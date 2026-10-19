import json

import httpx
import pytest
from fastapi.testclient import TestClient

from tokenwatch.advisor import TokenAdvisorEngine
from tokenwatch.decomposer import PromptSlotDecomposer
from tokenwatch.models import AdvisoryMessage, NodeHeartbeat, PromptBreakdown, TelemetryEvent
from tokenwatch.optimizer import DynamicToolPruner, PrefixCacheStabilizer, TelemetryCompactor
from tokenwatch.pricing import calculate_cost
from tokenwatch.security import DataRedactor, load_dotenv_if_present
from tokenwatch.sidecar import LocalTelemetryBuffer, create_sidecar_app
from tokenwatch.storage import SQLiteStorage


def test_decomposer():
    decomposer = PromptSlotDecomposer()
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a smart home assistant. API Key: sk-ant-api03-abcdef1234567890abcdef123456",
            },
            {
                "role": "user",
                "content": "Sensor temperature reading: 72F in living room. Password: mysecretpassword123",
            },
            {"role": "user", "content": "Turn down the thermostat to 68F."},
        ],
        "tools": [{"type": "function", "function": {"name": "set_climate", "parameters": {}}}],
    }
    breakdown = decomposer.decompose(payload, redact_secrets=True)
    assert breakdown.system_instructions > 0
    assert breakdown.tool_schemas > 0
    assert breakdown.iot_telemetry > 0
    assert breakdown.task_payload > 0
    assert breakdown.total_prompt_tokens > 0


def test_security_redactor():
    text = (
        "Connecting with sk-1234567890abcdef1234567890 and "
        "sk-ant-api03-1234567890abcdef1234567890 and password: supersecret and pin: 9988"
    )
    redacted = DataRedactor.redact_text(text)
    assert "sk-1234567890" not in redacted
    assert "[REDACTED_API_KEY]" in redacted
    assert "sk-ant-api03" not in redacted
    assert "[REDACTED_ANTHROPIC_KEY]" in redacted
    assert "supersecret" not in redacted
    assert "[REDACTED_PASSWORD]" in redacted
    assert "9988" not in redacted
    assert "[REDACTED_PIN]" in redacted


def test_dotenv_loader(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("TOKENWATCH_TEST_CUSTOM_VAR='loaded_from_dotenv'\n# Comment line\n", encoding="utf-8")
    monkeypatch.delenv("TOKENWATCH_TEST_CUSTOM_VAR", raising=False)
    load_dotenv_if_present(env_file)
    import os

    assert os.environ.get("TOKENWATCH_TEST_CUSTOM_VAR") == "loaded_from_dotenv"


def test_pricing_engine():
    cost_no_cache = calculate_cost("claude-3-5-sonnet-20241022", input_tokens=100_000, output_tokens=10_000)
    assert pytest.approx(cost_no_cache, 0.001) == 0.45

    cost_with_cache = calculate_cost(
        "claude-3-5-sonnet-20241022",
        input_tokens=100_000,
        output_tokens=10_000,
        cached_read_tokens=80_000,
    )
    assert pytest.approx(cost_with_cache, 0.001) == 0.234
    assert cost_with_cache < cost_no_cache

    # Verify gpt-4o-mini does not falsely match gpt-4o rate
    cost_mini = calculate_cost("gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000)
    assert pytest.approx(cost_mini, 0.001) == 0.75


def test_dynamic_tool_pruner():
    pruner = DynamicToolPruner(min_keep=2, max_keep=2)
    tools = [
        {"type": "function", "function": {"name": "turn_on_light", "description": "Controls indoor illumination and bulbs"}},
        {"type": "function", "function": {"name": "unlock_door", "description": "Controls smart lock deadbolt"}},
        {"type": "function", "function": {"name": "set_thermostat", "description": "Adjusts HVAC climate temperature"}},
        {"type": "function", "function": {"name": "water_lawn", "description": "Operates sprinkler system"}},
    ]
    kept, pruned = pruner.prune_tools(tools, user_query="Can you dim the kitchen lights?")
    kept_names = [t["function"]["name"] for t in kept]
    assert "turn_on_light" in kept_names
    assert len(kept) == 2
    assert "water_lawn" in pruned


def test_prefix_stabilizer():
    payload = {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "You are a butler."},
            {"role": "user", "content": "Turn on lights."},
        ]
    }
    stabilized = PrefixCacheStabilizer.stabilize_payload(payload)
    assert stabilized["messages"][0]["role"] == "system"


def test_telemetry_compactor():
    compactor = TelemetryCompactor()
    state1 = {"temp": 72, "humidity": 45}
    diff1 = compactor.compact_state_dict(state1)
    assert diff1 == state1

    diff2 = compactor.compact_state_dict(state1)
    assert diff2["status"] == "all_states_unchanged"

    diff3 = compactor.compact_state_dict({"temp": 73, "humidity": 45})
    assert diff3 == {"temp": 73}


def test_sqlite_storage_and_advisor():
    storage = SQLiteStorage(db_path=":memory:")
    advisor = TokenAdvisorEngine(storage=storage)

    event = TelemetryEvent(
        trace_id="t1",
        correlation_id="corr-123",
        node_id="n1",
        agent_id="hvac",
        model="claude-3-5-sonnet-20241022",
        input_tokens=3000,
        output_tokens=30,
        prompt_breakdown=PromptBreakdown(
            system_instructions=1000,
            tool_schemas=1500,
            conversation_history=100,
            iot_telemetry=300,
            task_payload=100,
            total_prompt_tokens=3000,
        ),
        tools_declared=["t1", "t2", "t3", "t4", "t5", "t6", "t7"],
        tools_invoked=["t1"],
    )
    advisor.record_event(event)

    summary = storage.get_agent_metrics_summary()
    assert len(summary) == 1
    assert summary[0]["agent_id"] == "hvac"

    advisories = advisor.generate_advisories("hvac")
    assert any(a.advisory_type == "TOOL_PRUNING" for a in advisories)
    assert any(a.advisory_type == "CACHE_ALIGNMENT" for a in advisories)
    assert any(a.advisory_type == "MODEL_CASCADING" for a in advisories)

    # Verify deterministic deduplication on repeated generate_advisories calls
    advisor.generate_advisories("hvac")
    persisted = storage.get_recent_advisories(agent_id="hvac")
    assert len(persisted) == len(advisories)


def test_heartbeat_and_circuit_breaker():
    storage = SQLiteStorage(db_path=":memory:")

    # Test heartbeat recording
    hb = NodeHeartbeat(
        node_id="edge_pi_1",
        status="ONLINE",
        memory_mb=128.5,
        backlog_events=0,
        active_agents=["agent_a"],
    )
    storage.record_heartbeat(hb)
    nodes = storage.get_node_statuses()
    assert len(nodes) == 1
    assert nodes[0]["node_id"] == "edge_pi_1"
    assert nodes[0]["status"] == "ONLINE"

    # Test circuit breaker tripping
    assert storage.is_circuit_tripped("loop-corr-99") is None
    storage.trip_circuit_breaker("loop-corr-99", reason="Loop detected")
    status = storage.is_circuit_tripped("loop-corr-99")
    assert status is not None
    assert status["reason"] == "Loop detected"

    storage.reset_circuit_breaker("loop-corr-99")
    assert storage.is_circuit_tripped("loop-corr-99") is None


def test_expanded_redactor():
    sample = "AWS: AKIAIOSFODNN7EXAMPLE, HF: hf_abcdefghijklmnopqrstuvwxyz01234567, Slack: xoxb-1234567890-abcdefghij"
    redacted = DataRedactor.redact_text(sample)
    assert "[REDACTED_AWS_KEY]" in redacted
    assert "[REDACTED_HF_KEY]" in redacted
    assert "[REDACTED_SLACK_KEY]" in redacted


def test_modern_pricing():
    cost_c37 = calculate_cost("claude-3-7-sonnet", input_tokens=1000, output_tokens=500)
    assert cost_c37 > 0
    cost_ds = calculate_cost("deepseek-chat", input_tokens=100_000, output_tokens=10_000)
    assert pytest.approx(cost_ds, 0.001) == 0.0168
    cost_r1 = calculate_cost("deepseek-r1", input_tokens=100_000, output_tokens=10_000)
    assert pytest.approx(cost_r1, 0.001) == 0.0769


def test_advisories_persistence_and_correlation_filter():
    storage = SQLiteStorage(db_path=":memory:")
    adv = AdvisoryMessage(
        advisory_type="TOOL_PRUNING",
        severity="WARNING",
        agent_id="test_agent",
        node_id="node_1",
        summary="Prune unused tools",
        estimated_savings_percent=45.0,
        recommendation="Use dynamic pruning",
    )
    storage.save_advisory(adv)
    advisories = storage.get_recent_advisories(agent_id="test_agent")
    assert len(advisories) == 1
    assert advisories[0].advisory_type == "TOOL_PRUNING"

    # Test correlation filter
    evt1 = TelemetryEvent(
        trace_id="t-c1",
        correlation_id="c-unique",
        node_id="n1",
        agent_id="a1",
        model="gpt-4o",
        input_tokens=100,
        output_tokens=50,
        prompt_breakdown=PromptBreakdown(system_instructions=100, total_prompt_tokens=100),
    )
    evt2 = TelemetryEvent(
        trace_id="t-c2",
        correlation_id="c-other",
        node_id="n1",
        agent_id="a1",
        model="gpt-4o",
        input_tokens=100,
        output_tokens=50,
        prompt_breakdown=PromptBreakdown(system_instructions=100, total_prompt_tokens=100),
    )
    storage.save_event(evt1)
    storage.save_event(evt2)
    filtered = storage.get_recent_events(correlation_id="c-unique")
    assert len(filtered) == 1
    assert filtered[0].trace_id == "t-c1"


def test_hub_api_client(monkeypatch):
    import tokenwatch.hub as hub_module
    from tokenwatch.security import API_KEY_HEADER_NAME, get_configured_api_key

    mem_storage = SQLiteStorage(db_path=":memory:")
    mem_advisor = TokenAdvisorEngine(storage=mem_storage)
    monkeypatch.setattr(hub_module, "storage", mem_storage)
    monkeypatch.setattr(hub_module, "advisor", mem_advisor)

    api_key = get_configured_api_key()
    headers = {API_KEY_HEADER_NAME: api_key} if api_key else {}

    client = TestClient(hub_module.app, headers=headers)
    # 1. Root health
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ONLINE"

    # 2. Ingest telemetry
    payload = {
        "trace_id": "test-trace-e2e",
        "correlation_id": "corr-e2e",
        "node_id": "edge_test",
        "agent_id": "thermostat_agent",
        "model": "gpt-4o-mini",
        "streaming": False,
        "input_tokens": 500,
        "output_tokens": 50,
        "prompt_breakdown": {
            "system_instructions": 200,
            "tool_schemas": 200,
            "conversation_history": 0,
            "iot_telemetry": 50,
            "task_payload": 50,
            "total_prompt_tokens": 500,
        },
    }
    ingest_resp = client.post("/api/v1/telemetry", json=payload)
    assert ingest_resp.status_code == 200
    assert ingest_resp.json()["status"] == "ok"

    # 3. Heartbeat populates client IP
    hb_resp = client.post("/api/v1/heartbeat", json={"node_id": "edge_test", "status": "ONLINE"})
    assert hb_resp.status_code == 200
    nodes_resp = client.get("/api/v1/nodes")
    assert nodes_resp.status_code == 200
    assert nodes_resp.json()[0]["ip_address"] is not None

    # 4. Analytics
    summary_resp = client.get("/api/v1/analytics/summary")
    assert summary_resp.status_code == 200
    assert any(s["agent_id"] == "thermostat_agent" for s in summary_resp.json())

    # 5. Advisories list endpoint
    adv_resp = client.get("/api/v1/advisories")
    assert adv_resp.status_code == 200

    # 6. Circuit breaker endpoints
    cb_check = client.get("/api/v1/circuit_breakers/check?correlation_id=corr-e2e")
    assert cb_check.status_code == 200
    assert cb_check.json()["is_tripped"] is False

    cb_reset = client.post("/api/v1/circuit_breakers/reset?circuit_key=corr-e2e")
    assert cb_reset.status_code == 200


def test_local_telemetry_buffer(tmp_path):
    db_file = str(tmp_path / "spool.db")
    buf = LocalTelemetryBuffer(db_path=db_file)
    assert buf.count_pending() == 0

    evt = TelemetryEvent(
        trace_id="t-spool-1",
        node_id="node_1",
        agent_id="agent_1",
        model="gpt-4o",
        input_tokens=100,
        output_tokens=20,
        prompt_breakdown=PromptBreakdown(total_prompt_tokens=100),
    )
    buf.push(evt)
    assert buf.count_pending() == 1

    batch = buf.fetch_batch(limit=10)
    assert len(batch) == 1
    row_id, data = batch[0]
    assert "t-spool-1" in data

    buf.remove([row_id])
    assert buf.count_pending() == 0


def test_sidecar_interception_and_streaming(tmp_path, monkeypatch):
    from tokenwatch.security import API_KEY_HEADER_NAME, get_configured_api_key

    spool_file = str(tmp_path / "sidecar_test_spool.db")
    sidecar_app = create_sidecar_app(
        upstream_url="http://mock-upstream.local/v1",
        node_id="node_test",
        hub_url=None,
        optimize_tools=True,
        spool_db_path=spool_file,
    )

    def mock_upstream_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        if body.get("stream"):
            sse_payload = (
                'data: {"choices":[{"delta":{"content":"Hello world"}}]}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(
                200,
                content=sse_payload.encode("utf-8"),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "choices": [{"message": {"role": "assistant", "content": "Thermostat set to 68F."}}],
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 8,
                    "prompt_tokens_details": {"cached_tokens": 60, "cached_write_tokens": 20},
                },
            },
        )

    mock_transport = httpx.MockTransport(mock_upstream_handler)

    api_key = get_configured_api_key()
    base_headers = {API_KEY_HEADER_NAME: api_key} if api_key else {}

    with TestClient(sidecar_app, headers=base_headers) as client:
        # Patch the upstream AsyncClient inside sidecar_app closure via transport replacement
        for route in sidecar_app.routes:
            if getattr(route, "path", "") == "/v1/chat/completions":
                closure_vars = dict(
                    zip(
                        route.endpoint.__code__.co_freevars,
                        [c.cell_contents for c in route.endpoint.__closure__],
                        strict=False,
                    )
                )
                upstream_client: httpx.AsyncClient = closure_vars["client"]
                monkeypatch.setattr(upstream_client, "_transport", mock_transport)

        # 1. Non-streaming completion with prefix stabilization and tool pruning
        resp = client.post(
            "/v1/chat/completions",
            headers={"X-Agent-ID": "climate_agent", "X-Stabilize-Prefix": "true"},
            json={
                "model": "claude-3-5-sonnet",
                "messages": [
                    {"role": "user", "content": "Set thermostat to 68F"},
                    {"role": "system", "content": "You are a home climate controller."},
                ],
                "tools": [
                    {"type": "function", "function": {"name": "set_thermostat", "description": "Set HVAC temp"}},
                    {"type": "function", "function": {"name": "lock_door", "description": "Lock door"}},
                    {"type": "function", "function": {"name": "water_lawn", "description": "Water lawn"}},
                    {"type": "function", "function": {"name": "play_music", "description": "Play speaker"}},
                ],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "Thermostat set to 68F."

        # 2. Streaming completion
        stream_resp = client.post(
            "/v1/chat/completions",
            headers={"X-Agent-ID": "stream_agent"},
            json={
                "model": "gpt-4o-mini",
                "stream": True,
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        assert stream_resp.status_code == 200
        assert "Hello world" in stream_resp.text


