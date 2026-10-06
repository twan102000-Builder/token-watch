import pytest
from tokenwatch.decomposer import PromptSlotDecomposer
from tokenwatch.advisor import TokenAdvisorEngine
from tokenwatch.models import TelemetryEvent, PromptBreakdown
from tokenwatch.pricing import calculate_cost
from tokenwatch.optimizer import DynamicToolPruner, PrefixCacheStabilizer, TelemetryCompactor
from tokenwatch.storage import DuckDBStorage

def test_decomposer():
    decomposer = PromptSlotDecomposer()
    payload = {
        "messages": [
            {"role": "system", "content": "You are a smart home assistant."},
            {"role": "user", "content": "Sensor temperature reading: 72F in living room."},
            {"role": "user", "content": "Turn down the thermostat to 68F."}
        ],
        "tools": [
            {"type": "function", "function": {"name": "set_climate", "parameters": {}}}
        ]
    }
    breakdown = decomposer.decompose(payload)
    assert breakdown.system_instructions > 0
    assert breakdown.tool_schemas > 0
    assert breakdown.iot_telemetry > 0
    assert breakdown.task_payload > 0
    assert breakdown.total_prompt_tokens > 0

def test_pricing_engine():
    # 100k input, 10k output on Claude 3.5 Sonnet
    # Without cache: 100k * 3/M ($0.30) + 10k * 15/M ($0.15) = $0.45
    cost_no_cache = calculate_cost("claude-3-5-sonnet-20241022", input_tokens=100_000, output_tokens=10_000)
    assert pytest.approx(cost_no_cache, 0.001) == 0.45

    # With 80k cached read: 20k * 3/M ($0.06) + 80k * 0.30/M ($0.024) + 10k * 15/M ($0.15) = $0.234
    cost_with_cache = calculate_cost("claude-3-5-sonnet-20241022", input_tokens=100_000, output_tokens=10_000, cached_read_tokens=80_000)
    assert pytest.approx(cost_with_cache, 0.001) == 0.234
    assert cost_with_cache < cost_no_cache

def test_dynamic_tool_pruner():
    pruner = DynamicToolPruner(min_keep=2, max_keep=2)
    tools = [
        {"type": "function", "function": {"name": "turn_on_light", "description": "Controls indoor illumination and bulbs"}},
        {"type": "function", "function": {"name": "unlock_door", "description": "Controls smart lock deadbolt"}},
        {"type": "function", "function": {"name": "set_thermostat", "description": "Adjusts HVAC climate temperature"}},
        {"type": "function", "function": {"name": "water_lawn", "description": "Operates sprinkler system"}}
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
            {"role": "user", "content": "Turn on lights."}
        ]
    }
    stabilized = PrefixCacheStabilizer.stabilize_payload(payload)
    assert stabilized["messages"][0]["role"] == "system"

def test_telemetry_compactor():
    compactor = TelemetryCompactor()
    state1 = {"temp": 72, "humidity": 45}
    diff1 = compactor.compact_state_dict(state1)
    assert diff1 == state1

    # Second poll with unchanged states
    diff2 = compactor.compact_state_dict(state1)
    assert diff2["status"] == "all_states_unchanged"

    # Third poll with single change
    diff3 = compactor.compact_state_dict({"temp": 73, "humidity": 45})
    assert diff3 == {"temp": 73}

def test_duckdb_storage_and_advisor():
    storage = DuckDBStorage(db_path=":memory:")
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
            total_prompt_tokens=3000
        ),
        tools_declared=["t1", "t2", "t3", "t4", "t5", "t6", "t7"],
        tools_invoked=["t1"]
    )
    advisor.record_event(event)

    # Check analytics summary
    summary = storage.get_agent_metrics_summary()
    assert len(summary) == 1
    assert summary[0]["agent_id"] == "hvac"

    # Check advisories
    advisories = advisor.generate_advisories("hvac")
    assert any(a.advisory_type == "TOOL_PRUNING" for a in advisories)
    assert any(a.advisory_type == "CACHE_ALIGNMENT" for a in advisories)
    assert any(a.advisory_type == "MODEL_CASCADING" for a in advisories)

def test_loop_detection_advisory():
    storage = DuckDBStorage(db_path=":memory:")
    advisor = TokenAdvisorEngine(storage=storage)

    # Simulate rapid 4-turn chatter loop with same correlation ID
    import time
    now = time.time()
    for i in range(4):
        event = TelemetryEvent(
            timestamp=now + (i * 2),
            trace_id=f"trace-{i}",
            correlation_id="loop-test-corr",
            node_id="node_edge",
            agent_id="chatter_agent",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=100,
            prompt_breakdown=PromptBreakdown(
                system_instructions=100,
                tool_schemas=100,
                conversation_history=700,
                task_payload=100,
                total_prompt_tokens=1000
            )
        )
        advisor.record_event(event)

    advisories = advisor.generate_advisories("chatter_agent")
    assert any(a.advisory_type == "LOOP_DETECTED" for a in advisories)
