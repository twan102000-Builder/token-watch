import argparse
import asyncio
import json
import logging
import os
import resource
import sqlite3
import sys
import time
import uuid
from contextlib import asynccontextmanager, contextmanager, suppress
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Request, Response
from starlette.responses import StreamingResponse

from tokenwatch.decomposer import PromptSlotDecomposer
from tokenwatch.models import NodeHeartbeat, TelemetryEvent
from tokenwatch.optimizer import DynamicToolPruner, PrefixCacheStabilizer
from tokenwatch.pricing import calculate_cost
from tokenwatch.security import (
    API_KEY_HEADER_NAME,
    DataRedactor,
    get_configured_api_key,
    load_dotenv_if_present,
    verify_api_key,
)

logger = logging.getLogger("tokenwatch.sidecar")


def _extract_usage_tokens(
    raw_usage: dict[str, Any] | None,
    fallback_input: int,
    fallback_output: int,
) -> tuple[int, int, int, int]:
    """
    Extracts (input_tokens, output_tokens, cached_read_tokens, cached_write_tokens)
    across OpenAI, Anthropic, and local vLLM/Ollama usage formats.
    """
    if not raw_usage:
        return fallback_input, fallback_output, 0, 0

    inp_tok = raw_usage.get("prompt_tokens", raw_usage.get("input_tokens", fallback_input))
    out_tok = raw_usage.get("completion_tokens", raw_usage.get("output_tokens", fallback_output))
    prompt_details = raw_usage.get("prompt_tokens_details") or {}
    cached_read = prompt_details.get(
        "cached_tokens",
        raw_usage.get("cache_read_input_tokens", raw_usage.get("cached_read_tokens", 0)),
    )
    cached_write = prompt_details.get(
        "cached_write_tokens",
        raw_usage.get("cache_creation_input_tokens", raw_usage.get("cached_write_tokens", 0)),
    )
    return int(inp_tok or 0), int(out_tok or 0), int(cached_read or 0), int(cached_write or 0)


class LocalTelemetryBuffer:
    """
    Local SQLite buffer to ensure zero data loss during network blips or Hub restarts.
    Guarantees clean connection lifecycle management with WAL mode.
    """

    def __init__(self, db_path: str | None = None):
        resolved_path = db_path or os.environ.get("TOKENWATCH_SPOOL_DB_PATH", "data/sidecar_spool.db")
        self.is_memory = resolved_path == ":memory:"
        if not self.is_memory:
            os.makedirs(os.path.dirname(os.path.abspath(resolved_path)), exist_ok=True)
        self.db_path = resolved_path
        self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False) if self.is_memory else None
        self._init_db()

    @contextmanager
    def _get_conn(self):
        if self.is_memory and self._mem_conn:
            with self._mem_conn:
                yield self._mem_conn
        else:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.execute("PRAGMA journal_mode = WAL;")
            try:
                with conn:
                    yield conn
            finally:
                conn.close()

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_json TEXT NOT NULL
                );
            """)

    def push(self, event: TelemetryEvent) -> None:
        with self._get_conn() as conn:
            conn.execute("INSERT INTO pending_telemetry (event_json) VALUES (?)", (event.model_dump_json(),))

    def fetch_batch(self, limit: int = 50) -> list[tuple[int, str]]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, event_json FROM pending_telemetry ORDER BY id ASC LIMIT ?", (limit,))
            return cursor.fetchall()

    def remove(self, ids: list[int]) -> None:
        if not ids:
            return
        with self._get_conn() as conn:
            placeholders = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM pending_telemetry WHERE id IN ({placeholders})", ids)

    def count_pending(self) -> int:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM pending_telemetry")
            row = cursor.fetchone()
            return row[0] if row else 0


def create_sidecar_app(
    upstream_url: str,
    node_id: str,
    hub_url: str | None = None,
    optimize_tools: bool = False,
    spool_db_path: str | None = None,
) -> FastAPI:
    load_dotenv_if_present()
    decomposer = PromptSlotDecomposer()
    tool_pruner = DynamicToolPruner()
    telemetry_buffer = LocalTelemetryBuffer(db_path=spool_db_path)
    telemetry_queue: asyncio.Queue[TelemetryEvent] = asyncio.Queue()
    active_agents_set: set[str] = set()

    def _hub_auth_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        configured_key = get_configured_api_key()
        if configured_key:
            headers[API_KEY_HEADER_NAME] = configured_key
        if extra:
            headers.update(extra)
        return headers

    client = httpx.AsyncClient(base_url=upstream_url, timeout=120.0)
    hub_client = httpx.AsyncClient(timeout=6.0) if hub_url else None

    async def flush_once() -> None:
        while not telemetry_queue.empty():
            evt = await telemetry_queue.get()
            await asyncio.to_thread(telemetry_buffer.push, evt)
            telemetry_queue.task_done()

        if hub_client and hub_url:
            batch = await asyncio.to_thread(telemetry_buffer.fetch_batch, 50)
            if batch:
                successful_ids = []
                for row_id, evt_json in batch:
                    try:
                        resp = await hub_client.post(
                            f"{hub_url.rstrip('/')}/api/v1/telemetry",
                            content=evt_json,
                            headers=_hub_auth_headers({"Content-Type": "application/json"}),
                        )
                        if resp.status_code == 200:
                            successful_ids.append(row_id)
                    except httpx.HTTPError:
                        break
                await asyncio.to_thread(telemetry_buffer.remove, successful_ids)

    # Background Flusher & Heartbeat
    async def background_worker() -> None:
        last_heartbeat = 0.0
        while True:
            now = time.time()
            try:
                await flush_once()

                # Send Node Heartbeat every 30s
                if hub_client and hub_url and (now - last_heartbeat >= 30.0):
                    mem_mb = 0.0
                    with suppress(OSError, ValueError):
                        rusage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        mem_mb = (rusage / (1024 * 1024)) if sys.platform == "darwin" else (rusage / 1024)

                    backlog_count = await asyncio.to_thread(telemetry_buffer.count_pending)
                    hb = NodeHeartbeat(
                        node_id=node_id,
                        timestamp=now,
                        status="ONLINE",
                        memory_mb=round(mem_mb, 1),
                        backlog_events=backlog_count,
                        active_agents=list(active_agents_set),
                    )
                    try:
                        await hub_client.post(
                            f"{hub_url.rstrip('/')}/api/v1/heartbeat",
                            json=hb.model_dump(),
                            headers=_hub_auth_headers(),
                        )
                        last_heartbeat = now
                    except httpx.HTTPError as exc:
                        logger.debug(f"Heartbeat delivery deferred: {exc}")

            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Background worker iteration error: {exc}")
            await asyncio.sleep(2.0)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        worker_task = asyncio.create_task(background_worker())
        try:
            yield
        finally:
            worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await worker_task
            with suppress(Exception):
                await flush_once()
            await client.aclose()
            if hub_client:
                await hub_client.aclose()

    app = FastAPI(title=f"Token Watch Sidecar - {node_id}", lifespan=lifespan)

    @app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
    async def intercept_chat(request: Request):
        start_time = time.time()
        body = await request.json()
        trace_id = str(uuid.uuid4())
        correlation_id = request.headers.get("X-Correlation-ID")
        agent_id = request.headers.get("X-Agent-ID", "general_agent")
        active_agents_set.add(agent_id)
        task_type = request.headers.get("X-Task-Type", "agent_action")
        model = body.get("model", "unknown-model")
        is_streaming = body.get("stream", False)

        # 1. Circuit Breaker Check (both conversation chain and rogue agent)
        if hub_client and hub_url and (correlation_id or agent_id):
            try:
                params = []
                if correlation_id:
                    params.append(f"correlation_id={correlation_id}")
                if agent_id:
                    params.append(f"agent_id={agent_id}")
                cb_resp = await hub_client.get(
                    f"{hub_url.rstrip('/')}/api/v1/circuit_breakers/check?{'&'.join(params)}",
                    headers=_hub_auth_headers(),
                )
                if cb_resp.status_code == 200 and cb_resp.json().get("is_tripped"):
                    reason = cb_resp.json().get("reason", "Circuit breaker tripped")
                    return Response(
                        content=json.dumps(
                            {
                                "error": "circuit_breaker_tripped",
                                "message": f"Circuit breaker tripped to prevent token exhaustion: {reason}",
                            }
                        ),
                        status_code=429,
                        media_type="application/json",
                    )
            except httpx.HTTPError as exc:
                logger.debug(f"Circuit breaker check skipped (fail-open): {exc}")

        # 2. Active Optimization: Prefix Cache Stabilization
        if request.headers.get("X-Stabilize-Prefix", "false").lower() == "true":
            body = PrefixCacheStabilizer.stabilize_payload(body)

        # 3. Active Optimization: Dynamic Tool Pruning
        declared_tools = []
        if "tools" in body and isinstance(body["tools"], list):
            declared_tools = [t.get("function", {}).get("name", "") for t in body["tools"]]
            if optimize_tools or request.headers.get("X-Optimize-Tools", "false").lower() == "true":
                messages = body.get("messages", [])
                user_query = ""
                for m in reversed(messages):
                    if m.get("role") == "user":
                        c = m.get("content")
                        if isinstance(c, str):
                            user_query = c
                        elif isinstance(c, list):
                            user_query = " ".join(
                                str(p.get("text", "")) for p in c if isinstance(p, dict)
                            )
                        elif c is not None:
                            user_query = str(c)
                        break
                kept, _pruned = tool_pruner.prune_tools(body["tools"], user_query)
                body["tools"] = kept

        # 4. Prompt Slot Decomposition (with secret redaction)
        breakdown = decomposer.decompose(body, redact_secrets=True)

        # Forward headers upstream (strip host, internal auth, and hop-by-hop headers)
        hop_by_hop = {
            "host",
            "content-length",
            "connection",
            "transfer-encoding",
            "keep-alive",
            API_KEY_HEADER_NAME.lower(),
        }
        upstream_headers = {k: v for k, v in request.headers.items() if k.lower() not in hop_by_hop}

        if is_streaming:
            if "stream_options" not in body:
                body["stream_options"] = {"include_usage": True}

            upstream_req = client.build_request("POST", "/chat/completions", json=body, headers=upstream_headers)
            upstream_resp = await client.send(upstream_req, stream=True)

            async def stream_generator():
                streamed_text_parts: list[str] = []
                sse_event_chunks = 0
                tools_invoked_set: set[str] = set()
                raw_usage: dict[str, Any] | None = None

                try:
                    async for chunk in upstream_resp.aiter_bytes():
                        chunk_text = chunk.decode("utf-8", errors="ignore")
                        for line in chunk_text.splitlines():
                            if line.startswith("data: ") and line != "data: [DONE]":
                                sse_event_chunks += 1
                                try:
                                    data = json.loads(line[6:])
                                    if data.get("usage"):
                                        raw_usage = data["usage"]
                                    choices = data.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        content_piece = delta.get("content")
                                        if content_piece:
                                            streamed_text_parts.append(str(content_piece))
                                        if delta.get("tool_calls"):
                                            for tc in delta["tool_calls"]:
                                                fn = tc.get("function", {})
                                                if fn.get("name"):
                                                    tools_invoked_set.add(fn["name"])
                                                if fn.get("arguments"):
                                                    streamed_text_parts.append(str(fn["arguments"]))
                                except (json.JSONDecodeError, TypeError, KeyError):
                                    pass
                        yield chunk
                finally:
                    await upstream_resp.aclose()
                    latency_ms = (time.time() - start_time) * 1000.0

                    combined_stream_text = "".join(streamed_text_parts)
                    estimated_out_tokens = (
                        decomposer.count(combined_stream_text)
                        if combined_stream_text
                        else max(1, sse_event_chunks)
                    )

                    inp_tok, out_tok, cached_read, cached_write = _extract_usage_tokens(
                        raw_usage,
                        fallback_input=breakdown.total_prompt_tokens,
                        fallback_output=estimated_out_tokens,
                    )

                    if upstream_resp.status_code >= 400:
                        out_tok = 0
                        cost = 0.0
                    else:
                        cost = calculate_cost(
                            model,
                            inp_tok,
                            out_tok,
                            cached_read_tokens=cached_read,
                            cached_write_tokens=cached_write,
                        )

                    evt = TelemetryEvent(
                        timestamp=time.time(),
                        trace_id=trace_id,
                        correlation_id=correlation_id,
                        node_id=node_id,
                        agent_id=agent_id,
                        task_type=task_type,
                        model=model,
                        streaming=True,
                        input_tokens=inp_tok,
                        output_tokens=out_tok,
                        cached_read_tokens=cached_read,
                        cached_write_tokens=cached_write,
                        cost_usd=cost,
                        latency_ms=latency_ms,
                        prompt_breakdown=breakdown,
                        tools_declared=declared_tools,
                        tools_invoked=list(tools_invoked_set),
                        raw_usage=DataRedactor.sanitize_dict(raw_usage),
                    )
                    telemetry_queue.put_nowait(evt)

            media_type = upstream_resp.headers.get("content-type", "text/event-stream")
            return StreamingResponse(stream_generator(), status_code=upstream_resp.status_code, media_type=media_type)

        resp = await client.post("/chat/completions", json=body, headers=upstream_headers)
        latency_ms = (time.time() - start_time) * 1000.0
        resp_data: dict[str, Any] = {}
        if resp.status_code == 200:
            with suppress(ValueError):
                resp_data = resp.json()

        raw_usage = resp_data.get("usage")
        tools_invoked: list[str] = []
        assistant_text_parts: list[str] = []
        if resp_data.get("choices"):
            msg = resp_data["choices"][0].get("message", {})
            if msg.get("content"):
                assistant_text_parts.append(str(msg["content"]))
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                if fn.get("name"):
                    tools_invoked.append(fn["name"])
                if fn.get("arguments"):
                    assistant_text_parts.append(str(fn["arguments"]))

        fallback_out = decomposer.count(" ".join(assistant_text_parts)) if assistant_text_parts else 0
        inp_tok, out_tok, cached_read, cached_write = _extract_usage_tokens(
            raw_usage,
            fallback_input=breakdown.total_prompt_tokens,
            fallback_output=fallback_out,
        )

        if resp.status_code >= 400:
            out_tok = 0
            cost = 0.0
        else:
            cost = calculate_cost(
                model,
                inp_tok,
                out_tok,
                cached_read_tokens=cached_read,
                cached_write_tokens=cached_write,
            )

        evt = TelemetryEvent(
            timestamp=time.time(),
            trace_id=trace_id,
            correlation_id=correlation_id,
            node_id=node_id,
            agent_id=agent_id,
            task_type=task_type,
            model=model,
            streaming=False,
            input_tokens=inp_tok,
            output_tokens=out_tok,
            cached_read_tokens=cached_read,
            cached_write_tokens=cached_write,
            cost_usd=cost,
            latency_ms=latency_ms,
            prompt_breakdown=breakdown,
            tools_declared=declared_tools,
            tools_invoked=tools_invoked,
            raw_usage=DataRedactor.sanitize_dict(raw_usage),
        )
        telemetry_queue.put_nowait(evt)

        media_type = resp.headers.get("content-type", "application/json")
        return Response(content=resp.content, status_code=resp.status_code, media_type=media_type)

    return app


def main() -> None:
    load_dotenv_if_present()
    default_port = int(os.environ.get("TOKENWATCH_SIDECAR_PORT", "8080"))
    default_host = os.environ.get("TOKENWATCH_SIDECAR_HOST", "127.0.0.1")
    default_upstream = os.environ.get("TOKENWATCH_UPSTREAM_URL", "http://localhost:11434/v1")
    default_hub_host = os.environ.get("TOKENWATCH_HUB_HOST", "localhost")
    default_hub_port = os.environ.get("TOKENWATCH_HUB_PORT", "9000")
    default_hub_url = os.environ.get("TOKENWATCH_HUB_URL", f"http://{default_hub_host}:{default_hub_port}")
    default_node_id = os.environ.get("TOKENWATCH_NODE_ID", "node_edge_1")
    default_opt_tools = os.environ.get("TOKENWATCH_OPTIMIZE_TOOLS", "false").lower() in ("true", "1", "yes")

    parser = argparse.ArgumentParser(description="Token Watch Sidecar Proxy")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--host", type=str, default=default_host)
    parser.add_argument("--upstream", type=str, default=default_upstream)
    parser.add_argument("--hub-url", type=str, default=default_hub_url)
    parser.add_argument("--node-id", type=str, default=default_node_id)
    parser.add_argument(
        "--optimize-tools",
        action="store_true",
        default=default_opt_tools,
        help="Enable dynamic tool pruning",
    )
    args = parser.parse_args()

    import uvicorn

    app = create_sidecar_app(
        upstream_url=args.upstream,
        node_id=args.node_id,
        hub_url=args.hub_url,
        optimize_tools=args.optimize_tools,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

