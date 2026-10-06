import argparse
import asyncio
import json
import time
import uuid
import httpx
from fastapi import FastAPI, Request, Response
from starlette.responses import StreamingResponse
from typing import Dict, Any, List, Optional
import sqlite3
import os

from tokenwatch.decomposer import PromptSlotDecomposer
from tokenwatch.models import TelemetryEvent, PromptBreakdown
from tokenwatch.pricing import calculate_cost
from tokenwatch.optimizer import DynamicToolPruner, PrefixCacheStabilizer

class LocalTelemetryBuffer:
    """
    Local SQLite buffer to ensure zero data loss during network blips or Hub restarts.
    """
    def __init__(self, db_path: str = "data/sidecar_spool.db"):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_json TEXT NOT NULL
                );
            """)

    def push(self, event: TelemetryEvent):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO pending_telemetry (event_json) VALUES (?)", (event.model_dump_json(),))

    def fetch_batch(self, limit: int = 50) -> List[tuple]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, event_json FROM pending_telemetry ORDER BY id ASC LIMIT ?", (limit,))
            return cursor.fetchall()

    def remove(self, ids: List[int]):
        if not ids:
            return
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM pending_telemetry WHERE id IN ({placeholders})", ids)

def create_sidecar_app(
    upstream_url: str,
    node_id: str,
    hub_url: Optional[str] = None,
    optimize_tools: bool = False
) -> FastAPI:
    app = FastAPI(title=f"Token Watch Sidecar - {node_id}")
    decomposer = PromptSlotDecomposer()
    tool_pruner = DynamicToolPruner()
    telemetry_buffer = LocalTelemetryBuffer()
    telemetry_queue: asyncio.Queue[TelemetryEvent] = asyncio.Queue()

    client = httpx.AsyncClient(base_url=upstream_url, timeout=120.0)
    hub_client = httpx.AsyncClient(timeout=10.0) if hub_url else None

    # Background flusher task
    async def telemetry_flusher():
        while True:
            try:
                # 1. Drain in-memory queue to disk buffer
                while not telemetry_queue.empty():
                    evt = await telemetry_queue.get()
                    telemetry_buffer.push(evt)
                    telemetry_queue.task_done()

                # 2. If hub is configured, drain buffer to hub
                if hub_client and hub_url:
                    batch = telemetry_buffer.fetch_batch(limit=50)
                    if batch:
                        successful_ids = []
                        for row_id, evt_json in batch:
                            try:
                                resp = await hub_client.post(
                                    f"{hub_url.rstrip('/')}/api/v1/telemetry",
                                    content=evt_json,
                                    headers={"Content-Type": "application/json"}
                                )
                                if resp.status_code == 200:
                                    successful_ids.append(row_id)
                            except Exception:
                                break  # Hub temporarily down, retry later
                        telemetry_buffer.remove(successful_ids)
            except Exception as e:
                pass
            await asyncio.sleep(2.0)

    @app.on_event("startup")
    async def startup_event():
        asyncio.create_task(telemetry_flusher())

    @app.on_event("shutdown")
    async def shutdown_event():
        await client.aclose()
        if hub_client:
            await hub_client.aclose()

    @app.post("/v1/chat/completions")
    async def intercept_chat(request: Request):
        start_time = time.time()
        body = await request.json()
        trace_id = str(uuid.uuid4())
        correlation_id = request.headers.get("X-Correlation-ID")
        agent_id = request.headers.get("X-Agent-ID", "general_agent")
        task_type = request.headers.get("X-Task-Type", "agent_action")
        model = body.get("model", "unknown-model")
        is_streaming = body.get("stream", False)

        # Active Optimization 1: Prefix Cache Stabilization
        if request.headers.get("X-Stabilize-Prefix", "false").lower() == "true":
            body = PrefixCacheStabilizer.stabilize_payload(body)

        # Active Optimization 2: Dynamic Tool Pruning
        declared_tools = []
        if "tools" in body and isinstance(body["tools"], list):
            declared_tools = [t.get("function", {}).get("name", "") for t in body["tools"]]
            if optimize_tools or request.headers.get("X-Optimize-Tools", "false").lower() == "true":
                messages = body.get("messages", [])
                user_query = messages[-1].get("content", "") if messages else ""
                kept, pruned = tool_pruner.prune_tools(body["tools"], str(user_query))
                body["tools"] = kept

        # Prompt Decomposition
        breakdown = decomposer.decompose(body)

        if is_streaming:
            # Enable usage reporting in stream if requested for OpenAI-compatible runtimes
            if "stream_options" not in body:
                body["stream_options"] = {"include_usage": True}

            # Forward streaming request
            upstream_req = client.build_request(
                "POST",
                "/chat/completions",
                json=body,
                headers={k: v for k, v in request.headers.items() if k.lower() not in ["host", "content-length"]}
            )
            upstream_resp = await client.send(upstream_req, stream=True)

            async def stream_generator():
                output_token_count = 0
                tools_invoked_set = set()
                raw_usage = None

                try:
                    async for chunk in upstream_resp.aiter_bytes():
                        # Parse SSE chunk
                        chunk_text = chunk.decode("utf-8", errors="ignore")
                        for line in chunk_text.splitlines():
                            if line.startswith("data: ") and line != "data: [DONE]":
                                try:
                                    data = json.loads(line[6:])
                                    # Extract usage if present
                                    if "usage" in data and data["usage"]:
                                        raw_usage = data["usage"]
                                    # Check delta tool calls
                                    choices = data.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        if "tool_calls" in delta and delta["tool_calls"]:
                                            for tc in delta["tool_calls"]:
                                                fn = tc.get("function", {})
                                                if "name" in fn:
                                                    tools_invoked_set.add(fn["name"])
                                except Exception:
                                    pass
                        output_token_count += max(1, len(chunk) // 4)
                        yield chunk
                finally:
                    await upstream_resp.aclose()
                    latency_ms = (time.time() - start_time) * 1000.0

                    # Resolve final token numbers
                    if raw_usage:
                        inp_tok = raw_usage.get("prompt_tokens", breakdown.total_prompt_tokens)
                        out_tok = raw_usage.get("completion_tokens", output_token_count)
                        cached_read = raw_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                    else:
                        inp_tok = breakdown.total_prompt_tokens
                        out_tok = output_token_count
                        cached_read = 0

                    cost = calculate_cost(model, inp_tok, out_tok, cached_read_tokens=cached_read)

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
                        cost_usd=cost,
                        latency_ms=latency_ms,
                        prompt_breakdown=breakdown,
                        tools_declared=declared_tools,
                        tools_invoked=list(tools_invoked_set),
                        raw_usage=raw_usage
                    )
                    await telemetry_queue.put(evt)

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        else:
            # Synchronous non-streaming path
            resp = await client.post(
                "/chat/completions",
                json=body,
                headers={k: v for k, v in request.headers.items() if k.lower() not in ["host", "content-length"]}
            )
            latency_ms = (time.time() - start_time) * 1000.0
            resp_data = resp.json() if resp.status_code == 200 else {}

            raw_usage = resp_data.get("usage")
            tools_invoked = []
            if "choices" in resp_data and resp_data["choices"]:
                msg = resp_data["choices"][0].get("message", {})
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    if "name" in fn:
                        tools_invoked.append(fn["name"])

            if raw_usage:
                inp_tok = raw_usage.get("prompt_tokens", breakdown.total_prompt_tokens)
                out_tok = raw_usage.get("completion_tokens", 0)
                cached_read = raw_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            else:
                inp_tok = breakdown.total_prompt_tokens
                out_tok = decomposer.count(str(resp_data))
                cached_read = 0

            cost = calculate_cost(model, inp_tok, out_tok, cached_read_tokens=cached_read)

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
                cost_usd=cost,
                latency_ms=latency_ms,
                prompt_breakdown=breakdown,
                tools_declared=declared_tools,
                tools_invoked=tools_invoked,
                raw_usage=raw_usage
            )
            await telemetry_queue.put(evt)

            return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")

    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token Watch Sidecar Proxy")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--upstream", type=str, default="http://localhost:11434/v1")
    parser.add_argument("--hub-url", type=str, default="http://localhost:9000")
    parser.add_argument("--node-id", type=str, default="node_edge_1")
    parser.add_argument("--optimize-tools", action="store_true", help="Enable dynamic tool pruning")
    args = parser.parse_args()

    import uvicorn
    app = create_sidecar_app(
        upstream_url=args.upstream,
        node_id=args.node_id,
        hub_url=args.hub_url,
        optimize_tools=args.optimize_tools
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port)
