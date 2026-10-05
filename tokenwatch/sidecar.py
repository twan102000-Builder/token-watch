import argparse
import uvicorn
from fastapi import FastAPI, Request, Response
import httpx
from tokenwatch.decomposer import PromptSlotDecomposer

def create_sidecar_app(upstream_url: str, node_id: str, hub_url: str = None) -> FastAPI:
    app = FastAPI(title=f"Token Watch Sidecar - {node_id}")
    decomposer = PromptSlotDecomposer()
    client = httpx.AsyncClient(base_url=upstream_url, timeout=60.0)

    @app.post("/v1/chat/completions")
    async def intercept_chat(request: Request):
        body = await request.json()
        agent_id = request.headers.get("X-Agent-ID", "default_agent")
        
        # Deconstruct tokens
        breakdown = decomposer.decompose(body)
        print(f"[Sidecar {node_id}][Agent {agent_id}] Prompt Tokens: {breakdown.total_prompt_tokens} (Tools: {breakdown.tool_schemas}, Telemetry: {breakdown.iot_telemetry})")

        # Forward upstream
        resp = await client.post("/chat/completions", json=body, headers=dict(request.headers))
        return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")

    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token Watch Sidecar Proxy")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--upstream", type=str, default="http://localhost:11434/v1")
    parser.add_argument("--node-id", type=str, default="node_edge_1")
    args = parser.parse_args()

    app = create_sidecar_app(upstream_url=args.upstream, node_id=args.node_id)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
