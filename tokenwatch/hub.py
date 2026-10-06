from fastapi import FastAPI, Query
from typing import List, Optional, Dict, Any
from tokenwatch.models import TelemetryEvent, AdvisoryMessage
from tokenwatch.storage import DuckDBStorage
from tokenwatch.advisor import TokenAdvisorEngine

app = FastAPI(
    title="Token Watch Central Hub",
    description="Telemetry aggregation, time-series storage, and active advisement engine for multi-agent networks."
)

storage = DuckDBStorage(db_path="data/tokenwatch.duckdb")
advisor = TokenAdvisorEngine(storage=storage)

@app.post("/api/v1/telemetry")
async def ingest_telemetry(event: TelemetryEvent) -> Dict[str, Any]:
    # Persist event
    advisor.record_event(event)
    # Generate any immediate critical advisories (e.g. loop breaker)
    immediate_advisories = advisor.generate_advisories(event.agent_id)
    return {
        "status": "ok",
        "recorded_trace": event.trace_id,
        "advisories_count": len(immediate_advisories),
        "circuit_breaker_triggered": any(a.advisory_type == "LOOP_DETECTED" for a in immediate_advisories)
    }

@app.get("/api/v1/advisories/{agent_id}", response_model=List[AdvisoryMessage])
async def get_advisories(agent_id: str):
    return advisor.generate_advisories(agent_id)

@app.get("/api/v1/analytics/summary", response_model=List[Dict[str, Any]])
async def get_metrics_summary():
    return storage.get_agent_metrics_summary()

@app.get("/api/v1/analytics/events", response_model=List[TelemetryEvent])
async def get_recent_events(
    limit: int = Query(50, ge=1, le=500),
    agent_id: Optional[str] = None
):
    return storage.get_recent_events(limit=limit, agent_id=agent_id)

@app.get("/")
async def root():
    summary = storage.get_agent_metrics_summary()
    total_spend = sum(s["total_spend_usd"] for s in summary)
    total_tokens = sum(s["total_input_tokens"] + s["total_output_tokens"] for s in summary)
    return {
        "service": "Token Watch Central Hub",
        "status": "ONLINE",
        "active_agents": len(summary),
        "total_spend_usd": round(total_spend, 4),
        "total_tokens_processed": total_tokens
    }

if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser(description="Token Watch Central Hub")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
