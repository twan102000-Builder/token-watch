import argparse
import os
from typing import Any

from fastapi import Depends, FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool

from tokenwatch.advisor import TokenAdvisorEngine
from tokenwatch.alerts import LocalAlertDispatcher
from tokenwatch.models import (
    AdvisoryMessage,
    AlertPayload,
    CircuitBreakerStatus,
    NodeHeartbeat,
    TelemetryEvent,
)
from tokenwatch.security import load_dotenv_if_present, verify_api_key
from tokenwatch.storage import SQLiteStorage

load_dotenv_if_present()

app = FastAPI(
    title="Token Watch Central Hub",
    description="100% private, zero-external-dependency token monitoring and advisement hub for multi-agent home networks.",
)

storage = SQLiteStorage()
advisor = TokenAdvisorEngine(storage=storage)
alert_dispatcher = LocalAlertDispatcher()


@app.post("/api/v1/telemetry", dependencies=[Depends(verify_api_key)])
async def ingest_telemetry(event: TelemetryEvent) -> dict[str, Any]:
    await run_in_threadpool(advisor.record_event, event)
    immediate_advisories = await run_in_threadpool(advisor.generate_advisories, event.agent_id)

    # Check if a circuit breaker was tripped
    loop_advisory = next((a for a in immediate_advisories if a.advisory_type == "LOOP_DETECTED"), None)
    if loop_advisory:
        await alert_dispatcher.send_alert(
            AlertPayload(
                alert_type="LOOP_TRIP",
                severity="CRITICAL",
                title=f"Circuit Breaker Tripped: {event.agent_id}",
                message=loop_advisory.summary,
                details={"correlation_id": event.correlation_id, "node_id": event.node_id},
            )
        )

    return {
        "status": "ok",
        "recorded_trace": event.trace_id,
        "advisories_count": len(immediate_advisories),
        "circuit_breaker_triggered": loop_advisory is not None,
    }


@app.post("/api/v1/heartbeat", dependencies=[Depends(verify_api_key)])
def record_heartbeat(heartbeat: NodeHeartbeat, request: Request) -> dict[str, str]:
    if not heartbeat.ip_address and request.client:
        heartbeat.ip_address = request.client.host
    storage.record_heartbeat(heartbeat)
    return {"status": "ok", "node_id": heartbeat.node_id}


@app.get("/api/v1/nodes", dependencies=[Depends(verify_api_key)])
def get_nodes() -> list[dict[str, Any]]:
    return storage.get_node_statuses()


@app.get("/api/v1/circuit_breakers/check", dependencies=[Depends(verify_api_key)])
def check_circuit_breaker(
    correlation_id: str | None = None,
    agent_id: str | None = None,
) -> CircuitBreakerStatus:
    if correlation_id:
        status = storage.is_circuit_tripped(correlation_id)
        if status:
            return CircuitBreakerStatus(
                correlation_id=correlation_id,
                is_tripped=True,
                reason=status["reason"],
                tripped_at=status["tripped_at"],
            )
    if agent_id:
        status = storage.is_circuit_tripped(agent_id)
        if status:
            return CircuitBreakerStatus(
                agent_id=agent_id,
                is_tripped=True,
                reason=status["reason"],
                tripped_at=status["tripped_at"],
            )
    return CircuitBreakerStatus(is_tripped=False)


@app.post("/api/v1/circuit_breakers/reset", dependencies=[Depends(verify_api_key)])
def reset_circuit_breaker(circuit_key: str) -> dict[str, str]:
    storage.reset_circuit_breaker(circuit_key)
    return {"status": "reset", "circuit_key": circuit_key}


@app.get("/api/v1/advisories", response_model=list[AdvisoryMessage], dependencies=[Depends(verify_api_key)])
def list_recent_advisories(
    limit: int = Query(20, ge=1, le=100),
    agent_id: str | None = None,
    max_age_seconds: float | None = Query(None, ge=1),
):
    return storage.get_recent_advisories(limit=limit, agent_id=agent_id, max_age_seconds=max_age_seconds)


@app.get("/api/v1/advisories/{agent_id}", response_model=list[AdvisoryMessage], dependencies=[Depends(verify_api_key)])
def get_advisories(agent_id: str):
    return advisor.generate_advisories(agent_id, persist=False, trip_circuits=False)


@app.get("/api/v1/analytics/summary", response_model=list[dict[str, Any]], dependencies=[Depends(verify_api_key)])
def get_metrics_summary():
    return storage.get_agent_metrics_summary()


@app.get("/api/v1/analytics/events", response_model=list[TelemetryEvent], dependencies=[Depends(verify_api_key)])
def get_recent_events(
    limit: int = Query(50, ge=1, le=500),
    agent_id: str | None = None,
    correlation_id: str | None = None,
):
    return storage.get_recent_events(limit=limit, agent_id=agent_id, correlation_id=correlation_id)


@app.get("/")
def root():
    summary = storage.get_agent_metrics_summary()
    nodes = storage.get_node_statuses()
    total_spend = sum(s["total_spend_usd"] for s in summary)
    total_tokens = sum(s["total_input_tokens"] + s["total_output_tokens"] for s in summary)
    return {
        "service": "Token Watch Central Hub",
        "status": "ONLINE",
        "storage_engine": "SQLite WAL (Zero SaaS/External Dependencies)",
        "active_nodes_count": len([n for n in nodes if n["status"] == "ONLINE"]),
        "active_agents_count": len(summary),
        "total_spend_usd": round(total_spend, 4),
        "total_tokens_processed": total_tokens,
    }


def main() -> None:
    load_dotenv_if_present()
    default_host = os.environ.get("TOKENWATCH_HUB_HOST", "127.0.0.1")
    default_port = int(os.environ.get("TOKENWATCH_HUB_PORT", "9000"))

    parser = argparse.ArgumentParser(description="Token Watch Central Hub")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--host", type=str, default=default_host)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

