from fastapi import FastAPI
from tokenwatch.models import TelemetryEvent, AdvisoryMessage
from tokenwatch.advisor import TokenAdvisorEngine
from typing import List

app = FastAPI(title="Token Watch Central Hub")
advisor = TokenAdvisorEngine()

@app.post("/api/v1/telemetry")
async def ingest_telemetry(event: TelemetryEvent):
    advisor.record_event(event)
    return {"status": "ok", "recorded_trace": event.trace_id}

@app.get("/api/v1/advisories/{agent_id}", response_model=List[AdvisoryMessage])
async def get_advisories(agent_id: str):
    return advisor.generate_advisories(agent_id)

@app.get("/")
async def root():
    return {"status": "Token Watch Hub Online", "events_indexed": len(advisor.history)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
