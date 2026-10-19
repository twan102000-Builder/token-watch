import time
from typing import Any

from pydantic import BaseModel, Field


class PromptBreakdown(BaseModel):
    system_instructions: int = 0
    tool_schemas: int = 0
    conversation_history: int = 0
    iot_telemetry: int = 0
    task_payload: int = 0
    total_prompt_tokens: int = 0


class TelemetryEvent(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    trace_id: str
    correlation_id: str | None = None  # Links multi-agent conversation turns
    node_id: str
    agent_id: str
    task_type: str | None = "general"
    model: str
    streaming: bool = False
    input_tokens: int
    output_tokens: int
    cached_read_tokens: int = 0
    cached_write_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    prompt_breakdown: PromptBreakdown
    tools_declared: list[str] = Field(default_factory=list)
    tools_invoked: list[str] = Field(default_factory=list)
    raw_usage: dict[str, Any] | None = None


class AdvisoryMessage(BaseModel):
    advisory_type: str  # e.g., 'TOOL_PRUNING', 'CACHE_ALIGNMENT', 'DELTA_COMPACTION', 'LOOP_DETECTED', 'MODEL_CASCADING'
    severity: str  # 'INFO', 'WARNING', 'CRITICAL'
    agent_id: str
    node_id: str
    summary: str
    estimated_savings_percent: float
    recommendation: str
    actionable_fix: dict[str, Any] | None = None


class NodeHeartbeat(BaseModel):
    node_id: str
    timestamp: float = Field(default_factory=time.time)
    status: str = "ONLINE"  # 'ONLINE', 'DEGRADED', 'OFFLINE'
    ip_address: str | None = None
    memory_mb: float = 0.0
    backlog_events: int = 0
    active_agents: list[str] = Field(default_factory=list)


class CircuitBreakerStatus(BaseModel):
    correlation_id: str | None = None
    agent_id: str | None = None
    is_tripped: bool = False
    reason: str | None = None
    tripped_at: float | None = None


class AlertPayload(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    alert_type: str  # e.g. 'LOOP_TRIP', 'QUOTA_EXCEEDED', 'NODE_OFFLINE'
    severity: str    # 'WARNING', 'CRITICAL'
    title: str
    message: str
    details: dict[str, Any] | None = None

