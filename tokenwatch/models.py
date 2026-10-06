from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import time

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
    correlation_id: Optional[str] = None  # Links multi-agent conversation turns
    node_id: str
    agent_id: str
    task_type: Optional[str] = "general"
    model: str
    streaming: bool = False
    input_tokens: int
    output_tokens: int
    cached_read_tokens: int = 0
    cached_write_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    prompt_breakdown: PromptBreakdown
    tools_declared: List[str] = Field(default_factory=list)
    tools_invoked: List[str] = Field(default_factory=list)
    raw_usage: Optional[Dict[str, Any]] = None

class AdvisoryMessage(BaseModel):
    advisory_type: str  # e.g., 'TOOL_PRUNING', 'CACHE_ALIGNMENT', 'DELTA_COMPACTION', 'LOOP_DETECTED', 'MODEL_CASCADING'
    severity: str  # 'INFO', 'WARNING', 'CRITICAL'
    agent_id: str
    node_id: str
    summary: str
    estimated_savings_percent: float
    recommendation: str
    actionable_fix: Optional[Dict[str, Any]] = None

class NodeHeartbeat(BaseModel):
    node_id: str
    timestamp: float = Field(default_factory=time.time)
    status: str = "ONLINE"  # 'ONLINE', 'DEGRADED', 'OFFLINE'
    ip_address: Optional[str] = None
    memory_mb: float = 0.0
    backlog_events: int = 0
    active_agents: List[str] = Field(default_factory=list)

class CircuitBreakerStatus(BaseModel):
    correlation_id: Optional[str] = None
    agent_id: Optional[str] = None
    is_tripped: bool = False
    reason: Optional[str] = None
    tripped_at: Optional[float] = None

class AlertPayload(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    alert_type: str  # e.g. 'LOOP_TRIP', 'QUOTA_EXCEEDED', 'NODE_OFFLINE'
    severity: str    # 'WARNING', 'CRITICAL'
    title: str
    message: str
    details: Optional[Dict[str, Any]] = None
