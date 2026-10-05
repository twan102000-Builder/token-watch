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
    node_id: str
    agent_id: str
    task_type: Optional[str] = "general"
    model: str
    input_tokens: int
    output_tokens: int
    cached_read_tokens: int = 0
    cached_write_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    prompt_breakdown: PromptBreakdown
    tools_declared: List[str] = Field(default_factory=list)
    tools_invoked: List[str] = Field(default_factory=list)

class AdvisoryMessage(BaseModel):
    advisory_type: str  # e.g., 'TOOL_PRUNING', 'CACHE_ALIGNMENT', 'DELTA_COMPACTION', 'LOOP_DETECTED'
    severity: str  # 'INFO', 'WARNING', 'CRITICAL'
    agent_id: str
    node_id: str
    summary: str
    estimated_savings_percent: float
    recommendation: str
