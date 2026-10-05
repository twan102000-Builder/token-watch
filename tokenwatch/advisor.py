from typing import List, Dict, Any
from tokenwatch.models import TelemetryEvent, AdvisoryMessage

class TokenAdvisorEngine:
    """
    Evaluates telemetry events across agents and nodes to generate actionable advisories.
    """
    def __init__(self):
        self.history: List[TelemetryEvent] = []

    def record_event(self, event: TelemetryEvent):
        self.history.append(event)
        # Keep sliding window of 500 events
        if len(self.history) > 500:
            self.history.pop(0)

    def generate_advisories(self, agent_id: str) -> List[AdvisoryMessage]:
        advisories = []
        agent_events = [e for e in self.history if e.agent_id == agent_id]
        if not agent_events:
            return advisories

        latest = agent_events[-1]

        # 1. Check Tool Schema Bloat
        if latest.tools_declared:
            total_tools = len(latest.tools_declared)
            invoked_set = set()
            for e in agent_events[-30:]:
                invoked_set.update(e.tools_invoked)
            
            dormant_tools = [t for t in latest.tools_declared if t not in invoked_set]
            if len(dormant_tools) >= max(3, total_tools * 0.5) and latest.prompt_breakdown.tool_schemas > 500:
                savings = (len(dormant_tools) / total_tools) * (latest.prompt_breakdown.tool_schemas / max(1, latest.input_tokens)) * 100
                advisories.append(
                    AdvisoryMessage(
                        advisory_type="TOOL_PRUNING",
                        severity="WARNING",
                        agent_id=agent_id,
                        node_id=latest.node_id,
                        summary=f"{len(dormant_tools)} of {total_tools} tools have been dormant across recent turns.",
                        estimated_savings_percent=round(savings, 1),
                        recommendation=f"Enable dynamic tool filtering to prune schemas for: {', '.join(dormant_tools[:4])}..."
                    )
                )

        # 2. Check Telemetry Delta Compaction
        if latest.prompt_breakdown.iot_telemetry > 400 and (latest.prompt_breakdown.iot_telemetry / max(1, latest.input_tokens)) > 0.4:
            advisories.append(
                AdvisoryMessage(
                    advisory_type="DELTA_COMPACTION",
                    severity="WARNING",
                    agent_id=agent_id,
                    node_id=latest.node_id,
                    summary="Raw continuous telemetry sensor dumps detected in prompt context.",
                    estimated_savings_percent=65.0,
                    recommendation="Switch agent prompt from full state dumps to delta/event-driven state diffs."
                )
            )

        # 3. Check Cache Alignment (Prompt Caching)
        if latest.cached_read_tokens == 0 and latest.prompt_breakdown.system_instructions > 800:
            advisories.append(
                AdvisoryMessage(
                    advisory_type="CACHE_ALIGNMENT",
                    severity="INFO",
                    agent_id=agent_id,
                    node_id=latest.node_id,
                    summary="Zero prompt cache hits detected on large static system instructions.",
                    estimated_savings_percent=70.0,
                    recommendation="Ensure static system instructions and tools are placed strictly at the prompt prefix before dynamic timestamps."
                )
            )

        return advisories
