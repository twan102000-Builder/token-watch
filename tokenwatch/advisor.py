from typing import List, Dict, Any, Optional
from tokenwatch.models import TelemetryEvent, AdvisoryMessage
from tokenwatch.storage import DuckDBStorage

class TokenAdvisorEngine:
    """
    Evaluates telemetry events across agents, nodes, and multi-agent correlation trees
    to generate actionable, high-impact token and cost reduction advisories.
    """
    def __init__(self, storage: Optional[DuckDBStorage] = None):
        self.storage = storage or DuckDBStorage(db_path=":memory:")

    def record_event(self, event: TelemetryEvent):
        self.storage.save_event(event)

    def generate_advisories(self, agent_id: str) -> List[AdvisoryMessage]:
        advisories: List[AdvisoryMessage] = []
        recent_events = self.storage.get_recent_events(limit=50, agent_id=agent_id)
        if not recent_events:
            return advisories

        latest = recent_events[0]

        # 1. Dormant Tool Schema Bloat Analysis
        if latest.tools_declared:
            total_tools = len(latest.tools_declared)
            invoked_set = set()
            for e in recent_events[:30]:
                invoked_set.update(e.tools_invoked)
            
            dormant_tools = [t for t in latest.tools_declared if t not in invoked_set]
            if len(dormant_tools) >= max(3, total_tools * 0.4) and latest.prompt_breakdown.tool_schemas > 400:
                # Estimate token savings
                tool_ratio = len(dormant_tools) / total_tools
                wasted_tokens = int(latest.prompt_breakdown.tool_schemas * tool_ratio)
                savings_pct = (wasted_tokens / max(1, latest.input_tokens)) * 100.0

                advisories.append(
                    AdvisoryMessage(
                        advisory_type="TOOL_PRUNING",
                        severity="WARNING",
                        agent_id=agent_id,
                        node_id=latest.node_id,
                        summary=f"{len(dormant_tools)} of {total_tools} tools declared were never invoked across the last {len(recent_events[:30])} turns.",
                        estimated_savings_percent=round(savings_pct, 1),
                        recommendation=f"Prune dormant tool schemas ({', '.join(dormant_tools[:4])}...). Save ~{wasted_tokens} tokens per request.",
                        actionable_fix={
                            "action": "enable_dynamic_tool_pruning",
                            "prune_candidates": dormant_tools,
                            "estimated_token_savings": wasted_tokens
                        }
                    )
                )

        # 2. Cache Alignment & Prefix Jitter Detection
        if latest.cached_read_tokens == 0 and latest.prompt_breakdown.system_instructions > 600:
            advisories.append(
                AdvisoryMessage(
                    advisory_type="CACHE_ALIGNMENT",
                    severity="INFO",
                    agent_id=agent_id,
                    node_id=latest.node_id,
                    summary="Zero prompt cache hits detected on large static system instructions.",
                    estimated_savings_percent=75.0,
                    recommendation="Reorder prompt: Place static system instructions and tool definitions at the prefix before dynamic telemetry.",
                    actionable_fix={"action": "reorder_prefix", "header": "X-Stabilize-Prefix: true"}
                )
            )

        # 3. Telemetry State Compaction (IoT sensor spam)
        if latest.prompt_breakdown.iot_telemetry > 350 and (latest.prompt_breakdown.iot_telemetry / max(1, latest.input_tokens)) > 0.35:
            advisories.append(
                AdvisoryMessage(
                    advisory_type="DELTA_COMPACTION",
                    severity="WARNING",
                    agent_id=agent_id,
                    node_id=latest.node_id,
                    summary="High proportion of raw IoT telemetry sensor state dumped into prompt.",
                    estimated_savings_percent=60.0,
                    recommendation="Enable Telemetry Compactor: pass entity state changes (diffs) instead of repeated static entity tables.",
                    actionable_fix={"action": "enable_telemetry_diffs"}
                )
            )

        # 4. Model Downgrade / Cascading Advisory
        expensive_models = ["claude-3-5-sonnet", "gpt-4o", "o1", "claude-3-opus"]
        if any(m in latest.model.lower() for m in expensive_models):
            # If agent tasks are simple (short output, few tools or binary answers)
            avg_out = sum(e.output_tokens for e in recent_events[:10]) / max(1, len(recent_events[:10]))
            if avg_out < 60 and len(latest.tools_invoked) <= 1:
                advisories.append(
                    AdvisoryMessage(
                        advisory_type="MODEL_CASCADING",
                        severity="INFO",
                        agent_id=agent_id,
                        node_id=latest.node_id,
                        summary=f"Agent '{agent_id}' uses tier-1 model ({latest.model}) for low-complexity turns (avg {int(avg_out)} tokens out).",
                        estimated_savings_percent=85.0,
                        recommendation="Route to fast tier model (e.g. 'gemini-2.5-flash' or local 'llama-3.1-8b') for 85%+ cost reduction.",
                        actionable_fix={"recommended_model": "gemini-2.5-flash"}
                    )
                )

        # 5. Multi-Agent Ping-Pong Loop Detection
        if latest.correlation_id:
            chain = [e for e in self.storage.get_recent_events(limit=20) if e.correlation_id == latest.correlation_id]
            if len(chain) >= 4:
                # Check rapid succession
                time_span = abs(chain[0].timestamp - chain[-1].timestamp)
                if time_span < 60.0:
                    advisories.append(
                        AdvisoryMessage(
                            advisory_type="LOOP_DETECTED",
                            severity="CRITICAL",
                            agent_id=agent_id,
                            node_id=latest.node_id,
                            summary=f"Rapid multi-agent chatter loop detected: {len(chain)} turns in {round(time_span, 1)}s across correlation ID {latest.correlation_id}.",
                            estimated_savings_percent=90.0,
                            recommendation="Trigger Circuit Breaker: force termination of recursive handoffs or inject exit directive.",
                            actionable_fix={"action": "circuit_breaker_trip", "correlation_id": latest.correlation_id}
                        )
                    )

        # Save generated advisories
        for adv in advisories:
            self.storage.save_advisory(adv)

        return advisories
