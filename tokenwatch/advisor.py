from tokenwatch.models import AdvisoryMessage, TelemetryEvent
from tokenwatch.storage import SQLiteStorage


class TokenAdvisorEngine:
    """
    Evaluates telemetry events across agents, nodes, and multi-agent correlation trees
    to generate actionable, high-impact token and cost reduction advisories.
    Backed by 100% private, zero-external-dependency SQLite with WAL mode.
    """

    def __init__(self, storage: SQLiteStorage | None = None):
        self.storage = storage or SQLiteStorage(db_path=":memory:")

    def record_event(self, event: TelemetryEvent) -> None:
        self.storage.save_event(event)

    def generate_advisories(
        self,
        agent_id: str,
        persist: bool = True,
        trip_circuits: bool = True,
    ) -> list[AdvisoryMessage]:
        advisories: list[AdvisoryMessage] = []
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
                        summary=(
                            f"{len(dormant_tools)} of {total_tools} tools declared were never invoked "
                            f"across the last {len(recent_events[:30])} turns."
                        ),
                        estimated_savings_percent=round(savings_pct, 1),
                        recommendation=(
                            f"Prune dormant tool schemas ({', '.join(dormant_tools[:4])}...). "
                            f"Save ~{wasted_tokens} tokens per request."
                        ),
                        actionable_fix={
                            "action": "enable_dynamic_tool_pruning",
                            "prune_candidates": dormant_tools,
                            "estimated_token_savings": wasted_tokens,
                        },
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
                    recommendation=(
                        "Reorder prompt: Place static system instructions and tool definitions "
                        "at the prefix before dynamic telemetry."
                    ),
                    actionable_fix={"action": "reorder_prefix", "header": "X-Stabilize-Prefix: true"},
                )
            )

        # 3. Telemetry State Compaction (IoT sensor spam)
        if (
            latest.prompt_breakdown.iot_telemetry > 350
            and (latest.prompt_breakdown.iot_telemetry / max(1, latest.input_tokens)) > 0.35
        ):
            advisories.append(
                AdvisoryMessage(
                    advisory_type="DELTA_COMPACTION",
                    severity="WARNING",
                    agent_id=agent_id,
                    node_id=latest.node_id,
                    summary="High proportion of raw IoT telemetry sensor state dumped into prompt.",
                    estimated_savings_percent=60.0,
                    recommendation=(
                        "Enable Telemetry Compactor: pass entity state changes (diffs) "
                        "instead of repeated static entity tables."
                    ),
                    actionable_fix={"action": "enable_telemetry_diffs"},
                )
            )

        # 4. Model Downgrade / Cascading Advisory
        expensive_models = ["claude-3-7-sonnet", "claude-3-5-sonnet", "gpt-4o", "o1", "claude-3-opus"]
        lower_model = latest.model.lower()
        if any(m in lower_model for m in expensive_models) and "mini" not in lower_model:
            avg_out = sum(e.output_tokens for e in recent_events[:10]) / max(1, len(recent_events[:10]))
            if avg_out < 60 and len(latest.tools_invoked) <= 1:
                advisories.append(
                    AdvisoryMessage(
                        advisory_type="MODEL_CASCADING",
                        severity="INFO",
                        agent_id=agent_id,
                        node_id=latest.node_id,
                        summary=(
                            f"Agent '{agent_id}' uses tier-1 model ({latest.model}) "
                            f"for low-complexity turns (avg {int(avg_out)} tokens out)."
                        ),
                        estimated_savings_percent=95.0,
                        recommendation=(
                            "Route to Tier 1 local model (e.g. 'gemma-4-12b' or 'gemma-4-9b' on cluster edge) "
                            "for 100% cost reduction, or fast cloud tier ('gemini-2.5-flash')."
                        ),
                        actionable_fix={"recommended_model": "gemma-4-12b"},
                    )
                )

        # 5. Multi-Agent Ping-Pong Loop Detection & Active Circuit Trip (4-turn sliding window)
        if latest.correlation_id:
            chain = self.storage.get_recent_events(limit=20, correlation_id=latest.correlation_id)
            if len(chain) >= 4:
                # Sliding window of the 4 most recent turns (chain is ordered timestamp DESC)
                time_span = abs(chain[0].timestamp - chain[3].timestamp)
                if time_span < 60.0:
                    trip_reason = (
                        f"Rapid multi-agent chatter loop detected: 4 turns in "
                        f"{round(time_span, 1)}s across correlation ID {latest.correlation_id}"
                    )
                    if trip_circuits:
                        self.storage.trip_circuit_breaker(latest.correlation_id, trip_reason)

                    advisories.append(
                        AdvisoryMessage(
                            advisory_type="LOOP_DETECTED",
                            severity="CRITICAL",
                            agent_id=agent_id,
                            node_id=latest.node_id,
                            summary=trip_reason,
                            estimated_savings_percent=90.0,
                            recommendation=(
                                "Circuit breaker TRIPPED: recursive handoffs halted. "
                                "Requests on this correlation ID will be blocked."
                            ),
                            actionable_fix={"action": "circuit_breaker_trip", "correlation_id": latest.correlation_id},
                        )
                    )

        # Save generated advisories when persist=True
        if persist:
            for adv in advisories:
                self.storage.save_advisory(adv)

        return advisories

