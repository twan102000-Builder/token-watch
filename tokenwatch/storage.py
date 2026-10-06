import os
import json
import duckdb
from typing import List, Optional, Dict, Any
from tokenwatch.models import TelemetryEvent, PromptBreakdown, AdvisoryMessage

class DuckDBStorage:
    def __init__(self, db_path: str = "data/tokenwatch.duckdb"):
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = duckdb.connect(database=db_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_events (
                trace_id VARCHAR PRIMARY KEY,
                correlation_id VARCHAR,
                timestamp DOUBLE,
                node_id VARCHAR,
                agent_id VARCHAR,
                task_type VARCHAR,
                model VARCHAR,
                streaming BOOLEAN,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cached_read_tokens INTEGER,
                cached_write_tokens INTEGER,
                cost_usd DOUBLE,
                latency_ms DOUBLE,
                prompt_breakdown_json VARCHAR,
                tools_declared_json VARCHAR,
                tools_invoked_json VARCHAR,
                raw_usage_json VARCHAR
            );
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS advisories (
                id VARCHAR DEFAULT uuid(),
                timestamp DOUBLE,
                advisory_type VARCHAR,
                severity VARCHAR,
                agent_id VARCHAR,
                node_id VARCHAR,
                summary VARCHAR,
                estimated_savings_percent DOUBLE,
                recommendation VARCHAR,
                actionable_fix_json VARCHAR
            );
        """)

    def save_event(self, event: TelemetryEvent):
        self.conn.execute("""
            INSERT OR REPLACE INTO telemetry_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.trace_id,
            event.correlation_id,
            event.timestamp,
            event.node_id,
            event.agent_id,
            event.task_type,
            event.model,
            event.streaming,
            event.input_tokens,
            event.output_tokens,
            event.cached_read_tokens,
            event.cached_write_tokens,
            event.cost_usd,
            event.latency_ms,
            event.prompt_breakdown.model_dump_json(),
            json.dumps(event.tools_declared),
            json.dumps(event.tools_invoked),
            json.dumps(event.raw_usage) if event.raw_usage else None
        ))

    def get_recent_events(self, limit: int = 100, agent_id: Optional[str] = None) -> List[TelemetryEvent]:
        query = "SELECT * FROM telemetry_events"
        params = []
        if agent_id:
            query += " WHERE agent_id = ?"
            params.append(agent_id)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        events = []
        for r in rows:
            events.append(TelemetryEvent(
                trace_id=r[0],
                correlation_id=r[1],
                timestamp=r[2],
                node_id=r[3],
                agent_id=r[4],
                task_type=r[5],
                model=r[6],
                streaming=r[7],
                input_tokens=r[8],
                output_tokens=r[9],
                cached_read_tokens=r[10],
                cached_write_tokens=r[11],
                cost_usd=r[12],
                latency_ms=r[13],
                prompt_breakdown=PromptBreakdown.model_validate_json(r[14]),
                tools_declared=json.loads(r[15]) if r[15] else [],
                tools_invoked=json.loads(r[16]) if r[16] else [],
                raw_usage=json.loads(r[17]) if r[17] else None
            ))
        return events

    def get_agent_metrics_summary(self) -> List[Dict[str, Any]]:
        query = """
            SELECT 
                agent_id,
                COUNT(*) as invocation_count,
                SUM(cost_usd) as total_spend_usd,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(cached_read_tokens) as total_cached_tokens,
                AVG(latency_ms) as avg_latency_ms
            FROM telemetry_events
            GROUP BY agent_id
            ORDER BY total_spend_usd DESC
        """
        results = self.conn.execute(query).fetchall()
        summaries = []
        for r in results:
            summaries.append({
                "agent_id": r[0],
                "invocation_count": r[1],
                "total_spend_usd": round(r[2], 6),
                "total_input_tokens": r[3],
                "total_output_tokens": r[4],
                "total_cached_tokens": r[5],
                "cache_hit_ratio": round(r[5] / max(1, r[3]), 3),
                "avg_latency_ms": round(r[6] or 0.0, 1)
            })
        return summaries

    def save_advisory(self, advisory: AdvisoryMessage):
        import time
        self.conn.execute("""
            INSERT INTO advisories (timestamp, advisory_type, severity, agent_id, node_id, summary, estimated_savings_percent, recommendation, actionable_fix_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(),
            advisory.advisory_type,
            advisory.severity,
            advisory.agent_id,
            advisory.node_id,
            advisory.summary,
            advisory.estimated_savings_percent,
            advisory.recommendation,
            json.dumps(advisory.actionable_fix) if advisory.actionable_fix else None
        ))
