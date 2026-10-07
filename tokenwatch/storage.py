import os
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import List, Optional, Dict, Any
from tokenwatch.models import TelemetryEvent, PromptBreakdown, AdvisoryMessage, NodeHeartbeat

class SQLiteStorage:
    """
    100% private, self-contained embedded storage powered by Python's built-in sqlite3.
    Uses Write-Ahead Logging (WAL) for high concurrency and zero external binary dependencies.
    """
    def __init__(self, db_path: str = "data/tokenwatch.db"):
        self.is_memory = db_path == ":memory:"
        if not self.is_memory:
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.db_path = db_path
        self._mem_conn = sqlite3.connect(":memory:") if self.is_memory else None
        if self._mem_conn:
            self._mem_conn.row_factory = sqlite3.Row
        self._init_schema()

    @contextmanager
    def _get_connection(self):
        """
        Yields an open SQLite connection with WAL enabled and guarantees
        explicit connection closing upon exit to prevent open file descriptor leaks.
        """
        if self.is_memory and self._mem_conn:
            with self._mem_conn:
                yield self._mem_conn
        else:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            try:
                with conn:
                    yield conn
            finally:
                conn.close()

    def _init_schema(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_events (
                    trace_id TEXT PRIMARY KEY,
                    correlation_id TEXT,
                    timestamp REAL,
                    node_id TEXT,
                    agent_id TEXT,
                    task_type TEXT,
                    model TEXT,
                    streaming INTEGER,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cached_read_tokens INTEGER,
                    cached_write_tokens INTEGER,
                    cost_usd REAL,
                    latency_ms REAL,
                    prompt_breakdown_json TEXT,
                    tools_declared_json TEXT,
                    tools_invoked_json TEXT,
                    raw_usage_json TEXT
                );
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_agent_time ON telemetry_events (agent_id, timestamp DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_corr ON telemetry_events (correlation_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_time ON telemetry_events (timestamp DESC);")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS advisories (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    advisory_type TEXT,
                    severity TEXT,
                    agent_id TEXT,
                    node_id TEXT,
                    summary TEXT,
                    estimated_savings_percent REAL,
                    recommendation TEXT,
                    actionable_fix_json TEXT
                );
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_advisories_agent ON advisories (agent_id, timestamp DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_advisories_time ON advisories (timestamp DESC);")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS node_heartbeats (
                    node_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    status TEXT,
                    ip_address TEXT,
                    memory_mb REAL,
                    backlog_events INTEGER,
                    active_agents_json TEXT
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS tripped_circuits (
                    circuit_key TEXT PRIMARY KEY, -- e.g. correlation_id or agent_id
                    tripped_at REAL,
                    reason TEXT
                );
            """)

    def save_event(self, event: TelemetryEvent):
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO telemetry_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.trace_id,
                event.correlation_id,
                event.timestamp,
                event.node_id,
                event.agent_id,
                event.task_type,
                event.model,
                1 if event.streaming else 0,
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

    def get_recent_events(
        self,
        limit: int = 100,
        agent_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> List[TelemetryEvent]:
        query = "SELECT * FROM telemetry_events"
        params: List[Any] = []
        conditions = []
        if agent_id:
            conditions.append("agent_id = ?")
            params.append(agent_id)
        if correlation_id:
            conditions.append("correlation_id = ?")
            params.append(correlation_id)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            events = []
            for r in rows:
                events.append(TelemetryEvent(
                    trace_id=r["trace_id"],
                    correlation_id=r["correlation_id"],
                    timestamp=r["timestamp"],
                    node_id=r["node_id"],
                    agent_id=r["agent_id"],
                    task_type=r["task_type"],
                    model=r["model"],
                    streaming=bool(r["streaming"]),
                    input_tokens=r["input_tokens"],
                    output_tokens=r["output_tokens"],
                    cached_read_tokens=r["cached_read_tokens"],
                    cached_write_tokens=r["cached_write_tokens"],
                    cost_usd=r["cost_usd"],
                    latency_ms=r["latency_ms"],
                    prompt_breakdown=PromptBreakdown.model_validate_json(r["prompt_breakdown_json"]),
                    tools_declared=json.loads(r["tools_declared_json"]) if r["tools_declared_json"] else [],
                    tools_invoked=json.loads(r["tools_invoked_json"]) if r["tools_invoked_json"] else [],
                    raw_usage=json.loads(r["raw_usage_json"]) if r["raw_usage_json"] else None
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
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            summaries = []
            for r in rows:
                inp = r["total_input_tokens"] or 0
                cached = r["total_cached_tokens"] or 0
                summaries.append({
                    "agent_id": r["agent_id"],
                    "invocation_count": r["invocation_count"],
                    "total_spend_usd": round(r["total_spend_usd"] or 0.0, 6),
                    "total_input_tokens": inp,
                    "total_output_tokens": r["total_output_tokens"] or 0,
                    "total_cached_tokens": cached,
                    "cache_hit_ratio": round(cached / max(1, inp), 3),
                    "avg_latency_ms": round(r["avg_latency_ms"] or 0.0, 1)
                })
            return summaries

    def save_advisory(self, advisory: AdvisoryMessage):
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO advisories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
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

    def get_recent_advisories(self, limit: int = 50, agent_id: Optional[str] = None) -> List[AdvisoryMessage]:
        query = "SELECT * FROM advisories"
        params: List[Any] = []
        if agent_id:
            query += " WHERE agent_id = ?"
            params.append(agent_id)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            advisories = []
            for r in rows:
                advisories.append(AdvisoryMessage(
                    advisory_type=r["advisory_type"],
                    severity=r["severity"],
                    agent_id=r["agent_id"],
                    node_id=r["node_id"],
                    summary=r["summary"],
                    estimated_savings_percent=r["estimated_savings_percent"],
                    recommendation=r["recommendation"],
                    actionable_fix=json.loads(r["actionable_fix_json"]) if r["actionable_fix_json"] else None
                ))
            return advisories

    def record_heartbeat(self, heartbeat: NodeHeartbeat):
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO node_heartbeats VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                heartbeat.node_id,
                heartbeat.timestamp,
                heartbeat.status,
                heartbeat.ip_address,
                heartbeat.memory_mb,
                heartbeat.backlog_events,
                json.dumps(heartbeat.active_agents)
            ))

    def get_node_statuses(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM node_heartbeats ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            now = time.time()
            nodes = []
            for r in rows:
                is_stale = (now - r["timestamp"]) > 90.0
                nodes.append({
                    "node_id": r["node_id"],
                    "last_seen_seconds_ago": round(now - r["timestamp"], 1),
                    "status": "OFFLINE" if is_stale else r["status"],
                    "ip_address": r["ip_address"],
                    "memory_mb": r["memory_mb"],
                    "backlog_events": r["backlog_events"],
                    "active_agents": json.loads(r["active_agents_json"]) if r["active_agents_json"] else []
                })
            return nodes

    def trip_circuit_breaker(self, circuit_key: str, reason: str):
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tripped_circuits VALUES (?, ?, ?)
            """, (circuit_key, time.time(), reason))

    def is_circuit_tripped(self, circuit_key: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tripped_circuits WHERE circuit_key = ?", (circuit_key,))
            row = cursor.fetchone()
            if row:
                return {
                    "circuit_key": row["circuit_key"],
                    "tripped_at": row["tripped_at"],
                    "reason": row["reason"]
                }
            return None

    def reset_circuit_breaker(self, circuit_key: str):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM tripped_circuits WHERE circuit_key = ?", (circuit_key,))
