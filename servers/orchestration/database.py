"""
Orchestration Database — SQLite schema and DatabaseManager.

Tables
------
server_profiles    Static capability profile per MCP server, updated on each
                   discovery or re-rate pass.

workflow_templates Reusable named workflow definitions: a task type maps to an
                   ordered JSON list of server IDs and tool names.

workflow_runs      Runtime execution log: every plan that was approved and
                   dispatched gets a row here so that governance reports can
                   always trace back to the allocation decision.

invocation_log     Per-tool-call telemetry used to feed the rater (latency,
                   success/failure, error message).

agent_assignments  Maps AI agent roles to server slots within a workflow run so
                   the coordinator can audit hot-swaps and role conflicts.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL — every table is created with IF NOT EXISTS so the manager is safe to
# call on an existing database during upgrades.
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS server_profiles (
    server_id       TEXT PRIMARY KEY,
    domain          TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    entry_point     TEXT NOT NULL DEFAULT '',
    tools_json      TEXT NOT NULL DEFAULT '[]',
    requires_gpu    INTEGER NOT NULL DEFAULT 0,
    min_ram_gb      REAL NOT NULL DEFAULT 0.0,
    extras_key      TEXT,
    annotations_json TEXT NOT NULL DEFAULT '{}',
    -- Composite score written by the rater (0.0 – 100.0)
    score           REAL,
    score_breakdown_json TEXT NOT NULL DEFAULT '{}',
    last_seen_at    TEXT NOT NULL,
    last_rated_at   TEXT
);

CREATE TABLE IF NOT EXISTS workflow_templates (
    template_id     TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    task_type       TEXT NOT NULL,
    -- Ordered JSON array of {server_id, tool_name, param_defaults} dicts
    steps_json      TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workflow_runs (
    run_id          TEXT PRIMARY KEY,
    template_id     TEXT REFERENCES workflow_templates(template_id),
    task_descriptor_json TEXT NOT NULL DEFAULT '{}',
    -- Full WorkflowPlan as JSON (including fallback alternatives)
    plan_json       TEXT NOT NULL DEFAULT '{}',
    status          TEXT NOT NULL DEFAULT 'pending',
    -- Governance gate decision: 'approved' | 'rejected'
    governance_decision TEXT NOT NULL DEFAULT 'pending',
    governance_notes_json TEXT NOT NULL DEFAULT '[]',
    started_at      TEXT,
    completed_at    TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS invocation_log (
    invocation_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT REFERENCES workflow_runs(run_id),
    server_id       TEXT NOT NULL,
    tool_name       TEXT NOT NULL,
    latency_ms      REAL,
    success         INTEGER NOT NULL DEFAULT 1,
    error_message   TEXT,
    invoked_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_assignments (
    assignment_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES workflow_runs(run_id),
    agent_role      TEXT NOT NULL,
    server_id       TEXT NOT NULL,
    -- Reason the allocator picked this server (score, rule name, etc.)
    selection_reason TEXT NOT NULL DEFAULT '',
    -- 'active' | 'hot_swapped' | 'completed'
    status          TEXT NOT NULL DEFAULT 'active',
    assigned_at     TEXT NOT NULL,
    released_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_profiles_domain ON server_profiles(domain);
CREATE INDEX IF NOT EXISTS idx_profiles_score  ON server_profiles(score);
CREATE INDEX IF NOT EXISTS idx_runs_status     ON workflow_runs(status);
CREATE INDEX IF NOT EXISTS idx_invocations_server ON invocation_log(server_id);
CREATE INDEX IF NOT EXISTS idx_assignments_run ON agent_assignments(run_id);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class DatabaseManager:
    """
    Thread-safe SQLite wrapper for the orchestration database.

    A single *DatabaseManager* instance is shared across the orchestration
    server process.  All writes go through a threading.Lock so that concurrent
    tool calls do not corrupt WAL state.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()
        logger.info("DatabaseManager ready — %s", self._db_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(_DDL)
                conn.commit()
            finally:
                conn.close()

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a connection inside a transaction; auto-commits on exit."""
        with self._lock:
            conn = self._connect()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # server_profiles
    # ------------------------------------------------------------------

    def upsert_server_profile(
        self,
        server_id: str,
        domain: str,
        description: str,
        entry_point: str,
        tools: List[Dict[str, Any]],
        requires_gpu: bool = False,
        min_ram_gb: float = 0.0,
        extras_key: Optional[str] = None,
        annotations: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or update a server profile (score fields are left unchanged on update)."""
        now = _utcnow()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO server_profiles
                    (server_id, domain, description, entry_point, tools_json,
                     requires_gpu, min_ram_gb, extras_key, annotations_json, last_seen_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(server_id) DO UPDATE SET
                    domain          = excluded.domain,
                    description     = excluded.description,
                    entry_point     = excluded.entry_point,
                    tools_json      = excluded.tools_json,
                    requires_gpu    = excluded.requires_gpu,
                    min_ram_gb      = excluded.min_ram_gb,
                    extras_key      = excluded.extras_key,
                    annotations_json = excluded.annotations_json,
                    last_seen_at    = excluded.last_seen_at
                """,
                (
                    server_id,
                    domain,
                    description,
                    entry_point,
                    json.dumps(tools),
                    int(requires_gpu),
                    min_ram_gb,
                    extras_key,
                    json.dumps(annotations or {}),
                    now,
                ),
            )

    def update_server_score(
        self,
        server_id: str,
        score: float,
        breakdown: Dict[str, Any],
    ) -> None:
        """Write a new composite score and breakdown for a server."""
        with self._tx() as conn:
            conn.execute(
                """
                UPDATE server_profiles
                SET score = ?, score_breakdown_json = ?, last_rated_at = ?
                WHERE server_id = ?
                """,
                (score, json.dumps(breakdown), _utcnow(), server_id),
            )

    def get_server_profile(self, server_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM server_profiles WHERE server_id = ?", (server_id,)
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def list_server_profiles(
        self,
        domain: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return profiles, optionally filtered by domain and minimum score."""
        query = "SELECT * FROM server_profiles WHERE 1=1"
        params: List[Any] = []
        if domain:
            query += " AND domain = ?"
            params.append(domain)
        if min_score is not None:
            query += " AND (score IS NULL OR score >= ?)"
            params.append(min_score)
        query += " ORDER BY COALESCE(score, 0) DESC"
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(query, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # workflow_templates
    # ------------------------------------------------------------------

    def upsert_workflow_template(
        self,
        template_id: str,
        name: str,
        description: str,
        task_type: str,
        steps: List[Dict[str, Any]],
    ) -> None:
        now = _utcnow()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO workflow_templates
                    (template_id, name, description, task_type, steps_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(template_id) DO UPDATE SET
                    name        = excluded.name,
                    description = excluded.description,
                    task_type   = excluded.task_type,
                    steps_json  = excluded.steps_json,
                    updated_at  = excluded.updated_at
                """,
                (template_id, name, description, task_type, json.dumps(steps), now, now),
            )

    def get_workflow_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM workflow_templates WHERE template_id = ?", (template_id,)
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def list_workflow_templates(self, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM workflow_templates"
        params: List[Any] = []
        if task_type:
            query += " WHERE task_type = ?"
            params.append(task_type)
        query += " ORDER BY updated_at DESC"
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(query, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # workflow_runs
    # ------------------------------------------------------------------

    def create_workflow_run(
        self,
        run_id: str,
        task_descriptor: Dict[str, Any],
        plan: Dict[str, Any],
        template_id: Optional[str] = None,
    ) -> None:
        now = _utcnow()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO workflow_runs
                    (run_id, template_id, task_descriptor_json, plan_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, template_id, json.dumps(task_descriptor), json.dumps(plan), now),
            )

    def update_workflow_run_status(
        self,
        run_id: str,
        status: str,
        governance_decision: Optional[str] = None,
        governance_notes: Optional[List[str]] = None,
    ) -> None:
        now = _utcnow()
        with self._tx() as conn:
            if status in ("running",):
                conn.execute(
                    "UPDATE workflow_runs SET status = ?, started_at = ? WHERE run_id = ?",
                    (status, now, run_id),
                )
            elif status in ("completed", "failed", "rejected"):
                updates = "status = ?, completed_at = ?"
                params: List[Any] = [status, now]
                if governance_decision is not None:
                    updates += ", governance_decision = ?"
                    params.append(governance_decision)
                if governance_notes is not None:
                    updates += ", governance_notes_json = ?"
                    params.append(json.dumps(governance_notes))
                params.append(run_id)
                conn.execute(
                    f"UPDATE workflow_runs SET {updates} WHERE run_id = ?", params
                )
            else:
                conn.execute(
                    "UPDATE workflow_runs SET status = ? WHERE run_id = ?",
                    (status, run_id),
                )

    def get_workflow_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM workflow_runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # invocation_log
    # ------------------------------------------------------------------

    def log_invocation(
        self,
        run_id: str,
        server_id: str,
        tool_name: str,
        latency_ms: Optional[float],
        success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO invocation_log
                    (run_id, server_id, tool_name, latency_ms, success, error_message, invoked_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, server_id, tool_name, latency_ms, int(success), error_message, _utcnow()),
            )

    def get_invocation_stats(self, server_id: str) -> Dict[str, Any]:
        """Return aggregate stats for a server — consumed by the rater."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT
                        COUNT(*)                        AS total_calls,
                        SUM(success)                    AS successful_calls,
                        AVG(latency_ms)                 AS avg_latency_ms,
                        MIN(latency_ms)                 AS min_latency_ms,
                        MAX(latency_ms)                 AS max_latency_ms
                    FROM invocation_log
                    WHERE server_id = ?
                    """,
                    (server_id,),
                ).fetchone()
                stats = dict(row) if row else {}
                total = stats.get("total_calls") or 0
                successful = stats.get("successful_calls") or 0
                stats["success_rate"] = (successful / total) if total > 0 else None
                return stats
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # agent_assignments
    # ------------------------------------------------------------------

    def assign_agent(
        self,
        run_id: str,
        agent_role: str,
        server_id: str,
        selection_reason: str,
    ) -> int:
        with self._tx() as conn:
            cursor = conn.execute(
                """
                INSERT INTO agent_assignments
                    (run_id, agent_role, server_id, selection_reason, assigned_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, agent_role, server_id, selection_reason, _utcnow()),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def release_agent(
        self,
        assignment_id: int,
        status: str = "completed",
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE agent_assignments SET status = ?, released_at = ? WHERE assignment_id = ?",
                (status, _utcnow(), assignment_id),
            )

    def list_agent_assignments(self, run_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM agent_assignments WHERE run_id = ? ORDER BY assigned_at",
                    (run_id,),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()
