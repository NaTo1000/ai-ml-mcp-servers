"""
Overlay Authority & Logic-Control Algorithm — Build Control for VS Code / Visual Studio.

Overview
--------
``BuildAuthority`` is the single control point that all VS Code tasks and
launch configurations call before any MCP server is started or any build step
is executed.  It implements a three-stage gate pipeline:

    Stage 1 — PROFILE GATE
        Verify that the target server has a current profile in the orchestration
        registry.  If the profile is missing, trigger an on-demand discovery
        pass.  A server with no discoverable profile is *always* rejected.

    Stage 2 — SCORE GATE
        Compare the server's composite score (set by the rater) against the
        ``min_score_threshold`` in ``orchestration_config.json``.  Unrated
        servers are allowed through only when ``allow_unrated_servers`` is
        ``true`` in the config (default: true) so that a fresh workspace where
        no rating pass has run yet does not block all development.

    Stage 3 — RESOURCE GATE
        Check the server's declared ``requires_gpu`` and ``min_ram_gb`` against
        the runtime budget.  If the host does not meet the minimum requirements,
        the gate rejects with an explanatory message so the developer knows
        exactly what hardware constraint was violated.

Every gate decision — pass *or* reject — is written as a structured JSON
record to ``.vscode/authority_log.json`` so that governance reports can
reconstruct the full audit trail of build decisions.

CLI Usage
---------
The module doubles as a command-line tool invoked by the VS Code tasks:

    python -m servers.orchestration.build_authority discover [--verbose]
        Run a full discovery + rating pass and print a summary table.

    python -m servers.orchestration.build_authority gate --server <id>
        Run all three gates for a server and exit 0 (pass) or 1 (reject).

    python -m servers.orchestration.build_authority run --server <id>
        Gate-check then launch the server's entry-point in a subprocess.

    python -m servers.orchestration.build_authority report
        Print the last N governance records from authority_log.json.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "orchestration_config.json"
_DEFAULT_AUTH_LOG = _REPO_ROOT / ".vscode" / "authority_log.json"
_DEFAULT_DB = Path.home() / ".local" / "share" / "ai-ml-mcp-servers" / "orchestration.db"

# ---------------------------------------------------------------------------
# Gate result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """Outcome of a single authority gate check."""

    gate: str          # "profile" | "score" | "resource"
    passed: bool
    server_id: str
    reason: str
    detail: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "server_id": self.server_id,
            "reason": self.reason,
            "detail": self.detail,
            "timestamp": self.timestamp,
        }


@dataclass
class AuthorityDecision:
    """Aggregate outcome after all gates have been evaluated."""

    server_id: str
    approved: bool
    gate_results: List[GateResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def rejection_reason(self) -> Optional[str]:
        for r in self.gate_results:
            if not r.passed:
                return f"[{r.gate.upper()} GATE] {r.reason}"
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "server_id": self.server_id,
            "approved": self.approved,
            "rejection_reason": self.rejection_reason,
            "gate_results": [g.to_dict() for g in self.gate_results],
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def _load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    if config_path is not None:
        path: Optional[Path] = config_path
    else:
        env_val = os.environ.get("ORCHESTRATION_CONFIG", "").strip()
        path = Path(env_val) if env_val else _DEFAULT_CONFIG
    if path and path.exists() and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    # Sensible built-in defaults when no config file is present.
    return {
        "governance": {
            "min_score_threshold": 60.0,
            "max_concurrent_servers": 4,
            "max_total_ram_gb": 32.0,
            "allow_unrated_servers": True,
        }
    }


# ---------------------------------------------------------------------------
# Runtime resource probe
# ---------------------------------------------------------------------------


def _probe_resources() -> Dict[str, Any]:
    """
    Return a dict of available system resources.

    Uses only stdlib so it works in environments where psutil is absent.
    """
    resources: Dict[str, Any] = {
        "platform": platform.system(),
        "python": sys.version,
        "ram_total_gb": None,
        "ram_available_gb": None,
        "gpu_available": False,
        "gpu_names": [],
    }

    # RAM via psutil if available, else /proc/meminfo on Linux.
    try:
        import psutil  # type: ignore[import]
        vm = psutil.virtual_memory()
        resources["ram_total_gb"] = round(vm.total / 1e9, 2)
        resources["ram_available_gb"] = round(vm.available / 1e9, 2)
    except ImportError:
        try:
            meminfo = Path("/proc/meminfo").read_text()
            for line in meminfo.splitlines():
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    resources["ram_total_gb"] = round(kb / 1e6, 2)
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    resources["ram_available_gb"] = round(kb / 1e6, 2)
        except Exception:
            pass

    # GPU via torch if available.
    try:
        import torch  # type: ignore[import]
        if torch.cuda.is_available():
            resources["gpu_available"] = True
            resources["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            resources["gpu_available"] = True
            resources["gpu_names"] = ["Apple MPS"]
    except Exception:
        pass

    return resources


# ---------------------------------------------------------------------------
# BuildAuthority — the overlay authority class
# ---------------------------------------------------------------------------


class BuildAuthority:
    """
    Overlay authority for build control.

    Instantiated once per process and reused for multiple gate checks.

    Parameters
    ----------
    config_path:
        Override the default config file location.
    db_path:
        Override the default SQLite database path.
    auth_log_path:
        Override the default authority log file location.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
        auth_log_path: Optional[Path] = None,
    ) -> None:
        self._config = _load_config(config_path)
        self._db_path = db_path or Path(
            self._config.get("database", {}).get("path", str(_DEFAULT_DB))
        ).expanduser()
        self._auth_log_path = auth_log_path or Path(
            self._config.get("authority_log_path", str(_DEFAULT_AUTH_LOG))
        ).expanduser()
        self._resources = _probe_resources()
        self._db: Optional[Any] = None  # Loaded lazily

    # ------------------------------------------------------------------
    # Database access (lazy — avoids import errors on cold installs)
    # ------------------------------------------------------------------

    def _get_db(self) -> Any:
        if self._db is None:
            from servers.orchestration.database import DatabaseManager
            self._db = DatabaseManager(self._db_path)
        return self._db

    # ------------------------------------------------------------------
    # Gate 1 — Profile Gate
    # ------------------------------------------------------------------

    def _gate_profile(self, server_id: str) -> GateResult:
        """
        Ensure the server has a current profile in the registry.

        On a cache-miss, triggers an on-demand discovery pass so that a server
        only recently added to pyproject.toml is still usable.
        """
        db = self._get_db()
        profile = db.get_server_profile(server_id)

        if profile is None:
            # Attempt on-demand discovery.
            logger.info("Profile not found for '%s'; running on-demand discovery.", server_id)
            try:
                from servers.orchestration.registry import Registry
                reg = Registry(db)
                reg.discover()
                profile = db.get_server_profile(server_id)
            except Exception as exc:
                logger.debug("On-demand discovery failed: %s", exc)

        if profile is None:
            return GateResult(
                gate="profile",
                passed=False,
                server_id=server_id,
                reason=f"No profile found for server '{server_id}' even after on-demand discovery. "
                       "Ensure the server is declared in [project.scripts] in pyproject.toml.",
            )

        return GateResult(
            gate="profile",
            passed=True,
            server_id=server_id,
            reason="Profile found.",
            detail={
                "domain": profile.get("domain"),
                "last_seen_at": profile.get("last_seen_at"),
                "tool_count": len(json.loads(profile.get("tools_json", "[]"))),
            },
        )

    # ------------------------------------------------------------------
    # Gate 2 — Score Gate
    # ------------------------------------------------------------------

    def _gate_score(self, server_id: str) -> GateResult:
        """
        Compare the server's composite score against the governance threshold.
        """
        threshold: float = self._config.get("governance", {}).get("min_score_threshold", 60.0)
        allow_unrated: bool = self._config.get("governance", {}).get("allow_unrated_servers", True)

        db = self._get_db()
        profile = db.get_server_profile(server_id)
        score = profile.get("score") if profile else None

        if score is None:
            if allow_unrated:
                return GateResult(
                    gate="score",
                    passed=True,
                    server_id=server_id,
                    reason="Server is unrated; proceeding because allow_unrated_servers=true.",
                    detail={"score": None, "threshold": threshold},
                )
            return GateResult(
                gate="score",
                passed=False,
                server_id=server_id,
                reason=(
                    f"Server '{server_id}' has no score and allow_unrated_servers=false. "
                    "Run 'Authority: Discover & Rate All Servers' first."
                ),
                detail={"score": None, "threshold": threshold},
            )

        passed = float(score) >= threshold
        return GateResult(
            gate="score",
            passed=passed,
            server_id=server_id,
            reason=(
                f"Score {score:.1f}% >= threshold {threshold:.1f}%."
                if passed
                else f"Score {score:.1f}% is below threshold {threshold:.1f}%."
            ),
            detail={
                "score": score,
                "threshold": threshold,
                "breakdown": json.loads(
                    (profile or {}).get("score_breakdown_json", "{}")
                ),
            },
        )

    # ------------------------------------------------------------------
    # Gate 3 — Resource Gate
    # ------------------------------------------------------------------

    def _gate_resource(self, server_id: str) -> GateResult:
        """
        Check that the host can satisfy the server's hardware requirements.
        """
        db = self._get_db()
        profile = db.get_server_profile(server_id)
        if profile is None:
            # Profile gate will have already failed; return a pass here so the
            # error is attributed to the correct gate.
            return GateResult(
                gate="resource",
                passed=True,
                server_id=server_id,
                reason="Skipped (profile gate failed).",
            )

        requires_gpu: bool = bool(profile.get("requires_gpu", False))
        min_ram_gb: float = float(profile.get("min_ram_gb", 0.0))
        available_ram: Optional[float] = self._resources.get("ram_available_gb")
        gpu_available: bool = bool(self._resources.get("gpu_available", False))

        failures: List[str] = []

        if requires_gpu and not gpu_available:
            failures.append(
                f"Server '{server_id}' requires a GPU but none was detected on this host."
            )

        if min_ram_gb > 0.0 and available_ram is not None and available_ram < min_ram_gb:
            failures.append(
                f"Server '{server_id}' needs {min_ram_gb:.1f} GB RAM but only "
                f"{available_ram:.1f} GB is available."
            )

        if failures:
            return GateResult(
                gate="resource",
                passed=False,
                server_id=server_id,
                reason=" | ".join(failures),
                detail={
                    "requires_gpu": requires_gpu,
                    "min_ram_gb": min_ram_gb,
                    "host_gpu_available": gpu_available,
                    "host_ram_available_gb": available_ram,
                },
            )

        return GateResult(
            gate="resource",
            passed=True,
            server_id=server_id,
            reason="Resource requirements met.",
            detail={
                "requires_gpu": requires_gpu,
                "min_ram_gb": min_ram_gb,
                "host_gpu_available": gpu_available,
                "host_ram_available_gb": available_ram,
            },
        )

    # ------------------------------------------------------------------
    # Main gate pipeline
    # ------------------------------------------------------------------

    def evaluate(self, server_id: str) -> AuthorityDecision:
        """
        Run all three gates sequentially and return an ``AuthorityDecision``.

        Gates are short-circuit: if the profile gate fails there is no point
        running the score or resource gates.
        """
        gate_results: List[GateResult] = []

        profile_result = self._gate_profile(server_id)
        gate_results.append(profile_result)

        if profile_result.passed:
            score_result = self._gate_score(server_id)
            gate_results.append(score_result)

            if score_result.passed:
                resource_result = self._gate_resource(server_id)
                gate_results.append(resource_result)

        approved = all(r.passed for r in gate_results)
        decision = AuthorityDecision(
            server_id=server_id,
            approved=approved,
            gate_results=gate_results,
        )
        self._log_decision(decision)
        return decision

    def evaluate_all(self) -> Dict[str, AuthorityDecision]:
        """Evaluate all known server profiles and return a mapping of decisions."""
        db = self._get_db()
        profiles = db.list_server_profiles()
        return {p["server_id"]: self.evaluate(p["server_id"]) for p in profiles}

    # ------------------------------------------------------------------
    # Authority log
    # ------------------------------------------------------------------

    def _log_decision(self, decision: AuthorityDecision) -> None:
        """Append the decision to the authority log file (JSON-lines format)."""
        try:
            self._auth_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._auth_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(decision.to_dict()) + "\n")
        except Exception as exc:
            logger.warning("Could not write authority log: %s", exc)

    def load_log(self, last_n: int = 50) -> List[Dict[str, Any]]:
        """Return the last N governance records from the authority log."""
        if not self._auth_log_path.exists():
            return []
        lines = self._auth_log_path.read_text(encoding="utf-8").splitlines()
        records = []
        for line in lines:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return records[-last_n:]

    # ------------------------------------------------------------------
    # Discovery + rating helper (used by CLI 'discover' sub-command)
    # ------------------------------------------------------------------

    def run_discovery(self, verbose: bool = False) -> List[Dict[str, Any]]:
        """Run a full registry discovery pass and return profile summaries."""
        from servers.orchestration.registry import Registry
        db = self._get_db()
        reg = Registry(db)
        profiles = reg.discover()
        if verbose:
            for p in profiles:
                score_str = f"{p.score:.1f}%" if p.score is not None else "unrated"
                tool_count = len(p.tools)
                print(
                    f"  {p.server_id:<20} domain={p.domain:<14} "
                    f"tools={tool_count:<3} score={score_str}"
                )
        return [p.to_dict() for p in profiles]

    # ------------------------------------------------------------------
    # Launch helper (used by CLI 'run' sub-command)
    # ------------------------------------------------------------------

    def launch_server(self, server_id: str) -> int:
        """
        Gate-check then launch a server's entry-point in a subprocess.

        Returns the subprocess exit code (or 1 if a gate rejected the server).
        """
        decision = self.evaluate(server_id)
        if not decision.approved:
            print(f"\n✗ AUTHORITY REJECTED: {decision.rejection_reason}", file=sys.stderr)
            return 1

        # Resolve entry point from the database profile.
        db = self._get_db()
        profile = db.get_server_profile(server_id)
        entry_point: str = profile.get("entry_point", "") if profile else ""
        if not entry_point or ":" not in entry_point:
            print(f"✗ Cannot resolve entry point for '{server_id}'.", file=sys.stderr)
            return 1

        module_name, _func = entry_point.rsplit(":", 1)
        cmd = [sys.executable, "-m", module_name]
        print(f"✓ Authority approved. Launching: {' '.join(cmd)}")
        result = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)})
        return result.returncode


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_decision(decision: AuthorityDecision) -> None:
    status = "✓ APPROVED" if decision.approved else "✗ REJECTED"
    print(f"\n{status} — {decision.server_id}")
    for r in decision.gate_results:
        icon = "  ✓" if r.passed else "  ✗"
        print(f"{icon} [{r.gate.upper()} GATE] {r.reason}")
    if not decision.approved:
        print(f"\n  Rejection reason: {decision.rejection_reason}")


def _cmd_discover(args: argparse.Namespace, authority: BuildAuthority) -> int:
    print("Running discovery pass…")
    profiles = authority.run_discovery(verbose=True)
    print(f"\nDiscovered {len(profiles)} server(s).")
    return 0


def _cmd_gate(args: argparse.Namespace, authority: BuildAuthority) -> int:
    server_id: str = args.server
    if server_id == "all":
        decisions = authority.evaluate_all()
        for d in decisions.values():
            _print_decision(d)
        rejected = [d for d in decisions.values() if not d.approved]
        print(f"\n{len(decisions) - len(rejected)}/{len(decisions)} servers approved.")
        return 1 if rejected else 0

    decision = authority.evaluate(server_id)
    _print_decision(decision)
    return 0 if decision.approved else 1


def _cmd_run(args: argparse.Namespace, authority: BuildAuthority) -> int:
    return authority.launch_server(args.server)


def _cmd_report(args: argparse.Namespace, authority: BuildAuthority) -> int:
    records = authority.load_log(last_n=getattr(args, "last", 20))
    if not records:
        print("No governance records found in authority log.")
        return 0
    print(f"Last {len(records)} governance record(s):\n")
    for rec in records:
        status = "✓" if rec.get("approved") else "✗"
        ts = rec.get("timestamp", "")[:19].replace("T", " ")
        reason = rec.get("rejection_reason") or "approved"
        print(f"  {status} {ts}  {rec.get('server_id', '?'):<20}  {reason}")
    return 0


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="build_authority",
        description="Overlay Authority — Build Control for AI/ML MCP Servers",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_discover = sub.add_parser("discover", help="Discover and profile all servers")
    p_discover.add_argument("--verbose", "-v", action="store_true")

    p_gate = sub.add_parser("gate", help="Run authority gate-check for a server")
    p_gate.add_argument("--server", "-s", required=True, help="Server id or 'all'")

    p_run = sub.add_parser("run", help="Gate-check then launch a server")
    p_run.add_argument("--server", "-s", required=True)

    p_report = sub.add_parser("report", help="Print recent governance records")
    p_report.add_argument("--last", "-n", type=int, default=20)

    args = parser.parse_args(argv)
    authority = BuildAuthority()

    dispatch = {
        "discover": _cmd_discover,
        "gate": _cmd_gate,
        "run": _cmd_run,
        "report": _cmd_report,
    }
    sys.exit(dispatch[args.command](args, authority))


if __name__ == "__main__":
    main()
