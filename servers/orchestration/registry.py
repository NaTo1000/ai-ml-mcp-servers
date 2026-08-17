"""
Orchestration Registry — auto-discovery and ServerProfile builder.

Discovery strategy
------------------
1. Read ``pyproject.toml`` in the repository root to find every declared
   ``[project.scripts]`` entry whose value starts with ``servers.``.
2. For each entry-point, derive the domain from the sub-package name and build
   an initial ``ServerProfile``.
3. Attempt to import the target module and look for a ``SERVER_METADATA``
   attribute of type ``servers.common.ServerMetadata`` to enrich the profile
   with structured tool information.
4. Fall back gracefully: if the module cannot be imported (e.g. optional
   heavy dependencies are absent) the profile is still created from the
   entry-point metadata alone.
5. Persist every discovered profile to the ``DatabaseManager`` so the rater
   and allocator can operate on up-to-date data.

All public symbols are safe to call from the orchestration MCP server at
startup and on-demand via the ``rerate_servers`` tool.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Path to the repository root (servers/orchestration/registry.py → repo root).
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Mapping from sub-package name → domain label used in the database.
_DOMAIN_MAP: Dict[str, str] = {
    "nlp": "nlp",
    "cv": "cv",
    "audio": "audio",
    "training": "training",
    "inference": "inference",
    "vector": "vector",
    "agents": "agents",
    "data": "data",
    "utils": "utils",
    "multimodal": "multimodal",
    "orchestration": "orchestration",
}

# Extras key associated with each domain (mirrors pyproject.toml optionals).
_EXTRAS_MAP: Dict[str, Optional[str]] = {
    "cv": "vision",
    "audio": "audio",
    "vector": "vector-extra",
    "training": "train",
    "multimodal": "vision",
}

# GPU-heavy domains that default to requires_gpu=True unless overridden.
_GPU_DOMAINS = {"training", "inference", "cv", "audio", "multimodal"}

# Minimum RAM estimate (GB) per domain — coarse defaults used when no
# ServerMetadata is available.
_RAM_DEFAULTS: Dict[str, float] = {
    "nlp": 2.0,
    "cv": 4.0,
    "audio": 4.0,
    "training": 8.0,
    "inference": 4.0,
    "vector": 1.0,
    "agents": 1.0,
    "data": 1.0,
    "utils": 0.5,
    "multimodal": 6.0,
    "orchestration": 0.5,
}


@dataclass
class ServerProfile:
    """
    In-memory representation of a discovered MCP server's capabilities.

    This is the canonical object passed between the registry, rater, and
    allocator.  It is persisted to ``server_profiles`` via
    ``DatabaseManager.upsert_server_profile``.
    """

    server_id: str
    domain: str
    description: str
    entry_point: str
    # Serialisable tool list (dicts, not ToolCapability objects, for easy JSON)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    requires_gpu: bool = False
    min_ram_gb: float = 0.0
    extras_key: Optional[str] = None
    annotations: Dict[str, Any] = field(default_factory=dict)
    # Set by the rater; None until first rating pass
    score: Optional[float] = None
    score_breakdown: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "server_id": self.server_id,
            "domain": self.domain,
            "description": self.description,
            "entry_point": self.entry_point,
            "tools": self.tools,
            "requires_gpu": self.requires_gpu,
            "min_ram_gb": self.min_ram_gb,
            "extras_key": self.extras_key,
            "annotations": self.annotations,
            "score": self.score,
            "score_breakdown": self.score_breakdown,
        }


class Registry:
    """
    Discovers, builds, and persists ``ServerProfile`` objects for every MCP
    server in the repository.

    Parameters
    ----------
    db:
        A ``DatabaseManager`` instance used to persist discovered profiles.
    pyproject_path:
        Override the default ``pyproject.toml`` path (useful in tests).
    """

    def __init__(
        self,
        db: Any,  # DatabaseManager — imported lazily to avoid circular imports
        pyproject_path: Optional[Path] = None,
    ) -> None:
        self._db = db
        self._pyproject_path = pyproject_path or (_REPO_ROOT / "pyproject.toml")
        self._profiles: Dict[str, ServerProfile] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(self) -> List[ServerProfile]:
        """
        Run a full discovery pass and return the list of profiles found.

        Profiles are also upserted into the database so they survive process
        restarts.
        """
        entry_points = self._read_entry_points()
        discovered: List[ServerProfile] = []
        for script_name, module_path in entry_points.items():
            try:
                profile = self._build_profile(script_name, module_path)
                self._profiles[profile.server_id] = profile
                self._persist(profile)
                discovered.append(profile)
                logger.info("Discovered server: %s (domain=%s)", profile.server_id, profile.domain)
            except Exception as exc:
                logger.warning("Failed to build profile for %s: %s", script_name, exc)
        return discovered

    def get_profile(self, server_id: str) -> Optional[ServerProfile]:
        """Return the in-memory profile for a known server id."""
        return self._profiles.get(server_id)

    def all_profiles(self) -> List[ServerProfile]:
        """Return all currently-held in-memory profiles."""
        return list(self._profiles.values())

    def refresh(self, server_id: str) -> Optional[ServerProfile]:
        """Re-discover a single server and update its profile in-place."""
        entry_points = self._read_entry_points()
        for script_name, module_path in entry_points.items():
            derived_id = _script_name_to_server_id(script_name)
            if derived_id == server_id:
                profile = self._build_profile(script_name, module_path)
                self._profiles[profile.server_id] = profile
                self._persist(profile)
                return profile
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_entry_points(self) -> Dict[str, str]:
        """
        Parse ``[project.scripts]`` from pyproject.toml without requiring the
        ``tomllib`` standard library (Python ≥ 3.11) — uses a simple regex
        parser that covers the subset of TOML we need.
        """
        path = self._pyproject_path
        if not path.exists():
            logger.warning("pyproject.toml not found at %s", path)
            return {}

        text = path.read_text(encoding="utf-8")

        # Try stdlib tomllib (3.11+) first, then tomli, then fallback parser.
        try:
            if sys.version_info >= (3, 11):
                import tomllib  # type: ignore[import]
                data = tomllib.loads(text)
            else:
                import tomli  # type: ignore[import]
                data = tomli.loads(text)
            scripts: Dict[str, str] = data.get("project", {}).get("scripts", {})
        except Exception:
            scripts = _parse_scripts_fallback(text)

        # Keep only entries that map to ``servers.*`` modules.
        return {k: v for k, v in scripts.items() if v.startswith("servers.")}

    def _build_profile(self, script_name: str, module_path: str) -> ServerProfile:
        """
        Construct a ``ServerProfile`` from entry-point metadata, optionally
        enriching it with ``SERVER_METADATA`` from the target module.
        """
        server_id = _script_name_to_server_id(script_name)
        module_name, _func = module_path.rsplit(":", 1)
        domain = _infer_domain(module_name)

        profile = ServerProfile(
            server_id=server_id,
            domain=domain,
            description=f"MCP server: {server_id}",
            entry_point=module_path,
            requires_gpu=domain in _GPU_DOMAINS,
            min_ram_gb=_RAM_DEFAULTS.get(domain, 1.0),
            extras_key=_EXTRAS_MAP.get(domain),
        )

        # Attempt rich introspection via SERVER_METADATA.
        metadata = _try_load_server_metadata(module_name)
        if metadata is not None:
            profile.description = metadata.description or profile.description
            profile.tools = [t.to_dict() for t in metadata.tools]
            profile.requires_gpu = metadata.requires_gpu
            profile.min_ram_gb = metadata.min_ram_gb or profile.min_ram_gb
            profile.extras_key = metadata.extras_key or profile.extras_key
            profile.annotations = metadata.annotations
        else:
            # Best-effort: try to extract tool names from the FastMCP instance.
            profile.tools = _infer_tools_from_module(module_name)

        return profile

    def _persist(self, profile: ServerProfile) -> None:
        self._db.upsert_server_profile(
            server_id=profile.server_id,
            domain=profile.domain,
            description=profile.description,
            entry_point=profile.entry_point,
            tools=profile.tools,
            requires_gpu=profile.requires_gpu,
            min_ram_gb=profile.min_ram_gb,
            extras_key=profile.extras_key,
            annotations=profile.annotations,
        )


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions — no side effects beyond logging)
# ---------------------------------------------------------------------------

def _script_name_to_server_id(script_name: str) -> str:
    """
    Normalise a pyproject script name to a server id.

    ``mcp-nlp-text`` → ``nlp-text``
    ``mcp-orchestrator`` → ``orchestrator``
    """
    if script_name.startswith("mcp-"):
        return script_name[4:]
    return script_name


def _infer_domain(module_name: str) -> str:
    """
    Extract the domain from a dotted module path.

    ``servers.nlp.text_analysis`` → ``nlp``
    ``servers.orchestration.server`` → ``orchestration``
    """
    parts = module_name.split(".")
    if len(parts) >= 2 and parts[0] == "servers":
        return _DOMAIN_MAP.get(parts[1], parts[1])
    return "utils"


def _try_load_server_metadata(module_name: str) -> Optional[Any]:
    """
    Import ``module_name`` and return its ``SERVER_METADATA`` attribute, or
    ``None`` if it does not exist or if the import fails (e.g. optional heavy
    dependency absent).
    """
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, "SERVER_METADATA", None)
    except Exception as exc:
        logger.debug("Could not import %s for metadata introspection: %s", module_name, exc)
        return None


def _infer_tools_from_module(module_name: str) -> List[Dict[str, Any]]:
    """
    Attempt to read tool names from the FastMCP instance in a module.

    Returns a list of minimal tool dicts if successful, otherwise an empty
    list.  This is a best-effort fallback used when ``SERVER_METADATA`` is not
    defined.
    """
    try:
        mod = importlib.import_module(module_name)
        # FastMCP instances are typically stored as a module-level ``mcp`` var.
        mcp_instance = getattr(mod, "mcp", None)
        if mcp_instance is None:
            return []
        # FastMCP exposes its tool registry; the attribute name differs across
        # versions — try the two known locations.
        tool_registry = (
            getattr(mcp_instance, "_tools", None)
            or getattr(mcp_instance, "tools", None)
            or {}
        )
        if hasattr(tool_registry, "items"):
            tools = []
            for name, tool_obj in tool_registry.items():
                desc = ""
                fn = getattr(tool_obj, "fn", None) or getattr(tool_obj, "func", None)
                if fn and fn.__doc__:
                    # Use only the first line of the docstring.
                    desc = fn.__doc__.strip().splitlines()[0]
                tools.append({"name": name, "description": desc, "parameters": [], "tags": []})
            return tools
    except Exception as exc:
        logger.debug("Could not infer tools from %s: %s", module_name, exc)
    return []


def _parse_scripts_fallback(toml_text: str) -> Dict[str, str]:
    """
    Minimal ``[project.scripts]`` parser that handles the subset of TOML
    produced by ``hatchling``/``flit`` — used when neither ``tomllib`` nor
    ``tomli`` is available.

    Parses lines of the form::

        script-name = "package.module:function"

    between the ``[project.scripts]`` and the next ``[`` section header.
    """
    import re

    scripts: Dict[str, str] = {}
    in_scripts = False
    for line in toml_text.splitlines():
        stripped = line.strip()
        if stripped == "[project.scripts]":
            in_scripts = True
            continue
        if in_scripts:
            if stripped.startswith("["):
                break
            m = re.match(r'^([\w\-]+)\s*=\s*"([^"]+)"', stripped)
            if m:
                scripts[m.group(1)] = m.group(2)
    return scripts
