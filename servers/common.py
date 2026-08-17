"""Shared utilities for all AI/ML MCP servers."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("ai-ml-mcp")


def create_server(name: str, instructions: str = "") -> FastMCP:
    """Create a FastMCP server instance with consistent defaults."""
    return FastMCP(
        name=name,
        instructions=instructions
        or f"Production AI/ML MCP server: {name}. Use the available tools to perform specialized ML tasks.",
    )


def safe_json(obj: Any) -> str:
    """Serialize to JSON, handling non-serializable types."""
    def default(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        if hasattr(o, "__dict__"):
            return str(o)
        return str(o)
    return json.dumps(obj, default=default, indent=2)


def get_device() -> str:
    """Return best available device string."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def require_env(key: str, default: Optional[str] = None) -> str:
    val = os.getenv(key, default)
    if val is None:
        raise ValueError(f"Environment variable {key} is required")
    return val


# ---------------------------------------------------------------------------
# Structured metadata types — optionally populated by each MCP server so the
# Orchestration registry can introspect richer capability information.
# ---------------------------------------------------------------------------

@dataclass
class ToolCapability:
    """Description of a single tool exposed by an MCP server."""

    name: str
    description: str
    # Parameter names expected by the tool (positional / keyword)
    parameters: List[str] = field(default_factory=list)
    # Broad task categories this tool covers, e.g. ["classification", "nlp"]
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCapability":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            parameters=data.get("parameters", []),
            tags=data.get("tags", []),
        )


@dataclass
class ServerMetadata:
    """Rich metadata that an MCP server can advertise to the Orchestration layer."""

    # Human-readable server id matching the pyproject.toml entry-point name
    server_id: str
    # Top-level domain: nlp | cv | audio | training | inference | vector | agents | data | utils
    domain: str
    description: str
    # Structured tool catalogue; populated by the server or inferred by the registry
    tools: List[ToolCapability] = field(default_factory=list)
    # Indicative resource hints — used by the allocator for budget checks
    requires_gpu: bool = False
    min_ram_gb: float = 0.0
    # Optional extras key from pyproject.toml (e.g. "audio", "vision")
    extras_key: Optional[str] = None
    # Free-form key/value annotations
    annotations: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "server_id": self.server_id,
            "domain": self.domain,
            "description": self.description,
            "tools": [t.to_dict() for t in self.tools],
            "requires_gpu": self.requires_gpu,
            "min_ram_gb": self.min_ram_gb,
            "extras_key": self.extras_key,
            "annotations": self.annotations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServerMetadata":
        return cls(
            server_id=data["server_id"],
            domain=data.get("domain", "utils"),
            description=data.get("description", ""),
            tools=[ToolCapability.from_dict(t) for t in data.get("tools", [])],
            requires_gpu=data.get("requires_gpu", False),
            min_ram_gb=data.get("min_ram_gb", 0.0),
            extras_key=data.get("extras_key"),
            annotations=data.get("annotations", {}),
        )
