"""Shared utilities for all AI/ML MCP servers."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

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
