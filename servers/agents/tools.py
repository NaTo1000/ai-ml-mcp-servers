"""
Agent Utilities MCP Server
Simple memory, planning helpers, and tool-routing primitives for multi-agent builds.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from servers.common import create_server, safe_json

logger = logging.getLogger(__name__)
mcp = create_server(
    "agent-tools",
    "Lightweight agent primitives: persistent key-value memory, simple planning notes, "
    "and hand-off helpers for multi-agent systems.",
)

_memory: Dict[str, Any] = {}
_notes: List[Dict[str, Any]] = []


@mcp.tool()
def memory_write(key: str, value: str, tags: Optional[List[str]] = None) -> str:
    """Store a value in agent memory under a key."""
    _memory[key] = {"value": value, "tags": tags or [], "ts": time.time()}
    return safe_json({"status": "stored", "key": key})


@mcp.tool()
def memory_read(key: str) -> str:
    """Retrieve a value from agent memory."""
    if key not in _memory:
        return safe_json({"error": "key not found", "key": key})
    return safe_json(_memory[key])


@mcp.tool()
def memory_list(tag: Optional[str] = None) -> str:
    """List all memory keys (optionally filtered by tag)."""
    items = []
    for k, v in _memory.items():
        if tag is None or tag in v.get("tags", []):
            items.append({"key": k, "tags": v.get("tags"), "ts": v.get("ts")})
    return safe_json({"items": items})


@mcp.tool()
def add_note(content: str, role: str = "planner") -> str:
    """Add a planning or observation note."""
    note = {"role": role, "content": content, "ts": time.time()}
    _notes.append(note)
    return safe_json({"status": "added", "index": len(_notes) - 1})


@mcp.tool()
def list_notes(limit: int = 20) -> str:
    """Return the most recent notes."""
    return safe_json({"notes": _notes[-limit:]})


@mcp.tool()
def clear_memory() -> str:
    """Clear all in-memory state (for testing)."""
    _memory.clear()
    _notes.clear()
    return safe_json({"status": "cleared"})


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
