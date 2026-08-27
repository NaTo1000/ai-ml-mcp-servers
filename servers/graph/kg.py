"""
Graph / Knowledge-Graph MCP Server
In-memory directed graph: nodes, edges, neighbors, shortest path, export.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from servers.common import create_server, safe_json

logger = logging.getLogger(__name__)
mcp = create_server(
    "graph",
    "Lightweight in-memory knowledge graph. Create nodes/edges, query neighbors, shortest path, export.",
)

_graphs: Dict[str, Dict[str, Any]] = {}


def _g(name: str) -> Dict[str, Any]:
    if name not in _graphs:
        _graphs[name] = {"nodes": {}, "edges": defaultdict(list)}
    return _graphs[name]


@mcp.tool()
def create_graph(name: str = "default") -> str:
    """Create (or reset) a named graph."""
    _graphs[name] = {"nodes": {}, "edges": defaultdict(list)}
    return safe_json({"status": "created", "name": name})


@mcp.tool()
def add_nodes(nodes: List[Dict[str, Any]], graph: str = "default") -> str:
    """Add nodes. Each node: {\"id\": \"...\", \"label\": \"...\", ...any attrs}."""
    g = _g(graph)
    for n in nodes:
        nid = str(n.get("id") or n.get("name"))
        g["nodes"][nid] = n
    return safe_json({"status": "ok", "node_count": len(g["nodes"])})


@mcp.tool()
def add_edges(edges: List[Dict[str, str]], graph: str = "default") -> str:
    """Add directed edges. Each edge: {\"source\": \"...\", \"target\": \"...\", \"relation\": \"...\"}."""
    g = _g(graph)
    for e in edges:
        src, tgt = str(e["source"]), str(e["target"])
        g["edges"][src].append({"target": tgt, "relation": e.get("relation", "related")})
    total = sum(len(v) for v in g["edges"].values())
    return safe_json({"status": "ok", "edge_count": total})


@mcp.tool()
def query_neighbors(node_id: str, graph: str = "default", direction: str = "out") -> str:
    """Return outgoing (or incoming) neighbors of a node."""
    g = _g(graph)
    if direction == "out":
        return safe_json({"node": node_id, "neighbors": g["edges"].get(node_id, [])})
    incoming = []
    for src, outs in g["edges"].items():
        for e in outs:
            if e["target"] == node_id:
                incoming.append({"source": src, "relation": e["relation"]})
    return safe_json({"node": node_id, "neighbors": incoming})


@mcp.tool()
def shortest_path(source: str, target: str, graph: str = "default") -> str:
    """BFS shortest path between two nodes."""
    g = _g(graph)
    if source not in g["nodes"] or target not in g["nodes"]:
        return safe_json({"error": "source or target missing"})
    q = deque([(source, [source])])
    seen = {source}
    while q:
        node, path = q.popleft()
        if node == target:
            return safe_json({"path": path, "length": len(path) - 1})
        for e in g["edges"].get(node, []):
            nxt = e["target"]
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, path + [nxt]))
    return safe_json({"path": None, "error": "no path"})


@mcp.tool()
def export_graph(graph: str = "default") -> str:
    """Export full graph as nodes + edge list."""
    g = _g(graph)
    edges = []
    for src, outs in g["edges"].items():
        for e in outs:
            edges.append({"source": src, "target": e["target"], "relation": e["relation"]})
    return safe_json({"nodes": list(g["nodes"].values()), "edges": edges})


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
