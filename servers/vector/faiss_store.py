"""
FAISS Vector Store MCP Server
In-memory / on-disk FAISS index for high-performance similarity search.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

from servers.common import create_server, safe_json

logger = logging.getLogger(__name__)
mcp = create_server(
    "vector-faiss",
    "FAISS-based vector index for fast approximate nearest-neighbor search. "
    "Create indexes, add vectors + metadata, query, and persist to disk.",
)

_indexes: Dict[str, Any] = {}
_metadatas: Dict[str, List[Dict]] = {}
_documents: Dict[str, List[str]] = {}


@mcp.tool()
def create_index(
    name: str,
    dim: int,
    index_type: str = "Flat",
    metric: str = "ip",
) -> str:
    """Create a new FAISS index. index_type: Flat | IVFFlat | HNSW. metric: ip | l2."""
    import faiss

    if metric == "ip":
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    if index_type == "HNSW":
        index = faiss.IndexHNSWFlat(dim, 32)
    elif index_type == "IVFFlat":
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, min(100, max(1, dim // 4)))

    _indexes[name] = index
    _metadatas[name] = []
    _documents[name] = []
    return safe_json({"status": "created", "name": name, "dim": dim, "type": index_type})


@mcp.tool()
def add_vectors(
    name: str,
    vectors: List[List[float]],
    documents: Optional[List[str]] = None,
    metadatas: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Add vectors (and optional documents/metadata) to an existing index."""
    if name not in _indexes:
        return safe_json({"error": f"Index '{name}' not found. Call create_index first."})
    import faiss

    index = _indexes[name]
    arr = np.asarray(vectors, dtype=np.float32)
    if not index.is_trained:
        index.train(arr)
    index.add(arr)
    n = len(vectors)
    _documents[name].extend(documents or [""] * n)
    _metadatas[name].extend(metadatas or [{}] * n)
    return safe_json({"status": "added", "count": n, "total": index.ntotal})


@mcp.tool()
def search(
    name: str,
    query_vectors: List[List[float]],
    top_k: int = 5,
) -> str:
    """Search the index. Returns distances, indices, documents and metadata."""
    if name not in _indexes:
        return safe_json({"error": f"Index '{name}' not found"})
    index = _indexes[name]
    q = np.asarray(query_vectors, dtype=np.float32)
    distances, indices = index.search(q, top_k)
    results = []
    for qi, (dists, idxs) in enumerate(zip(distances, indices)):
        hits = []
        for d, i in zip(dists, idxs):
            if i < 0:
                continue
            hits.append({
                "index": int(i),
                "distance": float(d),
                "document": _documents[name][i] if i < len(_documents[name]) else None,
                "metadata": _metadatas[name][i] if i < len(_metadatas[name]) else None,
            })
        results.append({"query_index": qi, "hits": hits})
    return safe_json({"results": results})


@mcp.tool()
def save_index(name: str, path: str) -> str:
    """Persist index + metadata to disk."""
    if name not in _indexes:
        return safe_json({"error": f"Index '{name}' not found"})
    import faiss

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    faiss.write_index(_indexes[name], path + ".faiss")
    with open(path + ".meta.pkl", "wb") as f:
        pickle.dump({"documents": _documents[name], "metadatas": _metadatas[name]}, f)
    return safe_json({"status": "saved", "path": path})


@mcp.tool()
def load_index(name: str, path: str) -> str:
    """Load a previously saved index."""
    import faiss

    index = faiss.read_index(path + ".faiss")
    with open(path + ".meta.pkl", "rb") as f:
        meta = pickle.load(f)
    _indexes[name] = index
    _documents[name] = meta["documents"]
    _metadatas[name] = meta["metadatas"]
    return safe_json({"status": "loaded", "name": name, "ntotal": index.ntotal})


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
