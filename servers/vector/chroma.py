"""
Chroma Vector Store MCP Server
Tools for RAG, persistent collections, semantic search, and document management.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from servers.common import create_server, safe_json

logger = logging.getLogger(__name__)
mcp = create_server(
    "vector-chroma",
    "Persistent Chroma vector database for RAG pipelines. "
    "Create collections, add documents with metadata, and perform semantic queries.",
)

_client = None
_collections: Dict[str, Any] = {}


def _get_client():
    global _client
    if _client is None:
        import chromadb
        from chromadb.config import Settings
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        _client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        logger.info("Chroma client initialized at %s", persist_dir)
    return _client


@mcp.tool()
def create_collection(name: str, metadata: Optional[Dict[str, str]] = None) -> str:
    """Create a new Chroma collection (or get existing)."""
    client = _get_client()
    col = client.get_or_create_collection(name=name, metadata=metadata or {})
    _collections[name] = col
    return safe_json({"status": "ok", "name": name, "count": col.count()})


@mcp.tool()
def add_documents(
    collection: str,
    documents: List[str],
    ids: Optional[List[str]] = None,
    metadatas: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Add documents (and optional metadata) to a collection. Auto-generates IDs if not provided."""
    client = _get_client()
    col = client.get_or_create_collection(collection)
    if ids is None:
        import uuid
        ids = [str(uuid.uuid4()) for _ in documents]
    col.add(documents=documents, ids=ids, metadatas=metadatas)
    return safe_json({"status": "added", "count": len(documents), "collection": collection, "total": col.count()})


@mcp.tool()
def query_collection(
    collection: str,
    query_texts: List[str],
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None,
) -> str:
    """Semantic query against a collection. Returns top-k documents with distances and metadata."""
    client = _get_client()
    col = client.get_collection(collection)
    results = col.query(query_texts=query_texts, n_results=n_results, where=where)
    return safe_json(results)


@mcp.tool()
def list_collections() -> str:
    """List all existing collections and their document counts."""
    client = _get_client()
    cols = client.list_collections()
    info = [{"name": c.name, "count": c.count(), "metadata": c.metadata} for c in cols]
    return safe_json({"collections": info})


@mcp.tool()
def delete_collection(name: str) -> str:
    """Delete an entire collection."""
    client = _get_client()
    client.delete_collection(name)
    _collections.pop(name, None)
    return safe_json({"status": "deleted", "name": name})


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
