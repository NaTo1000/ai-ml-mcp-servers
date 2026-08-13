"""
NLP Embeddings MCP Server
Tools: embed_text, batch_embed, cosine_similarity, semantic_search
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from servers.common import create_server, get_device, safe_json

logger = logging.getLogger(__name__)
mcp = create_server(
    "nlp-embeddings",
    "Sentence and document embedding tools powered by sentence-transformers. "
    "Default model: all-MiniLM-L6-v2 (fast & high quality).",
)

_model_cache: Dict[str, Any] = {}


def _get_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        device = get_device()
        _model_cache[model_name] = SentenceTransformer(model_name, device=device)
        logger.info("Loaded embedding model %s on %s", model_name, device)
    return _model_cache[model_name]


@mcp.tool()
def embed_text(
    text: str,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True,
) -> str:
    """Embed a single text into a dense vector. Returns the vector as a list of floats."""
    model_obj = _get_model(model)
    emb = model_obj.encode(text, normalize_embeddings=normalize)
    return safe_json({"embedding": emb.tolist(), "dim": len(emb), "model": model})


@mcp.tool()
def batch_embed(
    texts: List[str],
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True,
    batch_size: int = 32,
) -> str:
    """Embed a batch of texts. Returns list of vectors."""
    model_obj = _get_model(model)
    embs = model_obj.encode(texts, normalize_embeddings=normalize, batch_size=batch_size)
    return safe_json({
        "embeddings": [e.tolist() for e in embs],
        "count": len(embs),
        "dim": len(embs[0]) if len(embs) else 0,
        "model": model,
    })


@mcp.tool()
def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> str:
    """Compute cosine similarity between two embedding vectors."""
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return safe_json({"cosine_similarity": sim})


@mcp.tool()
def semantic_search(
    query: str,
    documents: List[str],
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5,
) -> str:
    """Semantic search: rank documents by similarity to the query."""
    model_obj = _get_model(model)
    q_emb = model_obj.encode(query, normalize_embeddings=True)
    doc_embs = model_obj.encode(documents, normalize_embeddings=True)
    scores = np.dot(doc_embs, q_emb)
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = [
        {"rank": i + 1, "index": int(idx), "score": float(scores[idx]), "document": documents[idx]}
        for i, idx in enumerate(top_idx)
    ]
    return safe_json({"results": results, "model": model})


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
