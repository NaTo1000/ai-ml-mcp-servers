"""
Code Intelligence MCP Server
Embed code, semantic search over snippets, language detection, import extraction.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from servers.common import create_server, get_device, safe_json

logger = logging.getLogger(__name__)
mcp = create_server(
    "code-intel",
    "Code intelligence: embed snippets, semantic search, detect language, extract imports.",
)

_model_cache: Dict[str, Any] = {}


def _get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[model_name] = SentenceTransformer(model_name, device=get_device())
    return _model_cache[model_name]


@mcp.tool()
def embed_code(code: str, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> str:
    """Embed a code snippet into a dense vector."""
    model_obj = _get_embedder(model)
    emb = model_obj.encode(code, normalize_embeddings=True)
    return safe_json({"embedding": emb.tolist(), "dim": len(emb), "model": model})


@mcp.tool()
def semantic_code_search(
    query: str,
    snippets: List[str],
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5,
) -> str:
    """Rank code snippets by semantic similarity to a natural-language or code query."""
    import numpy as np
    model_obj = _get_embedder(model)
    q = model_obj.encode(query, normalize_embeddings=True)
    docs = model_obj.encode(snippets, normalize_embeddings=True)
    scores = np.dot(docs, q)
    idx = np.argsort(scores)[::-1][:top_k]
    results = [
        {"rank": i + 1, "index": int(j), "score": float(scores[j]), "snippet": snippets[j][:500]}
        for i, j in enumerate(idx)
    ]
    return safe_json({"results": results})


@mcp.tool()
def detect_language(code: str) -> str:
    """Heuristic language detection from common patterns."""
    patterns = {
        "python": [r"\bdef\s+\w+\s*\(", r"\bimport\s+\w+", r"\bfrom\s+\w+\s+import", r":\s*$"],
        "javascript": [r"\bfunction\s+\w+", r"\bconst\s+\w+\s*=", r"=>", r"require\("],
        "typescript": [r"\binterface\s+\w+", r":\s*\w+\s*[=;]", r"\btype\s+\w+\s*="],
        "java": [r"\bpublic\s+class\b", r"\bSystem\.out\.println", r"\bvoid\s+\w+\s*\("],
        "go": [r"\bfunc\s+\w+", r"\bpackage\s+\w+", r":="],
        "rust": [r"\bfn\s+\w+", r"\blet\s+mut\b", r"\bimpl\s+"],
        "cpp": [r"#include\s*<", r"\bstd::", r"\bint\s+main\s*\("],
        "sql": [r"\bSELECT\b", r"\bFROM\b", r"\bWHERE\b"],
    }
    scores = {}
    for lang, pats in patterns.items():
        scores[lang] = sum(1 for p in pats if re.search(p, code, re.I | re.M))
    best = max(scores, key=scores.get) if any(scores.values()) else "unknown"
    return safe_json({"language": best, "scores": scores})


@mcp.tool()
def extract_imports(code: str, language: str = "python") -> str:
    """Extract import / require statements."""
    if language == "python":
        imports = re.findall(r"^(?:from\s+[\w.]+\s+)?import\s+.+$", code, re.M)
    elif language in ("javascript", "typescript"):
        imports = re.findall(r"^(?:import\s+.+from\s+['\"].+['\"]|const\s+\w+\s*=\s*require\(.+\))", code, re.M)
    else:
        imports = []
    return safe_json({"imports": imports, "count": len(imports)})


@mcp.tool()
def summarize_function(code: str) -> str:
    """Produce a one-line summary of a function/method body (heuristic + docstring)."""
    doc = re.search(r'"""(.*?)"""', code, re.S) or re.search(r"'''(.*?)'''", code, re.S)
    docstring = doc.group(1).strip().split("\n")[0] if doc else None
    name = re.search(r"(?:def|function|fn)\s+(\w+)", code)
    fname = name.group(1) if name else "anonymous"
    return safe_json({
        "name": fname,
        "summary": docstring or f"Function/method named {fname}",
        "has_docstring": bool(docstring),
    })


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
