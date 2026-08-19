"""
Hugging Face Hub MCP Server
Search models, datasets, spaces; get metadata; download; push helpers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from servers.common import create_server, safe_json

logger = logging.getLogger(__name__)
mcp = create_server(
    "hf-hub",
    "Full Hugging Face Hub tools: search models/datasets/spaces, get card info, "
    "download files, and list your own repos. Requires HF_TOKEN for private/write ops.",
)


def _api():
    from huggingface_hub import HfApi
    return HfApi()


@mcp.tool()
def search_models(
    query: str = "",
    task: Optional[str] = None,
    library: Optional[str] = None,
    limit: int = 10,
) -> str:
    """Search Hugging Face models. Optional filters: task (text-generation, image-classification...), library (transformers, diffusers...)."""
    api = _api()
    models = api.list_models(search=query, task=task, library=library, limit=limit, sort="downloads", direction=-1)
    results = [
        {
            "id": m.id,
            "downloads": m.downloads,
            "likes": m.likes,
            "tags": m.tags[:8] if m.tags else [],
            "pipeline_tag": m.pipeline_tag,
        }
        for m in models
    ]
    return safe_json({"models": results, "count": len(results)})


@mcp.tool()
def search_datasets(query: str = "", limit: int = 10) -> str:
    """Search Hugging Face datasets."""
    api = _api()
    datasets = api.list_datasets(search=query, limit=limit, sort="downloads", direction=-1)
    results = [
        {"id": d.id, "downloads": d.downloads, "likes": d.likes, "tags": d.tags[:6] if d.tags else []}
        for d in datasets
    ]
    return safe_json({"datasets": results, "count": len(results)})


@mcp.tool()
def get_model_info(model_id: str) -> str:
    """Get detailed model card / metadata for a specific model."""
    api = _api()
    info = api.model_info(model_id)
    return safe_json({
        "id": info.id,
        "downloads": info.downloads,
        "likes": info.likes,
        "pipeline_tag": info.pipeline_tag,
        "tags": info.tags,
        "siblings": [s.rfilename for s in (info.siblings or [])][:20],
        "card_data": str(info.cardData)[:500] if info.cardData else None,
    })


@mcp.tool()
def list_spaces(query: str = "", limit: int = 10) -> str:
    """Search or list Hugging Face Spaces."""
    api = _api()
    spaces = api.list_spaces(search=query, limit=limit)
    results = [{"id": s.id, "likes": s.likes, "sdk": s.sdk} for s in spaces]
    return safe_json({"spaces": results})


@mcp.tool()
def whoami() -> str:
    """Return the currently authenticated Hugging Face user (requires HF_TOKEN)."""
    api = _api()
    try:
        user = api.whoami()
        return safe_json(user)
    except Exception as e:
        return safe_json({"error": str(e), "hint": "Set HF_TOKEN environment variable"})


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
