"""
Dataset Preparation MCP Server
Load HF datasets, clean, split, push back to Hub.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from servers.common import create_server, safe_json

logger = logging.getLogger(__name__)
mcp = create_server(
    "data-prep",
    "Load, inspect, clean, split, and push Hugging Face datasets. Ideal for training pipelines.",
)

_cache: Dict[str, Any] = {}


@mcp.tool()
def load_dataset(
    path: str,
    split: Optional[str] = None,
    name: Optional[str] = None,
    streaming: bool = False,
) -> str:
    """Load a Hugging Face dataset (or local path). Returns basic stats and first rows."""
    from datasets import load_dataset as hf_load

    ds = hf_load(path, name=name, split=split, streaming=streaming)
    key = f"{path}:{name}:{split}"
    _cache[key] = ds

    if streaming:
        sample = list(ds.take(3))
        return safe_json({"status": "streaming", "sample": [str(r) for r in sample], "key": key})

    info = {
        "num_rows": len(ds),
        "features": str(ds.features),
        "column_names": ds.column_names,
        "first_rows": [dict(r) for r in ds.select(range(min(3, len(ds))))],
        "key": key,
    }
    return safe_json(info)


@mcp.tool()
def split_dataset(
    key: str,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> str:
    """Split a previously loaded dataset into train/test."""
    if key not in _cache:
        return safe_json({"error": "Dataset not loaded. Call load_dataset first."})
    ds = _cache[key]
    split = ds.train_test_split(test_size=1 - train_ratio, seed=seed)
    train_key = key + ":train"
    test_key = key + ":test"
    _cache[train_key] = split["train"]
    _cache[test_key] = split["test"]
    return safe_json({
        "train_rows": len(split["train"]),
        "test_rows": len(split["test"]),
        "train_key": train_key,
        "test_key": test_key,
    })


@mcp.tool()
def list_loaded() -> str:
    """List currently cached datasets."""
    return safe_json({"loaded_keys": list(_cache.keys())})


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
