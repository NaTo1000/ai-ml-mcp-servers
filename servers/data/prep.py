"""
Dataset Preparation MCP Server
Load HF datasets, clean, split, push back to Hub.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from servers.common import create_server, safe_json

logger = logging.getLogger(__name__)
mcp = create_server(
    "data-prep",
    "Load, inspect, clean, split, and push Hugging Face datasets. Ideal for training pipelines.",
)

_cache: Dict[str, Any] = {}


def _dataset_summary(ds: Any) -> Dict[str, Any]:
    if hasattr(ds, "items"):
        splits = {}
        for split_name, split_ds in ds.items():
            splits[split_name] = {
                "num_rows": len(split_ds),
                "features": str(split_ds.features),
                "column_names": split_ds.column_names,
            }
        return {"dataset_type": "dataset_dict", "splits": splits}
    return {
        "dataset_type": "dataset",
        "num_rows": len(ds),
        "features": str(ds.features),
        "column_names": ds.column_names,
        "first_rows": [dict(r) for r in ds.select(range(min(3, len(ds))))],
    }


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

    info = _dataset_summary(ds)
    info["key"] = key
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
    if hasattr(ds, "items"):
        return safe_json({"error": "Loaded object contains multiple splits. Load a single split first."})
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
def filter_dataset(
    key: str,
    column: str,
    equals: Optional[str] = None,
    contains: Optional[str] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    new_key: Optional[str] = None,
) -> str:
    """Filter a loaded dataset by exact match, substring match, or numeric range."""
    if key not in _cache:
        return safe_json({"error": "Dataset not loaded. Call load_dataset first."})
    ds = _cache[key]
    if hasattr(ds, "items"):
        return safe_json({"error": "Filtering a dataset dict is not supported. Load a single split first."})
    if column not in ds.column_names:
        return safe_json({"error": f"Column '{column}' not found", "available_columns": ds.column_names})

    def keep(row: Dict[str, Any]) -> bool:
        value = row.get(column)
        if equals is not None and str(value) != equals:
            return False
        if contains is not None and contains not in str(value):
            return False
        if min_value is not None:
            try:
                if value is None or float(value) < min_value:
                    return False
            except (TypeError, ValueError):
                return False
        if max_value is not None:
            try:
                if value is None or float(value) > max_value:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    filtered = ds.filter(keep)
    out_key = new_key or f"{key}:filtered"
    _cache[out_key] = filtered
    return safe_json({
        "source_key": key,
        "filtered_key": out_key,
        "num_rows": len(filtered),
        "column": column,
    })


@mcp.tool()
def map_dataset(
    key: str,
    column: str,
    operation: str,
    new_column: Optional[str] = None,
    value: Optional[str] = None,
    new_key: Optional[str] = None,
) -> str:
    """Apply a safe column transform: lower, upper, strip, length, prefix, suffix, fillna, json_parse."""
    if key not in _cache:
        return safe_json({"error": "Dataset not loaded. Call load_dataset first."})
    ds = _cache[key]
    if hasattr(ds, "items"):
        return safe_json({"error": "Mapping a dataset dict is not supported. Load a single split first."})
    if column not in ds.column_names:
        return safe_json({"error": f"Column '{column}' not found", "available_columns": ds.column_names})

    target_column = new_column or column

    def transform(example: Dict[str, Any]) -> Dict[str, Any]:
        source = example.get(column)
        if operation == "lower":
            result = "" if source is None else str(source).lower()
        elif operation == "upper":
            result = "" if source is None else str(source).upper()
        elif operation == "strip":
            result = "" if source is None else str(source).strip()
        elif operation == "length":
            result = len(str(source)) if source is not None else 0
        elif operation == "prefix":
            result = f"{value or ''}{'' if source is None else str(source)}"
        elif operation == "suffix":
            result = f"{'' if source is None else str(source)}{value or ''}"
        elif operation == "fillna":
            result = value if source in (None, "") else source
        elif operation == "json_parse":
            result = json.loads(source) if source else None
        else:
            raise ValueError(
                "Unsupported operation. Use lower, upper, strip, length, prefix, suffix, fillna, or json_parse."
            )
        return {target_column: result}

    try:
        mapped = ds.map(transform)
    except Exception as e:
        return safe_json({"error": str(e), "operation": operation, "column": column})

    out_key = new_key or f"{key}:mapped"
    _cache[out_key] = mapped
    return safe_json({
        "source_key": key,
        "mapped_key": out_key,
        "column": column,
        "target_column": target_column,
        "operation": operation,
        "num_rows": len(mapped),
    })


@mcp.tool()
def push_to_hub(
    key: str,
    repo_id: str,
    private: bool = False,
    token: Optional[str] = None,
    commit_message: str = "Upload dataset from ai-ml-mcp-servers",
) -> str:
    """Push a cached dataset split to the Hugging Face Hub."""
    if key not in _cache:
        return safe_json({"error": "Dataset not loaded. Call load_dataset first."})
    ds = _cache[key]
    if hasattr(ds, "items"):
        return safe_json({"error": "Pushing a dataset dict is not supported here. Load a single split first."})
    try:
        ds.push_to_hub(repo_id, private=private, token=token, commit_message=commit_message)
        return safe_json({"status": "pushed", "key": key, "repo_id": repo_id, "private": private})
    except Exception as e:
        return safe_json({"error": str(e), "repo_id": repo_id})


@mcp.tool()
def list_loaded() -> str:
    """List currently cached datasets."""
    return safe_json({"loaded_keys": list(_cache.keys())})


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
