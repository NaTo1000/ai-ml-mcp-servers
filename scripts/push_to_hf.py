#!/usr/bin/env python3
"""
Push the AI/ML MCP Servers suite to Hugging Face under NaTo1000.
Creates a Space (or Dataset/Model repo) and uploads the full source tree.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ID = os.getenv("HF_REPO_ID", "NaTo1000/ai-ml-mcp-servers")
REPO_TYPE = os.getenv("HF_REPO_TYPE", "space")  # space | model | dataset


def main():
    try:
        from huggingface_hub import HfApi, create_repo, login
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Set HF_TOKEN environment variable (write-scoped token)")
        sys.exit(1)

    api = HfApi(token=token)
    print(f"Authenticated as: {api.whoami()['name']}")

    try:
        create_repo(REPO_ID, repo_type=REPO_TYPE, exist_ok=True, token=token, private=False)
        print(f"Repo ready: https://huggingface.co/{REPO_TYPE}s/{REPO_ID}")
    except Exception as e:
        print(f"create_repo note: {e}")

    root = Path(__file__).resolve().parent.parent
    ignore = {
        ".git", "__pycache__", ".venv", "venv", "chroma_db",
        "*.pyc", ".DS_Store", "*.faiss", "*.meta.pkl",
    }

    print(f"Uploading from {root} → {REPO_ID} …")
    api.upload_folder(
        folder_path=str(root),
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        token=token,
        ignore_patterns=list(ignore) + ["*.pyc", "__pycache__/*", ".git/*"],
    )
    print("Done.")
    print(f"View: https://huggingface.co/{REPO_TYPE}s/{REPO_ID}")


if __name__ == "__main__":
    main()
