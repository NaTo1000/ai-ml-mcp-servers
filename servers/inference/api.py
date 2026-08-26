"""
Cloud / API Inference MCP Server
OpenAI-compatible endpoints, Hugging Face Inference API, Replicate-style calls.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from servers.common import create_server, safe_json

logger = logging.getLogger(__name__)
mcp = create_server(
    "inference-api",
    "Call remote inference APIs: OpenAI-compatible chat completions, "
    "Hugging Face Inference API, and generic HTTP model endpoints.",
)


@mcp.tool()
def openai_chat(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Chat completion via any OpenAI-compatible endpoint (OpenAI, Groq, Together, local vLLM, etc.)."""
    base = api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
    if not key:
        return safe_json({"error": "No API key. Set OPENAI_API_KEY or pass api_key."})

    url = base.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    with httpx.Client(timeout=120) as client:
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
    return safe_json(data)


@mcp.tool()
def hf_inference(
    model_id: str,
    inputs: str,
    parameters: Optional[Dict[str, Any]] = None,
    api_token: Optional[str] = None,
) -> str:
    """Call Hugging Face Inference API (serverless) for a model."""
    token = api_token or os.getenv("HF_TOKEN")
    if not token:
        return safe_json({"error": "HF_TOKEN required for Inference API"})

    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": inputs}
    if parameters:
        payload["parameters"] = parameters

    with httpx.Client(timeout=120) as client:
        r = client.post(url, json=payload, headers=headers)
        if r.status_code == 503:
            return safe_json({"error": "Model is loading", "hint": "Retry in a few seconds", "raw": r.text})
        r.raise_for_status()
        return safe_json(r.json())


@mcp.tool()
def generic_http_infer(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    method: str = "POST",
) -> str:
    """Generic HTTP call to any model endpoint. Useful for custom or self-hosted APIs."""
    with httpx.Client(timeout=120) as client:
        if method.upper() == "GET":
            r = client.get(url, params=payload, headers=headers or {})
        else:
            r = client.post(url, json=payload, headers=headers or {})
        try:
            return safe_json(r.json())
        except Exception:
            return safe_json({"status_code": r.status_code, "text": r.text[:2000]})


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
