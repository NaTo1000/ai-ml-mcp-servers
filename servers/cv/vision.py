"""
Computer Vision MCP Server
Image classification, object detection, captioning, and basic segmentation helpers.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional

from servers.common import create_server, get_device, safe_json

logger = logging.getLogger(__name__)
mcp = create_server(
    "cv-vision",
    "Computer vision tools: classify images, detect objects, generate captions, "
    "and run zero-shot vision models via Hugging Face transformers / torchvision.",
)

_pipeline_cache: Dict[str, Any] = {}


def _load_image(image_source: str):
    """Accept local path, URL, or base64 data URL / raw base64."""
    from PIL import Image
    import httpx

    if image_source.startswith("http://") or image_source.startswith("https://"):
        resp = httpx.get(image_source, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    if image_source.startswith("data:image"):
        b64 = image_source.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    # try raw base64
    try:
        return Image.open(io.BytesIO(base64.b64decode(image_source))).convert("RGB")
    except Exception:
        return Image.open(image_source).convert("RGB")


def _get_pipe(task: str, model: Optional[str] = None):
    key = f"{task}:{model or 'default'}"
    if key not in _pipeline_cache:
        from transformers import pipeline
        device = 0 if get_device() == "cuda" else -1
        defaults = {
            "image-classification": "google/vit-base-patch16-224",
            "object-detection": "facebook/detr-resnet-50",
            "image-to-text": "Salesforce/blip-image-captioning-base",
            "zero-shot-image-classification": "openai/clip-vit-base-patch32",
        }
        model_id = model or defaults.get(task)
        _pipeline_cache[key] = pipeline(task, model=model_id, device=device)
        logger.info("Loaded vision pipeline %s -> %s", task, model_id)
    return _pipeline_cache[key]


@mcp.tool()
def classify_image(
    image: str,
    model: Optional[str] = None,
    top_k: int = 5,
) -> str:
    """Classify an image. image can be a local path, URL, or base64."""
    img = _load_image(image)
    pipe = _get_pipe("image-classification", model)
    results = pipe(img, top_k=top_k)
    return safe_json(results)


@mcp.tool()
def detect_objects(
    image: str,
    model: Optional[str] = None,
    threshold: float = 0.5,
) -> str:
    """Detect objects with bounding boxes. Returns labels, scores, and boxes."""
    img = _load_image(image)
    pipe = _get_pipe("object-detection", model)
    results = pipe(img, threshold=threshold)
    # serialize boxes cleanly
    out = []
    for r in results:
        box = r.get("box", {})
        out.append({
            "label": r.get("label"),
            "score": float(r.get("score", 0)),
            "box": {k: float(v) for k, v in box.items()},
        })
    return safe_json(out)


@mcp.tool()
def caption_image(
    image: str,
    model: Optional[str] = None,
    max_new_tokens: int = 50,
) -> str:
    """Generate a natural-language caption for an image."""
    img = _load_image(image)
    pipe = _get_pipe("image-to-text", model)
    result = pipe(img, max_new_tokens=max_new_tokens)
    return safe_json(result)


@mcp.tool()
def zero_shot_classify(
    image: str,
    candidate_labels: List[str],
    model: Optional[str] = None,
) -> str:
    """Zero-shot image classification against arbitrary text labels (CLIP-style)."""
    img = _load_image(image)
    pipe = _get_pipe("zero-shot-image-classification", model)
    result = pipe(img, candidate_labels=candidate_labels)
    return safe_json(result)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
