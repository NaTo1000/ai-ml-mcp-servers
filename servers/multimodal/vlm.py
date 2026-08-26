"""
Vision-Language Model (VLM) MCP Server
Image QA, scene description, multimodal chat via BLIP / LLaVA-style models.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional

from servers.common import create_server, get_device, safe_json

logger = logging.getLogger(__name__)
mcp = create_server(
    "multimodal-vlm",
    "Vision-Language tools: ask questions about images, describe scenes, "
    "and run multimodal prompts with Hugging Face VLMs.",
)

_models: Dict[str, Any] = {}


def _load_image(image_source: str):
    from PIL import Image
    import httpx

    if image_source.startswith("http://") or image_source.startswith("https://"):
        resp = httpx.get(image_source, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    if image_source.startswith("data:image"):
        b64 = image_source.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    try:
        return Image.open(io.BytesIO(base64.b64decode(image_source))).convert("RGB")
    except Exception:
        return Image.open(image_source).convert("RGB")


def _get_vlm(model_id: str = "Salesforce/blip-vqa-base"):
    if model_id not in _models:
        from transformers import BlipProcessor, BlipForQuestionAnswering
        import torch

        device = get_device()
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForQuestionAnswering.from_pretrained(model_id)
        model = model.to(device)
        _models[model_id] = (processor, model, device)
        logger.info("Loaded VLM %s on %s", model_id, device)
    return _models[model_id]


@mcp.tool()
def image_qa(
    image: str,
    question: str,
    model: str = "Salesforce/blip-vqa-base",
) -> str:
    """Answer a natural-language question about an image."""
    img = _load_image(image)
    processor, model_obj, device = _get_vlm(model)
    inputs = processor(img, question, return_tensors="pt").to(device)
    out = model_obj.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return safe_json({"question": question, "answer": answer, "model": model})


@mcp.tool()
def describe_scene(
    image: str,
    model: str = "Salesforce/blip-image-captioning-base",
) -> str:
    """Generate a detailed scene description / caption."""
    from transformers import pipeline
    device = 0 if get_device() == "cuda" else -1
    pipe = pipeline("image-to-text", model=model, device=device)
    img = _load_image(image)
    result = pipe(img)
    return safe_json(result)


@mcp.tool()
def vlm_query(
    image: str,
    prompt: str,
    model: str = "Salesforce/blip-vqa-base",
) -> str:
    """Generic multimodal query: image + free-form prompt."""
    return image_qa(image=image, question=prompt, model=model)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
