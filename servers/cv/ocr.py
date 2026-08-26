"""
OCR & Document Understanding MCP Server
Image / PDF OCR, table extraction helpers, scientific document conversion.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional

from servers.common import create_server, get_device, safe_json

logger = logging.getLogger(__name__)
mcp = create_server(
    "cv-ocr",
    "OCR and document tools: extract text from images or PDFs, "
    "scientific paper OCR (Nougat-style), and simple table extraction.",
)

_ocr_pipe = None


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


def _get_ocr():
    global _ocr_pipe
    if _ocr_pipe is None:
        from transformers import pipeline
        device = 0 if get_device() == "cuda" else -1
        # TrOCR is lightweight and high quality for printed text
        _ocr_pipe = pipeline("image-to-text", model="microsoft/trocr-base-printed", device=device)
        logger.info("Loaded TrOCR pipeline")
    return _ocr_pipe


@mcp.tool()
def ocr_image(image: str, model: Optional[str] = None) -> str:
    """Extract text from an image (local path, URL, or base64)."""
    img = _load_image(image)
    if model:
        from transformers import pipeline
        device = 0 if get_device() == "cuda" else -1
        pipe = pipeline("image-to-text", model=model, device=device)
        result = pipe(img)
    else:
        result = _get_ocr()(img)
    return safe_json(result)


@mcp.tool()
def ocr_pdf_page(pdf_path: str, page: int = 0, dpi: int = 200) -> str:
    """Render a PDF page to image and run OCR. Requires pdf2image or pypdfium2."""
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(pdf_path)
        page_obj = doc[page]
        bitmap = page_obj.render(scale=dpi / 72)
        pil = bitmap.to_pil()
        result = _get_ocr()(pil)
        return safe_json({"page": page, "text": result})
    except ImportError:
        return safe_json({
            "error": "pypdfium2 not installed. pip install pypdfium2",
            "hint": "Alternatively convert PDF pages to images externally and call ocr_image",
        })


@mcp.tool()
def extract_text_blocks(image: str) -> str:
    """Heuristic block-level text extraction (OCR + simple layout split)."""
    img = _load_image(image)
    # Full-page OCR first
    result = _get_ocr()(img)
    text = result[0]["generated_text"] if isinstance(result, list) else str(result)
    # Simple paragraph split
    blocks = [b.strip() for b in text.split("\n") if b.strip()]
    return safe_json({"blocks": blocks, "raw": text})


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
