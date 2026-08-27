"""
Diffusion / Generative Image MCP Server
Helpers for Diffusers pipelines, VRAM estimates, and ready-to-run templates.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from servers.common import create_server, safe_json

logger = logging.getLogger(__name__)
mcp = create_server(
    "diffusion",
    "Helpers for Stable Diffusion / Diffusers pipelines. "
    "List popular models, estimate VRAM, and emit ready-to-run pipeline templates.",
)


POPULAR = [
    {"id": "stabilityai/stable-diffusion-xl-base-1.0", "type": "text2img", "approx_vram_gb": 8},
    {"id": "runwayml/stable-diffusion-v1-5", "type": "text2img", "approx_vram_gb": 4},
    {"id": "black-forest-labs/FLUX.1-dev", "type": "text2img", "approx_vram_gb": 16},
    {"id": "stabilityai/stable-diffusion-2-1", "type": "text2img", "approx_vram_gb": 5},
    {"id": "timbrooks/instruct-pix2pix", "type": "img2img", "approx_vram_gb": 6},
]


@mcp.tool()
def list_diffusers(limit: int = 10) -> str:
    """List popular diffusion models with approximate VRAM requirements."""
    return safe_json({"models": POPULAR[:limit]})


@mcp.tool()
def text2img_info(model_id: str = "runwayml/stable-diffusion-v1-5") -> str:
    """Return guidance and recommended settings for a text-to-image model."""
    return safe_json({
        "model_id": model_id,
        "recommended": {
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512,
            "dtype": "float16",
        },
        "notes": "Use torch.float16 + enable_model_cpu_offload() for low VRAM.",
    })


@mcp.tool()
def img2img_info(model_id: str = "timbrooks/instruct-pix2pix") -> str:
    """Return guidance for image-to-image / instruct-pix2pix style models."""
    return safe_json({
        "model_id": model_id,
        "recommended": {
            "num_inference_steps": 20,
            "image_guidance_scale": 1.5,
            "guidance_scale": 7.0,
            "strength": 0.75,
        },
    })


@mcp.tool()
def estimate_vram(model_id: str, resolution: int = 512, batch_size: int = 1) -> str:
    """Rough VRAM estimate in GB for a diffusion run."""
    base = 4.0
    for m in POPULAR:
        if m["id"] == model_id:
            base = float(m["approx_vram_gb"])
            break
    scale = (resolution / 512) ** 2 * batch_size
    return safe_json({
        "model_id": model_id,
        "resolution": resolution,
        "batch_size": batch_size,
        "estimated_vram_gb": round(base * scale, 1),
        "note": "Estimate only; actual usage depends on dtype, attention slicing, offload.",
    })


@mcp.tool()
def pipeline_template(model_id: str = "runwayml/stable-diffusion-v1-5", task: str = "text2img") -> str:
    """Return a ready-to-copy Diffusers Python snippet."""
    if task == "img2img":
        code = f'''from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "{model_id}", torch_dtype=torch.float16
).to("cuda")
pipe.enable_model_cpu_offload()

init = Image.open("input.png").convert("RGB").resize((512, 512))
image = pipe(prompt="your prompt", image=init, strength=0.75).images[0]
image.save("out.png")
'''
    else:
        code = f'''from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "{model_id}", torch_dtype=torch.float16
).to("cuda")
pipe.enable_model_cpu_offload()

image = pipe("your prompt", num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("out.png")
'''
    return safe_json({"task": task, "model_id": model_id, "code": code})


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
