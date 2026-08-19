"""
Local Model Inference MCP Server
Load Hugging Face / local transformers models and run generation / chat.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from servers.common import create_server, get_device, safe_json

logger = logging.getLogger(__name__)
mcp = create_server(
    "inference-local",
    "Load and run local Hugging Face transformers models for text generation and chat. "
    "Supports causal LMs and instruction-tuned models.",
)

_models: Dict[str, Any] = {}
_tokenizers: Dict[str, Any] = {}


def _load(model_id: str, device: Optional[str] = None):
    if model_id in _models:
        return _models[model_id], _tokenizers[model_id]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    device = device or get_device()
    logger.info("Loading %s on %s ...", model_id, device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)
    _models[model_id] = model
    _tokenizers[model_id] = tokenizer
    return model, tokenizer


@mcp.tool()
def load_model(model_id: str = "gpt2", device: Optional[str] = None) -> str:
    """Load a Hugging Face causal LM into memory. Returns basic info."""
    model, tokenizer = _load(model_id, device)
    return safe_json({
        "status": "loaded",
        "model_id": model_id,
        "device": str(next(model.parameters()).device),
        "vocab_size": tokenizer.vocab_size,
    })


@mcp.tool()
def generate(
    prompt: str,
    model_id: str = "gpt2",
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """Generate text from a prompt using a loaded (or auto-loaded) model."""
    model, tokenizer = _load(model_id)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return safe_json({"generated": text, "model_id": model_id})


@mcp.tool()
def chat(
    messages: List[Dict[str, str]],
    model_id: str = "gpt2",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Chat-style generation. messages = [{\"role\": \"user\", \"content\": \"...\"}, ...]. """
    model, tokenizer = _load(model_id)
    # Simple concatenation fallback if no chat template
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return safe_json({"response": text.strip(), "model_id": model_id})


@mcp.tool()
def list_loaded_models() -> str:
    """List currently loaded models and their devices."""
    info = []
    for mid, model in _models.items():
        device = str(next(model.parameters()).device)
        info.append({"model_id": mid, "device": device})
    return safe_json({"loaded": info})


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
