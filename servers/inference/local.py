"""
Local Model Inference MCP Server
Load Hugging Face / local transformers models and run generation / chat.
"""

from __future__ import annotations

import gc
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
        "parameters": int(sum(p.numel() for p in model.parameters())),
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


@mcp.tool()
def unload_model(model_id: str) -> str:
    """Unload a previously loaded model and tokenizer from memory."""
    if model_id not in _models:
        return safe_json({"error": f"Model '{model_id}' not found"})
    del _models[model_id]
    _tokenizers.pop(model_id, None)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return safe_json({"status": "unloaded", "model_id": model_id})


@mcp.tool()
def estimate_memory(
    model_id: str = "gpt2",
    seq_len: int = 2048,
    batch_size: int = 1,
    dtype: str = "auto",
) -> str:
    """Estimate rough inference memory needs without fully loading the model."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id)
        hidden = getattr(cfg, "hidden_size", 768)
        layers = getattr(cfg, "num_hidden_layers", 12)
        vocab = getattr(cfg, "vocab_size", 50257)
        params = layers * 12 * hidden * hidden + vocab * hidden
        resolved_dtype = "float16" if dtype == "auto" and get_device() != "cpu" else ("float32" if dtype == "auto" else dtype)
        bytes_per_param = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1}.get(resolved_dtype, 4)
        weights_gb = params * bytes_per_param / 1e9
        activations_gb = batch_size * seq_len * hidden * layers * bytes_per_param / 1e9
        total_gb = weights_gb + activations_gb + 0.5
        return safe_json({
            "model_id": model_id,
            "dtype": resolved_dtype,
            "approx_parameters": int(params),
            "estimated_weight_memory_gb": round(weights_gb, 2),
            "estimated_activation_memory_gb": round(activations_gb, 2),
            "estimated_total_memory_gb": round(total_gb, 2),
        })
    except Exception as e:
        return safe_json({"error": str(e), "model_id": model_id})


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
