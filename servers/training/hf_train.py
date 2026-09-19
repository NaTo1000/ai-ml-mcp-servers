"""
Hugging Face Training / PEFT MCP Server
Dataset prep helpers, LoRA config, lightweight fine-tune launch, adapter merge.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from servers.common import create_server, get_device, safe_json

logger = logging.getLogger(__name__)
mcp = create_server(
    "training-hf",
    "Training helpers: create LoRA/PEFT configs, prepare datasets for SFT, "
    "launch lightweight fine-tunes, and merge adapters. Designed for agent-driven ML workflows.",
)


@mcp.tool()
def create_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    task_type: str = "CAUSAL_LM",
) -> str:
    """Generate a PEFT LoRA config dictionary ready for use with transformers + peft."""
    cfg = {
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "bias": "none",
        "task_type": task_type,
        "target_modules": target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"],
    }
    return safe_json({"lora_config": cfg, "library": "peft"})


@mcp.tool()
def prepare_sft_dataset(
    dataset_id: str,
    text_field: str = "text",
    split: str = "train",
    max_samples: Optional[int] = 1000,
) -> str:
    """Load a dataset and return a summary + sample ready for SFT. Does not start training."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split)
    if text_field not in ds.column_names:
        return safe_json({
            "error": f"Field '{text_field}' not found",
            "available_columns": ds.column_names,
            "dataset_id": dataset_id,
        })
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    sample = [dict(r) for r in ds.select(range(min(3, len(ds))))]
    return safe_json({
        "dataset_id": dataset_id,
        "num_rows": len(ds),
        "columns": ds.column_names,
        "text_field": text_field,
        "sample": sample,
        "hint": "Use this with your training script or transformers Trainer",
    })


@mcp.tool()
def list_peft_methods() -> str:
    """List common PEFT / fine-tuning strategies and when to use them."""
    methods = [
        {
            "name": "LoRA",
            "best_for": ["causal-lm", "seq2seq", "vision-language"],
            "strengths": ["low memory", "simple merge", "wide ecosystem support"],
        },
        {
            "name": "QLoRA",
            "best_for": ["large language models", "single-GPU fine-tuning"],
            "strengths": ["4-bit base weights", "very low VRAM", "good quality/cost tradeoff"],
        },
        {
            "name": "Prefix Tuning",
            "best_for": ["generation tasks", "small adaptation footprint"],
            "strengths": ["few trainable params", "keeps base frozen"],
        },
        {
            "name": "Prompt Tuning",
            "best_for": ["classification", "instruction adaptation"],
            "strengths": ["smallest parameter footprint", "fast experiments"],
        },
        {
            "name": "Full Fine-Tune",
            "best_for": ["small models", "maximum adaptation quality"],
            "strengths": ["highest ceiling", "full control"],
            "tradeoffs": ["highest memory and compute cost"],
        },
    ]
    return safe_json({"methods": methods})


@mcp.tool()
def get_trainer_template(
    task: str = "sft",
    use_lora: bool = True,
    learning_rate: float = 2e-4,
    epochs: int = 3,
    per_device_batch_size: int = 4,
) -> str:
    """Return a practical trainer configuration template for common fine-tuning tasks."""
    task = task.lower()
    task_presets = {
        "sft": {
            "trainer": "transformers.Trainer",
            "dataset_expectation": "one text field containing the full prompt/response transcript",
            "max_seq_length": 2048,
        },
        "classification": {
            "trainer": "transformers.Trainer",
            "dataset_expectation": "text plus label columns",
            "max_seq_length": 512,
        },
        "seq2seq": {
            "trainer": "transformers.Seq2SeqTrainer",
            "dataset_expectation": "input text plus target text columns",
            "max_seq_length": 1024,
        },
    }
    preset = task_presets.get(task, task_presets["sft"])
    template = {
        "task": task,
        "trainer": preset["trainer"],
        "dataset_expectation": preset["dataset_expectation"],
        "training_arguments": {
            "learning_rate": learning_rate,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": per_device_batch_size,
            "per_device_eval_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": 4,
            "warmup_ratio": 0.03,
            "weight_decay": 0.01,
            "logging_steps": 10,
            "save_strategy": "epoch",
            "evaluation_strategy": "epoch",
            "fp16": get_device() == "cuda",
            "bf16": False,
            "report_to": [],
        },
        "modeling": {
            "use_lora": use_lora,
            "suggested_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"] if use_lora else [],
            "max_seq_length": preset["max_seq_length"],
        },
    }
    return safe_json(template)


@mcp.tool()
def estimate_train_memory(
    model_id: str = "gpt2",
    batch_size: int = 4,
    seq_len: int = 512,
    lora_r: int = 16,
) -> str:
    """Rough VRAM estimate for LoRA fine-tuning (rule-of-thumb)."""
    # Very approximate heuristic
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id)
        hidden = getattr(cfg, "hidden_size", 768)
        layers = getattr(cfg, "num_hidden_layers", 12)
        vocab = getattr(cfg, "vocab_size", 50257)
        # rough param count
        params_b = (layers * 12 * hidden * hidden + vocab * hidden) / 1e9
        # LoRA adds little; activations dominate
        activation_gb = batch_size * seq_len * hidden * layers * 4 / 1e9  # fp16-ish
        model_gb = params_b * 2  # fp16
        total = model_gb + activation_gb + 1.5  # optimizer states for LoRA small
        return safe_json({
            "model_id": model_id,
            "approx_params_B": round(params_b, 3),
            "estimated_vram_gb": round(total, 2),
            "notes": "Rough estimate only — real usage varies with implementation",
        })
    except Exception as e:
        return safe_json({"error": str(e)})


@mcp.tool()
def merge_lora_adapter(
    base_model_id: str,
    adapter_path: str,
    output_dir: str,
) -> str:
    """Merge a PEFT LoRA adapter into the base model and save the full weights."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch

        device = get_device()
        logger.info("Loading base %s ...", base_model_id)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        merged = model.merge_and_unload()
        os.makedirs(output_dir, exist_ok=True)
        merged.save_pretrained(output_dir)
        tok = AutoTokenizer.from_pretrained(base_model_id)
        tok.save_pretrained(output_dir)
        return safe_json({"status": "merged", "output_dir": output_dir})
    except Exception as e:
        return safe_json({"error": str(e), "hint": "Requires peft + transformers + enough RAM/VRAM"})


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
