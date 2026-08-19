# AI/ML MCP Servers by NaTo1000

**Full production suite of Model Context Protocol (MCP) servers** specialized for every major AI/ML build type.

> USB-C for AI agents — plug any LLM (Claude, Cursor, VS Code, etc.) into real ML tools, models, datasets, vector stores, training loops, multimodal pipelines, and more.

## Quick Start

```bash
pip install -e ".[full]"

# Run any server
mcp-nlp-embed          # embeddings + semantic search
mcp-nlp-text           # text analysis
mcp-vector-chroma      # Chroma vector DB
mcp-cv-vision          # computer vision
mcp-audio-speech       # speech / Whisper
mcp-infer-local        # local model inference
mcp-hf-hub             # Hugging Face Hub tools
# ... see all entry points in pyproject.toml
```

Claude Desktop / Cursor config example is in `configs/claude_desktop_config.example.json`.

---

## Complete Tool-Set Catalog by Build Type

### 1. Core AI/ML Infrastructure
| Tool Set | Server(s) | Key Tools |
|----------|-----------|-----------|
| Model Hub & Discovery | `utils/hf_hub` | search_models, search_datasets, get_model_info, download_model, list_spaces |
| Embeddings & Similarity | `nlp/embeddings` | embed_text, batch_embed, cosine_similarity, semantic_search |
| Vector Stores / RAG | `vector/chroma`, `vector/faiss_store` | create_collection, add_documents, query, delete, persist |
| Local Inference | `inference/local` | load_model, generate, chat, stream |
| Cloud / API Inference | `inference/api` | call_openai_compat, hf_inference, replicate_run |
| Prompt & Template Mgmt | `agents/tools` | list_prompts, render_prompt, save_template |

### 2. Training & Experimentation
| Tool Set | Server(s) | Key Tools |
|----------|-----------|-----------|
| Hugging Face Training | `training/hf_train` | prepare_dataset, start_finetune, monitor_run, save_adapter |
| LoRA / PEFT | `training/hf_train` | create_lora_config, apply_lora, merge_adapter |
| Metrics & Evaluation | `utils/metrics` | compute_bleu, compute_rouge, accuracy, latency_stats |
| Experiment Tracking | (extendable) | log_metric, log_artifact, list_runs |

### 3. NLP & Text
| Tool Set | Server(s) | Key Tools |
|----------|-----------|-----------|
| Text Analysis | `nlp/text_analysis` | summarize, classify, extract_entities, sentiment, keyword_extract |
| Embeddings | `nlp/embeddings` | (see above) |
| Document Understanding | `multimodal/vlm` + Nougat-style | pdf_to_markdown, ocr_scientific |

### 4. Computer Vision
| Tool Set | Server(s) | Key Tools |
|----------|-----------|-----------|
| Vision Models | `cv/vision` | classify_image, detect_objects, caption_image, segment |
| OCR & Documents | `cv/ocr` | ocr_image, ocr_pdf, extract_tables |

### 5. Multimodal & Audio
| Tool Set | Server(s) | Key Tools |
|----------|-----------|-----------|
| Vision-Language | `multimodal/vlm` | vlm_query, image_qa, describe_scene |
| Speech | `audio/speech` | transcribe, synthesize, voice_clone_info |

### 6. Data Engineering
| Tool Set | Server(s) | Key Tools |
|----------|-----------|-----------|
| Dataset Prep | `data/prep` | load_hf_dataset, clean, split, augment, push_to_hub |
| Feature Stores | (vector + data) | register_feature, query_features |

### 7. Agents & Orchestration
| Tool Set | Server(s) | Key Tools |
|----------|-----------|-----------|
| Agent Utilities | `agents/tools` | plan, tool_router, memory_write, memory_read, multi_agent_handoff |

### 8. Supporting Infrastructure (all builds)
- Filesystem, Git, GitHub, databases, web search, browser automation, Slack, Notion, etc. (use official MCP reference servers + community)

---

## Repository Structure

```
ai-ml-mcp-servers/
├── servers/
│   ├── common.py              # shared helpers
│   ├── nlp/
│   │   ├── embeddings.py      # production ready
│   │   └── text_analysis.py
│   ├── cv/
│   ├── multimodal/
│   ├── audio/
│   ├── training/
│   ├── inference/
│   ├── vector/
│   ├── data/
│   ├── agents/
│   └── utils/
├── configs/
│   └── claude_desktop_config.example.json
├── pyproject.toml
└── README.md
```

## Hugging Face

All servers are designed to work seamlessly with the official Hugging Face MCP server (`https://huggingface.co/mcp`) and with `huggingface_hub`.

To push this entire suite to Hugging Face under **NaTo1000**:

```bash
pip install huggingface_hub
huggingface-cli login
# Then create a Space or model repo and push the code
```

Or use the helper script (coming in next commit): `python -m servers.utils.push_to_hf`.

## License

MIT — free for any use, commercial or personal.

Built and maintained by **NaTo1000** (BPB BLUEPRINTBOT PTY LTD).
