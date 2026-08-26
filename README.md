# AI/ML MCP Servers by NaTo1000

**Full production suite of Model Context Protocol (MCP) servers** specialized for every major AI/ML build type.

> USB-C for AI agents — plug any LLM (Claude, Cursor, VS Code, Windsurf, etc.) into real ML tools, models, datasets, vector stores, training loops, multimodal pipelines, speech, and more.

**GitHub:** https://github.com/NaTo1000/ai-ml-mcp-servers  
**Author:** NaTo1000 (BPB BLUEPRINTBOT PTY LTD) — Melbourne

---

## Quick Start

```bash
git clone https://github.com/NaTo1000/ai-ml-mcp-servers.git
cd ai-ml-mcp-servers
pip install -e ".[full]"

# Run any server (stdio)
mcp-nlp-embed
mcp-nlp-text
mcp-vector-chroma
mcp-vector-faiss
mcp-cv-vision
mcp-cv-ocr
mcp-mm-vlm
mcp-audio-speech
mcp-infer-local
mcp-infer-api
mcp-train-hf
mcp-hf-hub
mcp-data-prep
mcp-agent-tools
mcp-metrics
```

Claude Desktop / Cursor example config → `configs/claude_desktop_config.example.json`

---

## Complete Tool-Set Catalog — All Build Types

### 1. Core Infrastructure & Model Hub
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **Hugging Face Hub** | `mcp-hf-hub` | `search_models`, `search_datasets`, `get_model_info`, `list_spaces`, `whoami` |
| **Local Inference** | `mcp-infer-local` | `load_model`, `generate`, `chat`, `list_loaded_models` |
| **API / Cloud Inference** | `mcp-infer-api` | `openai_chat`, `hf_inference`, `generic_http_infer` |
| **Embeddings** | `mcp-nlp-embed` | `embed_text`, `batch_embed`, `cosine_similarity`, `semantic_search` |

### 2. Vector Stores & RAG
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **Chroma** | `mcp-vector-chroma` | `create_collection`, `add_documents`, `query_collection`, `list_collections`, `delete_collection` |
| **FAISS** | `mcp-vector-faiss` | `create_index`, `add_vectors`, `search`, `save_index`, `load_index` |

### 3. NLP & Text Intelligence
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **Text Analysis** | `mcp-nlp-text` | `classify_text`, `extract_entities`, `sentiment_analysis`, `summarize`, `extract_keywords` |
| **Embeddings** | `mcp-nlp-embed` | (see above) |

### 4. Computer Vision
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **Vision Models** | `mcp-cv-vision` | `classify_image`, `detect_objects`, `caption_image`, `zero_shot_classify` |
| **OCR & Documents** | `mcp-cv-ocr` | `ocr_image`, `ocr_pdf_page`, `extract_text_blocks` |

### 5. Multimodal (Vision-Language)
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **VLM** | `mcp-mm-vlm` | `image_qa`, `describe_scene`, `vlm_query` |

### 6. Audio & Speech
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **Speech (Whisper)** | `mcp-audio-speech` | `transcribe`, `detect_language`, `list_whisper_models` |

### 7. Training, Fine-Tuning & PEFT
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **HF Training / LoRA** | `mcp-train-hf` | `create_lora_config`, `prepare_sft_dataset`, `estimate_train_memory`, `merge_lora_adapter` |

### 8. Data Engineering
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **Dataset Prep** | `mcp-data-prep` | `load_dataset`, `split_dataset`, `list_loaded` |

### 9. Evaluation & Metrics
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **Metrics** | `mcp-metrics` | `compute_accuracy`, `compute_bleu`, `compute_rouge`, `classification_report_simple`, `measure_latency` |

### 10. Agents & Orchestration
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **Agent Primitives** | `mcp-agent-tools` | `memory_write`, `memory_read`, `memory_list`, `add_note`, `list_notes`, `clear_memory` |

### 11. Supporting / Cross-Cutting (recommended official + community)
- Filesystem, Git, GitHub, Postgres, Time, Memory (official MCP reference servers)
- Browser automation, web search, Slack, Notion, etc.

---

## Recommended Combinations by Build Type

| Build Type | Recommended Servers |
|------------|---------------------|
| **RAG / Knowledge** | `mcp-nlp-embed` + `mcp-vector-chroma` (or faiss) + `mcp-hf-hub` + `mcp-data-prep` |
| **Chat / Local LLM** | `mcp-infer-local` + `mcp-agent-tools` + `mcp-hf-hub` |
| **Cloud Agent** | `mcp-infer-api` + `mcp-agent-tools` + `mcp-hf-hub` |
| **Fine-Tune / LoRA** | `mcp-train-hf` + `mcp-data-prep` + `mcp-metrics` + `mcp-hf-hub` |
| **Computer Vision App** | `mcp-cv-vision` + `mcp-cv-ocr` + `mcp-mm-vlm` |
| **Speech / Voice** | `mcp-audio-speech` + `mcp-infer-api` (for TTS providers) |
| **Multimodal Agent** | `mcp-mm-vlm` + `mcp-cv-vision` + `mcp-nlp-text` + `mcp-agent-tools` |
| **Full MLOps Loop** | All of the above + official GitHub / Filesystem / Postgres MCP servers |

---

## Repository Structure

```
ai-ml-mcp-servers/
├── servers/
│   ├── common.py
│   ├── nlp/
│   │   ├── embeddings.py
│   │   └── text_analysis.py
│   ├── cv/
│   │   ├── vision.py
│   │   └── ocr.py
│   ├── multimodal/
│   │   └── vlm.py
│   ├── audio/
│   │   └── speech.py
│   ├── training/
│   │   └── hf_train.py
│   ├── inference/
│   │   ├── local.py
│   │   └── api.py
│   ├── vector/
│   │   ├── chroma.py
│   │   └── faiss_store.py
│   ├── data/
│   │   └── prep.py
│   ├── agents/
│   │   └── tools.py
│   └── utils/
│       ├── hf_hub.py
│       └── metrics.py
├── configs/
│   └── claude_desktop_config.example.json
├── scripts/
│   └── push_to_hf.py
├── pyproject.toml
└── README.md
```

---

## Push to Hugging Face (NaTo1000)

```bash
pip install huggingface_hub
huggingface-cli login          # use a token with write scope
python scripts/push_to_hf.py   # creates/updates NaTo1000/ai-ml-mcp-servers Space or dataset repo
```

Or manually:

```bash
huggingface-cli repo create ai-ml-mcp-servers --type space --private false
git remote add hf https://huggingface.co/spaces/NaTo1000/ai-ml-mcp-servers
git push hf main
```

The suite is also fully compatible with the official Hugging Face MCP server at https://huggingface.co/mcp.

---

## Environment Variables

| Variable | Used By | Purpose |
|----------|---------|---------|
| `HF_TOKEN` | hf-hub, inference-api, training | Auth for Hub + Inference API |
| `OPENAI_API_KEY` | inference-api | OpenAI or compatible providers |
| `OPENAI_API_BASE` | inference-api | Custom base URL (Groq, Together, vLLM…) |
| `CHROMA_PERSIST_DIR` | vector-chroma | Where Chroma stores data (default `./chroma_db`) |

---

## License

MIT — free for commercial and personal use.

Built and maintained by **NaTo1000** · BPB BLUEPRINTBOT PTY LTD · Melbourne
