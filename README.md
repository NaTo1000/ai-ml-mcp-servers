# AI/ML MCP Servers by NaTo1000

**Complete production suite of Model Context Protocol (MCP) servers** for every major AI/ML build type.

> USB-C for AI agents — plug any LLM (Claude, Cursor, VS Code, Windsurf, Codex, etc.) into real ML tools, models, datasets, vector stores, training loops, multimodal pipelines, speech, tabular ML, diffusion, code intelligence, time-series, graphs, RL, and more.

**GitHub:** https://github.com/NaTo1000/ai-ml-mcp-servers  
**Hugging Face:** https://huggingface.co/NaTo1000/ai-ml-mcp-servers (Space / source mirror)  
**Author:** NaTo1000 (BPB BLUEPRINTBOT PTY LTD) — Melbourne · infinite2025.com

---

## Quick Start

```bash
git clone https://github.com/NaTo1000/ai-ml-mcp-servers.git
cd ai-ml-mcp-servers
pip install -e ".[full]"

# Core entry points (stdio)
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
mcp-tabular
mcp-diffusion
mcp-code-intel
mcp-timeseries
mcp-graph
mcp-rl
```

Claude Desktop / Cursor / VS Code example → `configs/claude_desktop_config.example.json`

---

## Full Tool-Set Catalog — All Build Types

### 1. Core Infrastructure & Model Hub
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **Hugging Face Hub** | `mcp-hf-hub` | `search_models`, `search_datasets`, `get_model_info`, `list_spaces`, `download_model`, `whoami`, `list_files` |
| **Local Inference** | `mcp-infer-local` | `load_model`, `generate`, `chat`, `list_loaded_models`, `unload_model`, `estimate_memory` |
| **API / Cloud Inference** | `mcp-infer-api` | `openai_chat`, `hf_inference`, `generic_http_infer`, `list_providers` |
| **Embeddings** | `mcp-nlp-embed` | `embed_text`, `batch_embed`, `cosine_similarity`, `semantic_search`, `list_embedding_models` |

### 2. Vector Stores & RAG
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **Chroma** | `mcp-vector-chroma` | `create_collection`, `add_documents`, `query_collection`, `list_collections`, `delete_collection`, `update_documents`, `get_collection_stats` |
| **FAISS** | `mcp-vector-faiss` | `create_index`, `add_vectors`, `search`, `save_index`, `load_index`, `index_info` |

### 3. NLP & Text Intelligence
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **Text Analysis** | `mcp-nlp-text` | `classify_text`, `extract_entities`, `sentiment_analysis`, `summarize`, `extract_keywords`, `translate`, `zero_shot_classify` |
| **Embeddings** | `mcp-nlp-embed` | (see above) |

### 4. Computer Vision
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **Vision Models** | `mcp-cv-vision` | `classify_image`, `detect_objects`, `caption_image`, `zero_shot_classify`, `segment_image`, `feature_extract` |
| **OCR & Documents** | `mcp-cv-ocr` | `ocr_image`, `ocr_pdf_page`, `extract_text_blocks`, `table_extract` |

### 5. Multimodal (Vision-Language)
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **VLM** | `mcp-mm-vlm` | `image_qa`, `describe_scene`, `vlm_query`, `visual_grounding` |

### 6. Audio & Speech
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **Speech (Whisper)** | `mcp-audio-speech` | `transcribe`, `detect_language`, `list_whisper_models`, `transcribe_timestamps`, `align_audio` |

### 7. Training, Fine-Tuning & PEFT
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **HF Training / LoRA** | `mcp-train-hf` | `create_lora_config`, `prepare_sft_dataset`, `estimate_train_memory`, `merge_lora_adapter`, `list_peft_methods`, `get_trainer_template` |

### 8. Data Engineering
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **Dataset Prep** | `mcp-data-prep` | `load_dataset`, `split_dataset`, `list_loaded`, `filter_dataset`, `map_dataset`, `push_to_hub` |

### 9. Evaluation & Metrics
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **Metrics** | `mcp-metrics` | `compute_accuracy`, `compute_bleu`, `compute_rouge`, `classification_report_simple`, `measure_latency`, `compute_f1`, `compute_perplexity` |

### 10. Agents & Orchestration
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **Agent Primitives** | `mcp-agent-tools` | `memory_write`, `memory_read`, `memory_list`, `add_note`, `list_notes`, `clear_memory`, `tool_trace` |

### 11. Tabular / Classic ML
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **Tabular ML** | `mcp-tabular` | `load_csv`, `describe_data`, `train_classifier`, `train_regressor`, `predict`, `feature_importance`, `cross_validate` |

### 12. Generative / Diffusion
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **Diffusion** | `mcp-diffusion` | `list_diffusers`, `text2img_info`, `img2img_info`, `estimate_vram`, `pipeline_template` |

### 13. Code Intelligence
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **Code Intel** | `mcp-code-intel` | `embed_code`, `semantic_code_search`, `summarize_function`, `detect_language`, `extract_imports` |

### 14. Time-Series
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **Time-Series** | `mcp-timeseries` | `load_series`, `forecast_naive`, `rolling_stats`, `detect_anomalies`, `resample` |

### 15. Graph / Knowledge
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **Graph Tools** | `mcp-graph` | `create_graph`, `add_nodes`, `add_edges`, `query_neighbors`, `shortest_path`, `export_graph` |

### 16. Reinforcement Learning
| Tool Set | Entry Point | Key Tools |
|----------|-------------|-----------|
| **RL Helpers** | `mcp-rl` | `list_gym_envs`, `env_info`, `create_rollout_template`, `compute_returns`, `policy_eval_template` |

### 17. Supporting / Cross-Cutting (recommended)
- Official MCP reference: Filesystem, Git, GitHub, Postgres, Time, Memory, Fetch, Sequential Thinking
- Browser automation, web search, Slack, Notion, Sentry, Docker, Kubernetes, AWS/Azure MCP servers

---

## Recommended Combinations by Build Type

| Build Type | Recommended Servers |
|------------|---------------------|
| **RAG / Knowledge Base** | `mcp-nlp-embed` + `mcp-vector-chroma` (or faiss) + `mcp-hf-hub` + `mcp-data-prep` + `mcp-agent-tools` |
| **Chat / Local LLM** | `mcp-infer-local` + `mcp-agent-tools` + `mcp-hf-hub` |
| **Cloud Agent** | `mcp-infer-api` + `mcp-agent-tools` + `mcp-hf-hub` |
| **Fine-Tune / LoRA / PEFT** | `mcp-train-hf` + `mcp-data-prep` + `mcp-metrics` + `mcp-hf-hub` |
| **Computer Vision App** | `mcp-cv-vision` + `mcp-cv-ocr` + `mcp-mm-vlm` |
| **Speech / Voice Agent** | `mcp-audio-speech` + `mcp-infer-api` (TTS) |
| **Multimodal Agent** | `mcp-mm-vlm` + `mcp-cv-vision` + `mcp-nlp-text` + `mcp-agent-tools` |
| **Tabular / AutoML** | `mcp-tabular` + `mcp-data-prep` + `mcp-metrics` |
| **Image Generation** | `mcp-diffusion` + `mcp-hf-hub` + `mcp-cv-vision` |
| **Code Assistant** | `mcp-code-intel` + `mcp-nlp-embed` + `mcp-infer-local` + official GitHub MCP |
| **Time-Series Forecasting** | `mcp-timeseries` + `mcp-tabular` + `mcp-metrics` |
| **Knowledge Graph RAG** | `mcp-graph` + `mcp-vector-chroma` + `mcp-nlp-embed` |
| **RL / Agents** | `mcp-rl` + `mcp-agent-tools` + `mcp-metrics` |
| **Full MLOps Loop** | All of the above + official GitHub / Filesystem / Postgres / Docker MCP servers |

---

## Repository Structure

```
ai-ml-mcp-servers/
├── servers/
│   ├── common.py
│   ├── nlp/          # embeddings, text_analysis
│   ├── cv/           # vision, ocr
│   ├── multimodal/   # vlm
│   ├── audio/        # speech
│   ├── training/     # hf_train
│   ├── inference/    # local, api
│   ├── vector/       # chroma, faiss_store
│   ├── data/         # prep
│   ├── agents/       # tools
│   ├── utils/        # hf_hub, metrics
│   ├── tabular/      # classic ML
│   ├── diffusion/    # generative image
│   ├── code/         # code intelligence
│   ├── timeseries/   # forecasting & anomalies
│   ├── graph/        # knowledge graphs
│   └── rl/           # reinforcement learning helpers
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
huggingface-cli login          # write-scoped token
export HF_TOKEN=hf_...
python scripts/push_to_hf.py   # creates/updates NaTo1000/ai-ml-mcp-servers
```

Or:

```bash
huggingface-cli repo create ai-ml-mcp-servers --type space --private false
# then upload via the script or git remote
```

Compatible with the official Hugging Face MCP server: https://huggingface.co/mcp

---

## Environment Variables

| Variable | Used By | Purpose |
|----------|---------|---------|
| `HF_TOKEN` | hf-hub, inference-api, training, data-prep | Hub + Inference auth |
| `OPENAI_API_KEY` | inference-api | OpenAI-compatible providers |
| `OPENAI_API_BASE` | inference-api | Custom base (Groq, Together, vLLM, Ollama…) |
| `CHROMA_PERSIST_DIR` | vector-chroma | Persistent storage (default `./chroma_db`) |
| `DEVICE` | most servers | Force `cpu` / `cuda` / `mps` |

---

## License

MIT — free for commercial and personal use.

Built and maintained by **NaTo1000** · BPB BLUEPRINTBOT PTY LTD · Melbourne · infinite2025.com
