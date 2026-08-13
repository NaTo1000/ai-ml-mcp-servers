# 🚀 AI/ML MCP Servers — Full Tool Suite by NaTo1000

**Production-ready Model Context Protocol (MCP) servers** for every major AI/ML build type.

Connect any MCP-compatible host (Claude Desktop, Cursor, VS Code, ChatGPT, Gemini CLI, custom agents) to powerful AI/ML tools with zero custom integration code.

> **Author**: [NaTo1000](https://github.com/NaTo1000) · BPB BLUEPRINTBOT PTY LTD  
> **License**: MIT  
> **Protocol**: [Model Context Protocol](https://modelcontextprotocol.io)  
> **Hugging Face**: designed for publishing under `NaTo1000`

---

## 📦 Complete Tool-Set Catalog (All Build Types)

| Category | Server | Purpose | Key Tools |
|----------|--------|---------|-----------|
| **NLP** | `nlp-text-analysis` | Text classification, NER, sentiment, summarization | `classify_text`, `extract_entities`, `sentiment_analysis`, `summarize`, `extract_keywords` |
| **NLP** | `nlp-embeddings` | Sentence & document embeddings | `embed_text`, `batch_embed`, `cosine_similarity`, `semantic_search` |
| **CV** | `cv-vision` | Image classification, detection, segmentation | `classify_image`, `detect_objects`, `segment_image`, `image_features` |
| **CV** | `cv-ocr` | OCR & document understanding *(stub ready for expansion)* | `ocr_image`, `extract_tables`, `layout_analysis` |
| **Multimodal** | `mm-vlm` | Vision-Language models *(stub)* | `image_caption`, `visual_qa`, `image_text_match` |
| **Multimodal** | `mm-diffusion` | Image / video generation *(stub)* | `generate_image`, `img2img`, `inpaint` |
| **Audio** | `audio-speech` | ASR, TTS, audio classification *(stub)* | `transcribe`, `synthesize_speech`, `classify_audio` |
| **Training** | `train-hf` | Hugging Face Trainer & LoRA helpers *(stub)* | `prepare_dataset`, `train_lora`, `evaluate_model` |
| **Inference** | `infer-local` | Local model serving (transformers / Ollama) | `list_local_models`, `generate`, `chat`, `stream_generate` |
| **Inference** | `infer-api` | Unified multi-provider inference *(stub)* | `openai_chat`, `anthropic_chat`, `hf_inference` |
| **Vector** | `vector-chroma` | ChromaDB vector store | `create_collection`, `add_documents`, `query`, `delete`, `collection_stats` |
| **Vector** | `vector-faiss` | FAISS index management *(stub)* | `build_index`, `search`, `save_index` |
| **Agents** | `agent-tools` | Agent orchestration helpers | `create_plan`, `tool_router`, `memory_store`, `memory_recall`, `reflect` |
| **Data** | `data-prep` | Dataset loading & preprocessing | `load_hf_dataset`, `split_dataset`, `tokenize`, `filter_rows`, `export_dataset` |
| **Utils** | `utils-metrics` | Evaluation metrics *(stub)* | `compute_bleu`, `compute_rouge`, `log_experiment` |
| **Utils** | `utils-hf-hub` | Hugging Face Hub operations | `search_models`, `search_datasets`, `download_model`, `push_to_hub`, `repo_info` |

**Fully implemented in this release**: NLP (text + embeddings), CV vision, Chroma vector, local inference, agent tools, data prep, HF Hub utils.

---

## 🏗️ Project Structure

```
ai-ml-mcp-servers/
├── servers/
│   ├── nlp/           # text_analysis.py, embeddings.py
│   ├── cv/            # vision.py
│   ├── multimodal/    # (ready for vlm / diffusion)
│   ├── audio/         # (ready for speech)
│   ├── training/      # (ready for hf_trainer)
│   ├── inference/     # local.py
│   ├── vector/        # chroma.py
│   ├── agents/        # tools.py
│   ├── data/          # prep.py
│   ├── utils/         # hf_hub.py
│   └── common.py      # shared helpers
├── configs/           # Claude Desktop / Cursor example configs
├── docs/
├── examples/
├── pyproject.toml
└── README.md
```

---

## ⚡ Quick Start

### 1. Install

```bash
git clone https://github.com/NaTo1000/ai-ml-mcp-servers.git
cd ai-ml-mcp-servers
pip install -e ".[full]"   # or: uv pip install -e ".[full]"
```

Core dependencies are declared in `pyproject.toml`. Optional extras: `vision`, `audio`, `full`.

### 2. Run a server (stdio — Claude Desktop / Cursor / VS Code)

```bash
python -m servers.nlp.text_analysis
python -m servers.nlp.embeddings
python -m servers.cv.vision
python -m servers.vector.chroma
python -m servers.inference.local
python -m servers.agents.tools
python -m servers.data.prep
python -m servers.utils.hf_hub
```

### 3. Claude Desktop config

Copy `configs/claude_desktop_config.example.json` into your Claude Desktop MCP settings and adjust paths / env vars as needed.

### 4. Streamable HTTP (remote / multi-client)

```bash
python -m servers.nlp.text_analysis --transport streamable-http --port 8001
```

(FastMCP supports the flag when using recent MCP SDK versions.)

---

## 🧩 Design Principles

- **One concern per server** — enable only what you need
- **Type-hinted tools** — automatic JSON Schema
- **Docstring-driven** — tool descriptions come from Python docstrings
- **Graceful degradation** — works offline when models are cached
- **Composable** — mix any combination of servers in one host
- **Production defaults** — caching, device auto-detect, safe serialization

---

## 🔗 Publish to Hugging Face (NaTo1000)

```bash
# Login once
huggingface-cli login

# Create a Space (or model repo) under your account
huggingface-cli repo create ai-ml-mcp-servers --type space --private false

# Add remote and push
git remote add hf https://huggingface.co/spaces/NaTo1000/ai-ml-mcp-servers
git push hf main
```

You can also turn any Gradio demo into an MCP endpoint by setting `mcp_server=True` in `launch()`.

The official Hugging Face MCP server (`https://huggingface.co/mcp`) can then discover and call tools from your Space.

---

## 📜 License

MIT © NaTo1000 / BPB BLUEPRINTBOT PTY LTD

---

**Built for the next generation of agentic AI systems.**  
*Understand the Universe.*
