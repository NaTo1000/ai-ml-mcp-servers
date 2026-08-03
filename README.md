# 🚀 AI/ML MCP Servers — Full Tool Suite by NaTo1000

**Production-ready Model Context Protocol (MCP) servers** for every major AI/ML build type.

Connect any MCP-compatible host (Claude Desktop, Cursor, VS Code, ChatGPT, Gemini CLI, custom agents) to powerful AI/ML tools with zero custom integration code.

> **Author**: [NaTo1000](https://github.com/NaTo1000) · BPB BLUEPRINTBOT PTY LTD  
> **License**: MIT  
> **Protocol**: [Model Context Protocol](https://modelcontextprotocol.io)

---

## 📦 Complete Tool-Set Catalog (All Build Types)

| Category | Server | Purpose | Key Tools |
|----------|--------|---------|-----------|
| **NLP** | `nlp-text-analysis` | Text classification, NER, sentiment, summarization | classify, extract_entities, sentiment, summarize |
| **NLP** | `nlp-embeddings` | Sentence & document embeddings | embed_text, similarity, batch_embed |
| **CV** | `cv-vision` | Image classification, detection, segmentation | classify_image, detect_objects, segment |
| **CV** | `cv-ocr` | OCR & document understanding | ocr_image, extract_tables, layout_analysis |
| **Multimodal** | `mm-vlm` | Vision-Language models | caption, vqa, image_text_match |
| **Multimodal** | `mm-diffusion` | Image / video generation | generate_image, img2img, inpaint |
| **Audio** | `audio-speech` | ASR, TTS, audio classification | transcribe, synthesize, classify_audio |
| **Training** | `train-hf` | Hugging Face Trainer & fine-tuning helpers | prepare_dataset, train_lora, evaluate |
| **Inference** | `infer-local` | Local model serving (Ollama / vLLM style) | list_models, generate, chat |
| **Inference** | `infer-api` | Unified multi-provider inference | openai_chat, anthropic_chat, hf_inference |
| **Vector** | `vector-chroma` | ChromaDB vector store | add_docs, query, delete, collection_stats |
| **Vector** | `vector-faiss` | FAISS index management | build_index, search, save_index |
| **Agents** | `agent-tools` | Agent orchestration helpers | plan, tool_router, memory_store |
| **Data** | `data-prep` | Dataset loading & preprocessing | load_hf_dataset, split, tokenize |
| **Utils** | `utils-metrics` | Evaluation metrics & logging | compute_bleu, compute_rouge, log_experiment |
| **Utils** | `utils-hf-hub` | Hugging Face Hub operations | search_models, download_model, push_to_hub |

---

## 🏗️ Project Structure

```
ai-ml-mcp-servers/
├── servers/
│   ├── nlp/
│   ├── cv/
│   ├── multimodal/
│   ├── audio/
│   ├── training/
│   ├── inference/
│   ├── vector/
│   ├── agents/
│   ├── data/
│   └── utils/
├── configs/          # Claude Desktop / Cursor / VS Code configs
├── docs/             # Architecture & contribution guides
├── examples/         # End-to-end usage examples
├── pyproject.toml
└── README.md
```

---

## ⚡ Quick Start

### 1. Install

```bash
pip install "mcp[cli]" huggingface_hub transformers torch pillow
# or with uv
uv add "mcp[cli]" huggingface_hub transformers torch pillow
```

### 2. Run a server (stdio — for Claude Desktop / Cursor)

```bash
python -m servers.nlp.text_analysis
```

### 3. Claude Desktop config (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "nlp-text-analysis": {
      "command": "python",
      "args": ["-m", "servers.nlp.text_analysis"]
    },
    "cv-vision": {
      "command": "python",
      "args": ["-m", "servers.cv.vision"]
    }
  }
}
```

### 4. Streamable HTTP (remote / multi-client)

```bash
python -m servers.nlp.text_analysis --transport streamable-http --port 8001
```

---

## 🧩 Design Principles

- **One concern per server** — easy to enable/disable
- **Type-hinted tools** — automatic JSON Schema generation
- **Docstring-driven** — descriptions come from Python docstrings
- **Graceful degradation** — works offline when models are cached
- **Composable** — mix any combination of servers in one host

---

## 🔗 Hugging Face

This repository is designed to be mirrored / published on Hugging Face Spaces or as a model collection under **NaTo1000**.

```bash
# After cloning
huggingface-cli login
huggingface-cli repo create ai-ml-mcp-servers --type space --private false
git remote add hf https://huggingface.co/spaces/NaTo1000/ai-ml-mcp-servers
git push hf main
```

---

## 📜 License

MIT © NaTo1000 / BPB BLUEPRINTBOT PTY LTD

---

**Built for the next generation of agentic AI systems.**  
*Understand the Universe.*
