---
title: Farm Agent
emoji: 👁
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
license: mit
short_description: one tool for all your farm queries
---

# 🌾 Farm Agent — Intelligent Multi-Modal Farm Assistant

A production-grade **LangGraph agent** combining RAG, vision, web search, and yield prediction to help Indian farmers with any farming query.

[![Demo](https://img.shields.io/badge/🤗-Live_Demo-yellow)](https://huggingface.co/spaces/shank-j/farm-agent)
[![CI/CD](https://github.com/shashank-sj/farm-agent/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/shashank-sj/farm-agent/actions/workflows/ci_cd.yml)

---

## Architecture

```
User Input (text + optional image)
            │
            ▼
    ┌───────────────────┐
    │   LangGraph Agent  │  ← Groq (Llama 3.1)
    └───────────────────┘
            │
    ┌───────┴────────────────────────────────┐
    │         Tool Router (automatic)         │
    └───┬──────────┬──────────┬──────────────┘
        │          │          │          │
        ▼          ▼          ▼          ▼
   📚 RAG      👁️ Vision  🔍 Search  📊 Yield
   (FAISS +   (YOLOv8    (Tavily/   (Rule-based)
   Gemini)    classifier) DuckDuckGo)
        │          │          │          │
        └──────────┴──────────┴──────────┘
                        │
                        ▼
        Structured into a FarmResponse (Pydantic)
                        │
                        ▼
               Final Answer to User
```

Every answer is passed through a `FarmResponse` Pydantic schema (`src/agent/schemas.py`) before
being shown to the user, so responses are consistently structured — topic, summary, concrete
recommendations (with how-to-use / where-to-buy / pros / cons), precautions, and sources.

**Memory:** each turn replays the full conversation (from Gradio's in-memory `history_state`)
back through the LLM, so the model does see prior turns — but it's session-only (lost on page
refresh or Space restart), unbounded (no truncation/summarization for long chats), and not
persisted anywhere. There's no cross-session recall.

## Tools

| Tool | Powered By | Handles |
|------|-----------|---------| 
| 📚 `farm_rag` | FAISS + BM25 + Gemini | Crop cultivation, soil, pests, schemes |
| 👁️ `farm_vision` | YOLOv8s-cls (31 classes) | Plant disease + pest photo analysis |
| 🔍 `farm_web_search` | Tavily / DuckDuckGo | Live prices, laws, mandi rates, eNAM |
| 📊 `yield_prediction` | Rule-based / XGBoost | Expected yield from farm parameters |

## Quickstart

```bash
git clone https://github.com/shashank-sj/farm-agent
cd farm-agent
pip install -r requirements.txt
cp .env.example .env   # add your keys

# Run the Gradio app
python app.py
```

## Project Structure

```
farm-agent/
├── app.py                     # Gradio UI — entry point (HF Spaces ready)
├── src/
│   ├── agent/
│   │   ├── graph.py           # LangGraph state machine + tool routing
│   │   └── schemas.py         # FarmResponse — structured output schema
│   └── tools/
│       ├── rag_tool.py        # Knowledge base search (FAISS + Gemini)
│       ├── vision_tool.py     # Plant/pest photo classification (YOLOv8)
│       ├── search_tool.py     # Live web search (Tavily / DuckDuckGo)
│       └── yield_tool.py      # Crop yield estimation
├── mlflow/
│   └── tracking.py            # MLflow experiment tracking
├── .github/workflows/
│   └── ci_cd.yml              # GitHub Actions CI/CD
├── data/
│   ├── knowledge_base/        # Farm PDFs/docs for RAG
│   └── faiss_index/           # Auto-generated vector index
├── outputs/
│   └── farm-vision/weights/   # YOLOv8 model weights
└── requirements.txt
```

## Setup Guide

### 1. Build the RAG Knowledge Base
`FarmRAGTool` (`src/tools/rag_tool.py`) only *loads* `data/faiss_index/` at query time — it
never builds it, and there's no index checked into this repo. Build one locally:

```bash
pip install -r requirements-ingest.txt   # pdfplumber, pytesseract — not needed to serve the app
python scripts/build_rag_index.py
```

Drop `.txt`/`.md`, `.pdf`, or `.png`/`.jpg` files into `data/knowledge_base/` first. The script
chunks each with `RecursiveCharacterTextSplitter` (~800 chars, 120 overlap), embeds with
`GoogleGenerativeAIEmbeddings`, and writes `data/faiss_index/`. PDFs get per-page text plus any
tables (rendered as Markdown tables so row/column structure survives); a page with no
extractable text (i.e. scanned) falls back to OCR via `pytesseract`, which also needs the
Tesseract OCR engine installed separately (see `requirements-ingest.txt`) — pip alone won't
install it. Note this is text retrieval, not image understanding: a photo with no text in it
yields nothing indexable, by design. Missing an optional dependency skips that file with a
warning instead of aborting the whole run.

Without an index present, the tool falls back to a plain "knowledge base not available" message.

### 2. Add the Vision Model
Copy a trained YOLOv8 classifier weights file to:
```
outputs/farm-vision/weights/best.pt
```

### 3. Deploy to HuggingFace Spaces
1. Create Space (Gradio SDK)
2. Add `HF_TOKEN`, `GROQ_API_KEY`, `GEMINI_API_KEY` to GitHub Secrets
3. Update `HF_SPACE` in `.github/workflows/ci_cd.yml`
4. Push to `main` → auto-deploys

> **Heads up:** `outputs/` and `data/faiss_index/` are both gitignored, and the CI/CD deploy
> step's `ignore_patterns` additionally excludes `data/faiss_index/` on top of that — so even a
> locally built index won't reach the Space through this pipeline as configured. To actually
> serve RAG/vision from the Space, either commit the built index (and drop it from
> `ignore_patterns`) or upload it directly to the Space outside of CI (e.g. `huggingface-cli
> upload` or `huggingface_hub`'s `upload_folder`).

## Environment Variables

```bash
GROQ_API_KEY=your_groq_key            # Required — agent LLM (Llama 3.1 via Groq)
GEMINI_API_KEY=your_gemini_key        # Required — RAG embeddings
TAVILY_API_KEY=your_tavily_key        # Optional — web search (falls back to DuckDuckGo)
HF_TOKEN=your_hf_token                # For HF Spaces deployment
```

## Resume Bullets

```
• Built production multi-modal farm agent using LangGraph with 4 tools:
  RAG (FAISS + Gemini embeddings), YOLOv8 vision, real-time web search,
  and rule-based yield prediction — automatically routed based on query type

• Structured every agent answer through a Pydantic schema (topic, summary,
  recommendations with pros/cons, precautions, sources) for consistent,
  farmer-readable output regardless of which tool answered the query

• Deployed end-to-end on HuggingFace Spaces via GitHub Actions CI/CD
  with MLflow experiment tracking across all agent interactions

• System handles text queries, image uploads (plant disease/pest
  detection), market price lookups, legal crop queries, and
  data-driven yield predictions in a single conversational interface
```
