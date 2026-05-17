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
    │   LangGraph Agent  │  ← Gemini 1.5 Flash / Fine-tuned Gemma-2B
    └───────────────────┘
            │
    ┌───────┴────────────────────────────────┐
    │         Tool Router (automatic)         │
    └───┬──────────┬──────────┬──────────────┘
        │          │          │          │
        ▼          ▼          ▼          ▼
   📚 RAG      👁️ Vision  🔍 Search  📊 Yield
   (FAISS +   (YOLOv8   (Tavily/   (XGBoost/
   BM25 +     31 classes) DuckDuckGo) Rule-based)
   Gemini)
        │          │          │          │
        └──────────┴──────────┴──────────┘
                        │
                        ▼
               Final Answer to User
                        │
                        ▼
               MLflow (logged)
```

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
python app/gradio_app.py
```

## Project Structure

```
farm-agent/
├── src/
│   ├── agent/
│   │   └── graph.py          # LangGraph state machine
│   └── tools/
│       └── tools.py          # All 4 tool implementations
├── app/
│   └── gradio_app.py         # Gradio UI (HF Spaces ready)
├── mlflow/
│   └── tracking.py           # MLflow experiment tracking
├── .github/workflows/
│   └── ci_cd.yml             # GitHub Actions CI/CD
├── data/
│   ├── knowledge_base/       # Farm PDFs/docs for RAG
│   └── faiss_index/          # Auto-generated vector index
├── outputs/
│   └── farm-vision/weights/  # YOLOv8 model weights
└── requirements.txt
```

## Setup Guide

### 1. Build RAG Knowledge Base
Add farm PDFs/TXTs to `data/knowledge_base/` then:
```python
from src.tools.tools import FarmRAGTool
rag = FarmRAGTool(gemini_api_key="your_key", index_path="data/faiss_index")
rag.build_index("data/knowledge_base")
```

### 2. Add Vision Model
Copy `best.pt` from Project 3 (Kaggle) to:
```
outputs/farm-vision/weights/best.pt
```

### 3. (Optional) Add Fine-tuned Gemma
Copy adapter from Project 2 to:
```
outputs/gemma-farm-qlora/final/
```
Then set `use_local_llm=True` in `FarmAgent(...)`.

### 4. Deploy to HuggingFace Spaces
1. Create Space (Gradio SDK)
2. Add `HF_TOKEN` + `GEMINI_API_KEY` to GitHub Secrets
3. Update `HF_SPACE` in `.github/workflows/ci_cd.yml`
4. Push to `main` → auto-deploys

## Environment Variables

```bash
GEMINI_API_KEY=your_gemini_key        # Required
TAVILY_API_KEY=your_tavily_key        # Optional (web search)
HF_TOKEN=your_hf_token                # For HF Spaces deployment
MODEL_PATH=outputs/farm-vision/...    # Vision model path
```

## Resume Bullets

```
• Built production multi-modal farm agent using LangGraph with 4 tools:
  RAG (FAISS+BM25), YOLOv8 vision (31 classes), real-time web search,
  and ML yield prediction — automatically routed based on query type

• Integrated fine-tuned Gemma-2B (QLoRA, Project 2) as agent LLM
  with Gemini fallback, enabling fully open-source inference pipeline

• Deployed end-to-end on HuggingFace Spaces via GitHub Actions CI/CD
  with MLflow experiment tracking across all agent interactions

• System handles text queries, image uploads (plant disease/pest
  detection), market price lookups, legal crop queries, and
  data-driven yield predictions in a single conversational interface
```
