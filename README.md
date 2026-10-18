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

# 🌾 Farm Agent — Multi-Modal Farm Assistant

A LangGraph agent for Indian farmers that combines hybrid RAG, a plant disease / pest image
classifier, live web search, and a trained yield model, with a fine-tuned small LLM as a
fallback and a deterministic safety layer in front of everything.

[![Demo](https://img.shields.io/badge/🤗-Live_Demo-yellow)](https://huggingface.co/spaces/shank-j/farm_agent)
[![CI/CD](https://github.com/shashank-sj/farm-agent/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/shashank-sj/farm-agent/actions/workflows/ci_cd.yml)

---

## Architecture

```
user message (+ optional photo)
        │
        ▼
 ┌─────────────────────┐   crisis / poisoning ──► fixed helpline / first-aid answer (no LLM)
 │ guardrails.screen() │   banned pesticide   ──► warning appended to the final answer
 └─────────────────────┘
        │
        ▼                       LangGraph
 ┌────────────────────────────────────────────────────────────────────────┐
 │  START ─ photo? ─ yes ─► vision node (YOLOv8s-cls, runs deterministically)│
 │            │                      │                                       │
 │            no                     ▼                                       │
 │            └──────────────► agent (Groq gpt-oss-20b) ◄──────┐            │
 │                                   │ tool calls?             │            │
 │                                   ▼                         │            │
 │                     tools: farm_rag │ farm_web_search │ yield_prediction  │
 └────────────────────────────────────────────────────────────────────────┘
        │                                    if the Groq call fails:
        ▼                                    fine-tuned Qwen2.5-0.5B + RAG context answers instead
 FarmResponse card (Pydantic structured output) ─► telemetry record ─► user
```

| Component | What it is | Where |
|---|---|---|
| `farm_rag` | Hybrid retrieval: BM25 (from scratch) + FAISS (BGE-small embeddings) → Reciprocal Rank Fusion → cross-encoder re-ranking (MiniLM) | `src/retrieval/`, `src/tools/rag_tool.py` |
| vision | YOLOv8s-cls, 27 classes (15 PlantVillage leaf classes + 12 insect classes), confidence-gated, beneficial insects flagged | `src/tools/vision_tool.py` |
| `farm_web_search` | Tavily restricted to agri/government domains, DuckDuckGo fallback, results framed as untrusted | `src/tools/search_tool.py` |
| `yield_prediction` | Gradient-boosted trees on Indian state crop data (1997–2020), time-based split; rule-based fallback | `scripts/train_yield_model.py`, `src/tools/yield_tool.py` |
| fallback LLM | Qwen2.5-0.5B-Instruct fine-tuned with QLoRA on 1,859 cleaned agri Q&A pairs (from 22.6K raw rows, 72% of them duplicates) | `finetune/`, `src/llm/local_llm.py` |
| guardrails | Pattern-based crisis / poisoning / banned-pesticide screen that runs before the LLM | `src/agent/guardrails.py` |
| telemetry | One structured JSON record per interaction; optional MLflow metrics | `src/observability/telemetry.py` |

### Design decisions

- **Vision is not an LLM tool.** When a photo is attached the graph always runs the classifier first.
  Routing an image needs no judgement, and an LLM-controlled file-path argument would let a prompt point
  the tool at any file on the server.
- **The model is never the safety mechanism.** Crisis and poisoning messages get fixed, reviewed text
  before any LLM call; a jailbreak can't talk a regex out of its rules.
- **The index is a build artifact.** CI rebuilds `data/rag_index/` on every deploy, and it's stored as
  raw FAISS + JSONL rather than LangChain's pickled docstore, which needs `allow_dangerous_deserialization`.
- **Bounded agent loop.** At most 6 LLM calls per question; the last call has no tools bound, so the user
  always gets an answer instead of an empty message.
- **Bounded memory.** Only the last 6 conversation turns are replayed to the LLM.

## Results

Each number comes from a script in this repo, run in the Kaggle notebooks in `notebooks/`.

| What | Result | Source |
|---|---|---|
| Vision, held-out test (shipped `best.pt`, reproducible split, 1,393 images, 27 classes) | top-1 96.5%, macro-F1 0.959, ECE 0.020; weakest classes beetle (F1 0.76) and caterpillar (0.78) | `results/vision/test_metrics.json` |
| Retrieval, 1,922 queries / 2,039 answers, 95% bootstrap CIs | cross-encoder re-ranking: MRR@10 0.543 → 0.570 and Recall@1 0.435 → 0.474 over dense (significant); RRF hybrid *without* re-ranking is worse than dense (MRR 0.498), so the no-re-ranker fallback is dense | `results/retrieval/retrieval_eval.md` |
| Yield model vs lookup baseline (test years ≥ 2016) | MAE 0.96 vs 1.10 t/ha; median error 16.0% vs 20.7%; R² (log) 0.935 vs 0.893. A random split would report 0.70, which is optimistic | `outputs/yield_model/metadata.json` |
| Test suite | 59 passed | `results/pytest.txt` |
| Fine-tune, 109 held-out questions: base → QLoRA → QLoRA + RAG (train-only index) | answer perplexity 9.97 → 6.82 → 4.94; ROUGE-L 0.134 → 0.231 → 0.313 | `results/finetune/SUMMARY.md` |
| LLM judge (gpt-oss-120b, same 100 questions, 1–5) | correctness: base 2.74, QLoRA 2.81 (flat: fine-tuning changed style, 95 → 30 words, not knowledge), base + RAG **3.24**. Retrieval is what makes answers more correct | `results/finetune/eval_*.json` |
| QLoRA vs LoRA vs full (same data, T4) | peak memory 6.97 / 6.82 / 8.19 GB; test perplexity 6.82 / 6.44 / 6.31. At 0.5B, QLoRA saves no memory | `results/finetune/*_train_summary.json` |
| Post-training quantisation of the QLoRA model | NF4: −49% memory, no perplexity loss; INT8: −34% memory, 2.8× slower | `results/finetune/eval_qlora-*.json` |

## Quickstart

```bash
git clone https://github.com/shashank-sj/farm-agent && cd farm-agent
pip install -r requirements.txt -r requirements-ingest.txt
cp .env.example .env                              # add GROQ_API_KEY
python -m scripts.build_rag_index --qa-dataset    # builds data/rag_index/
python app.py
```

Tests: `pip install -r requirements-dev.txt && pytest` (offline — the LLM, embedder and re-ranker are faked).

## Training and evaluation (Kaggle)

| Notebook | GPU | What it does |
|---|---|---|
| `notebooks/kaggle_1_tests_rag_yield.ipynb` | optional | test suite, retrieval benchmark, yield model |
| `notebooks/kaggle_2_finetune.ipynb` | T4 | data split, base eval, QLoRA, int8/nf4 eval, RAG eval, QLoRA vs LoRA vs full, merge + push |
| `notebooks/kaggle_3_vision.ipynb` | T4 | reproducible split, retrain YOLOv8s-cls, held-out test metrics |
| `notebooks/kaggle_4_judge.ipynb` | none | LLM-judge the saved answers (resumable; for when Groq's daily token limit cuts notebook 2 short) |

Keys go in **Kaggle → Add-ons → Secrets** (`HF_TOKEN`, `GROQ_API_KEY`), never in notebook cells.

## Project structure

```
farm-agent/
├── app.py                         # Gradio UI (HF Spaces entry point)
├── src/
│   ├── agent/      graph.py (LangGraph), guardrails.py, schemas.py (FarmResponse)
│   ├── tools/      rag_tool.py, vision_tool.py, search_tool.py, yield_tool.py
│   ├── retrieval/  bm25.py, embedder.py, index.py, reranker.py, hybrid.py
│   ├── llm/        local_llm.py (fine-tuned fallback), prompts.py (shared train/serve prompt)
│   ├── observability/telemetry.py
│   └── agri_qa.py  # dataset cleaning + leakage-safe grouped split (shared by RAG eval and fine-tuning)
├── scripts/        build_rag_index.py, eval_retrieval.py, train_yield_model.py, train_vision.py
├── finetune/       data.py, train.py, evaluate.py, metrics.py, merge_adapter.py
├── notebooks/      Kaggle notebooks
├── tests/          offline pytest suite
├── data/knowledge_base/          # source documents for RAG
└── outputs/farm-vision/weights/  # shipped YOLOv8 weights (Git LFS)
```

## Deployment

Push to `main` → GitHub Actions runs tests + lint → rebuilds the RAG index → uploads to the HF Space.
Set `HF_TOKEN` as a GitHub secret and `GROQ_API_KEY` (plus optional `TAVILY_API_KEY`, `LOCAL_LLM_MODEL`)
as HF Space secrets. To ship the yield model, commit it after training:
`git add -f outputs/yield_model/`.

## Environment variables

See `.env.example`.
