"""
Build the knowledge index used by farm_rag (src/tools/rag_tool.py).

This is an offline step (run in CI before deploy, or on Kaggle) — the app only
ever *loads* data/rag_index/. Ingestion-only dependencies (pdfplumber, OCR,
datasets) live in requirements-ingest.txt so the serving image stays small.

Usage:
    pip install -r requirements.txt -r requirements-ingest.txt
    python -m scripts.build_rag_index                         # documents only
    python -m scripts.build_rag_index --qa-dataset             # + cleaned agri Q&A pairs
    python -m scripts.build_rag_index --kb-dir none --qa-jsonl data/finetune/train.jsonl \
        --out data/rag_index_train_only                        # train-split-only index (fine-tune eval)

Output (data/rag_index/): index.faiss + chunks.jsonl + manifest.json — see src/retrieval/index.py.

Two kinds of content:
  * Documents (guides, PDFs) are split into ~800-character chunks.
  * Q&A pairs are indexed one pair per chunk ("Q: … / A: …") and never split:
    a pair is already a self-contained unit, and cutting it would separate an
    answer from the question that gives it meaning.

Supported input, dropped into data/knowledge_base/:
  .txt / .md   — read as plain text
  .pdf         — per-page text + tables (rendered as Markdown tables); a page
                 with no extractable text (i.e. a scanned page) falls back to
                 OCR via pytesseract, if installed
  .png/.jpg    — OCR'd via pytesseract; a photo with no text in it simply
                 yields nothing indexable — this is text retrieval, not image
                 understanding, so a picture of a wheat field won't be
                 "understood," only text visible in an image will be

Missing an optional dependency (pdfplumber, or Tesseract for OCR) doesn't
abort the run — that file is skipped with a warning so the rest still builds.
"""

import argparse
import json
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("build-rag-index")

KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")
INDEX_DIR = Path("data/rag_index")

CHUNK_SIZE = 800       # characters — short farm-advice paragraphs, not academic PDFs
CHUNK_OVERLAP = 120

TEXT_EXTENSIONS = {".txt", ".md"}
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# ── Loaders: file → list of {"text", "metadata"} units ──────────────────────

def load_text_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [{"text": text, "metadata": {"source": path.name}}] if text.strip() else []


def load_pdf_file(path: Path) -> list[dict]:
    try:
        import pdfplumber
    except ImportError:
        logger.warning(f"Skipping {path.name}: run `pip install pdfplumber` to index PDFs.")
        return []

    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()

            tables_md = [_table_to_markdown(t) for t in page.extract_tables()]
            tables_md = [t for t in tables_md if t]
            if tables_md:
                text = (text + "\n\n" + "\n\n".join(tables_md)).strip()

            if not text:
                text = _ocr_pdf_page(page, path, i)  # likely a scanned page

            if text:
                pages.append({"text": text, "metadata": {"source": path.name, "page": i}})
    return pages


def _table_to_markdown(table: list[list]) -> str:
    """Render a pdfplumber table (list of row lists) as a Markdown table, so
    row/column structure survives as readable text instead of being lost."""
    rows = [[cell or "" for cell in row] for row in table if row]
    if not rows:
        return ""
    header, *body = rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def _ocr_pdf_page(page, path: Path, page_num: int) -> str:
    try:
        import pytesseract
    except ImportError:
        logger.warning(
            f"{path.name} page {page_num} has no extractable text (likely scanned) — "
            f"install pytesseract + the Tesseract OCR engine to index it."
        )
        return ""
    try:
        image = page.to_image(resolution=200).original
        text = pytesseract.image_to_string(image).strip()
        if text:
            logger.info(f"  OCR'd {path.name} page {page_num} ({len(text)} chars)")
        return text
    except Exception as e:
        logger.warning(f"OCR failed on {path.name} page {page_num}: {e}")
        return ""


def load_image_file(path: Path) -> list[dict]:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        logger.warning(
            f"Skipping {path.name}: run `pip install pytesseract Pillow` "
            f"(+ install the Tesseract OCR engine) to index images."
        )
        return []
    try:
        text = pytesseract.image_to_string(Image.open(path)).strip()
    except Exception as e:
        logger.warning(f"OCR failed on {path.name}: {e}")
        return []
    if not text:
        logger.info(f"  {path.name}: OCR found no text — nothing to index from this image")
        return []
    return [{"text": text, "metadata": {"source": path.name}}]


def load_documents(directory: Path) -> list[dict]:
    if not directory.exists():
        logger.warning(f"{directory}/ doesn't exist — no documents indexed from it.")
        return []

    files = sorted(p for p in directory.rglob("*") if p.is_file())
    if not files:
        logger.warning(f"No files found in {directory}/.")
        return []

    docs = []
    for path in files:
        ext = path.suffix.lower()
        if ext in TEXT_EXTENSIONS:
            docs += load_text_file(path)
        elif ext in PDF_EXTENSIONS:
            docs += load_pdf_file(path)
        elif ext in IMAGE_EXTENSIONS:
            docs += load_image_file(path)
        else:
            logger.info(f"Skipping {path.name}: unsupported file type '{ext}'")
    return docs


# ── Q&A pairs ─────────────────────────────────────────────────────────────

def load_qa_dataset() -> list[dict]:
    """All cleaned pairs from the public agri Q&A dataset (see src/agri_qa.py)."""
    from src.agri_qa import DATASET_ID, clean_pairs, load_raw

    pairs, stats = clean_pairs(load_raw())
    logger.info(f"Q&A dataset {DATASET_ID}: {json.dumps(stats)}")
    return [{"question": p.question, "answer": p.answer, "source": DATASET_ID} for p in pairs]


def load_qa_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [{"question": r["question"], "answer": r["answer"], "source": path.name} for r in rows]


# ── Build ─────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kb-dir", default=str(KNOWLEDGE_BASE_DIR), help="documents folder, or 'none'")
    p.add_argument("--qa-dataset", action="store_true", help="add the cleaned agri Q&A dataset from the HF Hub")
    p.add_argument("--qa-jsonl", action="append", default=[], help="add Q&A pairs from a JSONL file (repeatable)")
    p.add_argument("--out", default=str(INDEX_DIR))
    p.add_argument("--embed-model", default=None, help="sentence-transformers model (default: BGE-small)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from src.retrieval.embedder import DEFAULT_EMBED_MODEL, SentenceTransformerEmbedder
    from src.retrieval.index import Chunk, KnowledgeIndex

    texts: list[tuple[str, str, dict]] = []   # (text, source, metadata)

    if args.kb_dir.lower() != "none":
        kb_dir = Path(args.kb_dir)
        logger.info(f"Loading documents from {kb_dir}/ ...")
        raw_docs = load_documents(kb_dir)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        for doc in raw_docs:
            meta = dict(doc["metadata"])
            source = meta.pop("source")
            for piece in splitter.split_text(doc["text"]):
                texts.append((piece, source, {**meta, "kind": "document"}))
        logger.info(f"Documents: {len(raw_docs)} source unit(s) → {len(texts)} chunk(s).")

    qa_rows = load_qa_dataset() if args.qa_dataset else []
    for path in args.qa_jsonl:
        qa_rows += load_qa_jsonl(Path(path))
    for row in qa_rows:
        texts.append((f"Q: {row['question']}\nA: {row['answer']}", row["source"], {"kind": "qa"}))
    if qa_rows:
        logger.info(f"Q&A pairs: {len(qa_rows)} chunk(s).")

    if not texts:
        sys.exit("Nothing indexable was found — check the warnings above.")

    chunks = [Chunk(id=i, text=t, source=s, metadata=m) for i, (t, s, m) in enumerate(texts)]
    embedder = SentenceTransformerEmbedder(args.embed_model or DEFAULT_EMBED_MODEL)
    logger.info(f"Embedding {len(chunks)} chunks with {embedder.model_name} ...")
    index = KnowledgeIndex.build(
        chunks, embedder,
        extra_manifest={"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
                        "n_qa_pairs": len(qa_rows), "n_document_chunks": len(chunks) - len(qa_rows)},
    )
    index.save(args.out)
    logger.info(f"Saved index to {args.out}/ — {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
