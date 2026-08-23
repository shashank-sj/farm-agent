"""
Build the FAISS index used by farm_rag from the documents in data/knowledge_base/.

This is an offline/local step, not something the app runs at serve time — the
RAG tool (src/tools/rag_tool.py) only ever *loads* data/faiss_index/, it never
builds it. That's why the dependencies here live in requirements-ingest.txt
instead of the main requirements.txt: the HF Space serving this app never
needs pdfplumber or an OCR engine, only whoever rebuilds the index does.

Usage:
    pip install -r requirements-ingest.txt
    python scripts/build_rag_index.py

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

import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("build-rag-index")

KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")
INDEX_DIR = Path("data/faiss_index")

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
        sys.exit(f"{directory}/ doesn't exist — create it and add some documents first.")

    files = sorted(p for p in directory.rglob("*") if p.is_file())
    if not files:
        sys.exit(f"No files found in {directory}/ — add some documents first.")

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


# ── Build ─────────────────────────────────────────────────────────────────

def main():
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        sys.exit("GEMINI_API_KEY not set — needed to embed the chunks (add it to .env).")
    os.environ["GOOGLE_API_KEY"] = gemini_key

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_community.vectorstores import FAISS

    logger.info(f"Loading documents from {KNOWLEDGE_BASE_DIR}/ ...")
    raw_docs = load_documents(KNOWLEDGE_BASE_DIR)
    if not raw_docs:
        sys.exit("Nothing indexable was extracted — check the warnings above.")
    logger.info(f"Loaded {len(raw_docs)} source unit(s) (pages/files).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = [
        Document(page_content=piece, metadata=doc["metadata"])
        for doc in raw_docs
        for piece in splitter.split_text(doc["text"])
    ]
    logger.info(f"Split into {len(chunks)} chunk(s) (~{CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap).")

    logger.info("Embedding chunks and building the FAISS index (calls the Gemini embeddings API)...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    logger.info(f"Saved index to {INDEX_DIR}/ — {len(chunks)} chunks from {len(raw_docs)} source unit(s).")


if __name__ == "__main__":
    main()
