# app/ingest.py
import os
import time
from pathlib import Path
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from app.config import settings


# ----------------- helpers -----------------

def ensure_index():
    idx_dir = Path(settings.INDEX_DIR)
    if not (idx_dir / "index.faiss").exists():
        build_index()


# Simple loaders (add more as needed)
def load_documents(doc_dir: str) -> List[Document]:
    docs: List[Document] = []
    for p in Path(doc_dir).glob("**/*"):
        if p.suffix.lower() in {".pdf", ".txt", ".md", ".html"}:
            text = extract_text(p)
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": str(p)}))
    return docs


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        # pypdf first (fast), you can add PyMuPDF fallback if needed
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    elif path.suffix.lower() == ".md":
        return path.read_text(encoding="utf-8", errors="ignore")
    elif path.suffix.lower() == ".html":
        from bs4 import BeautifulSoup
        return BeautifulSoup(path.read_text("utf-8", "ignore"), "html.parser").get_text(" ")
    else:
        return path.read_text(encoding="utf-8", errors="ignore")


def get_embedder():
    if settings.EMBEDDINGS_BACKEND == "ollama":
        # Ollama embeddings
        from langchain_ollama import OllamaEmbeddings
        # quick connectivity probe with light retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                emb = OllamaEmbeddings(model=settings.OLLAMA_EMBED)
                _ = emb.embed_query("ping")
                print("✓ Ollama embeddings ready")
                return emb
            except Exception as e:
                print(f"⚠ Ollama connect {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1 + attempt)
                else:
                    raise
    if settings.EMBEDDINGS_BACKEND == "hf":
        from langchain_huggingface.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=settings.HF_EMBED_MODEL)
    # default: OpenAI
    else:
        # OpenAI embeddings (hosted)
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=settings.OPENAI_EMBED_MODEL)


def _filter_chunks(chunks: List[Document], min_chars: int = 200) -> List[Document]:
    """Drop very short chunks and exact duplicates to save tokens/QPS."""
    seen = set()
    keep: List[Document] = []
    for d in chunks:
        txt = (d.page_content or "").strip()
        if len(txt) < min_chars:
            continue
        sig = (d.metadata.get("source", ""), txt[:160])
        if sig in seen:
            continue
        seen.add(sig)
        keep.append(d)
    return keep


def _embed_texts_throttled(
    texts: List[str],
    emb,
    batch_size: int = 5,
    sleep_between: float = 0.6,
    max_retries: int = 6,
) -> Tuple[List[List[float]], List[int]]:
    """
    Embed texts in tiny batches with exponential backoff.
    Returns (vectors, ok_indices) so we can drop failed items safely.
    """
    vectors: List[List[float]] = []
    ok_indices: List[int] = []
    total = len(texts)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = texts[start:end]
        attempt = 0
        while True:
            try:
                vecs = emb.embed_documents(batch)
                vectors.extend(vecs)
                ok_indices.extend(range(start, end))
                if sleep_between > 0:
                    time.sleep(sleep_between)  # be gentle with rate limits
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    print(f"✗ Batch {start//batch_size+1} failed permanently: {str(e)[:80]}...")
                    # skip this batch and continue; we won't index those items
                    break
                delay = min(8.0, 1.0 * (2 ** (attempt - 1)))
                print(f"⚠ 429/Transient error on batch {start//batch_size+1}, retry {attempt}/{max_retries} in {delay:.1f}s")
                time.sleep(delay)

    return vectors, ok_indices


# ----------------- main build -----------------

# app/ingest.py

def _emit(progress, msg):
    try:
        if progress:
            progress({"msg": msg})
    except Exception:
        pass

def build_index(progress=None, limit_chunks: int = 0, batch_size: int = 5, sleep_between: float = 0.6):
    _emit(progress, "Building index…")
    docs = load_documents(settings.DOC_DIR)
    _emit(progress, f"Loaded {len(docs)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    _emit(progress, f"Split into {len(chunks)} chunks")

    chunks = _filter_chunks(chunks, min_chars=200)
    _emit(progress, f"Kept {len(chunks)} chunks after filtering (<200 chars dropped, dups removed)")

    if limit_chunks and limit_chunks > 0:
        chunks = chunks[:limit_chunks]
        _emit(progress, f"Limiting to first {len(chunks)} chunks")

    if not chunks:
        raise RuntimeError("No usable chunks to index. Add documents to data/raw/ and try again.")

    emb = get_embedder()
    texts = [d.page_content for d in chunks]
    metas = [d.metadata for d in chunks]

    _emit(progress, f"Embedding {len(texts)} chunks (batch={batch_size}, sleep={sleep_between}s)…")
    vecs, ok_idx = _embed_texts_throttled(
        texts, emb, batch_size=batch_size, sleep_between=sleep_between, max_retries=6
    )

    if not ok_idx:
        raise RuntimeError("Failed to create embeddings (rate-limited or API error). Try a smaller N or prebuild locally.")

    ok_texts = [texts[i] for i in ok_idx]
    ok_metas = [metas[i] for i in ok_idx]
    pairs = list(zip(ok_texts, vecs))

    _emit(progress, "Building FAISS index…")
    vs = FAISS.from_embeddings(pairs, emb, metadatas=ok_metas)

    Path(settings.INDEX_DIR).mkdir(parents=True, exist_ok=True)
    vs.save_local(settings.INDEX_DIR)
    _emit(progress, f"✅ Index built and saved to {settings.INDEX_DIR} (indexed {len(ok_texts)}/{len(texts)})")

if __name__ == "__main__":
    build_index()
