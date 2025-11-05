import os, json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from app.config import settings


def ensure_index():
    idx_dir = Path(settings.INDEX_DIR)
    if not (idx_dir / "index.faiss").exists():
        build_index()

# Simple loaders (add more as needed)
def load_documents(doc_dir: str) -> list[Document]:
    docs = []
    for p in Path(doc_dir).glob("**/*"):
        if p.suffix.lower() in {".pdf", ".txt", ".md", ".html"}:
            text = extract_text(p)
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": str(p)}))
    return docs

def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
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
    if settings.MODE == "local":
        from langchain_ollama import OllamaEmbeddings
        import time
        
        # Test connection and retry if needed
        max_retries = 3
        for attempt in range(max_retries):
            try:
                embeddings = OllamaEmbeddings(model=settings.OLLAMA_EMBED)
                # Test with a simple embedding
                test_result = embeddings.embed_query("test connection")
                print(f"✓ Successfully connected to Ollama embedding service")
                return embeddings
            except Exception as e:
                print(f"⚠ Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"✗ Failed to connect after {max_retries} attempts")
                    raise
    else:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=settings.OPENAI_EMBED_MODEL)

def build_index():
    print("Building index...")
    docs = load_documents(settings.DOC_DIR)
    print(f"Loaded {len(docs)} documents")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    
    embeddings = get_embedder()
    
    # Process chunks in batches to avoid overwhelming the embedding service
    batch_size = 20  # Smaller batches for better reliability
    
    print(f"Processing {len(chunks)} chunks in batches of {batch_size}")
    
    vs = None
    failed_batches = []
    successful_batches = 0
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)", end=" ")
        
        # Retry logic for each batch
        max_batch_retries = 3
        batch_success = False
        
        for retry in range(max_batch_retries):
            try:
                if vs is None:
                    # Create initial index with first batch
                    vs = FAISS.from_documents(batch, embeddings)
                else:
                    # Add subsequent batches to existing index
                    batch_vs = FAISS.from_documents(batch, embeddings)
                    vs.merge_from(batch_vs)
                
                print("✓")
                successful_batches += 1
                batch_success = True
                break
                
            except Exception as e:
                if retry < max_batch_retries - 1:
                    wait_time = 1 + retry
                    print(f"⚠ (retry {retry + 1}/{max_batch_retries} in {wait_time}s)", end=" ")
                    import time
                    time.sleep(wait_time)
                else:
                    print(f"✗ Failed after {max_batch_retries} attempts: {str(e)[:50]}...")
                    failed_batches.append((batch_num, str(e)))
        
        # Continue processing even if some batches fail
        # The index will be created from successful batches
    
    # Summary
    print(f"\n� Processing Summary:")
    print(f"   ✓ Successful batches: {successful_batches}/{total_batches}")
    if failed_batches:
        print(f"   ✗ Failed batches: {len(failed_batches)}")
        print(f"   📝 Failed batch numbers: {[b[0] for b in failed_batches]}")
        print(f"   ℹ️  Note: Failed batches are skipped but don't affect the final index")
    
    if vs is None:
        raise Exception("Failed to create any embeddings")
        
    Path(settings.INDEX_DIR).mkdir(parents=True, exist_ok=True)
    vs.save_local(settings.INDEX_DIR)
    print(f"Index built and saved to {settings.INDEX_DIR}")

if __name__ == "__main__":
    build_index()
