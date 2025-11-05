RAG DocQ — Intelligent Document Q&A (RAG System)

A production-ish Retrieval-Augmented Generation (RAG) app that lets you ask questions about your documents and get grounded answers with citations.
It supports local (Ollama) and hosted (OpenAI) modes, hybrid retrieval (FAISS dense + BM25), cross-encoder re-ranking, a Streamlit UI, and a FastAPI endpoint.

✨ Features

Ingestion: PDFs/HTML/MD/TXT → clean → chunk → embed → FAISS index

Hybrid retrieval: FAISS dense + BM25, then bge-reranker cross-encoder

Grounded answers: citations with source filenames

Two modes:

Local: Ollama (free) — llama3.1 + nomic-embed-text

Hosted: OpenAI (gpt-4o-mini, text-embedding-3-small)

UI & API:

Streamlit app (file upload + rebuild index + examples)

FastAPI endpoint: POST /query

Eval (optional): RAGAS metrics (faithfulness, relevancy, etc.)

🗂️ Project structure
rag-docq/
  app/
    __init__.py
    api.py
    config.py
    ingest.py
    llm.py
    rag_chain.py
    retriever.py
    ui_streamlit.py
  data/
    raw/        # put your PDFs/HTML/MD/TXT here
    index/      # FAISS index gets built here
  .streamlit/
    config.toml   # (optional) theme
  .env.example
  requirements.txt
  runtime.txt
  README.md

⚙️ Configuration

Create .env (or set env vars in your platform). Example:

# MODE: "local" (Ollama) or "hosted" (OpenAI)
MODE=local

# OpenAI (only needed in hosted mode)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small

# Paths (resolved relative to project root by app/config.py)
INDEX_DIR=data/index
DOC_DIR=data/raw

# Retrieval knobs
# (tune for long legal PDFs)
TOP_K=10
RERANK_TOP_K=5


Local mode (free) requires Ollama
 with:

brew install ollama
ollama pull llama3.1:8b
ollama pull nomic-embed-text

🚀 Quickstart (local)

Using uv (recommended):

# from project root
uv sync                 # install deps from requirements.txt (or `uv add ...` in dev)
uv run python -m app.ingest     # build FAISS index from data/raw/
uv run streamlit run app/ui_streamlit.py   # open UI on http://localhost:8501


Using pip:

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.ingest
streamlit run app/ui_streamlit.py


FastAPI (optional):

uv run uvicorn app.api:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
# Test:
# curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question":"What is in my docs?"}'

📥 Add documents

Put files into data/raw/ (PDF/HTML/MD/TXT), then either:

Rebuild via CLI:

uv run python -m app.ingest


Or from the Streamlit sidebar → Upload & Reindex → Rebuild Index.

If you want auto-ingest on first run in the cloud, you can call ensure_index() at the start of ui_streamlit.py.

🌐 Deploy (Streamlit Community Cloud)

Repo prep

uv export -o requirements.txt --no-hashes
echo "python-3.11" > runtime.txt
git add .
git commit -m "ready for Streamlit Cloud"
git push


Create app

Go to share.streamlit.io → New app

Repo: <youruser>/rag-docq

Branch: main

Main file: app/ui_streamlit.py

Secrets / Env

In Settings → Secrets:

OPENAI_API_KEY = "sk-..."
MODE = "hosted"
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_EMBED_MODEL = "text-embedding-3-small"


First run

If you committed small docs into data/raw/, it will ingest.

Otherwise upload from the UI and click Rebuild Index.

Prefer Docker/Render? Use uvicorn app.api:app in a container and (optionally) run Streamlit as a second service.

🧠 Example questions (good demo prompts)

Cloud/Dev: “Compare Lambda Function URLs and API Gateway; list 3 trade-offs.”

FinTech/RBI: “Summarize RBI’s borrower consent + data storage rules for digital lending.”

Privacy: “Define ‘personal data’ under GDPR vs India’s DPDP; one key difference.”

Agri/FAO: “Which sampling issues can bias crop yield estimates?”

🧪 Evaluation (optional)

Use RAGAS to score a small eval set (10–20 Qs):

uv run python app/eval_ragas.py


Typical metrics: faithfulness, answer_relevancy, context_precision, context_recall.
Add results to your README for credibility.

🔧 Troubleshooting

IndexError: list index out of range in FAISS
Run app.ingest from the project root so DOC_DIR resolves correctly and ensure there are documents in data/raw/.
Add PyMuPDF fallback if PDFs don’t extract text:

uv add pymupdf


Import errors on Cloud
Ensure app/ has __init__.py and imports use from app.module import ....

Model slow on first query
The cross-encoder (BAAI/bge-reranker-base) downloads once; subsequent queries are faster.

Streamlit “state key” errors
Don’t mutate st.session_state keys bound to widgets outside callbacks. The UI provided uses unique keys to clear inputs safely.

🛠️ Tech stack

Python, FAISS, LangChain

Sentence-Transformers (bge-reranker-base)

BM25 (rank-bm25)

Streamlit (UI), FastAPI (API)

Ollama (local LLM + embeddings) / OpenAI (hosted)

RAGAS (evaluation)

📜 License

MIT — see LICENSE (add one if you haven’t yet).

🙌 Acknowledgements

AWS, RBI, GDPR, FAO documents for the sample corpus

Sentence-Transformers, FAISS, and Streamlit communities

