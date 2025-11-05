# --- path safety for Streamlit Cloud ---
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Streamlit Cloud: bridge secrets -> env vars ---
try:
    import streamlit as st  # available on Cloud
    for k in ("OPENAI_API_KEY", "MODE", "OPENAI_MODEL", "OPENAI_EMBED_MODEL",
              "EMBEDDINGS_BACKEND", "HF_EMBED_MODEL"):
        v = st.secrets.get(k) if hasattr(st, "secrets") else None
        if v and not os.getenv(k):
            os.environ[k] = str(st.secrets[k])
except Exception:
    pass
# ---------------------------------------------------

import streamlit as st
from typing import List, Dict
from app.rag_chain import answer, reset_retriever
from app.config import settings
from app import ingest
from pathlib import Path
import time

# ---------- Page & Styles ----------
st.set_page_config(page_title="RAG DocQ", page_icon="📄", layout="wide", initial_sidebar_state="expanded")

CUSTOM_CSS = """
<style>
/* tighten spacing */
.block-container { padding-top: 3rem; padding-bottom: 2rem; }
section[data-testid="stSidebar"] {width: 320px !important; }
div[data-testid="stToolbar"] { display: inherit; }

/* source card */
.source-card {
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 8px; 
    padding: 10px; 
    margin-bottom: 8px;
    background: rgba(128, 128, 128, 0.06);
}
.source-title { font-weight: 600; font-size: 0.93rem; margin-bottom: 6px; }
.source-snippet { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem; white-space: pre-wrap; }

/* answer bubble */
.answer {
    border-left: 4px solid #7aa2f7; 
    padding: 12px 12px 8px 12px; 
    border-radius: 6px;
    background: rgba(128, 128, 128, 0.08);
}

/* subtle code styling */
code, pre code { font-size: 0.85rem !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------- Session State ----------
if "history" not in st.session_state:
    st.session_state.history: List[Dict] = []  # [{q, a, sources}]
if "q" not in st.session_state:
    st.session_state.q = ""
if "is_indexing" not in st.session_state:
    st.session_state.is_indexing = False

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("Switch between local (Ollama) and hosted (OpenAI) in your `.env`.")

    st.write(f"**Mode:** `{settings.MODE}`")
    st.write(f"**Doc dir:** `{settings.DOC_DIR}`")
    st.write(f"**Index dir:** `{settings.INDEX_DIR}`")

    st.divider()
    st.subheader("📤 Upload & Reindex")

    uploaded = st.file_uploader(
        "Add PDF/Markdown/HTML/TXT",
        type=["pdf","md","html","txt"],
        accept_multiple_files=True,
        help="Drop one or more files to include in your document corpus. They will be stored in the Doc dir shown above."
    )

    # Smart defaults based on embedding backend
    backend = os.getenv("EMBEDDINGS_BACKEND", "hf").lower()
    if backend == "hf":
        default_batch = 8
        default_sleep = 0.0
        default_limit = 100
    else:  # openai or other HTTP API
        default_batch = 4
        default_sleep = 1.0
        default_limit = 100

    limit_chunks = st.number_input(
        "Index only first N chunks (0 = all)",
        min_value=0, max_value=5000, value=default_limit, step=20,
        help="How many chunks to embed/index this run. 0 = all. Use 100–200 first to smoke-test, then set to 0."
    )

    batch_size = st.slider(
        "Embedding batch size",
        min_value=1, max_value=16, value=default_batch,
        help="How many chunks to embed at once. Higher = faster but more CPU/RAM. Lower = safer on small machines."
    )

    sleep_between = st.slider(
        "Sleep between batches (sec)",
        min_value=0.0, max_value=2.0, value=default_sleep, step=0.1,
        help="Pause after each batch. Needed only for API rate limits (e.g., OpenAI). For HuggingFace (local), set to 0.0."
    )

    with st.expander("⚙️ Advanced tips", expanded=False):
        st.markdown("""
    - **HuggingFace (local) embeddings**: Batch **8–10**, Sleep **0.0**.
    - **OpenAI embeddings**: Batch **3–5**, Sleep **0.8–1.2s**; start with **100–200** chunks.
    - Seeing **429 rate limits**? Lower batch or increase sleep.
    - Switched embedding backend? **Rebuild the index** so dimensions match.
    - Big PDFs → lots of chunks. Try indexing **first 100** to smoke-test, then set to **0** (all).
        """)

    # Single-ingest lock + progress
    if st.button(
        "Rebuild Index",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.get("is_indexing", False),
    ):
        if st.session_state.get("is_indexing", False):
            st.warning("Indexing is already running…")
        else:
            st.session_state.is_indexing = True
            try:
                doc_dir = Path(settings.DOC_DIR)
                doc_dir.mkdir(parents=True, exist_ok=True)

                # Save any newly uploaded files
                if uploaded:
                    for f in uploaded:
                        (doc_dir / f.name).write_bytes(f.getbuffer())

                # Live status panel with log updates from ingest
                with st.status("Indexing…", expanded=True) as status:
                    log_area = st.empty()
                    lines = []

                    def _progress(evt: dict):
                        # evt is {"msg": "..."} from ingest
                        lines.append(evt.get("msg", ""))
                        log_area.text("\n".join(lines[-40:]))

                    ingest.build_index(
                        progress=_progress,
                        limit_chunks=int(limit_chunks),
                        batch_size=int(batch_size),
                        sleep_between=float(sleep_between),
                    )
                    status.update(label="Index built ✅", state="complete", expanded=False)

                reset_retriever()
                time.sleep(0.2)
                st.success("Index rebuilt and retriever reloaded.")
            except Exception as e:
                st.error(f"Indexing failed: {e}")
            finally:
                st.session_state.is_indexing = False

    st.divider()
    st.subheader("🧪 Example questions")
    for ex in [
        "Give me a 3-bullet summary of the main document.",
        "What are the key steps mentioned for deployment?",
        "List definitions and acronyms present in the docs.",
        "Which file discusses limitations or caveats?",
    ]:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:12]}"):
            st.session_state.q = ex

    st.divider()

tabs = st.tabs(["💬 Ask", "📈 Evaluate"])
with tabs[0]:
    # (move your existing ask UI into this tab)
    # ---------- Main Layout ----------
    col_left, col_right = st.columns([7,5], gap="large")

    with col_left:
        st.title("📄 RAG Document Query")
        st.caption("Ask questions about your indexed documents. Answers include citations.")

        q = st.text_area("Your question", key="q", height=90, placeholder="e.g., What does the document talk about?")
        go = st.button("Ask", type="primary")

        if go and q.strip():
            with st.spinner("Thinking…"):
                ans, docs = answer(q.strip())
            st.session_state.history.insert(0, {
                "q": q.strip(),
                "a": ans,
                "sources": [{"source": d.metadata.get("source",""), "text": d.page_content} for d in docs]
            })
            # Rerun to refresh the app and clear the input
            st.rerun()

        # Chat history render
        for item in st.session_state.history:
            st.markdown(f"#### ❓ {item['q']}")
            st.markdown(f'<div class="answer">{item["a"]}</div>', unsafe_allow_html=True)
            with st.expander("Show citations", expanded=False):
                for i, s in enumerate(item["sources"], start=1):
                    src_name = s["source"] or "(unknown source)"
                    st.markdown(f"**[{i}] {src_name}**")
                    # Truncate text to approximately 8 lines of 120 characters each
                    text_clean = s["text"].replace('\r', ' ').replace('\n', ' ')
                    max_chars = 120 * 8  # Approximate 8 lines
                    if len(text_clean) > max_chars:
                        text_clean = text_clean[:max_chars] + "…"
                    st.code(text_clean)
            st.markdown("---")

    with col_right:
        st.subheader("📚 Sources (latest answer)")
        if st.session_state.history:
            latest = st.session_state.history[0]
            if latest["sources"]:
                for i, s in enumerate(latest["sources"], start=1):
                    with st.container():
                        st.markdown(f'<div class="source-card">', unsafe_allow_html=True)
                        st.markdown(f'<div class="source-title">[{i}] {s["source"] or "(unknown)"} </div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="source-snippet">{s["text"][:800]}</div>', unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.button(f"Copy [{i}] snippet", key=f"copy_{i}", on_click=st.session_state.setdefault, args=("copied", True))
            else:
                st.info("Run a query to see sources here.")
        else:
            st.info("No queries yet. Ask a question to populate sources.")

        st.divider()
        st.subheader("📈 Session metrics")
        # Simple counters; you can wire real latency/usage later
        total_q = len(st.session_state.history)
        st.metric("Questions asked", total_q)
        unique_sources = len({s["source"] for item in st.session_state.history for s in item["sources"]})
        st.metric("Unique sources cited", unique_sources)
    
with tabs[1]:
    st.subheader("RAG Evaluation")
    st.caption("Runs a small benchmark and reports retrieval + RAGAS metrics.")
    eval_path = st.text_input("Eval file (JSONL)", "data/eval.jsonl")
    if st.button("Run evaluation", type="secondary"):
        with st.status("Evaluating…", expanded=True) as status:
            from app.eval_ragas import load_eval_set, run_batch_eval, save_summary
            items = load_eval_set(eval_path)
            st.write(f"Loaded {len(items)} items")
            rep = run_batch_eval(items)
            save_summary(rep)
            status.update(label="Evaluation complete ✅", state="complete", expanded=False)

    # show latest summary if present
    from pathlib import Path
    summ = Path("data/eval/summary.json")
    if summ.exists():
        import json
        st.markdown("### Latest results")
        data = json.loads(summ.read_text())
        cols = st.columns(4)
        cols[0].metric("N (questions)", data.get("n", 0))
        cols[1].metric("Recall@k", f'{(data.get("recall_at_k") or 0)*100:.1f}%')
        cols[2].metric("Latency p50", f'{data.get("lat_p50",0):.2f}s')
        cols[3].metric("Latency p95", f'{data.get("lat_p95",0):.2f}s')

        st.markdown("#### RAGAS (higher is better)")
        ragas_means = data.get("ragas_means", {})
        for k, v in ragas_means.items():
            st.write(f"- **{k}**: {v:.3f}")
    else:
        st.info("No evaluation summary found yet. Add `data/eval.jsonl` and run.")

