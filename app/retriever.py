from langchain_community.vectorstores import FAISS
from app.config import settings
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

# Load FAISS index
def load_vector_store():
    if settings.EMBEDDINGS_BACKEND == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings as OE
        emb = OE(model=settings.OLLAMA_EMBED)
    elif settings.EMBEDDINGS_BACKEND == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name=settings.HF_EMBED_MODEL)
    else:
        from langchain_openai import OpenAIEmbeddings
        emb = OpenAIEmbeddings(model=settings.OPENAI_EMBED_MODEL)

    from langchain_community.vectorstores import FAISS
    return FAISS.load_local(settings.INDEX_DIR, emb, allow_dangerous_deserialization=True)


# Optional: BM25 on the same chunks (kept in memory)
def corpus_from_vs(vs):
    # access the text of documents
    texts = [doc.page_content for doc in vs.docstore._dict.values()]
    return texts

class HybridRetriever:
    def __init__(self):
        self.vs = load_vector_store()
        self.docs_list = list(self.vs.docstore._dict.values())
        self.corpus = corpus_from_vs(self.vs)
        self.bm25 = BM25Okapi([t.split() for t in self.corpus])
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")

    def retrieve(self, query: str, k=8, rerank_k=4):
        # (1) Dense
        d_hits = self.vs.similarity_search(query, k=k)
        # (2) BM25
        bm_scores = self.bm25.get_scores(query.split())
        bm_top_idx = sorted(range(len(bm_scores)), key=lambda i: bm_scores[i], reverse=True)[:k]
        bm_hits = [self.docs_list[i] for i in bm_top_idx]
        # Merge
        pool = d_hits + bm_hits
        # (3) Re-rank
        pairs = [(query, d.page_content) for d in pool]
        scores = self.reranker.predict(pairs)
        ranked = [doc for _, doc in sorted(zip(scores, pool), key=lambda t: t[0], reverse=True)]
        # Deduplicate by content id
        seen, final = set(), []
        for d in ranked:
            sid = (d.metadata.get("source",""), d.page_content[:120])
            if sid not in seen:
                final.append(d)
                seen.add(sid)
            if len(final) >= rerank_k:
                break
        return final
