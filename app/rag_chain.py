from app.retriever import HybridRetriever
from app.llm import call_llm
from textwrap import dedent

RAG_TEMPLATE = """\
Answer the question using ONLY the context. Cite sources at the end as [n].
If answer is not in context, say "Sorry, I am unable give an answer to your query."

Question: {question}

Context:
{context}

Respond:"""

_retriever = None
def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever

def answer(question: str):
    retriever = _get_retriever()
    docs = retriever.retrieve(question, k=8, rerank_k=4)
    ctx = "\n\n".join([f"[{i+1}] {d.page_content[:1200]}\n(Source: {d.metadata.get('source','')})" for i,d in enumerate(docs)])
    prompt = RAG_TEMPLATE.format(question=question, context=ctx)
    out = call_llm(prompt)
    return out, docs

def reset_retriever():
    global _retriever
    _retriever = None
