from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_chain import answer

app = FastAPI(title="RAG DocQ")

class Q(BaseModel):
    question: str

@app.post("/query")
def query(q: Q):
    ans, docs = answer(q.question)
    return {
        "answer": ans,
        "sources": [{"source": d.metadata.get("source",""), "snippet": d.page_content[:300]} for d in docs]
    }
