from app.config import settings

def call_llm(prompt: str) -> str:
    if settings.MODE == "local":
        # Ollama local
        import httpx, json
        r = httpx.post("http://localhost:11434/api/generate",
            json={"model": settings.OLLAMA_LLM, "prompt": prompt, "stream": False}, timeout=120)
        r.raise_for_status()
        return r.json()["response"]
    else:
        # OpenAI hosted
        from openai import OpenAI
        client = OpenAI()
        res = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],
            temperature=0.2,
        )
        return res.choices[0].message.content

SYSTEM = "You are a helpful assistant that answers strictly based on the provided context. If unsure, say you don't know."
