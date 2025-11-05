from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1] # project root

class Settings(BaseSettings):
    # MODE: "local" (Ollama) or "hosted" (OpenAI)
    MODE: str = "local"

    # OpenAI
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBED_MODEL: str = "text-embedding-3-small"

    # Ollama
    OLLAMA_LLM: str = "llama3.1:8b"
    OLLAMA_EMBED: str = "nomic-embed-text"

    # Index paths
    INDEX_DIR: str = str(BASE_DIR / "data/index")
    DOC_DIR: str = str(BASE_DIR / "data/raw")

    # Retrieval knobs
    CHUNK_SIZE: int = 850
    CHUNK_OVERLAP: int = 120
    TOP_K: int = 8
    RERANK_TOP_K: int = 4

settings = Settings(_env_file=".env", _env_file_encoding="utf-8")
