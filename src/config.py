"""Central configuration. Values come from the environment / .env file."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "openai/gpt-oss-120b"


@dataclass
class Settings:
    hf_token: str | None = None
    model_name: str = DEFAULT_MODEL
    # Brand documents are short, so small chunks give precise retrieval
    chunk_size: int = 700
    chunk_overlap: int = 100
    top_k: int = 4


def load_settings() -> Settings:
    return Settings(
        hf_token=os.getenv("HF_TOKEN"),
        model_name=os.getenv("MODEL_NAME", DEFAULT_MODEL),
    )
