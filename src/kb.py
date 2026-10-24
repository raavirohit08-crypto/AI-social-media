"""Brand knowledge base: Knowledge/ folder -> chunks -> FAISS -> vector_db/.

The index is saved to disk, so documents are embedded only once. A manifest
(file name + size + modified time) detects added/removed/changed files and the
index is rebuilt automatically.
"""
import json
from pathlib import Path

from .config import Settings
from .loader import SUPPORTED, load_file
from .vector_store import build_vector_store, load_store, save_store, split_documents

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "Knowledge"
VECTOR_DIR = ROOT / "vector_db"


def list_files(folder: Path = KNOWLEDGE_DIR) -> list[Path]:
    folder.mkdir(exist_ok=True)
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED),
                  key=lambda p: p.name.lower())


def fingerprint(folder: Path = KNOWLEDGE_DIR) -> dict:
    return {p.name: [p.stat().st_size, int(p.stat().st_mtime)] for p in list_files(folder)}


def is_index_current(knowledge: Path = KNOWLEDGE_DIR, vector: Path = VECTOR_DIR) -> bool:
    manifest = vector / "manifest.json"
    if not (vector / "index.faiss").exists() or not manifest.exists():
        return False
    try:
        return json.loads(manifest.read_text()) == fingerprint(knowledge)
    except Exception:
        return False


def load_documents(folder: Path = KNOWLEDGE_DIR) -> tuple[dict, list[str]]:
    """Return ({file name: [Documents]}, [names of unreadable files])."""
    loaded, skipped = {}, []
    for path in list_files(folder):
        try:
            docs = load_file(str(path), path.name)
        except Exception:
            docs = []
        if docs:
            loaded[path.name] = docs
        else:
            skipped.append(path.name)
    return loaded, skipped


def build_or_load(embeddings, settings: Settings, force: bool = False,
                  knowledge: Path = KNOWLEDGE_DIR, vector: Path = VECTOR_DIR):
    """Return (store, loaded_files, skipped). Rebuilds only when needed."""
    loaded, skipped = load_documents(knowledge)
    if not loaded:
        return None, {}, skipped
    if not force and is_index_current(knowledge, vector):
        return load_store(embeddings, str(vector)), loaded, skipped
    chunks = []
    for docs in loaded.values():
        chunks += split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    store = build_vector_store(chunks, embeddings)
    vector.mkdir(exist_ok=True)
    save_store(store, str(vector))
    (vector / "manifest.json").write_text(json.dumps(fingerprint(knowledge)))
    return store, loaded, skipped
