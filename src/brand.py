"""Brand profile (voice, audience, words to avoid) + retrieval of brand facts."""
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROFILE_FILE = ROOT / "brand_profile.json"

PROFILE_FIELDS = {
    "name": "Brand / business name",
    "industry": "Industry",
    "about": "What you do (1-3 sentences)",
    "audience": "Default target audience",
    "voice": "Brand voice (e.g. warm, witty, no jargon)",
    "avoid": "Words / topics to avoid",
}


def load_profile(path: Path = PROFILE_FILE) -> dict:
    profile = {key: "" for key in PROFILE_FIELDS}
    try:
        profile.update({k: str(v) for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items()
                        if k in PROFILE_FIELDS})
    except Exception:
        pass
    return profile


def save_profile(profile: dict, path: Path = PROFILE_FILE) -> None:
    Path(path).write_text(json.dumps({k: profile.get(k, "") for k in PROFILE_FIELDS}, indent=2),
                          encoding="utf-8")


def profile_to_text(profile: dict) -> str:
    """Turn the profile into a prompt block; empty string if nothing is filled in."""
    lines = [f"- {label}: {profile[key].strip()}"
             for key, label in PROFILE_FIELDS.items() if profile.get(key, "").strip()]
    return "\n".join(lines)


@dataclass
class Passage:
    source: str
    text: str
    score: float  # cosine similarity (higher = more related)


def get_brand_context(store, query: str, k: int = 4) -> tuple[str, list[Passage]]:
    """Retrieve the brand-document passages most related to the query."""
    if store is None or not query.strip():
        return "", []
    passages = []
    for doc, distance in store.similarity_search_with_score(query, k=k):
        # embeddings are normalized, so cosine = 1 - squared L2 distance / 2
        passages.append(Passage(doc.metadata.get("source_name", "Unknown"), doc.page_content,
                                round(1 - float(distance) / 2, 3)))
    context = "\n\n".join(f"[{i}] (from {p.source})\n{p.text}" for i, p in enumerate(passages, 1))
    return context, passages
