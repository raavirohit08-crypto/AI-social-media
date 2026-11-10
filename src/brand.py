"""Brand profile (voice, audience, words to avoid) + retrieval of brand facts."""
import json
import re
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


# ------------------------------------------------------------------ brand restrictions
def avoid_terms_from_text(profile_text: str) -> list[str]:
    """Pull the short 'words / topics to avoid' entries out of the profile prompt block.

    Only short comma / semicolon / line separated entries are used (e.g. "cheap, guaranteed"),
    because long sentences describe topics and cannot be matched word for word.
    """
    label = PROFILE_FIELDS["avoid"]
    terms = []
    for line in (profile_text or "").splitlines():
        if line.strip().lower().startswith(f"- {label}:".lower()):
            value = line.split(":", 1)[1]
            for part in re.split(r"[,;\n]|\band\b|\bor\b", value):
                term = part.strip(" .\"'`-*")
                if 2 <= len(term) <= 30 and len(term.split()) <= 3:
                    terms.append(term)
    return list(dict.fromkeys(terms))


def find_avoided_terms(text: str, terms: list[str]) -> list[str]:
    """Return the avoided terms that appear in the text (whole words, any case)."""
    found = []
    for term in terms:
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text or "", flags=re.IGNORECASE):
            found.append(term)
    return found
