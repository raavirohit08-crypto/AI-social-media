"""Automatic checks for the real-model evaluation (pure Python, easy to test offline).

What is checked automatically:
  * platform format  - character limit, hashtag range, 'Visual idea:' line, video-script structure
  * factual grounding - every number/price in the post must appear in the brand documents,
                        brand profile or the request; required facts present; forbidden claims absent
  * variation quality - the right number of versions and clearly different from each other
  * tone proxies      - sentence length, emoji count, brand 'avoid' words. These are only PROXIES:
                        a person must still rate tone consistency (1-5) in the report.
"""
import re
from itertools import combinations

from src.brand import avoid_terms_from_text, find_avoided_terms
from src.platforms import PLATFORMS, analyze_post, extract_hashtags, split_visual_idea

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> set[str]:
    return {m.group(0).replace(",", "").rstrip(".") for m in _NUMBER.finditer(text or "")}


def check_format(text: str, platform: str) -> list[tuple[str, bool, str]]:
    spec = PLATFORMS[platform]
    stats = analyze_post(text, platform)
    results = []
    if spec.limit and spec.limit < 10000:
        results.append(("character limit", not stats["over_limit"], f"{stats['chars']} / {spec.limit}"))
    low, high = spec.hashtags
    ok = low <= stats["hashtags"] <= high if not spec.is_script else stats["hashtags"] <= high
    results.append(("hashtag count", ok, f"{stats['hashtags']} (target {low}-{high})"))
    if platform == "Instagram":
        _, visual = split_visual_idea(text)
        results.append(("has 'Visual idea:' line", bool(visual), ""))
    if spec.is_script:
        for part in ("HOOK", "SCENE", "CTA", "CAPTION"):
            results.append((f"script has {part}", part.lower() in text.lower(), ""))
    return results


def check_facts(text: str, allowed_text: str, case: dict) -> list[tuple[str, bool, str]]:
    results = []
    body = re.sub(r"(?<!\w)#\w+", "", split_visual_idea(text)[0])        # numbers inside hashtags do not count
    unsupported = sorted(numbers_in(body) - numbers_in(allowed_text))
    results.append(("every number appears in brand docs / request", not unsupported,
                    "unsupported: " + ", ".join(unsupported) if unsupported else ""))
    low = text.lower()
    for needle in case.get("must_include", []):
        results.append((f"mentions '{needle}'", needle.lower() in low, ""))
    anyof = case.get("must_include_any", [])
    if anyof:
        results.append((f"mentions one of {anyof}", any(n.lower() in low for n in anyof), ""))
    for needle in case.get("must_not_include", []):
        results.append((f"does NOT say '{needle}'", needle.lower() not in low, ""))
    for pattern in case.get("must_not_match", []):
        results.append((f"no match for /{pattern}/", re.search(pattern, text) is None, ""))
    return results


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def check_variations(texts: list[str], expected: int) -> list[tuple[str, bool, str]]:
    results = [("number of versions", len(texts) == expected, f"{len(texts)} of {expected}")]
    if expected > 1 and len(texts) > 1:
        worst = 0.0
        for a, b in combinations(texts, 2):
            wa, wb = _words(a), _words(b)
            worst = max(worst, len(wa & wb) / max(1, len(wa | wb)))
        results.append(("versions are distinct", worst < 0.7, f"max word overlap {worst:.0%}"))
    return results


def tone_proxies(text: str, profile_text: str) -> dict:
    body = split_visual_idea(text)[0]
    sentences = [s for s in re.split(r"[.!?\n]+", re.sub(r"(?<!\w)#\w+", "", body)) if s.strip()]
    avg = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
    avoided = find_avoided_terms(text, avoid_terms_from_text(profile_text))
    return {"avg_sentence_words": round(avg, 1), "emojis": analyze_post(text, "Facebook")["emojis"],
            "avoided_words_used": avoided, "avoid_check_passed": not avoided}


def evaluate_post(text: str, platform: str, allowed_text: str, case: dict, profile_text: str) -> dict:
    checks = check_format(text, platform) + check_facts(text, allowed_text, case)
    tone = tone_proxies(text, profile_text)
    checks.append(("no brand 'avoid' words", tone["avoid_check_passed"], ", ".join(tone["avoided_words_used"])))
    return {"checks": [{"name": n, "passed": ok, "detail": d} for n, ok, d in checks],
            "all_passed": all(ok for _, ok, _ in checks), "tone_proxies": tone}
