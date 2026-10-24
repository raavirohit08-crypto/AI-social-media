"""Content calendar: ask the model for a JSON plan, then validate it in Python."""
import json
import re
from datetime import date, timedelta

from .llm import LLM
from .platforms import PLATFORMS

SYSTEM_PROMPT = (
    "You are a social media strategist. You plan content calendars. "
    "Use brand facts ONLY from the context given; never invent offers, statistics or events. "
    "You reply with valid JSON only."
)

COLUMNS = ["date", "day", "platform", "format", "idea", "hook", "hashtags", "cta"]


def parse_json_items(text: str) -> list[dict]:
    """Pull a list of JSON objects out of model text.

    Handles ```json fences, extra chatter, and a reply that was cut off
    half-way (we then keep every complete {...} object found).
    """
    text = re.sub(r"```(?:json)?", "", text or "").strip()
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except json.JSONDecodeError:
            pass
    items = []
    for chunk in re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL):
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                items.append(obj)
        except json.JSONDecodeError:
            continue
    return items


def _clean_hashtags(value) -> str:
    if isinstance(value, str):
        value = re.findall(r"#?\w+", value)
    tags = [("#" + str(t).lstrip("#")) for t in (value or []) if str(t).strip("# ")]
    return " ".join(tags[:8])


def normalize_plan(items: list[dict], start: date, days: int, platforms: list[str]) -> list[dict]:
    """Keep only valid rows, compute real dates in Python, sort by date."""
    lookup = {p.lower(): p for p in platforms}
    rows = []
    for item in items:
        try:
            day = int(item.get("day"))
        except (TypeError, ValueError):
            continue
        platform = lookup.get(str(item.get("platform", "")).strip().lower())
        idea = str(item.get("idea", "")).strip()
        if not (1 <= day <= days) or not platform or not idea:
            continue
        when = start + timedelta(days=day - 1)
        rows.append({
            "date": when.isoformat(), "day": when.strftime("%A"), "platform": platform,
            "format": str(item.get("format", "")).strip(), "idea": idea,
            "hook": str(item.get("hook", "")).strip(),
            "hashtags": _clean_hashtags(item.get("hashtags")),
            "cta": str(item.get("cta", "")).strip(),
        })
    return sorted(rows, key=lambda r: (r["date"], r["platform"]))


def plan_calendar(llm: LLM, goal: str, platforms: list[str], days: int = 7, posts_per_week: int = 4,
                  start: date | None = None, theme: str = "", tone: str = "Friendly",
                  audience: str = "", profile_text: str = "", brand_context: str = "") -> list[dict]:
    if not platforms:
        raise ValueError("Choose at least one platform.")
    for p in platforms:
        if p not in PLATFORMS:
            raise ValueError(f"Unknown platform: {p}")
    start = start or date.today()
    total = max(1, round(days / 7 * posts_per_week))

    prompt = (
        (f"BRAND PROFILE:\n{profile_text}\n\n" if profile_text else "")
        + (f"BRAND CONTEXT:\n{brand_context}\n\n" if brand_context else "")
        + f"Plan a {days}-day social media content calendar with exactly {total} posts.\n"
        f"Platforms to use: {', '.join(platforms)} (spread posts across them).\n"
        f"Goal: {goal}\nTheme / focus: {theme or 'general brand content'}\n"
        f"Tone: {tone}\nAudience: {audience or 'general audience'}\n"
        "Mix post types (education, behind the scenes, social proof only if given in the context, "
        "promotion, engagement questions). Do not put more than 2 posts on one day.\n\n"
        "Return ONLY a JSON array. Each element must have exactly these keys:\n"
        '{"day": 1, "platform": "<one of the platforms>", "format": "e.g. text post / carousel / reel", '
        '"idea": "one sentence", "hook": "first line of the post", "hashtags": ["#tag1", "#tag2"], '
        '"cta": "short call to action"}\n'
        f'"day" is a number from 1 to {days}.'
    )
    rows = []
    for attempt in range(2):
        text = llm.ask(SYSTEM_PROMPT, prompt if attempt == 0 else
                       prompt + "\n\nYour previous answer was not valid JSON. Reply with ONLY the JSON array.",
                       max_tokens=2500, temperature=0.7)
        rows = normalize_plan(parse_json_items(text), start, days, platforms)
        if rows:
            break
    if not rows:
        raise ValueError("The model did not return a usable calendar. Please try again.")
    return rows
