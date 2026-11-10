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

MAX_POSTS_PER_DAY = 2
ATTEMPTS = 3

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


class IncompleteCalendarError(ValueError):
    """The model never returned a complete calendar. `rows` holds the best partial result."""
    def __init__(self, message: str, rows: list[dict] | None = None):
        super().__init__(message)
        self.rows = rows or []


def _key(text: str) -> str:
    return re.sub(r"\W+", " ", str(text).lower()).strip()


def clean_plan(rows: list[dict], max_per_day: int = MAX_POSTS_PER_DAY) -> list[dict]:
    """Remove duplicates and enforce the maximum posts per day (rows must be sorted by date).

    Duplicate = same idea text, or same date + platform + hook.
    """
    seen_ideas, seen_hooks, per_day, out = set(), set(), {}, []
    for row in rows:
        idea = _key(row["idea"])
        hook = (row["date"], row["platform"], _key(row["hook"])) if row["hook"] else None
        if idea in seen_ideas or (hook and hook in seen_hooks):
            continue
        if per_day.get(row["date"], 0) >= max_per_day:
            continue
        seen_ideas.add(idea)
        if hook:
            seen_hooks.add(hook)
        per_day[row["date"]] = per_day.get(row["date"], 0) + 1
        out.append(row)
    return out


def plan_problems(raw_count: int, rows: list[dict], total: int) -> list[str]:
    """Explain, in words the model can act on, why a calendar is not complete."""
    problems = []
    if len(rows) < total:
        problems.append(f"only {len(rows)} valid, unique posts were returned but exactly {total} are required")
    if raw_count > len(rows) and len(rows) < total:
        problems.append("some entries were invalid, duplicated, or put more than "
                        f"{MAX_POSTS_PER_DAY} posts on one day")
    return problems


def plan_calendar(llm: LLM, goal: str, platforms: list[str], days: int = 7, posts_per_week: int = 4,
                  start: date | None = None, theme: str = "", tone: str = "Friendly",
                  audience: str = "", profile_text: str = "", brand_context: str = "") -> list[dict]:
    """Return a COMPLETE calendar (exactly the requested number of posts) or raise.

    Each model answer is checked in Python: valid rows, no duplicates, at most
    MAX_POSTS_PER_DAY per day, and the right total. Incomplete answers are retried with a
    note saying what was wrong; if every try is incomplete an IncompleteCalendarError is raised
    (carrying the best partial rows) instead of showing a partial plan as if it were complete.
    """
    if not platforms:
        raise ValueError("Choose at least one platform.")
    for p in platforms:
        if p not in PLATFORMS:
            raise ValueError(f"Unknown platform: {p}")
    start = start or date.today()
    total = max(1, round(days / 7 * posts_per_week))
    if total > days * MAX_POSTS_PER_DAY:
        raise ValueError(f"{total} posts do not fit in {days} days with at most "
                         f"{MAX_POSTS_PER_DAY} posts per day. Lower the posts per week.")

    prompt = (
        (f"BRAND PROFILE:\n{profile_text}\n\n" if profile_text else "")
        + (f"BRAND CONTEXT:\n{brand_context}\n\n" if brand_context else "")
        + f"Plan a {days}-day social media content calendar with exactly {total} posts.\n"
        f"Platforms to use: {', '.join(platforms)} (spread posts across them).\n"
        f"Goal: {goal}\nTheme / focus: {theme or 'general brand content'}\n"
        f"Tone: {tone}\nAudience: {audience or 'general audience'}\n"
        "Mix post types (education, behind the scenes, social proof only if given in the context, "
        f"promotion, engagement questions). Do not put more than {MAX_POSTS_PER_DAY} posts on one day, "
        "and make every idea different.\n\n"
        "Return ONLY a JSON array. Each element must have exactly these keys:\n"
        '{"day": 1, "platform": "<one of the platforms>", "format": "e.g. text post / carousel / reel", '
        '"idea": "one sentence", "hook": "first line of the post", "hashtags": ["#tag1", "#tag2"], '
        '"cta": "short call to action"}\n'
        f'"day" is a number from 1 to {days}.'
    )
    best: list[dict] = []
    note = ""
    for attempt in range(ATTEMPTS):
        text = llm.ask(SYSTEM_PROMPT, prompt + note, max_tokens=2500, temperature=0.7)
        items = parse_json_items(text)
        rows = clean_plan(normalize_plan(items, start, days, platforms))
        if len(rows) >= total:
            return rows[:total]
        if len(rows) > len(best):
            best = rows
        problems = plan_problems(len(items), rows, total)
        if not items:
            problems = ["your previous answer was not valid JSON"]
        note = ("\n\nYour previous answer was rejected: " + "; ".join(problems) +
                f". Reply with ONLY a valid JSON array of exactly {total} posts.")

    if not best:
        raise IncompleteCalendarError("The model did not return a usable calendar. Please try again.")
    raise IncompleteCalendarError(
        f"The calendar is incomplete: the model returned {len(best)} of the {total} requested posts "
        f"after {ATTEMPTS} tries. Please try again.", best)
