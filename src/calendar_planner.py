
"""Content calendar: generate JSON plan and validate it in Python."""

import json
import re
from datetime import date, timedelta

from .llm import LLM
from .platforms import PLATFORMS


SYSTEM_PROMPT = (
    "You are a social media strategist. You plan content calendars. "
    "Use brand facts ONLY from the context given; never invent offers, "
    "statistics or events. Reply with valid JSON only."
)

COLUMNS = [
    "date", "day", "platform", "format",
    "idea", "hook", "hashtags", "cta"
]


def parse_json_items(text: str) -> list[dict]:
    """Extract JSON objects from model response."""

    text = re.sub(
        r"```(?:json)?",
        "",
        text or "",
        flags=re.IGNORECASE
    ).strip()

    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])

            if isinstance(data, list):
                return [
                    item for item in data
                    if isinstance(item, dict)
                ]

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
    """Convert hashtags into a clean string."""

    if isinstance(value, str):
        value = re.findall(r"#?\w+", value)

    if not isinstance(value, list):
        return ""

    tags = [
        "#" + str(tag).lstrip("#")
        for tag in value
        if str(tag).strip("# ")
    ]

    return " ".join(tags[:8])


def normalize_plan(
    items: list[dict],
    start: date,
    days: int,
    platforms: list[str]
) -> list[dict]:
    """Validate and normalize calendar rows."""

    lookup = {
        platform.lower(): platform
        for platform in platforms
    }

    rows = []

    for item in items:

        try:
            day = int(item.get("day"))
        except (TypeError, ValueError):
            continue

        platform = lookup.get(
            str(item.get("platform", "")).strip().lower()
        )

        idea = str(item.get("idea", "")).strip()

        if not (1 <= day <= days):
            continue

        if not platform or not idea:
            continue

        when = start + timedelta(days=day - 1)

        rows.append({
            "date": when.isoformat(),
            "day": when.strftime("%A"),
            "platform": platform,
            "format": str(item.get("format", "")).strip(),
            "idea": idea,
            "hook": str(item.get("hook", "")).strip(),
            "hashtags": _clean_hashtags(
                item.get("hashtags")
            ),
            "cta": str(item.get("cta", "")).strip()
        })

    return sorted(
        rows,
        key=lambda row: (
            row["date"],
            row["platform"]
        )
    )


def validate_calendar(
    rows: list[dict],
    expected_total: int
) -> bool:
    """Check count, duplicates and daily posting limits."""

    if len(rows) != expected_total:
        return False

    seen = set()
    daily_counts = {}

    for row in rows:

        # Duplicate date + platform + idea is not allowed.
        key = (
            row["date"],
            row["platform"],
            row["idea"].strip().lower()
        )

        if key in seen:
            return False

        seen.add(key)

        # Maximum two posts per day.
        daily_counts[row["date"]] = (
            daily_counts.get(row["date"], 0) + 1
        )

        if daily_counts[row["date"]] > 2:
            return False

    return True


def plan_calendar(
    llm: LLM,
    goal: str,
    platforms: list[str],
    days: int = 7,
    posts_per_week: int = 4,
    start: date | None = None,
    theme: str = "",
    tone: str = "Friendly",
    audience: str = "",
    profile_text: str = "",
    brand_context: str = ""
) -> list[dict]:

    """Generate a complete and validated content calendar."""

    if not platforms:
        raise ValueError("Choose at least one platform.")

    for platform in platforms:
        if platform not in PLATFORMS:
            raise ValueError(
                f"Unknown platform: {platform}"
            )

    if days < 1:
        raise ValueError("Days must be at least 1.")

    if posts_per_week < 1:
        raise ValueError(
            "Posts per week must be at least 1."
        )

    start = start or date.today()

    total = max(
        1,
        round(days / 7 * posts_per_week)
    )

    # Never allow more than two posts per day.
    if total > days * 2:
        raise ValueError(
            "Too many posts requested. Maximum two posts per day."
        )

    prompt = (
        (f"BRAND PROFILE:\n{profile_text}\n\n"
         if profile_text else "")

        + (f"BRAND CONTEXT:\n{brand_context}\n\n"
           if brand_context else "")

        + f"Plan a {days}-day social media content calendar "
        f"with exactly {total} posts.\n"

        + f"Platforms: {', '.join(platforms)}\n"
        + f"Goal: {goal}\n"
        + f"Theme: {theme or 'General brand content'}\n"
        + f"Tone: {tone}\n"
        + f"Audience: {audience or 'General audience'}\n"

        + "Use different content ideas. "
        "Do not repeat the same idea.\n"

        + "Maximum two posts per day.\n"

        + "Use only the supplied brand facts. "
        "Never invent offers, statistics or events.\n\n"

        + "Return ONLY a valid JSON array.\n"

        + "Each object must contain these keys:\n"

        + '{"day": 1, '
        '"platform": "Instagram", '
        '"format": "Carousel", '
        '"idea": "One sentence idea", '
        '"hook": "Opening line", '
        '"hashtags": ["#tag1", "#tag2"], '
        '"cta": "Call to action"}\n\n'

        + f"Generate exactly {total} objects.\n"
        + f"Day must be between 1 and {days}.\n"
        + "Do not return fewer objects."
    )

    rows = []

    # Retry incomplete responses three times.
    for attempt in range(3):

        retry_prompt = ""

        if attempt > 0:
            retry_prompt = (
                f"\n\nYour previous response was incomplete "
                f"or invalid. Generate exactly {total} "
                "complete, unique calendar entries. "
                "Return ONLY a JSON array. "
                "Maximum two posts per day."
            )

        text = llm.ask(
            SYSTEM_PROMPT,
            prompt + retry_prompt,
            max_tokens=3000,
            temperature=0.7
        )

        items = parse_json_items(text)

        rows = normalize_plan(
            items,
            start,
            days,
            platforms
        )

        if validate_calendar(rows, total):
            return rows

    raise ValueError(
        f"Calendar generation incomplete. "
        f"Expected {total} valid posts, but the AI "
        "could not generate a complete calendar after "
        "3 attempts. Please try again."
    )
