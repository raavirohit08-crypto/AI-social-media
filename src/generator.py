
"""Post generator: brand profile + brand facts (RAG) + platform rules -> posts."""

import re
from dataclasses import dataclass, field

from .llm import LLM
from .platforms import PLATFORMS, analyze_post


GOALS = [
    "Brand awareness",
    "Engagement",
    "Lead generation",
    "Sales / promotion",
    "Education",
    "Event / launch"
]

TONES = [
    "Friendly",
    "Professional",
    "Witty / playful",
    "Inspirational",
    "Bold / urgent",
    "Educational"
]

LANGUAGES = [
    "English",
    "Hindi",
    "Telugu",
    "Tamil",
    "Hinglish",
    "Spanish",
    "French"
]


SYSTEM_PROMPT = """You are an expert social media copywriter.

RULES:
1. Follow platform rules and character limits exactly.
2. Use brand facts only from BRAND CONTEXT or the user's request.
3. Never invent statistics, discounts, testimonials, awards or claims.
4. Match the brand voice shown in BRAND PROFILE.
5. Do not make false or misleading promises.
6. Follow the requested language and tone.
7. Output only the requested post content.
"""


VARIATION_MARK = "=== VARIATION {n} ==="

_SPLIT = re.compile(
    r"^===\s*VARIATION\s+(\d+)\s*===$",
    re.IGNORECASE | re.MULTILINE
)


@dataclass
class Post:
    platform: str
    variation: int
    text: str
    stats: dict
    topic: str = ""
    sources: list[str] = field(default_factory=list)


def build_prompt(
    topic,
    platform,
    tone,
    audience,
    goal,
    language,
    variations,
    emojis,
    hashtag_count,
    include_cta,
    profile_text,
    brand_context
) -> str:

    spec = PLATFORMS[platform]
    parts = []

    if profile_text:
        parts.append(f"BRAND PROFILE:\n{profile_text}")

    if brand_context:
        parts.append(
            f"BRAND CONTEXT (facts and example posts):\n{brand_context}"
        )

    limit_text = (
        f"Hard limit: {spec.limit} characters."
        if spec.limit and spec.limit < 10000
        else ""
    )

    hashtags = min(hashtag_count, spec.hashtags[1])

    parts.append(
        f"TASK: Write exactly {variations} different "
        f"{platform} version(s).\n"

        f"PLATFORM RULES: {spec.guidance}\n"
        f"{limit_text}\n"

        f"TOPIC / IDEA: {topic}\n"
        f"GOAL: {goal}\n"
        f"TONE: {tone}\n"
        f"TARGET AUDIENCE: {audience or 'general audience'}\n"
        f"LANGUAGE: {language}\n"

        f"EMOJIS: "
        f"{'Use them naturally' if emojis else 'Do not use emojis'}\n"

        f"HASHTAGS: "
        f"{f'Include about {hashtags} relevant hashtags' if hashtags else 'No hashtags'}\n"

        f"CALL TO ACTION: "
        f"{'Include a clear call to action' if include_cta else 'Not needed'}\n\n"

        + (
            "IMPORTANT OUTPUT FORMAT:\n"
            f"{VARIATION_MARK.format(n=1)}\n"
            "First post content\n\n"
            f"{VARIATION_MARK.format(n=2)}\n"
            "Second post content\n\n"
            f"{VARIATION_MARK.format(n=3)}\n"
            "Third post content\n\n"
            "Every variation must be different and complete.\n"
            "Do not skip any variation marker.\n"
            if variations > 1
            else "Return only one complete post.\n"
        )
    )

    return "\n\n".join(parts)


def split_variations(text: str, expected: int) -> list[str]:
    """
    Split model output using variation markers.
    Return an empty list if any requested variation is missing.
    """

    if not text:
        return []

    matches = list(_SPLIT.finditer(text))

    # Every requested variation must have a marker.
    if len(matches) != expected:
        return []

    numbers = [int(match.group(1)) for match in matches]

    # Check marker order and missing variations.
    if numbers != list(range(1, expected + 1)):
        return []

    pieces = []

    for i, match in enumerate(matches):

        start = match.end()

        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(text)
        )

        content = text[start:end].strip()

        if not content:
            return []

        pieces.append(content)

    return pieces


def enforce_limit(
    llm: LLM,
    text: str,
    platform: str,
    attempts: int = 2,
    profile_text: str = "",
    brand_context: str = "",
    tone: str = "Friendly",
    language: str = "English"
) -> str:

    """Shorten posts while preserving brand restrictions."""

    spec = PLATFORMS[platform]

    for _ in range(attempts):

        stats = analyze_post(text, platform)

        if not stats["over_limit"]:
            return text

        target = max(spec.limit - 15, 20)

        prompt = (
            f"BRAND PROFILE:\n{profile_text}\n\n"
            f"BRAND CONTEXT:\n{brand_context}\n\n"
            f"PLATFORM: {platform}\n"
            f"TONE: {tone}\n"
            f"LANGUAGE: {language}\n\n"

            f"Rewrite this post under {target} characters.\n"
            "Preserve the original meaning and important facts.\n"
            "Preserve brand restrictions and language.\n"
            "Do not invent any new information.\n"
            "Return only the rewritten post.\n\n"

            f"POST:\n{text}"
        )

        text = llm.ask(
            SYSTEM_PROMPT,
            prompt,
            max_tokens=400,
            temperature=0.4
        ).strip()

        if not text:
            raise ValueError("AI returned an empty shortened post.")

    final_stats = analyze_post(text, platform)

    if final_stats["over_limit"]:
        raise ValueError(
            f"Post exceeds the {platform} character limit. "
            "Please generate again."
        )

    return text


def generate_posts(
    llm: LLM,
    topic: str,
    platform: str,
    tone: str = "Friendly",
    audience: str = "",
    goal: str = "Engagement",
    language: str = "English",
    variations: int = 1,
    emojis: bool = True,
    hashtag_count: int = 3,
    include_cta: bool = True,
    profile_text: str = "",
    brand_context: str = "",
    sources: list[str] | None = None
) -> list[Post]:

    """Generate and validate social media posts."""

    topic = (topic or "").strip()

    if not topic:
        raise ValueError(
            "Please describe what the post should be about."
        )

    if platform not in PLATFORMS:
        raise ValueError(
            f"Unknown platform: {platform}"
        )

    variations = max(1, min(int(variations), 3))

    prompt = build_prompt(
        topic,
        platform,
        tone,
        audience,
        goal,
        language,
        variations,
        emojis,
        hashtag_count,
        include_cta,
        profile_text,
        brand_context
    )

    posts = []

    # Retry incomplete AI responses up to 3 times.
    for attempt in range(3):

        retry_instruction = ""

        if attempt > 0:
            retry_instruction = (
                f"\n\nYour previous response was incomplete. "
                f"Generate exactly {variations} complete posts. "
                "Include every required variation marker in order. "
                "Do not leave any variation empty. "
                "Do not combine multiple variations."
            )

        raw = llm.ask(
            SYSTEM_PROMPT,
            prompt + retry_instruction,
            max_tokens=700 * variations,
            temperature=0.8
        )

        if not raw:
            continue

        # Parse and validate variations.
        if variations == 1:
            texts = [raw.strip()]
        else:
            texts = split_variations(raw, variations)

        # Retry if the model returned incomplete output.
        if len(texts) != variations:
            continue

        validated_posts = []

        for i, text in enumerate(texts, start=1):

            try:
                text = enforce_limit(
                    llm,
                    text,
                    platform,
                    profile_text=profile_text,
                    brand_context=brand_context,
                    tone=tone,
                    language=language
                )

                stats = analyze_post(text, platform)

                if stats["over_limit"]:
                    raise ValueError(
                        f"Variation {i} exceeds character limit."
                    )

                validated_posts.append(
                    Post(
                        platform=platform,
                        variation=i,
                        text=text,
                        stats=stats,
                        topic=topic,
                        sources=sources or []
                    )
                )

            except Exception:
                validated_posts = []
                break

        if len(validated_posts) == variations:
            posts = validated_posts
            break

    # Never return incomplete results.
    if len(posts) != variations:
        raise ValueError(
            f"Could not generate all {variations} variations "
            "after 3 attempts. Please try again."
        )

    return posts
