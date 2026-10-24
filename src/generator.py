"""Post generator: brand profile + brand facts (RAG) + platform rules -> posts."""
import re
from dataclasses import dataclass, field

from .llm import LLM
from .platforms import PLATFORMS, analyze_post

GOALS = ["Brand awareness", "Engagement", "Lead generation", "Sales / promotion",
         "Education", "Event / launch"]
TONES = ["Friendly", "Professional", "Witty / playful", "Inspirational", "Bold / urgent", "Educational"]
LANGUAGES = ["English", "Hindi", "Telugu", "Tamil", "Hinglish", "Spanish", "French"]

SYSTEM_PROMPT = """You are an expert social media copywriter.
RULES
1. Follow the platform rules and the character limit exactly.
2. Facts about the brand, its products, prices, offers, dates and locations may come ONLY from the
   BRAND CONTEXT or from the user's request. NEVER invent statistics, discounts, testimonials,
   awards, reviews or claims. If a needed fact is missing, write around it or use a clear
   placeholder such as [DISCOUNT] or [DATE].
3. No false or misleading health, financial or income promises. No hateful, harassing or
   deceptive content, and no content targeting people for who they are.
4. Match the brand voice shown in the BRAND PROFILE and in the example posts inside BRAND CONTEXT.
5. Output ONLY the post text. No explanations, no preamble, no quotation marks around the post."""

VARIATION_MARK = "=== VARIATION {n} ==="
_SPLIT = re.compile(r"=+\s*VARIATION\s*\d+\s*=+", re.IGNORECASE)


@dataclass
class Post:
    platform: str
    variation: int
    text: str
    stats: dict
    topic: str = ""
    sources: list[str] = field(default_factory=list)


def build_prompt(topic, platform, tone, audience, goal, language, variations,
                 emojis, hashtag_count, include_cta, profile_text, brand_context) -> str:
    spec = PLATFORMS[platform]
    parts = []
    if profile_text:
        parts.append(f"BRAND PROFILE:\n{profile_text}")
    if brand_context:
        parts.append(f"BRAND CONTEXT (facts and example posts):\n{brand_context}")
    limit_text = f"Hard limit: {spec.limit} characters." if spec.limit and spec.limit < 10000 else ""
    hashtags = min(hashtag_count, spec.hashtags[1])
    parts.append(
        f"TASK: Write {variations} different {platform} version(s) about the topic below.\n"
        f"PLATFORM RULES: {spec.guidance} {limit_text}\n"
        f"TOPIC / IDEA: {topic}\n"
        f"GOAL: {goal}\nTONE: {tone}\nTARGET AUDIENCE: {audience or 'general audience'}\n"
        f"LANGUAGE: {language}\n"
        f"EMOJIS: {'use them naturally' if emojis else 'do not use any emojis'}\n"
        f"HASHTAGS: {f'include about {hashtags} relevant hashtags' if hashtags else 'do not use hashtags'}\n"
        f"CALL TO ACTION: {'include a clear one' if include_cta else 'not needed'}\n\n"
        + (f"Separate the versions with a line exactly like {VARIATION_MARK.format(n=1)} "
           f"(then 2, 3...). Start with the first marker.\n" if variations > 1 else
           "Return just the post.\n")
    )
    return "\n\n".join(parts)


def split_variations(text: str, expected: int) -> list[str]:
    """Split the model output on '=== VARIATION n ===' markers."""
    pieces = [p.strip() for p in _SPLIT.split(text) if p.strip()]
    if not pieces:
        return []
    return pieces[:expected]


def enforce_limit(llm: LLM, text: str, platform: str, attempts: int = 2) -> str:
    """If a post is over the hard limit, ask the model to shorten it (never cut blindly)."""
    spec = PLATFORMS[platform]
    for _ in range(attempts):
        stats = analyze_post(text, platform)
        if not stats["over_limit"]:
            break
        target = max(spec.limit - 15, 20)
        text = llm.ask(
            SYSTEM_PROMPT,
            f"Rewrite this {platform} post so it is under {target} characters (it is {stats['chars']}). "
            f"Keep the meaning, tone and hashtags. Output only the post.\n\nPOST:\n{text}",
            max_tokens=400, temperature=0.4,
        )
    return text


def generate_posts(llm: LLM, topic: str, platform: str, tone: str = "Friendly", audience: str = "",
                   goal: str = "Engagement", language: str = "English", variations: int = 1,
                   emojis: bool = True, hashtag_count: int = 3, include_cta: bool = True,
                   profile_text: str = "", brand_context: str = "",
                   sources: list[str] | None = None) -> list[Post]:
    topic = (topic or "").strip()
    if not topic:
        raise ValueError("Please describe what the post should be about.")
    if platform not in PLATFORMS:
        raise ValueError(f"Unknown platform: {platform}")
    variations = max(1, min(int(variations), 3))

    prompt = build_prompt(topic, platform, tone, audience, goal, language, variations,
                          emojis, hashtag_count, include_cta, profile_text, brand_context)
    raw = llm.ask(SYSTEM_PROMPT, prompt, max_tokens=700 * variations, temperature=0.8)

    texts = split_variations(raw, variations) if variations > 1 else [raw.strip()]
    if not texts:
        texts = [raw.strip()]
    posts = []
    for i, text in enumerate(texts, start=1):
        text = enforce_limit(llm, text, platform)
        posts.append(Post(platform, i, text, analyze_post(text, platform), topic, sources or []))
    return posts
