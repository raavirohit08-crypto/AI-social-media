"""Post generator: brand profile + brand facts (RAG) + platform rules -> posts.

Quality rules enforced in plain Python (never trusting the model to do them):
  * character limits (ask the model to shorten, never cut blindly)
  * the requested number of variations is really delivered (retry, top-up, or say so)
  * brand restrictions (profile, brand facts, language, tone) survive every rewrite
"""
import re
from dataclasses import dataclass, field
from typing import Callable

from .brand import avoid_terms_from_text, find_avoided_terms
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
   Never use a word or topic listed under "Words / topics to avoid" in the BRAND PROFILE.
5. Output ONLY the post text. No explanations, no preamble, no quotation marks around the post."""

VARIATION_MARK = "=== VARIATION {n} ==="
_SPLIT = re.compile(r"=+\s*VARIATION\s*\d+\s*=+", re.IGNORECASE)

MAX_VARIATIONS = 3
BATCH_ATTEMPTS = 2        # full tries to get every variation in one answer
TOPUP_ATTEMPTS = 2        # tries per missing variation after that


@dataclass
class Post:
    platform: str
    variation: int
    text: str
    stats: dict
    topic: str = ""
    sources: list[str] = field(default_factory=list)
    notice: str = ""      # set when the result is incomplete or needs attention


@dataclass
class PostSettings:
    """Everything that must stay the same when a post is rewritten or shortened."""
    tone: str = "Friendly"
    audience: str = ""
    goal: str = "Engagement"
    language: str = "English"        # "" = keep the language of the original text
    emojis: bool = True
    hashtag_count: int = 3
    include_cta: bool = True
    profile_text: str = ""
    brand_context: str = ""

    def prompt_block(self) -> str:
        """The brand + generation constraints, ready to put in front of a rewrite request."""
        parts = []
        if self.profile_text:
            parts.append(f"BRAND PROFILE:\n{self.profile_text}")
        if self.brand_context:
            parts.append(f"BRAND CONTEXT (facts and example posts):\n{self.brand_context}")
        language = self.language or "the same language as the original post"
        parts.append(
            f"SETTINGS TO KEEP: tone {self.tone}; goal {self.goal}; language {language}; "
            f"audience {self.audience or 'general audience'}; "
            f"emojis {'allowed' if self.emojis else 'not allowed'}."
        )
        return "\n\n".join(parts)


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
        + (f"Write exactly {variations} versions that use clearly different angles and openings. "
           f"Separate them with a line exactly like {VARIATION_MARK.format(n=1)} "
           f"(then {VARIATION_MARK.format(n=2)}, ...). Start with the first marker.\n" if variations > 1 else
           "Return just the post.\n")
    )
    return "\n\n".join(parts)


def split_variations(text: str, expected: int) -> list[str]:
    """Split the model output on '=== VARIATION n ===' markers.

    Empty pieces are dropped, and so is any chatter written BEFORE the first marker.
    At most `expected` versions are returned (it can be fewer: the caller must check).
    """
    text = text or ""
    first = _SPLIT.search(text)
    if first:
        text = text[first.end():]
    pieces = [p.strip() for p in _SPLIT.split(text) if p.strip()]
    return pieces[:expected]


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def unique_versions(texts: list[str]) -> list[str]:
    """Drop empty or repeated versions (same words ignoring case/punctuation)."""
    seen, out = set(), []
    for text in texts:
        key = _norm(text)
        if key and key not in seen:
            seen.add(key)
            out.append(text.strip())
    return out


def enforce_limit(llm: LLM, text: str, platform: str, attempts: int = 2,
                  settings: PostSettings | None = None) -> str:
    """If a post is over the hard limit, ask the model to shorten it (never cut blindly).

    `settings` carries the brand profile, brand facts, language, tone and emoji rules,
    so the shorter text still follows the brand's restrictions.
    """
    spec = PLATFORMS[platform]
    context = (settings.prompt_block() + "\n\n") if settings else ""
    for _ in range(attempts):
        stats = analyze_post(text, platform)
        if not stats["over_limit"]:
            break
        target = max(spec.limit - 15, 20)
        text = llm.ask(
            SYSTEM_PROMPT,
            f"{context}Rewrite this {platform} post so it is under {target} characters (it is {stats['chars']}). "
            f"Keep the meaning, tone, language and hashtags, and keep following the brand rules above. "
            f"Do not add new facts. Output only the post.\n\nPOST:\n{text}",
            max_tokens=400, temperature=0.4,
        )
    return text


def _ask(llm, prompt: str, tokens: int, on_token: Callable[[str], None] | None, temperature=0.8) -> str:
    """One generation call; streams when a callback is given and the client supports it."""
    streamer = getattr(llm, "ask_streaming", None) if on_token else None
    if streamer:
        return streamer(SYSTEM_PROMPT, prompt, on_token=on_token, max_tokens=tokens, temperature=temperature)
    text = llm.ask(SYSTEM_PROMPT, prompt, max_tokens=tokens, temperature=temperature)
    if on_token:
        on_token(text)
    return text


def _collect_versions(llm, prompt: str, wanted: int, on_token) -> list[str]:
    """Ask for all versions at once; retry with a reminder while some are missing."""
    texts: list[str] = []
    for attempt in range(BATCH_ATTEMPTS):
        reminder = "" if attempt == 0 else (
            f"\n\nYour previous answer did not contain {wanted} separate versions. Reply again with "
            f"exactly {wanted} versions, each one starting with its marker line "
            f"({VARIATION_MARK.format(n=1)}, {VARIATION_MARK.format(n=2)}, ...).")
        raw = _ask(llm, prompt + reminder, 700 * wanted, on_token if attempt == 0 else None)
        texts = unique_versions(split_variations(raw, wanted))
        if len(texts) >= wanted:
            break
    return texts


def generate_posts(llm: LLM, topic: str, platform: str, tone: str = "Friendly", audience: str = "",
                   goal: str = "Engagement", language: str = "English", variations: int = 1,
                   emojis: bool = True, hashtag_count: int = 3, include_cta: bool = True,
                   profile_text: str = "", brand_context: str = "",
                   sources: list[str] | None = None,
                   on_token: Callable[[str], None] | None = None) -> list[Post]:
    """Return the requested number of posts, or fewer posts with `notice` set on them.

    on_token(piece) is called while the first answer streams in (for a live preview).
    """
    topic = (topic or "").strip()
    if not topic:
        raise ValueError("Please describe what the post should be about.")
    if platform not in PLATFORMS:
        raise ValueError(f"Unknown platform: {platform}")
    wanted = max(1, min(int(variations), MAX_VARIATIONS))

    prompt = build_prompt(topic, platform, tone, audience, goal, language, wanted,
                          emojis, hashtag_count, include_cta, profile_text, brand_context)
    settings = PostSettings(tone, audience, goal, language, emojis, hashtag_count, include_cta,
                            profile_text, brand_context)

    if wanted == 1:
        raw = _ask(llm, prompt, 700, on_token)
        texts = [raw.strip()]
    else:
        texts = _collect_versions(llm, prompt, wanted, on_token)
        # Top-up: ask for each missing version on its own, different from the ones we have.
        for _ in range(TOPUP_ATTEMPTS * max(0, wanted - len(texts))):
            if len(texts) >= wanted:
                break
            single = build_prompt(topic, platform, tone, audience, goal, language, 1,
                                  emojis, hashtag_count, include_cta, profile_text, brand_context)
            single += ("\n\nWrite ONE new version with a clearly different angle and opening from these "
                       "existing versions:\n" + "\n---\n".join(texts))
            raw = _ask(llm, single, 700, None)
            texts = unique_versions(texts + [raw.strip()])

    texts = [t for t in texts if t][:wanted]
    if not texts:
        raise ValueError("The model returned no usable text. Please try again.")

    notice = ""
    if len(texts) < wanted:
        notice = (f"Incomplete result: only {len(texts)} of {wanted} versions could be generated. "
                  "Click Generate again to get the missing ones.")

    avoid = avoid_terms_from_text(profile_text)
    posts = []
    for i, text in enumerate(texts, start=1):
        text = enforce_limit(llm, text, platform, settings=settings)
        stats = analyze_post(text, platform)
        for term in find_avoided_terms(text, avoid):
            stats["warnings"].append(f'Uses "{term}", which the brand profile says to avoid.')
        posts.append(Post(platform, i, text, stats, topic, sources or [], notice))
    return posts
