
"""Rewrite an existing draft while preserving brand rules and settings."""

from .generator import SYSTEM_PROMPT, enforce_limit
from .llm import LLM
from .platforms import PLATFORMS, analyze_post


ACTIONS = {
    "Make it shorter": "Make it clearly shorter and tighter without losing the main message.",
    "Stronger hook": "Rewrite the first line as a much stronger scroll-stopping hook. Keep the rest, adjusting only what is needed.",
    "More engaging": "Make it more engaging and conversational. Add a question or a relatable moment if it fits.",
    "Change tone": "Rewrite it in the requested tone while keeping the facts the same.",
    "Fix grammar & spelling": "Fix grammar, spelling and punctuation only. Do not change the wording style or length.",
    "Add hashtags & CTA": "Keep the post text, then add a clear call to action and relevant hashtags suitable for the platform.",
    "Translate": "Translate it into the requested language, keeping the tone, emojis and hashtags natural for that language.",
}


def improve_post(
    llm: LLM,
    text: str,
    action: str,
    platform: str,
    tone: str = "Friendly",
    language: str = "English",
    profile_text: str = "",
    brand_context: str = "",
    emojis: bool = True,
    hashtag_count: int = 3,
    include_cta: bool = True,
) -> dict:
    """Improve a draft without violating brand or platform requirements."""

    text = (text or "").strip()

    if not text:
        raise ValueError("Paste a draft first.")

    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action}")

    if platform not in PLATFORMS:
        raise ValueError(f"Unknown platform: {platform}")

    spec = PLATFORMS[platform]

    brand_rules = ""

    if profile_text:
        brand_rules += f"\nBRAND PROFILE:\n{profile_text}\n"

    if brand_context:
        brand_rules += f"\nBRAND CONTEXT:\n{brand_context}\n"

    extras = [
        f"Requested tone: {tone}.",
        f"Required language: {language}.",
        f"Emojis: {'Use naturally' if emojis else 'Do not use emojis'}.",
        f"Hashtags: Include approximately {hashtag_count} relevant hashtags.",
        f"Call to action: {'Include a clear CTA' if include_cta else 'Do not add a CTA'}.",
    ]

    if action == "Translate":
        extras.append(f"Translate the post into {language}.")

    prompt = (
        f"PLATFORM: {platform}\n"
        f"PLATFORM RULES: {spec.guidance}\n"
        f"{brand_rules}\n"
        f"INSTRUCTION: {ACTIONS[action]}\n"
        f"{' '.join(extras)}\n\n"
        "IMPORTANT RULES:\n"
        "1. Preserve all brand restrictions and the brand voice.\n"
        "2. Do not invent facts, numbers, prices, offers, testimonials or claims.\n"
        "3. Use only facts from the original draft or supplied brand context.\n"
        "4. Follow the requested language and generation settings.\n"
        "5. Follow the platform rules and character limit.\n"
        "6. Return only the rewritten post, without explanations.\n\n"
        f"ORIGINAL DRAFT:\n{text}\n\n"
        "REWRITTEN POST:"
    )

    result = llm.ask(
        SYSTEM_PROMPT,
        prompt,
        max_tokens=800,
        temperature=0.5
    )

    result = (result or "").strip()

    if not result:
        raise ValueError("The model returned an empty rewrite. Please try again.")

    result = enforce_limit(
        llm,
        result,
        platform,
        profile_text=profile_text,
        brand_context=brand_context,
        tone=tone,
        language=language,
    )

    return {
        "platform": platform,
        "action": action,
        "original": text,
        "text": result,
        "stats": analyze_post(result, platform),
    }
