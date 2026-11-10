"""Rewrite an existing draft: shorten, stronger hook, change tone, fix grammar, translate...

The saved brand profile and (optionally) brand facts are passed to every rewrite, so
improved or shortened copy still follows the brand's voice and restrictions.
"""
from .brand import avoid_terms_from_text, find_avoided_terms
from .generator import SYSTEM_PROMPT, PostSettings, enforce_limit
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


def improve_post(llm: LLM, text: str, action: str, platform: str, tone: str = "Friendly",
                 language: str = "English", profile_text: str = "", brand_context: str = "",
                 audience: str = "") -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("Paste a draft first.")
    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action}")
    spec = PLATFORMS[platform]

    # The language only changes when the action is Translate; otherwise keep the draft's language.
    target_language = language if action == "Translate" else ""
    # The tone only changes for 'Change tone'; otherwise the brand voice leads.
    keep_tone = tone if action == "Change tone" else "the brand voice"
    settings = PostSettings(tone=keep_tone, audience=audience, language=target_language,
                            profile_text=profile_text, brand_context=brand_context)

    extras = []
    if action == "Change tone":
        extras.append(f"Requested tone: {tone}.")
    if action == "Translate":
        extras.append(f"Target language: {language}.")
    prompt = (
        f"{settings.prompt_block()}\n\n"
        f"Platform: {platform}. {spec.guidance}\n"
        f"Instruction: {ACTIONS[action]} {' '.join(extras)}\n"
        "Do not add facts, numbers or claims that are not in the original draft or the BRAND CONTEXT. "
        "Follow the brand profile, including the words and topics to avoid.\n\n"
        f"ORIGINAL DRAFT:\n{text}\n\nREWRITTEN POST:"
    )
    result = llm.ask(SYSTEM_PROMPT, prompt, max_tokens=800, temperature=0.5)
    result = enforce_limit(llm, result, platform, settings=settings)

    stats = analyze_post(result, platform)
    for term in find_avoided_terms(result, avoid_terms_from_text(profile_text)):
        stats["warnings"].append(f'Uses "{term}", which the brand profile says to avoid.')
    return {"platform": platform, "action": action, "original": text, "text": result, "stats": stats}
