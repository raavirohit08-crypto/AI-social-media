"""Image-prompt ideas and hashtag suggestions.

Both are AI SUGGESTIONS. The app has no access to live social-media data, so
hashtags are NOT 'trending' lists; brand hashtags are taken from your own
documents and the rest come from the model.
"""
import json
import re

from .calendar_planner import parse_json_items
from .llm import LLM
from .platforms import PLATFORMS, extract_hashtags

STYLES = ["Warm lifestyle photo", "Photorealistic", "Flat illustration", "Minimal & clean",
          "Bold typography poster", "3D render"]

# Commonly used image shapes per platform (a guide, not a hard rule)
ASPECT = {
    "LinkedIn": "1.91:1 landscape or 1:1 square",
    "Instagram": "4:5 portrait or 1:1 square",
    "X (Twitter)": "16:9 landscape",
    "Facebook": "1.91:1 landscape or 1:1 square",
    "Short video script (Reels/Shorts)": "9:16 vertical",
}

IMAGE_SYSTEM = (
    "You are a creative director who writes prompts for text-to-image tools. "
    "Never include real people's names, celebrities, copyrighted characters or other brands' logos. "
    "AI image tools draw text badly, so avoid prompts that depend on readable words in the image. "
    "You reply with valid JSON only."
)
HASHTAG_SYSTEM = (
    "You are a social media strategist suggesting hashtags. Suggest only hashtags that are relevant, "
    "respectful and not misleading. Never use hashtags about tragedies, politics or hate. "
    "You reply with valid JSON only."
)


def parse_json_object(text: str) -> dict:
    """First {...} object in model text (handles ```json fences and chatter)."""
    text = re.sub(r"```(?:json)?", "", text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        data = json.loads(text[start:end + 1])
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


# ------------------------------------------------------------------ image prompts
def image_prompts(llm: LLM, post_text: str, platform: str, style: str = STYLES[0], count: int = 3,
                  profile_text: str = "", brand_context: str = "") -> list[dict]:
    post_text = (post_text or "").strip()
    if not post_text:
        raise ValueError("Paste a post (or pick a generated one) first.")
    if platform not in PLATFORMS:
        raise ValueError(f"Unknown platform: {platform}")
    count = max(1, min(int(count), 4))
    aspect = ASPECT[platform]

    prompt = (
        (f"BRAND PROFILE:\n{profile_text}\n\n" if profile_text else "")
        + (f"BRAND CONTEXT:\n{brand_context}\n\n" if brand_context else "")
        + f"POST ({platform}):\n{post_text}\n\n"
        f"Suggest {count} different visual concepts to accompany this post. Visual style: {style}.\n"
        "For each, write a detailed text-to-image prompt covering: subject, setting, lighting, "
        "composition, color palette and mood. Keep each prompt under 80 words. "
        "Only show products or details that the post or brand context actually mention.\n"
        "Return ONLY a JSON array. Each element must have exactly these keys:\n"
        '{"concept": "short title", "prompt": "the text-to-image prompt", '
        '"alt_text": "accessible description of the image, max 120 characters"}'
    )
    items = []
    for attempt in range(2):
        text = llm.ask(IMAGE_SYSTEM, prompt if attempt == 0 else
                       prompt + "\n\nYour previous answer was not valid JSON. Reply with ONLY the JSON array.",
                       max_tokens=1200, temperature=0.8)
        items = [i for i in parse_json_items(text) if str(i.get("prompt", "")).strip()]
        if items:
            break
    if not items:
        raise ValueError("The model did not return usable image ideas. Please try again.")
    return [{
        "concept": str(i.get("concept", f"Idea {n}")).strip() or f"Idea {n}",
        "prompt": str(i["prompt"]).strip(),
        "alt_text": str(i.get("alt_text", "")).strip()[:125],
        "aspect": aspect, "style": style, "platform": platform,
    } for n, i in enumerate(items[:count], start=1)]


# ------------------------------------------------------------------ hashtags
def clean_tags(values, exclude: set[str] = frozenset(), limit: int = 12) -> list[str]:
    """Normalize model output to unique, valid #hashtags."""
    if isinstance(values, str):
        values = re.findall(r"#?\w+", values)
    out, seen = [], {t.lower() for t in exclude}
    for value in values or []:
        word = re.sub(r"[^\w]", "", str(value), flags=re.UNICODE)
        if not word or word.isdigit() or len(word) > 30:
            continue
        tag = "#" + word
        if tag.lower() not in seen:
            seen.add(tag.lower())
            out.append(tag)
    return out[:limit]


def recommended_set(groups: dict, platform: str) -> list[str]:
    """Pick a sensible number of hashtags: branded first, then niche, then broad."""
    maximum = PLATFORMS[platform].hashtags[1]
    picked = []
    for tag in groups["branded"][:2] + groups["niche"] + groups["broad"]:
        if tag not in picked:
            picked.append(tag)
    return picked[:maximum]


def hashtag_ideas(llm: LLM, topic: str, platform: str, profile_text: str = "",
                  brand_context: str = "") -> dict:
    topic = (topic or "").strip()
    if not topic:
        raise ValueError("Describe the topic or paste the post first.")
    if platform not in PLATFORMS:
        raise ValueError(f"Unknown platform: {platform}")

    # Brand hashtags come from the user's OWN documents, not from the model.
    branded = clean_tags(extract_hashtags(brand_context + "\n" + profile_text), limit=6)

    prompt = (
        (f"BRAND PROFILE:\n{profile_text}\n\n" if profile_text else "")
        + f"Platform: {platform}\nTopic / post:\n{topic}\n\n"
        "Suggest hashtags in two groups:\n"
        '- "broad": 4 popular, general hashtags for this topic (large audiences)\n'
        '- "niche": 6 specific hashtags for this topic or community (smaller, more targeted)\n'
        'Return ONLY a JSON object: {"broad": ["#a"], "niche": ["#b"]}'
    )
    data = {}
    for attempt in range(2):
        text = llm.ask(HASHTAG_SYSTEM, prompt if attempt == 0 else
                       prompt + "\n\nYour previous answer was not valid JSON. Reply with ONLY the JSON object.",
                       max_tokens=600, temperature=0.6)
        data = parse_json_object(text)
        if data.get("broad") or data.get("niche"):
            break
    else:
        raise ValueError("The model did not return usable hashtags. Please try again.")

    groups = {"branded": branded,
              "broad": clean_tags(data.get("broad"), exclude=set(branded), limit=6)}
    groups["niche"] = clean_tags(data.get("niche"), exclude=set(branded) | set(groups["broad"]), limit=8)
    groups["recommended"] = recommended_set(groups, platform)
    return groups
