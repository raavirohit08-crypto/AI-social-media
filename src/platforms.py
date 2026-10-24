"""Platform rules + a local post analyzer (character limits, hashtags, emojis).

The LLM is told these rules, but we also CHECK the result in plain Python,
because language models are unreliable at counting characters.
"""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Platform:
    name: str
    icon: str
    limit: int | None            # hard character limit (None = no practical limit)
    ideal: int | None            # recommended length
    hashtags: tuple[int, int]    # recommended (min, max)
    guidance: str
    is_script: bool = False


PLATFORMS: dict[str, Platform] = {p.name: p for p in [
    Platform(
        "LinkedIn", "💼", 3000, 1300, (3, 5),
        "Professional but human. The first line must be a strong hook (only about two lines show "
        "before 'see more'). Short paragraphs with blank lines between them. End with a takeaway or "
        "a question. Use at most 1-2 emojis. Put hashtags on the last line.",
    ),
    Platform(
        "Instagram", "📸", 2200, 900, (3, 10),
        "Visual, friendly and energetic. Open with a scroll-stopping first line. Short lines, "
        "emojis where natural, a clear call to action, hashtags at the end. After the caption add "
        "one final line that starts with 'Visual idea:' describing the photo, carousel or reel to pair with it.",
    ),
    Platform(
        "X (Twitter)", "🐦", 280, 280, (0, 2),
        "One punchy post that fits in 280 characters INCLUDING hashtags and spaces. Direct, clear, "
        "no filler. At most 1-2 hashtags. No threads.",
    ),
    Platform(
        "Facebook", "👍", 63206, 500, (0, 2),
        "Conversational and community-minded, about 40-90 words. Ask a question to start comments. "
        "Zero to two hashtags.",
    ),
    Platform(
        "Short video script (Reels/Shorts)", "🎬", None, None, (3, 5),
        "Write a 30-45 second vertical-video script (about 75-110 spoken words). Use exactly this "
        "structure: HOOK (first 3 seconds), then 3-4 numbered SCENES, each with 'On-screen text:' and "
        "'Voiceover:', then CTA, then a one-line CAPTION with hashtags.",
        is_script=True,
    ),
]}

_HASHTAG = re.compile(r"(?<!\w)#\w*[^\W\d_]\w*", re.UNICODE)
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U00002B50\U00002B06\U0000FE0F]"
)
_VISUAL_LINE = re.compile(r"^\s*visual idea\s*:.*$", re.IGNORECASE | re.MULTILINE)


def split_visual_idea(text: str) -> tuple[str, str]:
    """Separate the 'Visual idea:' line (Instagram) from the post itself."""
    match = _VISUAL_LINE.search(text)
    if not match:
        return text.strip(), ""
    return (text[:match.start()] + text[match.end():]).strip(), match.group(0).split(":", 1)[1].strip()


def extract_hashtags(text: str) -> list[str]:
    seen, tags = set(), []
    for tag in _HASHTAG.findall(text):
        if tag.lower() not in seen:
            seen.add(tag.lower())
            tags.append(tag)
    return tags


def analyze_post(text: str, platform: str) -> dict:
    """Return counts and a list of warnings for a post on the given platform."""
    spec = PLATFORMS[platform]
    body, _ = split_visual_idea(text)
    chars, words = len(body), len(body.split())
    tags = extract_hashtags(body)
    emojis = len(_EMOJI.findall(body))
    over = bool(spec.limit and chars > spec.limit)

    warnings = []
    if over:
        warnings.append(f"Over the {spec.limit}-character limit by {chars - spec.limit}.")
    elif spec.ideal and spec.limit != spec.ideal and chars > spec.ideal * 1.5:
        warnings.append(f"Longer than the recommended ~{spec.ideal} characters.")
    low, high = spec.hashtags
    if len(tags) > high:
        warnings.append(f"{len(tags)} hashtags; {high} or fewer works better here.")
    elif len(tags) < low and not spec.is_script:
        warnings.append(f"Only {len(tags)} hashtag(s); {low}+ is recommended.")
    if platform == "LinkedIn" and emojis > 4:
        warnings.append("Many emojis for LinkedIn; consider fewer.")

    info = {"chars": chars, "words": words, "hashtags": len(tags), "emojis": emojis,
            "limit": spec.limit, "over_limit": over, "warnings": warnings}
    if spec.is_script:
        info["seconds"] = round(words / 2.5)   # ~150 spoken words per minute
    return info
