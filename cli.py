"""Command-line version.

Usage:
    python cli.py "Launching our new cold brew this Friday" LinkedIn Instagram
Platforms: LinkedIn, Instagram, "X (Twitter)", Facebook, "Short video script (Reels/Shorts)"
"""
import sys

from src.brand import get_brand_context, load_profile, profile_to_text
from src.config import load_settings
from src.generator import generate_posts
from src.kb import build_or_load
from src.llm import LLM, LLMError
from src.platforms import PLATFORMS
from src.vector_store import get_embeddings


def main(args: list[str]) -> None:
    if not args:
        sys.exit(__doc__)
    topic, platforms = args[0], args[1:] or ["LinkedIn"]
    settings = load_settings()
    llm = LLM(settings.hf_token, settings.model_name)

    store, loaded, _ = build_or_load(get_embeddings(), settings)
    context, passages = get_brand_context(store, topic, settings.top_k)
    print(f"Brand documents used: {len(loaded)}")
    profile_text = profile_to_text(load_profile())

    for platform in platforms:
        if platform not in PLATFORMS:
            print(f"Skipping unknown platform: {platform}")
            continue
        try:
            for post in generate_posts(llm, topic, platform, profile_text=profile_text,
                                       brand_context=context):
                print(f"\n=== {platform} ({post.stats['chars']} chars) ===\n{post.text}")
                for w in post.stats["warnings"]:
                    print(f"  ! {w}")
        except (LLMError, ValueError) as error:
            print(f"Error for {platform}: {error}")


if __name__ == "__main__":
    main(sys.argv[1:])
