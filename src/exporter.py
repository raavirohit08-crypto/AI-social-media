"""Export generated content (Markdown for reading, CSV for spreadsheets / schedulers)."""
import csv
import io
from datetime import datetime

from .calendar_planner import COLUMNS


def posts_to_markdown(posts: list[dict]) -> str:
    out = [f"# Social media posts\n\n_Generated on {datetime.now():%Y-%m-%d %H:%M}_\n"]
    for p in posts:
        out.append(f"\n---\n\n## {p['platform']} · version {p['variation']}\n")
        if p.get("topic"):
            out.append(f"**Topic:** {p['topic']}\n")
        out.append(p["text"] + "\n")
        s = p["stats"]
        out.append(f"\n_{s['chars']} characters · {s['hashtags']} hashtags_")
        for w in s["warnings"]:
            out.append(f"\n⚠️ {w}")
        out.append("")
    return "\n".join(out)


def posts_to_csv(posts: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["platform", "version", "topic", "text", "characters", "hashtags", "warnings"])
    for p in posts:
        s = p["stats"]
        writer.writerow([p["platform"], p["variation"], p.get("topic", ""), p["text"],
                         s["chars"], s["hashtags"], " | ".join(s["warnings"])])
    return buf.getvalue()


def calendar_to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def calendar_to_markdown(rows: list[dict]) -> str:
    lines = ["# Content calendar\n", "| Date | Day | Platform | Format | Idea | Hook | Hashtags | CTA |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cells = [str(r.get(c, "")).replace("|", "/").replace("\n", " ") for c in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def image_ideas_to_markdown(ideas: list[dict]) -> str:
    lines = ["# Image ideas\n"]
    for n, i in enumerate(ideas, start=1):
        lines += [f"\n## {n}. {i['concept']} ({i['platform']}, {i['style']})\n",
                  f"**Prompt:** {i['prompt']}\n", f"**Alt text:** {i['alt_text']}\n",
                  f"**Suggested shape:** {i['aspect']}\n"]
    return "\n".join(lines)


def hashtags_to_text(sets: list[dict]) -> str:
    out = []
    for s in sets:
        out.append(f"{s['platform']} · {s['topic'][:60]}")
        out.append("Recommended: " + " ".join(s["groups"]["recommended"]))
        for key in ("branded", "niche", "broad"):
            if s["groups"][key]:
                out.append(f"{key.capitalize()}: " + " ".join(s["groups"][key]))
        out.append("")
    return "\n".join(out).strip()
