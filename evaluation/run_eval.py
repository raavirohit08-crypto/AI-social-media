"""Run the evaluation cases against the REAL model and write a report you can submit.

    python evaluation/run_eval.py                # real model (needs HF_TOKEN in .env)
    python evaluation/run_eval.py --limit 3      # only the first 3 cases (cheap smoke test)
    python evaluation/run_eval.py --offline      # scripted fake model: only proves the harness runs

Output: evaluation/results/eval_<time>.md and .json  (commit these files).
Automatic checks cover format and facts; you still fill in the tone score (1-5) by reading each output.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.checks import check_variations, evaluate_post          # noqa: E402
from src.brand import PROFILE_FIELDS, get_brand_context, profile_to_text  # noqa: E402
from src.generator import generate_posts                                # noqa: E402

CASES_FILE = Path(__file__).with_name("cases.json")
RESULTS_DIR = Path(__file__).with_name("results")


class OfflineModel:
    """Scripted stand-in so the harness can be tested without a key. NOT a real result."""
    def ask(self, system_prompt, user_prompt, **kw):
        if "VARIATION" in user_prompt:
            return "\n".join(f"=== VARIATION {n} ===\nOffline sample {n} about cold brew Rs 140 #ColdBrew #GreenLeafCoffee #ArakuCoffee"
                             for n in (1, 2, 3))
        return "Offline sample about cold brew Rs 140 #ColdBrew #GreenLeafCoffee #ArakuCoffee\nVisual idea: iced glass"


def load_context(topic: str, offline: bool, settings, store):
    if offline:
        text = "\n\n".join(p.read_text(encoding="utf-8") for p in sorted((ROOT / "Knowledge").glob("*.[mt][dx]*")))
        return text, ["(offline: all documents)"]
    context, passages = get_brand_context(store, topic, settings.top_k)
    return context, sorted({p.source for p in passages})


def run(limit: int | None, offline: bool) -> tuple[list[dict], str]:
    data = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    profile_text = profile_to_text({k: data["brand_profile"].get(k, "") for k in PROFILE_FIELDS})
    cases = data["cases"][:limit] if limit else data["cases"]

    settings = store = None
    if offline:
        llm, model_name = OfflineModel(), "OFFLINE FAKE MODEL (not a real result)"
    else:
        from src.config import load_settings
        from src.kb import build_or_load
        from src.llm import LLM
        from src.vector_store import get_embeddings
        settings = load_settings()
        llm, model_name = LLM(settings.hf_token, settings.model_name), settings.model_name
        store, _, _ = build_or_load(get_embeddings(), settings)

    results = []
    for case in cases:
        print(f"Case {case['id']}: {case['name']} ...", flush=True)
        context, sources = load_context(case["topic"], offline, settings, store)
        allowed = "\n".join([profile_text, context, case["topic"]])
        record = {"case": case, "sources": sources, "error": None, "posts": []}
        try:
            posts = generate_posts(
                llm, case["topic"], case["platform"], tone=case.get("tone", "Friendly"),
                audience=data["brand_profile"].get("audience", ""), goal=case.get("goal", "Engagement"),
                language=case.get("language", "English"), variations=case.get("variations", 1),
                profile_text=profile_text, brand_context=context, sources=sources)
            record["variation_checks"] = [{"name": n, "passed": ok, "detail": d} for n, ok, d in
                                          check_variations([p.text for p in posts], case.get("variations", 1))]
            for post in posts:
                record["posts"].append({"text": post.text, "notice": post.notice,
                                        **evaluate_post(post.text, case["platform"], allowed, case, profile_text)})
        except Exception as error:                      # keep going: one failure must not hide the others
            record["error"] = str(error)
        results.append(record)
    return results, model_name


def to_markdown(results: list[dict], model_name: str) -> str:
    total = sum(len(r["posts"]) for r in results)
    passed = sum(p["all_passed"] for r in results for p in r["posts"])
    out = [f"# Real-model evaluation: Social Media Content Generator\n",
           f"- Model: `{model_name}`\n- Run: {datetime.now():%Y-%m-%d %H:%M}\n- Cases: {len(results)} · outputs: {total} · "
           f"outputs passing every automatic check: {passed}/{total}\n",
           "Automatic checks cover platform format and facts. **Tone consistency must be rated by a person** "
           "(1 = off-brand, 5 = sounds exactly like the brand): fill in the `Tone score` lines.\n"]
    for r in results:
        c = r["case"]
        out.append(f"\n---\n\n## Case {c['id']}: {c['name']}\n")
        out.append(f"**Input:** platform = {c['platform']} · tone = {c.get('tone', 'Friendly')} · goal = {c.get('goal', 'Engagement')} · "
                   f"language = {c.get('language', 'English')} · versions = {c.get('variations', 1)}\n\n**Topic:** {c['topic']}\n")
        if c.get("note"):
            out.append(f"**What this tests:** {c['note']}\n")
        out.append(f"**Brand facts retrieved from:** {', '.join(r['sources']) or 'none'}\n")
        if r["error"]:
            out.append(f"\n❌ **Error:** {r['error']}\n")
            continue
        for v in r.get("variation_checks", []):
            out.append(f"- {'✅' if v['passed'] else '❌'} {v['name']} {v['detail']}")
        for n, p in enumerate(r["posts"], start=1):
            out.append(f"\n### Output {n}\n\n```\n{p['text']}\n```\n")
            if p["notice"]:
                out.append(f"⚠️ {p['notice']}\n")
            for chk in p["checks"]:
                out.append(f"- {'✅' if chk['passed'] else '❌'} {chk['name']} {('· ' + chk['detail']) if chk['detail'] else ''}")
            t = p["tone_proxies"]
            out.append(f"- Tone proxies: avg sentence {t['avg_sentence_words']} words · {t['emojis']} emoji(s)")
            out.append("- **Tone score (1-5, fill in): ___**   Notes: ___")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="run only the first N cases")
    parser.add_argument("--offline", action="store_true", help="use a scripted fake model (harness test only)")
    args = parser.parse_args()
    results, model_name = run(args.limit, args.offline)
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + ("_OFFLINE" if args.offline else "")
    (RESULTS_DIR / f"eval_{stamp}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    report = RESULTS_DIR / f"eval_{stamp}.md"
    report.write_text(to_markdown(results, model_name), encoding="utf-8")
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    main()
