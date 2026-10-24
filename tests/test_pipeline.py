"""Offline tests: no internet, no API key needed.

Run with:  python -m pytest -q      (or)   python tests/test_pipeline.py
"""
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from langchain_core.embeddings import Embeddings

from src import brand, kb
from src.calendar_planner import normalize_plan, parse_json_items, plan_calendar
from src.config import Settings
from src.creative import (ASPECT, IMAGE_SYSTEM, clean_tags, hashtag_ideas, image_prompts,
                          parse_json_object, recommended_set)
from src.exporter import (calendar_to_csv, calendar_to_markdown, hashtags_to_text, image_ideas_to_markdown,
                          posts_to_csv, posts_to_markdown)
from src.generator import SYSTEM_PROMPT, enforce_limit, generate_posts, split_variations
from src.improver import improve_post
from src.loader import load_file
from src.platforms import analyze_post, extract_hashtags, split_visual_idea
from src.vector_store import build_vector_store, split_documents

ROOT = Path(__file__).resolve().parent.parent


class FakeEmbeddings(Embeddings):
    """Bag-of-words hashing -> normalized vectors. Similar words => similar vectors."""
    dim = 256

    def _vec(self, text):
        v = np.zeros(self.dim, dtype="float32")
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    """Returns scripted replies (a list or a function) and records every prompt."""
    def __init__(self, replies="ok"):
        self.replies, self.calls = replies, []

    def ask(self, system_prompt, user_prompt, **kw):
        self.calls.append((system_prompt, user_prompt, kw))
        if callable(self.replies):
            return self.replies(user_prompt)
        if isinstance(self.replies, list):
            return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return self.replies


# ------------------------------------------------------------------ platforms
def test_analyze_post_counts_and_limits():
    s = analyze_post("Fresh cold brew is here ☕🌱 #ColdBrew #coldbrew #GreenLeaf", "X (Twitter)")
    assert s["hashtags"] == 2 and s["emojis"] == 2 and not s["over_limit"]
    long = analyze_post("a" * 300, "X (Twitter)")
    assert long["over_limit"] and "Over the 280" in long["warnings"][0]
    assert analyze_post("a" * 3500, "LinkedIn")["over_limit"]


def test_hashtag_extraction_ignores_numbers_and_duplicates():
    assert extract_hashtags("Top #1 pick #Coffee #coffee #Tea_Time") == ["#Coffee", "#Tea_Time"]


def test_visual_idea_not_counted():
    body, visual = split_visual_idea("Great morning! #Coffee\nVisual idea: latte art close-up")
    assert visual == "latte art close-up" and "Visual idea" not in body
    s = analyze_post("Hi\nVisual idea: " + "x" * 3000, "Instagram")
    assert s["chars"] == 2 and not s["over_limit"]


def test_script_has_seconds_estimate_and_no_limit():
    s = analyze_post("word " * 100, "Short video script (Reels/Shorts)")
    assert s["limit"] is None and s["seconds"] == 40 and not s["over_limit"]


# ------------------------------------------------------------------ generator
def test_split_variations():
    raw = "=== VARIATION 1 ===\nFirst post\n=== VARIATION 2 ===\nSecond post\n=== VARIATION 3 ===\nThird"
    assert split_variations(raw, 2) == ["First post", "Second post"]
    assert split_variations("no markers here", 3) == ["no markers here"]


def test_enforce_limit_asks_model_to_shorten():
    llm = FakeLLM(["short enough #x"])
    out = enforce_limit(llm, "b" * 400, "X (Twitter)")
    assert out == "short enough #x" and len(llm.calls) == 1
    assert "under 265" in llm.calls[0][1]
    llm2 = FakeLLM("c" * 50)                       # already fine -> no model call
    assert enforce_limit(llm2, "c" * 50, "X (Twitter)") == "c" * 50 and not llm2.calls


def test_enforce_limit_gives_up_but_does_not_truncate():
    llm = FakeLLM("d" * 500)                       # model never complies
    out = enforce_limit(llm, "d" * 500, "X (Twitter)", attempts=2)
    assert len(out) == 500 and len(llm.calls) == 2


def test_generate_posts_prompt_has_guardrails_and_brand_context():
    llm = FakeLLM("=== VARIATION 1 ===\nPost A #x\n=== VARIATION 2 ===\nPost B #y")
    posts = generate_posts(llm, "Launching cold brew", "Instagram", variations=2,
                           profile_text="- Brand: GreenLeaf", brand_context="[1] Cold brew costs Rs 140",
                           sources=["products.md"])
    assert [p.text for p in posts] == ["Post A #x", "Post B #y"]
    assert posts[0].sources == ["products.md"] and posts[0].stats["hashtags"] == 1
    system, user, kw = llm.calls[0]
    assert "NEVER invent" in system and system == SYSTEM_PROMPT
    assert "Cold brew costs Rs 140" in user and "GreenLeaf" in user and "Visual idea" in user
    assert kw["temperature"] >= 0.7                # creative task


def test_generate_posts_x_is_shortened_automatically():
    llm = FakeLLM(["z" * 400, "Short punchy post #ColdBrew"])
    post = generate_posts(llm, "cold brew", "X (Twitter)")[0]
    assert post.text == "Short punchy post #ColdBrew" and not post.stats["over_limit"]


def test_generate_posts_validation():
    llm = FakeLLM()
    for kwargs in ({"topic": "  ", "platform": "LinkedIn"}, {"topic": "x", "platform": "MySpace"}):
        try:
            generate_posts(llm, **kwargs)
            assert False
        except ValueError:
            pass
    assert not llm.calls


# ------------------------------------------------------------------ calendar
def test_parse_json_items_handles_messy_replies():
    clean = '[{"day": 1, "idea": "a"}]'
    assert parse_json_items(clean) == [{"day": 1, "idea": "a"}]
    fenced = "Sure! Here you go:\n```json\n" + clean + "\n```\nHope it helps"
    assert parse_json_items(fenced) == [{"day": 1, "idea": "a"}]
    truncated = '[{"day": 1, "idea": "a"}, {"day": 2, "idea": "b", "hashtags": ["#x"]}, {"day": 3, "ide'
    assert [d["day"] for d in parse_json_items(truncated)] == [1, 2]
    assert parse_json_items("no json at all") == []


def test_normalize_plan_validates_and_computes_dates():
    items = [
        {"day": 2, "platform": "linkedin", "idea": "Behind the scenes", "hashtags": ["Coffee", "#Roast"]},
        {"day": 99, "platform": "LinkedIn", "idea": "out of range"},
        {"day": 1, "platform": "TikTok", "idea": "not selected"},
        {"day": "x", "platform": "LinkedIn", "idea": "bad day"},
        {"day": 1, "platform": "LinkedIn", "idea": ""},
        {"day": 1, "platform": "Instagram", "idea": "Launch teaser", "hashtags": "#a #b"},
    ]
    rows = normalize_plan(items, date(2026, 10, 23), 7, ["LinkedIn", "Instagram"])
    assert len(rows) == 2
    assert rows[0]["date"] == "2026-10-23" and rows[0]["day"] == "Friday" and rows[0]["platform"] == "Instagram"
    assert rows[0]["hashtags"] == "#a #b"
    assert rows[1]["date"] == "2026-10-24" and rows[1]["platform"] == "LinkedIn"
    assert rows[1]["hashtags"] == "#Coffee #Roast"


def test_plan_calendar_retries_once_on_invalid_json():
    good = json.dumps([{"day": 1, "platform": "LinkedIn", "idea": "Intro post", "hook": "Hi",
                        "hashtags": ["#a"], "cta": "Follow", "format": "text"}])
    llm = FakeLLM(["Sorry, I cannot do JSON", good])
    rows = plan_calendar(llm, "Engagement", ["LinkedIn"], days=7, start=date(2026, 10, 23))
    assert len(rows) == 1 and len(llm.calls) == 2
    assert "not valid JSON" in llm.calls[1][1]
    try:
        plan_calendar(FakeLLM("never json"), "Engagement", ["LinkedIn"], start=date(2026, 10, 23))
        assert False
    except ValueError:
        pass
    try:
        plan_calendar(FakeLLM(), "Engagement", [])
        assert False
    except ValueError:
        pass


def test_plan_calendar_asks_for_right_number_of_posts():
    llm = FakeLLM("[]")
    try:
        plan_calendar(llm, "Sales", ["LinkedIn"], days=14, posts_per_week=3, start=date(2026, 10, 23))
    except ValueError:
        pass
    assert "exactly 6 posts" in llm.calls[0][1]


# ------------------------------------------------------------------ improver
def test_improve_post_actions_and_validation():
    llm = FakeLLM("Better post #x")
    r = improve_post(llm, "my draft", "Translate", "LinkedIn", language="Hindi")
    assert r["text"] == "Better post #x" and "Target language: Hindi" in llm.calls[0][1]
    assert "Do not add facts" in llm.calls[0][1]
    r = improve_post(FakeLLM("t" * 400), "x", "Make it shorter", "X (Twitter)")
    assert r["stats"]["chars"] == 400                      # could not shorten -> reported honestly
    for bad in (("", "Translate"), ("draft", "Nonsense")):
        try:
            improve_post(FakeLLM(), bad[0], bad[1], "LinkedIn")
            assert False
        except ValueError:
            pass


# ------------------------------------------------------------------ knowledge base
def _kb_dirs():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "Knowledge").mkdir()
    return tmp / "Knowledge", tmp / "vector_db"


def test_loader_reads_txt_md_and_pdf():
    know, _ = _kb_dirs()
    (know / "a.md").write_text("# Title\n\nHello brand", encoding="utf-8")
    (know / "b.txt").write_text("plain text", encoding="utf-8")
    (know / "empty.txt").write_text("   ", encoding="utf-8")
    assert load_file(str(know / "a.md"))[0].metadata == {"source_name": "a.md", "page": 0}
    assert load_file(str(know / "empty.txt")) == []
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(know / "c.pdf"))
    c.drawString(72, 700, "Pdf brand facts about tea")
    c.save()
    assert "tea" in load_file(str(know / "c.pdf"))[0].page_content
    try:
        load_file(str(know / "x.docx"))
        assert False
    except ValueError:
        pass


def test_kb_index_is_cached_and_rebuilt_on_change():
    know, vec = _kb_dirs()
    (know / "brand.md").write_text("We sell cold brew coffee in Vizag", encoding="utf-8")
    (know / "empty.txt").write_text("", encoding="utf-8")
    (know / "ignore.png").write_bytes(b"\x89PNG")

    class Counting(FakeEmbeddings):
        calls = 0
        def embed_documents(self, texts):
            Counting.calls += 1
            return super().embed_documents(texts)

    s = Settings()
    store, loaded, skipped = kb.build_or_load(Counting(), s, knowledge=know, vector=vec)
    assert store and list(loaded) == ["brand.md"] and skipped == ["empty.txt"] and Counting.calls == 1
    assert kb.is_index_current(know, vec)
    kb.build_or_load(Counting(), s, knowledge=know, vector=vec)
    assert Counting.calls == 1                              # loaded from disk
    (know / "more.txt").write_text("new product info", encoding="utf-8")
    assert not kb.is_index_current(know, vec)
    kb.build_or_load(Counting(), s, knowledge=know, vector=vec)
    assert Counting.calls == 2                              # rebuilt after a change
    empty_know, empty_vec = _kb_dirs()
    assert kb.build_or_load(FakeEmbeddings(), s, knowledge=empty_know, vector=empty_vec)[0] is None


def test_sample_knowledge_folder_retrieval():
    docs = []
    for path in sorted((ROOT / "Knowledge").glob("*")):
        docs += load_file(str(path))
    assert len(docs) == 3
    store = build_vector_store(split_documents(docs, 700, 100), FakeEmbeddings())
    context, passages = brand.get_brand_context(store, "price of salted caramel cold brew", k=3)
    assert passages and "products_and_offers.md" in {p.source for p in passages}
    assert all(-1 <= p.score <= 1 for p in passages) and context.startswith("[1]")
    assert brand.get_brand_context(None, "x") == ("", [])


def test_brand_profile_roundtrip():
    path = Path(tempfile.mkdtemp()) / "profile.json"
    assert brand.profile_to_text(brand.load_profile(path)) == ""      # missing file -> empty
    brand.save_profile({"name": "GreenLeaf", "voice": "warm", "junk": "ignored"}, path)
    loaded = brand.load_profile(path)
    assert loaded["name"] == "GreenLeaf" and "junk" not in loaded
    text = brand.profile_to_text(loaded)
    assert "GreenLeaf" in text and "warm" in text and "Industry" not in text
    path.write_text("not json")
    assert brand.load_profile(path)["name"] == ""                      # corrupt file -> defaults


# ------------------------------------------------------------------ exporter
def test_exporters():
    posts = [{"platform": "X (Twitter)", "variation": 1, "topic": "launch", "text": 'Hi, "friends" #a',
              "stats": analyze_post('Hi, "friends" #a', "X (Twitter)")}]
    assert "## X (Twitter) · version 1" in posts_to_markdown(posts)
    rows = list(__import__("csv").DictReader(posts_to_csv(posts).splitlines()))
    assert rows[0]["text"] == 'Hi, "friends" #a'                      # quotes/commas survive CSV
    cal = [{"date": "2026-10-23", "day": "Friday", "platform": "LinkedIn", "format": "text",
            "idea": "a | b", "hook": "h", "hashtags": "#a", "cta": "c", "extra": "dropped"}]
    assert calendar_to_csv(cal).splitlines()[0] == "date,day,platform,format,idea,hook,hashtags,cta"
    assert "a / b" in calendar_to_markdown(cal)


# ------------------------------------------------------------------ llm wrapper
def test_llm_fails_fast_on_billing_error_and_retries_others():
    import src.llm as mod
    from src.llm import LLM, LLMError

    class Err(Exception):
        def __init__(self, code): self.status_code = code

    class FakeClient:
        def __init__(self, errors):
            self.errors, self.calls = list(errors), []
            self.chat = self.completions = self
        def create(self, **kw):
            self.calls.append(kw)
            if self.errors:
                raise self.errors.pop(0)
            return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": " ok "})()})()]})

    mod.time.sleep = lambda s: None
    llm = LLM("tok", "m")
    llm.client = FakeClient([Err(402)])
    try:
        llm.chat([{"role": "user", "content": "hi"}])
        assert False
    except LLMError as e:
        assert "credits" in str(e) and len(llm.client.calls) == 1
    llm.client = FakeClient([Err(503), Err(503)])
    assert llm.chat([{"role": "user", "content": "hi"}]) == "ok" and len(llm.client.calls) == 3
    # gpt-oss on an OpenAI-compatible server: low reasoning + token headroom
    g = LLM("k", "openai/gpt-oss-120b", base_url="http://localhost:1/v1")
    g.client = FakeClient([])
    g.chat([{"role": "user", "content": "q"}], max_tokens=300)
    kw = g.client.calls[0]
    assert kw["max_tokens"] == 1300 and kw["extra_body"] == {"reasoning_effort": "low"}
    assert type(LLM("key", "llama", base_url="http://localhost:1/v1").client).__module__.startswith("openai")


# ------------------------------------------------------------------ image ideas
def test_parse_json_object_handles_messy_replies():
    assert parse_json_object('{"broad": ["#a"]}') == {"broad": ["#a"]}
    assert parse_json_object('Sure!\n```json\n{"niche": ["#b"]}\n```\nDone') == {"niche": ["#b"]}
    assert parse_json_object("no json") == {} and parse_json_object('{"cut": [') == {}


def test_image_prompts_adds_python_side_fields_and_validates():
    reply = json.dumps([
        {"concept": "Pour-over morning", "prompt": "A steaming pour-over on a wooden table, soft window light",
         "alt_text": "A" * 300},
        {"concept": "No prompt here", "prompt": "  "},
        {"concept": "Cold brew bottle", "prompt": "Cold brew bottle with ice, bright background"},
        {"concept": "Third", "prompt": "Third prompt"},
    ])
    llm = FakeLLM(reply)
    ideas = image_prompts(llm, "Cold brew launches Friday", "Instagram", "Flat illustration", count=2,
                          profile_text="- Brand: GreenLeaf", brand_context="[1] Cold brew costs Rs 140")
    assert [i["concept"] for i in ideas] == ["Pour-over morning", "Cold brew bottle"]   # blank dropped, count capped
    assert len(ideas[0]["alt_text"]) == 125                                            # accessibility text capped
    assert ideas[0]["aspect"] == ASPECT["Instagram"] and ideas[0]["style"] == "Flat illustration"
    system, user, _ = llm.calls[0]
    assert system == IMAGE_SYSTEM and "copyrighted" in system
    assert "Rs 140" in user and "GreenLeaf" in user and "Flat illustration" in user


def test_image_prompts_retry_and_errors():
    good = json.dumps([{"concept": "c", "prompt": "p"}])
    llm = FakeLLM(["not json", good])
    assert len(image_prompts(llm, "post", "LinkedIn")) == 1 and "not valid JSON" in llm.calls[1][1]
    for args in (("", "LinkedIn"), ("post", "MySpace")):
        try:
            image_prompts(FakeLLM(good), *args)
            assert False
        except ValueError:
            pass
    try:
        image_prompts(FakeLLM("never json"), "post", "LinkedIn")
        assert False
    except ValueError:
        pass


# ------------------------------------------------------------------ hashtags
def test_clean_tags():
    tags = clean_tags(["coffee", "#Coffee", "#cold brew!", "#2026", "##", "", "x" * 40, "#Tea_Time"])
    assert tags == ["#coffee", "#coldbrew", "#Tea_Time"]
    assert clean_tags("#a #b #a") == ["#a", "#b"]
    assert clean_tags(["#A", "#b"], exclude={"#a"}) == ["#b"]
    assert len(clean_tags([f"t{i}" for i in range(30)], limit=5)) == 5


def test_hashtag_ideas_uses_brand_tags_from_documents():
    llm = FakeLLM(json.dumps({"broad": ["#coffee", "#GreenLeafCoffee"], "niche": ["#coldbrewlovers", "#coffee"]}))
    groups = hashtag_ideas(llm, "cold brew launch", "LinkedIn",
                           profile_text="", brand_context="Our tags: #GreenLeafCoffee #ArakuCoffee")
    assert groups["branded"] == ["#GreenLeafCoffee", "#ArakuCoffee"]       # from the user's documents
    assert "#GreenLeafCoffee" not in groups["broad"]                        # no duplicates of brand tags
    assert groups["niche"] == ["#coldbrewlovers"]                           # '#coffee' already in broad
    assert groups["recommended"][:2] == ["#GreenLeafCoffee", "#ArakuCoffee"]
    assert len(groups["recommended"]) <= 5                                  # LinkedIn maximum


def test_recommended_set_respects_platform_limits():
    groups = {"branded": ["#b1", "#b2", "#b3"], "niche": ["#n1", "#n2"], "broad": ["#r1"]}
    assert recommended_set(groups, "X (Twitter)") == ["#b1", "#b2"]        # X: max 2
    assert recommended_set(groups, "Instagram") == ["#b1", "#b2", "#n1", "#n2", "#r1"]
    assert recommended_set({"branded": [], "niche": [], "broad": []}, "LinkedIn") == []


def test_hashtag_ideas_retry_and_validation():
    good = json.dumps({"broad": ["#a"], "niche": ["#b"]})
    llm = FakeLLM(["oops", good])
    assert hashtag_ideas(llm, "topic", "Instagram")["recommended"] == ["#b", "#a"] and len(llm.calls) == 2
    for fake, topic, platform in ((FakeLLM("never"), "t", "LinkedIn"), (FakeLLM(good), " ", "LinkedIn"),
                                  (FakeLLM(good), "t", "MySpace")):
        try:
            hashtag_ideas(fake, topic, platform)
            assert False
        except ValueError:
            pass


def test_sample_brand_hashtags_are_found():
    from src.platforms import extract_hashtags
    text = (ROOT / "Knowledge" / "brand_guide.md").read_text(encoding="utf-8")
    assert "#GreenLeafCoffee" in clean_tags(extract_hashtags(text))


def test_creative_exporters():
    ideas = [{"concept": "Pour-over", "prompt": "A pour-over", "alt_text": "Coffee", "aspect": "4:5",
              "style": "Photorealistic", "platform": "Instagram"}]
    md = image_ideas_to_markdown(ideas)
    assert "Pour-over" in md and "**Prompt:** A pour-over" in md and "4:5" in md
    txt = hashtags_to_text([{"platform": "X (Twitter)", "topic": "cold brew launch",
                             "groups": {"recommended": ["#a", "#b"], "branded": ["#a"], "niche": ["#b"], "broad": []}}])
    assert "Recommended: #a #b" in txt and "Broad" not in txt


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\nAll {len(tests)} tests passed")
