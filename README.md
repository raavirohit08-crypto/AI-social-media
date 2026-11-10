# 📣 Social Media Content Generator

A GenAI app that writes **brand-aware social media content**: platform-specific posts, a content calendar,
and a post improver. It reads your brand documents (brand guide, product info, past posts) with **RAG**, so the
posts use your real facts and your voice instead of generic AI text.

Built with the same stack as your Research Paper project: Hugging Face / Groq / Ollama · LangChain ·
sentence-transformers · FAISS · Streamlit.

**Live demo:** https://ai-social-media-rhuqfbwuvykscezl98d4my.streamlit.app/
**Source code:** https://github.com/raavirohit08-crypto/AI-social-media
**Demo recording:** _add your video link here before submitting_ (follow `docs/DEMO_SCRIPT.md`)

---

## 1. Features

| Page | What it does |
|---|---|
| 🏷️ **Brand & Knowledge** | Brand profile (voice, audience, words to avoid) + upload `.pdf / .txt / .md` brand documents → FAISS index |
| ✍️ **Generate Posts** | LinkedIn, Instagram, X, Facebook, short-video script. Tone, goal, language, 1-3 versions, hashtags, emojis, CTA |
| 🗓️ **Content Calendar** | 7 / 14 / 30-day plan (date, platform, format, idea, hook, hashtags, CTA) shown as a table |
| 🪄 **Improve Post** | Shorter, stronger hook, more engaging, change tone, fix grammar, add hashtags + CTA, translate |
| 🖼️ **Image Ideas** | Text-to-image prompts + accessibility alt text for a post, in a chosen visual style, with the common image shape for the platform |
| #️⃣ **Hashtag Helper** | Broad + niche hashtag suggestions, your own brand hashtags from `Knowledge/`, and a recommended set sized for the platform |
| ⬇️ **Export** | Markdown / CSV / text downloads for posts, calendar, image ideas and hashtag sets |

Important: hashtags are **AI suggestions, not live trends**. The app cannot see current social-media data, so check each tag on the platform before using it. Image prompts are written for you to paste into any text-to-image tool; the app does not generate images.

Extras: live **streaming** preview while the model writes, a guarantee that the requested number of versions is delivered (retry / top-up / clear warning), brand **avoid-word** warnings, automatic **character-limit check** (X = 280 etc.) with an automatic shorten step, hashtag / emoji counts,
Instagram "Visual idea" line, estimated seconds for video scripts, brand-source display.

## 2. How it works

```
Knowledge/*.pdf|txt|md ─► load ─► chunk ─► MiniLM embeddings ─► FAISS (vector_db/)        [done once, cached]

Generate:  topic ─► retrieve top-k brand passages ─┐
           brand profile ──────────────────────────┼─► prompt (rules + platform guidance) ─► LLM ─► post(s)
           platform rules (limits, hashtags) ──────┘                                          │
                                                          Python checks chars / hashtags ◄────┘
                                                          too long? ─► ask LLM to shorten (max 2 tries)
                                                          (shortening keeps brand profile, facts, language, tone)
           3 versions requested but fewer returned? ─► retry ─► ask for the missing ones ─► else show a warning

Calendar:  goal + brand context ─► LLM returns JSON ─► parsed + validated in Python ─► table
           (fenced / chatty / truncated JSON is handled; dates are computed in Python, not by the model)
           Checks: exact post count, no duplicates, max 2 posts per day. Incomplete? ─► retry (3 tries) ─► clear error.

Brand knowledge: Knowledge/ folder is fingerprinted. Add / change / delete a document ─► the session index is rebuilt
           automatically before the next generation. If the rebuild fails, old facts are NOT used.
```

| Concept from your GenAI notes | Where it appears |
|---|---|
| Prompt engineering: role, rules, constraints | `SYSTEM_PROMPT` in `src/generator.py` |
| Few-shot / style examples | Past posts in `Knowledge/` are retrieved as examples of your voice |
| Temperature | 0.8 for creative posts, 0.7 for the calendar, 0.5 for rewrites |
| RAG | `src/kb.py`, `src/vector_store.py`, `src/brand.py` |
| Hallucination control | "Never invent facts" rule, placeholders like `[DISCOUNT]`, facts only from brand context |
| Structured output (JSON) | `src/calendar_planner.py`, `src/creative.py` (parse + validate + retry) |
| Guardrails / responsible AI | Safety rules in the system prompt, human review reminder |
| Streamlit session state + caching | `src/state.py`, `st.cache_resource` |
| Streaming responses | `LLM.chat_stream` / `ask_streaming` in `src/llm.py`, live preview in `pages/2_Generate_Posts.py` |
| Output validation | variations in `src/generator.py`, calendar in `src/calendar_planner.py` |
| Evaluation with a real model | `evaluation/` (cases, automatic checks, report generator) |

## 3. Project structure

```
AI-social-media/                      # the GitHub repository (this folder)
├── app.py                    # home page (run this)
├── cli.py                    # command-line version
├── pages/                    # 1_Brand_Knowledge, 2_Generate_Posts, 3_Content_Calendar, 4_Improve_Post, 5_Image_Ideas, 6_Hashtag_Helper, 7_Export
├── src/
│   ├── config.py   llm.py    # settings; Hugging Face / Groq / Ollama client with retry, clear errors and streaming
│   ├── loader.py   vector_store.py   kb.py   brand.py     # knowledge base + brand profile + retrieval + avoid-word check
│   ├── platforms.py          # platform rules + post analyzer
│   ├── creative.py           # image-prompt ideas + hashtag suggestions
│   ├── generator.py   calendar_planner.py   improver.py   exporter.py
│   └── state.py              # shared Streamlit session logic (incl. automatic knowledge refresh)
├── Knowledge/                # SAMPLE brand (fictional café). Replace with your own documents.
├── evaluation/               # real-model evaluation: cases.json, checks.py, run_eval.py, results/
├── docs/DEMO_SCRIPT.md       # what to show in the demo / recording
├── tests/test_pipeline.py    # 48 offline tests
├── requirements.txt          # app dependencies
├── requirements-dev.txt      # + pytest and reportlab (needed by the tests)
└── .env.example   .gitignore
```

## 4. Setup (Windows)

```
git clone https://github.com/raavirohit08-crypto/AI-social-media.git
cd AI-social-media
pip install -r requirements.txt          # a virtual environment is optional
copy .env.example .env                   # then edit .env
streamlit run app.py
```

Edit `.env` so it contains only the option you use (no quotes, no spaces around `=`), for example Groq:

```
HF_TOKEN=your_groq_key
LLM_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=openai/gpt-oss-120b
```

Get a free Groq key at console.groq.com → API Keys. Restart Streamlit after changing `.env`.
**Never commit `.env` or share your key, and never put a real key in `.env.example`** (it must only contain placeholders
such as `your_api_key_here`). If a key leaks, revoke it in the provider's console, create a new one, and remove the old one from git history.
A test (`test_no_real_credentials_in_repository_files`) fails if a key-like value is committed.

## 5. Using the app

1. **Brand & Knowledge**: save your profile, replace the sample files in `Knowledge/` with your own, click **Build / refresh**
   (the first run downloads the embedding model). This step is optional but gives much better posts.
2. **Generate Posts**: write the topic, pick platforms, tone and goal, click **Generate**. Use the copy icon on each result.
3. **Content Calendar**: choose length, platforms, posts per week, optional theme.
4. **Improve Post**: paste your own draft and choose an action.
5. **Export**: download Markdown / CSV.

Try with the sample brand: *"Launching our Salted Caramel Cold Brew this Friday"* on LinkedIn, Instagram and X.
The posts should mention the real (sample) price and café details from `Knowledge/` and stay within X's 280 characters.

## 6. Testing

```
pip install -r requirements-dev.txt   # app dependencies + pytest + reportlab
python -m pytest -q                   # or: python tests/test_pipeline.py
```
48 tests run with a fake LLM and fake embeddings (no key, no internet): platform analyzer, character-limit shortening
(brand-aware), prompt guardrails, **complete-variation validation (retry / top-up / incomplete warning)**, JSON parsing of
messy model output, **calendar completeness (count, duplicates, max per day, retry)**, knowledge-base caching,
**stale-index refresh after adding/deleting documents**, brand profile and avoid-words, rewrites that keep the brand profile,
streaming client, image-prompt and hashtag logic, exporters, the LLM client's error handling, the evaluation checks, and a
credential scan of the repository.

### Real-model evaluation (at least 10 examples)

The offline tests prove the logic; `evaluation/` shows the quality of real model output:

```
python evaluation/run_eval.py            # runs 12 cases against your real model (needs HF_TOKEN in .env)
```
It writes `evaluation/results/eval_<time>.md` and `.json` with, for every case, the input, the output, platform-format checks,
factual checks (every number/price must come from the brand documents or the request; required facts present; forbidden claims
absent), distinct-version checks and tone proxies. Read each output and fill in the **Tone score (1-5)** line, then commit the
report. Cases include two traps: a sale with no facts given (no invented discount) and a health claim (must not be repeated).

## 7. Deploy on Streamlit Community Cloud

Push to GitHub (without `.env`; use only a NEW key, never one that was ever committed) → share.streamlit.io → main file `app.py` → **Advanced settings → Secrets** (TOML, quotes required):
```toml
HF_TOKEN = "your_groq_key"
LLM_BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "openai/gpt-oss-120b"
```
Files saved on the server (profile, uploads, index) disappear when the app restarts, so commit your brand documents
to `Knowledge/` in the repo. The index rebuilds automatically on the next visit.

## 8. Troubleshooting

| Message | Meaning / fix |
|---|---|
| 402 out of credits | Hugging Face credits used up → switch to Groq or Ollama in `.env` |
| 401 key rejected | Wrong or revoked key |
| 404 model not found | Model name retired → pick a current one from your provider's model list |
| 429 / rate limit | Free-tier limit → wait a minute, or use `openai/gpt-oss-20b` |
| "model returned an empty response" | Reasoning model used its whole token budget; `src/llm.py` already adds headroom, make sure it is the latest file |

## 9. Limitations (be honest in your viva)

- Streaming shows the first answer live; automatic retries (missing versions, shortening) are not streamed.
- The app **writes** content; it does not post to social networks, generate images, or know your real analytics.
- Character counts are exact for plain text, but platforms count some things differently (for example, X counts links as a fixed length).
- A language model can still make mistakes. The prompt forbids inventing facts, but **always review before publishing**.
- Posting-time advice is not included because it cannot be verified from your data.
- Quality depends on your brand documents and on the hosted model; free tiers have rate limits.

## 10. Ideas to extend

Direct scheduling via a platform API · calling an image-generation API from the Image Ideas page · A/B scoring of variations · sentiment / brand-safety check ·
real hashtag-trend data from a platform or analytics API · analytics upload to learn what performed · multi-brand workspaces.

## 11. Likely viva questions

- *Why RAG instead of fine-tuning for brand voice?* Cheaper, instantly updatable (drop in a new file), and facts stay traceable to documents.
- *How do you stop the model inventing offers or prices?* Rule in the system prompt, facts only from retrieved context, placeholders when missing, human review.
- *Why check character limits in Python?* LLMs are poor at counting characters; code is exact, and the model is only asked to rewrite when a limit is exceeded.
- *Why compute calendar dates in code?* Models make date mistakes; we ask only for a day number and compute the date ourselves.
- *How do you handle bad JSON?* Strip code fences, extract the array, salvage complete objects from truncated output, retry once, then show a clear error.
- *Why a higher temperature here than in the summarizer?* Social copy benefits from variety; summaries need accuracy.
- *Why not show "trending hashtags"?* A language model has no live data, so it would only be guessing. The app labels hashtags as suggestions, reuses your own brand tags, and a real trend feature would need a platform API.
- *Why do image prompts avoid readable text?* Text-to-image tools often misspell words, so text is better added afterwards in an editor.
- *Why is the knowledge base optional?* The app still works from the profile and topic alone, but retrieval makes posts specific.
