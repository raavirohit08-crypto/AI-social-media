# Corrections made after the project review

| # | Reviewer correction | What changed | Where |
|---|---|---|---|
| 1 | Replace the credential in `.env.example` | Now only placeholders (`HF_TOKEN=your_api_key_here`). A test scans the repo for key-like values. **You must still do the manual steps below.** | `.env.example`, `tests/` |
| 2 | Validate the number of variations | Markers are parsed strictly (chatter before the first marker and empty/duplicate versions dropped). Missing versions -> retry with a reminder -> ask for each missing one separately -> otherwise the result carries a visible "Incomplete result" warning. | `src/generator.py`, `pages/2_Generate_Posts.py` |
| 3 | Validate calendar completeness | Exact post count, duplicate removal (same idea / same date+platform+hook), max 2 posts per day. Incomplete answers are retried (3 tries) with the reason; if still incomplete an `IncompleteCalendarError` is shown (partial rows in an expander, never saved as a full plan). | `src/calendar_planner.py`, `pages/3_Content_Calendar.py` |
| 4 | Stop outdated brand info | The session index is fingerprinted against `Knowledge/`. Adding/deleting a document rebuilds it automatically (on upload, on delete, on every page load and right before generating). If the rebuild fails the old index is cleared; generation with brand knowledge is blocked, never run on old facts. | `src/state.py`, all pages |
| 5 | Carry brand restrictions into rewriting | New `PostSettings` (profile, brand facts, language, tone, audience, emojis) is passed to the automatic shortening. Improve Post now sends the saved brand profile (+ optional brand facts) and keeps the draft's language unless "Translate". Python also flags the brand's "avoid" words. | `src/generator.py`, `src/improver.py`, `src/brand.py`, `pages/4_Improve_Post.py` |
| 6 | Streaming + real-model evaluation | `LLM.chat_stream` / `ask_streaming` (retry before the first token, fallback if a provider does not stream); live preview on Generate Posts. `evaluation/` holds 12 cases, automatic checks and a report generator. | `src/llm.py`, `evaluation/` |
| 7 | Test setup and docs | `requirements-dev.txt` (pytest, reportlab); README folder structure, setup, testing, demo-link section; `docs/DEMO_SCRIPT.md`. | repo root, `README.md` |

## Things only you can do (cannot be done inside a zip)

1. **Revoke the exposed key now** in the Hugging Face (or Groq) console and create a new one. Treat the old one as public.
2. **Remove it from git history** (deleting the line in a new commit is not enough). Easiest: install `git-filter-repo`, then
   ```
   git filter-repo --replace-text <(echo 'THE_OLD_KEY==>REMOVED')     # or use BFG Repo-Cleaner
   git push --force origin main
   ```
   If this feels risky, create a fresh GitHub repository from this folder (no old history) and resubmit that link.
3. Put the **new** key only in `.env` (local) and in Streamlit Cloud **Secrets** (deployment). Redeploy.
4. Run `python evaluation/run_eval.py` with your real key, read the outputs, fill in the tone scores, and commit `evaluation/results/`.
5. Record the demo (see `docs/DEMO_SCRIPT.md`) and add the link to `README.md`.
