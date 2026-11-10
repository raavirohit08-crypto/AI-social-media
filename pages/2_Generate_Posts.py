import streamlit as st

st.set_page_config(page_title="Generate Posts", page_icon="✍️", layout="wide")

from src.generator import GOALS, LANGUAGES, TONES, generate_posts
from src.llm import LLMError
from src.platforms import PLATFORMS
from src.state import bootstrap, brand_inputs, get_llm, require_token, sync_knowledge_base

settings = bootstrap()
require_token(settings)
sync_knowledge_base(settings)
st.title("✍️ Generate Posts")

has_kb = st.session_state.store is not None
topic = st.text_area("What is the post about?", height=100,
                     placeholder="e.g. Launching our new cold-brew coffee this Friday")
c1, c2 = st.columns([2, 1])
chosen = c1.multiselect("Platforms", list(PLATFORMS), default=["LinkedIn", "Instagram"])
variations = c2.slider("Versions per platform", 1, 3, 1)

c1, c2, c3, c4 = st.columns(4)
goal = c1.selectbox("Goal", GOALS, index=1)
tone = c2.selectbox("Tone", TONES)
language = c3.selectbox("Language", LANGUAGES)
audience = c4.text_input("Audience", value=st.session_state.profile.get("audience", ""))

c1, c2, c3, c4 = st.columns(4)
emojis = c1.checkbox("Use emojis", value=True)
cta = c2.checkbox("Include call to action", value=True)
hashtags = c3.slider("Hashtags", 0, 10, 3)
use_kb = c4.checkbox("Use brand knowledge", value=has_kb, disabled=not has_kb,
                     help="Retrieves facts and example posts from your brand documents.")
if not has_kb:
    st.caption("ℹ️ No brand knowledge base built yet. Posts will rely on the profile and your topic only.")

if st.button("Generate", type="primary"):
    if not topic.strip():
        st.error("Please describe what the post should be about.")
    elif not chosen:
        st.error("Choose at least one platform.")
    else:
        try:
            llm = get_llm(settings)
            profile_text, context, sources = brand_inputs(topic, use_kb, settings)
            bar = st.progress(0.0, text="Starting...")
            new_posts = []
            live = st.empty()                       # live preview while the model is writing
            for i, platform in enumerate(chosen, start=1):
                bar.progress((i - 1) / len(chosen), text=f"Writing for {platform}...")
                buffer = []

                def show(piece, buffer=buffer, platform=platform):
                    buffer.append(piece)
                    live.markdown(f"**Writing for {platform}...**\n\n```\n{''.join(buffer)}\n```")

                new_posts += generate_posts(
                    llm, topic, platform, tone=tone, audience=audience, goal=goal, language=language,
                    variations=variations, emojis=emojis, hashtag_count=hashtags, include_cta=cta,
                    profile_text=profile_text, brand_context=context, sources=sources, on_token=show)
                live.empty()
            bar.empty()
            st.session_state.latest = [p.__dict__ for p in new_posts]
            st.session_state.generated += st.session_state.latest
            if sources:
                st.caption("Brand facts came from: " + ", ".join(sources))
        except (LLMError, ValueError) as error:
            st.error(str(error))

latest = st.session_state.get("latest", [])
if latest:
    st.subheader("Results")
    for platform in dict.fromkeys(p["platform"] for p in latest):   # keep order, no duplicates
        group = [p for p in latest if p["platform"] == platform]
        st.markdown(f"#### {PLATFORMS[platform].icon} {platform}")
        if group[0].get("notice"):
            st.warning(group[0]["notice"])
        cols = st.columns(len(group))
        for col, post in zip(cols, group):
            with col:
                s = post["stats"]
                st.code(post["text"], language=None, wrap_lines=True)
                size = f"{s['chars']}" + (f" / {s['limit']}" if s["limit"] and s["limit"] < 10000 else "")
                line = f"**{size}** characters · {s['hashtags']} hashtags · {s['emojis']} emojis"
                if "seconds" in s:
                    line += f" · about {s['seconds']} s spoken"
                st.caption(line)
                for w in s["warnings"]:
                    st.warning(w)
