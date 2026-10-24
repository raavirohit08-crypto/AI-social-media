import streamlit as st

st.set_page_config(page_title="Hashtag Helper", page_icon="#️⃣", layout="wide")

from src.creative import hashtag_ideas
from src.llm import LLMError
from src.platforms import PLATFORMS
from src.state import bootstrap, brand_inputs, get_llm, require_token

settings = bootstrap()
require_token(settings)
st.title("#️⃣ Hashtag Helper")
st.info("These are **AI suggestions, not live trending data**. This app cannot see current trends. "
        "Search each hashtag on the platform to check it is active and means what you expect. "
        "Hashtags starting with your brand's own tags come from your Knowledge documents.")

generated = st.session_state.generated
choices = ["✏️ Type a topic"] + [
    f"{p['platform']} · v{p['variation']} · {p['text'][:50].replace(chr(10), ' ')}..." for p in generated]
pick = st.selectbox("Based on", choices)
default_text, default_platform = ("", list(PLATFORMS)[0])
if pick != choices[0]:
    chosen_post = generated[choices.index(pick) - 1]
    default_text, default_platform = chosen_post["text"], chosen_post["platform"]

topic = st.text_area("Topic or post text", value=default_text, height=120, key=f"tag_text_{pick}")
c1, c2 = st.columns(2)
platform = c1.selectbox("Platform", list(PLATFORMS), index=list(PLATFORMS).index(default_platform))
use_kb = c2.checkbox("Use brand knowledge (picks up your own hashtags)",
                     value=st.session_state.store is not None, disabled=st.session_state.store is None)

if st.button("Suggest hashtags", type="primary"):
    try:
        with st.spinner("Finding hashtags..."):
            profile_text, context, _ = brand_inputs(topic, use_kb, settings)
            groups = hashtag_ideas(get_llm(settings), topic, platform, profile_text, context)
        st.session_state.last_tags = {"platform": platform, "topic": topic, "groups": groups}
        st.session_state.hashtag_sets.append(st.session_state.last_tags)
    except (LLMError, ValueError) as error:
        st.error(str(error))

result = st.session_state.get("last_tags")
if result:
    g = result["groups"]
    st.subheader(f"Recommended set for {result['platform']}")
    st.code(" ".join(g["recommended"]) or "(none)", language=None)
    a, b, c = st.columns(3)
    for col, key, label in ((a, "branded", "Your brand"), (b, "niche", "Niche"), (c, "broad", "Broad")):
        col.markdown(f"**{label}**")
        col.write(" ".join(g[key]) if g[key] else "—")
