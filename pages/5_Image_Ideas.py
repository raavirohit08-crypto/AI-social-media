import streamlit as st

st.set_page_config(page_title="Image Ideas", page_icon="🖼️", layout="wide")

from src.creative import ASPECT, STYLES, image_prompts
from src.llm import LLMError
from src.platforms import PLATFORMS
from src.state import bootstrap, brand_inputs, get_llm, require_token

settings = bootstrap()
require_token(settings)
st.title("🖼️ Image Ideas")
st.caption("Get text-to-image prompts and alt text to match a post. Paste a prompt into any image tool "
           "and add any words (like a price) afterwards in an editor, because AI tools draw text badly.")

generated = st.session_state.generated
choices = ["✏️ Paste my own text"] + [
    f"{p['platform']} · v{p['variation']} · {p['text'][:50].replace(chr(10), ' ')}..." for p in generated]
pick = st.selectbox("Post to illustrate", choices)
default_text, default_platform = ("", list(PLATFORMS)[0])
if pick != choices[0]:
    chosen_post = generated[choices.index(pick) - 1]
    default_text, default_platform = chosen_post["text"], chosen_post["platform"]

text = st.text_area("Post text", value=default_text, height=140, key=f"img_text_{pick}")
c1, c2, c3, c4 = st.columns(4)
platform = c1.selectbox("Platform", list(PLATFORMS), index=list(PLATFORMS).index(default_platform))
style = c2.selectbox("Visual style", STYLES)
count = c3.slider("Ideas", 1, 4, 3)
use_kb = c4.checkbox("Use brand knowledge", value=st.session_state.store is not None,
                     disabled=st.session_state.store is None)
st.caption(f"Common image shape for {platform}: {ASPECT[platform]}")

if st.button("Suggest images", type="primary"):
    try:
        with st.spinner("Thinking up visuals..."):
            profile_text, context, _ = brand_inputs(text, use_kb, settings)
            st.session_state.last_images = image_prompts(
                get_llm(settings), text, platform, style, count, profile_text, context)
        st.session_state.image_ideas += st.session_state.last_images
    except (LLMError, ValueError) as error:
        st.error(str(error))

for n, idea in enumerate(st.session_state.get("last_images", []), start=1):
    with st.container(border=True):
        st.markdown(f"**{n}. {idea['concept']}**")
        st.caption("Prompt (click the copy icon)")
        st.code(idea["prompt"], language=None, wrap_lines=True)
        st.write(f"**Alt text:** {idea['alt_text']}")
