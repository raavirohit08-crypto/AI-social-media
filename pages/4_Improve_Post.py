import streamlit as st

st.set_page_config(page_title="Improve Post", page_icon="🪄", layout="wide")

from src.generator import LANGUAGES, TONES
from src.improver import ACTIONS, improve_post
from src.llm import LLMError
from src.platforms import PLATFORMS
from src.state import bootstrap, get_llm, require_token

settings = bootstrap()
require_token(settings)
st.title("🪄 Improve a Post")

draft = st.text_area("Paste your draft", height=180)
c1, c2, c3, c4 = st.columns(4)
platform = c1.selectbox("Platform", list(PLATFORMS))
action = c2.selectbox("What should I do?", list(ACTIONS))
tone = c3.selectbox("Tone (for 'Change tone')", TONES)
language = c4.selectbox("Language (for 'Translate')", LANGUAGES)

if st.button("Improve", type="primary"):
    try:
        with st.spinner("Rewriting..."):
            result = improve_post(get_llm(settings), draft, action, platform, tone=tone, language=language)
        st.session_state.improved.append(result)
        st.session_state.last_improved = result
    except (LLMError, ValueError) as error:
        st.error(str(error))

result = st.session_state.get("last_improved")
if result:
    a, b = st.columns(2)
    a.markdown("**Original**")
    a.code(result["original"], language=None, wrap_lines=True)
    b.markdown(f"**{result['action']}**")
    b.code(result["text"], language=None, wrap_lines=True)
    s = result["stats"]
    b.caption(f"{s['chars']} characters · {s['hashtags']} hashtags")
    for w in s["warnings"]:
        b.warning(w)
