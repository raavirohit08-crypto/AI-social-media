import streamlit as st

st.set_page_config(page_title="Improve Post", page_icon="🪄", layout="wide")

from src.generator import LANGUAGES, TONES
from src.improver import ACTIONS, improve_post
from src.llm import LLMError
from src.platforms import PLATFORMS
from src.state import bootstrap, brand_inputs, get_llm, require_token, sync_knowledge_base

settings = bootstrap()
require_token(settings)
sync_knowledge_base(settings)
st.title("🪄 Improve a Post")

draft = st.text_area("Paste your draft", height=180)
c1, c2, c3, c4 = st.columns(4)
platform = c1.selectbox("Platform", list(PLATFORMS))
action = c2.selectbox("What should I do?", list(ACTIONS))
tone = c3.selectbox("Tone (for 'Change tone')", TONES)
language = c4.selectbox("Language (for 'Translate')", LANGUAGES)
has_kb = st.session_state.store is not None
use_kb = st.checkbox("Use brand knowledge", value=has_kb, disabled=not has_kb,
                     help="Keeps rewrites consistent with your brand facts. The saved brand profile "
                          "(voice, audience, words to avoid) is always used.")

if st.button("Improve", type="primary"):
    try:
        with st.spinner("Rewriting..."):
            profile_text, context, _ = brand_inputs(draft, use_kb, settings)
            result = improve_post(get_llm(settings), draft, action, platform, tone=tone, language=language,
                                  profile_text=profile_text, brand_context=context,
                                  audience=st.session_state.profile.get("audience", ""))
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
