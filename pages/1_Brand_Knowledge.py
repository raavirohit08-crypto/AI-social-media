import streamlit as st

st.set_page_config(page_title="Brand & Knowledge", page_icon="🏷️", layout="wide")

from src.brand import PROFILE_FIELDS, save_profile
from src.kb import KNOWLEDGE_DIR, VECTOR_DIR, is_index_current, list_files
from src.loader import SUPPORTED
from src.state import bootstrap, build_knowledge_base, require_token, sync_knowledge_base

settings = bootstrap()
require_token(settings)
sync_knowledge_base(settings)      # never show or use an out-of-date index
st.title("🏷️ Brand & Knowledge")

# ---- brand profile ----------------------------------------------------------
st.subheader("Brand profile")
st.caption("Used in every prompt so posts sound like your brand. Saved to brand_profile.json.")
with st.form("profile"):
    profile = {}
    c1, c2 = st.columns(2)
    for i, (key, label) in enumerate(PROFILE_FIELDS.items()):
        box = c1 if i % 2 == 0 else c2
        profile[key] = box.text_area(label, value=st.session_state.profile.get(key, ""), height=90)
    if st.form_submit_button("Save profile", type="primary"):
        st.session_state.profile = profile
        try:
            save_profile(profile)
            st.success("Profile saved.")
        except OSError as error:
            st.warning(f"Profile kept for this session only ({error}).")

# ---- knowledge documents --------------------------------------------------------
st.subheader("Brand documents")
st.caption(f"Brand guide, product sheets, FAQs and past posts go in the `{KNOWLEDGE_DIR.name}/` folder "
           f"(.pdf, .txt, .md). The index is saved in `{VECTOR_DIR.name}/`.")
uploads = st.file_uploader("Add brand documents", type=[s.lstrip(".") for s in SUPPORTED],
                           accept_multiple_files=True)
if uploads and st.button("Save to Knowledge folder"):
    KNOWLEDGE_DIR.mkdir(exist_ok=True)
    for f in uploads:
        (KNOWLEDGE_DIR / f.name).write_bytes(f.getbuffer())
    build_knowledge_base(settings)          # refresh right away so old facts are never used
    st.success(f"Saved {len(uploads)} file(s) and refreshed the knowledge base.")

files = list_files()
st.write(f"**Documents ({len(files)})**")
if not files:
    st.write("None yet. The app still works without them, but posts will be more generic.")
for path in files:
    a, b = st.columns([5, 1])
    a.write(f"📄 {path.name}  ·  {path.stat().st_size / 1024:.0f} KB")
    if b.button("Delete", key=f"del_{path.name}"):
        path.unlink()
        sync_knowledge_base(settings)       # drop the deleted document's facts immediately
        st.rerun()

with st.expander("Settings"):
    st.write(f"LLM model: `{settings.model_name}` (change MODEL_NAME in `.env`)")
    st.write(f"Chunk size: {settings.chunk_size} · overlap: {settings.chunk_overlap} · passages used: {settings.top_k}")

st.write("Index status:", "✅ up to date" if is_index_current() else "⚠️ needs (re)building")
c1, c2 = st.columns(2)
if c1.button("Build / refresh", type="primary"):
    build_knowledge_base(settings)
if c2.button("Force full rebuild"):
    build_knowledge_base(settings, force=True)
st.info(st.session_state.status)
for name in st.session_state.skipped:
    st.warning(f"No readable text in {name}. It was skipped.")
