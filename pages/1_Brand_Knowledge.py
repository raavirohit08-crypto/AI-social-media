
import streamlit as st

st.set_page_config(
    page_title="Brand & Knowledge",
    page_icon="🏷️",
    layout="wide"
)

from src.brand import PROFILE_FIELDS, save_profile
from src.kb import (
    KNOWLEDGE_DIR,
    VECTOR_DIR,
    is_index_current,
    list_files
)
from src.loader import SUPPORTED
from src.state import (
    bootstrap,
    build_knowledge_base,
    require_token,
    sync_knowledge_base
)

# ---- application setup --------------------------------------------------

settings = bootstrap()
require_token(settings)
sync_knowledge_base(settings)

st.title("🏷️ Brand & Knowledge")

# ---- brand profile ------------------------------------------------------

st.subheader("Brand profile")

st.caption(
    "Used in every prompt so posts sound like your brand. "
    "Saved to brand_profile.json."
)

# Initialize profile before creating widgets
if "profile" not in st.session_state:
    st.session_state.profile = {}

with st.form("profile"):
    profile = {}

    c1, c2 = st.columns(2)

    for i, (key, label) in enumerate(PROFILE_FIELDS.items()):
        box = c1 if i % 2 == 0 else c2

        profile[key] = box.text_area(
            label,
            value=st.session_state.profile.get(key, ""),
            height=90,
            key=f"profile_input_{key}"
        )

    submitted = st.form_submit_button(
        "Save profile",
        type="primary"
    )

    if submitted:
        try:
            # Update existing dictionary without replacing it
            st.session_state.profile.clear()
            st.session_state.profile.update(profile)

            save_profile(profile)

            st.success("Profile saved successfully!")

        except OSError as error:
            st.warning(
                f"Profile kept for this session only ({error})."
            )

# ---- knowledge documents -----------------------------------------------

st.subheader("Brand documents")

st.caption(
    f"Brand guide, product sheets, FAQs and past posts go in "
    f"the `{KNOWLEDGE_DIR.name}/` folder "
    f"(.pdf, .txt, .md). "
    f"The index is saved in `{VECTOR_DIR.name}/`."
)

uploads = st.file_uploader(
    "Add brand documents",
    type=[s.lstrip(".") for s in SUPPORTED],
    accept_multiple_files=True
)

if uploads and st.button("Save to Knowledge folder"):

    KNOWLEDGE_DIR.mkdir(exist_ok=True)

    for file in uploads:
        (KNOWLEDGE_DIR / file.name).write_bytes(
            file.getbuffer()
        )

    build_knowledge_base(settings)

    st.success(
        f"Saved {len(uploads)} file(s) and refreshed the knowledge base."
    )

# ---- display uploaded documents ----------------------------------------

files = list_files()

st.write(f"**Documents ({len(files)})**")

if not files:
    st.write(
        "None yet. The app still works without them, "
        "but posts will be more generic."
    )

for path in files:

    a, b = st.columns([5, 1])

    a.write(
        f"📄 {path.name} · "
        f"{path.stat().st_size / 1024:.0f} KB"
    )

    if b.button("Delete", key=f"del_{path.name}"):

        path.unlink()

        sync_knowledge_base(settings)

        st.success(f"{path.name} deleted successfully.")

        st.rerun()

# ---- settings -----------------------------------------------------------

with st.expander("Settings"):

    st.write(
        f"LLM model: `{settings.model_name}` "
        "(change MODEL_NAME in `.env`)"
    )

    st.write(
        f"Chunk size: {settings.chunk_size} "
        f"· overlap: {settings.chunk_overlap} "
        f"· passages used: {settings.top_k}"
    )

# ---- knowledge base status ---------------------------------------------

st.subheader("Knowledge Base Status")

st.write(
    "Index status:",
    "✅ up to date"
    if is_index_current()
    else "⚠️ needs (re)building"
)

c1, c2 = st.columns(2)

if c1.button("Build / refresh", type="primary"):

    build_knowledge_base(settings)

    st.success("Knowledge base refreshed successfully!")

    st.rerun()

if c2.button("Force full rebuild"):

    build_knowledge_base(settings, force=True)

    st.success("Knowledge base fully rebuilt!")

    st.rerun()

# ---- application messages -----------------------------------------------

st.info(st.session_state.status)

for name in st.session_state.skipped:

    st.warning(
        f"No readable text in {name}. It was skipped."
    )
