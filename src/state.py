
"""Shared Streamlit session logic used by every page."""

import os
import streamlit as st

from .brand import get_brand_context, load_profile, profile_to_text
from .config import load_settings


def bootstrap():
    """Call at the top of every page. Returns Settings."""

    # Load secrets from Streamlit Cloud into environment variables.
    try:
        for key in ("HF_TOKEN", "MODEL_NAME", "LLM_BASE_URL"):
            if key in st.secrets and not os.getenv(key):
                os.environ[key] = st.secrets[key]
    except Exception:
        pass

    settings = load_settings()

    defaults = {
        "store": None,
        "kb_files": {},
        "skipped": [],
        "generated": [],
        "calendar": [],
        "improved": [],
        "image_ideas": [],
        "hashtag_sets": [],
        "status": "Brand knowledge base not built yet (optional).",
        "profile": load_profile(),
        "kb_needs_rebuild": False,
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    return settings


def require_token(settings):
    """Stop the page if the Hugging Face token is missing."""

    if not settings.hf_token:
        st.error(
            "HF_TOKEN is missing. Create a `.env` file "
            "(see `.env.example`) and restart the app."
        )
        st.stop()


@st.cache_resource(show_spinner="Loading the embedding model (first run only)...")
def cached_embeddings():
    from .vector_store import get_embeddings
    return get_embeddings()


@st.cache_resource(show_spinner=False)
def cached_llm(token: str, model_name: str, base_url: str = ""):
    from .llm import LLM
    return LLM(
        token=token,
        model_name=model_name,
        base_url=base_url or None
    )


def get_llm(settings):
    """Return the cached language model."""

    return cached_llm(
        settings.hf_token,
        settings.model_name,
        os.getenv("LLM_BASE_URL", "")
    )


def invalidate_knowledge_base():
    """Mark the existing brand knowledge base as outdated."""

    st.session_state.store = None
    st.session_state.kb_files = {}
    st.session_state.kb_needs_rebuild = True
    st.session_state.status = (
        "Brand documents changed. Rebuild the knowledge base to use the latest files."
    )


def build_knowledge_base(settings, force: bool = False):
    """Build or reload the brand index from the Knowledge/ folder."""

    from .kb import build_or_load

    # Rebuild automatically when documents have changed.
    force = force or st.session_state.get("kb_needs_rebuild", False)

    try:
        with st.spinner(
            "Reading brand documents and preparing the vector database..."
        ):
            store, loaded, skipped = build_or_load(
                cached_embeddings(),
                settings,
                force=force
            )

    except Exception as error:
        st.session_state.status = (
            f"Could not build the brand knowledge base: {error}"
        )
        return

    st.session_state.skipped = skipped

    if store is None:
        st.session_state.update(
            store=None,
            kb_files={},
            kb_needs_rebuild=False,
            status="No readable brand documents found in the Knowledge/ folder."
        )
        return

    st.session_state.update(
        store=store,
        kb_files=loaded,
        kb_needs_rebuild=False,
        status=f"Ready: {len(loaded)} brand document(s) indexed."
    )


def brand_inputs(query: str, use_kb: bool, settings):
    """Return profile text, brand context and source names."""

    profile_text = profile_to_text(st.session_state.profile)
    context, passages = "", []

    if use_kb and st.session_state.store is not None:
        context, passages = get_brand_context(
            st.session_state.store,
            query,
            settings.top_k
        )

    return (
        profile_text,
        context,
        sorted({passage.source for passage in passages})
    )
