"""Shared Streamlit session logic used by every page (kept out of the UI files)."""
import os

import streamlit as st

from .brand import get_brand_context, load_profile, profile_to_text
from .config import load_settings


def bootstrap():
    """Call at the top of every page. Returns Settings."""
    try:  # Streamlit Cloud secrets -> environment
        for key in ("HF_TOKEN", "MODEL_NAME", "LLM_BASE_URL"):
            if key in st.secrets and not os.getenv(key):
                os.environ[key] = st.secrets[key]
    except Exception:
        pass
    settings = load_settings()
    defaults = {"store": None, "kb_files": {}, "skipped": [], "generated": [], "calendar": [],
                "improved": [], "image_ideas": [], "hashtag_sets": [], "status": "Brand knowledge base not built yet (optional).",
                "profile": load_profile()}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    return settings


def require_token(settings):
    if not settings.hf_token:
        st.error("HF_TOKEN is missing. Create a `.env` file (see `.env.example`) and restart the app.")
        st.stop()


@st.cache_resource(show_spinner="Loading the embedding model (first run only)...")
def cached_embeddings():
    from .vector_store import get_embeddings
    return get_embeddings()


@st.cache_resource(show_spinner=False)
def cached_llm(token: str, model_name: str, base_url: str = ""):
    from .llm import LLM
    return LLM(token=token, model_name=model_name, base_url=base_url or None)


def get_llm(settings):
    return cached_llm(settings.hf_token, settings.model_name, os.getenv("LLM_BASE_URL", ""))


def build_knowledge_base(settings, force: bool = False):
    """Build/load the brand index from Knowledge/ into session_state."""
    from .kb import build_or_load
    try:
        with st.spinner("Reading brand documents and preparing the vector database..."):
            store, loaded, skipped = build_or_load(cached_embeddings(), settings, force=force)
    except Exception as error:
        st.session_state.status = f"Could not build the brand knowledge base: {error}"
        return
    st.session_state.skipped = skipped
    if store is None:
        st.session_state.update(store=None, kb_files={},
                                status="No readable brand documents found in the Knowledge/ folder.")
        return
    st.session_state.update(
        store=store, kb_files=loaded,
        status=f"Ready: {len(loaded)} brand document(s) indexed.")


def brand_inputs(query: str, use_kb: bool, settings):
    """Return (profile_text, brand_context, source_names) for a generation request."""
    profile_text = profile_to_text(st.session_state.profile)
    context, passages = "", []
    if use_kb and st.session_state.store is not None:
        context, passages = get_brand_context(st.session_state.store, query, settings.top_k)
    return profile_text, context, sorted({p.source for p in passages})
