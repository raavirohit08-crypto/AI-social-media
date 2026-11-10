# Run with:  streamlit run app.py
import streamlit as st

st.set_page_config(page_title="Social Media Content Generator", page_icon="📣", layout="wide")

from src.kb import is_index_current, list_files
from src.state import bootstrap, require_token, sync_knowledge_base

settings = bootstrap()
require_token(settings)

# Keep the brand knowledge base in step with the Knowledge/ folder (it is optional).
# Rebuilds automatically if documents were added, changed or deleted.
sync_knowledge_base(settings)

st.title("📣 Social Media Content Generator")
st.caption("Brand-aware posts for LinkedIn, Instagram, X, Facebook and short videos, with a content calendar.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Brand documents", len(list_files()))
c2.metric("Index up to date", "Yes" if is_index_current() else "No")
c3.metric("Posts generated", len(st.session_state.generated))
c4.metric("Calendar entries", len(st.session_state.calendar))
st.info(st.session_state.status)

st.markdown(
    """
### How to use
1. **🏷️ Brand & Knowledge** *(recommended)*: fill in your brand profile and add brand documents
   (brand guide, product info, past posts), then click *Build / refresh*.
2. **✍️ Generate Posts**: pick platforms, tone, goal and topic. Get 1-3 versions per platform with
   character / hashtag checks.
3. **🗓️ Content Calendar**: a 7, 14 or 30 day plan with ideas, hooks and hashtags.
4. **🪄 Improve Post**: shorten, add a hook, fix grammar, change tone or translate your own draft.
5. **🖼️ Image Ideas**: text-to-image prompts and alt text to go with a post.
6. **#️⃣ Hashtag Helper**: suggested hashtags (AI suggestions, not live trends) plus your own brand tags.
7. **⬇️ Export**: download everything as Markdown, CSV or text.

> Always read AI-written posts before publishing. The app never posts anything for you.
"""
)
