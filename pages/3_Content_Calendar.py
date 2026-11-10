from datetime import date

import streamlit as st

st.set_page_config(page_title="Content Calendar", page_icon="🗓️", layout="wide")

from src.calendar_planner import COLUMNS, IncompleteCalendarError, plan_calendar
from src.generator import GOALS, TONES
from src.llm import LLMError
from src.platforms import PLATFORMS
from src.state import bootstrap, brand_inputs, get_llm, require_token, sync_knowledge_base

settings = bootstrap()
require_token(settings)
sync_knowledge_base(settings)
st.title("🗓️ Content Calendar")

has_kb = st.session_state.store is not None
c1, c2, c3 = st.columns(3)
days = c1.selectbox("Length", [7, 14, 30], format_func=lambda d: f"{d} days")
per_week = c2.slider("Posts per week", 2, 7, 4)
start = c3.date_input("Start date", value=date.today())
chosen = st.multiselect("Platforms", list(PLATFORMS), default=["LinkedIn", "Instagram"])
c1, c2, c3 = st.columns(3)
goal = c1.selectbox("Goal", GOALS)
tone = c2.selectbox("Tone", TONES)
use_kb = c3.checkbox("Use brand knowledge", value=has_kb, disabled=not has_kb)
theme = st.text_input("Theme / campaign focus (optional)", placeholder="e.g. Diwali offers, product launch, hiring")

if st.button("Plan calendar", type="primary"):
    if not chosen:
        st.error("Choose at least one platform.")
    else:
        try:
            with st.spinner("Planning your calendar..."):
                profile_text, context, _ = brand_inputs(theme or goal, use_kb, settings)
                st.session_state.calendar = plan_calendar(
                    get_llm(settings), goal, chosen, days=days, posts_per_week=per_week, start=start,
                    theme=theme, tone=tone, audience=st.session_state.profile.get("audience", ""),
                    profile_text=profile_text, brand_context=context)
        except IncompleteCalendarError as error:
            st.error(str(error))               # never shown as if it were a complete plan
            if error.rows:
                with st.expander(f"Partial result ({len(error.rows)} posts, NOT a complete plan)"):
                    st.dataframe(error.rows, column_order=COLUMNS, hide_index=True)
        except (LLMError, ValueError) as error:
            st.error(str(error))

rows = st.session_state.calendar
if rows:
    st.subheader(f"{len(rows)} planned posts")
    st.dataframe(rows, column_order=COLUMNS, hide_index=True)
    st.caption("Ideas and hooks are starting points. Turn any row into a full post on the Generate page.")
