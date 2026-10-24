import streamlit as st

st.set_page_config(page_title="Export", page_icon="⬇️", layout="wide")

from src.exporter import (calendar_to_csv, calendar_to_markdown, hashtags_to_text,
                          image_ideas_to_markdown, posts_to_csv, posts_to_markdown)
from src.state import bootstrap

bootstrap()
st.title("⬇️ Export")

posts = st.session_state.generated
if posts:
    st.write(f"**{len(posts)} generated post(s)**")
    a, b = st.columns(2)
    a.download_button("Posts (.md)", posts_to_markdown(posts), "social_posts.md", "text/markdown")
    b.download_button("Posts (.csv)", posts_to_csv(posts), "social_posts.csv", "text/csv")
else:
    st.caption("Generate some posts to export them.")

rows = st.session_state.calendar
if rows:
    st.write(f"**Calendar with {len(rows)} entries**")
    a, b = st.columns(2)
    a.download_button("Calendar (.csv)", calendar_to_csv(rows), "content_calendar.csv", "text/csv")
    b.download_button("Calendar (.md)", calendar_to_markdown(rows), "content_calendar.md", "text/markdown")
else:
    st.caption("Plan a calendar to export it.")

improved = st.session_state.improved
if improved:
    text = "\n\n---\n\n".join(f"{r['platform']} · {r['action']}\n\n{r['text']}" for r in improved)
    st.download_button("Improved posts (.txt)", text, "improved_posts.txt", "text/plain")

if st.session_state.image_ideas:
    st.write(f"**{len(st.session_state.image_ideas)} image idea(s)**")
    st.download_button("Image ideas (.md)", image_ideas_to_markdown(st.session_state.image_ideas),
                       "image_ideas.md", "text/markdown")

if st.session_state.hashtag_sets:
    st.write(f"**{len(st.session_state.hashtag_sets)} hashtag set(s)**")
    st.download_button("Hashtag sets (.txt)", hashtags_to_text(st.session_state.hashtag_sets),
                       "hashtags.txt", "text/plain")
