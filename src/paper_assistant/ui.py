"""Streamlit front end. It only talks to the FastAPI service; it contains no RAG logic.

Run:  streamlit run src/paper_assistant/ui.py
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Research Paper Assistant", page_icon="📄")
st.title("Research Paper Assistant")
st.caption(
    "Ask about 50 arXiv explainable-AI papers (2017-2023). Every answer cites paper and page; "
    "if the papers do not contain the answer, it says so."
)

question = st.text_input("Question", placeholder="What metrics are used to evaluate explanations?")
k = st.sidebar.slider("Passages to retrieve", 1, 10, 5)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Searching the papers..."):
        try:
            response = httpx.post(f"{API_URL}/ask", json={"question": question, "k": k}, timeout=90)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as error:
            st.error(f"API error {error.response.status_code}: {error.response.text}")
            st.stop()
        except httpx.HTTPError as error:
            st.error(f"Could not reach the API: {error}")
            st.stop()

    if data["answerable"]:
        st.markdown(data["answer"])
        st.subheader("Citations")
        for c in data["citations"]:
            st.markdown(f"- [{c['title'] or c['paper_id']} (p.{c['page']})]({c['url']})")
    else:
        st.warning("Not in the sources. The papers I have do not answer this question.")

    st.caption(f"{data['latency_ms']} ms")
    with st.expander("Retrieved passages"):
        for s in data["sources"]:
            st.markdown(f"**{s['title'] or s['paper_id']}**, p.{s['page']}")
            st.text(s["text"][:600])
