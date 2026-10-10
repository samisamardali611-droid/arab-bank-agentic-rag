"""Run with: python -m streamlit run app.py"""

import asyncio
import os

import streamlit as st

from bank_rag.config import INDEX_FILE, META_FILE

st.set_page_config(
    page_title="Financial Statements RAG",
    page_icon="💬",
    layout="centered",
)
st.title("Arab Bank Financial Statements Q&A")
st.caption("A portfolio RAG prototype using public reports from 2021–2024.")
st.info(
    "Numeric extraction is experimental. Check amounts, units, and years "
    "against the original financial statement."
)

if not os.getenv("OPENAI_API_KEY", "").strip():
    st.warning("Add your own OpenAI API key to a local .env file, then restart the app.")
    st.stop()

if not INDEX_FILE.is_file() or not META_FILE.is_file():
    st.error("The search index is missing. Follow the rebuild steps in README.md.")
    st.stop()

from bank_rag.agents.orchestrator import run_orchestrator

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question about the reports (English recommended)...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Searching financial statements..."):
            try:
                response = asyncio.run(run_orchestrator(user_input))
            except Exception:
                # Do not render raw exceptions that may contain provider/request data.
                response = (
                    "The request could not be completed. Check your API configuration, "
                    "model access, account quota, and local index files, then try again."
                )
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
