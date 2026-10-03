"""
Waste Sorting Assistant: demo app.

Run from the project folder (Ollama must be running):
    streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rag import WasteSortingRAG, answer_without_retrieval, DEFAULT_TOP_K  # noqa: E402

st.set_page_config(page_title="Waste Sorting Assistant", page_icon="♻️", layout="centered")

EXAMPLES = [
    "Can a greasy pizza box go in the recycling bin?",
    "Where should an old laptop go?",
    "How do I get rid of leftover paint?",
    "How do I renew my passport?",
]


@st.cache_resource
def load_rag():
    return WasteSortingRAG()


@st.cache_data
def load_source_log():
    path = ROOT / "data" / "source_log.csv"
    if not path.exists():
        return {}
    return pd.read_csv(path).set_index("file").to_dict("index")


def ask(question):
    st.session_state["question"] = question


st.title("♻️ Waste Sorting Assistant")
st.caption("Answers for Victorian households, using only official Victorian Government guidance.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Passages retrieved (Top-K)", 1, 10, DEFAULT_TOP_K)
    compare = st.toggle("Compare with the AI model on its own", value=False)
    st.caption("The comparison shows what the same AI model says without the official documents.")

st.write("Try an example:")
columns = st.columns(2)
for i, example in enumerate(EXAMPLES):
    columns[i % 2].button(example, on_click=ask, args=(example,), use_container_width=True)

question = st.text_input("Ask a waste-sorting question", key="question",
                         placeholder="e.g. Can batteries go in my rubbish bin?")

if question.strip():
    rag = load_rag()
    source_log = load_source_log()

    with st.spinner("Checking the official guidance..."):
        result = rag.answer(question.strip(), top_k=top_k)

    if compare:
        left, right = st.columns(2)
    else:
        left, right = st.container(), None

    with left:
        st.subheader("Our assistant")
        if result["refused"]:
            st.warning(result["answer_body"])
            st.caption("The assistant declines rather than guessing when the official "
                       "guidance does not cover the question.")
        else:
            st.success(result["answer_body"])
            st.markdown("**Official sources used**")
            for source in result["sources"]:
                info = source_log.get(source, {})
                title = info.get("title", source)
                url = info.get("url", "")
                updated = info.get("last_updated", "")
                line = f"- [{title}]({url})" if isinstance(url, str) and url else f"- {title}"
                if isinstance(updated, str) and updated:
                    line += f" (updated {updated})"
                st.markdown(line)

    if compare:
        with right:
            st.subheader("AI model on its own")
            with st.spinner("Asking the model without documents..."):
                baseline = answer_without_retrieval(question.strip())
            st.info(baseline["answer_body"])
            st.caption("No sources: this answer cannot be verified.")

    with st.expander("Show the passages the answer was based on"):
        for item in result["retrieved"]:
            st.markdown(f"**{item['source']}** (similarity {item['score']:.2f})")
            st.write(item["text"])
