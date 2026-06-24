"""
Streamlit UI for the Florida Statute Research Assistant.

Imports AgenticRAG directly (no HTTP hop) so the app works as a single process
on Streamlit Community Cloud.  The retriever and pipeline are cached across
reruns with @st.cache_resource.

Deploy:
    streamlit run app/streamlit_app.py

Or on Streamlit Community Cloud, set Main file path to: app/streamlit_app.py
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st

# Inject Streamlit Cloud secrets into os.environ before any module reads them
if "GROQ_API_KEY" not in os.environ and "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = str(st.secrets["GROQ_API_KEY"]).strip()

# Ensure project root is on sys.path when run from the app/ subdirectory
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.retriever import HybridRetriever
from agent.graph import AgenticRAG

ABSTAIN_MARKER = "not found in the source material"

EXAMPLES = [
    "What is the definition of a 'public record' under Florida law?",
    "What is the right of a person to inspect and copy public records?",
    "Which records are exempt from disclosure for law enforcement agencies?",
    "What are the criminal penalties for violating Florida's public records law?",
    "What exemptions exist for trade secrets in public records requests?",
]


# ── Cached pipeline ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_pipeline() -> AgenticRAG:
    retriever = HybridRetriever()
    return AgenticRAG(retriever=retriever)


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_abstain(answer: str) -> bool:
    return ABSTAIN_MARKER in answer.lower()


def dedup_sources(chunks) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for c in chunks:
        key = c.meta["section_num"]
        if key not in seen:
            seen.add(key)
            out.append({
                "section_num":   c.meta["section_num"],
                "section_title": c.meta["section_title"],
                "source_url":    c.meta.get("source_url"),
            })
    return out


def set_example(text: str) -> None:
    st.session_state["question_input"] = text


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Florida Statute Research Assistant",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚖️ FL Statute Assistant")

    st.markdown("""
    **Agentic RAG over Florida Statute Chapter 119 — Public Records Act.**

    Every answer is grounded in the statute text and cites the exact section.
    When a question is not covered, the system says so rather than guessing.
    """)

    st.markdown("---")
    st.subheader("Example questions")
    for ex in EXAMPLES:
        st.button(
            ex,
            key=f"ex_{ex[:30]}",
            on_click=set_example,
            args=[ex],
            use_container_width=True,
        )

    st.markdown("---")
    st.caption(
        "Source: [Florida Statutes Chapter 119]"
        "(https://www.leg.state.fl.us/statutes/index.cfm"
        "?App_mode=Display_Statute&URL=0100-0199/0119/0119.html) · "
        "Built with LangGraph + Groq + ChromaDB"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("Florida Statute Research Assistant")
st.caption("Ask anything about Chapter 119 · Public Records Act — cited answers only")

question = st.text_area(
    label="Question",
    key="question_input",
    height=80,
    placeholder="e.g. What are the exemptions for law enforcement records?",
    label_visibility="collapsed",
)

ask_col, _ = st.columns([1, 3])
ask_clicked = ask_col.button("Ask ▶", type="primary", use_container_width=True)

# ── Load models (shown once on first use) ────────────────────────────────────
if ask_clicked or st.session_state.get("_first_load"):
    st.session_state["_first_load"] = False
    with st.spinner("Loading models… (first run takes ~30 s)"):
        rag = load_pipeline()

# ── Handle query ──────────────────────────────────────────────────────────────
if ask_clicked:
    q = question.strip()
    if not q:
        st.error("Please enter a question.")
    else:
        rag = load_pipeline()
        with st.spinner("Researching…"):
            t0 = time.time()
            result = rag.ask(q)
            elapsed = time.time() - t0

        answer     = result["answer"]
        chunks     = result.get("chunks", [])
        sub_qs     = result.get("sub_questions", [])
        retry_ct   = result.get("retry_count", 0)
        searched   = result.get("searched_queries", [])

        st.markdown("---")

        # ── Answer ────────────────────────────────────────────────────────────
        if is_abstain(answer):
            st.warning(
                "**Not found in the Florida Statute source material.**\n\n" + answer,
                icon="⚠️",
            )
        else:
            st.subheader("Answer")
            st.markdown(answer)

        # ── Sources ───────────────────────────────────────────────────────────
        sources = dedup_sources(chunks)
        if sources:
            with st.expander(f"📄 Sources — {len(sources)} statute section(s)", expanded=True):
                for s in sources:
                    label = f"§ {s['section_num']} — {s['section_title']}"
                    url   = s["source_url"]
                    st.markdown(f"- [{label}]({url})" if url else f"- {label}")

        # ── Agent trace ───────────────────────────────────────────────────────
        with st.expander("🔍 Agent trace", expanded=False):
            st.markdown(f"**Response time:** {elapsed:.1f} s")
            if sub_qs:
                st.markdown("**Sub-questions decomposed:**")
                for sq in sub_qs:
                    st.markdown(f"  - {sq}")
            if retry_ct:
                st.markdown(f"**Retrieval retries:** {retry_ct}")
            if searched and searched != sub_qs:
                st.markdown("**All queries searched:**")
                for sq in searched:
                    st.markdown(f"  - {sq}")
