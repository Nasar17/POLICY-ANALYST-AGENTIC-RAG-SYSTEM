"""
Streamlit UI for the Florida Statute Research Assistant.

Imports AgenticRAG directly (no HTTP hop) so the app works as a single process
on Streamlit Community Cloud.  The retriever and pipeline are cached across
reruns with @st.cache_resource.
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Florida Statute Research Assistant",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Inject Streamlit Cloud secrets into os.environ before any module reads them
try:
    os.environ["GROQ_API_KEY"] = str(st.secrets["GROQ_API_KEY"]).strip()
except KeyError:
    pass  # local: key comes from .env file

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

# ── Styles ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.block-container { padding-top: 1.75rem; padding-bottom: 2rem; }

.app-header { margin-bottom: 0.25rem; }
.app-header h1 { font-size: 1.85rem; font-weight: 700; margin-bottom: 0.1rem; }
.app-header p  { color: #6c757d; font-size: 0.92rem; margin-top: 0; }

.source-card {
    background: #eef2f8;
    border-left: 3px solid #1a3a6b;
    padding: 0.4rem 0.75rem;
    margin: 0.25rem 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.88rem;
}
.source-card a { color: #1a3a6b; text-decoration: none; font-weight: 500; }
.source-card a:hover { text-decoration: underline; }

.meta-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 0.6rem 0 0.9rem; }
.meta-pill {
    background: #e8edf5;
    color: #1a3a6b;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
}
</style>
""", unsafe_allow_html=True)

# ── Pipeline ──────────────────────────────────────────────────────────────────

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


def render_answer(msg: dict) -> None:
    answer   = msg["answer"]
    sources  = msg.get("sources", [])
    elapsed  = msg.get("elapsed", 0.0)
    sub_qs   = msg.get("sub_questions", [])
    retry_ct = msg.get("retry_count", 0)
    searched = msg.get("searched_queries", [])

    if is_abstain(answer):
        st.warning(
            "**Not found in the Florida Statute source material.**\n\n" + answer,
            icon="⚠️",
        )
    else:
        st.markdown(answer)

    # Metric pills
    pills = [f"⏱ {elapsed:.1f} s"]
    if sources:
        n = len(sources)
        pills.append(f"📄 {n} section{'s' if n != 1 else ''}")
    if retry_ct:
        pills.append(f"🔄 {retry_ct} retr{'ies' if retry_ct != 1 else 'y'}")
    st.markdown(
        '<div class="meta-row">'
        + "".join(f'<span class="meta-pill">{p}</span>' for p in pills)
        + "</div>",
        unsafe_allow_html=True,
    )

    if sources:
        with st.expander(f"Sources — {len(sources)} statute section(s)", expanded=True):
            cards = []
            for s in sources:
                label = f"§ {s['section_num']} — {s['section_title']}"
                url   = s["source_url"]
                inner = f'<a href="{url}" target="_blank">{label}</a>' if url else label
                cards.append(f'<div class="source-card">{inner}</div>')
            st.markdown("\n".join(cards), unsafe_allow_html=True)

    with st.expander("Agent trace", expanded=False):
        if sub_qs:
            st.markdown("**Sub-questions decomposed:**")
            for q in sub_qs:
                st.markdown(f"- {q}")
        if searched and searched != sub_qs:
            st.markdown("**All queries searched:**")
            for q in searched:
                st.markdown(f"- {q}")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚖️ FL Statute Assistant")
    st.markdown(
        "Agentic RAG over **Chapter 119 — Public Records Act**. "
        "Every answer cites the exact statute section. "
        "Out-of-scope questions get a clear *not found* rather than a guess."
    )
    st.divider()

    st.markdown("**Example questions**")
    for ex in EXAMPLES:
        if st.button(ex, key=f"ex_{ex[:30]}", use_container_width=True):
            st.session_state["_prefill"] = ex
            st.rerun()

    st.divider()
    st.markdown(
        "<small>Source: [Florida Statutes Ch. 119]"
        "(https://www.leg.state.fl.us/statutes/index.cfm"
        "?App_mode=Display_Statute&URL=0100-0199/0119/0119.html)"
        " · Built with LangGraph, Groq, ChromaDB</small>",
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    "<div class='app-header'>"
    "<h1>⚖️ Florida Statute Research Assistant</h1>"
    "<p>Chapter 119 · Public Records Act — grounded, cited answers only</p>"
    "</div>",
    unsafe_allow_html=True,
)
st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_answer(msg)

# ── Input ─────────────────────────────────────────────────────────────────────

prefill = st.session_state.pop("_prefill", None)
prompt  = st.chat_input("Ask about Florida's public records law…") or prefill

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        spinner_msg = (
            "Loading models… (first run takes ~30 s)"
            if not st.session_state.get("_loaded")
            else "Researching…"
        )
        with st.spinner(spinner_msg):
            rag = load_pipeline()
            st.session_state["_loaded"] = True
            t0     = time.time()
            result = rag.ask(prompt)
            elapsed = time.time() - t0

        sources = dedup_sources(result.get("chunks", []))
        assistant_msg = {
            "role":             "assistant",
            "answer":           result["answer"],
            "sources":          sources,
            "elapsed":          elapsed,
            "sub_questions":    result.get("sub_questions", []),
            "retry_count":      result.get("retry_count", 0),
            "searched_queries": result.get("searched_queries", []),
        }
        render_answer(assistant_msg)
        st.session_state.messages.append(assistant_msg)
