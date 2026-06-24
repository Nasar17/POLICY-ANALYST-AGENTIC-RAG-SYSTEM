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

if not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Add it to Streamlit secrets or a local .env file.")
    st.stop()

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
<link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">

<style>
html, body,
.stMarkdown,
[data-testid="stMarkdownContainer"],
[data-testid="stChatMessageContent"],
[data-testid="stWidgetLabel"],
.stChatInput textarea,
input[type="text"],
.stButton > button > div > p {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Layout */
.block-container {
    padding-top: 1.25rem;
    padding-bottom: 3rem;
    max-width: 760px;
}

/* App header (shown when chat has messages) */
.app-header {
    padding: 0.25rem 0 0.5rem;
    border-bottom: 1px solid #e8edf5;
    margin-bottom: 0.75rem;
}
.app-header h1 {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0f1f3d;
    margin: 0 0 0.1rem;
    letter-spacing: -0.01em;
}
.app-header p {
    color: #8a94a6;
    font-size: 0.82rem;
    margin: 0;
}

/* Welcome hero (shown when no messages) */
.welcome-hero {
    text-align: center;
    padding: 3.5rem 1rem 2rem;
}
.welcome-hero .badge {
    display: inline-block;
    background: #e8edf5;
    color: #1a3a6b;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1.25rem;
}
.welcome-hero h2 {
    font-size: 2rem;
    font-weight: 700;
    color: #0f1f3d;
    letter-spacing: -0.02em;
    margin: 0 0 0.6rem;
    line-height: 1.2;
}
.welcome-hero p {
    color: #6c757d;
    font-size: 0.95rem;
    margin: 0 auto 2.5rem;
    max-width: 480px;
    line-height: 1.6;
}
.suggestions-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #adb5bd;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.75rem;
    text-align: center;
}

/* Suggestion + sidebar buttons */
.stButton > button {
    background: #ffffff;
    border: 1px solid #dce3ef;
    border-radius: 10px;
    color: #1a3a6b;
    font-size: 0.84rem;
    font-weight: 500;
    white-space: normal;
    text-align: left;
    padding: 12px 15px;
    line-height: 1.45;
    width: 100%;
    transition: background 0.15s, border-color 0.15s, box-shadow 0.15s;
}
.stButton > button:hover {
    background: #f0f4fb;
    border-color: #1a3a6b;
    box-shadow: 0 2px 8px rgba(26,58,107,0.08);
    color: #1a3a6b;
}

/* Source cards */
.source-card {
    background: #f4f7fc;
    border-left: 3px solid #1a3a6b;
    padding: 0.45rem 0.8rem;
    margin: 0.28rem 0;
    border-radius: 0 7px 7px 0;
    font-size: 0.85rem;
    line-height: 1.4;
}
.source-card a {
    color: #1a3a6b;
    text-decoration: none;
    font-weight: 500;
}
.source-card a:hover { text-decoration: underline; }

/* Meta pills */
.meta-row {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
    margin: 0.7rem 0 1rem;
}
.meta-pill {
    background: #edf1f9;
    color: #3a5a9b;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.01em;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #f8f9fc;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
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


def pick(example: str) -> None:
    st.session_state["_prefill"] = example


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚖️ FL Statute Assistant")
    st.markdown(
        "Agentic RAG over **Chapter 119 — Public Records Act**. "
        "Every answer cites the exact statute section. "
        "Out-of-scope questions return a clear *not found* — no hallucination."
    )
    st.divider()

    st.markdown("**Example questions**")
    for ex in EXAMPLES:
        if st.button(ex, key=f"sb_{ex[:30]}", use_container_width=True):
            pick(ex)
            st.rerun()

    st.divider()
    st.markdown(
        "<small>Source: [Florida Statutes Ch. 119]"
        "(https://www.leg.state.fl.us/statutes/index.cfm"
        "?App_mode=Display_Statute&URL=0100-0199/0119/0119.html)"
        "<br>Built with LangGraph · Groq · ChromaDB</small>",
        unsafe_allow_html=True,
    )

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Welcome state (no messages yet) ──────────────────────────────────────────

if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-hero">
        <div class="badge">Florida Statute Chapter 119</div>
        <h2>Public Records Research<br>Assistant</h2>
        <p>Ask any question about Florida's Public Records Act.<br>
        Every answer is grounded in the statute and cites the exact section.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="suggestions-label">Try asking</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    suggestions = EXAMPLES[:4]
    for i, ex in enumerate(suggestions):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(ex, key=f"sg_{i}", use_container_width=True):
                pick(ex)
                st.rerun()

else:
    # Compact header once conversation has started
    st.markdown(
        "<div class='app-header'>"
        "<h1>⚖️ Florida Statute Research Assistant</h1>"
        "<p>Chapter 119 · Public Records Act — grounded, cited answers only</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_answer(msg)

# ── Chat input ────────────────────────────────────────────────────────────────

prefill = st.session_state.pop("_prefill", None)
prompt  = st.chat_input("Ask about Florida's public records law…") or prefill

if prompt:
    if len(prompt) > 1000:
        st.error("Question is too long (max 1 000 characters). Please be more concise.")
        st.stop()
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
            t0      = time.time()
            result  = rag.ask(prompt)
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
