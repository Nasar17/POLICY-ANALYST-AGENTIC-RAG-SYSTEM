# Florida Statute Research Assistant

Agentic RAG over Florida Statute Chapter 119 — Public Records Act.

Ask a question in plain English. The system retrieves the relevant statute sections, reasons about whether they answer the question, and returns a grounded, cited answer. When the question isn't covered by the statute, it says so rather than guessing.

---

## Live demo

> Deploy to Streamlit Community Cloud (see [Deployment](#deployment) below) and paste the URL here.

---

## Architecture

```
Question
   │
   ▼
[analyze]  — decompose into sub-questions, extract metadata filters
   │
   ▼
[retrieve] — hybrid search: dense (ChromaDB) + BM25 → RRF fusion → cross-encoder rerank
   │
   ▼
[grade]    — LLM judges whether chunks are sufficient to answer
   │
   ├── sufficient ──▶ [generate] → cited answer
   │
   ├── insufficient + retries left ──▶ [rewrite] → back to retrieve
   │
   └── insufficient + retries exhausted ──▶ [abstain] → "not found"
```

**Stack:**
| Component | Library |
|---|---|
| Embedding | `BAAI/bge-small-en-v1.5` (sentence-transformers, CPU) |
| Vector store | ChromaDB (local persistent) |
| Keyword search | BM25Okapi (rank-bm25) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Agentic loop | LangGraph |
| LLM | Groq — `llama-3.1-8b-instant` |
| API | FastAPI |
| UI | Streamlit |

---

## Project structure

```
fl-policy-rag/
  ingest/          HTML fetch + section parser (Chapter 119)
  index/           Chunking, embedding, ChromaDB population
  retrieval/       HybridRetriever, RAGPipeline, generator
  agent/           LangGraph state graph (analyze/retrieve/grade/rewrite/generate/abstain)
  eval/            RAGAS-equivalent evaluation harness + test set
  app/
    api.py         FastAPI backend  (POST /ask)
    streamlit_app.py  Streamlit demo UI
  data/
    ch119_sections.json  Parsed statute sections (20 sections)
    chroma_db/     Persisted ChromaDB index (96 chunks, ~2.8 MB)
  .streamlit/
    config.toml    Streamlit theme
  requirements.txt
```

---

## Quick start

### 1. Clone and set up

```bash
git clone <repo-url>
cd fl-policy-rag
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add your Groq API key

```bash
# .env  (gitignored)
GROQ_API_KEY=gsk_...
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`.

### 4. Or run the FastAPI backend

```bash
uvicorn app.api:app --reload --port 8000
```

API docs at `http://localhost:8000/docs`.

Example request:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a public record under Florida law?", "mode": "agentic"}'
```

---

## Evaluation results (Phase 5)

Plain RAG vs. Agentic RAG on 15 answerable + 5 unanswerable questions from Chapter 119.

| Metric | Plain RAG | Agentic | Delta |
|---|---|---|---|
| Faithfulness | 0.800 | 0.780 | -0.020 |
| Answer Relevance | 0.869 | 0.869 | 0.000 |
| Context Precision | 0.687 | 0.607 | -0.080 |
| Context Recall | 0.680 | 0.467 | -0.213 |
| **Abstention Rate** | **100%** | **100%** | ±0% |

Both systems correctly refused all 5 unanswerable questions. The grounding discipline — answer only from retrieved text, abstain when coverage is absent — is the core credibility feature.

The plain pipeline slightly outperforms on context metrics because `llama-3.1-8b-instant` (used on the Groq free tier) is too small to reliably drive the analyze/grade/rewrite decisions. Swapping in a larger judge model would likely reverse this.

---

## Deployment

### Streamlit Community Cloud

1. Push this repo to GitHub (ChromaDB is bundled at `data/chroma_db/`, ~2.8 MB).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select repo, branch `main`, main file `app/streamlit_app.py`.
4. In **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
5. Deploy. First cold start downloads the embedding model (~130 MB); subsequent starts use the cache.

---

## Extending

- **Add more chapters**: run `ingest/` on additional chapters and `index/` to add chunks to ChromaDB. No architecture changes needed.
- **Add more jurisdictions**: set `jurisdiction` metadata on new chunks; the retriever's `where` filter routes to the right corpus.
- **Upgrade the LLM**: swap `MODEL` in `agent/nodes.py` and `retrieval/generator.py` to a larger Groq model for better agentic reasoning.
