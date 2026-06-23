# Florida Policy Assistant: Agentic RAG Build Plan

A working architecture and build plan. Each phase is a self-contained unit you can hand to Claude Code, with a verification checkpoint, the same way we did setup.

---

## 0. Context (the thing we are building)

An agentic RAG assistant over Florida government records and policy. The user is an internal records or policy analyst who needs fast, sourced answers to the questions staff and the public actually ask, instead of digging through statutes by hand. Every answer must cite the exact Title, Chapter, and Section it came from, and the system must say "not found in the source material" when the documents do not cover something. That abstain behavior is the credibility feature, not a nice-to-have.

Universal by design, seeded with Florida, spanning all state departments. Jurisdiction and department are metadata, so adding a second state later is just more ingestion into the same pipeline.

Goal beyond the build: a clickable, trustworthy demo good enough that a real Florida employee would vouch for it, which is what turns it into a referral.

---

## 1. Architecture at a glance

The flow is the agentic loop, not a straight pipeline:

1. Analyze the question. Classify it, split multi-part questions into sub-questions, and extract any metadata filters implied by the wording (a chapter, a topic, a department, a year).
2. Retrieve per sub-question. Hybrid search (vector + keyword) with metadata filtering, then rerank.
3. Grade the retrieved chunks for relevance and sufficiency.
4. Decide. If the chunks are weak, rewrite the query or broaden the filter and retry (capped). If still nothing after retries, route to abstain.
5. Generate a grounded answer with citations, combining sub-answers if the question was decomposed.
6. Abstain cleanly when the evidence is not there.

The grade-and-retry-or-abstain step is the whole reason this is agentic rather than plain RAG.

---

## 2. The data and what we ingest

Source: Florida Statutes on Online Sunshine (leg.state.fl.us), with the Florida Administrative Code (flrules.org) as a later layer.

Structure, confirmed from the real site: a three-level tree of Title, then Chapter, then Section. The Section is the atomic readable unit of law. The URLs are regular and parseable, following the pattern `.../0000-0099/0006/Sections/0006.01.html`, which decodes as chapter-group, chapter, section. This regularity means ingestion can walk the tree by pattern with no manual clicking.

Each Section page has a consistent internal shape: a section number, the title, a dash delimiter, the body text, and a trailing `History.` line with amendment citations.

Seed corpus to start (representative spread across departments, not everything):
- Title X, Public Officers, Employees, and Records (Chapters 110 to 122). Includes Chapter 119, the Public Records Act. This is the analyst's bread and butter.
- Title IV, Executive Branch (Chapters 14 to 24). The departments themselves.
- A few more Titles spanning different subject areas so the routing layer has a real job (for example procurement, ethics, public meetings).

You do not need all 49 Titles for the system to be all-departments by design or to demo convincingly. Scale ingestion up anytime; it is just more documents into the same pipeline.

---

## 3. Metadata schema (the spine)

Every stored chunk carries these fields, all of which fall straight out of the source structure:

- jurisdiction: "FL" (the universal-by-design hook)
- title_num: "X"
- title_name: "Public Officers, Employees, and Records"
- chapter_num: "119"
- chapter_name: "Public Records"
- section_num: "119.07" (stored as a string, since Florida inserts decimal sub-numbers like 6.075 between sections)
- section_title: "Inspection and copying of records; photographing..."
- department: tagged where applicable (the universal-by-design hook for cross-department routing)
- text: the clean body, history line stripped out
- history: kept as a side field, excluded from answer text
- source_url: the exact section URL, for citation links

This schema is what powers precise citations and lets the agent filter to the right place before retrieving.

---

## 4. Pipeline, component by component

### 4.1 Ingestion
Fetch section pages (or whole-chapter pages via the "View Entire Chapter" link, which is cleaner), then isolate the law from the surrounding site navigation. The real challenge here is stripping the menu and boilerplate chrome, not the law itself, since the legal content sits in a predictable spot on each page. Parse out number, title, body, and history using the consistent section shape. Output one clean record per section with the full metadata above. Your data-cleaning background is the asset here.

### 4.2 Chunking
One Section equals one chunk, in the normal case. Sections are short, a paragraph to roughly a page, so they fit a chunk and should not be split mid-thought, because splitting legal text mid-clause wrecks meaning. The exception is the rare long section that runs for pages with many subsections; split those on subsection boundaries only, and carry the same parent metadata onto each piece. Never use blind fixed-size splitting on this corpus.

### 4.3 Embedding and indexing
Embed each chunk locally with sentence-transformers (bge-small-en-v1.5), free and on CPU. Store vectors plus all metadata fields in ChromaDB (local, free). Metadata goes in as filterable fields so retrieval can hard-filter by title, chapter, department, or jurisdiction.

### 4.4 Retrieval
Hybrid retrieval: dense vector search for meaning plus BM25 keyword search (rank_bm25) for exact terms like section numbers and defined phrases. Apply metadata filters first when the question implies them (for example, restrict to Chapter 119). Then rerank the combined candidates with a cross-encoder reranker (sentence-transformers, CPU) to push the best chunks to the top. Document where keyword search rescues queries that vector search misses; that comparison is portfolio gold.

### 4.5 The agentic loop (LangGraph)
Build the loop as a LangGraph state graph with these nodes:
- analyze: classify and decompose the question, extract metadata filters.
- retrieve: run 4.4 for each sub-question.
- grade: an LLM step that judges whether the retrieved chunks are actually relevant and sufficient to answer.
- decide (conditional edge): good enough goes to generate; weak triggers a query rewrite or filter broadening and loops back to retrieve; exhausted retries route to abstain.
- generate: produce a grounded, cited answer, combining sub-answers.
- abstain: return a clean "not found in the source material" with what was searched.

Use shared graph state carrying the question, sub-questions, retrieved chunks, grades, and retry count. Cap retries and total LLM calls so it cannot loop forever or run up usage. There is a natural slot to add tools later (for example a citation-lookup or a date or threshold calculator), which is also the seed for the multi-agent project down the road.

### 4.6 Generation
Generation runs on Groq (llama-3.3-70b-versatile), free. The prompt forces three behaviors: answer only from the retrieved chunks, cite the Title, Chapter, and Section for every claim, and explicitly refuse to answer when the chunks do not contain the answer rather than guessing. No invented statute numbers, ever. This grounding-and-abstain discipline is what makes an insider trust it.

---

## 5. Evaluation

Build a test set of 40 to 60 question and answer pairs reflecting real analyst questions across several departments, including some deliberately unanswerable ones to test abstention. Measure with RAGAS: context precision, context recall, faithfulness (is the answer grounded in retrieved text), and answer relevance. Also track an abstention rate on the unanswerable set (a good system says "not found" instead of inventing).

The headline result: run plain RAG (phase 3) against the full agentic loop (phase 4) on the same test set and report them side by side. "The agentic loop raised completeness on multi-part questions from X to Y and cut confidently-wrong answers by Z percent" is the single most convincing thing in the whole project.

Note: RAGAS uses an LLM to score some metrics, so point it at Groq, and mind the free rate limits by running the eval in small batches.

---

## 6. Serving and deployment

API layer in FastAPI. UI in Streamlit, which doubles as the clickable demo. Deploy free on Streamlit Community Cloud so the demo can travel by link, which is the whole point for the referral goal.

Deployment notes to handle honestly:
- The Streamlit host runs the embedding model and calls Groq's API, both of which work the same deployed as locally, since Groq is an API.
- The ChromaDB index needs to be available to the deployed app. For a modest seed corpus, bundle the persisted Chroma directory in the repo and load it read-only, or rebuild it on first startup. A huge corpus would need a different approach, but the seed will be small enough.
- Expect a cold-start delay the first time, since the embedding model (about 130 MB) downloads on the host. Fine for a demo.

---

## 7. Universal by design

Two metadata facets make this scale beyond Florida and beyond one department without any architecture change:
- jurisdiction lets you add a second state later; the agent's routing then has to resolve which jurisdiction applies before retrieving.
- department lets the agent route across departments, resolving which department's policy governs a question before retrieving.

Both turn into real, impressive agentic routing problems the moment you load a second jurisdiction or span enough departments. You demo "works across jurisdictions and departments" while having loaded only a representative slice.

---

## 8. Build order (phased, for Claude Code)

Each phase ends with a verification checkpoint before moving on.

- Phase 1, Ingestion. Fetch and parse one chapter (use Chapter 119) into clean section records with full metadata. Verify: records have correct number, title, body, history stripped, and metadata populated.
- Phase 2, Index. Chunk (one section per chunk, split long sections), embed, store in ChromaDB with metadata. Verify: a plain similarity query returns the right sections.
- Phase 3, Baseline RAG. Hybrid retrieval plus reranking plus grounded, cited generation, with no agent yet. Verify: sensible cited answers on simple questions. This is also your plain-RAG comparison baseline.
- Phase 4, Agentic loop. Wrap phase 3 in the LangGraph graph: analyze, retrieve, grade, retry, abstain. Verify: multi-part questions get decomposed, weak retrievals trigger a retry, and unanswerable questions abstain.
- Phase 5, Evaluation. Build the test set, run RAGAS, produce the plain-versus-agentic comparison table. Verify: real numbers, including abstention rate.
- Phase 6, App and deploy. FastAPI plus Streamlit, deploy free, write the README. Verify: a working public link a stranger can use.

Hand Claude Code one phase at a time, review what it writes, and keep yourself the one making the design calls so you can explain every part in an interview.

---

## 9. Suggested repo structure

```
fl-policy-rag/
  ingest/        # fetching and parsing statute pages
  index/         # chunking, embedding, ChromaDB build
  retrieval/     # hybrid search + reranker
  agent/         # LangGraph loop (analyze, grade, decide, generate, abstain)
  eval/          # test set + RAGAS harness + comparison
  app/           # FastAPI + Streamlit
  data/          # raw and processed corpus, persisted Chroma index
  .env           # keys (gitignored)
  .gitignore
  README.md
```

