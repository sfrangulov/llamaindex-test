# Refactor Deliverables (A–F)

This document captures the requested structured outputs.

## A. Strategy matrix and selection

- Candidates (10):
  1) Remove unused imports and dead variables. Impact: low risk, quick gains.
  2) Collapse micro-wrappers; prefer direct library constructs.
  3) Minimize global state; keep runtime flags in env with sane defaults.
  4) Pin minimal runtime deps; split dev tools.
  5) Persist and reuse docstore/vector-store cleanly; avoid re-index when present.
  6) Cache BM25 nodes across queries; filter at runtime.
  7) Keep postprocessors minimal: rerank (optional), similarity cutoff, reorder.
  8) Avoid hidden coupling; expose a single async search API.
  9) Encode responses with explicit sources for UI.
  10) Keep agent optional/off by default.
- Weighted selection: 1, 3, 4, 5, 6, 7, 8, 9, 10 chosen first pass. (2) considered but retained SafeAutoPrevNext as a guard.

## B. Patch summary (unified diff available via VCS)

- app.py: removed unused logging bits and one dead variable; kept behavior.
- requirements.txt: rebuilt minimal, pinned runtime deps with comments.
- requirements-dev.txt: added dev tooling pins (ruff, black, mypy).
- README.md: fixed env defaults to match code; clarified docs location.

## C. Dependencies (runtime vs dev) and justification

Runtime

- llama-index-core==0.13.2 — core APIs used (VectorStoreIndex, Settings, StorageContext).
- llama-index-vector-stores-chroma==0.5.0 — ChromaVectorStore integration.
- llama-index-llms-google-genai==0.3.0 — GoogleGenAI LLM driver.
- llama-index-embeddings-google-genai==0.3.0 — Embeddings for text-embedding-004.
- llama-index-retrievers-bm25==0.6.3 — Sparse BM25 retriever.
- chromadb==1.0.17 — persistent vector DB.
- python-dotenv==1.1.1 — .env support.
- sentence-transformers==5.1.0 — optional cross-encoder reranker.

Dev

- black==24.10.0 — formatting.
- ruff==0.5.7 — linting.
- mypy==1.13.0 — optional type checks.

## D. Refactor report

- External behavior preserved: async `search_documents()` contract unchanged; CLI unchanged; env flags honored.
- Simplifications: removed unused code; clarified error paths; consistent metadata defaults; safer neighbor-windowing.
- No new runtime deps introduced; minimal pins align with imports. Dev tools isolated.

## E. Regression plan

- Sanity: run aquery on a sample query; expect JSON with `answer` and non-empty `sources` when data present.
- Edge cases: empty collection triggers indexing; reranker missing → still returns answer; filters that match zero docs → empty sources acceptable.
- Lint/format: ruff and black clean.
- Optional: add unit tests around `_make_filters` and `_build_sources`.

## F. Sources (preflight)

- LlamaIndex Releases: [github.com/run-llama/llama_index/releases](https://github.com/run-llama/llama_index/releases) — core 0.13.2; bm25 0.6.3 persist/load fix.
- ChromaDB PyPI 1.0.17: [pypi.org/project/chromadb/1.0.17](https://pypi.org/project/chromadb/1.0.17/)
- Sentence-Transformers v5.1.0: [sbert.net/docs/releases.html](https://www.sbert.net/docs/releases.html) and [pypi.org/project/sentence-transformers/5.1.0](https://pypi.org/project/sentence-transformers/5.1.0/)
- python-dotenv 1.1.1: [pypi.org/project/python-dotenv/1.1.1](https://pypi.org/project/python-dotenv/1.1.1/)
- Black 24.10.0 changelog: [black.readthedocs.io/change_log](https://black.readthedocs.io/en/stable/change_log.html)
- Ruff 0.5.7 release: [github.com/astral-sh/ruff/releases](https://github.com/astral-sh/ruff/releases) (select v0.5.7)
- Mypy 1.13.0 release: [github.com/python/mypy/releases/tag/v1.13.0](https://github.com/python/mypy/releases/tag/v1.13.0)
- Chroma persistence: [docs.trychroma.com/reference/clients/python#persisting-data](https://docs.trychroma.com/reference/clients/python#persisting-data) and [docs.trychroma.com/guides/persistence](https://docs.trychroma.com/guides/persistence)
