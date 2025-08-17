# LlamaIndex RAG Demo

This project demonstrates a production-leaning RAG pipeline powered by LlamaIndex and ChromaDB.

What’s included

- Hybrid retriever (BM25 + dense) with reciprocal rerank and keyword weight alpha≈0.35.
- SentenceWindow and LongContextReorder postprocessors.
- Similarity cutoff + optional cross-encoder reranker.
- HyDE query transform and deterministic LLM settings.
- Metadata filters (file_name, section, lang, version).
- Answers return citations (source_nodes) for UI highlighting.

Quick start

1. Export your Google API key:

```zsh
export GOOGLE_API_KEY="<your_key>"
# Optional toggles (defaults shown)
export TOP_K=15
export USE_FUSION=true    # hybrid (dense+BM25) with reciprocal rerank
export USE_HYDE=true      # synthetic query expansion (adds LLM latency)
export USE_RERANK=true    # cross-encoder reranker (CPU/GPU heavy)
export AGENT_ENABLED=false
```
 
1. Run:

```zsh
python app.py
```

Notes

- The first run indexes files in `data/` into `./chroma_db` and persists metadata to `./storage` for neighbor-windowing.
- Adjust chunking and top_k in `app.py` if needed.
- If reranker model is missing, it’s gracefully disabled.
- Defaults are tuned for latency. For higher recall/quality, try increasing `TOP_K` (e.g., 25) and enabling `USE_HYDE` or `USE_RERANK`.
- Agent mode is off by default to save round trips; enable with `AGENT_ENABLED=true` if you want tool-calling behavior.
- Chroma telemetry is disabled for privacy; see `PersistentClient(..., settings=...)`.

Docs

- See `docs/Deliverables.md` for the refactor strategy, dependency justifications, regression plan, and sources.

Environment variables

All flags are optional unless marked required:

- GOOGLE_API_KEY (required) — used by Gemini LLM/embeddings.
- TOP_K (default: 15) — retrieval top-k.
- USE_FUSION (default: true) — enable hybrid BM25+dense via QueryFusionRetriever.
- USE_HYDE (default: true) — enable HyDE query transform.
- PARALLEL_HYDE (default: true) — overlap HyDE generation with retrieval.
- USE_RERANK (default: true) — enable cross-encoder reranker.
- RESPONSE_MODE (default: compact) — response synthesizer mode.
- CHROMA_PATH (default: ./chroma_db) — Chroma persistent path.
- CHROMA_COLLECTION (default: test) — Chroma collection name.
- PERSIST_DIR (default: ./storage) — path for persisted docstore/index.
- AGENT_ENABLED (default: false) — enable AgentWorkflow wrapper.
- LOG_LEVEL (default: INFO) — logging verbosity.

Example `.env`:

```env
GOOGLE_API_KEY=your_key_here
TOP_K=15
USE_FUSION=true
USE_HYDE=true
PARALLEL_HYDE=true
USE_RERANK=true
RESPONSE_MODE=compact
CHROMA_PATH=./chroma_db
CHROMA_COLLECTION=test
PERSIST_DIR=./storage
AGENT_ENABLED=false
LOG_LEVEL=INFO
```
