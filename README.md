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
export USE_FUSION=false   # enable to try multi-retriever reciprocal rerank
export USE_HYDE=false     # enable synthetic query expansion (adds LLM latency)
export USE_RERANK=false   # enable cross-encoder reranker (CPU/GPU heavy)
export AGENT_ENABLED=false
```

1. Install deps (ideally in a venv):

```zsh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

1. Run:

```zsh
python app.py
```

Notes

- The first run indexes files in `data/` into `./chroma_db`.
- Adjust chunking and top_k in `app.py` if needed.
- If reranker model is missing, it’s gracefully disabled.
- Defaults are tuned for latency. For higher recall/quality, try increasing `TOP_K` (e.g., 25) and enabling `USE_HYDE` or `USE_RERANK`.
- Agent mode is off by default to save round trips; enable with `AGENT_ENABLED=true` if you want tool-calling behavior.
- Chroma telemetry is disabled for privacy; see `PersistentClient(..., settings=...)`.
