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
