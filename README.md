# LlamaIndex RAG Demo

A small local RAG project powered by LlamaIndex + ChromaDB + Google Gemini. It supports hybrid retrieval (BM25 + dense), HyDE, useful postprocessors, and returns answers with cited sources.

## Features

- Hybrid retriever: BM25 + dense via QueryFusionRetriever (reciprocal rerank)
- HyDE (including parallel hypothesis generation)
- Postprocessors: neighbor windows (AutoPrevNext with a safe wrapper), Similarity cutoff, LongContextReorder, optional cross‑encoder rerank
- Metadata filters: `file_name`, `section`, `lang`, `version`
- JSON answers with sources (`answer` + `sources[]`)
- Optional agent mode (AgentWorkflow)

## Requirements

- macOS, Python 3.13
- Google API key: `GOOGLE_API_KEY`
- Documents placed under `data/` (PDFs, etc.; loaded with `SimpleDirectoryReader`)

## Installation

```zsh
# 1) (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2) install dependencies
pip install -r requirements.txt -r requirements-dev.txt
```

## Configuration (.env)

All variables are optional unless noted otherwise.

- GOOGLE_API_KEY (required) — key for Gemini LLM/embeddings
- TOP_K (default 15) — retrieval top‑k
- USE_FUSION (true) — hybrid BM25 + dense
- USE_HYDE (true) — enable HyDE
- PARALLEL_HYDE (true) — run HyDE in parallel with base retrieval
- USE_RERANK (true) — cross‑encoder rerank
- RESPONSE_MODE (compact) — response synthesis mode
- CHROMA_PATH (./chroma_db) — Chroma path
- CHROMA_COLLECTION (test) — Chroma collection name
- PERSIST_DIR (./storage) — LlamaIndex docstore/index directory
- AGENT_ENABLED (false) — enable agent mode
- LOG_LEVEL (INFO) — logging level

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

## Quick start

```zsh
# activate venv and export the key (or rely on .env)
export GOOGLE_API_KEY="<your_key>"

# run an example query
python app.py "What is the scope of work?"
```

The output is JSON with `answer` and `sources` fields.

## Project structure

```text
.
├── app.py                # CLI wrapper over rag_engine
├── rag_engine.py         # Indexing/retrieval/synthesis logic
├── data/                 # Local documents
├── chroma_db/            # Persistent Chroma store
├── storage/              # LlamaIndex docstore/index
├── tests/                # Unit/integration tests
├── requirements.txt
├── requirements-dev.txt
└── docs/Deliverables.md
```

## How it works

1) On the first run, documents from `data/` are indexed into Chroma (`./chroma_db`), while metadata/docstore are persisted to `./storage` to enable neighbor-window postprocessing.
2) The retriever combines dense retrieval from the index and BM25 (if `USE_FUSION=true`) using reciprocal rerank.
3) If `USE_HYDE=true`, a hypothetical answer is generated (HyDE) and used to perform another retrieval; results are merged and synthesized once.
4) Postprocessors reorder/filter results and add adjacent context chunks.
5) The final answer is returned along with cited sources.

## Testing

```zsh
pytest -q
```

The integration test uses MockLLM and BM25 for an offline run.

## Quality and performance tips

- Higher answer quality: increase `TOP_K`, enable `USE_HYDE` and `USE_RERANK`
- Lower latency: disable HyDE/rerank, reduce `TOP_K`
- If needed, you can disable Chroma telemetry via `PersistentClient(..., settings=...)`

## Troubleshooting

- OpenAI API key error: this project uses Gemini; ensure `GOOGLE_API_KEY` is set and that LlamaIndex settings aren’t overridden before initialization
- Empty answers: verify `data/` has readable documents; temporarily disable `USE_FUSION`
- Slow responses: disable `USE_HYDE` and `USE_RERANK`, lower `TOP_K`

## Links

- LlamaIndex docs (retrievers/HyDE/postprocessors/Chroma)
