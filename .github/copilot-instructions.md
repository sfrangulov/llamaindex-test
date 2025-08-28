# AI agent guide for this repo

Local RAG app using LlamaIndex + ChromaDB with two UIs:
- `kb_ui.py` — chat Q&A over the index (Gradio, port 7860)
- `docs_ui.py` — document manager (Dash, port 7861)

Big picture
- Ingestion: `.docx` in `attachments/` → `md_reader.MarkItDownReader` → `Document(text, metadata)`; markdown copies saved to `data/markdown/<file_name>.docx.md`.
- Index: `storage.get_index()` attaches to/create Chroma collection (persistent; telemetry off). On first run, it loads DOCX, writes markdown, and builds a `VectorStoreIndex` over Chroma.
- Retrieval: `rag_engine.RagWorkflow` steps: reformulate → retrieve (top-k) → optional rerank → synthesize → `{answer, sources[]}`.
- UIs: both call `warmup(ensure_index=True)` to pre-init Settings and index.

Key configs (env or defaults)
- Storage (see `storage.Config`): `DATA_PATH=./attachments`, `CHROMA_PATH=./data/chroma_db`, `CHROMA_COLLECTION=knowledge_base`, `MD_DIR=./data/markdown`.
- RAG tuning (see `rag_engine.py`): `TOP_K` (default 10), `RERANK_ENABLED=true`, `GOOGLE_API_KEY` (required for Gemini), `HOST`, `PORT` (UI bind).
- Models: default `configure_settings_gemini()` uses Google embedding + `gemini-2.5-flash`; `configure_settings_local()` switches to HF embeddings + sentence-transformer reranker.

Core helpers to reuse (do not duplicate logic)
- `storage.get_index()` — singleton index over Chroma; don’t rebuild per request.
- `storage.add_docx_to_store(path)` — converts DOCX, saves markdown, deletes old vectors by `file_name`, then inserts.
- `storage.list_storage_files(search)` — filesystem listing of DOCX under `attachments/`.
- `storage.list_vector_file_names()` — distinct `file_name` values present in Chroma.
- `storage.delete_from_vector_store_by_file_names([...])` — delete embeddings by `metadata.file_name` (Chroma `where`).
- `storage.read_markdown(file_name)` — load saved markdown for preview.

Important conventions and gotchas
- Metadata from `MarkItDownReader`: `file_name`, `file_path`, `file_size`, `file_mtime_iso`, optional `sha256`; `Document.id_` is `sha256` when available.
- Idempotency: before insert, vectors with the same `file_name` are removed to avoid duplicates.
- Never delete originals from `attachments/` via vector deletions; deletions are metadata-filtered in Chroma only.
- Always call `ensure_dirs()` before file ops; `get_index()` does this internally.

Dev workflow
- Python 3.13. Install with `pip install -r requirements.txt`. Start UIs: `python kb_ui.py` (7860), `python docs_ui.py` (7861). Seed by placing `.docx` in `attachments/`.
- Programmatic RAG: `await RagWorkflow().run(input={"query": "..."})` or `await search_documents("...")` → JSON string with `answer` and `sources`.

Guardrails for AI edits
- Don’t add parallel ingestion paths—use `add_docx_to_store()` to keep markdown and vectors consistent.
- Don’t change default paths/collection without updating both UIs and `storage.Config` consumers.
- Prefer toggling models via `configure_settings_*()` rather than inlining new settings.

If any of the above is unclear (e.g., switching defaults to local models, adding filters/chunking changes), request confirmation before refactoring.
