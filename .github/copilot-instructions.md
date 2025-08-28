# AI agent working guide for this repo

This project is a small local RAG app using LlamaIndex + ChromaDB with two UIs:
- `kb_ui.py` — Q&A over indexed docs (chat style)
- `docs_ui.py` — document manager (list/upload/index/delete/preview)

Storage and indexing live in `storage.py`; retrieval and synthesis are in `rag_engine.py`. Prefer reusing these modules instead of duplicating logic.

## Architecture in one glance
- Ingestion
  - Source files: `attachments/` (DOCX). Markdown copies saved to `data/markdown/`.
  - `md_reader.MarkItDownReader` converts `.docx` to markdown text with rich metadata (file_name, size, mtime, sha256).
  - `storage.get_index()` initializes a persistent Chroma collection (path via `CHROMA_PATH`, collection via `CHROMA_COLLECTION`). On first run it loads `.docx` from `attachments/`, writes `.md` files, and builds a `VectorStoreIndex`.
  - `storage.add_docx_to_store(path)` converts and inserts a single file, saves `.md`, and upserts into the current index; it also optionally persists the original file in `attachments/`.
- Retrieval
  - `rag_engine.configure_settings_*()` configures `Settings.embed_model`, `Settings.llm`, `Settings.ranker`, and `Settings.node_parser`. Default is local HF models; Gemini variants exist but require `GOOGLE_API_KEY`.
  - `RagWorkflow` orchestrates: reformulate query → retrieve (top-k from index) → optional rerank → synthesize. Returns JSON with `answer` and `sources`.
- UIs
  - `kb_ui.py` launches chat Q&A (port 7860). Uses `warmup()` to pre-init settings/index.
  - `docs_ui.py` launches document manager (port 7861). Tabs: list (with search/sort), upload+index, delete from vector store (by file_name), preview Markdown.

## Conventions and important details
- Paths and config
  - Controlled via env or defaults in `storage.Config`: `DATA_PATH=./attachments`, `CHROMA_PATH=./data/chroma_db`, `CHROMA_COLLECTION=knowledge_base`, `MD_DIR=./data/markdown`.
  - Helper `ensure_dirs()` must be called before file ops.
- Vector store metadata
  - Each node carries `metadata["file_name"]` and others from the reader. Deletions target `file_name` via Chroma `where` filters.
  - `list_vector_file_names()` enumerates distinct `file_name` values from Chroma metadatas.
- Markdown persistence
  - Markdown is always saved as `data/markdown/<file_name>.md` where `<file_name>` includes the `.docx` suffix.
- Idempotency
  - `add_docx_to_store()` deletes existing vectors with the same `file_name` before insert to avoid duplicates.

## Developer workflows (do this first)
- Python 3.13 on macOS; create a venv and install deps from `requirements.txt`.
- Local run
  - Docs UI: `python docs_ui.py` (http://localhost:7861)
  - Chat UI: `python kb_ui.py` (http://localhost:7860)
- Retrieval engine
  - Uses local HF defaults. For Gemini, set `GOOGLE_API_KEY` and switch to `configure_settings_gemini()` in `rag_engine.configure_settings()`.
- Data seeding
  - Put `.docx` files into `attachments/` and start either UI; first run will index and generate markdown.

## Coding patterns to follow
- Reuse the public helpers in `storage.py` for:
  - Listing files: `list_storage_files(search)`
  - Adding files: `add_docx_to_store(path)`
  - Listing vector entries: `list_vector_file_names()`
  - Deleting vectors: `delete_from_vector_store_by_file_names([...])`
  - Reading Markdown: `read_markdown(file_name)`
- Always call `warmup(ensure_index=True)` before serving requests to avoid first‑hit latency.
- For new Gradio components, be mindful of v5 API (e.g., Dataframe lacks a `height` arg).

## External dependencies and versions
- LlamaIndex components: core, workflows, vector-stores-chroma, readers-file
- ChromaDB: persistent client; telemetry disabled via `ChromaSettings(anonymized_telemetry=False)`
- markitdown: DOCX → Markdown conversion
- Gradio v5 for UIs

## Examples from codebase
- Index init: `storage.get_index()` will either build from `attachments/` or attach to existing vectors.
- Delete by metadata: `collection.delete(where={"file_name": name})` (see `delete_from_vector_store_by_file_names`).
- RAG call: `await RagWorkflow().run(input={"query": "..."})` → JSON answer with ranked sources.

## Guardrails for agents
- Do not create parallel ingestion flows; use `add_docx_to_store` to keep markdown and vectors consistent.
- Do not delete source files; vector deletions should target only embeddings unless explicitly asked.
- Avoid changing default paths unless coordinated; UIs and storage helpers assume the defaults.

If anything is unclear (e.g., switching to Gemini everywhere, adding filters, or changing chunking), ask for confirmation before refactors.
