# Copilot instructions for this repo

Purpose: Make AI agents productive in this Dash + LlamaIndex app. Keep edits minimal, Russian text in UI, and JSON contracts stable.

## Big picture
- UI: single Dash app (`app.py`) with 3 tabs: Анализ (`fs_analyze_ui.py`), Чат (`fs_chat_ui.py`), Хранилище (`fs_storage_ui.py`).
- RAG: `rag_engine.py` (LlamaIndex + Chroma) → reformulate → retrieve (optional file filter) → rerank → synthesize → JSON {answer, sources}.
- Ingestion/storage: `storage.py` converts DOCX→Markdown via `md_reader.MarkItDownReader`, writes `data/markdown/`, upserts vectors in Chroma, lists/deletes files.
- FS analyzer: `fs_analyze_agent.py` runs per-section checks using `fs_analyze/*` and `fs_utils.SECTION_TITLES`; results saved to SQLite (`db.py`).

## Startup and env
- Entry: `python app.py` after installing `requirements.txt`. Main calls `rag_engine.start()`, `fs_analyze_agent.start()`, `db.start()`.
- .env keys: Google GenAI for Gemini (if used). Important vars: `TOP_K`, `RERANK_ENABLED`, `DATA_PATH` (./attachments), `CHROMA_PATH` (./data/chroma_db), `CHROMA_COLLECTION`, `MD_DIR` (./data/markdown), `HOST`, `PORT`.
- First run builds index from `attachments/*.docx` and writes Markdown to `data/markdown/`.

## Workflows (how to use)
- Add docs: drop `.docx` into `attachments/` or upload in “Анализ”. Internally uses `storage.add_docx_to_store()` (persist, write md, delete old vectors, insert new).
- Analyze FS: button in “Анализ” → background worker → `fs_analyze_agent.analyze_fs_sections()` → `db.save_fs_analysis()`; polling via `ANALYSIS_PROGRESS`.
- Ask questions: “Чат” and QA panel call `rag_engine.search_documents()`; pass `file_name` to scope retrieval to one FS.
- Manage storage: “Хранилище” lists via `storage.list_documents()`; preview uses `storage.read_markdown()`; delete uses `storage.delete_document()`.

## Conventions and patterns
- Tabs expose `get_layout()` + `register_callbacks(app, ...)`. Keep state in `dcc.Store`; mirror busy flags to button props.
- Component IDs for row actions are dicts (e.g., `{type:"view-btn", section:title}`); resolve with `callback_context.triggered_id`.
- Markdown rendered with `dcc.Markdown` within `dmc.TypographyStylesProvider` (see `assets/fs_analyze_ui.css`).
- RAG settings set once in `configure_settings()`; default Gemini models; local HF alternative in `configure_settings_local()`.
- Retrieval filter: LlamaIndex `MetadataFilters([{key:"file_name", value: name, operator: EQ}])`.
- Sources include `file_name`, `score`, snippet text; JSON from `search_documents()` is `{"answer": str, "sources": list}`.

## Key files
- `app.py`, `rag_engine.py`, `storage.py`, `md_reader.py`, `fs_analyze_agent.py`, `fs_analyze_ui.py`, `fs_chat_ui.py`, `fs_storage_ui.py`, `db.py`, `fs_analyze/*`.

## Examples
- Scoped search: `await search_documents("Вопрос", file_name="Some.docx")`.
- Programmatic ingest: `add_docx_to_store(Path("./attachments/Foo.docx"))`.
- Full delete: `delete_document("Foo.docx")` → removes vectors + md + original.

Notes
- No test suite yet (VS Code task exists for `pytest -q`). Lint with ruff; follow existing typing and structlog usage.
- If conventions drift (new tabs, envs), update this file in the same concise style (≈20–50 lines).
