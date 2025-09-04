# Repo AI Assistant Instructions

Purpose: Give AI/code assistants precise guidance for this project so changes stay consistent with our ingestion, indexing, and UI rendering flows.

## Project snapshot

- Python app (Dash + LlamaIndex + ChromaDB) to analyze functional specs/docs.
- Ingestion converts attachments (mostly .docx) to Markdown and indexes content.
- Images from .docx are extracted to disk and referenced via HTTP for the UI.

## Key files

- `storage.py` — builds/loads vector store and index; orchestrates ingestion and markdown persistence.
- `md_reader.py` — MarkItDown-based reader with DOCX image extraction and Markdown link rewrite.
- `app.py` — Dash app and Flask route to serve `/markdown_assets/<stem>/<file>`.
- `requirements.txt` / `_requirements.txt` — dependencies.
- `data/markdown/` — persisted Markdown; `data/markdown_assets/` — extracted images.

## Ingestion + images (rules)

- Use `MarkItDownReader` from `md_reader.py` for `.docx`/`.xlsx`.
- Extract images from DOCX zip (`word/media/*`) to `data/markdown_assets/<doc-stem>/` with SHA256-based dedupe.
- Rewrite Markdown `![](data:image/...)` to HTTP URLs: `/markdown_assets/<stem>/<filename>`.
- The app must serve these via the Flask route in `app.py`.
- Do not embed base64 images in Markdown for the UI; always rewrite to file-backed links.

## Metadata constraints

- LlamaIndex/Chroma metadata must be scalar: `str | int | float | None`.
- If you need to store lists/objects, serialize to JSON string (e.g., `images_saved`), and add numeric helpers (e.g., `images_count`).

## Environment variables

- `DATA_PATH` (default: `./data`)
- `CHROMA_PATH` (default: `./data/chroma_db`)
- `CHROMA_COLLECTION` (default: `knowledge_base`)
- `MD_DIR` (default: `./data/markdown`)
- `MD_SAVE_IMAGES` (default: `1` to enable extraction)
- `MD_IMG_DIR` (default: `./data/markdown_assets`)

## Implementation guidance

- Prefer small, targeted changes; keep public behavior/backwards-compat.
- Preserve `md_reader.py` helpers: `_extract_docx_images`, `_rewrite_md_data_uri_placeholders`, `_process_docx_images`.
- Keep URL shape stable: `/markdown_assets/<stem>/<filename>`.
- When adding new formats, mirror the DOCX flow: extract assets, persist, rewrite links, and serve via the static route.
- Gracefully handle missing files and unknown image types; skip rather than crash.

## UI serving

- The Flask route in `app.py` is the single source of truth for asset serving. If you change paths, update both the route and Markdown rewrite logic.

## Testing/checks

- Prefer fast, minimal checks. If adding code that changes public behavior, add a small test (pytest) where feasible.
- Honor the `pytest -q` task if tests are present.

## Common pitfalls

- Don’t store non-scalars in metadata.
- Don’t create file-system relative links in Markdown that aren’t served by HTTP.
- Avoid noisy regex; account for wrappers like bold around images when rewriting.

## Docs-first via Context7 (MCP)

When a question or task involves a third-party library, framework, SDK, or API:

- Always fetch up-to-date docs using the MCP server **context7** before proposing code.
- Procedure:
  1. Detect the target library and version from the repo (e.g., package.json, pyproject.toml, requirements.txt, go.mod, Cargo.toml).
  2. Call `resolve-library-id` with `libraryName` for the target.
  3. Call `get-library-docs` for the resolved ID (and version if available).
  4. Ground explanations and code examples strictly in the fetched docs.
- If docs cannot be retrieved, explicitly say so and continue with best-effort, marking the answer with: `Docs: not found in Context7`.
- Prefer examples that match this stack: TypeScript/React/Next.js and Python (FastAPI) unless the user states otherwise.
- At the end of the answer, include a brief note: `Docs: <lib>@<version> via Context7`.

Do not rely on model memory or generic web snippets for APIs if Context7 docs were retrieved.

## Nice-to-have next

- Optional EMF/WMF → PNG conversion during extraction for better browser support (only if needed).
