from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dash
from dash import Dash, Input, Output, State, dcc, html
from dash.dash_table import DataTable
import dash_bootstrap_components as dbc

import structlog

import storage
from md_reader import MarkItDownReader
from rag_engine import warmup


log = structlog.get_logger(__name__)


# --------------------------- Utilities ---------------------------

def bytes_to_human(n: Optional[int]) -> str:
    if n is None:
        return "—"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return f"{size:.0f} {u}" if u == "B" else f"{size:.1f} {u}"
        size /= 1024.0


def build_table_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Enrich with display fields and action columns
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "file_name": r.get("file_name"),
                "uploaded_at": r.get("uploaded_at_iso"),
                "size": bytes_to_human(r.get("size_bytes")),
                "view": "View",
                "delete": "Delete",
            }
        )
    return out


def extract_file_from_row(data: List[Dict[str, Any]], row: Optional[int]) -> Optional[str]:
    if row is None:
        return None
    try:
        return data[row]["file_name"]
    except Exception:
        return None


# ----------------------------- App -----------------------------

external_stylesheets = [dbc.themes.BOOTSTRAP]
app: Dash = Dash(__name__, external_stylesheets=external_stylesheets,
                 suppress_callback_exceptions=True)
server = app.server


DOCS_COLUMNS = [
    {"name": "File name", "id": "file_name", "type": "text"},
    {"name": "Uploaded at", "id": "uploaded_at", "type": "text"},
    {"name": "Size", "id": "size", "type": "text"},
    # action pseudo-columns (clickable via active_cell)
    {"name": "View", "id": "view", "type": "text"},
    {"name": "Delete", "id": "delete", "type": "text"},
]


app.layout = dbc.Container(
    [
        dcc.Store(id="docs-cache"),  # raw docs list
        # timestamp to force refresh (initialized)
        dcc.Store(id="refresh-ts", data=time.time()),
        dcc.Store(id="pending-delete-file"),
        dcc.Store(id="preview-file"),
        dcc.Tabs(
            id="tabs",
            value="tab-docs",
            children=[
                dcc.Tab(label="Documents", value="tab-docs", children=[
                    dbc.Row([
                        dbc.Col(
                            dcc.Input(
                                id="search-input",
                                placeholder="Search by file name…",
                                type="text",
                                debounce=True,
                                style={"width": "100%"},
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            html.Div(id="action-hint", className="text-muted",
                                     style={"paddingTop": "6px"}),
                            width=6,
                        ),
                    ], className="mb-2"),
                    DataTable(
                        id="docs-table",
                        columns=DOCS_COLUMNS,
                        data=[],
                        page_size=20,
                        sort_action="native",
                        sort_mode="multi",
                        sort_by=[
                            {
                                "column_id": "uploaded_at",
                                "direction": "desc"
                            }
                        ],
                        page_action="native",
                        filter_action="none",
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "minWidth": "120px",
                            "maxWidth": "420px",
                            "whiteSpace": "nowrap",
                            "textOverflow": "ellipsis",
                        },
                        style_header={"fontWeight": "600"},
                        style_data_conditional=[
                            {
                                "if": {"column_id": col},
                                "color": "#0d6efd",
                                "textDecoration": "underline",
                                "cursor": "pointer",
                            }
                            for col in ["view", "delete"]
                        ],
                    ),

                    # Preview Modal
                    dbc.Modal(
                        [
                            dbc.ModalHeader(
                                dbc.ModalTitle(id="preview-title")),
                            dbc.ModalBody(dcc.Markdown(
                                id="preview-content", style={"whiteSpace": "pre-wrap"})),
                            dbc.ModalFooter(
                                dbc.Button("Close", id="close-preview",
                                           className="ms-auto", color="secondary")
                            ),
                        ],
                        id="preview-modal",
                        is_open=False,
                        size="xl",
                        scrollable=True,
                        fullscreen=False,
                    ),

                    # Delete confirm Modal
                    dbc.Modal(
                        [
                            dbc.ModalHeader(
                                dbc.ModalTitle("Confirm deletion")),
                            dbc.ModalBody(id="delete-confirm-body"),
                            dbc.ModalFooter(
                                [
                                    dbc.Button(
                                        "Cancel", id="cancel-delete", color="secondary", className="me-2"),
                                    dbc.Button(
                                        "Delete", id="confirm-delete", color="danger"),
                                ]
                            ),
                        ],
                        id="delete-modal",
                        is_open=False,
                        centered=True,
                    ),

                    # Toasts
                    html.Div(
                        [
                            dbc.Toast(
                                id="toast-success",
                                header="Success",
                                is_open=False,
                                dismissable=True,
                                duration=4000,
                                icon="success",
                                children="",
                                style={"position": "fixed", "top": 10,
                                       "right": 10, "zIndex": 2000},
                            ),
                            dbc.Toast(
                                id="toast-error",
                                header="Error",
                                is_open=False,
                                dismissable=True,
                                duration=6000,
                                icon="danger",
                                children="",
                                style={"position": "fixed", "top": 10,
                                       "right": 10, "zIndex": 2000},
                            ),
                        ]
                    ),
                ]),
                dcc.Tab(label="Upload", value="tab-upload", children=[
                    dbc.Row([
                        dbc.Col(
                            dcc.Upload(
                                id="upload",
                                children=html.Div([
                                    "Drag and drop or ", html.A("select files")
                                ]),
                                multiple=True,
                                style={
                                    "width": "100%",
                                    "height": "120px",
                                    "lineHeight": "120px",
                                    "borderWidth": "1px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "5px",
                                    "textAlign": "center",
                                },
                            ),
                            width=12,
                        ),
                    ], className="my-3"),
                    dbc.Button("Upload & Index",
                               id="upload-btn", color="primary"),
                    dcc.Loading(id="upload-loading", type="default",
                                children=html.Div(id="upload-status", className="mt-3")),
                ]),
            ],
        ),
    ],
    fluid=True,
)


# ---------------------------- Callbacks ----------------------------


@app.callback(
    Output("docs-cache", "data"),
    Output("action-hint", "children"),
    Input("refresh-ts", "data"),
    prevent_initial_call=False,
)
def refresh_documents(_: Optional[Any]):
    t0 = time.time()
    try:
        docs = storage.list_documents()
        log.info("documents_listed", action="documents_listed",
                 count=len(docs), duration_ms=int((time.time()-t0)*1000))
        return docs, f"{len(docs)} documents"
    except Exception as e:
        log.error("documents_list_failed", error=str(e))
        return [], "Failed to load documents"


@app.callback(
    Output("docs-table", "data"),
    Input("docs-cache", "data"),
    Input("search-input", "value"),
)
def filter_and_render_table(docs: List[Dict[str, Any]] | None, search: Optional[str]):
    docs = docs or []
    if search:
        s = search.strip().lower()
        docs = [d for d in docs if s in (d.get("file_name", "").lower())]
    return build_table_rows(docs)


@app.callback(
    Output("preview-modal", "is_open"),
    Output("preview-title", "children"),
    Output("preview-content", "children"),
    Output("delete-modal", "is_open"),
    Output("delete-confirm-body", "children"),
    Output("pending-delete-file", "data"),
    Input("docs-table", "active_cell"),
    State("docs-table", "data"),
    Input("close-preview", "n_clicks"),
    Input("cancel-delete", "n_clicks"),
    prevent_initial_call=True,
)
def handle_table_click(active_cell, table_data, close_preview, cancel_delete):
    # Defaults
    open_preview = dash.no_update
    preview_title = dash.no_update
    preview_content = dash.no_update
    open_delete = dash.no_update
    delete_body = dash.no_update
    pending_file = dash.no_update

    ctx = dash.callback_context
    if not ctx.triggered:
        return open_preview, preview_title, preview_content, open_delete, delete_body, pending_file

    trig = ctx.triggered[0]["prop_id"].split(".")[0]

    if trig == "docs-table" and active_cell:
        row = active_cell.get("row")
        col = active_cell.get("column_id")
        file_name = extract_file_from_row(table_data, row)
        if not file_name:
            return open_preview, preview_title, preview_content, open_delete, delete_body, pending_file
        if col == "view":
            # Load markdown, fallback to plaintext via MarkItDown
            t0 = time.time()
            try:
                content = storage.read_markdown(file_name)
                if content.startswith("Файл Markdown не найден") or content.startswith("Ошибка"):
                    # Fallback to regenerate preview from source
                    path = storage.CFG.data_path / file_name
                    text = ""
                    if path.exists():
                        try:
                            reader = MarkItDownReader()
                            docs = reader.load_data(path)
                            if docs:
                                text = docs[0].text or ""
                        except Exception:
                            text = ""
                    content = text or "(No preview available)"
                log.info("document_previewed", action="document_previewed",
                         filename=file_name, duration_ms=int((time.time()-t0)*1000))
            except Exception as e:
                log.error("document_preview_failed",
                          filename=file_name, error=str(e))
                content = f"Error loading preview: {e}"

            open_preview = True
            preview_title = file_name
            preview_content = content

        elif col == "delete":
            open_delete = True
            delete_body = html.Div(
                [html.P(f"Delete '{file_name}'? This will remove the file and its vectors.")])
            pending_file = file_name

    elif trig == "close-preview":
        open_preview = False
    elif trig == "cancel-delete":
        open_delete = False

    return open_preview, preview_title, preview_content, open_delete, delete_body, pending_file


@app.callback(
    Output("delete-modal", "is_open", allow_duplicate=True),
    Output("toast-success", "is_open"),
    Output("toast-success", "children"),
    Output("toast-error", "is_open"),
    Output("toast-error", "children"),
    Output("refresh-ts", "data", allow_duplicate=True),
    Input("confirm-delete", "n_clicks"),
    State("pending-delete-file", "data"),
    prevent_initial_call=True,
)
def confirm_delete(n_clicks, file_name):
    if not n_clicks or not file_name:
        return dash.no_update, False, dash.no_update, False, dash.no_update, dash.no_update
    t0 = time.time()
    try:
        res = storage.delete_document(file_name)
        log.info(
            "document_deleted",
            action="document_deleted",
            filename=file_name,
            vectors_deleted=res.get("vectors_deleted"),
            duration_ms=int((time.time()-t0)*1000),
        )
        return False, True, f"Deleted {file_name}", False, dash.no_update, time.time()
    except Exception as e:
        log.error("document_delete_failed", filename=file_name, error=str(e))
        return False, False, dash.no_update, True, f"Failed to delete {file_name}: {e}", dash.no_update


def _parse_upload(contents: str, filename: str) -> Tuple[Optional[io.BytesIO], str]:
    if not contents:
        return None, filename
    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        return io.BytesIO(decoded), filename
    except Exception:
        return None, filename


@app.callback(
    Output("upload-status", "children"),
    Output("toast-success", "is_open", allow_duplicate=True),
    Output("toast-success", "children", allow_duplicate=True),
    Output("toast-error", "is_open", allow_duplicate=True),
    Output("toast-error", "children", allow_duplicate=True),
    Output("refresh-ts", "data", allow_duplicate=True),
    Input("upload-btn", "n_clicks"),
    State("upload", "contents"),
    State("upload", "filename"),
    prevent_initial_call=True,
)
def handle_upload(n_clicks, contents_list, filenames):
    if not n_clicks:
        return dash.no_update, False, dash.no_update, False, dash.no_update, dash.no_update
    if not contents_list or not filenames:
        return html.Div("No files selected."), False, dash.no_update, False, dash.no_update, dash.no_update

    successes = 0
    errors: List[str] = []
    started_at = time.time()
    log.info("upload_started", action="upload_started", count=len(filenames))

    for contents, fname in zip(contents_list, filenames):
        bio, original_name = _parse_upload(contents, fname)
        if bio is None:
            errors.append(f"Failed to parse {original_name}")
            continue
        try:
            tmp = Path(storage.CFG.data_path) / \
                f".__tmp__{int(time.time()*1000)}_{Path(original_name).name}"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                f.write(bio.getvalue())
            # Persist + index
            storage.add_docx_to_store(
                tmp, persist_original=True, target_file_name=Path(original_name).name)
            # Remove temp file if copy succeeded
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            successes += 1
        except Exception as e:
            errors.append(f"{original_name}: {e}")

    duration_ms = int((time.time() - started_at) * 1000)
    if errors:
        log.error("upload_failed", action="upload_failed",
                  error=errors, duration_ms=duration_ms)
    else:
        log.info("upload_finished", action="upload_finished",
                 count=successes, duration_ms=duration_ms)

    status_children = [
        html.Div(f"Uploaded {successes} / {len(filenames)} files.")]
    if errors:
        status_children.append(html.Ul([html.Li(e) for e in errors]))

    return (
        status_children,
        True if successes else False,
        f"Uploaded {successes} file(s)",
        True if errors else False,
        "\n".join(errors) if errors else dash.no_update,
        time.time() if successes else dash.no_update,
    )


if __name__ == "__main__":
    warmup()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7862"))
    app.run(host=host, port=port, debug=False)
