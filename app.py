from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import dash
from dash import Dash, Input, Output, State, dcc, html
import dash_mantine_components as dmc

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
    # Enrich with display fields
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "file_name": r.get("file_name"),
                "uploaded_at": (lambda s: (datetime.fromisoformat(s.replace("Z", "+00:00")).strftime("%d.%m.%Y %H:%M:%S") if s else "—"))(r.get("uploaded_at_iso")),
                "size": bytes_to_human(r.get("size_bytes")),
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


def _load_markdown_or_fallback(file_name: str) -> str:
    """Load saved markdown; if missing or error text, regenerate plain text preview."""
    try:
        md = storage.read_markdown(file_name)
    except Exception as e:
        log.error("read_markdown_failed", filename=file_name, error=str(e))
        md = ""
    if md and not (md.startswith("Файл Markdown не найден") or md.startswith("Ошибка")):
        return md
    # Fallback
    path = storage.CFG.data_path / file_name
    if not path.exists():
        return "(No preview available)"
    try:
        reader = MarkItDownReader()
        docs = reader.load_data(path)
        return (docs[0].text or "") if docs else "(No preview available)"
    except Exception as e:
        log.error("fallback_preview_failed", filename=file_name, error=str(e))
        return "(No preview available)"


def _process_single_upload(contents: str, original_name: str) -> Optional[str]:
    """Persist a single uploaded file and index it. Returns error message if any."""
    bio, _ = _parse_upload(contents, original_name)
    if bio is None:
        return f"Failed to parse {original_name}"
    try:
        tmp = Path(storage.CFG.data_path) / f".__tmp__{int(time.time()*1000)}_{Path(original_name).name}"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            f.write(bio.getvalue())
        storage.add_docx_to_store(tmp, persist_original=True, target_file_name=Path(original_name).name)
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return None
    except Exception as e:
        return f"{original_name}: {e}"


# ----------------------------- App -----------------------------

app: Dash = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

def render_docs_table(rows: List[Dict[str, Any]]):
    header = dmc.TableThead(
        dmc.TableTr(
            [
                dmc.TableTh("Document"),
                dmc.TableTh("Uploaded At"),
                dmc.TableTh("Size"),
                dmc.TableTh(""),
                dmc.TableTh(""),
            ]
        )
    )
    body_rows = []
    for r in rows:
        file = r.get("file_name")
        body_rows.append(
            dmc.TableTr(
                [
                    dmc.TableTd(file),
                    dmc.TableTd(r.get("uploaded_at")),
                    dmc.TableTd(r.get("size")),
                    dmc.TableTd(
                        dmc.Button(
                            "View",
                            id={"type": "view-doc", "file": file},
                            variant="subtle",
                            size="xs",
                        )
                    ),
                    dmc.TableTd(
                        dmc.Button(
                            "Delete",
                            id={"type": "delete-doc", "file": file},
                            color="red",
                            variant="subtle",
                            size="xs",
                        )
                    ),
                ]
            )
        )
    table = dmc.Table(
        [header, dmc.TableTbody(body_rows)],
        striped=True,
        highlightOnHover=True,
        withRowBorders=False,
    )
    return table


app.layout = dmc.MantineProvider(
    dmc.Container(
        [
            dmc.NotificationContainer(id="notification-container"),
            dcc.Store(id="docs-cache"),  # raw docs list
            dcc.Store(id="refresh-ts", data=time.time()),  # timestamp to force refresh
            dcc.Store(id="pending-delete-file"),
            dcc.Store(id="preview-file"),

            dmc.Tabs(
                value="tab-docs",
                children=[
                    # Tabs list
                    dmc.TabsList(
                        [
                            dmc.TabsTab("Documents", value="tab-docs"),
                            dmc.TabsTab("Upload", value="tab-upload"),
                        ]
                    ),

                    # Documents tab
                    dmc.TabsPanel(
                        children=[
                            dmc.Grid(
                                children=[
                                    dmc.GridCol(
                                        dmc.TextInput(
                                            id="search-input",
                                            placeholder="Search by file name…",
                                        ),
                                        span={"base": 12, "md": 6},
                                    ),
                                    dmc.GridCol(
                                        dmc.Text(id="action-hint", c="dimmed"),
                                        span={"base": 12, "md": 6},
                                    ),
                                ],
                                gutter="sm",
                            ),
                            html.Div(id="docs-table", style={"overflowX": "auto", "marginTop": "8px"}),

                            # Preview Modal
                            dmc.Modal(
                                id="preview-modal",
                                opened=False,
                                title="",
                                size="xl",
                                centered=True,
                                closeOnEscape=False,
                                closeOnClickOutside=False,
                                children=[
                                    dcc.Markdown(id="preview-content", style={"whiteSpace": "pre-wrap"}),
                                    dmc.Group(
                                        [
                                            dmc.Button("Close", id="close-preview", variant="light", color="gray"),
                                        ],
                                        justify="flex-end",
                                        mt="md",
                                    ),
                                ],
                            ),

                            # Delete confirm Modal
                            dmc.Modal(
                                id="delete-modal",
                                opened=False,
                                title="Confirm deletion",
                                centered=True,
                                children=[
                                    dmc.Text(id="delete-confirm-body"),
                                    dmc.Group(
                                        [
                                            dmc.Button("Cancel", id="cancel-delete", variant="light", color="gray"),
                                            dmc.Button("Delete", id="confirm-delete", color="red"),
                                        ],
                                        justify="flex-end",
                                        mt="md",
                                    ),
                                ],
                            ),
                        ],
                        value="tab-docs",
                    ),

                    # Upload tab
                    dmc.TabsPanel(
                        children=[
                            dcc.Upload(
                                id="upload",
                                children=html.Div(["Drag and drop or ", html.A("select files")]),
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
                            dmc.Button("Upload & Index", id="upload-btn", mt="md"),
                            dcc.Loading(id="upload-loading", type="default", children=html.Div(id="upload-status", style={"marginTop": "12px"})),
                        ],
                        value="tab-upload",
                    ),
                ],
            ),
        ]
    )
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
    Output("docs-table", "children"),
    Input("docs-cache", "data"),
    Input("search-input", "value"),
)
def filter_and_render_table(docs: List[Dict[str, Any]] | None, search: Optional[str]):
    docs = docs or []
    # sort by uploaded_at desc (string ISO comparable)
    try:
        docs = sorted(docs, key=lambda d: d.get("uploaded_at_iso") or "", reverse=True)
    except Exception:
        pass
    if search:
        s = search.strip().lower()
        docs = [d for d in docs if s in (d.get("file_name", "").lower())]
    rows = build_table_rows(docs)
    return render_docs_table(rows)


@app.callback(
    Output("preview-modal", "opened"),
    Output("preview-modal", "title"),
    Output("preview-content", "children"),
    Input({"type": "view-doc", "file": dash.ALL}, "n_clicks"),
    Input("close-preview", "n_clicks"),
    prevent_initial_call=True,
)
def handle_preview(view_clicks, close_preview):
    opened = dash.no_update
    title = dash.no_update
    content = dash.no_update

    ctx = dash.callback_context
    if not ctx.triggered:
        return opened, title, content

    trig_id = ctx.triggered[0]["prop_id"].split(".")[0]
    try:
        trig = dash.callback_context.triggered_id
    except Exception:
        trig = None

    if isinstance(trig, dict) and trig.get("type") == "view-doc":
        # Only open on an actual click (avoid spurious triggers when table rerenders)
        try:
            if not view_clicks or not any(((c or 0) > 0) for c in view_clicks):
                return opened, title, content
        except Exception:
            # If shape is unexpected, be conservative and do not open
            return opened, title, content
        file_name = trig.get("file")
        t0 = time.time()
        try:
            md = storage.read_markdown(file_name)
            if md.startswith("Файл Markdown не найден") or md.startswith("Ошибка"):
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
                md = text or "(No preview available)"
            log.info("document_previewed", action="document_previewed", filename=file_name, duration_ms=int((time.time()-t0)*1000))
        except Exception as e:
            log.error("document_preview_failed", filename=file_name, error=str(e))
            md = f"Error loading preview: {e}"
        opened = True
        title = file_name
        content = md
    elif trig_id == "close-preview":
        opened = False

    return opened, title, content


@app.callback(
    Output("delete-modal", "opened"),
    Output("delete-confirm-body", "children"),
    Output("pending-delete-file", "data"),
    Input({"type": "delete-doc", "file": dash.ALL}, "n_clicks"),
    Input("cancel-delete", "n_clicks"),
    prevent_initial_call=True,
)
def handle_delete_open(delete_clicks, cancel_delete):
    opened = dash.no_update
    body = dash.no_update
    pending = dash.no_update

    try:
        trig = dash.callback_context.triggered_id
    except Exception:
        trig = None

    if isinstance(trig, dict) and trig.get("type") == "delete-doc":
        # Only open on an actual click (avoid spurious triggers when table rerenders)
        try:
            if not delete_clicks or not any(((c or 0) > 0) for c in delete_clicks):
                return opened, body, pending
        except Exception:
            return opened, body, pending
        file_name = trig.get("file")
        opened = True
        body = html.Div([html.P(f"Delete '{file_name}'? This will remove the file and its vectors.")])
        pending = file_name
    elif trig == "cancel-delete":
        opened = False

    return opened, body, pending


@app.callback(
    Output("delete-modal", "opened", allow_duplicate=True),
    Output("notification-container", "sendNotifications"),
    Output("refresh-ts", "data", allow_duplicate=True),
    Input("confirm-delete", "n_clicks"),
    State("pending-delete-file", "data"),
    prevent_initial_call=True,
)
def confirm_delete(n_clicks, file_name):
    if not n_clicks or not file_name:
        return dash.no_update, dash.no_update, dash.no_update
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
        notif = [{
            "id": f"del-{int(time.time()*1000)}",
            "action": "show",
            "title": "Success",
            "message": f"Deleted {file_name}",
            "color": "green",
            "autoClose": 3000,
            "withBorder": True,
        }]
        return False, notif, time.time()
    except Exception as e:
        log.error("document_delete_failed", filename=file_name, error=str(e))
        notif = [{
            "id": f"err-{int(time.time()*1000)}",
            "action": "show",
            "title": "Error",
            "message": f"Failed to delete {file_name}: {e}",
            "color": "red",
            "autoClose": 6000,
            "withBorder": True,
        }]
        return False, notif, dash.no_update


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
    Output("notification-container", "sendNotifications", allow_duplicate=True),
    Output("refresh-ts", "data", allow_duplicate=True),
    Input("upload-btn", "n_clicks"),
    State("upload", "contents"),
    State("upload", "filename"),
    prevent_initial_call=True,
)
def handle_upload(n_clicks, contents_list, filenames):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update
    if not contents_list or not filenames:
        notif = [{
            "id": f"warn-{int(time.time()*1000)}",
            "action": "show",
            "title": "No files",
            "message": "No files selected.",
            "color": "yellow",
            "autoClose": 3000,
        }]
        return html.Div("No files selected."), notif, dash.no_update

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
            tmp = Path(storage.CFG.data_path) / f".__tmp__{int(time.time()*1000)}_{Path(original_name).name}"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                f.write(bio.getvalue())
            # Persist + index
            storage.add_docx_to_store(tmp, persist_original=True, target_file_name=Path(original_name).name)
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
        log.error("upload_failed", action="upload_failed", error=errors, duration_ms=duration_ms)
    else:
        log.info("upload_finished", action="upload_finished", count=successes, duration_ms=duration_ms)

    status_children = [html.Div(f"Uploaded {successes} / {len(filenames)} files.")]
    if errors:
        status_children.append(html.Ul([html.Li(e) for e in errors]))

    notifications = []
    if successes:
        notifications.append({
            "id": f"up-{int(time.time()*1000)}",
            "action": "show",
            "title": "Upload complete",
            "message": f"Uploaded {successes} file(s)",
            "color": "green",
            "autoClose": 3000,
        })
    if errors:
        notifications.append({
            "id": f"up-err-{int(time.time()*1000)}",
            "action": "show",
            "title": "Upload errors",
            "message": "\n".join(errors),
            "color": "red",
            "autoClose": 6000,
        })

    return (
        status_children,
        notifications if notifications else dash.no_update,
        time.time() if successes else dash.no_update,
    )


if __name__ == "__main__":
    warmup()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7862"))
    app.run(host=host, port=port, debug=False)
