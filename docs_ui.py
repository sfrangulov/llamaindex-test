import os
import io
import base64
from tempfile import NamedTemporaryFile
from typing import List, Dict

import dash
from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from storage import (
    list_storage_files,
    add_docx_to_store,
    list_vector_file_names,
    delete_from_vector_store_by_file_names,
    read_markdown,
    ensure_dirs,
    CFG,
)
from rag_engine import warmup


DOCX_EXT = ".docx"
COL_TITLE = "Название"
COL_SIZE = "Размер"
COL_DATE = "Дата"


def _format_size(n: int | None) -> str:
    if not n:
        return "-"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _load_rows(search: str, sort_by: str, sort_dir: str) -> List[Dict]:
    rows = list_storage_files(search)
    key_map = {
        "name": lambda r: r["file_name"].lower(),
        "size": lambda r: r.get("size_bytes") or 0,
        "date": lambda r: r.get("uploaded_at_iso") or "",
    }
    key = key_map.get(sort_by, key_map["name"])  # default by name
    rev = sort_dir == "desc"
    rows.sort(key=key, reverse=rev)
    data: List[Dict] = []
    for r in rows:
        data.append(
            {
                # keep original name for actions (hidden column in table)
                "file_name": r["file_name"],
                COL_TITLE: r["file_name"],
                COL_SIZE: _format_size(r.get("size_bytes")),
                COL_DATE: r.get("uploaded_at_iso") or "",
            }
        )
    return data


def _vector_count() -> int:
    try:
        return len(list_vector_file_names())
    except Exception:
        return 0


ensure_dirs()

external_stylesheets = [dbc.themes.COSMO]
app = Dash(__name__, external_stylesheets=external_stylesheets,
           title="Документы")
server = app.server  # for gunicorn/heroku if needed


# Initial table data
DEFAULT_SORT_BY = "name"
DEFAULT_SORT_DIR = "asc"
initial_data = _load_rows("", DEFAULT_SORT_BY, DEFAULT_SORT_DIR)


def _summary_badges():
    return dbc.Stack(
        direction="horizontal",
        gap=3,
        children=[
            dbc.Badge(f"Исходники: {CFG.data_path}",
                      color="primary", pill=True),
            dbc.Badge(f"Markdown: {CFG.md_dir}", color="info", pill=True),
            dbc.Badge(
                f"Chroma: {CFG.chroma_path} / {CFG.collection}",
                color="secondary",
                pill=True,
            ),
            dbc.Badge(f"Векторов по файлам: {_vector_count()}", color="success", pill=True,
                      id="badge-vector-count"),
        ],
    )


app.layout = dbc.Container(
    fluid=True,
    children=[
        html.Br(),
        dbc.Row(
            [
                dbc.Col(html.H2("Управление документами"), width="auto"),
            ],
            align="center",
        ),
        html.Br(),
        _summary_badges(),
        html.Hr(),
        dbc.Tabs(
            [
                dbc.Tab(label="Список файлов", tab_id="tab-list", children=[
                    html.Br(),
                    dbc.Row([
                        dbc.Col(dbc.Input(id="search",
                                placeholder="поиск по имени…"), md=5),
                        dbc.Col(dbc.Select(id="sort_by", options=[
                            {"label": "по имени", "value": "name"},
                            {"label": "по размеру", "value": "size"},
                            {"label": "по дате", "value": "date"},
                        ], value=DEFAULT_SORT_BY), md=3),
                        dbc.Col(dbc.Select(id="sort_dir", options=[
                            {"label": "возр.", "value": "asc"},
                            {"label": "убыв.", "value": "desc"},
                        ], value=DEFAULT_SORT_DIR), md=2),
                        dbc.Col(dbc.Button("Обновить", id="refresh",
                                color="secondary", outline=True), md=2),
                    ], className="gy-2"),
                    html.Br(),
                    dash_table.DataTable(
                        id="files-table",
                        columns=[
                            {"name": "file_name", "id": "file_name"},  # hidden technical column
                            {"name": COL_TITLE, "id": COL_TITLE},
                            {"name": COL_SIZE, "id": COL_SIZE},
                            {"name": COL_DATE, "id": COL_DATE},
                        ],
                        data=initial_data,
                        page_size=15,
                        sort_action="none",
                        row_selectable="single",
                        selected_rows=[],
                        hidden_columns=["file_name"],
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "textAlign": "left", "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"},
                        style_header={"fontWeight": "600"},
                    ),
                    html.Br(),
                    dbc.Row([
                        dbc.Col(dbc.Button("Предпросмотр",
                                id="btn-preview", color="primary"), md="auto"),
                        dbc.Col(dbc.Button("Удалить", id="btn-delete",
                                color="danger", outline=True), md="auto"),
                        dbc.Col(html.Div(id="status-area")),
                    ], className="gy-2"),
                ]),
                dbc.Tab(label="Загрузка и индексация", tab_id="tab-upload", children=[
                    html.Br(),
                    html.P(
                        "Загрузите один или несколько .docx — они будут конвертированы в Markdown и добавлены в векторное хранилище."),
                    dcc.Upload(
                        id="uploader",
                        multiple=True,
                        children=html.Div([
                            "Перетащите файлы или ", html.A(
                                "выберите на диске")
                        ]),
                        style={
                            "width": "100%",
                            "height": "120px",
                            "lineHeight": "120px",
                            "borderWidth": "1px",
                            "borderStyle": "dashed",
                            "borderRadius": "8px",
                            "textAlign": "center",
                            "backgroundColor": "#fafafa",
                        },
                        accept=".docx",
                    ),
                    html.Br(),
                    dbc.Button("Индексировать", id="btn-index",
                               color="success"),
                    html.Br(), html.Br(),
                    dbc.Alert(id="upload-status",
                              color="light", is_open=False),
                    dcc.Store(id="upload-cache"),
                ]),
            ],
            id="tabs",
            active_tab="tab-list",
        ),

        # Preview modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Предпросмотр Markdown")),
            dbc.ModalBody(dcc.Markdown(id="preview-md",
                          style={"whiteSpace": "pre-wrap"})),
        ], id="modal-preview", size="xl", fullscreen=False, is_open=False, scrollable=True),

        # Confirm delete modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Подтверждение удаления")),
            dbc.ModalBody(id="confirm-text"),
            dbc.ModalFooter([
                dbc.Button("Да", id="confirm-yes",
                           color="danger", className="me-2"),
                dbc.Button("Нет", id="confirm-no", outline=True),
            ]),
        ], id="modal-confirm", is_open=False),

        dcc.Store(id="files-data", data=initial_data),
        dcc.Store(id="pending-delete"),
    ],
)


# Refresh table when inputs change
@app.callback(
    Output("files-table", "data"),
    Output("files-table", "selected_rows"),
    Output("files-data", "data"),
    Output("badge-vector-count", "children"),
    Input("search", "value"),
    Input("sort_by", "value"),
    Input("sort_dir", "value"),
    Input("refresh", "n_clicks"),
    prevent_initial_call=False,
)
def refresh_table(search, sort_by, sort_dir, _):
    s = (search or "").strip()
    sb = sort_by or DEFAULT_SORT_BY
    sd = sort_dir or DEFAULT_SORT_DIR
    data = _load_rows(s, sb, sd)
    badge = f"Векторов по файлам: {_vector_count()}"
    return data, [], data, badge


def _selected_file_name(table_data: List[Dict], selected_rows: List[int]) -> str | None:
    if not table_data or not selected_rows:
        return None
    idx = selected_rows[0]
    if 0 <= idx < len(table_data):
        return table_data[idx].get("file_name")
    return None


# Preview flow
@app.callback(
    Output("preview-md", "children"),
    Output("modal-preview", "is_open"),
    Input("btn-preview", "n_clicks"),
    State("files-table", "data"),
    State("files-table", "selected_rows"),
    prevent_initial_call=True,
)
def on_preview(n_clicks, table_data, selected_rows):
    if not n_clicks:
        raise PreventUpdate
    name = _selected_file_name(table_data, selected_rows)
    if not name:
        return "Выберите файл", True
    return read_markdown(name), True


# Delete ask
@app.callback(
    Output("confirm-text", "children"),
    Output("modal-confirm", "is_open"),
    Output("pending-delete", "data"),
    Input("btn-delete", "n_clicks"),
    State("files-table", "data"),
    State("files-table", "selected_rows"),
    prevent_initial_call=True,
)
def on_delete_click(n_clicks, table_data, selected_rows):
    if not n_clicks:
        raise PreventUpdate
    name = _selected_file_name(table_data, selected_rows)
    if not name:
        return "Выберите файл для удаления", True, None
    return f"Удалить из векторного хранилища: {name}?", True, name


# Confirm delete
@app.callback(
    Output("status-area", "children"),
    Output("modal-confirm", "is_open"),
    Output("pending-delete", "data"),
    Output("refresh", "n_clicks"),  # trigger refresh
    Input("confirm-yes", "n_clicks"),
    Input("confirm-no", "n_clicks"),
    State("pending-delete", "data"),
    prevent_initial_call=True,
)
def on_confirm_delete(yes, no, pending):
    trigger = ctx.triggered_id
    if trigger == "confirm-no":
        return "", False, None, dash.no_update
    if trigger == "confirm-yes":
        if not pending:
            return dbc.Alert("Имя файла не задано", color="warning", dismissible=True), False, None, dash.no_update
        count = delete_from_vector_store_by_file_names([pending])
        msg = dbc.Alert(
            f"Удалено из векторного хранилища: {pending} (записей: {count})",
            color="success",
            dismissible=True,
        )
        # bump refresh clicks to trigger table reload
        return msg, False, None, (yes or 0) + 1
    raise PreventUpdate


# Upload cache: store raw upload payload
@app.callback(
    Output("upload-cache", "data"),
    Input("uploader", "contents"),
    State("uploader", "filename"),
    prevent_initial_call=True,
)
def on_upload(contents, filenames):
    if not contents or not filenames:
        raise PreventUpdate
    # Only keep DOCX files
    items = []
    for content, name in zip(contents, filenames):
        if not str(name).lower().endswith(DOCX_EXT):
            continue
        items.append({"name": name, "content": content})
    return items


# Index uploaded files on click
@app.callback(
    Output("upload-status", "children"),
    Output("upload-status", "is_open"),
    Output("upload-status", "color"),
    Output("refresh", "n_clicks"),  # trigger table refresh
    Input("btn-index", "n_clicks"),
    State("upload-cache", "data"),
    prevent_initial_call=True,
)
def on_index(n_clicks, items):
    if not n_clicks:
        raise PreventUpdate
    if not items:
        return "Файлы не выбраны", True, "warning", dash.no_update
    ensure_dirs()
    added: List[str] = []
    errors: List[str] = []

    def _process_items(items_list: List[Dict]) -> None:
        for it in items_list:
            try:
                name = it.get("name")
                content = it.get("content") or ""
                if not str(name).lower().endswith(DOCX_EXT) or not content.startswith("data:"):
                    continue
                _, b64data = content.split(",", 1)
                binary = base64.b64decode(b64data)
                with NamedTemporaryFile(suffix=DOCX_EXT, delete=True) as tf:
                    tf.write(binary)
                    tf.flush()
                    info = add_docx_to_store(tf.name)
                    added.append(info.get("file_name") or name)
            except Exception as e:
                errors.append(f"Ошибка: {it.get('name')} — {e}")

    _process_items(items)

    if added and not errors:
        msg = "Загружено:\n" + "\n".join(added)
        return msg, True, "success", (n_clicks or 0) + 1
    if added and errors:
        msg = "\n".join(["Частично загружено:"] + added + [""] + errors)
        return msg, True, "warning", (n_clicks or 0) + 1
    # type: ignore[name-defined]
    return ("\n".join(["Не удалось обработать файлы:"] + errors) if errors else "Нет поддерживаемых файлов"), True, "danger", dash.no_update


if __name__ == "__main__":
    warmup()
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=int(
        os.getenv("PORT", "7861")), debug=False)
