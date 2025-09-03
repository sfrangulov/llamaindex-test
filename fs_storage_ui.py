from __future__ import annotations

from typing import Any, Dict, List

import dash
from dash import dcc, Input, Output, State, callback_context, ALL
import dash_mantine_components as dmc

from storage import list_documents, read_markdown, delete_document


def _sort_rows_by_date_desc(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Sort by ISO timestamp (newest first); ISO strings sort lexicographically by time
    try:
        return sorted(rows or [], key=lambda r: (r or {}).get("uploaded_at_iso") or "", reverse=True)
    except Exception:
        return rows or []


def _human_size(n: int | float | None) -> str:
    try:
        n = float(n or 0)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if n < 1024:
                return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} PB"
    except Exception:
        return "—"


def _render_table(rows: List[Dict[str, Any]]):
    header = dmc.TableThead(dmc.TableTr([
        dmc.TableTh("#"),
        dmc.TableTh("Название", maw=640),
        dmc.TableTh("Размер"),
        dmc.TableTh("Изменен"),
        dmc.TableTh("Действия"),
    ]))
    body_rows = []
    for i, r in enumerate(rows or [], start=1):
        name = r.get("file_name", "")
        size = _human_size(r.get("size_bytes"))
        ts = (r.get("uploaded_at_iso") or "").replace("T", " ").replace("Z", "")
        name_btn = dmc.Tooltip(
            dmc.Button(
                name,
                id={"type": "stg-open-doc", "file": name},
                variant="subtle",
                color="blue",
                maw=620,
            ),
            label=name,
        )
        del_btn = dmc.Button(
            "Удалить",
            id={"type": "stg-delete-doc", "file": name},
            color="red",
            variant="light",
            size="xs",
        )
        body_rows.append(dmc.TableTr([
            dmc.TableTd(str(i)),
            dmc.TableTd(name_btn, maw=360),
            dmc.TableTd(size),
            dmc.TableTd(ts or "—"),
            dmc.TableTd(dmc.Group([del_btn], gap="xs")),
        ]))
    body = dmc.TableTbody(body_rows)
    return dmc.Table(highlightOnHover=True, striped=True, verticalSpacing="sm", horizontalSpacing="md", children=[header, body])


def get_layout() -> Any:
    return dmc.Stack([
        dmc.Paper(
            withBorder=True,
            p="md",
            radius="md",
            children=[
                dmc.Alert(
                    "Список файлов, загруженных в хранилище. Откройте предпросмотр Markdown или удалите ненужные документы. Используйте поиск по названию.",
                    title="Хранилище",
                    color="gray",
                ),
                dmc.Space(h=8),
                dmc.Group([
                    dmc.TextInput(id="stg-search", placeholder="Поиск по названию…", style={"flex": 1}),
                    dmc.Button("Обновить", id="stg-refresh", variant="light"),
                ], align="stretch"),
                dmc.Space(h=10),
                dmc.Box(id="stg-table-wrap", style={"overflowX": "auto"}),
                dmc.Space(h=6),
                dmc.Alert(id="stg-status", color="gray", children="", withCloseButton=False),
            ],
        ),
        # Stores and modals
        dcc.Store(id="stg-rows", data=[]),
        dcc.Store(id="stg-delete-target"),
        dmc.Modal(
            id="stg-preview-modal",
            title=dmc.Text(id="stg-preview-title", fw=600),
            children=[
                dmc.ScrollArea(
                    offsetScrollbars=True,
                    type="auto",
                    h=500,
                    children=[
                        dmc.TypographyStylesProvider(
                            dcc.Markdown(id="stg-preview-content", link_target="_blank", className="markdown")
                        )
                    ],
                )
            ],
            size="xl",
            centered=True,
            opened=False,
        ),
        dmc.Modal(
            id="stg-delete-modal",
            title="Удаление документа",
            children=[
                dmc.Stack([
                    dmc.Text(id="stg-delete-label"),
                    dmc.Group([
                        dmc.Button("Удалить", id="stg-confirm-delete", color="red"),
                        dmc.Button("Отмена", id="stg-cancel-delete", variant="light"),
                    ], gap="sm")
                ])
            ],
            size="md",
            centered=True,
            opened=False,
        ),
    ])


def register_callbacks(app: dash.Dash):
    # Auto-refresh when navigating to the Storage section
    @app.callback(
        Output("stg-rows", "data", allow_duplicate=True),
        Output("stg-table-wrap", "children", allow_duplicate=True),
        Output("stg-status", "children", allow_duplicate=True),
        Output("stg-status", "color", allow_duplicate=True),
        Input("top-nav", "value"),
        prevent_initial_call=True,
    )
    def _on_nav(val):
        if val != "storage":
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        rows = _sort_rows_by_date_desc(list_documents(None))
        return rows, _render_table(rows), f"Найдено: {len(rows)}", "gray"

    # Fetch and render table
    @app.callback(
        Output("stg-rows", "data"),
        Output("stg-table-wrap", "children"),
        Output("stg-status", "children"),
        Output("stg-status", "color"),
        Input("stg-search", "value"),  # realtime search on typing
        Input("stg-refresh", "n_clicks"),
        prevent_initial_call=True,
    )
    def refresh_table(search, _n_refresh):
        try:
            rows = list_documents((search or "").strip() or None)
            rows = _sort_rows_by_date_desc(rows)
            table = _render_table(rows)
            msg = f"Найдено: {len(rows)}"
            return rows, table, msg, "gray"
        except Exception as e:
            return dash.no_update, dash.no_update, f"Ошибка: {e}", "red"

    # Open preview
    @app.callback(
        Output("stg-preview-modal", "opened"),
        Output("stg-preview-title", "children"),
        Output("stg-preview-content", "children"),
        Input({"type": "stg-open-doc", "file": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def on_open(clicks):
        try:
            if not clicks or all(((c or 0) <= 0) for c in clicks):
                return False, dash.no_update, dash.no_update
            ctx = callback_context
            tid = getattr(ctx, "triggered_id", None)
            fname = None
            if isinstance(tid, dict):
                fname = tid.get("file")
            else:
                trig = getattr(ctx, "triggered", []) or []
                if trig:
                    comp = (trig[0].get("prop_id", "").split(".") or [""])[0]
                    if comp.startswith("{"):
                        import json as _json
                        try:
                            d = _json.loads(comp)
                            fname = d.get("file")
                        except Exception:
                            fname = None
            if not fname:
                return False, dash.no_update, dash.no_update
            md = read_markdown(fname)
            return True, fname, md
        except Exception:
            return False, dash.no_update, dash.no_update

    # Ask deletion confirmation
    @app.callback(
        Output("stg-delete-modal", "opened"),
        Output("stg-delete-label", "children"),
        Output("stg-delete-target", "data"),
        Input({"type": "stg-delete-doc", "file": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def ask_delete(clicks):
        try:
            if not clicks or all(((c or 0) <= 0) for c in clicks):
                return False, dash.no_update, dash.no_update
            ctx = callback_context
            tid = getattr(ctx, "triggered_id", None)
            fname = None
            if isinstance(tid, dict):
                fname = tid.get("file")
            if not fname:
                return False, dash.no_update, dash.no_update
            return True, f"Удалить документ: {fname}?", fname
        except Exception:
            return False, dash.no_update, dash.no_update

    # Delete confirmed / cancel
    @app.callback(
        Output("stg-delete-modal", "opened", allow_duplicate=True),
        Output("stg-status", "children", allow_duplicate=True),
        Output("stg-status", "color", allow_duplicate=True),
        Output("stg-rows", "data", allow_duplicate=True),
        Output("stg-table-wrap", "children", allow_duplicate=True),
        Input("stg-confirm-delete", "n_clicks"),
        Input("stg-cancel-delete", "n_clicks"),
        State("stg-delete-target", "data"),
        State("stg-search", "value"),
        prevent_initial_call=True,
    )
    def on_delete(confirm, cancel, target, search):
        try:
            ctx = callback_context
            trig = (getattr(ctx, "triggered", []) or [{}])[0].get("prop_id", "")
            if trig.startswith("stg-cancel-delete"):
                return False, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            if not confirm or not target:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            res = delete_document(str(target))
            rows = list_documents((search or "").strip() or None)
            rows = _sort_rows_by_date_desc(rows)
            table = _render_table(rows)
            ok = bool(res.get("file_deleted") or res.get("md_deleted") or (res.get("vectors_deleted", 0) > 0))
            msg = f"Удалено: {target}" if ok else f"Не удалось удалить: {target}"
            color = "green" if ok else "red"
            return False, msg, color, rows, table
        except Exception as e:
            return False, f"Ошибка удаления: {e}", "red", dash.no_update, dash.no_update
