from __future__ import annotations

from typing import Any, Dict, List

import dash
from dash import dcc, Input, Output, State, callback_context, ALL
import dash_mantine_components as dmc

from storage import list_documents, read_markdown, delete_document
from db import get_latest_fs_analysis


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
        dmc.TableTh("Оценки"),
        dmc.TableTh("Действия"),
    ]))
    body_rows = []
    for i, r in enumerate(rows or [], start=1):
        name = r.get("file_name", "")
        size = _human_size(r.get("size_bytes"))
        ts = (r.get("uploaded_at_iso") or "").replace("T", " ").replace("Z", "")
        # Ratings button (if available)
        if r.get("has_ratings"):
            ratings_btn = dmc.Tooltip(
                dmc.Button(
                    "Оценки",
                    id={"type": "stg-view-ratings", "file": name},
                    color="teal",
                    variant="light",
                    size="xs",
                ),
                label=r.get("ratings_tooltip") or "Просмотр оценок разделов",
            )
        else:
            ratings_btn = dmc.Text("—")
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
        analyze_btn = dmc.Button(
            "Анализировать",
            id={"type": "stg-analyze-doc", "file": name},
            color="blue",
            variant="light",
            size="xs",
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
            dmc.TableTd(name_btn, maw=640),
            dmc.TableTd(size),
            dmc.TableTd(ts or "—"),
            dmc.TableTd(ratings_btn),
            dmc.TableTd(dmc.Group([analyze_btn, del_btn], gap="xs")),
        ]))
    body = dmc.TableTbody(body_rows)
    return dmc.Table(highlightOnHover=True, striped=True, verticalSpacing="sm", horizontalSpacing="md", children=[header, body])


def _enrich_with_ratings(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach latest analysis summary counts per file to rows in place."""
    for r in rows or []:
        try:
            name = r.get("file_name")
            if not name:
                continue
            latest = get_latest_fs_analysis(str(name))
            if not latest:
                r["has_ratings"] = False
                continue
            sections = (latest or {}).get("analysis") or {}
            full = part = no = 0
            for _title, payload in (sections or {}).items():
                ov = ((payload or {}).get("overall_assessment") or "").strip().upper()
                if not ov:
                    continue
                if ov.startswith("ПОЛНОСТЬЮ"):
                    full += 1
                elif ov.startswith("ЧАСТИЧНО"):
                    part += 1
                else:
                    no += 1
            total = int((latest or {}).get("sections_count") or (full + part + no))
            r["has_ratings"] = True
            r["ratings_summary"] = {"full": full, "part": part, "no": no, "total": total}
            r["ratings_tooltip"] = f"П:{full} • Ч:{part} • Н:{no} из {total}"
        except Exception:
            r["has_ratings"] = False
    return rows


def get_layout() -> Any:
    return dmc.Stack([
        dmc.Paper(
            withBorder=True,
            p="md",
            radius="md",
            children=[
                dmc.Alert(
                    "Список файлов, загруженных в хранилище. Откройте предпросмотр Markdown или удалите ненужные документы. Используйте поиск по названию.",
                    title="Как работает хранилище",
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
        # Ratings modals
        dmc.Modal(
            id="stg-ratings-modal",
            title=dmc.Text(id="stg-ratings-title", fw=600),
            children=[
                dmc.ScrollArea(
                    offsetScrollbars=True,
                    type="auto",
                    h=500,
                    children=[
                        dmc.TypographyStylesProvider(
                            dmc.Box(id="stg-ratings-content")
                        )
                    ],
                )
            ],
            size="xl",
            centered=True,
            opened=False,
        ),
        dmc.Modal(
            id="stg-section-modal",
            title=dmc.Text(id="stg-section-title", fw=600),
            children=[
                dmc.ScrollArea(
                    offsetScrollbars=True,
                    type="auto",
                    h=500,
                    children=[
                        dmc.TypographyStylesProvider(
                            dcc.Markdown(id="stg-section-content", link_target="_blank", className="markdown")
                        )
                    ],
                )
            ],
            size="xl",
            centered=True,
            opened=False,
        ),
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
        rows = _sort_rows_by_date_desc(_enrich_with_ratings(list_documents(None)))
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
            rows = _enrich_with_ratings(rows)
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
            rows = _enrich_with_ratings(rows)
            rows = _sort_rows_by_date_desc(rows)
            table = _render_table(rows)
            ok = bool(res.get("file_deleted") or res.get("md_deleted") or (res.get("vectors_deleted", 0) > 0))
            msg = f"Удалено: {target}" if ok else f"Не удалось удалить: {target}"
            color = "green" if ok else "red"
            return False, msg, color, rows, table
        except Exception as e:
            return False, f"Ошибка удаления: {e}", "red", dash.no_update, dash.no_update

    # Jump to Analysis section and load selected file
    @app.callback(
        Output("top-nav", "value"),
        Output("current-file-name", "data", allow_duplicate=True),
        Output("main-tabs", "value"),
        Input({"type": "stg-analyze-doc", "file": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def on_analyze_from_storage(clicks):
        try:
            if not clicks or all(((c or 0) <= 0) for c in clicks):
                return dash.no_update, dash.no_update, dash.no_update
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
                return dash.no_update, dash.no_update, dash.no_update
            # Switch top-level section to Analysis and inner tab to Analysis panel
            return "analysis", fname, "analysis"
        except Exception:
            return dash.no_update, dash.no_update, dash.no_update

    # Open ratings modal
    @app.callback(
        Output("stg-ratings-modal", "opened"),
        Output("stg-ratings-title", "children"),
        Output("stg-ratings-content", "children"),
        Input({"type": "stg-view-ratings", "file": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def on_open_ratings(clicks):
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
            latest = get_latest_fs_analysis(str(fname))
            if not latest:
                return False, dash.no_update, dash.no_update
            sections = (latest or {}).get("analysis") or {}
            # Build table of sections and ratings
            header = dmc.TableThead(dmc.TableTr([
                dmc.TableTh("#"),
                dmc.TableTh("Раздел"),
                dmc.TableTh("Оценка"),
                dmc.TableTh("Детали"),
            ]))
            body_rows = []
            for i, (title, payload) in enumerate(sections.items(), start=1):
                overall = ((payload or {}).get("overall_assessment") or "—")
                if overall:
                    ot = overall.strip().upper()
                    if ot.startswith("ПОЛНОСТЬЮ"):
                        color = "green"
                    elif ot.startswith("ЧАСТИЧНО"):
                        color = "orange"
                    else:
                        color = "red"
                    overall_cell = dmc.Badge(overall, color=color, variant="light")
                else:
                    overall_cell = dmc.Text("—")
                details_btn = dmc.Button(
                    "Открыть",
                    id={"type": "stg-view-section-details", "file": fname, "section": title},
                    size="xs",
                    variant="light",
                )
                body_rows.append(dmc.TableTr([
                    dmc.TableTd(str(i)),
                    dmc.TableTd(title),
                    dmc.TableTd(overall_cell),
                    dmc.TableTd(details_btn),
                ]))
            table = dmc.Table(highlightOnHover=True, striped=True, verticalSpacing="sm", horizontalSpacing="md", children=[header, dmc.TableTbody(body_rows)])
            return True, f"Оценки разделов — {fname}", table
        except Exception:
            return False, dash.no_update, dash.no_update

    # Open section details modal from ratings
    @app.callback(
        Output("stg-section-modal", "opened"),
        Output("stg-section-title", "children"),
        Output("stg-section-content", "children"),
        Input({"type": "stg-view-section-details", "file": ALL, "section": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def on_open_section_details(clicks):
        try:
            # Guard: only open on actual click
            if not clicks or all(((c or 0) <= 0) for c in clicks):
                return False, dash.no_update, dash.no_update
            ctx = callback_context
            tid = getattr(ctx, "triggered_id", None)
            if not isinstance(tid, dict):
                return False, dash.no_update, dash.no_update
            fname = tid.get("file")
            section = tid.get("section")
            if not fname or not section:
                return False, dash.no_update, dash.no_update
            latest = get_latest_fs_analysis(str(fname))
            if not latest:
                return False, dash.no_update, dash.no_update
            sect = ((latest or {}).get("analysis") or {}).get(section) or {}
            details = (sect.get("details_markdown") or sect.get("summary") or "(нет деталей)")
            return True, f"{section} — {fname}", details
        except Exception:
            return False, dash.no_update, dash.no_update
