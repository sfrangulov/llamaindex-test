from rag_engine import warmup, search_documents
from fs_analyze_agent import get_fs, get_section_titles, analyze_fs_sections
from storage import CFG, ensure_dirs, add_docx_to_store, read_markdown
from md_reader import MarkItDownReader
from dotenv import load_dotenv
import os
import base64
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dash
from dash import Dash, html, dcc, Input, Output, State, callback_context, ALL
import dash_mantine_components as dmc
import plotly.io as pio

import structlog
log = structlog.get_logger(__name__)

load_dotenv()


# --------- App setup ---------
external_stylesheets: List[str] = []
app: Dash = dash.Dash(__name__, title="AI-анализ ФС",
                      external_stylesheets=external_stylesheets)
server = app.server


def _parse_upload(contents: str, filename: str) -> Tuple[bool, str]:
    """Persist uploaded DOCX to attachments and index; return (ok, message)."""
    try:
        if not contents or not filename:
            return False, "Нет файла для загрузки"
        _, content_string = contents.split(',')  # content_type unused
        decoded = base64.b64decode(content_string)
        # Ensure paths
        ensure_dirs()
        # Save temp file, then ingest using existing helper to keep md + vectors consistent
        tmp_path = Path("./attachments") / Path(filename).name
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, 'wb') as f:
            f.write(decoded)
        # Use storage helper to convert and store
        summary = add_docx_to_store(tmp_path, persist_original=True)
        return True, f"Загружен: {summary.get('file_name')}"
    except Exception as e:
        log.exception("upload_failed")
        return False, f"Ошибка загрузки: {e}"


COL_SECTION = "Раздел"
COL_STATUS = "Статус"
COL_ACTIONS = "Действия"
COL_ANALYSIS = "Анализ"
COL_OVERALL = "Оценка"


def _build_sections_table(file_name: str, analysis: Dict[str, Any] | None = None) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Return rows for DataTable and a dict of section->content (for modal).

    analysis: optional mapping section-> {summary, ok, issues_count, details_markdown}
    """
    try:
        md = read_markdown(file_name)
        # read_markdown returns error message as text if not found; we still pass to split
        sections_from_md = get_fs(str(
            CFG.md_dir / f"{file_name}.md")) if md and not md.startswith("Файл Markdown не найден") else {}
    except Exception:
        sections_from_md = {}

    titles = get_section_titles()
    rows: List[Dict[str, Any]] = []
    section_payload: Dict[str, str] = {}
    for i, title in enumerate(titles, start=1):
        content = sections_from_md.get(title)
        found = bool(content)
        section_payload[title] = content or "Раздел не найден"
        a_summary = None
        a_overall = None
        if analysis and isinstance(analysis, dict):
            a_data = analysis.get(title) or {}
            if isinstance(a_data, dict):
                a_summary = a_data.get("summary")
                a_overall = a_data.get("overall_assessment")
        rows.append({
            "#": i,
            COL_SECTION: title,
            COL_STATUS: "Найден" if found else "Не найден",
            COL_ANALYSIS: a_summary or ("—"),
            COL_OVERALL: a_overall or ("—"),
            COL_ACTIONS: "Просмотр" if found else "—",
        })
    return rows, section_payload


# --------- Layout ---------
app.layout = dmc.MantineProvider(
    children=dmc.Container(
        size="xl",
        px="md",
        children=[
            dmc.Title("AI-анализ ФС", order=2),
            dmc.Space(h=10),

            dmc.Paper(
                withBorder=True,
                p="md",
                radius="md",
                children=[
                    dmc.Group([
                        dmc.Text("Загрузка ФС (.docx)", fw=500),
                        dcc.Upload(
                            id="upload-docx",
                            children=dmc.Button("Выбрать файл"),
                            multiple=False,
                        ),
                    ], justify="space-between"),
                    dmc.Space(h=6),
                    dmc.Alert(id="upload-status", title="Статус", color="gray",
                              children="Ожидание загрузки", withCloseButton=False),
                ],
            ),

            dmc.Space(h=20),

            dmc.Badge(
                id="current-file",
                color="blue",
                variant="light",
                style={"display": "none"}),

            dmc.Space(h=20),

            dmc.Tabs(
                id="main-tabs",
                value="preview",
                style={"display": "none"},
                children=[
                    dmc.TabsList(children=[
                        dmc.TabsTab("Предпросмотр ФС", value="preview"),
                        dmc.TabsTab("Анализа ФС", value="analysis"),
                        dmc.TabsTab("Вопрос/ответ", value="qa"),
                    ]),

                    # Q&A Tab
                    dmc.TabsPanel(
                        value="qa",
                        children=[
                            dmc.Paper(
                                withBorder=True,
                                p="md",
                                radius="md",
                                children=[
                                    dmc.Stack([
                                        dmc.Textarea(
                                            id="chat-question", placeholder="Задайте вопрос по ФС…", autosize=True, minRows=2),
                                        dmc.Group([
                                            dmc.Button(
                                                "Спросить", id="chat-ask", variant="filled", color="blue"),
                                            dmc.Checkbox(
                                                id="chat-scope-file", label="Искать только в этой ФС", checked=True),
                                        ], gap="sm"),
                                        dcc.Loading(
                                            type="default",
                                            children=dmc.Alert(
                                                id="chat-answer",
                                                title="Ответ",
                                                color="gray",
                                                children="",
                                                withCloseButton=False,
                                            ),
                                        ),
                                    ])
                                ],
                            ),
                        ],
                    ),

                    # Analysis Tab
                    dmc.TabsPanel(
                        value="analysis",
                        children=[
                            dmc.Paper(
                                withBorder=True,
                                p="md",
                                radius="md",
                                children=[
                                    dmc.Group([
                                        dmc.Group([
                                            dmc.Button("Анализировать ФС", id="analyze-fs",
                                                       variant="filled", color="blue"),
                                            dmc.Button(
                                                "Выгрузить в Excel",
                                                id="export-excel",
                                                variant="outline",
                                                color="teal",
                                                disabled=True,
                                            ),
                                        ], gap="sm"),
                                    ], justify="space-between"),
                                    dmc.Space(h=10),
                                    dcc.Loading(
                                        id="analysis-loading",
                                        type="default",
                                        children=dmc.Box(
                                            id="sections-table-wrap",
                                            style={"overflowX": "auto"},
                                            children=[],
                                        ),
                                    ),
                                    dcc.Download(id="download-excel"),
                                    dcc.Store(id="sections-payload"),
                                    dcc.Store(id="analysis-result"),
                                ],
                            ),

                            # Modal for section preview (оставляем для разделов)
                            dmc.Modal(
                                id="section-modal",
                                title=dmc.Text(id="modal-title", fw=600),
                                children=[
                                    dmc.ScrollArea(
                                        offsetScrollbars=True,
                                        type="auto",
                                        h=500,
                                        children=[
                                            dcc.Markdown(
                                                id="modal-content", link_target="_blank", className="fs-md")
                                        ],
                                    )
                                ],
                                size="xl",
                                centered=True,
                                opened=False,
                            ),
                        ],
                    ),

                    # Inline Full Preview Tab
                    dmc.TabsPanel(
                        value="preview",
                        children=[
                            dmc.Paper(
                                withBorder=True,
                                p="md",
                                radius="md",
                                children=[
                                    dmc.Box(
                                        children=[
                                            dcc.Markdown(
                                                id="full-md-content", link_target="_blank", className="fs-md"),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            dcc.Store(id="current-file-name"),
            dcc.Store(id="analyzing-flag", data=False),
            dcc.Store(id="chatting-flag", data=False),
        ],
    )
)


# --------- Callbacks ---------
def _render_table(rows: List[Dict[str, Any]]):
    header = dmc.TableThead(dmc.TableTr([
        dmc.TableTh("#"),
        dmc.TableTh(COL_SECTION),
        dmc.TableTh(COL_STATUS),
        dmc.TableTh(COL_OVERALL),
        dmc.TableTh(COL_ANALYSIS),
        dmc.TableTh(COL_ACTIONS),
    ]))
    body_rows = []
    for r in rows:
        ok = (r.get(COL_STATUS) == "Найден")
        status = dmc.Badge("Найден", color="green", variant="light") if ok else dmc.Badge(
            "Не найден", color="red", variant="light")
        # Overall assessment badge
        overall_text = (r.get(COL_OVERALL) or "").strip().lower()
        if not overall_text or overall_text == "—":
            overall_cell = dmc.Text("—")
        else:
            if overall_text.startswith("полностью"):
                color = "green"
            elif overall_text.startswith("частично"):
                color = "orange"
            else:
                color = "red"
            overall_cell = dmc.Badge(
                r.get(COL_OVERALL), color=color, variant="light")
        analysis_text = (r.get(COL_ANALYSIS) or "").strip()
        if not analysis_text or analysis_text == "—":
            analysis_cell = dmc.Text("—")
        else:
            green = analysis_text.lower().startswith(
                "ok") or analysis_text.lower().startswith("ок")
            # Truncate badge label, show full text in tooltip for readability
            trimmed = analysis_text
            max_len = 40
            if len(trimmed) > max_len:
                trimmed = trimmed[: max_len - 1].rstrip() + "…"
            analysis_cell = dmc.Tooltip(
                label=dmc.Text(analysis_text, style={
                               "whiteSpace": "pre-wrap"}),
                multiline=True,
                withArrow=True,
                position="top-start",
                children=dmc.Badge(trimmed, color=(
                    "green" if green else "orange"), variant="light"),
            )
        action = (
            dmc.Button(
                "Просмотр",
                variant="light",
                size="xs",
                id={"type": "view-btn", "section": r.get(COL_SECTION)},
            ) if ok else dmc.Text("—")
        )
        body_rows.append(dmc.TableTr([
            dmc.TableTd(str(r.get("#", ""))),
            dmc.TableTd(r.get(COL_SECTION, "")),
            dmc.TableTd(status),
            dmc.TableTd(overall_cell),
            dmc.TableTd(analysis_cell),
            dmc.TableTd(action),
        ]))
    body = dmc.TableTbody(body_rows)
    return dmc.Table(highlightOnHover=True, striped=True, verticalSpacing="sm", horizontalSpacing="md", children=[header, body])


@app.callback(
    Output("upload-status", "children"),
    Output("upload-status", "color"),
    Output("current-file", "children"),
    Output("current-file-name", "data"),
    Output("sections-table-wrap", "children"),
    Output("sections-payload", "data"),
    Output("analyzing-flag", "data", allow_duplicate=True),
    Input("upload-docx", "contents"),
    State("upload-docx", "filename"),
    prevent_initial_call=True,
)
def on_upload(contents, filename):
    ok, msg = _parse_upload(contents, filename)
    if not ok:
        return msg, "red", "", None, [], {}, dash.no_update
    file_name = Path(filename).name
    rows, payload = _build_sections_table(file_name)
    table = _render_table(rows)
    return msg, "green", file_name, file_name, table, payload, False


@app.callback(
    Output("current-file", "style"),
    Output("main-tabs", "style"),
    Input("current-file-name", "data"),
)
def toggle_visibility(current_file_name):
    visible = bool(current_file_name)
    style_visible = {"display": "block"}  # Mantine Badge uses flex
    style_hidden = {"display": "none"}
    return (style_visible if visible else style_hidden), (style_visible if visible else style_hidden)


@app.callback(
    Output("section-modal", "opened"),
    Output("modal-title", "children"),
    Output("modal-content", "children"),
    Input({"type": "view-btn", "section": ALL}, "n_clicks"),
    State("sections-payload", "data"),
    prevent_initial_call=True,
)
def on_view(clicks, payload):
    try:
        # If no actual clicks yet (first render or after upload), do nothing
        if not clicks or all(((c or 0) <= 0) for c in clicks):
            return False, dash.no_update, dash.no_update
        ctx = callback_context
        tid = getattr(ctx, "triggered_id", None)
        if not tid or isinstance(tid, str):
            return False, dash.no_update, dash.no_update
        section_title = tid.get("section")
        if not section_title:
            return False, dash.no_update, dash.no_update
        content = (payload or {}).get(section_title) or "Раздел не найден"
        return True, section_title, content
    except Exception:
        return False, dash.no_update, dash.no_update


@app.callback(
    Output("full-md-content", "children"),
    Input("main-tabs", "value"),
    Input("current-file-name", "data"),
)
def update_full_preview(active_tab, file_name):
    try:
        # Render preview only when preview tab is active and a file is selected
        if active_tab != "preview" or not file_name:
            return dash.no_update
        md = read_markdown(file_name)
        if isinstance(md, str) and (md.startswith("Файл Markdown не найден") or md.startswith("Ошибка")):
            # Fallback to plain text preview from source DOCX (no persistence)
            try:
                path = CFG.data_path / file_name
                if path.exists():
                    reader = MarkItDownReader()
                    docs = reader.load_data(path)
                    if docs:
                        md = docs[0].text or "(Предпросмотр недоступен)"
                    else:
                        md = "(Предпросмотр недоступен)"
                else:
                    md = "(Markdown не найден, исходный файл отсутствует)"
            except Exception:
                md = "(Не удалось сформировать предпросмотр)"
        return md
    except Exception:
        return dash.no_update


@app.callback(
    Output("sections-table-wrap", "children", allow_duplicate=True),
    Output("sections-payload", "data", allow_duplicate=True),
    Output("analysis-result", "data", allow_duplicate=True),
    Output("upload-status", "children", allow_duplicate=True),
    Output("upload-status", "color", allow_duplicate=True),
    Input("analyze-fs", "n_clicks"),
    State("current-file-name", "data"),
    prevent_initial_call=True,
)
def on_analyze_fs(n_clicks, file_name):
    try:
        if not n_clicks or not file_name:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        # Build sections and run analysis
        try:
            md = read_markdown(file_name)
            sections_from_md = get_fs(str(
                CFG.md_dir / f"{file_name}.md")) if md and not md.startswith("Файл Markdown не найден") else {}
        except Exception:
            sections_from_md = {}

        analysis = analyze_fs_sections(sections_from_md)
        rows, payload = _build_sections_table(file_name, analysis=analysis)
        table = _render_table(rows)
        return table, payload, analysis, "Анализ завершен", "blue"
    except Exception as e:
        log.exception("analyze_failed")
        return dash.no_update, dash.no_update, dash.no_update, f"Ошибка анализа: {e}", "red"


# Separate fast callback to toggle analyzing flag when user clicks analyze
@app.callback(
    Output("analyzing-flag", "data"),
    Input("analyze-fs", "n_clicks"),
    prevent_initial_call=True,
)
def set_analyzing_flag(n_clicks):
    if not n_clicks:
        return dash.no_update
    return True


# Drive analyze button UI from analyzing flag
@app.callback(
    Output("analyze-fs", "loading"),
    Output("analyze-fs", "disabled"),
    Output("analyze-fs", "children"),
    Input("analyzing-flag", "data"),
)
def reflect_button(analyzing):
    analyzing = bool(analyzing)
    label = "Анализ…" if analyzing else "Анализировать ФС"
    return analyzing, analyzing, label


# Reset analyzing flag when analysis completes (based on status text change)
@app.callback(
    Output("analyzing-flag", "data", allow_duplicate=True),
    Input("upload-status", "children"),
    prevent_initial_call=True,
)
def reset_analyzing_on_status(status_text):
    # When analysis completes or errors, upload-status is updated
    if isinstance(status_text, str) and ("Анализ завершен" in status_text or "Ошибка анализа" in status_text):
        return False
    return dash.no_update


@app.callback(
    Output("chat-answer", "children"),
    Output("chat-answer", "color"),
    Input("chat-ask", "n_clicks"),
    State("chat-question", "value"),
    State("current-file-name", "data"),
    State("chat-scope-file", "checked"),
    prevent_initial_call=True,
)
def on_chat_ask(n_clicks, question, file_name, scope_file):
    try:
        if not n_clicks:
            return dash.no_update, dash.no_update
        q = (question or "").strip()
        if not q:
            return "Введите вопрос", "red"
        # Optionally scope by file name
        kwargs = {"file_name": file_name} if scope_file and file_name else {}
        # Call RAG; run coroutine safely in this thread
        import asyncio
        ans = asyncio.run(search_documents(q, **kwargs))
        import json as _json
        payload = _json.loads(ans)
        answer = payload.get("answer") or ""
        if not answer or answer == "Empty Response":
            return "Ответ не найден в контексте документов.", "yellow"
        # Render markdown in the alert body
        return dcc.Markdown(answer, link_target="_blank", className="fs-md"), "blue"
    except Exception as e:
        log.exception("chat_failed")
        return f"Ошибка: {e}", "red"


# Optional: show loading state on chat button
@app.callback(
    Output("chatting-flag", "data"),
    Input("chat-ask", "n_clicks"),
    prevent_initial_call=True,
)
def set_chatting_flag(n_clicks):
    if not n_clicks:
        return dash.no_update
    return True


@app.callback(
    Output("chat-ask", "loading"),
    Output("chat-ask", "disabled"),
    Output("chat-ask", "children"),
    Input("chatting-flag", "data"),
)
def reflect_chat_button(is_chatting):
    busy = bool(is_chatting)
    label = "Спросить…" if busy else "Спросить"
    return busy, busy, label


@app.callback(
    Output("chatting-flag", "data", allow_duplicate=True),
    Input("chat-answer", "children"),
    prevent_initial_call=True,
)
def reset_chatting_flag(_children):
    # When answer is updated (success or error), reset busy flag
    return False


# Enable/disable Export button based on analysis availability and analyzing state
@app.callback(
    Output("export-excel", "disabled"),
    Input("analysis-result", "data"),
    Input("analyzing-flag", "data"),
)
def toggle_export_button(analysis, analyzing):
    has_data = isinstance(analysis, dict) and len(analysis) > 0
    return bool(analyzing) or (not has_data)


# Generate Excel download from the current analysis
@app.callback(
    Output("download-excel", "data"),
    Input("export-excel", "n_clicks"),
    State("current-file-name", "data"),
    State("analysis-result", "data"),
    prevent_initial_call=True,
)
def export_analysis_excel(n_clicks, file_name, analysis):
    try:
        if not n_clicks or not file_name or not isinstance(analysis, dict) or not analysis:
            return dash.no_update
        # Rebuild rows using the helper to keep consistency with the UI
        rows, _ = _build_sections_table(file_name, analysis=analysis)
        # Keep a concise set of columns
        import pandas as _pd
        import io as _io
        from pathlib import Path as _Path

        df = _pd.DataFrame(rows)
        keep_cols = ['#', COL_SECTION, COL_STATUS, COL_OVERALL, COL_ANALYSIS]
        df = df[[c for c in keep_cols if c in df.columns]]

        buf = _io.BytesIO()
        with _pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Анализ", index=False)
        buf.seek(0)
        safe_name = _Path(file_name).stem or "analysis"
        return dcc.send_bytes(buf.read(), filename=f"Анализ_{safe_name}.xlsx")
    except Exception:
        log.exception("export_excel_failed")
        return dash.no_update


if __name__ == "__main__":
    # Lazy init dirs
    warmup()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7861"))
    app.run(host=host, port=port, debug=False)
