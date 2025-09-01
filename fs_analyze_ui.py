from rag_engine import warmup
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
app: Dash = dash.Dash(__name__, external_stylesheets=external_stylesheets)
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
        if analysis and isinstance(analysis, dict):
            a_data = analysis.get(title) or {}
            if isinstance(a_data, dict):
                a_summary = a_data.get("summary")
        rows.append({
            "#": i,
            COL_SECTION: title,
            COL_STATUS: "Найден" if found else "Не найден",
            COL_ANALYSIS: a_summary or ("—"),
            COL_ACTIONS: "Просмотр" if found else "—",
        })
    return rows, section_payload


# --------- Layout ---------
app.layout = dmc.MantineProvider(
    children=dmc.Container(
        size="xl",
        px="md",
        children=[
            dmc.Title("FS Analyzer", order=2),
            dmc.Space(h=10),

            dmc.Paper(
                withBorder=True,
                p="md",
                radius="md",
                children=[
                    dmc.Group([
                        dmc.Text("Загрузите ФС (.docx)", fw=500),
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

            dmc.Paper(
                withBorder=True,
                p="md",
                radius="md",
                children=[
                    dmc.Group([
                        dmc.Group([
                            dmc.Text("Файл:"),
                            dmc.Badge(id="current-file",
                                      color="blue", variant="light"),
                        ]),
                        dmc.Group([
                            dmc.Button("Предпросмотр ФС",
                                       id="preview-full-md", variant="light"),
                            dmc.Button("Анализировать ФС", id="analyze-fs",
                                       variant="filled", color="blue"),
                        ], gap="sm"),
                    ], justify="space-between"),
                    dmc.Space(h=10),
                    dmc.Box(id="sections-table-wrap", style={"overflowX": "auto"},
                            children=[],
                            ),
                    dcc.Store(id="sections-payload"),
                ],
            ),

            # Modal for preview
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
                                id="modal-content", link_target="_blank")
                        ],
                    )
                ],
                size="xl",
                centered=True,
                opened=False,
            ),

            # Modal for full FS preview
            dmc.Modal(
                id="full-md-modal",
                title=dmc.Text(id="full-md-title", fw=600),
                children=[
                    dmc.ScrollArea(
                        offsetScrollbars=True,
                        type="auto",
                        h=600,
                        children=[
                            dcc.Markdown(id="full-md-content",
                                         link_target="_blank"),
                        ],
                    )
                ],
                size="xl",
                centered=True,
                opened=False,
            ),

            dcc.Store(id="current-file-name"),
        ],
    )
)


# --------- Callbacks ---------
def _render_table(rows: List[Dict[str, Any]]):
    header = dmc.TableThead(dmc.TableTr([
        dmc.TableTh("#"),
        dmc.TableTh(COL_SECTION),
        dmc.TableTh(COL_STATUS),
        dmc.TableTh(COL_ANALYSIS),
        dmc.TableTh(COL_ACTIONS),
    ]))
    body_rows = []
    for r in rows:
        ok = (r.get(COL_STATUS) == "Найден")
        status = dmc.Badge("OK", color="green", variant="light") if ok else dmc.Badge(
            "Не найден", color="red", variant="light")
        analysis_text = (r.get(COL_ANALYSIS) or "").strip()
        if not analysis_text or analysis_text == "—":
            analysis_cell = dmc.Text("—")
        else:
            green = analysis_text.lower().startswith("ok") or analysis_text.lower().startswith("ок")
            # Truncate badge label, show full text in tooltip for readability
            trimmed = analysis_text
            max_len = 40
            if len(trimmed) > max_len:
                trimmed = trimmed[: max_len - 1].rstrip() + "…"
            analysis_cell = dmc.Tooltip(
                label=dmc.Text(analysis_text, style={"whiteSpace": "pre-wrap"}),
                multiline=True,
                withArrow=True,
                position="top-start",
                children=dmc.Badge(trimmed, color=("green" if green else "orange"), variant="light"),
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
    Input("upload-docx", "contents"),
    State("upload-docx", "filename"),
    prevent_initial_call=True,
)
def on_upload(contents, filename):
    ok, msg = _parse_upload(contents, filename)
    if not ok:
        return msg, "red", "", None, [], {}
    file_name = Path(filename).name
    rows, payload = _build_sections_table(file_name)
    table = _render_table(rows)
    return msg, "green", file_name, file_name, table, payload


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
    Output("full-md-modal", "opened"),
    Output("full-md-title", "children"),
    Output("full-md-content", "children"),
    Input("preview-full-md", "n_clicks"),
    State("current-file-name", "data"),
    prevent_initial_call=True,
)
def on_preview_full(n_clicks, file_name):
    try:
        if not n_clicks:
            return False, dash.no_update, dash.no_update
        if not file_name:
            return False, dash.no_update, dash.no_update
        md = read_markdown(file_name)
        if md.startswith("Файл Markdown не найден") or md.startswith("Ошибка"):
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
        return True, file_name, md
    except Exception:
        return False, dash.no_update, dash.no_update


@app.callback(
    Output("sections-table-wrap", "children", allow_duplicate=True),
    Output("sections-payload", "data", allow_duplicate=True),
    Output("upload-status", "children", allow_duplicate=True),
    Output("upload-status", "color", allow_duplicate=True),
    Input("analyze-fs", "n_clicks"),
    State("current-file-name", "data"),
    prevent_initial_call=True,
)
def on_analyze_fs(n_clicks, file_name):
    try:
        if not n_clicks or not file_name:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
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
        return table, payload, "Анализ завершен", "blue"
    except Exception as e:
        log.exception("analyze_failed")
        return dash.no_update, dash.no_update, f"Ошибка анализа: {e}", "red"


if __name__ == "__main__":
    # Lazy init dirs
    warmup()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7861"))
    app.run(host=host, port=port, debug=False)
