from rag_engine import search_documents
from fs_similarity import find_similar_subjects, llm_compare_subjects, SUBJECT_SECTION_TITLE
from fs_utils import get_fs, get_section_titles, split_by_sections_fs
from storage import CFG, ensure_dirs, add_docx_to_store, read_markdown
import json
import base64
from pathlib import Path
from typing import Any, Dict, List, Tuple
from threading import Thread, Lock
import time

import dash
from dash import dcc, Input, Output, State, callback_context, ALL
import dash_mantine_components as dmc

import structlog
log = structlog.get_logger(__name__)



# --------- Background analysis state (simple in-process store) ---------
PROGRESS_LOCK = Lock()
ANALYSIS_PROGRESS: Dict[str, Dict[str, Any]] = {}
ANALYSIS_RESULTS: Dict[str, Dict[str, Any]] = {}


def _progress_update(job_id: str, done: int, total: int, current: str):
    with PROGRESS_LOCK:
        st = ANALYSIS_PROGRESS.get(job_id) or {}
        st.update({
            "done": int(done or 0),
            "total": int(total or 0),
            "current": current or "",
            "status": st.get("status") or "running",
        })
        ANALYSIS_PROGRESS[job_id] = st


def _run_analysis_worker(job_id: str, file_name: str, sections_from_md: Dict[str, str]):
    try:
        from fs_analyze_agent import analyze_fs_sections
        from db import save_fs_analysis

        def cb(done: int, total: int, current: str):
            _progress_update(job_id, done, total, current)

        # init progress
        with PROGRESS_LOCK:
            ANALYSIS_PROGRESS[job_id] = {
                "done": 0,
                "total": len(get_section_titles()),
                "current": "",
                "status": "running",
                "file": file_name,
            }

        # Measure runtime
        t0 = time.time()
        analysis = analyze_fs_sections(sections_from_md, progress_cb=cb)
        runtime_ms = int((time.time() - t0) * 1000)

        # Persist results (best-effort)
        try:
            save_fs_analysis(
                file_name=file_name,
                analysis=analysis,
                job_id=job_id,
                runtime_ms=runtime_ms,
                metadata={
                    "source": "fs_analyze_ui",
                    "sections_total": len(get_section_titles()),
                },
            )
        except Exception:
            log.exception("save_analysis_failed")

        with PROGRESS_LOCK:
            ANALYSIS_RESULTS[job_id] = analysis
            st = ANALYSIS_PROGRESS.get(job_id) or {}
            st.update({"status": "done", "done": st.get("total", 0)})
            ANALYSIS_PROGRESS[job_id] = st
    except Exception as e:
        log.exception("analysis_worker_failed", error=str(e))
        with PROGRESS_LOCK:
            st = ANALYSIS_PROGRESS.get(job_id) or {}
            st.update({"status": "error", "error": str(e)})
            ANALYSIS_PROGRESS[job_id] = st


def _parse_upload(contents: str, filename: str) -> Tuple[bool, str]:
    """Persist uploaded DOCX to attachments and index; return (ok, message)."""
    try:
        if not contents or not filename:
            return False, "Нет файла для загрузки"
        # Validate extension: only .docx supported for ФС
        ext = Path(filename).suffix.lower()
        if ext != ".docx":
            return False, f"Неподдерживаемый формат: {ext}. Загрузите файл .docx"
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
COL_STATUS = "Предпросмотр"
COL_OVERALL = "Оценка"


def _build_sections_table(file_name: str, analysis: Dict[str, Any] | None = None) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Return rows for DataTable and a dict of section->content (for modal).

    analysis: optional mapping section-> {summary, details_markdown}
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
        a_overall = None
        if analysis and isinstance(analysis, dict):
            a_data = analysis.get(title) or {}
            if isinstance(a_data, dict):
                a_overall = a_data.get("overall_assessment")
        rows.append({
            "#": i,
            COL_SECTION: title,
            COL_STATUS: "НАЙДЕН" if found else "НЕ НАЙДЕН",
            COL_OVERALL: a_overall or ("—"),
        })
    return rows, section_payload

def get_layout() -> Any:
    return dmc.Stack(
        [
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
                            accept=".docx",
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
                styles={"panel": {"marginTop": 20}},
                children=[
                    dmc.TabsList(children=[
                        dmc.TabsTab("Предпросмотр ФС", value="preview"),
                        dmc.TabsTab("Анализа ФС", value="analysis"),
                        dmc.TabsTab("Схожие ФС", value="similar"),
                        dmc.TabsTab("Вопрос/ответ", value="qa"),
                    ]),

                    dmc.TabsPanel(
                        value="qa",
                        children=[
                            dmc.Paper(
                                withBorder=True,
                                p="md",
                                radius="md",
                                children=[
                                    dmc.Alert(
                                        "Задайте вопрос по содержанию ФС. При включённой опции ‘Искать только в этой ФС’ ответ формируется по выбранному документу; иначе — по всей базе. Источники ответа можно открыть в предпросмотре.",
                                        title="Как работает Вопрос/ответ",
                                        color="gray",
                                    ),
                                    dmc.Space(h=10),
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
                            dmc.Modal(
                                id="doc-preview-modal",
                                title=dmc.Text(id="doc-preview-title", fw=600),
                                children=[
                                    dmc.ScrollArea(
                                        offsetScrollbars=True,
                                        type="auto",
                                        h=500,
                                        children=[
                                            dmc.TypographyStylesProvider(
                                                dcc.Markdown(
                                                    id="doc-preview-content", link_target="_blank", className="markdown")
                                            )
                                        ],
                                    )
                                ],
                                size="xl",
                                centered=True,
                                opened=False,
                            ),
                        ],
                    ),

                    dmc.TabsPanel(
                        value="analysis",
                        children=[
                            dmc.Paper(
                                withBorder=True,
                                p="md",
                                radius="md",
                                children=[
                                    dmc.Alert(
                                        "Функция анализирует ФС по стандартным разделам: последовательно проходит документ, оценивает полноту каждого раздела и формирует краткое резюме. Прогресс ниже. После завершения можно открыть детали по разделу и выгрузить результат в Excel.",
                                        title="Как работает анализ",
                                        color="gray",
                                    ),
                                    dmc.Space(h=10),
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
                                    dmc.Space(h=8),
                                    dmc.Stack([
                                        dmc.Progress(
                                            id="analysis-progress-bar", value=0, color="blue", size="md"),
                                        dmc.Text(id="analysis-progress-text",
                                                 size="sm", c="dimmed"),
                                    ], gap="xs"),
                                    dmc.Space(h=10),
                                    dmc.Box(
                                        id="sections-table-wrap",
                                        style={"overflowX": "auto"},
                                        children=[],
                                    ),
                                    dcc.Download(id="download-excel"),
                                    dcc.Store(id="sections-payload"),
                                    dcc.Store(id="analysis-result"),
                                    dcc.Store(id="analysis-job-id"),
                                    dcc.Interval(
                                        id="analysis-progress-tick", interval=800, disabled=True, n_intervals=0),
                                ],
                            ),

                            dmc.Modal(
                                id="section-modal",
                                title=dmc.Text(id="modal-title", fw=600),
                                children=[
                                    dmc.ScrollArea(
                                        offsetScrollbars=True,
                                        type="auto",
                                        h=500,
                                        children=[
                                            dmc.TypographyStylesProvider(
                                                dcc.Markdown(
                                                    id="modal-content", link_target="_blank", className="markdown")
                                            )
                                        ],
                                    )
                                ],
                                size="xl",
                                centered=True,
                                opened=False,
                            ),

                            dmc.Modal(
                                id="analysis-modal",
                                title=dmc.Text(
                                    id="analysis-modal-title", fw=600),
                                children=[
                                    dmc.ScrollArea(
                                        offsetScrollbars=True,
                                        type="auto",
                                        h=500,
                                        children=[
                                            dmc.TypographyStylesProvider(
                                                dcc.Markdown(
                                                    id="analysis-modal-content", link_target="_blank", className="markdown")
                                            )
                                        ],
                                    )
                                ],
                                size="xl",
                                centered=True,
                                opened=False,
                            ),

                        ],
                    ),

                    dmc.TabsPanel(
                        value="similar",
                        children=[
                            dmc.Paper(
                                withBorder=True,
                                p="md",
                                radius="md",
                                children=[
                                    dmc.Alert(
                                        "Ищет документы с похожим разделом ‘Предмет разработки’ относительно выбранной ФС. Нажмите ‘Поиск’, затем откройте любой найденный документ для предпросмотра.",
                                        title="Как работает поиск похожих (экспериментальная функция)",
                                        color="yellow",
                                    ),
                                    dmc.Space(h=10),
                                    dmc.Group([
                                        dmc.Button(
                                            "Поиск",
                                            id="find-similar-subjects",
                                            variant="light",
                                            color="grape",
                                        ),
                                    ], justify="flex-start"),
                                    dmc.Space(h=10),
                                    dmc.Box(
                                        id="similar-subjects-table-wrap")
                                ],
                            ),
                        ],
                    ),

                    dmc.Modal(
                        id="similar-doc-preview-modal",
                        title=dmc.Text(id="similar-doc-preview-title", fw=600),
                        children=[
                            dmc.ScrollArea(
                                offsetScrollbars=True,
                                type="auto",
                                h=500,
                                children=[
                                    dmc.TypographyStylesProvider(
                                        dcc.Markdown(
                                            id="similar-doc-preview-content", link_target="_blank", className="markdown")
                                    )
                                ],
                            )
                        ],
                        size="xl",
                        centered=True,
                        opened=False,
                    ),

                    dmc.TabsPanel(
                        value="preview",
                        children=[
                            dmc.Paper(
                                withBorder=True,
                                p="md",
                                radius="md",
                                children=[
                                    dmc.Alert(
                                        "Показывает Markdown выбранной ФС. Если разделы распознаны, отображается их содержимое по порядку.",
                                        title="Как работает предпросмотр",
                                        color="gray",
                                    ),
                                    dmc.Space(h=10),
                                    dmc.Box(
                                        children=[
                                            dmc.TypographyStylesProvider(
                                                dcc.Markdown(
                                                    id="full-md-content", link_target="_blank", className="markdown"),
                                            )
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
            dcc.Store(id="finding-similar-flag", data=False),
            dcc.Store(id="similar-subjects-rows", data=[]),
        ]
    )

def _render_table(rows: List[Dict[str, Any]]):
    header = dmc.TableThead(dmc.TableTr([
        dmc.TableTh("#"),
        dmc.TableTh(COL_SECTION),
        dmc.TableTh(COL_STATUS),
        dmc.TableTh(COL_OVERALL),
    ]))
    body_rows = []
    for r in rows:
        ok = (r.get(COL_STATUS) == "НАЙДЕН")
        status_btn = (
            dmc.Button("НАЙДЕН", color="green", variant="light",
                       id={"type": "view-btn", "section": r.get(COL_SECTION)})
            if ok else dmc.Button("НЕ НАЙДЕН", color="red", variant="light", disabled=True)
        )
        overall_text = (r.get(COL_OVERALL) or "")
        if not overall_text or overall_text == "—":
            overall_cell = dmc.Text("—")
        else:
            if overall_text.startswith("ПОЛНОСТЬЮ"):
                color = "green"
            elif overall_text.startswith("ЧАСТИЧНО"):
                color = "orange"
            else:
                color = "red"
            overall_cell = dmc.Button(
                overall_text, color=color, variant="light", size="xs",
                id={"type": "analysis-btn", "section": r.get(COL_SECTION)})
        body_rows.append(dmc.TableTr([
            dmc.TableTd(str(r.get("#", ""))),
            dmc.TableTd(r.get(COL_SECTION, "")),
            dmc.TableTd(status_btn),
            dmc.TableTd(overall_cell),
        ]))
    body = dmc.TableTbody(body_rows)
    table = dmc.Table(highlightOnHover=True, striped=True,
                      verticalSpacing="sm", horizontalSpacing="md", children=[header, body])
    return table


def _render_similar_table(rows: List[Dict[str, Any]]):
    header = dmc.TableThead(dmc.TableTr([
        dmc.TableTh("#"),
        dmc.TableTh("Название документа"),
        dmc.TableTh("Процент совпадения"),
        dmc.TableTh("Сходства"),
        dmc.TableTh("Отличия"),
    ]))
    body_rows = []
    for i, r in enumerate(rows or [], start=1):
        name = r.get("file_name", "")
        score = r.get("score", 0)
        pct = round(float(score) * 100, 2)
        sim_txt = (r.get("llm_similar") or "—")
        diff_txt = (r.get("llm_diff") or "—")
        link_btn = dmc.Tooltip(
            dmc.Button(
                name,
                variant="subtle",
                color="blue",
                id={"type": "open-similar-doc", "file": name},
                maw=250,
            ),
            label=name,
        )
        body_rows.append(dmc.TableTr([
            dmc.TableTd(str(i)),
            dmc.TableTd(link_btn, maw=300),
            dmc.TableTd(f"{pct}%"),
            dmc.TableTd(sim_txt),
            dmc.TableTd(diff_txt),
        ]))
    body = dmc.TableTbody(body_rows)
    table = dmc.Table(highlightOnHover=True, striped=True,
                      verticalSpacing="sm", horizontalSpacing="md", children=[header, body])
    return table


def register_callbacks(app: dash.Dash):
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
        style_visible = {"display": "block"}
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
            if not clicks or all(((c or 0) <= 0) for c in clicks):
                return False, dash.no_update, dash.no_update
            ctx = callback_context
            tid = getattr(ctx, "triggered_id", None)
            section_title = None
            if isinstance(tid, dict):
                section_title = tid.get("section")
            else:
                trg = getattr(ctx, "triggered", []) or []
                if trg:
                    prop_id = trg[0].get("prop_id", "")
                    comp_id = prop_id.split(".")[0]
                    if comp_id.startswith("{"):
                        try:
                            d = json.loads(comp_id)
                            section_title = d.get("section")
                        except Exception:
                            section_title = None
            if not section_title:
                return False, dash.no_update, dash.no_update
            content = (payload or {}).get(section_title) or "Раздел не найден"
            return True, section_title, content
        except Exception:
            return False, dash.no_update, dash.no_update

    @app.callback(
        Output("analysis-modal", "opened"),
        Output("analysis-modal-title", "children"),
        Output("analysis-modal-content", "children"),
        Input({"type": "analysis-btn", "section": ALL}, "n_clicks"),
        State("analysis-result", "data"),
        prevent_initial_call=True,
    )
    def on_analysis_click(clicks, analysis):
        try:
            if not clicks or all(((c or 0) <= 0) for c in clicks):
                return False, dash.no_update, dash.no_update
            ctx = callback_context
            tid = getattr(ctx, "triggered_id", None)
            section = None
            if isinstance(tid, dict):
                section = tid.get("section")
            else:
                trg = getattr(ctx, "triggered", []) or []
                if trg:
                    prop_id = trg[0].get("prop_id", "")
                    comp_id = prop_id.split(".")[0]
                    if comp_id.startswith("{"):
                        try:
                            d = json.loads(comp_id)
                            section = d.get("section")
                        except Exception:
                            section = None
            if not section:
                return False, dash.no_update, dash.no_update
            details = ""
            if isinstance(analysis, dict):
                sec = analysis.get(section) or {}
                if isinstance(sec, dict):
                    details = (sec.get("details_markdown") or sec.get("summary") or "").strip()
            if not details:
                details = "(Детали анализа отсутствуют)"
            return True, section, details
        except Exception:
            return False, dash.no_update, dash.no_update

    @app.callback(
        Output("full-md-content", "children"),
        Input("main-tabs", "value"),
        Input("current-file-name", "data"),
    )
    def update_full_preview(active_tab, file_name):
        try:
            if active_tab != "preview" or not file_name:
                return dash.no_update
            md = read_markdown(file_name)
            if not isinstance(md, str) or md.startswith("Файл Markdown не найден"):
                return "(Предпросмотр недоступен: Markdown не найден)"
            sections = split_by_sections_fs(md)
            if sections:
                return "\n".join(sections.values())
            return md
        except Exception:
            return dash.no_update

    @app.callback(
        Output("analysis-job-id", "data"),
        Output("analysis-progress-tick", "disabled"),
        Output("analysis-progress-bar", "value"),
        Output("analysis-progress-text", "children"),
        Output("analysis-result", "data", allow_duplicate=True),
        Output("upload-status", "children", allow_duplicate=True),
        Output("upload-status", "color", allow_duplicate=True),
        Input("analyze-fs", "n_clicks"),
        State("current-file-name", "data"),
        prevent_initial_call=True,
    )
    def start_analysis_job(n_clicks, file_name):
        try:
            if not n_clicks or not file_name:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            try:
                md = read_markdown(file_name)
                sections_from_md = get_fs(str(
                    CFG.md_dir / f"{file_name}.md")) if md and not md.startswith("Файл Markdown не найден") else {}
            except Exception:
                sections_from_md = {}

            job_id = f"{file_name}:{int(time.time()*1000)}"
            with PROGRESS_LOCK:
                ANALYSIS_PROGRESS[job_id] = {"done": 0, "total": len(
                    get_section_titles()), "current": "", "status": "running", "file": file_name}
            t = Thread(target=_run_analysis_worker, args=(
                job_id, file_name, sections_from_md), daemon=True)
            t.start()
            return job_id, False, 0, "Подготовка…", {}, "Анализ запущен", "blue"
        except Exception as e:
            log.exception("analyze_start_failed")
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, f"Ошибка анализа: {e}", "red"

    @app.callback(
        Output("analysis-progress-tick", "disabled", allow_duplicate=True),
        Output("analysis-progress-bar", "value", allow_duplicate=True),
        Output("analysis-progress-text", "children", allow_duplicate=True),
        Output("sections-table-wrap", "children", allow_duplicate=True),
        Output("sections-payload", "data", allow_duplicate=True),
        Output("analysis-result", "data", allow_duplicate=True),
        Output("upload-status", "children", allow_duplicate=True),
        Output("upload-status", "color", allow_duplicate=True),
        Output("analyzing-flag", "data", allow_duplicate=True),
        Input("analysis-progress-tick", "n_intervals"),
        State("analysis-job-id", "data"),
        State("current-file-name", "data"),
        prevent_initial_call=True,
    )
    def poll_analysis_progress(_n, job_id, file_name):
        try:
            if not job_id:
                return True, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            with PROGRESS_LOCK:
                st = (ANALYSIS_PROGRESS.get(job_id) or {}).copy()
            total = int(st.get("total")) if st else 0
            done = int(st.get("done")) if st else 0
            current = st.get("current") or ""
            status = st.get("status") or "running"
            value = int((done / total) * 100) if total else 0
            text = f"{done}/{total} — Анализ раздела \"{current}\"" if current else "Подготовка…"

            if status == "done":
                analysis = ANALYSIS_RESULTS.get(job_id) or {}
                rows, payload = _build_sections_table(file_name, analysis=analysis)
                table = _render_table(rows)
                return True, 100, f"Готово: {done}/{total}", table, payload, analysis, "Анализ завершен", "blue", False
            if status == "error":
                err = st.get("error") or "Ошибка анализа"
                return True, value, text, dash.no_update, dash.no_update, dash.no_update, f"Ошибка анализа: {err}", "red", False
            return False, value, text, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        except Exception as e:
            log.exception("poll_progress_failed")
            return True, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, f"Ошибка анализа: {e}", "red", False

    @app.callback(
        Output("analyzing-flag", "data"),
        Input("analyze-fs", "n_clicks"),
        prevent_initial_call=True,
    )
    def set_analyzing_flag(n_clicks):
        if not n_clicks:
            return dash.no_update
        return True

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

    @app.callback(
        Output("analyzing-flag", "data", allow_duplicate=True),
        Input("upload-status", "children"),
        prevent_initial_call=True,
    )
    def reset_analyzing_on_status(status_text):
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
            kwargs = {"file_name": file_name} if scope_file and file_name else {}
            import asyncio
            ans = asyncio.run(search_documents(q, **kwargs))
            payload = json.loads(ans)
            answer = (payload.get("answer") or "").strip()
            sources = payload.get("sources") or []
            if not answer or answer == "Empty Response":
                return "Ответ не найден в контексте документов.", "yellow"
            if sources:
                parts = [answer, "", "---", "", "Источники:"]
                answer = "\n".join(parts)
            alert_body = dmc.Stack([
                dmc.TypographyStylesProvider(
                    dcc.Markdown(answer, link_target="_blank", className="markdown")),
            ])
            if sources:
                btns = []
                seen_files = set()
                for s in sources:
                    fname = (s.get("file_name") or "source").strip()
                    if fname in seen_files:
                        continue
                    seen_files.add(fname)
                    btns.append(
                        dmc.Button(
                            fname,
                            variant="light",
                            size="xs",
                            id={"type": "open-doc-btn", "file": fname},
                        )
                    )
                alert_body.children.append(dmc.Group(btns, gap="sm", wrap="wrap"))
            return alert_body, "blue"
        except Exception as e:
            log.exception("chat_failed")
            return f"Ошибка: {e}", "red"

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
        return False

    @app.callback(
        Output("doc-preview-modal", "opened"),
        Output("doc-preview-title", "children"),
        Output("doc-preview-content", "children"),
        Input({"type": "open-doc-btn", "file": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def open_doc_preview(clicks):
        try:
            if not clicks or all(((c or 0) <= 0) for c in clicks):
                return False, dash.no_update, dash.no_update
            ctx = callback_context
            tid = getattr(ctx, "triggered_id", None)
            file_name = None
            if isinstance(tid, dict):
                file_name = tid.get("file")
            else:
                trg = getattr(ctx, "triggered", []) or []
                if trg:
                    prop_id = trg[0].get("prop_id", "")
                    comp_id = prop_id.split(".")[0]
                    if comp_id.startswith("{"):
                        try:
                            d = json.loads(comp_id)
                            file_name = d.get("file")
                        except Exception:
                            file_name = None
            if not file_name:
                return False, dash.no_update, dash.no_update
            md = read_markdown(file_name)
            return True, file_name, md
        except Exception:
            return False, dash.no_update, dash.no_update

    @app.callback(
        Output("export-excel", "disabled"),
        Input("analysis-result", "data"),
        Input("analyzing-flag", "data"),
    )
    def toggle_export_button(analysis, analyzing):
        has_data = isinstance(analysis, dict) and len(analysis) > 0
        return bool(analyzing) or (not has_data)

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
            rows, _ = _build_sections_table(file_name, analysis=analysis)
            import pandas as _pd
            import io as _io
            from pathlib import Path as _Path

            df = _pd.DataFrame(rows)
            keep_cols = ['#', COL_SECTION, COL_STATUS, COL_OVERALL]
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

    @app.callback(
        Output("similar-subjects-table-wrap", "children"),
        Output("similar-subjects-rows", "data"),
        Input("find-similar-subjects", "n_clicks"),
        State("current-file-name", "data"),
        prevent_initial_call=True,
    )
    def on_find_similar_subjects(n_clicks, file_name):
        try:
            if not n_clicks or not file_name:
                return dash.no_update, dash.no_update
            matches = find_similar_subjects(file_name, top_k=7) or []
            if not matches:
                empty = dmc.Alert(
                    "Не найдено похожих ФС по разделу 'Предмет разработки'.", color="gray")
                return empty, []
            try:
                base_sections = get_fs(str(CFG.md_dir / f"{file_name}.md"))
                base_subject = (base_sections or {}).get(
                    SUBJECT_SECTION_TITLE) or ""
            except Exception:
                base_subject = ""
            enriched: List[Dict[str, Any]] = []
            for m in matches:
                cand_text = (m.get("text") or "").strip()
                sim, diff = ("—", "—")
                if base_subject and cand_text:
                    sim, diff = llm_compare_subjects(base_subject, cand_text)
                enriched.append({
                    **m,
                    "llm_similar": sim,
                    "llm_diff": diff,
                })
            table = _render_similar_table(enriched)
            return table, enriched
        except Exception as e:
            log.exception("similar_subjects_failed")
            return dmc.Alert(f"Ошибка поиска похожих: {e}", color="red"), []

    @app.callback(
        Output("finding-similar-flag", "data"),
        Input("find-similar-subjects", "n_clicks"),
        prevent_initial_call=True,
    )
    def set_finding_similar_flag(n_clicks):
        if not n_clicks:
            return dash.no_update
        return True

    @app.callback(
        Output("find-similar-subjects", "loading"),
        Output("find-similar-subjects", "disabled"),
        Output("find-similar-subjects", "children"),
        Input("finding-similar-flag", "data"),
    )
    def reflect_find_similar_button(is_busy):
        busy = bool(is_busy)
        label = "Ищем…" if busy else "Поиск"
        return busy, busy, label

    @app.callback(
        Output("finding-similar-flag", "data", allow_duplicate=True),
        Input("similar-subjects-table-wrap", "children"),
        prevent_initial_call=True,
    )
    def reset_finding_similar_flag(_):
        return False

    @app.callback(
        Output("similar-doc-preview-modal", "opened"),
        Output("similar-doc-preview-title", "children"),
        Output("similar-doc-preview-content", "children"),
        Input({"type": "open-similar-doc", "file": ALL}, "n_clicks"),
        State("similar-subjects-rows", "data"),
        prevent_initial_call=True,
    )
    def open_similar_doc_preview(clicks, rows):
        try:
            if not clicks or all(((c or 0) <= 0) for c in clicks):
                return False, dash.no_update, dash.no_update
            ctx = callback_context
            tid = getattr(ctx, "triggered_id", None)
            file_name = None
            if isinstance(tid, dict):
                file_name = tid.get("file")
            else:
                trg = getattr(ctx, "triggered", []) or []
                if trg:
                    prop_id = trg[0].get("prop_id", "")
                    comp_id = prop_id.split(".")[0]
                    if comp_id.startswith("{"):
                        try:
                            d = json.loads(comp_id)
                            file_name = d.get("file")
                        except Exception:
                            file_name = None
            if not file_name:
                return False, dash.no_update, dash.no_update
            md = read_markdown(file_name)
            return True, file_name, md
        except Exception:
            return False, dash.no_update, dash.no_update

