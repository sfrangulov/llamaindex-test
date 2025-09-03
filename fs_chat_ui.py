from __future__ import annotations

from typing import Any, Dict, List
import json

import dash
from dash import dcc, html, Input, Output, State, clientside_callback
import dash_mantine_components as dmc
from storage import read_markdown


def _render_chat_messages(history: List[Dict[str, Any]] | None) -> List[Any]:
    history = history or []
    ui: List[Any] = []
    for i, msg in enumerate(history):
        role = (msg.get("role") or "user").lower()
        text = (msg.get("text") or "").strip()
        sources = msg.get("sources") or []
        bubble_color = "gray" if role == "user" else "blue"
        align = "flex-end" if role == "user" else "flex-start"
        # Message bubble
        bubble_children = [
            dmc.TypographyStylesProvider(
                dcc.Markdown(text, link_target="_blank", className="markdown")
            )
        ]
        # Sources row (assistant only)
        if role == "assistant" and sources:
            btns = []
            seen = set()
            for s in sources:
                fname = (s.get("file_name") or "source").strip()
                if not fname or fname in seen:
                    continue
                seen.add(fname)
                btns.append(
                    dmc.Button(
                        fname,
                        variant="light",
                        size="xs",
                        id={"type": "dlg-open-doc-btn", "file": fname},
                    )
                )
            bubble_children.extend([
                dmc.Divider(variant="dashed", my=6),
                dmc.Text("Источники:", size="xs", c="dimmed"),
                dmc.Group(btns, gap="sm", wrap="wrap"),
            ])

        ui.append(
            dmc.Group(
                [
                    dmc.Paper(
                        withBorder=True,
                        shadow="xs",
                        radius="sm",
                        p="sm",
                        maw=800,
                        children=bubble_children,
                        style={
                            "background": "var(--mantine-color-{}-0)".format(bubble_color)},
                    )
                ],
                justify=align,
            )
        )
    # Add an anchor before the latest message to scroll into view
    if ui:
        ui.insert(len(ui) - 1, html.Div(id="dlg-scroll-anchor"))
    return ui


def get_layout() -> Any:
    return dmc.Stack(
        [
            dmc.Paper(
                withBorder=True,
                p="md",
                radius="md",
                children=[
                    dmc.Alert(
                        "Диалоговый режим с RAG: задавайте вопросы, ответы формируются по базе базе знаний. Нажмите на имея файла в источниках, чтобы открыть предпросмотр.",
                        title="Как работает чат с базой знаний",
                        color="gray",
                    ),
                    dmc.Space(h=10),
                    dmc.ScrollArea(
                        offsetScrollbars=True,
                        type="auto",
                        h=420,
                        children=dmc.Stack(id="dlg-thread", gap="xs"),
                    ),
                    dmc.Space(h=8),
                    dmc.Group(
                        [
                            dmc.Textarea(
                                id="dlg-input",
                                placeholder="Ваш вопрос…",
                                autosize=True,
                                minRows=2,
                                style={"flex": 1},
                            ),
                        ],
                        align="stretch",
                    ),
                    dmc.Space(h=6),
                    dmc.Group(
                        [
                            dmc.Button("Спросить", id="dlg-send",
                                       variant="filled", color="blue"),
                            dmc.Button("Очистить чат", id="dlg-clear",
                                       variant="outline", color="red"),
                        ],
                        gap="sm",
                    ),
                ],
            ),
            # Stores
            dcc.Store(id="dlg-history", data=[]),
            dcc.Store(id="dlg-busy", data=False),
            dcc.Store(id="dlg-scroll", data=None),
            # Chat-level document preview modal
            dmc.Modal(
                id="dlg-doc-preview-modal",
                title=dmc.Text(id="dlg-doc-preview-title", fw=600),
                children=[
                    dmc.ScrollArea(
                        offsetScrollbars=True,
                        type="auto",
                        h=500,
                        children=[
                            dmc.TypographyStylesProvider(
                                dcc.Markdown(
                                    id="dlg-doc-preview-content", link_target="_blank", className="markdown")
                            )
                        ],
                    )
                ],
                size="xl",
                centered=True,
                opened=False,
            ),
        ]
    )


def register_callbacks(app: dash.Dash, search_documents):
    # Mark busy when send is clicked (fast UI feedback)
    @app.callback(
        Output("dlg-busy", "data"),
        Input("dlg-send", "n_clicks"),
        prevent_initial_call=True,
    )
    def set_busy(n):
        if not n:
            return dash.no_update
        return True

    # Reflect busy state on the send button
    @app.callback(
        Output("dlg-send", "loading"),
        Output("dlg-send", "disabled"),
        Output("dlg-send", "children"),
        Input("dlg-busy", "data"),
    )
    def reflect_busy(busy):
        is_busy = bool(busy)
        label = "Спросить…" if is_busy else "Спросить"
        return is_busy, is_busy, label

    @app.callback(
        Output("dlg-history", "data"),
        Output("dlg-input", "value"),
        Input("dlg-send", "n_clicks"),
        State("dlg-input", "value"),
        State("dlg-history", "data"),
        prevent_initial_call=True,
    )
    def on_send(n, text, history):
        try:
            if not n:
                return dash.no_update, dash.no_update
            q = (text or "").strip()
            if not q:
                return dash.no_update, dash.no_update
            history = list(history or [])
            history.append({"role": "user", "text": q})

            # Build conversational context for the model (compact, last 6 turns)
            turns = []
            for m in history[-12:]:
                r = m.get("role")
                t = (m.get("text") or "").strip()
                if not r or not t:
                    continue
                turns.append(f"[{r}] {t}")
            conv = "\n".join(turns[-12:])
            conv_query = f"Контекст беседы (кратко):\n{conv}\n\nТекущий вопрос: {q}"

            # Scope by file if requested and available
            kwargs: Dict[str, Any] = {}

            import asyncio
            raw = asyncio.run(search_documents(conv_query, **kwargs))
            payload = json.loads(raw or "{}")
            answer = (payload.get("answer") or "").strip()
            sources = payload.get("sources") or []
            if not answer or answer == "Empty Response":
                answer = "(Ответ не найден в контексте документов.)"

            history.append({
                "role": "assistant",
                "text": answer,
                "sources": sources,
            })
            return history, ""
        except Exception as e:
            history = list(history or [])
            history.append({"role": "assistant", "text": f"Ошибка: {e}"})
            return history, dash.no_update

    # Reset busy once history updates (success or error)
    @app.callback(
        Output("dlg-busy", "data", allow_duplicate=True),
        Input("dlg-history", "data"),
        prevent_initial_call=True,
    )
    def reset_busy(_):
        return False

    @app.callback(
        Output("dlg-thread", "children"),
        Input("dlg-history", "data"),
    )
    def render_thread(history):
        return _render_chat_messages(history)

    @app.callback(
        Output("dlg-history", "data", allow_duplicate=True),
        Input("dlg-clear", "n_clicks"),
        prevent_initial_call=True,
    )
    def on_clear(n):
        if not n:
            return dash.no_update
        return []

    # Open preview of a source document from chat
    @app.callback(
        Output("dlg-doc-preview-modal", "opened"),
        Output("dlg-doc-preview-title", "children"),
        Output("dlg-doc-preview-content", "children"),
        Input({"type": "dlg-open-doc-btn", "file": dash.ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def on_open_chat_doc(clicks):
        try:
            if not clicks or all(((c or 0) <= 0) for c in clicks):
                return False, dash.no_update, dash.no_update
            from dash import callback_context as _ctx
            tid = getattr(_ctx, "triggered_id", None)
            fname = None
            if isinstance(tid, dict):
                fname = tid.get("file")
            else:
                trig = getattr(_ctx, "triggered", []) or []
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

    # Smooth scroll to newest message when the thread updates
    clientside_callback(
        """
        (children) => {
            const el = document.getElementById('dlg-scroll-anchor');
            if (el && typeof el.scrollIntoView === 'function') {
                try {
                    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
                } catch (e) {
                    el.scrollIntoView();
                }
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("dlg-scroll", "data"),
        Input("dlg-thread", "children"),
    )
