from __future__ import annotations

import os
from typing import Any

import dash
from flask import send_from_directory, abort
from dash import Input, Output
import dash_mantine_components as dmc
from dash_iconify import DashIconify

from dotenv import load_dotenv

# Load env (API keys, ports, etc.)
load_dotenv()

# Local modules expose per-tab layouts and callbacks
from fs_analyze_ui import get_layout as analyze_get_layout, register_callbacks as analyze_register_callbacks
from fs_chat_ui import get_layout as chat_get_layout, register_callbacks as chat_register_callbacks
from fs_storage_ui import get_layout as storage_get_layout, register_callbacks as storage_register_callbacks
from theme import THEME
from rag_engine import start as rag_start, search_documents
from fs_analyze_agent import start as fs_analyze_agent_start
from db import start as db_start


def _build_header() -> Any:
    theme_toggle = dmc.Switch(
        offLabel=DashIconify(
            icon="radix-icons:sun", width=15, color=dmc.DEFAULT_THEME["colors"]["yellow"][8]
        ),
        onLabel=DashIconify(
            icon="radix-icons:moon",
            width=15,
            color=dmc.DEFAULT_THEME["colors"]["yellow"][6],
        ),
        id="color-scheme-toggle",
        persistence=True,
        color="grey",
    )

    return dmc.AppShellHeader(
        dmc.Group(
            [
                dmc.Group(
                    [
                        DashIconify(icon="hugeicons:artificial-intelligence-04", width=40),
                        dmc.Title("AI-АНАЛИЗ ФС"),
                    ],
                    h="100%",
                    px="md",
                ),
                dmc.Group(
                    [
                        dmc.SegmentedControl(
                            id="top-nav",
                            value="analysis",
                            data=[
                                {"label": "Анализ", "value": "analysis"},
                                {"label": "Чат", "value": "chat"},
                                {"label": "Хранилище", "value": "storage"},
                            ],
                            size="sm",
                        ),
                        theme_toggle,
                    ],
                    gap="md",
                ),
            ],
            justify="space-between",
            style={"flex": 1},
            h="100%",
            px="md",
        )
    )


def create_app() -> dash.Dash:
    external_stylesheets: list[str] = []
    app = dash.Dash(__name__, title="AI-анализ ФС", external_stylesheets=external_stylesheets)
    app.config.suppress_callback_exceptions = True

    header = _build_header()

    # Sections (only one visible at a time)
    analysis_section = dmc.Box(id="analysis-section", children=[analyze_get_layout()])
    chat_section = dmc.Box(id="chat-section", style={"display": "none"}, children=[chat_get_layout()])
    storage_section = dmc.Box(id="storage-section", style={"display": "none"}, children=[storage_get_layout()])

    main = dmc.AppShellMain(children=[analysis_section, chat_section, storage_section])

    app.layout = dmc.MantineProvider(
        dmc.AppShell(children=[header, main], header={"height": 60}, padding="md"),
        theme=THEME,
    )

    # Static route for markdown image assets
    img_root = os.getenv("MD_IMG_DIR", "./data/markdown_assets")

    @app.server.route('/markdown_assets/<path:filename>')
    def _serve_markdown_asset(filename):  # type: ignore
        try:
            return send_from_directory(img_root, filename)
        except Exception:
            abort(404)

    # Register callbacks from individual tabs
    analyze_register_callbacks(app)
    chat_register_callbacks(app, search_documents)
    storage_register_callbacks(app)

    # Toggle which section is visible based on nav
    @app.callback(
        Output("analysis-section", "style"),
        Output("chat-section", "style"),
        Output("storage-section", "style"),
        Input("top-nav", "value"),
    )
    def _toggle_sections(nav_value):  # type: ignore
        if nav_value == "chat":
            return {"display": "none"}, {"display": "block"}, {"display": "none"}
        if nav_value == "storage":
            return {"display": "none"}, {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}, {"display": "none"}

    # Client-side color scheme toggle
    dash.clientside_callback(
        """
        (switchOn) => {
           document.documentElement.setAttribute('data-mantine-color-scheme', switchOn ? 'dark' : 'light');
           return window.dash_clientside.no_update;
        }
        """,
        Output("color-scheme-toggle", "id"),
        Input("color-scheme-toggle", "checked"),
    )

    return app


if __name__ == "__main__":
    rag_start()
    fs_analyze_agent_start()
    db_start()
    app = create_app()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7861"))
    app.run(host=host, port=port, debug=False)
