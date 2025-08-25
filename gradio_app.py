import os
import json
import html
import logging
import asyncio
from typing import List, Tuple, Any

from dotenv import load_dotenv

# Ensure env is loaded (API keys, toggles, etc.)
load_dotenv()

# Local imports kept light to avoid heavy side-effects at import time
from rag_engine import search_documents

import gradio as gr


# --------------- Helpers ---------------
async def _answer_once(message: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Call search_documents and format answer + sources."""
    try:
        result_json = await search_documents(message)
        payload = json.loads(result_json)
        answer = payload.get("answer") or ""
        sources = payload.get("sources") or []
        # Make a compact citations list
        cites: List[Tuple[str, str]] = []
        for s in sources:
            title = (s.get("file_name") or "source")
            section = s.get("section")
            if section:
                title = f"{title} · {section}"
            snippet = (s.get("text") or "").strip()
            if len(snippet) > 400:
                snippet = snippet[:400] + "…"
            cites.append((title, snippet))
        return answer, cites
    except Exception as e:
        logging.exception("search_documents failed: %s", e)
        return f"Ошибка: {e}", []


def _format_sources_html(rows: List[Tuple[str, str]]) -> str:
    if not rows:
        return ""
    items = []
    for title, snippet in rows:
        safe_title = html.escape(title)
        safe_snippet = html.escape(snippet)
        items.append(
            f"<div style='margin-bottom: 8px'><b>{safe_title}</b><br><small>{safe_snippet}</small></div>"
        )
    return "".join(items)


# --------------- Chat Logic ---------------
with gr.Blocks(title="Docs Chat") as demo:
    gr.Markdown("# Поиск по документам — чат")
    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(height=500, show_copy_button=True, type="messages")
            msg = gr.Textbox(placeholder="Задайте вопрос по документации…", label="Сообщение", lines=2)
            send = gr.Button("Отправить", variant="primary")
            clear = gr.Button("Очистить")
        with gr.Column(scale=2):
            gr.Markdown("### Источники")
            sources = gr.HTML(value="", label="Цитаты", elem_id="sources")

    async def user_submit(user_message: str, history: List[dict]):  # type: ignore[override]
        if not user_message or not user_message.strip():
            return gr.update(), history, gr.update(value="")
        history = history + [{"role": "user", "content": user_message}]
        answer, cites = await _answer_once(user_message)
        history = history + [{"role": "assistant", "content": answer}]
        return gr.update(value=history), _format_sources_html(cites), gr.update(value="")

    def on_clear():
        return [], "", ""

    send.click(user_submit, [msg, chat], [chat, sources, msg])
    msg.submit(user_submit, [msg, chat], [chat, sources, msg])
    clear.click(on_clear, None, [chat, sources, msg])


if __name__ == "__main__":
    # Launch Gradio app
    demo.queue().launch(server_name=os.getenv("HOST", "0.0.0.0"), server_port=int(os.getenv("PORT", "7860")))
