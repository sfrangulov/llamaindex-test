import os
import re
import json
import logging
from typing import List, Tuple

from dotenv import load_dotenv

# Ensure env is loaded (API keys, toggles, etc.)
load_dotenv()

# Local imports kept light to avoid heavy side-effects at import time
from rag_engine import search_documents, warmup

import gradio as gr


# --------------- Theming (light/dark) ---------------
# Adapt highlight colors for day/night themes. We style the <mark> tag used in snippets.
APP_CSS = """
:root {
    --hl-bg: #fff3a6;   /* soft amber for light */
    --hl-fg: #1a1a1a;   /* dark text on light */
    --hl-border: #e9d26a;
}

@media (prefers-color-scheme: dark) {
    :root {
        --hl-bg: #3a2f00; /* deep amber for dark */
        --hl-fg: #fff7cc; /* warm light text */
        --hl-border: #6b5d00;
    }
}

/* Also handle explicit dark themes if Gradio sets a data-theme */
html[data-theme*="dark"] :root {
    --hl-bg: #3a2f00;
    --hl-fg: #fff7cc;
    --hl-border: #6b5d00;
}

/* Highlight style */
mark {
    background-color: var(--hl-bg);
    color: var(--hl-fg);
    padding: 0 0.15em;
    border-radius: 0.2em;
    border: 1px solid var(--hl-border);
}
"""


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
            score = float(s.get("score") or 0)
            title = f"{title} · {score:.2f}"
            snippet = (s.get("text") or "").strip()
            cites.append((title, snippet))
        return answer, cites
    except Exception as e:
        logging.exception("search_documents failed: %s", e)
        return f"Ошибка: {e}", []


def _format_sources_md(rows: List[Tuple[str, str]], query: str) -> str:
    """Render sources as collapsible sections; keep snippet markdown intact.

    - Each source is wrapped in <details> to avoid a very long screen.
    - Titles include rounded scores prepared in _answer_once.
    """
    if not rows:
        return ""

    def _highlight(text: str, q: str) -> str:
        q = (q or "").strip()
        if len(q) < 3:
            return text
        # Highlight distinct words of length >= 3
        words = {w for w in re.split(r"\W+", q, flags=re.UNICODE) if len(w) >= 3}
        if not words:
            return text
        pattern = re.compile(r"(" + "|".join(map(re.escape, sorted(words, key=len, reverse=True))) + r")", re.IGNORECASE | re.UNICODE)
        return pattern.sub(r"<mark>\1</mark>", text)

    parts: List[str] = []
    parts.append(f"Источники: {len(rows)}\n")
    for i, (title, snippet) in enumerate(rows, start=1):
        # Collapsible block per source; allow raw HTML in Markdown
        parts.append(
            "\n".join(
                [
                    f"<details><summary><b>[{i}]</b> {title}</summary>",
                    "",
                    _highlight(snippet, query),
                    "",
                    "</details>",
                ]
            )
        )
    return "\n\n".join(parts)


# --------------- Chat Logic ---------------
with gr.Blocks(title="Поиск по документам", css=APP_CSS) as demo:
    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(height=450, show_copy_button=True, type="messages")
            msg = gr.Textbox(placeholder="Задайте вопрос по документации…", show_label=False, lines=2)
            with gr.Row():
                send = gr.Button("Отправить", variant="primary")
                clear = gr.Button("Очистить")

            # Quick start hints
            with gr.Row():
                hint1 = gr.Button("Как оформить документ «Оприходование излишков товаров»?")
                hint2 = gr.Button("Как рассчитать плановые коэффициенты вскрыши?")
                hint3 = gr.Button("Как загрузить проводки из Excel в документ «Операция»?")

    async def user_submit(user_message: str, history: List[dict]):  # type: ignore[override]
        if not user_message or not user_message.strip():
            return gr.update(), gr.update(value="")
        history = history + [{"role": "user", "content": user_message}]
        answer, cites = await _answer_once(user_message)
        sources_md = _format_sources_md(cites, user_message)
        # Also include context inside the assistant's chat bubble
        assistant_content = answer
        if sources_md:
            assistant_content = f"{answer}\n\n---\n\n{sources_md}"
        history = history + [{"role": "assistant", "content": assistant_content}]
        return gr.update(value=history), gr.update(value="")

    def on_clear():
        return [], ""

    send.click(user_submit, [msg, chat], [chat, msg])
    msg.submit(user_submit, [msg, chat], [chat, msg])
    clear.click(on_clear, None, [chat, msg])

    # Hint buttons: prefill message only (no auto-send)
    hint1.click(lambda: "Как создать документ «Оприходование излишков товаров»?", None, msg)
    hint2.click(lambda: "Как рассчитать коэффициент вскрыши по плановым данным?", None, msg)
    hint3.click(lambda: "Как загрузить проводки из Excel в документ «Операция»?", None, msg)


if __name__ == "__main__":
    warmup()
    demo.queue().launch(server_name=os.getenv("HOST", "0.0.0.0"), server_port=int(os.getenv("PORT", "7860")))
