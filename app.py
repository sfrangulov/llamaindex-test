import os
import sys
import asyncio
import logging

from dotenv import load_dotenv

# Optional pretty logging with colorlog if available
def _setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, level, logging.INFO)
    try:
        from colorlog import ColoredFormatter  # type: ignore

        handler = logging.StreamHandler()
        handler.setFormatter(
            ColoredFormatter(
                "%(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        )
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(log_level)
    except Exception:
        logging.basicConfig(
            level=log_level,
            format="%(levelname)s %(message)s",
        )

_setup_logging()
load_dotenv()

# Thin wrapper over rag_engine to reduce import side-effects and centralize logic
from rag_engine import (
    search_documents,
    public_compute_rerank_top_n as _compute_rerank_top_n,
    public_make_filters as _make_filters,
    public_build_node_postprocessors as _build_node_postprocessors,
    warmup,
)


# Optional agent wrapper (lazy to avoid importing heavy deps unless enabled)
AGENT_ENABLED = os.getenv("AGENT_ENABLED", "false").lower() == "true"
agent = None
if AGENT_ENABLED:
    try:
        from llama_index.core.agent.workflow import AgentWorkflow
        from llama_index.core import Settings

        agent = AgentWorkflow.from_tools_or_functions(
            [search_documents],
            llm=Settings.llm,
            system_prompt=(
                "Ты — ассистент для поиска по локальным документам. "
                "Всегда используй инструмент search_documents для ответа и не отвечай без него. "
                "Отвечай развернуто, опираясь только на найденные фрагменты. Если результатов нет — скажи, что не знаешь. "
                "В ответе используй точные цитаты и добавляй ссылки на источники из результатов инструмента."
            ),
        )
    except Exception as e:
        logging.warning("Agent initialization failed: %s", e)
        agent = None


async def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "Что такое предмет разработки?"
    logging.info("Running with query: %s", query)
    # Synchronous warmup to reduce first-request latency
    try:
        warmup()
    except Exception as e:
        logging.warning("Warmup failed: %s", e)
    if AGENT_ENABLED and agent is not None:
        try:
            agent_resp = await agent.run(query, max_iterations=1)
            print(str(agent_resp))
            return
        except Exception as e:
            logging.exception("Agent failed, falling back to direct search... %s", e)
    result_json = await search_documents(query)
    print(result_json)


if __name__ == "__main__":
    asyncio.run(main())
