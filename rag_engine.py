from dotenv import load_dotenv
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core import Settings
from llama_index.core.schema import MetadataMode
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.prompts import PromptTemplate
from llama_index.core.memory import Memory
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import (
    Workflow,
    Event,
    step,
    Context,
    StartEvent,
    StopEvent,
)
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from google.genai.types import EmbedContentConfig
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.google_genai import GoogleGenAI
from storage import get_index
import os
import time
import json
from typing import Any, Dict, List
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)

from storage import ensure_dirs

import structlog
log = structlog.get_logger(__name__)


load_dotenv()


# ---------------------------- Configuration ----------------------------

reformulate_template = (
    "Ты переписываешь пользовательский запрос для RAG-поиска.\n"
    "Цель — оставить только суть, чтобы улучшить векторный поиск.\n"
    "Требования: \n"
    "- Отвечай только на том же языке что и запрос.\n"
    "- Оставляй в ответе слова на том же языке что они написаны.\n"
    "- Сохраняй ключевые сущности, термины, даты, аббревиатуры, единицы.\n"
    "- Убери вводные слова, вежливость, местоимения и лишние подробности.\n"
    "- Нормализуй формулировку (кратко, без воды), до ~3–15 слов.\n"
    "- Если запрос уже короткий — оставь как есть.\n"
    "- Ответ — только переписанный запрос, без пояснений и кавычек.\n\n"
    "Запрос:\n{query}\n\n"
)

query_template = (
    "Инструкции:\n"
    "- Отвечай только на том же языке что и запрос.\n"
    "- Отвечай по делу и только по предоставленному контексту.\n"
    "- Если информации недостаточно — скажи об этом явно.\n"
    "- Приводи короткие цитаты в кавычках при необходимости.\n\n"
    "Контекст:\n{context_str}\n\n"
    "Вопрос пользователя:\n{query_str}\n\n"
)

# Tunables
TOP_K = int(os.getenv("TOP_K", 10))
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"


# ---------------------------- Conversational Memory ----------------------------

# Global memory for chat. For multi-user, key by session/user id.
MEMORY: Memory | None = None
AGENT: FunctionAgent | None = None


def _ensure_memory() -> Memory | None:
    """Lazy-init LlamaIndex Memory for conversational context."""
    global MEMORY
    if MEMORY is not None:
        return MEMORY
    try:
        MEMORY = Memory.from_defaults(
            session_id="chat_default",
            token_limit=20000,
            chat_history_token_ratio=0.7,
            token_flush_size=2000,
            insert_method="user",
        )
    except Exception:
        MEMORY = None
    return MEMORY


def reset_memory() -> None:
    """Reset global chat memory (best-effort sync wrapper)."""
    mem = _ensure_memory()
    if mem is None:
        return
    import asyncio

    async def _do():
        try:
            await mem.areset()
        except Exception:
            pass

    try:
        loop = asyncio.get_running_loop()
        # If we're in an event loop (e.g., async context), schedule without blocking
        loop.create_task(_do())
    except RuntimeError:
        # No running loop; run synchronously
        try:
            asyncio.run(_do())
        except Exception:
            pass


def configure_settings() -> None:
    """Init global Settings once (embedder, LLM, parser, transformations)."""
    configure_settings_gemini()
    # configure_settings_local()
    Settings.node_parser = MarkdownNodeParser()


def configure_settings_gemini() -> None:
    """Init global Settings once (embedder, LLM, parser, transformations)."""
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-001",
        embed_batch_size=80,
        retry_min_seconds=60,
        retry_max_seconds=120,
        embedding_config=EmbedContentConfig(
            output_dimensionality=768,
            task_type="RETRIEVAL_DOCUMENT",
        )
    )
    Settings.llm = GoogleGenAI(model="gemini-2.5-flash", temperature=0.1)
    Settings.ranker = LLMRerank()


def configure_settings_local() -> None:
    """Init global Settings once (embedder, LLM, parser, transformations)."""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_length=768,
        show_progress_bar=True
    )
    Settings.ranker = SentenceTransformerRerank(
        model="DiTy/cross-encoder-russian-msmarco", top_n=max(10, TOP_K)
    )  # "cross-encoder/ms-marco-MiniLM-L-2-v2"
    Settings.llm = GoogleGenAI(model="gemini-2.5-flash", temperature=0.1)


def _build_sources(response) -> List[Dict[str, Any]]:
    """Build a list of source documents from the response."""
    sources: List[Dict[str, Any]] = []
    for sn in getattr(response, "source_nodes", []) or []:
        meta = sn.node.metadata or {}
        try:
            score_val = float(sn.score) if getattr(
                sn, "score", None) is not None else None
        except Exception:
            score_val = None
        sources.append(
            {
                "score": score_val,
                "file_name": meta.get("file_name"),
                "text": sn.node.get_content(metadata_mode=MetadataMode.NONE),
            }
        )
        sources.sort(key=lambda x: x["score"] or 0, reverse=True)
    return sources


def _resp_text(res: object) -> str:
    """Extract text from common LlamaIndex agent/LLM responses."""
    if res is None:
        return ""
    if isinstance(res, str):
        return res
    for attr in ("output", "text", "message", "response"):
        val = getattr(res, attr, None)
        if isinstance(val, str):
            return val
        if getattr(val, "text", None):
            return str(getattr(val, "text"))
    return str(res)


async def _run_rag_workflow(query: str, file_name: str | None) -> Dict[str, Any]:
    """Run the existing RAG workflow and return the payload dict."""
    wf = RagWorkflow()
    try:
        result: Dict[str, Any] = await wf.run(input={"query": query, "file_name": file_name})
    except Exception as e:
        log.exception("Workflow failed, returning empty result: %s", e)
        result = {"answer": "", "sources": []}
    return result


def _ensure_agent() -> FunctionAgent | None:
    """Initialize and cache a FunctionAgent with a kb_search tool and memory."""
    global AGENT
    if AGENT is not None:
        return AGENT

    async def _kb_search_tool(query: str, file_name: str | None = None) -> str:
        """Tool: run the RAG pipeline and return strict JSON string."""
        payload = await _run_rag_workflow(query, file_name)
        return json.dumps(payload, ensure_ascii=False)

    try:
        tools = [
            FunctionTool.from_defaults(
                fn=_kb_search_tool,
                name="kb_search",
                description=(
                    "Искать ответ по базе знаний. Обязательно вызывать для каждого запроса. "
                    "Параметры: query (строка), file_name (опционально). Возвращает JSON со строкой 'answer' и списком 'sources'."
                ),
            )
        ]
    except Exception:
        tools = []

    system_prompt = (
        "Ты чат-ассистент по базе знаний. Всегда вызывай инструмент kb_search с текущим вопросом "
        "(и file_name если задан) и возвращай строго JSON, КОТОРЫЙ ВЕРНУЛ kb_search, без изменений."
    )

    try:
        AGENT = FunctionAgent.from_tools(
            tools=tools,
            llm=Settings.llm,
            memory=_ensure_memory(),
            verbose=False,
            system_prompt=system_prompt,
        )
    except Exception:
        AGENT = None
    return AGENT


# ---------------------------- Workflow Events ----------------------------


class QueryEvent(Event):
    query: str
    file_name: str | None


class ReformulatedRagQueryEvent(Event):
    rag_query: str
    query: str
    file_name: str | None


class RetrievedEvent(Event):
    query: str
    file_name: str | None
    nodes: List[Any]


class RerankEvent(Event):
    query: str
    file_name: str | None
    nodes: list[Any]


class SynthEvent(Event):
    query: str
    file_name: str | None
    response: Any
    nodes: List[Any]


class RagWorkflow(Workflow):
    """RAG pipeline implemented via llama_index.core.workflow."""

    def __init__(self, timeout: int = 600) -> None:
        super().__init__(timeout=timeout)

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> QueryEvent | StopEvent:
        """Entry point: read inputs from StartEvent and kick off retrieval."""
        data = getattr(ev, "input", None) or {}
        query = (data.get("query") or "").strip()
        file_name = (data.get("file_name") or None)
        if not query:
            # immediately stop with empty result
            return StopEvent(result={"answer": "", "sources": []})
        _ = get_index()
        return QueryEvent(query=query, file_name=file_name)

    @step
    async def reformulateRagQuery(self, ctx: Context, ev: QueryEvent) -> ReformulatedRagQueryEvent | StopEvent:
        """Transform the query for RAG using LLM to keep only the essence."""
        rag_query = ev.query
        try:
            prompt = PromptTemplate(
                reformulate_template).format(query=ev.query)
            resp = await Settings.llm.acomplete(prompt)
            log.debug(
                "RAG query reformulation done.", query=ev.query, rag_query=resp.text)
            candidate = (getattr(resp, "text", None) or "").strip()
            if candidate:
                # remove surrounding quotes if any
                if candidate.startswith(("'", '"')) and candidate.endswith(("'", '"')) and len(candidate) > 1:
                    candidate = candidate[1:-1].strip()
                rag_query = candidate or ev.query
        except Exception as e:
            log.warning("Query transform failed, using original", error=e)
        return ReformulatedRagQueryEvent(query=ev.query, rag_query=rag_query, file_name=ev.file_name)

    @step
    async def retrieve(self, ctx: Context, ev: ReformulatedRagQueryEvent) -> RetrievedEvent:
        """Retrieve candidate nodes using vector search with optional filters."""
        idx = get_index()
        if ev.file_name:
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="file_name", value=ev.file_name, operator=FilterOperator.EQ),
                ]
            )
            retriever = idx.as_retriever(
                similarity_top_k=TOP_K, filters=filters)
        else:
            retriever = idx.as_retriever(similarity_top_k=TOP_K)
        nodes = await retriever.aretrieve(ev.rag_query)
        return RetrievedEvent(query=ev.query, nodes=nodes, file_name=ev.file_name)

    @step
    async def rerank(self, ctx: Context, ev: RetrievedEvent) -> RerankEvent:
        """Rerank the nodes."""
        if not RERANK_ENABLED or not ev.nodes or not Settings.ranker:
            log.debug("Reranking skipped.")
            return RerankEvent(nodes=ev.nodes, query=ev.query, file_name=ev.file_name)
        new_nodes = Settings.ranker.postprocess_nodes(
            ev.nodes, query_str=ev.query)
        log.debug("Reranking done.", query=ev.query, input_count=len(
            ev.nodes), output_count=len(new_nodes))
        return RerankEvent(nodes=new_nodes, query=ev.query, file_name=ev.file_name)

    @step
    async def synthesize(self, ctx: Context, ev: RerankEvent) -> SynthEvent:
        """Generate final answer from nodes using a response synthesizer."""
        text_qa_template = PromptTemplate(query_template)
        synthesizer = get_response_synthesizer(
            llm=Settings.llm,
            response_mode="compact",
            use_async=True,
            text_qa_template=text_qa_template,
        )
        response = await synthesizer.asynthesize(ev.query, ev.nodes)
        return SynthEvent(query=ev.query, file_name=ev.file_name, response=response, nodes=ev.nodes)

    @step
    def finalize(self, ctx: Context, ev: SynthEvent) -> StopEvent:
        """Build payload and stop the workflow."""
        sources = _build_sources(ev.response)
        payload: Dict[str, Any] = {
            "answer": str(ev.response), "sources": sources}
        return StopEvent(result=payload)


async def search_documents(
    query: str,
    *, file_name: str | None = None,
) -> str:
    """Search and return JSON answer + sources using a Workflow pipeline."""
    t0 = time.time()
    agent = _ensure_agent()
    result: Dict[str, Any]
    if agent is not None:
        try:
            # Let the agent manage memory; pass file_name hint inline if present
            user_query = query if not file_name else f"{query}\n\n[file_name={file_name}]"
            resp = await agent.run(user_query, memory=_ensure_memory())
            txt = _resp_text(resp)
            data = json.loads((txt or "").strip() or "{}")
            if isinstance(data, dict) and "answer" in data and "sources" in data:
                result = data
            else:
                # Fallback to direct RAG if agent returned unexpected format
                result = await _run_rag_workflow(query, file_name)
        except Exception as e:
            log.warning("agent_run_failed", error=e)
            result = await _run_rag_workflow(query, file_name)
    else:
        result = await _run_rag_workflow(query, file_name)

    latency = int(time.time() - t0)
    log.debug("Workflow done.", query=query, latency=latency, answer=result.get("answer", "")[:50],
              sources=len(result.get("sources", "")))
    return json.dumps(result, ensure_ascii=False)


# ----------------------------- Warmup helpers -----------------------------
def start(
    *, ensure_index: bool = True
) -> None:
    """Warm heavy parts to reduce first-request latency."""
    ensure_dirs()
    configure_settings()
    _ensure_memory()
    _ensure_agent()
    if ensure_index:
        try:
            _ = get_index()
        except Exception as e:
            log.warning("Warmup get_index failed: %s", e)
    log.debug("Warmup done (index=%s)", ensure_index)
