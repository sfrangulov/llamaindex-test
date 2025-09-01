from dotenv import load_dotenv
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core import Settings
from llama_index.core.schema import MetadataMode
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.prompts import PromptTemplate
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
    # Settings.llm = HuggingFaceLLM(
    #    model_name="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit", )
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
        # Ensure settings/index are ready before retrieval
        configure_settings()
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
                    MetadataFilter(key="file_name", value=ev.file_name, operator=FilterOperator.EQ),
                ]
            )
            retriever = idx.as_retriever(similarity_top_k=TOP_K, filters=filters)
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
    # Run the workflow end-to-end
    wf = RagWorkflow()
    try:
        result: Dict[str, Any] = await wf.run(input={"query": query, "file_name": file_name})
    except Exception as e:
        log.exception("Workflow failed, returning empty result: %s", e)
        result = {"answer": "", "sources": []}

    latency = int(time.time() - t0)
    log.debug("Workflow done.", query=query, latency=latency, answer=result.get("answer", "")[:50],
              sources=len(result.get("sources", "")))
    return json.dumps(result, ensure_ascii=False)


# ----------------------------- Warmup helpers -----------------------------
def warmup(
    *, ensure_index: bool = True
) -> None:
    """Warm heavy parts to reduce first-request latency."""
    configure_settings()
    if ensure_index:
        try:
            _ = get_index()
        except Exception as e:
            log.warning("Warmup get_index failed: %s", e)
    log.debug("Warmup done (index=%s)", ensure_index)
