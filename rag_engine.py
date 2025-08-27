import os
import time
import json
from typing import Any, Dict, List

import logging
logger = logging.getLogger(__name__)

from storage import get_index
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.ollama import Ollama
from google.genai.types import EmbedContentConfig
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.workflow import (
    Workflow,
    Event,
    step,
    Context,
    StartEvent,
    StopEvent,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import MetadataMode
from llama_index.core import Settings
from llama_index.core.postprocessor.llm_rerank import LLMRerank

from dotenv import load_dotenv
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
    #configure_settings_gemini()
    configure_settings_ollama()
    Settings.transformations = [MarkdownNodeParser()]

def configure_settings_gemini() -> None:
    """Init global Settings once (embedder, LLM, parser, transformations)."""
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-001",          # новое имя модели
        embed_batch_size=100,
        embedding_config=EmbedContentConfig(
            output_dimensionality=768,
            task_type="RETRIEVAL_DOCUMENT"
        )
    )
    Settings.llm = GoogleGenAI(model="gemini-2.5-flash", temperature=0.1)

def configure_settings_ollama() -> None:
    """Init global Settings once (embedder, LLM, parser, transformations)."""
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-001",          # новое имя модели
        embed_batch_size=100,
        embedding_config=EmbedContentConfig(
            output_dimensionality=768,
            task_type="RETRIEVAL_DOCUMENT"
        )
    )
    Settings.llm = Ollama(model="qwen3:4b", temperature=0.1) # thinking=False, 

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


class ReformulatedRagQueryEvent(Event):
    rag_query: str
    query: str


class RetrievedEvent(Event):
    query: str
    nodes: List[Any]


class RerankEvent(Event):
    query: str
    nodes: list[Any]


class SynthEvent(Event):
    query: str
    response: Any
    nodes: List[Any]


class RagWorkflow(Workflow):
    """RAG pipeline implemented via llama_index.core.workflow."""

    def __init__(self, timeout: int = 120) -> None:
        super().__init__(timeout=timeout)

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> QueryEvent | StopEvent:
        """Entry point: read inputs from StartEvent and kick off retrieval."""
        data = getattr(ev, "input", None) or {}
        query = (data.get("query") or "").strip()
        if not query:
            # immediately stop with empty result
            return StopEvent(result={"answer": "", "sources": []})
        # Ensure settings/index are ready before retrieval
        configure_settings()
        _ = get_index()
        return QueryEvent(query=query)

    @step
    async def reformulateRagQuery(self, ctx: Context, ev: QueryEvent) -> ReformulatedRagQueryEvent | StopEvent:
        """Transform the query for RAG using LLM to keep only the essence."""
        rag_query = ev.query
        try:
            prompt = PromptTemplate(
                reformulate_template).format(query=ev.query)
            resp = await Settings.llm.acomplete(prompt)
            logger.debug(
                "RAG query transform | input=%s | output=%s", ev.query, resp)
            print(f"RAG query reformulated | input={ev.query} | output={resp}")
            candidate = (getattr(resp, "text", None) or "").strip()
            if candidate:
                # remove surrounding quotes if any
                if candidate.startswith(("'", '"')) and candidate.endswith(("'", '"')) and len(candidate) > 1:
                    candidate = candidate[1:-1].strip()
                rag_query = candidate or ev.query
        except Exception as e:
            logger.warning("Query transform failed, using original: %s", e)
        return ReformulatedRagQueryEvent(query=ev.query, rag_query=rag_query)

    @step
    async def retrieve(self, ctx: Context, ev: ReformulatedRagQueryEvent) -> RetrievedEvent:
        """Retrieve candidate nodes using vector search with optional filters."""
        idx = get_index()
        retriever = idx.as_retriever(similarity_top_k=TOP_K)
        nodes = await retriever.aretrieve(ev.rag_query)
        return RetrievedEvent(query=ev.query, nodes=nodes)

    @step
    async def rerank(self, ctx: Context, ev: RetrievedEvent) -> RerankEvent:
        """Rerank the nodes."""
        if not RERANK_ENABLED or not ev.nodes:
            return RerankEvent(nodes=ev.nodes, query=ev.query)
        ranker = LLMRerank()
        new_nodes = ranker.postprocess_nodes(ev.nodes, query_str=ev.query)
        return RerankEvent(nodes=new_nodes, query=ev.query)

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
        return SynthEvent(query=ev.query, response=response, nodes=ev.nodes)

    @step
    def finalize(self, ctx: Context, ev: SynthEvent) -> StopEvent:
        """Build payload and stop the workflow."""
        sources = _build_sources(ev.response)
        payload: Dict[str, Any] = {
            "answer": str(ev.response), "sources": sources}
        return StopEvent(result=payload)


async def search_documents(
    query: str,
) -> str:
    """Search and return JSON answer + sources using a Workflow pipeline."""

    t0 = time.time()
    # Run the workflow end-to-end
    wf = RagWorkflow()
    try:
        result: Dict[str, Any] = await wf.run(input={"query": query})
    except Exception as e:
        logger.exception("Workflow failed, returning empty result: %s", e)
        result = {"answer": "", "sources": []}

    latency = time.time() - t0
    logger.debug("Workflow done | %ds | sources=%d", int(
        latency), len(result.get("sources", []) or []))
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
            logger.warning("Warmup get_index failed: %s", e)
    logger.debug("Warmup done (index=%s)", ensure_index)
