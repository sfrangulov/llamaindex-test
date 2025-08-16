import os
import sys
import time
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine
from llama_index.core.schema import MetadataMode
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.postprocessor import (
    AutoPrevNextNodePostprocessor,
    LongContextReorder,
    SimilarityPostprocessor,
    SentenceTransformerRerank,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

import chromadb
from chromadb.config import Settings as ChromaSettings


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

# Load environment variables from .env file
load_dotenv()

# Settings control global defaults
# Speed/quality defaults (tunable via env if needed)
TOP_K = int(os.getenv("TOP_K", 15))
USE_FUSION = os.getenv("USE_FUSION", "false").lower() == "true"
USE_HYDE = os.getenv("USE_HYDE", "true").lower() == "true"
USE_RERANK = os.getenv("USE_RERANK", "true").lower() == "true"
PARALLEL_HYDE = os.getenv("PARALLEL_HYDE", "true").lower() == "true"
RESPONSE_MODE = os.getenv("RESPONSE_MODE", "compact")

Settings.embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-004",
    embed_batch_size=256,
)
Settings.llm = GoogleGenAI(
    model="gemini-2.5-flash",
    temperature=0.1,  # deterministic to reduce hallucinations
)
Settings.node_parser = SentenceSplitter(chunk_size=700, chunk_overlap=120)


# Create the document index
db = chromadb.PersistentClient(
    path="./chroma_db", settings=ChromaSettings(anonymized_telemetry=False)
)
PERSIST_DIR = "./storage"
chroma_collection = db.get_or_create_collection("test")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


def infer_language(text: str) -> str:
    # Simple heuristic: Cyrillic -> ru, else en
    return "ru" if any("\u0400" <= ch <= "\u04FF" for ch in text) else "en"


if chroma_collection.count() == 0:
    documents = SimpleDirectoryReader("data").load_data()
    # enrich metadata for filtering
    for d in documents:
        d.metadata.setdefault(
            "file_name",
            d.metadata.get("file_name") or d.metadata.get("filename") or "unknown",
        )
        d.metadata.setdefault("section", "unknown")
        d.metadata.setdefault("version", "v1")
        if "lang" not in d.metadata:
            d.metadata["lang"] = infer_language(d.text or "")

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    # Persist docstore / index metadata for neighbor-window postprocessors
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # Load persisted docstore if available so neighbor-window can resolve nodes
    try:
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=PERSIST_DIR
        )
    except Exception:
        logging.warning(
            "Failed to load persisted storage context, rebuilding index..."
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )

_bm25_nodes_cache: Optional[List[Any]] = None


def _load_nodes_for_bm25() -> List[Any]:
    # Build nodes from documents via the configured node parser
    documents = SimpleDirectoryReader("data").load_data()
    for d in documents:
        d.metadata.setdefault(
            "file_name",
            d.metadata.get("file_name") or d.metadata.get("filename") or "unknown",
        )
        d.metadata.setdefault("section", "unknown")
        d.metadata.setdefault("version", "v1")
        if "lang" not in d.metadata:
            d.metadata["lang"] = infer_language(d.text or "")
    return Settings.node_parser.get_nodes_from_documents(documents)


def _filter_nodes(nodes: List[Any], filters: Optional[MetadataFilters]) -> List[Any]:
    if not filters or not getattr(filters, "filters", None):
        return nodes
    result = []
    for n in nodes:
        meta = (
            getattr(n, "metadata", {})
            or getattr(getattr(n, "node", None), "metadata", {})
            or {}
        )
        ok = True
        for f in filters.filters:
            if meta.get(f.key) != f.value:
                ok = False
                break
        if ok:
            result.append(n)
    return result


def _get_bm25_retriever(filters: Optional[MetadataFilters]) -> BM25Retriever:
    global _bm25_nodes_cache
    if _bm25_nodes_cache is None:
        _bm25_nodes_cache = _load_nodes_for_bm25()
    nodes = _filter_nodes(_bm25_nodes_cache, filters)
    # Build a new retriever if filters subset changes size notably; simple approach: build per call
    return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=TOP_K)


def build_retriever(filters: Optional[MetadataFilters] = None):
    # Dense-only retriever
    dense = index.as_retriever(
        similarity_top_k=TOP_K,
        vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
        filters=filters,
    )
    # Real hybrid: BM25 (sparse) + dense with reciprocal rerank
    if USE_FUSION:
        bm25 = _get_bm25_retriever(filters)
        return QueryFusionRetriever(
            retrievers=[dense, bm25],
            similarity_top_k=TOP_K,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=True,
        )
    return dense


retriever = build_retriever()


def _compute_rerank_top_n(top_k: int, use_rerank: bool) -> Optional[int]:
    """Compute cross-encoder rerank top_n based on TOP_K and flag."""
    if not use_rerank:
        return None
    return min(12, max(1, top_k))


# Safe wrapper for neighbor windowing: if docstore misses nodes, skip gracefully
class SafeAutoPrevNext(BaseNodePostprocessor):
    def __init__(self, underlying: BaseNodePostprocessor) -> None:
        super().__init__()
        self._underlying = underlying

    def _postprocess_nodes(self, nodes, query_bundle=None):  # type: ignore[override]
        try:
            return self._underlying.postprocess_nodes(nodes, query_bundle=query_bundle)
        except Exception as e:
            logging.error("Failed to postprocess nodes: %s", e)
            return nodes


def _build_node_postprocessors() -> List[BaseNodePostprocessor]:
    node_postprocessors: List[BaseNodePostprocessor] = []
    # Cross-encoder rerank first for precision
    if USE_RERANK:
        try:
            top_n = _compute_rerank_top_n(TOP_K, USE_RERANK)
            if top_n and SentenceTransformerRerank is not None:
                node_postprocessors.append(
                    SentenceTransformerRerank(
                        top_n=top_n, model="cross-encoder/ms-marco-MiniLM-L-6-v2"
                    )
                )
        except Exception as e:
            logging.exception("Failed to compute rerank top N: %s", e)

    # Re-introduce sentence windowing without embedding smear
    try:
        node_postprocessors.append(
            SafeAutoPrevNext(AutoPrevNextNodePostprocessor(docstore=index.docstore))
        )
    except Exception as e:
        logging.exception("Failed to initialize AutoPrevNext postprocessor: %s", e)

    # Light or no similarity cutoff to avoid over-pruning
    node_postprocessors.append(SimilarityPostprocessor(similarity_cutoff=0.05))

    # Place salient chunks near the beginning/end for long prompts
    node_postprocessors.append(LongContextReorder())

    return node_postprocessors


node_postprocessors = _build_node_postprocessors()


base_query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=node_postprocessors,
)

# Query transforms: HyDE (+ optional user paraphrase in future)
hyde = HyDEQueryTransform(include_original=True)

query_engine = (
    TransformQueryEngine(base_query_engine, hyde) if USE_HYDE else base_query_engine
)


def _make_engine_with_filters(filters: Optional[List[ExactMatchFilter]]):
    if not filters:
        return query_engine
    filtered_retriever = build_retriever(MetadataFilters(filters=filters))
    filtered_engine = RetrieverQueryEngine(
        retriever=filtered_retriever,
        node_postprocessors=node_postprocessors,
    )
    if USE_HYDE and hyde is not None:
        return TransformQueryEngine(filtered_engine, hyde)
    return filtered_engine


def _make_filters(
    *,
    file_name: Optional[str],
    section: Optional[str],
    lang: Optional[str],
    version: Optional[str],
) -> List[ExactMatchFilter]:
    filters: List[ExactMatchFilter] = []
    if file_name:
        filters.append(ExactMatchFilter(key="file_name", value=file_name))
    if section:
        filters.append(ExactMatchFilter(key="section", value=section))
    if lang:
        filters.append(ExactMatchFilter(key="lang", value=lang))
    if version:
        filters.append(ExactMatchFilter(key="version", value=version))
    return filters


def _build_sources(response) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for sn in getattr(response, "source_nodes", []) or []:
        meta = sn.node.metadata or {}
        try:
            score_val = (
                float(sn.score) if getattr(sn, "score", None) is not None else None
            )
        except Exception as e:
            print(f"Failed to extract score from source node: {e}")
            score_val = None
        sources.append(
            {
                "score": score_val,
                "file_name": meta.get("file_name") or meta.get("filename"),
                "section": meta.get("section"),
                "lang": meta.get("lang"),
                "version": meta.get("version"),
                "window": meta.get("window"),
                "text": sn.node.get_content(metadata_mode=MetadataMode.NONE),
            }
        )
    return sources


def _make_fallback_engine(filters: Optional[List[ExactMatchFilter]]):
    fallback_retriever = index.as_retriever(
        similarity_top_k=TOP_K,
        vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
        filters=MetadataFilters(filters=filters) if filters else None,
    )
    base = RetrieverQueryEngine(
        retriever=fallback_retriever,
        node_postprocessors=node_postprocessors,
    )
    if USE_HYDE and hyde is not None:
        return TransformQueryEngine(base, hyde)
    return base


# --- Parallel HyDE utilities -------------------------------------------------
async def _generate_hyde_text(user_query: str) -> str:
    """Generate a short hypothetical answer for HyDE using the configured LLM.

    This approximates HyDE behavior without blocking other steps.
    """
    try:
        from llama_index.core.prompts import PromptTemplate as prompt_template_cls
    except Exception:
        prompt_template_cls = None  # type: ignore
    template_text = (
        "Сгенерируй краткий (80-120 слов) абзац, который мог бы содержаться в документации, "
        "отвечающий на запрос: \n{query}\n"
        "Пиши информативно и по делу, без вступлений."
    )
    try:
        if prompt_template_cls is not None:
            tmpl = prompt_template_cls(template_text)
            # Use async LLM call if available
            return await Settings.llm.apredict(tmpl, prompt_args={"query": user_query})  # type: ignore[attr-defined]
        else:
            # Fallback: pass as a formatted string
            return await Settings.llm.apredict(template_text.format(query=user_query))  # type: ignore[attr-defined]
    except AttributeError:
        # Fallback to sync if async isn't available
        if prompt_template_cls is not None:
            tmpl = prompt_template_cls(template_text)
            return Settings.llm.predict(tmpl, prompt_args={"query": user_query})
        return Settings.llm.predict(template_text.format(query=user_query))


def _merge_nodes_by_score(nodes_lists: List[List[Any]], limit: int) -> List[Any]:
    """Merge NodeWithScore lists, deduplicate by node_id, keep best score, return top-N."""
    best: Dict[str, Any] = {}
    for lst in nodes_lists:
        for n in lst or []:
            node_obj = getattr(n, "node", None)
            node_id = getattr(node_obj, "node_id", None) or getattr(n, "node_id", None)
            if node_id is None:
                # If cannot determine id, append as-is with a synthetic key
                node_id = f"_idx_{id(n)}"
            prev = best.get(node_id)
            if prev is None or (getattr(n, "score", 0.0) or 0.0) > (getattr(prev, "score", 0.0) or 0.0):
                best[node_id] = n
    # Sort by score desc if available
    merged = list(best.values())
    merged.sort(key=lambda x: (getattr(x, "score", 0.0) or 0.0), reverse=True)
    return merged[:limit]


async def search_documents(
    query: str,
    *,
    file_name: Optional[str] = None,
    section: Optional[str] = None,
    lang: Optional[str] = None,
    version: Optional[str] = None,
) -> str:
    """Search documents and return answer with citations.

    Optional metadata filters can be provided to narrow the search.
    In production, prefer to return a structured object; here we JSON-encode for the agent.
    """
    filters = _make_filters(
        file_name=file_name, section=section, lang=lang, version=version
    )

    engine = _make_engine_with_filters(filters)
    t0 = time.time()

    # Parallel HyDE path: overlap HyDE generation with baseline retrieval, then
    # combine results and do a single final synthesis. Falls back to regular path on error.
    if USE_HYDE and PARALLEL_HYDE:
        try:
            # 1) Kick off baseline retrieve and hyde text generation concurrently
            base_retriever = build_retriever(MetadataFilters(filters=filters)) if filters else retriever
            # We use raw retriever to get nodes without invoking LLM synthesis yet
            base_task = asyncio.create_task(base_retriever.aretrieve(query))  # type: ignore[attr-defined]
            hyde_task = asyncio.create_task(_generate_hyde_text(query))

            base_nodes, hyde_text = await asyncio.gather(base_task, hyde_task)

            # 2) Do a second retrieval for hyde text
            hyde_nodes = await base_retriever.aretrieve(hyde_text)  # type: ignore[attr-defined]

            # 3) Merge nodes and synthesize once
            merged_nodes = _merge_nodes_by_score([base_nodes, hyde_nodes], TOP_K)
            synthesizer = get_response_synthesizer(
                llm=Settings.llm,
                response_mode=RESPONSE_MODE,
                use_async=True,
            )
            response = await synthesizer.asynthesize(
                query,
                merged_nodes,
            )
        except Exception as e:
            logging.exception("Parallel HyDE failed, falling back to engine.aquery: %s", e)
            response = await engine.aquery(query)
    else:
        response = await engine.aquery(query)

    t_retrieve_llm = time.time() - t0
    # Fallback: if empty, retry with dense-only retriever without fusion
    if not str(response).strip() or not getattr(response, "source_nodes", None):
        response = await _make_fallback_engine(filters).aquery(query)

    # Build citation structure
    sources = _build_sources(response)

    logging.info("Found %d sources for query: %s", len(sources), query)
    logging.info("Latency: %.2fs (retrieval+LLM)", t_retrieve_llm)
    logging.debug("Response: %s", response)
    payload: Dict[str, Any] = {
        "answer": str(response),
        "sources": sources,
    }
    return json.dumps(payload, ensure_ascii=False)


# Optional agent wrapper (disabled by default to reduce latency)
AGENT_ENABLED = os.getenv("AGENT_ENABLED", "false").lower() == "true"
agent = None
if AGENT_ENABLED:
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


# Now we can ask questions about the documents or do calculations
async def main():
    # Allow passing a custom query via CLI; default to a sensible demo query
    query = sys.argv[1] if len(sys.argv) > 1 else "Что такое предмет разработки?"

    logging.info("Running with query: %s", query)
    if AGENT_ENABLED and agent is not None:
        try:
            agent_resp = await agent.run(query, max_iterations=1)
            text = str(agent_resp)
            print(text)
            return
        except Exception as e:
            logging.exception("Agent failed, falling back to direct search... %s", e)

    # Default path: call the retrieval tool directly for a fast answer with citations
    result_json = await search_documents(query)
    print(result_json)


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
