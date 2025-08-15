import os
import sys
import time
import asyncio
import json
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

# Load environment variables from .env file
load_dotenv() 

# Settings control global defaults
# Speed/quality defaults (tunable via env if needed)
TOP_K = int(os.getenv("TOP_K", 15))
USE_FUSION = os.getenv("USE_FUSION", "true").lower() == "true"
USE_HYDE = os.getenv("USE_HYDE", "true").lower() == "true"
USE_RERANK = os.getenv("USE_RERANK", "true").lower() == "true"

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
    path="./chroma_db",
    settings=ChromaSettings(anonymized_telemetry=False)
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
        d.metadata.setdefault("file_name", d.metadata.get("file_name") or d.metadata.get("filename") or "unknown")
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
        print("Failed to load persisted storage context, rebuilding index...")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

_bm25_retriever: Optional[BM25Retriever] = None
_bm25_nodes_cache: Optional[List[Any]] = None


def _load_nodes_for_bm25() -> List[Any]:
    # Build nodes from documents via the configured node parser
    documents = SimpleDirectoryReader("data").load_data()
    for d in documents:
        d.metadata.setdefault("file_name", d.metadata.get("file_name") or d.metadata.get("filename") or "unknown")
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
        meta = getattr(n, "metadata", {}) or getattr(getattr(n, "node", None), "metadata", {}) or {}
        ok = True
        for f in filters.filters:
            if meta.get(f.key) != f.value:
                ok = False
                break
        if ok:
            result.append(n)
    return result


def _get_bm25_retriever(filters: Optional[MetadataFilters]) -> BM25Retriever:
    global _bm25_retriever, _bm25_nodes_cache
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

node_postprocessors = []

# Cross-encoder rerank first for precision
def _compute_rerank_top_n(top_k: int, use_rerank: bool) -> Optional[int]:
    if not use_rerank:
        return None
    return min(12, max(1, top_k))


if USE_RERANK:
    try:
        top_n = _compute_rerank_top_n(TOP_K, USE_RERANK)
        if top_n and SentenceTransformerRerank is not None:
            node_postprocessors.append(
                SentenceTransformerRerank(
                    top_n=top_n, model="cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
            )
    except Exception:
        print("Failed to compute rerank top N")

# Safe wrapper for neighbor windowing: if docstore misses nodes, skip gracefully
class SafeAutoPrevNext(BaseNodePostprocessor):
    def __init__(self, underlying: BaseNodePostprocessor) -> None:
        super().__init__()
        self._underlying = underlying

    def _postprocess_nodes(self, nodes, query_bundle=None):  # type: ignore[override]
        try:
            return self._underlying.postprocess_nodes(nodes, query_bundle=query_bundle)
        except Exception:
            print("Failed to postprocess nodes")
            return nodes

# Re-introduce sentence windowing without embedding smear
try:
    node_postprocessors.append(
        SafeAutoPrevNext(AutoPrevNextNodePostprocessor(docstore=index.docstore))
    )
except Exception:
    print("Failed to initialize AutoPrevNext postprocessor")

# Light or no similarity cutoff to avoid over-pruning
node_postprocessors.append(SimilarityPostprocessor(similarity_cutoff=0.05))

# Place salient chunks near the beginning/end for long prompts
node_postprocessors.append(LongContextReorder())


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
    *, file_name: Optional[str], section: Optional[str], lang: Optional[str], version: Optional[str]
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
            score_val = float(sn.score) if getattr(sn, "score", None) is not None else None
        except Exception:
            print("Failed to extract score from source node")
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
    filters = _make_filters(file_name=file_name, section=section, lang=lang, version=version)

    engine = _make_engine_with_filters(filters)
    t0 = time.time()
    response = await engine.aquery(query)
    t_retrieve_llm = time.time() - t0
    # Fallback: if empty, retry with dense-only retriever without fusion
    if not str(response).strip() or not getattr(response, "source_nodes", None):
        response = await _make_fallback_engine(filters).aquery(query)

    # Build citation structure
    sources = _build_sources(response)

    print(f"Found {len(sources)} sources for query: {query}")
    print(f"Latency: {t_retrieve_llm:.2f}s (retrieval+LLM)")
    print(f"Response: {response}")
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
            "Отвечай кратко, опираясь только на найденные фрагменты. Если результатов нет — скажи, что не знаешь. "
            "В ответе используй точные цитаты и добавляй ссылки на источники из результатов инструмента."
        ),
    )


# Now we can ask questions about the documents or do calculations
async def main():
    # Allow passing a custom query via CLI; default to a sensible demo query
    query = sys.argv[1] if len(sys.argv) > 1 else "Что такое предмет разработки?"

    print(f"Running with query: {query}")
    if AGENT_ENABLED and agent is not None:
        try:
            agent_resp = await agent.run(query, max_iterations=1)
            text = str(agent_resp)
            print(text)
            return
        except Exception:
            print("Agent failed, falling back to direct search...")

    # Default path: call the retrieval tool directly for a fast answer with citations
    result_json = await search_documents(query)
    print(result_json)


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
