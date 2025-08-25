import os
import time
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# LlamaIndex imports kept local to functions to avoid heavy import side-effects during tests
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
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
from llama_index.core.retrievers import QueryFusionRetriever, BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

import chromadb
from chromadb.config import Settings as ChromaSettings


# ---------------------------- Configuration ----------------------------
load_dotenv()

TOP_K = int(os.getenv("TOP_K", 15))
USE_FUSION = os.getenv("USE_FUSION", "true").lower() == "true"
USE_HYDE = os.getenv("USE_HYDE", "true").lower() == "true"
USE_RERANK = os.getenv("USE_RERANK", "true").lower() == "true"
PARALLEL_HYDE = os.getenv("PARALLEL_HYDE", "true").lower() == "true"
RESPONSE_MODE = os.getenv("RESPONSE_MODE", "compact")

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "test")
PERSIST_DIR = os.getenv("PERSIST_DIR", "./storage")


_SETTINGS_CONFIGURED = False


def configure_settings() -> None:
    """Configure global LlamaIndex Settings lazily.

    Referenced from LlamaIndex docs to set embed model, llm, and parser.
    """
    global _SETTINGS_CONFIGURED
    if _SETTINGS_CONFIGURED:
        # Always refresh node parser for safety
        Settings.node_parser = SentenceSplitter(chunk_size=700, chunk_overlap=120)
        return

    # Set models explicitly to avoid default OpenAI resolution
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="text-embedding-004",
    )
    Settings.llm = GoogleGenAI(
        model="gemini-2.5-flash",
        temperature=0.1,
    )
    Settings.node_parser = SentenceSplitter(chunk_size=700, chunk_overlap=120)
    _SETTINGS_CONFIGURED = True


def infer_language(text: str) -> str:
    # Simple heuristic: Cyrillic -> ru, else en
    return "ru" if any("\u0400" <= ch <= "\u04FF" for ch in text) else "en"


# ---------------------------- Storage / Index ----------------------------
_db: Optional[chromadb.PersistentClient] = None
_vector_store: Optional[ChromaVectorStore] = None
_storage_context: Optional[StorageContext] = None
_index: Optional[VectorStoreIndex] = None
_bm25_nodes_cache: Optional[List[Any]] = None
_hyde: Optional[HyDEQueryTransform] = None


def _get_db() -> chromadb.PersistentClient:
    global _db
    if _db is None:
        _db = chromadb.PersistentClient(path=CHROMA_PATH, settings=ChromaSettings(anonymized_telemetry=False))
    return _db


def _get_vector_store() -> ChromaVectorStore:
    global _vector_store
    if _vector_store is None:
        collection = _get_db().get_or_create_collection(CHROMA_COLLECTION)
        _vector_store = ChromaVectorStore(chroma_collection=collection)
    return _vector_store


def _get_storage_context(load_persisted: bool = True) -> StorageContext:
    global _storage_context
    if _storage_context is not None:
        return _storage_context
    try:
        if load_persisted:
            _storage_context = StorageContext.from_defaults(
                vector_store=_get_vector_store(), persist_dir=PERSIST_DIR
            )
        else:
            raise RuntimeError("force rebuild")
    except Exception:
        logging.warning("Falling back to fresh StorageContext without persisted state")
        _storage_context = StorageContext.from_defaults(vector_store=_get_vector_store())
    return _storage_context


def _load_documents_with_metadata() -> List[Any]:
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
    return documents


def get_index() -> VectorStoreIndex:
    global _index
    if _index is not None:
        return _index

    configure_settings()
    collection = _get_db().get_or_create_collection(CHROMA_COLLECTION)
    if collection.count() == 0:
        # Build from documents
        documents = _load_documents_with_metadata()
        sc = _get_storage_context(load_persisted=False)
        _index = VectorStoreIndex.from_documents(documents, storage_context=sc)
        _index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        # Load from vector store (and persisted docstore if available)
        sc = _get_storage_context(load_persisted=True)
        _index = VectorStoreIndex.from_vector_store(_get_vector_store(), storage_context=sc)
    return _index


def _load_nodes_for_bm25() -> List[Any]:
    configure_settings()
    documents = _load_documents_with_metadata()
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
    return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=TOP_K)


def build_retriever(filters: Optional[MetadataFilters] = None) -> BaseRetriever:
    idx = get_index()
    dense = idx.as_retriever(
        similarity_top_k=TOP_K,
        vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
        filters=filters,
    )
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


def _compute_rerank_top_n(top_k: int, use_rerank: bool) -> Optional[int]:
    if not use_rerank:
        return None
    return min(12, max(1, top_k))


class SafeAutoPrevNext(BaseNodePostprocessor):
    def __init__(self, underlying: BaseNodePostprocessor) -> None:
        super().__init__()
        self._underlying = underlying

    def _postprocess_nodes(self, nodes, query_bundle=None):  # type: ignore[override]
        processed = []
        for n in nodes or []:
            try:
                out = self._underlying.postprocess_nodes([n], query_bundle=query_bundle)
                if out:
                    processed.extend(out)
                else:
                    processed.append(n)
            except Exception as e:
                node_obj = getattr(n, "node", n)
                node_id = getattr(node_obj, "node_id", "unknown")
                logging.warning("Skipping neighbor-window for node %s: %s", node_id, e)
                processed.append(n)
        return processed


def _build_node_postprocessors(docstore: Optional[Any] = None) -> List[BaseNodePostprocessor]:
    node_postprocessors: List[BaseNodePostprocessor] = []
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
            logging.exception("Failed to init reranker: %s", e)

    # Neighbor windowing if docstore is available; otherwise skip gracefully
    try:
        if docstore is None:
            # Try to get from index if already built; if not, skip
            if _index is not None and getattr(_index, "docstore", None) is not None:
                docstore = _index.docstore  # type: ignore[attr-defined]
        if docstore is not None:
            node_postprocessors.append(
                SafeAutoPrevNext(AutoPrevNextNodePostprocessor(docstore=docstore))
            )
    except Exception as e:
        logging.exception("Failed to initialize AutoPrevNext postprocessor: %s", e)

    node_postprocessors.append(SimilarityPostprocessor(similarity_cutoff=0.05))
    node_postprocessors.append(LongContextReorder())
    return node_postprocessors


def _get_hyde() -> HyDEQueryTransform:
    global _hyde
    if _hyde is None:
        configure_settings()
        _hyde = HyDEQueryTransform(include_original=True)
    return _hyde


def _make_engine_with_filters(filters: Optional[List[ExactMatchFilter]]):
    if not filters:
        # base engine lazily assembled
        retriever = build_retriever()
        return RetrieverQueryEngine(retriever=retriever, node_postprocessors=_build_node_postprocessors())
    filtered_retriever = build_retriever(MetadataFilters(filters=filters))
    engine = RetrieverQueryEngine(
        retriever=filtered_retriever,
        node_postprocessors=_build_node_postprocessors(),
    )
    if USE_HYDE:
        return TransformQueryEngine(engine, _get_hyde())
    return engine


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
    idx = get_index()
    fallback_retriever = idx.as_retriever(
        similarity_top_k=TOP_K,
        vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
        filters=MetadataFilters(filters=filters) if filters else None,
    )
    base = RetrieverQueryEngine(
        retriever=fallback_retriever,
        node_postprocessors=_build_node_postprocessors(),
    )
    if USE_HYDE:
        return TransformQueryEngine(base, _get_hyde())
    return base


async def _generate_hyde_text(user_query: str) -> str:
    configure_settings()
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
            return await Settings.llm.apredict(tmpl, prompt_args={"query": user_query})  # type: ignore[attr-defined]
        else:
            return await Settings.llm.apredict(template_text.format(query=user_query))  # type: ignore[attr-defined]
    except AttributeError:
        if prompt_template_cls is not None:
            tmpl = prompt_template_cls(template_text)
            return Settings.llm.predict(tmpl, prompt_args={"query": user_query})
        return Settings.llm.predict(template_text.format(query=user_query))


def _merge_nodes_by_score(nodes_lists: List[List[Any]], limit: int) -> List[Any]:
    best: Dict[str, Any] = {}
    for lst in nodes_lists:
        for n in lst or []:
            node_obj = getattr(n, "node", None)
            node_id = getattr(node_obj, "node_id", None) or getattr(n, "node_id", None)
            if node_id is None:
                node_id = f"_idx_{id(n)}"
            prev = best.get(node_id)
            if prev is None or (getattr(n, "score", 0.0) or 0.0) > (getattr(prev, "score", 0.0) or 0.0):
                best[node_id] = n
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
    """Search documents and return answer with citations (JSON string)."""
    filters = _make_filters(
        file_name=file_name, section=section, lang=lang, version=version
    )

    engine = _make_engine_with_filters(filters)
    t0 = time.time()

    if USE_HYDE and PARALLEL_HYDE:
        try:
            base_retriever = build_retriever(MetadataFilters(filters=filters)) if filters else build_retriever(None)
            base_task = asyncio.create_task(base_retriever.aretrieve(query))  # type: ignore[attr-defined]
            hyde_task = asyncio.create_task(_generate_hyde_text(query))

            base_nodes, hyde_text = await asyncio.gather(base_task, hyde_task)
            hyde_nodes = await base_retriever.aretrieve(hyde_text)  # type: ignore[attr-defined]

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

    if not str(response).strip() or not getattr(response, "source_nodes", None):
        response = await _make_fallback_engine(filters).aquery(query)

    sources = _build_sources(response)
    logging.info("Found %d sources for query: %s", len(sources), query)
    logging.info("Latency: %.2fs (retrieval+LLM)", t_retrieve_llm)
    payload: Dict[str, Any] = {
        "answer": str(response),
        "sources": sources,
    }
    return json.dumps(payload, ensure_ascii=False)


# ------------------------- Small public helpers -------------------------
# Re-exported by app.py to keep backward compatibility with tests

def public_compute_rerank_top_n(top_k: int, use_rerank: bool) -> Optional[int]:
    return _compute_rerank_top_n(top_k, use_rerank)


def public_make_filters(
    *, file_name: Optional[str], section: Optional[str], lang: Optional[str], version: Optional[str]
) -> List[ExactMatchFilter]:
    return _make_filters(file_name=file_name, section=section, lang=lang, version=version)


def public_build_node_postprocessors() -> List[BaseNodePostprocessor]:
    # Try to pass docstore if index is already initialized; otherwise let it skip gracefully
    docstore = _index.docstore if _index is not None else None  # type: ignore[attr-defined]
    return _build_node_postprocessors(docstore)
