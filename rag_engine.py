import os
import time
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# LlamaIndex core
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.postprocessor import (
    LongContextReorder,
    SimilarityPostprocessor,
    SentenceTransformerRerank,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import BaseRetriever

# Providers
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

import chromadb
from chromadb.config import Settings as ChromaSettings


# ---------------------------- Configuration ----------------------------
load_dotenv()

# Tunables
TOP_K = int(os.getenv("TOP_K", 10))
USE_HYDE = os.getenv("USE_HYDE", "false").lower() == "true"
USE_RERANK = os.getenv("USE_RERANK", "true").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RESPONSE_MODE = os.getenv("RESPONSE_MODE", "compact")

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "test")
PERSIST_DIR = os.getenv("PERSIST_DIR", "./storage")


def configure_settings() -> None:
    """Init global Settings once (embedder, LLM, parser)."""
    Settings.embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004")
    Settings.llm = GoogleGenAI(model="gemini-2.5-flash", temperature=0.1)
    Settings.node_parser = _make_node_parser()


def _make_node_parser():
    """Prefer Markdown parser; fallback to SentenceSplitter."""
    choice = os.getenv("NODE_PARSER", "markdown").lower()
    if choice in ("markdown", "md"):
        try:
            from llama_index.core.node_parser import MarkdownNodeParser  # type: ignore

            return MarkdownNodeParser()
        except Exception as e:
            logging.warning("MarkdownNodeParser unavailable, using SentenceSplitter: %s", e)
    return SentenceSplitter(chunk_size=700, chunk_overlap=120)


def infer_language(text: str) -> str:
    """Cyrillic -> ru, else en."""
    return "ru" if any("\u0400" <= ch <= "\u04FF" for ch in text) else "en"


# ---------------------------- Storage / Index ----------------------------
_db: Optional[chromadb.PersistentClient] = None
_vector_store: Optional[ChromaVectorStore] = None
_storage_context: Optional[StorageContext] = None
_index: Optional[VectorStoreIndex] = None


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
        logging.warning("StorageContext: no persisted state, creating fresh one")
        _storage_context = StorageContext.from_defaults(vector_store=_get_vector_store())
    return _storage_context


def _load_documents_with_metadata() -> List[Any]:
    """Load docs and fill key metadata fields.

    Restrict to Markdown/text to avoid heavy Excel/others and reduce embedding payload.
    """
    try:
        reader = SimpleDirectoryReader(
            input_dir="data",
            recursive=True,
            required_exts=[".md", ".markdown", ".txt"],  # newer LlamaIndex
        )
    except TypeError:
        # Older versions may use different arg name
        try:
            reader = SimpleDirectoryReader("data", recursive=True)
        except Exception:
            reader = SimpleDirectoryReader("data")
    documents = reader.load_data()
    for d in documents:
        d.metadata.setdefault(
            "file_name",
            d.metadata.get("file_name") or d.metadata.get("filename") or "unknown",
        )
        d.metadata.setdefault("section", "unknown")
        d.metadata.setdefault("version", "v1")
        d.metadata.setdefault("lang", infer_language(d.text or ""))
    return documents


def get_index() -> VectorStoreIndex:
    global _index
    if _index is not None:
        return _index

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

def build_retriever(filters: Optional[MetadataFilters] = None) -> BaseRetriever:
    idx = get_index()
    dense = idx.as_retriever(similarity_top_k=TOP_K, filters=filters)
    return dense


def _compute_rerank_top_n(top_k: int, use_rerank: bool) -> Optional[int]:
    return min(5, max(1, top_k)) if use_rerank else None


def _build_node_postprocessors() -> List[BaseNodePostprocessor]:
    """Lightweight, effective stack: optional cross-encoder -> cutoff -> reorder."""
    node_postprocessors: List[BaseNodePostprocessor] = []
    try:
        top_n = _compute_rerank_top_n(TOP_K, USE_RERANK)
        if top_n and SentenceTransformerRerank is not None:
            logging.info("Using SentenceTransformerRerank with top_n=%d", top_n)
            node_postprocessors.append(
                SentenceTransformerRerank(top_n=top_n, model=RERANK_MODEL)
            )
    except Exception as e:
        logging.warning("Reranker init failed: %s", e)

    node_postprocessors.append(SimilarityPostprocessor(similarity_cutoff=0.05))
    node_postprocessors.append(LongContextReorder())
    return node_postprocessors


async def _generate_hyde_text(user_query: str) -> str:
    """Generate a short synthetic paragraph (HyDE)."""
    template_text = (
        "Сгенерируй краткий (80-120 слов) абзац, который мог бы содержаться в документации, "
        "отвечающий на запрос: \n{query}\n"
        "Пиши информативно и по делу, без вступлений."
    )
    try:
        from llama_index.core.prompts import PromptTemplate as PromptTemplate  # type: ignore

        tmpl = PromptTemplate(template_text)
        return await Settings.llm.apredict(tmpl, prompt_args={"query": user_query})  # type: ignore
    except Exception:
        try:
            return await Settings.llm.apredict(template_text.format(query=user_query))  # type: ignore
        except Exception:
            return Settings.llm.predict(template_text.format(query=user_query))


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
            score_val = float(sn.score) if getattr(sn, "score", None) is not None else None
        except Exception:
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


def _synthesizer():
    return get_response_synthesizer(llm=Settings.llm, response_mode=RESPONSE_MODE, use_async=True)


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
    """Search and return JSON answer + sources."""
    configure_settings()
    _ = get_index()

    filters = _make_filters(file_name=file_name, section=section, lang=lang, version=version)
    metadata_filters = MetadataFilters(filters=filters) if filters else None
    retriever = build_retriever(metadata_filters)
    postprocessors = _build_node_postprocessors()
    synth = _synthesizer()

    t0 = time.time()

    try:
        if USE_HYDE:
            base_task = asyncio.create_task(retriever.aretrieve(query))  # type: ignore
            hyde_task = asyncio.create_task(_generate_hyde_text(query))
            base_nodes, hyde_text = await asyncio.gather(base_task, hyde_task)
            hyde_nodes = await retriever.aretrieve(hyde_text)  # type: ignore
            nodes = _merge_nodes_by_score([base_nodes, hyde_nodes], TOP_K)
        else:
            nodes = await retriever.aretrieve(query)  # type: ignore

        # Provide a lightweight QueryBundle replacement so rerankers receive the query
        class _QB:
            def __init__(self, s: str) -> None:
                self.query_str = s

        qb = _QB(query)
        for p in postprocessors:
            try:
                nodes = p.postprocess_nodes(nodes, query_bundle=qb)  # type: ignore
            except Exception as e:  # keep going if optional deps missing
                logging.warning("Postprocessor %s failed: %s", p.__class__.__name__, e)

        response = await synth.asynthesize(query, nodes)
    except Exception as e:
        logging.exception("Search failed, returning empty result: %s", e)
        response = type("Empty", (), {"__str__": lambda self: "", "source_nodes": []})()

    latency = time.time() - t0
    sources = _build_sources(response)
    logging.info("Found %d sources | %.2fs", len(sources), latency)
    payload: Dict[str, Any] = {"answer": str(response), "sources": sources}
    return json.dumps(payload, ensure_ascii=False)


# ------------------------- Public helpers (compat) -------------------------

def public_compute_rerank_top_n(top_k: int, use_rerank: bool) -> Optional[int]:
    return _compute_rerank_top_n(top_k, use_rerank)


def public_make_filters(
    *, file_name: Optional[str], section: Optional[str], lang: Optional[str], version: Optional[str]
) -> List[ExactMatchFilter]:
    return _make_filters(file_name=file_name, section=section, lang=lang, version=version)


def public_build_node_postprocessors() -> List[BaseNodePostprocessor]:
    return _build_node_postprocessors()


# ----------------------------- Warmup helpers -----------------------------
def warmup(
    *, ensure_index: bool = True, preload_reranker: bool = USE_RERANK
) -> None:
    """Warm heavy parts to reduce first-request latency."""
    configure_settings()
    if ensure_index:
        try:
            _ = get_index()
        except Exception as e:
            logging.warning("Warmup get_index failed: %s", e)
    if preload_reranker:
        try:
            _ = _build_node_postprocessors()
        except Exception as e:
            logging.warning("Warmup reranker failed: %s", e)
    logging.info("Warmup done (index=%s, reranker=%s)", ensure_index, preload_reranker)
