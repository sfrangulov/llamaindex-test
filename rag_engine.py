import os
import time
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
from llama_index.core.schema import MetadataMode
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.postprocessor import (
    LongContextReorder,
    SimilarityPostprocessor,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.prompts import PromptTemplate  # type: ignore

# Providers
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai.types import EmbedContentConfig
from llama_index.llms.google_genai import GoogleGenAI

import chromadb
from chromadb.config import Settings as ChromaSettings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------------------- Configuration ----------------------------
load_dotenv()

# Tunables
TOP_K = int(os.getenv("TOP_K", 10))
RESPONSE_MODE = os.getenv("RESPONSE_MODE", "compact")

DATA_PATH = os.getenv("DATA_PATH", "./data")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "knowledge_base")
PERSIST_DIR = os.getenv("PERSIST_DIR", "./storage")


def configure_settings() -> None:
    """Init global Settings once (embedder, LLM, parser, transformations)."""
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-001",          # новое имя модели
        embed_batch_size=100,
        embedding_config = EmbedContentConfig(
            output_dimensionality=768,
            task_type="RETRIEVAL_DOCUMENT"
        )
    )
    Settings.llm = GoogleGenAI(model="gemini-2.5-flash", temperature=0.1)
    Settings.transformations = [MarkdownNodeParser()]


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
            input_dir=DATA_PATH,
            recursive=True,
            required_exts=[".md"],  # newer LlamaIndex
        )
    except TypeError:
        # Older versions may use different arg name
        try:
            reader = SimpleDirectoryReader("data", recursive=True)
        except Exception:
            reader = SimpleDirectoryReader("data")
    documents = reader.load_data()
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

def _build_retriever(filters: Optional[MetadataFilters] = None) -> BaseRetriever:
    idx = get_index()
    dense = idx.as_retriever(similarity_top_k=TOP_K, filters=filters)
    return dense


def _build_node_postprocessors() -> List[BaseNodePostprocessor]:
    """Build list of node postprocessors to apply after retrieval."""
    node_postprocessors: List[BaseNodePostprocessor] = []
    node_postprocessors.append(SimilarityPostprocessor(similarity_cutoff=0.05))
    node_postprocessors.append(LongContextReorder())
    return node_postprocessors


def _make_filters(
    *,
    file_name: Optional[str],
) -> List[ExactMatchFilter]:
    """Build filters for document retrieval."""
    filters: List[ExactMatchFilter] = []
    if file_name:
        filters.append(ExactMatchFilter(key="file_name", value=file_name))
    return filters


def _build_sources(response) -> List[Dict[str, Any]]:
    """Build a list of source documents from the response."""
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
                "file_name": meta.get("file_name"),
                "text": sn.node.get_content(metadata_mode=MetadataMode.NONE),
            }
        )
        sources.sort(key=lambda x: x["score"] or 0, reverse=True)
    return sources


def __build_synthesizer():
    # If a system prompt is provided, build a custom QA template that embeds it.
    text_qa_template = None
    template_str = (
        "Инструкции:\n"
        "- Отвечай на русском языке.\n"
        "- Отвечай по делу и только по предоставленному контексту.\n"
        "- Если информации недостаточно — скажи об этом явно.\n"
        "- Приводи короткие цитаты в кавычках при необходимости.\n\n"
        "Контекст:\n{context_str}\n\n"
        "Вопрос пользователя:\n{query_str}\n\n" 
    )
    text_qa_template = PromptTemplate(template_str)

    return get_response_synthesizer(
        llm=Settings.llm,
        response_mode=RESPONSE_MODE,
        use_async=True,
        text_qa_template=text_qa_template,
    )


async def search_documents(
    query: str,
    *,
    file_name: Optional[str] = None,
) -> str:
    """Search and return JSON answer + sources."""

    filters = _make_filters(file_name=file_name)
    metadata_filters = MetadataFilters(filters=filters) if filters else None
    retriever = _build_retriever(metadata_filters)
    postprocessors = _build_node_postprocessors()
    synth = __build_synthesizer()

    t0 = time.time()

    try:
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


def public_make_filters(
    *, file_name: Optional[str]
) -> List[ExactMatchFilter]:
    return _make_filters(file_name=file_name)


def public_build_node_postprocessors() -> List[BaseNodePostprocessor]:
    return _build_node_postprocessors()


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
            logging.warning("Warmup get_index failed: %s", e)
    logging.info("Warmup done (index=%s)", ensure_index)
