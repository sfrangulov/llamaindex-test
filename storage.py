import os
import logging
from typing import Any, List, Optional

# LlamaIndex core
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)

from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb
from chromadb.config import Settings as ChromaSettings

DATA_PATH = os.getenv("DATA_PATH", "./data")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "knowledge_base")
PERSIST_DIR = os.getenv("PERSIST_DIR", "./storage")


# ---------------------------- Storage / Index ----------------------------
_db: Optional[chromadb.PersistentClient] = None
_vector_store: Optional[ChromaVectorStore] = None
_storage_context: Optional[StorageContext] = None
_index: Optional[VectorStoreIndex] = None


def _get_db() -> chromadb.PersistentClient:
    global _db
    if _db is None:
        _db = chromadb.PersistentClient(
            path=CHROMA_PATH, settings=ChromaSettings(anonymized_telemetry=False))
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
        logging.warning(
            "StorageContext: no persisted state, creating fresh one")
        _storage_context = StorageContext.from_defaults(
            vector_store=_get_vector_store())
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
        _index = VectorStoreIndex.from_vector_store(
            _get_vector_store(), storage_context=sc)
    return _index
