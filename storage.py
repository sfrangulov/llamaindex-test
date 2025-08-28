from chromadb.config import Settings as ChromaSettings
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
import os
from typing import Any, List, Optional
from md_reader import MarkItDownReader

import structlog
log = structlog.get_logger(__name__)

# LlamaIndex core


DATA_PATH = os.getenv("DATA_PATH", "./attachments")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "knowledge_base")
PERSIST_DIR = os.getenv("PERSIST_DIR", "./data/storage")
MD_DIR = os.getenv("MD_DIR", "./data/markdown")


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
        log.warning(
            "StorageContext: no persisted state, creating fresh one")
        _storage_context = StorageContext.from_defaults(
            vector_store=_get_vector_store())
    return _storage_context


def _load_documents_with_metadata() -> List[Any]:
    """Load docs and fill key metadata fields."""
    reader = SimpleDirectoryReader(
        input_dir=DATA_PATH,
        recursive=True,
        required_exts=[".docx"],  # , ".xlsx"
        file_extractor={
            ".docx": MarkItDownReader(),
            ".xlsx": MarkItDownReader(),
        }
    )
    documents = reader.load_data(show_progress=True)
    return documents


def get_index() -> VectorStoreIndex:
    global _index
    if _index is not None:
        return _index

    collection = _get_db().get_or_create_collection(CHROMA_COLLECTION)
    if collection.count() == 0:
        # Build from documents
        documents = _load_documents_with_metadata()
        os.makedirs(MD_DIR, exist_ok=True)
        for doc in documents:
            with open(os.path.join(MD_DIR, f"{doc.metadata['file_name']}.md"), "w", encoding="utf-8") as f:
                f.write(doc.text)
        sc = _get_storage_context(load_persisted=False)
        _index = VectorStoreIndex.from_documents(documents, storage_context=sc)
        _index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        # Load from vector store (and persisted docstore if available)
        sc = _get_storage_context(load_persisted=True)
        _index = VectorStoreIndex.from_vector_store(
            _get_vector_store(), storage_context=sc)
    return _index
