# путь может отличаться по версии
from llama_index.core.readers.file.base import BaseReader
from llama_index.core.schema import Document
from markitdown import MarkItDown
from chromadb.config import Settings as ChromaSettings
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from fsspec import AbstractFileSystem
import hashlib
import time

import structlog
log = structlog.get_logger(__name__)

# LlamaIndex core


DATA_PATH = os.getenv("DATA_PATH", "./attachments")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "knowledge_base")
PERSIST_DIR = os.getenv("PERSIST_DIR", "./data/storage")


md = MarkItDown(enable_plugins=False)


class MarkItDownReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Convert a single file to a Document with rich metadata.

        Note: Do not delete or move the source file here; ingestion should be
        idempotent and non-destructive.
        """
        # Compute file stats and a deterministic content hash for idempotency
        try:
            st = os.stat(file)
            with open(file, "rb") as f:
                blob = f.read()
            sha256 = hashlib.sha256(blob).hexdigest()
        except Exception:
            # If stats/hash failed, still try to convert and proceed without hash
            st = None
            sha256 = None

        result = md.convert(file)

        metadata: Dict[str, Any] = {
            "file_name": file.name,
            "file_path": str(file),
        }
        if st is not None:
            metadata.update(
                {
                    "file_size": getattr(st, "st_size", None),
                    "file_mtime": getattr(st, "st_mtime", None),
                    "file_mtime_iso": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(
                            getattr(st, "st_mtime", 0))
                    ),
                }
            )
        if sha256 is not None:
            metadata["sha256"] = sha256
        if extra_info is not None:
            metadata.update(extra_info)

        # Avoid logging document contents; log path and hash instead
        log.debug(
            "MarkItDownReader",
            file_path=str(file),
            sha256=sha256,
            size=getattr(st, "st_size", None),
        )

        # Use deterministic id_ when available for deduplication
        doc_id = sha256 if sha256 is not None else None
        return [Document(text=result.text_content, metadata=metadata or {}, id_=doc_id)]


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
        required_exts=[".docx"],
        file_extractor={
            ".docx": MarkItDownReader(),
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
        sc = _get_storage_context(load_persisted=False)
        _index = VectorStoreIndex.from_documents(documents, storage_context=sc)
        _index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        # Load from vector store (and persisted docstore if available)
        sc = _get_storage_context(load_persisted=True)
        _index = VectorStoreIndex.from_vector_store(
            _get_vector_store(), storage_context=sc)
    return _index
