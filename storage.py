from dataclasses import dataclass
from pathlib import Path
from chromadb.config import Settings as ChromaSettings
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
import os
from typing import Optional
from md_reader import MarkItDownReader
from llama_index.core.schema import Document

import structlog
log = structlog.get_logger(__name__)

# LlamaIndex core


@dataclass(frozen=True)
class Config:
    data_path: Path = Path(os.getenv("DATA_PATH", "./attachments"))
    chroma_path: Path = Path(os.getenv("CHROMA_PATH", "./data/chroma_db"))
    collection: str = os.getenv("CHROMA_COLLECTION", "knowledge_base")
    md_dir: Path = Path(os.getenv("MD_DIR", "./data/markdown"))


# ---------------------------- DB ----------------------------
_db: Optional[chromadb.PersistentClient] = None
_vector_store: Optional[ChromaVectorStore] = None
_index: Optional[VectorStoreIndex] = None

CFG = Config()


def ensure_dirs():
    CFG.chroma_path.mkdir(parents=True, exist_ok=True)
    CFG.md_dir.mkdir(parents=True, exist_ok=True)


def _get_db() -> chromadb.PersistentClient:
    global _db
    if _db is None:
        _db = chromadb.PersistentClient(
            path=str(CFG.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _db


def _get_vector_store() -> ChromaVectorStore:
    global _vector_store
    if _vector_store is None:
        col = _get_db().get_or_create_collection(CFG.collection)
        _vector_store = ChromaVectorStore(chroma_collection=col)
    return _vector_store


def load_documents() -> list[Document]:
    reader = SimpleDirectoryReader(
        input_dir=str(CFG.data_path),
        recursive=True,
        required_exts=[".docx"],  # , ".xlsx"
        file_extractor={
            ".docx": MarkItDownReader(), ".xlsx": MarkItDownReader()},
    )
    return reader.load_data(show_progress=True)


def get_index() -> VectorStoreIndex:
    global _index
    if _index is not None:
        return _index

    collection = _get_db().get_or_create_collection(CFG.collection)
    if collection.count() == 0:
        # Build from documents
        documents = load_documents()
        os.makedirs(CFG.md_dir, exist_ok=True)
        for doc in documents:
            with open(os.path.join(CFG.md_dir, f"{doc.metadata['file_name']}.md"), "w", encoding="utf-8") as f:
                f.write(doc.text)
        _index = VectorStoreIndex.from_documents(documents)
        _index.storage_context.persist(persist_dir=CFG.persist_dir)
    else:
        _index = VectorStoreIndex.from_vector_store(
            _get_vector_store())
    return _index
