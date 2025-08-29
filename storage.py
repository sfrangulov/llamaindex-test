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
import shutil

import structlog
log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Config:
    data_path: Path = Path(os.getenv("DATA_PATH", "./attachments"))
    chroma_path: Path = Path(os.getenv("CHROMA_PATH", "./data/chroma_db"))
    collection: str = os.getenv("CHROMA_COLLECTION", "knowledge_base")
    md_dir: Path = Path(os.getenv("MD_DIR", "./data/markdown"))


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

def _get_chroma_collection():
    return _get_db().get_or_create_collection(CFG.collection)


def _get_vector_store() -> ChromaVectorStore:
    global _vector_store
    if _vector_store is None:
        col = _get_chroma_collection()
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

    ensure_dirs()

    collection =_get_chroma_collection()
    if collection.count() == 0:
        documents = load_documents()
        os.makedirs(CFG.md_dir, exist_ok=True)
        for doc in documents:
            with open(os.path.join(CFG.md_dir, f"{doc.metadata['file_name']}.md"), "w", encoding="utf-8") as f:
                f.write(doc.text)
        storage_context = StorageContext.from_defaults(
            vector_store=_get_vector_store())
        _index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context)
    else:
        _index = VectorStoreIndex.from_vector_store(_get_vector_store())
    return _index


# ----------------- Helpers for UI and ingestion -----------------

def list_storage_files(search: str | None = None) -> list[dict]:
    """List DOCX files in data_path with basic metadata.

    Returns a list of dicts with keys: file_name, file_path, size_bytes, uploaded_at_iso.
    """
    base = CFG.data_path
    ensure_dirs()
    results: list[dict] = []
    try:
        for p in base.rglob("*.docx"):
            try:
                st = p.stat()
                row = {
                    "file_name": p.name,
                    "file_path": str(p),
                    "size_bytes": st.st_size,
                    "uploaded_at_iso": __import__("time").strftime(
                        "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime(getattr(st, "st_mtime", 0))
                    ),
                }
                results.append(row)
            except Exception:
                continue
    except FileNotFoundError:
        pass

    if search:
        s = search.lower().strip()
        results = [r for r in results if s in r["file_name"].lower()]
    # Sort by name by default
    results.sort(key=lambda r: r["file_name"].lower())
    return results


def list_vector_file_names() -> list[str]:
    """Return distinct file_name values present in the Chroma collection."""
    try:
        col = _get_chroma_collection()
        # Fetch metadatas (may be many; acceptable for small/medium sets)
        data = col.get(include=["metadatas"])  # valid include keys only
        names: set[str] = set()
        for md in (data.get("metadatas") or []):
            try:
                name = (md or {}).get("file_name")
                if name:
                    names.add(str(name))
            except Exception:
                continue
        return sorted(names, key=lambda x: x.lower())
    except Exception as e:
        log.warning("list_vector_file_names failed", error=e)
        return []


def delete_from_vector_store_by_file_names(file_names: list[str]) -> int:
    """Delete all embeddings whose metadata.file_name is in file_names.

    Returns the count of delete operations attempted (not the number of vectors).
    """
    if not file_names:
        return 0
    col = _get_chroma_collection()
    deleted = 0
    for name in file_names:
        try:
            col.delete(where={"file_name": name})
            deleted += 1
        except Exception as e:
            log.warning("Delete by file_name failed", file_name=name, error=e)
    return deleted


def add_docx_to_store(file: Path, *, persist_original: bool = True) -> dict:
    """Convert a single DOCX to markdown, upsert into vector store, and persist md.

    If vectors for the same file_name exist, they are removed before insert to avoid duplicates.
    Returns a short dict summary with file_name and bytes.
    """
    ensure_dirs()
    reader = MarkItDownReader()
    docs = reader.load_data(Path(file))
    if not docs:
        raise RuntimeError(f"No content extracted from {file}")
    doc = docs[0]
    file_name = doc.metadata.get("file_name", file.name)

    # Persist original .docx into storage for listing, if requested
    if persist_original:
        try:
            dest = CFG.data_path / file_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            # If source and dest are the same path, skip copy
            if str(Path(file).resolve()) != str(dest.resolve()):
                shutil.copy2(str(file), str(dest))
        except Exception as e:
            log.warning("Failed to persist original DOCX", src=str(file), error=e)

    # Save markdown alongside
    try:
        md_path = CFG.md_dir / f"{file_name}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(doc.text or "")
    except Exception as e:
        log.warning("Failed to write markdown", file=str(file), error=e)

    # Remove old entries for this file_name to avoid duplicates
    try:
        _get_chroma_collection().delete(where={"file_name": file_name})
    except Exception:
        pass

    # Insert into current index/vector store
    idx = get_index()
    idx.insert(doc)

    return {
        "file_name": file_name,
        "size_bytes": doc.metadata.get("file_size"),
        "sha256": doc.metadata.get("sha256"),
    }


def read_markdown(file_name: str) -> str:
    """Read markdown text for a given DOCX file_name (expects <name>.docx.md)."""
    ensure_dirs()
    md_path = CFG.md_dir / f"{file_name}.md"
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Файл Markdown не найден: {md_path}"
    except Exception as e:
        return f"Ошибка чтения Markdown: {e}"
