# путь может отличаться по версии
from llama_index.core.readers.file.base import BaseReader
from llama_index.core.schema import Document
from markitdown import MarkItDown
from pathlib import Path
from typing import Any, Dict, List, Optional
from fsspec import AbstractFileSystem
import hashlib
import time
import os
import structlog
log = structlog.get_logger(__name__)

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

        doc_id = sha256 if sha256 is not None else None
        return [Document(text=result.text_content, metadata=metadata or {}, id_=doc_id)]
