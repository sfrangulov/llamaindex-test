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
import zipfile
import re
import json
import structlog
log = structlog.get_logger(__name__)

md = MarkItDown(enable_plugins=False)


def _should_save_images() -> bool:
    val = os.getenv("MD_SAVE_IMAGES", "1").strip()
    return val in {"1", "true", "True"}


def _ensure_image_dir(src_file: Path) -> Path:
    stem = Path(src_file).stem
    img_root = Path(os.getenv("MD_IMG_DIR", "./data/markdown_assets"))
    out_dir = img_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _extract_docx_images(src_file: Path, out_dir: Path) -> list[str]:
    saved_files: list[str] = []
    seen_hashes: set[str] = set()
    try:
        with zipfile.ZipFile(src_file, "r") as zf:
            names = [n for n in zf.namelist() if n.startswith("word/media/")]
            for idx, name in enumerate(names, start=1):
                try:
                    data = zf.read(name)
                except Exception:
                    continue
                h = hashlib.sha256(data).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                ext = os.path.splitext(name)[1].lstrip(".") or "bin"
                file_name = f"img_{idx:03d}_{h[:8]}.{ext}"
                try:
                    with open(out_dir / file_name, "wb") as fimg:
                        fimg.write(data)
                    saved_files.append(file_name)
                except Exception:
                    continue
    except zipfile.BadZipFile:
        return []
    return saved_files


def _rewrite_md_data_uri_placeholders(text: str, saved_files: list[str], link_dir: Optional[Path] = None, url_base: Optional[str] = None) -> str:
    if not saved_files:
        return text
    # One regex that handles optional bold wrappers around the image
    pattern = re.compile(r'(\*\*)?!\[[^\]]*\]\(\s*data:image/[^)]*\)(\*\*)?')
    files_iter = iter(saved_files)

    def _repl(m: re.Match) -> str:
        try:
            fn = next(files_iter)
        except StopIteration:
            return m.group(0)
        prefix = m.group(1) or ''
        suffix = m.group(2) or ''
        if url_base:
            path = f"{url_base}/{fn}"
        else:
            if link_dir:
                path = str(link_dir / fn)
            else:
                path = fn
        # Use Flask-served relative URL, no host/port hardcode
        return f"{prefix}![](<{path}>){suffix}"

    return pattern.sub(_repl, text)


def _process_docx_images(src_file: Path, text: str) -> tuple[str, Optional[Path], Optional[list[str]]]:
    """Optionally extract images from DOCX and rewrite Markdown placeholders.

    Returns: (new_text, out_dir or None, saved_files or None)
    """
    if not (_should_save_images() and str(src_file).lower().endswith(".docx") and os.path.exists(src_file)):
        return text, None, None
    out_dir = _ensure_image_dir(src_file)
    saved_files = _extract_docx_images(src_file, out_dir)
    # Prefer serving via app route: /markdown_assets/<stem>/<file>
    url_base = f"/markdown_assets/{out_dir.name}"
    # Fallback relative path (if needed elsewhere)
    md_dir = Path(os.getenv("MD_DIR", "./data/markdown")).resolve()
    link_dir = Path(os.path.relpath(out_dir.resolve(), start=md_dir))
    new_text = _rewrite_md_data_uri_placeholders(
        text, saved_files, link_dir, url_base)
    return new_text, out_dir, saved_files


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
            st = None
            sha256 = None

        # Convert to markdown
        result = md.convert(file)
        text_content: str = result.text_content or ""

        # Optional: extract images from DOCX into files and rewrite placeholders
        try:
            text_content, out_dir, saved_files = _process_docx_images(
                file, text_content)
        except Exception as e:
            log.warning("docx_image_extraction_failed",
                        file=str(file), error=e)

        # Build metadata
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
                        "%Y-%m-%dT%H:%M:%SZ",
                        time.gmtime(getattr(st, "st_mtime", 0)),
                    ),
                }
            )
        if sha256 is not None:
            metadata["sha256"] = sha256
        if extra_info is not None:
            metadata.update(extra_info)

        # Images metadata
        try:
            if "out_dir" in locals() and out_dir.exists():
                metadata["image_dir"] = str(out_dir)
                # saved_files may not exist if branch skipped
                if "saved_files" in locals():
                    image_paths = [str(out_dir / fn)
                                   for fn in (saved_files or [])]
                    metadata["images_count"] = len(image_paths)
                    # store as JSON string to satisfy scalar-only metadata constraints
                    metadata["images_saved"] = json.dumps(
                        image_paths, ensure_ascii=False)
        except Exception:
            pass

        doc_id = sha256 if sha256 is not None else None
        return [Document(text=text_content, metadata=metadata or {}, id_=doc_id)]
