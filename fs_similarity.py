from typing import List, Dict, Tuple
import re

from llama_index.core import Settings
from llama_index.core import Document, VectorStoreIndex

from fs_utils import get_fs
from storage import CFG, list_storage_files, read_markdown


SUBJECT_SECTION_TITLE = "Предмет разработки"


# NOTE: manual embedding/cosine utilities removed in favor of LlamaIndex's VectorStoreIndex


def _load_subject_sections() -> List[Tuple[str, str]]:
    """Return a list of (file_name, section_text) for all docs that have the SUBJECT_SECTION_TITLE."""
    # Use storage listing to get canonical file names present under attachments
    rows = list_storage_files()
    results: List[Tuple[str, str]] = []
    for r in rows:
        fname = r.get("file_name")
        if not fname:
            continue
        md = read_markdown(fname)
        if not isinstance(md, str) or md.startswith("Файл Markdown не найден"):
            continue
        try:
            sections = get_fs(str(CFG.md_dir / f"{fname}.md"))
        except Exception:
            sections = {}
        text = (sections or {}).get(SUBJECT_SECTION_TITLE) or ""
        text = text.strip()
        if text:
            results.append((fname, text))
    return results


def find_similar_subjects(current_file_name: str, top_k: int = 10) -> List[Dict[str, object]]:
    """Find top-K most similar FS documents by the 'Предмет разработки' section.

    Returns list of { file_name: str, score: float, text: str } sorted by score desc.
    """
    if not current_file_name:
        return []
    all_sections = _load_subject_sections()

    # Extract target text
    target_text = None
    for fname, text in all_sections:
        if fname == current_file_name:
            target_text = text
            break
    if not target_text:
        return []

    # Build an in-memory VectorStoreIndex over candidate sections using LlamaIndex
    candidates: List[Tuple[str, str]] = [
        (fname, text) for (fname, text) in all_sections if fname != current_file_name and text.strip()
    ]
    if not candidates:
        return []

    docs: List[Document] = [
        Document(text=text, metadata={"file_name": fname, "section": SUBJECT_SECTION_TITLE})
        for fname, text in candidates
    ]
    # Use global Settings (embed model, etc.) implicitly
    try:
        index = VectorStoreIndex.from_documents(docs)
        retriever = index.as_retriever(similarity_top_k=max(1, int(top_k)))
        results = retriever.retrieve(target_text)
    except Exception:
        # Fallback: empty on failure (avoid partial/manual cosine to keep behavior simple)
        return []

    out: List[Dict[str, object]] = []
    for r in results:
        # r.node.get_content() returns the text; metadata were attached to the Document
        meta = r.node.metadata or {}
        fname = meta.get("file_name")
        if not fname or not isinstance(fname, str):
            continue
        out.append({
            "file_name": fname,
            "score": round(float(r.score or 0.0), 4),
            "text": r.node.get_content(),
        })

    # Ensure stable ordering by score desc
    out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return out


def _truncate(text: str, limit: int = 180) -> str:
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    return t[: max(0, limit - 1)].rstrip() + "…"


def _build_compare_prompt(base_text: str, cand_text: str) -> str:
    base = _truncate(base_text, 1200)
    cand = _truncate(cand_text, 1200)
    return (
        "Сравни два фрагмента раздела 'Предмет разработки'.\n"
        "Сравнивай только непосредственно предмет разработки.\n"
        "Ответ строго в 2 строках и кратко (1–2 фразы на строку):\n"
        "1) Схожи: <кратко по сути>\n"
        "2) Отличаются: <кратко по сути>\n"
        "Пиши на русском, без воды, без Markdown, без лишних пояснений.\n\n"
        f"Текст первой ФС:\n{base}\n\nТекст второй ФС:\n{cand}\n"
    )


def _parse_two_line_answer(text: str) -> tuple[str, str]:
    """Extract 'Схожи' and 'Отличаются' short phrases from LLM text."""
    text = (text or "").strip()
    if not text:
        return "—", "—"

    # Try to capture lines beginning with the expected labels
    sim_match = re.search(r"(?im)^\s*(?:1\)\s*)?схожи[^:]*:\s*(.*)$", text)
    diff_match = re.search(r"(?im)^\s*(?:2\)\s*)?отличаются[^:]*:\s*(.*)$", text)

    def _clean_fragment(s: str) -> str:
        s = _truncate((s or "").strip() or "—", 200)
        s = s.lstrip("1234567890). -:").strip() or "—"
        return s

    if sim_match or diff_match:
        sim = _clean_fragment(sim_match.group(1)) if sim_match else "—"
        diff = _clean_fragment(diff_match.group(1)) if diff_match else "—"
        return sim, diff

    # Fallback: use first two non-empty lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    first = lines[0] if lines else "—"
    second = lines[1] if len(lines) > 1 else "—"
    return _clean_fragment(first), _clean_fragment(second)


def llm_compare_subjects(base_text: str, cand_text: str) -> tuple[str, str]:
    """Return (similar_short, diff_short) via LLM based on two subject section texts.

    Produces two concise phrases in Russian. Falls back to '—' on error.
    """
    try:
        prompt = _build_compare_prompt(base_text, cand_text)
        # Prefer the sync interface exposed by LlamaIndex LLMs
        resp = Settings.llm.complete(prompt)
        return _parse_two_line_answer(getattr(resp, "text", None) or "")
    except Exception:
        return "—", "—"

