from typing import List, Dict, Tuple

import numpy as np
from llama_index.core import Settings

from fs_utils import get_fs
from storage import CFG, list_storage_files, read_markdown
from rag_engine import configure_settings


SUBJECT_SECTION_TITLE = "Предмет разработки"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors; returns 0.0 if invalid."""
    try:
        da = float(np.linalg.norm(a))
        db = float(np.linalg.norm(b))
        eps = 1e-12
        if da < eps or db < eps:
            return 0.0
        return float(np.dot(a, b) / (da * db))
    except Exception:
        return 0.0


def _embed_texts(texts: List[str]) -> List[np.ndarray]:
    """Embed a list of texts using the configured embedder.

    Falls back across common LlamaIndex APIs to stay compatible.
    """
    em = Settings.embed_model
    if not texts:
        return []
    # Best-effort: prefer batch APIs when available
    try:
        if hasattr(em, "get_text_embedding_batch"):
            vecs = em.get_text_embedding_batch(texts)
        elif hasattr(em, "embed_query") and callable(getattr(em, "embed_query")):
            vecs = [em.embed_query(t) for t in texts]
        elif hasattr(em, "get_text_embedding") and callable(getattr(em, "get_text_embedding")):
            vecs = [em.get_text_embedding(t) for t in texts]
        else:
            # Last resort: single-call attribute variants
            fn = getattr(em, "embed", None) or getattr(em, "__call__", None)
            if fn:
                vecs = [fn(t) for t in texts]
            else:
                vecs = [[] for _ in texts]
    except Exception:
        vecs = [[] for _ in texts]
    # Normalize to numpy arrays
    out: List[np.ndarray] = []
    for v in vecs:
        try:
            arr = np.asarray(v, dtype=float).reshape(-1)
        except Exception:
            arr = np.zeros((0,), dtype=float)
        out.append(arr)
    return out


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

    # Build embeddings for target and candidates (excluding self)
    cands = [(fname, text) for (fname, text) in all_sections if fname != current_file_name]
    if not cands:
        return []
    target_vec = _embed_texts([target_text])[0]
    cand_texts = [t for (_, t) in cands]
    cand_vecs = _embed_texts(cand_texts)

    scored: List[Tuple[str, float, str]] = []
    for (fname, text), vec in zip(cands, cand_vecs):
        score = _cosine(target_vec, vec)
        scored.append((fname, float(score), text))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[: max(1, int(top_k))]
    return [
        {"file_name": fname, "score": round(score, 4), "text": text}
        for (fname, score, text) in top
    ]


def _truncate(text: str, limit: int = 180) -> str:
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    return t[: max(0, limit - 1)].rstrip() + "…"


def llm_compare_subjects(base_text: str, cand_text: str) -> tuple[str, str]:
    """Return (similar_short, diff_short) via LLM based on two subject section texts.

    Produces two concise phrases in Russian. Falls back to '—' on error.
    """
    try:
        from llama_index.core import Settings as _Settings
        # Build compact prompt with explicit format
        base = _truncate(base_text, 1200)
        cand = _truncate(cand_text, 1200)
        prompt = (
            "Сравни два фрагмента раздела 'Предмет разработки'.\n"
            "Сравнивай только непосредственно предмет разработки.\n"
            "Ответ строго в 2 строках и кратко (1–2 фразы на строку):\n"
            "1) Схожи: <кратко по сути>\n"
            "2) Отличаются: <кратко по сути>\n"
            "Пиши на русском, без воды, без Markdown, без лишних пояснений.\n\n"
            f"Текст A:\n{base}\n\nТекст B:\n{cand}\n"
        )
        import asyncio as _asyncio
        resp = _asyncio.run(_Settings.llm.acomplete(prompt))
        text = (getattr(resp, "text", None) or "").strip()
        # Parse two lines heuristically
        sim, diff = "", ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in lines:
            low = ln.lower()
            if not sim and (low.startswith("1) схожи") or low.startswith("схожи")):
                sim = ln.split(":", 1)[-1].strip() if ":" in ln else ln
            elif not diff and (low.startswith("2) отличаются") or low.startswith("отличаются")):
                diff = ln.split(":", 1)[-1].strip() if ":" in ln else ln
        sim = _truncate(sim or text, 200)
        diff = _truncate(diff or "—", 200)
        # Clean potential numbering remnants
        sim = sim.lstrip("1234567890). -:").strip() or "—"
        diff = diff.lstrip("1234567890). -:").strip() or "—"
        return sim, diff
    except Exception:
        return "—", "—"

