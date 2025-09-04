"""SQLite-backed storage for FS analysis results.

Creates two tables:
- fs_analysis: one row per analysis run
- fs_analysis_section: one row per section in a run

This module is thread-safe for simple use: it opens a new connection per call.
"""

from __future__ import annotations

import os
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DB_PATH = Path("./data/results.db")


def _ensure_data_dir() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    _ensure_data_dir()
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_results_db() -> None:
    """Initialize database schema if not present."""
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fs_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                runtime_ms INTEGER,
                sections_count INTEGER,
                metadata_json TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fs_analysis_section (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                job_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                section_title TEXT NOT NULL,
                overall_assessment TEXT,
                details_markdown TEXT,
                FOREIGN KEY (analysis_id) REFERENCES fs_analysis(id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_fs_analysis_file_created
            ON fs_analysis(file_name, created_at DESC);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_fs_analysis_section_analysis
            ON fs_analysis_section(analysis_id);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_fs_analysis_section_title
            ON fs_analysis_section(file_name, section_title);
            """
        )
        conn.commit()


def save_fs_analysis(
    file_name: str,
    analysis: Dict[str, Dict[str, Any]],
    job_id: str,
    runtime_ms: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """Persist one FS analysis run and its per-section results.

    Returns the inserted analysis_id.
    """
    if not isinstance(analysis, dict) or not analysis:
        raise ValueError("analysis must be a non-empty dict")

    created_at = datetime.now(timezone.utc).isoformat()
    sections_count = len(analysis)
    meta_json = json.dumps(metadata or {}, ensure_ascii=False)

    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO fs_analysis (job_id, file_name, created_at, runtime_ms, sections_count, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (job_id, file_name, created_at, int(runtime_ms) if runtime_ms is not None else None, sections_count, meta_json),
        )
        analysis_id = int(cur.lastrowid)

        section_rows: List[Tuple[int, str, str, str, Optional[str], str]] = []
        for section_title, payload in analysis.items():
            payload = payload or {}
            overall = payload.get("overall_assessment")
            details = payload.get("details_markdown") or payload.get("summary") or ""
            section_rows.append(
                (analysis_id, job_id, file_name, str(section_title), overall, str(details))
            )

        cur.executemany(
            """
            INSERT INTO fs_analysis_section (analysis_id, job_id, file_name, section_title, overall_assessment, details_markdown)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            section_rows,
        )
        conn.commit()
        return analysis_id


def get_latest_fs_analysis(file_name: str) -> Optional[Dict[str, Any]]:
    """Return the latest analysis (header + sections) for a file."""
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, job_id, file_name, created_at, runtime_ms, sections_count, metadata_json\n             FROM fs_analysis WHERE file_name = ? ORDER BY datetime(created_at) DESC LIMIT 1",
            (file_name,),
        )
        row = cur.fetchone()
        if not row:
            return None
        analysis_id, job_id, file_name, created_at, runtime_ms, sections_count, metadata_json = row
        cur.execute(
            "SELECT section_title, overall_assessment, details_markdown FROM fs_analysis_section WHERE analysis_id = ?",
            (analysis_id,),
        )
        sections = {r[0]: {"overall_assessment": r[1], "details_markdown": r[2]} for r in cur.fetchall()}
        return {
            "id": analysis_id,
            "job_id": job_id,
            "file_name": file_name,
            "created_at": created_at,
            "runtime_ms": runtime_ms,
            "sections_count": sections_count,
            "metadata": json.loads(metadata_json or "{}"),
            "analysis": sections,
        }

