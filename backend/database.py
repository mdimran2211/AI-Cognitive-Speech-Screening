import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "outputs" / "analysis_history.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                prediction TEXT NOT NULL,
                cn_score REAL NOT NULL,
                ad_score REAL NOT NULL,
                quality_status TEXT NOT NULL,
                notes TEXT DEFAULT '',
                feedback TEXT DEFAULT '',
                features_json TEXT DEFAULT '{}',
                speech_visuals_json TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        # Default settings if not already set
        defaults = {
            "retention_days": "30",
            "save_audio_recordings": "false",
            "dark_theme": "true",
            "font_size": "medium"
        }
        for k, v in defaults.items():
            cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (k, v))
        conn.commit()


def save_analysis(
    filename: str,
    duration_seconds: float,
    prediction: str,
    cn_score: float,
    ad_score: float,
    quality_status: str,
    features_dict: Optional[Dict[str, Any]] = None,
    visuals_dict: Optional[Dict[str, Any]] = None
) -> int:
    init_db()
    now_str = datetime.now().isoformat()
    features_str = json.dumps(features_dict or {})
    visuals_str = json.dumps(visuals_dict or {})

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO analyses (
                filename, duration_seconds, prediction, cn_score, ad_score,
                quality_status, features_json, speech_visuals_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            filename, duration_seconds, prediction, cn_score, ad_score,
            quality_status, features_str, visuals_str, now_str
        ))
        conn.commit()
        return cursor.lastrowid


def get_analyses(
    search: str = "",
    sort_by: str = "date_desc",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
) -> List[Dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT id, filename, duration_seconds, prediction, cn_score, ad_score, quality_status, notes, feedback, created_at FROM analyses WHERE 1=1"
        params = []

        if search:
            query += " AND (filename LIKE ? OR notes LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])

        if date_from:
            query += " AND created_at >= ?"
            params.append(date_from)

        if date_to:
            query += " AND created_at <= ?"
            params.append(date_to)

        if sort_by == "date_asc":
            query += " ORDER BY created_at ASC"
        elif sort_by == "score_desc":
            query += " ORDER BY ad_score DESC"
        elif sort_by == "score_asc":
            query += " ORDER BY ad_score ASC"
        elif sort_by == "filename_asc":
            query += " ORDER BY filename ASC"
        else:
            query += " ORDER BY created_at DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_analysis_by_id(analysis_id: int) -> Optional[Dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
        row = cursor.fetchone()
        if not row:
            return None
        res = dict(row)
        res["features"] = json.loads(res.get("features_json") or "{}")
        res["speech_visuals"] = json.loads(res.get("speech_visuals_json") or "{}")
        return res


def delete_analysis(analysis_id: int) -> bool:
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        conn.commit()
        return cursor.rowcount > 0


def clear_all_analyses() -> int:
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analyses")
        conn.commit()
        return cursor.rowcount


def update_notes(analysis_id: int, notes: str, feedback: str = "") -> bool:
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE analyses
            SET notes = ?, feedback = ?
            WHERE id = ?
        """, (notes, feedback, analysis_id))
        conn.commit()
        return cursor.rowcount > 0


def get_settings() -> Dict[str, str]:
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM settings")
        rows = cursor.fetchall()
        return {row["key"]: row["value"] for row in rows}


def update_settings(settings: Dict[str, str]) -> Dict[str, str]:
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        for k, v in settings.items():
            cursor.execute("""
                INSERT INTO settings (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (str(k), str(v)))
        conn.commit()
    return get_settings()
