"""
SQLite-backed transcript history store for Sanketra v1.2 dashboard.

Schema per DASHBOARD_AND_WEB.html §3.2:
  sessions:    id, started_at, ended_at, client_kind, client_name, language
  transcripts: id, session_id, created_at, text, app_context, language, word_count
  transcripts_fts (FTS5 virtual table, external-content over transcripts)

Design choices:
  - One connection per HistoryDB instance. FastAPI serves a single event loop;
    sqlite3 connections are NOT threadsafe across threads, but every handler
    invokes this module synchronously from that loop. Writes execute in a
    local lock to serialize the rare multi-statement paths (log_transcript
    updates FTS + word_count via triggers in one transaction).
  - `check_same_thread=False` so the shared module-level singleton can be
    touched from both the event loop and the threadpool (used by _save_video
    and test fixtures). The internal Lock restores the serialization guarantee.
  - FTS5 kept in sync with transcripts via triggers (INSERT/DELETE/UPDATE) so
    a naive `DELETE FROM transcripts WHERE session_id = ?` correctly removes
    FTS rows without manual bookkeeping.
  - `logging_enabled` flag lives in a single-row `settings` table (id=1) rather
    than the existing ~/.config/sanketra/config.json so the dashboard can toggle
    it without going through the auth-config file and risking an accidental
    auth_token wipe.

The class is designed for tests to instantiate with an arbitrary path (including
":memory:"); the module-level `get_default_db()` returns a singleton pointed at
~/.config/sanketra/history.db for the server.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Iterable


DEFAULT_DB_PATH = os.path.join(
    os.path.expanduser("~"), ".config", "sanketra", "history.db"
)


def _word_count(text: str) -> int:
    if not text:
        return 0
    # Unicode-safe split — \w matches Devanagari too in Python's `re`.
    # We use simple whitespace split for speed since Hindi/English tokenization
    # is whitespace-delimited at the word level.
    return len([w for w in text.split() if w.strip()])


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   INTEGER NOT NULL,
    ended_at     INTEGER,
    client_kind  TEXT NOT NULL,
    client_name  TEXT,
    language     TEXT NOT NULL DEFAULT 'hi'
);

CREATE TABLE IF NOT EXISTS transcripts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    created_at   INTEGER NOT NULL,
    text         TEXT NOT NULL,
    app_context  TEXT,
    language     TEXT NOT NULL DEFAULT 'hi',
    word_count   INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_transcripts_session ON transcripts(session_id);
CREATE INDEX IF NOT EXISTS idx_transcripts_created ON transcripts(created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_started   ON sessions(started_at);

CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts
    USING fts5(text, content='transcripts', content_rowid='id', tokenize='unicode61');

CREATE TRIGGER IF NOT EXISTS transcripts_ai AFTER INSERT ON transcripts BEGIN
    INSERT INTO transcripts_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS transcripts_ad AFTER DELETE ON transcripts BEGIN
    INSERT INTO transcripts_fts(transcripts_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS transcripts_au AFTER UPDATE ON transcripts BEGIN
    INSERT INTO transcripts_fts(transcripts_fts, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO transcripts_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE IF NOT EXISTS settings (
    id               INTEGER PRIMARY KEY CHECK (id = 1),
    logging_enabled  INTEGER NOT NULL DEFAULT 1
);
INSERT OR IGNORE INTO settings (id, logging_enabled) VALUES (1, 1);
"""


class HistoryDB:
    """Thread-safe SQLite wrapper for transcript history."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        # check_same_thread=False: safe because _lock serializes writes and the
        # async event loop is single-threaded. Tests may cross thread boundaries.
        self._conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        # Enforce ON DELETE CASCADE on the transcripts.session_id FK.
        self._conn.execute("PRAGMA foreign_keys = ON")
        # WAL for better concurrent read while a write is in flight.
        if db_path != ":memory:":
            try:
                self._conn.execute("PRAGMA journal_mode = WAL")
            except sqlite3.DatabaseError:
                pass
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ sessions

    def create_session(
        self,
        client_kind: str,
        client_name: str | None = None,
        language: str = "hi",
        started_at_ms: int | None = None,
    ) -> int:
        """Insert a new session row and return its id."""
        started_at = started_at_ms if started_at_ms is not None else int(time.time() * 1000)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sessions (started_at, ended_at, client_kind, client_name, language) "
                "VALUES (?, NULL, ?, ?, ?)",
                (started_at, client_kind, client_name, language),
            )
            return int(cur.lastrowid)

    def end_session(self, session_id: int, ended_at_ms: int | None = None) -> None:
        """Mark a session ended. Idempotent — safe to call on an already-ended session."""
        ended_at = ended_at_ms if ended_at_ms is not None else int(time.time() * 1000)
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
                (ended_at, session_id),
            )

    # ------------------------------------------------------------------ transcripts

    def log_transcript(
        self,
        session_id: int,
        text: str,
        app_context: str | None = None,
        language: str = "hi",
        created_at_ms: int | None = None,
    ) -> int:
        """Insert a transcript. Caller is responsible for checking logging_enabled."""
        if not text:
            return 0
        created_at = created_at_ms if created_at_ms is not None else int(time.time() * 1000)
        wc = _word_count(text)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO transcripts (session_id, created_at, text, app_context, language, word_count) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, created_at, text, app_context, language, wc),
            )
            return int(cur.lastrowid)

    # ------------------------------------------------------------------ queries

    def list_sessions(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """Paginated sessions, newest first, with aggregate transcript count."""
        limit = max(1, min(500, int(limit)))
        offset = max(0, int(offset))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT s.id, s.started_at, s.ended_at, s.client_kind, s.client_name, s.language,
                       COUNT(t.id) AS transcript_count,
                       COALESCE(SUM(t.word_count), 0) AS total_words
                FROM sessions s
                LEFT JOIN transcripts t ON t.session_id = s.id
                GROUP BY s.id
                ORDER BY s.started_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id: int) -> dict[str, Any] | None:
        """Session row + all transcripts (chronological)."""
        with self._lock:
            sess = self._conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if sess is None:
                return None
            transcripts = self._conn.execute(
                "SELECT id, created_at, text, app_context, language, word_count "
                "FROM transcripts WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
        out = dict(sess)
        out["transcripts"] = [dict(t) for t in transcripts]
        return out

    def list_by_date(self, date_yyyy_mm_dd: str) -> list[dict[str, Any]]:
        """All transcripts whose created_at falls within [date 00:00, date 24:00)."""
        start_ms, end_ms = _date_bounds_ms(date_yyyy_mm_dd)
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT t.id, t.session_id, t.created_at, t.text, t.app_context, t.language, t.word_count
                FROM transcripts t
                WHERE t.created_at >= ? AND t.created_at < ?
                ORDER BY t.created_at ASC
                """,
                (start_ms, end_ms),
            ).fetchall()
        return [dict(r) for r in rows]

    def search(self, query: str, limit: int = 100) -> list[dict[str, Any]]:
        """FTS5 match against transcripts.text. Empty query → empty list (no full-table scan)."""
        q = (query or "").strip()
        if not q:
            return []
        limit = max(1, min(500, int(limit)))
        # Escape FTS5 syntax by quoting the phrase — users shouldn't need to
        # learn FTS5 operators. "I ate दाल" works verbatim.
        safe = '"' + q.replace('"', '""') + '"'
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT t.id, t.session_id, t.created_at, t.text, t.app_context, t.language, t.word_count
                FROM transcripts t
                JOIN transcripts_fts fts ON fts.rowid = t.id
                WHERE transcripts_fts MATCH ?
                ORDER BY t.created_at DESC
                LIMIT ?
                """,
                (safe, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ deletes

    def delete_session(self, session_id: int) -> int:
        """Delete a session (cascades to transcripts). Returns rows removed from sessions."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cur.rowcount or 0

    def delete_by_date(self, date_yyyy_mm_dd: str) -> int:
        """Delete transcripts (not sessions) on a given date. Returns count removed."""
        start_ms, end_ms = _date_bounds_ms(date_yyyy_mm_dd)
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM transcripts WHERE created_at >= ? AND created_at < ?",
                (start_ms, end_ms),
            )
            return cur.rowcount or 0

    def clear_all(self) -> tuple[int, int]:
        """Wipe transcripts + sessions. Returns (sessions_deleted, transcripts_deleted)."""
        with self._lock:
            t_cur = self._conn.execute("DELETE FROM transcripts")
            s_cur = self._conn.execute("DELETE FROM sessions")
            # Reset AUTOINCREMENT so next session starts at id=1 (cleaner for tests).
            self._conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('sessions','transcripts')")
        return (s_cur.rowcount or 0, t_cur.rowcount or 0)

    # ------------------------------------------------------------------ export

    def export_json(self) -> dict[str, Any]:
        """Full dump suitable for JSON/TXT/MD serialization by the caller."""
        with self._lock:
            sessions = [dict(r) for r in self._conn.execute(
                "SELECT * FROM sessions ORDER BY started_at ASC"
            ).fetchall()]
            transcripts = [dict(r) for r in self._conn.execute(
                "SELECT * FROM transcripts ORDER BY created_at ASC"
            ).fetchall()]
        return {
            "version": 1,
            "exported_at": int(time.time() * 1000),
            "sessions": sessions,
            "transcripts": transcripts,
        }

    def export_txt(self) -> str:
        """Human-readable flat text dump. One transcript per line with timestamp."""
        data = self.export_json()
        lines: list[str] = []
        # Group transcripts by session for readability.
        by_session: dict[int, list[dict[str, Any]]] = {}
        for t in data["transcripts"]:
            by_session.setdefault(t["session_id"], []).append(t)
        for s in data["sessions"]:
            lines.append(f"=== Session {s['id']} · {s['client_kind']}"
                         f" · {s.get('client_name') or '-'}"
                         f" · lang={s.get('language','hi')} ===")
            for t in by_session.get(s["id"], []):
                ts = _iso_from_ms(t["created_at"])
                lines.append(f"[{ts}] {t['text']}")
            lines.append("")
        return "\n".join(lines)

    def export_md(self) -> str:
        """Markdown dump. Same structure as export_txt but with headings + blockquotes."""
        data = self.export_json()
        lines: list[str] = ["# Sanketra transcript archive", ""]
        lines.append(f"*Exported:* `{_iso_from_ms(data['exported_at'])}`")
        lines.append("")
        by_session: dict[int, list[dict[str, Any]]] = {}
        for t in data["transcripts"]:
            by_session.setdefault(t["session_id"], []).append(t)
        for s in data["sessions"]:
            started = _iso_from_ms(s["started_at"])
            lines.append(f"## Session {s['id']}")
            lines.append("")
            lines.append(f"- **Started:** {started}")
            lines.append(f"- **Client:** {s['client_kind']} · {s.get('client_name') or '-'}")
            lines.append(f"- **Language:** {s.get('language','hi')}")
            lines.append("")
            for t in by_session.get(s["id"], []):
                ts = _iso_from_ms(t["created_at"])
                lines.append(f"> {t['text']}  ")
                lines.append(f"> *{ts}*")
                lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------ settings

    def get_settings(self) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute("SELECT logging_enabled FROM settings WHERE id = 1").fetchone()
        return {"logging_enabled": bool(row["logging_enabled"]) if row else True}

    def set_settings(self, *, logging_enabled: bool | None = None) -> dict[str, Any]:
        if logging_enabled is not None:
            with self._lock:
                self._conn.execute(
                    "UPDATE settings SET logging_enabled = ? WHERE id = 1",
                    (1 if logging_enabled else 0,),
                )
        return self.get_settings()


# ---------------------------------------------------------------------- helpers

def _date_bounds_ms(date_yyyy_mm_dd: str) -> tuple[int, int]:
    """Convert 'YYYY-MM-DD' into [start_ms, end_ms) local-time bounds.

    Uses local time deliberately — the user thinks in their wall clock, not UTC.
    If the server crosses a timezone boundary this needs revisiting, but for a
    single-user LAN product that's acceptable.
    """
    import datetime as _dt
    try:
        d = _dt.date.fromisoformat(date_yyyy_mm_dd)
    except ValueError as e:
        raise ValueError(f"invalid date '{date_yyyy_mm_dd}': {e}") from e
    start = _dt.datetime.combine(d, _dt.time.min)
    end = start + _dt.timedelta(days=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def _iso_from_ms(ms: int) -> str:
    import datetime as _dt
    try:
        return _dt.datetime.fromtimestamp(ms / 1000.0).isoformat(timespec="seconds")
    except Exception:
        return str(ms)


# ---------------------------------------------------------------------- singleton

_default_db: HistoryDB | None = None
_default_db_lock = threading.Lock()


def get_default_db() -> HistoryDB:
    """Return a module-level singleton pointed at DEFAULT_DB_PATH.

    Lazy so tests that never touch history don't create a file. The server
    imports this and calls it once from the first history endpoint or the
    ws_audio_stream hook.
    """
    global _default_db
    with _default_db_lock:
        if _default_db is None:
            _default_db = HistoryDB(DEFAULT_DB_PATH)
        return _default_db


def reset_default_db_for_testing(path: str | None = None) -> HistoryDB:
    """Swap the singleton (tests only). Returns the new instance."""
    global _default_db
    with _default_db_lock:
        if _default_db is not None:
            try:
                _default_db.close()
            except Exception:
                pass
        _default_db = HistoryDB(path or DEFAULT_DB_PATH)
        return _default_db
