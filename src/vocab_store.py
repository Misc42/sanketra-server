"""
JSON-backed vocabulary store for Sanketra v1.2 custom vocab.

Data model (per PRODUCT_VISION.html §6 and ANDROID_POLISH_V1.2.html §3.2):

    {
        "version": 3,
        "user_id": "<server-issued UUID>",
        "entries": [
            {"text": "Sanketra", "phonetic": "saa-ket-ra", "weight": 1.0},
            ...
        ],
        "last_modified": "2026-04-21T10:00:00Z"
    }

File location: ~/.config/sanketra/vocab.json. Writes are crash-safe via
write-to-temp + fsync + rename (same pattern as existing server_async.py
config handling). All writes hold a threading.Lock; reads do not, because
readers can tolerate the brief windows where the in-memory dict is being
rewritten — the file on disk is always a valid JSON snapshot.

The v1.2 spec uses this vocab to build Whisper `hotwords=` on each inference
call (§3.5). This module is storage-only; the Whisper integration lives in
the transcribe loop and imports `get_all()` when it needs the list.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any


DEFAULT_PATH = os.path.join(
    os.path.expanduser("~"), ".config", "sanketra", "vocab.json"
)

# Weight bounds per spec (§3.2): 0.0-2.0
MIN_WEIGHT = 0.0
MAX_WEIGHT = 2.0
DEFAULT_WEIGHT = 1.0
SCHEMA_VERSION = 3
# A modest cap prevents a misbehaving client from filling the disk or
# bloating Whisper's hotwords= beam search. 1k custom words is way more
# than any single user needs; the Android UI shows current count so users
# can self-prune.
MAX_ENTRIES = 1000


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sanitize_entry(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Coerce a raw client-supplied entry into our canonical shape.
    Returns None if the entry is unusable (empty text)."""
    if not isinstance(raw, dict):
        return None
    text = (raw.get("text") or "").strip()
    if not text:
        return None
    phonetic = raw.get("phonetic")
    if phonetic is not None:
        phonetic = str(phonetic).strip() or None
    try:
        weight = float(raw.get("weight", DEFAULT_WEIGHT))
    except (TypeError, ValueError):
        weight = DEFAULT_WEIGHT
    # Clamp without surfacing the rejection — client might send weight=10 by accident
    # and the result we apply is the clamped value. Silent clamp is OK because
    # weights above ~2.0 produce broken Whisper beam-search behaviour regardless.
    weight = max(MIN_WEIGHT, min(MAX_WEIGHT, weight))
    out = {"text": text, "weight": weight}
    if phonetic:
        out["phonetic"] = phonetic
    return out


class VocabStore:
    """Thread-safe JSON-backed vocab store."""

    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()
        existed = os.path.exists(path)
        self._data: dict[str, Any] = self._load()
        # Persist fresh state so the generated user_id survives subsequent reopens.
        # Without this, every process restart on an empty file would mint a new
        # UUID — and the Android client would see a "new user" after every reboot.
        if not existed:
            with self._lock:
                self._save()

    # ----------------------------------------------------------- persistence

    def _load(self) -> dict[str, Any]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    # Normalize on load — handles older (v2) files that may
                    # have malformed entries after an interrupted write.
                    data.setdefault("version", SCHEMA_VERSION)
                    data.setdefault("user_id", uuid.uuid4().hex)
                    data.setdefault("entries", [])
                    data.setdefault("last_modified", _now_iso())
                    clean: list[dict[str, Any]] = []
                    seen: set[str] = set()
                    for e in data.get("entries", []):
                        s = _sanitize_entry(e)
                        if s and s["text"] not in seen:
                            seen.add(s["text"])
                            clean.append(s)
                    data["entries"] = clean
                    return data
            except (json.JSONDecodeError, OSError):
                # Corrupt file — fall through to fresh state. We keep the broken
                # file in place (don't rm) so the user can rescue data manually
                # if it matters to them.
                pass
        return {
            "version": SCHEMA_VERSION,
            "user_id": uuid.uuid4().hex,
            "entries": [],
            "last_modified": _now_iso(),
        }

    def _save(self) -> None:
        """Atomic write: tmp + fsync + rename. Caller must hold self._lock."""
        self._data["last_modified"] = _now_iso()
        self._data["version"] = SCHEMA_VERSION
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # fsync may fail on tmpfs / test harnesses; that's OK — worst
                # case we lose the last write on an OS crash, same risk as any
                # other file write.
                pass
        os.replace(tmp_path, self.path)

    # ----------------------------------------------------------- queries

    def get_all(self) -> dict[str, Any]:
        """Return a deep copy of the whole vocab document."""
        with self._lock:
            return json.loads(json.dumps(self._data))

    def list_entries(self) -> list[dict[str, Any]]:
        """Return a copy of just the entries list (for Whisper hotwords=)."""
        with self._lock:
            return [dict(e) for e in self._data.get("entries", [])]

    # ----------------------------------------------------------- mutations

    def replace_all(self, entries: list[dict[str, Any]]) -> dict[str, Any]:
        """POST /api/vocab: replace full entries list (user_id stays)."""
        if not isinstance(entries, list):
            raise ValueError("entries must be a list")
        clean: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in entries[:MAX_ENTRIES]:
            s = _sanitize_entry(raw)
            if s and s["text"] not in seen:
                seen.add(s["text"])
                clean.append(s)
        with self._lock:
            self._data["entries"] = clean
            self._save()
            return json.loads(json.dumps(self._data))

    def add_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Idempotent add. Existing entry with same text is replaced (new weight
        / phonetic wins, because the user explicitly re-submitted it)."""
        s = _sanitize_entry(entry)
        if s is None:
            raise ValueError("entry has empty text")
        with self._lock:
            entries = self._data.get("entries", [])
            # Replace-if-exists semantics — this IS the "teach correct form"
            # path, and it needs to overwrite stale phonetic/weight from an
            # earlier teach cycle.
            existing_idx = next(
                (i for i, e in enumerate(entries) if e.get("text") == s["text"]),
                None,
            )
            if existing_idx is not None:
                entries[existing_idx] = s
            else:
                if len(entries) >= MAX_ENTRIES:
                    raise ValueError(f"vocab full (max {MAX_ENTRIES} entries)")
                entries.append(s)
            self._data["entries"] = entries
            self._save()
            return dict(s)

    def remove_entry(self, text: str) -> bool:
        """Remove by text. Returns True if removed, False if no match."""
        t = (text or "").strip()
        if not t:
            return False
        with self._lock:
            entries = self._data.get("entries", [])
            new_entries = [e for e in entries if e.get("text") != t]
            if len(new_entries) == len(entries):
                return False
            self._data["entries"] = new_entries
            self._save()
            return True

    def patch(self, add: list[dict[str, Any]] | None = None,
              remove: list[str] | None = None) -> dict[str, Any]:
        """PATCH /api/vocab: batch add+remove in one atomic write.

        Order: removals first, then additions. This matters for the edge
        case where the same text appears in both lists — the final state
        has it added (with possibly-new phonetic/weight), which is what
        the "edit entry" UI expects.
        """
        add = add or []
        remove = remove or []
        with self._lock:
            entries = self._data.get("entries", [])
            if remove:
                remove_set = {str(t).strip() for t in remove if t}
                entries = [e for e in entries if e.get("text") not in remove_set]
            if add:
                existing_by_text = {e.get("text"): i for i, e in enumerate(entries)}
                for raw in add:
                    s = _sanitize_entry(raw)
                    if s is None:
                        continue
                    if s["text"] in existing_by_text:
                        entries[existing_by_text[s["text"]]] = s
                    elif len(entries) < MAX_ENTRIES:
                        entries.append(s)
                        existing_by_text[s["text"]] = len(entries) - 1
            self._data["entries"] = entries
            self._save()
            return json.loads(json.dumps(self._data))


# ---------------------------------------------------------------- singleton

_default_store: VocabStore | None = None
_default_store_lock = threading.Lock()


def get_default_store() -> VocabStore:
    global _default_store
    with _default_store_lock:
        if _default_store is None:
            _default_store = VocabStore(DEFAULT_PATH)
        return _default_store


def reset_default_store_for_testing(path: str | None = None) -> VocabStore:
    global _default_store
    with _default_store_lock:
        _default_store = VocabStore(path or DEFAULT_PATH)
        return _default_store
