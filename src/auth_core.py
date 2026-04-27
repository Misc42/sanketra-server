"""Pure auth/rate-limit primitives — extracted from server_async.py so tests
can exercise the real production code instead of reimplementations.

Background: tests/test_auth_logic.py used to ship its own copies of
`check_auth_rate_limit` and `verify_session`. The reimplementation diverged
silently — the test version filtered attempts within `_AUTH_RATE_WINDOW`
(60s) while the production version filtered within `_AUTH_LOCKOUT` (600s).
Tests passed against fictional code. Pure-function module + thin wrappers
in server_async fixes both sides at once.

Constants live here so tests, server, and any future module share one
source of truth.
"""

from __future__ import annotations

import secrets
import time as _time
from typing import Optional

# ============================================================================
# Constants — single source for both server and tests.
# ============================================================================

PAIR_RATE_LIMIT: int = 5
PAIR_RATE_WINDOW: float = 60.0     # seconds

AUTH_RATE_LIMIT: int = 3
AUTH_RATE_WINDOW: float = 60.0     # seconds — used for "burst" detection
AUTH_LOCKOUT: float = 600.0        # seconds — once tripped, IP locked this long

SESSION_TTL: float = 30.0 * 24.0 * 3600.0  # 30 days


# ============================================================================
# Rate limiting
# ============================================================================

def is_pair_rate_limited(attempts: list, now: Optional[float] = None) -> bool:
    """Return True if the IP has hit the pair rate limit in the recent window.

    `attempts`: list of UNIX timestamps of past pair attempts for this IP.
    Caller is responsible for storing the list per-IP.
    """
    if now is None:
        now = _time.time()
    recent = [t for t in attempts if now - t < PAIR_RATE_WINDOW]
    return len(recent) >= PAIR_RATE_LIMIT


def is_auth_rate_limited(attempts: list, now: Optional[float] = None) -> bool:
    """Return True if the IP is currently in auth lockout.

    Filter window is `AUTH_LOCKOUT` (600 s), not `AUTH_RATE_WINDOW` (60 s) —
    once an IP exceeds `AUTH_RATE_LIMIT` failures, it stays locked out for
    the full lockout period from the last failure. Filtering by the shorter
    burst window would let attackers wait 60 s and resume — defeating the
    purpose of the lockout.
    """
    if now is None:
        now = _time.time()
    within_lockout = [t for t in attempts if now - t < AUTH_LOCKOUT]
    return len(within_lockout) >= AUTH_RATE_LIMIT


def prune_auth_attempts(attempts: list, now: Optional[float] = None) -> list:
    """Drop attempts older than the lockout window. Caller stores the result."""
    if now is None:
        now = _time.time()
    return [t for t in attempts if now - t < AUTH_LOCKOUT]


# ============================================================================
# Session token verification
# ============================================================================

def is_session_expired(session: dict, now: Optional[float] = None) -> bool:
    """A session dict is expired when its `created` timestamp + SESSION_TTL is
    in the past. Missing `created` defaults to 0 (expired)."""
    if now is None:
        now = _time.time()
    created = session.get("created", 0)
    return (now - created) > SESSION_TTL


def verify_session_token(
    token: str,
    master_token: str,
    sessions: dict,
    now: Optional[float] = None,
) -> bool:
    """Pure session verification.

    - Empty token → False.
    - Token equals `master_token` (the TOFU pair-time AUTH_TOKEN) → True
      (constant-time comparison via secrets.compare_digest).
    - Token matches an unexpired entry in `sessions` → True.
      We iterate `sessions` and compare via `secrets.compare_digest`
      rather than `sessions.get(token)` because dict lookup leaks the
      length of the longest shared prefix via PyHash timing. A targeted
      attacker who can submit O(n) probes against a paired client could
      reconstruct the token byte-by-byte. O(n) iteration costs us
      effectively nothing — `n` is the number of paired clients (a
      handful), and each comparison is a fixed-time 32-byte memcmp.
    - Otherwise False.

    Does NOT mutate `sessions`. Caller deletes expired entries if it wants;
    that mutation lives in the server wrapper since it has the lock.
    """
    if not token:
        return False
    if master_token and secrets.compare_digest(token, master_token):
        return True
    for stored_token, session in sessions.items():
        if not isinstance(stored_token, str):
            continue
        if not secrets.compare_digest(stored_token, token):
            continue
        if is_session_expired(session, now):
            return False
        return True
    return False
