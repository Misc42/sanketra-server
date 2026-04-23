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
    - Token in `sessions` and not expired → True.
    - Otherwise False.

    Does NOT mutate `sessions`. Caller deletes expired entries if it wants;
    that mutation lives in the server wrapper since it has the lock.
    """
    if not token:
        return False
    if master_token and secrets.compare_digest(token, master_token):
        return True
    session = sessions.get(token)
    if session and not is_session_expired(session, now):
        return True
    return False
