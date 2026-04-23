"""Offline license verifier for the Sanketra PC server.

Responsibilities:
- Load the bundled Ed25519 public key from `license/sanketra_signing_public.key`.
- Load the installed license key from `~/.config/sanketra/license.key` (if any).
- Expose `get_active_license()` and `has_track()` for feature gating.
- Write new license keys atomically (temp + fsync + os.replace) — partial
  writes must not corrupt an existing installed license.

Design notes:
- Strictly offline. No HTTP. No DNS. No network calls anywhere in this
  module — that's the invariant from MONETIZATION.md that makes the
  LAN-only privacy claim survive.
- Keep `server_async.py` diff small: that file imports this module and
  wires two HTTP endpoints. All the logic lives here.
- Cached — `get_active_license()` reads the file on first call, then
  returns the cached `License` for subsequent calls. Install flow
  (`install_license_key`) invalidates the cache on success. Call
  `reload()` explicitly from tests to force a re-read.

Error model:
- `load_public_key()` RAISES on missing pubkey — unrecoverable, a
  misconfigured PC-server install should fail loudly at startup rather
  than silently accepting unsigned licenses.
- `get_active_license()` NEVER raises. A missing / corrupt / tampered
  license file means "no active license" (return None). The PC server
  stays functional at the free tier.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

# ---- shared wire format ----------------------------------------------------
# The `license/` directory is deliberately not a Python package (its name
# collides with a stdlib namespace module). Import `license_format` via
# sys.path injection — same pattern `tests/test_auth_logic.py` uses.
_LICENSE_DIR = Path(__file__).resolve().parent.parent / "license"
if str(_LICENSE_DIR) not in sys.path:
    sys.path.insert(0, str(_LICENSE_DIR))

from license_format import License, verify_key  # noqa: E402


log = logging.getLogger("sanketra.license")


# --------------------------------------------------------------------------- #
# Paths — overridable via env for tests and for admins who want a non-default #
# install location.                                                           #
# --------------------------------------------------------------------------- #

def _default_public_key_path() -> Path:
    # Env override first so tests can swap the pubkey cleanly.
    env = os.environ.get("SANKETRA_PUBLIC_KEY_PATH")
    if env:
        return Path(env)
    return _LICENSE_DIR / "sanketra_signing_public.key"


def _default_license_path() -> Path:
    env = os.environ.get("SANKETRA_LICENSE_PATH")
    if env:
        return Path(env)
    return Path(os.path.expanduser("~/.config/sanketra/license.key"))


# --------------------------------------------------------------------------- #
# Public-key loader                                                           #
# --------------------------------------------------------------------------- #

_pubkey_lock = threading.Lock()
_pubkey_cache: Optional[ed25519.Ed25519PublicKey] = None
_pubkey_cache_path: Optional[Path] = None


def load_public_key(path: Optional[Path] = None) -> ed25519.Ed25519PublicKey:
    """Load and cache the bundled Ed25519 public key.

    Raises on missing / unreadable / wrong-algorithm key — an install
    whose pubkey bundle is broken is a misconfigured install, not a
    "just run free tier" situation."""
    global _pubkey_cache, _pubkey_cache_path
    target = path or _default_public_key_path()
    with _pubkey_lock:
        # Cache is keyed by path — if the env var flipped mid-test, reload.
        if _pubkey_cache is not None and _pubkey_cache_path == target:
            return _pubkey_cache
        if not target.is_file():
            raise FileNotFoundError(
                f"Sanketra public key not found at {target}. "
                f"Reinstall or set SANKETRA_PUBLIC_KEY_PATH."
            )
        pem = target.read_bytes()
        key = serialization.load_pem_public_key(pem)
        if not isinstance(key, ed25519.Ed25519PublicKey):
            raise ValueError(
                f"expected Ed25519 public key at {target}, "
                f"got {type(key).__name__}"
            )
        _pubkey_cache = key
        _pubkey_cache_path = target
        return key


# --------------------------------------------------------------------------- #
# License loader / cache                                                      #
# --------------------------------------------------------------------------- #

_license_lock = threading.Lock()
_license_cache: Optional[License] = None
_license_cache_loaded_from: Optional[Path] = None
_license_cache_valid: bool = False  # True once we've attempted a load


def _read_license_file(path: Path) -> Optional[str]:
    """Read the stored key-string, strip whitespace. Return None if absent."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as e:
        # Unreadable but present — log and treat as "no license".
        log.warning("license file at %s unreadable: %s", path, e)
        return None


def _try_verify(key_str: str, pub: ed25519.Ed25519PublicKey) -> Optional[License]:
    """Verify a raw key string. Return License on success, None on any
    failure (tampered, malformed, wrong signer, stale version)."""
    try:
        return verify_key(key_str, pub)
    except InvalidSignature:
        log.warning("installed license signature invalid — ignoring")
    except ValueError as e:
        log.warning("installed license malformed: %s", e)
    return None


def reload() -> Optional[License]:
    """Force-reread the license file from disk. Returns the active License
    or None. Resets the cache. Tests call this after mutating the file."""
    global _license_cache, _license_cache_valid, _license_cache_loaded_from
    path = _default_license_path()
    pub = load_public_key()
    with _license_lock:
        raw = _read_license_file(path)
        if raw is None:
            _license_cache = None
        else:
            _license_cache = _try_verify(raw, pub)
        _license_cache_valid = True
        _license_cache_loaded_from = path
        return _license_cache


def get_active_license() -> Optional[License]:
    """Return the active License, loading lazily on first call.

    NEVER raises. The whole point of this function is the server can
    call it from a hot path and get a definitive "None = free tier" on
    any failure mode."""
    global _license_cache, _license_cache_valid
    try:
        with _license_lock:
            if _license_cache_valid and _license_cache_loaded_from == _default_license_path():
                return _license_cache
        # First call OR path changed — do the full load outside the lock
        # so public-key loading (which takes its own lock) doesn't nest.
        return reload()
    except Exception as e:
        # Paranoid catch-all. The free-tier path must work even if the
        # license subsystem is on fire.
        log.exception("get_active_license failed (treating as free tier): %s", e)
        return None


def has_track(track: str) -> bool:
    """Does the active license cover `track`?

    - `track="phone"`  -> True if license.track in {"phone", "both"}
    - `track="desktop"` -> True if license.track in {"desktop", "both"}
    - `track="both"`    -> True only if license.track == "both"

    No license -> always False (free tier). The server's feature gates
    call `has_track(client_kind)` based on which client is connected."""
    lic = get_active_license()
    if lic is None:
        return False
    if track == "both":
        return lic.track == "both"
    if track in ("phone", "desktop"):
        return lic.track == track or lic.track == "both"
    # Unknown track string — refuse on principle, don't unlock by default.
    return False


# --------------------------------------------------------------------------- #
# Install flow — atomic write of the license file                             #
# --------------------------------------------------------------------------- #

class LicenseInstallError(ValueError):
    """Raised by install_license_key when the supplied key is rejected.

    Callers (the HTTP endpoint in server_async) translate this to 400."""


def install_license_key(key_str: str, path: Optional[Path] = None) -> License:
    """Verify + persist a new license key. Returns the `License` on success.

    Raises `LicenseInstallError` on any validation failure. On failure the
    destination file is NOT modified (we verify before we write).

    Write path is atomic:
        1. verify signature first
        2. mkdir -p parent directory
        3. write to `<path>.tmp`, fsync
        4. os.replace onto `<path>` (POSIX atomic rename on the same fs)
    A crash between steps 3 and 4 leaves `<path>` untouched — the next
    server startup loads the OLD license, not a half-written new one.
    """
    global _license_cache, _license_cache_valid

    if not isinstance(key_str, str) or not key_str.strip():
        raise LicenseInstallError("empty license key")

    key_str = key_str.strip()

    pub = load_public_key()
    try:
        lic = verify_key(key_str, pub)
    except InvalidSignature as e:
        raise LicenseInstallError("license signature invalid") from e
    except ValueError as e:
        raise LicenseInstallError(f"license malformed: {e}") from e

    dest = path or _default_license_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    # Plain text — the key is already armored (SKT-...). No wrapping.
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(key_str + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, dest)

    # Restrict perms to user-only on POSIX. The key isn't a secret (no
    # private-key material), but there's no reason for other local users
    # to see it either.
    try:
        os.chmod(dest, 0o600)
    except OSError:
        pass  # Windows / unusual fs — skip silently.

    # Reset cache so the next get_active_license() picks this up.
    with _license_lock:
        _license_cache = lic
        _license_cache_valid = True
        _license_cache_loaded_from = dest

    return lic


# --------------------------------------------------------------------------- #
# Test / reset helpers                                                        #
# --------------------------------------------------------------------------- #

def _reset_caches_for_tests() -> None:
    """Drop the in-memory caches. Tests call this between cases so env-var
    overrides take effect. Not part of the public API."""
    global _pubkey_cache, _pubkey_cache_path
    global _license_cache, _license_cache_valid, _license_cache_loaded_from
    with _pubkey_lock:
        _pubkey_cache = None
        _pubkey_cache_path = None
    with _license_lock:
        _license_cache = None
        _license_cache_valid = False
        _license_cache_loaded_from = None


__all__ = [
    "License",
    "LicenseInstallError",
    "load_public_key",
    "reload",
    "get_active_license",
    "has_track",
    "install_license_key",
]
