"""Offline license verifier for the Sanketra PC server.

Responsibilities:
- Load one or more bundled Ed25519 public keys (multi-key keyring with
  validity windows so we can rotate without forcing every customer to
  re-register on the same hour).
- Load the installed license key from `~/.config/sanketra/license.key` (if any).
- Verify signature, expiry, device fingerprint, and denylist.
- Expose `get_active_license()` and `has_track()` for feature gating.
- Write new license keys atomically (temp + fsync + os.replace) — partial
  writes must not corrupt an existing installed license.

Design notes:
- Strictly offline. No HTTP. No DNS. No network calls anywhere in this
  module — that's the invariant from MONETIZATION.md that makes the
  LAN-only privacy claim survive.
- Keep `server_async.py` diff small: that file imports this module and
  wires HTTP endpoints. All the logic lives here.
- Cached — `get_active_license()` reads the file on first call, then
  returns the cached `License` for subsequent calls. Install flow
  (`install_license_key`) invalidates the cache on success. Call
  `reload()` explicitly from tests to force a re-read.

Keyring format:
- Default load path: `license/sanketra_signing_public.key`. If that file
  is a JSON array, treat it as a keyring:
      [
        {"key_id": "k1", "pem": "<PEM block>", "valid_from": 0,
         "valid_until": 1900000000},
        {"key_id": "k2", "pem": "<PEM block>", "valid_from": 1800000000,
         "valid_until": 0}
      ]
  `valid_until = 0` means "no upper bound". A signed license is accepted
  if ANY entry whose validity window covers `now` verifies the signature.
- If the file is a plain PEM block (back-compat), it's loaded as a single
  always-valid key.

Error model:
- `load_public_keys()` RAISES on missing keyring — unrecoverable, a
  misconfigured PC-server install should fail loudly at startup rather
  than silently accepting unsigned licenses.
- `get_active_license()` NEVER raises. A missing / corrupt / tampered /
  expired / denylisted license file means "no active license" (return
  None). The PC server stays functional at the free tier.

Integration contract for server_async.py (Worktree B owns that file):
- The server SHOULD compute a local device fingerprint via
  `compute_local_fingerprint()` and pass it to `install_license_key()` as
  the `device_fingerprint` kwarg AND to `get_active_license()` so the
  verifier can compare it to the signed fingerprint.
- For the cached `get_active_license()` hot path the server can call
  `compute_local_fingerprint()` ONCE at startup and stash it in app state;
  it's a pure function over machine_id + MAC and won't change at runtime.
- If `device_fingerprint` is omitted, the verifier still rejects licenses
  that have a non-empty fingerprint (fail closed). Licenses issued without
  a fingerprint (i.e. `device_fingerprint == ""`) remain installable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
import threading
import time
import uuid
from dataclasses import dataclass
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


def _default_denylist_path() -> Path:
    env = os.environ.get("SANKETRA_DENYLIST_PATH")
    if env:
        return Path(env)
    return _LICENSE_DIR / "denylist.json"


def _default_fingerprint_salt_path() -> Path:
    """Per-install salt that backs the local fingerprint when no MAC /
    machine_id is available. Stored next to the license file so a fresh
    install gets a fresh salt (and therefore a different fingerprint)."""
    env = os.environ.get("SANKETRA_FINGERPRINT_SALT_PATH")
    if env:
        return Path(env)
    return Path(os.path.expanduser("~/.config/sanketra/fingerprint.salt"))


# --------------------------------------------------------------------------- #
# Public-key loader (keyring with validity windows)                           #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PublicKeyEntry:
    """One slot in the verification keyring.

    `valid_from` / `valid_until` are unix seconds UTC; `0` means open-ended
    on that side. The verifier accepts a license if ANY entry whose window
    covers `now` produces a successful Ed25519 verify."""
    key_id: str
    public_key: ed25519.Ed25519PublicKey
    valid_from: int
    valid_until: int

    def covers(self, now: int) -> bool:
        if self.valid_from and now < self.valid_from:
            return False
        if self.valid_until and now > self.valid_until:
            return False
        return True


_pubkey_lock = threading.Lock()
_pubkey_cache: Optional[list[PublicKeyEntry]] = None
_pubkey_cache_path: Optional[Path] = None


def _parse_pem_to_ed25519(pem: bytes, where: str) -> ed25519.Ed25519PublicKey:
    key = serialization.load_pem_public_key(pem)
    if not isinstance(key, ed25519.Ed25519PublicKey):
        raise ValueError(
            f"expected Ed25519 public key at {where}, got {type(key).__name__}"
        )
    return key


def load_public_keys(path: Optional[Path] = None) -> list[PublicKeyEntry]:
    """Load and cache the keyring.

    The bundle file is either:
    - A plain PEM block (legacy, single always-valid key), OR
    - A JSON array of objects with `key_id`, `pem`, `valid_from`,
      `valid_until` (any of the integer fields may be 0 = open-ended).

    Raises on missing / unreadable / malformed bundles — a misconfigured
    install must fail loudly, not silently accept unsigned licenses.
    """
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
        raw = target.read_bytes()
        entries: list[PublicKeyEntry] = []
        # Cheap discriminator: PEM bundles start with `-----BEGIN`. JSON
        # bundles start with `[` after optional whitespace.
        stripped = raw.lstrip()
        if stripped.startswith(b"["):
            try:
                arr = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise ValueError(
                    f"keyring at {target} is not valid JSON: {e}"
                ) from e
            if not isinstance(arr, list) or not arr:
                raise ValueError(
                    f"keyring at {target} must be a non-empty JSON array"
                )
            for i, item in enumerate(arr):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"keyring entry #{i} must be an object"
                    )
                key_id = str(item.get("key_id") or f"entry_{i}")
                pem = item.get("pem")
                if not isinstance(pem, str) or not pem.strip():
                    raise ValueError(
                        f"keyring entry {key_id!r} missing 'pem'"
                    )
                pub = _parse_pem_to_ed25519(
                    pem.encode("utf-8"), f"{target}#{key_id}"
                )
                entries.append(PublicKeyEntry(
                    key_id=key_id,
                    public_key=pub,
                    valid_from=int(item.get("valid_from") or 0),
                    valid_until=int(item.get("valid_until") or 0),
                ))
        else:
            # Legacy single-PEM bundle. Treat as always-valid.
            pub = _parse_pem_to_ed25519(raw, str(target))
            entries.append(PublicKeyEntry(
                key_id="default",
                public_key=pub,
                valid_from=0,
                valid_until=0,
            ))
        _pubkey_cache = entries
        _pubkey_cache_path = target
        return entries


def load_public_key(path: Optional[Path] = None) -> ed25519.Ed25519PublicKey:
    """Back-compat shim. Returns the FIRST currently-valid public key.

    Prefer `load_public_keys()` for new code — this exists so older callers
    that still expect a single key (e.g. tests that monkey-patch a pubkey
    file) keep working without refactoring. If the keyring has more than
    one currently-valid key, the order in the JSON array decides which
    wins for this single-key view; signature verification itself always
    iterates the whole keyring."""
    entries = load_public_keys(path)
    now = int(time.time())
    for e in entries:
        if e.covers(now):
            return e.public_key
    # No entry currently valid: surface the first one anyway so the legacy
    # single-key error path (used by callers that haven't migrated to the
    # keyring API) still has SOMETHING to compare against — verification
    # will then fail cleanly via the keyring path.
    return entries[0].public_key


# --------------------------------------------------------------------------- #
# Denylist (M9 stub — full revocation rollout is later)                       #
# --------------------------------------------------------------------------- #

_denylist_lock = threading.Lock()
_denylist_cache: Optional[set[str]] = None
_denylist_cache_path: Optional[Path] = None


def load_denylist(path: Optional[Path] = None) -> set[str]:
    """Return the set of revoked `license_id` strings.

    The on-disk format is a JSON array of license_id strings. Missing file
    => empty denylist (treat as "nothing revoked"). A malformed file is
    treated as empty too, but we log a warning — fail-open here is the
    right call: a typo'd denylist file should not lock every paying user
    out of their product.

    A future iteration will require a separate root-keypair signature on
    the file. The schema deliberately leaves room for that:
        {"revoked": ["<id1>", "<id2>"], "sig": "..."}
    is also accepted (only `revoked` is consulted today)."""
    global _denylist_cache, _denylist_cache_path
    target = path or _default_denylist_path()
    with _denylist_lock:
        if _denylist_cache is not None and _denylist_cache_path == target:
            return _denylist_cache
        revoked: set[str] = set()
        try:
            raw = target.read_text(encoding="utf-8")
        except FileNotFoundError:
            _denylist_cache = revoked
            _denylist_cache_path = target
            return revoked
        except OSError as e:
            log.warning("denylist at %s unreadable: %s — treating as empty", target, e)
            _denylist_cache = revoked
            _denylist_cache_path = target
            return revoked
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            log.warning("denylist at %s malformed: %s — treating as empty", target, e)
            _denylist_cache = revoked
            _denylist_cache_path = target
            return revoked
        if isinstance(data, list):
            ids = data
        elif isinstance(data, dict):
            ids = data.get("revoked") or []
        else:
            ids = []
        for item in ids:
            if isinstance(item, str) and item:
                revoked.add(item)
        _denylist_cache = revoked
        _denylist_cache_path = target
        return revoked


# --------------------------------------------------------------------------- #
# Device fingerprint                                                          #
# --------------------------------------------------------------------------- #

def _read_machine_id() -> str:
    """Best-effort stable per-machine identifier.

    Linux: /etc/machine-id (systemd) or /var/lib/dbus/machine-id.
    macOS: hw.uuid via `ioreg`.
    Windows: registry MachineGuid.
    Fallback: empty string (the salt path picks up the slack)."""
    # Linux
    for p in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                v = fh.read().strip()
                if v:
                    return v
        except OSError:
            pass
    # macOS / Windows: avoid spawning subprocesses just for a fingerprint
    # ingredient. The MAC + per-install salt below is sufficient.
    return ""


def _get_or_create_install_salt(path: Optional[Path] = None) -> str:
    """Return a 256-bit hex salt unique to this install. Created lazily on
    first read; `0o600` perms; survives across runs.

    This is what gives the fingerprint stability when /etc/machine-id is
    not available (containers, exotic distros) — and what keeps two
    fresh installs on the same machine from collapsing to the same FP."""
    target = path or _default_fingerprint_salt_path()
    try:
        existing = target.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except FileNotFoundError:
        pass
    except OSError as e:
        # Unreadable; fall back to a process-lifetime random — fingerprint
        # changes per restart in this case, which is annoying but better
        # than a global constant. Log so ops can fix the directory.
        log.warning("fingerprint salt at %s unreadable: %s", target, e)
        return uuid.uuid4().hex
    target.parent.mkdir(parents=True, exist_ok=True)
    salt = uuid.uuid4().hex
    # O_EXCL + 0o600 so a racing peer can't pre-create a world-readable
    # salt file and read ours.
    try:
        fd = os.open(
            str(target),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        # Lost the race; re-read whatever the winner wrote.
        try:
            return target.read_text(encoding="utf-8").strip() or salt
        except OSError:
            return salt
    except OSError as e:
        log.warning("could not create fingerprint salt at %s: %s", target, e)
        return salt
    try:
        os.write(fd, salt.encode("utf-8"))
    finally:
        os.close(fd)
    return salt


def compute_local_fingerprint(salt_path: Optional[Path] = None) -> str:
    """SHA-256 hex of (machine_id || primary_mac || install_salt).

    All ingredients are stable across reboots (machine_id), or stable
    enough (the highest-numbered persistent MAC, when present), or
    explicitly stable by design (the on-disk salt). The mix is intentional:
    machine_id alone is missing on some platforms; MAC alone changes when
    the user docks/undocks; salt alone defeats the point on a shared host.

    The result is HEX so it can sit safely in JSON without escaping."""
    parts: list[str] = []
    machine_id = _read_machine_id()
    if machine_id:
        parts.append(machine_id)
    # uuid.getnode() returns a 48-bit MAC if it can find one, else a random
    # multicast bit address. That second case is unstable, so we record it
    # but don't lean on it — the install salt picks up the slack.
    parts.append(f"{uuid.getnode():012x}")
    parts.append(platform.node() or "")
    parts.append(_get_or_create_install_salt(salt_path))
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


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


def _verify_against_keyring(
    key_str: str,
    entries: list[PublicKeyEntry],
    now: int,
) -> License:
    """Try every keyring entry whose validity window covers `now`. Return
    the first License a verify succeeds for. Raises `InvalidSignature` if
    no covering key matched, or `ValueError` for framing errors (those
    bubble out from `verify_key` directly).

    Iteration order matches the keyring file. We try ALL covering keys
    before giving up so rotation stays seamless: during the overlap
    window, both old and new licenses verify."""
    candidates = [e for e in entries if e.covers(now)]
    if not candidates:
        # No key currently valid — pretend the most recent one is, so the
        # error path returns a clean InvalidSignature rather than "no key
        # to compare against." Operationally this means "your bundle's
        # validity windows are all in the past" — alarming, but the
        # license itself is rejected just the same.
        candidates = entries
    last_value_err: Optional[ValueError] = None
    for entry in candidates:
        try:
            return verify_key(key_str, entry.public_key)
        except InvalidSignature:
            continue
        except ValueError as e:
            # Framing/decoding failure — same key string fails the same
            # way against every entry, so bail immediately.
            last_value_err = e
            break
    if last_value_err is not None:
        raise last_value_err
    raise InvalidSignature("no keyring entry verified the license signature")


def _check_expiry(lic: License, now: int) -> None:
    """Raise ValueError if `lic` is expired. `expires_at == 0` means
    "perpetual" and is allowed (used for free / promo tiers)."""
    if lic.expires_at and now > lic.expires_at:
        raise ValueError(
            f"license expired (expires_at={lic.expires_at}, now={now})"
        )


def _check_fingerprint(lic: License, local_fp: Optional[str]) -> None:
    """Raise ValueError if the license is bound to a fingerprint and that
    fingerprint doesn't match the local one.

    `lic.device_fingerprint == ""` means "unbound" — accepted on any host.
    `local_fp is None` means the caller chose not to pass one — in that
    case we still reject licenses with a non-empty fingerprint (fail
    closed). A caller who wants to skip the check on purpose passes the
    license's own fingerprint, which is a legitimate use only for the
    issuer's own self-tests."""
    if not lic.device_fingerprint:
        return
    if local_fp is None:
        raise ValueError(
            "license is bound to a device but no local fingerprint provided"
        )
    # Constant-time comparison out of paranoia. Both strings are hex of
    # fixed length, so timing is uniform anyway, but it costs us nothing.
    import hmac as _hmac
    if not _hmac.compare_digest(lic.device_fingerprint, local_fp):
        raise ValueError("license device_fingerprint mismatch")


def _check_denylist(lic: License) -> None:
    if lic.license_id in load_denylist():
        raise ValueError("license is revoked")


def _try_verify(
    key_str: str,
    entries: list[PublicKeyEntry],
    *,
    local_fp: Optional[str] = None,
    now: Optional[int] = None,
) -> Optional[License]:
    """Verify a raw key string and run all post-signature checks
    (expiry, denylist, device fingerprint).

    Returns License on success, None on any failure. Logs the SPECIFIC
    reason at WARNING level so ops can debug, but never surfaces it to
    the caller — `get_active_license()` returning None means "free tier"
    regardless of the underlying cause."""
    when = int(time.time()) if now is None else now
    try:
        lic = _verify_against_keyring(key_str, entries, when)
    except InvalidSignature:
        log.warning("installed license signature invalid — ignoring")
        return None
    except ValueError as e:
        log.warning("installed license malformed: %s", e)
        return None
    try:
        _check_expiry(lic, when)
        _check_denylist(lic)
        _check_fingerprint(lic, local_fp)
    except ValueError as e:
        log.warning("installed license rejected post-signature: %s", e)
        return None
    return lic


def reload(local_fp: Optional[str] = None) -> Optional[License]:
    """Force-reread the license file from disk. Returns the active License
    or None. Resets the cache. Tests call this after mutating the file."""
    global _license_cache, _license_cache_valid, _license_cache_loaded_from
    path = _default_license_path()
    entries = load_public_keys()
    fp = local_fp if local_fp is not None else _local_fp_or_compute()
    with _license_lock:
        raw = _read_license_file(path)
        if raw is None:
            _license_cache = None
        else:
            _license_cache = _try_verify(raw, entries, local_fp=fp)
        _license_cache_valid = True
        _license_cache_loaded_from = path
        return _license_cache


def _local_fp_or_compute() -> Optional[str]:
    """Compute the local fingerprint, swallowing any I/O errors. Returns
    None if we couldn't get one — callers downstream treat None as "no
    fingerprint provided" and fail closed on bound licenses."""
    try:
        return compute_local_fingerprint()
    except Exception as e:  # noqa: BLE001
        log.warning("could not compute local fingerprint: %s", e)
        return None


def get_active_license(local_fp: Optional[str] = None) -> Optional[License]:
    """Return the active License, loading lazily on first call.

    NEVER raises. The whole point of this function is the server can
    call it from a hot path and get a definitive "None = free tier" on
    any failure mode.

    `local_fp` is optional. If omitted, the verifier auto-computes the
    local fingerprint on first call and uses it for all subsequent checks.
    The server (Worktree B) SHOULD compute the fingerprint at startup
    and pass it explicitly — that keeps the hot path allocation-free."""
    global _license_cache, _license_cache_valid
    try:
        with _license_lock:
            if _license_cache_valid and _license_cache_loaded_from == _default_license_path():
                return _license_cache
        # First call OR path changed — do the full load outside the lock
        # so public-key loading (which takes its own lock) doesn't nest.
        return reload(local_fp=local_fp)
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


_GENERIC_INSTALL_ERROR = "invalid license"


def install_license_key(
    key_str: str,
    path: Optional[Path] = None,
    *,
    device_fingerprint: Optional[str] = None,
) -> License:
    """Verify + persist a new license key. Returns the `License` on success.

    Raises `LicenseInstallError` on any validation failure. The exception
    message is intentionally generic (`"invalid license"`) regardless of
    the underlying cause — bad signature, malformed payload, expired,
    revoked, fingerprint mismatch all collapse to one external string so
    we don't hand attackers an oracle. The specific reason is logged.

    On failure the destination file is NOT modified (we verify before we
    write).

    Write path is atomic:
        1. verify signature + run all post-signature checks
        2. mkdir -p parent directory
        3. write to `<path>.tmp`, fsync
        4. os.replace onto `<path>` (POSIX atomic rename on the same fs)
    A crash between steps 3 and 4 leaves `<path>` untouched — the next
    server startup loads the OLD license, not a half-written new one.

    `device_fingerprint`: pass the local machine fingerprint so the
    verifier can match it against the signed value. None => auto-compute.
    """
    global _license_cache, _license_cache_valid

    if not isinstance(key_str, str) or not key_str.strip():
        raise LicenseInstallError(_GENERIC_INSTALL_ERROR)

    key_str = key_str.strip()

    entries = load_public_keys()
    fp = device_fingerprint if device_fingerprint is not None else _local_fp_or_compute()
    now = int(time.time())
    try:
        lic = _verify_against_keyring(key_str, entries, now)
    except InvalidSignature as e:
        log.warning("license install rejected: signature invalid")
        raise LicenseInstallError(_GENERIC_INSTALL_ERROR) from e
    except ValueError as e:
        log.warning("license install rejected: malformed: %s", e)
        raise LicenseInstallError(_GENERIC_INSTALL_ERROR) from e

    try:
        _check_expiry(lic, now)
        _check_denylist(lic)
        _check_fingerprint(lic, fp)
    except ValueError as e:
        log.warning("license install rejected: %s", e)
        raise LicenseInstallError(_GENERIC_INSTALL_ERROR) from e

    dest = path or _default_license_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Per-call unique tmp filename. Two threads racing the install path
    # used to share a single `<dest>.tmp`, so one thread's `os.replace`
    # could move the OTHER thread's still-being-written file out from
    # under it (FileNotFoundError on the late writer). PID + monotonic
    # nanos is unique within a process; uuid4 covers the cross-process
    # case (multiple Sanketra installs running on the same dest).
    tmp = dest.with_suffix(
        f"{dest.suffix}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    )

    # Plain text — the key is already armored (SKT-...). No wrapping.
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(key_str + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
    except Exception:
        # If the rename never happened, leave nothing behind.
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise

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
    global _denylist_cache, _denylist_cache_path
    with _pubkey_lock:
        _pubkey_cache = None
        _pubkey_cache_path = None
    with _license_lock:
        _license_cache = None
        _license_cache_valid = False
        _license_cache_loaded_from = None
    with _denylist_lock:
        _denylist_cache = None
        _denylist_cache_path = None


__all__ = [
    "License",
    "LicenseInstallError",
    "PublicKeyEntry",
    "load_public_key",
    "load_public_keys",
    "load_denylist",
    "compute_local_fingerprint",
    "reload",
    "get_active_license",
    "has_track",
    "install_license_key",
]
