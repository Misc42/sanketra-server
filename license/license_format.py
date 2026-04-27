"""Sanketra license wire format — shared by the issuer and the PC-server verifier.

A license is a small signed blob that the PC server checks OFFLINE. No
network calls during verification — that preserves the LAN-only privacy
claim in `MONETIZATION.md`.

Payload shape
-------------
Canonical JSON with these fields (see `License` below):

    {
      "license_id":         "<uuid4>",
      "email":              "<buyer email>",
      "track":              "phone" | "desktop" | "both",
      "issued_at":          <unix seconds>,
      "expires_at":         <unix seconds> | 0,
      "device_fingerprint": "<hex digest>" | "",
      "v":                  2
    }

`expires_at = 0` means "perpetual" — used only for free / promo tiers.
Paid tracks (phone / desktop / both) MUST have a non-zero
`expires_at`; the issuer is responsible for setting it.

`device_fingerprint = ""` means "unbound" — license can be installed on
any machine. Paid tracks SHOULD have a fingerprint; the issuer accepts
the buyer's fingerprint at purchase time and signs it into the payload.

Wire format
-----------
A license key on the wire looks like this:

    SKT-XXXX-XXXX-...-XXXX

- Prefix is the literal string `SKT-`.
- Everything after `SKT-` is base32 (RFC 4648, uppercase, `=` padding
  stripped) of the following binary blob, then chunked into 4-char
  groups separated by `-`:

    +---------+----------------+---------+
    | version | payload_json   | sig     |
    | 1 byte  | varlen (uint16 | 64 byte |
    |         | big-endian LE) | Ed25519 |
    +---------+----------------+---------+

- `version` is `0x01`. Future versions bump this (e.g. for adding an
  expires_at field or a revocation epoch).
- `payload_json` is the canonical JSON of the `License` dataclass as
  bytes, prefixed by its length as big-endian uint16.
- `sig` is a raw 64-byte Ed25519 signature over
  `sha256(version_byte || len_prefix || payload_json)`.
  (Hashing first matches what `MONETIZATION.md` specifies; Ed25519
  itself also hashes internally, so the extra SHA-256 pass is purely
  for spec compliance — it does not weaken security.)

Why base32 (not base64)?
- Case-insensitive and telephone/email-safe — users will paste these
  into a CLI, or worst case read them out loud to support. `I/O/0/1`
  ambiguity is annoying but still better than `+/=/l/I` in base64.

Key material
------------
- Signing is Ed25519 (`cryptography.hazmat.primitives.asymmetric.ed25519`).
- Private key lives ONLY on the issuer host. Never in this repo.
- Public key bundles with the PC server (`license/sanketra_signing_public.key`).

Compatibility
-------------
Bumping the version byte will be read by `parse_key()` and rejected
cleanly so older servers don't silently misread newer keys. The issuer
and verifier share this file exactly — copy it into the server bundle,
don't fork it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import struct
from dataclasses import asdict, dataclass
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519


# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

WIRE_VERSION: int = 2
KEY_PREFIX: str = "SKT-"

# Maximum size of the canonical-JSON payload before we refuse to verify.
# Defense against pathological keys: a 1 MB email field would balloon the
# blob and force the verifier through a full SHA-256 + Ed25519 check on
# attacker-controlled bytes. The largest legitimate payload is well under
# 1 KiB (uuid + email + track + two ints + fingerprint hex). Cap well above
# that to absorb future fields, but well below the 64 KiB wire limit so a
# malicious buyer can't stuff junk into `email` or `device_fingerprint`.
MAX_PAYLOAD_BYTES: int = 4096
# 4-char base32 chunks so the key reads like a serial number. Users paste
# them into a text box; chunking is purely cosmetic — parse_key() strips
# hyphens before decoding.
CHUNK_SIZE: int = 4

Track = Literal["phone", "desktop", "both"]
VALID_TRACKS: tuple[Track, ...] = ("phone", "desktop", "both")


# --------------------------------------------------------------------------- #
# Dataclass                                                                   #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class License:
    """Signed payload. Fields here become wire-format fields — ANY field
    rename/addition requires a `WIRE_VERSION` bump so old servers reject
    new keys instead of silently misinterpreting them.

    `expires_at = 0` means "no expiry" (used for free / promo tiers).
    Paid tracks must have a non-zero, future expiry — the issuer is
    responsible for that; the verifier enforces it.

    `device_fingerprint = ""` means "unbound to a machine". The verifier
    skips the fingerprint check in that case. A non-empty fingerprint is
    compared byte-for-byte against the locally computed fingerprint."""

    license_id: str      # uuid4 hex, e.g. "f3a1b2c3..."
    email: str           # buyer's email — lower-cased before signing
    track: Track         # "phone" | "desktop" | "both"
    issued_at: int       # unix seconds UTC
    expires_at: int = 0          # 0 = perpetual; non-zero = unix seconds UTC
    device_fingerprint: str = ""  # "" = unbound; otherwise hex digest

    def __post_init__(self) -> None:
        if self.track not in VALID_TRACKS:
            raise ValueError(
                f"track must be one of {VALID_TRACKS}, got {self.track!r}"
            )
        if not self.email or "@" not in self.email:
            raise ValueError(f"invalid email: {self.email!r}")
        if not self.license_id:
            raise ValueError("license_id required")
        if self.issued_at <= 0:
            raise ValueError(f"issued_at must be positive, got {self.issued_at}")
        if self.expires_at < 0:
            raise ValueError(
                f"expires_at must be non-negative, got {self.expires_at}"
            )
        if self.expires_at and self.expires_at <= self.issued_at:
            raise ValueError(
                f"expires_at ({self.expires_at}) must be > "
                f"issued_at ({self.issued_at})"
            )
        if not isinstance(self.device_fingerprint, str):
            raise ValueError(
                f"device_fingerprint must be a string, "
                f"got {type(self.device_fingerprint).__name__}"
            )

    # --- serialization -----------------------------------------------------

    def to_canonical_json(self) -> bytes:
        """Byte-identical JSON every time, across Python versions / OSes.

        `sort_keys=True` + tight separators + version tag. This is what we
        sign and what we verify — both sides must produce the exact same
        bytes or the signature check fails."""
        payload = asdict(self)
        payload["v"] = WIRE_VERSION
        # Normalize email so a stray capital letter doesn't invalidate an
        # otherwise-valid license. Defense against case drift between
        # Razorpay (which may hand us the email as typed) and whatever the
        # user pastes later.
        payload["email"] = self.email.strip().lower()
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    @classmethod
    def from_canonical_json(cls, data: bytes) -> "License":
        obj = json.loads(data.decode("utf-8"))
        if obj.get("v") != WIRE_VERSION:
            raise ValueError(
                f"unsupported license version: {obj.get('v')!r} "
                f"(this server understands v{WIRE_VERSION})"
            )
        # `expires_at` and `device_fingerprint` are required-by-schema in
        # v2. Reject silently-missing fields rather than defaulting — a
        # v2 license without an expiry is not a thing this server should
        # accept just because the JSON happened to omit the key.
        if "expires_at" not in obj:
            raise ValueError("v2 license missing required field 'expires_at'")
        if "device_fingerprint" not in obj:
            raise ValueError(
                "v2 license missing required field 'device_fingerprint'"
            )
        return cls(
            license_id=obj["license_id"],
            email=obj["email"],
            track=obj["track"],
            issued_at=int(obj["issued_at"]),
            expires_at=int(obj["expires_at"]),
            device_fingerprint=str(obj["device_fingerprint"]),
        )


# --------------------------------------------------------------------------- #
# Binary blob <-> wire string                                                 #
# --------------------------------------------------------------------------- #

def _sign_digest(payload_bytes: bytes) -> bytes:
    """What we actually sign — sha256(version || len || payload)."""
    h = hashlib.sha256()
    h.update(bytes([WIRE_VERSION]))
    h.update(struct.pack(">H", len(payload_bytes)))
    h.update(payload_bytes)
    return h.digest()


def _pack_blob(payload_bytes: bytes, signature: bytes) -> bytes:
    if len(payload_bytes) > 0xFFFF:
        raise ValueError(
            f"payload too large: {len(payload_bytes)} bytes (max 65535)"
        )
    if len(signature) != 64:
        raise ValueError(
            f"expected 64-byte Ed25519 signature, got {len(signature)}"
        )
    return (
        bytes([WIRE_VERSION])
        + struct.pack(">H", len(payload_bytes))
        + payload_bytes
        + signature
    )


def _unpack_blob(blob: bytes) -> tuple[bytes, bytes]:
    """Return (payload_bytes, signature). Raises ValueError on any framing
    error — this is a trust boundary, so we are paranoid about lengths."""
    if len(blob) < 1 + 2 + 64:
        raise ValueError(f"blob too short: {len(blob)} bytes")
    ver = blob[0]
    if ver != WIRE_VERSION:
        raise ValueError(
            f"unsupported wire version: {ver} (expected {WIRE_VERSION})"
        )
    payload_len = struct.unpack(">H", blob[1:3])[0]
    if payload_len > MAX_PAYLOAD_BYTES:
        # Reject before we hash + verify. An attacker who could trick the
        # verifier into running Ed25519 over megabytes of data per request
        # has a cheap CPU-DoS primitive. Cap it.
        raise ValueError(
            f"payload too large: {payload_len} bytes "
            f"(max {MAX_PAYLOAD_BYTES})"
        )
    expected = 1 + 2 + payload_len + 64
    if len(blob) != expected:
        raise ValueError(
            f"blob length mismatch: {len(blob)} bytes, expected {expected}"
        )
    payload_bytes = blob[3 : 3 + payload_len]
    signature = blob[3 + payload_len :]
    return payload_bytes, signature


def _encode_wire(blob: bytes) -> str:
    """binary blob -> `SKT-XXXX-XXXX-...` string."""
    b32 = base64.b32encode(blob).decode("ascii").rstrip("=")
    chunks = [b32[i : i + CHUNK_SIZE] for i in range(0, len(b32), CHUNK_SIZE)]
    return KEY_PREFIX + "-".join(chunks)


# Regex to recognize what a Sanketra key should roughly look like. Loose
# on the inner groups (users fat-finger hyphens) — tight on the prefix.
_KEY_SHAPE_RE = re.compile(r"^SKT-[A-Z2-7\-]+$", re.IGNORECASE)


def _decode_wire(key_str: str) -> bytes:
    if not isinstance(key_str, str):
        raise ValueError(f"license key must be a str, got {type(key_str).__name__}")
    key_str = key_str.strip()
    if not _KEY_SHAPE_RE.match(key_str):
        raise ValueError("malformed license key (expected SKT-XXXX-XXXX-...)")
    body = key_str[len(KEY_PREFIX) :].replace("-", "").upper()
    # Re-pad to multiple of 8 for b32decode.
    pad = (-len(body)) % 8
    try:
        return base64.b32decode(body + "=" * pad)
    except Exception as e:
        raise ValueError(f"base32 decode failed: {e}") from e


# --------------------------------------------------------------------------- #
# Public API: issue + verify                                                  #
# --------------------------------------------------------------------------- #

def issue_license(
    license_obj: License,
    private_key: ed25519.Ed25519PrivateKey,
) -> str:
    """Sign a `License` and return the wire-format key.

    CALLER OWNS the private key and is expected to keep it on the issuer
    host. This function is symmetrical with `verify_key` below — they
    share the same framing, hashing, and signature scheme."""
    payload_bytes = license_obj.to_canonical_json()
    signature = private_key.sign(_sign_digest(payload_bytes))
    blob = _pack_blob(payload_bytes, signature)
    return _encode_wire(blob)


def verify_key(
    key_str: str,
    public_key: ed25519.Ed25519PublicKey,
) -> License:
    """Verify a wire-format key and return the `License` on success.

    Raises `ValueError` for framing/decoding failures and
    `InvalidSignature` for a key that was tampered with or signed by the
    wrong party. Callers should treat BOTH as "license is not valid" —
    the distinction only matters for logs."""
    blob = _decode_wire(key_str)
    payload_bytes, signature = _unpack_blob(blob)
    digest = _sign_digest(payload_bytes)
    # Raises cryptography.exceptions.InvalidSignature on mismatch — let
    # it propagate; callers distinguish by exception type.
    public_key.verify(signature, digest)
    return License.from_canonical_json(payload_bytes)


__all__ = [
    "WIRE_VERSION",
    "KEY_PREFIX",
    "MAX_PAYLOAD_BYTES",
    "VALID_TRACKS",
    "Track",
    "License",
    "issue_license",
    "verify_key",
    "InvalidSignature",
]
