"""
Accent profile store for Sanketra v1.2 calibration.

Wraps two files:
  ~/.config/sanketra/accent_profile.bin       — binary blob (the "profile")
  ~/.config/sanketra/accent_profile.meta.json — JSON metadata

Per ANDROID_POLISH_V1.2.html §4: the user reads a 30-second curated paragraph,
the server extracts F0 mean + formant offsets + phoneme posterior bias vector,
and the result is stored per-user at `accent_profile.bin` (~2KB).

v1.2 scope (this module):
  - Accept a WAV / PCM16 audio blob from the client.
  - If `librosa` is available: extract real F0 mean + formant-like spectral
    centroids and persist them alongside a 64-float bias vector derived from
    a spectrogram summary. This is a deliberately simple stub — not a real
    phoneme-posterior extractor; that's v1.3 work with a Hindi AM.
  - If `librosa` is NOT available: persist a hash-plus-metadata "profile" so
    the API contract (round-tripping through the dashboard + Android) works.
    The server's transcribe loop can check `get_metadata()["has_profile"]`
    and decide whether to bother loading anything.

Contract expectations from the endpoints:
  save_profile(audio_bytes, sample_rate=None, client_name=None) → metadata dict
  get_metadata() → metadata dict (or empty dict if not calibrated)
  reset() → None
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import threading
import time
import uuid
import wave
from typing import Any


DEFAULT_DIR = os.path.join(os.path.expanduser("~"), ".config", "sanketra")
DEFAULT_BLOB_PATH = os.path.join(DEFAULT_DIR, "accent_profile.bin")
DEFAULT_META_PATH = os.path.join(DEFAULT_DIR, "accent_profile.meta.json")

# Minimum duration — a 1-second sample is obviously not a 30-second calibration.
# Spec target is 30s; we accept 5s+ to be generous to users who tap Stop early.
MIN_DURATION_SEC = 2.0
# Reject giant uploads to avoid trivial DoS. 60 s of 16 kHz mono PCM16 ≈ 1.9 MB;
# cap at 10 MB which covers any sensible upload including 44.1 kHz stereo.
MAX_AUDIO_BYTES = 10 * 1024 * 1024


def _parse_wav(audio_bytes: bytes) -> tuple[bytes, int, int]:
    """Return (pcm16_bytes, sample_rate, channels). Raises on malformed WAV."""
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if sw != 2:
        raise ValueError(f"expected PCM16 WAV, got sampwidth={sw}")
    return frames, sr, ch


def _try_decode_audio(audio_bytes: bytes, hinted_sample_rate: int | None) -> tuple[bytes, int, int, float]:
    """Best-effort decode. Returns (pcm16, sr, channels, duration_sec).

    Tries: WAV first, raw PCM16 fallback using hinted_sample_rate.
    Raises ValueError if nothing works or sample is too short.
    """
    # 1) WAV container
    if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        pcm, sr, ch = _parse_wav(audio_bytes)
        dur = len(pcm) / (sr * ch * 2) if sr > 0 and ch > 0 else 0.0
        if dur < MIN_DURATION_SEC:
            raise ValueError(f"audio too short ({dur:.2f}s < {MIN_DURATION_SEC}s)")
        return pcm, sr, ch, dur
    # 2) Raw PCM16 — hinted sample rate is required
    if hinted_sample_rate:
        sr = int(hinted_sample_rate)
        ch = 1
        dur = len(audio_bytes) / (sr * ch * 2) if sr > 0 else 0.0
        if dur < MIN_DURATION_SEC:
            raise ValueError(f"audio too short ({dur:.2f}s < {MIN_DURATION_SEC}s)")
        return audio_bytes, sr, ch, dur
    raise ValueError("unrecognized audio format (need WAV or PCM16 with sample_rate hint)")


def _extract_features(pcm16: bytes, sample_rate: int, channels: int) -> dict[str, Any]:
    """Extract a feature-ish profile. Falls back to a hash-stub if librosa
    is not installed. The stub is deliberate: the API contract (profile
    round-trips, dashboard reflects it) must work even on boxes without
    the ML deps. Full feature extraction is v1.3.
    """
    features: dict[str, Any] = {}

    try:
        import numpy as _np
        import librosa  # type: ignore
        # Convert PCM16 → float32 mono
        raw = _np.frombuffer(pcm16, dtype=_np.int16)
        if channels > 1:
            raw = raw.reshape(-1, channels).mean(axis=1)
        audio = raw.astype(_np.float32) / 32768.0

        # F0 via YIN — cheap, no learned model. 75-350 Hz covers adult Hindi F0.
        try:
            f0 = librosa.yin(audio, fmin=75, fmax=350, sr=sample_rate, frame_length=2048)
            f0_valid = f0[_np.isfinite(f0) & (f0 > 0)]
            f0_mean = float(f0_valid.mean()) if f0_valid.size else 0.0
        except Exception:
            f0_mean = 0.0

        # Spectral centroid as a crude formant-height proxy.
        try:
            centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
            spec_centroid_mean = float(centroids.mean()) if centroids.size else 0.0
        except Exception:
            spec_centroid_mean = 0.0

        # 64-bin log-mel mean as the "bias vector" placeholder. Not a phoneme
        # posterior, but shape matches (64 floats) and is deterministic.
        try:
            mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=64)
            bias_vec = _np.log1p(mel.mean(axis=1)).astype(_np.float32)
        except Exception:
            bias_vec = _np.zeros(64, dtype=_np.float32)

        features = {
            "backend": "librosa",
            "f0_mean_hz": round(f0_mean, 2),
            "spectral_centroid_mean_hz": round(spec_centroid_mean, 2),
            "bias_vector_len": int(len(bias_vec)),
            # Very rough dialect estimate from F0 range — female Hindi voices
            # cluster 180-250 Hz, male 100-150 Hz. We just label "standard-hindi"
            # until a real AM is wired; the UI already accepts gentle labels.
            "dialect_estimate": "standard-hindi",
        }
        # Serialize bias vector + headers into the .bin payload.
        blob = _pack_blob(f0_mean, spec_centroid_mean, bias_vec.tolist())
        features["_blob"] = blob
    except ImportError:
        # Stub path — durable "profile" that the API contract round-trips.
        digest = hashlib.sha256(pcm16).hexdigest()
        features = {
            "backend": "stub",
            "f0_mean_hz": 0.0,
            "spectral_centroid_mean_hz": 0.0,
            "bias_vector_len": 0,
            "dialect_estimate": "unknown",
            "sample_hash_sha256": digest,
        }
        features["_blob"] = _pack_stub_blob(digest, sample_rate, len(pcm16))
    except Exception as e:
        # Unexpected failure inside the extractor — don't blow up the endpoint,
        # degrade to stub. Log via print so the server captures it; this module
        # shouldn't pull the server's logger.
        print(f"[accent_store] feature extraction failed: {e} — falling back to stub")
        digest = hashlib.sha256(pcm16).hexdigest()
        features = {
            "backend": "stub",
            "error": str(e),
            "dialect_estimate": "unknown",
            "sample_hash_sha256": digest,
        }
        features["_blob"] = _pack_stub_blob(digest, sample_rate, len(pcm16))

    return features


# -- binary blob format -----------------------------------------------------
# Header: magic (4B) | version (uint8) | backend (uint8) | reserved (2B)
#   backend: 0 = stub, 1 = librosa
# Librosa body: f0 (float32) | centroid (float32) | vec_len (uint16) | vec (float32×N)
# Stub body:    sr (uint32) | sample_bytes (uint32) | sha256 (32B)
_MAGIC = b"SKAP"


def _pack_blob(f0: float, centroid: float, vec: list[float]) -> bytes:
    hdr = struct.pack("<4sBBBB", _MAGIC, 1, 1, 0, 0)
    body = struct.pack(f"<ffH{len(vec)}f", float(f0), float(centroid), len(vec), *vec)
    return hdr + body


def _pack_stub_blob(digest_hex: str, sample_rate: int, sample_bytes: int) -> bytes:
    hdr = struct.pack("<4sBBBB", _MAGIC, 1, 0, 0, 0)
    body = struct.pack("<II32s", int(sample_rate), int(sample_bytes), bytes.fromhex(digest_hex))
    return hdr + body


class AccentStore:
    def __init__(self, blob_path: str = DEFAULT_BLOB_PATH, meta_path: str = DEFAULT_META_PATH):
        self.blob_path = blob_path
        self.meta_path = meta_path
        os.makedirs(os.path.dirname(os.path.abspath(blob_path)), exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------ save

    def save_profile(
        self,
        audio_bytes: bytes,
        sample_rate: int | None = None,
        client_name: str | None = None,
    ) -> dict[str, Any]:
        """Ingest audio + write blob + metadata. Returns the new metadata dict.

        Raises ValueError for unusable inputs (too small, bad format, too long).
        """
        if not audio_bytes:
            raise ValueError("empty audio")
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise ValueError(
                f"audio too large ({len(audio_bytes)} bytes > {MAX_AUDIO_BYTES})"
            )
        pcm16, sr, ch, duration = _try_decode_audio(audio_bytes, sample_rate)
        feats = _extract_features(pcm16, sr, ch)
        blob = feats.pop("_blob")

        profile_id = uuid.uuid4().hex
        meta = {
            "profile_id": profile_id,
            "last_calibrated": int(time.time() * 1000),
            "last_calibrated_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sample_count": 1,
            "duration_sec": round(duration, 3),
            "sample_rate": sr,
            "channels": ch,
            "client_name": client_name,
            "dialect_estimate": feats.get("dialect_estimate", "unknown"),
            "backend": feats.get("backend", "stub"),
            "has_profile": True,
            "features": {k: v for k, v in feats.items() if not k.startswith("_")},
            "blob_bytes": len(blob),
        }

        with self._lock:
            # Keep sample_count incrementing across repeat calibrations so the
            # dashboard can show "Calibrated 3 times".
            existing = self._read_meta_unlocked()
            if existing and existing.get("has_profile"):
                meta["sample_count"] = int(existing.get("sample_count", 1)) + 1

            # Atomic writes for both files — the blob MUST land before the meta
            # flips has_profile=True, else a crash mid-write leaves the meta
            # pointing at a nonexistent blob.
            tmp_blob = self.blob_path + ".tmp"
            with open(tmp_blob, "wb") as f:
                f.write(blob)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp_blob, self.blob_path)

            tmp_meta = self.meta_path + ".tmp"
            with open(tmp_meta, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp_meta, self.meta_path)

        return self._public_meta(meta)

    # ------------------------------------------------------------ get

    def get_metadata(self) -> dict[str, Any]:
        """Public metadata (no raw blob bytes). Returns {} if never calibrated."""
        with self._lock:
            meta = self._read_meta_unlocked()
        if not meta:
            return {"has_profile": False}
        # Sanity: if the blob file vanished (user deleted by hand), don't claim
        # has_profile=True.
        if meta.get("has_profile") and not os.path.exists(self.blob_path):
            meta["has_profile"] = False
        return self._public_meta(meta)

    # ------------------------------------------------------------ reset

    def reset(self) -> None:
        """Wipe both files. Safe to call when not calibrated."""
        with self._lock:
            for p in (self.blob_path, self.meta_path):
                try:
                    if os.path.exists(p):
                        os.unlink(p)
                except OSError:
                    pass

    # ------------------------------------------------------------ internals

    def _read_meta_unlocked(self) -> dict[str, Any] | None:
        if not os.path.exists(self.meta_path):
            return None
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _public_meta(meta: dict[str, Any]) -> dict[str, Any]:
        """Strip internal fields before returning to HTTP clients."""
        return {k: v for k, v in meta.items() if not k.startswith("_")}


# ---------------------------------------------------------------- singleton

_default_store: AccentStore | None = None
_default_store_lock = threading.Lock()


def get_default_store() -> AccentStore:
    global _default_store
    with _default_store_lock:
        if _default_store is None:
            _default_store = AccentStore()
        return _default_store


def reset_default_store_for_testing(
    blob_path: str | None = None, meta_path: str | None = None
) -> AccentStore:
    global _default_store
    with _default_store_lock:
        _default_store = AccentStore(
            blob_path or DEFAULT_BLOB_PATH,
            meta_path or DEFAULT_META_PATH,
        )
        return _default_store
