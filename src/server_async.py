#!/usr/bin/env python3
"""
Real-time STT WebSocket Server (Async)
FastAPI + uvicorn + async WebSocket + offloaded Whisper

Run: python server_async.py
"""

import os
import sys

# Handle pythonw.exe (no console) - redirect stdout/stderr
_is_pythonw = sys.executable.lower().endswith('pythonw.exe')
if _is_pythonw or sys.stdout is None or sys.stderr is None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)  # Go up from src/ to root
    logs_dir = os.path.join(root_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, "server_async_bg.log")
    try:
        log_handle = open(log_file, "w", buffering=1)
        sys.stdout = log_handle
        sys.stderr = log_handle
    except Exception:
        class NullWriter:
            def write(self, s): pass
            def flush(self): pass
        sys.stdout = NullWriter()
        sys.stderr = NullWriter()

import asyncio
import logging
import re
import numpy as np
import torch
import platform
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional
from contextlib import asynccontextmanager
import time
import ssl
import socket
import subprocess
import struct
import queue
import hashlib
import json
import signal
import uuid
from enum import Enum, auto

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, UploadFile, File, Form, Path as PathParam
from fastapi.responses import HTMLResponse, Response, JSONResponse, PlainTextResponse
from fastapi import Request
from starlette.middleware.gzip import GZipMiddleware
from starlette.websockets import WebSocketState
import uvicorn
try:
    import uvloop
    uvloop.install()
except ImportError:
    pass
import secrets
import qrcode
import qrcode.image.svg

# Auth token for WebSocket connections (persisted across restarts)
AUTH_TOKEN = secrets.token_urlsafe(24)  # default; overwritten by _load_auth_token() at startup

_paired_ips: set[str] = set()  # IPs that have paired (for logging)
_pair_attempts: dict[str, list[float]] = {}  # IP -> list of timestamps (rate limiting)
# Auth/rate-limit constants live in auth_core.py — single source for both
# server (here) and tests (tests/test_auth_logic.py). Re-exported as the
# old _-prefixed names for backward compat with all the inline references.
from auth_core import (
    PAIR_RATE_LIMIT as _PAIR_RATE_LIMIT,
    PAIR_RATE_WINDOW as _PAIR_RATE_WINDOW,
)
_pair_locked = False  # True after first successful pair — blocks further pairing
# P0-4: TOFU pair code — generated at startup, required in /api/pair body to prevent
# first-pair race on shared WiFi. 4-digit numeric code displayed in server console.
_pair_code: str = f"{secrets.randbelow(10000):04d}"
_blink_active = False
SERVER_PORT = 5000  # updated at startup from args
_SERVICE_MODE = False  # Set True by --service flag
_SERVER_START_TIME = time.time()  # For uptime in /api/health

def _read_version() -> str:
    """Read version from top-level VERSION file (single source of truth for Android + server)."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        with open(os.path.join(root_dir, "VERSION"), "r") as f:
            return f.read().strip() or "0.0.0"
    except Exception:
        return "0.0.0"

_SERVER_VERSION = _read_version()

# Active WS connection counters (atomically updated)
_active_trackpad_ws = 0
_active_screen_ws = 0
_active_audio_out_ws = 0

# Dashboard push channel — real-time transcripts to /dashboard clients.
# Set of WebSocket instances accepted on /ws-dashboard. Mutated only from the
# asyncio loop, so no lock needed. Broadcasts are best-effort: a dead socket
# is popped on the next send.
_dashboard_ws_clients: "set[WebSocket]" = set()
# Cap number of concurrent dashboards at 8 — more than one physical user per LAN
# is exotic; this stops a misconfigured client from leaking sockets.
_MAX_DASHBOARD_WS = 8

# Audio reconnection buffer: per-(IP, session_token_hash) ring buffer of
# recent audio frames. On reconnect, replay buffered frames so no speech
# is lost during WS drop.
# 3 seconds at 31.25 fps = ~94 frames × 1024 bytes = ~96KB per client (negligible).
#
# O-P1-2 / F-Apr22-02: keyed by (client_ip, session_token_hash) where the
# hash is sha256(token)[:16]. Previously keyed by IP alone, which meant two
# phones behind the same NAT could share — and cross-feed — each other's
# buffered audio on reconnect. Multi-phone is still blocked server-side by
# code 4030, so this is defensive hardening for when the cap is lifted.
_audio_reconnect_buffers: "dict[tuple[str, str], deque]" = {}
_RECONNECT_BUFFER_MAX_FRAMES = 94  # ~3 seconds of audio

# Server state machine (service mode)
class ServerState(Enum):
    IDLE = auto()       # No model loaded, waiting for auth + model selection
    LOADING = auto()    # Model download/load in progress
    ACTIVE = auto()     # Model loaded, ready for STT

_server_state = ServerState.ACTIVE  # Default: ACTIVE (legacy mode loads model at startup)
_model_load_progress = 0.0  # 0.0-1.0
_model_load_cancel = False
_loaded_model_name = None
_loaded_device = "cpu"  # "cuda" or "cpu" — actual device model is running on
_last_ws_activity = time.time()  # For auto-unload timer
_MODEL_UNLOAD_TIMEOUT = 900  # 15 minutes no WS connections

# Per-client session management
_client_sessions: dict[str, dict] = {}  # token -> {ip, created, last_used}
_auth_attempts: dict[str, list[float]] = {}  # IP -> timestamps
from auth_core import (
    AUTH_RATE_LIMIT as _AUTH_RATE_LIMIT,
    AUTH_RATE_WINDOW as _AUTH_RATE_WINDOW,
    AUTH_LOCKOUT as _AUTH_LOCKOUT,
    SESSION_TTL as _SESSION_TTL,
)
_SESSION_MAX_COUNT = 50        # Max concurrent sessions

# Config file for service mode
_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".config", "sanketra")
_CONFIG_FILE = os.path.join(_CONFIG_DIR, "config.json")
# Guards all config read-modify-write sequences.
# P3-6: This is a threading.Lock that may briefly block the asyncio event loop
# when called from async handlers (e.g., _verify_session, _save_client_sessions).
# Acceptable for single-user LAN use: config I/O is <1ms on local filesystem,
# contention is near-zero (max 1 phone + rare API calls), and the alternative
# (asyncio file I/O with aiofiles) adds complexity for no measurable benefit.
_config_lock = threading.Lock()

def _load_config():
    """Load config from ~/.config/sanketra/config.json.
    Caller MUST hold _config_lock when part of a read-modify-write."""
    if os.path.exists(_CONFIG_FILE):
        try:
            with open(_CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load config {_CONFIG_FILE}: {e}")
    return {}

def _save_config(config):
    """Save config to ~/.config/sanketra/config.json (atomic).
    Caller MUST hold _config_lock when part of a read-modify-write.
    Write to temp file, fsync, then os.replace — crash-safe."""
    os.makedirs(_CONFIG_DIR, exist_ok=True)
    tmp_path = _CONFIG_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(config, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, _CONFIG_FILE)

def _get_app_password_hash():
    """Get stored Argon2id password hash from config"""
    with _config_lock:
        config = _load_config()
        return config.get("password_hash", "")

def _set_app_password_hash(password_hash):
    """Store Argon2id password hash in config"""
    with _config_lock:
        config = _load_config()
        config["password_hash"] = password_hash
        _save_config(config)

def _save_client_sessions():
    """Persist client sessions to config (synchronous write — blocks fsync).
    Caller MUST hold _sessions_lock.

    Hot-path callers should prefer _mark_sessions_dirty() instead, which
    schedules a coalesced write via the background writer task. This sync
    function is still used by the shutdown flush and the initial bootstrap
    path where we want a guaranteed-durable write before returning."""
    with _config_lock:
        config = _load_config()
        config["sessions"] = dict(_client_sessions)
        _save_config(config)


# ─────────────────────────────────────────────────────────────────────────────
# Debounced session writer (O-P1-4 / F-Apr22-03)
#
# Problem: every WS auth and /api/auth call was writing ~ /.config/sanketra/
# config.json (with fsync + os.replace) inline. Under concurrent phone auth
# bursts the fsync cost stacks up and blocks the asyncio loop long enough to
# drop audio frames. Single-user LAN usage hid it; multi-device hardening
# and a future "refresh all tokens" admin action would not.
#
# Design: one long-lived writer task. Mutators call _mark_sessions_dirty()
# which sets an asyncio.Event. The writer wakes, sleeps 500 ms (coalesce
# window — any further mutations in that window collapse into the same
# fsync), snapshots _client_sessions under _sessions_lock, and writes once.
#
# Durability contract: every mutation is guaranteed a write within
# ~500 ms + fsync time. A crash in that window loses the tail edit — same
# order of magnitude as the OS page-cache flush interval, and far better
# than the prior "fsync per auth or lose the WS frame budget" tradeoff.
# Shutdown path flushes synchronously via _drain_sessions_writer() so a
# clean stop still persists everything.
_sessions_dirty: "asyncio.Event | None" = None
_sessions_writer_task: "asyncio.Task | None" = None
_sessions_writer_loop: "asyncio.AbstractEventLoop | None" = None
_SESSIONS_COALESCE_SECONDS = 0.5


def _mark_sessions_dirty():
    """Signal the background writer that _client_sessions changed.
    Safe to call from asyncio handlers or sync threads (e.g. ThreadPool).
    No-op if the writer isn't started yet (bootstrap / test import)."""
    evt = _sessions_dirty
    loop = _sessions_writer_loop
    if evt is None or loop is None:
        return
    # asyncio.Event.set() is not threadsafe — schedule it on the writer's loop.
    try:
        loop.call_soon_threadsafe(evt.set)
    except RuntimeError:
        # Loop closed (shutdown race) — fall through silently. The shutdown
        # path has already drained, or will drain synchronously below.
        pass


async def _sessions_writer_main():
    """Coalescing writer: wake on dirty, sleep the coalesce window, write once."""
    global _sessions_dirty
    assert _sessions_dirty is not None
    while True:
        await _sessions_dirty.wait()
        # Coalesce: any mutations during this sleep pile into the same write.
        await asyncio.sleep(_SESSIONS_COALESCE_SECONDS)
        _sessions_dirty.clear()
        try:
            # Snapshot under the threading lock so we never observe a half-
            # mutated dict from /api/auth's purge-evict-insert sequence.
            with _sessions_lock:
                snapshot = dict(_client_sessions)
            with _config_lock:
                config = _load_config()
                config["sessions"] = snapshot
                _save_config(config)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Write errors must NOT kill the writer — next mutation will
            # re-trigger, next disk free / permission fix will heal.
            logging.warning(f"[Sessions] debounced write failed: {e}")


def _start_sessions_writer():
    """Spawn the writer task. Called once from lifespan startup."""
    global _sessions_dirty, _sessions_writer_task, _sessions_writer_loop
    if _sessions_writer_task is not None:
        return
    _sessions_writer_loop = asyncio.get_running_loop()
    _sessions_dirty = asyncio.Event()
    _sessions_writer_task = asyncio.create_task(_sessions_writer_main())
    _sessions_writer_task.add_done_callback(_task_done)


async def _drain_sessions_writer():
    """Shutdown path: flush any pending mutations from the last ~500 ms,
    then cancel the writer. Unconditional write — cheaper than poking the
    Event flag across the threading/asyncio boundary, and config.json is
    small enough that one extra write at shutdown is invisible."""
    global _sessions_writer_task
    try:
        with _sessions_lock:
            _save_client_sessions()
    except Exception as e:
        logging.warning(f"[Sessions] shutdown flush failed: {e}")
    if _sessions_writer_task is not None:
        _sessions_writer_task.cancel()
        try:
            await _sessions_writer_task
        except (asyncio.CancelledError, Exception):
            pass
        _sessions_writer_task = None

def _load_client_sessions():
    """Load persisted client sessions"""
    global _client_sessions
    with _sessions_lock:
        with _config_lock:
            config = _load_config()
            _client_sessions = config.get("sessions", {})

def _load_auth_token():
    """Load persisted AUTH_TOKEN from config, or generate + save a new one.
    Ensures token survives server restarts so web clients don't lose auth."""
    global AUTH_TOKEN
    with _config_lock:
        config = _load_config()
        saved = config.get("auth_token", "")
        if saved:
            AUTH_TOKEN = saved
        else:
            # First run or config wiped — persist the token generated at import time
            config["auth_token"] = AUTH_TOKEN
            _save_config(config)

def _load_pair_lock():
    """Load pair lock state from config on startup."""
    global _pair_locked
    with _config_lock:
        config = _load_config()
        _pair_locked = config.get("pair_locked", False)

def _load_pair_code():
    """Load a pre-provisioned pair code from config.json (written by SSH setup).
    If config contains a valid 4-digit pair_code string, use it and delete from config."""
    global _pair_code
    with _config_lock:
        config = _load_config()
        code = config.get("pair_code")
        if isinstance(code, str) and len(code) == 4 and code.isdigit():
            _pair_code = code
            del config["pair_code"]
            _save_config(config)
            log_info(f"Loaded pre-provisioned pair code from config")

def _get_server_id():
    """Get or generate a persistent server identity.
    Stored in config.json, survives cert regeneration and IP changes.
    Format: 16-char hex string derived from a UUID4 generated once."""
    with _config_lock:
        config = _load_config()
        sid = config.get("server_id", "")
        if not sid:
            import uuid
            sid = uuid.uuid4().hex[:16]
            config["server_id"] = sid
            _save_config(config)
        return sid

def _set_pair_lock(locked: bool, device_ip: str = None):
    """Persist pair lock state to config. Sets paired_device info on lock."""
    global _pair_locked
    with _config_lock:
        config = _load_config()
        config["pair_locked"] = locked
        if locked and device_ip:
            config["paired_device"] = {
                "ip": device_ip,
                "paired_at": time.time(),
            }
        elif not locked:
            config.pop("paired_device", None)
        _save_config(config)
        _pair_locked = locked

from stt_common import (
    load_whisper, load_vad, to_roman, select_model, init_gpu, get_gpu_stats, cleanup_gpu,
    type_text, key_press, mouse_move, mouse_click, mouse_drag, mouse_scroll,
    mouse_move_absolute, get_screen_resolution, get_screen_resolution_physical, check_input_health,
    setup_logging, log_info, log_debug, log_error, log_warning,
    STREAMING_SAMPLE_RATE, STREAMING_CHUNK_SAMPLES, STREAMING_SILENCE_THRESHOLD,
    custom_model_selection, get_vad_type, get_available_models,
    get_display_server, get_linux_input_tool, get_input_tool_info, PLATFORM,
    preprocess_audio_frame, preprocess_audio_buffer, get_preprocessing_config, FilterState,
    pause_app_input, resume_app_input, is_app_input_paused, register_pause_callback,
    get_cursor_position, get_monitors, get_monitor_by_index,
)
import preflight
import input_monitor
import auth_core
import history_db
import vocab_store
import accent_store
import license_core  # License verifier + installer (offline, no network).

# =============================================================================
#                              PROTOCOL CONSTANTS
# =============================================================================

# Client -> Server
MSG_AUDIO_FRAME   = 0x01  # + 1024 bytes PCM16LE
MSG_START_SESSION = 0x10  # No payload
MSG_END_SESSION   = 0x11  # No payload (flush and close)

# Server -> Client
MSG_FINAL         = 0x02  # + UTF-8 text
MSG_VAD_STATUS    = 0x03  # + 1 byte (0x00=silence, 0x01=speech)
MSG_PARTIAL       = 0x04  # + UTF-8 text (interim transcript, not typed)
MSG_BACKPRESSURE  = 0xFE  # No payload (client must pause)
MSG_SESSION_READY = 0xF0  # No payload
MSG_SESSION_DONE  = 0xF1  # No payload (client may close)
MSG_INPUT_PAUSED  = 0xF2  # No payload (physical input detected, app paused)
MSG_INPUT_RESUMED = 0xF3  # No payload (phone activity, app resumed)
MSG_AUTH          = 0xFA  # + UTF-8 token (first message from client)
MSG_ERROR         = 0xFF  # + UTF-8 text

# Pre-allocated wire messages (avoid per-send allocation at 31+ fps)
_WIRE_VAD_SPEECH      = bytes([MSG_VAD_STATUS, 0x01])
_WIRE_VAD_SILENCE     = bytes([MSG_VAD_STATUS, 0x00])
_WIRE_BACKPRESSURE    = bytes([MSG_BACKPRESSURE])
_WIRE_SESSION_READY   = bytes([MSG_SESSION_READY])
_WIRE_SESSION_DONE    = bytes([MSG_SESSION_DONE])
_WIRE_INPUT_PAUSED    = bytes([MSG_INPUT_PAUSED])
_WIRE_INPUT_RESUMED   = bytes([MSG_INPUT_RESUMED])

# Audio constants
SAMPLE_RATE = 16000
FRAME_SAMPLES = 512      # 32ms
FRAME_BYTES = 1024       # 512 samples * 2 bytes (int16)
MAX_BUFFER_SECONDS = 30
MAX_BUFFER_FRAMES = int(MAX_BUFFER_SECONDS * SAMPLE_RATE / FRAME_SAMPLES)
MAX_SPEECH_SECONDS = 8  # Force-transcribe after 8s continuous speech
MAX_SPEECH_FRAMES = int(MAX_SPEECH_SECONDS * SAMPLE_RATE / FRAME_SAMPLES)

# Heartbeat
HEARTBEAT_INTERVAL = 15  # seconds

# =============================================================================
#                          LICENSE FEATURE GATES
# =============================================================================
#
# Free vs Pro enforcement lives on the PC server — neither client binary
# embeds any checkout surface (see MONETIZATION.md). Knobs are local
# constants so ops can tweak without a code change elsewhere:
#
#   FREE_TIER_MODELS             — Whisper model names a free-tier caller
#                                  may load. Everything not in this set
#                                  requires a license that covers the
#                                  caller's track.
#   FREE_TIER_MONTHLY_MINUTES    — rolling monthly STT inference cap for
#                                  free-tier callers. Resets on the first
#                                  WS auth of a new calendar month (UTC).
#   VALID_CLIENT_KINDS           — allow-list for the `X-Client-Kind` request
#                                  header. Unknown values are rejected at the
#                                  endpoint (400) rather than silently
#                                  defaulted — don't trust arbitrary strings.
#   DEFAULT_CLIENT_KIND          — what we assume when the header is missing.
#                                  `web` because the local browser UI is the
#                                  only client that legitimately omits it.
#
FREE_TIER_MODELS = frozenset({"tiny", "base", "small"})
FREE_TIER_MONTHLY_MINUTES = 200
VALID_CLIENT_KINDS = frozenset({"android", "desktop", "web"})
DEFAULT_CLIENT_KIND = "web"


def _now() -> float:
    """Clock seam for tests. Production code must use this rather than
    `time.time()` directly so `tests/test_feature_gates.py` can pin a
    deterministic clock (monkeypatched to return a fixed epoch)."""
    return time.time()


def _client_kind_from_headers(headers) -> "str | None":
    """Read + validate the `X-Client-Kind` header.

    Returns one of `VALID_CLIENT_KINDS`, or `DEFAULT_CLIENT_KIND` when the
    header is missing, or `None` when the header is present but outside the
    allow-list. Callers translate `None` to a 400.

    Accepts FastAPI `Request.headers` (starlette Headers) OR a plain dict
    so WS handlers that carry `ws.headers` can use the same function.
    """
    raw = headers.get("x-client-kind") if hasattr(headers, "get") else None
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return DEFAULT_CLIENT_KIND
    if not isinstance(raw, str):
        return None
    v = raw.strip().lower()
    if v not in VALID_CLIENT_KINDS:
        return None
    return v


def _track_for_client_kind(client_kind: str) -> "str | None":
    """Map client kind → license track whose claim unlocks it.

    `android`  -> `phone`   (a phone-track license OR a bundle license unlocks)
    `desktop`  -> `desktop` (a desktop-track license OR a bundle license unlocks)
    `web`      -> None      (browser client is always free tier — no native
                             binary to sell, per MONETIZATION.md)
    anything else -> None (defense: unknown → can't unlock anything)
    """
    if client_kind == "android":
        return "phone"
    if client_kind == "desktop":
        return "desktop"
    return None


def _has_license_for_client(client_kind: str) -> bool:
    """Is the currently-installed license sufficient for `client_kind`?

    Delegates to `license_core.has_track()`. Returns False when no license
    is installed, when the license doesn't cover this track, or when the
    client kind is one we never unlock (e.g. `web`). Never raises — the
    license subsystem is designed to fail into "free tier" on any error."""
    track = _track_for_client_kind(client_kind)
    if track is None:
        return False
    try:
        return license_core.has_track(track)
    except Exception as e:
        log_warning(f"[License] has_track({track}) errored — treating as free tier: {e}")
        return False

# =============================================================================
#                              GLOBAL STATE
# =============================================================================

whisper_model = None
_whisper_lock = threading.Lock()  # Guards model swap in _load_model_async and snapshot in transcribe_sync
vad_model = None
vad_utils = None
executor: Optional[ThreadPoolExecutor] = None      # Whisper inference
io_executor: Optional[ThreadPoolExecutor] = None    # type_text, blink, etc.
_model_load_executor: Optional[ThreadPoolExecutor] = None  # Model download/load (separate from transcription)

# --- async serialization (single uvicorn worker) ---
# asyncio.Lock for check-then-act in async handlers. Without these, concurrent
# POSTs race past the _server_state/_pair_locked bool check and both proceed —
# causing double-load OOM (two Whisper loads in parallel) and pair-lock bypass
# (two simultaneous /api/pair both "win"). Single-worker uvicorn makes one
# event-loop lock per invariant sufficient.
_model_load_lock: Optional[asyncio.Lock] = None   # created in lifespan
_pair_mutex: Optional[asyncio.Lock] = None         # created in lifespan

# --- cross-thread state guard ---
# _loaded_device is mutated from the model-load ThreadPoolExecutor and read
# from async /api/health + /api/model/status handlers. Reads across threads
# without a lock can see half-written strings on 32-bit arches; trivially
# safe here but spells "surprising bug" in 6 months.
_device_lock = threading.Lock()

# Lock for _client_sessions dict — accessed from asyncio handlers and ThreadPoolExecutor
_sessions_lock = threading.Lock()

# Track active ffmpeg subprocesses for shutdown cleanup
_active_ffmpeg_procs: set = set()
_ffmpeg_procs_lock = threading.Lock()

# =============================================================================
#                              SESSION STATE
# =============================================================================

@dataclass
class StreamingSession:
    """Per-connection state"""
    vad_iterator: any = None
    audio_buffer: deque = field(default_factory=lambda: deque(maxlen=MAX_BUFFER_FRAMES))
    filter_state: FilterState = field(default_factory=FilterState)
    is_speaking: bool = False
    had_speech: bool = False
    silence_frames: int = 0
    pending_inference: bool = False
    session_active: bool = False
    last_activity: float = field(default_factory=time.time)
    frames_processed: int = 0
    last_partial_time: float = 0
    partial_running: bool = False
    skip_next_partial: bool = False
    _input_health_warned: bool = False
    # AU-P0-1: Rolling byte buffer for variable-length resampled frames
    _raw_byte_buf: bytearray = field(default_factory=bytearray)
    # Per-session state — never share across connections
    cached_language: Optional[str] = None  # Detected language, reset per-session
    _last_drop_log_time: float = 0.0  # Throttle ring-buffer-full warnings to 1/sec
    # v1.2 history logging: sessions row id in history.db. Populated on
    # MSG_START_SESSION, cleared on MSG_END_SESSION / disconnect. None when
    # logging is disabled via the dashboard toggle so the run_inference hook
    # can skip DB writes entirely.
    history_session_id: Optional[int] = None
    client_name: Optional[str] = None  # 'Pixel 8' / 'Chrome on tman' — shown in dashboard

    def reset_vad(self):
        """Reset VAD state after transcription"""
        self.audio_buffer.clear()
        self.is_speaking = False
        self.had_speech = False
        self.silence_frames = 0
        # Filter-state leak fix: HPF's biquad memory (`zi`) persisted a transient
        # from the tail of utterance N into the head of utterance N+1. Reset it so
        # each utterance starts from silence, not residual high-frequency energy.
        self.filter_state.zi = None
        if self.vad_iterator:
            self.vad_iterator.reset_states()

    def add_audio(self, audio_float: np.ndarray):
        """Add audio frame to buffer (deque auto-trims oldest at maxlen)"""
        # AU-P0-2: Log warning when ring buffer is full and dropping oldest frames.
        # Throttled to 1/sec — at 31fps this was 31 log lines/sec of stdio + rotating-file
        # I/O, which ate another 10-30ms of CPU the system already didn't have.
        if len(self.audio_buffer) == self.audio_buffer.maxlen:
            now = time.time()
            if now - self._last_drop_log_time >= 1.0:
                self._last_drop_log_time = now
                log_warning("[Stream] Audio ring buffer full — dropping oldest frame (CPU backpressure)")
        self.audio_buffer.append(audio_float)
        self.last_activity = time.time()
        self.frames_processed += 1

    def get_audio(self) -> Optional[np.ndarray]:
        """Get concatenated audio buffer"""
        if not self.audio_buffer:
            return None
        return np.concatenate(list(self.audio_buffer))


def _task_done(t):
    """Callback for fire-and-forget asyncio tasks — logs exceptions instead of silently swallowing them."""
    if t.cancelled():
        return
    exc = t.exception()
    if exc:
        logging.error(f"Task failed: {exc}", exc_info=exc)

# =============================================================================
#                              WHISPER INFERENCE (SYNC - runs in thread pool)
# =============================================================================

def transcribe_sync(audio: np.ndarray, hint_language: Optional[str] = None) -> tuple:
    """
    Synchronous Whisper transcription.
    Runs in thread pool, not in async loop.

    hint_language: previously detected language for this session (saves ~200ms re-detection).
    Caller should pass `session.cached_language` and store the returned `lang` back.
    """
    with _whisper_lock:
        model = whisper_model  # Snapshot under lock to avoid race with _load_model_async / _unload_model
    if model is None:
        return "", ""
    if len(audio) < int(SAMPLE_RATE * 0.25):  # Min 250ms
        return "", ""

    try:
        segments, info = model.transcribe(
            audio,
            language=hint_language,
            vad_filter=False,
            beam_size=1,
            condition_on_previous_text=False,
            initial_prompt="Hindi aur English mein baat ho rahi hai."
        )
        text = " ".join([seg.text for seg in segments])
        return text.strip(), info.language
    except Exception as e:
        log_error(f"Transcription error: {e}")
        return "", ""

# =============================================================================
#                              WEBSOCKET HELPERS
# =============================================================================

def _extract_token(request, token_param: str = None) -> str:
    """Extract auth token from Authorization header or query param (web backward compat)."""
    auth_header = request.headers.get('authorization', '')
    if auth_header.lower().startswith('bearer '):
        return auth_header[7:].strip()
    return token_param or ''

def _check_any_token(token: str) -> bool:
    """Validate token against legacy AUTH_TOKEN or per-client sessions."""
    if not token:
        return False
    # Direct AUTH_TOKEN (TOFU pairing token — works in all modes)
    if secrets.compare_digest(token, AUTH_TOKEN):
        return True
    # Per-client session token
    return _verify_session(token)

def _validate_ws_origin(ws: WebSocket) -> bool:
    """Validate WebSocket Origin header to prevent Cross-Site WebSocket Hijacking (CSWSH).
    Allows: no Origin (native apps), localhost, and the server's own IP."""
    origin = ws.headers.get("origin", "")
    if not origin:
        return True  # Native apps (Android OkHttp) don't send Origin
    try:
        from urllib.parse import urlparse
        parsed = urlparse(origin)
        host = parsed.hostname or ""
        if host in ("localhost", "127.0.0.1", "::1"):
            return True
        # Allow server's own IP
        server_ip = get_local_ip()
        if host == server_ip:
            return True
        # Allow any RFC 1918 IP (LAN clients)
        import ipaddress
        try:
            addr = ipaddress.ip_address(host)
            return addr.is_private
        except ValueError:
            return False
    except Exception:
        return True  # Don't block on parse errors

async def _authenticate_ws(ws: WebSocket) -> "str | None":
    """Authenticate WS: check query params first, then wait for MSG_AUTH first message.
    Concurrent WS limit is enforced atomically in _register_ws() after auth succeeds.

    Returns the validated token string on success, or None on failure. Callers
    that only need a truthiness check (4 of 5 handlers) can keep using
    `if not <retval>:` — Python truthiness of a non-empty str is True, of None
    is False. ws_audio_stream uses the returned token to derive a per-session
    key for the reconnect-buffer dict (avoids NAT-shared-IP cross-feed risk)."""
    # Query-param path carries the token in the URL; surface it for callers.
    query_token = ws.query_params.get('token', '')
    if query_token and _check_any_token(query_token):
        return query_token
    # Wait for auth message (binary: [0xFA][utf8_token] or text: {"t":"auth","token":"..."})
    try:
        data = await asyncio.wait_for(ws.receive(), timeout=5.0)
        if 'bytes' in data and data['bytes']:
            raw = data['bytes']
            if raw[0] == MSG_AUTH:
                token = raw[1:].decode('utf-8', errors='ignore')
                if _check_any_token(token):
                    return token
        elif 'text' in data and data['text']:
            msg = json.loads(data['text'])
            if msg.get('t') == 'auth':
                token = msg.get('token', '')
                if _check_any_token(token):
                    return token
    except Exception as e:
        log_debug(f"[Auth] WS auth message receive failed: {e}")
    return None

async def send_message(ws: WebSocket, msg_type: int, payload: bytes = b""):
    """Send binary message if connection is open"""
    if ws.client_state == WebSocketState.CONNECTED:
        try:
            await ws.send_bytes(bytes([msg_type]) + payload)
        except Exception as e:
            log_debug(f"[WS] send_message failed (connection closed): {e}")

async def _send_raw(ws: WebSocket, data: bytes):
    """Send pre-allocated binary message if connection is open"""
    if ws.client_state == WebSocketState.CONNECTED:
        try:
            await ws.send_bytes(data)
        except Exception as e:
            log_debug(f"[WS] _send_raw failed (connection closed): {e}")

def _vad_sync(vad_iterator, audio_float: np.ndarray) -> dict:
    """Synchronous VAD inference — runs in thread pool"""
    audio_tensor = torch.from_numpy(audio_float)
    return vad_iterator(audio_tensor, return_seconds=False)

async def process_vad(session: StreamingSession, audio_float: np.ndarray) -> dict:
    """Run VAD on audio frame (offloaded to thread pool to avoid blocking event loop)"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(io_executor, _vad_sync, session.vad_iterator, audio_float)

async def run_inference(session: StreamingSession, ws: WebSocket):
    """Offload Whisper inference to thread pool"""
    if session.pending_inference:
        return  # Already running
    if whisper_model is None:
        session.reset_vad()
        return  # Model unloaded (service mode auto-unload)

    audio = session.get_audio()
    if audio is None or len(audio) < int(SAMPLE_RATE * 0.25):
        session.reset_vad()
        return

    session.pending_inference = True
    # AU-P1-5: Snapshot buffer length before inference so we can preserve
    # audio that arrives during the await (force-transcribe orphan gap fix)
    buf_len_at_start = len(session.audio_buffer)
    audio_duration = len(audio) / SAMPLE_RATE

    try:
        # POINT G: Apply noise filtering before transcription (no energy preservation needed)
        config = get_preprocessing_config()
        if config.enabled:
            audio = preprocess_audio_buffer(audio, SAMPLE_RATE, config)

        loop = asyncio.get_event_loop()
        log_debug(f"[Stream] Transcribing {audio_duration:.2f}s of audio")
        text, lang = await loop.run_in_executor(executor, transcribe_sync, audio, session.cached_language)
        # Cache detected language per-session — previously a module global, which
        # leaked language state between phone A (English) and phone B (Hindi)
        # sharing the same server instance.
        if lang in ("hi", "en", "ur"):
            session.cached_language = lang

        if text and lang in ('hi', 'en'):
            text = to_roman(text)
            lang_name = 'Hindi' if lang == 'hi' else 'English'
            result = f"[{lang_name}] {text}"
            await send_message(ws, MSG_FINAL, result.encode('utf-8'))
            # Codex F-Apr21-10: transcript content used to land in stdout via
            # log_info(...full text...) + print(result). With v1.2's history
            # dashboard, that's a privacy double-tap. Log only metadata
            # (length + language). Set SANKETRA_DEBUG_TRANSCRIPTS=1 to opt
            # back into full text for debugging.
            if os.environ.get("SANKETRA_DEBUG_TRANSCRIPTS") == "1":
                log_info(f"[Stream] Transcribed: {result}")
                print(result, flush=True)
            else:
                log_info(f"[Stream] Transcribed: {len(text)}ch {lang_name}")
            # v1.2 history: log raw transcript (without the [Hindi]/[English]
            # prefix) so the dashboard shows clean text. Best-effort — any
            # DB failure logs and moves on; this MUST NOT kill the WS.
            _history_log_transcript_safe(session, text, lang)

            # Type to cursor (run in executor to not block)
            await loop.run_in_executor(io_executor, type_text, text + ' ')

            # macOS: warn phone if Accessibility permission blocks input dispatch
            input_ok, input_err = check_input_health()
            if not input_ok and not session._input_health_warned:
                session._input_health_warned = True
                await send_message(ws, MSG_ERROR, input_err.encode('utf-8'))
        else:
            await send_message(ws, MSG_FINAL, b"(skipped)")
            log_debug(f"[Stream] Skipped: lang={lang}")
    except Exception as e:
        log_error(f"[Stream] Inference error: {e}")
        await send_message(ws, MSG_ERROR, b"Transcription failed")
    finally:
        session.pending_inference = False
        # P0-3: Clear backpressure — client paused mic on MSG_BACKPRESSURE,
        # now inference is done so tell it to resume via MSG_SESSION_READY
        # (Android AudioWebSocket already clears _backpressured on this message)
        await _send_raw(ws, _WIRE_SESSION_READY)
        # AU-P1-5: Preserve audio that arrived during inference (orphan gap fix).
        # Only remove the frames that were part of the transcribed audio.
        new_frames_count = len(session.audio_buffer) - buf_len_at_start
        if new_frames_count > 0:
            # Keep the new frames that arrived during inference
            overflow = list(session.audio_buffer)[-new_frames_count:]
            session.reset_vad()
            for frame in overflow:
                session.audio_buffer.append(frame)
            log_debug(f"[Stream] Preserved {new_frames_count} frames that arrived during inference")
        else:
            session.reset_vad()

PARTIAL_INTERVAL = 1.0  # seconds between partial transcripts

async def run_partial_inference(session: StreamingSession, ws: WebSocket):
    """Run interim transcription without resetting buffer or typing text"""
    # AU-P1-2: Skip partial if flagged (final inference is about to run)
    if session.skip_next_partial:
        session.skip_next_partial = False
        return
    if session.partial_running or session.pending_inference:
        return
    if whisper_model is None:
        return  # Model unloaded

    audio = session.get_audio()
    if audio is None or len(audio) < int(SAMPLE_RATE * 0.5):
        return

    session.partial_running = True
    session.last_partial_time = time.time()

    try:
        # AU-P1-2: Check skip flag again after acquiring partial_running
        if session.skip_next_partial:
            session.skip_next_partial = False
            return

        config = get_preprocessing_config()
        if config.enabled:
            audio = preprocess_audio_buffer(audio, SAMPLE_RATE, config)

        loop = asyncio.get_event_loop()
        text, lang = await loop.run_in_executor(executor, transcribe_sync, audio, session.cached_language)
        if lang in ("hi", "en", "ur"):
            session.cached_language = lang

        if text and lang in ('hi', 'en'):
            text = to_roman(text)
            lang_name = 'Hindi' if lang == 'hi' else 'English'
            partial = f"[{lang_name}] {text}"
            await send_message(ws, MSG_PARTIAL, partial.encode('utf-8'))
            log_debug(f"[Stream] Partial: {partial}")
    except Exception as e:
        log_error(f"[Stream] Partial inference error: {e}")
    finally:
        session.partial_running = False

async def handle_audio_frame(session: StreamingSession, ws: WebSocket, payload: bytes):
    """Process single audio frame — accumulates variable-length payloads into fixed-size chunks"""
    # AU-P0-1: Accumulate incoming bytes into rolling buffer, process in FRAME_BYTES chunks.
    # Android devices with non-standard sample rates produce different frame sizes after resampling.
    session._raw_byte_buf.extend(payload)

    while len(session._raw_byte_buf) >= FRAME_BYTES:
        chunk = bytes(session._raw_byte_buf[:FRAME_BYTES])
        del session._raw_byte_buf[:FRAME_BYTES]

        await _process_audio_chunk(session, ws, chunk)


async def _process_audio_chunk(session: StreamingSession, ws: WebSocket, chunk: bytes):
    """Process a single fixed-size audio chunk (1024 bytes = 512 samples Int16LE)"""
    # Convert Int16LE to float32
    audio_int16 = np.frombuffer(chunk, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32768.0

    # POINT C: Apply noise filtering before VAD (preserves energy for VAD accuracy)
    # Store RAW audio in buffer (not preprocessed) to avoid double pre-emphasis at POINT G
    config = get_preprocessing_config()
    if config.enabled:
        vad_audio = preprocess_audio_frame(audio_float, SAMPLE_RATE, config, preserve_original_energy=True, filter_state=session.filter_state)
    else:
        vad_audio = audio_float

    # Always buffer RAW audio (preprocessing applied once at POINT G before Whisper)
    session.add_audio(audio_float)

    # Run VAD on preprocessed copy
    speech_dict = await process_vad(session, vad_audio)

    was_speaking = session.is_speaking

    if speech_dict:
        if 'start' in speech_dict:
            session.is_speaking = True
            session.had_speech = True
            session.silence_frames = 0
            log_debug("[Stream] Speech started")
        if 'end' in speech_dict:
            session.is_speaking = False
            log_debug("[Stream] Speech ended")

    # Send VAD status change
    if session.is_speaking != was_speaking:
        await _send_raw(ws, _WIRE_VAD_SPEECH if session.is_speaking else _WIRE_VAD_SILENCE)

    # Partial transcripts during speech (every 2s)
    if session.is_speaking and session.had_speech:
        now = time.time()
        if now - session.last_partial_time >= PARTIAL_INTERVAL:
            task = asyncio.create_task(run_partial_inference(session, ws))
            task.add_done_callback(_task_done)

    # Track silence for auto-transcription
    if session.is_speaking:
        session.silence_frames = 0
    elif session.had_speech:
        session.silence_frames += 1

        # 500ms silence -> transcribe
        if session.silence_frames >= STREAMING_SILENCE_THRESHOLD:
            # AU-P1-2: Skip any pending partial so final inference runs immediately
            session.skip_next_partial = True
            await run_inference(session, ws)

    # Force-transcribe if buffer exceeds 8s continuous speech
    if session.had_speech and not session.pending_inference and len(session.audio_buffer) >= MAX_SPEECH_FRAMES:
        log_debug(f"[Stream] Buffer cap hit ({MAX_SPEECH_SECONDS}s), force-transcribing")
        # AU-P1-2: Skip any pending partial so final inference runs immediately
        session.skip_next_partial = True
        await run_inference(session, ws)

    # Backpressure: if inference running and buffer growing
    if session.pending_inference and len(session.audio_buffer) > MAX_BUFFER_FRAMES // 2:
        await _send_raw(ws, _WIRE_BACKPRESSURE)

# =============================================================================
#                              WEBSOCKET HANDLER
# =============================================================================

def _new_conn_id() -> str:
    """Generate 8-char hex connection ID for WS correlation logging."""
    return uuid.uuid4().hex[:8]

def _resolve_cid(ws) -> str:
    """Use the client-supplied `X-Conn-Id` header if present and sane, else mint one.
    Lets phone-side logs share a correlation ID with server logs for field debugging."""
    hdr = ws.headers.get("x-conn-id", "") if hasattr(ws, "headers") else ""
    if hdr and len(hdr) <= 32 and all(c in "0123456789abcdefABCDEF-" for c in hdr):
        return hdr[:16]
    return _new_conn_id()

def _ws_log(conn_id: str, level: str, msg: str):
    """Log with per-connection correlation ID prefix.
    level: 'info', 'debug', 'warning', 'error'."""
    tagged = f"[ws:{conn_id}] {msg}"
    if level == "info":
        log_info(tagged)
    elif level == "debug":
        log_debug(tagged)
    elif level == "warning":
        log_warning(tagged)
    elif level == "error":
        log_error(tagged)


async def _broadcast_dashboard(event: dict) -> None:
    """Push a JSON event to all /ws-dashboard clients (best-effort).
    Dead sockets are swept inline — a stale subscriber can't block live ones.
    `event` is a JSON-serializable dict; we emit it verbatim with json.dumps so
    the client sees exactly what the server logged, no schema translation.
    """
    if not _dashboard_ws_clients:
        return
    try:
        payload = json.dumps(event, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        log_warning(f"[Dashboard] broadcast payload not serializable: {e}")
        return
    dead: list[WebSocket] = []
    for ws in list(_dashboard_ws_clients):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _dashboard_ws_clients.discard(ws)


def _history_log_transcript_safe(session: "StreamingSession", text: str, language: str) -> None:
    """Synchronous helper called from the audio WS loop after a MSG_FINAL send.
    Swallows all exceptions — a broken history DB must NEVER kill dictation.
    """
    if session.history_session_id is None:
        return
    try:
        db = history_db.get_default_db()
        if not db.get_settings().get("logging_enabled"):
            return
        tid = db.log_transcript(
            session_id=session.history_session_id,
            text=text,
            language=language,
        )
        if tid > 0:
            # Fire-and-forget broadcast. We schedule on the loop so the
            # caller (inference path) doesn't wait on network I/O.
            event = {
                "type": "transcript",
                "id": tid,
                "session_id": session.history_session_id,
                "text": text,
                "language": language,
                "created_at": int(time.time() * 1000),
            }
            try:
                loop = asyncio.get_event_loop()
                task = loop.create_task(_broadcast_dashboard(event))
                task.add_done_callback(_task_done)
            except RuntimeError:
                # Not in a loop — caller is sync thread without event loop.
                # Fine: dashboard just won't see real-time, the DB row is logged.
                pass
    except Exception as e:
        log_warning(f"[History] log_transcript failed (non-fatal): {e}")


async def _accept_authenticated_ws(ws: WebSocket, cid: int, channel_label: str) -> "str | None":
    """
    Run the boilerplate handshake every WS endpoint shared:
      ws.accept() → origin check → auth → register

    Returns the validated token on full acceptance, or None on any failure
    (origin reject / unauthorized / concurrency cap). Callers that only
    care about success/failure can keep the `if not retval:` pattern —
    Python truthiness works on str vs None. ws_audio_stream uses the token
    to derive a per-session key for the reconnect buffer so two phones
    behind the same NAT don't share buffered audio (O-P1-2 hardening).

    Replaces ~20 lines × 4 handlers (audio, trackpad, screen, audio_output)
    of duplicated try/close/log boilerplate that drifted slightly between
    handlers (e.g. trackpad logged differently, audio_output's reason text
    differed). Single source kills future drift.
    """
    await ws.accept()
    if not _validate_ws_origin(ws):
        _ws_log(cid, "warning", f"[{channel_label}] Origin rejected: {ws.headers.get('origin', '')}")
        try:
            await ws.close(code=4403, reason="Origin not allowed")
        except Exception:
            pass
        return None
    token = await _authenticate_ws(ws)
    if not token:
        try:
            await ws.close(code=4401, reason="Unauthorized")
        except Exception:
            pass
        return None
    if not _register_ws(ws):
        _ws_log(cid, "warning", f"[{channel_label}] Max concurrent WS limit reached, rejecting")
        try:
            await ws.close(code=4429, reason="Too many connections")
        except Exception:
            pass
        return None
    return token


async def ws_audio_stream(ws: WebSocket):
    """Main WebSocket handler for audio streaming.

    Architecture: the receive loop and the frame processor run as SEPARATE
    asyncio tasks connected by an asyncio.Queue. Previously both ran serially
    in one task — while inference was running (300ms–3s), `ws.receive_bytes()`
    never got scheduled, so the kernel's TCP buffer accumulated incoming audio
    and the first syllable of the user's next sentence was lost after every
    transcription. Now frames queue up while inference runs; the processor
    catches up without stalling the receiver.
    """
    global _last_ws_activity
    cid = _resolve_cid(ws)
    client_ip = ws.client.host if ws.client else "unknown"
    token = await _accept_authenticated_ws(ws, cid, "Stream")
    if not token:
        return
    # O-P1-2: derive a stable per-session suffix so the reconnect buffer is
    # keyed by (ip, token_hash). sha256(token)[:16] = 64 bits of entropy —
    # collision-safe at any realistic session count, and the raw token never
    # lands in a dict key that might surface in logs/diagnostics.
    import hashlib as _hl
    buffer_key: tuple[str, str] = (client_ip, _hl.sha256(token.encode()).hexdigest()[:16])
    _ws_log(cid, "info", f"[Stream] Audio WS connected from {client_ip}")

    # Initialize session. vad_utils is a 5-tuple — either Silero's native tuple or
    # the EnergyVAD fallback's synthesized 5-tuple (see stt_common.load_vad). As
    # long as utils[3] is callable as `VADIterator(model, sampling_rate=...)` and
    # returns an object with `__call__(audio_tensor, return_seconds=False)` and
    # `reset_states()`, the streaming path works regardless of backend.
    if not vad_utils or len(vad_utils) < 4 or vad_utils[3] is None:
        _ws_log(cid, "error", "[Stream] VAD not available — cannot start session")
        await ws.close(code=4500, reason="VAD unavailable")
        _unregister_ws(ws)
        return
    VADIterator = vad_utils[3]
    session = StreamingSession()
    session.vad_iterator = VADIterator(vad_model, sampling_rate=SAMPLE_RATE)
    clean_end = False  # Codex F3: only set True on MSG_END_SESSION receipt

    input_ok, input_err = check_input_health()
    if not input_ok:
        session._input_health_warned = True
        await send_message(ws, MSG_ERROR, input_err.encode('utf-8'))

    # Bounded queue: 120 frames × 32ms = ~4 s of audio in flight. If the
    # processor falls more than 4 s behind (Whisper doing 30 s of audio on CPU),
    # drop-oldest rather than growing unbounded memory.
    frame_queue: asyncio.Queue = asyncio.Queue(maxsize=120)

    async def _processor_loop():
        """Drains frame_queue and feeds handle_audio_frame. Isolated from
        ws.receive_bytes() so inference never starves the receiver."""
        while True:
            item = await frame_queue.get()
            if item is None:
                return  # sentinel — shut down
            try:
                await handle_audio_frame(session, ws, item)
            except Exception as e:
                _ws_log(cid, "debug", f"[Stream] processor frame error: {e}")

    processor_task = asyncio.create_task(_processor_loop())
    processor_task.add_done_callback(_task_done)

    def _enqueue_frame(payload: bytes):
        """Drop-oldest on overflow so slow processor can't stall receive loop."""
        try:
            frame_queue.put_nowait(payload)
        except asyncio.QueueFull:
            try:
                frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                frame_queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    async def _drain_and_flush(final: bool):
        """Wait for processor to consume outstanding frames, then run final
        inference on any remaining speech buffer. Used for END_SESSION and
        the disconnect-cleanup path."""
        # Signal processor to stop after draining what's queued.
        try:
            frame_queue.put_nowait(None)
        except asyncio.QueueFull:
            # Queue full — drop a frame so sentinel fits.
            try:
                frame_queue.get_nowait()
                frame_queue.put_nowait(None)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass
        try:
            await asyncio.wait_for(processor_task, timeout=10.0)
        except asyncio.TimeoutError:
            _ws_log(cid, "warning", "[Stream] processor drain timed out, cancelling")
            processor_task.cancel()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _ws_log(cid, "debug", f"[Stream] processor drain error: {e}")

        if final and session.audio_buffer:
            if not session.had_speech:
                session.had_speech = True  # force flush audio-without-VAD-trigger
            if not session.pending_inference:
                try:
                    await run_inference(session, ws)
                except Exception as e:
                    _ws_log(cid, "debug", f"[Stream] final flush inference error: {e}")

    try:
        while True:
            try:
                data = await ws.receive_bytes()
            except WebSocketDisconnect as e:
                _ws_log(cid, "info", f"[Stream] Client disconnected (code={e.code})")
                break
            except Exception as e:
                _ws_log(cid, "error", f"[Stream] Receive error: {e}")
                break

            if not data:
                continue

            if is_app_input_paused():
                resume_app_input()

            msg_type = data[0]
            payload = data[1:] if len(data) > 1 else b""

            if msg_type == MSG_START_SESSION:
                session.session_active = True
                session.reset_vad()
                # v1.2 history: open a sessions row so subsequent WIRE_FINAL
                # transcripts can attach. Skipped if logging is disabled so we
                # don't accumulate empty session rows.
                try:
                    db = history_db.get_default_db()
                    if db.get_settings().get("logging_enabled"):
                        # Client kind inferred from User-Agent when possible — the
                        # Android app sets its UA; Chrome ext sets its own; the
                        # local web client leaves it blank.
                        ua = (ws.headers.get("user-agent") or "").lower()
                        if "sanketra" in ua and "android" in ua:
                            client_kind = "android"
                        elif "chrome-extension" in ua or "chromext" in ua:
                            client_kind = "chrome"
                        elif ua:
                            client_kind = "web"
                        else:
                            client_kind = "web"
                        session.history_session_id = db.create_session(
                            client_kind=client_kind,
                            client_name=session.client_name,
                            language="hi",
                        )
                except Exception as e:
                    log_warning(f"[History] create_session failed (non-fatal): {e}")
                    session.history_session_id = None
                await _send_raw(ws, _WIRE_SESSION_READY)
                # Replay buffered audio from previous connection (reconnection recovery).
                # Queued to the processor, not awaited inline — receive loop stays live.
                reconnect_buf = _audio_reconnect_buffers.get(buffer_key)
                if reconnect_buf and len(reconnect_buf) > 0:
                    replay_count = len(reconnect_buf)
                    for buffered_frame in reconnect_buf:
                        _enqueue_frame(buffered_frame)
                    reconnect_buf.clear()
                    _ws_log(cid, "info", f"[Stream] Queued {replay_count} buffered frames from reconnect")
                _ws_log(cid, "info", "[Stream] Session started")

            elif msg_type == MSG_END_SESSION:
                _ws_log(cid, "info", "[Stream] End session requested")
                await _drain_and_flush(final=True)
                await _send_raw(ws, _WIRE_SESSION_DONE)
                session.session_active = False
                # v1.2 history: close the sessions row. Idempotent.
                if session.history_session_id is not None:
                    try:
                        history_db.get_default_db().end_session(session.history_session_id)
                    except Exception as e:
                        log_warning(f"[History] end_session failed (non-fatal): {e}")
                clean_end = True  # Codex F3: signal finally to drop reconnect buffer
                await ws.close(code=1000, reason="Session complete")
                _ws_log(cid, "info", "[Stream] Session ended cleanly")
                break

            elif msg_type == MSG_AUDIO_FRAME:
                if session.session_active:
                    _last_ws_activity = time.time()
                    if buffer_key not in _audio_reconnect_buffers:
                        _audio_reconnect_buffers[buffer_key] = deque(maxlen=_RECONNECT_BUFFER_MAX_FRAMES)
                    _audio_reconnect_buffers[buffer_key].append(payload)
                    _enqueue_frame(payload)

            else:
                await send_message(ws, MSG_ERROR, b"Unknown message type")

    except Exception as e:
        _ws_log(cid, "error", f"[Stream] Handler error: {e}")
    finally:
        # A8-P2-3: Unregister WS BEFORE the flush — frees the WS slot immediately so
        # other clients can connect while we finish the (potentially slow) final inference.
        _unregister_ws(ws)
        # Codex F3: only drop the reconnect buffer on a CLEAN END_SESSION. On an
        # abnormal disconnect (network drop, OEM kill, cert hiccup) we MUST keep
        # the buffer so the next /ws-audio-stream MSG_START_SESSION can replay
        # the in-flight audio. Previously we wiped unconditionally → the replay
        # path at line 951 was effectively dead for disconnect recovery.
        if clean_end:
            _audio_reconnect_buffers.pop(buffer_key, None)
        # If we exited without END_SESSION (disconnect / error), drain + flush anyway.
        if not processor_task.done():
            try:
                await _drain_and_flush(final=True)
            except Exception as e:
                _ws_log(cid, "debug", f"[Stream] Disconnect flush failed (WS closing): {e}")
        session.session_active = False
        # v1.2 history: ensure the session row is closed even on abnormal disconnect.
        # Idempotent (WHERE ended_at IS NULL guard) so safe even after MSG_END_SESSION.
        if session.history_session_id is not None:
            try:
                history_db.get_default_db().end_session(session.history_session_id)
            except Exception as e:
                log_warning(f"[History] end_session (disconnect) failed: {e}")
        _ws_log(cid, "info", f"[Stream] WebSocket handler ended (processed {session.frames_processed} frames)")

# =============================================================================
#                              TRACKPAD WEBSOCKET
# =============================================================================

import queue

# ─── UDP sideband for low-latency input (gyro, trackpad move, scroll) ───
# Bypasses TCP/WebSocket entirely — no head-of-line blocking.
# Auth: first packet must be JSON {"t":"auth","token":"<valid_token>"}.
# After auth, accepts: {"t":"a","x":...,"y":...}, {"t":"m","x":...,"y":...}, {"t":"s","dy":...}
#
# O-P1-3 / F-Apr23-01: auth is tracked by two coordinated dicts.
#   _udp_authed_clients: (ip, port) -> (auth_timestamp, token_hash)
#   _udp_token_to_addr:  token_hash -> (ip, port)
# The reverse index lets a NAT rebind / Wi-Fi↔cellular handoff land on a new
# (ip, port) without being treated as a new client: if the first packet from
# the new tuple carries a token whose hash is already mapped, we evict the
# stale (ip, port) and rewrite the mapping atomically — no wait for the 1 h
# TTL to reclaim the slot, no stale-entry log-noise. token_hash is
# sha256(token)[:16] (64 bits, same shape as F-Apr22-02's reconnect-buffer
# key) so the raw token never lands in a key that might surface in logs.
_udp_transport = None
_udp_authed_clients: "dict[tuple, tuple[float, str]]" = {}  # (ip, port) -> (auth_ts, token_hash)
_udp_token_to_addr: "dict[str, tuple]" = {}  # token_hash -> (ip, port)
_UDP_CLIENT_TTL = 3600  # 1 hour — evict stale entries
_UDP_MAX_CLIENTS = 50


def _udp_token_hash(token: str) -> str:
    """64-bit SHA-256 prefix of a UDP-auth token. Matches F-Apr22-02 shape."""
    return hashlib.sha256(token.encode()).hexdigest()[:16]


def _udp_evict(addr: tuple, authed: dict, reverse: dict) -> None:
    """Remove one (addr) entry from both dicts, keeping them consistent.
    Safe on addresses that are not present."""
    entry = authed.pop(addr, None)
    if entry is None:
        return
    _, token_hash = entry
    # Only drop the reverse mapping if it still points at this addr — another
    # rebind may have already rewritten it to the new (ip, port).
    if reverse.get(token_hash) == addr:
        del reverse[token_hash]


def _udp_process_auth(
    addr: tuple,
    token_hash: str,
    now: float,
    authed: dict,
    reverse: dict,
    ttl: float,
    max_clients: int,
) -> str:
    """Install an authenticated (addr, token_hash) pair, evicting stale
    entries as needed. Returns a short status string for logging:
      'new'       — first time we've seen this token_hash
      'rebind'    — token_hash was already authed at a different (ip, port);
                    the old tuple is evicted and the mapping rewritten
      'refresh'   — same (addr) and same token_hash, just a timestamp bump
    Pure-ish: only touches the two dicts passed in (hence testable in
    isolation — the module-level dicts are the production binding).
    Ordering: a NAT rebind MUST be handled before the max-clients sweep, or
    the rebind would count against the cap even though it's replacing, not
    adding, an entry."""
    prior_entry = authed.get(addr)  # may be (ts, old_hash) if re-pair on same socket
    prior_addr = reverse.get(token_hash)
    if prior_addr is not None and prior_addr != addr:
        # NAT rebind: same token, new (ip, port). Drop the stale tuple.
        _udp_evict(prior_addr, authed, reverse)
        status = "rebind"
    elif prior_entry is not None:
        if prior_entry[1] == token_hash:
            # Same (addr), same token — just a resend/refresh. Not a new slot.
            status = "refresh"
        else:
            # Same (addr), different token (re-pair without tuple change).
            # Not new capacity, but the OLD token_hash's reverse entry is now
            # stale — drop it before we install the new hash.
            if reverse.get(prior_entry[1]) == addr:
                del reverse[prior_entry[1]]
            status = "refresh"
    else:
        status = "new"

    # Only enforce max-clients on genuinely new slots. 'rebind' and 'refresh'
    # don't grow the table.
    if status == "new" and len(authed) >= max_clients:
        expired = [k for k, v in authed.items() if now - v[0] > ttl]
        for k in expired:
            _udp_evict(k, authed, reverse)
        if len(authed) >= max_clients:
            # Still full after expiry sweep — evict oldest by auth_ts.
            oldest = min(authed, key=lambda k: authed[k][0])
            _udp_evict(oldest, authed, reverse)

    authed[addr] = (now, token_hash)
    reverse[token_hash] = addr
    return status


class _UdpInputProtocol(asyncio.DatagramProtocol):
    """UDP listener for latency-critical input events (gyro, move, scroll)."""

    def connection_made(self, transport):
        global _udp_transport
        _udp_transport = transport
        log_info(f"[UDP] Input listener ready on port {_udp_port}")

    def datagram_received(self, data: bytes, addr: tuple):
        try:
            msg = json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return  # Malformed packet — drop silently

        t = msg.get("t")

        # Auth handshake — must be first packet from this client
        if t == "auth":
            token = msg.get("token", "")
            if _check_any_token(token):
                status = _udp_process_auth(
                    addr,
                    _udp_token_hash(token),
                    time.time(),
                    _udp_authed_clients,
                    _udp_token_to_addr,
                    _UDP_CLIENT_TTL,
                    _UDP_MAX_CLIENTS,
                )
                if status == "rebind":
                    log_info(f"[UDP] Client rebound: {addr[0]}:{addr[1]} (NAT/network change, same token)")
                else:
                    log_info(f"[UDP] Client authenticated: {addr[0]}:{addr[1]}")
                # Send ACK so Android knows UDP is working (fire-and-forget)
                try:
                    ack = b'{"t":"ack"}'
                    _udp_transport.sendto(ack, addr)
                except Exception:
                    pass
            else:
                log_warning(f"[UDP] Auth failed from {addr[0]}:{addr[1]}")
            return

        # Reject unauthenticated or expired clients
        entry = _udp_authed_clients.get(addr)
        if entry is None or (time.time() - entry[0] > _UDP_CLIENT_TTL):
            if entry is not None:
                # Stale — drop it and its reverse mapping.
                _udp_evict(addr, _udp_authed_clients, _udp_token_to_addr)
            return
        # Refresh on activity. Token hash doesn't change within a session.
        _udp_authed_clients[addr] = (time.time(), entry[1])

        # Resume app input on any UDP activity (matches WSS path at ws_trackpad)
        if is_app_input_paused():
            resume_app_input()

        # Route through the same _tp_queue + worker thread as WSS.
        # Calling mouse_move() directly here blocks the asyncio event loop
        # (pynput SendInput on Windows can take 1-2ms) and skips coalescing.
        _ensure_tp_thread()
        try:
            _tp_queue.put_nowait(msg)
        except queue.Full:
            pass  # Drop under overload — UDP semantics, loss is expected

    def connection_lost(self, exc):
        log_info("[UDP] Input listener closed")

_udp_port = 5001  # Sideband port for UDP input

# Dedicated trackpad input queue and thread (bypasses asyncio for minimal latency).
# queue.Queue is inherently thread-safe (uses internal deque + mutex) — no external lock needed.
_tp_queue = queue.Queue(maxsize=500)
_tp_thread = None
_tp_running = False
_tp_scroll_stop = threading.Event()  # Signaled by async context, checked by worker to drain scroll events

# Cursor position broadcast: worker thread writes latest pos after each move dispatch.
# Async sender in ws_trackpad polls at ~30Hz and sends to phone client.
_cursor_pos_latest = None       # (x, y) tuple, written by _tp_worker
# A8-P2-4: This is a single shared Event for all trackpad clients. If multiple
# phones connect simultaneously, they all share the same cursor broadcast signal.
# In practice this is fine — only one phone controls the cursor at a time, and the
# 30Hz poll means all connected clients see the same latest position. Per-client
# events would only matter if clients needed independent cursor state, which they don't.
_cursor_pos_changed = threading.Event()  # Signaled by worker on cursor move

def _update_cursor_broadcast():
    """Called by _tp_worker after dispatching a move event. Reads cursor pos from stt_common."""
    global _cursor_pos_latest
    pos = get_cursor_position()
    if pos is not None:
        sw, sh = get_screen_resolution()
        _cursor_pos_latest = (pos[0], pos[1], sw, sh)
        _cursor_pos_changed.set()

def _tp_worker():
    """Dedicated thread for processing trackpad input - minimal latency"""
    global _tp_running
    # Event types that are continuous (safe to discard when paused)
    _CONTINUOUS_TYPES = {"m", "a", "rc", "s"}
    while _tp_running:
        try:
            # Check scroll-stop event (set by async context, thread-safe via Event)
            if _tp_scroll_stop.is_set():
                _tp_scroll_stop.clear()
                # Drain only continuous events from queue, preserve discrete
                drained = 0
                preserved = []
                while not _tp_queue.empty():
                    try:
                        item = _tp_queue.get_nowait()
                        if item.get("t") in _CONTINUOUS_TYPES:
                            drained += 1
                        else:
                            preserved.append(item)
                    except queue.Empty:
                        break
                for item in preserved:
                    try:
                        _tp_queue.put_nowait(item)
                    except queue.Full:
                        break
                if drained:
                    log_debug(f"[Trackpad] scroll-stop: drained {drained} continuous events")

            msg = _tp_queue.get(timeout=0.005)
            if msg is None:
                break
            t = msg.get("t")
            # If paused, discard only continuous events (mouse move / scroll).
            # Discrete events (key press, click, drag) survive — they represent
            # intentional user actions that must not be silently eaten.
            if is_app_input_paused():
                if t in _CONTINUOUS_TYPES:
                    # Drain remaining continuous events
                    while not _tp_queue.empty():
                        try:
                            peek = _tp_queue.get_nowait()
                            if peek.get("t") not in _CONTINUOUS_TYPES:
                                # Non-continuous event — put it back and stop draining
                                _tp_queue.put(peek)
                                break
                        except queue.Empty:
                            break
                    continue
                # Discrete event while paused — force resume (phone is active,
                # user tapped a button) and fall through to dispatch
                resume_app_input(force=True)
            if t == "m":
                # Coalesce: sum up queued relative moves into one dispatch
                total_x, total_y = msg["x"], msg["y"]
                while not _tp_queue.empty():
                    try:
                        peek = _tp_queue.get_nowait()
                        if peek.get("t") == "m":
                            total_x += peek["x"]
                            total_y += peek["y"]
                        else:
                            _tp_queue.put(peek)
                            break
                    except queue.Empty:
                        break
                mouse_move(total_x, total_y)
                _update_cursor_broadcast()
            elif t == "a":
                # Coalesce: skip to latest absolute position (intermediate ones are stale)
                while not _tp_queue.empty():
                    try:
                        peek = _tp_queue.get_nowait()
                        if peek.get("t") == "a":
                            msg = peek  # newer absolute replaces current
                        else:
                            _tp_queue.put(peek)  # non-absolute event, put back
                            break
                    except queue.Empty:
                        break
                mouse_move_absolute(msg["x"], msg["y"])
                _update_cursor_broadcast()
            elif t == "rc":
                sw, sh = get_screen_resolution()
                mouse_move_absolute(sw // 2, sh // 2)
                _update_cursor_broadcast()
            elif t == "c":
                mouse_click(msg.get("b", 1))
            elif t == "d":
                mouse_drag(msg.get("a", "down"))
            elif t == "k":
                key_press(msg.get("k", ""))
            elif t == "txt":
                type_text(msg.get("text", ""))
            elif t == "s":
                raw_dy = msg.get("dy", 0)
                dy = max(-50, min(50, raw_dy))
                mouse_scroll(dy * 0.1)
        except queue.Empty:
            pass
        except Exception as e:
            log_warning(f"[Trackpad] dispatch error: {e}")

def _ensure_tp_thread():
    """Start trackpad worker thread if not running"""
    global _tp_thread, _tp_running
    if _tp_thread is None or not _tp_thread.is_alive():
        _tp_running = True
        _tp_thread = threading.Thread(target=_tp_worker, daemon=True)
        _tp_thread.start()

async def ws_trackpad(ws: WebSocket):
    """Trackpad WebSocket handler - queues to dedicated thread for minimal latency"""
    cid = _resolve_cid(ws)
    client_ip = ws.client.host if ws.client else "unknown"
    if not await _accept_authenticated_ws(ws, cid, "Trackpad"):
        return
    _ws_log(cid, "info", f"[Trackpad] Connected from {client_ip}")
    _ensure_tp_thread()

    # Send initial cursor position on connect (so phone overlay starts at correct pos)
    init_pos = get_cursor_position()
    if init_pos is not None:
        try:
            sw, sh = get_screen_resolution()
            await ws.send_text(json.dumps({"t": "cp", "x": init_pos[0], "y": init_pos[1], "sw": sw, "sh": sh}))
        except Exception:
            pass

    # Warn phone on connect if macOS Accessibility is missing (all trackpad input will silently fail)
    input_ok, input_err = check_input_health()
    if not input_ok:
        try:
            await ws.send_text(json.dumps({"error": input_err}))
        except Exception:
            pass

    # Cursor position sender: reads _cursor_pos_changed at ~30Hz,
    # sends latest cursor pos back to phone for latency-free overlay on screen mirror.
    cursor_sender_stop = asyncio.Event()
    async def _cursor_sender():
        loop = asyncio.get_event_loop()
        last_sent = None
        while not cursor_sender_stop.is_set():
            # Wait for cursor change signal (up to 33ms) via executor to avoid blocking event loop
            changed = await loop.run_in_executor(None, lambda: _cursor_pos_changed.wait(0.033))
            if cursor_sender_stop.is_set():
                break
            if changed:
                _cursor_pos_changed.clear()
            pos = _cursor_pos_latest  # (x, y, sw, sh) tuple
            if pos is not None and pos != last_sent:
                last_sent = pos
                try:
                    await ws.send_text(json.dumps({"t": "cp", "x": pos[0], "y": pos[1], "sw": pos[2], "sh": pos[3]}))
                except Exception:
                    break  # WS closed

    cursor_task = asyncio.create_task(_cursor_sender())
    cursor_task.add_done_callback(_task_done)

    try:
        while True:
            try:
                data = await ws.receive_text()
            except WebSocketDisconnect:
                break
            except Exception as e:
                _ws_log(cid, "warning", f"[Trackpad] WS receive error: {e}")
                break

            if not data:
                continue

            # Phone activity → resume app input if paused
            if is_app_input_paused():
                resume_app_input()

            try:
                m = json.loads(data)
                if m.get('t') == 'ss':
                    # Scroll stop — signal the worker thread via Event to drain
                    # continuous events. This avoids the async context touching
                    # the queue concurrently with the worker thread.
                    _tp_scroll_stop.set()
                else:
                    _tp_queue.put_nowait(m)
            except queue.Full:
                _ws_log(cid, "warning", "[Trackpad] Queue full, dropping event")
            except Exception as e:
                _ws_log(cid, "warning", f"[Trackpad] WS message parse error: {e}")
    finally:
        cursor_sender_stop.set()
        _cursor_pos_changed.set()  # Wake sender so it can exit cleanly
        cursor_task.cancel()
        try:
            await cursor_task
        except (asyncio.CancelledError, Exception):
            pass
        _unregister_ws(ws)
        _ws_log(cid, "info", "[Trackpad] Disconnected")

# =============================================================================
#                              FASTAPI APP
# =============================================================================

def init_models(preference="balanced", direct_model=None, verbose=True):
    """Initialize Whisper and VAD models with preflight checks"""
    global whisper_model, vad_model, vad_utils, executor, io_executor, _model_load_executor
    global _server_state, _loaded_model_name

    setup_logging()

    # Thread pools always needed (even in IDLE for IO tasks like blink)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
    io_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="io")
    _model_load_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-load")

    # Load persisted auth token + pair lock state from config (survive restarts)
    _load_auth_token()
    _load_pair_lock()
    _load_pair_code()
    # F-Apr22-03 (+ F-Apr22-04 fix): session load + background writer must run
    # in BOTH modes. Previously nested inside `if _SERVICE_MODE:` which meant
    # legacy startup silently lost session persistence (every _mark_sessions_dirty
    # no-op'd because the writer never spun up).
    _load_client_sessions()
    _start_sessions_writer()

    if _SERVICE_MODE:
        # Service mode: start IDLE, no model loaded, load on-demand via API
        init_gpu()
        vad_model, vad_utils = load_vad()
        log_info(f"VAD: {get_vad_type()}")
        _server_state = ServerState.IDLE
        _loaded_model_name = None
        log_info("Service mode — IDLE, waiting for model load via API")
        return

    # Run preflight checks - prints startup summary if verbose
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pf = preflight.run(mode="web", preference=preference or "balanced", verbose=verbose, script_dir=script_dir)

    # Use preflight-detected GPU
    init_gpu()
    vram_before, _, _ = get_gpu_stats()

    # Load VAD (preflight already checked torch availability)
    vad_model, vad_utils = load_vad()
    vad_type = get_vad_type()
    log_info(f"VAD: {vad_type}")

    # Load Whisper using direct model, preflight-selected config, or custom
    if direct_model:
        model_name, compute_type = direct_model
        device = "cuda" if pf.has_gpu else "cpu"
    elif preference == "custom":
        model_name, compute_type, device = custom_model_selection()
    elif pf.transcription:
        model_name = pf.transcription.model
        compute_type = pf.transcription.compute_type
        device = pf.transcription.device
    else:
        model_name, compute_type, device = select_model(preference)

    global _loaded_device
    whisper_model, actual_device, actual_precision = load_whisper(
        model_name, device=device, compute_type=compute_type
    )
    _loaded_model_name = model_name
    # Record actual device (not requested): load_whisper silently falls back
    # cuda→cpu on failure. Reporting "cuda" when we're on CPU misled users
    # into blaming "GPU is slow" when their GPU never loaded at all.
    _loaded_device = actual_device
    _server_state = ServerState.ACTIVE
    if actual_device != device:
        log_warning(f"Whisper loaded on {actual_device} (requested {device})")

    vram_after, _, _ = get_gpu_stats()
    if vram_after > vram_before:
        log_info(f"VRAM used: {vram_after - vram_before:.2f} GB")

    log_info("Models loaded - ready to accept connections")


def _load_model_async(model_name, precision=None, force_device=None):
    """Load a Whisper model (called from API). Runs in executor."""
    global whisper_model, _server_state, _loaded_model_name, _loaded_device, _model_load_progress, _model_load_cancel

    if _server_state == ServerState.LOADING:
        return {"error": "Already loading a model"}

    _server_state = ServerState.LOADING
    _model_load_progress = 0.0
    _model_load_cancel = False

    try:
        import torch
        has_gpu = torch.cuda.is_available()

        # If CUDA unavailable but GPU physically exists, try recovery at load time
        # (catches cases where init_gpu() recovery failed at startup)
        if not has_gpu:
            import shutil
            if shutil.which('nvidia-smi'):
                from stt_common import _try_cuda_recovery
                log_info("CUDA unavailable at model load — attempting recovery")
                for attempt in range(2):
                    if _try_cuda_recovery():
                        import importlib
                        importlib.reload(torch.cuda)
                        if torch.cuda.is_available():
                            has_gpu = True
                            log_info(f"CUDA recovered at model load (attempt {attempt + 1})")
                            break

        if force_device in ("cuda", "cpu"):
            device = force_device
        else:
            device = "cuda" if has_gpu else "cpu"

        if precision is None:
            if device == "cuda":
                from stt_common import auto_select_precision, get_vram_free, gpu_supports_float16
                vram = get_vram_free()
                fp16 = gpu_supports_float16()
                precision = auto_select_precision(model_name, vram, fp16)
            else:
                precision = "int8"

        _model_load_progress = 0.1
        if _model_load_cancel:
            _server_state = ServerState.IDLE
            return {"status": "cancelled"}

        # Unload previous model if any
        with _whisper_lock:
            if whisper_model is not None:
                whisper_model = None
                if has_gpu:
                    import gc
                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

        _model_load_progress = 0.3

        try:
            new_model, actual_device, actual_precision = load_whisper(
                model_name, device=device, compute_type=precision
            )
        except Exception as cuda_err:
            if device == "cuda":
                log_info(f"CUDA load failed ({cuda_err}), falling back to CPU with int8")
                new_model, actual_device, actual_precision = load_whisper(
                    model_name, device="cpu", compute_type="int8"
                )
            else:
                raise
        # Use actual values returned by load_whisper (it may have fallen back
        # internally). device/precision args are the REQUEST; actual_* is the TRUTH.
        device = actual_device
        precision = actual_precision

        # Check cancel again after the expensive load_whisper() call
        if _model_load_cancel:
            new_model = None
            with _whisper_lock:
                whisper_model = None
            _loaded_model_name = None
            if has_gpu:
                import gc
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            _server_state = ServerState.IDLE
            _model_load_progress = 0.0
            log_info("Model load cancelled after download")
            return {"status": "cancelled"}

        with _whisper_lock:
            whisper_model = new_model
        _loaded_model_name = model_name
        _loaded_device = device
        _model_load_progress = 1.0
        _server_state = ServerState.ACTIVE
        log_info(f"Model loaded: {model_name} ({precision} on {device})")
        return {"status": "ready", "model": model_name, "precision": precision, "device": device}

    except Exception as e:
        _server_state = ServerState.IDLE
        _model_load_progress = 0.0
        log_error(f"Model load failed: {e}")
        # A9-P2-10: Don't leak internal exception details to client
        return {"error": "Model load failed. Check server logs for details."}


def _unload_model():
    """Unload current model and free VRAM."""
    global whisper_model, _server_state, _loaded_model_name
    with _whisper_lock:
        if whisper_model is not None:
            whisper_model = None
            _loaded_model_name = None
            try:
                import gc
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except Exception as e:
                log_warning(f"CUDA cache clear failed: {e}")
            _server_state = ServerState.IDLE
            log_info("Model unloaded, VRAM freed")

# =============================================================================
#                         INPUT GUARD — WS TRACKING + BROADCAST
# =============================================================================

_active_audio_ws = set()
_ws_track_lock = threading.Lock()
_event_loop = None
_input_guard_enabled = True  # Set False by --no-input-guard

def _register_ws(ws):
    """Register WS and check concurrent limit atomically. Returns True if registered, False if limit exceeded."""
    global _last_ws_activity
    with _ws_track_lock:
        if len(_active_audio_ws) >= _MAX_CONCURRENT_WS:
            return False
        _active_audio_ws.add(ws)
    _last_ws_activity = time.time()
    return True

def _unregister_ws(ws):
    global _last_ws_activity
    with _ws_track_lock:
        _active_audio_ws.discard(ws)
    _last_ws_activity = time.time()

async def _broadcast_pause_state(paused):
    """Send INPUT_PAUSED/RESUMED to all connected phone clients"""
    wire = _WIRE_INPUT_PAUSED if paused else _WIRE_INPUT_RESUMED
    with _ws_track_lock:
        connections = list(_active_audio_ws)
    for ws in connections:
        try:
            await _send_raw(ws, wire)
        except Exception as e:
            log_debug(f"[InputGuard] broadcast to WS failed: {e}")

def _on_pause_change(paused):
    """Sync callback from stt_common → schedule async broadcast to phones"""
    if _event_loop and not _event_loop.is_closed():
        _event_loop.call_soon_threadsafe(
            asyncio.ensure_future, _broadcast_pause_state(paused)
        )

async def _auto_unload_timer():
    """Service mode: unload model after 15min no WS connections."""
    while True:
        await asyncio.sleep(30)  # Check every 30s for tighter unload precision

        # P2-3: Periodic cleanup of rate limit dicts to prevent unbounded growth.
        # Remove IPs whose attempt lists are empty or fully expired.
        now = time.time()
        for rate_dict, window in [(_pair_attempts, _PAIR_RATE_WINDOW), (_auth_attempts, _AUTH_LOCKOUT)]:
            stale_ips = [
                ip for ip, attempts in rate_dict.items()
                if not attempts or all(now - t >= window for t in attempts)
            ]
            for ip in stale_ips:
                del rate_dict[ip]

        if not _SERVICE_MODE or _server_state != ServerState.ACTIVE:
            continue
        with _ws_track_lock:
            has_connections = len(_active_audio_ws) > 0
        if has_connections:
            continue
        elapsed = time.time() - _last_ws_activity
        if elapsed >= _MODEL_UNLOAD_TIMEOUT:
            log_info(f"No WS connections for {int(elapsed)}s — unloading model")
            _unload_model()

_WATCHDOG_GRACE_SECONDS = 10
_watchdog_armed = False

async def _watchdog_shutdown_timer():
    """Service mode: auto-shutdown after all clients disconnect."""
    global _watchdog_armed
    while True:
        await asyncio.sleep(5)
        if not _SERVICE_MODE:
            continue
        with _ws_track_lock:
            has_connections = len(_active_audio_ws) > 0
        if has_connections:
            _watchdog_armed = True
            continue
        if not _watchdog_armed:
            continue
        elapsed = time.time() - _last_ws_activity
        if elapsed >= _WATCHDOG_GRACE_SECONDS:
            log_info(f"[Watchdog] No clients for {int(elapsed)}s — shutting down server")
            os.kill(os.getpid(), signal.SIGTERM)
            return

def _set_tcp_nodelay_on_server():
    """N-P0-1: Set TCP_NODELAY on uvicorn's listening sockets to disable Nagle's algorithm.
    Reduces small-packet latency by 40-200ms for WS control messages."""
    try:
        loop = asyncio.get_event_loop()
        for server in getattr(loop, '_servers', []):
            for sock in server.sockets:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                log_debug(f"[Net] TCP_NODELAY set on {sock.getsockname()}")
    except Exception as e:
        log_debug(f"[Net] TCP_NODELAY set via loop._servers failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler - models already loaded in main"""
    global _event_loop, _model_load_lock, _pair_mutex
    _event_loop = asyncio.get_event_loop()

    # Create async locks on the running loop. Must happen inside lifespan;
    # asyncio.Lock() at module import time would bind to whichever loop is
    # current at import, which may not be the request-serving loop.
    _model_load_lock = asyncio.Lock()
    _pair_mutex = asyncio.Lock()

    # N-P0-1: Schedule TCP_NODELAY after uvicorn binds sockets (deferred to next tick)
    _event_loop.call_soon(_set_tcp_nodelay_on_server)

    # Start UDP sideband for low-latency input (gyro, trackpad move, scroll)
    udp_transport = None
    try:
        udp_transport, _ = await _event_loop.create_datagram_endpoint(
            _UdpInputProtocol,
            local_addr=("0.0.0.0", _udp_port),
        )
    except Exception as e:
        log_warning(f"[UDP] Failed to start input listener on port {_udp_port}: {e}")

    # Probe input health at startup (macOS Accessibility, etc.)
    input_ok, input_err = check_input_health()
    if not input_ok:
        log_warning(f"[Input] {input_err}")
    else:
        log_info("[Input] Input dispatch health check passed")

    log_info("Server ready")

    # Start physical input monitoring (Linux: /dev/input, Win/Mac: pynput listeners)
    if _input_guard_enabled:
        register_pause_callback(_on_pause_change)
        if input_monitor.start_monitor(pause_app_input):
            log_info("Physical input monitoring active")
        else:
            log_info("Physical input monitoring unavailable (permissions or pynput missing)")

    # Start auto-unload timer for service mode
    unload_task = None
    watchdog_task = None
    if _SERVICE_MODE:
        unload_task = asyncio.create_task(_auto_unload_timer())
        unload_task.add_done_callback(_task_done)
        watchdog_task = asyncio.create_task(_watchdog_shutdown_timer())
        watchdog_task.add_done_callback(_task_done)

    # systemd watchdog: send WATCHDOG=1 heartbeat if running under systemd
    sd_watchdog_task = None
    notify_socket = os.environ.get("NOTIFY_SOCKET")
    if notify_socket:
        async def _sd_watchdog_loop():
            """Send sd_notify WATCHDOG=1 at half the WatchdogSec interval."""
            import socket as _sock
            watchdog_usec = int(os.environ.get("WATCHDOG_USEC", "0"))
            if watchdog_usec <= 0:
                return
            interval = watchdog_usec / 2_000_000  # Half interval in seconds
            addr = notify_socket
            if addr.startswith("@"):
                addr = "\0" + addr[1:]  # Abstract socket
            sock = _sock.socket(_sock.AF_UNIX, _sock.SOCK_DGRAM)
            try:
                while True:
                    sock.sendto(b"WATCHDOG=1", addr)
                    await asyncio.sleep(interval)
            except asyncio.CancelledError:
                pass
            finally:
                sock.close()

        # Notify systemd we're ready
        try:
            import socket as _sock
            _sd_sock = _sock.socket(_sock.AF_UNIX, _sock.SOCK_DGRAM)
            _sd_addr = notify_socket if not notify_socket.startswith("@") else "\0" + notify_socket[1:]
            _sd_sock.sendto(b"READY=1", _sd_addr)
            _sd_sock.close()
        except Exception as e:
            log_debug(f"[systemd] sd_notify READY failed: {e}")

        sd_watchdog_task = asyncio.create_task(_sd_watchdog_loop())
        sd_watchdog_task.add_done_callback(_task_done)
        log_info("[systemd] Watchdog heartbeat active")

    # D-Bus lid close listener (Linux only) — close WS connections before sleep
    lid_close_task = None
    if PLATFORM == 'linux':
        async def _dbus_sleep_listener():
            """Listen for org.freedesktop.login1 PrepareForSleep signal.
            When laptop lid closes, close all WS cleanly so clients auto-reconnect on wake."""
            try:
                import dbus
                from dbus.mainloop.glib import DBusGMainLoop
                DBusGMainLoop(set_as_default=True)
                bus = dbus.SystemBus()
                def on_prepare_sleep(sleeping):
                    if sleeping:
                        log_info("[D-Bus] System going to sleep — clients will auto-reconnect on wake")
                    else:
                        log_info("[D-Bus] System woke up")
                bus.add_signal_receiver(
                    on_prepare_sleep,
                    signal_name='PrepareForSleep',
                    dbus_interface='org.freedesktop.login1.Manager',
                    bus_name='org.freedesktop.login1',
                )
                # Keep the listener alive
                while True:
                    await asyncio.sleep(60)
            except ImportError:
                log_debug("[D-Bus] python-dbus not installed — lid close detection disabled")
            except Exception as e:
                log_debug(f"[D-Bus] Sleep listener failed: {e}")

        lid_close_task = asyncio.create_task(_dbus_sleep_listener())
        lid_close_task.add_done_callback(_task_done)

    yield

    # Cleanup
    log_info("[Shutdown] Starting graceful shutdown...")

    # F-Apr22-03: flush any pending session writes BEFORE other teardown
    # so even a crash during GPU cleanup still leaves durable state on disk.
    try:
        await _drain_sessions_writer()
    except Exception as e:
        log_warning(f"[Shutdown] sessions drain error (non-fatal): {e}")

    if udp_transport:
        udp_transport.close()
    if unload_task:
        unload_task.cancel()
    if watchdog_task:
        watchdog_task.cancel()
    if sd_watchdog_task:
        sd_watchdog_task.cancel()
    if lid_close_task:
        lid_close_task.cancel()
    input_monitor.stop_monitor()

    # P2-29: Kill all active ffmpeg processes on shutdown (before GPU cleanup —
    # NVENC ffmpeg holds its own CUDA context that must be released first)
    with _ffmpeg_procs_lock:
        for proc in list(_active_ffmpeg_procs):
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=1)
            except Exception as e:
                log_warning(f"[Shutdown] ffmpeg cleanup error: {e}")
        _active_ffmpeg_procs.clear()

    # Wait for any in-flight Whisper inference to finish before releasing model.
    # wait=True with a short timeout so we don't hang on stuck threads.
    global executor
    if executor:
        executor.shutdown(wait=True, cancel_futures=True)
        executor = None

    # GPU cleanup: release CTranslate2 model + PyTorch tensors + CUDA cache + pynvml
    global whisper_model, vad_model, vad_utils
    try:
        with _whisper_lock:
            if whisper_model is not None:
                whisper_model = None
                log_info("[Shutdown] Whisper model released")
        if vad_model is not None:
            vad_model = None
            vad_utils = None
            log_info("[Shutdown] VAD model released")

        # Force Python GC before CUDA cleanup so all model tensor refs are dropped
        import gc
        gc.collect()

        # Flush CUDA memory and synchronize pending ops
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            log_info("[Shutdown] CUDA cache flushed")

        # Release pynvml handle
        cleanup_gpu()
        log_info("[Shutdown] pynvml shutdown complete")
    except Exception as e:
        log_warning(f"[Shutdown] GPU cleanup error (non-fatal): {e}")

    if io_executor:
        io_executor.shutdown(wait=False, cancel_futures=True)
    if _model_load_executor:
        _model_load_executor.shutdown(wait=False, cancel_futures=True)
    log_info("Server stopped")

app = FastAPI(lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=500)  # Compress responses >500 bytes (HTML=43KB->10KB)

# RFC 1918 middleware — reject non-LAN IPs in service mode
import ipaddress as _ipaddr
_LAN_NETS = [
    _ipaddr.ip_network("10.0.0.0/8"),
    _ipaddr.ip_network("172.16.0.0/12"),
    _ipaddr.ip_network("192.168.0.0/16"),
    _ipaddr.ip_network("127.0.0.0/8"),
    _ipaddr.ip_network("::1/128"),
    _ipaddr.ip_network("fe80::/10"),   # IPv6 link-local
    _ipaddr.ip_network("fc00::/7"),    # IPv6 ULA (private)
]
_MAX_CONCURRENT_WS = 8  # 4 WS types (audio-in/trackpad/screen/audio-out) + headroom for reconnect overlap

@app.middleware("http")
async def lan_only_middleware(request: Request, call_next):
    """Reject non-LAN IPs in service mode (RFC 1918 check)."""
    if _SERVICE_MODE:
        try:
            client_ip = _ipaddr.ip_address(request.client.host)
            # Unwrap IPv4-mapped IPv6 (e.g., ::ffff:192.168.1.5 -> 192.168.1.5)
            if hasattr(client_ip, 'ipv4_mapped') and client_ip.ipv4_mapped:
                client_ip = client_ip.ipv4_mapped
            if not any(client_ip in net for net in _LAN_NETS):
                return Response(status_code=403)
        except ValueError as e:
            log_debug(f"[LAN] Could not parse client IP '{request.client.host}': {e}")
    response = await call_next(request)
    # Security headers
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    # A19-P2-5: CSP — allows inline styles/scripts (embedded web client), wss: for WebSocket,
    # blob: for AudioWorklet, media: for video. Defense-in-depth against XSS injection.
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "connect-src 'self' wss: ws:; "
        "media-src 'self' blob:; "
        "worker-src 'self' blob:; "
        "img-src 'self' data:; "
        "frame-ancestors 'none'"
    )
    return response

# =============================================================================
#                         SCREEN MIRRORING (ffmpeg + NVENC)
# =============================================================================

# H.264 NAL unit types
_NAL_SLICE     = 1   # Non-IDR slice (P-frame)
_NAL_IDR       = 5   # IDR slice (keyframe)
_NAL_SEI       = 6   # Supplemental enhancement info
_NAL_SPS       = 7   # Sequence parameter set
_NAL_PPS       = 8   # Picture parameter set
_NAL_AUD       = 9   # Access unit delimiter

_SCREEN_FLAG_KEY = 0x01  # Keyframe flag in wire format

# Screen quality control: client → server [0x21][level:1]
# Levels: 0=480p, 1=720p, 2=1080p
MSG_QUALITY_CTRL = 0x21
_QUALITY_LEVELS = {0: '480p', 1: '720p', 2: '1080p'}
# A11-P3-1: Module-level constant (was re-created per WS session)
_RES_PRESETS = {'1080p': (1920, 1080), '720p': (1280, 720), '480p': (854, 480)}

def _get_nal_type_at(buf, start_code_pos):
    """Get NAL unit type byte at a start code position."""
    if buf[start_code_pos + 2] == 1:  # 3-byte start code
        return buf[start_code_pos + 3] & 0x1F
    else:  # 4-byte start code
        return buf[start_code_pos + 4] & 0x1F


def _read_h264_frames(pipe, frame_queue: queue.Queue, stop_event: threading.Event):
    """
    Thread: read ffmpeg stdout, split H.264 Annex B stream into access units (frames).

    H.264 Annex B stream from libx264 with baseline profile emits:
      [SPS] [PPS] [IDR] ... [Slice] [Slice] ... [SPS] [PPS] [IDR] ...

    Strategy: collect all NAL units between two consecutive VCL NALs (slice/IDR).
    A keyframe = SPS + PPS + IDR bundled together.  A P-frame = standalone slice.
    Non-VCL NALs (SEI, AUD) are included in the next frame's access unit.
    """
    buf = bytearray()
    start_time = time.monotonic()

    # Accumulator: list of (start_pos, end_pos) for NALs in current access unit
    pending_nals = bytearray()
    pending_is_key = False

    while not stop_event.is_set():
        try:
            chunk = pipe.read(65536)  # 64KB reads
            if not chunk:
                break  # ffmpeg exited / pipe closed
            buf.extend(chunk)
        except (OSError, ValueError):
            break

        # Parse all complete NAL units from buffer
        while True:
            # Find first start code
            sc1 = _find_start_code(buf, 0)
            if sc1 < 0:
                break
            # Discard any junk before first start code
            if sc1 > 0:
                del buf[:sc1]
                sc1 = 0

            # Find next start code (marks end of current NAL, start of next)
            sc2 = _find_start_code(buf, sc1 + 3)
            if sc2 < 0:
                break  # NAL not yet complete, need more data from pipe

            # Extract this NAL
            nal_bytes = bytes(buf[sc1:sc2])
            # NAL type: first byte after start code, masked to 5 bits
            if buf[sc1 + 2] == 1:  # 3-byte start code (00 00 01)
                nal_type = buf[sc1 + 3] & 0x1F
            else:                   # 4-byte start code (00 00 00 01)
                nal_type = buf[sc1 + 4] & 0x1F

            # Consume from buffer
            del buf[:sc2]

            # VCL NAL (actual picture data) = slice or IDR
            is_vcl = nal_type in (_NAL_SLICE, _NAL_IDR)

            if is_vcl:
                # This NAL completes an access unit — bundle pending + this NAL
                pending_nals.extend(nal_bytes)
                is_key = pending_is_key or (nal_type == _NAL_IDR)

                timestamp_ms = int((time.monotonic() - start_time) * 1000)
                try:
                    frame_queue.put_nowait((is_key, timestamp_ms, bytes(pending_nals)))
                except queue.Full:
                    # N-P2-6: Never drop keyframes — evict oldest frame to make room
                    if is_key:
                        try:
                            frame_queue.get_nowait()  # Evict oldest
                            frame_queue.put_nowait((is_key, timestamp_ms, bytes(pending_nals)))
                        except (queue.Empty, queue.Full):
                            pass  # Should not happen, but guard anyway
                    # P-frames can be dropped safely under backpressure

                # Reset accumulator for next access unit
                pending_nals = bytearray()
                pending_is_key = False
            else:
                # Non-VCL NAL (SPS, PPS, SEI, AUD, etc.) — accumulate
                pending_nals.extend(nal_bytes)
                if nal_type == _NAL_SPS:
                    pending_is_key = True

    # Signal end to consumer — sentinel MUST be delivered, so evict oldest if full
    try:
        frame_queue.put_nowait(None)
    except queue.Full:
        try:
            frame_queue.get_nowait()  # Evict oldest frame to make room
            frame_queue.put_nowait(None)
        except (queue.Empty, queue.Full):
            pass  # Should not happen after eviction, but guard anyway


def _find_start_code(buf, offset=0):
    """Find next H.264 start code (00 00 00 01 or 00 00 01) in buffer.
    Uses C-level memchr via bytes.find() instead of byte-by-byte Python loop."""
    # Search for 3-byte start code (00 00 01) — this also finds 4-byte (00 00 00 01)
    # since 00 00 01 appears at position+1 within 00 00 00 01.
    idx = buf.find(b'\x00\x00\x01', offset)
    if idx < 0:
        return -1
    # Check if preceded by 0x00 making it a 4-byte start code (00 00 00 01)
    if idx > 0 and idx - 1 >= offset and buf[idx - 1] == 0:
        return idx - 1  # 4-byte start code starts one byte earlier
    return idx  # 3-byte start code


def _get_avfoundation_screen_device(monitor_index=0):
    """Parse avfoundation device list to find screen capture device index.
    Returns device string like '2:' for -i flag. Defaults to '1:' if parsing fails.
    monitor_index: 0=primary, 1=secondary, etc. Matches CGGetActiveDisplayList order.
    UNTESTED: requires real multi-monitor macOS hardware."""
    try:
        p = subprocess.run(
            ['ffmpeg', '-nostdin', '-f', 'avfoundation', '-list_devices', 'true', '-i', ''],
            capture_output=True, text=True, timeout=5,
        )
        # Collect ALL "Capture screen N" devices — avfoundation lists them in
        # the same order as CGGetActiveDisplayList (documented Apple behavior)
        screen_devices = []
        for line in (p.stderr or '').splitlines():
            if 'Capture screen' in line:
                m = re.search(r'\[(\d+)\]', line)
                if m:
                    screen_devices.append(m.group(1))
        if screen_devices:
            idx = min(monitor_index, len(screen_devices) - 1)
            return f'{screen_devices[idx]}:'
    except Exception:
        pass
    return '1:'  # Default: screen is usually device index 1

def _check_hw_encoder():
    """Probe hardware encoder availability with a short real capture test.
    Returns encoder name string ('h264_nvenc', 'h264_vaapi', 'h264_qsv',
    'h264_videotoolbox') or None.

    Linux probe order (2026): NVENC → VAAPI → QSV → CPU. VAAPI added because
    NVENC has been intermittently broken on the user's PC (CUDA_ERROR_UNKNOWN
    after rapid kill/restart cycles); VAAPI uses a different driver path
    (DRI render node, not CUDA context) so it survives those failures.
    Encode latency: NVENC 2-5ms, VAAPI 3-6ms, QSV 4-8ms, libx264 8-15ms.
    """
    plat = PLATFORM
    if plat == 'linux':
        display = os.environ.get('DISPLAY', ':0')

        def _probe(extra_args, encoder_name):
            try:
                p = subprocess.run(
                    ['ffmpeg', '-nostdin', '-y', '-loglevel', 'error',
                     *extra_args,
                     '-f', 'x11grab', '-framerate', '10', '-video_size', '64x64',
                     '-i', display, '-frames:v', '3',
                     '-vf', 'format=nv12,hwupload' if encoder_name == 'h264_vaapi' else 'format=nv12',
                     '-c:v', encoder_name, '-f', 'null', '-'],
                    capture_output=True, timeout=8,
                )
                return p.returncode == 0
            except Exception as e:
                log_debug(f"[Screen] {encoder_name} probe failed: {e}")
                return False

        # 1. NVENC (lowest latency when it works)
        try:
            p = subprocess.run(
                ['ffmpeg', '-nostdin', '-y', '-loglevel', 'error',
                 '-f', 'x11grab', '-framerate', '10', '-video_size', '64x64',
                 '-i', display, '-frames:v', '3',
                 '-c:v', 'h264_nvenc', '-f', 'null', '-'],
                capture_output=True, timeout=8,
            )
            if p.returncode == 0:
                return 'h264_nvenc'
        except Exception as e:
            log_debug(f"[Screen] NVENC probe failed: {e}")

        # 2. VAAPI — universal fallback (Intel iGPU, AMD VCN, NVIDIA via interop)
        # Requires /dev/dri/renderD128 (or 129). If neither device exists, skip.
        for dri_node in ('/dev/dri/renderD128', '/dev/dri/renderD129'):
            if os.path.exists(dri_node):
                if _probe(['-vaapi_device', dri_node], 'h264_vaapi'):
                    log_info(f"[Screen] VAAPI available via {dri_node}")
                    return 'h264_vaapi'

        # 3. Intel QuickSync (`h264_qsv`)
        if _probe([], 'h264_qsv'):
            return 'h264_qsv'
    elif plat == 'windows':
        try:
            p = subprocess.run(
                ['ffmpeg', '-nostdin', '-y', '-loglevel', 'error',
                 '-f', 'gdigrab', '-framerate', '10', '-video_size', '64x64',
                 '-i', 'desktop', '-frames:v', '3',
                 '-c:v', 'h264_nvenc', '-f', 'null', '-'],
                capture_output=True, timeout=8,
                creationflags=0x08000000,  # CREATE_NO_WINDOW
            )
            if p.returncode == 0:
                return 'h264_nvenc'
        except Exception as e:
            log_debug(f"[Screen] NVENC probe failed on Windows: {e}")
        # Try QSV on Windows too (Intel laptops are everywhere)
        try:
            p = subprocess.run(
                ['ffmpeg', '-nostdin', '-y', '-loglevel', 'error',
                 '-f', 'gdigrab', '-framerate', '10', '-video_size', '64x64',
                 '-i', 'desktop', '-frames:v', '3',
                 '-c:v', 'h264_qsv', '-f', 'null', '-'],
                capture_output=True, timeout=8,
                creationflags=0x08000000,
            )
            if p.returncode == 0:
                return 'h264_qsv'
        except Exception as e:
            log_debug(f"[Screen] QSV probe failed on Windows: {e}")
    elif plat == 'macos':
        try:
            screen_dev = _get_avfoundation_screen_device()
            p = subprocess.run(
                ['ffmpeg', '-nostdin', '-y', '-loglevel', 'error',
                 '-f', 'avfoundation', '-framerate', '10', '-video_size', '64x64',
                 '-i', screen_dev, '-frames:v', '3',
                 '-c:v', 'h264_videotoolbox', '-f', 'null', '-'],
                capture_output=True, timeout=8,
            )
            if p.returncode == 0:
                return 'h264_videotoolbox'
        except Exception as e:
            log_debug(f"[Screen] VideoToolbox probe failed on macOS: {e}")
    return None

_hw_encoder = None  # None = untested, str = encoder name, False = none available

def _get_ffmpeg_display():
    """Return the X11 display string. Linux only."""
    if PLATFORM != 'linux':
        return None
    display = os.environ.get('DISPLAY', ':0')
    # Verify it works; if not, probe common displays
    try:
        r = subprocess.run(['xdpyinfo', '-display', display],
                           capture_output=True, timeout=2)
        if r.returncode == 0:
            return display
    except Exception as e:
        log_debug(f"[Screen] xdpyinfo probe failed for {display}: {e}")
    for d in [':0', ':1', ':2', ':3']:
        try:
            r = subprocess.run(['xdpyinfo', '-display', d],
                               capture_output=True, timeout=2)
            if r.returncode == 0:
                log_info(f"[Screen] Auto-detected DISPLAY={d}")
                os.environ['DISPLAY'] = d
                return d
        except Exception as e:
            log_debug(f"[Screen] xdpyinfo probe failed for {d}: {e}")
            continue
    return display  # Last resort, use whatever was set

def _build_ffmpeg_cmd(out_w, out_h, fps, hw_encoder=None,
                      screen_w=None, screen_h=None, display=None,
                      monitor_x=0, monitor_y=0, monitor_index=0):
    """Build ffmpeg command for screen capture (pure function, no side effects).

    Captures at full screen resolution and scales down to out_w x out_h
    so the entire desktop is visible (no cropping).
    monitor_x/monitor_y: pixel offset for multi-monitor (x11grab/gdigrab).
    Platform-aware: x11grab (Linux), gdigrab (Windows), avfoundation (macOS).
    A7-P1-4: On Wayland, x11grab captures via XWayland compatibility layer. This captures
    XWayland windows only (not native Wayland windows). PipeWire/wlr-screencopy would be needed
    for native capture but have no ffmpeg input support as of 2026.
    """
    plat = PLATFORM
    cap_w = screen_w or out_w
    cap_h = screen_h or out_h

    # Resolution-aware bitrate control (based on OUTPUT resolution)
    pixels = out_w * out_h
    if pixels > 1500000:       # 1080p+
        bitrate, maxrate, bufsize, level = '4M', '6M', '8M', '4.1'
    elif pixels > 500000:      # 720p
        bitrate, maxrate, bufsize, level = '2M', '3M', '4M', '4.0'
    else:                      # 480p and below
        bitrate, maxrate, bufsize, level = '1M', '1.5M', '2M', '3.1'

    cmd = ['ffmpeg', '-nostdin', '-loglevel', 'error']

    # VAAPI requires the device to be initialized BEFORE the input. If we're going
    # to use VAAPI encoding, declare the render node up front. /dev/dri/renderD128
    # is the standard name; renderD129 covers dual-GPU systems.
    if hw_encoder == 'h264_vaapi':
        for dri_node in ('/dev/dri/renderD128', '/dev/dri/renderD129'):
            if os.path.exists(dri_node):
                cmd += ['-vaapi_device', dri_node]
                break

    # Platform-specific input
    if plat == 'linux':
        # x11grab: display+offset for multi-monitor, e.g. ":0+2560,0"
        grab_input = display or ':0'
        if monitor_x or monitor_y:
            grab_input = f'{grab_input}+{monitor_x},{monitor_y}'
        cmd += [
            '-f', 'x11grab',
            '-framerate', str(fps),
            '-video_size', f'{cap_w}x{cap_h}',
            '-i', grab_input,
        ]
    elif plat == 'windows':
        cmd += [
            '-f', 'gdigrab',
            '-framerate', str(fps),
            '-offset_x', str(monitor_x), '-offset_y', str(monitor_y),
            '-video_size', f'{cap_w}x{cap_h}',
            '-i', 'desktop',
        ]
    elif plat == 'macos':
        screen_dev = _get_avfoundation_screen_device(monitor_index)
        cmd += [
            '-f', 'avfoundation',
            '-framerate', str(fps),
            '-capture_cursor', '1',
            '-i', screen_dev,
        ]
    else:
        raise RuntimeError(f"Screen mirror not supported on {plat}")

    # Scale filter chain. VAAPI needs an extra hwupload step to get frames onto
    # the GPU; the encoder consumes hardware surfaces. We do the scale on CPU
    # then upload (simpler than scale_vaapi which needs a fully-hardware pipeline
    # and tends to flake across vendors).
    needs_scale = (cap_w != out_w or cap_h != out_h) or plat == 'macos'
    vf_chain = []
    if needs_scale:
        vf_chain.append(f'scale={out_w}:{out_h}')
    if hw_encoder == 'h264_vaapi':
        vf_chain += ['format=nv12', 'hwupload']
    if vf_chain:
        cmd += ['-vf', ','.join(vf_chain)]

    # Encoder selection
    use_encoder = hw_encoder or 'libx264'
    if use_encoder == 'h264_nvenc':
        cmd += [
            '-c:v', 'h264_nvenc',
            '-preset', 'p1',
            '-tune', 'll',
            '-profile:v', 'baseline',
            '-level', level,
            '-rc', 'vbr',
            '-b:v', bitrate,
            '-maxrate', maxrate,
            '-bufsize', bufsize,
            '-g', str(fps * 2),
            '-bf', '0',
            '-zerolatency', '1',
        ]
    elif use_encoder == 'h264_videotoolbox':
        cmd += [
            '-c:v', 'h264_videotoolbox',
            '-profile:v', 'baseline',
            '-level', level,
            '-b:v', bitrate,
            '-maxrate', maxrate,
            '-bufsize', bufsize,
            '-g', str(fps * 2),
            '-bf', '0',
            '-realtime', '1',
        ]
    elif use_encoder == 'h264_vaapi':
        # VAAPI: universal Linux HW encoder. Bypasses CUDA context entirely so
        # survives the NVENC/CUDA corruption that has plagued our deployment.
        cmd += [
            '-c:v', 'h264_vaapi',
            '-profile:v', 'constrained_baseline',
            '-level', level,
            '-rc_mode', 'CBR',
            '-b:v', bitrate,
            '-maxrate', maxrate,
            '-bufsize', bufsize,
            '-g', str(fps * 2),
            '-bf', '0',
            '-low_power', '1',  # if supported, even lower latency
        ]
    elif use_encoder == 'h264_qsv':
        cmd += [
            '-c:v', 'h264_qsv',
            '-preset', 'veryfast',
            '-profile:v', 'baseline',
            '-level', level,
            '-b:v', bitrate,
            '-maxrate', maxrate,
            '-bufsize', bufsize,
            '-g', str(fps * 2),
            '-bf', '0',
            '-async_depth', '1',
        ]
    else:
        cmd += [
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-profile:v', 'baseline',
            '-level', level,
            '-b:v', bitrate,
            '-maxrate', maxrate,
            '-bufsize', bufsize,
            '-g', str(fps * 2),
            '-bf', '0',
        ]

    cmd += [
        '-pix_fmt', 'yuv420p',
        '-bsf:v', 'dump_extra',
        '-flush_packets', '1',
        '-f', 'h264',
        'pipe:1',
    ]
    return cmd, use_encoder


def _drain_stderr(pipe, label="ffmpeg"):
    """Drain stderr pipe in background to prevent deadlock. Log any output."""
    try:
        for line in pipe:
            text = line.decode('utf-8', errors='replace').rstrip()
            if text:
                log_error(f"[Screen] {label}: {text}")
    except (OSError, ValueError):
        pass
    finally:
        try:
            pipe.close()
        except Exception as e:
            log_debug(f"[Screen] stderr pipe close error: {e}")


def _start_ffmpeg(w, h, fps, screen_w, screen_h, monitor_x=0, monitor_y=0, monitor_index=0):
    """Start ffmpeg for screen capture. Probes HW encoder once, falls back to libx264.

    Key design: NO waiting after Popen — the reader thread must start immediately
    to prevent the pipe buffer from filling and blocking ffmpeg.
    monitor_x/monitor_y: pixel offset for multi-monitor capture.
    monitor_index: which monitor (0=primary) — used for macOS avfoundation device selection.
    """
    global _hw_encoder
    plat = PLATFORM

    display = _get_ffmpeg_display() if plat == 'linux' else None

    # One-time hardware encoder probe
    if _hw_encoder is None:
        _hw_encoder = _check_hw_encoder()
        if _hw_encoder:
            log_info(f"[Screen] Hardware encoder available: {_hw_encoder}")
        else:
            _hw_encoder = False
            log_info("[Screen] No hardware encoder, will use libx264")

    hw_enc = _hw_encoder if _hw_encoder else None

    # Build env for ffmpeg subprocess
    ffmpeg_env = dict(os.environ)
    if display:
        ffmpeg_env['DISPLAY'] = display

    # Build and start command
    cmd, encoder_name = _build_ffmpeg_cmd(w, h, fps, hw_encoder=hw_enc,
                                           screen_w=screen_w, screen_h=screen_h,
                                           display=display,
                                           monitor_x=monitor_x, monitor_y=monitor_y,
                                           monitor_index=monitor_index)
    log_info(f"[Screen] Starting ffmpeg ({encoder_name})")

    popen_kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        bufsize=0, env=ffmpeg_env)
    if plat == 'windows':
        popen_kwargs['creationflags'] = 0x08000000  # CREATE_NO_WINDOW

    proc = subprocess.Popen(cmd, **popen_kwargs)

    # Increase stdout pipe buffer to 1MB (Linux only, fcntl not available elsewhere)
    try:
        import fcntl
        F_SETPIPE_SZ = 1031
        fcntl.fcntl(proc.stdout.fileno(), F_SETPIPE_SZ, 1024 * 1024)
    except Exception as e:
        log_debug(f"[Screen] fcntl pipe resize skipped: {e}")  # Non-Linux or insufficient privileges

    # Track this ffmpeg process for shutdown cleanup (P2-29)
    with _ffmpeg_procs_lock:
        _active_ffmpeg_procs.add(proc)

    # Quick liveness check
    time.sleep(0.2)
    ret = proc.poll()
    if ret is not None:
        with _ffmpeg_procs_lock:
            _active_ffmpeg_procs.discard(proc)
        stderr_out = proc.stderr.read().decode('utf-8', errors='replace')[:500]
        log_error(f"[Screen] ffmpeg ({encoder_name}) exited immediately (code={ret}): {stderr_out}")

        if hw_enc:
            _hw_encoder = False
            log_info("[Screen] Falling back to libx264")
            cmd, encoder_name = _build_ffmpeg_cmd(w, h, fps, hw_encoder=None,
                                                   screen_w=screen_w, screen_h=screen_h,
                                                   display=display,
                                                   monitor_x=monitor_x, monitor_y=monitor_y,
                                                   monitor_index=monitor_index)
            proc = subprocess.Popen(cmd, **popen_kwargs)
            try:
                import fcntl
                fcntl.fcntl(proc.stdout.fileno(), F_SETPIPE_SZ, 1024 * 1024)
            except Exception as e:
                log_debug(f"[Screen] fcntl pipe resize skipped (fallback): {e}")
            with _ffmpeg_procs_lock:
                _active_ffmpeg_procs.add(proc)
            time.sleep(0.2)
            ret = proc.poll()
            if ret is not None:
                stderr2 = proc.stderr.read().decode('utf-8', errors='replace')[:300]
                raise RuntimeError(f"ffmpeg libx264 also failed (code={ret}): {stderr2}")
        else:
            raise RuntimeError(f"ffmpeg libx264 failed (code={ret}): {stderr_out}")

    # ffmpeg is alive — start background stderr drain
    threading.Thread(target=_drain_stderr, args=(proc.stderr, encoder_name),
                     daemon=True, name="ffmpeg-stderr").start()

    log_info(f"[Screen] Encoder started: {encoder_name}")
    return proc, encoder_name


async def ws_screen_mirror(ws: WebSocket):
    """WebSocket handler for screen mirroring. NVENC with automatic libx264 fallback."""
    cid = _resolve_cid(ws)
    client_ip = ws.client.host if ws.client else "unknown"
    if not await _accept_authenticated_ws(ws, cid, "Screen"):
        return
    _ws_log(cid, "info", f"[Screen] Client connected from {client_ip}")

    fps = 30
    monitor_x = 0
    monitor_y = 0
    selected_monitor_index = 0

    # Parse query params for resolution override and monitor selection
    query = ws.query_params if hasattr(ws, 'query_params') else {}
    req_res = query.get('res', '')

    # Multi-monitor: if monitor index specified, capture that monitor
    req_monitor = query.get('monitor', '')
    if req_monitor.isdigit():
        selected_monitor_index = int(req_monitor)
        monitor_info = get_monitor_by_index(int(req_monitor))
        if monitor_info:
            screen_w = monitor_info['width']
            screen_h = monitor_info['height']
            monitor_x = monitor_info['x']
            monitor_y = monitor_info['y']
            _ws_log(cid, "info", f"[Screen] Monitor {req_monitor}: {screen_w}x{screen_h} at ({monitor_x},{monitor_y})")
        else:
            _ws_log(cid, "warning", f"[Screen] Monitor {req_monitor} not found, using primary")
            screen_w, screen_h = get_screen_resolution_physical()
    else:
        screen_w, screen_h = get_screen_resolution_physical()

    # Fit screen into target box, preserving aspect ratio
    if req_res in _RES_PRESETS:
        box_w, box_h = _RES_PRESETS[req_res]
        scale = min(box_w / screen_w, box_h / screen_h)
        w = int(screen_w * scale) & ~1  # even dimensions (H.264 requirement)
        h = int(screen_h * scale) & ~1
    else:
        w, h = screen_w, screen_h

    req_fps = query.get('fps', '')
    if req_fps.isdigit():
        fps = max(1, min(60, int(req_fps)))

    _ws_log(cid, "info", f"[Screen] Capture {screen_w}x{screen_h} -> {w}x{h}@{fps}fps")

    # Send config FIRST so client can prepare decoder before frames arrive
    config = struct.pack('<HHB', w, h, fps)  # 5 bytes: width(2) + height(2) + fps(1)
    await ws.send_bytes(bytes([0x20]) + config)  # 0x20 = SCREEN_CONFIG

    # Adaptive quality: track current resolution for quality change detection
    current_quality = req_res if req_res in _RES_PRESETS else '1080p'
    quality_change_event = asyncio.Event()
    pending_quality = {}  # mutable dict shared between sender and receiver tasks
    # A11-P1-3: Cooldown to prevent oscillation between quality levels
    _last_quality_change_time = 0.0  # monotonic timestamp of last applied change

    loop = asyncio.get_event_loop()
    proc = None
    reader_thread = None
    stop_event = threading.Event()
    frame_queue = queue.Queue(maxsize=90)  # ~3 seconds buffer at 30fps
    receiver_task = None

    async def _receive_client_messages():
        """Listen for client->server messages (quality control) on the screen WS."""
        # A11-P3-2: Removed unnecessary nonlocal — current_quality and
        # _last_quality_change_time are only read here, not assigned
        try:
            while True:
                data = await ws.receive_bytes()
                if len(data) >= 2 and data[0] == MSG_QUALITY_CTRL:
                    level = data[1]
                    new_res = _QUALITY_LEVELS.get(level)
                    if new_res and new_res != current_quality:
                        # A11-P1-3: Prevent quality oscillation — 30s cooldown between changes
                        now = time.monotonic()
                        if now - _last_quality_change_time < 30.0:
                            _ws_log(cid, "debug", f"[Screen] Quality change {current_quality}->{new_res} suppressed (cooldown)")
                            continue
                        _ws_log(cid, "info", f"[Screen] Quality change requested: {current_quality} -> {new_res}")
                        pending_quality['res'] = new_res
                        quality_change_event.set()
        except (WebSocketDisconnect, RuntimeError):
            pass  # Client disconnected — sender loop will also break
        except Exception as e:
            _ws_log(cid, "debug", f"[Screen] Receiver error: {e}")

    def _stop_ffmpeg_proc(p, se, rt):
        """Stop an ffmpeg process and its reader thread."""
        se.set()
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        if p:
            with _ffmpeg_procs_lock:
                _active_ffmpeg_procs.discard(p)
        if rt and rt.is_alive():
            rt.join(timeout=2)

    try:
        # A8-P2-6: Use io_executor (not default) to avoid contending with
        # cursor_pos_changed.wait() and frame_queue.get() on the default pool
        proc, encoder = await loop.run_in_executor(
            io_executor, lambda: _start_ffmpeg(w, h, fps, screen_w, screen_h,
                                         monitor_x=monitor_x, monitor_y=monitor_y,
                                         monitor_index=selected_monitor_index))

        # Start reader thread IMMEDIATELY — ffmpeg is already writing to stdout.
        # Any delay here risks filling the pipe buffer and stalling ffmpeg.
        reader_thread = threading.Thread(
            target=_read_h264_frames,
            args=(proc.stdout, frame_queue, stop_event),
            daemon=True,
            name="screen-reader",
        )
        reader_thread.start()

        # Start receiver task for client messages (quality control)
        receiver_task = asyncio.create_task(_receive_client_messages())
        def _task_done_cb(t):
            if not t.cancelled():
                exc = t.exception()
                if exc:
                    _ws_log(cid, "debug", f"[Screen] Receiver task error: {exc}")
        receiver_task.add_done_callback(_task_done_cb)

        # Sender loop — read frames from queue, send over WebSocket
        # N-P1-4: Track send backpressure; drop P-frames if send is slow
        # N-P2-6: Track keyframe state — drop P-frames after lost keyframe
        _awaiting_keyframe = False  # Set True if a keyframe was dropped
        while True:
            # Check for pending quality change from client
            if quality_change_event.is_set():
                quality_change_event.clear()
                new_res = pending_quality.pop('res', None)
                if new_res and new_res in _RES_PRESETS:
                    box_w, box_h = _RES_PRESETS[new_res]
                    nscale = min(box_w / screen_w, box_h / screen_h)
                    new_w = int(screen_w * nscale) & ~1
                    new_h = int(screen_h * nscale) & ~1
                    _ws_log(cid, "info", f"[Screen] Quality change: {new_w}x{new_h}@{fps}fps ({new_res})")
                    new_stop = threading.Event()
                    new_queue = queue.Queue(maxsize=90)
                    try:
                        # A8-P2-6: Use io_executor for blocking ffmpeg ops
                        new_proc, new_enc = await loop.run_in_executor(
                            io_executor, lambda nw=new_w, nh=new_h: _start_ffmpeg(
                                nw, nh, fps, screen_w, screen_h,
                                monitor_x=monitor_x, monitor_y=monitor_y))
                    except Exception as qe:
                        _ws_log(cid, "error", f"[Screen] Quality change ffmpeg start failed: {qe}")
                        continue
                    old_proc, old_stop, old_rt = proc, stop_event, reader_thread
                    await loop.run_in_executor(io_executor, _stop_ffmpeg_proc, old_proc, old_stop, old_rt)
                    w, h = new_w, new_h
                    current_quality = new_res
                    _last_quality_change_time = time.monotonic()  # A11-P1-3: Reset cooldown
                    cfg = struct.pack('<HHB', w, h, fps)
                    await ws.send_bytes(bytes([0x20]) + cfg)
                    stop_event = new_stop
                    frame_queue = new_queue
                    proc, encoder = new_proc, new_enc
                    reader_thread = threading.Thread(
                        target=_read_h264_frames,
                        args=(proc.stdout, frame_queue, stop_event),
                        daemon=True, name="screen-reader",
                    )
                    reader_thread.start()
                    _awaiting_keyframe = False
                    continue
            try:
                frame = await loop.run_in_executor(None, lambda: frame_queue.get(timeout=1.0))
            except queue.Empty:
                # Check if ffmpeg is still alive — auto-restart up to 3 times
                if proc.poll() is not None:
                    _ws_log(cid, "error", f"[Screen] ffmpeg exited (code={proc.returncode})")
                    _ffmpeg_restart_count = getattr(proc, '_restart_count', 0)
                    if _ffmpeg_restart_count >= 3:
                        _ws_log(cid, "error", "[Screen] ffmpeg crashed 3 times, giving up")
                        break
                    backoff = (1 << _ffmpeg_restart_count)  # 1s, 2s, 4s
                    _ws_log(cid, "info", f"[Screen] Restarting ffmpeg (attempt {_ffmpeg_restart_count + 1}/3, backoff {backoff}s)")
                    await asyncio.sleep(backoff)
                    try:
                        new_stop = threading.Event()
                        new_queue = queue.Queue(maxsize=90)
                        new_proc, encoder = await loop.run_in_executor(
                            io_executor, lambda: _start_ffmpeg(w, h, fps, screen_w, screen_h,
                                                          monitor_x=monitor_x, monitor_y=monitor_y))
                        new_proc._restart_count = _ffmpeg_restart_count + 1
                        _stop_ffmpeg_proc(proc, stop_event, reader_thread)
                        proc, stop_event, frame_queue = new_proc, new_stop, new_queue
                        reader_thread = threading.Thread(
                            target=_read_h264_frames,
                            args=(proc.stdout, frame_queue, stop_event),
                            daemon=True, name="screen-reader",
                        )
                        reader_thread.start()
                        _awaiting_keyframe = False
                        _ws_log(cid, "info", f"[Screen] ffmpeg restarted successfully")
                    except Exception as restart_err:
                        _ws_log(cid, "error", f"[Screen] ffmpeg restart failed: {restart_err}")
                        break
                continue

            if frame is None:
                break  # Reader thread signaled end

            is_key, timestamp_ms, h264_data = frame

            # N-P2-6: If we lost a keyframe, drop all P-frames until next keyframe
            if _awaiting_keyframe:
                if is_key:
                    _awaiting_keyframe = False
                    _ws_log(cid, "debug", "[Screen] Keyframe received — resuming send")
                else:
                    continue  # Drop P-frame (would decode to garbage)

            # N-P1-4: If queue is backing up (>50% full), drop P-frames to reduce WS pressure.
            # Never drop keyframes — they are essential for decoder sync.
            if not is_key and frame_queue.qsize() > frame_queue.maxsize // 2:
                continue  # Drop P-frame under backpressure

            # Wire format: [flags:1][timestamp:4LE][h264_data:N]
            flags = _SCREEN_FLAG_KEY if is_key else 0
            # N-P2-2: Use bytearray + extend to avoid bytes concatenation allocations
            wire = bytearray(5 + len(h264_data))
            struct.pack_into('<BI', wire, 0, flags, timestamp_ms)
            wire[5:] = h264_data
            try:
                await ws.send_bytes(bytes(wire))
            except Exception as e:
                if is_key:
                    _awaiting_keyframe = True
                    _ws_log(cid, "warning", f"[Screen] Keyframe send failed — will skip until next keyframe: {e}")
                break

    except WebSocketDisconnect:
        _ws_log(cid, "info", "[Screen] Client disconnected")
    except RuntimeError as e:
        _ws_log(cid, "error", f"[Screen] Failed to start: {e}")
    except Exception as e:
        _ws_log(cid, "error", f"[Screen] Error: {e}")
    finally:
        if receiver_task and not receiver_task.done():
            receiver_task.cancel()
        _unregister_ws(ws)
        await loop.run_in_executor(None, _stop_ffmpeg_proc, proc, stop_event, reader_thread)
        _ws_log(cid, "info", "[Screen] Session ended")


# =============================================================================
#                         AUDIO OUTPUT (PC → Phone)
# =============================================================================

def _get_pulse_monitor_source():
    """Detect the PulseAudio/PipeWire monitor source for system audio capture.
    Returns the monitor source name (e.g. 'alsa_output.pci-xxx.analog-stereo.monitor')
    or None if detection fails."""
    try:
        result = subprocess.run(
            ['pactl', 'get-default-sink'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip() + '.monitor'
    except Exception:
        pass
    try:
        result = subprocess.run(
            ['pactl', 'list', 'short', 'sources'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                parts = line.split('\t')
                if len(parts) >= 2 and '.monitor' in parts[1]:
                    return parts[1]
    except Exception:
        pass
    return None

def _get_audio_capture_cmd():
    """Build ffmpeg command for system audio capture → Opus encoding.
    Returns (cmd_list, description) or (None, error_msg)."""
    if PLATFORM == 'linux':
        # PulseAudio/PipeWire: capture the MONITOR source of the default audio sink.
        # This captures what's playing through speakers, NOT the microphone input.
        monitor = _get_pulse_monitor_source()
        if monitor is None:
            return (None, "No PulseAudio monitor source found (is PulseAudio/PipeWire running?)")
        return ([
            'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
            '-f', 'pulse', '-i', monitor,
            '-ac', '1', '-ar', '48000',
            '-c:a', 'libopus', '-b:a', '64k',
            '-application', 'audio',
            '-f', 'opus', '-page_duration', '20000',
            'pipe:1',
        ], f"PulseAudio monitor: {monitor}")
    elif PLATFORM == 'windows':
        # Windows: WASAPI loopback via dshow virtual audio capturer
        # Requires "Stereo Mix" or similar enabled, or a virtual audio device
        return ([
            'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
            '-f', 'dshow', '-i', 'audio=Stereo Mix',
            '-ac', '1', '-ar', '48000',
            '-c:a', 'libopus', '-b:a', '64k',
            '-application', 'audio',
            '-f', 'opus', '-page_duration', '20000',
            'pipe:1',
        ], "Windows Stereo Mix (dshow)")
    elif PLATFORM == 'macos':
        # macOS: avfoundation audio device 0 (system audio requires BlackHole or similar)
        return ([
            'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
            '-f', 'avfoundation', '-i', ':0',
            '-ac', '1', '-ar', '48000',
            '-c:a', 'libopus', '-b:a', '64k',
            '-application', 'audio',
            '-f', 'opus', '-page_duration', '20000',
            'pipe:1',
        ], "macOS avfoundation audio device :0")
    return (None, f"Unsupported platform: {PLATFORM}")


async def ws_audio_output(ws: WebSocket):
    """WebSocket handler for streaming system audio (PC speaker → phone).
    Captures system audio via ffmpeg, encodes to Opus, sends as binary frames."""
    cid = _resolve_cid(ws)  # A8-P3-2: correlation ID (phone-supplied X-Conn-Id if present)
    if not await _accept_authenticated_ws(ws, cid, "AudioOut"):
        return
    _ws_log(cid, "info", "[AudioOut] Client connected")

    cmd, desc = _get_audio_capture_cmd()
    if cmd is None:
        _ws_log(cid, "error", f"[AudioOut] Cannot capture: {desc}")
        try:
            await ws.send_text(json.dumps({"error": desc}))
            await ws.close(code=4500, reason=desc)
        except Exception:
            pass
        _unregister_ws(ws)
        return

    _ws_log(cid, "info", f"[AudioOut] Capturing via {desc}")

    proc = None
    recv_task = None
    stop_event = threading.Event()
    audio_queue = queue.Queue(maxsize=200)  # ~4 seconds at 50 packets/sec

    def _read_opus_packets(stdout, q, stop_ev):
        """Read Opus packets from ffmpeg stdout pipe and queue them."""
        try:
            while not stop_ev.is_set():
                # Read OGG/Opus stream — each page contains one or more Opus packets.
                # For simplicity, read in fixed-size chunks. The Android decoder
                # handles partial/multiple packets per chunk gracefully.
                data = stdout.read(4096)
                if not data:
                    break
                try:
                    q.put_nowait(data)
                except queue.Full:
                    try:
                        q.get_nowait()  # Drop oldest
                    except queue.Empty:
                        pass
                    try:
                        q.put_nowait(data)
                    except queue.Full:
                        pass
        except Exception as e:
            if not stop_ev.is_set():
                log_warning(f"[AudioOut] Reader error: {e}")
        finally:
            q.put(None)  # Sentinel

    try:
        env = os.environ.copy()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        with _ffmpeg_procs_lock:
            _active_ffmpeg_procs.add(proc)

        reader_thread = threading.Thread(
            target=_read_opus_packets,
            args=(proc.stdout, audio_queue, stop_event),
            daemon=True,
            name="audio-out-reader",
        )
        reader_thread.start()

        # Also start a task to receive messages from client (e.g. disconnect signal)
        async def _recv_loop():
            try:
                while True:
                    await ws.receive()
            except (WebSocketDisconnect, Exception):
                pass

        recv_task = asyncio.create_task(_recv_loop())
        recv_task.add_done_callback(_task_done)

        loop = asyncio.get_event_loop()
        while True:
            try:
                data = await loop.run_in_executor(None, lambda: audio_queue.get(timeout=1.0))
            except queue.Empty:
                if proc.poll() is not None:
                    _ws_log(cid, "error", f"[AudioOut] ffmpeg exited (code={proc.returncode})")
                    break
                continue

            if data is None:
                break  # Reader done

            try:
                await ws.send_bytes(data)
            except Exception:
                break

    except WebSocketDisconnect:
        _ws_log(cid, "info", "[AudioOut] Client disconnected")
    except Exception as e:
        _ws_log(cid, "error", f"[AudioOut] Error: {e}")
    finally:
        if recv_task and not recv_task.done():
            recv_task.cancel()
            try:
                await recv_task
            except (asyncio.CancelledError, Exception):
                pass
        _unregister_ws(ws)
        stop_event.set()
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if proc:
            with _ffmpeg_procs_lock:
                _active_ffmpeg_procs.discard(proc)
        _ws_log(cid, "info", "[AudioOut] Session ended")


@app.websocket("/ws-audio-stream")
async def websocket_audio_endpoint(websocket: WebSocket):
    await ws_audio_stream(websocket)

@app.websocket("/ws-tp")
async def websocket_trackpad_endpoint(websocket: WebSocket):
    global _active_trackpad_ws
    _active_trackpad_ws += 1
    try:
        await ws_trackpad(websocket)
    finally:
        _active_trackpad_ws = max(0, _active_trackpad_ws - 1)

@app.websocket("/ws-screen")
async def websocket_screen_endpoint(websocket: WebSocket):
    global _active_screen_ws
    _active_screen_ws += 1
    try:
        await ws_screen_mirror(websocket)
    finally:
        _active_screen_ws = max(0, _active_screen_ws - 1)

@app.websocket("/ws-audio-out")
async def websocket_audio_output_endpoint(websocket: WebSocket):
    global _active_audio_out_ws
    _active_audio_out_ws += 1
    try:
        await ws_audio_output(websocket)
    finally:
        _active_audio_out_ws = max(0, _active_audio_out_ws - 1)

@app.get("/")
async def index():
    return Response(
        content=_html_bytes,
        media_type="text/html",
        headers={"Cache-Control": "private, max-age=300"},  # 5 min cache — only changes on restart
    )

# Codex F-Apr21-05: serve PC dashboard + its static assets. Dashboard HTML is
# public (no auth required for the page itself); the JS bootstraps the user's
# token from URL ?token= or localStorage and adds it to all subsequent
# /api/history/* + /api/vocab + /api/accent fetches + /ws-dashboard.
_static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "static")

@app.get("/dashboard")
async def dashboard_html():
    """Serve the dashboard SPA. Token bootstrapped client-side."""
    try:
        with open(os.path.join(_static_dir, "dashboard.html"), "rb") as f:
            html = f.read()
        return Response(content=html, media_type="text/html",
                        headers={"Cache-Control": "private, max-age=60"})
    except FileNotFoundError:
        return Response(status_code=404, content=b"dashboard.html missing")

@app.get("/static/{filename:path}")
async def static_files(filename: str):
    """Serve dashboard.css, dashboard.js (and future static assets)."""
    # Hard whitelist — only files that exist in static/. Prevents path traversal.
    safe = os.path.basename(filename)
    if safe != filename or ".." in filename:
        return Response(status_code=404)
    full = os.path.join(_static_dir, safe)
    if not os.path.isfile(full):
        return Response(status_code=404)
    ct = "text/css" if safe.endswith(".css") else \
         "application/javascript" if safe.endswith(".js") else \
         "text/html" if safe.endswith(".html") else \
         "application/octet-stream"
    with open(full, "rb") as f:
        return Response(content=f.read(), media_type=ct,
                        headers={"Cache-Control": "private, max-age=300"})

@app.get("/api/screen")
async def screen_info(request: Request, token: str = Query(None), monitor: int = Query(None)):
    if not _check_any_token(_extract_token(request, token)):
        return Response(status_code=401)
    # If monitor index specified, return that monitor's resolution + position
    if monitor is not None:
        m = get_monitor_by_index(monitor)
        if m:
            return {"w": m['width'], "h": m['height'], "x": m['x'], "y": m['y'], "monitor": monitor}
        return JSONResponse({"error": f"Monitor {monitor} not found"}, status_code=404)
    w, h = get_screen_resolution()
    return {"w": w, "h": h}

@app.get("/api/monitors")
async def api_monitors(request: Request, token: str = Query(None)):
    """Return list of connected monitors with index, name, resolution, and position."""
    if not _check_any_token(_extract_token(request, token)):
        return Response(status_code=401)
    monitors = get_monitors()
    return {"monitors": monitors}

def _get_preferred_ip():
    """Default-route IP — typically the Ethernet interface."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        return None

@app.get("/api/discover")
async def api_discover():
    """LAN discovery endpoint — no auth required, returns service identity.
    A9-P2-1: Exposes hostname (platform.node()), server_id, and udp_port to unauthenticated
    LAN clients. This is intentional — the Android app needs these fields for service discovery.
    The LAN middleware restricts this to RFC 1918 IPs in service mode."""
    resp = {"service": "sanketra", "name": platform.node(), "port": SERVER_PORT, "udp_port": _udp_port, "os": platform.system(), "server_id": _get_server_id(), "pair_code_required": not _pair_locked}
    pip = _get_preferred_ip()
    if pip:
        resp["preferred_ip"] = pip
    return resp

@app.post("/api/pair")
async def api_pair(request: Request):
    """SSH-style TOFU pairing — first connect is trusted, returns auth token + cert fingerprint.
    After first successful pair, pairing is locked. Use /api/unpair to unlock.

    Atomicity: the rate-limit check, pair-code verify, _pair_locked check, and
    _set_pair_lock() transition all happen under _pair_mutex. Previously they were
    4 separate steps — two concurrent correct-code POSTs could both reach the
    _pair_locked==False check and both "win" the pair.
    """
    client_ip = request.client.host

    if _pair_mutex is None:
        return JSONResponse({"error": "Server not ready"}, status_code=503)

    # Codex F4: parse the request body BEFORE acquiring _pair_mutex. Reading the
    # body is unbounded I/O — a slow or stalled client used to block every other
    # pairing attempt for the duration of its TCP read. Parsing first keeps the
    # mutex critical section purely to rate-limit + code-compare + lock-set.
    try:
        body = await request.json()
    except Exception:
        body = {}
    submitted_code = str(body.get("pair_code", "")).strip()

    async with _pair_mutex:
        now = time.time()
        attempts = _pair_attempts.get(client_ip, [])
        attempts = [t for t in attempts if now - t < _PAIR_RATE_WINDOW]
        if len(attempts) >= _PAIR_RATE_LIMIT:
            log_info(f"PAIR rate-limited: {client_ip} ({len(attempts)} attempts in {_PAIR_RATE_WINDOW}s)")
            return JSONResponse(
                {"error": "Too many attempts. Wait a minute.", "code": "rate_limited"},
                status_code=429,
            )
        attempts.append(now)
        _pair_attempts[client_ip] = attempts
        if not submitted_code or not secrets.compare_digest(submitted_code, _pair_code):
            log_info(f"PAIR rejected (bad pair_code): {client_ip}")
            return JSONResponse(
                {"error": "Invalid pair code", "code": "invalid_pair_code"},
                status_code=403,
            )

        if _pair_locked:
            log_info(f"PAIR rejected (locked): {client_ip}")
            return JSONResponse(
                {"error": "Pairing locked", "code": "pair_locked"},
                status_code=403,
            )

        from cryptography import x509
        from cryptography.hazmat.primitives.serialization import Encoding
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        cert_path = os.path.join(root_dir, 'cert.pem')
        try:
            with open(cert_path, "rb") as f:
                cert_obj = x509.load_pem_x509_certificate(f.read())
            cert_fp = hashlib.sha256(cert_obj.public_bytes(Encoding.DER)).hexdigest()
        except Exception as e:
            log_error(f"Failed to load cert for fingerprint: {e}")
            return Response(status_code=500)
        if not cert_fp:
            return Response(status_code=500)
        _paired_ips.add(client_ip)

        _set_pair_lock(True, device_ip=client_ip)
        log_info(f"PAIRED successfully with {client_ip} (TOFU) — pairing now LOCKED")
        return {"token": AUTH_TOKEN, "cert_fp": cert_fp}

@app.post("/api/unpair")
async def api_unpair(request: Request, token: str = Query(None)):
    """Unlock pairing so a new device can pair. Requires valid auth token."""
    if not _check_any_token(_extract_token(request, token)):
        return Response(status_code=401)
    if not _pair_locked:
        return JSONResponse({"status": "already_unlocked"})
    _set_pair_lock(False)
    log_info(f"UNPAIRED by {request.client.host} — pairing UNLOCKED")
    return JSONResponse({"status": "unpaired"})

@app.get("/api/token-check")
async def api_token_check(request: Request, token: str = Query(None)):
    """Lightweight token validation — web client calls on page load to detect stale tokens."""
    if _check_any_token(_extract_token(request, token)):
        return JSONResponse({"valid": True})
    return Response(status_code=401)

@app.post("/api/blink")
async def blink_screen(request: Request, token: str = Query(None)):
    """Flash the laptop screen 3 times for visual identification"""
    if not _check_any_token(_extract_token(request, token)):
        return Response(status_code=401)
    global _blink_active
    if _blink_active:
        return {"status": "already_blinking"}
    _blink_active = True
    loop = asyncio.get_event_loop()
    # A9-P3-2: Track the executor future so _blink_active always resets (already in finally,
    # but this ensures exceptions don't silently disappear). Fire-and-forget is acceptable here
    # since _do_screen_blink has its own try/finally.
    fut = loop.run_in_executor(io_executor, _do_screen_blink)
    def _blink_done(f):
        exc = f.exception()
        if exc:
            log_warning(f"[Blink] Error: {exc}")
    fut.add_done_callback(_blink_done)
    return {"status": "blinking"}

# =============================================================================
#                     SERVICE MODE API (Phase 3)
# =============================================================================

def _verify_session(token: str) -> bool:
    """Verify a client session token. Side-effects: deletes expired sessions
    and updates last_used. Pure logic lives in auth_core.verify_session_token."""
    if not token:
        return False
    with _sessions_lock:
        # auth_core handles the master-token equality + expiry check.
        if not auth_core.verify_session_token(token, AUTH_TOKEN, _client_sessions):
            # If a session exists but is expired, prune it so it doesn't accrete.
            session = _client_sessions.get(token)
            if session and auth_core.is_session_expired(session):
                del _client_sessions[token]
                # F-Apr22-03: debounced write — coalesce with other mutators.
                _mark_sessions_dirty()
            return False
        # Touch the session (only needed for non-master tokens)
        session = _client_sessions.get(token)
        if session is not None:
            session["last_used"] = time.time()
        return True

def _check_auth_rate_limit(ip: str) -> bool:
    """Returns True if IP is rate-limited (too many failed attempts).
    Lockout window is AUTH_LOCKOUT (600 s) — once tripped, IP stays locked
    for 10 min from last failure regardless of the burst window. Pure logic
    in auth_core; this wrapper handles the per-IP attempt-list pruning side-
    effect on the shared _auth_attempts dict."""
    now = time.time()
    attempts = _auth_attempts.get(ip, [])
    pruned = auth_core.prune_auth_attempts(attempts, now)
    _auth_attempts[ip] = pruned
    return auth_core.is_auth_rate_limited(pruned, now)

def _record_auth_failure(ip: str):
    """Record a failed auth attempt for rate limiting."""
    now = time.time()
    attempts = _auth_attempts.get(ip, [])
    attempts.append(now)
    _auth_attempts[ip] = attempts

@app.post("/api/auth")
async def api_auth(request: Request):
    """Authenticate with app password (Argon2id). Returns session token + GPU info."""
    if not _SERVICE_MODE:
        return JSONResponse({"error": "Auth API only available in service mode"}, status_code=400)

    client_ip = request.client.host
    if _check_auth_rate_limit(client_ip):
        return JSONResponse({"error": "Too many attempts. Try again later."}, status_code=429)

    try:
        body = await request.json()
    except Exception as e:
        log_warning(f"[Auth] Invalid request body from {client_ip}: {e}")
        return JSONResponse({"error": "Invalid request body"}, status_code=400)

    password = body.get("password", "")
    if not password:
        return JSONResponse({"error": "Password required"}, status_code=400)

    stored_hash = _get_app_password_hash()
    if not stored_hash:
        return JSONResponse({"error": "No app password set. Run setup first."}, status_code=503)

    # Verify password (Argon2id or PBKDF2 fallback)
    try:
        if stored_hash.startswith("pbkdf2:"):
            # PBKDF2 fallback format: pbkdf2:<salt>:<hash>
            _, salt, expected = stored_hash.split(":", 2)
            import hashlib as hl
            computed = hl.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000).hex()
            if not secrets.compare_digest(computed, expected):
                raise ValueError("mismatch")
        else:
            from argon2 import PasswordHasher
            # verify() doesn't need params — they're embedded in the hash.
            # PasswordHasher() with defaults is fine here.
            ph = PasswordHasher()
            ph.verify(stored_hash, password)
    except Exception as e:
        log_debug(f"[Auth] Password verification failed for {client_ip}: {e}")
        _record_auth_failure(client_ip)
        return JSONResponse({"error": "Invalid password"}, status_code=401)

    _auth_attempts.pop(client_ip, None)

    # Generate session token
    session_token = secrets.token_urlsafe(32)

    with _sessions_lock:
        # Purge expired sessions first, then evict oldest if still over capacity
        now = time.time()
        expired = [t for t, s in _client_sessions.items() if now - s.get("created", 0) > _SESSION_TTL]
        for t in expired:
            del _client_sessions[t]
        if len(_client_sessions) >= _SESSION_MAX_COUNT:
            sorted_tokens = sorted(
                _client_sessions.keys(),
                key=lambda t: _client_sessions[t].get("created", 0)
            )
            for old_token in sorted_tokens[:len(_client_sessions) - _SESSION_MAX_COUNT + 1]:
                del _client_sessions[old_token]

        _client_sessions[session_token] = {
            "ip": client_ip,
            "created": time.time(),
            "last_used": time.time(),
        }
        # F-Apr22-03: debounced write — concurrent /api/auth calls collapse
        # into a single fsync within the 500 ms coalesce window.
        _mark_sessions_dirty()

    # Return session + system info
    sys_info = get_available_models()
    return {
        "token": session_token,
        "system": sys_info["system"],
        "state": _server_state.name,
        "loaded_model": _loaded_model_name,
    }

@app.get("/api/models")
async def api_models(request: Request, token: str = Query(None)):
    """List available models with VRAM fit info and download status."""
    if not _verify_session(_extract_token(request, token)):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    result = get_available_models()
    result["server_state"] = _server_state.name
    result["loaded_model"] = _loaded_model_name
    return result

@app.post("/api/model/load")
async def api_model_load(request: Request, token: str = Query(None)):
    """Trigger model download/load. Transitions server IDLE → LOADING → ACTIVE.
    Serialized via _model_load_lock — two concurrent requests used to both pass
    the LOADING check and start two parallel Whisper loads (OOM risk)."""
    if not _verify_session(_extract_token(request, token)):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    # Validate X-Client-Kind up front — the license gate below keys off it.
    # Unknown header values are rejected rather than silently defaulted; a
    # misbehaving / tampered client must not accidentally fall into the
    # `web` (free-only) tier and wedge itself at `small`.
    client_kind = _client_kind_from_headers(request.headers)
    if client_kind is None:
        return JSONResponse(
            {"error": "invalid X-Client-Kind header",
             "allowed": sorted(VALID_CLIENT_KINDS)},
            status_code=400,
        )

    try:
        body = await request.json()
    except Exception as e:
        log_warning(f"[API] model/load invalid body: {e}")
        return JSONResponse({"error": "Invalid request body"}, status_code=400)

    model_name = body.get("model", "")
    precision = body.get("precision")
    force_device = body.get("device")
    valid_models = ["tiny", "base", "small", "medium", "large-v3-turbo", "distil-large-v3", "large-v3"]
    if model_name not in valid_models:
        return JSONResponse({"error": f"Invalid model. Choose from: {valid_models}"}, status_code=400)

    # License gate — Pro-only models need a license covering this client's
    # track (or a Bundle). Free-tier caps at `small` per MONETIZATION.md.
    # `requires_pro=True` lands in /api/models so client UIs can lock these
    # visually; this check is the authoritative one.
    if model_name not in FREE_TIER_MODELS and not _has_license_for_client(client_kind):
        return JSONResponse(
            {
                "error": "model_requires_pro",
                "model": model_name,
                "client_kind": client_kind,
                "free_tier_max": "small",
                "free_tier_models": sorted(FREE_TIER_MODELS),
            },
            status_code=403,
        )
    valid_precisions = [None, "auto", "float16", "float32", "int8", "int8_float16", "int8_float32"]
    if precision is not None and precision not in valid_precisions:
        return JSONResponse({"error": f"Invalid precision. Choose from: {[p for p in valid_precisions if p]}"}, status_code=400)
    if precision == "auto":
        precision = None

    if force_device != "cpu":
        try:
            from stt_common import get_vram_free, MODEL_VRAM
            vram_free = get_vram_free()
            if vram_free > 0 and model_name in MODEL_VRAM:
                vram_needed = MODEL_VRAM[model_name].get("float16", 0)
                if vram_needed > 0 and vram_free < vram_needed * 0.9:
                    return JSONResponse({
                        "error": f"Insufficient VRAM: {model_name} needs ~{vram_needed:.1f}GB, "
                                 f"only {vram_free:.1f}GB free. Try a smaller model or use CPU.",
                        "vram_free": round(vram_free, 2),
                        "vram_needed": round(vram_needed, 2),
                    }, status_code=400)
        except (ImportError, AttributeError):
            pass

    # Codex F7: the prior `if _model_load_lock.locked(): return 409` + `async with`
    # was non-atomic — between the locked() check and the await, another request
    # could acquire the lock; the loser then *waited* for a multi-second model
    # load instead of getting the intended 409. Use try-acquire pattern: if we
    # can't grab the lock immediately, return 409. Lock released after executor.
    if _model_load_lock is None:
        return JSONResponse({"error": "Server not ready"}, status_code=503)
    try:
        await asyncio.wait_for(_model_load_lock.acquire(), timeout=0.001)
    except asyncio.TimeoutError:
        return JSONResponse({"error": "Already loading a model"}, status_code=409)
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _model_load_executor, _load_model_async, model_name, precision, force_device
        )
    finally:
        _model_load_lock.release()
    if isinstance(result, dict) and "error" in result:
        sc = 409 if "Already loading" in result.get("error", "") else 500
        return JSONResponse(result, status_code=sc)
    return result

@app.get("/api/model/status")
async def api_model_status(request: Request, token: str = Query(None)):
    """Get current model loading status and progress."""
    if not _verify_session(_extract_token(request, token)):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    return {
        "state": _server_state.name,
        "model": _loaded_model_name,
        "progress": round(_model_load_progress, 2),
        "device": _loaded_device,
    }

@app.post("/api/model/cancel")
async def api_model_cancel(request: Request, token: str = Query(None)):
    """Cancel in-progress model download/load."""
    global _model_load_cancel
    if not _verify_session(_extract_token(request, token)):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    if _server_state != ServerState.LOADING:
        return JSONResponse({"error": "No model load in progress"}, status_code=400)

    _model_load_cancel = True
    return {"status": "cancelling"}

_admin_update_lock = asyncio.Lock()

@app.post("/api/admin/update")
async def api_admin_update(request: Request, token: str = Query(None)):
    """Pull latest code from git and self-restart if changed."""
    # P0-2: Restrict to localhost only — this endpoint runs git reset + pip install + restart
    client_ip = request.client.host
    if client_ip not in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
        log_warning(f"[API] admin/update blocked from non-local IP: {client_ip}")
        return JSONResponse({"error": "Forbidden — localhost only"}, status_code=403)

    if not _verify_session(_extract_token(request, token)):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    if _admin_update_lock.locked():
        return JSONResponse({"error": "Update already in progress"}, status_code=409)

    async with _admin_update_lock:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        install_dir = os.path.dirname(script_dir)  # parent of src/

        try:
            # Get current HEAD
            old_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=install_dir, capture_output=True, text=True, timeout=10
            ).stdout.strip()

            # Detect default branch (master or main)
            branch_result = subprocess.run(
                ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
                cwd=install_dir, capture_output=True, text=True, timeout=10
            )
            branch = "master"  # fallback
            if branch_result.returncode == 0 and branch_result.stdout.strip():
                branch = branch_result.stdout.strip().split("/")[-1]

            # Fetch
            fetch = subprocess.run(
                ["git", "fetch", "origin"],
                cwd=install_dir, capture_output=True, text=True, timeout=30
            )
            if fetch.returncode != 0:
                log_error(f"[API] git fetch failed: {fetch.stderr}")
                return JSONResponse({"error": "Git fetch failed"}, status_code=500)

            # Reset to remote branch
            reset = subprocess.run(
                ["git", "reset", "--hard", f"origin/{branch}"],
                cwd=install_dir, capture_output=True, text=True, timeout=10
            )
            if reset.returncode != 0:
                log_error(f"[API] git reset failed: {reset.stderr}")
                return JSONResponse({"error": "Git reset failed"}, status_code=500)

            # Get new HEAD
            new_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=install_dir, capture_output=True, text=True, timeout=10
            ).stdout.strip()

            if old_head and new_head and old_head != new_head:
                log_info(f"[API] admin/update: code updated {old_head[:8]}→{new_head[:8]}, installing deps + restarting...")

                # Install new dependencies if requirements changed
                venv_pip = os.path.join(install_dir, "venv", "bin", "pip")
                if not os.path.exists(venv_pip):
                    venv_pip = os.path.join(install_dir, "venv", "Scripts", "pip.exe")
                req_file = os.path.join(install_dir, "requirements.txt")
                if os.path.exists(venv_pip) and os.path.exists(req_file):
                    pip_result = subprocess.run(
                        [venv_pip, "install", "-r", req_file, "-q"],
                        cwd=install_dir, capture_output=True, text=True, timeout=120
                    )
                    # A9-P2-7: Log warning on pip failure but proceed with restart
                    # (code was already updated; deps may already be satisfied)
                    if pip_result.returncode != 0:
                        log_warning(f"[API] pip install returned {pip_result.returncode} — proceeding with restart")

                # Schedule self-restart: SIGTERM triggers clean shutdown,
                # service manager (systemd/launchd/schtasks) restarts us
                def _delayed_restart():
                    time.sleep(1)
                    os.kill(os.getpid(), signal.SIGTERM)
                threading.Thread(target=_delayed_restart, daemon=True).start()
                return {"status": "updating", "old_head": old_head[:8], "new_head": new_head[:8]}
            else:
                return {"status": "up_to_date", "old_head": old_head[:8], "new_head": new_head[:8]}

        except Exception as e:
            # A9-P2-8: Log full error server-side but don't leak internals to client
            log_error(f"[API] admin/update failed: {e}")
            return JSONResponse({"error": "Update failed — check server logs"}, status_code=500)

@app.post("/api/password/set")
async def api_set_password(request: Request):
    """Set app password (first-time setup, or from SSH remote setup)."""
    # Only allow from localhost or if no password set yet
    client_ip = request.client.host
    existing_hash = _get_app_password_hash()
    is_local = client_ip in ("127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1")

    if not is_local:
        return JSONResponse({"error": "Password can only be set from localhost."}, status_code=403)
    if existing_hash:
        return JSONResponse({"error": "Password already set. Use SSH to reset."}, status_code=403)

    try:
        body = await request.json()
    except Exception as e:
        log_warning(f"[API] password/set invalid body: {e}")
        return JSONResponse({"error": "Invalid request body"}, status_code=400)

    password = body.get("password", "")
    if len(password) < 4:
        return JSONResponse({"error": "Password too short (min 4 chars)"}, status_code=400)
    if len(password) > 128:
        return JSONResponse({"error": "Password too long"}, status_code=400)

    # Hash with Argon2id (RFC 9106 low-memory profile — 64 MiB, 3 iterations, 4 lanes).
    # HIGH_MEMORY (2 GiB) profile OOMs alongside Whisper on commodity PCs; LOW_MEMORY
    # is the RFC-recommended floor for memory-constrained hosts and is still
    # orders-of-magnitude stronger than the old default.
    try:
        from argon2 import PasswordHasher
        try:
            from argon2.profiles import RFC_9106_LOW_MEMORY
            ph = PasswordHasher.from_parameters(RFC_9106_LOW_MEMORY)
        except Exception:
            ph = PasswordHasher()
        password_hash = ph.hash(password)
        _set_app_password_hash(password_hash)
        return {"status": "ok"}
    except ImportError:
        import hashlib as hl
        salt = secrets.token_hex(16)
        h = hl.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000).hex()
        _set_app_password_hash(f"pbkdf2:{salt}:{h}")
        return {"status": "ok", "note": "Using PBKDF2 fallback (install argon2-cffi for Argon2id)"}

@app.get("/api/version")
async def api_version():
    """Public version endpoint — no auth needed.

    Useful for clients to check compatibility before pairing, and for
    Vercel-hosted dashboards to display server version. Exposes:
    - VERSION (e.g. "1.2.0")
    - git commit hash (if .git available)
    - server start time (uptime computable from this + now())
    """
    git_hash = "unknown"
    try:
        import subprocess
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            git_hash = result.stdout.strip() or "unknown"
    except Exception:
        pass
    return {
        "version": _SERVER_VERSION,
        "git": git_hash,
        "started_at": int(_SERVER_START_TIME),
        "service": "sanketra",
    }

@app.get("/api/health")
async def api_health(request: Request, token: str = Query(None)):
    """Server health: model state, GPU, connections, input guard, uptime."""
    if not _verify_session(_extract_token(request, token)):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    uptime_s = int(time.time() - _SERVER_START_TIME)
    gpu_used, gpu_total, gpu_util = get_gpu_stats()

    return {
        "status": "ok",
        "version": _SERVER_VERSION,
        "uptime_seconds": uptime_s,
        "model": {
            "state": _server_state.name,
            "name": _loaded_model_name,
            "device": _loaded_device,
        },
        "gpu": {
            "used_gb": round(gpu_used, 2),
            "total_gb": round(gpu_total, 2),
            "utilization_pct": gpu_util,
        },
        "connections": {
            "audio": len(_active_audio_ws),
            "trackpad": _active_trackpad_ws,
            "screen": _active_screen_ws,
            "audio_out": _active_audio_out_ws,
        },
        "input_guard": _input_guard_enabled,
        "service_mode": _SERVICE_MODE,
    }


# =============================================================================
#                              LICENSE (offline signed-key verification)
# =============================================================================
#
# See src/license_core.py for the verification logic and license/README.md
# for the wire format. Both endpoints require an authenticated session —
# the license only controls which features are unlocked, it doesn't replace
# auth.
#
# Feature gating itself is a follow-up commit: these endpoints expose the
# current tier so clients (Android + native desktop) can display "Free" /
# "Pro" in their UI without the server needing to re-verify on every RPC.

@app.get("/api/license-status")
async def api_license_status(request: Request, token: str = Query(None)):
    """Report the currently installed license — if any.

    Response shape:
        { "active": bool, "track": str|None, "email": str|None,
          "license_id": str|None, "issued_at": int|None }

    `active=false` with everything else None means the server is at free
    tier (no license installed, or an installed license failed to verify
    and has been ignored). This endpoint NEVER returns 500 for a license
    problem — license issues are never the reason the server stops
    working.
    """
    if (err := _require_auth(request, token)):
        return err
    lic = license_core.get_active_license()
    if lic is None:
        return {
            "active": False,
            "track": None,
            "email": None,
            "license_id": None,
            "issued_at": None,
        }
    return {
        "active": True,
        "track": lic.track,
        "email": lic.email,
        "license_id": lic.license_id,
        "issued_at": lic.issued_at,
    }


@app.post("/api/license-install")
async def api_license_install(request: Request, token: str = Query(None)):
    """Install a new license key (body: {"key": "SKT-..."}).

    Idempotent: posting the same key twice leaves the same file on disk
    and returns the same response. Posting a DIFFERENT valid key
    replaces the installed one atomically (tmp + fsync + os.replace, so
    a crash mid-install doesn't corrupt the existing license)."""
    if (err := _require_auth(request, token)):
        return err

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    key_str = (body or {}).get("key") if isinstance(body, dict) else None
    if not isinstance(key_str, str) or not key_str.strip():
        return JSONResponse(
            {"error": "body.key must be a non-empty string"}, status_code=400
        )

    try:
        lic = license_core.install_license_key(key_str)
    except license_core.LicenseInstallError as e:
        # Do NOT echo the submitted key back in the error — logs only.
        log_error(f"[License] install rejected: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        log_error(f"[License] install failed unexpectedly: {e}")
        return JSONResponse(
            {"error": "license install failed"}, status_code=500
        )

    return {
        "active": True,
        "track": lic.track,
        "email": lic.email,
        "license_id": lic.license_id,
        "issued_at": lic.issued_at,
    }


# =============================================================================
#                              v1.2: VOCAB / ACCENT / HISTORY
# =============================================================================
#
# All endpoints below require an authenticated session (same pattern as
# /api/health + /api/models — _verify_session on Bearer or ?token= query).
# They delegate to the three stores (history_db, vocab_store, accent_store)
# and never touch inference state.

def _require_auth(request: Request, token: str) -> JSONResponse | None:
    """Shared auth check — returns a 401 response to return directly from the
    endpoint, or None if authorized. Replaces the 2-line if-not-auth pattern
    with a single call site."""
    if not _verify_session(_extract_token(request, token)):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return None


# ---------- VOCAB ----------

@app.get("/api/vocab")
async def api_vocab_get(request: Request, token: str = Query(None)):
    if (err := _require_auth(request, token)):
        return err
    try:
        return vocab_store.get_default_store().get_all()
    except Exception as e:
        log_error(f"[Vocab] GET failed: {e}")
        return JSONResponse({"error": "vocab read failed"}, status_code=500)


@app.post("/api/vocab")
async def api_vocab_replace(request: Request, token: str = Query(None)):
    """Replace the full entries list. Body: {"entries": [...]}."""
    if (err := _require_auth(request, token)):
        return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    entries = body.get("entries")
    if not isinstance(entries, list):
        return JSONResponse({"error": "body.entries must be a list"}, status_code=400)
    try:
        data = vocab_store.get_default_store().replace_all(entries)
        return data
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        log_error(f"[Vocab] POST failed: {e}")
        return JSONResponse({"error": "vocab write failed"}, status_code=500)


@app.patch("/api/vocab")
async def api_vocab_patch(request: Request, token: str = Query(None)):
    """Batch add/remove. Body: {"add": [...entries], "remove": [...texts]}."""
    if (err := _require_auth(request, token)):
        return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    add = body.get("add") or []
    remove = body.get("remove") or []
    if not isinstance(add, list) or not isinstance(remove, list):
        return JSONResponse({"error": "add and remove must be lists"}, status_code=400)
    try:
        return vocab_store.get_default_store().patch(add=add, remove=remove)
    except Exception as e:
        log_error(f"[Vocab] PATCH failed: {e}")
        return JSONResponse({"error": "vocab patch failed"}, status_code=500)


@app.delete("/api/vocab/{text:path}")
async def api_vocab_delete(text: str, request: Request, token: str = Query(None)):
    """Delete a single entry by canonical text. `text:path` allows slashes/unicode."""
    if (err := _require_auth(request, token)):
        return err
    try:
        removed = vocab_store.get_default_store().remove_entry(text)
        if not removed:
            return JSONResponse({"error": "not found"}, status_code=404)
        return {"status": "ok", "text": text}
    except Exception as e:
        log_error(f"[Vocab] DELETE failed: {e}")
        return JSONResponse({"error": "vocab delete failed"}, status_code=500)


# ---------- ACCENT ----------

@app.post("/api/accent/calibrate")
async def api_accent_calibrate(
    request: Request,
    token: str = Query(None),
    audio: UploadFile = File(None),
    sample_rate: int = Form(None),
    client_name: str = Form(None),
):
    """Multipart audio (field name `audio`) → extract + persist profile.

    Fallback: if the client POSTs a raw audio body (no multipart), read
    request.body() and require `?sample_rate=` so we can decode PCM16.
    """
    if (err := _require_auth(request, token)):
        return err
    try:
        if audio is not None:
            data = await audio.read()
            if not data:
                return JSONResponse({"error": "empty upload"}, status_code=400)
        else:
            # Raw body fallback — Android may stream PCM16 directly.
            data = await request.body()
            if not data:
                return JSONResponse({"error": "missing audio (multipart field or raw body)"}, status_code=400)
            if sample_rate is None:
                try:
                    sample_rate = int(request.query_params.get("sample_rate", ""))
                except ValueError:
                    sample_rate = None
        try:
            meta = accent_store.get_default_store().save_profile(
                data,
                sample_rate=sample_rate,
                client_name=client_name,
            )
        except ValueError as ve:
            return JSONResponse({"error": str(ve)}, status_code=400)
        return {
            "profile_id": meta["profile_id"],
            "dialect_estimate": meta.get("dialect_estimate", "unknown"),
            "last_calibrated": meta["last_calibrated"],
            "sample_count": meta["sample_count"],
            "duration_sec": meta["duration_sec"],
            "backend": meta["backend"],
        }
    except Exception as e:
        log_error(f"[Accent] calibrate failed: {e}")
        return JSONResponse({"error": "calibration failed"}, status_code=500)


@app.get("/api/accent")
async def api_accent_get(request: Request, token: str = Query(None)):
    if (err := _require_auth(request, token)):
        return err
    try:
        return accent_store.get_default_store().get_metadata()
    except Exception as e:
        log_error(f"[Accent] GET failed: {e}")
        return JSONResponse({"error": "accent read failed"}, status_code=500)


@app.post("/api/accent/reset")
async def api_accent_reset(request: Request, token: str = Query(None)):
    if (err := _require_auth(request, token)):
        return err
    try:
        accent_store.get_default_store().reset()
        return {"status": "ok"}
    except Exception as e:
        log_error(f"[Accent] reset failed: {e}")
        return JSONResponse({"error": "accent reset failed"}, status_code=500)


# ---------- HISTORY ----------

@app.get("/api/history/sessions")
async def api_history_sessions(
    request: Request,
    token: str = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    if (err := _require_auth(request, token)):
        return err
    try:
        rows = history_db.get_default_db().list_sessions(limit=limit, offset=offset)
        return {"sessions": rows, "limit": limit, "offset": offset}
    except Exception as e:
        log_error(f"[History] list_sessions failed: {e}")
        return JSONResponse({"error": "history read failed"}, status_code=500)


@app.get("/api/history/sessions/{session_id}")
async def api_history_session_detail(
    session_id: int,
    request: Request,
    token: str = Query(None),
):
    if (err := _require_auth(request, token)):
        return err
    try:
        s = history_db.get_default_db().get_session(session_id)
        if s is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return s
    except Exception as e:
        log_error(f"[History] get_session failed: {e}")
        return JSONResponse({"error": "history read failed"}, status_code=500)


@app.get("/api/history/by-date/{date}")
async def api_history_by_date(
    date: str,
    request: Request,
    token: str = Query(None),
):
    if (err := _require_auth(request, token)):
        return err
    try:
        rows = history_db.get_default_db().list_by_date(date)
        return {"date": date, "transcripts": rows}
    except ValueError as ve:
        return JSONResponse({"error": str(ve)}, status_code=400)
    except Exception as e:
        log_error(f"[History] list_by_date failed: {e}")
        return JSONResponse({"error": "history read failed"}, status_code=500)


@app.get("/api/history/search")
async def api_history_search(
    request: Request,
    token: str = Query(None),
    q: str = Query("", max_length=500),
    limit: int = Query(100, ge=1, le=500),
):
    if (err := _require_auth(request, token)):
        return err
    try:
        rows = history_db.get_default_db().search(q, limit=limit)
        return {"query": q, "results": rows, "count": len(rows)}
    except Exception as e:
        log_error(f"[History] search failed: {e}")
        return JSONResponse({"error": "history search failed"}, status_code=500)


@app.delete("/api/history/sessions/{session_id}")
async def api_history_delete_session(
    session_id: int,
    request: Request,
    token: str = Query(None),
):
    if (err := _require_auth(request, token)):
        return err
    try:
        removed = history_db.get_default_db().delete_session(session_id)
        if removed == 0:
            return JSONResponse({"error": "not found"}, status_code=404)
        return {"status": "ok", "deleted_sessions": removed}
    except Exception as e:
        log_error(f"[History] delete_session failed: {e}")
        return JSONResponse({"error": "history delete failed"}, status_code=500)


@app.delete("/api/history/by-date/{date}")
async def api_history_delete_by_date(
    date: str,
    request: Request,
    token: str = Query(None),
):
    if (err := _require_auth(request, token)):
        return err
    try:
        count = history_db.get_default_db().delete_by_date(date)
        return {"status": "ok", "deleted_transcripts": count}
    except ValueError as ve:
        return JSONResponse({"error": str(ve)}, status_code=400)
    except Exception as e:
        log_error(f"[History] delete_by_date failed: {e}")
        return JSONResponse({"error": "history delete failed"}, status_code=500)


@app.delete("/api/history")
async def api_history_clear_all(request: Request, token: str = Query(None), confirm: str = Query(None)):
    """Clear all. Requires ?confirm=yes to prevent accidental DELETE.

    (Minimal confirm-token pattern from the spec — defense in depth against
    a misbehaving client that sends DELETE /api/history without confirmation.)
    """
    if (err := _require_auth(request, token)):
        return err
    if confirm != "yes":
        return JSONResponse({"error": "confirmation required: add ?confirm=yes"}, status_code=400)
    try:
        s_count, t_count = history_db.get_default_db().clear_all()
        return {"status": "ok", "deleted_sessions": s_count, "deleted_transcripts": t_count}
    except Exception as e:
        log_error(f"[History] clear_all failed: {e}")
        return JSONResponse({"error": "history clear failed"}, status_code=500)


@app.get("/api/history/export")
async def api_history_export(
    request: Request,
    token: str = Query(None),
    fmt: str = Query("json", pattern="^(json|txt|md)$"),
):
    if (err := _require_auth(request, token)):
        return err
    try:
        db = history_db.get_default_db()
        if fmt == "json":
            return db.export_json()
        if fmt == "txt":
            return PlainTextResponse(
                db.export_txt(),
                headers={"Content-Disposition": 'attachment; filename="sanketra-history.txt"'},
            )
        # md
        return PlainTextResponse(
            db.export_md(),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="sanketra-history.md"'},
        )
    except Exception as e:
        log_error(f"[History] export failed: {e}")
        return JSONResponse({"error": "history export failed"}, status_code=500)


@app.get("/api/history/settings")
async def api_history_settings_get(request: Request, token: str = Query(None)):
    if (err := _require_auth(request, token)):
        return err
    return history_db.get_default_db().get_settings()


@app.post("/api/history/settings")
async def api_history_settings_set(request: Request, token: str = Query(None)):
    if (err := _require_auth(request, token)):
        return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    le = body.get("logging_enabled")
    if le is None:
        return JSONResponse({"error": "body.logging_enabled required"}, status_code=400)
    return history_db.get_default_db().set_settings(logging_enabled=bool(le))


# ---------- DASHBOARD REAL-TIME WS ----------

@app.websocket("/ws-dashboard")
async def ws_dashboard(ws: WebSocket):
    """Lightweight push channel — broadcasts {"type":"transcript", ...}
    whenever a new final is logged.

    Auth reuses the shared `_accept_authenticated_ws` helper (WebSocket
    Subprotocol / first-message auth). This endpoint has no receive loop:
    it's purely server→client. We keep the socket alive until the client
    disconnects; any bytes the client sends are ignored.
    """
    cid = _resolve_cid(ws)
    client_ip = ws.client.host if ws.client else "unknown"
    if not await _accept_authenticated_ws(ws, cid, "Dashboard"):
        return
    # Soft cap — prevents a misconfigured browser refresh storm from leaking.
    if len(_dashboard_ws_clients) >= _MAX_DASHBOARD_WS:
        try:
            await ws.close(code=4429, reason="Too many dashboards")
        except Exception:
            pass
        _unregister_ws(ws)
        return
    _dashboard_ws_clients.add(ws)
    _ws_log(cid, "info", f"[Dashboard] WS connected from {client_ip} (total={len(_dashboard_ws_clients)})")
    try:
        await ws.send_text(json.dumps({"type": "hello", "version": _SERVER_VERSION}))
        # Passive keep-alive: just wait for disconnect. We don't need the
        # client to say anything; any receive errors close the socket.
        while True:
            try:
                await ws.receive_text()
            except WebSocketDisconnect:
                break
            except Exception:
                break
    finally:
        _dashboard_ws_clients.discard(ws)
        _unregister_ws(ws)
        _ws_log(cid, "info", f"[Dashboard] WS closed (remaining={len(_dashboard_ws_clients)})")


def _do_screen_blink():
    """Cross-platform fullscreen white flash — 3 blinks, no focus steal"""
    global _blink_active
    try:
        system = platform.system()
        if system == "Linux":
            _blink_linux()
        elif system == "Windows":
            _blink_windows()
        elif system == "Darwin":
            _blink_macos()
    except Exception as e:
        log_error(f"Screen blink failed: {e}")
    finally:
        _blink_active = False

def _blink_linux():
    """Linux screen flash — X11 override-redirect or tkinter fallback"""
    display_env = os.environ.get("DISPLAY", "")
    wayland = os.environ.get("WAYLAND_DISPLAY", "")

    if display_env and not wayland:
        # X11: override-redirect fullscreen white window (no focus steal)
        try:
            import ctypes
            import ctypes.util
            xlib = ctypes.cdll.LoadLibrary(ctypes.util.find_library("X11"))
            display = xlib.XOpenDisplay(None)
            if not display:
                raise RuntimeError("Cannot open X11 display")
            screen = xlib.XDefaultScreen(display)
            root = xlib.XRootWindow(display, screen)
            w = xlib.XDisplayWidth(display, screen)
            h = xlib.XDisplayHeight(display, screen)

            # Create override-redirect window (bypasses WM, no focus steal)
            win = xlib.XCreateSimpleWindow(display, root, 0, 0, w, h, 0, 0xFFFFFF, 0xFFFFFF)
            # Set override redirect
            from ctypes import Structure, c_int, c_long, c_ulong, byref
            class XSetWindowAttributes(Structure):
                _fields_ = [("background_pixmap", c_ulong), ("background_pixel", c_ulong),
                            ("border_pixmap", c_ulong), ("border_pixel", c_ulong),
                            ("bit_gravity", c_int), ("win_gravity", c_int),
                            ("backing_store", c_int), ("backing_planes", c_ulong),
                            ("backing_pixel", c_ulong), ("save_under", c_int),
                            ("event_mask", c_long), ("do_not_propagate_mask", c_long),
                            ("override_redirect", c_int), ("colormap", c_ulong), ("cursor", c_ulong)]
            attrs = XSetWindowAttributes()
            attrs.override_redirect = 1
            CWOverrideRedirect = 512
            xlib.XChangeWindowAttributes(display, win, CWOverrideRedirect, byref(attrs))

            for _ in range(3):
                xlib.XMapRaised(display, win)
                xlib.XFlush(display)
                time.sleep(0.2)
                xlib.XUnmapWindow(display, win)
                xlib.XFlush(display)
                time.sleep(0.1)

            xlib.XDestroyWindow(display, win)
            xlib.XCloseDisplay(display)
            return
        except Exception as e:
            log_warning(f"X11 blink failed, trying tkinter: {e}")

    # Wayland or X11 fallback: tkinter
    try:
        import tkinter as tk
        for _ in range(3):
            root = tk.Tk()
            root.attributes('-fullscreen', True)
            root.attributes('-topmost', True)
            root.configure(bg='white')
            root.update()
            time.sleep(0.2)
            root.destroy()
            time.sleep(0.1)
    except Exception as e:
        log_error(f"Tkinter blink failed: {e}")

def _blink_windows():
    """Windows screen flash — WS_EX_NOACTIVATE popup"""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        w = user32.GetSystemMetrics(0)
        h = user32.GetSystemMetrics(1)
        # Use PowerShell to flash (simpler, no Win32 registration)
        ps_script = f'''
Add-Type -AssemblyName System.Windows.Forms
for ($i = 0; $i -lt 3; $i++) {{
    $f = New-Object System.Windows.Forms.Form
    $f.FormBorderStyle = 'None'
    $f.WindowState = 'Maximized'
    $f.BackColor = 'White'
    $f.TopMost = $true
    $f.ShowInTaskbar = $false
    $f.Show()
    Start-Sleep -Milliseconds 200
    $f.Close()
    Start-Sleep -Milliseconds 100
}}'''
        subprocess.run(["powershell", "-Command", ps_script], timeout=5, capture_output=True)
    except Exception as e:
        log_error(f"Windows blink failed: {e}")

def _blink_macos():
    """macOS screen flash — tkinter fullscreen white window (no Accessibility/Automation perms needed)"""
    try:
        import tkinter as tk
        for _ in range(3):
            root = tk.Tk()
            root.attributes('-fullscreen', True)
            root.attributes('-topmost', True)
            root.configure(bg='white')
            root.overrideredirect(True)
            root.update()
            time.sleep(0.15)
            root.destroy()
            time.sleep(0.1)
    except Exception as e:
        log_error(f"macOS blink failed: {e}")

# =============================================================================
#                              HTML/JS CLIENT
# =============================================================================

def _load_html_bytes() -> bytes:
    """Load the embedded web client HTML from static/index.html.

    Cached at import time so per-request cost is the same as the previous
    inline-string approach. Splitting the HTML out (was 1893 lines wedged
    into a triple-quoted Python literal at this site) lets us actually
    lint, format, and test it as JavaScript instead of as a Python string.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    html_path = os.path.join(root_dir, "static", "index.html")
    try:
        with open(html_path, "rb") as f:
            return f.read()
    except Exception as e:
        try:
            log_warning(f"Failed to load static/index.html: {e}")
        except Exception:
            pass
        return (
            b"<!DOCTYPE html><html><body>"
            b"<h1>Sanketra server</h1>"
            b"<p>Static web client missing. Re-run <code>python setup.py</code> on this PC.</p>"
            b"</body></html>"
        )

# Pre-load HTML to bytes once at import time (skip ~98 KB UTF-8 read per request)
_html_bytes = _load_html_bytes()

# =============================================================================
#                              SSL CERT GENERATION
# =============================================================================

def _is_preferred_lan_ip(ip):
    """Check if IP is a typical LAN address (192.168.x.x or 172.16-31.x.x preferred over 10.x.x.x VPN)"""
    if ip.startswith('192.168.') or ip.startswith('172.'):
        return 2  # Most likely WiFi/LAN
    if ip.startswith('10.'):
        return 1  # Could be VPN or corporate LAN
    return 0

def get_local_ip():
    """Get local IP address. Prefers WiFi/LAN IPs over VPN on multi-NIC systems."""
    primary_ip = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            primary_ip = s.getsockname()[0]
        finally:
            s.close()
    except Exception as e:
        logging.debug(f"get_local_ip socket probe failed: {e}")

    # If primary IP looks good (not loopback), return it
    if primary_ip and not primary_ip.startswith('127.'):
        return primary_ip

    # Fallback: enumerate all interfaces, pick best RFC1918 address
    try:
        addrs = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
        candidates = [addr[4][0] for addr in addrs if not addr[4][0].startswith('127.')]
        if candidates:
            # Sort by LAN preference (192.168 > 10.x)
            candidates.sort(key=_is_preferred_lan_ip, reverse=True)
            return candidates[0]
    except Exception as e2:
        logging.debug(f"get_local_ip interface enum failed: {e2}")

    return primary_ip or '127.0.0.1'

def generate_ssl_certs(cert_path, key_path):
    """Generate self-signed SSL certificates"""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    import datetime
    import ipaddress

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "IN"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "STT"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])

    san_list = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        x509.IPAddress(ipaddress.IPv6Address("::1")),
    ]

    try:
        local_ip = get_local_ip()
        san_list.append(x509.IPAddress(ipaddress.IPv4Address(local_ip)))
    except Exception as e:
        log_warning(f"Could not add local IP to SSL SAN list: {e}")

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
        .sign(key, hashes.SHA256())
    )

    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))
    if platform.system() != 'Windows':
        os.chmod(key_path, 0o600)

    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print(f"Generated SSL certificates for localhost and {get_local_ip()}")

def print_qr(url, save_image=False):
    """Print QR code to console, optionally save as SVG"""
    qr = qrcode.QRCode(version=1, box_size=10, border=2)
    qr.add_data(url)
    qr.make(fit=True)

    if save_image:
        # Save QR as SVG for background mode (no console)
        img = qr.make_image(image_factory=qrcode.image.svg.SvgImage)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)  # Go up from src/ to root
        qr_path = os.path.join(root_dir, "qr_code.svg")
        img.save(qr_path)
        # Open the image
        if platform.system() == "Windows":
            os.startfile(qr_path)
        elif platform.system() == "Darwin":
            subprocess.Popen(['open', qr_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(['xdg-open', qr_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return qr_path
    else:
        qr.print_ascii(invert=True)
        return None

# =============================================================================
#                              MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Async STT WebSocket Server (FastAPI + uvicorn)")
    parser.add_argument("-p", "--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument("--background", "-bg", action="store_true",
                        help="Background mode (no console, saves QR as image)")
    parser.add_argument("--no-input-guard", action="store_true",
                        help="Disable physical input detection (auto-pause when laptop kb/mouse used)")
    parser.add_argument("--service", action="store_true",
                        help="Service mode: start IDLE, no model loaded, API-driven model selection")

    args = parser.parse_args()

    # Input guard toggle
    _input_guard_enabled = not args.no_input_guard

    # Service mode
    _SERVICE_MODE = args.service

    # Background mode detection (pythonw.exe or --background flag)
    is_pythonw = sys.executable.lower().endswith('pythonw.exe')
    bg_mode = args.background or is_pythonw or args.service

    # Model selection: always interactive unless background or service mode
    direct_model = None
    if args.service:
        model_preference = "service"  # No model load at startup
    elif bg_mode:
        model_preference = "balanced"  # auto-select in background (no console)
    else:
        model_preference = "custom"  # interactive selection

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)  # Go up from src/ to root
    cert_path = os.path.join(root_dir, 'cert.pem')
    key_path = os.path.join(root_dir, 'key.pem')

    # Generate certs if missing OR if IP changed (DHCP rotation invalidates SAN)
    need_regen = False
    if not os.path.exists(cert_path) or not os.path.exists(key_path):
        need_regen = True
        print("SSL certificates not found, generating...")
    else:
        # Check if current IP is in the cert's SAN list
        try:
            from cryptography import x509
            from cryptography.x509.oid import ExtensionOID
            import ipaddress
            with open(cert_path, "rb") as f:
                cert = x509.load_pem_x509_certificate(f.read())
            san = cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
            san_ips = [str(ip) for ip in san.value.get_values_for_type(x509.IPAddress)]
            current_ip = get_local_ip()
            if current_ip not in san_ips:
                need_regen = True
                print(f"IP changed: cert has {san_ips}, current is {current_ip}. Regenerating SSL certs...")
        except Exception as e:
            need_regen = True
            print(f"Could not verify cert SANs ({e}), regenerating...")

    if need_regen:
        generate_ssl_certs(cert_path, key_path)

    # Initialize models (verbose=False in background mode)
    init_models(model_preference, direct_model=direct_model, verbose=not bg_mode)

    ip = get_local_ip()
    port = args.port

    # macOS Monterey+ reserves port 5000 for AirPlay Receiver.
    # If user didn't specify a custom port and the default (5000) is in use, try 5001.
    if platform.system() == "Darwin" and port == 5000:
        try:
            _probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            _probe.settimeout(1)
            _probe.bind(("0.0.0.0", 5000))
            _probe.close()
        except OSError:
            port = 5001
            log_info(f"Port 5000 in use (likely AirPlay Receiver on macOS), using port {port}")
            print(f"WARNING: Port 5000 in use (AirPlay Receiver?). Switching to port {port}.")
            print(f"  To disable AirPlay Receiver: System Settings > General > AirDrop & Handoff > AirPlay Receiver OFF")

    SERVER_PORT = port
    _udp_port = port + 1  # Sideband UDP port always one above HTTP port

    # Self-heal: ensure systemd service file has adequate stop timeout for GPU cleanup
    if platform.system() == "Linux" and args.service:
        try:
            svc_path = os.path.expanduser("~/.config/systemd/user/sanketra.service")
            if os.path.isfile(svc_path):
                svc_text = open(svc_path).read()
                if "TimeoutStopSec=3" in svc_text:
                    svc_text = svc_text.replace("TimeoutStopSec=3", "TimeoutStopSec=10")
                    with open(svc_path, "w") as f:
                        f.write(svc_text)
                    # A9-P3-5: Use subprocess.run instead of os.system (no shell injection risk here,
                    # but consistent with the rest of the codebase)
                    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
                    log_info("Self-healed systemd TimeoutStopSec 3→10")
        except Exception:
            pass
    url = f"https://{ip}:{port}"

    # Show URL (P0-1: never log/print AUTH_TOKEN — it's already in config.json)
    if not bg_mode:
        print(f"\n{'='*50}")
        print(f"Open: {url}")
        print(f"Token: {AUTH_TOKEN}")
        print(f"  ^ Do NOT share this token publicly")
        if not _pair_locked:
            print(f"Pair code: {_pair_code}")
        print(f"(Accept certificate warning on phone)")
        print(f"{'='*50}")
        print_qr(f"{url}?token={AUTH_TOKEN}", save_image=False)
        print()
    else:
        log_info(f"URL: {url}")
        if not _pair_locked:
            log_info(f"Pair code: {_pair_code}")

    # Graceful shutdown: on Linux/macOS, let uvicorn handle SIGTERM natively
    # (it does proper lifespan teardown). Only override SIGBREAK on Windows
    # (schtasks /end sends CTRL_BREAK which uvicorn doesn't handle).
    if platform.system() == 'Windows':
        import signal
        def _shutdown_handler(signum, frame):
            raise SystemExit(0)
        signal.signal(signal.SIGBREAK, _shutdown_handler)

    # atexit safety net: release GPU resources even if lifespan teardown doesn't run
    # (e.g. SIGTERM during startup, or interpreter shutdown before lifespan completes)
    import atexit
    _atexit_state = {"done": False}
    def _atexit_gpu_cleanup():
        if _atexit_state["done"]:
            return
        _atexit_state["done"] = True
        global whisper_model, vad_model, vad_utils
        try:
            if whisper_model is not None:
                whisper_model = None
            if vad_model is not None:
                vad_model = None
                vad_utils = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            cleanup_gpu()
        except Exception:
            pass  # Best effort during interpreter shutdown
    atexit.register(_atexit_gpu_cleanup)

    # Run server
    if bg_mode and platform.system() == "Windows":
        # Windows background mode: run uvicorn in thread, main thread waits
        def run_server():
            try:
                uvicorn.run(
                    app,
                    host="0.0.0.0",
                    port=port,
                    ssl_certfile=cert_path,
                    ssl_keyfile=key_path,
                    log_level="warning",
                    ws_ping_interval=20,
                    ws_ping_timeout=20,
                )
            except Exception as e:
                print(f"Server error: {e}")
                import traceback
                traceback.print_exc()

        server_thread = threading.Thread(target=run_server, daemon=False)
        server_thread.start()

        # Give server time to start
        time.sleep(2)
        print(f"Server running at https://{ip}:{port}")

        # Block main thread until server exits
        server_thread.join()
    else:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            ssl_certfile=cert_path,
            ssl_keyfile=key_path,
            log_level="info" if not bg_mode else "warning",
            ws_ping_interval=20,
            ws_ping_timeout=20,
        )
