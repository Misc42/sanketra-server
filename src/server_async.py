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
from enum import Enum, auto

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, Response, JSONResponse
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

# Auth token for WebSocket connections (generated at startup)
AUTH_TOKEN = secrets.token_urlsafe(24)

_paired_ips: set[str] = set()  # IPs that have paired (for logging)
_pair_attempts: dict[str, list[float]] = {}  # IP -> list of timestamps (rate limiting)
_PAIR_RATE_LIMIT = 5   # max attempts per window
_PAIR_RATE_WINDOW = 60  # seconds
_pair_locked = False  # True after first successful pair — blocks further pairing
_blink_active = False
SERVER_PORT = 5000  # updated at startup from args
_SERVICE_MODE = False  # Set True by --service flag

# Server state machine (service mode)
class ServerState(Enum):
    IDLE = auto()       # No model loaded, waiting for auth + model selection
    LOADING = auto()    # Model download/load in progress
    ACTIVE = auto()     # Model loaded, ready for STT

_server_state = ServerState.ACTIVE  # Default: ACTIVE (legacy mode loads model at startup)
_model_load_progress = 0.0  # 0.0-1.0
_model_load_cancel = False
_loaded_model_name = None
_last_ws_activity = time.time()  # For auto-unload timer
_MODEL_UNLOAD_TIMEOUT = 900  # 15 minutes no WS connections

# Per-client session management
_client_sessions: dict[str, dict] = {}  # token -> {ip, created, last_used}
_auth_attempts: dict[str, list[float]] = {}  # IP -> timestamps
_AUTH_RATE_LIMIT = 3   # max failed attempts per window
_AUTH_RATE_WINDOW = 60  # seconds
_AUTH_LOCKOUT = 600  # 10 minute lockout after exceeding rate limit
_SESSION_MAX_COUNT = 50        # Max concurrent sessions
_SESSION_TTL = 30 * 24 * 3600  # 30 days in seconds

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
    """Save config to ~/.config/sanketra/config.json.
    Caller MUST hold _config_lock when part of a read-modify-write."""
    os.makedirs(_CONFIG_DIR, exist_ok=True)
    with open(_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

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
    """Persist client sessions to config.
    Caller MUST hold _sessions_lock."""
    with _config_lock:
        config = _load_config()
        config["sessions"] = dict(_client_sessions)
        _save_config(config)

def _load_client_sessions():
    """Load persisted client sessions"""
    global _client_sessions
    with _sessions_lock:
        with _config_lock:
            config = _load_config()
            _client_sessions = config.get("sessions", {})

def _load_pair_lock():
    """Load pair lock state from config on startup."""
    global _pair_locked
    with _config_lock:
        config = _load_config()
        _pair_locked = config.get("pair_locked", False)

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
    pause_app_input, resume_app_input, is_app_input_paused, register_pause_callback
)
import preflight
import input_monitor

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
#                              GLOBAL STATE
# =============================================================================

whisper_model = None
_whisper_lock = threading.Lock()  # Guards model swap in _load_model_async and snapshot in transcribe_sync
vad_model = None
vad_utils = None
executor: Optional[ThreadPoolExecutor] = None      # Whisper inference
io_executor: Optional[ThreadPoolExecutor] = None    # type_text, blink, etc.
_model_load_executor: Optional[ThreadPoolExecutor] = None  # Model download/load (separate from transcription)

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

    def reset_vad(self):
        """Reset VAD state after transcription"""
        self.audio_buffer.clear()
        self.is_speaking = False
        self.had_speech = False
        self.silence_frames = 0
        if self.vad_iterator:
            self.vad_iterator.reset_states()

    def add_audio(self, audio_float: np.ndarray):
        """Add audio frame to buffer (deque auto-trims oldest at maxlen)"""
        # AU-P0-2: Log warning when ring buffer is full and dropping oldest frames
        if len(self.audio_buffer) == self.audio_buffer.maxlen:
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

_cached_language = None  # Caches detected language to avoid ~200ms re-detection per utterance

def transcribe_sync(audio: np.ndarray) -> tuple:
    """
    Synchronous Whisper transcription.
    Runs in thread pool, not in async loop.
    """
    global _cached_language
    with _whisper_lock:
        model = whisper_model  # Snapshot under lock to avoid race with _load_model_async / _unload_model
    if model is None:
        return "", ""
    if len(audio) < int(SAMPLE_RATE * 0.25):  # Min 250ms
        return "", ""

    try:
        segments, info = model.transcribe(
            audio,
            language=_cached_language,
            vad_filter=False,
            beam_size=1,
            condition_on_previous_text=False,
            initial_prompt="Hindi aur English mein baat ho rahi hai."
        )
        # Cache detected language for subsequent calls (saves ~200ms)
        if info.language in ("hi", "en", "ur"):
            _cached_language = info.language
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

def _check_ws_token(ws: WebSocket) -> bool:
    """Validate auth token from WebSocket query params."""
    token = ws.query_params.get('token', '')
    return _check_any_token(token)

async def _authenticate_ws(ws: WebSocket) -> bool:
    """Authenticate WS: check query params first, then wait for MSG_AUTH first message.
    Also enforces max concurrent WS limit."""
    # Enforce max concurrent WS
    with _ws_track_lock:
        if len(_active_audio_ws) >= _MAX_CONCURRENT_WS:
            return False

    if _check_ws_token(ws):
        return True
    # Wait for auth message (binary: [0xFA][utf8_token] or text: {"t":"auth","token":"..."})
    try:
        data = await asyncio.wait_for(ws.receive(), timeout=5.0)
        if 'bytes' in data and data['bytes']:
            raw = data['bytes']
            if raw[0] == MSG_AUTH:
                token = raw[1:].decode('utf-8', errors='ignore')
                return _check_any_token(token)
        elif 'text' in data and data['text']:
            msg = json.loads(data['text'])
            if msg.get('t') == 'auth':
                token = msg.get('token', '')
                return _check_any_token(token)
    except Exception as e:
        log_debug(f"[Auth] WS auth message receive failed: {e}")
    return False

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
        text, lang = await loop.run_in_executor(executor, transcribe_sync, audio)

        if text and lang in ('hi', 'en'):
            text = to_roman(text)
            lang_name = 'Hindi' if lang == 'hi' else 'English'
            result = f"[{lang_name}] {text}"
            await send_message(ws, MSG_FINAL, result.encode('utf-8'))
            log_info(f"[Stream] Transcribed: {result}")
            print(result, flush=True)

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
        await send_message(ws, MSG_ERROR, str(e).encode('utf-8'))
    finally:
        session.pending_inference = False
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
        text, lang = await loop.run_in_executor(executor, transcribe_sync, audio)

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

async def ws_audio_stream(ws: WebSocket):
    """Main WebSocket handler for audio streaming"""
    global _last_ws_activity
    await ws.accept()
    if not await _authenticate_ws(ws):
        try:
            await ws.close(code=4401, reason="Unauthorized")
        except Exception:
            pass
        return
    _register_ws(ws)
    log_info("[Stream] WebSocket connected")

    # Initialize session
    # vad_utils is a 5-tuple from Silero: (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)
    # EnergyVAD fallback may return fewer elements — guard the unpacking
    if not vad_utils or len(vad_utils) < 4:
        log_error("[Stream] VAD utils unavailable or incomplete — cannot start session")
        await ws.close(code=4500, reason="VAD unavailable")
        _unregister_ws(ws)
        return
    VADIterator = vad_utils[3]
    session = StreamingSession()
    session.vad_iterator = VADIterator(vad_model, sampling_rate=SAMPLE_RATE)

    # Warn phone immediately if macOS Accessibility is missing — transcription will
    # succeed but type_text() will silently discard all output.
    input_ok, input_err = check_input_health()
    if not input_ok:
        session._input_health_warned = True
        await send_message(ws, MSG_ERROR, input_err.encode('utf-8'))

    try:
        while True:
            try:
                data = await ws.receive_bytes()
            except WebSocketDisconnect as e:
                log_info(f"[Stream] Client disconnected (code={e.code})")
                break
            except Exception as e:
                log_error(f"[Stream] Receive error: {e}")
                break

            if not data:
                continue

            # Phone activity → resume app input if paused
            if is_app_input_paused():
                resume_app_input()

            msg_type = data[0]
            payload = data[1:] if len(data) > 1 else b""

            if msg_type == MSG_START_SESSION:
                session.session_active = True
                session.reset_vad()
                await _send_raw(ws, _WIRE_SESSION_READY)
                log_info("[Stream] Session started")

            elif msg_type == MSG_END_SESSION:
                log_info("[Stream] End session requested")

                # Flush remaining audio.
                # P3-9: This synchronous inference call typically completes in <5s
                # (Whisper beam_size=1 on buffered audio). The client shows a
                # "Processing..." state while waiting for SESSION_DONE. The Android
                # client has a 10s safety timeout that clears isProcessing if
                # SESSION_DONE never arrives (network drop, server crash). That
                # 10s value is conservative — 99th percentile inference is ~3s on
                # GPU, ~8s on CPU for 30s audio. The web client relies on WS close
                # to exit processing state, which is acceptable for browser use.
                if session.audio_buffer and session.had_speech:
                    await run_inference(session, ws)
                elif session.audio_buffer:
                    # Had audio but no speech detected - transcribe anyway
                    session.had_speech = True  # Force it
                    await run_inference(session, ws)

                # Signal done
                await _send_raw(ws, _WIRE_SESSION_DONE)
                session.session_active = False

                # Clean close
                await ws.close(code=1000, reason="Session complete")
                log_info("[Stream] Session ended cleanly")
                break

            elif msg_type == MSG_AUDIO_FRAME:
                if session.session_active:
                    # P2-28: Keep _last_ws_activity fresh so auto_unload_timer
                    # doesn't unload the model during active audio streaming.
                    _last_ws_activity = time.time()
                    await handle_audio_frame(session, ws, payload)

            else:
                await send_message(ws, MSG_ERROR, b"Unknown message type")

    except Exception as e:
        log_error(f"[Stream] Handler error: {e}")
    finally:
        # AU-P0-3: On disconnect, run final inference if buffer has speech data
        if session.audio_buffer and session.had_speech and not session.pending_inference:
            try:
                log_info("[Stream] Disconnect flush — transcribing remaining buffered speech")
                await run_inference(session, ws)
            except Exception as e:
                log_debug(f"[Stream] Disconnect flush failed (WS closing): {e}")
        _unregister_ws(ws)
        session.session_active = False
        log_info(f"[Stream] WebSocket handler ended (processed {session.frames_processed} frames)")

# =============================================================================
#                              TRACKPAD WEBSOCKET
# =============================================================================

import queue

# ─── UDP sideband for low-latency input (gyro, trackpad move, scroll) ───
# Bypasses TCP/WebSocket entirely — no head-of-line blocking.
# Auth: first packet must be JSON {"t":"auth","token":"<valid_token>"}.
# After auth, accepts: {"t":"a","x":...,"y":...}, {"t":"m","x":...,"y":...}, {"t":"s","dy":...}
_udp_transport = None
_udp_authed_clients: dict[tuple, bool] = {}  # (ip, port) -> True after auth

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
                _udp_authed_clients[addr] = True
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

        # Reject unauthenticated clients
        if addr not in _udp_authed_clients:
            return

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
            elif t == "rc":
                sw, sh = get_screen_resolution()
                mouse_move_absolute(sw // 2, sh // 2)
            elif t == "c":
                mouse_click(msg.get("b", 1))
            elif t == "d":
                mouse_drag(msg.get("a", "down"))
            elif t == "k":
                key_press(msg.get("k", ""))
            elif t == "s":
                mouse_scroll(msg.get("dy", 0) * 0.1)
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
    await ws.accept()
    if not await _authenticate_ws(ws):
        try:
            await ws.close(code=4401, reason="Unauthorized")
        except Exception:
            pass
        return
    _register_ws(ws)
    print("[Trackpad] Connected", flush=True)
    _ensure_tp_thread()

    # Warn phone on connect if macOS Accessibility is missing (all trackpad input will silently fail)
    input_ok, input_err = check_input_health()
    if not input_ok:
        try:
            await ws.send_text(json.dumps({"error": input_err}))
        except Exception:
            pass

    try:
        while True:
            try:
                data = await ws.receive_text()
            except WebSocketDisconnect:
                break
            except Exception as e:
                log_warning(f"[Trackpad] WS receive error: {e}")
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
                log_warning("[Trackpad] Queue full, dropping event")
            except Exception as e:
                log_warning(f"[Trackpad] WS message parse error: {e}")
    finally:
        _unregister_ws(ws)
        print("[Trackpad] Disconnected", flush=True)

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

    # Load pair lock state from config (persists across restarts)
    _load_pair_lock()

    if _SERVICE_MODE:
        # Service mode: start IDLE, no model loaded, load on-demand via API
        init_gpu()
        vad_model, vad_utils = load_vad()
        log_info(f"VAD: {get_vad_type()}")
        _server_state = ServerState.IDLE
        _loaded_model_name = None
        _load_client_sessions()
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

    whisper_model = load_whisper(model_name, device=device, compute_type=compute_type)
    _loaded_model_name = model_name
    _server_state = ServerState.ACTIVE

    vram_after, _, _ = get_gpu_stats()
    if vram_after > vram_before:
        log_info(f"VRAM used: {vram_after - vram_before:.2f} GB")

    log_info("Models loaded - ready to accept connections")


def _load_model_async(model_name, precision=None):
    """Load a Whisper model (called from API). Runs in executor."""
    global whisper_model, _server_state, _loaded_model_name, _model_load_progress, _model_load_cancel

    if _server_state == ServerState.LOADING:
        return {"error": "Already loading a model"}

    _server_state = ServerState.LOADING
    _model_load_progress = 0.0
    _model_load_cancel = False

    try:
        import torch
        has_gpu = torch.cuda.is_available()
        device = "cuda" if has_gpu else "cpu"

        if precision is None:
            from stt_common import auto_select_precision, get_vram_free, gpu_supports_float16
            vram = get_vram_free()
            fp16 = gpu_supports_float16() if has_gpu else False
            precision = auto_select_precision(model_name, vram, fp16) if has_gpu else "int8"

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

        new_model = load_whisper(model_name, device=device, compute_type=precision)

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
        _model_load_progress = 1.0
        _server_state = ServerState.ACTIVE
        log_info(f"Model loaded: {model_name} ({precision} on {device})")
        return {"status": "ready", "model": model_name, "precision": precision, "device": device}

    except Exception as e:
        _server_state = ServerState.IDLE
        _model_load_progress = 0.0
        log_error(f"Model load failed: {e}")
        return {"error": str(e)}


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
    global _last_ws_activity
    with _ws_track_lock:
        _active_audio_ws.add(ws)
    _last_ws_activity = time.time()

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
        for rate_dict, window in [(_pair_attempts, _PAIR_RATE_WINDOW), (_auth_attempts, _AUTH_RATE_WINDOW)]:
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

_WATCHDOG_GRACE_SECONDS = 15
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
    global _event_loop
    _event_loop = asyncio.get_event_loop()

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

    yield

    # Cleanup
    log_info("[Shutdown] Starting graceful shutdown...")

    if udp_transport:
        udp_transport.close()
    if unload_task:
        unload_task.cancel()
    if watchdog_task:
        watchdog_task.cancel()
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
_MAX_CONCURRENT_WS = 6  # N-P1-1: 3 WS types (audio/trackpad/screen) + headroom for reconnect overlap

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

    # Signal end to consumer
    try:
        frame_queue.put_nowait(None)
    except queue.Full:
        pass


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


def _get_avfoundation_screen_device():
    """Parse avfoundation device list to find screen capture device index.
    Returns device string like '2:' for -i flag. Defaults to '1:' if parsing fails."""
    try:
        p = subprocess.run(
            ['ffmpeg', '-nostdin', '-f', 'avfoundation', '-list_devices', 'true', '-i', ''],
            capture_output=True, text=True, timeout=5,
        )
        # Device list is printed to stderr
        for line in (p.stderr or '').splitlines():
            # Look for "Capture screen" line, e.g.: [AVFoundation ...] [2] Capture screen 0
            if 'Capture screen' in line:
                m = re.search(r'\[(\d+)\]', line)
                if m:
                    return f'{m.group(1)}:'
    except Exception:
        pass
    return '1:'  # Default: screen is usually device index 1

def _check_hw_encoder():
    """Probe hardware encoder availability with a short real capture test.
    Returns encoder name string ('h264_nvenc', 'h264_videotoolbox') or None."""
    plat = PLATFORM
    if plat == 'linux':
        display = os.environ.get('DISPLAY', ':0')
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
            log_debug(f"[Screen] NVENC probe failed on Linux: {e}")
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
                      screen_w=None, screen_h=None, display=None):
    """Build ffmpeg command for screen capture (pure function, no side effects).

    Captures at full screen resolution and scales down to out_w x out_h
    so the entire desktop is visible (no cropping).
    Platform-aware: x11grab (Linux), gdigrab (Windows), avfoundation (macOS).
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

    # Platform-specific input
    if plat == 'linux':
        cmd += [
            '-f', 'x11grab',
            '-framerate', str(fps),
            '-video_size', f'{cap_w}x{cap_h}',
            '-i', display or ':0',
        ]
    elif plat == 'windows':
        cmd += [
            '-f', 'gdigrab',
            '-framerate', str(fps),
            '-offset_x', '0', '-offset_y', '0',  # Pin to primary monitor on multi-monitor setups
            '-video_size', f'{cap_w}x{cap_h}',
            '-i', 'desktop',
        ]
    elif plat == 'macos':
        screen_dev = _get_avfoundation_screen_device()
        cmd += [
            '-f', 'avfoundation',
            '-framerate', str(fps),
            '-capture_cursor', '1',
            '-i', screen_dev,
        ]
    else:
        raise RuntimeError(f"Screen mirror not supported on {plat}")

    # Scale filter: only if capture != output resolution
    if cap_w != out_w or cap_h != out_h:
        cmd += ['-vf', f'scale={out_w}:{out_h}']
    elif plat == 'macos':
        # macOS avfoundation captures at native (Retina) resolution, always need scale
        cmd += ['-vf', f'scale={out_w}:{out_h}']

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


def _start_ffmpeg(w, h, fps, screen_w, screen_h):
    """Start ffmpeg for screen capture. Probes HW encoder once, falls back to libx264.

    Key design: NO waiting after Popen — the reader thread must start immediately
    to prevent the pipe buffer from filling and blocking ffmpeg.
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
                                           display=display)
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
                                                   display=display)
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
    await ws.accept()
    if not await _authenticate_ws(ws):
        try:
            await ws.close(code=4401, reason="Unauthorized")
        except Exception:
            pass
        return
    _register_ws(ws)
    log_info("[Screen] Client connected")

    screen_w, screen_h = get_screen_resolution_physical()
    fps = 30

    # Parse query params for resolution override
    query = ws.query_params if hasattr(ws, 'query_params') else {}
    req_res = query.get('res', '')

    # Target box for each preset (fit screen into this box, preserving aspect ratio)
    _RES_PRESETS = {'1080p': (1920, 1080), '720p': (1280, 720), '480p': (854, 480)}
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

    log_info(f"[Screen] Capture {screen_w}x{screen_h} -> {w}x{h}@{fps}fps")

    # Send config FIRST so client can prepare decoder before frames arrive
    config = struct.pack('<HHB', w, h, fps)  # 5 bytes: width(2) + height(2) + fps(1)
    await ws.send_bytes(bytes([0x20]) + config)  # 0x20 = SCREEN_CONFIG

    proc = None
    reader_thread = None
    stop_event = threading.Event()
    frame_queue = queue.Queue(maxsize=90)  # ~3 seconds buffer at 30fps

    try:
        proc, encoder = _start_ffmpeg(w, h, fps, screen_w, screen_h)

        # Start reader thread IMMEDIATELY — ffmpeg is already writing to stdout.
        # Any delay here risks filling the pipe buffer and stalling ffmpeg.
        reader_thread = threading.Thread(
            target=_read_h264_frames,
            args=(proc.stdout, frame_queue, stop_event),
            daemon=True,
            name="screen-reader",
        )
        reader_thread.start()

        # Sender loop — read frames from queue, send over WebSocket
        # N-P1-4: Track send backpressure; drop P-frames if send is slow
        # N-P2-6: Track keyframe state — drop P-frames after lost keyframe
        loop = asyncio.get_event_loop()
        _awaiting_keyframe = False  # Set True if a keyframe was dropped
        while True:
            try:
                frame = await loop.run_in_executor(None, lambda: frame_queue.get(timeout=1.0))
            except queue.Empty:
                # Check if ffmpeg is still alive
                if proc.poll() is not None:
                    log_error(f"[Screen] ffmpeg exited (code={proc.returncode})")
                    break
                continue

            if frame is None:
                break  # Reader thread signaled end

            is_key, timestamp_ms, h264_data = frame

            # N-P2-6: If we lost a keyframe, drop all P-frames until next keyframe
            if _awaiting_keyframe:
                if is_key:
                    _awaiting_keyframe = False
                    log_debug("[Screen] Keyframe received — resuming send")
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
                    log_warning(f"[Screen] Keyframe send failed — will skip until next keyframe: {e}")
                break

    except WebSocketDisconnect:
        log_info("[Screen] Client disconnected")
    except RuntimeError as e:
        log_error(f"[Screen] Failed to start: {e}")
    except Exception as e:
        log_error(f"[Screen] Error: {e}")
    finally:
        _unregister_ws(ws)
        stop_event.set()
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        # Remove from active set (P2-29)
        if proc:
            with _ffmpeg_procs_lock:
                _active_ffmpeg_procs.discard(proc)
        if reader_thread and reader_thread.is_alive():
            reader_thread.join(timeout=2)
        log_info("[Screen] Session ended")


@app.websocket("/ws-audio-stream")
async def websocket_audio_endpoint(websocket: WebSocket):
    await ws_audio_stream(websocket)

@app.websocket("/ws-tp")
async def websocket_trackpad_endpoint(websocket: WebSocket):
    await ws_trackpad(websocket)

@app.websocket("/ws-screen")
async def websocket_screen_endpoint(websocket: WebSocket):
    await ws_screen_mirror(websocket)

@app.get("/")
async def index():
    return Response(
        content=_html_bytes,
        media_type="text/html",
        headers={"Cache-Control": "private, max-age=300"},  # 5 min cache — only changes on restart
    )

@app.get("/api/screen")
async def screen_info(request: Request, token: str = Query(None)):
    if not _check_any_token(_extract_token(request, token)):
        return Response(status_code=401)
    w, h = get_screen_resolution()
    return {"w": w, "h": h}

def _get_preferred_ip():
    """Default-route IP — typically the Ethernet interface."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None

@app.get("/api/discover")
async def api_discover():
    """LAN discovery endpoint — no auth required, returns service identity"""
    resp = {"service": "sanketra", "name": platform.node(), "port": SERVER_PORT, "udp_port": _udp_port, "os": platform.system(), "server_id": _get_server_id()}
    pip = _get_preferred_ip()
    if pip:
        resp["preferred_ip"] = pip
    return resp

@app.post("/api/pair")
async def api_pair(request: Request):
    """SSH-style TOFU pairing — first connect is trusted, returns auth token + cert fingerprint.
    After first successful pair, pairing is locked. Use /api/unpair to unlock."""
    client_ip = request.client.host

    # Per-IP rate limiting: max _PAIR_RATE_LIMIT attempts per _PAIR_RATE_WINDOW seconds
    # Always applied BEFORE lock check to prevent timing side-channels
    now = time.time()
    attempts = _pair_attempts.get(client_ip, [])
    attempts = [t for t in attempts if now - t < _PAIR_RATE_WINDOW]
    if len(attempts) >= _PAIR_RATE_LIMIT:
        log_info(f"PAIR rate-limited: {client_ip} ({len(attempts)} attempts in {_PAIR_RATE_WINDOW}s)")
        return Response(status_code=429)
    attempts.append(now)
    _pair_attempts[client_ip] = attempts

    # Reject if already paired — only /api/unpair (with valid token) can unlock
    if _pair_locked:
        log_info(f"PAIR rejected (locked): {client_ip}")
        return JSONResponse(
            {"error": "Pairing locked. Unpair from current device first."},
            status_code=403,
        )

    # Return token + cert DER fingerprint for TOFU pinning
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

    # Lock pairing after first success — persisted to config.json
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
    loop.run_in_executor(io_executor, _do_screen_blink)
    return {"status": "blinking"}

# =============================================================================
#                     SERVICE MODE API (Phase 3)
# =============================================================================

def _verify_session(token: str) -> bool:
    """Verify a client session token is valid. Rejects expired sessions (30d TTL)."""
    if not token:
        return False
    # Accept AUTH_TOKEN directly (TOFU pairing token — works in all modes)
    if secrets.compare_digest(token, AUTH_TOKEN):
        return True
    # Per-client session tokens
    with _sessions_lock:
        session = _client_sessions.get(token)
        if session:
            # Reject expired sessions (P2-18: 30-day runtime expiry)
            created = session.get("created", 0)
            if time.time() - created > _SESSION_TTL:
                del _client_sessions[token]
                _save_client_sessions()
                return False
            session["last_used"] = time.time()
            return True
    return False

def _check_auth_rate_limit(ip: str) -> bool:
    """Returns True if IP is rate-limited (too many failed attempts)."""
    now = time.time()
    attempts = _auth_attempts.get(ip, [])
    # Clean old attempts
    attempts = [t for t in attempts if now - t < _AUTH_RATE_WINDOW]
    _auth_attempts[ip] = attempts
    if len(attempts) >= _AUTH_RATE_LIMIT:
        # Check if still in lockout
        if attempts and (now - attempts[-1]) < _AUTH_LOCKOUT:
            return True
    return False

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
            ph = PasswordHasher()
            ph.verify(stored_hash, password)
    except Exception as e:
        log_debug(f"[Auth] Password verification failed for {client_ip}: {e}")
        _record_auth_failure(client_ip)
        return JSONResponse({"error": "Invalid password"}, status_code=401)

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
        _save_client_sessions()

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
    """Trigger model download/load. Transitions server IDLE → LOADING → ACTIVE."""
    if not _verify_session(_extract_token(request, token)):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    if _server_state == ServerState.LOADING:
        return JSONResponse({"error": "Already loading a model"}, status_code=409)

    try:
        body = await request.json()
    except Exception as e:
        log_warning(f"[API] model/load invalid body: {e}")
        return JSONResponse({"error": "Invalid request body"}, status_code=400)

    model_name = body.get("model", "")
    precision = body.get("precision")  # Optional, auto-select if None
    valid_models = ["tiny", "base", "small", "medium", "large-v3-turbo", "distil-large-v3", "large-v3"]
    if model_name not in valid_models:
        return JSONResponse({"error": f"Invalid model. Choose from: {valid_models}"}, status_code=400)

    # Load model in dedicated executor (keeps main whisper executor free for transcription)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_model_load_executor, _load_model_async, model_name, precision)
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

@app.post("/api/password/set")
async def api_set_password(request: Request):
    """Set app password (first-time setup, or from SSH remote setup)."""
    # Only allow from localhost or if no password set yet
    client_ip = request.client.host
    existing_hash = _get_app_password_hash()
    is_local = client_ip in ("127.0.0.1", "::1", "localhost")

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

    # Hash with Argon2id
    try:
        from argon2 import PasswordHasher
        ph = PasswordHasher()
        password_hash = ph.hash(password)
        _set_app_password_hash(password_hash)
        return {"status": "ok"}
    except ImportError:
        # Fallback: use hashlib if argon2 not installed
        import hashlib as hl
        salt = secrets.token_hex(16)
        h = hl.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000).hex()
        _set_app_password_hash(f"pbkdf2:{salt}:{h}")
        return {"status": "ok", "note": "Using PBKDF2 fallback (install argon2-cffi for Argon2id)"}

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

HTML_CONTENT = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no, viewport-fit=cover">
    <meta name="theme-color" content="#0A0A0B">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <title>Sanketra</title>
    <style>
        :root {
            /* Golden Ratio Spacing — Fibonacci sequence */
            --space-3xs: 3px; --space-2xs: 5px; --space-xs: 8px;
            --space-sm: 13px; --space-md: 21px; --space-lg: 34px; --space-xl: 55px;
            /* Button sizes — phi ratio 64:40 = 1.6 */
            --btn-primary: 64px; --btn-secondary: 44px;
            /* Typography — golden ratio scale */
            --fs-xs: 0.65rem; --fs-sm: 0.75rem; --fs-base: 0.875rem;
            --fs-md: 1rem; --fs-lg: 1.618rem;
            --fw-normal: 400; --fw-medium: 500; --fw-semibold: 600;
            --ls-normal: 0; --ls-wide: 0.02em; --ls-caps: 0.06em;
            /* Colors — neutral grey + red recording + green accent */
            --surface-base: #0A0A0B; --surface-raised: #141416;
            --surface-overlay: #1C1C1F; --surface-highest: #252528;
            --surface-sheet: #1A1A1D;
            --text-primary: #F5F5F7; --text-secondary: #8E8E93; --text-tertiary: #7C7C82;
            --border-subtle: rgba(255,255,255,0.06); --border-medium: rgba(255,255,255,0.12);
            --accent: #34C759; --accent-muted: rgba(52,199,89,0.15);
            --recording: #FF3B30; --recording-muted: rgba(255,59,48,0.15);
            --muted-bg: #FF3B30; --muted-stripe: #F5F5F7;
            /* Shadows */
            --shadow-sm: none;
            --shadow-md: none;
            --shadow-glow-recording: 0 0 20px rgba(255,59,48,0.25), 0 0 8px rgba(255,59,48,0.15);
            /* Animation */
            --duration-fast: 120ms; --duration-normal: 200ms; --duration-slow: 350ms;
            --ease-out: cubic-bezier(0.25,0.46,0.45,0.94);
            --ease-spring: cubic-bezier(0.34,1.56,0.64,1);
            /* Layout */
            --key-row-height: 48px;
            --safe-area-bottom: env(safe-area-inset-bottom, 0px);
            --z-sticky: 50; --z-backdrop: 99; --z-sheet: 100;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, system-ui, sans-serif;
            background: var(--surface-base);
            color: var(--text-primary);
            height: 100vh; height: 100dvh;
            display: flex; flex-direction: column;
            padding: var(--space-xs);
            padding-top: env(safe-area-inset-top, 8px);
            padding-bottom: env(safe-area-inset-bottom, 8px);
            touch-action: none; user-select: none; overflow: hidden;
            -webkit-font-smoothing: antialiased;
        }
        /* Controls — two-row layout */
        .controls { flex-shrink: 0; display: flex; flex-direction: column; align-items: center; gap: var(--space-sm); padding: var(--space-xs) 0; }
        .controls__primary { display: flex; align-items: center; justify-content: center; gap: var(--space-lg); }
        .controls__secondary { display: flex; align-items: center; justify-content: center; gap: var(--space-md); }
        /* Button base — aspect-ratio: 1 guarantees no ovals */
        .ctrl-btn {
            display: inline-flex; align-items: center; justify-content: center;
            border: none; cursor: pointer; position: relative; overflow: hidden;
            transition: background-color var(--duration-fast) var(--ease-out),
                        transform var(--duration-normal) var(--ease-spring),
                        box-shadow var(--duration-normal) var(--ease-out);
            font-family: inherit; font-weight: var(--fw-semibold);
            letter-spacing: var(--ls-caps); color: var(--text-primary);
            aspect-ratio: 1; flex-shrink: 0;
        }
        /* --- Mic Button --- */
        .mic-btn-wrapper { position: relative; display: flex; flex-direction: column; align-items: center; gap: var(--space-2xs); }
        .mic-btn {
            width: 80px; height: 80px; border-radius: 50%; background: var(--surface-overlay);
            border: none; cursor: pointer; position: relative; display: flex; align-items: center; justify-content: center;
            color: var(--text-primary); -webkit-tap-highlight-color: transparent; touch-action: none; z-index: 1;
            transition: transform var(--duration-normal) var(--ease-spring),
                        background-color var(--duration-fast) var(--ease-out),
                        box-shadow var(--duration-normal) var(--ease-out);
        }
        .mic-btn__icon { display: flex; align-items: center; justify-content: center; pointer-events: none; }
        .mic-btn__pulse { position: absolute; inset: -4px; border-radius: 50%; pointer-events: none; opacity: 0; }
        .mic-btn__hint { font-size: var(--fs-xs); color: var(--text-tertiary); letter-spacing: var(--ls-wide); text-align: center; min-height: 1.2em; }
        .mic-btn.holding { background: var(--recording); transform: scale(1.05); box-shadow: var(--shadow-glow-recording); }
        .mic-btn.holding .mic-btn__pulse { opacity: 1; animation: mic-pulse-ring 1.5s var(--ease-out) infinite; }
        .mic-btn.holding::before { content: '\2191'; position: absolute; top: -24px; font-size: var(--fs-sm); color: var(--text-tertiary); opacity: 0; animation: slide-hint 1.5s ease-in-out infinite; pointer-events: none; }
        .mic-btn.locked { background: var(--recording); box-shadow: var(--shadow-glow-recording); }
        .mic-btn.locked .mic-btn__pulse { opacity: 1; animation: mic-pulse-ring 1.5s var(--ease-out) infinite; }
        .mic-btn.locked::after { content: ''; position: absolute; inset: -3px; border-radius: 50%; border: 2.5px solid var(--recording); animation: locked-ring-pulse 2s ease-in-out infinite; pointer-events: none; }
        .mic-btn.waiting { background: var(--surface-highest); transform: scale(0.93); }
        .mic-btn.processing { background: var(--surface-highest); pointer-events: none; opacity: 0.7; }
        @keyframes mic-pulse-ring { 0% { box-shadow: 0 0 0 0 rgba(255,59,48,0.45); } 70% { box-shadow: 0 0 0 14px rgba(255,59,48,0); } 100% { box-shadow: 0 0 0 0 rgba(255,59,48,0); } }
        @keyframes locked-ring-pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
        @keyframes slide-hint { 0%,100% { opacity: 0; transform: translateY(4px); } 50% { opacity: 0.6; transform: translateY(-2px); } }
        .ctrl-btn--secondary {
            width: var(--btn-secondary); height: var(--btn-secondary);
            border-radius: 50%; background: var(--surface-overlay);
            font-size: var(--fs-xs); box-shadow: var(--shadow-sm);
        }
        .ctrl-btn--secondary:active { transform: scale(0.88); background: var(--surface-highest); }
        .ctrl-btn--icon-only { background: transparent; box-shadow: none; color: var(--text-secondary); }
        .ctrl-btn--icon-only:active { background: var(--surface-overlay); color: var(--text-primary); }
        /* Settings gear rotation */
        #settingsBtn .ctrl-btn__icon { transition: transform var(--duration-slow) var(--ease-spring); display: inline-block; }
        #settingsBtn.open .ctrl-btn__icon { transform: rotate(90deg); }
        .ctrl-btn__label { font-size: inherit; pointer-events: none; line-height: 1; }
        .ctrl-btn__icon { font-size: 1.25rem; pointer-events: none; line-height: 1; }
        .ctrl-btn:focus-visible, .key-btn:focus-visible, .btn-done:focus-visible, .seg-control__btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
        .ctrl-btn:disabled { opacity: 0.25; cursor: not-allowed; }
        #centerBtn:disabled { background: var(--surface-overlay); opacity: 0.25; }
        #centerBtn:not(:disabled) { background: var(--surface-overlay); color: var(--accent); border: 1.5px solid var(--accent-muted); }
        @keyframes center-pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
        #centerBtn.sampling { animation: center-pulse 0.5s ease-in-out infinite; }
        #centerBtn.set { color: var(--accent); }
        /* Mic mute — CSS strikethrough, icon never changes */
        #muteBtn { position: relative; overflow: visible; }
        #muteBtn::after {
            content: ''; position: absolute; top: 50%; left: 50%;
            width: 110%; height: 2.5px; background: var(--muted-stripe);
            border-radius: 2px; transform: translate(-50%,-50%) rotate(-45deg) scaleX(0);
            transition: transform var(--duration-normal) var(--ease-spring),
                        opacity var(--duration-fast) var(--ease-out);
            opacity: 0; z-index: 1; pointer-events: none;
        }
        #muteBtn.muted { background: var(--muted-bg); box-shadow: 0 0 12px rgba(255,59,48,0.3); }
        #muteBtn.muted::after { transform: translate(-50%,-50%) rotate(-45deg) scaleX(1); opacity: 1; }
        /* Mic mute bounce */
        @keyframes mute-bounce { 0% { transform: scale(1); } 40% { transform: scale(0.85); } 70% { transform: scale(1.08); } 100% { transform: scale(1); } }
        #muteBtn.mute-anim { animation: mute-bounce 0.3s var(--ease-spring); }
        /* Status & Result */
        #status {
            text-align: center; font-size: var(--fs-sm); font-weight: var(--fw-medium);
            letter-spacing: var(--ls-wide); color: var(--text-secondary);
            padding: var(--space-xs) var(--space-sm); background: var(--surface-raised);
            border-radius: var(--space-xs); margin: var(--space-2xs) 0; flex-shrink: 0;
            transition: color var(--duration-normal) var(--ease-out),
                        background-color var(--duration-normal) var(--ease-out);
            border: 1px solid var(--border-subtle);
        }
        #status.recording { color: var(--recording); background: var(--recording-muted); border-color: rgba(255,59,48,0.15); font-size: var(--fs-base); font-weight: 600; }
        #status.paused { color: #CCAA33; background: rgba(204,170,51,0.1); border-color: rgba(204,170,51,0.12); }
        #result {
            text-align: center; font-size: var(--fs-base); color: var(--text-primary);
            padding: var(--space-xs) var(--space-sm); max-height: 48px;
            background: var(--surface-raised); border-radius: var(--space-xs);
            border: 1px solid var(--border-subtle); overflow-y: auto; flex-shrink: 0;
            scrollbar-width: thin; scrollbar-color: var(--surface-highest) transparent;
        }
        #result:empty { display: none; }
        #result.partial { color: var(--text-tertiary); font-style: italic; }
        /* Segmented control — iOS-style with sliding indicator */
        .seg-control { flex-shrink: 0; padding: var(--space-2xs) var(--space-lg); margin: var(--space-2xs) 0; }
        .seg-control__track {
            display: flex; position: relative; background: var(--surface-raised);
            border-radius: var(--space-xs); padding: var(--space-3xs);
            border: 1px solid var(--border-subtle);
        }
        .seg-control__indicator {
            position: absolute; top: var(--space-3xs); left: var(--space-3xs);
            width: calc(50% - var(--space-3xs)); height: calc(100% - var(--space-3xs) * 2);
            background: var(--surface-overlay); border-radius: calc(var(--space-xs) - 2px);
            transition: transform var(--duration-slow) var(--ease-spring);
            box-shadow: var(--shadow-sm); z-index: 0;
        }
        .seg-control__indicator.right { transform: translateX(100%); }
        .seg-control__btn {
            flex: 1; padding: var(--space-xs) 0; border: none; background: transparent;
            color: var(--text-tertiary); font-family: inherit; font-size: var(--fs-xs);
            font-weight: var(--fw-semibold); letter-spacing: var(--ls-caps);
            cursor: pointer; position: relative; z-index: 1;
            transition: color var(--duration-normal) var(--ease-out),
                        transform var(--duration-normal) var(--ease-spring);
            border-radius: calc(var(--space-xs) - 2px);
        }
        .seg-control__btn:active { transform: scale(0.93); }
        .seg-control__btn--active { color: var(--text-primary); }
        /* Pad */
        #pad {
            flex: 1; min-height: 120px; background: var(--surface-raised);
            border-radius: var(--space-sm); margin: var(--space-xs) 0;
            margin-bottom: calc(var(--key-row-height) + var(--space-sm) + var(--safe-area-bottom));
            position: relative; display: flex; align-items: center; justify-content: center;
            border: 1px solid var(--border-subtle);
            transition: border-color var(--duration-normal) var(--ease-out),
                        background-color var(--duration-normal) var(--ease-out);
        }
        #pad.touching { border-color: var(--border-medium); background: #161618; }
        #pad::after { content: attr(data-label); color: var(--text-tertiary); font-size: var(--fs-lg); font-weight: var(--fw-semibold); letter-spacing: var(--ls-caps); pointer-events: none; opacity: 0.4; }
        /* Key row */
        .key-row {
            display: flex; gap: var(--space-xs); padding: var(--space-xs) var(--space-sm);
            padding-bottom: calc(var(--space-xs) + var(--safe-area-bottom));
            justify-content: center; position: fixed; bottom: 0; left: 0; right: 0;
            background: var(--surface-base); z-index: var(--z-sticky);
            border-top: 1px solid var(--border-subtle);
        }
        .key-btn {
            flex: 1; max-width: 160px; height: var(--key-row-height);
            border: none; border-radius: var(--space-xs);
            background: var(--surface-overlay); color: var(--text-primary);
            font-family: inherit; font-size: var(--fs-base); font-weight: var(--fw-medium);
            cursor: pointer;
            transition: background-color var(--duration-fast) var(--ease-out),
                        transform var(--duration-normal) var(--ease-spring);
        }
        .key-btn:active { background: var(--surface-highest); transform: scale(0.94); }
        .key-btn.active { background: rgba(52,199,89,0.08); }
        /* Settings sheet */
        .settings-backdrop {
            position: fixed; inset: 0; background: rgba(0,0,0,0.5);
            backdrop-filter: blur(4px); -webkit-backdrop-filter: blur(4px);
            z-index: var(--z-backdrop); opacity: 0; pointer-events: none;
            transition: opacity var(--duration-slow) var(--ease-out);
        }
        .settings-backdrop.open { opacity: 1; pointer-events: auto; }
        .settings-sheet {
            position: fixed; bottom: 0; left: 0; width: 100%; max-height: 70vh;
            background: var(--surface-sheet); border-radius: var(--space-md) var(--space-md) 0 0;
            padding: var(--space-sm) var(--space-md);
            padding-bottom: calc(var(--space-md) + var(--safe-area-bottom));
            z-index: var(--z-sheet); transform: translateY(100%);
            transition: transform var(--duration-slow) var(--ease-spring);
            overflow-y: auto; border-top: 1px solid var(--border-medium);
        }
        .settings-sheet.open { transform: translateY(0); }
        .sheet-handle { width: 36px; height: 4px; background: var(--text-tertiary); border-radius: 2px; margin: 0 auto var(--space-md); opacity: 0.6; }
        .setting-row { display: flex; align-items: center; justify-content: space-between; padding: var(--space-sm) 0; border-bottom: 1px solid var(--border-subtle); gap: var(--space-sm); }
        .setting-row:last-of-type { border-bottom: none; }
        .setting-row label { font-size: var(--fs-base); font-weight: var(--fw-medium); color: var(--text-primary); flex-shrink: 0; }
        .setting-row input[type=range] { width: 140px; height: 4px; -webkit-appearance: none; appearance: none; background: var(--surface-highest); border-radius: 2px; outline: none; }
        .setting-row input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 22px; height: 22px; border-radius: 50%; background: var(--text-primary); cursor: pointer; box-shadow: var(--shadow-md); transition: transform var(--duration-normal) var(--ease-spring), box-shadow var(--duration-normal) var(--ease-out); }
        .setting-row input[type=range]:active::-webkit-slider-thumb { transform: scale(1.25); box-shadow: 0 0 12px rgba(245,245,247,0.25); }
        .setting-row input[type=range]::-moz-range-track { height: 4px; background: var(--surface-highest); border-radius: 2px; border: none; }
        .setting-row input[type=range]::-moz-range-thumb { width: 22px; height: 22px; border-radius: 50%; background: var(--text-primary); cursor: pointer; box-shadow: var(--shadow-md); border: none; }
        .setting-row span { width: 40px; text-align: right; font-size: var(--fs-sm); color: var(--text-secondary); font-variant-numeric: tabular-nums; }
        .setting-row select { padding: var(--space-xs) var(--space-sm); border-radius: var(--space-xs); border: 1px solid var(--border-medium); background: var(--surface-highest); color: var(--text-primary); font-family: inherit; font-size: var(--fs-sm); -webkit-appearance: none; appearance: none; transition: border-color var(--duration-normal) var(--ease-out), box-shadow var(--duration-normal) var(--ease-out); outline: none; }
        .setting-row select:focus { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-muted); }
        .btn-done { width: 100%; padding: var(--space-sm); margin-top: var(--space-md); background: var(--surface-overlay); border: 1px solid var(--border-medium); border-radius: var(--space-xs); color: var(--text-primary); font-family: inherit; font-size: var(--fs-md); font-weight: var(--fw-semibold); letter-spacing: var(--ls-wide); cursor: pointer; transition: background-color var(--duration-fast) var(--ease-out), transform var(--duration-normal) var(--ease-spring); }
        .btn-done:active { background: var(--surface-highest); transform: scale(0.96); }
        .btn-done--flush { margin-top: 0; }
        /* Calibration status */
        .calib-status { font-size: var(--fs-sm); color: var(--text-tertiary); margin-top: var(--space-2xs); text-align: center; transition: color var(--duration-normal) var(--ease-out); }
        .calib-status.set { color: var(--accent); }
        /* Recording pulse animation */
        @keyframes pulse-ring { 0% { box-shadow: 0 0 0 0 rgba(255,59,48,0.4); } 70% { box-shadow: 0 0 0 10px rgba(255,59,48,0); } 100% { box-shadow: 0 0 0 0 rgba(255,59,48,0); } }
        @media (prefers-reduced-motion: reduce) { .mic-btn.holding .mic-btn__pulse, .mic-btn.locked .mic-btn__pulse, .mic-btn.locked::after, .mic-btn.holding::before, #muteBtn.mute-anim, #centerBtn.sampling { animation: none; } .mic-btn.waiting { transform: none; } }
        /* Welcome overlay */
        .welcome { position: fixed; inset: 0; z-index: 200; background: var(--surface-base); display: flex; flex-direction: column; align-items: center; justify-content: center; gap: var(--space-lg); padding: var(--space-xl); text-align: center; transition: opacity var(--duration-slow) var(--ease-out); }
        .welcome.fade-out { opacity: 0; pointer-events: none; }
        .welcome h1 { font-size: 1.8rem; font-weight: var(--fw-semibold); letter-spacing: var(--ls-wide); color: var(--text-primary); }
        .welcome p { font-size: var(--fs-base); color: var(--text-secondary); line-height: 1.6; max-width: 280px; }
        .welcome button { margin-top: var(--space-md); padding: var(--space-xs) var(--space-xl); font-size: var(--fs-base); font-weight: var(--fw-semibold); color: #fff; background: var(--accent); border: none; border-radius: 999px; cursor: pointer; letter-spacing: var(--ls-wide); transition: transform var(--duration-fast) var(--ease-spring), opacity var(--duration-fast); }
        .welcome button:active { transform: scale(0.95); opacity: 0.85; }
        /* Screen mirror overlay */
        .screen-overlay { position: fixed; inset: 0; z-index: 250; background: #000; display: none; flex-direction: column; }
        .screen-overlay.active { display: flex; }
        .screen-overlay canvas { flex: 1; width: 100%; height: 100%; object-fit: contain; }
        .screen-overlay__bar { position: absolute; top: 0; left: 0; right: 0; display: flex; align-items: center; justify-content: space-between; padding: var(--space-xs) var(--space-sm); background: rgba(0,0,0,0.6); z-index: 251; }
        .screen-overlay__bar button { background: var(--surface-overlay); color: var(--text-primary); border: none; border-radius: 8px; padding: var(--space-2xs) var(--space-sm); font-size: var(--fs-sm); cursor: pointer; }
        .screen-overlay__bar span { color: var(--text-secondary); font-size: var(--fs-xs); font-family: monospace; }
    </style>
</head>
<body>
    <div class="controls">
        <div class="controls__primary">
            <div class="mic-btn-wrapper">
                <button class="mic-btn" id="micBtn" aria-label="Record audio">
                    <span class="mic-btn__pulse"></span>
                    <span class="mic-btn__icon mic-btn__icon--mic">
                        <svg viewBox="0 0 24 24" width="32" height="32" fill="currentColor"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.48 6-3.3 6-6.72h-1.7z"/></svg>
                    </span>
                    <span class="mic-btn__icon mic-btn__icon--lock" style="display:none">
                        <svg viewBox="0 0 24 24" width="28" height="28" fill="currentColor"><path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm3.1-9H8.9V6c0-1.71 1.39-3.1 3.1-3.1s3.1 1.39 3.1 3.1v2z"/></svg>
                    </span>
                </button>
                <span class="mic-btn__hint" id="micHint"></span>
            </div>
        </div>
        <div class="controls__secondary">
            <button class="ctrl-btn ctrl-btn--secondary" id="muteBtn" aria-label="Mute microphone">
                <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor" style="pointer-events:none"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.48 6-3.3 6-6.72h-1.7z"/></svg>
            </button>
            <button class="ctrl-btn ctrl-btn--secondary" id="centerBtn" disabled aria-label="Set gyro center">
                <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" style="pointer-events:none"><circle cx="12" cy="12" r="3"/><line x1="12" y1="2" x2="12" y2="7"/><line x1="12" y1="17" x2="12" y2="22"/><line x1="2" y1="12" x2="7" y2="12"/><line x1="17" y1="12" x2="22" y2="12"/></svg>
            </button>
            <button class="ctrl-btn ctrl-btn--secondary" id="screenBtn" aria-label="Screen mirror">
                <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor" style="pointer-events:none"><path d="M21 2H3c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h7v2H8v2h8v-2h-2v-2h7c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H3V4h18v12z"/></svg>
            </button>
            <button class="ctrl-btn ctrl-btn--secondary ctrl-btn--icon-only" id="settingsBtn" aria-label="Settings">
                <span class="ctrl-btn__icon"><svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58a.49.49 0 00.12-.61l-1.92-3.32a.49.49 0 00-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54a.48.48 0 00-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96a.49.49 0 00-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.07.62-.07.94s.02.64.07.94l-2.03 1.58a.49.49 0 00-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6A3.6 3.6 0 1112 8.4a3.6 3.6 0 010 7.2z"/></svg></span>
            </button>
        </div>
    </div>
    <div id="status" aria-live="polite">Loading...</div>
    <div id="result" aria-live="polite"></div>
    <div class="seg-control" id="modeToggle" role="tablist" aria-label="Input mode">
        <div class="seg-control__track">
            <div class="seg-control__indicator" id="segIndicator"></div>
            <button class="seg-control__btn seg-control__btn--active" id="modeTrackpad" role="tab" aria-selected="true">TRACKPAD</button>
            <button class="seg-control__btn" id="modeGyro" role="tab" aria-selected="false">GYROSCOPE</button>
        </div>
    </div>
    <div id="pad" data-label="TRACKPAD"></div>
    <div class="key-row">
        <button class="key-btn" id="backspaceBtn" aria-label="Backspace">⌫ Backspace</button>
        <button class="key-btn" id="enterBtn" aria-label="Enter">Enter ↵</button>
    </div>
    <div class="settings-backdrop" id="settingsBackdrop"></div>
    <div class="settings-sheet" id="settingsSheet" role="dialog" aria-modal="true" aria-label="Settings">
        <div class="sheet-handle"></div>
        <div class="setting-row">
            <label>Trackpad Speed</label>
            <input type="range" min="0.5" max="4" step="0.25" value="1.5" id="sensitivitySlider">
            <span id="sensitivityVal">1.5x</span>
        </div>
        <!-- Input Mode selector removed: use main screen segmented control -->
        <div id="gyroSettings" style="display: none;">
            <div class="setting-row">
                <label>Gyro Sensitivity</label>
                <input type="range" min="1" max="10" step="0.5" value="3" id="gyroSensitivity">
                <span id="gyroSensVal">3x</span>
            </div>
            <div class="setting-row">
                <button id="calibrateBtn" class="btn-done btn-done--flush">
                    Set Center
                </button>
            </div>
            <div id="calibStatus" class="calib-status">
                Not calibrated
            </div>
            <div class="setting-row" style="margin-top: var(--space-xs);">
                <label for="autoSwitchChk" style="flex:1; cursor:pointer;">Auto-switch to Trackpad on mouse</label>
                <input type="checkbox" id="autoSwitchChk" checked style="width:18px; height:18px; accent-color:var(--accent); cursor:pointer;">
            </div>
        </div>
        <button class="btn-done" id="settingsDone">Done</button>
    </div>

    <div class="screen-overlay" id="screenOverlay">
        <div class="screen-overlay__bar">
            <button id="screenBackBtn">Back</button>
            <select id="screenRes" style="background:var(--surface-overlay);color:var(--text-primary);border:none;border-radius:8px;padding:4px 8px;font-size:var(--fs-xs);">
                <option value="">Native</option>
                <option value="1080p" selected>1080p</option>
                <option value="720p">720p</option>
                <option value="480p">480p</option>
            </select>
            <span id="screenFps">--</span>
        </div>
        <canvas id="screenCanvas"></canvas>
    </div>

    <div class="welcome" id="welcome">
        <h1>Sanketra</h1>
        <p>Turn your phone into a wireless mic &amp; trackpad. Speak to type. Swipe to move cursor.</p>
        <button id="welcomeBtn">Get Started</button>
    </div>

    <script>
        // Auth token from URL (passed via QR code)
        const _wsToken = new URLSearchParams(location.search).get('token') || '';

        // Welcome overlay
        const welcomeEl = document.getElementById('welcome');
        const dismissWelcome = () => { welcomeEl.classList.add('fade-out'); setTimeout(() => welcomeEl.style.display = 'none', 350); };
        document.getElementById('welcomeBtn').addEventListener('click', dismissWelcome);
        welcomeEl.addEventListener('click', (e) => { if (e.target === welcomeEl) dismissWelcome(); });

        // Protocol constants
        const MSG_AUDIO_FRAME = 0x01;
        const MSG_START_SESSION = 0x10;
        const MSG_END_SESSION = 0x11;
        const MSG_FINAL = 0x02;
        const MSG_VAD_STATUS = 0x03;
        const MSG_PARTIAL = 0x04;
        const MSG_BACKPRESSURE = 0xFE;
        const MSG_SESSION_READY = 0xF0;
        const MSG_SESSION_DONE = 0xF1;
        const MSG_INPUT_PAUSED = 0xF2;
        const MSG_INPUT_RESUMED = 0xF3;
        const MSG_ERROR = 0xFF;

        const pad = document.getElementById('pad');
        const micBtn = document.getElementById('micBtn');
        const micHint = document.getElementById('micHint');
        const micIconMic = micBtn.querySelector('.mic-btn__icon--mic');
        const micIconLock = micBtn.querySelector('.mic-btn__icon--lock');
        const muteBtn = document.getElementById('muteBtn');
        const status = document.getElementById('status');
        const result = document.getElementById('result');
        const settingsBtn = document.getElementById('settingsBtn');
        const centerBtn = document.getElementById('centerBtn');
        const modeTrackpadBtn = document.getElementById('modeTrackpad');
        const modeGyroBtn = document.getElementById('modeGyro');
        const settingsSheet = document.getElementById('settingsSheet');
        const sensitivitySlider = document.getElementById('sensitivitySlider');
        const sensitivityVal = document.getElementById('sensitivityVal');
        const settingsDone = document.getElementById('settingsDone');

        // ===== SETTINGS =====
        let sensitivity = parseFloat(localStorage.getItem('tpSensitivity') || '1.5');
        sensitivitySlider.value = sensitivity;
        sensitivityVal.textContent = sensitivity + 'x';

        const settingsBackdrop = document.getElementById('settingsBackdrop');
        function openSettings() { haptic(); settingsSheet.classList.add('open'); settingsBackdrop.classList.add('open'); settingsBtn.classList.add('open'); settingsDone.focus(); }
        function closeSettings() { haptic(); settingsSheet.classList.remove('open'); settingsBackdrop.classList.remove('open'); settingsBtn.classList.remove('open'); settingsBtn.focus(); }
        settingsBtn.onclick = openSettings;
        settingsDone.onclick = closeSettings;
        settingsBackdrop.onclick = closeSettings;
        // Focus trap inside settings dialog
        settingsSheet.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') { closeSettings(); return; }
            if (e.key !== 'Tab') return;
            const focusable = settingsSheet.querySelectorAll('button, input, select, [tabindex]:not([tabindex="-1"])');
            if (!focusable.length) return;
            const first = focusable[0], last = focusable[focusable.length - 1];
            if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
            else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
        });

        // Center button — set gyro center (debounced 600ms)
        let _centerDebounce = 0;
        centerBtn.ontouchstart = async (e) => {
            e.preventDefault();
            if (inputMode !== 'gyro') return;
            const now = Date.now();
            if (now - _centerDebounce < 600) return;
            _centerDebounce = now;
            const hasPermission = await requestGyroPermission();
            if (!hasPermission) return;
            haptic();
            centerBtn.classList.add('sampling');
            await fetchScreenSize();
            const samples = [];
            const interval = setInterval(() => samples.push({...currentOrientation}), 50);
            setTimeout(() => {
                clearInterval(interval);
                if (samples.length === 0) samples.push({...currentOrientation});
                gyroCenter = {
                    alpha: samples.reduce((s, o) => s + o.alpha, 0) / samples.length,
                    beta: samples.reduce((s, o) => s + o.beta, 0) / samples.length,
                };
                localStorage.setItem('gyroCenter', JSON.stringify(gyroCenter));
                window.gyroLastPos = {x: screenSize.w / 2, y: screenSize.h / 2};
                centerBtn.classList.remove('sampling');
                centerBtn.classList.add('set');
                updateCalibStatus();
                status.textContent = 'Center set!';
            }, 500);
        };
        sensitivitySlider.oninput = () => {
            sensitivity = parseFloat(sensitivitySlider.value);
            sensitivityVal.textContent = sensitivity + 'x';
            localStorage.setItem('tpSensitivity', sensitivity);
        };

        // ===== GYROSCOPE MODE =====
        let inputMode = localStorage.getItem('inputMode') || 'trackpad';
        let gyroSensitivity = parseFloat(localStorage.getItem('gyroSensitivity') || '3');
        let gyroCenter = JSON.parse(localStorage.getItem('gyroCenter') || 'null');
        let gyroAutoSwitch = localStorage.getItem('gyroAutoSwitch') !== 'false';
        const autoSwitchChk = document.getElementById('autoSwitchChk');
        autoSwitchChk.checked = gyroAutoSwitch;
        autoSwitchChk.addEventListener('change', () => { gyroAutoSwitch = autoSwitchChk.checked; localStorage.setItem('gyroAutoSwitch', gyroAutoSwitch); });
        // Discard old center format (had gamma, needs alpha)
        if (gyroCenter && gyroCenter.alpha === undefined) {
            gyroCenter = null;
            localStorage.removeItem('gyroCenter');
        }
        let screenSize = {w: 1920, h: 1080};
        let currentOrientation = {alpha: 0, beta: 0, gamma: 0};
        let lastGyroUpdate = 0;
        const GYRO_UPDATE_INTERVAL = 16; // ~60fps

        const inputModeSelect = document.getElementById('inputMode');
        const gyroSettings = document.getElementById('gyroSettings');
        const gyroSensSlider = document.getElementById('gyroSensitivity');
        const gyroSensVal = document.getElementById('gyroSensVal');
        const calibrateBtn = document.getElementById('calibrateBtn');
        let edgeHitStart = 0;

        async function fetchScreenSize() {
            try {
                const resp = await fetch('/api/screen', {
                    headers: {'Authorization': 'Bearer ' + _wsToken}
                });
                const data = await resp.json();
                screenSize = {w: data.w, h: data.h};
            } catch (e) {
                console.warn('Could not fetch screen size, using 1920x1080');
            }
        }

        // Initialize UI
        if (inputModeSelect) inputModeSelect.value = inputMode;
        gyroSensSlider.value = gyroSensitivity;
        gyroSensVal.textContent = gyroSensitivity + 'x';
        gyroSettings.style.display = (inputMode === 'gyro') ? 'block' : 'none';
        updateUIMode();
        if (inputMode === 'gyro') { fetchScreenSize(); requestGyroPermission(); }

        // Lock to portrait (prevents chaos in gyro mode)
        try { screen.orientation.lock('portrait').catch(() => {}); } catch(e) {}

        // Mode switching — settings dropdown and main toggle both use switchMode()
        if (inputModeSelect) inputModeSelect.onchange = () => { switchMode(inputModeSelect.value); };
        modeTrackpadBtn.onclick = () => { haptic(); switchMode('trackpad'); };
        modeGyroBtn.onclick = () => { haptic(); switchMode('gyro'); };

        gyroSensSlider.oninput = () => {
            gyroSensitivity = parseFloat(gyroSensSlider.value);
            gyroSensVal.textContent = gyroSensitivity + 'x';
            localStorage.setItem('gyroSensitivity', gyroSensitivity);
        };

        function updateUIMode() {
            const isGyro = inputMode === 'gyro';
            centerBtn.disabled = !isGyro;
            // Segmented control — slide indicator pill
            const indicator = document.getElementById('segIndicator');
            if (indicator) indicator.classList.toggle('right', isGyro);
            modeTrackpadBtn.classList.toggle('seg-control__btn--active', !isGyro);
            modeGyroBtn.classList.toggle('seg-control__btn--active', isGyro);
            modeTrackpadBtn.setAttribute('aria-selected', String(!isGyro));
            modeGyroBtn.setAttribute('aria-selected', String(isGyro));
            pad.dataset.label = isGyro ? 'GYRO POINTER' : 'TRACKPAD';
        }

        function switchMode(newMode) {
            if (inputMode === newMode) return;
            // Force-cleanup stuck gesture state
            _cancelBufferedClick();
            if (scrolling) { scrolling = false; sendTp({t: 'ss'}); }
            if (momentumRaf) { cancelAnimationFrame(momentumRaf); momentumRaf = null; }
            if (dragging) { dragging = false; sendTp({t: 'd', a: 'up'}); }
            moved = false;
            fingerCount = 0;
            totalDist = 0;
            scrollVelocity = 0;
            lastScrollTime = 0;

            inputMode = newMode;
            localStorage.setItem('inputMode', inputMode);
            if (inputModeSelect) inputModeSelect.value = inputMode;
            gyroSettings.style.display = (inputMode === 'gyro') ? 'block' : 'none';
            updateUIMode();
            if (inputMode === 'gyro') {
                requestGyroPermission();
                fetchScreenSize();
                updateCalibStatus();
            } else {
                if (!tpConnected) connectTrackpad();
            }
        }

        // iOS Permission Handling
        async function requestGyroPermission() {
            if (!window.DeviceOrientationEvent) {
                status.textContent = 'Gyroscope not supported';
                return false;
            }
            if (typeof DeviceOrientationEvent.requestPermission === 'function') {
                try {
                    const permission = await DeviceOrientationEvent.requestPermission();
                    if (permission === 'granted') {
                        window.addEventListener('deviceorientation', handleDeviceOrientation);
                        console.log('Gyro permission granted');
                        return true;
                    } else {
                        status.textContent = 'Gyro permission denied';
                        return false;
                    }
                } catch (err) {
                    console.error('Permission error:', err);
                    status.textContent = 'Gyro permission error';
                    return false;
                }
            } else {
                window.addEventListener('deviceorientation', handleDeviceOrientation);
                console.log('Gyro activated');
                return true;
            }
        }

        function handleDeviceOrientation(event) {
            currentOrientation = {
                alpha: event.alpha || 0,
                beta: event.beta || 0,
                gamma: event.gamma || 0
            };
            if (inputMode === 'gyro' && gyroCenter && tpConnected) {
                updateCursorFromGyro();
            }
        }

        // Set Center — capture current phone orientation as screen center
        let _calibDebounce = 0;
        calibrateBtn.onclick = async () => {
            const now = Date.now();
            if (now - _calibDebounce < 600) return;
            _calibDebounce = now;
            const hasPermission = await requestGyroPermission();
            if (!hasPermission) return;
            haptic();
            calibrateBtn.textContent = 'Sampling...';
            calibrateBtn.style.opacity = '0.5';
            await fetchScreenSize();
            const samples = [];
            const interval = setInterval(() => samples.push({...currentOrientation}), 50);
            setTimeout(() => {
                clearInterval(interval);
                if (samples.length === 0) samples.push({...currentOrientation});
                gyroCenter = {
                    alpha: samples.reduce((s, o) => s + o.alpha, 0) / samples.length,
                    beta: samples.reduce((s, o) => s + o.beta, 0) / samples.length,
                };
                localStorage.setItem('gyroCenter', JSON.stringify(gyroCenter));
                window.gyroLastPos = {x: screenSize.w / 2, y: screenSize.h / 2};
                calibrateBtn.textContent = 'Set Center';
                calibrateBtn.style.opacity = '1';
                updateCalibStatus();
                status.textContent = 'Center set!';
            }, 500);
        };

        // Helper functions
        function normalizeAngle(angle) {
            while (angle > 180) angle -= 360;
            while (angle < -180) angle += 360;
            return angle;
        }

        function updateCursorFromGyro() {
            const now = Date.now();
            if (now - lastGyroUpdate < GYRO_UPDATE_INTERVAL) return;
            lastGyroUpdate = now;

            // Direct mapping: orientation delta → screen pixels
            // alpha (yaw) → X axis, beta (pitch) → Y axis
            const pxPerDeg = gyroSensitivity * 20;
            let da = currentOrientation.alpha - gyroCenter.alpha;
            // Handle alpha wrapping (0↔360 boundary)
            if (da > 180) da -= 360;
            if (da < -180) da += 360;
            const db = currentOrientation.beta - gyroCenter.beta;

            let x = screenSize.w / 2 + da * pxPerDeg;
            let y = screenSize.h / 2 + db * pxPerDeg;

            // Clamp to screen bounds
            x = Math.max(0, Math.min(screenSize.w, x));
            y = Math.max(0, Math.min(screenSize.h, y));

            // Adaptive smoothing — fast move = responsive, still = stable
            if (!window.gyroLastPos) window.gyroLastPos = {x, y};
            const dx = x - window.gyroLastPos.x;
            const dy = y - window.gyroLastPos.y;
            const speed = Math.sqrt(dx * dx + dy * dy);
            const alpha = Math.min(0.85, Math.max(0.3, speed / 60));
            const smoothX = (1 - alpha) * window.gyroLastPos.x + alpha * x;
            const smoothY = (1 - alpha) * window.gyroLastPos.y + alpha * y;
            window.gyroLastPos = {x: smoothX, y: smoothY};

            // Send ABSOLUTE position
            if (tpConnected) {
                sendTp({t: 'a', x: Math.round(smoothX), y: Math.round(smoothY)});
            }

            // Auto-recenter: if stuck at edge for >2s
            checkEdgeRecenter(smoothX, smoothY, screenSize.w, screenSize.h);
        }

        function checkEdgeRecenter(x, y, sw, sh) {
            const margin = 20;
            const atEdge = x <= margin || x >= sw - margin || y <= margin || y >= sh - margin;
            if (atEdge) {
                if (!edgeHitStart) edgeHitStart = Date.now();
                if (Date.now() - edgeHitStart > 2000) {
                    // Auto-recenter: re-capture current orientation as center
                    gyroCenter = {
                        alpha: currentOrientation.alpha,
                        beta: currentOrientation.beta,
                    };
                    localStorage.setItem('gyroCenter', JSON.stringify(gyroCenter));
                    window.gyroLastPos = {x: sw / 2, y: sh / 2};
                    sendTp({t: 'rc'});
                    edgeHitStart = 0;
                }
            } else {
                edgeHitStart = 0;
            }
        }

        function updateCalibStatus() {
            const statusEl = document.getElementById('calibStatus');
            if (gyroCenter) {
                statusEl.textContent = 'Center set — point and move';
                statusEl.classList.add('set');
            } else {
                statusEl.textContent = 'Point at screen center, then Set Center';
                statusEl.classList.remove('set');
            }
        }


        // ===== TRACKPAD =====
        let tpWs = null;
        let lastX = 0, lastY = 0;
        let tpConnected = false;
        let touchStart = 0;
        let moved = false;
        let fingerCount = 0;
        let lastTap = 0;
        let dragging = false;
        let scrolling = false;
        let scrollAccum = 0;
        let scrollRaf = null;
        // Momentum scrolling state
        let scrollVelocity = 0;
        let lastScrollTime = 0;
        let momentumRaf = null;
        const SCROLL_FRICTION = 0.95;      // per-frame damping (time constant ~325ms)
        const SCROLL_MIN_VELOCITY = 0.3;   // stop threshold (sub-pixel)

        let _tpReconnectTimer = null;
        function connectTrackpad() {
            // Skip only if WS is already open or connecting
            if (tpWs && (tpWs.readyState === WebSocket.OPEN || tpWs.readyState === WebSocket.CONNECTING)) return;
            // Clean up stale reference (CLOSING or CLOSED state)
            if (tpWs) { tpWs = null; tpConnected = false; }
            tpWs = new WebSocket('wss://' + location.host + '/ws-tp');
            tpWs.onopen = () => {
                tpWs.send(JSON.stringify({t:'auth',token:_wsToken}));
                tpConnected = true; console.log('Trackpad connected');
            };
            tpWs.onclose = (e) => {
                tpWs = null; tpConnected = false;
                if (e.code === 4401) {
                    console.warn('Trackpad WS: session expired (4401)');
                    status.textContent = 'Session expired \u2014 refresh page';
                    return;
                }
                // Auto-reconnect after 500ms if not already scheduled
                if (!_tpReconnectTimer) {
                    _tpReconnectTimer = setTimeout(() => { _tpReconnectTimer = null; connectTrackpad(); }, 500);
                }
            };
            tpWs.onerror = () => { console.error('Trackpad WS error'); };
            tpWs.binaryType = 'arraybuffer';
            tpWs.onmessage = (e) => {
                const data = new Uint8Array(e.data);
                const type = data[0];
                if (type === MSG_INPUT_PAUSED) {
                    if (gyroAutoSwitch && inputMode === 'gyro') switchMode('trackpad');
                    status.textContent = 'Paused \u2014 laptop input active';
                    status.classList.add('paused');
                    status.classList.remove('recording');
                } else if (type === MSG_INPUT_RESUMED) {
                    status.textContent = sessionActive ? 'Listening...' : 'Ready';
                    status.classList.remove('paused');
                    if (sessionActive) status.classList.add('recording');
                }
            };
        }

        function sendTp(obj) {
            if (tpConnected && tpWs?.readyState === 1) {
                tpWs.send(JSON.stringify(obj));
            } else {
                // WS dead — trigger reconnect and queue the message for retry
                connectTrackpad();
            }
        }

        connectTrackpad();

        // Movement threshold (px) — finger jitter below this doesn't count as "moved"
        const MOVE_THRESHOLD = 8;
        const LONG_PRESS_MS = 300;
        const DOUBLE_TAP_WINDOW = 300;  // ms to wait before sending buffered click
        let startX = 0, startY = 0;
        let totalDist = 0;
        let longPressTimer = null;
        let longPressFired = false;
        // Double-tap-drag buffering (P1-10): buffer click on first tap,
        // cancel it if second tap arrives within DOUBLE_TAP_WINDOW to start drag instead.
        let _bufferedClickTimer = null;
        let _bufferedClickBtn = 1;

        function _flushBufferedClick() {
            if (_bufferedClickTimer) {
                clearTimeout(_bufferedClickTimer);
                _bufferedClickTimer = null;
                sendTp({t: 'c', b: _bufferedClickBtn});
            }
        }

        function _cancelBufferedClick() {
            if (_bufferedClickTimer) {
                clearTimeout(_bufferedClickTimer);
                _bufferedClickTimer = null;
            }
        }

        function onTouchStart(e) {
            e.preventDefault();
            e.currentTarget.classList.add('touching');
            if (!tpConnected) connectTrackpad();
            const now = Date.now();
            fingerCount = e.touches.length;
            lastX = e.touches[0].clientX;
            lastY = e.touches[0].clientY;

            if (fingerCount === e.changedTouches.length) {
                if (momentumRaf) { cancelAnimationFrame(momentumRaf); momentumRaf = null; sendTp({t: 'ss'}); }
                touchStart = now;
                moved = false;
                totalDist = 0;
                longPressFired = false;
                startX = lastX;
                startY = lastY;

                // Double-tap-drag: if there's a buffered click pending and second tap arrives,
                // cancel the buffered click and start drag instead (no click sent to desktop)
                if (fingerCount === 1 && _bufferedClickTimer && !dragging) {
                    _cancelBufferedClick();
                    dragging = true;
                    sendTp({t: 'd', a: 'down'});
                }
                lastTap = now;

                // Long press = right click (single finger, no drag)
                if (longPressTimer) clearTimeout(longPressTimer);
                if (fingerCount === 1 && !dragging) {
                    longPressTimer = setTimeout(() => {
                        longPressTimer = null;
                        if (!moved && !dragging && !scrolling) {
                            longPressFired = true;
                            _cancelBufferedClick();
                            sendTp({t: 'c', b: 3});
                            if (typeof haptic === 'function') haptic();
                        }
                    }, LONG_PRESS_MS);
                }
            }
        }
        pad.addEventListener('touchstart', onTouchStart, {passive: false});

        function onTouchMove(e) {
            e.preventDefault();
            fingerCount = e.touches.length;
            const t = e.touches[0];
            const dx = t.clientX - lastX;
            const dy = t.clientY - lastY;
            lastX = t.clientX;
            lastY = t.clientY;
            totalDist += Math.abs(dx) + Math.abs(dy);

            if (totalDist > MOVE_THRESHOLD) {
                moved = true;
                if (longPressTimer) { clearTimeout(longPressTimer); longPressTimer = null; }
            }

            if (fingerCount >= 2 && !dragging) {
                scrolling = true;
                if (momentumRaf) { cancelAnimationFrame(momentumRaf); momentumRaf = null; }
                const now = performance.now();
                const scrollDy = dy * sensitivity;
                if (lastScrollTime > 0) {
                    const dt = now - lastScrollTime;
                    if (dt > 0 && dt < 100) scrollVelocity = scrollDy / dt * 16.67;
                }
                lastScrollTime = now;
                scrollAccum += scrollDy;
                if (!scrollRaf) {
                    scrollRaf = requestAnimationFrame(() => {
                        scrollRaf = null;
                        if (scrollAccum !== 0) {
                            sendTp({t: 's', dy: scrollAccum});
                            scrollAccum = 0;
                        }
                    });
                }
            } else if (fingerCount === 1 && inputMode !== 'gyro') {
                sendTp({t: 'm', x: dx * sensitivity, y: dy * sensitivity});
            }
        }
        pad.addEventListener('touchmove', onTouchMove, {passive: false});

        function onTouchEnd(e) {
            e.preventDefault();
            const remaining = e.touches.length;
            if (remaining > 0) return;
            e.currentTarget.classList.remove('touching');

            if (scrolling) {
                scrolling = false;
                if (scrollRaf) { cancelAnimationFrame(scrollRaf); scrollRaf = null; }
                if (scrollAccum !== 0) sendTp({t: 's', dy: scrollAccum});
                scrollAccum = 0;
                if (Math.abs(scrollVelocity) > SCROLL_MIN_VELOCITY) {
                    let vel = scrollVelocity;
                    const momentumTick = () => {
                        vel *= SCROLL_FRICTION;
                        if (Math.abs(vel) < SCROLL_MIN_VELOCITY) { momentumRaf = null; sendTp({t: 'ss'}); return; }
                        sendTp({t: 's', dy: vel});
                        momentumRaf = requestAnimationFrame(momentumTick);
                    };
                    momentumRaf = requestAnimationFrame(momentumTick);
                } else {
                    sendTp({t: 'ss'});
                }
                scrollVelocity = 0;
                lastScrollTime = 0;
            }
            if (longPressTimer) { clearTimeout(longPressTimer); longPressTimer = null; }
            if (dragging) { dragging = false; sendTp({t: 'd', a: 'up'}); }
            else if (!longPressFired && !moved && Date.now() - touchStart < 300) {
                // P1-10: Buffer the click for DOUBLE_TAP_WINDOW ms.
                // If a second tap arrives within the window, it becomes a drag (no click sent).
                // If no second tap, the buffered click fires after the delay.
                const btn = fingerCount >= 2 ? 3 : 1;
                _cancelBufferedClick();
                _bufferedClickBtn = btn;
                _bufferedClickTimer = setTimeout(() => {
                    _bufferedClickTimer = null;
                    sendTp({t: 'c', b: _bufferedClickBtn});
                }, DOUBLE_TAP_WINDOW);
            }
            fingerCount = 0;
            longPressFired = false;
        }
        function onTouchCancel(e) {
            e.preventDefault();
            e.currentTarget.classList.remove('touching');
            if (longPressTimer) { clearTimeout(longPressTimer); longPressTimer = null; }
            _cancelBufferedClick();
            if (scrolling) { scrolling = false; if (scrollRaf) { cancelAnimationFrame(scrollRaf); scrollRaf = null; } if (momentumRaf) { cancelAnimationFrame(momentumRaf); momentumRaf = null; } scrollAccum = 0; scrollVelocity = 0; lastScrollTime = 0; sendTp({t: 'ss'}); }
            if (dragging) { dragging = false; sendTp({t: 'd', a: 'up'}); }
            moved = false; fingerCount = 0; totalDist = 0;
        }
        pad.addEventListener('touchend', onTouchEnd, {passive: false});
        pad.addEventListener('touchcancel', onTouchCancel, {passive: false});

        // Attach same touch handlers to screen canvas (trackpad works on screen mirror too)
        const screenCanvas = document.getElementById('screenCanvas');
        if (screenCanvas) {
            screenCanvas.addEventListener('touchstart', onTouchStart, {passive: false});
            screenCanvas.addEventListener('touchmove', onTouchMove, {passive: false});
            screenCanvas.addEventListener('touchend', onTouchEnd, {passive: false});
            screenCanvas.addEventListener('touchcancel', onTouchCancel, {passive: false});
        }

        // ===== HAPTIC FEEDBACK =====
        // Audio click (both) + vibration (Android)
        let clickCtx = null;

        const playClick = () => {
            try {
                if (!clickCtx) clickCtx = new (window.AudioContext || window.webkitAudioContext)();
                if (clickCtx.state === 'suspended') clickCtx.resume();
                const osc = clickCtx.createOscillator();
                const gain = clickCtx.createGain();
                osc.connect(gain);
                gain.connect(clickCtx.destination);
                osc.frequency.value = 1800;
                gain.gain.setValueAtTime(0.15, clickCtx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, clickCtx.currentTime + 0.05);
                osc.start(clickCtx.currentTime);
                osc.stop(clickCtx.currentTime + 0.05);
            } catch(e) {}
        };

        const haptic = () => {
            try { navigator.vibrate && navigator.vibrate(30); } catch(e) {}
            playClick();
        };

        // Micro haptic for repeat actions (softer vibration + quieter tick)
        const microHaptic = () => {
            try { navigator.vibrate && navigator.vibrate(10); } catch(e) {}
            try {
                if (!clickCtx) clickCtx = new (window.AudioContext || window.webkitAudioContext)();
                if (clickCtx.state === 'suspended') clickCtx.resume();
                const osc = clickCtx.createOscillator();
                const gain = clickCtx.createGain();
                osc.connect(gain);
                gain.connect(clickCtx.destination);
                osc.frequency.value = 1800;
                gain.gain.setValueAtTime(0.15, clickCtx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, clickCtx.currentTime + 0.02);
                osc.start(clickCtx.currentTime);
                osc.stop(clickCtx.currentTime + 0.02);
            } catch(e) {}
        };

        // ===== KEY BUTTONS =====
        const enterBtn = document.getElementById('enterBtn');
        const bsBtn = document.getElementById('backspaceBtn');

        // Enter button with glow + haptic
        enterBtn.addEventListener('touchstart', e => { e.preventDefault(); haptic(); enterBtn.classList.add('active'); sendTp({t: 'k', k: 'enter'}); });
        enterBtn.addEventListener('touchend', () => enterBtn.classList.remove('active'));
        enterBtn.addEventListener('touchcancel', () => enterBtn.classList.remove('active'));
        enterBtn.addEventListener('mousedown', () => { haptic(); enterBtn.classList.add('active'); sendTp({t: 'k', k: 'enter'}); });
        enterBtn.addEventListener('mouseup', () => enterBtn.classList.remove('active'));
        enterBtn.addEventListener('mouseleave', () => enterBtn.classList.remove('active'));

        // Backspace with hold-to-repeat, glow + haptic
        let bsInterval = null;
        let bsTimeout = null;
        const startBackspace = () => {
            haptic();
            bsBtn.classList.add('active');
            sendTp({t: 'k', k: 'backspace'});
            // Wait 300ms before starting repeat with micro haptic
            bsTimeout = setTimeout(() => {
                bsInterval = setInterval(() => { microHaptic(); sendTp({t: 'k', k: 'backspace'}); }, 40);
            }, 300);
        };
        const stopBackspace = () => {
            bsBtn.classList.remove('active');
            if (bsTimeout) { clearTimeout(bsTimeout); bsTimeout = null; }
            if (bsInterval) { clearInterval(bsInterval); bsInterval = null; }
        };
        bsBtn.addEventListener('touchstart', e => { e.preventDefault(); startBackspace(); });
        bsBtn.addEventListener('touchend', stopBackspace);
        bsBtn.addEventListener('touchcancel', stopBackspace);
        bsBtn.addEventListener('mousedown', startBackspace);
        bsBtn.addEventListener('mouseup', stopBackspace);
        bsBtn.addEventListener('mouseleave', stopBackspace);

        // ===== STREAMING AUDIO =====
        let audioWs = null;
        let audioContext = null;
        let workletNode = null;
        let sourceNode = null;
        let mediaStream = null;
        let sessionActive = false;
        let isStreaming = false;
        let micState = 'idle';
        let micLongPressTimer = null;
        let micTouchStartY = 0;
        let micTouchId = null;
        const MIC_LONG_PRESS_MS = 300;
        const SLIDE_UP_PX = 60;
        let paused = false;
        let isMuted = false;
        let micReady = false;

        // WebSocket reconnection state
        let reconnectAttempts = 0;
        let maxReconnectAttempts = 5;
        let reconnectDelays = [1000, 2000, 5000, 10000, 10000]; // Exponential backoff
        let isReconnecting = false;

        function setMicState(newState) {
            micState = newState;
            micBtn.classList.remove('waiting', 'holding', 'locked', 'processing');
            micIconMic.style.display = 'none';
            micIconLock.style.display = 'none';
            switch (newState) {
                case 'idle':
                    micIconMic.style.display = 'flex';
                    micHint.textContent = '';
                    micBtn.disabled = false;
                    break;
                case 'waiting':
                    micBtn.classList.add('waiting');
                    micIconMic.style.display = 'flex';
                    micHint.textContent = '';
                    break;
                case 'holding':
                    micBtn.classList.add('holding');
                    micIconMic.style.display = 'flex';
                    micHint.textContent = 'Slide up to lock';
                    break;
                case 'locked':
                    micBtn.classList.add('locked');
                    micIconLock.style.display = 'flex';
                    micHint.textContent = 'Tap to stop';
                    break;
                case 'processing':
                    micBtn.classList.add('processing');
                    micIconMic.style.display = 'flex';
                    micHint.textContent = '';
                    micBtn.disabled = true;
                    break;
            }
        }

        const processorCode = `
            class PCMProcessor extends AudioWorkletProcessor {
                constructor() {
                    super();
                    this.ratio = sampleRate / 16000;
                    this.buffer = new Float32Array(512);
                    this.idx = 0;
                    this.resamplePos = 0;
                }
                flush() {
                    const int16 = new Int16Array(512);
                    for (let j = 0; j < 512; j++) {
                        const s = Math.max(-1, Math.min(1, this.buffer[j]));
                        int16[j] = s < 0 ? s * 32768 : s * 32767;
                    }
                    this.port.postMessage(int16.buffer, [int16.buffer]);
                    this.buffer = new Float32Array(512);
                    this.idx = 0;
                }
                process(inputs) {
                    const input = inputs[0]?.[0];
                    if (!input) return true;
                    if (this.ratio <= 1.01) {
                        for (let i = 0; i < input.length; i++) {
                            this.buffer[this.idx++] = input[i];
                            if (this.idx >= 512) this.flush();
                        }
                    } else {
                        for (let i = 0; i < input.length; i++) {
                            this.resamplePos += 16000;
                            if (this.resamplePos >= sampleRate) {
                                this.resamplePos -= sampleRate;
                                this.buffer[this.idx++] = input[i];
                                if (this.idx >= 512) this.flush();
                            }
                        }
                    }
                    return true;
                }
            }
            registerProcessor('pcm-processor', PCMProcessor);
        `;

        async function initMic() {
            try {
                status.textContent = 'Requesting mic...';
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true  // Normalize volume for quiet/loud speakers
                    }
                });
                try { audioContext = new AudioContext({ sampleRate: 16000 }); }
                catch (_) { audioContext = new AudioContext(); }
                console.log('AudioContext sampleRate:', audioContext.sampleRate);
                const blob = new Blob([processorCode], { type: 'application/javascript' });
                await audioContext.audioWorklet.addModule(URL.createObjectURL(blob));
                micReady = true;
                console.log('Mic acquired');
            } catch (err) {
                console.error('Mic error:', err);
                status.textContent = 'Mic denied: ' + err.message;
                throw err;  // Propagate error to startStreaming
            }
        }

        function connectAudioWs() {
            return new Promise((resolve, reject) => {
                audioWs = new WebSocket('wss://' + location.host + '/ws-audio-stream');
                audioWs.binaryType = 'arraybuffer';

                audioWs.onopen = () => {
                    const tokenBytes = new TextEncoder().encode(_wsToken);
                    const authMsg = new Uint8Array(1 + tokenBytes.length);
                    authMsg[0] = 0xFA;
                    authMsg.set(tokenBytes, 1);
                    audioWs.send(authMsg.buffer);
                    console.log('Audio WS connected');
                    resolve();
                };

                audioWs.onmessage = (e) => {
                    const data = new Uint8Array(e.data);
                    const type = data[0];
                    const payload = new TextDecoder().decode(data.slice(1));

                    if (type === MSG_SESSION_READY) {
                        sessionActive = true;
                        status.textContent = 'Listening...';
                        startAudioCapture();
                    } else if (type === MSG_SESSION_DONE) {
                        sessionActive = false;
                        status.textContent = 'Done';
                        cleanup();
                    } else if (type === MSG_PARTIAL) {
                        result.textContent = payload;
                        result.classList.add('partial');
                    } else if (type === MSG_FINAL) {
                        result.textContent = payload;
                        result.classList.remove('partial');
                        if (sessionActive) status.textContent = 'Listening...';
                    } else if (type === MSG_VAD_STATUS) {
                        if (sessionActive) status.textContent = data[1] === 0x01 ? 'Streaming...' : 'Listening...';
                    } else if (type === MSG_BACKPRESSURE) {
                        paused = true;
                        setTimeout(() => { paused = false; }, 500);
                    } else if (type === MSG_INPUT_PAUSED) {
                        if (gyroAutoSwitch && inputMode === 'gyro') switchMode('trackpad');
                        status.textContent = 'Paused \u2014 laptop input active';
                        status.classList.add('paused');
                        status.classList.remove('recording');
                    } else if (type === MSG_INPUT_RESUMED) {
                            status.textContent = sessionActive ? 'Listening...' : 'Ready';
                        status.classList.remove('paused');
                        if (sessionActive) status.classList.add('recording');
                    } else if (type === MSG_ERROR) {
                        status.textContent = 'Error: ' + payload;
                        console.error('Server error:', payload);
                    }
                };

                audioWs.onclose = (e) => {
                    console.log('Audio WS closed:', e.code);

                    if (e.code === 4401) {
                        console.warn('Audio WS: session expired (4401)');
                        status.textContent = 'Session expired \u2014 refresh page';
                        cleanup();
                        return;
                    }

                    // Auto-reconnect on unexpected disconnection
                    if (e.code !== 1000 && isStreaming && !isReconnecting && reconnectAttempts < maxReconnectAttempts) {
                        isReconnecting = true;
                        const delay = reconnectDelays[reconnectAttempts];
                        reconnectAttempts++;
                        status.textContent = `Reconnecting (${reconnectAttempts}/${maxReconnectAttempts})...`;
                        console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts}/${maxReconnectAttempts})`);

                        setTimeout(async () => {
                            try {
                                await connectAudioWs();
                                console.log('Reconnected successfully');
                                reconnectAttempts = 0; // Reset on success
                                isReconnecting = false;

                                // Resume streaming if still in streaming mode
                                if (isStreaming && audioWs?.readyState === WebSocket.OPEN) {
                                    audioWs.send(new Uint8Array([MSG_START_SESSION]));
                                }
                            } catch (err) {
                                console.error('Reconnection failed:', err);
                                isReconnecting = false;

                                if (reconnectAttempts >= maxReconnectAttempts) {
                                    status.textContent = 'Connection lost - please refresh';
                                    cleanup();
                                }
                            }
                        }, delay);
                    } else if (e.code !== 1000 && isStreaming) {
                        status.textContent = 'Disconnected';
                        cleanup();
                    }
                };

                audioWs.onerror = (err) => {
                    console.error('Audio WS error:', err);
                    reject(err);
                };
            });
        }

        function startAudioCapture() {
            if (audioContext.state === 'suspended') audioContext.resume();

            workletNode = new AudioWorkletNode(audioContext, 'pcm-processor');
            workletNode.port.onmessage = (e) => {
                if (!sessionActive || paused || isMuted) return;
                if (audioWs?.readyState === WebSocket.OPEN) {
                    const frame = new Uint8Array(1 + e.data.byteLength);
                    frame[0] = MSG_AUDIO_FRAME;
                    frame.set(new Uint8Array(e.data), 1);
                    audioWs.send(frame);
                }
            };

            sourceNode = audioContext.createMediaStreamSource(mediaStream);
            sourceNode.connect(workletNode);
        }

        async function startStreaming() {
            if (isStreaming) return;
            isStreaming = true;
            result.textContent = '';

            try {
                // Request mic access only when starting recording
                if (!mediaStream) {
                    status.textContent = 'Requesting mic...';
                    await initMic();
                }

                await connectAudioWs();
                audioWs.send(new Uint8Array([MSG_START_SESSION]));
                status.textContent = 'Starting...';
                status.classList.add('recording');

            } catch (err) {
                status.textContent = 'Connection failed';
                isStreaming = false;
                setMicState('idle');
            }
        }

        function stopStreaming() {
            if (!isStreaming) return;

            if (audioWs?.readyState === WebSocket.OPEN && sessionActive) {
                status.textContent = 'Processing...';
                audioWs.send(new Uint8Array([MSG_END_SESSION]));
                // DO NOT close here - wait for SESSION_DONE
            } else {
                cleanup();
            }
        }

        function cleanup() {
            isStreaming = false;
            sessionActive = false;

            if (workletNode) { workletNode.disconnect(); workletNode = null; }
            if (sourceNode) { sourceNode.disconnect(); sourceNode = null; }
            if (audioWs) { audioWs.close(); audioWs = null; }

            // Release mic - stops the indicator on phone
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
                micReady = false;
                console.log('Mic released');
            }

            if (audioContext && audioContext.state !== 'closed') {
                audioContext.close();
                audioContext = null;
            }

            setMicState('idle');
            status.textContent = 'Ready';
            status.classList.remove('recording');
        }

        // ===== MIC BUTTON — Unified gesture handler =====
        function cancelLongPress() {
            if (micLongPressTimer) { clearTimeout(micLongPressTimer); micLongPressTimer = null; }
        }

        micBtn.addEventListener('touchstart', (e) => {
            e.preventDefault();
            if (micState === 'processing') return;
            if (micState === 'locked') {
                haptic();
                stopStreaming();
                setMicState('processing');
                status.textContent = 'Processing...';
                status.classList.remove('recording');
                return;
            }
            if (micState === 'idle') {
                const touch = e.changedTouches[0];
                micTouchStartY = touch.clientY;
                micTouchId = touch.identifier;
                setMicState('waiting');
                micLongPressTimer = setTimeout(async () => {
                    micLongPressTimer = null;
                    haptic();
                    setMicState('holding');
                    result.textContent = '';
                    status.classList.add('recording');
                    await startStreaming();
                }, MIC_LONG_PRESS_MS);
            }
        }, { passive: false });

        micBtn.addEventListener('touchmove', (e) => {
            if (micState !== 'holding' && micState !== 'waiting') return;
            const touch = Array.from(e.changedTouches).find(t => t.identifier === micTouchId);
            if (!touch) return;
            const deltaY = micTouchStartY - touch.clientY;
            if (micState === 'waiting' && Math.abs(deltaY) > 20) {
                cancelLongPress();
                setMicState('idle');
                return;
            }
            if (micState === 'holding' && deltaY > SLIDE_UP_PX) {
                haptic();
                setMicState('locked');
                micTouchId = null;
            }
        }, { passive: false });

        micBtn.addEventListener('touchend', (e) => {
            const touch = Array.from(e.changedTouches).find(t => t.identifier === micTouchId);
            if (micState === 'waiting') {
                cancelLongPress();
                setMicState('idle');
                return;
            }
            if (micState === 'holding' && touch) {
                stopStreaming();
                setMicState('processing');
                status.textContent = 'Processing...';
                status.classList.remove('recording');
                micTouchId = null;
                return;
            }
        }, { passive: false });

        micBtn.addEventListener('touchcancel', () => {
            cancelLongPress();
            if (micState === 'waiting') {
                setMicState('idle');
            } else if (micState === 'holding') {
                stopStreaming();
                setMicState('processing');
                status.textContent = 'Processing...';
                status.classList.remove('recording');
            }
            micTouchId = null;
        });

        // Mouse fallback for desktop testing
        let mouseIsDown = false;
        micBtn.addEventListener('mousedown', (e) => {
            e.preventDefault();
            if (micState === 'processing') return;
            if (micState === 'locked') {
                haptic();
                stopStreaming();
                setMicState('processing');
                status.textContent = 'Processing...';
                status.classList.remove('recording');
                return;
            }
            if (micState === 'idle') {
                mouseIsDown = true;
                micTouchStartY = e.clientY;
                setMicState('waiting');
                micLongPressTimer = setTimeout(async () => {
                    micLongPressTimer = null;
                    if (!mouseIsDown) return;
                    haptic();
                    setMicState('holding');
                    result.textContent = '';
                    status.classList.add('recording');
                    await startStreaming();
                }, MIC_LONG_PRESS_MS);
            }
        });
        document.addEventListener('mousemove', (e) => {
            if (!mouseIsDown || micState !== 'holding') return;
            const deltaY = micTouchStartY - e.clientY;
            if (deltaY > SLIDE_UP_PX) {
                haptic();
                setMicState('locked');
                mouseIsDown = false;
            }
        });
        document.addEventListener('mouseup', () => {
            if (!mouseIsDown) return;
            mouseIsDown = false;
            if (micState === 'waiting') {
                cancelLongPress();
                setMicState('idle');
            } else if (micState === 'holding') {
                stopStreaming();
                setMicState('processing');
                status.textContent = 'Processing...';
                status.classList.remove('recording');
            }
        });

        // MUTE button - actually turns mic on/off
        muteBtn.onclick = (e) => {
            e.preventDefault();
            haptic();
            isMuted = !isMuted;
            muteBtn.classList.toggle('muted', isMuted);
            // Bounce animation
            muteBtn.classList.remove('mute-anim');
            void muteBtn.offsetWidth; // reflow trigger for re-animation
            muteBtn.classList.add('mute-anim');
            if (sessionActive) status.textContent = isMuted ? 'Muted' : 'Listening...';

            // Actually disable/enable mic track
            if (mediaStream) {
                mediaStream.getAudioTracks().forEach(track => {
                    track.enabled = !isMuted;
                });
            }
        };
        muteBtn.addEventListener('animationend', () => muteBtn.classList.remove('mute-anim'));

        // ===== MOBILE LIFECYCLE MANAGEMENT =====

        // Handle background tab - suspend/resume AudioContext on iOS
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                // Tab went to background
                if (audioContext && audioContext.state === 'running') {
                    audioContext.suspend().then(() => {
                        console.log('AudioContext suspended (background)');
                    });
                }
            } else {
                // Tab came to foreground — reconnect trackpad WS if dead
                connectTrackpad();
                if (audioContext && audioContext.state === 'suspended') {
                    audioContext.resume().then(() => {
                        console.log('AudioContext resumed (foreground)');
                    });
                }
            }
        });

        // Handle network status - show offline/online feedback
        window.addEventListener('offline', () => {
            console.log('Network offline');
            status.textContent = 'Offline - check connection';
            // Don't cleanup, allow reconnection when back online
        });

        window.addEventListener('online', () => {
            console.log('Network online');
            // Reconnect trackpad WS immediately
            connectTrackpad();
            if (isStreaming && (!audioWs || audioWs.readyState !== WebSocket.OPEN)) {
                status.textContent = 'Reconnecting...';
                // Trigger reconnection attempt
                reconnectAttempts = 0;
                isReconnecting = false;
            } else {
                status.textContent = 'Ready';
            }
        });

        // iOS sample rate detection - log actual sample rate
        // (iOS often ignores requested 16kHz and uses 48kHz instead)
        // Our AudioWorklet already handles resampling to 512 samples
        // so this is mainly for debugging/logging
        if (navigator.mediaDevices && navigator.mediaDevices.getSupportedConstraints) {
            const constraints = navigator.mediaDevices.getSupportedConstraints();
            console.log('Supported audio constraints:', constraints);
        }

        // ===== SCREEN MIRROR =====
        const screenBtn = document.getElementById('screenBtn');
        const screenOverlay = document.getElementById('screenOverlay');
        const screenBackBtn = document.getElementById('screenBackBtn');
        const screenFpsEl = document.getElementById('screenFps');
        const screenCtx = screenCanvas ? screenCanvas.getContext('2d') : null;
        const screenResEl = document.getElementById('screenRes');
        if (screenResEl) screenResEl.value = localStorage.getItem('screenRes') || '1080p';
        let screenWs = null;
        let videoDecoder = null;
        let screenActive = false;
        let screenFrameCount = 0;
        let screenFpsTimer = null;

        function startScreenMirror() {
            if (screenActive) return;
            if (typeof VideoDecoder === 'undefined') {
                alert('WebCodecs not supported in this browser');
                return;
            }
            screenActive = true;
            screenOverlay.classList.add('active');

            videoDecoder = new VideoDecoder({
                output: (frame) => {
                    screenCanvas.width = frame.displayWidth;
                    screenCanvas.height = frame.displayHeight;
                    screenCtx.drawImage(frame, 0, 0);
                    frame.close();
                    screenFrameCount++;
                },
                error: (e) => { console.error('VideoDecoder error:', e); }
            });

            const resSelect = document.getElementById('screenRes');
            const resParam = resSelect ? resSelect.value : '';
            let screenUrl = 'wss://' + location.host + '/ws-screen';
            if (resParam) screenUrl += '?res=' + resParam;
            if (resSelect) localStorage.setItem('screenRes', resSelect.value);
            screenWs = new WebSocket(screenUrl);
            screenWs.binaryType = 'arraybuffer';

            screenWs.onopen = () => {
                // Send MSG_AUTH (0xFA + token)
                const tokenBytes = new TextEncoder().encode(_wsToken);
                const authMsg = new Uint8Array(1 + tokenBytes.length);
                authMsg[0] = 0xFA;
                authMsg.set(tokenBytes, 1);
                screenWs.send(authMsg.buffer);
            };

            let configured = false;

            screenWs.onmessage = (e) => {
                const data = new Uint8Array(e.data);
                if (data[0] === 0x20) {
                    // SCREEN_CONFIG: [0x20][width:2LE][height:2LE][fps:1]
                    const dv = new DataView(e.data);
                    const w = dv.getUint16(1, true);
                    const h = dv.getUint16(3, true);
                    console.log(`Screen: ${w}x${h}@${data[5]}fps`);
                    return;
                }

                // Frame: [flags:1][timestamp:4LE][h264_data:N]
                const flags = data[0];
                const isKey = !!(flags & 0x01);
                const timestamp = new DataView(e.data).getUint32(1, true);
                const h264Data = new Uint8Array(e.data, 5);

                if (!configured && isKey) {
                    // Configure decoder on first keyframe
                    // Extract codec string from SPS NAL
                    let codecStr = 'avc1.42001e'; // baseline profile level 3.0 default
                    // Find SPS in the data to build proper codec string
                    for (let i = 0; i < h264Data.length - 5; i++) {
                        if (h264Data[i] === 0 && h264Data[i+1] === 0 && h264Data[i+2] === 0 && h264Data[i+3] === 1) {
                            const nalType = h264Data[i+4] & 0x1F;
                            if (nalType === 7 && i + 7 < h264Data.length) {
                                // SPS found: profile_idc, constraint_flags, level_idc
                                const profile = h264Data[i+5].toString(16).padStart(2, '0');
                                const compat = h264Data[i+6].toString(16).padStart(2, '0');
                                const level = h264Data[i+7].toString(16).padStart(2, '0');
                                codecStr = `avc1.${profile}${compat}${level}`;
                                break;
                            }
                        }
                    }
                    try {
                        videoDecoder.configure({
                            codec: codecStr,
                            optimizeForLatency: true,
                        });
                        configured = true;
                        console.log('VideoDecoder configured:', codecStr);
                    } catch (err) {
                        console.error('Failed to configure decoder:', err);
                        return;
                    }
                }

                if (!configured) return; // Wait for first keyframe

                try {
                    videoDecoder.decode(new EncodedVideoChunk({
                        type: isKey ? 'key' : 'delta',
                        timestamp: timestamp * 1000, // microseconds
                        data: h264Data,
                    }));
                } catch (err) {
                    console.error('Decode error:', err);
                }
            };

            screenWs.onclose = (e) => {
                console.log('Screen WS closed:', e?.code);
                if (e?.code === 4401) {
                    console.warn('Screen WS: session expired (4401)');
                    status.textContent = 'Session expired \u2014 refresh page';
                }
                stopScreenMirror();
            };
            screenWs.onerror = () => { console.error('Screen WS error'); };

            // FPS counter
            screenFrameCount = 0;
            screenFpsTimer = setInterval(() => {
                screenFpsEl.textContent = screenFrameCount + ' fps';
                screenFrameCount = 0;
            }, 1000);
        }

        function stopScreenMirror() {
            screenActive = false;
            screenOverlay.classList.remove('active');
            if (screenWs && screenWs.readyState === WebSocket.OPEN) {
                screenWs.close(1000);
            }
            screenWs = null;
            if (videoDecoder && videoDecoder.state !== 'closed') {
                try { videoDecoder.close(); } catch(e) {}
            }
            videoDecoder = null;
            if (screenFpsTimer) {
                clearInterval(screenFpsTimer);
                screenFpsTimer = null;
            }
            screenFpsEl.textContent = '--';
        }

        screenBtn.addEventListener('click', () => { haptic(); startScreenMirror(); });
        screenBackBtn.addEventListener('click', () => { haptic(); stopScreenMirror(); });

        // Init - don't request mic until user starts recording
        status.textContent = 'Ready';
    </script>
</body>
</html>
'''

# Pre-encode HTML to bytes once at import time (skip 43KB UTF-8 encode per request)
_html_bytes = HTML_CONTENT.encode("utf-8")

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
                    os.system("systemctl --user daemon-reload")
                    log_info("Self-healed systemd TimeoutStopSec 3→10")
        except Exception:
            pass
    url = f"https://{ip}:{port}?token={AUTH_TOKEN}"

    # Show URL
    if not bg_mode:
        print(f"\n{'='*50}")
        print(f"Open: {url}")
        print(f"(Accept certificate warning on phone)")
        print(f"{'='*50}")
        print_qr(url, save_image=False)
        print()
    else:
        log_info(f"URL: {url}")

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
