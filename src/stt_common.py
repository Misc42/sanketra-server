#!/usr/bin/env python3
"""
Shared utilities for STT scripts
- Transliteration (Hindi/Urdu to Roman)
- GPU utilities with fallback
- Platform-specific typing/mouse
- Audio utilities
- Logging
"""

import sys
import os
import re
import subprocess
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import numpy as np

# ============================================================================
#                              LOGGING
# ============================================================================

_logger = None

def setup_logging(name="sanketra", log_dir=None):
    """
    Setup logging to both console and file.
    Log file: sanketra_YYYY-MM-DD.log
    """
    global _logger

    if _logger is not None:
        return _logger

    # Create logger
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers
    if _logger.handlers:
        return _logger

    # Log format
    fmt = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler (INFO and above)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    _logger.addHandler(console)

    # File handler (DEBUG and above)
    if log_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)  # Go up from src/ to root
        log_dir = os.path.join(root_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"sanketra_{datetime.now().strftime('%Y-%m-%d')}.log")

    try:
        file_handler = RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=3, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        _logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create log file: {e}")

    return _logger

def get_logger():
    """Get the logger instance"""
    global _logger
    if _logger is None:
        return setup_logging()
    return _logger

def log_info(msg): get_logger().info(msg)
def log_debug(msg): get_logger().debug(msg)
def log_warning(msg): get_logger().warning(msg)
def log_error(msg): get_logger().error(msg)

# ============================================================================
#                          APP INPUT PAUSE STATE
# ============================================================================

import threading as _threading
import time as _pause_time

# Import mark_app_output for pynput virtual event suppression (Win/Mac input guard)
try:
    from input_monitor import mark_app_output as _mark_app_output
except ImportError:
    def _mark_app_output():
        pass

_app_input_paused = False
_app_input_lock = _threading.Lock()
_pause_callbacks = []
_last_pause_time = 0.0
_PAUSE_COOLDOWN = 0.5  # seconds — don't allow resume until 500ms after last physical input

def pause_app_input():
    """Pause all app input dispatch — called by physical input monitor"""
    global _app_input_paused, _last_pause_time
    _last_pause_time = _pause_time.time()  # Refresh on every physical event
    with _app_input_lock:
        if _app_input_paused:
            return
        _app_input_paused = True
    log_info("[InputGuard] PAUSED - physical input detected")
    for cb in _pause_callbacks:
        try:
            cb(True)
        except Exception:
            pass

def resume_app_input(force=False):
    """Resume app input dispatch.

    Args:
        force: If True, bypass cooldown check. Use for discrete user actions
               (key press, click) where phone activity is unambiguous.
    """
    global _app_input_paused
    with _app_input_lock:
        if not _app_input_paused:
            return
        # Don't resume if physical input was recent (prevents flicker)
        # — unless force=True (discrete phone action like key press)
        if not force and _pause_time.time() - _last_pause_time < _PAUSE_COOLDOWN:
            return
        _app_input_paused = False
    log_info("[InputGuard] RESUMED - phone activity" + (" (forced)" if force else ""))
    for cb in _pause_callbacks:
        try:
            cb(False)
        except Exception:
            pass

def is_app_input_paused():
    """Check if app input is currently paused (lock-free read)"""
    return _app_input_paused

def register_pause_callback(cb):
    """Register callback for pause/resume state changes. cb(paused: bool)"""
    _pause_callbacks.append(cb)

# ============================================================================
#                           PLATFORM DETECTION
# ============================================================================

def get_platform():
    """Returns 'linux', 'windows', or 'macos'"""
    if sys.platform.startswith('linux'):
        return 'linux'
    elif sys.platform == 'win32':
        return 'windows'
    elif sys.platform == 'darwin':
        return 'macos'
    return 'unknown'

PLATFORM = get_platform()

def _detect_x11_display():
    """Auto-detect active X11 DISPLAY (validates existing, fixes broken ones)."""
    if PLATFORM != 'linux':
        return
    current = os.environ.get('DISPLAY', '')
    # Build probe list: try current first (if set), then common displays
    displays = []
    if current:
        displays.append(current)
    for d in [':0', ':1', ':2']:
        if d not in displays:
            displays.append(d)
    # Try each display
    for disp in displays:
        try:
            result = subprocess.run(
                ['xdpyinfo', '-display', disp],
                capture_output=True, timeout=2,
            )
            if result.returncode == 0:
                os.environ['DISPLAY'] = disp
                try: log_info(f"Auto-detected DISPLAY={disp}")
                except Exception: print(f"[stt_common] Auto-detected DISPLAY={disp}")
                return
        except Exception:
            continue
    # Fallback: parse from `w` command (logged-in user's display)
    try:
        result = subprocess.run(['w', '-hs'], capture_output=True, text=True, timeout=2)
        for line in result.stdout.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 3 and parts[2].startswith(':'):
                os.environ['DISPLAY'] = parts[2]
                try: log_info(f"Auto-detected DISPLAY={parts[2]} from w")
                except Exception: print(f"[stt_common] Auto-detected DISPLAY={parts[2]} from w")
                return
    except Exception:
        pass

# Lazy X11 display detection — runs on first use, not at import time
_x11_display_detected = False

def _ensure_x11_display():
    """Ensure DISPLAY env var is set. Runs detection once on first call."""
    global _x11_display_detected
    if _x11_display_detected:
        return
    _x11_display_detected = True
    _detect_x11_display()

def get_display_server():
    """
    Detect Linux display server: 'x11', 'wayland', or 'unknown'
    Returns 'n/a' on non-Linux platforms
    """
    if PLATFORM != 'linux':
        return 'n/a'
    _ensure_x11_display()

    # Check XDG_SESSION_TYPE (most reliable)
    session_type = os.environ.get('XDG_SESSION_TYPE', '').lower()
    if session_type in ('x11', 'wayland'):
        return session_type

    # Fallback: check for Wayland display
    if os.environ.get('WAYLAND_DISPLAY'):
        return 'wayland'

    # Fallback: check for X11 display
    if os.environ.get('DISPLAY'):
        return 'x11'

    return 'unknown'

def _check_uinput_access():
    """Check if we can access /dev/uinput directly (for old ydotool)"""
    import os
    try:
        return os.access('/dev/uinput', os.W_OK)
    except Exception:
        return False

def _get_linux_input_tool():
    """
    Get the appropriate input tool for Linux.
    Returns: ('xdotool', True), ('ydotool', True), ('ydotool-sudo', True), or (None, False)
    """
    import shutil
    display = get_display_server()

    # X11: prefer xdotool
    if display == 'x11':
        if shutil.which('xdotool'):
            return 'xdotool', True

    # Wayland: need ydotool
    if display == 'wayland':
        if shutil.which('ydotool'):
            # Old ydotool (0.x) needs direct uinput access or sudo
            # New ydotool (1.x) needs ydotoold daemon
            if _check_uinput_access():
                return 'ydotool', True
            else:
                # Need sudo for old ydotool without uinput perms
                return 'ydotool-sudo', True
        return None, False

    # Unknown: try xdotool first
    if shutil.which('xdotool'):
        return 'xdotool', True
    if shutil.which('ydotool'):
        return 'ydotool', True

    return None, False

# Cache the tool detection
_LINUX_INPUT_TOOL = None
_LINUX_INPUT_AVAILABLE = None

def get_linux_input_tool():
    """Get cached Linux input tool"""
    global _LINUX_INPUT_TOOL, _LINUX_INPUT_AVAILABLE
    if _LINUX_INPUT_TOOL is None:
        _LINUX_INPUT_TOOL, _LINUX_INPUT_AVAILABLE = _get_linux_input_tool()
    return _LINUX_INPUT_TOOL, _LINUX_INPUT_AVAILABLE

# ============================================================================
#                         STREAMING CONSTANTS
# ============================================================================

STREAMING_SAMPLE_RATE = 16000
STREAMING_CHUNK_SAMPLES = 512  # 32ms @ 16kHz - Silero VAD requirement
STREAMING_MIN_SPEECH_MS = 250
STREAMING_MIN_SILENCE_MS = 500
STREAMING_SILENCE_THRESHOLD = 16  # chunks for 500ms silence

# ============================================================================
#                              GPU UTILITIES
# ============================================================================

_gpu_handle = None
_gpu_available = False

def init_gpu():
    """Initialize GPU monitoring. Returns (handle, success)"""
    global _gpu_handle, _gpu_available
    try:
        import pynvml
        pynvml.nvmlInit()
        _gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        _gpu_available = True
        return _gpu_handle, True
    except Exception as e:
        print(f"GPU not available: {e}")
        _gpu_available = False
        return None, False

def get_gpu_stats():
    """Get GPU memory and utilization. Returns (used_gb, total_gb, util_percent)"""
    if not _gpu_available or _gpu_handle is None:
        return 0, 0, 0
    try:
        import pynvml
        mem = pynvml.nvmlDeviceGetMemoryInfo(_gpu_handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(_gpu_handle)
        return mem.used / (1024**3), mem.total / (1024**3), util.gpu
    except Exception:
        return 0, 0, 0

def cleanup_gpu():
    """Cleanup GPU resources"""
    global _gpu_handle, _gpu_available
    if _gpu_available:
        try:
            import pynvml
            pynvml.nvmlShutdown()
        except Exception:
            pass
    _gpu_handle = None
    _gpu_available = False

def get_gpu_compute_capability():
    """
    Get GPU compute capability (e.g., 7.5 for RTX 2080).
    Returns (major, minor) tuple, or (0, 0) if no GPU.

    float16 works well on compute capability >= 7.0 (Volta+)
    Older GPUs (Pascal 6.x, Maxwell 5.x) should use int8
    """
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return major, minor
    except Exception:
        pass
    return 0, 0

def gpu_supports_float16():
    """Check if GPU supports efficient float16 computation"""
    major, minor = get_gpu_compute_capability()
    # Compute capability >= 7.0 (Volta, Turing, Ampere, Ada, Hopper)
    return major >= 7

def get_model_download_size(model_name):
    """Get approximate download size for model"""
    sizes = {
        "tiny": "75 MB",
        "base": "150 MB",
        "small": "500 MB",
        "medium": "1.5 GB",
        "large-v3-turbo": "1.6 GB",
        "distil-large-v3": "1.5 GB",
        "large-v3": "3 GB",
    }
    return sizes.get(model_name, "unknown")

# ============================================================================
#                            MODEL LOADING
# ============================================================================

def load_whisper(model_name="large-v3", device="auto", compute_type="auto"):
    """
    Load Whisper model with automatic GPU/CPU fallback

    Args:
        model_name: Whisper model name (tiny, base, small, medium, large-v3, large-v3-turbo)
        device: "auto", "cuda", or "cpu"
        compute_type: "auto", "float16", "int8", "int8_float16"

    Returns:
        WhisperModel instance
    """
    import os
    import torch
    from faster_whisper import WhisperModel

    # Enable HuggingFace download progress bars
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

    # Auto-detect device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto-detect compute type based on GPU capability
    if compute_type == "auto":
        if device == "cuda":
            if gpu_supports_float16():
                compute_type = "float16"
            else:
                compute_type = "int8"  # Older GPUs use int8
                print(f"Note: GPU doesn't support efficient float16, using int8", flush=True)
        else:
            # Apple Silicon (arm64) supports float16 natively via CoreML/Accelerate
            import platform as _platform
            if _platform.machine() in ('arm64', 'aarch64'):
                compute_type = "float16"
            else:
                compute_type = "int8"

    print(f"Loading Whisper {model_name} on {device} ({compute_type})...", flush=True)
    print(f"(First run will download ~{get_model_download_size(model_name)} - please wait)", flush=True)

    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        print("Model loaded successfully", flush=True)
        return model
    except Exception as e:
        if device == "cuda":
            # Try fallback compute types on CUDA before going to CPU
            fallback_types = ["int8_float16", "int8"] if compute_type == "float16" else ["int8"]
            for fallback in fallback_types:
                if fallback == compute_type:
                    continue
                try:
                    print(f"Trying {fallback} on CUDA...", flush=True)
                    model = WhisperModel(model_name, device="cuda", compute_type=fallback)
                    print(f"Model loaded successfully with {fallback}", flush=True)
                    return model
                except Exception:
                    continue

            # Last resort: CPU
            print(f"CUDA failed, falling back to CPU...", flush=True)
            return WhisperModel(model_name, device="cpu", compute_type="int8")
        raise

# =============================================================================
#                         ENERGY-BASED VAD (FALLBACK)
# =============================================================================

class EnergyVAD:
    """Simple energy-based VAD using numpy only - fallback when Silero fails"""

    def __init__(self, threshold=0.02, min_speech_ms=250, min_silence_ms=500, sample_rate=16000):
        self.threshold = threshold
        self.min_speech_frames = int(min_speech_ms * sample_rate / 1000 / 512)  # 512 samples per frame
        self.min_silence_frames = int(min_silence_ms * sample_rate / 1000 / 512)
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speech = False

    def __call__(self, audio_chunk, sample_rate=16000):
        """Process audio chunk, return speech probability (0.0 or 1.0)"""
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))

        if energy > self.threshold:
            self.speech_frames += 1
            self.silence_frames = 0
            if self.speech_frames >= self.min_speech_frames:
                self.is_speech = True
        else:
            self.silence_frames += 1
            self.speech_frames = 0
            if self.silence_frames >= self.min_silence_frames:
                self.is_speech = False

        return 1.0 if self.is_speech else 0.0

    def reset_states(self):
        """Reset VAD state"""
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speech = False


# Flag to track VAD type
_VAD_TYPE = None  # "silero" or "energy"


def load_vad(force_energy=False):
    """
    Load VAD model with fallback to energy-based VAD.

    Returns:
        (model, utils) for Silero VAD, or
        (EnergyVAD(), None) for energy-based fallback
    """
    global _VAD_TYPE

    if force_energy:
        print("Using energy-based VAD (forced)...", flush=True)
        _VAD_TYPE = "energy"
        return EnergyVAD(), None

    try:
        import torch
        print("Loading Silero VAD...", flush=True)
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        _VAD_TYPE = "silero"
        return model, utils
    except Exception as e:
        print(f"Silero VAD failed: {e}", flush=True)
        print("Falling back to energy-based VAD...", flush=True)
        _VAD_TYPE = "energy"
        return EnergyVAD(), None


def get_vad_type():
    """Return current VAD type: 'silero' or 'energy'"""
    return _VAD_TYPE

# ============================================================================
#                         SMART MODEL SELECTION
# ============================================================================

# Model specifications — actual measured VRAM for faster-whisper (CTranslate2)
# These are runtime GPU memory footprint, NOT original OpenAI Whisper estimates.
MODEL_VRAM = {
    "tiny":             {"float16": 0.5, "int8_float16": 0.4, "int8": 0.3},
    "base":             {"float16": 0.5, "int8_float16": 0.4, "int8": 0.3},
    "small":            {"float16": 1.0, "int8_float16": 0.7, "int8": 0.5},
    "medium":           {"float16": 1.5, "int8_float16": 1.1, "int8": 0.8},
    "large-v3-turbo":   {"float16": 1.5, "int8_float16": 1.1, "int8": 0.8},
    "distil-large-v3":  {"float16": 1.5, "int8_float16": 1.1, "int8": 0.8},
    "large-v3":         {"float16": 3.0, "int8_float16": 2.1, "int8": 1.5},
}

MODEL_PARAMS = {
    "tiny": "39M", "base": "74M", "small": "244M",
    "medium": "769M", "large-v3-turbo": "809M",
    "distil-large-v3": "756M", "large-v3": "1.5B"
}

MODEL_LATENCY_GPU = {  # ms per 5s audio on RTX 3060+
    "tiny": 50, "base": 80, "small": 150,
    "medium": 300, "large-v3-turbo": 200,
    "distil-large-v3": 180, "large-v3": 500
}

MODEL_LATENCY_CPU = {  # ms per 5s audio on laptop i7/Ryzen 7 (int8, beam_size=5)
    "tiny": 400, "base": 700, "small": 1800,
    "medium": 4500, "large-v3-turbo": 3000,
    "distil-large-v3": 3000, "large-v3": 12000
}

MODEL_ACCURACY = {  # Approximate accuracy % on Hindi/English
    "tiny": 68, "base": 72, "small": 78,
    "medium": 83, "large-v3-turbo": 87,
    "distil-large-v3": 86, "large-v3": 89
}

MODEL_NOTES = {
    "tiny": "", "base": "", "small": "", "medium": "",
    "large-v3-turbo": "", "large-v3": "",
    "distil-large-v3": "English only",
}

INFERENCE_BUFFER_GB = 0.5  # Extra VRAM needed during inference

def get_vram_gb():
    """Get available GPU VRAM in GB, 0 if no GPU"""
    init_gpu()
    _, total, _ = get_gpu_stats()
    return total

def get_vram_free():
    """Get free GPU VRAM in GB"""
    init_gpu()
    used, total, _ = get_gpu_stats()
    return max(0, total - used)

def get_system_info():
    """Get comprehensive system info for display"""
    import platform as plat

    # GPU info
    gpu_name, vram_total, vram_used = "No GPU", 0, 0
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode('utf-8')
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_total = mem.total / (1024**3)
        vram_used = mem.used / (1024**3)
    except Exception:
        pass

    # CPU info — platform.processor() returns 'x86_64' on Linux, useless
    cpu_name = ""
    if PLATFORM == 'linux':
        try:
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if line.startswith('model name'):
                        cpu_name = line.split(':', 1)[1].strip()
                        break
        except Exception:
            pass
    elif PLATFORM == 'macos':
        try:
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                    capture_output=True, text=True, timeout=2)
            if result.returncode == 0 and result.stdout.strip():
                cpu_name = result.stdout.strip()
        except Exception:
            pass
    if not cpu_name:
        cpu_name = plat.processor() or "Unknown CPU"
    try:
        import psutil
        cpu_cores = psutil.cpu_count()
        ram_total = psutil.virtual_memory().total / (1024**3)
        ram_used = psutil.virtual_memory().used / (1024**3)
    except ImportError:
        cpu_cores = 0
        ram_total, ram_used = 0, 0

    return {
        "gpu_name": gpu_name,
        "vram_total": vram_total,
        "vram_used": vram_used,
        "vram_free": max(0, vram_total - vram_used),
        "cpu_name": cpu_name,
        "cpu_cores": cpu_cores,
        "ram_total": ram_total,
        "ram_used": ram_used,
        "ram_free": max(0, ram_total - ram_used),
    }

def is_model_downloaded(model_name):
    """Check if a faster-whisper model is already downloaded in HuggingFace cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        # faster-whisper models live under Systran/ org
        repo_id = f"Systran/faster-whisper-{model_name}"
        result = try_to_load_from_cache(repo_id, "model.bin")
        return result is not None
    except Exception:
        # Fallback: check common cache paths
        import pathlib
        cache_dirs = [
            pathlib.Path.home() / ".cache" / "huggingface" / "hub",
        ]
        slug = f"models--Systran--faster-whisper-{model_name}"
        for d in cache_dirs:
            if (d / slug).is_dir():
                snapshots = d / slug / "snapshots"
                if snapshots.is_dir() and any(snapshots.iterdir()):
                    return True
        return False


def get_available_models():
    """
    Return model catalog + system info + recommendation for REST API.
    No interactive UI — pure data.
    """
    sys_info = get_system_info()
    # Use torch.cuda for has_gpu — pynvml detects hardware but CUDA may be broken
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except Exception:
        has_gpu = sys_info["vram_total"] > 0
    vram_free = sys_info["vram_free"]
    supports_fp16 = gpu_supports_float16() if has_gpu else False

    models = []
    for name in ["tiny", "base", "small", "medium", "large-v3-turbo", "distil-large-v3", "large-v3"]:
        vram = MODEL_VRAM[name]
        fits_gpu = (vram["float16"] + INFERENCE_BUFFER_GB) <= vram_free if has_gpu else False
        precision = auto_select_precision(name, vram_free, supports_fp16) if has_gpu else "int8"
        models.append({
            "name": name,
            "params": MODEL_PARAMS[name],
            "vram": vram,
            "latency_gpu_ms": MODEL_LATENCY_GPU[name],
            "latency_cpu_ms": MODEL_LATENCY_CPU[name],
            "accuracy": MODEL_ACCURACY[name],
            "download_size": get_model_download_size(name),
            "notes": MODEL_NOTES.get(name, ""),
            "downloaded": is_model_downloaded(name),
            "fits_gpu": fits_gpu,
            "recommended_precision": precision,
        })

    # Recommended model
    model_names = [m["name"] for m in models]
    rec_idx = _get_recommended_model(vram_free, model_names)
    rec_model = model_names[rec_idx]
    rec_precision = auto_select_precision(rec_model, vram_free, supports_fp16) if has_gpu else "int8"

    return {
        "models": models,
        "system": {
            "gpu_name": sys_info["gpu_name"],
            "vram_total": round(sys_info["vram_total"], 2),
            "vram_free": round(vram_free, 2),
            "vram_used": round(sys_info["vram_used"], 2),
            "supports_fp16": supports_fp16,
            "has_gpu": has_gpu,
            "cpu_name": sys_info["cpu_name"],
            "cpu_cores": sys_info["cpu_cores"],
            "ram_total": round(sys_info["ram_total"], 2),
            "ram_free": round(sys_info["ram_free"], 2),
        },
        "recommended": {
            "model": rec_model,
            "precision": rec_precision,
            "device": "cuda" if has_gpu else "cpu",
        },
    }


def auto_select_precision(model, vram_free, supports_fp16=True):
    """Auto-select best precision based on available VRAM and fp16 support"""
    if model not in MODEL_VRAM:
        return "int8"

    model_vram = MODEL_VRAM[model]

    # Try float16 first (best quality) — needs GPU fp16 support
    if supports_fp16 and model_vram["float16"] + INFERENCE_BUFFER_GB <= vram_free:
        return "float16"
    # Try int8_float16 (balanced) — also needs fp16
    elif supports_fp16 and model_vram["int8_float16"] + INFERENCE_BUFFER_GB <= vram_free:
        return "int8_float16"
    # Fall back to int8 (smallest, no fp16 needed)
    else:
        return "int8"

def select_model(preference="balanced", vram_gb=None, show_info=True):
    """
    Select best model based on preference and VRAM.

    Thresholds based on actual faster-whisper CTranslate2 float16 VRAM:
      tiny/base ~0.5 GB, small ~1 GB, medium/turbo ~1.5 GB, large-v3 ~3 GB

    Args:
        preference: "fast", "accurate", or "balanced"
        vram_gb: Override VRAM detection (optional)
        show_info: Display model info after selection (default True)

    Returns:
        (model_name, compute_type, device)
    """
    if vram_gb is None:
        vram_gb = get_vram_free()

    has_gpu = vram_gb > 0
    device = "cuda" if has_gpu else "cpu"

    # Check if GPU supports float16 (compute capability >= 7.0)
    supports_fp16 = gpu_supports_float16() if has_gpu else False

    if preference == "fast":
        if vram_gb >= 2:
            model, compute = "large-v3-turbo", "float16" if supports_fp16 else "int8"
        elif vram_gb >= 1.5:
            model, compute = "small", "float16" if supports_fp16 else "int8"
        else:
            model, compute = "tiny", "int8"
            device = "cpu"

    elif preference == "accurate":
        if vram_gb >= 3.5:
            model, compute = "large-v3", "float16" if supports_fp16 else "int8"
        elif vram_gb >= 2:
            model, compute = "large-v3-turbo", "float16" if supports_fp16 else "int8"
        elif vram_gb >= 1.5:
            model, compute = "medium", "float16" if supports_fp16 else "int8"
        else:
            model, compute = "medium", "int8"
            device = "cpu"

    else:  # balanced (default)
        if vram_gb >= 3.5:
            model, compute = "large-v3", "float16" if supports_fp16 else "int8"
        elif vram_gb >= 2:
            model, compute = "large-v3-turbo", "float16" if supports_fp16 else "int8"
        elif vram_gb >= 1.5:
            model, compute = "small", "float16" if supports_fp16 else "int8"
        else:
            model, compute = "base", "int8"
            device = "cpu"

    if show_info:
        display_model_info(model, compute, device, preference, vram_gb)

    return model, compute, device


def display_model_info(model, compute, device, preference, vram_free):
    """Display selected model info to user"""
    vram_needed = MODEL_VRAM.get(model, {}).get(compute, 0)
    latency_gpu = MODEL_LATENCY_GPU.get(model, 0)
    latency_cpu = MODEL_LATENCY_CPU.get(model, 0)
    accuracy = MODEL_ACCURACY.get(model, 0)
    params = MODEL_PARAMS.get(model, "?")

    # Get RAM info for CPU mode
    ram_free = 0
    try:
        import psutil
        ram_free = psutil.virtual_memory().available / (1024**3)
    except Exception:
        pass

    print()
    print("=" * 60)
    print(f"  MODEL SELECTION ({preference.upper()} mode)")
    print("=" * 60)
    print(f"  Model:      {model} ({params} parameters)")
    print(f"  Precision:  {compute}")
    print(f"  Device:     {device.upper()}")
    print("-" * 60)

    if device == "cuda":
        print(f"  VRAM:       ~{vram_needed:.1f} GB needed (free: {vram_free:.1f} GB)")
        print(f"  Latency:    ~{latency_gpu}ms per 5s audio")
    else:
        # CPU mode - show RAM instead of VRAM
        ram_needed = vram_needed * 1.5  # CPU needs more RAM than GPU VRAM
        print(f"  RAM:        ~{ram_needed:.1f} GB needed (free: {ram_free:.1f} GB)")
        print(f"  Latency:    ~{latency_cpu/1000:.1f}s per 5s audio (CPU mode)")
        print(f"  Note:       CPU is ~10x slower than GPU")

    print(f"  Accuracy:   ~{accuracy}% (Hindi/English)")
    print("-" * 60)

    # Model download sizes (approximate)
    download_sizes = {
        "tiny": "75 MB", "base": "150 MB", "small": "500 MB",
        "medium": "1.5 GB", "large-v3-turbo": "1.6 GB",
        "distil-large-v3": "1.5 GB", "large-v3": "3 GB"
    }
    download_size = download_sizes.get(model, "unknown")
    print(f"  Download:   ~{download_size} (first run only)")
    print("=" * 60)
    print()

# ============================================================================
#                      TERMINAL UI — MODEL SELECTION
# ============================================================================

# ANSI escape codes — empty strings on Windows cmd.exe (no VT support)
def _supports_ansi():
    """Check if terminal supports ANSI escape codes"""
    if PLATFORM != 'windows':
        return True
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        result = kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        return bool(result)
    except Exception:
        return False

_ANSI = _supports_ansi()

_RST = '\033[0m'   if _ANSI else ''
_B   = '\033[1m'   if _ANSI else ''
_DIM = '\033[2m'   if _ANSI else ''
_GRN = '\033[92m'  if _ANSI else ''
_YLW = '\033[93m'  if _ANSI else ''
_RED = '\033[91m'  if _ANSI else ''
_CYN = '\033[96m'  if _ANSI else ''
_WHT = '\033[97m'  if _ANSI else ''
_GRY = '\033[90m'  if _ANSI else ''

_USE_UNICODE = (PLATFORM != 'windows')
_TL  = '┌' if _USE_UNICODE else '+'
_TR  = '┐' if _USE_UNICODE else '+'
_BL  = '└' if _USE_UNICODE else '+'
_BR  = '┘' if _USE_UNICODE else '+'
_H   = '─' if _USE_UNICODE else '-'
_V   = '│' if _USE_UNICODE else '|'
_LT  = '├' if _USE_UNICODE else '+'
_RT  = '┤' if _USE_UNICODE else '+'
_BLK = '█' if _USE_UNICODE else '#'
_EMP = '░' if _USE_UNICODE else '-'
_STR = '★' if _USE_UNICODE else '*'

_W = 68  # Box width (fits 80-col terminal)

def _clear_screen():
    """Clear terminal screen - cross-platform (ANSI escape works on Win10+ with VT)"""
    print("\033[2J\033[H", end="", flush=True)

def _box_top():
    print(f"  {_GRY}{_TL}{_H * _W}{_TR}{_RST}")

def _box_bot():
    print(f"  {_GRY}{_BL}{_H * _W}{_BR}{_RST}")

def _box_mid():
    print(f"  {_GRY}{_LT}{_H * _W}{_RT}{_RST}")

def _box_line(text=""):
    """Print a line inside the box. Auto-strips ANSI codes for width calculation."""
    import re
    raw_len = len(re.sub(r'\033\[[0-9;]*m', '', text))
    # _W is the horizontal line width; inner content = leading space + text + padding
    padding = max(0, _W - 1 - raw_len)
    print(f"  {_GRY}{_V}{_RST} {text}{' ' * padding}{_GRY}{_V}{_RST}")

def _vram_bar(used, total, width=24):
    """Draw a colored VRAM usage bar. Returns (colored_string, visible_length)."""
    if total <= 0:
        txt = _EMP * width + "  N/A"
        return f"{_GRY}{txt}{_RST}", len(txt)
    pct = min(1.0, used / total)
    filled = int(width * pct)
    bar = _BLK * filled + _EMP * (width - filled)
    free = max(0, total - used)
    color = _GRN if pct < 0.6 else (_YLW if pct < 0.8 else _RED)
    txt = f"{bar}  {used:.1f} / {total:.1f} GB  ({free:.1f} free)"
    return f"{color}{bar}{_RST}  {used:.1f} / {total:.1f} GB  ({free:.1f} free)", len(txt)

def _lat_color(latency_ms):
    """Return ANSI color based on latency (green=fast, yellow=ok, red=slow)."""
    if latency_ms <= 100:  return _GRN
    if latency_ms <= 300:  return _YLW
    return _RED

def _get_recommended_model(vram_free, models):
    """Get recommended model index using balanced logic."""
    priority = ["large-v3", "large-v3-turbo", "medium", "small", "base", "tiny"]
    for m in priority:
        if m not in models:
            continue
        if MODEL_VRAM[m]["float16"] + INFERENCE_BUFFER_GB <= vram_free:
            return models.index(m)
    return models.index("tiny") if "tiny" in models else 0

def custom_model_selection():
    """
    Interactive model selection — elegant terminal UI.
    Two-screen flow: pick model, pick precision.
    Returns: (model_name, precision, device)
    """
    import re as _re

    info = get_system_info()
    vram_free = info["vram_free"]
    vram_total = info["vram_total"]
    vram_used = info["vram_used"]
    has_gpu = vram_free > 0
    device = "cuda" if has_gpu else "cpu"
    supports_fp16 = gpu_supports_float16() if has_gpu else False

    # distil-large-v3 excluded: English-only model, catastrophic for Hindi/Hinglish target
    models = ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"]
    precisions = ["float16", "int8_float16", "int8"]

    rec_idx = _get_recommended_model(vram_free, models)

    # ===================== SCREEN 1: MODEL SELECTION =====================
    while True:
        _clear_screen()
        print()
        _box_top()
        _box_line()

        title = f"{_B}{_WHT}WHISPER MODEL{_RST}"
        _box_line(title)
        _box_line()

        if has_gpu:
            gpu_name = info["gpu_name"][:45]
            _box_line(f"{_GRY}GPU{_RST}    {_WHT}{gpu_name}{_RST}")
            bar_str, bar_len = _vram_bar(vram_used, vram_total)
            _box_line(f"{_GRY}VRAM{_RST}   {bar_str}")
            fp16_str = f"{_GRN}Supported{_RST}" if supports_fp16 else f"{_YLW}Not supported (int8 only){_RST}"
            _box_line(f"{_GRY}FP16{_RST}   {fp16_str}")
        else:
            _box_line(f"{_GRY}GPU{_RST}    {_RED}Not detected — CPU mode{_RST}")
            bar_str, bar_len = _vram_bar(info["ram_used"], info["ram_total"])
            _box_line(f"{_GRY}RAM{_RST}    {bar_str}")

        _box_line()
        _box_mid()
        _box_line()

        # Column widths: num=3, model=17, vram=6, speed=7, acc=5
        hdr = f"{_GRY} {'#':<2}  {'Model':<17}  {'VRAM':<6}   {'Speed':<7}   Accuracy{_RST}"
        _box_line(hdr)
        sep = f"{_GRY} {'─'*3}  {'─'*17}  {'─'*6}   {'─'*7}   {'─'*8}{_RST}"
        _box_line(sep)
        _box_line()

        for i, m in enumerate(models):
            num = i + 1
            vram = MODEL_VRAM[m]["float16"]
            lat = MODEL_LATENCY_GPU[m] if has_gpu else MODEL_LATENCY_CPU[m]
            acc = MODEL_ACCURACY[m]
            note = MODEL_NOTES.get(m, "")
            fits = (vram + INFERENCE_BUFFER_GB) <= vram_free if has_gpu else True
            is_rec = (i == rec_idx)
            lc = _lat_color(lat)

            # Fixed-width latency: 7 chars — GPU: "  60 ms", CPU: " 1.8 s"
            if has_gpu:
                lat_s = f"{lat:>4} ms"
            else:
                lat_s = f"{lat/1000:>4.1f} s " if lat < 10000 else f"{lat/1000:>4.0f} s "

            if not fits:
                colored = f"{_GRY} {num:<2}  {m:<17}  {vram:.1f} GB   {lat_s}   {acc:.1f}%  won't fit{_RST}"
            elif is_rec:
                colored = (f"{_YLW}{_B} {num:<2}{_RST}  {_YLW}{_B}{m:<17}{_RST}  {_WHT}{vram:.1f} GB{_RST}"
                           f"   {lc}{lat_s}{_RST}   {_GRN}{acc:.1f}%{_RST}  {_YLW}{_STR} BEST{_RST}")
            else:
                colored = (f"{_CYN} {num:<2}{_RST}  {_WHT}{m:<17}{_RST}  {_WHT}{vram:.1f} GB{_RST}"
                           f"   {lc}{lat_s}{_RST}   {_WHT}{acc:.1f}%{_RST}")
                if note:
                    colored += f"  {_GRY}{note}{_RST}"

            _box_line(colored)

        _box_line()
        _box_mid()

        rec_num = rec_idx + 1
        n = len(models)
        footer = (f"{_GRY}Enter{_RST} = {_YLW}{rec_num}{_RST} (recommended)"
                  f"  {_GRY}·{_RST}  {_GRY}1-{n} select{_RST}"
                  f"  {_GRY}·{_RST}  {_GRY}q quit{_RST}")
        _box_line(footer)
        _box_bot()
        print()

        try:
            choice = input(f"  {_CYN}>{_RST} ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            sys.exit(0)

        if choice in ('q', 'quit'):
            print(f"\n  {_GRY}Cancelled.{_RST}\n")
            sys.exit(0)

        if choice == '':
            model_idx = rec_idx
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < n:
                    model_idx = idx
                else:
                    continue
            except ValueError:
                continue

        chosen_model = models[model_idx]

        vram_needed = MODEL_VRAM[chosen_model]["float16"]
        if has_gpu and (vram_needed + INFERENCE_BUFFER_GB) > vram_free:
            print()
            print(f"  {_YLW}Warning:{_RST} {chosen_model} needs ~{vram_needed:.1f} GB, you have {vram_free:.1f} GB free.")
            print(f"  {_GRY}It may fall back to CPU. Continue anyway? [y/N]{_RST}")
            try:
                confirm = input(f"  {_CYN}>{_RST} ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print()
                sys.exit(0)
            if confirm not in ('y', 'yes'):
                continue
        break

    # ===================== SCREEN 2: PRECISION SELECTION =====================
    rec_prec_idx = 0
    for pi, prec in enumerate(precisions):
        if (MODEL_VRAM[chosen_model][prec] + INFERENCE_BUFFER_GB) <= vram_free:
            rec_prec_idx = pi
            break
    else:
        rec_prec_idx = 2

    if has_gpu and not supports_fp16:
        rec_prec_idx = 2

    precision_labels = {
        "float16": "Best quality",
        "int8_float16": "Good (mixed)",
        "int8": "Smallest footprint",
    }

    while True:
        _clear_screen()
        print()
        _box_top()
        _box_line()

        params = MODEL_PARAMS.get(chosen_model, "?")
        title = f"{_B}{_WHT}PRECISION{_RST}  {_GRY}·{_RST}  {_CYN}{chosen_model}{_RST} {_GRY}({params} params){_RST}"
        _box_line(title)
        _box_line()

        if has_gpu:
            bar_str, bar_len = _vram_bar(vram_used, vram_total)
            _box_line(f"{_GRY}VRAM{_RST}   {bar_str}")
        else:
            bar_str, bar_len = _vram_bar(info["ram_used"], info["ram_total"])
            _box_line(f"{_GRY}RAM{_RST}    {bar_str}")

        _box_line()
        _box_mid()
        _box_line()

        # Column widths: num=3, type=15, vram=6, quality=25
        hdr = f"{_GRY} {'#':<2}  {'Type':<15}  {'VRAM':<6}   Quality{_RST}"
        _box_line(hdr)
        sep = f"{_GRY} {'─'*3}  {'─'*15}  {'─'*6}   {'─'*25}{_RST}"
        _box_line(sep)
        _box_line()

        for pi, prec in enumerate(precisions):
            num = pi + 1
            vram_needed = MODEL_VRAM[chosen_model][prec]
            label = precision_labels[prec]
            fits = (vram_needed + INFERENCE_BUFFER_GB) <= vram_free if has_gpu else True
            is_rec = (pi == rec_prec_idx)

            if prec == "float16" and has_gpu and not supports_fp16:
                fits = False
                label = "No FP16 support"

            if not fits:
                colored = f"{_GRY} {num:<2}  {prec:<15}  {vram_needed:.1f} GB   {label}{_RST}"
            elif is_rec:
                colored = (f"{_YLW}{_B} {num:<2}{_RST}  {_YLW}{_B}{prec:<15}{_RST}  {_WHT}{vram_needed:.1f} GB{_RST}"
                           f"   {_GRN}{label}{_RST}  {_YLW}{_STR} BEST{_RST}")
            else:
                colored = (f"{_CYN} {num:<2}{_RST}  {_WHT}{prec:<15}{_RST}  {_WHT}{vram_needed:.1f} GB{_RST}"
                           f"   {_WHT}{label}{_RST}")

            _box_line(colored)

        _box_line()
        if not has_gpu:
            _box_line(f"{_GRY}No GPU detected. All options will run on CPU.{_RST}")
            _box_line()

        _box_mid()

        rec_p = rec_prec_idx + 1
        footer = (f"{_GRY}Enter{_RST} = {_YLW}{rec_p}{_RST} (recommended)"
                  f"  {_GRY}·{_RST}  {_GRY}1-3 select{_RST}"
                  f"  {_GRY}·{_RST}  {_GRY}b back{_RST}"
                  f"  {_GRY}·{_RST}  {_GRY}q quit{_RST}")
        _box_line(footer)
        _box_bot()
        print()

        try:
            choice = input(f"  {_CYN}>{_RST} ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            sys.exit(0)

        if choice in ('q', 'quit'):
            print(f"\n  {_GRY}Cancelled.{_RST}\n")
            sys.exit(0)
        if choice in ('b', 'back'):
            return custom_model_selection()  # Restart from Screen 1 (recursion OK — max ~3 depth in practice)

        if choice == '':
            prec_idx = rec_prec_idx
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < 3:
                    prec_idx = idx
                else:
                    continue
            except ValueError:
                continue

        chosen_precision = precisions[prec_idx]

        vram_needed = MODEL_VRAM[chosen_model][chosen_precision]
        if has_gpu and (vram_needed + INFERENCE_BUFFER_GB) > vram_free:
            print()
            print(f"  {_YLW}Warning:{_RST} {chosen_model} + {chosen_precision} "
                  f"needs ~{vram_needed:.1f} GB, you have {vram_free:.1f} GB free.")
            print(f"  {_GRY}May fall back to CPU. Continue? [y/N]{_RST}")
            try:
                confirm = input(f"  {_CYN}>{_RST} ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print()
                sys.exit(0)
            if confirm not in ('y', 'yes'):
                continue
        break

    # ===================== CONFIRMATION =====================
    _clear_screen()
    print()

    lat_gpu = MODEL_LATENCY_GPU.get(chosen_model, 0)
    lat_cpu = MODEL_LATENCY_CPU.get(chosen_model, 0)
    acc = MODEL_ACCURACY.get(chosen_model, 0)
    params = MODEL_PARAMS.get(chosen_model, "?")
    dl = get_model_download_size(chosen_model)

    _box_top()
    _box_line()
    _box_line(f"{_B}{_WHT}CONFIGURATION{_RST}")
    _box_line()
    _box_line(f"{_GRY}Model{_RST}      {_WHT}{chosen_model}{_RST} {_GRY}({params}){_RST}")
    _box_line(f"{_GRY}Precision{_RST}  {_WHT}{chosen_precision}{_RST}")
    _box_line(f"{_GRY}Device{_RST}     {_WHT}{device.upper()}{_RST}")

    if has_gpu:
        _box_line(f"{_GRY}Latency{_RST}    {_GRN}~{lat_gpu}ms per 5s audio{_RST}")
    else:
        _box_line(f"{_GRY}Latency{_RST}    {_YLW}~{lat_cpu/1000:.1f}s per 5s audio (CPU){_RST}")

    _box_line(f"{_GRY}Accuracy{_RST}   {_WHT}~{acc}%{_RST}")
    _box_line(f"{_GRY}Download{_RST}   {_WHT}~{dl}{_RST} {_GRY}(first run){_RST}")
    _box_line()
    _box_bot()

    print()
    print(f"  {_GRN}Starting...{_RST}")
    print()

    return chosen_model, chosen_precision, device

# ============================================================================
#                         PLATFORM-SPECIFIC TYPING
# ============================================================================

# macOS Accessibility permission state — checked once, cached forever.
# AXIsProcessTrusted() is the canonical check for whether pynput can generate
# keyboard/mouse events. Without it, ALL pynput operations silently fail.
_macos_accessibility_checked = False
_macos_accessibility_ok = True
_macos_accessibility_error = None  # Human-readable error string, or None if OK

def _check_macos_accessibility():
    """One-time probe for macOS Accessibility permission via AXIsProcessTrusted().
    Returns True if OK or not macOS. Caches result — safe to call from hot path."""
    global _macos_accessibility_checked, _macos_accessibility_ok, _macos_accessibility_error
    if _macos_accessibility_checked:
        return _macos_accessibility_ok
    if PLATFORM != 'macos':
        _macos_accessibility_checked = True
        return True
    _macos_accessibility_checked = True
    try:
        import ctypes
        import ctypes.util
        # ApplicationServices framework contains AXIsProcessTrusted
        lib_path = ctypes.util.find_library('ApplicationServices')
        if lib_path:
            hi = ctypes.cdll.LoadLibrary(lib_path)
        else:
            # Fallback to direct framework path
            hi = ctypes.cdll.LoadLibrary(
                '/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices'
            )
        hi.AXIsProcessTrusted.restype = ctypes.c_bool
        _macos_accessibility_ok = bool(hi.AXIsProcessTrusted())
    except Exception as e:
        # If the check itself fails (e.g., framework not found), assume OK
        # rather than blocking all input on a false negative.
        _macos_accessibility_ok = True
        try:
            log_debug(f"[macOS] Accessibility check failed (assuming OK): {e}")
        except Exception:
            pass
        return True
    if not _macos_accessibility_ok:
        _macos_accessibility_error = (
            "macOS Accessibility permission NOT granted. "
            "Input dispatch (typing, mouse, keyboard) will not work. "
            "Fix: System Settings > Privacy & Security > Accessibility — "
            "add the app that runs this server."
        )
        try:
            log_warning(f"[macOS] {_macos_accessibility_error}")
        except Exception:
            print(f"[stt_common] WARNING: {_macos_accessibility_error}")
    return _macos_accessibility_ok

def check_input_health():
    """Check if input dispatch is functional. Returns (ok: bool, error: str|None).
    Safe to call at any time — runs macOS Accessibility probe once on first call."""
    if PLATFORM == 'macos':
        _check_macos_accessibility()
        if not _macos_accessibility_ok:
            return False, _macos_accessibility_error
    return True, None

def is_typing_supported():
    """Check if typing at cursor is supported on this platform"""
    if PLATFORM == 'linux':
        tool, available = get_linux_input_tool()
        return available
    else:
        try:
            import pyautogui
            return True
        except ImportError:
            return False

def get_input_tool_info():
    """Get info about input tool for display to user"""
    if PLATFORM == 'linux':
        display = get_display_server()
        tool, available = get_linux_input_tool()
        if available:
            if tool == 'ydotool-sudo':
                return f"ydotool (sudo) on {display}"
            return f"{tool} on {display}"
        else:
            if display == 'wayland':
                return "Wayland detected but ydotool not installed"
            else:
                return "No input tool found (install xdotool or ydotool)"
    else:
        return f"{PLATFORM} (pyautogui)"

def type_text(text):
    """Type text at cursor position - cross-platform"""
    if not text:
        return
    if PLATFORM == 'macos' and not _check_macos_accessibility():
        return
    if is_app_input_paused():
        log_debug(f"[InputGuard] type_text suppressed: {text[:50]}")
        return

    if PLATFORM == 'linux':
        tool, available = get_linux_input_tool()
        if not available:
            log_warning(f"Cannot type: {get_input_tool_info()}")
            return

        if tool == 'xdotool':
            # Try Xlib direct first (no subprocess overhead), fallback to xdotool
            if not _xlib_type_text(text):
                subprocess.run(['xdotool', 'type', '--', text], check=False)
        elif tool in ('ydotool', 'ydotool-sudo'):
            _run_ydotool(['type', text])
    elif PLATFORM in ('windows', 'macos'):
        kb = _get_pynput_keyboard()
        if kb:
            try:
                _mark_app_output()
                kb.type(text)
                return
            except Exception:
                pass
        # Fallback: pyautogui is ASCII-only (typewrite drops non-ASCII chars).
        # For non-ASCII text on macOS, use clipboard paste (pyperclip + Cmd+V) as workaround.
        # On Windows, pyautogui.write() also only handles ASCII; clipboard paste uses Ctrl+V.
        if not text.isascii():
            try:
                import pyperclip
                pyperclip.copy(text)
                import pyautogui
                modifier = 'command' if PLATFORM == 'macos' else 'ctrl'
                pyautogui.hotkey(modifier, 'v')
                log_debug("pynput failed — used clipboard paste for non-ASCII text")
                return
            except ImportError:
                log_warning("pynput failed, pyperclip not available — non-ASCII text dropped")
            except Exception:
                log_warning("pynput failed — clipboard paste fallback also failed")
        try:
            import pyautogui
            pyautogui.typewrite(text, interval=0.01)
        except ImportError:
            log_warning("Cannot type — install pynput")

def key_press(key):
    """Press a special key (enter, backspace, etc.)"""
    if not key:
        return
    if PLATFORM == 'macos' and not _check_macos_accessibility():
        return
    if is_app_input_paused():
        return

    key_lower = key.lower()

    if PLATFORM == 'linux':
        display = get_display_server()

        # X11: direct Xlib (fastest), fallback to pynput, then xdotool
        if display == 'x11':
            if _init_xlib() and _xlib_key_press(key):
                return
            # Fallback to pynput
            kb = _get_pynput_keyboard()
            if kb:
                try:
                    from pynput.keyboard import Key
                    key_map = {
                        'enter': Key.enter, 'return': Key.enter,
                        'backspace': Key.backspace,
                        'tab': Key.tab, 'escape': Key.esc, 'esc': Key.esc,
                        'space': Key.space,
                        'up': Key.up, 'down': Key.down, 'left': Key.left, 'right': Key.right,
                        'delete': Key.delete, 'home': Key.home, 'end': Key.end,
                        'insert': Key.insert,
                        'pageup': Key.page_up, 'page_up': Key.page_up,
                        'pagedown': Key.page_down, 'page_down': Key.page_down,
                        'f1': Key.f1, 'f2': Key.f2, 'f3': Key.f3, 'f4': Key.f4,
                        'f5': Key.f5, 'f6': Key.f6, 'f7': Key.f7, 'f8': Key.f8,
                        'f9': Key.f9, 'f10': Key.f10, 'f11': Key.f11, 'f12': Key.f12,
                    }
                    pynput_key = key_map.get(key_lower)
                    if pynput_key:
                        kb.press(pynput_key)
                        kb.release(pynput_key)
                        return
                except Exception:
                    pass
            subprocess.run(['xdotool', 'key', key], check=False)
            return

        # Wayland: use uinput or ydotool
        if _init_uinput() and _uinput_fd:
            keycode = _UINPUT_KEY_MAP.get(key_lower)
            if keycode:
                _write_uinput_event(_EV_KEY, keycode, 1)
                _uinput_syn()
                _write_uinput_event(_EV_KEY, keycode, 0)
                _uinput_syn()
                return

        # Fallback to ydotool for Wayland
        tool, available = get_linux_input_tool()
        if not available:
            return

        if tool in ('ydotool', 'ydotool-sudo'):
            _run_ydotool(['key', key])
    else:
        # Windows / macOS: use pynput
        kb = _get_pynput_keyboard()
        if kb:
            try:
                from pynput.keyboard import Key
                key_map = {
                    'enter': Key.enter, 'return': Key.enter,
                    'backspace': Key.backspace,
                    'tab': Key.tab, 'escape': Key.esc, 'esc': Key.esc,
                    'space': Key.space,
                    'up': Key.up, 'down': Key.down, 'left': Key.left, 'right': Key.right,
                    'delete': Key.delete, 'home': Key.home, 'end': Key.end,
                    'insert': Key.insert,
                    'pageup': Key.page_up, 'page_up': Key.page_up,
                    'pagedown': Key.page_down, 'page_down': Key.page_down,
                    'f1': Key.f1, 'f2': Key.f2, 'f3': Key.f3, 'f4': Key.f4,
                    'f5': Key.f5, 'f6': Key.f6, 'f7': Key.f7, 'f8': Key.f8,
                    'f9': Key.f9, 'f10': Key.f10, 'f11': Key.f11, 'f12': Key.f12,
                }
                pynput_key = key_map.get(key_lower)
                if pynput_key:
                    _mark_app_output()
                    kb.press(pynput_key)
                    kb.release(pynput_key)
            except Exception:
                pass

# ============================================================================
#                        PLATFORM-SPECIFIC MOUSE
# ============================================================================

TRACKPAD_SENSITIVITY = 1.5
SCROLL_SPEED = 3

# Cached pynput controllers (for Windows/macOS - avoids creating new object each call)
_pynput_mouse = None
_pynput_keyboard = None

def _get_pynput_mouse():
    """Get cached pynput mouse controller"""
    global _pynput_mouse
    if _pynput_mouse is None:
        try:
            from pynput.mouse import Controller
            _pynput_mouse = Controller()
        except Exception:
            pass
    return _pynput_mouse

def _get_pynput_keyboard():
    """Get cached pynput keyboard controller"""
    global _pynput_keyboard
    if _pynput_keyboard is None:
        try:
            from pynput.keyboard import Controller
            _pynput_keyboard = Controller()
        except Exception:
            pass
    return _pynput_keyboard

# Direct Xlib for X11 (fastest - no pynput overhead, explicit flush)
# Per-thread Display objects — Xlib Display is NOT thread-safe, so each thread
# gets its own connection. Uses threading.local() to avoid races.
_xlib_thread_local = _threading.local()
_xlib_available = False
_xlib_init_attempted = False

def _get_xlib_display():
    """Get per-thread Xlib Display (thread-safe). Returns Display or None."""
    if not _xlib_available:
        return None
    if not hasattr(_xlib_thread_local, 'display') or _xlib_thread_local.display is None:
        try:
            import Xlib.display
            _xlib_thread_local.display = Xlib.display.Display()
        except Exception:
            _xlib_thread_local.display = None
    return _xlib_thread_local.display

def _init_xlib():
    """Initialize direct Xlib connection for X11 (fastest mouse control)"""
    global _xlib_available, _xlib_init_attempted

    if _xlib_init_attempted:
        return _xlib_available

    _xlib_init_attempted = True

    if get_display_server() != 'x11':
        _xlib_available = False
        return False

    try:
        from Xlib import X, display
        from Xlib.ext import xtest
        # Probe: create a display to verify Xlib works, then let per-thread handle ongoing usage
        d = display.Display()
        d.close()
        _xlib_available = True
        return True
    except Exception as e:
        log_debug(f"Xlib init failed: {e}")
        _xlib_available = False
        return False

def _xlib_mouse_move(dx, dy):
    """Move mouse using direct Xlib with explicit flush (fastest for X11)"""
    d = _get_xlib_display()
    if d is None:
        return False
    try:
        from Xlib import X
        from Xlib.ext import xtest
        # Relative move using XTest
        xtest.fake_input(d, X.MotionNotify, detail=True, x=int(dx), y=int(dy))
        d.flush()  # Critical! Without this, commands are buffered
        return True
    except Exception:
        return False

def _xlib_mouse_move_absolute(x, y):
    """Move cursor to absolute position using Xlib warp_pointer"""
    d = _get_xlib_display()
    if d is None:
        return False
    try:
        root = d.screen().root
        root.warp_pointer(int(x), int(y))
        d.flush()
        return True
    except Exception:
        return False

def _xlib_mouse_click(button=1):
    """Click mouse using direct Xlib"""
    d = _get_xlib_display()
    if d is None:
        return False
    try:
        from Xlib import X
        from Xlib.ext import xtest
        btn_map = {1: 1, 2: 2, 3: 3}  # X11 button numbers
        btn = btn_map.get(button, 1)
        xtest.fake_input(d, X.ButtonPress, btn)
        xtest.fake_input(d, X.ButtonRelease, btn)
        d.flush()
        return True
    except Exception:
        return False

_xlib_scroll_accum = 0.0

def _xlib_mouse_scroll(dy):
    """Scroll using direct Xlib button 4/5 (no subprocess). Accumulates fractional values."""
    global _xlib_scroll_accum
    d = _get_xlib_display()
    if d is None:
        return False
    try:
        from Xlib import X
        from Xlib.ext import xtest
        _xlib_scroll_accum += dy
        clicks = int(_xlib_scroll_accum)
        if clicks == 0:
            return True  # accumulated but not enough yet
        _xlib_scroll_accum -= clicks
        btn = 4 if clicks > 0 else 5
        for _ in range(abs(clicks)):
            xtest.fake_input(d, X.ButtonPress, btn)
            xtest.fake_input(d, X.ButtonRelease, btn)
        d.flush()
        return True
    except Exception:
        return False

def _xlib_mouse_drag(action="down"):
    """Mouse drag using direct Xlib"""
    d = _get_xlib_display()
    if d is None:
        return False
    try:
        from Xlib import X
        from Xlib.ext import xtest
        if action == "down":
            xtest.fake_input(d, X.ButtonPress, 1)
        else:
            xtest.fake_input(d, X.ButtonRelease, 1)
        d.flush()
        return True
    except Exception:
        return False

def _xlib_key_press(key):
    """Press key using direct Xlib"""
    d = _get_xlib_display()
    if d is None:
        return False
    try:
        from Xlib import X, XK
        from Xlib.ext import xtest

        key_lower = key.lower()
        keysym_map = {
            'enter': XK.XK_Return, 'return': XK.XK_Return,
            'backspace': XK.XK_BackSpace,
            'tab': XK.XK_Tab,
            'escape': XK.XK_Escape, 'esc': XK.XK_Escape,
            'space': XK.XK_space,
            'up': XK.XK_Up, 'down': XK.XK_Down, 'left': XK.XK_Left, 'right': XK.XK_Right,
            'delete': XK.XK_Delete, 'home': XK.XK_Home, 'end': XK.XK_End,
            'insert': XK.XK_Insert,
            'pageup': XK.XK_Page_Up, 'page_up': XK.XK_Page_Up,
            'pagedown': XK.XK_Page_Down, 'page_down': XK.XK_Page_Down,
            'f1': XK.XK_F1, 'f2': XK.XK_F2, 'f3': XK.XK_F3, 'f4': XK.XK_F4,
            'f5': XK.XK_F5, 'f6': XK.XK_F6, 'f7': XK.XK_F7, 'f8': XK.XK_F8,
            'f9': XK.XK_F9, 'f10': XK.XK_F10, 'f11': XK.XK_F11, 'f12': XK.XK_F12,
        }
        keysym = keysym_map.get(key_lower)
        if not keysym:
            return False

        keycode = d.keysym_to_keycode(keysym)
        if not keycode:
            return False

        xtest.fake_input(d, X.KeyPress, keycode)
        xtest.fake_input(d, X.KeyRelease, keycode)
        d.flush()
        return True
    except Exception:
        return False

def _xlib_type_text(text):
    """Type text using direct Xlib XTest (eliminates subprocess overhead)"""
    d = _get_xlib_display()
    if d is None:
        return False
    try:
        from Xlib import X, XK
        from Xlib.ext import xtest

        shift_keycode = d.keysym_to_keycode(XK.XK_Shift_L)

        for char in text:
            keysym = ord(char)
            keycode = d.keysym_to_keycode(keysym)
            if not keycode:
                continue

            # Check if shift is needed by comparing unshifted keysym at this keycode
            need_shift = (d.keycode_to_keysym(keycode, 0) != keysym)

            if need_shift:
                xtest.fake_input(d, X.KeyPress, shift_keycode)
            xtest.fake_input(d, X.KeyPress, keycode)
            xtest.fake_input(d, X.KeyRelease, keycode)
            if need_shift:
                xtest.fake_input(d, X.KeyRelease, shift_keycode)

        d.flush()
        return True
    except Exception:
        return False

# Direct uinput access (Linux only, no external packages needed)
import struct
if sys.platform == 'linux':
    import fcntl
else:
    fcntl = None

_uinput_fd = None
_uinput_available = False
_uinput_init_attempted = False

# uinput constants
_UINPUT_MAX_NAME_SIZE = 80
_UI_DEV_CREATE = 0x5501
_UI_DEV_DESTROY = 0x5502
_UI_SET_EVBIT = 0x40045564
_UI_SET_KEYBIT = 0x40045565
_UI_SET_RELBIT = 0x40045566

_EV_SYN = 0x00
_EV_KEY = 0x01
_EV_REL = 0x02
_SYN_REPORT = 0x00
_REL_X = 0x00
_REL_Y = 0x01
_REL_WHEEL = 0x08
_REL_WHEEL_HI_RES = 0x0B
_BTN_LEFT = 0x110
_BTN_RIGHT = 0x111
_BTN_MIDDLE = 0x112
# Linux key codes from input-event-codes.h
_KEY_BACKSPACE = 14
_KEY_TAB = 15
_KEY_ENTER = 28
_KEY_SPACE = 57
_KEY_ESC = 1
_KEY_INSERT = 110
_KEY_DELETE = 111
_KEY_HOME = 102
_KEY_END = 107
_KEY_PAGEUP = 104
_KEY_PAGEDOWN = 109
_KEY_UP = 103
_KEY_DOWN = 108
_KEY_LEFT = 105
_KEY_RIGHT = 106
_KEY_F1 = 59
_KEY_F2 = 60
_KEY_F3 = 61
_KEY_F4 = 62
_KEY_F5 = 63
_KEY_F6 = 64
_KEY_F7 = 65
_KEY_F8 = 66
_KEY_F9 = 67
_KEY_F10 = 68
_KEY_F11 = 87
_KEY_F12 = 88

# Mapping from key names to uinput key codes
_UINPUT_KEY_MAP = {
    'enter': _KEY_ENTER, 'return': _KEY_ENTER,
    'backspace': _KEY_BACKSPACE,
    'tab': _KEY_TAB,
    'space': _KEY_SPACE,
    'escape': _KEY_ESC, 'esc': _KEY_ESC,
    'insert': _KEY_INSERT,
    'delete': _KEY_DELETE,
    'home': _KEY_HOME,
    'end': _KEY_END,
    'pageup': _KEY_PAGEUP, 'page_up': _KEY_PAGEUP,
    'pagedown': _KEY_PAGEDOWN, 'page_down': _KEY_PAGEDOWN,
    'up': _KEY_UP, 'down': _KEY_DOWN, 'left': _KEY_LEFT, 'right': _KEY_RIGHT,
    'f1': _KEY_F1, 'f2': _KEY_F2, 'f3': _KEY_F3, 'f4': _KEY_F4,
    'f5': _KEY_F5, 'f6': _KEY_F6, 'f7': _KEY_F7, 'f8': _KEY_F8,
    'f9': _KEY_F9, 'f10': _KEY_F10, 'f11': _KEY_F11, 'f12': _KEY_F12,
}
# Unique key codes to register with uinput
_UINPUT_ALL_KEYS = list(set(_UINPUT_KEY_MAP.values()))

def _init_uinput():
    """Initialize uinput device directly (no evdev needed) - Linux only"""
    global _uinput_fd, _uinput_available, _uinput_init_attempted

    if _uinput_init_attempted:
        return _uinput_available

    _uinput_init_attempted = True

    # uinput is Linux-only
    if sys.platform != 'linux':
        _uinput_available = False
        return False

    try:
        # Open uinput device
        _uinput_fd = os.open('/dev/uinput', os.O_WRONLY | os.O_NONBLOCK)

        # Enable event types
        fcntl.ioctl(_uinput_fd, _UI_SET_EVBIT, _EV_KEY)
        fcntl.ioctl(_uinput_fd, _UI_SET_EVBIT, _EV_REL)

        # Enable mouse buttons
        fcntl.ioctl(_uinput_fd, _UI_SET_KEYBIT, _BTN_LEFT)
        fcntl.ioctl(_uinput_fd, _UI_SET_KEYBIT, _BTN_RIGHT)
        fcntl.ioctl(_uinput_fd, _UI_SET_KEYBIT, _BTN_MIDDLE)

        # Enable ALL keyboard keys used by key_press (not just enter/backspace)
        for kc in _UINPUT_ALL_KEYS:
            fcntl.ioctl(_uinput_fd, _UI_SET_KEYBIT, kc)

        # Enable relative axes
        fcntl.ioctl(_uinput_fd, _UI_SET_RELBIT, _REL_X)
        fcntl.ioctl(_uinput_fd, _UI_SET_RELBIT, _REL_Y)
        fcntl.ioctl(_uinput_fd, _UI_SET_RELBIT, _REL_WHEEL)
        fcntl.ioctl(_uinput_fd, _UI_SET_RELBIT, _REL_WHEEL_HI_RES)

        # Create device - uinput_user_dev structure
        name = b'sanketra-mouse'
        name = name[:_UINPUT_MAX_NAME_SIZE].ljust(_UINPUT_MAX_NAME_SIZE, b'\0')
        # struct uinput_user_dev: name[80], id{bustype,vendor,product,version}, ff_effects_max, absmax[64], absmin[64], absfuzz[64], absflat[64]
        user_dev = name + struct.pack('<HHHHi', 0x03, 0x1234, 0x5678, 1, 0) + b'\0' * (64 * 4 * 4)
        os.write(_uinput_fd, user_dev)

        # Create the device
        fcntl.ioctl(_uinput_fd, _UI_DEV_CREATE)

        _uinput_available = True
        log_info("Direct uinput initialized - fast trackpad enabled")
        return True
    except Exception as ex:
        log_warning(f"uinput unavailable: {ex}")
        _uinput_available = False
        if _uinput_fd:
            try:
                os.close(_uinput_fd)
            except Exception:
                pass
            _uinput_fd = None
        return False

def _write_uinput_event(ev_type, code, value):
    """Write a single input event"""
    if not _uinput_fd:
        return
    # struct input_event: time (16 bytes on 64-bit), type, code, value
    import time
    t = time.time()
    sec = int(t)
    usec = int((t - sec) * 1000000)
    event = struct.pack('@llHHi', sec, usec, ev_type, code, value)
    try:
        os.write(_uinput_fd, event)
    except Exception:
        pass

def _uinput_syn():
    """Send sync event"""
    _write_uinput_event(_EV_SYN, _SYN_REPORT, 0)

import time as _time

def _detect_screen_size():
    """Detect screen resolution using platform-native APIs.
    Windows: After SetProcessDPIAware(), GetSystemMetrics returns physical pixels.
    However SetCursorPos uses the same coordinate space, so this is consistent
    for mouse_move_absolute. For multi-monitor DPI-per-monitor awareness,
    we attempt GetSystemMetricsForDpi first (Win10 1607+)."""
    if PLATFORM == 'windows':
        try:
            import ctypes
            user32 = ctypes.windll.user32
            # Try per-monitor DPI-aware API (Win10 1607+) — returns physical pixels
            # without needing SetProcessDPIAware which is process-global
            try:
                shcore = ctypes.windll.shcore
                # PROCESS_PER_MONITOR_DPI_AWARE = 2
                shcore.SetProcessDpiAwareness(2)
            except Exception:
                # Fallback to legacy DPI awareness
                user32.SetProcessDPIAware()
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        except Exception:
            pass
    elif PLATFORM == 'macos':
        try:
            import Quartz
            main_display = Quartz.CGMainDisplayID()
            return Quartz.CGDisplayPixelsWide(main_display), Quartz.CGDisplayPixelsHigh(main_display)
        except Exception:
            pass
        try:
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'],
                                    capture_output=True, text=True, timeout=5)
            for line in result.stdout.split('\n'):
                match = re.search(r'Resolution:\s*(\d+)\s*x\s*(\d+)', line)
                if match:
                    return int(match.group(1)), int(match.group(2))
        except Exception:
            pass
    else:  # Linux
        try:
            result = subprocess.run(['xrandr'], capture_output=True, text=True, timeout=2)
            for line in result.stdout.split('\n'):
                if ' connected' in line and 'x' in line:
                    match = re.search(r'(\d+)x(\d+)', line)
                    if match:
                        return int(match.group(1)), int(match.group(2))
        except Exception:
            pass
    return 1920, 1080

_screen_cache = None
_screen_cache_time = 0
_screen_cache_lock = _threading.Lock()
_SCREEN_CACHE_TTL = 30  # seconds

def get_screen_resolution():
    """Return (width, height) of primary screen. Cached with 30s TTL. Thread-safe."""
    global _screen_cache, _screen_cache_time
    with _screen_cache_lock:
        now = _time.time()
        if _screen_cache and now - _screen_cache_time < _SCREEN_CACHE_TTL:
            return _screen_cache
        _screen_cache = _detect_screen_size()
        _screen_cache_time = now
        return _screen_cache

def get_screen_resolution_physical():
    """Return physical pixel resolution (for screen capture).
    On macOS Retina, logical != physical. On other platforms, same as get_screen_resolution()."""
    if PLATFORM == 'macos':
        try:
            import Quartz
            main_display = Quartz.CGMainDisplayID()
            mode = Quartz.CGDisplayCopyDisplayMode(main_display)
            pw = Quartz.CGDisplayModeGetPixelWidth(mode)
            ph = Quartz.CGDisplayModeGetPixelHeight(mode)
            if pw > 0 and ph > 0:
                return pw, ph
        except Exception:
            pass
    return get_screen_resolution()

# Lazy screen size init — actual detection happens on first get_screen_resolution() call.
# For _mouse_pos initial center, we defer to a function to avoid import-time detection.
def _get_initial_mouse_pos():
    """Get initial mouse position (screen center). Called lazily on first trackpad use."""
    w, h = get_screen_resolution()
    return {'x': w // 2, 'y': h // 2}

_mouse_pos = None  # Initialized lazily via _ensure_mouse_pos()

def _ensure_mouse_pos():
    """Ensure _mouse_pos is initialized (lazy init to avoid import-time screen detection)."""
    global _mouse_pos
    if _mouse_pos is None:
        _mouse_pos = _get_initial_mouse_pos()
_pending_move = {'dx': 0, 'dy': 0, 'last_send': 0}
_MOVE_THROTTLE_MS = 33  # ~30fps - balance between smoothness and performance

_ydotool_sudo_setup_done = False

def _setup_ydotool_sudo():
    """Setup passwordless sudo for ydotool (runs once)"""
    global _ydotool_sudo_setup_done
    if _ydotool_sudo_setup_done:
        return True

    # Check if passwordless sudo already works
    result = subprocess.run(['sudo', '-n', 'ydotool', '--help'],
                           capture_output=True, timeout=5)
    if result.returncode == 0:
        _ydotool_sudo_setup_done = True
        return True

    # Need to setup sudoers - will ask for password once
    print("\n[First-time setup] Configuring ydotool permissions (enter password once)...")
    user = os.environ.get('USER', 'root')
    sudoers_content = f"{user} ALL=(ALL) NOPASSWD: /usr/bin/ydotool"
    sudoers_file = "/etc/sudoers.d/ydotool"

    # Create sudoers file — use subprocess with input= to avoid shell injection via USER
    tee_result = subprocess.run(['sudo', 'tee', sudoers_file],
                                input=(sudoers_content + '\n').encode(),
                                capture_output=True)
    chmod_result = subprocess.run(['sudo', 'chmod', '440', sudoers_file],
                                  capture_output=True)
    result = tee_result.returncode or chmod_result.returncode

    if result == 0:
        print("[OK] ydotool configured - no password needed from now on\n")
        _ydotool_sudo_setup_done = True
        return True
    else:
        print("[FAILED] Could not configure ydotool permissions\n")
        return False

import threading
import queue

# Background thread for ydotool commands
_ydotool_queue = queue.Queue()
_ydotool_thread = None

def _ydotool_worker():
    """Background worker that processes ydotool commands"""
    while True:
        try:
            args = _ydotool_queue.get(timeout=1)
            if args is None:  # Shutdown signal
                break
            tool, _ = get_linux_input_tool()
            if tool == 'ydotool-sudo':
                subprocess.run(['sudo', '-n', 'ydotool'] + args,
                              capture_output=True, timeout=1)
            else:
                subprocess.run(['ydotool'] + args,
                              capture_output=True, timeout=1)
        except queue.Empty:
            continue
        except Exception:
            pass

def _ensure_ydotool_thread():
    """Start background thread if not running"""
    global _ydotool_thread
    if _ydotool_thread is None or not _ydotool_thread.is_alive():
        _ydotool_thread = threading.Thread(target=_ydotool_worker, daemon=True)
        _ydotool_thread.start()

def _run_ydotool(args):
    """Queue ydotool command to background thread (non-blocking)"""
    _setup_ydotool_sudo()  # Ensure sudo configured
    _ensure_ydotool_thread()
    try:
        _ydotool_queue.put_nowait(args)
    except queue.Full:
        pass  # Drop if queue full

def mouse_move(dx, dy):
    """Move mouse relative to current position"""
    if PLATFORM == 'macos' and not _check_macos_accessibility():
        return
    if is_app_input_paused():
        return
    _ensure_mouse_pos()
    global _mouse_pos, _pending_move

    dx = int(dx * TRACKPAD_SENSITIVITY)
    dy = int(dy * TRACKPAD_SENSITIVITY)

    if PLATFORM == 'linux':
        display = get_display_server()

        # X11: direct Xlib with flush (fastest), fallback to pynput, then xdotool
        if display == 'x11':
            if _init_xlib() and _xlib_mouse_move(dx, dy):
                return
            # Fallback to pynput
            mouse = _get_pynput_mouse()
            if mouse:
                mouse.move(dx, dy)
                return
            # Fallback to xdotool
            subprocess.run(['xdotool', 'mousemove_relative', '--', str(dx), str(dy)], check=False)
            return

        # Wayland: use uinput (needs permissions) or ydotool
        if _init_uinput() and _uinput_fd:
            _write_uinput_event(_EV_REL, _REL_X, dx)
            _write_uinput_event(_EV_REL, _REL_Y, dy)
            _uinput_syn()
            return

        # Fallback to ydotool for Wayland
        tool, available = get_linux_input_tool()
        if not available:
            return

        if tool in ('ydotool', 'ydotool-sudo'):
            _pending_move['dx'] += dx
            _pending_move['dy'] += dy
            now = _time.time() * 1000
            if now - _pending_move['last_send'] < _MOVE_THROTTLE_MS:
                return
            sw, sh = get_screen_resolution()
            _mouse_pos['x'] = max(0, min(sw, _mouse_pos['x'] + _pending_move['dx']))
            _mouse_pos['y'] = max(0, min(sh, _mouse_pos['y'] + _pending_move['dy']))
            _run_ydotool(['mousemove', str(_mouse_pos['x']), str(_mouse_pos['y'])])
            _pending_move['dx'] = 0
            _pending_move['dy'] = 0
            _pending_move['last_send'] = now
    else:
        # Windows / macOS
        mouse = _get_pynput_mouse()
        if mouse:
            _mark_app_output()
            mouse.move(dx, dy)

def mouse_move_absolute(x, y):
    """Move cursor to absolute screen position — for gyro pointer"""
    if PLATFORM == 'macos' and not _check_macos_accessibility():
        return
    if is_app_input_paused():
        return
    _ensure_mouse_pos()
    global _mouse_pos

    # Clamp to screen bounds (fresh resolution, handles runtime changes)
    sw, sh = get_screen_resolution()
    x = max(0, min(sw, int(x)))
    y = max(0, min(sh, int(y)))

    if PLATFORM == 'linux':
        display = get_display_server()

        # X11: direct Xlib warp_pointer (fastest), fallback to xdotool
        if display == 'x11':
            if _init_xlib() and _xlib_mouse_move_absolute(x, y):
                _mouse_pos['x'] = x
                _mouse_pos['y'] = y
                return
            # Fallback to pynput
            mouse = _get_pynput_mouse()
            if mouse:
                mouse.position = (x, y)
                _mouse_pos['x'] = x
                _mouse_pos['y'] = y
                return
            # Fallback to xdotool
            subprocess.run(['xdotool', 'mousemove', str(x), str(y)], check=False)
            _mouse_pos['x'] = x
            _mouse_pos['y'] = y
            return

        # Wayland: ydotool mousemove --absolute, or compute delta from tracked pos
        if _init_uinput() and _uinput_fd:
            # KNOWN LIMITATION: uinput only supports relative events. We compute the delta
            # from a tracked _mouse_pos, but this drifts if the user physically moves the
            # mouse or if another program repositions the cursor, because there is no way
            # to query the actual cursor position on Wayland without compositor cooperation.
            # The drift accumulates over time. On input monitor resume, _mouse_pos could
            # be re-synced if a compositor-specific API (e.g. wlr-foreign-toplevel) is available.
            dx = x - _mouse_pos['x']
            dy = y - _mouse_pos['y']
            if dx != 0 or dy != 0:
                _write_uinput_event(_EV_REL, _REL_X, dx)
                _write_uinput_event(_EV_REL, _REL_Y, dy)
                _uinput_syn()
            _mouse_pos['x'] = x
            _mouse_pos['y'] = y
            return

        # Fallback to ydotool absolute
        tool, available = get_linux_input_tool()
        if available and tool in ('ydotool', 'ydotool-sudo'):
            _run_ydotool(['mousemove', '--absolute', str(x), str(y)])
            _mouse_pos['x'] = x
            _mouse_pos['y'] = y
    else:
        # Windows / macOS: pynput absolute positioning
        mouse = _get_pynput_mouse()
        if mouse:
            try:
                _mark_app_output()
                mouse.position = (x, y)
                _mouse_pos['x'] = x
                _mouse_pos['y'] = y
            except Exception:
                pass

def mouse_click(button=1):
    """Click mouse button (1=left, 2=middle, 3=right)"""
    if PLATFORM == 'macos' and not _check_macos_accessibility():
        return
    if is_app_input_paused():
        return
    if PLATFORM == 'linux':
        display = get_display_server()

        # X11: direct Xlib (fastest), fallback to pynput, then xdotool
        if display == 'x11':
            if _init_xlib() and _xlib_mouse_click(button):
                return
            # Fallback to pynput
            mouse = _get_pynput_mouse()
            if mouse:
                try:
                    from pynput.mouse import Button
                    btn_map = {1: Button.left, 2: Button.middle, 3: Button.right}
                    mouse.click(btn_map.get(button, Button.left))
                    return
                except Exception:
                    pass
            subprocess.run(['xdotool', 'click', str(button)], check=False)
            return

        # Wayland: use uinput or ydotool
        if _init_uinput() and _uinput_fd:
            btn_map = {1: _BTN_LEFT, 2: _BTN_MIDDLE, 3: _BTN_RIGHT}
            btn = btn_map.get(button, _BTN_LEFT)
            _write_uinput_event(_EV_KEY, btn, 1)  # Press
            _uinput_syn()
            _write_uinput_event(_EV_KEY, btn, 0)  # Release
            _uinput_syn()
            return

        # Fallback to ydotool
        tool, available = get_linux_input_tool()
        if not available:
            return

        if tool in ('ydotool', 'ydotool-sudo'):
            btn_map = {1: '1', 2: '3', 3: '2'}
            _run_ydotool(['click', btn_map.get(button, '1')])
    else:
        # Windows / macOS
        mouse = _get_pynput_mouse()
        if mouse:
            try:
                from pynput.mouse import Button
                btn_map = {1: Button.left, 2: Button.middle, 3: Button.right}
                _mark_app_output()
                mouse.click(btn_map.get(button, Button.left))
            except Exception:
                pass

def mouse_drag(action="down"):
    """Press or release left mouse button for dragging"""
    if PLATFORM == 'macos' and not _check_macos_accessibility():
        return
    if is_app_input_paused():
        return
    if PLATFORM == 'linux':
        display = get_display_server()

        # X11: direct Xlib (fastest), fallback to pynput, then xdotool
        if display == 'x11':
            if _init_xlib() and _xlib_mouse_drag(action):
                return
            # Fallback to pynput
            mouse = _get_pynput_mouse()
            if mouse:
                try:
                    from pynput.mouse import Button
                    if action == "down":
                        mouse.press(Button.left)
                    else:
                        mouse.release(Button.left)
                    return
                except Exception:
                    pass
            if action == "down":
                subprocess.run(['xdotool', 'mousedown', '1'], check=False)
            else:
                subprocess.run(['xdotool', 'mouseup', '1'], check=False)
            return

        # Wayland: uinput or ydotool
        if _init_uinput() and _uinput_fd:
            if action == "down":
                _write_uinput_event(_EV_KEY, _BTN_LEFT, 1)
            else:
                _write_uinput_event(_EV_KEY, _BTN_LEFT, 0)
            _uinput_syn()
            return

        # Fallback to ydotool for Wayland
        tool, available = get_linux_input_tool()
        if not available:
            return

        if tool in ('ydotool', 'ydotool-sudo'):
            # ydotool >= 1.0: 0x40 = BTN_LEFT down, 0x80 = BTN_LEFT up
            if action == "down":
                _run_ydotool(['click', '0x40'])
            else:
                _run_ydotool(['click', '0x80'])
    else:
        # Windows / macOS
        mouse = _get_pynput_mouse()
        if mouse:
            try:
                from pynput.mouse import Button
                _mark_app_output()
                if action == "down":
                    mouse.press(Button.left)
                else:
                    mouse.release(Button.left)
            except Exception:
                pass

_scroll_hires_accum = 0.0

def _uinput_smooth_scroll(dy):
    """Send hi-res + legacy scroll via uinput. REL_WHEEL emitted only at 120-unit boundaries."""
    global _scroll_hires_accum
    hires = int(dy * 120)
    if hires == 0:
        return
    _write_uinput_event(_EV_REL, _REL_WHEEL_HI_RES, hires)
    # Accumulate for legacy REL_WHEEL — emit only at notch boundaries (120 units)
    _scroll_hires_accum += hires
    notches = int(_scroll_hires_accum / 120)
    if notches != 0:
        _write_uinput_event(_EV_REL, _REL_WHEEL, notches)
        _scroll_hires_accum -= notches * 120
    _uinput_syn()

_win_scroll_accum = 0.0

def _win_smooth_scroll(dy):
    """Hi-res scroll on Windows via SendInput with sub-notch WHEEL_DELTA values."""
    global _win_scroll_accum
    if PLATFORM != 'windows':
        return False
    try:
        import ctypes
        from ctypes import wintypes

        INPUT_MOUSE = 0
        MOUSEEVENTF_WHEEL = 0x0800

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                        ("mouseData", wintypes.LONG), ("dwFlags", wintypes.DWORD),
                        ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_void_p)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("mi", MOUSEINPUT)]

        _win_scroll_accum += dy * 120  # 120 = WHEEL_DELTA = 1 notch
        scroll_amount = int(_win_scroll_accum)
        if scroll_amount == 0:
            return True  # accumulated but not enough yet
        _win_scroll_accum -= scroll_amount

        _mark_app_output()
        mi = MOUSEINPUT(0, 0, scroll_amount, MOUSEEVENTF_WHEEL, 0, None)
        inp = INPUT(INPUT_MOUSE, mi)
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        return True
    except Exception:
        return False

_mac_scroll_accum = 0.0

def _mac_smooth_scroll(dy):
    """Pixel-level scroll on macOS via Quartz CGEvent with fractional accumulator."""
    global _mac_scroll_accum
    if PLATFORM != 'macos':
        return False
    try:
        import Quartz
        _mac_scroll_accum += dy * 30
        pixel_delta = int(_mac_scroll_accum)
        if pixel_delta == 0:
            return True  # accumulated but not enough yet
        _mac_scroll_accum -= pixel_delta
        _mark_app_output()
        event = Quartz.CGEventCreateScrollWheelEvent(
            None,
            0,  # kCGScrollEventUnitPixel
            1,  # one axis (vertical)
            pixel_delta
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        return True
    except Exception:
        return False

_pynput_scroll_accum = 0.0

def _pynput_accum_scroll(dy):
    """pynput scroll with fractional accumulator — prevents momentum from dying abruptly."""
    global _pynput_scroll_accum
    mouse = _get_pynput_mouse()
    if not mouse:
        return
    _pynput_scroll_accum += dy * SCROLL_SPEED
    clicks = int(_pynput_scroll_accum)
    if clicks != 0:
        _pynput_scroll_accum -= clicks
        try:
            _mark_app_output()
            mouse.scroll(0, clicks)
        except Exception:
            pass

def mouse_scroll(dy):
    """Scroll mouse wheel (positive=up, negative=down). Accepts float for smooth scrolling."""
    if PLATFORM == 'macos' and not _check_macos_accessibility():
        return
    if is_app_input_paused():
        return
    if PLATFORM == 'linux':
        display = get_display_server()

        # Prefer uinput on BOTH X11 and Wayland — REL_WHEEL_HI_RES gives pixel-smooth scroll
        # (xf86-input-libinput picks up uinput devices on X11 too)
        if _init_uinput() and _uinput_fd:
            _uinput_smooth_scroll(dy)
            return

        # X11 fallback: Xlib button 4/5 (notchy but works without uinput)
        if display == 'x11':
            if _init_xlib() and _xlib_mouse_scroll(dy):
                return
            # Last resort: xdotool subprocess
            clicks = int(dy)
            if clicks == 0:
                return
            btn = '4' if clicks > 0 else '5'
            for _ in range(abs(clicks)):
                subprocess.run(['xdotool', 'click', btn], check=False)
            return

        # Wayland fallback: ydotool (if uinput failed above)
        tool, available = get_linux_input_tool()
        if not available:
            return

        if tool in ('ydotool', 'ydotool-sudo'):
            clicks = int(dy)
            if clicks == 0:
                return
            btn = '4' if clicks > 0 else '5'
            for _ in range(abs(clicks)):
                _run_ydotool(['click', btn])
    else:
        # Windows: native hi-res scroll via SendInput
        if PLATFORM == 'windows':
            if _win_smooth_scroll(dy):
                return
        # macOS: native pixel scroll via Quartz CGEvent
        elif PLATFORM == 'macos':
            if _mac_smooth_scroll(dy):
                return
        # Fallback: pynput with fractional accumulator
        _pynput_accum_scroll(dy)

# ============================================================================
#                           AUDIO UTILITIES
# ============================================================================

def list_input_devices():
    """List available audio input devices"""
    import sounddevice as sd
    print("\nAvailable input devices:\n")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            default = " (default)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {d['name']} - {int(d['default_samplerate'])}Hz{default}")
    print()

def get_device_info(device_id=None):
    """Get device info. Returns (device_id, sample_rate)"""
    import sounddevice as sd
    if device_id is None:
        device_id = sd.default.device[0]
    info = sd.query_devices(device_id)
    return device_id, int(info['default_samplerate'])

def resample(audio, orig_rate, target_rate, target_samples=None):
    """Resample audio to target rate/samples"""
    if orig_rate == target_rate and target_samples is None:
        return audio
    from scipy import signal
    if target_samples:
        resampled = signal.resample(audio, target_samples).astype(np.float32)
    else:
        samples = int(len(audio) * target_rate / orig_rate)
        resampled = signal.resample(audio, samples).astype(np.float32)
    return resampled

# ============================================================================
#                          AUDIO PREPROCESSING
# ============================================================================

from audio_preprocessing import (
    preprocess_audio_frame,
    preprocess_audio_buffer,
    AudioPreprocessingConfig,
    FilterState,
    get_config as get_preprocessing_config,
    set_config as set_preprocessing_config,
    enable_preprocessing,
    disable_preprocessing
)

# ============================================================================
#                          TRANSLITERATION
# ============================================================================

def has_devanagari(text):
    """Check if text contains Devanagari characters"""
    return bool(re.search(r'[\u0900-\u097F]', text))

def has_urdu(text):
    """Check if text contains Urdu/Arabic script"""
    return bool(re.search(r'[\u0600-\u06FF\u0750-\u077F]', text))

HINDI_VOWELS = {
    'ँ':'n', 'ं':'n', 'ः':'a', 'अ':'a', 'आ':'aa', 'इ':'i', 'ई':'ee', 'उ':'u',
    'ऊ':'oo', 'ऋ':'ri', 'ए':'e', 'ऐ':'ae', 'ओ':'o', 'औ':'au', 'ा':'a', 'ि':'i',
    'ी':'i', 'ु':'u', 'ू':'oo', 'ृ':'ri', 'े':'e', 'ै':'ai', 'ो':'o', 'ौ':'au'
}

HINDI_CONSONANTS = {
    'क्ष':'ksh', 'त्र':'tr', 'ज्ञ':'gy', 'क':'k', 'क़':'q', 'ख':'kh', 'ख़':'kh',
    'ग':'g', 'ग़':'gh', 'घ':'gh', 'ङ':'ng', 'च':'ch', 'छ':'chh', 'ज':'j', 'ज़':'z',
    'झ':'jh', 'ञ':'nj', 'ट':'t', 'ठ':'th', 'ड':'d', 'ड़':'r', 'ढ':'dh', 'ढ़':'dh',
    'ण':'n', 'त':'t', 'थ':'th', 'द':'d', 'ध':'dh', 'न':'n', 'प':'p', 'फ':'ph',
    'फ़':'f', 'ब':'b', 'भ':'bh', 'म':'m', 'य':'y', 'य़':'y', 'र':'r', 'ल':'l',
    'व':'v', 'श':'sh', 'ष':'sh', 'स':'s', 'ह':'h'
}

URDU_TO_ROMAN = {
    'ا': 'a', 'آ': 'aa', 'ب': 'b', 'پ': 'p', 'ت': 't', 'ٹ': 't',
    'ث': 's', 'ج': 'j', 'چ': 'ch', 'ح': 'h', 'خ': 'kh', 'د': 'd',
    'ڈ': 'd', 'ذ': 'z', 'ر': 'r', 'ڑ': 'r', 'ز': 'z', 'ژ': 'zh',
    'س': 's', 'ش': 'sh', 'ص': 's', 'ض': 'z', 'ط': 't', 'ظ': 'z',
    'ع': 'a', 'غ': 'gh', 'ف': 'f', 'ق': 'q', 'ک': 'k', 'گ': 'g',
    'ل': 'l', 'م': 'm', 'ن': 'n', 'ں': 'n', 'و': 'o', 'ہ': 'h',
    'ھ': 'h', 'ء': '', 'ی': 'i', 'ے': 'e', 'ئ': 'i',
    'َ': 'a', 'ِ': 'i', 'ُ': 'u', 'ّ': '', 'ْ': '',
}

def devanagari_to_roman(content):
    """Convert Devanagari to Roman with schwa deletion"""
    content = content.replace('ज़','ज़').replace('ड़','ड़').replace('फ़','फ़')
    content = content.replace('क\u093C','\u0958').replace('\u0916\u093C','ख़')
    content = content.replace('\u0917\u093C','ग़').replace('\u0922\u093C','ढ़').replace('य\u093C','\u095F')

    content2 = ''
    i = 0
    while i < len(content):
        if content[i] == '\u094D':
            content2 += content[i]
        elif i+1 < len(content):
            if content[i+1] == '\u094D':
                if i+3 < len(content) and content[i] in HINDI_CONSONANTS and content[i+3] in HINDI_CONSONANTS:
                    content2 += content[i] + content[i+1] + content[i+2] + 'a'
                elif i+2 < len(content):
                    content2 += content[i] + content[i+1] + content[i+2]
                else:
                    content2 += content[i] + content[i+1]
                i += 2 if i+2 < len(content) else 1
            elif content[i] in HINDI_CONSONANTS and content[i+1] in HINDI_CONSONANTS:
                if i == 0 or content[i-1] in [' ','\n','\t',':']:
                    content2 += content[i] + 'a'
                elif content[i-1] in HINDI_VOWELS or content[i-1] in HINDI_CONSONANTS:
                    if i+2 == len(content) or content[i+2] not in HINDI_VOWELS:
                        content2 += content[i] + 'a'
                    else:
                        content2 += content[i]
                else:
                    content2 += content[i] + 'a'
            elif content[i+1] == 'ा':
                if i+2 < len(content) and (content[i+2] in HINDI_CONSONANTS or content[i+2] == 'ँ'):
                    content2 += content[i] + 'a'
                else:
                    content2 += content[i]
            elif content[i+1] == 'ं':
                if i+2 < len(content) and content[i] in HINDI_CONSONANTS and content[i+2] in HINDI_CONSONANTS:
                    content2 += content[i] + 'a'
                else:
                    content2 += content[i]
            else:
                content2 += content[i]
        else:
            content2 += content[i]
        i += 1
    content = content2

    for vk, vv in HINDI_VOWELS.items():
        content = content.replace(vk, vv)
    for ck, cv in HINDI_CONSONANTS.items():
        content = content.replace(ck, cv)

    content = content.replace('\u094D', '')
    content = content.replace('।', '.')
    return content

def urdu_to_roman(text):
    """Convert Urdu script to Roman"""
    result = []
    for char in text:
        if char in URDU_TO_ROMAN:
            result.append(URDU_TO_ROMAN[char])
        elif char == ' ':
            result.append(' ')
        elif char.isascii():
            result.append(char)
    return ''.join(result)

def to_roman(text):
    """Convert any script to Roman (ASCII only output)"""
    if has_devanagari(text):
        text = devanagari_to_roman(text)
    elif has_urdu(text):
        text = urdu_to_roman(text)
    return ''.join(char for char in text if ord(char) < 128)
