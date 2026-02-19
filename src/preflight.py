#!/usr/bin/env python3
"""
Preflight checks for mic_on_term.

This module runs before the application starts to verify dependencies,
detect hardware capabilities, and select optimal configurations.

Philosophy:
- Never crash silently
- Degrade gracefully when optional features are missing
- Exit early with clear error messages when required features are missing
- Print a startup summary showing what's available
"""

import sys
import os
import shutil
import subprocess
import importlib
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any, Tuple

# Minimum Python version
MIN_PYTHON = (3, 8)


# =============================================================================
# Dependency Classification
# =============================================================================

class DepType(Enum):
    REQUIRED = "required"        # App cannot run without this
    OPTIONAL = "optional"        # Feature disabled if missing
    ACCELERATION = "acceleration"  # Performance boost if present


@dataclass
class Dependency:
    name: str
    dep_type: DepType
    import_name: str
    check_func: Optional[Callable] = None
    fallback: Optional[str] = None
    install_hint: Optional[str] = None
    platforms: Optional[List[str]] = None  # None = all platforms


@dataclass
class DependencyStatus:
    name: str
    dep_type: DepType
    available: bool
    version: Optional[str] = None
    error: Optional[str] = None


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class TranscriptionConfig:
    device: str  # "cuda" or "cpu"
    model: str   # "tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"
    compute_type: str  # "float16", "int8_float16", "int8"
    min_vram_gb: float = 0.0


@dataclass
class VADConfig:
    type: str  # "silero" or "energy"
    model: Optional[str] = None
    threshold: float = 0.02


@dataclass
class AudioConfig:
    device_id: Optional[int] = None
    device_name: str = "default"
    sample_rate: int = 16000


@dataclass
class ServerConfig:
    type: str  # "fastapi" or "flask"
    module: str


@dataclass
class InputConfig:
    type: Optional[str] = None  # "uinput", "xdotool", "ydotool", "pynput", "pyautogui", None
    display: Optional[str] = None
    needs_daemon: bool = False
    needs_sudo: bool = False


@dataclass
class SSLConfig:
    type: str  # "generate", "existing", "none"
    cert_path: Optional[str] = None
    key_path: Optional[str] = None


# =============================================================================
# Preflight Result
# =============================================================================

@dataclass
class PreflightResult:
    """Complete results from preflight checks"""

    # Platform info
    platform: str = "unknown"
    display_server: str = "unknown"
    python_version: str = ""
    arch: str = ""

    # Hardware
    has_gpu: bool = False
    gpu_name: str = ""
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    supports_float16: bool = False
    cpu_cores: int = 0
    ram_gb: float = 0.0

    has_audio_input: bool = False
    audio_device_name: str = ""
    audio_device_id: Optional[int] = None
    audio_sample_rate: int = 16000

    # Dependencies
    dependencies: Dict[str, DependencyStatus] = field(default_factory=dict)

    # Selected configurations
    transcription: Optional[TranscriptionConfig] = None
    vad: Optional[VADConfig] = None
    audio: Optional[AudioConfig] = None
    server: Optional[ServerConfig] = None
    input_control: Optional[InputConfig] = None
    ssl: Optional[SSLConfig] = None

    # Messages
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def has_required_failures(self) -> bool:
        for status in self.dependencies.values():
            if status.dep_type == DepType.REQUIRED and not status.available:
                return True
        return False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_error(self, msg: str):
        self.errors.append(msg)


# =============================================================================
# Dependency Registry
# =============================================================================

DEPENDENCIES = [
    # === REQUIRED ===
    Dependency(
        name="numpy",
        dep_type=DepType.REQUIRED,
        import_name="numpy",
        install_hint="pip install numpy"
    ),
    Dependency(
        name="torch",
        dep_type=DepType.REQUIRED,
        import_name="torch",
        install_hint="pip install torch"
    ),
    Dependency(
        name="faster-whisper",
        dep_type=DepType.REQUIRED,
        import_name="faster_whisper",
        install_hint="pip install faster-whisper"
    ),
    Dependency(
        name="sounddevice",
        dep_type=DepType.OPTIONAL,
        import_name="sounddevice",
        fallback="Local audio input unavailable (phone mic still works via WebSocket)",
        install_hint="pip install sounddevice (requires libportaudio2 on Linux)"
    ),

    # === OPTIONAL ===
    Dependency(
        name="flask",
        dep_type=DepType.OPTIONAL,
        import_name="flask",
        fallback="Flask web server unavailable",
        install_hint="pip install flask flask-sock"
    ),
    Dependency(
        name="flask-sock",
        dep_type=DepType.OPTIONAL,
        import_name="flask_sock",
        fallback="Flask WebSocket unavailable",
        install_hint="pip install flask-sock"
    ),
    Dependency(
        name="fastapi",
        dep_type=DepType.OPTIONAL,
        import_name="fastapi",
        fallback="FastAPI server unavailable, use Flask",
        install_hint="pip install fastapi"
    ),
    Dependency(
        name="uvicorn",
        dep_type=DepType.OPTIONAL,
        import_name="uvicorn",
        fallback="uvicorn unavailable, use Flask",
        install_hint="pip install uvicorn[standard]"
    ),
    Dependency(
        name="qrcode",
        dep_type=DepType.OPTIONAL,
        import_name="qrcode",
        fallback="QR code generation disabled - print URL only",
        install_hint="pip install qrcode"
    ),
    Dependency(
        name="cryptography",
        dep_type=DepType.OPTIONAL,
        import_name="cryptography",
        fallback="SSL cert generation disabled",
        install_hint="pip install cryptography"
    ),
    Dependency(
        name="pyautogui",
        dep_type=DepType.OPTIONAL,
        import_name="pyautogui",
        fallback="Typing at cursor disabled",
        install_hint="pip install pyautogui",
        platforms=["windows", "macos"]
    ),
    Dependency(
        name="pynput",
        dep_type=DepType.OPTIONAL,
        import_name="pynput",
        fallback="Mouse control disabled",
        install_hint="pip install pynput",
        platforms=["windows", "macos"]
    ),

    # === ACCELERATION ===
    Dependency(
        name="nvidia-ml-py",
        dep_type=DepType.ACCELERATION,
        import_name="pynvml",
        fallback="GPU monitoring disabled - model selection may be less accurate",
        install_hint="pip install nvidia-ml-py"
    ),
    Dependency(
        name="psutil",
        dep_type=DepType.ACCELERATION,
        import_name="psutil",
        fallback="System info unavailable - using defaults",
        install_hint="pip install psutil"
    ),
    Dependency(
        name="scipy",
        dep_type=DepType.ACCELERATION,
        import_name="scipy",
        fallback="Audio resampling may be slower",
        install_hint="pip install scipy"
    ),
]


# =============================================================================
# Platform Detection
# =============================================================================

def get_platform() -> str:
    """Returns 'linux', 'windows', 'macos', or 'unknown'"""
    if sys.platform.startswith('linux'):
        return 'linux'
    elif sys.platform == 'win32':
        return 'windows'
    elif sys.platform == 'darwin':
        return 'macos'
    return 'unknown'


def get_display_server() -> str:
    """Returns 'x11', 'wayland', or 'n/a'"""
    if get_platform() != 'linux':
        return 'n/a'

    session_type = os.environ.get('XDG_SESSION_TYPE', '').lower()
    if session_type in ('x11', 'wayland'):
        return session_type

    if os.environ.get('WAYLAND_DISPLAY'):
        return 'wayland'
    if os.environ.get('DISPLAY'):
        return 'x11'

    return 'unknown'


def get_arch() -> str:
    """Returns architecture like 'x86_64', 'arm64', etc."""
    import platform
    return platform.machine()


# =============================================================================
# Dependency Checking
# =============================================================================

def check_dependency(dep: Dependency, result: PreflightResult) -> bool:
    """Check a single dependency, return True if available"""
    try:
        module = importlib.import_module(dep.import_name)

        # Get version - handle deprecation warnings
        version = 'unknown'
        try:
            version = getattr(module, '__version__', None)
            if version is None:
                # Try importlib.metadata as fallback
                try:
                    from importlib.metadata import version as get_version
                    version = get_version(dep.name)
                except Exception:
                    version = 'unknown'
        except Exception:
            version = 'unknown'

        # Run custom check if provided
        if dep.check_func and not dep.check_func(module):
            raise ImportError("Custom check failed")

        result.dependencies[dep.name] = DependencyStatus(
            name=dep.name,
            dep_type=dep.dep_type,
            available=True,
            version=str(version) if version else 'unknown'
        )
        return True

    except ImportError as e:
        result.dependencies[dep.name] = DependencyStatus(
            name=dep.name,
            dep_type=dep.dep_type,
            available=False,
            error=str(e)
        )

        if dep.dep_type == DepType.REQUIRED:
            result.add_error(f"Required: {dep.name} not installed. {dep.install_hint or ''}")
        elif dep.fallback:
            result.add_warning(f"{dep.name} not available: {dep.fallback}")

        return False


def should_check_dep(dep: Dependency, platform: str) -> bool:
    """Check if dependency is relevant for this platform"""
    if dep.platforms is None:
        return True
    return platform in dep.platforms


# =============================================================================
# Hardware Detection
# =============================================================================

def check_gpu(result: PreflightResult):
    """Detect GPU capabilities"""
    try:
        import torch
        if torch.cuda.is_available():
            result.has_gpu = True
            result.gpu_name = torch.cuda.get_device_name(0)

            # Get VRAM
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                result.vram_total_gb = info.total / (1024**3)
                result.vram_free_gb = info.free / (1024**3)
                pynvml.nvmlShutdown()
            except Exception:
                # Fallback to torch
                props = torch.cuda.get_device_properties(0)
                result.vram_total_gb = props.total_memory / (1024**3)
                result.vram_free_gb = result.vram_total_gb * 0.9  # Estimate

            # Check float16 support (compute capability >= 7.0)
            try:
                cap = torch.cuda.get_device_capability(0)
                result.supports_float16 = cap[0] >= 7
            except Exception:
                result.supports_float16 = False

    except Exception as e:
        result.has_gpu = False
        result.add_warning(f"GPU detection failed: {e}")


def check_system_info(result: PreflightResult):
    """Get CPU and RAM info"""
    try:
        import psutil
        result.cpu_cores = psutil.cpu_count(logical=True)
        result.ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        result.cpu_cores = os.cpu_count() or 1
        result.ram_gb = 0


def check_audio_devices(result: PreflightResult):
    """Check for audio input devices"""
    try:
        import sounddevice as sd
        devices = sd.query_devices()

        # Try default input first
        try:
            default_input = sd.default.device[0]
            if default_input is not None and default_input >= 0:
                dev = devices[default_input]
                if dev['max_input_channels'] > 0:
                    result.has_audio_input = True
                    result.audio_device_id = default_input
                    result.audio_device_name = dev['name']
                    result.audio_sample_rate = int(dev['default_samplerate'])
                    return
        except Exception:
            pass

        # Find any input device
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                result.has_audio_input = True
                result.audio_device_id = i
                result.audio_device_name = d['name']
                result.audio_sample_rate = int(d['default_samplerate'])
                return

        result.has_audio_input = False
        result.add_warning("No audio input device found")

    except Exception as e:
        result.has_audio_input = False
        result.add_warning(f"Audio device detection failed: {e}")


# =============================================================================
# Input Tool Detection (Linux)
# =============================================================================

def check_uinput_access() -> bool:
    """Check if /dev/uinput is writable"""
    return os.access('/dev/uinput', os.W_OK)


def check_ydotool_daemon() -> bool:
    """Check if ydotoold is running"""
    try:
        result = subprocess.run(['pgrep', '-x', 'ydotoold'],
                               capture_output=True, timeout=2)
        return result.returncode == 0
    except Exception:
        return False


def select_linux_input(display_server: str, result: PreflightResult) -> InputConfig:
    """Select best input method for Linux"""

    # Try direct uinput first (fastest)
    if check_uinput_access():
        return InputConfig(type="uinput", display=display_server)

    if display_server == 'x11':
        if shutil.which('xdotool'):
            return InputConfig(type="xdotool", display="x11")
        result.add_warning("xdotool not found - trackpad/typing disabled on X11")
        result.add_warning("  Fix: sudo apt install xdotool")

    elif display_server == 'wayland':
        if shutil.which('ydotool'):
            if check_ydotool_daemon():
                return InputConfig(type="ydotool", display="wayland", needs_daemon=True)
            else:
                result.add_warning("ydotool found but daemon not running")
                result.add_warning("  Fix: sudo systemctl start ydotoold")
                # Can still work with sudo
                return InputConfig(type="ydotool", display="wayland", needs_sudo=True)
        else:
            result.add_warning("ydotool not found - trackpad/typing disabled on Wayland")
            result.add_warning("  Fix: sudo apt install ydotool && sudo systemctl start ydotoold")

    return InputConfig(type=None)


def _check_macos_accessibility() -> bool:
    """Check if macOS Accessibility permission is granted for pynput to work."""
    try:
        import Quartz
        # CGRequestPostEventAccess() returns True if we can post events
        return Quartz.CGRequestPostEventAccess()
    except ImportError:
        # Quartz not available — try osascript probe
        try:
            result = subprocess.run(
                ['osascript', '-e', 'tell application "System Events" to keystroke ""'],
                capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return True  # Can't check, assume OK


def select_windows_macos_input(platform: str, result: PreflightResult) -> InputConfig:
    """Select input method for Windows/macOS"""

    pynput_ok = result.dependencies.get('pynput', DependencyStatus('', DepType.OPTIONAL, False)).available
    pyautogui_ok = result.dependencies.get('pyautogui', DependencyStatus('', DepType.OPTIONAL, False)).available

    if pynput_ok:
        # macOS: check Accessibility permission (pynput silently fails without it)
        if platform == 'macos' and not _check_macos_accessibility():
            result.add_warning("macOS Accessibility permission NOT granted — input will fail!")
            result.add_warning("  Fix: System Settings > Privacy & Security > Accessibility")
            result.add_warning("  Add your terminal app (Terminal, iTerm, VS Code) to allowed list")
        return InputConfig(type="pynput")
    elif pyautogui_ok:
        result.add_warning("pynput not available - using pyautogui (keyboard only)")
        return InputConfig(type="pyautogui")
    else:
        result.add_warning(f"No input control available on {platform}")
        result.add_warning("  Fix: pip install pynput pyautogui")
        return InputConfig(type=None)


# =============================================================================
# Feature Selection
# =============================================================================

def select_transcription(result: PreflightResult, preference: str = "balanced") -> TranscriptionConfig:
    """Select best transcription configuration"""

    vram = result.vram_free_gb if result.has_gpu else 0

    # Define model requirements for faster-whisper (CTranslate2) with float16
    # These are actual measured values, NOT original OpenAI Whisper estimates
    models = {
        "large-v3": {"vram": 3, "cpu_ram": 6},        # ~3 GB actual
        "large-v3-turbo": {"vram": 1.5, "cpu_ram": 4}, # ~1.5 GB actual
        "medium": {"vram": 1.5, "cpu_ram": 3},        # ~1.5 GB actual
        "small": {"vram": 1, "cpu_ram": 2},           # ~1 GB actual
        "base": {"vram": 0.5, "cpu_ram": 1},          # ~0.5 GB actual
        "tiny": {"vram": 0.5, "cpu_ram": 1},          # ~0.5 GB actual
    }

    if result.has_gpu:
        device = "cuda"
        compute = "float16" if result.supports_float16 else "int8"

        if preference == "fast":
            if vram >= 2:
                return TranscriptionConfig(device, "large-v3-turbo", compute)
            elif vram >= 1.5:
                return TranscriptionConfig(device, "small", compute)
            else:
                return TranscriptionConfig(device, "tiny", compute)

        elif preference == "accurate":
            if vram >= 3.5:
                return TranscriptionConfig(device, "large-v3", compute)
            elif vram >= 2:
                return TranscriptionConfig(device, "large-v3-turbo", compute)
            elif vram >= 1.5:
                return TranscriptionConfig(device, "medium", compute)
            else:
                return TranscriptionConfig("cpu", "medium", "int8")

        else:  # balanced
            if vram >= 3.5:
                return TranscriptionConfig(device, "large-v3", compute)
            elif vram >= 2:
                return TranscriptionConfig(device, "large-v3-turbo", compute)
            elif vram >= 1.5:
                return TranscriptionConfig(device, "small", compute)
            else:
                return TranscriptionConfig(device, "base", compute)

    else:
        # CPU fallback
        device = "cpu"
        compute = "int8"
        result.add_warning("No GPU detected - transcription will be slow")

        if preference == "fast":
            return TranscriptionConfig(device, "tiny", compute)
        elif preference == "accurate":
            return TranscriptionConfig(device, "medium", compute)
        else:  # balanced
            if result.ram_gb >= 4:
                return TranscriptionConfig(device, "small", compute)
            else:
                return TranscriptionConfig(device, "base", compute)


def select_vad(result: PreflightResult) -> VADConfig:
    """Select VAD configuration"""
    # Silero VAD is preferred, energy-based is fallback
    # The actual loading and fallback happens in stt_common.load_vad()
    if result.dependencies.get('torch', DependencyStatus('', DepType.REQUIRED, False)).available:
        return VADConfig(type="silero", model="silero_vad")
    else:
        return VADConfig(type="energy", threshold=0.02)


def select_server(result: PreflightResult, mode: str) -> Optional[ServerConfig]:
    """Select web server configuration"""
    if mode == "cli":
        return None

    fastapi_ok = result.dependencies.get('fastapi', DependencyStatus('', DepType.OPTIONAL, False)).available
    uvicorn_ok = result.dependencies.get('uvicorn', DependencyStatus('', DepType.OPTIONAL, False)).available
    flask_ok = result.dependencies.get('flask', DependencyStatus('', DepType.OPTIONAL, False)).available
    flask_sock_ok = result.dependencies.get('flask-sock', DependencyStatus('', DepType.OPTIONAL, False)).available

    if fastapi_ok and uvicorn_ok:
        return ServerConfig(type="fastapi", module="server_async")
    elif flask_ok and flask_sock_ok:
        result.add_warning("FastAPI not available - using Flask (may have WebSocket issues)")
        return ServerConfig(type="flask", module="stt_web")
    else:
        result.add_error("No web server available. Install: pip install fastapi uvicorn[standard]")
        return None


def select_ssl(result: PreflightResult, script_dir: str) -> SSLConfig:
    """Select SSL configuration"""
    cert_path = os.path.join(script_dir, 'cert.pem')
    key_path = os.path.join(script_dir, 'key.pem')

    # Check for existing certs
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return SSLConfig(type="existing", cert_path=cert_path, key_path=key_path)

    # Check if we can generate
    crypto_ok = result.dependencies.get('cryptography', DependencyStatus('', DepType.OPTIONAL, False)).available
    if crypto_ok:
        return SSLConfig(type="generate")

    result.add_warning("SSL unavailable - HTTPS disabled")
    result.add_warning("  Phone microphone will NOT work (browsers require HTTPS)")
    result.add_warning("  Fix: pip install cryptography")
    return SSLConfig(type="none")


# =============================================================================
# Startup Summary
# =============================================================================

def print_summary(result: PreflightResult, mode: str):
    """Print startup summary"""

    print()
    print("=" * 70)
    print("                    mic_on_term PREFLIGHT CHECK")
    print("=" * 70)
    print()

    # Platform
    print("  PLATFORM")
    print("  " + "-" * 8)
    print(f"  OS:              {result.platform} {result.arch}")
    if result.platform == 'linux':
        print(f"  Display:         {result.display_server}")
    print(f"  Python:          {result.python_version}")
    print()

    # Hardware
    print("  HARDWARE")
    print("  " + "-" * 8)
    if result.has_gpu:
        print(f"  GPU:             {result.gpu_name}")
        print(f"  VRAM:            {result.vram_free_gb:.1f} GB free / {result.vram_total_gb:.1f} GB total")
        print(f"  Float16:         {'Supported' if result.supports_float16 else 'Not supported'}")
    else:
        print("  GPU:             Not detected (CPU mode)")

    if result.has_audio_input:
        print(f"  Audio:           {result.audio_device_name} @ {result.audio_sample_rate}Hz")
    else:
        print("  Audio:           No input device")
    print()

    # Dependencies
    print("  DEPENDENCIES")
    print("  " + "-" * 12)

    required = [d for d in result.dependencies.values() if d.dep_type == DepType.REQUIRED]
    optional = [d for d in result.dependencies.values() if d.dep_type == DepType.OPTIONAL]
    accel = [d for d in result.dependencies.values() if d.dep_type == DepType.ACCELERATION]

    # Required
    ok_req = [d for d in required if d.available]
    fail_req = [d for d in required if not d.available]
    if ok_req:
        print(f"  [OK] {', '.join(d.name for d in ok_req)}")
    for d in fail_req:
        print(f"  [XX] {d.name} - NOT INSTALLED")

    # Optional
    ok_opt = [d for d in optional if d.available]
    fail_opt = [d for d in optional if not d.available]
    if ok_opt:
        print(f"  [OK] {', '.join(d.name for d in ok_opt)}")
    for d in fail_opt:
        print(f"  [!!] {d.name} - not installed")

    # Acceleration
    ok_acc = [d for d in accel if d.available]
    fail_acc = [d for d in accel if not d.available]
    if ok_acc:
        print(f"  [OK] {', '.join(d.name for d in ok_acc)} (acceleration)")
    for d in fail_acc:
        print(f"  [--] {d.name} - not installed (optional)")

    print()

    # Features
    print("  FEATURES")
    print("  " + "-" * 8)

    if result.transcription:
        t = result.transcription
        speed_note = "[SLOW]" if t.device == "cpu" else ""
        print(f"  Transcription:   {t.model} ({t.compute_type}) on {t.device.upper()} {speed_note}")
    else:
        print("  Transcription:   DISABLED")

    if result.vad:
        print(f"  VAD:             {result.vad.type.capitalize()}")

    if mode == "web" and result.server:
        print(f"  Server:          {result.server.type.capitalize()}")

    if result.input_control and result.input_control.type:
        ic = result.input_control
        extra = ""
        if ic.needs_daemon:
            extra = " (daemon)"
        elif ic.needs_sudo:
            extra = " (needs sudo)"
        print(f"  Input:           {ic.type}{extra}")
    else:
        print("  Input:           DISABLED")

    if mode == "web" and result.ssl:
        print(f"  SSL:             {result.ssl.type.capitalize()}")

    print()

    # Warnings
    if result.warnings:
        print(f"  WARNINGS ({len(result.warnings)})")
        print("  " + "-" * 8)
        for w in result.warnings:
            print(f"  {w}")
        print()

    # Status
    print("=" * 70)
    if result.has_required_failures():
        print("  STATUS: FATAL - Missing required dependencies")
    elif result.warnings:
        print("  STATUS: DEGRADED - Running with reduced capabilities")
    else:
        print("  STATUS: READY - All systems operational")
    print("=" * 70)
    print()


# =============================================================================
# Main Preflight Function
# =============================================================================

def run(mode: str = "cli", preference: str = "balanced",
        verbose: bool = True, script_dir: str = None) -> PreflightResult:
    """
    Run all preflight checks.

    Args:
        mode: "cli" or "web"
        preference: "fast", "balanced", or "accurate"
        verbose: Print status messages
        script_dir: Directory containing the scripts (for SSL certs)

    Returns:
        PreflightResult with all capability flags and configurations
    """
    result = PreflightResult()

    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)  # Go up from src/ to root

    # === Phase 1: Python version ===
    result.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info < MIN_PYTHON:
        print(f"FATAL: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required. You have {result.python_version}")
        sys.exit(1)

    # === Phase 2: Platform detection ===
    result.platform = get_platform()
    result.display_server = get_display_server()
    result.arch = get_arch()

    # === Phase 3: Check REQUIRED dependencies ===
    for dep in DEPENDENCIES:
        if dep.dep_type == DepType.REQUIRED:
            if should_check_dep(dep, result.platform):
                check_dependency(dep, result)

    # Exit early if required deps missing
    if result.has_required_failures():
        if verbose:
            print_summary(result, mode)
            print("\nTo fix, run:")
            for status in result.dependencies.values():
                if status.dep_type == DepType.REQUIRED and not status.available:
                    dep = next((d for d in DEPENDENCIES if d.name == status.name), None)
                    if dep and dep.install_hint:
                        print(f"  {dep.install_hint}")
            print("\nOr re-run setup:")
            print("  python3 setup.py")
        sys.exit(1)

    # === Phase 4: Hardware detection ===
    check_gpu(result)
    check_system_info(result)
    check_audio_devices(result)

    # === Phase 5: Check OPTIONAL dependencies ===
    for dep in DEPENDENCIES:
        if dep.dep_type == DepType.OPTIONAL:
            if should_check_dep(dep, result.platform):
                check_dependency(dep, result)

    # === Phase 6: Check ACCELERATION dependencies ===
    for dep in DEPENDENCIES:
        if dep.dep_type == DepType.ACCELERATION:
            if should_check_dep(dep, result.platform):
                check_dependency(dep, result)

    # === Phase 7: Select configurations ===
    result.transcription = select_transcription(result, preference)
    result.vad = select_vad(result)

    if mode == "web":
        result.server = select_server(result, mode)
        result.ssl = select_ssl(result, root_dir)

        # Check server availability
        if result.server is None:
            if verbose:
                print_summary(result, mode)
            sys.exit(1)

    # Input control
    if result.platform == 'linux':
        result.input_control = select_linux_input(result.display_server, result)
    elif result.platform in ('windows', 'macos'):
        result.input_control = select_windows_macos_input(result.platform, result)
    else:
        result.input_control = InputConfig(type=None)

    # Audio config
    if result.has_audio_input:
        result.audio = AudioConfig(
            device_id=result.audio_device_id,
            device_name=result.audio_device_name,
            sample_rate=result.audio_sample_rate
        )

    # === Phase 8: Print summary ===
    if verbose:
        print_summary(result, mode)

    return result


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run preflight checks")
    parser.add_argument("--mode", choices=["cli", "web"], default="cli")
    parser.add_argument("--preference", choices=["fast", "balanced", "accurate"], default="balanced")
    args = parser.parse_args()

    result = run(mode=args.mode, preference=args.preference)

    # Exit with appropriate code
    if result.has_required_failures():
        sys.exit(1)
    elif result.warnings:
        sys.exit(0)  # Degraded but functional
    else:
        sys.exit(0)  # All good
