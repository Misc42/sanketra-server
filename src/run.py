#!/usr/bin/env python3
"""
mic_on_term Interactive Launcher
Provides a polished interactive experience for mode and model selection.
"""

import sys
import os
import subprocess

# Get script directory and root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)  # Go up from src/ to root

# Platform detection
def get_platform():
    if sys.platform.startswith('linux'):
        return 'linux'
    elif sys.platform == 'win32':
        return 'windows'
    elif sys.platform == 'darwin':
        return 'macos'
    return 'unknown'

PLATFORM = get_platform()

# Colors - with Windows fallback
class C:
    if PLATFORM != 'windows':
        G = '\033[92m'   # green
        Y = '\033[93m'   # yellow
        R = '\033[91m'   # red
        D = '\033[90m'   # dim
        B = '\033[1m'    # bold
        CY = '\033[96m'  # cyan
        W = '\033[97m'   # white
        E = '\033[0m'    # end/reset
    else:
        G = Y = R = D = B = CY = W = E = ''

def clear_screen():
    """Clear terminal screen - ANSI escape works on Win10+ with VT processing"""
    print("\033[2J\033[H", end="", flush=True)

def get_venv_python():
    if PLATFORM == 'windows':
        return os.path.join(ROOT_DIR, 'venv', 'Scripts', 'python.exe')
    return os.path.join(ROOT_DIR, 'venv', 'bin', 'python3')

def get_venv_pythonw():
    """Get pythonw.exe path (Windows only, for background mode)"""
    if PLATFORM == 'windows':
        return os.path.join(ROOT_DIR, 'venv', 'Scripts', 'pythonw.exe')
    return get_venv_python()  # Non-Windows doesn't need this

def check_venv():
    venv_python = get_venv_python()
    if not os.path.exists(venv_python):
        print()
        print(f"  {C.R}Error:{C.E} Virtual environment not found.")
        print()
        print(f"  Run setup first:")
        print(f"    {C.CY}python3 setup.py{C.E}")
        print()
        sys.exit(1)

def get_gpu_compute_capability():
    """Get GPU compute capability. Returns (major, minor) or (0, 0)"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability(0)
    except Exception:
        pass
    return 0, 0

def gpu_supports_float16():
    """Check if GPU supports efficient float16 (compute capability >= 7.0)"""
    major, _ = get_gpu_compute_capability()
    return major >= 7

def get_hardware_info():
    """Get GPU and CPU/RAM info"""
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
        pynvml.nvmlShutdown()
    except Exception:
        pass

    cpu_name, cpu_cores, ram_total, ram_used = "Unknown", 0, 0, 0
    try:
        import platform
        import psutil
        cpu_name = platform.processor() or "Unknown CPU"
        cpu_cores = psutil.cpu_count()
        ram_total = psutil.virtual_memory().total / (1024**3)
        ram_used = psutil.virtual_memory().used / (1024**3)
    except Exception:
        pass

    # Check float16 support
    supports_fp16 = gpu_supports_float16() if vram_total > 0 else False

    return {
        "gpu_name": gpu_name,
        "vram_total": vram_total,
        "vram_used": vram_used,
        "vram_free": max(0, vram_total - vram_used),
        "has_gpu": vram_total > 0,
        "supports_fp16": supports_fp16,
        "cpu_name": cpu_name,
        "cpu_cores": cpu_cores,
        "ram_total": ram_total,
        "ram_used": ram_used,
        "ram_free": max(0, ram_total - ram_used),
    }

def draw_bar(used, total, width=25):
    """Draw ASCII progress bar"""
    if PLATFORM == 'windows':
        filled_char, empty_char = '#', '-'
    else:
        filled_char, empty_char = '█', '░'

    if total <= 0:
        return empty_char * width
    pct = min(1.0, used / total)
    filled = int(width * pct)
    return filled_char * filled + empty_char * (width - filled)

def show_header():
    clear_screen()
    print()
    if PLATFORM != 'windows':
        print(f"  {C.D}╔══════════════════════════════════════════════════════════╗{C.E}")
        print(f"  {C.D}║{C.E}                                                          {C.D}║{C.E}")
        print(f"  {C.D}║{C.E}   {C.B}mic_on_term{C.E}   {C.D}Voice-to-text for terminal{C.E}              {C.D}║{C.E}")
        print(f"  {C.D}║{C.E}                                                          {C.D}║{C.E}")
        print(f"  {C.D}╚══════════════════════════════════════════════════════════╝{C.E}")
    else:
        print(f"  +============================================================+")
        print(f"  |                                                            |")
        print(f"  |   {C.B}mic_on_term{C.E}   Voice-to-text for terminal                |")
        print(f"  |                                                            |")
        print(f"  +============================================================+")
    print()

def show_hardware_section(hw):
    """Show hardware info"""
    print(f"  {C.B}HARDWARE{C.E}")
    print(f"  {C.D}─────────────────────────────────────────────────────────{C.E}")

    if hw["has_gpu"]:
        # GPU info
        vram_pct = int(100 * hw["vram_used"] / hw["vram_total"]) if hw["vram_total"] > 0 else 0
        fp16_status = f"{C.G}fp16{C.E}" if hw.get("supports_fp16", False) else f"{C.Y}int8 only{C.E}"
        print(f"  {C.D}GPU:{C.E}  {hw['gpu_name'][:40]} ({fp16_status})")
        bar_now = draw_bar(hw["vram_used"], hw["vram_total"])
        print(f"  {C.D}VRAM:{C.E} {bar_now}  {hw['vram_used']:.1f} / {hw['vram_total']:.1f} GB ({vram_pct}%)")
    else:
        print(f"  {C.D}GPU:{C.E}  Not detected (CPU mode)")

    # RAM info
    print()
    ram_pct = int(100 * hw["ram_used"] / hw["ram_total"]) if hw["ram_total"] > 0 else 0
    print(f"  {C.D}CPU:{C.E}  {hw['cpu_name'][:45]}")
    bar_ram = draw_bar(hw["ram_used"], hw["ram_total"])
    print(f"  {C.D}RAM:{C.E}  {bar_ram}  {hw['ram_used']:.1f} / {hw['ram_total']:.1f} GB ({ram_pct}%)")

    print(f"  {C.D}─────────────────────────────────────────────────────────{C.E}")
    print()

def show_mode_selection():
    print(f"  {C.B}SELECT MODE{C.E}")
    print(f"  {C.D}─────────────────────────────────────────────────────────{C.E}")
    print()
    print(f"    {C.CY}1{C.E}  {C.W}CLI{C.E}   {C.D}─{C.E}  Local microphone (direct input)")
    print(f"                  {C.D}Use your computer's mic, text output to terminal{C.E}")
    print()
    print(f"    {C.CY}2{C.E}  {C.W}Web{C.E}   {C.D}─{C.E}  Phone mic + trackpad (via QR code)")
    print(f"                  {C.D}Use phone as wireless mic, touch as trackpad{C.E}")
    print()
    print(f"  {C.D}─────────────────────────────────────────────────────────{C.E}")
    print()

def get_choice(prompt, valid, default=None, allow_back=False):
    """Get user choice with validation"""
    back_hint = f" [{C.D}B=back{C.E}]" if allow_back else ""
    while True:
        try:
            choice = input(f"  {prompt}{back_hint}: ").strip().lower()
            if not choice and default:
                return default
            if allow_back and choice in ('b', 'back'):
                return 'back'
            if choice in valid:
                return choice
            # Check numeric
            if choice.isdigit() and int(choice) in [int(v) for v in valid if v.isdigit()]:
                return choice
        except (KeyboardInterrupt, EOFError):
            print()
            print()
            sys.exit(0)

        # Default fallback
        if default:
            return default
        print(f"  {C.D}Invalid choice, try again{C.E}")

def interactive_mode():
    """Interactive mode: pick cli or web, then launch.
    Model selection happens inside the Python app itself (interactive prompt)."""
    check_venv()

    while True:
        hw = get_hardware_info()
        show_header()
        show_hardware_section(hw)
        show_mode_selection()

        mode_choice = get_choice(
            f"{C.CY}Choice{C.E} [{C.D}1{C.E}/{C.D}2{C.E}]",
            ['1', '2', 'cli', 'web', 'q', 'quit'],
            default='2',
            allow_back=False
        )

        if mode_choice in ('q', 'quit'):
            print()
            print(f"  {C.D}Cancelled.{C.E}")
            print()
            sys.exit(0)

        if mode_choice in ('1', 'cli'):
            script = os.path.join(SCRIPT_DIR, 'stt_vad.py')
            mode_name = 'CLI'
            use_background = False
        else:
            script = os.path.join(SCRIPT_DIR, 'server_async.py')
            mode_name = 'Web'
            use_background = (PLATFORM == 'windows')

        # Launch — model selection happens inside the app
        show_header()
        print(f"  {C.D}Mode:{C.E} {C.G}{mode_name}{C.E}")
        print()
        print(f"  {C.D}─────────────────────────────────────────────────────────{C.E}")
        print()

        if use_background:
            print(f"  {C.Y}Starting in background mode...{C.E}")
            print(f"  {C.D}(Trackpad works when minimized){C.E}")
            print()
            print(f"  {C.D}QR code will open as image - scan with phone{C.E}")
            print(f"  {C.D}To stop: Task Manager > End pythonw.exe{C.E}")
            print()
            venv_pythonw = get_venv_pythonw()
            subprocess.Popen([venv_pythonw, script, '--background'],
                           creationflags=subprocess.CREATE_NO_WINDOW if PLATFORM == 'windows' else 0)
            import time
            time.sleep(3)
            print(f"  {C.G}Server started!{C.E} QR image should open automatically.")
            print()
        else:
            venv_python = get_venv_python()
            try:
                subprocess.run([venv_python, script])
            except KeyboardInterrupt:
                pass
        return

def show_help():
    print()
    print(f"  {C.B}Usage:{C.E}")
    print()
    print(f"  {C.D}Interactive (recommended):{C.E}")
    if PLATFORM == 'windows':
        print(f"    .\\run.bat              {C.D}# pick mode, then select model{C.E}")
    else:
        print(f"    ./run.sh               {C.D}# pick mode, then select model{C.E}")
    print()
    print(f"  {C.D}Direct:{C.E}")
    if PLATFORM == 'windows':
        print(f"    .\\run.bat cli          {C.D}# CLI mode (model selection on start){C.E}")
        print(f"    .\\run.bat web          {C.D}# Web mode (model selection on start){C.E}")
    else:
        print(f"    ./run.sh cli           {C.D}# CLI mode (model selection on start){C.E}")
        print(f"    ./run.sh web           {C.D}# Web mode (model selection on start){C.E}")
    print()
    print(f"  {C.D}CLI options:{C.E}")
    print(f"    -l              {C.D}List audio devices{C.E}")
    print(f"    -d <id>         {C.D}Use device ID{C.E}")
    print(f"    -t              {C.D}Type at cursor{C.E}")
    print()

def main():
    # If called directly with no args, run interactive mode
    if len(sys.argv) == 1:
        interactive_mode()
    elif sys.argv[1] in ('-h', '--help', 'help'):
        show_help()
    else:
        # Pass through to appropriate script
        mode = sys.argv[1]
        args = sys.argv[2:]

        if mode == 'cli':
            script = os.path.join(SCRIPT_DIR, 'stt_vad.py')
            use_background = False
        elif mode == 'web':
            script = os.path.join(SCRIPT_DIR, 'server_async.py')
            # Windows needs background mode for trackpad to work
            use_background = (PLATFORM == 'windows')
        else:
            print(f"Unknown mode: {mode}")
            print("Use 'cli' or 'web', or run without args for interactive mode")
            sys.exit(1)

        check_venv()

        if use_background:
            # Windows web mode - use pythonw.exe with --background
            print()
            print(f"  Starting in background mode...")
            print(f"  (Trackpad works when minimized)")
            print()
            print(f"  QR code will open as image - scan with phone")
            print(f"  To stop: Task Manager > End pythonw.exe")
            print()
            venv_pythonw = get_venv_pythonw()
            subprocess.Popen([venv_pythonw, script, '--background'] + args,
                           creationflags=subprocess.CREATE_NO_WINDOW if PLATFORM == 'windows' else 0)
            import time
            time.sleep(3)
            print(f"  Server started! QR image should open automatically.")
            print()
        else:
            venv_python = get_venv_python()
            subprocess.run([venv_python, script] + args)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        sys.exit(0)
