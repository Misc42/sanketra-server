#!/usr/bin/env python3
"""
Sanketra Setup Script
Usage: python3 setup.py [--uninstall]
"""

import sys
import os
import subprocess
import shutil
import platform
import glob
import time

# Force UTF-8 output on Windows (default codepage can't encode all pip/package output)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(SCRIPT_DIR, 'venv')

# Colors and symbols
class C:
    if platform.system() != 'Windows':
        G = '\033[92m'   # green
        Y = '\033[93m'   # yellow
        R = '\033[91m'   # red
        D = '\033[90m'   # dim
        B = '\033[1m'    # bold
        C = '\033[96m'   # cyan
        E = '\033[0m'    # end
        OK = '✓'         # checkmark
        X = '✗'          # cross
    else:
        G = Y = R = D = B = C = E = ''
        OK = 'OK'        # ASCII fallback
        X = 'X'          # ASCII fallback

def run(cmd, shell=True, capture=True):
    try:
        r = subprocess.run(cmd, shell=shell, capture_output=capture, text=True)
        return r.returncode == 0, r.stdout.strip() if r.stdout else ''
    except:
        return False, ''

def get_platform():
    s = platform.system().lower()
    return {'linux': 'linux', 'windows': 'windows', 'darwin': 'macos'}.get(s, 'unknown')

def get_venv_python():
    if get_platform() == 'windows':
        return os.path.join(VENV_DIR, 'Scripts', 'python.exe')
    return os.path.join(VENV_DIR, 'bin', 'python3')

def get_venv_pip():
    if get_platform() == 'windows':
        return os.path.join(VENV_DIR, 'Scripts', 'pip.exe')
    return os.path.join(VENV_DIR, 'bin', 'pip')

def format_time(seconds):
    if seconds < 60:
        return f"{int(seconds)}s"
    return f"{int(seconds//60)}m {int(seconds%60)}s"

def format_size(bytes):
    if bytes < 1024:
        return f"{bytes}B"
    elif bytes < 1024*1024:
        return f"{bytes/1024:.1f}KB"
    elif bytes < 1024*1024*1024:
        return f"{bytes/(1024*1024):.1f}MB"
    return f"{bytes/(1024*1024*1024):.2f}GB"

# =============================================================================
#                              SETUP STEPS
# =============================================================================

def step_python():
    v = sys.version_info
    if v.major == 3 and v.minor >= 10:
        return True, f"{v.major}.{v.minor}.{v.micro}"
    return False, f"{v.major}.{v.minor} (need 3.10+)"

def step_venv():
    if os.path.exists(get_venv_python()):
        return True, "exists"
    try:
        import venv
        venv.create(VENV_DIR, with_pip=True)
        return True, "created"
    except:
        return False, "failed"

def step_gpu():
    # Windows uses 2>nul, Linux/Mac use 2>/dev/null
    if get_platform() == 'windows':
        cmd = "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>nul"
    else:
        cmd = "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null"
    ok, out = run(cmd)
    if ok and out:
        parts = out.split(',')
        name = parts[0].strip()
        vram = f"{int(parts[1].strip())//1024}GB" if len(parts) > 1 else ""
        return True, f"{name} ({vram})"
    return True, "CPU mode (no GPU)"

def get_display_server():
    """Detect Linux display server: x11, wayland, or unknown"""
    if get_platform() != 'linux':
        return 'n/a'
    session = os.environ.get('XDG_SESSION_TYPE', '').lower()
    if session in ('x11', 'wayland'):
        return session
    if os.environ.get('WAYLAND_DISPLAY'):
        return 'wayland'
    if os.environ.get('DISPLAY'):
        return 'x11'
    return 'unknown'

def step_system_deps():
    plat = get_platform()
    if plat == 'linux':
        has_portaudio, _ = run("dpkg -l | grep -q libportaudio2")
        display = get_display_server()

        installed = []
        warnings = []

        # Install portaudio
        if not has_portaudio:
            subprocess.run(['sudo', 'apt', 'install', '-y', 'libportaudio2', 'portaudio19-dev'],
                           capture_output=True)
            installed.append("portaudio")

        # Install input tools based on display server
        if display == 'wayland':
            has_ydotool = shutil.which('ydotool') is not None
            if not has_ydotool:
                print(f"\n  {C.Y}Wayland detected - installing ydotool (may ask for password)...{C.E}\n")
                subprocess.run(['sudo', 'apt', 'install', '-y', 'ydotool'])
                has_ydotool = shutil.which('ydotool') is not None

            if not has_ydotool:
                return False, "ydotool required for Wayland but install failed (sudo apt install ydotool)"

            installed.append("ydotool")

            # Check ydotool version - old versions need uinput access, new need daemon
            has_daemon = shutil.which('ydotoold') is not None

            if has_daemon:
                # New ydotool - start daemon
                daemon_running, _ = run("pgrep -x ydotoold")
                if not daemon_running:
                    print(f"\n  {C.D}├─ Starting ydotool daemon...{C.E}")
                    subprocess.run(['sudo', 'systemctl', 'enable', 'ydotoold'])
                    subprocess.run(['sudo', 'systemctl', 'start', 'ydotoold'])
                    time.sleep(0.5)
                    daemon_running, _ = run("pgrep -x ydotoold")
                    if daemon_running:
                        installed.append("daemon started")
                    else:
                        return False, "ydotoold daemon failed to start (sudo systemctl start ydotoold)"
            else:
                # Old ydotool (0.x) - needs uinput access
                # Setup passwordless sudo for ydotool
                print(f"\n  {C.D}├─ Old ydotool detected, setting up permissions...{C.E}")
                sudoers_file = "/etc/sudoers.d/ydotool"
                user = os.environ.get('USER', 'root')

                # Create sudoers entry for passwordless ydotool
                # Use subprocess with input= to avoid shell injection via USER env var
                sudoers_content = f"{user} ALL=(ALL) NOPASSWD: /usr/bin/ydotool\n"
                subprocess.run(['sudo', 'tee', sudoers_file],
                               input=sudoers_content.encode(), capture_output=True)
                subprocess.run(['sudo', 'chmod', '440', sudoers_file])
                installed.append("sudo permissions")
        else:
            # X11 or unknown - xdotool is required
            has_xdotool = shutil.which('xdotool') is not None
            if not has_xdotool:
                print(f"\n  {C.D}├─ Installing xdotool...{C.E}")
                subprocess.run(['sudo', 'apt', 'install', '-y', 'xdotool'], capture_output=True)
                has_xdotool = shutil.which('xdotool') is not None

            if not has_xdotool:
                return False, "xdotool required for X11 but install failed (sudo apt install xdotool)"

            installed.append("xdotool")

        # Ensure user is in 'input' group (needed for physical input detection)
        user = os.environ.get('USER', '')
        if user and user != 'root':
            # Use list args to prevent shell injection via USER env var
            try:
                r = subprocess.run(['id', '-nG', user], capture_output=True, text=True)
                in_group = r.returncode == 0
                groups_out = r.stdout.strip() if r.stdout else ''
            except Exception:
                in_group, groups_out = False, ''
            if in_group and 'input' not in groups_out.split():
                print(f"\n  {C.D}├─ Adding {user} to 'input' group (physical input detection)...{C.E}")
                subprocess.run(['sudo', 'usermod', '-aG', 'input', user])
                installed.append("input group")
                warnings.append("logout/login needed for input group")

        result = []
        if installed:
            result.append(f"installed {', '.join(installed)}")
        else:
            result.append("all present")

        if display != 'unknown':
            result.append(f"[{display}]")

        if warnings:
            return True, f"{' '.join(result)} ⚠ {warnings[0]}"
        return True, ' '.join(result)
    elif plat == 'windows':
        installed = []
        warnings = []
        # Check/install git
        if not shutil.which('git'):
            ok, out = run('winget install Git.Git -e --accept-source-agreements --accept-package-agreements')
            if ok:
                installed.append("git")
            else:
                warnings.append("git not found, install manually from git-scm.com")
        # Check/install ffmpeg
        if not shutil.which('ffmpeg'):
            ok, out = run('winget install Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements')
            if ok:
                installed.append("ffmpeg")
            else:
                warnings.append("ffmpeg not found, install from ffmpeg.org")
        if installed:
            result = f"installed {', '.join(installed)}"
        else:
            result = "all present"
        if warnings:
            return True, f"{result} ⚠ {'; '.join(warnings)}"
        return True, result
    elif plat == 'macos':
        installed = []
        warnings = []
        has_brew = shutil.which('brew') is not None
        if has_brew:
            # Install portaudio and ffmpeg if missing
            for pkg in ['portaudio', 'ffmpeg']:
                check = run(f'brew list {pkg} 2>/dev/null')
                if not check[0]:
                    ok, _ = run(f'brew install {pkg}')
                    if ok:
                        installed.append(pkg)
                    else:
                        warnings.append(f"{pkg} install failed")
        else:
            warnings.append("Homebrew not found — install from https://brew.sh then run: brew install portaudio ffmpeg")
        # pyobjc-framework-Quartz is needed for input control on macOS
        # (installed via requirements.txt platform marker, but ensure it's present)
        venv_pip = get_venv_pip()
        if os.path.exists(venv_pip):
            subprocess.run([venv_pip, "install", "-q", "pyobjc-framework-Quartz>=9.0"],
                           capture_output=True)
        if installed:
            result = f"installed {', '.join(installed)}"
        else:
            result = "all present" if has_brew else "manual install needed"
        if warnings:
            return True, f"{result} ⚠ {'; '.join(warnings)}"
        return True, result
    return True, "ok"

def step_packages():
    venv_pip = get_venv_pip()
    venv_python = get_venv_python()
    req_file = os.path.join(SCRIPT_DIR, 'requirements.txt')

    if not os.path.exists(req_file):
        return False, "requirements.txt missing"

    # Upgrade pip quietly
    print(f"  {C.D}├─ upgrading pip...{C.E}", flush=True)
    subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip", "-q"],
                   capture_output=True)

    start_time = time.time()

    # Windows: Install PyTorch with CUDA from official index first
    # (default pip install torch on Windows = CPU only)
    if get_platform() == 'windows':
        # Check if NVIDIA GPU available
        has_gpu, _ = run("nvidia-smi", shell=True, capture=True)
        if has_gpu:
            print(f"  {C.D}├─ installing PyTorch with CUDA (Windows)...{C.E}")
            subprocess.run([
                venv_pip, "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu124",
                "--progress-bar", "on"
            ], cwd=SCRIPT_DIR)

    # Count packages
    with open(req_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    total = len(lines)

    print(f"  {C.D}├─ installing {total} packages (showing download speeds){C.E}")
    print(f"  {C.D}│{C.E}")

    # Use pip's native progress bar which shows actual download speed (MB/s)
    pip_cmd = [venv_pip, "install", "-r", req_file, "--progress-bar", "on"]
    # Windows + GPU: add CUDA wheel index so pip doesn't overwrite CUDA torch with CPU-only from PyPI
    if get_platform() == 'windows':
        has_gpu, _ = run("nvidia-smi", shell=True, capture=True)
        if has_gpu:
            pip_cmd += ["--extra-index-url", "https://download.pytorch.org/whl/cu124"]
    result = subprocess.run(pip_cmd, cwd=SCRIPT_DIR)

    total_time = time.time() - start_time
    print(f"  {C.D}│{C.E}")
    print(f"  {C.D}└─{C.E} {C.G}done{C.E} {C.D}in {format_time(total_time)}{C.E}")

    if result.returncode != 0:
        return False, "pip install failed"
    return True, f"{total} packages"

def step_verify():
    venv_python = get_venv_python()
    test_file = os.path.join(SCRIPT_DIR, '_test.py')

    code = '''
import sys
try:
    import torch
    cuda = "CUDA" if torch.cuda.is_available() else "CPU"
    import faster_whisper, sounddevice, flask
    print(f"ok:{cuda}")
except Exception as e:
    print(f"fail:{e}")
    sys.exit(1)
'''
    with open(test_file, 'w') as f:
        f.write(code)

    ok, out = run(f'"{venv_python}" "{test_file}"')
    os.remove(test_file)

    if ok and 'ok' in out:
        mode = out.split(':')[1] if ':' in out else 'ok'
        return True, f"all imports ok ({mode})"
    return False, out

def step_input_tool():
    """Check if input tool (trackpad/typing) is available - required"""
    plat = get_platform()

    if plat == 'linux':
        display = get_display_server()

        if display == 'wayland':
            has_ydotool = shutil.which('ydotool') is not None
            if not has_ydotool:
                return False, f"ydotool not found (sudo apt install ydotool)"
            # Check if ydotoold daemon is running (new ydotool needs it)
            has_daemon = shutil.which('ydotoold') is not None
            if has_daemon:
                daemon_ok, _ = run("pgrep -x ydotoold")
                if not daemon_ok:
                    return False, f"ydotool daemon not running (sudo systemctl start ydotoold)"
            return True, f"ydotool on {display}"
        else:
            # X11: xdotool is required
            has_xdotool = shutil.which('xdotool') is not None
            if not has_xdotool:
                return False, f"xdotool not found (sudo apt install xdotool)"
            return True, f"xdotool on {display}"

    elif plat == 'windows':
        try:
            import pynput
            return True, "pynput"
        except ImportError:
            return True, "pynput (will install)"

    elif plat == 'macos':
        try:
            import pynput
            return True, "pynput"
        except ImportError:
            return True, "pynput (will install)"

    return True, "ok"

def step_ffmpeg():
    """Check if ffmpeg is available (needed for screen mirroring)"""
    has_ffmpeg = shutil.which('ffmpeg') is not None
    if not has_ffmpeg:
        return True, "not found (screen mirror disabled)"
    # Check hardware encoder support (cross-platform)
    try:
        r = subprocess.run([shutil.which('ffmpeg'), '-encoders'], capture_output=True, text=True, timeout=10)
        out = r.stdout if r.returncode == 0 else ''
    except Exception:
        out = ''
    if 'h264_nvenc' in out:
        return True, "ffmpeg + NVENC"
    elif 'h264_videotoolbox' in out:
        return True, "ffmpeg + VideoToolbox"
    return True, "ffmpeg (CPU encoding)"

def step_runscript():
    plat = get_platform()
    if plat == 'windows':
        script = os.path.join(SCRIPT_DIR, 'scripts', 'windows', 'run_bg.bat')
    elif plat == 'macos':
        script = os.path.join(SCRIPT_DIR, 'scripts', 'mac', 'run.sh')
    else:
        script = os.path.join(SCRIPT_DIR, 'scripts', 'linux', 'run.sh')

    if os.path.exists(script):
        if plat != 'windows':
            os.chmod(script, 0o755)
        return True, "executable"
    return False, "not found"

# =============================================================================
#                              MAIN
# =============================================================================

def install():
    start = time.time()

    print(f"""
  {C.B}Sanketra{C.E} {C.D}installer{C.E}
  {C.D}─────────────────────{C.E}
""")

    steps = [
        ("python", step_python),
        ("venv", step_venv),
        ("gpu", step_gpu),
        ("system", step_system_deps),
    ]

    # Quick checks first
    for name, func in steps:
        print(f"  {C.D}●{C.E} {name:<12}", end="", flush=True)
        ok, msg = func()
        if ok:
            print(f"{C.G}{C.OK}{C.E} {C.D}{msg}{C.E}")
        else:
            print(f"{C.R}{C.X}{C.E} {msg}")
            sys.exit(1)

    # Packages (special handling with progress)
    print(f"\n  {C.D}●{C.E} packages")
    ok, msg = step_packages()
    if not ok:
        print(f"\n  {C.R}{C.X}{C.E} {msg}")
        sys.exit(1)

    # Final checks
    print()
    plat = get_platform()
    if plat == 'windows':
        run_script_name = "scripts/windows/run_bg.bat"
    elif plat == 'macos':
        run_script_name = "scripts/mac/run.sh"
    else:
        run_script_name = "scripts/linux/run.sh"

    # Required steps (fail = exit)
    required_steps = [
        ("verify", step_verify),
        ("input tool", step_input_tool),
        ("screen", step_ffmpeg),
        (run_script_name, step_runscript),
    ]

    for name, func in required_steps:
        print(f"  {C.D}●{C.E} {name:<12}", end="", flush=True)
        ok, msg = func()
        if ok:
            print(f"{C.G}{C.OK}{C.E} {C.D}{msg}{C.E}")
        else:
            print(f"{C.R}{C.X}{C.E} {msg}")
            sys.exit(1)

    total_time = time.time() - start
    if plat == 'windows':
        run_cmd = "scripts\\windows\\run_bg.bat"
    elif plat == 'macos':
        run_cmd = "scripts/mac/run.sh"
    else:
        run_cmd = "scripts/linux/run.sh"

    print(f"""
  {C.D}─────────────────────{C.E}
  {C.G}{C.OK} ready{C.E} {C.D}in {format_time(total_time)}{C.E}
""")

    # Print usage instructions
    print(f"""  {C.B}To run:{C.E}
    {run_cmd}              {C.D}# interactive launcher{C.E}
    {run_cmd} cli          {C.D}# local microphone{C.E}
    {run_cmd} web          {C.D}# phone mic + trackpad{C.E}

  {C.B}Model options:{C.E}
    --fast, --balanced, --accurate, --custom
""")

def uninstall():
    print(f"\n  {C.B}Sanketra{C.E} {C.D}uninstall{C.E}\n")

    removed = []
    freed = 0

    def dir_size(path):
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        return total

    # 1. Stop and remove service (current + pre-rebrand legacy)
    plat = get_platform()
    if plat == 'linux':
        for svc in ["sanketra", "mic_on_term"]:
            run(f"systemctl --user stop {svc} 2>/dev/null")
            run(f"systemctl --user disable {svc} 2>/dev/null")
            svc_path = os.path.expanduser(f"~/.config/systemd/user/{svc}.service")
            if os.path.exists(svc_path):
                os.remove(svc_path)
                removed.append(f"systemd {svc}")
        run("systemctl --user daemon-reload")
        # Remove sudoers rules installed by setup
        for sudoers in ["/etc/sudoers.d/sanketra-gpu", "/etc/sudoers.d/ydotool"]:
            if os.path.exists(sudoers):
                subprocess.run(['sudo', 'rm', '-f', sudoers], capture_output=True)
                removed.append(os.path.basename(sudoers))
    elif plat == 'windows':
        for task in ["sanketra", "mic_on_term"]:
            run(f'schtasks /end /tn "{task}" 2>nul')
            run(f'schtasks /delete /tn "{task}" /f 2>nul')
        # Remove firewall rules (current + legacy)
        for rule_name in ["sanketra", "sanketra UDP", "mic_on_term", "mic_on_term UDP"]:
            subprocess.run(['netsh', 'advfirewall', 'firewall', 'delete', 'rule',
                            f'name={rule_name}'], capture_output=True, check=False)
        removed.append("scheduled task + firewall")
    elif plat == 'macos':
        for plist_name in ["com.miconterm.server", "com.miconterm.mic_on_term"]:
            plist_path = os.path.expanduser(f"~/Library/LaunchAgents/{plist_name}.plist")
            run(f'launchctl unload "{plist_path}" 2>/dev/null')
            if os.path.exists(plist_path):
                os.remove(plist_path)
                removed.append(f"launchd {plist_name}")

    # 2. Remove venv (biggest item — torch + deps)
    if os.path.exists(VENV_DIR):
        freed += dir_size(VENV_DIR)
        shutil.rmtree(VENV_DIR)
        removed.append("venv")

    # 3. Remove config directories (~/.config/sanketra/ + legacy ~/.config/mic_on_term/)
    for cfg_name in ["sanketra", "mic_on_term"]:
        config_dir = os.path.expanduser(f"~/.config/{cfg_name}")
        if os.path.exists(config_dir):
            freed += dir_size(config_dir)
            shutil.rmtree(config_dir)
            removed.append(f"config ({cfg_name})")

    # 4. Remove Whisper model cache (~/.cache/huggingface/hub/models--Systran--faster-whisper-*)
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.isdir(hf_cache):
        for d in os.listdir(hf_cache):
            if d.startswith("models--Systran--faster-whisper"):
                full = os.path.join(hf_cache, d)
                freed += dir_size(full)
                shutil.rmtree(full)
                removed.append(f"model cache ({d.split('--')[-1]})")

    # 5. Remove Silero VAD cache
    for cache_name in ["silero-vad-models", "snakers4"]:
        vad_cache = os.path.expanduser(f"~/.cache/torch/hub/{cache_name}")
        if os.path.isdir(vad_cache):
            freed += dir_size(vad_cache)
            shutil.rmtree(vad_cache)
            removed.append("vad cache")
            break

    # 6. Remove logs
    logs_dir = os.path.join(SCRIPT_DIR, "logs")
    if os.path.exists(logs_dir):
        for f in glob.glob(os.path.join(logs_dir, "*.log")):
            freed += os.path.getsize(f)
            os.remove(f)
        removed.append("logs")

    # 7. Remove certs and cache
    for f in ["cert.pem", "key.pem"]:
        path = os.path.join(SCRIPT_DIR, f)
        if os.path.exists(path):
            freed += os.path.getsize(path)
            os.remove(path)
            removed.append(f)

    pycache = os.path.join(SCRIPT_DIR, "__pycache__")
    if os.path.exists(pycache):
        shutil.rmtree(pycache)

    # 8. Remove log files in project root
    for f in glob.glob(os.path.join(SCRIPT_DIR, "*.log")):
        freed += os.path.getsize(f)
        os.remove(f)

    if removed:
        print(f"  {C.G}{C.OK}{C.E} removed: {C.D}{', '.join(removed)}{C.E}")
        print(f"  {C.G}{C.OK}{C.E} freed: {C.D}{format_size(freed)}{C.E}")
    else:
        print(f"  {C.D}nothing to remove{C.E}")

    if get_platform() == 'windows':
        home = os.path.expanduser("~")
        print(f"\n  {C.D}To fully remove, run:{C.E} rmdir /s /q \"{os.path.join(home, 'sanketra')}\"")
        print(f"  {C.D}reinstall:{C.E} python setup.py\n")
    else:
        print(f"\n  {C.D}To fully remove, run:{C.E} rm -rf ~/sanketra")
        print(f"  {C.D}reinstall:{C.E} python3 setup.py\n")

def install_service():
    """Install platform-appropriate background service for Sanketra"""
    plat = get_platform()
    venv_python = get_venv_python()
    server_script = os.path.join(SCRIPT_DIR, "src", "server_async.py")

    if plat == 'linux':
        service_dir = os.path.expanduser("~/.config/systemd/user")
        os.makedirs(service_dir, exist_ok=True)
        service_content = f"""[Unit]
Description=Sanketra Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={SCRIPT_DIR}
ExecStart={venv_python} {server_script} --service
Restart=on-failure
RestartSec=5
TimeoutStopSec=10
KillSignal=SIGTERM
Environment=DISPLAY=:0
Environment=DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/%U/bus

[Install]
WantedBy=default.target
"""
        service_path = os.path.join(service_dir, "sanketra.service")
        with open(service_path, 'w') as f:
            f.write(service_content)
        run("systemctl --user daemon-reload")
        run("systemctl --user enable sanketra")
        return True, service_path

    elif plat == 'windows':
        # Use Task Scheduler to run at logon
        # pythonw.exe = no console window on login (python.exe spawns a visible cmd window)
        venv_pythonw = venv_python.replace('python.exe', 'pythonw.exe')
        task_cmd = f'"{venv_pythonw}" "{server_script}" --service'
        # Delete existing task (ignore errors)
        run('schtasks /delete /tn "sanketra" /f 2>nul')
        ok, out = run(
            f'schtasks /create /tn "sanketra" /sc ONLOGON /rl HIGHEST '
            f'/tr "{task_cmd}" /f'
        )
        if ok:
            # Add firewall rules for Sanketra (TCP 5000 for HTTP/WS, UDP 5001 for discovery)
            subprocess.run(['netsh', 'advfirewall', 'firewall', 'add', 'rule',
                            'name=sanketra', 'dir=in', 'action=allow',
                            'protocol=TCP', 'localport=5000'],
                           capture_output=True, check=False)
            subprocess.run(['netsh', 'advfirewall', 'firewall', 'add', 'rule',
                            'name=sanketra UDP', 'dir=in', 'action=allow',
                            'protocol=UDP', 'localport=5001'],
                           capture_output=True, check=False)
            return True, "Task Scheduler (ONLOGON)"
        return False, f"schtasks failed: {out}"

    elif plat == 'macos':
        plist_dir = os.path.expanduser("~/Library/LaunchAgents")
        os.makedirs(plist_dir, exist_ok=True)
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.miconterm.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>{venv_python}</string>
        <string>{server_script}</string>
        <string>--service</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{SCRIPT_DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>{os.path.join(SCRIPT_DIR, 'logs', 'server.log')}</string>
    <key>StandardErrorPath</key>
    <string>{os.path.join(SCRIPT_DIR, 'logs', 'server_err.log')}</string>
</dict>
</plist>
"""
        plist_path = os.path.join(plist_dir, "com.miconterm.server.plist")
        with open(plist_path, 'w') as f:
            f.write(plist_content)
        return True, plist_path

    return False, f"unsupported platform: {plat}"


def main():
    if '--uninstall' in sys.argv or 'uninstall' in sys.argv:
        uninstall()
    elif '--install-service' in sys.argv:
        ok, msg = install_service()
        if ok:
            print(f"  {C.G}{C.OK}{C.E} service installed: {C.D}{msg}{C.E}")
        else:
            print(f"  {C.R}{C.X}{C.E} {msg}")
            sys.exit(1)
    elif '--help' in sys.argv or '-h' in sys.argv:
        print(f"""
  {C.B}usage:{C.E}
    python3 setup.py                  {C.D}# install{C.E}
    python3 setup.py --uninstall      {C.D}# remove venv, logs, certs{C.E}
    python3 setup.py --install-service {C.D}# install systemd user service{C.E}
""")
    else:
        install()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n  {C.Y}cancelled{C.E}\n")
        sys.exit(1)
