#!/usr/bin/env bash
# Sanketra — One-click installer for Linux and macOS
# Usage: bash install.sh   OR   double-click install.command (macOS)
set -euo pipefail

REPO_URL="https://github.com/Misc42/sanketra-server.git"
INSTALL_DIR="$HOME/sanketra"
BRANCH="master"

# ─── Colors ───────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
DIM='\033[0;90m'
BOLD='\033[1m'
NC='\033[0m'

info()  { printf "${GREEN}[✓]${NC} %s\n" "$*"; }
warn()  { printf "${YELLOW}[!]${NC} %s\n" "$*"; }
fail()  { printf "${RED}[✗]${NC} %s\n" "$*"; exit 1; }
step()  { printf "\n${BOLD}── %s${NC}\n" "$*"; }

# ─── OS Detection ─────────────────────────────────────────────────────
detect_os() {
    case "$(uname -s)" in
        Linux*)  OS="linux";;
        Darwin*) OS="macos";;
        *)       fail "Unsupported OS: $(uname -s). Use install.bat for Windows.";;
    esac
}

# ─── Package Manager Detection (Linux) ────────────────────────────────
detect_pkg_manager() {
    if command -v apt-get &>/dev/null; then
        PKG="apt"
    elif command -v dnf &>/dev/null; then
        PKG="dnf"
    elif command -v pacman &>/dev/null; then
        PKG="pacman"
    else
        fail "No supported package manager found (need apt, dnf, or pacman)"
    fi
}

# ─── Install System Dependencies ──────────────────────────────────────
install_deps_linux() {
    detect_pkg_manager
    step "Installing dependencies via $PKG"

    case "$PKG" in
        apt)
            sudo apt-get update -qq
            sudo apt-get install -y -qq git python3 python3-venv python3-pip ffmpeg
            ;;
        dnf)
            sudo dnf install -y -q git python3 python3-pip ffmpeg
            ;;
        pacman)
            sudo pacman -Sy --noconfirm --needed git python python-pip ffmpeg
            ;;
    esac
    info "System dependencies installed"
}

install_deps_macos() {
    step "Checking macOS prerequisites"

    # Xcode CLT required for git and compilation
    if ! xcode-select -p &>/dev/null; then
        warn "Xcode Command Line Tools not installed"
        echo "    Installing... (this may take a few minutes)"
        xcode-select --install 2>/dev/null || true
        # Wait for user to complete the GUI dialog
        echo ""
        echo "    A dialog should have appeared on screen."
        echo "    Click 'Install' and wait for it to finish."
        echo ""
        read -rp "    Press Enter when installation is complete..."
        if ! xcode-select -p &>/dev/null; then
            fail "Xcode CLT installation failed. Run: xcode-select --install"
        fi
    fi
    info "Xcode Command Line Tools OK"

    # Homebrew
    if ! command -v brew &>/dev/null; then
        warn "Homebrew not found — installing..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        # Add brew to PATH for this session (Apple Silicon vs Intel)
        if [[ -f /opt/homebrew/bin/brew ]]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        elif [[ -f /usr/local/bin/brew ]]; then
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    fi
    info "Homebrew OK"

    brew install python3 git ffmpeg 2>/dev/null || brew upgrade python3 git ffmpeg 2>/dev/null || true
    info "macOS dependencies installed"
}

# ─── Clone or Update Repository ───────────────────────────────────────
setup_repo() {
    step "Setting up Sanketra"

    if [[ -d "$INSTALL_DIR/.git" ]]; then
        info "Existing install found — updating..."
        cd "$INSTALL_DIR"
        git fetch origin
        git reset --hard "origin/$BRANCH"
        cd - >/dev/null
    else
        info "Cloning repository..."
        git clone --depth 1 -b "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
    fi
    info "Repository ready at $INSTALL_DIR"
}

# ─── Run setup.py ─────────────────────────────────────────────────────
run_setup() {
    step "Running setup (venv, dependencies, GPU detection)"
    cd "$INSTALL_DIR"

    # setup.py creates venv, installs pip deps, configures GPU
    python3 setup.py
    info "Setup complete"
}

# ─── Install + Start Service ──────────────────────────────────────────
install_service() {
    step "Installing system service"
    cd "$INSTALL_DIR"

    python3 setup.py --install-service
    info "Service installed"

    if [[ "$OS" == "linux" ]]; then
        # systemd user service
        systemctl --user daemon-reload 2>/dev/null || true
        systemctl --user restart sanketra 2>/dev/null || true
        systemctl --user enable sanketra 2>/dev/null || true
        info "Service started (systemd user)"
    elif [[ "$OS" == "macos" ]]; then
        # launchd
        local plist="$HOME/Library/LaunchAgents/com.miconterm.server.plist"
        launchctl unload "$plist" 2>/dev/null || true
        launchctl load "$plist" 2>/dev/null || true
        info "Service started (launchd)"
    fi
}

# ─── macOS: Handle port 5000 AirPlay conflict ─────────────────────────
check_macos_port() {
    if [[ "$OS" == "macos" ]]; then
        if lsof -iTCP:5000 -sTCP:LISTEN &>/dev/null 2>&1; then
            warn "Port 5000 in use (probably AirPlay Receiver)"
            echo "    Server will auto-fallback to port 5001"
            echo "    Or: System Settings → General → AirDrop & Handoff → AirPlay Receiver → OFF"
        fi
    fi
}

# ─── Final Message ────────────────────────────────────────────────────
show_done() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "  ${GREEN}${BOLD}Sanketra is ready!${NC}\n"
    echo ""

    echo "  Make sure your phone is on the same WiFi."
    echo "  The app will find this computer automatically."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# ─── Main ─────────────────────────────────────────────────────────────
main() {
    echo ""
    printf "${BOLD}Sanketra Installer${NC}\n"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━"

    detect_os
    info "Detected OS: $OS"

    if [[ "$OS" == "linux" ]]; then
        install_deps_linux
    elif [[ "$OS" == "macos" ]]; then
        install_deps_macos
    fi

    setup_repo
    run_setup
    install_service
    check_macos_port
    show_done
}

main "$@"
