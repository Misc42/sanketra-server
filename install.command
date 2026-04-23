#!/usr/bin/env bash
# Sanketra — macOS double-click installer
# Finder opens Terminal and runs this script when double-clicked.
# Delivered inside a .zip to bypass Gatekeeper quarantine flag.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# If install.sh is bundled alongside, use it directly
if [[ -f "$SCRIPT_DIR/install.sh" ]]; then
    bash "$SCRIPT_DIR/install.sh"
else
    # Download and run install.sh from GitHub
    echo "Downloading Sanketra installer..."
    curl -fsSL "https://raw.githubusercontent.com/Misc42/sanketra-server/master/install.sh" -o "/tmp/sanketra_install.sh"
    bash "/tmp/sanketra_install.sh"
fi

echo ""
echo "You can close this window now."
read -rp "Press Enter to exit..."
