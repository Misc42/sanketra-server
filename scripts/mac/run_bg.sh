#!/bin/bash
# Sanketra - Background mode (no terminal needed)
# Use this to run server in background

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PYTHON="$ROOT_DIR/venv/bin/python"

# Check venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Virtual environment not found. Run setup.py first:"
    echo "  python3 setup.py"
    exit 1
fi

echo ""
echo "  Starting Sanketra in background mode..."
echo "  QR code will open as image - scan it with phone"
echo ""
echo "  To stop: pkill -f src/server_async.py"
echo ""

# Run in background with nohup
nohup "$VENV_PYTHON" "$ROOT_DIR/src/server_async.py" --background "$@" > /dev/null 2>&1 &

sleep 2
echo "  Server started! QR image should open automatically."
echo ""
