#!/bin/bash
# Sanketra - Update script (Linux/Mac)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

echo ""
echo "  Updating Sanketra..."
echo "  ─────────────────────────"
echo ""

# Check if git repo
if [ ! -d ".git" ]; then
    echo "  Error: Not a git repository"
    exit 1
fi

# Pull latest
echo "  Pulling latest changes..."
git pull

if [ $? -ne 0 ]; then
    echo ""
    echo "  Error: git pull failed"
    exit 1
fi

# Check if venv exists
VENV_PIP="$ROOT_DIR/venv/bin/pip"
if [ -f "$VENV_PIP" ]; then
    echo ""
    echo "  Updating packages..."
    "$VENV_PIP" install -r requirements.txt -q --upgrade
    echo "  Done."
else
    echo ""
    echo "  Note: venv not found. Run setup.py first:"
    echo "    python3 setup.py"
fi

echo ""
echo "  ─────────────────────────"
echo "  Update complete!"
echo ""
