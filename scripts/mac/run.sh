#!/bin/bash
# mic_on_term - Unified run script

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PYTHON="$ROOT_DIR/venv/bin/python3"

# Use all CPU cores for inference
NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
export OMP_NUM_THREADS=$NUM_CORES
export MKL_NUM_THREADS=$NUM_CORES
export OPENBLAS_NUM_THREADS=$NUM_CORES
export VECLIB_MAXIMUM_THREADS=$NUM_CORES
export NUMEXPR_NUM_THREADS=$NUM_CORES

# Check venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Virtual environment not found. Run setup.py first:"
    echo "  python3 setup.py"
    exit 1
fi

# If no arguments, run interactive Python launcher
if [ $# -eq 0 ]; then
    "$VENV_PYTHON" "$ROOT_DIR/src/run.py"
    exit 0
fi

# Direct mode - arguments provided
MODE=$1
shift

case $MODE in
    cli)
        "$VENV_PYTHON" "$ROOT_DIR/src/stt_vad.py" "$@"
        ;;
    web)
        "$VENV_PYTHON" "$ROOT_DIR/src/server_async.py" "$@"
        ;;
    -h|--help)
        "$VENV_PYTHON" "$ROOT_DIR/src/run.py" --help
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Use 'cli' or 'web', or run without args for interactive mode"
        exit 1
        ;;
esac
