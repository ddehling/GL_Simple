#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Quick run for GL_Simple (Ubuntu)"

if [ -d "venv" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
else
    echo "Virtual environment not found. Run ./setup_and_run.sh first." >&2
    exit 1
fi

echo "Starting application..."
echo "  Web control panel: http://localhost:5000"
echo "Press Ctrl+C to stop"

python Stories_OGL.py
