#!/usr/bin/env bash
set -euo pipefail

echo "====================================="
echo "  GL_Simple Setup & Launcher (Ubuntu)"
echo "====================================="

# This script is prescriptive for Ubuntu/Debian systems. It assumes `apt-get` is available
echo "[1/4] Updating apt and installing required system packages (non-interactive)..."
sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip build-essential \
    libportaudio2 portaudio19-dev libsndfile1-dev libasound2-dev pkg-config

echo "[2/4] Creating virtual environment (./venv) if missing..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Virtual environment created"
else
    echo "  Virtual environment already exists"
fi

echo "[3/4] Activating virtual environment and installing Python packages..."
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

REQ="info/requirements.txt"
if [ ! -f "$REQ" ]; then
    echo "ERROR: requirements file not found at $REQ" >&2
    echo "Please provide dependency file at info/requirements.txt" >&2
    exit 1
fi

echo "  Installing from $REQ (this can take several minutes)..."
python -m pip install -r "$REQ"

# Verify PortAudio / sounddevice is importable; if not, attempt to install system libs and reinstall sounddevice
echo "  Verifying audio support..."
python - <<'PY'
import sys
try:
    import sounddevice
    sys.exit(0)
except Exception as e:
    print('sounddevice import failed:', e)
    sys.exit(2)
PY
if [ $? -ne 0 ]; then
    echo "sounddevice failed to import. Installing PortAudio system packages and retrying..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get install -y libportaudio2 portaudio19-dev libsndfile1-dev libasound2-dev pkg-config
        source venv/bin/activate
        python -m pip install --force-reinstall --no-cache-dir sounddevice
        echo "  Re-checking sounddevice import..."
        python - <<'PY'
import sys
try:
    import sounddevice
    print('sounddevice import OK')
    sys.exit(0)
except Exception as e:
    print('sounddevice import still failing:', e)
    sys.exit(1)
PY
        if [ $? -ne 0 ]; then
            echo "ERROR: sounddevice still failing to import after installing system libs." >&2
            echo "Please check your system or install PortAudio manually (e.g. 'sudo apt-get install portaudio19-dev libsndfile1-dev')." >&2
            exit 1
        fi
    else
        echo "apt-get not available; cannot install PortAudio system packages automatically." >&2
        echo "Please install PortAudio (portaudio19-dev) and libsndfile (libsndfile1-dev) and re-run this script." >&2
        exit 1
    fi
fi

echo "[4/4] Setup complete. Launching application..."
echo "  Web control panel: http://localhost:5000"
echo "  Press Ctrl+C to stop"

python Stories_OGL.py

echo "Application exited."
