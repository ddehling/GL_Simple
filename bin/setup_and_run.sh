#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "====================================="
echo "  GL_Simple Setup & Launcher (Ubuntu)"
echo "====================================="

# This script is prescriptive for Ubuntu/Debian systems. It assumes `apt-get` is available
#
# Ensure git is available before anything else - the project auto-clone
# step below needs it. (On Linux we can install it via apt; on Windows
# the .ps1 script handles installation via winget.)
if ! command -v git >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        echo "[init] git not found - installing..."
        sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git
    else
        echo "ERROR: git is required but not installed and apt-get isn't available." >&2
        echo "       Install git manually and re-run." >&2
        exit 1
    fi
fi

# Ensure the active project's source tree is present. Per-project source
# (code + shaders + media) lives in its own private GitHub repo
# (GL_Simple_<id>) and is cloned into projects/<id>/ on first run.
# projects/<id>/ is gitignored in the main repo - engine code stays
# project-agnostic. The clone URL comes from deploy/catalog.yaml.
#
# Legacy: projects that still use a media-only submodule (declared in
# .gitmodules) are handled by the second branch below until they're
# migrated to the standalone-clone model.
#
# We grep config files rather than parse YAML so this step doesn't
# require Python yet (Python install happens below).
if [ -f config.yaml ]; then
    PROJECT_ID=$(grep -E '^project:' config.yaml | head -1 \
        | sed -E 's/^project:[[:space:]]*//; s/[[:space:]]*#.*$//; s/^["'"'"']//; s/["'"'"']$//; s/[[:space:]]+$//')
fi

if [ -n "${PROJECT_ID:-}" ]; then
    # New model: standalone clone from deploy/catalog.yaml.
    if [ ! -f "projects/$PROJECT_ID/project.yaml" ] && [ -f deploy/catalog.yaml ]; then
        # Pull the `repo:` line out of the catalog entry for this project.
        # Looks for the indented "<id>:" block then the first "repo:" under it.
        REPO_URL=$(awk -v pid="$PROJECT_ID" '
            $0 ~ "^  "pid":" { found=1; next }
            found && /^  [a-zA-Z_]/ { found=0 }
            found && /repo:/ { sub(/.*repo:[[:space:]]*/, ""); print; exit }
        ' deploy/catalog.yaml)
        if [ -n "$REPO_URL" ]; then
            echo "[init] Project '$PROJECT_ID' not deployed - cloning $REPO_URL ..."
            git clone "$REPO_URL" "projects/$PROJECT_ID" \
                || { echo "ERROR: clone failed. Set up GitHub auth (PAT or SSH key) and retry."; exit 1; }
        else
            echo "ERROR: Project '$PROJECT_ID' has no entry in deploy/catalog.yaml." >&2
            echo "       Add it, or change 'project:' in config.yaml to a deployed project." >&2
            exit 1
        fi
    fi
    # Legacy model: media-only submodule (will be removed once all projects
    # are migrated to standalone clones).
    if [ -f .gitmodules ] && grep -q "projects/$PROJECT_ID/media" .gitmodules 2>/dev/null; then
        echo "[init] Ensuring projects/$PROJECT_ID/media submodule is populated..."
        git submodule update --init "projects/$PROJECT_ID/media" \
            || echo "  (submodule init failed; retry: git submodule update --init projects/$PROJECT_ID/media)"
    fi
fi

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

REQ="requirements.txt"
if [ ! -f "$REQ" ]; then
    echo "ERROR: requirements file not found at $REQ" >&2
    echo "Please provide dependency file at requirements.txt" >&2
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

# Allow Python to bind to port 80 without sudo (Linux only, requires libcap2-bin)
VENV_PYTHON="$(readlink -f venv/bin/python3)"
if ! getcap "$VENV_PYTHON" 2>/dev/null | grep -q cap_net_bind_service; then
    echo "[4/5] Granting Python permission to bind port 80 (requires sudo)..."
    sudo setcap 'cap_net_bind_service=+ep' "$VENV_PYTHON"
else
    echo "[4/5] Python already has port-80 capability"
fi

echo "[5/5] Setup complete. Launching application..."
echo "  Web control panel: http://lucifera.local"
echo "  Press Ctrl+C to stop"

python Stories_OGL.py

echo "Application exited."
