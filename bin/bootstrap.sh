#!/usr/bin/env bash
# GL_Simple bootstrap for Linux (Ubuntu/Debian).
#
# Run on a fresh machine with nothing installed:
#
#   bash <(curl -fsSL https://raw.githubusercontent.com/ddehling/GL_Simple/main/bin/bootstrap.sh)
#
# What it does:
#   1. Installs git (if missing) via apt.
#   2. Configures git to cache GitHub credentials.
#   3. Clones the public GL_Simple engine repo into ./GL_Simple.
#   4. Runs bin/deploy.sh (interactive: pick which projects to install
#      and which is primary; first private-repo clone will prompt for
#      GitHub username + PAT).
#   5. Runs bin/setup_and_run.sh (installs Python + system deps + venv,
#      then launches the app).
#
# The deploy + setup scripts are interactive, so invoke this with
# `bash <(curl ...)` (process substitution) rather than `curl ... | bash`
# so prompts can read your terminal.

set -euo pipefail

echo "====================================="
echo "  GL_Simple Bootstrap (Linux)"
echo "====================================="

# --- 1. git ---
if ! command -v git >/dev/null 2>&1; then
    echo "[bootstrap] Installing git..."
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git
else
    echo "[bootstrap] git OK ($(git --version))"
fi

# --- 2. credential cache ---
if [ -z "$(git config --global credential.helper || true)" ]; then
    git config --global credential.helper store
    echo "[bootstrap] git credential helper set to 'store'"
fi

# --- 3. clone engine ---
REPO_DIR="GL_Simple"
if [ -d "$REPO_DIR/.git" ]; then
    echo "[bootstrap] $REPO_DIR/ already cloned; pulling latest..."
    git -C "$REPO_DIR" pull --ff-only
else
    echo "[bootstrap] Cloning GL_Simple engine..."
    git clone https://github.com/ddehling/GL_Simple.git "$REPO_DIR"
fi
cd "$REPO_DIR"

# --- 4. deploy (pick projects) ---
chmod +x bin/deploy.sh bin/setup_and_run.sh
echo ""
echo "[bootstrap] Launching project picker..."
./bin/deploy.sh

# --- 5. install deps + run ---
echo ""
echo "[bootstrap] Installing dependencies and launching app..."
./bin/setup_and_run.sh
