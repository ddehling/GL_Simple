#!/usr/bin/env bash
# GL_Simple everyday launch (Linux/macOS).
#
# Pulls latest engine + all deployed projects from their remotes,
# then launches the app. Offline-tolerant: any pull that can't reach
# its remote (no DNS, no internet, auth failure, hung connection) is
# logged and skipped, and the app launches with whatever's on disk.
#
# Use bin/setup.sh for the first install (also handles dep
# install + gh auth). Use bin/run.sh every subsequent launch.

set -uo pipefail
cd "$(dirname "$0")/.."

# Short timeout per pull so a slow / unreachable remote doesn't make
# the operator wait. `timeout` is GNU coreutils (default on Ubuntu);
# fall back to no-timeout if it's missing.
if command -v timeout >/dev/null 2>&1; then
    PULL="timeout 15 git"
else
    PULL="git"
fi

pull_repo() {
    local label="$1" dir="$2"
    if [ ! -d "$dir/.git" ]; then
        echo "  $label: not a git checkout, skipping"
        return 0
    fi
    # GIT_HTTP_LOW_SPEED_* aborts a stalled HTTPS clone in ~5s.
    # --ff-only refuses to merge if local commits diverge (safer than
    # silently merging during an unattended launch).
    if GIT_TERMINAL_PROMPT=0 GIT_HTTP_LOW_SPEED_LIMIT=1000 GIT_HTTP_LOW_SPEED_TIME=5 \
            $PULL -C "$dir" pull --ff-only 2>&1 | sed "s/^/  $label: /"; then
        return 0
    fi
    # Any nonzero exit (network, auth, conflict, divergent history)
    # gets a one-line log and we proceed.
    echo "  $label: pull failed (offline / auth / conflict?) - keeping local state"
    return 0
}

echo "[1/2] Pulling latest engine + deployed projects..."
pull_repo "engine" "."
for d in projects/*/; do
    [ -d "$d/.git" ] || continue
    id=$(basename "$d")
    pull_repo "$id" "$d"
done

echo ""
echo "[2/2] Launching application..."
if [ ! -d venv ]; then
    echo "ERROR: ./venv not found. Run bin/setup.sh first." >&2
    exit 1
fi
exec venv/bin/python Stories_OGL.py
