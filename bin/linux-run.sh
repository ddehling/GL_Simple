#!/usr/bin/env bash
# GL_Simple everyday launch (Linux/macOS).
#
# Pulls latest engine + all deployed projects from their remotes,
# then launches the app. Offline-tolerant: any pull that can't reach
# its remote (no DNS, no internet, auth failure, hung connection) is
# logged and skipped, and the app launches with whatever's on disk.
#
# Use bin/linux-install.sh for the first install (also handles dep
# install + gh auth). Use bin/linux-run.sh every subsequent launch.

set -uo pipefail
cd "$(dirname "$0")/.."

# Mirror everything up to the app launch into a logfile so a boot-time
# failure (network race, auth, etc.) can be read after the fact instead of
# only living in the autostart xterm. The app's own stdout still goes to the
# terminal via exec at the end.
LOG_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/gl-simple"
mkdir -p "$LOG_DIR" 2>/dev/null || true
LOG_FILE="$LOG_DIR/last-boot.log"
exec > >(tee "$LOG_FILE") 2>&1
echo "=== launch $(date -Is 2>/dev/null || true) ==="

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

# At boot the graphical autostart can fire before NetworkManager/WiFi has
# connectivity, so the very first pull fails to even resolve github.com
# ("could not resolve host" - looks like "can't find github"). Wait, bounded,
# for github.com to become reachable before pulling. Still offline-tolerant:
# if it never comes up within the timeout we proceed and launch with local code.
wait_for_github() {
    local deadline=$(( SECONDS + 15 ))   # cap the offline wait at 15s
    while [ "$SECONDS" -lt "$deadline" ]; do
        if getent hosts github.com >/dev/null 2>&1; then
            return 0
        fi
        echo "  waiting for network (github.com not resolvable yet)..."
        sleep 3
    done
    echo "  network still down after 15s - proceeding with local code."
    return 1
}

echo "[1/2] Pulling latest engine + deployed projects..."
wait_for_github
pull_repo "engine" "."
for d in projects/*/; do
    [ -d "$d/.git" ] || continue
    id=$(basename "$d")
    pull_repo "$id" "$d"
done

echo ""
echo "[2/2] Launching application..."
if [ ! -d venv ]; then
    echo "ERROR: ./venv not found. Run bin/linux-install.sh first." >&2
    exit 1
fi
# Re-apply the port-80 capability if a python upgrade stripped it (non-fatal).
bin/ensure_port_cap.sh || true
# On a Raspberry Pi only, make sure the GPIO libs are installed (no-op if they
# already are, or on any non-Pi system). Lets a Pi that was updated rather than
# reinstalled pick up the image-display button support. Non-fatal + offline-safe.
bash bin/ensure_pi_gpio.sh || true
# Show machines need the performance CPU governor or heavy seams stutter
# (see lib/audio_engine.py RING_TARGET_MS note). Warns + self-heals when the
# systemd unit from linux-install.sh is present; never blocks the launch.
bash bin/ensure_cpu_performance.sh || true
exec venv/bin/python Stories_OGL.py
