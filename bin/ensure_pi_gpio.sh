#!/usr/bin/env bash
# Ensure the Raspberry Pi GPIO Python libraries (gpiozero + lgpio) are present
# in the venv - but ONLY on a real Raspberry Pi, and ONLY if they're missing.
#
# Both bin/linux-install.sh and the everyday launcher bin/linux-run.sh call
# this, so:
#   * a fresh install on a Pi gets the libs, and
#   * a Pi that was UPDATED (pulled) rather than reinstalled picks them up on
#     the next launch (e.g. the first boot after the image-display buttons were
#     added).
#
# It is deliberately safe on every OTHER system: the /proc/device-tree/model
# check means a non-Pi Linux (x86 or non-Pi ARM), macOS, etc. does nothing at
# all here - it can't install anything or break a non-Pi box. It's also
# idempotent + offline-safe: once the libs are installed the cheap import test
# short-circuits with NO network call, so it adds no boot delay and never fails
# a launch when offline. Non-fatal throughout (callers use `|| true`).
#
# gpiozero is the button API the image-display event uses; lgpio is the pin
# backend that works on all current Pis (Pi 5 / Bookworm included).
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
PY="$ROOT/venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3 || command -v python || true)"
[ -n "$PY" ] || exit 0

# Non-Pi -> do nothing. This is the guard that keeps non-Pi systems untouched.
grep -qi "raspberry pi" /proc/device-tree/model 2>/dev/null || exit 0

# Already installed -> nothing to do (instant, no network).
if "$PY" -c "import gpiozero, lgpio" >/dev/null 2>&1; then
    exit 0
fi

echo "[gpio] Raspberry Pi detected and GPIO libs missing - installing gpiozero + lgpio into the venv..."
"$PY" -m pip install gpiozero lgpio \
    || echo "[gpio] install failed (offline?); the image-display button falls back to the SPACE bar until this succeeds"
