#!/usr/bin/env bash
# Autostart launcher wrapper, invoked by the XDG .desktop entry.
#
# Why this exists: on a Wayland desktop (Raspberry Pi OS with labwc/wayfire)
# the app was launched via xterm — an X11 program that runs through XWayland.
# When the autostart fires before XWayland is ready, xterm fails to open a
# window and exits immediately, so the app never launched ("half the time it
# doesn't start at the desktop"). The .desktop now calls THIS script (plain
# bash, no X dependency to even begin), and we retry launching the terminal
# until it sticks. If the terminal never comes up we launch headless anyway so
# a kiosk always ends up running (the app is headless; output is captured to
# ~/.local/state/gl-simple/last-boot.log by linux-run.sh regardless).

cd "$(dirname "$0")/.."
RUN="$(pwd)/bin/linux-run.sh"

max_attempts=10          # ~10 × 3s ≈ 30s of retries for XWayland to come up
attempt=0
while [ "$attempt" -lt "$max_attempts" ]; do
    attempt=$((attempt + 1))
    start=$SECONDS
    # -hold keeps the window open after the app exits so tracebacks stay
    # visible. While the app runs, this xterm stays in the foreground and
    # blocks here — that's the success path.
    xterm -T GL_Simple -geometry 120x30+50+50 -hold -e /bin/bash "$RUN"
    rc=$?
    # If xterm stayed up for a meaningful time it hosted the app — done. A
    # near-instant exit means it couldn't open a window (display not ready).
    if [ $(( SECONDS - start )) -ge 5 ]; then
        exit "$rc"
    fi
    echo "[autostart] xterm exited immediately (rc=$rc, attempt" \
         "$attempt/$max_attempts) — display not ready? retry in 3s" >&2
    sleep 3
done

# Terminal never came up. Launch headless so the rig still runs; all output is
# in ~/.local/state/gl-simple/last-boot.log.
echo "[autostart] no terminal after $max_attempts tries; launching headless" >&2
exec /bin/bash "$RUN"
