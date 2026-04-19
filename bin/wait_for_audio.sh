#!/usr/bin/env bash
# Block until a real (non-null) audio sink is the PipeWire/Pulse default.
# If a real sink exists in the graph but WirePlumber hasn't promoted it,
# force it via `pactl set-default-sink` — at boot (before login),
# WirePlumber often leaves the default as auto_null indefinitely.
# Called from gl-simple.service ExecStartPre.

set -u

MAX_WAIT=25
INTERVAL=1

# Kick the socket-activated PipeWire services so they're running.
systemctl --user start pipewire.service pipewire-pulse.service wireplumber.service 2>/dev/null || true

# Restore ALSA hardware mixer state (unmute Master/Front/Speaker/etc).
# Login normally triggers this; without a session, hardware mutes may persist.
alsactl restore >/dev/null 2>&1 || true
for card in $(aplay -l 2>/dev/null | awk '/^card/ {print $2}' | tr -d ':' | sort -u); do
    for ctl in Master Front Speaker Headphone PCM Surround Center LFE; do
        amixer -q -c "$card" sset "$ctl" unmute >/dev/null 2>&1 || true
    done
    amixer -q -c "$card" sset Master 100% >/dev/null 2>&1 || true
    amixer -q -c "$card" sset Front 100% >/dev/null 2>&1 || true
done

real_sink_regex='^(alsa_output|bluez_output|bluez_sink)'

for i in $(seq 1 "$MAX_WAIT"); do
    current_default=$(pactl info 2>/dev/null | awk -F': ' '/^Default Sink:/ {print $2}')

    if [[ "$current_default" =~ ^(alsa_output|bluez_output|bluez_sink) ]]; then
        pactl set-sink-mute "$current_default" false 2>/dev/null || true
        pactl set-sink-volume "$current_default" 100% 2>/dev/null || true
        echo "audio ready after ${i}s: default sink=${current_default} (unmuted, volume=100%)"
        exit 0
    fi

    # Real sink exists but isn't the default? Force it.
    real_sink=$(pactl list short sinks 2>/dev/null | awk '{print $2}' | grep -E "$real_sink_regex" | head -1)
    if [ -n "$real_sink" ]; then
        if pactl set-default-sink "$real_sink" 2>/dev/null; then
            pactl set-sink-mute "$real_sink" false 2>/dev/null || true
            pactl set-sink-volume "$real_sink" 100% 2>/dev/null || true
            echo "audio ready after ${i}s: forced default sink to ${real_sink} (was ${current_default:-unset}, unmuted, volume=100%)"
            exit 0
        fi
    fi

    # Every 5s, log what's visible for troubleshooting.
    if [ $((i % 5)) -eq 0 ]; then
        sinks=$(pactl list short sinks 2>/dev/null | awk '{print $2}' | paste -sd, -)
        echo "waiting (${i}s): default=${current_default:-unset}, sinks=[${sinks:-none}]"
    fi
    sleep "$INTERVAL"
done

echo "WARN: no real audio sink appeared in ${MAX_WAIT}s; starting anyway"
exit 0
