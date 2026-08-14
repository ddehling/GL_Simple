#!/usr/bin/env bash
# Ensure the CPU governor is `performance` on show machines.
#
# Why: the audio render producer must stay ahead of the speakers. On the
# N150 show box (2026-08-14, Burning Man prep) the default `powersave`
# governor plus a saturated 4-core load let stem-dual seams drain the
# entire render-ahead ring - multi-second audible stutter right after
# transitions (logged as audio_starved in logs/dj_*.jsonl). `performance`
# holds clocks up between bursts and avoids deep C-state wake latency.
#
# The real persistence is the cpu-performance.service systemd unit that
# bin/linux-install.sh installs and enables (survives reboots, needs no
# runtime sudo). This launch-time script is the safety net for machines
# that were updated rather than reinstalled: it WARNS loudly and prints
# the fix, but never blocks the launch (always exits 0). No-op on
# machines without cpufreq (VMs, containers, macOS).

set -u

GOV_FILE=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
[ -r "$GOV_FILE" ] || exit 0            # no cpufreq here - nothing to do

GOV="$(cat "$GOV_FILE" 2>/dev/null || true)"
[ "$GOV" = "performance" ] && exit 0    # already right - silent no-op

# Try the enabled unit first (covers "unit installed but governor was
# changed at runtime"); sudo -n so an unattended boot never blocks.
if systemctl list-unit-files cpu-performance.service >/dev/null 2>&1 \
        && sudo -n systemctl start cpu-performance.service 2>/dev/null; then
    echo "[cpu] governor was '$GOV' - restored to performance via cpu-performance.service"
    exit 0
fi

echo "[cpu] WARNING: CPU governor is '$GOV', not 'performance'."
echo "[cpu] Audio can stutter on heavy seams (stem duals). Fix once with:"
echo "[cpu]     sudo cpupower frequency-set -g performance"
echo "[cpu] or re-run bin/linux-install.sh to install the persistent"
echo "[cpu] cpu-performance.service unit."
exit 0
