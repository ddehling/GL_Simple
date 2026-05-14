"""Weight of Light — LD2412 radar router and per-frame derivation.

Each WoL box has an LD2412 24 GHz mmWave radar wired to its Q1+Q2 UART.
The firmware emits an OSC bundle every ~200 ms containing the sensor's
current detection state, distances, energies, and per-gate energy
arrays.  This module receives those messages, maintains per-object
state, and computes derived signals (smoothed distance, approach
speed, presence weight, baseline-subtracted gate energies) so effect
authors can read clean values without re-implementing the filtering.

OSC addresses consumed (already routed by `button_router._dispatch`
when ``kind == "radar"``):

    /wol/<host>/radar           <int state 0..3>
    /wol/<host>/radar/m         <int moving distance cm>
    /wol/<host>/radar/s         <int stationary distance cm>
    /wol/<host>/radar/d         <int closest detected distance cm>
    /wol/<host>/radar/me        <int moving energy 0..100>
    /wol/<host>/radar/se        <int stationary energy 0..100>
    /wol/<host>/radar/gm        <14 ints, moving gate energies, gates 0..13>
    /wol/<host>/radar/gs        <14 ints, stationary gate energies, gates 0..13>

Each gate spans 0.75 m of range, so gate 0 is 0–0.75 m and gate 13 is
9.75–10.5 m.

State exposed under ``env_system.scheduler.state['radar'][object_id]``.
The structure is documented on `_make_record()` below; effect code
should read from it like::

    radar = env_system.scheduler.state.get('radar', {}).get(object_id)
    if radar and radar['online']:
        if radar['approach_speed'] > 30:        # cm/s
            ...                                  # someone walking in
        weight = radar['presence_weight']        # 0..1, slow
        peak  = radar['gate_peak_idx']           # 0..13, where they are
        depth = radar['gs_above_baseline']       # 14-elt list, baseline-subtracted

Underscore-prefixed keys (``_baseline_gs_buckets`` etc.) are private to
this module — don't read them from effects.
"""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Iterable, Sequence


# ---------------------------------------------------------------------------
# Constants — keep in one place so tuning is obvious.
# ---------------------------------------------------------------------------

N_GATES = 14                   # LD2412 reports 14 range gates (0..13)
GATE_M  = 0.75                 # each gate spans this many metres

# EMA time constants (seconds). Tick-rate-independent because we use
# `dt`-aware EMA: alpha = 1 - exp(-dt / tau). After one tau the response
# has reached ~63% of the new target.
TAU_DETECT_DIST_S    = 0.7    # detect_smooth — fast enough to track approach
TAU_PRESENCE_S       = 1.5    # presence_weight — slow enough to ignore blips
TAU_PRESENCE_DECAY_S = 3.0    # decay factor when sensor offline / no target

# Presence-weight ramp: 1.0 when very close, 0.0 when far away. Linear
# between. Tune per installation; defaults assume the salient side is
# the ~0.5–3.5 m approach corridor.
PRESENCE_NEAR_CM = 50.0
PRESENCE_FAR_CM  = 350.0

# Online flag flips off after this long without a packet.
ONLINE_TIMEOUT_S = 3.0

# Rolling baseline: 20 buckets × 30 s = 600 s (10 min) window. Each
# bucket records the per-gate minimum observed during its 30-s slice;
# the live baseline is the elementwise minimum across all buckets plus
# the current in-progress bucket.
#
# Why minimum, not mean? When the field of view is empty the gate
# energies converge to the static room reflection. When a person is
# present the energies go up. The window-minimum captures the "no
# presence" floor, which is exactly the baseline we want to subtract
# off so above-baseline values represent real presence energy.
BASELINE_BUCKET_S  = 30.0
BASELINE_WINDOW_S  = 600.0
BASELINE_BUCKETS   = int(BASELINE_WINDOW_S / BASELINE_BUCKET_S)   # 20

# Initial baseline is "very high" so the first real observation always
# wins on min(). 200 is comfortably above the sensor's 0..100 range.
_BASELINE_INIT = 200


# ---------------------------------------------------------------------------
# Per-object record
# ---------------------------------------------------------------------------

def _make_record() -> dict:
    """Per-object state record. Public keys are stable API for effect
    authors; underscore-prefixed keys are this module's bookkeeping
    and may change. Numeric defaults are zeros so a freshly-seen
    object reads cleanly even before its first packet."""
    return {
        # ---- Raw, written by handle_osc when the corresponding message arrives.
        'state':         0,                 # 0=none 1=moving 2=stationary 3=both
        'moving_cm':     0,
        'static_cm':     0,
        'detect_cm':     0,
        'moving_e':      0,                 # 0..100 overall energies
        'static_e':      0,
        'gm':            [0] * N_GATES,     # raw gate arrays (latest packet)
        'gs':            [0] * N_GATES,
        'last_packet_t': 0.0,               # time.time() of last update

        # ---- Derived, refreshed each frame by tick().
        'detect_smooth':     0.0,           # EMA-smoothed detect_cm
        'approach_speed':    0.0,           # cm/s, positive when approaching
        'presence_weight':   0.0,           # 0..1, slow-onset presence
        'gate_peak_idx':     0,             # 0..13, gate with the largest
                                            #         baseline-subtracted static energy
        'gs_above_baseline': [0] * N_GATES, # gs - baseline_gs, clamped >= 0
        'gm_above_baseline': [0] * N_GATES, # gm - baseline_gm, clamped >= 0
        'online':            False,         # last packet within ONLINE_TIMEOUT_S

        # ---- Baseline state (private). Two separate baselines because
        # moving and stationary gates have very different floors — moving
        # tends to ~0 in an empty space, stationary picks up reflections.
        '_baseline_gs_buckets': deque(maxlen=BASELINE_BUCKETS),
        '_baseline_gm_buckets': deque(maxlen=BASELINE_BUCKETS),
        '_current_gs_min':      [_BASELINE_INIT] * N_GATES,
        '_current_gm_min':      [_BASELINE_INIT] * N_GATES,
        '_current_bucket_t':    0.0,
        '_baseline_gs':         [0] * N_GATES,   # latest computed (min over window)
        '_baseline_gm':         [0] * N_GATES,

        # ---- Tick bookkeeping (private).
        '_prev_smooth':  0.0,
        '_prev_tick_t':  0.0,
    }


def _radar_state(env_system) -> dict:
    """Get-or-create the radar state dict on the scheduler. Lazily
    initialised so this module imports cleanly even if the engine boots
    without OSC."""
    s = env_system.scheduler.state
    rs = s.get('radar')
    if rs is None:
        rs = {}
        s['radar'] = rs
    return rs


def _record_for(env_system, object_id: int) -> dict:
    rs = _radar_state(env_system)
    rec = rs.get(object_id)
    if rec is None:
        rec = _make_record()
        rs[object_id] = rec
    return rec


# ---------------------------------------------------------------------------
# OSC dispatch — called from button_router._dispatch when kind == "radar"
# ---------------------------------------------------------------------------

def handle_osc(env_system, object_id: int,
               sub_parts: Sequence[str], *args: Any) -> None:
    """Update the raw fields of the per-object record from one OSC
    message. ``sub_parts`` is the address remainder after
    ``/wol/<host>/radar``: empty for the bare state message,
    ``['m']`` / ``['s']`` / ``['d']`` / ``['me']`` / ``['se']`` for the
    summary fields, and ``['gm']`` / ``['gs']`` for the 14-element
    gate arrays.

    Per-frame derivation (smoothing, baseline, presence, etc.)
    happens in ``tick()`` — keep this hot path cheap so high-rate
    OSC bursts don't stall the listener thread.
    """
    rec = _record_for(env_system, object_id)
    rec['last_packet_t'] = time.time()

    if not sub_parts:
        # /wol/<host>/radar — bare = detection state int
        if args:
            rec['state'] = _safe_int(args[0])
        return

    key = sub_parts[0]
    if key == 'm':
        if args: rec['moving_cm'] = _safe_int(args[0])
    elif key == 's':
        if args: rec['static_cm'] = _safe_int(args[0])
    elif key == 'd':
        if args: rec['detect_cm'] = _safe_int(args[0])
    elif key == 'me':
        if args: rec['moving_e']  = _safe_int(args[0])
    elif key == 'se':
        if args: rec['static_e']  = _safe_int(args[0])
    elif key == 'gm':
        rec['gm'] = _ints_padded(args, N_GATES)
    elif key == 'gs':
        rec['gs'] = _ints_padded(args, N_GATES)
    # Anything else (e.g. legacy per-gate /gm/<n> from older firmware
    # builds) is silently ignored — the bundle form covers everything
    # the current firmware emits.


# ---------------------------------------------------------------------------
# Per-frame tick — called from random_events.run() once per frame
# ---------------------------------------------------------------------------

def tick(env_system) -> None:
    """Refresh derived values for every known object. Cheap (a few
    dozen floats per object), runs every frame from the existing
    per-frame hook."""
    now = time.time()
    rs = _radar_state(env_system)
    for rec in rs.values():
        _update_one(rec, now)


def _update_one(rec: dict, now: float) -> None:
    last = rec['last_packet_t']
    rec['online'] = (last > 0.0) and (now - last) <= ONLINE_TIMEOUT_S

    # Compute frame dt in seconds (clamped to avoid blow-ups on first tick).
    prev_t = rec['_prev_tick_t']
    dt = (now - prev_t) if prev_t > 0.0 else 0.0
    if dt <= 0.0 or dt > 1.0:    # absurdly long? skip derivative this tick
        dt = 1.0 / 40.0          # assume 40 fps
    rec['_prev_tick_t'] = now

    if not rec['online']:
        # No fresh data — fade derived signals so effects clean up
        # gracefully rather than holding a stale "approaching" reading.
        rec['approach_speed']  = 0.0
        rec['presence_weight'] = _ema_step(
            rec['presence_weight'], 0.0, dt, TAU_PRESENCE_DECAY_S)
        return

    # ---- Smooth detect_cm into detect_smooth -------------------------------
    raw = float(rec['detect_cm'])
    if rec['_prev_smooth'] <= 0.0:
        # Bootstrap on first valid sample so we don't spend a whole tau
        # ramping up from 0.
        rec['detect_smooth'] = raw
    else:
        rec['detect_smooth'] = _ema_step(
            rec['detect_smooth'], raw, dt, TAU_DETECT_DIST_S)

    # ---- Approach speed (cm/s, positive when distance shrinking) -----------
    delta = rec['_prev_smooth'] - rec['detect_smooth']
    rec['_prev_smooth'] = rec['detect_smooth']
    rec['approach_speed'] = max(0.0, delta / dt) if dt > 0 else 0.0

    # ---- Presence weight ---------------------------------------------------
    # Target value: 1 if a stationary or both-target is present at near
    # range, falling linearly to 0 at far range; 0 if no target. EMA on
    # this gives a "soft on" / "soft off" curve that won't flicker on
    # marginal detection.
    if rec['state'] in (2, 3):
        d = rec['detect_smooth']
        if d <= PRESENCE_NEAR_CM:
            target = 1.0
        elif d >= PRESENCE_FAR_CM:
            target = 0.0
        else:
            target = 1.0 - (d - PRESENCE_NEAR_CM) / (PRESENCE_FAR_CM - PRESENCE_NEAR_CM)
    else:
        target = 0.0
    rec['presence_weight'] = _ema_step(
        rec['presence_weight'], target, dt, TAU_PRESENCE_S)

    # ---- Rolling baseline + above-baseline gate arrays ---------------------
    _update_baseline(rec, now)
    bs = rec['_baseline_gs']
    bm = rec['_baseline_gm']
    gs = rec['gs']
    gm = rec['gm']
    for i in range(N_GATES):
        v_s = gs[i] - bs[i]
        v_m = gm[i] - bm[i]
        rec['gs_above_baseline'][i] = v_s if v_s > 0 else 0
        rec['gm_above_baseline'][i] = v_m if v_m > 0 else 0

    # ---- Peak gate index (using static, baseline-subtracted) ---------------
    peak_idx = 0
    peak_val = -1
    for i in range(N_GATES):
        v = rec['gs_above_baseline'][i]
        if v > peak_val:
            peak_val = v
            peak_idx = i
    rec['gate_peak_idx'] = peak_idx


# ---------------------------------------------------------------------------
# Rolling baseline implementation
# ---------------------------------------------------------------------------

def _update_baseline(rec: dict, now: float) -> None:
    """20 buckets × 30 s rolling minimum baseline. Each bucket holds the
    per-gate minimum observed during its 30-second slice; we periodically
    rotate the current bucket into the history deque (which auto-drops
    older entries beyond the 20-bucket cap) and start a fresh one. The
    live baseline is the elementwise min across the deque + the current
    in-progress bucket."""
    if rec['_current_bucket_t'] == 0.0:
        rec['_current_bucket_t'] = now

    # Update current bucket's per-gate min from the latest packet.
    cur_s = rec['_current_gs_min']
    cur_m = rec['_current_gm_min']
    gs    = rec['gs']
    gm    = rec['gm']
    for i in range(N_GATES):
        if gs[i] < cur_s[i]: cur_s[i] = gs[i]
        if gm[i] < cur_m[i]: cur_m[i] = gm[i]

    # Roll the bucket if its time slice has elapsed.
    if (now - rec['_current_bucket_t']) >= BASELINE_BUCKET_S:
        # deque(maxlen=...) auto-evicts the oldest when full, so we
        # don't need explicit pruning of expired buckets.
        rec['_baseline_gs_buckets'].append(list(cur_s))
        rec['_baseline_gm_buckets'].append(list(cur_m))
        rec['_current_gs_min'] = [_BASELINE_INIT] * N_GATES
        rec['_current_gm_min'] = [_BASELINE_INIT] * N_GATES
        rec['_current_bucket_t'] = now

    # Recompute live baseline = min across all buckets + current.
    bs = list(rec['_current_gs_min'])
    bm = list(rec['_current_gm_min'])
    for buc in rec['_baseline_gs_buckets']:
        for i in range(N_GATES):
            if buc[i] < bs[i]: bs[i] = buc[i]
    for buc in rec['_baseline_gm_buckets']:
        for i in range(N_GATES):
            if buc[i] < bm[i]: bm[i] = buc[i]

    # Cap at 100 (sensor's documented range) — anything higher is the
    # _BASELINE_INIT sentinel from buckets that haven't seen real data.
    for i in range(N_GATES):
        if bs[i] > 100: bs[i] = 0
        if bm[i] > 100: bm[i] = 0
    rec['_baseline_gs'] = bs
    rec['_baseline_gm'] = bm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema_step(prev: float, new: float, dt: float, tau: float) -> float:
    """Time-aware exponential moving average. ``tau`` is the smoothing
    time constant in seconds; after one tau the response reaches ~63 %
    of the step. dt-independent so framerate fluctuations don't change
    perceived response speed."""
    if tau <= 0.0 or dt <= 0.0:
        return new
    alpha = 1.0 - math.exp(-dt / tau)
    return prev + alpha * (new - prev)


def _safe_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _ints_padded(args: Iterable[Any], n: int) -> list[int]:
    """Coerce up-to-n OSC args to ints, padding with 0 if the bundle
    arrived short. Trims excess silently."""
    out = [0] * n
    for i, v in enumerate(args):
        if i >= n: break
        out[i] = _safe_int(v)
    return out
