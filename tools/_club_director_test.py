"""Offline test for the club music director v2 (steering + choreography).

Simulates the scheduler contract with scripted energy/build/drop signals
and asserts: heat/profile-table coverage, graph-respecting energy
corrections, phrase-aligned cuts, no churn in matched rooms, drop
snap-cuts with duration override, one-shot gating by floor heat, build
hold + tension, drop slam, and percentile self-calibration.

Note: the night-arc bias means exact target-heat values vary with the
wall clock (by up to +-0.25); assertions are behavioral with slack.

Usage: python tools/_club_director_test.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from projects.fan import weather_params as wp
from projects.fan.shaders.club_director import (shader_club_director,
                                                SCENE_HEAT, SCENE_PROFILE)

failures = []
def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)

DT = 1.0 / 40.0

def run(seconds, state, out, energy, build=0.0, drop_at=None, t0=0.0,
        conf=0.0, phrase_period=None):
    t = t0
    reqs, events, durs = [], [], []
    n = int(seconds / DT)
    for _ in range(n):
        t += DT
        state['elapsed_time'] = t
        out['audio_energy'] = energy
        out['build_level'] = build
        out['beat_confidence'] = conf
        if phrase_period:
            out['phrase_phase'] = (t / phrase_period) % 1.0
        fire = drop_at is not None and abs(t - drop_at) < DT * 0.6
        out['drop'] = fire
        out['drop_decay'] = 1.0 if fire else out.get('drop_decay', 0.0) * 0.94
        out['club_energy'] = 0.8
        shader_club_director(state, out)
        state['count'] += 1
        r = out.pop('_weather_transition_request', None)
        out.pop('_weather_transition_node', None)
        d = out.pop('_weather_transition_duration', None)
        if r:
            reqs.append((t, r))
            durs.append(d)
            out['current_weather_state'] = r
        e = out.pop('_event_request', None)
        if e:
            events.append((t, e[0]))
    return t, reqs, events, durs

def fresh(room):
    state = {'count': 0, 'elapsed_time': 0.0}
    out = {'current_weather_state': room, 'club_energy': 0.8,
           'audio_energy': 0.0, 'build_level': 0.0, 'drop': False,
           'drop_decay': 0.0, 'beat_confidence': 0.0, 'phrase_phase': 0.0,
           'bass_punch': 0.0, 'mid_punch': 0.0, 'high_punch': 0.0}
    shader_club_director(state, out)
    state['count'] = 1
    return state, out

print("club_director v2 offline test\n")

# 0. Tables cover exactly the set's states.
states = set(wp.WEATHER_SETS['club']['states'])
check("heat table coverage", set(SCENE_HEAT) == states,
      f"missing={states - set(SCENE_HEAT)} extra={set(SCENE_HEAT) - states}")
check("profile table coverage", set(SCENE_PROFILE) == states)

# 1. High energy in a cold room -> corrective jump hotter (no beat grid ->
# immediate).
state, out = fresh('club_afterhours')
t, reqs, _, _ = run(120.0, state, out, energy=0.6)
check("cold room corrected", len(reqs) >= 1 and SCENE_HEAT[reqs[0][1]] > 0.3,
      f"requests={[(round(a,1), b) for a, b in reqs[:3]]}")

# 2. Matched room -> no churn.
state, out = fresh('club_orbitarium')
t, reqs, _, _ = run(120.0, state, out, energy=0.40)
check("matched room left alone", len(reqs) == 0, f"requests={reqs}")

# 3. Phrase alignment: with a confident beat grid, the correction waits
# for the 16-beat wrap instead of firing at the 15s mismatch mark.
state, out = fresh('club_afterhours')
t, reqs, _, _ = run(40.0, state, out, energy=0.6, conf=0.8, phrase_period=7.5)
check("correction is phrase-aligned",
      len(reqs) >= 1 and abs((reqs[0][0] % 7.5)) < 0.35,
      f"first request at t={reqs[0][0]:.2f} (phrase period 7.5s)" if reqs
      else "no request fired")

# 4. Drop in a cold room with a hot floor -> SNAP cut (short duration).
state, out = fresh('club_mindblob')
t, reqs, events, durs = run(60.0, state, out, energy=0.45)
check("no churn while warming", len(reqs) == 0, f"requests={reqs}")
out['current_weather_state'] = 'club_pearl'
t, reqs2, events2, durs2 = run(4.0, state, out, energy=0.45,
                               drop_at=t + 1.0, t0=t)
jumped_hot = any(SCENE_HEAT[r] >= 0.7 for _, r in reqs2)
check("drop snap-cuts hot", jumped_hot, f"requests={reqs2}")
check("snap uses duration override",
      any(d is not None and d < 2.0 for d in durs2), f"durations={durs2}")
check("drop fires room-suited one-shot", len(events2) >= 1,
      f"events={events2}")

# 5. One-shot gating: a drop on a COLD floor fires nothing.
state, out = fresh('club_afterhours')
t, _, events, _ = run(30.0, state, out, energy=0.05, drop_at=25.0)
check("cold-floor drop holds one-shot back", len(events) == 0,
      f"events={events}")

# 6. Build holds transitions + compresses; drop slams.
state, out = fresh('club_runway')
t, _, _, _ = run(5.0, state, out, energy=0.4, build=0.9)
check("build holds transitions",
      out.get('_transition_hold_until', 0) > time.time() - 1)
check("build compresses energy", out['club_energy'] < 0.8,
      f"club_energy={out['club_energy']:.2f}")
check("build pre-selects drop destination",
      state.get('_drop_dest') in SCENE_HEAT, f"dest={state.get('_drop_dest')}")
out['_transition_hold_until'] = 0
t, _, _, _ = run(0.2, state, out, energy=0.4, drop_at=t + 0.05, t0=t)
check("drop slams energy", out['club_energy'] > 0.85,
      f"club_energy={out['club_energy']:.2f}")

# 7. Self-calibration: percentiles adapt to the observed energy range.
state, out = fresh('club_orbitarium')
t = 0.0
for seg in range(10):                    # alternate quiet/loud for ~6.7 min
    e = 0.30 if seg % 2 == 0 else 0.62
    t, _, _, _ = run(40.0, state, out, energy=e, t0=t)
check("calibration adapts",
      state['_cal_lo'] > 0.2 and state['_cal_hi'] > 0.55,
      f"cal=[{state['_cal_lo']:.2f}, {state['_cal_hi']:.2f}]")

print()
if failures:
    print("FAILED:", failures); sys.exit(1)
print("ALL PASS")
