"""Offline test for the club music director (scene steering + choreography).

Simulates the scheduler contract with scripted energy/build/drop signals
and asserts: heat-table coverage of the set's states, graph-respecting
energy corrections, no churn in matched rooms, drop jump + one-shot,
build hold + tension, and the drop slam.

Usage: python tools/_club_director_test.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from projects.fan import weather_params as wp
from projects.fan.shaders.club_director import (shader_club_director,
                                                SCENE_HEAT)

failures = []
def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)

# 0. Heat table exactly matches the set's states.
states = set(wp.WEATHER_SETS['club']['states'])
check("heat table coverage", set(SCENE_HEAT) == states,
      f"missing={states - set(SCENE_HEAT)} extra={set(SCENE_HEAT) - states}")

DT = 1.0 / 40.0

def run(seconds, state, out, energy, build=0.0, drop_at=None, t0=0.0):
    t = t0
    reqs, events = [], []
    n = int(seconds / DT)
    for i in range(n):
        t += DT
        state['elapsed_time'] = t
        out['audio_energy'] = energy
        out['build_level'] = build
        fire = drop_at is not None and abs(t - drop_at) < DT * 0.6
        out['drop'] = fire
        out['drop_decay'] = 1.0 if fire else out.get('drop_decay', 0.0) * 0.94
        out['club_energy'] = 0.8            # what send_variables would write
        shader_club_director(state, out)
        state['count'] += 1
        r = out.pop('_weather_transition_request', None)
        out.pop('_weather_transition_node', None)
        if r:
            reqs.append((t, r))
            out['current_weather_state'] = r
        e = out.pop('_event_request', None)
        if e:
            events.append((t, e[0]))
    return t, reqs, events

def fresh(room):
    state = {'count': 0, 'elapsed_time': 0.0}
    out = {'current_weather_state': room, 'club_energy': 0.8,
           'audio_energy': 0.0, 'build_level': 0.0, 'drop': False,
           'drop_decay': 0.0}
    shader_club_director(state, out)
    state['count'] = 1
    return state, out

print("club_director offline test\n")

# 1. High energy in a cold room -> corrective jump to a hotter neighbour.
state, out = fresh('club_afterhours')      # heat 0.12
t, reqs, _ = run(120.0, state, out, energy=0.6)   # target heat -> 1.0
check("cold room corrected", len(reqs) >= 1 and SCENE_HEAT[reqs[0][1]] > 0.3,
      f"requests={[(round(a,1), b) for a, b in reqs[:3]]}")

# 2. Matched room -> no churn.
state, out = fresh('club_orbitarium')      # heat 0.55
t, reqs, _ = run(120.0, state, out, energy=0.40)  # target ~0.55
check("matched room left alone", len(reqs) == 0, f"requests={reqs}")

# 3. Drop in a cold room with a hot floor -> immediate hot jump + one-shot.
# Warm the ema in a MATCHED room so the mismatch corrector stays quiet,
# then teleport to a cold room and drop.
state, out = fresh('club_mindblob')        # heat 0.60, target ~0.67 at e=0.45
t, reqs, events = run(60.0, state, out, energy=0.45)
check("no churn while warming", len(reqs) == 0, f"requests={reqs}")
out['current_weather_state'] = 'club_pearl'    # heat 0.18 << target
t, reqs2, events2 = run(4.0, state, out, energy=0.45, drop_at=t + 1.0, t0=t)
jumped_hot = any(SCENE_HEAT[r] >= 0.7 for _, r in reqs2)
check("drop jumps hot", jumped_hot, f"requests={reqs2}")
check("drop fires one-shot", len(events2) >= 1, f"events={events2}")

# 4. Build holds transitions + compresses club_energy; drop slams it.
state, out = fresh('club_runway')
t, _, _ = run(5.0, state, out, energy=0.4, build=0.9)
held = out.get('_transition_hold_until', 0) > time.time() - 1
check("build holds transitions", held)
check("build compresses energy", out['club_energy'] < 0.8,
      f"club_energy={out['club_energy']:.2f}")
out['_transition_hold_until'] = 0
t, _, _ = run(0.2, state, out, energy=0.4, drop_at=t + 0.05, t0=t)
check("drop slams energy", out['club_energy'] > 0.85,
      f"club_energy={out['club_energy']:.2f}")

print()
if failures:
    print("FAILED:", failures); sys.exit(1)
print("ALL PASS")
