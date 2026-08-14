"""Offline test for the club music director v2 (steering + choreography).

Simulates the scheduler contract with scripted energy/build/drop signals
and asserts: heat/profile-table coverage, graph-respecting energy
corrections, phrase-aligned cuts, no corrective churn in matched rooms,
drop snap-cuts with duration override, one-shot gating by floor heat,
build hold + tension, drop slam, percentile self-calibration, and the
visit-aware drift takeover (the director owns ALL room movement now -
the engine's Switch_rate roll is frozen every frame, and the director's
own drift roll must keep traveling instead of recycling the transition
graph's local cliques).

Note: the night-arc bias means exact target-heat values vary with the
wall clock (by up to +-0.25); assertions are behavioral with slack.

Usage: python tools/tests/_club_director_test.py
"""
import os
import random
import sys
import time

random.seed(20260813)   # drift roll timing/picks deterministic per run

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from projects.fan import weather_params as wp
from projects.fan.shaders.club_director import (shader_club_director,
                                                SCENE_HEAT, SCENE_PROFILE,
                                                SCENE_BUSY)

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
check("busy table coverage", set(SCENE_BUSY) == states,
      f"missing={states - set(SCENE_BUSY)}")

# 1. High energy in a cold room -> corrective jump hotter (no beat grid ->
# immediate).
state, out = fresh('club_afterhours')
t, reqs, _, _ = run(120.0, state, out, energy=0.6)
check("cold room corrected", len(reqs) >= 1 and SCENE_HEAT[reqs[0][1]] > 0.3,
      f"requests={[(round(a,1), b) for a, b in reqs[:3]]}")

# 2. Matched room -> no CORRECTIVE churn. The visit-aware drift still
#    roams (it replaced the engine's Switch_rate roll 1:1), but never as
#    an energy correction.
state, out = fresh('club_orbitarium')
t, reqs, _, _ = run(120.0, state, out, energy=0.40)
notes = " | ".join(j['msg'] for j in state['_journal'])
check("matched room: no corrections",
      'stepping up' not in notes and 'easing down' not in notes
      and 'stepped up' not in notes and 'cooled down' not in notes,
      f"journal={notes[-200:]}")
check("drift roams the matched room", len(reqs) >= 1,
      f"requests={[(round(a, 1), b) for a, b in reqs]}")

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
check("warming: only gentle drift, no snaps or corrections",
      all(d is None for d in durs), f"requests={reqs} durations={durs}")
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

# 8. Theme lean is BINDING: setting a theme moves the night into the family,
#    every subsequent pick stays inside it, the RNG drift is frozen, and the
#    director rotates the family's rooms on its own.
from projects.fan.shaders.club_director import SCENE_THEME
state, out = fresh("club_deep_space")          # cosmic room, molten lean
out['club_theme'] = 'molten'
t, reqs, _, _ = run(20.0, state, out, energy=0.5)
check("theme move fires", len(reqs) >= 1 and reqs[0][1] in SCENE_THEME['molten'],
      f"first request={reqs[0] if reqs else None}")
check("drift frozen while leaned",
      out.get('_transition_hold_until', 0.0) > time.time() - 1.0,
      f"hold_until={out.get('_transition_hold_until', 0.0):.0f}")
t, reqs2, _, _ = run(300.0, state, out, energy=0.5, t0=t)
all_moves = [r for _, r in reqs2]
check("every leaned pick stays molten",
      all_moves and all(r in SCENE_THEME['molten'] for r in all_moves),
      f"moves={[m.replace('club_', '') for m in all_moves]}")
check("leaned wander rotates rooms", len(set(all_moves)) >= 2,
      f"{len(set(all_moves))} distinct rooms in 5min")
# Release: the hold PERSISTS - the director owns drift for the whole
# night now, leaned or not (the engine roll stays frozen).
out['club_theme'] = ''
t, _, _, _ = run(5.0, state, out, energy=0.5, t0=t)
check("hold persists after release (director owns all drift)",
      out.get('_transition_hold_until', 0.0) > time.time() - 1.0,
      f"hold_until={out.get('_transition_hold_until', 0.0):.0f}")

# 9. Silence hush: the music stopping dims every pattern level toward
#    embers over a few seconds; sound returning wakes the room fast.
#    (Levels are reset each frame, mirroring send_variables.)
state, out = fresh("club_orbitarium")
t = 0.0
def hush_run(seconds, mood, t):
    lvl = 0.9
    for _ in range(int(seconds / DT)):
        t += DT
        state['elapsed_time'] = t
        out['orb_level'] = 0.9
        out['music_mood'] = mood
        out['audio_energy'] = 0.0 if mood == 'silent' else 0.5
        out['club_energy'] = 0.8
        shader_club_director(state, out)
        state['count'] += 1
        out.pop('_weather_transition_request', None)
        out.pop('_weather_transition_duration', None)
        lvl = out['orb_level']
    return lvl, t
lvl, t = hush_run(12.0, 'silent', t)
check("silence hushes the levels", lvl < 0.9 * 0.16,
      f"orb_level={lvl:.3f} after 12s of silence (from 0.9)")
lvl, t = hush_run(3.0, 'groove', t)
check("sound wakes the room", lvl > 0.9 * 0.9,
      f"orb_level={lvl:.3f} 3s after sound returns")

# 10. Earned night progression: the phase advances by danced heat -
#     a hot floor moves the night several times faster than a dead one.
state, out = fresh("club_orbitarium")
out['season'] = 0.10
t, _, _, _ = run(120.0, state, out, energy=0.7)
hot_adv = out['_season_autopilot'] - 0.10
state, out = fresh("club_orbitarium")
out['season'] = 0.10
t, _, _, _ = run(120.0, state, out, energy=0.0)
dead_adv = out['_season_autopilot'] - 0.10
check("night phase is earned", hot_adv > dead_adv * 2.5 and hot_adv > 0,
      f"2min advance: hot={hot_adv*100:.3f}% vs dead={dead_adv*100:.3f}% of the cycle")

# 11. Autonomous-DJ coupling: the DJ's published arc pulls the night phase
#     (director stays the only _season_autopilot writer), and the DJ's
#     next-seam ETA pre-arms the room exactly like a heard build.
state, out = fresh("club_orbitarium")
out['season'] = 0.10
out['dj_active'] = True
out['dj_arc_phase'] = 0.50
out['dj_arc_heat'] = 0.9
t, _, _, _ = run(120.0, state, out, energy=0.3)
pulled = out['_season_autopilot']
state2, out2 = fresh("club_orbitarium")
out2['season'] = 0.10
t, _, _, _ = run(120.0, state2, out2, energy=0.3)
alone = out2['_season_autopilot']
d_pull = abs((pulled - 0.50 + 0.5) % 1.0 - 0.5)
d_alone = abs((alone - 0.50 + 0.5) % 1.0 - 0.5)
check("dj arc pulls the night phase", d_pull < d_alone * 0.5,
      f"2min: with DJ {pulled:.3f} vs alone {alone:.3f} (target 0.50)")

state, out = fresh("club_runway")
out['dj_active'] = True
out['dj_next_drop_eta'] = 6.0
t, _, _, _ = run(3.0, state, out, energy=0.4)
check("dj drop-eta pre-arms the room",
      out.get('_transition_hold_until', 0) > time.time() - 1
      and state.get('_drop_dest') in SCENE_HEAT,
      f"dest={state.get('_drop_dest')}")

# ---- DJ swap choreography: the move matches the seam's character ---------
def _handover(style, energy=0.5, room="club_runway", last_jump=-1e9,
              room_since=None):
    state, out = fresh(room)
    out['dj_active'] = True
    out['dj_swap_eta'] = 3.0
    run(2.0, state, out, energy=energy)   # counting down, pre-armed
    out['dj_swap_eta'] = 0.4
    run(0.5, state, out, energy=energy)
    out['dj_swap_eta'] = None             # the handover landed
    out['dj_style'] = style
    state['_last_jump_t'] = last_jump     # -1e9 = ancient (refresh due)
    # Room age drives the refresh; keep it separately settable so the
    # "fresh room" case is fresh for the DRIFT roll too.
    state['_room_since'] = last_jump if room_since is None else room_since
    _, reqs, _, durs = run(1.0, state, out, energy=energy)
    return reqs, durs

reqs, durs = _handover('cut_at_drop')
check("punchy handover snap-cuts",
      len(reqs) == 1 and durs[0] == 0.7,
      f"reqs={reqs} durs={durs}")

reqs, durs = _handover('bass_swap')      # runway 0.72 vs mid target: glide
check("blend handover glides (no snap)",
      len(reqs) == 1 and durs[0] == 12.0,
      f"reqs={reqs} durs={durs}")

reqs, durs = _handover('long_fade', last_jump=-58.0, room_since=2.0)
check("fresh room: long fade rides in place", len(reqs) == 0,
      f"reqs={reqs} durs={durs}")

# STALE-ROOM REFRESH: heat fits, but the room hasn't changed in ages -
# the next musical handover (even a fade) glides somewhere new.
reqs, durs = _handover('long_fade', last_jump=-1e9)
check("stale room: fade handover freshens with a glide",
      len(reqs) == 1 and durs[0] == 12.0,
      f"reqs={reqs} durs={durs}")

# 12. CLIQUE RESISTANCE: parked at matched heat in the molten corner of
#     the transition graph (hearth/geyser/furnace/smoke_machine list each
#     other), the visit-aware drift must keep TRAVELING - the blind
#     engine roll it replaced recycled that clique for whole stretches
#     ('do they churn between a small number of rooms' - yes, it did).
state, out = fresh('club_hearth')
t, reqs, _, _ = run(600.0, state, out, energy=0.30)
walk = [r for _, r in reqs]
check("drift keeps rolling at engine cadence", len(walk) >= 8,
      f"{len(walk)} moves in 10min")
check("drift walk is diverse, not a clique",
      len(set(walk)) >= min(len(walk), 6),
      f"{len(set(walk))} distinct rooms in {len(walk)} moves: "
      f"{[w.replace('club_', '') for w in walk]}")
pingpong = sum(1 for i in range(2, len(walk)) if walk[i] == walk[i - 2])
check("no A-B-A ping-pong", pingpong <= 1,
      f"{pingpong} immediate revisits in {len(walk)} moves")

print()
if failures:
    print("FAILED:", failures); sys.exit(1)
print("ALL PASS")
