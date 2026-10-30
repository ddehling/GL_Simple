"""THE LIGHTS ARC - what RECORD mode tells the club's lights about the night (docs/DJ_SETS.md section 3, "The lights
get the night too", 2026-10-28).

Record mode used to publish dj_arc_phase = 0.0 all night, and club_director blends its night phase toward that
number, so the club sat at "Doors" all night. The fix: with no plan the phase is REMOVED (club_director's "no lean")
and the heat stays the number 0.5 (a None heat once killed the render loop); a planned night publishes its progress
and its chapter's energy (the status 'plan' block's plan_progress / plan_heat).

  HELPER        Stories_OGL.EnvironmentalSystem._dj_arc_outstate on fake Directors: record mode without a plan ->
                phase None (removed), heat 0.5; with plan keys -> those, clamped (phase held under 1.0, it is read
                mod 1); the layered engines unchanged (their arc as before, 0.0 for a missing progress)
  MIRROR        the mirror's apply: a phase left in outstate by an earlier engine is removed
  CLUB          club_director (projects/fan) fed the old 0.0 drags a Peak-time night toward Doors; fed the fix it
                keeps its own earned phase, and heat 0.5 leans the room exactly as no heat at all
  LIVE          (with --music) a real Director in record mode, headless (mixer pulled by hand, no audio device):
                its status() through the helper gives no phase and heat 0.5, and the playing engine's outstate
                does not bring a phase back

    python tools/tests/_dj_lights_arc.py [--music D:/Devel/music] [--seconds 30]

Prints ALL OK or FAIL (exit 1).
"""
import os
import random
import sys
import time
import types

os.environ["DJ_NO_PLAY_LOG"] = "1"
os.environ["DJ_NO_MIDI"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)
    if not cond:
        failures.append(name)


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def apply(state, keys):
    """The mirror's own apply (Stories_OGL._dj_mirror_status): None removes the key."""
    for k, v in keys.items():
        if v is None:
            state.pop(k, None)
        else:
            state[k] = v


def circ(a, b):
    return abs(((a - b + 0.5) % 1.0) - 0.5)


def main():
    from Stories_OGL import EnvironmentalSystem
    arc_of = EnvironmentalSystem._dj_arc_outstate
    rec = types.SimpleNamespace(player=object())       # a Director in RECORD mode
    lay = types.SimpleNamespace(player=None)           # the layered engines / the autoDJ under the Director
    # what Director.status() gives in record mode today: arc progress 0.0 (no layered engine), target None, 'plan' = the cut
    cut = {"state": "planned", "kind": "blend", "in_s": 40.0, "swap_in_s": 70.0, "end_in_s": 90.0}
    today = {"arc": {"progress": 0.0, "target": None}, "plan": cut}

    print("HELPER")
    o = arc_of(rec, today)
    check("record, no plan: no phase", o["dj_arc_phase"] is None, f"{o}")
    check("record, no plan: heat is the number 0.5", isinstance(o["dj_arc_heat"], float) and o["dj_arc_heat"] == 0.5, f"{o}")
    o = arc_of(rec, {"arc": {"progress": 0.0, "target": None}, "plan": None})
    check("record, plan block None: no phase, heat 0.5", o == {"dj_arc_phase": None, "dj_arc_heat": 0.5}, f"{o}")
    o = arc_of(rec, {"arc": {"error": "x"}})
    check("record, arc errored: no phase, heat 0.5", o == {"dj_arc_phase": None, "dj_arc_heat": 0.5}, f"{o}")
    o = arc_of(rec, {"arc": {"progress": 0.0}, "plan": dict(cut, plan_progress=0.42, plan_heat=0.8)})
    check("record, planned night (plan block): its progress and chapter heat", o == {"dj_arc_phase": 0.42, "dj_arc_heat": 0.8}, f"{o}")
    o = arc_of(rec, {"arc": {"progress": 0.0}, "plan": cut, "plan_progress": 0.3, "plan_heat": 0.2})
    check("record, planned night (top level): its progress and chapter heat", o == {"dj_arc_phase": 0.3, "dj_arc_heat": 0.2}, f"{o}")
    o = arc_of(rec, {"plan": {"plan_progress": 1.3, "plan_heat": 1.7}})
    check("record, a night run long: phase held under 1.0 (no wrap to Doors), heat clamped", o["dj_arc_phase"] < 1.0
          and o["dj_arc_phase"] > 0.99 and o["dj_arc_heat"] == 1.0, f"{o}")
    o = arc_of(rec, {"plan": {"plan_progress": 0.5, "plan_heat": None}})
    check("record, planned night without a heat: heat 0.5", o == {"dj_arc_phase": 0.5, "dj_arc_heat": 0.5}, f"{o}")
    o = arc_of(lay, {"arc": {"progress": 0.25, "target": 0.7}, "plan": {"plan_progress": 0.9}})
    check("layered: its own arc, as before", o == {"dj_arc_phase": 0.25, "dj_arc_heat": 0.7}, f"{o}")
    o = arc_of(lay, {"arc": {"error": "x"}})
    check("layered, arc errored: 0.0 / 0.5, as before", o == {"dj_arc_phase": 0.0, "dj_arc_heat": 0.5}, f"{o}")

    print("MIRROR")
    st = {"dj_arc_phase": 0.37, "dj_arc_heat": 0.9, "season": 0.6}
    apply(st, arc_of(rec, today))
    check("a phase an earlier engine left is removed", "dj_arc_phase" not in st and st["dj_arc_heat"] == 0.5
          and st["season"] == 0.6, f"{st}")

    print("CLUB")
    from projects.fan.shaders.club_director import shader_club_director
    DT = 1.0 / 40.0

    def night(arc_keys, seconds=300.0, season=0.6, seed=7):
        random.seed(seed)
        state = {"count": 0, "elapsed_time": 0.0}
        out = {"current_weather_state": "club_orbitarium", "club_energy": 0.8, "audio_energy": 0.3, "build_level": 0.0,
               "drop": False, "drop_decay": 0.0, "beat_confidence": 0.0, "phrase_phase": 0.0, "bass_punch": 0.0,
               "mid_punch": 0.0, "high_punch": 0.0, "season": season, "dj_active": True, "dj_energy": 0.4}
        apply(out, arc_keys)
        heats, t = [], 0.0
        for _ in range(int(seconds / DT)):
            t += DT
            state["elapsed_time"] = t
            out["club_energy"] = 0.8
            shader_club_director(state, out)
            state["count"] += 1
            for k in ("_weather_transition_request", "_weather_transition_node", "_weather_transition_duration", "_event_request"):
                out.pop(k, None)
            heats.append(((out.get("club_director_info") or {}).get("target_heat")))
        return out["_season_autopilot"], heats

    old, _ = night({"dj_arc_phase": 0.0, "dj_arc_heat": 0.5})
    new, heats_fix = night(arc_of(rec, today))
    check("the old 0.0 dragged a Peak-time night toward Doors", circ(old, 0.0) < 0.6 - 0.1,
          f"5 min from 0.60: phase {old:.3f}")
    check("the fix keeps the night's own earned phase", 0.6 <= new < 0.6 + 0.01, f"5 min from 0.60: phase {new:.4f}")
    _, heats_none = night({"dj_arc_phase": None})       # no heat key at all
    check("heat 0.5 leans the room as no heat at all", heats_fix == heats_none,
          f"{sum(1 for a, b in zip(heats_fix, heats_none) if a != b)} of {len(heats_fix)} frames differ")
    planned, _ = night(arc_of(rec, {"plan": {"plan_progress": 0.8, "plan_heat": 0.7}}))
    check("a planned night's progress pulls the lights its way", circ(planned, 0.8) < circ(0.6, 0.8) - 0.05,
          f"5 min from 0.60 toward 0.80: phase {planned:.3f}")

    music = arg("--music", None)
    if music:
        print("LIVE")
        live(music, arg("--seconds", 30.0, float), arc_of)

    print()
    if failures:
        print("FAIL:", failures)
        sys.exit(1)
    print("ALL OK")


def live(music, seconds, arc_of):
    import numpy as np
    from lib.audio_engine import AudioEngine
    from lib.dj import director as D
    engine = AudioEngine()                             # never opened: the mixer is pulled by hand below
    d = D.Director(music, engine=None, theme="groove", attach=engine.attach_track)
    ok = d.start(threaded=False)
    p = getattr(d, "player", None)
    if p is None or not ok:
        check("record mode starts", False, f"{d.last_error}")
        return
    gen = engine._mixer()
    next(gen)
    gen.send(256)
    BLOCK = 4410
    t0, played_blocks, state = time.time(), 0, {"dj_arc_phase": 0.0, "dj_arc_heat": 0.5}
    seen = []
    try:
        while played_blocks * BLOCK / 44100.0 < seconds:
            np.frombuffer(gen.send(BLOCK), dtype=np.float32)
            d.step()
            playing = p.record is not None and ((p.submix.telemetry or {}).get("decks") or {}).get(p.record, {}).get("playing")
            if not playing:
                if time.time() - t0 > 180.0:
                    check("a first record plays within 180 s", False, f"{d.last_error} / {p.last_error}")
                    return
                time.sleep(0.02)
                continue
            played_blocks += 1
            if played_blocks % 10 == 0:                # the mirror's 5 Hz-ish cadence, in rendered time
                dst = d.status()
                apply(state, arc_of(d, dst))
                for k, v in d.outstate_keys().items():  # the mirror applies the engine's outstate after the arc
                    state[k] = v
                seen.append((dst.get("arc") or {}).get("progress"))
                check_tick = ("dj_arc_phase" not in state and isinstance(state.get("dj_arc_heat"), float)
                              and state["dj_arc_heat"] == 0.5)
                if not check_tick:
                    check("every tick: no phase, heat 0.5", False, f"{ {k: state.get(k) for k in ('dj_arc_phase', 'dj_arc_heat')} }")
                    return
        check("every tick: no phase, heat 0.5", True, f"({len(seen)} ticks over {seconds:.0f} s of a record playing)")
        check("the Director's own arc still reads 0.0 in record mode (why record mode must not use it)",
              all(x in (0.0, None) for x in seen), f"progress values {sorted(set(seen))}")
        check("the record player reports active", bool(state.get("dj_active")), f"dj_active={state.get('dj_active')}")
        check("no errors", not d.last_error and not p.last_error, f"{d.last_error} / {p.last_error}")
    finally:
        try:
            d.stop(0.1)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
