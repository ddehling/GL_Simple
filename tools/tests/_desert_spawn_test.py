"""Measure how often desert cacti and toads actually spawn.

Drives the REAL scheduling path -- ``EnvironmentalSystem.random_events``
bound to a stub env_system, a real ``WeatherSetManager`` on desert_realm,
and the real ``projects/fan/random_events.py`` hook -- so the rates come
from the shipped config rather than a reimplementation of it.

Simulates frames at the project's real target_fps. No GL, no audio.

    python tools/tests/_desert_spawn_test.py [hours]
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import Stories_OGL as S                                   # noqa: E402
from lib.weather_set import WeatherSetManager             # noqa: E402
from core.project import load_project                     # noqa: E402
from core.shader_loader import load_project_shaders       # noqa: E402

# event_map resolves fx.shader_desert_* at import time, so the project's
# shaders must be grafted onto renderer.effects FIRST -- same order the
# app uses (Stories_OGL.py:267).
load_project_shaders(load_project("fan"))

from projects.fan import event_map as fan_event_map       # noqa: E402
from projects.fan import weather_params as fan_wp         # noqa: E402
from projects.fan import random_events as fan_hook        # noqa: E402

FPS = 40.0          # Fan declares no target_fps -> engine default (1/40)
CACTI = ("desert_saguaro", "desert_joshua_tree",
         "desert_prickly_pear", "desert_barrel_cactus")


class _StubScheduler:
    def __init__(self):
        self.state = {"media_root": "media"}

    def schedule_event(self, *a, **kw):
        return None


class _StubEnv:
    """Just enough surface for random_events() and the fan hook."""

    def __init__(self, set_name="desert_realm"):
        self.project = type("P", (), {"raw": {}, "load_hook": lambda self_, n: None})()
        self.weather_set = WeatherSetManager(
            event_map=fan_event_map.EVENT_MAP,
            weather_sets=fan_wp.WEATHER_SETS,
            default_set=set_name,
            weather_state_enum=fan_wp.WeatherState,
        )
        self.scheduler = _StubScheduler()
        self.season = 0.0
        self.weather_state = type("W", (), {"weather_params": {}})()
        self.fired = collections.Counter()      # actually reached the screen
        self.rolled = collections.Counter()     # dice said yes
        self.dropped = collections.Counter()    # lost to the duplicate check
        self.busy_until = collections.defaultdict(float)
        self.now = 0.0                          # sim seconds

    def _schedule_event_from_map(self, name, start, duration, frame_id=0):
        # Confirm the name really resolves to an effect before counting it:
        # a typo'd entry would otherwise inflate the rate silently.
        if self.weather_set.resolve_event(name) is None:
            self.fired[f"UNRESOLVED:{name}"] += 1
            return None
        self.rolled[name] += 1
        # Model EventScheduler's duplicate drop (lib/event_scheduler.py:41):
        # the SAME action already active on the SAME frame is skipped. With
        # long residence times this silently caps the realized rate, so a
        # roll only counts as a spawn if that cactus isn't already on screen.
        if self.now < self.busy_until[(name, frame_id)]:
            self.dropped[name] += 1
            return None
        self.busy_until[(name, frame_id)] = self.now + duration
        self.fired[name] += 1
        return None


def run(hours: float, seed: int = 0):
    np.random.seed(seed)
    env = _StubEnv()
    frames = int(hours * 3600 * FPS)

    cfg = env.weather_set.get_current_set_config()
    print(f"set            : {env.weather_set.current_set}")
    print(f"rate           : {cfg.get('random_event_rate')} /frame @ {FPS:.0f} fps")
    print(f"seasonal lock  : {cfg.get('random_event_seasonal', True)}")
    print(f"list           : {cfg.get('random_events')}")
    print(f"toad rate      : {fan_hook._TOAD_RATE} /frame")
    print(f"simulating     : {hours:g} h  ({frames:,} frames)\n")

    # The engine prints a line per spawn; at these volumes that dominates
    # runtime and buries the summary.
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(frames):
            env.now = i / FPS
            # Real free-running season: one cycle / 3600 s at season_speed 0.5.
            env.season = (env.now / 1800.0 * cfg.get("season_speed", 1.0)) % 1.0
            S.EnvironmentalSystem.random_events(env)
            fan_hook.run(env)

    minutes = hours * 60.0
    print(f"{'event':<24} {'spawned':>8} {'dropped':>8} {'per min':>9} {'gap':>8}")
    print("-" * 60)
    cactus_total = 0
    for name in CACTI + ("desert_toads",):
        n, d = env.fired[name], env.dropped[name]
        cactus_total += n if name in CACTI else 0
        gap = minutes / n if n else float("inf")
        print(f"{name:<24} {n:>8} {d:>8} {n/minutes:>9.3f} {gap:>7.1f}m")
    print("-" * 60)
    gap = minutes / cactus_total if cactus_total else float("inf")
    drops = sum(env.dropped[c] for c in CACTI)
    rolls = sum(env.rolled[c] for c in CACTI)
    print(f"{'CACTUS (any)':<24} {cactus_total:>8} {drops:>8} "
          f"{cactus_total/minutes:>9.3f} {gap:>7.1f}m")
    if rolls:
        print(f"\nduplicate-drop loss: {drops}/{rolls} rolls ({drops/rolls:.1%})")
    # Sum-of-durations over-counts once cacti overlap, so report the mean
    # simultaneous count AND the union (fraction of wall-clock with at
    # least one visible) -- those diverge sharply at high residence.
    dur = cfg.get("random_event_duration", 60)
    mean_count = sum(env.fired[c] for c in CACTI) * dur / (hours * 3600)
    busy_frac = [min(1.0, env.fired[c] * dur / (hours * 3600)) for c in CACTI]
    union = 1.0
    for b in busy_frac:
        union *= (1.0 - b)
    print(f"\nmean cacti on screen : {mean_count:.2f}")
    print(f"at least one visible : {1 - union:.0%} of the time")

    unresolved = {k: v for k, v in env.fired.items() if k.startswith("UNRESOLVED")}
    if unresolved:
        print(f"\n!! unresolved event names: {unresolved}")
    return env.fired


def run_seeds(hours: float, seeds: int):
    """Repeat the sim over several seeds and report mean +/- spread.

    One run can't distinguish "the rate is off" from "this seed was
    unlucky" -- at these counts one sigma is ~10%, so a single number
    reads as more precise than it is.
    """
    per_seed = [run(hours, seed=s) for s in range(seeds)]
    minutes = hours * 60.0
    print(f"\n===== {seeds} seeds x {hours:g} h =====")
    print(f"{'event':<24} {'mean/min':>9} {'sd':>7} {'mean gap':>10}")
    print("-" * 52)
    for name in CACTI + ("desert_toads",):
        rates = np.array([f[name] for f in per_seed]) / minutes
        print(f"{name:<24} {rates.mean():>9.3f} {rates.std():>7.3f} "
              f"{1/rates.mean():>9.1f}m")
    tot = np.array([sum(f[c] for c in CACTI) for f in per_seed]) / minutes
    print("-" * 52)
    print(f"{'CACTUS (any)':<24} {tot.mean():>9.3f} {tot.std():>7.3f} "
          f"{1/tot.mean():>9.1f}m")


if __name__ == "__main__":
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 24.0
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    run_seeds(hours, seeds) if seeds > 1 else run(hours)
