"""Weight of Light — per-frame random event rolls.

Wired by ``hooks: { random_events: projects.weight_of_light.random_events }``
in ``project.yaml``; ``run(env_system)`` is called once per frame from
``EnvironmentalSystem.random_events()`` (gated by
``enable_random_events: true``). Pure dice-rolling — no per-frame state
of its own.

Currently does just one thing:

  * **Lightning** rolls each frame with probability tuned so that at
    ``weather_params['lightning_probability'] == 1.0`` the event
    fires *roughly every 10 seconds* on a uniformly-chosen object.
    At lower values the rate scales linearly: at p=0.1 (the
    ``WOL_RAIN`` default) one strike every ~100 s on average; at
    p=0.0 nothing.

  Each fire delegates to ``button_router.fire_lightning_chain`` so
  random lightning gets the same multi-strike behaviour, sound
  layering, and Sky+Ground dispatch as a button-3 press.

Rate math: a per-frame Bernoulli with probability
``p / (target_period * fps)`` gives mean inter-event time of
``target_period / p`` seconds. ``target_period`` is hardcoded to 10 s
to match the user-facing intuition ("every ~10 s when fully on").
``fps`` is read from ``env_system.frame_time`` so the rate is robust
to the project's chosen target FPS.
"""
from __future__ import annotations

import random


# Mean seconds between lightning strikes when lightning_probability=1.0.
# Linear in lightning_probability — at p=0.1 the mean inter-event is
# 10 / 0.1 = 100 s; at p=0.5 it's 20 s; at p=1.0 it's 10 s.
_LIGHTNING_TARGET_PERIOD_S = 10.0


def run(env_system) -> None:
    """Roll once per frame for any project-specific random events,
    plus refresh per-frame derived state (radar smoothing, baseline,
    presence weight, etc.). Effects can read the radar derivation
    output via ``env_system.scheduler.state['radar'][object_id]`` —
    see ``projects.weight_of_light.radar`` for the field reference.
    """
    # ----- radar derivation -----
    # Cheap (a few dozen floats per object); skips silently if the
    # module isn't importable (e.g. radar.py removed).  Done first so
    # downstream consumers in this same hook see fresh radar values.
    try:
        from projects.weight_of_light import radar
        radar.tick(env_system)
    except Exception as e:
        print(f"[wol_radar] tick failed: {e}")

    weather_params = env_system.weather_state.weather_params

    # ----- lightning -----
    p = float(weather_params.get("lightning_probability", 0.0))
    if p > 0.0:
        # Per-frame Bernoulli probability scaled by frame time. Using
        # env_system.frame_time (set by _compute_frame_time, refreshed
        # on project swap) keeps the wall-clock rate stable across
        # different target_fps configs.
        frame_time = max(float(env_system.frame_time), 1e-3)
        per_frame_p = p * frame_time / _LIGHTNING_TARGET_PERIOD_S
        if random.random() < per_frame_p:
            _fire_random_lightning(env_system)


def _fire_random_lightning(env_system) -> None:
    """Pick a random object and route through button_router's
    multi-strike chain. Falls through silently if the project has no
    object_names map (very early boot, or a project that doesn't
    populate it)."""
    state = env_system.scheduler.state
    names = state.get("object_names") or {}
    if not names:
        return
    object_id = random.choice(list(names.keys()))
    # Lazy-import button_router so this hook stays usable even when
    # the OSC listener isn't running.
    from projects.weight_of_light.button_router import fire_lightning_chain
    fire_lightning_chain(env_system, object_id, tag="rand_lightning")
