"""Fan-specific random event scheduling.

Triggered by ``EnvironmentalSystem.random_events()`` once per frame via
the project's ``hooks.random_events`` declaration in ``project.yaml``.
Reads ``weather_state.weather_params`` for trigger thresholds and
schedules Fan-flavored effects (tree, aurora, lightning, sandstorm,
eye, meteor) plus paired one-shot sounds (thunder, wolves, owls).

Sound paths resolve relative to ``state['media_root']`` which the active
project seeds at startup. WoL doesn't declare this hook in its
``project.yaml`` so this module is never imported when WoL is active —
no Fan effects leak into WoL canvases.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from renderer import effects as fx


def run(env_system) -> None:
    """Roll the dice on Fan's hardcoded random events.

    Called once per call of ``EnvironmentalSystem.random_events()``,
    which itself fires once per ``random_state_change`` opportunity.
    Two independent dice rolls partition the events into "atmospheric"
    (tree / aurora / lightning) and "creature" (sandstorm / eye /
    wolf / owl / meteor) groups so they can fire concurrently.
    """
    weather_params = env_system.weather_state.weather_params
    scheduler = env_system.scheduler
    state = scheduler.state
    media_root = Path(state.get("media_root", "media"))
    sounds = media_root / "sounds"

    # ---- atmospheric group ----
    randcheck = np.random.random()

    # Tree-spawn event removed: didn't mesh visually with the forest
    # theme (single-instance projected tree clashed with the dense
    # procedural canopy). The ``tree_prob`` weather param is left in
    # presets as inert metadata; shader_tree itself is retained for
    # potential reuse but no longer scheduled by random_events.

    if randcheck < weather_params["Aurora_probability"] / 1000:
        scheduler.schedule_event(0, 50, fx.shader_aurora, frame_id=0)

    if randcheck < weather_params["lightning_probability"] / 500:
        scheduler.schedule_event(0, 1, fx.shader_lightning, frame_id=0)
        # Pair the visual lightning with a randomized thunder rumble.
        thunder_sounds = [
            "thunder-307513.mp3",
            "loud-thunder-192165.mp3",
            "peals-of-thunder-191992.mp3",
        ]
        thunder_file = sounds / np.random.choice(thunder_sounds)
        engine = state.get("soundengine")
        if engine:
            engine.schedule_event(thunder_file, volume=0.7)

    # ---- creature / weather-effect group (independent dice roll) ----
    randcheck = np.random.random()

    if randcheck < weather_params["sand_density"] / 2000:
        scheduler.schedule_event(0, 45, fx.shader_sandstorm, frame_id=0)

    if randcheck < weather_params["spookyness"] / 1000:
        scheduler.schedule_event(0, 30, fx.shader_eye, frame_id=0)

    if randcheck < weather_params["Wolfy"] / 2500:
        wolf_sounds = [
            "howling-wolves-6965.mp3",
            "wolf-howling-140235.mp3",
            "duskwolf-101348.mp3",
        ]
        wolf_file = sounds / np.random.choice(wolf_sounds)
        engine = state.get("soundengine")
        if engine:
            engine.schedule_event(wolf_file, volume=1.0)

    # Owl hoot one-shot sounds (low probability, half volume).
    # Divisor is 5x larger than wolves because the owl clip is ~23s
    # long — without this, hoots would overlap continuously.
    if randcheck < weather_params.get("Owly", 0.0) / 12500:
        owl_sounds = ["Owls Hooting.wav"]
        owl_file = sounds / np.random.choice(owl_sounds)
        engine = state.get("soundengine")
        if engine:
            engine.schedule_event(owl_file, volume=0.5)

    if randcheck < weather_params["meteor_rate"] / 800:
        scheduler.schedule_event(0, 25, fx.shader_meteor, frame_id=0)
