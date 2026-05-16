"""Project-agnostic schema for the weather machinery.

Each project's weather data (``WeatherState`` enum, ``WEATHER_PRESETS``,
``WEATHER_SETS``, ``DEFAULT_WEATHER_SET``) lives in that project's
own ``projects/<id>/weather_params.py``. Lib owns only what's shared
across every project: parameter type definitions, the param defaults,
the ``GLOBAL_PARAMETERS`` list, and a fallback list of background
event names for the web editor.

A minimal ``WeatherState`` (just ``CLEAR``) is kept here as a safety
fallback for code paths constructed without an active project (tests,
early boot). At runtime ``EnvironmentalSystem._refresh_weather_module``
pulls the active project's enum and never falls back to this one — so
adding new states here doesn't reach the engine.

See ``projects/<id>/weather_params.py`` for the runtime data.
"""
import numpy as np
from enum import Enum


class WeatherState(Enum):
    """Minimal default enum. Each project supplies its own
    ``WeatherState`` containing the states it actually uses; the
    engine prefers the project's enum at runtime."""
    CLEAR = "clear"


# Global parameters that are always available in every weather set —
# every project's editor exposes these as un-removable knobs.
GLOBAL_PARAMETERS = [
    "possible_transitions",
    "transition_weights",
    "transition_duration",
    "Sound_volume",
    "ARI",
    "fog",
    "fog_color",
    "season_preference",
    "Switch_rate",
    "on_transition_events",
]

# Fallback list of background-event names shown in the web editor's
# "add background event" dropdown when the active project hasn't
# pushed its own list via ``WebController.set_available_events``.
# In normal operation ``EnvironmentalSystem.__init__`` pushes the
# active project's event_map keys before the UI is reachable, so
# this list is effectively a safety net.
AVAILABLE_BACKGROUND_EVENTS = [
    'clouds',
    'firefly',
    'stars',
    'rain',
    'fog',
    'sandstorm',
    'fog_beings',
    'falling_leaves',
]

# Parameter definitions for the weather editor — type metadata that's
# shared across every project. Adding a new param here makes it
# editable in any project's web UI; the project's preset dicts pick
# whether to actually use it.
PARAMETER_DEFINITIONS = {
    'ARI': {'type': 'number', 'step': 1},
    'Aurora_probability': {'type': 'number', 'step': 0.1},
    'Owly': {'type': 'number', 'step': 0.1},
    'Sound_volume': {'type': 'number', 'step': 0.1},
    'Switch_rate': {'type': 'number', 'step': 0.1},
    'Weird': {'type': 'number', 'step': 0.1},
    'Wolfy': {'type': 'number', 'step': 0.1},
    'ambient_sound': {'type': 'text'},
    'bioluminescence': {'type': 'number', 'step': 0.05},
    'bubble_density': {'type': 'number', 'step': 0.05},
    'canopy_density': {'type': 'number', 'step': 0.05},
    'celestial_visibility': {'type': 'number', 'step': 0.1},
    'dapple_strength': {'type': 'number', 'step': 0.05},
    'data_flow_rate': {'type': 'number', 'step': 0.05},
    'drone_activity': {'type': 'number', 'step': 0.05},
    'electric_interference': {'type': 'number', 'step': 0.05},
    'eye_density': {'type': 'number', 'step': 0.05},
    'firefly_density': {'type': 'number', 'step': 0.1},
    'fog': {'type': 'number', 'step': 0.05},
    'fog_color': {'type': 'array', 'length': 3},
    'frost_level': {'type': 'number', 'step': 0.05},
    'glitch_probability': {'type': 'number', 'step': 0.05},
    'godray_strength': {'type': 'number', 'step': 0.05},
    'hologram_density': {'type': 'number', 'step': 0.05},
    'kelp_density': {'type': 'number', 'step': 0.05},
    'light_pollution': {'type': 'number', 'step': 0.05},
    'lightning_probability': {'type': 'number', 'step': 0.05},
    'marine_life_activity': {'type': 'number', 'step': 0.05},
    'meteor_rate': {'type': 'number', 'step': 0.05},
    'neon_intensity': {'type': 'number', 'step': 0.05},
    'on_transition_events': {'type': 'event-list'},
    'pollution_level': {'type': 'number', 'step': 0.05},
    'possible_transitions': {'type': 'array-string'},
    'pride_intensity': {'type': 'number', 'step': 0.05},
    'rain_rate': {'type': 'number', 'step': 0.1},
    'rainbow_intensity': {'type': 'number', 'step': 0.05},
    'sand_density': {'type': 'number', 'step': 0.1},
    'scan_line_intensity': {'type': 'number', 'step': 0.05},
    'season_preference': {'type': 'number', 'step': 0.025},
    'skiptime': {'type': 'number', 'step': 0.5},
    'snow_rate': {'type': 'number', 'step': 0.05},
    'spookyness': {'type': 'number', 'step': 0.1},
    'spore_color': {'type': 'number', 'step': 0.05},
    'spore_density': {'type': 'number', 'step': 0.05},
    'starryness': {'type': 'number', 'step': 0.1},
    'stream_flow_rate': {'type': 'number', 'step': 0.05},
    'tide_level': {'type': 'number', 'step': 0.05},
    'train_density': {'type': 'number', 'step': 0.1},
    'train_speed': {'type': 'number', 'step': 0.5},
    'transition_duration': {'type': 'number', 'step': 1},
    'transition_weights': {'type': 'array-number'},
    'tree_prob': {'type': 'number', 'step': 0.1},
    'vent_activity': {'type': 'number', 'step': 0.05},
    'volcano_level': {'type': 'number', 'step': 0.1},
    'wave_amplitude': {'type': 'number', 'step': 0.05},
    'wave_speed': {'type': 'number', 'step': 0.05},
    'wind_speed': {'type': 'number', 'step': 0.1},
}

# Default values for any param a project's preset doesn't override.
# Project-agnostic — every project's WeatherStateController uses
# these as the floor for ``get_weather_params``.
DEFAULT_WEATHER_PARAMS = {
    "wind_speed": 0,
    "rain_rate": 0,
    "lightning_probability": 0,
    "starryness": 1.0,
    "spookyness": 0.0,
    "fog": 0.0,
    "fog_color": np.array([0.7, 0.7, 0.7]),
    "possible_transitions": ['light_rain', 'foggy', 'windy_night'],
    "transition_weights": [1.0, 2.0, 0.5],
    "transition_duration": 20.0,
    "celestial_visibility": 1.0,
    "firefly_density": 0.0,
    "Aurora_probability": 0.0,
    "Wolfy": 0.0,
    "Switch_rate": 1.0,
    "meteor_rate": 0.0,
    "volcano_level": 0.0,
    "sand_density": 0.0,
    "skiptime": 0.0,
    "tree_prob": 0.0,
    "Weird": 0.0,
    "Sound_volume": 1.0,
    "season_preference": 0.375,
    "ambient_sound": None,
    "ARI": 0.0,
}

# Empty defaults — projects own their content. Used only as fallbacks
# when a project module doesn't provide its own (rare; ``_refresh_
# weather_module`` warns at boot if this happens). ``WEATHER_PRESETS``
# has an empty CLEAR entry so the controller's lookup
# ``weather_presets[weather_state]`` doesn't KeyError on the default
# fallback state.
WEATHER_PRESETS = {WeatherState.CLEAR: {}}
WEATHER_SETS = {}
DEFAULT_WEATHER_SET = None


def _validate_parameter_definitions():
    """Sanity-check that every parameter referenced by a weather set
    or preset has a ``PARAMETER_DEFINITIONS`` entry.

    Missing entries cause the web weather editor to silently skip
    the parameter (see the ``if (!paramDef) continue;`` in
    ``weather_editor.html``), so the parameter still affects
    rendering but the user can't see or change its value. Surfacing
    the problem at import time turns a "why can't I edit this"
    mystery into an obvious warning.

    With lib now holding no project data, this validates only
    against the empty default — projects can call this on their
    own data if they want the same check. ``EnvironmentalSystem``
    doesn't auto-invoke it on project modules; it runs only at
    lib import time.
    """
    import sys

    known = set(PARAMETER_DEFINITIONS.keys())

    missing_in_sets = {}
    for set_name, set_data in WEATHER_SETS.items():
        for param in set_data.get("allowed_parameters", []):
            if param not in known:
                missing_in_sets.setdefault(param, []).append(set_name)

    missing_in_presets = {}
    for state, preset in WEATHER_PRESETS.items():
        for param in preset.keys():
            if param not in known:
                missing_in_presets.setdefault(param, []).append(state.value)

    if not missing_in_sets and not missing_in_presets:
        return

    bar = "=" * 72
    lines = [
        "",
        bar,
        "[weather_params] parameters missing from PARAMETER_DEFINITIONS",
        "These will be silently skipped by the web weather editor.",
        "Add an entry in PARAMETER_DEFINITIONS for each one.",
        bar,
    ]
    for param in sorted(missing_in_sets):
        sets = ", ".join(sorted(missing_in_sets[param]))
        lines.append(f"  [set]    {param}  (in allowed_parameters of: {sets})")
    for param in sorted(missing_in_presets):
        states = ", ".join(sorted(missing_in_presets[param]))
        lines.append(f"  [preset] {param}  (in states: {states})")
    lines.append(bar)
    print("\n".join(lines), file=sys.stderr)


_validate_parameter_definitions()
