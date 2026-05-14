import numpy as np
from enum import Enum

class WeatherState(Enum):
    CLEAR = "clear"
    WOL_CLEAR = "wol_clear"
    WOL_RAIN = "wol_rain"
    WOL_STARS = "wol_stars"
    WOL_RAINBOW = "wol_rainbow"

# Global parameters that are always available in every weather set
# These cannot be removed from sets but their values can be customized per weather state
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

# Available background events (always-active effects)
# This is the single source of truth for which events can be used as continuous background effects.
# Add new background-capable events here. They must also exist in Stories_OGL.py's event_map.
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

# Parameter definitions for the weather editor
# Defines the type and input configuration for each parameter
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

# Default weather parameters
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

# Weather presets
# Weather state parameters
WEATHER_PRESETS = {
    WeatherState.CLEAR: {
        "ARI": 40,
        "Aurora_probability": 0.5,
        "Switch_rate": 0.9,
        "Weird": 1,
        "ambient_sound": None,
        "canopy_density": 0.8,
        "meteor_rate": 0.25,
        "possible_transitions": ["light_rain", "foggy", "windy_night", "firefly", "mushroom", "leaves", "bloom", "forest_morning", "desert_blazing_noon", "desert_dusk_embers", "desert_starlit_night", "desert_dust_devil", "desert_meteor_shower"],
        "season_preference": 0.375,
        "transition_weights": [1, 1, 0.75, 0.5, 0.2, 0.75, 0.75, 0.8, 1, 0.6, 0.5, 0.4, 0.2],
        "tree_prob": 1,
        "wind_speed": 0.2,
    },

    WeatherState.WOL_CLEAR: {
        "Switch_rate": 1,
        "celestial_visibility": 0,
        "fog": 0.05,
        "possible_transitions": ["wol_rain", "wol_stars", "wol_rainbow"],
        "rain_rate": 0,
        "rainbow_intensity": 0,
        "starryness": 0,
        "transition_duration": 12,
        "transition_weights": [1, 1, 0.4],
    },

    WeatherState.WOL_RAIN: {
        "Switch_rate": 1,
        "celestial_visibility": 0.4,
        "fog": 0.2,
        "lightning_probability": 0.1,
        "possible_transitions": ["wol_clear", "wol_stars", "wol_rainbow"],
        "rain_rate": 1,
        "rainbow_intensity": 0,
        "starryness": 0,
        "transition_duration": 12,
        "transition_weights": [2, 0.4, 0.3],
    },

    WeatherState.WOL_RAINBOW: {
        "Switch_rate": 1,
        "celestial_visibility": 0.2,
        "fog": 0,
        "possible_transitions": ["wol_clear"],
        "rain_rate": 0,
        "rainbow_intensity": 1,
        "starryness": 0,
        "transition_duration": 18,
        "transition_weights": [1],
    },

    WeatherState.WOL_STARS: {
        "Switch_rate": 1,
        "celestial_visibility": 1,
        "fog": 0,
        "possible_transitions": ["wol_clear", "wol_rain"],
        "rain_rate": 0,
        "rainbow_intensity": 0,
        "season_preference": 0,
        "starryness": 1,
        "transition_duration": 12,
        "transition_weights": [2, 0.4],
    },

}

# Weather Sets - Mutually exclusive collections of weather states
WEATHER_SETS = {
    "wol_gentle": {
        "allowed_parameters": [],
        "background_events": ["wol_fog_trunk", "wol_spots_leaves", "wol_voronoi_ambient"],
        "description": "Softer, slower-evolving patterns; quieter overall mood.",
        "name": "Weight of Light — Gentle",
        "random_event_rate": 0,
        "random_events": [],
        "season_extremity": 0,
        "season_speed": 0,
        "states": ["clear"],
        "transition_speed": 1,
    },

    "wol_geometric": {
        "allowed_parameters": [],
        "background_events": ["wol_isovalues_trunk", "wol_tentacle_leaves", "wol_isovalues_ambient"],
        "description": "Crystalline, angular patterns with sharper boundaries.",
        "name": "Weight of Light — Geometric",
        "random_event_rate": 0,
        "random_events": [],
        "season_extremity": 0,
        "season_speed": 0,
        "states": ["clear"],
        "transition_speed": 1,
    },

    "wol_natural": {
        "allowed_parameters": ["Switch_rate", "transition_duration", "rain_rate", "starryness", "rainbow_intensity", "celestial_visibility", "fog"],
        "background_events": ["wol_sky_daynight", "wol_stars", "wol_rain", "wol_ground_twinkle", "wol_rainbow"],
        "description": "Permanent day/night sky + ground twinkle, with rain / stars / rainbow weather overlays.",
        "name": "Weight of Light — Natural",
        "random_event_rate": 0,
        "random_events": [],
        "season_extremity": 0,
        "season_speed": 1,
        "states": ["wol_clear", "wol_rain", "wol_stars", "wol_rainbow"],
        "transition_speed": 1,
    },

    "wol_pattern": {
        "allowed_parameters": [],
        "background_events": ["wol_voronoi_trunk", "wol_wave_leaves", "wol_tunnel_ambient"],
        "description": "3D abstract patterns; each canvas shows a different cross-section of the underlying field.",
        "name": "Weight of Light — Pattern",
        "random_event_rate": 0,
        "random_events": [],
        "season_extremity": 0,
        "season_speed": 0,
        "states": ["clear"],
        "transition_speed": 1,
    },

    "wol_test": {
        "allowed_parameters": [],
        "background_events": ["wol_test_bouncing_ball"],
        "description": "Test set: bouncing blue ball in physical space, leaves group only. Other groups dark.",
        "name": "Weight of Light — Test",
        "random_event_rate": 0,
        "random_events": [],
        "season_extremity": 0,
        "season_speed": 0,
        "states": ["clear"],
        "transition_speed": 1,
    },

}

DEFAULT_WEATHER_SET = "wol_gentle"



def _validate_parameter_definitions():
    """Sanity-check that every parameter referenced by a weather set or
    preset has a PARAMETER_DEFINITIONS entry.

    Missing entries cause the web weather editor to silently skip the
    parameter (see the `if (!paramDef) continue;` in weather_editor.html),
    so even though the parameter still affects rendering, the user can't
    see or change its value. Surfacing the problem at import time turns a
    "why can't I edit this" mystery into an obvious warning.
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
