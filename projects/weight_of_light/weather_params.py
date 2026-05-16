"""Weight of Light project — weather machinery.

Owns the project's full weather data locally:

  * ``WeatherState`` enum (CLEAR + WOL_CLEAR / RAIN / STARS / RAINBOW)
  * ``WEATHER_PRESETS`` (per-state param overrides)
  * ``WEATHER_SETS`` (the five operator-switchable themes)
  * ``DEFAULT_WEATHER_SET`` (the boot default)

Lib (``lib/weather_params.py``) only provides project-agnostic schema:
``DEFAULT_WEATHER_PARAMS``, ``PARAMETER_DEFINITIONS``,
``GLOBAL_PARAMETERS``, and ``AVAILABLE_BACKGROUND_EVENTS``. Putting
project content in lib would silently leak between projects (and was
the cause of an earlier Fan-shows-WoL-options bug — see
``project_weather_data_lives_local.md`` in memory).

Each WoL set's ``background_events`` references entries in
``projects.weight_of_light.event_map`` — those carry
``{"group": ...}`` metadata so they dispatch onto WoL's group canvases
(Sky, Ground).
"""
from enum import Enum

# Schema bits stay shared — adding a new param definition lib-side
# is visible to every project automatically.
from lib.weather_params import (  # noqa: F401  explicit re-export
    DEFAULT_WEATHER_PARAMS,
    PARAMETER_DEFINITIONS,
    GLOBAL_PARAMETERS,
    AVAILABLE_BACKGROUND_EVENTS,
)


class WeatherState(Enum):
    """States WoL actually uses. CLEAR is kept so the legacy
    ``wol_pattern`` / ``wol_gentle`` / ``wol_geometric`` / ``wol_test``
    sets — which use CLEAR as their only state — still resolve."""
    CLEAR = "clear"
    WOL_CLEAR = "wol_clear"
    WOL_RAIN = "wol_rain"
    WOL_STARS = "wol_stars"
    WOL_RAINBOW = "wol_rainbow"


# Per-state param overrides. Anything a state doesn't list falls
# through to ``DEFAULT_WEATHER_PARAMS`` from lib via
# ``WeatherStateController.get_weather_params``.
WEATHER_PRESETS = {
    # Bare CLEAR — used as the active state by the legacy
    # self-running sets (wol_pattern / wol_gentle / wol_geometric /
    # wol_test), which gate their visuals on event presence rather
    # than per-state params. No transitions listed because those
    # sets stay on CLEAR by design.
    WeatherState.CLEAR: {
        "Switch_rate": 0.0,            # never auto-transition
        "transition_duration": 1.0,
        "possible_transitions": ["clear"],
        "transition_weights": [1.0],
    },

    WeatherState.WOL_CLEAR: {
        "Switch_rate": 1.0,
        "transition_duration": 12,
        "possible_transitions": ["wol_rain", "wol_stars", "wol_rainbow"],
        "transition_weights": [1.0, 1.0, 0.4],
        "celestial_visibility": 0.0,
        "fog": 0.05,
        "rain_rate": 0.0,
        "rainbow_intensity": 0.0,
        "starryness": 0.0,
    },

    WeatherState.WOL_RAIN: {
        "Switch_rate": 1.0,
        "transition_duration": 12,
        "possible_transitions": ["wol_clear", "wol_stars", "wol_rainbow"],
        "transition_weights": [2.0, 0.4, 0.3],
        "celestial_visibility": 0.4,
        "fog": 0.2,
        "lightning_probability": 0.1,
        "rain_rate": 1.0,
        "rainbow_intensity": 0.0,
        "starryness": 0.0,
    },

    WeatherState.WOL_STARS: {
        "Switch_rate": 1.0,
        "transition_duration": 12,
        "possible_transitions": ["wol_clear", "wol_rain"],
        "transition_weights": [2.0, 0.4],
        "celestial_visibility": 1.0,
        "fog": 0.0,
        "rain_rate": 0.0,
        "rainbow_intensity": 0.0,
        "season_preference": 0.0,
        "starryness": 1.0,
    },

    WeatherState.WOL_RAINBOW: {
        "Switch_rate": 1.0,
        "transition_duration": 18,
        "possible_transitions": ["wol_clear"],
        "transition_weights": [1.0],
        "celestial_visibility": 0.2,
        "fog": 0.0,
        "rain_rate": 0.0,
        "rainbow_intensity": 1.0,
        "starryness": 0.0,
    },
}


# Shared scaffolding for the three legacy self-running themes
# (wol_pattern / wol_gentle / wol_geometric / wol_test). All four
# sit on CLEAR and don't gate visuals on weather params.
_BASE = {
    "states": [WeatherState.CLEAR.value],
    "season_speed": 0.0,
    "transition_speed": 1.0,
    "season_extremity": 0.0,
    "allowed_parameters": [],
    "random_events": [],
    "random_event_rate": 0.0,
}


WEATHER_SETS = {
    # ----- Native weather set (proof of concept) -----
    # Continuous day/night sky + ground twinkle, with rain / stars /
    # rainbow weather overlays driven by per-state intensity params.
    # Permanent background events: day/night sky shader, stars, rain
    # on sky and ground, ground twinkle, rainbow, presence glow.
    # State transitions cross-fade smoothly because
    # WeatherStateController interpolates per-frame.
    "wol_natural": {
        "name": "Weight of Light — Natural",
        "description": "Permanent day/night sky + ground twinkle, with rain / "
                       "stars / rainbow weather overlays.",
        "states": [
            WeatherState.WOL_CLEAR.value,
            WeatherState.WOL_RAIN.value,
            WeatherState.WOL_STARS.value,
            WeatherState.WOL_RAINBOW.value,
        ],
        # season_speed > 0 keeps ``ambient_light`` (used by the
        # ground twinkle's brightness multiplier) actually moving.
        # season_extremity=0 keeps state transitions strictly
        # weighted by ``transition_weights``, not by season position.
        "season_speed": 1.0,
        "transition_speed": 1.0,
        "season_extremity": 0.0,
        "allowed_parameters": [
            "Switch_rate", "transition_duration",
            "rain_rate", "starryness",
            "rainbow_intensity",
            "celestial_visibility", "fog",
            "lightning_probability",
        ],
        "random_events": [],
        "random_event_rate": 0.0,
        # All run continuously; per-state intensity params decide
        # what's visible. The day/night sky shader runs forever;
        # the weather overlays gate on rain_rate / starryness /
        # rainbow_intensity and contribute zero alpha when their
        # gate is 0.
        "background_events": [
            "wol_sky_daynight",
            "wol_stars",
            "wol_rain",
            "wol_ground_twinkle",
            "wol_rain_ground",
            "wol_rainbow",
            "wol_presence_glow",
        ],
    },

    "wol_pattern": {
        **_BASE,
        "name": "Weight of Light — Pattern",
        "description": "3D abstract patterns; each canvas shows a different "
                       "cross-section of the underlying field.",
        "background_events": [
            "wol_voronoi_trunk",
            "wol_wave_leaves",
            "wol_tunnel_ambient",
        ],
    },
    "wol_gentle": {
        **_BASE,
        "name": "Weight of Light — Gentle",
        "description": "Softer, slower-evolving patterns; quieter overall mood.",
        "background_events": [
            "wol_fog_trunk",
            "wol_spots_leaves",
            "wol_voronoi_ambient",
        ],
    },
    "wol_geometric": {
        **_BASE,
        "name": "Weight of Light — Geometric",
        "description": "Crystalline, angular patterns with sharper boundaries.",
        "background_events": [
            "wol_isovalues_trunk",
            "wol_tentacle_leaves",
            "wol_isovalues_ambient",
        ],
    },
    # Test set — bouncing blue ball on the leaves canvas only.
    # Trunk and ambient have no events scheduled, so they stay
    # black. Useful as a smoke test for the per-pixel physical-
    # position metadata path (the ball's collision is in normalized
    # [-0.5, +0.5] space, not FBO pixels) and as a template for
    # future test shaders.
    "wol_test": {
        **_BASE,
        "name": "Weight of Light — Test",
        "description": "Test set: bouncing blue ball in physical space, "
                       "leaves group only. Other groups dark.",
        "background_events": [
            "wol_test_bouncing_ball",
        ],
    },
}


DEFAULT_WEATHER_SET = "wol_natural"
