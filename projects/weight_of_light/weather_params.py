"""Weight of Light project — weather machinery.

Inherits Fan's ``WeatherState`` enum, ``WEATHER_PRESETS`` (per-state param
preset dicts), ``DEFAULT_WEATHER_PARAMS`` and ``PARAMETER_DEFINITIONS``
from the shared library — so WoL doesn't have to redefine the entire
weather param schema. The two things WoL DOES override are
``WEATHER_SETS`` (the curated list of sets the operator can switch
between) and ``DEFAULT_WEATHER_SET`` (which set the project boots in).

Each WoL set's ``background_events`` references events that exist in
``projects.weight_of_light.event_map`` — those carry ``{"group": ...}``
metadata so they dispatch onto WoL's three group canvases (trunk, leaves,
ambient). No Fan events leak in.

Future per-state work for WoL (custom enum values, audio-reactive
presets, sensor-driven params) goes here too; for now the placeholder
set lights all three groups with weather-state-independent effects so
the operator sees obviously-different patterns per group.
"""
from copy import deepcopy

from lib.weather_params import *  # noqa: F401,F403
from lib.weather_params import (  # noqa: F401  explicit re-exports
    WeatherState,
    PARAMETER_DEFINITIONS,
    DEFAULT_WEATHER_PARAMS,
    WEATHER_PRESETS as _FAN_WEATHER_PRESETS,
)


# Strip ``ambient_sound`` entries off the inherited Fan presets — those
# point at .wav files that live in ``projects/fan/media/sounds/`` only.
# When WoL is the active project the audio engine resolves ambient
# paths against ``state['media_root']`` = WoL's media folder, where
# none of those files exist, and the lookup throws a popup
# FileNotFoundError. Silence ambient until WoL ships its own audio
# kit (or until the engine grows a multi-root resolver). Per-state
# audio reactivity unaffected; only the looped ambient backdrop.
WEATHER_PRESETS = deepcopy(_FAN_WEATHER_PRESETS)
for _state, _preset in WEATHER_PRESETS.items():
    if isinstance(_preset, dict) and "ambient_sound" in _preset:
        _preset["ambient_sound"] = None


# Three operator-switchable visual themes. Each theme is its own weather
# set — operator picks one from the web UI dropdown. All three share Fan's
# WeatherState.CLEAR as the active state; the picked effects don't gate on
# weather params (no pride_intensity / starryness / etc. dependency), so
# the visuals run cleanly without authoring per-state presets here.
#
# Effect choice per theme follows two rules:
#   1. SELF-RUNNING — no audio dependency (rooms can be silent)
#   2. PER-ROW VARIATION — different rows of each canvas produce visually
#      distinct output, so each physical strip looks different rather than
#      9 identical trunks / 18 identical leaves / 8 identical ambients.
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
    # ----- Native day/night cycle (proof of concept) -----
    # Linear progression DAWN → DAY → DUSK → NIGHT → DAWN, driven by
    # WeatherStateController's per-frame transition probability
    # (Switch_rate × transition_speed × 1/800 base). The 4 states each
    # set a sky_horizon_color / sky_zenith_color pair that the sky
    # gradient shader reads from outstate; transitions interpolate
    # automatically because both params are np.ndarray.
    #
    # Two background events:
    #   * wol_sky_daynight  → Sky group (300×19) — atmospheric gradient
    #   * wol_ground_passive → Ground group (300×9) — slow per-arc hue
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
        # weighted by transition_weights, not by season position.
        "season_speed": 1.0,
        "transition_speed": 1.0,
        "season_extremity": 0.0,
        "allowed_parameters": [
            "Switch_rate", "transition_duration",
            "rain_rate", "starryness",
            "rainbow_intensity",
            "celestial_visibility", "fog",
        ],
        "random_events": [],
        "random_event_rate": 0.0,
        # All five run continuously; per-state intensity params decide
        # what's visible. The day/night sky shader runs forever; the
        # weather overlays gate on rain_rate / starryness /
        # rainbow_intensity and contribute zero alpha when their gate
        # is 0.
        "background_events": [
            "wol_sky_daynight",
            "wol_stars",
            "wol_rain",
            "wol_ground_twinkle",
            "wol_rain_ground",
            "wol_rainbow",
        ],
    },

    "wol_pattern": {
        **_BASE,
        "name": "Weight of Light — Pattern",
        "description": "3D abstract patterns; each canvas shows a different cross-section "
                       "of the underlying field.",
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
    # Test set — bouncing blue ball on the leaves canvas only. Trunk
    # and ambient have no events scheduled, so they stay black. Useful
    # as a smoke test for the per-pixel physical-position metadata
    # path (the ball's collision is in normalized [-0.5, +0.5] space,
    # not FBO pixels) and as a template for future test shaders.
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
