"""Weather state interpolation and seasonal transition selection.

WeatherStateController tracks the current and target WeatherState, interpolates
parameters between them during a transition, and chooses the next state using
season-biased probability weights. Has no audio, scheduler, or web dependencies.

Phase-8 multi-project work: the WeatherState enum, default params dict, and
weather presets dict are project-specific. They can be passed in at
construction; ``lib.weather_params`` defaults are kept for backward compat.
"""

import numpy as np
import time

from lib.weather_params import (
    WeatherState as _DEFAULT_WEATHER_STATE_ENUM,
    DEFAULT_WEATHER_PARAMS as _DEFAULT_WEATHER_PARAMS,
    WEATHER_PRESETS as _DEFAULT_WEATHER_PRESETS,
)


class WeatherStateController:
    """Pure weather state management with no audio, scheduler, or web dependencies."""

    def __init__(
        self,
        initial_weather=None,
        weather_state_enum=None,
        default_weather_params: dict | None = None,
        weather_presets: dict | None = None,
    ):
        self._weather_state_enum = (
            weather_state_enum if weather_state_enum is not None else _DEFAULT_WEATHER_STATE_ENUM
        )
        if initial_weather is None:
            # Default to the enum's CLEAR if it has one; otherwise the first member.
            initial_weather = getattr(self._weather_state_enum, "CLEAR", None) \
                or next(iter(self._weather_state_enum))
        self.default_weather_params = (
            default_weather_params.copy() if default_weather_params is not None
            else _DEFAULT_WEATHER_PARAMS.copy()
        )
        self.weather_presets = (
            weather_presets if weather_presets is not None else _DEFAULT_WEATHER_PRESETS
        )
        self.current_weather = initial_weather
        self.target_weather = initial_weather
        self.transition_time = 0
        self.transition_start = 0
        # Start at "transition complete" so ``random_state_change``'s
        # ``progress >= 1.0`` gate is open from the first frame —
        # otherwise projects without an explicit ``startup_weather_set``
        # in project.yaml never bootstrap their first state change
        # (initial progress=0 stays stuck because the if-branch in
        # update() only runs when current != target). Fan-style
        # projects that DO use ``startup_weather_set`` are unaffected:
        # their main-thread kickoff calls transition_to_weather, which
        # sets target!=current and update() drives progress 0..1 from
        # whatever it was.
        self.progress = 1.0
        # Initialize weather_params from the actual initial state, not just defaults
        self.weather_params = self.get_weather_params(initial_weather)
        # Snapshot of weather_params at the moment a transition started (used as
        # interpolation start so mid-transition redirects don't snap back to presets)
        self.transition_start_params = self.weather_params.copy()

    def get_weather_params(self, weather_state) -> dict:
        """Get the complete set of parameters for a weather state by combining with defaults."""
        params = self.default_weather_params.copy()
        params.update(self.weather_presets[weather_state])
        return params

    def start_transition(
        self,
        new_weather,
        transition_duration: float,
        current_time: float,
    ) -> dict:
        """Set target state and timing. Returns target_params so the caller can handle
        audio and event scheduling without WeatherStateController having those dependencies."""
        self.target_weather = new_weather
        self.transition_time = transition_duration
        self.transition_start = current_time
        # Snapshot current interpolated state so redirected transitions don't snap
        self.transition_start_params = self.weather_params.copy()
        return self.get_weather_params(new_weather)

    def update(self, current_time: float) -> None:
        """Interpolate weather_params between current and target states; snap when done."""
        if self.current_weather != self.target_weather:
            self.progress = min(
                1.0, (current_time - self.transition_start) / self.transition_time
            )

            target_params = self.get_weather_params(self.target_weather)

            # Iterate over the UNION of source and target keys. A param that
            # exists in the source but not the target (e.g. spore_density when
            # leaving the "mushroom" state for one that doesn't define it, and
            # which also isn't in DEFAULT_WEATHER_PARAMS) used to be held at
            # the source value throughout the transition and then snap to its
            # fallback the moment progress hit 1.0 — visible as effects
            # disappearing in a single frame. Interpolating the union makes
            # orphan params fade smoothly toward their default (or 0).
            all_keys = set(target_params.keys()) | set(self.transition_start_params.keys())
            for param in all_keys:
                target_value = target_params.get(
                    param, self.default_weather_params.get(param, 0)
                )
                start_value = self.transition_start_params.get(
                    param, self.default_weather_params.get(param, 0)
                )

                if (isinstance(target_value, (int, float, np.ndarray))
                        and isinstance(start_value, (int, float, np.ndarray))):
                    self.weather_params[param] = (
                        target_value - start_value
                    ) * self.progress + start_value
                else:
                    self.weather_params[param] = target_value

            if self.progress >= 1.0:
                self.current_weather = self.target_weather
                self.weather_params = target_params.copy()

    def get_state_output(self, season: float, current_time: float) -> dict:
        """Return scheduler.state keys derived purely from weather_params.

        Does NOT include: sound, celestial_bodies, scale, season — those come from
        other subsystems and remain in send_variables().
        """
        fog = np.maximum(
            0,
            self.weather_params["fog"]
            * (0.75 - 0.25 * np.cos(np.pi * 2 * (season - 0.625))),
        )
        cloudyness = (
            (1 - self.weather_params["starryness"])
            + (1 - self.weather_params["celestial_visibility"])
            + fog
            + self.weather_params["rain_rate"]
            + self.weather_params["wind_speed"] / 3
        ) / 4

        return {
            "cloudyness": cloudyness,
            "fog_strength": fog,
            "fog_color": self.weather_params["fog_color"],
            "wind": self.weather_params["wind_speed"]
            * np.cos(np.pi * 2 * (season - 0.125)),
            "rain": self.weather_params["rain_rate"],
            "starryness": self.weather_params["starryness"],
            "celestial_visibility": self.weather_params["celestial_visibility"],
            "firefly_density": self.weather_params["firefly_density"],
            "meteor_rate": self.weather_params["meteor_rate"],
            "volcano_level": (np.sin(current_time / 100) * 0.5 + 0.5)
            * self.weather_params["volcano_level"],
            "sand_density": self.weather_params.get("sand_density", 0),
            "tree_growth": self.weather_params.get("tree_prob", 0) + 0.25,
            "wave_speed": self.weather_params.get("wave_speed", 0.5),
            "wave_amplitude": self.weather_params.get("wave_amplitude", 0.5),
            "tide_level": self.weather_params.get("tide_level", 0.5),
            "bioluminescence": self.weather_params.get("bioluminescence", 0.0),
            "bubble_density": self.weather_params.get("bubble_density", 0.0),
            "marine_life_activity": self.weather_params.get("marine_life_activity", 0.0),
            "kelp_density": self.weather_params.get("kelp_density", 0.0),
            "vent_activity": self.weather_params.get("vent_activity", 0.0),
            "train_speed": self.weather_params.get("train_speed", 8.0),
            "train_density": self.weather_params.get("train_density", 1.0),
            "pride_intensity": self.weather_params.get("pride_intensity", 0.0),
            "godray_strength": self.weather_params.get("godray_strength", 0.0),
            "canopy_density": self.weather_params.get("canopy_density", 0.0),
            "snow_rate": self.weather_params.get("snow_rate", 0.0),
            "spore_density": self.weather_params.get("spore_density", 0.0),
            "spore_color": self.weather_params.get("spore_color", 0.0),
            "dapple_strength": self.weather_params.get("dapple_strength", 0.0),
            "stream_flow_rate": self.weather_params.get("stream_flow_rate", 0.0),
            "eye_density": self.weather_params.get("eye_density", 0.0),
            "frost_level": self.weather_params.get("frost_level", 0.0),
            "spookyness": self.weather_params.get("spookyness", 0.0),
            "season_preference": self.weather_params.get("season_preference", 0.5),
            "rainbow_intensity": self.weather_params.get("rainbow_intensity", 0.0),
            "lightning_probability": self.weather_params.get("lightning_probability", 0.0),
            # Multiplier on forest_birds spawn rate. 1.0 = default, >1 boosts
            # density (e.g. migration state). Effects can read this from
            # outstate without weather_state itself needing to know what
            # forest_birds does with it.
            "bird_density": self.weather_params.get("bird_density", 1.0),
            # Ambient light level in [0.25, 1.0]: peaks at noon (season=0.5),
            # minimum at midnight (season=0 / 1). In the ocean set season is
            # repurposed as time of day, so effects that want to dim at
            # night (fish, kelp) scale their output by this value.
            "ambient_light": 0.25 + 0.75 * (0.5 - 0.5 * np.cos(2 * np.pi * season)),
            # ── Cyberpunk state-tied shader params ────────────────────
            # Without these, every cyber_* shader runs at its wrapper
            # default and is blind to per-state customization. See
            # renderer/effects/cyber_*.py wrappers — each does
            # outstate.get(<param>, <kwarg_default>).
            "neon_intensity": self.weather_params.get("neon_intensity", 0.0),
            "hologram_density": self.weather_params.get("hologram_density", 0.0),
            "pollution_level": self.weather_params.get("pollution_level", 0.0),
            "light_pollution": self.weather_params.get("light_pollution", 0.0),
            "drone_activity": self.weather_params.get("drone_activity", 0.0),
            "scan_line_intensity": self.weather_params.get("scan_line_intensity", 0.0),
            "data_flow_rate": self.weather_params.get("data_flow_rate", 0.0),
            "electric_interference": self.weather_params.get("electric_interference", 0.0),
            "glitch_probability": self.weather_params.get("glitch_probability", 0.0),
            "cyber_skyline_density": self.weather_params.get("cyber_skyline_density", 0.0),
            "cyber_signage_density": self.weather_params.get("cyber_signage_density", 0.0),
            "cyber_underway_intensity": self.weather_params.get("cyber_underway_intensity", 0.0),
            "cyber_transit_intensity": self.weather_params.get("cyber_transit_intensity", 0.0),
            "cyber_view_elevation": self.weather_params.get("cyber_view_elevation", 0.0),
            # Color arrays: per-state palette overrides for cyber shaders.
            # `cyber_rain_color` is consumed by cyber_rain to tint drops
            # acid green / electric blue / sodium amber per state.
            "cyber_rain_color": self.weather_params.get("cyber_rain_color"),
            # `velocity_direction` is a unit vector consumed by
            # velocity_streaks so each arc can pick its motion axis
            # (e.g. (0, -1) for upward ascent in Subroutine 9).
            "velocity_direction": self.weather_params.get("velocity_direction"),
        }

    def select_next_weather(
        self,
        current_weather,
        set_states: list,
        season: float,
        season_extremity: float,
    ):
        """Choose the next weather state using transition weights adjusted for season.

        Args:
            current_weather: The state we are transitioning away from.
            set_states: WeatherState values valid in the current set.
            season: Current season value in [0, 1).
            season_extremity: Scalar that controls how strongly season biases selection.

        Returns:
            A WeatherState chosen probabilistically from the candidates.
        """
        current_preset = self.weather_presets[current_weather]
        enum_cls = self._weather_state_enum
        # Filter out preset entries that don't correspond to states
        # this project's enum knows about. Projects can prune the
        # global lib WeatherState (e.g. WoL keeps only CLEAR + WOL_*),
        # leaving legacy ``possible_transitions`` strings that no
        # longer resolve. Skip them silently rather than crashing
        # with ValueError on the first bad entry.
        possible_states = []
        for s in current_preset["possible_transitions"]:
            try:
                possible_states.append(enum_cls(s))
            except (ValueError, KeyError):
                continue

        # Restrict to states that exist in the active set
        possible_states = [s for s in possible_states if s in set_states]

        if not possible_states:
            possible_states = set_states
            base_weights = [1.0] * len(possible_states)
        else:
            base_weights = []
            for state in possible_states:
                try:
                    idx = current_preset["possible_transitions"].index(state.value)
                    base_weights.append(current_preset["transition_weights"][idx])
                except (ValueError, IndexError):
                    base_weights.append(1.0)

        adjusted_weights = []
        for i, state in enumerate(possible_states):
            season_pref = self.weather_presets[state].get("season_preference", 0.375)
            multiplier = self.calculate_seasonal_weight_multiplier(season_pref, season)
            if season_extremity > 0:
                multiplier = 1.0 + (multiplier - 1.0) * season_extremity
                multiplier = max(0.01, multiplier)
            else:
                multiplier = 1.0
            adjusted_weights.append(base_weights[i] * multiplier)

        adjusted_weights = np.array(adjusted_weights)
        if np.sum(adjusted_weights) > 0:
            adjusted_weights /= np.sum(adjusted_weights)
        else:
            adjusted_weights = np.ones(len(adjusted_weights)) / len(adjusted_weights)

        return np.random.choice(possible_states, p=adjusted_weights)

    @staticmethod
    def calculate_seasonal_weight_multiplier(
        season_preference: float, current_season: float
    ) -> float:
        """Calculate a weight multiplier based on how close the current season is to the
        preferred season. Returns a value between 0.5 (furthest from preferred) and 3.0
        (at preferred season)."""
        distance = abs(current_season - season_preference)
        if distance > 0.5:
            distance = 1.0 - distance  # Take the shorter path around the cycle

        normalized_distance = distance * 2  # 0 = perfect match, 1 = opposite season
        multiplier = 1.0 - (normalized_distance * 0.95)
        return multiplier
