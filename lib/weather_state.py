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
        output_map: dict | None = None,
    ):
        self._weather_state_enum = (
            weather_state_enum if weather_state_enum is not None else _DEFAULT_WEATHER_STATE_ENUM
        )
        # Project-supplied outstate publish table (``OUTSTATE_PUBLISH`` in
        # the project's weather_params.py). Maps output key -> either a
        # plain default (published as ``weather_params.get(key, default)``)
        # or a callable ``(weather_params, season, current_time) -> value``
        # for derived outputs. get_state_output publishes the generic core
        # plus everything in this table — realm-specific params (cyber_*,
        # storm_*, club levels, ...) belong to their project, not here.
        self.output_map = dict(output_map) if output_map else {}
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
                elif (isinstance(target_value, (list, tuple))
                        and isinstance(start_value, (list, tuple))
                        and len(target_value) == len(start_value)
                        and len(target_value) > 0
                        and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                for x in target_value)
                        and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                for x in start_value)):
                    # NUMERIC colour / vector params (storm_tint, rain_color,
                    # fog_color, ...) are commonly stored as plain lists - the
                    # web weather editor saves them that way. Interpolate them
                    # elementwise too, otherwise they SNAP at the state change
                    # instead of cross-fading. Non-numeric lists (e.g.
                    # possible_transitions = list of state-name strings) are
                    # left to the snap branch below.
                    tv = np.asarray(target_value, dtype=float)
                    sv = np.asarray(start_value, dtype=float)
                    self.weather_params[param] = (tv - sv) * self.progress + sv
                else:
                    self.weather_params[param] = target_value

            if self.progress >= 1.0:
                self.current_weather = self.target_weather
                self.weather_params = target_params.copy()

    def get_state_output(self, season: float, current_time: float,
                         atmos_coupling: float = 1.0) -> dict:
        """Return scheduler.state keys derived purely from weather_params.

        Does NOT include: sound, celestial_bodies, scale, season — those come from
        other subsystems and remain in send_variables().

        Publishes the GENERIC CORE (derived atmosphere values plus the
        universal weather vocabulary consumed by the shared engine
        effects: stars, clouds, rain, fog, firefly, lightning), then
        applies the project's ``output_map`` (``OUTSTATE_PUBLISH`` in the
        project's weather_params.py). Realm-specific params — cyber_*,
        storm_*, club pattern levels, heart_*, ocean, elements gates —
        live in their project's table, NOT here. A weather param only
        reaches outstate (and thus its shader) if the core or the
        project table publishes it.

        ``atmos_coupling`` (0..1, from the active set's
        ``season_atmosphere_coupling``) blends the season modulation of
        the derived 'wind' and fog outputs: 1.0 = classic signed/scaled
        behavior, 0.0 = a state's wind_speed/fog render at face value
        (sets where season is a fast time-of-day clock, e.g. ocean).
        """
        c = float(np.clip(atmos_coupling, 0.0, 1.0))
        fog = np.maximum(
            0,
            self.weather_params["fog"]
            * ((1.0 - c) + c * (0.75 - 0.25 * np.cos(np.pi * 2 * (season - 0.625)))),
        )
        cloudyness = (
            (1 - self.weather_params["starryness"])
            + (1 - self.weather_params["celestial_visibility"])
            + fog
            + self.weather_params["rain_rate"]
            + self.weather_params["wind_speed"] / 3
        ) / 4

        out = {
            "cloudyness": cloudyness,
            "fog_strength": fog,
            "fog_color": self.weather_params["fog_color"],
            "wind": self.weather_params["wind_speed"]
            * ((1.0 - c) + c * np.cos(np.pi * 2 * (season - 0.125))),
            "rain": self.weather_params["rain_rate"],
            "starryness": self.weather_params["starryness"],
            "celestial_visibility": self.weather_params["celestial_visibility"],
            "firefly_density": self.weather_params["firefly_density"],
            "meteor_rate": self.weather_params["meteor_rate"],
            "lightning_probability": self.weather_params.get("lightning_probability", 0.0),
            "season_preference": self.weather_params.get("season_preference", 0.5),
            # Ambient light level in [0.25, 1.0]: peaks at noon (season=0.5),
            # minimum at midnight (season=0 / 1). In the ocean set season is
            # repurposed as time of day, so effects that want to dim at
            # night (fish, kelp) scale their output by this value.
            "ambient_light": 0.25 + 0.75 * (0.5 - 0.5 * np.cos(2 * np.pi * season)),
        }

        # Project-declared outputs: plain default -> passthrough;
        # callable -> derived value (params, season, current_time).
        params = self.weather_params
        for key, spec in self.output_map.items():
            if callable(spec):
                out[key] = spec(params, season, current_time)
            else:
                out[key] = params.get(key, spec)
        return out

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
