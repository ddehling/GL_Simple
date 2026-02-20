import numpy as np
import time

from lib.weather_params import (
    WeatherState, DEFAULT_WEATHER_PARAMS, WEATHER_PRESETS,
)


class WeatherStateController:
    """Pure weather state management with no audio, scheduler, or web dependencies."""

    def __init__(self, initial_weather: WeatherState = WeatherState.CLEAR):
        self.current_weather = initial_weather
        self.target_weather = initial_weather
        self.transition_time = 0
        self.transition_start = 0
        self.progress = 0
        self.default_weather_params = DEFAULT_WEATHER_PARAMS.copy()
        self.weather_params = self.default_weather_params.copy()
        self.weather_presets = WEATHER_PRESETS

    def get_weather_params(self, weather_state: WeatherState) -> dict:
        """Get the complete set of parameters for a weather state by combining with defaults."""
        params = self.default_weather_params.copy()
        params.update(self.weather_presets[weather_state])
        return params

    def start_transition(
        self,
        new_weather: WeatherState,
        transition_duration: float,
        current_time: float,
    ) -> dict:
        """Set target state and timing. Returns target_params so the caller can handle
        audio and event scheduling without WeatherStateController having those dependencies."""
        self.target_weather = new_weather
        self.transition_time = transition_duration
        self.transition_start = current_time
        return self.get_weather_params(new_weather)

    def update(self, current_time: float) -> None:
        """Interpolate weather_params between current and target states; snap when done."""
        if self.current_weather != self.target_weather:
            self.progress = min(
                1.0, (current_time - self.transition_start) / self.transition_time
            )

            start_params = self.get_weather_params(self.current_weather)
            target_params = self.get_weather_params(self.target_weather)

            for param in target_params:
                if isinstance(target_params[param], (int, float, np.ndarray)):
                    start_value = start_params.get(
                        param, self.default_weather_params.get(param, 0)
                    )
                    self.weather_params[param] = (
                        target_params[param] - start_value
                    ) * self.progress + start_value
                else:
                    self.weather_params[param] = target_params[param]

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
        }

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
