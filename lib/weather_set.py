"""Weather set management: active set, queued set changes, and per-set config.

WeatherSetManager owns the event_map (passed in at construction) and provides
typed accessors for all set-level configuration keys so call sites don't need
to dig into the raw WEATHER_SETS dict. Has no scheduler, web_controller, or
audio dependencies.
"""

from typing import Optional
from lib.weather_params import WeatherState, WEATHER_SETS, DEFAULT_WEATHER_SET


class WeatherSetManager:
    """Manages weather set selection, queued set changes, and per-set config access.

    Owns the event_map (passed in from EnvironmentalSystem) and provides
    accessors for all set-level configuration. Has no scheduler, web_controller,
    or audio dependencies.
    """

    def __init__(self, event_map: dict) -> None:
        self.current_set: str = DEFAULT_WEATHER_SET
        self.target_set: Optional[str] = None
        self.weather_sets: dict = WEATHER_SETS
        self.event_map: dict = event_map
        self._cached_set_config: Optional[tuple] = None  # (set_name, config)

    # ------------------------------------------------------------------
    # Config accessors
    # ------------------------------------------------------------------

    def get_current_set_config(self) -> dict:
        """Return config dict for the current set, using a cache keyed by set name."""
        if self._cached_set_config is None or self._cached_set_config[0] != self.current_set:
            self._cached_set_config = (self.current_set, self.weather_sets[self.current_set])
        return self._cached_set_config[1]

    def get_set_states(self, set_name: Optional[str] = None) -> list:
        """Return list of WeatherState enums for the given (or current) set."""
        target = set_name or self.current_set
        return [WeatherState(s) for s in self.weather_sets[target]["states"]]

    def is_valid_set(self, set_name: str) -> bool:
        return set_name in self.weather_sets

    def get_available_set_names(self) -> list:
        return list(self.weather_sets.keys())

    def get_background_events(self) -> list:
        return self.get_current_set_config().get("background_events", [])

    def get_random_events_config(self) -> tuple:
        """Return (random_events list, random_event_rate float) for current set."""
        cfg = self.get_current_set_config()
        return cfg.get("random_events", []), cfg.get("random_event_rate", 0.0001)

    def get_season_speed(self) -> float:
        return self.get_current_set_config().get("season_speed", 1.0)

    def get_transition_speed(self) -> float:
        return self.get_current_set_config().get("transition_speed", 1.0)

    def get_season_extremity(self) -> float:
        return self.get_current_set_config().get("season_extremity", 1.0)

    # ------------------------------------------------------------------
    # Event map accessors
    # ------------------------------------------------------------------

    def get_event_names(self) -> list:
        return list(self.event_map.keys())

    def resolve_event(self, event_name: str) -> Optional[tuple]:
        """Return (effect_func, params) tuple for event_name, or None if unknown."""
        return self.event_map.get(event_name)

    # ------------------------------------------------------------------
    # Set change management
    # ------------------------------------------------------------------

    def commit_set_change(self, new_set_name: str) -> None:
        """Apply a set change immediately. Invalidates the config cache."""
        self.current_set = new_set_name
        self.target_set = None
        self._cached_set_config = None

    def queue_set_change(self, new_set_name: str) -> None:
        self.target_set = new_set_name

    def has_pending_set_change(self) -> bool:
        return self.target_set is not None

    def consume_pending_set(self) -> str:
        """Commit the queued target_set and return the new set name."""
        new_set = self.target_set
        self.commit_set_change(new_set)
        return new_set
