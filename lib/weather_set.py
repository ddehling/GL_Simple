"""Weather set management: active set, queued set changes, and per-set config.

WeatherSetManager owns the event_map (passed in at construction) and provides
typed accessors for all set-level configuration keys so call sites don't need
to dig into the raw WEATHER_SETS dict. Has no scheduler, web_controller, or
audio dependencies.

Phase-8 multi-project work: weather sets, default set, and the WeatherState
enum are now project-specific. They can be passed in at construction; the
``lib.weather_params`` defaults are kept as a fallback so any standalone
caller (tests, editor utilities) that doesn't have a Project keeps working.
"""

from typing import Optional
from lib.weather_params import (
    WeatherState as _DEFAULT_WEATHER_STATE_ENUM,
    WEATHER_SETS as _DEFAULT_WEATHER_SETS,
    DEFAULT_WEATHER_SET as _DEFAULT_WEATHER_SET,
)


# Events that are scheduled automatically from per-set config fields
# (narrative_script, sound_pool_dir) rather than from the explicit
# `background_events` list. The editor should hide these from the
# "+ Add Background Event" dropdown -- the set's dedicated dropdowns
# are the single source of truth for them.
IMPLICIT_BACKGROUND_EVENTS = frozenset({"narrative_player", "sound_pool"})


class WeatherSetManager:
    """Manages weather set selection, queued set changes, and per-set config access.

    Owns the event_map (passed in from EnvironmentalSystem) and provides
    accessors for all set-level configuration. Has no scheduler, web_controller,
    or audio dependencies.
    """

    def __init__(
        self,
        event_map: dict,
        weather_sets: Optional[dict] = None,
        default_set: Optional[str] = None,
        weather_state_enum=None,
    ) -> None:
        self.weather_sets: dict = (
            weather_sets if weather_sets is not None else _DEFAULT_WEATHER_SETS
        )
        self.current_set: str = (
            default_set if default_set is not None else _DEFAULT_WEATHER_SET
        )
        # WeatherState enum used for casting state strings → enum values in
        # ``get_set_states``. Project-specific so WoL can use a different
        # enum (or just the same one with WoL-only sets).
        self._weather_state_enum = (
            weather_state_enum if weather_state_enum is not None else _DEFAULT_WEATHER_STATE_ENUM
        )
        self.target_set: Optional[str] = None
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
        enum_cls = self._weather_state_enum
        return [enum_cls(s) for s in self.weather_sets[target]["states"]]

    def is_valid_set(self, set_name: str) -> bool:
        return set_name in self.weather_sets

    def get_available_set_names(self) -> list:
        return list(self.weather_sets.keys())

    def get_background_events(self) -> list:
        """Return the effective background-events list for the active set.

        Starts from the explicit `background_events` list, then auto-adds
        `narrative_player` whenever the set has a non-empty `narrative_script`
        and `sound_pool` whenever `sound_pool_dir` is non-empty. The dropdown
        fields in the editor are the single source of truth for those two
        features -- users don't have to remember to add the events to the
        pill list too. Dedupes so an explicit entry won't cause duplicate
        scheduling.
        """
        cfg = self.get_current_set_config()
        events = list(cfg.get("background_events", []))

        implied = []
        if cfg.get("narrative_script"):
            implied.append("narrative_player")
        if cfg.get("sound_pool_dir"):
            implied.append("sound_pool")
        for ev in implied:
            if ev not in events:
                events.append(ev)
        return events

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

    def get_season_atmosphere_coupling(self) -> float:
        """How much season modulates the derived wind/fog outputs (0..1).

        1.0 (default) keeps the classic land-realm behavior: published
        'wind' is wind_speed * cos(2*pi*(season - 0.125)) — signed and
        season-scaled — and fog strength breathes +-25% over the season
        cycle. Sets that repurpose season as a fast time-of-day clock
        (ocean) set this to 0 so a state's declared wind_speed/fog render
        at face value instead of collapsing to ~0 whenever the clock
        crosses a cosine zero (e.g. a maelstrom with no current).
        """
        return float(self.get_current_set_config().get("season_atmosphere_coupling", 1.0))

    def get_narrative_script(self) -> Optional[str]:
        """Return the current set's narrative script path, or None if unset.

        Sets that omit the field (older saved data) behave as if it were None.
        Empty strings are treated as None too.
        """
        script = self.get_current_set_config().get("narrative_script")
        if not script:
            return None
        return script

    def get_sound_pool_dir(self) -> Optional[str]:
        """Return the current set's ambient sound-pool directory, or None.

        Same semantics as get_narrative_script: missing/empty -> None.
        """
        sdir = self.get_current_set_config().get("sound_pool_dir")
        if not sdir:
            return None
        return sdir

    def get_audio_source(self) -> Optional[str]:
        """Return the current set's preferred audio-reactivity source, or None.

        Sets that declare ``audio_source`` (one of "linein"/"loopback"/
        "internal"/"microphone") switch the analyzer to it on activation —
        e.g. WoL's "Elements" set uses "internal" so its own AudioEngine
        music mix drives the audio-reactive visuals without the operator
        touching config.yaml. Missing/empty -> None (leave the machine's
        configured source untouched).
        """
        src = self.get_current_set_config().get("audio_source")
        if not src:
            return None
        return src

    def get_narrative_node_delay(self) -> Optional[float]:
        """Return the current set's inter-node narrative delay in seconds.

        None (the default for sets that omit the key) keeps the
        narrative_player's own default; a number overrides the pause
        between one clip's audio ending and the next node starting.
        """
        raw = self.get_current_set_config().get("narrative_node_delay")
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def get_sound_pool_crossfade(self) -> float:
        """Return the current set's sound-pool crossfade window in seconds.

        0 (the default for sets that omit the key) keeps the original gap
        playback. A positive value makes the pool play clips gaplessly,
        crossfading each into the next over that many seconds.
        """
        try:
            return float(self.get_current_set_config().get("sound_pool_crossfade", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    # ------------------------------------------------------------------
    # Event map accessors
    # ------------------------------------------------------------------

    def get_event_names(self) -> list:
        return list(self.event_map.keys())

    def resolve_event(self, event_name: str) -> Optional[tuple]:
        """Return ``(effect_func, params, meta)`` for event_name, or None.

        Supports both 2-tuple (legacy: effect_func, params) and 3-tuple
        (Phase-6: effect_func, params, meta) entries. Meta is the
        per-entry options dict, e.g. ``{"group": "leaves"}``; an empty
        dict is returned when none was supplied.
        """
        entry = self.event_map.get(event_name)
        if entry is None:
            return None
        if len(entry) == 2:
            effect_func, params = entry
            return effect_func, params, {}
        if len(entry) == 3:
            return entry
        raise ValueError(
            f"event_map[{event_name!r}] must be a 2- or 3-tuple, got {entry!r}"
        )

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
