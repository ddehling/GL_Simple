# Event Map System Documentation

## Overview

The Event Map System provides a centralized, extensible way to manage visual effects and events in the weather system. Instead of hardcoding event scheduling with if statements, events are now defined in a map and referenced by name in weather state configurations.

## Architecture

### Event Map (`EVENT_MAP`)

Since the engine/project split, the event map is **defined by the active project**, not the engine. Each project declares an `EVENT_MAP` dict in its own repo.

**Location:** `projects/<id>/event_map.py` — module-level `EVENT_MAP`

```python
from renderer import effects as fx  # resolves to the active project's shaders/

EVENT_MAP = {
    # Background events (continuous)
    "clouds": (fx.shader_drifting_clouds, {}),
    "firefly": (fx.shader_firefly, {"squish_top_width": 0.1}),
    "stars": (fx.shader_stars, {}),
    "rain": (fx.shader_rain, {}),
    "fog": (fx.shader_fog, {
        "strength": 0.0,
        "color": (0.7, 0.7, 0.8),
        "fog_near": 20.0,
        "fog_far": 80.0
    }),

    # On-transition events (called when weather state starts)
    "sandstorm_event": (fx.shader_sandstorm, {}),
    "fog_beings_event": (fx.shader_chromatic_fog_beings, {}),
}
```

**Loading:** at startup (and on project swap), `Project.load_event_map()` (`core/project.py`) merges the project's `EVENT_MAP` over `core/default_events.py`'s `DEFAULT_EVENT_MAP` — universal features (currently `narrative_player` and `sound_pool`) that every project inherits without re-registering them. Project entries override defaults on key collision. At runtime the merged map is owned by `WeatherSetManager` (`lib/weather_set.py`).

### Event Format

Each entry is a tuple:

1. **`(function, params)`** — shader function plus its keyword parameters:
   ```python
   "firefly": (fx.shader_firefly, {"squish_top_width": 0.1})
   ```

2. **`(function, params, meta)`** — with a per-entry meta dict. `{"group": "<group-id>"}` routes the event to a specific canvas group from `project.yaml` (translated to a frame_id at schedule time):
   ```python
   "leaf_column": (fx.shader_falling_leaves, {}, {"group": "leaves"})
   ```

## Usage

### 1. Background Events (Weather Set Level)

Background events run continuously (10E9 seconds) when a weather set is active. They are defined in the weather set configuration.

**Configuration Location:** `projects/<id>/weather_params.py` - `WEATHER_SETS`

```python
WEATHER_SETS = {
    "peaceful_forest": {
        "states": ["clear", "light_rain", "foggy", ...],
        "background_events": ["clouds", "firefly", "stars"],  # Event names from event_map
        "season_speed": 1.0,
        "transition_speed": 1.0,
        "season_extremity": 1.0,
        "allowed_parameters": [...]
    }
}
```

**How it works:**
- When a weather set becomes active, `_initialize_weather_set_events()` is called
- All existing events are cancelled
- Background events listed in the set config are scheduled from the event map
- Events run until the weather set changes

### 2. On-Transition Events (Weather State Level)

On-transition events are triggered when a specific weather state becomes active. They run for a specified duration and then end.

**Configuration Location:** `projects/<id>/weather_params.py` - `WEATHER_PRESETS`

```python
WeatherState.SANDSTORM: {
    "ARI": 30,
    "ambient_sound": "26 Heavy Wind Gusts Blowing Sand EDITED.wav",
    # ... other parameters ...
    "on_transition_events": [("sandstorm_event", 100)],
    # Format: (event_name, duration_seconds, start_delay_seconds=0, frame_id=0)
}
```

**On-Transition Event Format:**

Each entry in `on_transition_events` is a tuple:
- `(event_name, duration)` - Starts immediately on frame 0
- `(event_name, duration, delay)` - Starts `delay` seconds after the transition
- `(event_name, duration, delay, frame_id)` - Also target a specific frame/canvas

Note: the third slot is a **start delay**, not a frame_id — frame_id moved to the fourth slot. (An event_map entry's `{"group": ...}` meta overrides frame_id either way.)

**Examples:**
```python
# Single event, immediate, for 100 seconds
"on_transition_events": [("sandstorm_event", 100)]

# Multiple events with different durations
"on_transition_events": [
    ("fog_beings_event", 80),
    ("lightning_event", 30),
]

# Staggered starts — the whale arrives 20 s after the dolphins
"on_transition_events": [
    ("dolphin_pod", 120),
    ("whale_passage", 90, 20),
]

# Explicit frame/canvas targeting
"on_transition_events": [
    ("main_effect", 60, 0, 0),       # Frame 0
    ("secondary_effect", 45, 0, 1),  # Frame 1
]
```

**How it works:**
- When `transition_to_weather()` is called, it reads `on_transition_events` from the target weather preset
- Each event is scheduled using `_schedule_event_from_map()`
- Events run for their specified duration and then automatically end
- Next weather transition can occur while events are still running

## Adding New Events

### Step 1: Add Event to Event Map

Add your event to the `EVENT_MAP` dictionary in the project's `projects/<id>/event_map.py`:

```python
EVENT_MAP = {
    # ... existing events ...

    # Your new event
    "my_new_effect": (fx.shader_my_effect, {}),

    # Or with parameters
    "my_parametric_effect": (fx.shader_my_effect, {
        "intensity": 0.8,
        "color": (1.0, 0.5, 0.2)
    }),
}
```

### Step 2A: Use as Background Event

Add to a weather set's `background_events` list:

```python
"my_custom_set": {
    "states": ["clear", "foggy"],
    "background_events": ["clouds", "my_new_effect"],  # Add here
    # ... other config ...
}
```

### Step 2B: Use as On-Transition Event

Add to a weather state's `on_transition_events` list:

```python
WeatherState.MY_STATE: {
    "ambient_sound": "my_sound.wav",
    "fog": 0.5,
    # ... other parameters ...
    "on_transition_events": [
        ("my_new_effect", 45),  # Runs for 45 seconds on transition
    ],
}
```

## Helper Methods

### `_schedule_event_from_map(event_name, start_time, duration, frame_id=0)`

Schedules an event from the event map.

**Parameters:**
- `event_name` (str) - Name of event in the merged event map
- `start_time` (float) - When to start (usually 0 for immediate)
- `duration` (float) - How long to run in seconds
- `frame_id` (int) - Which display frame (default 0)

**Returns:**
- Event ID from scheduler, or None if event not found

**Example:**
```python
# Schedule fog_beings for 60 seconds on frame 0
self._schedule_event_from_map("fog_beings", 0, 60, frame_id=0)
```

## Migration from Old System

### Before (Hardcoded):
```python
def transition_to_weather(self, new_weather: WeatherState, transition_duration: float = 10.0):
    # ... setup code ...
    
    if new_weather == WeatherState.SANDSTORM:
        self.scheduler.schedule_event(0, 100, fx.shader_sandstorm, frame_id=0)
    
    if new_weather == WeatherState.HEAVY_FOG:
        self.scheduler.schedule_event(0, 80, fx.shader_chromatic_fog_beings, frame_id=0)
    
    if new_weather == WeatherState.LEAVES:
        if not self.scheduler.state.get("has_leaves", False):
            self.scheduler.schedule_event(0, 60, fx.shader_falling_leaves, squish_top_width=self.scale, frame_id=0)
```

### After (Event Map System):

**In the project's `event_map.py`:**
```python
EVENT_MAP = {
    "sandstorm_event": (fx.shader_sandstorm, {}),
    "fog_beings_event": (fx.shader_chromatic_fog_beings, {}),
    "falling_leaves_event": (fx.shader_falling_leaves, {"squish_top_width": 0.1}),
}
```

**In the project's weather presets:**
```python
WeatherState.SANDSTORM: {
    # ... parameters ...
    "on_transition_events": [("sandstorm_event", 100)],
}

WeatherState.HEAVY_FOG: {
    # ... parameters ...
    "on_transition_events": [("fog_beings_event", 80)],
}

WeatherState.LEAVES: {
    # ... parameters ...
    "on_transition_events": [("falling_leaves_event", 60)],
}
```

**In the engine (`Stories_OGL.py`, project-agnostic — no per-project edits ever needed):**
```python
def transition_to_weather(self, new_weather, transition_duration: float = 10.0):
    # ... setup code ...

    # Entry schema: [name, duration, delay=0, frame_id=0]
    on_transition_events = target_params.get("on_transition_events", [])
    for event_config in on_transition_events:
        if isinstance(event_config, (tuple, list)) and len(event_config) >= 2:
            event_name, duration = event_config[:2]
            delay = float(event_config[2]) if len(event_config) > 2 else 0.0
            frame_id = int(event_config[3]) if len(event_config) > 3 else 0
            self._schedule_event_from_map(event_name, delay, duration, frame_id=frame_id)
```

## Benefits

### 1. **Centralized Event Definitions**
All of a project's events are defined in one place (its `event_map.py`), making them easy to find and maintain.

### 2. **Reusability**
Events can be used as both background events and on-transition events without duplication.

### 3. **Data-Driven Configuration**
Weather behavior is defined in project configuration (`projects/<id>/weather_params.py`) rather than code logic.

### 4. **Easy Extension**
Adding new events only requires:
1. Add to the project's `EVENT_MAP`
2. Reference by name in the project's weather config

### 5. **No Engine Changes for New Weather States or Events**
New weather states and events live entirely in the project repo — the engine (`Stories_OGL.py`) is never modified.

### 6. **Flexibility**
Same event can be used with different durations and frames in different weather states.

## Event Lifecycle

### Background Events:
```
Weather Set Activated
  ↓
_initialize_weather_set_events() called
  ↓
Cancel all existing events
  ↓
Schedule background_events from set config
  ↓
Events run for 10E9 seconds (effectively forever)
  ↓
Weather Set Changed → Repeat
```

### On-Transition Events:
```
Weather State Transition Triggered
  ↓
transition_to_weather() called
  ↓
Read on_transition_events from weather preset
  ↓
Schedule each event with specified duration
  ↓
Events run for specified duration
  ↓
Events automatically end
  ↓
(Can transition to new weather while events still running)
```

## Default Parameters

All weather states inherit from `DEFAULT_WEATHER_PARAMS`:

```python
DEFAULT_WEATHER_PARAMS = {
    # ... other defaults ...
    "on_transition_events": [],  # Empty by default
}
```

Weather states without `on_transition_events` simply don't trigger any events on transition.

## Examples

### Example 1: Simple Background Events
```python
# Weather set with continuous cloud and star rendering
"starry_sky": {
    "states": ["clear", "windy_night"],
    "background_events": ["clouds", "stars"],
    "season_speed": 1.0,
}
```

### Example 2: Single On-Transition Event
```python
WeatherState.FOGGY: {
    "fog": 0.8,
    "fog_color": np.array([0.7, 0.7, 0.8]),
    # ... other params ...
    "on_transition_events": [("fog_beings_event", 60)],  # Fog beings for 60s
}
```

### Example 3: Multiple On-Transition Events
```python
WeatherState.THUNDERSTORM: {
    "lightning_probability": 0.8,
    "rain_rate": 0.9,
    # ... other params ...
    "on_transition_events": [
        ("lightning_event", 120, 0),      # Lightning for 2 minutes
        ("heavy_rain_particles", 150, 0), # Rain particles for 2.5 minutes
        ("thunder_rumble", 90, 1),        # Thunder effect on secondary display
    ],
}
```

### Example 4: Event Variants via Wrapper Functions
`EVENT_MAP` values are plain tuples, so when two entries need the same shader with different baked-in behavior, define wrapper functions in `event_map.py`. This also gives each entry its own function identity — the scheduler dedups on `(action, frame_id)`, so mapping two event names straight to the same function makes one of them a silently-dropped "duplicate" when both run at once:

```python
def _strobe_burst(state, outstate, **kw):
    return fx.shader_club_strobe(state, outstate, burst=True, **kw)

EVENT_MAP = {
    "club_strobe":  (fx.shader_club_strobe, {}),   # continuous background
    "strobe_burst": (_strobe_burst, {}),           # one-shot variant
}
```

## Debugging

### Print Event Scheduling
The system automatically prints when events are scheduled:

```
🔄 Initializing events for weather set: 'peaceful_forest'
   📅 Scheduling background event: clouds
   📅 Scheduling background event: firefly
✓ Background events initialized for 'peaceful_forest'

   🎬 Transition event: fog_beings_event (80s)
```

### Check Active Events
```python
# Print number of active events
print(f"Active events: {len(self.scheduler.active_events)}")

# Inspect active events
for event_id, event_data in self.scheduler.active_events.items():
    print(f"Event {event_id}: {event_data}")
```

### Unknown Event Warning
If an event name doesn't exist in the merged map:
```
[WEATHER] Unknown event: nonexistent_event
```

## Performance Considerations

- **Background events** run continuously - use sparingly (2-5 per set recommended)
- **On-transition events** are temporary - can be more intensive
- Events automatically clean up when duration expires
- Use appropriate durations to avoid event buildup
- Consider `frame_id` to distribute load across multiple displays

## Future Enhancements

Potential additions to the event map system:

1. **Conditional Events** - Events that check conditions before scheduling
2. **Event Chains** - Events that trigger other events when they complete
3. **Dynamic Durations** - Durations based on weather parameters
4. **Event Priorities** - Control which events override others
5. **Event Categories** - Group events for easier management
6. **Hot Reload** - Update event_map without restarting

---

*Document Version: 2.0 — updated for the engine/project split (per-project `event_map.py` + `core/default_events.py` merge, delay slot in `on_transition_events`)*
*Created: 2025-12-30*
*Last Updated: 2026-08-02*
