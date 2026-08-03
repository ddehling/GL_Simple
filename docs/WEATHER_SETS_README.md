# Weather Set System - Quick Reference

## Overview
The weather set system allows you to create mutually exclusive collections of weather states for multi-night art installations. Each set has its own characteristics and stays isolated from other sets.

## Available Weather Sets

Weather sets are **defined per project** in the active project's `projects/<id>/weather_params.py` (`WEATHER_SETS` dict) — the engine ships no set list of its own. The default set is that file's `DEFAULT_WEATHER_SET`.

The Fan project, for example, currently defines: `peaceful_forest` (default), `storm_world`, `desert_realm`, `cosmic_night`, `ocean`, `cyberpunk`, `bartiki`, `beloved`, `club`, `fynewynz`, `full_spectrum`, and `test`.

To see the active project's sets with their states and descriptions, open `http://localhost:5000/weather_sets` or read the project's `weather_params.py` directly.

## Web Interface

### Accessing the Control Panel
1. Start Stories_OGL.py
2. Open browser to: `http://localhost:5000/weather_sets`
3. (Or use mDNS: `http://glsimple.local:5000/weather_sets`)

### Changing Sets
1. Click on any weather set card
2. System queues the change
3. On next weather transition (~4 min), system:
   - Switches to new set
   - Randomly picks a weather from that set
   - Transitions to it smoothly

### Status Display
- **Current Set**: Which set is active
- **Current Weather**: Current weather state
- **Season Progress**: 0-100% through the annual cycle
- **Active/Pending badges**: Shows current and queued sets

## How It Works

### Set Isolation
- Each set only transitions between its own states
- If a weather state's normal transitions include states outside the set, they're filtered out
- If no valid transitions exist, random state from set is chosen

### Parameters Per Set

**Season Speed**: How fast the 30-minute year cycle runs
- `0.5x` = 60 min per year (slow)
- `1.0x` = 30 min per year (normal)
- `2.0x` = 15 min per year (fast)

**Season Extremity**: How much seasons influence weather transitions
- `0.5x` = Subtle seasonal bias (storms can happen anytime)
- `1.0x` = Normal seasonal bias
- `2.0x` = Extreme seasonal bias (strong preference for seasonal weathers)

**Transition Speed**: How often weather changes
- `0.5x` = ~8 min between changes (slow)
- `1.0x` = ~4 min between changes (normal)
- `2.0x` = ~2 min between changes (fast)

### Transition Process
```
1. Random check every frame (~30 fps)
2. Base probability: 1/800 per frame
3. Multiplied by: weather's Switch_rate × set's transition_speed
4. If triggered:
   - If set change pending: switch sets, pick random weather
   - Otherwise: transition to weighted random state in current set
   - Weights affected by season_extremity
```

## Adding New Sets

Edit the active project's `projects/<id>/weather_params.py` (not the engine's `lib/weather_params.py` — that only provides shared defaults, and project modules override it at runtime):

```python
WEATHER_SETS = {
    "your_set_name": {
        "name": "Display Name",
        "description": "What makes this set special",
        "states": ["clear", "foggy", "spooky"],  # Only these weathers
        "season_speed": 1.0,      # Time multiplier
        "season_extremity": 1.0,  # Seasonal influence
        "transition_speed": 1.0,  # Change frequency
    },
}
```

New sets appear in the web UI automatically (generic 📦 icon); optionally add a custom icon in `web/templates/weather_sets.html`. Sets can also be created in the browser-based weather editor (`/weather_editor`), which saves back to the project's `weather_params.py`.

## Integration with Nightly Programs

You can combine this with the nightly program system:
- Each night selects a different weather set
- Sets provide the "world" each night
- Nightly programs can further tweak parameters within that world

## Troubleshooting

**Set not changing?**
- Check terminal for "Weather set change queued" message
- Wait for next weather transition (~4 min max)
- Check web interface shows "PENDING" badge

**Unexpected weather appearing?**
- Verify all states in set are spelled correctly
- Check `test_weather_sets.py` output for validation

**Transitions too fast/slow?**
- Adjust `transition_speed` in set configuration
- Adjust individual weather's `Switch_rate` parameter
