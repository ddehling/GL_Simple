# Weather Set System - Quick Reference

## Overview
The weather set system allows you to create mutually exclusive collections of weather states for multi-night art installations. Each set has its own characteristics and stays isolated from other sets.

## Available Weather Sets

### 🌲 Peaceful Forest (Default)
- **States**: clear, light_rain, foggy, firefly, mushroom, bloom, leaves
- **Season Speed**: 1.0x (30 min = 1 year)
- **Season Extremity**: 1.0x (normal seasonal influence)
- **Transition Speed**: 0.8x (slower, ~5 min between weather changes)
- **Vibe**: Gentle, natural, contemplative

### ⛈️ Storm World
- **States**: windy_night, heavy_rain, thunderstorm, foggy, spooky
- **Season Speed**: 1.5x (20 min = 1 year)
- **Season Extremity**: 0.5x (less seasonal variation)
- **Transition Speed**: 1.5x (faster, ~2.7 min between changes)
- **Vibe**: Intense, dramatic, energetic

### 🏜️ Desert Realm
- **States**: clear, sandstorm, volcano, windy_night
- **Season Speed**: 0.5x (60 min = 1 year)
- **Season Extremity**: 2.0x (extreme seasonal swings)
- **Transition Speed**: 0.6x (very slow, ~6.7 min between changes)
- **Vibe**: Harsh, alien, stark

### 🌫️ Ethereal Mist
- **States**: heavy_fog, foggy, spooky, mushroom, firefly
- **Season Speed**: 0.7x (43 min = 1 year)
- **Season Extremity**: 1.5x (strong seasonal influence)
- **Transition Speed**: 0.5x (very slow, ~8 min between changes)
- **Vibe**: Mysterious, dreamlike, otherworldly

### 🌌 Cosmic Night
- **States**: clear, asteroid, windy_night
- **Season Speed**: 2.0x (15 min = 1 year)
- **Season Extremity**: 1.0x (normal seasonal influence)
- **Transition Speed**: 2.0x (fast, ~2 min between changes)
- **Vibe**: Celestial, dynamic, cosmic

### 🌈 Full Spectrum
- **States**: All 15 weather states
- **Season Speed**: 1.0x
- **Season Extremity**: 1.0x
- **Transition Speed**: 1.0x
- **Vibe**: Maximum variety, unpredictable

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

Edit `lib/weather_params.py`:

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

Update `templates/weather_sets.html` to add icon and description.

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
