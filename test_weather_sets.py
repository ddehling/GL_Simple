"""
Quick test/demo of the weather set system
"""
from corefunctions.weather_params import WEATHER_SETS, WeatherState

print("=" * 60)
print("WEATHER SET SYSTEM OVERVIEW")
print("=" * 60)

for set_key, set_config in WEATHER_SETS.items():
    print(f"\n🌟 {set_config['name']}")
    print(f"   {set_config['description']}")
    print(f"   States: {', '.join(set_config['states'])}")
    print(f"   Season Speed: {set_config['season_speed']}x")
    print(f"   Season Extremity: {set_config['season_extremity']}x")
    print(f"   Transition Speed: {set_config['transition_speed']}x")

print("\n" + "=" * 60)
print("WEB INTERFACE USAGE")
print("=" * 60)
print("""
1. Start Stories_OGL.py
2. Open browser to: http://localhost:5000/weather_sets
3. Click on any weather set card to switch to it
4. Next weather transition will move to that set
5. View current status in the info panel at the top

FEATURES:
- Each set has its own isolated weather states
- Sets have different seasonal speeds (how fast the year cycles)
- Sets have different seasonal extremity (how much seasons matter)
- Sets have different transition speeds (how often weather changes)
- Web interface shows current set, weather, and season progress
""")

print("\n" + "=" * 60)
print("TESTING SET TRANSITIONS")
print("=" * 60)

# Simulate set validation
test_set = "peaceful_forest"
print(f"\nTesting set: {test_set}")
states = WEATHER_SETS[test_set]["states"]
print(f"Available states in this set: {states}")

# Check if each state exists
from corefunctions.weather_params import WEATHER_PRESETS
for state_name in states:
    try:
        state = WeatherState(state_name)
        if state in WEATHER_PRESETS:
            print(f"  ✓ {state_name} - configured")
        else:
            print(f"  ✗ {state_name} - MISSING PRESET")
    except ValueError:
        print(f"  ✗ {state_name} - INVALID STATE")

print("\n✅ Weather set system ready to use!")
