import numpy as np
from enum import Enum

class WeatherState(Enum):
    CLEAR = "clear"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"
    THUNDERSTORM = "thunderstorm"
    WINDY_NIGHT = "windy_night"
    FOGGY = "foggy"
    HEAVY_FOG = "heavy_fog"
    SPOOKY = "spooky"
    FIREFLY = "firefly"
    VOLCANO = "volcano"
    SANDSTORM = "sandstorm"
    HURRICANE = "hurricane"
    ASTEROID = "asteroid"
    MUSHROOM = "mushroom"
    LEAVES = "leaves"
    BLOOM = "bloom"
    CYBER_NEON_DRIZZLE = "cyber_neon_drizzle"
    CYBER_DATA_STORM = "cyber_data_storm"
    CYBER_SMOG_HAZE = "cyber_smog_haze"
    CYBER_ELECTRIC_STORM = "cyber_electric_storm"
    CYBER_ACID_RAIN = "cyber_acid_rain"
    CYBER_HOLOGRAM_NIGHT = "cyber_hologram_night"
    CYBER_BLACKOUT = "cyber_blackout"
    CYBER_GLITCH_FOG = "cyber_glitch_fog"
    CYBER_NEON_CLEAR = "cyber_neon_clear"
    CYBER_DRONE_PATROL = "cyber_drone_patrol"
    OCEAN_CALM_SHALLOWS = "ocean_calm_shallows"
    OCEAN_CHOPPY_SURFACE = "ocean_choppy_surface"
    OCEAN_STORM_SURGE = "ocean_storm_surge"
    OCEAN_DEEP_CALM = "ocean_deep_calm"
    OCEAN_BIOLUMINESCENT_SWARM = "ocean_bioluminescent_swarm"
    OCEAN_ABYSS = "ocean_abyss"
    OCEAN_KELP_FOREST = "ocean_kelp_forest"
    OCEAN_CORAL_REEF = "ocean_coral_reef"
    OCEAN_JELLYFISH_BLOOM = "ocean_jellyfish_bloom"
    OCEAN_MIDNIGHT_OPEN_WATER = "ocean_midnight_open_water"
    OCEAN_MAELSTROM = "ocean_maelstrom"
    BARTIKI_DAWN = "bartiki_dawn"
    BARTIKI_MORNING_RUSH = "bartiki_morning_rush"
    BARTIKI_MIDDAY = "bartiki_midday"
    BARTIKI_AFTERNOON = "bartiki_afternoon"
    BARTIKI_EVENING_RUSH = "bartiki_evening_rush"
    BARTIKI_DUSK = "bartiki_dusk"
    BARTIKI_NIGHT = "bartiki_night"
    BARTIKI_LATE_NIGHT = "bartiki_late_night"
    BARTIKI_STORMY = "bartiki_stormy"
    TEST_RGB = "test_rgb"
    TEST_HUE_BIN = "test_hue_bin"

# Global parameters that are always available in every weather set
# These cannot be removed from sets but their values can be customized per weather state
GLOBAL_PARAMETERS = [
    "possible_transitions",
    "transition_weights",
    "transition_duration",
    "Sound_volume",
    "ARI",
    "fog",
    "fog_color",
    "season_preference",
    "Switch_rate",
    "on_transition_events",
]

# Available background events (always-active effects)
# This is the single source of truth for which events can be used as continuous background effects.
# Add new background-capable events here. They must also exist in Stories_OGL.py's event_map.
AVAILABLE_BACKGROUND_EVENTS = [
    'clouds',
    'firefly',
    'stars',
    'rain',
    'fog',
    'sandstorm',
    'fog_beings',
    'falling_leaves',
]

# Parameter definitions for the weather editor
# Defines the type and input configuration for each parameter
PARAMETER_DEFINITIONS = {
    'ARI': {'type': 'number', 'step': 1},
    'Aurora_probability': {'type': 'number', 'step': 0.1},
    'Sound_volume': {'type': 'number', 'step': 0.1},
    'Switch_rate': {'type': 'number', 'step': 0.1},
    'Weird': {'type': 'number', 'step': 0.1},
    'Wolfy': {'type': 'number', 'step': 0.1},
    'ambient_sound': {'type': 'text'},
    'bioluminescence': {'type': 'number', 'step': 0.05},
    'bubble_density': {'type': 'number', 'step': 0.05},
    'celestial_visibility': {'type': 'number', 'step': 0.1},
    'data_flow_rate': {'type': 'number', 'step': 0.05},
    'drone_activity': {'type': 'number', 'step': 0.05},
    'electric_interference': {'type': 'number', 'step': 0.05},
    'firefly_density': {'type': 'number', 'step': 0.1},
    'fog': {'type': 'number', 'step': 0.05},
    'fog_color': {'type': 'array', 'length': 3},
    'glitch_probability': {'type': 'number', 'step': 0.05},
    'hologram_density': {'type': 'number', 'step': 0.05},
    'kelp_density': {'type': 'number', 'step': 0.05},
    'light_pollution': {'type': 'number', 'step': 0.05},
    'lightning_probability': {'type': 'number', 'step': 0.05},
    'marine_life_activity': {'type': 'number', 'step': 0.05},
    'meteor_rate': {'type': 'number', 'step': 0.05},
    'neon_intensity': {'type': 'number', 'step': 0.05},
    'on_transition_events': {'type': 'event-list'},
    'pollution_level': {'type': 'number', 'step': 0.05},
    'possible_transitions': {'type': 'array-string'},
    'rain_rate': {'type': 'number', 'step': 0.1},
    'sand_density': {'type': 'number', 'step': 0.1},
    'scan_line_intensity': {'type': 'number', 'step': 0.05},
    'season_preference': {'type': 'number', 'step': 0.025},
    'skiptime': {'type': 'number', 'step': 0.5},
    'spookyness': {'type': 'number', 'step': 0.1},
    'starryness': {'type': 'number', 'step': 0.1},
    'tide_level': {'type': 'number', 'step': 0.05},
    'train_density': {'type': 'number', 'step': 0.1},
    'train_speed': {'type': 'number', 'step': 0.5},
    'transition_duration': {'type': 'number', 'step': 1},
    'transition_weights': {'type': 'array-number'},
    'tree_prob': {'type': 'number', 'step': 0.1},
    'volcano_level': {'type': 'number', 'step': 0.1},
    'wave_amplitude': {'type': 'number', 'step': 0.05},
    'wave_speed': {'type': 'number', 'step': 0.05},
    'wind_speed': {'type': 'number', 'step': 0.1},
}

# Default weather parameters
DEFAULT_WEATHER_PARAMS = {
    "wind_speed": 0,
    "rain_rate": 0,
    "lightning_probability": 0,
    "starryness": 1.0,
    "spookyness": 0.0,
    "fog": 0.0,
    "fog_color": np.array([0.7, 0.7, 0.7]),
    "possible_transitions": ['light_rain', 'foggy', 'windy_night'],
    "transition_weights": [1.0, 2.0, 0.5],
    "transition_duration": 20.0,
    "celestial_visibility": 1.0,
    "firefly_density": 0.0,
    "Aurora_probability": 0.0,
    "Wolfy": 0.0,
    "Switch_rate": 1.0,
    "meteor_rate": 0.0,
    "volcano_level": 0.0,
    "sand_density": 0.0,
    "skiptime": 0.0,
    "tree_prob": 0.0,
    "Weird": 0.0,
    "Sound_volume": 1.0,
    "season_preference": 0.375,
    "ambient_sound": None,
    "ARI": 0.0,
}

# Weather presets
# Weather state parameters
WEATHER_PRESETS = {
    WeatherState.ASTEROID: {
        "ARI": 40,
        "Switch_rate": 3,
        "ambient_sound": "toot.toot",
        "celestial_visibility": 1,
        "meteor_rate": 0.2,
        "possible_transitions": ["clear"],
        "season_preference": 0.875,
        "starryness": 1,
        "transition_weights": [1],
    },

    WeatherState.BARTIKI_AFTERNOON: {
        "ARI": 0,
        "Switch_rate": 0.1,
        "ambient_sound": "01 Rain Light EDITED.wav",
        "celestial_visibility": 0,
        "fog": 0.05,
        "fog_color": np.array([0.15, 0.15, 0.25]),
        "meteor_rate": 0,
        "possible_transitions": ["bartiki_evening_rush", "bartiki_stormy"],
        "rain_rate": 0.2,
        "season_preference": 0.6,
        "starryness": 0,
        "train_density": 1,
        "train_speed": 8,
        "transition_duration": 20,
        "transition_weights": [1, 0.2],
        "wind_speed": 0.2,
    },

    WeatherState.BARTIKI_DAWN: {
        "ARI": 0,
        "Switch_rate": 0.1,
        "celestial_visibility": 0.3,
        "fog": 0.75,
        "fog_color": np.array([0.4, 0.25, 0.15]),
        "meteor_rate": 0,
        "possible_transitions": ["bartiki_morning_rush", "bartiki_stormy"],
        "rain_rate": 0,
        "season_preference": 0.2,
        "starryness": 0.3,
        "train_density": 1,
        "train_speed": 4,
        "transition_duration": 20,
        "transition_weights": [1, 0.1],
        "wind_speed": 0.1,
    },

    WeatherState.BARTIKI_DUSK: {
        "ARI": 0,
        "Switch_rate": 0.1,
        "celestial_visibility": 0.6,
        "fog": 0.2,
        "fog_color": np.array([0.15, 0.12, 0.2]),
        "meteor_rate": 0,
        "possible_transitions": ["bartiki_night"],
        "rain_rate": 0,
        "season_preference": 0.8,
        "starryness": 0.4,
        "train_density": 1,
        "train_speed": 6,
        "transition_duration": 20,
        "transition_weights": [1],
        "wind_speed": 0.1,
    },

    WeatherState.BARTIKI_EVENING_RUSH: {
        "ARI": 0,
        "Switch_rate": 0.1,
        "celestial_visibility": 0.2,
        "fog": 0.1,
        "fog_color": np.array([0.35, 0.2, 0.1]),
        "meteor_rate": 0,
        "possible_transitions": ["bartiki_dusk", "bartiki_stormy"],
        "rain_rate": 0,
        "season_preference": 0.7,
        "starryness": 0,
        "train_density": 1,
        "train_speed": 10,
        "transition_duration": 20,
        "transition_weights": [1, 0.1],
        "wind_speed": 0.15,
    },

    WeatherState.BARTIKI_LATE_NIGHT: {
        "ARI": 0,
        "Switch_rate": 0.08,
        "celestial_visibility": 1,
        "fog": 0,
        "fog_color": np.array([0.03, 0.03, 0.1]),
        "meteor_rate": 0.8,
        "possible_transitions": ["bartiki_dawn"],
        "rain_rate": 0,
        "season_preference": 0.05,
        "starryness": 1,
        "train_density": 1,
        "train_speed": 2,
        "transition_duration": 20,
        "transition_weights": [1],
        "wind_speed": 0,
    },

    WeatherState.BARTIKI_MIDDAY: {
        "ARI": 0,
        "Switch_rate": 0.1,
        "celestial_visibility": 0,
        "fog": 0,
        "fog_color": np.array([0.1, 0.1, 0.2]),
        "meteor_rate": 0,
        "possible_transitions": ["bartiki_afternoon", "bartiki_stormy"],
        "rain_rate": 0,
        "season_preference": 0.5,
        "starryness": 0,
        "train_density": 1,
        "train_speed": 8,
        "transition_duration": 20,
        "transition_weights": [1, 0.15],
        "wind_speed": 0.15,
    },

    WeatherState.BARTIKI_MORNING_RUSH: {
        "ARI": 0,
        "Switch_rate": 0.1,
        "celestial_visibility": 0,
        "fog": 0.6,
        "fog_color": np.array([0.1, 0.1, 0.2]),
        "meteor_rate": 0,
        "possible_transitions": ["bartiki_midday", "bartiki_stormy"],
        "rain_rate": 0,
        "season_preference": 0.3,
        "starryness": 0,
        "train_density": 1,
        "train_speed": 10,
        "transition_duration": 20,
        "transition_weights": [1, 0.15],
        "wind_speed": 0.1,
    },

    WeatherState.BARTIKI_NIGHT: {
        "ARI": 0,
        "Switch_rate": 0.08,
        "celestial_visibility": 1,
        "fog": 0,
        "fog_color": np.array([0.05, 0.05, 0.15]),
        "meteor_rate": 0.5,
        "possible_transitions": ["bartiki_late_night", "bartiki_stormy"],
        "rain_rate": 0.1,
        "season_preference": 0.95,
        "starryness": 1,
        "train_density": 1,
        "train_speed": 4,
        "transition_duration": 20,
        "transition_weights": [1, 0.05],
        "wind_speed": 0.05,
    },

    WeatherState.BARTIKI_STORMY: {
        "ARI": 30,
        "Switch_rate": 0.15,
        "ambient_sound": "Rain Heavy 01 EDITED.wav",
        "celestial_visibility": 0,
        "fog": 0.35,
        "fog_color": np.array([0.15, 0.15, 0.2]),
        "lightning_probability": 0.3,
        "meteor_rate": 0,
        "possible_transitions": ["bartiki_midday", "bartiki_afternoon", "bartiki_evening_rush", "bartiki_morning_rush"],
        "rain_rate": 0.8,
        "season_preference": 0.5,
        "starryness": 0,
        "train_density": 1,
        "train_speed": 5,
        "transition_duration": 20,
        "transition_weights": [1, 1, 0.8, 0.8],
        "wind_speed": 0.6,
    },

    WeatherState.BLOOM: {
        "ARI": 45,
        "Aurora_probability": 0,
        "Sound_volume": 1,
        "Switch_rate": 1,
        "Weird": 0,
        "ambient_sound": "09 Nightingale.mp3",
        "firefly_density": 0.5,
        "fog": 0,
        "meteor_rate": 0.05,
        "possible_transitions": ["clear", "windy_night", "leaves", "firefly", "mushroom"],
        "rain_rate": 0,
        "season_preference": 0.5,
        "skiptime": 0,
        "transition_weights": [1, 0.25, 1, 1, 0.5],
        "tree_prob": 0.1,
        "wind_speed": 0.2,
    },

    WeatherState.CLEAR: {
        "ARI": 40,
        "Aurora_probability": 0.5,
        "Switch_rate": 0.9,
        "Weird": 1,
        "ambient_sound": "Forest Cicadas EDITED.wav",
        "meteor_rate": 0.25,
        "possible_transitions": ["light_rain", "foggy", "windy_night", "firefly", "mushroom", "leaves", "bloom"],
        "season_preference": 0.375,
        "transition_weights": [1, 1, 0.75, 0.5, 0.2, 0.75, 0.75],
        "tree_prob": 1,
        "wind_speed": 0.2,
    },

    WeatherState.CYBER_ACID_RAIN: {
        "ARI": 42,
        "ambient_sound": "Rain Heavy 01 EDITED.wav",
        "celestial_visibility": 0,
        "fog": 0.6,
        "fog_color": np.array([0.5, 0.8, 0.2]),
        "light_pollution": 0.5,
        "neon_intensity": 0.5,
        "pollution_level": 0.8,
        "possible_transitions": ["cyber_smog_haze", "cyber_neon_drizzle", "cyber_electric_storm"],
        "rain_rate": 0.7,
        "season_preference": 0.7,
        "starryness": 0,
        "transition_weights": [1, 1.2, 0.6],
    },

    WeatherState.CYBER_BLACKOUT: {
        "ARI": 35,
        "ambient_sound": "Wind Strong EDITED.wav",
        "celestial_visibility": 0.8,
        "fog": 0.3,
        "fog_color": np.array([0.1, 0.1, 0.15]),
        "glitch_probability": 0.2,
        "hologram_density": 0,
        "light_pollution": 0.1,
        "neon_intensity": 0,
        "possible_transitions": ["cyber_neon_clear", "cyber_smog_haze", "cyber_glitch_fog"],
        "season_preference": 0.9,
        "starryness": 0.9,
        "transition_weights": [1.5, 0.5, 0.3],
    },

    WeatherState.CYBER_DATA_STORM: {
        "ARI": 40,
        "ambient_sound": "Wind Strong EDITED.wav",
        "celestial_visibility": 0,
        "data_flow_rate": 1,
        "electric_interference": 0.7,
        "fog": 0.3,
        "fog_color": np.array([0, 1, 0.3]),
        "glitch_probability": 0.5,
        "light_pollution": 0.5,
        "neon_intensity": 0.6,
        "possible_transitions": ["cyber_electric_storm", "cyber_glitch_fog", "cyber_neon_drizzle"],
        "season_preference": 0.5,
        "starryness": 0,
        "transition_weights": [0.8, 0.6, 1],
    },

    WeatherState.CYBER_DRONE_PATROL: {
        "ARI": 38,
        "ambient_sound": "Wind Strong EDITED.wav",
        "celestial_visibility": 0.3,
        "drone_activity": 1,
        "fog": 0.2,
        "fog_color": np.array([0.2, 0.2, 0.3]),
        "hologram_density": 0.2,
        "light_pollution": 0.7,
        "neon_intensity": 0.6,
        "possible_transitions": ["cyber_neon_clear", "cyber_hologram_night", "cyber_blackout"],
        "scan_line_intensity": 0.3,
        "season_preference": 0.4,
        "starryness": 0.4,
        "transition_weights": [1, 0.8, 0.3],
    },

    WeatherState.CYBER_ELECTRIC_STORM: {
        "ARI": 38,
        "ambient_sound": "Rain Heavy 01 EDITED.wav",
        "celestial_visibility": 0,
        "electric_interference": 1,
        "fog": 0.4,
        "fog_color": np.array([0, 0.5, 1]),
        "glitch_probability": 0.3,
        "light_pollution": 0.4,
        "lightning_probability": 1,
        "neon_intensity": 0.3,
        "possible_transitions": ["cyber_data_storm", "cyber_acid_rain", "cyber_neon_drizzle"],
        "rain_rate": 0.5,
        "season_preference": 0.85,
        "starryness": 0,
        "transition_weights": [0.5, 0.8, 1],
    },

    WeatherState.CYBER_GLITCH_FOG: {
        "ARI": 36,
        "ambient_sound": "Tinkle Atmosphere 01.wav",
        "celestial_visibility": 0,
        "data_flow_rate": 0.4,
        "electric_interference": 0.6,
        "fog": 0.8,
        "fog_color": np.array([0.6, 0, 0.6]),
        "glitch_probability": 0.8,
        "light_pollution": 0.6,
        "neon_intensity": 0.7,
        "possible_transitions": ["cyber_data_storm", "cyber_hologram_night", "cyber_neon_drizzle"],
        "season_preference": 0.25,
        "starryness": 0,
        "transition_weights": [0.7, 0.5, 1],
    },

    WeatherState.CYBER_HOLOGRAM_NIGHT: {
        "ARI": 40,
        "ambient_sound": "Forest Cicadas EDITED.wav",
        "celestial_visibility": 0.2,
        "drone_activity": 0.5,
        "fog": 0.1,
        "fog_color": np.array([0.3, 0.1, 0.5]),
        "hologram_density": 1,
        "light_pollution": 0.9,
        "neon_intensity": 0.8,
        "possible_transitions": ["cyber_neon_clear", "cyber_drone_patrol", "cyber_glitch_fog"],
        "season_preference": 0,
        "starryness": 0.3,
        "transition_weights": [1.2, 0.8, 0.4],
    },

    WeatherState.CYBER_NEON_CLEAR: {
        "ARI": 45,
        "ambient_sound": "Forest Cicadas EDITED.wav",
        "celestial_visibility": 0.1,
        "fog": 0.1,
        "fog_color": np.array([0.2, 0.1, 0.4]),
        "hologram_density": 0.3,
        "light_pollution": 1,
        "neon_intensity": 1,
        "on_transition_events": [['pixel_spots', 60, 0]],
        "possible_transitions": ["cyber_neon_drizzle", "cyber_hologram_night", "cyber_drone_patrol", "cyber_smog_haze"],
        "season_preference": 0.2,
        "starryness": 0.2,
        "transition_weights": [1.5, 1, 0.8, 0.5],
    },

    WeatherState.CYBER_NEON_DRIZZLE: {
        "ARI": 35,
        "ambient_sound": "01 Rain Light EDITED.wav",
        "celestial_visibility": 0.05,
        "fog": 0.2,
        "fog_color": np.array([0.1, 0.3, 0.6]),
        "light_pollution": 0.8,
        "neon_intensity": 0.9,
        "on_transition_events": [['sunrise', 60, 0]],
        "possible_transitions": ["cyber_neon_clear", "cyber_acid_rain", "cyber_data_storm"],
        "rain_rate": 0.3,
        "season_preference": 0.15,
        "starryness": 0.1,
        "transition_weights": [1.5, 0.5, 0.3],
    },

    WeatherState.CYBER_SMOG_HAZE: {
        "ARI": 50,
        "ambient_sound": "Wind Strong EDITED.wav",
        "celestial_visibility": 0.1,
        "fog": 0.9,
        "fog_color": np.array([0.7, 0.5, 0.2]),
        "light_pollution": 0.6,
        "neon_intensity": 0.4,
        "pollution_level": 1,
        "possible_transitions": ["cyber_acid_rain", "cyber_blackout", "cyber_neon_clear"],
        "season_preference": 0.6,
        "starryness": 0,
        "transition_weights": [0.7, 0.3, 1.2],
    },

    WeatherState.FIREFLY: {
        "ARI": 35,
        "Sound_volume": 2,
        "Weird": 0.5,
        "Wolfy": 0.2,
        "ambient_sound": "High Desert Crickets.wav",
        "celestial_visibility": 0.8,
        "firefly_density": 1,
        "fog": 0.3,
        "meteor_rate": 0.05,
        "possible_transitions": ["clear", "foggy", "spooky", "heavy_fog", "bloom", "light_rain"],
        "season_preference": 0.3,
        "skiptime": 2,
        "spookyness": 0.05,
        "transition_weights": [1, 0.5, 0.1, 0.1, 0.25, 0.25],
        "tree_prob": 0.2,
        "wind_speed": 0.2,
    },

    WeatherState.FOGGY: {
        "ARI": 40,
        "Sound_volume": 0.6,
        "Weird": 0.5,
        "ambient_sound": "25 Swamp Ambience 2 Special Mix Light Chorus of Frogs Croa EDITED.wav",
        "celestial_visibility": 0.5,
        "firefly_density": 0.05,
        "fog": 0.7,
        "fog_color": np.array([0.3, 0.8, 0.3]),
        "possible_transitions": ["clear", "light_rain", "spooky", "firefly", "heavy_fog", "mushroom"],
        "rain_rate": 0.05,
        "season_preference": 0.15,
        "spookyness": 0.05,
        "starryness": 0.8,
        "transition_weights": [0.9, 0.1, 0.1, 0.5, 0.35, 0.1],
        "tree_prob": 0.2,
        "wind_speed": 0.1,
    },

    WeatherState.HEAVY_FOG: {
        "ARI": 26,
        "Sound_volume": 6,
        "Switch_rate": 0.75,
        "Weird": -0.3,
        "Wolfy": 0.2,
        "ambient_sound": "Tinkle Atmosphere 01.wav",
        "celestial_visibility": 0.25,
        "firefly_density": 2,
        "fog": 1.25,
        "fog_color": np.array([0.6, 0, 0.6]),
        "on_transition_events": [['fog_beings', 80, 0]],
        "possible_transitions": ["spooky", "firefly", "mushroom"],
        "rain_rate": 0,
        "season_preference": 0.175,
        "spookyness": 0,
        "starryness": 0.2,
        "transition_weights": [0.1, 0.75, 0.2],
        "tree_prob": 0.05,
        "wind_speed": 0,
    },

    WeatherState.HEAVY_RAIN: {
        "ARI": 39,
        "Sound_volume": 2,
        "ambient_sound": "Rain Heavy 01 EDITED.wav",
        "celestial_visibility": 0.3,
        "fog": 0.5,
        "fog_color": np.array([0.3, 0.3, 0.7]),
        "lightning_probability": 0.1,
        "possible_transitions": ["light_rain", "thunderstorm", "windy_night", "hurricane"],
        "rain_rate": 0.8,
        "season_preference": 0.7,
        "skiptime": 2,
        "starryness": 0.1,
        "transition_weights": [2, 1, 0.5, 0.3],
        "tree_prob": 0.2,
        "wind_speed": 0.7,
    },

    WeatherState.HURRICANE: {
        "ARI": 100,
        "Sound_volume": 3,
        "Switch_rate": 2,
        "Wolfy": 0.2,
        "ambient_sound": "Hurricane_1.mp3",
        "celestial_visibility": 0,
        "fog": 0.6,
        "fog_color": np.array([0.35, 0.42, 0.28]),
        "lightning_probability": 0.8,
        "on_transition_events": [['hurricane', 50, 0], ['lightning', 6, 0], ['lightning', 6, 5], ['lightning', 6, 12]],
        "possible_transitions": ["thunderstorm", "heavy_rain", "windy_night"],
        "rain_rate": 1,
        "season_preference": 0.95,
        "skiptime": 2,
        "spookyness": 0.15,
        "starryness": 0,
        "transition_weights": [1.5, 1, 0.5],
        "tree_prob": 0.3,
        "wind_speed": 2.2,
    },

    WeatherState.LEAVES: {
        "ARI": 25,
        "Aurora_probability": 0,
        "Sound_volume": 2,
        "Switch_rate": 1,
        "Weird": 0,
        "ambient_sound": "030822_leaves-rustling-in-wind-79518.mp3",
        "fog": 0.25,
        "meteor_rate": 0,
        "on_transition_events": [['falling_leaves', 60, 0]],
        "possible_transitions": ["clear", "windy_night", "spooky"],
        "rain_rate": 0.2,
        "season_preference": 0.7,
        "skiptime": 2,
        "transition_weights": [1, 1, 0.25],
        "tree_prob": 0,
        "wind_speed": 0.4,
    },

    WeatherState.LIGHT_RAIN: {
        "ARI": 29,
        "Sound_volume": 2,
        "ambient_sound": "01 Rain Light EDITED.wav",
        "celestial_visibility": 0.8,
        "fog": 0.1,
        "fog_color": np.array([0.2, 0.5, 0.5]),
        "possible_transitions": ["clear", "heavy_rain", "foggy", "bloom"],
        "rain_rate": 0.2,
        "season_preference": 0.125,
        "skiptime": 0,
        "starryness": 0.5,
        "transition_weights": [1, 1.2, 0.5, 0.25],
        "tree_prob": 0.2,
        "wind_speed": 0.4,
    },

    WeatherState.MUSHROOM: {
        "ARI": 22,
        "Aurora_probability": 0.5,
        "Switch_rate": 1,
        "Weird": 0,
        "ambient_sound": "Frog Croaks.wav",
        "fog": 0.5,
        "meteor_rate": 0,
        "possible_transitions": ["clear", "foggy", "heavy_fog", "leaves", "bloom"],
        "rain_rate": 0.1,
        "season_preference": 0.5,
        "transition_weights": [1, 1, 0.25, 0.5, 0.25],
        "tree_prob": 1,
        "wind_speed": 0,
    },

    WeatherState.OCEAN_ABYSS: {
        "ARI": 60,
        "ambient_sound": "Tinkle Atmosphere 01.wav",
        "bioluminescence": 0.8,
        "bubble_density": 0,
        "fog": 0.95,
        "fog_color": np.array([0.02, 0.02, 0.08]),
        "kelp_density": 0,
        "marine_life_activity": 0.1,
        "possible_transitions": ["ocean_deep_calm", "ocean_bioluminescent_swarm"],
        "season_preference": 0,
        "tide_level": 0.05,
        "transition_weights": [1, 0.5],
        "wave_amplitude": 0.05,
        "wave_speed": 0.1,
        "wind_speed": 0.05,
    },

    WeatherState.OCEAN_BIOLUMINESCENT_SWARM: {
        "ARI": 50,
        "ambient_sound": "280 Water - Processed long big ocean wave whoosh by x5.mp3",
        "bioluminescence": 1,
        "bubble_density": 0.1,
        "fog": 0.3,
        "fog_color": np.array([0.15, 0.25, 0.4]),
        "kelp_density": 0,
        "marine_life_activity": 1,
        "possible_transitions": ["ocean_deep_calm", "ocean_calm_shallows", "ocean_kelp_forest", "ocean_jellyfish_bloom"],
        "season_preference": 0.875,
        "tide_level": 0.5,
        "transition_weights": [0.8, 1, 0.6, 0.7],
        "wave_amplitude": 0.2,
        "wave_speed": 0.3,
        "wind_speed": 0.1,
    },

    WeatherState.OCEAN_CALM_SHALLOWS: {
        "ARI": 50,
        "ambient_sound": "285 Water - Natural long small ocean wave by x5.mp3",
        "bioluminescence": 0.2,
        "bubble_density": 0.1,
        "fog": 0.1,
        "fog_color": np.array([0.3, 0.6, 0.9]),
        "kelp_density": 0.3,
        "marine_life_activity": 0.7,
        "possible_transitions": ["ocean_kelp_forest", "ocean_coral_reef", "ocean_choppy_surface", "ocean_midnight_open_water"],
        "season_preference": 0.15,
        "tide_level": 0.65,
        "transition_weights": [1, 1, 0.5, 0.4],
        "wave_amplitude": 0.3,
        "wave_speed": 0.4,
        "wind_speed": 0.2,
    },

    WeatherState.OCEAN_CHOPPY_SURFACE: {
        "ARI": 45,
        "ambient_sound": "46 Waves 2 Medium Size Waves Crash & Close Out with Foam H.mp3",
        "bioluminescence": 0,
        "bubble_density": 0.8,
        "fog": 0.4,
        "fog_color": np.array([0.2, 0.4, 0.6]),
        "kelp_density": 0.1,
        "marine_life_activity": 0.2,
        "possible_transitions": ["ocean_storm_surge", "ocean_calm_shallows", "ocean_deep_calm", "ocean_maelstrom", "ocean_midnight_open_water"],
        "season_preference": 0.55,
        "tide_level": 0.75,
        "transition_weights": [0.8, 1.2, 0.5, 0.4, 0.3],
        "wave_amplitude": 0.8,
        "wave_speed": 0.6,
        "wind_speed": 0.8,
    },

    WeatherState.OCEAN_CORAL_REEF: {
        "ARI": 58,
        "ambient_sound": "underwater_turbulent.mp3",
        "bioluminescence": 0.1,
        "bubble_density": 1,
        "fog": 0.15,
        "fog_color": np.array([0.25, 0.65, 0.95]),
        "kelp_density": 0,
        "marine_life_activity": 1,
        "possible_transitions": ["ocean_calm_shallows", "ocean_kelp_forest", "ocean_choppy_surface"],
        "season_preference": 0.2,
        "tide_level": 0.7,
        "transition_weights": [1.2, 0.8, 0.3],
        "wave_amplitude": 0.25,
        "wave_speed": 0.35,
        "wind_speed": 0.2,
    },

    WeatherState.OCEAN_DEEP_CALM: {
        "ARI": 45,
        "ambient_sound": "Tinkle Atmosphere 01.wav",
        "bioluminescence": 0.6,
        "bubble_density": 0,
        "fog": 0.6,
        "fog_color": np.array([0.1, 0.2, 0.35]),
        "kelp_density": 0,
        "marine_life_activity": 0.4,
        "possible_transitions": ["ocean_bioluminescent_swarm", "ocean_abyss", "ocean_calm_shallows", "ocean_choppy_surface", "ocean_jellyfish_bloom"],
        "season_preference": 0.95,
        "tide_level": 0.35,
        "transition_weights": [0.8, 0.3, 1, 0.6, 0.5],
        "wave_amplitude": 0.2,
        "wave_speed": 0.3,
        "wind_speed": 0,
    },

    WeatherState.OCEAN_JELLYFISH_BLOOM: {
        "ARI": 45,
        "ambient_sound": "280 Water - Processed long big ocean wave whoosh by x5.mp3",
        "bioluminescence": 0.65,
        "bubble_density": 0.05,
        "fog": 0.5,
        "fog_color": np.array([0.3, 0.1, 0.45]),
        "kelp_density": 0,
        "marine_life_activity": 0.4,
        "on_transition_events": [['tentacle', 180, 0]],
        "possible_transitions": ["ocean_deep_calm", "ocean_bioluminescent_swarm", "ocean_calm_shallows"],
        "season_preference": 0.1,
        "tide_level": 0.4,
        "transition_weights": [1, 0.8, 0.5],
        "wave_amplitude": 0.15,
        "wave_speed": 0.15,
        "wind_speed": 0.1,
    },

    WeatherState.OCEAN_KELP_FOREST: {
        "ARI": 42,
        "ambient_sound": "10 Sea from cliff top.mp3",
        "bioluminescence": 0.1,
        "bubble_density": 0.2,
        "fog": 0.25,
        "fog_color": np.array([0.2, 0.5, 0.3]),
        "kelp_density": 1,
        "marine_life_activity": 0.8,
        "possible_transitions": ["ocean_calm_shallows", "ocean_coral_reef", "ocean_choppy_surface"],
        "season_preference": 0.35,
        "tide_level": 0.6,
        "transition_weights": [1, 1, 0.4],
        "wave_amplitude": 0.5,
        "wave_speed": 0.5,
        "wind_speed": 0.3,
    },

    WeatherState.OCEAN_MAELSTROM: {
        "ARI": 35,
        "ambient_sound": "46 Waves 2 Medium Size Waves Crash & Close Out with Foam H.mp3",
        "bioluminescence": 0,
        "bubble_density": 0.9,
        "fog": 0.65,
        "fog_color": np.array([0.15, 0.25, 0.35]),
        "kelp_density": 0,
        "marine_life_activity": 0,
        "on_transition_events": [['vortex', 90, 0]],
        "possible_transitions": ["ocean_storm_surge", "ocean_deep_calm"],
        "season_preference": 0.8,
        "tide_level": 0.9,
        "transition_weights": [1.2, 0.4],
        "wave_amplitude": 1.2,
        "wave_speed": 1.2,
        "wind_speed": 2,
    },

    WeatherState.OCEAN_MIDNIGHT_OPEN_WATER: {
        "ARI": 40,
        "ambient_sound": "285 Water - Natural long small ocean wave by x5.mp3",
        "bioluminescence": 0.3,
        "bubble_density": 0,
        "fog": 0.05,
        "fog_color": np.array([0.05, 0.08, 0.2]),
        "kelp_density": 0,
        "marine_life_activity": 0.1,
        "on_transition_events": [['stars', 180, 0]],
        "possible_transitions": ["ocean_calm_shallows", "ocean_choppy_surface"],
        "season_preference": 0,
        "tide_level": 0.5,
        "transition_weights": [1.2, 0.5],
        "wave_amplitude": 0.2,
        "wave_speed": 0.3,
        "wind_speed": 0.2,
    },

    WeatherState.OCEAN_STORM_SURGE: {
        "ARI": 38,
        "ambient_sound": "Rain Heavy 01 EDITED.wav",
        "bioluminescence": 0,
        "bubble_density": 1,
        "fog": 0.7,
        "fog_color": np.array([0.2, 0.35, 0.45]),
        "kelp_density": 0,
        "marine_life_activity": 0.05,
        "possible_transitions": ["ocean_choppy_surface", "ocean_deep_calm", "ocean_maelstrom"],
        "season_preference": 0.7,
        "tide_level": 0.95,
        "transition_weights": [1.5, 0.5, 0.6],
        "wave_amplitude": 1,
        "wave_speed": 1,
        "wind_speed": 1.5,
    },

    WeatherState.SANDSTORM: {
        "ARI": 30,
        "Switch_rate": 1.5,
        "ambient_sound": "26 Heavy Wind Gusts Blowing Sand EDITED.wav",
        "celestial_visibility": 0.6,
        "fog": 0.65,
        "fog_color": np.array([0.6, 0.5, 0.35]),
        "on_transition_events": [['sandstorm', 100, 0]],
        "possible_transitions": ["clear", "windy_night", "spooky", "firefly"],
        "sand_density": 1,
        "season_preference": 0.6,
        "starryness": 0.25,
        "transition_weights": [0.3, 0.5, 0.1, 0.1],
        "wind_speed": 2,
    },

    WeatherState.SPOOKY: {
        "ARI": 15,
        "Switch_rate": 1.5,
        "Wolfy": 1,
        "ambient_sound": "294 Spooky Ghostly Moans (5) EDITED.wav",
        "celestial_visibility": 1,
        "firefly_density": 0.25,
        "fog": 0.65,
        "fog_color": np.array([0.7, 0.1, 0.1]),
        "possible_transitions": ["clear", "foggy", "firefly", "heavy_fog", "windy_night"],
        "sand_density": 0.1,
        "season_preference": 0.625,
        "spookyness": 1,
        "starryness": 1,
        "transition_weights": [1, 0.1, 0.3, 0.3, 0.2],
        "wind_speed": 0.2,
    },

    WeatherState.TEST_HUE_BIN: {
        "ARI": 9999,
        "Switch_rate": 1,
        "ambient_sound": "",
        "possible_transitions": ["test_rgb"],
        "transition_weights": [1],
    },

    WeatherState.TEST_RGB: {
        "ARI": 9999,
        "Switch_rate": 1,
        "ambient_sound": "",
        "possible_transitions": ["test_hue_bin"],
        "transition_weights": [1],
    },

    WeatherState.THUNDERSTORM: {
        "ARI": 39,
        "Sound_volume": 2,
        "ambient_sound": "Rain Heavy 01 EDITED.wav",
        "celestial_visibility": 0.1,
        "fog": 0.3,
        "fog_color": np.array([0.6, 0.6, 0.2]),
        "lightning_probability": 1,
        "possible_transitions": ["heavy_rain", "light_rain", "windy_night", "hurricane"],
        "rain_rate": 1,
        "season_preference": 0.9,
        "spookyness": 0.1,
        "starryness": 0,
        "transition_weights": [2, 0.3, 0.3, 0.6],
        "wind_speed": 1,
    },

    WeatherState.VOLCANO: {
        "ARI": 65,
        "Switch_rate": 1,
        "Wolfy": 0.2,
        "ambient_sound": "Volcano Lava Fire EDITED.wav",
        "celestial_visibility": 0.8,
        "firefly_density": 1,
        "fog": 0.3,
        "fog_color": np.array([0.7, 0.7, 0.7]),
        "meteor_rate": 0.1,
        "possible_transitions": ["clear", "foggy", "spooky", "sandstorm"],
        "season_preference": 0.5,
        "transition_weights": [1, 0.5, 0.2, 0.5],
        "volcano_level": 1,
        "wind_speed": 0.7,
    },

    WeatherState.WINDY_NIGHT: {
        "ARI": 20,
        "Aurora_probability": 0.5,
        "Weird": 0.1,
        "Wolfy": 0.5,
        "ambient_sound": "Wind Strong EDITED.wav",
        "lightning_probability": 0.05,
        "meteor_rate": 0.1,
        "possible_transitions": ["clear", "heavy_rain", "sandstorm", "thunderstorm", "leaves", "hurricane"],
        "rain_rate": 0.01,
        "sand_density": 0.2,
        "season_preference": 0.8,
        "spookyness": 0.01,
        "starryness": 1,
        "transition_weights": [1, 1, 0.6, 0.4, 0.5, 0.2],
        "tree_prob": 0.2,
        "wind_speed": 1.5,
    },

}

# Weather Sets - Mutually exclusive collections of weather states
WEATHER_SETS = {
    "bartiki": {
        "allowed_parameters": ["train_speed", "train_density", "fog", "fog_color", "Switch_rate", "ARI", "possible_transitions", "transition_weights", "season_preference", "starryness", "celestial_visibility", "rain_rate", "meteor_rate", "wind_speed", "firefly_density", "ambient_sound", "lightning_probability"],
        "background_events": ["bart_map", "stars", "clouds", "fog", "rain", "city_lights", "bay_shimmer"],
        "description": "Bay Area BART system with day/night cycle — map by day, constellations by night",
        "name": "BarTiki",
        "narrative_script": "media/sounds/bartiki/script.json",
        "random_event_rate": 5e-05,
        "random_events": ["sunrise"],
        "season_extremity": 0.8,
        "season_speed": 1,
        "sound_pool_dir": "media/sounds/bart_sounds",
        "states": ["bartiki_dawn", "bartiki_morning_rush", "bartiki_midday", "bartiki_afternoon", "bartiki_evening_rush", "bartiki_dusk", "bartiki_night", "bartiki_late_night", "bartiki_stormy"],
        "transition_speed": 1,
    },

    "cosmic_night": {
        "allowed_parameters": ["wind_speed", "starryness", "celestial_visibility", "meteor_rate", "Aurora_probability", "Switch_rate", "ambient_sound", "ARI", "possible_transitions", "transition_weights", "season_preference"],
        "background_events": ["stars", "clouds"],
        "description": "Clear skies with meteors and celestial events",
        "name": "Cosmic Night",
        "narrative_script": None,
        "random_event_rate": 1.1e-05,
        "random_events": ["audio_balls", "sunrise", "fractal_fog"],
        "season_extremity": 1,
        "season_speed": 2,
        "sound_pool_dir": None,
        "states": ["clear", "windy_night", "asteroid"],
        "transition_speed": 2,
    },

    "cyberpunk": {
        "allowed_parameters": ["rain_rate", "wind_speed", "fog", "fog_color", "lightning_probability", "neon_intensity", "pollution_level", "hologram_density", "electric_interference", "data_flow_rate", "light_pollution", "drone_activity", "glitch_probability", "scan_line_intensity", "celestial_visibility", "starryness", "Switch_rate", "ambient_sound", "ARI", "possible_transitions", "transition_weights", "season_preference"],
        "background_events": ["clouds", "rain"],
        "description": "Neon-lit dystopian cityscape with digital rain and holographic advertisements",
        "name": "Cyberpunk Metropolis",
        "narrative_script": None,
        "random_event_rate": 8e-05,
        "random_events": ["game_of_life", "tunnel", "pixel_spots"],
        "season_extremity": 0.5,
        "season_speed": 0.3,
        "sound_pool_dir": None,
        "states": ["cyber_neon_clear", "cyber_neon_drizzle", "cyber_data_storm", "cyber_smog_haze", "cyber_electric_storm", "cyber_acid_rain", "cyber_hologram_night", "cyber_blackout", "cyber_glitch_fog", "cyber_drone_patrol"],
        "transition_speed": 0.8,
    },

    "desert_realm": {
        "allowed_parameters": ["wind_speed", "sand_density", "volcano_level", "fog", "fog_color", "starryness", "celestial_visibility", "firefly_density", "Switch_rate", "ambient_sound", "ARI", "possible_transitions", "transition_weights", "season_preference"],
        "background_events": ["sandstorm", "stars", "clouds", "fog"],
        "description": "Harsh desert with sandstorms and volcanic activity",
        "name": "Desert Realm",
        "narrative_script": None,
        "random_event_rate": 7e-05,
        "random_events": ["wave_terrain", "noise_isovalues", "sandstorm_event"],
        "season_extremity": 2,
        "season_speed": 0.5,
        "sound_pool_dir": None,
        "states": ["clear", "sandstorm", "volcano", "windy_night"],
        "transition_speed": 0.6,
    },

    "ethereal_mist": {
        "allowed_parameters": ["wind_speed", "rain_rate", "fog", "fog_color", "spookyness", "starryness", "celestial_visibility", "firefly_density", "Wolfy", "Switch_rate", "tree_prob", "Weird", "Sound_volume", "ambient_sound", "ARI", "possible_transitions", "transition_weights", "season_preference"],
        "background_events": ["fog", "fog_beings", "firefly", "clouds"],
        "description": "Mysterious foggy realm with supernatural elements",
        "name": "Ethereal Mist",
        "narrative_script": None,
        "random_event_rate": 8e-05,
        "random_events": ["tentacle", "fractal_fog", "voronoi_sphere", "fog_beings"],
        "season_extremity": 1.5,
        "season_speed": 0.7,
        "sound_pool_dir": None,
        "states": ["heavy_fog", "foggy", "spooky", "mushroom", "firefly"],
        "transition_speed": 0.5,
    },

    "full_spectrum": {
        "allowed_parameters": ["wind_speed", "rain_rate", "lightning_probability", "starryness", "spookyness", "fog", "fog_color", "celestial_visibility", "firefly_density", "Aurora_probability", "Wolfy", "Switch_rate", "meteor_rate", "volcano_level", "sand_density", "skiptime", "tree_prob", "Weird", "Sound_volume", "season_preference", "ambient_sound", "ARI", "possible_transitions", "transition_weights", "transition_duration"],
        "background_events": ["clouds", "firefly", "stars", "rain", "fog"],
        "description": "All weather states available - maximum variety",
        "name": "Full Spectrum",
        "narrative_script": None,
        "random_event_rate": 0.0001,
        "random_events": [],
        "season_extremity": 1,
        "season_speed": 1,
        "sound_pool_dir": None,
        "states": ["clear", "light_rain", "heavy_rain", "thunderstorm", "windy_night", "foggy", "heavy_fog", "spooky", "firefly", "sandstorm", "mushroom", "leaves", "bloom"],
        "transition_speed": 1,
    },

    "ocean": {
        "allowed_parameters": ["wind_speed", "fog", "fog_color", "wave_speed", "wave_amplitude", "bioluminescence", "tide_level", "bubble_density", "marine_life_activity", "kelp_density", "Switch_rate", "ambient_sound", "ARI", "possible_transitions", "transition_weights", "season_preference"],
        "background_events": ["Bioluminescence", "ocean_waves", "kelp", "bubbles", "fish", "fog"],
        "description": "Oceanic environment with waves and aquatic life - fog represents water clarity, season represents time of day",
        "name": "Ocean Realm",
        "narrative_script": None,
        "random_event_rate": 8e-05,
        "random_events": [],
        "season_extremity": 1,
        "season_speed": 2,
        "sound_pool_dir": None,
        "states": ["ocean_calm_shallows", "ocean_choppy_surface", "ocean_storm_surge", "ocean_deep_calm", "ocean_bioluminescent_swarm", "ocean_abyss", "ocean_kelp_forest", "ocean_coral_reef", "ocean_jellyfish_bloom", "ocean_midnight_open_water", "ocean_maelstrom"],
        "transition_speed": 0.7,
    },

    "peaceful_forest": {
        "allowed_parameters": ["wind_speed", "rain_rate", "fog", "fog_color", "starryness", "celestial_visibility", "firefly_density", "Aurora_probability", "meteor_rate", "tree_prob", "Weird", "Sound_volume", "skiptime", "ambient_sound", "ARI", "possible_transitions", "transition_weights", "season_preference"],
        "background_events": ["clouds", "firefly", "stars", "rain", "fog", "falling_leaves"],
        "description": "Gentle natural cycles with fireflies and light weather",
        "name": "Peaceful Forest",
        "narrative_script": None,
        "random_event_rate": 8e-05,
        "random_events": ["sunrise", "audio_curve", "wave_equation", "falling_leaves_event"],
        "season_extremity": 1,
        "season_speed": 1,
        "sound_pool_dir": None,
        "states": ["clear", "light_rain", "foggy", "firefly", "mushroom", "bloom", "leaves"],
        "transition_speed": 0.8,
    },

    "storm_world": {
        "allowed_parameters": ["wind_speed", "rain_rate", "lightning_probability", "fog", "fog_color", "starryness", "spookyness", "celestial_visibility", "Wolfy", "Switch_rate", "tree_prob", "Sound_volume", "skiptime", "ambient_sound", "ARI", "possible_transitions", "transition_weights", "season_preference"],
        "background_events": ["clouds", "rain", "fog", "stars", "firefly"],
        "description": "Intense weather with storms and high winds",
        "name": "Storm World",
        "narrative_script": None,
        "random_event_rate": 0.0001,
        "random_events": [],
        "season_extremity": 0.5,
        "season_speed": 1.5,
        "sound_pool_dir": None,
        "states": ["windy_night", "heavy_rain", "thunderstorm", "hurricane", "foggy", "light_rain", "sandstorm", "leaves", "firefly"],
        "transition_speed": 1.5,
    },

    "test": {
        "allowed_parameters": [],
        "background_events": ["test_pattern"],
        "description": "test system",
        "name": "test",
        "narrative_script": None,
        "season_extremity": 1,
        "season_speed": 1,
        "sound_pool_dir": None,
        "states": ["test_rgb", "test_hue_bin"],
        "transition_speed": 1,
    },

}

DEFAULT_WEATHER_SET = "bartiki"



def _validate_parameter_definitions():
    """Sanity-check that every parameter referenced by a weather set or
    preset has a PARAMETER_DEFINITIONS entry.

    Missing entries cause the web weather editor to silently skip the
    parameter (see the `if (!paramDef) continue;` in weather_editor.html),
    so even though the parameter still affects rendering, the user can't
    see or change its value. Surfacing the problem at import time turns a
    "why can't I edit this" mystery into an obvious warning.
    """
    import sys

    known = set(PARAMETER_DEFINITIONS.keys())

    missing_in_sets = {}
    for set_name, set_data in WEATHER_SETS.items():
        for param in set_data.get("allowed_parameters", []):
            if param not in known:
                missing_in_sets.setdefault(param, []).append(set_name)

    missing_in_presets = {}
    for state, preset in WEATHER_PRESETS.items():
        for param in preset.keys():
            if param not in known:
                missing_in_presets.setdefault(param, []).append(state.value)

    if not missing_in_sets and not missing_in_presets:
        return

    bar = "=" * 72
    lines = [
        "",
        bar,
        "[weather_params] parameters missing from PARAMETER_DEFINITIONS",
        "These will be silently skipped by the web weather editor.",
        "Add an entry in PARAMETER_DEFINITIONS for each one.",
        bar,
    ]
    for param in sorted(missing_in_sets):
        sets = ", ".join(sorted(missing_in_sets[param]))
        lines.append(f"  [set]    {param}  (in allowed_parameters of: {sets})")
    for param in sorted(missing_in_presets):
        states = ", ".join(sorted(missing_in_presets[param]))
        lines.append(f"  [preset] {param}  (in states: {states})")
    lines.append(bar)
    print("\n".join(lines), file=sys.stderr)


_validate_parameter_definitions()
