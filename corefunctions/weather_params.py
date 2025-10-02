import numpy as np
from enum import Enum

class WeatherState(Enum):
    CLEAR = "clear"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"
    THUNDERSTORM = "thunderstorm"
    WINDY_NIGHT = "windy_night"
    FOGGY = "foggy"
    HEAVY_FOG = 'heavy_fog'
    SPOOKY = "spooky"
    FIREFLY = "firefly"
    VOLCANO = "volcano"
    SANDSTORM = "sandstorm"
    ASTEROID = "asteroid"
    MUSHROOM = "mushroom"
    FALLING_LEAVES="leaves"
    SUMMER_BLOOM="bloom"

# Default weather parameters
DEFAULT_WEATHER_PARAMS = {
    "wind_speed": 0,
    "rain_rate": 0,
    "lightning_probability": 0,
    "starryness": 1.0,
    "spookyness": 0.0,
    "fog": 0.0,
    "fog_color": np.array([0.7, 0.7, 0.7]),
    "possible_transitions": ["light_rain", "foggy", "windy_night"],
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
    "Sound_volume":1.0,
    "season_preference": 0.375,  # Default to summer
    "mountain":0
}

# Weather presets
# Weather state parameters
WEATHER_PRESETS = {
    WeatherState.CLEAR: {
        "wind_speed": 0.2,
        "ambient_sound": "Forest Cicadas EDITED.wav",
        "ARI": 40,
        "possible_transitions": [
            "light_rain",
            "foggy",
            "windy_night",
            "firefly",
            "volcano",
            "mushroom",
            "leaves",
            "bloom"
        ],
        "transition_weights": [1.0, 1.0, 0.75, 0.5, 0.1,0.2,0.75,0.75],
        "Aurora_probability": 0.5,
        "meteor_rate": 0.25,
        "tree_prob": 1,
        "Weird": 1,
        "Switch_rate": 0.9,
        "season_preference": 0.375,  # Summer
        "mountain":0.1
    },

    WeatherState.FALLING_LEAVES: {
        "wind_speed": 0.4,
        "rain_rate": 0.2,
        "fog": 0.25,
        "ambient_sound": "030822_leaves-rustling-in-wind-79518.mp3",
        "ARI": 25,
        "Sound_volume":2.0,
        "skiptime": 2.0,
        "possible_transitions": ["clear",'windy_night','spooky'],
        "transition_weights": [1.0,1.0,0.25],
        "Aurora_probability": 0.0,
        "meteor_rate": 0.0,
        "tree_prob": 0,
        "Weird": 0,
        "Switch_rate": 1,
        "mountain":0.1,
        "season_preference": 0.7,  # Summer
    },
    WeatherState.SUMMER_BLOOM: {
        "wind_speed": 0.2,
        "rain_rate": 0.0,
        "fog": 0.0,
        "ambient_sound": "09 Nightingale.mp3",
        "ARI": 45,
        "Sound_volume":1.0,
        "skiptime": 0.0,
        "possible_transitions": ["clear",'windy_night','leaves',"firefly","mushroom"],
        "transition_weights": [1.0,0.25,1.0,1.0,0.5],
        "Aurora_probability": 0.0,
        "meteor_rate": 0.05,
        "tree_prob": 0.1,
        "Weird": 0.0,
        "firefly_density": 0.5,
        "Switch_rate": 1.0,
        "season_preference": 0.5,  # late Summer
    },
    WeatherState.MUSHROOM: {
        "wind_speed": 0.0,
        "rain_rate": 0.1,
        "fog": 0.5,
        "ambient_sound": "Frog Croaks.wav",
        "ARI": 22,
        "possible_transitions": ["clear","foggy","heavy_fog","leaves","bloom"],
        "transition_weights": [1.0,1.0,0.25,0.5,0.25],
        "Aurora_probability": 0.5,
        "meteor_rate": 0.0,
        "tree_prob": 1,
        "Weird": 0,
        "Switch_rate": 1,
        "season_preference": 0.5,  # Summer
        "mountain":0.1
    },
    WeatherState.LIGHT_RAIN: {
        "wind_speed": 0.4,
        "rain_rate": 0.2,
        "ambient_sound": "01 Rain Light EDITED.wav",
        "ARI": 29,
        "skiptime": 0,
        "Sound_volume":2.0,
        "starryness": 0.5,
        "fog": 0.1,
        "fog_color": np.array([0.2, 0.5, 0.5]),
        "possible_transitions": ["clear", "heavy_rain", "foggy","bloom"],
        "transition_weights": [1.0, 1.2, 0.5,0.25],
        "celestial_visibility": 0.8,
        "tree_prob": 0.2,
        "mountain":0.2,
        "season_preference": 0.125  # Spring
    },
    WeatherState.HEAVY_RAIN: {
        "wind_speed": 0.7,
        "rain_rate": 0.8,
        "lightning_probability": 0.1,
        "ambient_sound": "Rain Heavy 01 EDITED.wav",
        "ARI": 39,
        "Sound_volume":2.0,
        "skiptime": 2.0,
        "starryness": 0.1,
        "fog": 0.5,
        "fog_color": np.array([0.3, 0.3, 0.7]),
        "possible_transitions": ["light_rain", "thunderstorm","windy_night"],
        "transition_weights": [2.0, 1.0,0.5],
        "celestial_visibility": 0.3,
        "mountain":1.0,
        "tree_prob": 0.2,
        "season_preference": 0.7  # Fall
    },
    WeatherState.THUNDERSTORM: {
        "wind_speed": 1.0,
        "rain_rate": 1.0,
        "lightning_probability": 1,
        "ambient_sound": "Rain Heavy 01 EDITED.wav",
        "ARI": 39,
        "Sound_volume":2.0,
        "starryness": 0.0,
        "spookyness": 0.1,
        "fog": 0.3,
        "fog_color": np.array([0.6, 0.6, 0.2]),
        "possible_transitions": ["heavy_rain", "light_rain","windy_night"],
        "transition_weights": [2.0, 0.3,0.3],
        "celestial_visibility": 0.1,
        "mountain":.5,
        "season_preference": 0.9  # Fall
    },
    WeatherState.WINDY_NIGHT: {
        "wind_speed": 1.5,
        "rain_rate": 0.01,
        "lightning_probability": 0.05,
        "ambient_sound": "Wind Strong EDITED.wav",
        "ARI": 20,
        "starryness": 1.0,
        "spookyness": 0.01,
        "possible_transitions": ["clear", "heavy_rain", "sandstorm","thunderstorm","leaves"],
        "transition_weights": [1.0, 1.0, 0.6,0.4,0.5],
        "Aurora_probability": 0.5,
        "Wolfy": 0.5,
        "meteor_rate": 0.1,
        "sand_density": 0.2,
        "Weird": 0.1,
        "tree_prob": 0.2,
        "mountain":0.3,
        "season_preference": 0.80  # Winter
    },
    WeatherState.FOGGY: {
        "wind_speed": 0.1,
        "rain_rate": 0.05,
        "ambient_sound": "25 Swamp Ambience 2 Special Mix Light Chorus of Frogs Croa EDITED.wav",
        "ARI": 40,
        "starryness": 0.8,
        "fog": 0.7,
        "fog_color": np.array([0.3, 0.8, 0.3]),
        "spookyness": 0.05,
        "possible_transitions": ["clear", "light_rain", "spooky","firefly","heavy_fog","mushroom"],
        "transition_weights": [0.9, 0.1, 0.1,0.5,0.35,0.1],
        "celestial_visibility": 0.5,
        "firefly_density": 0.05,
        "Weird": 0.5,
        "tree_prob": 0.2,
        "Sound_volume":0.6,
        "season_preference": 0.15  # Spring
    },
    WeatherState.HEAVY_FOG: {
        "wind_speed": 0.0,
        "rain_rate": 0.0,
        "ambient_sound": "Tinkle Atmosphere 01.wav",
        "ARI": 26,
        "starryness": 0.2,
        "fog": 1.25,
        "fog_color": np.array([0.6, 0.0, 0.6]),
        "spookyness": 0.0,
        "possible_transitions": ["spooky","firefly","mushroom"],
        "transition_weights": [0.1, 0.75,0.2],
        "celestial_visibility": 0.25,
        "firefly_density": 2.0,
        "Weird": -0.3,
        "tree_prob": 0.05,
        "Switch_rate":0.75,
        "Sound_volume":6,
        "Wolfy": 0.2,
        "season_preference": 0.175  # Spring
    },
    WeatherState.SPOOKY: {
        "wind_speed": 0.2,
        "ambient_sound": "294 Spooky Ghostly Moans (5) EDITED.wav",
        "ARI": 15,
        "starryness": 1,
        "fog": 0.65,
        "fog_color": np.array([0.7, 0.1, 0.1]),
        "spookyness": 1.0,
        "possible_transitions": ["clear", "foggy", "firefly","heavy_fog","windy_night"],
        "transition_weights": [1.0, 0.1, 0.3,0.3,0.2],
        "celestial_visibility": 1.0,
        "firefly_density": 0.25,
        "Wolfy": 1.0,
        "sand_density": 0.1,
        "Switch_rate": 1.5,
        "season_preference": 0.625  # Winter
    },
    WeatherState.FIREFLY: {
        "wind_speed": 0.2,
        "ambient_sound": "High Desert Crickets.wav",
        "ARI": 35,
        "fog": 0.3,
        "spookyness": 0.05,
        "possible_transitions": ["clear", "foggy", "spooky","heavy_fog","bloom"],
        "transition_weights": [1.0, 0.5, 0.1,0.1,0.25],
        "celestial_visibility": 0.80,
        "firefly_density": 1,
        "Wolfy": 0.2,
        "tree_prob": 0.2,
        "Weird":0.5,
        "meteor_rate": 0.05,
        "Sound_volume":2,
        "skiptime": 2.0,
        "season_preference": 0.3 # Summer
    },
    WeatherState.VOLCANO: {
        "wind_speed": 0.7,
        "ambient_sound": "Volcano Lava Fire EDITED.wav",
        "ARI": 65,
        "fog": 0.3,
        "fog_color": np.array([0.7, 0.7, 0.7]),
        "possible_transitions": ["clear", "foggy", "spooky", "sandstorm"],
        "transition_weights": [1.0, 0.5, 0.2, 0.5],
        "celestial_visibility": 0.80,
        "firefly_density": 1,
        "Wolfy": 0.2,
        "volcano_level": 1.0,
        "Switch_rate": 1,
        "meteor_rate": 0.1,
        "season_preference": 0.5 # Summer
    },
    WeatherState.SANDSTORM: {
        "wind_speed": 2.0,
        "ambient_sound": "26 Heavy Wind Gusts Blowing Sand EDITED.wav",
        "ARI": 30,
        "starryness": 0.25,
        "fog": 0.45,
        "fog_color": np.array([0.2, 0.1, 0.0]),  # Sandy colored fog
        "possible_transitions": ["clear", "windy_night", "spooky"],
        "transition_weights": [0.3, 0.5, 0.1],
        "celestial_visibility": 0.6,
        "sand_density": 1.0,
        "mountain":0.1,
        "season_preference": 0.6,  # Fall
        "Switch_rate": 1.5,
    },
    WeatherState.ASTEROID: {
        "ambient_sound": "toot.toot",
        "ARI": 40,
        "possible_transitions": ["clear"],
        "transition_weights": [1.0],
        "celestial_visibility": 1.0,
        "meteor_rate": 0.2,
        "starryness": 1.0,
        "Switch_rate": 3,
        "season_preference": 0.875 # Winter
    },
}
