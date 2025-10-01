"""
Shader effects package - import effects here for easy access
"""
from .base import ShaderEffect
from .rain import shader_rain, RainEffect
from .test_circle import shader_test_circle
from .firefly import shader_firefly, FireflyEffect
from .celestial_bodies import shader_celestial_bodies, CelestialBodiesEffect
from .stars import shader_stars, TwinklingStarsEffect

# As you add more effects:
# from .starfield import shader_starfield, StarfieldEffect
# from .snow import shader_snow, SnowEffect

__all__ = [
    'ShaderEffect',
    'shader_rain',
    'RainEffect',
    'shader_test_circle',
    'shader_firefly',
    'shader_celestial_bodies',
    'shader_stars',
    # Add new effects here
]