"""
Shader effects package - import effects here for easy access
"""
from .base import ShaderEffect
from .rain import shader_rain, RainEffect

# As you add more effects:
# from .starfield import shader_starfield, StarfieldEffect
# from .snow import shader_snow, SnowEffect

__all__ = [
    'ShaderEffect',
    'shader_rain',
    'RainEffect',
    # Add new effects here
]