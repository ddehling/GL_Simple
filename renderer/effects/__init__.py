"""
Shader effects package - automatically imports all effects
"""
import importlib
import pkgutil
from pathlib import Path

# Import base class explicitly
from .base import ShaderEffect

# Automatically discover and import all modules in this package
_package_path = Path(__file__).parent
_module_names = [
    name for _, name, is_pkg in pkgutil.iter_modules([str(_package_path)])
    if not is_pkg and name != 'base'  # Skip base module and subpackages
]

# Import all discovered modules and collect their exports
__all__ = ['ShaderEffect']
_shader_functions = {}
_effect_classes = {}

for module_name in _module_names:
    module = importlib.import_module(f'.{module_name}', package=__name__)
    
    # Collect functions starting with 'shader_'
    for attr_name in dir(module):
        if attr_name.startswith('shader_'):
            func = getattr(module, attr_name)
            globals()[attr_name] = func
            __all__.append(attr_name)
            _shader_functions[attr_name] = func
        # Collect classes ending with 'Effect' (excluding base ShaderEffect)
        elif attr_name.endswith('Effect') and attr_name != 'ShaderEffect':
            cls = getattr(module, attr_name)
            globals()[attr_name] = cls
            __all__.append(attr_name)
            _effect_classes[attr_name] = cls

print(f"Auto-loaded {len(_shader_functions)} shader functions and {len(_effect_classes)} effect classes")