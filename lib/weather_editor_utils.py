"""
Utility for saving weather editor changes back to weather_params.py
Handles file writing while preserving Python syntax and formatting.
"""
import os
from pathlib import Path
import ast
import numpy as np


def validate_weather_params(weather_states, weather_presets, weather_sets):
    """
    Validate weather parameters before saving.
    
    Returns:
        dict: {"valid": bool, "errors": list}
    """
    errors = []
    
    # Validate weather states
    if not weather_states or not isinstance(weather_states, list):
        errors.append("weather_states must be a non-empty list")
    
    for state in weather_states:
        if not isinstance(state, str) or not state:
            errors.append(f"Invalid weather state: {state}")
    
    # Validate weather presets
    if not isinstance(weather_presets, dict):
        errors.append("weather_presets must be a dictionary")
    else:
        for state, params in weather_presets.items():
            if state not in weather_states:
                errors.append(f"Preset '{state}' not in weather_states")
            if not isinstance(params, dict):
                errors.append(f"Preset '{state}' must be a dictionary")
    
    # Validate weather sets
    if not isinstance(weather_sets, dict):
        errors.append("weather_sets must be a dictionary")
    else:
        for set_id, set_data in weather_sets.items():
            if not isinstance(set_data, dict):
                errors.append(f"Set '{set_id}' must be a dictionary")
                continue
            
            if 'states' not in set_data:
                errors.append(f"Set '{set_id}' missing 'states' field")
            elif not isinstance(set_data['states'], list):
                errors.append(f"Set '{set_id}' states must be a list")
            else:
                for state in set_data['states']:
                    if state not in weather_states:
                        errors.append(f"Set '{set_id}' references unknown state '{state}'")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def save_weather_params(weather_states, weather_presets, weather_sets, global_parameters=None):
    """
    Save weather parameters back to weather_params.py file.
    
    Args:
        weather_states: List of weather state strings
        weather_presets: Dictionary of weather state -> parameters
        weather_sets: Dictionary of set_id -> set configuration
        global_parameters: List of parameter names that are global (optional)
    
    Returns:
        dict: {"success": bool, "message": str, "error": str (optional)}
    """
    try:
        # Default global parameters if not provided
        if global_parameters is None:
            global_parameters = ["possible_transitions", "transition_weights", "transition_duration", "Sound_volume"]
        
        # Validate first
        validation = validate_weather_params(weather_states, weather_presets, weather_sets)
        if not validation["valid"]:
            return {
                "success": False,
                "error": "Validation failed: " + "; ".join(validation["errors"])
            }
        
        # Get the path to weather_params.py (now lives in lib/)
        current_dir = Path(__file__).parent.parent
        weather_params_path = current_dir / "lib" / "weather_params.py"
        
        # Create backup
        backup_path = weather_params_path.with_suffix('.py.backup')
        if weather_params_path.exists():
            import shutil
            shutil.copy2(weather_params_path, backup_path)
        
        # Generate the Python file content
        content = generate_weather_params_file(weather_states, weather_presets, weather_sets, global_parameters)
        
        # Write to file
        with open(weather_params_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "success": True,
            "message": f"Successfully saved to {weather_params_path}. Backup created at {backup_path}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error saving file: {str(e)}"
        }


def generate_weather_params_file(weather_states, weather_presets, weather_sets, global_parameters=None):
    """
    Generate the complete weather_params.py file content.
    
    Args:
        global_parameters: List of parameter names that should be global
    
    Returns:
        str: Python file content
    """
    if global_parameters is None:
        global_parameters = ["possible_transitions", "transition_weights", "transition_duration", "Sound_volume"]
    
    lines = []
    
    # Header
    lines.append("import numpy as np")
    lines.append("from enum import Enum")
    lines.append("")
    
    # WeatherState enum
    lines.append("class WeatherState(Enum):")
    for state in weather_states:
        # Convert to UPPER_SNAKE_CASE for enum name
        enum_name = state.upper().replace('-', '_')
        lines.append(f'    {enum_name} = "{state}"')
    lines.append("")
    
    # Global parameters
    lines.append("# Global parameters that are always available in every weather set")
    lines.append("# These cannot be removed from sets but their values can be customized per weather state")
    lines.append("GLOBAL_PARAMETERS = [")
    for param in global_parameters:
        lines.append(f'    "{param}",')
    lines.append("]")
    lines.append("")
    
    # Available background events
    lines.append("# Available background events (always-active effects)")
    lines.append("# This is the single source of truth for which events can be used as continuous background effects.")
    lines.append("# Add new background-capable events here. They must also exist in Stories_OGL.py's event_map.")
    lines.append("AVAILABLE_BACKGROUND_EVENTS = [")
    background_events = [
        'clouds', 'firefly', 'stars', 'rain', 'fog', 'sandstorm', 'fog_beings', 'falling_leaves'
    ]
    for event in background_events:
        lines.append(f"    '{event}',")
    lines.append("]")
    lines.append("")
    
    # Parameter definitions
    lines.append("# Parameter definitions for the weather editor")
    lines.append("# Defines the type and input configuration for each parameter")
    lines.append("PARAMETER_DEFINITIONS = {")
    param_defs = {
        'wind_speed': {'type': 'number', 'step': 0.1},
        'rain_rate': {'type': 'number', 'step': 0.1},
        'lightning_probability': {'type': 'number', 'step': 0.05},
        'starryness': {'type': 'number', 'step': 0.1},
        'spookyness': {'type': 'number', 'step': 0.1},
        'fog': {'type': 'number', 'step': 0.05},
        'fog_color': {'type': 'array', 'length': 3},
        'celestial_visibility': {'type': 'number', 'step': 0.1},
        'firefly_density': {'type': 'number', 'step': 0.1},
        'Aurora_probability': {'type': 'number', 'step': 0.1},
        'Wolfy': {'type': 'number', 'step': 0.1},
        'Switch_rate': {'type': 'number', 'step': 0.1},
        'meteor_rate': {'type': 'number', 'step': 0.05},
        'volcano_level': {'type': 'number', 'step': 0.1},
        'sand_density': {'type': 'number', 'step': 0.1},
        'skiptime': {'type': 'number', 'step': 0.5},
        'tree_prob': {'type': 'number', 'step': 0.1},
        'Weird': {'type': 'number', 'step': 0.1},
        'Sound_volume': {'type': 'number', 'step': 0.1},
        'season_preference': {'type': 'number', 'step': 0.025},
        'mountain': {'type': 'number', 'step': 0.1},
        'ambient_sound': {'type': 'text'},
        'ARI': {'type': 'number', 'step': 1},
        'transition_duration': {'type': 'number', 'step': 1},
        'possible_transitions': {'type': 'array-string'},
        'transition_weights': {'type': 'array-number'},
        'on_transition_events': {'type': 'event-list'},
        'neon_intensity': {'type': 'number', 'step': 0.05},
        'pollution_level': {'type': 'number', 'step': 0.05},
        'hologram_density': {'type': 'number', 'step': 0.05},
        'electric_interference': {'type': 'number', 'step': 0.05},
        'data_flow_rate': {'type': 'number', 'step': 0.05},
        'light_pollution': {'type': 'number', 'step': 0.05},
        'drone_activity': {'type': 'number', 'step': 0.05},
        'glitch_probability': {'type': 'number', 'step': 0.05},
        'scan_line_intensity': {'type': 'number', 'step': 0.05}
    }
    for param_name, param_def in sorted(param_defs.items()):
        lines.append(f"    '{param_name}': {repr(param_def)},")
    lines.append("}")
    lines.append("")
    
    # DEFAULT_WEATHER_PARAMS (keep static for now)
    lines.append("# Default weather parameters")
    lines.append("DEFAULT_WEATHER_PARAMS = {")
    default_params = {
        "wind_speed": 0,
        "rain_rate": 0,
        "lightning_probability": 0,
        "starryness": 1.0,
        "spookyness": 0.0,
        "fog": 0.0,
        "fog_color": "np.array([0.7, 0.7, 0.7])",
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
        "Sound_volume": 1.0,
        "season_preference": 0.375,
        "mountain": 0,
        "ambient_sound": None,
        "ARI": 0.0
    }
    
    for key, value in default_params.items():
        if isinstance(value, str):
            lines.append(f'    "{key}": {value},')
        elif isinstance(value, list):
            lines.append(f'    "{key}": {repr(value)},')
        else:
            lines.append(f'    "{key}": {value},')
    lines.append("}")
    lines.append("")
    
    # WEATHER_PRESETS
    lines.append("# Weather presets")
    lines.append("# Weather state parameters")
    lines.append("WEATHER_PRESETS = {")
    
    for state, params in weather_presets.items():
        # Find the enum name for this state
        enum_name = state.upper().replace('-', '_')
        lines.append(f"    WeatherState.{enum_name}: {{")
        
        for key, value in sorted(params.items()):
            formatted_value = format_python_value(key, value)
            lines.append(f'        "{key}": {formatted_value},')
        
        lines.append("    },")
        lines.append("")
    
    lines.append("}")
    lines.append("")
    
    # WEATHER_SETS
    lines.append("# Weather Sets - Mutually exclusive collections of weather states")
    lines.append("WEATHER_SETS = {")
    
    for set_id, set_data in weather_sets.items():
        lines.append(f'    "{set_id}": {{')
        
        for key, value in sorted(set_data.items()):
            formatted_value = format_python_value(key, value)
            lines.append(f'        "{key}": {formatted_value},')
        
        lines.append("    },")
        lines.append("")
    
    lines.append("}")
    lines.append("")
    
    # DEFAULT_WEATHER_SET
    if weather_sets:
        first_set = list(weather_sets.keys())[0]
        lines.append(f'DEFAULT_WEATHER_SET = "{first_set}"')
    else:
        lines.append('DEFAULT_WEATHER_SET = "full_spectrum"')
    lines.append("")
    
    return '\n'.join(lines)


def format_python_value(key, value):
    """
    Format a value for Python code.
    Handles special cases like numpy arrays, lists, etc.
    """
    # Handle None
    if value is None:
        return "None"
    
    # Handle numpy-specific array for fog_color
    if key == "fog_color" and isinstance(value, (list, tuple)):
        return f"np.array({list(value)})"
    
    # Handle strings
    if isinstance(value, str):
        return f'"{value}"'
    
    # Handle lists
    if isinstance(value, list):
        # Check if it's a list of strings
        if all(isinstance(x, str) for x in value):
            formatted_items = [f'"{x}"' for x in value]
            return f"[{', '.join(formatted_items)}]"
        else:
            return repr(value)
    
    # Handle numbers
    if isinstance(value, (int, float)):
        return str(value)
    
    # Handle booleans
    if isinstance(value, bool):
        return str(value)
    
    # Default: use repr
    return repr(value)


def get_current_weather_params():
    """
    Load current weather parameters from weather_params.py
    
    Returns:
        dict: {"weather_states": list, "weather_presets": dict, "weather_sets": dict}
    """
    try:
        from lib.weather_params import (
            WeatherState, WEATHER_PRESETS, WEATHER_SETS
        )
        
        # Convert enum to list of strings
        weather_states = [state.value for state in WeatherState]
        
        # Convert WEATHER_PRESETS (with enum keys) to regular dict
        weather_presets = {}
        for state, params in WEATHER_PRESETS.items():
            state_key = state.value if hasattr(state, 'value') else str(state)
            params_copy = params.copy()
            
            # Convert numpy arrays to lists
            if 'fog_color' in params_copy:
                if hasattr(params_copy['fog_color'], 'tolist'):
                    params_copy['fog_color'] = params_copy['fog_color'].tolist()
            
            weather_presets[state_key] = params_copy
        
        return {
            "weather_states": weather_states,
            "weather_presets": weather_presets,
            "weather_sets": WEATHER_SETS
        }
        
    except Exception as e:
        raise Exception(f"Error loading current weather params: {str(e)}")


if __name__ == "__main__":
    # Test the utility
    print("Testing weather_editor_utils...")
    
    # Load current data
    data = get_current_weather_params()
    print(f"Loaded {len(data['weather_states'])} weather states")
    print(f"Loaded {len(data['weather_presets'])} weather presets")
    print(f"Loaded {len(data['weather_sets'])} weather sets")
    
    # Validate
    validation = validate_weather_params(
        data['weather_states'],
        data['weather_presets'],
        data['weather_sets']
    )
    print(f"Validation: {validation}")
    
    # Generate file content (but don't save)
    content = generate_weather_params_file(
        data['weather_states'],
        data['weather_presets'],
        data['weather_sets']
    )
    print(f"\nGenerated {len(content)} characters of Python code")
    print("\nFirst 500 characters:")
    print(content[:500])
