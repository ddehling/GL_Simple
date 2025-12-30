# Parameter Management System - Quick Guide

## Overview

The weather set editor now includes a comprehensive parameter management system that allows you to control which parameters are available for editing in each weather set. This prevents interface clutter and ensures each set only shows relevant parameters.

## Key Features

### 1. **Allowed Parameters Per Set**
Each weather set now has an `allowed_parameters` list that defines which parameters can be edited for weather states in that set.

**Example:**
```python
"cosmic_night": {
    "allowed_parameters": ["wind_speed", "starryness", "meteor_rate", "Switch_rate", ...],
    "states": ["clear", "asteroid", "windy_night"],
    ...
}
```

### 2. **Parameter Filtering**
When editing weather states, you'll only see the parameters that are allowed for the current set. This keeps the interface clean and focused.

### 3. **Dynamic Parameter Management**
You can add or remove parameters from a set's allowed list at any time through the UI.

### 4. **Custom Parameter Creation**
Create new parameters globally that can then be added to any set's allowed list.

## Using the Parameter System

### Viewing Allowed Parameters
1. Select a weather set from the left panel
2. The "Allowed Parameters for This Set" section shows all parameters that can be edited
3. Each parameter appears as a blue pill with an × button to remove it

### Adding Parameters to a Set
1. Click "+ Add Parameter" button in the "Allowed Parameters" section
2. Select from existing parameters in the dropdown
3. OR click "+ Create New Parameter" to create a custom parameter
4. The parameter is immediately added to the set's allowed list

### Removing Parameters from a Set
1. Hover over a parameter pill
2. Click the × button
3. The parameter is removed from the set (but still exists globally)

### Creating New Parameters
1. Click "+ Add Parameter" → "+ Create New Parameter"
2. Enter:
   - **Name**: lowercase_snake_case (e.g., `my_custom_param`)
   - **Type**: Number, Text, Array, Array of Strings, or Array of Numbers
   - **Default Value**: Initial value for the parameter
3. Click "Create"
4. The new parameter is now available to add to any set

### Editing Weather State Parameters
1. Select a weather set
2. Click on a weather state tab
3. Only parameters from the allowed list will be shown
4. Edit values as needed
5. All changes are tracked and saved when you click "Save Changes"

## Parameter Types

- **Number**: Single numeric value (e.g., `wind_speed`, `rain_rate`)
- **Text**: String value (e.g., `ambient_sound` filename)
- **Array**: 3-element RGB array (e.g., `fog_color`)
- **Array of Strings**: Comma-separated strings (e.g., `possible_transitions`)
- **Array of Numbers**: Comma-separated numbers (e.g., `transition_weights`)

## Default Parameters

All sets start with a curated list of allowed parameters:

- **Cosmic Night**: Celestial-focused (stars, meteors, aurora)
- **Desert Realm**: Environmental (sand, volcano, wind, fog)
- **Ethereal Mist**: Atmospheric (fog, fireflies, spooky elements)
- **Peaceful Forest**: Natural cycles (rain, fog, fireflies, trees)
- **Storm World**: Weather-intense (wind, rain, lightning, fog)
- **Full Spectrum**: All parameters available

## Benefits

✅ **Cleaner Interface**: Only see relevant parameters for each set
✅ **Prevent Errors**: Can't accidentally set irrelevant parameters
✅ **Organized Workflow**: Each set has its own focused parameter set
✅ **Extensible**: Easy to add new custom parameters as needed
✅ **Flexible**: Move parameters between sets or create set-specific lists

## Workflow Example

**Creating a "Desert Night" Set:**

1. Create new set with ID "desert_night"
2. Add weather states: "clear", "sandstorm", "windy_night"
3. Add allowed parameters:
   - `wind_speed` - for sandstorms
   - `sand_density` - control sand intensity
   - `starryness` - desert night sky
   - `celestial_visibility` - clear desert air
   - `ambient_sound` - desert sounds
   - `ARI` - for audio routing
   - `possible_transitions`, `transition_weights` - for state changes
4. Edit each weather state using only these parameters
5. Save and test

## Tips

- Start with a similar existing set and modify its allowed parameters
- Use Full Spectrum set as a reference for all available parameters
- Create custom parameters for unique effects specific to your installation
- Remove unused parameters to keep the interface clean
- Always validate before saving to catch any issues

## Technical Notes

- Parameters not in the allowed list are ignored when the set is active
- Removing a parameter from allowed list doesn't delete values from weather states
- Adding a parameter to allowed list immediately makes it visible in the editor
- Custom parameters are added to DEFAULT_WEATHER_PARAMS when saved
- The system maintains backward compatibility with existing weather_params.py files
