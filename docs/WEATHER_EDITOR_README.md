# Weather Set Editor

## Overview

The Weather Set Editor is a comprehensive visual interface for designing weather sets and weather states for the GL_Simple environmental effects system. It provides a mouse-driven, intuitive interface for creating, editing, and managing all weather-related configurations.

## Accessing the Editor

1. Start your GL_Simple application (`Stories_OGL.py`)
2. Navigate to: `http://localhost:5000/weather_editor`
3. Or use mDNS: `http://glsimple.local:5000/weather_editor`

## Features

### 🎨 Visual Interface
- Clean, modern design with intuitive controls
- Real-time preview of changes
- Organized layout with left panel for sets and right panel for editing
- Tabbed interface for editing individual weather states

### 📦 Weather Set Management
- **Create New Sets**: Click "+ New Set" to create a custom weather set
- **Edit Existing Sets**: Click any set in the left panel to edit
- **Delete Sets**: Hover over a set and click the × button
- **Configure Set Properties**:
  - Display Name
  - Description
  - Season Speed (time multiplier)
  - Season Extremity (seasonal influence)
  - Transition Speed (change frequency)

### 🌦️ Weather State Management
- **Add Weather to Sets**: Click "+ Add Weather" to add existing states to a set
- **Remove from Sets**: Click × on any weather pill
- **Create New Weather States**: Click "+ New Weather State" button
- **Delete Weather States**: Opens the weather state editor and click delete button
- **Edit Weather Parameters**: All weather state parameters are editable

### ⚙️ Parameter Editing
Each weather state has numerous configurable parameters:
- **Numeric Values**: wind_speed, rain_rate, fog, etc.
- **Text Fields**: ambient_sound file names
- **Arrays**: fog_color (RGB values)
- **Lists**: possible_transitions, transition_weights

### 💾 Save & Validation
- **Validate**: Click "Validate" to check data integrity without saving
- **Save Changes**: Overwrites `weather_params.py` (creates backup first)
- **Unsaved Changes Warning**: Browser warns before closing with unsaved work
- **Automatic Backup**: Original file saved as `weather_params.py.backup`

## Workflow

### Creating a New Weather Set

1. Click "+ New Set" button
2. Enter a unique ID (lowercase, underscores, no spaces)
3. Enter display name and description
4. Click "Create"
5. Add weather states using "+ Add Weather" button
6. Configure set parameters (season speed, etc.)
7. Click "Validate" to check for errors
8. Click "Save Changes" when ready

### Editing Weather State Parameters

1. Select a weather set from the left panel
2. Click on a weather state tab at the bottom
3. Edit parameters in the grid
4. Changes are tracked automatically
5. Click "Save Changes" to persist

### Creating a New Weather State

1. Click "+ New Weather State" button
2. Enter state name (UPPERCASE_SNAKE_CASE, e.g., "MYSTIC_RAIN")
3. Enter state value (lowercase, e.g., "mystic_rain")
4. Click "Create"
5. Edit parameters in the weather state editor
6. Add to weather sets as needed

### Deleting Weather States or Sets

**Sets**: Hover over set name → click × button → confirm

**States**: Open weather state editor → scroll to bottom → click "Delete Weather State" → confirm

**Note**: Deleting a weather state removes it from all sets and deletes all parameters

## Important Notes

### Before Saving
- **Create a Backup**: The system creates automatic backups, but consider manual backups too
- **Validate First**: Always click "Validate" before "Save Changes"
- **Restart Required**: Changes only take effect after restarting the application

### Data Structure
- **Weather States**: Global list of available weather types
- **Weather Presets**: Parameters for each weather state
- **Weather Sets**: Collections of weather states with their own characteristics

### File Location
- Original: `lib/weather_params.py`
- Backup: `lib/weather_params.py.backup`

## Parameter Reference

### Set Parameters
- **season_speed**: 0.5x = 60 min/year, 1.0x = 30 min/year, 2.0x = 15 min/year
- **season_extremity**: 0.5x = subtle, 1.0x = normal, 2.0x = extreme seasonal bias
- **transition_speed**: 0.5x = ~8 min, 1.0x = ~4 min, 2.0x = ~2 min between changes

### Common Weather Parameters
- **wind_speed**: Wind intensity (0.0 - 2.0)
- **rain_rate**: Rainfall amount (0.0 - 1.0)
- **fog**: Fog density (0.0 - 1.5)
- **fog_color**: RGB array [R, G, B] (0.0 - 1.0 each)
- **starryness**: Star visibility (0.0 - 1.0)
- **firefly_density**: Firefly count multiplier (0.0 - 2.0)
- **possible_transitions**: Array of weather state IDs this can transition to
- **transition_weights**: Probability weights for each transition
- **ambient_sound**: Audio file name from media/sounds/

## Troubleshooting

**Editor won't load**:
- Check Flask server is running (Stories_OGL.py)
- Check console for error messages
- Try refreshing the page

**Changes not saving**:
- Click "Validate" first to check for errors
- Check you have write permissions to lib/ folder
- Review error messages in status bar

**Application doesn't reflect changes**:
- Restart Stories_OGL.py after saving
- Check that weather_params.py was actually updated
- Restore from backup if needed

## Tips

1. **Start Small**: Edit existing sets before creating new ones
2. **Test Frequently**: Validate before saving to catch errors early
3. **Use Descriptive Names**: Makes sets easier to identify
4. **Document Changes**: Use the description field to note modifications
5. **Keep Backups**: The editor creates backups, but manual copies are wise

## Keyboard Shortcuts

- **Ctrl+S**: Not implemented (use "Save Changes" button)
- **Click outside dropdown**: Closes weather state dropdown
- **Tab**: Navigate between form fields

## API Endpoints

The editor uses these endpoints:
- `GET /api/weather_editor/all_data` - Load all weather data
- `POST /api/weather_editor/validate` - Validate without saving
- `POST /api/weather_editor/save` - Save to weather_params.py

## Support

For issues or questions:
1. Check the console (F12 in browser) for error messages
2. Review validation errors before saving
3. Restore from backup if needed: `weather_params.py.backup`
