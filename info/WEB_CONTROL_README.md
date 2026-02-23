# Web Control System for GL_Simple

This system provides a web-based interface for controlling GL_Simple environmental effects in real-time.

## Overview

The web control system consists of three main components:

1. **WebController** (`corefunctions/web_controller.py`) - Flask-based web server
2. **Control Panel HTML** (`templates/control_panel.html`) - Web interface
3. **Integration** (`Stories_OGL.py`) - Connected to the main application

## Features

- **Real-time Control**: Modify parameters while the application is running
- **Thread-safe**: Web server runs in a separate thread
- **Dynamic UI**: Control schema automatically generates the web interface
- **Multiple Control Types**: Sliders, checkboxes, dropdowns
- **Live Updates**: Interface polls for changes every 2 seconds

## Quick Start

1. Install Flask:
   ```bash
   pip install flask
   ```

2. Run the main application:
   ```bash
   python Stories_OGL.py
   ```

3. Open your web browser to:
   ```
   http://localhost:5000
   ```

## Usage

### Default Controls

The system includes these default controls:

- **Weather Intensity** (0.0 - 2.0): Multiplier for weather effects
- **Fog Strength** (0.0 - 1.0): Atmospheric fog density
- **Rain Amount** (0.0 - 1.0): Rain effect intensity
- **Audio Sensitivity** (0.1 - 3.0): Microphone input sensitivity
- **Enable Fireflies** (checkbox): Toggle firefly effects
- **Enable Stars** (checkbox): Toggle star effects
- **Color Mode** (dropdown): Color scheme selection
- **Effect Speed** (0.1 - 5.0): Speed multiplier for all effects

### Adding Custom Controls

You can add custom controls programmatically:

```python
# In Stories_OGL.py or your custom code
env_system.web_controller.add_control(
    key="my_custom_param",
    control_type="slider",
    label="My Custom Parameter",
    min=0,
    max=100,
    step=1,
    default=50
)
```

### Accessing Control Values

Control values are stored in the `web_controls` dictionary:

```python
# Read a control value
intensity = env_system.web_controls.get('weather_intensity', 1.0)

# Use control values in your effects
if env_system.web_controls.get('enable_fireflies', True):
    # Firefly logic here
    pass
```

### Control Types

**Slider:**
```python
web_controller.add_control(
    "my_slider",
    "slider",
    "My Slider",
    min=0.0,
    max=10.0,
    step=0.5,
    default=5.0
)
```

**Checkbox:**
```python
web_controller.add_control(
    "my_checkbox",
    "checkbox",
    "Enable Feature",
    default=True
)
```

**Dropdown:**
```python
web_controller.add_control(
    "my_select",
    "select",
    "Choose Option",
    options=["option1", "option2", "option3"],
    default="option1"
)
```

## Architecture

### WebController Class

Located in `corefunctions/web_controller.py`:

- Manages a shared dictionary (`control_dict`) with thread-safe access
- Provides REST API endpoints for reading/writing values
- Automatically generates UI from control schema
- Runs Flask server in a background thread

### Control Flow

1. User adjusts control in web interface
2. JavaScript sends POST request to `/api/update`
3. Flask server updates the control dictionary
4. Main application reads values from dictionary
5. Effects are modified based on current values

### API Endpoints

- `GET /` - Serves the control panel HTML
- `GET /api/schema` - Returns control definitions
- `GET /api/values` - Returns current control values
- `POST /api/update` - Updates a single control value
- `POST /api/batch_update` - Updates multiple values at once

## Customization

### Changing the Port

```python
web_controller = WebController(web_controls, port=8080)
```

### Custom Templates

The HTML template is located at `templates/control_panel.html`. You can modify it to:
- Change styling and colors
- Add custom JavaScript functionality
- Reorganize layout
- Add data visualizations

### Integrating with Existing Code

The `apply_web_controls()` method in `EnvironmentalSystem` shows how to use control values:

```python
def apply_web_controls(self):
    """Apply web control values to system parameters."""
    if 'weather_intensity' in self.web_controls:
        intensity = self.web_controls['weather_intensity']
        # Apply intensity to weather effects
        
    if 'fog_strength' in self.web_controls:
        fog_strength = self.web_controls['fog_strength']
        # Update fog in real-time
```

## Network Access

By default, the server listens on `0.0.0.0`, making it accessible from other devices on your network.

To access from another device:
1. Find your computer's IP address
2. Navigate to `http://YOUR_IP:5000` from the other device

To restrict to localhost only, modify `web_controller.py`:
```python
self.app.run(host='127.0.0.1', port=self.port, ...)
```

## Security Note

This web server is intended for local/trusted network use. For production environments:
- Add authentication
- Use HTTPS
- Implement rate limiting
- Validate all inputs

## Troubleshooting

**Port already in use:**
Change the port number when creating the WebController.

**Can't connect from browser:**
- Check firewall settings
- Ensure the application is running
- Verify the correct port number

**Controls not updating:**
- Check browser console for JavaScript errors
- Verify Flask server is running (check terminal output)
- Ensure control keys match between schema and application code

## Example Integration

Here's a complete example of integrating a custom effect:

```python
# Add a custom control for effect brightness
env_system.web_controller.add_control(
    "effect_brightness",
    "slider",
    "Effect Brightness",
    min=0.0,
    max=2.0,
    step=0.1,
    default=1.0
)

# In your update loop or effect code:
brightness = env_system.web_controls.get('effect_brightness', 1.0)
color = base_color * brightness
```
