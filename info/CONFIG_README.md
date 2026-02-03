# Configuration Guide

## Overview

The GL_Simple system now supports a comprehensive configuration file (`config.ini`) that allows you to control various features without modifying the code. This includes feature flags for enabling/disabling functionality, performance tuning, and startup behavior.

## Configuration File: config.ini

The configuration file is located in the root directory: `/config.ini`

### Configuration Sections

#### [Display]
Controls visual display and rendering settings:

- **show_rendering_window** (True/False): Enable or disable the OpenCV rendering window
  - Set to `False` for headless mode (no display, maximum performance)
  - Equivalent to setting `self.scheduler.state["simulate"]` in code
  
- **magnification** (1-5): Display magnification factor for the OpenCV window
  - Higher values make the display larger but don't affect LED output
  
- **show_fps** (True/False): Enable FPS counter display
  - When enabled, prints FPS stats every 500 frames
  - Previously was commented out code: `#print(f"FPS: {actual_fps:.1f}...")`
  
- **target_fps** (number): Target frame rate in frames per second
  - Default: 40 FPS
  - Adjust based on your system performance

#### [Audio]
Audio input configuration:

- **microphone_device** (string): Name of the microphone device to use
  - Common values: "TONOR", "" (empty for default), "default"
  - Leave empty to use system default microphone

#### [WebControl]
Web interface settings:

- **enable_web_control** (True/False): Enable/disable the web control interface
  - Set to `False` for maximum performance when web control isn't needed
  
- **web_port** (number): Port number for the web server
  - Default: 5000
  
- **service_name** (string): mDNS service name for discovery
  - Default: "glsimple"
  
- **admin_password** (string): Password for admin panel access
  - Change from default "admin123" for security

#### [Startup]
Initial state configuration when the program starts:

- **startup_weather_set** (string): Weather set to load on startup
  - Examples: "default", "ocean", "desert", "forest"
  - Leave empty to use default
  
- **startup_weather_state** (string): Initial weather state
  - Examples: "CLEAR", "OCEAN_KELP_FOREST", "RAIN", "FOG"
  - Must match a WeatherState enum value
  - Leave empty for default
  
- **immediate_startup** (True/False): Apply startup weather immediately
  - `True`: Jump directly to the weather state
  - `False`: Transition smoothly to the weather state

#### [Performance]
Performance-related settings:

- **frame_dimensions** (width,height pairs): Display dimensions
  - Format: `width1,height1;width2,height2;...`
  - Example: `128,300` for single display
  - Example: `128,300;256,256` for two displays
  
- **use_shader_renderer** (True/False): Enable shader-based rendering
  - Recommended: `True`
  
- **enable_precision_timing** (True/False): Enable high-precision timing on Windows
  - Requires winmm.dll
  - Only affects Windows systems

#### [Debug]
Debug and development features:

- **debug_mode** (True/False): Enable debug information output
  
- **show_tree** (True/False): Enable tree visualization (debug feature)
  
- **show_skyfull** (True/False): Enable skyfull mode (debug feature)

## Example Configurations

### High Performance (Headless)
```ini
[Display]
show_rendering_window = False
show_fps = False
target_fps = 60

[WebControl]
enable_web_control = False
```

### Development Mode
```ini
[Display]
show_rendering_window = True
show_fps = True
magnification = 4
target_fps = 40

[Debug]
debug_mode = True
```

### Custom Startup
```ini
[Startup]
startup_weather_set = ocean
startup_weather_state = OCEAN_KELP_FOREST
immediate_startup = True
```

## Migration from Hardcoded Values

The following previously hardcoded values are now configurable:

| Old Code Location | Config Setting |
|------------------|----------------|
| `headless=False` | `[Display] show_rendering_window` |
| `magnification=3` | `[Display] magnification` |
| `#print(f"FPS: ...")` | `[Display] show_fps` |
| `FRAME_TIME = 1/40` | `[Display] target_fps` |
| `device_name="TONOR"` | `[Audio] microphone_device` |
| `enable_web_control = True` | `[WebControl] enable_web_control` |
| `port=5000` | `[WebControl] web_port` |
| `admin_password="admin123"` | `[WebControl] admin_password` |
| `change_weather_set("ocean")` | `[Startup] startup_weather_set` |
| `transition_to_weather(...)` | `[Startup] startup_weather_state` |
| `scheduler.state["simulate"]` | `[Display] show_rendering_window` |
| `scheduler.state["tree"]` | `[Debug] show_tree` |
| `scheduler.state["skyfull"]` | `[Debug] show_skyfull` |

## Usage

1. Edit `config.ini` with your preferred settings
2. Run the program normally: `./quick_run.sh` or `python Stories_OGL.py`
3. The configuration is loaded automatically at startup

## Notes

- If `config.ini` doesn't exist, default values are used
- Invalid configuration values will fall back to defaults
- Changes to `config.ini` require restarting the program to take effect
- The configuration file uses standard INI format with `#` for comments
