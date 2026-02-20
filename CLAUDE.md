# GL_Simple — Claude Code Guide

## Project Overview

OpenGL-based DMX lighting control system with real-time GPU shader effects, audio-reactive visuals, weather simulation, and a web control panel. Primary use: live event visualization and stage lighting.

## Running the Application

### Scripts (recommended)

```bash
# First-time setup + launch (installs deps, creates venv, starts app)
./setup_and_run.sh       # Linux/macOS
setup_and_run.bat        # Windows (double-click or run in cmd)

# Quick launch after setup is done
./quick_run.sh           # Linux/macOS
quick_run.bat            # Windows
```

On Linux, make the scripts executable first if needed:
```bash
chmod +x setup_and_run.sh quick_run.sh
```

### Manual

```bash
# Activate venv (Linux/macOS)
source venv/bin/activate

# Activate venv (Windows/bash)
source venv/Scripts/activate

# Main application
python Stories_OGL.py

# Audio device configuration utility
python sound_editor.py

# Shader testing / alternate renderer
python computer.py

# Install / reinstall dependencies
pip install -r info/requirements.txt
```

Web control panel runs at `http://localhost:5000` when started.

## Architecture

```
Stories_OGL.py              # Entry point — wires everything together
corefunctions/
  shader_renderer.py        # GLFW window + OpenGL rendering loop
  Events.py                 # Timed event scheduler, frame orchestration
  soundinput.py             # Microphone capture and frequency analysis
  ImageToDMX.py             # Converts rendered frames to DMX lighting data
  web_controller.py         # Flask web control panel
  midi_controller.py        # Korg nanoKontrol2 MIDI integration
  weather_params.py         # Weather states, presets, environmental params
  shader_effects/           # 40+ individual shader effect modules
    base.py                 # ShaderEffect base class — all effects extend this
sceneutils/                 # Scene/biome-level event compositions
templates/                  # Flask HTML templates for web UI
DMXconfig/                  # DMX universe and fixture definitions (Unit*.txt)
media/                      # Audio files (sounds/) and images (images/)
info/                       # Documentation (see below)
```

## Key Configuration (in Stories_OGL.py)

| Lines | Setting |
|-------|---------|
| 22–25 | Frame dimensions and magnification |
| 40    | Audio input device name |
| 49    | Enable/disable web control |
| 57    | Web control port |
| 59    | Admin password |
| 639   | Initial weather set |
| 640   | Initial weather state |

## Adding a Shader Effect

1. Create a new file in `corefunctions/shader_effects/` extending `ShaderEffect` from `base.py`
2. Implement `__init__`, `update(dt, audio_data)`, and `render()` methods
3. Register it in the appropriate scene file under `sceneutils/` or directly in `Stories_OGL.py`

## Weather Sets

Forest, Desert, Ocean, Spooky, Mountain, Cyberpunk. Each set is a collection of weather states with transition logic defined in `corefunctions/weather_params.py`.

## DMX Output

Uses sACN/E1.31 protocol via the `sacn` library. Universe configs live in `DMXconfig/Unit*.txt`. `ImageToDMX.py` maps rendered pixel data to DMX channel values.

## Audio Input

`soundinput.py` captures microphone input and extracts frequency bands. `shader_effects/audio_*.py` effects subscribe to this data. Run `sound_editor.py` to identify the correct device name if audio isn't working.

## Documentation

- `info/WEATHER_EDITOR_README.md` — Weather system
- `info/WEB_CONTROL_README.md` — Web interface
- `info/MIDI_README.md` — MIDI controller setup
- `info/EVENT_MAP_SYSTEM.md` — Event scheduling
- `info/PARAMETER_MANAGEMENT.md` — Parameter system
- `info/shader_info.txt` — Shader effect reference
