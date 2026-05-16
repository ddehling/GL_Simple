# GL_Simple — Claude Code Guide

> **Working on a shader effect?** STOP and read
> [docs/shader_info.txt](docs/shader_info.txt) first — at minimum the TL;DR and
> "Alpha Output Rules" sections. Almost every shader regression in this
> codebase has been a re-violation of those rules (pre-multiplied alpha,
> wrong depth-state for fullscreen quads, missing uniform-wiring step,
> coupling `u_time` to a varying uniform — see "Time-based Animation").
> If (and only if) the shader is audio-reactive, also read
> [docs/shader_audio_reactivity.md](docs/shader_audio_reactivity.md).

## Project Overview

OpenGL-based DMX lighting control system with real-time GPU shader effects, audio-reactive visuals, weather simulation, and a web control panel. Primary use: live event visualization and stage lighting.

## Running the Application

### Scripts (recommended)

```bash
# First-time setup + launch (installs deps, creates venv, starts app)
./bin/setup_and_run.sh       # Linux/macOS
bin/setup_and_run.bat        # Windows (double-click or run in cmd)

# Quick launch after setup is done
./bin/quick_run.sh           # Linux/macOS
bin/quick_run.bat            # Windows
```

On Linux, make the scripts executable first if needed:
```bash
chmod +x bin/setup_and_run.sh bin/quick_run.sh
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
python tools/sound_editor.py

# Shader testing / alternate renderer
python tools/computer.py

# Install / reinstall dependencies
pip install -r requirements.txt
```

Web control panel runs at `http://localhost:5000` when started.

## Display Modes & Keyboard Shortcuts

The OpenGL window supports four view modes, toggled with keyboard shortcuts:

| Key / Input | Action |
|-------------|--------|
| F | Toggle flat/fan view |
| D | Toggle smooth/LED style |
| Scroll wheel | Zoom in/out (fan modes, centered on cursor) |
| Left-click drag | Pan the view (fan modes) |
| Middle-click | Reset zoom and pan |
| ESC | Quit |

The four resulting modes:

| Mode | Description |
|------|-------------|
| Flat Smooth | Magnified pixel blit of the 128×300 FBO (default) |
| Flat LED | Instanced circles showing individual LED values |
| Fan Smooth | Textured semicircle mesh simulating the physical fan layout |
| Fan LED | Instanced circles arranged in the physical fan semicircle |

## Web Preview

Visit `http://localhost:5000/preview` for a live WebGL preview in the browser. Supports the same four view modes via buttons. Frames are streamed as lossless PNG at 15 Hz via Socket.IO. Works in headless mode (no GLFW window needed).

## Architecture

```
Stories_OGL.py              # Entry point — wires everything together
engine/
  render_pipeline.py        # RenderPipeline — hardware init + per-frame loop
lib/
  event_scheduler.py        # TimedEvent + EventScheduler (pure event queue)
  audio_analyzer.py         # MicrophoneAnalyzer — mic capture + FFT analysis
  audio_engine.py           # AudioEngine, StreamingPlayer — audio playback
  ambient_audio.py          # AmbientAudioController — cross-fade management
  dmx_sender.py             # SACNPixelSender — sACN/E1.31 DMX output
  midi_controller.py        # KorgNanoKontrol2 MIDI integration
  weather_params.py         # Weather states, presets, environmental params
  weather_state.py          # WeatherStateController — transition interpolation
  weather_set.py            # WeatherSetManager — active set + event registry
renderer/
  shader_renderer.py        # GLFW window + OpenGL rendering loop
  fan_geometry.py           # Pure-numpy fan/polar geometry (shared by GL and web preview)
  effects/                  # 40+ individual shader effect modules
    base.py                 # ShaderEffect base class — all effects extend this
web/
  web_controller.py         # Flask web control panel + preview frame streaming
  templates/                # Flask HTML templates for web UI
  static/js/preview.js      # WebGL2 live preview client
  static/css/preview.css    # Preview page styles
tools/                      # Standalone utilities: sound_editor, computer, midi_integration_example, gl_test, wleddetect
config/                     # DMX universe and fixture definitions (Unit*.txt)
media/                      # Audio files (sounds/) and images (images/)
docs/                       # Documentation (see below)
bin/                        # Launch scripts (setup_and_run, quick_run)
```

## Key Configuration

Main settings are in `config.yaml` at the project root (display dimensions, headless mode, audio device, web port/password). Falls back to defaults if the file is missing.

Weather set/state defaults and DMX receiver config remain in `Stories_OGL.py`.

## Adding a Shader Effect

**REQUIRED — read [docs/shader_info.txt](docs/shader_info.txt) before writing or
modifying any shader.** It documents the depth/alpha-blending rules, the
fullscreen-quad vs 3D-particle patterns, the wrapper-uniform wiring checklist,
and known gotchas (pre-multiplied alpha, missing uniform updates in wrappers,
default-value pitfalls). Most shader bugs in this codebase have been the same
few violations of those rules — read the TL;DR section at the top of the file
at minimum.

**If (and only if) the shader is audio-reactive** (responds to microphone /
sound energy / `outstate['sound']`), ALSO read
[docs/shader_audio_reactivity.md](docs/shader_audio_reactivity.md). Skip it
otherwise — non-audio shaders don't need it.

1. Read [docs/shader_info.txt](docs/shader_info.txt) — pick Pattern A
   (3D-style geometry) or Pattern B (fullscreen-quad atmospheric layer) for
   what you're building.
2. If audio-reactive: also read
   [docs/shader_audio_reactivity.md](docs/shader_audio_reactivity.md).
3. Create a new file in `renderer/effects/` extending `ShaderEffect` from `base.py`
4. Implement `__init__`, `update(dt, audio_data)`, and `render()` methods
5. Register it in `Stories_OGL.py`'s `event_map` or schedule it directly in the `__main__` block
6. If the effect introduces new weather params, follow the wiring checklist in
   [docs/shader_info.txt](docs/shader_info.txt) ("Wiring a New Uniform").

## Weather Sets

Forest, Desert, Ocean, Spooky, Mountain, Cyberpunk. Each set is a collection of weather states with transition logic defined in `lib/weather_params.py`.

## DMX Output

Uses sACN/E1.31 protocol via the `sacn` library. Universe configs live in `config/Unit*.txt`. `lib/dmx_sender.py` maps rendered pixel data to DMX channel values.

## Audio Input

`lib/audio_analyzer.py` captures microphone input and extracts frequency bands. `renderer/effects/audio_*.py` effects subscribe to this data. Run `tools/sound_editor.py` to identify the correct device name if audio isn't working.

## Documentation

- `docs/WEATHER_EDITOR_README.md` — Weather system
- `docs/WEB_CONTROL_README.md` — Web interface
- `docs/MIDI_README.md` — MIDI controller setup
- `docs/EVENT_MAP_SYSTEM.md` — Event scheduling
- `docs/PARAMETER_MANAGEMENT.md` — Parameter system
- `docs/MEDIA_SUBMODULES.md` — Per-project media (submodule) workflow + new-project setup
- `docs/shader_info.txt` — Shader effect reference (read for ALL shader work)
- `docs/shader_audio_reactivity.md` — Audio-reactive shader guide (read ONLY for audio-reactive shaders)
