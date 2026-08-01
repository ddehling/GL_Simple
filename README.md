# GL_Simple

OpenGL-based DMX lighting control system with real-time GPU shader effects, audio-reactive visuals, weather simulation, and a web control panel. Designed for live event visualization and art installations.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [Weather System](#weather-system)
4. [Web Control Panel](#web-control-panel)
5. [MIDI Controller](#midi-controller)
6. [Event Map System](#event-map-system)
7. [Troubleshooting](#troubleshooting)
8. [Developer Reference](#developer-reference)

---

## Quick Start

### One-script setup (Recommended)

```bash
git clone https://github.com/ddehling/GL_Simple.git
cd GL_Simple
./bin/linux-install.sh        # Linux / Raspberry Pi
# or, on Windows:
powershell -ExecutionPolicy Bypass -File bin\windows-install.ps1
```

Installs git + the GitHub CLI, signs you into GitHub (browser device
flow — no PAT pasting), lets you pick which project(s) to install
and which is primary, clones them, installs Python deps, launches
the app. Full details: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

### Launching after setup

```bash
venv/bin/python Stories_OGL.py        # Linux/macOS
venv\Scripts\python Stories_OGL.py    # Windows
```

### Manual Launch

```bash
# Activate virtual environment
source venv/bin/activate        # Linux/macOS
source venv/Scripts/activate    # Windows/bash

# Run the main application
python Stories_OGL.py
```

Once running, the web control panel is at `http://localhost:5000`.

### Dependencies

Install or reinstall at any time:
```bash
pip install -r requirements.txt
```

Key packages: `glfw`, `PyOpenGL`, `numpy`, `scipy`, `opencv-python`, `sounddevice`, `librosa`, `soundfile`, `flask`, `sacn`, `pygame`, `numba`, `pyglm`, `scikit-image`, `zeroconf`.

### Requirements

- Python 3.8+ (3.10+ recommended)
- Windows 10/11, Ubuntu/Debian, or macOS
- OpenGL 3.3+ compatible GPU (OpenGL ES 3.1 for Raspberry Pi)
- Audio input device (optional, for audio-reactive effects)

---

## Configuration

Main settings live in `config.yaml` at the project root:

```yaml
display:
  width: 128          # LED strips (framebuffer columns)
  height: 300         # LEDs per strip (framebuffer rows)
  magnification: 0    # Window magnification (0 = auto-scale to monitor)
  headless: false     # true = no desktop window, web preview only

audio:
  device_name: "TONOR"  # Mic device (run tools/editors/sound_editor.py to find yours)

web:
  enabled: true
  port: 5000
  admin_password: "admin123"
```

If `config.yaml` is missing, the app starts with these defaults.

**DMX universes and fixtures**: edit files in `config/`.

**Weather set / state defaults**: configured in `Stories_OGL.py` (search for `DEFAULT_WEATHER_SET`).

### Linux-Specific Setup

If DMX receivers update at a very slow frame rate, the network send buffer may be too small. Add to `/etc/sysctl.conf`:
```
net.core.wmem_max=16777216
net.core.wmem_default=16777216
```

If you get PortAudio errors:
```bash
sudo apt-get install portaudio19-dev
```

---

## Weather System

### Available Weather Sets

| Set | States | Season Speed | Transition Speed | Vibe |
|-----|--------|-------------|-----------------|------|
| **Peaceful Forest** (default) | clear, light_rain, foggy, firefly, mushroom, bloom, leaves | 1.0x (30 min/yr) | 0.8x (~5 min) | Gentle, natural |
| **Storm World** | windy_night, heavy_rain, thunderstorm, foggy, spooky | 1.5x (20 min/yr) | 1.5x (~2.7 min) | Intense, dramatic |
| **Desert Realm** | clear, sandstorm, volcano, windy_night | 0.5x (60 min/yr) | 0.6x (~6.7 min) | Harsh, alien |
| **Ethereal Mist** | heavy_fog, foggy, spooky, mushroom, firefly | 0.7x (43 min/yr) | 0.5x (~8 min) | Mysterious |
| **Cosmic Night** | clear, asteroid, windy_night | 2.0x (15 min/yr) | 2.0x (~2 min) | Celestial |
| **Full Spectrum** | All 15 states | 1.0x | 1.0x | Maximum variety |

### Set Parameters

- **season_speed**: How fast the 30-minute year cycle runs. `0.5x` = 60 min/yr, `1.0x` = 30 min/yr, `2.0x` = 15 min/yr.
- **season_extremity**: How much seasons bias weather transitions. `0.5x` = subtle, `1.0x` = normal, `2.0x` = extreme.
- **transition_speed**: How often weather changes. `0.5x` = ~8 min, `1.0x` = ~4 min, `2.0x` = ~2 min.

### Transition Probability

Each frame (~30 fps): `P(transition) = (1/800) × weather.Switch_rate × set.transition_speed`

On transition: if a set change is pending, the new set activates and a random state from it is chosen. Otherwise, a weighted-random state from the current set is picked (weights influenced by `season_extremity`).

### Set Isolation

Each set only transitions between its own states. If a state's normal transitions include states outside the current set, those are filtered out. If no valid transitions remain, a random state from the set is chosen.

### Changing Sets via Web

1. Open `http://localhost:5000/weather_sets` (or `http://glsimple.local:5000/weather_sets`)
2. Click any weather set card — it queues the change
3. On the next weather transition (up to ~4 min), the system switches sets

### Adding a New Set in Code

Edit `lib/weather_params.py`:

```python
WEATHER_SETS = {
    "your_set_name": {
        "name": "Display Name",
        "description": "What makes this set special",
        "states": ["clear", "foggy", "spooky"],
        "season_speed": 1.0,
        "season_extremity": 1.0,
        "transition_speed": 1.0,
        "background_events": ["clouds", "stars"],   # Continuous effects
        "allowed_parameters": ["wind_speed", "fog", "starryness", ...],
    },
}
```

Then update `web/templates/weather_sets.html` to add its icon and description card.

### Weather State Parameters

Common parameters for each weather state in `WEATHER_PRESETS`:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `wind_speed` | 0.0–2.0 | Wind intensity |
| `rain_rate` | 0.0–1.0 | Rainfall amount |
| `fog` | 0.0–1.5 | Fog density |
| `fog_color` | [R, G, B] | Fog RGB color (0.0–1.0 each) |
| `starryness` | 0.0–1.0 | Star visibility |
| `firefly_density` | 0.0–2.0 | Firefly count multiplier |
| `ambient_sound` | filename | Audio file from `media/sounds/` |
| `possible_transitions` | string array | State IDs this can transition to |
| `transition_weights` | number array | Probability weights per transition |
| `on_transition_events` | list of tuples | Events triggered when this state activates |

### Weather Set Editor

A visual browser-based editor is available at `http://localhost:5000/weather_editor`.

**Workflow:**
1. Select a set from the left panel
2. Click a weather state tab to edit its parameters
3. Click **Validate** to check for errors before saving
4. Click **Save Changes** — this overwrites `lib/weather_params.py` (a backup is created at `.backup`)
5. **Restart the application** for changes to take effect

**Creating a new weather state:**
- Click **+ New Weather State**
- Name uses `UPPERCASE_SNAKE_CASE` (e.g., `MYSTIC_RAIN`)
- Value uses `lowercase` (e.g., `mystic_rain`)

**Allowed Parameters per Set:**
Each set has an `allowed_parameters` list — only those parameters appear in the editor for that set. This keeps the UI focused. You can add/remove parameters with the **+ Add Parameter** button, or create entirely new custom parameters (name, type, default value).

Parameter types: `Number`, `Text`, `Array` (3-element RGB), `Array of Strings`, `Array of Numbers`.

**Editor API endpoints:**
- `GET /api/weather_editor/all_data`
- `POST /api/weather_editor/validate`
- `POST /api/weather_editor/save`

---

## Web Control Panel

Access at `http://localhost:5000` (or from another device on your network at `http://YOUR_IP:5000`).

The server listens on `0.0.0.0` by default. To restrict to localhost, change `web_controller.py` to `host='127.0.0.1'`.

### Default Controls

- **Weather Intensity** (0.0–2.0): Effect multiplier
- **Fog Strength** (0.0–1.0): Fog density
- **Rain Amount** (0.0–1.0): Rain intensity
- **Audio Sensitivity** (0.1–3.0): Mic input scaling
- **Enable Fireflies** (checkbox)
- **Enable Stars** (checkbox)
- **Color Mode** (dropdown)

### Adding Custom Controls

```python
env_system.web_controller.add_control(
    key="my_param",
    control_type="slider",   # "slider", "checkbox", or "select"
    label="My Parameter",
    min=0, max=100, step=1, default=50
)

# Checkbox
web_controller.add_control("my_toggle", "checkbox", "Enable Feature", default=True)

# Dropdown
web_controller.add_control("my_select", "select", "Choose Option",
    options=["option1", "option2", "option3"], default="option1")
```

### Reading Control Values

```python
intensity = env_system.web_controls.get('weather_intensity', 1.0)

if env_system.web_controls.get('enable_fireflies', True):
    # firefly logic
```

### Web API

- `GET /` — Control panel HTML
- `GET /api/schema` — Control definitions
- `GET /api/values` — Current control values
- `POST /api/update` — Update a single value
- `POST /api/batch_update` — Update multiple values

The server runs in a background thread and updates the shared `web_controls` dict, which the main loop reads each frame.

---

## Display Modes

The physical installation is 128 LED strips arranged as a semicircle (fan), fanning out from a center point. Each strip has 300 LEDs. The renderer's FBO is 128×300 pixels — one column per strip, one row per LED.

The OpenGL window and web preview support four view modes:

| Mode | Description |
|------|-------------|
| **Flat Smooth** | Magnified pixel blit of the FBO (default) |
| **Flat LED** | Instanced circles showing individual LED values on a grid |
| **Fan Smooth** | Textured semicircle mesh simulating the physical fan layout |
| **Fan LED** | Instanced circles arranged in the physical fan semicircle |

### Keyboard Shortcuts (OpenGL window)

| Key | Action |
|-----|--------|
| **F** | Toggle flat / fan view |
| **D** | Toggle smooth / LED style |
| **Scroll wheel** | Zoom in/out (fan modes, centered on cursor) |
| **Left-click drag** | Pan the view (fan modes) |
| **Middle-click** | Reset zoom and pan |
| **ESC** | Quit |

### Web Preview

Visit `http://localhost:5000/preview` for a live WebGL preview in the browser. The same four view modes are available via buttons. Frames are streamed as lossless PNG at 15 Hz over Socket.IO. Works when the app is running in headless mode (no desktop window).

---

## MIDI Controller

Supports the **Korg nanoKontrol2** via pygame MIDI. Requires `pygame` (included in `requirements.txt`).

### Controller Layout

```
Channel:      1      2      3      4      5      6      7      8
           ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
Knobs:     │  ◯  │  ◯  │  ◯  │  ◯  │  ◯  │  ◯  │  ◯  │  ◯  │
           └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
           ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
Sliders:   │  |  │  |  │  |  │  |  │  |  │  |  │  |  │  |  │
           └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
           ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
S buttons: │  S  │  S  │  S  │  S  │  S  │  S  │  S  │  S  │
M buttons: │  M  │  M  │  M  │  M  │  M  │  M  │  M  │  M  │
R buttons: │  R  │  R  │  R  │  R  │  R  │  R  │  R  │  R  │
           └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
Transport: ⏮ ⏭  ↻  ●  ◄  ■  ►  ●
```

### Control Names

- **Knobs**: `knob_1` – `knob_8` → float 0.0–1.0
- **Sliders**: `slider_1` – `slider_8` → float 0.0–1.0
- **Channel buttons**: `s_button_1`–`8`, `m_button_1`–`8`, `r_button_1`–`8` → bool
- **Transport**: `track_prev`, `track_next`, `cycle`, `marker_set`, `marker_prev`, `marker_next`, `rewind`, `forward`, `stop`, `play`, `record` → bool

### Basic Usage

```python
from lib.midi_controller import KorgNanoKontrol2

midi = KorgNanoKontrol2(auto_connect=True)
midi.start_reading()   # Background thread, ~1000 Hz poll

# Read values
knob1 = midi.get_knob(1)           # 0.0–1.0
slider3 = midi.get_slider(3)       # 0.0–1.0
s1 = midi.get_button('s', 1)       # True/False

# Or manually each frame (instead of background thread)
changes = midi.update()            # Returns dict of changed controls
```

### Callbacks

```python
def on_slider1(name, value):
    scheduler.state['fog_strength'] = value

midi.register_callback('slider_1', on_slider1)
```

### Mapping Transport Buttons to Weather

```python
def setup_weather_controls(midi, env_system):
    def on_button(weather_state):
        def cb(name, pressed):
            if pressed:
                env_system.transition_to_weather(weather_state, 5.0)
        return cb

    midi.register_callback('marker_set',  on_button(WeatherState.CLEAR))
    midi.register_callback('marker_prev', on_button(WeatherState.RAIN))
    midi.register_callback('cycle',       on_button(WeatherState.SANDSTORM))
```

### Troubleshooting MIDI

**Device not found:**
```python
midi = KorgNanoKontrol2(auto_connect=False)
midi.list_devices()          # Shows all available MIDI devices
midi.device_name = "nano"   # Try partial name match
midi.connect()
```

**Button values stuck:** Some buttons send Note On instead of CC. If buttons aren't responding, enter setup mode by holding CYCLE + TRACK on power-on.

**Test standalone:**
```bash
python lib/midi_controller.py
python tools/hardware/midi_integration_example.py
```

---

## Sound Playback

There are two categories of sound and two mechanisms for playing them:

| | **Short (< ~30 s)** | **Long (voice, music)** |
|---|---|---|
| **Ambient** | Loop via `StreamingPlayer` | Loop via `StreamingPlayer` |
| **Event-triggered** | `play_oneshot()` | `StreamingPlayer(loop=False)` |

### Ambient sounds

Handled automatically by `transition_to_weather()` via `StreamingPlayer`. Defined per weather state in `WEATHER_PRESETS` as `ambient_sound` filename. Fade in/out on transition is built in.

### Short event sounds — `play_oneshot()`

Non-blocking fire-and-forget. The file loads in a background thread on first call; subsequent calls with the same file are served from cache instantly.

```python
soundengine = scheduler.state["soundengine"]
soundengine.play_oneshot(Path("media/sounds/thunder.wav"), volume=1.0)
```

To wire a short sound into the event map (so it can be scheduled like a shader effect):

```python
# In EnvironmentalSystem.__init__, add to self.event_map:
"thunder_crack": lambda: (audio_thunder_crack, {"volume": 1.0})

# Wrapper function (same pattern as shader effects):
def audio_thunder_crack(state, outstate, volume=1.0):
    if state['count'] == 0:
        outstate['soundengine'].play_oneshot(
            Path("media/sounds/thunder.wav"), volume=volume
        )
    # no cleanup needed — oneshot manages itself
```

Then reference it in a weather state's `on_transition_events` or in `random_events()` like any other event.

### Long event sounds — `StreamingPlayer`

Streams from disk, never fully loaded into memory. Interruptible: the scheduler's cleanup hook (`count == -1`) fires when the scene changes, giving the sound a chance to fade out gracefully.

```python
def audio_storm_narration(state, outstate, filepath="narration.wav", volume=1.0):
    if state['count'] == 0:
        player = StreamingPlayer(
            engine=outstate['soundengine'],
            filepath=Path("media/sounds") / filepath,
            name=f"event_{id(state)}",
            loop=False,
            volume=volume,
            fade_in=1.0,
        )
        player.start()
        state['player'] = player

    if state['count'] == -1:          # scene changed or event expired
        state['player'].fade_out(2.0)
```

Register and schedule exactly like a shader effect:

```python
"storm_narration": lambda: (audio_storm_narration, {"filepath": "storm_voice.wav"})

# In weather preset:
"on_transition_events": [("storm_narration", 60, 0)]
```

### Supported audio formats

WAV, FLAC, OGG Vorbis, MP3 — all streamed chunk-by-chunk via miniaudio for `StreamingPlayer`. `play_oneshot()` uses soundfile (WAV/FLAC) or librosa (MP3) and caches the result in RAM.

---

## Event Map System

Events are defined by name in a central map and referenced by name in weather configuration — no hardcoded `if` chains needed.

### Event Map (in `Stories_OGL.py`)

```python
self.event_map = {
    # Simple event
    "stars":          lambda: fx.shader_stars,
    # Event with parameters
    "firefly":        lambda: (fx.shader_firefly, {"squish_top_width": 0.1}),
    "falling_leaves": lambda: (fx.shader_falling_leaves, {"squish_top_width": self.scale}),
    "fog":            lambda: (fx.shader_fog, {"strength": 0.0, "color": (0.7, 0.7, 0.8)}),
}
```

### Background Events

Run continuously (effectively forever) while a weather set is active. Defined in the set config:

```python
"peaceful_forest": {
    "background_events": ["clouds", "firefly", "stars"],
    ...
}
```

When a set activates, all existing events are cancelled and the background events are scheduled.

### On-Transition Events

Triggered when a specific weather state becomes active. Defined per state in `WEATHER_PRESETS`:

```python
WeatherState.SANDSTORM: {
    "on_transition_events": [("sandstorm_event", 100, 0)],
    # Format: (event_name, duration_seconds, frame_id)
}

# Multiple events:
WeatherState.THUNDERSTORM: {
    "on_transition_events": [
        ("lightning_event", 120, 0),
        ("heavy_rain_particles", 150, 0),
        ("thunder_rumble", 90, 1),   # Secondary display
    ],
}
```

Events run for their specified duration and are automatically cleaned up.

### Adding a New Event

1. **Add to event_map** in `EnvironmentalSystem.__init__()`:
   ```python
   "my_effect": lambda: (fx.shader_my_effect, {"intensity": 0.8}),
   ```

2. **Reference by name** in set config (background) or weather preset (on-transition):
   ```python
   "background_events": ["clouds", "my_effect"]
   # or
   "on_transition_events": [("my_effect", 60, 0)]
   ```

No changes to `transition_to_weather()` needed.

### Helper Method

```python
# Schedule directly from code
self._schedule_event_from_map("fog_beings", start_time=0, duration=60, frame_id=0)
```

### Performance Notes

- Keep background events to 2–5 per set
- On-transition events are temporary and can be more intensive
- Use `frame_id` to distribute load across multiple displays

---

## Troubleshooting

### "Python not found"
Ensure Python is installed and on PATH. Verify with `python --version`. Restart terminal after installing.

### "Module not found"
Run `pip install -r requirements.txt` with the venv activated. Or re-run `bin\windows-install.ps1` / `./bin/linux-install.sh`.

### Audio device not found
- Run `sound_editor.py` to list available devices
- Update the device name at line 40 of `Stories_OGL.py`
- Remove the `device_name` parameter to use the system default

### PortAudio errors on Linux
```bash
sudo apt-get install portaudio19-dev
```

### OpenGL errors
Update graphics drivers. GPU must support OpenGL 3.3+ (or OpenGL ES 3.1 on Raspberry Pi).

### Performance issues
- Close other GPU-intensive applications
- Reduce magnification in `Stories_OGL.py` (lines 22–25)
- Set `enable_web_control = False` to disable Flask overhead

### Port 5000 already in use
Change the port at line 57 of `Stories_OGL.py`, or close whatever is using 5000.

### Weather editor won't save
- Click **Validate** first — fix any reported errors
- Ensure write permissions on `lib/`
- Restore from backup: `lib/weather_params.py.backup`

### Web controls not updating
- Check browser console (F12) for JS errors
- Verify Flask is still running in the terminal
- Ensure control keys match between schema and application code

### DMX receivers updating slowly (Linux)
Network send buffer too small. Add to `/etc/sysctl.conf`:
```
net.core.wmem_max=16777216
net.core.wmem_default=16777216
```

### Set not changing after web request
Check terminal for "Weather set change queued". The change applies on the next weather transition (up to ~4 min). The web UI shows a PENDING badge while waiting.

---

## Developer Reference

### Adding a Shader Effect

1. Create a new file in `renderer/effects/` that extends `ShaderEffect` from `base.py`
2. Name the wrapper function with `shader_` prefix and the class with `Effect` suffix — they are auto-discovered
3. See `docs/shader_info.txt` for the full guide covering: depth/alpha blending system, horizontal wrapping, event wrapper pattern, fade in/out, audio reactivity, and a complete template

### Project Structure

```
Stories_OGL.py              # Entry point — wires everything together
lib/
  ambient_audio.py          # Cross-fade ambient track controller
  audio_analyzer.py         # Microphone capture and frequency analysis
  audio_engine.py           # Audio playback engine (streaming + one-shot)
  dmx_sender.py             # sACN/E1.31 DMX pixel sender
  event_scheduler.py        # Pure timed event queue and shared state dict
  midi_controller.py        # Korg nanoKontrol2 MIDI integration
  weather_params.py         # Weather states, presets, and set configs
  weather_set.py            # Active set, event map, and per-set config access
  weather_state.py          # State interpolation and seasonal transitions
renderer/
  shader_renderer.py        # GLFW window + OpenGL rendering loop + display modes
  fan_geometry.py           # Pure-numpy fan/polar geometry (shared by GL and web)
  effects/                  # 40+ individual shader effect modules
    base.py                 # ShaderEffect base class
engine/
  render_pipeline.py        # Hardware integration: renderer + audio + DMX + per-frame loop
web/
  web_controller.py         # Flask web control panel + preview frame streaming
  templates/                # Flask HTML templates (control, editor, preview, admin)
  static/js/preview.js      # WebGL2 live preview client
  static/css/preview.css    # Preview page styles
config/                     # DMX universe and fixture definitions (Unit*.txt)
tools/                      # dj_planner.py, narrative_editor_v2_qt.py, layout_editor.py at the top
                            # level; dj/, editors/, tests/, hardware/, media/ below it
media/sounds/               # Ambient audio files
media/images/               # Image assets
docs/                       # Documentation
bin/                        # Setup and launch scripts
```

### Audio Data (`outstate['sound']`)

Updated at 40 FPS. Contains:
- `raw_bands`: `(1000, 32)` — raw power per frequency band
- `norm_short`: normalized to ~0.25s average (use for beat detection)
- `norm_long`: normalized to ~2.5s average (use for gradual changes)
- `norm_long_relu`: ReLU of norm_long (highlights peaks above baseline)
- `band_centers`: `(32,)` — center frequency of each band (40 Hz – 16 kHz)

Frequency ranges: Bass `[0:8]`, Mids `[8:20]`, Highs `[20:32]`.

### Design Documents

- `docs/CYBERPUNK_EVENTS.md` — Planned effects and events for the Cyberpunk weather set
- `docs/shader_info.txt` — Full shader development guide with templates
