# Korg nanoKontrol2 MIDI Controller Integration

This system provides real-time tracking of your Korg nanoKontrol2 USB MIDI controller, maintaining a state dictionary with current values for all knobs, sliders, and buttons.

## Features

- **Automatic device detection**: Finds and connects to your nanoKontrol2
- **Real-time state tracking**: Maintains current value of every control
- **Change detection**: Identifies which controls changed each frame
- **Callback system**: Register functions to respond to specific control changes
- **Background threading**: Optional continuous reading in separate thread
- **Normalized values**: Knobs/sliders return 0.0-1.0, buttons return True/False

## Installation

Install the required dependency:

```bash
pip install pygame
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from lib.midi_controller import KorgNanoKontrol2

# Create and connect to controller
midi = KorgNanoKontrol2(auto_connect=True)

# Start background reading thread
midi.start_reading()

# Get current values
knob1_value = midi.get_knob(1)          # Returns 0.0-1.0
slider3_value = midi.get_slider(3)      # Returns 0.0-1.0
button_pressed = midi.get_button('s', 1) # Returns True/False

# Get all values at once
all_values = midi.get_all_values()
print(all_values['knob_1'])
print(all_values['slider_3'])
print(all_values['s_button_1'])
```

### In Your Main Loop

```python
# Call this every frame to update MIDI state
changes = midi.update()

# Check what changed this frame
if 'knob_1' in changes:
    print(f"Knob 1 changed to: {changes['knob_1']}")

# Access current state
current_knob1 = midi.state['knob_1']
```

### With Callbacks

```python
# Register callback for specific control
def on_slider1_change(control_name, value):
    print(f"Slider 1: {value:.3f}")
    # Update your shader parameter here
    scheduler.state['some_param'] = value

midi.register_callback('slider_1', on_slider1_change)
```

## Controller Layout

### Korg nanoKontrol2 (Factory Settings)

```
Channel:      1      2      3      4      5      6      7      8
           ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
Knobs:     │  ◯  │  ◯  │  ◯  │  ◯  │  ◯  │  ◯  │  ◯  │  ◯  │
           └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
           ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
Sliders:   │  |  │  |  │  |  │  |  │  |  │  |  │  |  │  |  │
           │  |  │  |  │  |  │  |  │  |  │  |  │  |  │  |  │
           └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
           ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
S buttons: │  S  │  S  │  S  │  S  │  S  │  S  │  S  │  S  │
M buttons: │  M  │  M  │  M  │  M  │  M  │  M  │  M  │  M  │
R buttons: │  R  │  R  │  R  │  R  │  R  │  R  │  R  │  R  │
           └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

Transport controls:
  ⏮ ⏭   ↻   ●   ◄   ■   ►   ●
 Prev Next Cycle Set Rew Stop Fwd Rec
```

## Control Names

### Knobs (1-8)
- `knob_1` through `knob_8`
- Values: 0.0 to 1.0 (normalized)

### Sliders (1-8)
- `slider_1` through `slider_8`
- Values: 0.0 to 1.0 (normalized)

### Channel Buttons (1-8 each)
- `s_button_1` through `s_button_8` (Solo)
- `m_button_1` through `m_button_8` (Mute)
- `r_button_1` through `r_button_8` (Record)
- Values: True (pressed) or False (released)

### Transport Buttons
- `track_prev`, `track_next`
- `cycle`
- `marker_set`, `marker_prev`, `marker_next`
- `rewind`, `forward`
- `stop`, `play`, `record`
- Values: True (pressed) or False (released)

## Integration Examples

### Example 1: Control Shader Parameters

```python
from lib.midi_controller import KorgNanoKontrol2

# Setup
midi = KorgNanoKontrol2()
midi.start_reading()

# In your render loop:
def update_frame(state, outstate):
    # Update MIDI state
    midi.update()
    
    # Use MIDI values to control shader parameters
    state['fog_strength'] = midi.get_knob(1)
    state['rain_intensity'] = midi.get_slider(1)
    state['firefly_density'] = midi.get_knob(2) * 5.0  # Scale to 0-5
    
    # Toggle effects with buttons
    if midi.get_button('s', 1):
        state['enable_lightning'] = True
```

### Example 2: With EventScheduler

```python
# In Stories_OGL.py or similar

class EnvironmentalSystem:
    def __init__(self, scheduler):
        # ... your existing init code ...
        
        # Add MIDI controller
        self.midi = KorgNanoKontrol2(auto_connect=True)
        self.midi.start_reading()
        
        # Store in scheduler state for access in effects
        self.scheduler.state['midi'] = self.midi
    
    def update(self):
        # Update MIDI state each frame
        changes = self.midi.update()
        
        # Store current values in scheduler state
        self.scheduler.state['midi_values'] = self.midi.get_all_values()
        
        # React to specific changes
        if 'play' in changes and changes['play']:
            # Play button pressed - trigger something
            self.scheduler.schedule_event(0, 10, fx.shader_lightning, frame_id=0)
```

### Example 3: Using Callbacks

```python
midi = KorgNanoKontrol2()

# Setup callbacks for all knobs
for i in range(1, 9):
    def make_callback(index):
        def callback(name, value):
            print(f"Knob {index}: {value:.3f}")
            # Update your parameter here
        return callback
    
    midi.register_callback(f'knob_{i}', make_callback(i))

# Setup transport button callbacks
def on_play(name, pressed):
    if pressed:
        print("Play button pressed!")
        # Start something

midi.register_callback('play', on_play)

midi.start_reading()
```

### Example 4: Map to Weather System

```python
def setup_weather_controls(midi, env_system):
    """Map transport buttons to weather presets"""
    
    def on_button(weather_state):
        def callback(name, pressed):
            if pressed:
                env_system.transition_to_weather(weather_state, 5.0)
        return callback
    
    # Map transport buttons to weather states
    midi.register_callback('marker_set', on_button(WeatherState.CLEAR))
    midi.register_callback('marker_prev', on_button(WeatherState.RAIN))
    midi.register_callback('marker_next', on_button(WeatherState.HEAVY_FOG))
    midi.register_callback('cycle', on_button(WeatherState.SANDSTORM))
```

## API Reference

### Class: KorgNanoKontrol2

#### Methods

**`__init__(device_name="nanoKONTROL2", auto_connect=True)`**
- Initialize controller
- `device_name`: Partial name to match (case-insensitive)
- `auto_connect`: Auto-connect if device found

**`connect() -> bool`**
- Manually connect to device
- Returns True if successful

**`disconnect()`**
- Disconnect from device

**`update() -> Dict[str, any]`**
- Read MIDI messages and update state
- Returns dictionary of controls that changed this frame

**`get(control_name: str) -> float | bool`**
- Get current value of any control
- Returns 0.0-1.0 for knobs/sliders, True/False for buttons

**`get_knob(index: int) -> float`**
- Get knob value (1-8)
- Returns 0.0 to 1.0

**`get_slider(index: int) -> float`**
- Get slider value (1-8)
- Returns 0.0 to 1.0

**`get_button(button_type: str, index: int) -> bool`**
- Get button state
- `button_type`: 's', 'm', or 'r'
- `index`: 1-8
- Returns True if pressed

**`get_transport(button_name: str) -> bool`**
- Get transport button state
- Returns True if pressed

**`get_all_values() -> Dict[str, any]`**
- Get dictionary of all current control values

**`register_callback(control_name: str, callback: Callable)`**
- Register callback for control changes
- Callback receives (control_name, value)

**`start_reading()`**
- Start background thread for continuous MIDI reading

**`stop_reading()`**
- Stop background thread

**`list_devices()`**
- Print all available MIDI input devices

**`print_state()`**
- Print current state of all controls (for debugging)

#### Attributes

**`state: Dict[str, any]`**
- Current value of every control
- Key: control name (e.g., 'knob_1', 'slider_3', 's_button_1')
- Value: float (0.0-1.0) or bool (True/False)

## Troubleshooting

### Device Not Found

```python
# List all available MIDI devices
midi = KorgNanoKontrol2(auto_connect=False)
midi.list_devices()

# Try connecting manually with partial name
midi.device_name = "nano"  # or "KONTROL" or "nK2"
midi.connect()
```

### Values Not Updating

Make sure you're calling `update()` every frame:

```python
# In your main loop
while running:
    changes = midi.update()  # Must call this!
    # ... render code ...
```

Or use background thread mode:

```python
midi.start_reading()  # Updates happen automatically in background
```

### Button Values Stuck

Some buttons send Note On messages instead of CC. The handler supports both, but if you have issues, check your nanoKontrol2's control mode settings (hold CYCLE + TRACK buttons on power-on to enter setup mode).

## Testing

Run the standalone test:

```bash
python lib/midi_controller.py
```

Or run the integration example:

```bash
python tools/hardware/midi_integration_example.py
```

Move controls on your nanoKontrol2 to see real-time updates.

## Performance Notes

- Background thread mode (`start_reading()`) polls at ~1000 Hz
- Manual mode (`update()`) only processes messages when called
- No significant CPU overhead in either mode
- State dictionary updates are instant (no latency)

## License

Part of the GL_Simple project.
