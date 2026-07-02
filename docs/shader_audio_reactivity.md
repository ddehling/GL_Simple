# Audio-Reactive Shader Guide

> Read this **only** when writing or modifying an audio-reactive shader effect
> (one that responds to microphone / sound-energy input). General shader rules
> live in [shader_info.txt](shader_info.txt) — read that first.

**Shader effects can respond to real-time audio analysis.**

The audio analyzer processes microphone input at 40 FPS, extracting 32 frequency bands from 40 Hz to 16 kHz. This data is continuously updated in `outstate['sound']` and is available to all shader effects.

## Audio Data Structure

**Access audio data in your shader wrapper function:**

```python
def shader_myeffect(state, outstate, sensitivity=1.0):
    # Get audio analysis from outstate
    audio_data = outstate.get('sound')

    if audio_data is None:
        return  # No audio data available yet

    # Audio data contains:
    # - 'raw_bands': (1000 x 32) array - Raw power in each frequency band
    # - 'norm_short': (1000 x 32) array - Normalized to short-term average (~0.5s)
    # - 'norm_long': (1000 x 32) array - Normalized to long-term average (~2.5s)
    # - 'norm_long_relu': (1000 x 32) array - ReLU(norm_long - 1), highlights above-average
    # - 'band_centers': (32,) array - Center frequency of each band (Hz)
    # - 'band_edges': (33,) array - Edge frequencies defining bands
    # - 'timestamp': float - When data was captured
    # - 'averaging_method': str - 'exponential' or 'mean'
```

## Frequency Band Mapping

The 32 frequency bands span the audible spectrum:

```python
# Bass frequencies (deep, punchy sounds)
BASS_BANDS = slice(0, 8)       # ~40-300 Hz

# Mid frequencies (vocals, guitars, most instruments)
MID_BANDS = slice(8, 20)       # ~300-2000 Hz

# High frequencies (cymbals, hi-hats, brightness)
HIGH_BANDS = slice(20, 32)     # ~2000-16000 Hz

# Sub-bass (very low rumble)
SUB_BASS_BANDS = slice(0, 4)   # ~40-150 Hz

# Upper mids (presence, clarity)
UPPER_MID_BANDS = slice(15, 25) # ~1000-5000 Hz
```

## Data Indexing

**Most recent data is at index [0]:**

```python
# Current frame (most recent)
current_bands = audio_data['raw_bands'][0]        # Shape: (32,)

# One frame ago (1/40th second = 25ms ago)
previous_frame = audio_data['raw_bands'][1]      # Shape: (32,)

# Last second of data (40 frames)
last_second = audio_data['raw_bands'][0:40]      # Shape: (40, 32)

# All available history (up to 25 seconds)
full_history = audio_data['raw_bands']            # Shape: (1000, 32)
```

## Normalization Types

**Choose the normalization that fits your effect:**

```python
# 1. RAW BANDS - Absolute power levels
#    Use for: Effects that need consistent scaling regardless of volume
raw = audio_data['raw_bands'][0]

# 2. SHORT-TERM NORMALIZED - Relative to recent average (~0.5s)
#    Use for: Beat detection, transient response, rhythmic effects
#    Value >1.0 means "louder than recent average"
short_norm = audio_data['norm_short'][0]

# 3. LONG-TERM NORMALIZED - Relative to long average (~2.5s)
#    Use for: Detecting changes in song sections, energy shifts
#    Value >1.0 means "louder than typical"
long_norm = audio_data['norm_long'][0]

# 4. LONG-TERM RELU - Only above-average energy
#    Use for: Highlighting peaks, ignoring sustained sounds
#    Value >0 means "significantly above baseline"
relu = audio_data['norm_long_relu'][0]
```

## Complete Audio-Reactive Example

```python
def shader_audio_circles(state, outstate, bass_sensitivity=1.5, mid_sensitivity=1.0):
    """
    Circles that pulse with different frequency bands

    Usage:
        scheduler.schedule_event(0, 60, shader_audio_circles,
                               bass_sensitivity=2.0, frame_id=0)
    """
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    audio_data = outstate.get('sound')

    if shader_renderer is None:
        print("WARNING: shader_renderer not found!")
        return

    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        print(f"WARNING: viewport {frame_id} not found!")
        return

    # Initialize effect
    if state['count'] == 0:
        print(f"Initializing audio_circles for frame {frame_id}")

        try:
            effect = viewport.add_effect(
                AudioCirclesEffect,
                bass_sensitivity=bass_sensitivity,
                mid_sensitivity=mid_sensitivity
            )
            state['effect'] = effect
            print(f"Initialized shader audio_circles")
        except Exception as e:
            print(f"Failed to initialize audio_circles: {e}")
            import traceback
            traceback.print_exc()
            return

    # Update effect from audio data every frame
    if 'effect' in state and audio_data is not None:
        # Get current normalized bands (use short-term for beat response)
        bands = audio_data['norm_short'][0]

        # Extract frequency ranges
        bass_energy = np.mean(bands[0:8])      # Bass: 40-300 Hz
        mid_energy = np.mean(bands[8:20])      # Mids: 300-2000 Hz
        high_energy = np.mean(bands[20:32])    # Highs: 2000-16000 Hz

        # Apply sensitivity multipliers
        state['effect'].bass_intensity = bass_energy * bass_sensitivity
        state['effect'].mid_intensity = mid_energy * mid_sensitivity
        state['effect'].high_intensity = high_energy

        # Optional: Detect sudden changes (beat detection)
        if len(state.get('prev_bass', [])) > 0:
            bass_delta = bass_energy - state['prev_bass']
            if bass_delta > 0.5:  # Sudden bass increase
                state['effect'].trigger_beat_flash()

        state['prev_bass'] = bass_energy

    # Cleanup
    if state['count'] == -1:
        if 'effect' in state:
            print(f"Cleaning up audio_circles")
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()
            print(f"Cleaned up shader audio_circles")


class AudioCirclesEffect(ShaderEffect):
    """Circles that respond to audio frequency bands"""

    def __init__(self, viewport, bass_sensitivity=1.5, mid_sensitivity=1.0):
        super().__init__(viewport)
        self.bass_sensitivity = bass_sensitivity
        self.mid_sensitivity = mid_sensitivity

        # Audio response parameters (updated from wrapper)
        self.bass_intensity = 0.0
        self.mid_intensity = 0.0
        self.high_intensity = 0.0
        self.beat_flash = 0.0

        self._initialize_circles()

    def _initialize_circles(self):
        """Create three circles for bass, mid, high"""
        self.num_circles = 3
        self.base_positions = np.array([
            [self.viewport.width * 0.25, self.viewport.height * 0.5, 30],  # Bass
            [self.viewport.width * 0.50, self.viewport.height * 0.5, 30],  # Mid
            [self.viewport.width * 0.75, self.viewport.height * 0.5, 30],  # High
        ], dtype=np.float32)

        self.base_sizes = np.array([30, 25, 20], dtype=np.float32)
        self.colors = np.array([
            [1.0, 0.2, 0.2],  # Red for bass
            [0.2, 1.0, 0.2],  # Green for mids
            [0.2, 0.2, 1.0],  # Blue for highs
        ], dtype=np.float32)

    def trigger_beat_flash(self):
        """Trigger a flash on beat detection"""
        self.beat_flash = 1.0

    def update(self, dt: float, state: Dict):
        """Update circle sizes based on audio intensity"""
        if not self.enabled:
            return

        # Scale circles based on audio intensity
        intensities = np.array([
            self.bass_intensity,
            self.mid_intensity,
            self.high_intensity
        ])

        # Clamp intensities to reasonable range (0-3x base size)
        intensities = np.clip(intensities, 0, 3)

        # Update circle sizes
        self.current_sizes = self.base_sizes * (1.0 + intensities)

        # Decay beat flash
        self.beat_flash *= 0.9

    def render(self, state: Dict):
        """Render audio-reactive circles"""
        if not self.enabled:
            return

        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)

        # Set uniforms
        res_loc = glGetUniformLocation(self.shader, "resolution")
        glUniform2f(res_loc, self.viewport.width, self.viewport.height)

        beat_loc = glGetUniformLocation(self.shader, "beatFlash")
        glUniform1f(beat_loc, self.beat_flash)

        # Update instance data with current sizes
        instance_data = np.column_stack([
            self.base_positions,
            self.current_sizes,
            self.colors
        ]).astype(np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferSubData(GL_ARRAY_BUFFER, 0, instance_data.nbytes, instance_data)

        # Draw instanced circles
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, self.num_circles)

        glBindVertexArray(0)
        glUseProgram(0)

    # ... compile_shader, get_vertex_shader, get_fragment_shader, setup_buffers ...
```

## Rhythm & Structure Keys (BeatDetector / AudioStructure)

Beyond the raw band data in `outstate['sound']`, the engine publishes
pre-computed rhythm and song-structure scalars every frame. Prefer these
over rolling your own beat detection - they're PLL-tracked, silence-gated
and shared by every effect. All are plain floats/bools with safe defaults;
read them with `outstate.get(key, 0.0)`.

**Beat / tempo (lib/beat_detector.py):**

| key | type | semantics |
|---|---|---|
| `beat` | bool | True ONLY on the frame a beat fires |
| `beat_decay` | float 1->0 | flash envelope after each beat (tau 0.12s) |
| `bpm` | float | smoothed tempo; **0 until the tracker locks (~3s)** |
| `beat_phase` | float 0..1 | metronome position, snaps to 0 on each beat |
| `beat_intensity` | float >=0 | raw onset strength (not gated to beats) |
| `beat_confidence` | float 0..1 | tempo+phase lock quality; decays in silence |
| `beat_count` | int | monotonic beat index |
| `bar_phase` | float 0..1 | position in a 4-beat bar |
| `phrase_phase` | float 0..1 | position in a 16-beat phrase |

**IMPORTANT - bar/phrase origin is arbitrary.** Beats are counted from
detector start; there is no downbeat detection. Use `bar_phase` /
`phrase_phase` for slow cyclic evolution (palette drift, look rotation),
never for "flash on the One".

**Canonical confidence gating** - beat-locked motion must degrade to a
free-run instead of following a garbage phase. Blend on confidence and
integrate the free phase on the CPU (see club_lasers.py / club_tunnel.py):

```python
# in the effect's update(dt):
if self.conf >= 0.3:
    self.sweep_phase = self.beat_phase
    self._free_phase = self.beat_phase       # seed for seamless fallback
else:
    self._free_phase = (self._free_phase + dt * FREE_RATE) % 1.0
    self.sweep_phase = self._free_phase
```

**Song structure (lib/audio_signals.py):**

| key | type | semantics |
|---|---|---|
| `audio_bass` | float ~0..2 | smoothed bass (bands 0:8) vs 0.5s average; 0 in silence |
| `audio_mid` | float ~0..2 | same for mids (8:20) |
| `audio_high` | float ~0..2 | same for highs (20:32) |
| `bass_punch` | float 0..1 | transient envelope: 0 between hits, ~1 on a hit, ~0.16s release |
| `mid_punch` | float 0..1 | same for mids |
| `high_punch` | float 0..1 | same for highs (hat/cymbal hits) |
| `audio_punch` | float 0..1 | broadband max of the three |
| `audio_energy` | float 0..1 | slow overall loudness; ~0.5 steady groove, 0 in silence |
| `build_level` | float 0..1 | riser detector (energy climb x rising highs) |
| `drop` | bool | True ONLY on the frame a drop fires (quiet bass then slam) |
| `drop_decay` | float 1->0 | punch envelope after each drop (tau 0.35s) |

Unlike the raw `norm_*` bands (which read ~1.0 in silence - tiny/tiny AGC),
these scalars are gated against an absolute level tracker and genuinely go
to 0 when the room is quiet.

**DESIGNING FOR VISIBLE REACTIVITY (read before mapping audio -> visuals):**
the AGC-normalized signals (`norm_*` bands, `audio_bass/mid/high`) HOVER
AROUND 1.0 during steady music - that is what normalization means. Two
mappings that therefore always fail:

1. `brightness = 0.5 + 0.5 * audio_bass` -> a constant near-full wash with
   a few percent of wiggle; the brightness limiter flattens it to nothing.
2. `height = clamp(norm_band, 0, 1)` -> pinned at 1.0 with tiny dips.

What works:
- **Punch envelopes** for anything that should HIT: `x_punch` is 0 between
  hits and ~1 on them - multiply, don't add ("`(0.2 + 0.8*punch)`" reads;
  "`0.8 + 0.2*punch`" doesn't).
- **Deviation expansion** for anything continuous: map
  `clamp((norm - 0.55) / span, 0, 1)` so a resting band sits near ZERO and
  only real energy lights it. Thresholds against norm values must sit
  ABOVE 1.0 (e.g. audio_balls' lightning_threshold 1.3) or they are
  permanently true.
- Keep additive floors under ~0.2 of the output range; temporal contrast
  (dark -> flash) survives the limiter, mid-level wiggle does not.

Offline test for all of the above: `python tools/_club_signals_test.py`.

## Audio Reactivity Best Practices

- Use `norm_short` for beat/rhythm response — responds to transients and beats
- Use `norm_long` for gradual changes — smooths out fast fluctuations
- Use `norm_long_relu` for peak detection — only reacts to above-average sounds
- Check `audio_data is not None` — data may not be available immediately
- Apply sensitivity multipliers — different effects need different scaling
- Clamp audio values — prevent extreme values from breaking visuals
- Store previous values in `state` — enables beat detection and smoothing
- Use appropriate frequency bands — match your effect to relevant frequencies

## Common Audio Patterns

**Beat detection:**
```python
# Compare current to previous frame
current_bass = np.mean(audio_data['norm_short'][0][0:8])
prev_bass = np.mean(audio_data['norm_short'][1][0:8])

if current_bass > prev_bass + 0.5:  # Threshold for "beat"
    trigger_visual_effect()
```

**Smooth audio following:**
```python
# Use exponential smoothing for gradual response
if not hasattr(state['effect'], 'smoothed_energy'):
    state['effect'].smoothed_energy = 0.0

current_energy = np.mean(audio_data['norm_short'][0])
smoothing = 0.1  # 0-1, higher = faster response
state['effect'].smoothed_energy = (
    smoothing * current_energy +
    (1 - smoothing) * state['effect'].smoothed_energy
)
```

**Frequency-specific triggers:**
```python
# Different effects for different frequencies
bass_hit = np.mean(audio_data['norm_long_relu'][0][0:8]) > 0.3
high_hit = np.mean(audio_data['norm_long_relu'][0][24:32]) > 0.2

if bass_hit:
    spawn_heavy_particle()
if high_hit:
    spawn_sparkle()
```

## Performance Considerations

- Audio data is updated at 40 FPS (matches typical frame rate)
- Accessing `outstate['sound']` is very fast (simple dict lookup)
- NumPy operations on band data are highly optimized
- Avoid computing FFTs yourself — use provided band data
- Don't store large audio histories in effect state (use provided 1000-frame buffer)
