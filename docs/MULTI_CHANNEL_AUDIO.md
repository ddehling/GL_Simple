# Multi-Channel Audio Design

## Goal

Control where sound plays across output channels and set per-channel volume,
enabling spatial audio routing for stage/venue setups.

## Decision Points

### 1. Number of output channels

`miniaudio.PlaybackDevice` accepts any `nchannels` the hardware supports.
Change `CHANNELS = 2` in `lib/audio_engine.py` to match the physical setup.

Common choices for a stage system:
- **4** — quad (front-left, front-right, rear-left, rear-right)
- **8** — multi-zone venue PA

The mixer buffer in `_mixer()` becomes shape `(required_frames, CHANNELS)` automatically.

### 2. API style — gains array vs named zones

**Option A: Per-channel gains array** — simple and flexible
```python
play_ambient(path, channel_gains=[1.0, 0.0, 1.0, 0.0])
```

**Option B: Named zones** — nicer for weather state params
```python
ZONES = {
    "all":         [1.0, 1.0, 1.0, 1.0],
    "stage_left":  [1.0, 0.0, 0.0, 0.0],
    "stage_right": [0.0, 1.0, 0.0, 0.0],
    "front":       [1.0, 1.0, 0.0, 0.0],
    "rear":        [0.0, 0.0, 1.0, 1.0],
}
play_ambient(path, zone="stage_left")
```

Both can coexist — zones are sugar on top of the gains array.

### 3. Stereo source upmix

All audio files are 2-channel. To route to N output channels, average both
source channels to mono then scale by per-output gains:

```python
# chunk: (n, 2)   gains: (CHANNELS,)
mono   = chunk.mean(axis=1, keepdims=True)   # (n, 1)
output = mono * gains[np.newaxis, :]          # (n, CHANNELS)
```

For stereo field preservation across output pairs (e.g. L→ch0,ch2 / R→ch1,ch3)
use a full `(2, CHANNELS)` matrix multiply instead — but mono upmix is simpler
and works well for ambient.

## Implementation Checklist

| File | Change |
|------|--------|
| `lib/audio_engine.py` | `CHANNELS = N` |
| `lib/audio_engine.py` | Add `ZONES` dict |
| `lib/audio_engine.py` | `_Track.__init__` accepts `channel_gains` |
| `lib/audio_engine.py` | `_Track.read()` outputs `(n, CHANNELS)` via mono upmix |
| `lib/audio_engine.py` | `play_ambient` / `schedule_event` accept `channel_gains` or `zone` |
| `Stories_OGL.py` | Pass `channel_gains` / `zone` when calling `play_ambient` |
| `projects/<id>/weather_params.py` | Add `zone` or `channel_gains` to per-state params |
