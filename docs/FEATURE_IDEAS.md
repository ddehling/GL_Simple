# Feature Ideas

Loose collection of ideas to revisit. Not prioritized.

---

## Audio

### Named ambient layers
Allow multiple simultaneous ambient tracks in independent named slots, so e.g. a base environment sound and a weather overlay can be controlled separately without interfering.

```python
engine.play_ambient(forest_path, layer="base")
engine.play_ambient(rain_path,   layer="weather")
engine.stop_ambient(layer="weather")
```

Currently `play_ambient` enforces a single ambient slot — the mixer already supports multiple tracks, it's just the command handler that would need updating.

### Seamless loop crossfade
Short crossfade (e.g. 100ms overlap) at the loop boundary so files that don't start/end at silence don't produce an audible click when they loop.

### Movie playback with audio
`renderer/effects/movie.py` was removed. It used OpenCV for video frames and pushed decoded audio chunks manually to the engine to maintain A/V sync — a complex approach that was never wired up to any weather state.

If re-implemented, the clean approach with the current audio engine:
1. Extract the audio track from the video file to a temp WAV at load time (moviepy or ffmpeg)
2. Call `engine.schedule_event(temp_wav, skip_seconds=start_time)` at the same moment video playback begins
3. Both run on independent clocks at the same sample/frame rate — sync stays naturally tight

This eliminates all chunk-pushing logic. The video side stays as OpenCV frames uploaded to an OpenGL texture, same as before.
