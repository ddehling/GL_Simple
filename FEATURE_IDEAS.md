# Feature Ideas

Brainstormed feature candidates for GL_Simple. Not prioritized — use this as a backlog to draw from.

---

## Audio / Reactivity

### Beat Detection
Tap a tempo or auto-detect BPM from the microphone input. Sync visual transitions, effect swaps, and parameter pulses to the beat rather than just raw amplitude or frequency bands.

### Audio Cue Triggering
Trigger specific effects when a particular sound cue plays (e.g. a gunshot cue triggers lightning, a thunder crack triggers rain). Goes beyond continuous mic reactivity — more like show control. Could integrate with the event_map system.

### Spectrum Waterfall Shader
A shader effect showing frequency content over time as a scrolling heatmap / waterfall. Visualizes FFT history already captured in `MicrophoneAnalyzer.get_spectrum_history()`.

### Ambient Audio Reactivity
React to the ambient audio being played back (loopback capture or analysis of the decoded PCM), not just the external microphone. Effects could pulse in sync with the music rather than audience noise.

### Named Ambient Layers
Allow multiple simultaneous ambient tracks in independent named slots, so e.g. a base environment sound and a weather overlay can be controlled separately without interfering.

```python
engine.play_ambient(forest_path, layer="base")
engine.play_ambient(rain_path,   layer="weather")
engine.stop_ambient(layer="weather")
```

Currently `play_ambient` enforces a single ambient slot — the mixer already supports multiple tracks, it's just the command handler that would need updating.

### Seamless Loop Crossfade
Short crossfade (e.g. 100ms overlap) at the loop boundary so files that don't start/end at silence don't produce an audible click when they loop.

### Multi-Speaker / Positional Audio
Output to a large number of unique speakers to enable positional audio. Suggested by Dieter as a direction to grow toward — not currently implemented.

---

## Visuals / Shaders

### GPU Particle System
A shader effect implementing a large particle system (thousands of particles) driven by physics parameters (gravity, wind, turbulence) and modulated by audio amplitude/frequency. More flexible than individual bespoke effects like rain or fireflies.

### Video / Texture Input Pass
Accept a video file or live camera feed as an input texture, then run it through a shader pass for stylization, color grading, or compositing with other effects.

### Movie Playback with Audio (A/V Sync)
`renderer/effects/movie.py` was previously removed. If re-implemented, the clean approach with the current audio engine:
1. Extract the audio track from the video file to a temp WAV at load time (moviepy or ffmpeg)
2. Call `engine.schedule_event(temp_wav, skip_seconds=start_time)` at the same moment video playback begins
3. Both run on independent clocks — sync stays naturally tight

Video frames stay as OpenCV frames uploaded to an OpenGL texture. This eliminates the old manual chunk-pushing approach that was never wired up to any weather state.

### Day/Night Cycle
A gradual skybox/background transition tied to either wall-clock time or an internal simulation timer. Each weather set could define its own dawn/dusk color palette. Complements existing `season` / time-of-day parameters.

### Crowd Silhouette Layer
A composited layer of stylized human silhouettes rendered in front of world effects — good for concert and crowd atmosphere. Could be driven by audio energy (swaying, jumping).

### Reaction-Diffusion (Gray-Scott)
A generative texture effect using the Gray-Scott reaction-diffusion model, producing organic, morphing patterns. Parameters (feed rate, kill rate) map well to weather intensity or audio input.

---

## Weather / Environment

### Time-of-Day Parameter
Each weather set defines morning / afternoon / night variants. The system auto-advances through them on a configurable sim clock, adjusting ambient color, effect intensity, and audio.

### Continuous Intensity Slider
Replace discrete weather state transitions with a continuous 0–1 intensity control per set (calm → intense). States become waypoints on the continuum rather than hard cuts.

### Web-Based Weather Set Builder
A UI in the web control panel to create, edit, and save weather sets and states without editing Python source. Serializes to the existing `weather_params.py` format or a JSON equivalent.

---

## Web Control Panel

### Live Preview Thumbnail
Capture a screenshot of the rendered frame every few seconds and display it as a thumbnail in the web UI. Lets operators see what's on screen without being in the room.

### Effect Sequencer / Show Playlist
A drag-and-drop timeline in the web UI to queue effects with durations — a simple show playlist. Integrates with `EventScheduler` to fire cues at the right times.

### Mobile-Friendly Layout
Responsive / touch-optimized layout for the web control panel. Better touch targets, larger sliders, swipe gestures — designed to be used from a phone while moving around a venue.

### OSC Input
Receive Open Sound Control (OSC) messages from Ableton Live, QLab, or other show control software. Map OSC addresses to weather state changes, effect triggers, or parameter updates for tight show integration.

---

## DMX / Lighting

### Fixture Zone Grouping
Group DMX fixtures into named zones (stage left, stage right, upstage wash, etc.) and control each zone independently with separate color and intensity parameters.

### Chase Sequences
Programmable pixel chase / sweep patterns that run independently of the OpenGL renderer. Useful for simple running-light effects without needing a full shader.

### Per-Fixture Color Calibration
Store RGB correction profiles per fixture unit to compensate for hardware variation. Applied as a final transform in `dmx_sender.py` before values are written to sACN.

---

## Performance / Infrastructure

### Config Hot-Reload
Fully robust reload of weather params and event map without restarting the process. The web panel already has a partial version (`reload_weather_module`); make it reliable enough to use mid-show.

### Show Recording
Record rendered frames to a video file (e.g. via ffmpeg pipe or OpenGL pixel readback) for post-event review, archival, or live streaming.

### Multi-Output / Split Effects
Run two independent effects simultaneously on separate display outputs or windows — e.g. a world effect on the main display and a different ambient effect on a secondary monitor or projector.
