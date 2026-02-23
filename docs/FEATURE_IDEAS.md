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
