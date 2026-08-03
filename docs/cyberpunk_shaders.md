# Cyberpunk Shader Catalog

> Inventory of every shader in the cyberpunk weather set: what it looks like,
> what drives it, and which weather states activate it.

The cyberpunk set has **18 shaders**, split into two layers:

- **State-tied (12)** — driven by per-state weather params (`pollution_level`,
  `neon_intensity`, `rain_rate`, etc.). Each shader renders only when its
  driving param is non-zero in the active state.
- **Narrative-variable (6)** — driven by `outstate['story_*']` values
  published by `NarrativePlayer`. Threshold-gated at variable ≥ 0.20 (invisible
  below), linear scale 0 → 1 as variable climbs 0.20 → 1.0. These run during
  cyberpunk arcs regardless of weather state.

All 18 are registered in [projects/fan/event_map.py](../projects/fan/event_map.py)
and listed in [projects/fan/weather_params.py](../projects/fan/weather_params.py)
under `WEATHER_SETS["cyberpunk"]["background_events"]`.

---

## State-tied shaders

### 1. `cyber_smog_volume` — atmospheric base
**Driver:** `pollution_level` (0..1) · **Color:** `fog_color` (per-state)
**Render priority:** 1.0 (deepest atmospheric layer)

Soft volumetric smog with layered noise turbulence and a settle-to-bottom
gradient. Alpha capped at 0.55 so silhouettes still read through it. The
state's `fog_color` tints the smog — orange-brown for industrial smog
(SMOG_HAZE), acid green for ACID_RAIN, deep blue for night states.

**Appears in (≥ 0.05):**
| pollution | state |
|---|---|
| 1.00 | CYBER_SMOG_HAZE |
| 0.80 | CYBER_ACID_RAIN |
| 0.55 | CYBER_UNDERSIDE_FLOOD, CYBER_BROADCAST_NIGHT |
| 0.40 | CYBER_MIDDEN_MARKET |
| 0.35 | CYBER_DAWN_PIRATE |
| 0.30 | CYBER_HOUR_OF_STATIC |
| 0.25 | CYBER_AR_BLOOM |
| 0.20 | CYBER_TRANSIT_CORRIDOR, CYBER_RELAY_NODE |
| 0.10 | CYBER_CROWN_VAULT |

---

### 2. `cyber_underway_glow` — flooded-BART aesthetic
**Driver:** `cyber_underway_intensity` (0..1)
**Render priority:** 1.5 (just above smog)

Bioluminescent green algae blooms along the left/right edges (tunnel
walls); bright cyan caustic patterns at ankle height across the bottom 20%
of the screen (water surface). Cool teal ambient wash overall. Used to
make a frame feel submerged and underground.

**Appears in:**
| intensity | state |
|---|---|
| 1.00 | CYBER_UNDERSIDE_FLOOD |

---

### 3. `cyber_transit_flow` — tunnel/maglev motion
**Driver:** `cyber_transit_intensity` (0..1)
**Render priority:** 2.0

Three parallax layers of vertical edges sweeping horizontally across the
screen at 30%, 60%, 120% screen-width/second. Distant cool-blue layer is
dim and slow; near white-cyan layer is fast and bright. Concentrated at
mid-height (where the tunnel walls would be). Reads as kinetic motion
through a corridor.

**Appears in:**
| intensity | state |
|---|---|
| 1.00 | CYBER_TRANSIT_CORRIDOR |

---

### 4. `cyber_neon_grid` — perspective floor
**Driver:** `neon_intensity` (0..1) · **Audio:** pulses on `outstate['sound']`
**Render priority:** 5.0

Static cyan neon grid receding to a vanishing horizon at screen y=0.55.
12 vertical lines (radial spokes in fan view, emanating from the bottom-
center) plus 6 horizontal lines (concentric arcs in fan view). Spokes are
~1.3× brighter than arcs because in fan view they radiate from the fan
origin and read more naturally as "perspective floor" than concentric
rings. No scroll — perspective compression makes uniform-rate motion
jarring; audio pulse handles aliveness instead.

**Appears in (≥ 0.30):**
| neon | state |
|---|---|
| 1.00 | CYBER_NEON_CLEAR |
| 0.90 | CYBER_NEON_DRIZZLE, CYBER_MIDDEN_MARKET |
| 0.80 | CYBER_HOLOGRAM_NIGHT |
| 0.70 | CYBER_GLITCH_FOG, CYBER_AR_BLOOM |
| 0.60 | CYBER_DATA_STORM, CYBER_DRONE_PATROL, CYBER_BROADCAST_NIGHT |
| 0.50 | CYBER_ACID_RAIN, CYBER_TRANSIT_CORRIDOR |
| 0.40 | CYBER_SMOG_HAZE, CYBER_DAWN_PIRATE, CYBER_RELAY_NODE |
| 0.30 | CYBER_ELECTRIC_STORM, CYBER_CROWN_VAULT, CYBER_HOUR_OF_STATIC |

---

### 5. `cyber_city_skyline` — silhouetted buildings
**Driver:** `cyber_skyline_density` (0..1) — controls window-light count
**Render priority:** 6.0

24 vertical silhouetted towers with sparse window-light grids (6×18 cells
per building, ~20–65% of cells lit depending on density). Window colors:
mostly warm orange (1.0, 0.65, 0.30), 15% cool cyan (0.30, 0.85, 1.0).
Buildings are zero-output dark bodies; sky band behind them is a
night-blue → magenta gradient weighted by `light_pollution`. Per-window
flicker (~8% drop rate) gives it life. Per the contrast playbook: dark
bodies + bright sparse windows = high-contrast under the limiter.

**Appears in:**
| density | state |
|---|---|
| 0.90 | CYBER_CROWN_VAULT |
| 0.75 | CYBER_MIDDEN_MARKET |
| 0.70 | CYBER_AR_BLOOM |
| 0.60 | CYBER_DAWN_PIRATE, CYBER_HOUR_OF_STATIC |
| 0.55 | CYBER_BROADCAST_NIGHT |
| 0.30 | CYBER_TRANSIT_CORRIDOR |

---

### 6. `cyber_drone_spotlight` — sweeping searchlight
**Driver:** `drone_activity` (0..1)
**Render priority:** 6.5 (in front of skyline)

A volumetric red-amber beam cone sweeps across the scene with a
"pause-and-search" rhythm (sin-of-sin gives uneven motion). Cone widens
with distance from the drone; bright disc-splash where the beam hits the
ground. Vertical dust streaks inside the cone simulate atmospheric
scatter. The drone hovers near the top of the frame.

**Appears in:**
| activity | state |
|---|---|
| 1.00 | CYBER_DRONE_PATROL |
| 0.50 | CYBER_HOLOGRAM_NIGHT |
| 0.30 | CYBER_CROWN_VAULT |
| 0.20 | CYBER_AR_BLOOM |
| 0.15 | CYBER_MIDDEN_MARKET |
| 0.10 | CYBER_DAWN_PIRATE, CYBER_TRANSIT_CORRIDOR, CYBER_BROADCAST_NIGHT |

---

### 7. `cyber_neon_signs` — flickering storefronts
**Driver:** `cyber_signage_density` (0..1)
**Render priority:** 7.0 (in front of skyline, behind holograms)

Up to 48 sign slots in an 8×6 grid; per-slot hash gates each sign on/off
based on density. Signs are small horizontal bars (~2% wide × 0.7% tall)
with soft bloom halos. Each sign has independent flicker timer (0.4–2.9 s
period). Color picked per-sign from a 4-color palette: hot pink, cyan,
acid green, purple. Placed in the upper-middle vertical band (y ∈ 0.30..0.85)
so they hang on building faces.

**Appears in:**
| density | state |
|---|---|
| 0.95 | CYBER_MIDDEN_MARKET |
| 0.50 | CYBER_AR_BLOOM |
| 0.40 | CYBER_DAWN_PIRATE, CYBER_BROADCAST_NIGHT |
| 0.30 | CYBER_HOUR_OF_STATIC |
| 0.20 | CYBER_TRANSIT_CORRIDOR |
| 0.10 | CYBER_CROWN_VAULT |

---

### 8. `cyber_hologram_billboards` — floating advertisements
**Driver:** `hologram_density` (0..1) · also `story_defiance` for glitch rate
**Render priority:** 7.5

Up to 6 translucent rectangular billboards in the upper 2/3 of the screen,
each with hash-driven position, size, color (cyan/pink/lime base hues),
and scrolling content patterns. Bright thin edges + dimmer body. Per-row
hash gate creates ad-corruption stripes when `story_defiance` > 0 —
billboards visibly hijack themselves during defiant story beats.

**Appears in:**
| density | state |
|---|---|
| 1.00 | CYBER_HOLOGRAM_NIGHT, CYBER_AR_BLOOM |
| 0.50 | CYBER_CROWN_VAULT |
| 0.45 | CYBER_MIDDEN_MARKET |
| 0.40 | CYBER_RELAY_NODE |
| 0.30 | CYBER_NEON_CLEAR |
| 0.20 | CYBER_DRONE_PATROL, CYBER_DAWN_PIRATE |
| 0.15 | CYBER_TRANSIT_CORRIDOR, CYBER_BROADCAST_NIGHT |

---

### 9. `cyber_data_rain` — Matrix-style cascading characters
**Driver:** `data_flow_rate` (0..1)
**Render priority:** 8.0

32 vertical character streams across the wrap, each with per-column speed
(0.5–2.5×) and stream length (8–22 rows). Bright white-cyan "head" leads
a fading mid-green tail. Per-cell glyph hash changes 6× per second to
simulate character cycling. Density gates which columns are active. Falls
faster at higher rates.

**Appears in (≥ 0.20):**
| rate | state |
|---|---|
| 1.00 | CYBER_DATA_STORM, CYBER_RELAY_NODE |
| 0.40 | CYBER_GLITCH_FOG, CYBER_CROWN_VAULT, CYBER_AR_BLOOM |
| 0.30 | CYBER_MIDDEN_MARKET, CYBER_TRANSIT_CORRIDOR |
| 0.20 | CYBER_DAWN_PIRATE, CYBER_BROADCAST_NIGHT |

---

### 10. `cyber_rain` — neon-tinted vertical rain
**Driver:** `rain_rate` (0..1) · **Color:** `cyber_rain_color` (per-state)
**Render priority:** 8.2

60 vertical drop tracks across the wrap, sparsity gated by rain_rate.
Each drop is a thin bright streak with a fading upward tail (length
≤ 16% of screen). Per-column random speed, x-jitter, and ±15% brightness
variation. Head shifts toward white (hotter); tail stays palette color.
Phase-integrated fall (safe across rate transitions).

Color comes from per-state `cyber_rain_color`:
- **neon cyan** (0.20, 0.95, 1.00) — CYBER_NEON_DRIZZLE
- **acid green** (0.55, 1.00, 0.20) — CYBER_ACID_RAIN
- **electric blue** (0.30, 0.55, 1.00) — CYBER_ELECTRIC_STORM
- Any future state can set its own (sodium amber, bloodlight red, etc.)

**Appears in:**
| rate | state |
|---|---|
| 0.70 | CYBER_ACID_RAIN |
| 0.50 | CYBER_ELECTRIC_STORM |
| 0.30 | CYBER_NEON_DRIZZLE |

---

### 11. `cyber_electric_storm` — branching electric bolts
**Driver:** `lightning_probability` (0..1) + `electric_interference` (0..1)
**Render priority:** 9.5 (in front of city, behind sparks)

Up to 4 simultaneous bolts, CPU-managed. Each strike spawns stochastically
(rate ≈ probability × 2 strikes/sec); bolt is a jagged vertical line from
sky to a per-bolt ground level (y ∈ 0.55..0.85) with two diagonal fork
branches. White-hot core (~4 px) + electric-blue glow halo (~40 px).
Strike fades quadratically over 0.25–0.45 s.

Interference arcs (jagged horizontal twitches across building tops) fire
when `electric_interference` > 0, at a fixed 0.35 s shuffle period — NOT
30 Hz strobe.

**Appears in:**
| probability | state |
|---|---|
| 1.00 | CYBER_ELECTRIC_STORM |

---

### 12. `cyber_scan_lines` — CRT atmosphere
**Driver:** `scan_line_intensity` (0..1) · also `story_signal` (low boosts glitch)
**Render priority:** 10.5 (near top of stack)

Subtle horizontal scan-line dimming: every other pixel row is slightly
darkened (transparent black overlay, NOT colored). Wide slow vertical-sync
roll band (period ~17 s) drifts downward. Rare chromatic tear flashes (~25%
of 2 s buckets, visible for ~60 ms) — single magenta line, never a
30 Hz strobe. Effect reads as quiet CRT atmosphere, not strobe.

**Appears in (≥ 0.20):**
| intensity | state |
|---|---|
| 0.95 | CYBER_RELAY_NODE |
| 0.80 | CYBER_AR_BLOOM |
| 0.70 | CYBER_CROWN_VAULT |
| 0.55 | CYBER_BROADCAST_NIGHT |
| 0.50 | CYBER_TRANSIT_CORRIDOR |
| 0.40 | CYBER_MIDDEN_MARKET |
| 0.35 | CYBER_DAWN_PIRATE |
| 0.30 | CYBER_DRONE_PATROL |
| 0.20 | CYBER_UNDERSIDE_FLOOD |

---

## Narrative-variable shaders

These read `outstate['story_*']` values published by `NarrativePlayer`.
All are threshold-gated at variable ≥ 0.20 (below that, the shader's
`render()` returns immediately — no draw). Above threshold, the effect
scales linearly from 0 at value=0.20 to full at value=1.0.

### 13. `signal_carrier` — perception coherence
**Variable:** `story_signal`
**Render priority:** 10.0

A single scrolling waveform trace centered at mid-screen (slow sin
composition of three frequencies). At high signal: clean cyan-white,
continuous, steady. At low signal: per-column vertical jitter, random
dropouts (short missing segments), color shifts toward sickly phosphor
green. Single trace, no banding — signal quality is expressed through
the waveform's own coherence.

**Visible when:** cyberpunk narrative is active and the current node sets
`story_signal ≥ 0.20`. In Ghost in the Grid, peaks at 0.85+ during
intimacy/turn beats.

---

### 14. `dread_perimeter` — surveillance pressure
**Variable:** `story_dread`
**Render priority:** 5.5 (background edge layer)

Red/amber edge glow around the entire frame, pulsing with a slow
"breathing" rhythm. Sweeping search-beam line tracks across the screen
at a speed proportional to dread (faster at high dread). Rare alarm
pulses on the periphery (period scales down at high dread). Hot red on
alarm, red-amber otherwise.

**Visible when:** `story_dread ≥ 0.20`. Peaks at 0.85 during consequence
beats in Last Human Job, Listening Glass.

---

### 15. `yearning_gravity` — focal pull toward an absent person/place
**Variable:** `story_yearning`
**Render priority:** 8.5

A single bright warm pink/gold focal point slow-orbits in the upper-right
quadrant (orbit period ~30 s). Three nested intensity rings: bright core
(2% radius), gold bloom (18% radius), rose halo (45% radius). Opposite
corner of the frame darkens slightly (the "gravity" effect — the focal
point pulls visual attention while everything else dims away from it).

**Visible when:** `story_yearning ≥ 0.20`. Peaks at 0.90 during intimacy
beats — the sleeper's room in Ghost in the Grid, the unindexed sleeper
in Listening Glass.

---

### 16. `defiance_inversion` — active pushback
**Variable:** `story_defiance`
**Render priority:** 11.0 (topmost — sparks above everything)

Up to 6 simultaneous electric arcs flash briefly between random endpoints
(jagged line distance metric, ~0.5 s lifecycle each, count scales with
defiance). Localized "counter-content tiles" — small bright squares (~3%
× 2%) at random positions, lifetime ~0.4 × period, hot pink or acid
green. Rare full-screen white flash at period 5 → 2 s as defiance climbs.
**No horizontal bands** — those produce ugly stripes; tiles are
spatially localized instead.

**Visible when:** `story_defiance ≥ 0.20`. Peaks at 0.90+ during turn
beats — the moment of the broadcast in Signal/Noise, the file exfiltration
in Last Human Job.

---

### 17. `dissolution_drift` — fading toward an ending
**Variable:** `story_dissolution`
**Render priority:** 9.0

Bright cool-white particles drift **upward** across the frame on a
scrolling 30×25 grid; each cell has a hash-gated visibility and per-cell
position offset. Slight vertical motion-smear above each particle. Corner
desaturation wash creeps in from the four corners toward the center
(reach scales 0.05 → 0.30 of frame, capped at gray-blue tint). Phase-
integrated drift rate scales with dissolution.

**Visible when:** `story_dissolution ≥ 0.20`. Climbs across an arc as
the protagonist's resources deplete; peaks 0.70+ at stillness beats.

---

### 18. `velocity_streaks` — kinetic motion
**Variable:** `story_velocity` · **Direction:** `velocity_direction` (uv-space)
**Render priority:** 9.5

Short dashes (length ≤ 14% of screen, hard-capped) drift along the arc's
chosen direction. 10–20 rows perpendicular to direction; each row has an
independent dash with its own spawn offset so dashes don't align across
rows. Sparser at low velocity (~45% of rows visible) → denser at high
velocity (~80%). Cool blue-white color. **Not full-width lines** —
that produced ugly horizontal/concentric banding.

Each arc can set `velocity_direction` (unit vector) — e.g. `(1, 0)` for
rightward, `(0, -1)` for upward (Subroutine 9's stairwell ascent).

**Visible when:** `story_velocity ≥ 0.20`. Peaks during chase/transit
beats — courier in Faraday Run, repossessor driving, drone evasion.

---

## State → active shaders cross-reference

For each cyberpunk state, this lists the shaders that produce visible output
(driver param > 0.20). Lower threshold params still render but are subtle.

### Existing states (10)

**CYBER_NEON_CLEAR** (hub state) — `cyber_neon_grid` (full), `cyber_hologram_billboards` (light), `cyber_smog_volume` (light)

**CYBER_NEON_DRIZZLE** — `cyber_neon_grid` (full), `cyber_rain` (cyan, 0.3), `cyber_smog_volume` (light)

**CYBER_DATA_STORM** — `cyber_data_rain` (max), `cyber_neon_grid`, `cyber_smog_volume`

**CYBER_SMOG_HAZE** — `cyber_smog_volume` (max, orange-brown), `cyber_neon_grid` (light)

**CYBER_ELECTRIC_STORM** — `cyber_electric_storm` (max), `cyber_rain` (electric blue, 0.5), `cyber_neon_grid`

**CYBER_ACID_RAIN** — `cyber_rain` (acid green, 0.7), `cyber_smog_volume` (heavy), `cyber_neon_grid`

**CYBER_HOLOGRAM_NIGHT** — `cyber_hologram_billboards` (max), `cyber_neon_grid`, `cyber_drone_spotlight` (light)

**CYBER_BLACKOUT** — most shaders dimmed/off (this is the "city goes dark" state)

**CYBER_GLITCH_FOG** — `cyber_neon_grid`, `cyber_data_rain` (light), `cyber_smog_volume`

**CYBER_DRONE_PATROL** — `cyber_drone_spotlight` (max), `cyber_neon_grid`, `cyber_scan_lines`

### New states (9)

**CYBER_DAWN_PIRATE** — `cyber_smog_volume` (amber), `cyber_city_skyline`, `cyber_neon_grid`, `cyber_neon_signs` (medium), `cyber_scan_lines` (light), `cyber_hologram_billboards` (very light)

**CYBER_CROWN_VAULT** — `cyber_city_skyline` (max, vertical density), `cyber_scan_lines` (heavy), `cyber_hologram_billboards`, `cyber_drone_spotlight` (medium), `cyber_data_rain` (light), `cyber_neon_grid` (subdued)

**CYBER_MIDDEN_MARKET** — `cyber_neon_signs` (max), `cyber_neon_grid`, `cyber_city_skyline`, `cyber_smog_volume`, `cyber_hologram_billboards`, `cyber_data_rain` (light), `cyber_scan_lines`

**CYBER_UNDERSIDE_FLOOD** — `cyber_underway_glow` (max), `cyber_smog_volume` (water-tinted), `cyber_scan_lines` (light). No skyline (`cyber_skyline_density: 0`), no signs.

**CYBER_HOUR_OF_STATIC** — `cyber_city_skyline`, `cyber_smog_volume` (light), `cyber_neon_signs` (subdued), `cyber_neon_grid` (subdued). `cyber_scan_lines: 0` because the Overlay is OFF.

**CYBER_AR_BLOOM** — `cyber_hologram_billboards` (max + glitched), `cyber_scan_lines` (heavy), `cyber_city_skyline`, `cyber_neon_grid`, `cyber_neon_signs`, `cyber_data_rain` (medium)

**CYBER_TRANSIT_CORRIDOR** — `cyber_transit_flow` (max), `cyber_scan_lines`, `cyber_neon_grid`, `cyber_data_rain`, `cyber_city_skyline` (sparse), `cyber_smog_volume` (light)

**CYBER_RELAY_NODE** — `cyber_scan_lines` (max), `cyber_data_rain` (max), `cyber_neon_grid`, `cyber_hologram_billboards`, `cyber_smog_volume` (deep cold). No skyline, no signs — interior location.

**CYBER_BROADCAST_NIGHT** — `cyber_smog_volume` (amber), `cyber_neon_grid`, `cyber_city_skyline`, `cyber_scan_lines`, `cyber_neon_signs`, `cyber_data_rain` (light)

---

## Rendering order (priority, back to front)

| Priority | Shader | Layer |
|---|---|---|
| 1.0 | cyber_smog_volume | atmospheric base |
| 1.5 | cyber_underway_glow | submerged/tunnel atmospheric base |
| 2.0 | cyber_transit_flow | tunnel motion bands |
| 5.0 | cyber_neon_grid | perspective floor |
| 5.5 | dread_perimeter | edge ambient |
| 6.0 | cyber_city_skyline | building silhouettes + windows |
| 6.5 | cyber_drone_spotlight | beam cone |
| 7.0 | cyber_neon_signs | storefront signage |
| 7.5 | cyber_hologram_billboards | floating ads |
| 8.0 | cyber_data_rain | Matrix glyphs |
| 8.2 | cyber_rain | weather rain (tinted) |
| 8.5 | yearning_gravity | focal point |
| 9.0 | dissolution_drift | upward fade particles |
| 9.5 | cyber_electric_storm | bolts |
| 9.5 | velocity_streaks | motion dashes |
| 10.0 | signal_carrier | waveform |
| 10.5 | cyber_scan_lines | CRT overlay |
| 11.0 | defiance_inversion | sparks, tiles, flash |

---

## Files

- Shader implementations: [projects/fan/shaders/cyber_*.py](../projects/fan/shaders/) plus
  [signal_carrier.py](../projects/fan/shaders/signal_carrier.py),
  [dread_perimeter.py](../projects/fan/shaders/dread_perimeter.py),
  [yearning_gravity.py](../projects/fan/shaders/yearning_gravity.py),
  [defiance_inversion.py](../projects/fan/shaders/defiance_inversion.py),
  [dissolution_drift.py](../projects/fan/shaders/dissolution_drift.py),
  [velocity_streaks.py](../projects/fan/shaders/velocity_streaks.py)
- Event-map registration: [projects/fan/event_map.py](../projects/fan/event_map.py)
- State presets + parameters: [projects/fan/weather_params.py](../projects/fan/weather_params.py)
- Narrative-variable source: [renderer/effects/narrative_player.py](../renderer/effects/narrative_player.py)
  publishes `outstate['story_*']` keys for each `variables[]` entry in the
  active narrative script ([projects/fan/media/sounds/cyberpunk/sounds.json](../projects/fan/media/sounds/cyberpunk/sounds.json))
