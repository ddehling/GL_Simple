# Cyberpunk Weather Set - Event Design Document

This document outlines proposed visual effects and events for the Cyberpunk Metropolis weather set.

## Background Events (Continuous - 10E9 seconds)

These events run continuously when a cyberpunk weather set is active, providing the persistent visual atmosphere.

### 1. neon_grid
**Purpose:** Ground-level illuminated street grid with pulsing neon lines

**Parameters Used:**
- `neon_intensity` - Controls brightness of grid lines
- `light_pollution` - Affects overall glow intensity

**Visual Description:**
- Glows brighter/dimmer based on city activity
- Grid lines pulse with audio input for reactive beats
- Hexagonal or rectangular grid pattern
- Colors: Cyan, magenta, electric blue

**Implementation Notes:**
- Could use shader with distance-based brightness falloff
- Pulse rate synced to `sound` input from analyzer

---

### 2. hologram_billboards
**Purpose:** Floating 3D advertisements and signage

**Parameters Used:**
- `hologram_density` - Number of billboards visible
- `glitch_probability` - Chance of corruption/distortion

**Visual Description:**
- Animated textures cycling through advertisements
- Glitch effects when `glitch_probability` is high
- Semi-transparent projected images
- Could display scrolling text, rotating 3D objects
- Flicker and scan-line artifacts

**Implementation Notes:**
- Billboard positions randomized but persistent
- Texture atlas for different ad content
- Alpha blending for transparency
- Distortion shader for glitch effects

---

### 3. data_rain
**Purpose:** Matrix-style cascading characters/symbols

**Parameters Used:**
- `data_flow_rate` - Speed of falling characters
- `electric_interference` - Intensity variation

**Visual Description:**
- Green/cyan color (#00FF41, Matrix green)
- Vertical streams of changing characters
- Characters: numbers, letters, symbols, Japanese katakana
- Trails fade as they fall
- Random column start times

**Implementation Notes:**
- Character texture atlas
- Particle system or instanced rendering
- Each column has independent speed variation
- Brightness varies with `electric_interference`

---

### 4. city_fog
**Purpose:** Volumetric fog with colored neon lighting

**Parameters Used:**
- `pollution_level` - Base density
- `fog` - Standard fog amount
- `neon_intensity` - Light scattering color
- `light_pollution` - Overall tint

**Visual Description:**
- Not standard gray fog - tinted by neon sources
- Swirls and eddies around buildings
- Colored by nearby light pollution (blues, pinks, purples)
- Depth-based density
- Animated movement patterns

**Implementation Notes:**
- Different from standard `shader_fog`
- Multiple color layers
- Perlin noise for movement
- Light scattering from neon sources

---

### 5. scan_lines
**Purpose:** CRT monitor horizontal line effect overlay

**Parameters Used:**
- `scan_line_intensity` - Visibility of effect

**Visual Description:**
- Horizontal lines across entire display
- Subtle rolling interference pattern
- Intensifies during glitch events
- RGB chromatic aberration on edges
- Occasional vertical sync roll

**Implementation Notes:**
- Post-processing effect applied last
- Simple sine wave pattern
- Screen-space shader
- Very low performance cost

---

### 6. neon_signs
**Purpose:** Flickering storefronts and advertisements

**Parameters Used:**
- `neon_intensity` - Overall brightness

**Visual Description:**
- Random on/off states for individual signs
- Occasional flicker/buzz effects
- Japanese, English, Chinese characters
- Colors: Hot pink (#FF006E), cyan (#00F5FF), purple (#BF00FF)
- Tube-style neon glow with bloom

**Implementation Notes:**
- Each sign has independent flicker timer
- Bloom/glow post-processing
- Could use texture sprites or vector shapes
- Randomized placement along "buildings"

---

### 7. ambient_particles
**Purpose:** Floating debris, steam, smoke, sparks

**Parameters Used:**
- `wind_speed` - Drift direction and speed
- `pollution_level` - Particle density

**Visual Description:**
- Industrial particles rising from vents
- Mix of bright particles (sparks, embers) and dark (soot, ash)
- Steam billowing from grates
- Affected by wind direction
- Denser near ground during high pollution

**Implementation Notes:**
- Particle system with multiple emitters
- Different particle types (bright/dark)
- Wind vector affects velocity
- Alpha blending for smoke/steam
- Additive blending for sparks

---

## Limited Time Events (Temporary - 20-100 seconds)

These events are triggered randomly or by specific weather conditions for dramatic visual moments.

### 1. cyber_lightning
**Purpose:** Electric arc discharge

**Trigger Conditions:**
- Primary: CYBER_ELECTRIC_STORM state
- Random chance during high `lightning_probability`

**Duration:** 0.5-2 seconds per strike

**Visual Description:**
- Branching blue/white/cyan lightning bolts
- Multiple parallel arcs
- Causes brief screen flicker/glitch on strike
- Power surge effect with bloom
- Afterglow/retina burn effect

**Audio:**
- Sharp electrical crack
- Reverberating thunder
- Static interference

**Implementation Notes:**
- Similar to existing `shader_lightning`
- Modified color palette (blue instead of yellow)
- Screen shake/distortion on strike
- Brief `electric_interference` spike

---

### 2. glitch_wave
**Purpose:** Reality distortion sweep across screen

**Trigger Conditions:**
- Random when `glitch_probability` > 0.5
- Chance: 1/1000 per frame

**Duration:** 2-5 seconds

**Visual Description:**
- Horizontal scan line sweeping top to bottom
- RGB channel separation (chromatic aberration)
- Pixel displacement/scrambling in wave
- Vertical jitter along scan line
- Digital artifacts and blockiness

**Audio:**
- Digital static/corruption sound
- Pitched whine

**Implementation Notes:**
- Screen-space post-process shader
- UV displacement based on noise
- RGB channel offset
- Sweeping mask for effect region

---

### 3. drone_spotlight
**Purpose:** Police/security drone searchlight sweep

**Trigger Conditions:**
- Primary: CYBER_DRONE_PATROL state
- Random during `drone_activity` > 0.5

**Duration:** 15-30 seconds

**Visual Description:**
- Sweeping cone of light across scene
- Red or white beam
- Can "pause" and "search" areas
- Dust particles visible in beam
- Lens flare at source
- Red targeting reticle when "locked"

**Audio:**
- Drone motor hum
- Servo/gimbal movement sounds
- Alarm when "detecting"

**Implementation Notes:**
- Cone/spotlight shader
- Raymarching for volumetric beam
- Bezier path for sweep motion
- Particle system for dust in beam

---

### 4. hologram_malfunction
**Purpose:** Advertisement corruption and failure

**Trigger Conditions:**
- Random when `hologram_density` > 0.5
- Chance: 1/5000 per frame per hologram

**Duration:** 10-15 seconds

**Visual Description:**
- Hologram flickers rapidly
- Color channels separate and drift
- Partial collapse then reformation
- Scan lines through corrupted sections
- Pixel dropout
- Image stretching/tearing

**Audio:**
- Electrical buzzing
- Digital corruption sounds

**Implementation Notes:**
- Per-billboard effect
- UV scrambling
- Alpha modulation for flicker
- Color channel displacement

---

### 5. power_surge
**Purpose:** Electrical overload flash

**Trigger Conditions:**
- Random during CYBER_ELECTRIC_STORM
- Can occur during state transitions
- Chance: 1/2000 per frame

**Duration:** 1-3 seconds (flash + recovery)

**Visual Description:**
- Brief intense white/blue screen flash
- All neon signs flash simultaneously
- Screen blooms heavily
- Quick fade to dimmer state
- Gradual recovery to normal brightness

**Audio:**
- Loud electrical crack
- Transformer exploding sound
- Crackling/sizzling

**Implementation Notes:**
- Screen-space brightness multiplication
- Brief HDR spike
- All light sources pulse together
- Bloom intensity spike

---

### 6. data_overflow
**Purpose:** Cascading data flood

**Trigger Conditions:**
- Primary: CYBER_DATA_STORM state
- Random when `data_flow_rate` > 0.7

**Duration:** 20-30 seconds

**Visual Description:**
- Massive increase in data rain density
- Screen fills with green characters
- Falls faster than normal
- "Entering the Matrix" aesthetic
- Screen tint shifts to green
- Some columns reverse direction

**Audio:**
- Digital waterfall sound
- Layered typing/keyboard sounds
- Data transmission noises

**Implementation Notes:**
- Temporarily multiply particle count
- Increase fall speed parameter
- Green color filter overlay
- Return to normal gradually

---

### 7. police_chase
**Purpose:** Emergency vehicle lighting

**Trigger Conditions:**
- Random during CYBER_DRONE_PATROL or CYBER_NEON_CLEAR
- Chance: 1/8000 per frame

**Duration:** 5-10 seconds

**Visual Description:**
- Rapid alternating red and blue light washes
- Movement direction (passes left to right or vice versa)
- Doppler-shifted audio
- Light intensity fades with distance
- Multiple vehicles in sequence possible

**Audio:**
- Siren (approaching, passing, receding)
- Doppler effect on pitch

**Implementation Notes:**
- Animated directional light source
- Color alternates red/blue at 2-3 Hz
- Position moves across screen
- Distance-based intensity falloff

---

### 8. neon_cascade_failure
**Purpose:** Sequential light shutdown

**Trigger Conditions:**
- Can precede CYBER_BLACKOUT state transition
- Random rare event, chance: 1/15000 per frame

**Duration:** 8-12 seconds

**Visual Description:**
- Neon signs turning off in sequence
- Darkness spreading across scene (ripple or sweep)
- Brief moment of total blackout
- Gradual recovery or stay dark (if transitioning to BLACKOUT)
- Sparks/flashes during shutdown

**Audio:**
- Sequential power-down sounds
- Electrical pops
- Fading hum
- Emergency alert tone

**Implementation Notes:**
- Timed sequence of light deactivations
- Region-based shutdown pattern
- `neon_intensity` parameter override
- Can trigger state transition

---

### 9. acid_puddles
**Purpose:** Glowing toxic rain accumulation

**Trigger Conditions:**
- Primary: CYBER_ACID_RAIN state
- Spawns during rain

**Duration:** 40-60 seconds per puddle

**Visual Description:**
- Yellow/green glowing spots on ground level
- Bubbling/steaming effect
- Caustic texture animation
- Grows during rain, evaporates after
- Emits colored light upward

**Audio:**
- Sizzling/hissing
- Bubbling liquid sounds

**Implementation Notes:**
- Decal system or ground texture overlay
- Animated caustic noise
- Particle emitters for steam
- Point lights for glow
- Size increases over time then shrinks

---

### 10. billboard_hack
**Purpose:** Advertisement takeover/hijacking

**Trigger Conditions:**
- Random during high `hologram_density`
- Chance: 1/6000 per frame

**Duration:** 15-25 seconds

**Visual Description:**
- Hologram suddenly glitches and changes content
- Displays counter-culture messages, rebellion symbols
- Skull logo, anarchy symbol, cryptic text
- Glitch aesthetic transition effect
- Returns to normal ads with corruption artifacts

**Audio:**
- Digital intrusion sound
- Static burst
- Alert tone

**Implementation Notes:**
- Billboard texture swap
- Transition with glitch shader
- Alternative texture set (hacked content)
- Return transition

---

### 11. window_lights
**Purpose:** Building window pattern animation

**Trigger Conditions:**
- Random rare event
- Seasonal/time-based patterns
- Chance: 1/10000 per frame

**Duration:** 30-45 seconds

**Visual Description:**
- Mass synchronized window light changes
- Creates patterns or messages in building faces
- Various colors cycling (not just white)
- Could spell words like "OBEY" or show corporate logos
- Wave patterns, checkerboards, animations

**Audio:**
- Subtle electrical hum increase
- Synchronized switch clicks

**Implementation Notes:**
- Building facade texture with controllable pixels
- Pattern generator (text rendering, shapes)
- Color cycle shader
- Grid-based animation

---

### 12. smog_bank
**Purpose:** Rolling wall of pollution

**Trigger Conditions:**
- Primary: CYBER_SMOG_HAZE state
- Random during high `pollution_level`

**Duration:** 35-50 seconds

**Visual Description:**
- Thick wall of fog rolling through scene
- Reduces visibility significantly as it passes
- Orange/brown color (rust, industrial)
- Turbulent motion within
- Objects obscured then revealed

**Audio:**
- Deep rumbling
- Wind sounds
- Muffled city ambience while inside

**Implementation Notes:**
- Traveling fog region
- Density gradient (leading edge, core, trailing edge)
- Noise-based internal motion
- Visibility reduction shader

---

## Seasonal/Rare Events

Ultra-rare events with high visual impact, tied to seasonal timing or very low probability.

### 1. corporate_announcement
**Purpose:** Massive holographic propaganda broadcast

**Trigger Conditions:**
- Seasonal event (specific times of year)
- Very rare random: 1/50000 per frame

**Duration:** 25-35 seconds

**Visual Description:**
- Giant projected face/figure appears in sky
- CEO, government official, or AI entity
- 1984-style "Big Brother" aesthetic
- Monochrome or single color (blue tint)
- Lips sync with audio message
- Everyone must watch and listen

**Audio:**
- Deep authoritative voice
- Echo/reverb
- Corporate jingle before/after
- Propaganda slogans

**Implementation Notes:**
- Large quad or sphere-mapped texture
- Animated facial features (lip sync)
- Audio playback synchronized
- Overrides most other effects while active
- Screen darkens/focuses attention

---

### 2. augmented_reality_glitch
**Purpose:** AR layer malfunction revealing impossible geometry

**Trigger Conditions:**
- Seasonal event
- Tied to `season_preference` specific values

**Duration:** 45-60 seconds

**Visual Description:**
- Virtual objects bleeding into real world
- Impossible geometry (Escher-like structures)
- Recursive patterns
- Objects existing in multiple states
- Non-Euclidean architecture glimpses
- Reality tears/seams visible

**Audio:**
- Reality bending sounds
- Discordant tones
- Audio pitch shifting

**Implementation Notes:**
- Complex shader effects
- Fractal geometry rendering
- Portal/recursion effects
- Expensive - performance consideration
- Could use existing `shader_fractal_fog` as base

---

### 3. quantum_distortion
**Purpose:** Reality flickering between timelines

**Trigger Conditions:**
- Very rare: 1/100000 per frame
- Time-of-day specific (midnight)

**Duration:** 20-30 seconds

**Visual Description:**
- Scene switching between alternate versions
- Time stutter effect (frame skipping)
- Multiple timelines overlapping
- Same scene with different lighting/weather
- Ghosting/afterimages of alternate realities
- Colors shift between versions

**Audio:**
- Time rewind sounds
- Echoing/repeating audio snippets
- Distorted reverb

**Implementation Notes:**
- Multiple render buffers
- Cross-fade between weather states
- Temporal aliasing effect
- Very high visual impact
- Performance intensive

---

## Implementation Priority

### Phase 1 - Essential Background (Implement First)
1. **neon_grid** - Foundational cyberpunk aesthetic
2. **data_rain** - Iconic visual element
3. **city_fog** - Modified fog system
4. **scan_lines** - Low-cost overlay effect

### Phase 2 - Core Atmosphere
5. **neon_signs** - Key visual identifier
6. **hologram_billboards** - Primary background event
7. **ambient_particles** - Environmental detail

### Phase 3 - Weather Events
8. **cyber_lightning** - Electric storm support
9. **acid_puddles** - Acid rain visual
10. **glitch_wave** - Glitch fog effect
11. **drone_spotlight** - Drone patrol mechanic

### Phase 4 - Polish & Variety
12. **power_surge** - Dramatic moment
13. **data_overflow** - Data storm enhancement
14. **police_chase** - Environmental storytelling
15. **billboard_hack** - Interactive feeling

### Phase 5 - Advanced/Optional
16. **smog_bank** - Smog haze enhancement
17. **window_lights** - Building detail
18. **neon_cascade_failure** - State transition
19. **hologram_malfunction** - Detail variation

### Phase 6 - Special Events (If Time/Resources)
20. **corporate_announcement** - Rare narrative moment
21. **augmented_reality_glitch** - High visual impact
22. **quantum_distortion** - Ultimate rare event

---

## Parameter Usage Reference

### New Cyberpunk Parameters:
- `neon_intensity` (0.0-1.0) - Brightness of neon lighting
- `pollution_level` (0.0-1.0) - Smog/haze density
- `hologram_density` (0.0-1.0) - Number of holograms
- `electric_interference` (0.0-1.0) - Glitch/static intensity
- `data_flow_rate` (0.0-1.0) - Data rain speed
- `light_pollution` (0.0-1.0) - Ambient city glow
- `drone_activity` (0.0-1.0) - Drone presence
- `glitch_probability` (0.0-1.0) - Visual glitch chance
- `scan_line_intensity` (0.0-1.0) - CRT effect strength

### Standard Parameters Used:
- `fog` - Base fog amount
- `fog_color` - Fog tint (repurposed for colored lighting)
- `rain_rate` - Precipitation
- `lightning_probability` - Electric storm strikes
- `wind_speed` - Particle drift
- `celestial_visibility` - Sky visibility
- `starryness` - Star visibility
- `sound` - Audio reactivity

---

## Audio Asset Needs

### Background Ambience:
- City traffic drone
- Distant sirens
- Industrial machinery hum
- Electronic buzz/hum
- Digital data sounds

### Event Sounds:
- Lightning cracks (electric tone)
- Drone motors
- Digital glitches/corruption
- Power surge/transformer explosion
- Siren (doppler effect)
- Sizzling/hissing (acid)
- Alert tones
- Corporate jingles
- Propaganda voice clips

---

## Color Palette Reference

### Neon Colors:
- **Neon Pink:** RGB(255, 0, 110) / #FF006E
- **Neon Cyan:** RGB(0, 245, 255) / #00F5FF
- **Neon Purple:** RGB(191, 0, 255) / #BF00FF
- **Matrix Green:** RGB(0, 255, 65) / #00FF41
- **Electric Blue:** RGB(0, 128, 255) / #0080FF

### Pollution/Industrial:
- **Smog Orange:** RGB(179, 102, 51) / #B36633
- **Toxic Yellow:** RGB(230, 230, 51) / #E6E633
- **Rust Brown:** RGB(140, 80, 40) / #8C5028
- **Acid Green:** RGB(128, 204, 51) / #80CC33

### Atmosphere:
- **Night Blue:** RGB(10, 20, 40) / #0A1428
- **Dark Purple:** RGB(40, 10, 60) / #280A3C
- **Deep Red:** RGB(80, 10, 20) / #500A14

---

## Technical Notes

### Performance Considerations:
- Background events should be optimized for constant rendering
- Particle systems need LOD (Level of Detail) scaling
- Post-processing effects should be toggleable
- Complex shaders (quantum distortion) should be rare
- Consider GPU memory for texture assets

### Integration Points:
- Background events added to `EVENT_MAP` in `projects/fan/event_map.py`
- Scheduled via `_initialize_weather_set_events()`
- Temporary events triggered via the project's `random_events` hook (`projects/fan/random_events.py`)
- Weather state transitions can trigger specific events
- Audio synchronized through existing `soundengine`

### Shader Architecture:
- Most effects implemented as fragment shaders
- Use existing shader framework from `shader_effects/`
- Create new files: `shader_neon_grid.py`, `shader_data_rain.py`, etc.
- Follow existing pattern for parameter passing
- Use `state` and `outstate` dictionaries

---

## Future Expansion Ideas

### Additional Weather States:
- **CYBER_NETWORK_OUTAGE** - Communication blackout
- **CYBER_SOLAR_FLARE** - EM interference
- **CYBER_QUANTUM_FLUX** - Reality instability
- **CYBER_CORPORATE_LOCKDOWN** - Martial law

### Advanced Effects:
- Face recognition scanning overlay
- Neural link connection visuals
- Cryptocurrency price tickers
- Social credit score displays
- Augmented reality UI elements

### Interactive Elements:
- Hackable billboards (user can trigger)
- Controllable drones
- Traffic simulation
- Building light patterns (user designed)

---

*Document Version: 1.0*
*Created: 2025-12-30*
*Last Updated: 2025-12-30*
