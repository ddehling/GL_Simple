# Shader Contrast Playbook

> Read [docs/shader_info.txt](shader_info.txt) for alpha/depth correctness rules
> first. This document covers a separate concern: **how to make shaders
> read clearly under the per-receiver brightness limit.**

## The constraint

`brightness_limit` in `project.yaml` divides every pixel's output by the
average energy of the pixels driving each physical receiver — a power-cap
on the LED installation that can't be raised. For Fan that limit is
**0.1**, meaning the limiter compresses output toward 1/10th-scale
whenever a shader paints a lot of pixels brightly.

You will not get more headroom by asking. The job is to **spend the
energy budget where the eye can perceive contrast.**

## The key insight

The limiter is **per-receiver-energy**, not per-pixel-cap. Two scenes
with identical total energy land at the same effective brightness, but
look very different:

| Scene A | Scene B |
|---|---|
| 100% of pixels at 0.1 brightness | 10% of pixels at 1.0 brightness |
| Reads as: dim uniform color field | Reads as: sparse bright points against darkness |
| Same energy budget | Same energy budget |
| Low perceptual contrast | High perceptual contrast |

Stage lighting has known this forever: a single bright instrument against
darkness reads stronger than a wash at the same total output. The same
physics applies to LED pixels under an energy cap.

## Three levers

These compound — apply more than one whenever possible.

### 1. Hue separation between paired layers

Even at low brightness, opponent hues (warm/cool, red/cyan,
yellow/violet) read as visually distinct. Same-family palettes (e.g. all
warm tones) collapse into mud once the limiter compresses them.

If two layers will be visible simultaneously, give them *complementary*
chroma. Examples:

- **Sky horizon + dune silhouette** → cool sky band immediately above
  the ridge so warm dunes pop, even if the zenith is something else.
- **Forest canopy + falling leaves** → cool deep-green canopy, leaves
  authored in warm autumn tones — not green-on-green.
- **Lightning + cloud bank** → silver-blue bolt against warm-cream
  cloud rather than white-on-white.

**Common failure**: copying `fog_color` or `u_tint` into multiple layers.
The atmospheric tint then dominates every layer's hue, dragging them all
into the same chromatic neighborhood. Use `u_tint` sparingly — let each
layer carry its own characteristic hue.

### 2. Negative space — concentrate, don't wash

Audit a shader's pixel coverage. If most of the canvas is being painted
to a moderate brightness, the limiter is spreading your budget thin.
Refactor toward sparse bright features against a dark or zero-output
background.

**Wash-style (avoid by default):**
- Fullscreen-quad gradients at high alpha that hit every pixel
- Atmospheric overlays that tint the whole canvas at moderate intensity
- "Filled-in" shapes (a dune body that's uniformly mid-bright)

**Sparse-bright (prefer):**
- Stars: single bright points against black
- Crest rim-light: dark dune body, only the leeward edge / sun-facing
  crest is bright
- Light shafts: dark canopy with thin bright wedges of god-rays
- Silhouettes: 0-output figure against a bright partial-coverage sky band

If a wash *is* the design intent (e.g. a peaceful fog scene), accept the
brightness compression and rely on lever 1 (hue separation) instead.

### 3. Saturation over brightness

At the same luminance, a saturated chroma value reads more distinctly
than a desaturated neutral. The limiter compresses luminance more than
it compresses chroma in shader code — most shaders write `vec3 col`
direct to the framebuffer with no tone-mapping, so high-saturation
colors survive the limiter better than high-luminance ones.

**Prefer**: `vec3(0.8, 0.1, 0.1)` over `vec3(0.4, 0.4, 0.4)` if you
want "red, visible."

**Don't desaturate toward white** to "add brightness" — you're spending
energy on all three channels for less perceptual gain.

## Patterns

### Rim light (best per-energy ratio)

A dark body with a thin bright edge. Lambert slope shading with a wide
output range — e.g. `0.05 .. 1.0` rather than `0.35 .. 1.30` — leaves
most of the body near-zero output and concentrates the energy on the
sun-facing crest. The eye reads strong form from the bright edge alone.

### Silhouette against a band

A figure (dunes, canopy, mountains) at zero or near-zero output, sitting
in front of a sky region whose brightest band is at the figure's
horizon. The sky pays the energy cost; the figure is free because it's
dark. The silhouette shape reads clearly because of the bright backdrop.

### Sparse feature emission

Stars, fireflies, lightning seeds, sparks. Output is zero almost
everywhere and high in isolated points. Naturally budget-friendly. The
risk: too sparse and there's nothing to see — calibrate density to the
pixel count of the target group canvas.

## Anti-patterns

| Pattern | Why it fails |
|---|---|
| Fullscreen gradient at high alpha | Spends ~all the budget on broad coverage; limiter then crushes it |
| Two layers with same hue family | Limiter compresses them into indistinguishable mud |
| Atmospheric perspective via brightness only | Loses depth cues when the limiter compresses; use hue/saturation shifts instead |
| `mix(layer_color, u_tint, 0.4)` on every layer | Pulls all layers into atmospheric mud — fog_color contaminates everything |
| Wide slope-shading range like `0.5 .. 1.0` | Body still mostly bright; rim-light effect is lost |

## Where the limiter bites unexpectedly

The brightness limiter operates per-receiver, after composing all the
event shaders for that group. So:

- A single "background" shader that paints the whole canvas at moderate
  alpha is the worst offender — even with no other effects active, it
  still triggers compression.
- Adding a sparse-bright feature shader (stars, sparks) on top of a
  full-canvas wash gets compressed *with* the wash — the sparse feature
  doesn't reclaim its budget independently.
- Removing or dimming a wash *increases* the headroom available to
  every other shader on that group. Sometimes the best shader edit is
  to lower a backdrop's alpha rather than touch the foreground.

## Auditing an existing shader

Quick read:

1. What fraction of pixels does this shader paint above ~0.2 alpha?
   (Eyeball from the GLSL: are most pixels written, or only the masked
   feature region?)
2. What's its primary hue? Where else in the active weather set does
   that same hue appear? (Open the other shaders' final-color math.)
3. Could the body be darkened and the feature edge brightened without
   changing the overall energy?
4. Could the hue shift toward a complement of whatever it'll be paired
   with?

Worked examples lower in this doc as we go (TBD: desert_sky / dunes
audit, forest audit).
