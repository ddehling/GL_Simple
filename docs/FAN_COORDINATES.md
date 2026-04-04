# Fan Coordinate System Guide

## Overview

The LED array is mapped to a semicircle with an inner radius of 4 ft and an outer radius of 20.6 ft. Effects render to a 128x300 rectangular framebuffer (FBO), which is then displayed on this fan. The **fan coordinate system** (`renderer/fan_coords.py`) provides tools to convert between three coordinate spaces:

1. **Buffer UV** — normalized `(u, v)` in `[0, 1]`, where `u` = column/127 and `v` = row/299
2. **Physical space** — real-world `(x, y)` in feet on the fan surface
3. **Buffer pixels** — integer `(col, row)` in the 128x300 FBO

## When to Use Fan Coordinates

Use fan coordinates when you want patterns to appear **geometrically correct on the physical fan**. Without compensation, a circle drawn in buffer space appears as a distorted wedge on the fan, and features near the inner edge are compressed while features near the outer edge are stretched.

**Use fan coords for:**
- Geographic maps (BART map, terrain)
- Rectangular grids, text, or UI elements that should look square
- Any effect where uniform physical spacing matters
- Scaling object sizes so they appear consistent across the fan

**Don't use fan coords for:**
- Effects that naturally work in polar space (aurora, radial waves)
- Abstract patterns where distortion is acceptable or desired
- Post-processing effects (fog, bloom) that operate on the full buffer

## Physical Space Layout

The semicircle spans:
- **x**: -20.6 to +20.6 ft (left to right)
- **y**: 0 to 20.6 ft (bottom to top)
- Inner radius: 4.0 ft, outer radius: 20.6 ft
- Angle: 180 degrees (pi radians), left = pi, right = 0

```
                    y = 20.6 ft
                        |
           outer_r _____|_____
                 /      |      \
                /       |       \
               /        |        \
    x=-20.6   |  inner_r|         |  x=+20.6
              |    _____|_____    |
               \  /     |     \  /
                \/______|______\/
                    y = 0 ft
```

## GLSL Usage

### Include in Fragment Shader

```python
from renderer.fan_coords import FAN_COORDS_UNIFORMS, FAN_COORDS_GLSL, FanCoords

_FRAG_SRC = f"""#version 310 es
precision highp float;
in vec2 v_texcoord;
out vec4 fragColor;

{FAN_COORDS_UNIFORMS}
{FAN_COORDS_GLSL}

void main() {{
    vec2 phys = fan_uv_to_physical(v_texcoord);
    // phys.x, phys.y are in feet
    // Draw patterns using physical coordinates here
}}"""
```

### Set Uniforms (once at init)

```python
fan = FanCoords(viewport.width, viewport.height)

# In setup_buffers() or after compile:
glUseProgram(self.shader)
fan.set_uniforms(self.shader)
glUseProgram(0)
```

### Available GLSL Functions

| Function | Signature | Returns |
|----------|-----------|---------|
| `fan_uv_to_physical` | `vec2 fan_uv_to_physical(vec2 uv)` | Physical `(x, y)` in feet from buffer UV |
| `fan_physical_to_uv` | `vec2 fan_physical_to_uv(vec2 xy)` | Buffer UV from physical `(x, y)` feet, clamped to `[0, 1]` |
| `fan_radius_ft` | `float fan_radius_ft(vec2 uv)` | Physical radius in feet at buffer UV |
| `fan_theta` | `float fan_theta(vec2 uv)` | Angle in radians at buffer UV (pi at left, 0 at right) |
| `fan_pixel_scale` | `float fan_pixel_scale(vec2 uv)` | Local scale factor: 1.0 = average density, <1.0 near inner edge, >1.0 near outer edge |
| `fan_arc_width_ft` | `float fan_arc_width_ft(vec2 uv)` | Physical width of one pixel in feet at UV |
| `fan_radial_height_ft` | `float fan_radial_height_ft()` | Physical height of one pixel in feet (constant everywhere) |

### GLSL Uniforms (set by `FanCoords.set_uniforms`)

| Uniform | Value | Description |
|---------|-------|-------------|
| `u_inner_r_ft` | 4.0 | Inner radius in feet |
| `u_outer_r_ft` | 20.6 | Outer radius in feet |
| `u_num_cols` | 128.0 | Buffer width in pixels |
| `u_num_rows` | 300.0 | Buffer height in pixels |

## Python Usage

```python
from renderer.fan_coords import FanCoords

fc = FanCoords(128, 300)

# Convert physical position to buffer UV
u, v = fc.physical_to_uv(5.0, 12.0)   # 5 ft right, 12 ft up

# Convert physical position to pixel coordinates
px, py = fc.physical_to_px(5.0, 12.0)

# Convert buffer UV to physical feet
x, y = fc.uv_to_physical(0.5, 0.5)    # center of buffer

# Get local pixel scale (for size compensation)
scale = fc.pixel_scale_at_uv(u, v)    # 1.0 = average, >1 = outer, <1 = inner

# Physical size of one pixel
arc_w = fc.arc_width_ft(v)            # varies with radius
rad_h = fc.radial_height_ft()         # constant: 0.055 ft

# Vectorized numpy versions
x_arr, y_arr = fc.uv_to_physical_np(u_array, v_array)
u_arr, v_arr = fc.physical_to_uv_np(x_array, y_array)
```

## Common Patterns

### Drawing a Rectangular Grid on the Fan

In the fragment shader, convert each pixel to physical space, then compute grid lines in feet:

```glsl
vec2 phys = fan_uv_to_physical(v_texcoord);
float spacing = 2.0;  // feet
float x_dist = abs(mod(phys.x + spacing * 0.5, spacing) - spacing * 0.5);
float y_dist = abs(mod(phys.y + spacing * 0.5, spacing) - spacing * 0.5);
float grid = max(
    smoothstep(0.25, 0.1, x_dist),
    smoothstep(0.25, 0.1, y_dist)
);
```

### Mapping Geographic Coordinates to the Fan

Map a lat/lon bounding box to a physical region on the fan, then convert to buffer pixels:

```python
fc = FanCoords()

# Define physical region the map covers
PHYS_X_MIN, PHYS_X_MAX = -20.6, 20.6
PHYS_Y_MIN, PHYS_Y_MAX =   0.0, 20.6

def geo_to_fan_px(lat, lon, w, h):
    nx = (lon - LON_MIN) / (LON_MAX - LON_MIN)
    ny = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
    phys_x = PHYS_X_MIN + nx * (PHYS_X_MAX - PHYS_X_MIN)
    phys_y = PHYS_Y_MIN + ny * (PHYS_Y_MAX - PHYS_Y_MIN)
    return fc.physical_to_px(phys_x, phys_y)
```

### Compensating Object Size for Radial Position

Objects (circles, clouds, sprites) should be scaled by the inverse of `pixel_scale` so they appear the same physical size everywhere on the fan:

```python
# At spawn time:
v_frac = spawn_y / viewport_height
scale = fc.pixel_scale(v_frac)
pixel_radius = desired_physical_radius / scale

# Or with precomputed per-object scales:
station_scales = np.array([
    fc.pixel_scale_at_uv(*fc.physical_to_uv(x, y))
    for x, y in station_positions
])
compensated_radii = base_radius / station_scales
```

### Inverse Mapping: Physical to Lat/Lon

For effects that need to know the geographic position of a pixel (e.g., spatial fog):

```glsl
vec2 phys = fan_uv_to_physical(v_texcoord);
// Inverse of geo_to_physical mapping
float nx = (phys.x - PHYS_X_MIN) / (PHYS_X_MAX - PHYS_X_MIN);
float ny = (phys.y - PHYS_Y_MIN) / (PHYS_Y_MAX - PHYS_Y_MIN);
float lon = LON_MIN + nx * (LON_MAX - LON_MIN);
float lat = LAT_MIN + ny * (LAT_MAX - LAT_MIN);
```

## Coordinate Math Reference

### Buffer UV to Physical

```
theta = pi * (1.0 - u)
r     = inner_r_ft + v * (outer_r_ft - inner_r_ft)
x     = r * cos(theta)
y     = r * sin(theta)
```

### Physical to Buffer UV

```
r     = sqrt(x^2 + y^2)
theta = atan2(y, x)
u     = 1.0 - theta / pi
v     = (r - inner_r_ft) / (outer_r_ft - inner_r_ft)
```

### Pixel Scale

A pixel at row fraction `v` covers physical arc width `r * pi / num_cols` horizontally and `(outer_r - inner_r) / num_rows` radially. The scale factor relative to the midpoint radius:

```
mid_r = (inner_r_ft + outer_r_ft) / 2
scale = r / mid_r
```

- `scale < 1.0` near inner edge (pixels cover less area, higher density)
- `scale > 1.0` near outer edge (pixels cover more area, lower density)
- `scale = 1.0` at the midpoint radius (~12.3 ft)

## Files

- `renderer/fan_coords.py` — GLSL string constants + Python `FanCoords` class
- `renderer/fan_geometry.py` — Physical constants (`PHYSICAL_INNER_FT`, `PHYSICAL_OUTER_FT`), display-mode geometry
- `renderer/effects/test_fan_coords.py` — Visual test effect (rectangular grid in physical space)
