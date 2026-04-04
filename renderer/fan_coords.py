"""Fan coordinate mapping utility for physically-correct pattern generation.

Effects render to a 128x300 rectangular FBO that maps onto a semicircular
LED fan (4 ft inner radius, 20.6 ft outer radius).  This module provides
both GLSL functions (as string constants to concatenate into shaders) and
a Python class mirroring the same math for CPU-side use.

Usage is opt-in per-effect.  Existing effects that work in r/theta buffer
space continue unchanged.

GLSL usage::

    from renderer.fan_coords import FAN_COORDS_UNIFORMS, FAN_COORDS_GLSL

    frag_src = f\"\"\"#version 310 es
    precision highp float;
    in vec2 vTexCoord;
    out vec4 fragColor;
    {FAN_COORDS_UNIFORMS}
    {FAN_COORDS_GLSL}
    void main() {{
        vec2 phys = fan_uv_to_physical(vTexCoord);
        // phys.x, phys.y are in feet
    }}\"\"\"

Python usage::

    from renderer.fan_coords import FanCoords
    fc = FanCoords(128, 300)
    x, y = fc.uv_to_physical(0.5, 0.5)   # center of buffer -> feet
    u, v = fc.physical_to_uv(0.0, 12.0)  # 12 ft up -> buffer UV
"""

import math
import numpy as np
from OpenGL.GL import glGetUniformLocation, glUniform1f, GL_FLOAT
from renderer.fan_geometry import FanGeometry


# ---------------------------------------------------------------------------
# Physical constants (feet) — sourced from FanGeometry
# ---------------------------------------------------------------------------
INNER_R_FT = FanGeometry.PHYSICAL_INNER_FT   # 4.0
OUTER_R_FT = FanGeometry.PHYSICAL_OUTER_FT   # 20.6

_PI = math.pi


# ---------------------------------------------------------------------------
# GLSL string constants
# ---------------------------------------------------------------------------

FAN_COORDS_UNIFORMS = """
// Fan coordinate mapping uniforms (set via FanCoords.set_uniforms)
uniform float u_inner_r_ft;   // 4.0  — inner radius in feet
uniform float u_outer_r_ft;   // 20.6 — outer radius in feet
uniform float u_num_cols;     // 128.0
uniform float u_num_rows;     // 300.0
"""

FAN_COORDS_GLSL = """
// --- Fan coordinate mapping functions ---
// Maps between buffer UV space and physical (x, y) feet on the fan.

const float FAN_PI = 3.14159265358979;

// Physical radius in feet at a buffer UV position.
float fan_radius_ft(vec2 uv) {
    return u_inner_r_ft + uv.y * (u_outer_r_ft - u_inner_r_ft);
}

// Angle in radians at a buffer UV position (pi at left, 0 at right).
float fan_theta(vec2 uv) {
    return FAN_PI * (1.0 - uv.x);
}

// Buffer UV -> physical (x, y) in feet.
// The semicircle spans x ~ [-20.6, 20.6], y ~ [0, 20.6].
vec2 fan_uv_to_physical(vec2 uv) {
    float r = fan_radius_ft(uv);
    float theta = fan_theta(uv);
    return vec2(r * cos(theta), r * sin(theta));
}

// Physical (x, y) feet -> buffer UV.  Clamped to [0, 1].
vec2 fan_physical_to_uv(vec2 xy) {
    float r = length(xy);
    float theta = atan(xy.y, xy.x);
    float u = 1.0 - theta / FAN_PI;
    float v = (r - u_inner_r_ft) / (u_outer_r_ft - u_inner_r_ft);
    return clamp(vec2(u, v), 0.0, 1.0);
}

// Local pixel scale factor.  1.0 = average pixel density.
// < 1.0 near inner edge (pixels are physically smaller),
// > 1.0 near outer edge (pixels are physically larger).
float fan_pixel_scale(vec2 uv) {
    float r = fan_radius_ft(uv);
    float mid_r = (u_inner_r_ft + u_outer_r_ft) * 0.5;
    return r / mid_r;
}

// Physical arc width of one pixel in feet at the given UV.
float fan_arc_width_ft(vec2 uv) {
    return fan_radius_ft(uv) * FAN_PI / u_num_cols;
}

// Physical radial height of one pixel in feet (constant everywhere).
float fan_radial_height_ft() {
    return (u_outer_r_ft - u_inner_r_ft) / u_num_rows;
}
"""


# ---------------------------------------------------------------------------
# Python class — mirrors GLSL math + uniform upload
# ---------------------------------------------------------------------------

class FanCoords:
    """CPU-side fan coordinate mapping and uniform helper.

    Parameters
    ----------
    num_cols : int
        Number of LED strips / buffer columns (default 128).
    num_rows : int
        Number of LEDs per strip / buffer rows (default 300).
    """

    def __init__(self, num_cols: int = 128, num_rows: int = 300):
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.inner_r_ft = INNER_R_FT
        self.outer_r_ft = OUTER_R_FT
        self._mid_r = (self.inner_r_ft + self.outer_r_ft) * 0.5

    # -- coordinate conversions --

    def uv_to_physical(self, u: float, v: float):
        """Buffer UV -> physical (x, y) in feet."""
        theta = _PI * (1.0 - u)
        r = self.inner_r_ft + v * (self.outer_r_ft - self.inner_r_ft)
        return r * math.cos(theta), r * math.sin(theta)

    def physical_to_uv(self, x: float, y: float):
        """Physical (x, y) feet -> buffer (u, v), clamped to [0, 1]."""
        r = math.sqrt(x * x + y * y)
        theta = math.atan2(y, x)
        u = max(0.0, min(1.0, 1.0 - theta / _PI))
        v = max(0.0, min(1.0, (r - self.inner_r_ft) / (self.outer_r_ft - self.inner_r_ft)))
        return u, v

    def physical_to_px(self, x: float, y: float):
        """Physical (x, y) feet -> buffer pixel coordinates."""
        u, v = self.physical_to_uv(x, y)
        return u * self.num_cols, v * self.num_rows

    def uv_to_physical_np(self, u: np.ndarray, v: np.ndarray):
        """Vectorized: buffer UV arrays -> physical (x, y) arrays in feet."""
        theta = _PI * (1.0 - u)
        r = self.inner_r_ft + v * (self.outer_r_ft - self.inner_r_ft)
        return r * np.cos(theta), r * np.sin(theta)

    def physical_to_uv_np(self, x: np.ndarray, y: np.ndarray):
        """Vectorized: physical (x, y) feet arrays -> buffer (u, v) arrays."""
        r = np.sqrt(x * x + y * y)
        theta = np.arctan2(y, x)
        u = np.clip(1.0 - theta / _PI, 0.0, 1.0)
        v = np.clip((r - self.inner_r_ft) / (self.outer_r_ft - self.inner_r_ft), 0.0, 1.0)
        return u, v

    # -- scale helpers --

    def radius_ft(self, v: float) -> float:
        """Physical radius in feet at buffer row fraction v."""
        return self.inner_r_ft + v * (self.outer_r_ft - self.inner_r_ft)

    def pixel_scale(self, v: float) -> float:
        """Local scale factor at row fraction v (1.0 = average density)."""
        return self.radius_ft(v) / self._mid_r

    def pixel_scale_at_uv(self, u: float, v: float) -> float:
        """Local scale factor at buffer UV (1.0 = average density)."""
        return self.radius_ft(v) / self._mid_r

    def arc_width_ft(self, v: float) -> float:
        """Physical arc width of one pixel in feet at row fraction v."""
        return self.radius_ft(v) * _PI / self.num_cols

    def radial_height_ft(self) -> float:
        """Physical radial height of one pixel in feet (constant)."""
        return (self.outer_r_ft - self.inner_r_ft) / self.num_rows

    # -- OpenGL uniform upload --

    def set_uniforms(self, shader_program):
        """Upload fan coordinate uniforms to an active shader program.

        Call this after ``glUseProgram(shader_program)``.
        Silently skips uniforms the shader doesn't declare.
        """
        loc = glGetUniformLocation(shader_program, "u_inner_r_ft")
        if loc != -1:
            glUniform1f(loc, self.inner_r_ft)
        loc = glGetUniformLocation(shader_program, "u_outer_r_ft")
        if loc != -1:
            glUniform1f(loc, self.outer_r_ft)
        loc = glGetUniformLocation(shader_program, "u_num_cols")
        if loc != -1:
            glUniform1f(loc, float(self.num_cols))
        loc = glGetUniformLocation(shader_program, "u_num_rows")
        if loc != -1:
            glUniform1f(loc, float(self.num_rows))
