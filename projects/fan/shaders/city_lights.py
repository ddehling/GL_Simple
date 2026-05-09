"""City lights and bridge lights effect for the bartiki weather set.

At night, urban areas twinkle with warm single-pixel lights and the
Golden Gate and Bay Bridge light up with strings of white dots.
Controlled by time-of-day (season): invisible during day, fade in
at dusk, full brightness at night.
"""

import numpy as np
import ctypes
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict, List, Tuple
from renderer.effects.base import ShaderEffect
from .bart_map import (
    _geo_to_fan_px, _URBAN_POLYGONS, _point_in_polygon,
    _LAT_MIN, _LAT_MAX, _LON_MIN, _LON_MAX,
)

# ---------------------------------------------------------------------------
# Bridge coordinates (lat, lon polylines)
# ---------------------------------------------------------------------------
_GOLDEN_GATE = [
    (37.8085, -122.4776), (37.8115, -122.4770), (37.8145, -122.4764),
    (37.8175, -122.4758), (37.8205, -122.4752), (37.8235, -122.4746),
    (37.8265, -122.4740), (37.8295, -122.4735), (37.8324, -122.4730),
    (37.8345, -122.4725), (37.8365, -122.4718), (37.8385, -122.4710),
    (37.8400, -122.4700), (37.8415, -122.4688), (37.8430, -122.4675),
]

_BAY_BRIDGE = [
    (37.7883, -122.3935), (37.7903, -122.3885), (37.7923, -122.3835),
    (37.7943, -122.3785), (37.7963, -122.3735), (37.7983, -122.3685),
    (37.8003, -122.3635), (37.8023, -122.3585), (37.8043, -122.3535),
    (37.8063, -122.3485), (37.8083, -122.3435), (37.8103, -122.3385),
    (37.8123, -122.3335), (37.8143, -122.3285), (37.8163, -122.3235),
    (37.8183, -122.3185), (37.8200, -122.3135), (37.8215, -122.3085),
    (37.8225, -122.3035), (37.8230, -122.2985),
]


# ---------------------------------------------------------------------------
# Event wrapper
# ---------------------------------------------------------------------------

def shader_city_lights(state, outstate):
    """City lights + bridge lights background effect."""
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(CityLightsEffect)
            state['effect'] = effect
            print(f"  [OK] Initialized city_lights for frame {frame_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to initialize city_lights: {e}")
            import traceback
            traceback.print_exc()
            return

    if 'effect' in state:
        state['effect'].time_of_day = outstate.get('season', 0.5)

    if state['count'] == -1:
        if 'effect' in state:
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()


# ---------------------------------------------------------------------------
# Shaders
# ---------------------------------------------------------------------------

_VERT = """
#version 310 es
precision highp float;

layout(location = 0) in vec2 aQuad;
layout(location = 1) in vec2 iPos;
layout(location = 2) in vec3 iColor;
layout(location = 3) in float iPhase;

out vec3 vColor;
out float vPhase;
out vec2 vQuad;

uniform vec2 uResolution;

void main() {
    // Single-pixel points: quad size = 1 pixel
    vec2 world = iPos + aQuad * 0.5;
    vec2 clip = (world / uResolution) * 2.0 - 1.0;
    float depth = 97.0 / 100.0;
    gl_Position = vec4(clip, depth, 1.0);
    vColor = iColor;
    vPhase = iPhase;
    vQuad = aQuad;
}
"""

_FRAG = """
#version 310 es
precision highp float;

in vec3 vColor;
in float vPhase;
in vec2 vQuad;
out vec4 FragColor;

uniform float uTime;
uniform float uNightFactor;

void main() {
    float dist = length(vQuad);
    if (dist > 1.0) discard;

    // Slow twinkle
    float twinkle = 0.7 + 0.3 * sin(uTime * 1.5 + vPhase);

    float alpha = uNightFactor * twinkle;
    if (alpha < 0.01) discard;

    FragColor = vec4(vColor * twinkle, alpha);
}
"""


# ---------------------------------------------------------------------------
# Effect class
# ---------------------------------------------------------------------------

class CityLightsEffect(ShaderEffect):
    """Instanced single-pixel lights for urban areas and bridges."""

    def __init__(self, viewport):
        super().__init__(viewport)
        self.render_priority = 5.3
        self.time_of_day = 0.5
        self._time = 0.0

        w, h = float(viewport.width), float(viewport.height)

        # Generate city light positions within urban polygons
        city_positions = []
        city_colors = []
        city_phases = []

        np.random.seed(42)  # deterministic placement
        for poly in _URBAN_POLYGONS:
            # Bounding box of polygon
            lats = [p[0] for p in poly]
            lons = [p[1] for p in poly]
            lat_min, lat_max = min(lats), max(lats)
            lon_min, lon_max = min(lons), max(lons)

            # Scatter points inside polygon
            n_attempts = 80
            for _ in range(n_attempts):
                lat = np.random.uniform(lat_min, lat_max)
                lon = np.random.uniform(lon_min, lon_max)
                if _point_in_polygon(lon, lat, poly):
                    px, py = _geo_to_fan_px(lat, lon, w, h)
                    city_positions.append([px, py])
                    # Warm yellow-orange colors with variation
                    r = np.random.uniform(0.9, 1.0)
                    g = np.random.uniform(0.6, 0.85)
                    b = np.random.uniform(0.1, 0.3)
                    city_colors.append([r, g, b])
                    city_phases.append(np.random.uniform(0, 2 * np.pi))

        # Generate bridge light positions
        bridge_positions = []
        bridge_colors = []
        bridge_phases = []

        for lat, lon in _GOLDEN_GATE:
            px, py = _geo_to_fan_px(lat, lon, w, h)
            bridge_positions.append([px, py])
            bridge_colors.append([1.0, 0.85, 0.6])  # warm white
            bridge_phases.append(np.random.uniform(0, 2 * np.pi))

        for lat, lon in _BAY_BRIDGE:
            px, py = _geo_to_fan_px(lat, lon, w, h)
            bridge_positions.append([px, py])
            bridge_colors.append([0.9, 0.95, 1.0])  # cool white
            bridge_phases.append(np.random.uniform(0, 2 * np.pi))

        np.random.seed(None)  # restore random seed

        # Combine all lights
        all_pos = city_positions + bridge_positions
        all_col = city_colors + bridge_colors
        all_phase = city_phases + bridge_phases

        self._n_lights = len(all_pos)
        if self._n_lights == 0:
            return

        self._positions = np.array(all_pos, dtype=np.float32)
        self._colors = np.array(all_col, dtype=np.float32)
        self._phases = np.array(all_phase, dtype=np.float32)

    def compile_shader(self):
        return shaders.compileProgram(
            shaders.compileShader(_VERT, GL_VERTEX_SHADER),
            shaders.compileShader(_FRAG, GL_FRAGMENT_SHADER),
        )

    def setup_buffers(self):
        if self._n_lights == 0:
            return

        # Unit quad
        quad = np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1],
                        dtype=np.float32)

        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        # Quad VBO (location 0)
        quad_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        self.VBOs.append(quad_vbo)

        # Instance data: pos(2) + color(3) + phase(1) = 6 floats per instance
        instance_data = np.column_stack([
            self._positions,
            self._colors,
            self._phases,
        ]).astype(np.float32)

        inst_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, inst_vbo)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_STATIC_DRAW)
        self.VBOs.append(inst_vbo)

        stride = 6 * 4
        # iPos (location 1)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glVertexAttribDivisor(1, 1)
        # iColor (location 2)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glVertexAttribDivisor(2, 1)
        # iPhase (location 3)
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))
        glVertexAttribDivisor(3, 1)

        glBindVertexArray(0)

    def update(self, dt: float, state: Dict):
        self._time += dt

    def render(self, state: Dict):
        super().render(state)
        if not self.shader or self._n_lights == 0:
            return

        # Night factor: 0 during day, 1 at night
        day_night = -np.cos(self.time_of_day * 2 * np.pi)
        night_factor = max(-day_night, 0.0)  # 0 at noon, 1 at midnight
        # Start fading in at dusk (season ~0.75), full by night (~0.9)
        night_factor = min(night_factor * 1.5, 1.0)

        if night_factor < 0.01:
            return  # invisible during day

        glUseProgram(self.shader)
        glUniform2f(glGetUniformLocation(self.shader, "uResolution"),
                    float(self.viewport.width), float(self.viewport.height))
        glUniform1f(glGetUniformLocation(self.shader, "uTime"), self._time)
        glUniform1f(glGetUniformLocation(self.shader, "uNightFactor"), night_factor)

        glBindVertexArray(self.VAO)
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, self._n_lights)
        glBindVertexArray(0)
        glUseProgram(0)
