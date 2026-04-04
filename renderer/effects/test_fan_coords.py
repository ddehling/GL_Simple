"""Test effect for fan coordinate mapping verification.

Draws concentric circles at 2-foot intervals and radial lines at fixed
angles, all in physical (x, y) feet space using the fan_coords utility.

On the fan display, circles should appear as evenly-spaced arcs and
radial lines should be straight.  On the flat display, the pre-distortion
should be visible (circles wider at top/outer rows).

Event wrapper::

    from renderer.effects.test_fan_coords import shader_test_fan_coords
    scheduler.add_event(TimedEvent("test_fan_coords", ...,
                                   action=shader_test_fan_coords))
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect
from renderer.fan_coords import FAN_COORDS_UNIFORMS, FAN_COORDS_GLSL, FanCoords


# ---------------------------------------------------------------------------
# Event wrapper (scheduler-compatible)
# ---------------------------------------------------------------------------

def shader_test_fan_coords(state, outstate):
    """Test fan coordinate mapping — compatible with EventScheduler."""
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        if state['count'] == 0:
            print(f"  [WARN] test_fan_coords: shader_renderer not in outstate")
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        if state['count'] == 0:
            print(f"  [WARN] test_fan_coords: no viewport for frame {frame_id}")
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(TestFanCoordsEffect)
            state['effect'] = effect
            print(f"  [OK] Initialized test_fan_coords for frame {frame_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to initialize test_fan_coords: {e}")
            import traceback
            traceback.print_exc()
            return

    if state['count'] == -1:
        if 'effect' in state:
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()


# ---------------------------------------------------------------------------
# Fullscreen quad vertex shader (same pattern as shader_fog, etc.)
# ---------------------------------------------------------------------------

_VERT_SRC = """#version 310 es
precision highp float;
in vec2 position;
out vec2 v_texcoord;
void main() {
    v_texcoord = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

# ---------------------------------------------------------------------------
# Fragment shader — draws a rectangular grid in physical feet space
# On the fan display this should appear as a proper Cartesian grid with
# straight horizontal/vertical lines and uniform spacing.
# ---------------------------------------------------------------------------

_FRAG_SRC = f"""#version 310 es
precision highp float;

in vec2 v_texcoord;
out vec4 fragColor;

{FAN_COORDS_UNIFORMS}
{FAN_COORDS_GLSL}

uniform float u_time;

void main() {{
    // Convert buffer UV to physical (x, y) in feet
    vec2 phys = fan_uv_to_physical(v_texcoord);

    // --- Rectangular grid in physical space ---
    float spacing = 2.0;       // grid spacing in feet
    float line_width = 0.25;   // line width in feet

    // Distance to nearest vertical line (constant x)
    float x_dist = abs(mod(phys.x + spacing * 0.5, spacing) - spacing * 0.5);
    float v_line = smoothstep(line_width, line_width * 0.4, x_dist);

    // Distance to nearest horizontal line (constant y)
    float y_dist = abs(mod(phys.y + spacing * 0.5, spacing) - spacing * 0.5);
    float h_line = smoothstep(line_width, line_width * 0.4, y_dist);

    // Major lines every 10 ft
    float major_spacing = 10.0;
    float mx_dist = abs(mod(phys.x + major_spacing * 0.5, major_spacing) - major_spacing * 0.5);
    float my_dist = abs(mod(phys.y + major_spacing * 0.5, major_spacing) - major_spacing * 0.5);
    float major = max(
        smoothstep(line_width * 1.5, line_width * 0.3, mx_dist),
        smoothstep(line_width * 1.5, line_width * 0.3, my_dist)
    );

    float grid = max(v_line, h_line);

    // Minor lines dim cyan, major lines bright cyan
    vec3 color = mix(
        vec3(0.0, 0.4, 0.5),
        vec3(0.0, 0.9, 1.0),
        major
    );

    // Subtle animation to confirm effect is live
    float pulse = 0.85 + 0.15 * sin(u_time * 1.5);
    color *= pulse;

    fragColor = vec4(color * grid, grid * 0.9);
}}
"""


# ---------------------------------------------------------------------------
# ShaderEffect implementation
# ---------------------------------------------------------------------------

class TestFanCoordsEffect(ShaderEffect):
    """Visual test: grid of circles and radial lines in physical space."""

    def __init__(self, viewport):
        super().__init__(viewport)
        self.render_priority = 5
        self._fan = FanCoords(viewport.width, viewport.height)
        self._time = 0.0

    def compile_shader(self):
        return shaders.compileProgram(
            shaders.compileShader(_VERT_SRC, GL_VERTEX_SHADER),
            shaders.compileShader(_FRAG_SRC, GL_FRAGMENT_SHADER),
        )

    def setup_buffers(self):
        quad = np.array([
            -1, -1,  1, -1,  1, 1,
            -1, -1,  1,  1, -1, 1,
        ], dtype=np.float32)

        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        self.VBOs.append(vbo)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)

        # Set static fan coord uniforms once
        glUseProgram(self.shader)
        self._fan.set_uniforms(self.shader)
        glUseProgram(0)

    def update(self, dt: float, state: Dict):
        self._time += dt

    def render(self, state: Dict):
        super().render(state)
        if not self.shader:
            return

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glUseProgram(self.shader)
        loc = glGetUniformLocation(self.shader, "u_time")
        if loc != -1:
            glUniform1f(loc, self._time)

        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
