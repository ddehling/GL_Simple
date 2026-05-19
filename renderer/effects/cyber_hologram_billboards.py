"""
Cyber hologram billboards — translucent floating advertisements with
internal scrolling content and occasional glitch corruption. Each
billboard is a rectangular region with chromatic-aberration edges and
internal scan-line texture. Reads `outstate['cyber_hologram_density']`
(0..1) for count, and `outstate['story_defiance']` for glitch rate
(defiance = billboards corrupting / showing counter-content).
"""
import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


def shader_cyber_hologram_billboards(state, outstate, density=0.5):
    frame_id = state.get('frame_id', 0)
    renderer = outstate.get('shader_renderer')
    if renderer is None:
        return
    viewport = renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(CyberHologramBillboardsEffect,
                                          density=density)
            state['effect'] = effect
        except Exception as e:
            print(f"[cyber_hologram_billboards] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is None:
        return

    eff.density = float(outstate.get('hologram_density', density))
    # Defiance drives glitch corruption rate
    eff.glitch_level = float(outstate.get('story_defiance', 0.0))

    if state['count'] == -1:
        if 'effect' in state:
            viewport.effects.remove(state['effect'])
            state['effect'].cleanup()


VERTEX = """#version 310 es
precision highp float;
layout(location = 0) in vec2 position;
out vec2 v_uv;
void main() {
    v_uv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

FRAGMENT = """#version 310 es
precision highp float;
in vec2 v_uv;
uniform float u_time;
uniform float u_density;
uniform float u_glitch;
out vec4 fragColor;

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

void main() {
    vec2 uv = v_uv;
    vec3 col = vec3(0.0);
    float a = 0.0;

    // Up to 6 billboard slots arranged in the upper 2/3 of the screen.
    // Each slot RELOCATES periodically (per-slot lifetime 20..36s) so
    // billboards don't cluster permanently in one spot.
    const int NUM = 6;
    for (int i = 0; i < NUM; i++) {
        float fi = float(i);
        // Per-slot lifetime — staggered so all 6 don't move simultaneously
        float lifetime = 20.0 + hash(vec2(fi, 0.7)) * 16.0;     // 20..36 sec
        float bucket = floor(u_time / lifetime);
        // Per-slot in-cycle progress (0..1)
        float life_t = mod(u_time, lifetime) / lifetime;

        // On/off gate uses a per-bucket seed so it can vary across cycles
        float on_seed = hash(vec2(fi * 11.0, bucket * 1.3));
        if (on_seed > u_density) continue;

        // Position changes EACH BUCKET — billboard relocates between cycles.
        // Spread across full width and upper 2/3 to avoid clustering.
        float cx = 0.08 + hash(vec2(fi * 7.0,  bucket * 3.1)) * 0.84;
        float cy = 0.08 + hash(vec2(fi * 11.0, bucket * 5.7)) * 0.58;
        // Size also re-rolls per bucket so the same slot doesn't look
        // identical every time it relocates.
        float w = 0.06 + hash(vec2(fi * 13.0, bucket * 7.1)) * 0.06;
        float h = 0.09 + hash(vec2(fi * 17.0, bucket * 9.3)) * 0.07;

        // Fade in / fade out at the start / end of each life cycle so the
        // billboard appears at a new location instead of teleporting.
        float life_fade = smoothstep(0.00, 0.04, life_t)
                        * smoothstep(1.00, 0.94, life_t);

        // Local coords inside the billboard, -1..+1
        vec2 d = (uv - vec2(cx, cy)) / vec2(w, h);

        // Outside billboard?
        if (abs(d.x) > 1.0 || abs(d.y) > 1.0) continue;

        // Edge mask — bright thin border, body slightly translucent
        float ex = abs(d.x);
        float ey = abs(d.y);
        float edge = max(smoothstep(0.85, 0.98, ex), smoothstep(0.85, 0.98, ey));

        // Body content — animated scroll
        float scroll = u_time * (0.3 + hash(vec2(fi, 5.9)) * 0.7);
        float content_row = floor((d.y * 0.5 + 0.5 + scroll) * 12.0);
        float content_col = floor((d.x * 0.5 + 0.5) * 8.0);
        float content_seed = hash(vec2(content_col + fi * 19.0, content_row));
        float content = step(0.35, content_seed) * content_seed;

        // Glitch: when story_defiance high, rows of the billboard get
        // displaced / palette-flipped. Per-row chance scales with u_glitch.
        float row_seed = hash(vec2(fi * 7.0, floor(d.y * 8.0 + u_time * 4.0)));
        float glitched = step(1.0 - u_glitch * 0.6, row_seed);

        // Color: each billboard has a base hue (cyan/pink/lime)
        vec3 base_col;
        float hue = hash(vec2(fi, 6.1));
        if (hue < 0.40) base_col = vec3(0.0, 0.95, 1.0);
        else if (hue < 0.70) base_col = vec3(1.0, 0.2, 0.65);
        else base_col = vec3(0.4, 1.0, 0.2);

        // When glitched, flip palette toward complementary + add red corruption
        vec3 glitch_col = vec3(1.0, 0.10, 0.10);
        base_col = mix(base_col, glitch_col, glitched);

        float body_bright = content * 0.55 + edge * 1.0;
        body_bright = clamp(body_bright, 0.0, 1.0);

        // Per-billboard alpha — translucent but not weak. Multiplied by
        // life_fade so the billboard smoothly fades in at start of cycle
        // and fades out at end of cycle (just before it relocates).
        float bb_alpha = body_bright * 0.85 * life_fade;

        // Accumulate (saturate)
        col = max(col, base_col * body_bright * life_fade);
        a = max(a, bb_alpha);
    }

    if (a < 0.04) discard;
    fragColor = vec4(col, a);
}
"""


class CyberHologramBillboardsEffect(ShaderEffect):
    def __init__(self, viewport, density: float = 0.5):
        super().__init__(viewport)
        self.render_priority = 7.5    # In front of signs (7.0), behind data_rain (8.0)
        self.density = density
        self.glitch_level = 0.0
        self._time = 0.0

    def compile_shader(self):
        try:
            vert = shaders.compileShader(VERTEX, GL_VERTEX_SHADER)
            frag = shaders.compileShader(FRAGMENT, GL_FRAGMENT_SHADER)
            return shaders.compileProgram(vert, frag)
        except Exception as e:
            print(f"CyberHologramBillboards compile error: {e}")
            raise

    def setup_buffers(self):
        verts = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        vbo = glGenBuffers(1)
        self.VBOs = [vbo]
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glBindVertexArray(0)

    def update(self, dt: float, state: Dict):
        if not self.enabled:
            return
        self._time += dt

    def render(self, state: Dict):
        if not self.enabled or self.density < 0.01:
            return
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glUniform1f(glGetUniformLocation(self.shader, "u_density"), self.density)
        glUniform1f(glGetUniformLocation(self.shader, "u_glitch"), self.glitch_level)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
