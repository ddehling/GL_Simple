"""
Cyber drone spotlight — sweeping volumetric beam cone that moves across
the scene. Beam motion is parametric (period-driven), beam color is
red/amber, with a bright "head" where the beam hits the ground. Dust
particles visible in the beam (vertical bright streaks within the cone).
Reads `outstate['cyber_drone_intensity']` (0..1).
"""
import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect


def shader_cyber_drone_spotlight(state, outstate, intensity=0.6):
    frame_id = state.get('frame_id', 0)
    renderer = outstate.get('shader_renderer')
    if renderer is None:
        return
    viewport = renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        try:
            effect = viewport.add_effect(CyberDroneSpotlightEffect,
                                          intensity=intensity)
            state['effect'] = effect
        except Exception as e:
            print(f"[cyber_drone_spotlight] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is None:
        return

    eff.intensity = float(outstate.get('drone_activity', intensity))

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
uniform float u_sweep_phase;     // CPU-integrated sweep
uniform float u_time;
uniform float u_intensity;
out vec4 fragColor;

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
float hash1(float x) { return fract(sin(x * 12.9898) * 43758.5453); }

void main() {
    vec2 uv = v_uv;

    // DUTY CYCLE — at intensity 1.0, the drone is on duty 100% of the time.
    // At intensity 0.3, it's on duty only 30% of each 18s patrol cycle.
    // This is what makes "drone occasionally appears" instead of "always on."
    const float PATROL_PERIOD = 18.0;
    float cycle_t = mod(u_time, PATROL_PERIOD) / PATROL_PERIOD;
    if (cycle_t > u_intensity) discard;     // off-duty pixel — no draw

    float duty_fade = smoothstep(0.0, 0.05, cycle_t)
                    * smoothstep(u_intensity, u_intensity - 0.05, cycle_t);

    // Searcher drone position: hovers at top of screen, sweeps left-right
    // with a pause-and-search rhythm (sin of sin gives uneven motion)
    float sweep_t = u_sweep_phase * 0.5;
    float drone_x = 0.5 + sin(sweep_t) * 0.4 + sin(sweep_t * 0.27) * 0.08;
    float drone_y = 0.05 + sin(sweep_t * 0.3) * 0.03;

    // Beam target on the ground
    float target_x = drone_x + sin(sweep_t * 1.3 + 0.5) * 0.15;
    float target_y = 0.95;

    vec2 d_to_drone = uv - vec2(drone_x, drone_y);
    vec2 beam_dir = normalize(vec2(target_x, target_y) - vec2(drone_x, drone_y));
    float along = dot(d_to_drone, beam_dir);
    float perp  = length(d_to_drone - beam_dir * along);

    vec3 beam_color = vec3(1.0, 0.35, 0.10);

    // --- Beam cone (only renders for pixels below the searcher) ---
    float beam = 0.0;
    if (along > 0.0) {
        float beam_width = 0.02 + along * 0.10;
        beam = smoothstep(beam_width, beam_width * 0.3, perp);
        // Dust in the beam — vertical bright striations
        float dust_seed = hash(vec2(floor(uv.x * 80.0), floor(u_time * 4.0)));
        float dust = step(0.85, dust_seed) * 0.4;
        beam += beam * dust;
    }

    // Ground splash where beam hits — bright disc at target
    float splash_dist = length(uv - vec2(target_x, target_y));
    float splash = smoothstep(0.08, 0.0, splash_dist) * 0.8;
    beam = max(beam, splash);

    // --- Target drones: 3 quadcopters wandering in air space, invisible
    //     until the searcher's cone passes over them, then briefly lit.
    float drone_layer = 0.0;
    for (int i = 0; i < 3; i++) {
        float fi = float(i);
        // Wandering air-space position (LFOs, period ~80-150s per axis).
        // Kept in upper / middle bands (the air, not the ground).
        float tx = 0.12 + 0.76 * (sin(u_time * (0.043 + fi * 0.013) + fi * 1.81) * 0.5 + 0.5);
        float ty = 0.20 + 0.42 * (sin(u_time * (0.061 + fi * 0.019) + fi * 2.41) * 0.5 + 0.5);

        // Is THIS drone's center inside the searcher's cone right now?
        // (Compute once per drone — independent of the pixel's position.)
        vec2  t_off    = vec2(tx, ty) - vec2(drone_x, drone_y);
        float t_along  = dot(t_off, beam_dir);
        float t_in_cone = 0.0;
        if (t_along > 0.0) {
            float t_perp = length(t_off - beam_dir * t_along);
            float t_cone_width = 0.02 + t_along * 0.10;
            // Slightly wider tolerance than the visible beam so drones at
            // the cone edge still light up.
            t_in_cone = smoothstep(t_cone_width * 1.4, 0.0, t_perp);
        }
        if (t_in_cone < 0.02) continue;     // not lit — skip drawing

        // Quadcopter silhouette: small bright body + four short rotor arms.
        vec2  dp     = uv - vec2(tx, ty);
        float dd     = length(dp);
        float body   = smoothstep(0.018, 0.0, dd);
        float arm_h  = smoothstep(0.0035, 0.0, abs(dp.y)) * step(abs(dp.x), 0.036);
        float arm_v  = smoothstep(0.0035, 0.0, abs(dp.x)) * step(abs(dp.y), 0.036);
        float halo   = smoothstep(0.055, 0.0, dd) * 0.35;
        float shape  = max(max(body, halo), max(arm_h * 0.85, arm_v * 0.85));

        drone_layer = max(drone_layer, shape * t_in_cone * 1.30);
    }

    // Combine beam + drones (both use the same red-amber color)
    float intensity = max(beam, drone_layer);
    float alpha = intensity * duty_fade * 0.95;
    if (alpha < 0.03) discard;
    fragColor = vec4(beam_color, clamp(alpha, 0.0, 1.0));
}
"""


class CyberDroneSpotlightEffect(ShaderEffect):
    def __init__(self, viewport, intensity: float = 0.6):
        super().__init__(viewport)
        self.render_priority = 6.5    # In front of skyline (6.0), behind signs (7.0)
        self.intensity = intensity
        self._time = 0.0
        self._sweep_phase = 0.0

    def compile_shader(self):
        try:
            vert = shaders.compileShader(VERTEX, GL_VERTEX_SHADER)
            frag = shaders.compileShader(FRAGMENT, GL_FRAGMENT_SHADER)
            return shaders.compileProgram(vert, frag)
        except Exception as e:
            print(f"CyberDroneSpotlight compile error: {e}")
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
        self._sweep_phase += dt

    def render(self, state: Dict):
        if not self.enabled or self.intensity < 0.01:
            return
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        glUniform1f(glGetUniformLocation(self.shader, "u_sweep_phase"), self._sweep_phase)
        glUniform1f(glGetUniformLocation(self.shader, "u_time"), self._time)
        glUniform1f(glGetUniformLocation(self.shader, "u_intensity"), self.intensity)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
