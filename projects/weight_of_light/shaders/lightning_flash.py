"""WoL one-shot lightning flash, scoped to a single box.

Pattern B (fullscreen-quad atmospheric layer). Designed to feel like
real lightning: instantaneous peak, sweep along each strip, leap
between strips on the same receiver, then a longer "blinded" tail
where the surrounding sky is dimmed and recovers.

Per-pixel addressing reads two atlases packed into one RG32F texture:
  * R channel: ``object_id`` — gate the bright flash to one box
  * G channel: ``strip_idx`` — order strips within the box for the
    inter-strip leap

Firing geometry (in seconds since event start):

    firing_time = (strip_idx - first_strip_for_target) * strip_delay
                  + vUV.x * propagation_time

So strips fire in increasing-strip_idx order, and within each strip
the wavefront sweeps from vUV.x=0 (LED 0, bottom of strip / horizon)
toward vUV.x=1 (last LED, top / zenith). Each pixel that has fired
emits ``exp(-age / decay)`` brightness; pixels that haven't fired
yet stay dark, so the wavefront edge is sharp.

Sky dim: pixels that aren't on the target box get a translucent
black overlay that ramps in within 50 ms (synchronized with the
flash), holds, then fades out linearly over ``dim_recovery`` seconds.
``dim_enable=False`` disables the dim entirely — used on the Ground
group where surrounding-arc dimming has no narrative purpose.

Schedule the same shader on both Sky and Ground groups (with
distinct ``functools.partial`` instances so the scheduler's dedup
doesn't drop one). Each group's instance reads its group's atlas
through the wrapper's group_metadata lookup.
"""
from __future__ import annotations

import ctypes
from typing import Dict

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders

from renderer.effects.base import ShaderEffect


def shader_wol_lightning_flash(state, outstate,
                               target_object_id: int = -1,
                               color_rgb=(1.0, 1.0, 1.0),
                               strip_delay: float = 0.04,
                               propagation_time: float = 0.06,
                               decay: float = 0.4,
                               dim_alpha: float = 0.4,
                               dim_recovery: float = 2.5,
                               dim_enable: bool = True):
    """Spawn / tick / clean up a dramatic lightning flash.

    Schedule with ``frame_id`` set to the target group's frame id
    and a duration that covers the full effect (strip propagation +
    decay tail + dim recovery). 3.0–3.5 s is a comfortable default.

    ``target_object_id``: numeric id of the box to flash; -1 disables.
    ``strip_delay``: seconds between successive strips firing on the
        target box. 0.04 s reads as a tight inter-strip leap.
    ``propagation_time``: seconds for the wavefront to traverse one
        strip end-to-end. 0.06 s feels like a near-instantaneous bolt
        with just enough sweep to register.
    ``decay``: e-fold time of brightness after a pixel fires. 0.4 s
        gives a sharp head + lingering glow.
    ``dim_alpha``: peak black-overlay opacity on non-target sky
        pixels. 0.4 = 40 % darkening.
    ``dim_recovery``: seconds for the dim to fade back to 0.
    ``dim_enable``: True for the Sky group, False for Ground.
    """
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        return

    if state['count'] == 0:
        # Resolve which group this frame_id maps to so we read the
        # right atlas (Sky on Sky-frame, Ground on Ground-frame).
        group_ids = getattr(shader_renderer, 'group_ids', None) or []
        if 0 <= frame_id < len(group_ids):
            gid = group_ids[frame_id]
        else:
            gid = "Sky"
        meta = (outstate.get("group_metadata") or {}).get(gid)
        if meta is None:
            print(f"[wol_lightning_flash] no group_metadata for {gid!r}; "
                  f"flash inert")
            return
        object_id_atlas = meta.get("object_id")
        strip_idx_atlas = meta.get("strip_idx")
        if object_id_atlas is None or strip_idx_atlas is None:
            print(f"[wol_lightning_flash] group {gid!r} missing object_id "
                  f"or strip_idx atlas; flash inert")
            return

        # Find the lowest strip_idx among strips on this group that
        # belong to the target object — used as the base for the
        # inter-strip firing offset. If the box has 3 sky strips at
        # idx {3,4,5}, the offsets within the flash are {0, delay,
        # 2*delay}.
        strip_table = outstate.get("strip_table") or []
        first_strip_idx = 0
        target_idxs = [
            int(e["strip_idx"]) for e in strip_table
            if int(e.get("object_id", -1)) == int(target_object_id)
            and e.get("group_id") == gid
        ]
        if target_idxs:
            first_strip_idx = min(target_idxs)

        try:
            eff = viewport.add_effect(
                LightningFlashEffect,
                object_id_atlas=object_id_atlas,
                strip_idx_atlas=strip_idx_atlas,
                target_object_id=int(target_object_id),
                first_strip_idx=int(first_strip_idx),
                color=tuple(color_rgb),
                strip_delay=float(strip_delay),
                propagation_time=float(propagation_time),
                decay=float(decay),
                dim_alpha=float(dim_alpha),
                dim_recovery=float(dim_recovery),
                dim_enable=bool(dim_enable),
            )
            state['effect'] = eff
        except Exception as e:
            print(f"[wol_lightning_flash] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    eff = state.get('effect')
    if eff is not None:
        # Push elapsed time directly (in seconds) — the shader does
        # all timing in seconds rather than fractions, so the firing
        # offsets stay in real units regardless of event duration.
        eff.elapsed = float(state.get('elapsed_time', 0.0))

    if state['count'] == -1:
        if 'effect' in state:
            try:
                viewport.effects.remove(state['effect'])
            except ValueError:
                pass
            state['effect'].cleanup()


class LightningFlashEffect(ShaderEffect):
    def __init__(self, viewport,
                 object_id_atlas: np.ndarray,
                 strip_idx_atlas: np.ndarray,
                 target_object_id: int,
                 first_strip_idx: int,
                 color: tuple = (1.0, 1.0, 1.0),
                 strip_delay: float = 0.04,
                 propagation_time: float = 0.06,
                 decay: float = 0.4,
                 dim_alpha: float = 0.4,
                 dim_recovery: float = 2.5,
                 dim_enable: bool = True):
        super().__init__(viewport)
        # Front of the stack so the bright bolt overlays the day/night
        # sky, stars, rain, etc.
        self.render_priority = 9.0

        self.target_object_id = int(target_object_id)
        self.first_strip_idx = int(first_strip_idx)
        self.color = (float(color[0]), float(color[1]), float(color[2]))
        self.strip_delay = float(strip_delay)
        self.propagation_time = float(propagation_time)
        self.decay = max(float(decay), 1e-3)
        self.dim_alpha = float(dim_alpha)
        self.dim_recovery = max(float(dim_recovery), 1e-3)
        self.dim_enable = bool(dim_enable)
        self.elapsed = 0.0

        # Pack the two atlases into one RG32F texture so the fragment
        # shader does one fetch per pixel. Same vertical-flip
        # convention as test_bouncing_ball / rainbow.
        if object_id_atlas.shape != strip_idx_atlas.shape:
            raise ValueError(
                f"atlas shape mismatch: object_id={object_id_atlas.shape} "
                f"strip_idx={strip_idx_atlas.shape}"
            )
        h, w = object_id_atlas.shape
        rg = np.empty((h, w, 2), dtype=np.float32)
        rg[..., 0] = object_id_atlas
        rg[..., 1] = strip_idx_atlas
        self._atlas_rg = np.ascontiguousarray(np.flipud(rg))
        self.atlas_tex = None

        self.position_VBO = None
        self.EBO = None
        self._vertices = np.array([
            [-1.0, -1.0],
            [ 1.0, -1.0],
            [ 1.0,  1.0],
            [-1.0,  1.0],
        ], dtype=np.float32)
        self._indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        layout(location = 0) in vec2 position;
        out vec2 vUV;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            vUV = position * 0.5 + 0.5;
        }
        """

    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        in vec2 vUV;
        uniform sampler2D uMeta;       // RG32F: R=object_id, G=strip_idx
        uniform float u_target;
        uniform float u_first_strip;
        uniform float u_elapsed;
        uniform float u_strip_delay;
        uniform float u_propagation;
        uniform float u_decay;
        uniform float u_dim_alpha;
        uniform float u_dim_recovery;
        uniform float u_dim_enable;
        uniform vec3  u_color;
        out vec4 fragColor;

        void main() {
            vec2 m = texture(uMeta, vUV).rg;
            float oid = m.r;
            float sidx = m.g;

            // No-strip pixels (gaps in the canvas) — never paint.
            if (oid < -0.5) { fragColor = vec4(0.0); return; }

            bool is_target = abs(oid - u_target) < 0.5;

            if (is_target) {
                // Per-strip leap: each strip in the target object
                // fires at increasing offsets driven by strip_idx.
                float strip_offset = (sidx - u_first_strip) * u_strip_delay;
                // Per-pixel sweep along the strip.
                float pixel_offset = vUV.x * u_propagation;
                float firing = strip_offset + pixel_offset;
                float age = u_elapsed - firing;
                if (age < 0.0) {
                    fragColor = vec4(0.0);
                    return;
                }
                // Instant peak (no attack ramp), exponential decay.
                float a = exp(-age / u_decay);
                fragColor = vec4(u_color, clamp(a, 0.0, 1.0));
            } else {
                // Surrounding sky dim — only on Sky pass.
                if (u_dim_enable < 0.5) {
                    fragColor = vec4(0.0);
                    return;
                }
                // Dim ramps in over 50 ms (matches the flash front),
                // then linearly fades back to 0 over u_dim_recovery.
                float ramp_in = clamp(u_elapsed / 0.05, 0.0, 1.0);
                float fade_out = max(0.0,
                    1.0 - u_elapsed / u_dim_recovery);
                float dim = u_dim_alpha * ramp_in * fade_out;
                fragColor = vec4(0.0, 0.0, 0.0, clamp(dim, 0.0, 1.0));
            }
        }
        """

    def compile_shader(self):
        vert = shaders.compileShader(self.get_vertex_shader(), GL_VERTEX_SHADER)
        frag = shaders.compileShader(self.get_fragment_shader(), GL_FRAGMENT_SHADER)
        return shaders.compileProgram(vert, frag)

    def setup_buffers(self):
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        self.position_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.position_VBO)
        glBufferData(GL_ARRAY_BUFFER, self._vertices.nbytes,
                     self._vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self._indices.nbytes,
                     self._indices, GL_STATIC_DRAW)
        glBindVertexArray(0)

        h, w, _ = self._atlas_rg.shape
        self.atlas_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.atlas_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, w, h, 0,
                     GL_RG, GL_FLOAT, self._atlas_rg)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)

    def update(self, dt: float, state: Dict):
        # Wrapper pushes elapsed each frame; nothing to integrate here.
        pass

    def render(self, state: Dict):
        if not self.enabled or not self.shader:
            return
        # Cheap early-out once everything has decayed below visible.
        # Max remaining alpha = max(flash decay, dim fade-out).
        max_firing = (
            (max(0.0, 18.0 - self.first_strip_idx)) * self.strip_delay
            + self.propagation_time
        )
        flash_remaining = (
            self.elapsed > max_firing
            and (self.elapsed - max_firing) / self.decay > 6.0
        )
        dim_remaining = (
            self.dim_enable and self.elapsed < self.dim_recovery
        )
        if flash_remaining and not dim_remaining:
            return

        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        try:
            glUseProgram(self.shader)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.atlas_tex)
            glUniform1i(glGetUniformLocation(self.shader, "uMeta"), 0)
            glUniform1f(glGetUniformLocation(self.shader, "u_target"),
                        float(self.target_object_id))
            glUniform1f(glGetUniformLocation(self.shader, "u_first_strip"),
                        float(self.first_strip_idx))
            glUniform1f(glGetUniformLocation(self.shader, "u_elapsed"),
                        float(self.elapsed))
            glUniform1f(glGetUniformLocation(self.shader, "u_strip_delay"),
                        float(self.strip_delay))
            glUniform1f(glGetUniformLocation(self.shader, "u_propagation"),
                        float(self.propagation_time))
            glUniform1f(glGetUniformLocation(self.shader, "u_decay"),
                        float(self.decay))
            glUniform1f(glGetUniformLocation(self.shader, "u_dim_alpha"),
                        float(self.dim_alpha))
            glUniform1f(glGetUniformLocation(self.shader, "u_dim_recovery"),
                        float(self.dim_recovery))
            glUniform1f(glGetUniformLocation(self.shader, "u_dim_enable"),
                        1.0 if self.dim_enable else 0.0)
            glUniform3f(glGetUniformLocation(self.shader, "u_color"),
                        *self.color)
            glBindVertexArray(self.VAO)
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)
            glBindTexture(GL_TEXTURE_2D, 0)
            glUseProgram(0)
        finally:
            glDepthFunc(GL_LESS)
            glDepthMask(GL_TRUE)

    def cleanup(self):
        try:
            if self.atlas_tex is not None:
                glDeleteTextures(1, [self.atlas_tex])
                self.atlas_tex = None
            if self.position_VBO is not None:
                glDeleteBuffers(1, [self.position_VBO])
                self.position_VBO = None
            if self.EBO is not None:
                glDeleteBuffers(1, [self.EBO])
                self.EBO = None
            if self.VAO is not None:
                glDeleteVertexArrays(1, [self.VAO])
                self.VAO = None
            if self.shader:
                glDeleteProgram(self.shader)
                self.shader = None
        except Exception:
            pass
