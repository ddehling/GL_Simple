"""Test shader — a blue ball that bounces around in *physical layout
space* (not FBO space).

Each fragment samples its own physical (u, v) position from a per-group
metadata atlas (state["group_metadata"][gid]["pos_u"/"pos_v"]) uploaded
as an RG32F texture. Coordinates are in normalized [-0.5, +0.5] with
origin at the composite canvas center, so the ball moves through real
physical space — pixels not covered by any strip carry NaN in the atlas
and are skipped (output transparent). The same shader on different
group canvases would only light up the LEDs whose physical positions
fall inside the ball's radius.

Pattern B (fullscreen-quad atmospheric layer): straight-alpha output,
depth disabled around render() and restored on exit. See
docs/shader_info.txt for the alpha and depth-state rules.
"""
from __future__ import annotations

import ctypes
from typing import Dict

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders

from renderer.effects.base import ShaderEffect


# ---------------------------------------------------------------------------
# Wrapper — the function the EventScheduler calls
# ---------------------------------------------------------------------------

def shader_test_bouncing_ball(state, outstate,
                              radius: float = 0.1,
                              speed_x: float = 0.25,
                              speed_y: float = 0.31,
                              color_rgb=(0.0, 0.3, 1.0)):
    """Spawn / tick / clean up the bouncing-ball test shader.

    Reads ``state["group_metadata"][group_id]`` to get this group's
    per-pixel ``pos_u`` / ``pos_v`` atlases, uploads them as a texture
    once on first call, then animates the ball in normalized
    [-0.5, +0.5] space. The ball starts at the canvas center.

    ``radius`` is in normalized space (0.1 = 20% of the canvas's
    smaller-axis half-width). Velocities are chosen to be slightly
    incommensurate so the bounce path doesn't loop trivially.
    """
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    if shader_renderer is None:
        print("[test_bouncing_ball] shader_renderer missing in outstate")
        return
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        print(f"[test_bouncing_ball] viewport {frame_id} missing")
        return

    # First-call init: pull pos_u / pos_v from the metadata atlas, hand
    # them to the effect so it can build the physical-position texture
    # at GL setup time.
    if state['count'] == 0:
        # Resolve the active group_id for this frame_id. ``group_ids``
        # on the renderer is the list of canvas ids in frame-order, so
        # frame_id 0 → first group, etc. (mirrors how event_map's
        # ``{"group": "leaves"}`` is dispatched.)
        group_ids = getattr(shader_renderer, 'group_ids', None) or []
        if 0 <= frame_id < len(group_ids):
            gid = group_ids[frame_id]
        else:
            gid = "leaves"
        meta = (outstate.get("group_metadata") or {}).get(gid)
        if meta is None:
            print(f"[test_bouncing_ball] no group_metadata for {gid!r}; "
                  f"need state['group_metadata'][gid]['pos_u'/'pos_v']")
            return
        pos_u = meta["pos_u"]
        pos_v = meta["pos_v"]
        try:
            eff = viewport.add_effect(
                BouncingBallEffect,
                pos_u=pos_u, pos_v=pos_v,
                radius=radius,
                speed=(float(speed_x), float(speed_y)),
                color=tuple(color_rgb),
            )
            state['effect'] = eff
            print(f"[test_bouncing_ball] init on frame {frame_id} (group={gid})")
        except Exception as e:
            print(f"[test_bouncing_ball] init failed: {e}")
            import traceback; traceback.print_exc()
            return

    if state['count'] == -1:
        if 'effect' in state:
            try:
                viewport.effects.remove(state['effect'])
            except ValueError:
                pass
            state['effect'].cleanup()
            print(f"[test_bouncing_ball] cleaned up on frame {frame_id}")


# ---------------------------------------------------------------------------
# Effect — Pattern B fullscreen quad with a position-atlas texture lookup
# ---------------------------------------------------------------------------

class BouncingBallEffect(ShaderEffect):
    """Fullscreen-quad shader that lights pixels whose physical (u, v)
    lies inside a bouncing ball's radius.

    Per-pixel positions come from ``pos_u`` / ``pos_v`` numpy arrays
    (same shape as the FBO) packed into an RG32F texture. The shader
    samples this texture at each fragment to get its physical
    coordinates; fragments without a physical layout (NaN in the
    atlas) emit zero alpha so they don't paint."""

    def __init__(self, viewport,
                 pos_u: np.ndarray, pos_v: np.ndarray,
                 radius: float = 0.1,
                 speed: tuple = (0.25, 0.31),
                 color: tuple = (0.0, 0.3, 1.0)):
        super().__init__(viewport)
        self.render_priority = 5

        # Static physics state
        self.radius = float(radius)
        self.color = (float(color[0]), float(color[1]), float(color[2]))
        # Position + velocity in normalized [-0.5, +0.5] space.
        self.pos = np.array([0.0, 0.0], dtype=np.float32)
        self.vel = np.array([float(speed[0]), float(speed[1])], dtype=np.float32)

        # Position atlas — packed into RG32F at setup_buffers time.
        # Keep references so they don't get GC'd before upload.
        if pos_u.shape != pos_v.shape:
            raise ValueError(
                f"pos_u shape {pos_u.shape} != pos_v shape {pos_v.shape}"
            )
        self._pos_u = np.ascontiguousarray(pos_u, dtype=np.float32)
        self._pos_v = np.ascontiguousarray(pos_v, dtype=np.float32)
        self.pos_tex = None

        # Buffers populated by setup_buffers()
        self.position_VBO = None
        self.EBO = None

        # Fullscreen quad geometry (Pattern B template)
        self._vertices = np.array([
            [-1.0, -1.0],
            [ 1.0, -1.0],
            [ 1.0,  1.0],
            [-1.0,  1.0],
        ], dtype=np.float32)
        self._indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

    # ----- shader source -----
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
        # ``uPosMap`` carries (pos_u, pos_v) per pixel. NaN marks
        # pixels with no physical layout — those output 0 alpha so
        # they don't paint into the FBO.
        return """
        #version 310 es
        precision highp float;
        in vec2 vUV;
        uniform sampler2D uPosMap;
        uniform vec2  uBallPos;
        uniform float uRadius;
        uniform vec3  uColor;
        out vec4 fragColor;

        void main() {
            vec2 p = texture(uPosMap, vUV).rg;
            if (isnan(p.x) || isnan(p.y)) {
                fragColor = vec4(0.0);
                return;
            }
            float d = length(p - uBallPos);
            if (d > uRadius) {
                fragColor = vec4(0.0);
                return;
            }
            // Soft edge for visual nicety: full alpha at the center,
            // smoothstep falloff over the outer 30% of the radius.
            float a = 1.0 - smoothstep(uRadius * 0.7, uRadius, d);
            fragColor = vec4(uColor, a);
        }
        """

    def compile_shader(self):
        vert = shaders.compileShader(self.get_vertex_shader(), GL_VERTEX_SHADER)
        frag = shaders.compileShader(self.get_fragment_shader(), GL_FRAGMENT_SHADER)
        return shaders.compileProgram(vert, frag)

    def setup_buffers(self):
        # VAO + VBO + EBO for the fullscreen quad.
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

        # Pack pos_u + pos_v into a single RG32F texture sized to the
        # FBO. NEAREST filtering — the per-pixel atlas is exact, and
        # bilinear smoothing across NaN/non-NaN boundaries would yield
        # NaN inside legitimate strip pixels.
        #
        # The atlas is uploaded VERTICALLY FLIPPED. The atlas is
        # indexed in the post-flipud numpy convention used by the rest
        # of the engine (np[0] = top of display). OpenGL stores the
        # first scanline of the upload at v=0 (bottom of texture in
        # GLSL convention), and a fragment at vUV.y=0 renders to the
        # OpenGL viewport bottom — which after the runtime's
        # np.flipud at readback becomes the bottom of the post-flipud
        # numpy frame. So fragment writing post-flipud row H-1 needs
        # to sample atlas[H-1]; uploading the atlas flipped lines
        # those up. Confirmed by tracing: without this flip, all
        # rows' colors get assigned to inverted output rows and the
        # ball appears on the wrong strip relative to its physical
        # position.
        h, w = self._pos_u.shape
        rg = np.empty((h, w, 2), dtype=np.float32)
        rg[..., 0] = self._pos_u
        rg[..., 1] = self._pos_v
        rg = np.ascontiguousarray(np.flipud(rg))
        self.pos_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.pos_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, w, h, 0,
                     GL_RG, GL_FLOAT, rg)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)

    # ----- per-frame -----
    def update(self, dt: float, state: Dict):
        if not self.enabled:
            return
        self.pos += self.vel * dt
        # Bounce against the ±0.5 walls, with the ball edge (radius)
        # held inside so the ball doesn't graze through.
        bound = 0.5 - self.radius
        for axis in (0, 1):
            if self.pos[axis] > bound:
                self.pos[axis] = bound
                self.vel[axis] = -self.vel[axis]
            elif self.pos[axis] < -bound:
                self.pos[axis] = -bound
                self.vel[axis] = -self.vel[axis]
        # Diagnostic: log position once per second so we can tell
        # whether the ball is actually moving (vs stuck at center).
        # Remove once the test is stable.
        self._debug_t = getattr(self, "_debug_t", 0.0) + dt
        if self._debug_t >= 1.0:
            print(f"[test_ball] pos=({self.pos[0]:+.3f}, {self.pos[1]:+.3f}) "
                  f"vel=({self.vel[0]:+.3f}, {self.vel[1]:+.3f})")
            self._debug_t = 0.0

    def render(self, state: Dict):
        if not self.enabled or not self.shader:
            return
        # Pattern B: depth disabled while drawing this layer; restore
        # the renderer's defaults on exit so the next effect (or the
        # absence of one) sees a clean depth state.
        glDepthFunc(GL_ALWAYS)
        glDepthMask(GL_FALSE)
        try:
            glUseProgram(self.shader)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.pos_tex)
            glUniform1i(glGetUniformLocation(self.shader, "uPosMap"), 0)
            glUniform2f(glGetUniformLocation(self.shader, "uBallPos"),
                        float(self.pos[0]), float(self.pos[1]))
            glUniform1f(glGetUniformLocation(self.shader, "uRadius"),
                        float(self.radius))
            glUniform3f(glGetUniformLocation(self.shader, "uColor"),
                        float(self.color[0]),
                        float(self.color[1]),
                        float(self.color[2]))
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
            if self.pos_tex is not None:
                glDeleteTextures(1, [self.pos_tex])
                self.pos_tex = None
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
