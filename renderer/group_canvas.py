"""GroupCanvas — one logical canvas (FBO + effects + readback).

A GroupCanvas owns:
  - an OpenGL framebuffer at native LED resolution (width x height)
  - the list of shader effects that draw into it
  - ping-pong Pixel Pack Buffers for asynchronous CPU-side readback
  - the per-frame update/render/get_frame loop scoped to this canvas

It does NOT own anything related to the GLFW window or how the canvas is
displayed on the desktop — that's ShaderViewport's job.

Phase 2 of the multi-project refactor introduces this class. ShaderViewport
keeps its public API (still the iteration unit driven by RenderPipeline)
and delegates FBO concerns to its inner GroupCanvas. Later phases will
allow multiple GroupCanvas instances per project (one per logical group)
without touching the window-display layer.
"""
from __future__ import annotations

import ctypes
from typing import Dict

import glfw
import numpy as np
from OpenGL.GL import *


class GroupCanvas:
    def __init__(self, group_id: str, width: int, height: int,
                 glfw_window, frame_id: int = 0):
        self.id = group_id
        self.frame_id = frame_id  # legacy index; kept until Phase 6 migration
        self.width = width        # FBO width  (e.g. number of LED strips)
        self.height = height      # FBO height (e.g. LEDs per strip)
        self.glfw_window = glfw_window
        self.flip_x = False       # Horizontal flip applied at FBO readback

        self.effects: list = []

        self.fbo = None
        self.color_texture = None
        self.depth_texture = None

        # Ping-pong PBO readback state — populated by init_framebuffer().
        self._pbos = None
        self._pbo_size = 0
        self._pbo_index = 0
        self._pbo_primed = False

    # --- aliases retained for compatibility with FanGeometry / display modes ---
    @property
    def led_width(self) -> int:
        return self.width

    @property
    def led_height(self) -> int:
        return self.height

    def _make_current(self):
        """Activate GL context (no-op when window is None, i.e. EGL headless)."""
        if self.glfw_window is not None:
            glfw.make_context_current(self.glfw_window)

    # ------------------------------------------------------------------
    # Framebuffer setup
    # ------------------------------------------------------------------

    def init_framebuffer(self):
        """Create the FBO + ping-pong PBOs."""
        self._make_current()

        # Color texture
        self.color_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.color_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # Depth texture
        self.depth_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.depth_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16,
                     self.width, self.height,
                     0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # Framebuffer
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, self.color_texture, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                               GL_TEXTURE_2D, self.depth_texture, 0)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Framebuffer incomplete: {status}")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Pixel Pack Buffers for asynchronous glReadPixels (ping-pong, 2 PBOs).
        # Frame N issues a non-blocking read into pbo[write], while frame N-1's
        # data is mapped from pbo[read]. This decouples GPU readback from CPU.
        self._pbo_size = self.width * self.height * 4  # RGBA8
        self._pbos = glGenBuffers(2)
        for pbo in self._pbos:
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo)
            glBufferData(GL_PIXEL_PACK_BUFFER, self._pbo_size, None, GL_STREAM_READ)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        self._pbo_index = 0
        self._pbo_primed = False

        print(f"[GroupCanvas] Framebuffer created: {self.width}x{self.height} "
              f"(group={self.id})")

    # ------------------------------------------------------------------
    # Effect pipeline
    # ------------------------------------------------------------------

    def add_effect(self, effect_class, **params):
        """Add a shader effect to this canvas's render pipeline."""
        self._make_current()
        effect = effect_class(self, **params)
        effect.init()
        self.effects.append(effect)
        if not getattr(effect, '_silent', False):
            print(f"[GroupCanvas] Added effect: {effect.__class__.__name__} "
                  f"(group={self.id})")
        return effect

    def clear(self):
        """Clear the FBO."""
        self._make_current()
        glDepthMask(GL_TRUE)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)
        glScissor(0, 0, self.width, self.height)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def update(self, dt: float, state: Dict):
        """Update all effects (sorted by priority)."""
        sorted_effects = sorted(self.effects, key=lambda e: getattr(e, 'render_priority', 0))
        for effect in sorted_effects:
            if effect.enabled:
                effect.update(dt, state)

    def render(self, state: Dict):
        """Render effects into the FBO at native LED resolution."""
        sorted_effects = sorted(self.effects, key=lambda e: getattr(e, 'render_priority', 0))
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)
        glScissor(0, 0, self.width, self.height)
        for effect in sorted_effects:
            if effect.enabled:
                effect.render(state)

    # ------------------------------------------------------------------
    # Frame readback for DMX output
    # ------------------------------------------------------------------

    def get_frame(self) -> np.ndarray:
        """Read FBO into a numpy RGB array (async via ping-pong PBOs).

        Returns a (height, width, 3) uint8 array. The first call returns
        zeros because PBO ping-pong introduces 1 frame of latency.
        """
        self._make_current()
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        write_idx = self._pbo_index
        read_idx = 1 - self._pbo_index

        # 1) Issue async read of the current frame into pbo[write_idx]
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self._pbos[write_idx])
        glReadPixels(0, 0, self.width, self.height,
                     GL_RGBA, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))

        # 2) Map and copy out the PREVIOUS frame's data from pbo[read_idx]
        if self._pbo_primed:
            glBindBuffer(GL_PIXEL_PACK_BUFFER, self._pbos[read_idx])
            ptr = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, self._pbo_size, GL_MAP_READ_BIT)
            try:
                buf = (ctypes.c_ubyte * self._pbo_size).from_address(int(ptr))
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(
                    self.height, self.width, 4).copy()
            finally:
                glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
        else:
            frame = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            self._pbo_primed = True

        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        self._pbo_index = read_idx

        # Flip Y axis and drop alpha
        frame = np.flipud(frame)
        if self.flip_x:
            frame = np.fliplr(frame)
        return np.ascontiguousarray(frame[:, :, :3])

    def cleanup(self):
        """Release effects, FBO, PBOs, textures."""
        self._make_current()
        for effect in self.effects:
            effect.cleanup()
        self.effects = []
        if self.fbo:
            glDeleteFramebuffers(1, [self.fbo])
            self.fbo = None
        if self._pbos is not None:
            glDeleteBuffers(2, self._pbos)
            self._pbos = None
        if self.color_texture:
            glDeleteTextures([self.color_texture])
            self.color_texture = None
        if self.depth_texture:
            glDeleteTextures([self.depth_texture])
            self.depth_texture = None

    def resize(self, new_width: int, new_height: int):
        """Tear down effects + GL handles and rebuild the FBO at new dims.

        Used by project-swap to repoint the canvas at the new project's
        native resolution without recreating the GLFW window. Effects are
        gone after this; the caller is responsible for re-adding any that
        belong to the new project.
        """
        self.cleanup()
        self.width = int(new_width)
        self.height = int(new_height)
        self.init_framebuffer()
