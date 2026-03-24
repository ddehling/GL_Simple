"""Flat Smooth display mode — magnified pixel blit of the FBO."""

from OpenGL.GL import *
from .base import DisplayMode


class FlatSmooth(DisplayMode):
    """Blits the FBO to the window with GL_NEAREST scaling."""

    def init(self, viewport):
        # Nothing to compile or allocate
        self._initialized = True

    def render(self, viewport):
        glDisable(GL_SCISSOR_TEST)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, viewport.fbo)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        glViewport(0, 0, viewport.display_width, viewport.display_height)
        glBlitFramebuffer(
            0, 0, viewport.width, viewport.height,
            0, 0, viewport.display_width, viewport.display_height,
            GL_COLOR_BUFFER_BIT, GL_NEAREST,
        )
        glEnable(GL_SCISSOR_TEST)

    def cleanup(self):
        pass
