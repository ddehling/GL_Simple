"""LED Dots display mode — instanced circles, one per physical LED.

Works in both flat (rectangular grid) and fan (semicircle) layouts,
controlled by the ``fan`` parameter passed at construction time.
"""

import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from .base import DisplayMode


class LEDDots(DisplayMode):
    """Renders each LED as an instanced circle colored by the FBO texel."""

    _VERT = """#version 310 es
precision highp float;
layout(location = 0) in vec2 aQuad;
layout(location = 1) in vec2 iPos;
layout(location = 2) in vec2 iTexCoord;
out vec2 vQuad;
flat out vec2 vUV;
uniform float uRadius;
uniform float uZoom;
uniform vec2 uPan;
void main() {
    vec2 world = (iPos + aQuad * uRadius) * uZoom + uPan;
    gl_Position = vec4(world, 0.0, 1.0);
    vQuad = aQuad;
    vUV = iTexCoord;
}
"""

    _FRAG = """#version 310 es
precision highp float;
in vec2 vQuad;
flat in vec2 vUV;
uniform sampler2D uTexture;
out vec4 outColor;
void main() {
    float d = length(vQuad);
    if (d > 1.0) discard;
    float alpha = smoothstep(1.0, 0.7, d);
    vec4 color = texture(uTexture, vUV);
    outColor = vec4(color.rgb * alpha, alpha);
}
"""

    def __init__(self, fan=False):
        super().__init__()
        self.fan = fan
        self._shader = None
        self._vao = None
        self._quad_vbo = None
        self._inst_vbo = None
        self._n_instances = 0
        self._tex_loc = None
        self._radius_loc = None
        self._dot_radius = 0.0

    def init(self, viewport):
        viewport._ensure_geometry()

        if self.fan:
            inst_2d, self._dot_radius = viewport._geometry.compute_fan_dots()
        else:
            inst_2d, self._dot_radius = viewport._geometry.compute_flat_dots()

        inst_np = inst_2d.flatten()
        self._n_instances = viewport.led_width * viewport.led_height

        # Compile shader (only once — survives mark_dirty)
        if self._shader is None:
            vert = shaders.compileShader(self._VERT, GL_VERTEX_SHADER)
            frag = shaders.compileShader(self._FRAG, GL_FRAGMENT_SHADER)
            self._shader = shaders.compileProgram(vert, frag)
            self._tex_loc = glGetUniformLocation(self._shader, "uTexture")
            self._radius_loc = glGetUniformLocation(self._shader, "uRadius")
            self._zoom_loc = glGetUniformLocation(self._shader, "uZoom")
            self._pan_loc = glGetUniformLocation(self._shader, "uPan")

        # Unit quad (6 vertices, 2 triangles)
        quad = np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1],
                        dtype=np.float32)

        # Clean up old buffers if re-initializing
        if self._vao:
            glDeleteVertexArrays(1, [self._vao])
        if self._quad_vbo:
            glDeleteBuffers(1, [self._quad_vbo])
        if self._inst_vbo:
            glDeleteBuffers(1, [self._inst_vbo])

        self._vao = glGenVertexArrays(1)
        glBindVertexArray(self._vao)

        self._quad_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

        self._inst_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._inst_vbo)
        glBufferData(GL_ARRAY_BUFFER, inst_np.nbytes, inst_np, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glVertexAttribDivisor(1, 1)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
        glVertexAttribDivisor(2, 1)

        glBindVertexArray(0)
        self._initialized = True

        layout = 'fan' if self.fan else 'flat'
        print(f"[LEDDots] {layout}: {self._n_instances} instances, "
              f"radius={self._dot_radius:.4f}")

    def render(self, viewport):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, viewport.display_width, viewport.display_height)
        glDisable(GL_SCISSOR_TEST)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glClearColor(0.02, 0.02, 0.02, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self._shader)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, viewport.color_texture)
        glUniform1i(self._tex_loc, 0)
        glUniform1f(self._radius_loc, self._dot_radius)
        zoom = viewport.zoom if self.fan else 1.0
        pan_x = viewport.pan_x if self.fan else 0.0
        pan_y = viewport.pan_y if self.fan else 0.0
        glUniform1f(self._zoom_loc, zoom)
        glUniform2f(self._pan_loc, pan_x, pan_y)

        glBindVertexArray(self._vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, self._n_instances)
        glBindVertexArray(0)

        glUseProgram(0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_SCISSOR_TEST)

    def cleanup(self):
        if self._vao:
            glDeleteVertexArrays(1, [self._vao])
            self._vao = None
        if self._quad_vbo:
            glDeleteBuffers(1, [self._quad_vbo])
            self._quad_vbo = None
        if self._inst_vbo:
            glDeleteBuffers(1, [self._inst_vbo])
            self._inst_vbo = None
        if self._shader:
            glDeleteProgram(self._shader)
            self._shader = None
