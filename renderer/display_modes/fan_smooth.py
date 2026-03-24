"""Fan Smooth display mode — textured semicircle mesh."""

import ctypes
from OpenGL.GL import *
from OpenGL.GL import shaders
from .base import DisplayMode


class FanSmooth(DisplayMode):
    """Renders the FBO onto a semicircular fan mesh simulating the physical LED layout."""

    _VERT = """#version 310 es
precision highp float;
layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec2 aTexCoord;
out vec2 vTexCoord;
void main() {
    gl_Position = vec4(aPosition, 0.0, 1.0);
    vTexCoord = aTexCoord;
}
"""

    _FRAG = """#version 310 es
precision highp float;
in vec2 vTexCoord;
uniform sampler2D uTexture;
out vec4 outColor;
void main() {
    outColor = texture(uTexture, vTexCoord);
}
"""

    def __init__(self):
        super().__init__()
        self._shader = None
        self._vao = None
        self._vbo = None
        self._ebo = None
        self._index_count = 0
        self._tex_loc = None

    def init(self, viewport):
        viewport._ensure_geometry()
        verts_2d, indices_np = viewport._geometry.compute_fan_mesh()
        verts_np = verts_2d.flatten()
        self._index_count = len(indices_np)

        # Compile shader (only once — survives mark_dirty)
        if self._shader is None:
            vert = shaders.compileShader(self._VERT, GL_VERTEX_SHADER)
            frag = shaders.compileShader(self._FRAG, GL_FRAGMENT_SHADER)
            self._shader = shaders.compileProgram(vert, frag)
            self._tex_loc = glGetUniformLocation(self._shader, "uTexture")

        # Clean up old buffers if re-initializing
        if self._vao:
            glDeleteVertexArrays(1, [self._vao])
        if self._vbo:
            glDeleteBuffers(1, [self._vbo])
        if self._ebo:
            glDeleteBuffers(1, [self._ebo])

        # VAO / VBO / EBO
        self._vao = glGenVertexArrays(1)
        glBindVertexArray(self._vao)

        self._vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, verts_np.nbytes, verts_np, GL_STATIC_DRAW)

        # position: location 0, 2 floats, stride 16, offset 0
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        # texcoord: location 1, 2 floats, stride 16, offset 8
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))

        self._ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_np.nbytes, indices_np, GL_STATIC_DRAW)

        glBindVertexArray(0)
        self._initialized = True
        print(f"[FanSmooth] Fan mesh: {viewport.led_width} strips x {viewport.led_height} LEDs, "
              f"{self._index_count // 3} triangles")

    def render(self, viewport):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, viewport.display_width, viewport.display_height)
        glDisable(GL_SCISSOR_TEST)
        glDisable(GL_DEPTH_TEST)

        glClearColor(0.05, 0.05, 0.05, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self._shader)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, viewport.color_texture)
        glUniform1i(self._tex_loc, 0)

        glBindVertexArray(self._vao)
        glDrawElements(GL_TRIANGLES, self._index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        glUseProgram(0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_SCISSOR_TEST)

    def cleanup(self):
        if self._vao:
            glDeleteVertexArrays(1, [self._vao])
            self._vao = None
        if self._vbo:
            glDeleteBuffers(1, [self._vbo])
            self._vbo = None
        if self._ebo:
            glDeleteBuffers(1, [self._ebo])
            self._ebo = None
        if self._shader:
            glDeleteProgram(self._shader)
            self._shader = None
