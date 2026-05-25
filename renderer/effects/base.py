"""Base classes for shader effects"""
from OpenGL.GL import *
from typing import Dict
import numpy as np

class ShaderEffect:
    """Base class for shader-based effects"""
    def __init__(self, viewport):
        # `viewport` is whichever object owns the FBO this effect draws into.
        # Phase 2 of the multi-project refactor: this is now a GroupCanvas
        # (passed by GroupCanvas.add_effect). The `self.viewport` name is
        # kept as an alias of `self.canvas` for backwards compatibility with
        # existing effects that read self.viewport.{width,height,fbo,...}.
        self.canvas = viewport
        self.viewport = viewport
        self.enabled = True
        self.shader = None
        self.VAO = None
        self.VBOs = []
        self.EBO = None
        self.render_priority = 0  # Higher priority renders later (post-processing = 1000)
        # Lazy cache for glGetUniformLocation results - see uniform() below.
        self._uniform_cache = {}
        
    def init(self):
        """Initialize shader and buffers"""
        try:
            self.shader = self.compile_shader()
            self.setup_buffers()
            print(f"    [OK] {self.__class__.__name__} shader compiled successfully")
        except Exception as e:
            print(f"    [ERROR] Error initializing {self.__class__.__name__}: {e}")
            self.enabled = False
            raise
        
    def compile_shader(self):
        """Compile and link shaders - override in subclasses"""
        pass
        
    def setup_buffers(self):
        """Set up VAO, VBO, etc. - override in subclasses"""
        pass
        
    def update(self, dt: float, state: Dict):
        """Update animation state - override in subclasses"""
        pass
        
    def render(self, state: Dict):
        """Render the effect - override in subclasses"""
        if not self.enabled or not self.shader:
            return
            
    def cleanup(self):
        """Clean up OpenGL resources"""
        try:
            if self.VAO:
                glDeleteVertexArrays(1, [self.VAO])
            if self.VBOs:
                glDeleteBuffers(len(self.VBOs), self.VBOs)
            if self.EBO:
                glDeleteBuffers(1, [self.EBO])
            if self.shader:
                glDeleteProgram(self.shader)
        except:
            pass  # Ignore cleanup errors

    def uniform(self, program, name):
        """Lazily-cached drop-in replacement for glGetUniformLocation.

        glGetUniformLocation is a driver call that walks the shader's
        string symbol table - cheap once, wasteful per-frame. This
        method memoizes the result per (program, name), so repeated
        calls from render() pay only a Python dict lookup.

        Usage in any effect (drop-in for glGetUniformLocation):
            loc = self.uniform(self.shader, "iTime")
            glUniform1f(loc, time)

        Inactive uniforms (declared but unused in the linked program)
        come back as -1 from the driver and stay cached as -1;
        glUniform* calls with -1 are silent no-ops, so it's safe to
        speculatively ask for any name.

        Note: the cache is keyed on the program *handle* (an integer).
        If you delete a program and create a new one that happens to
        reuse the same GLuint, cached locations could be stale. None
        of the current effects do that; if you add one that recompiles
        shaders at runtime, clear _uniform_cache after the recompile.
        """
        cache = self._uniform_cache
        key = (program, name)
        loc = cache.get(key)
        if loc is None:
            loc = glGetUniformLocation(program, name)
            cache[key] = loc
        return loc