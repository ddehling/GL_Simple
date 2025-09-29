"""Base classes for shader effects"""
from OpenGL.GL import *
from typing import Dict
import numpy as np

class ShaderEffect:
    """Base class for shader-based effects"""
    def __init__(self, viewport):
        self.viewport = viewport
        self.enabled = True
        self.shader = None
        self.VAO = None
        self.VBOs = []
        self.EBO = None
        
    def init(self):
        """Initialize shader and buffers"""
        try:
            self.shader = self.compile_shader()
            self.setup_buffers()
            print(f"    ✓ {self.__class__.__name__} shader compiled successfully")
        except Exception as e:
            print(f"    ✗ Error initializing {self.__class__.__name__}: {e}")
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