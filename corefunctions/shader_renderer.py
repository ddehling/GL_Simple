import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
import numpy as np
import ctypes
from typing import List, Tuple, Dict, Optional
import time
import platform

# Detect platform
IS_RASPBERRY_PI = platform.machine() in ['aarch64', 'armv7l', 'armv8']

class ShaderRenderer:
    """GPU-based renderer with visible OpenGL window and multiple viewports"""
    def __init__(self, frame_dimensions: List[Tuple[int, int]], window_width=1200, window_height=800):
        self.frame_dimensions = frame_dimensions
        self.num_frames = len(frame_dimensions)
        self.window_width = window_width
        self.window_height = window_height
        self.window = None
        self.viewports = []
        self.ctx_initialized = False
        
        self.init_glfw()
        self.create_window()
        
    def init_glfw(self):
        """Initialize GLFW with OpenGL ES 3.1"""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        if IS_RASPBERRY_PI:
            print("Configuring for Raspberry Pi (OpenGL ES 3.1 + EGL)")
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
            glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
        else:
            print("Configuring for Desktop (OpenGL ES 3.1)")
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        
    def create_window(self):
        """Create a visible OpenGL window"""
        self.window = glfw.create_window(self.window_width, self.window_height, 
                                        "LED Renderer", None, None)
        if not self.window:
            raise RuntimeError("Failed to create OpenGL window")
            
        glfw.make_context_current(self.window)
        
        # OpenGL setup
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_SCISSOR_TEST)
        
        version = glGetString(GL_VERSION)
        if version:
            print(f"OpenGL Version: {version.decode()}")
        glsl_version = glGetString(GL_SHADING_LANGUAGE_VERSION)
        if glsl_version:
            print(f"GLSL Version: {glsl_version.decode()}")
        
        print(f"Created OpenGL window: {self.window_width}x{self.window_height}")
        self.ctx_initialized = True
        
    def create_viewport(self, frame_id: int) -> 'ShaderViewport':
        """Create a viewport for a specific frame"""
        if frame_id >= self.num_frames:
            raise ValueError(f"Invalid frame_id: {frame_id}")
            
        width, height = self.frame_dimensions[frame_id]
        
        # Calculate display scaling - make viewports fill the window nicely
        padding = 20
        
        # Calculate total width needed and scale to fit
        total_content_width = sum(w for w, h in self.frame_dimensions)
        total_padding = padding * (self.num_frames + 1)
        available_width = self.window_width - total_padding
        
        # Scale to fill most of the window
        scale = available_width / total_content_width
        
        # Also check height constraint
        max_height = max(h for w, h in self.frame_dimensions)
        height_scale = (self.window_height - 2 * padding) / max_height
        
        # Use the smaller scale to ensure everything fits
        scale = min(scale, height_scale)
        
        # Calculate scaled dimensions for display
        display_width = int(width * scale)
        display_height = int(height * scale)
        
        # Calculate x position (accumulate previous widths)
        x_offset = padding
        for i in range(frame_id):
            prev_width, _ = self.frame_dimensions[i]
            x_offset += int(prev_width * scale) + padding
        
        # Center vertically
        y_offset = (self.window_height - display_height) // 2
        
        print(f"Creating viewport {frame_id}:")
        print(f"  Framebuffer (LED): {width}x{height}")
        print(f"  Display: {display_width}x{display_height} at ({x_offset}, {y_offset})")
        print(f"  Scale factor: {scale:.2f}")
        
        viewport = ShaderViewport(frame_id, width, height, 
                                 x_offset, y_offset, 
                                 display_width, display_height,
                                 self.window)
        viewport.init_framebuffer()
        self.viewports.append(viewport)
        return viewport
    
    def get_viewport(self, frame_id: int) -> Optional['ShaderViewport']:
        """Get viewport by frame_id"""
        for vp in self.viewports:
            if vp.frame_id == frame_id:
                return vp
        return None
    
    def poll_events(self):
        """Poll GLFW events"""
        glfw.poll_events()
    
    def should_close(self):
        """Check if window should close"""
        return glfw.window_should_close(self.window)
    
    def swap_buffers(self):
        """Swap window buffers"""
        glfw.swap_buffers(self.window)
    
    def clear_window(self):
        """Clear the entire window"""
        glfw.make_context_current(self.window)
        glViewport(0, 0, self.window_width, self.window_height)
        glScissor(0, 0, self.window_width, self.window_height)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
    
    def cleanup(self):
        """Clean up resources"""
        for vp in self.viewports:
            vp.cleanup()
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()


class ShaderViewport:
    """Individual viewport with shader effect pipeline and framebuffer for LED output"""
    def __init__(self, frame_id: int, width: int, height: int, 
                 window_x: int, window_y: int, 
                 display_width: int, display_height: int,
                 glfw_window):
        self.frame_id = frame_id
        self.width = width  # Actual framebuffer size (for LED output)
        self.height = height
        self.window_x = window_x  # Position in window
        self.window_y = window_y
        self.display_width = display_width  # Display size in window (scaled)
        self.display_height = display_height
        self.glfw_window = glfw_window
        self.effects = []
        
        # Framebuffer for LED output (separate from window rendering)
        self.fbo = None
        self.color_texture = None
        self.depth_renderbuffer = None
        
    def init_framebuffer(self):
        """Create framebuffer for offscreen rendering (for LED output)"""
        glfw.make_context_current(self.glfw_window)
        
        # Create color texture
        self.color_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.color_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 
                     0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        # Create depth renderbuffer
        self.depth_renderbuffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_renderbuffer)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, 
                             self.width, self.height)
        
        # Create framebuffer
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 
                              GL_TEXTURE_2D, self.color_texture, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                 GL_RENDERBUFFER, self.depth_renderbuffer)
        
        # Check framebuffer completeness
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Framebuffer incomplete: {status}")
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        print(f"  Framebuffer created: {self.width}x{self.height}")
    
    def add_effect(self, effect_class, **params):
        """Add a shader effect to the rendering pipeline"""
        glfw.make_context_current(self.glfw_window)
        
        effect = effect_class(self, **params)
        effect.init()
        self.effects.append(effect)
        print(f"  Added effect: {effect.__class__.__name__} to frame {self.frame_id}")
        return effect
    
    def clear(self):
        """Clear the viewport in both window and framebuffer"""
        glfw.make_context_current(self.glfw_window)
        
        # Clear framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)
        glScissor(0, 0, self.width, self.height)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        # Clear window viewport region
        glViewport(self.window_x, self.window_y, self.display_width, self.display_height)
        glScissor(self.window_x, self.window_y, self.display_width, self.display_height)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
    
    def update(self, dt: float, state: Dict):
        """Update all effects"""
        for effect in self.effects:
            if effect.enabled:
                effect.update(dt, state)
    
    def render(self, state: Dict):
        """Render effects to both framebuffer and window"""
        glfw.make_context_current(self.glfw_window)
        
        # Render to framebuffer (for LED output at actual resolution)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)
        glScissor(0, 0, self.width, self.height)
        
        for effect in self.effects:
            if effect.enabled:
                effect.render(state)
        
        # Render to window (for visualization at display resolution)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(self.window_x, self.window_y, self.display_width, self.display_height)
        glScissor(self.window_x, self.window_y, self.display_width, self.display_height)
        
        for effect in self.effects:
            if effect.enabled:
                effect.render(state)
    
    def get_frame(self) -> np.ndarray:
        """Read framebuffer into numpy array for LED output"""
        glfw.make_context_current(self.glfw_window)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        pixels = glReadPixels(0, 0, self.width, self.height, 
                             GL_RGBA, GL_UNSIGNED_BYTE)
        frame = np.frombuffer(pixels, dtype=np.uint8).reshape(
            self.height, self.width, 4)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        # Flip Y axis (OpenGL convention)
        return np.flip(frame, axis=0)
    
    def cleanup(self):
        """Clean up resources"""
        glfw.make_context_current(self.glfw_window)
        for effect in self.effects:
            effect.cleanup()
        if self.fbo:
            glDeleteFramebuffers(1, [self.fbo])
        if self.color_texture:
            glDeleteTextures([self.color_texture])
        if self.depth_renderbuffer:
            glDeleteRenderbuffers(1, [self.depth_renderbuffer])


class ShaderEffect:
    """Base class for shader-based effects"""
    def __init__(self, viewport: ShaderViewport):
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


class RainEffect(ShaderEffect):
    """GPU-based rain effect using instanced rendering - adapted from multi_viewport.py"""
    def __init__(self, viewport: ShaderViewport, num_raindrops: int = 100):
        super().__init__(viewport)
        self.num_raindrops = num_raindrops
        self.raindrops = []
        self.instance_VBO = None
        
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec2 offset;
        layout(location = 2) in vec2 size;
        layout(location = 3) in vec4 color;

        out vec4 fragColor;
        uniform vec2 resolution;

        void main() {
            vec2 pos = position * size + offset;
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            gl_Position = vec4(clipPos, 0.0, 1.0);
            fragColor = color;
        }
        """
        
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec4 fragColor;
        out vec4 outColor;

        void main() {
            outColor = fragColor;
        }
        """
    
    def compile_shader(self):
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            vert = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            frag = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            shader = shaders.compileProgram(vert, frag)
            
            # Set resolution uniform
            glUseProgram(shader)
            loc = glGetUniformLocation(shader, "resolution")
            if loc != -1:
                glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
            glUseProgram(0)
            
            return shader
        except Exception as e:
            print(f"Shader compilation error: {e}")
            raise
    
    def setup_buffers(self):
        self.raindrops = [Raindrop(self.viewport.width, self.viewport.height) 
                         for _ in range(self.num_raindrops)]
        
        # Quad vertices
        vertices = np.array([
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Vertex buffer
        vertex_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        self.VBOs.append(vertex_VBO)
        
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Element buffer
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Instance buffer
        self.instance_VBO = glGenBuffers(1)
        self.VBOs.append(self.instance_VBO)
        
        glBindVertexArray(0)
    
    def update(self, dt: float, state: Dict):
        if not self.enabled:
            return
        for drop in self.raindrops:
            drop.update(dt)
    
    def render(self, state: Dict):
        if not self.enabled or not self.shader:
            return
            
        glUseProgram(self.shader)
        
        # Update resolution uniform
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        
        # Prepare instance data
        instance_data = []
        for drop in self.raindrops:
            instance_data.extend([
                drop.x, drop.y,           # offset
                drop.width, drop.length,  # size
                0.3, 0.7, 1.0, drop.alpha # color (light blue)
            ])
        
        if not instance_data:
            return
            
        instance_array = np.array(instance_data, dtype=np.float32)
        
        # Upload instance data
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, instance_array.nbytes, instance_array, GL_DYNAMIC_DRAW)
        
        glBindVertexArray(self.VAO)
        
        # Setup instance attributes
        stride = 8 * 4  # 8 floats * 4 bytes
        
        # Offset
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)
        
        # Size
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        
        # Color
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        
        # Draw
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, self.num_raindrops)
        
        glBindVertexArray(0)
        glUseProgram(0)


# Raindrop class - copied directly from multi_viewport.py
class Raindrop:
    def __init__(self, surface_width, surface_height):
        self.surface_width = surface_width
        self.surface_height = surface_height
        self.reset(True)
        
    def reset(self, randomize_y=False):
        import random
        self.x = random.uniform(0, self.surface_width)
        self.y = random.uniform(0, self.surface_height) if randomize_y else -10
        self.speed = random.uniform(100, 300)
        self.width = random.uniform(1.0, 2.0)
        self.length = random.uniform(10, 20)
        self.alpha = random.uniform(0.5, 0.8)
        
    def update(self, dt):
        self.y += self.speed * dt
        if self.y > self.surface_height + 10:
            self.reset()