import glfw
from OpenGL.GL import *
import numpy as np
from typing import List, Tuple, Dict, Optional
import platform
from corefunctions.shader_effects.base import ShaderEffect
# Detect platform
IS_RASPBERRY_PI = platform.machine() in ['aarch64', 'armv7l', 'armv8']

class ShaderRenderer:
    """GPU-based renderer with visible OpenGL window and multiple viewports"""
    def __init__(self, frame_dimensions: List[Tuple[int, int]], padding=20, headless=False):
        self.frame_dimensions = frame_dimensions
        self.num_frames = len(frame_dimensions)
        self.padding = padding
        self.headless = headless
        self.window = None
        self.viewports = []
        self.ctx_initialized = False
        
        # Calculate window size based on viewport dimensions
        total_width = sum(w for w, h in frame_dimensions)
        max_height = max(h for w, h in frame_dimensions)
        
        self.window_width = total_width + padding * (self.num_frames + 1)
        self.window_height = max_height + padding * 2
        
        print(f"Calculated window size: {self.window_width}x{self.window_height} (native viewport size)")
        
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
        
        # No scaling - use native dimensions for display
        display_width = width
        display_height = height
        
        # Calculate x position (accumulate previous widths)
        x_offset = self.padding
        for i in range(frame_id):
            prev_width, _ = self.frame_dimensions[i]
            x_offset += prev_width + self.padding
        
        # Center vertically
        y_offset = (self.window_height - display_height) // 2
        
        if not self.headless:
            print(f"Creating viewport {frame_id}:")
            print(f"  Framebuffer (LED): {width}x{height}")
            print(f"  Display: {display_width}x{display_height} at ({x_offset}, {y_offset})")
            print(f"  Scale factor: 1.0 (native)")
        else:
            print(f"Creating viewport {frame_id}: {width}x{height} (headless)")
        
        viewport = ShaderViewport(frame_id, width, height, 
                                 x_offset, y_offset, 
                                 display_width, display_height,
                                 self.window, headless=self.headless)
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
        """Swap window buffers (skip in headless mode)"""
        if not self.headless:
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
        
    def sync_gpu(self):
        """Wait for all GPU operations to complete"""
        glfw.make_context_current(self.window)
        glFinish()


class ShaderViewport:
    """Individual viewport with shader effect pipeline and framebuffer for LED output"""
    def __init__(self, frame_id: int, width: int, height: int, 
                 window_x: int, window_y: int, 
                 display_width: int, display_height: int,
                 glfw_window, headless=False):
        self.frame_id = frame_id
        self.width = width  # Actual framebuffer size (for LED output)
        self.height = height
        self.window_x = window_x  # Position in window
        self.window_y = window_y
        self.display_width = display_width  # Display size in window (scaled)
        self.display_height = display_height
        self.glfw_window = glfw_window
        self.headless = headless  # Add headless flag
        self.effects = []
        
        # Framebuffer for LED output (separate from window rendering)
        self.fbo = None
        self.color_texture = None
        self.depth_texture = None  # Changed from depth_renderbuffer
        
        
    def init_glfw(self):
        """Initialize GLFW with OpenGL ES 3.1"""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        # Set visibility based on headless mode
        if self.headless:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            print("Configuring headless mode (window hidden)")
        
        if IS_RASPBERRY_PI:
            print("Configuring for Raspberry Pi (OpenGL ES 3.1 + EGL)")
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
            glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
        else:
            if not self.headless:
                print("Configuring for Desktop (OpenGL ES 3.1)")
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        
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
        
        # Create depth texture (changed from renderbuffer so effects can read it)
        self.depth_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.depth_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, 
                     self.width, self.height,
                     0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        # Create framebuffer
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 
                              GL_TEXTURE_2D, self.color_texture, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                              GL_TEXTURE_2D, self.depth_texture, 0)
        
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
        
        # Clear framebuffer (including depth!)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)
        glScissor(0, 0, self.width, self.height)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear depth too
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        # Clear window viewport region
        if not self.headless:
            glViewport(self.window_x, self.window_y, self.display_width, self.display_height)
            glScissor(self.window_x, self.window_y, self.display_width, self.display_height)
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear depth too

    
    def update(self, dt: float, state: Dict):
        """Update all effects"""
        for effect in self.effects:
            if effect.enabled:
                effect.update(dt, state)
    
    def render(self, state: Dict):
        """Render effects to framebuffer (and optionally to window)"""
        glfw.make_context_current(self.glfw_window)
        
        # Always render to framebuffer (for LED output at actual resolution)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)
        glScissor(0, 0, self.width, self.height)
        
        for effect in self.effects:
            if effect.enabled:
                effect.render(state)
        
        # CRITICAL: Ensure rendering is complete before unbinding
        glFlush()  # Submit all commands
        
        # Only render to window if not in headless mode
        if not self.headless:
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
        
        # No need for glFinish() here since it's called before reading frames
        pixels = glReadPixels(0, 0, self.width, self.height, 
                            GL_RGBA, GL_UNSIGNED_BYTE)
        frame = np.frombuffer(pixels, dtype=np.uint8).reshape(
            self.height, self.width, 4)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        # Flip Y axis and drop alpha
        frame = np.flip(frame, axis=0)
        return frame[:, :, :3]
    
    def cleanup(self):
        """Clean up resources"""
        glfw.make_context_current(self.glfw_window)
        for effect in self.effects:
            effect.cleanup()
        if self.fbo:
            glDeleteFramebuffers(1, [self.fbo])
        if self.color_texture:
            glDeleteTextures([self.color_texture])
        if self.depth_texture:
            glDeleteTextures([self.depth_texture])

