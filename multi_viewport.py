import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
import numpy as np
import random
import sys
import time
import ctypes
import platform

# Default dimensions
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 600

# Detect platform
IS_RASPBERRY_PI = platform.machine() in ['aarch64', 'armv7l', 'armv8']
IS_WINDOWS = sys.platform == 'win32'


class ViewportRenderer:
    """Single window with multiple rendering viewports"""
    def __init__(self, window_width=1200, window_height=800):
        self.window_width = window_width
        self.window_height = window_height
        self.window = None
        self.viewports = []
        self.running = True
        self.fps = 0
        self.frame_count = 0
        self.fps_update_time = 0
        
        self.init_glfw()
        self.create_window()
        
    def init_glfw(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        if IS_RASPBERRY_PI:
            # Raspberry Pi: Use OpenGL ES with EGL
            print("Configuring for Raspberry Pi (OpenGL ES 3.1 + EGL)")
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
            glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
        else:
            # Windows/Desktop: Use desktop OpenGL
            print("Configuring for Desktop (OpenGL 3.3 Core)")
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
    def create_window(self):
        self.window = glfw.create_window(self.window_width, self.window_height, 
                                       "Multi-Viewport Renderer", None, None)
        if not self.window:
            raise RuntimeError("Failed to create window")
            
        glfw.make_context_current(self.window)
        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_window_close_callback(self.window, self._close_callback)
        
        # OpenGL setup
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_SCISSOR_TEST)
        
        print(f"Created main window: {self.window_width}x{self.window_height}")
        
    def add_viewport(self, x, y, width, height, title="Viewport"):
        """Add a new rendering viewport"""
        viewport = Viewport(x, y, width, height, title, len(self.viewports) + 1)
        self.viewports.append(viewport)
        
        print(f"Added viewport: {title} at ({x}, {y}) size {width}x{height}")
        return viewport
        
    def _key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            # Toggle effects with number keys for each viewport
            elif glfw.KEY_1 <= key <= glfw.KEY_9:
                viewport_index = key - glfw.KEY_1
                if viewport_index < len(self.viewports):
                    viewport = self.viewports[viewport_index]
                    if viewport.effects:
                        viewport.effects[0].toggle()  # Toggle first effect
                        print(f"Toggled {viewport.effects[0].name} in {viewport.title}")
            # Function keys to toggle viewports by effect type
            elif key == glfw.KEY_F1:  # Toggle all rain effects
                for viewport in self.viewports:
                    for effect in viewport.effects:
                        if "Rain" in effect.name:
                            effect.toggle()
                            print(f"Toggled {effect.name} in {viewport.title}")
            elif key == glfw.KEY_F2:  # Toggle all starfield effects
                for viewport in self.viewports:
                    for effect in viewport.effects:
                        if "Starfield" in effect.name:
                            effect.toggle()
                            print(f"Toggled {effect.name} in {viewport.title}")
                            
    def _close_callback(self, window):
        self.running = False
        
    def run(self):
        last_time = glfw.get_time()
        
        print("Starting render loop...")
        print("\nControls:")
        print("  1-9: Toggle effects in viewports 1-9")
        print("  F1: Toggle all rain effects")
        print("  F2: Toggle all starfield effects")
        print("  ESC: Exit")
        print()
        
        # Initialize all viewport effects
        for viewport in self.viewports:
            viewport.init_effects()
        
        while not glfw.window_should_close(self.window) and self.running:
            current_time = glfw.get_time()
            dt = current_time - last_time
            last_time = current_time
            
            glfw.poll_events()
            
            # Clear entire window with dark gray background
            glViewport(0, 0, self.window_width, self.window_height)
            glScissor(0, 0, self.window_width, self.window_height)
            glClearColor(0.1, 0.1, 0.1, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            
            # Render each viewport
            for viewport in self.viewports:
                self.render_viewport(viewport, dt)
            
            # Draw viewport borders and labels
            self.draw_viewport_info()
            
            glfw.swap_buffers(self.window)
            
            # Calculate FPS
            self.frame_count += 1
            if current_time - self.fps_update_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.fps_update_time)
                self.frame_count = 0
                self.fps_update_time = current_time
                glfw.set_window_title(self.window, f"Multi-Viewport Renderer - FPS: {self.fps:.1f}")
            
            time.sleep(0.016)  # ~60 FPS cap
            
        self.cleanup()
        
    def render_viewport(self, viewport, dt):
        """Render a single viewport"""
        # Set viewport and scissor for this region
        glViewport(viewport.x, viewport.y, viewport.width, viewport.height)
        glScissor(viewport.x, viewport.y, viewport.width, viewport.height)
        
        # Clear viewport area with black background
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        
        # Update and render viewport effects
        viewport.update(dt)
        viewport.render()
        
    def draw_viewport_info(self):
        """Draw borders and info for viewports (simple colored borders)"""
        # Reset viewport to full window for border drawing
        glViewport(0, 0, self.window_width, self.window_height)
        glScissor(0, 0, self.window_width, self.window_height)
        
        # This is a simple approach - you could implement text rendering here
        # For now, we'll just draw colored borders using line primitives
        # (This would require a simple line shader, but we'll skip for simplicity)
        
    def cleanup(self):
        print("Cleaning up viewports...")
        for viewport in self.viewports:
            viewport.cleanup()
        self.viewports.clear()
        
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()
        print("Cleanup complete")


class Viewport:
    """A rendering viewport within the main window"""
    def __init__(self, x, y, width, height, title, viewport_id):
        self.x = x
        self.y = y  
        self.width = width
        self.height = height
        self.title = title
        self.viewport_id = viewport_id
        self.effects = []
        self.initialized = False
        
    def add_effect(self, effect_class, *args, **kwargs):
        kwargs['surface_width'] = self.width
        kwargs['surface_height'] = self.height
        effect = effect_class(*args, **kwargs)
        self.effects.append(effect)
        print(f"Added {effect.name} to {self.title}")
        return effect
        
    def init_effects(self):
        """Initialize all effects for this viewport"""
        if self.initialized:
            return
            
        # Set viewport for initialization
        glViewport(self.x, self.y, self.width, self.height)
        glScissor(self.x, self.y, self.width, self.height)
        
        for effect in self.effects:
            try:
                effect.init()
                print(f"Initialized {effect.name} for {self.title}")
            except Exception as e:
                print(f"Failed to initialize {effect.name} for {self.title}: {e}")
                effect.enabled = False
                
        self.initialized = True
        
    def update(self, dt):
        for effect in self.effects:
            if effect.enabled:
                effect.update(dt)
                
    def render(self):
        for effect in self.effects:
            if effect.enabled:
                effect.render()
                
    def get_frame_buffer(self):
        """Get frame buffer content for this viewport"""
        # Set viewport first
        glViewport(self.x, self.y, self.width, self.height)
        pixels = glReadPixels(self.x, self.y, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE)
        frame = np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 4)
        return np.flip(frame, axis=0)  # Flip Y axis
        
    def cleanup(self):
        for effect in self.effects:
            effect.cleanup()


# Base ShaderEffect class
class ShaderEffect:
    """Base class for all shader effects"""
    def __init__(self, name, enabled=True, surface_width=DEFAULT_WIDTH, surface_height=DEFAULT_HEIGHT):
        self.name = name
        self.enabled = enabled
        self.surface_width = surface_width
        self.surface_height = surface_height
        self.shader = None
        self.VAO = None
        self.VBOs = []
        self.EBO = None
        
    def init(self):
        """Initialize shader and buffers"""
        try:
            self.shader = self.compile_shader()
            self.setup_buffers()
            print(f"  ✓ {self.name} shader compiled successfully")
        except Exception as e:
            print(f"  ✗ Error initializing {self.name}: {e}")
            self.enabled = False
            raise
        
    def compile_shader(self):
        """Compile and link shaders - override in subclasses"""
        pass
        
    def setup_buffers(self):
        """Set up VAO, VBO, etc. - override in subclasses"""
        pass
        
    def update(self, dt):
        """Update animation state - override in subclasses"""
        pass
        
    def render(self):
        """Render the effect - override in subclasses"""
        if not self.enabled or not self.shader:
            return
            
    def toggle(self):
        """Toggle effect on/off"""
        self.enabled = not self.enabled
        
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
    def __init__(self, num_raindrops=100, enabled=True, surface_width=DEFAULT_WIDTH, surface_height=DEFAULT_HEIGHT):
        super().__init__("Rain", enabled, surface_width, surface_height)
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
                glUniform2f(loc, self.surface_width, self.surface_height)
            glUseProgram(0)
            
            return shader
        except Exception as e:
            print(f"Shader compilation error in {self.name}: {e}")
            raise
    
    def setup_buffers(self):
        self.raindrops = [Raindrop(self.surface_width, self.surface_height) 
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
    
    def update(self, dt):
        if not self.enabled:
            return
        for drop in self.raindrops:
            drop.update(dt)
    
    def render(self):
        if not self.enabled or not self.shader:
            return
            
        glUseProgram(self.shader)
        
        # Update resolution uniform
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, self.surface_width, self.surface_height)
        
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


class StarfieldEffect(ShaderEffect):
    def __init__(self, num_stars=200, enabled=True, surface_width=DEFAULT_WIDTH, surface_height=DEFAULT_HEIGHT):
        super().__init__("Starfield", enabled, surface_width, surface_height)
        self.num_stars = num_stars
        self.stars = []
        self.instance_VBO = None
        
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec2 center;
        layout(location = 2) in float size;
        layout(location = 3) in vec4 color;

        out vec4 fragColor;
        out vec2 fragCoord;
        uniform vec2 resolution;

        void main() {
            vec2 pos = position * size + center;
            vec2 clipPos = (pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            gl_Position = vec4(clipPos, 0.0, 1.0);
            fragColor = color;
            fragCoord = position - vec2(0.5);
        }
        """
        
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec4 fragColor;
        in vec2 fragCoord;
        out vec4 outColor;

        void main() {
            float dist = length(fragCoord);
            float intensity = 1.0 - smoothstep(0.0, 0.5, dist);
            outColor = vec4(fragColor.rgb, fragColor.a * intensity);
        }
        """
    
    def compile_shader(self):
        try:
            vert = shaders.compileShader(self.get_vertex_shader(), GL_VERTEX_SHADER)
            frag = shaders.compileShader(self.get_fragment_shader(), GL_FRAGMENT_SHADER)
            shader = shaders.compileProgram(vert, frag)
            
            glUseProgram(shader)
            loc = glGetUniformLocation(shader, "resolution")
            if loc != -1:
                glUniform2f(loc, self.surface_width, self.surface_height)
            glUseProgram(0)
            
            return shader
        except Exception as e:
            print(f"Shader compilation error in {self.name}: {e}")
            raise
    
    def setup_buffers(self):
        self.stars = [Star(self.surface_width, self.surface_height) 
                     for _ in range(self.num_stars)]
        
        vertices = np.array([
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        vertex_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        self.VBOs.append(vertex_VBO)
        
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        self.instance_VBO = glGenBuffers(1)
        self.VBOs.append(self.instance_VBO)
        
        glBindVertexArray(0)
    
    def update(self, dt):
        if not self.enabled:
            return
        for star in self.stars:
            star.update(dt)
    
    def render(self):
        if not self.enabled or not self.shader:
            return
            
        glUseProgram(self.shader)
        
        # Update resolution uniform
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, self.surface_width, self.surface_height)
        
        instance_data = []
        for star in self.stars:
            instance_data.extend([
                star.x, star.y,    # center
                star.size,         # size
                1.0, 1.0, 0.8, star.twinkle * star.alpha  # color (warm white)
            ])
        
        instance_array = np.array(instance_data, dtype=np.float32)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, instance_array.nbytes, instance_array, GL_DYNAMIC_DRAW)
        
        glBindVertexArray(self.VAO)
        stride = 7 * 4
        
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)
        
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, self.num_stars)
        
        glBindVertexArray(0)
        glUseProgram(0)


# Data classes
class Raindrop:
    def __init__(self, surface_width, surface_height):
        self.surface_width = surface_width
        self.surface_height = surface_height
        self.reset(True)
        
    def reset(self, randomize_y=False):
        self.x = random.uniform(0, self.surface_width)
        self.y = random.uniform(0, self.surface_height) if randomize_y else -10
        self.speed = random.uniform(100, 300)  # Slower for smaller viewports
        self.width = random.uniform(1.0, 2.0)
        self.length = random.uniform(10, 20)
        self.alpha = random.uniform(0.5, 0.8)
        
    def update(self, dt):
        self.y += self.speed * dt
        if self.y > self.surface_height + 10:
            self.reset()


class Star:
    def __init__(self, surface_width, surface_height):
        self.x = random.uniform(0, surface_width)
        self.y = random.uniform(0, surface_height)
        self.size = random.uniform(2, 6)
        self.twinkle_speed = random.uniform(1, 3)
        self.twinkle_phase = random.uniform(0, 2 * np.pi)
        self.twinkle = 1.0
        self.alpha = random.uniform(0.6, 1.0)
        
    def update(self, dt):
        self.twinkle_phase += self.twinkle_speed * dt
        self.twinkle = 0.4 + 0.6 * abs(np.sin(self.twinkle_phase))


def main():
    """Main function with viewport-based rendering"""
    
    try:
        print("Creating multi-viewport renderer...")
        
        # Create main renderer window
        renderer = ViewportRenderer(1200, 900)
        
        print("\nCreating viewports...")
        
        # Create viewports (positioned like separate windows)
        # Top row
        vp1 = renderer.add_viewport(10, 10, 380, 280, "Light Rain")
        vp1.add_effect(RainEffect, num_raindrops=30)
        
        vp2 = renderer.add_viewport(410, 10, 380, 280, "Heavy Rain")
        vp2.add_effect(RainEffect, num_raindrops=100)
        
        vp3 = renderer.add_viewport(810, 10, 380, 280, "Starfield")
        vp3.add_effect(StarfieldEffect, num_stars=150)
        
        # Bottom row
        vp4 = renderer.add_viewport(10, 310, 580, 280, "Mixed Effects")
        vp4.add_effect(StarfieldEffect, num_stars=80)
        vp4.add_effect(RainEffect, num_raindrops=40)
        
        vp5 = renderer.add_viewport(610, 310, 580, 280, "Dense Stars")
        vp5.add_effect(StarfieldEffect, num_stars=250)
        
        # Bottom center
        vp6 = renderer.add_viewport(210, 610, 780, 280, "Storm")
        vp6.add_effect(RainEffect, num_raindrops=200)
        vp6.add_effect(StarfieldEffect, num_stars=50)
        
        print(f"\nCreated {len(renderer.viewports)} viewports successfully")
        print("Starting application...")
        
        # Run the renderer
        renderer.run()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'renderer' in locals():
            renderer.cleanup()
        
    print("Application closed.")


if __name__ == "__main__":
    main()