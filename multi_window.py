import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
import numpy as np
import random
import sys
import time
import ctypes

# Default window dimensions
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 600


class RenderWindow:
    """Independent GLFW window with its own OpenGL context"""
    _window_count = 0
    _windows = {}  # Will use id(window) as key
    
    def __init__(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, title=None, x=None, y=None):
        RenderWindow._window_count += 1
        self.window_id = RenderWindow._window_count
        self.width = width
        self.height = height
        self.title = title or f"Render Window {self.window_id}"
        self.x = x if x is not None else 50 + (self.window_id - 1) * 50
        self.y = y if y is not None else 50 + (self.window_id - 1) * 50
        
        self.window = None
        self.effects = []
        self.running = True
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0
        self.fps_update_time = 0
        
        self.create_window()
        
    def create_window(self):
        """Create GLFW window and OpenGL context"""
        # Create window
        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            raise RuntimeError(f"Failed to create window: {self.title}")
            
        # Position window
        glfw.set_window_pos(self.window, self.x, self.y)
        
        # Store window reference using window pointer as integer key
        window_ptr = id(self.window) if hasattr(self.window, '__hash__') else self.window.__int__()
        RenderWindow._windows[window_ptr] = self
        
        # Set user pointer for callbacks (alternative method)
        glfw.set_window_user_pointer(self.window, window_ptr)
        
        # Set callbacks
        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_window_close_callback(self.window, self._close_callback)
        
        # Make context current for initialization
        glfw.make_context_current(self.window)
        
        # Initialize OpenGL settings
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glViewport(0, 0, self.width, self.height)
        
        # Initialize effects
        for effect in self.effects:
            effect.init()
            
        # Release context so other windows can use it
        glfw.make_context_current(None)
        
    @staticmethod
    def _get_window_instance(window):
        """Get RenderWindow instance from GLFW window handle"""
        window_ptr = glfw.get_window_user_pointer(window)
        return RenderWindow._windows.get(window_ptr)
        
    @staticmethod
    def _key_callback(window, key, scancode, action, mods):
        """Handle key events for a window"""
        self = RenderWindow._get_window_instance(window)
        if not self:
            return
            
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            # Toggle effects with number keys
            elif glfw.KEY_1 <= key <= glfw.KEY_9:
                effect_index = key - glfw.KEY_1
                if effect_index < len(self.effects):
                    self.effects[effect_index].toggle()
                    print(f"Window {self.window_id}: Toggled {self.effects[effect_index].name}")
                    
    @staticmethod
    def _close_callback(window):
        """Handle window close event"""
        self = RenderWindow._get_window_instance(window)
        if self:
            self.running = False
            
    def add_effect(self, effect_class, *args, **kwargs):
        """Add an effect to this window"""
        kwargs['surface_width'] = self.width
        kwargs['surface_height'] = self.height
        effect = effect_class(*args, **kwargs)
        self.effects.append(effect)
        
        # Initialize if window exists
        if self.window:
            glfw.make_context_current(self.window)
            effect.init()
            glfw.make_context_current(None)
            
        return effect
        
    def remove_effect(self, effect):
        """Remove an effect from this window"""
        if effect in self.effects:
            glfw.make_context_current(self.window)
            effect.cleanup()
            glfw.make_context_current(None)
            self.effects.remove(effect)
            
    def update(self, dt):
        """Update all effects"""
        for effect in self.effects:
            effect.update(dt)
            
    def render(self):
        """Render all effects"""
        if not self.window or glfw.window_should_close(self.window):
            self.running = False
            return False
            
        # Make this window's context current
        glfw.make_context_current(self.window)
        
        # Clear screen
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Render effects
        for effect in self.effects:
            effect.render()
            
        # Swap buffers
        glfw.swap_buffers(self.window)
        
        # Calculate FPS
        self.frame_count += 1
        current_time = glfw.get_time()
        if current_time - self.fps_update_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.fps_update_time)
            self.frame_count = 0
            self.fps_update_time = current_time
            glfw.set_window_title(self.window, f"{self.title} - FPS: {self.fps:.1f}")
            
        return True
        
    def cleanup(self):
        """Clean up resources"""
        if self.window:
            glfw.make_context_current(self.window)
            for effect in self.effects:
                effect.cleanup()
                
            # Remove from registry
            window_ptr = glfw.get_window_user_pointer(self.window)
            if window_ptr in RenderWindow._windows:
                del RenderWindow._windows[window_ptr]
                
            glfw.destroy_window(self.window)
            self.window = None


class WindowManager:
    """Manages multiple GLFW windows"""
    def __init__(self):
        self.windows = []
        self.running = True
        
        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
            
        # Set OpenGL version hints (optional, for compatibility)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            
    def create_window(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, title=None, x=None, y=None):
        """Create a new render window"""
        window = RenderWindow(width, height, title, x, y)
        self.windows.append(window)
        return window
        
    def remove_window(self, window):
        """Remove a window"""
        if window in self.windows:
            window.cleanup()
            self.windows.remove(window)
            
    def run(self):
        """Main render loop for all windows"""
        last_time = glfw.get_time()
        
        while self.windows and self.running:
            # Poll events
            glfw.poll_events()
            
            # Calculate delta time
            current_time = glfw.get_time()
            dt = current_time - last_time
            last_time = current_time
            
            # Update and render each window
            windows_to_remove = []
            for window in self.windows:
                window.update(dt)
                if not window.render():
                    windows_to_remove.append(window)
                    
            # Remove closed windows
            for window in windows_to_remove:
                self.remove_window(window)
                print(f"Window closed: {window.title}")
                
        # Cleanup
        self.cleanup()
        
    def cleanup(self):
        """Clean up all windows and terminate GLFW"""
        for window in self.windows[:]:
            window.cleanup()
        self.windows.clear()
        glfw.terminate()


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
        except Exception as e:
            print(f"Error initializing {self.name}: {e}")
            self.enabled = False
        
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
        if not self.enabled:
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
        #version 330 core
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
        #version 330 core
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
        
        # Update resolution uniform in case window was resized
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, self.surface_width, self.surface_height)
        
        # Prepare instance data
        instance_data = []
        for drop in self.raindrops:
            instance_data.extend([
                drop.x, drop.y,           # offset
                drop.width, drop.length,  # size
                0.5, 0.7, 1.0, drop.alpha # color (light blue)
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
        #version 330 core
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
        #version 330 core
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
                1.0, 1.0, 1.0, star.twinkle * star.alpha  # color with twinkle
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
        self.speed = random.uniform(200, 500)
        self.width = random.uniform(1.5, 3)
        self.length = random.uniform(15, 30)
        self.alpha = random.uniform(0.6, 0.9)
        
    def update(self, dt):
        self.y += self.speed * dt
        if self.y > self.surface_height + 10:
            self.reset()


class Star:
    def __init__(self, surface_width, surface_height):
        self.x = random.uniform(0, surface_width)
        self.y = random.uniform(0, surface_height)
        self.size = random.uniform(2, 8)
        self.twinkle_speed = random.uniform(1, 5)
        self.twinkle_phase = random.uniform(0, 2 * np.pi)
        self.twinkle = 1.0
        self.alpha = random.uniform(0.5, 1.0)
        
    def update(self, dt):
        self.twinkle_phase += self.twinkle_speed * dt
        self.twinkle = 0.3 + 0.7 * abs(np.sin(self.twinkle_phase))


def main():
    """Main function with multiple independent windows"""
    
    try:
        # Create window manager
        manager = WindowManager()
        
        print("Creating multiple windows...")
        print("\nControls for each window:")
        print("  1-9: Toggle effects")
        print("  Esc: Close window")
        print("\n")
        
        # Create Window 1: Light rain
        window1 = manager.create_window(400, 300, "Light Rain", x=50, y=50)
        window1.add_effect(RainEffect, num_raindrops=30)
        print(f"Created: {window1.title}")
        
        # Create Window 2: Heavy rain
        window2 = manager.create_window(600, 450, "Heavy Rain", x=500, y=50)
        window2.add_effect(RainEffect, num_raindrops=150)
        print(f"Created: {window2.title}")
        
        # Create Window 3: Starfield
        window3 = manager.create_window(500, 500, "Starfield", x=50, y=400)
        window3.add_effect(StarfieldEffect, num_stars=200)
        print(f"Created: {window3.title}")
        
        # Create Window 4: Mixed effects
        window4 = manager.create_window(800, 300, "Mixed Effects", x=600, y=520)
        window4.add_effect(StarfieldEffect, num_stars=100)
        window4.add_effect(RainEffect, num_raindrops=50)
        print(f"Created: {window4.title}")
        
        print("\nAll windows created. Running...")
        print("Close all windows or press Ctrl+C to exit\n")
        
        # Run all windows
        manager.run()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'manager' in locals():
            manager.cleanup()
        
    print("All windows closed.")


if __name__ == "__main__":
    main()