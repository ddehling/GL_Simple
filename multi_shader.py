import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL import shaders
import numpy as np
import random
import sys
import ctypes

# Screen parameters
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Initialize pygame
pygame.init()
display = (SCREEN_WIDTH, SCREEN_HEIGHT)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
pygame.display.set_caption("OpenGL Visual Effects")

# Enable alpha blending
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

class ShaderEffect:
    """Base class for all shader effects"""
    def __init__(self, name, enabled=True):
        self.name = name
        self.enabled = enabled
        self.shader = None
        self.VAO = None
        self.VBOs = []
        self.EBO = None
        
    def init(self):
        """Initialize shader and buffers"""
        self.shader = self.compile_shader()
        self.setup_buffers()
        
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
        if self.VAO:
            glDeleteVertexArrays(1, [self.VAO])
        if self.VBOs:
            glDeleteBuffers(len(self.VBOs), self.VBOs)
        if self.EBO:
            glDeleteBuffers(1, [self.EBO])


class RainEffect(ShaderEffect):
    def __init__(self, num_raindrops=100, enabled=True):
        super().__init__("Rain", enabled)
        self.num_raindrops = num_raindrops
        self.raindrops = []
        self.instance_VBO = None
        
        # Shader source code
        self.vertex_shader_src = """
        #version 330 core
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec2 offset;
        layout(location = 2) in vec2 size;     // width, height
        layout(location = 3) in vec4 hsva;     // hsva

        out vec4 fragHsva;

        void main() {
            // Transform vertex by raindrop properties
            vec2 pos = position * size + offset;
            
            // Convert to clip space (-1 to 1)
            vec2 clipPos;
            clipPos.x = (pos.x / float(""" + str(SCREEN_WIDTH) + """)) * 2.0 - 1.0;
            clipPos.y = 1.0 - (pos.y / float(""" + str(SCREEN_HEIGHT) + """)) * 2.0;
            
            gl_Position = vec4(clipPos, 0.0, 1.0);
            fragHsva = hsva;
        }
        """

        self.fragment_shader_src = """
        #version 330 core
        in vec4 fragHsva;
        out vec4 outColor;

        // HSV to RGB conversion function
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }

        void main() {
            // Convert HSV to RGB
            vec3 rgb = hsv2rgb(vec3(fragHsva.x, fragHsva.y, fragHsva.z));
            outColor = vec4(rgb, fragHsva.a);
        }
        """
    
    def compile_shader(self):
        vert = shaders.compileShader(self.vertex_shader_src, GL_VERTEX_SHADER)
        frag = shaders.compileShader(self.fragment_shader_src, GL_FRAGMENT_SHADER)
        return shaders.compileProgram(vert, frag)
    
    def setup_buffers(self):
        # Initialize raindrops
        self.raindrops = [Raindrop() for _ in range(self.num_raindrops)]
        
        # Define raindrop shape (unit square)
        vertices = np.array([
            0.0, 0.0,  # top left
            1.0, 0.0,  # top right
            1.0, 1.0,  # bottom right
            0.0, 1.0   # bottom left
        ], dtype=np.float32)
        
        # Create indices for drawing quads
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # Create and bind VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Create and bind VBO for vertices
        vertex_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        self.VBOs.append(vertex_VBO)
        
        # Set up vertex attributes
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Create and bind EBO
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Create and bind instance VBO
        self.instance_VBO = glGenBuffers(1)
        self.VBOs.append(self.instance_VBO)
        
        glBindVertexArray(0)
    
    def update(self, dt):
        if not self.enabled:
            return
            
        # Update raindrops
        for drop in self.raindrops:
            drop.update(dt)
    
    def render(self):
        if not self.enabled:
            return
            
        # Use shader
        glUseProgram(self.shader)
        
        # Draw all raindrops and their trails
        all_instances = []
        
        # First add trails (so they're drawn first)
        for drop in self.raindrops:
            # Add trail segments with decreasing alpha
            for i, (x, y, alpha) in enumerate(drop.trail_points):
                # Calculate alpha fade based on position in trail
                fade = i / len(drop.trail_points) if drop.trail_points else 0
                trail_alpha = alpha * 0.5 * fade
                
                # Add trail segment (smaller than the raindrop)
                trail_width = drop.width * 0.8
                trail_length = drop.length * 0.5
                all_instances.extend([
                    x, y,                                # offset
                    trail_width, trail_length,           # size
                    drop.color[0], drop.color[1], drop.color[2], trail_alpha  # hsva with reduced alpha
                ])
        
        # Then add raindrops themselves
        for drop in self.raindrops:
            all_instances.extend([
                drop.x, drop.y,                          # offset
                drop.width, drop.length,                 # size
                drop.color[0], drop.color[1], drop.color[2], drop.color[3]  # hsva
            ])
        
        # Convert to numpy array
        instance_data = np.array(all_instances, dtype=np.float32)
        
        # Upload instance data
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        
        # Configure instance attributes
        glBindVertexArray(self.VAO)
        stride = 8 * 4  # 8 floats per instance * 4 bytes per float
        
        # Offset attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)  # 1 = advance once per instance
        
        # Size attribute
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        
        # Color attribute
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        
        # Draw all instances
        num_instances = len(instance_data) // 8
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, num_instances)
        
        glBindVertexArray(0)


class PlanetarySystemEffect(ShaderEffect):
    def __init__(self, num_planets=10, enabled=True):
        super().__init__("Planetary System", enabled)
        self.num_planets = num_planets
        self.planets = []
        self.instance_VBO = None
        self.time = 0.0  # Time counter for animation
        
        # Shader source code
        self.vertex_shader_src = """
        #version 330 core
        layout(location = 0) in vec2 position;        // Base shape coordinates (quad)
        layout(location = 1) in vec4 orbitData;       // orbital radius, inclination, size, speed
        layout(location = 2) in vec4 hsva;            // HSVA color
        layout(location = 3) in float phase;          // Initial orbital phase
        layout(location = 4) in float longOffset;     // Longitude offset for wrapping
        
        uniform float time;  // Time for animation
        
        out vec2 fragCoord;
        out vec4 fragHsva;
        out float fragAngle;
        out float fragLatitude;
        out float fragLongFactor;
        
        // Convert 3D position to longitude/latitude view
        vec2 sphericalToView(vec3 pos) {
            // Calculate longitude (angle in x-z plane)
            float longitude = degrees(atan(pos.z, pos.x));
            
            // Ensure longitude is in [0, 360] range
            longitude = mod(longitude + 360.0, 360.0);
            
            // Apply longitude offset for wrapping
            longitude = longitude + longOffset;
            
            // Calculate latitude from y coordinate
            float latitude = degrees(asin(pos.y / length(pos)));
            
            // Map longitude to x-coordinate in clip space [-1, 1]
            float clipX = (longitude / 180.0) - 1.0;
            
            // Map latitude to y-coordinate in clip space [-1, 1]
            float clipY = (latitude / 37.5) - 1.0;
            
            return vec2(clipX, clipY);
        }
        
        void main() {
            // Extract planet data
            float radius = orbitData.x;        // Orbital radius
            float baseInclination = orbitData.y;  // Base orbital inclination
            float size = orbitData.z;          // Planet size
            float speed = orbitData.w;         // Orbital speed
            
            // Calculate current angle in orbit
            float angle = mod(phase + time * speed*0.1, 360.0);
            float angleRad = radians(angle);
            
            // Calculate inclination with sinusoidal variation
            float inclination = baseInclination + 30.0 * sin(angleRad);
            float inclinationRad = radians(inclination);
            
            // Calculate 3D position using spherical coordinates
            vec3 planetPos;
            planetPos.x = radius * cos(angleRad) * cos(inclinationRad);
            planetPos.y = radius * sin(inclinationRad);
            planetPos.z = radius * sin(angleRad) * cos(inclinationRad);
            
            // Convert to view coordinates
            vec2 viewPos = sphericalToView(planetPos);
            
            // Calculate distance for size adjustment
            float distance = length(vec3(0.0, 0.0, 10.0) - planetPos);
            float adjustedSize = size * (10.0 / distance);
            
            // Calculate latitude for aspect ratio correction
            float latitude = degrees(asin(planetPos.y / length(planetPos)));
            
            // Calculate longitude correction factor based on latitude
            // cos(latitude) gives the appropriate scaling factor
            float longFactor = cos(radians(latitude));
            
            // Apply stronger correction - use a power function to enhance the effect
            // This makes the correction more pronounced at higher latitudes
            longFactor = pow(longFactor, 0.7);
            
            // Limit minimum scaling to prevent extreme stretching
            longFactor = max(longFactor, 0.15);
            
            // Center the quad
            vec2 vertexPos = position - vec2(0.5, 0.5);
            
            // Apply different scaling in x and y directions
            // For x: divide by longFactor to make planets wider at high latitudes
            vertexPos.x = vertexPos.x * adjustedSize / longFactor;
            vertexPos.y = vertexPos.y * adjustedSize;
            
            // Add to view position to place the planet
            vertexPos = vertexPos + viewPos;
            
            gl_Position = vec4(vertexPos, 0.0, 1.0);
            
            // Pass data to fragment shader
            fragCoord = position - vec2(0.5, 0.5);  // Center-relative coordinates
            fragHsva = hsva;
            fragAngle = angle;
            fragLatitude = latitude;
            fragLongFactor = longFactor;  // Pass longitude factor directly
        }
        """

        self.fragment_shader_src = """
        #version 330 core
        in vec2 fragCoord;
        in vec4 fragHsva;
        in float fragAngle;
        in float fragLatitude;
        in float fragLongFactor;
        
        out vec4 outColor;
        
        // HSV to RGB conversion function
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        
        void main() {
            // Original coordinates (centered at 0,0 with range [-0.5,0.5])
            vec2 centeredCoord = fragCoord;
            
            // Calculate an elliptical shape to account for the stretching in the vertex shader
            // Instead of scaling coordinates, we use an elliptical distance function
            float ellipticalDist = length(vec2(centeredCoord.x , centeredCoord.y));
            
            // Create hard boundary at 0.5 (edge of the normalized unit circle)
            // This ensures we stay within the quad regardless of stretching
            if (ellipticalDist > 0.5) {
                discard; // Discard fragments outside the circle
            }
            
            // Create a soft edge near the boundary
            float alpha = 1.0 - smoothstep(0.4, 0.5, ellipticalDist);
            
            // Convert HSV to RGB
            vec3 rgb = hsv2rgb(vec3(fragHsva.x, fragHsva.y, fragHsva.z));
            
            // Output final color with alpha
            outColor = vec4(rgb, fragHsva.a * alpha);
        }
        """
    
    def compile_shader(self):
        vert = shaders.compileShader(self.vertex_shader_src, GL_VERTEX_SHADER)
        frag = shaders.compileShader(self.fragment_shader_src, GL_FRAGMENT_SHADER)
        return shaders.compileProgram(vert, frag)
    
    def setup_buffers(self):
        # Initialize planets
        self.planets = [Planet() for _ in range(self.num_planets)]
        
        # Define planet shape (unit square)
        vertices = np.array([
            0.0, 0.0,  # top left
            1.0, 0.0,  # top right
            1.0, 1.0,  # bottom right
            0.0, 1.0   # bottom left
        ], dtype=np.float32)
        
        # Create indices for drawing quads
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # Create and bind VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Create and bind VBO for vertices
        vertex_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        self.VBOs.append(vertex_VBO)
        
        # Set up vertex attributes
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Create and bind EBO
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Create instance VBO
        self.instance_VBO = glGenBuffers(1)
        self.VBOs.append(self.instance_VBO)
        
        glBindVertexArray(0)
    
    def update(self, dt):
        if not self.enabled:
            return
            
        # Update animation time
        self.time += dt
    
    def render(self):
        if not self.enabled:
            return
            
        # Use shader
        glUseProgram(self.shader)
        
        # Set time uniform
        time_loc = glGetUniformLocation(self.shader, "time")
        glUniform1f(time_loc, self.time)
        
        # Prepare instance data with wrapping
        instance_data = []
        
        for planet in self.planets:
            # Add normal instance (with no longitude offset)
            instance_data.extend([
                planet.orbital_radius, planet.inclination, planet.size, planet.speed,
                planet.color[0], planet.color[1], planet.color[2], planet.color[3],
                planet.phase,
                0.0  # No longitude offset
            ])
            
            # Add copy that wraps to the left (appears on the right edge)
            instance_data.extend([
                planet.orbital_radius, planet.inclination, planet.size, planet.speed,
                planet.color[0], planet.color[1], planet.color[2], planet.color[3],
                planet.phase,
                -360.0  # Negative longitude offset makes it appear on the right
            ])
            
            # Add copy that wraps to the right (appears on the left edge)
            instance_data.extend([
                planet.orbital_radius, planet.inclination, planet.size, planet.speed,
                planet.color[0], planet.color[1], planet.color[2], planet.color[3],
                planet.phase,
                360.0  # Positive longitude offset makes it appear on the left
            ])
        
        # Convert to numpy array
        instance_data = np.array(instance_data, dtype=np.float32)
        
        # Upload instance data
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        
        # Configure instance attributes
        glBindVertexArray(self.VAO)
        stride = 10 * 4  # 10 floats per instance * 4 bytes per float
        
        # Orbit data attribute
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)
        
        # Color attribute
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        
        # Phase attribute
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(32))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        
        # Longitude offset attribute
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(36))
        glEnableVertexAttribArray(4)
        glVertexAttribDivisor(4, 1)
        
        # Draw all instances
        num_instances = len(instance_data) // 10
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, num_instances)
        
        glBindVertexArray(0)
        


class Planet:
    def __init__(self):
        # Orbital parameters
        self.orbital_radius = random.uniform(0.5, 3.0)
        self.inclination = random.uniform(10, 60)  # Base inclination (middle of visible range)
        self.size = random.uniform(0.03, 0.15)    # Relative size
        self.speed = random.uniform(10, 60)       # Degrees per second
        self.phase = random.uniform(0, 360)       # Initial position in orbit (degrees)
        
        # Color in HSV
        self.color = (
            random.uniform(0, 1),                # Hue - random color
            random.uniform(0.7, 1.0),            # Saturation - fairly saturated
            random.uniform(0.6, 1.0),            # Value - bright
            1.0                                  # Alpha - fully opaque
        )

class FireflyEffect(ShaderEffect):
    def __init__(self, num_fireflies=50, enabled=True):
        super().__init__("Fireflies", enabled)
        self.num_fireflies = num_fireflies
        self.fireflies = []
        self.instance_VBO = None
        
        # Shader source code
        self.vertex_shader_src = """
        #version 330 core
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec2 center;
        layout(location = 2) in float size;
        layout(location = 3) in vec4 hsva;

        out vec2 fragCoord;
        out vec4 fragHsva;

        void main() {
            // Scale and position the firefly
            vec2 pos = position * size + center;
            
            // Convert to clip space (-1 to 1)
            vec2 clipPos;
            clipPos.x = (pos.x / float(""" + str(SCREEN_WIDTH) + """)) * 2.0 - 1.0;
            clipPos.y = 1.0 - (pos.y / float(""" + str(SCREEN_HEIGHT) + """)) * 2.0;
            
            gl_Position = vec4(clipPos, 0.0, 1.0);
            
            // Pass to fragment shader
            fragCoord = position - vec2(0.5, 0.5);  // Center-relative coordinates
            fragHsva = hsva;
        }
        """

        self.fragment_shader_src = """
        #version 330 core
        in vec2 fragCoord;
        in vec4 fragHsva;
        out vec4 outColor;

        // HSV to RGB conversion function
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }

        void main() {
            // Calculate distance from center
            float dist = length(fragCoord);
            
            // Create a soft, glowing circle
            float intensity = 1.0 - smoothstep(0.0, 0.5, dist);
            
            // Convert HSV to RGB
            vec3 rgb = hsv2rgb(vec3(fragHsva.x, fragHsva.y, fragHsva.z));
            
            // Apply glow effect
            outColor = vec4(rgb, fragHsva.a * intensity);
        }
        """
    
    def compile_shader(self):
        vert = shaders.compileShader(self.vertex_shader_src, GL_VERTEX_SHADER)
        frag = shaders.compileShader(self.fragment_shader_src, GL_FRAGMENT_SHADER)
        return shaders.compileProgram(vert, frag)
    
    def setup_buffers(self):
        # Initialize fireflies
        self.fireflies = [Firefly() for _ in range(self.num_fireflies)]
        
        # Define firefly shape (unit square)
        vertices = np.array([
            0.0, 0.0,  # top left
            1.0, 0.0,  # top right
            1.0, 1.0,  # bottom right
            0.0, 1.0   # bottom left
        ], dtype=np.float32)
        
        # Create indices for drawing quads
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # Create and bind VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Create and bind VBO for vertices
        vertex_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        self.VBOs.append(vertex_VBO)
        
        # Set up vertex attributes
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Create and bind EBO
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Create instance VBO
        self.instance_VBO = glGenBuffers(1)
        self.VBOs.append(self.instance_VBO)
        
        glBindVertexArray(0)
    
    def update(self, dt):
        if not self.enabled:
            return
            
        # Update fireflies
        for fly in self.fireflies:
            fly.update(dt)
    
    def render(self):
        if not self.enabled:
            return
            
        # Use shader
        glUseProgram(self.shader)
        
        # Prepare instance data
        instance_data = []
        for fly in self.fireflies:
            instance_data.extend([
                fly.x, fly.y,                          # center
                fly.size,                              # size
                fly.color[0], fly.color[1], fly.color[2], fly.color[3]  # hsva
            ])
        
        # Convert to numpy array
        instance_data = np.array(instance_data, dtype=np.float32)
        
        # Upload instance data
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        
        # Configure instance attributes
        glBindVertexArray(self.VAO)
        stride = 7 * 4  # 7 floats per instance * 4 bytes per float
        
        # Center attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)
        
        # Size attribute
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        
        # Color attribute
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        
        # Draw all instances
        num_instances = len(instance_data) // 7
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, num_instances)
        
        glBindVertexArray(0)


class StarfieldEffect(ShaderEffect):
    def __init__(self, num_stars=200, enabled=True):
        super().__init__("Stars", enabled)
        self.num_stars = num_stars
        self.stars = []
        self.instance_VBO = None
        
        # Shader source code - similar to fireflies but with twinkling
        self.vertex_shader_src = """
        #version 330 core
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec2 center;
        layout(location = 2) in float size;
        layout(location = 3) in vec4 hsva;
        layout(location = 4) in float twinkle;

        out vec2 fragCoord;
        out vec4 fragHsva;
        out float fragTwinkle;

        void main() {
            // Scale and position the star
            vec2 pos = position * size + center;
            
            // Convert to clip space (-1 to 1)
            vec2 clipPos;
            clipPos.x = (pos.x / float(""" + str(SCREEN_WIDTH) + """)) * 2.0 - 1.0;
            clipPos.y = 1.0 - (pos.y / float(""" + str(SCREEN_HEIGHT) + """)) * 2.0;
            
            gl_Position = vec4(clipPos, 0.0, 1.0);
            
            // Pass to fragment shader
            fragCoord = position - vec2(0.5, 0.5);  // Center-relative coordinates
            fragHsva = hsva;
            fragTwinkle = twinkle;
        }
        """

        self.fragment_shader_src = """
        #version 330 core
        in vec2 fragCoord;
        in vec4 fragHsva;
        in float fragTwinkle;
        out vec4 outColor;

        // HSV to RGB conversion function
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }

        void main() {
            // Calculate distance from center
            float dist = length(fragCoord);
            
            // Create a soft, glowing circle with twinkle effect
            float intensity = 1.0 - smoothstep(0.0, 0.5, dist);
            intensity *= fragTwinkle;  // Apply twinkle factor
            
            // Convert HSV to RGB
            vec3 rgb = hsv2rgb(vec3(fragHsva.x, fragHsva.y, fragHsva.z));
            
            // Apply glow effect
            outColor = vec4(rgb, fragHsva.a * intensity);
        }
        """
    
    def compile_shader(self):
        vert = shaders.compileShader(self.vertex_shader_src, GL_VERTEX_SHADER)
        frag = shaders.compileShader(self.fragment_shader_src, GL_FRAGMENT_SHADER)
        return shaders.compileProgram(vert, frag)
    
    def setup_buffers(self):
        # Initialize stars
        self.stars = [Star() for _ in range(self.num_stars)]
        
        # Define star shape (unit square)
        vertices = np.array([
            0.0, 0.0,  # top left
            1.0, 0.0,  # top right
            1.0, 1.0,  # bottom right
            0.0, 1.0   # bottom left
        ], dtype=np.float32)
        
        # Create indices for drawing quads
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # Create and bind VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Create and bind VBO for vertices
        vertex_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        self.VBOs.append(vertex_VBO)
        
        # Set up vertex attributes
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Create and bind EBO
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Create instance VBO
        self.instance_VBO = glGenBuffers(1)
        self.VBOs.append(self.instance_VBO)
        
        glBindVertexArray(0)
    
    def update(self, dt):
        if not self.enabled:
            return
            
        # Update stars (twinkling)
        for star in self.stars:
            star.update(dt)
    
    def render(self):
        if not self.enabled:
            return
            
        # Use shader
        glUseProgram(self.shader)
        
        # Prepare instance data
        instance_data = []
        for star in self.stars:
            instance_data.extend([
                star.x, star.y,                          # center
                star.size,                               # size
                star.color[0], star.color[1], star.color[2], star.color[3],  # hsva
                star.twinkle                             # twinkle factor
            ])
        
        # Convert to numpy array
        instance_data = np.array(instance_data, dtype=np.float32)
        
        # Upload instance data
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        
        # Configure instance attributes
        glBindVertexArray(self.VAO)
        stride = 8 * 4  # 8 floats per instance * 4 bytes per float
        
        # Center attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)
        
        # Size attribute
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        
        # Color attribute
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        
        # Twinkle attribute
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(28))
        glEnableVertexAttribArray(4)
        glVertexAttribDivisor(4, 1)
        
        # Draw all instances
        num_instances = len(instance_data) // 8
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, num_instances)
        
        glBindVertexArray(0)


# Effect data classes
class Raindrop:
    def __init__(self):
        self.reset(True)
        self.trail_points = []
        self.max_trail_length = random.randint(5, 15)
    
    def reset(self, randomize_y=False):
        self.x = random.uniform(0, SCREEN_WIDTH)
        if randomize_y:
            self.y = random.uniform(0, SCREEN_HEIGHT)
        else:
            self.y = -10  # Start above the screen
        self.speed = random.uniform(200, 500)
        self.width = random.uniform(1.5, 3)
        self.length = random.uniform(15, 30)
        self.alpha = random.uniform(0.7, 0.9)
        # Light blue color in HSV (h, s, v, alpha)
        self.color = (0.6, 0.3, 1.0, self.alpha)  # H=0.6 is blue, S=0.3 for slight desaturation, V=1.0 for brightness
        
    def update(self, dt):
        # Save current position for trail
        self.trail_points.append((self.x, self.y, self.alpha))
        if len(self.trail_points) > self.max_trail_length:
            self.trail_points.pop(0)
        
        # Move raindrop
        self.y += self.speed * dt
        
        # Reset when off screen
        if self.y > SCREEN_HEIGHT + 10:
            self.reset()


class Firefly:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.x = random.uniform(0, SCREEN_WIDTH)
        self.y = random.uniform(0, SCREEN_HEIGHT)
        self.size = random.uniform(5, 12)
        self.speed_x = random.uniform(-30, 30)
        self.speed_y = random.uniform(-30, 30)
        self.pulse_speed = random.uniform(1, 3)
        self.pulse_phase = random.uniform(0, 2 * np.pi)
        # Yellowish color in HSV (h, s, v, alpha)
        self.base_alpha = random.uniform(0.5, 0.9)
        self.color = (0.15, 0.8, 1.0, self.base_alpha)  # H=0.15 is yellowish, S=0.8 for saturation, V=1.0 for brightness
        
    def update(self, dt):
        # Move firefly
        self.x += self.speed_x * dt
        self.y += self.speed_y * dt
        
        # Pulse brightness (in HSV, we modify the V component)
        pulse = (np.sin(self.pulse_phase) + 1) * 0.5  # 0-1 range
        self.pulse_phase += self.pulse_speed * dt
        self.color = (self.color[0], self.color[1], 0.5 + 0.5 * pulse, 
                      self.base_alpha)
        
        # Change direction occasionally
        if random.random() < 0.01:
            self.speed_x = random.uniform(-30, 30)
            self.speed_y = random.uniform(-30, 30)
        
        # Wrap around screen
        if self.x < -20:
            self.x = SCREEN_WIDTH + 20
        elif self.x > SCREEN_WIDTH + 20:
            self.x = -20
        if self.y < -20:
            self.y = SCREEN_HEIGHT + 20
        elif self.y > SCREEN_HEIGHT + 20:
            self.y = -20


class Star:
    def __init__(self):
        self.x = random.uniform(0, SCREEN_WIDTH)
        self.y = random.uniform(0, SCREEN_HEIGHT)
        self.size = random.uniform(1, 4)
        self.twinkle_speed = random.uniform(1, 5)
        self.twinkle_phase = random.uniform(0, 2 * np.pi)
        self.twinkle = 1.0
        
        # Slight color variations in HSV (h, s, v, alpha)
        hue = random.uniform(0.0, 1.0)  # Random hue for color variety
        self.color = (
            hue,
            random.uniform(0.0, 0.2),  # Low saturation for white-ish stars
            1.0,  # Full brightness
            random.uniform(0.7, 1.0)   # Alpha
        )
        
    def update(self, dt):
        # Update twinkle effect
        self.twinkle_phase += self.twinkle_speed * dt
        self.twinkle = 0.5 + 0.5 * np.sin(self.twinkle_phase)


class EffectManager:
    def __init__(self):
        self.effects = []
        self.fps = 0  # Current FPS
        self.fps_samples = []  # Store recent FPS samples
        self.fps_sample_count = 30  # Number of samples for averaging
        
    def add_effect(self, effect):
        effect.init()
        self.effects.append(effect)
        
        
    def update(self, dt):
        # Update FPS with rolling average
        if dt > 0:
            current_fps = 1.0 / dt
            self.fps_samples.append(current_fps)
            
            # Keep only the most recent samples
            if len(self.fps_samples) > self.fps_sample_count:
                self.fps_samples.pop(0)
                
            # Calculate average FPS
            self.fps = sum(self.fps_samples) / len(self.fps_samples)
            
        for effect in self.effects:
            effect.update(dt)
            
    def render(self):
        for effect in self.effects:
            effect.render()
            
    def toggle_effect(self, index):
        if 0 <= index < len(self.effects):
            self.effects[index].toggle()
            return True
        return False
    
    def cleanup(self):
        for effect in self.effects:
            effect.cleanup()

class WaveDistortionEffect(ShaderEffect):
    """Creates a wave distortion effect using vertex manipulation"""
    def __init__(self, enabled=True):
        super().__init__("Wave Distortion", enabled)
        self.time = 0.0
        
        self.vertex_shader_src = """
        #version 330 core
        layout(location = 0) in vec2 position;
        
        uniform float time;
        uniform vec2 resolution;
        
        out vec2 fragCoord;
        
        void main() {
            fragCoord = position;
            gl_Position = vec4(position, 06.0, 1.0);
        }
        """
        
        self.fragment_shader_src = """
        #version 330 core
        in vec2 fragCoord;
        out vec4 outColor;
        
        uniform float time;
        uniform vec2 resolution;
        
        void main() {
            vec2 uv = (fragCoord + 1.0) * 0.5;
            
            // Create wave distortion
            float wave1 = sin(uv.x * 10.0 + time * 2.0) * 0.1;
            float wave2 = cos(uv.y * 8.0 - time * 1.5) * 0.1;
            
            uv.x += wave1 * sin(uv.y * 15.0 + time);
            uv.y += wave2 * cos(uv.x * 12.0 - time);
            
            // Create color pattern
            vec3 color = vec3(0.0);
            color.r = sin(uv.x * 20.0 + time) * 0.5 + 0.5;
            color.g = sin(uv.y * 20.0 - time * 0.7) * 0.5 + 0.5;
            color.b = sin((uv.x + uv.y) * 10.0 + time * 1.3) * 0.5 + 0.5;
            
            // Add glow effect
            float glow = 1.0 - length(uv - 0.5) * 2.0;
            color *= glow;
            
            outColor = vec4(color, 0.7);
        }
        """
    
    def compile_shader(self):
        vert = shaders.compileShader(self.vertex_shader_src, GL_VERTEX_SHADER)
        frag = shaders.compileShader(self.fragment_shader_src, GL_FRAGMENT_SHADER)
        return shaders.compileProgram(vert, frag)
    
    def setup_buffers(self):
        # Create fullscreen quad
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        self.VBOs.append(VBO)
        
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        glBindVertexArray(0)
    
    def update(self, dt):
        if self.enabled:
            self.time += dt
    
    def render(self):
        if not self.enabled:
            return
            
        glUseProgram(self.shader)
        
        # Set uniforms
        glUniform1f(glGetUniformLocation(self.shader, "time"), self.time)
        glUniform2f(glGetUniformLocation(self.shader, "resolution"), 
                   SCREEN_WIDTH, SCREEN_HEIGHT)
        
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)


class FractalNoiseEffect(ShaderEffect):
    """Generates fractal noise patterns using multiple octaves"""
    def __init__(self, enabled=True):
        super().__init__("Fractal Noise", enabled)
        self.time = 0.0
        self.noise_scale = 5.0
        
        self.vertex_shader_src = """
        #version 330 core
        layout(location = 0) in vec2 position;
        out vec2 fragCoord;
        
        void main() {
            fragCoord = position;
            gl_Position = vec4(position, 0.0, 1.0);
        }
        """
        
        self.fragment_shader_src = """
        #version 330 core
        in vec2 fragCoord;
        out vec4 outColor;
        
        uniform float time;
        uniform float noiseScale;
        
        // Simple pseudo-random function
        float random(vec2 st) {
            return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
        }
        
        // 2D noise function
        float noise(vec2 st) {
            vec2 i = floor(st);
            vec2 f = fract(st);
            
            float a = random(i);
            float b = random(i + vec2(1.0, 0.0));
            float c = random(i + vec2(0.0, 1.0));
            float d = random(i + vec2(1.0, 1.0));
            
            vec2 u = f * f * (3.0 - 2.0 * f);
            
            return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + 
                   (d - b) * u.x * u.y;
        }
        
        // Fractal Brownian Motion
        float fbm(vec2 st, int octaves) {
            float value = 0.0;
            float amplitude = 0.5;
            float frequency = 1.0;
            
            for(int i = 0; i < octaves; i++) {
                value += amplitude * noise(st * frequency);
                frequency *= 2.0;
                amplitude *= 0.5;
            }
            
            return value;
        }
        
        void main() {
            vec2 uv = (fragCoord + 1.0) * 0.5;
            
            // Animate the noise
            vec2 st = uv * noiseScale + vec2(time * 0.1, time * 0.05);
            
            // Generate fractal noise with different octaves
            float n1 = fbm(st, 4);
            float n2 = fbm(st * 2.0 + vec2(100.0), 3);
            float n3 = fbm(st * 4.0 + vec2(200.0), 2);
            
            // Create color from noise
            vec3 color = vec3(0.0);
            color.r = n1 * 0.8 + 0.1;
            color.g = n2 * 0.6 + 0.2;
            color.b = n3 * 0.9 + 0.1;
            
            // Add some contrast
            color = pow(color, vec3(1.2));
            
            outColor = vec4(color, 0.6);
        }
        """
    
    def compile_shader(self):
        vert = shaders.compileShader(self.vertex_shader_src, GL_VERTEX_SHADER)
        frag = shaders.compileShader(self.fragment_shader_src, GL_FRAGMENT_SHADER)
        return shaders.compileProgram(vert, frag)
    
    def setup_buffers(self):
        # Fullscreen quad
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        self.VBOs.append(VBO)
        
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        glBindVertexArray(0)
    
    def update(self, dt):
        if self.enabled:
            self.time += dt
    
    def render(self):
        if not self.enabled:
            return
            
        glUseProgram(self.shader)
        
        glUniform1f(glGetUniformLocation(self.shader, "time"), self.time)
        glUniform1f(glGetUniformLocation(self.shader, "noiseScale"), self.noise_scale)
        
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)


class GeometryMorphEffect(ShaderEffect):
    """Demonstrates geometry morphing using vertex shader animation"""
    def __init__(self, num_shapes=20, enabled=True):
        super().__init__("Geometry Morph", enabled)
        self.num_shapes = num_shapes
        self.time = 0.0
        
        self.vertex_shader_src = """
        #version 330 core
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec3 shapeData;  // x, y, size
        layout(location = 2) in vec4 color;
        layout(location = 3) in float phase;
        
        uniform float time;
        
        out vec4 fragColor;
        
        mat2 rotate2D(float angle) {
            float s = sin(angle);
            float c = cos(angle);
            return mat2(c, -s, s, c);
        }
        
        void main() {
            // Morph between circle and star shape
            float morphFactor = sin(time + phase) * 0.5 + 0.5;
            
            // Calculate angle for this vertex
            float angle = atan(position.y, position.x);
            float dist = length(position);
            
            // Star shape calculation
            float starPoints = 5.0;
            float starFactor = 0.5 + 0.5 * cos(starPoints * angle);
            
            // Interpolate between circle and star
            float finalDist = mix(dist, dist * starFactor, morphFactor);
            vec2 morphedPos = normalize(position) * finalDist;
            
            // Apply rotation
            morphedPos = rotate2D(time * 0.5 + phase) * morphedPos;
            
            // Scale and translate
            morphedPos = morphedPos * shapeData.z + shapeData.xy;
            
            // Convert to clip space
            vec2 clipPos;
            clipPos.x = (morphedPos.x / float(""" + str(SCREEN_WIDTH) + """)) * 2.0 - 1.0;
            clipPos.y = 1.0 - (morphedPos.y / float(""" + str(SCREEN_HEIGHT) + """)) * 2.0;
            
            gl_Position = vec4(clipPos, 0.0, 1.0);
            fragColor = color;
        }
        """
        
        self.fragment_shader_src = """
        #version 330 core
        in vec4 fragColor;
        out vec4 outColor;
        
        void main() {
            outColor = fragColor;
        }
        """
    
    def compile_shader(self):
        vert = shaders.compileShader(self.vertex_shader_src, GL_VERTEX_SHADER)
        frag = shaders.compileShader(self.fragment_shader_src, GL_FRAGMENT_SHADER)
        return shaders.compileProgram(vert, frag)
    
    def setup_buffers(self):
        # Create circle vertices (will be morphed to star)
        num_segments = 32
        vertices = []
        for i in range(num_segments):
            angle = 2.0 * np.pi * i / num_segments
            vertices.extend([np.cos(angle), np.sin(angle)])
        vertices = np.array(vertices, dtype=np.float32)
        
        # Create indices for triangle fan
        indices = []
        for i in range(num_segments - 1):
            indices.extend([0, i, i + 1])
        indices.extend([0, num_segments - 1, 0])
        indices = np.array(indices, dtype=np.uint32)
        
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        self.VBOs.append(VBO)
        
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Create instance buffer
        self.instance_VBO = glGenBuffers(1)
        self.VBOs.append(self.instance_VBO)
        
        # Generate random shape data
        self.shape_data = []
        for i in range(self.num_shapes):
            x = random.uniform(50, SCREEN_WIDTH - 50)
            y = random.uniform(50, SCREEN_HEIGHT - 50)
            size = random.uniform(20, 60)
            r = random.random()
            g = random.random()
            b = random.random()
            phase = random.uniform(0, 2 * np.pi)
            self.shape_data.extend([x, y, size, r, g, b, 0.8, phase])
        
        glBindVertexArray(0)
        self.num_indices = len(indices)
    
    def update(self, dt):
        if self.enabled:
            self.time += dt
    
    def render(self):
        if not self.enabled:
            return
            
        glUseProgram(self.shader)
        glUniform1f(glGetUniformLocation(self.shader, "time"), self.time)
        
        # Upload instance data
        instance_data = np.array(self.shape_data, dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        
        glBindVertexArray(self.VAO)
        
        # Setup instance attributes
        stride = 8 * 4
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)
        
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(28))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        
        glDrawElementsInstanced(GL_TRIANGLES, self.num_indices, GL_UNSIGNED_INT, 
                                None, self.num_shapes)
        glBindVertexArray(0)


def main():
    # Create effect manager
    manager = EffectManager()
    
    # Add effects
    manager.add_effect(StarfieldEffect(200))
    manager.add_effect(RainEffect(100))
    manager.add_effect(FireflyEffect(50))
    manager.add_effect(PlanetarySystemEffect(15))
    
    # Add new advanced effects
    manager.add_effect(WaveDistortionEffect())
    manager.add_effect(FractalNoiseEffect())
    manager.add_effect(GeometryMorphEffect(20))
    
    
    # Main loop
    clock = pygame.time.Clock()
    running = True
    
    # FPS display variables
    fps_update_time = 0
    fps_update_interval = 0.5  # Update FPS display every 0.5 seconds
    
    # Help text
    print("Controls:")
    print("  1-7: Toggle effects")
    print("  1: Stars")
    print("  2: Rain")
    print("  3: Fireflies")
    print("  4: Planetary System")
    print("  5: Wave Distortion")
    print("  6: Fractal Noise")
    print("  7: Geometry Morph")
    print("  Esc: Quit")
    
    # For debugging
    print("Starting main loop...")
    
    # Main game loop
    while running:
        try:
            dt = clock.tick(60) / 1000.0  # Delta time in seconds
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    # Toggle effects with number keys
                    elif pygame.K_1 <= event.key <= pygame.K_9:
                        effect_index = event.key - pygame.K_1
                        if manager.toggle_effect(effect_index):
                            effect_name = manager.effects[effect_index].name
                            status = "enabled" if manager.effects[effect_index].enabled else "disabled"
                            print(f"{effect_name} {status}")
            
            # Update effects
            manager.update(dt)
            
            # Update FPS display periodically (to avoid updating too frequently)
            fps_update_time += dt
            if fps_update_time >= fps_update_interval:
                fps_update_time = 0
                # Update window title with FPS
                pygame.display.set_caption(f"OpenGL Visual Effects - FPS: {manager.fps:.1f}")
                # Also print to console periodically
                print(f"FPS: {manager.fps:.1f}")
            
            # Clear screen
            glClearColor(0.0, 0.0, 0.0, 1.0)  # Pure black background
            glClear(GL_COLOR_BUFFER_BIT)
            
            # Render effects
            manager.render()
            
            # Swap buffers
            pygame.display.flip()
            
            # Capture framebuffer to numpy array
            frame_data = glReadPixels(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE)
            frame_array = np.frombuffer(frame_data, dtype=np.uint8).reshape(SCREEN_HEIGHT, SCREEN_WIDTH, 4)
            
            # OpenGL returns the image flipped vertically, so we need to flip it back
            frame_array = np.flipud(frame_array)
            
            # Now frame_array contains the rendered image as a numpy array
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            running = False
    
    print("Exiting main loop...")
    
    # Clean up
    try:
        manager.cleanup()
        pygame.quit()
    except Exception as e:
        print(f"Error during cleanup: {e}")
    
    print("Program terminated")
    sys.exit()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error starting program: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)