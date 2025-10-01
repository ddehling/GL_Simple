"""
Shader-based celestial body rendering with viewport coordinate mapping
Handles arbitrary viewport distortions through corner coordinate interpolation
"""
import numpy as np
import math
import time
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict, List, Tuple
from .base import ShaderEffect

# ============================================================================
# CelestialBody Class - Handles orbital mechanics
# ============================================================================

class CelestialBody:
    def __init__(self, 
                 size: float,           
                 roughness: float,      
                 orbital_speed: float,  
                 color_h: float,        
                 color_s: float,        
                 color_v: float,        
                 tilt: float = 0,      # INPUT: Angle from vertical in DEGREES
                 shift: float = 0,      # INPUT: Azimuthal angle in DEGREES
                 glow_factor: float = 0.2,         
                 corona_size: float = 1.5,         
                 name: str = "unnamed",             
                 distance: float = 1
                 ):
        self.size = size
        self.roughness = roughness
        # Convert orbital speed to radians per second
        self.orbital_speed = orbital_speed * (2 * math.pi) / 600
        self.color_h = color_h
        self.color_s = color_s
        self.color_v = color_v
        
        # Store original degree values for reference
        self.tilt_deg = tilt
        self.shift_deg = shift
        
        # Convert input degrees to radians for calculations
        self.tilt = math.radians(tilt)  # Convert to radians from horizontal
        self.shift = math.radians(shift)      # Convert to radians
        
        self.glow_factor = glow_factor
        self.corona_size = corona_size
        self.name = name
        self.distance = distance
        self.last_update = time.time()
        self.angle = np.random.random() * 2 * math.pi  # Internal angle in RADIANS random start

    def update(self, current_time: float, whomp: float = 0):
        """Update orbital position (works in RADIANS)"""
        speed = 0.25
        delta_time = current_time - self.last_update
        self.angle += self.orbital_speed * delta_time * speed
        self.angle %= 2 * math.pi  # Keep angle in [0, 2π]
        self.last_update = current_time

    def get_true_position(self) -> tuple:
        """Calculate position in horizontal coordinates
        Internal calculations in RADIANS, converts to DEGREES for output"""
        # Start with position in vertical orbital plane (XZ plane)
        orbit_y = math.sin(self.angle)   # Horizontal component
        orbit_z = math.cos(self.angle)   # Vertical component
        orbit_x = 0  # No horizontal offset

        # Apply tilt (rotation around Y axis)
        tilted_x = orbit_x * math.cos(self.tilt) - orbit_z * math.sin(self.tilt)
        tilted_y = orbit_y
        tilted_z = orbit_x * math.sin(self.tilt) + orbit_z * math.cos(self.tilt)

        # Apply shift (rotation around Z axis)
        final_x = tilted_x * math.cos(self.shift) - tilted_y * math.sin(self.shift)
        final_y = tilted_x * math.sin(self.shift) + tilted_y * math.cos(self.shift)
        final_z = tilted_z

        # Convert to spherical coordinates in DEGREES
        azimuth = math.degrees(math.atan2(final_x, final_y))
        # Normalize azimuth to [-180, 180]
        if azimuth < 0:
            azimuth += 360
        if azimuth > 180:
            azimuth -= 360
            
        # Elevation
        r_xy = math.sqrt(final_x**2 + final_y**2)
        elevation = math.degrees(math.atan2(final_z, r_xy))

        return (azimuth, elevation)


# ============================================================================
# Celestial Body Configurations
# ============================================================================

CELESTIAL_BODIES = [
    CelestialBody(
        size=8,
        roughness=0.3,
        orbital_speed=1,
        color_h=0.15,  # Yellowish
        color_s=0.25,
        color_v=0.6,
        tilt=-15,  # Vertical orbital plane
        shift=5,  # Rise in the east
        glow_factor=0.4,
        corona_size=2.0,
        name="moon",
        distance=1,
    ),
    CelestialBody(
        size=4.5,
        roughness=0.2,
        orbital_speed=0.7,
        color_h=0.0,  # Red
        color_s=0.9,
        color_v=0.7,
        tilt=30,  # 30° from vertical
        shift=5,  # Rise slightly north of east
        glow_factor=0.3,
        corona_size=1.5,
        name="red_planet",
        distance=1.5,
    ),
    CelestialBody(
        size=16,
        roughness=0.3,
        orbital_speed=0.30,
        color_h=0.65,  # Blueish
        color_s=0.9,
        color_v=0.7,
        tilt=35,  # 45° from vertical
        shift=-3,  # Rise slightly south of east
        glow_factor=1.3,
        corona_size=2.3,
        name="blue_planet",
        distance=5,
    ),
    CelestialBody(
        size=20,
        roughness=0.6,
        orbital_speed=-2,
        color_h=0.25,  # Yellow-green
        color_s=0.9,
        color_v=0.6,
        tilt=-40,  # 45° from vertical
        shift=3,  # Rise slightly south of east
        glow_factor=1.3,
        corona_size=1.3,
        name="yellow_planet",
        distance=6,
    ),
    CelestialBody(
        size=2.5,
        roughness=0.3,
        orbital_speed=-9,
        color_h=0.65,  # Blueish
        color_s=0.2,
        color_v=0.7,
        tilt=20,  # 45° from vertical
        shift=6,  # Rise slightly south of east
        glow_factor=0.5,
        corona_size=1.75,
        name="asteroid",
        distance=1.2,
    ),
    CelestialBody(
        size=6,
        roughness=0.5,
        orbital_speed=-0.50,
        color_h=0.35,  # Green
        color_s=1,
        color_v=0.5,
        tilt=-5,  # 45° from vertical
        shift=6,  # Rise slightly south of east
        glow_factor=1.3,
        corona_size=2.3,
        name="green_planet",
        distance=4,
    ),
    CelestialBody(
        size=12,
        roughness=0.25,
        orbital_speed=2.60,
        color_h=0.35,  # Green (but invisible)
        color_s=1,
        color_v=0.0,  # Zero brightness = invisible/ghost
        tilt=-10,  # 45° from vertical
        shift=-6,  # Rise slightly south of east
        glow_factor=1.3,
        corona_size=2.3,
        name="ghost_planet",
        distance=.4,
    )
]


# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_celestial_bodies(state, outstate, corners, **kwargs):
    """
    Shader-based celestial bodies compatible with EventScheduler
    
    Usage:
        corners = [
            (az_tl, el_tl),  # Top-left: azimuth, elevation in degrees
            (az_tr, el_tr),  # Top-right
            (az_br, el_br),  # Bottom-right
            (az_bl, el_bl)   # Bottom-left
        ]
        scheduler.schedule_event(0, 60, shader_celestial_bodies, 
                                corners=corners, frame_id=0)
    
    Args:
        state: Event state dict
        outstate: Global state dict (contains 'celestial_bodies' list)
        corners: List of 4 (azimuth, elevation) tuples defining viewport bounds
    """
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    
    if shader_renderer is None:
        print("WARNING: shader_renderer not found in state!")
        return
    
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        print(f"WARNING: viewport {frame_id} not found!")
        return
    
    # Get celestial bodies from global state
    celestial_bodies = outstate.get('celestial_bodies', [])
    
    # Initialize effect on first call
    if state['count'] == 0:
        print(f"Initializing celestial bodies for frame {frame_id}")
        
        try:
            effect = viewport.add_effect(
                CelestialBodiesEffect,
                corners=corners,
                celestial_bodies=celestial_bodies
            )
            state['celestial_effect'] = effect
            print(f"✓ Initialized shader celestial bodies for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize celestial bodies: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Bodies update externally, but we can update visibility here
    if 'celestial_effect' in state:
        state['celestial_effect'].celestial_visibility = outstate.get('celestial_visibility', 1.0)
    
    # Cleanup on close
    if state['count'] == -1:
        if 'celestial_effect' in state:
            print(f"Cleaning up celestial bodies for frame {frame_id}")
            viewport.effects.remove(state['celestial_effect'])
            state['celestial_effect'].cleanup()
            print(f"✓ Cleaned up shader celestial bodies for frame {frame_id}")


# ============================================================================
# Rendering Class
# ============================================================================

class CelestialBodiesEffect(ShaderEffect):
    """GPU-based celestial body rendering with viewport distortion correction"""
    
    def __init__(self, viewport, corners: List[Tuple[float, float]], 
                 celestial_bodies: List):
        super().__init__(viewport)
        
        # Store viewport corner coordinates (azimuth, elevation in degrees)
        self.corners = np.array(corners, dtype=np.float32)
        if self.corners.shape != (4, 2):
            raise ValueError("corners must be a list of 4 (azimuth, elevation) tuples")
        
        # Reference to external celestial bodies (not a copy!)
        self.celestial_bodies = celestial_bodies
        self.celestial_visibility = 1.0
        
        # Pre-compute coordinate grids for the viewport
        self._setup_coordinate_grid()
        
    def _setup_coordinate_grid(self):
        """Pre-compute angular coordinates for each pixel in viewport"""
        height, width = self.viewport.height, self.viewport.width
        
        # Create normalized coordinate grids (0 to 1)
        y_norm = np.linspace(0, 1, height)
        x_norm = np.linspace(0, 1, width)
        X_norm, Y_norm = np.meshgrid(x_norm, y_norm)
        
        # Bilinear interpolation of corner coordinates
        # corners: [TL, TR, BR, BL] = [(az, el), ...]
        tl, tr, br, bl = self.corners
        
        # Interpolate azimuth
        az_top = tl[0] * (1 - X_norm) + tr[0] * X_norm
        az_bottom = bl[0] * (1 - X_norm) + br[0] * X_norm
        self.azimuth_grid = az_top * (1 - Y_norm) + az_bottom * Y_norm
        
        # Interpolate elevation
        el_top = tl[1] * (1 - X_norm) + tr[1] * X_norm
        el_bottom = bl[1] * (1 - X_norm) + br[1] * X_norm
        self.elevation_grid = el_top * (1 - Y_norm) + el_bottom * Y_norm
        
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;  // Quad vertex position
        layout(location = 1) in vec3 instance_data;  // x, y, z (screen pos + depth)
        layout(location = 2) in vec4 color_size;  // r, g, b, radius
        layout(location = 3) in vec2 body_params;  // corona_size, glow_factor
        
        out vec4 fragColor;
        out vec2 fragPos;
        out float fragRadius;
        out float fragCoronaSize;
        out float fragGlowFactor;
        
        uniform vec2 resolution;
        
        void main() {
            // Scale quad by radius (for rendering size)
            vec2 scaled = position * color_size.a;
            
            // Translate to body center
            vec2 screen_pos = scaled + instance_data.xy;
            
            // Convert to clip space
            vec2 clipPos = (screen_pos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
            // Use Z for depth
            float depth = instance_data.z / 100.0;
            
            gl_Position = vec4(clipPos, depth, 1.0);
            
            // Pass data to fragment shader
            fragColor = vec4(color_size.rgb, 1.0);
            // IMPORTANT: Pass scaled position so distance calculation works
            fragPos = scaled;  // Now in pixel space
            fragRadius = color_size.a;
            fragCoronaSize = body_params.x;
            fragGlowFactor = body_params.y;
        }
        """
        
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec4 fragColor;
        in vec2 fragPos;
        in float fragRadius;
        in float fragCoronaSize;
        in float fragGlowFactor;
        
        out vec4 outColor;
        
        void main() {
            // Distance from center of celestial body (now in pixel space)
            float dist = length(fragPos);
            
            // Core body radius (the solid part)
            float core_radius = fragRadius / fragCoronaSize;
            
            // Core body with anti-aliasing
            float core_alpha = smoothstep(core_radius + 0.5, core_radius - 0.5, dist);
            
            // Corona glow (extends to full radius)
            float corona_alpha = smoothstep(fragRadius + 0.5, fragRadius - 0.5, dist);
            corona_alpha = corona_alpha * fragGlowFactor * (1.0 - core_alpha);
            
            // Combine core and corona
            float total_alpha = core_alpha + corona_alpha;
            
            // Reduce saturation in corona
            vec3 final_color = fragColor.rgb;
            if (core_alpha < 0.5) {
                // In corona region, desaturate
                float gray = dot(final_color, vec3(0.299, 0.587, 0.114));
                final_color = mix(vec3(gray), final_color, 0.6);
            }
            
            // Apply brightness falloff in corona for more realistic glow
            if (dist > core_radius) {
                float corona_falloff = 1.0 - smoothstep(core_radius, fragRadius, dist);
                final_color = final_color * (0.5 + 0.5 * corona_falloff);
            }
            
            outColor = vec4(final_color, total_alpha);
            
            // Discard fully transparent pixels
            if (total_alpha < 0.01) discard;
        }
        """
    
    def compile_shader(self):
        """Compile and link celestial body shaders"""
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
        """Initialize OpenGL buffers"""
        # Quad vertices (centered, will be scaled per body)
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # Create VAO
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
        
        # Instance buffer (will be updated each frame)
        self.instance_VBO = glGenBuffers(1)
        self.VBOs.append(self.instance_VBO)
        
        glBindVertexArray(0)

    def _get_screen_position(self, azimuth: float, elevation: float) -> Tuple[float, float, bool]:
        """
        Convert angular coordinates to screen position
        Returns (x, y, is_visible)
        """
        # Find closest pixel in the grid
        # This is a simple nearest-neighbor approach; could be optimized
        az_diff = np.abs(self.azimuth_grid - azimuth)
        el_diff = np.abs(self.elevation_grid - elevation)
        
        # Handle azimuth wraparound (e.g., 179° and -179° are close)
        az_diff = np.minimum(az_diff, 360 - az_diff)
        
        # Combined distance
        dist = np.sqrt(az_diff**2 + el_diff**2)
        
        # Find minimum distance
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        min_dist = dist[min_idx]
        
        # If too far from any grid point, not visible
        if min_dist > 5.0:  # 5 degree tolerance
            return (0, 0, False)
        
        # Convert array indices to screen coordinates
        y, x = min_idx
        
        return (float(x), float(y), True)

    def update(self, dt: float, state: Dict):
        """Update is handled externally by CelestialBody.update()"""
        pass

    def render(self, state: Dict):
        """Render all visible celestial bodies"""
        if not self.enabled or not self.shader:
            return
        
        # Get visibility from state
        visibility = self.celestial_visibility
        if visibility <= 0.01:
            return
        
        # Collect visible bodies
        visible_bodies = []
        
        for body in self.celestial_bodies:
            # Get angular position
            pos = body.get_true_position()
            if pos is None:
                continue
            
            azimuth, elevation = pos
            
            # Convert to screen coordinates
            screen_x, screen_y, is_visible = self._get_screen_position(azimuth, elevation)
            
            if not is_visible:
                continue
            
            # Calculate apparent size in pixels (base size scaled by corona)
            pixel_radius = body.size * body.corona_size
            
            # Calculate depth (closer bodies render on top)
            # Use inverse of distance so closer = larger depth value
            depth = 100.0 / max(body.distance, 0.1)
            
            # Convert HSV color to RGB
            h, s, v = body.color_h, body.color_s, body.color_v
            # Simple HSV to RGB conversion
            c = v * s
            x = c * (1 - abs((h * 6) % 2 - 1))
            m = v - c
            
            if h < 1/6:
                r, g, b = c, x, 0
            elif h < 2/6:
                r, g, b = x, c, 0
            elif h < 3/6:
                r, g, b = 0, c, x
            elif h < 4/6:
                r, g, b = 0, x, c
            elif h < 5/6:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            r, g, b = r + m, g + m, b + m
            
            # Apply visibility and roughness
            brightness = visibility * (1.0 + np.random.random() * body.roughness)
            
            visible_bodies.append({
                'position': (screen_x, screen_y, depth),
                'color': (r * brightness, g * brightness, b * brightness),
                'radius': pixel_radius,
                'corona_size': body.corona_size,
                'glow_factor': body.glow_factor
            })
        
        if not visible_bodies:
            return
        
        # Sort by depth (far to near) for proper rendering
        visible_bodies.sort(key=lambda b: b['position'][2])
        
        # Build instance data
        instance_data = []
        for body_data in visible_bodies:
            pos = body_data['position']
            col = body_data['color']
            instance_data.extend([
                pos[0], pos[1], pos[2],  # position + depth
                col[0], col[1], col[2], body_data['radius'],  # color + radius
                body_data['corona_size'], body_data['glow_factor']  # body params
            ])
        
        instance_data = np.array(instance_data, dtype=np.float32)
        
        # Upload to GPU
        glUseProgram(self.shader)
        
        # Update resolution uniform
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_VBO)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        
        glBindVertexArray(self.VAO)
        
        # Setup instance attributes
        stride = 9 * 4  # 9 floats per instance
        
        # Position (location 1) - vec3
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)
        
        # Color + Radius (location 2) - vec4
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        
        # Body params (location 3) - vec2
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(28))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Draw all bodies with instancing
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, len(visible_bodies))
        
        glDisable(GL_BLEND)
        glBindVertexArray(0)
        glUseProgram(0)