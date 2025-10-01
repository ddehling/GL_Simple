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
        speed = 2.5
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
        
        # Store corners for inverse mapping
        self.tl, self.tr, self.br, self.bl = self.corners
        self.width = width
        self.height = height
        
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

    def _inverse_bilinear_map(self, azimuth: float, elevation: float) -> Tuple[float, float, bool]:
        """
        Inverse bilinear interpolation to find screen position from angular coordinates
        Uses iterative solver (Newton-Raphson)
        Returns (x, y, converged)
        """
        # Unpack corners
    # Unpack corners
        az_tl, el_tl = self.tl
        az_tr, el_tr = self.tr
        az_br, el_br = self.br
        az_bl, el_bl = self.bl
        
        # SPECIAL CASE: Check if viewport spans full azimuth range (360° wraparound)
        az_range = max(az_tl, az_tr, az_br, az_bl) - min(az_tl, az_tr, az_br, az_bl)
        if az_range > 350:  # Full panoramic view
            # For panoramic views, azimuth maps linearly across width
            # Normalize azimuth to [0, 360]
            az_normalized = azimuth % 360
            if az_normalized < 0:
                az_normalized += 360
            
            # Map to u coordinate [0, 1]
            # Assume left edge is at -180, right edge at +180
            u = (az_normalized + 180) / 360.0
            
            # Elevation still uses standard interpolation
            # Linear interpolation for v based on elevation
            el_top = (el_tl + el_tr) / 2  # Average top elevation
            el_bottom = (el_bl + el_br) / 2  # Average bottom elevation
            
            if abs(el_top - el_bottom) < 0.01:  # Degenerate case
                v = 0.5
            else:
                v = (elevation - el_top) / (el_bottom - el_top)
            
            # Convert to pixels
            x = u * self.width
            y = v * self.height
            
            # Check if within viewport bounds
            converged = (0 <= x <= self.width) and (0 <= y <= self.height)
            return (x, y, converged)
        
        # STANDARD CASE: Normal bilinear interpolation for limited FOV
        # Handle azimuth wraparound - normalize target and corners to same range
        def normalize_azimuth(az, reference):
            """Normalize azimuth to be close to reference (handle wraparound)"""
            while az - reference > 180:
                az -= 360
            while az - reference < -180:
                az += 360
            return az
        
        # Normalize all azimuths relative to target
        az_tl = normalize_azimuth(az_tl, azimuth)
        az_tr = normalize_azimuth(az_tr, azimuth)
        az_br = normalize_azimuth(az_br, azimuth)
        az_bl = normalize_azimuth(az_bl, azimuth)
        
        # Initial guess (center of viewport)
        u, v = 0.5, 0.5
        
        # Newton-Raphson iteration
        error = float('inf')  # Initialize error
        for iteration in range(20):  # Max iterations
            # Current position estimate
            az_est = ((1-u)*(1-v)*az_tl + u*(1-v)*az_tr + 
                    u*v*az_br + (1-u)*v*az_bl)
            el_est = ((1-u)*(1-v)*el_tl + u*(1-v)*el_tr + 
                    u*v*el_br + (1-u)*v*el_bl)
            
            # Error
            error_az = azimuth - az_est
            error_el = elevation - el_est
            error = np.sqrt(error_az**2 + error_el**2)
            
            if error < 0.01:  # Converged
                break
            
            # Compute Jacobian
            daz_du = (-(1-v)*az_tl + (1-v)*az_tr + v*az_br - v*az_bl)
            daz_dv = (-(1-u)*az_tl - u*az_tr + u*az_br + (1-u)*az_bl)
            del_du = (-(1-v)*el_tl + (1-v)*el_tr + v*el_br - v*el_bl)
            del_dv = (-(1-u)*el_tl - u*el_tr + u*el_br + (1-u)*el_bl)
            
            # Jacobian matrix determinant
            det = daz_du * del_dv - daz_dv * del_du
            
            if abs(det) < 1e-10:  # Singular, can't solve
                return (self.width * u, self.height * v, False)
            
            # Newton-Raphson update
            du = (del_dv * error_az - daz_dv * error_el) / det
            dv = (-del_du * error_az + daz_du * error_el) / det
            
            u += du
            v += dv
            
            # Don't clamp u,v - allow extrapolation for off-screen bodies
        
        # Convert normalized coordinates to pixels
        x = u * self.width
        y = v * self.height
        
        # More lenient convergence check for extrapolation
        # Accept solution if error is reasonable OR if we've done many iterations
        converged = error < 2.0 or iteration >= 15  # Relaxed threshold
        
        return (x, y, converged)


        
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 position;  // Quad vertex position
        layout(location = 1) in vec3 instance_data;  // x, y, z (screen pos + depth)
        layout(location = 2) in vec4 color_size;  // r, g, b, radius
        layout(location = 3) in vec3 body_params;  // corona_size, glow_factor, roughness
        
        out vec4 fragColor;
        out vec2 fragPos;
        out float fragRadius;
        out float fragCoronaSize;
        out float fragGlowFactor;
        out float fragRoughness;
        
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
            fragPos = scaled;  // In pixel space
            fragRadius = color_size.a;
            fragCoronaSize = body_params.x;
            fragGlowFactor = body_params.y;
            fragRoughness = body_params.z;
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
        in float fragRoughness;
        
        out vec4 outColor;
        
        // Simple noise function for surface texture
        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
        }
        
        void main() {
            // Distance from center of celestial body (now in pixel space)
            float dist = length(fragPos);
            
            // Core body radius (the solid part)
            float core_radius = fragRadius / fragCoronaSize;
            
            // Add per-pixel noise for surface texture (only in core)
            float noise = 0.0;
            if (dist < core_radius) {
                noise = (hash(fragPos * 0.1) - 0.5) * fragRoughness;
            }
            
            // Core body with anti-aliasing and noise
            float core_alpha = smoothstep(core_radius + 0.5, core_radius - 0.5, dist);
            
            // Corona glow (extends to full radius)
            float corona_alpha = smoothstep(fragRadius + 0.5, fragRadius - 0.5, dist);
            corona_alpha = corona_alpha * fragGlowFactor * (1.0 - core_alpha);
            
            // Combine core and corona
            float total_alpha = core_alpha + corona_alpha;
            
            // Apply noise to brightness (only in core)
            vec3 final_color = fragColor.rgb * (1.0 + noise * core_alpha);
            
            // Reduce saturation in corona
            if (core_alpha < 0.5) {
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


    def _get_screen_position(self, azimuth: float, elevation: float, apparent_radius: float = 0) -> Tuple[float, float, bool]:
        """
        Convert angular coordinates to screen position
        Returns (x, y, is_visible)
        """
        # Use inverse bilinear interpolation
        x, y, converged = self._inverse_bilinear_map(azimuth, elevation)
        
        # Expand check margin for bodies near viewport edges
        margin = apparent_radius + 10  # Extra pixels for smooth transitions
        
        # Check if any part of the body could be visible (with margin)
        x_min = x - margin
        x_max = x + margin
        y_min = y - margin
        y_max = y + margin
        
        # Check intersection with viewport [0, width] x [0, height]
        intersects_viewport = not (
            x_max < 0 or x_min > self.width or
            y_max < 0 or y_min > self.height
        )
        
        # Trust the bounding box check - if any part of the body could be visible,
        # render it even if the center coordinate mapping is approximate
        return (x, y, intersects_viewport)




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
            
            # Calculate apparent size in pixels (base size scaled by corona)
            pixel_radius = body.size * body.corona_size
            
            # Convert to screen coordinates WITH radius check
            screen_x, screen_y, is_visible = self._get_screen_position(
                azimuth, elevation, apparent_radius=pixel_radius
            )
            
            if not is_visible:
                continue
            # Calculate apparent size in pixels (base size scaled by corona)
            pixel_radius = body.size * body.corona_size
            
            # Calculate depth for celestial bodies (second furthest objects)
            # Map distance to z-range: 90-98 (with closer bodies slightly nearer)
            # Bodies with larger distance values are physically further, so lower z
            # Normalize to a reasonable range and invert so closer bodies have lower z
            z_base = 98.0  # Base depth for furthest bodies
            z_range = 8.0  # Range of depths (90-98)
            
            # Normalize distance (assuming distances range from ~0.4 to ~6)
            # Closer bodies (smaller distance) get slightly lower z values but still far
            distance_factor = np.clip(body.distance / 10.0, 0.0, 1.0)
            depth = z_base - (distance_factor * z_range)  # 90 to 98
            
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
            
            # Apply visibility (but NOT random roughness here!)
            brightness = visibility
            
            visible_bodies.append({
                'position': (screen_x, screen_y, depth),
                'color': (r * brightness, g * brightness, b * brightness),
                'radius': pixel_radius,
                'corona_size': body.corona_size,
                'glow_factor': body.glow_factor,
                'roughness': body.roughness  # Pass roughness to shader instead
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
                body_data['corona_size'], body_data['glow_factor'], body_data['roughness']  # body params
            ])
        
        instance_data = np.array(instance_data, dtype=np.float32)
        
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
        # Setup instance attributes
        stride = 10 * 4  # 10 floats per instance (was 9)
        
        # Position (location 1) - vec3
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)
        
        # Color + Radius (location 2) - vec4
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        
        # Body params (location 3) - vec3 (now includes roughness)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(28))
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