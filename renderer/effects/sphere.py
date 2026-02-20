"""
Complete sphere effect - rendering + event integration
A sphere that moves slowly through 3D space
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
from .base import ShaderEffect

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_sphere(state, outstate, speed=1.0, radius=50.0):
    """
    Shader-based sphere effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_sphere, speed=1.0, radius=50.0, frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        speed: Movement speed multiplier
        radius: Sphere radius in pixels
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
    
    # Initialize sphere effect on first call
    if state['count'] == 0:
        print(f"Initializing sphere effect for frame {frame_id}")
        
        try:
            sphere_effect = viewport.add_effect(
                SphereEffect,
                speed=speed,
                radius=radius
            )
            state['sphere_effect'] = sphere_effect
            print(f"✓ Initialized shader sphere for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize sphere: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Update speed if it changes in global state
    if 'sphere_effect' in state:
        state['sphere_effect'].speed = outstate.get('sphere_speed', speed)
    
    # On close event, clean up
    if state['count'] == -1:
        if 'sphere_effect' in state:
            print(f"Cleaning up sphere effect for frame {frame_id}")
            viewport.effects.remove(state['sphere_effect'])
            state['sphere_effect'].cleanup()
            print(f"✓ Cleaned up shader sphere for frame {frame_id}")


# ============================================================================
# Rendering Classes
# ============================================================================

class SphereEffect(ShaderEffect):
    """GPU-based sphere effect with 3D movement and lighting"""
    
    def __init__(self, viewport, speed: float = 1.0, radius: float = 50.0):
        super().__init__(viewport)
        self.speed = speed
        self.radius = radius
        self.wrap_margin = self.radius + 10  # Wrapping margin (must exceed sphere radius)
        
        # Sphere position and movement
        self.position = np.array([
            viewport.width / 2,
            viewport.height / 2,
            0.0  # depth
        ], dtype=np.float32)
        
        # Movement path parameters
        self.time = 0.0
        self.path_scale = min(viewport.width, viewport.height) * 0.3
        
        # Sphere color
        self.color = np.array([0.2, 0.6, 1.0], dtype=np.float32)
        
        # Generate sphere mesh
        self._generate_sphere_mesh()
        
    def _generate_sphere_mesh(self, segments=32, rings=16):
        """Generate a UV sphere mesh"""
        vertices = []
        normals = []
        indices = []
        
        # Generate vertices and normals
        for ring in range(rings + 1):
            theta = ring * np.pi / rings
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            
            for seg in range(segments + 1):
                phi = seg * 2 * np.pi / segments
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)
                
                # Vertex position (unit sphere)
                x = sin_theta * cos_phi
                y = cos_theta
                z = sin_theta * sin_phi
                
                vertices.extend([x, y, z])
                normals.extend([x, y, z])  # For a sphere, normal = normalized position
        
        # Generate indices
        for ring in range(rings):
            for seg in range(segments):
                first = ring * (segments + 1) + seg
                second = first + segments + 1
                
                # First triangle
                indices.extend([first, second, first + 1])
                # Second triangle
                indices.extend([second, second + 1, first + 1])
        
        self.vertices = np.array(vertices, dtype=np.float32)
        self.normals = np.array(normals, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.uint32)
        self.num_indices = len(self.indices)
        

    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;
        
        uniform vec2 resolution;
        uniform vec3 spherePos;    // x, y, depth (0-100)
        uniform float sphereRadius;
        
        out vec3 fragNormal;
        out vec3 fragPos;
        out float fragDepth;
        
        void main() {
            // Scale by radius
            vec3 scaledPos = position * sphereRadius;
            
            // Rotate slightly for visual interest
            vec3 worldPos = scaledPos + spherePos;
            
            // Transform normal
            fragNormal = normalize(normal);
            fragPos = worldPos;
            
            // Convert to clip space (same method as rain/circles)
            vec2 clipPos = (worldPos.xy / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;
            
                        // Map sphere depth to 0-1 range (z=0 near, z=100 far)
            float depth = worldPos.z / 100.0;
            depth = clamp(depth, 0.0, 1.0);
            
            gl_Position = vec4(clipPos, depth, 1.0);
            fragDepth = depth;
        }
        """
        
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec3 fragNormal;
        in vec3 fragPos;
        in float fragDepth;
        
        uniform vec3 sphereColor;
        uniform vec3 lightPos;
        uniform vec2 resolution;
        
        out vec4 outColor;
        
        void main() {
            // Ambient
            vec3 ambient = 0.3 * sphereColor;
            
            // Diffuse
            vec3 norm = normalize(fragNormal);
            vec3 lightDir = normalize(lightPos - fragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * sphereColor;
            
            // Specular
            vec3 viewDir = normalize(-fragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
            vec3 specular = 0.5 * spec * vec3(1.0);
            
            // Combine
            vec3 result = ambient + diffuse + specular;
            
                        // Depth-based alpha (z=0 near/opaque, z=100 far/transparent)
            float alpha = 0.5 + 0.5 * (1.0 - fragDepth);
            
            outColor = vec4(result, alpha);
        }
        """
    
    def compile_shader(self):
        """Compile and link sphere shaders"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            vert = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            frag = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            shader = shaders.compileProgram(vert, frag)
            
            return shader
        except Exception as e:
            print(f"Shader compilation error: {e}")
            raise

    def setup_buffers(self):
        """Initialize OpenGL buffers for sphere rendering"""
        # Create VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        # Vertex buffer
        vertex_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        self.VBOs.append(vertex_VBO)
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Normal buffer
        normal_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, normal_VBO)
        glBufferData(GL_ARRAY_BUFFER, self.normals.nbytes, self.normals, GL_STATIC_DRAW)
        self.VBOs.append(normal_VBO)
        
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        
        # Element buffer
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        
        glBindVertexArray(0)

    def _create_projection_matrix(self):
        """Create a perspective projection matrix"""
        aspect = self.viewport.width / self.viewport.height
        fov = np.radians(45.0)
        near = 0.1
        far = 1000.0
        
        f = 1.0 / np.tan(fov / 2.0)
        
        projection = np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)
        
        return projection
    
    def _create_view_matrix(self):
        """Create a view matrix"""
        # Camera position (looking at the viewport)
        camera_pos = np.array([
            self.viewport.width / 2,
            self.viewport.height / 2,
            -300.0
        ])
        
        # Look at center of viewport
        target = np.array([
            self.viewport.width / 2,
            self.viewport.height / 2,
            0.0
        ])
        
        up = np.array([0.0, -1.0, 0.0])  # Y points down in screen space
        
        # Create view matrix
        z = target - camera_pos
        z = z / np.linalg.norm(z)
        x = np.cross(up, z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = x
        view[1, :3] = y
        view[2, :3] = -z
        view[:3, 3] = -np.array([np.dot(x, camera_pos), np.dot(y, camera_pos), np.dot(-z, camera_pos)])
        
        return view, camera_pos
    
    def _create_model_matrix(self):
        """Create model matrix for sphere transformation"""
        model = np.eye(4, dtype=np.float32)
        
        # Scale by radius
        model[0, 0] = self.radius
        model[1, 1] = self.radius
        model[2, 2] = self.radius
        
        # Translate to position
        model[0, 3] = self.position[0]
        model[1, 3] = self.position[1]
        model[2, 3] = self.position[2]
        
        return model

    def update(self, dt: float, state: Dict):
        """Update sphere position"""
        if not self.enabled:
            return
        
        self.time += dt * self.speed
        
        # Horizontal movement that crosses screen boundaries
        self.position[0] = self.viewport.width / 2 + (self.viewport.width * 0.6) * np.sin(self.time * 0.5)
        
        # Vertical movement (stays within bounds for simplicity)
        self.position[1] = self.viewport.height / 2 + self.path_scale * np.sin(self.time * 0.7)
        
        # Depth oscillation (z=0 near, z=100 far)
        self.position[2] = 50 + 50 * np.sin(self.time * 0.3)
        
        # Wrap position to stay within [0, viewport.width] range
        self.position[0] = self.position[0] % self.viewport.width

    def render(self, state: Dict):
        #"""Render the sphere with horizontal wrapping"""
        if not self.enabled or not self.shader:
            return
        
        # Determine all positions to render (for seamless wrapping)
        render_positions = [self.position.copy()]  # Always render original
        
        # Check if sphere is near left edge
        if self.position[0] < self.wrap_margin:
            # Create duplicate on right side
            right_duplicate = self.position.copy()
            right_duplicate[0] += self.viewport.width
            render_positions.append(right_duplicate)
        
        # Check if sphere is near right edge
        if self.position[0] > (self.viewport.width - self.wrap_margin):
            # Create duplicate on left side
            left_duplicate = self.position.copy()
            left_duplicate[0] -= self.viewport.width
            render_positions.append(left_duplicate)
        
        glUseProgram(self.shader)
        
        # Set common uniforms (same for all duplicates)
        loc = glGetUniformLocation(self.shader, "resolution")
        if loc != -1:
            glUniform2f(loc, float(self.viewport.width), float(self.viewport.height))
        
        loc = glGetUniformLocation(self.shader, "sphereRadius")
        if loc != -1:
            glUniform1f(loc, self.radius)
        
        loc = glGetUniformLocation(self.shader, "sphereColor")
        if loc != -1:
            glUniform3f(loc, self.color[0], self.color[1], self.color[2])
        
        # Light position (top-left-front)
        light_pos = np.array([
            self.viewport.width * 0.2,
            self.viewport.height * 0.2,
            -200.0
        ])
        loc = glGetUniformLocation(self.shader, "lightPos")
        if loc != -1:
            glUniform3f(loc, light_pos[0], light_pos[1], light_pos[2])
        
        # NO depth testing toggle - use global depth state
        # Depth values are written in shader, compatible with rain/circles (0-100 range)
        
        # Draw sphere at each position (original + duplicates for wrapping)
        glBindVertexArray(self.VAO)
        
        for pos in render_positions:
            # Update spherePos uniform for this instance
            loc = glGetUniformLocation(self.shader, "spherePos")
            if loc != -1:
                glUniform3f(loc, pos[0], pos[1], pos[2])
            
            # Draw sphere
            glDrawElements(GL_TRIANGLES, self.num_indices, GL_UNSIGNED_INT, None)
        
        glBindVertexArray(0)
        glUseProgram(0)
