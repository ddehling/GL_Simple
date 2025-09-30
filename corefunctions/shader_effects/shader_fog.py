import numpy as np
from OpenGL.GL import *
from .base import ShaderEffect

# Vertex shader for fullscreen quad
VERTEX_SHADER = """#version 310 es
precision highp float;

in vec2 position;
out vec2 v_texcoord;

void main() {
    v_texcoord = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

# Fragment shader with depth-based fog and blur
FRAGMENT_SHADER = """#version 310 es
precision highp float;

in vec2 v_texcoord;
out vec4 fragColor;

uniform sampler2D u_color_texture;
uniform sampler2D u_depth_texture;
uniform vec3 u_fog_color;
uniform float u_fog_strength;
uniform float u_fog_near;
uniform float u_fog_far;
uniform vec2 u_resolution;

// Linearize depth value
float linearize_depth(float depth) {
    float near = u_fog_near;
    float far = u_fog_far;
    float z = depth * 2.0 - 1.0; // Back to NDC
    return (2.0 * near * far) / (far + near - z * (far - near));
}

// Gaussian blur based on depth
vec4 blur_at_depth(vec2 uv, float depth_factor) {
    vec2 texel_size = 1.0 / u_resolution;
    float blur_amount = depth_factor * 8.0; // Max blur radius
    
    if (blur_amount < 0.5) {
        return texture(u_color_texture, uv);
    }
    
    vec4 result = vec4(0.0);
    float total_weight = 0.0;
    
    // 5x5 gaussian kernel (simplified for performance)
    int radius = int(ceil(blur_amount));
    radius = min(radius, 8); // Limit max radius
    
    for (int x = -radius; x <= radius; x++) {
        for (int y = -radius; y <= radius; y++) {
            vec2 offset = vec2(float(x), float(y)) * texel_size;
            float dist = length(vec2(float(x), float(y)));
            float weight = exp(-dist * dist / (2.0 * blur_amount * blur_amount));
            
            result += texture(u_color_texture, uv + offset) * weight;
            total_weight += weight;
        }
    }
    
    return result / total_weight;
}

void main() {
    // Sample depth
    float depth = texture(u_depth_texture, v_texcoord).r;
    
    // Linearize depth to get distance from camera
    float linear_depth = linearize_depth(depth);
    
    // Calculate fog factor based on depth (0 = no fog, 1 = full fog)
    float depth_range = u_fog_far - u_fog_near;
    float fog_factor = clamp((linear_depth - u_fog_near) / depth_range, 0.0, 1.0);
    fog_factor = fog_factor * u_fog_strength;
    
    // Apply depth-based blur
    vec4 blurred_color = blur_at_depth(v_texcoord, fog_factor);
    
    // Mix blurred color with fog color based on fog factor
    vec3 final_color = mix(blurred_color.rgb, u_fog_color, fog_factor * 0.7);
    
    fragColor = vec4(final_color, blurred_color.a);
}
"""


class ShaderFog(ShaderEffect):
    """Post-processing fog effect with depth-based blur"""
    
    def __init__(self, viewport, strength=0.5, color=(0.7, 0.7, 0.8), 
                 fog_near=10.0, fog_far=100.0):
        super().__init__(viewport)
        self.strength = strength
        self.fog_color = color
        self.fog_near = fog_near
        self.fog_far = fog_far
        
        # OpenGL resources
        self.program = None
        self.vao = None
        self.vbo = None
        # No need for separ

    def create_program(self, vertex_src, fragment_src):
        """Compile and link shader program"""
        # Compile vertex shader
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, vertex_src)
        glCompileShader(vertex_shader)
        
        # Check vertex shader compilation
        if not glGetShaderiv(vertex_shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(vertex_shader).decode()
            raise RuntimeError(f"Vertex shader compilation failed:\n{error}")
        
        # Compile fragment shader
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, fragment_src)
        glCompileShader(fragment_shader)
        
        # Check fragment shader compilation
        if not glGetShaderiv(fragment_shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(fragment_shader).decode()
            raise RuntimeError(f"Fragment shader compilation failed:\n{error}")
        
        # Link program
        program = glCreateProgram()
        glAttachShader(program, vertex_shader)
        glAttachShader(program, fragment_shader)
        glLinkProgram(program)
        
        # Check program linking
        if not glGetProgramiv(program, GL_LINK_STATUS):
            error = glGetProgramInfoLog(program).decode()
            raise RuntimeError(f"Program linking failed:\n{error}")
        
        # Clean up shaders (they're linked into the program now)
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        
        return program
    

    def init(self):
        """Initialize shader program and geometry"""
        self.program = self.create_program(VERTEX_SHADER, FRAGMENT_SHADER)
        
        # Create fullscreen quad
        vertices = np.array([
            -1, -1,
             1, -1,
            -1,  1,
             1,  1,
        ], dtype=np.float32)
        
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        pos_loc = glGetAttribLocation(self.program, "position")
        glEnableVertexAttribArray(pos_loc)
        glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 0, None)
        
        glBindVertexArray(0)
        
        # Create depth texture
        self.depth_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.depth_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16,
                     self.viewport.width, self.viewport.height,
                     0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    
    def update(self, dt, state):
        """Update fog parameters from state"""
        # Allow dynamic control from state dict
        if f'fog_strength_{self.viewport.frame_id}' in state:
            self.strength = state[f'fog_strength_{self.viewport.frame_id}']
        elif 'fog_strength' in state:
            self.strength = state['fog_strength']
        
        if f'fog_color_{self.viewport.frame_id}' in state:
            self.fog_color = state[f'fog_color_{self.viewport.frame_id}']
        elif 'fog_color' in state:
            self.fog_color = state['fog_color']
    
    def render(self, state):
        """Apply fog as post-processing effect"""
        # Disable depth testing for post-process
        glDisable(GL_DEPTH_TEST)
        
        glUseProgram(self.program)
        glBindVertexArray(self.vao)
        
        # Bind color texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.viewport.color_texture)
        glUniform1i(glGetUniformLocation(self.program, "u_color_texture"), 0)
        
        # Bind depth texture from viewport
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.viewport.depth_texture)
        glUniform1i(glGetUniformLocation(self.program, "u_depth_texture"), 1)
        
        # Set uniforms
        glUniform3f(glGetUniformLocation(self.program, "u_fog_color"),
                   self.fog_color[0], self.fog_color[1], self.fog_color[2])
        glUniform1f(glGetUniformLocation(self.program, "u_fog_strength"),
                   self.strength)
        glUniform1f(glGetUniformLocation(self.program, "u_fog_near"),
                   self.fog_near)
        glUniform1f(glGetUniformLocation(self.program, "u_fog_far"),
                   self.fog_far)
        glUniform2f(glGetUniformLocation(self.program, "u_resolution"),
                   float(self.viewport.width), float(self.viewport.height))
        
        # Draw fullscreen quad
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        
        glBindVertexArray(0)
        glUseProgram(0)
        
        # Re-enable depth testing
        glEnable(GL_DEPTH_TEST)
    
    def cleanup(self):
        """Clean up resources"""
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
        if self.vbo:
            glDeleteBuffers(1, [self.vbo])
        if self.program:
            glDeleteProgram(self.program)
