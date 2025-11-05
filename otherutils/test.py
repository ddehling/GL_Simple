import numpy as np
import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders
from OpenGL.GL import *
import glfw
import glm

# Vertex shader (shared)
VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fragColor;

void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
    fragColor = color;
}
"""

# Fragment shader for near objects (more opaque)
FRAGMENT_SHADER_NEAR = """
#version 330 core
in vec3 fragColor;
out vec4 finalColor;

uniform float alpha;

void main() {
    finalColor = vec4(fragColor, alpha * 0.3);  // Near objects more opaque
}
"""

# Fragment shader for mid-range objects
FRAGMENT_SHADER_MID = """
#version 330 core
in vec3 fragColor;
out vec4 finalColor;

uniform float alpha;

void main() {
    finalColor = vec4(fragColor * 1.2, alpha * 0.5);  // Mid range, medium transparency
}
"""

# Fragment shader for far objects (more transparent)
FRAGMENT_SHADER_FAR = """
#version 330 core
in vec3 fragColor;
out vec4 finalColor;

uniform float alpha;

void main() {
    finalColor = vec4(fragColor * 0.8, alpha );  // Far objects more transparent
}
"""

class TransparentObject:
    def __init__(self, position, size, color, shader_program):
        self.position = position
        self.size = size
        self.color = color
        self.shader_program = shader_program
        self.distance_to_camera = 0.0
        
        # Create cube vertices
        vertices = np.array([
            # Front face
            -0.5, -0.5,  0.5,  color[0], color[1], color[2],
             0.5, -0.5,  0.5,  color[0], color[1], color[2],
             0.5,  0.5,  0.5,  color[0], color[1], color[2],
            -0.5,  0.5,  0.5,  color[0], color[1], color[2],
            # Back face
            -0.5, -0.5, -0.5,  color[0], color[1], color[2],
             0.5, -0.5, -0.5,  color[0], color[1], color[2],
             0.5,  0.5, -0.5,  color[0], color[1], color[2],
            -0.5,  0.5, -0.5,  color[0], color[1], color[2],
        ], dtype=np.float32)
        
        indices = np.array([
            0, 1, 2, 2, 3, 0,  # Front
            1, 5, 6, 6, 2, 1,  # Right
            5, 4, 7, 7, 6, 5,  # Back
            4, 0, 3, 3, 7, 4,  # Left
            3, 2, 6, 6, 7, 3,  # Top
            4, 5, 1, 1, 0, 4   # Bottom
        ], dtype=np.uint32)
        
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)
        
        glBindVertexArray(self.vao)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Color attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        
        glBindVertexArray(0)
    
    def calculate_distance(self, camera_pos):
        """Calculate distance from camera for sorting"""
        self.distance_to_camera = glm.length(glm.vec3(*self.position) - camera_pos)
    
    def render(self, view, projection, alpha=1.0):
        glUseProgram(self.shader_program)
        
        # Set uniforms
        model = glm.mat4(1.0)
        model = glm.translate(model, glm.vec3(*self.position))
        model = glm.scale(model, glm.vec3(self.size, self.size, self.size))
        
        glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "model"), 
                          1, GL_FALSE, glm.value_ptr(model))
        glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "view"), 
                          1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "projection"), 
                          1, GL_FALSE, glm.value_ptr(projection))
        glUniform1f(glGetUniformLocation(self.shader_program, "alpha"), alpha)
        
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

def main():
    # Initialize GLFW
    if not glfw.init():
        return
    
    # Create window
    window = glfw.create_window(800, 600, "Multi-Shader Transparent Rendering", None, None)
    if not window:
        glfw.terminate()
        return
    
    glfw.make_context_current(window)
    
    # Enable depth testing and blending
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    # Compile shaders
    shader_near = shaders.compileProgram(
        shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        shaders.compileShader(FRAGMENT_SHADER_NEAR, GL_FRAGMENT_SHADER)
    )
    
    shader_mid = shaders.compileProgram(
        shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        shaders.compileShader(FRAGMENT_SHADER_MID, GL_FRAGMENT_SHADER)
    )
    
    shader_far = shaders.compileProgram(
        shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        shaders.compileShader(FRAGMENT_SHADER_FAR, GL_FRAGMENT_SHADER)
    )
    
    # Create transparent objects at different ranges with different shaders
    objects = [
        # Near objects (z = -5 to -10) - red cubes
        TransparentObject([2, 0, -5], 1.0, [1.0, 0.2, 0.2], shader_near),
        TransparentObject([-2, 1, -7], 1.0, [1.0, 0.3, 0.3], shader_near),
        TransparentObject([0, -1, -8], 1.0, [1.0, 0.1, 0.1], shader_near),
        
        # Mid-range objects (z = -12 to -18) - green cubes
        TransparentObject([1, 0, -12], 1.2, [0.2, 1.0, 0.2], shader_mid),
        TransparentObject([-1, 1, -15], 1.2, [0.3, 1.0, 0.3], shader_mid),
        TransparentObject([0, -1, -17], 1.2, [0.1, 1.0, 0.1], shader_mid),
        
        # Far objects (z = -20 to -25) - blue cubes
        TransparentObject([2, 0, -20], 1.5, [0.2, 0.2, 1.0], shader_far),
        TransparentObject([-2, 1, -23], 1.5, [0.3, 0.3, 1.0], shader_far),
        TransparentObject([0, -1, -25], 1.5, [0.1, 0.1, 1.0], shader_far),
    ]
    
    # Camera setup
    camera_pos = glm.vec3(0, 0, 0)
    
    # Main loop
    angle = 0
    while not glfw.window_should_close(window):
        glfw.poll_events()
        
        # Clear buffers
        glClearColor(0.1, 0.1, 0.15, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Update camera rotation
        angle += 0.01
        camera_pos = glm.vec3(np.sin(angle) * 5, 2, np.cos(angle) * 5)
        
        # Setup matrices
        view = glm.lookAt(camera_pos, glm.vec3(0, 0, -15), glm.vec3(0, 1, 0))
        projection = glm.perspective(glm.radians(45.0), 800/600, 0.1, 100.0)
        
        # Calculate distances and sort objects back-to-front
        for obj in objects:
            obj.calculate_distance(camera_pos)
        objects.sort(key=lambda x: x.distance_to_camera, reverse=True)
        
        # Render with depth writes disabled for transparency
        glDepthMask(GL_FALSE)
        
        for obj in objects:
            obj.render(view, projection, alpha=1.0)
        
        glDepthMask(GL_TRUE)
        
        glfw.swap_buffers(window)
    
    glfw.terminate()

if __name__ == "__main__":
    main()