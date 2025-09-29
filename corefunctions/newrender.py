import cv2
import numpy as np
from OpenGL.GL import *  # noqa: F403
from OpenGL.GLU import *  # noqa: F403
from PIL import Image
from dataclasses import dataclass
from typing import Union, Tuple, Optional, Dict, List
import uuid
import moderngl

@dataclass
class ImagePlane:
    """Class to hold all properties of a rendered image plane"""
    texture: moderngl.Texture
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: Tuple[float, float] = (1.0, 1.0)
    alpha: float = 1.0
    ambient: Tuple[float, float, float] = (0.2, 0.2, 0.2)
    diffuse: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    specular: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    shininess: float = 32.0
    enabled: bool = True
    
    def __post_init__(self):
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("Alpha must be between 0.0 and 1.0")
        if self.shininess < 0:
            raise ValueError("Shininess must be non-negative")

class ImageRenderer:
    """Handles the actual OpenGL rendering"""
    def __init__(self, frame_dimensions: List[Tuple[int, int]], enable_lighting=True):
        """
        Initialize renderer with multiple frames
        
        Args:
            frame_dimensions: List of (width, height) tuples for each frame
            enable_lighting: Whether to enable lighting effects
        """
        self.frame_dimensions = frame_dimensions
        self.num_frames = len(frame_dimensions)
        self.enable_lighting = enable_lighting
        self.ctx = moderngl.create_standalone_context()
        
        # Create framebuffers for each frame
        self.fbos = []
        for width, height in frame_dimensions:
            fbo = self.ctx.framebuffer(
                color_attachments=[self.ctx.texture((width, height), 4)]
            )
            self.fbos.append(fbo)
        
        # Set up OpenGL state
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        # Create fog parameters for each frame
        self.fog_params = []
        for _ in range(self.num_frames):
            self.fog_params.append({
                'amount': 0.0, 
                'color': (0.0, 0.0, 0.7), 
                'dir_scale': (1.0, 1.0)
            })

        # Create basic shader program
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 mvp;
                uniform vec2 translation;
                in vec3 in_position;
                in vec2 in_texcoord;
                out vec2 v_texcoord;
                out float v_fog_factor;  // Add fog factor output
                
                void main() {
                    vec4 pos = mvp * vec4(in_position, 1.0);
                    pos.xy += translation;
                    gl_Position = pos;
                    v_texcoord = in_texcoord;
                    
                    // Calculate fog factor based on distance from camera
                    float distance = length(pos.xyz);
                    v_fog_factor = clamp(distance/20, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D texture0;
                uniform float alpha;
                uniform float fog_amount;
                uniform vec3 fog_color;
                uniform vec2 fog_dir_scale;  // Added directional scaling for fog

                in vec2 v_texcoord;
                in float v_fog_factor;
                out vec4 f_color;
                
                vec4 blur(sampler2D image, vec2 uv, vec2 resolution, float radius) {
                    vec4 total = vec4(0.0);
                    float samples = 0.0;
                    
                    // Larger sampling area with directional scaling
                    float blur_size = radius * 0.15; // Base blur size
                    
                    // Sample in a grid pattern with directional scaling
                    for (float x = -2.0; x <= 2.0; x += 1.0) {
                        for (float y = -2.0; y <= 2.0; y += 1.0) {
                            vec2 scaled_offset = vec2(x * fog_dir_scale.x, y * fog_dir_scale.y) * blur_size / resolution;
                            total += texture(image, uv + scaled_offset);
                            samples += 1.0;
                        }
                    }
                    
                    return total/2;
                }
                
                void main() {
                    vec2 resolution = textureSize(texture0, 0);
                    
                    // Calculate blur radius based on distance and fog amount
                    float blur_radius = v_fog_factor * fog_amount * 5;
                    
                    // Get original color
                    vec4 original = texture(texture0, v_texcoord);
                    
                    // Get blurred color
                    vec4 blurred = blur(texture0, v_texcoord, resolution, blur_radius);
                    
                    // Mix between original and blurred based on fog factor
                    vec4 tex_color = mix(original, blurred, v_fog_factor * fog_amount/2);
                    
                    // Apply fog color blend
                    vec3 final_color = mix(tex_color.rgb, fog_color, v_fog_factor * fog_amount*0.75 );
                    
                    f_color = vec4(final_color, tex_color.a * alpha);
                }
            '''
        )

        # Create quad vertices for rendering
        vertices = np.array([
            # x      y     z     u     v
            -0.5, -0.5,  0.0,  0.0,  0.0,  # Bottom left
             0.5, -0.5,  0.0,  1.0,  0.0,  # Bottom right
             0.5,  0.5,  0.0,  1.0,  1.0,  # Top right
            -0.5,  0.5,  0.0,  0.0,  1.0,  # Top left
        ], dtype='f4')

        indices = np.array([0, 1, 2, 0, 2, 3], dtype='i4')
        
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())
        
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo, '3f 2f', 'in_position', 'in_texcoord'),
            ],
            self.ibo
        )
        
        # Set reference_z for use in MVP calculation
        self.reference_z = 5.0

    def clear_frame(self, frame_id: int):
        """Clear a specific framebuffer"""
        if not 0 <= frame_id < self.num_frames:
            raise ValueError(f"Invalid frame_id: {frame_id}. Must be between 0 and {self.num_frames-1}")
            
        fbo = self.fbos[frame_id]
        width, height = self.frame_dimensions[frame_id]
        
        fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        fbo.clear(0.0, 0.0, 0.0, 0.0)

    def create_texture_from_array(self, array: np.ndarray) -> moderngl.Texture:
        """Create a texture from a numpy array
        
        Args:
            array: numpy array with shape (H, W, C) and dtype uint8
                  C should be 3 (RGB) or 4 (RGBA)
        """
        if array.dtype != np.uint8:
            raise ValueError("Array must be uint8")
        
        if len(array.shape) != 3:
            raise ValueError("Array must be 3-dimensional (H, W, C)")
            
        height, width, channels = array.shape
        if channels not in [3, 4]:
            raise ValueError("Array must have 3 (RGB) or 4 (RGBA) channels")
            
        # Convert RGB to RGBA if necessary
        if channels == 3:
            rgba = np.zeros((height, width, 4), dtype=np.uint8)
            rgba[..., :3] = array
            rgba[..., 3] = 255
            array = rgba
            
        # Ensure array is contiguous and correct format
        array = np.ascontiguousarray(array)
        
        # Create texture
        texture = self.ctx.texture((width, height), 4, array.tobytes())
        texture.use(0)
        texture.repeat_x = False
        texture.repeat_y = False
        
        return texture

    def load_texture(self, source: Union[str, np.ndarray]) -> moderngl.Texture:
        """Load texture from either file or numpy array"""
        if isinstance(source, str):
            with Image.open(source) as img:
                img = img.convert('RGBA')
                texture = self.ctx.texture(img.size, 4, img.tobytes())
                texture.use(0)
                return texture
        elif isinstance(source, np.ndarray):
            return self.create_texture_from_array(source)
        else:
            raise ValueError("Source must be either a file path or numpy array")

    def draw_image_plane(self, plane, frame_id: int):
        """Draw an image plane to a specific frame buffer"""
        if not 0 <= frame_id < self.num_frames:
            raise ValueError(f"Invalid frame_id: {frame_id}. Must be between 0 and {self.num_frames-1}")
            
        fbo = self.fbos[frame_id]
        width, height = self.frame_dimensions[frame_id]
        fog_params = self.fog_params[frame_id]
        
        fbo.use()
        self.ctx.viewport = (0, 0, width, height)

        mvp = self.get_mvp_matrix(plane, frame_id)
        plane.texture.use(0)
        self.prog['mvp'].write(mvp.astype('f4').tobytes())
        self.prog['translation'].value = (plane.position[0], plane.position[1])
        self.prog['alpha'].value = plane.alpha
        self.prog['fog_amount'].value = fog_params['amount']
        self.prog['fog_color'].value = fog_params['color']
        self.prog['fog_dir_scale'].value = fog_params['dir_scale']
        self.vao.render()

    def set_fog(self, frame_id: int, amount: float, color: Tuple[float, float, float] = None, 
                dir_scale: Tuple[float, float] = None):
        """Set fog parameters for specified frame"""
        if not 0 <= frame_id < self.num_frames:
            raise ValueError(f"Invalid frame_id: {frame_id}. Must be between 0 and {self.num_frames-1}")
            
        fog_params = self.fog_params[frame_id]
        fog_params['amount'] = max(0.0, min(1.0, amount))
        if color is not None:
            fog_params['color'] = color
        if dir_scale is not None:
            fog_params['dir_scale'] = dir_scale
    
    def get_mvp_matrix(self, plane, frame_id: int):
        """Calculate Model-View-Projection matrix with calibrated perspective"""
        width, height = self.frame_dimensions[frame_id]
        
        fov_y = np.radians(60)
        # Adjust reference_z to be closer
        
        # Standard perspective projection matrix
        aspect = width / height
        near = 0.1
        far = 100.0
        f = 1.0 / np.tan(fov_y / 2)
        
        projection = np.array([
            [f/aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far+near)/(far-near), (-2*far*near)/(far-near)],
            [0, 0, 1, 0]
        ], dtype=np.float32)

        # Move camera back along z-axis and look at center
        view = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -self.reference_z],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Scale objects to be visible in viewport
        viewport_scale = self.reference_z * np.tan(fov_y/2) / (height/2)
        
        # Model matrix combines scale, rotation, and translation
        model = np.identity(4, dtype=np.float32)
        
        # Scale matrix - scale by texture dimensions and viewport scale
        tex_width = plane.texture.width
        tex_height = plane.texture.height
        
        scale_mat = np.array([
            [tex_width * viewport_scale * plane.scale[0], 0, 0, 0],
            [0, tex_height * viewport_scale * plane.scale[1], 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Rotation matrices
        rx, ry, rz = np.radians(plane.rotation)
        
        rotx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(rx), -np.sin(rx), 0],
            [0, np.sin(rx), np.cos(rx), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        roty = np.array([
            [np.cos(ry), 0, np.sin(ry), 0],
            [0, 1, 0, 0],
            [-np.sin(ry), 0, np.cos(ry), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        rotz = np.array([
            [np.cos(rz), -np.sin(rz), 0, 0],
            [np.sin(rz), np.cos(rz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Scale positions by viewport scale to match scaled geometry
        trans = np.array([
            [1, 0, 0, plane.position[0] * viewport_scale],
            [0, 1, 0, plane.position[1] * viewport_scale],
            [0, 0, 1, plane.position[2]],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        trans = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, plane.position[2]],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Combine transformations
        model = trans @ rotz @ roty @ rotx @ scale_mat @ model
        mvp = projection @ view @ model

        return mvp
   
    def get_frame(self, frame_id: int) -> np.ndarray:
        """Get a specific frame as a numpy array"""
        if not 0 <= frame_id < self.num_frames:
            raise ValueError(f"Invalid frame_id: {frame_id}. Must be between 0 and {self.num_frames-1}")
            
        fbo = self.fbos[frame_id]
        width, height = self.frame_dimensions[frame_id]
        
        fbo.use()
        data = fbo.read(components=4, dtype='f4')
        image = np.frombuffer(data, dtype=np.float32).reshape(height, width, 4)
        return (image * 255).astype(np.uint8)
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'ctx'):
            self.ctx.release()

    def get_frame_dimensions(self, frame_id: int) -> Tuple[int, int]:
        """Get the width and height of a specific frame
        
        Args:
            frame_id: The frame index to query
            
        Returns:
            Tuple of (width, height)
            
        Raises:
            ValueError: If frame_id is invalid
        """
        if not 0 <= frame_id < self.num_frames:
            raise ValueError(f"Invalid frame_id: {frame_id}. Must be between 0 and {self.num_frames-1}")
        return self.frame_dimensions[frame_id]

class Scene:
    """Manages collection of ImagePlanes and their lifecycle"""
    def __init__(self, renderer: ImageRenderer, frame_id: int = 0):
        self.renderer = renderer
        self.frame_id = frame_id
        self.width, self.height = renderer.get_frame_dimensions(frame_id)
        self.image_planes: Dict[str, ImagePlane] = {}
        
    def create_image_plane(self, source: Union[str, np.ndarray], **kwargs) -> str:
        texture = self.renderer.load_texture(source)
        plane = ImagePlane(texture=texture, **kwargs)
        plane_id = str(uuid.uuid4())
        self.image_planes[plane_id] = plane
        return plane_id
    
    def update_image_plane_texture(self, plane_id: str, source: Union[str, np.ndarray]):
        """Update the texture of an existing ImagePlane"""
        if plane_id not in self.image_planes:
            raise KeyError(f"No image plane with id {plane_id}")
        plane = self.image_planes[plane_id]
        
        if isinstance(source, np.ndarray):
            # Ensure the data is in the correct format
            if source.dtype != np.uint8:
                source = source.astype(np.uint8)
                
            # Get the current texture dimensions
            current_width, current_height = plane.texture.width, plane.texture.height
            source_height, source_width = source.shape[:2]
            
            # Check if dimensions match
            if current_width == source_width and current_height == source_height:
                # Ensure we have 4 channels (RGBA)
                if source.shape[2] == 3:  # RGB needs conversion to RGBA
                    rgba = np.zeros((source_height, source_width, 4), dtype=np.uint8)
                    rgba[..., :3] = source
                    rgba[..., 3] = 255
                    source = rgba
                # Write directly to the existing texture
                plane.texture.write(source.tobytes())
            else:
                # Dimensions don't match, need to create new texture
                # Release old texture
                plane.texture.release()
                # Create new texture
                plane.texture = self.renderer.load_texture(source)
        else:
            # For file sources, create new texture
            plane.texture.release()
            new_texture = self.renderer.load_texture(source)
            plane.texture = new_texture

    def remove_image_plane(self, plane_id: str):
        """Remove an ImagePlane and clean up its resources"""
        if plane_id in self.image_planes:
            plane = self.image_planes[plane_id]
            plane.texture.release()  # Release the texture
            del self.image_planes[plane_id]
    
    def get_image_plane(self, plane_id: str) -> Optional[ImagePlane]:
        """Get an ImagePlane by its ID"""
        return self.image_planes.get(plane_id)
    
    def update_image_plane(self, plane_id: str, **kwargs):
        """Update properties of an existing ImagePlane"""
        if plane_id in self.image_planes:
            plane = self.image_planes[plane_id]
            for key, value in kwargs.items():
                if hasattr(plane, key):
                    setattr(plane, key, value)
    
    def render(self) -> np.ndarray:
        self.renderer.clear_frame(self.frame_id)

        sorted_planes = sorted(
            self.image_planes.values(),
            key=lambda p: p.position[2],
            reverse=True
        )
        
        for plane in sorted_planes:
            if plane.enabled:
                self.renderer.draw_image_plane(plane, self.frame_id)
                
        return self.renderer.get_frame(self.frame_id)

def create_animated_texture(size, frame):
    """Create a dynamic texture pattern"""
    array = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Create a moving pattern
    x = np.linspace(0, 10, size)
    y = np.linspace(0, 10, size)
    X, Y = np.meshgrid(x, y)
    
    # Animated sine pattern
    Z = np.sin(X + frame * 0.1) * np.cos(Y + frame * 0.1)
    Z = (Z + 1) * 127.5  # Convert to 0-255 range
    
    array[:, :, 0] = Z  # Red channel
    array[:, :, 1] = np.roll(Z, shift=size//3, axis=0)  # Green channel
    array[:, :, 2] = np.roll(Z, shift=2*size//3, axis=0)  # Blue channel
    
    return array

def main():
    """Example usage with multiple frames"""
    # Define dimensions for multiple frames
    frame_dimensions = [
        (120, 60),   # Frame 0 (previously primary)
        (300, 32),   # Frame 1 (previously secondary)
        (200, 200)   # Frame 2 (new additional frame)
    ]
    
    renderer = ImageRenderer(frame_dimensions, enable_lighting=True)
    
    # Set fog for frame 0
    renderer.set_fog(0, 1.0, (0.0, 1.0, 0.0))
    
    # Create scenes for different frames
    scene0 = Scene(renderer, frame_id=0)
    scene1 = Scene(renderer, frame_id=1)
    scene2 = Scene(renderer, frame_id=2)
    
    # Create content for frame 0
    array1 = np.zeros((100, 200, 3), dtype=np.uint8)
    array1[:50, :, 0] = 255  # Red top half
    array1[50:, :, 1] = 255  # Green bottom half
    
    plane1_id = scene0.create_image_plane(
        array1,
        position=(0, 0, 40),
        rotation=(0, 0, 0),
        scale=(2, 2)
    )
    
    # Create content for frame 1
    plane2_id = scene1.create_image_plane(
        "C:\\Users\\diete\\Desktop\\devel-local\\LED-Sign\\media\\images\\PCshirt2.png",
        position=(0, 0, 50),
        rotation=(0, 0, 0),
        scale=(10, 10)
    )
    
    # Create content for frame 2 (new frame)
    plane3_id = scene2.create_image_plane(
        create_animated_texture(100, 0),
        position=(0, 0, 30),
        rotation=(0, 0, 0),
        scale=(1.5, 1.5)
    )
    
    cv2.namedWindow('Frame 0', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Frame 1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Frame 2', cv2.WINDOW_NORMAL)
    
    try:
        angle = 0
        while True:
            # Oscillating fog effect for frame 0
            fog_amount = (np.sin(angle * 0.02) * 0.5 + 0.5)
            renderer.set_fog(0, fog_amount)
            
            # Update animated texture for frame 2
            scene2.update_image_plane_texture(plane3_id, create_animated_texture(100, angle))
            
            # Render all frames
            frame0 = scene0.render()
            frame1 = scene1.render()
            frame2 = scene2.render()
            
            # Display all frames
            cv2.imshow('Frame 0', cv2.cvtColor(frame0, cv2.COLOR_RGBA2BGR))
            cv2.imshow('Frame 1', cv2.cvtColor(frame1, cv2.COLOR_RGBA2BGR))
            cv2.imshow('Frame 2', cv2.cvtColor(frame2, cv2.COLOR_RGBA2BGR))
            
            if cv2.waitKey(10) == 27:  # ESC
                break
                
            angle += 1
            
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()