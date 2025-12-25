"""
Movie playback shader effect - GPU-accelerated video with scaling and rotation
Plays video files with real-time transformations and audio playback
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Dict
import cv2
import os
from .base import ShaderEffect

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("WARNING: moviepy not available - audio playback disabled")

# ============================================================================
# Event Wrapper Function - Integrates with EventScheduler
# ============================================================================

def shader_movie(state, outstate, video_path="", x=None, y=None, scale=1.0, 
                 rotation=0.0, depth=50.0, loop=True, fade_duration=2.0, start_time=0.0,
                 enable_audio=False, audio_volume=1.0):
    """
    Movie playback effect compatible with EventScheduler
    
    Usage:
        scheduler.schedule_event(0, 60, shader_movie, 
                               video_path="media/video.mp4",
                               x=512, y=384, scale=1.5, 
                               rotation=15.0, start_time=5.0, 
                               enable_audio=True, audio_volume=0.8,
                               frame_id=0)
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, count, frame_id)
        outstate: Global state dict (from EventScheduler)
        video_path: Path to video file (relative to workspace root or absolute)
        x: X position of video center (default: viewport center)
        y: Y position of video center (default: viewport center)
        scale: Scale multiplier (1.0 = original size, 2.0 = double size)
        rotation: Rotation angle in degrees (clockwise)
        depth: Z-depth for 3D ordering (0=near, 100=far, default=50)
        loop: Whether to loop the video (default=True)
        fade_duration: Duration of fade in/out in seconds (default 2.0)
        start_time: Starting position in video in seconds (default 0.0)
        enable_audio: Whether to play audio track (default=False, requires moviepy)
        audio_volume: Audio volume multiplier 0.0-1.0 (default=1.0)
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
    
    # Initialize effect on first call
    if state['count'] == 0:
        print(f"Initializing movie effect for frame {frame_id}")
        
        # Default to viewport center if position not specified
        if x is None:
            x = viewport.width / 2.0
        if y is None:
            y = viewport.height / 2.0
        
        try:
            effect = viewport.add_effect(
                MovieEffect,
                video_path=video_path,
                x=x,
                y=y,
                scale=scale,
                rotation=rotation,
                depth=depth,
                loop=loop,
                start_time=start_time,
                enable_audio=enable_audio,
                audio_volume=audio_volume
            )
            state['movie_effect'] = effect
            state['movie_outstate'] = outstate  # Store reference for audio injection
            print(f"✓ Initialized shader movie for frame {frame_id}: {video_path}")
        except Exception as e:
            import traceback
            print(f"ERROR initializing movie effect: {e}")
            traceback.print_exc()
            return
    
    # Update effect parameters from state
    if 'movie_effect' in state:
        effect = state['movie_effect']
        
        # Update fade factor based on elapsed time
        elapsed_time = state['elapsed_time']
        total_duration = state.get('duration', 60)  # Default 60s if not set
        
        # Calculate fade factor (0.0 to 1.0)
        if elapsed_time < fade_duration:
            # Fade in
            fade_factor = elapsed_time / fade_duration
        elif elapsed_time > (total_duration - fade_duration):
            # Fade out
            fade_factor = (total_duration - elapsed_time) / fade_duration
        else:
            # Full visibility
            fade_factor = 1.0
        
        # Update effect's fade factor (clip to 0-1 range)
        effect.fade_factor = np.clip(fade_factor, 0, 1)
        
        # Allow dynamic parameter updates from outstate
        effect.rotation = outstate.get('movie_rotation', rotation)
        effect.scale = outstate.get('movie_scale', scale)
        
        # Inject audio into outstate if enabled
        if effect.enable_audio and effect.audio_buffer is not None:
            effect.inject_audio_to_outstate(outstate)
    
    # Cleanup on close event
    if state['count'] == -1:
        if 'movie_effect' in state:
            effect = state['movie_effect']
            if effect in viewport.effects:
                viewport.effects.remove(effect)
            effect.cleanup()
            del state['movie_effect']
            print(f"✓ Cleaned up shader movie for frame {frame_id}")


# ============================================================================
# Movie Effect Class
# ============================================================================

class MovieEffect(ShaderEffect):
    """GPU-based movie playback with scaling and rotation"""
    
    def __init__(self, viewport, video_path: str = "", x: float = 0.0, y: float = 0.0,
                 scale: float = 1.0, rotation: float = 0.0, depth: float = 50.0,
                 loop: bool = True, start_time: float = 0.0, enable_audio: bool = False,
                 audio_volume: float = 1.0):
        super().__init__(viewport)
        self.video_path = video_path
        self.x = x
        self.y = y
        self.scale = scale
        self.rotation = rotation
        self.depth = depth
        self.loop = loop
        self.start_time = start_time
        self.enable_audio = enable_audio
        self.audio_volume = audio_volume
        self.fade_factor = 0.0  # For fade in/out (updated by event wrapper)
        
        # Video capture
        self.video_capture = None
        self.video_width = 0
        self.video_height = 0
        self.fps = 30.0
        self.frame_count = 0
        self.current_frame_index = 0
        self.start_frame_index = 0  # Where to loop back to
        self.current_frame = None
        
        # Audio handling
        self.audio_clip = None
        self.audio_buffer = None
        self.audio_sample_rate = 44100
        self.audio_playback_time = 0.0  # Current audio playback position
        
        # OpenGL texture
        self.texture = None
        
        # Time tracking for frame updates
        self.time_accumulator = 0.0
        
        # Load video
        if video_path:
            self._load_video(video_path)
    
    def _load_video(self, video_path: str):
        """Load video file using OpenCV"""
        # Handle relative paths (from workspace root)
        if not os.path.isabs(video_path):
            # Try relative to workspace root
            workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            full_path = os.path.join(workspace_root, video_path)
            if not os.path.exists(full_path):
                # Try relative to current directory
                full_path = video_path
        else:
            full_path = video_path
        
        if not os.path.exists(full_path):
            print(f"ERROR: Video file not found: {full_path}")
            return
        
        self.video_capture = cv2.VideoCapture(full_path)
        
        if not self.video_capture.isOpened():
            print(f"ERROR: Failed to open video: {full_path}")
            return
        
        # Get video properties
        self.video_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"✓ Loaded video: {full_path}")
        print(f"  Size: {self.video_width}x{self.video_height}")
        print(f"  FPS: {self.fps}")
        print(f"  Frames: {self.frame_count}")
        
        # Seek to start time if specified
        if self.start_time > 0:
            self.start_frame_index = int(self.start_time * self.fps)
            # Clamp to valid range
            self.start_frame_index = min(self.start_frame_index, self.frame_count - 1)
            
            # Seek to the position
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame_index)
            
            # Verify the seek worked
            actual_pos = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            if actual_pos != self.start_frame_index:
                print(f"  WARNING: Seek requested frame {self.start_frame_index}, got {actual_pos}")
                # Try alternative method: read and discard frames
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                for _ in range(self.start_frame_index):
                    self.video_capture.read()
                actual_pos = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                print(f"  Sequential seek to frame {actual_pos}")
            
            self.current_frame_index = actual_pos
            print(f"  Starting at: {self.start_time}s (frame {actual_pos})")
        
        # Read first frame
        self._read_next_frame()
        
        # Load audio if enabled (non-blocking - errors won't prevent video playback)
        if self.enable_audio:
            try:
                self._load_audio(full_path)
            except Exception as e:
                print(f"  WARNING: Failed to load audio (video will still play): {e}")
                self.audio_buffer = None
    
    def _read_next_frame(self):
        """Read the next frame from video"""
        if self.video_capture is None:
            return False
        
        ret, frame = self.video_capture.read()
        
        if not ret:
            # End of video
            if self.loop and self.frame_count > 0:
                # Loop back to start frame (respects start_time)
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame_index)
                self.current_frame_index = self.start_frame_index
                ret, frame = self.video_capture.read()
                if not ret:
                    return False
            else:
                return False
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip vertically (OpenGL texture coords are bottom-up)
        frame = np.flip(frame, axis=0)
        
        self.current_frame = frame
        self.current_frame_index += 1
        
        return True
        
    
    def _load_audio(self, video_path: str):
        """Load audio track from video using moviepy"""
        if not MOVIEPY_AVAILABLE:
            print("  Audio disabled: moviepy not installed")
            return
        
        try:
            # Load video with moviepy to access audio
            self.audio_clip = VideoFileClip(video_path)
            
            if self.audio_clip.audio is None:
                print("  No audio track found in video")
                return
            
            # Get audio properties
            self.audio_sample_rate = self.audio_clip.audio.fps
            duration = self.audio_clip.duration
            
            # Extract entire audio as numpy array
            audio_array = self.audio_clip.audio.to_soundarray(fps=self.audio_sample_rate)
            
            # Convert to mono if stereo (average channels)
            if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Apply volume
            audio_array = audio_array * self.audio_volume
            
            # Store audio buffer
            self.audio_buffer = audio_array.astype(np.float32)
            
            # Set playback time to match video start
            self.audio_playback_time = self.start_time
            
            print(f"  Audio loaded: {self.audio_sample_rate}Hz, {duration:.2f}s, {len(self.audio_buffer)} samples")
            
        except Exception as e:
            print(f"  WARNING: Failed to load audio: {e}")
            self.audio_buffer = None
    
    def inject_audio_to_outstate(self, outstate: Dict):
        """Inject current audio samples into outstate playback buffer"""
        if self.audio_buffer is None or not self.enable_audio:
            return
        
        # Get or create audio output buffer in outstate
        if 'movie_audio_output' not in outstate:
            outstate['movie_audio_output'] = np.zeros(4096, dtype=np.float32)  # Buffer for mixing
        
        # Calculate sample position
        sample_position = int(self.audio_playback_time * self.audio_sample_rate)
        
        # Clamp to valid range
        if sample_position >= len(self.audio_buffer):
            if self.loop:
                # Loop audio back to start
                sample_position = int(self.start_time * self.audio_sample_rate)
                self.audio_playback_time = self.start_time
            else:
                return  # Audio finished
        
        # Get chunk of audio for this frame (typical 1024-4096 samples at 44.1kHz)
        chunk_size = min(2048, len(self.audio_buffer) - sample_position)
        
        if chunk_size > 0:
            audio_chunk = self.audio_buffer[sample_position:sample_position + chunk_size]
            
            # Apply fade factor to audio
            audio_chunk = audio_chunk * self.fade_factor
            
            # Store in outstate (resize buffer to match chunk)
            outstate['movie_audio_output'] = audio_chunk
        self.current_frame = frame
        self.current_frame_index += 1
        
        return True
    
    def _upload_frame_to_texture(self):
        """Upload current frame to OpenGL texture"""
        if self.current_frame is None:
            return
        
        if self.texture is None:
            # Create texture on first upload
            self.texture = glGenTextures(1)
        
        glBindTexture(GL_TEXTURE_2D, self.texture)
        
        # Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # Upload frame data
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.video_width, self.video_height,
                     0, GL_RGB, GL_UNSIGNED_BYTE, self.current_frame)
        
        glBindTexture(GL_TEXTURE_2D, 0)
    
    def update(self, dt: float, state: Dict):
        """Update movie playback"""
        if not self.enabled or self.video_capture is None:
            return
        
        # Accumulate time and advance frames based on video FPS
        self.time_accumulator += dt
        frame_duration = 1.0 / self.fps if self.fps > 0 else 1.0 / 30.0
        
        while self.time_accumulator >= frame_duration:
            self.time_accumulator -= frame_duration
            self._read_next_frame()
        
        # Update audio playback time
        if self.enable_audio and self.audio_buffer is not None:
            self.audio_playback_time += dt
    
    def render(self, state: Dict):
        """Render movie frame as textured quad"""
        if not self.enabled:
            return
        
        if self.current_frame is None:
            return
        
        # Upload current frame to texture
        self._upload_frame_to_texture()
        
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        
        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        
        # Set uniforms
        res_loc = glGetUniformLocation(self.shader, "resolution")
        glUniform2f(res_loc, self.viewport.width, self.viewport.height)
        
        pos_loc = glGetUniformLocation(self.shader, "position")
        glUniform2f(pos_loc, self.x, self.y)
        
        size_loc = glGetUniformLocation(self.shader, "videoSize")
        glUniform2f(size_loc, self.video_width, self.video_height)
        
        scale_loc = glGetUniformLocation(self.shader, "scale")
        glUniform1f(scale_loc, self.scale)
        
        rotation_loc = glGetUniformLocation(self.shader, "rotation")
        glUniform1f(rotation_loc, np.radians(self.rotation))
        
        depth_loc = glGetUniformLocation(self.shader, "depth")
        glUniform1f(depth_loc, self.depth)
        
        fade_loc = glGetUniformLocation(self.shader, "fadeAlpha")
        glUniform1f(fade_loc, self.fade_factor)
        
        tex_loc = glGetUniformLocation(self.shader, "videoTexture")
        glUniform1i(tex_loc, 0)
        
        # Draw quad
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindVertexArray(0)
        glUseProgram(0)
    
    def get_vertex_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        layout(location = 0) in vec2 vertexPos;  // Quad vertices (-1 to 1)
        layout(location = 1) in vec2 texCoord;   // Texture coordinates (0 to 1)
        
        out vec2 fragTexCoord;
        
        uniform vec2 resolution;
        uniform vec2 position;      // Center position in screen space
        uniform vec2 videoSize;     // Original video dimensions
        uniform float scale;        // Scale multiplier
        uniform float rotation;     // Rotation angle in radians
        uniform float depth;        // Z-depth (0-100)
        
        void main() {
            // Scale video to its original size, then apply scale factor
            vec2 scaledSize = videoSize * scale;
            
            // Apply rotation matrix
            float cosR = cos(rotation);
            float sinR = sin(rotation);
            mat2 rotationMatrix = mat2(cosR, -sinR, sinR, cosR);
            
            // Rotate and scale vertex
            vec2 rotatedVertex = rotationMatrix * (vertexPos * scaledSize * 0.5);
            
            // Translate to position
            vec2 worldPos = rotatedVertex + position;
            
            // Convert to clip space
            vec2 clipPos = (worldPos / resolution) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;  // Flip Y for screen coords
            
            // Map depth to 0.0-1.0 range
            float depthValue = depth / 100.0;
            depthValue = clamp(depthValue, 0.0, 1.0);
            
            gl_Position = vec4(clipPos, depthValue, 1.0);
            fragTexCoord = texCoord;
        }
        """
    
    def get_fragment_shader(self):
        return """
        #version 310 es
        precision highp float;
        
        in vec2 fragTexCoord;
        out vec4 outColor;
        
        uniform sampler2D videoTexture;
        uniform float fadeAlpha;
        
        void main() {
            vec4 texColor = texture(videoTexture, fragTexCoord);
            
            // Apply fade factor to alpha
            outColor = vec4(texColor.rgb, texColor.a * fadeAlpha);
        }
        """
    
    def compile_shader(self):
        """Compile and link movie shaders - REQUIRED by ShaderEffect base class"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        
        try:
            vertex = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
            fragment = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            return shaders.compileProgram(vertex, fragment)
        except Exception as e:
            print(f"Shader compilation error: {e}")
            raise
    
    def setup_buffers(self):
        """Initialize OpenGL buffers for textured quad"""
        # Quad vertices with texture coordinates
        # Layout: [x, y, u, v]
        vertices = np.array([
            # Vertex positions    Texture coords (U flipped to fix mirroring)
            -1.0, -1.0,           1.0, 0.0,  # Bottom-left
             1.0, -1.0,           0.0, 0.0,  # Bottom-right
             1.0,  1.0,           0.0, 1.0,  # Top-right
            -1.0,  1.0,           1.0, 1.0   # Top-left
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
        
        stride = 4 * 4  # 4 floats per vertex
        
        # Position attribute (location 0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Texture coordinate attribute (location 1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        
        # Element buffer
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        glBindVertexArray(0)
    
    def cleanup(self):
        """Clean up resources"""
        # Release video capture
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        
        # Release audio clip
        if self.audio_clip is not None:
            self.audio_clip.close()
            self.audio_clip = None
        
        # Clear audio buffer
        self.audio_buffer = None
        
        # Delete texture
        if self.texture is not None:
            glDeleteTextures([self.texture])
            self.texture = None
        
        # Call parent cleanup
        super().cleanup()
