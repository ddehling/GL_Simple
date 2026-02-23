import numpy as np
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
import glfw
import time

# Compute shader with multiple blended functions
COMPUTE_SHADER = """
#version 430 core
layout(local_size_x = 256) in;

// Input: 3D positions
layout(std430, binding = 0) buffer PositionBuffer {
    vec3 positions[];
};

// Output: Final blended RGBA values
layout(std430, binding = 1) buffer ColorBuffer {
    vec4 colors[];
};

uniform float time;  // For animation

// ============================================================================
// Define your multiple compute functions here
// ============================================================================

// Function 1: Near-range effect (red glow)
vec4 computeFunction1(vec3 pos, float t) {
    float dist = length(pos);
    float pulse = sin(t * 2.0) * 0.5 + 0.5;
    float intensity = smoothstep(10.0, 0.0, dist) * pulse;
    
    vec3 color = vec3(1.0, 0.2, 0.2);
    float alpha = intensity * 0.8;
    
    return vec4(color, alpha);
}

// Function 2: Mid-range effect (green bands)
vec4 computeFunction2(vec3 pos, float t) {
    float height = pos.y + sin(t) * 3.0;
    float band = sin(height * 0.5) * 0.5 + 0.5;
    
    vec3 color = vec3(0.2, 1.0, 0.2) * band;
    float alpha = band * 0.6;
    
    return vec4(color, alpha);
}

// Function 3: Far-range effect (blue gradient)
vec4 computeFunction3(vec3 pos, float t) {
    float dist = length(pos);
    float rotation = atan(pos.x, pos.z) + t * 0.5;
    float intensity = smoothstep(5.0, 20.0, dist) * (sin(rotation * 4.0) * 0.5 + 0.5);
    
    vec3 color = vec3(0.2, 0.2, 1.0);
    float alpha = intensity * 0.4;
    
    return vec4(color, alpha);
}

// Function 4: Position-based texture
vec4 computeFunction4(vec3 pos, float t) {
    float r = sin(pos.x * 0.3 + t) * 0.5 + 0.5;
    float g = cos(pos.y * 0.3 + t) * 0.5 + 0.5;
    float b = sin(pos.z * 0.3 + t) * 0.5 + 0.5;
    
    vec3 color = vec3(r, g, b);
    float alpha = 0.5;
    
    return vec4(color, alpha);
}

// ============================================================================
// Alpha blending function
// ============================================================================
vec4 blendOver(vec4 source, vec4 dest) {
    float srcAlpha = source.a;
    float dstAlpha = dest.a * (1.0 - srcAlpha);
    float outAlpha = srcAlpha + dstAlpha;
    
    vec3 outColor;
    if (outAlpha > 0.0001) {
        outColor = (source.rgb * srcAlpha + dest.rgb * dstAlpha) / outAlpha;
    } else {
        outColor = vec3(0.0);
    }
    
    return vec4(outColor, outAlpha);
}

// ============================================================================
// Main compute function
// ============================================================================
void main() {
    uint idx = gl_GlobalInvocationID.x;
    
    if (idx >= positions.length()) return;
    
    vec3 pos = positions[idx];
    
    // Start with transparent black
    vec4 result = vec4(0.0, 0.0, 0.0, 0.0);
    
    // Blend functions in order (back to front)
    vec4 layer1 = computeFunction3(pos, time);
    result = blendOver(layer1, result);
    
    vec4 layer2 = computeFunction4(pos, time);
    result = blendOver(layer2, result);
    
    vec4 layer3 = computeFunction2(pos, time);
    result = blendOver(layer3, result);
    
    vec4 layer4 = computeFunction1(pos, time);
    result = blendOver(layer4, result);
    
    // Store final blended result
    colors[idx] = result;
}
"""

class ComputeShaderBenchmark:
    def __init__(self, num_pixels=200000):
        """
        Initialize compute shader benchmark
        
        Parameters:
        -----------
        num_pixels : int
            Number of pixels to process per frame
        """
        self.num_pixels = num_pixels
        
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(1, 1, "Benchmark", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Window creation failed")
        
        glfw.make_context_current(self.window)
        
        # Compile compute shader
        compute_shader = shaders.compileShader(COMPUTE_SHADER, GL_COMPUTE_SHADER)
        self.program = shaders.compileProgram(compute_shader)
        
        # Generate random 3D positions
        print(f"Generating {num_pixels:,} random 3D positions...")
        np.random.seed(42)
        self.positions_3d = np.random.randn(num_pixels, 3).astype(np.float32) * 8.0
        
        # Create persistent buffers
        self.position_buffer = glGenBuffers(1)
        self.color_buffer = glGenBuffers(1)
        
        # Upload positions (only once)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.position_buffer)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.positions_3d.nbytes, 
                     self.positions_3d, GL_STATIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.position_buffer)
        
        # Allocate output buffer
        self.rgba_values = np.zeros((num_pixels, 4), dtype=np.float32)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.color_buffer)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.rgba_values.nbytes, 
                     None, GL_DYNAMIC_READ)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.color_buffer)
        
        # Calculate work groups
        self.work_groups = (num_pixels + 255) // 256
        
        print(f"✓ Compute shader initialized")
        print(f"  Work groups: {self.work_groups}")
        print(f"  Threads per group: 256")
        print(f"  Total threads: {self.work_groups * 256:,}")
    
    def compute_frame(self, frame_time):
        """Compute one frame"""
        glUseProgram(self.program)
        glUniform1f(glGetUniformLocation(self.program, "time"), frame_time)
        
        # Dispatch compute shader
        glDispatchCompute(self.work_groups, 1, 1)
        
        # Wait for completion with proper synchronization
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        glFinish()  # Force GPU to complete
    
    def compute_frame_with_readback(self, frame_time):
        """Compute one frame and read back results"""
        self.compute_frame(frame_time)
        
        # Read back results - use simpler method
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.color_buffer)
        
        # Map buffer method (faster)
        ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, self.rgba_values.nbytes, 
                               GL_MAP_READ_BIT)
        if ptr:
            import ctypes
            ctypes.memmove(self.rgba_values.ctypes.data, ptr, self.rgba_values.nbytes)
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        else:
            # Fallback
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 
                              self.rgba_values.nbytes, self.rgba_values)
        
        return self.rgba_values
    
    def run_benchmark(self, num_frames=1000, readback_interval=0):
        """
        Run benchmark
        
        Parameters:
        -----------
        num_frames : int
            Number of frames to compute
        readback_interval : int
            If > 0, read back data every N frames (0 = never)
        """
        print("\n" + "=" * 70)
        print(f"BENCHMARK: {num_frames:,} frames × {self.num_pixels:,} pixels")
        print("=" * 70)
        
        frame_times = []
        readback_times = []
        
        print("\nRunning benchmark...")
        total_start = time.time()
        
        for frame in range(num_frames):
            frame_time = frame * 0.016  # Simulate 60 FPS timing
            
            frame_start = time.time()
            
            if readback_interval > 0 and frame % readback_interval == 0:
                # Compute with readback
                self.compute_frame_with_readback(frame_time)
                readback_times.append(time.time() - frame_start)
            else:
                # Compute only
                self.compute_frame(frame_time)
            
            frame_times.append(time.time() - frame_start)
            
            # Progress indicator
            if (frame + 1) % 100 == 0:
                elapsed = time.time() - total_start
                current_fps = (frame + 1) / elapsed
                eta = (num_frames - frame - 1) / current_fps if current_fps > 0 else 0
                print(f"  Frame {frame + 1:4d}/{num_frames} | "
                      f"{current_fps:.1f} FPS | "
                      f"ETA: {eta:.1f}s", end='\r')
        
        total_elapsed = time.time() - total_start
        
        # Calculate statistics
        frame_times = np.array(frame_times) * 1000  # Convert to ms
        
        print("\n\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        
        avg_fps = num_frames / total_elapsed
        
        print(f"\n>>> FRAMES PER SECOND: {avg_fps:.2f} FPS <<<")
        print(f"\nTotal time: {total_elapsed:.2f}s")
        print(f"Total frames: {num_frames:,}")
        
        print(f"\nFrame timing (compute only):")
        print(f"  Average: {frame_times.mean():.3f}ms per frame")
        print(f"  Median:  {np.median(frame_times):.3f}ms")
        print(f"  Min:     {frame_times.min():.3f}ms")
        print(f"  Max:     {frame_times.max():.3f}ms")
        print(f"  Std dev: {frame_times.std():.3f}ms")
        
        if readback_times:
            readback_times = np.array(readback_times) * 1000
            print(f"\nFrame timing (with GPU→CPU readback):")
            print(f"  Average: {readback_times.mean():.3f}ms per frame")
            print(f"  Median:  {np.median(readback_times):.3f}ms")
        
        print(f"\nThroughput:")
        total_pixels = num_frames * self.num_pixels
        pixels_per_sec = total_pixels / total_elapsed
        print(f"  Total pixels processed: {total_pixels:,}")
        print(f"  Pixels per second: {pixels_per_sec:,.0f}")
        print(f"  Megapixels per second: {pixels_per_sec / 1e6:.2f} MP/s")
        print(f"  Gigapixels per second: {pixels_per_sec / 1e9:.3f} GP/s")
        
        print(f"\nPer-frame throughput:")
        pixels_per_frame = self.num_pixels / (frame_times.mean() / 1000)
        print(f"  Pixels per frame: {self.num_pixels:,}")
        print(f"  Processing rate: {pixels_per_frame:,.0f} pixels/sec")
        print(f"  Megapixels/sec: {pixels_per_frame / 1e6:.2f} MP/s")
        
        print("\n" + "=" * 70)
        
        return {
            'total_time': total_elapsed,
            'num_frames': num_frames,
            'fps': num_frames / total_elapsed,
            'frame_times_ms': frame_times,
            'pixels_per_sec': pixels_per_sec,
            'megapixels_per_sec': pixels_per_sec / 1e6
        }
    
    def verify_output(self):
        """Compute one frame and verify output"""
        print("\nVerifying output...")
        rgba = self.compute_frame_with_readback(0.0)
        
        print(f"\nFirst 5 pixels:")
        print("Position (x, y, z) -> RGBA (r, g, b, a)")
        print("-" * 70)
        for i in range(5):
            pos = self.positions_3d[i]
            color = rgba[i]
            print(f"({pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f}) -> "
                  f"({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}, {color[3]:.3f})")
        
        print(f"\nOutput statistics:")
        print(f"  R: [{rgba[:,0].min():.3f}, {rgba[:,0].max():.3f}] "
              f"mean={rgba[:,0].mean():.3f}")
        print(f"  G: [{rgba[:,1].min():.3f}, {rgba[:,1].max():.3f}] "
              f"mean={rgba[:,1].mean():.3f}")
        print(f"  B: [{rgba[:,2].min():.3f}, {rgba[:,2].max():.3f}] "
              f"mean={rgba[:,2].mean():.3f}")
        print(f"  A: [{rgba[:,3].min():.3f}, {rgba[:,3].max():.3f}] "
              f"mean={rgba[:,3].mean():.3f}")
        
        # Check for valid output
        if np.any(np.isnan(rgba)):
            print("  ⚠ WARNING: NaN values detected!")
        if np.all(rgba == 0):
            print("  ⚠ WARNING: All zeros output!")
        else:
            print("  ✓ Output looks valid")
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        glDeleteBuffers(2, [self.position_buffer, self.color_buffer])
        glDeleteProgram(self.program)
        glfw.terminate()


def main():
    # Configuration
    NUM_PIXELS = 2000000
    NUM_FRAMES = 10000
    
    print("=" * 70)
    print("MULTI-LAYER COMPUTE SHADER BENCHMARK")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Pixels per frame: {NUM_PIXELS:,}")
    print(f"  Number of frames: {NUM_FRAMES:,}")
    print(f"  Compute functions: 4 (blended with alpha)")
    print(f"  Total computations: {NUM_PIXELS * NUM_FRAMES * 4:,}")
    
    try:
        # Initialize benchmark
        print("\n[1/5] Initializing OpenGL context...")
        benchmark = ComputeShaderBenchmark(num_pixels=NUM_PIXELS)
        print("      ✓ OpenGL initialized")
        
        # Verify output works
        print("\n[2/5] Verifying compute shader output...")
        benchmark.verify_output()
        print("      ✓ Output verified")
        
        # Run benchmark (compute only, no readback)
        print("\n[3/5] Running compute-only benchmark...")
        print("=" * 70)
        print("Test 1: Compute-only benchmark (no GPU→CPU transfer)")
        print("=" * 70)
        results1 = benchmark.run_benchmark(num_frames=NUM_FRAMES, readback_interval=0)
        print("      ✓ Test 1 complete")
        
        # Run benchmark with periodic readback
        print("\n[4/5] Running compute + readback benchmark...")
        print("=" * 70)
        print("Test 2: Compute + readback every 100 frames")
        print("=" * 70)
        results2 = benchmark.run_benchmark(num_frames=NUM_FRAMES, readback_interval=100)
        print("      ✓ Test 2 complete")
        
        # Comparison
        print("\n[5/5] Generating comparison...")
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)
        print(f"\nCompute-only:")
        print(f"  >>> {results1['fps']:.2f} FPS <<<")
        print(f"  {results1['megapixels_per_sec']:.2f} MP/s")
        
        print(f"\nWith readback (every 100 frames):")
        print(f"  >>> {results2['fps']:.2f} FPS <<<")
        print(f"  {results2['megapixels_per_sec']:.2f} MP/s")
        
        slowdown = (results1['fps'] - results2['fps']) / results1['fps'] * 100
        print(f"\nReadback overhead: {slowdown:.1f}% slower")
        
        # Cleanup
        benchmark.cleanup()
        
        print("\n" + "=" * 70)
        print("✓ Benchmark complete!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nBenchmark failed. Check error above.")
        return


if __name__ == "__main__":
    main()