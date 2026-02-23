"""
Fast pixel extraction using Numba JIT compilation
Releases GIL for parallel processing without blocking render thread
"""
import numpy as np
from numba import njit, prange

@njit(parallel=True, nogil=True, cache=True, fastmath=True)
def extract_and_pack_pixels_unchecked(source, x_coords, y_coords, output_buffer):
    """
    Ultra-fast pixel extraction assuming coordinates are already validated.
    Uses fastmath and skips bounds checking for maximum performance.
    
    Args:
        source: Source image array (height, width, 3) uint8
        x_coords: Pre-validated X coordinates (N,) int32
        y_coords: Pre-validated Y coordinates (N,) int32  
        output_buffer: Output buffer to fill (N*3,) uint8
    """
    pixel_count = x_coords.shape[0]
    
    # Parallel pixel extraction - no bounds checking
    for i in prange(pixel_count):
        x = x_coords[i]
        y = y_coords[i]
        
        # Pack RGB values directly - compiler can vectorize this
        idx = i * 3
        output_buffer[idx] = source[x, y, 0]
        output_buffer[idx + 1] = source[x, y, 1]
        output_buffer[idx + 2] = source[x, y, 2]


@njit(parallel=True, nogil=True, cache=True)
def extract_and_pack_pixels(source, x_coords, y_coords, output_buffer, height, width):
    """
    Extract pixels from source array using coordinate arrays and pack into output buffer.
    Runs in parallel without holding Python GIL.
    
    Args:
        source: Source image array (height, width, 3) uint8
        x_coords: X coordinates to extract (N,) int32
        y_coords: Y coordinates to extract (N,) int32
        output_buffer: Output buffer to fill (N*3,) uint8
        height: Source image height
        width: Source image width
    """
    pixel_count = x_coords.shape[0]
    
    # Parallel pixel extraction and packing
    for i in prange(pixel_count):
        x = min(x_coords[i], height - 1)
        y = min(y_coords[i], width - 1)
        
        # Pack RGB values directly into output buffer
        output_buffer[i * 3] = source[x, y, 0]      # R
        output_buffer[i * 3 + 1] = source[x, y, 1]  # G
        output_buffer[i * 3 + 2] = source[x, y, 2]  # B


@njit(parallel=True, nogil=True, cache=True, fastmath=True)
def process_all_universes(pixel_buffer, starts, ends, output_buffers):
    """
    Process all universes in parallel with optimal memory access.
    
    Args:
        pixel_buffer: Flat pixel data (N*3,) uint8
        starts: Start pixel indices for each universe (M,) int32
        ends: End pixel indices for each universe (M,) int32
        output_buffers: 2D array of universe buffers (M, 510) uint8
    """
    num_universes = starts.shape[0]
    
    for u in prange(num_universes):
        start_byte = starts[u] * 3
        end_byte = ends[u] * 3
        byte_count = end_byte - start_byte
        
        # Copy pixel data
        for i in range(byte_count):
            output_buffers[u, i] = pixel_buffer[start_byte + i]
        
        # Zero padding
        for i in range(byte_count, 510):
            output_buffers[u, i] = 0


@njit(parallel=True, nogil=True, cache=True)
def pack_universes(pixel_buffer, universe_starts, universe_ends, output_buffers):
    """
    Pack pixel data into universe buffers in parallel without GIL.
    
    Args:
        pixel_buffer: Flat pixel data (N*3,) uint8
        universe_starts: Start indices for each universe
        universe_ends: End indices for each universe
        output_buffers: List of 510-byte buffers for each universe
    """
    num_universes = len(universe_starts)
    
    for u in prange(num_universes):
        start = universe_starts[u] * 3
        end = universe_ends[u] * 3
        byte_count = end - start
        
        # Copy pixel data to universe buffer
        for i in range(byte_count):
            output_buffers[u][i] = pixel_buffer[start + i]
        
        # Zero padding
        for i in range(byte_count, 510):
            output_buffers[u][i] = 0


@njit(nogil=True, cache=True)
def extract_pixels_single(source, x_coords, y_coords, output_buffer, height, width):
    """
    Single-threaded version for small arrays (faster for < 1000 pixels).
    Releases GIL so it doesn't block main thread.
    """
    pixel_count = x_coords.shape[0]
    
    for i in range(pixel_count):
        x = min(x_coords[i], height - 1)
        y = min(y_coords[i], width - 1)
        
        output_buffer[i * 3] = source[x, y, 0]
        output_buffer[i * 3 + 1] = source[x, y, 1]
        output_buffer[i * 3 + 2] = source[x, y, 2]
