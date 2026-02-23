import time
import numpy as np
from skimage import color
import random
import cv2
#from pathlib import Path
import numba  # Adding numba for JIT compilation

def forest_scene(instate, outstate):
    if instate['count'] == 0:
        # Initialize forest parameters
        
        # Constants and dimensions
        instate['width'] = 120
        instate['height'] = 60
        
        # Main buffers
        instate['forest_window'] = np.zeros((instate['height'], instate['width'], 4))  # HSVA format
        instate['rgb_out'] = np.zeros((instate['height'], instate['width'], 4), dtype=np.uint8)
        
        # Create image plane
        instate['forest_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((instate['height'], instate['width'], 4), dtype=np.uint8),
            position=(0, 0, 49.5),
            rotation=(0, 0, 0),
            scale=(9, 9)
        )
        
        # Generate ground texture (once during initialization)
        ground_texture = generate_ground_texture(instate['width'], instate['height'])
        instate['ground_texture'] = ground_texture

        # Get current season from outstate (default to 0 if not provided)
        current_season = outstate.get('season', 0.0)
        print("Season = " + str(current_season))
        # Create trees data structure
        instate['trees'] = generate_forest(instate['width'], instate['height'], 
                                           density_factor=0.8, 
                                           season=current_season)
        
        # Pre-compute coordinate grids for faster rendering
        y_grid, x_grid = np.meshgrid(np.arange(instate['height']), np.arange(instate['width']), indexing='ij')
        instate['y_grid'] = y_grid
        instate['x_grid'] = x_grid
        
        # Pre-compute tree masks for faster rendering
        precompute_tree_data(instate)
        
        # Generate a random buffer for tree rendering (reused for stability)
        instate['random_buffer'] = np.random.random((instate['height'], instate['width']))
        
        # Store time for animation
        instate['start_time'] = time.time()
        instate['last_update'] = time.time()
        
        # Cache for frame-to-frame stability (static parts)
        instate['trunks_buffer'] = np.zeros((instate['height'], instate['width'], 4))
        pre_render_tree_trunks(instate['trunks_buffer'], instate['trees'])
        
        # Get total duration from outstate or use default (added for fade effect)
        #instate['duration'] = instate.get('duration', 30.0)
        
        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['forest_plane'])
        return
    
    # Get timing and parameters
    current_time = time.time()
    #dt = current_time - instate['last_update']
    instate['last_update'] = current_time
    
    # Calculate fade factor based on elapsed time (added for fade effect)
    elapsed_time = current_time - instate['start_time']
    total_duration = instate['duration']
    
    # Define fade durations for smooth transitions
    fade_duration = 6.0  # 3 seconds for fade in/out
    
    # Calculate fade factor
    if elapsed_time < fade_duration:
        # Smooth fade in
        fade_factor = elapsed_time / fade_duration
    elif elapsed_time > (total_duration - fade_duration):
        # Smooth fade out
        fade_factor = (total_duration - elapsed_time) / fade_duration
    else:
        # Fully visible during the middle part
        fade_factor = 1.0
    
    # Ensure fade_factor is between 0 and 1
    fade_factor = np.clip(fade_factor, 0, 1)
    
    # Get wind value from outstate
    wind_strength = outstate.get('wind', 0) * 8.0  # Scale for more visible effect
    
    # Start with an empty window
    window = instate['forest_window']
    window.fill(0)  # Clear the window
    
    # Apply the ground texture with fade factor
    ground_texture = instate['ground_texture'].copy()
    ground_texture[..., 3] *= fade_factor  # Apply fade factor to ground alpha
    
    # Copy ground texture to window where alpha > 0
    ground_mask = ground_texture[..., 3] > 0
    window[ground_mask] = ground_texture[ground_mask]
    
    # Apply the pre-rendered tree trunks (static parts) with fade factor
    window[:] = np.where(
        instate['trunks_buffer'][..., 3:4] > 0,
        np.concatenate([instate['trunks_buffer'][..., 0:3], instate['trunks_buffer'][..., 3:4] * fade_factor], axis=-1),
        window
    )
    
    # Render tree segments with wind effect
    render_tree_segments(
        window,
        instate['trees'], 
        wind_strength, 
        current_time, 
        instate['y_grid'], 
        instate['x_grid'],
        instate['random_buffer'],
        fade_factor  # Pass fade_factor to the rendering function
    )
    
    # Convert HSVA to RGBA for rendering using vectorized operations
    rgb = color.hsv2rgb(window[..., 0:3])
    alpha = window[..., 3:4]
    rgb_out = instate['rgb_out']
    rgb_out[..., :3] = rgb * 255
    rgb_out[..., 3:] = alpha * 255
    
    # Update the image plane
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['forest_plane'], 
        rgb_out
    )

def precompute_tree_data(instate):
    """Pre-compute tree data for faster rendering"""
    print("Pre-computing tree data for faster rendering...")
    
    # Get the coordinate grids
    y_grid = instate['y_grid']
    x_grid = instate['x_grid']
    
    # Cache data for all trees
    for tree in instate['trees']:
        # Calculate trunk parameters
        trunk_width = tree['base_width'] * 0.11
        trunk_height = tree['height'] * 0.2
        
        trunk_left = max(0, int(tree['x'] - trunk_width/2))
        trunk_right = min(instate['width']-1, int(tree['x'] + trunk_width/2))
        trunk_top = max(0, int(tree['y'] - trunk_height))
        trunk_bottom = min(instate['height']-1, int(tree['y']))
        
        # Pre-compute trunk mask
        trunk_mask = (
            (y_grid >= trunk_top) & 
            (y_grid <= trunk_bottom) & 
            (x_grid >= trunk_left) & 
            (x_grid <= trunk_right)
        )
        tree['trunk_mask'] = trunk_mask
        
        # Pre-compute segment data
        segment_height = (tree['height'] - trunk_height) / tree['segments']
        tree['segment_height'] = segment_height
        tree['trunk_height'] = trunk_height
        
        # Cache segment width factors
        tree['segment_widths'] = []
        tree['segment_masks'] = []
        
        # Generate segment-specific color variations for gradient effect within tree
        tree['segment_hues'] = []
        for i in range(tree['segments']):
            # Vary hue slightly from top to bottom (more variation at top)
            segment_hue_variation = tree['needle_color_variation'] * (i / tree['segments'])
            segment_hue = tree['needle_hue'] + segment_hue_variation
            tree['segment_hues'].append(segment_hue)
            
            width_factor = (tree['segments'] - i) / tree['segments']
            segment_width = tree['base_width'] * np.power(width_factor, 1.5)
            tree['segment_widths'].append(segment_width)
            
            # Calculate segment position
            segment_y = tree['y'] - trunk_height - i * segment_height
            
            # Pre-compute segment bounds
            center_x = int(tree['x'])
            top_y = int(segment_y - segment_height)
            bottom_y = int(segment_y)
            
            # Store segment data
            tree['segment_masks'].append({
                'center_x': center_x,
                'top_y': top_y,
                'bottom_y': bottom_y,
                'width': segment_width,
                'height': segment_height,
                'factor': (i + 1) / tree['segments']  # For brightness and wind
            })


def pre_render_tree_trunks(buffer, trees):
    """Pre-render just the tree trunks (static parts that don't sway)"""
    # Render each tree trunk
    for tree in trees:
        # Apply the trunk using pre-computed mask
        trunk_mask = tree['trunk_mask']
        buffer[trunk_mask, 0] = tree['trunk_hue']
        buffer[trunk_mask, 1] = tree['trunk_saturation']  # Use tree-specific trunk saturation
        buffer[trunk_mask, 2] = tree['trunk_value']       # Use tree-specific trunk value
        buffer[trunk_mask, 3] = 1.0  # Alpha



def render_tree_segments(window, trees, wind_strength, current_time, y_grid, x_grid, random_buffer, fade_factor):
    """Render tree segments with wind effect and fade factor"""
    # Render each tree's segments
    for tree in trees:
        # Calculate wind effect on this tree
        #wind_time = current_time * 0.5
        wind_effect = wind_strength * tree['sway_amount'] #* np.sin(wind_time + tree['sway_phase'])
        
        # Render each segment with appropriate wind displacement
        for i, segment in enumerate(tree['segment_masks']):
            # Calculate displacement from wind (more at top)
            segment_factor = segment['factor']
            segment_displacement = wind_effect * segment_factor**2
            
            # Draw the segment with wind displacement and fade factor
            draw_pine_segment_numba(
                window,
                segment['center_x'] + segment_displacement,  # Add wind displacement
                segment['top_y'] + segment['height'],  # Bottom Y of segment
                segment['width'],
                segment['height'],
                tree['segment_hues'][i],  # Use segment-specific hue
                segment_factor,  # Brightness factor
                y_grid, x_grid,
                random_buffer,
                fade_factor,  # Pass fade_factor to the drawing function
                tree['needle_saturation'],  # Use tree-specific saturation
                tree['needle_value']        # Use tree-specific value
            )


def generate_ground_texture(width, height):
    """Generate a natural-looking ground texture using vectorized operations"""
    # Ground starts at 3/4 of the screen height
    ground_height = height * 5 // 6 -1
    
    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Create multi-scale noise for ground (vectorized)
    scale_1 = np.random.uniform(-1, 1, (height//8, width//8))
    scale_2 = np.random.uniform(-1, 1, (height//4, width//4))
    scale_3 = np.random.uniform(-1, 1, (height//2, width//2))
    
    # Resize to full dimensions
    scale_1 = cv2.resize(scale_1, (width, height))
    scale_2 = cv2.resize(scale_2, (width, height))
    scale_3 = cv2.resize(scale_3, (width, height))
    
    # Combine scales (vectorized)
    noise = scale_1 * 0.5 + scale_2 * 0.3 + scale_3 * 0.2
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    
    # Create height mask with smooth gradient (vectorized)
    #height_factor = 1.0 / (1 + np.exp((y_coords - ground_height + 5 * noise) / 2))
    
    # Create ground texture array
    texture = np.zeros((height, width, 4))
    
    # Height mask for ground
    ground_mask = y_coords >= (ground_height - 5 * noise)
    
    # Calculate ground factors vectorized
    ground_factor = np.zeros_like(y_coords, dtype=float)
    ground_factor[ground_mask] = (y_coords[ground_mask] - ground_height) / (height - ground_height)
    ground_factor = np.clip(ground_factor, 0, 1)
    
    # Apply base ground texture (vectorized)
    texture[ground_mask, 0] = 0.10 + noise[ground_mask] * 0.05  # Hue (brown)
    texture[ground_mask, 1] = 0.4 + noise[ground_mask] * 0.2    # Saturation
    texture[ground_mask, 2] = 0.3 - ground_factor[ground_mask] * 0.1 + noise[ground_mask] * 0.1  # Value
    texture[ground_mask, 3] = 1.0  # Alpha
    
    # Add green patches (vectorized)
    green_mask = (noise > 0.7) & (y_coords < ground_height + 3) & ground_mask
    texture[green_mask, 0] = 0.3 + noise[green_mask] * 0.05  # Green hue
    texture[green_mask, 1] = 0.5 + noise[green_mask] * 0.2   # Higher saturation for grass
    
    return texture


def generate_forest(width, height, density_factor=1.5, season=0.0):
    """
    Generate a forest of pine trees with seasonal variations
    
    Parameters:
        width, height: dimensions of the forest scene
        density_factor: controls tree density
        season: value from 0-1 representing the seasonal cycle
    """
    trees = []
    ground_height = height * 5 // 6
    
    # Number of trees with 50% increase
    num_trees = int((width // 8) * density_factor)
    
    # Define color palette types for different seasons
    color_palettes = [
        # Spring/Summer Palettes
        # Fresh light greens (spring)
        {"hue_range": (0.25, 0.30), "sat_range": (0.7, 0.9), "val_range": (0.4, 0.6), 
         "trunk_sat_range": (0.5, 0.7), "trunk_val_range": (0.3, 0.5)},
        
        # Standard greens (summer)
        {"hue_range": (0.28, 0.35), "sat_range": (0.75, 0.9), "val_range": (0.25, 0.4), 
         "trunk_sat_range": (0.5, 0.7), "trunk_val_range": (0.3, 0.5)},
        
        # Blue-greens (spruce/fir like)
        {"hue_range": (0.35, 0.43), "sat_range": (0.7, 0.85), "val_range": (0.3, 0.45), 
         "trunk_sat_range": (0.45, 0.65), "trunk_val_range": (0.25, 0.4)},
        
        # Yellow-greens (pine like)
        {"hue_range": (0.22, 0.28), "sat_range": (0.7, 0.9), "val_range": (0.35, 0.5), 
         "trunk_sat_range": (0.55, 0.75), "trunk_val_range": (0.3, 0.45)},
        
        # Darker forest greens
        {"hue_range": (0.30, 0.35), "sat_range": (0.8, 0.95), "val_range": (0.2, 0.3), 
         "trunk_sat_range": (0.5, 0.7), "trunk_val_range": (0.25, 0.4)},
        
        # Fall Palettes
        # Early autumn yellows
        {"hue_range": (0.15, 0.20), "sat_range": (0.8, 0.9), "val_range": (0.45, 0.6), 
         "trunk_sat_range": (0.5, 0.7), "trunk_val_range": (0.3, 0.45)},
        
        # Golden autumn
        {"hue_range": (0.10, 0.15), "sat_range": (0.8, 0.95), "val_range": (0.5, 0.65), 
         "trunk_sat_range": (0.5, 0.7), "trunk_val_range": (0.3, 0.45)},
        
        # Orange autumn
        {"hue_range": (0.05, 0.10), "sat_range": (0.85, 0.95), "val_range": (0.45, 0.6), 
         "trunk_sat_range": (0.5, 0.7), "trunk_val_range": (0.3, 0.45)},
        
        # Red autumn
        {"hue_range": (0.02, 0.07), "sat_range": (0.85, 0.95), "val_range": (0.4, 0.55), 
         "trunk_sat_range": (0.5, 0.7), "trunk_val_range": (0.3, 0.45)},
        
        # Winter Palettes
        # Evergreen winter (slightly blue-tinted)
        {"hue_range": (0.35, 0.45), "sat_range": (0.6, 0.8), "val_range": (0.2, 0.35), 
         "trunk_sat_range": (0.4, 0.6), "trunk_val_range": (0.2, 0.35)},
        
        # Dark winter green
        {"hue_range": (0.33, 0.38), "sat_range": (0.7, 0.85), "val_range": (0.1, 0.25), 
         "trunk_sat_range": (0.4, 0.6), "trunk_val_range": (0.2, 0.3)},
        
        # Snow-dusted evergreen 
        {"hue_range": (0.3, 0.4), "sat_range": (0.05, 0.4), "val_range": (0.3, 0.5), 
         "trunk_sat_range": (0.4, 0.6), "trunk_val_range": (0.2, 0.35)},
    ]
    
    # Calculate seasonal weights based on the season parameter (0-1)
    # Divide the year into 4 seasons
    spring_center = 0.125
    summer_center = 0.375
    fall_center = 0.625
    winter_center = 0.875
    season_width = 0.25  # Width of seasonal influence
    
    # Initialize weights for each palette
    palette_weights = [0] * len(color_palettes)
    

    # Helper function to calculate circular distance in the 0-1 range
    def circular_distance(a, b):
        direct_distance = abs(a - b)
        return min(direct_distance, 1 - direct_distance)

    # Spring weights
    spring_influence = max(0, 1 - circular_distance(season, spring_center) / season_width)
    palette_weights[0] = 30 * spring_influence  # Fresh light greens

    # Summer weights
    summer_influence = max(0, 1 - circular_distance(season, summer_center) / season_width)
    palette_weights[1] = 20 * summer_influence  # Standard greens
    palette_weights[2] = 15 * summer_influence  # Blue-greens
    palette_weights[3] = 15 * summer_influence  # Yellow-greens
    palette_weights[4] = 10 * summer_influence  # Darker forest greens

    # Fall weights
    fall_influence = max(0, 1 - circular_distance(season, fall_center) / season_width)
    palette_weights[5] = 15 * fall_influence  # Early autumn yellows
    palette_weights[6] = 20 * fall_influence  # Golden autumn
    palette_weights[7] = 15 * fall_influence  # Orange autumn
    palette_weights[8] = 10 * fall_influence  # Red autumn

    # Winter weights
    winter_influence = max(0, 1 - circular_distance(season, winter_center) / season_width)
    palette_weights[9] = 20 * winter_influence   # Evergreen winter
    palette_weights[10] = 15 * winter_influence  # Dark winter green
    palette_weights[11] = 100 * winter_influence  # Snow-dusted eve
    
    # Add a small baseline weight to avoid zero probabilities
    palette_weights = [max(1, w) for w in palette_weights]
    
    for _ in range(num_trees):
        # Tree position
        x = random.uniform(0, width)
        y = random.uniform(ground_height - 2, ground_height + 2)
        
        # Tree size (height, width)
        tree_height = random.uniform(20, 35)
        
        # Base width proportional to height (wider)
        base_width = tree_height * random.uniform(0.6, 0.8)
        
        # Tree shape parameters
        num_segments = random.randint(5, 8)
        
        # Choose a color palette based on seasonal weights
        palette = random.choices(color_palettes, weights=palette_weights, k=1)[0]
        
        # Color variations from the chosen palette
        trunk_hue = 0.08 + random.random() * 0.04  # Brown trunks remain similar
        
        # Get needle color from palette
        hue_min, hue_max = palette["hue_range"]
        sat_min, sat_max = palette["sat_range"]
        val_min, val_max = palette["val_range"]
        trunk_sat_min, trunk_sat_max = palette["trunk_sat_range"]
        trunk_val_min, trunk_val_max = palette["trunk_val_range"]
        
        needle_hue = random.uniform(hue_min, hue_max)
        needle_saturation = random.uniform(sat_min, sat_max)
        needle_value = random.uniform(val_min, val_max)
        trunk_saturation = random.uniform(trunk_sat_min, trunk_sat_max)
        trunk_value = random.uniform(trunk_val_min, trunk_val_max)
        
        # Color variation within the tree (gradient from bottom to top)
        needle_color_variation = random.uniform(-0.05, 0.05)
        
        # Swaying parameters
        sway_amount = random.uniform(0.4, 1.2)
        sway_phase = random.uniform(0, 6.28)
        
        trees.append({
            'x': x,
            'y': y,
            'height': tree_height,
            'base_width': base_width,
            'segments': num_segments,
            'trunk_hue': trunk_hue,
            'trunk_saturation': trunk_saturation,
            'trunk_value': trunk_value,
            'needle_hue': needle_hue,
            'needle_saturation': needle_saturation,
            'needle_value': needle_value,
            'needle_color_variation': needle_color_variation,
            'sway_amount': sway_amount,
            'sway_phase': sway_phase
        })
    
    # Sort trees by y coordinate (depth) to render back to front
    trees.sort(key=lambda t: t['y'])
    
    return trees


# Numba JIT-compiled function for drawing tree segments
@numba.njit
def draw_pine_segment_numba(window, x, y, width, height, hue, brightness_factor, y_grid, x_grid, random_buffer, fade_factor, saturation, value):
    """Draw a triangle pine segment using Numba JIT compilation with fade factor and custom color parameters"""
    # Convert to integers
    center_x = int(x)
    top_y = int(y - height)
    bottom_y = int(y)
    
    # Calculate bounds for drawing
    h, w = window.shape[0:2]
    y_min = max(0, top_y)
    y_max = min(h-1, bottom_y)
    
    # Early exit if segment is out of bounds
    if y_min > y_max:
        return
    
    # Loop through rows (Numba likes explicit loops)
    for row in range(y_min, y_max + 1):
        # Calculate row width as a function of height
        row_factor = (row - y_min) / max(1, y_max - y_min)
        row_width = width * (1 - row_factor * 0.2)
        
        # Calculate x bounds for this row
        left_bound = max(0, int(center_x - row_width / 2))
        right_bound = min(w-1, int(center_x + row_width / 2))
        
        # Draw the pixels in this row
        for col in range(left_bound, right_bound + 1):
            # Calculate edge factor for needle density
            edge_factor = 2 * abs(col - center_x) / (row_width + 0.001)
            edge_factor = edge_factor ** 2 * 0.5
            
            # Use pre-computed random values from the buffer
            # This gives consistent tree appearance while still allowing movement
            if random_buffer[row % h, col % w] > edge_factor * 0.8:
                # Use more pre-computed random values by sampling different positions
                value_random = random_buffer[(row + 7) % h, (col + 3) % w] * 0.1
                sat_random = random_buffer[(row + 11) % h, (col + 5) % w] * 0.1
                hue_random = (random_buffer[(row + 13) % h, (col + 7) % w] - 0.5) * 0.05
                
                # Set pixel values with fade factor applied to alpha
                window[row, col, 0] = hue + hue_random  # Hue with slight variation
                window[row, col, 1] = saturation + sat_random  # Custom saturation with slight variation
                window[row, col, 2] = value + brightness_factor * 0.2 + value_random  # Custom value with height-based brightness
                window[row, col, 3] = fade_factor  # Apply fade factor to alpha