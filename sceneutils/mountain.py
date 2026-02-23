import time
import numpy as np
from skimage import color
import random
import numba

def mountain_scene(instate, outstate):
    if instate['count'] == 0:
        # Initialize mountain parameters
        outstate['has_mountain']  = True
        # Constants and dimensions - define these first
        instate['width'] = 120
        instate['height'] = 60
        instate['season'] = outstate['season']  # Store season for updates
        
        # Store season constants
        instate['spring_center'] = 0.125
        instate['summer_center'] = 0.375
        instate['fall_center'] = 0.625
        instate['winter_center'] = 0.875
        instate['season_width'] = 0.25
        
        # Main buffers
        instate['mountain_window'] = np.zeros((instate['height'], instate['width'], 4))  # HSVA format
        instate['rgb_out'] = np.zeros((instate['height'], instate['width'], 4), dtype=np.uint8)
        
        # Create image plane
        instate['mountain_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((instate['height'], instate['width'], 4), dtype=np.uint8),
            position=(0, 0, 49.8),
            rotation=(0, 0, 0),
            scale=(9, 9)
        )
        
        # Generate mountain profiles
        instate['mountain_profiles'] = generate_mountain_profiles(instate['width'], instate['height'])
        
        # Pre-compute coordinate grids for faster rendering
        y_grid, x_grid = np.meshgrid(np.arange(instate['height']), np.arange(instate['width']), indexing='ij')
        instate['y_grid'] = y_grid
        instate['x_grid'] = x_grid
        
        # Generate a random buffer for texture rendering (reused for stability)
        instate['random_buffer'] = np.random.random((instate['height'], instate['width']))
        
        # Add a structure texture for enhanced physical details
        instate['structure_texture'] = generate_structure_texture(instate['width'], instate['height'])
        
        # Calculate snow line parameters
        snow_line_base = int(instate['height'] * 0.45)
        instate['snow_line_base'] = snow_line_base
        
        # Pre-calculate tree line that follows terrain features
        # Base tree line starts below snow line
        tree_line_base = snow_line_base + int(instate['height'] * 0.1) + 3
        instate['tree_line_base'] = tree_line_base
        
        # Create a persistent tree line array that varies across the entire width
        tree_line_array = np.zeros(instate['width'], dtype=np.int32)
        
        # Calculate tree line variations based on structure texture - but with fixed seed
        np.random.seed(42)  # Use fixed seed for consistency
        for x in range(instate['width']):
            # Sample a vertical slice of the structure texture
            structure_slice = instate['structure_texture'][tree_line_base-10:tree_line_base+10, x]
            
            # Calculate average structure value in this area
            local_structure = np.mean(structure_slice)
            
            # Trees grow higher in valleys (low structure) and lower on ridges (high structure)
            structure_factor = (local_structure - 0.5) * 2.0  # -1.0 to 1.0 range
            
            # Calculate tree line variation - make it less dramatic (±5 pixels)
            variation = int(-structure_factor * 5)
            
            # Add some random variation for natural look (but consistent)
            random_variation = np.random.randint(-2, 3)
            
            # Set tree line for this column
            tree_line_array[x] = tree_line_base + variation + random_variation
        
        # Apply small smoothing to avoid extreme jumps in tree line
        from scipy.ndimage import gaussian_filter1d
        tree_line_array = gaussian_filter1d(tree_line_array, sigma=1.5).astype(np.int32)
        
        # Store the tree line for reuse
        instate['tree_line_array'] = tree_line_array
        
        # Reset random seed to avoid affecting other random processes
        np.random.seed(None)
        
        # Store time for animation
        instate['start_time'] = time.time()
        instate['last_update'] = time.time()
        
        # Pre-render the mountain base texture (static parts)
        instate['mountain_base_texture'] = np.zeros((instate['height'], instate['width'], 4))
        pre_render_mountain_texture(instate['mountain_base_texture'], 
                                  instate['mountain_profiles']['profiles'],
                                  instate['mountain_profiles']['hues'],
                                  instate['mountain_profiles']['saturations'],
                                  instate['mountain_profiles']['values'],
                                  instate['random_buffer'],
                                  instate['structure_texture'])
        
        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['mountain_plane'])
        outstate['has_mountain']  = False
        return
   
    # Get timing and parameters
    current_time = time.time()
    # dt = min(current_time - instate['last_update'], 0.1)  # Cap dt to avoid large jumps
    elapsed_time = current_time - instate['start_time']
    total_duration = instate.get('duration', 60.0)
    
    # Calculate fade factor
    fade_duration = 6.0
    if elapsed_time < fade_duration:
        fade_factor = elapsed_time / fade_duration
    elif elapsed_time > (total_duration - fade_duration):
        fade_factor = (total_duration - elapsed_time) / fade_duration
    else:
        fade_factor = 1.0
    fade_factor = np.clip(fade_factor, 0, 1)
    
    # Calculate snow level based on season
    season = outstate['season']
    winter_center = instate['winter_center']
    
    # Calculate seasonal snow amount (0 = no snow, 1 = full snow)
    winter_distance = min(
        abs(season - winter_center),
        abs(season - (winter_center + 1)),
        abs(season - (winter_center - 1))
    )
    
    # Snow is present mostly in winter
    snow_amount = max(0, 1 - (winter_distance * 4))
    
    # Re-render the mountain texture with seasonal snow
    buffer = np.zeros_like(instate['mountain_base_texture'])  
    
    # Get mountain data
    profiles = instate['mountain_profiles']['profiles']
    hues = instate['mountain_profiles']['hues']
    saturations = instate['mountain_profiles']['saturations']
    values = instate['mountain_profiles']['values']
    random_buffer = instate['random_buffer']
    
    height, width = buffer.shape[0:2]
    
    # ELEVATION-BASED SNOW - use cached base snow line
    snow_line_base = instate['snow_line_base']
    winter_snow_extension = int(height * 0.1 * snow_amount)
    snow_line = snow_line_base + winter_snow_extension
    
    # Use the precalculated tree line array
    tree_line_array = instate['tree_line_array']
    
    # Calculate season-based colors using the same logic
    spring_center = instate['spring_center']
    # summer_center = instate['summer_center']
    fall_center = instate['fall_center']
    
    spring_distance = min(
        abs(season - spring_center),
        abs(season - (spring_center + 1)),
        abs(season - (spring_center - 1))
    )
    
    # summer_distance = min(
    #     abs(season - summer_center),
    #     abs(season - (summer_center + 1)),
    #     abs(season - (summer_center - 1))
    # )
    
    fall_distance = min(
        abs(season - fall_center),
        abs(season - (fall_center + 1)),
        abs(season - (fall_center - 1))
    )
    
    # Tree coloration parameters based on season
    tree_hue_base = 0.3
    tree_sat_base = 0.7
    tree_val_base = 0.5
    
    spring_factor = max(0, 1 - (spring_distance * 4))
    if spring_factor > 0:
        tree_hue_base = tree_hue_base * (1 - spring_factor) + 0.25 * spring_factor
        tree_sat_base = tree_sat_base * (1 - spring_factor) + 0.8 * spring_factor
        tree_val_base = tree_val_base * (1 - spring_factor) + 0.7 * spring_factor
    
    fall_factor = max(0, 1 - (fall_distance * 4))
    if fall_factor > 0:
        tree_hue_base = tree_hue_base * (1 - fall_factor) + 0.1 * fall_factor
        tree_sat_base = tree_sat_base * (1 - fall_factor) + 0.8 * fall_factor
        tree_val_base = tree_val_base * (1 - fall_factor) + 0.6 * fall_factor
    
    if snow_amount > 0:
        tree_sat_base = tree_sat_base * (1 - snow_amount * 0.5)
        tree_val_base = tree_val_base * (1 - snow_amount * 0.3)
    
    # A simpler but still vectorized approach
    # Create a base mountain texture first using the original function
    pre_render_mountain_texture(buffer, profiles, hues, saturations, values, 
                               random_buffer, instate['structure_texture'])
    
    # Now apply seasonal changes using vectorized operations
    # We'll work with the entire buffer at once where possible
    
    # Create the tree mask using the variable tree line
    # Create tree mask using vectorized operations
    tree_mask = np.zeros((height, width), dtype=bool)
    
    # Create indices arrays once
    x_indices = np.arange(width)
    y_indices = np.arange(height)
    Y, X = np.meshgrid(y_indices, x_indices, indexing='ij')
    
    # Create the condition array for tree mask
    mask_condition = Y >= tree_line_array[X]
    # Apply the alpha condition as well
    tree_mask = mask_condition & (buffer[Y, X, 3] > 0)
    
    if np.any(tree_mask):
        # Get coordinates of all masked pixels
        mask_y, mask_x = np.where(tree_mask)
        
        # Calculate distances in a vectorized way
        tree_distance = (mask_y - tree_line_array[mask_x]) / (height * 0.05)
        
        # Apply minimum and calculate tree factor
        tree_factor = np.minimum(1.0, tree_distance * 1.5 + 0.2)
        
        # Create transition area around tree line - vectorized
        transition_zone = np.zeros((height, width), dtype=bool)
        
        # For each x, mark pixels in the 3-pixel zone above tree line
        for i in range(3):
            # Create mask where Y is at position i above tree line
            transition_level_mask = Y == (tree_line_array[X] + i)
            # Only include pixels with content
            transition_zone |= transition_level_mask & (buffer[Y, X, 3] > 0)
        
        # Alternative fully vectorized approach for transition zone:
        # transition_zone = (Y >= tree_line_array[X]) & (Y < tree_line_array[X] + 3) & (buffer[Y, X, 3] > 0)
        
        transition_mask = transition_zone
        
        if np.any(transition_mask):
            # Calculate transition factors
            # Find intersection of transition and tree masks
            transition_indices = np.where(transition_mask.flatten())[0]
            tree_mask_indices = np.where(tree_mask.flatten())[0]
            
            # Find common indices (vectorized intersection)
            common_indices = np.intersect1d(transition_indices, tree_mask_indices)
            
            if len(common_indices) > 0:
                # Convert flat indices back to 2D coordinates
                common_y, common_x = np.unravel_index(common_indices, (height, width))
                
                # Get corresponding tree lines for these positions
                common_tree_lines = tree_line_array[common_x]
                
                # Calculate relative positions in transition zone
                rel_pos = (common_y - common_tree_lines) / 3
                
                # Calculate transition tree factors
                tree_factor_transition = np.minimum(1.0, 
                                                ((common_y - common_tree_lines) / (height * 0.05)) * 1.5 + 0.2)
                tree_factor_transition *= rel_pos
                
                # Find matching positions in the tree_factor array
                # This maps common indices back to positions in the tree mask
                for i, idx in enumerate(common_indices):
                    # Find position in tree_mask_indices array
                    tree_idx = np.where(tree_mask_indices == idx)[0][0]
                    tree_factor[tree_idx] = tree_factor_transition[i]
                
                # Alternative approach using vectorized operations:
                # tree_idx_mapping = np.searchsorted(tree_mask_indices, common_indices)
                # tree_factor[tree_idx_mapping] = tree_factor_transition
        
        # Extract coordinates for tree pixels
        tree_y, tree_x = np.where(tree_mask)
        
        # Apply noise to tree density - increase variability for more distinct trees
        tree_noise = random_buffer[(tree_y * 5) % height, (tree_x * 11) % width] * 0.4  # Increase from 0.3 to 0.4
        
        # Use structure texture to affect tree density - but less dramatically
        structure_influence = instate['structure_texture'][tree_y, tree_x]
        # Make the influence less extreme - linear falloff instead of quadratic
        structure_tree_factor = 1.0 - structure_influence * 0.6  # Less reduction in tree density
        
        # Apply stronger base tree density and structure influence
        tree_density = np.minimum(1.0, (tree_factor + tree_noise - 0.05) * structure_tree_factor)  # Reduce threshold from 0.1 to 0.05
        
        # Only apply to visible trees - lower threshold for visibility
        visible_trees = tree_density > 0.05  # Reduce threshold from 0.1 to 0.05
        if np.any(visible_trees):
            # Calculate tree colors with more contrast
            tree_hue = np.full_like(tree_density, tree_hue_base)
            tree_hue += tree_noise * 0.15  # Increase variation
            
            tree_sat = np.full_like(tree_density, tree_sat_base)
            tree_sat *= (0.8 + tree_noise * 0.5)  # Increase variation
            
            tree_val = np.full_like(tree_density, tree_val_base)
            tree_val *= (0.8 + tree_noise * 0.5)  # Increase variation
            
            # Enhance the contrast between trees and background
            # Trees in valleys are darker, trees on ridges get more light
            structure_shade = (structure_influence - 0.5) * 0.4  # -0.2 to +0.2 range
            tree_val += structure_shade
            
            # Make tree color more distinct from background
            tree_sat += 0.2  # Increase saturation for more vivid trees

            # Calculate blend factor - increase influence of tree color
            blend_factor = tree_density * np.minimum(1.0, tree_distance * 1.2 + 0.3)  # Add 0.3 baseline
            
            # Prepare arrays for color blending 
            hue_tree = buffer[tree_mask, 0].copy()
            sat_tree = buffer[tree_mask, 1].copy()
            val_tree = buffer[tree_mask, 2].copy()
            
            # Blend colors with greater tree influence
            hue_tree[visible_trees] = hue_tree[visible_trees] * (1 - blend_factor[visible_trees]) + tree_hue[visible_trees] * blend_factor[visible_trees]
            sat_tree[visible_trees] = sat_tree[visible_trees] * (1 - blend_factor[visible_trees]) + tree_sat[visible_trees] * blend_factor[visible_trees]
            val_tree[visible_trees] = val_tree[visible_trees] * (1 - blend_factor[visible_trees]) + tree_val[visible_trees] * blend_factor[visible_trees]
            
            # Update buffer
            buffer[tree_mask, 0] = hue_tree
            buffer[tree_mask, 1] = sat_tree
            buffer[tree_mask, 2] = val_tree
    
    # --- Apply snow effects ---
    # Create a mask for snow areas (above snow line and has mountain content)
    y_coords, x_coords = np.indices((height, width))
    snow_mask = (y_coords < snow_line) & (buffer[:, :, 3] > 0)
    
    if np.any(snow_mask):
        # Create a transition zone mask
        transition_mask = (y_coords > (snow_line - 5)) & (y_coords < snow_line) & (buffer[:, :, 3] > 0)
        
        # Calculate snow factor - default to 1.0 (full snow)
        snow_factor = np.ones(snow_mask.shape)
        
        # Apply transition at snow line
        if np.any(transition_mask):
            rel_pos = ((snow_line - y_coords[transition_mask]) / 5)
            snow_factor[transition_mask] = rel_pos
        
        # Apply noise to snow line
        edge_noise = random_buffer[(y_coords * 3) % height, (x_coords * 7) % width] * 0.4
        if np.any(transition_mask):
            snow_factor[transition_mask] *= (1.0 - edge_noise[transition_mask])
        
        # Prepare arrays for color blending (only for snow pixels)
        hue_snow = buffer[snow_mask, 0].copy()
        sat_snow = buffer[snow_mask, 1].copy()
        val_snow = buffer[snow_mask, 2].copy()
        
        # Use structure texture to affect snow accumulation
        structure_snow = instate['structure_texture'][snow_mask]
        snow_factor_masked = snow_factor[snow_mask]
        
        # Less snow on steep areas (high structure values)
        snow_factor_masked *= (1.0 - structure_snow * 0.7)
        
        # Visible snow has a factor > 0.1
        visible_snow = snow_factor_masked > 0.1
        
        # Blend with snow color
        if np.any(visible_snow):
            hue_snow[visible_snow] = hue_snow[visible_snow] * (1 - snow_factor_masked[visible_snow]) + 0.05 * snow_factor_masked[visible_snow]
            sat_snow[visible_snow] = sat_snow[visible_snow] * (1 - snow_factor_masked[visible_snow]) + 0.05 * snow_factor_masked[visible_snow]
            val_snow[visible_snow] = val_snow[visible_snow] * (1 - snow_factor_masked[visible_snow]) + 0.95 * snow_factor_masked[visible_snow]
            
            # Structure affects snow brightness - shadows in crevices
            snow_shadow = (structure_snow[visible_snow] - 0.5) * 0.2
            val_snow[visible_snow] -= snow_shadow * snow_factor_masked[visible_snow]
        
        # Update buffer
        buffer[snow_mask, 0] = hue_snow
        buffer[snow_mask, 1] = sat_snow
        buffer[snow_mask, 2] = val_snow
    
    # Clamp values to valid range
    buffer[:, :, 0:3] = np.clip(buffer[:, :, 0:3], 0, 1)
    
    # Copy result to window buffer
    instate['mountain_window'][:] = buffer
    
    # Convert HSVA to RGBA for rendering using vectorized operations
    rgb = color.hsv2rgb(instate['mountain_window'][..., 0:3])
    alpha = instate['mountain_window'][..., 3:4]
    rgb_out = instate['rgb_out']
    rgb_out[..., :3] = rgb * 255
    rgb_out[..., 3:] = alpha * 255*fade_factor
    
    # Update the image plane
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['mountain_plane'], 
        rgb_out
    )
    
def generate_mountain_profiles(width, height):
    """Generate a single mountain range with distinct, wide but sharp peaks and low ground"""
    # Base height level (where mountains start) - moved much lower
    base_height = height * 7 // 8  # Lower ground level
    
    # We'll just create one mountain range with 2-3 distinct peaks
    num_ranges = 1
    
    # Create arrays for mountain data (Numba-compatible)
    profiles = np.zeros((num_ranges, width))
    hues = np.zeros(num_ranges)
    saturations = np.zeros(num_ranges)
    values = np.zeros(num_ranges)
    depths = np.zeros(num_ranges)
    
    # Mountain range characteristics
    range_height = 0.8 * height  # Maximum height of the range
    range_base = base_height
    
    # Start with a flat baseline
    profile = np.ones(width) * range_base
    
    # Create 2-3 distinct peaks
    num_peaks = 3
    
    # Distribute peak positions across the width
    peak_positions = []
    for i in range(num_peaks):
        # Divide the width into sections, place peaks in each section
        section_width = width / num_peaks
        min_pos = section_width * i + section_width * 0.15  # Keep away from edges
        max_pos = section_width * (i + 1) - section_width * 0.15
        peak_pos = random.uniform(min_pos, max_pos)
        peak_positions.append(int(peak_pos))
    
    # Create each peak - wider but still sharp
    for peak_pos in peak_positions:
        # Random peak height
        peak_height = random.uniform(range_height * 0.6, range_height * 0.95)
        
        # Random peak width - wider now (20-30% of width)
        peak_width = random.uniform(width * 0.15, width * 0.25)
        
        # Generate peak shape
        for x in range(width):
            # Distance from peak center
            dist = abs(x - peak_pos) / peak_width
            
            if dist < 1.0:
                # Calculate peak height at this point
                # Power function with exponent < 1 creates sharper peaks
                peak_factor = (1.0 - dist ** 0.7)  # Keep the sharp peaks
                
                # Add some small random variations for a more natural look
                jitter = random.uniform(-0.05, 0.05) * peak_height
                
                # Calculate new height at this point
                new_height = range_base - (peak_height * peak_factor + jitter)
                
                # Keep the lowest point (highest peak)
                profile[x] = min(profile[x], new_height)
    
    # Add subtle variations to the mountain silhouette
    # This adds small details without creating noise
    for x in range(1, width-1):
        # Small random adjustments
        if random.random() < 0.3:  # Only adjust some points
            adj = random.uniform(-2, 2)
            # Limit the adjustment to keep the overall shape
            profile[x] += adj * 0.5
    
    # Color information for the range
    hue_base = random.uniform(0.05, 0.12)  # Brownish to grayish
    saturation_base = random.uniform(0.3, 0.5)
    value_base = random.uniform(0.4, 0.7)
    
    # Store profile data
    profiles[0] = profile
    hues[0] = hue_base
    saturations[0] = saturation_base
    values[0] = value_base
    depths[0] = 0.0  # Single layer, so depth is 0
    
    return {
        'profiles': profiles,
        'hues': hues,
        'saturations': saturations,
        'values': values,
        'num_ranges': num_ranges
    }

@numba.njit
def pre_render_mountain_texture(buffer, profiles, hues, saturations, values, random_buffer, structure_texture):
    """Pre-render the mountain texture with rock features using Numba-compatible arrays"""
    height, width = buffer.shape[0:2]
    num_ranges = profiles.shape[0]
    
    # Clear buffer
    buffer.fill(0)
    
    # Create mountain shapes by iterating through profiles (back to front)
    for p_idx in range(num_ranges):
        profile = profiles[p_idx]
        hue_base = hues[p_idx]
        sat_base = saturations[p_idx]
        val_base = values[p_idx]
        
        # Iterate through each column
        for x in range(width):
            mountain_height = profile[x]
            
            # Find all pixels in this column that should be mountain
            for y in range(height):
                # Only draw if below the profile line and above any previously drawn mountains
                if y >= mountain_height and buffer[y, x, 3] == 0:
                    # Calculate factors for this pixel
                    height_ratio = (y - mountain_height) / (height - mountain_height + 1e-5)
                    
                    # Base color variations
                    noise_val = random_buffer[y % height, x % width] * 0.1
                    
                    # Height-based darkening - gentler gradient
                    darkness = height_ratio * 0.3
                    
                    # Get structure value for this pixel and use it to create stronger lighting effects
                    structure_val = structure_texture[y, x]
                    # More dramatic effect: -0.4 to +0.4 range
                    structure_effect = (structure_val - 0.5) * 0.8
                    
                    # Rock texture effect using random buffer
                    rock_noise = random_buffer[(y + 7) % height, (x + 13) % width] * 0.1
                    
                    # Create horizontal strata effect (rock layers)
                    strata_y = y % 10
                    strata_effect = 0.0
                    if strata_y < 2:
                        strata_effect = 0.05  # Slightly lighter bands
                    
                    # Adjust the hue slightly based on structure - different rock composition
                    structure_hue_shift = (structure_val - 0.5) * 0.05  # Small hue shift
                    
                    # Set color values with variation and stronger structure influence
                    hue = hue_base + rock_noise * 0.1 + structure_hue_shift
                    
                    # Structure affects saturation - more saturated in valleys, less on ridges
                    sat = sat_base - darkness * 0.3 + noise_val - structure_effect * 0.3
                    
                    # Structure has stronger effect on value (brightness)
                    # Ridges catch light (brighter), valleys are shadowed (darker)
                    val = val_base - darkness + rock_noise + strata_effect + structure_effect
                    
                    # Base snow threshold - will be modified in the non-Numba function
                    snow_threshold = 0.15  # Default snow level
                    
                    # Apply elevation-based snow effect for high areas
                    if height_ratio < snow_threshold:
                        snow_amount = 1.0 - (height_ratio / snow_threshold)
                        # Structure strongly affects snow depth - much less snow on ridges/steep areas
                        snow_amount *= (1.0 - structure_val * 0.7)
                        # Blend with snow color
                        hue = hue * (1 - snow_amount) + 0.05 * snow_amount  # Slight blue tint for snow
                        sat = sat * (1 - snow_amount) + 0.1 * snow_amount    # Low saturation for snow
                        val = val * (1 - snow_amount) + 0.9 * snow_amount    # High value for snow
                        
                        # Snow shadows in crevices - structure affects snow brightness
                        snow_shadow = structure_effect * 0.2 * snow_amount
                        val = val - snow_shadow
                    
                    # Ensure values are in valid range
                    hue = min(max(hue, 0), 1)
                    sat = min(max(sat, 0), 1)
                    val = min(max(val, 0), 1)
                    
                    # Set the pixel in the buffer
                    buffer[y, x, 0] = hue
                    buffer[y, x, 1] = sat
                    buffer[y, x, 2] = val
                    buffer[y, x, 3] = 1.0  # Full alpha

def generate_structure_texture(width, height):
    """Generate a texture representing the physical structure of mountains with pronounced features"""
    # Create a multi-scale noise texture for physical features
    structure = np.zeros((height, width))
    
    # Use multiple noise scales and techniques for more realistic mountain structure
    from scipy.ndimage import gaussian_filter
    
    # Base large-scale ridge patterns
    x_coords = np.linspace(0, 5, width)
    y_coords = np.linspace(0, 5, height)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Create primary ridge-lines that run diagonally
    ridge_pattern = np.sin(X*1.5 + Y*1.5) * 0.5 + 0.5
    # Add some variation with a second set of ridges
    ridge_pattern += np.sin(X*2.5 - Y*0.8) * 0.3 + 0.3
    
    # Create detailed noise for rock texture
    noise_small = np.random.random((height, width)) * 0.2
    noise_medium = gaussian_filter(np.random.random((height//2, width//2)), sigma=1.0)
    noise_medium = np.repeat(np.repeat(noise_medium, 2, axis=0), 2, axis=1)
    noise_medium = noise_medium[:height, :width] * 0.3
    
    # Create erosion patterns - streaks running down the mountain
    erosion = np.zeros((height, width))
    for i in range(width//10):
        # Start points along width
        x_start = np.random.randint(0, width)
        # Create a wavy line down the height
        x_pos = x_start
        strength = np.random.random() * 0.5 + 0.5  # Erosion strength
        width_factor = np.random.randint(3, 8)  # Width of erosion line
        
        for y in range(height):
            # Vary position slightly as we move down
            x_pos += np.random.randint(-1, 2)
            x_pos = x_pos % width  # Wrap around width
            
            # Draw erosion line with falloff from center
            for x_offset in range(-width_factor, width_factor+1):
                x = (x_pos + x_offset) % width
                # Falloff based on distance from center of line
                falloff = 1.0 - abs(x_offset)/width_factor
                erosion[y, x] = max(erosion[y, x], strength * falloff)
    
    # Combine features
    structure = ridge_pattern * 0.6 + noise_medium + noise_small
    
    # Add strong vertical bias for sharper ridges and gullies
    for x in range(width):
        for y in range(1, height):
            # Propagate structure downward with slight modification
            if y < height-1:
                # Each row is influenced by the row above
                downward_flow = structure[y-1, x] * 0.2
                structure[y, x] = structure[y, x] * 0.8 + downward_flow
    
    # Add erosion patterns
    structure = structure * (1.0 - erosion * 0.5) + erosion * 0.5
    
    # Create sharper contrasts for more defined features
    structure = np.power(structure, 1.5)  # Increase contrast
    
    # Normalize to 0-1 range
    structure = (structure - structure.min()) / (structure.max() - structure.min())
    
    # Add final detail - small rock texture
    rock_detail = np.random.random((height, width)) * 0.15
    structure = structure * 0.85 + rock_detail
    
    # Re-normalize
    structure = (structure - structure.min()) / (structure.max() - structure.min())
    
    return structure