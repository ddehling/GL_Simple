import time
import numpy as np
from skimage import color
from pathlib import Path
ParentPath = Path(__file__).parent.parent
media_path = ParentPath / 'media'
sound_path = media_path / 'sounds'


def falling_leaves(instate, outstate):
    if instate['count'] == 0:
        outstate['has_leaves'] = True
        # Initialize constants in instate
        instate['MAX_LEAVES'] = 25
        instate['BATCH_SIZE'] = 25
        
        # Initialize separate arrays for leaf properties
        instate.update({
            'leaves_window': np.zeros((60, 120, 4)),  # HSVA format
            'leaf_x': np.zeros(instate['MAX_LEAVES'], dtype=np.float32),
            'leaf_y': np.zeros(instate['MAX_LEAVES'], dtype=np.float32),
            'leaf_vx': np.zeros(instate['MAX_LEAVES'], dtype=np.float32),
            'leaf_vy': np.zeros(instate['MAX_LEAVES'], dtype=np.float32),
            'leaf_size': np.zeros(instate['MAX_LEAVES'], dtype=np.float32),
            'leaf_rotation': np.zeros(instate['MAX_LEAVES'], dtype=np.float32),
            'leaf_rotation_speed': np.zeros(instate['MAX_LEAVES'], dtype=np.float32),
            'leaf_flutter_phase': np.zeros(instate['MAX_LEAVES'], dtype=np.float32),
            'leaf_flutter_amplitude': np.zeros(instate['MAX_LEAVES'], dtype=np.float32),
            'leaf_color': np.zeros((instate['MAX_LEAVES'], 3), dtype=np.float32),  # HSV
            'leaf_alpha': np.zeros(instate['MAX_LEAVES'], dtype=np.float32),
            'leaf_count': 0,
            'last_update': time.time(),
            'start_time': time.time(),
            'rgb_out': np.zeros((60, 120, 4), dtype=np.uint8)
        })
        
        # Create image plane
        instate['leaves_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((60, 120, 4), dtype=np.uint8),
            position=(0, 0, 9.5),
            rotation=(0, 0, 0),
            scale=(1, 1)
        )
        
        # Pre-compute sin/cos for common angles to speed up rotation calculations
        angles = np.linspace(0, 2*np.pi, 100, endpoint=False)
        instate['cos_cache'] = np.cos(angles)
        instate['sin_cache'] = np.sin(angles)
        instate['angle_step'] = 2*np.pi / 100
        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['leaves_plane'])
        outstate['has_leaves'] = False
        return

    # Get current state
    current_time = time.time()
    dt = current_time - instate['last_update']
    elapsed_time = current_time - instate['start_time']
    instate['last_update'] = current_time
    
    # Calculate fade factor based on duration
    total_duration = instate.get('duration', 45.0)
    fade_in_duration = 5.0
    fade_out_start = total_duration - 5.0
    
    if elapsed_time < fade_in_duration:
        global_alpha = elapsed_time / fade_in_duration
    elif elapsed_time > fade_out_start:
        global_alpha = (total_duration - elapsed_time) / 5.0
    else:
        global_alpha = 1.0
    global_alpha = np.clip(global_alpha, 0, 1)

    # Get environment parameters
    wind = outstate.get('wind', 0)
    whomp = outstate.get('whomp', 0.0)
    season = outstate.get('season', 0.625)  # Default to fall
    
    # Adjust leaf generation rate based on wind and whomp
    leaf_rate = 0.25 + 0.2 * abs(wind) + 0.3 * whomp
    
    # Clear the window efficiently
    window = instate['leaves_window']
    window.fill(0)

    fall_distance = min(abs(season - 0.625), 1 - abs(season - 0.625))
    fall_factor = 1 - 1.9*fall_distance

    # Generate new leaves
    if np.random.random() < leaf_rate * global_alpha*fall_factor:
        available_slots = instate['MAX_LEAVES'] - instate['leaf_count']
        new_count = min(np.random.randint(1, 3), available_slots)
        
        if new_count > 0:
            start_idx = instate['leaf_count']
            end_idx = start_idx + new_count
            
            # Initialize new leaves
            instate['leaf_x'][start_idx:end_idx] = np.random.uniform(0, 120, new_count)
            instate['leaf_y'][start_idx:end_idx] = np.random.uniform(-5, 0, new_count)  # Start above screen
            instate['leaf_vx'][start_idx:end_idx] = np.random.uniform(-0.5, 0.5, new_count)
            instate['leaf_vy'][start_idx:end_idx] = np.random.uniform(1.0, 2.0, new_count)
            instate['leaf_size'][start_idx:end_idx] = np.random.uniform(2.0, 3.5, new_count)
            instate['leaf_rotation'][start_idx:end_idx] = np.random.uniform(0, 2*np.pi, new_count)
            instate['leaf_rotation_speed'][start_idx:end_idx] = np.random.uniform(-1.0, 1.0, new_count)
            instate['leaf_flutter_phase'][start_idx:end_idx] = np.random.uniform(0, 2*np.pi, new_count)
            instate['leaf_flutter_amplitude'][start_idx:end_idx] = np.random.uniform(0.5, 1.2, new_count)
            instate['leaf_alpha'][start_idx:end_idx] = np.random.uniform(0.9, 1, new_count)
            
            # Set colors based on season
            colors = np.zeros((new_count, 3), dtype=np.float32)
            
            # Calculate distance from spring center
            spring_distance = min(abs(season - 0.125), 1 - abs(season - 0.125))
            fall_distance = min(abs(season - 0.625), 1 - abs(season - 0.625))
            
            # Spring factor (1.0 at spring, decreases as we move away)
            spring_factor = max(0, 1 - spring_distance * 4)  # Sharp falloff
            
            # Fall factor (1.0 at fall, decreases as we move away)
            fall_factor = max(0, 1 - fall_distance * 4)  # Sharp falloff
            
            # If we're in spring, all leaves are green
            if spring_factor > 0.5:
                # All green leaves in spring
                colors[:, 0] = np.random.uniform(0.25, 0.35, new_count)  # Green hue
                colors[:, 1] = np.random.uniform(0.7, 0.9, new_count)    # High saturation
                colors[:, 2] = np.random.uniform(0.3, 0.5, new_count)    # Medium value
            else:
                # Otherwise use seasonal mix with fall colors where appropriate
                # Vectorized color generation
                color_types = np.random.random(new_count)
                
                # Red leaves (proportion increases during fall)
                red_proportion = 0.1 + 0.3 * fall_factor
                red_mask = color_types < red_proportion
                colors[red_mask, 0] = np.random.uniform(0.00, 0.05, np.sum(red_mask))
                colors[red_mask, 1] = np.random.uniform(0.8, 0.95, np.sum(red_mask))
                colors[red_mask, 2] = np.random.uniform(0.4, 0.6, np.sum(red_mask))
                
                # Orange leaves (proportion increases during fall)
                orange_proportion = red_proportion + (0.1 + 0.2 * fall_factor)
                orange_mask = (color_types >= red_proportion) & (color_types < orange_proportion)
                colors[orange_mask, 0] = np.random.uniform(0.05, 0.10, np.sum(orange_mask))
                colors[orange_mask, 1] = np.random.uniform(0.85, 0.95, np.sum(orange_mask))
                colors[orange_mask, 2] = np.random.uniform(0.45, 0.65, np.sum(orange_mask))
                
                # Yellow leaves (consistent, but more vibrant in fall)
                yellow_proportion = orange_proportion + (0.2 + 0.1 * fall_factor)
                yellow_mask = (color_types >= orange_proportion) & (color_types < yellow_proportion)
                colors[yellow_mask, 0] = np.random.uniform(0.10, 0.15, np.sum(yellow_mask))
                colors[yellow_mask, 1] = np.random.uniform(0.8, 0.9, np.sum(yellow_mask))
                colors[yellow_mask, 2] = np.random.uniform(0.5, 0.7, np.sum(yellow_mask))
                
                # Brown leaves (proportion increases during fall)
                brown_proportion = yellow_proportion + (0.05 + 0.15 * fall_factor)
                brown_mask = (color_types >= yellow_proportion) & (color_types < brown_proportion)
                colors[brown_mask, 0] = np.random.uniform(0.07, 0.12, np.sum(brown_mask))
                colors[brown_mask, 1] = np.random.uniform(0.6, 0.8, np.sum(brown_mask))
                colors[brown_mask, 2] = np.random.uniform(0.3, 0.4, np.sum(brown_mask))
                
                # Green leaves (remaining leaves)
                green_mask = color_types >= brown_proportion
                colors[green_mask, 0] = np.random.uniform(0.25, 0.35, np.sum(green_mask))
                colors[green_mask, 1] = np.random.uniform(0.7, 0.9, np.sum(green_mask))
                colors[green_mask, 2] = np.random.uniform(0.3, 0.5, np.sum(green_mask))
            
            instate['leaf_color'][start_idx:end_idx] = colors

            instate['leaf_count'] += new_count

    # Process active leaves
    if instate['leaf_count'] > 0:
        # Update leaf positions and properties
        active_slice = slice(0, instate['leaf_count'])
        
        # Update flutter phase
        instate['leaf_flutter_phase'][active_slice] += dt * 3.0
        
        # Calculate flutter effect
        flutter_x = np.sin(instate['leaf_flutter_phase'][active_slice]) * instate['leaf_flutter_amplitude'][active_slice]
        
        # Apply wind and flutter to velocity
        instate['leaf_vx'][active_slice] = flutter_x + wind * 5
        
        # Update positions
        instate['leaf_x'][active_slice] += instate['leaf_vx'][active_slice] * dt * 5
        instate['leaf_y'][active_slice] += instate['leaf_vy'][active_slice] * dt * 5 * (1- whomp*1.5)
        
        # Update rotation
        instate['leaf_rotation'][active_slice] += instate['leaf_rotation_speed'][active_slice] * dt * 2
        
        # Filter out-of-bounds leaves
        valid_mask = (
            (instate['leaf_y'][active_slice] < 65) & 
            (instate['leaf_x'][active_slice] > -5) & 
            (instate['leaf_x'][active_slice] < 125)
        )
        
        if not np.all(valid_mask):
            # Compact arrays efficiently - use numpy's boolean indexing to do this in one step
            valid_indices = np.where(valid_mask)[0]
            valid_count = len(valid_indices)
            
            # Optimize by doing a single copy operation for each array
            for arr_name in ['leaf_x', 'leaf_y', 'leaf_vx', 'leaf_vy', 'leaf_size', 
                            'leaf_rotation', 'leaf_rotation_speed', 'leaf_flutter_phase',
                            'leaf_flutter_amplitude', 'leaf_alpha']:
                instate[arr_name][:valid_count] = instate[arr_name][active_slice][valid_indices]
            
            # Handle color array separately since it's 2D
            instate['leaf_color'][:valid_count] = instate['leaf_color'][active_slice][valid_indices]
            
            instate['leaf_count'] = valid_count
        
        # Render leaves efficiently while preserving the original look
        render_leaves_optimized(instate, global_alpha, window)

    # Convert to RGB efficiently 
    rgb = color.hsv2rgb(window[..., 0:3])
    alpha = window[..., 3:4]
    rgb_out = instate['rgb_out']
    rgb_out[..., :3] = (rgb * 255).astype(np.uint8)
    rgb_out[..., 3:] = (alpha * 255).astype(np.uint8)
    
    # Update texture
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['leaves_plane'],
        rgb_out
    )

def get_cached_sin_cos(instate, angle):
    """Get cached sin and cos values for an angle"""
    angle_step = instate['angle_step']
    idx = int((angle % (2*np.pi)) / angle_step)
    if idx >= len(instate['sin_cache']):
        idx = 0
    return instate['sin_cache'][idx], instate['cos_cache'][idx]

def render_leaves_optimized(instate, global_alpha, window):
    """Optimized leaf rendering that preserves original appearance"""
    # Get active leaves
    active_count = instate['leaf_count']
    if active_count == 0:
        return
    
    # Process only leaves that are visible on screen
    x = instate['leaf_x'][:active_count]
    y = instate['leaf_y'][:active_count]
    
    # Create a mask for leaves that are on-screen or near the edges
    visible_mask = (x >= -10) & (x < 130) & (y >= -10) & (y < 70)
    visible_indices = np.where(visible_mask)[0]
    
    if len(visible_indices) == 0:
        return
    
    # Extract properties for visible leaves
    sizes = instate['leaf_size'][visible_indices]
    rotations = instate['leaf_rotation'][visible_indices]
    alphas = instate['leaf_alpha'][visible_indices]
    colors = instate['leaf_color'][visible_indices]
    x_vis = x[visible_mask]
    y_vis = y[visible_mask]
    
    # Precompute common values for each leaf
    # This avoids recalculating them inside the pixel loops
    leaf_count = len(visible_indices)
    rect_sizes = np.array([int(sizes[i] * 3) for i in range(leaf_count)], dtype=np.int32)
    
    # Use cached sin/cos values to avoid expensive trig calculations
    sin_vals = np.zeros(leaf_count, dtype=np.float32)
    cos_vals = np.zeros(leaf_count, dtype=np.float32)
    
    for i in range(leaf_count):
        sin_vals[i], cos_vals[i] = get_cached_sin_cos(instate, rotations[i])
    
    # Batch process leaves - still iterate over leaves but optimize inner operations
    for i in range(leaf_count):
        leaf_idx = visible_indices[i]
        rect_size = rect_sizes[i]
        sin_rot = sin_vals[i]
        cos_rot = cos_vals[i]
        
        # Create coordinate grid more efficiently
        # Instead of using mgrid, use integer ranges and broadcasting
        y_range = np.arange(-rect_size, rect_size+1)
        x_range = np.arange(-rect_size, rect_size+1)
        
        # Use meshgrid which is more memory efficient than mgrid
        dy_grid, dx_grid = np.meshgrid(y_range, x_range, indexing='ij')
        
        # Apply rotation - this is still needed for correct leaf shape
        rotated_x = dx_grid * cos_rot - dy_grid * sin_rot
        rotated_y = dx_grid * sin_rot + dy_grid * cos_rot
        
        # Scale and apply offset
        scale_factor = sizes[i] / 3
        px_grid = np.round(x_vis[i] + rotated_x * scale_factor).astype(np.int32)
        py_grid = np.round(y_vis[i] + rotated_y * scale_factor).astype(np.int32)
        
        # Calculate valid coordinates efficiently
        valid_x = (px_grid >= 0) & (px_grid < 120)
        valid_y = (py_grid >= 0) & (py_grid < 60)
        valid_mask = valid_x & valid_y
        
        # Skip if no valid pixels
        if not np.any(valid_mask):
            continue
        
        # Normalize coordinates for leaf shape calculation
        nx_grid = rotated_x / sizes[i]
        ny_grid = rotated_y / sizes[i]
        
        # Create leaf shape mask - keep original logic to preserve leaf appearance
        leaf_factor = ((nx_grid) ** 2 / 1.2) + ((ny_grid) ** 2)
        taper = 0.3 * (nx_grid + 0.8)
        leaf_factor += np.maximum(0, taper)
        shape_mask = leaf_factor <= 1.0
        
        # Create vein masks
        main_vein_mask = np.abs(ny_grid) < 0.1 * (1 - np.abs(nx_grid * 0.5))
        
        # Side veins
        side_vein_mask = np.zeros_like(main_vein_mask, dtype=bool)
        vein_positions = np.linspace(0.1, 0.6, 5)
        for vein_pos in vein_positions:
            side_vein_mask |= (np.abs(np.abs(ny_grid) - vein_pos) < 0.12) & (nx_grid < 0.5)
        
        # Combined valid pixels mask
        combined_mask = valid_mask & shape_mask
        
        if not np.any(combined_mask):
            continue
            
        # Get pixel coordinates for all valid pixels
        valid_px = px_grid[combined_mask]
        valid_py = py_grid[combined_mask]
        
        # Get vein status for valid pixels
        vein_pixels = (main_vein_mask | side_vein_mask)[combined_mask]
        
        # Calculate alpha values
        leaf_factors = leaf_factor[combined_mask]
        alphas_pix = alphas[i] * (1.0 - 0.3*leaf_factors) * global_alpha
        
        # Get colors for all valid pixels - optimized to do this once per leaf
        hue, sat, val = colors[i]
        hues = np.full_like(alphas_pix, hue)
        sats = np.full_like(alphas_pix, sat)
        vals = np.full_like(alphas_pix, val)
        
        # Adjust color for veins
        vals[vein_pixels] *= 0.7
        
        # Create a mask where our new pixels have higher alpha than existing ones
        idx_pairs = (valid_py, valid_px)
        alpha_mask = alphas_pix > window[idx_pairs + (3,)]
        
        if np.any(alpha_mask):
            # Update only those pixels where our alpha is higher
            final_idx = (idx_pairs[0][alpha_mask], idx_pairs[1][alpha_mask])
            window[final_idx + (0,)] = hues[alpha_mask]
            window[final_idx + (1,)] = sats[alpha_mask]
            window[final_idx + (2,)] = vals[alpha_mask]
            window[final_idx + (3,)] = alphas_pix[alpha_mask]

def secondary_falling_leaves(instate, outstate):
    if instate['count'] == 0:
        # Initialize constants in instate
        instate['MAX_LEAVES'] = 40
        instate['BATCH_SIZE'] = 40
        
        # Initialize separate arrays for leaf properties
        instate.update({
            'leaves_window': np.zeros((32, 300, 4)),  # HSVA format for polar coordinates
            'leaf_theta': np.zeros(instate['MAX_LEAVES'], dtype=np.float32),  # Angular position
            'leaf_r': np.zeros(instate['MAX_LEAVES'], dtype=np.float32),      # Radial position
            'leaf_vtheta': np.zeros(instate['MAX_LEAVES'], dtype=np.float32), # Angular velocity
            'leaf_vr': np.zeros(instate['MAX_LEAVES'], dtype=np.float32),     # Radial velocity
            'leaf_size': np.zeros(instate['MAX_LEAVES'], dtype=np.float32),
            'leaf_rotation': np.zeros(instate['MAX_LEAVES'], dtype=np.float32),
            'leaf_rotation_speed': np.zeros(instate['MAX_LEAVES'], dtype=np.float32),
            'leaf_flutter_phase': np.zeros(instate['MAX_LEAVES'], dtype=np.float32),
            'leaf_flutter_amplitude': np.zeros(instate['MAX_LEAVES'], dtype=np.float32),
            'leaf_color': np.zeros((instate['MAX_LEAVES'], 3), dtype=np.float32),  # HSV
            'leaf_alpha': np.zeros(instate['MAX_LEAVES'], dtype=np.float32),
            'leaf_count': 0,
            'last_update': time.time(),
            'start_time': time.time(),
            'rgb_out': np.zeros((32, 300, 4), dtype=np.uint8)
        })
        
        # Create image plane for secondary display
        instate['leaves_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((32, 300, 4), dtype=np.uint8),
            position=(0, 0, 49.5),
            rotation=(0, 0, 0),
            scale=(9, 9)
        )
        
        # Pre-generate leaf shapes for different sizes
        # For the simplified approach, we'll use a 1D Gaussian shape for each leaf
        instate['leaf_shapes'] = {}
        for size in range(1, 16):  # Different leaf sizes (1-15 pixels)
            # Create a Gaussian profile for leaf intensity
            x = np.linspace(-3, 3, size*2+1)
            shape = np.exp(-x**2)  # Gaussian distribution
            instate['leaf_shapes'][size] = shape
        
        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['leaves_plane'])
        return

    # Get current state
    current_time = time.time()
    dt = current_time - instate['last_update']
    elapsed_time = current_time - instate['start_time']
    instate['last_update'] = current_time
    
    # Calculate fade factor based on duration
    total_duration = instate.get('duration', 45.0)
    fade_in_duration = 5.0
    fade_out_start = total_duration - 5.0
    
    if elapsed_time < fade_in_duration:
        global_alpha = elapsed_time / fade_in_duration
    elif elapsed_time > fade_out_start:
        global_alpha = (total_duration - elapsed_time) / 5.0
    else:
        global_alpha = 1.0
    global_alpha = np.clip(global_alpha, 0, 1)

    # Get environment parameters
    wind = outstate.get('wind', 0)
    whomp = outstate.get('whomp', 0.0)
    season = outstate.get('season', 0.625)  # Default to fall
    
    # Adjust leaf generation rate based on wind and whomp
    leaf_rate = 0.2 + 0.15 * abs(wind) + 0.25 * whomp
    
    # Clear the window efficiently
    window = instate['leaves_window']
    window.fill(0)

    # Generate new leaves
    if np.random.random() < leaf_rate * global_alpha:
        available_slots = instate['MAX_LEAVES'] - instate['leaf_count']
        new_count = min(np.random.randint(1, 4), available_slots)
        
        if new_count > 0:
            start_idx = instate['leaf_count']
            end_idx = start_idx + new_count
            
            # Initialize new leaves in polar coordinates
            instate['leaf_theta'][start_idx:end_idx] = np.random.uniform(0, 2*np.pi, new_count)
            instate['leaf_r'][start_idx:end_idx] = np.random.uniform(0.9, 1.0, new_count)  # Start near edge
            instate['leaf_vtheta'][start_idx:end_idx] = np.random.uniform(-0.2, 0.2, new_count)
            instate['leaf_vr'][start_idx:end_idx] = np.random.uniform(-0.05, -0.2, new_count)  # Negative values for inward movement
            
            instate['leaf_size'][start_idx:end_idx] = np.random.uniform(0.5, 1.5, new_count)
            instate['leaf_rotation'][start_idx:end_idx] = np.random.uniform(0, 2*np.pi, new_count)
            instate['leaf_rotation_speed'][start_idx:end_idx] = np.random.uniform(-0.5, 0.5, new_count)
            instate['leaf_flutter_phase'][start_idx:end_idx] = np.random.uniform(0, 2*np.pi, new_count)
            instate['leaf_flutter_amplitude'][start_idx:end_idx] = np.random.uniform(0.02, 0.08, new_count)
            instate['leaf_alpha'][start_idx:end_idx] = np.random.uniform(0.6, 0.9, new_count)
            
            # Initialize color array
            colors = np.zeros((new_count, 3), dtype=np.float32)
            
            # Calculate distance from spring center
            spring_distance = min(abs(season - 0.125), 1 - abs(season - 0.125))
            fall_distance = min(abs(season - 0.625), 1 - abs(season - 0.625))
            
            # Spring factor (1.0 at spring, decreases as we move away)
            spring_factor = max(0, 1 - spring_distance * 4)  # Sharp falloff
            
            # Fall factor (1.0 at fall, decreases as we move away)
            fall_factor = max(0, 1 - fall_distance * 4)  # Sharp falloff
            
            # If we're in spring, all leaves are green
            if spring_factor > 0.5:
                # All green leaves in spring
                colors[:, 0] = np.random.uniform(0.25, 0.35, new_count)  # Green hue
                colors[:, 1] = np.random.uniform(0.7, 0.9, new_count)    # High saturation
                colors[:, 2] = np.random.uniform(0.3, 0.5, new_count)    # Medium value
            else:
                # Otherwise use seasonal mix with fall colors where appropriate
                # Vectorized color generation
                color_types = np.random.random(new_count)
                
                # Red leaves (proportion increases during fall)
                red_proportion = 0.1 + 0.3 * fall_factor
                red_mask = color_types < red_proportion
                colors[red_mask, 0] = np.random.uniform(0.00, 0.05, np.sum(red_mask))
                colors[red_mask, 1] = np.random.uniform(0.8, 0.95, np.sum(red_mask))
                colors[red_mask, 2] = np.random.uniform(0.4, 0.6, np.sum(red_mask))
                
                # Orange leaves (proportion increases during fall)
                orange_proportion = red_proportion + (0.1 + 0.2 * fall_factor)
                orange_mask = (color_types >= red_proportion) & (color_types < orange_proportion)
                colors[orange_mask, 0] = np.random.uniform(0.05, 0.10, np.sum(orange_mask))
                colors[orange_mask, 1] = np.random.uniform(0.85, 0.95, np.sum(orange_mask))
                colors[orange_mask, 2] = np.random.uniform(0.45, 0.65, np.sum(orange_mask))
                
                # Yellow leaves (consistent, but more vibrant in fall)
                yellow_proportion = orange_proportion + (0.2 + 0.1 * fall_factor)
                yellow_mask = (color_types >= orange_proportion) & (color_types < yellow_proportion)
                colors[yellow_mask, 0] = np.random.uniform(0.10, 0.15, np.sum(yellow_mask))
                colors[yellow_mask, 1] = np.random.uniform(0.8, 0.9, np.sum(yellow_mask))
                colors[yellow_mask, 2] = np.random.uniform(0.5, 0.7, np.sum(yellow_mask))
                
                # Brown leaves (proportion increases during fall)
                brown_proportion = yellow_proportion + (0.05 + 0.15 * fall_factor)
                brown_mask = (color_types >= yellow_proportion) & (color_types < brown_proportion)
                colors[brown_mask, 0] = np.random.uniform(0.07, 0.12, np.sum(brown_mask))
                colors[brown_mask, 1] = np.random.uniform(0.6, 0.8, np.sum(brown_mask))
                colors[brown_mask, 2] = np.random.uniform(0.3, 0.4, np.sum(brown_mask))
                
                # Green leaves (remaining leaves)
                green_mask = color_types >= brown_proportion
                colors[green_mask, 0] = np.random.uniform(0.25, 0.35, np.sum(green_mask))
                colors[green_mask, 1] = np.random.uniform(0.7, 0.9, np.sum(green_mask))
                colors[green_mask, 2] = np.random.uniform(0.3, 0.5, np.sum(green_mask))
            
            instate['leaf_color'][start_idx:end_idx] = colors
            instate['leaf_count'] += new_count

    # Process active leaves
    if instate['leaf_count'] > 0:
        # Update leaf positions and properties
        active_slice = slice(0, instate['leaf_count'])
        
        # Update flutter phase
        instate['leaf_flutter_phase'][active_slice] += dt * 3.0
        
        # Calculate flutter effect
        flutter_theta = np.sin(instate['leaf_flutter_phase'][active_slice]) * instate['leaf_flutter_amplitude'][active_slice]
        
        # Apply wind and flutter to angular velocity
        instate['leaf_vtheta'][active_slice] = flutter_theta + wind * 3
        
        # Update positions
        instate['leaf_theta'][active_slice] -= instate['leaf_vtheta'][active_slice] * dt * 2  # Note the minus sign here
        instate['leaf_r'][active_slice] += instate['leaf_vr'][active_slice] * dt*(1-whomp*0.75)
        
        # Ensure theta is in [0, 2π]
        instate['leaf_theta'][active_slice] = instate['leaf_theta'][active_slice] % (2 * np.pi)
        
        # Update rotation
        instate['leaf_rotation'][active_slice] += instate['leaf_rotation_speed'][active_slice] * dt * 2
        
        # Filter out-of-bounds leaves
        valid_mask = instate['leaf_r'][active_slice] > 0.05 
        
        if not np.all(valid_mask):
            # Compact arrays more efficiently using numpy's boolean indexing
            valid_indices = np.where(valid_mask)[0]
            valid_count = len(valid_indices)
            
            # Do a single copy operation for each array
            for arr_name in ['leaf_theta', 'leaf_r', 'leaf_vtheta', 'leaf_vr', 'leaf_size', 
                           'leaf_rotation', 'leaf_rotation_speed', 'leaf_flutter_phase',
                           'leaf_flutter_amplitude', 'leaf_alpha']:
                instate[arr_name][:valid_count] = instate[arr_name][active_slice][valid_indices]
            
            # Handle color array separately since it's 2D
            instate['leaf_color'][:valid_count] = instate['leaf_color'][active_slice][valid_indices]
            
            instate['leaf_count'] = valid_count
        
        # Render leaves with simplified approach
        render_secondary_leaves_simplified(instate, global_alpha, window)

    # Convert to RGB efficiently
    rgb = color.hsv2rgb(window[..., 0:3])
    alpha = window[..., 3:4]
    rgb_out = instate['rgb_out']
    rgb_out[..., :3] = (rgb * 255).astype(np.uint8)
    rgb_out[..., 3:] = (alpha * 255).astype(np.uint8)
    
    # Update texture
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['leaves_plane'],
        rgb_out
    )

def render_secondary_leaves_simplified(instate, global_alpha, window):
    """Simplified leaf rendering - each leaf occupies a single row in polar coordinates"""
    # Get active leaves
    active_count = instate['leaf_count']
    if active_count == 0:
        return
    
    # Get leaf properties
    theta = instate['leaf_theta'][:active_count]
    r = instate['leaf_r'][:active_count]
    sizes = instate['leaf_size'][:active_count]
    alphas = instate['leaf_alpha'][:active_count]
    colors = instate['leaf_color'][:active_count]
    
    # Convert polar coordinates to pixel indices
    theta_idx = (theta / (2 * np.pi) * 32).astype(np.int32) % 32
    r_idx = (r * 299).astype(np.int32)
    
    # Filter valid leaves (those in the visible area)
    valid_mask = (r_idx >= 0) & (r_idx < 300)
    if not np.any(valid_mask):
        return
    
    # Process each valid leaf
    valid_indices = np.where(valid_mask)[0]
    
    for i in valid_indices:
        # Get the row (angular position) for this leaf
        row = theta_idx[i]
        
        # Get the center position along the radial axis
        center = r_idx[i]
        
        # Get the size of this leaf
        size_pixels = max(1, int(sizes[i] * 6))  # Scale size appropriately
        
        # Limit size to pre-computed shapes
        size_key = min(size_pixels, 15)
        
        # Get the pre-computed leaf shape (Gaussian profile)
        leaf_shape = instate['leaf_shapes'][size_key]
        
        # Calculate start and end positions for the leaf
        half_width = len(leaf_shape) // 2
        start_pos = center - half_width
        end_pos = start_pos + len(leaf_shape)
        
        # Clip to window boundaries
        if start_pos < 0:
            shape_start = -start_pos
            start_pos = 0
        else:
            shape_start = 0
            
        if end_pos > 300:
            shape_end = len(leaf_shape) - (end_pos - 300)
            end_pos = 300
        else:
            shape_end = len(leaf_shape)
        
        # Skip if leaf is completely outside window
        if start_pos >= 300 or end_pos <= 0 or shape_start >= shape_end:
            continue
        
        # Get the part of the shape that's within window
        shape_section = leaf_shape[shape_start:shape_end]
        
        # Calculate alpha values - multiply shape by alpha factor
        alpha_values = shape_section * alphas[i] * global_alpha
        
        # Get color for this leaf
        hue, sat, val = colors[i]
        
        # Check which pixels to update (where our alpha is higher)
        existing_alphas = window[row, start_pos:end_pos, 3]
        update_mask = alpha_values > existing_alphas
        
        if not np.any(update_mask):
            continue
        
        # Update only those pixels where our alpha is higher
        update_indices = np.where(update_mask)[0]
        window_indices = start_pos + update_indices
        
        # Set color and alpha values
        window[row, window_indices, 0] = hue
        window[row, window_indices, 1] = sat
        window[row, window_indices, 2] = val
        window[row, window_indices, 3] = alpha_values[update_mask]