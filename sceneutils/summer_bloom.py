import time
import numpy as np
from skimage import color
import numba

def bioluminescent_wildflowers(instate, outstate):
    if instate['count'] == 0:
        # Initialize parameters
        instate['flower_window'] = np.zeros((60, 120, 4))  # HSVA format
        instate['flower_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((60, 120, 4), dtype=np.uint8),
            position=(0, 0,9.99),  # Positioned in front of most backgrounds
            rotation=(0, 0, 0),
            scale=(1, 1)
        )
        
        # Initialize timing parameters
        instate['start_time'] = time.time()
        instate['last_update'] = time.time()
        instate['growth_duration'] = 15.0  # Grass grows in over 15 seconds
        
        # Create grass field with flowers
        num_stalks = np.random.randint(100, 140)  # More stalks for full coverage
        
        # Generate positions across full width of screen
        base_x = np.random.uniform(0, 120, size=num_stalks)
        
        # Ground parameters
        ground_height = 6  # Height of ground layer
        instate['ground_height'] = ground_height
        instate['ground_color'] = np.random.uniform(0.05, 0.15)  # Dark brown
        instate['ground_texture'] = np.random.random((ground_height, 120)) * 0.1  # Random texture
        
        # Start positions randomly within ground
        base_y = np.random.randint(60 - ground_height, 60, size=num_stalks)
        
        # Grass properties
        max_heights = np.random.randint(7, 45, size=num_stalks)  # Final heights
        thicknesses = np.random.uniform(0.5, 1.2, size=num_stalks)
        
        # Growth parameters
        growth_speeds = np.random.uniform(0.4, 1.2, size=num_stalks)  # Variation in growth speed
        
        # Animation properties
        sway_phases = np.random.uniform(0, 2 * np.pi, size=num_stalks)
        sway_speeds = np.random.uniform(0.5, 1.5, size=num_stalks)
        
        # Every stalk has a flower but they start with size 0 (not bloomed)
        has_flower = np.ones(num_stalks, dtype=bool)
        
        # Diverse color range for flowers
        # Use full color spectrum but favor certain ranges
        hues = np.zeros(num_stalks)
        
        # Set flower colors - wide range across the spectrum
        # Create an array of possible flower colors with different probabilities
        colors_with_weights = [
            # Red/pink flowers (0.95-1.0, 0-0.05)
            (np.random.uniform(0.95, 1.0) % 1.0, 0.1),
            (np.random.uniform(0.0, 0.05), 0.1),
            
            # Orange/yellow flowers (0.05-0.2)
            (np.random.uniform(0.05, 0.2), 0.15),
            
            # Green flowers (uncommon) (0.25-0.35)
            (np.random.uniform(0.25, 0.35), 0.05),
            
            # Blue/cyan flowers (0.5-0.6)
            (np.random.uniform(0.5, 0.6), 0.2),
            
            # Purple/magenta flowers (0.7-0.85)
            (np.random.uniform(0.7, 0.85), 0.25),
            
            # White flowers (special handling - high value, low saturation)
            (np.random.uniform(0.0, 1.0), 0.15)  # Hue doesn't matter for white
        ]
        
        # Extract just the colors and weights
        colors = [c[0] for c in colors_with_weights]
        weights = [c[1] for c in colors_with_weights]
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Choose colors based on weights
        chosen_colors = np.random.choice(colors, size=num_stalks, p=weights)
        hues = chosen_colors
        
        # Mark white flowers (will be handled specially during rendering)
        white_flowers = np.random.random(size=num_stalks) < 0.15
        # For white flowers, hue doesn't matter but we'll tag them by setting negative
        hues[white_flowers] = -1.0
        
        # Add slight hue variation to non-white flowers
        valid_hues = (hues >= 0)
        if np.sum(valid_hues) > 0:
            hues[valid_hues] += np.random.uniform(-0.03, 0.03, size=np.sum(valid_hues))
            hues[valid_hues] = hues[valid_hues] % 1.0  # Keep in valid range
        
        # Pulse properties
        pulse_phases = np.random.uniform(0, 2 * np.pi, size=num_stalks)
        pulse_speeds = np.random.uniform(1.0, 2.0, size=num_stalks)
        
        # Flower sizes (start at 0, will grow after stalks reach full height)
        flower_max_sizes = np.random.randint(3, 6, size=num_stalks)
        flower_sizes = np.zeros(num_stalks, dtype=np.float32)  # Current sizes (will grow)
        flower_growth_speeds = np.random.uniform(0.8, 1.2, size=num_stalks)
        
        # Varying bloom delays so not all flowers bloom at exactly the same time
        bloom_delays = np.random.uniform(0.0, 14.0, size=num_stalks)
        
        # Store in vectorized format
        instate['grass_data'] = {
            'base_x': base_x,
            'base_y': base_y,
            'max_heights': max_heights,
            'current_heights': np.zeros(num_stalks),  # Start with zero height
            'growth_speeds': growth_speeds,
            'thicknesses': thicknesses,
            'sway_phases': sway_phases,
            'sway_speeds': sway_speeds,
            'has_flower': has_flower,
            'hues': hues,
            'pulse_phases': pulse_phases,
            'pulse_speeds': pulse_speeds,
            'flower_max_sizes': flower_max_sizes,
            'flower_sizes': flower_sizes,
            'flower_growth_speeds': flower_growth_speeds,
            'bloom_delays': bloom_delays,
            'white_flowers': white_flowers,
            'fully_grown': np.zeros(num_stalks, dtype=bool)  # Track which stalks are fully grown
        }
        
        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['flower_plane'])
        return

    current_time = time.time()
    dt = current_time - instate['last_update']
    elapsed_time = current_time - instate['start_time']
    total_duration = instate.get('duration', 60.0)
    
    # Calculate event fade
    fade_duration = 5.0
    if elapsed_time < fade_duration:
        event_fade = elapsed_time / fade_duration
    elif elapsed_time > (total_duration - fade_duration):
        event_fade = (total_duration - elapsed_time) / fade_duration
    else:
        event_fade = 1.0
    event_fade = np.clip(event_fade, 0, 1)
    
    # Apply external factors like weather
    wind_factor = outstate.get('wind', 0) * 6  # More responsive to wind
    
    # Clear the window
    instate['flower_window'].fill(0)
    
    # Draw ground layer
    ground_y_start = 60 - instate['ground_height']
    for y in range(ground_y_start, 60):
        # Calculate dirt color with texture variation
        texture_row = y - ground_y_start
        texture = instate['ground_texture'][texture_row, :]
        
        # Darker at top of ground, lighter at bottom for pseudo-3D effect
        depth_factor = (y - ground_y_start) / instate['ground_height']
        brightness = 0.25 + 0.15 * depth_factor
        
        # Set ground color
        instate['flower_window'][y, :, 0] = instate['ground_color']  # Hue
        instate['flower_window'][y, :, 1] = 0.7 - 0.2 * depth_factor  # Saturation
        instate['flower_window'][y, :, 2] = (brightness + texture) * event_fade  # Value
        instate['flower_window'][y, :, 3] = 1.0 * event_fade  # Alpha
    
    # Update global time for animations
    global_sway_time = elapsed_time * 0.8
    
    # Update grass and flower data
    grass_data = instate['grass_data']
    
    # Update animation phases
    grass_data['sway_phases'] += dt * grass_data['sway_speeds'] * (1.0 + wind_factor * 0.8)
    grass_data['pulse_phases'] += dt * grass_data['pulse_speeds']
    
    # Update grass growth based on elapsed time
    growth_progress = min(1.0, elapsed_time / instate['growth_duration'])
    
    # Grow each stalk proportionally to its growth speed
    for i in range(len(grass_data['current_heights'])):
        # Grow the stalk if not fully grown
        if not grass_data['fully_grown'][i]:
            target_height = grass_data['max_heights'][i] * growth_progress
            growth_rate = grass_data['growth_speeds'][i] * dt * 5.0  # Adjust for reasonable growth speed
            
            # Smoothly approach target height
            height_diff = target_height - grass_data['current_heights'][i]
            if height_diff > 0:
                growth = min(height_diff, growth_rate)
                grass_data['current_heights'][i] += growth
            
            # Check if fully grown
            if grass_data['current_heights'][i] >= grass_data['max_heights'][i] * 0.95:
                grass_data['fully_grown'][i] = True
    
    # Update flower blooming - only after stalks are fully grown
    bloom_threshold = instate['growth_duration']
    
    for i in range(len(grass_data['flower_sizes'])):
        if grass_data['fully_grown'][i]:
            # Only start blooming after the stalk is fully grown and delay period passed
            bloom_time = elapsed_time - bloom_threshold - grass_data['bloom_delays'][i]
            
            if bloom_time > 0:
                # Bloom over a 5 second period
                bloom_progress = min(1.0, bloom_time / 5.0)
                
                # Smoothly grow flower
                target_size = grass_data['flower_max_sizes'][i] * bloom_progress
                growth_rate = grass_data['flower_growth_speeds'][i] * dt * 2.0
                
                size_diff = target_size - grass_data['flower_sizes'][i]
                if size_diff > 0:
                    growth = min(size_diff, growth_rate)
                    grass_data['flower_sizes'][i] += growth
    
    # Calculate sway amounts for all grass stalks
    base_sway = np.sin(grass_data['sway_phases']) * 1.5
    wind_sway = np.sin(global_sway_time + grass_data['base_x'] * 0.05) * wind_factor * 2.0
    total_sway = base_sway + wind_sway
    
    # Calculate pulse for glow
    pulse_factors = 0.7 + 0.3 * np.sin(grass_data['pulse_phases'])
    
    # Draw all grass stalks
    draw_grass_field(
        instate['flower_window'],
        grass_data['base_x'],
        grass_data['base_y'],
        grass_data['current_heights'],  # Use current growth height
        total_sway,
        grass_data['thicknesses'],
        grass_data['hues'],
        grass_data['has_flower'],
        grass_data['flower_sizes'],  # Use current bloom size
        pulse_factors,
        event_fade,
        grass_data['white_flowers']
    )
    
    # Convert HSVA to RGBA for rendering
    rgb_main = color.hsv2rgb(instate['flower_window'][:,:,0:3])
    alpha_main = instate['flower_window'][:,:,3:4]
    rgb_main_out = np.concatenate([rgb_main*255, alpha_main*255], axis=2).astype(np.uint8)
    
    # Update the image plane
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['flower_plane'],
        rgb_main_out[:,:,:]
    )
    
    instate['last_update'] = current_time


def draw_grass_field(window, base_x, base_y, heights, sway_amounts, thicknesses, 
                    hues, has_flower, flower_sizes, pulse_factors, event_fade, white_flowers):
    """Draw entire field of grass with vectorized operations where possible"""
    # Process each grass stalk individually (could be optimized further with Numba)
    for i in range(len(base_x)):
        # Only draw if has some height
        if heights[i] <= 0:
            continue
            
        # Draw single grass stalk
        top_x, top_y = draw_grass_stalk(
            window,
            base_x[i],
            base_y[i],
            int(heights[i]),  # Convert to integer height
            sway_amounts[i],
            thicknesses[i],
            0.3,  # Green hue for grass
            0.5 * event_fade,  # Brightness
            0.8 * event_fade,  # Alpha
            pulse_factors[i]
        )
        
        # Draw flower if this stalk has one and flower has started growing
        if has_flower[i] and flower_sizes[i] > 0:
            is_white = white_flowers[i]
            draw_cross_flower(
                window,
                top_x,
                top_y,
                int(flower_sizes[i]),  # Convert to integer size
                hues[i],
                0.9 * event_fade,  # Brighter
                pulse_factors[i],
                0.9 * event_fade,
                is_white
            )


@numba.njit
def draw_grass_stalk(window, base_x, base_y, height, sway_amount, thickness, 
                     hue, brightness, alpha, pulse_factor):
    """Draw a single grass stalk with sway - optimized with Numba"""
    # Handle edge case
    if height <= 0:
        return int(base_x), int(base_y)
        
    # Prepare points for the entire stalk at once
    stalk_y = np.zeros(height, dtype=np.int32)
    stalk_x = np.zeros(height, dtype=np.int32)
    
    for i in range(height):
        # Calculate sway (increases with height)
        sway_factor = (i / height) ** 1.5  # More sway at the top
        x_offset = sway_amount * sway_factor
        
        # Calculate position
        stalk_x[i] = int(base_x + x_offset)
        stalk_y[i] = base_y - i  # Going upward from base
    
    # Draw each point with thickness
    for i in range(height):
        x, y = stalk_x[i], stalk_y[i]
        
        # Skip if offscreen
        if y < 0 or y >= 60:
            continue
            
        # Calculate brightness gradient - brighter at top
        brightness_factor = 0.7 + 0.3 * (i / height)
        actual_brightness = brightness * brightness_factor * pulse_factor
        
        # Draw with thickness
        for t in range(-int(thickness), int(thickness) + 1):
            px = x + t
            
            # Check bounds
            if 0 <= px < 120:
                # Calculate fade based on distance from center
                thickness_fade = 1.0 - (abs(t) / thickness) ** 2 if thickness > 0 else 1.0
                
                # Set colors
                window[y, px, 0] = hue  # Hue
                window[y, px, 1] = 0.9  # Saturation
                window[y, px, 2] = actual_brightness * thickness_fade  # Value
                window[y, px, 3] = alpha * thickness_fade  # Alpha
    
    # Return top position for flower placement (last point)
    return stalk_x[height-1], stalk_y[height-1]


@numba.njit
def draw_cross_flower(window, center_x, center_y, size, hue, brightness, pulse, alpha, is_white):
    """Draw a simple cross-shaped flower with diagonals - optimized with Numba"""
    # Check if size is valid
    if size <= 0:
        return
        
    # Handle white flowers specially
    if is_white:
        saturation = 0.2  # Low saturation for white
        value_boost = 1.4  # Extra bright
        flower_hue = 0.0  # Doesn't matter for white, but set a default
    else:
        saturation = 0.9
        value_boost = 1.0
        flower_hue = hue
    
    # Draw center point
    if 0 <= center_x < 120 and 0 <= center_y < 60:
        window[center_y, center_x, 0] = flower_hue
        window[center_y, center_x, 1] = saturation
        window[center_y, center_x, 2] = brightness * pulse * 1.2 * value_boost  # Brighter center
        window[center_y, center_x, 3] = alpha
    
    # Draw petals (up, down, left, right)
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # (dx, dy)
    
    for dx, dy in directions:
        for i in range(1, size):
            px = center_x + dx * i
            py = center_y + dy * i
            
            # Check bounds
            if 0 <= px < 120 and 0 <= py < 60:
                # Fade towards the tips
                tip_fade = 1.0 - (i / (size + 1)) ** 2
                
                window[py, px, 0] = flower_hue
                window[py, px, 1] = saturation
                window[py, px, 2] = brightness * pulse * tip_fade * value_boost
                window[py, px, 3] = alpha * tip_fade
    
    # Draw diagonals (just one pixel in each diagonal direction)
    diagonal_directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonal (dx, dy)
    
    for dx, dy in diagonal_directions:
        # Just draw one pixel in each diagonal
        px = center_x + dx
        py = center_y + dy
        
        # Check bounds
        if 0 <= px < 120 and 0 <= py < 60:
            # Use slightly less brightness than center but more than petal tips
            diag_fade = 0.85
            
            window[py, px, 0] = flower_hue
            window[py, px, 1] = saturation
            window[py, px, 2] = brightness * pulse * diag_fade * value_boost
            window[py, px, 3] = alpha * diag_fade