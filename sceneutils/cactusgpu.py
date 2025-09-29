import time
import cupy as cp
import numpy as np  # Keep numpy for CPU operations
from skimage import color
from pathlib import Path

ParentPath = Path(__file__).parent.parent
media_path = ParentPath / 'media'
sound_path = media_path / 'sounds'

def dancing_joshua(instate, outstate):
    if instate['count'] == 0:
        # Initialize with simpler structure
        outstate['has_cactus'] = True
        
        # Create image plane - maintain original resolution for quality
        instate['joshua_window'] = cp.zeros((60, 120, 4))  # HSVA format on GPU
        instate['joshua_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((60, 120, 4), dtype=np.uint8),  # Initial CPU array for plane
            position=(0, 0, 19.5),
            rotation=(0, 0, 0),
            scale=(3, 3)
        )
        
        # Initialize timing parameters
        instate['start_time'] = time.time()
        instate['last_update'] = time.time()
        
        # Store random state on GPU
        instate['rng'] = cp.random.RandomState(int(time.time()))
        
        # Simplified wind parameters (use Python scalars for simple values)
        instate['wind'] = {
            'strength': 0,
            'target_strength': float(instate['rng'].uniform(0.2, 0.4)),
            'change_time': time.time(),
            'change_interval': float(instate['rng'].uniform(3, 7))
        }
        
        # Ground level
        instate['ground_y'] = 50
        instate['soil_color'] = float(instate['rng'].uniform(0.05, 0.12))
        
        # Pre-calculate coordinate grids once (keep on GPU)
        y, x = cp.mgrid[0:60, 0:120]
        instate['y_indices'] = y
        instate['x_indices'] = x
        
        # Create a simplified Joshua tree structure
        # Main trunk (use Python scalars for simple values)
        trunk_base_x = 60 + int(instate['rng'].randint(-15, 16))
        trunk_base_y = instate['ground_y']
        trunk_height = 15
        
        instate['trunk'] = {
            'base_x': trunk_base_x,
            'base_y': trunk_base_y,
            'height': trunk_height,
            'width': 2.5,  # Slender trunk
            'sway_phase': float(instate['rng'].uniform(0, 2*cp.pi)),
            'color_variation': float(instate['rng'].uniform(-0.02, 0.02))
        }
        
        # Create main branches with better spread distribution
        instate['branches'] = []
        num_branches = int(instate['rng'].randint(2, 6))
        
        # Define angle ranges for left and right sides
        right_angle_range = cp.linspace(cp.pi/6, cp.pi/3, num_branches//2 + num_branches%2)
        left_angle_range = cp.linspace(-cp.pi/6, -cp.pi/3, num_branches//2)
        
        # Add some noise to each angle to make it look more natural
        right_angle_range += instate['rng'].uniform(-0.1, 0.1, size=len(right_angle_range))
        left_angle_range += instate['rng'].uniform(-0.1, 0.1, size=len(left_angle_range))
        
        # Convert to CPU for easier Python loop processing (one-time cost at initialization)
        right_angle_range_cpu = cp.asnumpy(right_angle_range)
        left_angle_range_cpu = cp.asnumpy(left_angle_range)
        
        # Combine angles, alternating sides
        branch_angles = []
        for i in range(max(len(right_angle_range_cpu), len(left_angle_range_cpu))):
            if i < len(right_angle_range_cpu):
                branch_angles.append(right_angle_range_cpu[i])
            if i < len(left_angle_range_cpu):
                branch_angles.append(left_angle_range_cpu[i])
        
        # Keep only the number of branches we want
        branch_angles = branch_angles[:num_branches]
        
        # Distribute branches along trunk height (avoid too many at same height)
        height_positions = cp.linspace(0.6, 1.0, num_branches)
        height_positions = cp.asnumpy(height_positions)  # Move to CPU for Python shuffling
        np.random.shuffle(height_positions)
        
        for i in range(num_branches):
            # Branch height along trunk (more toward top)
            height_ratio = height_positions[i]
            branch_y = trunk_base_y - height_ratio * trunk_height
            
            # Get the angle for this branch
            angle = branch_angles[i]
            
            # Upper branches tend to be more vertical
            if height_ratio > 0.8:
                angle *= 0.6  # Reduce angle to make more vertical
            
            # Branch length
            length = float(instate['rng'].uniform(18, 25))
            
            # Branch width
            width = 2.5 * (1.0 - height_ratio * 0.2)
            
            # Create branch with secondary branches
            branch = {
                'height_ratio': height_ratio,
                'angle': angle-cp.pi/2,
                'length': length,
                'width': width,
                'sway_phase': float(instate['rng'].uniform(0, 2*cp.pi)),
                'secondary_branches': []
            }
            
            # Add 2-4 secondary branches with better distribution
            num_secondary = int(instate['rng'].randint(2, 5))
            
            # Distribute secondary branches with better spacing
            # Define positions along the branch, favoring the outer part
            sec_positions = cp.linspace(0.5, 1.0, num_secondary)
            sec_positions = sec_positions**1.5  # Bias toward outer part of branch
            
            # Define angle range to cover both upward and slightly downward
            sec_angle_range = cp.linspace(-cp.pi/6, cp.pi/6, num_secondary)
            # Add noise to angles
            sec_angle_range += instate['rng'].uniform(-0.1, 0.1, size=num_secondary)
            
            # Convert to CPU for easier Python loop processing
            sec_positions_cpu = cp.asnumpy(sec_positions)
            sec_angle_range_cpu = cp.asnumpy(sec_angle_range)
            
            # Shuffle the positions and angles for more natural look
            np.random.shuffle(sec_positions_cpu)
            np.random.shuffle(sec_angle_range_cpu)
            
            for j in range(num_secondary):
                # Position along primary branch (with better spacing)
                position = sec_positions_cpu[j]
                
                # Angle for this secondary branch (with better distribution)
                sec_angle = sec_angle_range_cpu[j]
                
                # Length and width
                sec_length = length * float(instate['rng'].uniform(0.5, 0.7))
                sec_width = width * float(instate['rng'].uniform(0.8, 0.95))
                
                secondary = {
                    'position': position,
                    'angle': sec_angle,
                    'length': sec_length,
                    'width': sec_width,
                    'sway_phase': float(instate['rng'].uniform(0, 2*cp.pi)),
                    'leaf_cluster': {
                        'size': float(instate['rng'].uniform(2, 3.5)),
                        'sway_phase': float(instate['rng'].uniform(0, 2*cp.pi)),
                    }
                }
                branch['secondary_branches'].append(secondary)
            
            # Add a leaf cluster to the primary branch end
            branch['leaf_cluster'] = {
                'size': float(instate['rng'].uniform(0.5, 1)),
                'sway_phase': float(instate['rng'].uniform(0, 2*cp.pi)),
            }
            
            instate['branches'].append(branch)
        
        # Add some rocks and desert vegetation (simplified)
        instate['decorations'] = []
        num_decorations = int(instate['rng'].randint(3, 7))
        
        for i in range(num_decorations):
            decoration_type = int(instate['rng'].randint(0, 2))  # 0 = rock, 1 = desert shrub
            pos_x = float(instate['rng'].uniform(10, 110))
            pos_y = float(instate['rng'].uniform(instate['ground_y'], 58))
            
            decoration = {
                'type': decoration_type,
                'pos': cp.array([pos_x, pos_y]),
                'size': float(instate['rng'].uniform(1, 2.5)),
                'color_variation': float(instate['rng'].uniform(-0.05, 0.05))
            }
            instate['decorations'].append(decoration)
            
        # Set drawing parameters for optimization
        instate['trunk_steps'] = 12
        instate['branch_steps'] = 10
        instate['sec_branch_steps'] = 8
        
        return

    if instate['count'] == -1:
        outstate['has_cactus'] = False
        outstate['render'][instate['frame_id']].remove_image_plane(instate['joshua_plane'])
        return

    current_time = time.time()
    dt = min(current_time - instate['last_update'], 0.1)  # Cap dt to avoid large jumps
    elapsed_time = current_time - instate['start_time']
    total_duration = instate.get('duration', 60.0)
    
    # Calculate fade factor
    fade_duration = 4.0
    if elapsed_time < fade_duration:
        fade_factor = elapsed_time / fade_duration
    elif elapsed_time > (total_duration - fade_duration):
        fade_factor = (total_duration - elapsed_time) / fade_duration
    else:
        fade_factor = 1.0
    fade_factor = min(max(fade_factor, 0), 1)  # Python min/max for scalar value
    
    season = outstate['season']
    s_wind = (1 - 0.5 * cp.cos(cp.pi * 2 * (season - 0.125)))
    
    # Update wind parameters
    wind = instate['wind']
    if current_time - wind['change_time'] > wind['change_interval']:
        wind['target_strength'] = float(instate['rng'].uniform(0.2, 0.4) * s_wind)
        wind['change_time'] = current_time
        wind['change_interval'] = float(instate['rng'].uniform(3, 7))
    
    # Smoothly adjust wind strength
    wind['strength'] += (wind['target_strength'] - wind['strength']) * min(dt * 0.5, 0.1)
    
    # Update sway phases
    trunk = instate['trunk']
    trunk['sway_phase'] += dt * 0.8
    
    for branch in instate['branches']:
        branch['sway_phase'] += dt * 0.6
        branch['leaf_cluster']['sway_phase'] += dt * 1.0
        
        # Update secondary branches
        for sec_branch in branch['secondary_branches']:
            sec_branch['sway_phase'] += dt * 0.7
            sec_branch['leaf_cluster']['sway_phase'] += dt * 1.1
    
    # Clear the window (on GPU)
    window = instate['joshua_window']
    window.fill(0)
    
    # Get coordinate grids (already on GPU)
    y_indices = instate['y_indices']
    x_indices = instate['x_indices']
    
    # Draw ground with transparent sky (vectorized operation on GPU)
    ground_mask = y_indices >= instate['ground_y']
    window[ground_mask, 0] = instate['soil_color']  # Hue
    window[ground_mask, 1] = 0.5  # Saturation
    window[ground_mask, 2] = 0.3  # Value
    window[ground_mask, 3] = fade_factor  # Alpha
    
    # Draw desert decorations - process all at once for performance (on GPU)
    for decoration in instate['decorations']:
        dec_x, dec_y = decoration['pos']
        # Limit calculation to a region around the decoration (optimization)
        size = decoration['size']
        x_min = max(0, int(dec_x - size - 1))
        x_max = min(120, int(dec_x + size + 1))
        y_min = max(0, int(dec_y - size - 1))
        y_max = min(60, int(dec_y + size + 1))
        
        if x_min < x_max and y_min < y_max:  # Make sure region is valid
            sub_x = x_indices[y_min:y_max, x_min:x_max]
            sub_y = y_indices[y_min:y_max, x_min:x_max]
            
            dec_distance = cp.sqrt((sub_x - dec_x)**2 + (sub_y - dec_y)**2)
            dec_mask = dec_distance < size
            
            if decoration['type'] == 0:  # Rock
                window[y_min:y_max, x_min:x_max][dec_mask, 0] = 0.05 + decoration['color_variation']
                window[y_min:y_max, x_min:x_max][dec_mask, 1] = 0.2
                window[y_min:y_max, x_min:x_max][dec_mask, 2] = 0.4
                window[y_min:y_max, x_min:x_max][dec_mask, 3] = fade_factor
            else:  # Desert shrub
                window[y_min:y_max, x_min:x_max][dec_mask, 0] = 0.25 + decoration['color_variation']
                window[y_min:y_max, x_min:x_max][dec_mask, 1] = 0.4
                window[y_min:y_max, x_min:x_max][dec_mask, 2] = 0.3
                window[y_min:y_max, x_min:x_max][dec_mask, 3] = fade_factor
    
    # Calculate trunk sway (scalar calculation)
    trunk_sway = float(cp.sin(trunk['sway_phase']) * wind['strength'] * 2.0)
    
    # Draw trunk
    trunk_base_x = trunk['base_x']
    trunk_base_y = trunk['base_y']
    trunk_height = trunk['height']
    trunk_width = trunk['width']
    
    # Draw trunk with optimized number of segments
    steps = instate['trunk_steps']
    y_coords = cp.linspace(trunk_base_y, trunk_base_y - trunk_height, steps).get()  # Get as CPU array for loop
    
    for i, y in enumerate(y_coords):
        # Height ratio (0 at base, 1 at top)
        height_ratio = i / max(steps - 1, 1)
        
        # Add some sway that increases with height
        sway_amount = trunk_sway * height_ratio**2
        x = trunk_base_x + sway_amount
        
        # Width tapers slightly toward top
        width = trunk_width * (1.0 - height_ratio * 0.3)
        
        # Process region around this trunk segment
        x_min = max(0, int(x - width - 1))
        x_max = min(120, int(x + width + 1))
        y_min = max(0, int(y - 2))
        y_max = min(60, int(y + 2))
        
        if x_min < x_max and y_min < y_max:
            sub_x = x_indices[y_min:y_max, x_min:x_max]
            sub_y = y_indices[y_min:y_max, x_min:x_max]
            
            dx = sub_x - x
            dy = sub_y - y
            dist_squared = (dx**2 / (width**2 + 0.001)) + (dy**2 / 4)
            trunk_mask = dist_squared <= 1
            
            window[y_min:y_max, x_min:x_max][trunk_mask, 0] = 0.08 + trunk['color_variation']
            window[y_min:y_max, x_min:x_max][trunk_mask, 1] = 0.4
            window[y_min:y_max, x_min:x_max][trunk_mask, 2] = 0.4
            window[y_min:y_max, x_min:x_max][trunk_mask, 3] = fade_factor
    
    # Draw branches and leaf clusters
    for branch in instate['branches']:
        # Calculate branch start position
        height_ratio = branch['height_ratio']
        branch_y = trunk_base_y - height_ratio * trunk_height
        
        # Add trunk sway at this height
        branch_x = trunk_base_x + trunk_sway * height_ratio**2
        
        # Calculate branch sway
        branch_sway = float(cp.sin(branch['sway_phase']) * wind['strength'] * 2.0)
        branch_angle = branch['angle'] + branch_sway * 0.2
        
        # Calculate branch end position
        end_x = branch_x + float(cp.cos(branch_angle) * branch['length'])
        end_y = branch_y + float(cp.sin(branch_angle) * branch['length'])
        
        # Draw primary branch with optimized number of steps
        steps = instate['branch_steps']
        for i in range(steps):
            t = i / max(steps - 1, 1)
            
            # Add a slight waviness - Joshua trees have gnarly branches
            bend = float(cp.sin(t * cp.pi) * 0.3)
            local_angle = branch_angle + bend
            
            # Calculate position along branch
            pos_x = branch_x + float(cp.cos(local_angle) * branch['length'] * t)
            pos_y = branch_y + float(cp.sin(local_angle) * branch['length'] * t)
            
            # Width tapers toward end
            width = branch['width'] * (1.0 - t * 0.7)
            width = max(width, 0.1)  # Ensure minimum width
            
            # Process region around this branch segment
            x_min = max(0, int(pos_x - width - 1))
            x_max = min(120, int(pos_x + width + 1))
            y_min = max(0, int(pos_y - 2))
            y_max = min(60, int(pos_y + 2))
            
            if x_min < x_max and y_min < y_max:
                sub_x = x_indices[y_min:y_max, x_min:x_max]
                sub_y = y_indices[y_min:y_max, x_min:x_max]
                
                dx = sub_x - pos_x
                dy = sub_y - pos_y
                dist_squared = (dx**2 / (width**2 + 0.001)) + (dy**2 / 2)
                branch_mask = dist_squared <= 1
                
                window[y_min:y_max, x_min:x_max][branch_mask, 0] = 0.09 + trunk['color_variation']
                window[y_min:y_max, x_min:x_max][branch_mask, 1] = 0.45
                window[y_min:y_max, x_min:x_max][branch_mask, 2] = 0.4
                window[y_min:y_max, x_min:x_max][branch_mask, 3] = fade_factor
        
        # Draw secondary branches
        for sec_branch in branch['secondary_branches']:
            # Calculate secondary branch start position along primary branch
            sec_pos = sec_branch['position']
            
            # Find position along primary branch
            sec_base_x = branch_x + float(cp.cos(branch_angle) * branch['length'] * sec_pos)
            sec_base_y = branch_y + float(cp.sin(branch_angle) * branch['length'] * sec_pos)
            
            # Calculate secondary branch angle, combining primary angle and its own
            sec_angle = branch_angle + sec_branch['angle']
            
            # Add secondary sway
            sec_sway = float(cp.sin(sec_branch['sway_phase']) * wind['strength'] * 1.5)
            sec_angle += sec_sway * 0.1
            
            # Calculate secondary branch end position
            sec_end_x = sec_base_x + float(cp.cos(sec_angle) * sec_branch['length'])
            sec_end_y = sec_base_y + float(cp.sin(sec_angle) * sec_branch['length'])
            
            # Draw secondary branch with optimized number of steps
            steps = instate['sec_branch_steps']
            for i in range(steps):
                t = i / max(steps - 1, 1)
                
                # Add waviness
                bend = float(cp.sin(t * cp.pi) * 0.4)
                local_angle = sec_angle + bend
                
                # Position along secondary branch
                pos_x = sec_base_x + float(cp.cos(local_angle) * sec_branch['length'] * t)
                pos_y = sec_base_y + float(cp.sin(local_angle) * sec_branch['length'] * t)
                
                # Width tapers toward end
                width = sec_branch['width'] * (1.0 - t * 0.7)
                width = max(width, 0.1)  # Ensure minimum width
                
                # Process region around this branch segment
                x_min = max(0, int(pos_x - width - 1))
                x_max = min(120, int(pos_x + width + 1))
                y_min = max(0, int(pos_y - 2))
                y_max = min(60, int(pos_y + 2))
                
                if x_min < x_max and y_min < y_max:
                    sub_x = x_indices[y_min:y_max, x_min:x_max]
                    sub_y = y_indices[y_min:y_max, x_min:x_max]
                    
                    dx = sub_x - pos_x
                    dy = sub_y - pos_y
                    dist_squared = (dx**2 / (width**2 + 0.001)) + (dy**2 / 2)
                    sec_mask = dist_squared <= 1
                    
                    window[y_min:y_max, x_min:x_max][sec_mask, 0] = 0.095 + trunk['color_variation']
                    window[y_min:y_max, x_min:x_max][sec_mask, 1] = 0.48
                    window[y_min:y_max, x_min:x_max][sec_mask, 2] = 0.4
                    window[y_min:y_max, x_min:x_max][sec_mask, 3] = fade_factor
            
            # Draw leaf cluster at end of secondary branch
            draw_leaf_cluster_optimized(
                window, x_indices, y_indices,
                sec_end_x, sec_end_y,
                sec_branch['leaf_cluster']['size'],
                wind['strength'],
                sec_branch['leaf_cluster']['sway_phase'],
                trunk['color_variation'],
                fade_factor,
                instate['rng']  # Pass RNG to avoid creating new one
            )
        
        # Draw leaf cluster at end of primary branch
        draw_leaf_cluster_optimized(
            window, x_indices, y_indices,
            end_x, end_y,
            branch['leaf_cluster']['size'],
            wind['strength'],
            branch['leaf_cluster']['sway_phase'],
            trunk['color_variation'],
            fade_factor,
            instate['rng']  # Pass RNG to avoid creating new one
        )
    
    # Draw a small leaf cluster at the top of the trunk for a full look
    trunk_top_x = trunk_base_x + trunk_sway
    trunk_top_y = trunk_base_y - trunk_height
    draw_leaf_cluster_optimized(
        window, x_indices, y_indices,
        trunk_top_x, trunk_top_y,
        2.5,  # Size
        wind['strength'],
        trunk['sway_phase'],
        trunk['color_variation'],
        fade_factor,
        instate['rng']  # Pass RNG to avoid creating new one
    )
    
    # Convert HSVA to RGBA for rendering
    # Only transfer to CPU at the very end
    cpu_window = cp.asnumpy(window)  # This is the only GPU->CPU transfer in the main loop
    rgb = color.hsv2rgb(cpu_window[..., 0:3])
    alpha = cpu_window[..., 3:4]
    rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)
    
    # Update the image plane
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['joshua_plane'],
        rgb_out[:,:,:]
    )
    
    # Update timestamp
    instate['last_update'] = current_time

# Optimized leaf cluster drawing function - updated to keep operations on GPU
def draw_leaf_cluster_optimized(window, x_indices, y_indices, center_x, center_y, size, 
                               wind_strength, sway_phase, color_variation, fade_factor, rng):
    # Skip if out of bounds
    if center_x < 0 or center_x >= 120 or center_y < 0 or center_y >= 60:
        return
    
    # Apply leaf cluster sway
    cluster_sway = float(cp.sin(sway_phase) * wind_strength * 1.5)
    center_x += cluster_sway
    center_y += cluster_sway * 0.3
    
    # Optimize by only processing the region around the leaf cluster
    max_radius = size * 2.0  # Maximum possible extent of the cluster
    x_min = max(0, int(center_x - max_radius))
    x_max = min(120, int(center_x + max_radius + 1))
    y_min = max(0, int(center_y - max_radius))
    y_max = min(60, int(center_y + max_radius + 1))
    
    if x_min >= x_max or y_min >= y_max:
        return
    
    # Get the subset of coordinate arrays
    sub_x = x_indices[y_min:y_max, x_min:x_max]
    sub_y = y_indices[y_min:y_max, x_min:x_max]
    
    # Base center area
    center_radius = size * 0.6
    dx = sub_x - center_x
    dy = sub_y - center_y
    dist = cp.sqrt(dx**2 + dy**2)
    center_mask = dist < center_radius
    
    # Apply colors to center
    window[y_min:y_max, x_min:x_max][center_mask, 0] = 0.33 + color_variation  # Green hue
    window[y_min:y_max, x_min:x_max][center_mask, 1] = 0.5  # Moderate saturation
    window[y_min:y_max, x_min:x_max][center_mask, 2] = 0.5  # Medium brightness
    window[y_min:y_max, x_min:x_max][center_mask, 3] = fade_factor
    
    # Add spiky leaves with optimized number
    num_spikes = min(int(size * 2.5), 10)  # Use fewer spikes for performance
    
    # OPTIMIZATION: Draw all spikes at once instead of looping
    # Create arrays for all spike properties
    
    # Distribute spike angles evenly around the circle
    base_angles = cp.linspace(0, 2*cp.pi, num_spikes, endpoint=False)
    
    # Add random variation to each angle (but not so much they overlap)
    angle_noise = rng.uniform(-0.2, 0.2, size=num_spikes)
    spike_angles = base_angles + angle_noise
    
    # Create random lengths for all spikes at once
    spike_lengths = size * rng.uniform(0.8, 1.4, size=num_spikes)
    
    # Simplified drawing - use fewer steps per spike
    spike_steps = min(int(size * 2), 4)
    
    # For each spike angle and length
    for i in range(num_spikes):
        actual_angle = spike_angles[i]
        spike_length = spike_lengths[i]
        
        for step in range(spike_steps):
            d = center_radius + (spike_length - center_radius) * (step / max(spike_steps - 1, 1))
            spike_x = center_x + cp.cos(actual_angle) * d
            spike_y = center_y + cp.sin(actual_angle) * d
            
            # Skip if this point is outside our processing region
            if not (x_min <= spike_x < x_max) or not (y_min <= spike_y < y_max):
                continue
            
            # Width tapers toward tip
            width_factor = 1.0 - (d - center_radius) / max(spike_length - center_radius, 0.1)
            spike_width = max(size * 0.3 * width_factor, 0.2)
            
            # Calculate distances within our subregion
            spike_dx = sub_x - spike_x
            spike_dy = sub_y - spike_y
            spike_dist_squared = (spike_dx**2 + spike_dy**2) / (spike_width**2 + 0.001)
            spike_mask = spike_dist_squared <= 1
            
            # Apply colors
            window[y_min:y_max, x_min:x_max][spike_mask, 0] = 0.33 + color_variation
            window[y_min:y_max, x_min:x_max][spike_mask, 1] = 0.5
            window[y_min:y_max, x_min:x_max][spike_mask, 2] = 0.5 * max(width_factor, 0.3)
            window[y_min:y_max, x_min:x_max][spike_mask, 3] = fade_factor

def dancing_prickly_pear(instate, outstate):
    if instate['count'] == 0:
        # Initialize with simpler structure
        outstate['has_cactus'] = True
        
        # Create image plane - maintain original resolution for quality
        instate['cactus_window'] = np.zeros((60, 120, 4))  # HSVA format
        instate['cactus_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((60, 120, 4), dtype=np.uint8),
            position=(0, 0, 19.5),
            rotation=(0, 0, 0),
            scale=(3, 3)
        )
        
        # Initialize timing parameters
        instate['start_time'] = time.time()
        instate['last_update'] = time.time()
        
        # Wind parameters
        instate['wind'] = {
            'strength': 0,
            'target_strength': np.random.uniform(0.1, 0.3),
            'change_time': time.time(),
            'change_interval': np.random.uniform(3, 7)
        }
        
        # Ground level
        instate['ground_y'] = 50
        instate['soil_color'] = np.random.uniform(0.05, 0.12)
        
        # Pre-calculate coordinate grids once
        y, x = np.mgrid[0:60, 0:120]
        instate['y_indices'] = y
        instate['x_indices'] = x
        
        # Create a prickly pear cactus structure
        base_x = 60 + np.random.randint(-10, 11)  # Centered with slight variation
        base_y = instate['ground_y']
        
        # Create pads array
        instate['pads'] = []
        
        # Start with a base pad - make it larger
        base_pad = {
            'x': base_x,
            'y': base_y - 8,  # Place slightly higher above ground
            'width': np.random.uniform(15, 20),  # Increased width
            'height': np.random.uniform(18, 22),  # Increased height
            'angle': 0,  # Upright position
            'sway_phase': np.random.uniform(0, 2*np.pi),
            'sway_amount': np.random.uniform(0.8, 1.2),
            'color_variation': np.random.uniform(-0.03, 0.03),
            'has_flower': False,
            'parent_idx': -1,  # No parent
            'depth': 0,  # Base pad is at depth 0
            'attach_points': []  # Will store attachment points for child pads
        }
        instate['pads'].append(base_pad)
        
        # Create branching structure
        max_pads = np.random.randint(8, 14)
        
        # Use breadth-first approach to ensure connected structure
        current_depth = 0
        while len(instate['pads']) < max_pads and current_depth < 4:  # Limit depth to prevent excessive branching
            # Find pads at current depth
            parent_indices = [i for i, pad in enumerate(instate['pads']) if pad['depth'] == current_depth]
            
            # For each parent pad, potentially add 1-3 child pads
            for parent_idx in parent_indices:
                parent = instate['pads'][parent_idx]
                
                # Number of pads to add to this parent
                num_new_pads = min(np.random.randint(1, 4), max_pads - len(instate['pads']))
                if num_new_pads <= 0:
                    continue
                
                # Determine angles around parent pad for attachment points - favor upward growth
                # Avoid bottom angles (anything more than 120° from vertical up)
                base_angles = np.linspace(-np.pi/3-np.pi/2, np.pi/3-np.pi/2, num_new_pads)
                
                for i in range(num_new_pads):
                    # Attachment angle with slight noise
                    attachment_angle = base_angles[i] + np.random.uniform(-0.15, 0.15)
                    
                    # Calculate attachment position (at edge of parent pad)
                    parent_angle = parent['angle']
                    # Rotate attachment angle relative to parent orientation
                    global_attach_angle = parent_angle + attachment_angle
                    
                    # Position EXACTLY on parent ellipse edge
                    attach_x = parent['x'] + 0.5 * parent['width'] * np.cos(global_attach_angle)
                    attach_y = parent['y'] + 0.5 * parent['height'] * np.sin(global_attach_angle)
                    
                    # New pad grows from attachment point at edge
                    # Pad angle is tangential to the ellipse at attachment point
                    # Tangent angle is perpendicular to radius vector at attachment point
                    tangent_angle = global_attach_angle + np.pi/2
                    # Add slight variation and upward bias
                    upward_bias = -np.pi/6  # Slight upward tendency
                    pad_angle = tangent_angle + upward_bias + np.random.uniform(-np.pi/12, np.pi/12)
                    
                    # Ensure pad doesn't grow downward
                    if np.sin(pad_angle) > 0.3:  # If pad is pointing too downward
                        pad_angle = pad_angle - np.pi/2  # Adjust to point more horizontally
                    
                    # Size decreases less with depth to keep pads larger
                    size_factor = 0.9 - 0.05 * current_depth
                    pad_width = parent['width'] * np.random.uniform(0.75, 0.95) * size_factor
                    pad_height = parent['height'] * np.random.uniform(0.75, 0.95) * size_factor
                    
                    # Calculate new pad center position
                    # Center of the pad is half its height away from attachment point, in pad angle direction
                    pad_x = attach_x + (pad_height/2) * np.cos(pad_angle)
                    pad_y = attach_y + (pad_height/2) * np.sin(pad_angle)
                    
                    # Store attachment point on parent pad
                    parent['attach_points'].append({
                        'x': attach_x,
                        'y': attach_y,
                        'angle': global_attach_angle
                    })
                    
                    # Create the pad
                    pad = {
                        'x': pad_x,
                        'y': pad_y,
                        'width': pad_width,
                        'height': pad_height,
                        'angle': pad_angle,
                        'sway_phase': np.random.uniform(0, 2*np.pi),
                        'sway_amount': np.random.uniform(1.0, 1.5),  # Increased sway for more motion
                        'color_variation': np.random.uniform(-0.03, 0.03),
                        'has_flower': np.random.random() < 0.4,  # 40% chance of having a flower
                        'parent_idx': parent_idx,
                        'depth': current_depth + 1,
                        'attachment': {
                            'x': attach_x,
                            'y': attach_y,
                            'angle': global_attach_angle
                        },
                        'attach_points': []  # Will store attachment points for child pads
                    }
                    
                    # If has flower, add flower properties (smaller flowers)
                    if pad['has_flower']:
                        # Create 1-3 flowers per pad
                        pad['flowers'] = []
                        num_flowers = np.random.randint(1, 3 if pad['depth'] > 1 else 4)
                        
                        # Place flowers around edge of pad
                        flower_angles = np.linspace(0, 2*np.pi, num_flowers, endpoint=False)
                        flower_angles += np.random.uniform(0, np.pi/4)  # Rotate randomly
                        
                        for f_angle in flower_angles:
                            global_flower_angle = pad_angle + f_angle
                            flower_offset = min(pad_width, pad_height) * 0.45  # Place at edge
                            
                            flower = {
                                'x': pad_x + flower_offset * np.cos(global_flower_angle),
                                'y': pad_y + flower_offset * np.sin(global_flower_angle),
                                'size': np.random.uniform(1.5, 3.0),  # Larger flowers to be more visible
                                'color': np.random.choice([0.05, 0.95]),  # Red or orange
                                'sway_phase': np.random.uniform(0, 2*np.pi)
                            }
                            pad['flowers'].append(flower)
                    
                    instate['pads'].append(pad)
            
            # Move to next depth level
            current_depth += 1
        
        # Add some rocks and desert vegetation (simplified)
        instate['decorations'] = []
        num_decorations = np.random.randint(3, 7)
        
        for i in range(num_decorations):
            decoration_type = np.random.randint(0, 2)  # 0 = rock, 1 = desert shrub
            pos_x = np.random.uniform(10, 110)
            pos_y = np.random.uniform(instate['ground_y'], 58)
            
            decoration = {
                'type': decoration_type,
                'pos': np.array([pos_x, pos_y]),
                'size': np.random.uniform(1, 2.5),
                'color_variation': np.random.uniform(-0.05, 0.05)
            }
            instate['decorations'].append(decoration)
        
        # Sort pads by depth for proper drawing order
        instate['pads'].sort(key=lambda x: x['depth'])
        
        # Pre-calculate some values to avoid computational work during animation
        instate['pad_shapes'] = []
        for pad in instate['pads']:
            # For each pad, pre-calculate an approximation of its shape for faster drawing
            instate['pad_shapes'].append({
                'width': pad['width'], 
                'height': pad['height'],
                'cos_angle': np.cos(-pad['angle']),
                'sin_angle': np.sin(-pad['angle'])
            })
        
        return

    if instate['count'] == -1:
        outstate['has_cactus'] = False
        outstate['render'][instate['frame_id']].remove_image_plane(instate['cactus_plane'])
        return

    current_time = time.time()
    dt = min(current_time - instate['last_update'], 0.1)  # Cap dt to avoid large jumps
    elapsed_time = current_time - instate['start_time']
    total_duration = instate.get('duration', 60.0)
    
    # Calculate fade factor
    fade_duration = 4.0
    if elapsed_time < fade_duration:
        fade_factor = elapsed_time / fade_duration
    elif elapsed_time > (total_duration - fade_duration):
        fade_factor = (total_duration - elapsed_time) / fade_duration
    else:
        fade_factor = 1.0
    fade_factor = np.clip(fade_factor, 0, 1)
    
    # Seasonal wind strength
    season = outstate['season']
    s_wind = (1 - 0.5 * np.cos(np.pi * 2 * (season - 0.125)))
    
    # Update wind parameters
    wind = instate['wind']
    if current_time - wind['change_time'] > wind['change_interval']:
        wind['target_strength'] = np.random.uniform(0.2, 0.4) * s_wind  # Increased for more visible motion
        wind['change_time'] = current_time
        wind['change_interval'] = np.random.uniform(3, 7)
    
    # Smoothly adjust wind strength
    wind['strength'] += (wind['target_strength'] - wind['strength']) * min(dt * 0.5, 0.1)
    
    # Update sway phases for all pads
    for pad in instate['pads']:
        pad['sway_phase'] += dt * (0.8 + 0.3 * pad['depth'])  # Deeper pads sway faster
    
    # Clear the window
    window = instate['cactus_window']
    window.fill(0)
    
    # Get coordinate grids
    y_indices = instate['y_indices']
    x_indices = instate['x_indices']
    
    # Draw ground with transparent sky (vectorized operation)
    ground_mask = y_indices >= instate['ground_y']
    window[ground_mask, 0] = instate['soil_color']  # Hue
    window[ground_mask, 1] = 0.5  # Saturation
    window[ground_mask, 2] = 0.3  # Value
    window[ground_mask, 3] = fade_factor  # Alpha
    
    # Draw desert decorations - process all at once for performance
    for decoration in instate['decorations']:
        dec_x, dec_y = decoration['pos']
        # Limit calculation to a region around the decoration (optimization)
        size = decoration['size']
        x_min = max(0, int(dec_x - size - 1))
        x_max = min(120, int(dec_x + size + 1))
        y_min = max(0, int(dec_y - size - 1))
        y_max = min(60, int(dec_y + size + 1))
        
        if x_min < x_max and y_min < y_max:  # Make sure region is valid
            sub_x = x_indices[y_min:y_max, x_min:x_max]
            sub_y = y_indices[y_min:y_max, x_min:x_max]
            
            dec_distance = np.sqrt((sub_x - dec_x)**2 + (sub_y - dec_y)**2)
            dec_mask = dec_distance < size
            
            if decoration['type'] == 0:  # Rock
                window[y_min:y_max, x_min:x_max][dec_mask, 0] = 0.05 + decoration['color_variation']
                window[y_min:y_max, x_min:x_max][dec_mask, 1] = 0.2
                window[y_min:y_max, x_min:x_max][dec_mask, 2] = 0.4
                window[y_min:y_max, x_min:x_max][dec_mask, 3] = fade_factor
            else:  # Desert shrub
                window[y_min:y_max, x_min:x_max][dec_mask, 0] = 0.25 + decoration['color_variation']
                window[y_min:y_max, x_min:x_max][dec_mask, 1] = 0.4
                window[y_min:y_max, x_min:x_max][dec_mask, 2] = 0.3
                window[y_min:y_max, x_min:x_max][dec_mask, 3] = fade_factor
    
    # Calculate all pad positions with sway
    pad_positions = []
    
    # First calculate new positions for all pads (with wind effects)
    for i, pad in enumerate(instate['pads']):
        # Base sway from pad's own motion
        pad_sway = np.sin(pad['sway_phase']) * wind['strength'] * pad['sway_amount']
        
        if pad['parent_idx'] == -1:
            # Base pad sway
            pad_x = pad['x'] + pad_sway * 2.0
            pad_y = pad['y'] + pad_sway * 0.5
            pad_angle = pad['angle'] + pad_sway * 0.1  # More rotation
            
            # No attachment points to update for base pad
            attachment_points = []
        else:
            # Get parent position from the updated positions
            parent_pos = pad_positions[pad['parent_idx']]
            parent_x = parent_pos['x']
            parent_y = parent_pos['y']
            parent_angle = parent_pos['angle']
            
            # Calculate attachment point relative to parent center
            parent_pad = instate['pads'][pad['parent_idx']]  # Get the actual parent pad
            attach_vector_x = pad['attachment']['x'] - parent_pad['x']
            attach_vector_y = pad['attachment']['y'] - parent_pad['y']
            
            # Rotate attachment vector by parent's new angle
            angle_diff = parent_angle - parent_pad['angle']
            rotated_x = attach_vector_x * np.cos(angle_diff) - attach_vector_y * np.sin(angle_diff)
            rotated_y = attach_vector_x * np.sin(angle_diff) + attach_vector_y * np.cos(angle_diff)
            
            # Updated attachment point
            attach_x = parent_x + rotated_x
            attach_y = parent_y + rotated_y
            
            # Vector from attachment to pad center - keep same relative position
            vector_x = pad['x'] - pad['attachment']['x']
            vector_y = pad['y'] - pad['attachment']['y']
            vector_length = np.sqrt(vector_x**2 + vector_y**2)
            
            # Rotate by parent angle change plus this pad's sway
            total_angle_change = angle_diff + pad_sway * 0.15
            pad_angle = pad['angle'] + total_angle_change
            
            # New position based on attachment point
            pad_x = attach_x + vector_length * np.cos(pad_angle)
            pad_y = attach_y + vector_length * np.sin(pad_angle)
            
            # Update attachment points for this pad
            attachment_points = []
            for attach_point in pad.get('attach_points', []):
                # Calculate relative to original pad center
                rel_x = attach_point['x'] - pad['x']
                rel_y = attach_point['y'] - pad['y']
                
                # Rotate and position relative to new pad center
                new_attach_x = pad_x + rel_x * np.cos(total_angle_change) - rel_y * np.sin(total_angle_change)
                new_attach_y = pad_y + rel_x * np.sin(total_angle_change) + rel_y * np.cos(total_angle_change)
                
                attachment_points.append({
                    'x': new_attach_x,
                    'y': new_attach_y,
                    'angle': attach_point['angle'] + total_angle_change
                })
        
        # Store updated position
        pad_positions.append({
            'x': pad_x,
            'y': pad_y,
            'angle': pad_angle,
            'attachment_points': attachment_points
        })
    
    # Simplified drawing for better performance - draw pads in proper depth order
    for i, pad in enumerate(instate['pads']):
        pos = pad_positions[i]
        shape = instate['pad_shapes'][i]
        
        # Skip if far off screen
        if (pos['x'] < -shape['width']*2 or pos['x'] >= 120+shape['width']*2 or 
            pos['y'] < -shape['height']*2 or pos['y'] >= 60+shape['height']*2):
            continue
        
        # Calculate region around pad
        width = shape['width']
        height = shape['height']
        x_min = max(0, int(pos['x'] - width - 2))
        x_max = min(120, int(pos['x'] + width + 2))
        y_min = max(0, int(pos['y'] - height - 2))
        y_max = min(60, int(pos['y'] + height + 2))
        
        if x_min >= x_max or y_min >= y_max:
            continue
        
        # Get the subset of coordinate arrays
        sub_x = x_indices[y_min:y_max, x_min:x_max]
        sub_y = y_indices[y_min:y_max, x_min:x_max]
        
        # Get coordinates relative to pad center
        dx = sub_x - pos['x']
        dy = sub_y - pos['y']
        
        # Apply current angle
        cos_angle = np.cos(-pos['angle'])
        sin_angle = np.sin(-pos['angle'])
        rot_x = dx * cos_angle - dy * sin_angle
        rot_y = dx * sin_angle + dy * cos_angle
        
        # Create pad shape (ellipse)
        ellipse_val = (rot_x / (width/2))**2 + (rot_y / (height/2))**2
        #pad_mask = ellipse_val <= 1.0
        
        # Create edge mask (only 1 pixel wide)
        edge_mask = (ellipse_val > 0.92) & (ellipse_val <= 1.0)
        inner_mask = ellipse_val <= 0.92
        
        # Apply cactus color - more vibrant green
        window[y_min:y_max, x_min:x_max][inner_mask, 0] = 0.33 + pad['color_variation']  # Green hue
        window[y_min:y_max, x_min:x_max][inner_mask, 1] = 0.8  # High saturation
        window[y_min:y_max, x_min:x_max][inner_mask, 2] = 0.5  # Medium-high brightness
        window[y_min:y_max, x_min:x_max][inner_mask, 3] = fade_factor
        
        # Edge is slightly darker green (just 1 pixel)
        window[y_min:y_max, x_min:x_max][edge_mask, 0] = 0.31 + pad['color_variation']  # Slightly darker
        window[y_min:y_max, x_min:x_max][edge_mask, 1] = 0.7  # Slightly less saturated
        window[y_min:y_max, x_min:x_max][edge_mask, 2] = 0.4  # Slightly darker
        window[y_min:y_max, x_min:x_max][edge_mask, 3] = fade_factor
        
        # Draw spine clusters more efficiently - just around edge
        spine_density = min(20, int(width + height))
        # Generate all angles at once
        t_values = np.linspace(0, 2*np.pi, spine_density, endpoint=False)
        # Calculate all unit circle coordinates at once
        cx_values = np.cos(t_values)
        cy_values = np.sin(t_values)

        # Calculate all spine positions at once
        cos_angle = np.cos(pos['angle'])
        sin_angle = np.sin(pos['angle'])
        spine_x_values = pos['x'] + (width/2) * cx_values * cos_angle - (height/2) * cy_values * sin_angle
        spine_y_values = pos['y'] + (width/2) * cx_values * sin_angle + (height/2) * cy_values * cos_angle

        # Convert to integers for pixel positions
        spine_x_int = np.round(spine_x_values).astype(int)
        spine_y_int = np.round(spine_y_values).astype(int)

        # Create mask for valid positions
        valid_mask = (0 <= spine_x_int) & (spine_x_int < 120) & (0 <= spine_y_int) & (spine_y_int < 60)

        # Apply color to all valid spine positions at once
        valid_spine_x = spine_x_int[valid_mask]
        valid_spine_y = spine_y_int[valid_mask]

        # Only access valid indices
        if len(valid_spine_x) > 0:
            window[valid_spine_y, valid_spine_x, 0] = 0.15  # Yellow-white hue
            window[valid_spine_y, valid_spine_x, 1] = 0.3   # Low saturation
            window[valid_spine_y, valid_spine_x, 2] = 0.9   # High value
            window[valid_spine_y, valid_spine_x, 3] = fade_factor
                
        # Draw flowers for this pad
        if pad['has_flower']:
            flowers = pad['flowers']
            if flowers:  # Skip if no flowers
                # Pre-calculate constants used for all flowers
                angle_diff = pos['angle'] - pad['angle']
                cos_angle = np.cos(angle_diff)
                sin_angle = np.sin(angle_diff)
                sway_factor = wind['strength'] * 0.5
                
                # Extract arrays of flower properties for vectorized operations
                flower_xs = np.array([f['x'] for f in flowers])
                flower_ys = np.array([f['y'] for f in flowers])
                flower_sizes = np.array([f['size'] for f in flowers])
                flower_colors = np.array([f['color'] for f in flowers])
                flower_phases = np.array([f['sway_phase'] for f in flowers])
                
                # Calculate vectors from old pad center to flowers (all at once)
                flower_vec_xs = flower_xs - pad['x']
                flower_vec_ys = flower_ys - pad['y']
                
                # Rotate all flower vectors at once
                rotated_xs = flower_vec_xs * cos_angle - flower_vec_ys * sin_angle
                rotated_ys = flower_vec_xs * sin_angle + flower_vec_ys * cos_angle
                
                # Calculate new positions for all flowers
                new_flower_xs = pos['x'] + rotated_xs
                new_flower_ys = pos['y'] + rotated_ys
                
                # Calculate sway for all flowers at once
                flower_sways = np.sin(flower_phases + elapsed_time) * sway_factor
                new_flower_xs += flower_sways * 0.3
                new_flower_ys += flower_sways * 0.2
                
                # Draw all flowers (still have to loop, but with fewer calculations)
                for i in range(len(flowers)):
                    draw_flower(
                        window, x_indices, y_indices,
                        new_flower_xs[i], new_flower_ys[i],
                        flower_sizes[i],
                        flower_colors[i],
                        fade_factor
                    )
    
    # Convert HSVA to RGBA for rendering
    rgb = color.hsv2rgb(window[..., 0:3])
    alpha = window[..., 3:4]
    rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)
    
    # Update the image plane
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['cactus_plane'],
        rgb_out[:,:,:]
    )
    
    # Update timestamp
    instate['last_update'] = current_time

def draw_flower(window, x_indices, y_indices, center_x, center_y, size, color_hue, fade_factor):
    """Draw a flower more visibly"""
    # Skip if out of bounds 
    if (center_x < 0 or center_x >= 120 or center_y < 0 or center_y >= 60 or size < 0.5):
        return
    
    # Calculate region around flower for optimization
    x_min = max(0, int(center_x - size - 1))
    x_max = min(120, int(center_x + size + 1))
    y_min = max(0, int(center_y - size - 1))
    y_max = min(60, int(center_y + size + 1))
    
    if x_min >= x_max or y_min >= y_max:
        return
    
    # Get coordinate arrays
    sub_x = x_indices[y_min:y_max, x_min:x_max]
    sub_y = y_indices[y_min:y_max, x_min:x_max]
    
    # Distance from center
    dist = np.sqrt((sub_x - center_x)**2 + (sub_y - center_y)**2)
    
    # Create petal mask and center mask
    center_mask = dist < (size * 0.3)
    petal_mask = dist < size
    
    # Draw petals (whole flower)
    window[y_min:y_max, x_min:x_max][petal_mask, 0] = color_hue  # Red or orange
    window[y_min:y_max, x_min:x_max][petal_mask, 1] = 0.9        # High saturation
    window[y_min:y_max, x_min:x_max][petal_mask, 2] = 0.8        # Bright
    window[y_min:y_max, x_min:x_max][petal_mask, 3] = fade_factor
    
    # Draw yellow center
    window[y_min:y_max, x_min:x_max][center_mask, 0] = 0.15  # Yellow
    window[y_min:y_max, x_min:x_max][center_mask, 1] = 0.9   # High saturation
    window[y_min:y_max, x_min:x_max][center_mask, 2] = 0.9   # Very bright
    window[y_min:y_max, x_min:x_max][center_mask, 3] = fade_factor



def dancing_cactuses(instate, outstate):
    if instate['count'] == 0:
        instate['cactus_rng'] = np.random.RandomState(int(time.time()))
        outstate['has_cactus'] = True
        # Initialize cactus scene parameters
        instate['cactus_window'] = np.zeros((60, 120, 4))  # HSVA format
        instate['cactus_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((60, 120, 4), dtype=np.uint8),
            position=(0, 0, 19.5),  # Place in front of background but behind other elements
            rotation=(0, 0, 0),
            scale=(3, 3)
        )
        
        # Initialize timing parameters
        instate['start_time'] = time.time()
        instate['last_update'] = time.time()
        instate['last_render'] = 0  # Track when we last did a full render
        instate['render_interval'] = 0.033  # ~30fps, adjust as needed
        
        # Wind parameters
        instate['windC'] = {
            'direction': instate['cactus_rng'].uniform(0, 2*np.pi),
            'strength': 0,
            'target_strength': instate['cactus_rng'].uniform(0.3, 0.8),
            'change_time': time.time(),
            'change_interval': instate['cactus_rng'].uniform(3, 7)
        }
        
        # Set ground level at bottom 1/4 of screen
        instate['ground_y'] = 50  # Bottom 1/4 of screen (60 * 3/4 = 45)
        instate['sand_color'] = instate['cactus_rng'].uniform(0.05, 0.12)  # Darker sand for night
        
        # Pre-calculate coordinate grids once
        y, x = np.mgrid[0:60, 0:120]
        instate['y_indices'] = y
        instate['x_indices'] = x
        
        # Create ONE cactus in the center
        instate['cactuses'] = []
        
        # Base position - centered
        base_x = 60 + np.random.randint(-15, 16)  # Center of screen
        base_y = instate['ground_y']  # Base at ground level
        
        # Create cactus data structure
        cactus = {
            'pos': np.array([base_x, base_y]),
            'size': 7.5,  # Base width
            'color_variation': instate['cactus_rng'].uniform(-0.03, 0.03),  # Slight color variation
            'sway_phase': instate['cactus_rng'].uniform(0, 2*np.pi),  # Random starting phase
            'sway_amplitude': 3.0,  # Sway intensity
            'last_blink': time.time(),
            'blink_interval': instate['cactus_rng'].uniform(3, 8),
            'is_blinking': False,
            'blink_duration': 0.2,
            # Eye movement parameters (like in eye function)
            'eye_target_x': 0,
            'eye_target_y': 0,
            'eye_current_x': 0,
            'eye_current_y': 0,
            'last_eye_movement_time': time.time(),
            'eye_movement_interval': instate['cactus_rng'].uniform(2.0, 4.0),
            # New eye blinking parameters with animation
            'eye_blink_state': 'open',      # States: open, closing, closed, opening
            'eye_blink_start_time': time.time(),
            'eye_blink_progress': 0.0,      # 0 = fully open, 1 = fully closed
            'eye_close_duration': 0.15,     # Time to close the eye
            'eye_open_duration': 0.2,       # Time to open the eye (slightly slower)
            'eye_closed_duration': 0.1,     # Time eye stays fully closed  
            'eye_open_interval': instate['cactus_rng'].uniform(5, 10),  # Time between blinks
            # Texture noise
            'texture_seed': instate['cactus_rng'].randint(0, 1000)
        }
        
        # Taller cactus - reaching close to top of screen
        cactus['height'] = 42  # From ground to about 10-15 pixels from top
        
        # Configure two arms (one on each side)
        cactus['arms'] = [
            # Left arm
            {
                'side': 'left',
                'height_ratio': 0.45,  # About 45% from the top
                'out_length': 12,      # Horizontal segment length
                'up_length': 14,       # Vertical segment length
                'sway_phase': instate['cactus_rng'].uniform(0, 2*np.pi),
                'texture_seed': instate['cactus_rng'].randint(0, 1000)  # For arm texture
            },
            # Right arm
            {
                'side': 'right',
                'height_ratio': 0.35,  # About 35% from the top (slightly higher)
                'out_length': 14,      # Horizontal segment length
                'up_length': 16,       # Vertical segment length
                'sway_phase': instate['cactus_rng'].uniform(0, 2*np.pi),
                'texture_seed': instate['cactus_rng'].randint(0, 1000)  # For arm texture
            }
        ]
        
        # Face position near top of cactus
        cactus['face_y'] = base_y - cactus['height'] * 0.75  # Face at 75% of height
        
        instate['cactuses'].append(cactus)
        
        # Add some random rocks and small desert plants
        instate['decorations'] = []
        num_decorations = instate['cactus_rng'].randint(5, 10)
        
        for _ in range(num_decorations):
            decoration_type = instate['cactus_rng'].randint(0, 2)  # 0 = rock, 1 = small plant
            decoration = {
                'type': decoration_type,
                'pos': np.array([
                    instate['cactus_rng'].uniform(10, 110),
                    instate['cactus_rng'].uniform(instate['ground_y'], 58)
                ]),
                'size': instate['cactus_rng'].uniform(1, 3)
            }
            instate['decorations'].append(decoration)
            
        # Create texture noise once for reuse
        texture_seed = cactus['texture_seed']
        instate['cactus_rng'].seed(texture_seed)
        instate['cactus_texture_noise'] = instate['cactus_rng'].uniform(-0.08, 0.08, (60, 120))
        
        for arm in cactus['arms']:
            instate['cactus_rng'].seed(arm['texture_seed'])
            arm['arm_texture'] = instate['cactus_rng'].uniform(-0.08, 0.08, (60, 120))
            
        return

    if instate['count'] == -1:
        outstate['has_cactus'] = False
        outstate['render'][instate['frame_id']].remove_image_plane(instate['cactus_plane'])
        return

    current_time = time.time()
    dt = current_time - instate['last_update']
    elapsed_time = current_time - instate['start_time']
    total_duration = instate['duration']
    
    # Throttle updates and rendering
    should_render = (current_time - instate['last_render']) >= instate['render_interval']
    
    # Always update state logic even when not rendering
    # Calculate fade factor
    fade_duration = 4.0
    if elapsed_time < fade_duration:
        fade_factor = elapsed_time / fade_duration
    elif elapsed_time > (total_duration - fade_duration):
        fade_factor = (total_duration - elapsed_time) / fade_duration
    else:
        fade_factor = 1.0
    fade_factor = np.clip(fade_factor, 0, 1)

    # Update wind parameters
    wind = instate['windC']
    if current_time - wind['change_time'] > wind['change_interval']:
        wind['target_strength'] = instate['cactus_rng'].uniform(0.3, 0.8)
        wind['change_time'] = current_time
        wind['change_interval'] = instate['cactus_rng'].uniform(3, 7)
    
    # Smoothly adjust wind strength
    wind['strength'] += (wind['target_strength'] - wind['strength']) * dt * 0.5
    
    # Update cactus states
    cactus = instate['cactuses'][0]  # Just one cactus
    
    # Update sway phase
    cactus['sway_phase'] += dt * 0.8
    
    # Update pupil blinking (affects pupil size)
    if cactus['is_blinking']:
        if current_time - cactus['last_blink'] > cactus['blink_duration']:
            cactus['is_blinking'] = False
            cactus['last_blink'] = current_time
    elif current_time - cactus['last_blink'] > cactus['blink_interval']:
        cactus['is_blinking'] = True
        cactus['last_blink'] = current_time
    
    # Update eye blinking animation state
    blink_elapsed = current_time - cactus['eye_blink_start_time']
    
    if cactus['eye_blink_state'] == 'open':
        if blink_elapsed > cactus['eye_open_interval']:
            # Start closing
            cactus['eye_blink_state'] = 'closing'
            cactus['eye_blink_start_time'] = current_time
            cactus['eye_blink_progress'] = 0.0
    
    elif cactus['eye_blink_state'] == 'closing':
        # Update progress based on close duration
        cactus['eye_blink_progress'] = blink_elapsed / cactus['eye_close_duration']
        if cactus['eye_blink_progress'] >= 1.0:
            # Transition to closed state
            cactus['eye_blink_state'] = 'closed'
            cactus['eye_blink_start_time'] = current_time
            cactus['eye_blink_progress'] = 1.0
    
    elif cactus['eye_blink_state'] == 'closed':
        if blink_elapsed > cactus['eye_closed_duration']:
            # Start opening
            cactus['eye_blink_state'] = 'opening'
            cactus['eye_blink_start_time'] = current_time
    
    elif cactus['eye_blink_state'] == 'opening':
        # Update progress based on open duration (1.0 to 0.0)
        open_progress = blink_elapsed / cactus['eye_open_duration']
        cactus['eye_blink_progress'] = 1.0 - open_progress
        if cactus['eye_blink_progress'] <= 0.0:
            # Transition back to open state
            cactus['eye_blink_state'] = 'open'
            cactus['eye_blink_start_time'] = current_time
            cactus['eye_blink_progress'] = 0.0
            # Randomize next interval
            cactus['eye_open_interval'] = instate['cactus_rng'].uniform(5, 10)
    
    # Update arm sway phases
    for arm in cactus['arms']:
        arm['sway_phase'] += dt * 0.5
        
    # Update eye movement (similar to eye function)
    if current_time - cactus['last_eye_movement_time'] > cactus['eye_movement_interval']:
        # Generate new random target for eye to look - INCREASED RANGE
        angle = instate['cactus_rng'].random() * 2 * np.pi
        max_radius = 1.1  # Increased from 0.6 to 1.1 (beyond sclera bounds)
        r = instate['cactus_rng'].random() * max_radius
        
        # Scale x and y differently for elliptical movement
        cactus['eye_target_x'] = r * np.cos(angle) * 1.2
        cactus['eye_target_y'] = r * np.sin(angle)
        
        cactus['last_eye_movement_time'] = current_time
        cactus['eye_movement_interval'] = instate['cactus_rng'].uniform(2.0, 4.0)
    
    # Smoothly move eye position toward target
    dxC = cactus['eye_target_x'] - cactus['eye_current_x']
    dyC = cactus['eye_target_y'] - cactus['eye_current_y']
    distance = np.sqrt(dxC*dxC + dyC*dyC)
    
    if distance > 0.001:
        # Movement speed factor
        movement_speed = 2.0
        
        # Calculate movement amount based on time and speed
        move_amount = min(distance, movement_speed * dt)
        
        if distance > 0:  # Avoid division by zero
            cactus['eye_current_x'] += (dxC / distance) * move_amount
            cactus['eye_current_y'] += (dyC / distance) * move_amount
    
    # Skip rendering if not needed
    if not should_render:
        instate['last_update'] = current_time
        return
    
    # Clear the window - only if we're rendering
    instate['cactus_window'].fill(0)
    window = instate['cactus_window']
    
    # Helper function to get coordinates grid (cache these values)
    y_indices = instate['y_indices']
    x_indices = instate['x_indices']
    
    # Draw ground with transparent sky
    ground_mask = y_indices >= instate['ground_y']
    window[ground_mask, 0] = instate['sand_color']  # Hue
    window[ground_mask, 1] = 0.5  # Saturation
    window[ground_mask, 2] = 0.3  # Value
    window[ground_mask, 3] = fade_factor  # Alpha
    
    # Draw decorations (these rarely change, so we could cache them too)
    for decoration in instate['decorations']:
        dec_x, dec_y = decoration['pos']
        # Use vectorized operations for distance calculations
        dec_distance = np.sqrt((x_indices - dec_x)**2 + (y_indices - dec_y)**2)
        dec_mask = dec_distance < decoration['size']
        
        if decoration['type'] == 0:  # Rock
            window[dec_mask, 0] = 0  # Hue (gray)
            window[dec_mask, 1] = 0.1  # Low saturation for gray
            window[dec_mask, 2] = 0.4  # Value
            window[dec_mask, 3] = fade_factor  # Alpha
        else:  # Small plant/shrub
            window[dec_mask, 0] = 0.3  # Hue (green)
            window[dec_mask, 1] = 0.6  # Saturation
            window[dec_mask, 2] = 0.3  # Value
            window[dec_mask, 3] = fade_factor  # Alpha
    
    # Calculate sway offset based on wind
    sway_offset = np.sin(cactus['sway_phase']) * wind['strength'] * cactus['sway_amplitude'] * 3
    
    # Draw main body
    cactus_base_x, cactus_base_y = cactus['pos']
    height = cactus['height']
    width = cactus['size']
    
    # Use cached texture noise
    texture_noise = instate['cactus_texture_noise']
    
    # Simple approach to draw cactus body
    segments = min(int(height), 40)  # Limit number of segments
    for i in range(segments):
        y_offset = -height + i * (height / segments)
        
        # Calculate sway at this height
        height_ratio = abs(y_offset) / height
        local_sway = sway_offset * height_ratio * height_ratio
        
        center_x = cactus_base_x + local_sway
        center_y = cactus_base_y + y_offset
        
        # Skip if out of bounds
        if not (0 <= center_y < 60):
            continue
            
        # Calculate width at this height
        local_width = width * (0.8 + 0.2 * (1 - height_ratio))#* (1 - (height_ratio**4)/5)
        
        # Calculate distance for slice
        dx = x_indices - center_x
        dy = y_indices - center_y
        dist_squared = (dx**2 / local_width**2) + (dy**2 / 4)
        segment_mask = dist_squared <= 1
        
        if np.any(segment_mask):
            # Create texture pattern
            base_hue = 0.33 + cactus['color_variation']
            texture_y = np.sin(x_indices * 0.8 + 123) * 0.05
            texture_x = np.cos(y_indices * 0.5 + 369) * 0.04
            # Get mask indices for indexing
            ys, xs = np.where(segment_mask)
            
            # Extract the pre-calculated texture components for the masked pixels
            masked_texture_y = texture_y[ys, xs]
            masked_texture_x = texture_x[ys, xs]
            masked_texture_noise = texture_noise[ys, xs]
            
            # Combine noise components for all pixels at once
            combined_noise = masked_texture_noise + masked_texture_y + masked_texture_x
            
            # Apply to all pixels at once
            window[ys, xs, 0] = base_hue + combined_noise * 0.1
            window[ys, xs, 1] = 0.6 + combined_noise * 0.2
            window[ys, xs, 2] = 0.4 + combined_noise * 0.3
            window[ys, xs, 3] = fade_factor
    
    # Draw arms with 90-degree bend (out then up)
    for arm in cactus['arms']:
        # Calculate arm start position
        arm_height = arm['height_ratio'] * height
        arm_base_y = cactus_base_y - arm_height
        
        # Calculate sway at arm height
        height_ratio = arm_height / height
        body_sway = sway_offset * height_ratio * height_ratio
        arm_sway = np.sin(arm['sway_phase']) * wind['strength'] * 2.0
        
        # Determine horizontal direction based on side
        if arm['side'] == 'left':
            direction = -1  # Left side
            arm_base_x = cactus_base_x - (width/2) * 0.9 + body_sway
        else:  # right side
            direction = 1   # Right side
            arm_base_x = cactus_base_x + (width/2) * 0.9 + body_sway
        
        # Calculate elbow position (after horizontal segment)
        elbow_x = arm_base_x + direction * arm['out_length']
        elbow_y = arm_base_y
        
        # Apply slight sway to elbow
        elbow_x += arm_sway
        
        # Use cached arm texture
        arm_texture = arm['arm_texture']
        
        # Draw horizontal segment (out from trunk)
# Draw horizontal segment (out from trunk)
        h_segment_points = 5

        # Pre-calculate all points at once
        t_values = np.linspace(0, 1, h_segment_points)
        segment_xs = arm_base_x + direction * arm['out_length'] * t_values + arm_sway * t_values
        thicknesses = width * 0.6 * (1 - 0.2 * t_values)

        # Draw all segments in a single pass
        for i, (segment_x, thickness) in enumerate(zip(segment_xs, thicknesses)):
            segment_y = arm_base_y
            
            # Skip if out of bounds
            if not (0 <= segment_y < 60):
                continue
            
            # Define region to process (avoids unnecessary calculations)
            region_width = thickness * 2
            x_min = max(0, int(segment_x - region_width))
            x_max = min(120, int(segment_x + region_width + 1))
            y_min = max(0, int(segment_y - 2))
            y_max = min(60, int(segment_y + 2))
            
            if x_min >= x_max or y_min >= y_max:
                continue
            
            # Process only the relevant region
            sub_x = x_indices[y_min:y_max, x_min:x_max]
            sub_y = y_indices[y_min:y_max, x_min:x_max]
            
            # Calculate distance efficiently
            dx = sub_x - segment_x
            dy = sub_y - segment_y
            dist_squared = (dx**2 / thickness**2) + (dy**2 / 4)
            segment_mask = dist_squared <= 1
            
            if np.any(segment_mask):
                # Get base hue
                base_hue = 0.33 + cactus['color_variation']
                
                # Create texture pattern for the masked region
                masked_y, masked_x = np.where(segment_mask)
                global_y = masked_y + y_min  # Convert to global coordinates
                global_x = masked_x + x_min
                
                # Get actual texture values directly
                sin_pattern = np.sin(global_x * 0.5 + global_y * 0.3 + arm['texture_seed']) * 0.06
                texture_values = arm_texture[global_y, global_x] * 0.5
                texture_pattern = sin_pattern + texture_values
                
                # Apply to window all at once
                window[global_y, global_x, 0] = base_hue + texture_pattern * 0.1
                window[global_y, global_x, 1] = 0.6 + texture_pattern * 0.2
                window[global_y, global_x, 2] = 0.4 + texture_pattern * 0.3
                window[global_y, global_x, 3] = fade_factor
        
        # Draw vertical segment (up from elbow)
        # Draw vertical segment (up from elbow)
        v_segment_points = 7

        # Pre-calculate all segment positions and properties at once
        t_values = np.linspace(0, 1, v_segment_points)
        segment_xs = elbow_x + arm_sway * 0.2 * (1 + t_values)
        segment_ys = elbow_y - arm['up_length'] * t_values
        thicknesses = width * 0.6 * (1 - 0.3 * t_values)

        # Process all segments with improved region-based calculation
        for i, (segment_x, segment_y, thickness) in enumerate(zip(segment_xs, segment_ys, thicknesses)):
            # Skip if out of bounds
            if not (0 <= segment_y < 60):
                continue
            
            # Define region to process (avoids unnecessary calculations)
            region_width = thickness * 2
            x_min = max(0, int(segment_x - region_width))
            x_max = min(120, int(segment_x + region_width + 1))
            y_min = max(0, int(segment_y - 2))
            y_max = min(60, int(segment_y + 2))
            
            if x_min >= x_max or y_min >= y_max:
                continue
            
            # Process only the relevant region
            sub_x = x_indices[y_min:y_max, x_min:x_max]
            sub_y = y_indices[y_min:y_max, x_min:x_max]
            
            # Calculate distance efficiently
            dx = sub_x - segment_x
            dy = sub_y - segment_y
            dist_squared = (dx**2 / thickness**2) + (dy**2 / 4)
            segment_mask = dist_squared <= 1
            
            if np.any(segment_mask):
                # Get base hue
                base_hue = 0.33 + cactus['color_variation']
                
                # Get global coordinates of the masked region
                masked_y, masked_x = np.where(segment_mask)
                global_y = masked_y + y_min  # Convert to global coordinates
                global_x = masked_x + x_min
                
                # More efficient texture pattern calculation
                sin_pattern = np.sin(global_y * 0.7 + global_x * 0.2 + arm['texture_seed']) * 0.06
                texture_values = arm_texture[global_y, global_x] * 0.5
                texture_pattern = sin_pattern + texture_values
                
                # Apply colors in one operation
                window[global_y, global_x, 0] = base_hue + texture_pattern * 0.1
                window[global_y, global_x, 1] = 0.6 + texture_pattern * 0.2
                window[global_y, global_x, 2] = 0.4 + texture_pattern * 0.3
                window[global_y, global_x, 3] = fade_factor
    
    # Draw eye
    face_x = cactus_base_x
    face_y = cactus['face_y']
    
    # Add sway to face position
    height_ratio = (cactus_base_y - face_y) / height
    face_x += sway_offset * height_ratio * height_ratio
    
    # Calculate eye size
    eye_size = width * 0.75
    eye_y = face_y
    
    eye_center_x = face_x
    eye_center_y = eye_y
    eye_radius_x = eye_size * 1.2
    eye_radius_y = eye_size * (1.0 - 0.9 * cactus['eye_blink_progress'])*0.8
    
    # Draw elliptical eye
    dx = x_indices - eye_center_x
    dy = y_indices - eye_center_y
    eye_dist_squared = (dx / eye_radius_x)**2 + (dy / eye_radius_y)**2
    eye_mask = eye_dist_squared <= 1
    
    # Draw eye white
    if np.any(eye_mask):
        # Create a separate mask for the eyelid line
        eyelid_line = (cactus['eye_blink_progress'] > 0.9) & (np.abs(y_indices - eye_center_y) < 2)
        
        # Combine with eye mask to get two distinct regions
        eyelid_mask = eye_mask & eyelid_line
        eye_white_mask = eye_mask & ~eyelid_line
        
        # Set colors for each region in one operation
        if np.any(eyelid_mask):
            window[eyelid_mask] = [0, 0, 0, fade_factor * 0.8]  # Black eyelid
        
        if np.any(eye_white_mask):
            window[eye_white_mask] = [0.6, 0.5, 0.1, fade_factor * 0.8]  # White of eye
    
    # Only draw iris and pupil if eye is at least 30% open
    if cactus['eye_blink_progress'] < 0.7:
        # Adjust iris visibility based on eye openness
        iris_visibility = 1.0 - (cactus['eye_blink_progress'] / 0.7)
        
        # Calculate iris position with movement
        max_x_offset = eye_size * 0.8
        max_y_offset = eye_size * 0.6 * (1.0 - cactus['eye_blink_progress'])
        
        iris_x = eye_center_x + (cactus['eye_current_x'] * max_x_offset)
        iris_y = eye_center_y + (cactus['eye_current_y'] * max_y_offset)
        
        # Calculate distortion based on position from center
        h_stretch = 1.0
        v_stretch = 1.0 + (0.3 * abs(cactus['eye_current_x']))
        
        # Adjust iris size based on eye openness
        iris_radius = eye_size * 0.9 * iris_visibility
        
        # Calculate distance to iris center
        dx_iris = x_indices - iris_x
        dy_iris = y_indices - iris_y
        
        # Apply distortion to distance calculation
        iris_dist = np.sqrt((dx_iris/h_stretch)**2 + (dy_iris/v_stretch)**2)
        
        # Create iris mask within eye boundary
        iris_mask = (iris_dist <= iris_radius) & eye_mask
        
        if np.any(iris_mask):
            # Get the masked arrays directly
            masked_dy_iris = dy_iris[iris_mask]
            masked_dx_iris = dx_iris[iris_mask]
            masked_dist_ratio = iris_dist[iris_mask] / iris_radius
            
            # Calculate angles and patterns in a vectorized way
            angles = np.arctan2(masked_dy_iris/v_stretch, masked_dx_iris/h_stretch)
            patterns = (np.sin(angles * 8) * 0.1) + (masked_dist_ratio * 0.2)
            
            # Apply colors all at once
            window[iris_mask, 0] = 0.55 + patterns  # Hue (blue-green)
            window[iris_mask, 1] = 0.7              # Saturation
            window[iris_mask, 2] = 0.6              # Value
            window[iris_mask, 3] = fade_factor * iris_visibility  # 
                
        # Calculate pupil size with variations
        base_pupil_size = 0.6
        breathing_variation = np.sin(current_time * 1.5) * 0.1
        
        # Add separate blink effect for pupil
        blink_variation = -0.3 if cactus['is_blinking'] else 0
        
        # Combine all variations
        pupil_size = base_pupil_size + breathing_variation + blink_variation
        pupil_size = np.clip(pupil_size, 0.3, 1.0)
        
        # Draw pupil
        pupil_radius = iris_radius * pupil_size * 0.7
        pupil_mask = (iris_dist <= pupil_radius) & eye_mask
        
        if np.any(pupil_mask):
            ys, xs = np.where(pupil_mask)
            
            for idx in range(len(ys)):
                y, x = ys[idx], xs[idx]
                window[y, x] = [0, 0, 0, fade_factor * iris_visibility]  # Black pupil
        
        # Only draw highlight if eye is mostly open
        if cactus['eye_blink_progress'] < 0.5:
            highlight_offset_x = -pupil_radius * 0.5
            highlight_offset_y = -pupil_radius * 0.5
            highlight_x = iris_x + highlight_offset_x
            highlight_y = iris_y + highlight_offset_y
            highlight_radius = eye_size * 0.15 * (1.0 - cactus['eye_blink_progress'])
            
            # Calculate distance to highlight center
            dx_highlight = x_indices - highlight_x
            dy_highlight = y_indices - highlight_y
            highlight_dist = np.sqrt((dx_highlight/h_stretch)**2 + (dy_highlight/v_stretch)**2)
            
            # Create highlight mask within eye boundary
            highlight_mask = (highlight_dist <= highlight_radius) & eye_mask
            
            if np.any(highlight_mask):
                # Calculate intensity with falloff for all highlighted pixels at once
                masked_highlight_dist = highlight_dist[highlight_mask]
                intensities = 1 - (masked_highlight_dist / highlight_radius)
                
                # Adjust all intensities by eye openness
                adjusted_intensities = intensities * (1.0 - cactus['eye_blink_progress']/0.5)
                
                # Apply to all pixels at once
                window[highlight_mask, 0] = 0  # Hue
                window[highlight_mask, 1] = 0  # Saturation
                window[highlight_mask, 2] = 1.0 * adjusted_intensities  # Value
                window[highlight_mask, 3] = fade_factor * adjusted_intensities  # Alpha
    
    # Convert HSVA to RGBA for rendering
    rgb = color.hsv2rgb(window[..., 0:3])
    alpha = window[..., 3:4]
    rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)
    
    # Update the image plane
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['cactus_plane'],
        rgb_out[:,:,:]
    )
    
    # Update timestamps
    instate['last_update'] = current_time
    instate['last_render'] = current_time


def dancing_barrel_cactus(instate, outstate):
    if instate['count'] == 0:
        # Initialize with simpler structure
        outstate['has_cactus'] = True
        
        # Create image plane - maintain original resolution for quality
        instate['barrel_window'] = np.zeros((60, 120, 4))  # HSVA format
        instate['barrel_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((60, 120, 4), dtype=np.uint8),
            position=(0, 0, 19.5),
            rotation=(0, 0, 0),
            scale=(3, 3)
        )
        
        # Initialize timing parameters
        instate['start_time'] = time.time()
        instate['last_update'] = time.time()
        
        # Wind parameters
        instate['wind'] = {
            'strength': 0,
            'target_strength': np.random.uniform(0.1, 0.3),
            'change_time': time.time(),
            'change_interval': np.random.uniform(3, 7)
        }
        
        # Bounce parameters (small vertical bounce)
        instate['bounce'] = {
            'active': False,
            'start_time': 0,
            'duration': 0,
            'intensity': 0,
            'next_bounce': time.time() + np.random.uniform(15, 30)
        }
        
        # Hop parameters (larger horizontal movement)
        instate['hop'] = {
            'active': False,
            'start_time': 0,
            'duration': 0,
            'distance': 0,
            'direction': 1,  # 1 for right, -1 for left
            'start_x': 0,
            'next_hop': time.time() + np.random.uniform(10, 20),
            'frame_count': 0
        }
        
        # Quick wiggle parameters
        instate['wiggle'] = {
            'active': False,
            'start_time': 0,
            'duration': 0,
            'next_wiggle': time.time() + np.random.uniform(5, 15)
        }
        
        # Ground level
        instate['ground_y'] = 50
        instate['soil_color'] = np.random.uniform(0.05, 0.12)
        
        # Pre-calculate coordinate grids once
        y, x = np.mgrid[0:60, 0:120]
        instate['y_indices'] = y
        instate['x_indices'] = x
        
        # Create a barrel cactus - make sure it's taller than wide
        base_x = 60 + np.random.randint(-15, 16)
        base_y = instate['ground_y']
        
        # Size parameters - explicitly taller than wide
        width = np.random.uniform(12, 16)
        height = width * np.random.uniform(2.0, 2.5)
        
        # Create barrel cactus structure
        instate['barrel'] = {
            'x': base_x,
            'y': base_y - height/2,  # Center point of the cactus
            'base_x': base_x,  # Original x position
            'base_y': base_y,  # Store ground level
            'width': width,
            'height': height,
            'sway_phase': np.random.uniform(0, 2*np.pi),
            'sway_amount': np.random.uniform(0.8, 1.2),
            'has_flower': True,  # 30% chance of having a flower
            'vertical_offset': 0,  # For bouncing
            'horizontal_offset': 0,  # For swaying
            'hop_offset_x': 0,  # For hopping movement
            'hop_offset_y': 0,  # For hopping arc
            'num_ribs': np.random.randint(12, 18),  # Number of rib lines
            'shadow_offset': 0,  # For shadow when bouncing
            'shadow_scale': 1.0,  # For shadow scaling when bouncing
            'total_hop_x': 0  # Track total distance hopped for flower positioning
        }
        
        # Generate noise texture for the barrel
        instate['noise_texture'] = np.random.uniform(-0.05, 0.05, (60, 120))
        
        # Flowers if the cactus has them
        if instate['barrel']['has_flower']:
            instate['barrel']['flowers'] = []
            num_flowers = np.random.randint(1, 4)
            
            # Place flowers at top of cactus with variation
            for i in range(num_flowers):
                angle = np.random.uniform(0, 2*np.pi)
                radius = width * 0.4
                
                flower = {
                    'x': base_x + np.cos(angle) * radius,
                    'y': (base_y - height) + np.random.uniform(1, 5),  # Near the top with variation
                    'size': np.random.uniform(2.0, 3.5),
                    'color': np.random.choice([0.05, 0.95]),  # Red or orange
                    'sway_phase': np.random.uniform(0, 2*np.pi)
                }
                instate['barrel']['flowers'].append(flower)
        
        # Add some rocks and desert vegetation (simplified)
        instate['decorations'] = []
        num_decorations = np.random.randint(3, 7)
        
        for i in range(num_decorations):
            decoration_type = np.random.randint(0, 2)  # 0 = rock, 1 = desert shrub
            pos_x = np.random.uniform(10, 110)
            pos_y = np.random.uniform(instate['ground_y'], 58)
            
            decoration = {
                'type': decoration_type,
                'pos': np.array([pos_x, pos_y]),
                'size': np.random.uniform(1, 2.5),
                'color_variation': np.random.uniform(-0.05, 0.05)
            }
            instate['decorations'].append(decoration)
        
        return

    if instate['count'] == -1:
        outstate['has_cactus'] = False
        outstate['render'][instate['frame_id']].remove_image_plane(instate['barrel_plane'])
        return

    current_time = time.time()
    dt = min(current_time - instate['last_update'], 0.1)  # Cap dt to avoid large jumps
    elapsed_time = current_time - instate['start_time']
    total_duration = instate.get('duration', 60.0)
    
    # Calculate fade factor
    fade_duration = 4.0
    if elapsed_time < fade_duration:
        fade_factor = elapsed_time / fade_duration
    elif elapsed_time > (total_duration - fade_duration):
        fade_factor = (total_duration - elapsed_time) / fade_duration
    else:
        fade_factor = 1.0
    fade_factor = np.clip(fade_factor, 0, 1)
    
    # Seasonal wind strength
    season = outstate['season']
    s_wind = (1 - 0.5 * np.cos(np.pi * 2 * (season - 0.125)))
    
    # Update wind parameters
    wind = instate['wind']
    if current_time - wind['change_time'] > wind['change_interval']:
        wind['target_strength'] = np.random.uniform(0.2, 0.4) * s_wind
        wind['change_time'] = current_time
        wind['change_interval'] = np.random.uniform(3, 7)
    
    # Smoothly adjust wind strength
    wind['strength'] += (wind['target_strength'] - wind['strength']) * min(dt * 0.5, 0.1)
    
    # Update sway phases for barrel
    barrel = instate['barrel']
    barrel['sway_phase'] += dt * 0.75
    hopframes=20
    # Handle hop behavior (larger movement side to side with arc)
    hop = instate['hop']
    
    if hop['active']:
        hop['frame_count'] += 1
        
        # Use frame count for consistent timing (0 to 100 frames)
        # Maximum of 100 frames for the complete hop
        if hop['frame_count'] >= hopframes:
            # End hop
            hop['active'] = False
            hop['frame_count'] = 0
            
            # Update total hop distance for flower positioning
            barrel['total_hop_x'] += hop['distance']
            
            # Update base position to the landing position
            barrel['base_x'] += hop['distance']  # Final landing position
            barrel['hop_offset_x'] = 0  # Reset for next hop
            barrel['hop_offset_y'] = 0  # Reset Y to ground level
            
            # Update flower positions to new cactus position
            if barrel['has_flower']:
                for flower in barrel['flowers']:
                    flower['x'] += hop['distance']  # Update flower's base position
            
            # Schedule next hop
            hop['next_hop'] = current_time + np.random.uniform(10, 20)
        else:
            # Calculate hop progress (0 to 1)
            progress = hop['frame_count'] / hopframes
            
            # Calculate horizontal movement (linear from start to finish)
            barrel['hop_offset_x'] = hop['distance'] * progress
            
            # Calculate vertical arc (parabolic)
            # Maximum height at middle of hop (progress = 0.5)
            arc_height = 7.0 * 4 * progress * (1 - progress)  # Parabola peaking at 10 pixels
            barrel['hop_offset_y'] = -arc_height  # Negative because up is negative y
            
            # Shadow effects during hop
            barrel['shadow_offset'] = arc_height * 0.3
            barrel['shadow_scale'] = 1.0 - arc_height * 0.02
    else:
        # Check if it's time for a new hop (and not already bouncing)
        bounce = instate['bounce']
        if current_time >= hop['next_hop'] and not bounce['active']:
            # Start a new hop
            hop['active'] = True
            hop['start_time'] = current_time
            hop['frame_count'] = 0
            path=sound_path / 'bonk-46000.mp3'
            outstate['soundengine'].schedule_event(path, time.time(), 6)
            # Determine hop direction based on current position
            current_center_x = barrel['base_x']  # Use base_x (current position after previous hops)
            
            # Choose a direction that keeps cactus within bounds (30-90)
            if current_center_x <= 35:
                hop['direction'] = 1  # Right
            elif current_center_x >= 85:
                hop['direction'] = -1  # Left
            else:
                hop['direction'] = 1 if np.random.random() < 0.5 else -1  # Random
                
            # Set hop distance (7-15 pixels in chosen direction)
            hop_distance = np.random.uniform(7, 15)
            hop['distance'] = hop_distance * hop['direction']
            
            # Reset offsets at start of hop
            barrel['hop_offset_x'] = 0
            barrel['hop_offset_y'] = 0
    
    # Handle bounce behavior (small vertical bounce)
    bounce = instate['bounce']
    
    if bounce['active']:
        bounce_elapsed = current_time - bounce['start_time']
        if bounce_elapsed >= bounce['duration']:
            # End bounce
            bounce['active'] = False
            barrel['vertical_offset'] = 0
            barrel['shadow_offset'] = 0
            barrel['shadow_scale'] = 1.0
            # Schedule next bounce
            bounce['next_bounce'] = current_time + np.random.uniform(15, 30)
        else:
            # Calculate bounce progress (0 to 1 and back to 0)
            progress = bounce_elapsed / bounce['duration']
            bounce_factor = np.sin(progress * np.pi) * np.exp(-2 * progress)  # Damped sine curve
            
            # Apply bounce only if not hopping
            if not hop['active']:
                max_bounce = bounce['intensity'] * 3
                barrel['vertical_offset'] = -max_bounce * bounce_factor  # Negative because up is negative y
                
                # Shadow effects
                barrel['shadow_offset'] = max_bounce * bounce_factor * 0.3
                barrel['shadow_scale'] = 1.0 - bounce_factor * 0.1
    else:
        # Check if it's time for a new bounce (and not hopping)
        if current_time >= bounce['next_bounce'] and not hop['active']:
            # Start a new bounce
            bounce['active'] = True
            bounce['start_time'] = current_time
            bounce['duration'] = np.random.uniform(2.5, 4.0)
            bounce['intensity'] = np.random.uniform(0.3, 0.6)
    
    # Handle quick wiggle behavior
    wiggle = instate['wiggle']
    
    if wiggle['active']:
        wiggle_elapsed = current_time - wiggle['start_time']
        if wiggle_elapsed >= wiggle['duration']:
            # End wiggle
            wiggle['active'] = False
            # Schedule next wiggle
            wiggle['next_wiggle'] = current_time + np.random.uniform(5, 15)
    else:
        # Check if it's time for a new wiggle
        if current_time >= wiggle['next_wiggle']:
            # Start a new wiggle
            wiggle['active'] = True
            wiggle['start_time'] = current_time
            wiggle['duration'] = np.random.uniform(0.5, 1.2)  # Quick wiggle duration
    
    # Calculate sway offset based on wind
    sway_offset = np.sin(barrel['sway_phase']) * wind['strength'] * barrel['sway_amount'] * 3
    
    # Add quick wiggle if active
    if wiggle['active'] and not hop['active']:  # Don't wiggle during hop
        wiggle_progress = (current_time - wiggle['start_time']) / wiggle['duration']
        # Fast wiggle that diminishes as it progresses
        wiggle_amount = np.sin(wiggle_progress * 50) * 3.0 * (1 - wiggle_progress)
        sway_offset += wiggle_amount
    
    barrel['horizontal_offset'] = sway_offset
    
    # Clear the window
    window = instate['barrel_window']
    window.fill(0)
    
    # Get coordinate grids
    y_indices = instate['y_indices']
    x_indices = instate['x_indices']
    
    # Draw ground with transparent sky (vectorized operation)
    ground_mask = y_indices >= instate['ground_y']
    window[ground_mask, 0] = instate['soil_color']  # Hue
    window[ground_mask, 1] = 0.5  # Saturation
    window[ground_mask, 2] = 0.3  # Value
    window[ground_mask, 3] = fade_factor  # Alpha
    
    # Draw desert decorations - process all at once for performance
    for decoration in instate['decorations']:
        dec_x, dec_y = decoration['pos']
        # Limit calculation to a region around the decoration (optimization)
        size = decoration['size']
        x_min = max(0, int(dec_x - size - 1))
        x_max = min(120, int(dec_x + size + 1))
        y_min = max(0, int(dec_y - size - 1))
        y_max = min(60, int(dec_y + size + 1))
        
        if x_min < x_max and y_min < y_max:  # Make sure region is valid
            sub_x = x_indices[y_min:y_max, x_min:x_max]
            sub_y = y_indices[y_min:y_max, x_min:x_max]
            
            dec_distance = np.sqrt((sub_x - dec_x)**2 + (sub_y - dec_y)**2)
            dec_mask = dec_distance < size
            
            if decoration['type'] == 0:  # Rock
                window[y_min:y_max, x_min:x_max][dec_mask, 0] = 0.05 + decoration['color_variation']
                window[y_min:y_max, x_min:x_max][dec_mask, 1] = 0.2
                window[y_min:y_max, x_min:x_max][dec_mask, 2] = 0.4
                window[y_min:y_max, x_min:x_max][dec_mask, 3] = fade_factor
            else:  # Desert shrub
                window[y_min:y_max, x_min:x_max][dec_mask, 0] = 0.25 + decoration['color_variation']
                window[y_min:y_max, x_min:x_max][dec_mask, 1] = 0.4
                window[y_min:y_max, x_min:x_max][dec_mask, 2] = 0.3
                window[y_min:y_max, x_min:x_max][dec_mask, 3] = fade_factor
    
    # Calculate shadow position and size for barrel cactus
    shadow_x = barrel['base_x'] + barrel['hop_offset_x']
    shadow_y = instate['ground_y'] - 1  # Just above ground level
    shadow_width = barrel['width'] * barrel['shadow_scale'] * 0.8
    shadow_height = barrel['width'] * 0.3 * barrel['shadow_scale']  # Shadow always oval regardless of barrel height
    
    # Apply horizontal offset but reverse (so shadow moves opposite to barrel)
    shadow_x -= barrel['horizontal_offset'] * 0.3  # Reduced shadow movement
    shadow_x += barrel['shadow_offset'] * 0.5  # Add bounce/hop offset
    
    # Draw shadow (flatten to oval)
    dx = x_indices - shadow_x
    dy = y_indices - shadow_y
    shadow_dist = ((dx / shadow_width)**2 + (dy / shadow_height)**2)
    shadow_mask = (shadow_dist <= 1.0) & (y_indices >= instate['ground_y'])
    
    # Instead of replacing the ground color entirely, blend with it
    if np.any(shadow_mask):
        # Get current ground color where shadow will appear
        shadow_blend_factor = (1 - shadow_dist[shadow_mask]**2) * 0.6
        if hop['active']:
            # Reduce shadow alpha based on hop height
            hop_progress = hop['frame_count'] / hopframes
            hop_height = 7.0 * 4 * hop_progress * (1 - hop_progress)
            shadow_reduction = hop_height / 7.0 * 0.5  # Up to 50% transparency at max height
            shadow_blend_factor *= max(0.5, 1.0 - shadow_reduction)
        
        # Darken existing ground colors rather than replacing them
        # Keep the ground hue and saturation, just reduce the value/brightness
        window[shadow_mask, 2] *= (1.0 - shadow_blend_factor * 0.7)  # Reduce value by up to 70%
        window[shadow_mask, 3] = fade_factor  # Maintain ground alpha
    
    
    # Alpha with falloff from center (fade when hopping high)
    shadow_alpha = (1 - shadow_dist[shadow_mask]**2) * 0.6 * fade_factor
    if hop['active']:
        # Reduce shadow alpha based on hop height
        hop_progress = hop['frame_count'] / 100.0
        hop_height = 10.0 * 4 * hop_progress * (1 - hop_progress)
        shadow_reduction = hop_height / 10.0 * 0.5  # Up to 50% transparency at max height
        shadow_alpha *= max(0.5, 1.0 - shadow_reduction)
    window[shadow_mask, 3] = 1#shadow_alpha
    
    # Calculate barrel cactus position with all offsets
    barrel_x = barrel['base_x'] + barrel['horizontal_offset'] + barrel['hop_offset_x']
    barrel_y = barrel['y'] + barrel['vertical_offset'] + barrel['hop_offset_y']
    
    # Ensure cactus stays within allowed bounds (30-90 for center)
    barrel_x = np.clip(barrel_x, 30, 90)
    
    # Calculate barrel dimensions
    width = barrel['width']
    height = barrel['height']
    
    # Calculate the barrel shape using vectorized operations
    dx = x_indices - barrel_x
    dy = y_indices - barrel_y
    
    # Normalize y position for barrel shape
    rel_y = dy / (height/2)
    # Width is slightly less at top and bottom (barrel shape)
    width_factor = 1 - 0.15 * np.clip(rel_y, -1, 1)**2
    
    # Calculate barrel shape (elliptical with width varying by height)
    barrel_dist = (dx / (width * width_factor))**2 + (dy / (height/2))**2
    barrel_mask = barrel_dist <= 1.0
    
    # Apply base green color with noise
    noise = instate['noise_texture']
    
    # Set base green color with noise variation - STRICTLY GREEN
    window[barrel_mask, 0] = 0.33 + noise[barrel_mask] * 0.03  # Green hue with small variation
    window[barrel_mask, 1] = 0.6 + noise[barrel_mask] * 0.1   # Medium-high saturation with variation
    window[barrel_mask, 2] = 0.5 + noise[barrel_mask] * 0.1   # Medium brightness with variation
    window[barrel_mask, 3] = fade_factor
    
    # Calculate top and bottom points of the cactus
    top_y = barrel_y - height/2
    bottom_y = barrel_y + height/2
    
    # Create a normalized height grid (0 at top, 1 at bottom)
    #height_grid = np.clip((y_indices - top_y) / height, 0, 1)
    
    # Number of vertical rib lines
    num_ribs = barrel['num_ribs']
    
    # Create a clean rib mask 
    #rib_mask = np.zeros_like(barrel_mask, dtype=bool)
    
    # Draw each rib line separately
    # Vectorized rib drawing
    # Generate all angles for all ribs at once
    rib_angles = np.linspace(0, 2 * np.pi, num_ribs, endpoint=False)
    cos_angles = np.cos(rib_angles)

    # Generate all spine y positions at once
    spine_y_positions = np.linspace(top_y, bottom_y, 15, endpoint=False)

    # Filter to only include positions within bounds
    valid_spine_indices = (spine_y_positions >= top_y) & (spine_y_positions <= bottom_y)
    spine_y_positions = spine_y_positions[valid_spine_indices]

    # Initialize the spine mask
    spine_mask = np.zeros_like(barrel_mask, dtype=bool)

    # If no valid spine positions, skip
    if len(spine_y_positions) > 0:
        # Create arrays to store all spine coordinates
        all_spine_x = []
        all_spine_y = []
        
        for i, spine_y in enumerate(spine_y_positions):
            # Calculate height ratio for this spine row
            height_ratio = (spine_y - top_y) / height
            
            # Width at this height
            local_width = width * (1 - 0.15 * (2*height_ratio - 1)**2)
            
            # Calculate all x positions for this row of spines
            row_spine_x = barrel_x + local_width * 0.9 * cos_angles
            
            # Create corresponding y positions (same y for all spines in this row)
            row_spine_y = np.full_like(row_spine_x, spine_y)
            
            # Add to the arrays
            all_spine_x.extend(row_spine_x)
            all_spine_y.extend(row_spine_y)
        
        # Convert to numpy arrays
        spine_xs = np.array(all_spine_x)
        spine_ys = np.array(all_spine_y)
        
        # Filter out spines outside the image bounds
        valid_spines = ((spine_xs >= 0) & (spine_xs < 120) & 
                        (spine_ys >= 0) & (spine_ys < 60))
        spine_xs = spine_xs[valid_spines]
        spine_ys = spine_ys[valid_spines]
        
        # Round to nearest pixel
        spine_xs_int = np.round(spine_xs).astype(int)
        spine_ys_int = np.round(spine_ys).astype(int)
        
        # For each spine, update the pixels within a small radius
        spine_radius = 0.8
        spine_radius_squared = spine_radius**2
        
        # Define a small region to check around each spine
        region_size = int(np.ceil(spine_radius)) + 1
        
        # Process each spine
        for sx, sy in zip(spine_xs_int, spine_ys_int):
            # Define region around spine
            x_min = max(0, sx - region_size)
            x_max = min(120, sx + region_size + 1)
            y_min = max(0, sy - region_size)
            y_max = min(60, sy + region_size + 1)
            
            if x_min < x_max and y_min < y_max:
                # Get region of coordinates
                region_x = x_indices[y_min:y_max, x_min:x_max]
                region_y = y_indices[y_min:y_max, x_min:x_max]
                
                # Calculate distance squared to spine
                dx_squared = (region_x - sx)**2
                dy_squared = (region_y - sy)**2
                dist_squared = dx_squared + dy_squared
                
                # Create mask for this spine
                region_spine = (dist_squared <= spine_radius_squared) & barrel_mask[y_min:y_max, x_min:x_max]
                
                # Update the spine mask
                if np.any(region_spine):
                    spine_mask[y_min:y_max, x_min:x_max] |= region_spine
    
    # Apply bright yellow-white color to spines
    window[spine_mask, 0] = 0.13  # Yellow-white hue
    window[spine_mask, 1] = 0.25  # Low saturation
    window[spine_mask, 2] = 0.95  # Very bright
    
    # Draw flowers if present
    if barrel['has_flower']:
        for flower in barrel['flowers']:
            # Calculate flower position based on barrel's current position
            # Use the base_x (includes accumulated hop distance)
            # And add current hop_offset_x and hop_offset_y for active hops
            flower_x = barrel['base_x'] + barrel['hop_offset_x'] + barrel['horizontal_offset']
            flower_y = barrel['y'] + barrel['hop_offset_y'] + barrel['vertical_offset']
            
            # Calculate flower position relative to cactus center
            rel_x = flower['x'] - barrel['base_x']
            rel_y = flower['y'] - barrel['y']
            
            # Apply this relative position to the current cactus position
            flower_x += rel_x
            flower_y += rel_y
            
            # Add slight sway to flowers
            flower_sway = np.sin(flower['sway_phase'] + elapsed_time) * wind['strength'] * 1.0
            flower_x += flower_sway * 0.5
            flower_y += flower_sway * 0.3
            
            # Draw the flower
            draw_flower(
                window, x_indices, y_indices,
                flower_x, flower_y,
                flower['size'],
                flower['color'],
                fade_factor
            )
    
    # Convert HSVA to RGBA for rendering
    rgb = color.hsv2rgb(window[..., 0:3])
    alpha = window[..., 3:4]
    rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)
    
    # Update the image plane
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['barrel_plane'],
        rgb_out[:,:,:]
    )
    
    # Update timestamp
    instate['last_update'] = current_time