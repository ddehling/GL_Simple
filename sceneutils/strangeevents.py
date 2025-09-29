import time
import numpy as np
from skimage import color
from pathlib import Path
ParentPath = Path(__file__).parent.parent
media_path = ParentPath / 'media'
sound_path = media_path / 'sounds'


def chromatic_fog_beings(instate, outstate):
    if instate['count'] == 0:
        # Initialize fog beings parameters
        instate['fog_window'] = np.zeros((60, 120, 4))  # RGBA format for final output
        instate['fog_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((60, 120, 4), dtype=np.uint8),
            position=(0, 0, 19),
            rotation=(0, 0, 0),
            scale=(3, 3)
        )
        
        # Create 3-5 fog beings with different properties
        num_beings = np.random.randint(3, 6)
        instate['beings'] = []
        
        for i in range(num_beings):
            # Each being has its own parameters
            being = {
                'position': np.array([np.random.uniform(20, 100), np.random.uniform(15, 45)]),
                'velocity': np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)]),
                'size': np.random.uniform(6, 10),
                'base_hue': np.random.uniform(0, 1),
                'hue_drift_rate': np.random.uniform(0.05, 0.2),
                'hue_drift_phase': np.random.uniform(0, 2*np.pi),
                'target_behavior': np.random.randint(0, 3),  # 0=wander, 1=seek, 2=mimic
                'target_entity': None,
                'shape_complexity': np.random.uniform(2, 5),
                'shape_phase': np.random.uniform(0, 2*np.pi),
                'shape_evolution_rate': np.random.uniform(0.1, 0.3),
                'last_behavior_change': time.time(),
                'behavior_duration': np.random.uniform(5, 15),
                'memory': [],  # Stores previous shapes/positions for mimicry
                'memory_capacity': 10,
                'color_pulses': [],  # For communication events
                'tentacles': np.random.randint(0, 5),  # Number of extended tendrils
                'tentacle_params': []
            }
            
            # Initialize tentacles if any
            for _ in range(being['tentacles']):
                being['tentacle_params'].append({
                    'angle': np.random.uniform(0, 2*np.pi),
                    'length': np.random.uniform(5, 15),
                    'wave_rate': np.random.uniform(0.5, 2.0),
                    'wave_phase': np.random.uniform(0, 2*np.pi)
                })
            
            instate['beings'].append(being)
        
        # Communication event timing
        instate['next_communication'] = time.time() + np.random.uniform(3, 8)
        
        # Initialize timing parameters
        instate['start_time'] = time.time()
        instate['last_update'] = time.time()
        
        # Pre-compute coordinate grids for vectorized calculations
        instate['y_coords'], instate['x_coords'] = np.mgrid[0:60, 0:120]
        
        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['fog_plane'])
        return

    # Time calculations
    current_time = time.time()
    dt = current_time - instate['last_update']
    elapsed_time = current_time - instate['start_time']
    total_duration = instate.get('duration', 60)
    
    # Calculate fade factor more efficiently
    fade_duration = 10.0
    if elapsed_time < fade_duration:
        fade_factor = elapsed_time / fade_duration
    elif elapsed_time > (total_duration - fade_duration):
        fade_factor = (total_duration - elapsed_time) / fade_duration
    else:
        fade_factor = 1.0
    fade_factor = np.clip(fade_factor, 0, 1)

    # Clear the window more efficiently
    instate['fog_window'].fill(0)
    
    # Get coordinate grids - already stored in instate
    y_coords, x_coords = instate['y_coords'], instate['x_coords']
    
    # Sort beings by y-position for back-to-front rendering
    beings_sorted = sorted(instate['beings'], key=lambda b: b['position'][1])
    
    # Pre-compute common trigonometric values
    sin_time = np.sin(current_time)
    
    # Update each being
    for being in beings_sorted:
        # Update behavior if needed
        if current_time - being['last_behavior_change'] > being['behavior_duration']:
            being['target_behavior'] = np.random.randint(0, 3)
            being['behavior_duration'] = np.random.uniform(5, 15)
            being['last_behavior_change'] = current_time
            
            # If seeking behavior, choose a target entity
            if being['target_behavior'] == 1 and len(instate['beings']) > 1:
                potential_targets = [b for b in instate['beings'] if b is not being]
                being['target_entity'] = np.random.choice(potential_targets)
            else:
                being['target_entity'] = None
        
        # Handle different behaviors
        if being['target_behavior'] == 0:  # Wander
            # Occasionally change direction
            if np.random.random() < 0.02:
                angle = np.random.uniform(0, 2*np.pi)
                speed = np.linalg.norm(being['velocity']) or np.random.uniform(0.5, 1.5)
                being['velocity'] = np.array([np.cos(angle), np.sin(angle)]) * speed
        
        elif being['target_behavior'] == 1 and being['target_entity']:  # Seek
            # Move toward target entity
            direction = being['target_entity']['position'] - being['position']
            distance = np.linalg.norm(direction)
            
            if distance > 0.1:
                direction = direction / distance
                target_velocity = direction * np.random.uniform(0.5, 1.5)
                being['velocity'] += (target_velocity - being['velocity']) * 0.1
                
                # Limit speed
                speed = np.linalg.norm(being['velocity'])
                if speed > 2.0:
                    being['velocity'] = being['velocity'] / speed * 2.0
        
        elif being['target_behavior'] == 2:  # Mimic
            # Evolve toward a more complex/interesting shape
            being['shape_complexity'] = 3 + np.sin(current_time * 0.2) * 2
            
            # Store current state in memory for other beings to potentially mimic
            if len(being['memory']) >= being['memory_capacity']:
                being['memory'].pop(0)
            being['memory'].append({
                'position': being['position'].copy(),
                'shape_phase': being['shape_phase'],
                'shape_complexity': being['shape_complexity']
            })
        
        # Update position
        being['position'] += being['velocity'] * dt
        
        # Boundary checks with soft bouncing
        padding = 10
        dim_limits = np.array([120 - padding, 60 - padding])
        
        for i in range(2):
            if being['position'][i] < padding:
                being['position'][i] = padding
                being['velocity'][i] = abs(being['velocity'][i]) * 0.8
            elif being['position'][i] > dim_limits[i]:
                being['position'][i] = dim_limits[i]
                being['velocity'][i] = -abs(being['velocity'][i]) * 0.8
        
        # Update appearance properties
        being['shape_phase'] += being['shape_evolution_rate'] * dt
        being['hue_drift_phase'] += being['hue_drift_rate'] * dt
        
        # Calculate current hue with drift
        current_hue = (being['base_hue'] + 0.1 * np.sin(being['hue_drift_phase'])) % 1.0
        
        # Draw the being using metaballs for smooth, organic shape
        center_x, center_y = being['position']
        size = being['size']
        complexity = being['shape_complexity']
        phase = being['shape_phase']
        
        # Create base density field - more efficient distance calculation
        dx = x_coords - center_x
        dy = y_coords - center_y
        base_distance_squared = dx*dx + dy*dy
        
        # Pre-calculate size values once
        size_squared = 2 * size * size
        density = np.exp(-base_distance_squared / size_squared)
        
        # Only calculate base_distance when needed (for pulses)
        base_distance = None
        
        # Add organic variations based on complexity - optimize
        n_lobes = int(complexity)
        offset_factor = size * 0.7
        lobe_size = size * 0.6
        lobe_size_squared = 2 * lobe_size * lobe_size
        
        if n_lobes > 0:
            # Pre-compute angles for all lobes at once
            lobe_angles = phase + np.arange(n_lobes) * (2*np.pi / complexity)
            cos_angles = np.cos(lobe_angles)
            sin_angles = np.sin(lobe_angles)
            
            # Calculate all lobe positions at once
            lobe_xs = center_x + cos_angles * offset_factor
            lobe_ys = center_y + sin_angles * offset_factor
            
            # Add all lobes to density
            for i in range(n_lobes):
                lobe_x, lobe_y = lobe_xs[i], lobe_ys[i]
                
                lobe_dx = x_coords - lobe_x
                lobe_dy = y_coords - lobe_y
                lobe_distance_squared = lobe_dx*lobe_dx + lobe_dy*lobe_dy
                
                # Add lobe density - avoid redundant exponential calculation
                density += 0.7 * np.exp(-lobe_distance_squared / lobe_size_squared)
        
        # Draw tentacles if any - this is a high-cost area, optimize carefully
        if being['tentacles'] > 0:
            segment_sizes = size * 0.3  # Base segment size
            
            for tentacle in being['tentacle_params']:
                angle = tentacle['angle'] + np.sin(current_time * tentacle['wave_rate'] + tentacle['wave_phase']) * 0.5
                length = tentacle['length']
                
                # Create points along the tentacle - use fewer segments for efficiency
                segments = min(int(length), 8)  # Cap the number of segments
                if segments <= 0:
                    continue
                    
                segment_ratios = np.linspace(0, 1, segments)
                segment_lengths = length * segment_ratios
                
                # Vectorize undulation calculation
                undulation = segment_ratios * sin_time
                angles = angle + undulation
                
                # Pre-calculate trig functions
                cos_angles = np.cos(angles)
                sin_angles = np.sin(angles)
                
                # Calculate segment positions
                segment_xs = center_x + cos_angles * segment_lengths
                segment_ys = center_y + sin_angles * segment_lengths
                
                # Taper sizes toward end
                segment_sizes_all = segment_sizes * (1 - segment_ratios * 0.8)
                segment_sizes_squared = 2 * segment_sizes_all * segment_sizes_all
                
                # Process each segment - check bounds first to avoid unnecessary calculations
                in_bounds_mask = (
                    (segment_xs >= 0) & (segment_xs < 120) &
                    (segment_ys >= 0) & (segment_ys < 60)
                )
                
                valid_indices = np.where(in_bounds_mask)[0]
                
                for j in valid_indices:
                    segment_x, segment_y = segment_xs[j], segment_ys[j]
                    
                    # Add density for this segment
                    segment_dx = x_coords - segment_x
                    segment_dy = y_coords - segment_y
                    segment_distance_squared = segment_dx*segment_dx + segment_dy*segment_dy
                    
                    segment_size_squared = segment_sizes_squared[j]
                    # Use a multiplier directly in the exponent calculation
                    density += 0.5 * np.exp(-segment_distance_squared / segment_size_squared)
        
        # Process any active color pulses (communication)
        # Create a single pulse density field for better efficiency
        pulse_density = np.zeros_like(density) if being['color_pulses'] else None
        
        remaining_pulses = []
        for pulse in being['color_pulses']:
            pulse['age'] += dt
            if pulse['age'] < pulse['duration']:
                # Calculate pulse radius
                max_radius = size * 2.5
                pulse_progress = pulse['age'] / pulse['duration']
                pulse_radius = max_radius * pulse_progress
                
                # Calculate base_distance only if needed and not already calculated
                if base_distance is None:
                    base_distance = np.sqrt(base_distance_squared)
                
                # Create ring effect
                ring_width = size * 0.5
                ring_inner = pulse_radius - ring_width/2
                ring_outer = pulse_radius + ring_width/2
                
                ring_mask = (base_distance >= ring_inner) & (base_distance <= ring_outer)
                
                if np.any(ring_mask):
                    # Calculate intensity based on distance
                    normalized_distance = (base_distance[ring_mask] - ring_inner) / ring_width
                    ring_intensity = np.sin(normalized_distance * np.pi) * (1 - pulse_progress)
                    
                    # Add pulse to pulse density field
                    pulse_density[ring_mask] += ring_intensity * 0.3
                
                remaining_pulses.append(pulse)
        
        being['color_pulses'] = remaining_pulses
        
        # Add pulses to main density if there were any
        if pulse_density is not None:
            density += pulse_density
        
        # Normalize density for consistent appearance
        max_density = np.max(density)
        if max_density > 0:
            density = density / max_density
        
        # Create mask for areas with sufficient density
        fog_mask = density > 0.05
        
        # Skip if no visible elements
        if not np.any(fog_mask):
            continue
            
        # Apply being's appearance to the window using alpha blending
        # Define colors for this being
        masked_density = density[fog_mask]
        saturation = 0.9 - masked_density * 0.3
        value = 0.2 + masked_density * 0.6 * fade_factor
        alpha = masked_density * 0.4 * fade_factor
        
        # Create HSV values for this being
        hsv = np.zeros((np.sum(fog_mask), 3))
        hsv[:, 0] = current_hue
        hsv[:, 1] = saturation
        hsv[:, 2] = value
        
        # Convert to RGB
        rgb = color.hsv2rgb(hsv)
        
        # Get existing values from the fog window where this being will be drawn
        existing_rgba = instate['fog_window'][fog_mask]
        
        # Perform alpha compositing more efficiently
        src_color = rgb  # Shape: (N, 3)
        src_alpha = alpha.reshape(-1, 1)  # Shape: (N, 1)
        dst_color = existing_rgba[:, 0:3]  # Shape: (N, 3)
        dst_alpha = existing_rgba[:, 3:4]  # Shape: (N, 1)
        
        # Calculate new alpha
        new_alpha = src_alpha + dst_alpha * (1 - src_alpha)  # Shape: (N, 1)
        
        # Initialize the result array
        new_rgba = np.zeros_like(existing_rgba)  # Shape: (N, 4)
        
        # Only blend where alpha > 0 to avoid division by zero
        blend_mask = new_alpha[:, 0] > 0
        
        if np.any(blend_mask):
            # Vectorized calculation for RGB channels
            blend_factor = 1 / new_alpha[blend_mask]
            new_rgba[blend_mask, 0:3] = (
                src_color[blend_mask] * src_alpha[blend_mask] + 
                dst_color[blend_mask] * dst_alpha[blend_mask] * (1 - src_alpha[blend_mask])
            ) * blend_factor
        
        # Set alpha
        new_rgba[:, 3] = new_alpha[:, 0]
        
        # Update the fog window
        instate['fog_window'][fog_mask] = new_rgba
    
    # Check for communication events
    if current_time >= instate['next_communication'] and len(instate['beings']) > 1:
        # Choose two random beings to communicate
        sender, receiver = np.random.choice(instate['beings'], 2, replace=False)
        
        # Create a communication pulse
        pulse = {
            'age': 0.0,
            'duration': np.random.uniform(1.0, 3.0),
            'hue': sender['base_hue']
        }
        sender['color_pulses'].append(pulse)
        
        # Schedule next communication
        instate['next_communication'] = current_time + np.random.uniform(3, 8)
        
        # Receiver might change behavior in response
        if np.random.random() < 0.3:
            receiver['target_behavior'] = np.random.randint(0, 3)
            receiver['target_entity'] = sender
            receiver['last_behavior_change'] = current_time
    
    # Convert to RGBA for rendering - do the multiplication and type conversion in one step
    rgb_out = (instate['fog_window'] * 255).astype(np.uint8)
    
    # Update the image plane
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['fog_plane'], 
        rgb_out
    )
    
    instate['last_update'] = current_time