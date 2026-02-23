import time
import numpy as np
from skimage import color
from pathlib import Path
ParentPath = Path(__file__).parent.parent
media_path = ParentPath / 'media'
sound_path = media_path / 'sounds'

def growing_mushrooms(instate, outstate):
    if instate['count'] == 0:
        # Initialize mushroom parameters
        instate['mushroom_window'] = np.zeros((60, 120, 4))  # HSVA format
        instate['mushroom_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((60, 120, 4), dtype=np.uint8),
            position=(0, 0, 10),
            rotation=(0, 0, 0),
            scale=(1, 1)
        )
        outstate['has_mushrooms'] = True
        # Initialize timing parameters
        instate['start_time'] = time.time()
        instate['last_update'] = time.time()
        
        # Initialize toad parameters
        instate['toad'] = {
            'active': False,
            'entry_time': 20.0,  # Seconds after start when toad appears
            'current_mushroom': None,
            'next_mushroom': None,
            'pos_x': -10,  # Start offscreen
            'pos_y': 0,
            'target_x': 0,
            'target_y': 0,
            'state': 'waiting',  # States: waiting, hopping, sitting
            'hop_progress': 0.0,
            'sit_timer': 0.0,
            'sit_duration': 0.0,
            'size': 5.0,
            'facing_right': True,
            'hop_height': 0.0,
            'z_index': 0  # For rendering behind mushrooms
        }
        
        # Create multiple mushrooms with varied properties
        instate['mushrooms'] = []
        num_mushrooms = np.random.randint(4, 9)
        
        # Ground parameters
        instate['ground_height'] = 50  # Ground level from top of screen
        instate['ground_color_base'] = 0.1  # Brown hue
        
        # Create mushroom clusters at varied locations
        for _ in range(num_mushrooms):
            # Try to find a spot that's not too close to existing mushrooms
            while True:
                pos_x = np.random.randint(10, 110)
                too_close = False
                
                # Check distance from other mushrooms
                for mushroom in instate['mushrooms']:
                    if abs(pos_x - mushroom['pos_x']) < 8:
                        too_close = True
                        break
                
                if not too_close:
                    break
            
            # Determine whether this is a cluster or single mushroom
            is_cluster = np.random.random() < 0.3
            cluster_size = np.random.randint(2, 4) if is_cluster else 1
            cluster = []
            
            # Create mushrooms in this cluster
            for i in range(cluster_size):
                # Variation within cluster
                cluster_offset = np.random.randint(-4, 5) if is_cluster else 0
                
                # Mushroom characteristics
                mushroom = {
                    'pos_x': pos_x + cluster_offset,
                    'pos_y': instate['ground_height'],  # Starting at ground level
                    'type': np.random.choice(['normal', 'slim', 'round', 'umbrella', 'conical', 'bell', 'flat']),
                    'size': np.random.uniform(0.7, 1.8),
                    'cap_width_ratio': np.random.uniform(1.5, 3.5),
                    'stem_height_ratio': np.random.uniform(0.8, 1.5),
                    'start_delay': np.random.uniform(0, 10.0),  # Staggered growth
                    'growth_stage': 0.0,  # 0 to 1 for growth animation
                    'growth_rate': np.random.uniform(0.05, 0.2),
                    'mature_age': 0.0,  # Time since reaching full growth
                    'spore_start': False,  # Whether spores have started releasing
                    'spore_intensity': 0.0,  # Intensity of spore release
                    'spore_color': np.random.uniform(0.05, 0.25),  # Hue variation
                    'spores': [],  # Individual spore particles
                    'exploded': False,  # Whether mushroom has exploded
                    'color_variation': np.random.uniform(-0.05, 0.05),  # Color variation
                    'gills_density': np.random.uniform(6, 12),  # For gill detail
                    'sway_phase': np.random.uniform(0, 2*np.pi),  # For gentle swaying
                    'sway_amount': np.random.uniform(0.2, 0.6),  # How much it sways
                    'has_spots': np.random.random() < 0.6,  # Whether mushroom has spots
                    'has_striations': np.random.random() < 0.3,  # Striations on cap
                    'ring_present': np.random.random() < 0.4,  # Ring on stem
                    'color_scheme': np.random.choice(['brown', 'red', 'white', 'yellow', 'purple']),  # Base color scheme
                    'render_depth': 0  # For depth sorting with toad
                }
                cluster.append(mushroom)
            
            # Add all mushrooms from this cluster
            instate['mushrooms'].extend(cluster)
        
        # Spore system parameters
        instate['spore_system'] = {
            'particles': [],
            'max_particles': 1000,  # Limit for performance
            'global_wind': np.random.uniform(-0.5, 0.5),  # Horizontal bias
            'updraft': np.random.uniform(0.2, 0.5),  # Vertical bias
            'turbulence_phase': 0.0  # For wind variation
        }
        
        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['mushroom_plane'])
        outstate['has_mushrooms'] = False
        return

    current_time = time.time()
    dt = min(current_time - instate['last_update'], 0.05)  # Cap delta time
    elapsed_time = current_time - instate['start_time']
    total_duration = instate.get('duration', 60.0)
    
    # Calculate fade factor
    fade_duration = 3.0
    if elapsed_time < fade_duration:
        fade_factor = elapsed_time / fade_duration
    elif elapsed_time > (total_duration - fade_duration):
        fade_factor = (total_duration - elapsed_time) / fade_duration
    else:
        fade_factor = 1.0
    fade_factor = np.clip(fade_factor, 0, 1)

    # Clear the window
    instate['mushroom_window'].fill(0)
    
    # Update wind and turbulence
    instate['spore_system']['turbulence_phase'] += dt * 0.8
    wind_variation = np.sin(instate['spore_system']['turbulence_phase']) * 0.3
    current_wind = instate['spore_system']['global_wind'] + wind_variation
    
    # Determine explosion phase (triggered in later part of animation)
    explosion_phase = (elapsed_time - 0.7 * total_duration) / (0.2 * total_duration)
    explosion_active = 0.0 <= explosion_phase <= 1.0
    
    # Draw ground - simple implementation
    ground_y = instate['ground_height']
    ground_height = 60 - ground_y
    if ground_height > 0:
        # Create coordinate grids for the ground area
        y_indices, x_indices = np.mgrid[ground_y:60, 0:120]
        
        # Calculate depth factor as a 2D array
        depth_factor = (y_indices - ground_y) / max(1, ground_height)
        
        # Create noise patterns using vectorized operations
        x_noise = np.sin(x_indices * 0.2) * 0.05
        y_noise = np.cos(y_indices * 0.3) * 0.05
        noise = x_noise + y_noise
        
        # Calculate colors using array operations
        h_values = instate['ground_color_base'] + noise - depth_factor * 0.05  # Hue
        s_values = 0.5 - depth_factor * 0.2  # Saturation
        v_values = 0.4 - depth_factor * 0.2  # Value
        a_values = np.full_like(h_values, fade_factor)  # Alpha
        
        # Assign values to the window buffer in a single operation
        instate['mushroom_window'][ground_y:60, 0:120, 0] = h_values
        instate['mushroom_window'][ground_y:60, 0:120, 1] = s_values
        instate['mushroom_window'][ground_y:60, 0:120, 2] = v_values
        instate['mushroom_window'][ground_y:60, 0:120, 3] = a_values

    # Process each mushroom
    for mushroom in instate['mushrooms']:
        # Only process if animation has started for this mushroom
        if elapsed_time < mushroom['start_delay']:
            continue
        
        #active_time = elapsed_time - mushroom['start_delay']
        
        # Grow until full size
        if mushroom['growth_stage'] < 1.0:
            mushroom['growth_stage'] += mushroom['growth_rate'] * dt
            mushroom['growth_stage'] = min(mushroom['growth_stage'], 1.0)
            
            # If just reached maturity, record time
            if mushroom['growth_stage'] == 1.0:
                mushroom['mature_age'] = 0.0
        else:
            # Track time since maturity
            mushroom['mature_age'] += dt
        
        # Handle explosions during explosion phase
        if explosion_active and not mushroom['exploded'] and mushroom['growth_stage'] >= 0.9:
            # Stagger explosions for more natural feel
            explosion_chance = explosion_phase * 0.1 + mushroom['mature_age'] * 0.01
            if np.random.random() < explosion_chance * dt:
                mushroom['exploded'] = True
                
                path = sound_path / 'Whoosh By 06.wav'
                outstate['soundengine'].schedule_event(path, time.time(), 6)
                

                # Generate large burst of spores
                spore_count = int(50 + 150 * mushroom['size'])
                
                # Calculate cap position (top of mushroom)
                cap_center_x = mushroom['pos_x']
                stem_height = 8 * mushroom['stem_height_ratio'] * mushroom['size'] * mushroom['growth_stage']
                cap_height = 6 * mushroom['size'] * mushroom['growth_stage']
                cap_center_y = mushroom['pos_y'] - stem_height - cap_height/2
                
                # Generate spores in all directions
                for _ in range(spore_count):
                    angle = np.random.uniform(0, 2*np.pi)
                    speed = np.random.uniform(10, 30)
                    size = np.random.uniform(0.3, 0.8)
                    
                    spore = {
                        'pos_x': cap_center_x + np.random.normal(0, 2),
                        'pos_y': cap_center_y + np.random.normal(0, 2),
                        'vel_x': np.cos(angle) * speed,
                        'vel_y': np.sin(angle) * speed,
                        'size': size,
                        'color': mushroom['spore_color'] + np.random.uniform(-0.02, 0.02),
                        'alpha': np.random.uniform(0.5, 1.0),
                        'lifetime': 0.0,  # Make sure this is float
                        'max_lifetime': np.random.uniform(4, 8)
                    }
                    instate['spore_system']['particles'].append(spore)
        
        # Start natural spore release after maturity
        if mushroom['growth_stage'] >= 0.95 and not mushroom['spore_start'] and not mushroom['exploded']:
            maturity_threshold = 5.0  # Seconds after full growth
            if mushroom['mature_age'] > maturity_threshold:
                mushroom['spore_start'] = True
        
        # Calculate current size based on growth stage
        current_size = mushroom['size'] * mushroom['growth_stage']
        
        # Subtle swaying motion
        mushroom['sway_phase'] += dt * 0.5
        sway_offset = np.sin(mushroom['sway_phase']) * mushroom['sway_amount']
        
        # Draw mushroom stem
        stem_width = int(3 * current_size)
        stem_height = int(8 * mushroom['stem_height_ratio'] * current_size)
        
        # Initialize stem_top with a default value
        stem_top = mushroom['pos_y'] - stem_height
        
        if stem_width > 0 and stem_height > 0:
            stem_top = mushroom['pos_y'] - stem_height
            stem_bottom = mushroom['pos_y']
            stem_left = int(mushroom['pos_x'] - stem_width/2 + sway_offset)
            stem_right = int(mushroom['pos_x'] + stem_width/2 + sway_offset)
            
            # Enforce bounds
            stem_left = max(0, stem_left)
            stem_right = min(119, stem_right)
            stem_top = max(0, stem_top)
            stem_bottom = min(59, stem_bottom)
            
            if stem_right > stem_left and stem_bottom > stem_top:
                # Create coordinate grids for vectorized operations
                y_indices, x_indices = np.mgrid[stem_top:stem_bottom, stem_left:stem_right]
                
                # Calculate normalized positions for shading
                x_size = max(1, stem_right - stem_left - 1)
                y_size = max(1, stem_bottom - stem_top - 1)
                norm_x = (x_indices - stem_left) / x_size
                norm_y = (y_indices - stem_top) / y_size
                
                # Brighter on top, darker at bottom
                brightness = 0.7 - norm_y * 0.2
                # Curved shading
                horizontal_shade = 0.7 + np.sin(norm_x * np.pi) * 0.3
                
                # Calculate final brightness
                final_brightness = brightness * horizontal_shade
                
                # Set stem colors for all pixels at once
                instate['mushroom_window'][y_indices, x_indices, 0] = 0.08 + mushroom['color_variation']  # Hue
                instate['mushroom_window'][y_indices, x_indices, 1] = 0.3  # Saturation
                instate['mushroom_window'][y_indices, x_indices, 2] = final_brightness  # Value
                instate['mushroom_window'][y_indices, x_indices, 3] = fade_factor  # Alpha

        
        # Draw mushroom cap
        cap_width = int(6 * mushroom['cap_width_ratio'] * current_size)
        cap_height = int(6 * current_size)
        
        if cap_width > 0 and cap_height > 0:
            cap_top = int(mushroom['pos_y'] - stem_height - cap_height)
            cap_center_y = mushroom['pos_y'] - stem_height - cap_height/2
            cap_center_x = mushroom['pos_x'] + sway_offset
            
            # Update mushroom render depth based on its VISUAL position (top of cap, not center)
            # Using cap_top for depth sorting (lower number = further back)
            mushroom['render_depth'] = cap_top
            
            
            # Define cap region
            cap_left = int(cap_center_x - cap_width/2)
            cap_right = int(cap_center_x + cap_width/2)
            
            # Enforce bounds
            cap_left = max(0, cap_left)
            cap_right = min(119, cap_right)
            cap_top = max(0, cap_top)
            cap_bottom = min(stem_top, 59)
            
            if cap_right > cap_left and cap_bottom > cap_top:
                # Define base colors
                if mushroom['color_scheme'] == 'brown':
                    base_hue = 0.08 + mushroom['color_variation']
                    base_saturation = 0.7
                    base_value = 0.8
                elif mushroom['color_scheme'] == 'red':
                    base_hue = 0.01 + mushroom['color_variation']
                    base_saturation = 0.8
                    base_value = 0.7
                elif mushroom['color_scheme'] == 'white':
                    base_hue = 0.05 + mushroom['color_variation']
                    base_saturation = 0.1
                    base_value = 0.9
                elif mushroom['color_scheme'] == 'yellow':
                    base_hue = 0.14 + mushroom['color_variation']
                    base_saturation = 0.7
                    base_value = 0.9
                elif mushroom['color_scheme'] == 'purple':
                    base_hue = 0.75 + mushroom['color_variation']
                    base_saturation = 0.5
                    base_value = 0.7
                else:
                    # Default fallback
                    base_hue = 0.1 + mushroom['color_variation']
                    base_saturation = 0.7
                    base_value = 0.8
                
                if mushroom['exploded']:
                    # Darkened color for exploded mushrooms
                    base_hue += 0.02
                    base_saturation = 0.3
                    base_value = 0.4
                
                # Generate spot pattern if needed
                has_pattern = mushroom['has_spots'] or mushroom['has_striations']
                uses_spots = mushroom['type'] in ['normal', 'round']
                
                # Vectorized approach - create coordinate grids
                y_indices, x_indices = np.mgrid[cap_top:cap_bottom, cap_left:cap_right]
                
                # Calculate normalized coordinates for all pixels at once
                dx = (x_indices - cap_center_x) / (cap_width/2)
                dy = (y_indices - cap_center_y) / (cap_height/2)
                dist = np.sqrt(dx**2 + dy**2)
                
                # Initialize mask for cap shape
                in_cap = np.zeros((cap_bottom-cap_top, cap_right-cap_left), dtype=bool)
                
                # Apply the appropriate shape mask based on mushroom type
                if mushroom['type'] == 'normal':
                    # Standard mushroom with rounded top becoming flatter with maturity
                    flatness = mushroom['growth_stage'] * 0.4
                    adjusted_dy = dy + flatness * (1 - dx**2) * (dy < 0)
                    in_cap = np.sqrt(dx**2 + adjusted_dy**2) <= 1.0
                
                elif mushroom['type'] == 'slim':
                    # Tall, conical mushroom
                    in_cap = np.sqrt(dx**2 + (dy*1.2)**2) <= 1.0
                    
                elif mushroom['type'] == 'round':
                    # Very round mushroom
                    in_cap = dist <= 1.0
                
                elif mushroom['type'] == 'umbrella':
                    # Umbrella-shaped with very flat top
                    flatness = 0.7
                    adjusted_dy = dy + flatness * (1 - dx**2) * (dy < 0)
                    # Thinner at the edges
                    in_cap = np.sqrt(dx**2 + adjusted_dy**2) <= (1.0 - 0.2 * np.abs(dx))
                
                elif mushroom['type'] == 'conical':
                    # Distinctly conical/pointy cap
                    # More pointed at top, wider at bottom
                    height_factor = 1.5 - dy  # Varies based on height
                    in_cap = np.sqrt((dx*height_factor)**2 + (dy*0.8)**2) <= 1.0
                
                elif mushroom['type'] == 'bell':
                    # Bell-shaped cap with curved sides
                    # More rounded at top, slightly concave sides
                    bell_shape = 1.0 - 0.3 * np.sin(dy * np.pi)
                    in_cap = np.sqrt((dx/bell_shape)**2 + dy**2) <= 1.0
                
                elif mushroom['type'] == 'flat':
                    # Very flat pancake-like cap
                    # Thin vertical height, wide horizontal
                    in_cap = (np.abs(dy) <= 0.5) & (np.abs(dx) <= (1.0 - np.abs(dy)/0.5 * 0.2))
                    
                    # Ensure the bottom of the cap reaches the stem
                    if mushroom['growth_stage'] > 0.5:  # Only for sufficiently grown mushrooms
                        stem_connection = (np.abs(dx) <= stem_width/(cap_width)) & (dy > 0) & (dy < 0.5)
                        in_cap = in_cap | stem_connection

                
                # Only process pixels within the cap shape
                where_in_cap = np.where(in_cap)
                if len(where_in_cap[0]) > 0:
                    # For cap shading
                    cap_dy = dy[where_in_cap]
                    cap_dx = dx[where_in_cap]
                    cap_dist = dist[where_in_cap]
                    
                    # Calculate spot pattern for all pixels at once
                    spot_value = np.zeros_like(cap_dist)
                    if has_pattern and uses_spots:
                        if mushroom['has_spots']:
                            # Simplify spot calculation for performance
                            spot_value = (np.sin(cap_dx * 10) * np.sin(cap_dy * 10)) * 0.5 + 0.5
                        elif mushroom['has_striations']:
                            # Simpler striations
                            angle = np.arctan2(cap_dy, cap_dx)
                            striation_density = 20
                            spot_value = (np.cos(angle * striation_density) * 0.5 + 0.5) * cap_dist
                    
                    # Cap shading based on shape
                    cap_shade = 0.7 + 0.3 * (1 - cap_dist**2)  # Brighter in center
                    
                    # Apply top highlight
                    highlight = np.exp(-((cap_dy + 0.5)**2 + cap_dx**2) / 0.3) * 0.3
                    
                    # Apply gill shadowing at bottom of cap
                    gill_shadow = (cap_dy > 0.2) * (cap_dy - 0.2) * 0.5
                    
                    # Calculate final cap brightness
                    brightness = base_value * cap_shade + highlight - gill_shadow
                    
                    # Get the actual pixel coordinates for assignment
                    y_pixels = y_indices[where_in_cap]
                    x_pixels = x_indices[where_in_cap]
                    
                    # Set cap colors with spots
                    instate['mushroom_window'][y_pixels, x_pixels, 0] = base_hue + spot_value * 0.05
                    instate['mushroom_window'][y_pixels, x_pixels, 1] = base_saturation - spot_value * 0.3
                    instate['mushroom_window'][y_pixels, x_pixels, 2] = brightness
                    instate['mushroom_window'][y_pixels, x_pixels, 3] = fade_factor
                    
                    # Draw gills where mature enough and at bottom of cap
                    gill_mask = (mushroom['growth_stage'] > 0.7) & (cap_dy > 0.0)
                    if np.any(gill_mask):
                        gill_indices = np.where(gill_mask)
                        gill_y = y_pixels[gill_indices]
                        gill_x = x_pixels[gill_indices]
                        gill_dy = cap_dy[gill_indices]
                        gill_dx = cap_dx[gill_indices]
                        gill_dist = cap_dist[gill_indices]
                        
                        # Angle from center for radial gills
                        angle = np.arctan2(gill_dy, gill_dx)
                        density = mushroom['gills_density']
                        gill_pattern = (np.cos(angle * density) * 0.5 + 0.5) ** 0.5
                        
                        # Gills darkening
                        gill_darkness = 0.3 + 0.2 * (gill_dist ** 0.5)
                        
                        # Update colors for gills
                        instate['mushroom_window'][gill_y, gill_x, 0] = base_hue - 0.01
                        instate['mushroom_window'][gill_y, gill_x, 1] *= 0.8
                        instate['mushroom_window'][gill_y, gill_x, 2] *= (1.0 - gill_darkness * gill_pattern)

                
                # Natural spore release from mature mushrooms
                if mushroom['spore_start'] and not mushroom['exploded']:
                    if len(instate['spore_system']['particles']) < instate['spore_system']['max_particles']:
                        # Gradually increase spore production

                        mushroom['spore_intensity'] = min(1.0, mushroom['spore_intensity'] + dt * 0.1)
                        
                        # Emit spores at a rate proportional to size
                        emit_count = int(mushroom['spore_intensity'] * mushroom['size'] * dt * 10)
                        
                        for _ in range(emit_count):
                            # Emit from bottom of cap
                            emit_angle = np.random.uniform(-np.pi/2 - 0.5, -np.pi/2 + 0.5)  # Downward angle
                            emit_x_offset = np.random.uniform(-cap_width/3, cap_width/3)
                            
                            spore = {
                                'pos_x': cap_center_x + emit_x_offset,
                                'pos_y': cap_bottom - 1,
                                'vel_x': np.cos(emit_angle) * np.random.uniform(1, 3),
                                'vel_y': np.sin(emit_angle) * np.random.uniform(1, 3),
                                'size': np.random.uniform(0.2, 0.5),
                                'color': mushroom['spore_color'] + np.random.uniform(-0.02, 0.02),
                                'alpha': np.random.uniform(0.3, 0.7),
                                'lifetime': 0.0,  # Ensure this is float
                                'max_lifetime': np.random.uniform(3, 7)
                            }
                            instate['spore_system']['particles'].append(spore)
    
    # Update toad behavior
    if elapsed_time >= instate['toad']['entry_time'] and not instate['toad']['active'] and not instate.get('toad_exited', False):
        # Activate toad and find initial position
        instate['toad']['active'] = True

        
        # Play the same hopping sound when toad first enters
        path=sound_path / 'bonk-46000.mp3'
        outstate['soundengine'].schedule_event(path, time.time(), 6)
        

        # Find the 5 largest mushrooms
        largest_mushrooms = sorted(
            [m for m in instate['mushrooms'] if m['growth_stage'] >= 0.5],  # Lower threshold to find more candidates
            key=lambda m: m['size'] * m['cap_width_ratio'],
            reverse=True
        )[:5]
        
        if largest_mushrooms:
            # Pick a random mushroom to start
            target = largest_mushrooms[np.random.randint(0, len(largest_mushrooms))]
            instate['toad']['largest_mushrooms'] = largest_mushrooms
            instate['toad']['current_mushroom'] = None  # Start with no current mushroom
            instate['toad']['next_mushroom'] = target
            
            # Position toad offscreen to hop in
            cap_center_x = target['pos_x'] + np.sin(target['sway_phase']) * target['sway_amount']
            stem_height = 8 * target['stem_height_ratio'] * target['size'] * target['growth_stage']
            cap_height = 6 * target['size'] * target['growth_stage']
            cap_top = target['pos_y'] - stem_height - cap_height
            
            instate['toad']['pos_x'] = -10  # Start offscreen
            instate['toad']['pos_y'] = cap_top  # Position directly on top of cap surface
            instate['toad']['target_x'] = cap_center_x
            instate['toad']['target_y'] = cap_top  # Position directly on top of cap surface
            instate['toad']['facing_right'] = True
            instate['toad']['state'] = 'hopping'
            instate['toad']['hop_progress'] = 0.0
            instate['toad']['hop_height'] = 15.0
            instate['toad']['z_index'] = 1000  # Set to a high value for initial hop
            
            # Add exit time right before explosion phase (around 65% of total duration)
            instate['toad']['exit_time'] = 0.65 * total_duration
            instate['toad']['has_exited'] = False
    
    if instate['toad']['active']:
        toad = instate['toad']
        
        # Check if it's time for the toad to exit the scene
        if elapsed_time >= toad['exit_time'] and not toad['has_exited']:
            # Begin exit hop toward right side of screen
            toad['state'] = 'hopping'
            toad['hop_progress'] = 0.0
            toad['target_x'] = 140  # Far off the right edge
            toad['target_y'] = toad['pos_y'] - 5  # Slightly upward trajectory for a big leap
            toad['facing_right'] = True
            toad['hop_height'] = 25.0  # Higher hop for dramatic exit
            toad['has_exited'] = True
            path=sound_path / 'bonk-46000.mp3'
            outstate['soundengine'].schedule_event(path, time.time(), 6)
        
        if toad['state'] == 'hopping':
            # Update hopping animation
            toad['hop_progress'] += dt * 0.5  # Control hop speed
            
            if toad['hop_progress'] >= 1.0:
                # Finished hop
                toad['hop_progress'] = 1.0
                
                # If this was the exit hop, permanently remove the toad
                if toad['has_exited']:
                    # Toad has left the scene permanently - mark as BOTH inactive AND exited
                    toad['active'] = False
                    instate['toad_exited'] = True  # Add this global flag to prevent reactivation
                else:
                    # Normal hop, now sit
                    toad['state'] = 'sitting'
                    toad['sit_timer'] = 0.0
                    toad['sit_duration'] = np.random.uniform(8.0, 14.0)  # Sit for 3-6 seconds
                    toad['current_mushroom'] = toad['next_mushroom']
                    
                    # Update z-index to match the mushroom's render depth
                    if toad['current_mushroom'] is not None:
                        toad['z_index'] = toad['current_mushroom']['render_depth'] - 0.1  # Slightly in front


            else:
                # In mid-hop, update position using arc trajectory
                t = toad['hop_progress']
                toad['pos_x'] = toad['pos_x'] * (1-t) + toad['target_x'] * t
                
                # Add vertical arc for hopping
                base_y = toad['pos_y'] * (1-t) + toad['target_y'] * t
                hop_arc = np.sin(np.pi * t) * toad['hop_height']
                toad['pos_y'] = base_y - hop_arc
        
        elif toad['state'] == 'sitting':
            # Update sitting timer
            toad['sit_timer'] += dt
            
            if toad['sit_timer'] >= toad['sit_duration']:
                # Time to hop to a new mushroom
                toad['state'] = 'hopping'
                toad['hop_progress'] = 0.0
                path=sound_path / 'bonk-46000.mp3'
                outstate['soundengine'].schedule_event(path, time.time(), 6)
                # Choose a new mushroom target (different from current)
                available_targets = [m for m in toad['largest_mushrooms'] if m != toad['current_mushroom']]
                if available_targets:
                    target = available_targets[np.random.randint(0, len(available_targets))]
                    toad['next_mushroom'] = target
                    
                    # Calculate target position on the new mushroom
                    cap_center_x = target['pos_x'] + np.sin(target['sway_phase']) * target['sway_amount']
                    stem_height = 8 * target['stem_height_ratio'] * target['size'] * target['growth_stage']
                    cap_height = 6 * target['size'] * target['growth_stage']
                    cap_top = target['pos_y'] - stem_height - cap_height
                    
                    # Set hop parameters
                    toad['target_x'] = cap_center_x
                    toad['target_y'] = cap_top  # Position directly on top of cap surface
                    toad['facing_right'] = toad['target_x'] > toad['pos_x']
                    toad['hop_height'] = np.random.uniform(10.0, 20.0)
                    # Update z-index to match the mushroom's render depth so toad sits on TOP of target mushroom
                    # Use a value slightly lower than the cap_top (further "up" in screen coordinates)
                    toad['z_index'] = target['render_depth'] - 1  # -1 ensures toad is drawn on top of the target mushroom

        # Determine z-order for toad rendering in relation to mushrooms
        # Only draw toad if it should appear in front of other mushrooms based on y-position
        should_draw_toad = True  # Default to drawing

        # When sitting on a mushroom, use z-index from current mushroom
        if toad['active']:
            # During hops, always show the toad
            if toad['state'] == 'hopping':
                should_draw_toad = True
            # When sitting, we should ALWAYS draw the toad
            # The previous logic was incorrectly hiding the toad sometimes
            elif toad['state'] == 'sitting' and toad['current_mushroom']:
                # Calculate toad position and bounds
                toad_x_min = toad['pos_x'] - toad['size'] * 3  # Approximate toad width
                toad_x_max = toad['pos_x'] + toad['size'] * 3
                
                # The problem is here - we shouldn't hide the toad completely when it's sitting
                # Instead, just ensure proper z-ordering with its current mushroom
                should_draw_toad = True
                
                # Check if any mushroom (except current one) should occlude the toad
                for mushroom in instate['mushrooms']:
                    if mushroom != toad['current_mushroom']:
                        # Calculate mushroom cap bounds
                        cap_width = 6 * mushroom['cap_width_ratio'] * mushroom['size'] * mushroom['growth_stage']
                        sway_offset = np.sin(mushroom['sway_phase']) * mushroom['sway_amount']
                        mushroom_x_min = mushroom['pos_x'] - cap_width/2 + sway_offset
                        mushroom_x_max = mushroom['pos_x'] + cap_width/2 + sway_offset
                        
                        # Check for horizontal overlap
                        has_overlap = not (toad_x_max < mushroom_x_min or toad_x_min > mushroom_x_max)
                        
                        # Only consider occluding if there's overlap AND mushroom is in front of toad's position
                        # (Remember that lower render_depth values mean further up on screen)
                        if has_overlap and mushroom['render_depth'] < toad['pos_y']:
                            should_draw_toad = True
                            break

        # Draw the toad if active
        if toad['active'] and should_draw_toad:
            # Define toad shape based on state
            toad_width = int(toad['size'] * 6.0)  # Wide toad body
            toad_height = int(toad['size'] * 4.0)  # Increased height for more toad-like proportions
            
            # Adjust size during hop
            if toad['state'] == 'hopping':
                # Squash and stretch during hop
                hop_factor = np.sin(np.pi * toad['hop_progress'])
                toad_width = int(toad['size'] * (6.0 - 1.0 * hop_factor))
                toad_height = int(toad['size'] * (4.0 + 1.0 * hop_factor))
            elif toad['state'] == 'sitting':
                # Slightly squatter when sitting
                toad_width = int(toad['size'] * 6.2)
                toad_height = int(toad['size'] * 3.8)
            
            # Define toad drawing region - position toad ON TOP of the mushroom
            toad_left = int(toad['pos_x'] - toad_width/2)
            toad_right = int(toad['pos_x'] + toad_width/2)
            toad_top = int(toad['pos_y'] - toad_height/2)  # Center vertically on the target y position
            toad_bottom = int(toad['pos_y'] + toad_height/2)
            
            # Enforce bounds
            toad_left = max(0, toad_left)
            toad_right = min(119, toad_right)
            toad_top = max(0, toad_top)
            toad_bottom = min(59, toad_bottom)
            
            if toad_right > toad_left and toad_bottom > toad_top:
                # Create coordinate grids for vectorized operations
                y_indices, x_indices = np.mgrid[toad_top:toad_bottom, toad_left:toad_right]
                
                # Calculate normalized coordinates
                norm_x = (x_indices - toad_left) / max(1, toad_right - toad_left - 1)
                if not toad['facing_right']:
                    norm_x = 1 - norm_x  # Flip horizontally
                norm_y = (y_indices - toad_top) / max(1, toad_bottom - toad_top - 1)
                
                # Calculate center positions for features
                center_x = 0.5
                center_y = 0.5
                
                # Define eye positions
                eye_distance = 0.20  # Distance between eyes
                eye_forward_shift = 0.05  # How far forward the eyes are positioned
                eye_y = center_y - 0.15
                
                if toad['facing_right']:
                    eye_left_x = center_x + eye_forward_shift - eye_distance/2
                    eye_right_x = center_x + eye_forward_shift + eye_distance/2
                else:
                    eye_left_x = center_x - eye_forward_shift - eye_distance/2
                    eye_right_x = center_x - eye_forward_shift + eye_distance/2
                
                # Define pupil positions and size
                pupil_size = 0.03
                pupil_shift = 0.04
                pupil_y = eye_y
                
                if toad['facing_right']:
                    pupil_left_x = eye_left_x + pupil_shift
                    pupil_right_x = eye_right_x + pupil_shift
                else:
                    pupil_left_x = eye_left_x - pupil_shift
                    pupil_right_x = eye_right_x - pupil_shift

                # Define mouth position
                mouth_y = center_y + 0.1
                mouth_width = 0.3
                mouth_x = center_x + (0.05 if toad['facing_right'] else -0.05)
                
                # Create toad body shape
                dx = norm_x - center_x
                dy = (norm_y - center_y) * 1.2  # Slightly elongate vertically
                
                # Main body shape
                body_dist = np.sqrt((dx/0.45)**2 + (dy/0.4)**2)
                
                # Add lumpy back
                back_lumps = 0.05 * np.sin(norm_x * 10) * np.exp(-(norm_y - 0.3)**2 / 0.1) * (norm_y < 0.5)
                
                # Create legs/feet bulges
                leg_front = 0.15 * np.exp(-((norm_x - (0.8 if toad['facing_right'] else 0.2))**2 + (norm_y - 0.7)**2) / 0.05)
                leg_back = 0.15 * np.exp(-((norm_x - (0.2 if toad['facing_right'] else 0.8))**2 + (norm_y - 0.7)**2) / 0.05)
                
                # Calculate distances for front leg
                front_leg_x_base = 0.7 if toad['facing_right'] else 0.3
                front_leg_y_base = 0.55
                front_leg_x_end = 0.8 if toad['facing_right'] else 0.2
                front_leg_y_end = 0.8
                
                front_leg_length = np.sqrt((front_leg_x_end - front_leg_x_base)**2 + 
                                          (front_leg_y_end - front_leg_y_base)**2)
                front_leg_t = np.maximum(0, np.minimum(1, 
                                         ((norm_x - front_leg_x_base) * (front_leg_x_end - front_leg_x_base) + 
                                          (norm_y - front_leg_y_base) * (front_leg_y_end - front_leg_y_base)) / 
                                         (front_leg_length**2)))
                front_leg_px = front_leg_x_base + front_leg_t * (front_leg_x_end - front_leg_x_base)
                front_leg_py = front_leg_y_base + front_leg_t * (front_leg_y_end - front_leg_y_base)
                front_leg_dist = np.sqrt((norm_x - front_leg_px)**2 + (norm_y - front_leg_py)**2)
                front_leg = (front_leg_dist < 0.08)
                
                # Calculate distances for back leg
                back_leg_x_base = 0.3 if toad['facing_right'] else 0.7
                back_leg_y_base = 0.55
                back_leg_x_end = 0.2 if toad['facing_right'] else 0.8
                back_leg_y_end = 0.8
                
                back_leg_length = np.sqrt((back_leg_x_end - back_leg_x_base)**2 + 
                                         (back_leg_y_end - back_leg_y_base)**2)
                back_leg_t = np.maximum(0, np.minimum(1, 
                                        ((norm_x - back_leg_x_base) * (back_leg_x_end - back_leg_x_base) + 
                                         (norm_y - back_leg_y_base) * (back_leg_y_end - back_leg_y_base)) / 
                                        (back_leg_length**2)))
                back_leg_px = back_leg_x_base + back_leg_t * (back_leg_x_end - back_leg_x_base)
                back_leg_py = back_leg_y_base + back_leg_t * (back_leg_y_end - back_leg_y_base)
                back_leg_dist = np.sqrt((norm_x - back_leg_px)**2 + (norm_y - back_leg_py)**2)
                back_leg = (back_leg_dist < 0.08)
                
                # Add feet
                front_foot = np.exp(-((norm_x - front_leg_x_end)**2 + (norm_y - front_leg_y_end)**2) / 0.005) > 0.1
                back_foot = np.exp(-((norm_x - back_leg_x_end)**2 + (norm_y - back_leg_y_end)**2) / 0.005) > 0.1
                
                # Combine shapes for toad body
                toad_shape = body_dist - back_lumps - leg_front - leg_back
                in_toad_body = (toad_shape <= 1.0) | front_leg | back_leg | front_foot | back_foot
                
                # Check for eyes
                eye_left_dist = np.sqrt((norm_x - eye_left_x)**2 + (norm_y - eye_y)**2)
                eye_right_dist = np.sqrt((norm_x - eye_right_x)**2 + (norm_y - eye_y)**2)
                in_eye = (eye_left_dist <= 0.08) | (eye_right_dist <= 0.08)
                
                # Check for pupils
                pupil_left_dist = np.sqrt((norm_x - pupil_left_x)**2 + (norm_y - pupil_y)**2)
                pupil_right_dist = np.sqrt((norm_x - pupil_right_x)**2 + (norm_y - pupil_y)**2)
                in_pupil = (pupil_left_dist <= pupil_size) | (pupil_right_dist <= pupil_size)
                
                # Check for mouth
                mouth_shape = np.abs(norm_x - mouth_x) / mouth_width + np.abs(norm_y - mouth_y) * 5
                in_mouth = (mouth_shape < 1.0) & (norm_y > mouth_y - 0.02) & (norm_y < mouth_y + 0.02)
                
                # Create shading/pattern elements
                pattern = (np.sin(norm_x * 15) * np.sin(norm_y * 15)) * 0.1
                speckles = np.random.random(norm_x.shape) * 0.1
                
                # Shading
                toad_shade = 0.7 - body_dist * 0.3 + back_lumps * 0.3
                # Darker belly
                toad_shade = np.where(norm_y > 0.5, toad_shade * 0.8, toad_shade)
                
                # Create HSVA arrays for each component
                h_array = np.zeros_like(norm_x)
                s_array = np.zeros_like(norm_x)
                v_array = np.zeros_like(norm_x)
                a_array = np.zeros_like(norm_x)
                
                # Apply pupil colors (black)
                h_array[in_pupil] = 0.0
                s_array[in_pupil] = 0.0
                v_array[in_pupil] = 0.0
                a_array[in_pupil] = fade_factor
                
                # Apply eye colors (white)
                h_array[in_eye & ~in_pupil] = 0.15
                s_array[in_eye & ~in_pupil] = 0.1
                v_array[in_eye & ~in_pupil] = 0.9
                a_array[in_eye & ~in_pupil] = fade_factor
                
                # Apply mouth colors (dark)
                h_array[in_mouth & ~in_pupil & ~in_eye] = 0.0
                s_array[in_mouth & ~in_pupil & ~in_eye] = 0.0
                v_array[in_mouth & ~in_pupil & ~in_eye] = 0.2
                a_array[in_mouth & ~in_pupil & ~in_eye] = fade_factor
                
                # Apply toad body colors (green with patterns)
                body_mask = in_toad_body & ~in_pupil & ~in_eye & ~in_mouth
                h_array[body_mask] = 0.3 + pattern[body_mask] - back_lumps[body_mask] * 0.2
                s_array[body_mask] = 0.7 + speckles[body_mask]
                v_array[body_mask] = toad_shade[body_mask]
                a_array[body_mask] = fade_factor
                
                # Apply the colors to the mushroom window
                instate['mushroom_window'][y_indices, x_indices, 0] = np.where(
                    in_toad_body | in_eye | in_pupil | in_mouth,
                    h_array, 
                    instate['mushroom_window'][y_indices, x_indices, 0]
                )
                
                instate['mushroom_window'][y_indices, x_indices, 1] = np.where(
                    in_toad_body | in_eye | in_pupil | in_mouth,
                    s_array, 
                    instate['mushroom_window'][y_indices, x_indices, 1]
                )
                
                instate['mushroom_window'][y_indices, x_indices, 2] = np.where(
                    in_toad_body | in_eye | in_pupil | in_mouth,
                    v_array, 
                    instate['mushroom_window'][y_indices, x_indices, 2]
                )
                
                instate['mushroom_window'][y_indices, x_indices, 3] = np.where(
                    in_toad_body | in_eye | in_pupil | in_mouth,
                    a_array, 
                    instate['mushroom_window'][y_indices, x_indices, 3]
                )

    # Update and draw all spores
    new_particles = []
    
    # Update and draw all spores
    if instate['spore_system']['particles']:
        # Convert particle list to numpy arrays for vectorized operations
        particles_array = np.array([(
            spore['pos_x'], 
            spore['pos_y'], 
            spore['vel_x'], 
            spore['vel_y'], 
            spore['size'],
            spore['color'],
            spore['alpha'],
            spore['lifetime'],
            spore['max_lifetime']
        ) for spore in instate['spore_system']['particles']], 
        dtype=[
            ('pos_x', 'f4'), 
            ('pos_y', 'f4'), 
            ('vel_x', 'f4'), 
            ('vel_y', 'f4'), 
            ('size', 'f4'),
            ('color', 'f4'),
            ('alpha', 'f4'),
            ('lifetime', 'f4'),
            ('max_lifetime', 'f4')
        ])
        
        # Update age
        particles_array['lifetime'] += dt
        
        # Filter out expired particles
        valid_mask = particles_array['lifetime'] <= particles_array['max_lifetime']
        particles_array = particles_array[valid_mask]
        
        if len(particles_array) > 0:
            # Calculate alpha based on lifetime
            age_factor = np.ones(len(particles_array))
            fade_in_mask = particles_array['lifetime'] < 0.5
            fade_out_mask = particles_array['lifetime'] > particles_array['max_lifetime'] - 1.0
            
            # Apply fade in and fade out
            age_factor[fade_in_mask] = particles_array['lifetime'][fade_in_mask] / 0.5
            age_factor[fade_out_mask] = (particles_array['max_lifetime'][fade_out_mask] - 
                                         particles_array['lifetime'][fade_out_mask]) / 1.0
            
            # Apply physics to all particles at once
            # Gravity affects velocity
            particles_array['vel_y'] += 2.0 * dt
            
            # Apply wind and updraft
            particles_array['vel_x'] += current_wind * dt
            # Add random component individually (not vectorizable in a simple way)
            particles_array['vel_x'] += np.random.normal(0, 0.5, len(particles_array)) * dt
            particles_array['vel_y'] -= instate['spore_system']['updraft'] * dt
            
            # Apply drag
            particles_array['vel_x'] *= 0.99
            particles_array['vel_y'] *= 0.99
            
            # Update position
            particles_array['pos_x'] += particles_array['vel_x'] * dt
            particles_array['pos_y'] += particles_array['vel_y'] * dt
            
            # Filter out particles that went out of bounds
            bounds_mask = ((particles_array['pos_x'] >= 0) & 
                           (particles_array['pos_x'] < 120) & 
                           (particles_array['pos_y'] >= 0) & 
                           (particles_array['pos_y'] < 60))
            particles_array = particles_array[bounds_mask]
            
            if len(particles_array) > 0:
                # Recompute age factor for remaining particles
                age_factor = age_factor[bounds_mask]
                
                # Round positions to integers for drawing
                x_indices = np.clip(particles_array['pos_x'].astype(np.int32), 0, 119)
                y_indices = np.clip(particles_array['pos_y'].astype(np.int32), 0, 59)
                
                # Calculate final alpha
                final_alpha = particles_array['alpha'] * age_factor * fade_factor
                
                # Create a mask to handle overlapping particles at the same pixel
                # (we'll use a simple approach - later particles overwrite earlier ones)
                positions = np.column_stack((y_indices, x_indices))
                unique_positions, position_indices = np.unique(positions, return_index=True, axis=0)
                
                # Draw only unique positions (last particle at each position)
                for idx in position_indices:
                    y, x = y_indices[idx], x_indices[idx]
                    instate['mushroom_window'][y, x, 0] = particles_array['color'][idx]
                    instate['mushroom_window'][y, x, 1] = 0.7
                    instate['mushroom_window'][y, x, 2] = 0.9
                    instate['mushroom_window'][y, x, 3] = final_alpha[idx]
            
            # Convert back to list format
            new_particles = []
            for i in range(len(particles_array)):
                spore = {
                    'pos_x': float(particles_array['pos_x'][i]),
                    'pos_y': float(particles_array['pos_y'][i]),
                    'vel_x': float(particles_array['vel_x'][i]),
                    'vel_y': float(particles_array['vel_y'][i]),
                    'size': float(particles_array['size'][i]),
                    'color': float(particles_array['color'][i]),
                    'alpha': float(particles_array['alpha'][i]),
                    'lifetime': float(particles_array['lifetime'][i]),
                    'max_lifetime': float(particles_array['max_lifetime'][i])
                }
                new_particles.append(spore)
        else:
            new_particles = []
    else:
        new_particles = []
    
    # Update particle list with maximum size limit for performance
    max_particles = min(1000, instate['spore_system']['max_particles'])
    if len(new_particles) > max_particles:
        # If too many particles, keep the newest ones
        new_particles = new_particles[-max_particles:]
    
    instate['spore_system']['particles'] = new_particles
    
    # Convert HSVA to RGBA for rendering
    rgb = color.hsv2rgb(instate['mushroom_window'][..., 0:3])
    alpha = instate['mushroom_window'][..., 3:4]
    rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)
    
    # Update the image plane
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['mushroom_plane'],
        rgb_out
    )
    
    instate['last_update'] = current_time