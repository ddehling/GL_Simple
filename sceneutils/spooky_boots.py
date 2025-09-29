import time
import numpy as np
from skimage import color
from pathlib import Path
ParentPath = Path(__file__).parent.parent
media_path = ParentPath / 'media'
sound_path = media_path / 'sounds'


def ghostly_boots(instate, outstate):
    if instate['count'] == 0:
        # Initialize parameters
        instate['boots_window'] = np.zeros((60, 120, 4))  # HSVA format
        instate['boots_plane'] = outstate['render'].create_image_plane(
            np.zeros((60, 120, 4), dtype=np.uint8),
            position=(0, 0, 19.7),  # In front of aurora but behind meteors
            rotation=(0, 0, 0),
            scale=(3,3)
        )
        
        # Boot parameters
        instate['position'] = -20  # Start off-screen to the left
        # ADJUSTED: Slow down the speed to cross screen in ~20 seconds (140 pixels in ~20 seconds)
        instate['speed'] = 70.0 / 20.0  # 7 pixels per second (140 pixels / 20 seconds = 7 pps)
        instate['boot_separation'] = 12  # Pixels between boots
        instate['step_phase'] = 0  # For walking animation
        # ADJUSTED: Slow down step speed to match slower movement
        instate['step_speed'] = 1.5  # Steps per second
        instate['step_height'] = 4  # Maximum height of step
        instate['opacity'] = 0.0  # Start invisible for fade-in
        # ADJUSTED: Set maximum opacity higher for more visible boots
        instate['max_opacity'] = 1  # Maximum opacity value
        
        # Boot design parameters
        instate['boot_width'] = 10
        instate['boot_height'] = 15
        instate['heel_height'] = 3
        
        # Sound parameters
        instate['last_step_time'] = time.time()
        instate['next_step_delay'] = 0.5  # Time between footsteps
        instate['boot_color'] = 0.08 + np.random.random() * 0.03  # Brown hue
        
        # Load footstep sound
        path = sound_path / 'Boot_Footsteps.wav'
        if not path.exists():
            path = sound_path / 'Footstep.wav'  # Fallback sound
        instate['footstep_sound'] = path
        
        instate['start_time'] = time.time()
        instate['last_update'] = time.time()
        return

    if instate['count'] == -1:
        outstate['render'].remove_image_plane(instate['boots_plane'])
        return

    # Get time delta
    current_time = time.time()
    dt = current_time - instate['last_update']
    instate['last_update'] = current_time
    elapsed_time = current_time - instate['start_time']
    total_duration = instate.get('duration', 20)
    
    # Calculate fade factors
    fade_in_time = 2.0
    fade_out_start = total_duration - 2.0
    max_opacity = instate['max_opacity']  # Use the configured maximum opacity
    
    if elapsed_time < fade_in_time:
        instate['opacity'] = (elapsed_time / fade_in_time) * max_opacity
    elif elapsed_time > fade_out_start:
        instate['opacity'] = ((total_duration - elapsed_time) / 2.0) * max_opacity
    else:
        instate['opacity'] = max_opacity
    
    # Clear the window
    window = instate['boots_window']
    window.fill(0)
    
    # Update position
    instate['position'] += instate['speed'] * dt
    
    # Update step phase
    instate['step_phase'] += dt * instate['step_speed'] * np.pi
    
    # Play footstep sound
    step_cycle_position = instate['step_phase'] % (2 * np.pi)
    if (step_cycle_position < np.pi/4 or (step_cycle_position > np.pi and step_cycle_position < np.pi + np.pi/4)) and \
       (current_time - instate['last_step_time'] > instate['next_step_delay']):
        # Play footstep sound
        outstate['soundengine'].schedule_event(instate['footstep_sound'], current_time, 1)
        instate['last_step_time'] = current_time
    
    # Base positions
    pos_x = int(instate['position'])
    base_y = 45  # Ground level
    
    # If boots have moved off screen to the right, end the event
    if pos_x > 140:
        instate['count'] = -1
        return
    
    # IMPROVED WALKING ANIMATION:
    # Left and right boot phases are opposite
    left_phase = instate['step_phase']
    right_phase = instate['step_phase'] + np.pi  # 180 degrees out of phase
    
    # Forward movement during each step
    left_x_offset = np.sin(left_phase) * 6
    right_x_offset = np.sin(right_phase) * 6
    
    # Up-down movement during step (only up when moving forward)
    left_y_offset = max(0, np.sin(left_phase)) * instate['step_height']
    right_y_offset = max(0, np.sin(right_phase)) * instate['step_height']
    
    # Boot positions (left and right)
    left_x = int(pos_x - instate['boot_separation'] // 2 + left_x_offset)
    right_x = int(pos_x + instate['boot_separation'] // 2 + right_x_offset)
    
    left_y = int(base_y - left_y_offset)
    right_y = int(base_y - right_y_offset)
    
    # Draw ghostly boots
    draw_boot(window, left_x, left_y, instate['boot_width'], instate['boot_height'], 
              instate['heel_height'], instate['boot_color'], instate['opacity'], True)  # Left boot
    draw_boot(window, right_x, right_y, instate['boot_width'], instate['boot_height'], 
              instate['heel_height'], instate['boot_color'], instate['opacity'], False)  # Right boot
    
    # Add ghostly trail effects
    trail_length = 10
    trail_opacity_factor = 0.6
    
    for i in range(1, trail_length):
        trail_opacity = instate['opacity'] * trail_opacity_factor * (1 - i/trail_length)
        trail_x_offset = -i * 2  # Trail extends behind boots
        
        # Only draw trail if boots are visible
        if left_x + trail_x_offset > -instate['boot_width']:
            draw_boot(window, left_x + trail_x_offset, left_y, 
                      instate['boot_width'], instate['boot_height'], 
                      instate['heel_height'], instate['boot_color'], trail_opacity, True)
        
        if right_x + trail_x_offset > -instate['boot_width']:
            draw_boot(window, right_x + trail_x_offset, right_y, 
                      instate['boot_width'], instate['boot_height'], 
                      instate['heel_height'], instate['boot_color'], trail_opacity, False)
    
    # Add dust cloud particles when stepping
    if left_y_offset < 0.5:  # Left boot touching ground
        add_dust_particles(window, left_x, base_y, instate['opacity'])
    
    if right_y_offset < 0.5:  # Right boot touching ground
        add_dust_particles(window, right_x, base_y, instate['opacity'])
    
    # Convert HSVA to BGRA for rendering
    rgb = color.hsv2rgb(window[..., 0:3])
    alpha = window[..., 3:4]
    rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)
    
    # Update the image plane
    outstate['render'].update_image_plane_texture(
        instate['boots_plane'], 
        rgb_out[:,:,:]
    )

def draw_boot(window, x, y, width, height, heel_height, color_hue, opacity, is_left=True):
    """Draw a cowboy boot at the specified position"""
    # Define boot shape parameters
    toe_length = 3
    ankle_height = height // 3
    shaft_height = height - ankle_height
    
    # Calculate bounds to avoid out-of-bounds drawing
    x_min = max(0, x - width//2 - 5)  # Expanded bounds for wider shaft
    x_max = min(120, x + width//2 + toe_length + 5)
    y_min = max(0, y - height - 2)
    y_max = min(60, y + heel_height + 2)
    
    # Skip if out of bounds
    if x_max <= x_min or y_max <= y_min:
        return
    
    # Generate a coordinate grid
    y_coords, x_coords = np.meshgrid(
        np.arange(y_min, y_max),
        np.arange(x_min, x_max),
        indexing='ij'
    )
    
    # Boot direction determines which side the toe and heel are on
    boot_direction = 1 if is_left else -1
    
    # Create boot mask with clearer shape
    boot_mask = np.zeros_like(y_coords, dtype=bool)
    pattern_mask = np.zeros_like(y_coords, dtype=bool)
    
    # Build the boot shape pixel by pixel for better control
    for py in range(y_min, y_max):
        for px in range(x_min, x_max):
            # Relative coordinates
            rel_y = py - y
            rel_x = (px - x) * boot_direction  # Apply direction to make mirroring work correctly
            
            # Foot part (bottom of boot)
            if -ankle_height <= rel_y <= 0:
                # Main foot
                if -width//2 <= rel_x <= width//2:
                    boot_mask[py-y_min, px-x_min] = True
                # Toe extension (curved shape)
                elif width//2 < rel_x <= width//2 + toe_length:
                    # Create a curved toe shape
                    toe_curve = (rel_y + ankle_height) / ankle_height
                    if toe_curve >= (rel_x - width//2) / (toe_length * 2):
                        boot_mask[py-y_min, px-x_min] = True
            
            # Heel part
            elif 0 < rel_y <= heel_height:
                heel_width = width // 3
                if -width//2 <= rel_x <= -width//2 + heel_width:
                    boot_mask[py-y_min, px-x_min] = True
            
            # Shaft part (upper part of boot)
            elif -height <= rel_y < -ankle_height:
                # Calculate how wide the shaft should be at this height
                # Wider at top, narrower at bottom
                height_ratio = abs(rel_y + ankle_height) / shaft_height
                shaft_width = width//2 + int(height_ratio * 2)
                
                # Shaft with slight curve on top
                if -shaft_width <= rel_x <= shaft_width:
                    # Add a slight curve at the top of the shaft
                    if rel_y < -height + 4:
                        # Round the top edges
                        edge_curve = 1 - (abs(rel_x) / shaft_width)
                        if edge_curve > (rel_y + height) / 4:
                            boot_mask[py-y_min, px-x_min] = True
                    else:
                        boot_mask[py-y_min, px-x_min] = True
            
            # Add stitching pattern at top of shaft
            if -height + 2 <= rel_y <= -height + 4:
                if -shaft_width + 1 <= rel_x <= shaft_width - 1:
                    pattern_mask[py-y_min, px-x_min] = True
            
            # Add decorative stitching on shaft
            if -height + shaft_height//3 <= rel_y <= -height + shaft_height//3 + 1:
                if -width//2 - 1 <= rel_x <= width//2 + 1:
                    pattern_mask[py-y_min, px-x_min] = True
            
            # Add seam line along the back of shaft
            if -height <= rel_y <= -ankle_height:
                if -1 <= rel_x <= 1:
                    pattern_mask[py-y_min, px-x_min] = True
            
            # Add detail at ankle
            if -ankle_height - 2 <= rel_y <= -ankle_height + 2:
                if -width//2 - 1 <= rel_x <= width//2 + 1:
                    pattern_mask[py-y_min, px-x_min] = True
    
    # Apply boot color with ghostly effect
    base_color = np.array([
        color_hue,              # Hue (brown)
        0.8,                    # Saturation
        0.4,                    # Value
        opacity * 0.8           # Alpha (ghostly)
    ])
    
    # Pattern color (slightly different shade)
    pattern_color = np.array([
        color_hue + 0.02,       # Hue (slightly different)
        0.9,                    # Saturation
        0.5,                    # Value
        opacity * 0.9           # Alpha (slightly more visible)
    ])
    
    # Apply colors
    if np.any(boot_mask):
        window[y_coords[boot_mask], x_coords[boot_mask]] = base_color
    
    if np.any(pattern_mask):
        window[y_coords[pattern_mask], x_coords[pattern_mask]] = pattern_color
    
    # Add ghostly glow effect
    glow_radius = 2
    for dy in range(-glow_radius, glow_radius + 1):
        for dx in range(-glow_radius, glow_radius + 1):
            if dx == 0 and dy == 0:
                continue
                
            # Calculate distance for glow falloff
            distance = np.sqrt(dx*dx + dy*dy) / glow_radius
            glow_intensity = (1 - distance) * 0.3 * opacity
            
            if glow_intensity <= 0:
                continue
                
            # Apply glow around boot
            glow_mask = np.zeros_like(boot_mask)
            
            # Shift the boot mask by (dx, dy)
            shift_y_min = max(0, y_min - dy)
            shift_y_max = min(60, y_max - dy)
            shift_x_min = max(0, x_min - dx)
            shift_x_max = min(120, x_max - dx)
            
            # Skip if shift is out of bounds
            if shift_y_max <= shift_y_min or shift_x_max <= shift_x_min:
                continue
                
            # Get the coordinates for the shifted region
            shifted_y_coords, shifted_x_coords = np.meshgrid(
                np.arange(shift_y_min, shift_y_max),
                np.arange(shift_x_min, shift_x_max),
                indexing='ij'
            )
            
            # Calculate source coordinates
            source_y = shifted_y_coords + dy
            source_x = shifted_x_coords + dx
            
            # Create mask for valid source coordinates
            valid_source = (
                (source_y >= y_min) & (source_y < y_max) &
                (source_x >= x_min) & (source_x < x_max)
            )
            
            if not np.any(valid_source):
                continue
                
            # Get boot mask values for valid source coordinates
            source_indices = (
                source_y[valid_source] - y_min,
                source_x[valid_source] - x_min
            )
            
            # Apply glow to shifted positions
            glow_positions = (
                shifted_y_coords[valid_source],
                shifted_x_coords[valid_source]
            )
            
            # Only apply glow around the boot (not inside it)
            glow_mask = np.zeros_like(boot_mask[0:shift_y_max-shift_y_min, 0:shift_x_max-shift_x_min], dtype=bool)
            valid_boot_values = boot_mask[source_indices]
            glow_mask[glow_positions[0]-shift_y_min, glow_positions[1]-shift_x_min] = valid_boot_values
            
            # Apply glow color
            glow_color = np.array([
                color_hue,         # Hue
                0.5,               # Saturation
                0.7,               # Value (brighter than boot)
                glow_intensity     # Alpha (faint glow)
            ])
            
            if np.any(glow_mask):
                y_indices = shift_y_min + np.arange(glow_mask.shape[0])
                x_indices = shift_x_min + np.arange(glow_mask.shape[1])
                yy, xx = np.meshgrid(y_indices, x_indices, indexing='ij')
                window[yy[glow_mask], xx[glow_mask]] = glow_color

def add_dust_particles(window, x, y, opacity):
    """Add dust particles when boot hits the ground"""
    num_particles = int(8 * opacity)
    
    for _ in range(num_particles):
        # Random position around boot
        dx = np.random.randint(-10, 10)
        dy = np.random.randint(-2, 3)
        px = x + dx
        py = y + dy
        
        # Skip if out of bounds
        if px < 0 or px >= 120 or py < 0 or py >= 60:
            continue
            
        # Random size
        size = np.random.randint(1, 3)
        
        # Draw dust particle (small circle)
        for pdy in range(-size, size + 1):
            for pdx in range(-size, size + 1):
                particle_x = px + pdx
                particle_y = py + pdy
                
                if 0 <= particle_x < 120 and 0 <= particle_y < 60:
                    dist = np.sqrt(pdx*pdx + pdy*pdy)
                    if dist <= size:
                        # Dust color (grayish-brown, partially transparent)
                        intensity = (1 - dist/size) * opacity * 0.7
                        if intensity > 0:
                            window[particle_y, particle_x] = [
                                0.1,        # Hue (grayish-brown)
                                0.3,        # Saturation (desaturated)
                                0.8,        # Value
                                intensity   # Alpha
                            ]