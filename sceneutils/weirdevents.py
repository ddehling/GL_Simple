import time
import numpy as np
from skimage import color
from pathlib import Path
import cv2
ParentPath = Path(__file__).parent.parent
media_path = ParentPath / 'media'
sound_path = media_path / 'sounds'


def eye(instate, outstate):
    if instate['count'] == 0:
        # Initialize eye parameters
        instate['eye_window'] = np.zeros((60, 120, 4))  # HSVA format
        instate['eye_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((60, 120, 4), dtype=np.uint8),
            position=(0, 0, 20),  # Positioned relatively close to viewer
            rotation=(0, 0, 0),
            scale=(3, 3)
        )
        instate['start_time'] = time.time()
        instate['last_movement_time'] = time.time()
        instate['target_x'] = 0
        instate['target_y'] = 0
        instate['current_x'] = 0
        instate['current_y'] = 0
        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['eye_plane'])
        return

    current_time = time.time()
    elapsed_time = current_time - instate['start_time']
    total_duration = instate.get('duration', 30)

    # Movement logic
    movement_interval = outstate.get('eye_movement_interval', 3.0)  # Time between new target positions
    movement_speed = outstate.get('eye_movement_speed', 2.0)  # Speed of iris movement
    
    # Check if it's time to get a new target position
    if current_time - instate['last_movement_time'] > movement_interval:
        # Calculate maximum safe radius for iris movement
        angle = np.random.random() * 2 * np.pi
        max_radius = 0.7  # Reduce to 70% of possible range to ensure iris stays in bounds
        r = np.random.random() * max_radius
        
        # Scale x and y differently due to elliptical shape
        instate['target_x'] = r * np.cos(angle) * 1.5  # x can move more due to wider eye
        instate['target_y'] = r * np.sin(angle)
        
        instate['last_movement_time'] = current_time

    # Smoothly move current position toward target
    dx = instate['target_x'] - instate['current_x']
    dy = instate['target_y'] - instate['current_y']
    distance = np.sqrt(dx*dx + dy*dy)
    
    if distance > 0.001:  # Only move if we're not already very close to target
        move_amount = min(distance, movement_speed * (current_time - instate['last_update_time'])) if 'last_update_time' in instate else 0
        if distance > 0:  # Prevent division by zero
            instate['current_x'] += (dx / distance) * move_amount
            instate['current_y'] += (dy / distance) * move_amount

    instate['last_update_time'] = current_time

    # Calculate fade factor
    fade_duration = 5.0
    if elapsed_time < fade_duration:
        fade_factor = elapsed_time / fade_duration
    elif elapsed_time > (total_duration - fade_duration):
        fade_factor = (total_duration - elapsed_time) / fade_duration
    else:
        fade_factor = 1.0
    fade_factor = np.clip(fade_factor, 0, 1)

    # Get eye parameters from outstate or use defaults
    pupil_size = outstate.get('eye_pupil_size', 1.0)  # 0 to 1 range
    
    # Clear the window
    instate['eye_window'].fill(0)
    
    # Calculate centers and constraints
    center_x, center_y = 59.5, 29  # Center of the eye
    
    # Calculate iris position ensuring it stays within ellipse bounds
    max_x_offset = 35  # Maximum horizontal movement (pixels)
    max_y_offset = 15  # Maximum vertical movement (pixels)
    
    iris_x = center_x + (instate['current_x'] * max_x_offset)
    iris_y = center_y + (instate['current_y'] * max_y_offset)
    
    # Calculate distortion factor based on horizontal position
    h_stretch = 1.0
    v_stretch = 1.0 + (0.4 * abs(instate['current_x']))

    # Helper function for eye boundary check
    def is_within_eye_ellipse(x, y):
        dx = (x - center_x) / 60  # Divided by half-width
        dy = (y - center_y) / 30  # Divided by half-height
        return (dx*dx + dy*dy) <= 1

    # Draw white of the eye (sclera)
# Create coordinate grids directly (or use the ones already created)
    y_coords, x_coords = np.mgrid[0:60, 0:120]

    # Calculate normalized distances from center
    dx = (x_coords - center_x) / 60
    dy = (y_coords - center_y) / 30
    dist = np.sqrt(dx*dx + dy*dy)

    # Create mask for points inside the ellipse
    eye_mask = dist <= 1

    # Apply color to all masked points at once
    instate['eye_window'][eye_mask] = [0.6, 0.5, 0.1, fade_factor*0.15]

    # Only draw iris and pupil if their center is within the eye ellipse
    if is_within_eye_ellipse(iris_x, iris_y):
        # Create a grid for the iris area
        iris_radius = 30
        
        # Define the bounding box
        y_min = max(0, int(iris_y-iris_radius))
        y_max = min(60, int(iris_y+iris_radius))
        x_min = max(0, int(iris_x-iris_radius))
        x_max = min(120, int(iris_x+iris_radius))
        
        # Create coordinate grids for the bounding box
        y_slice, x_slice = np.mgrid[y_min:y_max, x_min:x_max]
        
        # Calculate distances and angles for all points
        dx = x_slice - iris_x
        dy = y_slice - iris_y
        
        # Apply distortion
        dist = np.sqrt((dx/h_stretch)**2 + (dy/v_stretch)**2)
        
        # Create masks for valid points
        iris_mask = dist <= iris_radius
        
        # Only calculate for points within both iris radius and eye ellipse
        within_eye = np.zeros_like(iris_mask, dtype=bool)
        
        # Vectorize the eye ellipse check (assuming this is the function definition):
        # def is_within_eye_ellipse(x, y):
        #     dx = (x - center_x) / 60
        #     dy = (y - center_y) / 30
        #     return (dx*dx + dy*dy) <= 1
        
        eye_dx = (x_slice - center_x) / 60
        eye_dy = (y_slice - center_y) / 30
        within_eye = (eye_dx**2 + eye_dy**2) <= 1
        
        # Combine the masks
        valid_points = iris_mask & within_eye
        
        if np.any(valid_points):
            # Calculate angles only for valid points
            angles = np.arctan2(dy[valid_points]/v_stretch, dx[valid_points]/h_stretch)
            
            # Calculate pattern
            dist_ratio = dist[valid_points]/iris_radius
            pattern = (np.sin(angles * 8) * 0.1) + (dist_ratio * 0.2)
            
            # Set colors for valid points
            instate['eye_window'][y_slice[valid_points], x_slice[valid_points], 0] = 0.55 + pattern
            instate['eye_window'][y_slice[valid_points], x_slice[valid_points], 1] = 0.7
            instate['eye_window'][y_slice[valid_points], x_slice[valid_points], 2] = 0.5
            instate['eye_window'][y_slice[valid_points], x_slice[valid_points], 3] = fade_factor

        # Draw pupil with distortion
        base_pupil_size = outstate.get('eye_pupil_size', 1.0)
    
    # Slow subtle breathing-like variation (period of about 4 seconds)
        breathing_variation = np.sin(current_time * 1.5) * 0.1
        
        # Occasional rapid contractions (blinks)
        blink_interval = outstate.get('eye_blink_interval', 7.0)
        blink_phase = ((current_time - instate['start_time']) % blink_interval) / blink_interval
        if blink_phase < 0.1:  # Quick contraction during blink
            blink_variation = -0.3 * np.sin(blink_phase * np.pi * 10)
        else:
            blink_variation = 0
        
        # Combine all variations
        pupil_size = base_pupil_size + breathing_variation + blink_variation
        pupil_size = np.clip(pupil_size, 0.3, 1.0)  # Ensure pupil stays within reasonable bounds

        # Use pupil_size in the pupil drawing section
        pupil_radius = 15 * pupil_size

# Define the bounding box for the pupil
    y_min = max(0, int(iris_y-pupil_radius))
    y_max = min(60, int(iris_y+pupil_radius))
    x_min = max(0, int(iris_x-pupil_radius))
    x_max = min(120, int(iris_x+pupil_radius))

    # Create coordinate grids for the bounding box
    y_slice, x_slice = np.mgrid[y_min:y_max, x_min:x_max]

    # Calculate distances from pupil center with distortion
    dx = x_slice - iris_x
    dy = y_slice - iris_y
    dist = np.sqrt((dx/h_stretch)**2 + (dy/v_stretch)**2)

    # Create mask for points within pupil radius
    pupil_mask = dist <= pupil_radius

    # Create mask for points within eye ellipse
    eye_dx = (x_slice - center_x) / 60
    eye_dy = (y_slice - center_y) / 30
    within_eye = (eye_dx**2 + eye_dy**2) <= 1

    # Combine masks for valid pupil points
    valid_pupil = pupil_mask & within_eye

    # Set color for all valid pupil points at once
    if np.any(valid_pupil):
        instate['eye_window'][y_slice[valid_pupil], x_slice[valid_pupil]] = [0, 0, 0, fade_factor*0.5]

        # Add highlights with distortion
        highlight_offset_x = -pupil_radius * 0.5
        highlight_offset_y = -pupil_radius * 0.5
        highlight_x = iris_x + highlight_offset_x
        highlight_y = iris_y + highlight_offset_y
        highlight_radius = 5
        
# Define the bounding box for the highlight
        y_min = max(0, int(highlight_y-highlight_radius))
        y_max = min(60, int(highlight_y+highlight_radius))
        x_min = max(0, int(highlight_x-highlight_radius))
        x_max = min(120, int(highlight_x+highlight_radius))

        # Create coordinate grids for the bounding box
        y_slice, x_slice = np.mgrid[y_min:y_max, x_min:x_max]

        # Calculate distances from highlight center with distortion
        dx = x_slice - highlight_x
        dy = y_slice - highlight_y
        dist = np.sqrt((dx/h_stretch)**2 + (dy/v_stretch)**2)

        # Create mask for points within highlight radius
        highlight_mask = dist <= highlight_radius

        # Create mask for points within eye ellipse
        eye_dx = (x_slice - center_x) / 60
        eye_dy = (y_slice - center_y) / 30
        within_eye = (eye_dx**2 + eye_dy**2) <= 1

        # Combine masks for valid highlight points
        valid_highlight = highlight_mask & within_eye

        if np.any(valid_highlight):
            # Calculate intensity with falloff
            intensity = 1 - (dist[valid_highlight]/highlight_radius)
            
            # Set colors for all valid points at once
            instate['eye_window'][y_slice[valid_highlight], x_slice[valid_highlight], 0] = 0
            instate['eye_window'][y_slice[valid_highlight], x_slice[valid_highlight], 1] = 0
            instate['eye_window'][y_slice[valid_highlight], x_slice[valid_highlight], 2] = 1.0 * intensity
            instate['eye_window'][y_slice[valid_highlight], x_slice[valid_highlight], 3] = fade_factor * intensity

    # Convert HSVA to BGRA for rendering
    rgb = color.hsv2rgb(instate['eye_window'][:,:,0:3])
    alpha = instate['eye_window'][:,:,3:4]
    rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)
    
    # Update the image plane
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['eye_plane'], 
        rgb_out[:,:,:]
    )


def psychedelic_spiral(instate, outstate):
    if instate['count'] == 0:
        # Initialize spiral parameters
        instate['spiral_window'] = np.zeros((60, 120, 4))  # HSVA format
        instate['spiral_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((60, 120, 4), dtype=np.uint8),
            position=(0, 0, 49.95),
            rotation=(0, 0, 0),
            scale=(9, 9)
        )
        instate['start_time'] = time.time()
        instate['phase'] = 0
        instate['color_shift'] = 0
        instate['last_update'] = time.time()

        # Create coordinate grids for spiral calculation
        x = np.linspace(-2, 2, 120)
        y = np.linspace(-1, 1, 60)
        instate['X'], instate['Y'] = np.meshgrid(x, y)
        instate['R'] = np.sqrt(instate['X']**2 + instate['Y']**2)
        instate['Theta'] = np.arctan2(instate['Y'], instate['X'])

        # Get total duration from outstate or use default
        #instate['duration'] = outstate.get('duration', 30.0)  # Changed from 'duration' to 'length'
        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['spiral_plane'])
        return

    # Get parameters from outstate or use defaults
    spiral_speed = outstate.get('spiral_speed', 1.0)
    color_speed = outstate.get('color_speed', 1.0)
    spiral_density = outstate.get('spiral_density', 5.0)
    distortion = outstate.get('spiral_distortion', 0.5)
    brightness = outstate.get('spiral_brightness', 0.8)

    # Calculate fade factor based on event time
    current_time = time.time()
    elapsed_time = current_time - instate['start_time']
    total_duration = instate['duration']
    
    # Define fade durations (in seconds)
    fade_in_duration = 5.0  # Fade in over 3 seconds
    fade_out_duration = 5.0  # Fade out over 3 seconds
    
    # Calculate fade factor - Fixed timing logic
    if elapsed_time < fade_in_duration:
        # Smooth fade in using sine interpolation
        fade_factor = np.sin((elapsed_time / fade_in_duration) * np.pi / 2)
    elif elapsed_time > (total_duration - fade_out_duration):
        # Smooth fade out using sine interpolation
        time_from_end = total_duration - elapsed_time
        fade_factor = np.sin((time_from_end / fade_out_duration) * np.pi / 2)
    else:
        fade_factor = 1.0

    # Update time and phases
    dt = current_time - instate['last_update']
    instate['last_update'] = current_time
    
    # Scale phase speeds by fade factor for smoother transitions
    instate['phase'] += dt * spiral_speed * (0.2 + 0.8 * fade_factor)
    instate['color_shift'] += dt * color_speed * (0.2 + 0.8 * fade_factor)

    # Calculate spiral pattern
    R = instate['R']
    Theta = instate['Theta']
    
    # Create base spiral pattern
    spiral = (Theta + spiral_density * R + instate['phase']) % (2 * np.pi)
    
    # Add distortion
    distorted_spiral = spiral + distortion * np.sin(5 * spiral + instate['phase'])
    
    # Create color variations
    hue_base = (distorted_spiral / (2 * np.pi) + instate['color_shift']) % 1.0
    
    # Add radial color variation
    radial_variation = 0.1 * np.sin(R * 4 + instate['phase'])
    hue = (hue_base + radial_variation) % 1.0

    # Create pulsing saturation and value
    saturation = 0.8 + 0.2 * np.sin(R * 3 - instate['phase'])
    value = brightness * (0.7 + 0.3 * np.cos(spiral * 2 + instate['phase']))

    # Apply fade factor to saturation and value
    saturation *= fade_factor
    value *= fade_factor

    # Combine into HSVA
    instate['spiral_window'][:,:,0] = hue
    instate['spiral_window'][:,:,1] = saturation
    instate['spiral_window'][:,:,2] = value
    instate['spiral_window'][:,:,3] = fade_factor  # Apply fade to alpha channel

    # Add some sparkle effects (scaled by fade factor)
    sparkle = np.random.random(R.shape) > 0.99
    instate['spiral_window'][sparkle,2] = fade_factor  # Scale sparkle brightness

    # Create pulsing center point (scaled by fade factor)
    center_x, center_y = 60, 30  # Center of display
    center_radius = 3
    y_coords, x_coords = np.ogrid[-center_y:60-center_y, -center_x:120-center_x]
    center_mask = x_coords**2 + y_coords**2 <= center_radius**2
    center_brightness = (0.8 + 0.2 * np.sin(instate['phase'] * 2)) * fade_factor
    instate['spiral_window'][center_mask,2] = center_brightness

    # Add subtle pulse during transitions
    if elapsed_time < fade_in_duration or elapsed_time > total_duration - fade_out_duration:
        transition_pulse = 0.2 * np.sin(elapsed_time * 8)
        instate['spiral_window'][:,:,2] *= (1 + transition_pulse)

    # Convert to RGB for display
    rgb = color.hsv2rgb(instate['spiral_window'][:,:,0:3])
    alpha = instate['spiral_window'][:,:,3:4]* fade_factor
    rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)
    
    # Update the image plane
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['spiral_plane'],
        rgb_out[:,:,:]
    )

def secondary_psychedelic_spiral(instate, outstate):
    if instate['count'] == 0:
        # Initialize secondary spiral parameters
        instate['secondary_spiral_window'] = np.zeros((32, 300, 4))  # HSVA format
        instate['secondary_spiral_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((32, 300, 4), dtype=np.uint8),
            position=(0, 0, 49.2),
            rotation=(0, 0, 0),
            scale=(9, 9)
        )

        # Initialize or share timing parameters
        if 'start_time' not in instate:
            instate['start_time'] = time.time()
        if 'duration' not in instate:
            instate['duration'] = outstate.get('duration', 30.0)
        if 'phase' not in instate:
            instate['phase'] = 0
        if 'color_shift' not in instate:
            instate['color_shift'] = 0
        if 'last_update' not in instate:
            instate['last_update'] = time.time()
        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['secondary_spiral_plane'])
        return

    # Get parameters from outstate
    spiral_speed = outstate.get('spiral_speed', 1.0)
    color_speed = outstate.get('color_speed', 1.0)
    spiral_density = outstate.get('spiral_density', 5.0)
    distortion = outstate.get('spiral_distortion', 0.5)
    brightness = outstate.get('spiral_brightness', 0.8)

    # Calculate fade factor based on event time
    current_time = time.time()
    elapsed_time = current_time - instate['start_time']
    total_duration = instate['duration']
    
    # Define fade durations (matching primary spiral)
    fade_in_duration = 10.0
    fade_out_duration = 10.0
    
    # Calculate fade factor
    if elapsed_time < fade_in_duration:
        fade_factor = np.sin((elapsed_time / fade_in_duration) * np.pi / 2)
    elif elapsed_time > total_duration - fade_out_duration:
        remaining_time = total_duration - elapsed_time
        fade_factor = np.sin((remaining_time / fade_out_duration) * np.pi / 2)
    else:
        fade_factor = 1.0
    
    # Update time and phases
    dt = current_time - instate['last_update']
    instate['last_update'] = current_time
    
    # Update shared phase variables (if not updated by primary spiral)
    instate['phase'] += dt * spiral_speed * (0.2 + 0.8 * fade_factor)
    instate['color_shift'] += dt * color_speed * (0.2 + 0.8 * fade_factor)
    
    # Create polar coordinates for secondary display
    theta = np.linspace(0, 2*np.pi, 32)[:, np.newaxis]
    r = np.linspace(0, 1, 300)[np.newaxis, :]
    
    # Calculate spiral pattern in polar coordinates
    spiral = (theta + spiral_density * r + instate['phase']) % (2 * np.pi)
    
    # Add distortion
    distorted_spiral = spiral + distortion * np.sin(5 * spiral + instate['phase'])
    
    # Create color variations
    hue_base = (distorted_spiral / (2 * np.pi) + instate['color_shift']) % 1.0
    
    # Add radial color variation
    radial_variation = 0.1 * np.sin(r * 4 + instate['phase'])
    hue = (hue_base + radial_variation) % 1.0

    # Create dynamic patterns scaled by fade factor
    radial_pulse = 0.2 * np.sin(r * 10 - instate['phase'])
    angular_pulse = 0.2 * np.sin(theta * 5 + instate['phase'])
    
    # Combine pulses for saturation and value
    saturation = (0.8 + radial_pulse) * fade_factor
    value = brightness * (0.7 + 0.3 * np.cos(distorted_spiral * 2 + instate['phase']) + angular_pulse) * fade_factor

    # Add transition pulse during fade in/out
    if elapsed_time < fade_in_duration or elapsed_time > total_duration - fade_out_duration:
        transition_pulse = 0.2 * np.sin(elapsed_time * 8)
        value *= (1 + transition_pulse)

    # Combine into HSVA
    instate['secondary_spiral_window'][:,:,0] = hue
    instate['secondary_spiral_window'][:,:,1] = saturation
    instate['secondary_spiral_window'][:,:,2] = value
    instate['secondary_spiral_window'][:,:,3] = fade_factor

    # Add sparkles scaled by fade factor
    sparkle = np.random.random((32, 300)) > 0.995
    sparkle_brightness = fade_factor * (0.8 + 0.2 * np.sin(current_time * 10))
    instate['secondary_spiral_window'][sparkle, 2] = sparkle_brightness

    # Add central glow (corrected dimensions)
    center_radius = 20
    # Create properly shaped radius grid
    theta_grid, r_grid = np.meshgrid(np.arange(32), np.arange(300), indexing='ij')
    center_mask = r_grid < center_radius
    center_pulse = 0.2 * np.sin(current_time * 4)
    
    # Calculate glow with proper dimensions
    glow_factor = np.exp(-r_grid[center_mask] / center_radius) * fade_factor
    glow_value = (0.8 + center_pulse) * glow_factor * brightness
    
    # Apply the glow
    instate['secondary_spiral_window'][center_mask, 2] = np.maximum(
        instate['secondary_spiral_window'][center_mask, 2],
        glow_value
    )

    # Add radial waves
    wave_pattern = 0.1 * np.sin(r_grid * 20 + instate['phase']) * fade_factor
    instate['secondary_spiral_window'][:,:,2] += wave_pattern

    # Ensure value and alpha stay in valid range
    instate['secondary_spiral_window'][:,:,2] = np.clip(
        instate['secondary_spiral_window'][:,:,2], 0, 1)
    instate['secondary_spiral_window'][:,:,3] = np.clip(
        instate['secondary_spiral_window'][:,:,3], 0, 1)

    # Convert to RGB
    rgb = color.hsv2rgb(instate['secondary_spiral_window'][:,:,0:3])
    alpha = instate['secondary_spiral_window'][:,:,3:4]
    rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)
    
    # Update the image plane
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['secondary_spiral_plane'],
        rgb_out[:,:,:]
    )

def display_text(instate, outstate, text="No Text Provided"):
    if instate['count'] == 0:
        # Initialize text display parameters
        instate['text_window'] = np.zeros((60, 120, 4))  # HSVA format
        instate['text_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((60, 120, 4), dtype=np.uint8),
            position=(0, 0, 15),
            rotation=(0, 0, 0),
            scale=(1.8, 1.8)
        )
        
        # Parse text from args
        text_chunks = [chunk.strip() for chunk in text.split(',')]
        instate['text_chunks'] = text_chunks
        instate['chunk_count'] = len(text_chunks)
        
        # Assign random base hue for each letter in each chunk (including spaces)
        instate['letter_hues'] = []
        for chunk in text_chunks:
            instate['letter_hues'].append([np.random.random() for _ in chunk])
        
        # Pre-calculate text dimensions
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        thickness = 1
        outline_thickness = 1
        top_bottom_padding = 0
        
        # Calculate font scale that fits the widest line including letter spacing
        max_width = 110  # Allow for some side padding
        test_scale = 2.5  # Start with larger scale
        final_scale = test_scale
        
        # Test each chunk's total width including letter spacing and spaces
        for chunk in text_chunks:
            total_width = 0
            for letter in chunk:
                if letter == ' ':
                    # Add space width
                    total_width += cv2.getTextSize(' ', font, test_scale, thickness)[0][0]
                else:
                    letter_width = cv2.getTextSize(letter, font, test_scale, thickness)[0][0]
                    total_width += letter_width
                total_width += test_scale * 1.5  # Add letter spacing
            
            if total_width > max_width:
                scale_factor = max_width / total_width
                final_scale = min(final_scale, test_scale * scale_factor)
        
        base_font_scale = final_scale
        
        # Calculate total height of all text chunks with the final scale
        total_text_height = 0
        chunk_heights = []
        
        for chunk in text_chunks:
            (_, text_height), _ = cv2.getTextSize(chunk, font, base_font_scale, thickness)
            chunk_heights.append(text_height)
            total_text_height += text_height
        
        # Add spacing between lines
        line_spacing = int(max(chunk_heights) * 0.25)
        total_text_height += line_spacing * (len(text_chunks) - 1)
        
        # Calculate starting Y position to center the entire text block with padding
        available_height = 60 - (2 * top_bottom_padding)
        start_y = top_bottom_padding + (available_height - total_text_height) // 2 + max(chunk_heights)
        
        # Calculate Y positions for each chunk
        y_positions = []
        current_y = start_y
        for height in chunk_heights:
            y_positions.append(current_y)
            current_y += height + line_spacing
        
        instate['y_positions'] = y_positions
        instate['font_scale'] = base_font_scale
        
        # Initialize animation parameters
        instate['start_time'] = time.time()
        instate['fade_states'] = np.zeros(len(text_chunks))
        instate['fade_in_delays'] = [i * 0.5 for i in range(len(text_chunks))]
        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['text_plane'])
        return

    current_time = time.time()
    elapsed_time = current_time - instate['start_time']
    total_duration = instate['duration']
    
    # Clear the window
    instate['text_window'].fill(0)
    
    # Calculate global fade based on total duration
    fade_duration = 2.0
    if elapsed_time < fade_duration:
        global_fade = elapsed_time / fade_duration
    elif elapsed_time > (total_duration - fade_duration):
        global_fade = (total_duration - elapsed_time) / fade_duration
    else:
        global_fade = 1.0
    global_fade = np.clip(global_fade, 0, 1)

    # Update fade states for each chunk
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    thickness = 1
    outline_thickness = 3
    letter_spacing = int(instate['font_scale'] * 1.5)
    
    for chunk_idx, (chunk, delay) in enumerate(zip(instate['text_chunks'], instate['fade_in_delays'])):
        chunk_elapsed = elapsed_time - delay
        if chunk_elapsed > 0:
            instate['fade_states'][chunk_idx] = min(1.0, chunk_elapsed / 1.0)
        
        if instate['fade_states'][chunk_idx] > 0:
            font_scale = instate['font_scale']
            chunk_fade = instate['fade_states'][chunk_idx] * global_fade
            
            # Calculate total width of chunk including spacing and spaces
            total_chunk_width = 0
            for letter in chunk:
                if letter == ' ':
                    total_chunk_width += cv2.getTextSize(' ', font, font_scale, thickness)[0][0] + letter_spacing
                else:
                    letter_width = cv2.getTextSize(letter, font, font_scale, thickness)[0][0]
                    total_chunk_width += letter_width + letter_spacing
            total_chunk_width -= letter_spacing  # Remove extra spacing after last letter
            
            # Center the chunk
            start_x = (120 - total_chunk_width) // 2
            y = int(instate['y_positions'][chunk_idx])
            
            # Process each letter separately
            current_x = start_x
            for letter_idx, letter in enumerate(chunk):
                # Handle spaces
                if letter == ' ':
                    space_width = cv2.getTextSize(' ', font, font_scale, thickness)[0][0]
                    current_x += space_width + letter_spacing
                    continue
                
                letter_width = cv2.getTextSize(letter, font, font_scale, thickness)[0][0]
                base_hue = instate['letter_hues'][chunk_idx][letter_idx]
                
                # Create temporary images for letter and its outline
                temp_img = np.zeros((60, 120), dtype=np.uint8)
                outline_img = np.zeros((60, 120), dtype=np.uint8)
                
                # Draw outline (black ring)
                cv2.putText(outline_img, letter, (current_x, y), font, font_scale, 255, outline_thickness, cv2.LINE_AA)
                
                # Draw main letter
                cv2.putText(temp_img, letter, (current_x, y), font, font_scale, 255, thickness, cv2.LINE_AA)
                
                # Calculate pulsing effect
                pulse = 0.8 + 0.2 * np.sin(current_time * 2 + letter_idx * np.pi/6)
                
                # Apply outline (pure black)
                outline_mask = outline_img > 0
                text_mask = temp_img > 0
                outline_only_mask = outline_mask & ~text_mask
                
                # Set outline to black
                instate['text_window'][outline_only_mask, 0] = 0
                instate['text_window'][outline_only_mask, 1] = 0
                instate['text_window'][outline_only_mask, 2] = 0
                instate['text_window'][outline_only_mask, 3] = chunk_fade
                
                # Apply letter with color and noise
                text_pixels = np.where(text_mask)
                if len(text_pixels[0]) > 0:
                    # Generate color noise for each pixel
                    hue_noise = np.random.normal(0, 0.02, len(text_pixels[0]))
                    sat_noise = np.random.normal(0, 0.05, len(text_pixels[0]))
                    val_noise = np.random.normal(0, 0.05, len(text_pixels[0]))
                    
                    # Apply base color with noise
                    instate['text_window'][text_pixels[0], text_pixels[1], 0] = (base_hue + hue_noise) % 1.0
                    instate['text_window'][text_pixels[0], text_pixels[1], 1] = np.clip(0.8 + sat_noise, 0, 1)
                    instate['text_window'][text_pixels[0], text_pixels[1], 2] = np.clip(pulse * chunk_fade + val_noise, 0, 1)
                    instate['text_window'][text_pixels[0], text_pixels[1], 3] = chunk_fade
                
                # Move to next letter position
                current_x += letter_width + letter_spacing
    
    # Convert HSVA to RGBA for rendering
    rgb = color.hsv2rgb(instate['text_window'][..., 0:3])
    alpha = instate['text_window'][..., 3:4]
    rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)
    
    # Update the image plane
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['text_plane'], 
        rgb_out[:,:,:]
    )


def colorful_conway(instate, outstate):
    if instate['count'] == 0:
        # Initialize Conway's Life parameters
        instate['conway_window'] = np.zeros((60, 120, 4))  # HSVA format
        instate['conway_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((60, 120, 4), dtype=np.uint8),
            position=(0, 0, 10),
            rotation=(0, 0, 0),
            scale=(1, 1)
        )
        
        # Add rule set definitions 
        instate['rule_sets'] = [
            # Standard Conway's rules (B3/S23)
            {'name': 'Conway', 'birth': {3}, 'survival': {2, 3}},
            
            # HighLife (B36/S23) - Similar to Conway but with interesting replicators
            {'name': 'HighLife', 'birth': {3, 6}, 'survival': {2, 3}},
            
            # Day & Night (B3678/S34678) - Produces symmetric patterns
            {'name': 'DayNight', 'birth': {3,6,7,8}, 'survival': {3,4,6,7,8}},
            
            # Seeds (B2/S) - Explosive growth patterns
            {'name': 'Seeds', 'birth': {2}, 'survival': set()},
            
            # Maze (B3/S12345) - Creates maze-like patterns
            {'name': 'Maze', 'birth': {3}, 'survival': {1,2,3,4,5}},
            
            # Coral (B3/S45678) - Grows coral-like structures
            {'name': 'Coral', 'birth': {3}, 'survival': {4,5,6,7,8}}
        ]
        
        # Initialize rule state
        instate['current_rule_idx'] = 0
        instate['rule_change_interval'] = outstate.get('rule_change_interval', 10.0)
        instate['last_rule_change'] = time.time()
        
        # Create initial random grid with 30% live cells
        instate['grid'] = np.random.choice([0, 1], size=(60, 120), p=[0.7, 0.3])
        
        # Initialize color grid (hue values for each cell)
        instate['color_grid'] = np.random.random((60, 120))
        
        # Initialize timing parameters
        instate['start_time'] = time.time()
        instate['last_update'] = time.time()
        instate['update_interval'] = outstate.get('conway_speed', 0.05)
        
        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['conway_plane'])
        return

    current_time = time.time()
    elapsed_time = current_time - instate['start_time']
    total_duration = instate['duration']
    
    # Calculate fade factor
    fade_duration = 3.0
    if elapsed_time < fade_duration:
        fade_factor = elapsed_time / fade_duration
    elif elapsed_time > (total_duration - fade_duration):
        fade_factor = (total_duration - elapsed_time) / fade_duration
    else:
        fade_factor = 1.0
    fade_factor = np.clip(fade_factor, 0, 1)

    # Check if it's time to change rules
    if current_time - instate['last_rule_change'] >= instate['rule_change_interval']:
        instate['current_rule_idx'] = (instate['current_rule_idx'] + 1) % len(instate['rule_sets'])
        instate['last_rule_change'] = current_time
        
        # Add visual feedback when rules change
        instate['color_grid'] = (instate['color_grid'] + 0.3) % 1.0

    # Get current ruleset
    current_rules = instate['rule_sets'][instate['current_rule_idx']]

    # Update the grid based on the update interval
    if current_time - instate['last_update'] >= instate['update_interval']:
        # Create padded grid for easier neighbor counting
        padded_grid = np.pad(instate['grid'], ((1, 1), (1, 1)), mode='wrap')
        new_grid = np.zeros_like(instate['grid'])
        new_color_grid = np.copy(instate['color_grid'])
        
        # Calculate neighbors using convolution
        neighbors = np.zeros_like(instate['grid'])
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                neighbors += padded_grid[i:i+60, j:j+120]

        # Apply current ruleset
        # Birth rules
        births = (instate['grid'] == 0) & np.isin(neighbors, list(current_rules['birth']))
        new_grid[births] = 1
        
        # Survival rules 
        survival = (instate['grid'] == 1) & np.isin(neighbors, list(current_rules['survival']))
        new_grid[survival] = 1

        # Update colors for new cells (births only)
        for i, j in zip(*np.where(births)):
            neighbor_colors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = (i + di) % 60, (j + dj) % 120
                    if instate['grid'][ni, nj]:
                        neighbor_colors.append(instate['color_grid'][ni, nj])
            
            if neighbor_colors:
                # Mix colors based on current ruleset
                if current_rules['name'] in ['DayNight', 'Coral']:
                    # More dramatic color shifts for these rules
                    new_color_grid[i, j] = (np.mean(neighbor_colors) + 0.2) % 1.0
                elif current_rules['name'] == 'Seeds':
                    # Quick color mutations for explosive growth
                    new_color_grid[i, j] = (np.mean(neighbor_colors) + np.random.normal(0, 0.1)) % 1.0
                else:
                    # Standard color inheritance
                    new_color_grid[i, j] = (np.mean(neighbor_colors) + np.random.normal(0, 0.05)) % 1.0
            else:
                new_color_grid[i, j] = np.random.random()

        # Population control based on ruleset
        live_cell_count = np.sum(new_grid)
        min_threshold = 0.001 if current_rules['name'] != 'Seeds' else 0.0005  # Lower threshold for Seeds
        max_threshold = 0.3 if current_rules['name'] != 'DayNight' else 0.5  # Higher threshold for Day & Night

        if live_cell_count < (60 * 120 * min_threshold):
            cells_to_add = int((60 * 120 * 0.10) - live_cell_count)
            dead_cells = new_grid == 0
            dead_cell_indices = np.where(dead_cells)
            
            if len(dead_cell_indices[0]) > 0:
                random_indices = np.random.choice(
                    len(dead_cell_indices[0]),
                    size=min(cells_to_add, len(dead_cell_indices[0])),
                    replace=False
                )
                
                for idx in random_indices:
                    i, j = dead_cell_indices[0][idx], dead_cell_indices[1][idx]
                    new_grid[i, j] = 1
                    new_color_grid[i, j] = np.random.random()
        
        # Clear excess cells if over maximum threshold
        elif live_cell_count > (60 * 120 * max_threshold):
            live_cells = np.where(new_grid == 1)
            cells_to_remove = int(live_cell_count - (60 * 120 * max_threshold))
            remove_indices = np.random.choice(len(live_cells[0]), size=cells_to_remove, replace=False)
            for idx in remove_indices:
                new_grid[live_cells[0][idx], live_cells[1][idx]] = 0

        instate['grid'] = new_grid
        instate['color_grid'] = new_color_grid
        instate['last_update'] = current_time

    # Create the visual output
    window = instate['conway_window']
    window.fill(0)
    
    # Add pulsing effect
    pulse = 0.9 + 0.1 * np.sin(current_time * 2)
    
    # Set colors for live cells
    live_cells = instate['grid'] > 0
    window[live_cells, 0] = instate['color_grid'][live_cells]  # Hue
    window[live_cells, 1] = 1.0  # Saturation
    window[live_cells, 2] = pulse  # Value
    window[live_cells, 3] = fade_factor  # Alpha

    # Convert HSVA to RGBA for rendering
    rgb = color.hsv2rgb(window[..., 0:3])
    alpha = window[..., 3:4]
    rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)
    
    # Update the image plane
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['conway_plane'],
        rgb_out[:,:,:]
    )

def secondary_alarm(instate, outstate):
    if instate['count'] == 0:
        # Initialize alarm parameters
        instate['alarm_window'] = np.zeros((32, 300, 4))  # HSVA format for polar display
        instate['alarm_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((32, 300, 4), dtype=np.uint8),
            position=(0, 0, 49.5),  # Place in front of most effects
            rotation=(0, 0, 0),
            scale=(9, 9)
        )
        instate['start_time'] = time.time()
        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['alarm_plane'])
        return

    # Get current time and calculate elapsed time
    current_time = time.time()
    elapsed_time = current_time - instate['start_time']
    total_duration = instate['duration']
    
    # Calculate event fade factor (5 second fade in/out)
    fade_duration = 5.0
    if elapsed_time < fade_duration:
        event_fade = elapsed_time / fade_duration
    elif elapsed_time > (total_duration - fade_duration):
        event_fade = (total_duration - elapsed_time) / fade_duration
    else:
        event_fade = 1.0
    event_fade = np.clip(event_fade, 0, 1)
    
    # Calculate sweep position (1 second period, moving down only)
    sweep_position = (elapsed_time % 1.0) * 32  # Moves through all 32 rows in 1 second
    sweep_center = int(sweep_position)  # Integer position for current center row
    
    # Calculate global blink (5 second period)
    blink_factor = np.cos(elapsed_time * 2 * np.pi / 5) * 0.5 + 0.5  # 5 second period
    
    # Combine event fade with blink factor
    combined_fade = event_fade * blink_factor
    
    # Clear the window
    window = instate['alarm_window']
    window.fill(0)
    
    # Draw sweep (5 rows wide)
    for i in range(-2, 3):  # 5 rows (-2, -1, 0, 1, 2)
        row = (sweep_center + i) % 32  # Wrap around using modulo
        # Fade intensity based on distance from center row
        intensity = 1.0 - abs(i) * 0.2  # Full intensity at center, fading to edges
        
        # Set the color for the entire row (red alarm)
        window[row, :] = [
            0.0,        # Hue (red)
            1.0,        # Saturation (full)
            intensity * combined_fade,  # Value (varies with sweep position, blink, and event fade)
            intensity * combined_fade   # Alpha (matches value for proper blending)
        ]
    
    # Convert HSVA to RGBA for rendering
    rgb = color.hsv2rgb(window[..., 0:3])
    alpha = window[..., 3:4]
    rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)
    
    # Update the image plane
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['alarm_plane'],
        rgb_out[:,:,:]
    )

def display_text_short(instate, outstate, text="No Text Provided", num_frames=2, gap_frames=1):
    if instate['count'] == 0:
        # Initialize text display parameters 
        instate['text_window'] = np.zeros((60, 120, 4))  # HSVA format
        instate['text_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((60, 120, 4), dtype=np.uint8), 
            position=(0, 0, 10),
            rotation=(0, 0, 0),
            scale=(1, 1)
        )
        
        # Store words and initialize counters
        instate['text_words'] = text.strip().split()
        instate['frame_count'] = 0
        instate['gap_count'] = 0
        instate['num_frames'] = num_frames
        instate['gap_frames'] = gap_frames
        instate['in_gap'] = False
        instate['is_end_gap'] = False
        
        # Assign random base hue for each letter in each word
        instate['letter_hues'] = []
        for word in instate['text_words']:
            instate['letter_hues'].append([np.random.random() for _ in word])
            
        # Initialize animation parameters
        instate['start_time'] = time.time()
        instate['current_word_idx'] = 0
        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['text_plane'])
        return
        
    # Clear the window
    instate['text_window'].fill(0)
    
    # Handle gap period
    if instate['in_gap']:
        instate['gap_count'] += 1
        # Use double gap length if at end of sequence
        current_gap_frames = instate['gap_frames'] * 5 if instate['is_end_gap'] else instate['gap_frames']
        
        if instate['gap_count'] >= current_gap_frames:
            instate['in_gap'] = False
            instate['gap_count'] = 0
            instate['frame_count'] = 0
            instate['is_end_gap'] = False
        
        # Update image plane with blank frame
        rgb_out = np.zeros((60, 120, 4), dtype=np.uint8)
        outstate['render'][instate['frame_id']].update_image_plane_texture(
            instate['text_plane'],
            rgb_out
        )
        return
        
    # ... existing code ...
    
    current_time = time.time()
    
    # Get current word
    word = instate['text_words'][instate['current_word_idx']]
    
    # Calculate optimal font scale to fit word in 2/3 of screen width
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    thickness = 1
    max_width = 80  # 2/3 of 120
    test_scale = 4.0  # Start large
    
    # Calculate total width including letter spacing
    letter_spacing = int(test_scale * 1.5)
    total_width = 0
    for letter in word:
        letter_width = cv2.getTextSize(letter, font, test_scale, thickness)[0][0] 
        total_width += letter_width + letter_spacing
    total_width -= letter_spacing  # Remove extra final spacing
    
    # Scale down if too wide
    if total_width > max_width:
        scale_factor = max_width / total_width
        font_scale = test_scale * scale_factor
    else:
        font_scale = test_scale
    
    # Calculate total word width with final scale
    total_width = 0
    for letter in word:
        letter_width = cv2.getTextSize(letter, font, font_scale, thickness)[0][0]
        total_width += letter_width + int(font_scale * 1.5)
    total_width -= int(font_scale * 1.5)  # Remove extra spacing
    
    # Center horizontally and vertically  
    start_x = (120 - total_width) // 2
    _, text_height = cv2.getTextSize(word, font, font_scale, thickness)[0]
    y = 30 + text_height//2  # Center vertically
    
    current_x = start_x
    letter_spacing = int(font_scale * 1.5)
    outline_thickness = 3
    
    # Draw each letter
    for letter_idx, letter in enumerate(word):
        # Get letter width
        letter_width = cv2.getTextSize(letter, font, font_scale, thickness)[0][0]
        base_hue = instate['letter_hues'][instate['current_word_idx']][letter_idx]
        
        # Create temporary images for letter and outline
        temp_img = np.zeros((60, 120), dtype=np.uint8)
        outline_img = np.zeros((60, 120), dtype=np.uint8)
        
        # Draw outline (black ring)
        cv2.putText(outline_img, letter, (current_x, y), font, font_scale, 255, outline_thickness, cv2.LINE_AA)
        
        # Draw main letter
        cv2.putText(temp_img, letter, (current_x, y), font, font_scale, 255, thickness, cv2.LINE_AA)
        
        # Calculate pulsing effect
        pulse = 0.8 + 0.2 * np.sin(current_time * 2 + letter_idx * np.pi/6)
        
        # Apply outline
        outline_mask = outline_img > 0
        text_mask = temp_img > 0
        outline_only_mask = outline_mask & ~text_mask
        
        # Set outline to black
        instate['text_window'][outline_only_mask, 0] = 0  
        instate['text_window'][outline_only_mask, 1] = 0
        instate['text_window'][outline_only_mask, 2] = 0
        instate['text_window'][outline_only_mask, 3] = 1.0  # Full opacity
        
        # Apply letter with color and noise
        text_pixels = np.where(text_mask)
        if len(text_pixels[0]) > 0:
            # Generate color noise
            hue_noise = np.random.normal(0, 0.02, len(text_pixels[0]))
            sat_noise = np.random.normal(0, 0.05, len(text_pixels[0]))  
            val_noise = np.random.normal(0, 0.05, len(text_pixels[0]))
            
            # Apply base color with noise
            instate['text_window'][text_pixels[0], text_pixels[1], 0] = (base_hue + hue_noise) % 1.0
            instate['text_window'][text_pixels[0], text_pixels[1], 1] = np.clip(0.8 + sat_noise, 0, 1)
            instate['text_window'][text_pixels[0], text_pixels[1], 2] = np.clip(pulse + val_noise, 0, 1)
            instate['text_window'][text_pixels[0], text_pixels[1], 3] = 1.0  # Full opacity
            
        # Move to next letter position
        current_x += letter_width + letter_spacing
        
    # Convert HSVA to RGBA
    rgb = color.hsv2rgb(instate['text_window'][..., 0:3])
    alpha = instate['text_window'][..., 3:4] 
    rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)
    
    # Update image plane
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['text_plane'],
        rgb_out[:,:,:]
    )
    
    # Increment frame counter
    instate['frame_count'] += 1
    
    # Check if current word is done
    if instate['frame_count'] >= instate['num_frames']:
        # Check if this is the last word
        next_word_idx = (instate['current_word_idx'] + 1) % len(instate['text_words'])
        if next_word_idx == 0:
            instate['is_end_gap'] = True
            
        instate['current_word_idx'] = next_word_idx
        instate['frame_count'] = 0
        instate['in_gap'] = True
        
        # Regenerate hues for next cycle if we're wrapping around
        if instate['current_word_idx'] == 0:
            instate['letter_hues'] = []
            for word in instate['text_words']:
                instate['letter_hues'].append([np.random.random() for _ in word])

def fluid_pond(instate, outstate):
    if instate['count'] == 0:
        # Initialize fluid simulation parameters
        grid_size = (60, 120)  # Match display dimensions
        instate['fluid_window'] = np.zeros((*grid_size, 4))  # HSVA format
        instate['fluid_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((*grid_size, 4), dtype=np.uint8),
            position=(0, 0, 10),
            rotation=(0, 0, 0),
            scale=(1, 1)
        )
        
        # Initialize simulation grids
        instate['height'] = np.zeros(grid_size)  # Current height
        instate['velocity'] = np.zeros(grid_size)  # Vertical velocity
        instate['last_update'] = time.time()
        instate['start_time'] = time.time()
        
        # Adjusted simulation parameters for more visible ripples
        instate['dampening'] = 0.98  # Less dampening for longer-lasting waves
        instate['tension'] = 0.3  # Higher tension for crisper waves
        instate['propagation'] = 3  # Slightly slower propagation for more defined waves
        
        # Enhanced drop parameters
        instate['last_drop'] = time.time()
        instate['drop_interval'] = np.random.uniform(0.3, 1.0)  # More frequent drops
        
        # Create pond boundary mask (elliptical shape)
        y, x = np.ogrid[:60, :120]
        center_y, center_x = 30, 60
        instate['boundary_mask'] = ((x - center_x)**2 / (60**2) + 
                                  (y - center_y)**2 / (30**2) <= 1)
        
        # Visual parameters
        instate['base_brightness'] = 0.7
        instate['height_limit'] = 0.3
        instate['wave_contrast'] = 0.3

        # Initialize multiple fish parameters
        instate['fish'] = []
        num_fish = np.random.randint(3, 5)
        
        for _ in range(num_fish):
            angle = np.random.uniform(0, 2*np.pi)
            radius = np.random.uniform(0.2, 0.85)
            start_x = 60 + np.cos(angle) * 52 * radius
            start_y = 30 + np.sin(angle) * 25 * radius
            
            size_variation = np.random.uniform(0.8, 1.2)
            
            fish = {
                'pos': np.array([start_x, start_y]),
                'vel': np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)]),
                'target': np.array([start_x, start_y]),
                'angle': np.random.uniform(0, 2*np.pi),
                'last_target_time': time.time() + np.random.uniform(0, 2.0),
                'target_change_interval': np.random.uniform(2.0, 4.0),
                'speed': np.random.uniform(5.0, 10.0),
                'size': 3.0 * size_variation,
                'trail': [],
                'max_trail_length': 10,
                'hue': np.random.uniform(0.08, 0.95),
                'personality': {
                    'turn_factor': np.random.uniform(4.0, 6.0),
                    'sociability': np.random.uniform(0.5, 1.5)
                }
            }
            instate['fish'].append(fish)

        # Initialize lily pads
        num_lily_pads = np.random.randint(3, 6)
        instate['lily_pads'] = []
        
        for _ in range(num_lily_pads):
            while True:
                angle = np.random.uniform(0, 2*np.pi)
                radius = np.random.uniform(0.2, 0.7)
                pad_x = 60 + np.cos(angle) * 45 * radius
                pad_y = 30 + np.sin(angle) * 20 * radius
                
                too_close = False
                for pad in instate['lily_pads']:
                    dist = np.sqrt((pad_x - pad['pos'][0])**2 + (pad_y - pad['pos'][1])**2)
                    if dist < 15:
                        too_close = True
                        break
                
                if not too_close:
                    lily_pad = {
                        'pos': np.array([pad_x, pad_y]),
                        'base_pos': np.array([pad_x, pad_y]),
                        'size': np.random.uniform(6, 8),
                        'angle': np.random.uniform(0, 2*np.pi),
                        'drift_offset': np.array([0., 0.]),
                        'drift_phase': np.random.uniform(0, 2*np.pi),
                    }
                    instate['lily_pads'].append(lily_pad)
                    break

        # Initialize frog
        initial_pad = np.random.choice(instate['lily_pads'])
        instate['frog'] = {
            'current_pad': initial_pad,
            'pos': initial_pad['pos'].copy(),
            'size': 3.0,
            'state': 'sitting',
            'jump_start_time': 0,
            'jump_duration': 1.0,
            'last_jump_time': time.time(),
            'jump_interval': np.random.uniform(5, 15),
            'target_pad': None,
            'start_pos': None
        }

        # Add fireflies
        instate['fireflies'] = []
        num_fireflies = np.random.randint(2, 4)
        for _ in range(num_fireflies):
            firefly = {
                'pos': np.array([
                    np.random.uniform(30, 90),
                    np.random.uniform(10, 50)
                ]),
                'vel': np.array([0., 0.]),
                'phase': np.random.uniform(0, 2*np.pi),
                'blink_rate': np.random.uniform(0.5, 2.0),
                'hover_radius': np.random.uniform(3, 8)
            }
            instate['fireflies'].append(firefly)

        # Add water plants
        instate['water_plants'] = []
        num_plants = np.random.randint(3, 6)
        for _ in range(num_plants):
            while True:
                angle = np.random.uniform(0, 2*np.pi)
                radius = np.random.uniform(0.3, 0.8)
                pos = np.array([
                    60 + np.cos(angle) * 45 * radius,
                    30 + np.sin(angle) * 20 * radius
                ])
                
                # Check distance from lily pads and other plants
                too_close = False
                for pad in instate['lily_pads']:
                    if np.linalg.norm(pos - pad['pos']) < 12:
                        too_close = True
                        break
                        
                for plant in instate['water_plants']:
                    if np.linalg.norm(pos - plant['pos']) < 8:
                        too_close = True
                        break
                
                if not too_close:
                    plant = {
                        'pos': pos,
                        'height': np.random.uniform(5, 8),
                        'sway_phase': np.random.uniform(0, 2*np.pi),
                        'segments': np.random.randint(3, 6),
                        'color_variation': np.random.uniform(-0.05, 0.05)
                    }
                    instate['water_plants'].append(plant)
                    break

        # Add dragonflies
        instate['dragonflies'] = []
        num_dragonflies = np.random.randint(1, 2)
        for _ in range(num_dragonflies):
            dragonfly = {
                'pos': np.array([
                    np.random.uniform(30, 90),
                    np.random.uniform(10, 50)
                ]),
                'vel': np.array([0., 0.]),
                'hover_point': np.array([
                    np.random.uniform(30, 90),
                    np.random.uniform(10, 50)
                ]),
                'hover_time': time.time(),
                'hover_duration': np.random.uniform(3, 8),
                'wing_phase': 0,
                'wing_speed': np.random.uniform(10, 15),
                'size': np.random.uniform(2, 3)
            }
            instate['dragonflies'].append(dragonfly)

        # Add wind effects
        instate['wind'] = {
            'direction': np.random.uniform(0, 2*np.pi),
            'strength': 0,
            'target_strength': 0,
            'change_time': time.time(),
            'pattern_offset': 0
        }

        # Add floating leaves
        instate['floating_leaves'] = []
        num_leaves = np.random.randint(8, 15)
        for _ in range(num_leaves):
            leaf = {
                'pos': np.array([
                    np.random.uniform(30, 90),
                    np.random.uniform(10, 50)
                ]),
                'angle': np.random.uniform(0, 2*np.pi),
                'size': np.random.uniform(1.5, 2.5),
                'drift_phase': np.random.uniform(0, 2*np.pi),
                'color_variation': np.random.uniform(-0.05, 0.05)
            }
            instate['floating_leaves'].append(leaf)

        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['fluid_plane'])
        return

    current_time = time.time()
    dt = min(current_time - instate['last_update'], 0.033)
    elapsed_time = current_time - instate['start_time']
    total_duration = instate['duration']
    
    # Calculate fade factor
    fade_duration = 3.0
    if elapsed_time < fade_duration:
        fade_factor = elapsed_time / fade_duration
    elif elapsed_time > (total_duration - fade_duration):
        fade_factor = (total_duration - elapsed_time) / fade_duration
    else:
        fade_factor = 1.0
    fade_factor = np.clip(fade_factor, 0, 1)

    # Generate random drops
    if current_time - instate['last_drop'] > instate['drop_interval']:
        valid_positions = np.where(instate['boundary_mask'])
        if len(valid_positions[0]) > 0:
            num_drops = np.random.randint(1, 2)
            for _ in range(num_drops):
                idx = np.random.randint(len(valid_positions[0]))
                drop_y, drop_x = valid_positions[0][idx], valid_positions[1][idx]
                
                drop_radius = np.random.uniform(2, 4)
                y, x = np.ogrid[:60, :120]
                drop_mask = ((x - drop_x)**2 + (y - drop_y)**2 <= drop_radius**2)
                drop_strength = np.random.uniform(-15, -8)
                
                distance = np.sqrt((x - drop_x)**2 + (y - drop_y)**2)
                wave_pattern = np.exp(-distance / drop_radius) * drop_strength
                instate['velocity'][drop_mask] = wave_pattern[drop_mask]
            
        instate['drop_interval'] = np.random.uniform(2, 7)
        instate['last_drop'] = current_time

    # Update wind effects
    if current_time - instate['wind']['change_time'] > 5.0:
        instate['wind']['target_strength'] = np.random.uniform(0, 1.0)
        instate['wind']['direction'] = np.random.uniform(0, 2*np.pi)
        instate['wind']['change_time'] = current_time
        
    # Smoothly adjust wind strength
    instate['wind']['strength'] += (instate['wind']['target_strength'] - 
                                  instate['wind']['strength']) * dt
    
    # Apply wind ripples
    instate['wind']['pattern_offset'] += dt * 2
    wind_x = np.cos(instate['wind']['direction'])
    wind_y = np.sin(instate['wind']['direction'])
    y, x = np.ogrid[:60, :120]
    wind_pattern = np.sin(x * wind_x * 0.1 + y * wind_y * 0.1 + 
                         instate['wind']['pattern_offset'])
    instate['velocity'] += wind_pattern * instate['wind']['strength'] * dt

    # Update lily pad positions
   
    for pad in instate['lily_pads']:
        pad['drift_phase'] += dt * 0.5
        pad['drift_offset'] = np.array([
            np.cos(pad['drift_phase']) * 1.5,
            np.sin(pad['drift_phase'] * 0.7) * 1.0
        ])
        
        local_height = instate['height'][
            int(pad['pos'][1]), 
            int(pad['pos'][0])
        ]
        fluid_influence = np.array([
            local_height * 2.0,
            local_height * 2.0
        ])
        
        pad['pos'] = pad['base_pos'] + pad['drift_offset'] + fluid_influence

    # Update frog behavior
    frog = instate['frog']
    if frog['state'] == 'sitting':
        frog['pos'] = frog['current_pad']['pos'].copy()
        
        if current_time - frog['last_jump_time'] > frog['jump_interval']:
            available_pads = [pad for pad in instate['lily_pads'] if pad is not frog['current_pad']]
            frog['target_pad'] = np.random.choice(available_pads)
            frog['start_pos'] = frog['pos'].copy()
            frog['state'] = 'jumping'
            frog['jump_start_time'] = current_time
            frog['jump_duration'] = np.random.uniform(0.8, 1.2)
            frog['last_jump_time'] = current_time
            frog['jump_interval'] = np.random.uniform(5, 15)
            
            jump_x, jump_y = int(frog['pos'][0]), int(frog['pos'][1])
            if 0 <= jump_y < 60 and 0 <= jump_x < 120:
                radius = 5
                jump_mask = ((x - jump_x)**2 + (y - jump_y)**2 <= radius**2)
                instate['velocity'][jump_mask] -= 10.0

    elif frog['state'] == 'jumping':
        jump_progress = (current_time - frog['jump_start_time']) / frog['jump_duration']
        
        if jump_progress >= 1.0:
            frog['state'] = 'sitting'
            frog['current_pad'] = frog['target_pad']
            frog['pos'] = frog['target_pad']['pos'].copy()
            
            land_x, land_y = int(frog['pos'][0]), int(frog['pos'][1])
            if 0 <= land_y < 60 and 0 <= land_x < 120:
                radius = 6
                land_mask = ((x - land_x)**2 + (y - land_y)**2 <= radius**2)
                instate['velocity'][land_mask] -= 25.0
        else:
            t = jump_progress
            height_factor = 4.0 * t * (1 - t)
            frog['pos'] = (frog['start_pos'] * (1 - t) + 
                          frog['target_pad']['pos'] * t + 
                          np.array([0, -15 * height_factor]))

    # Update all fish positions
    separation_radius = 10.0
    
    for fish in instate['fish']:
        if current_time - fish['last_target_time'] > fish['target_change_interval']:
            angle = np.random.uniform(0, 2*np.pi)
            radius = np.random.uniform(0.2, 0.85)
            target_x = 60 + np.cos(angle) * 57 * radius
            target_y = 30 + np.sin(angle) * 27 * radius
            fish['target'] = np.array([target_x, target_y])
            fish['target_change_interval'] = np.random.uniform(4.0, 10.0)
            fish['last_target_time'] = current_time

        separation = np.zeros(2)
        for other_fish in instate['fish']:
            if other_fish is not fish:
                diff = fish['pos'] - other_fish['pos']
                dist = np.linalg.norm(diff)
                if dist < separation_radius:
                    separation += diff * (1.0 - dist/separation_radius) * fish['personality']['sociability']

        to_target = fish['target'] - fish['pos']
        distance = np.linalg.norm(to_target)
        
        if distance > 0.1:
            direction = to_target / distance
            if np.any(separation):
                separation_dir = separation / np.linalg.norm(separation)
                direction = direction + separation_dir
                direction = direction / np.linalg.norm(direction)
            
            desired_velocity = direction * fish['speed']
            steering = desired_velocity - fish['vel']
            steering *= 3.0 * dt
            fish['vel'] += steering
            
            speed = np.linalg.norm(fish['vel'])
            if speed > fish['speed']:
                fish['vel'] *= fish['speed'] / speed

            fish['pos'] += fish['vel'] * dt
            
            target_angle = np.arctan2(fish['vel'][1], fish['vel'][0])
            angle_diff = (target_angle - fish['angle'] + np.pi) % (2*np.pi) - np.pi
            fish['angle'] += angle_diff * fish['personality']['turn_factor'] * dt

        fish['trail'].append(np.copy(fish['pos']))
        if len(fish['trail']) > fish['max_trail_length']:
            fish['trail'].pop(0)

    # Update fireflies
    for firefly in instate['fireflies']:
        firefly['phase'] += dt * firefly['blink_rate']
        hover_offset = np.array([
            np.cos(firefly['phase']) * firefly['hover_radius'],
            np.sin(firefly['phase'] * 0.7) * firefly['hover_radius'] * 0.5
        ])
        firefly['pos'] += hover_offset * dt
        
        # Keep fireflies within bounds
        firefly['pos'] = np.clip(firefly['pos'], [30, 10], [90, 50])

    # Update water plants
    for plant in instate['water_plants']:
        plant['sway_phase'] += dt
        # Plant movement is handled in the drawing phase

    # Update dragonflies
    for dragonfly in instate['dragonflies']:
        dragonfly['wing_phase'] += dt * dragonfly['wing_speed']
        
        if current_time - dragonfly['hover_time'] > dragonfly['hover_duration']:
            dragonfly['hover_point'] = np.array([
                np.random.uniform(30, 90),
                np.random.uniform(10, 50)
            ])
            dragonfly['hover_time'] = current_time
            dragonfly['hover_duration'] = np.random.uniform(3, 8)
        
        to_target = dragonfly['hover_point'] - dragonfly['pos']
        dist = np.linalg.norm(to_target)
        if dist > 0.1:
            dragonfly['vel'] += to_target * dt * 2
            dragonfly['vel'] *= 0.95
            
        dragonfly['pos'] += dragonfly['vel'] * dt

    # Update floating leaves
    for leaf in instate['floating_leaves']:
        leaf['drift_phase'] += dt * 0.5
        drift = np.array([
            np.cos(leaf['drift_phase']) * 0.5,
            np.sin(leaf['drift_phase'] * 0.7) * 0.3
        ])
        
        local_height = instate['height'][
            int(leaf['pos'][1]), 
            int(leaf['pos'][0])
        ]
        
        leaf['pos'] += (drift + 
                       np.array([wind_x, wind_y]) * instate['wind']['strength'] * 0.5 +
                       np.array([0, local_height])) * dt
        
        leaf['pos'] = np.clip(leaf['pos'], [30, 10], [90, 50])

    # Update fluid simulation
    laplacian = (np.roll(instate['height'], 1, axis=0) + 
                np.roll(instate['height'], -1, axis=0) + 
                np.roll(instate['height'], 1, axis=1) + 
                np.roll(instate['height'], -1, axis=1) - 
                4 * instate['height'])

    instate['velocity'] += instate['tension'] * laplacian * instate['propagation']
    instate['velocity'] *= instate['dampening']
    
    max_velocity = 1.2
    instate['velocity'] = np.clip(instate['velocity'], -max_velocity, max_velocity)
    
    instate['height'] += instate['velocity'] * dt
    instate['height'] = np.clip(instate['height'], -instate['height_limit'], instate['height_limit'])
    
    distance_from_edge = np.zeros_like(instate['height'])
    center_y, center_x = 30, 60
    distance_from_edge = np.sqrt(((x - center_x)/60)**2 + ((y - center_y)/30)**2)
    edge_falloff = np.clip(1 - (distance_from_edge - 0.9) * 5, 0, 1)
    
    instate['height'] *= edge_falloff
    instate['velocity'] *= edge_falloff

    # Render scene
    mask = instate['boundary_mask']
    height_normalized = (instate['height'] + instate['height_limit']) / (2 * instate['height_limit'])
    wave_intensity = (height_normalized - 0.5) * instate['wave_contrast']
    
    # Clear the window
    instate['fluid_window'].fill(0)
    
    # Draw water
    base_hue = 0.6
    instate['fluid_window'][:,:,0][mask] = base_hue + wave_intensity[mask] * 0.5
    instate['fluid_window'][:,:,1][mask] = 0.8 + wave_intensity[mask] * 0.1
    brightness = instate['base_brightness'] + wave_intensity[mask]*0.3
    instate['fluid_window'][:,:,2][mask] = np.clip(brightness, 0.2, 0.9)
    instate['fluid_window'][:,:,3][mask] = fade_factor

    # Draw underwater plants
    for plant in instate['water_plants']:
        base_x, base_y = plant['pos']
        for i in range(plant['segments']):
            segment_height = plant['height'] / plant['segments']
            sway_amount = (i + 1) / plant['segments'] * 2
            offset_x = np.sin(plant['sway_phase'] + i * 0.2) * sway_amount
            
            segment_mask = ((x - (base_x + offset_x))**2 + 
                          (y - (base_y - i * segment_height))**2 < 2**2)
            
            instate['fluid_window'][segment_mask,0] = 0.35 + plant['color_variation']
            instate['fluid_window'][segment_mask,1] = 0.9
            instate['fluid_window'][segment_mask,2] = 0.4
            instate['fluid_window'][segment_mask,3] = fade_factor

    # Draw fish
    for fish in instate['fish']:
        fish_x, fish_y = fish['pos']
        fish_distance = np.sqrt((x - fish_x)**2 + (y - fish_y)**2)
        fish_mask = fish_distance < fish['size']
        
        if np.any(fish_mask):
            instate['fluid_window'][fish_mask,0] = fish['hue']
            instate['fluid_window'][fish_mask,1] = 0.9
            instate['fluid_window'][fish_mask,2] = 0.9
            instate['fluid_window'][fish_mask,3] = fade_factor
            
            eye_offset = np.array([
                np.cos(fish['angle']) * fish['size'] * 0.5,
                np.sin(fish['angle']) * fish['size'] * 0.5
            ])
            eye_pos = fish['pos'] + eye_offset
            eye_distance = np.sqrt((x - eye_pos[0])**2 + (y - eye_pos[1])**2)
            eye_mask = eye_distance < fish['size'] * 0.2
            instate['fluid_window'][eye_mask,0] = 0.0
            instate['fluid_window'][eye_mask,1] = 0.0
            instate['fluid_window'][eye_mask,2] = 0.0
            instate['fluid_window'][eye_mask,3] = fade_factor

    # Draw floating leaves
    for leaf in instate['floating_leaves']:
        leaf_x, leaf_y = leaf['pos']
        leaf_mask = ((x - leaf_x)**2 + (y - leaf_y)**2 < leaf['size']**2)
        
        instate['fluid_window'][leaf_mask,0] = 0.1 + leaf['color_variation']
        instate['fluid_window'][leaf_mask,1] = 0.8
        instate['fluid_window'][leaf_mask,2] = 0.4
        instate['fluid_window'][leaf_mask,3] = fade_factor

    # Draw lily pads
    for pad in instate['lily_pads']:
        pad_x, pad_y = pad['pos']
        pad_angle = pad['angle']
        stretched_distance = np.sqrt(
            ((x - pad_x) * np.cos(pad_angle) + (y - pad_y) * np.sin(pad_angle))**2 +
            ((x - pad_x) * -np.sin(pad_angle) + (y - pad_y) * np.cos(pad_angle))**2 / 1.5**2
        )
        pad_mask = stretched_distance < pad['size']
        
        instate['fluid_window'][pad_mask,0] = 0.3
        instate['fluid_window'][pad_mask,1] = 0.7
        instate['fluid_window'][pad_mask,2] = 0.4
        instate['fluid_window'][pad_mask,3] = fade_factor

        edge_mask = (stretched_distance < pad['size']) & (stretched_distance > pad['size'] - 1.0)
        instate['fluid_window'][edge_mask,2] *= 0.7


    # Draw frog
    frog_x, frog_y = frog['pos']
    if 0 <= frog_y < 60 and 0 <= frog_x < 120:
        frog_distance = np.sqrt((x - frog_x)**2 + (y - frog_y)**2)
        frog_mask = frog_distance < frog['size']
        
        if np.any(frog_mask):
            instate['fluid_window'][frog_mask,0] = 0.45
            instate['fluid_window'][frog_mask,1] = 0.8
            instate['fluid_window'][frog_mask,2] = 0.7
            instate['fluid_window'][frog_mask,3] = fade_factor
            
            eye_offset = 1.0
            for eye_x_offset in [-eye_offset, eye_offset]:
                eye_pos = frog['pos'] + np.array([eye_x_offset, -0.5])
                eye_distance = np.sqrt((x - eye_pos[0])**2 + (y - eye_pos[1])**2)
                eye_mask = eye_distance < 0.8
                instate['fluid_window'][eye_mask,0] = 0.0
                instate['fluid_window'][eye_mask,1] = 0.0
                instate['fluid_window'][eye_mask,2] = 0.0
                instate['fluid_window'][eye_mask,3] = fade_factor

    # Draw dragonflies
    for dragonfly in instate['dragonflies']:
        fly_x, fly_y = dragonfly['pos']
        body_mask = ((x - fly_x)**2 + (y - fly_y)**2 < dragonfly['size']**2)
        
        instate['fluid_window'][body_mask,0] = 0.9
        instate['fluid_window'][body_mask,1] = 0.9
        instate['fluid_window'][body_mask,2] = 0.8
        instate['fluid_window'][body_mask,3] = fade_factor
        
        wing_spread = np.sin(dragonfly['wing_phase']) * dragonfly['size'] * 2
        for wing_offset in [-1, 1]:
            wing_mask = ((x - (fly_x + wing_offset * wing_spread))**2 + 
                        (y - fly_y)**2 < (dragonfly['size'] * 0.8)**2)
            instate['fluid_window'][wing_mask,0] = 0
            instate['fluid_window'][wing_mask,1] = 0
            instate['fluid_window'][wing_mask,2] = 0.9
            instate['fluid_window'][wing_mask,3] = 0.3 * fade_factor

    # Draw fireflies (last to ensure they're on top)
    for firefly in instate['fireflies']:
        glow_intensity = (np.sin(firefly['phase'] * 2) * 0.5 + 0.5) ** 2
        if glow_intensity > 0.1:
            fly_x, fly_y = firefly['pos']
            glow_distance = np.sqrt((x - fly_x)**2 + (y - fly_y)**2)
            glow_mask = glow_distance < 3
            glow_falloff = np.exp(-glow_distance[glow_mask] / 2)
            
            instate['fluid_window'][glow_mask,0] = 0.2
            instate['fluid_window'][glow_mask,1] = 0.8
            instate['fluid_window'][glow_mask,2] = np.maximum(
                instate['fluid_window'][glow_mask,2],
                glow_falloff * glow_intensity
            )
            instate['fluid_window'][glow_mask,3] = fade_factor

    # Convert HSVA to RGBA for rendering
    rgb = color.hsv2rgb(instate['fluid_window'][..., 0:3])
    alpha = instate['fluid_window'][..., 3:4]
    rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)
    
    # Update the image plane
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['fluid_plane'],
        rgb_out[:,:,:]
    )
    
    instate['last_update'] = current_time

def drifting_clouds(instate, outstate):
    if instate['count'] == 0:
        outstate['has_clouds'] = True
        cloud_depth = 40
        
        # Create image plane
        instate['cloud_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((60, 120, 4), dtype=np.uint8),
            position=(0, 0, cloud_depth),
            rotation=(0, 0, 0),
            scale=(7, 7)
        )
        
        # Enhanced cloud system
        instate['cloud_system'] = {
            'last_update': time.time(),
            'start_time': time.time(),
            'cloud_images': [],           # Base cloud images
            'cloud_positions': [],        # Current positions
            'cloud_speeds': [],           # Movement speeds
            'cloud_opacities': [],        # Base opacity values
            'current_opacities': [],      # Current transition opacity 
            'noise_offsets': [],          # For noise animation
            'cloud_turbulence': [],       # Internal turbulence parameters
            'cloud_sizes': [],            # Size variations
            'subpixel_offsets': [],       # For smoother motion
            'z_indices': []               # For proper layering
        }
        
        # Create clouds
        num_clouds = np.random.randint(6, 9)
        
        for i in range(num_clouds):
            # Create larger cloud images for smoother edges
            width = np.random.randint(60, 120)
            height = np.random.randint(25, 45)
            
            # Generate base cloud shape
            cloud_img = np.zeros((height, width, 4), dtype=np.uint8)
            
            # Create cloud field with multiple overlapping blobs for natural shape
            density = np.zeros((height, width))
            
            # Create coordinate grids
            y, x = np.mgrid[0:height, 0:width]
            
            # Add main cloud mass with multiple blobs
            num_blobs = np.random.randint(10, 20)  # More blobs for varied shape
            
            # Choose a more varied pattern type for this cloud
            pattern_type = np.random.randint(0, 4)
            
            if pattern_type == 0:  # Horizontal stretched pattern
                main_axis_ratio = np.random.uniform(1.5, 3.0)
                secondary_axis_ratio = np.random.uniform(0.6, 1.2)
                blob_distribution_x = np.random.beta(2, 5, num_blobs) * width
                blob_distribution_y = np.random.normal(height/2, height/4, num_blobs)
            elif pattern_type == 1:  # Vertical stretched pattern
                main_axis_ratio = np.random.uniform(0.6, 1.2)
                secondary_axis_ratio = np.random.uniform(1.5, 3.0)
                blob_distribution_x = np.random.normal(width/2, width/4, num_blobs)
                blob_distribution_y = np.random.beta(2, 5, num_blobs) * height
            elif pattern_type == 2:  # Clustered multi-center pattern
                main_axis_ratio = np.random.uniform(0.8, 1.5)
                secondary_axis_ratio = np.random.uniform(0.8, 1.5)
                # Create 2-3 cluster centers
                centers = [(np.random.uniform(0.2, 0.8) * width, 
                           np.random.uniform(0.2, 0.8) * height) 
                           for _ in range(np.random.randint(2, 4))]
                # Distribute blobs around these centers
                center_idx = np.random.randint(0, len(centers), num_blobs)
                blob_distribution_x = np.array([centers[i][0] + np.random.normal(0, width/5) for i in center_idx])
                blob_distribution_y = np.array([centers[i][1] + np.random.normal(0, height/5) for i in center_idx])
            else:  # Random scattered pattern
                main_axis_ratio = np.random.uniform(0.7, 1.8)
                secondary_axis_ratio = np.random.uniform(0.7, 1.8)
                blob_distribution_x = np.random.uniform(0.1, 0.9, num_blobs) * width
                blob_distribution_y = np.random.uniform(0.1, 0.9, num_blobs) * height
            
            # Ensure main dense area by adding a few central blobs
            num_central = np.random.randint(3, 6)
            for i in range(num_central):
                # Place near center with some variation
                cx = width * (0.4 + np.random.uniform(-0.2, 0.2))
                cy = height * (0.4 + np.random.uniform(-0.2, 0.2))
                
                # Larger size for main mass
                rx = np.random.uniform(0.2, 0.5) * width * main_axis_ratio
                ry = np.random.uniform(0.2, 0.5) * height * secondary_axis_ratio
                
                # Random rotation
                angle = np.random.uniform(0, np.pi)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                
                # Rotated coordinates
                dx = (x - cx) * cos_a - (y - cy) * sin_a
                dy = (x - cx) * sin_a + (y - cy) * cos_a
                
                # Elliptical distance
                dist = np.sqrt((dx/rx)**2 + (dy/ry)**2)
                
                # Add to density field with soft falloff
                falloff = np.random.uniform(1.5, 3.0)
                blob_density = np.exp(-dist**falloff)
                density += blob_density * np.random.uniform(0.7, 1.0)  # Stronger weight for central mass
            
            # Add varied peripheral blobs for texture and shape
            for i in range(num_blobs):
                # Use the pre-calculated positions
                cx = blob_distribution_x[i]
                cy = blob_distribution_y[i]
                
                # Size varies based on position - smaller at edges
                edge_factor = max(0.001, min(cx/width, (width-cx)/width, cy/height, (height-cy)/height))
                size_factor = 0.3 + 0.7 * (edge_factor ** 0.5)  # Non-linear scaling

                
                # More varied sizes
                rx = np.random.uniform(0.05, 0.3) * width * size_factor
                ry = np.random.uniform(0.05, 0.3) * height * size_factor
                
                # Random rotation for more natural shapes
                angle = np.random.uniform(0, np.pi)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                
                # Rotated coordinates
                dx = (x - cx) * cos_a - (y - cy) * sin_a
                dy = (x - cx) * sin_a + (y - cy) * cos_a
                
                # Elliptical distance with optional squaring for varied shapes
                if np.random.random() > 0.7:
                    dist = ((dx/rx)**2 + (dy/ry)**2)  # Squared for sharper edges on some blobs
                else:
                    dist = np.sqrt((dx/rx)**2 + (dy/ry)**2)
                
                # Add to density field with varied falloff
                falloff = np.random.uniform(1.2, 4.0)  # More variation in edge softness
                blob_density = np.exp(-dist**falloff)
                density += blob_density * np.random.uniform(0.3, 0.9)  # Varied blob strength
            
            # Normalize and clip the density field
            density = np.clip(density / (np.max(density) + 1e-10), 0, 1)
            alpha = np.clip(density, 0, 1) ** 4 * 255  # Much stronger power falloff
            alpha = alpha.astype(np.uint8)  # Convert to uint8 after
            
            # Add noise texture for detail
            noise = np.random.random((height, width))
            
            # Blur the noise for smoother texture
            from scipy.ndimage import gaussian_filter
            noise = gaussian_filter(noise, sigma=1.0)
            
            # Create wispy edges
            edge_mask = (density > 0.1) & (density < 0.5)
            if np.any(edge_mask):
                density[edge_mask] *= (0.6 + 0.8 * noise[edge_mask])
            
            # Apply blur for smoother edges
            density = gaussian_filter(density, sigma=0.8)
            
            # Create brightness with top-down lighting
            brightness = np.zeros_like(density)
            vertical_gradient = 1.0 - y / height * 0.3
            brightness = (220 + noise * 35) * vertical_gradient
            
            # Create alpha with much stronger falloff at edges
            alpha = (density**2 * 255).astype(np.uint8)  # Much stronger power falloff
            
            # Apply enhanced edge softening to guarantee zero alpha at edges
            edge_strength = 2.0  # Increased significantly for stronger edge falloff
            # Wider soft edge region to ensure gradual transition to zero
            soft_edge = (density > 0) & (density < 0.6)  
            if np.any(soft_edge):
                # Create non-linear mapping that approaches zero at the boundaries
                edge_factor = (density[soft_edge] / 0.6) ** 1.5  # Steeper curve toward zero
                edge_alpha = alpha[soft_edge].astype(float)
                
                # Apply stronger falloff near edges with noise variation
                edge_multiplier = edge_factor**edge_strength * (0.4 + 0.6 * noise[soft_edge])
                edge_alpha *= edge_multiplier
                
                # Force very low density areas to zero
                extreme_edge = density[soft_edge] < 0.2
                if np.any(extreme_edge):
                    # Additional exponential falloff at the very edge
                    extreme_factor = (density[soft_edge][extreme_edge] / 0.2) ** 3.0
                    edge_alpha[extreme_edge] *= extreme_factor
                
                alpha[soft_edge] = np.clip(edge_alpha, 0, 255).astype(np.uint8)
            
            # Apply stronger blur to alpha for ultra-smooth transition to zero
                alpha = gaussian_filter(alpha, sigma=1.5)
                
                # Force a zero-alpha border around the entire cloud
                border_width = 3
                if height > 2*border_width and width > 2*border_width:
                    alpha[:border_width, :] = 0
                    alpha[-border_width:, :] = 0
                    alpha[:, :border_width] = 0
                    alpha[:, -border_width:] = 0
                    
                # Create a zero-alpha border and gradient falloff in a vectorized way
                if height > 2*border_width and width > 2*border_width:
                    # Create zero border
                    alpha[:border_width, :] = 0
                    alpha[-border_width:, :] = 0
                    alpha[:, :border_width] = 0
                    alpha[:, -border_width:] = 0
                    
                    # Create gradient masks for each border region
                    gradient_width = border_width
                    
                    # Top gradient (if within bounds)
                    if border_width + gradient_width <= alpha.shape[0]:
                        rows = np.arange(border_width, border_width + gradient_width)
                        factors = (rows - border_width) / gradient_width
                        factors = factors.reshape(-1, 1)  # Column vector for broadcasting
                        alpha[border_width:border_width + gradient_width, :] = (
                            alpha[border_width:border_width + gradient_width, :] * factors
                        ).astype(np.uint8)
                    
                    # Bottom gradient
                    if border_width + gradient_width <= alpha.shape[0]:
                        rows = np.arange(border_width, border_width + gradient_width)
                        factors = (rows - border_width) / gradient_width
                        factors = factors.reshape(-1, 1)  # Column vector for broadcasting
                        alpha[-(border_width + gradient_width):-border_width, :] = (
                            alpha[-(border_width + gradient_width):-border_width, :] * factors[::-1]
                        ).astype(np.uint8)
                    
                    # Left gradient
                    if border_width + gradient_width <= alpha.shape[1]:
                        cols = np.arange(border_width, border_width + gradient_width)
                        factors = (cols - border_width) / gradient_width
                        factors = factors.reshape(1, -1)  # Row vector for broadcasting
                        alpha[:, border_width:border_width + gradient_width] = (
                            alpha[:, border_width:border_width + gradient_width] * factors
                        ).astype(np.uint8)
                    
                    # Right gradient
                    if border_width + gradient_width <= alpha.shape[1]:
                        cols = np.arange(border_width, border_width + gradient_width)
                        factors = (cols - border_width) / gradient_width
                        factors = factors.reshape(1, -1)  # Row vector for broadcasting
                        alpha[:, -(border_width + gradient_width):-border_width] = (
                            alpha[:, -(border_width + gradient_width):-border_width] * factors[::-1]
                        ).astype(np.uint8)


            
            # Create RGBA cloud image
            valid = alpha > 0
            cloud_img = np.zeros((height, width, 4), dtype=np.uint8)
            for c in range(3):
                cloud_img[:,:,c][valid] = np.clip(brightness[valid], 0, 255).astype(np.uint8)
            cloud_img[:,:,3] = alpha
            
            # Store the cloud
            instate['cloud_system']['cloud_images'].append(cloud_img)
            
            # Set initial position with z-ordering
            start_x = np.random.uniform(-120, 180)
            start_y = np.random.uniform(3, 35)
            instate['cloud_system']['cloud_positions'].append([start_x, start_y])
            
            # Set speed
            speed = np.random.uniform(8, 15)
            instate['cloud_system']['cloud_speeds'].append(speed)
            
            # Set opacity
            base_opacity = np.random.uniform(0.6, 1.0)
            current_opacity = 0.0 if start_x > -60 and start_x < 120 else base_opacity
            instate['cloud_system']['cloud_opacities'].append(base_opacity)
            instate['cloud_system']['current_opacities'].append(current_opacity)
            
            # Set noise offset for animation
            instate['cloud_system']['noise_offsets'].append(np.random.uniform(0, 10, 2))
            
            # Set turbulence parameters for internal dynamics
            instate['cloud_system']['cloud_turbulence'].append({
                'phase': np.random.uniform(0, 2*np.pi, 3),  # For x, y, rotation
                'speed': np.random.uniform(0.1, 0.3, 3),    # How fast it changes
                'amount': [                                 # How much it moves
                    np.random.uniform(0.5, 1.5),            # X movement
                    np.random.uniform(0.3, 0.8),            # Y movement
                    np.random.uniform(0.1, 0.3)           # Rotation
                ]
            })
            
            # Size variation
            instate['cloud_system']['cloud_sizes'].append(np.random.uniform(0.8, 1.3))
            
            # Subpixel offsets for smooth motion
            instate['cloud_system']['subpixel_offsets'].append([0.0, 0.0])
            
            # Z-index for proper layering (clouds are drawn back-to-front)
            # Higher y position = closer to front
            z_index = start_y + np.random.uniform(-5, 5)  # Add slight randomness
            instate['cloud_system']['z_indices'].append(z_index)
        
        # Pre-allocate buffers
        instate['rgba_buffer'] = np.zeros((60, 120, 4), dtype=np.uint8)
        instate['noise_time'] = 0.0
        
        return

    if instate['count'] == -1:
        outstate['has_clouds'] = False
        outstate['render'][instate['frame_id']].remove_image_plane(instate['cloud_plane'])
        return

    # Get references and timing
    cloud_system = instate['cloud_system']
    current_time = time.time()
    dt = current_time - cloud_system['last_update']
    elapsed_time = current_time - cloud_system['start_time']
    total_duration = instate.get('duration', 60.0)
    outwind=outstate.get('wind',0)*50
    fog=outstate.get('fog_level',0)
    # Only update at reasonable FPS
    if dt < 0.042:
        return
        
    # Clear output buffer
    output_buffer = instate['rgba_buffer']
    output_buffer.fill(0)
    
    # Calculate fade factor for start/end
    if elapsed_time < 5.0:
        fade_factor = elapsed_time / 5.0
    elif elapsed_time > (total_duration - 5.0):
        fade_factor = (total_duration - elapsed_time) / 5.0
    else:
        fade_factor = 1.0
    fade_factor = max(0, min(1, fade_factor))
    
    # Update noise time for animation
    instate['noise_time'] += dt * 0.2
    
    # Global wave for coordinated motion
    global_wave_y = np.sin(elapsed_time * 0.05) * 0.3 + np.sin(elapsed_time * 0.13) * 0.1
    
    # Get cloud indices sorted by z-index for proper back-to-front rendering
    cloud_indices = list(range(len(cloud_system['cloud_images'])))
    sorted_indices = sorted(cloud_indices, key=lambda i: cloud_system['z_indices'][i])
    
    # Update and render each cloud in back-to-front order
    for idx in sorted_indices:
        cloud_img = cloud_system['cloud_images'][idx]
        position = cloud_system['cloud_positions'][idx]
        speed = cloud_system['cloud_speeds'][idx]
        base_opacity = cloud_system['cloud_opacities'][idx]
        current_opacity = cloud_system['current_opacities'][idx]
        noise_offset = cloud_system['noise_offsets'][idx]
        turbulence = cloud_system['cloud_turbulence'][idx]
        size = cloud_system['cloud_sizes'][idx]
        subpixel = cloud_system['subpixel_offsets'][idx]
        
        # Update opacity with smooth transition
        opacity_diff = base_opacity - current_opacity
        if abs(opacity_diff) > 0.01:
            # Faster fade in, slower fade out
            transition_speed = 0.8 if opacity_diff > 0 else 0.4
            cloud_system['current_opacities'][idx] += opacity_diff * transition_speed * dt
            current_opacity = cloud_system['current_opacities'][idx]
        
        # Update turbulence phases
        for j in range(3):
            turbulence['phase'][j] += turbulence['speed'][j] * dt
        
        # Calculate turbulence offsets
        #turbulence_x = np.sin(turbulence['phase'][0]) * turbulence['amount'][0]
        #turbulence_y = np.sin(turbulence['phase'][1]) * turbulence['amount'][1]
        
        # Update subpixel offsets for smoother motion
        subpixel[0] = (subpixel[0] + speed * dt) % 1.0
        subpixel[1] = (subpixel[1] + dt * 0.1) % 1.0
        
        # Apply movement with turbulence
        position[0] += (speed+outwind) * dt #+ turbulence_x * dt
        position[1] += global_wave_y * dt #+ turbulence_y * dt
        
        # Keep clouds from drifting too far vertically
        if position[1] < 0:
            position[1] = 0
        elif position[1] > 50:
            position[1] = 50
        
        # Update z-index based on vertical position
        cloud_system['z_indices'][idx] = position[1] + np.sin(turbulence['phase'][2]) * 3
        
        # Handle cloud recycling
        cloud_height, cloud_width = cloud_img.shape[:2]
        
        # If cloud has moved offscreen, recycle it
        if position[0] > 120 + cloud_width:
            # Reset cloud to enter from left
            position[0] = -cloud_width - np.random.uniform(0, 60)
            position[1] = np.random.uniform(3, 50)
            cloud_system['cloud_sizes'][idx] = np.random.uniform(0.8, 1.3)
            cloud_system['current_opacities'][idx] = 0.0  # Fade in again
            continue
        
        # Apply cloud size scaling with subpixel precision
        scaled_width = int(cloud_width * size)
        scaled_height = int(cloud_height * size)
        
        # Calculate screen coordinates with subpixel offset
        x_start = int(position[0] - subpixel[0])
        y_start = int(position[1] - subpixel[1])
        
        # Skip if entirely offscreen
        if x_start + scaled_width < 0 or x_start >= 120 or y_start + scaled_height < 0 or y_start >= 60:
            continue
        
        # Scale cloud if needed
        if scaled_width != cloud_width or scaled_height != cloud_height:
            try:
                import cv2
                scaled_cloud = cv2.resize(cloud_img, (scaled_width, scaled_height), 
                                          interpolation=cv2.INTER_LINEAR)
            except (ImportError, cv2.error):
                # Simple scaling if cv2 not available
                x_scale = scaled_width / cloud_width
                y_scale = scaled_height / cloud_height
                
                # Nearest neighbor but with fractional coordinates
                scaled_cloud = np.zeros((scaled_height, scaled_width, 4), dtype=np.uint8)
                for y in range(scaled_height):
                    src_y = min(int(y / y_scale), cloud_height-1)
                    for x in range(scaled_width):
                        src_x = min(int(x / x_scale), cloud_width-1)
                        scaled_cloud[y, x] = cloud_img[src_y, src_x]
        else:
            scaled_cloud = cloud_img
        
        # Calculate visible region of cloud
        cloud_x_start = max(0, -x_start)
        cloud_y_start = max(0, -y_start)
        cloud_x_end = min(scaled_width, 120 - x_start)
        cloud_y_end = min(scaled_height, 60 - y_start)
        
        # Calculate screen region
        screen_x_start = max(0, x_start)
        screen_y_start = max(0, y_start)
        screen_x_end = min(120, x_start + scaled_width)
        screen_y_end = min(60, y_start + scaled_height)
        
        # Skip if invalid region
        if cloud_x_end <= cloud_x_start or cloud_y_end <= cloud_y_start:
            continue
        
        # Extract the visible portion of the cloud
        cloud_region = scaled_cloud[cloud_y_start:cloud_y_end, cloud_x_start:cloud_x_end].copy()
        
        # Add dynamic noise to alpha channel
        h, w = cloud_region.shape[:2]
        if h > 0 and w > 0:
            # Generate noise pattern unique to this frame
            noise_y, noise_x = np.mgrid[0:h, 0:w].astype(np.float32)
            noise_y = (noise_y / h + noise_offset[1] + instate['noise_time'] * 0.1)
            noise_x = (noise_x / w + noise_offset[0] + instate['noise_time'] * 0.07)
            
            # Compute noise values (simplified perlin-like noise)
            noise_values = (np.sin(noise_x * 5) * np.cos(noise_y * 5) * 0.25 + 
                           np.sin(noise_x * 10 + 2) * np.cos(noise_y * 8 + 1) * 0.125 + 0.5)
            
            # Apply noise to alpha with edge sensitivity
            # Apply stronger noise influence at edges
            alpha_mask = cloud_region[:, :, 3] > 0
            if np.any(alpha_mask):
                alpha_values = cloud_region[:, :, 3][alpha_mask].astype(np.float32) / 255.0
                
                # Even stronger edge effect
                edge_factor = alpha_values**0.4  # Even smaller exponent for more influence at edges
                
                # Noise has greater impact at low alpha values
                noise_impact = (noise_values[alpha_mask] * 0.7 + 0.6) * (1.0 - edge_factor * 0.95)
                
                # Apply noise to fade edges more dramatically
                new_alpha = alpha_values * noise_impact * current_opacity * fade_factor * 255
                cloud_region[:, :, 3][alpha_mask] = np.clip(new_alpha, 0, 255).astype(np.uint8)


        
        # Get screen region
        screen_region = output_buffer[screen_y_start:screen_y_end, screen_x_start:screen_x_end]
        
        # Proper alpha blending (pre-multiplied alpha)
        # Proper alpha blending (pre-multiplied alpha)
        alpha_mask = cloud_region[:, :, 3] > 0
        if np.any(alpha_mask):
            # Convert to premultiplied alpha for better blending
            cloud_alpha = cloud_region[:, :, 3][alpha_mask].astype(np.float32) / 255.0
            
            # Calculate effective alpha (cloud over background)
            dest_alpha = screen_region[:, :, 3][alpha_mask].astype(np.float32) / 255.0
            out_alpha = cloud_alpha + dest_alpha * (1 - cloud_alpha)*2
            
            # Ensure we don't divide by zero
            blend_mask = out_alpha > 0.001
            if np.any(blend_mask):
                # Only blend visible pixels
                masked_indices = np.where(alpha_mask)
                blend_indices = tuple(arr[blend_mask] for arr in masked_indices)
                
                # Vectorized version - process all color channels at once
                src_colors = cloud_region[:, :, :3][blend_indices].astype(np.float32) / 255.0
                dst_colors = screen_region[:, :, :3][blend_indices].astype(np.float32) / 255.0
                
                # Expand dimensions to allow broadcasting
                cloud_alpha_expanded = cloud_alpha[blend_mask][:, np.newaxis]
                out_alpha_expanded = out_alpha[blend_mask][:, np.newaxis]
                
                # Perform blending with premultiplied alpha for all channels
                src_premult = src_colors * cloud_alpha_expanded
                dst_premult = dst_colors * dest_alpha[blend_mask][:, np.newaxis]
                
                # Final color
                out_premult = src_premult + dst_premult * (1 - cloud_alpha_expanded)
                out_colors = np.divide(out_premult, out_alpha_expanded, 
                                     out=np.zeros_like(out_premult), 
                                     where=out_alpha_expanded>0.001)
                
                # Store the result for all channels at once
                screen_region[:, :, :3][blend_indices] = np.clip(out_colors * 255/(1+fog*3.0), 0, 255).astype(np.uint8)

                
                # Store final alpha
                screen_region[:, :, 3][blend_indices] = np.clip(out_alpha[blend_mask] * 255, 0, 255).astype(np.uint8)
    
    # Update the image plane
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['cloud_plane'],
        output_buffer
    )
    
    # Update timestamp
    cloud_system['last_update'] = current_time