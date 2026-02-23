import time
import numpy as np
from skimage import color
from imgutils import lightning as lt
from pathlib import Path
import math 
import random
import cv2
from numba import jit
ParentPath = Path(__file__).parent.parent
media_path = ParentPath / 'media'
sound_path = media_path / 'sounds'


def wind(elapsed_time, state, maxspeed):
    wind = state.get('wind', 0)
    wind = np.cos(time.time()/15)*maxspeed
    state['wind_pos'] = wind
    
def smooth_wind(t, period=15, sharpness=4):
    """
    Generate a smooth wind oscillation between -1 and 1 with small derivative at zero.
    
    Parameters:
    - t: time value
    - period: time to complete one full oscillation
    - sharpness: controls how quickly the function transitions through zero
                (higher values = slower transition at zero)
    
    Returns:
    - Wind value between -1 and 1
    """
    # Using a transformed sine function that flattens near zero
    normalized_t = 2 * np.pi * t / period
    raw_value = 0.5*np.sin(normalized_t)+0.5
    wind_switch=np.sign(np.sin(normalized_t*0.25))
    # Apply transformation to flatten the curve near zero
    return wind_switch * ((raw_value) ** sharpness)


def lightning(instate, outstate,distance=10):
    
    if instate['count']==0:
        img = lt.generate_lightning(
        width=300, 
        height=300, 
        color=(np.random.random(3)/2+0.5), 
        branch_probability=0.95, 
        max_branches=15, 
        jitter=0.3
        )

        #make a string with thunder sound location
        spath=np.random.choice(['Thunder Clap Loud.wav','loud-thunder-192165.mp3','thunder-307513.mp3','peals-of-thunder-191992.mp3'])
        path=sound_path / spath

        outstate['soundengine'].schedule_event(path, time.time() + distance/10, 10)

        distance=np.random.random()*40+7
        instate['img']= outstate['render'][instate['frame_id']].create_image_plane(img,position=((np.random.random()-0.5)*50.0, 0, distance),rotation=(0, 0, 0),scale=(1, 1))

        #outstate['render'].update_image_plane_texture(instate['img'],np.random.randint(0,255,(400,400,4)).astype(np.uint8))
        instate['window']=img

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['img'])
        return
    #update the image plane position
    outstate['render'][instate['frame_id']].update_image_plane_texture(instate['img'], instate['window'])
    instate['window']=(instate['window']*0.98).astype(np.uint8)
    #cv2.imshow('Rain',instate['img'].image)
    #cv2.waitKey(1)



def Aurora(instate, outstate):
    if instate['count'] == 0:
        # Initialize aurora parameters
        instate['aurora_window'] = np.zeros((60, 120, 4))  # HSVA format
        instate['aurora_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((60, 120, 4), dtype=np.uint8),
            position=(0, 0, 49.9),  # Place aurora in background
            rotation=(0, 0, 0),
            scale=(9, 9)
        )
        # Wave parameters
        instate['base_hue'] = 0.3+np.random.random()*0.4  # Default to green
        instate['wave_points'] = np.linspace(0, 120, 20)  # X coordinates for wave control points
        instate['wave_heights'] = np.random.uniform(20, 35, len(instate['wave_points']))  # Y coordinates
        instate['time_offset'] = time.time()
        instate['start_time'] = time.time()
        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['aurora_plane'])
        return

    # Calculate fade factor based on duration
    current_time = time.time()
    elapsed_time = current_time - instate['start_time']
    total_duration = instate.get('duration', 30)
    fade_in_duration = total_duration * 0.2
    fade_out_start = total_duration * 0.8
    whompin=outstate.get('whomp',0.0)
    instate['whomp'] = whompin*0.3+0.7*instate.get('whomp',0.0)
    whomp=instate['whomp']
    # Calculate fade factor
    if elapsed_time < fade_in_duration:
        fade_factor = elapsed_time / fade_in_duration
    elif elapsed_time > fade_out_start:
        fade_factor = (total_duration - elapsed_time) / (total_duration * 0.2)
    else:
        fade_factor = 1.0
    fade_factor = np.clip(fade_factor, 0, 1)

    # Get aurora parameters from outstate or use defaults
    intensity = outstate.get('aurora_intensity', 1.0) * fade_factor
    base_hue = instate.get('base_hue', 0.3)  # Default to green
    
    # Update wave heights with smooth movement
    wave_speed = 0.5
    
    time_factor = (current_time - instate['time_offset']) * wave_speed
    
    # Create smooth wave movement using sine functions with different frequencies
    instate['wave_heights'] += np.sin(time_factor + np.arange(len(instate['wave_heights'])) * 0.2) * 0.1
    instate['wave_heights'] = np.clip(instate['wave_heights'], 15, 40)
    
    # Interpolate wave points for smooth curve
    x_interp = np.arange(120)
    wave_curve = np.interp(x_interp, instate['wave_points'], instate['wave_heights'])
    
    # Clear the window
    instate['aurora_window'].fill(0)
    
    # Create coordinate matrices for vectorized operations
    x_coords, y_coords = np.meshgrid(np.arange(120), np.arange(60))
    
    # Calculate the base curtain shape
    heights = wave_curve[x_coords]
    vertical_falloff = np.clip((heights - y_coords) / 20, 0, 1)
    
    # Add vertical streaks
    streaks = np.sin(x_coords / 2 + time_factor) * 0.1
    vertical_falloff *= (1 + streaks)
    
    # Add some noise to make it more organic
    noise = np.random.random((60, 120)) * 0.1
    vertical_falloff = np.clip(vertical_falloff + noise, 0, 1)
    
    # Create color variations
    hue_variation = np.sin(x_coords / 30 + time_factor) * 0.05
    saturation = np.ones_like(vertical_falloff) * 0.8
    
    # Assign colors to the aurora window
    
    instate['aurora_window'][..., 0] = base_hue + hue_variation  # Hue
    instate['aurora_window'][..., 1] = saturation  # Saturation
    instate['aurora_window'][..., 2] = vertical_falloff * intensity*0.7  # Value
    instate['aurora_window'][..., 3] = vertical_falloff * 0.4 * intensity*np.clip((1+whomp),0,2)  # Alpha
    
    # Convert HSVA to BGRA for rendering
    rgb = color.hsv2rgb(instate['aurora_window'][..., 0:3])
    alpha = instate['aurora_window'][..., 3:4]
    rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)
    
    # Update the image plane
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['aurora_plane'], 
        rgb_out[:,: :]
    )

def Awooo_Wolf_Howl(instate, outstate):
    
    if instate['count']==0:
        switch=np.random.randint(0,4)
        #make a string with thunder sound location
        if switch==0:
            path=sound_path / '125 Wolf Howl (4).mp3'
            outstate['soundengine'].schedule_event(path, time.time(), 6)
        if switch==1:
            path=sound_path / 'wolf-howling-140235.mp3'
            outstate['soundengine'].schedule_event(path, time.time(), 6)
        if switch==2:
            path=sound_path / 'duskwolf-101348.mp3'
            outstate['soundengine'].schedule_event(path, time.time(), 6)
        if switch==3:
            path=sound_path / 'howling-wolves-6965.mp3'
            outstate['soundengine'].schedule_event(path, time.time(), 7)

def meteor_shower(instate, outstate):
    if instate['count'] == 0:
        # Initialize meteor shower parameters - identical to original
        instate['meteor_window'] = np.zeros((60, 120, 4))  # HSVA format
        instate['meteor_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((60, 120, 4), dtype=np.uint8),
            position=(0, 0, 49.9),  # Just in front of aurora layer
            rotation=(0, 0, 0),
            scale=(9, 9)
        )
        instate['meteors'] = []  # Keep the original dictionary-based meteors
        instate['start_time'] = time.time()
        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['meteor_plane'])
        return

    # Clear the window - same as original
    instate['meteor_window'].fill(0)
    
    current_time = time.time()
    elapsed_time = current_time - instate['start_time']
    total_duration = instate.get('duration', 30.0)
    fade_in_duration = total_duration * 0.2
    fade_out_start = total_duration * 0.8
    
    # Calculate fade factor
    if elapsed_time < fade_in_duration:
        fade_factor = elapsed_time / fade_in_duration
    elif elapsed_time > fade_out_start:
        fade_factor = (total_duration - elapsed_time) / (total_duration * 0.2)
    else:
        fade_factor = 1.0
    fade_factor = np.clip(fade_factor, 0, 1)

    # Generate new meteors based on meteor_rate - identical to original
    if random.random() < outstate.get('meteor_rate', 0)/2:
        # Create new meteor - identical to original
        meteor = {
            'x': random.uniform(0, 120),  # Start within screen width
            'y': 0,  # Start at top of screen
            'angle': math.radians(random.uniform(50, 130)),  # Angle for downward diagonal movement
            'speed': random.uniform(1.5, 3.0),
            'size': random.uniform(0.5, 2.5),
            'trail_length': 30 + random.random() * 30,
            'life': 1.0
        }
        instate['meteors'].append(meteor)

        # Add whoosh sound occasionally - identical to original
        if random.random() < 0.1:
            whoosh_path = sound_path / 'Whoosh By 04.wav'
            outstate['soundengine'].schedule_event(
                whoosh_path, 
                time.time(), 
                2
            )

    # Update and draw meteors
    new_meteors = []
    for meteor in instate['meteors']:
        # Update meteor position - identical to original
        meteor['x'] += math.cos(meteor['angle']) * meteor['speed']
        meteor['y'] += math.sin(meteor['angle']) * meteor['speed']
        meteor['life'] -= 0.02

        # Keep meteor if still alive and on screen - identical to original
        if meteor['life'] > 0 and meteor['y'] < 70 and meteor['x'] > -10 and meteor['x'] < 130:
            new_meteors.append(meteor)
            
            # Pre-compute trail values for better performance
            angle_cos = math.cos(meteor['angle'])
            angle_sin = math.sin(meteor['angle'])
            base_x = meteor['x']
            base_y = meteor['y']
            trail_length = int(meteor['trail_length'])
            meteor_size = meteor['size']  # Use size parameter
            
            # Draw all valid trail points
            for i in range(trail_length):
                tx = base_x - angle_cos * i * 0.5
                ty = base_y - angle_sin * i * 0.5
                
                # Skip if off-screen
                if not (0 <= tx < 120 and 0 <= ty < 60):
                    continue
                
                # Calculate intensity
                intensity = (1 - i/meteor['trail_length']) * meteor['life']
                
                # Calculate pixel size based on meteor size and position in trail
                pixel_size = max(1, int(meteor_size * (1 - i/trail_length) * 2))
                
                # Determine if this is core or trail
                is_core = i < 2
                
                # Draw a multi-pixel point with size based on the meteor size
                # Vectorized approach for drawing meteor pixels
                # Pre-calculate the grid of offsets and distances once
                if pixel_size > 0:  # Only process if we have size
                    # Generate coordinate arrays for the full pixel block
                    y_indices, x_indices = np.mgrid[-pixel_size:pixel_size+1, -pixel_size:pixel_size+1]
                    
                    # Calculate distances from center (vectorized)
                    distances = np.sqrt(x_indices**2 + y_indices**2) / pixel_size
                    
                    # Create mask for valid distances
                    valid_mask = distances <= 1.0
                    
                    # Get pixel coordinates in the output image
                    px = int(tx) + x_indices[valid_mask]
                    py = int(ty) + y_indices[valid_mask]
                    
                    # Create mask for on-screen pixels
                    screen_mask = (px >= 0) & (px < 120) & (py >= 0) & (py < 60)
                    
                    if np.any(screen_mask):
                        # Get final valid coordinates
                        final_px = px[screen_mask]
                        final_py = py[screen_mask]
                        final_distances = distances[valid_mask][screen_mask]
                        
                        # Calculate intensities for all pixels at once
                        pixel_intensities = intensity * (1.0 - final_distances)
                        
                        # Process all pixels based on core vs trail
                        if is_core:  # Core - simpler processing
                            # Set all core pixels at once
                            for idx in range(len(final_px)):
                                instate['meteor_window'][final_py[idx], final_px[idx]] = [
                                    0.6, 0.2, pixel_intensities[idx], pixel_intensities[idx]
                                ]
                        else:  # Trail - need to check brightness
                            for idx in range(len(final_px)):
                                px_idx, py_idx = final_px[idx], final_py[idx]
                                p_intensity = pixel_intensities[idx]
                                
                                # Only update if brighter
                                if p_intensity > instate['meteor_window'][py_idx, px_idx, 2]:
                                    hue_variation = 0.5 + random.random() * 0.1
                                    trail_value = 0.7 * p_intensity * (1 - i/meteor['trail_length'])
                                    
                                    instate['meteor_window'][py_idx, px_idx] = [
                                        hue_variation, 0.8, trail_value, p_intensity * 0.5
                                    ]

    instate['meteors'] = new_meteors

    # Convert HSVA to BGRA for rendering - identical to original
    rgb = color.hsv2rgb(instate['meteor_window'][..., 0:3])
    alpha = instate['meteor_window'][..., 3:4]* fade_factor
    rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)
    
    # Update the image plane - identical to original
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['meteor_plane'], 
        rgb_out[:,:,:]
    )

def volcanic_mountain(instate, outstate):
    
    if instate['count'] == 0:
        path=sound_path / 'Volcano Eruption.wav'
        outstate['soundengine'].schedule_event(path, time.time()+25, 10)
        
        # Initialize volcano window
        instate['volcano_window'] = np.zeros((60, 120, 4))  # HSVA format
        instate['volcano_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((60, 120, 4), dtype=np.uint8),
            position=(0, 0, 49.8),
            rotation=(0, 0, 0),
            scale=(9, 9)
        )
        instate['peak_height'] = 35
        peak_height = 35

        # Generate mountain profile
        base_points = np.array([
            [0, 60], [30, 45], [45, 40], 
            [55, peak_height], [60, peak_height+3], [65, peak_height],
            [75, 40], [90, 50], [120, 60]
        ])
        
        # Precompute coordinate grids
        instate['y_grid'], instate['x_grid'] = np.meshgrid(np.arange(60), np.arange(120), indexing='ij')
        
        # Generate smooth mountain profile
        x_coords = np.arange(121)
        y_coords = np.zeros(121)
        
        # Use quadratic interpolation for smoother curves
        for i in range(len(base_points) - 1):
            mask = (x_coords >= base_points[i][0]) & (x_coords <= base_points[i+1][0])
            x_segment = (x_coords[mask] - base_points[i][0]) / (base_points[i+1][0] - base_points[i][0])
            y_segment = (1 - x_segment)**2 * base_points[i][1] + \
                       2 * (1 - x_segment) * x_segment * (base_points[i][1] + base_points[i+1][1])/2 + \
                       x_segment**2 * base_points[i+1][1]
            y_coords[mask] = y_segment

        # Precompute transition line
        base_transition = 45
        noise_amplitude = 3
        transition_noise = base_transition + noise_amplitude * (
            0.3 * np.sin(np.arange(120) * 0.1) +
            0.4 * np.sin(np.arange(120) * 0.05) +
            0.3 * np.sin(np.arange(120) * 0.02) +
            0.5 * np.random.random(120)
        )
        
        instate['transition_line'] = transition_noise
        instate['mountain_profile'] = y_coords
        
        # Precompute mountain masks and factors
        mountain_heights = np.tile(y_coords[:-1], (60, 1))
        height_factors = (60 - instate['y_grid']) / (60 - mountain_heights+1e-10)
        instate['height_factors'] = np.clip(height_factors, 0, 1)
        instate['mountain_mask'] = instate['y_grid'] >= mountain_heights
        
        # Precompute transition masks
        transition_heights = np.tile(transition_noise, (60, 1))
        instate['transition_masks'] = {
            'below': instate['y_grid'] >= (transition_heights + 2),
            'gradient': (instate['y_grid'] >= transition_heights) & (instate['y_grid'] < (transition_heights + 2))
        }
        
        # Precompute crater rim factors
        rim_dx = np.abs(instate['x_grid'] - 60)
        rim_dy = np.abs(instate['y_grid'] - peak_height)
        instate['rim_mask'] = (rim_dy < 3) & (rim_dx < 10)
        instate['rim_factors'] = (1 - rim_dx/10) * (1 - rim_dy/3)
        
        # Generate multi-layered texture
        texture_res = (60, 120)
        scale_1 = cv2.resize(np.random.uniform(-1, 1, (15, 30)), texture_res[::-1])
        scale_2 = cv2.resize(np.random.uniform(-1, 1, (30, 60)), texture_res[::-1])
        scale_3 = np.random.uniform(-1, 1, texture_res)
        
        surface_texture = scale_1 * 0.5 + scale_2 * 0.3 + scale_3 * 0.2
        instate['surface_texture'] = (surface_texture - surface_texture.min()) / (surface_texture.max() - surface_texture.min())
        
        # Initialize particle systems using preallocated arrays
        max_particles = 1000
        instate['max_particles'] = max_particles
        
        # Main particles (lava/ash)
        instate['particle_count'] = 0
        instate['particle_x'] = np.zeros(max_particles)
        instate['particle_y'] = np.zeros(max_particles)
        instate['particle_vx'] = np.zeros(max_particles)
        instate['particle_vy'] = np.zeros(max_particles)
        instate['particle_life'] = np.zeros(max_particles)
        instate['particle_type'] = np.zeros(max_particles, dtype=np.int8)  # 0=ash, 1=lava
        instate['particle_size'] = np.zeros(max_particles)
        instate['particle_alpha'] = np.zeros(max_particles)
        
        # Smoke particles - slightly increased for more prominence
        max_smoke = 1200
        instate['max_smoke'] = max_smoke
        instate['smoke_count'] = 0
        instate['smoke_x'] = np.zeros(max_smoke)
        instate['smoke_y'] = np.zeros(max_smoke)
        instate['smoke_vx'] = np.zeros(max_smoke)
        instate['smoke_vy'] = np.zeros(max_smoke)
        instate['smoke_life'] = np.zeros(max_smoke)
        instate['smoke_size'] = np.zeros(max_smoke)
        instate['smoke_alpha'] = np.zeros(max_smoke)
        
        # Generate lava channels as numpy arrays
        instate['lava_channels'] = []
        peak_x = 60
        for _ in range(5):
            points = []
            x = peak_x + np.random.uniform(-15, 15)
            y = peak_height + 2
            while y < 60:
                points.append([x, y])
                x += np.random.uniform(-1, 1)
                y += np.random.uniform(0.5, 1.5)
            instate['lava_channels'].append(np.array(points))
        
        # Precompute lava channel base points
        instate['lava_points'] = []
        for channel in instate['lava_channels']:
            instate['lava_points'].append(channel.astype(int))
        
        # Precompute offsets for lava channel expansion
        instate['lava_offsets'] = np.array([
            [-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2],
            [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
            [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
            [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
            [2, -2], [2, -1], [2, 0], [2, 1], [2, 2]
        ])
        
        # Reusable buffers for rendering
        instate['lava_mask'] = np.zeros((60, 120), dtype=bool)
        instate['pulse_buffer'] = np.zeros((60, 120))
        instate['rgb_out'] = np.zeros((60, 120, 4), dtype=np.uint8)
        
        instate['start_time'] = time.time()
        instate['last_update'] = time.time()
        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['volcano_plane'])
        return

    # Time calculations
    current_time = time.time()
    dt = current_time - instate['last_update']
    instate['last_update'] = current_time
    elapsed_time = current_time - instate['start_time']
        
    # Calculate fade factor
    fade_duration = 5.0
    total_duration = instate.get('duration', 60.0)
    
    if elapsed_time < fade_duration:
        global_alpha = elapsed_time / fade_duration
    elif elapsed_time > total_duration - fade_duration:
        global_alpha = (total_duration - elapsed_time) / fade_duration
    else:
        global_alpha = 1.0
    global_alpha = np.clip(global_alpha, 0, 1)

    level = outstate.get('volcano_level', 0)
    window = np.zeros((60, 120, 4))

    # Vectorized mountain rendering
    mountain_mask = instate['mountain_mask']
    height_factors = instate['height_factors']
    
    # Base colors and heat influence - vectorized
    rock_hues = np.full_like(instate['x_grid'], 0.05, dtype=float)
    rock_hues[instate['transition_masks']['below']] = 0.3
    
    # Apply gradient in transition zone - vectorized
    gradient_mask = instate['transition_masks']['gradient']
    if gradient_mask.any():
        gradient_factor = (instate['y_grid'][gradient_mask] - instate['transition_line'][instate['x_grid'][gradient_mask]]) / 2
        rock_hues[gradient_mask] = 0.05 + (0.3 - 0.05) * gradient_factor
    
    # Heat and texture calculations - vectorized
    heat_factor = np.clip(level * 2 * height_factors, 0, 1)
    texture_val = instate['surface_texture']
    banding = np.sin(instate['y_grid'] * 0.8) * 0.1
    texture_val = np.clip(texture_val + banding, 0, 1)
    
    # Apply mountain colors - vectorized
    window[mountain_mask, 0] = rock_hues[mountain_mask]
    window[mountain_mask, 1] = 0.5 + texture_val[mountain_mask] * 0.2
    window[mountain_mask, 2] = 0.25 + texture_val[mountain_mask] * 0.15
    window[mountain_mask, 3] = 1.0

    # Apply heat influence - vectorized
    window[mountain_mask, 1] = window[mountain_mask, 1] * (1-heat_factor[mountain_mask]) + 0.9 * heat_factor[mountain_mask]
    window[mountain_mask, 2] = window[mountain_mask, 2] * (1-heat_factor[mountain_mask]) + (0.4 + texture_val[mountain_mask] * 0.2) * heat_factor[mountain_mask]

    # Vectorized lava rendering
    lava_intensity = np.clip(level * 1.5, 0.2, 1)
    if lava_intensity > 0:
        # Reset lava mask
        lava_mask = instate['lava_mask']
        lava_mask.fill(False)
        mountain_profile = instate['mountain_profile'][:-1]
        
        # Process all lava channels efficiently
        for points in instate['lava_points']:
            # Filter points above mountain surface
            valid_mask = (
                (points[:, 0] >= 0) & (points[:, 0] < 120) & 
                (points[:, 1] >= 0) & (points[:, 1] < 60) &
                (points[:, 1] >= mountain_profile[np.clip(points[:, 0], 0, 119)])
            )
            valid_points = points[valid_mask]
            
            if len(valid_points) > 0:
                # For each valid point, apply offsets to create a wider channel
                for offset in instate['lava_offsets']:
                    # Apply offset to all points at once
                    shifted = valid_points + offset
                    
                    # Filter valid shifted points
                    valid_shifted = (
                        (shifted[:, 0] >= 0) & (shifted[:, 0] < 120) & 
                        (shifted[:, 1] >= 0) & (shifted[:, 1] < 60) &
                        (shifted[:, 1] >= mountain_profile[np.clip(shifted[:, 0], 0, 119)])
                    )
                    
                    # Set mask for valid positions
                    if np.any(valid_shifted):
                        y_coords = shifted[valid_shifted, 1]
                        x_coords = shifted[valid_shifted, 0]
                        lava_mask[y_coords, x_coords] = True
        
        # Apply pulsating effect
        pulse = 0.8 + 0.2 * np.sin(current_time * 2 + instate['x_grid'] * 0.1)
        
        # Get lava pixel coordinates
        lava_y, lava_x = np.where(lava_mask)
        
        if len(lava_y) > 0:
            # Generate random hue variations for all lava pixels at once
            hue_variations = 0.05 + np.random.random(len(lava_y)) * 0.05
            
            # Apply colors to all lava pixels in one operation
            window[lava_y, lava_x, 0] = hue_variations
            window[lava_y, lava_x, 1] = 0.9
            window[lava_y, lava_x, 2] = 0.8 * lava_intensity * pulse[lava_y, lava_x]
            window[lava_y, lava_x, 3] = 1.0

    # Smoke system - slightly enhanced but still similar to original
    smoke_chance = 0.3 + level * 0.1  # Slightly increased but not too much
    
    if np.random.random() < smoke_chance:
        wind = outstate.get('wind', 0)
        
        # Add new smoke particle
        if instate['smoke_count'] < instate['max_smoke']:
            idx = instate['smoke_count']
            
            # Position with slight variation based on level
            spread = 3 * level  # More spread at higher levels
            instate['smoke_x'][idx] = 60 + np.random.uniform(-3 - spread, 3 + spread)
            instate['smoke_y'][idx] = instate['peak_height']
            
            # Velocity with slight wind influence
            instate['smoke_vx'][idx] = np.random.uniform(-0.5, 0.5) + wind
            instate['smoke_vy'][idx] = np.random.uniform(-1.9, -0.8) * (1 + level * 0.2)  # Slightly faster rise at higher levels
            
            # Slightly enhanced properties for more visibility
            instate['smoke_life'][idx] = 1.0
            instate['smoke_size'][idx] = np.random.uniform(2, 4) * (1 + level * 0.2)  # Slightly larger at higher levels
            instate['smoke_alpha'][idx] = np.random.uniform(0.4, 0.7)  # Slightly more opaque
            
            instate['smoke_count'] += 1

    # Update all smoke particles vectorized
    if instate['smoke_count'] > 0:
        active = slice(0, instate['smoke_count'])
        
        # Update positions and properties vectorized - similar to original
        wind = outstate.get('wind', 0)
        instate['smoke_x'][active] += (instate['smoke_vx'][active] + wind * 0.5) * dt
        instate['smoke_y'][active] += instate['smoke_vy'][active] * dt
        instate['smoke_vx'][active] += np.random.uniform(-0.1, 0.1, instate['smoke_count'])
        instate['smoke_vy'][active] *= 0.99  # Gradually slow vertical movement
        instate['smoke_size'][active] *= 1.03  # Same growth rate as original
        instate['smoke_life'][active] -= 0.009  # Same life decrease as original
        instate['smoke_alpha'][active] *= 0.99  # Same alpha decrease
        
        # Filter dead particles
        valid_mask = instate['smoke_life'][active] > 0
        if not np.all(valid_mask):
            valid_count = np.sum(valid_mask)
            
            # Compact arrays - move valid particles to front
            for attr in ['smoke_x', 'smoke_y', 'smoke_vx', 'smoke_vy', 
                       'smoke_life', 'smoke_size', 'smoke_alpha']:
                instate[attr][:valid_count] = instate[attr][active][valid_mask]
            
            instate['smoke_count'] = valid_count
        
        # Process smoke particles in batches
        batch_size = 50
        for batch_start in range(0, instate['smoke_count'], batch_size):
            batch_end = min(batch_start + batch_size, instate['smoke_count'])
            batch_indices = np.arange(batch_start, batch_end)
            
            # Filter on-screen particles
            x_positions = instate['smoke_x'][batch_indices].astype(int)
            y_positions = instate['smoke_y'][batch_indices].astype(int)
            #sizes = instate['smoke_size'][batch_indices].astype(int)
            
            on_screen = (
                (x_positions >= 0) & (x_positions < 120) &
                (y_positions >= 0) & (y_positions < 60)
            )
            
            if not np.any(on_screen):
                continue
                
            # Process visible particles
            visible_indices = batch_indices[on_screen]
            for idx in visible_indices:
                x, y = int(instate['smoke_x'][idx]), int(instate['smoke_y'][idx])
                size = int(instate['smoke_size'][idx])
                
                # Calculate smoke area bounds
                y_min = max(0, y - size)
                y_max = min(60, y + size + 1)
                x_min = max(0, x - size)
                x_max = min(120, x + size + 1)
                
                # Skip if area is too small
                if x_max <= x_min or y_max <= y_min:
                    continue
                
                # Create coordinate grids for the smoke area
                y_grid, x_grid = np.meshgrid(
                    np.arange(y_min, y_max),
                    np.arange(x_min, x_max),
                    indexing='ij'
                )
                
                # Calculate distances and mask
                distances = np.sqrt((x_grid - x)**2 + (y_grid - y)**2)
                mask = distances <= size
                
                if mask.any():  # Only process if we have pixels to draw
                    # Calculate intensity with slight enhancement for visibility
                    intensity = np.zeros_like(distances)
                    intensity[mask] = (1 - distances[mask]/size) * instate['smoke_life'][idx] * instate['smoke_alpha'][idx] * 1.2  # 20% boost
                    
                    # Create smoke color array with slightly improved opacity
                    smoke_color = np.zeros((*distances.shape, 4))
                    smoke_color[..., 0] = 0.1  # Hue
                    smoke_color[..., 1] = 0.1  # Saturation
                    smoke_color[..., 2] = 0.5  # Value
                    smoke_color[..., 3] = intensity * 0.8  # Alpha - slightly increased from 0.7
                    
                    # Update window only where smoke is more opaque
                    update_mask = mask & (smoke_color[..., 3] > window[y_grid, x_grid, 3])
                    if update_mask.any():
                        window[y_grid[update_mask], x_grid[update_mask]] = smoke_color[update_mask]

    # Particle system - vectorized
    particle_chance = level * 0.3
    if np.random.random() < particle_chance:
        # Add new particles
        if instate['particle_count'] < instate['max_particles']:
            spread = 8 + level * 15
            count = min(10, instate['max_particles'] - instate['particle_count'])
            start_idx = instate['particle_count']
            end_idx = start_idx + count
            
            # Initialize multiple particles at once
            instate['particle_x'][start_idx:end_idx] = 60 + np.random.uniform(-spread, spread, count)
            instate['particle_y'][start_idx:end_idx] = instate['peak_height']
            instate['particle_vx'][start_idx:end_idx] = np.random.uniform(-1, 1, count)
            instate['particle_vy'][start_idx:end_idx] = np.random.uniform(-2, -1, count)
            instate['particle_life'][start_idx:end_idx] = 1.0
            instate['particle_type'][start_idx:end_idx] = (np.random.random(count) < level).astype(np.int8)  # 1=lava, 0=ash
            instate['particle_size'][start_idx:end_idx] = np.random.uniform(2, 4, count)
            instate['particle_alpha'][start_idx:end_idx] = np.random.uniform(0.3, 0.6, count)
            
            instate['particle_count'] += count

    # Update existing particles vectorized
    if instate['particle_count'] > 0:
        active = slice(0, instate['particle_count'])
        
        # Update positions and properties vectorized
        instate['particle_x'][active] += instate['particle_vx'][active] * dt
        instate['particle_y'][active] += instate['particle_vy'][active] * dt
        instate['particle_vy'][active] += 0.1 * dt
        instate['particle_life'][active] -= 0.02
        
        # Filter valid particles
        valid_mask = (
            (instate['particle_life'][active] > 0) & 
            (instate['particle_y'][active] >= 0) & 
            (instate['particle_y'][active] < 60) & 
            (instate['particle_x'][active] >= 0) & 
            (instate['particle_x'][active] < 120)
        )
        
        if not np.all(valid_mask):
            valid_count = np.sum(valid_mask)
            
            # Compact arrays - move valid particles to front
            for attr in ['particle_x', 'particle_y', 'particle_vx', 'particle_vy', 
                       'particle_life', 'particle_type', 'particle_size', 'particle_alpha']:
                instate[attr][:valid_count] = instate[attr][active][valid_mask]
            
            instate['particle_count'] = valid_count
        
        # Draw all particles at once
        x_positions = instate['particle_x'][:instate['particle_count']].astype(int)
        y_positions = instate['particle_y'][:instate['particle_count']].astype(int)
        
        for i in range(instate['particle_count']):
            x, y = x_positions[i], y_positions[i]
            
            # Skip if off-screen
            if not (0 <= x < 120 and 0 <= y < 60):
                continue
                
            # Set color based on type
            if instate['particle_type'][i] == 1:  # Lava
                window[y, x] = [0.05, 0.9, 0.9 * instate['particle_life'][i], instate['particle_life'][i]]
            else:  # Ash
                window[y, x] = [0.1, 0.2, 0.3 * instate['particle_life'][i], instate['particle_life'][i] * 0.5]

    # Apply global alpha
    window[..., 3] *= global_alpha
    
    # Convert to RGB
    rgb = color.hsv2rgb(window[..., 0:3])
    alpha = window[..., 3:4]
    rgb_out = instate['rgb_out']
    rgb_out[..., :3] = rgb * 255
    rgb_out[..., 3:] = alpha * 255
    
    # Update the texture
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['volcano_plane'], 
        rgb_out
    )

@jit(nopython=True)
def _update_particle_positions(x, y, speed, wind, dt, current_time):
    """Numba-accelerated particle position update"""
    wind_effect = wind * 30
    new_x = x + (speed * np.sign(wind) + wind_effect) * dt
    new_y = y + np.sin(current_time + x * 0.1) * dt
    return new_x, new_y

@jit(nopython=True)
def _calculate_particle_intensity(x, y, size, alpha, density):
    """Numba-accelerated intensity calculation"""
    intensity = np.zeros((60, 120), dtype=np.float32)
    
    # Calculate affected region bounds
    x_min = max(0, int(x - size))
    x_max = min(120, int(x + size + 1))
    y_min = max(0, int(y - size))
    y_max = min(60, int(y + size + 1))
    
    # Calculate intensities for affected pixels
    for yi in range(y_min, y_max):
        for xi in range(x_min, x_max):
            dist = np.sqrt((xi - x)**2 + (yi - y)**2)
            if dist <= size:
                intensity[yi, xi] = (1 - dist/size) * alpha * density
                
    return intensity

def sandstorm(instate, outstate):
    if instate['count'] == 0:
        # Initialize constants in instate
        instate['MAX_PARTICLES'] = 30
        instate['BATCH_SIZE'] = 30
        
        # Initialize separate arrays for particle properties
        instate.update({
            'sand_window': np.zeros((60, 120, 4)),  # HSVA format
            'particle_x': np.zeros(instate['MAX_PARTICLES'], dtype=np.float32),
            'particle_y': np.zeros(instate['MAX_PARTICLES'], dtype=np.float32),
            'particle_speed': np.zeros(instate['MAX_PARTICLES'], dtype=np.float32),
            'particle_size': np.zeros(instate['MAX_PARTICLES'], dtype=np.float32),
            'particle_color': np.zeros(instate['MAX_PARTICLES'], dtype=np.float32),
            'particle_alpha': np.zeros(instate['MAX_PARTICLES'], dtype=np.float32),
            'particle_count': 0,
            'last_update': time.time(),
            'rgb_out': np.zeros((60, 120, 4), dtype=np.uint8)
        })
        
        # Create image plane
        instate['sand_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((60, 120, 4), dtype=np.uint8),
            position=(0, 0, 49.4),
            rotation=(0, 0, 0),
            scale=(9, 9)
        )
        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['sand_plane'])
        return

    # Get current state
    current_time = time.time()
    dt = current_time - instate['last_update']
    instate['last_update'] = current_time

    sand_density = outstate.get('sand_density', 0)
    wind = outstate.get('wind', 0)
    
    # Clear the window efficiently
    window = instate['sand_window']
    window.fill(0)

    # Generate new particles
    particle_count = int(sand_density * 2)
    if particle_count > 0:
        available_slots = instate['MAX_PARTICLES'] - instate['particle_count']
        new_count = min(particle_count, available_slots)
        
        if new_count > 0:
            start_idx = instate['particle_count']
            end_idx = start_idx + new_count
            
            # Initialize new particles
            instate['particle_x'][start_idx:end_idx] = -5 if wind > 0 else 125
            instate['particle_y'][start_idx:end_idx] = np.random.uniform(0, 60, new_count)
            instate['particle_speed'][start_idx:end_idx] = np.random.uniform(30, 50, new_count)
            instate['particle_size'][start_idx:end_idx] = np.random.uniform(0.5, 2.0, new_count)
            instate['particle_color'][start_idx:end_idx] = np.random.uniform(0.08, 0.12, new_count)
            instate['particle_alpha'][start_idx:end_idx] = np.random.uniform(0.3, 0.7, new_count)
            
            instate['particle_count'] += new_count

    # Process active particles
    if instate['particle_count'] > 0:
        # Update particle positions
        active_slice = slice(0, instate['particle_count'])
        new_x, new_y = _update_particle_positions(
            instate['particle_x'][active_slice],
            instate['particle_y'][active_slice],
            instate['particle_speed'][active_slice],
            wind, dt, current_time
        )
        
        # Update positions
        instate['particle_x'][active_slice] = new_x
        instate['particle_y'][active_slice] = new_y
        
        # Filter out-of-bounds particles
        valid_mask = (new_x >= -10) & (new_x <= 130) & \
                    (new_y >= -5) & (new_y <= 65)
        
        if not np.all(valid_mask):
            # Compact arrays
            valid_count = np.sum(valid_mask)
            for arr_name in ['particle_x', 'particle_y', 'particle_speed', 
                           'particle_size', 'particle_color', 'particle_alpha']:
                instate[arr_name][:valid_count] = instate[arr_name][active_slice][valid_mask]
            instate['particle_count'] = valid_count
        
        # Process particles in batches
        for batch_start in range(0, instate['particle_count'], instate['BATCH_SIZE']):
            batch_end = min(batch_start + instate['BATCH_SIZE'], instate['particle_count'])
            
            for i in range(batch_start, batch_end):
                intensity = _calculate_particle_intensity(
                    instate['particle_x'][i],
                    instate['particle_y'][i],
                    instate['particle_size'][i],
                    instate['particle_alpha'][i],
                    sand_density
                )
                
                # Update window where intensity is higher
                mask = intensity > window[..., 3]
                window[mask, 0] = instate['particle_color'][i]
                window[mask, 1] = 0.6
                window[mask, 2] = 0.9
                window[mask, 3] = intensity[mask]

    # Convert to RGB efficiently
    rgb = color.hsv2rgb(window[..., 0:3])
    alpha = window[..., 3:4]
    rgb_out = instate['rgb_out']
    rgb_out[..., :3] = rgb * 255
    rgb_out[..., 3:] = alpha * 255
    
    # Update texture
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['sand_plane'],
        rgb_out
    )


def secondary_sandstorm(instate, outstate):
    if instate['count'] == 0:
        # Initialize constants in instate
        instate['MAX_PARTICLES'] = 500
        instate['BATCH_SIZE'] = 100
        
        # Initialize separate arrays for particle properties
        instate.update({
            'sand_window': np.zeros((32, 300, 4)),  # HSVA format
            'particle_x': np.zeros(instate['MAX_PARTICLES'], dtype=np.float32),  # Cartesian coordinates
            'particle_y': np.zeros(instate['MAX_PARTICLES'], dtype=np.float32),
            'particle_r': np.zeros(instate['MAX_PARTICLES'], dtype=np.float32),  # Polar coordinates
            'particle_theta': np.zeros(instate['MAX_PARTICLES'], dtype=np.float32),
            'particle_speed': np.zeros(instate['MAX_PARTICLES'], dtype=np.float32),
            'particle_size': np.zeros(instate['MAX_PARTICLES'], dtype=np.float32),
            'particle_color': np.zeros(instate['MAX_PARTICLES'], dtype=np.float32),
            'particle_alpha': np.zeros(instate['MAX_PARTICLES'], dtype=np.float32),
            'particle_count': 0,
            'last_update': time.time(),
            'rgb_out': np.zeros((32, 300, 4), dtype=np.uint8)
        })
        
        # Create image plane for secondary display
        instate['sand_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((32, 300, 4), dtype=np.uint8),
            position=(0, 0, 49.6),
            rotation=(0, 0, 0),
            scale=(9, 9)
        )
        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['sand_plane'])
        return

    # Get current state
    current_time = time.time()
    dt = current_time - instate['last_update']
    instate['last_update'] = current_time

    sand_density = outstate.get('sand_density', 0)
    wind = outstate.get('wind', 0)
    Full_sky = outstate.get('skyfull', True)
    
    # Determine sky width based on Full_sky setting
    sky_width = 32 if Full_sky else 64
    
    # Clear the window efficiently
    window = instate['sand_window']
    window.fill(0)

    # Generate new particles
    particle_count = int(sand_density * 2)
    if particle_count > 0:
        available_slots = instate['MAX_PARTICLES'] - instate['particle_count']
        new_count = min(particle_count, available_slots)
        
        if new_count > 0:
            start_idx = instate['particle_count']
            end_idx = start_idx + new_count
            
            # Initialize new particles in cartesian space
            x_start = -1 if wind > 0 else 1
            instate['particle_x'][start_idx:end_idx] = x_start
            instate['particle_y'][start_idx:end_idx] = np.random.uniform(-5, 5, new_count)
            
            # Convert to polar coordinates
            x = instate['particle_x'][start_idx:end_idx]
            y = instate['particle_y'][start_idx:end_idx]
            instate['particle_r'][start_idx:end_idx] = np.sqrt(x**2 + y**2)
            instate['particle_theta'][start_idx:end_idx] = np.arctan2(y, x)
            # Ensure theta is positive
            neg_theta_mask = instate['particle_theta'][start_idx:end_idx] < 0
            instate['particle_theta'][start_idx:end_idx][neg_theta_mask] += 2 * np.pi
            
            # Other particle properties
            instate['particle_speed'][start_idx:end_idx] = np.random.uniform(0.5, 1.0, new_count)
            instate['particle_size'][start_idx:end_idx] = np.random.uniform(0.01, 0.11, new_count)
            instate['particle_color'][start_idx:end_idx] = np.random.uniform(0.08, 0.12, new_count)
            instate['particle_alpha'][start_idx:end_idx] = np.random.uniform(0.7, 0.9, new_count)
            
            instate['particle_count'] += new_count

    # Process active particles
    if instate['particle_count'] > 0:
        active_slice = slice(0, instate['particle_count'])
        
        # Update cartesian positions
        wind_effect = wind * dt
        instate['particle_x'][active_slice] += (
            instate['particle_speed'][active_slice] * np.sign(wind) + wind_effect
        ) * dt
        instate['particle_y'][active_slice] += np.sin(
            current_time + instate['particle_x'][active_slice] * 0.1
        ) * dt * 0.1
        
        # Update polar coordinates
        instate['particle_r'][active_slice] = np.sqrt(
            instate['particle_x'][active_slice]**2 + 
            instate['particle_y'][active_slice]**2
        )
        instate['particle_theta'][active_slice] = np.arctan2(
            instate['particle_y'][active_slice],
            instate['particle_x'][active_slice]
        )
        # Ensure theta is positive
        neg_theta_mask = instate['particle_theta'][active_slice] < 0
        instate['particle_theta'][active_slice][neg_theta_mask] += 2 * np.pi
        
        # Filter out-of-bounds particles
        valid_mask = instate['particle_r'][active_slice] <= 1.2
        
        if not np.all(valid_mask):
            # Compact arrays
            valid_count = np.sum(valid_mask)
            for arr_name in ['particle_x', 'particle_y', 'particle_r', 'particle_theta',
                           'particle_speed', 'particle_size', 'particle_color', 'particle_alpha']:
                instate[arr_name][:valid_count] = instate[arr_name][active_slice][valid_mask]
            instate['particle_count'] = valid_count
            
        # Process particles in batches
        for batch_start in range(0, instate['particle_count'], instate['BATCH_SIZE']):
            batch_end = min(batch_start + instate['BATCH_SIZE'], instate['particle_count'])
            
            for i in range(batch_start, batch_end):
                # Convert particle position to buffer coordinates using sky_width
                theta = instate['particle_theta'][i]
                r = instate['particle_r'][i]
                
                # Use sky_width for theta calculations
                theta_idx = int((theta / (2 * np.pi)) * sky_width) % sky_width
                r_idx = int(r * 299)  # Map to 300 radial steps
                
                # Skip if outside the visible range when Full_sky is False
                if not Full_sky and theta_idx >= 32:
                    continue
                
                if 0 <= theta_idx < 32 and 0 <= r_idx < 299:
                    size = instate['particle_size'][i]
                    
                    # Calculate affected region
                    theta_min = max(0, theta_idx - int(size * 100))
                    theta_max = min(32, theta_idx + int(size * 100) + 1)
                    r_min = max(0, r_idx - int(size * 100))
                    r_max = min(300, r_idx + int(size * 100) + 1)
                    
                    # Create meshgrid for vectorized calculations
                    r_grid, theta_grid = np.meshgrid(
                        np.arange(r_min, r_max),
                        np.arange(theta_min, theta_max),
                        indexing='ij'
                    )
                    
                    # Apply mask for Full_sky mode
                    if not Full_sky:
                        valid_mask = theta_grid < 32
                    else:
                        valid_mask = np.ones_like(theta_grid, dtype=bool)
                    
                    # Calculate distances vectorized
                    if Full_sky:
                        dtheta = (theta_grid - theta_idx) / 32
                    else:
                        dtheta = (theta_grid - theta_idx) / sky_width
                    
                    dr = (r_grid - r_idx) / 300
                    dist = np.sqrt(dtheta**2 + dr**2)
                    
                    # Create mask for points within particle radius
                    particle_mask = (dist <= size/5) & valid_mask
                    
                    if np.any(particle_mask):
                        # Calculate intensity for all valid points
                        #r_factor = r_grid / 180  # Radial factor
                        intensity = np.zeros_like(dist)
                        intensity[particle_mask] = (
                            (1 - dist[particle_mask]/size) * 
                            instate['particle_alpha'][i] * 
                            sand_density
                        )
                        
                        # Create mask for points where new intensity is higher
                        update_mask = (intensity > window[theta_grid, r_grid, 3]) & particle_mask
                        
                        if np.any(update_mask):
                            # Update only necessary points
                            r_update = r_grid[update_mask]
                            theta_update = theta_grid[update_mask]
                            intensity_update = intensity[update_mask]
                            
                            # Update window efficiently
                            window[theta_update, r_update, 0] = instate['particle_color'][i]
                            window[theta_update, r_update, 1] = 0.4
                            window[theta_update, r_update, 2] = 0.8
                            window[theta_update, r_update, 3] = intensity_update

    # Convert to RGB efficiently
    rgb = color.hsv2rgb(window[..., 0:3])
    alpha = window[..., 3:4]
    rgb_out = instate['rgb_out']
    rgb_out[..., :3] = rgb * 255
    rgb_out[..., 3:] = alpha * 255
    
    # Update texture
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['sand_plane'],
        rgb_out
    )


@jit(nopython=True)
def calculate_fade_factor(elapsed_time, total_duration, fade_in_duration=3.0):
    """Calculate the fade factor for the meteor shower based on time."""
    fade_out_start = total_duration - 5.0
    
    if elapsed_time < fade_in_duration:
        fade_factor = elapsed_time / fade_in_duration
    elif elapsed_time > fade_out_start:
        fade_factor = (total_duration - elapsed_time) / (total_duration - fade_out_start)
    else:
        fade_factor = 1.0
    return max(0.0, min(1.0, fade_factor))

@jit(nopython=True)
def update_meteor_positions(theta, radius, direction, speed, life, dt=0.016):
    """Update meteor positions using vectorized operations."""
    # Move toward center
    radius -= speed
    
    # Scale angular movement based on radius (faster at edge, slower near center)
    angular_speed = direction * (0.04 / (radius + 0.2))
    theta += angular_speed
    
    # Decrease life
    life -= 0.015
    
    return theta, radius, life

@jit(nopython=True)
def render_trail_points(theta, radius, life, size, speed, direction, fade_factor, output_buffer):
    """Render meteor trails efficiently."""
    # Calculate buffer dimensions
    buffer_height, buffer_width = output_buffer.shape[:2]
    
    for i in range(len(theta)):
        if life[i] <= 0 or radius[i] <= 0.05 or radius[i] >= 1.0:
            continue
            
        # Calculate trail length based on size
        trail_length = int(30 + size[i] * 10)
        
        # Generate trail points
        for t in range(trail_length):
            # Calculate trail position with proper perspective
            trail_r = radius[i] + (t * speed[i] * 0.5)
            trail_theta = theta[i] - (t * direction[i] * 0.02 / (trail_r + 0.2))
            
            # Skip if outside visible range
            if trail_r >= 1.0:
                continue
                
            # Convert to display coordinates
            theta_idx = int((trail_theta % (2 * np.pi)) / (2 * np.pi) * buffer_height)
            r_idx = int(trail_r * buffer_width)
            
            if theta_idx < 0 or theta_idx >= buffer_height or r_idx < 0 or r_idx >= buffer_width:
                continue
                
            # Calculate intensity with fade
            intensity = (1 - t/trail_length) * life[i] * fade_factor
            
            # Calculate trail width based on radius (wider at edge, thinner at center)
            trail_width = int(2 + size[i] * (trail_r + 0.3))
            
            # Determine core vs. trail
            is_core = t < 3
            
            # Process pixels in trail width
            width_range = trail_width if is_core else max(1, trail_width // 2)
            
            for dy in range(-width_range, width_range + 1):
                for dx in range(-width_range, width_range + 1):
                    # Apply radial compression (less spread in radial direction)
                    dr = int(dx * (1 - trail_r * 0.7))
                    
                    # Calculate pixel position with offsets
                    px = r_idx + dr
                    py = (theta_idx + dy) % buffer_height  # Wrap around for angles
                    
                    if px < 0 or px >= buffer_width:
                        continue
                        
                    # Calculate distance from center for smooth falloff
                    dist = np.sqrt(dy*dy + dx*dx) / width_range
                    if dist > 1.0:
                        continue
                        
                    # Calculate intensity with falloff
                    pixel_intensity = intensity * (1.0 - dist)
                    
                    # Skip if too dim
                    if pixel_intensity < 0.05:
                        continue
                    
                    # Set color based on core/trail and add to buffer
                    if is_core:
                        # Core (white-blue)
                        if pixel_intensity > output_buffer[py, px, 2]:
                            output_buffer[py, px, 0] = 0.6  # Hue
                            output_buffer[py, px, 1] = 0.2  # Saturation
                            output_buffer[py, px, 2] = pixel_intensity  # Value
                            output_buffer[py, px, 3] = pixel_intensity  # Alpha
                    else:
                        # Trail (blue/cyan)
                        if pixel_intensity > output_buffer[py, px, 2]:
                            output_buffer[py, px, 0] = 0.5 + 0.05  # Hue
                            output_buffer[py, px, 1] = 0.8  # Saturation
                            output_buffer[py, px, 2] = 0.7 * pixel_intensity * (1 - t/trail_length)  # Value
                            output_buffer[py, px, 3] = pixel_intensity * 0.6  # Alpha

def secondary_meteor_shower(instate, outstate):
    if instate['count'] == 0:
        # Initialize meteor shower parameters
        instate['meteor_window'] = np.zeros((32, 300, 4), dtype=np.float32)  # HSVA format
        instate['buffer'] = np.zeros((32, 300, 4), dtype=np.float32)  # Working buffer
        instate['meteor_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((32, 300, 4), dtype=np.uint8),
            position=(0, 0, 49.8),
            rotation=(0, 0, 0),
            scale=(9, 9)
        )
        
        # Pre-allocate arrays for meteor data
        max_meteors = 20
        instate['max_meteors'] = max_meteors
        instate['meteor_count'] = 0
        instate['theta'] = np.zeros(max_meteors, dtype=np.float32)
        instate['radius'] = np.zeros(max_meteors, dtype=np.float32)
        instate['direction'] = np.zeros(max_meteors, dtype=np.float32)
        instate['speed'] = np.zeros(max_meteors, dtype=np.float32)
        instate['size'] = np.zeros(max_meteors, dtype=np.float32)
        instate['life'] = np.zeros(max_meteors, dtype=np.float32)
        
        instate['start_time'] = time.time()
        instate['last_time'] = time.time()
        instate['duration'] = instate.get('duration', 30.0)
        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['meteor_plane'])
        return

    # Get time information
    current_time = time.time()
    dt = current_time - instate['last_time']
    instate['last_time'] = current_time
    elapsed_time = current_time - instate['start_time']
    
    # Calculate fade factor
    fade_factor = calculate_fade_factor(elapsed_time, instate['duration'])
    
    # Clear the buffer
    buffer = instate['buffer']
    buffer.fill(0)

    # Generate new meteors based on meteor_rate and fade factor
    meteor_rate = outstate.get('meteor_rate', 0)
    if random.random() < meteor_rate * fade_factor:
        # Add new meteor if we have space
        if instate['meteor_count'] < instate['max_meteors']:
            idx = instate['meteor_count']
            
            # Initialize meteor
            instate['theta'][idx] = random.uniform(0, 2 * np.pi)
            instate['radius'][idx] = 0.98  # Start near edge
            instate['direction'][idx] = random.uniform(-0.8, 0.8)  # Angular movement
            instate['speed'][idx] = random.uniform(0.008, 0.012)  # Radial movement
            instate['size'][idx] = random.uniform(1.0, 2.5)  # Size
            instate['life'][idx] = 1.0  # Full life
            
            instate['meteor_count'] += 1
            
            # Add whoosh sound occasionally
            if random.random() < 0.1:
                whoosh_path = sound_path / 'Whoosh By 04.wav'
                outstate['soundengine'].schedule_event(
                    whoosh_path, 
                    time.time(), 
                    2
                )

    # Update meteor positions
    if instate['meteor_count'] > 0:
        active_meteors = instate['meteor_count']
        
        # Update positions
        instate['theta'][:active_meteors], instate['radius'][:active_meteors], instate['life'][:active_meteors] = update_meteor_positions(
            instate['theta'][:active_meteors],
            instate['radius'][:active_meteors],
            instate['direction'][:active_meteors],
            instate['speed'][:active_meteors],
            instate['life'][:active_meteors],
            dt
        )
        
        # Filter invalid meteors
        valid_mask = (instate['life'][:active_meteors] > 0) & (instate['radius'][:active_meteors] > 0.05)
        valid_count = np.sum(valid_mask)
        
        if valid_count < active_meteors:
            # Compact arrays to remove dead meteors
            for i, idx in enumerate(np.where(valid_mask)[0]):
                if i != idx:  # Only copy if position changed
                    instate['theta'][i] = instate['theta'][idx]
                    instate['radius'][i] = instate['radius'][idx]
                    instate['direction'][i] = instate['direction'][idx]
                    instate['speed'][i] = instate['speed'][idx]
                    instate['size'][i] = instate['size'][idx]
                    instate['life'][i] = instate['life'][idx]
                    
            instate['meteor_count'] = valid_count
        
        # Render meteor trails
        render_trail_points(
            instate['theta'][:instate['meteor_count']],
            instate['radius'][:instate['meteor_count']],
            instate['life'][:instate['meteor_count']],
            instate['size'][:instate['meteor_count']],
            instate['speed'][:instate['meteor_count']],  # Added speed parameter
            instate['direction'][:instate['meteor_count']],  # Added direction parameter
            fade_factor,
            buffer
        )
    
    # Combine buffer with main window (add intensity)
    instate['meteor_window'] = buffer  # Replace instead of add for cleaner trails
    
    # Apply global fade to final result
    instate['meteor_window'] *= fade_factor
    
    # Convert HSVA to BGRA for rendering
    rgb = color.hsv2rgb(instate['meteor_window'][..., 0:3])
    alpha = instate['meteor_window'][..., 3:4]
    rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)
    
    # Update the image plane
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['meteor_plane'], 
        rgb_out[:,:,:]
    )

def secondary_Aurora(instate, outstate):
    if instate['count'] == 0:
        # Initialize aurora parameters for polar display
        instate['aurora_window'] = np.zeros((32, 300, 4))  # HSVA format for 32 angle divisions, 300 radius steps
        instate['aurora_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((32, 300, 4), dtype=np.uint8),
            position=(0, 0, 49.9),  # Place aurora in background
            rotation=(0, 0, 0),
            scale=(9, 9)
        )
        # Wave parameters in polar coordinates - include overlap for continuity
        instate['base_hue'] = 0.3 + np.random.random() * 0.4  # Default to green
        # Add extra points to ensure continuity
        n_points = 21  # Odd number to avoid duplicate at wrap point
        instate['wave_points'] = np.linspace(0, 2*np.pi, n_points)
        instate['wave_heights'] = np.random.uniform(0.3, 0.7, n_points)
        # Make sure start and end heights are equal for continuity
        instate['wave_heights'][-1] = instate['wave_heights'][0]
        instate['time_offset'] = time.time()
        instate['start_time'] = time.time()
        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['aurora_plane'])
        return

    # Calculate fade factor based on duration
    current_time = time.time()
    elapsed_time = current_time - instate['start_time']
    total_duration = instate.get('duration', 30)
    fade_in_duration = total_duration * 0.2
    fade_out_start = total_duration * 0.8
    whomp=outstate.get('whomp',0.0)
    # Calculate fade factor
    if elapsed_time < fade_in_duration:
        fade_factor = elapsed_time / fade_in_duration
    elif elapsed_time > fade_out_start:
        fade_factor = (total_duration - elapsed_time) / (total_duration * 0.2)
    else:
        fade_factor = 1.0
    fade_factor = np.clip(fade_factor, 0, 1)

    # Get aurora parameters
    intensity = outstate.get('aurora_intensity', 1.0) * fade_factor
    base_hue = instate.get('base_hue', 0.3)
    
    # Update wave heights with smooth movement
    wave_speed = 0.5
    time_factor = (current_time - instate['time_offset']) * wave_speed
    
    # Create smooth wave movement using sine functions with different frequencies
    # Keep first and last points linked for continuity
    phase_offset = np.linspace(0, 2*np.pi, len(instate['wave_heights']))
    movement = np.sin(time_factor + phase_offset * 0.2) * 0.01
    instate['wave_heights'] += movement
    instate['wave_heights'] = np.clip(instate['wave_heights'], 0.2, 0.8)
    # Ensure continuity by making first and last points equal
    instate['wave_heights'][-1] = instate['wave_heights'][0]
    
    # Create extended angle array for interpolation
    theta_ext = np.linspace(-np.pi/16, 2*np.pi + np.pi/16, 34)  # Extra points for smooth wrapping
    wave_points_ext = np.concatenate([
        [instate['wave_points'][-2] - 2*np.pi],  # Add wrapped point at start
        instate['wave_points'],
        [instate['wave_points'][1] + 2*np.pi]    # Add wrapped point at end
    ])
    wave_heights_ext = np.concatenate([
        [instate['wave_heights'][-2]],  # Match second-to-last point
        instate['wave_heights'],
        [instate['wave_heights'][1]]    # Match second point
    ])
    
    # Interpolate wave points for smooth curve in polar coordinates with proper wrapping
    wave_curve = np.interp(theta_ext, wave_points_ext, wave_heights_ext)
    # Extract the valid portion (removing extra padding)
    wave_curve = wave_curve[1:-1]
    wave_curve = wave_curve[:, np.newaxis]
    
    # Create polar coordinate grids
    theta = np.linspace(0, 2*np.pi, 32)[:, np.newaxis]
    r = np.linspace(0, 1, 300)[np.newaxis, :]
    
    # Clear the window
    instate['aurora_window'].fill(0)
    
    # Calculate the base curtain shape in polar coordinates
    radial_falloff = np.clip((wave_curve - r) / 0.2, 0, 1)
    
    # Add angular streaks with proper wrapping
    streak_freq = 4  # Number of streak cycles around the circle
    streak_phase = time_factor
    streaks = np.sin(theta * streak_freq + streak_phase) * 0.1
    streaks = np.vstack([streaks, streaks[0]])  # Add wrapped point
    radial_falloff *= (1 + streaks[:-1])
    
    # Add some noise to make it more organic
    noise = np.random.random((32, 300)) * 0.1
    radial_falloff = np.clip(radial_falloff + noise, 0, 1)
    
    # Create color variations with proper wrapping
    hue_freq = 2  # Number of color variation cycles
    hue_phase = time_factor * 0.5
    hue_variation = np.sin(theta * hue_freq + hue_phase) * 0.05
    saturation = np.ones_like(radial_falloff) * 0.8
    
    # Assign colors to the aurora window
    
    instate['aurora_window'][..., 0] = base_hue + hue_variation  # Hue
    instate['aurora_window'][..., 1] = saturation  # Saturation
    instate['aurora_window'][..., 2] = radial_falloff * intensity * 0.7  # Value
    instate['aurora_window'][..., 3] = radial_falloff * 0.4 * intensity*np.clip((1+whomp),0,2)  # Alpha
    
    # Add swirling motion with proper wrapping
    swirl_freq = 3  # Number of swirl cycles
    swirl_phase = time_factor * 0.2
    swirl = np.sin(theta * swirl_freq + r * 10 + swirl_phase) * 0.3 + 0.7
    swirl = np.vstack([swirl, swirl[0]])  # Add wrapped point
    swirl_mask = swirl[:-1]
    instate['aurora_window'][..., 2] *= swirl_mask
    instate['aurora_window'][..., 3] *= swirl_mask
    
    # Add radial fade towards center
    center_fade = np.clip(r * 1.5, 0, 1)
    instate['aurora_window'][..., 2] *= center_fade
    instate['aurora_window'][..., 3] *= center_fade
    
    # Ensure continuity at the wrap point by copying first row to last
    instate['aurora_window'][0] = instate['aurora_window'][-1]
    
    # Convert HSVA to BGRA for rendering
    rgb = color.hsv2rgb(instate['aurora_window'][..., 0:3])
    alpha = instate['aurora_window'][..., 3:4]
    rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)
    
    # Update the image plane
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['aurora_plane'], 
        rgb_out[:,:,:]
    )

def secondary_lightning(instate, outstate):
    if instate['count'] == 0:
        # Initialize lightning parameters
        instate['lightning_window'] = np.zeros((32, 300, 4))  # HSVA format
        instate['lightning_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((32, 300, 4), dtype=np.uint8),
            position=(0, 0, 10),
            rotation=(0, 0, 0),
            scale=(1, 1)
        )
        
        # Choose a random row for the lightning
        instate['row'] = np.random.randint(0, 32)
        
        # Choose random start point and length
        instate['start'] = np.random.randint(0, 150)
        instate['length'] = np.random.randint(175, 250)
        instate['end'] = min(instate['start'] + instate['length'], 299)
        instate['color']=np.random.random()
        # Generate a single path
        path = []
        x = instate['start']
        while x < instate['end']:
            path.append(x)
            x += np.random.randint(1, 4)  # Random step size
        instate['path'] = np.array(path)
        
        # Initialize parameters
        instate['intensity'] = 1.0
        instate['color_shift'] = 0.0
        instate['last_update'] = time.time()
        instate['start_time'] = time.time()
        
        # Schedule thunder sound
        instate['thunder_distance'] = np.random.random() * 20 + 3  # 3-23 units away
        instate['thunder_delay'] = instate['thunder_distance'] / 340 * 1000  # Realistic speed of sound delay in seconds
        instate['thunder_scheduled'] = False
        distance=np.random.random()*10+7

        #make a string with thunder sound location
        spath=np.random.choice(['Thunder Clap Loud.wav','loud-thunder-192165.mp3','thunder-307513.mp3','peals-of-thunder-191992.mp3'])
        path=sound_path / spath

        outstate['soundengine'].schedule_event(path, time.time() + distance/10, 10)

        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['lightning_plane'])
        return
    
    # Time calculations
    current_time = time.time()
    dt = current_time - instate['last_update']
    elapsed = current_time - instate['start_time']
    instate['last_update'] = current_time
    
    # Handle thunder sound based on distance

    # Decay from previous frame
    instate['lightning_window'] *= np.exp(-dt * 3)
    
    # Update intensity with decay and flicker
    flicker = 0.6 + 0.8 * np.random.random()
    decay_rate = 0.98 if elapsed < 0.8 else 0.9  # Faster decay after initial flash
    instate['intensity'] *= decay_rate * flicker
    
    # Shift color over time (blue to white to yellow)
    instate['color_shift'] = min(instate['color_shift'] + dt * 0.5, 1.0)
    
    # Only continue if lightning is still visible
    if instate['intensity'] > 0.01:
        row = instate['row']
        path = instate['path']
        
        # Create mask for valid positions
        valid_mask = (path >= 0) & (path < 300)
        valid_path = path[valid_mask]
        
        if len(valid_path) > 0:
            # Generate random brightnesses along the path
            segment_count = 8
            segments = np.array_split(np.arange(len(valid_path)), segment_count)
            segment_brightnesses = np.random.random(segment_count) * 0.7 + 0.3
            
            # Map segment brightnesses to each point in the path
            brightnesses = np.zeros(len(valid_path))
            for i, segment in enumerate(segments):
                brightnesses[segment] = segment_brightnesses[i]
            
            # Calculate final intensity for each point
            intensities = instate['intensity'] * brightnesses * (0.8 + 0.4 * np.random.random(len(valid_path)))
            
            # Determine color parameters - shift from blue to white/yellow
            hues = instate['color'] - (instate['color_shift'] * 0.1 * np.random.random(len(valid_path)))
            saturations = 0.8 - (instate['color_shift'] * 0.8 * intensities)
            
            # Update the lightning window for the main path
            instate['lightning_window'][row, valid_path, 0] = hues
            instate['lightning_window'][row, valid_path, 1] = saturations
            instate['lightning_window'][row, valid_path, 2] = intensities
            instate['lightning_window'][row, valid_path, 3] = np.minimum(intensities * 1.2, 1.0)
            
            # Add glow/halo effect - slightly wider than the main path
            width = 2  # Half-width of the glow
            for dx in range(-width, width+1):
                if dx == 0:  # Skip the center as we already processed it
                    continue
                    
                # Calculate falloff based on distance from center
                intensity_factor = 0.7 * (1.0 - abs(dx)/width)
                
                # Calculate positions with offset
                glow_positions = valid_path + dx
                valid_glow = (glow_positions >= 0) & (glow_positions < 300)
                
                if np.any(valid_glow):
                    glow_path = glow_positions[valid_glow]
                    glow_intensities = intensities[valid_glow] * intensity_factor
                    
                    # Only update if the new value is brighter
                    current = instate['lightning_window'][row, glow_path, 2]
                    update_mask = glow_intensities > current
                    
                    if np.any(update_mask):
                        update_positions = glow_path[update_mask]
                        update_intensities = glow_intensities[update_mask]
                        update_hues = hues[valid_glow][update_mask]
                        update_sats = saturations[valid_glow][update_mask] + 0.2  # More saturated glow
                        
                        instate['lightning_window'][row, update_positions, 0] = update_hues
                        instate['lightning_window'][row, update_positions, 1] = update_sats
                        instate['lightning_window'][row, update_positions, 2] = update_intensities
                        instate['lightning_window'][row, update_positions, 3] = np.minimum(update_intensities, 1.0)
    
    # Apply overall fade based on elapsed time
    if elapsed > 0.7:  # Start fading after 0.7 seconds
        fade_factor = np.exp(-(elapsed - 0.7) * 1.5)  # Exponential decay
        instate['lightning_window'] *= fade_factor
    
    # Convert to RGB
    rgb = color.hsv2rgb(instate['lightning_window'][:,:,0:3])
    alpha = instate['lightning_window'][:,:,3:4]
    rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)
    
    # Update the image plane
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['lightning_plane'],
        rgb_out[:,:,:]
    )

def secondary_tree(instate, outstate):
    if instate['count'] == 0:
        # Initialize constants in instate
        depth = 10
        scale_factor = depth / 10
        base_dims = np.array([32, 300])  # 32 angle divisions, 300 radius steps
        layer_dims = (int(base_dims[0] * scale_factor), int(base_dims[1] * scale_factor))
        
        # Get current season from outstate (default to 0 if not provided)
        current_season = outstate.get('season', 0.0)
        
        # Define color palette types for different seasons - same as in forest.py
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
        spring_influence = max(0, 1 - circular_distance(current_season, spring_center) / season_width)
        palette_weights[0] = 15 * spring_influence  # Fresh light greens

        # Summer weights
        summer_influence = max(0, 1 - circular_distance(current_season, summer_center) / season_width)
        palette_weights[1] = 20 * summer_influence  # Standard greens
        palette_weights[2] = 15 * summer_influence  # Blue-greens
        palette_weights[3] = 15 * summer_influence  # Yellow-greens
        palette_weights[4] = 10 * summer_influence  # Darker forest greens

        # Fall weights
        fall_influence = max(0, 1 - circular_distance(current_season, fall_center) / season_width)
        palette_weights[5] = 15 * fall_influence  # Early autumn yellows
        palette_weights[6] = 20 * fall_influence  # Golden autumn
        palette_weights[7] = 15 * fall_influence  # Orange autumn
        palette_weights[8] = 10 * fall_influence  # Red autumn

        # Winter weights
        winter_influence = max(0, 1 - circular_distance(current_season, winter_center) / season_width)
        palette_weights[9] = 20 * winter_influence   # Evergreen winter
        palette_weights[10] = 15 * winter_influence  # Dark winter green
        palette_weights[11] = 100 * winter_influence  # Snow-dusted eve
        
        # Add a small baseline weight to avoid zero probabilities
        palette_weights = [max(1, w) for w in palette_weights]
        
        # Store the palettes and weights in instate
        instate['color_palettes'] = color_palettes
        instate['palette_weights'] = palette_weights
        
        # Pre-allocate arrays in instate
        instate['depth'] = depth
        instate['scale_factor'] = scale_factor
        instate['layer_dims'] = layer_dims
        # Add particle speed to the leaf_layer array (now storing [theta, radius, speed])
        instate['leaf_layer'] = np.zeros((0, 3), dtype=np.float32)  # [theta, radius, speed]
        instate['window'] = np.zeros((*layer_dims, 4))
        instate['trunk_height'] = 120  # Adjustable trunk height (0-300)
        instate['start_time'] = time.time()
        outstate['tree'] = True
        
        # Create image plane
        instate['tree_plane'] = outstate['render'][instate['frame_id']].create_image_plane(
            np.zeros((*layer_dims, 4), dtype=np.uint8),
            position=(0, 0, depth),
            rotation=(0, 0, 0),
            scale=(scale_factor*1.25, scale_factor)
        )
        return

    if instate['count'] == -1:
        outstate['render'][instate['frame_id']].remove_image_plane(instate['tree_plane'])
        outstate['tree'] = False
        return

    # Calculate event fade factor based on time
    current_time = time.time()
    elapsed_time = current_time - instate['start_time']
    total_duration = instate.get('duration', 30)
    fade_in_duration = total_duration * 0.1  # First 10% of event
    fade_out_start = total_duration * 0.9    # Last 10% of event

    # Calculate fade factor
    if elapsed_time < fade_in_duration:
        event_fade = elapsed_time / fade_in_duration
    elif elapsed_time > fade_out_start:
        event_fade = (total_duration - elapsed_time) / (total_duration * 0.1)
    else:
        event_fade = 1.0
    event_fade = np.clip(event_fade, 0, 1)

    # Get state variables
    growth_rate = outstate.get('tree_growth', 0.5)
    window = instate['window']
    height, width = instate['layer_dims']
    trunk_height = instate['trunk_height']
    whomp = outstate.get('whomp', 0.0)
    firefly_density = outstate.get('firefly_density', 0.0)
    if firefly_density >1.01:
        whomp=0
    # Clear previous frame
    window[:,:,2] *= 0.93
    window[:,:,3] = window[:,:,2] > 0.3

    # Generate new particles
    particle_count = np.random.binomial(2, growth_rate * 0.4)
    if particle_count > 0:
        new_particles = np.column_stack((
            np.random.uniform(0, 2*np.pi, particle_count),  # Angular position in radians
            np.random.uniform(0, 0.05, particle_count),     # Start near center
            np.random.uniform(0.25, 2, particle_count)     # Individual speed factor
        ))
        instate['leaf_layer'] = np.vstack((instate['leaf_layer'], new_particles)) if instate['leaf_layer'].size > 0 else new_particles

    if len(instate['leaf_layer']) > 0:
        # Update all particle positions vectorized
        wind = outstate.get('wind', 0)
        base_movement_speed = 0.003 * (0.1 + (growth_rate+whomp*6) * 0.3 + np.abs(wind))
        wind_effect = wind * 0.05  # Scale wind effect for angular movement
        
        # Apply individual speed factors to each particle
        individual_speeds = base_movement_speed * instate['leaf_layer'][:, 2]
        
        # Move particles radially outward and apply wind to angular position
        instate['leaf_layer'][:, 1] += individual_speeds  # Radial movement with individual speeds
        instate['leaf_layer'][:, 0] += wind_effect    # Angular movement (wind effect)
        
        # Wrap angular positions around 2π
        instate['leaf_layer'][:, 0] = instate['leaf_layer'][:, 0] % (2 * np.pi)
        
        # Filter particles vectorized
        mask = instate['leaf_layer'][:, 1] < 1.0  # Keep particles within radius
        instate['leaf_layer'] = instate['leaf_layer'][mask]
        
        # Convert coordinates vectorized
        theta_indices = ((instate['leaf_layer'][:, 0]) / (2 * np.pi) * height).astype(int) % height
        radius_indices = (instate['leaf_layer'][:, 1] * (width-1)).astype(int)
        
        # Valid particles mask
        valid_mask = (radius_indices >= 0) & (radius_indices < width)
        
        # Draw valid particles vectorized
        if np.any(valid_mask):
            valid_thetas = theta_indices[valid_mask]
            valid_radii = radius_indices[valid_mask]
            
            # Create colors using the seasonal palettes
            colors = []
            color_palettes = instate['color_palettes']
            palette_weights = instate['palette_weights']
            
            trunk_mask = valid_radii < trunk_height
            foliage_mask = ~trunk_mask

            # Pre-allocate colors array
            colors = np.zeros((len(valid_radii), 4), dtype=np.float32)

            # Handle trunk and foliage particles in batches
            if np.any(trunk_mask):
                # Select random palettes for all trunk particles at once
                trunk_indices = np.random.choice(
                    len(color_palettes), 
                    size=np.sum(trunk_mask), 
                    p=np.array(palette_weights) / sum(palette_weights)
                )
                
                # Extract trunk parameters for each selected palette
                trunk_sat_ranges = np.array([color_palettes[i]["trunk_sat_range"] for i in trunk_indices])
                trunk_val_ranges = np.array([color_palettes[i]["trunk_val_range"] for i in trunk_indices])
                
                # Generate random colors for all trunk particles at once
                colors[trunk_mask, 0] = 0.08 + np.random.random(np.sum(trunk_mask)) * 0.04  # Brown hue
                colors[trunk_mask, 1] = np.random.uniform(
                    trunk_sat_ranges[:, 0], 
                    trunk_sat_ranges[:, 1]
                )  # Saturation
                colors[trunk_mask, 2] = np.random.uniform(
                    trunk_val_ranges[:, 0], 
                    trunk_val_ranges[:, 1]
                ) + 0.5  # Value
                colors[trunk_mask, 3] = 1.0  # Alpha

            if np.any(foliage_mask):
                # Select random palettes for all foliage particles at once
                foliage_indices = np.random.choice(
                    len(color_palettes), 
                    size=np.sum(foliage_mask), 
                    p=np.array(palette_weights) / sum(palette_weights)
                )
                
                # Extract foliage parameters for each selected palette
                foliage_hue_ranges = np.array([color_palettes[i]["hue_range"] for i in foliage_indices])
                foliage_sat_ranges = np.array([color_palettes[i]["sat_range"] for i in foliage_indices])
                foliage_val_ranges = np.array([color_palettes[i]["val_range"] for i in foliage_indices])
                
                # Generate random colors for all foliage particles at once
                leaf_hues = np.random.uniform(
                    foliage_hue_ranges[:, 0], 
                    foliage_hue_ranges[:, 1]
                )
                hue_variations = np.random.uniform(-0.05, 0.05, np.sum(foliage_mask))
                
                colors[foliage_mask, 0] = leaf_hues + hue_variations  # Hue with variation
                colors[foliage_mask, 1] = np.random.uniform(
                    foliage_sat_ranges[:, 0], 
                    foliage_sat_ranges[:, 1]
                )  # Saturation
                colors[foliage_mask, 2] = np.random.uniform(
                    foliage_val_ranges[:, 0], 
                    foliage_val_ranges[:, 1]
                ) + 0.4  # Value
                colors[foliage_mask, 3] = 1.0  # Alpha

            # Apply event fade to all particles
            colors[:, 2] *= event_fade  # Adjust brightness
            colors[:, 3] *= event_fade  # Adjust alpha

            # Update window
            window[valid_thetas, valid_radii] = colors

    # Convert HSVA to RGBA for rendering
    rgb = color.hsv2rgb(window[:,:,0:3])
    alpha = window[:,:,3:4]
    rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)
    
    # Update the image plane
    outstate['render'][instate['frame_id']].update_image_plane_texture(
        instate['tree_plane'], 
        rgb_out[:,:,:]
    )