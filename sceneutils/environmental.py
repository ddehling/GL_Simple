import time
import numpy as np
from skimage import color
from pathlib import Path
import math 

ParentPath = Path(__file__).parent.parent
media_path = ParentPath / 'media'
sound_path = media_path / 'sounds'
print(media_path)


class CelestialBody:
    def __init__(self, 
                 size: float,           
                 roughness: float,      
                 orbital_speed: float,  
                 color_h: float,        
                 color_s: float,        
                 color_v: float,        
                 tilt: float = 0,      # INPUT: Angle from vertical in DEGREES
                 shift: float = 0,      # INPUT: Azimuthal angle in DEGREES
                 glow_factor: float = 0.2,         
                 corona_size: float = 1.5,         
                 name: str = "unnamed",             
                 distance: float = 1
                 ):
        self.size = size
        self.roughness = roughness
        # Convert orbital speed to radians per second
        self.orbital_speed = orbital_speed * (2 * math.pi) / 600
        self.color_h = color_h
        self.color_s = color_s
        self.color_v = color_v
        
        # Store original degree values for reference
        self.tilt_deg = tilt
        self.shift_deg = shift
        
        # Convert input degrees to radians for calculations
        self.tilt = math.radians(tilt)  # Convert to radians from horizontal
        self.shift = math.radians(shift)      # Convert to radians
        
        self.glow_factor = glow_factor
        self.corona_size = corona_size
        self.name = name
        self.distance = distance
        self.last_update = time.time()
        self.angle = np.random.random()*2 * math.pi  # Internal angle in RADIANS random start
        
        # FOV constants (in DEGREES)
        self.FOV_H_MIN = -180  # Expanded to full azimuth range
        self.FOV_H_MAX = 180
        self.FOV_V_MIN = 0
        self.FOV_V_MAX = 90   # Expanded to zenith
        self.PRIMARY_H_MIN = -18
        self.PRIMARY_H_MAX = 18
        self.PRIMARY_V_MIN = 0
        self.PRIMARY_V_MAX = 18    
        # Additional Secondary Display FOV
        self.FOV_SECONDARY_V_MIN = 45  # Secondary display shows 45° to 90°
        self.FOV_SECONDARY_V_MAX = 90

    def update(self, current_time: float,whomp:float):
        """Update orbital position (works in RADIANS)"""
        speed=0.25
        #speed=2.5
        delta_time = current_time - self.last_update
        self.angle += self.orbital_speed * delta_time*speed#*(1+whomp*3)
        self.angle %= 2 * math.pi  # Keep angle in [0, 2π]
        self.last_update = current_time

    def get_true_position(self) -> tuple:
        """Calculate position in horizontal coordinates
        Internal calculations in RADIANS, converts to DEGREES for FOV check"""
        # Start with position in vertical orbital plane (XZ plane)
        # self.angle is in RADIANS
        orbit_y = math.sin(self.angle)   # Horizontal component
        orbit_z = math.cos(self.angle)   # Vertical component
        orbit_x = 0  # No horizontal offset

        # Apply tilt (rotation around Y axis)
        # self.tilt is in RADIANS
        tilted_x = orbit_x * math.cos(self.tilt) - orbit_z * math.sin(self.tilt)
        tilted_y = orbit_y
        tilted_z = orbit_x * math.sin(self.tilt) + orbit_z * math.cos(self.tilt)

        # Apply shift (rotation around Z axis)
        # self.shift is in RADIANS
        final_x = tilted_x * math.cos(self.shift) - tilted_y * math.sin(self.shift)
        final_y = tilted_x * math.sin(self.shift) + tilted_y * math.cos(self.shift)
        final_z = tilted_z

        # Convert to spherical coordinates
        # Convert to DEGREES for FOV comparison
        azimuth = math.degrees(math.atan2(final_x, final_y))
        # Normalize azimuth to [-180, 180]
        if azimuth < 0:
            azimuth += 360
        if azimuth > 180:
            azimuth -= 360
            
        # Elevation
        r_xy = math.sqrt(final_x**2 + final_y**2)
        elevation = math.degrees(math.atan2(final_z, r_xy))

        # Return raw position for both displays to use
        return (azimuth, elevation)

    def get_position(self) -> tuple:
        """Get position and visibility info for primary display
        Returns (viewport_x, viewport_y, is_fully_visible) or None if completely outside FOV"""
        pos = self.get_true_position()
        if pos is None:
            return None
            
        azimuth, elevation = pos
        
        # Calculate apparent radius in degrees
        apparent_radius = (self.size * self.corona_size) / 2
        
        # Check if body is completely outside the viewable area
        if (azimuth + apparent_radius < self.PRIMARY_H_MIN or 
            azimuth - apparent_radius > self.PRIMARY_H_MAX or
            elevation + apparent_radius < self.PRIMARY_V_MIN or 
            elevation - apparent_radius > self.PRIMARY_V_MAX):
            return None
            
        # Map to viewport coordinates
        # For azimuth: map [-18, 18] to [0, 120] with 0° at center (60)
        viewport_x = 60 + (azimuth / 18) * 60
        
        # For elevation: map [0, 18] to [60, 0] (inverted Y axis)
        viewport_y = 60-(elevation / 18) * 60
        
        # Check if body is fully within the viewable area
        is_fully_visible = (
            self.PRIMARY_H_MIN <= azimuth - apparent_radius and 
            azimuth + apparent_radius <= self.PRIMARY_H_MAX and
            self.PRIMARY_V_MIN <= elevation - apparent_radius and 
            elevation + apparent_radius <= self.PRIMARY_V_MAX
        )
        
        return (viewport_x, viewport_y, is_fully_visible)
    


def multilayer_world(instate, outstate,):
    # Initialize on first run
    if instate['count'] == 0:
        frame_id=instate['frame_id']
        depths = np.array([10, 12, 17, 22])
        scale_factors = (depths) / 10
        base_dims = np.array([60, 120])
        layer_dims = [(int(base_dims[0] * scale), int(base_dims[1] * scale)) for scale in scale_factors]
        #path=sound_path / 'Rain Into Puddle.wav'
        #outstate['soundengine'].schedule_event(path, time.time(), 20,repeat_interval=19,inname='toot')
        # Pre-allocate arrays in instate
        instate['depths'] = depths
        instate['scale_factors'] = scale_factors
        instate['layer_dims'] = layer_dims
        instate['rain_layers'] = [np.zeros((0, 2), dtype=np.float32) for _ in range(5)]
        instate['windows'] = [np.zeros((dims[0], dims[1], 4)) for dims in layer_dims]
        instate['img_planes'] = []
        instate['sky_planes'] = []
        instate['stars'] = []  # Will store (x, y, brightness, color_h, color_s) for each star
        instate['max_stars'] = 120  # Maximum number of stars in the sky
        instate['fireflies'] = [
        np.array([], dtype=[
                ('x', 'f4'), 
                ('y', 'f4'), 
                ('phase', 'f4'), 
                ('speed', 'f4'),
                ('lifetime', 'f4')
            ]) 
            for _ in range(4)
        ]  # x, y, phase, speed, lifetime
        for _ in range(instate['max_stars']):
            x = np.random.uniform(0, 119)
            y = np.random.uniform(0, 59)
            brightness = 0.25 + np.random.random() * 0.75
            color_h = np.random.random()  # Hue
            color_s = 0.5 + np.random.random() * 0.2  # Saturation
            speed=0.5+np.random.random()
            instate['stars'].append([x, y, brightness, color_h, color_s,speed])
        
        instate['sky'] = np.zeros((60, 120,4)) 
        # Create image planes
        instate['sky_planes'] = outstate['render'][frame_id].create_image_plane(
                np.zeros((60, 120,4), dtype=np.uint8) ,
                position=(0, 0, 50),
                rotation=(0, 0, 0),
                scale=(9, 9)
        )
        
        for depth, scale, dims in zip(depths, scale_factors, layer_dims):
            img_plane = outstate['render'][frame_id].create_image_plane(
                np.zeros((*dims, 4), dtype=np.uint8),
                position=(0, 0, depth),
                rotation=(0, 0, 0),
                scale=(scale, scale)
            )
            instate['img_planes'].append(img_plane)
        return

    if instate['count'] == -1:
        outstate['render'][frame_id].remove_image_plane(instate['sky_planes'])
        for img_plane in instate['img_planes']:
            outstate['render'][frame_id].remove_image_plane(img_plane)
        return

    # Get state variables
    frame_id=instate['frame_id']
    rainrate = outstate.get('rainrate', 0.5)
    wind = outstate.get('wind', 0)
    windows = instate['windows']
    rain_layers = instate['rain_layers']
    scale_factors = instate['scale_factors']
    layer_dims = instate['layer_dims']
    firefly_density = outstate.get('firefly_density', 0.0)
    whomp=outstate.get('whomp',0.0)
    if firefly_density > 0:
        if firefly_density >1:
            whomp=0
        for layer_idx in range(4):
            height, width = layer_dims[layer_idx]
            
            # Add new fireflies - now as structured arrays with position, phase, speed, and lifetime
            if np.random.random() < firefly_density * 0.025:
                new_firefly = np.array([
                    (np.random.rand() * width,     # x
                     np.random.rand() * height,    # y
                     np.random.rand() * 2 * np.pi, # phase
                     0.1 + np.random.rand() * 0.2, # speed
                     1.0)                          # lifetime
                ], dtype=[('x', 'f4'), ('y', 'f4'), ('phase', 'f4'), ('speed', 'f4'), ('lifetime', 'f4')])
                
                instate['fireflies'][layer_idx] = np.concatenate((instate['fireflies'][layer_idx], new_firefly)) \
                    if instate['fireflies'][layer_idx].size > 0 else new_firefly
            #
            if instate['fireflies'][layer_idx].size > 0:
                # Update firefly positions and properties first
                fireflies = instate['fireflies'][layer_idx]
                
                # Update phases
                fireflies['phase'] += 0.1
                
                # Move in smooth random directions using phase
                angle = fireflies['phase'] * 0.1
                fireflies['x'] += np.cos(angle) * fireflies['speed']*(1+whomp*12)
                fireflies['y'] += np.sin(angle) * fireflies['speed']*(1+whomp*12)
                
                # Wrap around screen edges
                fireflies['x'] %= width
                fireflies['y'] %= height
                
                # Calculate brightness based on phase
                brightness = 0.8 + 0.2 * np.sin(fireflies['phase'])
                
                # Slowly decrease lifetime
                fireflies['lifetime'] -= 0.001
                
                # Remove dead fireflies
                mask = fireflies['lifetime'] > 0
                instate['fireflies'][layer_idx] = fireflies[mask]
                brightness = brightness[mask]
                
                # If we have no fireflies left after removing dead ones, skip rendering
                if instate['fireflies'][layer_idx].size == 0:
                    continue
                
                # Now render the fireflies that remain
                fireflies = instate['fireflies'][layer_idx]  # Get updated firefly array
                x_ints = fireflies['x'].astype(np.int32)
                y_ints = fireflies['y'].astype(np.int32)
                
                # Filter out fireflies outside valid range immediately
                valid_mask = (x_ints >= 0) & (x_ints < width) & (y_ints >= 0) & (y_ints < height)
                if not np.any(valid_mask):
                    continue
                    
                # Extract only valid fireflies
                x_ints = x_ints[valid_mask]
                y_ints = y_ints[valid_mask]
                lifetimes = fireflies['lifetime'][valid_mask]
                valid_brightness = brightness[valid_mask]
                
                # Pre-generate random hue variations for all fireflies at once
                random_hues = 0.1 + np.random.random(len(x_ints)) * 0.25
                
                # Create a reusable glow kernel (distance array from center)
                glow_radius = 4
                y_k, x_k = np.ogrid[-glow_radius:glow_radius+1, -glow_radius:glow_radius+1]
                distances_kernel = np.sqrt(x_k**2 + y_k**2)
                glow_kernel = np.clip(1 - distances_kernel/glow_radius, 0, 1)  # Normalized distances
                
                # Process each valid firefly with the pre-computed kernel
                for i, (x_int, y_int) in enumerate(zip(x_ints, y_ints)):
                    # Calculate window bounds
                    y_min, y_max = max(0, y_int-glow_radius), min(height, y_int+glow_radius+1)
                    x_min, x_max = max(0, x_int-glow_radius), min(width, x_int+glow_radius+1)
                    
                    # Calculate kernel bounds
                    k_y_min = glow_radius - (y_int - y_min)
                    k_y_max = glow_radius + (y_max - y_int)
                    k_x_min = glow_radius - (x_int - x_min)
                    k_x_max = glow_radius + (x_max - x_int)
                    
                    # Extract relevant slices
                    kernel_slice = glow_kernel[k_y_min:k_y_max, k_x_min:k_x_max]
                    window_slice = windows[layer_idx][y_min:y_max, x_min:x_max]
                    
                    # Calculate intensity
                    intensity = valid_brightness[i] * lifetimes[i] * 0.8
                    
                    # Calculate glow and mask in one step
                    glow = kernel_slice * intensity
                    update_mask = glow > window_slice[..., 2]
                    
                    # Apply updates where needed
                    if np.any(update_mask):
                        window_slice[update_mask, 0] = random_hues[i]
                        window_slice[update_mask, 1] = 0.9
                        window_slice[update_mask, 2] = glow[update_mask]
                        window_slice[update_mask, 3] = 1.0

    # Update each layer using vectorized operations where possible
    for layer_idx in range(4):
        height, width = layer_dims[layer_idx]
        scale = scale_factors[layer_idx]
        
        # Fade existing drops (vectorized)
        windows[layer_idx][:,:,2] *= 0.9*(1-0*whomp/3)
        windows[layer_idx][:,:,3] = windows[layer_idx][:,:,2] > 0.3
        
        # Generate new drops
        drop_count = np.random.binomial(2, rainrate *0.4* (1.0 - layer_idx * 0.1))
        if drop_count > 0:
            new_drops = np.random.rand(drop_count, 2)
            new_drops[:, 0] *= (height - 1)
            new_drops[:, 1] *= (width - 1)
            rain_layers[layer_idx] = np.vstack((rain_layers[layer_idx], new_drops)) if rain_layers[layer_idx].size > 0 else new_drops

        if rain_layers[layer_idx].size > 0:
            # Update positions (vectorized)
            rain_layers[layer_idx][:, 0] += (0.75 + rainrate * 0.5)/2+whomp*3
            rain_layers[layer_idx][:, 1] += wind 

            # Filter out-of-bounds drops (vectorized)
            mask = (rain_layers[layer_idx][:, 0] > 0) & (rain_layers[layer_idx][:, 0] < height-1) & \
                  (rain_layers[layer_idx][:, 1] > 0) & (rain_layers[layer_idx][:, 1] < width-1)
            rain_layers[layer_idx] = rain_layers[layer_idx][mask]

            # Draw drops
            if rain_layers[layer_idx].size > 0:
                drop_coords = rain_layers[layer_idx].astype(np.int32)
                drop_size = 1
                
                # Create drop color array (vectorized)
                drop_colors = np.array([0.55 + np.random.random(len(drop_coords))*0.20+0.3*(np.random.random()-0.5)*whomp*2,
                                      np.ones(len(drop_coords)),
                                      np.ones(len(drop_coords)),
                                      np.ones(len(drop_coords))]).T
                
                # Update drop positions
                for i in range(len(drop_coords)):
                    x, y = drop_coords[i]
                    x_slice = slice(x, min(x + drop_size, height))
                    y_slice = slice(y, min(y + drop_size, width))
                    windows[layer_idx][x_slice, y_slice] = drop_colors[i]

        # Convert to RGB (vectorized)
        rgb = color.hsv2rgb(windows[layer_idx][:,:,0:3])
        alpha = windows[layer_idx][:,:,3:4]
        rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)
        outstate['render'][frame_id].update_image_plane_texture(instate['img_planes'][layer_idx], rgb_out[:,:,:])
    #dim the sky
# Inside multilayer_world function, replace the celestial body rendering section:

    #dim the sky
    instate['sky'][:,:,:] = 0
    
    # First render stars if enabled
    if outstate['starryness'] > 0:
        if 'stars' not in instate:
            instate['stars'] = []
            instate['max_stars'] = 30
        if len(instate['stars']) < instate['max_stars'] and np.random.random() < outstate['starryness'] * 0.05:
            # Start new stars from either left or right edge
            start_side = np.random.choice(['left', 'right'])
            x = 0 if start_side == 'left' else 119
            y = np.random.randint(0, 60)
            brightness = 0.25 + np.random.random() * 0.75
            color_h = np.random.random()  # Hue
            color_s = 0.5 + np.random.random() * 0.2  # Saturation
            speed = 0.5 + np.random.random()
            instate['stars'].append([x, y, brightness, color_h, color_s, speed])

        # Update existing stars
        new_stars = []
        for star in instate['stars']:
            x, y, brightness, color_h, color_s, speed = star
            
            # Move stars left to right
            x += 0.0025*(speed)  # Adjust speed as needed
            
            # Twinkle effect
            brightness += np.random.normal(0, 0.05)  # Random brightness fluctuation
            brightness = np.clip(brightness, 0.3, 1.0)  # Keep brightness in reasonable range
            
            # If star moves off screen, create new star at left edge
            if x >= 120:
                x = 0
                y = np.random.uniform(0, 59)
                brightness = 0.5 + np.random.random() * 0.5
                color_h = np.random.random()
                color_s = 0.5 + np.random.random() * 0.2
                speed = 0.5 + np.random.random()
            
            new_stars.append([x, y, brightness, color_h, color_s, speed])
                
            # Draw the star only if no celestial body has been drawn at this position
            x_int, y_int = int(x), int(y)
            if instate['sky'][y_int, x_int, 3] == 0:  # Only draw if position is empty
                instate['sky'][y_int, x_int] = np.array([color_h, color_s, np.clip(brightness * outstate['starryness'],0,1), 1])

        instate['stars'] = new_stars

    celestial_visibility = outstate.get('celestial_visibility', 1.0)
    celestial_bodies = outstate.get('celestial_bodies', [])
    
    # Sort celestial bodies by distance (farthest first)
    sorted_bodies = sorted(celestial_bodies, key=lambda x: x.distance, reverse=True)
    
    for body in sorted_bodies:
        position = body.get_position()  # Now returns (x, y, is_fully_visible) or None
        if position is None:  # Skip if completely outside FOV
            continue
            
        pos_x, pos_y, is_fully_visible = position
        
        # Calculate sub-pixel position
        screen_x = pos_x  # Keep as float for sub-pixel accuracy
        screen_y = pos_y  # Keep as float for sub-pixel accuracy
        
        # Calculate render boundaries for the celestial body
        radius = body.size / 2
        corona_radius = radius * body.corona_size
        
        # Calculate render boundaries with extra padding for sub-pixel rendering
        y_min = max(0, int(screen_y - corona_radius - 1))
        y_max = min(instate['sky'].shape[0], int(screen_y + corona_radius + 2))
        x_min = max(0, int(screen_x - corona_radius - 1))
        x_max = min(instate['sky'].shape[1], int(screen_x + corona_radius + 2))
        
        # Skip if the region is empty or invalid
        if x_max <= x_min or y_max <= y_min:
            continue
        
        # Create sub-pixel sampling grid
        y_coords, x_coords = np.mgrid[y_min:y_max, x_min:x_max]
        
        # Calculate distances to sub-pixel center
        dx = x_coords - screen_x
        dy = y_coords - screen_y
        distances = np.sqrt(dx*dx + dy*dy)
        
        # Calculate core mask with anti-aliasing
        core_mask = np.clip(radius + 0.5 - distances, 0, 1)
        corona_mask = np.clip(corona_radius + 0.5 - distances, 0, 1) * body.glow_factor
        
        # Create noise array for roughness
        noise = np.random.random(core_mask.shape) * body.roughness
        
        # Calculate intensities
        core_intensities = core_mask * (celestial_visibility + noise)
        corona_intensities = corona_mask * celestial_visibility
        
        # Combine intensities
        total_intensities = np.maximum(core_intensities, corona_intensities)
        
        # Create color arrays
        h_array = np.full_like(total_intensities, body.color_h)
        s_array = np.where(core_mask > 0,
                          body.color_s,
                          body.color_s * 0.8)  # Reduced saturation in corona
        v_array = body.color_v * total_intensities
        
        # Create mask for non-zero intensities
        update_mask = total_intensities > 0.01  # Threshold to avoid very faint pixels
        
        # Update sky buffer - always overwrite with new celestial body values where mask is true
        instate['sky'][y_min:y_max, x_min:x_max][update_mask] = np.stack(
            [h_array[update_mask], s_array[update_mask], v_array[update_mask], np.ones_like(v_array[update_mask])],
            axis=-1
        )

    # Convert final sky buffer to RGB
    rgb = color.hsv2rgb(instate['sky'][:,:,0:3])
    alpha = instate['sky'][:,:,3:4]
    rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)   
    outstate['render'][frame_id].update_image_plane_texture(instate['sky_planes'], rgb_out[:,:,:])


def test_pattern(instate, outstate):
    if instate['count'] == 0:
        # Initialize test pattern
        instate['pattern_window'] = np.zeros((60, 120, 4))  # HSVA format
        instate['pattern_plane'] = outstate['render'].create_image_plane(
            np.zeros((60, 120, 4), dtype=np.uint8),
            position=(0, 0, 10),
            rotation=(0, 0, 0),
            scale=(1, 1)
        )
        return

    if instate['count'] == -1:
        outstate['render'].remove_image_plane(instate['pattern_plane'])
        return

    # Define the pattern colors in RGB format
    colors = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [1, 1, 1],  # White
        [0, 0, 0],   # Black
        [1, 1, 0],   # Yellow
        [0, 1, 1]    # Cyan
    ]

    # Create the pattern
    pattern = np.zeros((60, 120, 3))
    
    # Assign colors to each column
    for col in range(120):
        color_index = col % 7  # Cycle through 5 colors
        pattern[:, col] = colors[color_index]

    # Add alpha channel (fully opaque)
    alpha = np.ones((60, 120, 1))
    
    # Combine RGB with alpha channel
    rgb_out = np.concatenate([pattern*255, alpha*255], axis=2).astype(np.uint8)
    
    # Update the image plane
    outstate['render'].update_image_plane_texture(
        instate['pattern_plane'],
        rgb_out[:,:,:]
    )

def secondary_multilayer_world(instate, outstate):
    # Initialize on first run
    frame_id=instate['frame_id']
    if instate['count'] == 0:
        numstars=150
        depth = 10  # Single depth for the layer
        scale_factor = depth / 10
        base_dims = np.array([32, 300])  # 40 vertical steps for angle, 180 steps for radius
        layer_dims = (int(base_dims[0] * scale_factor), int(base_dims[1] * scale_factor))

        instate['el_to_r'] = lambda el: (90 - el) / 45  # Maps 90° to r=0 (center) and 45° to r=1 (edge)
        instate['az_to_theta'] = lambda az: (az + 180) * np.pi / 180  # Maps -180° to 0 and 180° to 2π
        
        # Pre-allocate arrays in instate
        instate['depth'] = depth
        instate['scale_factor'] = scale_factor
        instate['layer_dims'] = layer_dims
        instate['rain_layer'] = np.zeros((0, 2), dtype=np.float32)  # [theta, radius]
        instate['window'] = np.zeros((*layer_dims, 4))
        instate['fireflies'] = np.zeros(0, dtype=[
            ('theta', 'f4'), ('radius', 'f4'), ('phase', 'f4'),
            ('speed', 'f4'), ('lifetime', 'f4')
        ])
        
        # Initialize stars with structured array
        instate['stars'] = np.zeros(numstars, dtype=[
            ('theta', 'f4'),      # Base angle position
            ('radius', 'f4'),     # Base radius position
            ('brightness', 'f4'),
            ('color_h', 'f4'),
            ('color_s', 'f4'),
            ('phase', 'f4'),      # Phase for twinkling
            ('twinkle_speed', 'f4'),  # Individual twinkle speed
            ('move_speed', 'f4')      # Individual movement speed
        ])
        
        # Set persistent initial positions
        instate['stars']['theta'] = np.random.uniform(0, 2 * np.pi, numstars)  # Base angle
        instate['stars']['radius'] = np.sqrt(np.random.random(numstars))       # Base radius
        instate['stars']['brightness'] = 0.25 + np.random.random(numstars) * 0.75
        instate['stars']['color_h'] = np.random.random(numstars)
        instate['stars']['color_s'] = 0.5 + np.random.random(numstars) * 0.2
        instate['stars']['phase'] = np.random.uniform(0, 2 * np.pi, numstars)
        instate['stars']['twinkle_speed'] = 2.0 + np.random.random(numstars) * 4.0
        instate['stars']['move_speed'] = 0.2 + np.random.random(numstars) * 0.3  # Slow movement speeds

        # Initialize current positions

        
        # Precompute coordinate grids for vectorized operations
        theta_grid, radius_grid = np.meshgrid(
            np.linspace(0, 2*np.pi, 32),
            np.linspace(0, 1, 300),
            indexing='ij'
        )
        instate['theta_grid'] = theta_grid
        instate['radius_grid'] = radius_grid
        
        instate['sky'] = np.zeros((32, 300, 4))
        
        # Create image planes
        instate['sky_plane'] = outstate['render'][frame_id].create_image_plane(
            np.zeros((32, 300, 4), dtype=np.uint8),
            position=(0, 0, 50),
            rotation=(0, 0, 0),
            scale=(9, 9)
        )
        
        instate['img_plane'] = outstate['render'][frame_id].create_image_plane(
            np.zeros((*layer_dims, 4), dtype=np.uint8),
            position=(0, 0, depth),
            rotation=(0, 0, 0),
            scale=(scale_factor*1.25, scale_factor)
        )
        return

    if instate['count'] == -1:
        outstate['render']['frame_id'].remove_image_plane(instate['sky_plane'])
        outstate['render']['frame_id'].remove_image_plane(instate['img_plane'])
        return

    # Get state variables
    whomp=outstate.get('whomp',0.0)
    rainrate = outstate.get('rainrate', 0.5)
    wind = outstate.get('wind', 0)
    window = instate['window']
    height, width = instate['layer_dims']
    firefly_density = outstate.get('firefly_density', 0.0)
    Full_sky=outstate.get('skyfull',True)
    sky_width = 32 if Full_sky else 64
    # Handle fireflies vectorized
    if firefly_density > 0:
        if firefly_density >1:
            whomp=0
        # Add new fireflies
        if np.random.random() < firefly_density * 0.1:
            new_firefly = np.array([(
                np.random.rand() * 2 * np.pi,
                0.8 + np.random.rand() * 0.2,
                np.random.rand() * 2 * np.pi,
                0.05 + np.random.rand() * 0.4,
                1.0
            )], dtype=instate['fireflies'].dtype)
            
            instate['fireflies'] = np.concatenate((instate['fireflies'], new_firefly))
        #
        if len(instate['fireflies']) > 0:
            # Update firefly positions
            # Update phases
            instate['fireflies']['phase'] += 0.1
            
            # Move in smooth random directions
            # angle = instate['fireflies']['phase'] * 0.1
            instate['fireflies']['radius'] -= instate['fireflies']['speed'] * 0.003 *(1+ whomp*4)
            instate['fireflies']['theta'] += np.sin(instate['fireflies']['phase']) * 0.07
            
            # Wrap around
            instate['fireflies']['theta'] %= (2 * np.pi)
            
            # Decrease lifetime
            instate['fireflies']['lifetime'] -= 0.0006
            
            # Remove dead or out of range fireflies
            mask = (instate['fireflies']['lifetime'] > 0) & (instate['fireflies']['radius'] > 0.1)
            instate['fireflies'] = instate['fireflies'][mask]
            
            # If no fireflies remain after filtering, skip rendering
            if len(instate['fireflies']) == 0:
                pass
            else:
                # Calculate brightness for all fireflies
                brightness = 0.7 + 0.3 * np.sin(instate['fireflies']['phase'])
                
                # Extract and filter valid fireflies in one step
                theta_indices = (instate['fireflies']['theta'] / (2*np.pi) * height).astype(int)
                radius_indices = (instate['fireflies']['radius'] * (width-1)).astype(int)
                
                # Create validity mask
                valid_mask = (0 <= theta_indices) & (theta_indices < height) & (0 <= radius_indices) & (radius_indices < width)
                
                # Only proceed if there are valid fireflies
                if np.any(valid_mask):
                    # Filter to valid fireflies only
                    valid_indices = np.where(valid_mask)[0]
                    valid_thetas = theta_indices[valid_mask]
                    valid_radii = radius_indices[valid_mask]
                    valid_brightness = brightness[valid_mask]
                    valid_lifetime = instate['fireflies']['lifetime'][valid_mask]
                    
                    # Pre-compute random hues for all valid fireflies at once
                    random_hues = 0.1 + np.random.random(len(valid_indices)) * 0.25
                    
                    # Create the glow kernel once
                    glow_size = 2
                    y_k, x_k = np.ogrid[-glow_size:glow_size+1, -glow_size:glow_size+1]
                    distances = np.sqrt(y_k*y_k + x_k*x_k)
                    glow_kernel = np.clip(1 - distances/glow_size, 0, 1)
                    
                    # Process all fireflies at once using a temporary buffer approach
                    # Create a buffer for each channel that will be applied to the window
                    h_buffer = np.zeros_like(window[:,:,0], dtype=float)
                    s_buffer = np.zeros_like(window[:,:,1], dtype=float)
                    v_buffer = np.zeros_like(window[:,:,2], dtype=float)
                    a_buffer = np.zeros_like(window[:,:,3], dtype=float)
                    
                    # For each valid firefly, update the buffer
                    for i, idx in enumerate(valid_indices):
                        theta_idx = valid_thetas[i]
                        r_idx = valid_radii[i]
                        
                        # Calculate affected region
                        theta_min = max(0, theta_idx - glow_size)
                        theta_max = min(height, theta_idx + glow_size + 1)
                        r_min = max(0, r_idx - glow_size)
                        r_max = min(width, r_idx + glow_size + 1)
                        
                        # Calculate slice bounds in kernel coordinates
                        k_theta_min = glow_size - (theta_idx - theta_min)
                        k_theta_max = glow_size + (theta_max - theta_idx)
                        k_r_min = glow_size - (r_idx - r_min)
                        k_r_max = glow_size + (r_max - r_idx)
                        
                        # Extract relevant part of kernel
                        kernel_slice = glow_kernel[k_theta_min:k_theta_max, k_r_min:k_r_max]
                        
                        # Calculate glow intensity
                        glow_intensity = kernel_slice * valid_brightness[i] * valid_lifetime[i]
                        
                        # Update the buffers with the maximum value at each position
                        h_slice = h_buffer[theta_min:theta_max, r_min:r_max]
                        s_slice = s_buffer[theta_min:theta_max, r_min:r_max]
                        v_slice = v_buffer[theta_min:theta_max, r_min:r_max]
                        a_slice = a_buffer[theta_min:theta_max, r_min:r_max]
                        
                        # Create a mask for where this firefly's glow is stronger than current buffer
                        update_mask = glow_intensity > v_slice
                        
                        # Update the buffer where this firefly's glow is stronger
                        if np.any(update_mask):
                            h_slice[update_mask] = random_hues[i]
                            s_slice[update_mask] = 0.9
                            v_slice[update_mask] = glow_intensity[update_mask]
                            a_slice[update_mask] = 1.0
                    
                    # Apply the buffer to the window only where v_buffer is greater than current values
                    update_mask = v_buffer > window[:,:,2]
                    if np.any(update_mask):
                        window[update_mask, 0] = h_buffer[update_mask]
                        window[update_mask, 1] = s_buffer[update_mask]
                        window[update_mask, 2] = v_buffer[update_mask]
                        window[update_mask, 3] = a_buffer[update_mask]

    # Handle rain vectorized
    window[:,:,2] *= 0.9
    window[:,:,3] = window[:,:,2] > 0.3

    # Generate new raindrops vectorized
    drop_count = np.random.binomial(2, rainrate * 0.5)
    if drop_count > 0:
        new_drops = np.column_stack((
            np.random.uniform(0, 2*np.pi, drop_count),  # Angular position in radians
            np.random.uniform(0.95, 1.0, drop_count),    # Radial position
            np.random.uniform(0.25,1.5, drop_count)  #drop speed
        ))
        instate['rain_layer'] = np.vstack((instate['rain_layer'], new_drops)) if instate['rain_layer'].size > 0 else new_drops

    if len(instate['rain_layer']) > 0:
        # Update all rain positions vectorized
        movement_speed = 0.003 * (0.5 + (rainrate+whomp*6) * 0.5)
        instate['rain_layer'][:, 1] -= movement_speed*instate['rain_layer'][:, 2]  # Move inward radially
        instate['rain_layer'][:, 0] += wind * 0.05    # Angular movement
        # Wrap angular positions around 2π
        instate['rain_layer'][:, 0] = instate['rain_layer'][:, 0] % (2 * np.pi)
        
        # Filter drops vectorized (only remove drops that get too close to center)
        mask = instate['rain_layer'][:, 1] > 0.1
        instate['rain_layer'] = instate['rain_layer'][mask]
        
        # Convert coordinates vectorized
        theta_indices = ((instate['rain_layer'][:, 0]) / (2*np.pi) * height).astype(int) % height
        radius_indices = (instate['rain_layer'][:, 1] * (width-1)).astype(int)
        
        # Valid drops mask
        valid_mask = (radius_indices >= 0) & (radius_indices < width)
        
        # Draw valid drops vectorized
        if np.any(valid_mask):
            valid_thetas = theta_indices[valid_mask]
            valid_radii = radius_indices[valid_mask]
            
            # Create rain colors vectorized
            rain_colors = np.column_stack((
                0.55 + np.random.random(np.sum(valid_mask)) * 0.20+0.3*(np.random.random()-0.5)*whomp*2,
                np.ones(np.sum(valid_mask)),
                np.ones(np.sum(valid_mask)),
                np.ones(np.sum(valid_mask))
            ))
            
            # Update window
            window[valid_thetas, valid_radii] = rain_colors

   # Clear sky
    instate['sky'].fill(0)

    # Update sky for celestial bodies
    celestial_visibility = outstate.get('celestial_visibility', 1.0)
    celestial_bodies = outstate.get('celestial_bodies', [])
    

    # Update stars vectorized
    if outstate['starryness'] > 0:
        
        
        # Update phases for twinkling
        instate['stars']['phase'] += 0.025 * instate['stars']['twinkle_speed']
        
        # Move stars (counter-clockwise rotation)
        instate['stars']['theta'] += 0.0002 * instate['stars']['move_speed']
        instate['stars']['theta'] %= (2 * np.pi)  # Wrap around
        
        # When stars complete a rotation, slightly adjust their radius
        wrap_mask = instate['stars']['theta'] < 0.0002 * instate['stars']['move_speed']
        if np.any(wrap_mask):
            # Small random adjustment to radius while maintaining distribution
            instate['stars']['radius'][wrap_mask] = np.clip(
                instate['stars']['radius'][wrap_mask] + np.random.uniform(-0.1, 0.1, np.sum(wrap_mask)),
                0.1, 0.95
            )
        
        # Calculate twinkle effect (combine multiple frequencies for more natural twinkling)
        twinkle = (np.sin(instate['stars']['phase']) * 0.5 + 
                  np.sin(instate['stars']['phase'] * 1.3) * 0.3 +
                  np.sin(instate['stars']['phase'] * 2.7) * 0.2)
        
        # Calculate current brightness with more dramatic twinkling
        current_brightness = np.clip(
            instate['stars']['brightness'] + twinkle * 0.7,
            0.1, 1.0
        )
        
        # Convert coordinates vectorized
        theta_indices = (instate['stars']['theta'] / (2 * np.pi) * 32).astype(int) % 32
        radius_indices = (instate['stars']['radius'] * 300).astype(int)
        
        # Apply stars to sky (single pixel per star)
        for idx in range(len(instate['stars'])):
            theta_idx = theta_indices[idx]
            r_idx = radius_indices[idx]
            
            if 0 <= theta_idx < 32 and 0 <= r_idx < 300:
                # Single pixel star with current brightness
                instate['sky'][theta_idx, r_idx] = [
                    instate['stars']['color_h'][idx],
                    instate['stars']['color_s'][idx],
                    current_brightness[idx] * outstate['starryness'],
                    1.0
                ]

    for body in celestial_bodies:
        position = body.get_true_position()  # Get raw position in degrees
        if position is None:  # Skip if body is not visible
            continue
            
        az, el = position[0], position[1]
        
        # Calculate the apparent radius in degrees
        apparent_radius = (body.size * body.corona_size) / 2
        
        # Calculate the upper edge elevation accounting for corona size
        edge_elevation = el + apparent_radius  # Upper edge of corona
        
        # Skip if corona is completely below 45 degrees
        if edge_elevation < 45:
            continue
            
        # Convert celestial coordinates to polar display coordinates
        r = float(instate['el_to_r'](el))  # Convert elevation to radius (90° -> 0, 45° -> 1)
        theta = float(instate['az_to_theta'](az))  # Convert azimuth to angle
        
        # Calculate the minimum allowed radius that keeps any part of the corona in view
        r_with_corona = r - (apparent_radius / 45.0)  # Convert degree radius to r-space
        
        # Skip if corona is completely outside display bounds
        if r_with_corona > 1:
            continue
            
        # Convert to pixel coordinates with sub-pixel precision
# Inside the secondary_multilayer_world function, replacing the celestial body rendering section:

        # Convert to pixel coordinates with sub-pixel precision
        theta_pos = float(((theta / (2 * np.pi)) * sky_width) % sky_width)   # 32 vertical divisions
        r_pos = float(120 + r * 179)  # Map r=[0,1] to columns [120,299]
        
        # Calculate base radius in pixels - fixed size regardless of position
        pixel_radius = float(body.size * 2)  # Base size multiplier
        
        # Calculate corona radius
        corona_pixel_radius = float(pixel_radius * body.corona_size)
        
        # Calculate render boundaries including wraparound
        theta_min = int(np.floor(theta_pos - corona_pixel_radius - 1))
        theta_max = int(np.ceil(theta_pos + corona_pixel_radius + 2))
        r_min = max(120, int(np.floor(r_pos - corona_pixel_radius)))
        r_max = min(300, int(np.ceil(r_pos + corona_pixel_radius + 2)))
        
        # Generate all theta positions we need to render, including wrapped positions
        theta_range = []
        for t in range(theta_min, theta_max):
            wrapped_t = t % sky_width
            # Only add if it's in visible range (0-31) when not using full sky
            if Full_sky or wrapped_t < 32:
                if wrapped_t not in theta_range:
                    theta_range.append(wrapped_t)
        
        # Create render region
        for t in theta_range:
            for rad in range(r_min, r_max):
                # Calculate the shortest angular distance considering wraparound
                if not Full_sky and t >= 32:
                    continue
                dt = float(min(
                    (t - theta_pos) % sky_width,
                    (theta_pos - t) % sky_width
                ))
                
                dr = float(rad - r_pos)
                
                # Scale the angular distance based on the radius
                # This is the key fix: scale dt by the radius position
                # to maintain consistent circular appearance
                radial_position = rad - 120  # Distance from center in pixels
                if radial_position > 0:
                    # Angular distance converted to equivalent arc length at this radius
                    dt_scaled = float(dt * (2 * np.pi / sky_width) * radial_position)
                else:
                    dt_scaled = dt  # Near center, minimal scaling needed
                
                # Calculate Euclidean distance in pixel space
                dist = float(np.sqrt(dr * dr + dt_scaled * dt_scaled))
                
                # Calculate sub-pixel coverage for anti-aliasing
                core_intensity = float(max(0, min(1, pixel_radius + 0.5 - dist)))
                corona_intensity = float(max(0, min(1, corona_pixel_radius + 0.5 - dist)) * body.glow_factor)
                
                # Apply intensity with sub-pixel anti-aliasing
                if dist <= pixel_radius:
                    # Core of celestial body
                    intensity = celestial_visibility * core_intensity
                    noise = np.random.random() * body.roughness
                    if intensity > instate['sky'][t, rad, 2]:
                        instate['sky'][t, rad] = [
                            body.color_h,
                            body.color_s,
                            body.color_v * min(intensity + noise, 1),
                            1.0
                        ]
                elif dist <= corona_pixel_radius:
                    # Corona/glow effect
                    intensity = corona_intensity * celestial_visibility
                    if intensity > instate['sky'][t, rad, 2]:
                        instate['sky'][t, rad] = [
                            body.color_h,
                            body.color_s * 0.8,
                            body.color_v * intensity,
                            1.0
                        ]

    # Convert sky to RGB
    rgb = color.hsv2rgb(instate['sky'][:,:,0:3])
    alpha = instate['sky'][:,:,3:4]
    rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)
    outstate['render'][frame_id].update_image_plane_texture(instate['sky_plane'], rgb_out[:,:,:])
    
    # Convert window (rain/firefly layer) to RGB
    rgb = color.hsv2rgb(window[:,:,0:3])
    alpha = window[:,:,3:4]
    rgb_out = np.concatenate([rgb*255, alpha*255], axis=2).astype(np.uint8)
    outstate['render'][frame_id].update_image_plane_texture(instate['img_plane'], rgb_out[:,:,:])
