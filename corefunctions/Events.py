import time
import heapq
import numpy as np
import corefunctions.soundtestthreaded as sound
import corefunctions.ImageToDMX as imdmx
from corefunctions.shader_renderer import ShaderRenderer
import threading

class TimedEvent:
    def __init__(self, start_time, duration, action, args=(), kwargs={}, name=None, frame_id=None):
        self.start_time = start_time
        self.duration = duration
        self.action = action
        self.args = args
        self.kwargs = kwargs
        # Use action name as default if name not provided
        self.name = name if name is not None else (action.__name__ if hasattr(action, '__name__') else str(action))
        self.state = {}
        self.state['count'] = 0
        self.state['start_time'] = start_time
        self.state['duration'] = duration
        self.state['elapsed_time'] = 0
        self.state['elapsed_fraction'] = 0
        self.frame_duration=[]
        # Store frame_id in state if provided
        if frame_id is not None:
            self.state['frame_id'] = frame_id

    def __lt__(self, other):
        return self.start_time < other.start_time

    def update(self, outstate):
        # Use high precision timer
        start = time.perf_counter_ns()
        
        self.state['elapsed_time'] = outstate['current_time'] - self.state['start_time']
        self.state['elapsed_fraction'] = self.state['elapsed_time'] / self.state['duration']
        if self.state['elapsed_time'] > self.state['duration']:
            self.closeevent(outstate)
            return False
            
        self.action(self.state, outstate, *self.args, **self.kwargs)
        self.state['count'] += 1
        
        # Calculate duration in microseconds for higher precision
        elapsed = (time.perf_counter_ns() - start) / 1.0E9  # Convert ns to seconds
        if self.state['count']<1000:
            self.frame_duration.append(elapsed)
        return True
    
    def closeevent(self, outstate):
        median_duration = np.median(self.frame_duration) if self.frame_duration else 0
        print(f"Event closed: {self.name} Length:{median_duration:.6f}s")
        
        # Add logging to file
        # with open("log.txt", "a") as log_file:
        #     log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Event: {self.name}, Duration: {median_duration:.6f}s\n")
            
        self.state['count'] = -1
        self.action(self.state, outstate, *self.args, **self.kwargs)


class EventScheduler:
    def __init__(self, use_shader_renderer=False, headless=False,frames=[(128,300)], magnification=1):
        self.event_queue = []
        self.active_events = []
        self.state = {}
        self.should_exit = False
        self._cleaned_up = False
        

        frame_dimensions = frames
        # Determine which renderer to use
        self.use_shader_renderer = use_shader_renderer
        
        if use_shader_renderer:
            mode_str = "headless GPU" if headless else "GPU"
            print(f"Initializing {mode_str} shader renderer...")
            # Create shader renderer - window size calculated automatically
            self.shader_renderer = ShaderRenderer(
                frame_dimensions=frame_dimensions,
                headless=headless,
                magnification=magnification
            )
            
            # Create viewports for each frame
            for frame_id in range(len(frame_dimensions)):
                viewport = self.shader_renderer.create_viewport(frame_id)
                if not headless:
                    print(f"  Created viewport {frame_id}: {frame_dimensions[frame_id]}")
                
            self.state['shader_renderer'] = self.shader_renderer
            
            # Create a placeholder for legacy render compatibility
            self.state['render'] = [None] * len(frame_dimensions)
            print(f"[OK] {mode_str} shader renderer initialized")
        else:
            print("no CPU renderer...")

        
        self.state['last_time'] = time.time()
        self.state['soundengine'] = sound.ThreadedAudioEngine()
        self.state['soundengine'].start()
        self.state['current_time'] = time.time()
        self.state['wind'] = 0
        self.state['tree_frame'] = np.zeros((60, 120, 4))
        self.state['rainrate'] = 0.5
        self.state['thunderrate'] = 0.0
        self.state['starryness'] = 0.0
        self.state['simulate'] = True
        
        # Store frame dimensions for per-frame brightness calculations
        self.frame_dimensions = frame_dimensions
        
        # Brightness limiting configuration
        # Setpoint is normalized 0-1 where 1.0 = all pixels at full white (255,255,255)
        # Set to 0.25 for 25% of maximum possible brightness, etc.
        self.brightness_setpoint = 0.1  # Adjust this value (0.0 - 1.0)
        
        self.brightness_config = {
            'red_factor': 1.0,      # Brightnessfactors are based on LED per color power draws
            'green_factor': 1.0,
            'blue_factor': 1.0,
            'threshold': 0.8,         # Start limiting at 80% of setpoint
            'smoothing': 0.05          # Lower = smoother but slower response (0.05-0.2 recommended)
        }
        # Store per-frame brightness state
        self.brightness_state = [{
            'divisor': 1.0,           # Current brightness divisor
            'bright_factor': 0.0      # Last calculated brightness factor
        } for _ in frame_dimensions]
        
        total_pixels = sum(width * height for width, height in frame_dimensions)
        print(f"Brightness limiting: {len(frame_dimensions)} displays, {total_pixels} total pixels")
        
        # Performance monitoring
        self.perf_stats = {
            'event_update': [],
            'rendering': [],
            'display_send': [],
            'total_frame': [],
            'last_report_time': time.time()
        }
        
        # Define receivers for each display
        receivers = [            
            [
                {
                    'ip': '192.168.68.140',
                    'pixel_count': 300*32,
                    'addressing_array': imdmx.make_indices_V_rect_alternate(32,300,0)
                },  
                                         {
                    'ip': '192.168.68.141',
                    'pixel_count': 300*32,
                    'addressing_array': imdmx.make_indices_V_rect_alternate(32,300,32)
                },      
                                         {
                    'ip': '192.168.68.142',
                    'pixel_count': 300*32,
                    'addressing_array': imdmx.make_indices_V_rect_alternate(32,300,64)
                },
                                                {
                    'ip': '192.168.68.143',
                    'pixel_count': 300*32,
                    'addressing_array': imdmx.make_indices_V_rect_alternate(32,300,96)
                },
            ],
            # Primary display receivers (frame 0)
            [
                {
                    'ip': '192.168.68.111',
                    'pixel_count': 2019,
                    'addressing_array': imdmx.make_indicesHS(r"./DMXconfig/UnitA.txt")
                },
                {
                    'ip': '192.168.68.125',
                    'pixel_count': 1777,
                    'addressing_array': imdmx.make_indicesHS(r"./DMXconfig/UnitB.txt")
                },         
                {
                    'ip': '192.168.68.124',
                    'pixel_count': 1793,
                    'addressing_array': imdmx.make_indicesHS(r"./DMXconfig/UnitC.txt")
                }
            ],
        ]
        
        # Create pixel senders for each display
        self.state['screens'] = []
        for i in range(len(receivers)):
            if i < len(receivers):
                sender = imdmx.SACNPixelSender(receivers[i], skip_network=False, use_raw_udp=True,per_receiver_universe=True)
                # Enable async sending for better performance
                sender.enable_async_send()
                self.state['screens'].append(sender)
            else:
                # For displays without physical receivers, add None as placeholder
                self.state['screens'].append(None)
        
        print("[OK] sACN senders initialized with async mode enabled")
        

    
    def has_action(self, action):
        return any(event.action == action for event in self.event_queue) or \
               any(event.action == action for event in self.active_events)

    def schedule_event(self, delay, duration, action, *args, **kwargs):
        """Schedule an event with optional frame_id"""
        # Check if an event with the same action is already running OR queued
        if any(event.action == action for event in self.active_events) or \
           any(event.action == action for event in self.event_queue):
            # Skip scheduling this event - an instance is already running or queued
            action_name = action.__name__ if hasattr(action, '__name__') else str(action)
            print(f"Skipping duplicate event: {action_name} (already running or queued)")
            return None
        
        event_time = time.time() + delay
        
        # Extract special kwargs
        name = kwargs.pop('name', None)
        frame_id = kwargs.pop('frame_id', None)
        
        # Create event with frame_id if provided
        event = TimedEvent(event_time, duration, action, args, kwargs, name=name, frame_id=frame_id)
        heapq.heappush(self.event_queue, event)
        return event

    def schedule_frame_event(self, delay, duration, action, frame_id=0, *args, **kwargs):
        """Convenience method to schedule an event for a specific frame"""
        kwargs['frame_id'] = frame_id
        return self.schedule_event(delay, duration, action, *args, **kwargs)

    def cancel_all_events(self):
        # Run close events for all active events
        for event in self.active_events:
            event.closeevent(self.state)
        self.event_queue = []
        self.active_events = []

    def set_fog(self, frame_id, amount, color=None, dir_scale=None):
        """Convenience method to set fog parameters for a specific frame"""
        if self.use_shader_renderer:
            # TODO: Implement fog for shader renderer
            pass
        else:
            self.renderer.set_fog(frame_id, amount, color, dir_scale)

    def update(self):
        # Process OSC messages if needed
        # osc_messages = self.get_osc_messages()
        # if osc_messages != []:
        #     self.state['osc_messages'] = osc_messages

        # Poll window events if using shader renderer
        if self.use_shader_renderer:
            self.shader_renderer.poll_events()
            if self.shader_renderer.should_close():
                print("Window closed by user")
                self.cleanup()
                self.should_exit = True
                return

        self.state['current_time'] = time.time()
        
        # Process events that should start now
        while self.event_queue and self.event_queue[0].state['start_time'] <= self.state['current_time']:
            self.active_events.append(heapq.heappop(self.event_queue))
        
        # Update active events
        i = 0
        while i < len(self.active_events):
            event = self.active_events[i]
            if event.update(self.state):
                i += 1
            else:
                self.active_events.pop(i)
        
        # Calculate delta time
        dt = self.state['current_time'] - self.state['last_time']
        self.state['last_time'] = self.state['current_time']
        
        # Render based on active renderer
        if self.use_shader_renderer:
            frames = self._render_shader(dt)
        else:
            frames = self._render_legacy()
        
        # Send to physical displays
        self._send_to_displays(frames)
    
    def _render_shader(self, dt):
        """Render using shader renderer"""
        # Clear window
        self.shader_renderer.clear_window()
        
        # STEP 1: Clear and render ALL viewports first
        for viewport in self.shader_renderer.viewports:
            viewport.clear()
            viewport.update(dt, self.state)
            viewport.render(self.state)
        
        # STEP 2: CRITICAL - Ensure ALL rendering completes before reading ANY frames
        self.shader_renderer.sync_gpu()
        
        # STEP 3: Now safely read all frames
        frames = []
        for viewport in self.shader_renderer.viewports:
            frames.append(viewport.get_frame())
        
        return frames
    
    def _render_legacy(self):
        """Render using existing moderngl renderer"""
        frames = []
        for scene in self.state['render']:
            if scene is not None:
                frames.append(scene.render())
        return frames
    
    def _send_to_displays(self, frames):
        """Send frames to physical displays"""
        gamma = 2.8
        
        # Process and send frames
        for i, frame in enumerate(frames):
            # Convert RGBA to RGB (drop alpha channel) - avoid copy if already RGB
            if frame.shape[2] == 4:
                frame_rgb = frame[:, :, :3]
            else:
                frame_rgb = frame
            
            # Apply gamma correction (keep as float) - only if gamma != 1
            if gamma != 1:
                frame_corrected = np.power(frame_rgb / 255.0, gamma) * 255.0
            else:
                # Skip gamma correction if gamma is 1
                frame_corrected = frame_rgb.astype(np.float32)
            
            # Apply brightness limiting and convert to uint8
            frame_corrected = self._apply_brightness_limiting(frame_corrected, i)
            
            # Send to physical display if available
            if i < len(self.state['screens']) and self.state['screens'][i] is not None:
                try:
                    # sACN expects BGR order
                    self.state['screens'][i].send(frame_corrected[:, :, [0, 1, 2]])
                except OSError as e:
                    print(f"Network error while sending sACN data to display {i}: {e}")
        
        # Swap OpenGL buffers if using shader renderer
        if self.use_shader_renderer:
            self.shader_renderer.swap_buffers()

    def _apply_brightness_limiting(self, frame_corrected, frame_index):
        """Apply brightness limiting to prevent total brightness from exceeding setpoint.
        Uses exponential smoothing to prevent flickering.
        
        Args:
            frame_corrected: RGB frame (height, width, 3) as float (0-255 range)
            frame_index: Index of the frame being processed (for per-frame state)
            
        Returns:
            Limited frame as uint8
        """
        cfg = self.brightness_config
        state = self.brightness_state[frame_index]
        
        # Calculate maximum possible weighted brightness for this frame
        # When all pixels are full white (255,255,255), weighted sum = pixels * 255 * sum_of_weights
        height, width = frame_corrected.shape[:2]
        actual_pixels = height * width
        sum_of_weights = cfg['red_factor'] + cfg['green_factor'] + cfg['blue_factor']  # = 3.0
        max_possible_brightness = actual_pixels * 255.0 * sum_of_weights
        
        # Calculate weighted brightness factor (frame already in float)
        # Optimize: compute all channel sums in one pass
        bright_factor = (np.sum(frame_corrected[:, :, 0]) * cfg['red_factor'] + 
                        np.sum(frame_corrected[:, :, 1]) * cfg['green_factor'] + 
                        np.sum(frame_corrected[:, :, 2]) * cfg['blue_factor'])
        
        # Normalize to 0-1 range
        normalized_brightness = bright_factor / max_possible_brightness
        
        state['bright_factor'] = normalized_brightness
        
        # Calculate target divisor using normalized setpoint (0-1)
        threshold_value = self.brightness_setpoint * cfg['threshold']
        
        if normalized_brightness <= threshold_value:
            # Below threshold - no limiting needed
            target_divisor = 1.0
        else:
            # Above threshold - calculate divisor to bring brightness to setpoint
            # Use a smooth transition that gets stronger as we exceed the setpoint
            target_divisor = normalized_brightness / self.brightness_setpoint
            
            # Ensure divisor is at least 1.0
            target_divisor = max(1.0, target_divisor)
        
        # Smooth the divisor using exponential moving average to prevent flickering
        # Higher smoothing = slower response but smoother transitions
        alpha = cfg['smoothing']
        state['divisor'] = (alpha * target_divisor) + ((1 - alpha) * state['divisor'])
        
        # Apply the divisor if it's significantly above 1.0
        if state['divisor'] > 1.001:
            frame_corrected = frame_corrected / state['divisor']
            print(f"Applying brightness divisor: {state['divisor']}")
        
        # Convert to uint8 once at the end
        return np.clip(frame_corrected, 0, 255).astype(np.uint8)

    def cleanup(self):
        """Clean up all resources"""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        print("Cleaning up EventScheduler...")

        # Cancel all events
        self.cancel_all_events()

        # Clean up renderer
        if self.use_shader_renderer:
            self.shader_renderer.cleanup()

        # Clean up sound engine
        if hasattr(self.state.get('soundengine'), 'stop'):
            self.state['soundengine'].stop()

        print("[OK] Cleanup complete")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass
