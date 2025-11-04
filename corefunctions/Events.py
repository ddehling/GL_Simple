import time
import heapq
import numpy as np
import corefunctions.soundtestthreaded as sound
import corefunctions.ImageToDMX as imdmx
from corefunctions.shader_renderer import ShaderRenderer
import threading
import queue
import socket

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
    def __init__(self, use_shader_renderer=False, headless=False,frames=[(100,100)]):
        self.event_queue = []
        self.active_events = []
        self.state = {}
        

        frame_dimensions = frames
        # Determine which renderer to use
        self.use_shader_renderer = use_shader_renderer
        
        if use_shader_renderer:
            mode_str = "headless GPU" if headless else "GPU"
            print(f"Initializing {mode_str} shader renderer...")
            # Create shader renderer - window size calculated automatically
            self.shader_renderer = ShaderRenderer(
                frame_dimensions=frame_dimensions,
                
                headless=headless
            )
            
            # Create viewports for each frame
            for frame_id in range(len(frame_dimensions)):
                viewport = self.shader_renderer.create_viewport(frame_id)
                if not headless:
                    print(f"  Created viewport {frame_id}: {frame_dimensions[frame_id]}")
                
            self.state['shader_renderer'] = self.shader_renderer
            
            # Create a placeholder for legacy render compatibility
            self.state['render'] = [None] * len(frame_dimensions)
            print(f"✓ {mode_str} shader renderer initialized")
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
        
        # Define receivers for each display
        receivers = [
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
            # Secondary display receivers (frame 1)
            [
                {
                    'ip': '192.168.68.130',
                    'pixel_count': 2160,
                    'addressing_array': imdmx.make_indicesHS(r"./DMXconfig/UnitD.txt")
                },
                {
                    'ip': '192.168.68.131',
                    'pixel_count': 2040,
                    'addressing_array': imdmx.make_indicesHS(r"./DMXconfig/UnitE.txt")
                },
                {
                    'ip': '192.168.68.132',
                    'pixel_count': 2520,
                    'addressing_array': imdmx.make_indicesHS(r"./DMXconfig/UnitF.txt")
                }
            ]
        ]
        
        # Create pixel senders for each display
        self.state['screens'] = []
        for i in range(len(receivers)):
            if i < len(receivers):
                self.state['screens'].append(imdmx.SACNPixelSender(receivers[i]))
            else:
                # For displays without physical receivers, add None as placeholder
                self.state['screens'].append(None)
        

    
    def has_action(self, action):
        return any(event.action == action for event in self.event_queue) or \
               any(event.action == action for event in self.active_events)

    def schedule_event(self, delay, duration, action, *args, **kwargs):
        """Schedule an event with optional frame_id"""
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
                import sys
                sys.exit(0)

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
            # Convert RGBA to RGB (drop alpha channel)
            if frame.shape[2] == 4:
                frame_rgb = frame[:, :, :3]
            else:
                frame_rgb = frame
            
            #frame = self._generate_test_pattern(frame.shape[0], frame.shape[1], pattern_type=0)/4

            # Apply gamma correction
            frame_corrected = np.power(frame_rgb / 255.0, gamma) * 255.0
            frame_corrected = frame_corrected.astype(np.uint8)
            
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

    def _generate_test_pattern(self, height, width, pattern_type=0):
        """
        Generate test patterns for debugging
        
        Args:
            height: Frame height in pixels
            width: Frame width in pixels
            pattern_type: Which pattern to generate
                0: Color bars (RGBCMYW)
                1: Gradient (left to right)
                2: Checkerboard
                3: Solid red
                4: Solid green
                5: Solid blue
                6: Solid white
        """
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        if pattern_type == 0:  # Color bars
            bar_width = width // 7
            # Red, Green, Blue, Cyan, Magenta, Yellow, White
            colors = [
                [255, 0, 0],    # Red
                [0, 255, 0],    # Green
                [0, 0, 255],    # Blue
                [0, 255, 255],  # Cyan
                [255, 0, 255],  # Magenta
                [255, 255, 0],  # Yellow
                [255, 255, 255] # White
            ]
            for i, color in enumerate(colors):
                x_start = i * bar_width
                x_end = min((i + 1) * bar_width, width)
                frame[:, x_start:x_end] = color
                
        elif pattern_type == 1:  # Horizontal gradient (left to right)
            gradient = np.linspace(0, 255, width, dtype=np.uint8)
            frame[:, :, 0] = gradient  # Red channel
            frame[:, :, 1] = gradient  # Green channel
            frame[:, :, 2] = gradient  # Blue channel
            
        elif pattern_type == 2:  # Checkerboard
            checker_size = 8
            for y in range(height):
                for x in range(width):
                    if ((x // checker_size) + (y // checker_size)) % 2 == 0:
                        frame[y, x] = [255, 255, 255]
                        
        elif pattern_type == 3:  # Solid red
            frame[:, :] = [255, 0, 0]
            
        elif pattern_type == 4:  # Solid green
            frame[:, :] = [0, 255, 0]
            
        elif pattern_type == 5:  # Solid blue
            frame[:, :] = [0, 0, 255]
            
        elif pattern_type == 6:  # Solid white
            frame[:, :] = [255, 255, 255]
            
        else:  # Default: diagonal gradient
            for y in range(height):
                for x in range(width):
                    value = int((x + y) / (width + height) * 255)
                    frame[y, x] = [value, value, value]
        
        return frame



    def cleanup(self):
        """Clean up all resources"""
        print("Cleaning up EventScheduler...")
        
        # Cancel all events
        self.cancel_all_events()
        
        # Clean up renderer
        if self.use_shader_renderer:
            self.shader_renderer.cleanup()
        
        # Clean up sound engine
        if hasattr(self.state.get('soundengine'), 'stop'):
            self.state['soundengine'].stop()
        
        print("✓ Cleanup complete")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass