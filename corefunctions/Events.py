import time
import heapq
import numpy as np
import corefunctions.soundtestthreaded as sound
import corefunctions.ImageToDMX as imdmx
import corefunctions.newrender as sr
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
        
        # Define dimensions for multiple frames
        # frame_dimensions = [
        #     (120, 60),   # Frame 0 (primary/main display)
        #     (300, 32),   # Frame 1 (secondary display)
        # ]
        frame_dimensions = frames
        # Determine which renderer to use
        self.use_shader_renderer = use_shader_renderer
        
        if use_shader_renderer:
            mode_str = "headless GPU" if headless else "GPU"
            print(f"Initializing {mode_str} shader renderer...")
            # Create shader renderer - window size calculated automatically
            self.shader_renderer = ShaderRenderer(
                frame_dimensions=frame_dimensions,
                padding=20,
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
            print("Initializing CPU renderer...")
            # Initialize renderer with multiple frames
            self.renderer = sr.ImageRenderer(
                frame_dimensions=frame_dimensions,
                enable_lighting=True
            )
            
            # Create array of scenes for each frame
            self.state['render'] = []
            for frame_id in range(len(frame_dimensions)):
                self.state['render'].append(sr.Scene(self.renderer, frame_id=frame_id))
            print("✓ CPU renderer initialized")
        
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
        
        frames = []
        
        for viewport in self.shader_renderer.viewports:
            viewport.clear()
            viewport.update(dt, self.state)
            viewport.render(self.state)
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
            
            # Apply gamma correction
            frame_corrected = np.power(frame_rgb / 255.0, gamma) * 255.0
            frame_corrected = frame_corrected.astype(np.uint8)
            
            # Send to physical display if available
            if i < len(self.state['screens']) and self.state['screens'][i] is not None:
                try:
                    # sACN expects BGR order
                    self.state['screens'][i].send(frame_corrected[:, :, [2, 1, 0]])
                except OSError as e:
                    print(f"Network error while sending sACN data to display {i}: {e}")
        
        # Swap OpenGL buffers if using shader renderer
        if self.use_shader_renderer:
            self.shader_renderer.swap_buffers()
    
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