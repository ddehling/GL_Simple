import time
import heapq
import numpy as np
import cv2
import corefunctions.soundtestthreaded as sound
import corefunctions.ImageToDMX as imdmx
import corefunctions.newrender as sr
#from pythonosc.osc_server import ThreadingOSCUDPServer
#from pythonosc.dispatcher import Dispatcher
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

        elapsed = (time.perf_counter_ns() - start) / 1.0E9  # Convert ns to μs
        if self.state['count']<1000:
            self.frame_duration.append(elapsed)
        return True
    
    def closeevent(self, outstate):
        median_duration = np.median(self.frame_duration)
        print(f"Event closed: {self.name} Length:{median_duration}")
        
        # Add logging to file
        # with open("log.txt", "a") as log_file:
        #     log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Event: {self.name}, Duration: {median_duration:.6f}s\n")
            
        self.state['count'] = -1
        self.action(self.state, outstate, *self.args, **self.kwargs)

        #print(np.median(self.frame_duration))


class EventScheduler:
    def __init__(self):
        self.event_queue = []
        self.active_events = []
        self.state = {}
        
        # Define dimensions for multiple frames
        frame_dimensions = [
            (120, 60),   # Frame 0 (primary/main display)
            (300, 32),   # Frame 1 (secondary display)
        ]
        
        # Initialize renderer with multiple frames
        self.renderer = sr.ImageRenderer(
            frame_dimensions=frame_dimensions,
            enable_lighting=True
        )
        
        # Create array of scenes for each frame
        self.state['render'] = []
        for frame_id in range(len(frame_dimensions)):
            self.state['render'].append(sr.Scene(self.renderer, frame_id=frame_id))
        
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
        #self.state['osc_messages'] = []
        
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
        
        # Initialize OSC server and message queue
        # self.osc_messages = queue.Queue(maxsize=1000)
        # self.dispatcher = Dispatcher()
        # self.dispatcher.set_default_handler(self._handle_osc)
        
        # # Start OSC server on port 5005
        # self.osc_server = ThreadingOSCUDPServer(("0.0.0.0", 5005), self.dispatcher)
        # # Set socket options after creation
        # self.osc_server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # self.osc_server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        
        # self.osc_thread = threading.Thread(target=self._run_osc_server)
        # self.osc_thread.daemon = True
        # self.osc_thread.start()

    # def _handle_osc(self, address, *args):
    #     """Default handler for all OSC messages"""
    #     try:
    #         self.osc_messages.put_nowait((address, args))
    #     except queue.Full:
    #         print("Warning: OSC message queue full, dropping message")

    # def _run_osc_server(self):
    #     """Run the OSC server in a separate thread"""
    #     print("OSC server starting on port 5005")
    #     try:
    #         self.osc_server.serve_forever()
    #     except Exception as e:
    #         print(f"OSC server error: {e}")

    # def get_osc_messages(self):
    #     """Get all OSC messages received since last call"""
    #     messages = []
    #     try:
    #         while True:
    #             messages.append(self.osc_messages.get_nowait())
    #     except queue.Empty:
    #         pass
    #     return messages
    
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
        self.renderer.set_fog(frame_id, amount, color, dir_scale)

    def update(self):
        # Process OSC messages if needed
        # osc_messages = self.get_osc_messages()
        # if osc_messages != []:
        #     self.state['osc_messages'] = osc_messages

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
        
        self.state['last_time'] = self.state['current_time']
        
        # Render all frames
        frames = []
        for scene in self.state['render']:
            frames.append(scene.render())
        
        # Process and send frames to physical displays
        gamma = 2.8
        display_frames = []  # Store processed frames for simulation
        
        for i, frame in enumerate(frames):
            # Convert from RGBA to BGR for OpenCV and sACN
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # Store original frame for simulation
            if self.state['simulate']:
                display_frames.append(frame_bgr.copy())
            
            # Apply gamma correction
            frame_bgr = np.power(frame_bgr / 255.0, gamma) * 255.0
            frame_bgr = frame_bgr.astype(np.uint8)
            
            # Send to physical display if a screen sender exists
            if i < len(self.state['screens']) and self.state['screens'][i] is not None:
                try:
                    # Send with RGB channels swapped to match the hardware expectation
                    self.state['screens'][i].send(frame_bgr[:, :, [2, 1, 0]])
                except OSError as e:
                    print(f"Network error while sending sACN data to display {i}: {e}")

        # Display frames in simulation mode
        if self.state['simulate']:
            window_names = ["Frame 0", "Frame 1", "Frame 2", "Frame 3", "Frame 4"]  # Add more if needed
            window_sizes = [(900, 450), (900, 96), (400, 400), (400, 400), (400, 400)]  # Default sizes
            
            for i, frame in enumerate(display_frames):
                if i < len(window_names):
                    win_name = window_names[i]
                else:
                    win_name = f"Frame {i}"
                    
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                
                if i < len(window_sizes):
                    cv2.resizeWindow(win_name, *window_sizes[i])
                    
                cv2.imshow(win_name, frame)
                
            cv2.waitKey(1)