"""
MIDI Controller Input Handler
Tracks Korg nanoKontrol2 USB MIDI controller state in real-time
"""

import pygame
import pygame.midi
import time
from typing import Dict, Optional, Callable
import threading


class KorgNanoKontrol2:
    """
    Handler for Korg nanoKontrol2 MIDI controller
    Maintains current state of all knobs, sliders, and buttons
    
    nanoKontrol2 Layout (Factory Settings, Control Mode):
    - 8 channels with:
      - 1 knob per channel (CC 16-23)
      - 1 slider per channel (CC 0-7)
      - 3 buttons per channel: S, M, R (CC 32-47, 48-63, 64-79)
    - Transport buttons:
      - Track < > (CC 58, 59)
      - Cycle (CC 46)
      - Set/Marker (CC 60, 62)
      - Rewind/Forward (CC 43, 44)
      - Stop/Play/Record (CC 42, 41, 45)
    """
    
    # Default MIDI CC mappings for nanoKontrol2 (Factory preset)
    KNOBS = list(range(16, 24))      # CC 16-23 (8 knobs)
    SLIDERS = list(range(0, 8))      # CC 0-7 (8 sliders)
    S_BUTTONS = list(range(32, 40))  # CC 32-39 (Solo buttons)
    M_BUTTONS = list(range(48, 56))  # CC 48-55 (Mute buttons)
    R_BUTTONS = list(range(64, 72))  # CC 64-71 (Record buttons)
    
    # Transport buttons
    TRANSPORT = {
        'track_prev': 58,
        'track_next': 59,
        'cycle': 46,
        'marker_set': 60,
        'marker_prev': 61,
        'marker_next': 62,
        'rewind': 43,
        'forward': 44,
        'stop': 42,
        'play': 41,
        'record': 45,
    }
    
    def __init__(self, device_name: str = "nanoKONTROL2", auto_connect: bool = True):
        """
        Initialize MIDI controller handler
        
        Args:
            device_name: Partial name to match MIDI device (case-insensitive)
            auto_connect: Automatically connect to device if found
        """
        self.device_name = device_name
        self.input_device = None
        self.device_id = None
        
        # State dictionary - current value of every control
        self.state: Dict[str, float] = {}
        
        # Previous state for change detection
        self._prev_state: Dict[str, float] = {}
        
        # Callbacks for control changes
        self.callbacks: Dict[str, Callable] = {}
        
        # Threading for continuous reading
        self.running = False
        self.thread = None
        
        # Initialize pygame.midi
        pygame.midi.init()
        
        # Initialize all controls to default values
        self._initialize_state()
        
        # Auto-connect if requested
        if auto_connect:
            self.connect()
    
    def _initialize_state(self):
        """Initialize all controls to default values (0 for knobs/sliders, False for buttons)"""
        # Knobs (0-127 range, default to 0)
        for i, cc in enumerate(self.KNOBS):
            self.state[f'knob_{i+1}'] = 0.0
            
        # Sliders (0-127 range, default to 0)
        for i, cc in enumerate(self.SLIDERS):
            self.state[f'slider_{i+1}'] = 0.0
            
        # S buttons (Solo)
        for i, cc in enumerate(self.S_BUTTONS):
            self.state[f's_button_{i+1}'] = False
            
        # M buttons (Mute)
        for i, cc in enumerate(self.M_BUTTONS):
            self.state[f'm_button_{i+1}'] = False
            
        # R buttons (Record)
        for i, cc in enumerate(self.R_BUTTONS):
            self.state[f'r_button_{i+1}'] = False
            
        # Transport buttons
        for name in self.TRANSPORT.keys():
            self.state[name] = False
            
        # Copy to previous state
        self._prev_state = self.state.copy()
    
    def list_devices(self):
        """List all available MIDI input devices"""
        print("\nAvailable MIDI Input Devices:")
        print("-" * 60)
        for i in range(pygame.midi.get_count()):
            info = pygame.midi.get_device_info(i)
            # info = (interface, name, is_input, is_output, opened)
            if info[2]:  # is_input
                print(f"ID {i}: {info[1].decode()} (Interface: {info[0].decode()})")
        print("-" * 60)
    
    def find_device(self) -> Optional[int]:
        """
        Find MIDI input device by name
        
        Returns:
            Device ID if found, None otherwise
        """
        device_name_lower = self.device_name.lower()
        
        for i in range(pygame.midi.get_count()):
            info = pygame.midi.get_device_info(i)
            if info[2]:  # is_input
                name = info[1].decode().lower()
                if device_name_lower in name:
                    print(f"[MidiController] Found device: {info[1].decode()} (ID: {i})")
                    return i
        
        return None
    
    def connect(self) -> bool:
        """
        Connect to the MIDI controller
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.input_device is not None:
            print("[MidiController] Already connected")
            return True

        self.device_id = self.find_device()

        if self.device_id is None:
            print(f"[MidiController] Device '{self.device_name}' not found")
            self.list_devices()
            return False

        try:
            self.input_device = pygame.midi.Input(self.device_id)
            print(f"[MidiController] Connected to device ID {self.device_id}")
            return True
        except Exception as e:
            print(f"[MidiController] Error connecting: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MIDI device"""
        if self.input_device:
            self.input_device.close()
            self.input_device = None
            print("[MidiController] Disconnected")
    
    def _get_control_name(self, cc: int) -> Optional[str]:
        """Map MIDI CC number to control name"""
        # Knobs
        if cc in self.KNOBS:
            return f'knob_{self.KNOBS.index(cc) + 1}'
        
        # Sliders
        if cc in self.SLIDERS:
            return f'slider_{self.SLIDERS.index(cc) + 1}'
        
        # S buttons
        if cc in self.S_BUTTONS:
            return f's_button_{self.S_BUTTONS.index(cc) + 1}'
        
        # M buttons
        if cc in self.M_BUTTONS:
            return f'm_button_{self.M_BUTTONS.index(cc) + 1}'
        
        # R buttons
        if cc in self.R_BUTTONS:
            return f'r_button_{self.R_BUTTONS.index(cc) + 1}'
        
        # Transport buttons
        for name, transport_cc in self.TRANSPORT.items():
            if cc == transport_cc:
                return name
        
        return None
    
    def _is_button(self, control_name: str) -> bool:
        """Check if a control is a button (vs knob/slider)"""
        return 'button' in control_name or control_name in self.TRANSPORT
    
    def update(self) -> Dict[str, any]:
        """
        Read and process MIDI messages, update state
        
        Returns:
            Dictionary of controls that changed this frame
        """
        if not self.input_device:
            return {}
        
        changes = {}
        
        # Read all available MIDI messages
        if self.input_device.poll():
            midi_events = self.input_device.read(100)  # Read up to 100 messages
            
            for event in midi_events:
                # event = [[status, data1, data2, data3], timestamp]
                status, cc, value, _ = event[0]
                
                # Only process Control Change messages (status 176 = CC on channel 1)
                if status == 176 or status == 144:  # CC or Note On (some buttons send Note On)
                    control_name = self._get_control_name(cc)
                    
                    if control_name:
                        # Store previous value
                        prev_value = self.state.get(control_name)
                        
                        # Update state
                        if self._is_button(control_name):
                            # Buttons: value > 0 means pressed
                            new_value = value > 0
                        else:
                            # Knobs/Sliders: normalize to 0.0-1.0
                            new_value = value / 127.0
                        
                        self.state[control_name] = new_value
                        
                        # Track changes
                        if prev_value != new_value:
                            changes[control_name] = new_value
                            
                            # Call registered callback if exists
                            if control_name in self.callbacks:
                                self.callbacks[control_name](control_name, new_value)
        
        return changes
    
    def get(self, control_name: str, normalized: bool = True) -> float:
        """
        Get current value of a control
        
        Args:
            control_name: Name of control (e.g., 'knob_1', 'slider_3', 's_button_2')
            normalized: If True, return 0.0-1.0 for knobs/sliders, bool for buttons
        
        Returns:
            Current value of control
        """
        return self.state.get(control_name, 0.0 if not self._is_button(control_name) else False)
    
    def get_knob(self, index: int, normalized: bool = True) -> float:
        """Get knob value (1-8)"""
        return self.get(f'knob_{index}', normalized)
    
    def get_slider(self, index: int, normalized: bool = True) -> float:
        """Get slider value (1-8)"""
        return self.get(f'slider_{index}', normalized)
    
    def get_button(self, button_type: str, index: int) -> bool:
        """
        Get button state (True = pressed)
        
        Args:
            button_type: 's', 'm', or 'r'
            index: 1-8
        """
        return self.get(f'{button_type}_button_{index}')
    
    def get_transport(self, button_name: str) -> bool:
        """Get transport button state"""
        return self.get(button_name)
    
    def register_callback(self, control_name: str, callback: Callable):
        """
        Register callback for when a specific control changes
        
        Args:
            control_name: Name of control
            callback: Function to call with (control_name, value) when control changes
        """
        self.callbacks[control_name] = callback
    
    def start_reading(self):
        """Start background thread to continuously read MIDI input"""
        if self.running:
            print("[MidiController] Already reading")
            return

        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        print("[MidiController] Reading thread started")
    
    def stop_reading(self):
        """Stop background thread"""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        print("[MidiController] Reading thread stopped")
    
    def _read_loop(self):
        """Background thread loop for reading MIDI"""
        while self.running:
            self.update()
            time.sleep(0.001)  # ~1000 Hz polling rate
    
    def get_all_values(self) -> Dict[str, any]:
        """Get dictionary of all current control values"""
        return self.state.copy()
    
    def print_state(self):
        """Print current state of all controls (for debugging)"""
        print("\n" + "="*60)
        print("MIDI Controller State")
        print("="*60)
        
        # Knobs
        print("\nKnobs:")
        for i in range(1, 9):
            value = self.get_knob(i)
            print(f"  Knob {i}: {value:.3f}")
        
        # Sliders
        print("\nSliders:")
        for i in range(1, 9):
            value = self.get_slider(i)
            print(f"  Slider {i}: {value:.3f}")
        
        # Buttons
        print("\nS Buttons:")
        for i in range(1, 9):
            value = self.get_button('s', i)
            print(f"  S{i}: {'ON' if value else 'OFF'}")
        
        print("\nM Buttons:")
        for i in range(1, 9):
            value = self.get_button('m', i)
            print(f"  M{i}: {'ON' if value else 'OFF'}")
        
        print("\nR Buttons:")
        for i in range(1, 9):
            value = self.get_button('r', i)
            print(f"  R{i}: {'ON' if value else 'OFF'}")
        
        print("\nTransport:")
        for name in self.TRANSPORT.keys():
            value = self.get_transport(name)
            print(f"  {name}: {'ON' if value else 'OFF'}")
        
        print("="*60)
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop_reading()
        self.disconnect()
        pygame.midi.quit()


if __name__ == "__main__":
    """Example usage and testing"""
    print("Korg nanoKontrol2 MIDI Handler Test")
    print("Move controls on your nanoKontrol2 to see updates")
    print("Press Ctrl+C to exit\n")
    
    # Create controller instance
    controller = KorgNanoKontrol2()
    
    if not controller.input_device:
        print("Failed to connect to nanoKontrol2")
        exit(1)
    
    # Example: Register callback for knob 1
    def on_knob1_change(name, value):
        print(f"Knob 1 changed: {value:.3f}")
    
    controller.register_callback('knob_1', on_knob1_change)
    
    # Start background reading
    controller.start_reading()
    
    try:
        last_print = time.time()
        while True:
            # Print state every 2 seconds
            if time.time() - last_print > 2.0:
                controller.print_state()
                last_print = time.time()
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nStopping...")
        controller.stop_reading()
        controller.disconnect()
