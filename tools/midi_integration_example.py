# Standalone tool: example MIDI controller integration for Stories_OGL (not part of the main app).
"""
Example integration of Korg nanoKontrol2 with Stories_OGL.py
Shows how to use MIDI controller to control shader effects and weather
"""

import time
from pathlib import Path
from lib.midi_controller import KorgNanoKontrol2
from corefunctions.Events import EventScheduler
from corefunctions import shader_effects as fx


def integrate_midi_controller():
    """
    Example showing how to integrate MIDI controller with your shader system
    """

    # Initialize MIDI controller
    midi = KorgNanoKontrol2(auto_connect=True)

    if not midi.input_device:
        print("No MIDI controller found, continuing without MIDI support")
        return None

    # Start background reading thread
    midi.start_reading()

    return midi


def setup_midi_callbacks(midi_controller, scheduler, env_system):
    """
    Setup callbacks to map MIDI controls to shader parameters

    Example mappings:
    - Slider 1-8: Control different shader intensities
    - Knob 1-8: Control shader parameters
    - S Buttons: Toggle effects on/off
    - M Buttons: Mute/unmute layers
    - R Buttons: Record triggers/snapshots
    - Transport: Weather changes
    """

    # Example: Use knob 1 to control fog density
    def on_knob1(name, value):
        scheduler.state['fog_strength'] = value
        print(f"Fog strength: {value:.2f}")

    midi_controller.register_callback('knob_1', on_knob1)

    # Example: Use slider 1 to control firefly density
    def on_slider1(name, value):
        scheduler.state['firefly_density'] = value * 3.0  # Scale to 0-3
        print(f"Firefly density: {value * 3.0:.2f}")

    midi_controller.register_callback('slider_1', on_slider1)

    # Example: S button 1 toggles rain
    def on_s1(name, value):
        if value:  # Button pressed
            scheduler.state['rain_enabled'] = not scheduler.state.get('rain_enabled', True)
            print(f"Rain: {'ON' if scheduler.state['rain_enabled'] else 'OFF'}")

    midi_controller.register_callback('s_button_1', on_s1)

    # Example: Play button triggers special effect
    def on_play(name, value):
        if value:
            print("Play pressed - triggering lightning!")
            scheduler.schedule_event(0, 5, fx.shader_lightning, frame_id=0)

    midi_controller.register_callback('play', on_play)

    print("MIDI callbacks configured")


def midi_update_loop(midi_controller, scheduler):
    """
    Main update loop - call this every frame to get MIDI changes

    Returns:
        Dictionary of controls that changed this frame
    """
    # Get all changes since last frame
    changes = midi_controller.update()

    # Store current MIDI state in scheduler state for easy access
    scheduler.state['midi'] = midi_controller.get_all_values()

    return changes


# Example usage in your main loop:
if __name__ == "__main__":
    """
    Standalone test - demonstrates MIDI controller integration
    """

    print("MIDI Controller Integration Test")
    print("="*60)

    # Setup MIDI controller
    midi = KorgNanoKontrol2(auto_connect=True)

    if not midi.input_device:
        print("No MIDI controller found!")
        exit(1)

    # Start reading
    midi.start_reading()

    # Setup some example callbacks
    def on_any_knob_change(name, value):
        print(f"{name} -> {value:.3f}")

    for i in range(1, 9):
        midi.register_callback(f'knob_{i}', on_any_knob_change)
        midi.register_callback(f'slider_{i}', on_any_knob_change)

    def on_any_button(name, value):
        if value:
            print(f"{name} pressed!")

    for i in range(1, 9):
        midi.register_callback(f's_button_{i}', on_any_button)
        midi.register_callback(f'm_button_{i}', on_any_button)
        midi.register_callback(f'r_button_{i}', on_any_button)

    for transport_name in midi.TRANSPORT.keys():
        midi.register_callback(transport_name, on_any_button)

    print("\nMIDI controller ready!")
    print("Move controls to see updates...")
    print("Press Ctrl+C to exit\n")

    try:
        last_state_print = time.time()

        while True:
            # This is what you'd call in your main render loop
            changes = midi.update()

            # Print full state every 5 seconds
            if time.time() - last_state_print > 5.0:
                print("\n" + "="*60)
                print("Current MIDI State:")
                print("="*60)

                # Print knobs
                knobs = [midi.get_knob(i) for i in range(1, 9)]
                print(f"Knobs:   {' '.join([f'{v:.2f}' for v in knobs])}")

                # Print sliders
                sliders = [midi.get_slider(i) for i in range(1, 9)]
                print(f"Sliders: {' '.join([f'{v:.2f}' for v in sliders])}")

                print("="*60 + "\n")
                last_state_print = time.time()

            time.sleep(0.01)  # ~100 FPS

    except KeyboardInterrupt:
        print("\nExiting...")
        midi.stop_reading()
        midi.disconnect()
