"""
Shader-based event functions that integrate with the existing EventScheduler
"""
import numpy as np
from corefunctions.shader_renderer import RainEffect

def shader_rain(state, outstate, intensity=1.0):
    """
    Shader-based rain effect compatible with EventScheduler
    
    Args:
        state: Event state dict (contains start_time, elapsed_time, etc.)
        outstate: Global state dict (from EventScheduler)
        intensity: Rain intensity multiplier
    """
    # Get the appropriate viewport from shader renderer
    frame_id = state.get('frame_id', 0)
    shader_renderer = outstate.get('shader_renderer')
    
    if shader_renderer is None:
        print("WARNING: shader_renderer not found in state!")
        return
    
    viewport = shader_renderer.get_viewport(frame_id)
    if viewport is None:
        print(f"WARNING: viewport {frame_id} not found!")
        return
    
    # Initialize rain effect on first call
    if state['count'] == 0:
        num_drops = int(100 * intensity)
        print(f"Initializing rain effect for frame {frame_id} with {num_drops} drops")
        
        try:
            rain_effect = viewport.add_effect(
                RainEffect,
                num_raindrops=num_drops
            )
            state['rain_effect'] = rain_effect
            print(f"✓ Initialized shader rain for frame {frame_id}")
        except Exception as e:
            print(f"✗ Failed to initialize rain: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Effect updates are handled automatically in viewport.update()
    # No per-frame work needed here
    
    # On close event, mark for removal
    if state['count'] == -1:
        if 'rain_effect' in state:
            print(f"Cleaning up rain effect for frame {frame_id}")
            viewport.effects.remove(state['rain_effect'])
            state['rain_effect'].cleanup()
            print(f"✓ Cleaned up shader rain for frame {frame_id}")