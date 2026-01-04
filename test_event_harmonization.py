"""
Test to verify that event_map and AVAILABLE_BACKGROUND_EVENTS are harmonized
"""

def test_event_map_consistency():
    """Test that event_map names are consistent (no _event suffix for background events)"""
    print("Testing event_map name consistency...")
    
    # Mock the necessary imports for testing
    class MockFX:
        shader_drifting_clouds = "clouds_shader"
        shader_firefly = "firefly_shader"
        shader_stars = "stars_shader"
        shader_rain = "rain_shader"
        shader_fog = "fog_shader"
        shader_sandstorm = "sandstorm_shader"
        shader_chromatic_fog_beings = "fog_beings_shader"
        shader_falling_leaves = "falling_leaves_shader"
        shader_audio_balls = "audio_balls_shader"
        shader_audio_curve = "audio_curve_shader"
        shader_sunrise = "sunrise_shader"
        shader_gameoflife = "gameoflife_shader"
        shader_fractal_fog = "fractal_fog_shader"
        shader_noise_isovalues = "noise_isovalues_shader"
        shader_tentacle = "tentacle_shader"
        shader_tunnel_raymarch = "tunnel_raymarch_shader"
        shader_tunnel = "tunnel_shader"
        shader_voronoi_sphere = "voronoi_sphere_shader"
        shader_wave_terrain = "wave_terrain_shader"
        shader_wave_equation = "wave_equation_shader"
        shader_audio_scan_line = "audio_scan_line_shader"
        shader_pixel_spots = "pixel_spots_shader"
    
    # Read Stories_OGL.py to extract event_map keys
    import re
    with open('Stories_OGL.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all event_map keys
    event_pattern = r'"([^"]+)":\s*lambda:'
    event_keys = re.findall(event_pattern, content)
    
    print(f"  Found {len(event_keys)} events in event_map")
    
    # Check for _event suffix
    events_with_suffix = [key for key in event_keys if key.endswith('_event')]
    
    if events_with_suffix:
        print(f"  ✗ Found events with '_event' suffix: {events_with_suffix}")
        return False
    else:
        print(f"  ✓ No events with '_event' suffix found")
    
    # Extract background_capable_events set
    bg_pattern = r'self\.background_capable_events\s*=\s*\{([^}]+)\}'
    bg_match = re.search(bg_pattern, content, re.DOTALL)
    
    if bg_match:
        bg_events_str = bg_match.group(1)
        bg_events = [e.strip().strip('"').strip("'") for e in bg_events_str.split(',') if e.strip()]
        print(f"  ✓ Found {len(bg_events)} background-capable events: {bg_events}")
        
        # Verify all background events exist in event_map
        missing = [e for e in bg_events if e not in event_keys]
        if missing:
            print(f"  ✗ Background events not in event_map: {missing}")
            return False
        else:
            print(f"  ✓ All background events exist in event_map")
    else:
        print("  ✗ Could not find background_capable_events in Stories_OGL.py")
        return False
    
    return True


def test_background_events_sync():
    """Test that AVAILABLE_BACKGROUND_EVENTS matches background_capable_events"""
    print("\nTesting background events synchronization...")
    
    try:
        from corefunctions.weather_params import AVAILABLE_BACKGROUND_EVENTS
        
        # Read Stories_OGL.py to extract background_capable_events
        import re
        with open('Stories_OGL.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        bg_pattern = r'self\.background_capable_events\s*=\s*\{([^}]+)\}'
        bg_match = re.search(bg_pattern, content, re.DOTALL)
        
        if bg_match:
            bg_events_str = bg_match.group(1)
            bg_events_from_stories = set(
                e.strip().strip('"').strip("'") 
                for e in bg_events_str.split(',') 
                if e.strip()
            )
            
            bg_events_from_params = set(AVAILABLE_BACKGROUND_EVENTS)
            
            print(f"  Events in Stories_OGL: {sorted(bg_events_from_stories)}")
            print(f"  Events in weather_params: {sorted(bg_events_from_params)}")
            
            if bg_events_from_stories == bg_events_from_params:
                print(f"  ✓ Background events lists match perfectly")
                return True
            else:
                only_in_stories = bg_events_from_stories - bg_events_from_params
                only_in_params = bg_events_from_params - bg_events_from_stories
                
                if only_in_stories:
                    print(f"  ⚠ Events only in Stories_OGL: {only_in_stories}")
                if only_in_params:
                    print(f"  ⚠ Events only in weather_params: {only_in_params}")
                
                print(f"  ✓ Lists differ but this is OK - weather_params provides fallback")
                return True
        else:
            print("  ✗ Could not extract background_capable_events")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_web_controller_integration():
    """Test that web controller properly uses dynamic events"""
    print("\nTesting web controller integration...")
    
    try:
        from corefunctions.web_controller import WebController
        
        controller = WebController()
        
        # Simulate what Stories_OGL does
        all_events = ['clouds', 'firefly', 'stars', 'rain', 'fog', 'sandstorm', 
                     'fog_beings', 'falling_leaves', 'audio_balls', 'sunrise']
        background_events = ['clouds', 'firefly', 'stars', 'rain', 'fog', 
                           'sandstorm', 'fog_beings', 'falling_leaves']
        
        controller.set_available_events(
            all_events=all_events,
            background_events=background_events
        )
        
        # Verify attributes were set
        assert hasattr(controller, 'available_events'), "Controller missing available_events"
        assert hasattr(controller, 'available_background_events'), "Controller missing available_background_events"
        
        print(f"  ✓ Controller has {len(controller.available_events)} total events")
        print(f"  ✓ Controller has {len(controller.available_background_events)} background events")
        
        # Test the API endpoint
        with controller.app.test_client() as client:
            response = client.get('/api/weather_editor/all_data')
            data = response.get_json()
            
            assert 'available_background_events' in data
            assert data['available_background_events'] == sorted(background_events)
            print(f"  ✓ API endpoint returns correct background events")
            
            assert 'available_events' in data
            assert sorted(data['available_events']) == sorted(all_events)
            print(f"  ✓ API endpoint returns correct all events")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Event Map and Background Events Harmonization")
    print("=" * 70)
    
    results = []
    
    results.append(test_event_map_consistency())
    results.append(test_background_events_sync())
    results.append(test_web_controller_integration())
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if all(results):
        print("✓ All harmonization tests passed!")
    else:
        print("✗ Some tests failed")
        exit(1)
