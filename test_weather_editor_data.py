"""
Test script to verify that weather editor data is properly configured
and pulling from the correct locations rather than being hard-coded.
"""

def test_weather_params_imports():
    """Test that all new data structures can be imported from weather_params"""
    try:
        from corefunctions.weather_params import (
            GLOBAL_PARAMETERS,
            AVAILABLE_BACKGROUND_EVENTS,
            PARAMETER_DEFINITIONS,
            DEFAULT_WEATHER_PARAMS,
            WEATHER_PRESETS,
            WEATHER_SETS
        )
        
        print("✓ Successfully imported all weather parameter data structures")
        
        # Verify GLOBAL_PARAMETERS
        assert isinstance(GLOBAL_PARAMETERS, list), "GLOBAL_PARAMETERS should be a list"
        assert len(GLOBAL_PARAMETERS) > 0, "GLOBAL_PARAMETERS should not be empty"
        print(f"✓ GLOBAL_PARAMETERS contains {len(GLOBAL_PARAMETERS)} parameters")
        
        # Verify AVAILABLE_BACKGROUND_EVENTS
        assert isinstance(AVAILABLE_BACKGROUND_EVENTS, list), "AVAILABLE_BACKGROUND_EVENTS should be a list"
        assert len(AVAILABLE_BACKGROUND_EVENTS) > 0, "AVAILABLE_BACKGROUND_EVENTS should not be empty"
        print(f"✓ AVAILABLE_BACKGROUND_EVENTS contains {len(AVAILABLE_BACKGROUND_EVENTS)} events: {AVAILABLE_BACKGROUND_EVENTS}")
        
        # Verify PARAMETER_DEFINITIONS
        assert isinstance(PARAMETER_DEFINITIONS, dict), "PARAMETER_DEFINITIONS should be a dict"
        assert len(PARAMETER_DEFINITIONS) > 0, "PARAMETER_DEFINITIONS should not be empty"
        print(f"✓ PARAMETER_DEFINITIONS contains {len(PARAMETER_DEFINITIONS)} parameter definitions")
        
        # Verify structure of parameter definitions
        for param_name, param_def in list(PARAMETER_DEFINITIONS.items())[:3]:
            assert 'type' in param_def, f"Parameter {param_name} missing 'type' field"
            assert param_def['type'] in ['number', 'text', 'array', 'array-string', 'array-number', 'event-list'], \
                f"Parameter {param_name} has invalid type: {param_def['type']}"
            print(f"  ✓ {param_name}: {param_def}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error importing weather parameters: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_web_controller_endpoint():
    """Test that web controller provides the new data"""
    try:
        from corefunctions.web_controller import WebController
        
        # Create a test controller
        controller = WebController()
        
        # Simulate getting the weather data
        with controller.app.test_client() as client:
            response = client.get('/api/weather_editor/all_data')
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            data = response.get_json()
            
            # Verify all expected fields are present
            expected_fields = [
                'weather_states',
                'default_params',
                'weather_presets',
                'weather_sets',
                'global_parameters',
                'parameter_definitions',
                'available_background_events',
                'available_sounds',
                'available_events'
            ]
            
            for field in expected_fields:
                assert field in data, f"Missing expected field: {field}"
            
            print("✓ Web controller endpoint returns all expected fields")
            
            # Verify parameter_definitions structure
            param_defs = data['parameter_definitions']
            assert isinstance(param_defs, dict), "parameter_definitions should be a dict"
            assert len(param_defs) > 0, "parameter_definitions should not be empty"
            print(f"✓ Endpoint provides {len(param_defs)} parameter definitions")
            
            # Verify background events
            bg_events = data['available_background_events']
            assert isinstance(bg_events, list), "available_background_events should be a list"
            assert len(bg_events) > 0, "available_background_events should not be empty"
            print(f"✓ Endpoint provides {len(bg_events)} background events: {bg_events}")
            
            return True
            
    except Exception as e:
        print(f"✗ Error testing web controller: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_and_regenerate():
    """Test that saving weather data properly includes new structures"""
    try:
        from corefunctions.weather_editor_utils import generate_weather_params_file
        from corefunctions.weather_params import (
            WeatherState, WEATHER_PRESETS, WEATHER_SETS, GLOBAL_PARAMETERS
        )
        
        # Convert weather states
        weather_states = [state.value for state in WeatherState]
        
        # Convert presets
        weather_presets = {}
        for state, params in WEATHER_PRESETS.items():
            state_key = state.value if hasattr(state, 'value') else str(state)
            weather_presets[state_key] = params
        
        # Generate file content
        content = generate_weather_params_file(
            weather_states,
            weather_presets,
            WEATHER_SETS,
            GLOBAL_PARAMETERS
        )
        
        # Verify the generated content includes our new structures
        assert 'AVAILABLE_BACKGROUND_EVENTS' in content, "Generated file missing AVAILABLE_BACKGROUND_EVENTS"
        assert 'PARAMETER_DEFINITIONS' in content, "Generated file missing PARAMETER_DEFINITIONS"
        assert 'GLOBAL_PARAMETERS' in content, "Generated file missing GLOBAL_PARAMETERS"
        
        print("✓ Generated file includes all required data structures")
        
        # Verify specific content
        assert 'clouds' in content, "Background events should include 'clouds'"
        assert 'wind_speed' in content, "Parameter definitions should include 'wind_speed'"
        
        print("✓ Generated file contains expected parameter and event data")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing save/regenerate: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Weather Editor Data Configuration")
    print("=" * 60)
    print()
    
    results = []
    
    print("Test 1: Weather Parameters Import")
    print("-" * 60)
    results.append(test_weather_params_imports())
    print()
    
    print("Test 2: Web Controller Endpoint")
    print("-" * 60)
    results.append(test_web_controller_endpoint())
    print()
    
    print("Test 3: Save and Regenerate")
    print("-" * 60)
    results.append(test_save_and_regenerate())
    print()
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if all(results):
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
        exit(1)
