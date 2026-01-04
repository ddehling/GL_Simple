import numpy as np
import time
from pathlib import Path
from corefunctions.Events import EventScheduler

from corefunctions.soundinput import MicrophoneAnalyzer
from corefunctions.weather_params import (
    WeatherState, DEFAULT_WEATHER_PARAMS, WEATHER_PRESETS,
    WEATHER_SETS, DEFAULT_WEATHER_SET
)
from corefunctions.shader_effects.shader_fog import ShaderFog
from corefunctions.shader_effects.celestial_bodies import (
     shader_celestial_bodies, 
     CELESTIAL_BODIES
 )
from corefunctions import shader_effects as fx
from corefunctions.web_controller import WebController

class EnvironmentalSystem:
    def __init__(self, scheduler):
        frame_dimensions = [
            (128, 300),   # Frame 0 (primary/main display)
              # Frame 1 (secondary display)
        ]
        self.scheduler = EventScheduler(
        use_shader_renderer=True,
        headless=False,frames=frame_dimensions, magnification=3
    )
        # Weather set management
        self.current_set = DEFAULT_WEATHER_SET
        self.target_set = None  # For pending set changes
        self.weather_sets = WEATHER_SETS
        
        self.current_weather = WeatherState.CLEAR
        self.target_weather = WeatherState.CLEAR
        self.transition_time = 0
        self.transition_start = 0
        self.progress = 0
        self.season = 0.0  # Initialize season value
        self.analyzer = MicrophoneAnalyzer(device_name="TONOR")
        self.analyzer.start()
        #self.specdat = np.zeros([513, 1000])
        self.scale = 1.0
        
        # Initialize web control system (set to False to disable for max performance)
        self.enable_web_control = True
        self.web_controller = None
        
        if self.enable_web_control:
            self.web_controls = {
                "current_weather_set": self.current_set,
                "available_sets": list(WEATHER_SETS.keys()),
            }
            self.web_controller = WebController(
                self.web_controls, 
                port=5000, 
                service_name="glsimple",
                admin_password="admin123"  # Change this to your desired password
            )
            self.web_controller.start(threaded=True)
        
        # Initialize celestial bodies
        self.celestial_bodies = CELESTIAL_BODIES.copy()
        # sort celestial bodies by distance, farthest first
        self.celestial_bodies.sort(key=lambda x: x.distance, reverse=True)

        # Keep track of active weather effects
        self.default_weather_params = DEFAULT_WEATHER_PARAMS.copy()
        self.weather_params = self.default_weather_params.copy()

        # Weather state parameters
        self.weather_presets = WEATHER_PRESETS
        self.scheduler.state["tree"] = False
        self.scheduler.state["skyfull"] = False
        self.scheduler.state["simulate"] = True  # Display the leds in an opencv window for visualization
        self.active_effects = {"world": None, "ambient_sound": None}
        self._prewarm_audio_cache()
        corners_frame0 = [
        (-18, 18),  # Top-left
        (18, 18),   # Top-right
        (18, 0),    # Bottom-right
        (-18, 0)    # Bottom-left
    ]
    
    # Define corners for frame 1 (upward-facing view)
        corners_frame1 = [
        (-0, 40),  # Top-left
        (180, 40),   # Top-right
        (180, 80),   # Bottom-right
        (-0, 80)   # Bottom-left
    ]
        # Event map - maps event names to shader effects and their parameters
        # This is used for both background events and on-transition events
        self.event_map = {
            "clouds": lambda: fx.shader_drifting_clouds,
            "firefly": lambda: (fx.shader_firefly, {"squish_top_width": 0.1}),
            "stars": lambda: fx.shader_stars,
            "rain": lambda: fx.shader_rain,
            "fog": lambda: (fx.shader_fog, {
                "strength": 0.0,
                "color": (0.7, 0.7, 0.8),
                "fog_near": 20.0,
                "fog_far": 80.0
            }),

            "sandstorm_event": lambda: fx.shader_sandstorm,
            "fog_beings_event": lambda: fx.shader_chromatic_fog_beings,
            "falling_leaves_event": lambda: (fx.shader_falling_leaves, {"squish_top_width": self.scale}), 
            "audio_balls": lambda:(fx.shader_audio_balls, {"squish_top_width": self.scale}),       # Position 0
            "audio_curve": lambda:    fx.shader_audio_curve,                                     # Position 1
            "sunrise": lambda:    (fx.shader_sunrise, {"squish_top_width": 1/self.scale}),         # Position 2
            "game_of_life": lambda:    fx.shader_gameoflife,                                      # Position 3
            "fractal_fog": lambda:    fx.shader_fractal_fog,                                     # Position 4
            "noise_isovalues": lambda:    fx.shader_noise_isovalues,                                 # Position 5
            "tentacle": lambda:    fx.shader_tentacle,                                        # Position 6
            "tunnel_raymarch": lambda:    fx.shader_tunnel_raymarch,                                  # Position 7
            "tunnel": lambda:    fx.shader_tunnel,                                          # Position 8
            "voronoi_sphere": lambda:    fx.shader_voronoi_sphere,                                  # Position 10
            "wave_terrain": lambda:    fx.shader_wave_terrain,                                    # Position 11
            "wave_equation": lambda:    fx.shader_wave_equation,                                   # Position 12
            "audio_scan_line": lambda:    (fx.shader_audio_scan_line, {
                    "scan_speed": 50.0,
                    "trail_length": 75,
                        "intensity_sensitivity": 2.0,
                        "width_sensitivity": 0.5,
                        "base_width": 2.0,
                        "max_width": 20.0,
                        "color_hue": 0.5}),    
            "pixel_spots": lambda:    fx.shader_pixel_spots   
        }
        
        # Pass event_map keys to web controller if enabled
        if self.enable_web_control:
            self.web_controller.set_available_events(list(self.event_map.keys()))
        
        # Initialize background events for the starting weather set
        self._initialize_weather_set_events()
        
        # Schedule world rendering events for each frame, keeping the original function names
        #self.active_effects["world"] = self.scheduler.schedule_event(0, 999999999, multilayer_world, frame_id=0) # noqa: F405
        #self.active_effects["secondary_world"] = self.scheduler.schedule_event(0, 999999999, secondary_multilayer_world, frame_id=1) # noqa: F405
        self.whompcount = 0

    def _prewarm_audio_cache(self):
        """Pre-warm the audio cache with all weather sound effects"""
        print("Pre-warming audio cache...")
        for weather_state, params in self.weather_presets.items():
            if "ambient_sound" in params:
                sound_path = Path("media") / Path("sounds") / params["ambient_sound"]
                duration = self.get_weather_params(weather_state).get("ARI", 40)
                skip_time = self.get_weather_params(weather_state).get("skiptime", 0)
                volume = self.get_weather_params(weather_state).get("Sound_volume", 0)
                try:
                    # This will automatically cache the audio in AudioCache
                    self.scheduler.state["soundengine"].load_audio(
                        sound_path, duration, skip_time, volume
                    )
                    print(f"Cached: {sound_path.name}")
                except Exception as e:
                    print(f"Failed to cache {sound_path.name}: {str(e)}")

    def get_weather_params(self, weather_state: WeatherState):
        """Get the complete set of parameters for a weather state by combining with defaults"""
        params = self.default_weather_params.copy()
        params.update(self.weather_presets[weather_state])
        return params
    
    def get_current_set_config(self):
        """Get the configuration for the current weather set"""
        return self.weather_sets[self.current_set]
    
    def get_set_states(self, set_name=None):
        """Get list of weather states in a set"""
        if set_name is None:
            set_name = self.current_set
        return [WeatherState(state) for state in self.weather_sets[set_name]["states"]]
    
    def change_weather_set(self, new_set_name: str, immediate: bool = False):
        """Request a change to a different weather set
        
        Args:
            new_set_name: Name of the weather set to change to
            immediate: If True, change immediately. If False, queue for next transition.
        """
        if new_set_name not in self.weather_sets:
            print(f"⚠️ Unknown weather set: {new_set_name}")
            return False
        
        if new_set_name == self.current_set:
            print(f"Already in set '{new_set_name}'")
            return True
        
        if immediate:
            # Apply the change immediately
            print(f"🌈 Switching weather set immediately: '{self.current_set}' → '{new_set_name}'")
            self.current_set = new_set_name
            self.target_set = None
            if self.enable_web_control:
                self.web_controls["current_weather_set"] = self.current_set
            
            # Cancel all existing events and start new background events for the set
            self._initialize_weather_set_events()
            
            # Pick a random weather from the new set
            set_states = self.get_set_states()
            new_weather = np.random.choice(set_states)
            print(f"   Starting with: {new_weather.value}")
            
            new_weather_params = self.get_weather_params(new_weather)
            t_duration = new_weather_params["transition_duration"]
            self.transition_to_weather(new_weather, transition_duration=t_duration)
        else:
            # Queue for next transition
            self.target_set = new_set_name
            print(f"🔄 Weather set change queued: '{self.current_set}' → '{new_set_name}'")
            print(f"   Will transition on next weather change...")
        
        return True
    
    def _initialize_weather_set_events(self):
        """Cancel all events and start background events for the current weather set"""
        print(f"🔄 Initializing events for weather set: '{self.current_set}'")
        
        # Cancel all active events
        self.scheduler.cancel_all_events()
        
        # Get the background events for this set
        set_config = self.get_current_set_config()
        background_events = set_config.get("background_events", [])
        
        # Schedule the permanent background events based on set configuration
        sim_forever = 10E9  # 10 billion seconds (over 300 years)
        
        # Schedule background events for this set
        for event_name in background_events:
            print(f"   📅 Scheduling background event: {event_name}")
            self._schedule_event_from_map(event_name, 0, sim_forever, frame_id=0)
        
        print(f"✓ Background events initialized for '{self.current_set}'")
    
    def _schedule_event_from_map(self, event_name: str, start_time: float, duration: float, frame_id: int = 0):
        """Schedule an event from the event map"""
        if event_name not in self.event_map:
            print(f"   ⚠️ Unknown event: {event_name}")
            return None
        
        event_config = self.event_map[event_name]()
        if isinstance(event_config, tuple):
            # Event with parameters
            effect_func, params = event_config
            return self.scheduler.schedule_event(start_time, duration, effect_func, frame_id=frame_id, **params)
        else:
            # Simple event
            return self.scheduler.schedule_event(start_time, duration, event_config, frame_id=frame_id)

    def transition_to_weather(self, new_weather: WeatherState, transition_duration: float = 10.0):
        """Start a transition to a new weather state"""
        self.target_weather = new_weather
        print(self.target_weather)
        self.transition_time = transition_duration
        self.transition_start = time.time()

        # Start new effects if needed, one offs that occur when a weather state happens
        target_params = self.get_weather_params(new_weather)
        
        # Schedule events based on on_transition_events in weather preset
        on_transition_events = target_params.get("on_transition_events", [])
        for event_config in on_transition_events:
            if isinstance(event_config, tuple) and len(event_config) >= 2:
                event_name, duration = event_config[:2]
                frame_id = event_config[2] if len(event_config) > 2 else 0
                print(f"   🎬 Transition event: {event_name} ({duration}s)")
                self._schedule_event_from_map(event_name, 0, duration, frame_id=frame_id)
            else:
                print(f"   ⚠️ Invalid on_transition_event format: {event_config}")

        # Handle ambient sound transition
        if self.active_effects["ambient_sound"]:
            # Fade out the currently playing sound
            self.scheduler.state["soundengine"].fade_out_audio(self.active_effects["ambient_sound"], 5)

        # Schedule new ambient sound
        sound_path = Path("media") / Path("sounds") / target_params["ambient_sound"]
        self.active_effects["ambient_sound"] = target_params["ambient_sound"]
        self.scheduler.state["soundengine"].schedule_event(
            sound_path,
            time.time(),
            target_params["ARI"],
            repeat_interval=target_params["ARI"],
            inname=self.active_effects["ambient_sound"],
            fade_in_duration=5.0,
            skip_time=target_params["skiptime"],
        )

    def calculate_seasonal_weight_multiplier(self, season_preference, current_season):
        """
        Calculate a weight multiplier based on how close the current season is to the preferred season.
        Returns a value between 0.5 (furthest from preferred) and 3.0 (at preferred season).
        """
        # Calculate distance between current season and preferred season
        # Since seasons are cyclical (0-1), we need to find the shortest distance
        distance = abs(current_season - season_preference)
        if distance > 0.5:
            distance = 1.0 - distance  # Take the shorter path around the cycle
        
        # Normalize distance to range [0, 1] where 0 means perfect match and 1 means opposite season
        normalized_distance = distance * 2  # Now 0 = perfect match, 1 = opposite season
        
        # Calculate multiplier that varies from 3.0 (perfect match) to 0.5 (opposite season)
        multiplier = 1.0 - (normalized_distance * .95)
        
        return multiplier

    # def get_whomp(self):
    #     thresh = 1.0
    #     maxsound = 6
    #     # loud = self.analyzer.get_sound()
    #     loud = self.analyzer.get_all_sound()
    #     swloud = (loud > thresh) * 1
    #     self.whomp = swloud * (np.clip(loud, 0, maxsound) - thresh) / (maxsound - thresh)

    def apply_web_controls(self):
        """Apply web control values to system parameters."""
        # Skip entirely if web control is disabled
        if not self.enable_web_control or self.web_controller is None:
            return
        
        # Only check web controls occasionally - not every frame!
        if not hasattr(self, '_last_web_check'):
            self._last_web_check = 0
        
        # Only check every 0.2 seconds instead of every frame (reduces from 30Hz to 5Hz)
        if self.current_time - self._last_web_check < 0.2:
            return
        
        self._last_web_check = self.current_time
        
        # Check for weather set change requests (only if present)
        new_set = self.web_controller.get('request_weather_set')
        if new_set and new_set != self.current_set:
            self.change_weather_set(new_set)
            self.web_controller.set('request_weather_set', None)
        
        # Update status values every 0.5 seconds
        if not hasattr(self, '_last_status_update'):
            self._last_status_update = 0
        
        if self.current_time - self._last_status_update > 0.5:
            # Batch update to minimize lock acquisitions
            self.web_controller._dict_lock.acquire()
            try:
                self.web_controls['current_weather_set'] = self.current_set
                self.web_controls['current_weather'] = self.current_weather.value
                self.web_controls['season'] = float(self.season)
                self.web_controller._values_cache = None  # Invalidate cache
            finally:
                self.web_controller._dict_lock.release()
            self._last_status_update = self.current_time
        
        # Read control values (only when we actually check)
        weather_intensity = self.web_controller.get('weather_intensity')
        if weather_intensity is not None:
            # Apply intensity to weather effects
            pass
            
        fog_strength = self.web_controller.get('fog_strength')
        if fog_strength is not None:
            # Update fog strength in real-time
            pass
            
        audio_sensitivity = self.web_controller.get('audio_sensitivity')
        if audio_sensitivity is not None:
            # Adjust audio sensitivity
            self.analyzer.sensitivity = audio_sensitivity
    
    def transition_update(self):
        # self.progress = 1.0
        if self.current_weather != self.target_weather:
            self.progress = min(
                1.0, (self.current_time - self.transition_start) / self.transition_time
            )

            start_params = self.get_weather_params(self.current_weather)
            target_params = self.get_weather_params(self.target_weather)

            # Interpolate parameters
            for param in target_params:
                if isinstance(target_params[param], (int, float, np.ndarray)):
                    self.weather_params[param] = (
                        target_params[param] - start_params[param]
                    ) * self.progress + start_params[param]
                else:
                    # For non-numeric parameters, just use the target value
                    self.weather_params[param] = target_params[param]

            if self.progress >= 1.0:
                self.current_weather = self.target_weather
                self.weather_params = target_params.copy()

    def send_variables(self):
        # Apply season speed from current weather set (cache to avoid repeated lookups)
        if not hasattr(self, '_cached_set_config') or self._cached_set_config[0] != self.current_set:
            self._cached_set_config = (self.current_set, self.get_current_set_config())
        
        set_config = self._cached_set_config[1]
        season_speed = set_config.get("season_speed", 1.0)
        self.season = ((time.time() / 1800) * season_speed) % 1
        
        fog = np.maximum(0,self.weather_params["fog"] * (0.75 - 0.25 * np.cos(np.pi * 2 * (self.season - 0.625))))
        self.cloudyness = ((1 - self.weather_params["starryness"]) + (1 - self.weather_params["celestial_visibility"]) + fog + self.weather_params["rain_rate"] + self.weather_params["wind_speed"] / 3)/4
        
        # Batch all state updates to reduce dictionary overhead
        state = self.scheduler.state
        state["cloudyness"] = self.cloudyness
        state["fog_strength"] = fog
        state["fog_color"] = self.weather_params["fog_color"]
        state["wind"] = self.weather_params["wind_speed"] * np.cos(np.pi * 2 * (self.season - 0.125))
        state["season"] = self.season
        state["rain"] = self.weather_params["rain_rate"]
        state["starryness"] = self.weather_params["starryness"]
        state["sound"] = self.analyzer.get_extended_analysis()
        state["celestial_bodies"] = self.celestial_bodies
        state["celestial_visibility"] = self.weather_params["celestial_visibility"]
        state["firefly_density"] = self.weather_params["firefly_density"]
        state["meteor_rate"] = self.weather_params["meteor_rate"]
        state["volcano_level"] = (np.sin(self.current_time / 100) * 0.5 + 0.5) * self.weather_params["volcano_level"]
        state["sand_density"] = self.weather_params.get("sand_density", 0)
        state["tree_growth"] = (self.weather_params.get("tree_prob", 0) + 0.25)

    def random_events(self):
        randcheck = np.random.random()
        
        # Random events from current weather set configuration
        set_config = self.get_current_set_config()
        random_events = set_config.get("random_events", [])
        random_event_rate = set_config.get("random_event_rate", 0.0001)
        
        # Check if a random event should trigger based on the set's rate
        if random_events and randcheck < random_event_rate:
            # Assign each event a seasonal position based on its index
            # and select the one closest to the current season
            num_events = len(random_events)
            event_positions = np.linspace(0, 1, num_events, endpoint=False)
            
            # Find the event closest to the current season
            seasonal_distances = np.abs(event_positions - self.season)
            # Account for wraparound (e.g., season 0.95 is close to position 0.05)
            seasonal_distances = np.minimum(seasonal_distances, 1 - seasonal_distances)
            closest_index = np.argmin(seasonal_distances)
            
            event_name = random_events[closest_index]
            print(f"   🎲 Seasonal event triggered: {event_name} (season: {self.season:.3f}, position: {event_positions[closest_index]:.3f})")
            self._schedule_event_from_map(event_name, 0, 60, frame_id=0)
        
        if (randcheck < self.weather_params["tree_prob"] / 10000):
            # self.scheduler.schedule_event(0, 100, secondary_tree, frame_id=1) # noqa: F405
            self.scheduler.schedule_event(0, 80, fx.shader_tree,squish_top_width=self.scale, frame_id=0) # noqa: F405
        
        # Wolf howl
        # if (randcheck < (self.weather_params["Wolfy"] + self.weather_params["spookyness"] / 10) / 2000):
        #     self.scheduler.schedule_event(0, 10, Awooo_Wolf_Howl, frame_id=0) # noqa: F405

        # # Giant auroras in the sky
        if randcheck < self.weather_params["Aurora_probability"] / 1000:
            self.scheduler.schedule_event(0, 50, fx.shader_aurora, frame_id=0) # noqa: F405
        #     #self.scheduler.schedule_event(0, 50, secondary_Aurora, frame_id=1) # noqa: F405

        if randcheck < self.weather_params["lightning_probability"] / 500:
            self.scheduler.schedule_event(0, 1, fx.shader_lightning, frame_id=0) # noqa: F405

                
        randcheck = np.random.random()

        # Sand storms
        if randcheck < self.weather_params["sand_density"] / 2000:
            self.scheduler.schedule_event(0, 45, fx.shader_sandstorm, frame_id=0) # noqa: F405


        # # Spooky giant eye
        if randcheck < self.weather_params["spookyness"] / 1000:
            self.scheduler.schedule_event(0, 30, fx.shader_eye, squish_top_width=self.scale,frame_id=0) # noqa: F405

        # # Random meteor events
        if randcheck < self.weather_params["meteor_rate"] / 800:
            self.scheduler.schedule_event(0, 25, fx.shader_meteor, frame_id=0) # noqa: F405


        # Dancing cactus events
        randcheck = np.random.random()
                
    def random_state_change(self):
        # Apply set-specific transition speed (use cached config)
        if not hasattr(self, '_cached_set_config') or self._cached_set_config[0] != self.current_set:
            self._cached_set_config = (self.current_set, self.get_current_set_config())
        
        set_config = self._cached_set_config[1]
        transition_speed_mult = set_config.get("transition_speed", 1.0)
        
        randcheck = np.random.random()
        if (randcheck < (1 / 800) * self.weather_params["Switch_rate"] * transition_speed_mult) and (self.progress >= 0.99):
            self.progress = 0
            
            # Check if we need to change weather sets
            if self.target_set is not None:
                print(f"🌈 Switching weather set: '{self.current_set}' → '{self.target_set}'")
                self.current_set = self.target_set
                self.target_set = None
                self.web_controls["current_weather_set"] = self.current_set
                
                # Cancel all existing events and start new background events for the set
                self._initialize_weather_set_events()
                
                # Pick a random weather from the new set
                set_states = self.get_set_states()
                new_weather = np.random.choice(set_states)
                print(f"   Starting with: {new_weather.value}")
                
                new_weather_params = self.get_weather_params(new_weather)
                t_duration = new_weather_params["transition_duration"]
                self.transition_to_weather(new_weather, transition_duration=t_duration)
                return
            
            # Normal transition within current set
            current_preset = self.weather_presets[self.current_weather]
            possible_states = [WeatherState(state) for state in current_preset["possible_transitions"]]
            
            # Filter to only states in current set
            set_states = self.get_set_states()
            possible_states = [state for state in possible_states if state in set_states]
            
            if not possible_states:
                # If no valid transitions in set, pick random state from set
                possible_states = set_states
                base_weights = [1.0] * len(possible_states)
            else:
                # Use weights from preset, but only for states in the set
                base_weights = []
                for state in possible_states:
                    # Find the weight for this state
                    try:
                        idx = current_preset["possible_transitions"].index(state.value)
                        base_weights.append(current_preset["transition_weights"][idx])
                    except (ValueError, IndexError):
                        base_weights.append(1.0)
            
            # Apply seasonal modifiers to weights
            season_extremity = set_config.get("season_extremity", 1.0)
            adjusted_weights = []
            for i, state in enumerate(possible_states):
                # Get the season preference for this weather state
                target_season_pref = self.weather_presets[state].get("season_preference", 0.375)
                
                # Calculate seasonal multiplier (modified by extremity)
                season_multiplier = self.calculate_seasonal_weight_multiplier(target_season_pref, self.season)
                
                # Apply extremity: interpolate between 1.0 (no effect) and season_multiplier
                # Use max to ensure we never go below a small positive value
                if season_extremity > 0:
                    season_multiplier = 1.0 + (season_multiplier - 1.0) * season_extremity
                    season_multiplier = max(0.01, season_multiplier)  # Ensure non-negative
                else:
                    season_multiplier = 1.0  # No seasonal effect
                
                # Apply the seasonal modifier to the base weight
                adjusted_weight = base_weights[i] * season_multiplier
                adjusted_weights.append(adjusted_weight)
            
            # Normalize weights
            adjusted_weights = np.array(adjusted_weights)
            if np.sum(adjusted_weights) > 0:
                adjusted_weights = adjusted_weights / np.sum(adjusted_weights)
            else:
                adjusted_weights = np.ones(len(adjusted_weights)) / len(adjusted_weights)
            
            # Choose new weather state
            new_weather = np.random.choice(possible_states, p=adjusted_weights)
            
            # Find the transition duration
            new_weather_params = self.get_weather_params(new_weather)
            t_duration = new_weather_params["transition_duration"]
            self.transition_to_weather(new_weather, transition_duration=t_duration)

    def update(self):
        """Update the environmental system - should be called each frame"""
        #self.get_whomp()
        self.current_time = time.time()
        
        # Apply web control values
        self.apply_web_controls()
       
        # OSC handling
        #messages = self.scheduler.get_osc_messages()
        #if messages != []:
        #    print(messages)  # Eventually want to pass these to the scheduler
            
        # Handle transitions
        self.transition_update()

        # Update celestial bodies
        for body in self.celestial_bodies:
            body.update(self.current_time)
            
        # Apply current parameters to scheduler state
        self.send_variables()
        
        # Random events
        self.random_events()
        self.random_state_change()
        
        # Update the scheduler
        self.scheduler.update()


# Main execution
if __name__ == "__main__":
    scheduler = EventScheduler()
    env_system = EnvironmentalSystem(scheduler)

    #change to a specific weather set on startup
    env_system.change_weather_set("ethereal_mist", immediate=True)
    env_system.transition_to_weather(WeatherState.HEAVY_RAIN)
    env_system.scheduler.schedule_event(0, 160, fx.shader_pixel_spots, 
        # 0=red, 0.33=green, 0.66=blue
                        frame_id=0)
    #env_system.scheduler.schedule_event(0, 90, fx.shader_sandstorm,frame_id=0)
    #env_system.scheduler.schedule_event(10, 20, fx.shader_gameoflife,frame_id=0)  # noqa: F405
    #env_system.scheduler.schedule_event(0, 500, fx.shader_meteor,frame_id=0)
    last_time = time.time()
    FRAME_TIME = 1 / 50
    first_time = time.time()
    frame_count = 0
    fps_start_time = time.time()
    
    # For better sleep precision on Windows
    import sys
    if sys.platform == 'win32':
        import ctypes
        winmm = ctypes.WinDLL('winmm')
        winmm.timeBeginPeriod(1)  # Set 1ms timer resolution
    
    try:
        while True:
            frame_start = time.perf_counter()
            
            # Update environmental system (includes scheduler.update())
            env_system.update()

            # Calculate time taken
            frame_time = time.perf_counter() - frame_start
            
            # Sleep to maintain target framerate (busy-wait for last 1ms for precision)
            sleep_time = FRAME_TIME - frame_time
            if sleep_time > 0.001:
                time.sleep(sleep_time - 0.001)
                # Busy wait for the last millisecond for precision
                while time.perf_counter() - frame_start < FRAME_TIME:
                    pass
            
            frame_count += 1
            if frame_count % 50 == 0:  # Print FPS every 50 frames
                current_time = time.time()
                actual_fps = 50.0 / (current_time - fps_start_time)
                fps_start_time = current_time
                print(f"FPS: {actual_fps:.1f}, Frame time: {frame_time*1000:.1f}ms, Active events: {len(scheduler.active_events)}")
            # Print stats if needed
            # print(["%.2f" % (1/(time.time()-lasttime)), "%.2f" % len(scheduler.active_events), len(scheduler.event_queue),"%.3f" %((lasttime-first_time)/3600)])
            #last_time = time.time()

    except KeyboardInterrupt:
        print("Done!")