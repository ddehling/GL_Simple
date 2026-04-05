import numpy as np
import time
from pathlib import Path
import yaml
from engine.render_pipeline import RenderPipeline
import lib.dmx_sender as imdmx

from lib.audio_analyzer import MicrophoneAnalyzer
from lib.weather_params import (
    WeatherState, WEATHER_SETS, DEFAULT_WEATHER_SET
)
from lib.weather_state import WeatherStateController
from lib.weather_set import WeatherSetManager
from renderer.effects.celestial_bodies import CELESTIAL_BODIES
from renderer import effects as fx
from web.web_controller import WebController

def load_config():
    """Load config.yaml, falling back to defaults if missing."""
    config_path = Path(__file__).parent / "config.yaml"
    defaults = {
        "display": {"width": 128, "height": 300, "magnification": 0, "headless": False},
        "audio": {"enabled": True, "device_name": "TONOR"},
        "web": {"enabled": True, "port": 5000, "admin_password": "admin123", "bind_ip": ""},
        "dmx": {"bind_ip": "", "receivers": [
            {"ip": "192.168.68.140", "columns": 32, "column_offset": 0},
            {"ip": "192.168.68.141", "columns": 32, "column_offset": 32},
            {"ip": "192.168.68.142", "columns": 32, "column_offset": 64},
            {"ip": "192.168.68.143", "columns": 32, "column_offset": 96},
        ]},
    }
    if config_path.exists():
        with open(config_path, "r") as f:
            loaded = yaml.safe_load(f) or {}
        # Merge: loaded sections override defaults
        for section in defaults:
            if section in loaded:
                defaults[section].update(loaded[section])
        print(f"[Config] Loaded from {config_path}")
    else:
        print(f"[Config] {config_path} not found, using defaults")
    return defaults


class EnvironmentalSystem:
    def __init__(self):
        cfg = load_config()
        disp = cfg["display"]
        audio_cfg = cfg["audio"]
        web_cfg = cfg["web"]
        dmx_cfg = cfg["dmx"]

        frame_dimensions = [
            (disp["width"], disp["height"]),  # Frame 0 (primary/main display)
        ]

        # Hardware receiver configuration — built from config.yaml
        receivers = [
            [
                {
                    'ip': rx['ip'],
                    'pixel_count': disp["height"] * rx['columns'],
                    'addressing_array': imdmx.make_indices_V_rect_alternate(
                        rx['columns'], disp["height"], rx['column_offset']
                    ),
                }
                for rx in dmx_cfg['receivers']
            ],
#                  'ip': '192.168.68.111',
#                     'pixel_count': 2019,
#                     'addressing_array': imdmx.make_indicesHS(r"./config/UnitA.txt")
#                 },
#                 {
#                     'ip': '192.168.68.125',
#                     'pixel_count': 1777,
#                     'addressing_array': imdmx.make_indicesHS(r"./config/UnitB.txt")
#                 },
#                 {
#                     'ip': '192.168.68.124',
#                     'pixel_count': 1793,
#                     'addressing_array': imdmx.make_indicesHS(r"./config/UnitC.txt")
#                 },
            # [
            #     {
            #         'ip': '192.168.68.140',
            #         'pixel_count': disp["height"] * 32,
            #         'addressing_array': imdmx.make_indices_V_rect_alternate(32, disp["height"], 0),
            #     },
            #     {
            #         'ip': '192.168.68.141',
            #         'pixel_count': disp["height"] * 32,
            #         'addressing_array': imdmx.make_indices_V_rect_alternate(32, disp["height"], 32),
            #     },
            #     {
            #         'ip': '192.168.68.142',
            #         'pixel_count': disp["height"] * 32,
            #         'addressing_array': imdmx.make_indices_V_rect_alternate(32, disp["height"], 64),
            #     },
            #     {
            #         'ip': '192.168.68.143',
            #         'pixel_count': disp["height"] * 32,
            #         'addressing_array': imdmx.make_indices_V_rect_alternate(32, disp["height"], 96),
            #     },
            # ],
            ]

        self.scheduler = RenderPipeline(
            frame_dimensions=frame_dimensions,
            receivers=receivers,
            magnification=disp["magnification"],
            headless=disp["headless"],
            dmx_bind_ip=dmx_cfg.get("bind_ip", ""),
        )
        self.weather_state = WeatherStateController()
        self.season = 0.0
        self.analyzer = None
        if audio_cfg.get("enabled", True):
            try:
                self.analyzer = MicrophoneAnalyzer(device_name=audio_cfg["device_name"])
                self.analyzer.start()
            except Exception as e:
                print(f"[Audio] Failed to initialize microphone: {e}")
                print("[Audio] Continuing without audio input")
                self.analyzer = None
        self.scale = 0.2

        # Initialize web control system
        self.enable_web_control = web_cfg["enabled"]
        self.web_controller = None

        if self.enable_web_control:
            self.web_controls = {
                "current_weather_set": DEFAULT_WEATHER_SET,
                "available_sets": list(WEATHER_SETS.keys()),
                "available_weather_states": list(WEATHER_SETS[DEFAULT_WEATHER_SET]["states"]),
                "all_weather_states": [s.value for s in WeatherState],
                "state_switch_locked": True,
                "weather_state_locked": False,
                "led_width": frame_dimensions[0][0],
                "led_height": frame_dimensions[0][1],
            }
            self.web_controller = WebController(
                self.web_controls,
                port=web_cfg["port"],
                service_name="glsimple",
                admin_password=web_cfg["admin_password"],
                bind_ip=web_cfg.get("bind_ip", ""),
            )
            self.web_controller.start(threaded=True)
        
        # Initialize celestial bodies
        self.celestial_bodies = CELESTIAL_BODIES.copy()
        # sort celestial bodies by distance, farthest first
        self.celestial_bodies.sort(key=lambda x: x.distance, reverse=True)

        self.scheduler.state["tree"] = False
        self.scheduler.state["skyfull"] = False
        self.scheduler.state["simulate"] = True  # Display the leds in an opencv window for visualization
        self.active_effects = {"world": None, "ambient_sound": None}
        # Event map - maps event names to shader effects and their parameters
        # This is used for both background events and on-transition events
        # Format: "event_name": (shader_function, {params_dict})
        self.event_map = {
            "clouds": (fx.shader_drifting_clouds, {}),
            "firefly": (fx.shader_firefly, {}),
            "stars": (fx.shader_stars, {}),
            "rain": (fx.shader_rain, {}),
            "fog": (fx.shader_fog, {
                "strength": 0.0,
                "color": (0.7, 0.7, 0.8),
                "fog_near": 20.0,
                "fog_far": 80.0
            }),
            "sandstorm": (fx.shader_sandstorm, {}),
            "fog_beings": (fx.shader_chromatic_fog_beings, {}),
            "falling_leaves": (fx.shader_falling_leaves, {}), 
            "audio_balls": (fx.shader_audio_balls, {}),
            "audio_curve": (fx.shader_audio_curve, {}),
            "sunrise": (fx.shader_sunrise, {}),
            "game_of_life": (fx.shader_gameoflife, {}),
            "fractal_fog": (fx.shader_fractal_fog, {}),
            "noise_isovalues": (fx.shader_noise_isovalues, {}),
            "tentacle": (fx.shader_tentacle, {}),
            "tunnel_raymarch": (fx.shader_tunnel_raymarch, {}),
            "tunnel": (fx.shader_tunnel, {}),
            "voronoi_sphere": (fx.shader_voronoi_sphere, {}),
            "wave_terrain": (fx.shader_wave_terrain, {}),
            "wave_equation": (fx.shader_wave_equation, {}),
            "audio_scan_line": (fx.shader_audio_scan_line, {
                "scan_speed": 50.0,
                "trail_length": 75,
                "intensity_sensitivity": 2.0,
                "width_sensitivity": 0.5,
                "base_width": 2.0,
                "max_width": 20.0,
                "color_hue": 0.5
            }),
            "pixel_spots": (fx.shader_pixel_spots, {}),
            "vortex": (fx.shader_vortex, {}),
            "ocean_waves": (fx.shader_ocean_waves, {}),
            "kelp": (fx.shader_kelp, {}),
            "Bioluminescence": (fx.shader_bioluminescence, {}),
            "bubbles": (fx.shader_bubbles, {}),
            "fish": (fx.shader_fish, {}),
            "test_pattern": (fx.shader_test_pattern, {"orientation": "vertical"}),
            "bart_map": (fx.shader_bart_map, {}),
            "highway_traffic": (fx.shader_highway_traffic, {}),
            "test_fan_coords": (fx.shader_test_fan_coords, {}),
            "city_lights": (fx.shader_city_lights, {}),
            "bay_shimmer": (fx.shader_bay_shimmer, {}),
            "narrative_player": (fx.shader_narrative_player, {
                "script_path": "media/sounds/bartiki/script.json",
                "node_delay": 3.0,
                "restart_delay": 10.0,
            }),
        }
        
        # WeatherSetManager owns the event_map from here on
        self.weather_set = WeatherSetManager(self.event_map)
        del self.event_map  # WeatherSetManager is the single owner

        # Pass event names to web controller if enabled
        if self.enable_web_control:
            event_list = self.weather_set.get_event_names()
            self.web_controller.set_available_events(
                all_events=event_list,
                background_events=event_list
            )
            # Sync initial set name now that WeatherSetManager is ready
            self.web_controller.set("current_weather_set", self.weather_set.current_set)
        
        # Initialize background events for the starting weather set
        self._initialize_weather_set_events()
        
        self.whompcount = 0
    
    def update(self):
        """Update the environmental system - should be called each frame"""
        self.current_time = time.time()

        # Apply web control values
        self.apply_web_controls()

        # Handle transitions
        self.weather_state.update(self.current_time)

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

        # Copy PNG frame to web controller for preview streaming
        if hasattr(self, 'web_controller'):
            png = self.scheduler.state.get('_frame_png')
            if png is not None:
                self.web_controller.control_dict['_frame_png'] = png

    def change_weather_set(self, new_set_name: str, immediate: bool = False,
                           initial_weather: WeatherState = None):
        """Request a change to a different weather set.

        Args:
            new_set_name: Name of the weather set to change to.
            immediate: If True, change immediately. If False, queue for next transition.
            initial_weather: When immediate=True, start with this specific state instead
                             of a random one. Ignored if the state is not in the new set.
        """
        if not self.weather_set.is_valid_set(new_set_name):
            print(f"[WEATHER] Unknown weather set: {new_set_name}")
            return False

        if new_set_name == self.weather_set.current_set:
            print(f"[WEATHER] Already in set '{new_set_name}', skipping")
            return True

        if immediate:
            # Apply the change immediately
            print(f"[WEATHER] Switching weather set immediately: '{self.weather_set.current_set}' -> '{new_set_name}'")
            self.weather_set.commit_set_change(new_set_name)
            if self.enable_web_control:
                self.web_controller.set("current_weather_set", self.weather_set.current_set)
                self.web_controller.set("available_weather_states",
                                        list(self.weather_set.get_current_set_config()["states"]))

            # Cancel all existing events and start new background events for the set
            self._initialize_weather_set_events()

            # Pick initial weather: use caller's choice if valid, else random
            set_states = self.weather_set.get_set_states()
            if initial_weather is not None and initial_weather in set_states:
                new_weather = initial_weather
            else:
                new_weather = np.random.choice(set_states)
            print(f"[WEATHER]   Starting with: {new_weather.value}")

            new_weather_params = self.weather_state.get_weather_params(new_weather)
            t_duration = new_weather_params["transition_duration"]
            if self.enable_web_control and self.web_controller.get('instant_transitions', False):
                t_duration = 0.01
            # Snap instantly on immediate set change — don't blend from previous set's params
            self.transition_to_weather(new_weather, transition_duration=0.01)
        else:
            # Queue for next transition
            self.weather_set.queue_set_change(new_set_name)
            print(f"[WEATHER] Weather set change queued: '{self.weather_set.current_set}' -> '{new_set_name}'")
            print(f"[WEATHER]   Will transition on next weather change...")
        
        return True
    
    def _initialize_weather_set_events(self):
        """Cancel all events and start background events for the current weather set"""
        print(f"[WEATHER] Initializing events for weather set: '{self.weather_set.current_set}'")

        # Cancel all active events and fade out all audio
        self.scheduler.cancel_all_events()
        engine = self.scheduler.state.get("soundengine")
        if engine:
            engine.stop_all(duration=2.0)

        # Schedule the permanent background events based on set configuration
        sim_forever = 10E9  # 10 billion seconds (over 300 years)

        for event_name in self.weather_set.get_background_events():
            print(f"[WEATHER]   Background event: {event_name}")
            self._schedule_event_from_map(event_name, 0, sim_forever, frame_id=0)

        print(f"[WEATHER] Background events initialized for '{self.weather_set.current_set}'")
    
    def _schedule_event_from_map(self, event_name: str, start_time: float, duration: float, frame_id: int = 0):
        """Schedule an event from the event map"""
        entry = self.weather_set.resolve_event(event_name)
        if entry is None:
            print(f"[WEATHER] Unknown event: {event_name}")
            return None

        effect_func, params = entry
        return self.scheduler.schedule_event(start_time, duration, effect_func, frame_id=frame_id, **params)

    def transition_to_weather(self, new_weather: WeatherState, transition_duration: float = 10.0):
        """Start a transition to a new weather state"""
        target_params = self.weather_state.start_transition(new_weather, transition_duration, time.time())
        
        # Schedule events based on on_transition_events in weather preset
        on_transition_events = target_params.get("on_transition_events", [])
        for event_config in on_transition_events:
            if isinstance(event_config, (tuple, list)) and len(event_config) >= 2:
                event_name, duration = event_config[:2]
                frame_id = event_config[2] if len(event_config) > 2 else 0
                print(f"[WEATHER]   Transition event: {event_name} ({duration}s)")
                self._schedule_event_from_map(event_name, 0, duration, frame_id=frame_id)
            else:
                print(f"[WEATHER]   Invalid on_transition_event format: {event_config!r}")

        ambient_sound = target_params.get("ambient_sound")
        skip_time = target_params.get("skiptime", 0.0)
        ari = target_params.get("ARI", 0.0)
        engine = self.scheduler.state["soundengine"]
        if ambient_sound:
            sound_path = Path("media") / Path("sounds") / ambient_sound
            engine.play_ambient(sound_path, skip_seconds=skip_time, ari=ari)
        else:
            engine.stop_ambient()
        self.active_effects["ambient_sound"] = ambient_sound

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

        # Check for weather set/state change requests (read and clear atomically)
        with self.web_controller._dict_lock:
            new_set = self.web_controller.control_dict.pop('request_weather_set', None)
            new_state = self.web_controller.control_dict.pop('request_weather_state', None)
            trigger_event = self.web_controller.control_dict.pop('request_trigger_event', False)

        if new_set is not None and new_set != self.weather_set.current_set:
            self.change_weather_set(new_set, immediate=True)

        if new_state is not None:
            locked = self.web_controller.get('state_switch_locked', True)
            all_states = [s.value for s in WeatherState]
            if new_state in all_states and (not locked or new_state in [s.value for s in self.weather_set.get_set_states()]):
                state_enum = WeatherState(new_state)
                t_duration = self.weather_state.get_weather_params(state_enum)["transition_duration"]
                if self.web_controller.get('instant_transitions', False):
                    t_duration = 0.01
                self.transition_to_weather(state_enum, transition_duration=t_duration)
            else:
                print(f"[WEATHER] Requested state '{new_state}' rejected (locked={locked})")

        if trigger_event:
            if isinstance(trigger_event, str) and trigger_event:
                # Specific event requested
                event_name = trigger_event
            else:
                # Random pick from current set
                random_events, _ = self.weather_set.get_random_events_config()
                event_name = np.random.choice(random_events) if random_events else None
            if event_name:
                print(f"[WEB] Triggered event: {event_name}")
                self._schedule_event_from_map(event_name, 0, 60, frame_id=0)

        # Apply audio sensitivity from global modifiers
        if self.analyzer:
            with self.web_controller._dict_lock:
                audio_sens = self.web_controller.global_modifiers.get('audio_sensitivity', 1.0)
            self.analyzer.sensitivity = audio_sens

        # Update audio summary frequently (every web check = 0.2s / 5 Hz)
        # This is lightweight: just 32 floats + 2 numbers
        try:
            current_bands = self.analyzer.get_current_bands(normalize='long') if self.analyzer else None
            if current_bands is not None:
                audio_summary = {
                    "bands": current_bands.tolist(),
                    "peak_band": int(np.argmax(current_bands)),
                    "total_power": float(np.sum(current_bands)),
                    "sensitivity": self.analyzer.sensitivity,
                }
                with self.web_controller._dict_lock:
                    self.web_controller.control_dict['audio_summary'] = audio_summary
        except Exception:
            pass

        # Update status values every 0.5 seconds
        if not hasattr(self, '_last_status_update'):
            self._last_status_update = 0

        if self.current_time - self._last_status_update > 0.5:
            # Build transition state info
            transitioning = self.weather_state.current_weather != self.weather_state.target_weather
            transition_state = {
                "current": self.weather_state.current_weather.value,
                "target": self.weather_state.target_weather.value,
                "progress": float(self.weather_state.progress) if hasattr(self.weather_state, 'progress') else 1.0,
                "transitioning": transitioning,
            }

            # Use the last frame's output values (post-override, what effects see)
            # Falls back to raw weather_params on first frame
            params_snapshot = {}
            source = getattr(self, '_last_web_output', self.weather_state.weather_params)
            for k, v in source.items():
                if isinstance(v, np.ndarray):
                    params_snapshot[k] = v.tolist()
                elif isinstance(v, (int, float, str, bool, list)):
                    params_snapshot[k] = v

            # Build active effects list (snapshot to avoid race with scheduler thread)
            try:
                active_effects = [e.name for e in list(self.scheduler._scheduler.active_events)]
            except Exception as e:
                print(f"[WebController] Error reading active_events: {e}")
                active_effects = []

            # Batch update to minimize lock acquisitions
            self.web_controller._dict_lock.acquire()
            try:
                d = self.web_controller.control_dict
                d['current_weather_set'] = self.weather_set.current_set
                d['available_weather_states'] = list(self.weather_set.get_current_set_config()["states"])
                d['random_events'] = list(self.weather_set.get_current_set_config().get("random_events", []))
                d['current_weather'] = self.weather_state.current_weather.value
                d['season'] = float(self.season)
                d['brightness_limiting_factor'] = round(self.scheduler.brightness_state[0]['divisor'], 3)
                d['weather_params_snapshot'] = params_snapshot
                d['transition_state'] = transition_state
                d['active_overrides'] = dict(self.web_controller.web_param_overrides)
                d['global_modifiers'] = dict(self.web_controller.global_modifiers)
                d['fps'] = getattr(self, '_current_fps', 0)
                d['active_effects'] = active_effects
                d['ambient_sound'] = self.active_effects.get("ambient_sound")
                d['allowed_output_params'] = self._get_allowed_output_params()
                self.web_controller._values_cache = None  # Invalidate cache
            finally:
                self.web_controller._dict_lock.release()
            self._last_status_update = self.current_time
    
    # Output keys from get_state_output() that weather_intensity should scale
    WEATHER_INTENSITY_KEYS = {"rain", "wind", "sand_density", "volcano_level"}

    # Maps allowed_parameters input names → output keys from get_state_output()
    # Parameters not in this map pass through with the same name
    _INPUT_TO_OUTPUT_PARAM = {
        "fog": "fog_strength",
        "wind_speed": "wind",
        "rain_rate": "rain",
        "tree_prob": "tree_growth",
    }

    def _get_allowed_output_params(self):
        """Return the set of output param keys allowed for the current weather set."""
        set_config = WEATHER_SETS.get(self.weather_set.current_set, {})
        allowed_input = set_config.get("allowed_parameters", [])
        if not allowed_input:
            return None  # No restrictions (e.g. test set)
        result = set()
        for inp in allowed_input:
            out = self._INPUT_TO_OUTPUT_PARAM.get(inp, inp)
            result.add(out)
        # cloudyness is always derived from multiple params, include it if any contributor is allowed
        cloud_contributors = {"fog", "wind_speed", "rain_rate", "starryness", "celestial_visibility"}
        if cloud_contributors & set(allowed_input):
            result.add("cloudyness")
        return list(result)

    def send_variables(self):
        season_speed = self.weather_set.get_season_speed()
        self.season = ((time.time() / 1800) * season_speed) % 1

        state = self.scheduler.state
        output = self.weather_state.get_state_output(self.season, self.current_time)

        # Apply global modifiers and overrides to the output (not to weather_params)
        if self.enable_web_control and self.web_controller is not None:
            with self.web_controller._dict_lock:
                intensity = self.web_controller.global_modifiers.get('weather_intensity', 1.0)
                brightness_mod = self.web_controller.global_modifiers.get('brightness', 1.0)
                overrides = dict(self.web_controller.web_param_overrides)

            # Scale weather intensity on output keys
            if intensity != 1.0:
                for key in self.WEATHER_INTENSITY_KEYS:
                    if key in output:
                        output[key] = output[key] * intensity

            # Apply direct overrides to output (these replace values entirely)
            for param, value in overrides.items():
                if param in output:
                    output[param] = value

            # Store brightness modifier in state for render pipeline to apply
            # after the hardware limiter (can only dim, never brighten past limiter)
            state["web_brightness"] = brightness_mod

            # Volume controls — both applied in real time in the audio engine mixer
            master_vol = self.web_controller.global_modifiers.get('master_volume', 1.0)
            narrative_vol = self.web_controller.global_modifiers.get('narrative_volume', 1.0)
            state["soundengine"].master_volume = master_vol
            state["soundengine"].narrative_volume = narrative_vol

            # Cache the final output for the web UI snapshot (post-overrides)
            self._last_web_output = output

        state.update(output)
        state["season"] = self.season
        state["scale"] = self.scale
        state["sound"] = self.analyzer.get_extended_analysis() if self.analyzer else None
        state["celestial_bodies"] = self.celestial_bodies

    def random_events(self):
        randcheck = np.random.random()
        
        # Random events from current weather set configuration
        random_events, random_event_rate = self.weather_set.get_random_events_config()
        
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
        
        if (randcheck < self.weather_state.weather_params["tree_prob"] / 10000):
            self.scheduler.schedule_event(0, 80, fx.shader_tree, frame_id=0) # noqa: F405

        if randcheck < self.weather_state.weather_params["Aurora_probability"] / 1000:
            self.scheduler.schedule_event(0, 50, fx.shader_aurora, frame_id=0) # noqa: F405

        if randcheck < self.weather_state.weather_params["lightning_probability"] / 500:
            self.scheduler.schedule_event(0, 1, fx.shader_lightning, frame_id=0) # noqa: F405


        randcheck = np.random.random()

        # Sand storms
        if randcheck < self.weather_state.weather_params["sand_density"] / 2000:
            self.scheduler.schedule_event(0, 45, fx.shader_sandstorm, frame_id=0) # noqa: F405


        # # Spooky giant eye
        if randcheck < self.weather_state.weather_params["spookyness"] / 1000:
            self.scheduler.schedule_event(0, 30, fx.shader_eye, frame_id=0) # noqa: F405

        # # Random meteor events
        if randcheck < self.weather_state.weather_params["meteor_rate"] / 800:
            self.scheduler.schedule_event(0, 25, fx.shader_meteor, frame_id=0) # noqa: F405


        # Dancing cactus events
        randcheck = np.random.random()

    def random_state_change(self):
        if self.enable_web_control and self.web_controller.get('weather_state_locked', False):
            return

        transition_speed_mult = self.weather_set.get_transition_speed()

        randcheck = np.random.random()
        if (randcheck < (1 / 800) * self.weather_state.weather_params["Switch_rate"] * transition_speed_mult) and (self.weather_state.progress >= 0.99):
            self.weather_state.progress = 0

            # Check if we need to change weather sets
            if self.weather_set.has_pending_set_change():
                old_set = self.weather_set.current_set
                new_set_name = self.weather_set.consume_pending_set()
                print(f"[WEATHER] Switching weather set: '{old_set}' -> '{new_set_name}'")
                if self.enable_web_control:
                    self.web_controller.set("current_weather_set", self.weather_set.current_set)

                # Cancel all existing events and start new background events for the set
                self._initialize_weather_set_events()

                # Pick a random weather from the new set
                set_states = self.weather_set.get_set_states()
                new_weather = np.random.choice(set_states)
                print(f"[WEATHER]   Starting with: {new_weather.value}")

                new_weather_params = self.weather_state.get_weather_params(new_weather)
                t_duration = new_weather_params["transition_duration"]
                if self.enable_web_control and self.web_controller.get('instant_transitions', False):
                    t_duration = 0.01
                self.transition_to_weather(new_weather, transition_duration=t_duration)
                return

            # Normal transition within current set
            new_weather = self.weather_state.select_next_weather(
                self.weather_state.current_weather,
                self.weather_set.get_set_states(),
                self.season,
                self.weather_set.get_season_extremity(),
            )

            # Find the transition duration
            new_weather_params = self.weather_state.get_weather_params(new_weather)
            t_duration = new_weather_params["transition_duration"]
            if self.enable_web_control and self.web_controller.get('instant_transitions', False):
                t_duration = 0.01
            self.transition_to_weather(new_weather, transition_duration=t_duration)

    def shutdown(self):
        """Stop all audio and background threads cleanly."""
        engine = self.scheduler.state.get("soundengine")
        if engine:
            engine.stop_ambient()
        if self.analyzer:
            self.analyzer.stop()


# Main execution
if __name__ == "__main__":
    env_system = EnvironmentalSystem()

    #TODO: A way to set a weather state independently of set for testing

    # Change to a specific weather set and state on startup
    env_system.change_weather_set("bartiki", immediate=True,
                                  initial_weather=WeatherState.BARTIKI_MIDDAY)
    last_time = time.time()
    FRAME_TIME = 1 / 40
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

            if env_system.scheduler.should_exit:
                break

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
            if frame_count % 500 == 0:  # Print FPS every 500 frames
                current_time = time.time()
                actual_fps = 500.0 / (current_time - fps_start_time)
                env_system._current_fps = round(actual_fps, 1)
                fps_start_time = current_time

    except KeyboardInterrupt:
        pass
    finally:
        env_system.shutdown()
        print("Done!")