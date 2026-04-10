"""
Web-based control system for GL_Simple environmental effects.
Provides a Flask web server with real-time WebSocket control interface.
"""
import threading
import socket
import hashlib
import secrets
import time
import copy
from flask import Flask, render_template, jsonify, request, session
from flask_socketio import SocketIO, emit, join_room, leave_room
from pathlib import Path
import json
from zeroconf import Zeroconf, ServiceInfo
from renderer.fan_geometry import FanGeometry
import numpy as np


def _sanitize_for_json(obj):
    """Recursively convert numpy scalars/arrays to Python-native types so
    json.dumps (used by socketio.emit) doesn't choke on float32 etc."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.complexfloating)):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


class WebController:
    """
    Web-based control interface for modifying runtime parameters.
    Thread-safe dictionary updates with real-time web interface.
    """
    
    def __init__(self, control_dict=None, port=5000, service_name="lucifera", admin_password=None, bind_ip=""):
        """
        Initialize the web controller.
        
        Args:
            control_dict: Dictionary to be controlled (default: creates new dict)
            port: Port number for the web server (default: 5000)
            service_name: mDNS service name (default: "lucifera")
                         Will be accessible at http://{service_name}.local:{port}
            admin_password: Password for admin panel (default: None = no admin access)
        """
        self.control_dict = control_dict if control_dict is not None else {}
        self.port = port
        self.service_name = service_name
        self.bind_ip = bind_ip
        
        # Add thread lock for thread-safe dictionary access
        self._dict_lock = threading.RLock()
        
        # Cache for frequently accessed values to reduce lock contention
        self._values_cache = None
        self._values_cache_time = 0
        self._cache_duration = 0.1  # Cache for 100ms
        
        self.app = Flask(__name__,
                        template_folder=str(Path(__file__).parent / 'templates'),
                        static_folder=str(Path(__file__).parent / 'static'),
                        static_url_path='/static')

        # Configure Flask session with a secret key
        self.app.secret_key = secrets.token_hex(32)

        # Custom JSON provider so jsonify() handles numpy types everywhere.
        from flask.json.provider import DefaultJSONProvider
        class _NumpyJSONProvider(DefaultJSONProvider):
            def default(self, o):
                if isinstance(o, np.ndarray):
                    return o.tolist()
                if isinstance(o, (np.floating, np.complexfloating)):
                    return float(o)
                if isinstance(o, np.integer):
                    return int(o)
                if isinstance(o, np.bool_):
                    return bool(o)
                return super().default(o)
        self.app.json_provider_class = _NumpyJSONProvider
        self.app.json = _NumpyJSONProvider(self.app)

        # Initialize SocketIO with threading async mode
        self.socketio = SocketIO(self.app, async_mode='threading',
                                 cors_allowed_origins='*', logger=False,
                                 engineio_logger=False)

        # Hash the admin password if provided
        self.available_events = []  # Will be set by EnvironmentalSystem
        self.admin_password_hash = None
        if admin_password:
            self.admin_password_hash = hashlib.sha256(admin_password.encode()).hexdigest()

        self.server_thread = None
        self.zeroconf = None
        self.service_info = None
        self._emitter_thread = None
        self._emitter_running = False
        self._setup_routes()
        self._setup_socketio_events()

        # Global modifiers — multiplicative scalers (0.0–2.0, default 1.0)
        self.global_modifiers = {
            "weather_intensity": 1.0,
            "effect_speed": 1.0,
            "audio_sensitivity": 1.0,
            "brightness": 1.0,
            "master_volume": 1.0,
            "narrative_volume": 1.0,
        }

        # Global modifier metadata for UI generation
        self.global_modifier_schema = {
            "weather_intensity": {
                "label": "Weather Intensity",
                "min": 0.0, "max": 2.0, "step": 0.05, "default": 1.0,
                "description": "Scales dramatic weather effects (rain, wind, lightning, etc.)"
            },
            "effect_speed": {
                "label": "Effect Speed",
                "min": 0.0, "max": 2.0, "step": 0.05, "default": 1.0,
                "description": "Scales animation and transition timing"
            },
            "audio_sensitivity": {
                "label": "Audio Sensitivity",
                "min": 0.1, "max": 3.0, "step": 0.05, "default": 1.0,
                "description": "Scales how much audio input affects visuals"
            },
            "brightness": {
                "label": "Brightness",
                "min": 0.0, "max": 1.0, "step": 0.05, "default": 1.0,
                "description": "Global brightness dimmer (cannot exceed power limiter)"
            },
            "master_volume": {
                "label": "Master Volume",
                "min": 0.0, "max": 2.0, "step": 0.05, "default": 1.0,
                "description": "Controls all audio output volume"
            },
            "narrative_volume": {
                "label": "Narrative Volume",
                "min": 0.0, "max": 2.0, "step": 0.05, "default": 1.0,
                "description": "Controls narrative player TTS volume"
            },
        }

        # Weather parameter overrides — direct value overrides (param_name -> value)
        self.web_param_overrides = {}

        # Preview frame streaming
        # Preview clients are tracked via Socket.IO 'preview' room
        self._geometry_json = None  # cached FanGeometry JSON
    
    def _setup_routes(self):
        """Setup Flask routes for the web interface."""
        
        @self.app.route('/')
        def index():
            """Serve the main control page."""
            return render_template('control_panel.html')
        
        @self.app.route('/weather_sets')
        def weather_sets():
            """Serve the weather sets control page."""
            return render_template('weather_sets.html')
        
        @self.app.route('/weather_editor')
        def weather_editor():
            """Serve the weather set editor page."""
            return render_template('weather_editor.html')
        
        @self.app.route('/preview')
        def preview():
            """Serve the live preview page."""
            return render_template('preview.html')

        @self.app.route('/api/preview/geometry')
        def preview_geometry():
            """Return fan geometry JSON (fetched once by the WebGL client)."""
            if self._geometry_json is None:
                # Build geometry using LED dimensions from control_dict
                w = self.control_dict.get('led_width', 128)
                h = self.control_dict.get('led_height', 300)
                # Use the fan window aspect ratio (900x500)
                geo = FanGeometry(w, h, 900 / 500)
                self._geometry_json = geo.to_json()
            return jsonify(self._geometry_json)

        @self.app.route('/admin')
        def admin_panel():
            """Serve the admin panel page."""
            if not self.admin_password_hash:
                return jsonify({"error": "Admin panel not configured"}), 403
            return render_template('admin_panel.html')
        
        @self.app.route('/api/admin/login', methods=['POST'])
        def admin_login():
            """Authenticate admin user."""
            if not self.admin_password_hash:
                return jsonify({"success": False, "error": "Admin panel not configured"}), 403
            
            data = request.json
            password = data.get('password', '')
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            if password_hash == self.admin_password_hash:
                session['admin_authenticated'] = True
                return jsonify({"success": True})
            else:
                return jsonify({"success": False, "error": "Invalid password"}), 401
        
        @self.app.route('/api/admin/logout', methods=['POST'])
        def admin_logout():
            """Logout admin user."""
            session.pop('admin_authenticated', None)
            return jsonify({"success": True})
        
        @self.app.route('/api/admin/check')
        def admin_check():
            """Check if admin is authenticated."""
            is_authenticated = session.get('admin_authenticated', False)
            has_admin = self.admin_password_hash is not None
            return jsonify({
                "authenticated": is_authenticated,
                "admin_enabled": has_admin
            })
        
        @self.app.route('/api/admin/system_info')
        def system_info():
            """Return system information (admin only)."""
            if not session.get('admin_authenticated', False):
                return jsonify({"error": "Unauthorized"}), 401
            
            import platform
            
            info = {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "hostname": socket.gethostname()
            }
            
            # Try to get psutil info, but don't fail if not available
            try:
                import psutil
                info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
                info["memory_percent"] = psutil.virtual_memory().percent
            except ImportError:
                info["cpu_percent"] = "N/A (install psutil)"
                info["memory_percent"] = "N/A (install psutil)"
            
            return jsonify(info)
        
        @self.app.route('/api/globals/schema')
        def get_globals_schema():
            """Return the global modifier schema for UI generation."""
            return jsonify(self.global_modifier_schema)

        @self.app.route('/api/globals/values')
        def get_globals_values():
            """Return current global modifier values."""
            with self._dict_lock:
                return jsonify(dict(self.global_modifiers))

        @self.app.route('/api/globals/update', methods=['POST'])
        def update_global():
            """Update a global modifier value."""
            data = request.json
            modifier = data.get('modifier')
            value = data.get('value')

            if modifier not in self.global_modifier_schema:
                return jsonify({"success": False, "error": f"Unknown modifier: {modifier}"}), 400

            schema = self.global_modifier_schema[modifier]
            value = max(schema["min"], min(schema["max"], float(value)))

            with self._dict_lock:
                self.global_modifiers[modifier] = value
            self._values_cache = None

            return jsonify({"success": True, "modifier": modifier, "value": value})

        @self.app.route('/api/params/override', methods=['POST'])
        def set_param_override():
            """Set a weather parameter override (direct value)."""
            data = request.json
            param = data.get('param')
            value = data.get('value')

            if param is None or value is None:
                return jsonify({"success": False, "error": "Must provide 'param' and 'value'"}), 400

            with self._dict_lock:
                self.web_param_overrides[param] = float(value)
            self._values_cache = None

            return jsonify({"success": True, "param": param, "value": float(value)})

        @self.app.route('/api/params/clear_override', methods=['POST'])
        def clear_param_override():
            """Remove a weather parameter override."""
            data = request.json
            param = data.get('param')

            with self._dict_lock:
                removed = self.web_param_overrides.pop(param, None)
            self._values_cache = None

            return jsonify({"success": True, "param": param, "was_overridden": removed is not None})

        @self.app.route('/api/params/clear_all_overrides', methods=['POST'])
        def clear_all_overrides():
            """Remove all weather parameter overrides."""
            with self._dict_lock:
                count = len(self.web_param_overrides)
                self.web_param_overrides.clear()
            self._values_cache = None

            return jsonify({"success": True, "cleared": count})

        @self.app.route('/api/state/snapshot')
        def get_state_snapshot():
            """Return current system state: weather params, globals, transition, audio."""
            import time
            current_time = time.time()

            if (self._values_cache is not None and
                current_time - self._values_cache_time < self._cache_duration):
                return self._values_cache

            with self._dict_lock:
                snapshot = {
                    "global_modifiers": dict(self.global_modifiers),
                    "active_overrides": dict(self.web_param_overrides),
                    "weather_params": self.control_dict.get('weather_params_snapshot', {}),
                    "transition": self.control_dict.get('transition_state', {}),
                    "audio_summary": self.control_dict.get('audio_summary', {}),
                    "fps": self.control_dict.get('fps', 0),
                    "current_weather_set": self.control_dict.get('current_weather_set', 'unknown'),
                    "current_weather": self.control_dict.get('current_weather', 'unknown'),
                    "season": self.control_dict.get('season', 0.0),
                    "brightness_limiting_factor": self.control_dict.get('brightness_limiting_factor', 1.0),
                    "active_effects": self.control_dict.get('active_effects', []),
                    "allowed_output_params": self.control_dict.get('allowed_output_params'),
                }

            response = jsonify(snapshot)
            self._values_cache = response
            self._values_cache_time = current_time
            return response

        @self.app.route('/api/values')
        def get_values():
            """Return current values from the control dictionary."""
            with self._dict_lock:
                values_snapshot = self.control_dict.copy()
            return jsonify(values_snapshot)
        
        @self.app.route('/api/weather_set/change', methods=['POST'])
        def change_weather_set():
            """Request a weather set change."""
            data = request.json
            new_set = data.get('set_name')
            
            if not new_set:
                return jsonify({"success": False, "error": "No set_name provided"}), 400
            
            # Check if set exists (read with lock)
            with self._dict_lock:
                available_sets = self.control_dict.get('available_sets', [])
            
            if new_set not in available_sets:
                return jsonify({"success": False, "error": f"Unknown set: {new_set}"}), 400
            
            # Set the request in control dict for main loop to pick up
            with self._dict_lock:
                self.control_dict['request_weather_set'] = new_set
            
            # Invalidate cache
            self._values_cache = None
            
            return jsonify({
                "success": True, 
                "message": f"Weather set change to '{new_set}' queued",
                "set_name": new_set
            })
        
        @self.app.route('/api/weather_state/change', methods=['POST'])
        def change_weather_state():
            """Request a weather state change."""
            data = request.json
            new_state = data.get('state_name')

            if not new_state:
                return jsonify({"success": False, "error": "No state_name provided"}), 400

            with self._dict_lock:
                locked = self.control_dict.get('state_switch_locked', True)
                valid_states = (
                    self.control_dict.get('available_weather_states', [])
                    if locked
                    else self.control_dict.get('all_weather_states', [])
                )

            if new_state not in valid_states:
                return jsonify({"success": False, "error": f"Unknown state: {new_state}"}), 400

            with self._dict_lock:
                self.control_dict['request_weather_state'] = new_state

            self._values_cache = None

            return jsonify({
                "success": True,
                "message": f"Weather state change to '{new_state}' queued",
                "state_name": new_state
            })

        @self.app.route('/api/weather_state/set_lock', methods=['POST'])
        def set_weather_state_lock():
            """Set the weather state switch lock."""
            data = request.json
            locked = data.get('locked')
            if locked is None:
                return jsonify({"success": False, "error": "No 'locked' value provided"}), 400

            with self._dict_lock:
                self.control_dict['state_switch_locked'] = bool(locked)

            self._values_cache = None

            return jsonify({"success": True, "locked": bool(locked)})

        @self.app.route('/api/weather_state/set_state_locked', methods=['POST'])
        def set_state_locked():
            """Lock or unlock automatic weather state transitions."""
            data = request.json
            locked = data.get('locked')
            if locked is None:
                return jsonify({"success": False, "error": "No 'locked' value provided"}), 400

            with self._dict_lock:
                self.control_dict['weather_state_locked'] = bool(locked)

            self._values_cache = None

            return jsonify({"success": True, "locked": bool(locked)})

        @self.app.route('/api/weather_set/info')
        def weather_set_info():
            """Get current weather set information."""
            with self._dict_lock:
                info = {
                    "current_set": self.control_dict.get('current_weather_set', 'unknown'),
                    "available_sets": self.control_dict.get('available_sets', []),
                    "available_weather_states": self.control_dict.get('available_weather_states', []),
                    "all_weather_states": self.control_dict.get('all_weather_states', []),
                    "current_weather": self.control_dict.get('current_weather', 'unknown'),
                    "season": self.control_dict.get('season', 0.0),
                    "state_switch_locked": self.control_dict.get('state_switch_locked', True),
                    "weather_state_locked": self.control_dict.get('weather_state_locked', False),
                    "brightness_limiting_factor": self.control_dict.get('brightness_limiting_factor', 1.0),
                    "active_effects": self.control_dict.get('active_effects', []),
                    "available_events": sorted(self.available_events) if self.available_events else [],
                    "random_events": self.control_dict.get('random_events', []),
                }
            return jsonify(info)
        
        @self.app.route('/api/weather_editor/all_data')
        def get_all_weather_data():
            """Get all weather states, presets, and sets for editing."""
            # Check if cache should be refreshed (allow manual refresh via query param)
            refresh = request.args.get('refresh', 'false').lower() == 'true'
            
            # Cache this expensive operation (but allow refresh)
            if not hasattr(self, '_weather_data_cache') or refresh:
                from lib.weather_params import (
                    WeatherState, DEFAULT_WEATHER_PARAMS, WEATHER_PRESETS, WEATHER_SETS,
                    GLOBAL_PARAMETERS, PARAMETER_DEFINITIONS, AVAILABLE_BACKGROUND_EVENTS
                )
                import numpy as np
                from pathlib import Path
                
                def convert_to_json_serializable(obj):
                    """Recursively convert numpy arrays to lists"""
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_json_serializable(item) for item in obj]
                    else:
                        return obj
                
                # Get available sound files (only once)
                sound_files = []
                sounds_dir = Path(__file__).parent.parent / 'media' / 'sounds'
                if sounds_dir.exists():
                    sound_files = [f.name for f in sounds_dir.iterdir() if f.is_file()]
                    sound_files.sort()
                
                # Convert WeatherState enum to list of strings
                weather_states = [state.value for state in WeatherState]
                
                # Convert WEATHER_PRESETS dict (with WeatherState keys) to JSON-friendly format
                presets = {}
                for state, params in WEATHER_PRESETS.items():
                    state_key = state.value if hasattr(state, 'value') else str(state)
                    params_copy = convert_to_json_serializable(params.copy())
                    presets[state_key] = params_copy
                
                # Convert default params
                default_params = convert_to_json_serializable(DEFAULT_WEATHER_PARAMS.copy())
                
                # Convert weather sets
                weather_sets = convert_to_json_serializable(WEATHER_SETS.copy())
                
                # Use dynamically provided background events if available, otherwise fall back to static list
                background_events = (
                    sorted(self.available_background_events) 
                    if hasattr(self, 'available_background_events') 
                    else AVAILABLE_BACKGROUND_EVENTS
                )
                
                # Cache the result
                self._weather_data_cache = {
                    "weather_states": weather_states,
                    "default_params": default_params,
                    "weather_presets": presets,
                    "weather_sets": weather_sets,
                    "global_parameters": GLOBAL_PARAMETERS,
                    "parameter_definitions": PARAMETER_DEFINITIONS,
                    "available_background_events": background_events,
                    "available_sounds": sound_files,
                    "available_events": sorted(self.available_events) if hasattr(self, 'available_events') else []
                }
            
            return jsonify(self._weather_data_cache)
        
        @self.app.route('/api/weather_editor/save', methods=['POST'])
        def save_weather_data():
            """Save modified weather data back to weather_params.py."""
            try:
                data = request.json
                from lib.weather_editor_utils import save_weather_params
                
                result = save_weather_params(
                    weather_states=data.get('weather_states', []),
                    weather_presets=data.get('weather_presets', {}),
                    weather_sets=data.get('weather_sets', {}),
                    global_parameters=data.get('global_parameters', [])
                )
                
                # Clear cache since data has changed
                self.clear_weather_data_cache()
                
                if result['success']:
                    return jsonify(result)
                else:
                    return jsonify(result), 500
                    
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/weather_editor/reload', methods=['POST'])
        def reload_weather_module():
            """Reload the weather_params module to reflect saved changes."""
            try:
                import importlib
                from lib import weather_params
                importlib.reload(weather_params)
                
                # Clear cache since module has been reloaded
                self.clear_weather_data_cache()
                
                return jsonify({
                    "success": True,
                    "message": "Weather parameters reloaded"
                })
                    
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/weather_editor/validate', methods=['POST'])
        def validate_weather_data():
            """Validate weather data without saving."""
            try:
                data = request.json
                from lib.weather_editor_utils import validate_weather_params
                
                result = validate_weather_params(
                    weather_states=data.get('weather_states', []),
                    weather_presets=data.get('weather_presets', {}),
                    weather_sets=data.get('weather_sets', {})
                )
                
                return jsonify(result)
                    
            except Exception as e:
                return jsonify({
                    "valid": False,
                    "errors": [str(e)]
                }), 500
    
    def clear_weather_data_cache(self):
        """Clear cached weather data (call after saving changes)."""
        if hasattr(self, '_weather_data_cache'):
            delattr(self, '_weather_data_cache')

    def _setup_socketio_events(self):
        """Register WebSocket event handlers for real-time client communication."""

        @self.socketio.on('connect')
        def handle_connect():
            # Send initial state snapshot on connect
            with self._dict_lock:
                snapshot = {
                    "global_modifiers": dict(self.global_modifiers),
                    "active_overrides": dict(self.web_param_overrides),
                    "weather_params": self.control_dict.get('weather_params_snapshot', {}),
                    "transition": self.control_dict.get('transition_state', {}),
                    "audio_summary": self.control_dict.get('audio_summary', {}),
                    "fps": self.control_dict.get('fps', 0),
                    "current_weather_set": self.control_dict.get('current_weather_set', 'unknown'),
                    "current_weather": self.control_dict.get('current_weather', 'unknown'),
                    "season": self.control_dict.get('season', 0.0),
                    "brightness_limiting_factor": self.control_dict.get('brightness_limiting_factor', 1.0),
                    "active_effects": self.control_dict.get('active_effects', []),
                }
            emit('state_update', _sanitize_for_json(snapshot))

        @self.socketio.on('subscribe_preview')
        def handle_subscribe_preview():
            join_room('preview')
            with self._dict_lock:
                self.control_dict['_preview_subscribers'] = (
                    self.control_dict.get('_preview_subscribers', 0) + 1
                )
            print(f"[WebController] Preview client subscribed")

        @self.socketio.on('unsubscribe_preview')
        def handle_unsubscribe_preview():
            leave_room('preview')
            with self._dict_lock:
                self.control_dict['_preview_subscribers'] = max(
                    0, self.control_dict.get('_preview_subscribers', 0) - 1
                )

        @self.socketio.on('disconnect')
        def handle_disconnect():
            with self._dict_lock:
                if self.control_dict.get('_preview_subscribers', 0) > 0:
                    self.control_dict['_preview_subscribers'] -= 1

        @self.socketio.on('update_global')
        def handle_update_global(data):
            modifier = data.get('modifier')
            value = data.get('value')
            if modifier in self.global_modifier_schema:
                schema = self.global_modifier_schema[modifier]
                value = max(schema["min"], min(schema["max"], float(value)))
                with self._dict_lock:
                    self.global_modifiers[modifier] = value
                self._values_cache = None

        @self.socketio.on('set_override')
        def handle_set_override(data):
            param = data.get('param')
            value = data.get('value')
            if param is not None and value is not None:
                with self._dict_lock:
                    self.web_param_overrides[param] = float(value)
                self._values_cache = None

        @self.socketio.on('clear_override')
        def handle_clear_override(data):
            param = data.get('param')
            with self._dict_lock:
                self.web_param_overrides.pop(param, None)
            self._values_cache = None

        @self.socketio.on('clear_all_overrides')
        def handle_clear_all_overrides(data=None):
            with self._dict_lock:
                self.web_param_overrides.clear()
            self._values_cache = None

        @self.socketio.on('set_season')
        def handle_set_season(data):
            """Set the manual season value and lock auto-advance."""
            value = data.get('value')
            locked = data.get('locked')
            with self._dict_lock:
                if value is not None:
                    self.control_dict['season_override'] = float(value) % 1.0
                if locked is not None:
                    self.control_dict['season_locked'] = bool(locked)
            self._values_cache = None

        # Allowed keys for the set_flag WebSocket event
        ALLOWED_FLAGS = {'instant_transitions', 'flip_x'}

        @self.socketio.on('set_flag')
        def handle_set_flag(data):
            key = data.get('key')
            value = data.get('value')
            if key in ALLOWED_FLAGS:
                with self._dict_lock:
                    self.control_dict[key] = bool(value)
                self._values_cache = None
                # Apply flip_x directly to renderer viewports (avoid per-frame polling)
                if key == 'flip_x':
                    viewports = self.control_dict.get('_viewports')
                    if viewports:
                        for vp in viewports:
                            vp.flip_x = bool(value)

        @self.socketio.on('change_weather_set')
        def handle_change_set(data):
            new_set = data.get('set_name')
            if new_set:
                with self._dict_lock:
                    available = self.control_dict.get('available_sets', [])
                if new_set in available:
                    with self._dict_lock:
                        self.control_dict['request_weather_set'] = new_set
                    self._values_cache = None

        @self.socketio.on('change_weather_state')
        def handle_change_state(data):
            new_state = data.get('state_name')
            if new_state:
                with self._dict_lock:
                    locked = self.control_dict.get('state_switch_locked', True)
                    valid_states = (
                        self.control_dict.get('available_weather_states', [])
                        if locked
                        else self.control_dict.get('all_weather_states', [])
                    )
                if new_state in valid_states:
                    with self._dict_lock:
                        self.control_dict['request_weather_state'] = new_state
                    self._values_cache = None

        @self.socketio.on('trigger_random_event')
        def handle_trigger_random_event(data=None):
            event_name = data.get('event_name', '') if data else ''
            with self._dict_lock:
                self.control_dict['request_trigger_event'] = event_name or True
            self._values_cache = None

    def _start_emitter(self):
        """Start the background thread that pushes state updates via WebSocket."""
        self._emitter_running = True

        def emitter_loop():
            state_interval = 0.2    # 200ms for state updates (5 Hz)
            audio_interval = 0.033  # ~33ms for audio updates (30 Hz)
            frame_interval = 0.066  # ~66ms for frame updates (15 Hz)
            last_state_push = 0
            last_audio_push = 0
            last_frame_push = 0
            last_weather = None
            last_set = None

            while self._emitter_running:
                now = time.time()

                # Push state updates at 5 Hz
                if now - last_state_push >= state_interval:
                    last_state_push = now
                    try:
                        with self._dict_lock:
                            snapshot = {
                                "global_modifiers": dict(self.global_modifiers),
                                "active_overrides": dict(self.web_param_overrides),
                                "weather_params": copy.copy(self.control_dict.get('weather_params_snapshot', {})),
                                "transition": copy.copy(self.control_dict.get('transition_state', {})),
                                "fps": self.control_dict.get('fps', 0),
                                "current_weather_set": self.control_dict.get('current_weather_set', 'unknown'),
                                "current_weather": self.control_dict.get('current_weather', 'unknown'),
                                "season": self.control_dict.get('season', 0.0),
                                "season_locked": self.control_dict.get('season_locked', False),
                                "brightness_limiting_factor": self.control_dict.get('brightness_limiting_factor', 1.0),
                                "active_effects": list(self.control_dict.get('active_effects', [])),
                                "ambient_sound": self.control_dict.get('ambient_sound'),
                                "allowed_output_params": self.control_dict.get('allowed_output_params'),
                            }

                        # Emit to all connected clients
                        snapshot = _sanitize_for_json(snapshot)
                        self.socketio.emit('state_update', snapshot)

                        # Detect weather set/state changes for event-driven push
                        cur_weather = snapshot.get('current_weather')
                        cur_set = snapshot.get('current_weather_set')
                        if cur_weather != last_weather or cur_set != last_set:
                            if last_weather is not None:  # skip initial
                                self.socketio.emit('weather_changed', {
                                    'weather': cur_weather,
                                    'set': cur_set,
                                    'previous_weather': last_weather,
                                    'previous_set': last_set,
                                })
                            last_weather = cur_weather
                            last_set = cur_set

                    except Exception as e:
                        print(f"[WebController] Emitter state error: {e}")

                # Push audio updates at 10 Hz
                if now - last_audio_push >= audio_interval:
                    last_audio_push = now
                    try:
                        with self._dict_lock:
                            audio = copy.copy(self.control_dict.get('audio_summary', {}))
                        if audio:
                            self.socketio.emit('audio_update', _sanitize_for_json(audio))
                    except Exception as e:
                        print(f"[WebController] Emitter audio error: {e}")

                # Push preview frames at 15 Hz to subscribed clients
                if now - last_frame_push >= frame_interval:
                    last_frame_push = now
                    try:
                        frame_png = self.control_dict.get('_frame_png')
                        if frame_png is not None:
                            self.socketio.emit('frame', frame_png, room='preview')
                    except Exception as e:
                        print(f"[WebController] Emitter frame error: {e}")

                # Sleep a short interval to avoid busy-waiting
                time.sleep(0.015)

        self._emitter_thread = threading.Thread(target=emitter_loop, daemon=True)
        self._emitter_thread.start()

    def start(self, threaded=True):
        """
        Start the web server.

        Args:
            threaded: If True, run server in a separate thread (non-blocking)
        """
        # Register mDNS service
        self._register_mdns()

        # Start the WebSocket emitter thread
        self._start_emitter()

        if threaded:
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self.server_thread.start()
            print(f"[WebController] Started at:")
            print(f"[WebController]   http://localhost:{self.port}")
            print(f"[WebController]   http://{self.service_name}.local:{self.port}")
        else:
            self._run_server()

    def _run_server(self):
        """Internal method to run the Flask+SocketIO server."""
        # Disable Flask request logging
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        self.socketio.run(self.app, host='0.0.0.0', port=self.port,
                          debug=False, use_reloader=False, log_output=False,
                          allow_unsafe_werkzeug=True)
    
    def _register_mdns(self):
        """Register the service with mDNS/Bonjour for easy discovery."""
        # Do this in background thread - it can be slow
        def register_async():
            try:
                # Use bind_ip if configured, otherwise auto-detect
                if self.bind_ip:
                    local_ip = self.bind_ip
                else:
                    hostname = socket.gethostname()
                    try:
                        local_ip = socket.gethostbyname(hostname)
                    except:
                        local_ip = '127.0.0.1'

                # Create service info
                service_type = "_http._tcp.local."
                service_name = f"{self.service_name}.{service_type}"

                # Restrict Zeroconf to specific interface if bind_ip is set
                if self.bind_ip:
                    from zeroconf import InterfaceChoice
                    self.zeroconf = Zeroconf(interfaces=[self.bind_ip])
                    print(f"[WebController] mDNS restricted to interface {self.bind_ip}")
                else:
                    self.zeroconf = Zeroconf()

                self.service_info = ServiceInfo(
                    service_type,
                    service_name,
                    addresses=[socket.inet_aton(local_ip)],
                    port=self.port,
                    properties={
                        'path': '/',
                        'description': 'GL_Simple Control Panel'
                    },
                    server=f"{self.service_name}.local."
                )

                try:
                    self.zeroconf.register_service(self.service_info)
                    print(f"[WebController] mDNS registered as '{self.service_name}.local' ({local_ip})")
                except Exception as e:
                    print(f"[WebController] Warning: mDNS registration failed: {e}")

            except Exception as e:
                print(f"[WebController] Warning: Could not register mDNS service: {e}")
                print("[WebController] Service still accessible via IP address")

        # Start async registration in background
        mdns_thread = threading.Thread(target=register_async, daemon=True)
        mdns_thread.start()
    def stop(self):
        """Stop the web server, emitter, and unregister mDNS service."""
        self._emitter_running = False
        if self.zeroconf and self.service_info:
            try:
                self.zeroconf.unregister_service(self.service_info)
                self.zeroconf.close()
                print("[WebController] mDNS unregistered")
            except Exception as e:
                print(f"[WebController] Warning: Error unregistering mDNS service: {e}")
    
    def set_available_events(self, all_events=None, background_events=None):
        """
        Set the list of available event names from the event_map.
        
        Args:
            all_events: List of all event names (for on-transition events)
            background_events: List of events suitable for background use (for weather sets)
        """
        if all_events is not None:
            self.available_events = all_events
        if background_events is not None:
            self.available_background_events = background_events
        
        # Clear the weather data cache so it gets regenerated with new events
        if hasattr(self, '_weather_data_cache'):
            delattr(self, '_weather_data_cache')
    
    def get(self, key, default=None):
        """Get a value from the control dictionary."""
        with self._dict_lock:
            return self.control_dict.get(key, default)
    
    def set(self, key, value):
        """Set a value in the control dictionary."""
        with self._dict_lock:
            self.control_dict[key] = value
        self._values_cache = None  # Invalidate cache
    
    def __getitem__(self, key):
        """Allow dictionary-style access."""
        with self._dict_lock:
            return self.control_dict[key]
    
    def __setitem__(self, key, value):
        """Allow dictionary-style setting."""
        with self._dict_lock:
            self.control_dict[key] = value
        self._values_cache = None  # Invalidate cache


if __name__ == "__main__":
    # Example usage
    control_dict = {}
    controller = WebController(control_dict)

    # Start the server (blocking mode for standalone testing)
    controller.start(threaded=False)
