"""
Web-based control system for GL_Simple environmental effects.
Provides a Flask web server with real-time control interface.
"""
import threading
import socket
import hashlib
import secrets
from flask import Flask, render_template, jsonify, request, session
from pathlib import Path
import json
from zeroconf import Zeroconf, ServiceInfo


class WebController:
    """
    Web-based control interface for modifying runtime parameters.
    Thread-safe dictionary updates with real-time web interface.
    """
    
    def __init__(self, control_dict=None, port=5000, service_name="glsimple", admin_password=None):
        """
        Initialize the web controller.
        
        Args:
            control_dict: Dictionary to be controlled (default: creates new dict)
            port: Port number for the web server (default: 5000)
            service_name: mDNS service name (default: "glsimple")
                         Will be accessible at http://{service_name}.local:{port}
            admin_password: Password for admin panel (default: None = no admin access)
        """
        self.control_dict = control_dict if control_dict is not None else {}
        self.port = port
        self.service_name = service_name
        
        # Add thread lock for thread-safe dictionary access
        self._dict_lock = threading.RLock()
        
        # Cache for frequently accessed values to reduce lock contention
        self._values_cache = None
        self._values_cache_time = 0
        self._cache_duration = 0.1  # Cache for 100ms
        
        self.app = Flask(__name__, 
                        template_folder=str(Path(__file__).parent.parent / 'templates'))
        
        # Configure Flask session with a secret key
        self.app.secret_key = secrets.token_hex(32)
        
        # Hash the admin password if provided
        self.available_events = []  # Will be set by EnvironmentalSystem
        self.admin_password_hash = None
        if admin_password:
            self.admin_password_hash = hashlib.sha256(admin_password.encode()).hexdigest()
        
        self.server_thread = None
        self.zeroconf = None
        self.service_info = None
        self._setup_routes()
        
        # Default control schema - defines what controls are available
        self.control_schema = {
            "weather_intensity": {
                "type": "slider",
                "min": 0.0,
                "max": 2.0,
                "step": 0.1,
                "default": 1.0,
                "label": "Weather Intensity"
            },
            "fog_strength": {
                "type": "slider",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "default": 0.3,
                "label": "Fog Strength"
            },
            "rain_amount": {
                "type": "slider",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "default": 0.5,
                "label": "Rain Amount"
            },
            "audio_sensitivity": {
                "type": "slider",
                "min": 0.1,
                "max": 3.0,
                "step": 0.1,
                "default": 1.0,
                "label": "Audio Sensitivity"
            },
            "enable_fireflies": {
                "type": "checkbox",
                "default": True,
                "label": "Enable Fireflies"
            },
            "enable_stars": {
                "type": "checkbox",
                "default": True,
                "label": "Enable Stars"
            },
            "color_mode": {
                "type": "select",
                "options": ["default", "warm", "cool", "monochrome"],
                "default": "default",
                "label": "Color Mode"
            },
            "effect_speed": {
                "type": "slider",
                "min": 0.1,
                "max": 5.0,
                "step": 0.1,
                "default": 1.0,
                "label": "Effect Speed Multiplier"
            }
        }
        
        # Initialize control_dict with defaults
        self._init_defaults()
    
    def _init_defaults(self):
        """Initialize control dictionary with default values from schema."""
        for key, config in self.control_schema.items():
            if key not in self.control_dict:
                self.control_dict[key] = config["default"]
    
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
        
        @self.app.route('/api/schema')
        def get_schema():
            """Return the control schema for dynamic UI generation."""
            return jsonify(self.control_schema)
        
        @self.app.route('/api/values')
        def get_values():
            """Return current values of all controls."""
            # Use cached values if available and fresh
            import time
            current_time = time.time()
            
            if (self._values_cache is not None and 
                current_time - self._values_cache_time < self._cache_duration):
                return self._values_cache
            
            # Create a snapshot with lock to avoid holding it during JSON serialization
            with self._dict_lock:
                values_snapshot = self.control_dict.copy()
            
            # Serialize outside the lock
            response = jsonify(values_snapshot)
            
            # Cache the response
            self._values_cache = response
            self._values_cache_time = current_time
            
            return response
        
        @self.app.route('/api/update', methods=['POST'])
        def update_value():
            """Update a control value."""
            data = request.json
            key = data.get('key')
            value = data.get('value')
            
            if key in self.control_schema:
                # Convert value to appropriate type
                schema_type = self.control_schema[key]["type"]
                if schema_type == "checkbox":
                    value = bool(value)
                elif schema_type in ["slider", "number"]:
                    value = float(value)
                
                with self._dict_lock:
                    self.control_dict[key] = value
                
                # Invalidate cache
                self._values_cache = None
                
                return jsonify({"success": True, "key": key, "value": value})
            
            return jsonify({"success": False, "error": "Unknown key"}), 400
        
        @self.app.route('/api/batch_update', methods=['POST'])
        def batch_update():
            """Update multiple control values at once."""
            data = request.json
            updated = {}
            
            # Prepare updates outside lock
            for key, value in data.items():
                if key in self.control_schema:
                    schema_type = self.control_schema[key]["type"]
                    if schema_type == "checkbox":
                        value = bool(value)
                    elif schema_type in ["slider", "number"]:
                        value = float(value)
                    updated[key] = value
            
            # Apply all updates at once under lock
            with self._dict_lock:
                self.control_dict.update(updated)
            
            # Invalidate cache
            self._values_cache = None
            
            return jsonify({"success": True, "updated": updated})
        
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
        
        @self.app.route('/api/weather_set/info')
        def weather_set_info():
            """Get current weather set information."""
            with self._dict_lock:
                info = {
                    "current_set": self.control_dict.get('current_weather_set', 'unknown'),
                    "available_sets": self.control_dict.get('available_sets', []),
                    "current_weather": self.control_dict.get('current_weather', 'unknown'),
                    "season": self.control_dict.get('season', 0.0)
                }
            return jsonify(info)
        
        @self.app.route('/api/weather_editor/all_data')
        def get_all_weather_data():
            """Get all weather states, presets, and sets for editing."""
            # Cache this expensive operation
            if not hasattr(self, '_weather_data_cache'):
                from corefunctions.weather_params import (
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
                from corefunctions.weather_editor_utils import save_weather_params
                
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
                from corefunctions import weather_params
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
                from corefunctions.weather_editor_utils import validate_weather_params
                
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
    
    def add_control(self, key, control_type, label, **kwargs):
        """
        Add a new control to the schema.
        
        Args:
            key: Unique identifier for the control
            control_type: Type of control ('slider', 'checkbox', 'select', 'number')
            label: Display label for the control
            **kwargs: Additional parameters (min, max, step, default, options, etc.)
        """
        self.control_schema[key] = {
            "type": control_type,
            "label": label,
            **kwargs
        }
        
        # Set default value if provided and not already in dict
        if "default" in kwargs and key not in self.control_dict:
            self.control_dict[key] = kwargs["default"]
    
    def start(self, threaded=True):
        """
        Start the web server.
        
        Args:
            threaded: If True, run server in a separate thread (non-blocking)
        """
        # Register mDNS service
        self._register_mdns()
        
        if threaded:
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self.server_thread.start()
            print(f"Web control panel started at:")
            print(f"  - http://localhost:{self.port}")
            print(f"  - http://{self.service_name}.local:{self.port}")
        else:
            self._run_server()
    
    def _run_server(self):
        """Internal method to run the Flask server."""
        # Disable Flask request logging
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        self.app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False)
    
    def _register_mdns(self):
        """Register the service with mDNS/Bonjour for easy discovery."""
        # Do this in background thread - it can be slow
        def register_async():
            try:
                # Get local IP address (use a timeout to avoid blocking)
                hostname = socket.gethostname()
                # Try to get IP quickly, fallback to 127.0.0.1
                try:
                    local_ip = socket.gethostbyname(hostname)
                except:
                    local_ip = '127.0.0.1'
                
                # Create service info
                service_type = "_http._tcp.local."
                service_name = f"{self.service_name}.{service_type}"
                
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
                
                # Register in a separate thread to avoid blocking
                def register_with_timeout():
                    try:
                        self.zeroconf.register_service(self.service_info)
                        print(f"mDNS service registered as '{self.service_name}.local'")
                    except Exception as e:
                        print(f"Warning: mDNS registration failed: {e}")
                
                register_with_timeout()
                
            except Exception as e:
                print(f"Warning: Could not register mDNS service: {e}")
                print("Service will still be accessible via IP address")
        
        # Start async registration in background
        mdns_thread = threading.Thread(target=register_async, daemon=True)
        mdns_thread.start()
    def stop(self):
        """Stop the web server and unregister mDNS service."""
        if self.zeroconf and self.service_info:
            try:
                self.zeroconf.unregister_service(self.service_info)
                self.zeroconf.close()
                print("mDNS service unregistered")
            except Exception as e:
                print(f"Warning: Error unregistering mDNS service: {e}")
    
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
    
    # Add custom controls
    controller.add_control(
        "custom_param",
        "slider",
        "Custom Parameter",
        min=0,
        max=100,
        step=1,
        default=50
    )
    
    # Start the server (blocking mode for standalone testing)
    controller.start(threaded=False)
