"""
Web-based control system for GL_Simple environmental effects.
Provides a Flask web server with real-time WebSocket control interface.
"""
import threading
import socket
import time
import copy
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from pathlib import Path
import json
from zeroconf import Zeroconf, ServiceInfo
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
    
    def __init__(self, control_dict=None, port=5000, service_name="lucifera", bind_ip="", geometry_provider=None):
        """
        Initialize the web controller.

        Args:
            control_dict: Dictionary to be controlled (default: creates new dict)
            port: Port number for the web server (default: 5000)
            service_name: mDNS service name (default: "lucifera")
                         Will be accessible at http://{service_name}.local:{port}
            geometry_provider: Active GeometryProvider for /api/preview/geometry.
                If None, a default FanGeometryProvider is built lazily so the
                preview keeps working in standalone use.
        """
        self.control_dict = control_dict if control_dict is not None else {}
        self.port = port
        self.service_name = service_name
        self.bind_ip = bind_ip
        self._geometry_provider = geometry_provider
        
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

        self.available_events = []  # Will be set by EnvironmentalSystem

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
            "audio_sensitivity": 1.0,
            "brightness": 1.0,
            "gamma": 2.0,
            "brightness_limit": 0.1,
            "master_volume": 1.0,
            "narrative_volume": 1.0,
            "soundpool_volume": 1.0,
            "audio_balance": 0.0,
        }

        # Global modifier metadata for UI generation
        self.global_modifier_schema = {
            "weather_intensity": {
                "label": "Weather Intensity",
                "min": 0.0, "max": 2.0, "step": 0.05, "default": 1.0,
                "description": "Scales dramatic weather effects (rain, wind, lightning, etc.)"
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
            "gamma": {
                "label": "Gamma",
                "min": 1.0, "max": 3.0, "step": 0.05, "default": 2.0,
                "description": "Output gamma correction exponent (applied before brightness limiter)"
            },
            "brightness_limit": {
                # SAFETY CEILING — this max is a hardware guard, not a UI
                # preference. Above the active piece's declared budget the
                # PSUs are over-drawn, so the slider must not be able to
                # go there: a fat-fingered drag has to stop at the safe
                # value. ``max`` is therefore rewritten per project at
                # boot and on swap (set_project_ceiling), to exactly that
                # project's ``brightness_limit:`` — Fan 0.1, WoL 0.4. The
                # 0.1 here is the conservative fallback for a project that
                # declares nothing, matching RenderPipeline's own default.
                # The API clamps every write to this range, and the render
                # pipeline clamps again against the project value, so no
                # single mistake can raise the ceiling.
                "label": "Brightness Limit",
                "min": 0.0, "max": 0.1, "step": 0.005, "default": 0.1,
                "description": "Power limiter threshold, as a fraction of full-scale output. The maximum is the active project's PSU budget and cannot be exceeded from here; lower values cap output harder to save power"
            },
            "master_volume": {
                "label": "Master Volume",
                "min": 0.0, "max": 2.0, "step": 0.05, "default": 1.0,
                "description": "Controls all audio output volume"
            },
            "narrative_volume": {
                "label": "Narrative Volume",
                "min": 0.0, "max": 4.0, "step": 0.05, "default": 1.0,
                "description": "Controls narrative player TTS volume"
            },
            "soundpool_volume": {
                "label": "Soundpool Volume",
                "min": 0.0, "max": 4.0, "step": 0.05, "default": 1.0,
                "description": "Controls random sound-pool clip volume (independent of master/narrative/ambient)"
            },
            "audio_balance": {
                "label": "Balance",
                "min": -1.0, "max": 1.0, "step": 0.05, "default": 0.0,
                "description": "Stereo balance: 0 = centered, -1 = left speaker only, +1 = right speaker only (opposite side fades linearly)"
            },
        }

        # Weather parameter overrides — direct value overrides (param_name -> value)
        self.web_param_overrides = {}

        # Timed set switch (DJ panel): {'hours', 'target', 'end'} while a
        # countdown is armed, None when off. Fired by the emitter loop.
        self._set_switch_timer = None

        # Per-weather-set interaction panels (set id -> normalized panel).
        # Pushed by EnvironmentalSystem from the active project's
        # ``interaction`` hook; empty means no set offers an interaction
        # tab, which is the default for a project that declares none.
        self._interaction_panels = {}

        # Preview frame streaming
        # Preview clients are tracked via Socket.IO 'preview' room
        self._geometry_json = None  # cached provider.to_json()

    def replace_geometry_provider(self, provider):
        """Hot-swap the active geometry provider (used by project-swap)."""
        self._geometry_provider = provider
        self._geometry_json = None

    def set_interaction_panels(self, panels):
        """Install the active project's normalized interaction panels.

        Called by ``EnvironmentalSystem`` at boot and after every project
        swap. ``panels`` maps weather-set id -> normalized panel (see
        ``lib.interaction``); sets absent from the mapping simply get no
        interaction tab, which is the default.
        """
        with self._dict_lock:
            self._interaction_panels = dict(panels or {})

    def _active_interaction_panel(self):
        """Panel for the live weather set, or None.

        Applies the panel's ``requires`` gate (the club panel points at
        the DJ page and asks for ``dj``, so the tab stays hidden when the
        DJ subsystem isn't available even though club is running)."""
        from lib.interaction import panel_for_set
        with self._dict_lock:
            panels = getattr(self, '_interaction_panels', {})
            cur = self.control_dict.get('current_weather_set')
            snapshot = {'dj_info': self.control_dict.get('dj_info'),
                        'gen_info': self.control_dict.get('gen_info')}
        return panel_for_set(panels, cur, snapshot)

    def set_weather_module(self, *, weather_state_enum=None,
                           weather_presets=None, weather_sets=None,
                           default_weather_params=None,
                           default_weather_set=None,
                           weather_module_path=None,
                           weather_module_name=None):
        """Push the active project's weather data into the web UI's
        backing store.

        Called by ``EnvironmentalSystem`` at boot and after every
        project swap so the weather editor's ``/api/weather/all``
        endpoint surfaces the active project's enum / presets / sets
        rather than the lib-level globals (which are a union of
        every project's contributions and would confuse the UI).

        ``weather_module_path`` is the filesystem path to the active
        project's ``weather_params.py`` (typically
        ``projects/<id>/weather_params.py``). The save endpoint writes
        edited data back to this path so per-project overrides
        (WEATHER_SETS, etc.) actually take effect — saves to the
        lib-level file get clobbered by the project's runtime
        override.

        ``weather_module_name`` is the importable module name
        (e.g. ``projects.weight_of_light.weather_params``); the
        reload endpoint uses it to refresh the right module.

        Any argument left as ``None`` keeps the previous value.
        """
        if weather_state_enum is not None:
            self._project_weather_state_enum = weather_state_enum
        if weather_presets is not None:
            self._project_weather_presets = weather_presets
        if weather_sets is not None:
            self._project_weather_sets = weather_sets
        if default_weather_params is not None:
            self._project_default_weather_params = default_weather_params
        if default_weather_set is not None:
            self._project_default_weather_set = default_weather_set
        if weather_module_path is not None:
            self._project_weather_module_path = str(weather_module_path)
        if weather_module_name is not None:
            self._project_weather_module_name = str(weather_module_name)
        # Drop the cached weather_data response so the next GET sees
        # the new project. set_available_events does the same thing
        # for its dimension; both are needed.
        if hasattr(self, '_weather_data_cache'):
            delattr(self, '_weather_data_cache')

    def _resolve_active_media_sounds_dir(self):
        """Return the active project's ``media/sounds`` directory, or fall
        back to the legacy repo-root ``media/sounds`` for callers that
        haven't set ``control_dict['media_root']`` yet (tests, standalone)."""
        from pathlib import Path
        media_root = self.control_dict.get('media_root') if self.control_dict else None
        if media_root:
            return Path(media_root) / 'sounds'
        return Path(__file__).parent.parent / 'media' / 'sounds'
    
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
        
        @self.app.route('/club')
        def club_panel():
            return render_template('club_panel.html')

        @self.app.route('/api/club/active')
        def club_active():
            """Whether the club set is live (gates the Club nav tab)."""
            with self._dict_lock:
                cur = self.control_dict.get('current_weather_set')
            return jsonify({"active": cur == 'club'})

        @self.app.route('/dj')
        def dj_panel():
            return render_template('dj_panel.html')

        @self.app.route('/gen')
        def gen_panel():
            """Generative music control surface (lib/gen) - the DJ's
            sibling: its own page, its own action whitelist."""
            return render_template('gen_panel.html')

        @self.app.route('/api/gen/surface')
        def gen_surface():
            """The declarative UI spec the /gen page renders (lib/gen/ui.py)."""
            from lib.gen.ui import surface_spec
            return jsonify(surface_spec())

        @self.app.route('/api/gen/status')
        def gen_status():
            """The full gen_info snapshot (what the socket ships at 5 Hz), for
            the native console's remote mode (tools/gen_console.py --remote)."""
            with self._dict_lock:
                info = self.control_dict.get('gen_info') or {"available": False, "active": False}
            return jsonify(_sanitize_for_json(info))

        @self.app.route('/api/gen/active')
        def gen_active():
            """Generative subsystem availability/liveness (gates the nav
            tab, same idiom as /api/dj/active)."""
            with self._dict_lock:
                info = self.control_dict.get('gen_info') or {}
            return jsonify({"available": bool(info.get("available")),
                            "active": bool(info.get("active")),
                            "state": info.get("state") or "",
                            "style": info.get("style") or "",
                            "error": info.get("error") or ""})

        @self.app.route('/api/dj/active')
        def dj_active():
            """DJ availability/liveness (gates the DJ nav tab). Also the
            planner's Push-to-live VERIFICATION endpoint: enough state to
            confirm a pushed setlist actually landed (loaded name, mode,
            the show's library dir, last error) - a queued action that the
            show then rejects must not read as success."""
            with self._dict_lock:
                info = self.control_dict.get('dj_info') or {}
            return jsonify({"available": bool(info.get("available")),
                            "active": bool(info.get("active")),
                            "state": info.get("state") or "",
                            "setlist": info.get("setlist"),
                            "setlist_mode": info.get("setlist_mode") or "",
                            "music_dir": info.get("music_dir") or "",
                            "error": info.get("error") or ""})

        @self.app.route('/interaction')
        def interaction_panel():
            """Generic per-weather-set interaction page.

            Renders whatever the live set declared; shows a "nothing
            here" card when the set that owns the panel is no longer
            running (someone left the tab open through a set change)."""
            return render_template('interaction_panel.html')

        @self.app.route('/api/interaction/info')
        def interaction_info():
            """What the interaction nav slot should show right now.

            One endpoint drives the tab on every page: hidden when the
            live set declares no panel, labelled and pointed at the
            set's own page (``/dj`` for club) otherwise."""
            panel = self._active_interaction_panel()
            if panel is None:
                return jsonify({"available": False})
            with self._dict_lock:
                dj = self.control_dict.get('dj_info') or {}
                gen = self.control_dict.get('gen_info') or {}
            return jsonify({
                "available": True,
                "set": panel["set"],
                "label": panel["label"],
                "href": panel.get("page") or "/interaction",
                "accent": panel["theme"]["accent"],
                # "something is running behind this tab" — currently only
                # the DJ-backed panel has a liveness notion; it lights the
                # tab green exactly as the old dedicated DJ tab did.
                "live": (bool(dj.get("active")) if panel.get("requires") == "dj"
                         else bool(gen.get("active")) if panel.get("requires") == "gen"
                         else False),
            })

        @self.app.route('/api/interaction/spec')
        def interaction_spec():
            """Full panel spec + live values for the generic renderer."""
            from lib.interaction import public_panel
            panel = self._active_interaction_panel()
            if panel is None:
                with self._dict_lock:
                    cur = self.control_dict.get('current_weather_set', 'unknown')
                return jsonify({"available": False, "current_set": cur})
            with self._dict_lock:
                overrides = dict(self.web_param_overrides)
                signals = copy.deepcopy(
                    self.control_dict.get('interaction_signals', {}))
                current_state = self.control_dict.get('current_weather', '')
            return jsonify({
                "available": True,
                "panel": public_panel(panel),
                "values": {"params": overrides, "signals": signals},
                "current_state": current_state,
            })

        @self.app.route('/preview')
        def preview():
            """Serve the live preview page."""
            return render_template('preview.html')

        @self.app.route('/api/preview/geometry')
        def preview_geometry():
            """Return geometry JSON from the active GeometryProvider.

            The JS client branches on the top-level ``type`` field
            (currently ``"fan"``; multi-object lands later).
            """
            if self._geometry_json is None:
                if self._geometry_provider is None:
                    # Standalone fallback: rebuild a Fan provider from the
                    # LED dimensions cached in control_dict.
                    from core.geometry.fan import FanGeometryProvider
                    w = self.control_dict.get('led_width', 128)
                    h = self.control_dict.get('led_height', 300)
                    self._geometry_provider = FanGeometryProvider(
                        num_strips=w, num_leds=h, display_aspect=900 / 500,
                    )
                self._geometry_json = self._geometry_provider.to_json()
            return jsonify(self._geometry_json)

        @self.app.route('/api/project/info')
        def project_info():
            """Return active project + the list of switchable projects."""
            with self._dict_lock:
                current = self.control_dict.get('current_project', 'unknown')
                current_name = self.control_dict.get('current_project_name', current)
                available = self.control_dict.get('available_projects', [])
            return jsonify({
                "current": current,
                "current_name": current_name,
                "available": available,
            })

        @self.app.route('/api/project/change', methods=['POST'])
        def project_change():
            """Swap the active project. Mirrors the ``change_project``
            Socket.IO handler so HTTP clients (the layout editor) can
            request a swap without a Socket.IO dependency."""
            data = request.get_json(silent=True) or {}
            new_id = data.get('project_id')
            if not new_id:
                return jsonify({"success": False,
                                "error": "missing project_id"}), 400
            with self._dict_lock:
                avail = self.control_dict.get('available_projects', []) or []
                ids = {p.get('id') for p in avail if isinstance(p, dict)}
                current = self.control_dict.get('current_project')
            if new_id not in ids:
                return jsonify({"success": False,
                                "error": f"unknown project {new_id!r}"}), 400
            if new_id == current:
                return jsonify({"success": True, "current": current,
                                "note": "already active"})
            with self._dict_lock:
                self.control_dict['request_project_swap'] = new_id
            self._values_cache = None
            return jsonify({"success": True, "requested": new_id})

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
                    "narrative_vars": list(self.control_dict.get('narrative_vars', [])),
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
            print(f"[WebController] Weather state lock: {'ON' if locked else 'OFF'}")

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
                # Prefer the active project's stashed weather data
                # (pushed in via set_weather_module on boot / project
                # swap). Fall back to the lib-level globals only when
                # the engine hasn't wired anything yet — useful for
                # standalone tests that boot the WebController without
                # an EnvironmentalSystem.
                from lib.weather_params import (
                    WeatherState as _LIB_STATE_ENUM,
                    DEFAULT_WEATHER_PARAMS as _LIB_DEFAULT_PARAMS,
                    WEATHER_PRESETS as _LIB_PRESETS,
                    WEATHER_SETS as _LIB_SETS,
                    GLOBAL_PARAMETERS, PARAMETER_DEFINITIONS,
                )
                WeatherState = getattr(self, "_project_weather_state_enum",
                                       None) or _LIB_STATE_ENUM
                DEFAULT_WEATHER_PARAMS = getattr(
                    self, "_project_default_weather_params", None) \
                    or _LIB_DEFAULT_PARAMS
                WEATHER_PRESETS = getattr(self, "_project_weather_presets",
                                          None) or _LIB_PRESETS
                WEATHER_SETS = getattr(self, "_project_weather_sets",
                                       None) or _LIB_SETS
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
                
                # Get available sound files (only once). Resolves the
                # active project's media folder via the geometry provider's
                # group cache (set at startup) — falls back to the legacy
                # repo-root ``media/sounds`` if no project context.
                sound_files = []
                sounds_dir = self._resolve_active_media_sounds_dir()
                if sounds_dir.exists():
                    sound_files = [f.name for f in sounds_dir.iterdir() if f.is_file()]
                    sound_files.sort()

                # Discover narrative scripts: any media/sounds/*/script.json.
                # Each entry carries the repo-relative path (what gets stored
                # in WEATHER_SETS[*]["narrative_script"]) and a display name
                # pulled from the JSON's "name" field when present.
                import json as _json
                narrative_scripts = []
                sound_pool_dirs = []
                audio_exts = {'.mp3', '.wav', '.ogg', '.flac', '.m4a'}
                if sounds_dir.exists():
                    for sub in sorted(sounds_dir.iterdir()):
                        if not sub.is_dir():
                            continue
                        script_file = sub / 'script.json'
                        if script_file.is_file():
                            # Stored as media_root-relative so the narrative
                            # player resolves it against the active project's
                            # media folder. (Was ``media/sounds/...`` when
                            # all media lived at the repo root.)
                            rel_path = f"sounds/{sub.name}/script.json"
                            display = sub.name
                            try:
                                data = _json.loads(script_file.read_text(encoding='utf-8'))
                                display = data.get('name') or sub.name
                            except Exception:
                                pass  # Keep directory name as fallback label.
                            narrative_scripts.append({
                                "path": rel_path,
                                "name": display,
                            })

                        # Any subdir with audio files is eligible as a
                        # sound pool. Count non-recursively so the UI can
                        # show how many clips the user is selecting.
                        audio_count = sum(
                            1 for f in sub.iterdir()
                            if f.is_file() and f.suffix.lower() in audio_exts
                        )
                        if audio_count > 0:
                            sound_pool_dirs.append({
                                "path": f"sounds/{sub.name}",
                                "name": sub.name,
                                "count": audio_count,
                            })
                
                # Scope to the active project's relevant states.
                # WoL's weather_params.py inherits the lib-level
                # WeatherState enum (which carries every project's
                # values, hundreds of them), so iterating it raw
                # would surface forest/bartiki/ocean/etc. states in
                # the UI even when the project doesn't use them.
                # Compute the union of state values across this
                # project's WEATHER_SETS and filter the enum +
                # presets through that.
                _project_state_values = set()
                for _set in WEATHER_SETS.values():
                    if isinstance(_set, dict):
                        for _sv in (_set.get("states") or []):
                            _project_state_values.add(_sv)

                if _project_state_values:
                    weather_states = [
                        s.value for s in WeatherState
                        if s.value in _project_state_values
                    ]
                    presets = {}
                    for state, params in WEATHER_PRESETS.items():
                        state_key = state.value if hasattr(state, 'value') else str(state)
                        if state_key not in _project_state_values:
                            continue
                        params_copy = convert_to_json_serializable(params.copy())
                        presets[state_key] = params_copy
                else:
                    # No project sets configured (very early boot);
                    # fall back to the unfiltered enum + presets so
                    # standalone tests of the WebController still
                    # have something to render.
                    weather_states = [s.value for s in WeatherState]
                    presets = {}
                    for state, params in WEATHER_PRESETS.items():
                        state_key = state.value if hasattr(state, 'value') else str(state)
                        params_copy = convert_to_json_serializable(params.copy())
                        presets[state_key] = params_copy

                # Convert default params
                default_params = convert_to_json_serializable(DEFAULT_WEATHER_PARAMS.copy())

                # Convert weather sets
                weather_sets = convert_to_json_serializable(WEATHER_SETS.copy())
                
                # Background events: prefer the active project's dynamic
                # list (pushed by ``EnvironmentalSystem.set_available_events``
                # at boot + on project swap). Fall back to an empty list
                # if the engine hasn't wired one in yet — using lib's
                # static list would surface Fan-flavored event names
                # on a WoL boot before set_available_events runs, which
                # was a subtle source of stale dropdown options. An
                # empty fallback shows "no options yet" instead.
                background_events = (
                    sorted(self.available_background_events)
                    if hasattr(self, 'available_background_events')
                    else []
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
                    "available_narratives": narrative_scripts,
                    "available_sound_pools": sound_pool_dirs,
                    "available_events": sorted(self.available_events) if hasattr(self, 'available_events') else []
                }
            
            return jsonify(self._weather_data_cache)
        
        @self.app.route('/api/weather_editor/save', methods=['POST'])
        def save_weather_data():
            """Save modified weather data back to the active project's
            weather_params.py.

            The target file path is the active project's module
            (pushed in via ``set_weather_module(weather_module_path=...)``
            on engine boot / project swap). Falls back to the lib-level
            file only when no project has been wired in — kept that way
            so standalone WebController tests still write somewhere
            sensible.
            """
            try:
                data = request.json
                from lib.weather_editor_utils import save_weather_params

                target = getattr(self, "_project_weather_module_path", None)

                result = save_weather_params(
                    weather_states=data.get('weather_states', []),
                    weather_presets=data.get('weather_presets', {}),
                    weather_sets=data.get('weather_sets', {}),
                    global_parameters=data.get('global_parameters', []),
                    target_path=target,
                )

                if result['success']:
                    # Re-import the project's weather module and re-stash
                    # the controller's in-memory references so the next
                    # /load returns the values we just wrote. Without
                    # this, the cache rebuild reads stale references
                    # captured at engine boot — the editor saw "Save"
                    # succeed but then displayed the original values
                    # because /load served pre-edit data.
                    try:
                        self._refresh_weather_module()
                    except Exception as refresh_err:
                        # Save succeeded; only the in-memory refresh
                        # failed. Surface as a warning but don't roll
                        # the save back — the file is correct, just
                        # may need a manual /reload to pick up.
                        result["refresh_warning"] = str(refresh_err)
                    return jsonify(result)
                else:
                    self.clear_weather_data_cache()
                    return jsonify(result), 500

            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500

        @self.app.route('/api/weather_editor/reload', methods=['POST'])
        def reload_weather_module():
            """Reload the active project's weather_params module so
            the editor sees just-saved changes. The engine still needs
            a separate restart to pick up new presets/sets at runtime
            (no in-flight reload of the WeatherStateController + scheduler
            yet)."""
            try:
                self._refresh_weather_module()
                proj_mod_name = getattr(self, "_project_weather_module_name", None)
                return jsonify({
                    "success": True,
                    "message": "Weather parameters reloaded "
                               + (f"({proj_mod_name})" if proj_mod_name else "(lib only)")
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
    
    def _refresh_weather_module(self):
        """Reload the active project's weather module and re-stash the
        controller's in-memory references to its WEATHER_PRESETS,
        WEATHER_SETS, etc. so the next /api/weather_editor/load returns
        the freshly-saved values.

        Called from both the save endpoint (so saves are immediately
        visible) and the reload endpoint (operator-triggered refresh).
        Without re-stashing, ``importlib.reload`` only replaces the
        module's attributes in-place; the controller's
        ``self._project_weather_presets`` reference still points at
        the OLD dict captured at engine boot, and the cache rebuild
        serves stale data.
        """
        import importlib
        from lib import weather_params as _lib_wp
        importlib.reload(_lib_wp)

        proj_mod_name = getattr(self, "_project_weather_module_name", None)
        if proj_mod_name:
            proj_mod = importlib.import_module(proj_mod_name)
            importlib.reload(proj_mod)
            # Re-stash references from the reloaded module so cache
            # rebuilds pick up the new values.
            if hasattr(proj_mod, "WeatherState"):
                self._project_weather_state_enum = proj_mod.WeatherState
            if hasattr(proj_mod, "WEATHER_PRESETS"):
                self._project_weather_presets = proj_mod.WEATHER_PRESETS
            if hasattr(proj_mod, "WEATHER_SETS"):
                self._project_weather_sets = proj_mod.WEATHER_SETS
            if hasattr(proj_mod, "DEFAULT_WEATHER_PARAMS"):
                self._project_default_weather_params = proj_mod.DEFAULT_WEATHER_PARAMS

        self.clear_weather_data_cache()

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
                    "narrative_vars": list(self.control_dict.get('narrative_vars', [])),
                    "club": self.control_dict.get('club_director_info'),
                    "club_controls": {
                        "heat_bias": self.control_dict.get('club_heat_bias', 0.0),
                        "hold_room": self.control_dict.get('club_hold_room', False),
                        "palette_override": self.control_dict.get('club_palette_override'),
                        "density": self.control_dict.get('club_density', 1.0),
                        "theme": self.control_dict.get('club_theme', ''),
                        "ttls": self.control_dict.get('club_ttls', {}),
                    },
                    "dj": self.control_dict.get('dj_info'),
                                "gen": self.control_dict.get('gen_info'),
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

        @self.socketio.on('set_weather_time_scale')
        def handle_set_weather_time_scale(data):
            """Scale every weather state's dwell time (0.5x .. 2x)."""
            try:
                value = float((data or {}).get('value', 1.0))
            except (TypeError, ValueError):
                return
            with self._dict_lock:
                self.control_dict['weather_time_scale'] = max(0.5, min(2.0, value))
            self._values_cache = None

        @self.socketio.on('set_club_heat_bias')
        def handle_set_club_heat_bias(data):
            """Operator calmer/hotter bias for the club music director."""
            try:
                value = float((data or {}).get('value', 0.0))
            except (TypeError, ValueError):
                return
            with self._dict_lock:
                self.control_dict['club_heat_bias'] = max(-0.35, min(0.35, value))
            self._values_cache = None

        @self.socketio.on('set_club_hold')
        def handle_set_club_hold(data):
            """Hold the current club room (director keeps choreographing)."""
            with self._dict_lock:
                self.control_dict['club_hold_room'] = bool((data or {}).get('hold', False))
            self._values_cache = None

        @self.socketio.on('force_club_drop')
        def handle_force_club_drop(data=None):
            """Fire a drop moment now (one-shot; engine consumes the flag)."""
            with self._dict_lock:
                self.control_dict['club_force_drop'] = True

        @self.socketio.on('set_club_palette')
        def handle_set_club_palette(data):
            """Lock the club palette to a hue (value=None returns to auto)."""
            value = (data or {}).get('value')
            if value is not None:
                try:
                    value = float(value) % 1.0
                except (TypeError, ValueError):
                    return
            with self._dict_lock:
                self.control_dict['club_palette_override'] = value
            self._values_cache = None

        @self.socketio.on('set_club_density')
        def handle_set_club_density(data):
            """Scale every club pattern level (0.4 sparse .. 1.4 dense)."""
            try:
                value = float((data or {}).get('value', 1.0))
            except (TypeError, ValueError):
                return
            with self._dict_lock:
                self.control_dict['club_density'] = max(0.4, min(1.4, value))
            self._values_cache = None

        CLUB_THEMES = {'', 'organic', 'geometric', 'cosmic', 'heavy',
                       'molten', 'flow', 'psychedelic'}

        @self.socketio.on('set_club_theme')
        def handle_set_club_theme(data):
            """Lean the director's room picks toward a visual family."""
            theme = (data or {}).get('theme', '')
            if theme in CLUB_THEMES:
                with self._dict_lock:
                    self.control_dict['club_theme'] = theme
                self._values_cache = None

        DJ_ACTIONS = {'start', 'stop', 'skip', 'theme', 'autopilot',
                      'nudge', 'pulse', 'next_id', 'setlist', 'setlist_pool',
                      'seek', 'seek_rel', 'to_exit', 'mix_now', 'flavor',
                      'hold', 'reroll', 'seam_fb', 'seam_fb_edit', 'arc',
                      'moment', 'abort',
                      'persona', 'layer', 'set_length'}

        def queue_dj_action(data):
            """Validate + clamp one DJ control action and queue it for the
            app's 5 Hz bridge. Shared by the socketio event AND the HTTP
            route (the planner's Push-to-live) - one whitelist, one
            sanitizer. Returns True when the action was queued.

            All DJ controls funnel through ONE queued-action channel;
            Stories_OGL._apply_dj_controls drains it on the render thread
            where DJSystem lives.
            """
            data = data or {}
            action = data.get('action')
            if action not in DJ_ACTIONS:
                return False
            arg = data.get('value')
            if action in ('nudge', 'pulse'):
                try:
                    arg = max(-0.4, min(0.4, float(arg)))
                except (TypeError, ValueError):
                    return False
            elif action == 'next_id':
                try:
                    arg = int(arg)
                except (TypeError, ValueError):
                    return False
            elif action in ('theme', 'setlist', 'setlist_pool', 'persona'):
                if not isinstance(arg, str) or len(arg) > 80:
                    return False
            elif action == 'autopilot':
                arg = bool(arg)
            elif action in ('seek', 'seek_rel'):
                try:
                    arg = float(arg)
                except (TypeError, ValueError):
                    return False
            elif action == 'set_length':
                # Operator's set-length choice, seconds; same clamp the
                # engine applies (30 min .. 12 h).
                try:
                    arg = max(1800.0, min(43200.0, float(arg)))
                except (TypeError, ValueError):
                    return False
            elif action == 'seam_fb':
                arg = bool(arg)
            elif action == 'seam_fb_edit':
                # {n: tracklist entry sequence, fb: true/false/null(clear)}
                try:
                    fb = arg.get('fb')
                    arg = {'n': int(arg.get('n')),
                           'fb': None if fb is None else bool(fb)}
                except (TypeError, ValueError, AttributeError):
                    return False
            elif action == 'moment':
                arg = arg if arg in ('drop', 'spinback', 'stall',
                                     'nextdrop') else 'drop'
            elif action == 'layer':
                # Optional loop label from the picker; None = let the
                # engine take the best-fitting one. Press with a layer
                # already riding = kill it (DJSystem._do_layer toggles).
                if arg is not None and (not isinstance(arg, str)
                                        or len(arg) > 120):
                    return False
            elif action == 'arc':
                if not isinstance(arg, list):
                    return False
                pts = []
                for it in arg[:8]:
                    try:
                        pts.append((max(0.0, min(1.0, float(it[0]))),
                                    max(0.0, min(1.0, float(it[1])))))
                    except (TypeError, ValueError, IndexError):
                        return False
                arg = pts
            elif action == 'flavor':
                # Sanitize: known sections only, str tags, weights 0..1.
                if not isinstance(arg, dict):
                    return False
                clean = {}
                for sec in ('prefer_tags', 'avoid_tags', 'axis_targets'):
                    part = arg.get(sec)
                    if not isinstance(part, dict):
                        continue
                    kept = {}
                    for k, v in list(part.items())[:24]:
                        try:
                            kept[str(k)[:32]] = max(0.0, min(1.0, float(v)))
                        except (TypeError, ValueError):
                            continue
                    if kept:
                        clean[sec] = kept
                # require_tags: HARD filter - a list of tag strings.
                req = arg.get('require_tags')
                if isinstance(req, (list, tuple)):
                    clean['require_tags'] = [str(t)[:32] for t in req[:24]
                                             if isinstance(t, str) and t]
                arg = clean
            with self._dict_lock:
                q = self.control_dict.setdefault('request_dj_actions', [])
                q.append((action, arg))
            return True

        @self.socketio.on('dj_action')
        def handle_dj_action(data):
            queue_dj_action(data)

        @self.app.route('/api/dj/action', methods=['POST'])
        def api_dj_action():
            """HTTP twin of the dj_action socketio event, so out-of-process
            tools (the set planner's Push-to-live) can drive the same
            whitelisted queue with a plain POST. Identical validation."""
            ok = queue_dj_action(request.get_json(silent=True))
            return jsonify({'ok': bool(ok)}), (200 if ok else 400)

        def queue_gen_action(data):
            """Validate + clamp one generative-music action and queue it
            for Stories_OGL._apply_gen_controls (render thread, 5 Hz).
            The whitelist/sanitizer lives in lib/gen/actions.py so the
            standalone gen_server and the show can never disagree."""
            from lib.gen.actions import sanitize_gen_action
            pair = sanitize_gen_action(data)
            if pair is None:
                return False
            with self._dict_lock:
                q = self.control_dict.setdefault('request_gen_actions', [])
                q.append(pair)
            return True

        @self.socketio.on('gen_action')
        def handle_gen_action(data):
            queue_gen_action(data)

        @self.app.route('/api/gen/action', methods=['POST'])
        def api_gen_action():
            ok = queue_gen_action(request.get_json(silent=True))
            return jsonify({'ok': bool(ok)}), (200 if ok else 400)

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

        @self.socketio.on('change_project')
        def handle_change_project(data):
            new_id = (data or {}).get('project_id') if isinstance(data, dict) else None
            if not new_id:
                return
            with self._dict_lock:
                avail = self.control_dict.get('available_projects', []) or []
                ids = {p.get('id') for p in avail if isinstance(p, dict)}
                current = self.control_dict.get('current_project')
            if new_id not in ids:
                print(f"[WebController] change_project: unknown project {new_id!r}")
                return
            if new_id == current:
                return
            with self._dict_lock:
                self.control_dict['request_project_swap'] = new_id
            self._values_cache = None

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

        @self.socketio.on('set_switch_timer')
        def handle_set_switch_timer(data):
            """Arm / retarget / cancel the timed weather-set switch.

            {hours: 0..6 | 'setlist', target: set_name}. hours == 0
            cancels. A changed hour count restarts the countdown; a
            changed target while the same mode is armed retargets without
            restarting. 'setlist' fires when the DJ's loaded setlist
            finishes instead of on a clock.
            """
            data = data or {}
            hours = data.get('hours', 0)
            if hours != 'setlist':
                try:
                    hours = int(hours)
                except (TypeError, ValueError):
                    return
            target = data.get('target')
            with self._dict_lock:
                if hours == 0 or (isinstance(hours, int) and hours <= 0):
                    self._set_switch_timer = None
                    return
                available = self.control_dict.get('available_sets', [])
                if (isinstance(hours, int) and hours > 6) \
                        or target not in available:
                    return
                t = self._set_switch_timer
                if t and t['hours'] == hours:
                    t['target'] = target
                elif hours == 'setlist':
                    dj = self.control_dict.get('dj_info') or {}
                    self._set_switch_timer = {
                        'hours': 'setlist', 'target': target, 'end': None,
                        # arm-time state: a setlist already loaded counts
                        # as seen; else wait for one to load, THEN finish.
                        'seen': bool(dj.get('setlist'))}
                else:
                    self._set_switch_timer = {
                        'hours': hours, 'target': target,
                        'end': time.time() + hours * 3600}

        @self.socketio.on('change_audio_source')
        def handle_change_audio_source(data):
            src = data.get('source')
            if src in ('linein', 'loopback', 'internal', 'microphone', 'bluetooth'):
                with self._dict_lock:
                    self.control_dict['request_audio_source'] = src
                self._values_cache = None

        # --- Bluetooth audio sink (lucifera) ---------------------------------
        # Requests are queued as (action, arg) tuples and drained by the main
        # loop, which owns the BluetoothAudioReceiver. Open to the control panel
        # (no admin gating) — per-device pairing approval is the access control.
        def _queue_bt_action(action, arg=None):
            with self._dict_lock:
                self.control_dict.setdefault(
                    'request_bluetooth_actions', []).append((action, arg))
            self._values_cache = None

        @self.socketio.on('toggle_bluetooth_audio')
        def handle_toggle_bluetooth(data=None):
            if not self.get('bt_available', False):
                emit('bluetooth_error', {'error': 'Bluetooth sink not available on this host'})
                return
            enable = bool((data or {}).get('enabled'))
            _queue_bt_action('enable' if enable else 'disable')

        @self.socketio.on('approve_pairing')
        def handle_approve_pairing(data):
            mac = (data or {}).get('mac')
            if mac:
                _queue_bt_action('approve', mac)

        @self.socketio.on('deny_pairing')
        def handle_deny_pairing(data):
            mac = (data or {}).get('mac')
            if mac:
                _queue_bt_action('deny', mac)

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

        @self.socketio.on('interaction_action')
        def handle_interaction_action(data):
            """Queue one interaction-panel control press for the render thread.

            The client sends only a control id (+ value for sliders and
            selects). The action itself is looked up in the LIVE set's
            panel, so a tab left open across a set change can't fire the
            old set's events, and values are clamped to the declared
            range before they leave this function.
            """
            from lib.interaction import resolve_action
            data = data or {}
            control_id = data.get('control')
            if not isinstance(control_id, str):
                return
            panel = self._active_interaction_panel()
            if panel is None or panel.get('page'):
                return
            # A stale tab may name the set it was rendered for; drop the
            # press rather than applying it to whatever is live now.
            wanted_set = data.get('set')
            if isinstance(wanted_set, str) and wanted_set != panel['set']:
                return
            action = resolve_action(panel, control_id, data.get('value'))
            if action is None:
                return
            with self._dict_lock:
                queue = self.control_dict.setdefault('request_interaction_actions', [])
                if len(queue) < 64:  # backstop against a stuck client
                    queue.append(action)
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

                # Timed set switch: fire on the countdown, or - in
                # 'setlist' mode - the moment the DJ's loaded setlist
                # finishes (its name leaves the status while the DJ is
                # still active; pool completion clears it the same way).
                with self._dict_lock:
                    t = self._set_switch_timer
                    fire = False
                    if t and t['hours'] == 'setlist':
                        dj = self.control_dict.get('dj_info') or {}
                        if dj.get('setlist'):
                            t['seen'] = True
                        elif t.get('seen') and dj.get('active'):
                            fire = True
                    elif t and t.get('end') and now >= t['end']:
                        fire = True
                    if fire:
                        self._set_switch_timer = None
                        if t['target'] in self.control_dict.get(
                                'available_sets', []):
                            self.control_dict['request_weather_set'] = \
                                t['target']
                            self._values_cache = None
                            print(f"[WebController] Set-switch timer fired "
                                  f"({t['hours']}): -> {t['target']}")

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
                                "weather_time_scale": self.control_dict.get('weather_time_scale', 1.0),
                                "club": self.control_dict.get('club_director_info'),
                                "club_controls": {
                                    "heat_bias": self.control_dict.get('club_heat_bias', 0.0),
                                    "hold_room": self.control_dict.get('club_hold_room', False),
                                    "palette_override": self.control_dict.get('club_palette_override'),
                                    "density": self.control_dict.get('club_density', 1.0),
                                    "theme": self.control_dict.get('club_theme', ''),
                                    "ttls": self.control_dict.get('club_ttls', {}),
                                },
                                "dj": self.control_dict.get('dj_info'),
                                "gen": self.control_dict.get('gen_info'),
                                "available_sets": list(
                                    self.control_dict.get('available_sets', [])),
                                "set_timer": (
                                    {"hours": self._set_switch_timer['hours'],
                                     "target": self._set_switch_timer['target'],
                                     "remaining_s": (max(
                                         0.0, self._set_switch_timer['end'] - now)
                                         if self._set_switch_timer.get('end')
                                         else None)}
                                    if self._set_switch_timer else None),
                                "brightness_limiting_factor": self.control_dict.get('brightness_limiting_factor', 1.0),
                                "active_effects": list(self.control_dict.get('active_effects', [])),
                                "ambient_sound": self.control_dict.get('ambient_sound'),
                                "allowed_output_params": self.control_dict.get('allowed_output_params'),
                                "narrative_vars": list(self.control_dict.get('narrative_vars', [])),
                                "bluetooth": {
                                    "available": self.control_dict.get('bt_available', False),
                                    "enabled": self.control_dict.get('bt_enabled', False),
                                    "reason": self.control_dict.get('bt_unavailable_reason', ''),
                                    "pending": list(self.control_dict.get('bt_pending', [])),
                                    "connected": list(self.control_dict.get('bt_connected', [])),
                                },
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

                # Push preview frames at 15 Hz to subscribed clients, but
                # only when the encoder has produced a new PNG since last
                # broadcast (prevents re-emitting identical bytes).
                if now - last_frame_push >= frame_interval:
                    last_frame_push = now
                    try:
                        frame_png = self.control_dict.get('_frame_png')
                        if frame_png is not None:
                            frame_id = id(frame_png)
                            if frame_id != getattr(self, '_last_broadcast_frame_id', None):
                                self._last_broadcast_frame_id = frame_id
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
                # Use bind_ip if configured, otherwise auto-detect via the
                # default route (gethostbyname can return a virtual adapter's
                # IP — WSL/VirtualBox — on multi-homed machines)
                if self.bind_ip:
                    local_ip = self.bind_ip
                else:
                    try:
                        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        try:
                            probe.connect(("8.8.8.8", 80))  # no packet sent
                            local_ip = probe.getsockname()[0]
                        finally:
                            probe.close()
                    except OSError:
                        try:
                            local_ip = socket.gethostbyname(socket.gethostname())
                        except Exception:
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

    def set_global_modifier(self, name, value, make_default=False) -> bool:
        """Move a global modifier (a control-panel slider) from code.

        Global modifiers live in their own dict, NOT in ``control_dict``
        — ``set()`` cannot reach them, and a value written to the wrong
        dict is read by nobody.

        ``make_default`` also rewrites the schema default, which is what
        the panel's "reset" restores to. Pass it when the value IS the
        right resting position for the active project — otherwise a reset
        would drop the piece back to whatever the last project wanted.

        Clamps to the modifier's schema and returns False for an unknown
        name.
        """
        schema = self.global_modifier_schema.get(name)
        if schema is None:
            return False
        value = max(schema["min"], min(schema["max"], float(value)))
        with self._dict_lock:
            self.global_modifiers[name] = value
            if make_default:
                schema["default"] = value
        self._values_cache = None
        return True

    def set_project_ceiling(self, name, ceiling) -> bool:
        """Re-scale a slider to the active project's hardware limit.

        Some sliders are guards rather than preferences —
        ``brightness_limit`` is a PSU budget, and above it the supplies
        are over-drawn. The safe value differs per piece (Fan 0.1, Weight
        of Light 0.4), so the slider's ``max`` follows the project rather
        than being one number for every rig: the operator can drag freely
        and still cannot leave the safe range.

        Sets max, default and current value to ``ceiling``, so the piece
        boots at its full budget and every later write — REST, socket, or
        code — is clamped to it. Call on boot and on every project swap.
        """
        schema = self.global_modifier_schema.get(name)
        if schema is None:
            return False
        ceiling = max(schema["min"], float(ceiling))
        with self._dict_lock:
            schema["max"] = ceiling
            schema["default"] = ceiling
            self.global_modifiers[name] = ceiling
        self._values_cache = None
        return True


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
