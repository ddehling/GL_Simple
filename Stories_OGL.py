import argparse
import sys

# Force stdout/stderr to UTF-8 so the many print statements scattered
# through the engine + project shaders that use unicode glyphs (check
# marks, arrows, etc.) don't crash on Windows where the default console
# encoding is cp1252. 'replace' errors mode means an unencodable char
# becomes '?' rather than raising. No-op on Linux/macOS where stdout is
# already UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, Exception):
    pass

import numpy as np
import time
import threading
from pathlib import Path
import yaml
from engine.render_pipeline import RenderPipeline
import lib.dmx_sender as imdmx
from lib.osc_listener import OscListener
from lib.mdns_resolve import resolve as mdns_resolve

from lib.audio_analyzer import MicrophoneAnalyzer
from lib.beat_detector import BeatDetector
from lib.audio_signals import AudioStructure, HarmonicTracker
from lib.bluetooth_audio import create_bluetooth_receiver
# WeatherState is kept as a fallback type for projects that don't override
# the enum; runtime code reads ``self._weather_state_enum`` instead so a
# project can swap in its own enum.
from lib.weather_params import (
    WeatherState as _LIB_WEATHER_STATE,
    WEATHER_SETS as _LIB_WEATHER_SETS,
    DEFAULT_WEATHER_SET as _LIB_DEFAULT_WEATHER_SET,
)
from lib.weather_state import WeatherStateController
from lib.weather_set import WeatherSetManager
from renderer.effects.celestial_bodies import CELESTIAL_BODIES
from renderer import effects as fx
from web.web_controller import WebController
from core.project import load_project, list_projects
from core.shader_loader import load_project_shaders

def load_config(project_override: str | None = None):
    """Load config.yaml, falling back to defaults if missing.

    Returns (cfg, project) where ``project`` is the resolved active
    Project (loaded from ``projects/<id>/project.yaml``). The active id
    comes from (in order):
      1. ``project_override`` (CLI --project flag) if given
      2. ``active_project.yaml`` at repo root (per-machine, gitignored;
         written by bin/setup.* so swapping the active project on a
         machine doesn't cause merge churn in shared config.yaml)
      3. config.yaml's top-level ``project:`` field (tracked default)
      4. ``"fan"`` as the hard-coded fallback
    """
    config_path = Path(__file__).parent / "config.yaml"
    active_project_path = Path(__file__).parent / "active_project.yaml"
    defaults = {
        "project": "fan",
        "display": {"width": 128, "height": 300, "magnification": 0, "headless": False},
        "audio": {"enabled": True, "source": "linein", "linein_device": "",
                  "loopback_device": "", "device_name": "TONOR"},
        "web": {"enabled": True, "port": 5000, "bind_ip": ""},
        # OSC listener — observability only at this stage. Receives messages
        # from Weight_Of_Light boxes (button presses, analog samples,
        # 1-Wire temps) and prints them. Mapping into actual events is a
        # later step.
        "osc": {"enabled": True, "port": 9001, "bind_ip": "0.0.0.0"},
        # Autonomous DJ. enabled=True makes the DJ AVAILABLE (web tab +
        # start button); it never auto-plays on boot. music_dir empty =
        # <repo_parent>/music (the library travels parallel to the repo).
        # stretch_engine: rubberband (default when pylibrb is installed -
        # pip install -r requirements-dj-keylock.txt; falls back to vari
        # otherwise) | rubberband-crisp | vari | wsola | pv. Applied at
        # startup via DJ_STRETCH_ENGINE.
        "dj": {"enabled": True, "music_dir": "", "theme": "groove",
               "night_hours": 6.0, "stretch_max": 1.08, "record": False,
               "stretch_engine": ""},
        "dmx": {"bind_ip": "", "receivers": [
            {"ip": "192.168.68.140", "columns": 32, "column_offset": 0},
            {"ip": "192.168.68.141", "columns": 32, "column_offset": 32},
            {"ip": "192.168.68.142", "columns": 32, "column_offset": 64},
            {"ip": "192.168.68.143", "columns": 32, "column_offset": 96},
        ]},
    }
    if config_path.exists():
        # Force UTF-8 — Python on Windows defaults to cp1252, which trips
        # on non-ASCII bytes anywhere in the file (a stray smart-quote, an
        # editor's BOM, etc.). YAML is UTF-8 by spec.
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        # Top-level scalars (project) merge directly; named sections do a
        # shallow update so user values override defaults key-by-key.
        if "project" in loaded:
            defaults["project"] = loaded["project"]
        for section in defaults:
            if section == "project":
                continue
            if section in loaded:
                defaults[section].update(loaded[section])
        print(f"[Config] Loaded from {config_path}")
    else:
        print(f"[Config] {config_path} not found, using defaults")

    # Per-machine override for the active project. Lives outside config.yaml
    # so 'switch active project on this machine' doesn't dirty a shared file.
    if active_project_path.exists():
        try:
            with open(active_project_path, "r", encoding="utf-8") as f:
                ap = yaml.safe_load(f) or {}
            if isinstance(ap, dict) and ap.get("project"):
                defaults["project"] = ap["project"]
                print(f"[Config] Active project from {active_project_path.name}: {defaults['project']}")
        except Exception as e:
            print(f"[Config] {active_project_path.name} parse failed ({e}); ignoring")

    project_id = project_override or defaults["project"]
    project = load_project(project_id)
    print(f"[Project] Active: {project.display_name} (id={project.id})")

    # project.yaml may override machine-local display dims so each piece
    # gets its native canvas size without the operator editing config.yaml.
    proj_display = project.raw.get("display") if isinstance(project.raw, dict) else None
    if isinstance(proj_display, dict):
        defaults["display"].update(proj_display)
        print(f"[Project] display override: "
              f"{defaults['display']['width']}x{defaults['display']['height']}")
    return defaults, project


def _parse_cli_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--project",
        default=None,
        help="Active project id (overrides config.yaml). Looks for projects/<id>/project.yaml.",
    )
    parser.add_argument(
        "--emulator-port",
        type=int,
        default=None,
        help="If set, broadcast per-frame group canvases on this localhost "
             "TCP port for the layout editor's emulator preview. Off by default.",
    )
    parser.add_argument(
        "--parent-pid",
        type=int,
        default=None,
        help="If set, the engine self-terminates when this PID disappears. "
             "Used by the layout editor's launch button so a crashed editor "
             "doesn't leave the engine orphaned.",
    )
    # parse_known_args so future flags / IDE-injected args don't blow up.
    args, _ = parser.parse_known_args()
    return args


def _start_parent_watcher(parent_pid: int) -> None:
    """Spawn a daemon thread that polls the given PID once a second and
    forces the engine to exit when the parent process disappears.

    On POSIX, ``os.kill(pid, 0)`` is the standard "is this process
    alive" probe — sends no signal, only does the permission check.
    DO NOT USE IT ON WINDOWS: CPython implements ``os.kill`` on
    Windows by calling ``TerminateProcess(pid, sig)``, which actually
    *terminates* the target. Using sig=0 there silently killed the
    editor every poll. The Windows path uses ``OpenProcess`` +
    ``GetExitCodeProcess`` instead, which observes the parent without
    affecting it.
    """
    import os
    import sys
    import threading
    import time

    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259

        def _alive() -> bool:
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, parent_pid
            )
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return False
                return code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
    else:
        def _alive() -> bool:
            try:
                os.kill(parent_pid, 0)
                return True
            except (ProcessLookupError, PermissionError, OSError):
                return False

    def _watch():
        while True:
            if not _alive():
                # Parent gone — exit immediately. Use os._exit so we
                # skip cleanup paths that might hang on hardware locks
                # the parent's death cut us off from.
                print(f"[ParentWatcher] Parent PID {parent_pid} gone; exiting.")
                os._exit(0)
            time.sleep(1.0)

    threading.Thread(target=_watch, name="ParentWatcher", daemon=True).start()


class EnvironmentalSystem:
    def __init__(self, project_override: str | None = None,
                 emulator_port: int | None = None):
        cfg, self.project = load_config(project_override=project_override)
        # Hold onto cfg so swap_project can rebuild from machine-local
        # settings (DMX bind IP, legacy receiver fallback) without
        # re-reading from disk.
        self._cfg = cfg

        # Submodule sanity check. Per-project media (projects/<id>/media/)
        # lives in its own GitHub repo mounted as a submodule. If the
        # operator cloned main but didn't init the submodule, the directory
        # exists but is empty — sounds will fail to load with cryptic
        # FileNotFoundErrors later. Catch it up front with the exact fix
        # command. Warning-only, not fatal: new projects pre-submodule
        # legitimately have empty media trees, and developers may want to
        # boot without ambient audio.
        self._check_media_submodule()
        disp = cfg["display"]
        audio_cfg = cfg["audio"]
        web_cfg = cfg["web"]
        dmx_cfg = cfg["dmx"]
        osc_cfg = cfg.get("osc", {})

        # One canvas per project group (Phase 6). Order is authoritative —
        # frame_id == index in self.project.groups.
        frame_dimensions = [(g.width, g.height) for g in self.project.groups]
        group_ids = [g.id for g in self.project.groups]

        # Hardware receiver configuration. Source of truth (in priority):
        #   1. project.yaml's top-level `receivers:` list (strip-based).
        #   2. config.yaml's `dmx.receivers` (legacy column-rect / HS / VS).
        # Each entry's address is resolved via `_resolve_receiver_address` so
        # mDNS hostnames work either way; per-receiver `protocol` (default
        # "sacn"; "ddp" supported) is passed through to the sender.
        receivers_list = self._build_receivers(
            project_receivers=self.project.raw.get('receivers'),
            legacy_receivers=dmx_cfg['receivers'],
            display_height=disp["height"],
        )
        receivers = [receivers_list]

        # Project-local shaders: merge into ``renderer.effects`` namespace
        # so existing ``fx.shader_xxx`` references resolve to the project's
        # files. Fan moves all its themed shaders (BarTiki/Ocean/Forest/
        # Desert/Spooky/Beloved) into projects/fan/shaders/ and this call
        # auto-loads them; WoL has none today so this is a no-op.
        load_project_shaders(self.project)

        # Geometry provider — drives the web preview's geometry JSON and the
        # PNG-encode composite path. Built once from the active project's
        # ``geometry:`` block and shared with both RenderPipeline and the
        # WebController.
        self.geometry_provider = self.project.load_geometry()

        # Project-specific weather machinery (Phase 8): the project's
        # weather_params module exports its own WEATHER_SETS / DEFAULT_WEATHER_SET
        # / WeatherState / WEATHER_PRESETS / DEFAULT_WEATHER_PARAMS. When a
        # project re-exports lib.weather_params via ``from ... import *``
        # (Fan today), these resolve to lib's globals; when a project
        # overrides one (WoL: WEATHER_SETS + DEFAULT_WEATHER_SET only), only
        # the override is project-specific.
        self._refresh_weather_module(self.project)

        # Emulator broadcaster — only instantiated when the editor passed
        # --emulator-port. start() may fail (port in use); on failure we
        # log and continue without emulation rather than abort the run.
        emu = None
        if emulator_port is not None:
            from lib.emulator_broadcaster import EmulatorBroadcaster
            emu = EmulatorBroadcaster(emulator_port)
            if not emu.start():
                emu = None

        self.scheduler = RenderPipeline(
            frame_dimensions=frame_dimensions,
            receivers=receivers,
            magnification=disp["magnification"],
            headless=disp["headless"],
            dmx_bind_ip=dmx_cfg.get("bind_ip", ""),
            geometry_provider=self.geometry_provider,
            group_ids=group_ids,
            emulator=emu,
        )
        self._emulator = emu

        # Per-project brightness ceiling. Each piece has its own PSU
        # budget; the project.yaml ``brightness_limit:`` key seeds
        # both the pipeline's hardware limiter and the web slider's
        # initial position. Falls through to RenderPipeline's default
        # when the project doesn't declare one.
        if self.project.brightness_limit is not None:
            self.scheduler.brightness_setpoint = self.project.brightness_limit
            self.scheduler.state["brightness_limit"] = self.project.brightness_limit
            print(f"[Project] brightness_limit = {self.project.brightness_limit}")

        # Per-project target render FPS. Read from project.yaml's
        # ``target_fps:`` key; main loop reads ``env_system.frame_time``
        # each tick so a project swap re-paces the render loop without
        # restart. Default 40 fps when the project doesn't declare one.
        self.frame_time = self._compute_frame_time(self.project)
        print(f"[Project] target_fps = {1.0 / self.frame_time:.1f} "
              f"(frame_time = {self.frame_time*1000:.1f} ms)")

        # Expose strip metadata to the rendering engine so effects can
        # query the active project's strip table — e.g. drive
        # per-object behaviours, count strips per group, look up which
        # receiver a particular strip lives on. Updated again in
        # _swap_project_unsafe whenever the project changes.
        self._publish_strip_metadata(receivers_list)

        # Boot resilience: receivers whose mDNS hostname didn't resolve at
        # startup (box powered on after the controller, or network not ready
        # yet) are skipped by _build_receivers and would never get data until
        # a restart. Track how many the project wants vs how many are live and
        # spin a background watcher that re-resolves the missing ones and asks
        # the render thread to rebuild the senders when one appears. Receivers
        # configured with a literal static IP always resolve, so a static-only
        # project never needs the watcher. See _start_receiver_watch.
        self._receiver_lock = threading.Lock()
        self._pending_receivers = None         # bg watcher -> render thread
        self._receiver_resolve_interval = 15.0  # seconds between retries
        self._live_receiver_count = len(receivers_list)
        self._receiver_watch_thread = None
        self._receiver_watch_stop = False
        self._start_receiver_watch()
        self.weather_state = WeatherStateController(
            weather_state_enum=self._weather_state_enum,
            weather_presets=self._weather_presets,
            default_weather_params=self._default_weather_params,
            output_map=self._outstate_publish,
        )
        self.season = 0.0
        self.analyzer = None
        self._audio_tap_wired = False
        if audio_cfg.get("enabled", True):
            try:
                self.analyzer = MicrophoneAnalyzer(
                    source=audio_cfg.get("source", "linein"),
                    device_name=audio_cfg.get("device_name"),
                    linein_device=audio_cfg.get("linein_device") or None,
                    loopback_device=audio_cfg.get("loopback_device") or None,
                )
                self.analyzer.start()
            except Exception as e:
                print(f"[Audio] Failed to initialize audio analyzer: {e}")
                print("[Audio] Continuing without audio input")
                self.analyzer = None

        # Bluetooth audio sink (advertised as "lucifera"). Linux + BlueZ only;
        # an inert stub elsewhere (e.g. Windows), so this never fails. Defaults
        # OFF — only turned on from the web UI. State is mirrored into the web
        # control_dict each web-check tick (see _apply_bluetooth_controls).
        self.bt_receiver = create_bluetooth_receiver()
        self._bt_prev_source = None            # source to restore on disconnect
        self._bt_last_connected = 0            # connected-device count last tick

        # Autonomous DJ (lib/dj). Constructed lazily on the first web
        # "start" action so boot cost is zero when unused; config only
        # gates AVAILABILITY. See _apply_dj_controls for the 5 Hz bridge.
        self.dj_cfg = cfg.get("dj", {"enabled": False})
        # Tempo-engine choice from config.yaml (dj.stretch_engine) - the
        # no-env-vars way to pick vari/rubberband/rubberband-crisp/wsola/
        # pv. An explicit DJ_STRETCH_ENGINE in the environment still wins.
        eng = str(self.dj_cfg.get("stretch_engine", "") or "")
        eng = eng.strip().lower()
        if eng and "DJ_STRETCH_ENGINE" not in os.environ:
            if eng.startswith("rubberband") or eng == "rb":
                os.environ["DJ_STRETCH_ENGINE"] = "rubberband"
                os.environ.setdefault(
                    "DJ_RB_ENGINE",
                    "faster" if eng.endswith("crisp") else "finer")
            else:
                os.environ["DJ_STRETCH_ENGINE"] = eng
        self._dj = None
        self._dj_prev_source = None            # analyzer source to restore
        self._dj_pending_setlist = None        # armed while idle, load on start
        self._dj_pending_flavor = {}           # armed while idle, set on start
        self._dj_pending_arc = []              # arc waypoints armed while idle
        self._dj_pending_nudge = 0.0
        self._dj_idle_vocab = (0.0, [])        # (stamp, tag vocab) for chips
        self._dj_last_error = ""

        # Beat / tempo detector — a pure consumer of the analyzer output,
        # published into outstate by send_variables() so any shader can sync
        # to the beat. Harmless when there's no audio (returns zeros).
        self._beat_detector = BeatDetector()
        self._prev_beat_time = time.time()
        # Song-structure signals (bass/mid/high scalars, energy, build, drop)
        # — same consumer contract as the beat detector, published alongside.
        self._audio_structure = AudioStructure()
        self._harmonic_tracker = HarmonicTracker()

        # MIDI is intentionally NOT wired into the show. The club set (and
        # everything else) is a fully autonomous audio-visual experience -
        # visuals are driven by the analyzer/beat/structure signals and the
        # weather machinery only. lib/midi_controller.py remains available
        # for standalone tools, but the render loop takes no MIDI input.
        self.midi = None
        self.scale = 0.2

        # Initialize web control system
        self.enable_web_control = web_cfg["enabled"]
        self.web_controller = None

        if self.enable_web_control:
            self.web_controls = {
                "current_weather_set": self._default_weather_set,
                "available_sets": list(self._weather_sets.keys()),
                "available_weather_states": list(
                    self._weather_sets[self._default_weather_set]["states"]
                ),
                "all_weather_states": [s.value for s in self._weather_state_enum],
                "state_switch_locked": True,
                "weather_state_locked": False,
                "led_width": frame_dimensions[0][0],
                "led_height": frame_dimensions[0][1],
                "current_project": self.project.id,
                "current_project_name": self.project.display_name,
                "available_projects": list_projects(),
                # Project-local media root, exposed so the narrative
                # editor can enumerate scripts/pools in the active
                # project's folder rather than a hardcoded ``media/``.
                "media_root": str(self.project.media_root),
            }
            self.web_controller = WebController(
                self.web_controls,
                port=web_cfg["port"],
                service_name="lucifera",
                bind_ip=web_cfg.get("bind_ip", ""),
                geometry_provider=self.geometry_provider,
            )
            self.web_controller.start(threaded=True)
            # Register viewports so socket handlers can mutate them directly
            self.web_controller.control_dict['_viewports'] = self.scheduler._shader_renderer.viewports

        # OSC listener — observability + project-side routing. The
        # listener stays a print-only catch-all by default; the active
        # project's ``button_router`` hook (if declared) registers
        # prefix routes to handle structured input addresses (button
        # presses, sensor streams). Failure to bind is logged but
        # non-fatal — the show keeps running.
        self.osc_listener = None
        if osc_cfg.get("enabled", True):
            # ``log_unrouted`` toggles the catch-all console log for
            # any OSC message no project hook has claimed. Off by
            # default so radar firmware bursts don't flood the
            # console when a non-WoL project is active; opt back in
            # via ``osc.log_unrouted: true`` in config.yaml when
            # diagnosing message-shape problems.
            self.osc_listener = OscListener(
                port=int(osc_cfg.get("port", 9001)),
                bind_ip=osc_cfg.get("bind_ip", "0.0.0.0"),
                log_unrouted=bool(osc_cfg.get("log_unrouted", False)))
            self.osc_listener.start()

        # Project ``button_router`` hook: project.yaml may declare
        # ``hooks: { button_router: <module.path> }``. The module's
        # ``register(env_system)`` is called once at boot — typically
        # to wire OSC route handlers, build per-project name maps, etc.
        # Errors are logged but never fatal (a buggy router shouldn't
        # take the show down).
        self._call_button_router_hook()

        # Initialize celestial bodies
        self.celestial_bodies = CELESTIAL_BODIES.copy()
        # sort celestial bodies by distance, farthest first
        self.celestial_bodies.sort(key=lambda x: x.distance, reverse=True)

        self.scheduler.state["tree"] = False
        self.scheduler.state["skyfull"] = False
        self.scheduler.state["simulate"] = True  # Display the leds in an opencv window for visualization
        # Project-local media root. Effects + random_events resolve sound /
        # narrative paths against this rather than against the cwd, so
        # WoL doesn't try to load Fan media (and vice versa).
        self.scheduler.state["media_root"] = str(self.project.media_root)
        self.active_effects = {"world": None, "ambient_sound": None}
        # Wall-clock time the current weather state finished transitioning
        # in. None until the active state's transition completes. Drives the
        # deterministic ``state_duration`` timer in random_state_change;
        # reset to None on every transition_to_weather so it restarts.
        self._state_hold_start = None
        # Event map — pulled from the active project module. Copied so any
        # in-place mutation here doesn't bleed back into the module-level
        # constant.
        self.event_map = dict(self.project.load_event_map())

        # WeatherSetManager owns the event_map from here on
        self.weather_set = WeatherSetManager(
            self.event_map,
            weather_sets=self._weather_sets,
            default_set=self._default_weather_set,
            weather_state_enum=self._weather_state_enum,
        )
        del self.event_map  # WeatherSetManager is the single owner

        # Pass event names to web controller if enabled
        if self.enable_web_control:
            from lib.weather_set import IMPLICIT_BACKGROUND_EVENTS
            event_list = self.weather_set.get_event_names()
            # Hide implicitly-scheduled events (narrative_player, sound_pool)
            # from the "+ Add Background Event" dropdown; they're controlled
            # via per-set dropdown fields instead.
            bg_events = [e for e in event_list if e not in IMPLICIT_BACKGROUND_EVENTS]
            self.web_controller.set_available_events(
                all_events=event_list,
                background_events=bg_events,
            )
            # Push the active project's weather machinery into the web
            # UI so the editor surfaces the right enum / presets / sets
            # instead of the lib-level union of every project's states.
            self.web_controller.set_weather_module(
                weather_state_enum=self._weather_state_enum,
                weather_presets=self._weather_presets,
                weather_sets=self._weather_sets,
                default_weather_params=self._default_weather_params,
                default_weather_set=self._default_weather_set,
                weather_module_path=self._weather_module_path,
                weather_module_name=self._weather_module_name,
            )
            # Sync initial set name now that WeatherSetManager is ready
            self.web_controller.set("current_weather_set", self.weather_set.current_set)
        
        # Initialize background events for the starting weather set
        self._initialize_weather_set_events()

        self.whompcount = 0

    @staticmethod
    def _compute_frame_time(project) -> float:
        """Return seconds-per-frame for the project's target FPS.
        Falls back to 1/40 when the project doesn't declare one."""
        fps = getattr(project, "target_fps", None)
        if fps is None or fps <= 0:
            return 1.0 / 40.0
        return 1.0 / float(fps)

    def _refresh_weather_module(self, project) -> None:
        """Cache the project's weather machinery on self.

        Pulled out as its own helper so ``__init__`` and ``swap_project``
        share the same logic. ``getattr(..., default)`` falls back to the
        ``lib.weather_params`` globals so a project that doesn't override
        a particular field (e.g. WoL inherits Fan's WeatherState enum and
        presets) keeps working.
        """
        try:
            mod = project.load_weather_module()
        except Exception as e:
            print(f"[Project] weather module load failed ({e}); using lib defaults")
            mod = None
        from lib.weather_params import (
            DEFAULT_WEATHER_PARAMS as _LIB_DEFAULT_PARAMS,
            WEATHER_PRESETS as _LIB_PRESETS,
        )
        self._weather_state_enum = getattr(mod, "WeatherState", _LIB_WEATHER_STATE)
        self._weather_sets = getattr(mod, "WEATHER_SETS", _LIB_WEATHER_SETS)
        self._default_weather_set = getattr(mod, "DEFAULT_WEATHER_SET", _LIB_DEFAULT_WEATHER_SET)
        self._weather_presets = getattr(mod, "WEATHER_PRESETS", _LIB_PRESETS)
        self._default_weather_params = getattr(mod, "DEFAULT_WEATHER_PARAMS", _LIB_DEFAULT_PARAMS)
        # Project outstate publish table (realm-specific params). The
        # engine core only publishes the generic weather vocabulary; a
        # project without this table gets ONLY the core — its realm
        # shaders would run blind at their wrapper defaults, so warn
        # loudly rather than fail silently.
        self._outstate_publish = getattr(mod, "OUTSTATE_PUBLISH", None)
        if self._outstate_publish is None:
            print("[Project] WARNING: weather module defines no OUTSTATE_PUBLISH "
                  "table — only core weather outputs (fog/wind/rain/stars/...) "
                  "will reach effects. Realm-specific params need an entry there.")
            self._outstate_publish = {}
        # Module identity — used by the web editor save / reload paths
        # to write back to the project's own weather_params.py rather
        # than lib's (saves to lib are silently overridden by any
        # project module that defines its own WEATHER_SETS).
        self._weather_module_name = getattr(project, "weather_sets_module", None)
        self._weather_module_path = getattr(mod, "__file__", None) if mod else None

    def _check_media_submodule(self) -> None:
        """Warn the operator if the active project's media submodule
        looks uninitialized.

        Per-project media lives in its own GitHub repo (``GL_Simple_<id>_media``)
        mounted at ``projects/<id>/media/``. A fresh clone of the main
        repo leaves that directory empty until the operator runs
        ``git submodule update --init``. Without this check, sounds
        fail to load downstream with FileNotFoundErrors that don't
        suggest the actual cause — print the fix command up front.

        Heuristic: the directory exists, contains no real files (only
        possibly ``.git`` / ``.gitkeep`` markers), AND the project is
        listed in ``.gitmodules``. New projects without submodules
        wired up legitimately have empty media trees, so we gate on
        the .gitmodules presence to distinguish.
        """
        media_root = self.project.media_root
        if not media_root.is_dir():
            return    # No media dir declared at all — fine.

        gitmodules = Path(__file__).parent / ".gitmodules"
        if not gitmodules.is_file():
            return    # Not a submodule-managed repo; skip.

        # Is this project's media listed in .gitmodules?
        try:
            gm_text = gitmodules.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        rel_path = f"projects/{self.project.id}/media"
        if rel_path not in gm_text:
            return    # This project's media isn't a submodule — fine.

        # Filter out .git / .gitkeep / hidden markers; if anything real
        # remains, the submodule is populated and we're done.
        real_entries = [p for p in media_root.rglob("*")
                        if p.is_file() and not any(part.startswith(".")
                                                   for part in p.relative_to(media_root).parts)]
        if real_entries:
            return

        print()
        print("=" * 70, file=sys.stderr)
        print(f"  WARNING: {rel_path}/ is empty.", file=sys.stderr)
        print(f"  This project's media is in a git submodule that hasn't", file=sys.stderr)
        print(f"  been initialized yet. Audio/narrative will fail to load.", file=sys.stderr)
        print(f"", file=sys.stderr)
        print(f"  Fix:", file=sys.stderr)
        print(f"    git submodule update --init {rel_path}", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print()

    def _build_receivers(self, project_receivers, legacy_receivers, display_height: int):
        """Resolve receiver entries from project.yaml strips or config.yaml legacy.

        Returns a list of receiver dicts ready for ``SACNPixelSender``: each
        either carries a ``strips`` list (project form) or
        ``addressing_array`` + ``pixel_count`` (legacy form). The sender
        normalizes both into the same Nx2 internal layout, so output is
        byte-identical regardless of which path produced the receiver.

        mDNS resolution runs in parallel: each unresolvable hostname costs
        ~3s on the render thread, so a project with 9 missing hosts would
        otherwise serialize into ~27s of pause during a swap. The pool
        also warms ``mdns_resolve``'s success cache so subsequent rebuilds
        for the same hosts are instant.
        """
        from concurrent.futures import ThreadPoolExecutor
        from core.strip import strips_from_yaml_list

        def _resolve_pool(entries):
            entries = list(entries) if entries else []
            if not entries:
                return []
            workers = min(16, max(1, len(entries)))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                return list(ex.map(
                    lambda rx: self._resolve_receiver_address(rx)
                    if isinstance(rx, dict) else None,
                    entries,
                ))

        out: list[dict] = []
        if project_receivers:
            ips = _resolve_pool(project_receivers)
            for rx, ip in zip(project_receivers, ips):
                if not isinstance(rx, dict):
                    print(f"[Project] Skipping malformed receiver entry: {rx!r}")
                    continue
                if ip is None:
                    continue
                strips = strips_from_yaml_list(rx['strips'])
                # Tag each strip with its receiver's object_id so
                # effects can correlate strips with physical objects.
                oid = int(rx.get('object_id', -1))
                for s in strips:
                    s.object_id = oid
                out.append({
                    'ip': ip,
                    'protocol': rx.get('protocol', 'sacn'),
                    'strips': strips,
                })
            print(f"[Project] Loaded {len(out)} receivers from project.yaml "
                  f"({sum(len(r['strips']) for r in out)} strips)")
            return out

        # Legacy fallback — config.yaml column-rect / HS / VS receivers.
        legacy_ips = _resolve_pool(legacy_receivers)
        for rx, ip in zip(legacy_receivers, legacy_ips):
            if not isinstance(rx, dict):
                print(f"[Config] Skipping malformed receiver entry: {rx!r}")
                continue
            if ip is None:
                continue
            protocol = rx.get('protocol', 'sacn')
            if 'addressing' in rx:
                mode = rx['addressing']['mode']
                filepath = rx['addressing']['file']
                if mode == 'hs':
                    addr = imdmx.make_indicesHS(filepath)
                elif mode == 'vs':
                    addr = imdmx.make_indicesVS(filepath)
                else:
                    raise ValueError(f"Unknown addressing mode: {mode}")
                out.append({
                    'ip': ip,
                    'pixel_count': len(addr),
                    'addressing_array': addr,
                    'protocol': protocol,
                })
            else:
                out.append({
                    'ip': ip,
                    'pixel_count': display_height * rx['columns'],
                    'addressing_array': imdmx.make_indices_V_rect_alternate(
                        rx['columns'], display_height, rx['column_offset']
                    ),
                    'protocol': protocol,
                })
        return out

    @staticmethod
    def _resolve_receiver_address(rx: dict):
        """Pick a literal IP for one receiver entry.

        - ``host:`` (mDNS name) is preferred when set; resolved via
          ``lib.mdns_resolve.resolve``.
        - ``ip:`` is treated as a candidate too — also routed through
          ``mdns_resolve`` so invalid addresses (typos like
          ``192.168.68.401``) get caught here rather than blowing up
          the render loop with ``socket.gaierror`` on every send.
        - Failed resolutions log and return None — the caller skips the
          entry so one missing box doesn't block the rest of the show
          from coming up.
        """
        host = rx.get('host')
        ip = rx.get('ip')
        if host:
            resolved = mdns_resolve(host)
            if resolved is not None:
                print(f"[mDNS] {host} -> {resolved}")
                return resolved
            # Fall through to ip if both are given.
            if ip:
                print(f"[mDNS] {host!r} not found; trying ip {ip!r}")
            else:
                print(f"[mDNS] {host!r} not found; skipping receiver")
                return None
        if ip:
            # mdns_resolve passes valid IPv4 strings through unchanged; an
            # invalid one (bad octet, unreachable hostname-in-ip-field,
            # etc.) returns None and the receiver is skipped.
            resolved = mdns_resolve(ip)
            if resolved is not None:
                return resolved
            print(f"[mDNS] {ip!r} is not a valid IP and didn't resolve as "
                  f"a hostname; skipping receiver")
            return None
        print(f"[mDNS] receiver entry has neither 'host' nor 'ip'; skipping: {rx}")
        return None

    # ------------------------------------------------------------------
    # Receiver boot-resilience: re-resolve late-arriving boxes at runtime
    # ------------------------------------------------------------------
    def _receiver_source(self):
        """(project_receivers, legacy_receivers, display_height) the active
        project resolves receivers from — captured so the watcher can
        re-resolve without re-deriving it."""
        return (
            self.project.raw.get('receivers'),
            self._cfg['dmx']['receivers'],
            self.project.groups[0].height if self.project.groups
            else int(self._cfg['display']['height']),
        )

    def _desired_receiver_count(self) -> int:
        """How many receiver entries the active project declares (resolved or
        not). Once this many are live, every box is online."""
        proj = self.project.raw.get('receivers')
        if proj:
            return sum(1 for rx in proj if isinstance(rx, dict))
        legacy = self._cfg['dmx']['receivers'] or []
        return sum(1 for rx in legacy if isinstance(rx, dict))

    def _rebuild_senders(self, receivers_list) -> None:
        """Replace the DMX senders with ones built for ``receivers_list``.
        Render-thread only — mutates ``scheduler.state['screens']``. Mirrors
        the sender rebuild in _swap_project_unsafe, minus the project teardown.
        """
        for sender in list(self.scheduler.state.get('screens', [])):
            try:
                sender.close()
            except Exception as e:
                print(f"[Receivers] sender close error (ignoring): {e}")
        new_screens = []
        if receivers_list:
            sender = imdmx.SACNPixelSender(
                receivers_list, skip_network=False,
                use_raw_udp=True, per_receiver_universe=True,
                bind_ip=self._cfg['dmx'].get('bind_ip', ''),
            )
            sender.enable_async_send()
            new_screens.append(sender)
        self.scheduler.state['screens'] = new_screens
        self._publish_strip_metadata(receivers_list)
        self._live_receiver_count = len(receivers_list)

    def _start_receiver_watch(self) -> None:
        """Spin a background thread to re-resolve receivers that weren't live
        at startup, if any. Stops once all declared receivers are live, so a
        fully-online rig does no ongoing mDNS chatter. Safe to call again on a
        project swap — it cancels any prior watcher first.
        """
        # Cancel a prior watcher (e.g. from before a project swap).
        self._receiver_watch_stop = True
        prev = getattr(self, '_receiver_watch_thread', None)
        if prev is not None and prev.is_alive():
            prev.join(timeout=0.1)

        desired = self._desired_receiver_count()
        if self._live_receiver_count >= desired:
            return  # all receivers already live — nothing to watch for

        print(f"[Receivers] {self._live_receiver_count}/{desired} live at startup; "
              f"watching for the rest (re-resolve every "
              f"{self._receiver_resolve_interval:.0f}s)...")
        self._receiver_watch_stop = False
        proj_id = self.project.id
        src = self._receiver_source()

        def _loop():
            applied = self._live_receiver_count   # local: avoids racing the render thread
            while not self._receiver_watch_stop:
                time.sleep(self._receiver_resolve_interval)
                if self._receiver_watch_stop or self.project.id != proj_id:
                    return
                try:
                    resolved = self._build_receivers(
                        project_receivers=src[0],
                        legacy_receivers=src[1],
                        display_height=src[2],
                    )
                except Exception as e:
                    print(f"[Receivers] re-resolve error (will retry): {e}")
                    continue
                if len(resolved) > applied:
                    print(f"[Receivers] {len(resolved)}/{desired} now resolvable "
                          f"(was {applied}); queuing sender rebuild")
                    with self._receiver_lock:
                        self._pending_receivers = resolved
                    applied = len(resolved)
                if applied >= desired:
                    print("[Receivers] all declared receivers online; watcher done")
                    return

        t = threading.Thread(target=_loop, name="receiver-watch", daemon=True)
        self._receiver_watch_thread = t
        t.start()

    def update(self):
        """Update the environmental system - should be called each frame"""
        self.current_time = time.time()

        # Render-thread: apply a receiver set the background watcher resolved
        # (a previously-missing box came online). Done here, not in the watcher
        # thread, so the sender rebuild happens on the render thread like a
        # project swap does.
        pending = None
        with self._receiver_lock:
            if self._pending_receivers is not None:
                pending = self._pending_receivers
                self._pending_receivers = None
        if pending is not None:
            print(f"[Receivers] Rebuilding senders for {len(pending)} receiver(s)")
            self._rebuild_senders(pending)

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

        # Service a narrative-driven weather-transition request, if the
        # narrative_player published one this frame. Backwards compatible
        # — old scripts without per-node ``trigger_state`` fields never
        # publish, and this block is a no-op. The source node id is
        # published alongside the state name so every log line names
        # which node fired the trigger.
        # Service an event request published by a director event this frame
        # (e.g. the club director firing a one-shot on a detected drop).
        # Value: (event_name, duration_seconds). Unknown names are logged
        # and ignored by _schedule_event_from_map.
        ev_req = self.scheduler.state.pop('_event_request', None)
        if ev_req is not None:
            try:
                ev_name, ev_dur = ev_req
                self._schedule_event_from_map(str(ev_name), 0, float(ev_dur),
                                              frame_id=0)
            except (TypeError, ValueError) as e:
                print(f"[EVENT] bad _event_request {ev_req!r}: {e}")

        req      = self.scheduler.state.pop('_weather_transition_request', None)
        src_node = self.scheduler.state.pop('_weather_transition_node', None)
        # Optional duration override for the request (seconds). Lets a
        # director snap into a room (e.g. sub-second cut ON a drop) instead
        # of taking the target preset's usual crossfade time.
        req_dur  = self.scheduler.state.pop('_weather_transition_duration', None)
        if req is not None:
            node_tag = f"node '{src_node}'" if src_node else "node (unknown)"
            try:
                target = self._weather_state_enum(req)
            except (ValueError, KeyError):
                print(f"[NARRATIVE] {node_tag}: unknown weather state "
                      f"'{req}' - ignored")
                target = None
            if target is not None:
                set_states = self.weather_set.get_set_states()
                if target not in set_states:
                    print(f"[NARRATIVE] {node_tag}: state '{req}' not in "
                          f"active set '{self.weather_set.current_set}' "
                          f"- ignored")
                elif (target == self.weather_state.target_weather
                      and getattr(self.weather_state, 'progress', 1.0) >= 1.0):
                    # Already in target state and not transitioning;
                    # skip the redundant snap but log so it's visible.
                    print(f"[NARRATIVE] {node_tag}: already in '{req}' "
                          f"- transition skipped")
                else:
                    target_params = self.weather_state.get_weather_params(target)
                    duration = float(target_params.get('transition_duration', 10.0))
                    if req_dur is not None:
                        try:
                            duration = max(0.05, float(req_dur))
                        except (TypeError, ValueError):
                            pass
                    print(f"[NARRATIVE] {node_tag} triggered transition "
                          f"to '{req}' ({duration:.1f}s)")
                    self.transition_to_weather(target, duration)

        # Copy PNG frame to web controller for preview streaming
        if hasattr(self, 'web_controller'):
            png = self.scheduler.state.get('_frame_png')
            if png is not None:
                self.web_controller.control_dict['_frame_png'] = png

    def change_weather_set(self, new_set_name: str, immediate: bool = False,
                           initial_weather=None):
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

        # Already in this set: normally a no-op, BUT if a specific
        # initial_weather was requested (e.g. a startup_weather_state whose
        # set is also the default set), still fall through so we apply that
        # state — otherwise the boot stays on the controller's CLEAR fallback.
        if new_set_name == self.weather_set.current_set and initial_weather is None:
            print(f"[WEATHER] Already in set '{new_set_name}', skipping")
            return True

        # Changing SET is the operator asking for that set's full experience,
        # soundscape included - hand the soundtrack back if the DJ still owns
        # it. (Measured 2026-07-12: the DJ kept running across a set change,
        # its ambient-suppression guard stayed engaged, and the new set
        # arrived silent with no hint why.)
        dj = getattr(self, "_dj", None)
        if dj is not None and dj.active:
            print("[DJ] weather set changed - stopping the DJ and handing "
                  "the soundtrack back")
            self._dj_stop()

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
    
    def _publish_strip_metadata(self, receivers_list: list) -> None:
        """Expose the active project's strip table + per-pixel
        metadata atlas to effects via the scheduler's shared state.

        Keys published:
          * ``state["project"]`` — the active Project
          * ``state["strips_by_group"]`` — group_id → list[StripBinding]
          * ``state["strips_by_object"]`` — object_id → list[StripBinding]
          * ``state["receivers"]`` — raw list as built for the DMX
            sender (entries: {ip, host, protocol, strips, object_id})

        Plus the *render-aware* additions:
          * ``state["strip_table"]`` — list[dict], one per strip, each
            with a globally unique ``uid``, plus ``group_id``,
            ``strip_idx``, ``receiver_idx``, ``object_id``, ``length``,
            ``positions`` (Nx2 float32 of physical (x, y) per LED, NaN
            where layout is unknown). Effects iterate this list to do
            per-strip work without juggling receivers.
          * ``state["group_metadata"]`` — dict[group_id, dict] where
            each entry is a stack of (H, W) numpy arrays mirroring
            the FBO. Per-pixel lookup answers the four awareness
            questions:
                strip_uid    — which strip owns this pixel (-1 = none)
                strip_idx    — wire-order index within receiver
                receiver_idx — which receiver
                object_id    — which physical object
                led_chain_idx— LED's index within its strip (0..L-1)
                pos_x, pos_y — real-world (x, y) on composite canvas,
                              in pixels (NaN if no physical layout)
                pos_u, pos_v — same position, NORMALIZED to
                              ``[-0.5, +0.5]`` per-axis, origin at
                              canvas center. Recommended for pattern
                              math because it's scale-independent —
                              ``r = sqrt(u² + v²)`` works identically
                              on Fan (1024×600) and WoL (1024×768).
            Effects sample these with ``arr[row, col]``; shaders that
            want them can upload as textures.
          * ``state["composite_canvas_size"]`` — ``(width, height)``
            tuple in pixels. Useful when an effect mixes normalized
            and pixel coordinates.
        """
        import numpy as _np
        from core.strip import strips_from_yaml_list

        # Build the strip table and metadata atlases from the project
        # YAML's FULL receiver list — *not* the runtime-filtered
        # ``receivers_list``. The latter drops receivers whose mDNS
        # hostnames couldn't be resolved (correct policy for DMX
        # output: don't try to send to unreachable hosts), but a
        # strip's PHYSICAL POSITION on the layout canvas is the same
        # whether or not its receiver is online. Effects need to know
        # about every strip the project declares so per-pixel
        # rendering uses the right coordinates regardless of network
        # state. Without this decoupling, only strips on resolved
        # receivers had non-NaN entries in the atlas, and shaders
        # using ``isnan(pos_u)`` would discard every other strip's
        # fragments — exactly the symptom that surfaced on WoL where
        # only one receiver hostname resolves locally.
        project_receivers = self.project.raw.get("receivers") or []

        strips_by_group: dict = {}
        strips_by_object: dict = {}
        # Flat strip table — unique id per strip, all metadata in one
        # row so effects can iterate without joining across dicts.
        strip_table: list[dict] = []

        canvas_w, canvas_h = self.geometry_provider.composite_canvas_size()
        canvas_w = max(int(canvas_w), 1)
        canvas_h = max(int(canvas_h), 1)

        for rx_idx, rx in enumerate(project_receivers):
            if not isinstance(rx, dict):
                continue
            try:
                strips = strips_from_yaml_list(rx.get("strips", []) or [])
            except Exception as e:
                print(f"[Project] receiver {rx_idx} strip load failed for "
                      f"metadata atlas: {e}")
                continue
            oid = int(rx.get("object_id", -1))
            for s in strips:
                s.object_id = oid
                uid = len(strip_table)
                positions = self.geometry_provider.led_positions(
                    s.group_id, s.strip_idx, s.length
                )
                positions_norm = _np.empty_like(positions)
                positions_norm[:, 0] = positions[:, 0] / canvas_w - 0.5
                positions_norm[:, 1] = positions[:, 1] / canvas_h - 0.5
                strip_table.append({
                    "uid": uid,
                    "group_id": s.group_id,
                    "strip_idx": s.strip_idx,
                    "receiver_idx": rx_idx,
                    "object_id": oid,
                    "length": s.length,
                    "pixel_indices": s.pixel_indices,
                    "positions": positions,
                    "positions_norm": positions_norm,
                })
                strips_by_group.setdefault(s.group_id, []).append(s)
                strips_by_object.setdefault(oid, []).append(s)

        # Per-group metadata atlas. Allocate sized to each group's FBO
        # (the same dims the renderer's GroupCanvas uses), then write
        # each strip's per-LED metadata at its FBO indices. Pixels not
        # covered by any strip stay at the "no-strip" sentinel values.
        group_metadata: dict = {}
        group_dims = {g.id: (g.height, g.width) for g in self.project.groups}
        for gid, (gh, gw) in group_dims.items():
            atlas = {
                "strip_uid":     _np.full((gh, gw), -1, dtype=_np.int32),
                "strip_idx":     _np.full((gh, gw), -1, dtype=_np.int32),
                "receiver_idx":  _np.full((gh, gw), -1, dtype=_np.int32),
                "object_id":     _np.full((gh, gw), -1, dtype=_np.int32),
                "led_chain_idx": _np.full((gh, gw), -1, dtype=_np.int32),
                # Pixel-space coords on the composite canvas (NaN if
                # the project has no physical layout for this pixel).
                "pos_x":         _np.full((gh, gw), _np.nan, dtype=_np.float32),
                "pos_y":         _np.full((gh, gw), _np.nan, dtype=_np.float32),
                # Normalized to [-0.5, +0.5] per axis, origin at the
                # composite canvas center. Recommended for patterns
                # — scale-independent, doesn't change with project.
                "pos_u":         _np.full((gh, gw), _np.nan, dtype=_np.float32),
                "pos_v":         _np.full((gh, gw), _np.nan, dtype=_np.float32),
            }
            group_metadata[gid] = atlas

        for entry in strip_table:
            atlas = group_metadata.get(entry["group_id"])
            if atlas is None:
                continue
            idx = entry["pixel_indices"]
            if idx is None or len(idx) == 0:
                continue
            rows = idx[:, 0]
            cols = idx[:, 1]
            gh, gw = atlas["strip_uid"].shape
            ok = ((rows >= 0) & (rows < gh) & (cols >= 0) & (cols < gw))
            if not ok.all():
                rows = rows[ok]; cols = cols[ok]
                positions = entry["positions"][ok]
                chain = _np.arange(len(idx))[ok]
            else:
                positions = entry["positions"]
                chain = _np.arange(len(idx))
            atlas["strip_uid"][rows, cols] = entry["uid"]
            atlas["strip_idx"][rows, cols] = entry["strip_idx"]
            atlas["receiver_idx"][rows, cols] = entry["receiver_idx"]
            atlas["object_id"][rows, cols] = entry["object_id"]
            atlas["led_chain_idx"][rows, cols] = chain
            if positions.shape[0] == len(rows):
                atlas["pos_x"][rows, cols] = positions[:, 0]
                atlas["pos_y"][rows, cols] = positions[:, 1]
                atlas["pos_u"][rows, cols] = positions[:, 0] / canvas_w - 0.5
                atlas["pos_v"][rows, cols] = positions[:, 1] / canvas_h - 0.5

        # Object names + bidirectional id<->name maps. The
        # multi_object geometry provider exposes its parsed
        # ``objects: [{id, name, x, y}]`` list as ``.objects``;
        # geometries that don't model named objects (e.g. Fan) just
        # leave both maps empty. Effects/handlers can then reference
        # boxes by display name ("center", "north") instead of
        # memorising numeric ids.
        object_names: dict[int, str] = {}
        name_to_object_id: dict[str, int] = {}
        objects_yaml = list(getattr(self.geometry_provider, "objects", []) or [])
        for o in objects_yaml:
            if not isinstance(o, dict):
                continue
            try:
                oid = int(o.get("id"))
                name = (o.get("name") or "").strip()
            except (TypeError, ValueError):
                continue
            if name:
                object_names[oid] = name
                name_to_object_id[name] = oid

        # Outbound OSC targets per object_id. Built from the receiver
        # list's ``host``/``ip`` fields. Hostnames stay unresolved here
        # — ProjectOscSender resolves lazily on first send so an
        # unreachable host doesn't stall startup. Port defaults to the
        # project-level ``osc.return_port`` (default 9000).
        osc_cfg = (self.project.raw or {}).get("osc") or {}
        return_port = int(osc_cfg.get("return_port", 9000))
        object_osc_targets: dict[int, tuple[str, int]] = {}
        for rx in project_receivers:
            if not isinstance(rx, dict):
                continue
            try:
                oid = int(rx.get("object_id", -1))
            except (TypeError, ValueError):
                continue
            if oid < 0:
                continue
            host = rx.get("host") or rx.get("ip")
            if host:
                object_osc_targets[oid] = (str(host), return_port)

        # Replace any prior sender (project-swap path) with one bound
        # to this project's targets.
        from lib.osc_sender import ProjectOscSender, make_send_callable
        self._osc_sender = ProjectOscSender(
            targets_by_id=object_osc_targets,
            name_to_id=name_to_object_id,
            return_port=return_port,
        )

        st = self.scheduler.state
        st["project"] = self.project
        st["strips_by_group"] = strips_by_group
        st["strips_by_object"] = strips_by_object
        st["receivers"] = receivers_list
        st["strip_table"] = strip_table
        st["group_metadata"] = group_metadata
        st["composite_canvas_size"] = (canvas_w, canvas_h)
        # Per-project name maps + outbound OSC. Effects, button
        # handlers, and weather hooks can reach any box by name or id.
        st["object_names"] = object_names
        st["name_to_object_id"] = name_to_object_id
        st["object_osc_targets"] = object_osc_targets
        st["osc_send"] = make_send_callable(self._osc_sender)

    def swap_project(self, new_project_id: str) -> bool:
        """Hot-swap the active art-piece project.

        Must run on the render thread (apply_web_controls is the entry
        point). Pre-validates the new project so a failed load doesn't
        tear down the running one. Steps:

          1. Load + validate the new project (no side effects yet).
          2. Cancel events, stop ambient + transient audio.
          3. Resize each canvas to the new project's display dims
             (this also tears down all current effects via cleanup()).
          4. Close current DMX senders, open new ones for the new
             project's receivers.
          5. Replace geometry provider in the pipeline + web controller.
          6. Replace event_map + WeatherSetManager + WeatherStateController.
          7. Reseed the web control_dict so the UI reflects the new project.
          8. Re-initialize background events for the new active weather set.
        """
        if new_project_id == self.project.id:
            print(f"[Project] Already on '{new_project_id}'; no-op")
            return True
        print(f"[Project] Swap requested: {self.project.id} -> {new_project_id}")

        # Same rule as a weather-set change: a project swap wants the new
        # project's soundscape - the DJ must hand the soundtrack back first
        # (and never survive across a swap that resets active_effects).
        dj = getattr(self, "_dj", None)
        if dj is not None and dj.active:
            print("[DJ] project swap - stopping the DJ")
            self._dj_stop()

        # ---- Phase A: pre-validate ----
        # Loading the new project's event_map is the gating step:
        # project event_map modules reference project-local shader
        # symbols (e.g. ``fx.shader_test_bouncing_ball``) which must
        # already be on ``renderer.effects`` at import time. So we
        # swap the project-local shader namespace BEFORE importing
        # the event_map. If the import fails we restore the previous
        # project's shaders so the running show keeps rendering.
        prev_project = self.project
        try:
            new_project = load_project(new_project_id)
            new_geometry = new_project.load_geometry()
            load_project_shaders(new_project)
            try:
                new_event_map = dict(new_project.load_event_map())
            except Exception:
                # Restore prior project's shader namespace before
                # propagating so the running effects' references
                # remain valid.
                try:
                    load_project_shaders(prev_project)
                except Exception as restore_err:
                    print(f"[Project] !! shader rollback after load fail "
                          f"raised: {restore_err}")
                raise
        except Exception as e:
            print(f"[Project] Swap aborted; load failed: {e}")
            return False

        # display_height is only consumed by the legacy column-rect fallback
        # in _build_receivers. Use the new project's first group's height.
        first_group_h = new_project.groups[0].height if new_project.groups else \
            int(self._cfg["display"]["height"])

        new_receivers = self._build_receivers(
            project_receivers=new_project.raw.get("receivers"),
            legacy_receivers=self._cfg["dmx"]["receivers"],
            display_height=first_group_h,
        )

        try:
            return self._swap_project_unsafe(new_project, new_geometry, new_event_map, new_receivers)
        except Exception as e:
            import traceback
            print(f"[Project] !! swap raised mid-flight: {e}")
            traceback.print_exc()
            print("[Project] !! state may be inconsistent; subsequent swap should recover")
            return False

    def _swap_project_unsafe(self, new_project, new_geometry, new_event_map, new_receivers):
        # Phase 6: groups vary in count between projects (Fan: 1, WoL: 3).
        # Fully tear down + rebuild the canvas list rather than just
        # resizing, so canvas count can grow or shrink across swaps.
        new_groups = new_project.groups
        new_dims = [(g.width, g.height) for g in new_groups]
        new_group_ids = [g.id for g in new_groups]

        # ---- Phase B: tear down current state ----
        self.scheduler.cancel_all_events()
        engine = self.scheduler.state.get("soundengine")
        if engine is not None:
            try:
                engine.stop_all(duration=0.0)
            except Exception as e:
                print(f"[Project]   stop_all error (ignoring): {e}")
            try:
                engine.stop_ambient()
            except Exception as e:
                print(f"[Project]   stop_ambient error (ignoring): {e}")

        renderer = self.scheduler._shader_renderer
        for vp in renderer.viewports:
            try:
                vp.cleanup()
            except Exception as e:
                print(f"[Project]   viewport cleanup error (ignoring): {e}")
        renderer.viewports = []
        renderer.frame_dimensions = new_dims
        renderer.num_frames = len(new_dims)
        renderer.group_ids = list(new_group_ids)
        renderer.group_to_frame_id = {gid: i for i, gid in enumerate(new_group_ids)}
        for fid in range(len(new_dims)):
            renderer.create_viewport(fid)

        for sender in list(self.scheduler.state.get("screens", [])):
            try:
                sender.close()
            except Exception as e:
                print(f"[Project]   sender close error (ignoring): {e}")

        # ---- Phase C: build new state ----
        new_screens = []
        if new_receivers:
            sender = imdmx.SACNPixelSender(
                new_receivers, skip_network=False,
                use_raw_udp=True, per_receiver_universe=True,
                bind_ip=self._cfg["dmx"].get("bind_ip", ""),
            )
            sender.enable_async_send()
            new_screens.append(sender)
        self.scheduler.state["screens"] = new_screens

        self.geometry_provider = new_geometry
        self.scheduler.replace_geometry_provider(new_geometry)
        if self.web_controller is not None:
            self.web_controller.replace_geometry_provider(new_geometry)

        # Phase 8: refresh project-specific weather machinery before
        # rebuilding the managers so they get the new project's enum, sets,
        # default, presets, and default params.
        self._refresh_weather_module(new_project)

        # Project-local shaders were already swapped in by
        # ``swap_project``'s Phase A (must happen before the new
        # project's event_map can import). Keep it that way — calling
        # load_project_shaders here would be a no-op (idempotent for
        # the same project id) but is misleading.

        self.event_map = new_event_map
        self.weather_set = WeatherSetManager(
            self.event_map,
            weather_sets=self._weather_sets,
            default_set=self._default_weather_set,
            weather_state_enum=self._weather_state_enum,
        )
        self.weather_state = WeatherStateController(
            weather_state_enum=self._weather_state_enum,
            weather_presets=self._weather_presets,
            default_weather_params=self._default_weather_params,
            output_map=self._outstate_publish,
        )
        self.active_effects = {"world": None, "ambient_sound": None}

        # ---- Phase D: republish to the web UI ----
        if self.web_controller is not None:
            from lib.weather_set import IMPLICIT_BACKGROUND_EVENTS
            event_list = self.weather_set.get_event_names()
            bg_events = [e for e in event_list if e not in IMPLICIT_BACKGROUND_EVENTS]
            self.web_controller.set_available_events(
                all_events=event_list,
                background_events=bg_events,
            )
            # Repush the new project's weather machinery so the editor
            # endpoint serves the right enum / presets / sets after
            # the swap. Both calls invalidate the editor cache.
            self.web_controller.set_weather_module(
                weather_state_enum=self._weather_state_enum,
                weather_presets=self._weather_presets,
                weather_sets=self._weather_sets,
                default_weather_params=self._default_weather_params,
                default_weather_set=self._default_weather_set,
                weather_module_path=self._weather_module_path,
                weather_module_name=self._weather_module_name,
            )
            self.web_controller.set("current_weather_set", self.weather_set.current_set)
            self.web_controller.set("available_sets", list(self._weather_sets.keys()))
            self.web_controller.set(
                "available_weather_states",
                list(self._weather_sets[self.weather_set.current_set]["states"]),
            )
            self.web_controller.set(
                "all_weather_states",
                [s.value for s in self._weather_state_enum],
            )
            # led_width/led_height: the first group's dims (used by the
            # JS preview's flat-mode aspect calc on Fan-style projects;
            # multi-object projects override aspect via their own
            # geometry JSON anyway).
            self.web_controller.set("led_width", new_dims[0][0])
            self.web_controller.set("led_height", new_dims[0][1])
            self.web_controller.set("current_project", new_project.id)
            self.web_controller.set("current_project_name", new_project.display_name)
            self.web_controller.set("media_root", str(new_project.media_root))
            # Stale parameter overrides reference the old project's params;
            # drop them rather than carry forward what may be invalid keys.
            self.web_controller.web_param_overrides = {}

        self.project = new_project
        # Repoint state['media_root'] at the new project's media folder so
        # subsequent random_events / ambient sound lookups go to the right place.
        self.scheduler.state["media_root"] = str(new_project.media_root)
        self.scheduler.frame_dimensions = new_dims
        self.scheduler.group_ids = list(new_group_ids)

        # Per-project brightness ceiling — same logic as __init__, run
        # again on swap so each piece's PSU budget is respected.
        if new_project.brightness_limit is not None:
            self.scheduler.brightness_setpoint = new_project.brightness_limit
            self.scheduler.state["brightness_limit"] = new_project.brightness_limit
            if self.web_controller is not None:
                self.web_controller.set("brightness_limit",
                                        new_project.brightness_limit)
            print(f"[Project] brightness_limit = {new_project.brightness_limit}")

        # Per-project target FPS. Main loop re-reads
        # ``env_system.frame_time`` each tick so this updates without a
        # restart. Default 40 fps when the new project omits it.
        self.frame_time = self._compute_frame_time(new_project)
        print(f"[Project] target_fps = {1.0 / self.frame_time:.1f} "
              f"(frame_time = {self.frame_time*1000:.1f} ms)")

        # Refresh strip metadata for effects to read.
        self._publish_strip_metadata(new_receivers)
        # Resize per-canvas brightness state to match the new canvas count.
        self.scheduler.brightness_state = [
            {'divisor': 1.0, 'bright_factor': 0.0, 'last_logged_divisor': 0.0}
            for _ in new_dims
        ]
        # Re-register the project's OSC button routes against the new
        # project's name maps. The previous project's prefix handlers
        # remain on the listener (first-match-wins still routes
        # correctly because the new ones are appended); a future
        # cleanup hook can drop stale routes if duplication starts to
        # matter.
        self._call_button_router_hook()
        self._initialize_weather_set_events()

        # Receiver boot-resilience: drop any rebuild the watcher queued for the
        # OLD project, then (re)start it for the new one so its late-arriving
        # boxes recover too. Runs after self.project is the new project
        # (set earlier in this method) — _start_receiver_watch reads its id.
        with self._receiver_lock:
            self._pending_receivers = None
        self._live_receiver_count = len(new_receivers)
        self._start_receiver_watch()

        dims_str = ", ".join(f"{gid}={w}x{h}" for gid, (w, h) in zip(new_group_ids, new_dims))
        print(f"[Project] Swap complete: now {new_project.display_name} [{dims_str}]")
        return True

    def _call_button_router_hook(self) -> None:
        """Run the active project's ``button_router`` hook (if any).

        Convention: project.yaml declares ``hooks: { button_router: <mod> }``
        and that module exposes ``register(env_system)``. Called once at
        boot AND after every project swap so a hot-swap to a new project
        re-registers OSC routes against the new project's name maps.

        The router is responsible for clearing any prior project's
        prefix handlers from ``self.osc_listener`` if it cares about
        idempotence — for the typical PoC case (re-register on top of
        the previous list) the listener happily fans out duplicate
        prefixes; first-match-wins still reaches the right handler.
        """
        hook = self.project.load_hook("button_router")
        if hook is None:
            return
        register = getattr(hook, "register", None)
        if not callable(register):
            print(f"[Project] button_router hook missing register(env_system) "
                  f"function; skipping")
            return
        try:
            register(self)
        except Exception as e:
            print(f"[Project] button_router hook failed: {e}")
            import traceback; traceback.print_exc()

    def _initialize_weather_set_events(self):
        """Cancel all events and start background events for the current weather set"""
        print(f"[WEATHER] Initializing events for weather set: '{self.weather_set.current_set}'")

        # Cancel all active events and fade out all audio
        self.scheduler.cancel_all_events()
        engine = self.scheduler.state.get("soundengine")
        if engine:
            engine.stop_all(duration=2.0)

        # Belt-and-suspenders: cancel_all_events relies on each event's
        # count==-1 wrapper to remove ITS OWN effect from the canvas. If any
        # wrapper deviates from that contract, or its viewport.effects.remove()
        # throws, the effect is ORPHANED -- left in the canvas's render list
        # and still drawing, now fed the NEW set's params. That renders as
        # unrecognizable geometry/colors / the previous set "blending through",
        # and it compounds the longer a session runs (orphans accumulate).
        # Authoritatively clear every canvas so a set switch always starts
        # from a clean slate. A non-zero count here means the leak was real.
        renderer = getattr(self.scheduler, '_shader_renderer', None)
        if renderer is not None:
            orphans = 0
            for vp in getattr(renderer, 'viewports', []):
                effects = getattr(vp, 'effects', None)
                if not effects:
                    continue
                orphans += len(effects)
                try:
                    vp.canvas._make_current()
                except Exception:
                    pass
                for eff in list(effects):
                    try:
                        eff.cleanup()
                    except Exception:
                        pass
                effects.clear()
            if orphans:
                print(f"[WEATHER] Leak guard: force-cleared {orphans} orphaned "
                      f"effect(s) left on canvas after cancel_all_events")

        # Schedule the permanent background events based on set configuration
        sim_forever = 10E9  # 10 billion seconds (over 300 years)

        for event_name in self.weather_set.get_background_events():
            print(f"[WEATHER]   Background event: {event_name}")
            self._schedule_event_from_map(event_name, 0, sim_forever, frame_id=0)

        # Per-set audio source. A set that declares ``audio_source`` (e.g.
        # WoL's "Elements" set uses "internal" so its own music drives the
        # audio-reactive visuals) switches the analyzer when activated. Sets
        # that omit it leave the machine-configured source untouched.
        src = self.weather_set.get_audio_source()
        if src:
            self.set_audio_source(src)

        print(f"[WEATHER] Background events initialized for '{self.weather_set.current_set}'")
    
    def _schedule_event_from_map(self, event_name: str, start_time: float, duration: float, frame_id: int = 0):
        """Schedule an event from the event map.

        If the event_map entry includes a ``{"group": "..."}`` meta dict,
        the group name is translated to the matching canvas's frame_id
        (overriding the default 0). Unknown group names fall back to the
        passed ``frame_id`` with a warning. Phase 6+ effects use this to
        target one canvas; legacy 2-tuple entries route to frame_id=0
        ("main") on Fan, the default group on any project.
        """
        entry = self.weather_set.resolve_event(event_name)
        if entry is None:
            print(f"[WEATHER] Unknown event: {event_name}")
            return None

        effect_func, params, meta = entry
        target_group = meta.get("group") if isinstance(meta, dict) else None
        if target_group is not None:
            mapping = self.scheduler._shader_renderer.group_to_frame_id
            if target_group in mapping:
                frame_id = mapping[target_group]
            else:
                print(f"[WEATHER]   '{event_name}' targets unknown group "
                      f"{target_group!r}; falling back to frame_id={frame_id}")
        return self.scheduler.schedule_event(start_time, duration, effect_func, frame_id=frame_id, **params)

    def transition_to_weather(self, new_weather, transition_duration: float = 10.0):
        """Start a transition to a new weather state"""
        # Restart the deterministic state-duration timer: it re-arms once
        # this new state's transition completes (progress >= 1.0). See
        # random_state_change.
        self._state_hold_start = None
        target_params = self.weather_state.start_transition(new_weather, transition_duration, time.time())
        
        # Schedule events based on on_transition_events in weather preset.
        # Entry schema: [name, duration, delay=0, frame_id=0]. The third
        # slot is a START DELAY in seconds — every preset author wrote it
        # that way (staggered lightning strikes, a whale 20s after the
        # dolphins) — but it was historically consumed as a frame_id,
        # which silently no-op'd those entries on single-canvas projects
        # (the viewport lookup failed). frame_id moved to the 4th slot.
        on_transition_events = target_params.get("on_transition_events", [])
        for event_config in on_transition_events:
            if isinstance(event_config, (tuple, list)) and len(event_config) >= 2:
                event_name, duration = event_config[:2]
                delay = float(event_config[2]) if len(event_config) > 2 else 0.0
                frame_id = int(event_config[3]) if len(event_config) > 3 else 0
                tag = f" +{delay:.0f}s" if delay else ""
                print(f"[WEATHER]   Transition event: {event_name} ({duration}s{tag})")
                self._schedule_event_from_map(event_name, delay, duration, frame_id=frame_id)
            else:
                print(f"[WEATHER]   Invalid on_transition_event format: {event_config!r}")

        ambient_sound = target_params.get("ambient_sound")
        skip_time = target_params.get("skiptime", 0.0)
        ari = target_params.get("ARI", 0.0)
        engine = self.scheduler.state["soundengine"]
        dj = getattr(self, "_dj", None)
        if dj is not None and dj.active:
            # The DJ owns the soundtrack: weather changes track the visuals
            # but must not start ambient beds under the mix. The new state's
            # ambient is remembered and restored by _dj_stop.
            self.active_effects["ambient_sound"] = ambient_sound
        elif ambient_sound:
            media_root = Path(self.scheduler.state.get("media_root", "media"))
            sound_path = media_root / "sounds" / ambient_sound
            engine.play_ambient(sound_path, skip_seconds=skip_time, ari=ari)
            self.active_effects["ambient_sound"] = ambient_sound
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

        # Check for project swap first — it invalidates everything else
        # below, so we handle and return early on swap.
        with self.web_controller._dict_lock:
            request_swap = self.web_controller.control_dict.pop('request_project_swap', None)
        if request_swap:
            self.swap_project(request_swap)
            return

        # Check for weather set/state change requests (read and clear atomically)
        with self.web_controller._dict_lock:
            new_set = self.web_controller.control_dict.pop('request_weather_set', None)
            new_state = self.web_controller.control_dict.pop('request_weather_state', None)
            trigger_event = self.web_controller.control_dict.pop('request_trigger_event', False)
            audio_source_req = self.web_controller.control_dict.pop('request_audio_source', None)

        if audio_source_req is not None:
            self.set_audio_source(audio_source_req)

        # Bluetooth sink: drain queued web actions + mirror live state.
        self._apply_bluetooth_controls()

        # Autonomous DJ: drain queued web actions + mirror status.
        self._apply_dj_controls()

        if new_set is not None and new_set != self.weather_set.current_set:
            self.change_weather_set(new_set, immediate=True)

        if new_state is not None:
            locked = self.web_controller.get('state_switch_locked', True)
            all_states = [s.value for s in self._weather_state_enum]
            if new_state in all_states and (not locked or new_state in [s.value for s in self.weather_set.get_set_states()]):
                state_enum = self._weather_state_enum(new_state)
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
                duration = float(self.weather_set.get_current_set_config()
                                 .get("random_event_duration", 60))
                print(f"[WEB] Triggered event: {event_name} ({duration:.0f}s)")
                self._schedule_event_from_map(event_name, 0, duration, frame_id=0)

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
                st = self.scheduler.state
                audio_summary = {
                    "bands": current_bands.tolist(),
                    "peak_band": int(np.argmax(current_bands)),
                    "total_power": float(np.sum(current_bands)),
                    "sensitivity": self.analyzer.sensitivity,
                    "source": getattr(self.analyzer, "_active_source", None),
                    # Rhythm/structure signals for the club page's meters.
                    "beat_decay": float(st.get("beat_decay", 0.0) or 0.0),
                    "punch": [float(st.get(k, 0.0) or 0.0)
                              for k in ("bass_punch", "mid_punch", "high_punch")],
                    "energy": float(st.get("audio_energy", 0.0) or 0.0),
                    "build": float(st.get("build_level", 0.0) or 0.0),
                    "drop_decay": float(st.get("drop_decay", 0.0) or 0.0),
                    "health": self.analyzer.get_input_health(),
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
                d['fps_target'] = getattr(self, '_target_fps', 0)
                d['fps_uncapped'] = getattr(self, '_uncapped_fps', 0)
                # Club-page telemetry: only meaningful while the club set is
                # active (the page and its nav tab hide otherwise).
                d['club_director_info'] = (
                    self.scheduler.state.get('club_director_info')
                    if self.weather_set.current_set == 'club' else None)
                d['active_effects'] = active_effects
                d['ambient_sound'] = self.active_effects.get("ambient_sound")
                d['allowed_output_params'] = self._get_allowed_output_params()
                # Only surface narrative variables when the active set
                # actually has a narrative script. Otherwise the stale
                # list from a previous set (e.g. switching cyberpunk →
                # forest) would hang around because NarrativePlayer's
                # update() short-circuits when disabled.
                if self.weather_set.get_narrative_script():
                    d['narrative_vars'] = list(self.scheduler.state.get('narrative_vars', []))
                else:
                    d['narrative_vars'] = []
                self.web_controller._values_cache = None  # Invalidate cache
            finally:
                self.web_controller._dict_lock.release()
            self._last_status_update = self.current_time
    
    # Club-page steering settings auto-release so a forgotten setting can't
    # shape the whole night (anti-rut). Re-tapping restarts the clock.
    CLUB_THEME_TTL_S = 20 * 60.0
    CLUB_HOLD_TTL_S = 12 * 60.0

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
        set_config = self._weather_sets.get(self.weather_set.current_set, {})
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

        # Season can be manually locked to a user-chosen value via the web UI.
        season_locked = False
        season_override = None
        club_heat_bias = 0.0
        club_hold = False
        club_force = False
        club_palette_override = None
        club_density = 1.0
        club_theme = ''
        if self.enable_web_control and self.web_controller is not None:
            with self.web_controller._dict_lock:
                season_locked = bool(self.web_controller.control_dict.get('season_locked', False))
                season_override = self.web_controller.control_dict.get('season_override')
                # Club-page operator controls (consumed by club_director).
                club_heat_bias = float(self.web_controller.control_dict.get('club_heat_bias', 0.0) or 0.0)
                club_hold = bool(self.web_controller.control_dict.get('club_hold_room', False))
                club_palette_override = self.web_controller.control_dict.get('club_palette_override')
                club_density = float(self.web_controller.control_dict.get('club_density', 1.0) or 1.0)
                club_theme = self.web_controller.control_dict.get('club_theme', '') or ''
                if self.web_controller.control_dict.get('club_force_drop'):
                    club_force = True          # one-shot: consume the flag
                    self.web_controller.control_dict['club_force_drop'] = False

                # Anti-rut: theme and hold auto-release so a setting made at
                # 11pm and forgotten can't shape the whole night. Re-tapping
                # on the club page restarts the clock.
                cd = self.web_controller.control_dict
                now_t = time.time()
                ttls = {}
                if club_theme:
                    if cd.get('_club_theme_val') != club_theme:
                        cd['_club_theme_val'] = club_theme
                        cd['_club_theme_since'] = now_t
                    remain = self.CLUB_THEME_TTL_S - (now_t - cd.get('_club_theme_since', now_t))
                    if remain <= 0:
                        cd['club_theme'] = ''
                        club_theme = ''
                    else:
                        ttls['theme'] = int(remain)
                else:
                    cd['_club_theme_val'] = ''
                if club_hold:
                    if not cd.get('_club_hold_since'):
                        cd['_club_hold_since'] = now_t
                    remain = self.CLUB_HOLD_TTL_S - (now_t - cd['_club_hold_since'])
                    if remain <= 0:
                        cd['club_hold_room'] = False
                        club_hold = False
                        cd['_club_hold_since'] = 0
                    else:
                        ttls['hold'] = int(remain)
                else:
                    cd['_club_hold_since'] = 0
                cd['club_ttls'] = ttls

        # Season resolution order: manual lock > set-driven autopilot >
        # wall-clock ramp. The autopilot hook lets a set's own logic drive
        # its season (the club director advances the night phase by how
        # hard the floor is actually going); the freshness stamp makes a
        # stale value from a departed set fall back to the clock.
        sstate = self.scheduler.state
        _auto = sstate.get('_season_autopilot')
        _auto_fresh = (time.time() - sstate.get('_season_autopilot_t', 0.0)) < 2.0
        if season_locked and season_override is not None:
            self.season = float(season_override) % 1.0
        elif _auto is not None and _auto_fresh:
            self.season = float(_auto) % 1.0
        else:
            self.season = ((time.time() / 1800) * season_speed) % 1

        state = self.scheduler.state
        state['club_heat_bias'] = club_heat_bias
        state['club_hold_room'] = club_hold
        state['club_force_drop'] = club_force
        state['club_palette_override'] = club_palette_override
        state['club_density'] = club_density
        state['club_theme'] = club_theme
        output = self.weather_state.get_state_output(
            self.season, self.current_time,
            self.weather_set.get_season_atmosphere_coupling())

        # Apply global modifiers and overrides to the output (not to weather_params)
        if self.enable_web_control and self.web_controller is not None:
            with self.web_controller._dict_lock:
                intensity = self.web_controller.global_modifiers.get('weather_intensity', 1.0)
                brightness_mod = self.web_controller.global_modifiers.get('brightness', 1.0)
                gamma_mod = self.web_controller.global_modifiers.get('gamma', 2.0)
                brightness_limit_mod = self.web_controller.global_modifiers.get('brightness_limit', 0.1)
                overrides = dict(self.web_controller.web_param_overrides)

            # Scale weather intensity on output keys
            if intensity != 1.0:
                for key in self.WEATHER_INTENSITY_KEYS:
                    if key in output:
                        output[key] = output[key] * intensity

            # Apply direct overrides to output (these replace values entirely)
            # Narrative variables (story_*) live in state, not output — they
            # are published by NarrativePlayer each frame, so a web override
            # is applied directly to state and stomps the player's value.
            for param, value in overrides.items():
                if param in output:
                    output[param] = value
                elif param.startswith('story_'):
                    state[param] = value

            # Store brightness modifier in state for render pipeline to apply
            # after the hardware limiter (can only dim, never brighten past limiter)
            state["web_brightness"] = brightness_mod
            state["web_gamma"] = gamma_mod
            state["brightness_limit"] = brightness_limit_mod

            # Volume controls — all applied in real time in the audio
            # engine mixer. Master + narrative come from the web UI's
            # global_modifiers; ambient comes from the active weather
            # state's Sound_volume param (blended during transitions
            # by WeatherStateController). Without the ambient hookup
            # Sound_volume was edit-but-no-effect — visible in the
            # weather editor but never reaching playback.
            master_vol = self.web_controller.global_modifiers.get('master_volume', 1.0)
            narrative_vol = self.web_controller.global_modifiers.get('narrative_volume', 1.0)
            soundpool_vol = self.web_controller.global_modifiers.get('soundpool_volume', 1.0)
            ambient_vol = float(self.weather_state.weather_params.get('Sound_volume', 1.0))
            state["soundengine"].master_volume = master_vol
            state["soundengine"].narrative_volume = narrative_vol
            state["soundengine"].soundpool_volume = soundpool_vol
            state["soundengine"].ambient_volume = ambient_vol

            # Cache the final output for the web UI snapshot (post-overrides)
            self._last_web_output = output

        state.update(output)
        state["season"] = self.season
        state["current_weather_state"] = self.weather_state.current_weather.value
        # Publish the active set's narrative script path (or None). The
        # narrative_player background event reads this and reloads when it
        # changes, so switching weather sets cleanly swaps scripts.
        state["narrative_script"] = self.weather_set.get_narrative_script()
        # Same deal for the random ambient-sound pool directory.
        state["sound_pool_dir"] = self.weather_set.get_sound_pool_dir()
        # Opt-in crossfade window (seconds) for the sound pool: 0 = original
        # gap playback, >0 = gapless crossfaded stream.
        state["sound_pool_crossfade"] = self.weather_set.get_sound_pool_crossfade()
        if self.enable_web_control and self.web_controller is not None:
            state["_preview_active"] = (
                self.web_controller.control_dict.get('_preview_subscribers', 0) > 0
            )
        else:
            state["_preview_active"] = False
        state["scale"] = self.scale
        # Wire the AudioEngine -> analyzer monitor tap once (drives the
        # 'internal' audio source). feed() no-ops unless that source is active,
        # so it's safe to keep wired regardless of the current source.
        if self.analyzer is not None and not self._audio_tap_wired:
            engine = state.get("soundengine")
            if engine is not None:
                engine.set_monitor_tap(self.analyzer.feed)
                self._audio_tap_wired = True

        state["sound"] = self.analyzer.get_extended_analysis() if self.analyzer else None
        # Raw time-domain waveform (128 samples, AGC-normalized) for the
        # MilkDrop-style oscilloscope shaders. None when no analyzer.
        state["waveform"] = self.analyzer.get_waveform() if self.analyzer else None
        # 12-bin pitch-class distribution for harmony-aware shaders.
        state["chroma"] = self.analyzer.get_chroma() if self.analyzer else None
        # Beat / tempo detection over the analyzer output. Published so any
        # shader can sync to the beat (Club set's lasers/strobe/eq_bars).
        # Returns zeros when there's no audio, so this is always safe.
        now = self.current_time
        audio_dt = now - self._prev_beat_time
        beat = self._beat_detector.update(state["sound"], audio_dt)
        self._prev_beat_time = now
        state["beat"] = beat["onset"]
        state["beat_decay"] = beat["decay"]
        state["bpm"] = beat["bpm"]
        state["beat_phase"] = beat["phase"]
        state["beat_intensity"] = beat["strength"]
        state["beat_confidence"] = beat["confidence"]
        state["beat_count"] = beat["count"]
        state["bar_phase"] = beat["bar_phase"]
        state["phrase_phase"] = beat["phrase_phase"]
        # Song-structure signals (see lib/audio_signals.py). Same dt as the
        # beat detector; safe zeros when there's no audio.
        sig = self._audio_structure.update(state["sound"], beat, audio_dt)
        state["audio_bass"] = sig["bass"]
        state["audio_mid"] = sig["mid"]
        state["audio_high"] = sig["high"]
        state["bass_punch"] = sig["bass_punch"]
        state["mid_punch"] = sig["mid_punch"]
        state["high_punch"] = sig["high_punch"]
        state["audio_punch"] = sig["punch"]
        state["audio_energy"] = sig["energy"]
        # When the autonomous DJ is playing, it KNOWS the energy of what's
        # on the decks (per-track level x playhead energy curve) - the DSP
        # estimate reads every steady track as ~medium (AGC bands hover at
        # 1.0), which left the club unable to tell chill from peak. Ground
        # truth wins; the DSP value remains the mic-mode fallback. Smoothed
        # here (~0.8s) - just enough to hide the 2 Hz curve steps.
        dj_e = state.get("dj_energy")
        if state.get("dj_active") and dj_e is not None:
            prev = getattr(self, "_dj_energy_sm", None)
            k = 1.0 - float(np.exp(-(audio_dt or 0.025) / 0.8))
            sm = dj_e if prev is None else prev + (float(dj_e) - prev) * k
            self._dj_energy_sm = sm
            state["audio_energy"] = sm
        else:
            self._dj_energy_sm = None
        state["build_level"] = sig["build"]
        # DJ FOREKNOWLEDGE -> deterministic build: the decks publish the
        # next drop/seam ETA; ramp build_level through the final 8s so
        # every pattern's coil-up (and the director's squeeze) lands
        # BEFORE every known drop - the DSP riser detector only catches
        # builds the mastering makes obvious.
        dj_eta = state.get("dj_next_drop_eta")
        if state.get("dj_active") and dj_eta is not None and dj_eta < 8.0:
            state["build_level"] = max(state["build_level"],
                                       min(1.0, 1.0 - dj_eta / 8.0))
        state["drop"] = sig["drop"]
        state["drop_decay"] = sig["drop_decay"]
        # DJ ground-truth drops: the decks KNOW when a drop section lands
        # (published as a wall-time stamp). Fire the same drop/drop_decay
        # signals the DSP detector would - hard sets never give the DSP
        # path the quiet episode it needs to arm, so without this the
        # club sat still through the hardest-hitting moments.
        ddt = state.get("dj_drop_t")
        if (state.get("dj_active") and ddt
                and ddt != getattr(self, "_dj_drop_seen", None)):
            self._dj_drop_seen = ddt
            self._dj_drop_env = 1.0
            state["drop"] = True
        env = getattr(self, "_dj_drop_env", 0.0)
        if env > 0.001:
            state["drop_decay"] = max(state["drop_decay"], env)
            self._dj_drop_env = env * float(
                np.exp(-(audio_dt or 0.025) / 0.35))
        # GROUND-TRUTH BEAT: while the DJ plays, the audible deck's stored
        # grid IS the beat - sample-tight bpm/phase/bar/phrase for every
        # beat-synced shader, where the DSP detector on the mix lags and
        # quantizes ('doesn't respond to beats in a clear manner' - user).
        # Punches get a grid-pulse FLOOR scaled by the section's bass
        # share: relentless hard sets flatten the AGC punch envelopes
        # exactly when the floor hits hardest.
        lb = None
        if self._dj is not None and self._dj.active:
            try:
                lb = self._dj.live_beat()
            except Exception:
                lb = None
        if lb is not None:
            state["bpm"] = lb["bpm"]
            ph = lb["phase"]
            prev_ph = getattr(self, "_dj_beat_prev", None)
            onset = prev_ph is not None and ph < prev_ph - 0.5
            self._dj_beat_prev = ph
            drive = lb.get("drive", 1.0)
            if onset and drive >= 0.2:
                self._dj_beat_env = drive
            benv = getattr(self, "_dj_beat_env", 0.0)
            # Phases/bpm stay grid-true through breakdowns (motion should
            # keep gliding); PULSES follow the section's actual rhythm -
            # a resting kick must not flash the room.
            state["beat"] = bool(state["beat"]) or (onset and drive >= 0.2)
            state["beat_decay"] = max(state["beat_decay"], benv)
            state["beat_phase"] = ph
            state["bar_phase"] = lb["bar_phase"]
            state["phrase_phase"] = lb["phrase_phase"]
            state["beat_confidence"] = max(state["beat_confidence"], 0.95)
            pulse = benv * min(1.0, lb["bass_share"] * 2.5)
            state["beat_intensity"] = max(state["beat_intensity"], pulse)
            state["bass_punch"] = max(state["bass_punch"], pulse)
            state["audio_punch"] = max(state["audio_punch"], pulse)
            self._dj_beat_env = benv * float(
                np.exp(-(audio_dt or 0.025) / 0.18))
        else:
            self._dj_beat_prev = None
            self._dj_beat_env = 0.0
        state["music_mood"] = sig["mood"]
        state["music_perc"] = sig["perc"]
        state["rhythm_density"] = sig["density"]
        # DIAGNOSTIC (AUDIO_DEBUG=1): the EXACT reactive values reaching the
        # patterns, whatever the source (loopback / internal-DJ / mic). One
        # line/sec ends the guesswork about where reactivity dies.
        if getattr(self, "_audio_dbg", None) is None:
            import os as _os
            self._audio_dbg = _os.environ.get("AUDIO_DEBUG") == "1"
            self._audio_dbg_n = 0
        if self._audio_dbg:
            self._audio_dbg_n += 1
            if self._audio_dbg_n % 40 == 0:
                snd = state.get("sound") or {}
                dj = bool(getattr(self, "_dj", None) and self._dj.active)
                print(f"[AUDIO] dj={dj} src={getattr(self.analyzer,'_active_source','?')} "
                      f"gate={float(snd.get('gate', -1)):.2f} "
                      f"energy={state.get('audio_energy',0):.2f} "
                      f"bass_p={state.get('bass_punch',0):.2f} "
                      f"mid_p={state.get('mid_punch',0):.2f} "
                      f"high_p={state.get('high_punch',0):.2f} "
                      f"beat_d={state.get('beat_decay',0):.2f} "
                      f"drop_d={state.get('drop_decay',0):.2f} "
                      f"cE={state.get('club_energy',0):.2f} "
                      f"bpm={state.get('bpm',0):.0f} conf={state.get('beat_confidence',0):.2f}")
        key = self._harmonic_tracker.update(state.get("chroma"), audio_dt)
        state["key_center"] = key["center"]
        state["key_strength"] = key["strength"]
        state["key_changed"] = key["changed"]
        state["celestial_bodies"] = self.celestial_bodies

    def random_events(self):
        """Per-frame random-event roll.

        Two responsibilities, both project-aware:

        1. Generic: trigger one of the active set's ``random_events`` list
           based on its ``random_event_rate`` and the current season. This
           works for any project that defines a ``random_events:`` list in
           a weather set.
        2. Project hook: any further random scheduling that's project-
           specific lives in a ``random_events`` hook module declared in
           project.yaml's ``hooks:`` block. Fan declares one in
           ``projects.fan.random_events``; WoL doesn't declare one so
           nothing extra fires there.

        The legacy global gate ``enable_random_events: false`` in
        project.yaml still short-circuits both halves for projects that
        want zero random scheduling.
        """
        if not self.project.raw.get("enable_random_events", True):
            return

        # ---- generic set-level random events ----
        random_events, random_event_rate = self.weather_set.get_random_events_config()
        if random_events:
            randcheck = np.random.random()
            if randcheck < random_event_rate:
                num_events = len(random_events)
                event_positions = np.linspace(0, 1, num_events, endpoint=False)
                seasonal_distances = np.abs(event_positions - self.season)
                seasonal_distances = np.minimum(seasonal_distances, 1 - seasonal_distances)
                closest_index = np.argmin(seasonal_distances)
                event_name = random_events[closest_index]
                # Per-set dwell time; 60 s when the set doesn't specify.
                duration = float(self.weather_set.get_current_set_config()
                                 .get("random_event_duration", 60))
                print(f"   🎲 Seasonal event triggered: {event_name} "
                      f"(season: {self.season:.3f}, "
                      f"position: {event_positions[closest_index]:.3f}, "
                      f"{duration:.0f}s)")
                self._schedule_event_from_map(event_name, 0, duration, frame_id=0)

        # ---- project-specific hook ----
        hook = self.project.load_hook("random_events")
        if hook is not None:
            try:
                hook.run(self)
            except Exception as e:
                print(f"[Project] random_events hook error: {e}")

    # ----------------------------------------------------------------
    # Audio input source switching (linein / loopback / internal / mic)
    # ----------------------------------------------------------------
    _AUDIO_SOURCES = ["linein", "loopback", "internal", "microphone", "bluetooth"]

    def set_audio_source(self, source):
        """Switch the audio-reactive input source at runtime."""
        if self.analyzer is None:
            return
        if source not in self._AUDIO_SOURCES:
            print(f"[Audio] unknown source '{source}' (valid: {self._AUDIO_SOURCES})")
            return
        self.analyzer.set_source(source)
        if self.enable_web_control and self.web_controller is not None:
            self.web_controller.set("audio_source", source)

    def _apply_bluetooth_controls(self):
        """Drain queued Bluetooth web actions and mirror receiver state.

        Called from apply_web_controls (~5 Hz). Owns the BluetoothAudioReceiver
        so all BlueZ interaction stays on the app's control path. On a non-Linux
        host the receiver is an inert stub, so this just reports unavailable.
        """
        bt = getattr(self, "bt_receiver", None)
        if bt is None or self.web_controller is None:
            return

        # 1. Drain queued actions (enable / disable / approve / deny).
        with self.web_controller._dict_lock:
            actions = self.web_controller.control_dict.pop(
                'request_bluetooth_actions', [])
        for action, arg in actions:
            try:
                if action == 'enable':
                    bt.enable()
                elif action == 'disable':
                    bt.disable()
                elif action == 'approve':
                    bt.approve(arg)
                elif action == 'deny':
                    bt.deny(arg)
            except Exception as e:
                print(f"[BT] action '{action}' failed: {e}")

        # 2. Auto-route the analyzer to/from the live BT stream as devices
        #    connect and drop, and keep the capture-node hint current.
        connected = bt.connected_devices()
        n = len(connected)
        if self.analyzer is not None:
            self.analyzer.set_bluetooth_hint(bt.source_hint())
            cur = getattr(self.analyzer, "_active_source", None)
            if n > 0 and self._bt_last_connected == 0 and cur != "bluetooth":
                # First device connected: switch to it, remember where to return.
                self._bt_prev_source = cur
                self.set_audio_source("bluetooth")
            elif n == 0 and self._bt_last_connected > 0 and cur == "bluetooth":
                # Last device dropped: fall back to the prior source.
                self.set_audio_source(self._bt_prev_source or "linein")
        self._bt_last_connected = n

        # 3. Mirror state into the web control_dict for the UI snapshot.
        self.web_controller.set('bt_available', bool(getattr(bt, 'available', False)))
        self.web_controller.set('bt_enabled', bt.is_enabled())
        self.web_controller.set('bt_pending', bt.pending_pairings())
        self.web_controller.set('bt_connected', connected)
        if not getattr(bt, 'available', False):
            self.web_controller.set(
                'bt_unavailable_reason', getattr(bt, 'unavailable_reason', ''))

    # ----------------------------------------------------------------
    # Autonomous DJ bridge (web -> lib/dj/system.py), ~5 Hz
    # ----------------------------------------------------------------

    def _apply_dj_controls(self):
        """Drain queued DJ web actions, mirror status, publish outstate keys.

        The DJ owns the soundtrack while active: state ambient is silenced
        (see the guard in _trigger_state_ambient) and the analyzer follows
        the mix via the 'internal' monitor tap, so every audio-reactive
        shader dances to what the DJ is actually playing.
        """
        if self.web_controller is None or not self.dj_cfg.get("enabled", False):
            return
        with self.web_controller._dict_lock:
            actions = self.web_controller.control_dict.pop(
                'request_dj_actions', [])
        for action, arg in actions:
            try:
                if action == 'start':
                    self._dj_start()
                elif action == 'stop':
                    self._dj_stop()
                elif action == 'theme':
                    # Works idle (arms the start theme) or live (retheme).
                    self.dj_cfg['theme'] = str(arg)
                    if self._dj is not None and self._dj.active:
                        self._dj.set_theme(str(arg))
                elif action == 'persona':
                    # Works idle (arms the start persona) or live (the
                    # night changes character on the next pick).
                    self.dj_cfg['persona'] = str(arg)
                    if self._dj is not None and self._dj.active:
                        self._dj.set_persona(str(arg))
                elif action in ('setlist', 'setlist_pool'):
                    # Idle: arm the setlist to load on start. Live: load now.
                    # 'setlist' plays the list in order; 'setlist_pool'
                    # confines the DJ to the list but lets it steer the
                    # order (arc / flavor / nudge all apply).
                    mode = 'pool' if action == 'setlist_pool' else 'order'
                    name = str(arg or '') or None
                    self._dj_pending_setlist = (name, mode) if name else None
                    self._dj_idle_vocab = (0.0, [])   # re-scope the chips
                    if self._dj is not None and self._dj.active:
                        self._dj.load_setlist(str(arg or ''), mode=mode)
                elif action in ('nudge', 'arc') and (
                        self._dj is None or not self._dj.active):
                    # IDLE STEERING: arm now, applied the moment START hits.
                    if action == 'nudge':
                        self._dj_pending_nudge = float(arg)
                    else:
                        self._dj_pending_arc = list(arg or [])
                elif action == 'flavor':
                    # Live music-type steering (tag leans / axis pulls).
                    self._dj_pending_flavor = dict(arg or {})
                    if self._dj is not None and self._dj.active:
                        self._dj.set_flavor(self._dj_pending_flavor)
                elif self._dj is not None and self._dj.active:
                    if action == 'skip':
                        self._dj.request_skip()
                    elif action == 'autopilot':
                        self._dj.set_autopilot(bool(arg))
                    elif action == 'nudge':
                        self._dj_pending_nudge = float(arg)
                        self._dj.set_energy_nudge(float(arg))
                    elif action == 'pulse':
                        self._dj.set_energy_pulse(float(arg))
                    elif action == 'next_id':
                        self._dj.request_next(int(arg))
                    elif action == 'seek':
                        self._dj.seek(float(arg))
                    elif action == 'seek_rel':
                        self._dj.seek_relative(float(arg))
                    elif action == 'to_exit':
                        self._dj.to_exit()
                    elif action == 'hold':
                        self._dj.hold()
                    elif action == 'reroll':
                        self._dj.reroll_next()
                    elif action == 'seam_fb':
                        self._dj.seam_feedback(bool(arg))
                    elif action == 'arc':
                        self._dj_pending_arc = list(arg or [])
                        self._dj.set_arc_waypoints(arg or [])
                    elif action == 'moment':
                        self._dj.moment()
                    elif action == 'mix_now':
                        self._dj.mix_now()
                    elif action == 'abort':
                        self._dj.abort_transition()
            except Exception as e:
                print(f"[DJ] action '{action}' failed: {e}")

        # Mirror into the web snapshot + scheduler outstate.
        if self._dj is not None:
            info = self._dj.status()
            info.pop("deck_telemetry", None)   # heavy; web uses compact 'decks'
            info["available"] = True
            info["active"] = self._dj.active
            for k, v in self._dj.outstate_keys().items():
                self.scheduler.state[k] = v
        else:
            from lib.dj.persona import PERSONAS
            info = {"available": True, "active": False, "state": "idle",
                    "theme": self.dj_cfg.get("theme", "groove"),
                    "persona_mode": self.dj_cfg.get("persona", "auto"),
                    "personas": [(p.name, p.tagline)
                                 for p in PERSONAS.values()
                                 if p.name != "neutral"],
                    "autopilot": True, "energy_nudge": 0.0,
                    "arc_phase": 0.0, "arc_heat": 0.5,
                    "setlist": (self._dj_pending_setlist[0]
                                if isinstance(self._dj_pending_setlist,
                                              tuple)
                                else self._dj_pending_setlist),
                    "setlist_mode": (self._dj_pending_setlist[1]
                                     if isinstance(self._dj_pending_setlist,
                                                   tuple) else "order"),
                    "setlists": self._dj_list_setlists(),
                    "music_dir": self._dj_music_dir_display(),
                    "error": self._dj_last_error}
            # PRE-START STEERING: the page must offer the whole steering
            # surface BEFORE the start button - themes, flavor chips (tag
            # vocab straight from the DB), and a draggable arc, all armed
            # and applied the moment the DJ starts.
            info.update(self._dj_idle_steer_info())
            self.scheduler.state['dj_active'] = False
        # DELTA payloads: heavy blobs ship only when they change (weak
        # venue Wi-Fi shouldn't carry an identical arc curve at 5 Hz).
        import json as _json
        if not hasattr(self, '_dj_sent'):
            self._dj_sent = {}
            self._dj_sent_n = 0
        # Every ~3s ship the FULL payload regardless: the sent-cache is
        # server-global, so without this a freshly loaded page never
        # receives fields that haven't changed since some earlier client
        # saw them (reload -> empty flavor chips; user-reported).
        self._dj_sent_n += 1
        if self._dj_sent_n % 15 == 0:
            self._dj_sent = {}
        for k in ('arc_curve', 'track_map', 'next_map', 'horizon',
                  'history', 'tags', 'themes', 'setlists'):
            if k in info:
                h = hash(_json.dumps(info[k], sort_keys=True, default=str))
                if self._dj_sent.get(k) == h:
                    del info[k]
                else:
                    self._dj_sent[k] = h
        self.web_controller.set('dj_info', info)

    def _dj_idle_steer_info(self):
        import time as _t
        import json as _json
        from lib.dj.themes import BUILTIN_THEMES, get_theme
        stamp, vocab = self._dj_idle_vocab
        if _t.time() - stamp > 30.0:
            vocab = []
            try:
                from lib.dj.db import LibraryDB
                from lib.dj import resolve_music_dir
                db = LibraryDB(resolve_music_dir(
                    self.dj_cfg.get('music_dir', '')))
                # An armed setlist scopes the chips to ITS songs, so the
                # prep page steers with the vocabulary of what will play.
                scope = None
                if self._dj_pending_setlist:
                    nm = (self._dj_pending_setlist[0]
                          if isinstance(self._dj_pending_setlist, tuple)
                          else self._dj_pending_setlist)
                    try:
                        from lib.dj.setlist import get_setlist
                        sl = get_setlist(db, name=nm)
                        if sl:
                            scope = {e['track_id'] for e in sl['entries']}
                    except Exception:
                        pass
                user, auto = {}, {}
                for r in db.conn.execute("SELECT track_id, tag FROM tags"):
                    if scope is not None and r['track_id'] not in scope:
                        continue
                    user[r['tag']] = user.get(r['tag'], 0) + 1
                for r in db.conn.execute(
                        "SELECT id, auto_tags FROM tracks WHERE error"
                        " IS NULL AND missing = 0"):
                    if scope is not None and r['id'] not in scope:
                        continue
                    for t in _json.loads(r['auto_tags'] or '[]'):
                        auto[t] = auto.get(t, 0) + 1
                db.close()
                vocab = [(t, n, True) for t, n in
                         sorted(user.items(), key=lambda kv: -kv[1])]
                vocab += [(t, n, False) for t, n in
                          sorted(auto.items(), key=lambda kv: -kv[1])
                          if t not in user][:64 - len(vocab)]
            except Exception as e:
                print(f"[DJ] idle vocab skipped: {e}")
            self._dj_idle_vocab = (_t.time(), vocab)
        theme = get_theme(self.dj_cfg.get('theme', 'groove'))
        wps = self._dj_pending_arc

        def base(p):
            if wps:
                xs = [w[0] for w in wps]
                ys = [w[1] for w in wps]
                if p <= xs[0]:
                    return ys[0]
                if p >= xs[-1]:
                    return ys[-1]
                for i in range(len(xs) - 1):
                    if xs[i] <= p <= xs[i + 1]:
                        f = (p - xs[i]) / max(xs[i + 1] - xs[i], 1e-6)
                        return ys[i] + f * (ys[i + 1] - ys[i])
            return theme.arc_target(p)
        return {
            "energy_nudge": self._dj_pending_nudge,
            "arc_heat": max(0.0, min(1.0, base(0.0)
                                     + self._dj_pending_nudge)),
            "themes": sorted(BUILTIN_THEMES),
            "tags": vocab,
            "flavor": dict(self._dj_pending_flavor),
            "arc_waypoints": list(wps),
            "arc_cycle_s": (float(self.dj_cfg.get('night_hours', 6.0))
                            * 3600.0 if theme.arc == 'all_night'
                            else 90 * 60.0),
            "arc_curve": [round(max(0.0, min(1.0, base(i / 24.0))), 3)
                          for i in range(25)],
        }

    def _dj_list_setlists(self):
        """Setlist names available in the library DB, without a running DJ."""
        try:
            from lib.dj import resolve_music_dir
            from lib.dj.db import LibraryDB
            from lib.dj.setlist import list_setlists
            import os
            root = resolve_music_dir(self.dj_cfg.get("music_dir", ""))
            if not os.path.isfile(os.path.join(root, "dj_library.sqlite3")):
                return []
            db = LibraryDB(root)
            names = [s["name"] for s in list_setlists(db)]
            db.close()
            return names
        except Exception:
            return []

    def _dj_music_dir_display(self):
        from lib.dj import resolve_music_dir
        return resolve_music_dir(self.dj_cfg.get("music_dir", ""))

    def _dj_start(self):
        if self._dj is not None and self._dj.active:
            return
        from lib.dj import resolve_music_dir
        from lib.dj.system import DJSystem
        engine = self.scheduler.state.get("soundengine")
        # FULL soundtrack takeover BEFORE the DJ mounts its submix (queue
        # order matters: this stop_all must precede the attach or it would
        # fade the DJ's own track). stop_ambient alone left long ONESHOTS
        # (random-event clips, narrative beds) playing under the mix, and
        # its generation bump kills any oneshot still decoding. Muting
        # keeps weather events (which continue for the visuals) from
        # layering new sounds over the set.
        if engine is not None:
            try:
                engine.stop_all(duration=1.5)
                engine.oneshots_muted = True
            except Exception:
                pass
        self._dj = DJSystem(
            resolve_music_dir(self.dj_cfg.get("music_dir", "")),
            engine=engine,
            theme=self.dj_cfg.get("theme", "groove"),
            night_hours=float(self.dj_cfg.get("night_hours", 6.0)),
            stretch_max=float(self.dj_cfg.get("stretch_max", 1.08)),
            record=bool(self.dj_cfg.get("record", False)),
            persona=str(self.dj_cfg.get("persona", "auto")))
        # Armed idle steering queues BEFORE start(): start() spawns the
        # step thread, and its very first step picks the OPENING track -
        # queuing after it raced that pick (user-heard: a pool night
        # opened with an off-pool song, then corrected).
        if self._dj_pending_setlist:
            name, mode = (self._dj_pending_setlist
                          if isinstance(self._dj_pending_setlist, tuple)
                          else (self._dj_pending_setlist, 'order'))
            self._dj.load_setlist(name, mode=mode)
        if self._dj_pending_flavor:
            self._dj.set_flavor(self._dj_pending_flavor)
        if self._dj_pending_arc:
            self._dj.set_arc_waypoints(self._dj_pending_arc)
        if self._dj_pending_nudge:
            self._dj.set_energy_nudge(self._dj_pending_nudge)
        if not self._dj.start():
            self._dj_last_error = self._dj.last_error or "DJ failed to start"
            self._dj = None
            print(f"[DJ] start failed: {self._dj_last_error}")
            # Give the soundtrack back - we already stopped/muted it above.
            if engine is not None:
                engine.oneshots_muted = False
                self._restore_state_ambient(engine)
            return
        self._dj_last_error = ""
        # The DJ takes the soundtrack: silence state ambient, point the
        # analyzer at the engine's own output.
        try:
            engine.stop_ambient()
        except Exception:
            pass
        if self.analyzer is not None:
            self._dj_prev_source = getattr(self.analyzer, "_active_source",
                                           None)
            self.set_audio_source("internal")
        print("[DJ] live - ambient handed off, analyzer on internal mix")

    def _dj_stop(self):
        if self._dj is None:
            return
        self._dj.stop()
        self._dj = None
        if self.analyzer is not None and self._dj_prev_source:
            self.set_audio_source(self._dj_prev_source)
            self._dj_prev_source = None
        self.scheduler.state['dj_active'] = False
        # Hand the soundtrack back: unmute event sounds and re-trigger the
        # current state's ambient (remembered by _trigger-time bookkeeping
        # even while the DJ played).
        engine = self.scheduler.state.get("soundengine")
        if engine is not None:
            engine.oneshots_muted = False
            self._restore_state_ambient(engine)
        print("[DJ] stopped - state ambient restored")

    def _restore_state_ambient(self, engine):
        """Re-trigger the current weather state's ambient bed (used when
        the DJ hands the soundtrack back, or fails to take it)."""
        try:
            ambient = self.active_effects.get("ambient_sound")
            if ambient:
                media_root = Path(self.scheduler.state.get("media_root",
                                                           "media"))
                params = self.weather_state.get_weather_params(
                    self.weather_state.current_weather)
                engine.play_ambient(media_root / "sounds" / ambient,
                                    skip_seconds=0.0,
                                    ari=params.get("ARI", 0.0))
        except Exception as e:
            print(f"[DJ] ambient restore failed: {e}")

    def _cycle_audio_source(self):
        if self.analyzer is None:
            return
        # Bluetooth is auto-routed when a phone connects + explicitly
        # selectable in the web UI, but excluded from the MIDI cycle so the
        # DJ wheel never lands on a silent BT source with nothing connected.
        order = [s for s in self._AUDIO_SOURCES if s != "bluetooth"]
        cur = getattr(self.analyzer, "_active_source", None)
        nxt = order[(order.index(cur) + 1) % len(order)] if cur in order else order[0]
        self.set_audio_source(nxt)

    def _perform_weather_transition(self):
        """Execute one weather transition right now.

        If a weather-set change is queued, commit it and start a random
        state from the new set; otherwise pick the next state within the
        current set (which, for a linear set whose presets list a single
        ``possible_transitions`` target, is deterministic). Shared by the
        probabilistic ``Switch_rate`` path and the deterministic
        ``state_duration`` timer so both behave identically.
        """
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

    def random_state_change(self):
        if self.enable_web_control and self.web_controller.get('weather_state_locked', False):
            return

        # A director event may HOLD transitions for a moment (e.g. the club
        # director freezes scene changes while a build-up is running so the
        # drop lands in the room that earned it). Wall-clock deadline in
        # scheduler.state; expired or absent = no hold.
        if self.scheduler.state.get('_transition_hold_until', 0.0) > time.time():
            return

        # Operator "Weather Time" slider (web control panel): scales every
        # state's dwell time, 0.5x (fast cycle) .. 2x (slow). Applies to
        # both the deterministic state_duration timer and the probabilistic
        # Switch_rate roll below.
        time_scale = 1.0
        if self.enable_web_control:
            try:
                time_scale = float(
                    self.web_controller.get('weather_time_scale', 1.0) or 1.0)
            except (TypeError, ValueError):
                time_scale = 1.0
        time_scale = max(0.5, min(2.0, time_scale))

        # Deterministic per-state duration (Weight of Light "Elements" set):
        # when the active state declares state_duration > 0, ignore the
        # probabilistic Switch_rate roll entirely and advance on a fixed
        # timer measured from when the transition INTO this state completed
        # (progress >= 1.0). Gives the even per-stage pacing a linear
        # designed sequence wants, instead of the geometric-variance roll.
        state_duration = float(
            self.weather_state.weather_params.get("state_duration", 0.0) or 0.0
        )
        if state_duration > 0.0:
            if self.weather_state.progress >= 1.0:
                now = time.time()
                if self._state_hold_start is None:
                    self._state_hold_start = now
                elif now - self._state_hold_start >= state_duration * time_scale:
                    self._perform_weather_transition()
            return

        transition_speed_mult = self.weather_set.get_transition_speed()

        randcheck = np.random.random()
        if (randcheck < (1 / 800) * self.weather_state.weather_params["Switch_rate"] * transition_speed_mult / time_scale) and (self.weather_state.progress >= 1.0):
            self._perform_weather_transition()

    def shutdown(self):
        """Stop all audio and background threads cleanly."""
        engine = self.scheduler.state.get("soundengine")
        if engine:
            engine.stop_ambient()
        if self.analyzer:
            self.analyzer.stop()
        if getattr(self, "bt_receiver", None) is not None:
            self.bt_receiver.shutdown()
        if self.osc_listener is not None:
            self.osc_listener.stop()


# Main execution
if __name__ == "__main__":
    _args = _parse_cli_args()
    if _args.parent_pid is not None:
        _start_parent_watcher(_args.parent_pid)
    env_system = EnvironmentalSystem(
        project_override=_args.project,
        emulator_port=_args.emulator_port,
    )

    # Optional startup weather pick (project.yaml fields):
    #   startup_weather_set:   name of a weather set in this project's
    #                          WEATHER_SETS (defaults: just stay on the
    #                          project's DEFAULT_WEATHER_SET, no transition).
    #   startup_weather_state: name of a state in that set; the controller
    #                          transitions to it on launch. Falls back to a
    #                          random pick from the set's states if absent.
    _startup_set = env_system.project.raw.get("startup_weather_set")
    _startup_state = env_system.project.raw.get("startup_weather_state")
    if _startup_set:
        try:
            _initial_weather = (
                env_system._weather_state_enum(_startup_state)
                if _startup_state else None
            )
            env_system.change_weather_set(
                _startup_set,
                immediate=True,
                initial_weather=_initial_weather,
            )
        except Exception as _e:
            print(f"[Project] startup weather pick failed ({_e}); "
                  f"using project default")
    # Frame pacing: read the per-project target FPS from env_system on
    # every iteration so a project swap re-paces the loop without
    # restart. ``env_system.frame_time`` is set by ``_compute_frame_time``
    # in __init__ and refreshed in _swap_project_unsafe.
    frame_count = 0
    fps_start_time = time.perf_counter()
    work_time_accum = 0.0  # sum of per-frame work time (no waits) over the window

    # For better sleep precision on Windows
    import sys
    if sys.platform == 'win32':
        import ctypes
        winmm = ctypes.WinDLL('winmm')
        winmm.timeBeginPeriod(1)  # Set 1ms timer resolution

    next_deadline = time.perf_counter() + env_system.frame_time

    try:
        while True:
            frame_start = time.perf_counter()
            frame_time = env_system.frame_time   # may change on project swap

            # Update environmental system (includes scheduler.update())
            env_system.update()

            if env_system.scheduler.should_exit:
                break

            # Measure pure work time (render + send) before any frame-rate waits
            work_time_accum += time.perf_counter() - frame_start

            # Deadline-based frame pacing: fast frames compensate for slow ones
            remaining = next_deadline - time.perf_counter()
            if remaining > 0.002:
                time.sleep(remaining - 0.0015)
            while time.perf_counter() < next_deadline:
                pass

            next_deadline += frame_time
            # If we're more than 2 frames behind, drop the debt rather than burst-render
            if time.perf_counter() - next_deadline > 2 * frame_time:
                next_deadline = time.perf_counter() + frame_time

            frame_count += 1
            if frame_count % 500 == 0:  # Print FPS every 500 frames
                current_time = time.perf_counter()
                window = current_time - fps_start_time
                actual_fps = 500.0 / window
                target_fps = 1.0 / frame_time
                avg_work = work_time_accum / 500.0
                uncapped_fps = (1.0 / avg_work) if avg_work > 0 else 0.0
                env_system._current_fps = round(actual_fps, 1)
                env_system._target_fps = round(target_fps, 1)
                env_system._uncapped_fps = round(uncapped_fps, 1)
                print(f"[Main] FPS actual={actual_fps:.1f} target={target_fps:.1f} uncapped={uncapped_fps:.1f}")
                fps_start_time = current_time
                work_time_accum = 0.0

    except KeyboardInterrupt:
        pass
    finally:
        env_system.shutdown()
        print("Done!")