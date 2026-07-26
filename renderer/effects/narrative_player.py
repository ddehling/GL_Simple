"""
NarrativePlayer — Frame-driven narrative audio playback.

Walks a narrative script JSON, plays each node's audio file via the main
AudioEngine, waits for playback to complete, applies a configurable
inter-node delay, then advances to the next node via weighted random selection.

Story variables defined in the script are interpolated linearly over a
configurable ramp duration whenever a new node starts, and ramped to zero
when the story reaches a terminal node.  Current values are injected into
the shared state dict every frame so other effects can read them.

No OpenGL resources are used.  All logic lives in update(); render() is a no-op.

Usage — add to event_map in Stories_OGL.py:

    "play_my_story": (narrative_play_event, {
        "script_path": "media/sounds/my_story/script.json",
        "delay": 3.0,
    }),

Then schedule it from the __main__ block or a weather-state event like any
other effect.
"""

import json
import random
from pathlib import Path
from typing import Dict, Optional

from renderer.effects.base import ShaderEffect


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _audio_duration(path: Path) -> float:
    """Return playback length of an audio file in seconds, 0.0 on failure.

    NOTE: We deliberately do NOT use miniaudio.mp3_get_file_info() — it is
    unreliable for VBR MP3s without a Xing/VBRI header (estimates from
    bitrate × filesize and routinely under-reports), which previously caused
    long narrative tracks to cut out partway. Always decode the full file
    so the frame count is exact.
    """
    try:
        import miniaudio
        decoded = miniaudio.decode_file(
            str(path), nchannels=1, sample_rate=44100,
            output_format=miniaudio.SampleFormat.SIGNED16,
        )
        return decoded.num_frames / 44100
    except Exception:
        return 0.0


def _weighted_choice(nexts: list, weights: list,
                     recency: dict = None) -> Optional[str]:
    """Pick a node ID from nexts using weights, penalised by recency counters.

    Each node's effective weight is  w * 4^(-counter).  A counter of 1 divides
    the probability by 4; 2 divides it by 16, etc.  Counters decay toward 0 over time
    so the penalty fades after roughly 10 hours.
    """
    if not nexts:
        return None
    effective = []
    for nid, w in zip(nexts, weights):
        if recency and nid in recency:
            w *= 10.0 ** (-recency[nid])
        effective.append(w)
    total = sum(effective) or 1.0
    r = random.random() * total
    acc = 0.0
    for nid, w in zip(nexts, effective):
        acc += w
        if r <= acc:
            return nid
    return nexts[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Effect class
# ─────────────────────────────────────────────────────────────────────────────

class NarrativePlayer(ShaderEffect):
    """
    Frame-driven narrative script player.  No OpenGL resources.

    Phase machine per node:
        idle    → pick a random start node → play
        play    → phase_elapsed >= audio_duration → delay
        delay   → phase_elapsed >= delay → pick next node → play  (or restart)
        restart → phase_elapsed >= restart_delay → idle

    Story variables are linearly interpolated from their current values to the
    new node's target values over ``var_ramp_duration`` seconds.  When a
    terminal node finishes (no next[]), all variables ramp to 0.
    """

    IDLE    = 'idle'
    PLAY    = 'play'
    DELAY   = 'delay'
    RESTART = 'restart'
    # Terminal phase: the script has nothing playable (an empty graph, or a
    # next[] pointing at a node that no longer exists). Both `_play_node`
    # and `update` already routed here, but the constant was never defined —
    # so instead of going quiet the player raised AttributeError every frame
    # and took the whole engine down with it. Reaching DONE is a dead end by
    # design: the player parks, ramps its variables to zero so the narrative
    # shaders fade out cleanly, and waits for a reload.
    DONE    = 'done'

    def __init__(self, viewport, script_path: str = '', delay: float = 3.0,
                 restart_delay: float = 10.0, var_ramp_duration: float = 10.0):
        super().__init__(viewport)
        self.delay             = delay            # seconds between nodes
        self.restart_delay     = restart_delay    # seconds before looping
        self.var_ramp_duration = var_ramp_duration  # seconds for variable transitions

        self._phase:         str            = self.IDLE
        self._phase_elapsed: float          = 0.0
        self._audio_dur:     float          = 0.0
        self._current_node:  Optional[str]  = None

        self._nodes:       dict = {}
        self._start_nodes: list = []
        self._audio_dir:   Path = Path('.')
        self._script_path: str  = ''    # currently-loaded script (empty = none)

        # ── Story variable interpolation state ───────────────────────────
        self._var_defs:    list = []   # [{"name": ..., "description": ...}, ...]
        self._var_current: dict = {}   # name -> current interpolated value
        self._var_start:   dict = {}   # name -> value at ramp start
        self._var_target:  dict = {}   # name -> ramp target value
        self._var_elapsed: float = 0.0 # time since ramp started
        self._var_ramping: bool  = False

        # ── Recency suppression (avoids repeating the same nodes) ─────
        # Maps node_id -> float counter.  Incremented by 1 each time a node
        # plays; decays by 1/hour continuously.  Effective weight is
        # original_weight * 2^(-counter).
        self._recency: dict = {}   # node_id -> float
        self._DECAY_PER_SEC: float = 1.0 / 36000.0  # lose 1 count per 10 hours

        # ── Pending weather-transition request ─────────────────────────
        # When a node with a non-null ``trigger_state`` field plays, that
        # state name is stashed here along with the source node id (so
        # logs can report which node fired the transition). ``update()``
        # publishes both to the frame state dict as
        # ``_weather_transition_request`` and ``_weather_transition_node``
        # so the owning environmental system can pick them up and call
        # ``transition_to_weather()``. Old scripts without the field are
        # unaffected — ``nd.get('trigger_state')`` simply returns None.
        self._pending_weather_request: Optional[str] = None
        self._pending_weather_node:    Optional[str] = None

        # Disabled by default; _load_script flips enabled on success.
        self.enabled = False
        if script_path:
            self._load_script(script_path)

    def _load_script(self, script_path: str) -> bool:
        """Load a narrative script JSON. Resets state-machine and variables.

        Returns True on success, False otherwise. A False return leaves
        the player disabled so it stays silent until a valid path arrives.
        """
        p = Path(script_path.replace('\\', '/'))
        if not p.exists():
            print(f'[NarrativePlayer] Script not found: {script_path}')
            self._script_path = ''
            self.enabled = False
            return False

        try:
            data = json.loads(p.read_text(encoding='utf-8'))
        except Exception as e:
            print(f'[NarrativePlayer] Failed to parse {script_path}: {e}')
            self._script_path = ''
            self.enabled = False
            return False

        self._nodes       = data.get('nodes', {})
        self._start_nodes = data.get('start_nodes', [])
        self._var_defs    = data.get('variables', [])
        self._audio_dir   = p.parent
        self._script_path = script_path

        # Reset state machine and variables so the new script starts fresh.
        self._phase         = self.IDLE
        self._phase_elapsed = 0.0
        self._audio_dur     = 0.0
        self._current_node  = None
        self._var_current.clear()
        self._var_start.clear()
        self._var_target.clear()
        self._var_elapsed = 0.0
        self._var_ramping = False
        self._recency.clear()
        for v in self._var_defs:
            name = v['name']
            self._var_current[name] = 0.0
            self._var_start[name]   = 0.0
            self._var_target[name]  = 0.0

        var_names = [v['name'] for v in self._var_defs]
        print(f'[NarrativePlayer] Loaded {p.name}  '
              f'({len(self._nodes)} nodes, '
              f'{len(self._start_nodes)} start nodes'
              f'{", vars: " + ", ".join(var_names) if var_names else ""})')

        self.enabled = True
        return True

    def unload(self) -> None:
        """Disable the player without destroying the instance.

        Used when the active weather set declares no narrative_script,
        so the effect stays silent but the same object can be reloaded
        later without reconstructing OpenGL resources (there are none,
        but the pattern matches other effects).
        """
        self._script_path = ''
        self._phase       = self.IDLE
        self._current_node = None
        self._nodes.clear()
        self._start_nodes = []
        self._var_defs    = []
        self.enabled = False

    @property
    def script_path(self) -> str:
        return self._script_path

    def init(self):
        """No shaders or GPU buffers needed."""
        print('    [OK] NarrativePlayer initialised')

    # ── Variable interpolation ───────────────────────────────────────────

    def _start_var_ramp(self, target_vars: dict) -> None:
        """Begin a linear ramp from current values to *target_vars*."""
        if not self._var_defs:
            return
        for v in self._var_defs:
            name = v['name']
            self._var_start[name]  = self._var_current[name]
            self._var_target[name] = float(target_vars.get(name, 0.0))
        self._var_elapsed = 0.0
        self._var_ramping = True

    def _ramp_to_zero(self) -> None:
        """Begin a linear ramp of all variables to 0."""
        self._start_var_ramp({})

    def _update_var_ramp(self, dt: float) -> None:
        """Advance the variable ramp by *dt* seconds."""
        if not self._var_ramping or not self._var_defs:
            return
        self._var_elapsed += dt
        progress = min(1.0, self._var_elapsed / self.var_ramp_duration) \
                   if self.var_ramp_duration > 0 else 1.0
        for v in self._var_defs:
            name = v['name']
            start  = self._var_start[name]
            target = self._var_target[name]
            self._var_current[name] = start + (target - start) * progress
        if progress >= 1.0:
            self._var_ramping = False

    def _inject_vars(self, state: Dict) -> None:
        """Write current variable values into the shared state dict.

        Also publishes a `narrative_vars` metadata list (name, narrative-
        driven value, description) so the web control panel can render
        sliders for whatever variables the active script declares. The
        listed value is always the narrative-driven current value; web
        overrides are applied later by Stories_OGL and stomp state[...]
        independently.
        """
        meta = []
        for v in self._var_defs:
            name = v['name']
            cur  = self._var_current[name]
            state[f'story_{name}'] = cur
            meta.append({
                'name': name,
                'value': cur,
                'description': v.get('description', ''),
            })
        state['narrative_vars'] = meta

    # ── Internal state-machine helpers ──────────────────────────────────────

    def _play_node(self, node_id: str, engine) -> None:
        """Start playback of node_id's audio file and enter PLAY phase."""
        nd = self._nodes.get(node_id)
        if not nd:
            self._phase = self.DONE
            return

        self._current_node  = node_id
        self._phase_elapsed = 0.0

        # Record play for recency suppression
        prev_count = self._recency.get(node_id, 0.0)
        self._recency[node_id] = prev_count + 1.0

        # Start variable ramp toward this node's values
        self._start_var_ramp(nd.get('vars', {}))

        # Stash any node-triggered weather-state transition. Most nodes
        # won't have this field (backwards-compatible with pre-feature
        # scripts). ``update()`` publishes it to the shared state dict
        # this frame so the env-system picks it up. Source node id is
        # stashed alongside for logging.
        trig = nd.get('trigger_state')
        if trig:
            self._pending_weather_request = str(trig)
            self._pending_weather_node    = node_id

        # Use the node's explicit "file" field if set, otherwise fall back
        # to the convention of {node_id}.mp3 in the script directory.
        #
        # Resolution order (in priority):
        #   1. ``basename(file)`` next to the script — works for the
        #      common case where audio is colocated with script.json
        #      (true after the Phase-9 media move into projects/<id>/media/).
        #   2. The literal ``file:`` value resolved relative to the repo
        #      root — legacy behavior for pre-Phase-9 scripts that stored
        #      ``media/sounds/foo/bar.mp3`` paths.
        #   3. ``{node_id}.mp3`` next to the script (fallback when file
        #      field is missing).
        file_rel = nd.get('file', '')
        if file_rel:
            file_rel = file_rel.replace('\\', '/')
            colocated = self._audio_dir / Path(file_rel).name
            if colocated.exists():
                audio_file = colocated
            else:
                audio_file = Path(file_rel)
                if not audio_file.is_absolute():
                    repo_root = Path(__file__).parent.parent.parent
                    audio_file = repo_root / audio_file
        else:
            audio_file = self._audio_dir / f'{node_id}.mp3'
        if audio_file.exists():
            self._audio_dur = _audio_duration(audio_file)
            if engine:
                # duration cap = measured length + 0.5 s safety margin
                engine.schedule_event(
                    str(audio_file), volume=1.0,
                    duration=self._audio_dur + 0.5,
                    narrative=True,
                )
            print(f'[NarrativePlayer] ▶ {node_id}  ({self._audio_dur:.1f}s)  recency={prev_count:.2f}')
        else:
            print(f'[NarrativePlayer] ▶ {node_id}  (no audio — skipping)  recency={prev_count:.2f}')
            self._audio_dur = 0.0

        self._phase = self.PLAY

    def _advance(self, engine) -> None:
        """Pick the next node via weighted random and play it, or finish."""
        nd      = self._nodes.get(self._current_node, {})
        nexts   = nd.get('next', [])
        weights = nd.get('weights', [1.0] * len(nexts))
        nxt = _weighted_choice(nexts, weights, self._recency)
        if nxt:
            self._play_node(nxt, engine)
        else:
            # Terminal node — ramp all variables to zero
            self._ramp_to_zero()
            print(f'[NarrativePlayer] Script complete — restarting in {self.restart_delay:.0f}s.')
            self._phase         = self.RESTART
            self._phase_elapsed = 0.0

    # ── Per-frame interface ──────────────────────────────────────────────────

    def update(self, dt: float, state: Dict) -> None:
        if not self.enabled:
            return

        self._phase_elapsed += dt
        engine = state.get('soundengine')

        # Decay recency counters (1 count per hour)
        if self._recency:
            decay = self._DECAY_PER_SEC * dt
            to_remove = []
            for nid in self._recency:
                self._recency[nid] -= decay
                if self._recency[nid] <= 0.0:
                    to_remove.append(nid)
            for nid in to_remove:
                del self._recency[nid]

        if self._phase == self.DONE:
            # Nothing playable. Stay silent and let the variables finish
            # ramping to zero; do not spin on the state machine.
            self._update_var_ramp(dt)
            self._inject_vars(state)
            return

        if self._phase == self.IDLE:
            starts = self._start_nodes or list(self._nodes.keys())
            if starts:
                pick = _weighted_choice(starts, [1.0] * len(starts),
                                        self._recency)
                self._play_node(pick or starts[0], engine)
            else:
                print('[NarrativePlayer] Script has no playable nodes - '
                      'parking. (Check that the script still has a `nodes` '
                      'block; the v2 editor writes one file for both.)')
                self._ramp_to_zero()
                self._phase = self.DONE

        elif self._phase == self.PLAY:
            if self._phase_elapsed >= self._audio_dur:
                # Audio has finished; begin inter-node delay
                self._phase         = self.DELAY
                self._phase_elapsed = 0.0

        elif self._phase == self.DELAY:
            if self._phase_elapsed >= self.delay:
                self._advance(engine)

        elif self._phase == self.RESTART:
            if self._phase_elapsed >= self.restart_delay:
                print('[NarrativePlayer] Restarting.')
                self._phase         = self.IDLE
                self._phase_elapsed = 0.0

        # Publish a pending weather-transition request, if any. The
        # request was set in ``_play_node`` when a node with a
        # ``trigger_state`` field began playing. The env-system consumes
        # (pops) both keys in its main update loop. The node id is
        # included so log messages can report which node fired it.
        if self._pending_weather_request is not None:
            state['_weather_transition_request'] = self._pending_weather_request
            state['_weather_transition_node']    = self._pending_weather_node
            self._pending_weather_request = None
            self._pending_weather_node    = None

        # Interpolate variables and inject into shared state every frame
        self._update_var_ramp(dt)
        self._inject_vars(state)


    def render(self, state: Dict) -> None:
        pass  # no OpenGL output

    def cleanup(self) -> None:
        pass  # no OpenGL resources to free


# ─────────────────────────────────────────────────────────────────────────────
# Event-map wrapper
# ─────────────────────────────────────────────────────────────────────────────

def shader_narrative_player(state: dict, outstate: dict,
                             script_path: str = '',
                             node_delay: float = 3.0,
                             restart_delay: float = 10.0,
                             var_ramp_duration: float = 10.0,
                             frame_id: int = 0) -> None:
    """
    Event-map wrapper for NarrativePlayer.

    The script path is driven by outstate['narrative_script'], published by
    WeatherSetManager from the active set's configuration. Any explicit
    `script_path` kwarg here is only a fallback for cases where the
    outstate key is absent (standalone scheduling, tests, etc.). When the
    active set changes its narrative_script, this wrapper calls reload on
    the effect so the story swaps cleanly without restarting the effect
    instance.

    count == 0   : first call — instantiate the effect (disabled) and
                   load whatever script is currently declared.
    count  > 0   : subsequent frames — sync effect.script_path with
                   outstate['narrative_script'].
    count == -1  : cleanup call — remove effect from viewport.
    """
    renderer = outstate.get('shader_renderer')
    if not renderer:
        return

    # outstate is the single source of truth for which script plays.
    # Fall back to the kwarg only when the key is absent entirely.
    if 'narrative_script' in outstate:
        desired = outstate['narrative_script'] or ''
    else:
        desired = script_path or ''

    # Resolve relative paths against the active project's media root so
    # weather sets can use ``narrative_script: 'sounds/foo/script.json'``
    # without hardcoding the project subdir. Absolute paths pass through.
    if desired:
        from pathlib import Path as _Path
        p = _Path(desired)
        if not p.is_absolute() and 'media_root' in outstate:
            desired = str(_Path(outstate['media_root']) / p)

    if state['count'] == 0:
        viewport = renderer.get_viewport(frame_id)
        # Construct disabled; _sync_script below loads the correct script.
        viewport.add_effect(NarrativePlayer,
                            script_path='',
                            delay=node_delay,
                            restart_delay=restart_delay,
                            var_ramp_duration=var_ramp_duration)
        state['effect'] = viewport.effects[-1]
        print('[shader_narrative_player] Started')

    elif state['count'] == -1:
        effect   = state.get('effect')
        viewport = renderer.get_viewport(frame_id)
        if effect and effect in viewport.effects:
            viewport.effects.remove(effect)
            effect.cleanup()
        print('[shader_narrative_player] Stopped')
        return

    # Every frame (including the first): if the desired script differs
    # from what the effect has loaded, swap.
    effect = state.get('effect')
    if effect is None:
        return
    current = effect.script_path
    if desired != current:
        if desired:
            effect._load_script(desired)
        else:
            effect.unload()
