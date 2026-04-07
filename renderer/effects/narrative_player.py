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
    """Return playback length of an audio file in seconds, 0.0 on failure."""
    try:
        import miniaudio
        info = miniaudio.mp3_get_file_info(str(path))
        return info.num_frames / info.sample_rate
    except Exception:
        # Fallback: decode a small portion to measure sample rate
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

        p = Path(script_path.replace('\\', '/'))
        if p.exists():
            data              = json.loads(p.read_text(encoding='utf-8'))
            self._nodes       = data.get('nodes', {})
            self._start_nodes = data.get('start_nodes', [])
            self._var_defs    = data.get('variables', [])
            self._audio_dir   = p.parent

            # Initialise all variables to 0
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
        else:
            print(f'[NarrativePlayer] Script not found: {script_path}')
            self.enabled = False

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
        """Write current variable values into the shared state dict."""
        for v in self._var_defs:
            name = v['name']
            state[f'story_{name}'] = self._var_current[name]

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

        # Use the node's explicit "file" field if set, otherwise fall back
        # to the convention of {node_id}.mp3 in the script directory.
        # The "file" field is stored relative to the project root.
        file_rel = nd.get('file', '')
        if file_rel:
            # Normalize path separators for cross-platform compatibility
            file_rel = file_rel.replace('\\', '/')
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
        self._volume = state.get('narrative_volume', 1.0)

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

        if self._phase == self.IDLE:
            starts = self._start_nodes or list(self._nodes.keys())
            if starts:
                pick = _weighted_choice(starts, [1.0] * len(starts),
                                        self._recency)
                self._play_node(pick or starts[0], engine)
            else:
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

    count == 0   : first call — instantiate and register the effect.
    count  > 0   : subsequent frames — effect drives itself via update().
    count == -1  : cleanup call — remove effect from viewport.
    """
    renderer = outstate.get('shader_renderer')
    if not renderer:
        return

    if state['count'] == 0:
        viewport = renderer.get_viewport(frame_id)
        viewport.add_effect(NarrativePlayer,
                            script_path=script_path,
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
