"""
NarrativePlayer — Frame-driven narrative audio playback.

Walks a narrative script JSON, plays each node's audio file via the main
AudioEngine, waits for playback to complete, applies a configurable
inter-node delay, then advances to the next node via weighted random selection.

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


def _weighted_choice(nexts: list, weights: list) -> Optional[str]:
    """Pick a node ID from nexts using the corresponding weights."""
    if not nexts:
        return None
    total = sum(weights) or 1.0
    r = random.random() * total
    acc = 0.0
    for nid, w in zip(nexts, weights):
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
    """

    IDLE    = 'idle'
    PLAY    = 'play'
    DELAY   = 'delay'
    RESTART = 'restart'

    def __init__(self, viewport, script_path: str = '', delay: float = 3.0,
                 restart_delay: float = 10.0):
        super().__init__(viewport)
        self.delay         = delay          # seconds between nodes
        self.restart_delay = restart_delay  # seconds to wait before looping

        self._phase:         str            = self.IDLE
        self._phase_elapsed: float          = 0.0
        self._audio_dur:     float          = 0.0
        self._current_node:  Optional[str]  = None

        self._nodes:       dict = {}
        self._start_nodes: list = []
        self._audio_dir:   Path = Path('.')

        p = Path(script_path)
        if p.exists():
            data              = json.loads(p.read_text(encoding='utf-8'))
            self._nodes       = data.get('nodes', {})
            self._start_nodes = data.get('start_nodes', [])
            self._audio_dir   = p.parent
            print(f'[NarrativePlayer] Loaded {p.name}  '
                  f'({len(self._nodes)} nodes, '
                  f'{len(self._start_nodes)} start nodes)')
        else:
            print(f'[NarrativePlayer] Script not found: {script_path}')
            self.enabled = False

    def init(self):
        """No shaders or GPU buffers needed."""
        print('    [OK] NarrativePlayer initialised')

    # ── Internal state-machine helpers ──────────────────────────────────────

    def _play_node(self, node_id: str, engine) -> None:
        """Start playback of node_id's audio file and enter PLAY phase."""
        nd = self._nodes.get(node_id)
        if not nd:
            self._phase = self.DONE
            return

        self._current_node  = node_id
        self._phase_elapsed = 0.0

        audio_file = self._audio_dir / f'{node_id}.mp3'
        if audio_file.exists():
            self._audio_dur = _audio_duration(audio_file)
            if engine:
                # duration cap = measured length + 0.5 s safety margin
                engine.schedule_event(
                    str(audio_file), volume=1.0,
                    duration=self._audio_dur + 0.5,
                )
            print(f'[NarrativePlayer] ▶ {node_id}  ({self._audio_dur:.1f}s)')
        else:
            print(f'[NarrativePlayer] ▶ {node_id}  (no audio — skipping)')
            self._audio_dur = 0.0

        self._phase = self.PLAY

    def _advance(self, engine) -> None:
        """Pick the next node via weighted random and play it, or finish."""
        nd      = self._nodes.get(self._current_node, {})
        nexts   = nd.get('next', [])
        weights = nd.get('weights', [1.0] * len(nexts))
        nxt = _weighted_choice(nexts, weights)
        if nxt:
            self._play_node(nxt, engine)
        else:
            print(f'[NarrativePlayer] Script complete — restarting in {self.restart_delay:.0f}s.')
            self._phase         = self.RESTART
            self._phase_elapsed = 0.0

    # ── Per-frame interface ──────────────────────────────────────────────────

    def update(self, dt: float, state: Dict) -> None:
        if not self.enabled:
            return

        self._phase_elapsed += dt
        engine = state.get('soundengine')

        if self._phase == self.IDLE:
            starts = self._start_nodes or list(self._nodes.keys())
            if starts:
                self._play_node(random.choice(starts), engine)
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
                            restart_delay=restart_delay)
        state['effect'] = viewport.effects[-1]
        print('[shader_narrative_player] Started')

    elif state['count'] == -1:
        effect   = state.get('effect')
        viewport = renderer.get_viewport(frame_id)
        if effect and effect in viewport.effects:
            viewport.effects.remove(effect)
            effect.cleanup()
        print('[shader_narrative_player] Stopped')
