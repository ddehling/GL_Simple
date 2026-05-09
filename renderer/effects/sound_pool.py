"""
SoundPoolEffect - Random ambient sound playback from a directory pool.

Scans a directory for audio files and plays a random clip at configurable
intervals via the shared AudioEngine. Extracted from the original inline
BART-sound scheduler in `bart_map.py` so any weather set can specify its
own pool directory via `sound_pool_dir` in WEATHER_SETS.

No OpenGL resources are used. All logic lives in update(); render() is a
no-op.

The active pool directory is driven by outstate['sound_pool_dir'], which
WeatherSetManager publishes from the active set's config. Swapping sets
reloads the pool on the next frame, so the effect instance itself doesn't
need to be restarted.
"""
import time as _time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from renderer.effects.base import ShaderEffect


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_AUDIO_EXTS = ('.mp3', '.wav', '.ogg', '.flac', '.m4a')


def _audio_duration_seconds(path: Path) -> float:
    """Best-effort duration lookup for a sound file.

    Mirrors the behavior of the original bart sound scheduler: try
    miniaudio's fast info path first, fall back to a conservative default
    so scheduling never fails outright.
    """
    try:
        import miniaudio
        info = miniaudio.mp3_get_file_info(str(path))
        return info.num_frames / info.sample_rate
    except Exception:
        return 15.0


def _scan_pool(dir_path: str) -> list:
    """Return a list of Path objects for every audio file under *dir_path*.

    Path is interpreted as repo-relative if not absolute.
    """
    if not dir_path:
        return []
    p = Path(dir_path.replace('\\', '/'))
    if not p.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent.parent
        p = repo_root / p
    if not p.exists() or not p.is_dir():
        return []
    files = []
    for ext in _AUDIO_EXTS:
        files.extend(p.glob(f'*{ext}'))
    return sorted(files)


# ─────────────────────────────────────────────────────────────────────────────
# Effect class
# ─────────────────────────────────────────────────────────────────────────────

class SoundPoolEffect(ShaderEffect):
    """Pick a random sound from a directory and play it at intervals.

    Parameters
    ----------
    sound_dir : str
        Repo-relative (or absolute) path to the pool directory.
    probability : float
        Per-frame chance of starting a new clip while cooldown is clear.
        Default 1/400 matches the original bart scheduler's rate.
    min_cooldown : float
        Minimum seconds between clips, on top of the last clip's duration.
    volume : float
        Playback volume handed to AudioEngine.schedule_event.
    """

    def __init__(self, viewport, sound_dir: str = '', probability: float = 1/400,
                 min_cooldown: float = 5.0, volume: float = 0.8):
        super().__init__(viewport)
        self.probability: float = float(probability)
        self.min_cooldown: float = float(min_cooldown)
        self.volume: float = float(volume)

        self._sound_dir: str = ''
        self._files: list = []
        self._next_play_time: float = 0.0

        # Disabled until a valid pool loads.
        self.enabled = False
        if sound_dir:
            self._load_pool(sound_dir)

    # ── Pool lifecycle ───────────────────────────────────────────────────────

    def _load_pool(self, sound_dir: str) -> bool:
        """Scan *sound_dir* for audio files. Returns True if any were found.

        On success the effect is enabled; on failure it stays silent.
        """
        self._files = _scan_pool(sound_dir)
        self._sound_dir = sound_dir
        if not self._files:
            print(f'[SoundPool] No audio files in: {sound_dir}')
            self.enabled = False
            return False

        # Reset the cooldown so a freshly-loaded pool doesn't immediately
        # fire a clip on top of whatever was playing before, but also
        # doesn't wait an unpredictable amount of time from a prior pool.
        self._next_play_time = _time.time() + self.min_cooldown
        self.enabled = True
        print(f'[SoundPool] Loaded {len(self._files)} files from {sound_dir}')
        return True

    def unload(self) -> None:
        """Disable the effect and drop the cached file list."""
        self._sound_dir = ''
        self._files = []
        self.enabled = False

    @property
    def sound_dir(self) -> str:
        return self._sound_dir

    # ── Per-frame interface ─────────────────────────────────────────────────

    def init(self) -> None:
        """No shaders or GPU buffers needed."""
        print('    [OK] SoundPoolEffect initialised')

    def update(self, dt: float, state: Dict) -> None:
        if not self.enabled or not self._files:
            return

        engine = state.get('soundengine')
        if not engine:
            return

        now = _time.time()
        if now < self._next_play_time:
            return

        # Probabilistic trigger; same knob as the old bart scheduler.
        if np.random.random() >= self.probability:
            return

        pick = np.random.choice(self._files)
        dur = _audio_duration_seconds(pick)
        engine.schedule_event(str(pick), volume=self.volume)
        # Block further picks for the clip's duration + cooldown pad so
        # clips don't overlap themselves.
        self._next_play_time = now + dur + self.min_cooldown

    def render(self, state: Dict) -> None:
        pass  # no OpenGL output

    def cleanup(self) -> None:
        pass  # no OpenGL resources to free


# ─────────────────────────────────────────────────────────────────────────────
# Event-map wrapper
# ─────────────────────────────────────────────────────────────────────────────

def shader_sound_pool(state: dict, outstate: dict,
                      sound_dir: str = '',
                      probability: float = 1/400,
                      min_cooldown: float = 5.0,
                      volume: float = 0.8,
                      frame_id: int = 0) -> None:
    """
    Event-map wrapper for SoundPoolEffect.

    The pool directory is driven by outstate['sound_pool_dir'], published
    by WeatherSetManager from the active set's config. The `sound_dir`
    kwarg is only a fallback when the outstate key is absent entirely.

    count == 0   : instantiate the effect (disabled); sync below loads it.
    count  > 0   : subsequent frames - sync effect.sound_dir with
                   outstate['sound_pool_dir'].
    count == -1  : cleanup call - remove effect from viewport.
    """
    renderer = outstate.get('shader_renderer')
    if not renderer:
        return

    if 'sound_pool_dir' in outstate:
        desired = outstate['sound_pool_dir'] or ''
    else:
        desired = sound_dir or ''

    # Resolve relative paths against the active project's media_root so
    # weather sets can use ``sound_pool_dir: 'sounds/foo'`` portably.
    if desired:
        from pathlib import Path as _Path
        p = _Path(desired)
        if not p.is_absolute() and 'media_root' in outstate:
            desired = str(_Path(outstate['media_root']) / p)

    if state['count'] == 0:
        viewport = renderer.get_viewport(frame_id)
        viewport.add_effect(SoundPoolEffect,
                            sound_dir='',
                            probability=probability,
                            min_cooldown=min_cooldown,
                            volume=volume)
        state['effect'] = viewport.effects[-1]
        print('[shader_sound_pool] Started')

    elif state['count'] == -1:
        effect = state.get('effect')
        viewport = renderer.get_viewport(frame_id)
        if effect and effect in viewport.effects:
            viewport.effects.remove(effect)
            effect.cleanup()
        print('[shader_sound_pool] Stopped')
        return

    # Every frame (including the first): sync the pool.
    effect = state.get('effect')
    if effect is None:
        return
    if desired != effect.sound_dir:
        if desired:
            effect._load_pool(desired)
        else:
            effect.unload()
