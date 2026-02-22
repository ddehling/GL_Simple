"""Ambient audio cross-fade controller.

Manages the single looping ambient track that plays during a weather state.
Cross-fades by starting the incoming track (with fade-in) immediately while
the outgoing track fades out — both coexist as concurrent events inside the
AudioEngine's sounddevice mixer, giving a true simultaneous overlap.

Audio loading runs on a background thread so the render loop is never blocked
by librosa/soundfile decoding (especially important for large MP3 files).
"""

import threading
import time
from pathlib import Path
from typing import Optional


class AmbientAudioController:
    """Cross-fades between ambient tracks using the AudioEngine event mixer.

    Fades out the outgoing track immediately, then loads and starts the
    incoming track on a background thread. Both coexist as concurrent
    AudioEvent objects in the sounddevice mixer callback once loaded,
    giving a true simultaneous cross-fade with no render-loop stalls.
    """

    FADE_OUT_DURATION = 5.0   # seconds to fade the outgoing track
    FADE_IN_DURATION  = 5.0   # seconds to fade in the incoming track

    def __init__(self) -> None:
        self._current_name: Optional[str] = None
        self._engine = None

    def transition(self, filepath: Path, skip_time: float, ari: float, engine) -> None:
        """Cross-fade to a new ambient track without blocking the render loop.

        Fades out the current track immediately, then spawns a background
        thread to load (or retrieve from cache) and schedule the new one.
        On first use of a file the load may take a moment; subsequent
        transitions to the same state are served instantly from cache.

        Args:
            filepath:  Path to the audio file.
            skip_time: Seconds into the file to start playback from.
            ari:       Ambient Repeat Interval — seconds of audio to load and
                       loop (keeps RAM bounded for long files like 1-hour wavs).
            engine:    The AudioEngine instance (from state['soundengine']).
        """
        # Fade out old track immediately (fast — just sets flags on the event)
        if self._current_name is not None:
            engine.fade_out_audio(self._current_name, self.FADE_OUT_DURATION)

        name = filepath.name  # filename used as unique event name
        self._current_name = name
        self._engine = engine
        print(f"[AmbientAudio] Cross-fading to: {filepath.name}")

        def _load_and_schedule():
            # load_audio warms the AudioCache (librosa/soundfile decode happens here)
            engine.schedule_event(
                filepath,
                time.time(),
                ari,
                repeat_interval=ari,
                inname=name,
                fade_in_duration=self.FADE_IN_DURATION,
                skip_time=skip_time,
            )

        threading.Thread(target=_load_and_schedule, daemon=True,
                         name=f"ambient-load-{filepath.name}").start()

    def stop(self) -> None:
        """Fade out the current ambient track."""
        if self._current_name is not None and self._engine is not None:
            self._engine.fade_out_audio(self._current_name, self.FADE_OUT_DURATION)
            self._current_name = None
