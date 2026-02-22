"""Ambient audio cross-fade controller.

Manages the single looping ambient track that plays during a weather state.
Cross-fades by starting the incoming track (with fade-in) immediately while
the outgoing track fades out — both coexist as concurrent events inside the
AudioEngine's sounddevice mixer, giving a true simultaneous overlap.
"""

from pathlib import Path
from typing import Optional


class AmbientAudioController:
    """Cross-fades between ambient tracks using the AudioEngine event mixer.

    Calls engine.fade_out_audio() on the outgoing track and
    engine.schedule_event() on the incoming one immediately — no background
    thread needed because the AudioEngine's sounddevice callback handles
    mixing both at the same time.
    """

    FADE_OUT_DURATION = 5.0   # seconds to fade the outgoing track
    FADE_IN_DURATION  = 5.0   # seconds to fade in the incoming track

    def __init__(self) -> None:
        self._current_name: Optional[str] = None
        self._engine = None

    def transition(self, filepath: Path, skip_time: float, ari: float, engine) -> None:
        """Cross-fade to a new ambient track.

        Fades out the current track and starts the new one simultaneously so
        there is a brief overlap rather than a gap between scenes.

        Args:
            filepath:  Absolute or relative path to the audio file.
            skip_time: Seconds into the file to start playback from.
            ari:       Ambient Repeat Interval — seconds of audio to load and
                       loop (keeps RAM bounded for long files).
            engine:    The AudioEngine instance (from state['soundengine']).
        """
        if self._current_name is not None:
            engine.fade_out_audio(self._current_name, self.FADE_OUT_DURATION)

        import time
        name = filepath.name  # filename used as unique event name
        engine.schedule_event(
            filepath,
            time.time(),
            ari,
            repeat_interval=ari,
            inname=name,
            fade_in_duration=self.FADE_IN_DURATION,
            skip_time=skip_time,
        )
        self._current_name = name
        self._engine = engine
        print(f"[AmbientAudio] Cross-fading to: {filepath.name}")

    def stop(self) -> None:
        """Fade out the current ambient track."""
        if self._current_name is not None and self._engine is not None:
            self._engine.fade_out_audio(self._current_name, self.FADE_OUT_DURATION)
            self._current_name = None
