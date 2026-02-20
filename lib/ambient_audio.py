import time
import threading
from pathlib import Path
from typing import Optional

from corefunctions.soundtestthreaded import StreamingPlayer


class AmbientAudioController:
    """Manages the single ambient StreamingPlayer: fade-out old, fade-in new.

    The cross-fade runs on a background thread so the render loop is never
    blocked by pygame.mixer.music.load() or fadeout().
    """

    FADE_OUT_DURATION = 2.0   # seconds to fade the outgoing track
    FADE_IN_DURATION  = 5.0   # seconds to fade in the incoming track

    def __init__(self) -> None:
        self._current_player: Optional[StreamingPlayer] = None

    def transition(self, filepath: Path, volume: float, skip_time: float, engine) -> None:
        """Start a cross-fade to a new ambient track on a background thread."""
        old_player = self._current_player
        self._current_player = None

        def _do_transition():
            if old_player is not None:
                old_player.fade_out(self.FADE_OUT_DURATION)
                time.sleep(self.FADE_OUT_DURATION)
            new_player = StreamingPlayer(
                engine=engine,
                filepath=filepath,
                name="ambient",
                loop=True,
                volume=volume,
                fade_in=self.FADE_IN_DURATION,
                skip_time=skip_time,
            )
            new_player.start()
            self._current_player = new_player

        threading.Thread(target=_do_transition, daemon=True,
                         name="audio-transition").start()

    def stop(self) -> None:
        """Stop the current ambient track immediately."""
        if self._current_player is not None:
            self._current_player.stop()
            self._current_player = None
