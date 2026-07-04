"""Show themes: what kind of night the DJ is running.

A Theme shapes track selection (bpm window, energy arc, mood/spectral
leans) and transition style weighting. The arc maps set progress (0..1)
to an energy TARGET the brain chases when picking the next track; the
`all_night` theme stretches its arc over config `night_hours` of wall time
and is the source of outstate['dj_arc_phase'] for the visual night phase.
"""
import math
from dataclasses import dataclass, field


@dataclass
class Theme:
    name: str
    bpm_range: tuple = (100.0, 130.0)
    energy_base: float = 0.5          # arc floor (0..1)
    energy_span: float = 0.4          # arc swing above the floor
    arc: str = "flat"                 # flat | rise | peak_wave | wind_down | all_night
    mood_weights: dict = field(default_factory=dict)   # mood -> weight
    spectral_lean: str = ""           # "" | "bass" | "high"
    style_weights: dict = field(default_factory=lambda: {
        "long_blend": 1.0, "bass_swap": 1.0, "cut_at_drop": 0.6,
        "loop_roll_exit": 0.6, "bassline_layer": 0.9, "double_drop": 0.5,
        "loop_build": 0.6, "long_fade": 0.3})
    min_play_s: float = 150.0
    max_play_s: float = 420.0

    def arc_target(self, progress):
        """Energy target in 0..1 for set progress 0..1."""
        p = max(0.0, min(1.0, progress))
        if self.arc == "rise":
            shape = p
        elif self.arc == "wind_down":
            shape = 1.0 - p
        elif self.arc == "peak_wave":
            # Two swells with a breather between - a club night in miniature.
            shape = 0.5 - 0.5 * math.cos(2 * math.pi * (0.25 + 0.75 * p)) \
                if p < 0.85 else 1.0 - (p - 0.85) / 0.15 * 0.4
            shape = max(0.0, min(1.0, shape))
        elif self.arc == "all_night":
            # Warm-up -> long peak plateau -> gentle landing.
            if p < 0.30:
                shape = p / 0.30 * 0.8
            elif p < 0.75:
                shape = 0.8 + 0.2 * math.sin(math.pi * (p - 0.30) / 0.45)
            else:
                shape = max(0.15, 0.8 - (p - 0.75) / 0.25 * 0.65)
        else:                          # flat
            shape = 0.5
        return max(0.0, min(1.0, self.energy_base
                            + self.energy_span * (shape - 0.5) * 2.0))


BUILTIN_THEMES = {t.name: t for t in [
    Theme("chill_evening", bpm_range=(85.0, 115.0),
          energy_base=0.35, energy_span=0.2, arc="flat",
          mood_weights={"chill": 1.0, "ambient": 0.6, "groove": 0.4},
          style_weights={"long_blend": 1.2, "bass_swap": 0.6,
                         "cut_at_drop": 0.1, "loop_roll_exit": 0.3,
                         "bassline_layer": 0.4, "long_fade": 0.8},
          min_play_s=210.0, max_play_s=480.0),
    Theme("groove", bpm_range=(105.0, 128.0),
          energy_base=0.55, energy_span=0.25, arc="peak_wave",
          mood_weights={"groove": 1.0, "chill": 0.4, "peak": 0.5}),
    Theme("peak_heavy", bpm_range=(122.0, 145.0),
          energy_base=0.75, energy_span=0.25, arc="peak_wave",
          mood_weights={"peak": 1.0, "groove": 0.6},
          spectral_lean="bass",
          style_weights={"long_blend": 0.7, "bass_swap": 1.2,
                         "cut_at_drop": 1.2, "loop_roll_exit": 1.0,
                         "bassline_layer": 1.2, "double_drop": 1.0,
                         "loop_build": 1.0, "long_fade": 0.1},
          min_play_s=120.0, max_play_s=300.0),
    Theme("wind_down", bpm_range=(80.0, 112.0),
          energy_base=0.35, energy_span=0.3, arc="wind_down",
          mood_weights={"chill": 1.0, "ambient": 0.8, "groove": 0.3},
          style_weights={"long_blend": 1.2, "bass_swap": 0.4,
                         "cut_at_drop": 0.0, "loop_roll_exit": 0.2,
                         "long_fade": 1.0},
          min_play_s=240.0, max_play_s=540.0),
    Theme("all_night", bpm_range=(95.0, 138.0),
          energy_base=0.55, energy_span=0.45, arc="all_night",
          mood_weights={"groove": 1.0, "peak": 0.7, "chill": 0.5}),
]}


def get_theme(name):
    return BUILTIN_THEMES.get(name, BUILTIN_THEMES["groove"])
