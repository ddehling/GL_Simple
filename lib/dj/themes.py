"""Show themes: what kind of night the DJ is running.

A Theme shapes track selection (bpm window, energy arc, mood/spectral
leans) and transition style weighting. The arc maps set progress (0..1)
to an energy TARGET the brain chases when picking the next track; the
`all_night` theme stretches its arc over config `night_hours` of wall time
and is the source of outstate['dj_arc_phase'] for the visual night phase.
"""
import math
from dataclasses import dataclass, field, replace


@dataclass
class Theme:
    name: str
    bpm_range: tuple = (100.0, 130.0)
    energy_base: float = 0.5          # arc floor (0..1)
    energy_span: float = 0.4          # arc swing above the floor
    arc: str = "flat"                 # flat | rise | peak_wave | wind_down | all_night
    mood_weights: dict = field(default_factory=dict)   # mood -> weight
    spectral_lean: str = ""           # "" | "bass" | "high"
    # Clean, reliable transitions dominate; the elaborate techniques are
    # occasional accents (they sound bad if even slightly off, so they must
    # be the exception, not the rule).
    # Long smooth blends dominate; punchy exits (echo/cut/drops) are RARE
    # accents - at equal weights the night sounded fast and harsh.
    style_weights: dict = field(default_factory=lambda: {
        "long_blend": 1.7, "bass_swap": 1.2, "cut_at_drop": 0.08,
        "loop_roll_exit": 0.15, "bassline_layer": 0.15, "double_drop": 0.08,
        "loop_build": 0.12, "long_fade": 0.3,
        "filter_sweep": 0.6, "echo_out": 0.12})
    min_play_s: float = 150.0
    max_play_s: float = 420.0
    # FLAVOR: what KIND of music this night leans on, expressed in the
    # library's own vocabulary. prefer_tags/avoid_tags act on each track's
    # auto+user tags; axis_targets pull toward positions on the analysis
    # axes (hardness / hypnotic / vocal / speed, all 0..1). Without this
    # every theme converged on the same groove-optimal picks night after
    # night (user-reported).
    prefer_tags: dict = field(default_factory=dict)   # tag -> weight 0..1
    avoid_tags: dict = field(default_factory=dict)    # tag -> weight 0..1
    axis_targets: dict = field(default_factory=dict)  # axis -> target 0..1
    # ML-mood lever (Music2Emo): danceability target the brain pulls toward,
    # 0..1 or None. ONLY bites once tracks are mood-scored (lib/dj/mood_ml);
    # None on unscored libraries, so it never changes pre-mood behavior.
    dance_target: float = None                        # 0..1 or None

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


# Where each theme LIVES in a library's tempo landscape, as percentiles of
# the library's own bpm distribution. The authored bpm_range numbers below
# assume an EDM-shaped library (~110-130); on an eclectic library (this
# user's: median 100, p10 77) those absolute windows strand the brain in a
# thin corner - measured on the real 573-track library: 31% of tracks had
# NO compatible successor, stretches were forced to 7.7%, and 54% of seams
# fell back to long_fade. adapt_theme() keeps each theme's CHARACTER (its
# relative position: peak_heavy = the fast end) while fitting the numbers
# to the music that's actually there.
THEME_BPM_PERCENTILES = {
    "chill_evening": (0.08, 0.45),
    "groove": (0.35, 0.75),
    "peak_heavy": (0.62, 0.97),
    "wind_down": (0.03, 0.40),
    "all_night": (0.10, 0.92),
    "hypnotic_deep": (0.30, 0.68),
    "vocal_journey": (0.25, 0.65),
    "hard_drive": (0.55, 0.90),
    "gentle_organic": (0.15, 0.55),
}


def adapt_theme(theme, bpms, min_width=14.0):
    """Return a copy of ``theme`` with bpm_range fitted to THIS library's
    tempo distribution (see THEME_BPM_PERCENTILES). On an EDM library the
    percentiles land back on EDM numbers, so well-matched libraries are
    unchanged in practice. Unknown/custom themes and tiny libraries pass
    through untouched."""
    pcts = THEME_BPM_PERCENTILES.get(theme.name)
    v = sorted(b for b in (bpms or []) if b and b > 0)
    if not pcts or len(v) < 20:
        return theme
    lo = v[int(pcts[0] * (len(v) - 1))]
    hi = v[int(pcts[1] * (len(v) - 1))]
    if hi - lo < min_width:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - min_width / 2.0, mid + min_width / 2.0
    return replace(theme, bpm_range=(round(lo, 1), round(hi, 1)))


BUILTIN_THEMES = {t.name: t for t in [
    Theme("chill_evening", bpm_range=(85.0, 115.0),
          energy_base=0.35, energy_span=0.2, arc="flat",
          mood_weights={"chill": 1.0, "ambient": 0.6, "groove": 0.4},
          style_weights={"long_blend": 1.5, "bass_swap": 0.8,
                         "cut_at_drop": 0.0, "loop_roll_exit": 0.15,
                         "bassline_layer": 0.15, "long_fade": 0.8,
                         "filter_sweep": 0.7, "echo_out": 0.3},
          dance_target=0.3,
          prefer_tags={"relaxing": 0.5, "calm": 0.4},
          min_play_s=210.0, max_play_s=480.0),
    Theme("groove", bpm_range=(105.0, 128.0),
          energy_base=0.55, energy_span=0.25, arc="peak_wave",
          dance_target=0.65,
          mood_weights={"groove": 1.0, "chill": 0.4, "peak": 0.5}),
    Theme("peak_heavy", bpm_range=(122.0, 145.0),
          energy_base=0.75, energy_span=0.25, arc="peak_wave",
          mood_weights={"peak": 1.0, "groove": 0.6},
          spectral_lean="bass",
          style_weights={"long_blend": 1.2, "bass_swap": 1.3,
                         "cut_at_drop": 0.25, "loop_roll_exit": 0.25,
                         "bassline_layer": 0.3, "double_drop": 0.25,
                         "loop_build": 0.25, "long_fade": 0.1,
                         "filter_sweep": 0.7, "echo_out": 0.25},
          dance_target=0.85,
          prefer_tags={"party": 0.6, "energetic": 0.5},
          min_play_s=120.0, max_play_s=300.0),
    Theme("wind_down", bpm_range=(80.0, 112.0),
          energy_base=0.35, energy_span=0.3, arc="wind_down",
          mood_weights={"chill": 1.0, "ambient": 0.8, "groove": 0.3},
          style_weights={"long_blend": 1.2, "bass_swap": 0.4,
                         "cut_at_drop": 0.0, "loop_roll_exit": 0.2,
                         "long_fade": 1.0, "filter_sweep": 0.6},
          dance_target=0.25,
          prefer_tags={"relaxing": 0.5, "melancholic": 0.3},
          min_play_s=240.0, max_play_s=540.0),
    Theme("all_night", bpm_range=(95.0, 138.0),
          energy_base=0.55, energy_span=0.45, arc="all_night",
          dance_target=0.6,
          mood_weights={"groove": 1.0, "peak": 0.7, "chill": 0.5}),
    # FLAVORED nights - same machinery, different corners of the library.
    Theme("hypnotic_deep", bpm_range=(105.0, 124.0),
          energy_base=0.5, energy_span=0.2, arc="flat",
          mood_weights={"groove": 1.0, "chill": 0.6},
          prefer_tags={"hypnotic": 1.0, "instrumental": 0.5},
          avoid_tags={"peaky": 0.7, "vocals": 0.4},
          axis_targets={"hypnotic": 0.95, "hardness": 0.45},
          dance_target=0.7,
          min_play_s=210.0, max_play_s=480.0),
    Theme("vocal_journey", bpm_range=(100.0, 124.0),
          energy_base=0.5, energy_span=0.3, arc="peak_wave",
          mood_weights={"groove": 1.0, "chill": 0.5},
          prefer_tags={"vocals": 1.0, "vocal-heavy": 0.8},
          avoid_tags={"instrumental": 0.5},
          axis_targets={"vocal": 0.6},
          dance_target=0.6),
    Theme("hard_drive", bpm_range=(115.0, 132.0),
          energy_base=0.7, energy_span=0.3, arc="rise",
          mood_weights={"peak": 1.0, "groove": 0.7},
          spectral_lean="bass",
          prefer_tags={"hard": 1.0, "driving": 0.8},
          avoid_tags={"gentle": 0.8, "mellow": 0.6},
          axis_targets={"hardness": 0.9, "energy": 0.85},
          dance_target=0.85,
          min_play_s=120.0, max_play_s=300.0),
    Theme("gentle_organic", bpm_range=(95.0, 120.0),
          energy_base=0.4, energy_span=0.25, arc="flat",
          mood_weights={"chill": 1.0, "groove": 0.6},
          prefer_tags={"gentle": 1.0, "mellow": 0.7, "relaxing": 0.5},
          avoid_tags={"hard": 0.8, "peaky": 0.6},
          axis_targets={"hardness": 0.2},
          dance_target=0.35,
          min_play_s=210.0, max_play_s=480.0),
]}


def get_theme(name):
    return BUILTIN_THEMES.get(name, BUILTIN_THEMES["groove"])
