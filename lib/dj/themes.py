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
    # Long smooth blends DOMINATE; punchy exits are accents - at equal
    # weights the night sounded fast and harsh (user-confirmed), and that
    # ordering is not up for revision.
    #
    # The accent WEIGHTS were, though. Measured over 560 logged seams
    # (tools/dj/dj_review.py), the assumption behind the 0.08-0.15 tier -
    # "they sound bad if even slightly off" - is only true of one of them:
    #
    #     echo_out        median flam 0.034 beats   (best of any style)
    #     loop_build                   0.044
    #     loop_roll_exit               0.056
    #     filter_sweep                 0.061
    #     long_blend                   0.064
    #     bass_swap                    0.068
    #     cut_at_drop                  0.247        <- 4x everything else
    #
    # The short decisive exits measure BETTER than the workhorse blends,
    # because a short overlap gives the lock less time to drift. At 0.12
    # they fired 7 and 9 times in 560 seams - not "rare accent", closer to
    # never, and a 30-seam night saw none of them at all. Raised to a tier
    # where a night gets one or two, still an order of magnitude under the
    # blends. cut_at_drop stays at the bottom: a spectacle move whose job
    # is the engineered moment, and the only technique the measurements
    # actually indict. (bassline_layer and double_drop removed 2026-08-02.)
    style_weights: dict = field(default_factory=lambda: {
        "long_blend": 1.7, "bass_swap": 1.2, "cut_at_drop": 0.08,
        "loop_roll_exit": 0.30,
        "loop_build": 0.28, "long_fade": 0.3,
        "filter_sweep": 0.6, "echo_out": 0.26,
        # Stem styles (inert until tools/dj/dj_stems.py has rendered stems):
        # accents, same tier as the other elaborate techniques.
        "stem_drum_swap": 0.3, "acapella_out": 0.2})
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
# Bands widened APART 2026-07-24. The pairs that measured as duplicates
# (tools/tests/_dj_theme_sim.py: hard_drive/peak_heavy at 0.45 track overlap,
# chill_evening/gentle_organic at 0.33) were the pairs whose percentile
# windows sat almost on top of each other - hard_drive 0.55-0.90 inside
# peak_heavy's 0.62-0.97 is not a different tempo landscape, it is the
# same one. peak_heavy now owns the TOP of the library and hard_drive the
# relentless upper-middle.
#
# TEMPO SEPARATION MATTERS MORE THAN FLAVOR HERE, because a set walks a
# tempo CORRIDOR: every pick must sit within a few percent of the last
# one, so two themes that start in the same band stay in the same band no
# matter how differently their tag leans are written. chill_evening and
# gentle_organic kept 33% of their records in common through three rounds
# of sharper flavor levers, and only separated when their windows stopped
# touching. gentle_organic moved UP rather than chill moving down: warm
# melodic organic house genuinely lives at 100-120, and the bottom of the
# library belongs to wind_down and chill_evening.
THEME_BPM_PERCENTILES = {
    "chill_evening": (0.08, 0.32),
    "groove": (0.35, 0.75),
    "peak_heavy": (0.72, 1.00),
    "wind_down": (0.00, 0.24),
    "all_night": (0.10, 0.92),
    "hypnotic_deep": (0.30, 0.68),
    "vocal_journey": (0.25, 0.65),
    "hard_drive": (0.50, 0.80),
    "gentle_organic": (0.36, 0.64),
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
                         "long_fade": 0.8,
                         "filter_sweep": 0.7, "echo_out": 0.3},
          dance_target=0.3,
          # SPARSE AND BRIGHT is chill_evening's own corner - room music
          # with air in it. Distinguishes it from gentle_organic below,
          # which owns the warm/melodic corner at the same tempo and
          # energy; the two measured 33% the same records without this.
          prefer_tags={"relaxing": 0.5, "calm": 0.4, "sparse": 0.7,
                       "bright": 0.5},
          avoid_tags={"bass-heavy": 0.5, "peaky": 0.5},
          axis_targets={"hardness": 0.25, "hypnotic": 0.4},
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
                         "loop_build": 0.4, "long_fade": 0.1,
                         "filter_sweep": 0.7, "echo_out": 0.25},
          dance_target=0.85,
          # peak_heavy is the SPECTACLE peak: the biggest, most dynamic
          # records in the box, breakdowns and drops included. hard_drive
          # below is the opposite temperament at a similar energy - no
          # breakdowns, just relentless. Same axis targets would collapse
          # them again, so this one leans on dynamics, that one on
          # steadiness.
          prefer_tags={"party": 0.6, "energetic": 0.5, "drops": 0.6,
                       "dynamic": 0.5},
          axis_targets={"energy": 0.9},
          min_play_s=120.0, max_play_s=300.0),
    Theme("wind_down", bpm_range=(80.0, 112.0),
          energy_base=0.35, energy_span=0.3, arc="wind_down",
          mood_weights={"chill": 1.0, "ambient": 0.8, "groove": 0.3},
          style_weights={"long_blend": 1.2, "bass_swap": 0.4,
                         "cut_at_drop": 0.0, "loop_roll_exit": 0.2,
                         "long_fade": 1.0, "filter_sweep": 0.6},
          dance_target=0.25,
          prefer_tags={"relaxing": 0.5, "melancholic": 0.3, "mellow": 0.7,
                       "long-intro": 0.3},
          avoid_tags={"peaky": 0.8, "hard": 0.7, "drops": 0.5},
          axis_targets={"energy": 0.15},
          min_play_s=240.0, max_play_s=540.0),
    Theme("all_night", bpm_range=(95.0, 138.0),
          energy_base=0.55, energy_span=0.45, arc="all_night",
          dance_target=0.6,
          mood_weights={"groove": 1.0, "peak": 0.7, "chill": 0.5}),
    # FLAVORED nights - same machinery, different corners of the library.
    Theme("hypnotic_deep", bpm_range=(105.0, 124.0),
          energy_base=0.5, energy_span=0.2, arc="flat",
          mood_weights={"groove": 1.0, "chill": 0.6},
          # A hypnotic night does not break down - the whole point is that
          # it never lets go. 'steady'/'breakdowny'/'dynamic' are real
          # percentile tags the scanner writes (25% of the library each),
          # so these are live levers, not aspirational words.
          prefer_tags={"hypnotic": 1.0, "instrumental": 0.5, "steady": 0.8},
          avoid_tags={"peaky": 0.7, "vocals": 0.4, "breakdowny": 0.6,
                      "dynamic": 0.4},
          axis_targets={"hypnotic": 0.95, "hardness": 0.45},
          dance_target=0.7,
          min_play_s=210.0, max_play_s=480.0),
    Theme("vocal_journey", bpm_range=(100.0, 124.0),
          energy_base=0.5, energy_span=0.3, arc="peak_wave",
          mood_weights={"groove": 1.0, "chill": 0.5},
          prefer_tags={"vocals": 1.0, "vocal-heavy": 0.8, "melodic": 0.4},
          avoid_tags={"instrumental": 0.5},
          # 0.9 on the RANKED vocal axis (load_library) = the most sung
          # material this collection has. The old 0.6 was read against the
          # raw demucs fraction, whose 90th percentile is 0.44 - a target
          # no track in the library could reach, so the pull was flat.
          axis_targets={"vocal": 0.9},
          dance_target=0.6),
    Theme("hard_drive", bpm_range=(115.0, 132.0),
          energy_base=0.7, energy_span=0.3, arc="rise",
          mood_weights={"peak": 1.0, "groove": 0.7},
          spectral_lean="bass",
          # Tag names must exist in the library's actual vocabulary or the
          # lever is a silent no-op. ('driving'/'mellow' matched 0 tracks
          # on the real library until 2026-07-24 - not because the words
          # were wrong but because ten NaN energy axes poisoned the
          # percentile that produced them; see scan._recalibrate_tags.)
          #
          # RELENTLESS, not spectacular: hard_drive keeps the floor moving
          # without a single moment of release, where peak_heavy spends
          # its energy on drops and breakdowns. They measured 45% the same
          # records while both simply asked for "loud and fast".
          prefer_tags={"hard": 1.0, "energetic": 0.7, "steady": 0.8,
                       "percussive": 0.6, "driving": 0.6},
          avoid_tags={"gentle": 0.8, "calm": 0.6, "breakdowny": 0.7,
                      "sparse": 0.5},
          axis_targets={"hardness": 0.95, "energy": 0.75,
                        "hypnotic": 0.75},
          dance_target=0.85,
          min_play_s=120.0, max_play_s=300.0),
    Theme("gentle_organic", bpm_range=(95.0, 120.0),
          energy_base=0.4, energy_span=0.25, arc="flat",
          mood_weights={"chill": 1.0, "groove": 0.6},
          # WARM AND PLAYED, not sparse and ambient - that corner belongs
          # to chill_evening, and without the split the two shared a third
          # of their records.
          prefer_tags={"gentle": 1.0, "calm": 0.7, "warm": 0.9,
                       "melodic": 0.7},
          avoid_tags={"hard": 0.8, "peaky": 0.6, "sparse": 0.5},
          axis_targets={"hardness": 0.1, "hypnotic": 0.35},
          dance_target=0.35,
          min_play_s=210.0, max_play_s=480.0),
]}


# WEB PICKER. Cut to five earlier on 2026-07-24 because the nine themes
# measured as about five behaviors (hard_drive 53% the same records as
# peak_heavy; a chill_evening/wind_down/gentle_organic overlap triangle;
# hypnotic_deep 30% on groove). The criterion was "offer only what
# measures distinct", and after the axis and band work later the same day
# they all do - tools/tests/_dj_theme_sim.py, one 3h night each on the real
# 529-track library:
#
#     hard_drive / peak_heavy          0.45 -> 0.20
#     chill_evening / gentle_organic   0.33 -> 0.12
#     chill_evening / wind_down        0.23 -> 0.14
#     hypnotic_deep / anything         0.30 -> 0.06 or less
#     vocal_journey / anything         ---- -> 0.09 or less
#
# The only pairs left above 0.20 involve all_night, which spans the whole
# library on purpose. So the menu goes back to the full table. Re-run the
# sim before adding or retiring a theme; the same criterion still applies.
PICKER_THEMES = ["chill_evening", "gentle_organic", "groove",
                 "hypnotic_deep", "vocal_journey", "hard_drive",
                 "peak_heavy", "wind_down", "all_night"]


def get_theme(name):
    return BUILTIN_THEMES.get(name, BUILTIN_THEMES["groove"])
