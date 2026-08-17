"""DJ personas: HOW the DJ plays the night it was given.

A persona is orthogonal to a Theme. The theme is the operator's order -
what music, what tempo window, what arc. The persona is the mixing
temperament layered on top: patience, theatricality, harmonic taste,
exploration appetite, one signature habit. Two `groove` nights under
different personas should be audibly different DJs playing the same brief.

HARD RULE: personas multiply weights only inside the menus the defensive
gates have already approved. Every safety gate in the brain (flam, kick
offset, grid confidence, vocal clash, meter clash) is absolute and persona
-blind. `neutral` is all-identity and byte-identical to pre-persona
behavior; it is the default everywhere.

Selection-side effects (key strictness, exploration, vocal pull, bpm
widening) are deliberately weaker than technique-side effects - persona
must never fight the theme's flavor steering or the operator's chips.
"""
import datetime
import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Persona:
    name: str
    tagline: str                     # banner line: "tonight: <name> - <tagline>"
    # -- technique (seam styles, pacing) -----------------------------------
    style_bias: dict = field(default_factory=dict)  # style -> menu multiplier
    theatrics: float = 1.0           # scales the punchy accent tier as a whole
    moment_cooldown_x: float = 1.0   # scales the engineered-moment cooldown
    p96: float = 0.35                # chance a long_blend stretches to 96 beats
    play_len_x: float = 1.0          # scales the arc-coupled play length
    # Scales the entry-runway floor (brain.best_pair): how much of the
    # theme's minimum play an entry point must leave room for. 1.0 =
    # every record must be able to host a full minimum song; below 1.0
    # the persona is allowed drop-in/drop-out records - a taste knob for
    # deliberate fast rotation, distinct from play_len_x which paces the
    # WHOLE night. Keep it on exactly the personas whose identity is
    # short records, or the floor flattens the pacing spread it protects.
    entry_floor_x: float = 1.0
    # -- selection leans (half-strength by design) -------------------------
    key_strictness: float = 1.0      # exponent on the key-compat score (>1 = stricter)
    explore: float = 1.0             # widens selection dice + flattens finalist sampling
    groove_tolerance: float = 1.0    # >1 softens the groove/rhythm selection leans
    vocal_pull: float = 0.0          # -1..1 lean away from / toward vocal tracks
    bpm_widen: float = 1.0           # >1 relaxes the theme tempo window (taste, not safety)


NEUTRAL = Persona(
    name="neutral",
    tagline="no persona - the house style",
)

PERSONAS = {p.name: p for p in [
    NEUTRAL,
    # Patient, orthodox, strict keys, near-zero theatrics. 96-beat
    # marathons, digs one harmonic pocket all night, tracks breathe long.
    Persona("monk", "playing it patient - long blends, one deep pocket",
            style_bias={"long_blend": 1.6, "filter_sweep": 1.2},
            theatrics=0.3, moment_cooldown_x=1.6,
            p96=0.60, play_len_x=1.30,
            key_strictness=1.6, explore=0.7, vocal_pull=-0.3),
    # Moments early and often, punchy tier up, shorter records; the
    # signature is the hard cut onto the incoming drop.
    # (The signature was loop_build until 2026-08-13, by which point that
    # style had been retired for nine days - so showman's entire
    # style_bias was multiplying a weight that kill() zeroed on every
    # seam. cut_at_drop was reinstated the same week and is what the
    # tagline literally describes; double_drop went 2026-08-02, the
    # nextdrop MOMENT owns the synced-drop spectacle.)
    # entry_floor_x 0.55: "short records" is this persona's literal
    # tagline, so it keeps the right to drop into a record's back half
    # and ride only its last drop (~70-90s) - the entry-runway floor
    # that protects every other persona from accidental 80-second songs
    # (2026-08-16) is, for showman, the feature itself.
    Persona("showman", "big moments, short records, drops on drops",
            style_bias={"cut_at_drop": 2.0},
            theatrics=1.9, moment_cooldown_x=0.55,
            p96=0.15, play_len_x=0.80, entry_floor_x=0.55,
            key_strictness=0.75, vocal_pull=0.15),
    # Harmonic journeys: strictest keys of all, surgical single-bassline
    # handoffs, spectacle declined in favor of a clean modulation.
    # SEPARATED FROM MONK 2026-08-13. Measured over 3x6h nights, purist
    # differed from NEUTRAL on exactly one axis - key coherence 83.6% vs
    # 71.0% - and that signature was indistinguishable from monk's 84.3%.
    # Everything else (p96, play_len_x, moment_cooldown_x) sat at the
    # neutral default, so "the purist" was a night nobody could pick out
    # of a lineup that already contained monk.
    # Its own line says SURGICAL: short decisive swaps, not marathons, and
    # spectacle declined for a clean modulation. p96 down and the moment
    # cooldown up say that in the levers, and they are exactly where monk
    # goes the other way (0.60 / 1.6) - so the two tidy-key personas now
    # differ on how they SPEND that tidiness.
    Persona("purist", "clean key journeys, surgical swaps",
            style_bias={"bass_swap": 1.8},
            theatrics=0.8, moment_cooldown_x=1.35,
            p96=0.18, play_len_x=0.95,
            key_strictness=2.2, explore=0.8),
    # Range over pocket: exploration up, groove-lean softened, tempo
    # window blurred at the edges - and the deliberate palate-cleanser
    # fade into an outlier is a feature, not a failure.
    # (First sim pass read too close to neutral: explore 1.8 / widen 1.10
    # barely moved the histograms, and widening REDUCED fades - more of the
    # library became reachable. These values measured distinct.)
    # (style_bias was loop_roll_exit until 2026-08-13 - retired 2026-08-04,
    # so for nine days this persona's signature multiplied a weight kill()
    # zeroed on every seam. It stayed measurably distinct anyway, on
    # explore/key_strictness: lowest key coherence of any persona at 62.9%
    # against neutral's 71.0%. filter_sweep is the technique that fits what
    # it actually does - a sweep is how you bridge two records that do not
    # share a pocket, which is this persona's whole premise.)
    Persona("crate_digger", "range over pocket - expect a curveball",
            style_bias={"filter_sweep": 1.6},
            theatrics=1.1, p96=0.25, play_len_x=0.90,
            key_strictness=0.45, explore=2.6,
            groove_tolerance=2.0, bpm_widen=1.18),
    # Voices staged as entrances: stem styles up, vocal tracks clustered
    # into acts, bigger breathing room around a singer.
    # long_blend ADDED 2026-08-13: both stem signatures are gated on
    # rendered stems, which ~42% of the library has, and those styles are
    # 1-5% of seams even then - so this persona's entire technique bias
    # was inert on most nights and its identity rested on vocal_pull,
    # which was itself dead until the same day. A voice needs ROOM, and
    # the long blend is the only room-making style that always exists.
    # (style_bias was acapella_out 2.2 until 2026-08-16 - benched that
    # day on the operator's Lab verdicts ("it just felt pointless"), and
    # a signature multiplying a benched weight is the exact inert-bias
    # trap crate_digger fell into for nine days. acapella_in carries the
    # voices-as-entrances premise instead: the surviving vocal-stem
    # style, and the one whose mechanic - a voice arrives, then its own
    # full mix - actually stages an entrance.)
    Persona("storyteller", "the voices carry the night",
            style_bias={"acapella_in": 2.2, "stem_drum_swap": 1.8,
                        "long_blend": 1.3},
            theatrics=1.1, moment_cooldown_x=0.9,
            p96=0.55, play_len_x=1.10,
            key_strictness=1.2, vocal_pull=0.5),
]}

# Auto-rotation draws from the real characters only.
ROTATION = [n for n in PERSONAS if n != "neutral"]


def for_night(date=None, avoid=None):
    """The night's persona under auto-rotation: date-seeded (one character
    per calendar night, stable across restarts), avoiding yesterday's name
    when given. Local Random - never touches global RNG state."""
    d = date or datetime.date.today()
    rng = random.Random(d.toordinal() * 2654435761 % (2 ** 31))
    pool = [n for n in ROTATION if n != avoid] or ROTATION
    return PERSONAS[rng.choice(pool)]
