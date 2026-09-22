# The blend: seams that are long, blended, and shaped

*2026-09-21, after four "bad transition / seam" flags in an hour and "the transitions are really rough right
now. fast, no blending, no artistry". One page. "yes" the same evening; the status is at the end.*

## What a seam is today, in numbers (SEAMS = normal, the default)

| step | what happens | how long it crosses |
|---|---|---|
| B's drums in under A's harmony | a stem gain ramp | 2 bars |
| A darkens (bass swap only) | high shelf to 0.7 | 4 bars |
| the harmony swap: A's bass + melody out, B's in | two ramps, a low shelf handing the bass over | 2 bars |
| A's vocal | goes with its harmony | 2 bars |
| A's drums out (bass swap) | a ramp | 2 bars |
| A leaves (whatever it still holds) | gain to zero | **1.5 beats** |
| the cut (NEXT, a hole, a score's first groove) | 4-bar strip, then B whole | 0.5 beat |

So a "16-bar morph" is three switches a phrase apart and a slam at the end. The ramps are linear in gain,
which dips the bus at the middle of every cross. B's drums may enter from a build or an intro (Astra's build
drums under In The Beginning, 20:15). And when a rehearsal hears anything wrong, the shape collapses to the
cut. None of this is a blend, and the ear says so.

The autoDJ's long fade, tuned by the user's ear over weeks, was never rejected: a whole-record crossfade over
many bars with the low end handed over. The layered engine has stems, which let it do that fade BETTER - the
drums can lead, the bass can hand over cleanly - but it has been using them to switch.

## The blend (the default from here)

1. **Long.** The default seam is 32 bars; 64 when both records have the body ("blend"). Every cross is a
   phrase: B's drums in over 8 bars, the harmony swap over 8 bars, A's drums out over 4, and A's last stems
   leave over a phrase - never 1.5 beats. SEAMS: quick = today's normal (2-bar crosses), normal = 8-bar
   crosses, long = 16-bar crosses. The cut stays a cut (MIXING = cut, NEXT, a first-groove score, a hole).
2. **Equal power.** A cross follows an equal-power curve, not a straight line: the bus does not dip at the
   middle of a cross. One curve in the deck's stem ramp; every cross gets it.
3. **Level continuity through the whole cross.** The rehearsal's step correction becomes a rule of the blend:
   B's arriving stems are scaled so A + B stays within 1.5 dB of A alone across the cross, then ease to B's own
   level over the phrase after. (Built today for the step at the swap; extended to the whole cross.)
4. **The low end is handed over, never doubled and never empty.** Through the harmony cross A's low shelf goes
   down as B's comes up (the code has this for crosses over 2 beats; now it spans the 8 bars), and the kick
   stays with whichever record holds the drums - two kicks never sound full at once.
5. **B enters only from a groove that carries.** Its drums come in from a groove of its own (never a build or
   an intro) that its stems carry through the whole cross; the swap lands on a phrase boundary of BOTH
   records; the landmark rule stays. Where no such place exists in time, the blend is shorter (16 bars), not
   a cut - the cut is for holes and for what you ask for.
6. **The vocal in the blend.** A's vocal leaves with its harmony over the phrase; B's vocal comes only after
   its harmony has landed, and only where it sings (rule f: yours holds).
7. **Shape, not FX.** The EQ hand-over and the darkening are the shapes; no filter sweeps, no echo (rejected
   before). A blend may end on the one of B's drop when B's landmark is a drop: the harmony lands the bar
   before, as now.

## Feedback: hear it before it plays

The rehearsal already renders every seam two phrases ahead. **HEAR THE SEAM** (the PLAN's SEAM row, right-click,
like PAIRS' HEAR) plays that render on the cue output; the SEAM row shows the rehearsal's verdict and its
levels. You hear what is coming while it can still be moved or cut - the headphone cue the design promised on
2026-09-13 and never delivered. And every flagged seam is re-rendered to `logs/seams/<hms>.wav` at the flag,
so "bad transition" comes with the sound attached, not only the room.

## How it is proved

- The remix gate and the control gate as now (holes, lurches, dead bars, the contract).
- The record campaign (eight twelve-minute nights): the lurch and dead-air numbers per seam before and after;
  the blend must not raise either. A new number: the bus level's range across each cross (target within 3 dB).
- Six seams rendered to wav from the campaign, chosen at random, for the user's ear - the pass mark.

## Status (2026-09-21, night)

**Built.** SEAM_BARS morph 32 / blend 64 (quick halves, long doubles). The crosses: quick = two bars (what
normal was), normal = a phrase, long = two phrases - `_step_beats`; the drums following a harmony cross over
four bars at most; the harmony lead and the landmark offset follow the cross length (`_seam_lm_off`); A's
vocal leaves with its harmony over the same cross. The deck's stem ramp is EQUAL POWER for any ramp of a
second or more (sin in, cos out); switches stay linear. B's leading stems enter only from a groove of B that
carries through the cross (`entry_ok` in `_seam_b_start`); the placement ladder is 32 → 16 → 8 → the cut. The
rehearsal models the crosses as equal-power ramps of their real length (`_gain_schedule`), renders every
rehearsed seam as AUDIO to `logs/seams/seam_<seq>_<hms>_<B>.wav` (the last twelve kept; not in headless runs),
and a 'seam' flag copies the last one beside it (`flag_<stamp>_seam.wav`, named in the flag's `seam_wav`).
HEAR THE SEAM: right-click the PLAN's SEAM row - the rehearsal's wav on the cue output, or a render now.
Level continuity through the cross = equal power + the rehearsed step correction at the swap (built earlier
the same day); the continuous scaling of item 3 is not separate code.

**Not yet.** The campaign's before/after numbers and the six renders for the ear; a bus-level-range measure per
cross in the campaign.

## Cost, in order

1. The blend's lengths, the phrase-long leave, equal power, level continuity through the cross, the entry rule:
   a day.
2. HEAR THE SEAM on the cue output, the flag's wav: half a day.
3. The campaign and the six renders: an evening of machine time.
