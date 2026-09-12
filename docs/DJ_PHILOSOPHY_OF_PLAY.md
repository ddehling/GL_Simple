# The DJ's philosophy of play

*Why the Director plays the way it does. Written 2026-09-12 after the user asked for "a philosophy of
musical play that isn't just schizophrenically jamming everything together at random", from what working
DJs say they do and what measurements of real mixes show they do. Every principle names the mechanism
that carries it and the dial that bends it. The user's ear outranks all of it.*

## What the evidence says DJs actually do

**Measured on real mixes**

- 1,557 mixes from 1001Tracklists (Kim et al. 2020, "A computational analysis of real-world DJ mixes"):
  tempo is barely touched (86 % of tracks adjusted under 5 %, 94.5 % under 10 %); key is almost never
  transposed (2.5 % of tracks, nearly all by one semitone); **transition lengths peak at multiples of 32
  beats** (the phrase); different DJs cue the same track at the same places (74 % of cue points within 8
  bars of each other) - songs have canonical in and out points, and DJs find them; a 60-minute mix holds
  about 15 tracks, roughly four minutes each.
- 114 BBC Essential Mixes (Kell & Tzanetakis, ISMIR 2013): consecutive tracks are **timbrally close**
  (closer than random dance music, looser than an album); **loudness and tempo are held nearly constant**;
  **key showed no significant pattern** - DJs do not order by key; **order matters** (a shuffle of the same
  tracks measures rougher); and **evenly sized steps beat nearest-neighbour ordering** - a set is a walk of
  similar-sized steps, not a slide down a similarity ranking that drifts.

**What working DJs say** (teaching sites, forums, interviews; the same points recur everywhere)

- *Phrasing*: dance music moves in 8/16/32-bar phrases; you mix on the phrase, out of an outro or a
  stripped section into an intro; blends run 16-64 bars in house and techno, cuts in hip-hop and open
  format; the bass swaps on a phrase boundary, never mid-phrase.
- *Payoff*: never interrupt a track before it has delivered what it was played for - "like telling a joke
  without the punchline"; a track earns its exit after its hook / drop has been heard.
- *Play length*: about 2.5 minutes in quick-mix and party sets, 3.5 as the club default, 5 with long blends;
  quick mixing all night "tires the crowd out".
- *Energy*: waves, not a constant peak - a build of small steps, a release, then space; after a peak, pull
  back a level for two or three tracks so the next peak hits; the biggest tricks (double drops, key-up
  boosts) "one or two per set"; ten bangers in a row stop being bangers.
- *Contrast*: at the same energy, change texture (bright after dark, vocal after instrumental); vary the
  technique too - all long blends sounds as monotonous as all cuts.
- *Reading and testing*: try a track, watch, and if it works follow that path; give a track time - the floor
  needs a phrase to arrive.
- *Stems*: an acapella over another instrumental, a drums swap before a melody swap, strip to bass or voice
  to build tension; two basslines never; two singers never; a lone vocal stem for long sounds thin.

## The principles, and where each one lives

| # | Principle | Mechanism | Dial that bends it |
|---|-----------|-----------|-------------------|
| 1 | **A song is heard to its payoff.** Nothing joins or replaces a song before its drop / hook has played. | autoDJ: exits drawn past the payoff (`_draw_exit`, entry runway floor). Conductor: a song enters at its **landmark** (the groove after its first build); a **voice takes the bed at its own drop** (`_drop_in_bars`), else after `VOICE_MAX_PHRASES`. | PACE (how much longer than the payoff) |
| 2 | **Everything happens on the phrase.** Moves land on bar and phrase boundaries; seams are 1-2 phrases long. | autoDJ: phrase-aligned seams, blend lengths in beats. Conductor: moves every `change_bars`, snapped to the master's section boundaries (`_snap_to_section`). | PACE (4 / 8 / 16 bars), SEAMS |
| 3 | **Let it breathe.** A new bed is heard as itself before the next song arrives. | Conductor: `SETTLE_PHRASES` (3) after a bed change before a voice may enter; a bed **building to its drop** admits nothing on top of it; a breakdown of the bed opens the door early. | PACE |
| 4 | **Tension, then release.** A bed change is a breakdown first, then the new drums and bass land on the one. | Conductor: `_bed_to` with `strip_before_bed` - the old bed's drums + bass out for `STRIP_BARS` (4), the new bed lands as a cut. | MOMENTS (rare = clean crosses), MIXING (blend / morph cross gradually) |
| 5 | **Peaks are spaced and earned.** The DJ's own drops and breaks never stack, and come when the arc is hot. | Director: `MOMENT_GAP_S` (8 min at some, 4 at lots) since the last moment - yours or the DJ's; at "some" only while the arc target is warm. | MOMENTS, ENERGY |
| 6 | **Small, even steps.** Consecutive songs are close in timbre and energy; the night walks, it does not lurch or drift. | Brain: arc energy target with a tight pull, timbral variety penalty against clones, genre / era coherence, valence continuity, blendability of the seam in AND out. | ENERGY, VARIETY, VOCALS |
| 7 | **Tempo and loudness hold; key is a courtesy, not a law.** | Brain: the stretch wall (~5.5 %), the tempo arc; key as a soft term with chroma refinement and a one-semitone rescue priced below an honest match. Submix: peak guard, level trims. Conductor: harmonic guard on the tonal lanes. | TEMPO, LEVEL |
| 8 | **One bed, one or two voices.** Drums and bass belong together and to one song; other songs speak over them; never two basslines, never two singers. | Conductor arrangement policy: BED = drums + bass, VOICES = other / vocals; max voices by LAYERS; the vocal lane follows measured singing. | LAYERS, VOCALS |
| 9 | **Contrast without chaos.** Same energy, different texture; different techniques over the night. | Brain: the similarity penalty (`s_var`) and anti-streak on seam styles; the dice on "auto" mixing. | MIXING auto, VARIETY, TONE |
| 10 | **Follow the room.** GOOD / BAD change tonight's weights at once and the next picks lean toward what was liked. | Director: seam-style weights, move-kind weights, and **taste** (the rated song's tags as prefer / avoid leans, halving with every new verdict). BAD also acts now. | GOOD / BAD |
| 11 | **Say why.** Every wait and every move has a sentence. | Conductor `wait_why` + `arrangement` status; Director WHY and the intent line ("bed: X for 12 bars, settles 12 more"; "voice Y takes the bed at its drop in ~6 bars"; "arc building"). | - |
| 13 | **Know the parts, not just the labels.** The hook is the payoff; the chords that overlap are what clash. | `lib/dj/measure.py`: per-section chroma from the harmonic stems and the HOOK from the vocal stem (loudness × repetition, ML chorus as a bonus, the vocal curve against bleed). Conductor: the tonal guard compares overlapping sections; a voice takes the bed at its hook or its drop. Brain: the seam's key term reads the chroma at the out / in points. | - |
| 12 | **The night has a shape you can see and hold.** Warm-up, build, peak, release - a plan, and the freedom to abandon it. | Director arc: the ARC STRIP (the plan as a curve, the songs that played as dots at their energy, the playhead, the peak ahead); ARC dial (theme / steady / build / waves / down / yours), LENGTH dial (45 m / 90 m / 3 h / night); click the strip = "we are here", drag = bend the plan. Both engines chase the same curve (waypoints). | ARC, LENGTH, ENERGY |

## What this is not

- Not a set of hard gates. Every principle is a lean the dials can overrule; a floor that is dying is the
  operator's call (NEXT, DROP, the energy dial), and the arrangement steps aside for it.
- Not the last word. The numbers (3 settle phrases, 2-5 voice phrases, 4-bar strip, 8/4-minute moment
  spacing) are first guesses from the literature; the user's ear at a party decides them.

## Sources

- Kim, Kim, Nam - "A Computational Analysis of Real-World DJ Mixes using Mix-To-Track Subsequence
  Alignment" (ISMIR 2020), arXiv:2008.10267.
- Kell, Tzanetakis - "Empirical Analysis of Track Selection and Ordering in Electronic Dance Music using
  Audio Feature Extraction" (ISMIR 2013).
- Williams, Meehan, Lattner, Pauwels, Barthet - "Temporal Considerations in DJ Mix Information Retrieval
  and Generation" (TIME 2025): DJ decisions at macro (set), meso (sequence) and micro (seam) time scales.
- Digital DJ Tips: "How pro DJs know where to transition", "How much of a song should DJs play?".
- Vibes / Mixgraph / Setflow / ZIPDJ / Point Blank guides on phrase mixing, energy flow, set structure.
- Relentless Beats "How DJs read a crowd", DJ City "Building tension & release", Mixed In Key "5 mixing
  techniques" (energy boost: "one or two per set").
- Serato / Traktor stems guides and DJ TechTools on live stem practice.
