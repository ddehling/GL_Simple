# The record plays: layered mode as a DJ plays

*The design agreed 2026-09-13 evening ("how can we make the system plan well and play tastefully?" →
"what is the gap?" → "yes"). Replaces the conductor's arrangement policy (one bed, voices arriving and
taking the bed every few phrases) in `lib/dj/remix.py`. The philosophy page stays; this is what its
principles look like when the unit of play is the record, not the lane.*

## The gap this closes

Tonight's nine layered sessions: 1.5 to 2.8 minutes per song, 17 to 35 lane moves per ten minutes.
A DJ: four to five minutes per record, one transition each, sixteen to sixty-four bars long, mixed at
the edges (intro over outro), heard in the headphones before the room hears it, with the stem tricks
one or two an hour. We mixed middles, in beats, thirty times an hour, unheard. Every guard added this
week compensated for that.

## The rules

1. **The unit is the record.** In layers two and three the room has ONE current record, all four stems,
   its own balance, playing whole. It plays from its IN point for at least `PLAY_MIN` (PACE: quick 150 s,
   normal 210 s, long 300 s) and never past its OUT point. At most one change is in flight.
2. **In and out points, at the edges, from the stems.** OUT = the last phrase boundary at or after
   `PLAY_MIN` where the record's drums still carry for the whole seam and its body has not ended (its outro
   start when it has one and the seam fits inside it). IN = the first bar where the record's drums carry;
   its LANDMARK stays what it is (the first groove after a build, backed by the stems). The next record is
   cued so that its landmark lands ON THE BAR THE SEAM COMPLETES: the room hears the new record's payoff
   exactly when it has fully taken over. Nothing is placed by a label the stems contradict.
3. **One seam per record, decided once.** When the next record is staged, the seam is planned: the bar of
   A (its OUT, snapped to a phrase), the bar of B (landmark − seam length), the seam's length and order.
   It is shown as the plan ("next: B in 3:10 at bar 212, a 16-bar morph") and drawn on the picture. It
   moves only when you move it (NEXT, STAY, a drag), when UP NEXT replaces B, or when the rehearsal fails.
4. **The seam is the stem morph, over phrases.** Length by MIXING × SEAMS: cut = 4-bar strip then a cut on
   the one; morph = 16 bars; blend = 32; auto = the dice between 16 and 32 (all the same technique is
   monotonous). Order, drums first by default (the literature's "a drum swap before a melody swap"):
   - bar 0 (a phrase boundary of A): B's drums in as A's drums go out, crossing over one bar; A's harmony
     and vocal carry on over B's drums;
   - the bars before the harmony swap: A's harmony DARKENS (its high shelf down over the last four bars,
     the autoDJ's fade-to-darkness) so the swap is a release, not a jump;
   - the harmony swap on the one, half way (bar 8 of 16, bar 16 of 32): A's bass and melody out, B's bass
     and melody in, crossing over two beats. HARMONY TRAVELS TOGETHER: never a melody over a foreign bass.
     A's vocal goes out here unless it fits B's harmony, in which case it may sing on until it stops;
   - after the swap B is the record; A's deck stops when its last lane is gone; B's vocal follows its own
     singing bar by bar (the vocal tick, unchanged).
   Two other orders exist for variety and for the pair: BASS FIRST (the harmony swap at bar 0 under A's
   drums, A's drums follow at the phrase: the TAKEOVER shape, the change asked for on day one) and VOCAL
   FIRST (B sings at its landmark and A is instrumental at its OUT: B's vocal enters two phrases before the
   drum swap, an acapella lead-in). Auto mixing rotates them; MIXING morph pins drums first.
5. **The rehearsal: the headphone cue.** Two phrases before the seam (at least 30 s), the seam is rendered
   offline from the stems already loaded on the decks, exactly as scheduled (gains, order, ramps, the
   shelf), in a thread, mono, decimated. It is judged on what we can measure: no bar under −40 dBFS; every
   bar within 6 dB of the bar before the seam; the vocal-over-harmony fit at the overlapped bars. A seam
   that fails moves one phrase later (twice at most), then tries another order, then falls back to the
   strip-and-cut at the phrase. The verdict is in the WHY panel and the log ("rehearsed: clean" /
   "rehearsed: moved a phrase - a dead bar at bar 6").
6. **Stems are events.** The five moments stay as they are (SLAM, BREAK, SING, TAKEOVER, BRIDGE), armed
   by you or by the DJ at the MOMENTS spacing (8 min at "some"); a moment never arms inside a seam. The
   acapella over the current record is the VOCAL FIRST seam order, or your right-click "vocals in" as
   before. Layers three: a third staged record may lend its vocal as an event; two never means two records
   swapping all the time.
7. **Safety nets kept, not steering:** the runway loop (a record whose seam failed and whose body ends
   loops its last bars, once), the bed-never-empty rule, the low carve on any song holding neither drums
   nor bass, the loudness matcher, the dropout detector.

## How the steps cross (2026-09-19: "transitions seem pretty abrupt")

Nine record changes that day, seven of them a bassline and melody on the one, loud. Three causes, three
rules:

- **The steps cross for as long as the SEAMS dial says** (THE BLEND, docs/DJ_THE_BLEND.md, 2026-09-21: "fast,
  no blending, no artistry" - two-bar steps were three switches a phrase apart). A seam is 32 bars (64 as a
  blend); quick = two-bar crosses, normal = a phrase per cross, long = two phrases; every cross is equal
  power. The harmony step is placed so its cross COMPLETES on the bar before B's landmark,
  where the switch used to happen; over the cross the old harmony's lows go down to the carve as the new
  one's come up from it (the bass handed over under a low shelf - never two full basslines, never a snap).
  The cut and the breakdown strip stay switches: a cut is a cut.
- **The DJ's own seams and events land at unity.** The bass swap's 1.25 × bass is gone; the DJ's takeover
  event lands at unity. Loud is for the moments you press yourself (SLAM, TAKEOVER, a stem brought in loud).
- **The records change through seams unless MIXING says cut.** The takeover event - the record's exit as a
  bass swap on the one - is planned only under MIXING cut. Under auto, blend or morph the plan shown is the
  plan played (a takeover exit planned three seconds after a morph was shown replaced it: "the system
  basically lies about its transitions", 2026-09-19). When a takeover exit does replace a seam, the story row
  says so ("the seam into X is dropped: the takeover with Y at its drop is this record's exit") and the log
  carries a `plan / exit` row. And a rehearsal that fails no longer turns a morph into a bass swap: the seam
  moves a phrase when the record has the body, else it becomes the cut; the shape you set is kept.
- **A level step is a level, not a shape (2026-09-21: "no fade between songs, just a bland switch").** When
  the rehearsal hears a step at the swap (the incoming harmony louder or quieter than the outgoing by more
  than the records' own steps), the judge says its sign, and the arriving stems come in trimmed by that much
  and ease back to their own level over the phrase after the cross (`step_trim`; `_lane_gain` reads it, so a
  re-post in between lands on the eased value). The morph stays a morph. Only a HOLE still moves the seam or
  falls back to the cut.
- **The placement must leave a record (2026-09-21).** The pick's runway rule reads the section map; the seam's
  placement reads the stems. Medes' bass carried nowhere the seam could use before 6:09 of its 7:33, the bass
  swap put it in there, and it was the record for thirty seconds. Now a placement that leaves the next record
  less than its minimum play plus a runway makes that song UNFIT as the next record tonight (`unfit_ids`: the
  picker, the chain and the seam skip it; the story row and an `unfit` log row say why) - unless it is YOUR
  pick, which plays as a short record, said.

## Removed

The arrival rule and the voice/bed policy (`_arrangement_move` sections 1–3), `SETTLE_PHRASES`,
`VOICE_PHRASES`, `VOICE_MAX_PHRASES`, the calm counter, `_plan_check`'s three plans, the arrival bar
search, evictions as a way of changing songs, the drum swap switch. The code of the moments, the vocal
tick, the level trims, the stem-backed positions, the flags and the picture is untouched.

## What you see

- The intent line: "Believe plays to 3:42 (2:10 more); next: Antrea, a 16-bar morph at bar 212,
  rehearsed clean" then, in the seam, "seam: Antrea's drums in · harmony swaps in 6 bars".
- The picture: the current record as four full clips, the seam as a gold bracket with its two steps, the
  next record's landmark as the white tick where the bracket ends.
- NEXT = the seam now (at the next phrase, the same order); STAY = one more phrase before it; UP NEXT
  replaces B and re-plans the seam; PACE lengthens or shortens the play; SEAMS and MIXING shape the seam.

## The gate

Play length per record ≥ `PLAY_MIN` (bar the user's NEXT), one seam per record, seam length in {4+cut,
16, 32}, harmony split 0 bars outside the operator's own brings, no dead bars, every seam rehearsed with
its verdict logged, the landmark of every record on the bar its seam completed (± 1 bar), lock p95 ≤ 0.08.

## Build order

1. The record-play engine: IN / OUT points, the seam plan per record, the morph in three orders with the
   darkening, the intent line and the picture. One piece; the gate rewritten alongside. BUILT 2026-09-13
   (lib/dj/record.py).
2. The rehearsal render and judge; the fallbacks. BUILT 2026-09-16 (lib/dj/rehearse.py: the render from
   the decks' stems at bar granularity, mono, decimated, in a thread; the hole and lurch judges from the
   campaign; the fallbacks move a phrase twice, swap the order, then the cut; `rehearsal_log`).
3. The dice on the order, the acapella lead-in from a third record (layers three), the guide. The dice
   and the guide BUILT; the third record's vocal became the EVENTS (docs/DJ_PAIRINGS.md).
