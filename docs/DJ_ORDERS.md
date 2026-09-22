# Orders: why commands keep failing, and the fix

*2026-09-20. Written after the Resurrection pair missed its groove - the fourth pair failure in a day, each
"fixed" and each followed by another. One page. "Do it all" the same evening; the status is at the end.*

## Why these things keep failing

An order (PLAY THIS PAIR, TAKEOVER with X, UP NEXT, a stem in) is not executed. It is stored as a hint -
`_planted`, `_pending_moment`, `up_next` - and then four or five independent planners, later, each decide
again what to do with it, each by its own rules:

| step | who decides | its own rule | how it broke an order |
|---|---|---|---|
| where the record starts | `pair_record_start` (geometry, sections only) | five phrases before the groove, or a cut two phrases before | said "starts at 0:31" in PAIRS and THE PLAN |
| where the record actually enters | `_seam_b_cut` / `_seam_b_start` (stem envelopes) | drums AND bass must carry two phrases | Resurrection's intro has no bass: entered ON the groove |
| when the vocal comes | `_event_step` | a phrase boundary, one phrase of margin, eight bars of groove left | the only phrase that fit was inside the margin |
| whether it may come | `arm_moment` | the record must not sing there (by the stem) | nine shown IDEAs refused at their bar (09-18) |
| whether the song is there | `stage_track` | a free deck, the wall, the string of a refusal | UP NEXT popped when decks were full |

Three structural faults, not five bugs:

1. **Three models of one fact.** Where the record starts is computed three times by three pieces of code
   with three inputs. They agree on easy songs and disagree on the rest. Every fix so far aligned two of
   them for one song.
2. **The DJ's taste rules judge the user's orders.** "Drums and bass must carry", "events land on phrases",
   "one phrase of margin", "no vocal over a singer" were written for the DJ's OWN choices, where refusing is
   free. Applied to an order, a refusal minutes later is the failure the user sees.
3. **The display renders state, not the plan.** THE PLAN and the picture show a row only once some pending
   object exists. An order has no object until its record plays, so the plan shows "the opening part" and
   nothing else - and what it shows is computed by yet another piece of code (row 1 above).

And one fault of method: each fix was verified on the pair at hand (the best-scoring pair in the smoke), not
on the class. The smoke never held a first-groove pair whose intro carries no bass.

## The fix: an order becomes a SCORE, computed once, executed verbatim, displayed verbatim

**A score** is one object per order: every concrete decision, in bars, decided at take time and never
re-decided. For a pair:

```
score = {record R, start_t (R's own time), entry ("cut"|"drums"), seam_bar K_seam (once the current
         record's seam exists), event_bar K_ev, bars, S's cue_t, state, why}
```

- **Computed once, from the real inputs.** PLAY THIS PAIR stages R and S at once (they are the inputs). When
  R's stems are decoded (a few seconds), `plan_pair` computes `start_t` with the same `_stem_carries` the
  seam will use, `K_ev` from `start_t` and the groove, and checks that eight bars fit. If they do not, the
  order is **refused then, said, by name** ("Resurrection cannot hold this pair: its groove is 16 bars and its
  drums carry only from 0:55") and R leaves the queue. Until then the row reads "planning (its stems)". There
  is no fourth outcome.
- **Executed verbatim.** The seam into R reads `score.start_t` and `score.entry`; it searches nothing. The
  event reads `score.K_ev` relative to R's first bar; it searches no window. The arm takes the score's bar. If
  a step CANNOT execute (the deck vanished), the score is marked failed with the reason, said - never silently
  re-planned.
- **The DJ's taste rules do not apply to a score.** Carry checks, phrase margins, "no vocal over a singer",
  runway floors: the DJ's own. A score is what the user asked for; the judge may say it, not obey it (as `bring`
  already does for a key clash).
- **Displayed verbatim.** THE PLAN's rows and the picture's marks for an order come from the score object -
  its bars, its state - not from a re-computation. One code path writes the row; the same object drives play.
  What is shown is by construction what is played.

**The same shape for the other orders**: TAKEOVER / BRIDGE with X (a score with X's landing bar, computed
when X is staged, refused then or fixed), UP NEXT (a score with the seam bar once placed; a full deck WAITS,
said), a stem in (already close: staged, then `bring`, refused only when not on a deck).

## How it is proved

- **The pair census** (`tools/tests/_dj_pair_census.py`, new): for every pair the PAIRS list would show, run
  `plan_pair` against the real stems (no audio, decode only) and count: feasible / refused, with reasons. The
  list then shows only the feasible ones - the list is derived from the planner that executes, not from a
  geometry stand-in. Target: 0 pairs shown that the planner refuses.
- **The pair smoke** gains `--record --part` (done today) and a set of NAMED cases run every time: a
  first-groove pair with a bassless intro (Resurrection / The White City), a pair on a late groove, a pair
  whose record is the record playing, a bass pair. Each must land on its scored bar, within a bar.
- **The control gate** checks, after every order, that a score exists within two bars and that its rows are
  in THE PLAN, and at the end that every score either ran on its bar or was refused with a reason.

## Cost and order of work

1. `Score` object + `plan_pair` + refusal at take time, the seam and event reading it: a day.
2. THE PLAN and the picture from scores: half a day.
3. The pair census and the named smoke cases: half a day.
4. TAKEOVER / BRIDGE / UP NEXT as scores: a day.

Nothing of today's patches is undone by this; they become the DJ's own rules again, and the scores go
around them.

## Status (2026-09-20, evening)

**Built.** `lib/dj/orders.py` (the score: `new_order`, `plan_pair_entry` - pure, shared with the census).
`RecordPlay.place_order` / `_orders_step` / `_order_event_step` / `_order_end` / `orders_status` in
lib/dj/record.py: a pair order is "staging" until its record's stems are on a deck, then planned ONCE
(fixed with entry, entry time, event time, bars, the other song's cue) or refused by name; the seam into
the record reads the score (`_planted_start` → `_seam_b_cut` / `_seam_b_start` / `_plan_seam` search
nothing for a scored record); the event bar is geometry from the score's event time and where the record is
(no phrase grid, no window); a seam in the way MOVES past the pair; the arm lets a scored acapella through
where the record's stem sings (said, not refused); every end is a said state (done / refused / failed +
reason) through the Director's `_on_order` hook, and a refused pair's record leaves UP NEXT. THE PLAN's PAIR
rows and the picture's marks come from `orders_status()` - the object itself. TAKEOVER / BRIDGE with a song
now have a MOMENT row while staging. The pair census writes `.stems/pair_plans.json`; the PAIRS list hides
what it refused and says the entry it decided. The control gate presses PLAY THIS PAIR and checks at the end
that no order died silently; the pair smoke fails on a refused or failed order and has the named cases.

**2026-09-21, the morning after:** "the plan does not match the timeline" - the next-record ghost drew the
first staged song, not the seam's B; the PLAYING list did not redraw on a state word; a 0:00 groove said
0:00 for an entry that is heard at 0:12 - all three fixed, the smoke checks the ghost's identity. Then "the
vocals from another song last a pathetic amount of time" (8 and 16 bars live, then "ACAPELLA over: the vocal
goes" and its song off the deck): the acapella script's two-to-four-phrases-and-gone was the DJ's rule for
its own idea, applied to the user's choice. A vocal YOU chose (a pair, or an idea taken with "do it") now
holds the lane under the vocal hold's rules - until it has sung its last, or you take it out; the moment ends
a bar after the vocal is in and the hold carries it. The smoke measures the hold for six phrases.

**Same morning, "the system refused a suggested pair":** Raw's groove at 0:00, 16 bars - the list said "enters
at 0:00" from the census of the night before; the planner, changed that morning (a cut is heard six bars in at
the earliest), refused it at take time: the census was stale against its own planner. Now the census verdicts
carry `PLAN_VERSION` and a planner of another version ignores them, and the list's own geometry applies the
same floor for a groove at the record's very start (door 3 is the only door there, so the list can be exact
without the stems). Lesson: a census is only as true as the planner that wrote it - bump the version and re-run
on every planner change, the same day. Remaining gap: a pair on a record's first bars can only begin ten bars
in; beginning the vocal ON the cut itself would need the acapella armed across the seam.

**Same day, the evening's flags:** "this pair sucked" / "the vocals in the pair are too faint to hear mostly" -
God Sigma's vocal at unity (a stem against another song is never trimmed up; its stem is quiet by the
library's measure) under Duskraver's bed at 0.86, and the hold RESTING the lane in every four-bar pause and
fading it back on the first detected beat: eight rests and returns in three minutes, the head of every phrase
lost. Now a chosen vocal is brought to the library's vocal level (up to +6 dB) and its lane stays through its
pauses. "bad transition / seam" - the score chose a cut into My Darkside at 2:17 because door 1 tried drums
first at exactly five phrases before the groove (2:32, where the drums do not carry) and never at 2:17 where
they do; door 1 now tries five, four, six, three, seven and eight phrases before it (PLAN_VERSION 3, census
re-run).

**Not yet.** UP NEXT and TAKEOVER / BRIDGE as full scores (they refuse at once for what is known, wait for a
deck, and show a row; their landing is still the arm's decision when the song is ready). The census is run by
hand, not by the pairings pass. The bass pair (TAKEOVER at the other song's drop) keeps its own placement.
