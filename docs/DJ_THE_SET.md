# The set: the plan reaches five records ahead, the clock travels, a request becomes playable

*2026-09-19. Asked for as "do the planning horizon and floor work" after the gap list of that day. Two of the
listed gaps were wrong and are not built again: the deck already transposes independently of tempo (two
semitones, `Deck.set_pitch`, chosen by `_key_fit`), and the Director already has its nanoKONTROL2 mapping
(`tools/dj/planner/director.py`, `MIDI_FADERS` / `MIDI_KNOBS`). The rest was real.*

## The gap this closes

- The layered engine plans one seam ahead. THE PLAN ends at "RECORD: X plays · its seam is planned when its
  next record is staged" and UP NEXT. The one-song engine has had a six-deep horizon for a year
  (`Brain.plan_horizon`, `DJSystem._maybe_horizon`); the record engine has none.
- The clock never leaves the opener. `base_bpm` is set once, at the first record; `_tempo_step` drifts at
  most ±6 % around it all night. A set that opens at 118 cannot reach 128, whatever the records.
- A queued song outside the tempo wall is dropped from UP NEXT one tick later with a line in the log. The menu
  hides the action, so this only bites from the web panel or a chain, but the honest answer would be a route.
- A song without stems has no path to the decks during a night: `rank` greys it "no stems", and the only way
  to render them is the Library tab.

## The rules

1. **THE SET is a chain, not a star.** After the record and whatever you queued, the Director holds up to
   five records more (`horizon`), each chosen against the one before it: key fit at 0.55 or better with the
   deck's two-semitone shifts, tempo inside the wall against the PROJECTED clock (rule 3), the energy the arc
   asks for at the projected time, stems present, the minimum play and a runway left, nothing played tonight,
   and the pairing file's word on how it can arrive (drums first, the bass swap, else the cut). A beam search
   of width four, seeded, so the same room gives the same chain. `lib/dj/setplan.py`, pure.
2. **The chain does not churn.** A hop changes only when it becomes impossible (played, its predecessor
   changed, the wall, you said "not tonight") or when you change the head of the queue; a dial turn re-plans
   from the first hop that no longer fits, and keeps the ones that still do. THE PLAN shows each hop as a
   `THEN` row with its projected time, tempo step, key, energy word and arrival. The DJ's spare pick IS the
   first hop (`rc.pick_hook`): the live pick follows the displayed plan, as the one-song engine's does.
3. **The clock follows the record.** When a record takes the clock, `base_bpm` becomes ITS native tempo and
   the clock steps toward it half a percent a bar (sixteen bars for an eight-percent step), every deck
   re-rated together, never past the wall for any deck. A record that entered at 0.93 is at its own tempo a
   phrase and a half later, and the next hop is measured from there. The arc's span and the TEMPO dial's lean
   still sit on top. This is what makes a tempo journey possible at all.
4. **A far song is a route, not a refusal.** Queue a song outside the wall and the Director looks for a chain
   of at most three records, each hop inside the wall and fitting in key, that ends at it; found, the
   intermediates go into UP NEXT ahead of it, marked "via", and the story row says "to reach X (132 bpm)
   from 120: via A (125), B (129)". Not found, it says so with the distance. The menu offers "UP NEXT via N
   records" instead of nothing; SONGS shows "via 2" instead of grey.
5. **A request becomes playable during the night.** A song in the library without stems offers "make it
   playable" in its menu: its stems render in the planner's one-slot GPU queue, the pairing lists are rebuilt
   for it, the Director learns the track without a restart. A file dropped into the music folder during the
   night is found within twenty seconds and goes through the same path (analysis, stems, pairings). Each
   request is a `REQUEST` row at the top of THE PLAN with its stage; when it is ready the row says so for a
   minute and the song is in SONGS as a ✓. One job at a time: demucs owns the GPU, and the player is in the
   same process.

## What you see

- THE PLAN, after UP NEXT: `THEN  12:40  Antrea · 124 (+2 %) · 8A 0.85 · same energy · drums first`, five
  rows at most, then `CLOCK  122 now → 126 by 12:40 → 128 by 25:00` when the chain moves the tempo.
- Right-click a THEN row: put it UP NEXT · not tonight · show its parts.
- A REQUEST row: `REQUEST  Antrea: separating (40 %) → pairings → ready in about a minute`, then
  `REQUEST  Antrea is ready · put it UP NEXT?`.
- In SONGS, a walled song reachable by a route reads `via 2` instead of grey.

## The gate

The control gate presses `queue_far` (a song outside the wall: a route said and queued, or an honest
refusal naming the distance) and `set_skip` (a THEN row's "not tonight": the hop leaves and does not come
back); after every press in layered mode with a record past its first phrase the plan has THEN rows; the
head of the chain may change only on a trigger (a record change, a queue change, a skip, a dial) — a change
without one is BROKEN. The clock's travel is measured by the campaign: the records' native tempo against the
clock at each seam (`clock_off_pct`), and the wall is never crossed (`_tempo_step`'s rule, already there).
The floor path is smoked headless on a temporary music root with one file (`dj_scan.py --add`), never on the
library; demucs is not run by any gate.

## Build order

1. The clock follows the record (`remix.py`: `follow_clock`, the two handovers, `_tempo_step`).
2. `lib/dj/setplan.py`: the pair score, the chain, the route.
3. The Director: `horizon`, `set_skips`, the triggers, `pick_hook`, `queue` with a route, `plan_rows` THEN and
   CLOCK rows, `refresh_track`.
4. The planner tab: THEN rows and their menu, the walled menu's route, `via` in SONGS, "make it playable",
   the drop box, REQUEST rows.
5. `dj_scan.py --add`, the gate cases, the docs.
