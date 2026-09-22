# The composer: every record gets a score

*2026-09-21, night, after the blend: "We are not making a system that really does anything interesting. It
slowly and ponderously plays songs with slight overlap. This is not a good system." One page, for a yes.
Nothing here is built.*

## What plays tonight, honestly

The campaign on the blend (eight nights): 4.4 minutes per record, six minutes of one song alone at the worst,
eight seams in ninety-six minutes, the DJ's own events zero. The seams are smooth now (the level step across a
seam is a third of a decibel) and the records are whole. It is a playlist mixer with a long crossfade.

Two rules I wrote to end the clashes and the churn also ended the play: **"the record plays whole"** and
**"one idea per record, only if you say do it"**. Every interesting thing the system can do exists - a vocal
from another record over a groove (the pair census: 40 of 40 offered pairs playable), a bassline taking over
under the drums, a loop tightening into a drop, the next record's vocal as a lead-in, a SLAM, a BREAK - as an
optional side order in a tab or a button, waiting for a yes, landing a phrase or two later. None of it is the
music's grain.

## What "interesting" means, in numbers

Per record, two to four things happen that a listener notices, placed where the record's structure invites
them, and the exit is a payoff rather than a fade:

| when | what | from |
|---|---|---|
| the record's first groove (after a phrase of it alone) | a guest VOCAL from its pairing list, held until it has sung its last | the acapella order, as built |
| its second groove or its breakdown | a BASSLINE SWAP under its drums for a phrase or two, then back - or on to the takeover | the takeover order, with a return |
| the bars before its own drop | a LOOP tightening (8 → 4 → 2 bars), the hole, the drop on the one | BUILD, as built |
| its exit | the next record's drop as a TAKEOVER, or its vocal as the lead-in, the blend only when neither has a place | the seam's orders |

Density is the MOMENTS dial: rare = one event, some = two, lots = four. Records get shorter when they are
full: the minimum play counts events (a record with three events may move on at two and a half minutes), so
PACE means how dense the night is, not only how long each record runs.

## The composer

1. **A score per record, composed when it is staged.** From the record's pairing lists (vocal over, bass
   under), its section map (grooves, breakdowns, its drop) and the next record's entry, the composer places
   the events on its grooves and its drop, checks each with the executor's own planner (`plan_pair_entry`,
   the arm's rules) against the stems - the census showed that works - and writes the score. THE PLAN shows
   the record's events as rows the moment it is staged; the picture shows their marks. What is shown is what
   plays: the orders machinery executes each event verbatim, as PLAY THIS PAIR does today.
2. **The DJ takes its own ideas.** An event in the score fires on its bar. "Ask me first" becomes a setting
   (the IDEA rows come back for those who want them); the default is a DJ who plays.
3. **The exit is a payoff by default.** When the next record has a drop that can land on a phrase of this
   one, the seam is a TAKEOVER at that drop (its bassline under these drums, its drums on the next phrase - as
   built, but the seam's default, not a button); when it sings before its drums, the acapella lead-in; else
   the blend.
4. **The same treatment for the buttons.** MOMENTS and READY become scores too: pressed, they land within a
   phrase, or refuse at once by name. Nothing waits on a decode without a row in THE PLAN.
5. **Variety is composed, not rolled.** No two records in a row get the same score shape; a night alternates
   vocal records, bass records, build records; the arc dial leans which.

## What stays

The record is still the spine (its drums and bass are the room's unless an event says otherwise); harmony
travels together; a vocal you bring holds; the blend is the fallback seam; the guards say, they do not refuse,
for anything the user asked; nothing is synthesized.

## How it is proved

- **The composer census**: over the library, how many records get a full score of two, three, four events
  from their own lists - the number the MOMENTS dial can actually deliver.
- **The control gate** as now, plus: every scored event either fires on its bar or is refused with a reason,
  none silently.
- **The campaign** with new columns: events per record, minutes per record, seconds of one song alone, and
  the exit kinds (takeover / lead-in / blend). Target: two or more events per record at MOMENTS some, under
  three minutes per record, no stretch alone over two minutes.
- **The ear**: the rendered seams (HEAR THE SEAM, the flag wavs) and the user's hour.

## Cost, in order

1. The composer (the score per record from lists + structure, the plan rows, the executor hooks for the
   return of a bass swap and the payoff exit): two days.
2. The census and the campaign columns: half a day.
3. The buttons as scores: a day.
