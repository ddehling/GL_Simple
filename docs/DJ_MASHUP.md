# MASHUP: three songs taken apart and played as one

*2026-10-16. "look, I want to be able to do this" (a short: "3 of my own TRACKS in one mix") → "sorry, but the
engine is boring and only does slow things" → "yes". One page.*

## What it is

A second mode beside THE RECORD PLAYS (LAYERS = **mashup**). Three songs live at once, chosen so their
tempos and keys lock and their pairing lists say their parts fit. The engine plays a ROUTINE of
combinations of their parts - a SCENE every 8 or 16 bars - instead of one record and a blend:

- a scene is who holds what: the DRUMS from one song, the HARMONY (bass + melody, always together - the
  clash rule) from one song, the VOCAL from one song or none;
- each scene changes one or two of those; the change lands on the scene's bar, quick (a beat), the way a DJ
  hits the stems on a controller;
- every few scenes a shaped one: a DROP (everything out but the vocal for the last two beats, then the new
  scene slams in on the one) or a BREAKDOWN (no drums for a scene, then the drop);
- a song coming in is cued to where its part sounds (its groove for drums and harmony, its sung stretch for
  the vocal) on the scene's bar; a song that keeps a part is never jumped;
- two vocals never sound together; the vocal comes from a song whose pairing says it sits over the harmony's
  song when the lists know, from any singing song otherwise;
- after a dozen scenes the song used least leaves and a new one comes in that fits the two that stay (the
  set moves on by itself).

Density is MOMENTS: rare = a scene every 32 bars, some = 16, lots = 8.

## You on top

THE MASHUP tab: three rows (the three songs) × four pads (drums, bass, melody, vocal), lit where the song
holds the part. A pad press moves that part to that song on the next bar (harmony travels together: bass or
melody moves both); pressing a lit pad takes it out. Keys: 1-4 the first row, Q-R the second, A-F the third.
The routine carries on from where you leave it; the next scene is always in THE PLAN with its bar.

## What stays

The beat lock, the key shift, the level trims, the low-end carve (only the song holding the bass keeps its
lows), the clock (the first song's), the moment buttons. THE RECORD PLAYS is the other setting of LAYERS.

## Proof

The mashup smoke (`tools/tests/_dj_mashup_smoke.py`): left alone for ten rendered minutes - three songs on,
a scene change every 16 bars as planned, every scene with at least two parts sounding, never two vocals, a
rotation, the pads acting on the next bar, no dead bars. The control gate and the remix gate unchanged.
