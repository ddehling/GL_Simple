# Pairings: the events come from fits, not from whoever is next

*The design agreed 2026-09-14 after the first live set on the record policy: "even when it was working it
was pretty boring and was basically being an automixer with slightly longer overlapped transitions. It also
basically refused instructions I gave it." Companion to `DJ_THE_RECORD_PLAYS.md` (the records and the
seams) and `DJ_MOMENTS_DESIGN.md` (the composed shapes).*

## What this is for

The records and their seams are the plain part of a layered night; they are an automixer with stem seams
and they should stay unremarkable. What makes layered mode worth having is the EVENT: a vocal from another
record over this groove, another record's bassline swapping under these drums, a break, a drop. Tonight's
events came from whatever song happened to be next, so they clashed and were pushed out to a few an hour.
From now on they come from a list computed once, offline, from the stems: for every record in the pool,
the other songs whose parts sit on it. The DJ arms one event per record from that list at the MOMENTS
rate; you see it coming and can veto or pick another; your own picks are orders.

## What is measured, per song, once (`tools/dj/dj_pairings.py --envelopes`)

From the four stems, stored as `.stems/<id>/env.npz` (2 Hz envelopes, each stem against its own loud
level - the same envelopes the conductor reads live):

- **sung stretches**: where the vocal sings for eight bars or more (short gaps merged), and whether the
  measured hook falls inside one;
- **instrumental grooves**: stretches of sixteen bars or more where the song is a groove by its map, its
  drums and bass carry, and its vocal is silent - the places another vocal can sit;
- **the bass from the landmark**: whether the bassline carries for sixteen bars from the song's landmark
  (the first drop) and how much body follows it - what a bass swap needs;
- key, tempo, the landmark, the body start.

## The pairings (`--pairs`, `.stems/pairings.json`)

For every record R, two ranked lists over the whole pool:

- **VOCAL OVER**: songs S with a sung stretch that fits one of R's instrumental grooves. Key fit ≥ 0.68
  over shifts of at most two semitones (each semitone costs a little), tempo within ±6 % (a vocal shifted
  further than that sounds wrong), the stretch at least eight bars, the hook inside it a bonus. Score =
  key fit × tempo closeness × stretch length × hook bonus. Never a second singer: R must be instrumental
  where S sings.
- **BASS UNDER**: songs S whose bass and melody swap under R's drums and then become the record (the
  TAKEOVER shape): key fit ≥ 0.68, tempo within the wall, the bass carrying sixteen bars from S's landmark,
  the minimum play left after it.

The lists are cheap to compute once the per-song measures exist (a key and tempo filter over the pool,
then arithmetic on the stretches), and are rebuilt by the planner's analysis stage when stems are added.

## How the night uses them

- **The DJ's events** (the MOMENTS dial: rare = none, some = one per record, lots = two). During a record's
  play, after its first phrase and clear of its seam, the conductor arms an event from the record's list:
  a VOCAL OVER becomes the new **ACAPELLA** moment - S staged, its sung stretch cued to begin on the next
  phrase over R's instrumental groove, in for the stretch (two to four phrases), out when the stretch ends
  or when R sings again; S leaves, unless it is UP NEXT, in which case its seam follows. A BASS UNDER becomes
  the TAKEOVER moment with S, which then IS the next record (its seam replaced by the moment). The picture
  draws the event in gold and the intent line says "event in 0:40: the vocal of X over this groove (fit
  0.81)".
- **Your picks are orders.** SONGS shows, for the current record, "sits over it" and "swaps under it" with
  their fits, and the Song parts window shows the same. Right-click as before (that part in now, TAKEOVER or
  BRIDGE with this, UP NEXT): a staged song is cued to where its part sounds whatever its cursor says, a key
  clash is said and not obeyed, a running seam gives way to your moment, and the only refusal left is a song
  that is not on a deck.
- **Harmony travels together** still: a VOCAL OVER never brings S's melody or bass; a BASS UNDER brings S's
  bass and melody together and takes R's harmony out on the same bar.

## What you see and hear

A record plays; forty seconds in, another record's vocal arrives over its groove for two phrases and goes;
the record plays on; its seam comes at its section boundary. The next record's event is a bassline swapped
under its drums at its drop. Two or three such things per record at "lots", one at "some", none at "rare".

## Build order

1. `dj_pairings.py`: the envelope pass (parallel, resumable), the per-song measures, the pairing lists, a
   `--show <song>` that prints a record's lists in words; the analysis stage calls it.
2. The ACAPELLA moment; the DJ arming one event per record from the lists; the SONGS list's fits section
   and the Song parts window's words.
3. The rehearsal render (`DJ_THE_RECORD_PLAYS.md`, step 2) judging each event and each seam before it plays.
