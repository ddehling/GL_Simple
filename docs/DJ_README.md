# Autonomous DJ Subsystem

Scans a music library offline, understands each track's structure, and
mixes real DJ sets all night — beat-matched, time-stretched (constant
pitch), EQ-blended, loop-rolled — fully autonomously or following a
preplanned setlist. The mix feeds the analyzer's `internal` source, so
every audio-reactive shader dances to what the DJ is actually playing,
and the DJ's planned arc drives the club set's night phase.

## Quick start

```bash
# 1. Put music in <repo_parent>/music (or pass --dir anywhere below)
# 2. Analyze the library - the FULL pipeline (scan, chroma, rhythm,
#    vocals, enrich, mood, structure; every stage incremental), headless:
python tools/dj/dj_analyze.py            # add --stems for the stem render, --instruments for the per-song instrument pass
#    (same stage list as the planner's "Analyze all" - lib/dj/analyze.py)
#    or just the scanner: python tools/dj/dj_scan.py
#    add --refine-grids to also re-run tracks whose beat-grid confidence
#    sits below 0.75 (vote-weighted confidence can promote them back into
#    the precision transition styles)

# 3a. Hear it standalone:
python tools/dj/dj_player.py --live --theme groove
# 3b. ...or in the show: open the web panel -> DJ tab -> START THE DJ.

# Plan a set for tonight (native desktop app; audition every seam):
python tools/dj_planner.py

# Read back what the DJ actually did last night:
python tools/dj/dj_review.py --all
```

## Pieces

| Piece | What it does |
|---|---|
| `tools/dj/dj_scan.py` | Incremental scanner → `music/dj_library.sqlite3` (BPM + ms-accurate beat grid, downbeats, Camelot key, structure sections with busyness/vocalness, loops, mix points, loudness, live-pipeline cross-check). Finishes with a **beat-grid health line** — the loose-grid tail is the ceiling on everything the DJ can do, so it is now stated out loud (`--refine-grids` is the fix). `--retag` re-derives the library-relative auto tags from stored analysis in seconds, no audio decode |
| `tools/dj/dj_review.py` | **Reads the DJ's own night logs** (`logs/dj_*.jsonl`) — night summary, style/verdict report, `--terms` selection-term validation against what each seam measured, `--gates` why a technique never reached the dice, `--skips` what the operator kept rejecting. The data layer lives in `lib/dj/review.py`, and the planner reads the SAME evidence: the **Nights tab** (per-night play-by-play + per-seam verdicts) and the Set tab's night badges ("✖ flammed live 07-12" on a pairing that measured rough) |
| `tools/dj/dj_analyze.py` | **Headless "Analyze all"** — the planner's full 8-stage pipeline as a CLI (`--dir`, `--stems`, `--only <stage>`, `--list`), for overnight/scripted analysis with no Qt window. One shared stage list (`lib/dj/analyze.py`) so the GUI and CLI can't drift; the Library tab also shows a per-pass **coverage line** ("gaps: mood 310/430 · stems 0/430") so "is my library ready?" has an answer in the UI |
| `lib/dj/` | `features` analysis · `db` library · `rb_stretch` **Rubber Band R3 keylock — the DEFAULT tempo engine** (2026-07-22, picked by ear: constant pitch, warble-free, enables the ±1-semitone key rescue; needs `pip install -r requirements-dj-keylock.txt`, otherwise the engine resolves to varispeed automatically) · `varispeed` turntable tempo engine (pitch rides tempo, zero stretch artifacts; the brain bends BOTH decks to a meeting tempo so each song shifts half as far — under a fifth of a semitone on a typical seam) · `stretch` WSOLA keylock / `pv_stretch` phase-vocoder keylock / `rb_stretch` **Rubber Band keylock** (the Mixxx-grade library via `pylibrb` — wheels for Windows/Linux x86_64/macOS, `pip install -r requirements-dj-keylock.txt`; R2 "faster" engine by default — measured onset-preserving at every DJ rate at ~2% CPU, R3 via `DJ_RB_ENGINE=finer` trades attack crispness for tonal smoothness; falls back to varispeed with a warning when the wheel is missing, and the brain's planning semantics follow the *resolved* engine so transpose-rescue/dual-bend decisions always match the decks) (keylock opt-in via `DJ_STRETCH_ENGINE=wsola\|pv\|rubberband`) · `eq` LR4 3-band · `deck`/`submix` playback (ONE engine track, sample-accurate automation, sync PLL) · `brain` selection + transition planning · `themes` arcs · `setlist` compiler · `system` conductor |
| `tools/dj/dj_player.py` | Standalone: `--live`, `--wav out.wav --minutes N`, `--audition A B` (render one seam), `--file X --rate r`, `--setlist NAME` |
| MusicBrainz enrichment | Built into the planner's **Library tab** — the "Enrich (MusicBrainz)" button pulls genre, release year/era, label and canonical identity (free, live, no key) for every track that lacks it, in a background thread with live progress; genres + decade fold into `TrackInfo.all_tags` and appear in the tag browser as they land, steering selection, flavor, and the Set Copilot. Stored per track (DB v9), incremental + resumable, ~1 track/sec. (Spotify audio-features and AcousticBrainz APIs are both dead as of 2024/2022 — MusicBrainz is the durable open source; local acoustic descriptors are the **Mood (ML)** pass below.) `tools/dj/dj_enrich.py` remains as an optional CLI (`--limit/--force/--stats`) but the GUI needs no scripts. |
| Mood (ML) descriptors | Built into the planner's **Library tab** — the "Mood (ML)" button runs the [Music2Emo](https://github.com/AMAAI-Lab/Music2Emotion) model (PyTorch, same torch+CUDA stack as the vocal pass) over each track and stores real **valence/arousal** (0..1) + **mood tags** (dark, party, epic, melancholic…). `character.py` PREFERS these over its heuristic derivation, so `danceable`/`dark`/`uplifting` tags, valence steering and mood vocabulary in the tag browser + Set Copilot all become ML-grounded. This is the local-acoustic-descriptor path Essentia can't provide on Windows (no wheels); Music2Emo runs natively. Stored per track (DB v10), incremental + resumable, ~3-5 s/track on GPU. Needs a Music2Emotion clone + `requirements-dj-mood.txt` (see that file); `tools/dj/dj_mood.py` (`--limit/--force/--stats/--model-dir`) is the optional CLI. The scanner also now keeps each file's embedded **genre** ID3 tag (free, `file_genre`) as another `all_tags` source. **The live autoDJ + set generation actively USE this** once the library is ≥80% mood-scored: valence continuity at seams (don't cut dark→bubbly), an arousal-blended energy arc, and per-theme danceability targets (`Theme.dance_target`) + ML-mood `prefer_tags`. Below 80% coverage the steering stays OFF (a partially-scored library would bias the scored few against the unscored majority), so it's all-or-nothing. |
| Beat power | **Does the track actually THUMP on its own beats?** (`lib/dj/beatpower.py`, `logs/beat_power.json`). Grid *confidence* measures whether a periodic lattice fits the audio; beat power measures whether low-band **attack energy lands ON that lattice** (on-beat vs off-beat attack peaks, ~30s at the track midpoint, density-neutral). Measured 2026-08-04: **38% of the library scores below 1.2** — confident grids over diffuse grooves (organic percussion, rolling basslines) — and beat-matching those is matching air: sample-perfect sync, audible mess (one pair with stored kick-agreement 0.99 rendered at 3.2× kick density). **Overlapped-drum styles require BOTH sides ≥ 1.5**; diffuse tracks keep the dipped fade, clean cuts and acapella paths. Scan CLI: `python -m lib.dj.beatpower --music <dir>` (incremental, resumable, below-normal priority). Unmeasured tracks pass until scored. |
| Grid-phase profile | **Where do the audible kicks sit relative to the stored grid?** (`lib/dj/beatpower.py --phase`, same `logs/beat_power.json`). The 2026-08-04 discovery that ended the beat-match hunt: stored grids are periodically right (high confidence) but their **phase misses the real kicks by ~48ms median** in seam regions, with signs differing per track — so grid-to-grid sync aligns *lattices* while the ear hears flam (gates showed 5ms "lock" over audible double beats), and cuts scheduled on grid beats land mid-flam. The scan measures every grid beat's low-band attack-peak offset and buckets it **every 20s along the whole track** (phase is a LOCAL property — a 3-region first cut immediately missed seams landing 60-100s from a measurement). Consumed in three places, all from ONE lookup per seam: `build_events` shifts `out_s`/`in_s` from lattice time to music time (fixes cut placement), computes the **kick bias** (B-offset − A-offset, in beats) and ships it inside the `sync` events; the submix snap + PLL hold that biased target (kicks in register, grids intentionally apart). Zeroed for live tempo-fixed grids (offsets don't apply to a re-anchored grid; write-back invalidates the track's record for rescan). Validated end-to-end by rendering seams and measuring kick-to-kick in the audio. Scan CLI: `python -m lib.dj.beatpower --music <dir> --phase`. Unmeasured tracks fall back to plain grid sync. |
| Rhythm signatures | Beat-sync GROOVE fingerprint per track (DB v13, `lib/dj/rhythm.py`, sig v2): kick/snare/hat 16th-note step patterns folded over 2 bars, one-beat fine folds, **swing** (0.50 straight → 0.67 shuffle, onset-latency corrected), **density**, **meter** (3/4 vs 4/4, beat-level accent contrast — only a confident 3/4 claim counts) and **region patterns** (~48s folds at the primary mix-in/mix-out, so a seam compares A's EXIT pattern against B's INTRO pattern — the material the blend actually overlaps — not two whole-track averages). Computed inline on scan (mix-derived); the planner's **"Rhythm"** pass (`tools/dj/dj_rhythm.py`, in "Analyze all" after stems) backfills old libraries, upgrades v1→v2 formats, and upgrades to **drum-stem-derived** where stems exist. Powers: the seam **chips** (word-first limiting factor — `kick clash`, `swung vs straight`, `3/4 vs 4/4`, `half-time`, `flam risk 40ms`, `stretch +5.8%`; a trailing `?` = shaky grids), compile warnings, the **seam inspector** (click a ↳ seam: both region step grids aligned at the planned tempo-read, low-band contradictions marked, one-beat flam microscope), the library **rhythm column** (kick-pattern glyph, sorts by density), arc-strip tooltips, the Copilot's seam explanations, the web `/dj` armed-seam chips, and the LIVE engine (see below). |
| Per-song instruments | **Which sounds is THIS song built from, and what does each play on every beat?** (`lib/dj/instruments.py`, `tools/dj/dj_instruments.py`, result in `.stems/<id>/instruments.json`, versioned). Not a fixed taxonomy: inside each stem the recording's own sounds are discovered. **Drums** — every onset gets a timbre vector (attack + body mel spectra, decay, centroid), the onsets cluster, and the clusters' stacked-in-time templates are explained by each other with NNLS that may not overshoot (a superset never explains its own subset), so a cluster that is only a **coincidence** of other sounds (a kick under a clap) credits its hits to each component at near-full level and goes, a same-sound split (a hat at two velocities) merges, and a coincidence with something of its own keeps the residual — no whole-stem NMF peak picking, which split look-alike sounds arbitrarily. **Pitched stems** (bass/other/vocals) — plucked/struck onsets on the percussive (HPSS) part get a pitch by harmonic summation with everything already sounding **subtracted** (a pluck over a pad reads as its own note) and cluster on their **harmonic profile** (pitch-invariant, raw dB, a split needs ≥ 6 dB rms between profiles), so one instrument stays one instrument whatever line it plays; unpitched onsets cluster on mel timbre. **Held notes** — the harmonic part beat by beat: a note counts only if prominent through BOTH halves of the beat (a pluck's tail is loud then gone; a pad holds), the beat must be TONAL (peak/median salience ≥ 4 — reverb, room and percussion residue read as flat salience; a real pad measures 15+), and a note that a pitched hit instrument STRUCK and is still sounding belongs to that hit, not to a pad (v2: before this a lead's long notes were reported twice and the pad row swallowed them); the remaining beats cluster by MFCC envelope + register + polyphony into sustained instruments (pad vs lead by polyphony), up to 3 notes per beat in `other`. Every instrument carries a measured description (register · envelope · note range · hits · level vs the stem's loudest), a role **hint** (kick? pad? — a reading aid only), an **exemplar** cut, and events `[beat, 16th step, midi, duration, velocity, confidence]` on the DJ grid's beats (main segment extrapolated over the track, so beat indices are the ones the decks count). Gate: `tools/tests/_dj_instruments_test.py` — a synthesised 32-bar song with known parts (kick/clap/hat/shaker, a bass line, a melody on 8ths, a pad changing triads): kick/clap exact, bass and melody lines 100% of notes, pad chord root on 97% of beats; hat vs shaker (written with the SAME noise spectrum, only envelopes differ) tolerates a quarter leaking. Known limits: sub-kick bleed in the bass stem reads as a low pitched sound at the range floor (C1); spectrally identical sounds are separated only by envelope. ~16 s of CPU per minute of audio (measured over a library sample: a 5-minute track ≈ 80 s); `dj_instruments.py --show <id>` prints the palette and a few beats. Opt-in in "Analyze all" (**+ instruments**) and `dj_analyze.py --instruments`. **Reconstruction and the gen bridge** (`lib/dj/resynth.py`): `render(result, stems, ids)` plays any set of instruments back from their notes with the instrument's own sampled exemplar (repitched for pitched sounds, looped for held ones, restored to the level its loud hits had via `level_dbfs`, each event at its read velocity) on the DJ grid — the Analysis tab's **hear: reconstruction** source plays the SongProgram render of it (below) behind the same S / M mixer as the stems, and it is the honest check of a reading: a spurious note is a wrong note there; `dj_instruments.py --render <id> --ids a,b --out x.wav` does it headless. **Levels artifact found 2026-09-09** (the "garbage" the reconstruction used to return in the planner): a pitched voice's per-pitch samples are 'un-velocitied' by up to 26 dB and then played at event gains up to +17 dB, and on real tracks one `other` voice came out 15-20 dB hotter than its stem (raw peak 6.5 against a stem peak of 0.93 on Deacon Blue) — and `songprogram.render` scaled the WHOLE track by its single loudest sample, so that one voice took the reconstruction down 15 dB and everything else vanished under it. The planner now renders each voice raw (`render(limit=None)`), caps a voice at its stem's level inside its own event windows (`songprogram.voice_ceiling_db`, a one-sided correction: a part cannot be louder than the stem it came from), and limits the SUM with a look-ahead peak limiter (attack 5 ms, release 80 ms) instead of scaling it. The evaluator still uses the whole-track clamp; its "level" column on tracks with such a voice is that clamp, not the voices — a lever to measure. `export_gen(...)` (tab button **→ Gen**, CLI `--export-gen <id>`) writes the same material in the generative console's SongScript form under `logs/analysis/<title>/` — kit one-shots per drum slot with `kit_db`, per-pitch note banks for the lead, keys and bass sounds, the pad sample with its base note, per-bar `drums_bars`, `melody` and `bass_line` per section, sections from the DJ analysis — so the gen console's Analysis tab (Open) recreates and varies the song from it (verified: the exported script renders through `lib/gen/script.render`). The two consoles are linked both ways through `lib/dj/gen_link.py`: the planner's **→ Gen** and the gen console's Analysis tab **DJ track** button (a picker over the library's stem-rendered tracks, ✓ = reading on disk, otherwise the pass runs first) both write the same folder — script + samples + `original.wav` + `features.json` (per-bar features, bar times, the DJ facts and the instrument list) — and the gen tab opens it whole: strip, compare view (A = the source), Recreate, Score, Tune, all against the planner's reading. **Measured honestly** (`tools/tests/_dj_recon_eval.py <ids>`: each stem's reconstruction against the real stem, 6 varied tracks, medians): drums onset F1 0.90 but a 20 dB per-beat spectral gap and -2 dB (dry one-shots, no tails or room); bass onset F1 0.78, chroma 0.63, 14 dB; other onset F1 0.20, chroma 0.53, 11 dB (polyphony and pads are read poorly); vocals are the stem passed through (no sample model sings); onset timing within ±12 ms after the 10 ms sample pre-roll; the mix 8 dB / chroma 0.62. Events keep their exact onset offset from the grid step (v3), held chords play as one sample, pitched hits play their own pitch's recording. The reconstruction is a rhythmic and harmonic SKETCH of the song, not a re-render: what it lacks is drum tails/room/velocity layers, real polyphonic transcription for `other`, and per-note timbre. **Replay** in the gen tab is the programmatic recreation itself: `gen_link.replay` plays every note of the reading through the song's own sampled instruments on its grid into `replay.wav`, side B of the compare view, scorable against the source (Final Voyage: 83 global / 96 structure, against 75 for the composer's **Recreate** of the same script). The gen console's Analysis tab has no file browsing any more: the DJ library is its only source (the planner is where songs are analysed, the gen tab is where they are recreated and varied). Spurious-note guards (v2): onsets more than 18 dB under their instrument's loud hits are bleed, not notes (a quarter of bass "notes" were, mostly kick bleed at the range floor); held-chord candidates on a chosen note's harmonic series (octave, twelfth, two octaves + third...) are its harmonics unless nearly as loud. The Library tab shows an **instr** column next to **stems** (✓ = a current reading exists; its tooltip lists the sounds found per stem; sortable), stamped at library load like `has_stems` (`TrackInfo.has_instruments`). **The reading checks itself (v7, `lib/dj/explain.py`):** every instrument is played back alone and compared with its stem inside its own event windows — `explained` (the share of the stem it accounts for; the **%** beside each name in the instrument panel, green ≥ 60, red < 25, tooltip has overshoot and events-in-silence), and on that evidence events in silence are dropped, voices explaining nothing pruned and look-alike splits merged — each step a closed loop: kept only when the whole stem rendered from the program gets closer to the recording (`lib/dj/fidelity.spectral_gap`; measured 2026-09-09: Sussudio drums 20 → 9 dB, while the same rules unverified cost Final Voyage 7 dB and were rejected). `result["pruned"]` lists what went and why (the tab's status line and `--show` print it). **Fidelity readouts in the gen console** (`lib/dj/fidelity.py`, written as `fidelity.json` by the link, the **Fidelity** button recomputes): NOTES AS CODE — events → patterns + ops, events per pattern+op, the share of events patterns carry vs. written one by one, verbatim bars, vocal phrase reuse, per voice how much of its stem it explains; WAVEFORM vs THE ORIGINAL — the program rendered per stem against the real stems and the mix (spectral gap dB: 0 identical, 6–8 the same part on another instrument, 12+ a different sound; level, rhythm r, missed/extra 16ths, onset F1, chroma). `tools/tests/_dj_recon_eval.py` quotes the same module. **The judge since 2026-09-09 is the TRUTH SET** (`tools/tests/_dj_truthset.py build | eval`): four composed songs rendered through the GM SoundFont and separated with the library's demucs model, every note known — notes F1 (onset within 60 ms and the same pitch) per pitched stem and per drum sound is what a reading change must improve; the spectral gap on real stems is reported, not trusted (it was gamed twice). **Readers v9**: the bass is one monophonic line from a pitch track with envelope re-attacks (`lib/dj/bassreader.py`; truth set 0.61 → 0.80), `other` takes its notes from a polyphonic transcription and assigns them to voices by timbre (`lib/dj/polyreader.py`; 0.29 → 0.61; the note-to-voice assignment is the open problem), drums detect quiet hats (onset threshold 0.03) and decompose a kick that always carries a hat (pop kit hats 0.47 → 0.83). Real songs have no truth: the gen tab's NOTES vs AN INDEPENDENT TRANSCRIPTION line is weak evidence, and the ear decides. |
| `tools/dj/dj_beatport.py` | Beatport discovery CLI: `login` (paste token / `--pkce`), `search "…" --bpm 118-126 --fit`, `fit <id> --deep` (analyze the preview clip), `wish add/list/open`. Public v4 API = search + metadata + preview audio + your library; NO cart API, so buying is a browser click on the track page. See `lib/dj/beatport.py` for the auth story |
| `tools/dj_planner.py` | PyQt6 planner, seven tabs — **Library** (scan button w/ live populate + **"Rescan all"** to force full re-analysis + **"Refine grids"** to re-run low-confidence beat grids — promoting a track past bpm_conf 0.70 unlocks the precision transition styles for it — search, user tags, auto-classification tags, multi-select add, **"🚫 Do not use"** flag — right-click or button — that greys a track out and removes it from EVERYTHING that auto-selects: set generation, the Copilot, and the live autoDJ; the track stays in the browser, "Allow" clears it, DB v11), **Analysis** (zoomable **log-frequency spectrogram** — 30 Hz–16 kHz, toggleable back to the min/max waveform — down to beat level w/ sections + vocal regions + beat grid, play/scrub, user IN/OUT/INTEREST cues that override the analyzer; plus per-song **stem tools**: render this ONE track's stems (htdemucs subprocess, `dj_stems.py --track`), **stem lanes** showing each stem's energy on the same zoomed timeline — separation quality and bleed visible at a glance — **ONE MIXER** over everything that plays: every stem lane / header and every instrument row carries **S / M** boxes with bus semantics (a muted stem silences its rows, solos win over mutes), and a **hear:** switch on the transport row picks the SOURCE those boxes act on — **stems** (real audio: every stem lit = the original mix, a stem S / M = a stem sum, what the separation extracted; a row S = the recording gated to that sound's hits, the honest check of a row against the real audio; a row M is refused here with a hint, real stems cannot drop one sound) or **reconstruction** (the reading played back; every row's S / M works; voices render one at a time on the first switch — ~10 s to build the program, then a few seconds per voice, additive voices longer — and are cached, so every later S / M is an instant sum, and the mix fills in voice by voice with the status line narrating). The spectrogram/waveform follow whatever is heard (per-stem band powers make a stem selection's picture a cheap sum; gated and reconstruction pictures come from a worker once the mix is complete). Before 2026-09-09 this was three overlapping controls (four stem checkboxes acting on the stems, S / M boxes acting only on a reconstruction that a click silently switched to and that took a minute to appear, and a "hear reconstruction" checkbox) — the M/S looked dead and the checkboxes did nothing to what was heard. `tools/tests/_dj_analysis_mixer_test.py` drives the tab headless through all of it. Delete-stems, and **Identify instruments** — the song's OWN instruments found inside its stems (see the row below), drawn as a **tracker-style instrument panel** under the waveform (vertical splitter): a bar ruler, one collapsible header per stem carrying its energy envelope (the stem lanes fold into it), one row per discovered sound with a two-line gutter (the role read for it, then register · length · hits · level · note range), pitched hits as cells with the **note name written in** at bar zoom, held notes as spans with the chord's names, unpitched hits as strength bars, the beat under the playhead lit; minor sounds (under 4% of their stem) hide unless "all sounds" is ticked; a **beat readout** line grouped by stem (`bass: bass+2 E2 | other: pad A2 E3`), and the panel is a **sampler**: clicking a sound's name selects it and plays its exemplar cut from the stem, clicking any note cell plays that very note from the stem, and with a sound selected the keys `z s x d c v g b h n j m` play it at C..B of its exemplar's octave (`q 2 w 3 e r 5 t 6 y 7 u` the octave above, `[ ]` shift octaves) by repitching the exemplar; pitched cells are coloured by pitch class and a hit's measured ring-out trails its cell. The analyzer's text dump (axes, sections) is behind an "ⓘ details" toggle and the track line's tooltip, so the space goes to the pictures), **Set** (the v3 set creator: arc strip showing energy-vs-theme + bpm path + seam quality while you build; one-button **✦ Build set** (suggest → shape → optimize composed; the four individual ops live under More ▾); **Ctrl+Z / Ctrl+Y undo-redo on every set edit**, including the whole-list replaces the ordering ops do; a ranked clickable **worst-seams list** under the report card; per-track "played Nd ago" recency chips (play_history); beam-search Optimize order; anchor timing solver Auto-fill that actually lands timed anchors; right-click repair — slot alternatives + insert-bridge; **▶ Push to live** (save + load into the running show, order or pool mode); a per-set **notes** field; per-seam fade-risk/blend-floor/groove-offset/pair-memory badges; report card; seam audition; and the **Set Copilot** — a conversational Claude tool-loop that searches the library, edits the set, and runs the planning ops, every change visible and one-click revertible; it can also **pin seam styles** (through the same gates the live engine runs), read **night evidence** (`night_history`: live-measured flams per pairing + played-recency per track, so it stops rebuilding last Saturday), see which tracks have **stems**, and **save / push the set to the running show** (executed on the GUI thread after its reply; push requires your explicit go-ahead in the chat); runs through your Claude Code session via the `claude` CLI with NO API key when it's installed — same as the narrative editor — otherwise falls back to the `anthropic` SDK with a key you paste in the panel; **▶ Play set** plays the whole compiled set right there without switching to the Mix tab), **Mix** (DJ-style timeline w/ overlap, beat ticks, real gain/EQ envelopes; play the whole set, jump tracks/seams), **Seam Lab** (a rating treadmill: generates seams with a random arm point, the real brain's choice + plan, renders each through the shared audition renderer, plays it, and advances on a one-key verdict: 1 good / 2 passable / 3 bad / 4 skip, R replays; the NEXT seam renders while you listen. Style selection defaults to **(balance coverage)** — biased hard toward the styles with the least evidence, because you cannot learn why a style fails or where it works from four seams; `(brain's choice)` reproduces a real night's distribution instead, and a named style pins every seam. All three go through the REAL gates, the log records wanted-vs-got, and a style that is asked for repeatedly and never once lands (a retired one, e.g. `cut_at_drop`) drops out of the rotation by itself. Tracks are sampled **without replacement**: both sides of every seam are vetoed for the rest of the session, recycling only once ~70% of the library has been heard (measured over a 120-seam session: 41 tracks repeated before, zero after). A **seam scope** under the plan card draws the audition's mechanics on the mix timeline, zoomed to the rendered seam (not the two whole songs): per-deck **gain and EQ (low/mid/high) envelopes** plus stem gains when a stem style diverges one — so the staged highs→mids→bass migration, the vocal duck and the quiet-intro **entry trim** (B ridden up to +3 dB, drawn above the unity line, released as its own body arrives) are all visible as curves; both decks' **beat grids meeting in the middle**, bar-numbered relative to the seam, with the scripted sync snap simulated and audible-window downbeat flams bridged; a **marker rail** for every other command the script issues (cue, start, loop, brake, echo, filter, stop) plus blend start, seam, A-out and the point of no return; and ONE continuous playhead down the whole picture. It draws from the exact event list the renderer ran (`render_seam`'s `info` out-param), so the picture cannot drift from the audio — the one approximation is the PLL, simulated as its initial snap only. Hover for per-deck position/gain/EQ at that instant; click or drag anywhere to seek, and playback **stops where the scope stops** (past the analysed region the render is just the incoming track playing on). The scope is deliberately bounded to ~150-210 px — the tab's main readout is the **analysis pane** below it (`seamstats.py`), which answers what is failing and HOW: a ranked *What is failing* / *What is working* table (every feature bucket whose good-share departs from the baseline, weighted by evidence), a **Where each style works, and why it fails** section giving every style its *works when* / *fails when* conditions measured against that style's OWN average (so it answers where to reach for a style, not merely whether it is good — and says plainly when a style reads as uniformly weak instead), the engine multiplier currently applied to each style's dice, tracks that keep producing bad seams, gate/pin refusals with reasons, fast-vs-late bad calls, per-session trend, and what the cross-night memory is steering right now. Features covered: style, key fit, stretch, pitch shift, pair score, groove fit, drum alignment (flam window), grid confidence, blend length, stems, theme and engine. Ratings are logged with full diagnostics, and anything an older row predates is **back-filled by joining the track ids against the live library** — `seam_rhythm` is a pure function of the two tracks and the rate, so the whole rating history gets groove/flam analysis, not just sessions after the field was added (memoised, so a refresh after each rating stays ~70 ms even at thousands of ratings). Every number carries its n, buckets under 5 ratings are dropped rather than shown as confident percentages, thin ones are marked, and `passable` counts in the totals but abstains from every good-share. Every verdict lands in `logs/seam_lab_ratings.jsonl` with full plan context (pair, style, rate, pitch, pair score, arm point, engine, listen time) — the analyzable dataset — and good/bad additionally write the same cross-night `seam_feedback` the live thumbs teach, source `lab`, so a rating session directly trains pair/class/style memory), **Discover** (Beatport: sign in with your Beatport **username + password** — the app does the OAuth automatically, no token hunting; password goes only to Beatport's server, only the token is stored. Then search live results with per-result fit vs your set's last track + library; preview audio; ♥ wishlist; open-on-Beatport to buy; "add to set" ghosts a track from its analyzed preview so you can audition the seam before purchasing), **Nights** (read-only post-mortem: each night's play-by-play and the engine's own per-seam verdicts from `logs/dj_*.jsonl` — the same evidence `dj_review.py` reports on — so last weekend's flams are visible while building next weekend's set) |
| Planned sets play as planned | A saved set carries its **theme, compiled length and notes** (DB v14), and the live engine honors all of it on load: the theme applies automatically (no re-picking in the web panel), the night's energy **arc runs on the set's own clock** (a 90-min set traverses its whole arc in ~90 min, not the generic night cycle), and a **pinned seam style** (`style_override`) goes through `plan_transition` itself — geometry and all — both in the compiled preview and live order-mode. Pins are preferences: safety gates (no stems, shaky grid, flam pair) still win, and a refused pin is warned at compile time and logged live (`style_pin` events). The planner's **▶ Push to live** button saves and loads the set into the running show in one click (order or pool mode, via `POST /api/dj/action` — same whitelist as the socket channel). And the live engine's nightly tempo re-measure now **writes back** (`bpm_source='live_verified'`, original kept in `bpm_scan`), so the planner compiles against the tempo the decks actually verified instead of re-discovering the same wrong BPM every night |
| Web `/dj` tab | LIVE control only: start/stop, now/next + blend countdown, theme, energy nudge, autopilot, skip, ABORT MIX (recalls an armed transition before its point of no return), setlist picker. **Music-type chips** (the library's tag vocabulary, incl. a **genre** group from MusicBrainz + embedded genre tags): tap once = **ONLY THIS** (green — a HARD filter; only tracks carrying at least one lit tag may play, not merely boosted), tap again = avoid (red, soft). Composes with the setlist pool (steer within it). **Mix verdicts**: 👍/👎 judge the seam in your EARS — a press while a blend is audibly in flight targets THAT seam (it rides until the swap stamps it), otherwise the last completed one. Every row in the tonight tracklist carries its verdict on the via badge, and the badge is **clickable — 👍 → 👎 → clear** — so a rating that landed on the wrong song can be moved or removed; the stored evidence row is deleted and tonight's style weighting is rebuilt from the verdicts that remain. |

## Config (`config.yaml`)

```yaml
dj:
  enabled: true        # availability only - never auto-plays on boot
  music_dir: ""        # empty = <repo_parent>/music
  theme: groove        # chill_evening / groove / peak_heavy / wind_down / all_night
  night_hours: 6.0     # all_night arc length
  stretch_max: 1.10     # outer tempo wall; config can only TIGHTEN it
```

## How it stays musical

- Track selection couples SONG choice to MIX quality: tempo fit (≤10%
  stretch, half/double-time reads), Camelot compatibility, energy vs the
  theme's arc, recency — AND section-pair mixability: transitions land on
  detected structure boundaries, and two busy/vocal sections never blend
  over each other.
- Selection is TRANSITION-AWARE: candidates whose best seam would be
  forced to a long_fade (loose grid, beatless seam, vocal-over-vocal)
  lean down, as do pairs whose groove offsets differ enough to flam.
- Selection is GROOVE-AWARE (rhythm signatures, DB v13): contradicting
  kick patterns, swung-vs-straight microtiming and flam-band near-misses
  lean a pairing down at the actual tempo-read of the seam (half/double
  reads resample the pattern). Evidence-gated — unscanned tracks are
  neutral — and soft (0.78× floor), because EQ discipline survives most
  kick clashes and a fade opts out of beat physics entirely.
- When a rhythm-rough pair plays anyway (setlist order, thin pool), the
  STYLE hides the clash instead of exposing it: kick clash → the
  one-low-bed styles (`bass_swap`/`stem_drum_swap`/`stem_bass_swap`/
  `cut_at_drop`), never both lows open; swing clash → `stem_drum_swap`
  (removing one percussion bed is the only real fix) or short decisive
  overlaps; flam-band near-misses → the punchy short-dual styles come
  off the menu; a confident 3/4-vs-4/4 meter clash → deliberate fade,
  same rule as a tempo clash. Rough grooves also cap blends at 32 beats
  — don't ride a known clash through two extra phrases.
- SIX stem styles once stems are rendered (`.stems/<id>/`):
  `stem_drum_swap` (drums-only entry), `acapella_out` (A's vocal tail
  rides B's instrumental), `acapella_in` (B's isolated vocal rides A's
  bed, full mix lands at the swap), `stem_bass_swap` (the actual bass
  STEMS trade — zero crossover spill), `drum_bridge` (both tracks strip
  to percussion for 8 beats — the key-clash rescue, boosted exactly
  where harmony fails), `melody_carry` (A's pad/lead bed sustains under
  B for a phrase — tight-key glue). Plus the VOCAL DUCK: when two sung
  passages would overlap on a blend and A has stems, A's vocal stem is
  zeroed through the overlap instead of surrendering the seam to a
  `long_fade` (vocal_over_vocal was a top logged fade reason). All of
  it degrades gracefully — no stems keeps the classic styles; a failed
  stem decode at arm time downgrades to `bass_swap` and logs
  `stem_downgrade`.
- The groove terms are a PREDICTION the system checks against itself:
  every armed plan carries them, and the seam self-assessment logs
  prediction next to measurement (`seam_quality` events,
  `predicted_rhythm`) while class memory buckets feedback by groove
  match too — the data that will eventually tune the term weights.
  Pair scoring walks the 2 Hz energy curves through the seam so a blend
  never lands in (or hands over into) a near-silent stretch. Result on
  the real library: long_fade share 55% → 38%.
- Track energy is grounded in MEASURED loudness + the 2 Hz energy curve
  (how much of the track sits near its own peak), not just mood buckets —
  so energy arcs and energy-based selection actually discriminate a quiet
  ambient piece from a slammed club master.
- Adjacent tracks LEAN toward sharing a genre and era (MusicBrainz +
  embedded genre tags): free-play nights hang together by default, while
  a deliberate pivot stays one good seam away. Missing metadata is
  neutral — no evidence, no penalty.
- Every finished seam is SELF-ASSESSED from its own measurements (worst
  audible grid flam, level holes); a measured train-wreck is stored as a
  gentle auto thumbs-down in cross-night pair memory (half an operator
  vote) — the DJ improves nightly with nobody touching a button. The same
  feedback also generalizes: it aggregates into feature-class memory
  (key fit × groove-offset gap × grid confidence) and per-style memory,
  so one night's lesson steers every future seam of that kind, not just
  the exact same two tracks.
- **Learned execution tuning** (2026-08-02, `lib/dj/tuning.py`). 41 constants
  inside `build_events` — swap position and crossfade width, B's entry
  EQ shelves, the entry-trim ceiling, blend length, `long_fade`'s recede
  level, two-stage arrival and its two A-side carves
  (`fade_a_low_out`, `fade_a_high`), `echo_out`'s delay/feedback/wet/tail, the
  spinback and brake lengths, the vocal-duck depth, the loop-roll shrink
  schedule, the pre-swap dip, the exit reservation — are now named knobs
  (`brain.TUNE_DEFAULTS`) instead of literals. `build_events` resolves each
  as **per-seam override → learned value → original constant**, so a Seam
  Lab verdict eventually changes how the live engine mixes. The Lab nudges
  two style-relevant knobs per seam at random; because the nudge is
  independent of the music it separates from the pair across enough seams,
  so no repeated renders are needed. The update is a **gradient step along
  the nudge/verdict correlation** (which vanishes at the optimum, so it
  converges and stops) — not a jump to an estimated best value, which only
  shifts by the asymmetry in the good-rate and crawls. Guards: 2.5 sigma
  AND |r| >= 0.12 before anything moves (a plain 2-sigma bar drifted knobs
  on noise), steps capped at 22% of the explored range, values confined to
  that range, every move journalled with its evidence in
  `logs/seam_tuning.json`, and `tuning.reset()` restores the constant.
  Verified: defaults reproduce the previous event scripts exactly
  (340/340 plans), every knob measurably changes the automation where it
  is offered, and a closed-loop test against a hidden optimum closed 66%
  of the gap in 8 sessions while no effect-free knob moved.
- Style memory is **conditional, and evidence-weighted** (2026-08-02). The
  same votes are re-aggregated per `(style, condition)` over the coarse
  axes in `brain.seam_conditions` — grid precision, key fit, groove fit,
  flam window — and `Brain.style_multiplier` reads the memory FOR THE SEAM
  IN FRONT OF IT: conditional evidence leads, the global average is a weak
  prior, and it is halved where no conditional evidence applies (an
  average earned elsewhere should not condemn a situation it never saw).
  Both the broad memories are also pulled toward neutral in proportion to
  the votes behind them (`_shrink`, k=8) — measured before this, five
  votes pinned `phrase_cut` to the 0.60 FLOOR while `long_fade` (the
  can't-beat-match FALLBACK) was the only boosted style, which is a
  rich-get-richer collapse toward wall-to-wall fades. **This localizes
  blame, it does not protect anything**: a style that is bad in every
  condition still lands at the bottom of the same 0.6–1.4 band the flat
  memory always used. On the real feedback table it immediately separated
  `long_fade` into ×1.31 on loose grids / ×1.30 on clashing keys — where
  it genuinely is the right tool — from ~×1.05 everywhere else.
- Transition/technique repertoire (all beat-matched via a sync SNAP at
  launch — the incoming deck's phase is instantly aligned to the playing
  track, then a PLL holds it, ±1.2% authority):
  - `long_blend` / `bass_swap` — staged-EQ blends (highs→mids→bass swap)
  - `cut_at_drop` — hard cut on the incoming track's drop. Retired
    2026-08-02, reinstated 2026-08-12 after the retirement's evidence
    (flam 0.247 beats) was re-measured post-grid-phase-fix at 0.017 —
    rebuilt to enter at B's strongest MEASURED drop (`_drop_entries`
    scans B's own energy curve; the old `pre_drop` mix-in hints sat in
    the intro). 2026-08-14, two rounds: first the kick-return check was
    found reading the beat-power profile below its 20 s bucket
    resolution — it silently killed most real drops (6% of the library
    passed while 85% of tracks carry labelled drops; operator: "drops
    seem common... the system thinks they don't happen often") — fixed
    via `drop_kick_levels` (dip→landing at bucket resolution, 21%
    pass). Then the operator rated 25 `cut_drop_shape` Gate Check
    trials: the strict shape bars were wrong 20/25 and no measured
    quantity separated bad from fine, so the bars moved to the rated
    band's floors (step 1.5, land 0.50, run-up 0.65) and the kick kill
    came off entirely (fine at ×0.06, bad at ×0.95 — measurement only
    now). The wider reach then exposed a real defect the render gate
    caught: stored grids are SEGMENTED, and a breakdown can carry a
    garbage segment (72 bpm at score 0.25 inside a conf-0.99 track)
    that the cut's run-in and landing get scheduled on — deterministic
    159 ms grid sawtooth. Entries whose landing or run-in sit in a
    segment >5% off the track's meter are now structurally skipped.
    **72% of the library is cut-eligible** (was 6%). The trial stays in
    the Lab's Gate box serving the next unheard band (step 1.25–1.5)
    since the new bars are themselves unrated
  - `breakdown_swap` — blend over A's breakdown carrying B's build; the
    drop that follows is the payoff. Benched 2026-08-04 (EQ restore
    stacked on B's drop, 9.1 dB slam), rebuilt 2026-08-13 (entry takes
    the build whose drop actually ARRIVES within 4–40 beats; the
    restore clears it by ≥4 beats — lurch 3.3 dB median), un-benched
    2026-08-14 after 12/14 good in the Lab. Note its payoff drop is
    label-vouched (`drop_moments`), and 24% of payoffs measure below
    ×1.25 on the energy curve — if live nights hear "where was the
    drop?", the measured-step vet is the ready lever (rate first)
  - `loop_roll_exit` — shrinking loop-roll outro (retired 2026-08-04
    with `loop_in` and `loop_build` — user verdict on the whole roll
    family: "I don't like the loop rolls at all"; the quality gate had
    also caught loop_in lurching 7.8 dB. spinback_cut retired the same
    day — the slowdown-into-cut mechanic reads "cheesy and overdone",
    and phrase_cut's optional brake is off via the brake_chance knob.
    Old pins for all of them refuse politely.)
  - `loop_build` — stutter a shrinking loop into A's drop to build tension,
    release exactly on the drop as B slams in
  - `long_fade` — fallback for low-confidence grids. Its decks are
    unsynced by design, so it enforces ONE KICK AT A TIME by band
    instead: B enters with its low closed, and at the seam the low band
    is handed over as a baton (A's low out and B's in on the same
    `fade_a_low_out` clock) rather than crossed — the two kick
    fundamentals used to ramp through each other at ~-10 dB for 2-3 s.
    The carve is always applied to the DEPARTING track; B's identity
    arrives whole and its quiet entry is what masks the mismatch.
    Measured by `perc_overlap` (`lib/dj/seamverify.py`), the only rhythm
    instrument that means anything on unsynced decks — kick clash 1.19 s
    → 0.25 s mean over four rhythmic pairs. 2026-08-14, the PERCUSSION
    baton: on predicted-clash pairs (seam kick_agreement < 0.6 at conf
    ≥ 0.5 — evidence-gated) B's entry moves closer to the seam
    (`fade_clash_lead_x`), enters with its high closed, and the top end
    is handed over on the baton clocks at the seam while A's mids leave
    decisively — perc co-presence halved (2.5→1.25 s, 4.75→2.25 s) on
    rendered clash pairs. EQ alone couldn't do it: the 2500 Hz
    crossover leaves half the percussion band inside B's identity mids,
    so the lever is TIME. Separately, seams whose overlap crosses an
    off-meter GRID SEGMENT (the fictitious-lattice defect) divert to
    the fade with `fade_reason: off_meter_segment` (~12% of seams) —
    a blend there would lock a lie. `_dj_quality_test.py` gates
    it on both fade populations. (`fade_a_high`, an air shelf on A,
    exists as a knob but defaults OFF: measured, trimming the louder
    deck moved its transients *toward* the other's rather than out of
    the way, and co-presence got slightly worse.)
  (`bassline_layer` and `double_drop` were removed 2026-08-02 — 3 live
  plays ever, and the fx one-shot holdout respectively; the nextdrop
  MOMENT owns the synced-drop spectacle, on the music alone.)
  Styles are gated by per-track analysis confidence and theme weights.
- **LOOP LAYER — BUILT, THEN SHELVED 2026-08-08.** A percussion bed
  ridden UNDER the playing track on **deck C**. The capability is intact
  and tested; only the two controls are hidden. Operator verdict: *"the
  bed loop isn't clean... it's also not particularly impressive."*
  - **What exists** (all still live): `Deck("c")` in `DJSubmix.decks`;
    `lib/dj/looplayer.py` (loop sourcing); `DJSystem.layer()` /
    `_do_layer` / `_cancel_layer` / `_layer_tick` +
    `LAYER_BARS`/`LAYER_GAIN`/`LAYER_FADE_BARS`; `layer` in the web
    `DJ_ACTIONS` whitelist and the `Stories_OGL` bridge;
    `audition.render_layer`; gate `tools/tests/_dj_layer_test.py`.
  - **What is hidden**: the `◍ LAYER` button on `/dj`, and the **Layer
    Lab** tab (`tools/dj/planner/layerlab.py`, file kept). Both are
    three-line restores, marked in place.
  - **Why it was shelved**: the loop material. Sources were a
    `db.loops_for()` point sliced from a track's demucs **drums stem**,
    or a curated `media/loops/*.wav` DJ tool (BPM in the filename). Only
    the first was ever exercised, because `media/loops/` was empty — and
    demucs drums are bleedy, carry the original room and reverb tails,
    and were never mixed to sit under other music. A real DJ tool is
    dry, single-instrument and mixed with a hole in it. **The feature was
    never tested with the material it was designed for.**
  - **What was measured** (so nobody re-derives it): the reported click
    is NOT the loop wrap — max sample step at the wrap is 0.00065 against
    a 99.9th-percentile step of 0.065 elsewhere in the same loop, 100x
    below its own transients, unchanged by any crossfade. Remaining
    suspects are artefacts inside the drum stem (which recur once per
    loop and read the same way) and slices that are simply not musically
    seamless. Separately: **total-mix RMS is the wrong instrument for
    judging an added layer** — a bed 13 dB down moves summed RMS by
    ~0.2 dB, which says nothing about audibility; measure the layer's own
    level against the track instead (`wet - dry`).
  - **If it is revived**: put real DJ tools in `media/loops/` first
    (Beatport's "DJ Tools" genre, Splice, Loopmasters, or bounce 8 good
    bars of a drum stem yourself). Then consider carving the bed —
    high-pass ~200-300 Hz so it adds percussion above the track's low end
    instead of fighting the kick, and/or sidechain it to the master's
    kick (`Deck.filter` and the submix `duck` already exist). Phase 2
    (persistent across seams, autonomous selection) was never started.
  - **Rules that must survive any revival**, both learned elsewhere in
    this file: drums only (anything pitched clashes in key with what it
    rides under), and never a bed built from the song it plays under —
    four operator moments were retired in a day because "the payoff was
    still the same song". It also must never run over an armed seam
    (`_do_moment`: "layering anything on top of it read as garbage every
    time it was tried"); `_arm` calls `_cancel_layer` for exactly that.
- Incoming deck launches bar-aligned from the DB grid, stretched to the
  running tempo, then a PLL trims ±0.3% on measured beat-phase error;
  after the handover the new track glides back to its natural tempo.
- An armed transition can be RECALLED (web ABORT MIX, or a skip during
  the armed window) up to its point of no return — every style stamps
  the decisive clock (bass swap / cut / drop) into its plan; past it,
  finishing sounds better than any rescue.
- A CONTINUITY WATCHDOG guarantees the music never simply runs out: if
  nothing is armed ~20s before the current track ends (persistent
  "no compatible next", stuck decode, dead planner), it force-picks
  (ignoring tempo gates if it must), buys time with a safety loop over
  the last phrase, and hands off with a clock-domain fade.
- Audit everything before the night: planner seam audition, or
  `dj_player --audition "trackA" "trackB"`. Audit the night AFTER it with
  `python tools/dj/dj_review.py --all`.
- THEMES ARE MEASURED, not asserted. `tools/tests/_dj_theme_sim.py` runs the real
  brain for N nights per theme and prints the pairwise track-set overlap
  plus a dead-flavor-lever audit (a prefer/avoid tag that doesn't exist
  inside that theme's own tempo window is a theme silently having no
  opinion — it has happened twice). Run it after touching `themes.py`.
  Current worst non-`all_night` pair: 0.20 Jaccard, down from 0.45.
- The character AXES the themes steer on are library percentiles
  (`TrackInfo.axes_rank`), not raw analyzer output. Raw `hardness` clipped
  at 1.0 with 81% of a real library tied there, which made every
  hardness target resolve to the same number; `features.hardness_raw` is
  deliberately unbounded because it is only ever consumed through a rank.

## Tests (all self-checking, `ALL PASS` gates)

`_dj_features_test` · `_dj_stretch_test` · `_dj_mix_test` ·
`_dj_brain_test` (incl. full autonomous end-to-end through the hand-pumped
engine judged by the live signals pipeline) · `_dj_setlist_test` ·
`_dj_rescue_test` (abort/skip-while-armed/watchdog through the offline
DJSystem) · `_dj_quality_test` / `_dj_theory_test` / `_dj_soak_test`
(real-library audio invariants, DJ-practice conformance, full-night soak) ·
`_dj_enrich_test` (MusicBrainz + DB v9) · `_dj_mood_test` (Music2Emo mood-pass
wiring + DB v10 — canned blobs, no torch) · `_dj_exclude_test` (do-not-use
flag DB v11 + boundary filter + save-set invariant) · `_dj_rhythm_test`
(rhythm signatures DB v13: synthetic known patterns → extraction, swing,
pairwise clash/flam terms, tempo-multiple recovery, chips vocabulary) ·
`_dj_moment_test` (the operator MOMENT — the nextdrop double-drop —
measured in the rendered audio: the build, the hole, the landing on the
incoming drop, the abort recall, and that every refusal is visible; pass
a path to also write a WAV you can listen to) · `_dj_moment_vis_test`
(the moment's visual choreography through the real coupler: build ramp,
breath-hold, hard drop stamp) · `_dj_spectral_test` (the frequency-domain
sibling of `_dj_quality_test`: seam renders with per-deck post-EQ
low-band taps gate one-bassline-at-a-time, hat/mud stacking, low-end
holes and cliff swaps against the pair's own solo behavior — A and B
rendered alone absolve the music's own moves — plus stuck-filter /
carved-EQ restoration after the seam, and a fast no-audio audit that the
spectral shares, section bass info and `spectral_lean` steering engage
at all).

Sims (not pass/fail — they print distributions you read):
`_dj_persona_sim` (are the personas audibly different DJs?) ·
`_dj_theme_sim` (do the themes reach different music?).

## Gotchas (hard-won)

- OPEN DEFECT — 16-BEAT `long_blend` LOSES ITS STAGED ENTRY. `set_gain`
  replaces any pending ramp, so when the swap lands at the blend start
  (`mid` == S0) the stage-1 event is overwritten in the same sample and B
  arrives at full instead of riding under A; `stage1_gain` / `stage1_frac`
  do nothing there. Measured over 40 planned blends: stage-1 lives 17.2s
  at 64 beats and 25.4s at 96, but **0.000s on all four 16-beat seams**.
  Not fixed — see the note at the `long_stage` branch in `brain.py`, and
  A/B any fix by ear rather than by trough depth.
- A CROSSFADE'S DIP IS NOT ALWAYS THE FADE LAW. The decks are genuinely
  uncorrelated (|rho| < 0.06 in every band — beat *alignment* is not
  waveform *correlation*), so linear ramps should cost 3 dB at the cross
  and `Deck.set_gain(curve="power")` now interpolates g² for overlapped
  styles. Measured gain: **±0.2 dB**, because the automation staggers the
  two fades rather than crossing them. The mid-blend sag is spectral —
  B enters bass-cut and mid-shelved while A's low hands over — and that
  carve is deliberate.
- ONE NaN POISONS A WHOLE PERCENTILE. `np.percentile` over a list holding a
  single NaN returns NaN, every `value >= NaN` is False, and the tag
  vanishes library-wide. Ten near-silent tracks with a NaN energy axis
  erased the `driving` and `mellow` tags from all 649 tracks while the
  themes went on asking for them. `features._finite` is the guard; run
  `dj_scan.py --retag` after any tagging-rule change.
- A percentile rank over a heavily TIED axis is a lie in a different way:
  mid-ranking the ties is honest ("everyone ~0.55") but it means the axis
  steers nothing. Fix the axis, don't fix the rank.
- The live `BeatDetector` quantizes BPM to integer 40 fps lags (±2.5%);
  measure tempo precision with `features.estimate_beat_grid`, never the
  live detector.
- Analyzer chroma is A-origin: `c_origin[j] = a_origin[(j+3)%12]`.
- Spectral-flux onsets LEAD the true transient by ~28 ms with our 4096
  framing — `features.ONSET_LATENCY_S` compensates; don't remove it.
- WSOLA legitimately duplicates/skips the odd transient beyond ±5%
  stretch; the brain prefers small ratios for a reason.
- Windows can't decode m4a/aac via miniaudio — PyAV fallback handles it.
- A one-shot layered over the live mix is inaudible at any sane gain: the
  MOMENT button used to just play a riser + impact over an unchanged track
  (riser RMS −17 dBFS under a −9 dBFS master) and read as nothing. Crowd
  moments are CONTRAST *with a PAYOFF* — contrast alone (sweep out, hole,
  same bar resumes) still read as the song pausing. And the synth riser
  is gone from the moments entirely (third strike, 2026-07-29: "that
  shitty whoosh") — at ANY honest level it reads as a cheap sample pasted
  on the song; the build is the track shaping itself (sweep, trim push,
  pitch rise, loop-roll). `fx.at_peak` exists because the `gain` args of
  `make_riser`/`make_impact` are pre-filter amplitudes, not peaks
  (filtered noise has a crest factor near 5 — `gain=0.26` clips). at_peak
  is only for PERCUSSIVE one-shots: peaking a riser's squared swell
  (crest ~11) buried its body 13 dB under the track — same failure,
  different knob (`fx.at_tail` states a swell in tail RMS; the brain's
  seam styles still use quiet risers, that's their call).
- MOMENT is ONE gesture (`_do_moment` → `_moment_nextdrop`): the set
  double-drops forward into the NEXT track's real drop. Four flavors
  shipped 2026-07-29 and the operator's next-day verdict was final —
  "only next is good", "drop is awful", "stall is worthless", "spinback
  is basically another next". Every same-track gesture (build-and-
  resume, build-and-jump, echo stall, spinback dive) failed three
  consecutive rebuilds across three different sound designs, because
  the payoff was still the same song. The lesson, in one line: **a
  crowd moment must change the music; everything else is a wet fart no
  matter how it's dressed.**
  - The build runs on the dying track: HP sweep 30→600 Hz spanning the
    whole 8–24-beat wait, ~2 dB trim push into the freed headroom, a
    snare roll, and the LOOP-ROLL (beat-repeat 1 → ½ → ¼ — the deck
    loop is a virtual→source map, so a bare clear at the hole lands
    exactly where the track would have been unlooped). No synth riser
    anywhere ("that shitty whoosh").
  - One beat of hole; the incoming deck pre-rolls silently under it and
    its drop slams in cold at full gain on the landing downbeat.
  - Arms as a real transition (style `moment_nextdrop`, swap on the
    landing): `_finish_swap` does the handover bookkeeping, and recall
    (second press or ABORT MIX) is `_do_abort`.
  - Anything it can't deliver it refuses OUT LOUD: `_moment_skip`
    stamps `moment_denied` into `status()` and the panel flashes the
    reason on the button ("no next queued", "next has no drop", "mix in
    progress", "deck not up"). A silent refusal is indistinguishable
    from a dead button. The incoming drop needs `PLAN_LEAD_S + 25 +
    ride` (~115 s) of runway after it, or entering the new track at its
    drop would arm the NEXT blend seconds later (user-heard 2026-07-29:
    track gone 80 s in).
  The visuals get `dj_moment_eta` (build ramp),
  `dj_moment_hole` (breath-hold: build pinned, pulses suppressed) and a
  HARD drop stamp at the landing (`dj_drop_hard` → ~3× longer slam than
  a passing musical drop).
- Anything that shapes the LIVE deck outside a transition must be recalled
  by whatever takes the deck over next (`_cancel_moment` from `_arm`,
  `_do_seek`, the watchdog handoff; today's one gesture arms as a
  transition so `_do_abort` is its recall, but the guard stays for any
  future txn-tagged gesture). A half-fired build leaves the deck
  high-passed at 600 Hz and 7 % gain, which is a dead room.
