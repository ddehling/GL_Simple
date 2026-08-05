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
python tools/dj/dj_analyze.py            # add --stems for the stem render
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
| Rhythm signatures | Beat-sync GROOVE fingerprint per track (DB v13, `lib/dj/rhythm.py`, sig v2): kick/snare/hat 16th-note step patterns folded over 2 bars, one-beat fine folds, **swing** (0.50 straight → 0.67 shuffle, onset-latency corrected), **density**, **meter** (3/4 vs 4/4, beat-level accent contrast — only a confident 3/4 claim counts) and **region patterns** (~48s folds at the primary mix-in/mix-out, so a seam compares A's EXIT pattern against B's INTRO pattern — the material the blend actually overlaps — not two whole-track averages). Computed inline on scan (mix-derived); the planner's **"Rhythm"** pass (`tools/dj/dj_rhythm.py`, in "Analyze all" after stems) backfills old libraries, upgrades v1→v2 formats, and upgrades to **drum-stem-derived** where stems exist. Powers: the seam **chips** (word-first limiting factor — `kick clash`, `swung vs straight`, `3/4 vs 4/4`, `half-time`, `flam risk 40ms`, `stretch +5.8%`; a trailing `?` = shaky grids), compile warnings, the **seam inspector** (click a ↳ seam: both region step grids aligned at the planned tempo-read, low-band contradictions marked, one-beat flam microscope), the library **rhythm column** (kick-pattern glyph, sorts by density), arc-strip tooltips, the Copilot's seam explanations, the web `/dj` armed-seam chips, and the LIVE engine (see below). |
| `tools/dj/dj_beatport.py` | Beatport discovery CLI: `login` (paste token / `--pkce`), `search "…" --bpm 118-126 --fit`, `fit <id> --deep` (analyze the preview clip), `wish add/list/open`. Public v4 API = search + metadata + preview audio + your library; NO cart API, so buying is a browser click on the track page. See `lib/dj/beatport.py` for the auth story |
| `tools/dj_planner.py` | PyQt6 planner, seven tabs — **Library** (scan button w/ live populate + **"Rescan all"** to force full re-analysis + **"Refine grids"** to re-run low-confidence beat grids — promoting a track past bpm_conf 0.70 unlocks the precision transition styles for it — search, user tags, auto-classification tags, multi-select add, **"🚫 Do not use"** flag — right-click or button — that greys a track out and removes it from EVERYTHING that auto-selects: set generation, the Copilot, and the live autoDJ; the track stays in the browser, "Allow" clears it, DB v11), **Analysis** (zoomable **log-frequency spectrogram** — 30 Hz–16 kHz, toggleable back to the min/max waveform — down to beat level w/ sections + vocal regions + beat grid, play/scrub, user IN/OUT/INTEREST cues that override the analyzer; plus per-song **stem tools**: render this ONE track's stems (htdemucs subprocess, `dj_stems.py --track`), **stem lanes** showing each stem's energy on the same zoomed timeline — separation quality and bleed visible at a glance — **solo/mute audition** (uncheck stems to hear exactly what was extracted; all four checked plays the original mix), and delete-stems), **Set** (the v3 set creator: arc strip showing energy-vs-theme + bpm path + seam quality while you build; one-button **✦ Build set** (suggest → shape → optimize composed; the four individual ops live under More ▾); **Ctrl+Z / Ctrl+Y undo-redo on every set edit**, including the whole-list replaces the ordering ops do; a ranked clickable **worst-seams list** under the report card; per-track "played Nd ago" recency chips (play_history); beam-search Optimize order; anchor timing solver Auto-fill that actually lands timed anchors; right-click repair — slot alternatives + insert-bridge; **▶ Push to live** (save + load into the running show, order or pool mode); a per-set **notes** field; per-seam fade-risk/blend-floor/groove-offset/pair-memory badges; report card; seam audition; and the **Set Copilot** — a conversational Claude tool-loop that searches the library, edits the set, and runs the planning ops, every change visible and one-click revertible; it can also **pin seam styles** (through the same gates the live engine runs), read **night evidence** (`night_history`: live-measured flams per pairing + played-recency per track, so it stops rebuilding last Saturday), see which tracks have **stems**, and **save / push the set to the running show** (executed on the GUI thread after its reply; push requires your explicit go-ahead in the chat); runs through your Claude Code session via the `claude` CLI with NO API key when it's installed — same as the narrative editor — otherwise falls back to the `anthropic` SDK with a key you paste in the panel; **▶ Play set** plays the whole compiled set right there without switching to the Mix tab), **Mix** (DJ-style timeline w/ overlap, beat ticks, real gain/EQ envelopes; play the whole set, jump tracks/seams), **Seam Lab** (a rating treadmill: generates seams with a random arm point, the real brain's choice + plan, renders each through the shared audition renderer, plays it, and advances on a one-key verdict: 1 good / 2 passable / 3 bad / 4 skip, R replays; the NEXT seam renders while you listen. Style selection defaults to **(balance coverage)** — biased hard toward the styles with the least evidence, because you cannot learn why a style fails or where it works from four seams; `(brain's choice)` reproduces a real night's distribution instead, and a named style pins every seam. All three go through the REAL gates, the log records wanted-vs-got, and a style that is asked for repeatedly and never once lands (a retired one, e.g. `cut_at_drop`) drops out of the rotation by itself. Tracks are sampled **without replacement**: both sides of every seam are vetoed for the rest of the session, recycling only once ~70% of the library has been heard (measured over a 120-seam session: 41 tracks repeated before, zero after). A **seam scope** under the plan card draws the audition's mechanics on the mix timeline, zoomed to the rendered seam (not the two whole songs): per-deck **gain and EQ (low/mid/high) envelopes** plus stem gains when a stem style diverges one — so the staged highs→mids→bass migration, the vocal duck and the quiet-intro **entry trim** (B ridden up to +3 dB, drawn above the unity line, released as its own body arrives) are all visible as curves; both decks' **beat grids meeting in the middle**, bar-numbered relative to the seam, with the scripted sync snap simulated and audible-window downbeat flams bridged; a **marker rail** for every other command the script issues (cue, start, loop, brake, echo, filter, stop) plus blend start, seam, A-out and the point of no return; and ONE continuous playhead down the whole picture. It draws from the exact event list the renderer ran (`render_seam`'s `info` out-param), so the picture cannot drift from the audio — the one approximation is the PLL, simulated as its initial snap only. Hover for per-deck position/gain/EQ at that instant; click or drag anywhere to seek, and playback **stops where the scope stops** (past the analysed region the render is just the incoming track playing on). The scope is deliberately bounded to ~150-210 px — the tab's main readout is the **analysis pane** below it (`seamstats.py`), which answers what is failing and HOW: a ranked *What is failing* / *What is working* table (every feature bucket whose good-share departs from the baseline, weighted by evidence), a **Where each style works, and why it fails** section giving every style its *works when* / *fails when* conditions measured against that style's OWN average (so it answers where to reach for a style, not merely whether it is good — and says plainly when a style reads as uniformly weak instead), the engine multiplier currently applied to each style's dice, tracks that keep producing bad seams, gate/pin refusals with reasons, fast-vs-late bad calls, per-session trend, and what the cross-night memory is steering right now. Features covered: style, key fit, stretch, pitch shift, pair score, groove fit, drum alignment (flam window), grid confidence, blend length, stems, theme and engine. Ratings are logged with full diagnostics, and anything an older row predates is **back-filled by joining the track ids against the live library** — `seam_rhythm` is a pure function of the two tracks and the rate, so the whole rating history gets groove/flam analysis, not just sessions after the field was added (memoised, so a refresh after each rating stays ~70 ms even at thousands of ratings). Every number carries its n, buckets under 5 ratings are dropped rather than shown as confident percentages, thin ones are marked, and `passable` counts in the totals but abstains from every good-share. Every verdict lands in `logs/seam_lab_ratings.jsonl` with full plan context (pair, style, rate, pitch, pair score, arm point, engine, listen time) — the analyzable dataset — and good/bad additionally write the same cross-night `seam_feedback` the live thumbs teach, source `lab`, so a rating session directly trains pair/class/style memory), **Discover** (Beatport: sign in with your Beatport **username + password** — the app does the OAuth automatically, no token hunting; password goes only to Beatport's server, only the token is stored. Then search live results with per-result fit vs your set's last track + library; preview audio; ♥ wishlist; open-on-Beatport to buy; "add to set" ghosts a track from its analyzed preview so you can audition the seam before purchasing), **Nights** (read-only post-mortem: each night's play-by-play and the engine's own per-seam verdicts from `logs/dj_*.jsonl` — the same evidence `dj_review.py` reports on — so last weekend's flams are visible while building next weekend's set) |
| Planned sets play as planned | A saved set carries its **theme, compiled length and notes** (DB v14), and the live engine honors all of it on load: the theme applies automatically (no re-picking in the web panel), the night's energy **arc runs on the set's own clock** (a 90-min set traverses its whole arc in ~90 min, not the generic night cycle), and a **pinned seam style** (`style_override`) goes through `plan_transition` itself — geometry and all — both in the compiled preview and live order-mode. Pins are preferences: safety gates (no stems, shaky grid, flam pair) still win, and a refused pin is warned at compile time and logged live (`style_pin` events). The planner's **▶ Push to live** button saves and loads the set into the running show in one click (order or pool mode, via `POST /api/dj/action` — same whitelist as the socket channel). And the live engine's nightly tempo re-measure now **writes back** (`bpm_source='live_verified'`, original kept in `bpm_scan`), so the planner compiles against the tempo the decks actually verified instead of re-discovering the same wrong BPM every night |
| Web `/dj` tab | LIVE control only: start/stop, now/next + blend countdown, theme, energy nudge, autopilot, skip, ABORT MIX (recalls an armed transition before its point of no return), setlist picker. **Music-type chips** (the library's tag vocabulary, incl. a **genre** group from MusicBrainz + embedded genre tags): tap once = **ONLY THIS** (green — a HARD filter; only tracks carrying at least one lit tag may play, not merely boosted), tap again = avoid (red, soft). Composes with the setlist pool (steer within it). |

## Config (`config.yaml`)

```yaml
dj:
  enabled: true        # availability only - never auto-plays on boot
  music_dir: ""        # empty = <repo_parent>/music
  theme: groove        # chill_evening / groove / peak_heavy / wind_down / all_night
  night_hours: 6.0     # all_night arc length
  stretch_max: 1.08
```

## How it stays musical

- Track selection couples SONG choice to MIX quality: tempo fit (≤8%
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
- **Learned execution tuning** (2026-08-02, `lib/dj/tuning.py`). 39 constants
  inside `build_events` — swap position and crossfade width, B's entry
  EQ shelves, the entry-trim ceiling, blend length, `long_fade`'s recede
  level and two-stage arrival, `echo_out`'s delay/feedback/wet/tail, the
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
  - `cut_at_drop` — hard cut on the incoming track's drop (retired
    2026-08-02: 0/2000 rolls, phrase_cut does its job; old pins refuse
    politely)
  - `loop_roll_exit` — shrinking loop-roll outro (retired 2026-08-04
    with `loop_in` and `loop_build` — user verdict on the whole roll
    family: "I don't like the loop rolls at all"; the quality gate had
    also caught loop_in lurching 7.8 dB. spinback_cut retired the same
    day — the slowdown-into-cut mechanic reads "cheesy and overdone",
    and phrase_cut's optional brake is off via the brake_chance knob.
    Old pins for all of them refuse politely.)
  - `loop_build` — stutter a shrinking loop into A's drop to build tension,
    release exactly on the drop as B slams in
  - `long_fade` — fallback for low-confidence grids
  (`bassline_layer` and `double_drop` were removed 2026-08-02 — 3 live
  plays ever, and the fx one-shot holdout respectively; the nextdrop
  MOMENT owns the synced-drop spectacle, on the music alone.)
  Styles are gated by per-track analysis confidence and theme weights.
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
