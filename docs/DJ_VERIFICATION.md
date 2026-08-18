# DJ Verification Stack

How the DJ's mixing quality is measured, simulated, and gated — without
playing music at 1x. Written 2026-08-17 after the investigation that
root-caused the 08-16 live disaster, built the offline night simulator,
and put every measurement instrument through an ear exam. Read this
before trusting any DJ quality number or adding a new instrument.

## The instruments — which numbers to believe

| Instrument | Where | Status |
|---|---|---|
| **Isolated-kick alignment** (`seamverify.measured_kick_alignment`) | harness renders (`kick_iso` / `kick_flam_ms`) | **EAR-VALIDATED, the flam ground truth.** Per-deck low-band attacks vs each deck's own trace-projected grid (beatpower --phase method, ±90ms bounded, IQR-gated); only the two kick clocks are compared — the decks never meet, so rhythm patterns can't cross-contaminate. Declines on diffuse material rather than guess. Bar: >45ms. Exam: operator-good seams med 12.1ms, bad 37.6ms, monotonic. |
| Grid-phase lock (PLL telemetry, `max_err_beats`) | live + harness | Trustworthy **only where grids are trusted** — it measures lattice lock, and at bpm_conf ~0.66 the lattice doesn't describe the music (the 08-16 disaster measured 0.042 beats "clean" while kicks flammed 125ms). This is the bar that charges pair memory live (0.12 beats sustained). |
| Level lurch vs the pair's own solo dynamics | harness (`lurch_db` vs `lurch_solo_db`) | Believable. Bar: > max(6dB, solo+1.5). |
| Dead air (`rms_min_ratio`), double bass (`bass_bump_db`), clipping | harness | Believable. Bars: <0.15, >+3.5dB, >0. |
| Env-xcorr kick lag (`lag_med_ms`) | harness | **OVERREADS — comparison only.** Cross-correlating two full-band rhythm envelopes returns the *pattern-similarity* offset (A's shaker vs B's congas ≈ an eighth note), not kick alignment. Failed the ear exam *inverted*: good seams med 75ms, bad 50ms. Never gate on it. |
| Wide audible meter (`audible_err_beats`, `max_audible_beats`) | live submix + logs | **NO EAR SIGNAL — diagnostic count only.** Same xcorr family. Had live verdict power for one day (08-16→17); revoked. Fields still logged; nothing surfaces them as flags. |

**The standing rule: no instrument gets believed until it passes the
ear anchor** (below) — it must order the operator's existing verdicts.
Both deprecated instruments looked plausible until that exam.

## The tools (all from repo root, plain `python`, no venv)

**Always pass `--music D:/Devel/music` on this machine** — defaults in
some tools point at an empty Desktop DB.

| Tool | Command | Time | Purpose |
|---|---|---|---|
| Phase-waiver gate | `python tools/tests/_dj_phasewaiver_test.py` | sec | Gate waivers must ask the correction's own question (region-specific phase lookups, reach bounds). Run after touching `beatpower.phase_offset`, `_local_ok`, or gate standdowns. |
| Selection audit | `python tools/tests/_dj_quality_test.py --audit-only --music D:/Devel/music` | ~4 min | Full-library pick sweep: style reachability, persona targets, fade share, stretch wall. Run after any `brain.py` planning/gate/weight change. |
| Full quality gate | `python tools/tests/_dj_quality_test.py --music D:/Devel/music` | ~15 min | Audit + rendered seam measurements. `--wav` dumps audio, `--diag` per-seam events. Run before committing `build_events`/submix/deck changes. |
| Night census | 4 parallel workers: `python tools/tests/_dj_night_sim.py --worker N --nights 3 --seams 84 --music D:/Devel/music` (N=0..3), then `python tools/tests/_dj_night_sim.py --report` | ~55 min | 1,008 seams as 12 chained nights (recency, arc, personas, fades, stems) — "what breaks in normal operation." Shards in `logs/night_sim/`; **ARCHIVE `w*.jsonl` (rename the dir) before a fresh census, never delete** — per-seam before/after diffs across censuses need them, and the workers are seed-deterministic so paired diffs are exact (lesson: the v2 shards were deleted on 2026-08-17 and a severity comparison became unprovable). |
| Ear anchor | `python tools/tests/_dj_ear_anchor.py --music D:/Devel/music --limit 80` | ~15 min | Re-renders operator-rated Lab seams; a valid instrument must order good < passable < bad. **Run before trusting any new/changed instrument.** Fresh run: delete `logs/ear_anchor.jsonl`. |
| Audible-meter calibration | `python tools/tests/_dj_audible_calib.py --music D:/Devel/music --n 40` | ~10 min | Only if the wide meter's thresholds are revisited. Rows v3+ in `logs/audible_calib.jsonl`; threshold sweeps re-run free from stored series. |
| Persona sim | `python tools/tests/_dj_persona_sim.py --music D:/Devel/music` | ~5 min | After persona/style_bias changes. |
| Setlist gate | `python tools/tests/_dj_setlist_test.py --music D:/Devel/music` | ~5 min | After setlist/pool/queue changes in `system.py`. |

Simulation speed: ~13s per rendered seam single-process (~4–7× realtime
on the audio, 15–25× on seams vs a live night); 4 workers ≈ 1,000
seams/hour. `render_seam` accepts `pair=`, `tune=`, `decoded=` (decode
cache for chained renders), `test_gates=` (forensic renders of
gate-refused pairs) — and **attaches stems** for stem-style renders.

## Census baseline (2026-08-17, 1,008 seams, validated instruments)

| Issue | Share | Notes |
|---|---|---|
| Level lurch | 11.4% → 10.9% with the gap policy (census v3) | The worst class — hush-then-slam stop-gaps landing late in blends — is fixed by `lib/dj/gapscan` (predicted-exposure policy, A/B-verified: 15.5→3.4dB on the worst specimen, verified present in the fleet render). The remaining ~11% is heterogeneous: 33 are fades (deliberate-dip shape + craters, different fix), the biggest surviving blend lurchers (20–23dB) are NOT the hush-slam class and are undiagnosed, and the Condor-class B-side stop-gap evades the envelope prediction when the swap lands late. Each next class needs its own anatomy pass (the lurch_anatomy method: signed band-split steps against the event timeline). |
| True kick flam (>45ms) | 6.5% of synced | Median 16.7ms overall — nights are genuinely well-aligned. Flams concentrate at min(conf)<0.8 (37/54) and large opposite-sign per-deck kick offsets (±60ms each side, beyond bias correction). |
| Dead air | 6.5% | Fade craters at dead tails; 'Dunes'-class dying outros recur as A-side. |
| Double bass | 1.2% | Minor. |
| Clipping | 0% | Clean. |

**(1) Fade craters — FIXED 2026-08-17 in exit-anchor selection**
(`03d7b22`), after two arm-time timing patches were tried and reverted.
Anchors are now scored by `_exit_life`: the A-side's own 2 Hz curve
across the fade's exposure window, 1s-smoothed **min** (not a
quantile — that lost twice on Dunes; the dead-air gate is a min and a
1–2s hush notch IS the crater), squared, against 0.6× body energy, and
applied OUTSIDE the weighted-sum fit floor. Specimens: Mukadderat→
06_DEADLIFE 35.2→6.5dB lurch and 0.003→0.318 floor; Symmetry and Take
Me Home 0.06/0.08→0.33. Census re-run pending at commit time.

### Levers that DIED against the operator's verdicts (do not rebuild)

Both were swept 2026-08-17 against ~470 rated seams. Neither survives
rule 2 — and the sweeps cost an afternoon, so they are recorded here
rather than re-derived:

- **Plan-time cap on combined per-deck kick offset** (the old ranked
  lever #2). The sync bias already consumes the entire offset: the
  residual after the ±0.25-beat clip is zero for ~97% of seams in
  *every* verdict class. Raw combined offset does not separate either
  (good 71.7ms vs bad 76.6ms; refusal precision 44% against a 41% base
  rate) — a 50ms cap would refuse 74 of 115 GOOD seams.
- **Raising the grid-confidence gate.** 39–44% precision at every
  threshold from 0.70 to 0.90, against a 39% base rate. min(bpm_conf)
  medians: good 0.77, bad 0.75.

Key fit (camelot) and arc-energy distance were swept in the same pass
and also sort nothing (both tiers, all verdicts ≈ identical).

### What the remaining bad verdicts actually are

Rendered isolated-kick flam DOES track the ear (good med 12.1ms, bad
37.6ms; 44% of bads over 45ms vs 8% of goods) — but **nothing available
at plan time predicts it**, per the dead levers above. So the next
real lever is an **arm-time pre-flight**: seams arm 60–110s ahead and
the decks already hold the audio, so `seamverify.measured_kick_alignment`
can measure each deck's own kick clock at the planned anchors before
committing, and divert to the fade over ~45ms. Untested — the
experiment that earns it is: measure isolated-kick flam at plan time
for the already-rated seams and check it predicts the verdicts where
the stored offsets did not.

Remaining smaller levers: extend the kick-agreement 0.35–0.6 damp
beyond kit styles (weakest evidence, n=2); the Condor-class B-gap that
evades the blend policy's envelope prediction.

### Style benches (2026-08-17)

`drum_bridge` (2/1/5 good/passable/bad since 08-15) and
`stem_bass_swap` (19/23/44) joined `melody_carry`/`acapella_out` on the
bench — benched rather than gated for the reason above: no measurable
quantity sorts their verdicts. Their seams reassigned to long_blend and
bass_swap, NOT to fades (fade share 412→413 of 1206). `stem_drum_swap`
stays live (10/10/8, a different animal). The Lab's `allow_benched`
hatch still plays all four for revival listens.

## Hard-won rules (each cost a wrong conclusion)

1. **Waiver = correction, one lookup.** A gate standdown must ask the
   exact question the correction will ask (the 08-16 disaster's root
   cause; `_dj_phasewaiver_test` holds it).
2. **Ear-anchor before belief.** Every instrument must order the
   operator's existing verdicts before it gates, verdicts, or flags.
3. **Never mutate shared singletons in harnesses.** `force_style`
   rebinding `style_weights` on the `get_theme` singleton silently
   pinned a "natural" sample to one style (same defect class as the
   08-14 hashseed leak). Copy first.
4. **Harness fidelity mirrors live exactly**: settled windows, stems
   attached, the live collector's cadence. Two calibrations were
   invalidated by fidelity gaps before this stuck.
5. **The operator's words outrank stats**, and their one-line
   reactions ("thats a lot of flam") have caught what summaries hid —
   surface anomalies, don't average them away.
6. **What the operator dislikes is mostly not sync.** Even the honest
   meter finds <half of their "bad" verdicts are kick flam; the rest
   are musical (the benched "pointless" tail styles). Sync gates can't
   fix taste.
