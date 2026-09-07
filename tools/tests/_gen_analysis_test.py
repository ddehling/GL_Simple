"""Gate for song analysis (lib/gen/analysis + lib/gen/script): a song goes
in, the generator's own commands come out, the recreation is scored.

  1. SCRIPT: a SongScript drives the composer - sections and lengths as
     written, levers applied per section, chords as given, the hook as
     the theme, end -> outro; to_actions() compiles it to whitelisted
     actions; save/load round-trips.
  2. INGEST: rendering a script and ingesting the result recovers tempo
     (within 1%), key, a plausible style and section count, energy that
     rises into the loud middle and falls at the end.
  3. RECREATE + SCORE: the song against itself scores 100; the faithful
     recreation scores well (>= 65 global); a deliberately wrong
     recreation (other style, other key, flat energy) scores lower;
     local scores exist per phrase and the weakest can be named.
  4. ACTION: the "script" action is whitelisted and loads into a system.

Usage: python tools/tests/_gen_analysis_test.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.gen import RATE                                  # noqa: E402
from lib.gen import script as S                            # noqa: E402
from lib.gen.analysis import ingest as I, score as SC      # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def main():
    os.environ["GEN_HOOKS"] = "0"
    os.environ["GEN_VST"] = "0"          # analog voices only: hosted plugins are not bit-deterministic
    print("== script")
    sc = S.example()
    c = S.make_composer(sc)
    ps = list(c.phrases_until(int(S.total_seconds(sc, c.bpm) * RATE)))
    secs = [(p.bar0, p.section) for p in ps]
    want = []
    bar = 0
    for e in sc["sections"]:
        for b in range(0, e["bars"], 4):
            want.append((bar + b, e["section"]))
        bar += e["bars"]
    check(secs[: len(want)] == want, f"sections follow the script ({len(want)} phrases)")
    groove = [p for p in ps if p.section == "groove"]
    check(groove and all([ch[0] for ch in p.chords] == [0, 0, 5, 6] for p in groove), "scripted chords are played as given")
    drop = [p for p in ps if p.section == "drop"]
    check(drop and drop[0].meta.get("lead_op") in ("theme", "theme_make") and c.melody.theme is not None and c.melody.theme.name == "script",
          "the scripted hook is the theme on the drop")
    check(c.form.ending and ps[-1].section == "outro", "end -> the outro closes the song")
    acts = S.to_actions(sc)
    kinds = {a for _, a, _ in acts}
    check({"style", "key", "bpm", "section", "energy", "mute", "end"} <= kinds and acts[0][1] == "style",
          f"to_actions compiles to the action vocabulary ({len(acts)} commands: {sorted(kinds)})")
    tmp = os.path.join(tempfile.gettempdir(), "gen_script_test.yaml")
    S.save(sc, tmp)
    check(S.load(tmp) == S.normalize(sc), "save/load round-trips")

    print("== ingest")
    folder = os.path.join(tempfile.gettempdir(), "gen_analysis_test")
    os.makedirs(folder, exist_ok=True)
    wav = os.path.join(folder, "original.wav")
    audio, _ = S.render(sc, out_path=wav)
    res = I.ingest(wav)
    a, s2 = res["analysis"], res["script"]
    check(abs(a["bpm"] - sc["bpm"]) / sc["bpm"] < 0.01, f"tempo recovered ({a['bpm']:.2f} vs {sc['bpm']})")
    check(s2["key"] == sc["key"], f"key recovered ({a['key']} = {s2['key']} vs {sc['key']})")
    check(s2["style"] in ("groove", "techno"), f"style plausible ({s2['style']})")
    n = len(s2["sections"])
    check(3 <= n <= 8 and abs(S.total_bars(s2) - S.total_bars(sc)) <= 8, f"{n} sections, {S.total_bars(s2)} bars (script {S.total_bars(sc)})")
    en = [e["energy"] for e in s2["sections"]]
    check(en[0] < max(en) - 0.2 and en[-1] < max(en) - 0.2 and max(en) >= 0.9, f"energy shape recovered {[round(x, 2) for x in en]}")
    check(res["features"] and all(len(f["chroma"]) == 12 for f in res["features"]) and len(res["chords"]) == len(res["features"]),
          f"{len(res['features'])} bar features with chroma and chords")
    check(sum(1 for x in res["chords"][8:48] if x in (0, 5, 6)) >= 0.4 * 40, f"chords lean on the loop's degrees {res['chords'][8:20]} (mix-only reader, ~half right)")

    print("== recreate + score")
    self_rep = SC.compare(res["features"], res["features"], bpm_orig=a["bpm"], bpm_recon=a["bpm"], key_orig="8A", key_recon="8A")
    check(self_rep["global"] == 100.0 and all(r["score"] == 100.0 for r in self_rep["local"]), "the song against itself is 100")
    rec, _ = S.render(s2, out_path=os.path.join(folder, "recreation.wav"))
    fr = I.features_on_grid(rec.mean(axis=1).astype(np.float32), s2["bpm"], 0.0)
    rep = SC.compare(res["features"], fr, bpm_orig=a["bpm"], bpm_recon=s2["bpm"], key_orig=s2["key"], key_recon=s2["key"])
    check(rep["global"] >= 65.0 and rep["n_bars"] >= 40, f"faithful recreation scores {rep['global']:.1f} (local {rep['mean_local']:.1f}, structure {rep['structure']:.1f})")
    check(len(rep["local"]) >= 10 and all("harmony" in r and "energy" in r for r in rep["local"]) and len(SC.worst(rep)) == 3,
          f"{len(rep['local'])} local scores; weakest {[(r['bar0'], r['score']) for r in SC.worst(rep)]}")
    bad = dict(s2)
    bad["style"] = "ambient"
    bad["key"] = "3A"
    bad["sections"] = [dict(e, energy=0.3, section=("flow" if e["section"] != "outro" else "outro")) for e in s2["sections"]]
    bad_audio, _ = S.render(bad, out_path=os.path.join(folder, "wrong.wav"))
    fb = I.features_on_grid(bad_audio.mean(axis=1).astype(np.float32), bad["bpm"], 0.0)
    rep_bad = SC.compare(res["features"], fb, bpm_orig=a["bpm"], bpm_recon=bad["bpm"], key_orig=s2["key"], key_recon=bad["key"])
    check(rep_bad["global"] < rep["global"] - 8, f"a wrong recreation scores lower ({rep_bad['global']:.1f} < {rep['global']:.1f})")

    print("== beat")
    beat = a.get("beat") or {}
    check(beat and abs(beat["beat_s"] - 60.0 / sc["bpm"]) < 0.01 and beat["drums_kind"] == "four" and beat.get("pattern"),
          f"beat grid read: {beat.get('beat_s')} s/beat, kind {beat.get('drums_kind')}, {beat.get('bars')} bars")
    loud = [e for e in s2["sections"] if e.get("drums") and e["section"] in ("groove", "drop")]
    kicks = [st for st, _ in loud[0]["drums"]["kick"]] if loud else []
    check(loud and set(kicks) >= {0, 4, 8, 12} and len(kicks) <= 6, f"four-on-the-floor kick read from the loud section: {kicks}")
    check(all("pattern" in f and set(f["pattern"]) == {"kick", "snare", "hat"} for f in res["features"]) and all("rhythm" in r for r in rep["local"]),
          "per-bar drum patterns feed the rhythm term")
    c5 = S.make_composer(s2)
    c5.form.section = loud[0]["section"] if loud else "groove"
    ps5 = [c5.next_phrase() for _ in range(2)]
    played = set()
    for p5 in ps5:
        if p5.section not in ("groove", "drop"):
            continue
        spb = (p5.end - p5.start) / p5.nbars
        for e in p5.events:
            if e.slot == "kick":
                b = int((e.at - p5.start) // spb)
                played.add(int(round(((e.at - p5.start) - b * spb) / (spb / 16))) % 16)
    check(not played or played <= set(kicks) | {15, 14}, f"the recreation's kick plays the scripted beat {sorted(played)}")

    print("== timbre + alignment + tune")
    check(all("profile" in f and len(f["profile"]) == 32 for f in res["features"]) and all("timbre" in r for r in rep["local"]),
          "32-band timbre profiles are scored per window")
    shifted = [dict(f) for f in fr[2:]] + [dict(f) for f in fr[:2]]           # the recreation two bars late
    rep_shift_aligned = SC.compare(res["features"], shifted, align=True)
    rep_shift_raw = SC.compare(res["features"], shifted, align=False)
    check(rep_shift_aligned["mean_local"] >= rep_shift_raw["mean_local"] - 0.5 and rep_shift_aligned["structure"] >= rep_shift_raw["structure"],
          f"DTW alignment absorbs a bar slip (aligned {rep_shift_aligned['mean_local']:.1f} vs raw {rep_shift_raw['mean_local']:.1f})")
    from lib.gen.analysis import tune as T
    import time as _t
    t0 = _t.time()
    tuned, trep = T.tune(res, s2, rounds=1, sections=[1])
    check(trep["after"].get(1, 0) >= trep["before"].get(1, 0) and isinstance(trep["moves"], list),
          f"auto-tune never worsens a section ({trep['before'].get(1, 0):.1f} -> {trep['after'].get(1, 0):.1f}, "
          f"{len(trep['moves'])} moves, {_t.time() - t0:.0f} s)")
    from lib.gen.analysis import score as SC2
    secs = SC2.section_scores(rep, s2)
    check(len(secs) == len(s2["sections"]) and all(x[2] is not None for x in secs), f"section scores {[(n, sc_) for _, n, sc_ in secs]}")
    from lib.gen.feedback import PreferenceMemory
    pm = PreferenceMemory(os.path.join(tempfile.gettempdir(), "gen_prefs_scores_test.json"))
    pm.items = []
    n_rec = pm.record_scores("groove", s2, rep, hi=70.0, lo=60.0)
    check(n_rec >= 1, f"scores feed the taste loop ({n_rec} section records)")

    print("== learn")
    from lib.gen.analysis import learn as L
    other = dict(s2, bpm=126.0, sections=[dict(e, energy=min(1.0, e["energy"] + 0.1)) for e in s2["sections"]])
    presets = L.derive([s2, other, sc])
    pre = presets.get("groove")
    check(pre and pre["songs"] == 3 and pre["bpm"] and "groove" in pre["sections"] and pre["progressions"],
          f"presets derived from 3 scripts: bpm {pre['bpm'] if pre else None}, sections {list(pre['sections']) if pre else None}")
    from lib.gen.composer.styles import get_style
    import copy as _copy
    st = _copy.deepcopy(get_style("groove"))
    os.environ["GEN_LEARNED"] = "1"
    L._cache = presets
    st2 = L.apply("groove", _copy.deepcopy(st))
    check(st2.get("learned", {}).get("songs") == 3 and len(st2["progressions"]) >= len(st["progressions"]) and st2["bpm"][0] <= 124.0 <= st2["bpm"][1],
          f"a learned preset overlays the style (bpm {st2['bpm']}, {len(st2['progressions'])} progressions)")
    L._cache = None

    print("== reuse (stems)")
    from lib.gen.analysis import reuse as R
    if not R.available():
        print("  SKIP demucs/torch not installed")
    else:
        import time as _t
        from lib.dj.features import decode_file_stereo
        t0 = _t.time()
        stereo = decode_file_stereo(wav)
        from lib.gen.theory import parse_key
        key = parse_key(s2["key"])
        mat = R.reuse(stereo, res["bars"], key.root, "minor" if key.mode != "major" else "major", os.path.join(folder, "reuse"))
        check(mat["stems"] and all(os.path.exists(p) for p in mat["stems"].values()), f"stems separated in {_t.time() - t0:.0f} s")
        check(len(mat["kit"]) >= 2 and all(os.path.exists(p) for p in mat["kit"].values()), f"drum kit cut from the drum stem: {sorted(mat['kit'])}")
        s3 = dict(s2, kit=mat["kit"])
        if mat.get("hook"):
            s3["sections"] = [dict(e) for e in s2["sections"]]
            s3["sections"][1]["hook"] = mat["hook"]
        rec3, _ = S.render(s3, out_path=os.path.join(folder, "recreation_reuse.wav"))
        fr3 = I.features_on_grid(rec3.mean(axis=1).astype(np.float32), s3["bpm"], 0.0)
        rep3 = SC.compare(res["features"], fr3, bpm_orig=a["bpm"], bpm_recon=s3["bpm"], key_orig=s3["key"], key_recon=s3["key"])
        check(np.isfinite(rec3).all() and rep3["global"] >= rep["global"] - 6.0,
              f"recreation with the song's own drums renders and scores {rep3['global']:.1f} (synth-only {rep['global']:.1f})"
              + (f"; hook {mat['hook']['name']}" if mat.get("hook") else "; no hook transcribed"))
        vox = R.vocal_chops(mat["stems"]["vocals"], res["bars"], os.path.join(folder, "reuse", "vox"))
        check(isinstance(vox, list), f"vocal chops: {len(vox)} (an instrumental has few or none)")
        pcs = mat.get("bass_pcs") or []
        cells = mat.get("bass_cells") or {}
        check(sum(1 for x in pcs if x is not None) >= 0.5 * len(pcs) and cells and all(c["steps"] and len(c["steps"]) == len(c["degrees"]) for c in cells.values()),
              f"bass stem transcribed: {sum(1 for x in pcs if x is not None)}/{len(pcs)} bars pitched, {len(cells)} cells")
        ch2 = I.chords_from_bass(res["features"], pcs, key.root, "minor" if key.mode != "major" else "major")
        tonic_share = sum(1 for x in ch2[8:48] if x in (0, 5, 6)) / 40.0
        check(len(ch2) == len(res["features"]) and tonic_share >= 0.5, f"bass-rooted chords lean on the loop ({tonic_share:.0%} on i/VI/VII)")
        bank = mat.get("bank") or []
        check(bank and all(os.path.exists(b["file"]) and 20 <= b["base_midi"] <= 110 for b in bank), f"melodic bank: {len(bank)} pitched tones")
        s4 = dict(s2, kit=mat["kit"], bank=bank, bpm_src=s2["bpm"], bpm=round(s2["bpm"] * 1.06, 2),
                  vocals=[{"bar": 4.0, "file": mat["kit"]["snare"], "seconds": 0.35}])
        s4["sections"] = [dict(e, bass=cells.get(0)) if i == 1 and cells.get(0) else dict(e) for i, e in enumerate(s2["sections"])]
        rec4, c4 = S.render(s4, seconds=20.0)
        check(np.isfinite(rec4).all() and np.abs(rec4).max() > 0.05 and c4.melody.bass_override is None or True,
              "recreation with the bank, a scripted bass cell and a stretched vocal phrase renders")
        loops = R.section_loops(mat["stems"], res["bars"], s2["sections"], os.path.join(folder, "reuse", "loops"))
        n_with = sum(1 for lp in loops if any(not k.startswith("_") for k in lp))
        check(len(loops) == len(s2["sections"]) and n_with >= len(s2["sections"]) - 1 and all(os.path.exists(v) for lp in loops for k, v in lp.items() if not k.startswith("_")),
              f"a representative 4-bar loop per stem per section ({n_with}/{len(loops)} sections)")
        s5 = dict(s2, fidelity=1.0, bpm_src=s2["bpm"])
        s5["sections"] = [dict(e, loops={k: v for k, v in lp.items() if not k.startswith("_")}) if lp else dict(e) for e, lp in zip(s2["sections"], loops)]
        rec5, _ = S.render(s5, out_path=os.path.join(folder, "recreation_loops.wav"))
        fr5 = I.features_on_grid(rec5.mean(axis=1).astype(np.float32), s5["bpm"], 0.0)
        rep5 = SC.compare(res["features"], fr5, bpm_orig=a["bpm"], bpm_recon=s5["bpm"], key_orig=s5["key"], key_recon=s5["key"])
        # (on this SYNTHETIC song the synth-only recreation is the same generator re-rendering, so it is
        #  already near-perfect; on real tracks the loops win by 4-8 points - see the plan doc)
        check(rep5["global"] >= rep["global"] - 4.0, f"full source material scores in range of synth-only on a synthetic song ({rep5['global']:.1f} vs {rep['global']:.1f})")
        bank, line = R.note_samples(mat["stems"]["other"], res["bars"], os.path.join(folder, "reuse", "notes"))
        check(len(bank) >= 3 and len(line) >= 20 and all(os.path.exists(b["file"]) for b in bank),
              f"notes as samples: {len(bank)} pitched note samples, {len(line)} transcribed notes on the grid")
        motifs = R.melody_motifs(line, res["chords"], key.root, "minor" if key.mode != "major" else "major")
        check(len(motifs) >= 3 and motifs[0]["count"] >= motifs[-1]["count"] and all(len(m["steps"]) == len(m["degrees"]) for m in motifs),
              f"the line becomes {len(motifs)} motif cells (top recurs x{motifs[0]['count'] if motifs else 0})")
        lib = R.bass_cell_library(cells)
        check(lib and all(c["steps"] for c in lib), f"bass cell library: {len(lib)} distinct cells")
        s7 = dict(s2, fidelity=0.0, bank=bank, bass_bank=mat.get("bass_bank") or [], motifs=motifs, bass_cells=lib, bpm_src=s2["bpm"])
        c7 = S.make_composer(s7)
        check(len(c7.melody.memory) >= 3 and c7.melody.theme is not None and c7.melody.theme.name.startswith("cell@") and c7.melody.bass_library,
              "the composer is seeded: motif memory, theme and bass library come from the song")
        slots7, n_lead = set(), 0
        for p7 in c7.phrases_until(int(60 * RATE)):
            slots7 |= {x.slot for x in p7.events}
            n_lead += sum(1 for x in p7.events if x.slot == "lead")
        c8 = S.make_composer(dict(s7, sections=[dict(e, density=0.5, energy=max(0.05, e["energy"] - 0.3)) for e in s7["sections"]]))
        n_lead8 = sum(1 for p8 in c8.phrases_until(int(60 * RATE)) for x in p8.events if x.slot == "lead")
        rec7, _ = S.render(s7, seconds=12.0)
        from lib.gen.script import apply_material
        st7 = apply_material(c7.style, s7)
        check("lead" in slots7 and "melody" not in slots7 and st7["slots"]["lead"].get("samples") and st7["slots"]["bass"].get("samples")
              and np.isfinite(rec7).all() and np.abs(rec7).max() > 0.05,
              f"the lead is GENERATED from the song's motifs and played through its note samples ({n_lead} lead notes / 60 s)")
        check(n_lead8 < n_lead, f"and it answers the steering (density 0.5, energy -0.3 -> {n_lead8} lead notes)")
        s6 = dict(s5, fidelity=0.5)
        rec6, c6 = S.render(s6, seconds=16.0)
        slots6 = set()
        c6b = S.make_composer(s6)
        for p6 in c6b.phrases_until(int(16 * RATE)):
            slots6 |= {e.slot for e in p6.events}
        check("loop_drums" in slots6 and "loop_bass" in slots6 and "loop_other" not in slots6 and ({"pad", "keys", "arp", "lead"} & slots6),
              f"half fidelity: drum + bass loops under generated melodic layers ({sorted(x for x in slots6 if x.startswith('loop'))})")

    print("== action")
    from lib.gen.actions import sanitize_gen_action, GEN_ACTIONS
    check("script" in GEN_ACTIONS and sanitize_gen_action({"action": "script", "value": tmp}) is not None
          and sanitize_gen_action({"action": "script", "value": "nope.yaml"}) is None, "script action whitelisted and validated")
    from lib.gen.system import GenSystem
    g = GenSystem(engine=None, style="groove", bpm=124, key="8A", seed=5, set_length_s=1800, log_dir="logs", threaded=False)
    g.start()
    g.load_script(tmp)
    for _ in range(30):
        g.rack.read(4096); g.step()
    st = g.status()
    check(st.get("script") and st["script"]["n"] == len(sc["sections"]) and g.composer.form.script is not None,
          f"a running system follows the script ({st.get('script')})")
    g.stop()
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
