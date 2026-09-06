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
    check(sum(1 for x in res["chords"][8:48] if x in (0, 5, 6)) >= 0.5 * 40, f"chords lean on the loop's degrees {res['chords'][8:20]}")

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
