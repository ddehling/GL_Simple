"""Song analysis CLI (lib/gen/analysis): ingest a song into the
generator's own command language, recreate it, score the recreation.

    python tools/gen/analyze.py ingest song.wav                 # -> logs/analysis/<name>/script.yaml + actions.txt + features
    python tools/gen/analyze.py recreate logs/analysis/<name>   # renders recreation.wav from script.yaml (edit the yaml first if you like)
    python tools/gen/analyze.py score logs/analysis/<name>      # per-phrase and global scores -> score.json + a text report
    python tools/gen/analyze.py all song.wav                    # the three in a row
    python tools/gen/analyze.py tune logs/analysis/<name>       # auto-tune the script against the score -> script_tuned.yaml + recreation_tuned.wav
    python tools/gen/analyze.py batch <folder> [--reuse]        # ingest a whole folder, then learn style presets from every script
    python tools/gen/analyze.py learn                           # (re)derive lib/gen/composer/data/learned_styles.json from logs/analysis/*/script.yaml
    python tools/gen/analyze.py play logs/analysis/<name>       # send the script to the running show (POST /api/gen/action)

Scores: local = per 4-bar window (energy, spectrum, rhythm, harmony),
global = local mean + structure (energy envelopes rise and fall
together) + tempo + key. The song against itself is 100.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def folder_for(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return os.path.join("logs", "analysis", name)


def do_ingest(path, out=None, reuse=False):
    from lib.gen import script as S
    from lib.gen.analysis import ingest as I
    out = out or folder_for(path)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    res = I.ingest(path, progress=lambda p, what: print(f"  {p:4.0%} {what}", flush=True), reuse=reuse, out_dir=out)
    if reuse:
        for r in res["analysis"].get("reuse_reasons", []):
            print("  note:", r)
    S.save(res["script"], os.path.join(out, "script.yaml"))
    with open(os.path.join(out, "features.json"), "w", encoding="utf-8") as fh:
        json.dump({"features": res["features"], "bars": res["bars"], "chords": res["chords"],
                   "analysis": {k: v for k, v in res["analysis"].items() if k != "sections"},
                   "sections": res["analysis"]["sections"], "source": os.path.abspath(path)}, fh)
    with open(os.path.join(out, "actions.txt"), "w", encoding="utf-8") as fh:
        for bar, action, value in S.to_actions(res["script"]):
            fh.write(f"bar {bar:4d}  {action:10s} {json.dumps(value) if not isinstance(value, str) else value}\n")
    print(f"ingested in {time.time() - t0:.1f} s -> {out}")
    print(S.describe(res["script"]))
    a = res["analysis"]
    print(f"bpm {a['bpm']:.2f} (conf {a['bpm_conf']:.2f})  key {a['key']} ({a['camelot']}, conf {a['key_conf']:.2f})  "
          f"sections {a['n_sections']}  swing {a['rhythm'].get('swing')}")
    return out


def do_recreate(folder, seed=None):
    from lib.gen import script as S
    sc = S.load(os.path.join(folder, "script.yaml"))
    t0 = time.time()
    audio, c = S.render(sc, out_path=os.path.join(folder, "recreation.wav"), seed=seed, stems=True,
                        progress=lambda p: print(f"  {p:4.0%}", end="\r", flush=True))
    trims = [e.get("trim_db") for e in (c.script or {}).get("sections", [])]
    if any(t is not None for t in trims) and not any(e.get("trim_db") is not None for e in sc["sections"]):
        for e, t in zip(sc["sections"], trims):
            e["trim_db"] = t
        sc["master_db"] = float((c.script or {}).get("master_db") or 0.0)
        S.save(sc, os.path.join(folder, "script.yaml"))          # keep the level calibration with the script
        print(f"  level calibration: {' '.join(f'{t:+.1f}' for t in trims if t is not None)} dB per section, master {sc['master_db']:+.1f} dB (saved)")
    print(f"recreated {audio.shape[0] / 44100:.0f} s in {time.time() - t0:.1f} s -> {folder}/recreation.wav")
    return folder


def do_score(folder):
    import numpy as np
    from lib.gen import script as S
    from lib.gen.analysis import ingest as I, score as SC
    from lib.dj.features import decode_file
    with open(os.path.join(folder, "features.json"), encoding="utf-8") as fh:
        saved = json.load(fh)
    sc = S.load(os.path.join(folder, "script.yaml"))
    rec = decode_file(os.path.join(folder, "recreation.wav"))
    fr = I.features_on_grid(rec, sc["bpm"], 0.0)
    a = saved["analysis"]
    rep = SC.compare(saved["features"], fr, bpm_orig=a["bpm"], bpm_recon=sc["bpm"], key_orig=sc["key"], key_recon=sc["key"])
    with open(os.path.join(folder, "score.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    lines = [f"global {rep['global']:.1f}   (local mean {rep['mean_local']:.1f}, structure {rep['structure']:.1f}, "
             f"tempo {rep['tempo']:.0f}, key {rep['key']:.0f}, {rep['n_bars']} bars)",
             f"{'bar':>5s} {'t':>6s} {'score':>6s} {'energy':>7s} {'spect':>6s} {'rhythm':>7s} {'harm':>6s}"]
    for r in rep["local"]:
        lines.append(f"{r['bar0']:5d} {r['t']:6.1f} {r['score']:6.1f} {r['energy']:7.1f} {r['spectrum']:6.1f} {r['rhythm']:7.1f} {r['harmony']:6.1f}")
    lines.append("weakest: " + ", ".join(f"bar {r['bar0']} ({r['score']:.0f})" for r in SC.worst(rep)))
    stems = {n: os.path.join(folder, "stems", n + ".wav") for n in ("drums", "bass", "other", "vocals")}
    if all(os.path.exists(p) for p in stems.values()) and not os.environ.get("GEN_NO_STEM_SCORE"):
        # the strict measure: the recreation's own stems against the original's, beat by beat
        for p in (os.path.join(folder, "recon_stems", n + ".wav") for n in stems):
            if os.path.exists(p):
                os.remove(p)
        sf_rep = SC.stem_fidelity(stems, os.path.join(folder, "recreation.wav"), float(a.get("first_bar_s", 0.0)), sc["bpm"],
                                  os.path.join(folder, "recon_stems"), n_bars=len(saved["features"]))
        rep["stems"] = sf_rep
        with open(os.path.join(folder, "score.json"), "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
        # the mix, identified: fold the measured level differences into the script's per-stem trims (recreate again to apply)
        mix = dict(sc.get("mix_db") or {})
        for n, v in sf_rep.items():
            if isinstance(v, dict) and v.get("level_db") is not None and abs(v["level_db"]) >= 0.5:
                mix[n] = round(float(max(-18.0, min(18.0, mix.get(n, 0.0) - v["level_db"]))), 1)
        if mix != (sc.get("mix_db") or {}):
            sc["mix_db"] = mix
            S.save(sc, os.path.join(folder, "script.yaml"))
            lines.append("mix trims saved to the script (dB): " + "  ".join(f"{k} {v:+.1f}" for k, v in mix.items()) + "  -> recreate again to apply")
        lines.append("stems (fine log-mel per beat, mean |dB|; 0 = identical): " + "  ".join(
            f"{n} {v['db']:.1f} dB (level {v['level_db']:+.1f}, activity r {v['corr']:.2f})" for n, v in sf_rep.items() if isinstance(v, dict))
                     + f"  | mean {sf_rep['mean_db']:.1f} dB")
    with open(os.path.join(folder, "score.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:1] + lines[-2:]))
    return rep


def do_tune(folder, rounds=2):
    """Auto-tune script.yaml against the saved original features; writes
    script_tuned.yaml + recreation_tuned.wav + score_tuned.json."""
    import numpy as np
    from lib.gen import script as S
    from lib.gen.analysis import ingest as I, score as SC, tune as T
    with open(os.path.join(folder, "features.json"), encoding="utf-8") as fh:
        saved = json.load(fh)
    sc = S.load(os.path.join(folder, "script.yaml"))
    res = {"features": saved["features"], "analysis": saved["analysis"]}
    t0 = time.time()
    tuned, rep = T.tune(res, sc, rounds=rounds, progress=lambda p, what: print(f"  {p:4.0%} {what}", flush=True))
    S.save(tuned, os.path.join(folder, "script_tuned.yaml"))
    audio, _ = S.render(tuned, out_path=os.path.join(folder, "recreation_tuned.wav"))
    fr = I.features_on_grid(audio.mean(axis=1).astype(np.float32), tuned["bpm"], 0.0)
    a = saved["analysis"]
    score = SC.compare(saved["features"], fr, bpm_orig=a["bpm"], bpm_recon=tuned["bpm"], key_orig=tuned["key"], key_recon=tuned["key"])
    with open(os.path.join(folder, "score_tuned.json"), "w", encoding="utf-8") as fh:
        json.dump({"score": score, "tune": rep}, fh, indent=1)
    print(f"tuned in {time.time() - t0:.0f} s: {len(rep['moves'])} moves, mean section gain {rep['gain']:+.1f} -> global {score['global']:.1f}")
    for m in rep["moves"][:12]:
        print(f"   section {m['section']} {m['lever']}: {m['from']} -> {m['to']} ({m['gain']:+.1f})")
    try:
        from lib.gen.feedback import PreferenceMemory
        n = PreferenceMemory(os.path.join("logs", "gen_prefs.json")).record_scores(tuned["style"], tuned, score)
        print(f"   taste: {n} section records from the scores")
    except Exception as e:  # noqa: BLE001
        print(f"   taste not recorded ({e})")
    return score


def do_batch(folder_in, reuse=False):
    """Ingest every audio file in a folder; then learn presets from all scripts under logs/analysis."""
    exts = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif")
    files = sorted(f for f in os.listdir(folder_in) if f.lower().endswith(exts))
    print(f"{len(files)} songs in {folder_in}")
    for f in files:
        try:
            do_ingest(os.path.join(folder_in, f), None, reuse)
        except Exception as e:  # noqa: BLE001
            print(f"  {f}: failed ({type(e).__name__}: {e})")
    return do_learn()


def do_learn():
    from lib.gen import script as S
    from lib.gen.analysis import learn as L
    root = os.path.join("logs", "analysis")
    scripts = []
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        p = os.path.join(root, name, "script.yaml")
        if os.path.exists(p):
            try:
                scripts.append(S.load(p))
            except Exception:
                pass
    presets = L.derive(scripts)
    path = L.save(presets)
    print(f"learned presets from {len(scripts)} scripts -> {path}")
    for style, pre in presets.items():
        print(f"  {style:10s} songs {pre['songs']}  bpm {pre['bpm']}  swing {pre['swing']}  sections {list(pre['sections'])}  progressions {pre['progressions'][:3]}")
    return presets


def do_play(folder, base="http://localhost:5000"):
    import urllib.request
    path = os.path.abspath(os.path.join(folder, "script.yaml"))
    data = json.dumps({"action": "script", "value": path}).encode("utf-8")
    req = urllib.request.Request(base + "/api/gen/action", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        print(r.read().decode("utf-8")[:200])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["ingest", "recreate", "score", "tune", "all", "play", "batch", "learn"])
    ap.add_argument("target", nargs="?", default="")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--show", default="http://localhost:5000")
    ap.add_argument("--reuse", action="store_true", help="separate stems and reuse the song's drums, vocals and hook")
    args = ap.parse_args()
    if args.cmd == "ingest":
        do_ingest(args.target, args.out, args.reuse)
    elif args.cmd == "recreate":
        do_recreate(args.target, args.seed)
    elif args.cmd == "score":
        do_score(args.target)
    elif args.cmd == "tune":
        do_tune(args.target, args.rounds)
    elif args.cmd == "all":
        folder = do_ingest(args.target, args.out, args.reuse)
        do_recreate(folder, args.seed)
        do_score(folder)
        do_tune(folder, args.rounds)
    elif args.cmd == "batch":
        do_batch(args.target, args.reuse)
    elif args.cmd == "learn":
        do_learn()
    else:
        do_play(args.target, args.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
