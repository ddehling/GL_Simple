"""Song analysis CLI (lib/gen/analysis): ingest a song into the
generator's own command language, recreate it, score the recreation.

    python tools/gen/analyze.py ingest song.wav                 # -> logs/analysis/<name>/script.yaml + actions.txt + features
    python tools/gen/analyze.py recreate logs/analysis/<name>   # renders recreation.wav from script.yaml (edit the yaml first if you like)
    python tools/gen/analyze.py score logs/analysis/<name>      # per-phrase and global scores -> score.json + a text report
    python tools/gen/analyze.py all song.wav                    # the three in a row
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
    audio, c = S.render(sc, out_path=os.path.join(folder, "recreation.wav"), seed=seed,
                        progress=lambda p: print(f"  {p:4.0%}", end="\r", flush=True))
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
    with open(os.path.join(folder, "score.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:1] + lines[-1:]))
    return rep


def do_play(folder, base="http://localhost:5000"):
    import urllib.request
    path = os.path.abspath(os.path.join(folder, "script.yaml"))
    data = json.dumps({"action": "script", "value": path}).encode("utf-8")
    req = urllib.request.Request(base + "/api/gen/action", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        print(r.read().decode("utf-8")[:200])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["ingest", "recreate", "score", "all", "play"])
    ap.add_argument("target")
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
    elif args.cmd == "all":
        folder = do_ingest(args.target, args.out, args.reuse)
        do_recreate(folder, args.seed)
        do_score(folder)
    else:
        do_play(args.target, args.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
