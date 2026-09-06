"""VST3 plugin library tool for the generative rack (lib/gen/synth/plugins.py).

    python tools/gen/plugins.py scan media/plugins              # write plugins.json from the .vst3 bundles in a folder
    python tools/gen/plugins.py scan "C:/Program Files/Common Files/VST3" --out media/plugins   # scan a system folder, manifest in ours
    python tools/gen/plugins.py list                            # names the styles can reference as vst:<name>
    python tools/gen/plugins.py test dexed [--program 3]        # render a chord, print level / speed / programs
    python tools/gen/plugins.py programs dexed                  # list the plugin's programs (presets it exposes)

Scan probes each bundle with pedalboard to learn whether it is an
instrument or an effect; bundles that fail to load are listed but
skipped. Names are the bundle stem, lower-cased.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def scan(folder, out=None):
    from lib.gen.synth.plugins import binary_path
    from pedalboard import VST3Plugin
    out = out or folder
    entries = {}
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith(".vst3"):
            continue
        path = os.path.join(folder, fn)
        name = os.path.splitext(fn)[0].lower().replace(" ", "_")
        try:
            p = VST3Plugin(binary_path(path))
            kind = "instrument" if p.is_instrument else "effect"
            rel = os.path.relpath(path, out) if os.path.abspath(out) in os.path.abspath(path) else os.path.abspath(path)
            entries[name] = {"path": rel.replace("\\", "/"), "kind": kind, "tags": []}
            print(f"  {name:24s} {kind:10s} {p.name}")
        except Exception as e:  # noqa: BLE001
            print(f"  {name:24s} SKIP ({type(e).__name__}: {str(e)[:80]})")
    os.makedirs(out, exist_ok=True)
    mpath = os.path.join(out, "plugins.json")
    old = {}
    try:
        with open(mpath, encoding="utf-8") as fh:
            old = json.load(fh)
    except Exception:
        pass
    for k, v in entries.items():                       # keep hand-edited program/params/preset
        if k in old:
            for keep in ("program", "params", "preset", "tags"):
                if keep in old[k]:
                    v[keep] = old[k][keep]
    old.update(entries)
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(old, fh, indent=1)
    print(f"wrote {mpath}: {len(old)} plugins")
    return old


def test(name, program=None):
    import numpy as np
    from lib.gen import RATE
    from lib.gen.events import NoteEvent
    from lib.gen.synth import plugins
    inst = plugins.instrument({"plugin": f"vst:{name}", "program": program})
    if inst is None:
        print(f"cannot load vst:{name} (pedalboard {'present' if plugins.available() else 'MISSING'}, manifest {plugins.names()})")
        return 1
    evs = [NoteEvent(0, "x", 60.0, 0.9, RATE, {}), NoteEvent(RATE // 2, "x", 64.0, 0.8, RATE, {}), NoteEvent(RATE, "x", 67.0, 0.8, RATE, {})]
    t0 = time.perf_counter()
    a = inst.render(evs, 0, 3.0)
    dt = time.perf_counter() - t0
    rms = 20 * np.log10(float(np.sqrt(np.mean(a ** 2))) + 1e-9)
    print(f"{inst.name}: program {getattr(inst.plugin, 'program', None)!r}, latency {inst.latency} smp, "
          f"render {dt * 1000:.0f} ms for 3 s ({3 / max(dt, 1e-6):.0f}x realtime), peak {np.abs(a).max():.3f}, rms {rms:.1f} dBFS")
    return 0


def programs(name):
    from lib.gen.synth import plugins
    inst = plugins.instrument({"plugin": f"vst:{name}"})
    if inst is None:
        print("cannot load")
        return 1
    p = inst.plugin
    try:
        n = p.num_programs if hasattr(p, "num_programs") else None
    except Exception:
        n = None
    print(f"{inst.name}: current program {getattr(p, 'program', None)!r}; num_programs {n}")
    print("parameters:", ", ".join(list(p.parameters.keys())[:40]), "..." if len(p.parameters) > 40 else "")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["scan", "list", "test", "programs"])
    ap.add_argument("arg", nargs="?", default=os.path.join("media", "plugins"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--program", default=None)
    args = ap.parse_args()
    if args.cmd == "scan":
        scan(args.arg, args.out)
    elif args.cmd == "list":
        from lib.gen.synth import plugins
        for folder, man in plugins.manifests():
            print(folder)
            for k, v in man.items():
                print(f"  vst:{k:22s} {v.get('kind', 'instrument'):10s} {v.get('path')}  program={v.get('program')}")
    elif args.cmd == "test":
        return test(args.arg, int(args.program) if args.program and args.program.isdigit() else args.program)
    else:
        return programs(args.arg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
