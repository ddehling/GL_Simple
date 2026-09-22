"""The composer census (docs/DJ_THE_COMPOSER.md): over the stem library, how many events each record gets from
its own lists and structure at MOMENTS some (2) and lots (4) - the density the dial can actually deliver. No
audio, seconds.

    python tools/tests/_dj_compose_census.py --music D:/Devel/music
"""
import os
import sys
from collections import Counter

os.environ.setdefault("DJ_NO_PLAY_LOG", "1")
os.environ.setdefault("DJ_NO_MIDI", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.tests._low_priority import PRIORITY  # noqa: E402


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    music = arg("--music", "D:/Devel/music")
    from lib.dj import director as D
    from lib.dj.composer import compose
    d = D.Director(music, engine=None, theme="groove")
    recs = (d._pairings_file() or {}).get("records") or {}
    lib = [t for t in d.library if getattr(t, "has_stems", False) and t.bpm and str(t.id) in recs]
    print(f"{len(lib)} records with pairing lists")
    for n in (2, 4):
        counts, kinds, leads = Counter(), Counter(), Counter()
        for t in lib:
            rec = recs.get(str(t.id)) or {}
            secs = t.sections or []
            entry = next((float(s["start_s"]) for s in secs if s.get("kind") != "intro"), 0.0)
            evs = compose(t, rec, entry, 8, 4.0 * float(t.period_s), n, next_tid=None)
            evs = [e for e in evs if not e.get("exit")]
            counts[len(evs)] += 1
            for e in evs:
                kinds[e["kind"]] += 1
            if evs:
                leads[evs[0]["kind"]] += 1
        total = sum(counts.values())
        print(f"\nMOMENTS {'some' if n == 2 else 'lots'} ({n} asked): events per record " + ", ".join(f"{k}: {v} ({100 * v / max(total, 1):.0f} %)" for k, v in sorted(counts.items())))
        print(f"  kinds {dict(kinds)}; leads {dict(leads)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
