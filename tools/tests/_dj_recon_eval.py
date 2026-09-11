"""Evaluate the programmatic reconstruction (lib/dj/resynth.render) of
library tracks against the real stems - the critical instrument, not a
score that flatters.

Per track and per stem, the stem's instruments are rendered from the
reading and compared with the stem itself:
    level     rms difference (dB)
    spectral  mean |dB| on a 64-band log-mel, per beat, over beats where
              the stem sounds (timbre + notes + balance in one number;
              0 would be identical, 6-8 is "the same part on another
              synth", 12+ is a different sound)
    activity  Pearson r between 16th-note rms envelopes - the RHYTHM
              of the part; then the share of active 16ths the recon
              MISSES (stem sounds, recon silent) and adds SPURIOUSLY
              (recon sounds, stem silent)
    onsets    F1 of onset times within 40 ms (drums / plucked parts)
    lag       the envelope cross-correlation lag (ms; a positive value
              means the recon is late)
    chroma    per-beat chroma correlation on active beats (pitched
              stems: the notes, octave-blind)
plus the full mix. Usage:
    python tools/tests/_dj_recon_eval.py 1 51 101        # library ids
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.dj import instruments as INS, resynth as RS      # noqa: E402
from lib.dj import resolve_music_dir                       # noqa: E402

from lib.dj import fidelity as F                           # noqa: E402

RATE = F.RATE
HOP = F.HOP
# the measure itself lives in lib/dj/fidelity.py (shared with the reading's check and the gen console)
_db, _mel_db, _env16, _onsets, _f1, _chroma_beats, _lag_ms, _compare = F.db, F.mel_db, F.env16, F.onsets, F.f1, F.chroma_beats, F.lag_ms, F.compare


PROGRAM = "--program" in sys.argv          # render from the SongProgram instead of the raw reading
EXPLAIN = "--explain" in sys.argv          # with --program: prune/merge voices by synthesis first (lib/dj/explain.py)
_PROGS = {}


def _render(res, stems, ids, t0, t1, tid):
    if not PROGRAM:
        return RS.render(res, stems, ids=ids, t0=t0, t1=t1, use_stems=())
    from lib.dj import songprogram as SP
    if tid not in _PROGS:
        _PROGS[tid] = SP.build(res, stems, explain=EXPLAIN)          # vocals as reused phrases, `other` residual bars marked
        d = SP.describe(_PROGS[tid]).splitlines()
        print("   program:", d[0])
        for line in d[1:]:
            if "PRUNED" in line or "MERGED" in line:
                print("  ", line)
    prog = _PROGS[tid]
    ids = [i for i in ids if i in prog["voices"]] or ids      # pruned/merged voices are gone from the program
    # voices only: the verbatim residual bars would flatter the "other" row (their share is printed above)
    return SP.render(prog, stems, ids=ids, t0=t0, t1=t1, use_residual=False, limit=None)   # levels as rendered, no peak clamp


def evaluate(root, tid, t0=None, t1=None):
    res = INS.load(root, tid)
    if res is None:
        return None
    stems = RS.load_stems_mono(root, tid)
    beats = np.asarray(res["beats"], dtype=np.float64)
    period = float(res["period_s"])
    dur = min(len(y) for y in stems.values()) / RATE
    t0 = 0.0 if t0 is None else t0
    t1 = dur if t1 is None else min(t1, dur)
    sel = (beats >= t0) & (beats < t1 - period)
    bts = beats[sel]
    rows = {}
    mix_ref = np.zeros(int((t1 - t0) * RATE), dtype=np.float32)
    mix_rec = np.zeros_like(mix_ref)
    for stem in INS.STEM_ORDER:
        ids = [i["id"] for i in INS.instruments(res) if i["stem"] == stem]
        ref = stems[stem][int(t0 * RATE):int(t1 * RATE)]
        if not ids or _db(ref) < -60:
            rows[stem] = None
            continue
        rec = _render(res, stems, ids, t0, t1, tid)[:, 0][: len(ref)]
        if len(rec) < len(ref):
            rec = np.concatenate([rec, np.zeros(len(ref) - len(rec), dtype=np.float32)])
        mix_ref[: len(ref)] += ref
        mix_rec[: len(rec)] += rec
        rows[stem] = _compare(ref, rec, bts - t0, period, pitched=stem != "drums")
    rows["mix"] = _compare(mix_ref, mix_rec, bts - t0, period, pitched=True)
    return rows


def main():
    root = resolve_music_dir(os.environ.get("DJ_MUSIC", ""))
    ids = [int(a) for a in sys.argv[1:] if a.isdigit()] or [1]
    print("mode:", "SongProgram" if PROGRAM else "raw reading (resynth)")
    from lib.dj.db import LibraryDB
    db = LibraryDB(root)
    titles = {r["id"]: (r.get("title") or "")[:26] for r in db.all_tracks()}
    db.close()
    hdr = f"{'stem':6s} {'level':>6s} {'spect':>6s} {'act r':>6s} {'missed':>6s} {'spur':>6s} {'lag':>6s} {'onF1':>5s} {'off':>5s} {'sd':>4s} {'chroma':>6s}"
    agg = {}
    for tid in ids:
        rows = evaluate(root, tid, t0=30.0, t1=150.0)
        if rows is None:
            print(f"== {tid}: no reading")
            continue
        print(f"== {tid} {titles.get(tid, '')}\n   {hdr}")
        for stem, r in rows.items():
            if r is None:
                print(f"   {stem:6s} (silent / no instruments)")
                continue
            ch = r.get("chroma_r", float("nan"))
            tag = "  (stem passthrough)" if stem == "vocals" and r["spectral"] < 0.5 else ("  (reused phrases)" if stem == "vocals" and PROGRAM else "")
            print(f"   {stem:6s} {r['level']:+6.1f} {r['spectral']:6.1f} {r['activity_r']:6.2f} {r['missed']:6.2f} {r['spurious']:6.2f} "
                  f"{r['lag_ms']:+6.0f} {r['onset_f1']:5.2f} {r['onset_off_ms']:+5.0f} {r['onset_off_sd']:4.0f} {ch:6.2f}{tag}")
            for k, v in r.items():
                agg.setdefault((stem, k), []).append(v)
    print("\n== medians over tracks")
    print(f"   {hdr}")
    for stem in list(INS.STEM_ORDER) + ["mix"]:
        if (stem, "level") not in agg:
            continue
        g = lambda k: float(np.nanmedian(agg.get((stem, k), [np.nan])))
        print(f"   {stem:6s} {g('level'):+6.1f} {g('spectral'):6.1f} {g('activity_r'):6.2f} {g('missed'):6.2f} {g('spurious'):6.2f} "
              f"{g('lag_ms'):+6.0f} {g('onset_f1'):5.2f} {g('onset_off_ms'):+5.0f} {g('onset_off_sd'):4.0f} {g('chroma_r'):6.2f}")


if __name__ == "__main__":
    main()
