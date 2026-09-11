"""The bridge from the DJ planner's instrument reading to the generative
console's Analysis tab.

link(root, track_id) takes a library track (stems on disk; the
instrument pass is run if there is no current reading) and writes
everything the gen console's Analysis tab opens as one folder,
logs/analysis/<title>/:

    script.yaml     the reading as a SongScript (lib/dj/resynth.export_gen:
                    kit one-shots, note banks, pad, per-bar drum grids,
                    melody / bass lines, sections)
    *.wav           the samples the script plays
    original.wav    the source, decoded (the A side of the compare view
                    and the score's reference)
    features.json   the per-bar features, bar times and analysis facts
    fidelity.json   how programmatic the program is and how close its
                    render is to the original stems (lib/dj/fidelity.py)
                    the tab's strip, compare view and scorer read

Both the planner's "-> Gen" button and the gen console's "DJ track"
button come through here, so the two directions produce the same folder.
"""
import json
import os
import re

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RATE = 44100


def folder_for(title, track_id):
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(title or ""))[:60] or f"track_{track_id}"
    return os.path.join(_REPO_ROOT, "logs", "analysis", name)


def tracks_with_stems(root):
    """[(id, title, artist, has_reading)] for the picker, sorted by title."""
    from lib.dj.db import LibraryDB
    from lib.dj import stems as ST, instruments as INS
    db = LibraryDB(root)
    rows = [r for r in db.all_tracks() if not r.get("missing") and not r.get("error")]
    db.close()
    out = []
    for r in rows:
        if not ST.has_stems(root, r["id"]):
            continue
        out.append((r["id"], r.get("title") or r["path"], r.get("artist") or "", INS.has_instruments(root, r["id"])))
    return sorted(out, key=lambda t: (t[1].lower(), t[2].lower()))


def link(root, track_id, out_dir=None, stems=None, progress=None, write_original=True):
    """-> the folder written. stems: {name: (n,) or (n,2) float arrays}
    already in memory (the planner's), else decoded from disk."""
    from lib.dj.db import LibraryDB
    from lib.dj import instruments as INS, resynth as RS
    from lib.dj.features import decode_file_stereo
    db = LibraryDB(root)
    row = next((r for r in db.all_tracks() if r["id"] == int(track_id)), None)
    if row is None:
        db.close()
        raise KeyError(f"no track {track_id} in {root}")
    sections = db.sections_for(row["id"])
    src_path = db.abs(row["path"])
    db.close()
    title = row.get("title") or f"track_{track_id}"
    out_dir = out_dir or folder_for(title, track_id)
    os.makedirs(out_dir, exist_ok=True)
    res = INS.load(root, row["id"])
    if res is None:
        if progress:
            progress("identifying instruments (no current reading)")
        res = INS.identify_track(root, row["id"], row.get("beat_grid") or [], row.get("downbeat_offset") or 0,
                                 row.get("duration_s") or 0.0, bpm=row.get("bpm"), progress=progress)
    if stems is None:
        if progress:
            progress("decoding stems")
        stems = RS.load_stems_mono(root, row["id"])
        if stems is None:
            raise FileNotFoundError("no stems on disk for this track")
    else:
        stems = {n: RS._mono(a) for n, a in stems.items()}
    # the SongProgram first: its chord track feeds the script's sections
    prog = None
    try:
        from lib.dj import songprogram as SP
        if progress:
            progress("building the song program")
        prog = SP.build(res, stems, sections=[{k: v for k, v in s.items() if k in ("kind", "start_s", "end_s", "energy")} for s in sections],
                        progress=progress)
        SP.save(prog, out_dir)
        if progress:
            progress("program: " + SP.describe(prog).splitlines()[0][:160])
    except Exception as e:  # noqa: BLE001
        if progress:
            progress(f"program build failed ({type(e).__name__}: {str(e)[:80]}) - replay falls back to the raw reading")
    # the two honest figures the gen tab shows: how much of the notes the program language holds,
    # and how close its render is to the original stems (lib/dj/fidelity.py, the evaluator's measure)
    if prog is not None:
        try:
            from lib.dj import fidelity as F
            rep = F.report(prog, stems, res, progress=progress)
            F.save(rep, out_dir)
            if progress:
                w = rep["waveform"].get("mix") or {}
                progress(f"fidelity: mix gap {w.get('spectral', float('nan')):.1f} dB; {rep['notes']['events_per_entry']:.1f} events per pattern+op")
        except Exception as e:  # noqa: BLE001
            if progress:
                progress(f"fidelity report skipped ({type(e).__name__}: {str(e)[:80]})")
    if progress:
        progress("writing samples and script")
    script_path = RS.export_gen(res, stems, out_dir, title=title, bpm=row.get("bpm"), key=row.get("camelot") or "8A",
                                sections=sections, progress=progress, chords=(prog or {}).get("chords"))
    # the source and its per-bar features: the tab's A side, strip and scorer
    if progress:
        progress("decoding the source for the compare view")
    stereo = decode_file_stereo(src_path)
    source = src_path
    if write_original:
        import soundfile as sf
        source = os.path.join(out_dir, "original.wav")
        sf.write(source, np.clip(stereo, -1.0, 1.0), RATE, subtype="PCM_16")
    beats = np.asarray(res["beats"], dtype=np.float64)
    down0 = int(res.get("down0", 0))
    first_bar_s = float(beats[down0]) if len(beats) > down0 else 0.0
    bpm = float(row.get("bpm") or 60.0 / float(res.get("period_s") or 0.5))
    try:
        from lib.gen.analysis import ingest as I
        feats = I.features_on_grid(stereo.mean(axis=1).astype(np.float32), bpm, first_bar_s)
    except Exception as e:  # noqa: BLE001
        feats = []
        if progress:
            progress(f"features skipped ({type(e).__name__}: {str(e)[:60]})")
    bars = [round(float(t), 4) for t in beats[down0::4]]
    analysis = {"bpm": bpm, "bpm_conf": float(row.get("bpm_conf") or 0.0), "key_conf": float(row.get("key_conf") or 0.0),
                "duration_s": float(row.get("duration_s") or len(stereo) / RATE), "first_bar_s": first_bar_s,
                "downbeat_conf": float(row.get("downbeat_conf") or 0.0), "camelot": row.get("camelot"),
                "dj_track_id": int(row["id"]), "dj_library": os.path.abspath(root),
                "instruments": [f"{i['id']} {INS.display_name(i)}: {INS.facts(i)}" for i in INS.instruments(res)]}
    with open(os.path.join(out_dir, "instruments.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, separators=(",", ":"))          # the reading itself

    with open(os.path.join(out_dir, "features.json"), "w", encoding="utf-8") as fh:
        json.dump({"features": feats, "bars": bars, "chords": [], "analysis": analysis,
                   "sections": [{k: v for k, v in s.items() if k in ("kind", "start_s", "end_s", "energy", "label")} for s in sections],
                   "source": os.path.abspath(source)}, fh)
    if progress:
        progress(f"linked: {script_path}")
    return out_dir


def replay(folder, out_path=None, ids=None, progress=None):
    """The PROGRAMMATIC recreation of a linked track: every event of the
    reading (instruments.json in the folder) played by its instrument's
    exemplar cut from the library's stems (lib/dj/resynth.render) -
    notes, samples and levels as read, nothing composed. Returns
    (audio (n,2) float32, wav path)."""
    from lib.dj import resynth as RS
    with open(os.path.join(folder, "features.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(os.path.join(folder, "instruments.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    a = meta.get("analysis") or {}
    root, tid = a.get("dj_library"), a.get("dj_track_id")
    if progress:
        progress("decoding the library's stems")
    stems = RS.load_stems_mono(root, tid) if root and tid is not None else None
    if stems is None:
        raise FileNotFoundError("the track's stems are not on disk any more")
    if os.path.exists(os.path.join(folder, "program.json")):
        from lib.dj import songprogram as SP
        if progress:
            progress("rendering the song program")
        prog = SP.load(folder)
        # the program the way it should be heard: raw voices under per-voice ceilings and a limiter on the
        # sum (render()'s whole-track peak clamp used to sink a replay 7-15 dB under one hot voice)
        audio = SP.mixdown(prog, stems, ids=ids, t0=0.0, t1=float(a.get("duration_s") or 0.0) or None,
                           progress=(lambda i, n, name: progress(f"replay {i}/{n} {name}")) if progress else None)
    else:
        if progress:
            progress("playing the reading back (no program in this folder)")
        audio = RS.render(res, stems, ids=ids, t0=0.0, t1=float(a.get("duration_s") or 0.0) or None,
                          progress=(lambda i, n, name: progress(f"replay {i}/{n} {name}")) if progress else None)
    out_path = out_path or os.path.join(folder, "replay.wav")
    import soundfile as sf
    sf.write(out_path, audio, RATE, subtype="PCM_16")
    if progress:
        progress(f"replay written: {out_path}")
    return audio, out_path
