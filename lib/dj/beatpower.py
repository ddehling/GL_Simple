"""Beat power: does this track actually THUMP on its own beats?

The missing column that explained a day of chasing ghosts (2026-08-04).
Grid confidence measures periodicity - whether a steady lattice fits the
audio. It never asked whether the music puts low-band ATTACK energy ON
that lattice. Measured across the library: 38% of tracks score below
1.2 - confident grids over diffuse grooves (organic percussion, rolling
basslines, tribal walls). Beat-matching those is matching air: the sync
can be sample-perfect while the ear hears an unrelated mess, which is
exactly what the user reported ("the beats are fundamentally off...
double beat... all the time").

Score = mean on-beat low-band attack peak / mean off-beat (half-beat
later) attack peak, over ~30s at the track's midpoint. Density-neutral:
a busy full mix scores high if its kicks land on its beats, low only
when the off-beats carry just as much attack. ~1.0 = no beat to match;
>=2 = clean four-on-floor.

CLI (fills logs/beat_power.json incrementally, resumable):
    python -m lib.dj.beatpower --music D:/Devel/music
"""
import json
import os
import time

import numpy as np

RATE = 44100
# Bars lowered 2026-08-05 (were 1.5/1.2): they were set while sync could
# only align GRIDS - blending borderline-groove material then smeared
# ("matching air"). With kicks now measured and phase-locked (validated
# 2-17ms on rendered seams) and the pattern screens (density mismatch,
# 1:1 interleave, swing clash) still standing, the bars only need to
# exclude genuinely beatless material. A week of over-gating turned
# nights into fade marathons ("a week ago it was pretty good. now every
# dj mix sucks" - user; the live log: five fades in a row).
BLEND_MIN = 1.3        # the INCOMING side: it becomes the foundation
BLEND_MIN_EXIT = 1.05  # the OUTGOING side: it only hands off


_MUSIC_ROOT = None


def set_music_root(root):
    """beat_power.json LIVES WITH THE LIBRARY IT DESCRIBES (2026-08-14).

    It used to live in the repo's logs/ - a per-machine, gitignored
    artifact keyed by track ids from the library DB that DOES sync with
    the music folder. The playing machine therefore had the DB but NO
    beat_power.json at all, and the whole precision stack (kick-true
    anchors, phase interpolation, the local grid_conf standdown, dense
    beat power) was silently inert on the one machine that plays music -
    diagnosed 2026-08-14 after a day of "bad beat matching" reports
    that no local render could reproduce. The file now sits next to the
    DB it is keyed to, so the operator's own music sync carries it.
    Called by LibraryDB.__init__ - the one place that knows the root."""
    global _MUSIC_ROOT
    _MUSIC_ROOT = root


def path():
    # DJ_BEATPOWER_PATH: hermetic override for test harnesses. The
    # synthetic-fleet e2e uses tiny track ids that COLLIDE with real
    # library ids in the production file - its fake tracks inherited
    # real tracks' scores and lost their blends to another song's beat
    # power (caught 2026-08-05 when phrase_cut's retirement unmasked it).
    env = os.environ.get("DJ_BEATPOWER_PATH")
    if env:
        return env
    if _MUSIC_ROOT:
        return os.path.join(_MUSIC_ROOT, "beat_power.json")
    # Legacy repo-logs location: only reachable before any LibraryDB
    # exists (no DJ tool touches beatpower before opening the library).
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "logs", "beat_power.json")


_CACHE = {"mtime": None, "scores": {}}


def scores():
    """{track_id: score} as last scanned. Cheap mtime-cached re-read."""
    p = path()
    try:
        m = os.path.getmtime(p)
    except OSError:
        return {}
    if _CACHE["mtime"] != m:
        try:
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
            _CACHE["scores"] = {
                int(k): float(v["score"] if isinstance(v, dict) else v)
                for k, v in doc.get("scores", {}).items()
                if (v["score"] if isinstance(v, dict) else v) is not None}
            _CACHE["mtime"] = m
        except (OSError, ValueError):
            return {}
    return _CACHE["scores"]


def band_scores(track_id, region="mid"):
    """Per-band rhythmicity for a track region ('mid' body, 'in' intro,
    'out' exit), or None until the --bands pass has scored it. Falls back
    to the body when a region was unmeasurable."""
    p = path()
    try:
        m = os.path.getmtime(p)
    except OSError:
        return None
    if _CACHE.get("bands_mtime") != m:
        try:
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
            _CACHE["bands"] = {int(k): v.get("bands")
                               for k, v in doc.get("scores", {}).items()
                               if isinstance(v, dict) and v.get("bands")}
            # accept both flat {band: s} and region {region: {band: s}}
            _CACHE["bands_mtime"] = m
        except (OSError, ValueError):
            return None
    b = _CACHE.get("bands", {}).get(track_id)
    if b is None:
        return None
    if "low" in b:                      # old flat format
        return b
    return b.get(region) or b.get("mid")


def blendable(track_id):
    """True / False / None (not yet measured - existing gates decide)."""
    s = scores().get(track_id)
    return None if s is None else s >= BLEND_MIN


def profile_coverage(track_id):
    """Fraction of a track's phase-profile buckets that pass the trust
    bar - i.e. how much of the track has a MEASURED-good grid (attacks
    on the lattice, consistent phase, which also pins the local period).
    0.0 when unscanned. A high coverage outranks a stale whole-track
    bpm_conf scalar: the profile is direct evidence, the scalar a fit
    score from the original scan."""
    p = path()
    try:
        m = os.path.getmtime(p)
    except OSError:
        return 0.0
    if _CACHE.get("phase_mtime") != m:
        phase_offset(track_id)           # refresh the shared cache
    d = _CACHE.get("phase", {}).get(track_id)
    prof = (d or {}).get("prof")
    if not prof:
        return 0.0
    ok = sum(1 for r in prof
             if r.get("n", 0) >= 10 and r.get("iqr", 999.0) <= 55.0)
    return ok / len(prof)


def phase_offset(track_id, region="mid", at_s=None):
    """Measured music-vs-grid offset in SECONDS for a track region, or
    None when unmeasured or unreliable. Positive = the audible low-band
    attacks land LATE relative to the stored grid.

    Pass at_s (source seconds of the seam anchor) to select the NEAREST
    measured region by POSITION - a seam forced mid-track (the lab, the
    knob sweep) must not read the offset measured at the primary mix-out
    100s away; phase is a local property. Without at_s, the region label
    ('in'/'mid'/'out') selects, with 'mid' fallback.

    This is the column the 2026-08-04 beat-match hunt ended on: stored
    grids are periodically right (high confidence) but their PHASE misses
    the actual kicks by ~48ms median in seam regions, with signs
    differing per track - so grid-to-grid sync aligns lattices while the
    ear hears flam. The sync bias consumes the DIFFERENCE of these
    offsets to put kicks, not grids, in register."""
    p = path()
    # The mtime freshness probe is a syscall, and selection makes ~1650
    # phase lookups per pick - throttle it to twice a second. A freshly
    # written profile is picked up within 500ms, which is instant on the
    # only timescale that matters (a scan finishing mid-session).
    now = time.monotonic()
    m = _CACHE.get("phase_mtime")
    if now - _CACHE.get("phase_mtime_checked", 0.0) > 0.5:
        _CACHE["phase_mtime_checked"] = now
        try:
            m = os.path.getmtime(p)
        except OSError:
            return None
    if m is None:
        return None
    if _CACHE.get("phase_mtime") != m:
        try:
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
            _CACHE["phase"] = {int(k): v.get("phase")
                               for k, v in doc.get("scores", {}).items()
                               if isinstance(v, dict) and v.get("phase")}
            _CACHE["phase_mtime"] = m
            _CACHE["phase_arrs"] = {}   # derived arrays follow the doc
            _CACHE["phase_memo"] = {}   # ...and the result memo with them
        except (OSError, ValueError):
            return None
    d = _CACHE.get("phase", {}).get(track_id)
    if not d:
        return None
    # RESULT MEMO: the same (track, anchor) is asked thousands of times
    # per pick - cur's handful of exit anchors repeat for every one of
    # ~1000 candidates. Keyed at half-second granularity (well inside
    # one profile bucket); cleared with the doc cache on reload.
    if at_s is not None:
        mk = (track_id, region, round(at_s * 2.0))
        memo = _CACHE.setdefault("phase_memo", {})
        if mk in memo:
            return memo[mk]
    else:
        mk = None

    def _ok(rec):
        # Trust bar: enough beats with real attacks, and a consistent
        # story. A wide IQR means the bucket has no single phase
        # (organic/rubato) - correcting with its median is a coin flip.
        return (rec is not None and rec.get("n", 0) >= 10
                and rec.get("iqr", 999.0) <= 55.0)

    prof = d.get("prof")
    if prof is not None:
        if at_s is None:
            at_s = prof[len(prof) // 2]["at_s"]    # label-less: track body
        # PER-TRACK ARRAY CACHE (2026-08-14). Selection calls this
        # ~1650x per pick (the blendability mirror runs best_pair per
        # candidate, and the anchor-trust lean added more) and the
        # per-call list comprehensions + np builds were HALF of all
        # planning time (measured 135ms of a 272ms pick on the dev box;
        # ~3x that on the N150's planner thread, all of it GIL-bound
        # against the audio producer and the GL loop). The profile is
        # immutable per file mtime, so the trusted-point arrays and the
        # untrusted-bucket positions are computed once per track and
        # invalidated with the same mtime the raw doc cache uses.
        ck = _CACHE.setdefault("phase_arrs", {})
        arrs = ck.get(track_id)
        if arrs is None:
            pts = [(r["at_s"], r["ms"]) for r in prof if _ok(r)]
            bad = np.asarray([r["at_s"] for r in prof if not _ok(r)],
                             dtype=np.float64)
            if pts:
                arrs = (np.asarray([p[0] for p in pts]),
                        np.asarray([p[1] for p in pts]), bad)
            else:
                arrs = (None, None, bad)
            ck[track_id] = arrs
        xs, ys, bad = arrs
        # If the bucket AT this position was measured but failed the
        # trust bar, the phase HERE is genuinely unstable (fill, break,
        # rubato) - interpolating neighbors across it is a guess dressed
        # as a measurement (validation: -44ms miss on exactly this case).
        # Decline; no correction beats a wrong one.
        if xs is None:
            val = None
        elif len(bad) and float(np.min(np.abs(bad - at_s))) \
                <= PROF_BUCKET_S / 2:
            val = None
        elif float(np.min(np.abs(xs - at_s))) > PROF_REACH_S:
            val = None        # nothing trustworthy near this position
        else:
            # INTERPOLATE, don't stair-step: measured profiles are
            # smooth RAMPS (~-0.44 ms/s on most of this library - the
            # stored grid period is systematically ~0.04% off the
            # music, a scanner tempo quantization artifact), so the
            # local phase at a seam is the line through the neighboring
            # buckets, not the nearest median.
            val = float(np.interp(at_s, xs, ys)) / 1000.0
        if mk is not None:
            _CACHE.setdefault("phase_memo", {})[mk] = val
        return val
    # legacy labeled-region records (pre-profile format)
    if at_s is not None:
        best, dist = None, None
        for rec in d.values():
            if not _ok(rec) or rec.get("at_s") is None:
                continue
            dd = abs(rec["at_s"] - at_s)
            if dist is None or dd < dist:
                best, dist = rec, dd
        if best is not None:
            return best["ms"] / 1000.0
    p_ = d.get(region) or d.get("mid")
    if not _ok(p_):
        return None
    return p_["ms"] / 1000.0


BANDS = {"low": ("lowpass", 110.0),
         "mid": ("bandpass", (250.0, 2000.0)),
         "high": ("highpass", 4000.0)}


def compute(track, db, bands=False):
    """Score one track from its raw audio + stored grid.

    Returns a scalar (low-band score, the blend gate) or, with
    bands=True, {"low":s,"mid":s,"high":s} - the per-band rhythmicity
    the band-aware style selection consumes: different frequency bands
    mismatch independently, and each mix style is a strategy for which
    bands may overlap (user, 2026-08-04)."""
    from lib.dj.features import decode_file_stereo
    from scipy.signal import butter, sosfilt
    try:
        x = decode_file_stereo(db.abs(track.path))
    except Exception:
        return None
    mid = track.duration_s * 0.5
    lo = int(max(mid - 15, 0) * RATE)
    hi = int(min(mid + 15, track.duration_s) * RATE)
    seg = x[lo:hi].mean(axis=1).astype(np.float64)
    if len(seg) < 5 * RATE:
        return None
    if bands:
        # REGION-AWARE (user, 2026-08-04: timing per band, where it
        # matters): a blend overlaps A's EXIT with B's INTRO, and a track
        # that thumps in its body can be a diffuse wash in its intro. One
        # score set per region: "mid" (body), "in" (around the primary
        # mix-in), "out" (around the primary mix-out).
        def _span(center):
            l = int(max(center - 15, 0) * RATE)
            h = int(min(center + 15, track.duration_s) * RATE)
            return x[l:h].mean(axis=1).astype(np.float64), l
        regions = {"mid": (seg, lo)}
        try:
            if track.mix_ins:
                regions["in"] = _span(track.mix_ins[0]["time_s"] + 10)
            if track.mix_outs:
                regions["out"] = _span(track.mix_outs[0]["time_s"] - 10)
        except Exception:
            pass
        out = {}
        for rname, (rseg, rlo) in regions.items():
            if len(rseg) < 5 * RATE:
                continue
            out[rname] = {name: _band_score(track, rseg, rlo, kind, freq)
                          for name, (kind, freq) in BANDS.items()}
        return out or None
    sos = butter(4, 110.0, btype="lowpass", fs=RATE, output="sos")
    env = np.abs(sosfilt(sos, seg))
    w = max(int(0.01 * RATE), 1)
    env = np.convolve(env, np.ones(w) / w, mode="same")
    att = np.diff(env)
    att[att < 0] = 0.0
    beats = []
    for g in (track.grid or []):
        per = g.get("period_s") or 0
        if per <= 0:
            continue
        b = g["first_beat_s"]
        span_lo, span_hi = lo / RATE, hi / RATE
        if b < span_lo:
            b += np.ceil((span_lo - b) / per) * per
        while b <= min(span_hi, g["end_s"]):
            beats.append((b - span_lo, per))
            b += per
    if len(beats) < 16:
        return None
    on = off = 0.0
    n_on = n_off = 0
    w2 = int(0.03 * RATE)
    for b, per in beats:
        i = int(b * RATE)
        if i + w2 >= len(att):
            break
        on += float(np.max(att[max(i - w2, 0):i + w2]))
        n_on += 1
        j = int((b + per * 0.5) * RATE)
        if j + w2 < len(att):
            off += float(np.max(att[j - w2:j + w2]))
            n_off += 1
    if not n_on or not n_off or off <= 0:
        return None
    return round((on / n_on) / (off / n_off), 3)


PROF_BUCKET_S = 20.0     # phase profile resolution along the track
PROF_REACH_S = 30.0      # farthest a bucket may be from the asked position


def compute_phase(track, db):
    """DENSE music-vs-grid phase profile from the raw audio.

    One full-track pass: low-band attack envelope, then for EVERY grid
    beat find the strongest attack peak within +/-75ms (capped at 0.3
    beat) and record peak_time - beat_time. Offsets are bucketed every
    ~20s; each bucket keeps median/iqr/count. Returns
    {"prof": [{"at_s", "ms", "iqr", "n"}, ...]}.

    Dense on purpose (2026-08-04): the first cut measured 3 labeled
    regions (in/mid/out) and validation immediately caught seams landing
    60-100s from the nearest measurement, where the local phase had
    drifted 40-90ms away - phase is a LOCAL property, so the profile
    must cover wherever a seam can land. Same instrument that DIAGNOSED
    the 48ms defect (rendered attacks vs trace-projected beats) applied
    at the source, so correction and diagnosis can never disagree about
    what a kick is."""
    from lib.dj.features import decode_file_stereo
    from scipy.signal import butter, sosfilt
    try:
        x = decode_file_stereo(db.abs(track.path))
    except Exception:
        return None
    mono = x.mean(axis=1).astype(np.float32)
    del x
    sos = butter(4, 110.0, btype="lowpass", fs=RATE, output="sos")
    env = np.abs(sosfilt(sos, mono)).astype(np.float32)
    del mono
    w = max(int(0.01 * RATE), 1)
    env = np.convolve(env, np.ones(w, dtype=np.float32) / w, mode="same")
    att = np.diff(env)
    del env
    att[att < 0] = 0.0

    peaks = []                       # (beat_s, peak_value, offset_s)
    for g in (track.grid or []):
        per = g.get("period_s") or 0
        if per <= 0:
            continue
        half = int(min(0.075, 0.3 * per) * RATE)
        b = g["first_beat_s"]
        if b < 0:
            b += np.ceil(-b / per) * per
        while b <= g["end_s"]:
            i = int(b * RATE)
            if i - half >= 0 and i + half < len(att):
                wseg = att[i - half:i + half]
                k = int(np.argmax(wseg))
                peaks.append((b, float(wseg[k]), (k - half) / RATE))
            b += per
    del att
    if len(peaks) < 24:
        return None
    ref = float(np.median([p for _, p, _ in peaks]))
    prof = []
    n_buckets = int(track.duration_s / PROF_BUCKET_S) + 1
    for bi in range(n_buckets):
        lo, hi = bi * PROF_BUCKET_S, (bi + 1) * PROF_BUCKET_S
        offs = [o for b, p, o in peaks
                if lo <= b < hi and p > 0 and p >= 0.3 * ref]
        if len(offs) < 10:
            continue
        q1, q3 = np.percentile(offs, [25, 75])
        prof.append({"at_s": round(lo + PROF_BUCKET_S / 2, 1),
                     "ms": round(float(np.median(offs)) * 1000.0, 1),
                     "iqr": round(float(q3 - q1) * 1000.0, 1),
                     "n": len(offs)})
    return {"prof": prof} if prof else None


def compute_power_prof(track, db):
    """DENSE beat-power profile from the raw audio - the same on/off
    attack ratio `compute` measures, but for EVERY ~20s of the track
    instead of three fixed windows.

    Why (2026-08-12): beat power was stored as three 30s samples - the
    midpoint, and windows around mix_ins[0]/mix_outs[0] - and the gate
    picked one by label, falling back to the MIDPOINT whenever the seam
    landed >45s from the primary mix point. Measured on this library
    that fallback fired on 35% of planned seams, and it is not a
    harmless approximation: 46.7% of tracks fail the incoming bar when
    judged at their midpoint against 27.2% judged at their entry, so
    182 tracks (19.5%) were refused as 'no beat to match' over a beat
    they have exactly where the blend would have landed. Within-track
    spread is the reason - median |in-mid| is 0.43 and 35.3% of tracks
    STRADDLE the bar, passing in one region and failing in another.
    Beat power is a local property, and the code treated it as a
    track-level scalar.

    This is the same lesson, and the same fix, that compute_phase
    already learned for phase in 2026-08-04 - hence the same shape:
    {"prof": [{"at_s", "s", "n"}, ...]}, read back through power_at().

    One full-track pass over the low-band attack envelope; per bucket
    the mean on-beat peak over the mean off-beat (half-period later)
    peak, identical in definition to `compute` so the stored bars
    (BLEND_MIN / BLEND_MIN_EXIT) keep their meaning.

    PHASE-CORRECTED (v2, same day): the first cut measured at the RAW
    grid positions and read 0.8-0.97 on mid-groove sections of tracks
    whose midpoint scores 2-4x - because stored grids miss the real
    kicks by ~48ms median in seam regions, drifting as a ramp (see
    compute_phase's docstring), and a +/-30ms on-beat window loses the
    kick entirely once the drift passes it. Late-track buckets - the
    exact regions this profile exists to score - are where the drift is
    largest, so uncorrected readings sagged precisely where the old
    midpoint fallback was WRONG in the other direction. Each bucket's
    beats are shifted by the measured local phase offset before the
    windows are placed, mirroring what the deck's sync bias does at mix
    time: the blend aligns KICKS, so the on-beat power that predicts it
    is at the kick-true positions. Buckets where the phase is untrusted
    (phase_offset returns None) fall back to the raw grid - a best
    effort, not a refusal, since a scalar existed for years with no
    correction at all."""
    from lib.dj.features import decode_file_stereo
    from scipy.signal import butter, sosfilt
    try:
        x = decode_file_stereo(db.abs(track.path))
    except Exception:
        return None
    mono = x.mean(axis=1).astype(np.float32)
    del x
    sos = butter(4, 110.0, btype="lowpass", fs=RATE, output="sos")
    env = np.abs(sosfilt(sos, mono)).astype(np.float32)
    del mono
    w = max(int(0.01 * RATE), 1)
    env = np.convolve(env, np.ones(w, dtype=np.float32) / w, mode="same")
    att = np.diff(env)
    del env
    att[att < 0] = 0.0

    # (beat_s, on_peak, off_peak) for every grid beat with both windows
    # fully inside the audio - bucketed after the walk, so a beat is
    # measured once no matter how the buckets fall. Each beat is shifted
    # to its kick-true position first (one phase lookup per ~20s bucket,
    # cached - phase is smooth at that scale).
    w2 = int(0.03 * RATE)
    _ph_cache = {}

    def _ph(b):
        key = int(b / PROF_BUCKET_S)
        if key not in _ph_cache:
            _ph_cache[key] = phase_offset(
                track.id, at_s=(key + 0.5) * PROF_BUCKET_S)
        return _ph_cache[key] or 0.0

    obs = []
    for g in (track.grid or []):
        per = g.get("period_s") or 0
        if per <= 0:
            continue
        b = g["first_beat_s"]
        if b < 0:
            b += np.ceil(-b / per) * per
        while b <= g["end_s"]:
            bc = b + _ph(b)
            i = int(bc * RATE)
            j = int((bc + per * 0.5) * RATE)
            if i - w2 >= 0 and j + w2 < len(att):
                obs.append((b,
                            float(np.max(att[i - w2:i + w2])),
                            float(np.max(att[j - w2:j + w2]))))
            b += per
    del att
    if len(obs) < 24:
        return None
    prof = []
    n_buckets = int(track.duration_s / PROF_BUCKET_S) + 1
    for bi in range(n_buckets):
        lo, hi = bi * PROF_BUCKET_S, (bi + 1) * PROF_BUCKET_S
        cell = [(on, off) for b, on, off in obs if lo <= b < hi]
        # Same floor as _band_score's whole-window walk: fewer than 16
        # beats is not a measurement, it is a fill or a gap.
        if len(cell) < 16:
            continue
        on_m = float(np.mean([c[0] for c in cell]))
        off_m = float(np.mean([c[1] for c in cell]))
        if off_m <= 0:
            continue
        prof.append({"at_s": round(lo + PROF_BUCKET_S / 2, 1),
                     "s": round(on_m / off_m, 3),
                     "n": len(cell)})
    # v2 = phase-corrected windows; the scan recomputes any stored
    # profile without this marker.
    return {"prof": prof, "v": 2} if prof else None


def power_at(track_id, at_s):
    """Beat power AT a position in seconds, or None when unmeasured or
    nothing trustworthy sits near it. Companion to phase_offset(at_s=),
    and deliberately the same contract: None means 'no evidence', which
    every caller must treat as 'no penalty' rather than 'bad'.

    Interpolated, not stair-stepped, for the same reason phase is: the
    profile samples a continuous property every PROF_BUCKET_S, so the
    value at a seam is the line through its neighbours. Positions
    farther than PROF_REACH_S from any measured bucket return None -
    a breakdown long enough to have no scored bucket is exactly where a
    guess would be worst."""
    p = path()
    try:
        m = os.path.getmtime(p)
    except OSError:
        return None
    if _CACHE.get("power_mtime") != m:
        try:
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
            _CACHE["power"] = {int(k): (v.get("power") or {}).get("prof")
                               for k, v in doc.get("scores", {}).items()
                               if isinstance(v, dict) and v.get("power")}
            _CACHE["power_mtime"] = m
        except (OSError, ValueError):
            return None
    prof = _CACHE.get("power", {}).get(track_id)
    if not prof:
        return None
    xs = np.asarray([r["at_s"] for r in prof], dtype=np.float64)
    if float(np.min(np.abs(xs - at_s))) > PROF_REACH_S:
        return None
    return float(np.interp(at_s, xs,
                           np.asarray([r["s"] for r in prof],
                                      dtype=np.float64)))


def _band_score(track, seg, lo, kind, freq):
    from scipy.signal import butter, sosfilt
    sos = butter(4, freq, btype=kind, fs=RATE, output="sos")
    env = np.abs(sosfilt(sos, seg))
    w = max(int(0.01 * RATE), 1)
    env = np.convolve(env, np.ones(w) / w, mode="same")
    att = np.diff(env)
    att[att < 0] = 0.0
    beats = []
    for g in (track.grid or []):
        per = g.get("period_s") or 0
        if per <= 0:
            continue
        b = g["first_beat_s"]
        span_lo = lo / RATE
        span_hi = span_lo + len(seg) / RATE
        if b < span_lo:
            b += np.ceil((span_lo - b) / per) * per
        while b <= min(span_hi, g["end_s"]):
            beats.append((b - span_lo, per))
            b += per
    if len(beats) < 16:
        return None
    on = off = 0.0
    n_on = n_off = 0
    w2 = int(0.03 * RATE)
    for b, per in beats:
        i = int(b * RATE)
        if i + w2 >= len(att):
            break
        on += float(np.max(att[max(i - w2, 0):i + w2]))
        n_on += 1
        j = int((b + per * 0.5) * RATE)
        if j + w2 < len(att):
            off += float(np.max(att[j - w2:j + w2]))
            n_off += 1
    if not n_on or not n_off or off <= 0:
        return None
    return round((on / n_on) / (off / n_off), 3)


def main():
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--music", required=True)
    ap.add_argument("--bands", action="store_true",
                    help="also fill per-band scores (for band-aware "
                         "style selection)")
    ap.add_argument("--phase", action="store_true",
                    help="also fill per-region music-vs-grid phase "
                         "offsets (kick alignment bias)")
    ap.add_argument("--power", action="store_true",
                    help="also fill the DENSE beat-power profile, so the "
                         "blend gates can score the seam's actual "
                         "position instead of the track midpoint")
    args = ap.parse_args()
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception:
        pass
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from lib.dj.brain import load_library
    from lib.dj.db import LibraryDB
    db = LibraryDB(args.music)
    lib = load_library(db)
    done = {}
    try:
        with open(path(), encoding="utf-8") as f:
            done = json.load(f).get("scores", {})
    except (OSError, ValueError):
        pass
    print(f"{len(lib)} tracks, {len(done)} already scored", flush=True)
    for i, t in enumerate(lib):
        key = str(t.id)
        have = done.get(key)
        need_bands = args.bands and (not isinstance(have, dict)
                                     or "bands" not in have)
        need_phase = args.phase and (not isinstance(have, dict)
                                     or "phase" not in have
                                     or "prof" not in have["phase"])
        need_power = args.power and (not isinstance(have, dict)
                                     or "power" not in have
                                     or "prof" not in have["power"]
                                     or have["power"].get("v") != 2)
        if have is not None and not need_bands and not need_phase \
                and not need_power:
            continue
        if args.bands or args.phase or args.power:
            rec = dict(have) if isinstance(have, dict) else \
                ({"score": have} if have is not None else {"score": None})
            if need_bands:
                b = compute(t, db, bands=True)
                if b is not None:
                    rec["bands"] = b
                    if rec.get("score") is None:
                        rec["score"] = b.get("low")
            if need_phase:
                p = compute_phase(t, db)
                if p is not None:
                    rec["phase"] = p
            if need_power:
                pw = compute_power_prof(t, db)
                if pw is not None:
                    rec["power"] = pw
                    # A track with no stored scalar (an old partial scan)
                    # gets one from the profile's own body bucket, so the
                    # legacy `evid` path stays populated.
                    if rec.get("score") is None and pw["prof"]:
                        rec["score"] = pw["prof"][len(pw["prof"]) // 2]["s"]
            if rec:
                done[key] = rec
        else:
            s = compute(t, db)
            if s is not None:
                done[key] = s
        if i % 10 == 0 or i == len(lib) - 1:
            os.makedirs(os.path.dirname(path()), exist_ok=True)
            with open(path(), "w", encoding="utf-8") as f:
                json.dump({"t": time.time(), "scores": done}, f)
            print(f"[{i + 1}/{len(lib)}] scored {len(done)}", flush=True)
        time.sleep(0.3)                  # breathing room - desktop first
    with open(path(), "w", encoding="utf-8") as f:
        json.dump({"t": time.time(), "scores": done}, f)
    vals = np.array([float(v["score"] if isinstance(v, dict) else v)
                     for v in done.values()
                     if (v["score"] if isinstance(v, dict) else v)
                     is not None])
    print(f"done: {len(done)} scored, median {np.median(vals):.2f}, "
          f"{np.mean(vals < BLEND_MIN):.0%} below the blend bar", flush=True)


if __name__ == "__main__":
    main()
