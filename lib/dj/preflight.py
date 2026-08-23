"""Arm-time seam pre-flight - SHADOW MODE (2026-08-23).

Flam is diagnosable only AFTER rendering (docs/DJ_VERIFICATION.md: every
plan-time predictor sorts verdicts at base rate, incl. the span-drift
sweep of 2026-08-23), and a seam arms 60-110s before it plays. This
module renders the armed plan OFFLINE during that lead and measures it
with the one instrument that passed the ear exam
(seamverify.measured_kick_alignment) - validated in sim 2026-08-23:
0/12 operator-good seams tripped the 45ms bar, and folding the measured
residuals back dropped a tripped drum_bridge from 58.8 to 7.2ms.

SHADOW ONLY: it measures and logs (`preflight` night-log events); it
never touches the armed seam. Verdict power comes later, if the shadow
data orders the live thumbs - the standing rule for every instrument.

DEPLOY SAFETY (the reason this file looks paranoid): live boxes are
GIL-bound against the audio producer, so the render runs in a separate
PROCESS at the OS's lowest priority - the scheduler, not Python, keeps
it off the audio path - and every failure fails OPEN:
  - skip when a worker is already running (never more than one),
  - skip when free memory is low (the worker decodes two tracks),
  - skip on short arm leads (urgent/skip seams),
  - a worker that dies, hangs, or declines just logs that fact.
The `audio_starved` night-log event carries a `preflight` flag so the
shadow data itself answers "does the pre-flight cause skips".

Worker entry: python -m lib.dj.preflight <job.pkl> <result.json>
"""
import json
import os
import pickle
import subprocess
import sys
import tempfile
import threading
import time

RATE = 44100
BLOCK = 1024
BAR_MS = 45.0            # the ear-validated flam bar (DJ_VERIFICATION.md)
MIN_LEAD_S = 25.0        # below this the result can't even inform a log
WORKER_TIMEOUT_S = 150.0
MIN_FREE_MB = 1500.0     # worker decodes 2 tracks (~100MB each) + render

_lock = threading.Lock()
_active = None           # {"proc","t0","meta"} while a worker runs
_results = []            # drained by DJSystem's tick via poll()


def active():
    """True while a shadow worker is running (audio_starved context)."""
    return _active is not None


def poll():
    """Drain finished shadow results (thread-safe, non-blocking)."""
    global _results
    with _lock:
        out, _results = _results, []
    return out


def _free_mb():
    """Best-effort free physical memory; None = unknown (don't block)."""
    try:
        if sys.platform == "win32":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual",
                             ctypes.c_ulonglong)]
            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return ms.ullAvailPhys / (1024.0 * 1024.0)
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return None


def launch_shadow(music_root, a_id, b_id, plan, theme, lead_s,
                  blend_wall_s, log_dir=None):
    """Spawn the shadow worker for an armed seam. Returns a skip-reason
    string when it does not launch (all guards fail OPEN), else None.
    Results arrive later via poll()."""
    global _active
    from lib.dj import tuning as _tuning
    if not _tuning.value("preflight_shadow", 1.0):
        return "disabled"
    if plan.get("style") == "long_fade":
        return "unsynced_style"     # instrument is meaningless on fades
    if lead_s < MIN_LEAD_S:
        return "short_lead"
    with _lock:
        if _active is not None:
            return "worker_busy"
        _active = {"t0": time.time()}
    free = _free_mb()
    if free is not None and free < MIN_FREE_MB:
        with _lock:
            _active = None
        return f"low_memory_{free:.0f}mb"

    tmp = tempfile.mkdtemp(prefix="dj_preflight_")
    job_p = os.path.join(tmp, "job.pkl")
    res_p = os.path.join(tmp, "result.json")
    # The plan is passed BY VALUE (pickle) so the worker renders exactly
    # what was armed - same anchors, same phase_applied, same bias. It
    # never re-plans: re-planning in a 2-track library diverges.
    with open(job_p, "wb") as f:
        pickle.dump({"music_root": music_root, "a_id": a_id, "b_id": b_id,
                     "plan": plan, "theme": theme,
                     "fast": bool(_tuning.value("preflight_fast", 1.0))},
                    f)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    kw = {"cwd": repo, "stdout": subprocess.DEVNULL,
          "stderr": subprocess.DEVNULL}
    if sys.platform == "win32":
        # IDLE_PRIORITY_CLASS | CREATE_NO_WINDOW
        kw["creationflags"] = 0x40 | 0x08000000
    else:
        kw["preexec_fn"] = lambda: os.nice(19)
    try:
        proc = subprocess.Popen([sys.executable, "-m", "lib.dj.preflight",
                                 job_p, res_p], **kw)
    except Exception as ex:
        with _lock:
            _active = None
        return f"spawn_failed:{ex}"

    meta = {"style": plan.get("style"), "a_id": a_id, "b_id": b_id,
            "lead_s": round(lead_s, 1),
            "blend_wall_s": blend_wall_s, "tmp": tmp}
    with _lock:
        _active.update({"proc": proc, "meta": meta})

    def _watch():
        global _active
        row = {"style": meta["style"], "lead_s": meta["lead_s"]}
        try:
            proc.wait(timeout=WORKER_TIMEOUT_S)
            row["wall_s"] = round(time.time() - _active["t0"], 1)
            with open(res_p, encoding="utf-8") as f:
                row.update(json.load(f))
        except subprocess.TimeoutExpired:
            proc.kill()
            row["error"] = "timeout"
            row["wall_s"] = round(WORKER_TIMEOUT_S, 1)
        except Exception as ex:
            row["error"] = f"result:{ex}"
            row["wall_s"] = round(time.time() - _active["t0"], 1)
        # Did the measurement land before the blend actually started?
        # (Shadow doesn't act either way - but this is THE deploy
        # question for a future acting mode.)
        row["in_time"] = (time.time() < meta["blend_wall_s"]
                          if meta.get("blend_wall_s") else None)
        flam = row.get("flam_med_ms")
        row["over_bar"] = None if flam is None else bool(flam > BAR_MS)
        for p in (job_p, res_p):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp)
        except OSError:
            pass
        with _lock:
            _results.append(row)
            _active = None

    threading.Thread(target=_watch, daemon=True,
                     name="dj-preflight-watch").start()
    return None


# ========================================================================
# Worker side (separate process, idle priority - never imported live)
# ========================================================================

def _load_track(db, track_id):
    from lib.dj.brain import TrackInfo
    row = db.get_track(track_id=track_id)
    if row is None:
        raise RuntimeError(f"track {track_id} not in DB")
    return TrackInfo(row, db.sections_for(track_id),
                     db.loops_for(track_id), db.mix_points_for(track_id),
                     cues=db.cues_for(track_id),
                     user_tags=db.tags_for(track_id))


def render_and_measure(music_root, a_id, b_id, plan, theme_name,
                       fast=False):
    """Render the armed plan offline and measure kick alignment.

    Mirrors tools/tests/_dj_quality_test.render_seam's render path (real
    submix, real build_events, stems attached for stem styles) but takes
    the LIVE plan by value instead of re-planning, and collects only
    what measured_kick_alignment needs. Gap-policy adjustments already
    live in the plan's fields; the policy itself is not re-run here
    (its input telemetry no longer exists), so gap-adjusted seams are
    rendered from the adjusted plan - `gap` rides the result so readers
    can segment. Returns the result dict (never raises).

    `fast` (2026-08-23, after the live shadow run showed a ~2fps render
    dip in worker windows): 4x block size - the render loop is Python-
    iteration-bound, not DSP-bound - and stop rendering once the
    instrument's own measurement span (blend + 24s + margin) is
    covered instead of running to swap+8s. Event scheduling and the
    PLL coarsen from 23ms to 93ms granularity; _dj_preflight_test's
    A/B holds fast-vs-full flam agreement, which is the only fidelity
    that matters here."""
    import numpy as np
    t_start = time.time()
    block = BLOCK * 4 if fast else BLOCK
    res = {"a_id": a_id, "b_id": b_id, "style": plan.get("style"),
           "gap": bool(plan.get("gap")), "fast": bool(fast)}
    try:
        from lib.audio_engine import AudioEngine
        from lib.dj import beatpower, features as F, seamverify
        from lib.dj.brain import Brain
        from lib.dj.db import LibraryDB
        from lib.dj.submix import DJSubmix
        from lib.dj.themes import get_theme

        beatpower.set_music_root(music_root)
        db = LibraryDB(music_root)
        cur = _load_track(db, a_id)
        cand = _load_track(db, b_id)
        a = F.decode_file_stereo(db.abs(cur.path))
        b = F.decode_file_stereo(db.abs(cand.path))
        res["decode_s"] = round(time.time() - t_start, 1)

        # A 2-track Brain is enough for build_events (knobs + curves);
        # the plan itself is the live one and is never recomputed.
        brain = Brain([cur, cand], get_theme(theme_name), seed=7)
        plan = dict(plan)

        engine = AudioEngine()
        sub = DJSubmix()
        engine.attach_track("dj", sub)
        pre_roll = max(20.0, (plan.get("beats") or 0) * cur.period_s + 8.0)
        cue_a = max(cur.nearest_downbeat(plan["out_s"] - pre_roll), 0.0)
        sub.post({"cmd": "load", "deck": "a", "samples": a,
                  "grid": cur.grid, "track_id": cur.id,
                  "gain_db": cur.gain_db, "cue_s": cue_a})
        sub.post({"cmd": "gain", "deck": "a", "value": 1.0,
                  "ramp_s": 0.01})
        sub.post({"cmd": "start", "deck": "a"})
        gen = engine._mixer()
        next(gen)
        n_blocks = 0
        for _ in range(int(2.0 * RATE) // block):      # telemetry warm-up
            gen.send(block)
            n_blocks += 1
        sub.post({"cmd": "load", "deck": "b", "samples": b,
                  "grid": cand.grid, "track_id": cand.id,
                  "gain_db": cand.gain_db, "cue_s": plan["in_s"]})
        if plan.get("style") in ("stem_drum_swap", "drum_bridge",
                                 "stem_bass_swap", "acapella_out",
                                 "acapella_in", "melody_carry") \
                or plan.get("duck_vocal_a"):
            from lib.dj.stems import load_stems
            for deck, t, arr in (("a", cur, a), ("b", cand, b)):
                if getattr(t, "has_stems", False):
                    st_ = load_stems(music_root, t.id,
                                     expected_len=len(arr))
                    if st_:
                        sub.post({"cmd": "attach_stems", "deck": deck,
                                  "stems": st_})
        for _ in range(4):
            gen.send(block)
            n_blocks += 1
        events, swap_at, blend_at = brain.build_events(
            plan, sub.telemetry, "a", "b", cur, cand)
        sub.post_many(events)

        deck_pcm = {"a": [], "b": []}
        pos_trace = {"a": [], "b": []}
        for _nm, _d in sub.decks.items():
            def _wrap(orig, nm):
                def f(n):
                    blk = orig(n)
                    deck_pcm[nm].append(
                        (sub.clock, blk.mean(axis=1).astype(np.float32)))
                    return blk
                return f
            _d.read = _wrap(_d.read, _nm)

        if fast:
            # The instrument reads [blend, min(swap, blend+24s)] - render
            # just past that, not to swap+8s.
            end_clock = min(swap_at, blend_at + int(24.0 * RATE)) \
                + int(4.0 * RATE)
        else:
            end_clock = swap_at + int(8.0 * RATE)
        # Hard block cap: a wedged deck must not spin this worker forever
        # (fail open = fail FAST).
        cap = n_blocks + int((end_clock - sub.clock) / block) + 2048
        pos_every = 1 if fast else 4     # keep trace density ~constant
        while sub.clock < end_clock and n_blocks < cap:
            gen.send(block)
            n_blocks += 1
            if n_blocks % pos_every == 0:
                for _nm in ("a", "b"):
                    _d = sub.decks[_nm]
                    if _d.playing:
                        pos_trace[_nm].append((sub.clock,
                                               _d.source_time_s()))
        total_len = n_blocks * block
        _sc0 = sub.clock - total_len
        deck_arr = {}
        for _nm in ("a", "b"):
            _arr = np.zeros(total_len, dtype=np.float32)
            for _clk, _blk in deck_pcm[_nm]:
                _i0 = int(_clk - _sc0)
                if 0 <= _i0 and _i0 + len(_blk) <= total_len:
                    _arr[_i0:_i0 + len(_blk)] = _blk
            deck_arr[_nm] = _arr
        marks = {"blend_s": (blend_at - _sc0) / RATE,
                 "swap_s": (swap_at - _sc0) / RATE,
                 "pos": {nm: [((c - _sc0) / RATE, s)
                              for c, s in pos_trace[nm]]
                         for nm in ("a", "b")}}
        ki = seamverify.measured_kick_alignment(deck_arr, marks,
                                                cur, cand)
        if ki is None:
            res["declined"] = True
        else:
            res.update({"flam_med_ms": ki.get("flam_med_ms"),
                        "flam_p90_ms": ki.get("flam_p90_ms"),
                        "off_a_ms": ki.get("off_a_ms"),
                        "off_b_ms": ki.get("off_b_ms"),
                        "n_kicks": ki.get("n")})
    except Exception as ex:
        res["error"] = f"{type(ex).__name__}: {ex}"
    res["render_s"] = round(time.time() - t_start, 1)
    return res


def main():
    job_p, res_p = sys.argv[1], sys.argv[2]
    with open(job_p, "rb") as f:
        job = pickle.load(f)
    # NOTE (2026-08-23): swapping the worker's stretch engine to
    # varispeed was A/B'd for more speed (1.7x) and REJECTED: the live
    # decks play rubberband, and measuring a varispeed rendition moved
    # flam by up to 19.9ms (mean 9.2ms) on the same seams - the exact
    # harness-fidelity violation DJ_VERIFICATION.md rule 4 exists for.
    # Fast mode stays: bigger blocks + early-stop, bit-identical flam.
    res = render_and_measure(job["music_root"], job["a_id"], job["b_id"],
                             job["plan"], job["theme"],
                             fast=bool(job.get("fast")))
    with open(res_p, "w", encoding="utf-8") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
