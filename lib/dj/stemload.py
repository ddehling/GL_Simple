"""Decoding OFF THE GIL: a song's mix and four stems decoded (and measured) in a helper PROCESS.

A stem song is ~5 s of PyAV decoding plus half a second of numpy over the stems. Done in a thread of the
show process it holds the GIL for most of that, and the audio producer - which must render 100 ms of four
stretched decks in well under 100 ms - fell behind and underran (user's log 2026-09-12: "the render thread
cannot keep up", right at a song's arrival). In a separate process the decode costs the show nothing but
the copy of the result (~200 MB for a five-minute song, a few hundred ms of memcpy).

One warm worker, started on first use; every call falls back to the in-process path when the pool is
unavailable (a frozen build without a spawnable interpreter, an import error in the child), so a failure
here degrades to the old behaviour rather than silence.

    samples, stems, levels = decode_song(abs_path, music_root, track_id, body_start_s, sections)

`levels` is what RemixConductor._stem_levels used to return: per-stem body RMS, "_env" (half-second
envelopes, float16, each stem scaled to its 95th percentile) and "_vox_sections" (the vocal stem's RMS per
section - the measured singing map).
"""
import math
import os
import threading

import numpy as np

RATE = 44100
POOL_TIMEOUT_S = 90.0             # a decode past this is a dead or hung helper: fall back in-process
_pool = None
_pool_lock = threading.Lock()
_pool_dead = False


def stem_levels(stems, body_start_s, sections):
    """Body RMS, half-second envelopes and the singing map from the stems - one pass per stem."""
    hop = RATE // 2
    i0 = int(body_start_s * RATE)
    out, env, energies = {}, {}, {}
    for name, arr in stems.items():
        a = np.asarray(arr)
        n = len(a) // hop
        if n <= 0:
            out[name] = 0.0
            continue
        e = np.empty(n, dtype=np.float32)
        step = 240
        for j in range(0, n, step):
            j1 = min(n, j + step)
            blk = a[j * hop:j1 * hop].astype(np.float32, copy=False).reshape(j1 - j, hop, -1)
            np.square(blk, out=blk)
            e[j:j1] = blk.mean(axis=(1, 2))
        energies[name] = e
        k0 = i0 // hop
        k1 = min(n, k0 + 240)
        if k1 - k0 < 2:
            k0, k1 = 0, min(n, 240)
        out[name] = float(math.sqrt(max(float(e[k0:k1].mean()), 0.0))) if k1 > k0 else 0.0
        vals = np.sqrt(e)
        top = float(np.percentile(vals, 95)) or 1e-6
        env[name] = np.clip(vals / top, 0.0, 1.0).astype(np.float16)
    out["_env"] = env
    e_vox = energies.get("vocals")
    if e_vox is not None and sections:
        per = []
        for s in sections:
            k0, k1 = int(s["start_s"] * 2), min(len(e_vox), int(s["end_s"] * 2))
            per.append(math.sqrt(max(float(e_vox[k0:k1].mean()), 0.0)) if k1 - k0 >= 1 else 0.0)
        out["_vox_sections"] = per
    return out


def _decode_in_process(abs_path, music_root, track_id, body_start_s, sections, with_levels=True):
    """The work itself - runs in the helper process (or inline as the fallback)."""
    from lib.dj.features import decode_file_stereo
    from lib.dj.stems import load_stems
    samples = decode_file_stereo(abs_path)
    stems = load_stems(music_root, track_id, expected_len=len(samples))
    if not stems:
        raise ValueError("no stems on disk")
    levels = stem_levels(stems, body_start_s, [dict(start_s=s["start_s"], end_s=s["end_s"]) for s in (sections or [])]) if with_levels else None
    return samples, stems, levels


def _get_pool():
    global _pool, _pool_dead
    if _pool_dead:
        return None
    with _pool_lock:
        if _pool is None:
            try:
                import multiprocessing
                from concurrent.futures import ProcessPoolExecutor
                ctx = multiprocessing.get_context("spawn")
                _pool = ProcessPoolExecutor(max_workers=1, mp_context=ctx)
            except Exception as e:  # noqa: BLE001
                print(f"[DJ] stem decode pool unavailable ({type(e).__name__}: {e}) - decoding in-process")
                _pool_dead = True
                return None
        return _pool


def decode_song(abs_path, music_root, track_id, body_start_s=0.0, sections=None, with_levels=True):
    """Decode + measure a song in the helper process; in-process when the pool cannot be used."""
    global _pool_dead
    pool = _get_pool() if os.environ.get("DJ_DECODE_INPROC", "") != "1" else None
    if pool is not None:
        try:
            fut = pool.submit(_decode_in_process, abs_path, music_root, int(track_id), float(body_start_s),
                              [dict(start_s=float(s["start_s"]), end_s=float(s["end_s"])) for s in (sections or [])], with_levels)
            return fut.result(timeout=POOL_TIMEOUT_S)
        except ValueError:
            raise
        except Exception as e:  # noqa: BLE001
            # a broken pool (the child died, spawn refused or hung - a spawn with no importable __main__
            # hangs rather than errors): fall back for good this session
            print(f"[DJ] stem decode pool failed ({type(e).__name__}: {e}) - decoding in-process from now on")
            _kill_pool()
    return _decode_in_process(abs_path, music_root, track_id, body_start_s, sections, with_levels)


def _kill_pool():
    global _pool, _pool_dead
    _pool_dead = True
    p, _pool = _pool, None
    if p is not None:
        try:
            p.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


def _decode_mix_in_process(abs_path):
    from lib.dj.features import decode_file_stereo
    return decode_file_stereo(abs_path)


def decode_mix(abs_path):
    """Just the mix (the autoDJ's decode), in the helper process when it can be."""
    global _pool_dead
    pool = _get_pool() if os.environ.get("DJ_DECODE_INPROC", "") != "1" else None
    if pool is not None:
        try:
            return pool.submit(_decode_mix_in_process, abs_path).result(timeout=POOL_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001
            print(f"[DJ] stem decode pool failed ({type(e).__name__}: {e}) - decoding in-process from now on")
            _kill_pool()
    return _decode_mix_in_process(abs_path)


def warm():
    """Start the helper process early (the first spawn costs a second or two of imports in the child)."""
    pool = _get_pool()
    if pool is not None:
        def _check(fut):
            # the child must answer within a generous minute (imports on a cold disk); a spawn that
            # cannot import __main__ hangs forever, and the first real decode would then wait on it
            try:
                fut.result(timeout=0)
            except Exception:
                pass
        try:
            fut = pool.submit(_ping)
            threading.Timer(60.0, lambda: (_kill_pool() if not fut.done() else None)).start()
            fut.add_done_callback(_check)
        except Exception:
            _kill_pool()


def _ping():
    return 1
