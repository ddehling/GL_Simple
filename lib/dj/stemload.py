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

THE HANDOFF (2026-10-25): the result used to come back PICKLED through the pool's pipe, and unpickling a
five-minute song (~318 MB: the float32 mix and four float16 stems) held the show process's GIL 85-119 ms
(measured: a 1 ms probe thread woke that late) - an LED stutter at every song's arrival, longer on the
N150. Now the helper writes the arrays into shared memory and returns only their names; the parent copies
them out in 8 MB slices (the GIL changes hands between slices: worst stall 5-11 ms), and the helper then
frees exactly that call's blocks (see "the shared-memory handoff" below for ownership and cleanup). The caller gets ordinary private ndarrays, the same dtypes as before, that live as long as it
holds them - nothing here depends on the helper outliving a deck. Any failure on either side falls back
to the pickled result.
"""
import math
import os
import threading
import time

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


# --- the shared-memory handoff ----------------------------------------------------------------------------
# OWNERSHIP. The helper creates every block, keeps its own handle to each one under the block's name, and is
# the only side that ever unlinks: the parent attaches, copies, closes, then asks the helper to release
# exactly THAT call's names (never "everything" - two decks decode at once, and a blanket release freed one
# call's blocks under the other's parent on Windows, where a block dies with its last handle).
# CLEANUP. Created blocks stay registered with multiprocessing's resource tracker (on POSIX the spawned
# helper shares the parent's tracker process), so a block the parent never collects - a decode that timed
# out, a show quit mid-decode, a parent-side error - is unlinked when the app exits; a long-running helper
# also sweeps anything older than STALE_S. Release = close + unlink, and unlink also unregisters, so the
# tracker never tries a name twice.
COPY_SLICE_BYTES = 8 << 20        # parent copies out this much per slice, then lets the GIL go round
STALE_S = 300.0                   # helper side: a block nobody collected in this long is abandoned
_child_keep = {}                  # helper side: name -> (SharedMemory, created at, monotonic)


class _HandoffError(RuntimeError):
    """The parent could not copy a block out: decode that one song in-process, keep the helper."""


def _shm_create(size):
    from multiprocessing import shared_memory
    return shared_memory.SharedMemory(create=True, size=max(1, int(size)))


def _shm_attach(name):
    """Parent side. 3.13+ can attach without registering with the tracker; before that an attach also
    registers (bpo-39959), which on POSIX is the SAME tracker entry the helper's create made - harmless,
    the helper's unlink removes it once."""
    from multiprocessing import shared_memory
    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:
        return shared_memory.SharedMemory(name=name)


def _drop_block(name):
    """Helper side: close our handle and unlink (POSIX; on Windows close alone frees it)."""
    ent = _child_keep.pop(name, None)
    if ent is None:
        return
    shm = ent[0]
    try:
        shm.close()
    except Exception:
        pass
    try:
        shm.unlink()
    except Exception:
        pass


def _release_child(names=None):
    """Helper side. With names: release exactly those (the parent has copied them). Without: only blocks
    older than STALE_S - something no parent is still reading."""
    if names is not None:
        for n in names:
            _drop_block(n)
        return 1
    now = time.monotonic()
    for n in [n for n, (_s, t0) in _child_keep.items() if now - t0 > STALE_S]:
        _drop_block(n)
    return 1


def _to_shm(arr, made):
    a = np.ascontiguousarray(arr)
    shm = _shm_create(a.nbytes)
    _child_keep[shm.name] = (shm, time.monotonic())
    made.append(shm.name)
    np.ndarray(a.shape, a.dtype, buffer=shm.buf)[...] = a
    return ("shm", shm.name, a.shape, a.dtype.str)


def _is_desc(d):
    return isinstance(d, tuple) and len(d) == 4 and d[0] == "shm"


def _from_shm(desc):
    """Parent side: a private copy of the block. The helper unlinks it afterwards (see _collect)."""
    if not _is_desc(desc):
        return desc                                   # the pickled fallback: already an array
    _, name, shape, dt = desc
    shm = _shm_attach(name)
    try:
        src = np.ndarray(shape, np.dtype(dt), buffer=shm.buf)
        dst = np.empty(shape, np.dtype(dt))
        row_bytes = max(1, src.itemsize * int(np.prod(shape[1:], dtype=np.int64)))
        rows = max(1, COPY_SLICE_BYTES // row_bytes)
        for i in range(0, shape[0] if len(shape) else 1, rows):
            dst[i:i + rows] = src[i:i + rows]
        del src
    finally:
        shm.close()
    return dst


def _song_via_shm(abs_path, music_root, track_id, body_start_s, sections, with_levels):
    """Helper side of decode_song: the arrays go out through shared memory, levels (small) pickled."""
    _release_child()                                  # only the long-abandoned
    samples, stems, levels = _decode_in_process(abs_path, music_root, track_id, body_start_s, sections, with_levels)
    made = []
    try:
        return _to_shm(samples, made), {k: _to_shm(v, made) for k, v in stems.items()}, levels
    except Exception:                                 # no shared memory here: the pickle, as before
        _release_child(made)
        return samples, stems, levels


def _mix_via_shm(abs_path):
    _release_child()
    samples = _decode_mix_in_process(abs_path)
    made = []
    try:
        return _to_shm(samples, made)
    except Exception:
        _release_child(made)
        return samples


def _collect(pool, desc_or_tree):
    """Parent side: copy every block out, then have the helper release exactly this call's blocks."""
    is_song = isinstance(desc_or_tree, tuple) and len(desc_or_tree) == 3 and isinstance(desc_or_tree[1], dict)
    descs = ([desc_or_tree[0]] + list(desc_or_tree[1].values())) if is_song else [desc_or_tree]
    names = [d[1] for d in descs if _is_desc(d)]
    try:
        if is_song:
            s, st, lv = desc_or_tree
            return (_from_shm(s), {k: _from_shm(v) for k, v in st.items()}, lv)
        return _from_shm(desc_or_tree)
    except Exception as e:
        raise _HandoffError(f"{type(e).__name__}: {e}") from e
    finally:
        if names:
            try:
                pool.submit(_release_child, names)    # fire and forget: names, not "everything"
            except Exception:
                pass                                  # pool gone: the tracker unlinks them at exit


def _watch_parent():
    """Helper-process initializer: exit the moment the show process is gone.

    A ProcessPoolExecutor worker only exits when its parent shuts the pool down. A parent that is KILLED
    (a crashed show, a closed planner, a timed-out gate) never does, and on Windows the idle worker then
    lives forever - 23 of them were found on the dev box on 2026-10-25, 1.4 GB between them. The parent's
    sentinel (its process handle on Windows, the spawn pipe on POSIX) becomes ready when it exits."""
    import multiprocessing
    pp = multiprocessing.parent_process()
    if pp is None or pp.sentinel is None:
        return

    def _wait():
        try:
            from multiprocessing.connection import wait
            wait([pp.sentinel])
        except Exception:
            return
        os._exit(0)

    threading.Thread(target=_wait, daemon=True, name="parent-watch").start()


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
                _pool = ProcessPoolExecutor(max_workers=1, mp_context=ctx, initializer=_watch_parent)
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
            fut = pool.submit(_song_via_shm, abs_path, music_root, int(track_id), float(body_start_s),
                              [dict(start_s=float(s["start_s"]), end_s=float(s["end_s"])) for s in (sections or [])], with_levels)
            return _collect(pool, fut.result(timeout=POOL_TIMEOUT_S))
        except ValueError:
            raise
        except _HandoffError as e:
            print(f"[DJ] stem decode handoff failed ({e}) - decoding this song in-process")
            return _decode_in_process(abs_path, music_root, track_id, body_start_s, sections, with_levels)
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
            return _collect(pool, pool.submit(_mix_via_shm, abs_path).result(timeout=POOL_TIMEOUT_S))
        except _HandoffError as e:
            print(f"[DJ] decode handoff failed ({e}) - decoding this song in-process")
            return _decode_mix_in_process(abs_path)
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
