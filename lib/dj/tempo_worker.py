"""Out-of-process beat-grid measurement for DJSystem._verify_tempo.

Run as ``python -m lib.dj.tempo_worker``. Reads length-prefixed pickled
mono windows on stdin, writes length-prefixed pickled results on stdout,
one response per request, until EOF.

WHY A HAND-ROLLED SUBPROCESS instead of multiprocessing/ProcessPoolExecutor:
both the 'spawn' and 'forkserver' start methods re-import the parent's
``__main__`` in the child (forkserver's preload list is literally
``['__main__']``). In the show that means a SECOND full import of
Stories_OGL - every shader, Flask, GL binding - inside the helper
process, landing right when a transition is being planned on a 4-core
box. This module imports only lib.dj.features. 'fork' is not an option
either: forking a process that has a live audio callback thread can
deadlock the child on a lock that thread was holding.
"""
import pickle
import struct
import sys

_HDR = struct.Struct("<Q")


def _read_exactly(stream, n):
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _send(stream, obj):
    blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_HDR.pack(len(blob)))
    stream.write(blob)
    stream.flush()


class _ParentGone(Exception):
    """The show exited while we were mid-reply - not an error worth a
    traceback on the console."""


def main():
    # Anything the analysis imports may print; keep stdout pure for the
    # protocol by pointing stray writes at stderr.
    real_stdout = sys.stdout.buffer
    sys.stdout = sys.stderr

    from lib.dj.features import verify_tempo_window

    stdin = sys.stdin.buffer
    while True:
        hdr = _read_exactly(stdin, _HDR.size)
        if hdr is None:
            return 0
        (n,) = _HDR.unpack(hdr)
        blob = _read_exactly(stdin, n)
        if blob is None:
            return 0
        try:
            mono = pickle.loads(blob)
            result = verify_tempo_window(mono)
        except Exception as e:                   # never take the parent down
            result = ("error", f"{type(e).__name__}: {e}")
        try:
            _send(real_stdout, result)
        except (BrokenPipeError, ValueError):
            return 0                             # parent shut down; just go


if __name__ == "__main__":
    sys.exit(main())
