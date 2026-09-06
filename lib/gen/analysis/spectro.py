"""High-resolution spectrograms for comparing an original with its
recreation: log-frequency (30 Hz .. 16 kHz, 256 bins), 86 frames per
second (n_fft 2048, hop 512 at 44.1 kHz), dB from -80 to 0, rendered to
an RGB image (frames x bins x 3, uint8) with a perceptual colour ramp.

    spec, fps = compute(samples_mono)          # spec: (frames, bins) float32 dB, top bin = highest frequency
    rgb = to_rgb(spec)                         # (frames, bins, 3) uint8
    load(path) -> (stereo float32 (n,2), mono float32)

Cached next to the wav as <name>.spec.npz so reopening a folder is instant."""
from __future__ import annotations

import os

import numpy as np

from lib.gen import RATE

N_FFT = 2048
HOP = 512
BINS = 256
FMIN, FMAX = 30.0, 16000.0
DB_LO, DB_HI = -80.0, 0.0


def load(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != RATE:
        idx = np.arange(0, x.shape[0], sr / RATE)
        x = np.stack([np.interp(idx, np.arange(x.shape[0]), x[:, ch]) for ch in range(x.shape[1])], axis=1).astype(np.float32)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return x, x.mean(axis=1).astype(np.float32)


def compute(mono, cache_path=None):
    """(frames, bins) dB spectrogram, log-frequency, plus frames per second."""
    if cache_path and os.path.exists(cache_path):
        try:
            z = np.load(cache_path)
            if z["n_fft"] == N_FFT and z["hop"] == HOP and z["bins"] == BINS:
                return z["spec"], float(z["fps"])
        except Exception:
            pass
    y = np.asarray(mono, dtype=np.float32)
    n = max(0, (len(y) - N_FFT) // HOP + 1)
    if n <= 0:
        return np.full((1, BINS), DB_LO, dtype=np.float32), RATE / HOP
    win = np.hanning(N_FFT).astype(np.float32)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / RATE)
    targets = np.geomspace(FMIN, FMAX, BINS)
    # each log bin averages the linear bins it covers (at least one)
    edges = np.geomspace(FMIN / (targets[1] / targets[0]) ** 0.5, FMAX * (targets[1] / targets[0]) ** 0.5, BINS + 1)
    lin_idx = np.searchsorted(freqs, edges)
    spec = np.empty((n, BINS), dtype=np.float32)
    chunk = 512
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        idx = np.arange(c0, c1)[:, None] * HOP + np.arange(N_FFT)[None, :]
        frames = y[idx] * win
        mag = np.abs(np.fft.rfft(frames, axis=1)) ** 2 / (N_FFT * 0.375)
        for b in range(BINS):
            a, e = lin_idx[b], max(lin_idx[b] + 1, lin_idx[b + 1])
            spec[c0:c1, b] = mag[:, a:e].mean(axis=1)
    spec = 10.0 * np.log10(spec + 1e-12)
    ref = float(np.percentile(spec, 99.5))
    spec = np.clip(spec - ref, DB_LO, DB_HI).astype(np.float32)
    fps = RATE / HOP
    if cache_path:
        try:
            np.savez_compressed(cache_path, spec=spec, fps=fps, n_fft=N_FFT, hop=HOP, bins=BINS)
        except Exception:
            pass
    return spec, fps


# a compact perceptual ramp (dark -> blue -> magenta -> orange -> yellow)
_RAMP = np.array([[0, 0, 4], [20, 12, 60], [70, 20, 110], [130, 30, 120], [190, 55, 90], [235, 110, 40], [250, 180, 30], [252, 250, 160]],
                 dtype=np.float32)


def to_rgb(spec):
    """(frames, bins) dB -> (frames, bins, 3) uint8."""
    u = (np.asarray(spec, dtype=np.float32) - DB_LO) / (DB_HI - DB_LO)
    u = np.clip(u, 0.0, 1.0) * (len(_RAMP) - 1)
    i = np.floor(u).astype(int)
    f = (u - i)[..., None]
    i1 = np.minimum(i + 1, len(_RAMP) - 1)
    rgb = _RAMP[i] * (1.0 - f) + _RAMP[i1] * f
    return np.clip(rgb, 0, 255).astype(np.uint8)


def prepare(path):
    """Load a wav, compute (cached) its spectrogram, return
    {"stereo", "mono", "spec", "rgb", "fps", "seconds"}."""
    stereo, mono = load(path)
    cache = os.path.splitext(path)[0] + ".spec.npz"
    spec, fps = compute(mono, cache)
    return {"stereo": stereo, "mono": mono, "spec": spec, "rgb": to_rgb(spec), "fps": fps, "seconds": len(mono) / RATE, "path": path}
