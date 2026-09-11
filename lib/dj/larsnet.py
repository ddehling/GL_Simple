"""LarsNet: the drum stem split into KICK, SNARE, TOMS, HI-HAT and CYMBALS
(five U-Nets, Mezza et al. 2023, github.com/polimi-ispl/larsnet) - the
family split drumsep lacks: the hi-hat apart from the other cymbals.

Model code vendored under lib/dj/vendor/larsnet (unet.py); the checkpoints
(562 MB, CC BY-NC 4.0 - NON-COMMERCIAL: measured here for research, a
commercial deployment must not ship them) live in models/larsnet/
pretrained_larsnet_models/<stem>/pretrained_<stem>_unet.pth.

    separate(y) -> {"kick", "snare", "toms", "hihat", "cymbals": mono float32}
"""
import os

import numpy as np

RATE = 44100
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "larsnet")
STEMS = ("kick", "snare", "toms", "hihat", "cymbals")
SEGMENT_S, OVERLAP_S = 11.85, 1.0
MIN_FREE_GPU_MB = 1500
_models = {}
_recent = []
RECENT_MAX = 2


def available():
    return all(os.path.isfile(os.path.join(ROOT, "pretrained_larsnet_models", s, f"pretrained_{s}_unet.pth")) for s in STEMS)


def _load(progress=None):
    if _models:
        return _models
    import torch
    from lib.dj.vendor.larsnet.unet import UNetWaveform
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            free, _t = torch.cuda.mem_get_info()
            if free / 2 ** 20 < MIN_FREE_GPU_MB:
                device = "cpu"
                if progress:
                    progress("larsnet: the GPU is held by other processes - separating on the CPU")
        except Exception:  # noqa: BLE001
            pass
    for stem in STEMS:
        p = os.path.join(ROOT, "pretrained_larsnet_models", stem, f"pretrained_{stem}_unet.pth")
        m = UNetWaveform(input_size=(2, 2048, 512), device=device)
        ck = torch.load(p, map_location=device, weights_only=False)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        _models[stem] = m.to(device)
    _models["_device"] = device
    return _models


def _key(a):
    return (int(a.shape[0]), float(np.abs(a[:: max(1, a.shape[0] // 4096)]).sum()))


def separate(y, progress=None):
    """y: (n,) or (n, 2) float @44100 -> {stem: (n,) float32}, the track processed in overlapping segments."""
    import torch
    a = np.asarray(y, dtype=np.float32)
    key = _key(a)
    for k, fam in _recent:
        if k == key:
            return {n: v.copy() for n, v in fam.items()}
    models = _load(progress)
    device = models["_device"]
    stereo = np.stack([a, a], axis=1) if a.ndim == 1 else a
    n = len(stereo)
    seg, ov = int(SEGMENT_S * RATE), int(OVERLAP_S * RATE)
    out = {s: np.zeros(n, dtype=np.float32) for s in STEMS}
    weight = np.zeros(n, dtype=np.float32)
    start = 0
    with torch.no_grad():
        while start < n:
            end = min(n, start + seg)
            chunk = stereo[start:end]
            if len(chunk) < RATE // 2 and start > 0:
                break
            x = torch.from_numpy(np.ascontiguousarray(chunk.T)).unsqueeze(0).to(device)
            w = np.ones(end - start, dtype=np.float32)
            f = min(ov, (end - start) // 4)
            if f > 0:
                ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
                if start > 0:
                    w[:f] = ramp
                if end < n:
                    w[-f:] = ramp[::-1]
            for s in STEMS:
                yhat, _mask = models[s](x)
                mono = yhat.squeeze(0).detach().cpu().numpy().astype(np.float32).mean(axis=0)[: end - start]
                out[s][start:start + len(mono)] += mono * w[: len(mono)]
            weight[start:end] += w
            if end >= n:
                break
            start = end - ov
    weight[weight == 0] = 1.0
    fam = {s: (v / weight).astype(np.float32) for s, v in out.items()}
    _recent.append((key, fam))
    del _recent[:-RECENT_MAX]
    return {n_: v.copy() for n_, v in fam.items()}
