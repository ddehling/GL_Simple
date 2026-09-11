"""Kit decomposition with a LEARNED separator: the drum stem split into kick,
snare, toms and cymbals by drumsep (a hybrid-demucs checkpoint trained for
that, github.com/inagoy/drumsep), so the reader sees each family alone.

Why (2026-09-09, the truth set): the cluster reader's ceiling on a mixed
drum stem was the coincidences - a hat that never plays without a kick,
toms and crashes folded into big clusters. Neither a residual test nor
more clusters could recover them (a handful of broadband templates with
free gains explains any onset patch). Reading each drumsep family on its
own: all-hits F1 0.77 -> 0.87, kick 0.87-1.00 on every song, toms 0.2 ->
0.5-1.0; the open questions move inside the cymbal family.

    separate(y) -> {"kick": mono, "snare": mono, "toms": mono, "cymbals": mono}

The checkpoint (167 MB) lives in models/drumsep/ (gitignored) and is
fetched on first use with gdown from the project's Google Drive file.
"""
import os

import numpy as np

RATE = 44100
MODEL_NAME = "49469ca8"
GDRIVE_ID = "1-Dm666ScPkg8Gt2-lK3Ua0xOudWHZBGC"
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "drumsep")
FAMILY = {"bombo": "kick", "redoblante": "snare", "toms": "toms", "platillos": "cymbals"}
_model = {}


def model_path():
    return os.path.join(ROOT, MODEL_NAME + ".th")


def available():
    return os.path.isfile(model_path())


def ensure_model(progress=None):
    """Download the checkpoint if missing. -> path or None when it cannot be fetched."""
    p = model_path()
    if os.path.isfile(p):
        return p
    os.makedirs(ROOT, exist_ok=True)
    try:
        import gdown
    except Exception:  # noqa: BLE001
        return None
    if progress:
        progress("drumsep: downloading the checkpoint (167 MB)")
    try:
        gdown.download(id=GDRIVE_ID, output=p, quiet=True)
    except Exception:  # noqa: BLE001
        return None
    return p if os.path.isfile(p) else None


def _load(progress=None):
    if "m" in _model:
        return _model["m"]
    if ensure_model(progress) is None:
        raise FileNotFoundError("drumsep checkpoint unavailable (models/drumsep/%s.th)" % MODEL_NAME)
    import torch
    from pathlib import Path
    from demucs.pretrained import get_model
    import demucs.hdemucs, demucs.demucs, demucs.htdemucs  # noqa: E401
    # the checkpoint is a full-model pickle; torch >= 2.6 loads weights-only unless the classes are allowlisted
    with torch.serialization.safe_globals([demucs.hdemucs.HDemucs, demucs.demucs.Demucs, demucs.htdemucs.HTDemucs]):
        m = get_model(MODEL_NAME, repo=Path(ROOT))
    m.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    _model["m"] = m
    return m


MIN_FREE_GPU_MB = 1500         # under this much free GPU memory the separation runs on the CPU: with the GPU
                               # oversubscribed (several torch processes at once) Windows pages its memory and a
                               # 100 s separation that takes 5 s ran for hours (2026-09-09)


def _device(m, progress=None):
    import torch
    dev = next(m.parameters()).device
    if dev.type == "cuda":
        try:
            free, _total = torch.cuda.mem_get_info()
            if free / 2 ** 20 < MIN_FREE_GPU_MB:
                if progress:
                    progress(f"drumsep: {free / 2 ** 20:.0f} MB free on the GPU (other processes hold it) - separating on the CPU")
                m.to("cpu")
                return torch.device("cpu")
        except Exception:  # noqa: BLE001
            pass
    return dev


_recent = []                   # [(key, families)]: the last separations, so the reading and the program build of the
                               # same track (the planner: identify, then hear: reconstruction) separate ONCE
RECENT_MAX = 2


def _key(a):
    return (int(a.shape[0]), float(np.abs(a[:: max(1, a.shape[0] // 4096)]).sum()))


def separate(y, progress=None):
    """y: (n,) or (n, 2) float @44100 -> {family: (n,) float32}."""
    import torch
    from demucs.apply import apply_model
    a = np.asarray(y, dtype=np.float32)
    key = _key(a)
    for k, fam in _recent:
        if k == key:
            return {n: v.copy() for n, v in fam.items()}
    m = _load(progress)
    stereo = np.stack([a, a], axis=1) if a.ndim == 1 else a
    audio = torch.from_numpy(np.ascontiguousarray(stereo.T)).unsqueeze(0)
    ref = audio.mean(0)
    std = ref.std() + 1e-8
    device = _device(m, progress)
    with torch.no_grad():
        out = apply_model(m, audio / std, device=device, shifts=0, split=True, overlap=0.1, progress=False)[0].cpu().numpy() * float(std)
    fam = {FAMILY.get(name, name): out[i].mean(axis=0).astype(np.float32) for i, name in enumerate(m.sources)}
    if device.type == "cuda":
        torch.cuda.empty_cache()                     # the card is shared (the planner, a YourMT3 process)
    _recent.append((key, fam))
    del _recent[:-RECENT_MAX]
    return {n: v.copy() for n, v in fam.items()}
