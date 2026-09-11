"""The `other` stem split into GUITAR, PIANO and the rest with demucs'
six-stem model (htdemucs_6s), so the `other` reader sees each family alone.

Why (2026-09-09, the render gate): the parts that play through the wrong
instrument in the reconstruction are the ones sharing one voice in the
`other` stem - piano, pluck and pad as one voice (house), horns inside the
clav voice (funk), a lead guitar inside the power-chord voice (rock). No
clustering rule separated them (k-means, GMM, agglomerative all measured);
a learned separation of exactly those instruments is the cheapest attack.

    separate_other(y) -> {"guitar": mono, "piano": mono, "rest": mono}

The model runs on the `other` STEM (not the mix): its guitar and piano
outputs are the families, everything else it emits (its "other", and the
bleed it puts into "drums", "bass", "vocals") is summed as the rest, so no
content is lost. Weights come from demucs' own remote on first use.
"""
import numpy as np

RATE = 44100
MODEL_NAME = "htdemucs_6s"
FAMILIES = ("guitar", "piano", "rest")
MIN_FREE_GPU_MB = 1500
_model = {}
_recent = []
RECENT_MAX = 2


def _load(progress=None):
    if "m" in _model:
        return _model["m"]
    import torch
    from demucs.pretrained import get_model
    if progress:
        progress(f"sixstem: loading {MODEL_NAME}")
    m = get_model(MODEL_NAME)
    m.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    _model["m"] = m
    return m


def _device(m, progress=None):
    import torch
    dev = next(m.parameters()).device
    if dev.type == "cuda":
        try:
            free, _total = torch.cuda.mem_get_info()
            if free / 2 ** 20 < MIN_FREE_GPU_MB:
                if progress:
                    progress(f"sixstem: {free / 2 ** 20:.0f} MB free on the GPU - separating on the CPU")
                m.to("cpu")
                return torch.device("cpu")
        except Exception:  # noqa: BLE001
            pass
    return dev


def _key(a):
    return (int(a.shape[0]), float(np.abs(a[:: max(1, a.shape[0] // 4096)]).sum()))


def separate_other(y, progress=None):
    """y: (n,) or (n, 2) float @44100 (the `other` stem) -> {family: (n,) float32}."""
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
    mono = {name: out[i].mean(axis=0).astype(np.float32) for i, name in enumerate(m.sources)}
    fam = {"guitar": mono.get("guitar", np.zeros_like(a)), "piano": mono.get("piano", np.zeros_like(a))}
    rest = np.zeros_like(fam["guitar"])
    for name, v in mono.items():
        if name not in ("guitar", "piano"):
            rest += v
    fam["rest"] = rest
    _recent.append((key, fam))
    del _recent[:-RECENT_MAX]
    return {n: v.copy() for n, v in fam.items()}
