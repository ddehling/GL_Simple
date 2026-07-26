"""Pre-rendered stem storage for stem-aware mixing.

The vocal pass already runs htdemucs over every track and throws the
separated audio away; this module is the KEEP-IT path. tools/dj/dj_stems.py
renders each track's four stems (drums/bass/other/vocals) once and
encodes them under <music_root>/.stems/<track_id>/; the live system
decodes them at load time so a deck can play any weighted subset
(drums-only entries, acapella tails - the stem_drum_swap / acapella_out
transition styles).

Storage: lossy stereo files (~4-6 MB/stem at 160 kbps -> ~20 MB/track).
Encoding prefers aac, then libmp3lame, then flac, whichever this PyAV
build ships. Presence of all four files == "this track has stems";
there is no DB column - the filesystem is the source of truth, checked
once per library load (TrackInfo.has_stems).

Deps: rendering needs torch + demucs (requirements-dj-vocals.txt, same
stack as the vocal pass) + PyAV (already a core dep for decode).
Playback of existing stems needs neither torch nor demucs.
"""
import os

import numpy as np

RATE = 44100
STEM_NAMES = ("drums", "bass", "other", "vocals")
STEMS_DIRNAME = ".stems"
_EXTS = (".m4a", ".mp3", ".flac")

# Encoder preference: (av codec name, extension, format-ish kwargs)
_ENCODERS = (("aac", ".m4a"), ("libmp3lame", ".mp3"), ("flac", ".flac"))


def stems_dir(music_root, track_id):
    return os.path.join(music_root, STEMS_DIRNAME, str(int(track_id)))


def stem_paths(music_root, track_id):
    """{stem: abs_path} when ALL four stems exist on disk, else None."""
    d = stems_dir(music_root, track_id)
    if not os.path.isdir(d):
        return None
    out = {}
    for name in STEM_NAMES:
        for ext in _EXTS:
            p = os.path.join(d, name + ext)
            if os.path.isfile(p):
                out[name] = p
                break
        else:
            return None
    return out


def has_stems(music_root, track_id):
    return stem_paths(music_root, track_id) is not None


def load_stems(music_root, track_id, expected_len=None, dtype=np.float16):
    """Decode a track's four stems -> {name: (n,2) dtype array}, or None.

    float16 by default: half the RAM of the deck's mix buffer per stem;
    Deck._src_slice upcasts per block. Lengths are aligned to
    expected_len (the deck's mix buffer) when given - encoders pad."""
    paths = stem_paths(music_root, track_id)
    if paths is None:
        return None
    from lib.dj.features import decode_file_stereo
    out = {}
    for name, p in paths.items():
        arr = decode_file_stereo(p)
        if expected_len is not None:
            if len(arr) < expected_len:
                arr = np.concatenate(
                    [arr, np.zeros((expected_len - len(arr), 2),
                                   dtype=arr.dtype)])
            else:
                arr = arr[:expected_len]
        out[name] = arr.astype(dtype)
    return out


def encode_stem(arr, path_no_ext):
    """Encode (n,2) float32 @44100 to the first codec this PyAV build
    supports. Returns the written path."""
    import av
    last_err = None
    # s16 packed = interleaved: row-major (n,2) reshape IS the interleave.
    pcm16 = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
    interleaved = np.ascontiguousarray(pcm16).reshape(1, -1)
    for codec, ext in _ENCODERS:
        path = path_no_ext + ext
        try:
            with av.open(path, "w") as container:
                stream = container.add_stream(codec, rate=RATE)
                try:
                    stream.layout = "stereo"
                except Exception:
                    pass
                if codec in ("aac", "libmp3lame"):
                    stream.bit_rate = 160_000
                frame = av.AudioFrame.from_ndarray(
                    interleaved, format="s16", layout="stereo")
                frame.rate = RATE
                # Encoders want exactly frame_size samples per frame -
                # chunk through a fifo, flush the partial tail at the end.
                fifo = av.AudioFifo()
                fifo.write(frame)
                fs = stream.codec_context.frame_size or 4096
                while True:
                    f = fifo.read(fs)
                    if f is None:
                        break
                    f.pts = None
                    for packet in stream.encode(f):
                        container.mux(packet)
                tail = fifo.read()
                if tail is not None and tail.samples:
                    tail.pts = None
                    for packet in stream.encode(tail):
                        container.mux(packet)
                for packet in stream.encode(None):
                    container.mux(packet)
            return path
        except Exception as e:
            last_err = e
            try:
                os.remove(path)
            except OSError:
                pass
    raise RuntimeError(f"no usable stem encoder in this PyAV build "
                       f"({last_err})")


class StemRenderer:
    """htdemucs full-track separation -> encoded stem files. Construct
    only if lib.dj.vocals.available() (same torch+demucs stack)."""

    def __init__(self):
        import torch
        from demucs.pretrained import get_model
        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = get_model("htdemucs").to(self._device)
        self._model.eval()
        self._sources = list(self._model.sources)   # drums/bass/other/vocals
        if self._device == "cuda":
            print("[DJ stems] using CUDA")

    def render(self, samples_stereo, music_root, track_id):
        """Separate a full track ((n,2) float32 @44100) and write its four
        stem files. Returns the stems dir. split=True chunks internally,
        so full tracks fit in GPU memory."""
        torch = self._torch
        from demucs.apply import apply_model
        d = stems_dir(music_root, track_id)
        os.makedirs(d, exist_ok=True)
        audio = torch.from_numpy(
            np.ascontiguousarray(samples_stereo.T)).unsqueeze(0)
        ref = audio.mean(0)
        std = ref.std() + 1e-8
        audio = audio / std
        with torch.no_grad():
            out = apply_model(self._model, audio, device=self._device,
                              shifts=0, split=True, overlap=0.1,
                              progress=False)[0].cpu().numpy() * float(std)
        for i, name in enumerate(self._sources):
            encode_stem(out[i].T.astype(np.float32),
                        os.path.join(d, name))
        return d
