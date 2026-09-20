"""THE HEADPHONE PREVIEW: a pair from the pairing file rendered offline, eight bars, from the two records' stems
(docs/DJ_MORE_THAN_A_MIXER.md 1).

    vocal over : R's drums, bass and melody from one of its instrumental grooves; S's vocal from its sung stretch,
                 time-stretched to R's tempo and shifted by the pair's semitones (RubberBand offline, R3)
    bass under : R's drums from a groove; S's bass and melody from its landmark, stretched and shifted the same way

No engine, no device: a float32 stereo buffer at the mix rate (and `write_wav` for a file). The decode of two
songs' stems is seconds: render on a worker, never on the UI thread, and keep the `cache` between renders.
"""
import numpy as np

from lib.dj.submix import RATE


def _stems(music_root, track_id, cache=None, loader=None):
    """A song's stems {name: (frames, 2)}: from `loader(track_id)` when given (the planner passes the helper-process
    decoder so this process never decodes next to the audio producer), else decoded here."""
    if cache is not None and track_id in cache:
        return cache[track_id]
    if loader is not None:
        st = loader(int(track_id))
    else:
        from lib.dj.stems import load_stems
        st = load_stems(music_root, int(track_id), dtype=np.float32)
    if cache is not None:
        cache[track_id] = st
    return st


def _slice(arr, t0, n):
    """`n` frames of a (frames, 2) array from time `t0` (zero-padded past the ends)."""
    a = int(max(0.0, t0) * RATE)
    out = np.zeros((n, 2), dtype=np.float32)
    seg = arr[a:a + n]
    out[:len(seg)] = seg
    return out


def _stretch(x, rate, semitones):
    """`x` (frames, 2) played at `rate` (source seconds per output second) and shifted by `semitones`; RubberBand
    offline, or a plain resample when it is not installed (the pitch rides the tempo then)."""
    if abs(rate - 1.0) < 1e-3 and abs(semitones) < 1e-3:
        return x
    try:
        from pylibrb import Option, RubberBandStretcher
    except Exception:
        n_out = int(round(len(x) / max(rate, 1e-6)))
        idx = np.minimum((np.arange(n_out) * rate).astype(np.int64), max(len(x) - 1, 0))
        return x[idx]
    opts = Option.PROCESS_OFFLINE | Option.ENGINE_FINER | Option.CHANNELS_TOGETHER | Option.TRANSIENTS_CRISP | Option.THREADING_NEVER
    st = RubberBandStretcher(sample_rate=RATE, channels=2, options=opts)
    st.time_ratio = 1.0 / max(rate, 1e-6)
    st.pitch_scale = float(2.0 ** (semitones / 12.0))
    data = np.ascontiguousarray(x.T.astype(np.float32))          # (channels, frames)
    st.set_expected_input_duration(data.shape[1])
    st.study(data, True)
    st.process(data, True)
    out = st.retrieve(st.available())
    return np.ascontiguousarray(out.T.astype(np.float32))


def render_pair(music_root, kind, r_track, s_track, r_start_s, s_start_s, bars=8, shift=0, cache=None, s_gain=1.0, loader=None):
    """The pair as heard: (frames, 2) float32 at RATE, peak 0.9. `kind`: "vocal" (S's vocal over R's groove) or
    "bass" (S's bass and melody under R's drums). `r_start_s`: where R's groove begins; `s_start_s`: where S's
    stretch / landmark begins. `loader(track_id) -> stems` decodes elsewhere. Raises ValueError when stems are
    missing."""
    r_st = _stems(music_root, r_track.id, cache, loader)
    s_st = _stems(music_root, s_track.id, cache, loader)
    if not r_st or not s_st:
        raise ValueError("stems missing for " + (s_track.title if r_st else r_track.title))
    bar_r = 4.0 * 60.0 / float(r_track.bpm)
    n = int(round(bars * bar_r * RATE))
    rate = float(r_track.bpm) / float(s_track.bpm)                 # S plays at this rate to sit on R's grid
    if kind == "vocal":
        r_lanes, s_lanes = ("drums", "bass", "other"), ("vocals",)
    else:
        r_lanes, s_lanes = ("drums",), ("bass", "other")
    mix = np.zeros((n, 2), dtype=np.float32)
    for ln in r_lanes:
        if ln in r_st:
            mix += _slice(r_st[ln], r_start_s, n)
    n_src = int(round(n * rate)) + RATE // 10
    s_part = np.zeros((n_src, 2), dtype=np.float32)
    for ln in s_lanes:
        if ln in s_st:
            s_part += _slice(s_st[ln], s_start_s, n_src)
    s_out = _stretch(s_part, rate, shift)
    if len(s_out) < n:
        s_out = np.concatenate([s_out, np.zeros((n - len(s_out), 2), dtype=np.float32)])
    mix += float(s_gain) * s_out[:n]
    peak = float(np.max(np.abs(mix))) if n else 0.0
    if peak > 1e-6:
        mix *= 0.9 / max(peak, 0.9)
    return mix


def write_wav(path, samples, rate=RATE):
    import wave
    x = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(pcm.tobytes())
