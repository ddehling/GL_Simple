"""Offline beat-detector test against real music.

Replicates MicrophoneAnalyzer's exact per-frame pipeline (4096 Hann FFT ->
RMS log-bands -> 0.95/0.05 smoothing -> rolling-20-frame norm_short) on a
decoded audio file, then feeds norm_short[0] to BeatDetector each frame at
40 fps - exactly what it sees live. Cross-checks BeatDetector's BPM against
an independent onset-envelope autocorrelation tempo estimate.

Usage: python tools/tests/_beat_offline_test.py "D:/Media/Desert Whomp" [N_tracks] [window_s] [start_s]
"""
import sys, glob, os
import numpy as np
import miniaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from lib.beat_detector import BeatDetector

RATE = 44100
FPS = 40
CHUNK = 4096
HOP = RATE // FPS                 # samples between analysis frames
WIN_SHORT = 20                    # analyzer default short window (frames)
NUM_BANDS = 32

# Analyzer's log band edges (40 Hz .. 16 kHz) and masks.
freq_bins = np.fft.rfftfreq(CHUNK, 1.0 / RATE)
band_edges = np.logspace(np.log10(40), np.log10(16000), NUM_BANDS + 1)
band_masks = []
for i in range(NUM_BANDS):
    m = (freq_bins >= band_edges[i]) & (freq_bins < band_edges[i + 1])
    if not np.any(m):
        m = (freq_bins >= band_edges[i] * 0.9) & (freq_bins < band_edges[i + 1] * 1.1)
    band_masks.append(m)
hann = np.hanning(CHUNK)


def band_powers_for_window(win):
    mags = np.abs(np.fft.rfft(win * hann))
    mags[0] = 0.0
    bp = np.zeros(NUM_BANDS, dtype=np.float64)
    for i, m in enumerate(band_masks):
        if np.any(m):
            bp[i] = np.sqrt(np.mean(mags[m] ** 2))
    return bp


def autocorr_bpm(onset_env):
    """Independent tempo estimate: autocorrelate the onset envelope, pick the
    strongest lag in the 50..200 BPM range."""
    x = onset_env - onset_env.mean()
    ac = np.correlate(x, x, mode='full')[len(x) - 1:]
    lo = int(round(60.0 / 200.0 * FPS))   # 200 BPM -> small lag
    hi = int(round(60.0 / 50.0 * FPS))    # 50 BPM  -> large lag
    seg = ac[lo:hi + 1]
    if len(seg) == 0 or seg.max() <= 0:
        return 0.0
    lag = lo + int(np.argmax(seg))
    return 60.0 * FPS / lag


def analyze(path, window_s, start_s):
    dec = miniaudio.decode_file(path, output_format=miniaudio.SampleFormat.FLOAT32,
                                nchannels=1, sample_rate=RATE)
    samples = np.asarray(dec.samples, dtype=np.float32)
    start = int(start_s * RATE)
    end = min(len(samples), start + int(window_s * RATE))
    if end - start < CHUNK * 2:
        return None

    bd = BeatDetector()
    dt = 1.0 / FPS
    prev_bp = None
    recent = []                      # rolling raw-band frames (newest first)
    beats = []
    bpm_traj = []
    onset_env = []
    t = 0.0
    pos = start
    while pos + CHUNK <= end:
        win = samples[pos:pos + CHUNK].astype(np.float64)
        bp = band_powers_for_window(win)
        if prev_bp is None:
            prev_bp = bp.copy()
        else:
            bp = 0.95 * bp + 0.05 * prev_bp     # analyzer's light smoothing
            prev_bp = bp.copy()

        recent.insert(0, bp)
        if len(recent) > WIN_SHORT:
            recent.pop()
        mean_short = np.mean(recent, axis=0)
        mean_short = np.where(mean_short < 1e-10, 1e-10, mean_short)
        norm_short_row = bp / mean_short

        onset_env.append(float(np.mean(np.maximum(0.0, norm_short_row[0:24] - 1.0))))

        r = bd.update({"norm_short": np.array([norm_short_row])}, dt)
        if r["onset"]:
            beats.append(t)
        bpm_traj.append(r["bpm"])
        t += dt
        pos += HOP

    beats = np.array(beats)
    ivals = np.diff(beats)
    med_bpm = (60.0 / np.median(ivals)) if len(ivals) >= 2 else 0.0
    # final BPM = mode-ish: median of the last third of the trajectory (settled)
    settled = np.array(bpm_traj[len(bpm_traj) // 3:]) if bpm_traj else np.array([0.0])
    final_bpm = float(np.median(settled[settled > 0])) if np.any(settled > 0) else 0.0
    ac_bpm = autocorr_bpm(np.array(onset_env))
    dur = t
    return dict(beats=len(beats), bps=len(beats) / dur if dur else 0,
                med_bpm=med_bpm, final_bpm=final_bpm, ac_bpm=ac_bpm, dur=dur)


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "D:/Media/Desert Whomp"
    n_tracks = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    window_s = float(sys.argv[3]) if len(sys.argv) > 3 else 75.0
    start_s = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0

    files = sorted(glob.glob(os.path.join(folder, "*.mp3")))[:n_tracks]
    print(f"Testing {len(files)} tracks | {window_s:.0f}s window from {start_s:.0f}s | "
          f"replicated analyzer pipeline @ {FPS}fps\n")
    print(f"{'track':52s} {'beats':>6s} {'b/s':>5s} {'detBPM':>7s} {'medBPM':>7s} {'acBPM':>6s} {'match'}")
    print("-" * 100)
    for path in files:
        name = os.path.basename(path)[:50]
        try:
            r = analyze(path, window_s, start_s)
        except Exception as e:
            print(f"{name:52s}  ERROR: {type(e).__name__}: {e}")
            continue
        if r is None:
            print(f"{name:52s}  (too short)")
            continue
        # "match" if BeatDetector's BPM agrees with the independent autocorr
        # estimate within 4% OR within 4% of an octave (half/double-time).
        d, a = r["final_bpm"], r["ac_bpm"]
        ratios = [d / a if a else 0, d / (2 * a) if a else 0, d / (0.5 * a) if a else 0]
        match = any(abs(x - 1.0) < 0.04 for x in ratios)
        print(f"{name:52s} {r['beats']:6d} {r['bps']:5.2f} {r['final_bpm']:7.1f} "
              f"{r['med_bpm']:7.1f} {r['ac_bpm']:6.1f}  {'OK' if match else 'CHECK'}")


if __name__ == "__main__":
    main()
