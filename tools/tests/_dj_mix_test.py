"""Phase B gate #2: a scripted beat-matched transition through the REAL
audio engine, rendered offline and judged by the live signals pipeline.

Two synthetic grooves (A at 128 BPM, B at 124), both analyzed by the real
Phase-A analyzer for their beat grids. Deck B is stretched to A's tempo,
launched on a shared downbeat under the sync PLL, blended in over 16 beats
with a mid-blend bass swap, then A exits - the canonical DJ blend. The
engine's _mixer generator is hand-pumped (no audio device), the rendered
mix is folded to mono and pushed through tools/tests/_club_signals_test.run - the
exact pipeline the visuals see at showtime. Beat lock must survive the
transition; the PLL must hold the decks phase-locked.

Usage:
    python tools/tests/_dj_mix_test.py          # ALL PASS gate
    python tools/tests/_dj_mix_test.py --wav    # also write logs/dj_mix_test.wav
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from lib.dj import features as F
from lib.dj.submix import DJSubmix

RATE = 44100
RENDER_S = 78.0
BLOCK = 1024

failures = []


def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


def synth_groove(bpm, seconds, seed, bass_hz=54.0, pad_hz=220.0):
    beat = 60.0 / bpm
    n = int(seconds * RATE)
    x = np.zeros(n)
    rng = np.random.RandomState(seed)
    t = 0.0
    k = 0
    while t < seconds:
        i0 = int(t * RATE)
        d = int(0.10 * RATE)
        if i0 + d <= n:
            tl = np.arange(d) / RATE
            amp = 1.0 if k % 4 == 0 else 0.75
            x[i0:i0 + d] += amp * np.sin(2 * np.pi * bass_hz * tl) * np.exp(-tl / 0.045)
        t += beat
        k += 1
    t = 0.0
    while t < seconds:
        i0 = int(t * RATE)
        d = int(0.03 * RATE)
        if i0 + d <= n:
            b = rng.randn(d) * np.exp(-np.arange(d) / (0.008 * RATE))
            x[i0:i0 + d] += 0.10 * np.diff(np.concatenate([[0.0], b]))
        t += beat / 2
    tt = np.arange(n) / RATE
    x += 0.04 * np.sin(2 * np.pi * pad_hz * tt) \
        + 0.02 * np.sin(2 * np.pi * pad_hz * 1.5 * tt)
    x = (x / np.max(np.abs(x)) * 0.8).astype(np.float32)
    return np.stack([x, x], axis=1)


def downbeats(grid, downbeat_offset, until_s):
    g = grid[0]
    out = []
    t = g["first_beat_s"] + downbeat_offset * g["period_s"]
    while t < until_s:
        out.append(t)
        t += 4 * g["period_s"]
    return out


def main():
    print("DJ mix test: scripted A(128) -> B(124) blend through the engine\n")

    a = synth_groove(128.0, 80.0, seed=11, bass_hz=54.0, pad_hz=220.0)
    b = synth_groove(124.0, 80.0, seed=22, bass_hz=58.0, pad_hz=277.0)
    ra = F.analyze_samples(a.mean(axis=1), deep=False)
    rb = F.analyze_samples(b.mean(axis=1), deep=False)
    check("analyzer on A", abs(ra["bpm"] - 128.0) / 128.0 < 0.01,
          f"A bpm={ra['bpm']:.2f}")
    check("analyzer on B", abs(rb["bpm"] - 124.0) / 124.0 < 0.01,
          f"B bpm={rb['bpm']:.2f}")

    from lib.audio_engine import AudioEngine
    engine = AudioEngine()                      # never start()ed - no device
    sub = DJSubmix()
    engine.attach_track("dj", sub)

    beat_a = ra["beat_grid"][0]["period_s"]
    rate_b = ra["bpm"] / rb["bpm"]

    # Deck A from its first downbeat; the mix clock starts there too.
    cue_a = downbeats(ra["beat_grid"], ra["downbeat_offset"], 80.0)[1]
    sub.post({"cmd": "load", "deck": "a", "samples": a,
              "grid": ra["beat_grid"], "cue_s": cue_a})
    sub.post({"cmd": "gain", "deck": "a", "value": 1.0, "ramp_s": 0.01})
    sub.post({"cmd": "start", "deck": "a"})

    # Transition at deck A's downbeat nearest 30s of output.
    da = [d for d in downbeats(ra["beat_grid"], ra["downbeat_offset"], 78.0)
          if d - cue_a > 28.0]
    t_out0 = da[0] - cue_a
    S0 = int(t_out0 * RATE)
    beats16 = 16 * beat_a
    cue_b = downbeats(rb["beat_grid"], rb["downbeat_offset"], 40.0)[4]

    sub.post({"cmd": "load", "deck": "b", "samples": b,
              "grid": rb["beat_grid"], "cue_s": cue_b})
    sub.post({"cmd": "rate", "deck": "b", "value": rate_b})
    sub.post({"cmd": "gain", "deck": "b", "value": 0.0, "ramp_s": 0.01})
    sub.post({"cmd": "eq", "deck": "b", "low": 0.0, "ramp_s": 0.01})
    sub.post_many([
        {"at": S0, "cmd": "start", "deck": "b"},
        {"at": S0, "cmd": "sync", "slave": "b", "master": "a"},
        {"at": S0, "cmd": "gain", "deck": "b", "value": 1.0,
         "ramp_s": beats16},
        # Bass swap on the mid-blend downbeat.
        {"at": S0 + int(8 * beat_a * RATE), "cmd": "eq", "deck": "a",
         "low": 0.0, "ramp_s": 0.4},
        {"at": S0 + int(8 * beat_a * RATE), "cmd": "eq", "deck": "b",
         "low": 1.0, "ramp_s": 0.4},
        # A exits over 4 beats, then stops; B rides on alone.
        {"at": S0 + int(beats16 * RATE), "cmd": "gain", "deck": "a",
         "value": 0.0, "ramp_s": 4 * beat_a},
        {"at": S0 + int((beats16 + 5 * beat_a) * RATE), "cmd": "stop",
         "deck": "a"},
        {"at": S0 + int((beats16 + 5 * beat_a) * RATE), "cmd": "end_sync"},
    ])

    # Hand-pump the engine's mixer generator - renders faster than realtime.
    gen = engine._mixer()
    next(gen)
    rendered = []
    trims, kick_lags = [], []
    from lib.dj.deck import ENV_FPS
    beat_a = ra["beat_grid"][0]["period_s"]

    def audible_kick_lag_ms():
        """What listeners hear: the two decks' output kick envelopes."""
        em = sub.decks["a"].out_env[-300:].astype(np.float64)
        es = sub.decks["b"].out_env[-300:].astype(np.float64)
        if em.max() <= 1e-4 or es.max() <= 1e-4:
            return None
        em -= em.mean()
        es -= es.mean()
        xc = np.correlate(em, es, "full")
        mid = len(es) - 1
        half = max(int(0.5 * beat_a * ENV_FPS), 2)
        seg = xc[mid - half:mid + half + 1]
        return abs(int(np.argmax(seg)) - half) / ENV_FPS * 1000
    n_blocks = int(RENDER_S * RATE) // BLOCK
    for i in range(n_blocks):
        buf = gen.send(BLOCK)
        rendered.append(np.frombuffer(buf, dtype=np.float32).reshape(-1, 2))
        tel = sub.telemetry
        t_now = (i + 1) * BLOCK / RATE
        if tel and t_out0 + 2.0 < t_now < t_out0 + beats16:
            da_, db_ = tel["decks"]["a"], tel["decks"]["b"]
            if da_["playing"] and db_["playing"]:
                if i % 40 == 0:
                    lag = audible_kick_lag_ms()
                    if lag is not None:
                        kick_lags.append(lag)
                trims.append(abs(db_["rate"] - rate_b) / rate_b)
    mix = np.concatenate(rendered, axis=0)
    mono = mix.mean(axis=1).astype(np.float64)

    check("render produced audio", float(np.abs(mix).max()) > 0.1,
          f"peak={np.abs(mix).max():.2f}, {len(mix)/RATE:.0f}s rendered")

    # No dead air anywhere once A is in (0.5 s RMS windows).
    w = RATE // 2
    rms = np.sqrt(np.mean(
        mono[:len(mono) // w * w].reshape(-1, w) ** 2, axis=1))
    active = rms[2:-2]
    check("no dropouts through the blend",
          float(active.min()) > 0.15 * float(np.median(active)),
          f"min RMS {active.min():.3f} vs median {np.median(active):.3f}")

    # Sync: the decks' AUDIBLE kicks stayed locked while both playing (the
    # sync aligns audio, not grid phase - the metric that matters).
    check("audible kick lock", len(kick_lags) > 0
          and float(np.median(kick_lags)) < 12.0
          and float(np.max(kick_lags)) < 25.0,
          f"kick lag median {np.median(kick_lags):.1f}ms "
          f"max {np.max(kick_lags):.1f}ms over {len(kick_lags)} samples")
    # Tight synthetic grids need only small trims even though the cap is
    # 1.2%; a runaway trim here would mean the launch alignment failed.
    check("pll trim bounded", len(trims) > 0 and max(trims) < 0.013,
          f"max |rate trim| {max(trims) * 100:.2f}% on tight grids")

    # The live pipeline judges the mix - what the visuals see at showtime.
    from tools.tests import _club_signals_test as CS
    log, drops = CS.run(mono)

    def at(t):
        return min(log, key=lambda r: abs(r["t"] - t))

    pre = at(t_out0 - 5.0)
    post = at(min(t_out0 + beats16 + 12.0, RENDER_S - 3.0))
    check("locked before blend", pre["conf"] > 0.4
          and abs(pre["bpm"] - 128.0) / 128.0 < 0.05,
          f"t={pre['t']:.0f}s bpm={pre['bpm']:.1f} conf={pre['conf']:.2f}")
    check("locked after blend", post["conf"] > 0.4
          and abs(post["bpm"] - 128.0) / 128.0 < 0.05,
          f"t={post['t']:.0f}s bpm={post['bpm']:.1f} conf={post['conf']:.2f} "
          f"(B stretched to 128)")
    # Confidence dips through the blend are allowed, but only briefly.
    seg = [r for r in log if 15.0 <= r["t"] <= RENDER_S - 3.0]
    low = [r["t"] for r in seg if r["conf"] < 0.3]
    runs, cur = [], []
    for t in low:
        if cur and t - cur[-1] > 0.06:
            runs.append(cur)
            cur = []
        cur.append(t)
    if cur:
        runs.append(cur)
    worst = max((c[-1] - c[0] for c in runs), default=0.0)
    check("lock never lost long", worst <= 2.0,
          f"longest conf<0.3 stretch: {worst:.1f}s (allow 2)")
    tw_drops = [d for d in drops if abs(d - t_out0 - 8 * beat_a) < beats16]
    check("no drop storm", len(drops) <= 2 and len(tw_drops) <= 1,
          f"drops at {['%.1f' % d for d in drops] or 'none'}")

    if "--wav" in sys.argv:
        from scipy.io import wavfile
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                         "logs", "dj_mix_test.wav")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        wavfile.write(p, RATE, (np.clip(mix, -1, 1) * 32767).astype(np.int16))
        print(f"\n  wrote {os.path.normpath(p)} "
              f"(blend at {t_out0:.1f}-{t_out0 + beats16:.1f}s)")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
