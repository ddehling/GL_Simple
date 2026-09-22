"""THE REHEARSAL - the headphone cue (docs/DJ_THE_RECORD_PLAYS.md, step 2).

Two phrases before a seam the conductor renders it from the stems already loaded on the two decks, exactly
as scheduled (the order's steps, the lane gains, the records' level factors), mono, decimated, in a thread,
and judges it with the campaign's instruments: no hole (the 0.5 s RMS never under 0.15 of its median inside
the seam) and no lurch (no 0.5 s step inside the seam beyond the pair's own steps outside it, and never over
6 dB). A seam that fails moves a phrase, then changes its order, then falls back to the breakdown and cut.
The verdict is in the WHY panel and the log: "rehearsed: clean" / "rehearsed: moved a phrase - a hole at
bar 6".
"""
import threading

import numpy as np

from lib.dj.submix import RATE

DECIM = 4                          # 11 kHz mono is plenty for a level series
WIN_S = 0.5
PRE_BARS = 2
TAIL_BARS = 4
DEAD_RATIO = 0.15
LURCH_DB = 6.0
LURCH_MARGIN_DB = 1.5


def _gain_schedule(order, L, n_bars, first, cross_d=1.0, cross_h=1.0):
    """Per bar (index in the render), the gain multiplier of every stem of A and of B: what the seam's steps do,
    at bar granularity. A cross of `n` bars is an EQUAL-POWER ramp over those bars (docs/DJ_THE_BLEND.md 2) -
    the deck ramps the same way; a cut is a switch. `cross_d` / `cross_h`: the drums' and the harmony's cross in
    bars (the SEAMS dial); the harmony cross starts early so it COMPLETES the bar before B's landmark, as the
    conductor places it (_harmony_lead)."""
    ga = {ln: np.ones(n_bars, dtype=np.float32) for ln in ("drums", "bass", "other", "vocals")}
    gb = {ln: np.zeros(n_bars, dtype=np.float32) for ln in ("drums", "bass", "other", "vocals")}

    def ramp(arr, i0, n, up):
        n = max(1, int(round(n)))
        for k in range(n):
            i = i0 + k
            if 0 <= i < n_bars:
                f = (k + 0.5) / n
                arr[i] = float(np.sin(0.5 * np.pi * f)) if up else float(np.cos(0.5 * np.pi * f))
        if i0 + n < n_bars:
            arr[max(0, i0 + n):] = 1.0 if up else 0.0

    def step(bar, a_lanes, b_lanes, cross=1.0):
        i = first + bar
        if i >= n_bars:
            return
        for ln in a_lanes:
            if cross <= 0.0:
                ga[ln][max(0, i):] = 0.0
            else:
                ramp(ga[ln], i, cross, up=False)
        for ln in b_lanes:
            if cross <= 0.0:
                gb[ln][max(0, i):] = 1.0
            else:
                ramp(gb[ln], i, cross, up=True)
    hl = max(0, int(np.ceil(cross_h)) - 1)
    if order == "cut":
        step(0, ("drums", "bass"), (), cross=0.0)
        step(4, ("other", "vocals"), ("drums", "bass", "other"), cross=0.0)
    elif order == "bass":
        h = max(1, 4 - hl)
        step(h, ("bass", "other", "vocals"), ("bass", "other"), cross=cross_h)
        step(L, ("drums",), ("drums",), cross=min(cross_d, 4.0))
    elif order == "vocal":
        step(0, ("vocals",), ("vocals",), cross=0.5)
        step(16, ("drums",), ("drums",), cross=cross_d)
        step(16 + L - hl, ("bass", "other"), ("bass", "other"), cross=cross_h)
    else:
        step(0, ("drums",), ("drums",), cross=cross_d)
        step(max(1, L - hl), ("bass", "other", "vocals"), ("bass", "other"), cross=cross_h)
    return ga, gb


def render(rc, sm):
    """The seam as a level series: (rms_db per 0.5 s, dict of bar marks) or None when it cannot be rendered
    (a deck without stems, a song already sounding)."""
    a, b = sm["a"], sm["b"]
    da, db_ = rc.submix.decks.get(a), rc.submix.decks.get(b)
    sa, sb = rc.songs.get(a), rc.songs.get(b)
    if da is None or db_ is None or da.stems is None or db_.stems is None or sa is None or sb is None or sb.entered:
        return None
    order, L = sm["order"], int(sm["len"])
    last = {"cut": 4, "bass": L, "vocal": 16 + L, "drums": L}.get(order, L)
    n_bars = PRE_BARS + last + TAIL_BARS
    bar_c = rc._bar_clock()
    bar_s = bar_c / RATE
    at_first = rc._next_bar_clock() + (sm["bar"] - rc.bar_n - 1) * bar_c
    clock0 = at_first - PRE_BARS * bar_c
    t_a0 = rc._song_time_at(a, clock0)
    bar_a = float(rc._bar_s(sa.track, t_a0))               # each song's own bar, in its own seconds
    bar_b = float(rc._bar_s(sb.track, float(sm["tb"])))
    # B's time at the seam's first bar: tb is its time at the drum bar (the cue, _seam_cue_b)
    t_b_first = float(sm["tb"]) - (int(sm["drums_bar"]) - int(sm["bar"])) * bar_b
    tb0 = t_b_first - PRE_BARS * bar_b
    fa = rc._trim_whole(a) * float(getattr(da, "loudness_gain", 1.0) or 1.0)
    fb = rc._trim_whole(b) * float(getattr(db_, "loudness_gain", 1.0) or 1.0)
    try:
        cross_d, cross_h = rc._step_beats("drums") / 4.0, rc._step_beats("harmony") / 4.0
    except Exception:
        cross_d = cross_h = 1.0
    ga, gb = _gain_schedule(order, L, n_bars, PRE_BARS, cross_d=cross_d, cross_h=cross_h)
    n_out = int(round(bar_s * RATE / DECIM))
    out = np.zeros(n_bars * n_out, dtype=np.float32)
    alone = np.zeros(n_bars * n_out, dtype=np.float32)   # A playing on by itself: the seam is judged against THAT (A's own steps and breaks are the record's)

    def take(deck, t0, bar_src):
        """One bar of a deck's stems from song time t0 (bar_src seconds of it), mono, decimated to n_out samples."""
        n_src = int(round(bar_src * RATE))
        i0 = int(round(t0 * RATE))
        segs = {}
        for ln, arr in deck.stems.items():
            if i0 >= len(arr) or i0 + n_src <= 0:
                segs[ln] = np.zeros(n_out, dtype=np.float32)
                continue
            j0, j1 = max(0, i0), min(len(arr), i0 + n_src)
            m = np.asarray(arr[j0:j1:DECIM], dtype=np.float32).mean(axis=1) if j1 > j0 else np.zeros(0, dtype=np.float32)
            if len(m) != n_out:
                m = np.interp(np.linspace(0, max(len(m) - 1, 0), n_out), np.arange(len(m)), m) if len(m) > 1 else np.zeros(n_out, dtype=np.float32)
            segs[ln] = m.astype(np.float32)
        return segs
    for i in range(n_bars):
        sa_i = take(da, t_a0 + i * bar_a, bar_a)
        sb_i = take(db_, tb0 + i * bar_b, bar_b)
        acc = np.zeros(n_out, dtype=np.float32)
        acc_a = np.zeros(n_out, dtype=np.float32)
        for ln in ("drums", "bass", "other", "vocals"):
            if ln in sa_i:
                acc += sa_i[ln] * (fa * ga[ln][i])
                acc_a += sa_i[ln] * fa
            if ln in sb_i:
                acc += sb_i[ln] * (fb * gb[ln][i])
        out[i * n_out:(i + 1) * n_out] = acc
        alone[i * n_out:(i + 1) * n_out] = acc_a
        threading.Event().wait(0.0)                    # (let the audio thread through between bars)
    w = int(WIN_S * RATE / DECIM)
    n = len(out) // w
    rms = np.sqrt(np.mean(out[:n * w].reshape(-1, w) ** 2, axis=1))
    rms_a = np.sqrt(np.mean(alone[:n * w].reshape(-1, w) ** 2, axis=1))
    marks = {"first": PRE_BARS, "swap": PRE_BARS + last, "bar_s": bar_s, "win_s": WIN_S}
    return rms, marks, rms_a


def judge(rms, marks, rms_a=None):
    """(ok, why): a hole or a lurch inside the seam that the record ALONE would not have had, named by its
    bar (a 9 dB step that is A's own breakdown is A's, and the seam moved a phrase for nothing)."""
    win_per_bar = max(1, int(round(marks["bar_s"] / marks["win_s"])))
    i0 = marks["first"] * win_per_bar
    i1 = min(len(rms), (marks["swap"] + 1) * win_per_bar)
    if i1 - i0 < 4 or len(rms) < i1:
        return True, "too short to judge"
    act = np.maximum(rms, 1e-6)
    act_a = np.maximum(rms_a, 1e-6) if rms_a is not None and len(rms_a) == len(rms) else None
    inside = act[i0:i1]
    ratio = float(inside.min() / max(np.median(act), 1e-9))
    if ratio < DEAD_RATIO:
        k = int(np.argmin(inside)) + i0
        own = act_a is not None and float(act_a[i0:i1].min()) < 2.0 * float(inside.min())   # A alone is as quiet there (within 6 dB): its own break
        if not own:
            return False, f"a hole at bar {k // win_per_bar - marks['first']} ({20 * np.log10(ratio):.0f} dB under the median)"
    steps = 20 * np.abs(np.diff(np.log10(act)))
    ins = steps[max(0, i0 - 1):i1]
    outs = np.concatenate([steps[:max(0, i0 - 1)], steps[i1:]])
    lurch = float(ins.max()) if len(ins) else 0.0
    solo = float(outs.max()) if len(outs) else 0.0
    if act_a is not None:
        steps_a = 20 * np.abs(np.diff(np.log10(act_a)))
        own_in = steps_a[max(0, i0 - 1):i1]
        solo = max(solo, float(own_in.max()) if len(own_in) else 0.0)   # the record's own steps inside the window count as its own
    line = max(LURCH_DB, solo + LURCH_MARGIN_DB)
    if lurch > line:
        k = int(np.argmax(ins)) + max(0, i0 - 1)
        signed = float(20 * np.log10(act[k + 1] / act[k])) if k + 1 < len(act) else lurch   # + = louder after the step
        if lurch <= line + 1.0:
            # within a decibel of the line: the instrument's own noise, not a lurch - a seam re-shaped or moved on
            # a 9 dB step against the records' own 8 changed what arrived for nothing (the remix gate, 2026-09-19)
            return True, f"clean within a decibel of the line (a {lurch:.0f} dB step at bar {k // win_per_bar - marks['first']} against the records' own {solo:.0f} dB)"
        return False, f"a {signed:+.0f} dB step at bar {k // win_per_bar - marks['first']} (the records' own steps: {solo:.0f} dB)"
    return True, f"clean (steps inside {lurch:.0f} dB, the records' own {solo:.0f} dB, floor {ratio:.2f} of the median)"


def render_wav(rc, sm, pre_bars=PRE_BARS, tail_bars=TAIL_BARS):
    """The seam as AUDIO - stereo, full rate, the same schedule and levels as the judge's render: what you would
    hear (docs/DJ_THE_BLEND.md, "hear it before it plays"). Returns (samples (n, 2) float32, RATE) or None."""
    a, b = sm["a"], sm["b"]
    da, db_ = rc.submix.decks.get(a), rc.submix.decks.get(b)
    sa, sb = rc.songs.get(a), rc.songs.get(b)
    if da is None or db_ is None or da.stems is None or db_.stems is None or sa is None or sb is None:
        return None
    order, L = sm["order"], int(sm["len"])
    last = {"cut": 4, "bass": L, "vocal": 16 + L, "drums": L}.get(order, L)
    n_bars = pre_bars + last + tail_bars
    bar_c = rc._bar_clock()
    at_first = rc._next_bar_clock() + (sm["bar"] - rc.bar_n - 1) * bar_c
    clock0 = at_first - pre_bars * bar_c
    t_a0 = rc._song_time_at(a, clock0)
    bar_a = float(rc._bar_s(sa.track, t_a0))
    bar_b = float(rc._bar_s(sb.track, float(sm["tb"])))
    t_b_first = float(sm["tb"]) - (int(sm["drums_bar"]) - int(sm["bar"])) * bar_b
    tb0 = t_b_first - pre_bars * bar_b
    fa = rc._trim_whole(a) * float(getattr(da, "loudness_gain", 1.0) or 1.0)
    fb = rc._trim_whole(b) * float(getattr(db_, "loudness_gain", 1.0) or 1.0)
    try:
        cross_d, cross_h = rc._step_beats("drums") / 4.0, rc._step_beats("harmony") / 4.0
    except Exception:
        cross_d = cross_h = 1.0
    ga, gb = _gain_schedule(order, L, n_bars, pre_bars, cross_d=cross_d, cross_h=cross_h)
    n_bar = int(round(bar_c))                          # the room's bar in samples (both songs run at the clock)
    out = np.zeros((n_bars * n_bar, 2), dtype=np.float32)

    def take(deck, t0, bar_src):
        n_src = int(round(bar_src * RATE))
        i0 = int(round(t0 * RATE))
        segs = {}
        for ln, arr in deck.stems.items():
            if i0 >= len(arr) or i0 + n_src <= 0:
                segs[ln] = np.zeros((n_bar, 2), dtype=np.float32)
                continue
            j0, j1 = max(0, i0), min(len(arr), i0 + n_src)
            m = np.asarray(arr[j0:j1], dtype=np.float32) if j1 > j0 else np.zeros((0, 2), dtype=np.float32)
            if len(m) != n_bar:
                idx = np.linspace(0, max(len(m) - 1, 0), n_bar)
                m = np.stack([np.interp(idx, np.arange(len(m)), m[:, c]) for c in range(2)], axis=1) if len(m) > 1 else np.zeros((n_bar, 2), dtype=np.float32)
            segs[ln] = m.astype(np.float32)
        return segs
    for i in range(n_bars):
        sa_i = take(da, t_a0 + i * bar_a, bar_a)
        sb_i = take(db_, tb0 + i * bar_b, bar_b)
        acc = np.zeros((n_bar, 2), dtype=np.float32)
        for ln in ("drums", "bass", "other", "vocals"):
            if ln in sa_i:
                acc += sa_i[ln] * (fa * ga[ln][i])
            if ln in sb_i:
                acc += sb_i[ln] * (fb * gb[ln][i])
        out[i * n_bar:(i + 1) * n_bar] = acc
        threading.Event().wait(0.0)
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 0.98:
        out *= 0.98 / peak
    return out, RATE


def write_wav(path, samples, rate):
    import wave
    x = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(pcm.tobytes())


def rehearse(rc, sm, done):
    """Render and judge in a thread; `done(ok, why)` is called on the conductor's next step via rc._rehearsal_result.
    When `rc.seam_wav_dir` is set the seam is also rendered as audio to `<dir>/seam_<seq>_<B>.wav` (the last
    twelve kept) and named in `rc.last_seam_wav` - HEAR THE SEAM plays it; a 'seam' flag keeps it."""
    seq = sm.get("seq")

    def work():
        try:
            r = render(rc, sm)
            if r is None:
                rc._rehearsal_result = (seq, None, "not rendered (a deck without stems, or the song is sounding already)")
                return
            ok, why = judge(*r)
            rc._rehearsal_result = (seq, ok, why)
        except Exception as e:  # noqa: BLE001
            rc._rehearsal_result = (seq, None, f"{type(e).__name__}: {e}")
            return
        wdir = getattr(rc, "seam_wav_dir", None)
        if wdir:
            try:
                import os
                import time
                os.makedirs(wdir, exist_ok=True)
                w = render_wav(rc, sm)
                if w is not None:
                    sb = rc.songs.get(sm["b"])
                    safe = "".join(ch if ch.isalnum() else "_" for ch in (sb.track.title[:24] if sb else "b"))
                    path = os.path.join(wdir, f"seam_{seq:04d}_{time.strftime('%H%M%S')}_{safe}.wav")
                    write_wav(path, *w)
                    rc.last_seam_wav = (seq, path)
                    old = sorted(f for f in os.listdir(wdir) if f.startswith("seam_") and f.endswith(".wav"))
                    for f in old[:-12]:
                        try:
                            os.remove(os.path.join(wdir, f))
                        except OSError:
                            pass
            except Exception as e:  # noqa: BLE001
                rc.last_error = f"seam wav: {type(e).__name__}: {e}"
    threading.Thread(target=work, daemon=True, name="remix-rehearse").start()
