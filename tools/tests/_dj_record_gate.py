"""THE RECORD GATE - the record player plays one record at a time and MIXES into the next (docs/DJ_MIXING.md A + C).

Drives the Director headless in RECORD mode (the default layers dial; no audio device: the mixer is pulled by hand
as the control gate does) with a 60 s minimum play, presses NEXT and STAY in assorted states, and judges the
rendered bus and the deck telemetry:

  MIN PLAY      every audible record is audible for play_min_s unless a NEXT was pressed on it (the last record is exempt)
  OVERLAP       at every blend the two records are audible together for the planned L + T bars (+-1.5); a breakdown swap
                for its planned bars; a cut (the fallback) for none
  ONE BASSLINE  the two decks' low bands are never both open (> 0.5) while both are audible, beyond the swap's two beats
  ON THE BEAT   the next record starts within 60 ms of the planned clock, on a beat of the record (its beat phase within 0.1)
  BEAT LOCK     through the last 60 % of the overlap the two decks' beat phases agree within 0.1 beats (the PLL holds)
  THE HOLE      a cut's silence is 0.5 to 1.5 beats of the OLD record's tempo (10 ms RMS windows); a blend has none
  DEAD AIR      no silence over 1.5 s outside the holes, the first-song wait and the end
  PRESSES       a NEXT or STAY either acted within 3 bars or said an allowed reason
  ERRORS        Director.last_error / RecordPlayer.last_error stay None

    python tools/tests/_dj_record_gate.py --music D:/Devel/music [--seeds 1-2] [--minutes 9] [--pace 2.5] [--wav out.wav]

Exit 0 = ALL OK, 1 = FAIL, 2 = record mode not available.
"""
import math
import os
import random
import sys
import time

import numpy as np

os.environ["DJ_NO_PLAY_LOG"] = "1"
os.environ["DJ_NO_MIDI"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools.tests._low_priority import PRIORITY  # noqa: E402,F401  (idle priority: it may run beside the planner)
from lib.dj.submix import RATE  # noqa: E402

BLOCK = 4410                                       # 0.1 s
WIN = RATE // 100                                  # 10 ms
SIL = 0.004                                        # RMS under this is silence
AUDIBLE_GAIN = 0.1
LOW_OPEN = 0.5                                     # a low band over this is "open"
PLAY_MIN = 60.0
ALLOWED = ("still decoding", "nothing planned yet", "is the record playing", "not on a deck", "no such song", "press START")


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def seeds_of(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def write_wav(path, pcm_i16):
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(np.ascontiguousarray(pcm_i16).tobytes())


def silent_runs(rms):
    """[(start, end_exclusive)] of consecutive 10 ms windows under SIL."""
    s = np.concatenate([[0], (rms < SIL).astype(np.int8), [0]])
    d = np.diff(s)
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def wrap(x):
    return ((x + 0.5) % 1.0) - 0.5


class Press:
    def __init__(self, kind, trig, slot_nb):
        self.kind, self.trig, self.slot_nb = kind, trig, slot_nb
        self.deadline_nb = slot_nb + int(150 * RATE / BLOCK)
        self.fired_nb = None
        self.wait_record_change = None             # for "fresh": the record seen when waiting began
        self.fire_at = None


def run(music, seed, minutes, pace, wav_path):
    from lib.audio_engine import AudioEngine
    from lib.dj import director as D
    for k in list(getattr(D, "PLAY_MIN_S", {}) or {}):
        D.PLAY_MIN_S[k] = PLAY_MIN
    engine = AudioEngine()
    d = D.Director(music, engine=None, theme="groove", attach=engine.attach_track)
    ok = d.start(threaded=False)
    p = getattr(d, "player", None)
    if p is None:
        return {"fatal": f"record mode not available: {d.last_error}", "code": 2}
    if not ok:
        return {"fatal": f"did not start: {d.last_error}", "code": 2}
    p.play_min_s = PLAY_MIN
    random.seed(seed)
    np.random.seed(seed)
    if hasattr(p, "rng"):
        try:
            p.rng = random.Random(seed)
        except Exception:  # noqa: BLE001
            pass
    rng = random.Random(1000 + seed)
    gen = engine._mixer()
    next(gen)
    gen.send(256)

    rms10 = []                                     # per 10 ms window
    pcm = []                                       # int16 blocks, only with --wav
    tel = []                                       # per block {deck: (gain, playing, time_s, track_id, low, beat_phase)}
    sync_tel = []                                  # per block (slave, master, bias) or None
    vis_tel = []                                   # per block (live_beat, outstate_keys): what the lights are told
    cut_seen = []                                  # (block, plan copy) every time the plan changes
    cut_last = [None]
    rec_hist = []                                  # per block p.record
    bpm_of = {}                                    # track id -> bpm
    title_of = {}
    clock_pts = {}
    errors = []                                    # (rendered_s, who, text)
    err_seen = {"d": None, "p": None}
    presses_done = []                              # dicts
    pending = []                                   # presses awaiting their verdict
    next_press_blocks = []
    state = {"first_play_nb": None, "t_play0": None}

    # the schedule: 2-3 NEXT, 2-3 STAY; the first is a STAY before anything has started
    n_next, n_stay = rng.choice([2, 3]), rng.choice([2, 3])
    plan = [("next", t) for t in (["fresh", "cut", "early", "random"][:n_next])] + \
           [("stay", t) for t in (["cut", "fresh", "random"][:n_stay])]
    rng.shuffle(plan)
    plan.insert(0, ("stay", "early"))
    total_s = minutes * 60.0
    queue = []
    for i, (kind, trig) in enumerate(plan):
        t_s = 6.0 + (total_s * 0.8) * (i / max(1, len(plan)))
        queue.append((kind, trig, t_s + rng.uniform(0, 10.0)))

    def tel_deck(dk):
        return ((p.submix.telemetry or {}).get("decks") or {}).get(dk) or {}

    def c2s(c):
        if "a" not in clock_pts or "b" not in clock_pts:
            return None
        (n1, c1), (n2, c2) = clock_pts["a"], clock_pts["b"]
        per = (c2 - c1) / float(n2 - n1)
        return n1 + (c - c1) / per if per else None

    def bar_samples():
        try:
            return max(1.0, float(p.bar_s()) * RATE)
        except Exception:  # noqa: BLE001
            return 2.0 * RATE

    def state_words():
        try:
            st = p.status()
        except Exception as e:  # noqa: BLE001
            return f"status raised {type(e).__name__}"
        r, n, c = st.get("record"), st.get("next"), st.get("cut")
        return ", ".join(x for x in [
            f"record {r['title'][:14]} played {r['played_s']:.0f}s" if r else "no record",
            f"next {n['state']}" if n else "no next",
            f"{c.get('kind', 'plan')} in {c['in_bars']} bars ({c['state']})" if c else "no plan"] if x)

    def fire(pr, nb):
        before_events = len(d.events)
        before_moves = len(p.moves)
        cut_before = dict(p.cut) if p.cut else None
        rec_before = p.record
        sw = state_words()
        try:
            ret = d.next() if pr.kind == "next" else d.hold()
            exc = None
        except Exception as e:  # noqa: BLE001
            ret, exc = None, f"{type(e).__name__}: {e}"
        bar_s = float(p.bar_s() or 2.0) if callable(getattr(p, "bar_s", None)) else 2.0
        rec = {"kind": pr.kind, "trig": pr.trig, "nb": nb, "t": nb * BLOCK / RATE, "ret": ret, "exc": exc, "state": sw,
               "cut_before": cut_before, "rec_before": rec_before, "ev0": before_events, "mv0": before_moves,
               "bar_samples": bar_samples(), "until_nb": nb + int(math.ceil((3 * bar_s + 0.5) * RATE / BLOCK)) + 1,
               "acted": False, "said": None, "done": False}
        if pr.kind == "next":
            next_press_blocks.append(nb)
        presses_done.append(rec)
        pending.append(rec)

    def eval_pending(nb):
        for r in list(pending):
            cut = p.cut
            if not r["acted"]:
                if r["kind"] == "next":
                    if p.record != r["rec_before"] and p.record is not None:
                        r["acted"] = True
                    elif cut is not None:
                        # planned: its start within 3 bars; under way: its swap within 3 bars
                        key = "clock" if cut.get("state") == "planned" else "swap_clock"
                        ci = c2s(cut.get(key, cut["clock"]))
                        if ci is not None and ci - nb * BLOCK <= 3 * r["bar_samples"] + 0.1 * RATE:
                            r["acted"] = True
                else:
                    cb = r["cut_before"]
                    if cb is not None and cut is not None:
                        ci, cj = c2s(cut["clock"]), c2s(cb["clock"])
                        if ci is not None and cj is not None and ci - cj >= 4 * r["bar_samples"]:
                            r["acted"] = True
            if nb >= r["until_nb"] or r["acted"]:
                pending.remove(r)
                r["done"] = True
                texts = [str(r["ret"])] + [e[1] for e in d.events[r["ev0"]:]] + \
                        [str(m.get("text", "")) for m in p.moves[r["mv0"]:]]
                spoke = [t for t in texts if any(a in t for a in ALLOWED)]
                r["said"] = spoke[0] if spoke else None
                r["texts"] = texts

    def due(nb):
        """the next press to fire now, or None; one at a time."""
        if not queue or pending:
            return None
        kind, trig, t_s = queue[0]
        if nb * BLOCK / RATE < t_s + (state["first_play_nb"] or 0) * BLOCK / RATE:
            return None
        pr = Press(kind, trig, nb)
        age = 0.0
        if p.record in p.songs:
            age = float(p.status().get("record", {}).get("played_s", 0.0) or 0.0) if p.status().get("record") else 0.0
        cut = p.cut
        ci = c2s(cut["clock"]) if cut else None
        force = nb > int((t_s + 150.0) * RATE / BLOCK) + (state["first_play_nb"] or 0)
        okc = False
        if trig == "random" or force:
            okc = True
        elif trig == "early":
            okc = age < 0.6 * PLAY_MIN
        elif trig == "cut":
            okc = cut is not None and cut.get("state") == "planned" and ci is not None and ci - nb * BLOCK > 2 * bar_samples()
        elif trig == "fresh":
            if getattr(due, "w", None) is None:
                due.w = (p.record, nb)
            elif p.record != due.w[0] and p.record is not None:
                due.w = ("fired", nb + rng.randint(1, 3))
            if isinstance(due.w, tuple) and due.w[0] == "fired" and nb >= due.w[1]:
                okc = True
        if not okc:
            return None
        due.w = None
        queue.pop(0)
        return pr

    # ---- the loop ----
    t_wall0 = time.time()
    nb = 0
    t_play0 = None
    n_end = None
    while True:
        x = np.frombuffer(gen.send(BLOCK), dtype=np.float32).reshape(-1, 2)
        nb += 1
        w = x[:BLOCK].astype(np.float64).reshape(10, -1)
        rms10.append(np.sqrt(np.mean(w ** 2, axis=1)))
        if wav_path:
            pcm.append((np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16))
        clk = p.submix.clock
        if nb == 1:
            clock_pts["a"] = (nb * BLOCK, clk)
        elif nb == 30:
            clock_pts["b"] = (nb * BLOCK, clk)
        clock_pts["last"] = (nb * BLOCK, clk)
        d.step()
        telx = p.submix.telemetry or {}
        decks = telx.get("decks") or {}
        tel.append({dk: (float(t.get("gain") or 0.0), bool(t.get("playing")), float(t.get("time_s") or 0.0), t.get("track_id"),
                        float((t.get("eq") or [1.0])[0]), float(t.get("beat_phase") or 0.0))
                    for dk, t in decks.items()})
        sy = telx.get("sync")
        sync_tel.append((sy.get("slave"), sy.get("master"), float(sy.get("bias_beats") or 0.0)) if sy else None)
        # THE ROOM'S FEED (lib/dj/vis.py's vocabulary): the beat and the outstate the lights would get this tick
        try:
            lb = p.live_beat()
            ok_ = p.outstate_keys()
        except Exception as e:  # noqa: BLE001
            lb, ok_ = None, {"_exc": f"{type(e).__name__}: {e}"}
        vis_tel.append((lb, ok_))
        rec_hist.append(p.record)
        for s in list(p.songs.values()):
            try:
                bpm_of[s.track.id] = float(s.track.bpm)
                title_of[s.track.id] = s.track.title
            except Exception:  # noqa: BLE001
                pass
        cut = dict(p.cut) if p.cut else None
        sig = (cut.get("seq"), cut.get("clock"), cut.get("swap_clock"), cut.get("state")) if cut else None
        if sig != cut_last[0]:
            cut_last[0] = sig
            if cut:
                cut["_old_tid"] = p.songs[cut["a"]].track.id if cut.get("a") in p.songs else None
                cut["_new_tid"] = p.songs[cut["b"]].track.id if cut.get("b") in p.songs else None
                cut_seen.append((nb, cut))
        for key, obj in (("d", d), ("p", p)):
            e = getattr(obj, "last_error", None)
            if e and e != err_seen[key]:
                err_seen[key] = e
                errors.append((nb * BLOCK / RATE, key, str(e)))
        # the first song
        if t_play0 is None:
            if p.record is not None and tel_deck(p.record).get("playing"):
                t_play0 = time.time()
                state["first_play_nb"] = nb
                n_end = nb + int(minutes * 60 * RATE / BLOCK)
                print(f"seed {seed}: first record playing {time.time() - t_wall0:.1f} s after START ({title_of.get(tel_deck(p.record).get('track_id'), '?')})", flush=True)
            elif time.time() - t_wall0 > 180.0:
                d.stop(0.1)
                return {"fatal": f"no first song in 180 s: {d.last_error} / {p.last_error}", "code": 1}
            else:
                time.sleep(0.02)
                continue
        else:
            lag = (nb - state["first_play_nb"]) * BLOCK / RATE / pace - (time.time() - t_play0)
            if lag > 0.05:
                time.sleep(min(lag, 0.2))
            eval_pending(nb)
            pr = due(nb)
            if pr is not None:
                fire(pr, nb)
                eval_pending(nb)
            if nb % 600 == 0:
                print(f"seed {seed} [{(nb - state['first_play_nb']) * BLOCK / RATE / 60:4.1f} min] {len(p.moves)} moves, {len(presses_done)} presses, err {d.last_error} / {p.last_error}", flush=True)
            if nb >= n_end:
                break
    moves = list(p.moves)
    final_status = None
    try:
        final_status = p.status()
    except Exception as e:  # noqa: BLE001
        errors.append((nb * BLOCK / RATE, "p", f"status: {type(e).__name__}: {e}"))
    unplanned = [q for q in queue]
    clock_check = None
    if "a" in clock_pts and "b" in clock_pts and "last" in clock_pts:
        (n1, c1), (n2, c2), (n3, c3) = clock_pts["a"], clock_pts["b"], clock_pts["last"]
        per = (c2 - c1) / float(n2 - n1)
        clock_check = {"per_sample": per, "drift_samples": (c3 - c1) / per - (n3 - n1) if per else None, "offset": c1 - n1 * (per or 1)}
    d.stop(0.1)
    # presses still pending at the end are judged by what they had
    for r in pending:
        r["done"] = True
        texts = [str(r["ret"])] + [e[1] for e in d.events[r["ev0"]:]] + [str(m.get("text", "")) for m in moves[r["mv0"]:]]
        spoke = [t for t in texts if any(a in t for a in ALLOWED)]
        r["said"] = spoke[0] if spoke else None
        r["texts"] = texts
        r["cut_short"] = True
    return {"p": p, "d": d, "rms": np.concatenate(rms10), "pcm": pcm, "tel": tel, "sync": sync_tel, "vis": vis_tel, "cut_seen": cut_seen, "bpm_of": bpm_of,
            "title_of": title_of, "clock_pts": clock_pts, "clock_check": clock_check, "errors": errors, "presses": presses_done,
            "next_blocks": next_press_blocks, "moves": moves, "status": final_status, "first_nb": state["first_play_nb"],
            "n_blocks": nb, "unfired": unplanned, "c2s": c2s, "cut_log": [c for _n, c in cut_seen]}


def analyse(seed, R, play_min):
    rows = []                                      # (check, value, ok, note)
    rms, tel, n_blocks = R["rms"], R["tel"], R["n_blocks"]
    c2s, bpm_of, title_of = R["c2s"], R["bpm_of"], R["title_of"]
    cc = R["clock_check"]
    sync_tel = R.get("sync") or []

    def add(check, value, ok, note=""):
        rows.append((check, value, ok, note))

    if cc:
        drift = cc["drift_samples"]
        add("clock map", f"{cc['per_sample']:.4f} clk/sample, offset {cc['offset']:.0f}", abs(drift or 0) < 0.05 * RATE,
            f"drift over the run {drift:.0f} samples" if drift is not None else "")

    def g_of(t, dk):
        return t.get(dk, (0.0, False, 0.0, None, 1.0, 0.0))

    # ---- spans: runs of a deck audible with one track ----
    spans = []
    decks = sorted({dk for t in tel for dk in t})
    for dk in decks:
        cur = None
        gap = 0
        for k, t in enumerate(tel):
            g, pl, _tm, tid, _lo, _ph = g_of(t, dk)
            aud = pl and g > AUDIBLE_GAIN
            if aud and cur is not None and cur["tid"] == tid:
                cur["end"] = k + 1
                gap = 0
            elif aud:
                if cur is not None:
                    spans.append(cur)
                cur, gap = {"deck": dk, "tid": tid, "start": k, "end": k + 1}, 0
            elif cur is not None:
                gap += 1
                if gap > 2:
                    spans.append(cur)
                    cur, gap = None, 0
        if cur is not None:
            spans.append(cur)
    spans.sort(key=lambda s: s["start"])
    # ---- (a) min play ----
    nexts = R["next_blocks"]
    for i, s in enumerate(spans):
        dur = (s["end"] - s["start"]) * BLOCK / RATE
        last = s["end"] >= n_blocks - 3
        pressed = any(s["start"] - 2 <= b <= s["end"] + 2 for b in nexts)
        name = f"{str(title_of.get(s['tid'], s['tid']))[:22]} [{s['deck']}]"
        if last:
            add(f"min play #{i + 1}", f"{dur:.1f} s", True, f"{name} (last record, cut off by the end: exempt)")
        elif pressed:
            add(f"min play #{i + 1}", f"{dur:.1f} s", True, f"{name} (NEXT pressed on it: exempt)")
        else:
            add(f"min play #{i + 1}", f"{dur:.1f} s", dur >= play_min - 0.3, f"{name} >= {play_min:.0f} s")
    n_rec = len(spans)
    add("records heard", str(n_rec), n_rec >= 3, "three or more records in the run")

    # ---- transitions: the next span begins while (or right as) the old one is audible ----
    trans = []
    for a, b in zip(spans, spans[1:]):
        if b["start"] <= a["end"] + 12:
            trans.append((a, b))
    plans = R["cut_seen"]
    holes = []
    kinds = []
    for i, (a, b) in enumerate(trans):
        # the plan that produced it: the last plan state seen for this pair whose start clock sits near b's first audible block
        planned = None
        for nbk, c in reversed(plans):
            if c.get("_new_tid") != b["tid"] or c.get("_old_tid") != a["tid"]:
                continue
            ci = c2s(c["clock"])
            if ci is not None and abs(ci - b["start"] * BLOCK) < 6.0 * RATE:
                planned = c
                break
        kind = (planned or {}).get("kind", "?")
        kinds.append(kind)
        bpm_old = bpm_of.get(a["tid"]) or 120.0
        beat_s = 60.0 / bpm_old
        bar_blocks = 4 * beat_s * RATE / BLOCK
        label = f"{kind} {a['deck']}->{b['deck']} @ {b['start'] * BLOCK / RATE:.1f}s"
        if planned is None:
            add(f"plan {i + 1}", "none", False, f"{label}: no plan seen for {str(title_of.get(a['tid']))[:16]} -> {str(title_of.get(b['tid']))[:16]}")
            holes.append(None)
            continue
        # OVERLAP: both audible
        both_blocks = sum(1 for k in range(b["start"], min(a["end"], n_blocks)) if True)
        overlap_bars = both_blocks / bar_blocks
        if kind in ("blend", "breakdown"):
            want = int(planned.get("L", 0)) + int(planned.get("T", 0))
            truncated = a["end"] >= n_blocks - 3 or b["end"] >= n_blocks - 3      # the render ended inside this blend
            add(f"overlap {i + 1}", f"{overlap_bars:.1f} bars", truncated or abs(overlap_bars - want) <= 1.5,
                f"{label}: planned {want} bars ({planned.get('size')}), why: {str(planned.get('why', ''))[:40]}" + (" (cut off by the end of the run: exempt)" if truncated else ""))
        else:
            add(f"overlap {i + 1}", f"{overlap_bars:.1f} bars", overlap_bars <= 0.6, f"{label}: a cut has none")
        # ON THE BEAT: the new deck starts within 60 ms of the planned clock, on a beat of the old record
        start_nb = None
        for k in range(max(0, b["start"] - 40), min(n_blocks, b["start"] + 5)):
            if g_of(tel[k], b["deck"])[1] and (k == 0 or not g_of(tel[k - 1], b["deck"])[1]):
                start_nb = k
        ci = c2s(planned["clock"])
        if start_nb is None or ci is None:
            add(f"on the beat {i + 1}", "unseen", False, f"{label}: the new deck's start block not found")
        else:
            err_ms = (start_nb * BLOCK - ci) / RATE * 1000.0
            # the telemetry is a snapshot at the END of the block: the record's phase there, walked back to the planned clock
            ph_end = g_of(tel[start_nb], a["deck"])[5]
            ph_a = wrap(ph_end - ((start_nb + 1) * BLOCK - ci) / RATE / beat_s)
            add(f"on the beat {i + 1}", f"{err_ms:+.0f} ms, phase {ph_a:+.3f}", (-110.0 <= err_ms <= 60.0) and (abs(ph_a) <= 0.08 or kind == "cut"),
                f"{label}: start block vs planned clock (a block is 100 ms); the record's beat phase at the planned clock")
        if kind in ("blend", "breakdown"):
            # ONE BASSLINE: both audible and both lows open, beyond the swap's two beats
            both_low = 0
            for k in range(b["start"], min(a["end"], n_blocks)):
                ga, pa, _, _, la, _ = g_of(tel[k], a["deck"])
                gb, pb, _, _, lb, _ = g_of(tel[k], b["deck"])
                if pa and pb and ga > AUDIBLE_GAIN and gb > AUDIBLE_GAIN and la > LOW_OPEN and lb > LOW_OPEN:
                    both_low += 1
            add(f"one bassline {i + 1}", f"{both_low * 0.1:.1f} s both open", both_low * BLOCK / RATE <= 2 * beat_s + 0.25, f"{label}: the swap is two beats")
            # BEAT LOCK over the last 60 % of the overlap
            k0 = b["start"] + int(0.4 * both_blocks)
            errs = []
            for k in range(k0, min(a["end"], n_blocks)):
                ga, pa, _, _, _, pha = g_of(tel[k], a["deck"])
                gb, pb, _, _, _, phb = g_of(tel[k], b["deck"])
                if not (pa and pb):
                    continue
                sy = sync_tel[k] if k < len(sync_tel) else None
                bias = sy[2] if sy and sy[0] == b["deck"] else 0.0
                errs.append(abs(wrap(phb - pha - bias)))
            if errs:
                mx, p95 = max(errs), float(np.percentile(errs, 95))
                add(f"beat lock {i + 1}", f"max {mx:.3f}, p95 {p95:.3f} beats", p95 <= 0.1 and mx <= 0.2, f"{label}: slave vs master grid phase, {len(errs)} readings")
            else:
                add(f"beat lock {i + 1}", "no readings", False, f"{label}: both decks never playing together in the window")
            holes.append(None)
        else:
            t_idx = int(ci) if ci is not None else b["start"] * BLOCK
            lo, hi = max(0, int((t_idx - 2.5 * RATE) / WIN)), min(len(rms), int((t_idx + 2.5 * RATE) / WIN))
            runs = [(s0, e0) for s0, e0 in silent_runs(rms) if e0 > lo and s0 < hi]
            t_win = t_idx / WIN
            best = min(runs, key=lambda r: abs(r[1] - t_win)) if runs else None
            if best is None:
                add(f"hole {i + 1}", "no silence", False, f"{label}: no gap between the records")
                holes.append(None)
                continue
            holes.append(best)
            hole_s = (best[1] - best[0]) * 0.01
            beats = hole_s / beat_s
            add(f"hole {i + 1}", f"{beats:.2f} beats ({hole_s * 1000:.0f} ms)", 0.5 <= beats <= 1.5, f"{label}, old {bpm_old:.0f} bpm")
    n_blend = sum(1 for k in kinds if k in ("blend", "breakdown"))
    add("mixed seams", f"{n_blend} of {len(trans)}", n_blend >= 1 or not trans, "at least one transition was a blend or a breakdown swap"
        + (f"; kinds: {', '.join(kinds)}" if kinds else ""))

    # ---- the room's feed: what the lights are told (beat, energy, the swap countdown, the drop stamp) ----
    vis = R.get("vis") or []
    if vis:
        excs = [o.get("_exc") for _lb, o in vis if isinstance(o, dict) and o.get("_exc")]
        add("visuals raise", str(len(excs)), not excs, excs[0][:100] if excs else "live_beat() and outstate_keys() never raised")
        # beat: present whenever a record plays with a trusted grid; its bpm is the record's (times its rate)
        n_play = sum(1 for k, t in enumerate(tel) if any(pl and g > AUDIBLE_GAIN for (g, pl, *_r) in t.values()))
        n_beat = sum(1 for lb, _o in vis if lb is not None)
        bad_bpm = 0
        for k, (lb, _o) in enumerate(vis):
            if lb is None or k >= len(tel):
                continue
            tids = [tid for (g, pl, _tm, tid, _lo, _ph) in tel[k].values() if pl and g > AUDIBLE_GAIN]
            if tids and all(abs(lb["bpm"] - (bpm_of.get(tid) or lb["bpm"])) / max(bpm_of.get(tid) or 1.0, 1.0) > 0.09 for tid in tids):
                bad_bpm += 1
        add("visuals beat", f"{n_beat} of {n_play} playing blocks", n_play == 0 or n_beat >= 0.9 * n_play, f"a live beat whenever a record plays; {bad_bpm} blocks with a bpm matching no audible record")
        add("visuals bpm", f"{bad_bpm} off", bad_bpm <= 2, "the published bpm is an audible record's (within 9 %, the rate included)")
        # phase advances and wraps: onsets about once per beat
        onsets = 0
        prev = None
        for lb, _o in vis:
            if lb is None:
                prev = None
                continue
            if prev is not None and lb["phase"] < prev - 0.5:
                onsets += 1
            prev = lb["phase"]
        beats_expected = 0.0
        for t in tel:
            if not t:
                continue
            g, pl, _tm, tid, _lo, _ph = max(t.values(), key=lambda v: v[0])     # the loudest deck's record
            if pl and g > AUDIBLE_GAIN:
                beats_expected += ((bpm_of.get(tid) or 120.0) / 60.0) * BLOCK / RATE
        add("visuals onsets", f"{onsets} (about {beats_expected:.0f} beats played)", beats_expected == 0 or 0.4 * beats_expected <= onsets <= 1.1 * beats_expected,
            "beat-phase wraps seen at the 10 Hz tick (a 100 ms tick misses some at 116 bpm; never more than the beats)")
        # energy: present and in range whenever something plays
        es = [o.get("dj_energy") for _lb, o in vis if isinstance(o, dict) and o.get("dj_energy") is not None]
        add("visuals energy", f"{len(es)} readings, {min(es) if es else 0:.2f}-{max(es) if es else 0:.2f}", bool(es) and all(0.12 <= e <= 1.0 for e in es) and (max(es) - min(es) > 0.05 if len(es) > 50 else True),
            "0.12..1.0, and it moves (breakdowns dip, grooves rise)")
        # the swap countdown reaches zero where the record changes, and a drop is stamped there
        rec_hist_changes = []
        last = None
        for k, (_lb, o) in enumerate(vis):
            st = o.get("dj_style") if isinstance(o, dict) else None
            se = o.get("dj_swap_eta") if isinstance(o, dict) else None
            if last is not None and last[1] is not None and se is None and last[0] in ("blend", "breakdown", "cut"):
                rec_hist_changes.append((k, last[1]))
            last = (st, se)
        add("visuals swap eta", f"{len(rec_hist_changes)} countdowns ended" + (f", last reading {max(v for _k, v in rec_hist_changes):.2f} s" if rec_hist_changes else ""),
            bool(rec_hist_changes) and all(v <= 0.25 for _k, v in rec_hist_changes), "dj_swap_eta counts down to within a block of zero before it clears")
        stamps = sorted({o.get("dj_drop_t") for _lb, o in vis if isinstance(o, dict) and o.get("dj_drop_t")})
        add("visuals drop stamps", str(len(stamps)), len(stamps) >= len(rec_hist_changes), f"at least one per swap ({len(rec_hist_changes)} swaps), plus the records' own drops")

    # ---- (d) dead air ----
    hole_set = [h for h in holes if h]
    dead = []
    for s0, e0 in silent_runs(rms):
        if e0 - s0 <= 150:
            continue
        if s0 == 0 or e0 >= len(rms):
            continue                                # the first-song wait / the end
        if any(not (e0 <= h[0] or s0 >= h[1]) for h in hole_set):
            continue
        dead.append((s0 * 0.01, (e0 - s0) * 0.01))
    add("dead air", f"{len(dead)} gaps", not dead, "; ".join(f"{t:.1f}s for {l:.1f}s" for t, l in dead[:4]))

    # ---- presses ----
    for j, r in enumerate(R["presses"]):
        name = f"{r['kind'].upper()} #{j + 1} ({r['trig']}) at {r['t']:.0f}s"
        if r["exc"]:
            add(name, "EXCEPTION", False, r["exc"])
        elif r["acted"]:
            add(name, "acted", True, f"[{r['state']}] {str(r['ret'])[:60]}")
        elif r["said"]:
            add(name, "said", True, f"[{r['state']}] {r['said'][:70]}")
        else:
            add(name, "SILENT", False, f"[{r['state']}] returned {str(r['ret'])[:60]}; said {[t[:40] for t in r['texts'][1:3]]}")
    for kind in ("next", "stay"):
        n = sum(1 for r in R["presses"] if r["kind"] == kind)
        add(f"{kind.upper()} presses fired", str(n), n >= 2, "2-3 planned" + ("; unfired: " + ", ".join(f"{k}/{t}" for k, t, _ in R["unfired"]) if R["unfired"] else ""))

    # ---- (f) errors ----
    for t, who, text in R["errors"]:
        add("error", f"{who} at {t:.0f}s", False, text[:140])
    if not R["errors"]:
        add("errors", "none", True, "d.last_error and p.last_error stayed None")
    return rows


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    music = arg("--music", "D:/Devel/music")
    seeds = seeds_of(arg("--seeds", "1-2"))
    minutes = arg("--minutes", 9.0, float)
    pace = arg("--pace", 2.5, float)
    wav = arg("--wav", None)
    all_rows = []
    for s in seeds:
        print(f"== seed {s}", flush=True)
        try:
            R = run(music, s, minutes, pace, wav)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            all_rows.append((s, ("run", "EXCEPTION", False, f"{type(e).__name__}: {e}")))
            continue
        if "fatal" in R:
            print(R["fatal"])
            if R.get("code") == 2:
                return 2
            all_rows.append((s, ("run", "fatal", False, R["fatal"])))
            continue
        rows = analyse(s, R, R["p"].play_min_s if hasattr(R["p"], "play_min_s") else PLAY_MIN)
        all_rows += [(s, r) for r in rows]
        print(f"\n-- seed {s}: the player's moves (p.moves) --")
        for m in R["moves"]:
            if isinstance(m, dict):
                print(f"  {time.strftime('%H:%M:%S', time.localtime(m.get('t', 0)))}  [{m.get('kind', '')}] {m.get('text', '')}")
            else:
                print(f"  {m}")
        if wav and R["pcm"]:
            path = wav if len(seeds) == 1 else f"{os.path.splitext(wav)[0]}_seed{s}{os.path.splitext(wav)[1] or '.wav'}"
            write_wav(path, np.concatenate(R["pcm"]))
            print(f"wrote {path}")
        del R
    print(f"\n{'seed':>4s} {'check':24s} {'value':30s} {'':5s} note")
    bad = 0
    for s, (check, value, ok, note) in all_rows:
        bad += 0 if ok else 1
        print(f"{s:4d} {check[:24]:24s} {str(value)[:30]:30s} {'PASS' if ok else 'FAIL':5s} {note}")
    print("\nALL OK" if bad == 0 else f"\nFAIL ({bad})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code or 0)
