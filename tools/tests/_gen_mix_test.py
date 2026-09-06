"""Gate for listening session 3 (docs/GENERATIVE_MUSIC_PLAN.md): the mix
moves, the space is real, the parts are layered, the melody has
function, the kit has a language, styles morph, the box holds a
loudness and never clips, and the operator's taste steers the form.

  1. AUTOMATION: the form writes lane events; the build's high-pass
     climbs phrase by phrase and snaps open on the drop; a lane ramps
     smoothly in the rack.
  2. SPACE: the FDN reverb tail is smooth (no single dominant comb) and
     decays; chorus widens a mono send.
  3. LAYERS: a slot with layers renders differently from the same slot
     without; the crossover works.
  4. POLYPHONY: the per-slot cap steals the oldest note.
  5. KARPLUS: the physical pluck is pitched and rings.
  6. MELODY FUNCTION: lead notes on strong beats are chord tones; pad
     voicings move by small intervals.
  7. RHYTHM LANGUAGE: dnb uses breaks, hip-hop halftime, builds double
     the hats, fills are shared across kick/snare/tom.
  8. MORPH: a style swap with morph glides the slot gain.
  9. LOUDNESS + LIMITER: the rack converges toward the style target and
     the lookahead limiter never exceeds its ceiling.
 10. TASTE: liked sections get heavier form weights; a like boosts the
     playing motif.
 11. ONE-SHOTS: the manifest resolves and a sample layer plays.
 12. STYLES: every style composes and renders clean.

Usage: python tools/tests/_gen_mix_test.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.gen import RATE                                  # noqa: E402
from lib.gen.composer import Composer                      # noqa: E402
from lib.gen.composer.styles import get_style, STYLES      # noqa: E402
from lib.gen.events import NoteEvent                       # noqa: E402
from lib.gen.synth import SynthRack, fx                    # noqa: E402
from lib.gen.synth.voices import VOICES                    # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def render_rack(rack, seconds, blk=2048):
    out = []
    while rack.clock < seconds * RATE:
        out.append(rack.render(blk))
    return np.concatenate(out)


def hold(c, section):
    c.form.section = section
    c.form.bars_left = 10 ** 6
    c.form.hold = True
    return c


def main():
    print("== automation")
    c = Composer("groove", bpm=124, key="8A", seed=21)
    ps = list(c.phrases_until(int(10 * 60 * RATE)))
    autos = [(p.section, e) for p in ps for e in p.events if e.slot == "auto"]
    check(len(autos) > 20, f"{len(autos)} automation events over 10 min")
    hp_build = [e.params["to"] for s, e in autos if s == "build" and e.params["lane"] == "hp"]
    runs, cur = [], []
    for v in hp_build:                      # one run per build (a new build starts low again)
        if cur and v < cur[-1]:
            runs.append(cur); cur = []
        cur.append(v)
    runs.append(cur)
    check(len(hp_build) >= 2 and all(all(b > a for a, b in zip(r_, r_[1:])) for r_ in runs) and max(hp_build) > 300,
          f"build high-pass climbs within each build: {[[round(v) for v in r_] for r_ in runs[:3]]}")
    drop_hp = [e.params["to"] for s, e in autos if s == "drop" and e.params["lane"] == "hp"]
    check(drop_hp and all(v <= 20.0 for v in drop_hp), "drop snaps the high-pass open")
    r = SynthRack(get_style("groove"), 124.0)
    r.set_lane("lp", 1000.0, int(1.0 * RATE))
    vals = []
    for _ in range(int(RATE / 2048) + 1):
        r.render(2048)
        vals.append(r.lanes["lp"][0])
    check(vals[0] > 15000 and vals[-1] <= 1000.5 and all(b <= a for a, b in zip(vals, vals[1:])), f"lane ramps smoothly {round(vals[0])} -> {round(vals[-1])}")

    print("== space")
    rv = fx.FDNReverb()
    imp = np.zeros((4096, 2), np.float32); imp[0] = 1.0
    tail = np.concatenate([rv.process(imp)] + [rv.process(np.zeros_like(imp)) for _ in range(25)])
    mono = tail.mean(axis=1)
    early = float(np.sqrt(np.mean(mono[4096:8192] ** 2)))
    late = float(np.sqrt(np.mean(mono[-8192:] ** 2)))
    spec = np.abs(np.fft.rfft(mono[4096:4096 + 16384]))
    peaky = float(spec.max() / (np.median(spec) + 1e-9))
    check(early > 1e-4 and late < early * 0.1, f"reverb tail decays ({20 * np.log10(late / early + 1e-12):.0f} dB over the tail)")
    check(peaky < 60, f"tail is diffuse, not a comb (peak/median spectrum {peaky:.0f})")
    ch = fx.Chorus()
    x = np.repeat(np.sin(2 * np.pi * 220 * np.arange(RATE) / RATE).astype(np.float32)[:, None], 2, axis=1)
    wet = np.concatenate([ch.process(x[i:i + 2048]) for i in range(0, RATE - 2048, 2048)])
    corr = float(np.corrcoef(wet[RATE // 4:, 0], wet[RATE // 4:, 1])[0, 1])
    check(corr < 0.98, f"chorus widens a mono source (L/R corr {corr:.3f})")

    print("== layers")
    st = get_style("groove")
    r1 = SynthRack(st, 124.0, seed=1)
    a = r1._synth(VOICES["bass"](), NoteEvent(0, "bass", 45.0, 0.9, RATE // 4, {}), st["slots"]["bass"])[0]
    plain = {k: v for k, v in st["slots"]["bass"].items() if k != "layers"}
    r2 = SynthRack(st, 124.0, seed=1)
    b = r2._synth(VOICES["bass"](), NoteEvent(0, "bass", 45.0, 0.9, RATE // 4, {}), plain)[0]
    n = min(a.shape[0], b.shape[0])
    hi_a = np.abs(np.fft.rfft(a[:n].astype(np.float64) if a.ndim == 1 else a[:n].mean(axis=1)))
    hi_b = np.abs(np.fft.rfft(b[:n].astype(np.float64) if b.ndim == 1 else b[:n].mean(axis=1)))
    f = np.fft.rfftfreq(n, 1 / RATE)
    top = (f > 1000) & (f < 5000)
    ratio = float(hi_a[top].sum() / (hi_b[top].sum() + 1e-9))
    check(ratio > 1.1, f"a layered bass carries more top than the sub alone ({ratio:.2f}x in 1-5 kHz)")

    print("== polyphony")
    st_np = get_style("groove"); st_np.pop("vst", None)        # analog pad: hosted slots render per phrase, not per note
    r = SynthRack(st_np, 124.0, seed=3)
    from lib.gen.synth.rack import POLY
    cap = POLY["pad"]
    evs = [NoteEvent(1000 + i * 100, "pad", 57.0 + (i % 5), 0.6, RATE, {}) for i in range(cap + 3)]
    r.schedule(evs)
    r.render(4096)
    live = [it for it in r._active if it[2] == "pad" and it[0] + it[1].shape[0] > 4096]
    check(len(live) <= cap and r.stats["stolen"] >= 3, f"pad polyphony capped at {cap} ({r.stats['stolen']} stolen)")

    print("== karplus")
    rng = np.random.default_rng(4)
    ks = VOICES["ks"]().render(69.0, 0.9, int(0.6 * RATE), {"decay": 0.997}, {}, rng)
    seg = ks[int(0.1 * RATE):int(0.4 * RATE)].astype(np.float64)
    ac = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
    lag = int(np.argmax(ac[40:400])) + 40
    peak_hz = RATE / lag
    check(abs(peak_hz - 440.0) < 8.0 and np.abs(ks[int(0.4 * RATE):int(0.5 * RATE)]).max() > 0.02,
          f"Karplus-Strong A4 rings at {peak_hz:.0f} Hz (autocorrelation)")

    print("== melody function")
    c = Composer("groove", bpm=124, key="8A", seed=8)
    hold(c, "drop")
    ps = [c.next_phrase() for _ in range(12)]
    strong = tot = 0
    for p in ps:
        spb = (p.end - p.start) / p.nbars
        sps = spb / 16
        for e in p.events:
            if e.slot != "lead":
                continue
            b = int((e.at - p.start) // spb)
            s = int(round(((e.at - p.start) - b * spb) / sps))
            if s % 4 == 0:
                tot += 1
                pcs = {m % 12 for m in c.harmony.notes(p.chords[min(b, 3)], 4, 4)}
                strong += int(round(e.pitch)) % 12 in pcs
    check(tot > 30 and strong >= 0.9 * tot, f"lead strong beats on chord tones: {strong}/{tot}")
    moves = []
    prev = None
    for p in ps:
        spb = (p.end - p.start) / p.nbars
        by_bar = {}
        for e in p.events:
            if e.slot == "pad":
                by_bar.setdefault(int((e.at - p.start) // spb), []).append(int(round(e.pitch)))
        for b in sorted(by_bar):
            v = sorted(by_bar[b])
            if prev is not None and len(v) == len(prev):
                moves.append(sum(abs(x - y) for x, y in zip(v, prev)) / len(v))
            prev = v
    check(moves and float(np.mean(moves)) < 4.0, f"pad voice leading: mean move {np.mean(moves) if moves else 0:.1f} semitones per voice")

    print("== rhythm language")
    def kit(style, section, seed=5, phrases=6):
        c = Composer(style, key="8A", seed=seed)
        hold(c, section)
        out = {}
        for _ in range(phrases):
            p = c.next_phrase()
            spb = (p.end - p.start) / p.nbars
            sps = spb / 16
            for e in p.events:
                b = int((e.at - p.start) // spb)
                s = int(round(((e.at - p.start) - b * spb) / sps))
                out.setdefault(e.slot, []).append(s % 16)
        return out
    d = kit("dnb", "drop")
    snare_steps = set(d.get("snare", []))
    check({4, 12} <= snare_steps and any(s in d.get("kick", []) for s in (10, 6, 2)), f"dnb plays breaks (snare steps {sorted(snare_steps)[:6]})")
    h = kit("hiphop", "groove")
    check(set(h.get("snare", [])) <= {8, 14, 15, 2, 4, 6, 10, 12, 9, 13} and h.get("snare", []).count(8) >= 0.6 * len(h.get("snare", [1])),
          "hip-hop halftime: snare on 3")
    hb = kit("groove", "build")
    hg = kit("groove", "groove")
    check(len(hb.get("hat", [])) > 1.1 * len(hg.get("hat", [])), f"builds fill the hats ({len(hb.get('hat', []))} vs {len(hg.get('hat', []))} in groove)")
    c = Composer("groove", bpm=124, key="8A", seed=9)
    c.form.section = "groove"; c.form.bars_left = 4
    p = c.next_phrase()
    spb = (p.end - p.start) / p.nbars
    last = [e for e in p.events if int((e.at - p.start) // spb) == 3 and e.slot in ("kick", "snare", "tom")]
    check(len(last) >= 6, f"fill bar carries a library fill ({len(last)} kit hits)")

    print("== morph")
    r = SynthRack(get_style("groove"), 124.0, seed=1)
    r.warm_up()
    r.set_style(get_style("techno"), 130.0, at=2048, morph=8 * 2048)
    r.schedule([NoteEvent(0, "pad", 57.0, 0.8, 20 * 2048, {})])
    gains = []
    for _ in range(12):
        r.render(2048)
        gains.append(float(r._slot_gain("pad", r.slots["pad"])))
    check(r._morph is None and abs(gains[1] - gains[-1]) > 1e-3 and all(abs(b - a) < 0.1 for a, b in zip(gains, gains[1:])),
          f"style morph glides the slot gain ({gains[1]:.3f} -> {gains[-1]:.3f})")

    print("== loudness + limiter")
    lim = fx.LookaheadLimiter(ceiling=0.9)
    burst = np.zeros((4096, 2), np.float32); burst[2000:2010] = 3.0
    out = np.concatenate([lim.process(burst), lim.process(np.zeros_like(burst))])
    check(float(np.abs(out).max()) <= 0.9 + 1e-4, f"lookahead limiter holds a 3.0 spike under 0.9 (peak {np.abs(out).max():.3f})")
    c = Composer("groove", bpm=124, key="8A", seed=12)
    hold(c, "drop")
    r = SynthRack(c.style, 124.0, seed=12)
    r.warm_up()
    for p in c.phrases_until(int(40 * RATE)):
        r.schedule(p.events)
    x = render_rack(r, 40)
    lufs = float(r.loud.lufs())
    check(abs(lufs + r.norm_db - (r.target_lufs)) < 3.0 or abs(lufs - r.target_lufs) < 2.5,
          f"loudness holds near the style target ({lufs:.1f} LUFS-ish, trim {r.norm_db:+.1f} dB, target {r.target_lufs})")
    check(np.isfinite(x).all() and float(np.abs(x).max()) <= 0.96, f"drop never exceeds the ceiling (peak {np.abs(x).max():.3f})")

    print("== taste")
    from lib.gen.feedback import PreferenceMemory
    import tempfile
    pm = PreferenceMemory(os.path.join(tempfile.gettempdir(), "gen_prefs_test.json"))
    pm.items = []
    for _ in range(3):
        pm.record({"style": "groove", "section": "break", "energy": 0.4}, True)
    pm.record({"style": "groove", "section": "drop", "energy": 1.0}, False)
    bias = pm.section_bias("groove")
    check(bias.get("break", 1) > 1.5 and bias.get("drop", 1) < 1.0, f"liked sections weigh more: {bias}")
    c = Composer("groove", bpm=124, key="8A", seed=2)
    c.form.taste = {"break": 2.0}
    hits = 0
    for _ in range(200):
        c.form.section = "groove"
        hits += c.form._next_section() == "break"
    c2 = Composer("groove", bpm=124, key="8A", seed=2)
    hits2 = 0
    for _ in range(200):
        c2.form.section = "groove"
        hits2 += c2.form._next_section() == "break"
    check(hits > hits2, f"taste steers the form (break chosen {hits} vs {hits2} of 200)")
    c3 = Composer("groove", bpm=124, key="8A", seed=3)
    hold(c3, "drop")
    c3.next_phrase()
    m = c3.melody.last_motif
    c3.melody.like()
    check(m is not None and m.liked == 1, "a like boosts the motif that played")

    print("== one-shots")
    from lib.gen.synth import oneshots
    path, base = oneshots.resolve("oneshots:kick_909")
    check(path is not None and os.path.exists(path) and base == 36, f"manifest resolves kick_909 -> base {base}")
    smp = VOICES["sample"]().render(36.0, 0.9, 100, {"file": "oneshots:kick_909"}, {}, np.random.default_rng(1))
    check(smp.shape[0] > 1000 and np.abs(smp).max() > 0.3, "sample layer plays the one-shot")

    print("== timeline")
    os.environ["GEN_HOOKS"] = "0"
    from lib.gen.system import GenSystem
    g = GenSystem(engine=None, style="groove", bpm=124, key="8A", seed=5, set_length_s=1800, log_dir="logs", threaded=False)
    g.start()
    for _ in range(40):
        g.rack.read(4096); g.step()
    tl = g.timeline()
    hz = tl["horizon"]
    ahead = [p for p in tl["phrases"] if p["end_s"] > tl["now_s"]]
    check(tl["now_s"] > 3 and len(tl["phrases"]) >= 2 and ahead and hz["composed_to_s"] >= tl["now_s"],
          f"timeline: now {tl['now_s']:.1f}s, {len(tl['phrases'])} phrases, composed to +{hz['composed_to_s'] - tl['now_s']:.0f}s")
    check(hz["section"] in ("intro", "groove") and hz["bars_left"] > 0 and hz["next"] and abs(sum(w for _, w in hz["next"]) - 1.0) < 0.05
          and len(hz["arc"]) == 21, f"horizon knows the section end, likely next {hz['next'][0]}, and the arc ahead")
    check("timeline" in g.status(), "timeline rides in status (so the remote console sees it)")
    g.stop()

    print("== styles")
    bad = []
    for name in STYLES:
        try:
            c = Composer(name, key="8A", seed=4)
            r = SynthRack(c.style, c.bpm, seed=4)
            for p in c.phrases_until(int(12 * RATE)):
                r.schedule(p.events)
            x = render_rack(r, 12)
            rms = 20 * np.log10(float(np.sqrt(np.mean(x ** 2))) + 1e-9)
            if not (np.isfinite(x).all() and np.abs(x).max() <= 0.96 and rms > -40):
                bad.append(f"{name}:{rms:.0f}dB")
        except Exception as e:  # noqa: BLE001
            bad.append(f"{name}:{type(e).__name__}:{e}")
    check(not bad, f"every style composes and renders clean ({len(STYLES)} styles){' bad: ' + ', '.join(bad) if bad else ''}")
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
