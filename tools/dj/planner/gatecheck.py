"""Gate Check: put ONE screen on trial and rate it by ear.

Eleven conjunctive screens decide whether a blend is allowed. Measured
2026-08-07: long_blend carries the highest weight in the groove theme
(1.7) and reaches the dice on 19% of seams - not out-rolled, gated out.
Three screens do most of it (band_clash_high, no_beat_power_A,
kick_offset>20ms, ~30% of seams between them) and none of them had any
route to being shown wrong, because a threshold nobody may cross can
never be disproved.

So: pick a screen, get a seam it refused, hear the blend it refused, and
say whether it was right. One question, one pair of buttons.

Everything on screen is about THIS seam:
  * WHY it fired - the measured numbers beside the bar they missed,
    read through lib/dj/gateprobe.py, which imports its bars from the
    gates themselves so the panel cannot drift from the engine
  * the song picture - where on each track this is happening
  * the seam scope - what the blend actually does to the sound, drawn
    from the exact events the renderer ran
  * the groove chips - the pair's rhythmic limiting factor in words

Verdicts append to logs/gate_ratings.jsonl. They are deliberately NOT
written to seam_feedback: that table teaches STYLE taste, and "this gate
was wrong" is a statement about a threshold, not about long_blend.
"""
import json
import os
import random
import time

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton,
                             QVBoxLayout, QWidget)

from lib.dj import gateprobe
from lib.dj.rhythm import seam_chips
from tools.dj.planner.exitcompare import ExitLanes

RATE = 44100
_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "logs",
    "gate_ratings.jsonl")

# Styles worth putting on trial: the overlapped-drum tier the screens
# govern. force_style needs one of these for the seam to exist at all.
TRIAL_STYLES = ("long_blend", "bass_swap", "filter_sweep")


class _FindWorker(QThread):
    """Find a seam the chosen gate refused, plan it THROUGH that gate, and
    render it. All off the UI thread."""
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, db, brain, library, gate, rng, seen=()):
        super().__init__()
        self.db, self.brain, self.library = db, brain, library
        self.gate, self.rng = gate, rng
        # Pairs already rated this session. Without this the finder handed
        # the same pair back repeatedly (measured: one pair rated 3x in a
        # 23-verdict run), which silently inflates n and lets one opinion
        # count as three against a threshold.
        self.seen = set(seen)

    def run(self):
        try:
            from lib.dj.audition import render_seam
            pool = [t for t in self.library
                    if 150 <= t.duration_s <= 420 and t.mix_outs and t.sections]
            for attempt in range(400):
                if attempt % 25 == 0:
                    self.status.emit(f"looking for a seam {self.gate} "
                                     f"refused... ({attempt})")
                a = self.rng.choice(pool)
                b, meta = self.brain.choose_next(a, 0.5, a.bpm)
                if b is None or not (150 <= b.duration_s <= 420):
                    continue
                if (a.id, b.id) in self.seen:
                    continue
                want = self.rng.choice(TRIAL_STYLES)
                try:
                    plan = self.brain.plan_transition(
                        a, b, dict(meta),
                        after_s=a.duration_s * self.rng.uniform(0.40, 0.62),
                        force_style=want, test_gates=True)
                except Exception:
                    continue
                if plan["style"] != want:
                    continue
                # The pin must have been let through BECAUSE of this gate -
                # that is what makes the seam evidence about it.
                tested = (plan.get("diag") or {}).get("gate_test") or ""
                if self.gate not in tested:
                    continue
                # ATTRIBUTABLE EVIDENCE FIRST. The override needs EVERY
                # kill reason to be testable, so a seam can arrive with two
                # or three screens overridden at once - and then "the gate
                # was wrong" does not say WHICH. Spend most of the search
                # insisting the chosen gate refused this seam alone; only
                # settle for a combination if that turns up nothing, and
                # say so on screen when it does.
                solo = (len([x for x in tested.split(",") if x.strip()]) == 1)
                if not solo and attempt < 300:
                    continue
                rows = gateprobe.probe(a, b, plan)
                self.status.emit(f"rendering {plan['style']}: "
                                 f"{a.title[:24]} -> {b.title[:24]}...")
                info = {}
                audio = render_seam(self.db, a, b, plan, info=info)
                self.ready.emit({
                    "a": a, "b": b, "plan": plan, "rows": rows,
                    "audio": audio, "info": info,
                    "gate": self.gate, "also_tested": tested, "solo": solo,
                    "chips": seam_chips(plan, {"rhythm": plan.get("rhythm")}),
                })
                return
            self.failed.emit(
                f"no seam found that {self.gate} refused in 400 tries - "
                f"either it rarely fires on this library, or every seam it "
                f"refused was also blocked by a structural gate")
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class GateCheckTab(QWidget):
    """One gate, one seam, one question."""

    def __init__(self, planner):
        super().__init__()
        self.planner = planner
        from tools.dj.planner.player import TrackPlayer
        self.player = TrackPlayer()
        self.rng = random.Random()
        self.brain = None
        self.seam = None
        self._worker = None
        self._seen = self._rated_pairs()   # (a_id, b_id) already judged

        v = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Put this gate on trial:"))
        self.gate_box = QComboBox()
        for g in gateprobe.gate_names():
            self.gate_box.addItem(g)
        self.gate_box.setToolTip(
            "The screen under test. Listed most-costly first: the top "
            "three block ~30% of all seams between them.")
        row.addWidget(self.gate_box)
        self.find_btn = QPushButton("Find a seam it refused")
        self.find_btn.clicked.connect(self._find)
        row.addWidget(self.find_btn)
        stop = QPushButton("■ Stop")
        stop.clicked.connect(lambda: self.planner.stop_all_playback())
        row.addWidget(stop)
        row.addStretch(1)
        v.addLayout(row)

        self.card = QLabel(
            "Pick a gate and press Find. You get a seam that gate refused, "
            "rendered as it would have played, plus the numbers it was "
            "judged on. Listen, then say whether the gate was right.")
        self.card.setWordWrap(True)
        self.card.setStyleSheet("font-size: 14px; padding: 8px;")
        v.addWidget(self.card)

        self.why = QLabel("")
        self.why.setWordWrap(True)
        self.why.setTextFormat(Qt.TextFormat.RichText)
        self.why.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 11px; "
            "padding: 6px 10px; background: #16161a;")
        v.addWidget(self.why)

        self.lanes = ExitLanes()
        self.lanes.setMinimumHeight(190)
        v.addWidget(self.lanes, 2)

        from tools.dj.planner.seamscope import SeamScope
        self.strip = SeamScope()
        self.strip.setMinimumHeight(150)
        self.strip.setMaximumHeight(210)
        self.strip.seekRequested.connect(self._seek)
        v.addWidget(self.strip, 2)

        self.chips = QLabel("")
        self.chips.setStyleSheet("color:#c8ccd4; padding: 2px 10px;")
        v.addWidget(self.chips)

        row2 = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play the blend it refused")
        self.play_btn.clicked.connect(self._play)
        self.play_btn.setEnabled(False)
        row2.addWidget(self.play_btn)
        self.right_btn = QPushButton("✓ Gate was RIGHT — this sounds bad")
        self.right_btn.setStyleSheet("color:#5adc78;")
        self.right_btn.clicked.connect(lambda: self._rate(True))
        self.wrong_btn = QPushButton("✗ Gate was WRONG — this sounds fine")
        self.wrong_btn.setStyleSheet("color:#f0645a;")
        self.wrong_btn.clicked.connect(lambda: self._rate(False))
        for b in (self.right_btn, self.wrong_btn):
            b.setEnabled(False)
            row2.addWidget(b)
        row2.addStretch(1)
        v.addLayout(row2)

        self.tally = QLabel("")
        self.tally.setStyleSheet("color:#9aa0a8; padding: 0 10px;")
        v.addWidget(self.tally)
        self._refresh_tally()

        self._tick = QTimer(self)
        self._tick.timeout.connect(self._follow)
        self._tick.start(60)

    # -- finding -----------------------------------------------------------
    def _ensure_brain(self):
        if self.brain is None:
            from lib.dj.brain import Brain
            from lib.dj.themes import get_theme
            self.brain = Brain(self.planner.library, get_theme("groove"),
                               seed=self.rng.randrange(1 << 30))
            try:
                self.brain.load_pair_memory(self.planner.db)
            except Exception:
                pass
        return self.brain

    def _find(self):
        if not self.planner.library:
            self.card.setText("No library loaded.")
            return
        if self._worker is not None and self._worker.isRunning():
            return
        self.find_btn.setEnabled(False)
        for b in (self.play_btn, self.right_btn, self.wrong_btn):
            b.setEnabled(False)
        self.card.setText("searching...")
        self._worker = _FindWorker(self.planner.db, self._ensure_brain(),
                                   self.planner.library,
                                   self.gate_box.currentText(), self.rng,
                                   seen=self._seen)
        self._worker.ready.connect(self._ready)
        self._worker.failed.connect(self._failed)
        self._worker.status.connect(self.card.setText)
        self._worker.start()

    def _ready(self, seam):
        self.seam = seam
        a, b, plan = seam["a"], seam["b"], seam["plan"]
        self.find_btn.setEnabled(True)
        for btn in (self.play_btn, self.right_btn, self.wrong_btn):
            btn.setEnabled(True)
        self.card.setText(
            f"{a.title}  →  {b.title}      [{plan['style']}, "
            f"{plan['beats']} beats, rate {plan['rate']:.4f}]")
        self.why.setText(self._why_html(seam))
        blend_s = plan["beats"] * a.period_s or 14.0
        self.lanes.set_seam({"a": a, "b": b,
                             "cur": {"out_s": plan["out_s"], "after_s": 0.0,
                                     "fallback": False},
                             "new": {"out_s": plan["out_s"], "after_s": 0.0,
                                     "fallback": False},
                             "in_s": plan["in_s"], "beats": plan["beats"],
                             "blend_s": blend_s}, simple=True)
        info = seam.get("info") or {}
        render_s = len(seam["audio"]) / RATE if seam.get("audio") is not None else 0.0
        self.strip.set_seam(a, b, info, plan.get("out_s", 0.0), render_s)
        cw = seam.get("chips") or []
        self.chips.setText(
            "   ".join(f"<b>[ {c} ]</b>" for c in cw)
            or "<i>rhythmically clean — no groove chips on this pair</i>")

    def _why_html(self, seam):
        g = seam["gate"]
        out = [f"<b style='color:#f0a03c'>{g}</b> refused "
               f"<b>{seam['plan']['style']}</b> on this seam.<br>"]
        if not seam.get("solo", True):
            out.append(
                "<span style='color:#f0a03c'>&nbsp;&nbsp;heads up: "
                f"{seam.get('also_tested')} ALL refused this seam, so a "
                "verdict here judges the combination, not "
                f"{g} alone.</span><br>")
        for r in seam["rows"]:
            if r["gate"] == g:
                out.append(f"&nbsp;&nbsp;measured &nbsp;<b>{r['detail']}</b>"
                           f"<br>&nbsp;&nbsp;bar &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                           f"<b>{r['bar']}</b><br>"
                           f"&nbsp;&nbsp;it also blocks: "
                           f"{', '.join(r['kills'][:6])}<br><br>")
        out.append("<span style='color:#8a9098'>every other screen on this "
                   "seam:</span><br>")
        for r in seam["rows"]:
            if r["gate"] == g:
                continue
            if r["skipped"]:
                mark, col = "skip ", "#6a7078"
            elif r["fired"]:
                mark, col = "FIRED", "#f0645a"
            else:
                mark, col = "ok   ", "#5adc78"
            note = r["skipped"] or r["detail"]
            out.append(f"<span style='color:{col}'>&nbsp;&nbsp;{mark}</span> "
                       f"<span style='color:#9aa0a8'>{r['gate']:<18} "
                       f"{note}</span><br>")
        return "".join(out)

    def _failed(self, msg):
        self.find_btn.setEnabled(True)
        self.card.setText(msg)

    # -- playback ----------------------------------------------------------
    def _play(self):
        if self.seam is None:
            return
        self.planner.claim_playback("gatecheck")
        self.player.load(self.seam["audio"])
        self.player.play()

    def _seek(self, t):
        if self.seam is None:
            return
        self.planner.claim_playback("gatecheck")
        self.player.load(self.seam["audio"])
        self.player.seek(t)
        self.player.play()

    def _follow(self):
        if self.seam is None:
            return
        if self.player.playing:
            t = self.player.time_s()
            self.strip.set_playhead(t)
            end = self.strip.window_end()
            if end and t >= end:
                self.player.pause()
        else:
            self.strip.set_playhead(None)

    # -- verdict -----------------------------------------------------------
    def _rate(self, gate_was_right):
        if self.seam is None:
            return
        s, plan = self.seam, self.seam["plan"]
        row = next((r for r in s["rows"] if r["gate"] == s["gate"]), {})
        rec = {"t": time.time(), "gate": s["gate"],
               "gate_was_right": bool(gate_was_right),
               "style": plan["style"], "beats": plan["beats"],
               "rate": round(plan["rate"], 4),
               "a": s["a"].title, "b": s["b"].title,
               "a_id": s["a"].id, "b_id": s["b"].id,
               "measured": row.get("detail"), "bar": row.get("bar"),
               "also_tested": s.get("also_tested"),
               "solo": bool(s.get("solo", True)),
               "chips": s.get("chips"),
               "listened_s": round(self.player.time_s(), 1)}
        try:
            os.makedirs(os.path.dirname(_LOG), exist_ok=True)
            with open(_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError as e:
            self.card.setText(f"could not write the verdict: {e}")
            return
        self._seen.add((s["a"].id, s["b"].id))
        self.player.pause()
        self._refresh_tally()
        self._find()               # straight on to the next one

    def _rated_pairs(self):
        """Pairs judged in EARLIER sessions too - a threshold should not be
        re-argued with the same two songs a week later either."""
        out = set()
        try:
            with open(_LOG, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        out.add((r.get("a_id"), r.get("b_id")))
                    except ValueError:
                        pass
        except OSError:
            pass
        return out

    def _refresh_tally(self):
        """What the verdicts say so far, per gate. A gate the ear keeps
        overruling is the one to re-tune."""
        rows = []
        try:
            with open(_LOG, encoding="utf-8") as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass
        except OSError:
            pass
        if not rows:
            self.tally.setText("no verdicts yet — "
                               "logs/gate_ratings.jsonl is empty")
            return
        # Only SOLO verdicts are attributable to one screen; combined
        # ones are logged but never tallied against a gate.
        agg = {}
        combined = 0
        for r in rows:
            if not r.get("solo", True):
                combined += 1
                continue
            g = r.get("gate")
            a = agg.setdefault(g, [0, 0])
            a[0 if r.get("gate_was_right") else 1] += 1
        bits = []
        for g, (right, wrong) in sorted(agg.items(),
                                        key=lambda kv: -sum(kv[1])):
            n = right + wrong
            flag = "  ← the ear disagrees" if wrong > right and n >= 5 else ""
            bits.append(f"{g}: right {right} / wrong {wrong}{flag}")
        extra = (f"  (+{combined} on multi-gate seams, not attributable)"
                 if combined else "")
        self.tally.setText(f"{len(rows)} verdicts — "
                           + (" · ".join(bits) or "none attributable yet")
                           + extra)

    def stop_playback(self):
        try:
            self.player.pause()
        except Exception:
            pass

    def close_audio(self):
        try:
            self.player.close()
        except Exception:
            pass
