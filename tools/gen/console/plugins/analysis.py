"""Analysis tab: ingest a song, read it as the generator's own commands,
recreate it, score the recreation locally and globally, and send the
script to the show.

  Ingest     decode + analyse the file (lib/gen/analysis/ingest.py) ->
             the inferred SongScript (sections table, editable) and its
             command list (the whitelisted actions that regenerate it)
  Recreate   render the script offline (lib/gen/script.render) to
             logs/analysis/<name>/recreation.wav
  Score      per-phrase + global scores (lib/gen/analysis/score.py),
             drawn as two energy strips (original / recreation) and a
             score bar per phrase; the weakest phrases listed
  Save       script.yaml (edits in the table are applied first)
  Play       send the script to the running show (the "script" action)
  Open       load a previously saved script.yaml

All heavy work runs on a worker thread; the tab polls it at the
console's refresh rate. The table edits section / bars / energy /
density / brightness / swing / layers / chords in place."""
from __future__ import annotations

import json
import os
import threading
import traceback

import numpy as np
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QFont
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog,
                             QTableWidget, QTableWidgetItem, QPlainTextEdit, QProgressBar, QSplitter, QCheckBox)

COLS = ["section", "bars", "energy", "density", "brightness", "swing", "layers", "chords", "lanes"]
SECTION_COLOURS = {"intro": "#4a5a7a", "groove": "#3f7a5a", "build": "#a8772a", "drop": "#b03a3a", "break": "#5a4a8a",
                   "outro": "#4a4a4a", "flow": "#3f7a7a", "swell": "#a85a2a", "calm": "#4a6a8a"}


def _score_colour(v):
    v = max(0.0, min(100.0, float(v))) / 100.0
    r = int(220 * (1.0 - v) + 60 * v)
    g = int(80 * (1.0 - v) + 200 * v)
    return QColor(r, g, 70)


class ScoreStrip(QWidget):
    """Original vs recreation energy per bar, section blocks, and the local score per window."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.orig = []
        self.recon = []
        self.report = None
        self.script = None

    def set(self, orig=None, recon=None, report=None, script=None):
        if orig is not None:
            self.orig = orig
        if recon is not None:
            self.recon = recon
        self.report = report
        if script is not None:
            self.script = script
        self.update()

    def paintEvent(self, ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        qp.fillRect(0, 0, w, h, QColor("#16181d"))
        n = max(len(self.orig), len(self.recon), 1)
        bw = w / n
        font = QFont(); font.setPointSize(8); qp.setFont(font)
        # sections from the script
        if self.script:
            bar = 0
            for e in self.script.get("sections", []):
                x0, x1 = bar * bw, (bar + e["bars"]) * bw
                col = QColor(SECTION_COLOURS.get(e["section"], "#555")); col.setAlpha(120)
                qp.setPen(QPen(QColor("#0e1013"))); qp.setBrush(QBrush(col))
                qp.drawRect(QRectF(x0, 2, max(1.0, x1 - x0 - 1), 16))
                if x1 - x0 > 40:
                    qp.setPen(QPen(QColor("#f0f2f5"))); qp.drawText(int(x0) + 3, 14, f"{e['section']} {e['bars']}")
                bar += e["bars"]
        # energy curves (dB, normalised to the original's range)
        def curve(feats, y0, y1, colour):
            if not feats:
                return
            vals = np.array([f["energy_db"] for f in feats])
            lo, hi = float(np.percentile(vals, 5)) - 3, float(np.percentile(vals, 98)) + 1
            qp.setPen(QPen(QColor(colour), 2))
            last = None
            for i, v in enumerate(vals):
                y = y1 - (min(max(v, lo), hi) - lo) / max(hi - lo, 1e-6) * (y1 - y0)
                x = (i + 0.5) * bw
                if last is not None:
                    qp.drawLine(int(last[0]), int(last[1]), int(x), int(y))
                last = (x, y)
        curve(self.orig, 24, 70, "#7fd1a8")
        curve(self.recon, 24, 70, "#6cc3ff")
        qp.setPen(QPen(QColor("#7fd1a8"))); qp.drawText(4, 34, "original")
        qp.setPen(QPen(QColor("#6cc3ff"))); qp.drawText(4, 46, "recreation")
        # local scores
        if self.report:
            for r in self.report.get("local", []):
                x0, x1 = r["bar0"] * bw, (r["bar0"] + r["bars"]) * bw
                col = _score_colour(r["score"])
                qp.setPen(QPen(QColor("#0e1013"))); qp.setBrush(QBrush(col))
                hh = 40 * r["score"] / 100.0
                qp.drawRect(QRectF(x0, h - 12 - hh, max(1.0, x1 - x0 - 1), hh))
                if x1 - x0 > 26:
                    qp.setPen(QPen(QColor("#f0f2f5"))); qp.drawText(int(x0) + 2, h - 2, f"{r['score']:.0f}")
            qp.setPen(QPen(QColor("#c8ccd4"))); qp.drawText(4, h - 56, f"global {self.report['global']:.1f}")
        qp.end()


class AnalysisPage(QWidget):
    def __init__(self, console):
        super().__init__()
        self.console = console
        self.folder = None
        self.result = None          # ingest result
        self.script = None
        self.recon_feats = None
        self.report = None
        self.busy = False
        self.msg = ""
        self.progress = 0.0
        self._pending = None
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.path = QLineEdit(""); self.path.setPlaceholderText("song file (wav/mp3/flac...) or a logs/analysis/<name> folder")
        top.addWidget(self.path, 1)
        self.reuse = QCheckBox("reuse stems"); self.reuse.setToolTip("separate the song (demucs) and reuse its drums, vocal phrases and hook")
        top.addWidget(self.reuse)
        for text, fn in (("Browse", self.browse), ("Ingest", self.ingest), ("Recreate", self.recreate), ("Score", self.score),
                         ("Save", self.save), ("Play", self.play), ("Open", self.open_script)):
            b = QPushButton(text); b.clicked.connect(fn); top.addWidget(b)
        lay.addLayout(top)
        self.bar = QProgressBar(); self.bar.setRange(0, 100); self.bar.setTextVisible(False); self.bar.setMaximumHeight(6)
        lay.addWidget(self.bar)
        self.info = QLabel(""); lay.addWidget(self.info)
        self.strip = ScoreStrip(); lay.addWidget(self.strip)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.table = QTableWidget(0, len(COLS)); self.table.setHorizontalHeaderLabels(COLS)
        self.table.horizontalHeader().setStretchLastSection(True); self.table.verticalHeader().setVisible(False)
        split.addWidget(self.table)
        self.cmds = QPlainTextEdit(); self.cmds.setReadOnly(True); self.cmds.setPlaceholderText("the command list that regenerates the song")
        split.addWidget(self.cmds)
        split.setSizes([700, 400])
        lay.addWidget(split, 1)

    # -- helpers --------------------------------------------------------------
    def _run(self, label, fn):
        if self.busy:
            self.msg = "busy"
            return
        self.busy = True
        self.msg = label
        self.progress = 0.0

        def work():
            try:
                fn()
                self.msg = label + " done"
            except Exception as e:  # noqa: BLE001
                self.msg = f"{label} failed: {type(e).__name__}: {e}"
                traceback.print_exc()
            finally:
                self.busy = False
                self.progress = 1.0
        threading.Thread(target=work, name=f"analysis-{label}", daemon=True).start()

    def _folder(self):
        p = self.path.text().strip()
        if os.path.isdir(p):
            return p
        name = os.path.splitext(os.path.basename(p))[0] or "song"
        return os.path.join("logs", "analysis", name)

    def _script_from_table(self):
        if self.script is None:
            return None
        from lib.gen import script as S
        sc = dict(self.script)
        rows = []
        for i in range(self.table.rowCount()):
            def cell(j):
                it = self.table.item(i, j)
                return it.text().strip() if it else ""
            try:
                e = {"section": cell(0) or "groove", "bars": int(float(cell(1) or 8))}
                for j, k in ((2, "energy"), (3, "density"), (4, "brightness"), (5, "swing")):
                    if cell(j):
                        e[k] = float(cell(j))
                if cell(6):
                    e["layers"] = [x.strip() for x in cell(6).replace("+", ",").split(",") if x.strip()]
                if cell(7):
                    e["chords"] = [int(x) for x in cell(7).replace(",", " ").split()]
                if cell(8):
                    e["lanes"] = json.loads(cell(8))
                old = self.script["sections"][i] if i < len(self.script["sections"]) else {}
                if old.get("hook"):
                    e["hook"] = old["hook"]
                rows.append(e)
            except Exception as ex:  # noqa: BLE001
                self.msg = f"row {i + 1}: {ex}"
                return None
        sc["sections"] = rows
        return S.normalize(sc)

    def _fill_table(self):
        sc = self.script or {"sections": []}
        self.table.setRowCount(len(sc["sections"]))
        for i, e in enumerate(sc["sections"]):
            vals = [e.get("section", ""), str(e.get("bars", "")), *(f"{e[k]:.2f}" if e.get(k) is not None else "" for k in ("energy", "density", "brightness", "swing")),
                    "+".join(e.get("layers") or []), " ".join(str(x) for x in (e.get("chords") or [])),
                    json.dumps(e["lanes"]) if e.get("lanes") else ""]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j == 0:
                    it.setBackground(QColor(SECTION_COLOURS.get(v, "#555")))
                self.table.setItem(i, j, it)
        from lib.gen import script as S
        self.cmds.setPlainText("\n".join(f"bar {b:4d}  {a:10s} {json.dumps(v) if not isinstance(v, str) else v}" for b, a, v in S.to_actions(sc)) if sc.get("sections") else "")

    # -- actions --------------------------------------------------------------
    def browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Song", "", "Audio (*.wav *.flac *.mp3 *.ogg *.m4a *.aiff);;All files (*)")
        if p:
            self.path.setText(p)

    def ingest(self):
        p = self.path.text().strip()
        if not p or not os.path.exists(p):
            self.msg = "pick a file first"
            return
        folder = self._folder()

        def work():
            from lib.gen import script as S
            from lib.gen.analysis import ingest as I
            os.makedirs(folder, exist_ok=True)

            def prog(x, what):
                self.progress = x; self.msg = f"ingest: {what}"
            res = I.ingest(p, progress=prog, reuse=self.reuse.isChecked(), out_dir=folder)
            self.result = res
            for r in res["analysis"].get("reuse_reasons", []):
                print("[analysis]", r)
            self.script = res["script"]
            self.folder = folder
            S.save(self.script, os.path.join(folder, "script.yaml"))
            with open(os.path.join(folder, "features.json"), "w", encoding="utf-8") as fh:
                json.dump({"features": res["features"], "bars": res["bars"], "chords": res["chords"],
                           "analysis": {k: v for k, v in res["analysis"].items() if k != "sections"},
                           "sections": res["analysis"]["sections"], "source": os.path.abspath(p)}, fh)
            self.recon_feats = None
            self.report = None
            self._pending = "table"
        self._run("ingest", work)

    def recreate(self):
        sc = self._script_from_table()
        if sc is None:
            return
        folder = self.folder or self._folder()

        def work():
            from lib.gen import script as S
            from lib.gen.analysis import ingest as I
            self.script = sc
            self.folder = folder
            S.save(sc, os.path.join(folder, "script.yaml"))

            def prog(x):
                self.progress = x
            audio, _ = S.render(sc, out_path=os.path.join(folder, "recreation.wav"), progress=prog)
            self.recon_feats = I.features_on_grid(audio.mean(axis=1).astype(np.float32), sc["bpm"], 0.0)
            self.report = None
            self._pending = "strip"
        self._run("recreate", work)

    def score(self):
        if self.result is None or self.recon_feats is None:
            self.msg = "ingest and recreate first"
            return

        def work():
            from lib.gen.analysis import score as SC
            a = self.result["analysis"]
            self.report = SC.compare(self.result["features"], self.recon_feats, bpm_orig=a["bpm"], bpm_recon=self.script["bpm"],
                                     key_orig=self.script["key"], key_recon=self.script["key"])
            if self.folder:
                with open(os.path.join(self.folder, "score.json"), "w", encoding="utf-8") as fh:
                    json.dump(self.report, fh, indent=1)
            self._pending = "strip"
        self._run("score", work)

    def save(self):
        sc = self._script_from_table()
        if sc is None:
            return
        from lib.gen import script as S
        folder = self.folder or self._folder()
        self.script = sc
        path = S.save(sc, os.path.join(folder, "script.yaml"))
        self.msg = f"saved {path}"
        self._fill_table()

    def play(self):
        sc = self._script_from_table()
        if sc is None:
            return
        from lib.gen import script as S
        folder = self.folder or self._folder()
        path = os.path.abspath(S.save(sc, os.path.join(folder, "script.yaml")))
        self.console.ctx.emit("script", path)
        self.msg = f"sent {os.path.basename(folder)}/script.yaml to the show"

    def open_script(self):
        p = self.path.text().strip()
        cand = os.path.join(p, "script.yaml") if os.path.isdir(p) else p
        if not cand.lower().endswith((".yaml", ".yml", ".json")) or not os.path.exists(cand):
            p2, _ = QFileDialog.getOpenFileName(self, "Script", "logs/analysis", "SongScript (*.yaml *.yml *.json)")
            if not p2:
                return
            cand = p2
        from lib.gen import script as S
        self.script = S.load(cand)
        self.folder = os.path.dirname(cand)
        feats = os.path.join(self.folder, "features.json")
        if os.path.exists(feats):
            with open(feats, encoding="utf-8") as fh:
                saved = json.load(fh)
            self.result = {"features": saved["features"], "analysis": dict(saved["analysis"], sections=saved.get("sections", [])),
                           "bars": saved["bars"], "chords": saved["chords"], "script": self.script}
        self._fill_table()
        self.strip.set(orig=(self.result or {}).get("features", []), recon=[], report=None, script=self.script)
        self.msg = f"opened {cand}"

    # -- refresh (console timer) ----------------------------------------------
    def refresh(self, state):
        self.bar.setValue(int(100 * self.progress))
        if self._pending == "table":
            self._pending = None
            self._fill_table()
            a = self.result["analysis"]
            mat = ""
            if self.script.get("kit") or self.script.get("vocals"):
                mat = f"   reused: kit {','.join(self.script.get('kit') or {})}  vocals {len(self.script.get('vocals') or [])}" + \
                      ("  hook" if any(e.get("hook") for e in self.script["sections"]) else "")
            self.info.setText(f"{self.script.get('title')}: {self.script['style']} {self.script['bpm']:.1f} bpm {self.script['key']} "
                              f"(key conf {a['key_conf']:.2f}, tempo conf {a['bpm_conf']:.2f}), {len(self.script['sections'])} sections, "
                              f"{a['duration_s']:.0f} s{mat}")
            self.strip.set(orig=self.result["features"], recon=[], report=None, script=self.script)
        elif self._pending == "strip":
            self._pending = None
            self.strip.set(orig=(self.result or {}).get("features", []), recon=self.recon_feats or [], report=self.report, script=self.script)
            if self.report:
                from lib.gen.analysis import score as SC
                worst = ", ".join(f"bar {r['bar0']} ({r['score']:.0f})" for r in SC.worst(self.report))
                self.info.setText(f"global {self.report['global']:.1f}  local mean {self.report['mean_local']:.1f}  "
                                  f"structure {self.report['structure']:.1f}  tempo {self.report['tempo']:.0f}  key {self.report['key']:.0f}"
                                  f"   weakest: {worst}")
        if self.msg:
            self.console.notify(self.msg, quiet=True) if hasattr(self.console, "notify") else None
            self.msg = ""


def register(console):
    page = AnalysisPage(console)
    console.add_tab("Analysis", page)
