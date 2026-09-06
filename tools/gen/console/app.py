"""The native generative-music console: a PyQt6 window that renders the
same surface spec as the web page (lib/gen/ui.py) with native widgets
(tools/gen/console/widgets.py), against a LocalBackend (audio here) or a
RemoteBackend (the show box). Two column stacks, foldable cards, 10 Hz
refresh, keyboard shortcuts (Space start/stop, Esc stop, Ctrl+Q quit)."""
from __future__ import annotations

import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton, QScrollArea,
                             QSizePolicy, QVBoxLayout, QWidget)

from lib.gen.ui import surface_spec
from tools.gen.console.widgets import REGISTRY

QSS = """
QMainWindow, QScrollArea, QWidget#root { background: #14161c; color: #e0e2e8; }
QLabel { color: #e0e2e8; font-size: 13px; }
QLabel#dim { color: #9aa0ab; font-size: 12px; }
QLabel#headline { font-size: 22px; font-weight: bold; }
QLabel#banner { padding: 8px 12px; border-radius: 8px; border: 1px solid #444; background: #23232a; }
QLabel#banner[tone="on"] { border-color: #3a6; background: #123020; color: #9f9; }
QLabel#banner[tone="err"] { border-color: #a33; background: #301515; color: #f99; }
QLabel#status { font-size: 12px; } QLabel#status[tone="ok"] { color: #9f9; } QLabel#status[tone="err"] { color: #f99; } QLabel#status[tone="busy"] { color: #9ef; }
QLabel#chord { padding: 8px 2px; border-radius: 6px; border: 1px solid #3a3a44; background: #1e1e26; font-size: 15px; }
QLabel#chord[on="true"] { border-color: #4bd; background: #123a44; color: #9ef; font-weight: bold; }
QLabel#countdown { color: #e64; font-size: 15px; min-height: 20px; } QLabel#countdown[hot="true"] { font-size: 22px; font-weight: bold; }
QFrame#beat { border-radius: 4px; background: #26262e; border: 1px solid #333; }
QFrame#beat[on="true"] { background: #4bd; } QFrame#beat[down="true"][on="true"] { background: #e64; }
QFrame#card { background: #1b1e26; border: 1px solid #2c3140; border-radius: 10px; }
QLabel#cardTitle { font-size: 15px; font-weight: bold; }
QPushButton { border-radius: 8px; border: 2px solid #555; background: #2a2a2e; color: #ddd; padding: 9px 10px; font-size: 13px; }
QPushButton:pressed { background: #3a3a44; }
QPushButton[style="go"] { border-color: #3a6; background: #123a22; color: #9f9; font-weight: bold; }
QPushButton[style="stop"] { border-color: #a33; background: #401515; color: #f99; }
QPushButton[style="alt"] { border-color: #46a; background: #14203a; color: #9bf; }
QPushButton[on="true"] { border-color: #4bd; background: #123a44; color: #9ef; }
QPushButton#chip { border-radius: 14px; border: 1px solid #555; background: #26262e; color: #bbb; padding: 6px 12px; font-size: 12px; }
QPushButton#chip[live="true"] { border-color: #3a6; color: #9f9; } QPushButton#chip[muted="true"] { border-color: #a33; color: #f99; }
QPushButton#choice { border-radius: 6px; border: 1px solid #555; background: #26262e; color: #ccc; font-size: 12px; }
QPushButton#choice:checked { border-color: #4bd; background: #123a44; color: #9ef; }
QPushButton#fold { border: none; background: transparent; color: #9aa0ab; padding: 2px 6px; }
QSlider::groove:horizontal { height: 8px; background: #2a2f3c; border-radius: 4px; }
QSlider::handle:horizontal { width: 18px; margin: -6px 0; border-radius: 9px; background: #4bd; }
QProgressBar { border: none; background: #222; border-radius: 6px; } QProgressBar::chunk { border-radius: 6px; background: #4bd; }
QProgressBar[palette="energy"]::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2a6, stop:0.5 #dc4, stop:1 #e64); }
QProgressBar[palette="section"]::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2a6, stop:1 #4bd); }
QProgressBar[palette="arc"]::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #235, stop:0.5 #a24, stop:1 #722); }
QComboBox, QLineEdit { padding: 7px; background: #26262e; color: #ccc; border: 1px solid #555; border-radius: 6px; }
QPlainTextEdit#log { background: #15151c; color: #cfd3dc; border: 1px solid #2c3140; border-radius: 6px; font-family: Menlo, Consolas, monospace; font-size: 11px; }
QPlainTextEdit#code { background: #15151c; color: #ddd; border: 1px solid #555; border-radius: 6px; font-family: Menlo, Consolas, monospace; font-size: 12px; }
"""


class Card(QFrame):
    def __init__(self, card, ctx):
        super().__init__()
        self.card = card
        self.setObjectName("card")
        self.widgets = []
        lay = QVBoxLayout(self); lay.setContentsMargins(12, 10, 12, 12); lay.setSpacing(8)
        self.body = QWidget(); body_lay = QVBoxLayout(self.body); body_lay.setContentsMargins(0, 0, 0, 0); body_lay.setSpacing(8)
        if card.get("title"):
            head = QHBoxLayout()
            t = QLabel(card["title"] + (f"  <span style='color:#9aa0ab;font-weight:normal;font-size:12px'>{card['hint']}</span>" if card.get("hint") else ""))
            t.setObjectName("cardTitle"); head.addWidget(t); head.addStretch(1)
            if card.get("foldable"):
                self.fold_btn = QPushButton("▾"); self.fold_btn.setObjectName("fold"); self.fold_btn.setFixedWidth(28)
                self.fold_btn.clicked.connect(self.toggle_fold); head.addWidget(self.fold_btn)
            lay.addLayout(head)
        for w in card.get("widgets", []):
            cls = REGISTRY.get(w["type"])
            if cls is None:
                body_lay.addWidget(QLabel(f"unknown widget {w['type']}")); continue
            inst = cls(w, ctx); body_lay.addWidget(inst); self.widgets.append(inst)
        lay.addWidget(self.body)
        if card.get("foldable") and card.get("folded"):
            self.toggle_fold()

    def toggle_fold(self):
        vis = not self.body.isVisible()
        self.body.setVisible(vis)
        if hasattr(self, "fold_btn"):
            self.fold_btn.setText("▾" if vis else "▸")

    def refresh(self, state):
        live = bool(state and state.get("active"))
        rule = self.card.get("show_when", "always")
        vis = bool(state) and (rule == "always" or (rule == "live" and live) or (rule == "idle" and not live))
        if self.card.get("kind") == "banner":
            vis = True
        self.setVisible(vis)
        if not vis:
            return
        for w in self.widgets:
            try:
                w.update_state(state or {})
            except Exception as e:  # noqa: BLE001
                print(f"[console] widget {w.spec.get('type')} failed: {e}")


class Ctx:
    def __init__(self, backend):
        self.backend = backend
        self._last = {}

    def emit(self, action, value=None):
        return self.backend.act(action, value)

    def emit_throttled(self, action, value, ms=120):
        now = time.time()
        if now - self._last.get(action, 0) > ms / 1000.0:
            self._last[action] = now
            return self.emit(action, value)
        return False


class ConsoleWindow(QMainWindow):
    def __init__(self, backend, spec=None, refresh_ms=100):
        super().__init__()
        self.backend = backend
        self.spec = spec or surface_spec()
        self.ctx = Ctx(backend)
        self.setWindowTitle(self.spec.get("title", "Lucifera Gen") + " · console")
        self.setStyleSheet(QSS)
        root = QWidget(); root.setObjectName("root"); self.setCentralWidget(root)
        outer = QVBoxLayout(root); outer.setContentsMargins(10, 10, 10, 10); outer.setSpacing(10)
        self.cards = []
        cols = QHBoxLayout(); cols.setSpacing(10)
        stacks = {1: QVBoxLayout(), 2: QVBoxLayout()}
        for st in stacks.values():
            st.setSpacing(10); st.setAlignment(Qt.AlignmentFlag.AlignTop)
        for card in self.spec.get("cards", []):
            c = Card(card, self.ctx); self.cards.append(c)
            if card.get("kind") in ("banner", "transport"):
                outer.addWidget(c)
            else:
                stacks[2 if card.get("col") == 2 else 1].addWidget(c)
        for i in (1, 2):
            w = QWidget(); w.setLayout(stacks[i]); cols.addWidget(w, 1)
        scroll_body = QWidget(); scroll_body.setObjectName("root"); scroll_body.setLayout(cols)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(scroll_body); scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll, 1)
        self.state = None
        self.timer = QTimer(self); self.timer.timeout.connect(self.refresh); self.timer.start(refresh_ms)
        QShortcut(QKeySequence("Space"), self, activated=self._space)
        QShortcut(QKeySequence("Escape"), self, activated=lambda: self.ctx.emit("stop"))
        QShortcut(QKeySequence("Ctrl+Q"), self, activated=self.close)
        self.resize(1100, 900)
        self.refresh()

    def _space(self):
        self.ctx.emit("stop" if (self.state and self.state.get("active")) else "start")

    def refresh(self):
        try:
            self.state = self.backend.status()
        except Exception as e:  # noqa: BLE001
            self.state = {"available": False, "active": False, "error": str(e)}
        for c in self.cards:
            c.refresh(self.state)

    def closeEvent(self, ev):
        self.timer.stop()
        try:
            self.backend.close()
        except Exception:
            pass
        super().closeEvent(ev)


def run(backend, argv=None):
    app = QApplication.instance() or QApplication(argv or [])
    win = ConsoleWindow(backend)
    win.show()
    return app.exec()
