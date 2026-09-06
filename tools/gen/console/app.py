"""The native generative-music console (PyQt6).

Layout: a persistent HEADER STRIP (banner, transport, the now-strip) over a
QTabWidget whose tabs come from the surface spec (lib/gen/ui.py "tabs") -
Play / Steer / Scenes / Patterns / Log - plus a Setup tab the console adds
itself (where to play: here, headless, or a show over HTTP). A status bar
carries backend, uptime and engine health.

Extension points (see docs/GENERATIVE_UI.md):
  * spec: add a card to a tab, or a tab, in lib/gen/ui.py - no console code
  * widgets: tools/gen/console/widgets.py registry
  * plugins: tools/gen/console/plugins/<name>.py with `def register(console)`;
    a plugin can console.add_tab(), add_shortcut(), add_status(), on_state(),
    add_menu_action(). Discovered automatically; disable with --no-plugins.
"""
from __future__ import annotations

import importlib
import pkgutil
import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QApplication, QComboBox, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit,
                             QMainWindow, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QTabWidget,
                             QVBoxLayout, QWidget)

from lib.gen.ui import surface_spec
from tools.gen.console.widgets import REGISTRY

QSS = """
QMainWindow, QScrollArea, QWidget#root, QTabWidget::pane, QStatusBar { background: #14161c; color: #e0e2e8; }
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
QFrame#strip { background: #1b1e26; border: 1px solid #2c3140; border-radius: 10px; }
QLabel#cardTitle { font-size: 15px; font-weight: bold; }
QLabel#tabhint { color: #7f8694; font-size: 12px; padding: 2px 4px 6px 4px; }
QPushButton { border-radius: 8px; border: 2px solid #555; background: #2a2a2e; color: #ddd; padding: 9px 10px; font-size: 13px; }
QPushButton:pressed { background: #3a3a44; } QPushButton:disabled { color: #666; border-color: #333; }
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
QTabWidget::pane { border: 1px solid #2c3140; border-radius: 10px; top: -1px; }
QTabBar::tab { background: #1b1e26; color: #9aa0ab; padding: 9px 18px; border: 1px solid #2c3140; border-bottom: none; border-top-left-radius: 8px; border-top-right-radius: 8px; margin-right: 2px; font-size: 13px; }
QTabBar::tab:selected { background: #232838; color: #e0e2e8; font-weight: bold; }
QTabBar::tab:hover { color: #e0e2e8; }
QStatusBar QLabel { color: #9aa0ab; font-size: 12px; padding: 0 8px; }
QMenuBar { background: #14161c; color: #e0e2e8; } QMenuBar::item:selected { background: #232838; }
QMenu { background: #1b1e26; color: #e0e2e8; border: 1px solid #2c3140; } QMenu::item:selected { background: #232838; }
"""


def _visible_by_rule(rule, state):
    live = bool(state and state.get("active"))
    return bool(state) and (rule in (None, "always") or (rule == "live" and live) or (rule == "idle" and not live))


class Card(QFrame):
    def __init__(self, card, ctx):
        super().__init__()
        self.card = card
        self.setObjectName("strip" if card.get("kind") == "strip" else "card")
        self.widgets = []
        strip = card.get("kind") == "strip"
        lay = QVBoxLayout(self); lay.setContentsMargins(12, 6 if strip else 10, 12, 6 if strip else 12); lay.setSpacing(4 if strip else 8)
        self.body = QWidget(); body_lay = QVBoxLayout(self.body); body_lay.setContentsMargins(0, 0, 0, 0); body_lay.setSpacing(4 if strip else 8)
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
        vis = _visible_by_rule(self.card.get("show_when"), state) or self.card.get("kind") == "banner"
        self.setVisible(vis)
        if not vis:
            return
        for w in self.widgets:
            try:
                w.update_state(state or {})
            except Exception as e:  # noqa: BLE001
                print(f"[console] widget {w.spec.get('type')} failed: {e}")


class Ctx:
    """What widgets get: emit(), emit_throttled(), and the console itself."""

    def __init__(self, console):
        self.console = console
        self._last = {}

    def emit(self, action, value=None):
        be = self.console.backend
        if be is None:
            self.console.notify("Connect first (Setup tab)")
            return False
        ok = be.act(action, value)
        self.console.notify(f"{action}" + (f" {value}" if value not in (None, "") else ""), quiet=True)
        return ok

    def emit_throttled(self, action, value, ms=120):
        now = time.time()
        if now - self._last.get(action, 0) > ms / 1000.0:
            self._last[action] = now
            return self.emit(action, value)
        return False


class TabPage(QWidget):
    """One tab: two column stacks of cards (col 1 / col 2) in a scroll area."""

    def __init__(self, tab, cards, ctx):
        super().__init__()
        self.tab = tab
        self.cards = []
        outer = QVBoxLayout(self); outer.setContentsMargins(8, 6, 8, 8); outer.setSpacing(6)
        if tab.get("hint"):
            h = QLabel(tab["hint"]); h.setObjectName("tabhint"); outer.addWidget(h)
        cols = QHBoxLayout(); cols.setSpacing(10)
        stacks = {1: QVBoxLayout(), 2: QVBoxLayout()}
        for st in stacks.values():
            st.setSpacing(10); st.setAlignment(Qt.AlignmentFlag.AlignTop)
        for card in cards:
            c = Card(card, ctx); self.cards.append(c)
            stacks[2 if card.get("col") == 2 else 1].addWidget(c)
        used = [i for i in (1, 2) if stacks[i].count()]
        for i in used:
            w = QWidget(); w.setLayout(stacks[i]); cols.addWidget(w, 1)
        body = QWidget(); body.setObjectName("root"); body.setLayout(cols)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(body); scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll, 1)

    def refresh(self, state):
        for c in self.cards:
            c.refresh(state)


class SetupPage(QWidget):
    """Where to play. Shown first when the console starts unconnected."""

    def __init__(self, console):
        super().__init__()
        self.console = console
        lay = QVBoxLayout(self); lay.setContentsMargins(16, 12, 16, 12); lay.setSpacing(10)
        title = QLabel("Where should the music play?"); title.setObjectName("cardTitle"); lay.addWidget(title)
        hint = QLabel("Pick one. You can change it any time from this tab; the Play tab is where the night happens."); hint.setObjectName("dim"); hint.setWordWrap(True); lay.addWidget(hint)
        row = QHBoxLayout()
        # local
        loc = QFrame(); loc.setObjectName("card"); ll = QVBoxLayout(loc)
        ll.addWidget(self._h("This machine"))
        ll.addWidget(self._p("The composer and synth run here and play on this computer's speakers. For rehearsal, sound design and listening sessions."))
        form = QFormLayout(); self.style = QComboBox(); self.style.addItems(["groove", "downtempo", "ambient"])
        self.key = QLineEdit("8A"); self.key.setPlaceholderText("Camelot (8A) or name (Am)")
        self.fluid = QLineEdit(""); self.fluid.setPlaceholderText("SoundFont slots, e.g. keys,pad")
        form.addRow("style", self.style); form.addRow("key", self.key); form.addRow("SoundFont", self.fluid); ll.addLayout(form)
        b1 = QPushButton("▶ play here"); b1.setProperty("style", "go"); b1.clicked.connect(lambda: self.console.connect_local(audio=True, cfg=self._cfg()))
        b2 = QPushButton("dry run (no sound device)"); b2.clicked.connect(lambda: self.console.connect_local(audio=False, cfg=self._cfg()))
        ll.addWidget(b1); ll.addWidget(b2); row.addWidget(loc, 1)
        # remote
        rem = QFrame(); rem.setObjectName("card"); rl = QVBoxLayout(rem)
        rl.addWidget(self._h("The show box"))
        rl.addWidget(self._p("Remote-control a running show (or tools/gen/gen_server.py) over the venue network. The music plays there; this console steers it."))
        self.url = QLineEdit("http://lucifera.local:5000"); rl.addWidget(self.url)
        b3 = QPushButton("⇄ connect"); b3.setProperty("style", "alt"); b3.clicked.connect(lambda: self.console.connect_remote(self.url.text().strip()))
        rl.addWidget(b3); rl.addStretch(1); row.addWidget(rem, 1)
        lay.addLayout(row)
        self.status = QLabel(""); self.status.setObjectName("status"); self.status.setWordWrap(True); lay.addWidget(self.status)
        lay.addStretch(1)

    def _cfg(self):
        return {"style": self.style.currentText(), "key": self.key.text().strip() or "8A", "fluid_slots": self.fluid.text().strip(), "log_dir": "logs"}

    @staticmethod
    def _h(t):
        l = QLabel(t); l.setObjectName("cardTitle"); return l

    @staticmethod
    def _p(t):
        l = QLabel(t); l.setObjectName("dim"); l.setWordWrap(True); return l

    def refresh(self, state):
        be = self.console.backend
        if be is None:
            self.status.setProperty("tone", ""); self.status.setText("Not connected.")
        elif state and state.get("available"):
            self.status.setProperty("tone", "ok"); self.status.setText(f"Connected: {state.get('backend', '')}. Go to Play.")
        else:
            self.status.setProperty("tone", "err"); self.status.setText((state or {}).get("error") or "Connecting…")
        self.status.style().unpolish(self.status); self.status.style().polish(self.status)


class ConsoleWindow(QMainWindow):
    def __init__(self, backend=None, spec=None, refresh_ms=100, plugins=True):
        super().__init__()
        self.backend = backend
        self.spec = spec or surface_spec()
        self.ctx = Ctx(self)
        self.state = None
        self._state_hooks = []
        self._shortcuts = []          # (sequence, label)
        self.setWindowTitle(self.spec.get("title", "Lucifera Gen") + " · console")
        self.setStyleSheet(QSS)
        root = QWidget(); root.setObjectName("root"); self.setCentralWidget(root)
        outer = QVBoxLayout(root); outer.setContentsMargins(10, 8, 10, 6); outer.setSpacing(8)
        # header strip: every card not in a tab
        by_id = {c["id"]: c for c in self.spec.get("cards", [])}
        in_tabs = {cid for t in self.spec.get("tabs", []) for cid in t.get("cards", [])}
        self.strip_cards = []
        for c in self.spec.get("cards", []):
            if c["id"] not in in_tabs:
                card = Card(c, self.ctx); self.strip_cards.append(card); outer.addWidget(card)
        # tabs from the spec
        self.tabs = QTabWidget(); self.tabs.setDocumentMode(True); outer.addWidget(self.tabs, 1)
        self.pages = {}
        for t in self.spec.get("tabs", []):
            page = TabPage(t, [by_id[cid] for cid in t.get("cards", []) if cid in by_id], self.ctx)
            self.pages[t["id"]] = page; self.tabs.addTab(page, t["label"])
        self.setup = SetupPage(self); self.pages["setup"] = self.setup; self.tabs.addTab(self.setup, "Setup")
        # status bar
        sb = self.statusBar(); sb.setSizeGripEnabled(False)
        self.sb_left = QLabel(""); self.sb_msg = QLabel(""); self.sb_health = QLabel("")
        sb.addWidget(self.sb_left, 1); sb.addPermanentWidget(self.sb_msg); sb.addPermanentWidget(self.sb_health)
        # menu
        m = self.menuBar(); self.menus = {"File": m.addMenu("&File"), "View": m.addMenu("&View"), "Help": m.addMenu("&Help")}
        self.add_menu_action("File", "Play here", lambda: self.connect_local(True))
        self.add_menu_action("File", "Dry run (no sound device)", lambda: self.connect_local(False))
        self.add_menu_action("File", "Connect to show…", lambda: self.tabs.setCurrentWidget(self.setup))
        self.add_menu_action("File", "Quit", self.close, "Ctrl+Q")
        for i in range(self.tabs.count()):
            self.add_menu_action("View", self.tabs.tabText(i), lambda _c=False, i=i: self.tabs.setCurrentIndex(i), f"Ctrl+{i + 1}")
        # base shortcuts
        self.add_shortcut("Space", self._space, "start / stop")
        self.add_shortcut("Escape", lambda: self.ctx.emit("stop"), "stop")
        # plugins
        self.plugins = []
        if plugins:
            self.load_plugins()
        self.timer = QTimer(self); self.timer.timeout.connect(self.refresh); self.timer.start(refresh_ms)
        self.resize(1100, 860)
        self.tabs.setCurrentWidget(self.setup if self.backend is None else self.pages.get("play", self.setup))
        self.refresh()

    @property
    def cards(self):
        """Every spec card the window built: header strip + all tab pages."""
        out = list(self.strip_cards)
        for page in self.pages.values():
            out += getattr(page, "cards", [])
        return out

    # -- extension API -------------------------------------------------------
    def add_tab(self, title, widget, index=None):
        """A plugin's own tab. `widget` may implement refresh(state)."""
        if index is None:
            index = self.tabs.count() - 1          # before Setup
        self.tabs.insertTab(index, widget, title)
        self.pages[title.lower()] = widget
        return widget

    def add_shortcut(self, sequence, fn, label=""):
        sc = QShortcut(QKeySequence(sequence), self); sc.activated.connect(fn)
        self._shortcuts.append((sequence, label)); return sc

    def shortcuts(self):
        return list(self._shortcuts)

    def add_status(self, label_widget):
        self.statusBar().addPermanentWidget(label_widget); return label_widget

    def on_state(self, fn):
        """fn(state) after every refresh."""
        self._state_hooks.append(fn); return fn

    def add_menu_action(self, menu, text, fn, shortcut=None):
        a = QAction(text, self); a.triggered.connect(fn)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if menu not in self.menus:
            self.menus[menu] = self.menuBar().addMenu(menu)
        self.menus[menu].addAction(a); return a

    def notify(self, text, quiet=False):
        self.sb_msg.setText(text)
        if not quiet:
            QTimer.singleShot(4000, lambda: self.sb_msg.setText("") if self.sb_msg.text() == text else None)

    def load_plugins(self):
        import tools.gen.console.plugins as pkg
        for mod in pkgutil.iter_modules(pkg.__path__):
            try:
                m = importlib.import_module(f"{pkg.__name__}.{mod.name}")
                if hasattr(m, "register"):
                    m.register(self); self.plugins.append(mod.name)
            except Exception as e:  # noqa: BLE001
                print(f"[console] plugin {mod.name} failed: {e}")

    # -- backends ---------------------------------------------------------------
    def set_backend(self, backend):
        if self.backend is not None and self.backend is not backend:
            try:
                self.backend.close()
            except Exception:
                pass
        self.backend = backend
        self.refresh()
        self.tabs.setCurrentWidget(self.pages.get("play", self.setup))

    def connect_local(self, audio=True, cfg=None):
        from tools.gen.console.backend import LocalBackend
        self.set_backend(LocalBackend(cfg or {"style": "groove", "key": "8A", "log_dir": "logs"}, audio=audio))
        self.notify("playing here" if audio else "dry run (no sound device)")

    def connect_remote(self, url):
        from tools.gen.console.backend import RemoteBackend
        if not url:
            return
        self.set_backend(RemoteBackend(url)); self.notify(f"connected to {url}")

    # -- loop -------------------------------------------------------------------
    def _space(self):
        self.ctx.emit("stop" if (self.state and self.state.get("active")) else "start")

    def refresh(self):
        if self.backend is None:
            self.state = None
        else:
            try:
                self.state = self.backend.status()
            except Exception as e:  # noqa: BLE001
                self.state = {"available": False, "active": False, "error": str(e)}
        s = self.state
        for c in self.strip_cards:
            c.refresh(s)
        for page in self.pages.values():
            if hasattr(page, "refresh"):
                page.refresh(s)
        left = (s or {}).get("backend", "not connected") if s else "not connected"
        if s and s.get("active"):
            left += f"   ·   {s.get('style')} · {s.get('section')} · bar {s.get('bar')} · {s.get('bpm')} bpm · {s.get('camelot')} · up {int(s.get('uptime_s', 0)) // 60} min"
            re = s.get("render_errors", 0)
            self.sb_health.setText(f"ahead {s.get('lead_s')} s · peak {s.get('peak')}" + (f" · render errors {re}" if re else ""))
        else:
            self.sb_health.setText("")
        self.sb_left.setText(left)
        for fn in self._state_hooks:
            try:
                fn(s)
            except Exception as e:  # noqa: BLE001
                print(f"[console] state hook failed: {e}")

    def closeEvent(self, ev):
        self.timer.stop()
        try:
            if self.backend is not None:
                self.backend.close()
        except Exception:
            pass
        super().closeEvent(ev)


def run(backend=None, argv=None, plugins=True):
    app = QApplication.instance() or QApplication(argv or [])
    win = ConsoleWindow(backend, plugins=plugins)
    win.show()
    return app.exec()
