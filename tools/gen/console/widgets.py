"""Native (PyQt6) widgets for the /gen surface spec (lib/gen/ui.py).

One class per widget type, registered by name - the same contract as the
web renderer: create from a spec, update(state) at 10 Hz, send operator
input through ctx.emit(action, value). To add a widget type: subclass
GenWidget, implement build()/update(), and @register('name'). The
validator in lib/gen/ui.py accepts a registry from here too."""
from __future__ import annotations

import json

from PyQt6.QtCore import Qt, QRect, QSize, QPoint
from PyQt6.QtWidgets import (QComboBox, QFrame, QGridLayout, QHBoxLayout, QInputDialog, QLabel, QLayout,
                             QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy,
                             QSlider, QVBoxLayout, QWidget)

from lib.gen.ui import _widget_keys  # noqa: F401  (keys contract shared with the web renderer)

REGISTRY = {}


def register(name):
    def deco(cls):
        REGISTRY[name] = cls
        cls.type_name = name
        return cls
    return deco


def _fmt(pattern, *args):
    """The web store's tiny Python-ish format, in Python."""
    named = args[0] if (len(args) == 1 and isinstance(args[0], dict)) else None
    import re

    def rep(m):
        ref, spec = m.group(1), m.group(2)
        if ref.isdigit():
            v = args[int(ref)] if int(ref) < len(args) else None
        else:
            mm = re.match(r"^([a-zA-Z_]+)\[(\d+)\]$", ref)
            if mm:
                o = (named or {}).get(mm.group(1))
                v = o[int(mm.group(2))] if isinstance(o, (list, tuple)) and int(mm.group(2)) < len(o) else None
            else:
                v = (named or {}).get(ref)
        if v is None:
            return "–"
        if spec:
            s = re.match(r"^([+])?\.(\d)f$", spec)
            if s and isinstance(v, (int, float)):
                t = f"{v:.{int(s.group(2))}f}"
                return ("+" + t) if (s.group(1) and v >= 0) else t
        return " / ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)
    return re.sub(r"\{([^{}:]+)(?::([^{}]+))?\}", rep, pattern)


def _duration(sec):
    sec = int(max(0, sec or 0))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")


class FlowLayout(QLayout):
    """Chips that wrap (Qt's flow layout example, trimmed)."""

    def __init__(self, parent=None, spacing=6):
        super().__init__(parent)
        self._items = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item): self._items.append(item)
    def count(self): return len(self._items)
    def itemAt(self, i): return self._items[i] if 0 <= i < len(self._items) else None
    def takeAt(self, i): return self._items.pop(i) if 0 <= i < len(self._items) else None
    def expandingDirections(self): return Qt.Orientation(0)
    def hasHeightForWidth(self): return True
    def heightForWidth(self, w): return self._do(QRect(0, 0, w, 0), True)
    def sizeHint(self): return self.minimumSize()

    def minimumSize(self):
        s = QSize()
        for it in self._items:
            s = s.expandedTo(it.minimumSize())
        return s + QSize(2, 2)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do(rect, False)

    def _do(self, rect, test):
        x, y, line_h = rect.x(), rect.y(), 0
        sp = self.spacing()
        for it in self._items:
            w = it.sizeHint().width()
            if x + w > rect.right() and line_h > 0:
                x = rect.x(); y += line_h + sp; line_h = 0
            if not test:
                it.setGeometry(QRect(QPoint(x, y), it.sizeHint()))
            x += w + sp
            line_h = max(line_h, it.sizeHint().height())
        return y + line_h - rect.y()

    def clear(self):
        while self._items:
            it = self._items.pop()
            if it.widget():
                it.widget().deleteLater()


class GenWidget(QWidget):
    def __init__(self, spec, ctx):
        super().__init__()
        self.spec = spec
        self.ctx = ctx
        self.build()

    def build(self): ...
    def update_state(self, s): ...


@register("banner")
class Banner(GenWidget):
    def build(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.lab = QLabel("Connecting…"); self.lab.setObjectName("banner"); self.lab.setWordWrap(True); lay.addWidget(self.lab)

    def update_state(self, s):
        if not s or not s.get("available"):
            self.lab.setProperty("tone", "err"); self.lab.setText(s.get("error") if s else "No generative subsystem.")
        elif s.get("error"):
            self.lab.setProperty("tone", "err"); self.lab.setText("Error: " + str(s["error"]))
        elif s.get("active"):
            mode = {"ending": "ending after the outro", "hold": "holding this section"}.get(s.get("state"), "autonomous")
            self.lab.setProperty("tone", "on"); self.lab.setText(f"Playing · {s.get('style')} · {s.get('bpm')} bpm · {s.get('key')} ({s.get('camelot')}) · {mode} · {s.get('backend', '')}")
        else:
            self.lab.setProperty("tone", ""); self.lab.setText(f"Idle. Steering arms the next start. · {s.get('backend', '')}")
        self.lab.style().unpolish(self.lab); self.lab.style().polish(self.lab)


@register("buttons")
class Buttons(GenWidget):
    def build(self):
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.btns = []
        for it in self.spec.get("items", []):
            b = QPushButton(it.get("label", "")); b.setProperty("style", it.get("style", "")); b._item = it
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            b.clicked.connect(lambda _c=False, b=b: self._press(b))
            lay.addWidget(b); self.btns.append(b)
        self.trail = QLabel(""); self.trail.setObjectName("dim")
        if self.spec.get("trailing_key"):
            lay.addWidget(self.trail)

    def _press(self, b):
        it = b._item
        if it.get("confirm") and QMessageBox.question(self, "Confirm", it["confirm"]) != QMessageBox.StandardButton.Yes:
            return
        v = it.get("value")
        if it.get("toggle_key"):
            v = not b.property("on")
        self.ctx.emit(it["action"], v)

    def update_state(self, s):
        live = bool(s and s.get("active"))
        for b in self.btns:
            rule = b._item.get("show_when", "always")
            b.setVisible(rule == "always" or (rule == "live" and live) or (rule == "idle" and not live))
            if b._item.get("toggle_key") and s:
                on = s.get(b._item["toggle_key"]) == b._item.get("toggle_value")
                b.setProperty("on", on); b.style().unpolish(b); b.style().polish(b)
        if self.spec.get("trailing_key") and s:
            self.trail.setText(_fmt(self.spec.get("trailing_format", "{0}"), s.get(self.spec["trailing_key"]) or {}))


@register("headline")
class Headline(GenWidget):
    def build(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.lab = QLabel("--"); self.lab.setObjectName("headline"); lay.addWidget(self.lab)

    def update_state(self, s):
        sub = "".join(f" · {k} {s.get(k)}" for k in self.spec.get("sub_keys", []))
        a = s.get(self.spec.get("arrow_key") or "")
        self.lab.setText(f"<span style='color:#9ef'>{str(s.get(self.spec['key'], '--')).upper()}</span>{sub}" + (f" <span style='color:#e64;font-size:12px'>→ {a}</span>" if a else ""))


@register("keyline")
class Keyline(GenWidget):
    def build(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.lab = QLabel("--"); self.lab.setObjectName("dim"); lay.addWidget(self.lab)

    def update_state(self, s):
        k = self.spec.get("keys", [])
        self.lab.setText(f"{s.get(k[0])} ({s.get(k[1])}) · {s.get(k[2])} bpm · chord {s.get(k[3]) or '–'}" + (f" · motif {s.get(k[4])}" if len(k) > 4 and s.get(k[4]) else ""))


@register("beats")
class Beats(GenWidget):
    def build(self):
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 4, 0, 4)
        self.cells = []
        for i in range(4):
            f = QFrame(); f.setFixedHeight(14); f.setObjectName("beat"); f.setProperty("down", i == 0)
            lay.addWidget(f); self.cells.append(f)

    def update_state(self, s):
        b = s.get(self.spec["key"])
        for i, f in enumerate(self.cells):
            f.setProperty("on", b == i + 1); f.style().unpolish(f); f.style().polish(f)


@register("chords")
class Chords(GenWidget):
    def build(self):
        self.lay = QHBoxLayout(self); self.lay.setContentsMargins(0, 0, 0, 0); self.labels = []

    def update_state(self, s):
        chords = s.get(self.spec["key"]) or []
        if len(self.labels) != len(chords):
            for l in self.labels:
                l.deleteLater()
            self.labels = []
            for _ in chords:
                l = QLabel(); l.setObjectName("chord"); l.setAlignment(Qt.AlignmentFlag.AlignCenter); self.lay.addWidget(l); self.labels.append(l)
        cur = int((s.get(self.spec.get("phase_key") or "", 0) or 0) * len(chords)) if chords else -1
        for i, (l, c) in enumerate(zip(self.labels, chords)):
            l.setText(str(c)); l.setProperty("on", i == cur); l.style().unpolish(l); l.style().polish(l)


@register("countdown")
class Countdown(GenWidget):
    def build(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.lab = QLabel(""); self.lab.setObjectName("countdown"); self.lab.setAlignment(Qt.AlignmentFlag.AlignCenter); lay.addWidget(self.lab)

    def update_state(self, s):
        v = s.get(self.spec["key"])
        self.setVisible(v is not None)
        if v is None:
            self.lab.setText(""); self.lab.setProperty("hot", False)
        else:
            self.lab.setText(f"{self.spec.get('label', '')} {float(v):.1f} s"); self.lab.setProperty("hot", float(v) < self.spec.get("hot_below", 0))
        self.lab.style().unpolish(self.lab); self.lab.style().polish(self.lab)


@register("meter")
class Meter(GenWidget):
    def build(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 2, 0, 2); lay.setSpacing(2)
        row = QHBoxLayout(); self.left = QLabel(self.spec.get("label", "")); self.left.setObjectName("dim")
        self.right = QLabel(""); self.right.setObjectName("dim"); self.right.setAlignment(Qt.AlignmentFlag.AlignRight)
        row.addWidget(self.left); row.addWidget(self.right); lay.addLayout(row)
        self.bar = QProgressBar(); self.bar.setRange(0, 1000); self.bar.setTextVisible(False); self.bar.setFixedHeight(12)
        self.bar.setProperty("palette", self.spec.get("palette", "plain")); lay.addWidget(self.bar)

    def update_state(self, s):
        sp = self.spec
        if sp.get("done_key") and sp.get("total_key"):
            tot, left = (s.get(sp["total_key"]) or 0), (s.get(sp["done_key"]) or 0)
            frac = ((tot - left) / tot if sp.get("inverse") else left / tot) if tot else 0.0
        else:
            frac = float(s.get(sp["key"]) or 0.0)
        self.bar.setValue(int(1000 * max(0.0, min(1.0, frac))))
        if sp.get("right_keys"):
            self.right.setText(_fmt(sp.get("right_format", "{0}"), *[s.get(k) for k in sp["right_keys"]]))


@register("kv")
class KV(GenWidget):
    def build(self):
        g = QGridLayout(self); g.setContentsMargins(0, 0, 0, 0); g.setHorizontalSpacing(14); g.setVerticalSpacing(2)
        self.vals = []
        for i, it in enumerate(self.spec.get("items", [])):
            k = QLabel(it["label"]); k.setObjectName("dim"); v = QLabel("–"); v.setAlignment(Qt.AlignmentFlag.AlignRight)
            g.addWidget(k, i // 2, (i % 2) * 2); g.addWidget(v, i // 2, (i % 2) * 2 + 1); self.vals.append((it, v))

    def update_state(self, s):
        for it, lab in self.vals:
            v = s.get(it["key"]); f = it.get("format")
            if f == "duration": v = _duration(v)
            elif f == "list": v = (", ".join(v) or "none") if isinstance(v, list) else (v or "none")
            elif f == "json": v = json.dumps(v) if v else "none"
            elif f: v = _fmt(f, v)
            lab.setText("–" if v is None else str(v))


@register("chips")
class Chips(GenWidget):
    def build(self):
        self.flow = FlowLayout(self); self._key = None

    def update_state(self, s):
        items = s.get(self.spec["items_key"]) or []
        idf, labf = self.spec.get("id_field", "id"), self.spec.get("label_field", "label")
        key = "|".join(str(i.get(idf)) for i in items)
        if key != self._key:
            self._key = key; self.flow.clear()
            for it in items:
                b = QPushButton(str(it.get(labf))); b.setObjectName("chip")
                b.clicked.connect(lambda _c=False, v=it.get(idf): self.ctx.emit(self.spec["action"], v))
                self.flow.addWidget(b)


@register("choice")
class Choice(GenWidget):
    def build(self):
        self.lay = QHBoxLayout(self); self.lay.setContentsMargins(0, 0, 0, 0); self.btns = {}; self._key = None

    def update_state(self, s):
        opts = s.get(self.spec["options_key"]) or []
        idf = self.spec.get("id_field", "id")
        key = "|".join(str(o.get(idf)) for o in opts)
        if key != self._key:
            self._key = key
            for b in self.btns.values():
                b.deleteLater()
            self.btns = {}
            for o in opts:
                sub = _fmt(self.spec["sub_format"], o) if self.spec.get("sub_format") else ""
                short = str(o.get("label") or o.get(idf)).split(" (")[0].split(" /")[0]
                b = QPushButton(str(o.get(idf)) + ("\n" + short if short and short != str(o.get(idf)) else "")); b.setObjectName("choice"); b.setCheckable(True)
                if sub:
                    b.setToolTip(sub)
                b.clicked.connect(lambda _c=False, v=o.get(idf): self.ctx.emit(self.spec["action"], v))
                self.lay.addWidget(b); self.btns[str(o.get(idf))] = b
        cur = str(s.get(self.spec["key"]))
        for k, b in self.btns.items():
            b.setChecked(k == cur)


@register("slider")
class Slider(GenWidget):
    SCALE = 1000

    def build(self):
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lab = QLabel(self.spec.get("label", self.spec["key"])); lab.setFixedWidth(92); lab.setObjectName("dim")
        self.sl = QSlider(Qt.Orientation.Horizontal); self.lo, self.hi = float(self.spec["min"]), float(self.spec["max"])
        self.sl.setRange(0, self.SCALE); self.sl.setSingleStep(1)
        self.val = QLabel("--"); self.val.setFixedWidth(60); self.val.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.sl.valueChanged.connect(self._changed); self.sl.sliderReleased.connect(self._released)
        lay.addWidget(lab); lay.addWidget(self.sl); lay.addWidget(self.val)
        self._dragging = False; self.sl.sliderPressed.connect(lambda: setattr(self, "_dragging", True))

    def _to_value(self, pos):
        v = self.lo + (self.hi - self.lo) * pos / self.SCALE
        step = float(self.spec.get("step", 0.01))
        return round(round(v / step) * step, 6)

    def _show(self, v):
        d = int(self.spec.get("decimals", 2)); self.val.setText(("+" if self.spec.get("signed") and v >= 0 else "") + f"{v:.{d}f}")

    def _changed(self, pos):
        v = self._to_value(pos); self._show(v)
        if self.sl.isSliderDown():
            self.ctx.emit_throttled(self.spec["action"], v)
        elif not getattr(self, "_syncing", False):
            self.ctx.emit(self.spec["action"], v)

    def _released(self):
        self._dragging = False; self.ctx.emit(self.spec["action"], self._to_value(self.sl.value()))

    def update_state(self, s):
        v = s.get(self.spec["key"])
        if v is None or self.sl.isSliderDown():
            return
        pos = int(round((float(v) - self.lo) / (self.hi - self.lo) * self.SCALE))
        if pos != self.sl.value():
            self._syncing = True; self.sl.setValue(pos); self._syncing = False
        self._show(float(v))


@register("select")
class Select(GenWidget):
    def build(self):
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lab = QLabel(self.spec.get("label", self.spec["key"])); lab.setFixedWidth(92); lab.setObjectName("dim")
        self.cb = QComboBox()
        opts = self.spec.get("options")
        if opts == "camelot":
            names = {"1A": "Ab min", "2A": "Eb min", "3A": "Bb min", "4A": "F min", "5A": "C min", "6A": "G min", "7A": "D min", "8A": "A min",
                     "9A": "E min", "10A": "B min", "11A": "F# min", "12A": "C# min", "1B": "B maj", "2B": "F# maj", "3B": "Db maj", "4B": "Ab maj",
                     "5B": "Eb maj", "6B": "Bb maj", "7B": "F maj", "8B": "C maj", "9B": "G maj", "10B": "D maj", "11B": "A maj", "12B": "E maj"}
            opts = [{"id": f"{n}{ab}", "label": f"{n}{ab}  {names[f'{n}{ab}']}"} for n in range(1, 13) for ab in "AB"]
        for o in opts or []:
            self.cb.addItem(str(o["label"]), o["id"])
        self.cb.activated.connect(lambda i: self.ctx.emit(self.spec["action"], self.cb.itemData(i)))
        lay.addWidget(lab); lay.addWidget(self.cb)
        self.trail = QLabel(""); self.trail.setObjectName("dim")
        if self.spec.get("trailing_key"):
            lay.addWidget(self.trail)

    def update_state(self, s):
        v = s.get(self.spec["key"])
        if v is None and self.spec.get("idle_key"):
            v = s.get(self.spec["idle_key"])
        if v is not None:
            target = int(round(v)) if isinstance(v, (int, float)) else v
            i = self.cb.findData(target)
            if i >= 0 and i != self.cb.currentIndex():
                self.cb.setCurrentIndex(i)
        if self.spec.get("trailing_key"):
            self.trail.setText(str(s.get(self.spec["trailing_key"]) or ""))


@register("text")
class Text(GenWidget):
    def build(self):
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lab = QLabel(self.spec.get("label", self.spec["key"])); lab.setFixedWidth(92); lab.setObjectName("dim")
        self.ed = QLineEdit(); self.ed.setPlaceholderText(self.spec.get("placeholder", ""))
        self.ed.editingFinished.connect(lambda: self.ctx.emit(self.spec["action"], self.ed.text().strip()))
        lay.addWidget(lab); lay.addWidget(self.ed)

    def update_state(self, s):
        if self.ed.hasFocus():
            return
        v = s.get(self.spec["key"]); self.ed.setText(",".join(v) if isinstance(v, list) else (v or ""))


@register("toggles")
class Toggles(GenWidget):
    def build(self):
        self.flow = FlowLayout(self); self.btns = {}; self._key = None

    def update_state(self, s):
        items = s.get(self.spec["items_key"]) or []
        key = "|".join(items)
        if key != self._key:
            self._key = key; self.flow.clear(); self.btns = {}
            for it in items:
                b = QPushButton(it); b.setObjectName("chip"); b._item = it
                b.clicked.connect(lambda _c=False, b=b: self._press(b))
                self.flow.addWidget(b); self.btns[it] = b
        on, off, badge = set(s.get(self.spec["on_key"]) or []), set(s.get(self.spec["off_key"]) or []), set(s.get(self.spec.get("badge_key") or "") or [])
        for it, b in self.btns.items():
            b.setProperty("live", it in on and it not in off); b.setProperty("muted", it in off)
            b.setText(it + (f" {self.spec.get('badge', '')}" if it in badge else "")); b.style().unpolish(b); b.style().polish(b)

    def _press(self, b):
        nxt = not b.property("muted")
        vf = self.spec.get("value_format")
        value = {k: (b._item if v == "$item" else nxt if v == "$next" else v) for k, v in vf.items()} if isinstance(vf, dict) else b._item
        self.ctx.emit(self.spec["action"], value)


@register("ask")
class Ask(GenWidget):
    def build(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 6, 0, 0)
        row = QHBoxLayout(); self.ed = QLineEdit(); self.ed.setPlaceholderText(self.spec.get("placeholder", ""))
        go = QPushButton("ASK"); go.setProperty("style", "alt"); go.clicked.connect(self._send); self.ed.returnPressed.connect(self._send)
        row.addWidget(self.ed); row.addWidget(go); lay.addLayout(row)
        self.status = QLabel(""); self.status.setObjectName("status"); self.status.setWordWrap(True); lay.addWidget(self.status)

    def _send(self):
        t = self.ed.text().strip()
        if t:
            self.ctx.emit(self.spec["action"], t); self.ed.clear()

    def update_state(self, s):
        d = s.get(self.spec["status_key"]) or {}
        last = d.get("last") or {}
        if not d.get("available"):
            tone, text = "err", "director offline (install Claude Code `claude`, or pip install anthropic + ANTHROPIC_API_KEY) - gestures still work"
        elif d.get("busy"):
            tone, text = "busy", "director thinking about: " + str(last.get("text", ""))
        elif last.get("error"):
            tone, text = "err", "director error: " + str(last["error"])
        elif "say" in last:
            tone, text = "ok", "director: " + (last.get("say") or "(done)") + ("  · " + "; ".join(last["warn"]) if last.get("warn") else "")
        else:
            tone, text = "", f"director ready ({d.get('mode') or ''})"
        self.status.setProperty("tone", tone); self.status.setText(text); self.status.style().unpolish(self.status); self.status.style().polish(self.status)


@register("director_log")
class DirectorLog(GenWidget):
    def build(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.box = QPlainTextEdit(); self.box.setReadOnly(True); self.box.setObjectName("log"); self.box.setFixedHeight(110); lay.addWidget(self.box)

    def update_state(self, s):
        d = s.get(self.spec["key"]) or {}
        rows = list(d.get("log") or [])[-int(self.spec.get("limit", 8)):][::-1]
        text = "\n".join(f"{r.get('kind', ''):8s} {r.get('text', '')} → {', '.join(r.get('done') or []) or r.get('say', '')}" for r in rows)
        if text != self.box.toPlainText():
            self.box.setPlainText(text)


@register("scenes")
class Scenes(GenWidget):
    def build(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout(); lab = QLabel("scene"); lab.setFixedWidth(92); lab.setObjectName("dim"); self.cb = QComboBox(); row.addWidget(lab); row.addWidget(self.cb); lay.addLayout(row)
        btns = QHBoxLayout(); a = self.spec["actions"]
        save = QPushButton("＋ save as…"); save.setProperty("style", "go"); save.clicked.connect(self._save)
        load = QPushButton("▶ recall"); load.setProperty("style", "alt"); load.clicked.connect(lambda: self.cb.currentData() and self.ctx.emit(a["load"], self.cb.currentData()))
        dele = QPushButton("✕ delete"); dele.clicked.connect(self._delete)
        for b in (save, load, dele):
            btns.addWidget(b)
        lay.addLayout(btns); self._key = None

    def _save(self):
        name, ok = QInputDialog.getText(self, "Save scene", "Scene name")
        if ok and name.strip():
            self.ctx.emit(self.spec["actions"]["save"], name.strip())

    def _delete(self):
        n = self.cb.currentData()
        if n and QMessageBox.question(self, "Delete scene", f'Delete scene "{n}"?') == QMessageBox.StandardButton.Yes:
            self.ctx.emit(self.spec["actions"]["delete"], n)

    def update_state(self, s):
        scenes = s.get(self.spec["key"]) or []
        key = "|".join(sc["name"] for sc in scenes)
        if key == self._key:
            return
        self._key = key; cur = self.cb.currentData(); self.cb.clear()
        if not scenes:
            self.cb.addItem("(no scenes saved yet)", None)
        for sc in scenes:
            self.cb.addItem(f"{sc['name']}  · {sc.get('style') or ''} {sc.get('bpm') or ''} {sc.get('key') or ''}", sc["name"])
        i = self.cb.findData(cur)
        if i >= 0:
            self.cb.setCurrentIndex(i)


@register("code")
class Code(GenWidget):
    def build(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.ed = QPlainTextEdit(); self.ed.setObjectName("code"); self.ed.setPlaceholderText(self.spec.get("placeholder", "")); self.ed.setFixedHeight(150)
        self.ed.textChanged.connect(lambda: setattr(self, "_dirty", True)); self._dirty = False
        lay.addWidget(self.ed)
        btns = QHBoxLayout(); ev = QPushButton("▶ EVAL (next phrase)"); ev.setProperty("style", "go"); ev.clicked.connect(lambda: self.ctx.emit(self.spec["action"], self.ed.toPlainText()))
        cl = QPushButton("CLEAR → autonomous"); cl.clicked.connect(lambda: (self.ctx.emit(self.spec["clear_action"]), setattr(self, "_dirty", False)))
        btns.addWidget(ev); btns.addWidget(cl); lay.addLayout(btns)
        self.status = QLabel(""); self.status.setObjectName("status"); self.status.setWordWrap(True); lay.addWidget(self.status)
        if self.spec.get("help"):
            h = QLabel(self.spec["help"]); h.setObjectName("dim"); h.setWordWrap(True); lay.addWidget(h)

    def update_state(self, s):
        sp = self.spec
        if not self.ed.hasFocus() and s.get(sp["key"]) and not self._dirty and self.ed.toPlainText() != s[sp["key"]]:
            self.ed.setPlainText(s[sp["key"]]); self._dirty = False
        slots = s.get(sp.get("slots_key") or "") or []
        if s.get(sp.get("available_key")) is False: tone, text = "err", "Strudel unavailable (pip install mini-racer)"
        elif s.get(sp.get("status_key")): tone, text = "err", "pattern error: " + str(s[sp["status_key"]])
        elif s.get(sp["key"]): tone, text = "ok", f"whole-rack pattern live ({s.get(sp.get('engine_key')) or ''})"
        elif slots: tone, text = "ok", f"slot patterns live: {', '.join(slots)}"
        else: tone, text = "", "autonomous (rule composer)" if s.get("active") else "idle - a pattern set now is applied at start"
        self.status.setProperty("tone", tone); self.status.setText(text); self.status.style().unpolish(self.status); self.status.style().polish(self.status)


@register("phrase_log")
class PhraseLog(GenWidget):
    def build(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.box = QPlainTextEdit(); self.box.setReadOnly(True); self.box.setObjectName("log"); self.box.setMinimumHeight(180); lay.addWidget(self.box)

    def update_state(self, s):
        rows = list(s.get(self.spec["key"]) or [])[-int(self.spec.get("limit", 14)):][::-1]
        lines = []
        for r in rows:
            if r.get("event") != "phrase":
                lines.append(f"{'':5s} {r.get('event', ''):8s} {r.get('style') or r.get('key') or r.get('seed') or r.get('name') or r.get('error') or ''}")
            else:
                lines.append(f"{r.get('bar', ''):>5} {r.get('section', ''):8s} {float(r.get('energy', 0)):.2f}  {' '.join(r.get('chords') or []):18s} · {r.get('key')}  {r.get('lead') or ''}")
        text = "\n".join(lines)
        if text != self.box.toPlainText():
            self.box.setPlainText(text)


@register("timeline_strip")
class TimelineStripWidget(GenWidget):
    """The song strip (the Timeline tab draws the full version; this is
    the same painter inside a spec card)."""

    def build(self):
        from tools.gen.console.plugins.timeline import TimelineStrip
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.strip = TimelineStrip()
        self.strip.setMinimumHeight(int(self.spec.get("height", 120)) + 50)
        self.strip.set_window(float(self.spec.get("window_s", 300)))
        lay.addWidget(self.strip)

    def update_state(self, s):
        self.strip.refresh({"timeline": s.get(self.spec.get("key", "timeline"))})
