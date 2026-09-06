"""Night log tab: reads logs/gen_*.jsonl (what the conductor wrote) so a
night can be reviewed after the fact - phrases, gestures, asks, scenes,
movements, errors - without leaving the console."""
import glob
import json
import os

from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget


class NightLog(QWidget):
    def __init__(self, console, log_dir="logs"):
        super().__init__()
        self.console = console
        self.log_dir = log_dir
        lay = QVBoxLayout(self); lay.setContentsMargins(12, 8, 12, 8)
        row = QHBoxLayout(); row.addWidget(QLabel("night")); self.pick = QComboBox(); row.addWidget(self.pick, 1)
        b = QPushButton("reload"); b.clicked.connect(self.reload); row.addWidget(b)
        self.filter = QComboBox(); self.filter.addItems(["everything", "phrases", "operator (gestures, asks, scenes)", "movements + errors"]); row.addWidget(self.filter)
        lay.addLayout(row)
        self.box = QPlainTextEdit(); self.box.setReadOnly(True); self.box.setObjectName("log"); lay.addWidget(self.box, 1)
        self.pick.currentIndexChanged.connect(self.render); self.filter.currentIndexChanged.connect(self.render)
        self.reload()

    def reload(self):
        files = sorted(glob.glob(os.path.join(self.log_dir, "gen_*.jsonl")), reverse=True)
        cur = self.pick.currentData()
        self.pick.blockSignals(True); self.pick.clear()
        for f in files:
            self.pick.addItem(os.path.basename(f)[4:-6], f)
        if cur:
            i = self.pick.findData(cur)
            if i >= 0:
                self.pick.setCurrentIndex(i)
        self.pick.blockSignals(False)
        self.render()

    def render(self):
        path = self.pick.currentData()
        if not path:
            self.box.setPlainText("(no night logs yet)"); return
        mode = self.filter.currentIndex()
        keep = {0: None, 1: {"phrase"}, 2: {"gesture", "ask", "ask_error", "scene_save", "scene_load", "pattern", "slot_pattern", "feedback", "reseed", "style", "section_request", "ramp"},
                3: {"movement", "error", "start", "stop", "end_requested"}}[mode]
        lines = []
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    ev = r.get("event", "")
                    if keep and ev not in keep:
                        continue
                    t = r.get("t", 0)
                    stamp = f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"
                    if ev == "phrase":
                        lines.append(f"{stamp} {r.get('bar', ''):>5} {r.get('section', ''):8s} {float(r.get('energy', 0)):.2f} {' '.join(r.get('chords') or []):18s} {r.get('key', '')}  {r.get('lead') or ''}")
                    else:
                        rest = {k: v for k, v in r.items() if k not in ("event", "t", "snapshot", "intent")}
                        lines.append(f"{stamp} {ev:14s} {json.dumps(rest, default=str)[:160]}")
        except Exception as e:  # noqa: BLE001
            lines = [f"could not read {path}: {e}"]
        self.box.setPlainText("\n".join(lines))

    def refresh(self, state):
        pass


def register(console):
    console.add_tab("Nights", NightLog(console))
