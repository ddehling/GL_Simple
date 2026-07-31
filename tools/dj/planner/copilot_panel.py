"""Qt panel for the Set Copilot (backend: tools/dj/planner/copilot.py).

Lives in the Set tab's right column. Each turn runs on a QThread; tool
activity streams into the chat view; the copilot's edits are pushed to the
Set tab through entriesApplied and are one-click revertible.
"""
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                             QPushButton, QVBoxLayout, QWidget)

from tools.dj.planner.copilot import SetCopilot


class TurnWorker(QThread):
    event = pyqtSignal(str, object)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, copilot, text):
        super().__init__()
        self.copilot, self.text = copilot, text

    def run(self):
        try:
            reply = self.copilot.run_turn(
                self.text, on_event=lambda k, p: self.event.emit(k, p))
            self.done.emit(reply)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class CopilotPanel(QWidget):
    # (entries, theme_name) the Set tab should adopt after a turn.
    entriesApplied = pyqtSignal(object, str)

    def __init__(self, planner, set_tab):
        super().__init__()
        self.planner = planner
        self.set_tab = set_tab
        self.copilot = None
        self._worker = None
        self._revert = None          # (entries, theme) before the last turn

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 6, 0, 0)
        head = QHBoxLayout()
        head.addWidget(QLabel("SET COPILOT"))
        head.addStretch(1)
        self.revert_btn = QPushButton("↩ Revert last change")
        self.revert_btn.setEnabled(False)
        self.revert_btn.clicked.connect(self._do_revert)
        head.addWidget(self.revert_btn)
        v.addLayout(head)
        self.chat = QPlainTextEdit()
        self.chat.setReadOnly(True)
        self.chat.setPlaceholderText(
            'Talk to the copilot: "build me a 90 minute set that starts '
            'dreamy and peaks around an hour in", "why is seam 4 a fade?", '
            '"swap slot 2 for something more hypnotic"...')
        v.addWidget(self.chat, 1)
        # API-key entry, shown only when no credential resolves. Paste once
        # and it's saved (user-level, out of the repo); no env-var juggling.
        self.key_row = QWidget()
        kr = QHBoxLayout(self.key_row)
        kr.setContentsMargins(0, 0, 0, 0)
        kr.addWidget(QLabel("Anthropic API key:"))
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("sk-ant-...")
        self.key_edit.returnPressed.connect(self._save_key)
        kr.addWidget(self.key_edit, 1)
        kb = QPushButton("Save")
        kb.clicked.connect(self._save_key)
        kr.addWidget(kb)
        self.key_row.hide()
        v.addWidget(self.key_row)
        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.returnPressed.connect(self._send)
        row.addWidget(self.input, 1)
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._send)
        row.addWidget(self.send_btn)
        v.addLayout(row)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        v.addWidget(self.status)

    def _ensure_copilot(self):
        if self.copilot is None:
            bp, wl = None, None
            try:                     # optional Beatport discovery tools
                from lib.dj import beatport as BP
                bp = BP.BeatportClient()
                wl = BP.Wishlist(self.planner.music_dir)
            except Exception:
                pass
            self.copilot = SetCopilot(
                self.planner.library,
                theme_name=self.set_tab.theme_combo.currentText(),
                pair_memory=getattr(self.planner, "pair_memory", None),
                beatport=bp, wishlist=wl,
                night_verdicts=getattr(self.planner, "night_verdicts", None),
                last_played=getattr(self.planner, "last_played", None))
        else:
            # Re-sync the working library so do-not-use flags (and re-tags)
            # applied since the copilot was built take effect immediately.
            self.copilot.set_library(self.planner.library)
            # ...and the night evidence (fresh after a Nights-tab reload).
            self.copilot.night_verdicts = dict(
                getattr(self.planner, "night_verdicts", None) or {})
            self.copilot.last_played = dict(
                getattr(self.planner, "last_played", None) or {})
        if not self.copilot.available():
            self.status.setText("Copilot needs an API key: "
                                + self.copilot.why_unavailable())
            # Offer the inline key field unless the blocker is the package.
            self.key_row.setVisible("package" not in
                                    self.copilot.why_unavailable())
            return False
        self.key_row.hide()
        return True

    def _save_key(self):
        key = self.key_edit.text().strip()
        if not key:
            return
        if self.copilot is None:
            self._ensure_copilot()
        if self.copilot and self.copilot.set_api_key(key):
            self.key_edit.clear()
            self.key_row.hide()
            self.status.setText("API key saved - copilot ready.")
        else:
            self.status.setText("That key didn't resolve; check it and "
                                "try again.")

    def _append(self, text):
        self.chat.appendPlainText(text)
        sb = self.chat.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _send(self):
        text = self.input.text().strip()
        if not text or (self._worker and self._worker.isRunning()):
            return
        if not self._ensure_copilot():
            return
        # The human may have edited the set since the last turn - the UI
        # is the truth, the copilot adopts it.
        self.copilot.sync(self.set_tab.entries,
                          self.set_tab.theme_combo.currentText())
        self._revert = ([dict(e) for e in self.set_tab.entries],
                        self.set_tab.theme_combo.currentText())
        self.input.clear()
        self.input.setEnabled(False)
        self.send_btn.setEnabled(False)
        self._append(f"You: {text}")
        self.status.setText("thinking...")
        self._worker = TurnWorker(self.copilot, text)
        self._worker.event.connect(self._on_event)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_event(self, kind, payload):
        if kind == "tool":
            args = ", ".join(f"{k}={v}" for k, v in
                             (payload.get("input") or {}).items())
            self._append(f"   [{payload['name']}({args[:80]})]")
            self.status.setText(f"running {payload['name']}...")

    def _first_turn_banner(self):
        mode = self.copilot.auth_mode() if self.copilot else None
        if mode == "cli" and not getattr(self, "_bannered", False):
            self._bannered = True
            self._append("[using your Claude Code session - no API key "
                         "needed]\n")

    def _on_done(self, reply):
        self._first_turn_banner()
        self._append(f"Copilot: {reply}\n")
        self.input.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.status.setText("")
        changed = (self.copilot.entries != self._revert[0]
                   or self.copilot.theme_name != self._revert[1])
        if changed:
            self.revert_btn.setEnabled(True)
            self.entriesApplied.emit(list(self.copilot.entries),
                                     self.copilot.theme_name)
        # Deferred UI actions (save / push-to-live): the tool loop runs on
        # a worker thread where neither the DB nor the show belong; the
        # tools queued the intent, and HERE - on the GUI thread, after the
        # Set tab adopted the entries - is where it happens.
        actions, self.copilot.pending_ui = self.copilot.pending_ui, []
        for kind, args in actions:
            try:
                if kind == "save":
                    self._do_save(args.get("name"))
                    self._append("[set saved]\n")
                elif kind == "push":
                    self.set_tab._push_live(args.get("mode", "order"))
                    self._append(f"[{self.set_tab.status.text()}]\n")
            except Exception as e:
                self._append(f"[error] {kind}: {type(e).__name__}: {e}\n")
        self.input.setFocus()

    def _do_save(self, name):
        """Save via the Set tab's own path (persists theme + compiled
        length + notes). A given name selects-or-creates that setlist;
        without one, the tab's current setlist is used (prompting the
        user if the set is unnamed - the human stays in charge of names)."""
        st = self.set_tab
        if name:
            from lib.dj import setlist as SL
            row = self.planner.db.conn.execute(
                "SELECT id FROM setlists WHERE name = ?", (name,)).fetchone()
            st.setlist_id = row["id"] if row else SL.create_setlist(
                self.planner.db, name,
                theme=st.theme_combo.currentText())
        st.save_set()

    def _on_failed(self, msg):
        self._append(f"[error] {msg}\n")
        self.input.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.status.setText("")

    def _do_revert(self):
        if self._revert is None:
            return
        entries, theme = self._revert
        self.revert_btn.setEnabled(False)
        self._append("[reverted the copilot's last change]\n")
        self.entriesApplied.emit([dict(e) for e in entries], theme)
