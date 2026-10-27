"""Music Studio - PyQt6 front end for the ElevenLabs Music batch generator.

Left: the numbered prompts parsed from a prompts markdown (default
projects/fan/forest_music_prompts.md), with a count of generated takes.
Right: the selected prompt, editable, with the shared house style block;
generation settings; a log; and the takes for that prompt with a player.

The API key is entered in the top bar and remembered between runs (QSettings,
stored per user on this machine). Nothing is sent anywhere but ElevenLabs.

Usage: python tools/music_studio.py [--prompts <file.md>]
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QUrl, QSettings, QFileSystemWatcher, QTimer, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QFont
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QPushButton, QComboBox,
    QSplitter, QMessageBox, QDoubleSpinBox, QSpinBox, QPlainTextEdit,
    QSlider, QCheckBox, QProgressBar, QFileDialog, QGroupBox, QFormLayout,
    QStyle, QToolButton,
)

from tools.media import music_elevenlabs as me

MODELS = ["music_v2_5", "music_v2", "music_v1"]


# ── Background generation ─────────────────────────────────────────────────

class GenWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)          # done, total
    take_done = pyqtSignal(int, str)         # prompt number, mp3 path
    finished_all = pyqtSignal(int, int)      # ok, failed

    def __init__(self, api_key, jobs, model, length_ms, parallel=3, stagger=1.0):
        super().__init__()
        self.api_key = api_key
        self.jobs = jobs        # list of (Prompt, full_prompt, out_path, seed)
        self.model = model
        self.length_ms = length_ms
        self.parallel = max(1, parallel)
        self.stagger = stagger  # seconds between launching requests
        self._stop = False
        self._lock = threading.Lock()
        self._ok = self._failed = 0

    def stop(self):
        self._stop = True

    def _one(self, idx, job):
        """Runs on a pool thread. Returns (idx, True | False | None=skipped)."""
        p, full, out_path, seed = job
        if self._stop:
            return idx, None
        tag = out_path.name
        self.log.emit(f"[{idx}] start {tag}")
        t0 = time.time()
        good = me.compose(self.api_key, full, self.length_ms, self.model,
                          seed, out_path,
                          log=lambda m: self.log.emit(f"[{idx}] {m.strip()}"),
                          should_stop=lambda: self._stop)
        if good:
            me.write_sidecar(out_path, p, self.model, self.length_ms, seed, full)
            mb = out_path.stat().st_size / 1e6
            self.log.emit(f"[{idx}] saved {tag}  {mb:.1f} MB in {time.time() - t0:.0f}s")
            self.take_done.emit(p.number, str(out_path))
            with self._lock:
                self._ok += 1
            return idx, True
        if self._stop:
            return idx, None          # cancelled mid-request, not a failure
        with self._lock:
            self._failed += 1
            first_failure_before_any_success = self._ok == 0
        if first_failure_before_any_success:
            # Almost certainly key/plan. Flip the flag here, on the pool
            # thread, so queued jobs see it before they start.
            self.log.emit("A request failed before any succeeded; stopping the batch.")
            self._stop = True
        return idx, False

    def run(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        done = 0
        total = len(self.jobs)
        futures = []
        with ThreadPoolExecutor(max_workers=self.parallel) as pool:
            for i, job in enumerate(self.jobs, 1):
                if self._stop:
                    break
                futures.append(pool.submit(self._one, i, job))
                # Stagger launches a little so a burst doesn't trip the rate limit.
                if i < total and self.stagger > 0 and i <= self.parallel:
                    time.sleep(self.stagger)
            for fut in as_completed(futures):
                _, good = fut.result()
                if good is None:          # skipped or cancelled
                    continue
                done += 1
                self.progress.emit(done, total)
        self.finished_all.emit(self._ok, self._failed)


# ── Main window ───────────────────────────────────────────────────────────

class MusicStudio(QMainWindow):
    def __init__(self, prompts_path: Path):
        super().__init__()
        self.setWindowTitle("Music Studio - ElevenLabs")
        self.resize(1320, 860)
        self.settings = QSettings("GL_Simple", "MusicStudio")

        self.prompts_path = prompts_path
        self.house = ""
        self.prompts = []
        self.worker = None
        self._dirty = False

        self.player = QMediaPlayer()
        self.audio = QAudioOutput()
        self.player.setAudioOutput(self.audio)
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(self._on_duration)
        self.player.playbackStateChanged.connect(self._on_play_state)
        self.player.errorOccurred.connect(
            lambda err, msg: self._log(f"player error: {msg}"))

        # Auto-reload when the markdown changes on disk. Editors usually
        # write a new file over the old one, so the path is re-added after
        # every change; a short debounce absorbs multi-step saves.
        self.watcher = QFileSystemWatcher()
        self.watcher.fileChanged.connect(self._on_file_changed)
        self._reload_timer = QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.setInterval(400)
        self._reload_timer.timeout.connect(self._auto_reload)

        self._build_ui()
        self._restore_settings()
        self.load_prompts()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)

        # Top bar: key + paths
        top = QGroupBox("Connection and files")
        form = QFormLayout(top)

        key_row = QHBoxLayout()
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("ElevenLabs API key (xi-api-key)")
        self.key_show = QToolButton()
        self.key_show.setText("show")
        self.key_show.setCheckable(True)
        self.key_show.toggled.connect(
            lambda on: self.key_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password))
        self.key_edit.editingFinished.connect(self._save_settings)
        key_row.addWidget(self.key_edit, 1)
        key_row.addWidget(self.key_show)
        form.addRow("API key", key_row)

        p_row = QHBoxLayout()
        self.prompts_edit = QLineEdit(str(self.prompts_path))
        self.prompts_edit.setReadOnly(True)
        b = QPushButton("Browse")
        b.clicked.connect(self._pick_prompts)
        r = QPushButton("Reload now")
        r.setToolTip("Re-read the prompts file. It also reloads by itself whenever the file changes on disk.")
        r.clicked.connect(self.load_prompts)
        p_row.addWidget(self.prompts_edit, 1)
        p_row.addWidget(b)
        p_row.addWidget(r)
        form.addRow("Prompts file", p_row)

        o_row = QHBoxLayout()
        self.out_edit = QLineEdit(str(me.default_out_dir(self.prompts_path)))
        self.out_edit.editingFinished.connect(self._on_out_changed)
        ob = QPushButton("Browse")
        ob.clicked.connect(self._pick_out)
        of = QPushButton("Open folder")
        of.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(self.out_dir()))))
        o_row.addWidget(self.out_edit, 1)
        o_row.addWidget(ob)
        o_row.addWidget(of)
        form.addRow("Output folder", o_row)
        outer.addWidget(top)

        split = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(split, 1)

        # Left: prompt list
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("Prompts"))
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._on_select)
        ll.addWidget(self.list, 1)
        split.addWidget(left)

        # Right: editor + settings + takes
        right = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(right)
        split.setSizes([300, 1020])

        # Editor
        ed = QWidget()
        el = QVBoxLayout(ed)
        el.setContentsMargins(0, 0, 0, 0)
        hs_row = QHBoxLayout()
        self.house_chk = QCheckBox("Prepend house style to every prompt")
        self.house_chk.setChecked(True)
        self.house_chk.toggled.connect(self._update_counts)
        self.house_toggle = QToolButton()
        self.house_toggle.setText("edit house style")
        self.house_toggle.setCheckable(True)
        self.house_toggle.toggled.connect(lambda on: self.house_edit.setVisible(on))
        hs_row.addWidget(self.house_chk)
        hs_row.addWidget(self.house_toggle)
        hs_row.addStretch(1)
        el.addLayout(hs_row)
        self.house_edit = QPlainTextEdit()
        self.house_edit.setMaximumHeight(120)
        self.house_edit.setVisible(False)
        self.house_edit.textChanged.connect(self._on_house_edited)
        el.addWidget(self.house_edit)

        self.title_lbl = QLabel("")
        f = QFont()
        f.setBold(True)
        f.setPointSize(11)
        self.title_lbl.setFont(f)
        el.addWidget(self.title_lbl)
        self.body_edit = QPlainTextEdit()
        self.body_edit.textChanged.connect(self._on_body_edited)
        el.addWidget(self.body_edit, 1)
        cnt_row = QHBoxLayout()
        self.count_lbl = QLabel("")
        self.save_btn = QPushButton("Save edits to markdown")
        self.save_btn.clicked.connect(self.save_prompts)
        self.save_btn.setEnabled(False)
        cnt_row.addWidget(self.count_lbl, 1)
        cnt_row.addWidget(self.save_btn)
        el.addLayout(cnt_row)
        right.addWidget(ed)

        # Settings + generate + log
        gen = QWidget()
        gl = QVBoxLayout(gen)
        gl.setContentsMargins(0, 0, 0, 0)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Length (s)"))
        self.len_spin = QDoubleSpinBox()
        self.len_spin.setRange(me.MIN_LENGTH_S, me.MAX_LENGTH_S)
        self.len_spin.setDecimals(0)
        self.len_spin.setValue(me.DEFAULT_LENGTH_S)
        srow.addWidget(self.len_spin)
        srow.addWidget(QLabel("Takes"))
        self.takes_spin = QSpinBox()
        self.takes_spin.setRange(1, 10)
        srow.addWidget(self.takes_spin)
        srow.addWidget(QLabel("Model"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODELS)
        srow.addWidget(self.model_combo)
        srow.addWidget(QLabel("Seed"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 2_000_000_000)
        self.seed_spin.setSpecialValueText("random")
        self.seed_spin.setValue(-1)
        srow.addWidget(self.seed_spin)
        srow.addWidget(QLabel("Parallel"))
        self.par_spin = QSpinBox()
        self.par_spin.setRange(1, 8)
        self.par_spin.setValue(2)
        self.par_spin.setToolTip("Requests in flight at once. ElevenLabs caps this per plan (2 on this "
                                 "account). Above the cap, requests wait for a free slot and retry.")
        srow.addWidget(self.par_spin)
        srow.addStretch(1)
        gl.addLayout(srow)

        brow = QHBoxLayout()
        self.gen_all_btn = QPushButton("GENERATE ALL PROMPTS")
        self.gen_all_btn.setMinimumHeight(44)
        bf = QFont()
        bf.setBold(True)
        bf.setPointSize(11)
        self.gen_all_btn.setFont(bf)
        self.gen_all_btn.setToolTip("Generate every prompt in the list, with the takes count above, "
                                    "appending to any takes that already exist")
        self.gen_all_btn.clicked.connect(self._generate_all)
        self.gen_btn = QPushButton("Generate this prompt only")
        self.gen_btn.setMinimumHeight(44)
        self.gen_btn.clicked.connect(self._generate_selected)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumHeight(44)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        brow.addWidget(self.gen_all_btn, 2)
        brow.addWidget(self.gen_btn, 1)
        brow.addWidget(self.cancel_btn)
        gl.addLayout(brow)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        gl.addWidget(self.progress)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(2000)
        gl.addWidget(self.log_edit, 1)
        right.addWidget(gen)

        # Takes + player
        tk = QWidget()
        tl = QVBoxLayout(tk)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(QLabel("Takes for this prompt"))
        self.takes_list = QListWidget()
        self.takes_list.itemDoubleClicked.connect(lambda _: self._play())
        self.takes_list.currentRowChanged.connect(self._on_take_select)
        tl.addWidget(self.takes_list, 1)
        prow = QHBoxLayout()
        st = self.style()
        self.play_btn = QPushButton(st.standardIcon(QStyle.StandardPixmap.SP_MediaPlay), "Play")
        self.play_btn.clicked.connect(self._play)
        self.stop_btn = QPushButton(st.standardIcon(QStyle.StandardPixmap.SP_MediaStop), "Stop")
        self.stop_btn.clicked.connect(self.player.stop)
        self.pos_slider = QSlider(Qt.Orientation.Horizontal)
        self.pos_slider.sliderReleased.connect(
            lambda: self.player.setPosition(self.pos_slider.value()))
        self.time_lbl = QLabel("0:00 / 0:00")
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(80)
        self.vol_slider.setMaximumWidth(120)
        self.vol_slider.valueChanged.connect(lambda v: self.audio.setVolume(v / 100))
        self.audio.setVolume(0.8)
        self.del_btn = QPushButton("Delete take")
        self.del_btn.clicked.connect(self._delete_take)
        self.show_btn = QPushButton("Show prompt used")
        self.show_btn.clicked.connect(self._show_sidecar)
        prow.addWidget(self.play_btn)
        prow.addWidget(self.stop_btn)
        prow.addWidget(self.pos_slider, 1)
        prow.addWidget(self.time_lbl)
        prow.addWidget(QLabel("Vol"))
        prow.addWidget(self.vol_slider)
        prow.addWidget(self.show_btn)
        prow.addWidget(self.del_btn)
        tl.addLayout(prow)
        right.addWidget(tk)
        right.setSizes([380, 230, 220])

    # ── Settings ──────────────────────────────────────────────────────────

    def _restore_settings(self):
        s = self.settings
        self.key_edit.setText(s.value("api_key", "", str))
        out = s.value("out_dir", "", str)
        if out:
            self.out_edit.setText(out)
        self.takes_spin.setValue(int(s.value("takes", 1)))
        m = s.value("model", me.DEFAULT_MODEL, str)
        if m in MODELS:
            self.model_combo.setCurrentText(m)
        self.house_chk.setChecked(s.value("house", True, bool))
        self.vol_slider.setValue(int(s.value("volume", 80)))
        self.par_spin.setValue(int(s.value("parallel", 2)))

    def _save_settings(self):
        s = self.settings
        s.setValue("api_key", self.key_edit.text().strip())
        s.setValue("out_dir", self.out_edit.text().strip())
        s.setValue("takes", self.takes_spin.value())
        s.setValue("model", self.model_combo.currentText())
        s.setValue("house", self.house_chk.isChecked())
        s.setValue("volume", self.vol_slider.value())
        s.setValue("parallel", self.par_spin.value())
        s.setValue("prompts_path", str(self.prompts_path))

    def closeEvent(self, ev):
        if self.worker and self.worker.isRunning():
            if QMessageBox.question(self, "Generation running",
                                    "A batch is still running. Cancel it and quit?") \
                    != QMessageBox.StandardButton.Yes:
                ev.ignore()
                return
            self.worker.stop()
            self.worker.wait(5000)
        if self._dirty and QMessageBox.question(
                self, "Unsaved edits", "Save prompt edits to the markdown file?") \
                == QMessageBox.StandardButton.Yes:
            self.save_prompts()
        self.player.stop()
        self._save_settings()
        ev.accept()

    # ── Prompts ───────────────────────────────────────────────────────────

    def out_dir(self) -> Path:
        return Path(self.out_edit.text().strip() or me.default_out_dir(self.prompts_path))

    def _on_out_changed(self):
        self._save_settings()
        self._refresh_list()
        self._on_select(self.list.currentRow())

    def _pick_prompts(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Prompts markdown",
                                            str(self.prompts_path.parent), "Markdown (*.md)")
        if fn:
            self.prompts_path = Path(fn)
            self.prompts_edit.setText(fn)
            self.out_edit.setText(str(me.default_out_dir(self.prompts_path)))
            self._save_settings()
            self.load_prompts()

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", str(self.out_dir()))
        if d:
            self.out_edit.setText(d)
            self._on_out_changed()

    def _watch_prompts_file(self):
        for f in self.watcher.files():
            self.watcher.removePath(f)
        if self.prompts_path.exists():
            self.watcher.addPath(str(self.prompts_path))

    def _on_file_changed(self, _path):
        self._reload_timer.start()

    def _auto_reload(self):
        self._watch_prompts_file()          # re-arm; the editor may have replaced the file
        if time.time() - getattr(self, "_self_saved_at", 0) < 2.0:
            return                          # our own save, nothing new to read
        if not self.prompts_path.exists():
            self._log(f"{self.prompts_path.name} disappeared; keeping the loaded prompts")
            return
        if self._dirty:
            if QMessageBox.question(
                    self, "File changed on disk",
                    f"{self.prompts_path.name} changed on disk and you have unsaved edits here.\n"
                    "Reload from disk and discard your edits?") != QMessageBox.StandardButton.Yes:
                return
            self._dirty = False
        self.load_prompts(auto=True)

    def load_prompts(self, auto=False):
        if self._dirty and QMessageBox.question(
                self, "Unsaved edits", "Discard unsaved prompt edits and reload?") \
                != QMessageBox.StandardButton.Yes:
            return
        if not self.prompts_path.exists():
            QMessageBox.critical(self, "Missing", f"Prompts file not found:\n{self.prompts_path}")
            return
        try:
            self.house, self.prompts = me.parse_prompts(self.prompts_path)
        except SystemExit:
            QMessageBox.critical(self, "Parse error",
                                 "No '## N. Title' prompt sections found in the file.")
            return
        keep = self.current_prompt().number if self.current_prompt() else None
        self._loading = True
        self.house_edit.setPlainText(self.house)
        self._loading = False
        self._dirty = False
        self.save_btn.setEnabled(False)
        self._refresh_list()
        row = next((i for i, p in enumerate(self.prompts) if p.number == keep), 0)
        if self.prompts:
            self.list.setCurrentRow(row)
            self._on_select(row)            # refresh editor even if the row didn't change
        self._watch_prompts_file()
        self._log(f"{'file changed, re' if auto else ''}loaded {len(self.prompts)} prompts "
                  f"from {self.prompts_path.name}")

    def _refresh_list(self):
        row = self.list.currentRow()
        self.list.blockSignals(True)
        self.list.clear()
        for p in self.prompts:
            n = len(me.find_takes(self.out_dir(), p))
            mark = f"  [{n} take{'s' if n != 1 else ''}]" if n else ""
            it = QListWidgetItem(f"{p.number:2d}. {p.title}{mark}")
            it.setToolTip(p.states)
            self.list.addItem(it)
        self.list.blockSignals(False)
        if 0 <= row < self.list.count():
            self.list.setCurrentRow(row)

    def current_prompt(self):
        r = self.list.currentRow()
        return self.prompts[r] if 0 <= r < len(self.prompts) else None

    def _on_select(self, row):
        p = self.current_prompt()
        if p is None:
            return
        self._loading = True
        self.title_lbl.setText(f"{p.number}. {p.title}    ({p.states})")
        self.body_edit.setPlainText(p.body)
        self._loading = False
        self._update_counts()
        self._refresh_takes()

    def _on_body_edited(self):
        if getattr(self, "_loading", False):
            return
        p = self.current_prompt()
        if p is None:
            return
        p.body = self.body_edit.toPlainText()
        self._mark_dirty()
        self._update_counts()

    def _on_house_edited(self):
        if getattr(self, "_loading", False):
            return
        self.house = self.house_edit.toPlainText()
        self._mark_dirty()
        self._update_counts()

    def _mark_dirty(self):
        self._dirty = True
        self.save_btn.setEnabled(True)

    def _effective_house(self) -> str:
        return self.house if self.house_chk.isChecked() else ""

    def _update_counts(self):
        p = self.current_prompt()
        if p is None:
            return
        n = len(me.build_prompt(self._effective_house(), p))
        if n > me.PROMPT_MAX_CHARS:
            self.count_lbl.setText(f"{n} characters: OVER the {me.PROMPT_MAX_CHARS} limit, "
                                   f"shorten by {n - me.PROMPT_MAX_CHARS} before generating")
            self.count_lbl.setStyleSheet("color: #d03030; font-weight: bold")
        elif n > me.PROMPT_WARN_CHARS:
            self.count_lbl.setText(f"{n} characters will be sent (limit {me.PROMPT_MAX_CHARS})")
            self.count_lbl.setStyleSheet("color: #c08000")
        else:
            self.count_lbl.setText(f"{n} characters will be sent (limit {me.PROMPT_MAX_CHARS})")
            self.count_lbl.setStyleSheet("")

    def save_prompts(self):
        try:
            self._self_saved_at = time.time()
            me.save_prompts(self.prompts_path, self.house, self.prompts)
        except OSError as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return
        self._dirty = False
        self.save_btn.setEnabled(False)
        self._log(f"saved edits to {self.prompts_path.name}")

    # ── Generation ────────────────────────────────────────────────────────

    def _jobs_for(self, prompts):
        out = self.out_dir()
        out.mkdir(parents=True, exist_ok=True)
        takes = self.takes_spin.value()
        seed0 = self.seed_spin.value()
        jobs = []
        house = self._effective_house()
        for p in prompts:
            full = me.build_prompt(house, p)
            if len(full) > me.PROMPT_MAX_CHARS:
                self._log(f"skipping {p.number}. {p.title}: {len(full)} chars is over the "
                          f"{me.PROMPT_MAX_CHARS} limit")
                continue
            start = me.next_take_number(out, p)
            for k in range(takes):
                seed = None if seed0 < 0 else seed0 + k
                jobs.append((p, full, me.take_path(out, p, start + k), seed))
        return jobs

    def _start(self, jobs):
        key = self.key_edit.text().strip()
        if not key:
            QMessageBox.warning(self, "No API key", "Enter your ElevenLabs API key first.")
            return
        if self.worker and self.worker.isRunning():
            return
        if not jobs:
            QMessageBox.warning(self, "Nothing to generate",
                                "Every selected prompt is over the character limit. See the log.")
            return
        self._save_settings()
        length_ms = int(round(self.len_spin.value() * 1000))
        mins = len(jobs) * self.len_spin.value() / 60
        self._log(f"starting {len(jobs)} track(s), {self.len_spin.value():.0f}s each "
                  f"({mins:.0f} min of audio), model {self.model_combo.currentText()}, "
                  f"{self.par_spin.value()} at a time")
        self.worker = GenWorker(key, jobs, self.model_combo.currentText(), length_ms,
                                parallel=self.par_spin.value())
        self.worker.log.connect(self._log)
        self.worker.progress.connect(self._on_progress)
        self.worker.take_done.connect(self._on_take_done)
        self.worker.finished_all.connect(self._on_finished)
        self.progress.setRange(0, len(jobs))
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self._set_busy(True)
        self.worker.start()

    def _generate_selected(self):
        p = self.current_prompt()
        if p:
            self._start(self._jobs_for([p]))

    def _generate_all(self):
        self._start(self._jobs_for(self.prompts))

    def _cancel(self):
        if self.worker:
            self.worker.stop()
            self._log("cancelling: requests already in flight finish, nothing new starts...")
            self.cancel_btn.setEnabled(False)

    def _set_busy(self, busy):
        self.gen_btn.setEnabled(not busy)
        self.gen_all_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)

    def _on_progress(self, done, total):
        self.progress.setValue(done)

    def _on_take_done(self, number, path):
        self._refresh_list()
        p = self.current_prompt()
        if p and p.number == number:
            self._refresh_takes()
            self.takes_list.setCurrentRow(self.takes_list.count() - 1)

    def _on_finished(self, ok, failed):
        self._set_busy(False)
        self.progress.setVisible(False)
        self._log(f"batch finished: {ok} ok, {failed} failed")
        self._refresh_list()

    def _log(self, msg):
        self.log_edit.appendPlainText(msg)

    # ── Takes / player ────────────────────────────────────────────────────

    def _refresh_takes(self):
        p = self.current_prompt()
        self.takes_list.clear()
        if p is None:
            return
        for f in me.find_takes(self.out_dir(), p):
            mb = f.stat().st_size / 1e6
            it = QListWidgetItem(f"{f.name}   ({mb:.1f} MB)")
            it.setData(Qt.ItemDataRole.UserRole, str(f))
            self.takes_list.addItem(it)
        has = self.takes_list.count() > 0
        for b in (self.play_btn, self.del_btn, self.show_btn):
            b.setEnabled(has)
        if has:
            self.takes_list.setCurrentRow(0)

    def _current_take(self):
        it = self.takes_list.currentItem()
        return Path(it.data(Qt.ItemDataRole.UserRole)) if it else None

    def _on_take_select(self, row):
        # Selecting a different take stops whatever is playing.
        f = self._current_take()
        if f is None:
            return
        cur = self.player.source().toLocalFile()
        if cur and Path(cur) != f:
            self.player.stop()

    def _play(self):
        f = self._current_take()
        if f is None:
            return
        cur = self.player.source().toLocalFile()
        if cur and Path(cur) == f:
            if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self.player.pause()
            else:
                self.player.play()
            return
        self.player.setSource(QUrl.fromLocalFile(str(f)))
        self.player.play()

    def _on_play_state(self, state):
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        icon = QStyle.StandardPixmap.SP_MediaPause if playing else QStyle.StandardPixmap.SP_MediaPlay
        self.play_btn.setIcon(self.style().standardIcon(icon))
        self.play_btn.setText("Pause" if playing else "Play")

    def _on_duration(self, d):
        self.pos_slider.setRange(0, max(0, d))
        self._on_position(self.player.position())

    def _on_position(self, pos):
        if not self.pos_slider.isSliderDown():
            self.pos_slider.setValue(pos)
        self.time_lbl.setText(f"{_mmss(pos)} / {_mmss(self.player.duration())}")

    def _delete_take(self):
        f = self._current_take()
        if f is None:
            return
        if QMessageBox.question(self, "Delete take", f"Delete {f.name}?") \
                != QMessageBox.StandardButton.Yes:
            return
        if Path(self.player.source().toLocalFile() or "") == f:
            self.player.stop()
            self.player.setSource(QUrl())
        try:
            f.unlink()
            side = f.with_suffix(".txt")
            if side.exists():
                side.unlink()
        except OSError as e:
            QMessageBox.critical(self, "Delete failed", str(e))
            return
        self._log(f"deleted {f.name}")
        self._refresh_takes()
        self._refresh_list()

    def _show_sidecar(self):
        f = self._current_take()
        if f is None:
            return
        side = f.with_suffix(".txt")
        if not side.exists():
            QMessageBox.information(self, "No record", "No prompt record was saved for this take.")
            return
        box = QMessageBox(self)
        box.setWindowTitle(side.name)
        box.setText(side.read_text(encoding="utf-8"))
        box.exec()


def _mmss(ms: int) -> str:
    s = max(0, int(ms // 1000))
    return f"{s // 60}:{s % 60:02d}"


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Music Studio - ElevenLabs music generator GUI")
    ap.add_argument("--prompts", type=Path, help="prompts markdown file")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    path = args.prompts
    if path is None:
        saved = QSettings("GL_Simple", "MusicStudio").value("prompts_path", "", str)
        if saved and Path(saved).exists():
            path = Path(saved)
    if path is None:
        path = me.DEFAULT_PROMPTS
    w = MusicStudio(path.resolve())
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
