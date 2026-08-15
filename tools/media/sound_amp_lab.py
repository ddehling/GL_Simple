"""Sound Amp Lab — interactive amplification test bench for quiet-but-peaky
sounds (e.g. the weight_of_light thunder files: too faint overall, but with
transient peaks that clip audibly under plain linear gain).

Three tabs:

* "Stack" (default) - one toggle per method, one intensity slider under each.
  Enabled modules process simultaneously as a chain, top to bottom
  (Gain -> Compressor -> Parallel comp -> Soft clip -> Limiter); any toggle
  or slider change re-renders the whole stack live. The compressor modules
  are peak-restored, so their sliders add density without changing level -
  loudness comes from the Gain/Limiter sliders and the pieces compose
  predictably.

* "Presets" - set ONE "Boost" amount and every preset chain in the
  table is rendered at once. The table shows peak / RMS / clipped-sample stats
for each, so the clippers expose themselves without listening. Click a row to
see its waveform + spectrogram against the input; double-click to hear it.
Save writes the selected result at the SAME sample rate and bit depth
(libsndfile subtype) as the original file — the input is never modified.

Playback is live: hit "▶ Output" (Loop is on by default) and keep tweaking —
every slider/toggle/preset change re-renders and splices the new audio in at
the current playhead, so you HEAR the filtering change while the sound plays.

Preset chains (all loudness-matched by the shared Boost knob):
    Linear gain          - the baseline that clips
    Peak normalize       - the most clean linear gain possible
    Limiter              - pre-gain into a look-ahead brick wall
    Comp gentle -> Lim   - 3:1 slow compressor, makeup, safety limiter
    Comp punchy -> Lim   - 6:1 fast compressor, makeup, safety limiter
    Parallel comp -> Lim - 50% wet crushed compressor (keeps transients)
    Soft clip tanh/cubic - waveshapers: linear when quiet, round the peaks
    Hard clip            - "just turn it up", for comparison
    RMS -16 -> Lim       - loudness-normalize then limit (boost-independent)

* "Custom" - the full per-method parameter panel for fine-tuning a winning
  approach.

Usage:
    python tools/media/sound_amp_lab.py [path/to/sound.wav]
"""

import argparse
import sys
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy import signal
from scipy.ndimage import minimum_filter1d

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
import pyqtgraph as pg

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:      # slow but functional fallback
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f

pg.setConfigOptions(imageAxisOrder='row-major', background='k', foreground='w')

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO_ROOT / 'projects' / 'weight_of_light' / 'media' / 'sounds' / 'thunder'

EPS = 1e-12
SPEC_FLOOR_DB = -100.0


# ---------------------------------------------------------------------------
# DSP primitives
# ---------------------------------------------------------------------------

@njit(cache=True)
def _smooth_asym(x, up_coef, down_coef):
    """One-pole smoother with separate coefficients for rising vs falling
    input. Coefficients are exp(-1/(tau*sr)) - closer to 1 = slower."""
    out = np.empty_like(x)
    s = x[0]
    for i in range(x.size):
        v = x[i]
        c = up_coef if v > s else down_coef
        s = c * s + (1.0 - c) * v
        out[i] = s
    return out


def _warmup_numba():
    if HAVE_NUMBA:
        _smooth_asym(np.zeros(4), 0.5, 0.5)


def _coef(tau_ms, sr):
    tau = max(tau_ms, 1e-3) * 1e-3
    return float(np.exp(-1.0 / (tau * sr)))


def _db_to_lin(db):
    return 10.0 ** (db / 20.0)


def _detector(data):
    """Per-sample level: max of |x| across channels."""
    return np.max(np.abs(data), axis=1)


def amp_linear(data, sr, p):
    return data * _db_to_lin(p['gain_db'])


def amp_normalize_peak(data, sr, p):
    peak = np.max(np.abs(data))
    if peak < EPS:
        return data.copy()
    return data * (_db_to_lin(p['target_db']) / peak)


def amp_normalize_rms(data, sr, p):
    rms = np.sqrt(np.mean(data ** 2))
    if rms < EPS:
        return data.copy()
    return data * (_db_to_lin(p['target_rms_db']) / rms)


def amp_hard_clip(data, sr, p):
    ceiling = _db_to_lin(p['ceiling_db'])
    return np.clip(data * _db_to_lin(p['pre_gain_db']), -ceiling, ceiling)


def amp_soft_clip_tanh(data, sr, p):
    # Slope at zero equals the drive gain, so quiet material is amplified
    # linearly while peaks saturate toward +/-1.
    g = _db_to_lin(p['drive_db'])
    return np.tanh(data * g) * _db_to_lin(p['trim_db'])


def amp_soft_clip_cubic(data, sr, p):
    g = _db_to_lin(p['drive_db'])
    x = np.clip(data * g, -1.0, 1.0)
    y = 1.5 * x - 0.5 * x ** 3
    return y * _db_to_lin(p['trim_db'])


def amp_compressor(data, sr, p):
    thresh = p['threshold_db']
    ratio = max(p['ratio'], 1.0)
    knee = max(p['knee_db'], 1e-6)
    mix = np.clip(p['mix_pct'], 0.0, 100.0) / 100.0

    lvl_db = 20.0 * np.log10(_detector(data) + EPS)
    lvl_db = _smooth_asym(lvl_db, _coef(p['attack_ms'], sr),
                          _coef(p['release_ms'], sr))

    over = lvl_db - thresh
    slope = 1.0 / ratio - 1.0
    gain_db = np.where(
        over <= -knee / 2.0, 0.0,
        np.where(over >= knee / 2.0, slope * over,
                 slope * (over + knee / 2.0) ** 2 / (2.0 * knee)))
    gain = _db_to_lin(gain_db + p['makeup_db'])
    wet = data * gain[:, None]
    return mix * wet + (1.0 - mix) * data


def amp_limiter(data, sr, p):
    x = data * _db_to_lin(p['pre_gain_db'])
    ceiling = _db_to_lin(p['ceiling_db'])
    det = _detector(x)
    g = np.minimum(1.0, ceiling / np.maximum(det, EPS))

    look = max(int(p['lookahead_ms'] * 1e-3 * sr), 1)
    if look % 2 == 0:
        look += 1
    # Future-looking running minimum: gmin[i] = min(g[i : i+look]), so the
    # gain starts ducking BEFORE the peak arrives (zero added latency).
    gmin = minimum_filter1d(g, size=look, mode='nearest', origin=-(look // 2))
    # Fast attack (over ~half the lookahead), slow release.
    gsm = _smooth_asym(gmin, _coef(p['release_ms'], sr),
                       _coef(p['lookahead_ms'] * 0.5, sr))
    return x * gsm[:, None]


def run_chain(data, sr, chain):
    for func, params in chain:
        data = func(data, sr, params)
    return data


LIMITER_DEFAULT = dict(pre_gain_db=0.0, ceiling_db=-1.0,
                       lookahead_ms=3.0, release_ms=80.0)
SAFETY_LIMITER = dict(pre_gain_db=0.0, ceiling_db=-0.1,
                      lookahead_ms=3.0, release_ms=80.0)


def _comp(threshold_db, ratio, attack_ms, release_ms, makeup_db, mix_pct=100.0):
    return dict(threshold_db=threshold_db, ratio=ratio, knee_db=6.0,
                attack_ms=attack_ms, release_ms=release_ms,
                makeup_db=makeup_db, mix_pct=mix_pct)


# Preset bank: name -> chain builder taking the shared boost amount in dB.
# Order matters - it is the table order, cleanest candidates first.
PRESETS = [
    ('Limiter',
     lambda b: [(amp_limiter, dict(LIMITER_DEFAULT, pre_gain_db=b))]),
    ('Comp gentle 3:1 → Lim',
     lambda b: [(amp_compressor, _comp(-24, 3, 10, 200, b)),
                (amp_limiter, SAFETY_LIMITER)]),
    ('Comp punchy 6:1 → Lim',
     lambda b: [(amp_compressor, _comp(-30, 6, 2, 120, b)),
                (amp_limiter, SAFETY_LIMITER)]),
    ('Parallel comp 50% → Lim',
     lambda b: [(amp_compressor, _comp(-35, 8, 5, 150, b, mix_pct=50)),
                (amp_limiter, SAFETY_LIMITER)]),
    ('Soft clip tanh',
     lambda b: [(amp_soft_clip_tanh, dict(drive_db=b, trim_db=-0.1))]),
    ('Soft clip cubic',
     lambda b: [(amp_soft_clip_cubic, dict(drive_db=b, trim_db=-0.1))]),
    ('RMS → -16 → Lim',
     lambda b: [(amp_normalize_rms, dict(target_rms_db=-16.0)),
                (amp_limiter, SAFETY_LIMITER)]),
    ('Peak normalize -1',
     lambda b: [(amp_normalize_peak, dict(target_db=-1.0))]),
    ('Linear gain (clips!)',
     lambda b: [(amp_linear, dict(gain_db=b))]),
    ('Hard clip',
     lambda b: [(amp_hard_clip, dict(pre_gain_db=b, ceiling_db=-0.1))]),
]


# --- Stack modules: one toggle + one intensity slider each, processed in
# --- list order (top to bottom = signal flow). Sliders map to sensible
# --- macro-parameters so each module needs exactly one number.

def amp_stack_gain(data, sr, v):
    return data * _db_to_lin(v)


def amp_stack_comp(data, sr, v):
    """Compression amount 0-100%. Peak-restored: the slider adds density
    (quiet parts come up) without changing the peak level, so it composes
    predictably with the gain/limiter modules."""
    a = v / 100.0
    if a <= 0.0:
        return data
    out = amp_compressor(data, sr,
                         _comp(-10.0 - 30.0 * a, 1.0 + 7.0 * a, 5.0, 150.0, 0.0))
    peak_in, peak_out = np.max(np.abs(data)), np.max(np.abs(out))
    if peak_out > EPS:
        out *= peak_in / peak_out
    return out


def amp_stack_parallel(data, sr, v):
    """Parallel (NY) compression: blend in a heavily crushed copy. Keeps the
    transient shape of the dry signal while thickening the quiet tail.
    Result is peak-restored like the compressor module."""
    a = v / 100.0
    if a <= 0.0:
        return data
    wet = amp_compressor(data, sr, _comp(-40.0, 10.0, 5.0, 150.0, 0.0))
    peak_in, peak_wet = np.max(np.abs(data)), np.max(np.abs(wet))
    if peak_wet > EPS:
        wet *= peak_in / peak_wet
    out = data + a * wet
    peak_out = np.max(np.abs(out))
    if peak_out > EPS:
        out *= peak_in / peak_out
    return out


def amp_stack_soft(data, sr, v):
    if v <= 0.0:
        return data
    return np.tanh(data * _db_to_lin(v)) * _db_to_lin(-0.1)


def amp_stack_limit(data, sr, v):
    return amp_limiter(data, sr, dict(LIMITER_DEFAULT, pre_gain_db=v))


# (name, hint, slider min, max, default, suffix, default-enabled, func)
STACK_MODULES = [
    ('Gain', 'linear boost into the chain', 0, 36, 12, ' dB', True, amp_stack_gain),
    ('Compressor', 'density — quiet parts up, peak unchanged', 0, 100, 40, ' %', False, amp_stack_comp),
    ('Parallel comp', 'crushed copy blended under the dry signal', 0, 100, 0, ' %', False, amp_stack_parallel),
    ('Soft clip', 'tanh drive — rounds peaks instead of clipping', 0, 24, 6, ' dB', False, amp_stack_soft),
    ('Limiter', 'push into a -1 dBFS look-ahead wall', 0, 24, 0, ' dB', True, amp_stack_limit),
]


# Custom tab: (label, key, min, max, default, step, decimals, suffix)
METHODS = {
    'Linear Gain': (amp_linear, [
        ('Gain', 'gain_db', -24.0, 48.0, 12.0, 1.0, 1, ' dB'),
    ]),
    'Normalize Peak': (amp_normalize_peak, [
        ('Target peak', 'target_db', -24.0, 0.0, -1.0, 0.5, 1, ' dBFS'),
    ]),
    'Normalize RMS': (amp_normalize_rms, [
        ('Target RMS', 'target_rms_db', -40.0, 0.0, -18.0, 0.5, 1, ' dBFS'),
    ]),
    'Compressor': (amp_compressor, [
        ('Threshold', 'threshold_db', -60.0, 0.0, -28.0, 1.0, 1, ' dB'),
        ('Ratio', 'ratio', 1.0, 20.0, 4.0, 0.5, 1, ' :1'),
        ('Knee', 'knee_db', 0.0, 24.0, 6.0, 1.0, 1, ' dB'),
        ('Attack', 'attack_ms', 0.1, 200.0, 5.0, 1.0, 1, ' ms'),
        ('Release', 'release_ms', 10.0, 1000.0, 150.0, 10.0, 0, ' ms'),
        ('Makeup', 'makeup_db', 0.0, 36.0, 10.0, 1.0, 1, ' dB'),
        ('Mix (wet)', 'mix_pct', 0.0, 100.0, 100.0, 5.0, 0, ' %'),
    ]),
    'Limiter (lookahead)': (amp_limiter, [
        ('Pre-gain', 'pre_gain_db', 0.0, 48.0, 12.0, 1.0, 1, ' dB'),
        ('Ceiling', 'ceiling_db', -12.0, 0.0, -1.0, 0.1, 1, ' dBFS'),
        ('Lookahead', 'lookahead_ms', 0.5, 20.0, 3.0, 0.5, 1, ' ms'),
        ('Release', 'release_ms', 10.0, 1000.0, 80.0, 10.0, 0, ' ms'),
    ]),
    'Soft Clip (tanh)': (amp_soft_clip_tanh, [
        ('Drive', 'drive_db', 0.0, 48.0, 12.0, 1.0, 1, ' dB'),
        ('Output trim', 'trim_db', -12.0, 0.0, -0.1, 0.1, 1, ' dB'),
    ]),
    'Soft Clip (cubic)': (amp_soft_clip_cubic, [
        ('Drive', 'drive_db', 0.0, 48.0, 12.0, 1.0, 1, ' dB'),
        ('Output trim', 'trim_db', -12.0, 0.0, -0.1, 0.1, 1, ' dB'),
    ]),
    'Hard Clip': (amp_hard_clip, [
        ('Pre-gain', 'pre_gain_db', 0.0, 48.0, 12.0, 1.0, 1, ' dB'),
        ('Ceiling', 'ceiling_db', -12.0, 0.0, -0.1, 0.1, 1, ' dBFS'),
    ]),
}


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def spectrogram_db(mono, sr):
    nper = 2048 if sr >= 32000 else 1024
    f, t, S = signal.stft(mono, sr, nperseg=nper, noverlap=nper * 3 // 4)
    return f, t, np.maximum(20.0 * np.log10(np.abs(S) + EPS), SPEC_FLOOR_DB)


def waveform_xy(mono, sr, bins=4000):
    """Min/max envelope interleaved into one polyline for fast drawing."""
    n = mono.size
    if n <= bins * 2:
        return np.arange(n) / sr, mono.copy()
    hop = n // bins
    seg = mono[:hop * bins].reshape(bins, hop)
    xs = np.repeat((np.arange(bins) * hop + hop / 2) / sr, 2)
    ys = np.empty(bins * 2)
    ys[0::2] = seg.min(axis=1)
    ys[1::2] = seg.max(axis=1)
    return xs, ys


def audio_stats(data):
    det = np.abs(data)
    peak = float(det.max()) if det.size else 0.0
    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2))) if det.size else 0.0
    peak_db = 20 * np.log10(peak + EPS)
    rms_db = 20 * np.log10(rms + EPS)
    clipped = int(np.count_nonzero(det >= 0.9999))
    pct = 100.0 * clipped / max(det.size, 1)
    return peak_db, rms_db, clipped, pct


def stats_line(data):
    peak_db, rms_db, clipped, pct = audio_stats(data)
    return (f"peak {peak_db:+.1f} dBFS   rms {rms_db:+.1f} dBFS   "
            f"crest {peak_db - rms_db:.1f} dB   "
            f"clipped {clipped} ({pct:.3f}%)")


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

STAT_COLS = ['Preset', 'Peak dB', 'RMS dB', 'Clip %']


class AmpLab(QtWidgets.QMainWindow):
    def __init__(self, initial_path=None):
        super().__init__()
        self.setWindowTitle('Sound Amp Lab')
        self.resize(1500, 900)

        self.data = None          # (n, ch) float64, original
        self.processed = None     # currently displayed output
        self.processed_name = None
        self.sr = None
        self.info = None          # soundfile info of the loaded file
        self.path = None
        self.renders = {}         # preset name -> (n, ch) float32
        self._param_widgets = {}
        self._stack_rows = []     # (checkbox, slider, value_label, name, func)
        self._presets_dirty = True

        # Streaming playback state. The audio callback reads _play_buf at
        # _play_pos; re-renders hot-swap the buffer at the SAME position so
        # filter changes are audible mid-sound. Guarded by _play_lock
        # (callback runs on the PortAudio thread).
        self._play_lock = threading.Lock()
        self._play_buf = None
        self._play_pos = 0
        self._play_src = None     # 'input' | 'output' | None
        self._stream = None
        self._loop = True

        self._build_ui()

        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self._recompute_presets)

        self._custom_debounce = QtCore.QTimer(self)
        self._custom_debounce.setSingleShot(True)
        self._custom_debounce.setInterval(250)
        self._custom_debounce.timeout.connect(self._apply_custom)

        self._stack_debounce = QtCore.QTimer(self)
        self._stack_debounce.setSingleShot(True)
        self._stack_debounce.setInterval(150)
        self._stack_debounce.timeout.connect(self._apply_stack)

        threading.Thread(target=_warmup_numba, daemon=True).start()

        if initial_path:
            self._load(Path(initial_path))

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        tb = self.addToolBar('main')
        tb.setMovable(False)
        tb.addAction('Load…', self._load_dialog)
        tb.addAction('Save output…', self._save_dialog)
        tb.addSeparator()
        tb.addAction('▶ Input', lambda: self._play(self.data, 'input'))
        tb.addAction('▶ Output', lambda: self._play(self.processed, 'output'))
        tb.addAction('■ Stop', self._stop_playback)
        loop_box = QtWidgets.QCheckBox('Loop')
        loop_box.setChecked(True)
        loop_box.toggled.connect(lambda on: setattr(self, '_loop', bool(on)))
        tb.addWidget(loop_box)
        tb.addSeparator()
        self.file_label = QtWidgets.QLabel('  no file loaded')
        tb.addWidget(self.file_label)

        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # --- left: tabs -----------------------------------------------------
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)

        self.tabs = QtWidgets.QTabWidget()
        lv.addWidget(self.tabs, 1)

        # -- Stack tab: toggle + intensity slider per module, run in order
        stab = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(stab)
        sv.addWidget(QtWidgets.QLabel(
            '<b>Enabled modules run top to bottom</b>'))
        for name, hint, lo, hi, default, suffix, enabled, func in STACK_MODULES:
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(enabled)
            cb.setToolTip(hint)
            val_lab = QtWidgets.QLabel(f'{default}{suffix}')
            val_lab.setAlignment(Qt.AlignmentFlag.AlignRight |
                                 Qt.AlignmentFlag.AlignVCenter)
            head = QtWidgets.QHBoxLayout()
            head.addWidget(cb)
            head.addStretch(1)
            head.addWidget(val_lab)
            sv.addLayout(head)

            slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
            slider.setRange(lo, hi)
            slider.setValue(default)
            slider.setEnabled(enabled)
            slider.valueChanged.connect(
                lambda v, lab=val_lab, sfx=suffix: (lab.setText(f'{v}{sfx}'),
                                                    self._schedule_stack()))
            cb.toggled.connect(
                lambda on, s=slider: (s.setEnabled(on), self._schedule_stack()))
            sv.addWidget(slider)

            hint_lab = QtWidgets.QLabel(hint)
            hint_lab.setStyleSheet('color: gray; font-size: 10px')
            sv.addWidget(hint_lab)
            sv.addSpacing(6)
            self._stack_rows.append((cb, slider, val_lab, name, func))
        sv.addStretch(1)
        self.tabs.addTab(stab, 'Stack')

        # -- Presets tab
        ptab = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(ptab)

        boost_row = QtWidgets.QHBoxLayout()
        boost_row.addWidget(QtWidgets.QLabel('<b>Boost</b>'))
        self.boost = QtWidgets.QDoubleSpinBox()
        self.boost.setRange(0.0, 36.0)
        self.boost.setValue(12.0)
        self.boost.setSingleStep(1.0)
        self.boost.setDecimals(1)
        self.boost.setSuffix(' dB')
        self.boost.valueChanged.connect(self._boost_changed)
        boost_row.addWidget(self.boost)
        boost_row.addWidget(QtWidgets.QLabel('drives every preset below'))
        boost_row.addStretch(1)
        pv.addLayout(boost_row)

        self.table = QtWidgets.QTableWidget(len(PRESETS), len(STAT_COLS))
        self.table.setHorizontalHeaderLabels(STAT_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for c in range(1, len(STAT_COLS)):
            self.table.horizontalHeader().setSectionResizeMode(
                c, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        for r, (name, _) in enumerate(PRESETS):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
            for c in range(1, len(STAT_COLS)):
                item = QtWidgets.QTableWidgetItem('—')
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight |
                                      Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, item)
        self.table.itemSelectionChanged.connect(self._preset_selected)
        self.table.itemDoubleClicked.connect(
            lambda *_: self._play(self.processed, 'output'))
        pv.addWidget(self.table, 1)

        hint = QtWidgets.QLabel('click = view    double-click = play')
        hint.setStyleSheet('color: gray')
        pv.addWidget(hint)
        self.tabs.addTab(ptab, 'Presets')

        # -- Custom tab
        ctab = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(ctab)
        cv.addWidget(QtWidgets.QLabel('<b>Method</b>'))
        self.method_box = QtWidgets.QComboBox()
        self.method_box.addItems(METHODS.keys())
        self.method_box.currentTextChanged.connect(self._rebuild_params)
        cv.addWidget(self.method_box)

        self.param_form = QtWidgets.QFormLayout()
        self.param_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        pf = QtWidgets.QWidget()
        pf.setLayout(self.param_form)
        cv.addWidget(pf)

        self.post_limit = QtWidgets.QCheckBox(
            'Safety limiter after method  (-0.1 dBFS, 3 ms lookahead)')
        self.post_limit.stateChanged.connect(self._schedule_custom)
        cv.addWidget(self.post_limit)

        row = QtWidgets.QHBoxLayout()
        apply_btn = QtWidgets.QPushButton('Apply')
        apply_btn.clicked.connect(self._apply_custom)
        self.auto_apply = QtWidgets.QCheckBox('Auto-apply')
        self.auto_apply.setChecked(True)
        row.addWidget(apply_btn)
        row.addWidget(self.auto_apply)
        row.addStretch(1)
        cv.addLayout(row)
        cv.addStretch(1)
        self.tabs.addTab(ctab, 'Custom')

        # -- shared stats + note
        self.in_stats = QtWidgets.QLabel('Input:  —')
        self.out_stats = QtWidgets.QLabel('Output: —')
        lv.addWidget(self.in_stats)
        lv.addWidget(self.out_stats)
        note = QtWidgets.QLabel(
            'Saving clips to ±1.0 and keeps the original sample rate '
            'and bit depth. The input file is never modified.')
        note.setWordWrap(True)
        note.setStyleSheet('color: gray')
        lv.addWidget(note)

        # --- right: plots ---------------------------------------------------
        glw = pg.GraphicsLayoutWidget()

        self.wave_in = glw.addPlot(row=0, col=0, title='Input waveform')
        self.spec_in = glw.addPlot(row=1, col=0, title='Input spectrogram (dBFS)')
        self.wave_out = glw.addPlot(row=2, col=0, title='Output waveform')
        self.spec_out = glw.addPlot(row=3, col=0, title='Output spectrogram (dBFS)')

        for p in (self.wave_in, self.wave_out):
            p.setYRange(-1.05, 1.05)
            p.addLine(y=1.0, pen=pg.mkPen('r', style=Qt.PenStyle.DashLine))
            p.addLine(y=-1.0, pen=pg.mkPen('r', style=Qt.PenStyle.DashLine))
            p.setLabel('left', 'amp')
        for p in (self.spec_in, self.spec_out):
            p.setLabel('left', 'Hz')
        self.spec_out.setLabel('bottom', 'time', units='s')

        self.wave_in_curve = self.wave_in.plot(pen=pg.mkPen('#4fc3f7', width=1))
        self.wave_out_curve = self.wave_out.plot(pen=pg.mkPen('#aed581', width=1))

        try:
            cmap = pg.colormap.get('inferno')
        except Exception:
            cmap = pg.colormap.get('viridis')
        self.spec_in_img = pg.ImageItem()
        self.spec_out_img = pg.ImageItem()
        for img in (self.spec_in_img, self.spec_out_img):
            img.setLookupTable(cmap.getLookupTable())
            img.setLevels((SPEC_FLOOR_DB, 0.0))
        self.spec_in.addItem(self.spec_in_img)
        self.spec_out.addItem(self.spec_out_img)

        for p in (self.spec_in, self.wave_out, self.spec_out):
            p.setXLink(self.wave_in)
        self.spec_out.setYLink(self.spec_in)

        splitter.addWidget(left)
        splitter.addWidget(glw)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1080])

        self.tabs.currentChanged.connect(self._refresh_active_tab)
        self._rebuild_params(self.method_box.currentText())

    # ---------------------------------------------------------------- stack
    def _schedule_stack(self, *_):
        if self.data is not None:
            self._stack_debounce.start()

    def _apply_stack(self):
        if self.data is None:
            return
        out = self.data
        names = []
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for cb, slider, _lab, name, func in self._stack_rows:
                if cb.isChecked():
                    out = func(out, self.sr, float(slider.value()))
                    names.append(name)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Processing failed', str(e))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        label = 'Stack: ' + (' → '.join(names) if names else 'bypass')
        self._show_output(out, label)

    # ----------------------------------------------------------- tab switch
    def _refresh_active_tab(self, *_):
        if self.data is None:
            return
        i = self.tabs.currentIndex()
        if i == 0:
            self._apply_stack()
        elif i == 1:
            if self._presets_dirty:
                self._recompute_presets()
            else:
                self._preset_selected()
        else:
            self._apply_custom()

    def _boost_changed(self, *_):
        self._presets_dirty = True
        if self.data is not None and self.tabs.currentIndex() == 1:
            self._debounce.start()

    # --------------------------------------------------------------- custom
    def _rebuild_params(self, method_name):
        while self.param_form.rowCount():
            self.param_form.removeRow(0)
        self._param_widgets = {}
        for label, key, lo, hi, default, step, decimals, suffix in METHODS[method_name][1]:
            box = QtWidgets.QDoubleSpinBox()
            box.setRange(lo, hi)
            box.setValue(default)
            box.setSingleStep(step)
            box.setDecimals(decimals)
            box.setSuffix(suffix)
            box.valueChanged.connect(self._schedule_custom)
            self.param_form.addRow(label, box)
            self._param_widgets[key] = box
        self._schedule_custom()

    def _schedule_custom(self, *_):
        if self.data is not None and self.auto_apply.isChecked():
            self._custom_debounce.start()

    def _apply_custom(self):
        if self.data is None:
            return
        method = self.method_box.currentText()
        func = METHODS[method][0]
        params = {k: w.value() for k, w in self._param_widgets.items()}
        chain = [(func, params)]
        if self.post_limit.isChecked():
            chain.append((amp_limiter, SAFETY_LIMITER))
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            out = run_chain(self.data, self.sr, chain)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Processing failed', str(e))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._show_output(out, f'Custom — {method}')

    # -------------------------------------------------------------- presets
    def _recompute_presets(self):
        if self.data is None:
            return
        boost = self.boost.value()
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.renders = {}
            for r, (name, builder) in enumerate(PRESETS):
                out = run_chain(self.data, self.sr, builder(boost))
                self.renders[name] = out.astype(np.float32)
                peak_db, rms_db, clipped, pct = audio_stats(out)
                self.table.item(r, 1).setText(f'{peak_db:+.1f}')
                self.table.item(r, 2).setText(f'{rms_db:+.1f}')
                self.table.item(r, 3).setText(f'{pct:.3f}')
                warn = pct > 0.001
                for c in range(len(STAT_COLS)):
                    self.table.item(r, c).setForeground(
                        pg.mkColor('#ef5350' if warn else '#eeeeee'))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._presets_dirty = False

        if not self.table.selectedItems():
            self.table.selectRow(0)     # triggers _preset_selected
        else:
            self._preset_selected()

    def _preset_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        name = PRESETS[rows[0].row()][0]
        out = self.renders.get(name)
        if out is not None:
            self._show_output(out, name)

    # -------------------------------------------------------------- display
    def _show_output(self, out, name):
        self.processed = out
        self.processed_name = name
        self._swap_output_playback(out)
        self.out_stats.setText(f'Output ({name}):  {stats_line(out)}')
        self.wave_out.setTitle(f'Output waveform — {name}')
        self.spec_out.setTitle(f'Output spectrogram (dBFS) — {name}')

        mono = out.mean(axis=1, dtype=np.float64)
        xs, ys = waveform_xy(mono, self.sr)
        self.wave_out_curve.setData(xs, ys)
        f, t, S = spectrogram_db(mono, self.sr)
        self.spec_out_img.setImage(S, autoLevels=False)
        dur = out.shape[0] / self.sr
        self.spec_out_img.setRect(QtCore.QRectF(0, 0, t[-1] if t.size else dur, f[-1]))

    # ------------------------------------------------------------ file I/O
    def _load_dialog(self):
        start = str(self.path.parent if self.path else
                    (DEFAULT_DIR if DEFAULT_DIR.is_dir() else REPO_ROOT))
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Load sound', start,
            'Sound files (*.wav *.flac *.ogg *.mp3 *.aiff *.aif);;All files (*)')
        if fn:
            self._load(Path(fn))

    def _load(self, path):
        try:
            data, sr = sf.read(str(path), always_2d=True, dtype='float64')
            info = sf.info(str(path))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Load failed', f'{path}\n\n{e}')
            return
        self._stop_playback()   # sample rate / channel count may change
        self.data, self.sr, self.info, self.path = data, sr, info, path
        self.processed = None
        dur = data.shape[0] / sr
        self.file_label.setText(
            f'  {path.name}   —   {sr} Hz, {data.shape[1]} ch, '
            f'{info.subtype}, {dur:.2f} s')
        self.in_stats.setText(f'Input:  {stats_line(data)}')

        mono = data.mean(axis=1)
        xs, ys = waveform_xy(mono, sr)
        self.wave_in_curve.setData(xs, ys)
        f, t, S = spectrogram_db(mono, sr)
        self.spec_in_img.setImage(S, autoLevels=False)
        self.spec_in_img.setRect(QtCore.QRectF(0, 0, t[-1] if t.size else dur, f[-1]))
        self.wave_in.setXRange(0, dur, padding=0.01)
        self.spec_in.setYRange(0, f[-1], padding=0)

        self.renders = {}
        self._presets_dirty = True
        self._refresh_active_tab()

    def _save_dialog(self):
        if self.processed is None:
            QtWidgets.QMessageBox.information(self, 'Nothing to save',
                                              'Load a file first.')
            return
        suffix = self.path.suffix
        tag = (self.processed_name or 'amp').split('(')[0].strip()
        tag = ''.join(ch if ch.isalnum() else '_' for ch in tag).strip('_').lower()
        default = str(self.path.with_name(f'{self.path.stem}_{tag}{suffix}'))
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save output', default, f'Same format (*{suffix});;All files (*)')
        if not fn:
            return
        out = np.clip(self.processed.astype(np.float64), -1.0, 1.0)
        try:
            sf.write(fn, out, self.sr, subtype=self.info.subtype)
        except Exception:
            # Subtype may be invalid for a changed extension - let libsndfile
            # pick the default subtype for the chosen format instead.
            try:
                sf.write(fn, out, self.sr)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'Save failed', str(e))
                return
        self.statusBar().showMessage(
            f'Saved {fn}  ({self.sr} Hz, {self.info.subtype})', 8000)

    # ------------------------------------------------------------ playback
    @staticmethod
    def _as_play_buf(data):
        # Clip for playback so out-of-range floats behave like the DAC
        # would make them behave - you hear what saving would produce.
        buf = np.ascontiguousarray(np.clip(data, -1.0, 1.0), dtype=np.float32)
        return buf[:, None] if buf.ndim == 1 else buf

    def _audio_cb(self, outdata, frames, time_info, status):
        with self._play_lock:
            buf = self._play_buf
            if buf is None:
                outdata[:] = 0
                raise sd.CallbackStop
            n = buf.shape[0]
            pos = self._play_pos % n
            if self._loop:
                idx = (pos + np.arange(frames)) % n
                outdata[:] = buf[idx]
                self._play_pos = (pos + frames) % n
            else:
                end = min(pos + frames, n)
                k = end - pos
                outdata[:k] = buf[pos:end]
                if k < frames:
                    outdata[k:] = 0
                    self._play_pos = 0
                    self._play_src = None
                    raise sd.CallbackStop
                self._play_pos = end

    def _close_stream(self):
        if self._stream is not None:
            try:
                self._stream.abort()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _stop_playback(self):
        with self._play_lock:
            self._play_buf = None
            self._play_src = None
            self._play_pos = 0
        self._close_stream()

    def _play(self, data, src):
        if data is None:
            return
        buf = self._as_play_buf(data)
        with self._play_lock:
            self._play_buf = buf
            self._play_pos = 0
            self._play_src = src
        if self._stream is None or not self._stream.active:
            self._close_stream()
            try:
                self._stream = sd.OutputStream(
                    samplerate=self.sr, channels=buf.shape[1],
                    dtype='float32', callback=self._audio_cb)
                self._stream.start()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'Playback failed', str(e))

    def _swap_output_playback(self, out):
        """If the output is currently playing, splice the new render in at
        the current playhead so the change is heard immediately."""
        if self._play_src != 'output':
            return
        buf = self._as_play_buf(out)
        with self._play_lock:
            if self._play_src == 'output':
                self._play_pos %= buf.shape[0]
                self._play_buf = buf


def main():
    ap = argparse.ArgumentParser(description='Interactive sound amplification test bench')
    ap.add_argument('path', nargs='?', help='sound file to load on startup')
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = AmpLab(args.path)
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
