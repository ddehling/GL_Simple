"""Drive the planner's Analysis tab headless (offscreen Qt, no audio
device) through the mixer: open a stem-rendered track with a reading,
mute a stem, solo a row on the stems source (the gated recording),
refuse a row mute there, switch to the reconstruction (voices render
one by one and cache), mute a voice instantly, mute a stem, switch
back. Plus the limiter / gate primitives and the transport button
states. Needs a library with stems + a reading: --music D:/Devel/music
(track 51 by default).

    python tools/tests/_dj_analysis_mixer_test.py [--music DIR] [--track ID]
"""
import argparse
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from PyQt6.QtWidgets import QApplication

app = QApplication([])

from tools import dj_planner as P                      # noqa: E402
from tools.dj.planner.player import TrackPlayer         # noqa: E402

RATE = 44100
FAILS = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


def wait_until(pred, timeout_s, what):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        app.processEvents()
        if pred():
            return True
        time.sleep(0.02)
    print(f"  TIMEOUT waiting for {what} ({timeout_s:.0f}s)")
    FAILS.append("timeout: " + what)
    return False


def dbfs(x):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-12)


# -- primitives ------------------------------------------------------------------
print("primitives")
n = RATE * 2
x = np.zeros((n, 2), dtype=np.float32)
x[:, 0] = x[:, 1] = 0.5 * np.sin(np.arange(n) * 2 * np.pi * 220 / RATE)
x[RATE: RATE + 40] *= 8.0                                   # one 4.0 spike
P.MixWorker.limit(x)
check(float(np.abs(x).max()) <= 1.0, "limiter: nothing above 1.0")
check(abs(float(np.abs(x[: RATE // 2]).max()) - 0.5) < 0.01, "limiter: the untouched half keeps its level")
check(float(np.abs(x[RATE + 20000:]).max()) > 0.45, "limiter: the gain is back 0.45 s after the spike")
g = P.MixWorker.gate([(0.5, 0.7), (1.0, 1.1)], n)
check(g[int(0.6 * RATE)] > 0.99 and g[int(0.3 * RATE)] < 0.01 and g[int(1.05 * RATE)] > 0.99, "gate: 1 inside the windows, 0 outside")
check(0.2 < g[int(0.5 * RATE)] < 0.8, "gate: a smoothed edge")

pl = TrackPlayer()
pl.load(np.zeros((RATE, 2), dtype=np.float32))
pl.seek(0.5)
pl.replace(np.ones((RATE, 2), dtype=np.float32) * 0.1)
check(abs(pl.time_s() - 0.5) < 1e-3, "player.replace keeps the position")
pl.seek(10.0)
check(pl.at_end(), "player.at_end after a seek past the end")

# -- the tab -----------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--music", default="D:/Devel/music")
ap.add_argument("--track", type=int, default=51)
args = ap.parse_args()

from lib.dj.db import LibraryDB
from lib.dj.brain import load_library

db = LibraryDB(args.music)
library = load_library(db)
track = next((t for t in library if t.id == args.track), None)
if track is None or not getattr(track, "has_stems", False):
    print(f"track {args.track} not in the library or without stems; skipping the tab test")
    sys.exit(1 if FAILS else 0)


class _Viewport:
    def update(self):
        pass


class _Table:
    def viewport(self):
        return _Viewport()


class _LibTab:
    table = _Table()


class FakePlanner:
    def __init__(self):
        self.db, self.music_dir = db, args.music
        self.library = self.library_all = library
        self.library_tab = _LibTab()
        self._stem_proc = None

    def claim_playback(self, owner):
        pass

    def reload_library(self, keep_analysis=False):
        pass


tab = P.AnalysisTab(FakePlanner())
tab.player.play = lambda: setattr(tab.player, "playing", True)     # no audio device in a test
tab.resize(1400, 900)
print(f"open track {track.id} '{track.title}'")
tab.open_track(track)
wait_until(lambda: tab._samples is not None, 60, "decode")
wait_until(lambda: tab._stems is not None, 120, "stems")
wait_until(lambda: tab._instruments is not None, 30, "reading")
if tab._instruments is None:
    print("no reading for this track: identify instruments first"); sys.exit(1)
lanes = tab.inst_lanes
ids = lanes._all_ids()
print("  instruments:", ids)
check(tab.src_recon.isEnabled(), "reconstruction source enabled with a reading")
check(tab._mix_key == tab._ALL_STEMS and tab.player.samples is tab._samples, "the original is in the player")
orig = tab._samples

# stem mute on the stems source: a stem sum, instantly
lanes.toggle_stem("drums", "M")
wait_until(lambda: tab._mix_key == ("bass", "other", "vocals"), 10, "drums muted -> stem sum")
want = sum(np.asarray(tab._stems[s], dtype=np.float32) for s in ("bass", "other", "vocals"))
P.MixWorker.limit(want)                                     # the mix is limited the same way
got = tab.player.samples
check(got is not orig and float(np.abs(got[:len(want)] - want).max()) < 0.05, "player holds bass+other+vocals")
check(tab.lanes.mute == {"drums"}, "the stem lanes mirror the stem mute")
check(not lanes._audible_now(ids[0]), "a muted stem silences its rows")

# a row solo on the stems source: the recording gated to that sound's hits
lanes.toggle_stem("drums", "M")                             # un-mute
wait_until(lambda: tab._mix_key == tab._ALL_STEMS, 10, "back to the original")
drum_rows = [i for i in ids if i.startswith("drums.")]
lanes.toggle_rows([drum_rows[0]], "S")
wait_until(lambda: tab._mix_key == ("gate", drum_rows[0]), 10, "row solo -> gated recording")
buf = tab.player.samples
wins = lanes.event_windows(drum_rows[0])
inside = int((wins[5][0] + wins[5][1]) / 2 * RATE)
stem = np.asarray(tab._stems["drums"], dtype=np.float32)
check(abs(float(buf[inside, 0]) - float(stem[inside, 0])) < 0.02, "inside a window: the drums stem itself")
# a spot at least 0.3 s away from every window is silent
mids = np.array([(a + b) / 2 for a, b in wins])
gaps = [(wins[i][1], wins[i + 1][0]) for i in range(len(wins) - 1) if wins[i + 1][0] - wins[i][1] > 0.6]
if gaps:
    a, b = gaps[0]
    seg = buf[int((a + 0.25) * RATE): int((b - 0.25) * RATE)]
    check(float(np.abs(seg).max()) < 1e-4, "between windows: silence")
check("gated" in tab.stems_lbl.text(), "status says gated")

# a row mute on the stems source is refused with a hint, state unchanged
lanes.toggle_rows([drum_rows[0]], "S")                      # un-solo
tab.stems_lbl.setText("")
from PyQt6.QtCore import QPointF, Qt, QEvent
from PyQt6.QtGui import QMouseEvent
row_y = None
for row, y0, rh in lanes._row_tops():
    if row[0] == "inst" and row[1]["id"] == drum_rows[0]:
        row_y = y0 + rh / 2
from tools.dj.planner.stemlanes import SM_W, SM_GAP
mx = lanes.SM_X + SM_W + SM_GAP + SM_W / 2
ev = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(mx, row_y), Qt.MouseButton.LeftButton,
                 Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
lanes.mouseReleaseEvent(ev)
app.processEvents()
check(not lanes.mute and "reconstruction" in tab.stems_lbl.text(), "row M on the stems source: refused, hint shown")
ev = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(lanes.SM_X + SM_W / 2, row_y), Qt.MouseButton.LeftButton,
                 Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
lanes.mouseReleaseEvent(ev)
app.processEvents()
check(lanes.solo == {drum_rows[0]}, "a click on the S box solos the row")
lanes.clear_solo_mute()
wait_until(lambda: tab._mix_key == tab._ALL_STEMS, 10, "clear S/M -> original")

# the reconstruction: voices render one by one and cache; then S/M is a sum
t0 = time.time()
tab.src_recon.setChecked(True)
app.processEvents()
check(tab._source == "recon" and lanes.row_mute_enabled, "source switched: row mutes enabled")
wait_until(lambda: any(v is not None for v in tab._units.values()), 90, "the first voice")
print(f"  first voice after {time.time() - t0:.1f}s: {tab.stems_lbl.text()[:100]}")
units_all = tab._recon_units(ids)        # voice units + the vocals phrases + a "recording" unit per note stem (the hybrid program)
wait_until(lambda: all(u in tab._units for u in units_all) and tab._resynth is None
           and tab._mix_key == ("recon",) + tuple(u for u in units_all if u in tab._units), 240, "every voice rendered and mixed")
print(f"  all {len(units_all)} units after {time.time() - t0:.1f}s")
rec = tab.player.samples
lvl = dbfs(rec[:, 0])
print(f"  reconstruction rms {lvl:.1f} dBFS (original {dbfs(orig[:, 0]):.1f}), peak {float(np.abs(rec).max()):.2f}")
check(float(np.abs(rec).max()) <= 1.0, "reconstruction peak <= 1.0")
check(abs(lvl - dbfs(orig[:, 0])) < 6.0, "reconstruction within 6 dB of the original's level")
if "other.1" in tab._units and tab._units["other.1"] is not None:
    from lib.dj import instruments as INS
    print("  other.1 rms %.1f dBFS" % dbfs(np.asarray(tab._units["other.1"], dtype=np.float32)))

# mute one voice: instant, no render
t1 = time.time()
victim = next(u for u in units_all if u != "vocals")
lanes.toggle_rows([victim], "M")
wait_until(lambda: tab._mix_key == ("recon",) + tuple(u for u in units_all if u != victim), 10, f"{victim} muted from the sum")
check(tab._resynth is None and time.time() - t1 < 5.0, "a voice mute did not re-render")
diff = rec[:, 0] - tab.player.samples[:, 0]
check(dbfs(diff) > -60, "the muted voice is gone from the buffer")
# mute a whole stem in the reconstruction
lanes.toggle_stem("bass", "M")
wait_until(lambda: tab._mix_key == ("recon",) + tuple(u for u in units_all if u != victim and u != "bass" and not u.startswith("bass.")), 10,
           "bass stem muted -> its voices and its recording unit out of the sum")
# everything muted -> silence
lanes.stem_mute.update(("drums", "other", "vocals"))
lanes._changed()
wait_until(lambda: tab._mix_key == (), 10, "everything muted -> silence")
check(float(np.abs(tab.player.samples).max()) == 0.0, "silent buffer")
lanes.clear_solo_mute()
wait_until(lambda: tab._mix_key == ("recon",) + tuple(units_all), 10, "clear -> the whole reconstruction")

# back to the stems with a stem mute in place
lanes.toggle_stem("drums", "M")
tab.src_stems.setChecked(True)
wait_until(lambda: tab._mix_key == ("bass", "other", "vocals"), 10, "stems source again: drums still muted")
check(not lanes.row_mute_enabled, "row mutes disabled on the stems source")

# transport button states
tab.play_btn.setText("▶ Play")
tab._toggle_play()
check(tab.player.playing and tab.play_btn.text() == "⏸ Pause", "play")
tab.player.playing = False                                   # the track ends on its own
tab._tick()
check(tab.play_btn.text() == "▶ Play", "the button follows the end of the track")
tab.player.seek(1e9)
tab._toggle_play()
check(tab.player.playing and tab.player.pos < RATE, "play after the end starts over")
tab.player.playing = False

print("\nALL OK" if not FAILS else f"\n{len(FAILS)} FAILED:\n  " + "\n  ".join(FAILS))
sys.exit(1 if FAILS else 0)
