"""Discover tab: a crate-digging workspace over Beatport.

Not just a search box - a results/detail split with set-aware discovery
("for my set" seeds a search from the track you're mixing out of; "more
like this" from a selected track), fit scoring vs your set + library, a
preview transport, and a wishlist panel with a running total. Buying still
happens on beatport.com (no cart API); everything else - login, search,
metadata, previews, fit - is live in here.

Login is username/password (see lib/dj/beatport.password_login); the
password goes only to Beatport, we store only the token.
"""
import os
import tempfile

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QCheckBox, QHBoxLayout, QHeaderView, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem,
                             QPushButton, QSplitter, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from lib.dj import beatport as BP
from tools.djplanner.player import TrackPlayer


class SearchWorker(QThread):
    """Runs one or SEVERAL (query, filters) pairs and merges the results,
    deduped by Beatport id - so 'For my set' can cast a wider net across a
    track's genres and still return one ranked list."""
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, client, queries):
        super().__init__()
        self.client = client
        self.queries = queries         # [(query_str, filters_dict), ...]

    def run(self):
        try:
            seen, rows = set(), []
            errors = []
            for q, f in self.queries:
                try:
                    for t in self.client.search(q, per_page=60, **f):
                        r = BP.beatport_row(t)
                        if r["bp_id"] not in seen:
                            seen.add(r["bp_id"])
                            rows.append(r)
                except Exception as e:
                    errors.append(str(e))
            if not rows and errors:
                raise RuntimeError(errors[0])
            self.done.emit(rows)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class LoginWorker(QThread):
    ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, auth, username, password):
        super().__init__()
        self.auth, self.username, self.password = auth, username, password

    def run(self):
        try:
            acct = BP.password_login(self.auth, self.username, self.password)
            self.ok.emit(acct.get("username") or self.username)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class PreviewWorker(QThread):
    """Download + decode a preview; optionally analyze it for real bpm/key."""
    done = pyqtSignal(object, object)          # samples, analysis-or-None
    failed = pyqtSignal(str)

    def __init__(self, client, row, analyze=False):
        super().__init__()
        self.client, self.row, self.analyze = client, row, analyze

    def run(self):
        try:
            from lib.dj.features import decode_file_stereo
            dest = os.path.join(tempfile.gettempdir(),
                                f"bp_prev_{self.row['bp_id']}.mp3")
            self.client.download_preview(self.row["raw"], dest)
            samples = decode_file_stereo(dest)
            analysis = None
            if self.analyze:
                from lib.dj.features import analyze_samples
                analysis = analyze_samples(samples.mean(axis=1), deep=False)
            try:
                os.remove(dest)
            except OSError:
                pass
            self.done.emit(samples, analysis)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


COLS = ["Title", "Artist", "BPM", "Key", "Genre", "Price", "Fit"]
_VERDICT_RANK = {"great": 4, "good": 3, "workable": 2}


class DiscoverTab(QWidget):
    def __init__(self, planner):
        super().__init__()
        self.planner = planner
        self.client = BP.BeatportClient()
        self.wishlist = BP.Wishlist(planner.music_dir)
        self.rows = []
        self._fit = {}                 # row id -> (fit_vs_track, neighbours)
        self._search = self._preview = self._login_worker = None
        self.player = TrackPlayer()

        v = QVBoxLayout(self)

        # -- toolbar: modes + query + filters ---------------------------------
        top = QHBoxLayout()
        self.query = QLineEdit()
        self.query.setPlaceholderText("Search Beatport: artist, track, "
                                      "label, genre...")
        self.query.returnPressed.connect(self.search)
        top.addWidget(self.query, 1)
        top.addWidget(QLabel("BPM"))
        self.bpm_lo = QLineEdit(placeholderText="min")
        self.bpm_lo.setFixedWidth(48)
        top.addWidget(self.bpm_lo)
        self.bpm_hi = QLineEdit(placeholderText="max")
        self.bpm_hi.setFixedWidth(48)
        top.addWidget(self.bpm_hi)
        sb = QPushButton("Search")
        sb.clicked.connect(self.search)
        top.addWidget(sb)
        v.addLayout(top)

        # Set-aware discovery + view controls.
        row2 = QHBoxLayout()
        b1 = QPushButton("For my set →")
        b1.setToolTip("Find tracks that would follow your set's last track "
                      "(seeded by its genre + tempo).")
        b1.clicked.connect(self.discover_for_set)
        row2.addWidget(b1)
        b2 = QPushButton("More like selected")
        b2.setToolTip("Search for tracks like the highlighted result "
                      "(same genre, nearby tempo).")
        b2.clicked.connect(self.more_like_selected)
        row2.addWidget(b2)
        row2.addSpacing(16)
        self.fit_only = QCheckBox("mixable only")
        self.fit_only.setToolTip("Hide results that can't beat-match your "
                                 "set's last track (tempo).")
        self.fit_only.stateChanged.connect(self._rerender)
        row2.addWidget(self.fit_only)
        self.harmonic = QCheckBox("in key")
        self.harmonic.setToolTip("Hide results that clash harmonically with "
                                 "your set's last track (Camelot-compatible "
                                 "only).")
        self.harmonic.stateChanged.connect(self._rerender)
        row2.addWidget(self.harmonic)
        self.sort_fit = QCheckBox("sort by fit")
        self.sort_fit.setChecked(True)
        self.sort_fit.stateChanged.connect(self._rerender)
        row2.addWidget(self.sort_fit)
        row2.addStretch(1)
        v.addLayout(row2)

        # -- results | detail + wishlist split --------------------------------
        split = QSplitter(Qt.Orientation.Horizontal)
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.doubleClicked.connect(lambda _: self.preview())
        split.addWidget(self.table)

        right = QSplitter(Qt.Orientation.Vertical)
        # Detail / fit panel.
        detail = QWidget()
        dv = QVBoxLayout(detail)
        dv.setContentsMargins(8, 4, 4, 4)
        self.detail = QLabel("Select a track to see its fit against your "
                             "set and library.")
        self.detail.setWordWrap(True)
        self.detail.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.detail.setTextFormat(Qt.TextFormat.RichText)
        dv.addWidget(self.detail, 1)
        # Preview transport.
        prow = QHBoxLayout()
        self.play_btn = QPushButton("▶ Preview")
        self.play_btn.clicked.connect(self.preview)
        prow.addWidget(self.play_btn)
        stop = QPushButton("■")
        stop.setFixedWidth(32)
        stop.clicked.connect(self.stop_preview)
        prow.addWidget(stop)
        self.prev_lbl = QLabel("")
        prow.addWidget(self.prev_lbl, 1)
        dv.addLayout(prow)
        # Actions.
        arow = QHBoxLayout()
        for label, fn in (("♥ Wishlist", self.add_wishlist),
                          ("\U0001f6d2 Buy", self.open_track),
                          ("+ Add to set", self.add_to_set),
                          ("⊕ Analyze seam", lambda: self.preview(True))):
            b = QPushButton(label)
            b.clicked.connect(fn)
            arow.addWidget(b)
        dv.addLayout(arow)
        right.addWidget(detail)

        # Wishlist panel.
        wl = QWidget()
        wv = QVBoxLayout(wl)
        wv.setContentsMargins(8, 4, 4, 4)
        wh = QHBoxLayout()
        self.wish_lbl = QLabel("Wishlist")
        wh.addWidget(self.wish_lbl, 1)
        ob = QPushButton("Open all on Beatport")
        ob.clicked.connect(lambda: self.wishlist.open_in_browser())
        wh.addWidget(ob)
        rb = QPushButton("Remove")
        rb.clicked.connect(self._wish_remove)
        wh.addWidget(rb)
        wv.addLayout(wh)
        self.wish_list = QListWidget()
        self.wish_list.itemDoubleClicked.connect(self._wish_open_one)
        wv.addWidget(self.wish_list)
        right.addWidget(wl)
        right.setSizes([320, 220])
        split.addWidget(right)
        split.setSizes([620, 420])
        v.addWidget(split, 1)

        # -- login row (when unsigned) ----------------------------------------
        self.signin_row = QWidget()
        sr = QHBoxLayout(self.signin_row)
        sr.setContentsMargins(0, 0, 0, 0)
        sr.addWidget(QLabel("Beatport login:"))
        self.user_edit = QLineEdit(placeholderText="username / email")
        sr.addWidget(self.user_edit, 1)
        self.pass_edit = QLineEdit(placeholderText="password")
        self.pass_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.pass_edit.returnPressed.connect(self._log_in)
        sr.addWidget(self.pass_edit, 1)
        self.login_btn = QPushButton("Log in")
        self.login_btn.clicked.connect(self._log_in)
        sr.addWidget(self.login_btn)
        v.addWidget(self.signin_row)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        v.addWidget(self.status)

        self._ptimer = QTimer(self)
        self._ptimer.timeout.connect(self._preview_tick)
        self._ptimer.start(250)
        self._refresh_wishlist()
        self._refresh_auth()

    # -- login --------------------------------------------------------------
    def _log_in(self):
        user, pw = self.user_edit.text().strip(), self.pass_edit.text()
        if not (user and pw):
            self.status.setText("enter your Beatport username and password.")
            return
        if self._login_worker and self._login_worker.isRunning():
            return
        self.login_btn.setEnabled(False)
        self.status.setText("logging in to Beatport...")
        self._login_worker = LoginWorker(self.client.auth, user, pw)
        self._login_worker.ok.connect(self._login_ok)
        self._login_worker.failed.connect(self._login_failed)
        self._login_worker.start()

    def _login_ok(self, who):
        self.pass_edit.clear()
        self.login_btn.setEnabled(True)
        self.status.setText(f"signed in as {who}.")
        self._refresh_auth()

    def _login_failed(self, msg):
        self.login_btn.setEnabled(True)
        self.status.setText("login failed: " + msg)

    def _refresh_auth(self):
        signed = self.client.available()
        self.signin_row.setVisible(not signed)
        for w in (self.query, self.bpm_lo, self.bpm_hi):
            w.setEnabled(signed)
        if not signed:
            self.status.setText(
                "Enter your Beatport username and password to sign in "
                "(same as beatport.com; the password goes only to Beatport, "
                "we store only the token). Then search, check fit, preview, "
                "wishlist, and buy on beatport.com.")
        else:
            self.status.setText("Signed in. Search, or use 'For my set' to "
                                "find tracks that follow your current set.")

    # -- helpers ------------------------------------------------------------
    def _cur_out_track(self):
        entries = self.planner.set_tab.entries
        if not entries:
            return None
        return next((t for t in self.planner.library
                     if t.id == entries[-1]["track_id"]), None)

    # -- search / discovery -------------------------------------------------
    def _run_search(self, queries, note):
        if not self.client.available() or (self._search
                                           and self._search.isRunning()):
            return
        self.status.setText(note)
        self._search = SearchWorker(self.client, queries)
        self._search.done.connect(self._results)
        self._search.failed.connect(
            lambda m: self.status.setText("search failed: " + m))
        self._search.start()

    def search(self):
        q = self.query.text().strip()
        if not q:
            return
        self._run_search([(q, self._bpm_filters())], "searching Beatport...")

    def _bpm_filters(self):
        f = {}
        try:
            if self.bpm_lo.text():
                f["bpm_low"] = int(float(self.bpm_lo.text()))
            if self.bpm_hi.text():
                f["bpm_high"] = int(float(self.bpm_hi.text()))
        except ValueError:
            pass
        return f

    def _tempo_filter(self, bpm):
        return ({"bpm_low": int(bpm * 0.92), "bpm_high": int(bpm * 1.08)}
                if bpm else {})

    def discover_for_set(self):
        t = self._cur_out_track()
        if t is None:
            self.status.setText("your set is empty - add a track first, then "
                                "'For my set' finds what follows it.")
            return
        # Cast a wider net: one query per genre we know for the anchor (up to
        # 3), each tempo-boxed to +/-8%, merged. Then auto-enable the
        # harmonic + fit sort so only in-key, mixable tracks surface, best
        # first. Falls back to artist when the track isn't enriched.
        genres = (getattr(t, "genres", None) or [])[:3]
        seeds = genres or [t.artist or t.title]
        f = self._tempo_filter(t.bpm)
        queries = [(g, f) for g in seeds]
        self.query.setText(seeds[0])
        self.harmonic.setChecked(True)
        self.sort_fit.setChecked(True)
        self._run_search(
            queries, f"finding in-key, tempo-matched tracks to follow "
            f"'{t.title[:26]}' ({t.bpm:.0f} bpm, {t.camelot})...")

    def more_like_selected(self):
        r = self._selected_row()
        if not r:
            self.status.setText("select a result first.")
            return
        q = r.get("genre") or r["artist"]
        self._run_search([(q, self._tempo_filter(r["bpm"]))],
                         f"more like '{r['title'][:30]}'...")

    # -- results rendering --------------------------------------------------
    def _results(self, rows):
        # Precompute fit for every row once (bpm/key from Beatport metadata).
        cur = self._cur_out_track()
        lib = self.planner.library
        self._fit = {}
        for r in rows:
            if not r["camelot"]:
                continue
            ghost = BP.ghost_trackinfo(r)
            ft = BP.fit_vs_track(cur, ghost) if cur is not None else None
            nb = BP.fit_vs_library(lib, ghost)["mixable_neighbours"] \
                if lib else 0
            self._fit[r["id"]] = (ft, nb)
        self.rows = rows
        self._rerender()

    def _fit_key(self, r):
        ft, nb = self._fit.get(r["id"], (None, 0))
        if ft is None:
            return (1, nb)
        if not ft["mixable"]:
            return (0, nb)
        return (2 + _VERDICT_RANK.get(ft["verdict"], 0), nb)

    def _rerender(self):
        rows = list(self.rows)
        cur = self._cur_out_track()
        if self.fit_only.isChecked() and cur is not None:
            rows = [r for r in rows if (self._fit.get(r["id"], (None,))[0]
                                        or {}).get("mixable")]
        if self.harmonic.isChecked() and cur is not None:
            # Camelot-compatible: same/relative/neighbour on the wheel (>=0.55).
            rows = [r for r in rows
                    if (self._fit.get(r["id"], (None,))[0] or {}
                        ).get("key_fit", 0) >= 0.55]
        if self.sort_fit.isChecked():
            rows.sort(key=self._fit_key, reverse=True)
        self._view = rows
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            fit_txt, col = self._fit_cell(r, cur)
            cells = [r["title"], r["artist"], f"{r['bpm']:.0f}", r["camelot"],
                     r["genre"], str(r["price"] or ""), fit_txt]
            for c, val in enumerate(cells):
                it = QTableWidgetItem(val)
                if c == 6 and col:
                    it.setForeground(col)
                self.table.setItem(i, c, it)
        self.status.setText(
            f"{len(rows)} results" + (f" (of {len(self.rows)})"
            if len(rows) != len(self.rows) else "")
            + ("  · fit vs your set's last track + library"
               if cur is not None else "  · sign of fit needs a track in the set"))

    def _fit_cell(self, r, cur):
        ft, nb = self._fit.get(r["id"], (None, 0))
        parts, col = [], None
        if ft is not None:
            if ft["mixable"]:
                parts.append(f"{ft['verdict']} {ft['stretch_pct']:+.1f}%")
                col = (QColor(120, 200, 140) if ft["verdict"] == "great"
                       else QColor(200, 180, 90))
            else:
                parts.append("fade only")
                col = QColor(200, 120, 120)
        if nb:
            parts.append(f"{nb} nbrs")
        return "  ".join(parts), col

    def _selected_row(self):
        i = self.table.currentRow()
        return getattr(self, "_view", self.rows)[i] \
            if 0 <= i < len(getattr(self, "_view", self.rows)) else None

    def _selection_changed(self):
        r = self._selected_row()
        if not r:
            return
        cur = self._cur_out_track()
        ft, nb = self._fit.get(r["id"], (None, 0))
        lines = [f"<b>{r['title']}</b><br>{r['artist']}",
                 f"{r['bpm']:.0f} bpm &nbsp; {r['camelot']} &nbsp; "
                 f"{r['genre'] or ''}",
                 f"{r.get('label') or ''} &nbsp; {r.get('released') or ''} "
                 f"&nbsp; <b>{r['price'] or ''}</b>"]
        if ft is not None:
            if ft["mixable"]:
                lines.append(f"<br><b>Into your set</b> (after "
                             f"{cur.title[:24]}): <b>{ft['verdict']}</b>, "
                             f"{ft['stretch_pct']:+.1f}% stretch, key fit "
                             f"{ft['key_fit']}")
            else:
                lines.append(f"<br><b>Into your set</b>: fade only "
                             f"(tempo gap beyond ±8%)")
        # Best library neighbours.
        if self.planner.library and r["camelot"]:
            fl = BP.fit_vs_library(self.planner.library, BP.ghost_trackinfo(r),
                                   top=4)
            lines.append(f"<b>Library:</b> {fl['mixable_neighbours']} "
                         f"mixable neighbours")
            for b in fl["best"]:
                lines.append(f"&nbsp;&nbsp;• {b['title'][:28]} — "
                             f"{b['bpm']:.0f} {b['key']} (key {b['key_fit']})")
        self.detail.setText("<br>".join(lines))

    # -- preview ------------------------------------------------------------
    def preview(self, analyze=False):
        r = self._selected_row()
        if not r or (self._preview and self._preview.isRunning()):
            return
        self.status.setText("analyzing preview..." if analyze
                            else "loading preview...")
        self.planner.claim_playback("discover")
        self._preview = PreviewWorker(self.client, r, analyze=analyze)
        self._preview.done.connect(lambda s, a: self._play_samples(s, a, r))
        self._preview.failed.connect(
            lambda m: self.status.setText("preview failed: " + m))
        self._preview.start()

    def _play_samples(self, samples, analysis, r):
        self.player.load(samples)
        self.player.play()
        if analysis:
            self.status.setText(
                f"analyzed '{r['title'][:26]}': measured "
                f"{analysis['bpm']:.0f} bpm {analysis['camelot']} "
                f"(Beatport says {r['bpm']:.0f} {r['camelot']})")
        else:
            self.status.setText(f"previewing {r['title'][:30]}")

    def stop_preview(self):
        self.player.pause()

    def _preview_tick(self):
        if self.player.playing:
            self.prev_lbl.setText(f"{self.player.time_s():.0f}s")

    # -- wishlist -----------------------------------------------------------
    def add_wishlist(self):
        r = self._selected_row()
        if r:
            added = self.wishlist.add(r)
            self._refresh_wishlist()
            self.status.setText(("added to wishlist: " if added else
                                 "already on wishlist: ") + r["title"])

    def _refresh_wishlist(self):
        self.wish_list.clear()
        total = 0.0
        for it in self.wishlist.items:
            self.wish_list.addItem(QListWidgetItem(
                f"{it['title'][:28]} — {it['artist'][:18]}  "
                f"{it.get('bpm', 0):.0f} {it.get('camelot', '')}  "
                f"{it.get('price') or ''}"))
            p = str(it.get("price") or "").replace("$", "").replace("€", "")
            try:
                total += float(p)
            except ValueError:
                pass
        self.wish_lbl.setText(
            f"Wishlist — {len(self.wishlist.items)} tracks"
            + (f", ~${total:.2f}" if total else ""))

    def _wish_remove(self):
        i = self.wish_list.currentRow()
        if 0 <= i < len(self.wishlist.items):
            self.wishlist.remove(self.wishlist.items[i].get("bp_id"))
            self._refresh_wishlist()

    def _wish_open_one(self, _item):
        i = self.wish_list.currentRow()
        if 0 <= i < len(self.wishlist.items):
            self.wishlist.open_in_browser(self.wishlist.items[i].get("bp_id"))

    # -- misc ---------------------------------------------------------------
    def open_track(self):
        r = self._selected_row()
        if r and r["url"]:
            import webbrowser
            webbrowser.open(r["url"])
            self.status.setText("opened on Beatport - add to cart there.")

    def add_to_set(self):
        r = self._selected_row()
        if not r:
            return
        self.status.setText("analyzing preview for a real seam plan...")
        try:
            ghost = BP.deep_ghost(self.client, r)
        except Exception as e:
            self.status.setText(f"couldn't analyze preview: {e}")
            return
        ghost.is_ghost = True
        self.planner.library.append(ghost)
        self.planner.set_tab.add_tracks([ghost])
        self.status.setText(f"added '{r['title']}' as a ghost track - "
                            "audition the seam in Set/Mix, then buy it if it "
                            "works. (Ghosts aren't saved to setlists.)")

    def close(self):
        self.player.close()
