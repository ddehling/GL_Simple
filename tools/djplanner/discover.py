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
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                             QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QListWidget, QListWidgetItem, QPushButton,
                             QSlider, QSplitter, QTableWidget,
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
                    for t in self._fetch(q, f):
                        r = BP.beatport_row(t)
                        if r["bp_id"] not in seen:
                            seen.add(r["bp_id"])
                            rows.append(r)
                except Exception as e:
                    errors.append(str(e))
            if not rows and errors:
                raise RuntimeError(errors[0])
            # Fit scoring runs on the GUI thread per row (vs the set AND
            # the whole library) - cap the merged pool so a 4-chart sweep
            # can't stall the window.
            self.done.emit(rows[:500])
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")

    def _fetch(self, q, f):
        if q is not None:
            return self.client.search(q, per_page=60, **f)
        # Genre mode: FULL-CATALOG genre search - empty q + genre_id +
        # bpm window is server-side filtered and paginated (verified live
        # 2026-07-20, pages disjoint). Two pages = up to 200 in-tempo
        # tracks per genre seed, vs the top-100 chart whose tempo/key/
        # owned filtering left "a very small number of songs". The chart
        # stays as the fallback.
        rows = []
        try:
            for page in (1, 2):
                got = self.client.search("", per_page=100, page=page, **f)
                rows.extend(got)
                if len(got) < 100:
                    break
        except Exception:
            rows = []
        if rows:
            return rows
        f = dict(f)
        gid = f.pop("genre_id")
        lo, hi = f.pop("bpm_low", 0), f.pop("bpm_high", 999)
        return [t for t in self.client.top(gid)
                if lo <= float(t.get("bpm") or 0) <= hi]


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


class GenresWorker(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, client):
        super().__init__()
        self.client = client

    def run(self):
        try:
            self.done.emit(self.client.genres())
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
        # Two-step-confirm arming state for the wishlist buttons.
        self._wish_armed = self._open_armed = None
        self._wish_total = 0.0
        self.rows = []
        self._fit = {}                 # row id -> (fit_vs_track, neighbours)
        self._search = self._preview = self._login_worker = None
        self._genres_worker = None
        self._genre_ids = {}           # lower-cased genre name -> beatport id
        self._gap_pair = None          # (a, b) TrackInfos: bridge-fit mode
        self._variety_cap = None       # max results per artist (discovery)
        self._owned_idx = None         # title-root -> artist tokens (lazy)
        self._owned_idx_n = -1         # library size the index was built at
        self.player = TrackPlayer()

        v = QVBoxLayout(self)

        # -- toolbar: modes + query + filters ---------------------------------
        top = QHBoxLayout()
        self.query = QLineEdit()
        self.query.setPlaceholderText("Search Beatport: artist, track, "
                                      "label... (leave empty + pick a genre "
                                      "for its top 100)")
        self.query.returnPressed.connect(self.search)
        top.addWidget(self.query, 1)
        top.addWidget(QLabel("Genre"))
        self.genre_combo = QComboBox()
        self.genre_combo.addItem("any genre", None)
        self.genre_combo.setMinimumWidth(170)
        self.genre_combo.setToolTip(
            "Filter results to a real Beatport genre. With an empty query, "
            "shows the genre's top-100 chart instead of a text search.")
        top.addWidget(self.genre_combo)
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
        self.hide_owned = QCheckBox("hide owned")
        self.hide_owned.setChecked(True)
        self.hide_owned.setToolTip("Hide tracks that are already in your "
                                   "library (matched by song title root + "
                                   "artist) - discovery should show what "
                                   "you DON'T have.")
        self.hide_owned.stateChanged.connect(self._rerender)
        row2.addWidget(self.hide_owned)
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
        # Preview transport: ⏮/⏭ hop through the RESULT ROWS (crate-dig
        # flow: hear one, hop to the next), slider scrubs the 2-min
        # preview clip live while dragging.
        prow = QHBoxLayout()
        pb_prev = QPushButton("⏮")
        pb_prev.setFixedWidth(32)
        pb_prev.setToolTip("Preview the previous result")
        pb_prev.clicked.connect(lambda: self._step_preview(-1))
        prow.addWidget(pb_prev)
        self.play_btn = QPushButton("▶ Preview")
        self.play_btn.clicked.connect(self.preview)
        prow.addWidget(self.play_btn)
        pb_next = QPushButton("⏭")
        pb_next.setFixedWidth(32)
        pb_next.setToolTip("Preview the next result")
        pb_next.clicked.connect(lambda: self._step_preview(1))
        prow.addWidget(pb_next)
        stop = QPushButton("■")
        stop.setFixedWidth(32)
        stop.clicked.connect(self.stop_preview)
        prow.addWidget(stop)
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setMaximum(1000)
        self.seek_slider.setToolTip("Scrub the preview clip (seeks live "
                                    "while dragging)")
        self._seek_drag = False
        self.seek_slider.sliderPressed.connect(
            lambda: setattr(self, "_seek_drag", True))
        self.seek_slider.sliderMoved.connect(self._seek_frac)
        self.seek_slider.sliderReleased.connect(self._seek_released)
        prow.addWidget(self.seek_slider, 1)
        self.prev_lbl = QLabel("-:-- / -:--")
        prow.addWidget(self.prev_lbl)
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
        self.wish_open_btn = QPushButton("Open all on Beatport")
        self.wish_open_btn.clicked.connect(self._wish_open)
        wh.addWidget(self.wish_open_btn)
        self.wish_rm_btn = QPushButton("Remove")
        self.wish_rm_btn.clicked.connect(self._wish_remove)
        wh.addWidget(self.wish_rm_btn)
        wv.addLayout(wh)
        self.wish_list = QListWidget()
        # Multi-select: ctrl/shift-click, ctrl+A, rubber-band drag.
        self.wish_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.wish_list.itemDoubleClicked.connect(self._wish_open_one)
        self.wish_list.itemSelectionChanged.connect(self._wish_sel_changed)
        wv.addWidget(self.wish_list)
        # Delete key removes the selection (same two-step confirm).
        self._wish_del = QShortcut(QKeySequence.StandardKey.Delete,
                                   self.wish_list)
        self._wish_del.setContext(Qt.ShortcutContext.WidgetShortcut)
        self._wish_del.activated.connect(self._wish_remove)
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
        for w in (self.query, self.genre_combo, self.bpm_lo, self.bpm_hi):
            w.setEnabled(signed)
        if signed and not self._genre_ids:
            self._load_genres()
        if not signed:
            self.status.setText(
                "Enter your Beatport username and password to sign in "
                "(same as beatport.com; the password goes only to Beatport, "
                "we store only the token). Then search, check fit, preview, "
                "wishlist, and buy on beatport.com.")
        else:
            self.status.setText("Signed in. Search, or use 'For my set' to "
                                "find tracks that follow your current set.")

    # -- genres -------------------------------------------------------------
    def _load_genres(self):
        if self._genres_worker and self._genres_worker.isRunning():
            return
        self._genres_worker = GenresWorker(self.client)
        self._genres_worker.done.connect(self._genres_loaded)
        self._genres_worker.failed.connect(self._search_failed)
        self._genres_worker.start()

    def _genres_loaded(self, genres):
        keep = self.genre_combo.currentText()
        self.genre_combo.blockSignals(True)
        self.genre_combo.clear()
        self.genre_combo.addItem("any genre", None)
        for g in sorted(genres, key=lambda g: g.get("name", "")):
            if g.get("id") and g.get("name"):
                self.genre_combo.addItem(g["name"], g["id"])
                self._genre_ids[g["name"].lower()] = g["id"]
        i = self.genre_combo.findText(keep)
        self.genre_combo.setCurrentIndex(max(i, 0))
        self.genre_combo.blockSignals(False)

    def _genre_id(self, name):
        return self._genre_ids.get((name or "").lower())

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
        self._search.failed.connect(self._search_failed)
        self._search.start()

    def _search_failed(self, msg):
        low = msg.lower()
        if ("unauthorized" in low or "token expired" in low
                or "not authenticated" in low):
            self._session_expired()
        else:
            self.status.setText("search failed: " + msg)

    def _session_expired(self):
        """The stored token is dead and refresh failed - drop it so the
        login row reappears instead of a dead-end error."""
        self.client.auth.clear()
        self._refresh_auth()
        self.status.setText(
            "Your Beatport session expired and couldn't be refreshed. "
            "Log in again below (same username/password as beatport.com), "
            "then re-run the search.")

    def search(self):
        self._gap_pair = None
        self._variety_cap = None      # manual search: show what matches
        q = self.query.text().strip()
        gid = self.genre_combo.currentData()
        f = self._bpm_filters()
        if gid:
            f["genre_id"] = gid
        if q:
            self._run_search([(q, f)], "searching Beatport...")
        elif gid:
            self._run_search(
                [(None, f)],
                f"browsing {self.genre_combo.currentText()}"
                + (" in your BPM window" if ("bpm_low" in f
                                             or "bpm_high" in f) else "")
                + "...")
        else:
            self.status.setText("type a search, or pick a genre for its "
                                "top-100 chart.")

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

    def _last_set_tracks(self, n=3):
        entries = getattr(self.planner.set_tab, "entries", []) or []
        by_id = {t.id: t for t in self.planner.library}
        return [by_id[e["track_id"]] for e in entries[-n:]
                if e["track_id"] in by_id]

    def discover_for_set(self):
        self._gap_pair = None
        t = self._cur_out_track()
        if t is None:
            self.status.setText("your set is empty - add a track first, then "
                                "'For my set' finds what follows it.")
            return
        # Cast a wider net: one query per genre we know for the anchor (up to
        # 3), each tempo-boxed to +/-8%, merged. Then auto-enable the
        # harmonic + fit sort so only in-key, mixable tracks surface, best
        # first (styles, not names), then narrows: GENRE CHARTS fuzzy-
        # mapped from the last 3 set tracks' genres (MusicBrainz names
        # like 'downtempo' rarely EXACTLY match Beatport charts like
        # 'Organic House / Downtempo' - the old exact mapping failed and
        # everything degraded to artist text searches; user-reported as
        # 'mostly songs by the same authors'), then the anchor's LABEL
        # (labels are style-tight in electronic music), and the artist as
        # ONE query, last.
        f = self._tempo_filter(t.bpm)
        names, gids = [], []
        for tr in (self._last_set_tracks(3) or [t]):
            for g in self._gap_genre_names(tr):
                if g.lower() not in {n.lower() for n in names}:
                    names.append(g)
        for nm in names:
            for gid in self._genre_ids_fuzzy(nm):
                if gid not in gids:
                    gids.append(gid)
        queries = [(None, {**f, "genre_id": g}) for g in gids[:4]]
        label = ((getattr(t, "enrichment", None) or {}).get("label")
                 or "").strip()
        if label:
            queries.append((label, dict(f)))
        if t.artist:
            queries.append((t.artist, dict(f)))
        if not queries:
            queries = [(g, dict(f)) for g in names[:2]] \
                or [(t.title, dict(f))]
        self._variety_cap = 2            # discovery = breadth, not one name
        self.query.setText("")
        self.harmonic.setChecked(True)
        self.sort_fit.setChecked(True)
        self._run_search(
            queries, f"following '{t.title[:24]}' ({t.bpm:.0f} bpm, "
            f"{t.camelot}): {len(gids[:4])} genre charts"
            + (" + label" if label else "") + " + artist...")

    def more_like_selected(self):
        self._gap_pair = None
        r = self._selected_row()
        if not r:
            self.status.setText("select a result first.")
            return
        f = self._tempo_filter(r["bpm"])
        gids = self._genre_ids_fuzzy(r.get("genre"))
        queries = [(None, {**f, "genre_id": g}) for g in gids[:2]]
        if r.get("artist"):
            queries.append((r["artist"], dict(f)))
        if not queries:
            queries = [(r.get("genre") or r["title"], f)]
        self._variety_cap = 3
        self._run_search(queries, f"more like '{r['title'][:30]}'...")

    # -- gap shopping (Set tab hands us a flagged seam) ----------------------
    def shop_gap(self, a, b, label):
        """Shop Beatport for a CONNECTOR between two set neighbours whose
        seam the compiler flagged: genre charts drawn from both tracks'
        genres, BPM boxed to what is tempo-reachable from BOTH sides, and
        results fit-ranked as bridges (fit_between - both directions must
        work)."""
        if not self.client.available():
            self.status.setText("sign in to Beatport first (below).")
            return
        self._gap_pair = (a, b)
        self._variety_cap = 2
        self.fit_only.setChecked(False)
        self.harmonic.setChecked(False)
        self.sort_fit.setChecked(True)
        lo, hi = BP.connector_bpm_window(a, b)
        f = {"bpm_low": int(lo), "bpm_high": int(hi + 0.999)}
        gids = []
        for name in self._gap_genre_names(a) | self._gap_genre_names(b):
            for gid in self._genre_ids_fuzzy(name):
                if gid not in gids:
                    gids.append(gid)
        queries = [(None, {**f, "genre_id": g}) for g in gids[:4]]
        if not queries:
            # No genre mapping: text-search both artists in the BPM box.
            queries = [(q, dict(f))
                       for q in {a.artist, b.artist} if q]
        if not queries:
            self.status.setText("gap shop: no genre or artist to search by.")
            self._gap_pair = None
            return
        self.query.setText("")
        self._run_search(
            queries, f"shopping the gap {label}: connectors "
            f"{int(lo)}-{int(hi)} bpm, fit-ranked as bridges...")

    @staticmethod
    def _gap_genre_names(t):
        names = {g.strip() for g in getattr(t, "genres", []) or []
                 if g and g.strip()}
        fg = getattr(t, "file_genre", "") or ""
        for part in fg.replace("/", ",").replace(";", ",").split(","):
            if part.strip():
                names.add(part.strip())
        return names

    def _genre_ids_fuzzy(self, name):
        """Beatport genre ids for a free-text genre name: exact match
        first, else substring containment either way ('house' hits a few
        charts - shortest names win, callers cap the total)."""
        low = (name or "").lower().strip()
        if not low:
            return []
        if low in self._genre_ids:
            return [self._genre_ids[low]]
        hits = [(len(k), gid) for k, gid in self._genre_ids.items()
                if low in k or k in low]
        return [gid for _, gid in sorted(hits)[:2]]

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
            if self._gap_pair is not None:
                # Bridge mode: rank as a CONNECTOR between the two flagged
                # set neighbours, not against the set's last track.
                ft = BP.fit_between(self._gap_pair[0], self._gap_pair[1],
                                    ghost)
            else:
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

    def _owned(self, r):
        """Already in the library? Matched by song-identity (title root)
        plus loose artist-token overlap, so 'Track (Extended Mix)' on
        Beatport matches the library's 'Track (Original Mix)'."""
        from lib.dj.brain import _title_root
        lib = self.planner.library
        if self._owned_idx is None or self._owned_idx_n != len(lib):
            idx = {}
            for t in lib:
                root = _title_root(t.title) or (t.title or "").lower()
                idx.setdefault(root, set()).update(
                    w for w in (t.artist or "").lower().replace(",", " ")
                    .split() if len(w) > 2)
            self._owned_idx = idx
            self._owned_idx_n = len(lib)
        root = _title_root(r.get("title") or "") \
            or (r.get("title") or "").lower()
        toks = self._owned_idx.get(root)
        if toks is None:
            return False
        rt = {w for w in (r.get("artist") or "").lower().replace(",", " ")
              .split() if len(w) > 2}
        return not toks or not rt or bool(toks & rt)

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
        if self.hide_owned.isChecked():
            rows = [r for r in rows if not self._owned(r)]
        if self.sort_fit.isChecked():
            rows.sort(key=self._fit_key, reverse=True)
        if self._variety_cap:
            # Discovery modes: breadth over one prolific name - keep only
            # the best N per artist (rows are already fit-sorted).
            seen, capped = {}, []
            for r in rows:
                k = (r.get("artist") or "").lower()
                if seen.get(k, 0) >= self._variety_cap:
                    continue
                seen[k] = seen.get(k, 0) + 1
                capped.append(r)
            rows = capped
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

    def _step_preview(self, d):
        """Hop the selection ±1 result row and preview it - rapid
        crate-digging without touching the table."""
        n = self.table.rowCount()
        if not n:
            return
        row = self.table.currentRow()
        row = (row + d) % n if row >= 0 else 0
        self.table.selectRow(row)
        if self._preview and self._preview.isRunning():
            self.status.setText("still loading the previous preview - "
                                "press again in a moment")
            return
        self.preview()

    def _preview_dur(self):
        p = self.player
        return (len(p.samples) / 44100.0
                if p.samples is not None and len(p.samples) else 0.0)

    def _seek_frac(self, v):
        d = self._preview_dur()
        if d > 0:
            self.player.seek(v / 1000.0 * d)

    def _seek_released(self):
        self._seek_drag = False
        self._seek_frac(self.seek_slider.value())

    @staticmethod
    def _mmss(t):
        return f"{int(t // 60)}:{int(t % 60):02d}"

    def _preview_tick(self):
        d = self._preview_dur()
        if d <= 0:
            self.prev_lbl.setText("-:-- / -:--")
            if not self._seek_drag:
                self.seek_slider.setValue(0)
            return
        t = self.player.time_s()
        self.prev_lbl.setText(f"{self._mmss(t)} / {self._mmss(d)}")
        if not self._seek_drag:
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(int(t / d * 1000))
            self.seek_slider.blockSignals(False)

    # -- wishlist -----------------------------------------------------------
    def add_wishlist(self):
        r = self._selected_row()
        if r:
            added = self.wishlist.add(r)
            self._refresh_wishlist()
            self.status.setText(("added to wishlist: " if added else
                                 "already on wishlist: ") + r["title"])

    @staticmethod
    def _price_of(it):
        p = str(it.get("price") or "").replace("$", "").replace("€", "")
        try:
            return float(p)
        except ValueError:
            return 0.0

    def _refresh_wishlist(self):
        """Rebuild the list, preserving the selection by bp_id (rows shift
        when items are removed, so never carry selection by index)."""
        keep = set(self._wish_selected_ids())
        self.wish_list.blockSignals(True)
        self.wish_list.clear()
        total = 0.0
        for it in self.wishlist.items:
            row = QListWidgetItem(
                f"{it['title'][:28]} — {it['artist'][:18]}  "
                f"{it.get('bpm', 0):.0f} {it.get('camelot', '')}  "
                f"{it.get('price') or ''}")
            row.setData(Qt.ItemDataRole.UserRole, it.get("bp_id"))
            self.wish_list.addItem(row)
            if it.get("bp_id") in keep:
                row.setSelected(True)
            total += self._price_of(it)
        self.wish_list.blockSignals(False)
        self._wish_total = total
        self._wish_sel_changed()

    def _wish_selected_ids(self):
        return [i.data(Qt.ItemDataRole.UserRole)
                for i in self.wish_list.selectedItems()
                if i.data(Qt.ItemDataRole.UserRole) is not None]

    def _wish_sel_changed(self):
        """Selection drives both button labels; any change disarms a
        pending confirm so a stale 'Confirm' can't fire on a new set."""
        self._wish_disarm()
        n = len(self.wish_list.selectedItems())
        total = getattr(self, "_wish_total", 0.0)
        self.wish_lbl.setText(
            f"Wishlist — {len(self.wishlist.items)} tracks"
            + (f", ~${total:.2f}" if total else "")
            + (f"   ({n} selected)" if n else ""))
        self.wish_rm_btn.setText(f"Remove ({n})" if n else "Remove")
        self.wish_rm_btn.setEnabled(bool(n))
        self.wish_open_btn.setText(
            f"Open {n} on Beatport" if n else "Open all on Beatport")

    def _wish_disarm(self):
        self._wish_armed = None
        if getattr(self, "wish_rm_btn", None):
            n = len(self.wish_list.selectedItems())
            self.wish_rm_btn.setText(f"Remove ({n})" if n else "Remove")
        if getattr(self, "wish_open_btn", None) and \
                getattr(self, "_open_armed", None):
            self._open_armed = None
            n = len(self.wish_list.selectedItems())
            self.wish_open_btn.setText(
                f"Open {n} on Beatport" if n else "Open all on Beatport")

    def _wish_remove(self):
        """Two-step inline confirm (no modal dialogs): first click arms the
        button, second commits. Auto-disarms after 5s or on any selection
        change."""
        ids = self._wish_selected_ids()
        if not ids:
            self.status.setText("select one or more wishlist tracks first "
                                "(ctrl/shift-click, or ctrl+A for all)")
            return
        if self._wish_armed != tuple(ids):
            self._wish_armed = tuple(ids)
            self.wish_rm_btn.setText(f"Confirm remove {len(ids)}?")
            self.status.setText(f"click again to remove {len(ids)} track(s) "
                                "from the wishlist")
            QTimer.singleShot(5000, self._wish_disarm)
            return
        removed = self.wishlist.remove_many(ids)
        self._wish_armed = None
        self._refresh_wishlist()
        self.status.setText(f"removed {removed} track(s) from the wishlist")

    def _wish_open(self):
        """Open the selection (or everything when nothing is selected).
        Opening many tabs at once gets the same two-step confirm - the
        wishlist can hold hundreds of tracks."""
        ids = self._wish_selected_ids()
        if not ids:
            # A None id would make open_in_browser open EVERYTHING.
            ids = [it.get("bp_id") for it in self.wishlist.items
                   if it.get("bp_id") is not None]
        if not ids:
            return
        if len(ids) > 8 and self._open_armed != tuple(ids):
            self._open_armed = tuple(ids)
            self.wish_open_btn.setText(f"Confirm open {len(ids)} tabs?")
            self.status.setText(f"that opens {len(ids)} browser tabs - "
                                "click again to confirm")
            QTimer.singleShot(5000, self._wish_disarm)
            return
        self._open_armed = None
        for b in ids:
            self.wishlist.open_in_browser(b)
        self._wish_sel_changed()
        self.status.setText(f"opened {len(ids)} track(s) on Beatport")

    def _wish_open_one(self, item):
        b = item.data(Qt.ItemDataRole.UserRole)
        if b is not None:
            self.wishlist.open_in_browser(b)

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
