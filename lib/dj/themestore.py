"""SAVED THEMES and "A NIGHT LIKE THESE" (docs/DJ_SETS.md sections 2 and 5; 2026-10-29). No Qt, nothing heavy at the
top: the planner app's NIGHT tab and the planner's own process both import this.

  the store     the presets stay in code (lib/dj/setplanner.PRESETS) and are never deleted; the themes you SAVE live in
                the library DB (schema v17: LibraryDB.save_theme / load_theme / delete_theme / list_themes), so they
                travel with the music folder. A theme is the set planner's object: name, crate, leans, chapters,
                keys_mode, plus the night's seed ("night") and its length ("length"), so a saved theme plans the same
                night again.
  like these    infer(): the theme of a night like 2 to 5 example records, measured against the library it will play
                from. The crate is each measure's range across the examples plus a margin (tempo +-4 %, the energy rank
                +-0.10), widened step by step until it holds about twice the records the night needs - every widening
                is kept and shown. Leans are weighted by their LIFT over the library ("sexy" is on 77 % of the records:
                it says almost nothing). Vocals come from the stems' vocal envelope (the planner's `sung`, the player's
                -24 dB test), never the ML vocal axis. A narrow energy spread gives one flat chapter, a wide one a rise
                from the lowest example to the highest. The example ids are stored with the theme and the theme is
                inferred again on load, because every rank is relative to the library it was measured in - unless you
                edited it after the inference (then your edits are the theme). An example without stems says nothing
                about vocals (none measured: vocals "any"; the count is kept as inferred.vox_unknown).
                The crate's count toward its target leaves out the records the night cannot play (`banned`: played
                tonight, not tonight). Where the widening stopped is kept as inferred.ladder: when the night planned
                from an unedited inferred theme does not reach its length, the set planner climbs on from there
                (widen(), 2026-10-30). (2026-10-30) Your folders and setlists are words too ("@groovy"): one becomes a
                lean only when its lift says it is informative, as any word; it never narrows the crate. Where the
                examples agree, the chapter gets the new courses: mode (every example minor, or every one major) and
                percussive (their ranks within AGREE_SPREAD).
"""
import json
import math

LIKE_TEMPO = 0.04            # the crate's tempo margin around the examples (DJ_SETS.md: +-4 %)
LIKE_RANK = 0.10             # ...and the energy rank's (+-0.10)
NEED_PLAY_S = 45.0           # a record plays its minimum and about this much more (its way out): the night's need in records
CRATE_X = 2.0                # the crate should hold about twice what the night needs
FLAT_SPREAD = 0.15           # the examples' energy spread under this: one flat chapter; over it: a rise
AGREE_SPREAD = 0.25          # the examples' percussive ranks within this of each other: the chapter gets a percussive course
LEAN_MAX = 8
LEAN_MIN_W = 0.2             # a lean lighter than this says nothing (a word most of the library carries)
# the widenings, in order, each kept only while the crate is still short of its target
WIDEN = (("tempo", 0.06), ("rank", 0.15), ("tempo", 0.08), ("rank", 0.20), ("vocals", "any"), ("genres", None),
         ("tempo", 0.12), ("rank", 0.30), ("tempo", 0.16), ("rank", 0.50))


def _copy(x):
    return json.loads(json.dumps(x))


# ---- the store ------------------------------------------------------------------------------------------------------
def preset_names():
    from lib.dj.setplanner import preset_names as _pn
    return _pn()


def get_preset(name):
    from lib.dj.setplanner import get_preset as _gp
    return _gp(name)


def is_preset(name):
    return str(name or "").strip().lower() in {n.lower() for n in preset_names()}


def saved(db):
    """[{name, example_ids, updated_at}] of the saved themes (none when the DB is older than v17 or unreadable)."""
    try:
        return db.list_themes()
    except Exception:
        return []


def load(db, name):
    """A saved theme as a dict (its examples under "examples"), or None."""
    try:
        row = db.load_theme(name)
    except Exception:
        return None
    if row is None:
        return None
    th = row.get("theme") if isinstance(row, dict) else None
    if not isinstance(th, dict):
        return None                                   # (a row whose JSON is not a theme: unreadable, never a crash)
    th["name"] = row["name"]
    if row.get("example_ids"):
        th["examples"] = [int(i) for i in row["example_ids"]]
    return th


def exists(db, name):
    """A saved theme of exactly this name (the presets are not rows)."""
    return any(r.get("name") == str(name or "").strip() for r in saved(db))


def free_name(db, base):
    """`base`, or `base 2`, `base 3`... - the first name no saved theme or preset holds (SAVE's default never
    silently replaces an older theme)."""
    base = str(base or "my night").strip() or "my night"
    taken = {r.get("name") for r in saved(db)}
    if base not in taken and not is_preset(base):
        return base
    stem = base[:-1].rstrip() if base.endswith(")") else base
    for k in range(2, 100):
        nm = f"{stem} {k})" if base.endswith(")") else f"{base} {k}"
        if nm not in taken and not is_preset(nm):
            return nm
    return base


SAVE_BUSY_MS = 1000          # a write waits at most this long for another writer (the show, a scan): the Qt thread is never held 5 s


def save(db, name, theme):
    """Save `theme` under `name`. A preset's name is refused (the presets stay as they are) - unless a saved theme of
    that name already exists (a preset added later with your theme's name: yours can still be saved and deleted).
    Returns (ok, words); never raises (a locked or unwritable library is words, not a crash of the planner app)."""
    name = str(name or "").strip()
    if not name:
        return False, "a theme needs a name"
    had = exists(db, name)
    if is_preset(name) and not had:
        return False, f"{name} is a preset: save it under another name"
    th = _copy(theme)
    th["name"] = name
    conn = getattr(db, "conn", None)
    try:
        if conn is not None:
            try:
                conn.execute(f"PRAGMA busy_timeout = {int(SAVE_BUSY_MS)}")
            except Exception:
                conn = None
        db.save_theme(name, th, th.get("examples") or None)
    except Exception as e:  # noqa: BLE001
        try:
            db.conn.rollback()                        # (a write half done is not left open on a shared connection)
        except Exception:
            pass
        return False, f"not saved: {type(e).__name__}: {e}"
    finally:
        if conn is not None:
            try:
                conn.execute("PRAGMA busy_timeout = 5000")     # (sqlite3's own default, for every other reader of the connection)
            except Exception:
                pass
    return True, (f"replaced: {name}" if had else f"saved: {name}")


def delete(db, name):
    """Delete a saved theme (only the themes table: a preset of the same name is never touched)."""
    try:
        return bool(db.delete_theme(name))
    except Exception:
        return False


def needs_reinfer(theme):
    """A "like these" theme is inferred again on load unless it was edited after the inference."""
    return bool(theme and theme.get("examples") and not (theme.get("inferred") or {}).get("edited"))


# ---- a night like these ---------------------------------------------------------------------------------------------
def need_records(length_min, play_min_s):
    return max(4, int(math.ceil(60.0 * float(length_min) / (float(play_min_s) + NEED_PLAY_S))))


def _vocals_rule(sung):
    """The narrowest crate vocal rule every example passes (setplanner.VOCALS_OK)."""
    if max(sung) <= 0.05:
        return "none"
    if max(sung) <= 0.35:
        return "some"
    if min(sung) >= 0.25:
        return "lots"
    return "any"


def _short(title, n=18):
    import re
    t = re.sub(r"\s*[\(\[][^\)\]]*(mix|edit|version|remaster)[^\)\]]*[\)\]]\s*$", "", str(title or "?"), flags=re.I) or str(title or "?")
    return t if len(t) <= n else t[:n - 1].rstrip() + "…"


def _ladder_crate(lad):
    """The crate rule at a rung of the like-these ladder: the examples' tempo and energy-rank span plus its margins, its
    vocal rule and genres (`lad`: {bpm: [lo, hi], e: [lo, hi], tempo, rank, vocals, genres})."""
    (b0, b1), (e0, e1) = lad["bpm"], lad["e"]
    return {"source": None, "genresIn": list(lad["genres"]), "genresOut": [], "moodsIn": [], "tagsIn": [], "tagsOut": [],
            "bpm": [math.floor(b0 * (1.0 - lad["tempo"]) * 10.0) / 10.0, math.ceil(b1 * (1.0 + lad["tempo"]) * 10.0) / 10.0],
            "energy": [max(0, int(math.floor(100.0 * (e0 - lad["rank"])))), min(100, int(math.ceil(100.0 * (e1 + lad["rank"]))))],
            "vocals": lad["vocals"]}


def _ladder_step(lad, what, val):
    """One WIDEN step applied to the ladder state in place: its words, or None when it changes nothing (vocals or genres
    already open)."""
    if what == "tempo":
        lad["tempo"] = val
        return f"tempo ±{val * 100:.0f} %"
    if what == "rank":
        lad["rank"] = val
        return f"energy ±{val:.2f}"
    if what == "vocals":
        if lad["vocals"] == "any":
            return None
        lad["vocals"] = "any"
        return "vocals any"
    if not lad["genres"]:
        return None
    lad["genres"] = []
    return "any genre"


def _available(ids, banned):
    return sum(1 for i in ids if i not in banned) if banned else len(ids)


def infer(facts, example_ids, crate_ids, length_min, play_min_s, name=None, banned=()):
    """The theme of a night like the examples. `facts`: {id: record facts} as the set planner reads them (bpm, e = the
    energy rank, words, genres, sung = the share of the record that sings by its stems, title); `crate_ids(crate)`: the
    planner's own crate rule (a list of ids); `banned`: records the night cannot play (played tonight, not tonight) - the
    crate's count toward its target leaves them out (2026-10-30: examples played tonight were counted, then banned by the
    plan). Returns {theme, widened: [{what, records, from, available}], crate, available, need, target, missing: [titles of
    examples no crate can hold], lifts: {word: [share of the examples, share of the library]}}. The theme's
    inferred.ladder is where the widening stopped: the planner climbs on from there (widen) when the night does not
    finish."""
    ex = [facts[i] for i in dict.fromkeys(int(i) for i in example_ids) if i in facts]
    if not ex:
        return {"theme": None, "error": "none of these records is in the library", "widened": [], "crate": 0}
    banned = {int(i) for i in banned or ()}
    need = need_records(length_min, play_min_s)
    target = int(math.ceil(CRATE_X * need))
    bpms = [f.bpm for f in ex]
    es = [f.e for f in ex]
    # vocals only from the examples the stems measured: a record with no envelope is "nothing known", not silent (its
    # sung = 0.0 would make vocal records read as instrumentals and narrow the crate to vocals "none")
    measured = [f for f in ex if getattr(f, "vox", None) is not None and len(f.vox)]
    m_ids = {id(f) for f in measured}
    vox_unknown = [_short(f.title) for f in ex if id(f) not in m_ids]
    st = {"bpm": [min(bpms), max(bpms)], "e": [min(es), max(es)], "tempo": LIKE_TEMPO, "rank": LIKE_RANK,
          "vocals": _vocals_rule([f.sung for f in measured]) if measured else "any",
          "genres": [] if any(not f.genres for f in ex) else sorted(set().union(*[f.genres for f in ex])), "at": 0}

    def crate():
        return _ladder_crate(st)
    ids = list(crate_ids(crate()))
    widened = []
    for k, (what, val) in enumerate(WIDEN):
        if _available(ids, banned) >= target:
            break
        st["at"] = k + 1
        words = _ladder_step(st, what, val)
        if words is None:
            continue
        before = len(ids)
        ids = list(crate_ids(crate()))
        widened.append({"what": words, "records": len(ids), "from": before, "available": _available(ids, banned)})
    in_crate = set(ids)
    missing = [f.title for f in ex if f.id not in in_crate]
    # LEANS BY LIFT: a word on most of the examples counts as much as it is rarer in the library than among them
    n_lib = max(1, len(facts))
    lib_n = {}
    for f in facts.values():
        for w in f.words:
            lib_n[w] = lib_n.get(w, 0) + 1
    ex_n = {}
    for f in ex:
        for w in f.words:
            ex_n[w] = ex_n.get(w, 0) + 1
    at_least = max(1, int(math.ceil(len(ex) / 2.0)))
    scored, lifts = [], {}
    for w, k in ex_n.items():
        if k < at_least:
            continue
        p_ex, p_lib = k / len(ex), max(lib_n.get(w, 0), 1) / n_lib
        wt = min(2.0, 0.5 * p_ex * math.log2(max(p_ex / p_lib, 1.0)))
        lifts[w] = [round(p_ex, 2), round(p_lib, 3)]
        if wt >= LEAN_MIN_W:
            scored.append((wt, w))
    scored.sort(key=lambda x: (-x[0], x[1]))
    leans = {w: round(wt, 2) for wt, w in scored[:LEAN_MAX]}
    # THE STORY from the examples' energy spread; the tempo course as the examples sit among the crate's tempos
    cb = sorted(facts[i].bpm for i in ids) or sorted(bpms)

    def pct(b):
        lo = sum(1 for x in cb if x < b)
        return round(min(1.0, max(0.0, lo / max(1, len(cb) - 1))), 2)
    lo_f, hi_f = min(ex, key=lambda f: f.e), max(ex, key=lambda f: f.e)
    if hi_f.e - lo_f.e < FLAT_SPREAD:
        m = round(sum(es) / len(es), 3)
        q = round(sum(pct(b) for b in bpms) / len(bpms), 2)
        chapters = [{"name": "The night", "share": 1.0, "energy": [m, m], "tempo": [q, q], "leans": [], "dims": {}}]
        story = f"one flat chapter at {m:.2f} (the examples sit within {hi_f.e - lo_f.e:.2f} of each other)"
    else:
        chapters = [{"name": "Rise", "share": 1.0, "energy": [round(lo_f.e, 3), round(hi_f.e, 3)],
                     "tempo": [pct(lo_f.bpm), pct(hi_f.bpm)], "leans": [], "dims": {}}]
        story = f"a rise from {_short(lo_f.title)} ({lo_f.e:.2f}) to {_short(hi_f.title)} ({hi_f.e:.2f})"
    # THE TWO STORY CONTROLS (2026-10-30), proposed only where the examples agree: mode when every example reads minor
    # (or every one major) by the key, percussive when their ranks sit within AGREE_SPREAD; a flat course at their mean
    agree = {}
    if len(ex) >= 2:
        md = [(getattr(f, "dims", None) or {}).get("mode") for f in ex]
        if all(v is not None for v in md) and (all(v < 0.5 for v in md) or all(v > 0.5 for v in md)):
            agree["mode"] = round(sum(md) / len(md), 2)
        pc = [(getattr(f, "dims", None) or {}).get("percussive") for f in ex]
        if all(v is not None for v in pc) and max(pc) - min(pc) <= AGREE_SPREAD:
            agree["percussive"] = round(sum(pc) / len(pc), 2)
    if agree:
        chapters[0]["dims"] = {k: [v, v] for k, v in agree.items()}
        story += "; " + ", ".join(("all minor" if v < 0.5 else "all major") if k == "mode" else f"percussive {v:.2f}" for k, v in agree.items())
    titles = [_short(f.title) for f in ex]
    nm = name or ("Like " + ", ".join(titles[:2]) + (f" +{len(titles) - 2}" if len(titles) > 2 else ""))
    theme = {"name": nm, "blurb": f"A night like {', '.join(titles)}: {story}.", "crate": crate(), "leans": leans,
             "chapters": chapters, "keys_mode": "veto", "length": float(length_min),
             "inferred": {"widened": widened, "need": need, "target": target, "crate": len(ids), "story": story, "missing": missing,
                          "vox_unknown": len(vox_unknown), "vox_unknown_titles": vox_unknown, "ladder": _copy(st)}}
    return {"theme": theme, "widened": widened, "crate": len(ids), "available": _available(ids, banned), "need": need,
            "target": target, "missing": missing, "lifts": lifts}


def can_widen(theme):
    """An inferred theme the planner may widen on its own: it has its examples and its ladder, and you did not edit it
    (an edited theme or a preset is never widened behind your back)."""
    return needs_reinfer(theme) and isinstance((theme.get("inferred") or {}).get("ladder"), dict)


def widen(theme, crate_ids, banned=()):
    """THE NEXT RUNG for an unedited "like these" theme whose night does not finish (2026-10-30; the set planner calls it,
    lib/dj/setplanner.Planner._plan): the WIDEN steps after the one infer() stopped at, applied in turn until the crate
    holds more records the night can still play (not in `banned`) than before - a step that adds none is folded into the
    next. Every step is kept, as infer() keeps them, marked by "plan". Returns (the widened theme, [its steps]), or
    (None, []) when the ladder has run out (or the theme is not one to widen)."""
    if not can_widen(theme):
        return None, []
    banned = {int(i) for i in banned or ()}
    lad = _copy(theme["inferred"]["ladder"])
    ids0 = list(crate_ids(theme.get("crate") or _ladder_crate(lad)))
    av0 = _available(ids0, banned)
    n, steps = len(ids0), []
    while int(lad.get("at") or 0) < len(WIDEN):
        what, val = WIDEN[int(lad.get("at") or 0)]
        lad["at"] = int(lad.get("at") or 0) + 1
        words = _ladder_step(lad, what, val)
        if words is None:
            continue
        ids = list(crate_ids(_ladder_crate(lad)))
        av = _available(ids, banned)
        steps.append({"what": words, "records": len(ids), "from": n, "available": av, "by": "plan"})
        n = len(ids)
        if av > av0:
            th = _copy(theme)
            th["crate"] = _ladder_crate(lad)
            inf = th["inferred"]
            inf["ladder"] = lad
            inf["widened"] = list(inf.get("widened") or []) + steps
            inf["crate"] = len(ids)
            return th, steps
    return None, []
