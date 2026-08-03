"""What the ratings have taught, across every Seam Lab session.

Reads logs/seam_lab_ratings.jsonl (good, passable AND bad, with each
seam's plan context) and answers the question the treadmill exists to
answer: WHAT is failing, and HOW.

Three layers:

  * `enrich` fills each row's diagnostic features. Rows logged before a
    field existed are back-filled by joining the track ids against the
    live library, so an old dataset still yields key fit, tempo gap,
    grid confidence and stem availability.
  * `findings` ranks every feature bucket by how far its good-share
    departs from the overall baseline, weighted by how much evidence
    stands behind it - that ranking is the "what is failing" list, and
    each failing style then gets its own worst condition, the "how".
  * `report_html` lays it out.

Statistical honesty is the whole point of this being useful: every
number carries its n, buckets thinner than MIN_BUCKET are dropped rather
than shown as confident percentages, and `passable` is counted in totals
but never votes in a good-share (it is the explicit "no opinion").
"""
import json
import math
import os
import time

MIN_BUCKET = 5          # below this a percentage is noise, not a finding
MIN_LIFT = 0.10         # smaller departures from baseline aren't worth a line
THIN = 8                # n below this is flagged as provisional

GOOD_C = "#2e8b57"      # readable on light AND dark themes
BAD_C = "#c0392b"
DIM_C = "#8a8f98"

# (a_id, b_id, rate) -> rhythm terms. Process-lifetime; a Rhythm pass that
# rewrites signatures mid-session won't be picked up until restart.
_RHYTHM_MEMO = {}


def log_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))), "logs",
        "seam_lab_ratings.jsonl")


def cohort_path():
    return os.path.join(os.path.dirname(log_path()), "seam_cohorts.json")


def suspect_before():
    """Ratings older than this were collected under a configuration since
    found to be broken, and are excluded from the CHOICE-level statistics
    (styles, conditions, context model).

    They are NOT excluded from the knob analysis: the execution nudge is
    randomised independently of pair and style, so a bad sampling config
    added noise to those seams but no bias to the nudge/verdict relation -
    throwing that evidence away would cost real listening for nothing.

    Recorded as a timestamp rather than by rewriting the log, so nothing
    is lost if a rating lands while this is being set."""
    try:
        with open(cohort_path(), encoding="utf-8") as f:
            return float(json.load(f).get("suspect_before") or 0.0)
    except (OSError, ValueError, TypeError):
        return 0.0


def mark_suspect_before(ts, note=""):
    os.makedirs(os.path.dirname(cohort_path()), exist_ok=True)
    with open(cohort_path(), "w", encoding="utf-8") as f:
        json.dump({"suspect_before": float(ts), "note": note}, f, indent=2)


def read_ratings(path=None):
    """Every logged verdict, oldest first. Missing/corrupt lines skipped."""
    path = path or log_path()
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("verdict") in ("good", "passable", "bad"):
                    rows.append(r)
    except OSError:
        return []
    return rows


# ---------------------------------------------------------------- enrich ---
def _camelot_compat(a, b):
    if not a or not b:
        return None
    try:
        from lib.dj.brain import camelot_compat
        return camelot_compat(a, b)
    except Exception:
        return None


def enrich(rows, library=None):
    """Add derived diagnostic fields in place-ish (returns new dicts).

    Old rows predate the richer logging; anything recoverable from the
    track ids is joined back in from the library so the whole dataset
    stays analysable instead of only the newest sessions."""
    by_id = {t.id: t for t in (library or [])}
    try:
        from lib.dj.rhythm import seam_rhythm
    except Exception:
        seam_rhythm = None
    out = []
    for r in rows:
        d = dict(r)
        ta, tb = by_id.get(r.get("a_id")), by_id.get(r.get("b_id"))
        if not d.get("rhythm") and seam_rhythm is not None \
                and ta is not None and tb is not None:
            # Rhythm terms are a pure function of the two tracks and the
            # rate, so the WHOLE history gets groove/flam analysis - not
            # just the sessions logged after the field was added. Memoised
            # because this re-runs on the UI thread after every rating and
            # only ONE row is ever new (measured: 600 ms uncached at 3k
            # ratings, which would stall the listen-press-listen loop).
            ck = (r.get("a_id"), r.get("b_id"), round(r.get("rate") or 1.0, 4))
            if ck in _RHYTHM_MEMO:
                d["rhythm"] = _RHYTHM_MEMO[ck]
            else:
                try:
                    d["rhythm"] = _RHYTHM_MEMO[ck] = \
                        seam_rhythm(ta, tb, r.get("rate") or 1.0) or {}
                except Exception:
                    pass
        for side, t in (("a", ta), ("b", tb)):
            if t is None:
                continue
            d.setdefault(f"camelot_{side}", t.camelot)
            d.setdefault(f"bpm_{side}", t.bpm)
            d.setdefault(f"conf_{side}", t.bpm_conf or 0.0)
            d.setdefault(f"stems_{side}", bool(getattr(t, "has_stems",
                                                       False)))
            if d.get(f"title_{side}") is None:
                d[f"title_{side}"] = r.get(f"{side}_title") or t.title
        d["key_fit"] = _camelot_compat(d.get("camelot_a"),
                                       d.get("camelot_b"))
        if d.get("bpm_a") and d.get("bpm_b") and d.get("rate"):
            # The tempo gap the stretch had to close, as the engine saw it.
            d["bpm_gap"] = abs(d["bpm_a"] - d["bpm_b"] * d["rate"])
        rh = d.get("rhythm") or {}
        d["rhythm_score"] = rh.get("score")
        d["flam_ms"] = rh.get("flam_ms")
        d["meter_clash"] = rh.get("meter_clash")
        d["kick_agreement"] = rh.get("kick_agreement")
        out.append(d)
    return out


def _stretch_pct(r):
    return None if r.get("rate") is None else abs(r["rate"] - 1.0) * 100.0


def _band(v, edges, labels):
    if v is None:
        return None
    i = 0
    while i < len(edges) and v >= edges[i]:
        i += 1
    return labels[i]


# (key, human group name, extractor)
FEATURES = [
    ("style", "transition style", lambda r: r.get("style")),
    ("key", "key fit", lambda r: _band(
        r.get("key_fit"), [0.55, 0.9],
        ["clashing keys", "workable keys", "matched keys"])),
    ("stretch", "stretch", lambda r: _band(
        _stretch_pct(r), [1.0, 3.0],
        ["stretch under 1%", "stretch 1-3%", "stretch over 3%"])),
    ("pitch", "pitch shift", lambda r: None if r.get("pitch_st") is None
     else ("pitch shifted" if abs(r.get("pitch_st") or 0) > 0.01
           else "no pitch shift")),
    ("pair", "pair score", lambda r: _band(
        r.get("pair_score"), [0.45, 0.60],
        ["pair score under 0.45", "pair score 0.45-0.60",
         "pair score over 0.60"])),
    ("rhythm", "groove fit", lambda r: _band(
        r.get("rhythm_score"), [0.45, 0.60],
        ["grooves fight", "grooves half-agree", "grooves lock"])),
    ("flam", "drum alignment", lambda r: _band(
        r.get("flam_ms"), [15.0, 80.0],
        ["hits lock (<15ms)", "flam risk (15-80ms)",
         "hits far apart (>80ms)"])),
    ("grid", "grid confidence", lambda r: None if (
        r.get("conf_a") is None or r.get("conf_b") is None) else
     ("precise grids" if min(r["conf_a"], r["conf_b"]) >= 0.7
      else "loose grid")),
    ("beats", "blend length", lambda r: _band(
        r.get("beats"), [17, 33],
        ["short blend (<=16 beats)", "32-beat blend",
         "long blend (64+ beats)"])),
    ("stems", "stems", lambda r: None if r.get("stems_a") is None else
     ("both tracks have stems" if r.get("stems_a") and r.get("stems_b")
      else "one track has stems" if (r.get("stems_a") or r.get("stems_b"))
      else "no stems")),
    ("theme", "theme", lambda r: r.get("theme")),
    ("engine", "stretch engine", lambda r: r.get("engine")),
]


def _rate(rows):
    """(good share among decided, n decided, n total). Passable abstains."""
    good = sum(1 for r in rows if r["verdict"] == "good")
    bad = sum(1 for r in rows if r["verdict"] == "bad")
    n = good + bad
    return (good / n if n else 0.0), n, len(rows)


def group(rows, fn):
    out = {}
    for r in rows:
        k = fn(r)
        if k is not None:
            out.setdefault(k, []).append(r)
    return out


def findings(rows, baseline=None, min_bucket=MIN_BUCKET, min_lift=MIN_LIFT):
    """Every feature bucket whose good-share departs from the baseline,
    ranked by departure x evidence. The 'what is failing' list."""
    base, base_n, _ = _rate(rows) if baseline is None else baseline
    out = []
    for key, gname, fn in FEATURES:
        for bucket, rs in group(rows, fn).items():
            share, n, tot = _rate(rs)
            if n < min_bucket:
                continue
            lift = share - base
            if abs(lift) < min_lift:
                continue
            out.append({"feature": key, "group": gname, "bucket": bucket,
                        "share": share, "n": n, "total": tot, "lift": lift,
                        "weight": abs(lift) * math.sqrt(n)})
    out.sort(key=lambda f: -f["weight"])
    return out


def worst_condition(rows, style_rows, baseline):
    """Within one style's seams, the condition its failures cluster on."""
    fs = findings(style_rows, baseline=baseline, min_bucket=3, min_lift=0.12)
    fs = [f for f in fs if f["lift"] < 0 and f["feature"] != "style"]
    return fs[0] if fs else None


# Conditions worth reporting a style's competence in - the same axes the
# brain now learns per (lib/dj/brain.seam_conditions), so the panel's
# words and the engine's dice agree about WHY and WHERE.
COND_FEATURES = ("grid", "key", "groove", "flam", "stems", "beats",
                 "stretch", "pair")


def style_profile(style_rows, min_n=3):
    """Where a style works and where it fails, on its OWN seams.

    Compared against the style's own average rather than the global one:
    the question is not "is this style good" (the style table answers
    that) but "given that I am using it, when does it work" - which is
    what turns a bad average into something actionable, or confirms that
    it fails everywhere."""
    own, own_n, _ = _rate(style_rows)
    good, bad = [], []
    for key, gname, fn in FEATURES:
        if key not in COND_FEATURES:
            continue
        for bucket, rs in group(style_rows, fn).items():
            share, n, _tot = _rate(rs)
            if n < min_n:
                continue
            lift = share - own
            entry = {"bucket": bucket, "share": share, "n": n, "lift": lift}
            if lift >= 0.15 or (own_n and share >= 0.75 and own < 0.75):
                good.append(entry)
            elif lift <= -0.15 or (share <= 0.25 and own > 0.25):
                bad.append(entry)
    good.sort(key=lambda e: (-e["lift"], -e["n"]))
    bad.sort(key=lambda e: (e["lift"], -e["n"]))
    return good[:3], bad[:3], own, own_n


def track_trouble(rows, min_n=3):
    """Tracks that keep producing bad seams, and ones that never do."""
    acc = {}
    for r in rows:
        for side in ("a", "b"):
            tid = r.get(f"{side}_id")
            if tid is None:
                continue
            title = r.get(f"{side}_title") or r.get(f"title_{side}") or "?"
            e = acc.setdefault(tid, {"title": title, "good": 0, "bad": 0})
            if r["verdict"] == "good":
                e["good"] += 1
            elif r["verdict"] == "bad":
                e["bad"] += 1
    rated = [e for e in acc.values() if e["good"] + e["bad"] >= min_n]
    for e in rated:
        e["n"] = e["good"] + e["bad"]
        e["share"] = e["good"] / e["n"]
    hard = sorted([e for e in rated if e["share"] <= 0.34],
                  key=lambda e: (e["share"], -e["n"]))
    easy = sorted([e for e in rated if e["share"] >= 0.75],
                  key=lambda e: (-e["share"], -e["n"]))
    return hard, easy


def sessions(rows):
    """Per-day totals, newest first - is it getting better or worse?"""
    by_day = {}
    for r in rows:
        if not r.get("t"):
            continue
        day = time.strftime("%Y-%m-%d", time.localtime(r["t"]))
        by_day.setdefault(day, []).append(r)
    out = []
    for day in sorted(by_day, reverse=True):
        share, n, tot = _rate(by_day[day])
        out.append({"day": day, "share": share, "n": n, "total": tot})
    return out


def coverage(rows):
    """How much of the dataset actually carries each diagnostic - old rows
    predate the richer logging, and a finding drawn from 6 of 90 seams
    should say so."""
    n = len(rows)
    fields = (("groove fit", "rhythm_score"), ("drum alignment", "flam_ms"),
              ("key fit", "key_fit"), ("pair score", "pair_score"),
              ("stems", "stems_a"))
    return [(label, sum(1 for r in rows if r.get(key) is not None), n)
            for label, key in fields]


def analyze(rows, library=None):
    """Everything the panel needs, as plain data.

    Choice-level statistics use only the trustworthy cohort; the knob
    analysis (seamtune) is handed the FULL set separately."""
    if not rows:
        return {"n": 0}
    rs_all = enrich(rows, library)
    cut = suspect_before()
    rs = [r for r in rs_all if not cut or (r.get("t") or 0) >= cut]
    excluded = len(rs_all) - len(rs)
    if not rs:
        return {"n": 0, "excluded": excluded, "rows": rs_all}
    base = _rate(rs)
    counts = {v: sum(1 for r in rs if r["verdict"] == v)
              for v in ("good", "passable", "bad")}
    styles = []
    for s, srs in group(rs, lambda r: r.get("style")).items():
        share, n, tot = _rate(srs)
        works, fails, _own, _on = style_profile(srs)
        styles.append({"style": s, "share": share, "n": n, "total": tot,
                       "passable": sum(1 for r in srs
                                       if r["verdict"] == "passable"),
                       "worst": worst_condition(rs, srs, base) if n >= 4
                       else None,
                       "works": works, "fails": fails})
    styles.sort(key=lambda s: (-s["n"], s["style"]))
    pins = [r for r in rs if r.get("want_style")]
    refused = [r for r in pins if r.get("style") != r.get("want_style")]
    why = {}
    for r in refused:
        why[r.get("pin_why") or "gates"] = why.get(
            r.get("pin_why") or "gates", 0) + 1
    quick = [r for r in rs
             if r["verdict"] == "bad" and (r.get("listened_s") or 99) < 8.0]
    slow_bad = [r for r in rs
                if r["verdict"] == "bad" and (r.get("listened_s") or 0) >= 20]
    hard, easy = track_trouble(rs)
    eras = {}
    for r in rs:
        v = r.get("ver")
        if isinstance(v, dict):
            k = f"{v.get('code', '?')}/{v.get('knobs', '?')}"
        else:
            k = v or "unstamped"
        eras[k] = eras.get(k, 0) + 1
    return {"n": len(rs), "counts": counts, "baseline": base,
            "rows": rs_all, "excluded": excluded,
            "eras": eras,
            "findings": findings(rs), "styles": styles,
            "sessions": sessions(rs), "coverage": coverage(rs),
            "pins": len(pins), "refused": len(refused), "why": why,
            "quick_bad": len(quick), "slow_bad": len(slow_bad),
            "hard": hard[:6], "easy": easy[:6]}


# ------------------------------------------------------------------ html ---
def _pct(x):
    return f"{x * 100:.0f}%"


def _colour(share, base):
    return GOOD_C if share >= base + 0.08 else \
        BAD_C if share <= base - 0.08 else DIM_C


def _thin(n):
    return f" <span style='color:{DIM_C}'>(thin)</span>" if n < THIN else ""


def report_html(sm, brain=None):
    """The bottom pane. Sections in order of what you'd act on first."""
    if not sm.get("n"):
        if sm.get("excluded"):
            # There ARE ratings, just none in the trustworthy cohort. The
            # knob section still has evidence and must not be dropped.
            head = (f"<p><b>{sm['excluded']} ratings held back.</b> They "
                    f"were collected under a sampling configuration since "
                    f"found to starve the candidate pool, so their style "
                    f"and condition statistics reflect that rather than "
                    f"your taste. Rate under the fixed configuration and "
                    f"this fills in again.<br><span style='color:{DIM_C}'>"
                    f"Their execution-knob evidence is still counted below "
                    f"— that nudge is randomised independently of pair and "
                    f"style.</span></p>")
        else:
            head = ("<p>No ratings logged yet — every verdict you give "
                    "here becomes the dataset this panel reads. Rate a few "
                    "dozen seams and it will start telling you which "
                    "styles and which conditions are failing.</p>")
        try:
            from tools.dj.planner import seamprobe
            from tools.dj.planner.seamtune import RANGES
            head += seamprobe.report_html(RANGES)
        except Exception:
            pass
        try:
            from tools.dj.planner import seamtune
            head += seamtune.report_html(sm.get("rows") or [])
        except Exception:
            pass
        return head
    base, base_n, _tot = sm["baseline"]
    c = sm["counts"]
    style_mem = getattr(brain, "style_memory", {}) or {}
    cond_mem = getattr(brain, "style_cond_memory", {}) or {}
    h = []
    h.append(
        f"<p><b>{sm['n']} seams rated</b> over {len(sm['sessions'])} "
        f"session day{'s' if len(sm['sessions']) != 1 else ''} &nbsp; "
        f"<span style='color:{GOOD_C}'>{c['good']} good</span> · "
        f"{c['passable']} passable · "
        f"<span style='color:{BAD_C}'>{c['bad']} bad</span> &nbsp;—&nbsp; "
        f"baseline <b>{_pct(base)}</b> good of {base_n} decided verdicts "
        f"<span style='color:{DIM_C}'>(passable abstains)</span></p>")

    # -- what is failing / working -----------------------------------------
    fs = sm["findings"]
    bad_f = [f for f in fs if f["lift"] < 0][:7]
    good_f = [f for f in fs if f["lift"] > 0][:7]
    h.append("<h4 style='margin:6px 0 2px'>What is failing</h4>")
    if bad_f:
        h.append("<table width='100%' cellspacing='0' cellpadding='2'>")
        for f in bad_f:
            h.append(
                f"<tr><td width='42%'>{f['bucket']}"
                f"<span style='color:{DIM_C}'> — {f['group']}</span></td>"
                f"<td width='14%' align='right'><b style='color:{BAD_C}'>"
                f"{_pct(f['share'])}</b> good</td>"
                f"<td width='14%' align='right' style='color:{DIM_C}'>"
                f"{f['n']} rated{_thin(f['n'])}</td>"
                f"<td style='color:{BAD_C}'>{_pct(f['lift'])} vs "
                f"baseline</td></tr>")
        h.append("</table>")
    else:
        h.append(f"<p style='color:{DIM_C}'>Nothing yet departs from the "
                 f"baseline by {int(MIN_LIFT * 100)} points with "
                 f"{MIN_BUCKET}+ ratings behind it — keep rating.</p>")
    h.append("<h4 style='margin:6px 0 2px'>What is working</h4>")
    if good_f:
        h.append("<table width='100%' cellspacing='0' cellpadding='2'>")
        for f in good_f:
            h.append(
                f"<tr><td width='42%'>{f['bucket']}"
                f"<span style='color:{DIM_C}'> — {f['group']}</span></td>"
                f"<td width='14%' align='right'><b style='color:{GOOD_C}'>"
                f"{_pct(f['share'])}</b> good</td>"
                f"<td width='14%' align='right' style='color:{DIM_C}'>"
                f"{f['n']} rated{_thin(f['n'])}</td>"
                f"<td style='color:{GOOD_C}'>+{_pct(f['lift'])} vs "
                f"baseline</td></tr>")
        h.append("</table>")
    else:
        h.append(f"<p style='color:{DIM_C}'>No condition is clearly "
                 f"outperforming yet.</p>")

    # -- the diagnosis: per style, where it works and why it fails ---------
    h.append("<h4 style='margin:8px 0 2px'>Where each style works, and "
             "why it fails</h4>"
             f"<p style='color:{DIM_C};margin:0 0 4px'>Conditions are "
             f"measured against that style's OWN average, so this says "
             f"where to reach for it — not whether it is good. "
             f"<i>Engine ×</i> is what the memory currently does to its "
             f"dice.</p>")
    h.append("<table width='100%' cellspacing='0' cellpadding='2'>")
    for s in sm["styles"]:
        if not s["n"] and not s["works"] and not s["fails"]:
            continue
        mult = style_mem.get(s["style"])
        head = (f"<b>{s['style']}</b> "
                f"<span style='color:{_colour(s['share'], base)}'>"
                f"{_pct(s['share']) if s['n'] else '—'}</span>"
                f"<span style='color:{DIM_C}'> of {s['n']} decided"
                + (f", {s['passable']} passable" if s["passable"] else "")
                + "</span>")
        if mult:
            head += (f" <span style='color:"
                     f"{GOOD_C if mult > 1 else BAD_C}'>engine "
                     f"×{mult:.2f}</span>")
        detail = []
        if s["works"]:
            detail.append(
                f"<span style='color:{GOOD_C}'>works when</span> " +
                " · ".join(f"{e['bucket']} {_pct(e['share'])} of {e['n']}"
                           for e in s["works"]))
        if s["fails"]:
            detail.append(
                f"<span style='color:{BAD_C}'>fails when</span> " +
                " · ".join(f"{e['bucket']} {_pct(e['share'])} of {e['n']}"
                           for e in s["fails"]))
        if not detail:
            detail.append(
                f"<span style='color:{DIM_C}'>" +
                ("no condition separates its wins from its losses yet — "
                 "it reads as uniformly weak so far"
                 if s["n"] >= MIN_BUCKET and s["share"] < 0.4 else
                 "not enough seams yet to say where it works")
                + "</span>")
        h.append(f"<tr><td>{head}<br>&nbsp;&nbsp;"
                 + "<br>&nbsp;&nbsp;".join(detail) + "</td></tr>")
    h.append("</table>")

    h.append("<h4 style='margin:8px 0 2px'>Every style at a glance</h4>")
    h.append("<table width='100%' cellspacing='0' cellpadding='2'>"
             f"<tr style='color:{DIM_C}'><td width='24%'><i>style</i></td>"
             f"<td width='9%' align='right'><i>good</i></td>"
             f"<td width='13%' align='right'><i>rated</i></td>"
             f"<td><i>its failures cluster on</i></td></tr>")
    for s in sm["styles"]:
        if not s["n"]:
            h.append(f"<tr><td>{s['style']}</td><td align='right' "
                     f"style='color:{DIM_C}'>—</td><td align='right' "
                     f"style='color:{DIM_C}'>{s['total']} (all passable)"
                     f"</td><td></td></tr>")
            continue
        w = s["worst"]
        note = (f"{w['bucket']} → {_pct(w['share'])} of {w['n']}"
                if w else f"<span style='color:{DIM_C}'>no clear "
                          f"pattern</span>")
        h.append(
            f"<tr><td>{s['style']}</td>"
            f"<td align='right'><b style='color:"
            f"{_colour(s['share'], base)}'>{_pct(s['share'])}</b></td>"
            f"<td align='right' style='color:{DIM_C}'>{s['n']}"
            f"{_thin(s['n'])}</td><td>{note}</td></tr>")
    h.append("</table>")

    # -- specific tracks ----------------------------------------------------
    if sm["hard"] or sm["easy"]:
        h.append("<h4 style='margin:8px 0 2px'>Specific tracks</h4>")
        if sm["hard"]:
            h.append(f"<p><span style='color:{BAD_C}'>Hard to mix</span> "
                     + " · ".join(
                         f"{e['title'][:34]} ({e['good']}/{e['n']})"
                         for e in sm["hard"]) + "</p>")
        if sm["easy"]:
            h.append(f"<p><span style='color:{GOOD_C}'>Reliable</span> "
                     + " · ".join(
                         f"{e['title'][:34]} ({e['good']}/{e['n']})"
                         for e in sm["easy"]) + "</p>")

    # -- gates, pins, listening behaviour -----------------------------------
    bits = []
    if sm["pins"]:
        wh = ", ".join(f"{k} ×{v}" for k, v in
                       sorted(sm["why"].items(), key=lambda kv: -kv[1])[:3])
        bits.append(f"{sm['refused']}/{sm['pins']} style pins refused by "
                    f"the gates" + (f" ({wh})" if wh else ""))
    if sm["quick_bad"]:
        bits.append(f"{sm['quick_bad']} bad calls inside 8s — obvious "
                    f"failures, audible immediately")
    if sm["slow_bad"]:
        bits.append(f"{sm['slow_bad']} bad calls after 20s+ — these went "
                    f"wrong late in the blend")
    if bits:
        h.append("<h4 style='margin:8px 0 2px'>Gates and listening</h4>"
                 "<p>" + "<br>".join(bits) + "</p>")

    # -- sessions -----------------------------------------------------------
    ss = sm["sessions"]
    if len(ss) > 1:
        h.append("<h4 style='margin:8px 0 2px'>By session</h4><p>"
                 + " · ".join(
                     f"{s['day'][5:]} <b style='color:"
                     f"{_colour(s['share'], base)}'>{_pct(s['share'])}</b>"
                     f"<span style='color:{DIM_C}'>/{s['total']}</span>"
                     for s in ss[:10]) + "</p>")

    # -- what is steering right now ----------------------------------------
    steer = steering_line(brain)
    if steer:
        h.append(f"<h4 style='margin:8px 0 2px'>Steering the DJ right "
                 f"now</h4><p>{steer}</p>")
    if cond_mem:
        # The conditional memory IS the "where" the engine acts on - show
        # it, so the words above and the dice below can be compared.
        by_style = {}
        for (st, cond), v in cond_mem.items():
            by_style.setdefault(st, []).append((cond, v))
        h.append(f"<p style='color:{DIM_C}'>Learned per condition "
                 f"(applied to the style dice for seams that match):</p>"
                 "<table width='100%' cellspacing='0' cellpadding='2'>")
        for st in sorted(by_style):
            items = sorted(by_style[st], key=lambda kv: kv[1])
            h.append(
                f"<tr><td width='24%'>{st}</td><td>" + " · ".join(
                    f"<span style='color:{GOOD_C if v > 1 else BAD_C}'>"
                    f"{cond} ×{v:.2f}</span>" for cond, v in items)
                + "</td></tr>")
        h.append("</table>")

    # -- the execution model ------------------------------------------------
    try:
        from tools.dj.planner import seamprobe
        from tools.dj.planner.seamtune import RANGES
        h.append(seamprobe.report_html(RANGES))
    except Exception as e:
        h.append(f"<p style='color:{DIM_C}'>probe panel unavailable: "
                 f"{e}</p>")
    try:
        from tools.dj.planner import seamtune
        h.append(seamtune.report_html(sm.get("rows") or []))
    except Exception as e:
        h.append(f"<p style='color:{DIM_C}'>execution model "
                 f"unavailable: {e}</p>")

    # -- excluded cohort -----------------------------------------------------
    if sm.get("excluded"):
        h.append(
            f"<p style='color:{DIM_C}'><b>{sm['excluded']} earlier ratings "
            f"are excluded</b> from everything above: they were collected "
            f"under a sampling configuration since found to starve the "
            f"candidate pool, which depressed seam quality independently "
            f"of taste. They are still used for the execution knobs below "
            f"— the nudge there is randomised independently of pair and "
            f"style, so those seams carry noise but no bias.</p>")

    # -- engine eras ---------------------------------------------------------
    eras = sm.get("eras") or {}
    if len(eras) > 1:
        top = sorted(eras.items(), key=lambda kv: -kv[1])
        h.append(
            f"<p style='color:{BAD_C}'><b>Mixed engine versions.</b> These "
            f"{sm['n']} ratings span {len(eras)} builds — "
            + " · ".join(f"{k} ×{n}" for k, n in top[:4])
            + f". Verdicts from before a transition change describe audio "
              f"the engine no longer produces, so treat cross-era totals "
              f"as approximate.</p>")
    elif eras:
        k, n = next(iter(eras.items()))
        h.append(f"<p style='color:{DIM_C}'>All {n} ratings from one "
                 f"engine build ({k}).</p>")

    # -- honesty about the sample ------------------------------------------
    cov = [f"{label} {have}/{tot}" for label, have, tot in sm["coverage"]
           if have < tot]
    if cov:
        h.append(f"<p style='color:{DIM_C}'><i>Measured on part of the "
                 f"dataset only: " + " · ".join(cov)
                 + ". Older ratings predate some diagnostics; missing "
                   "fields are back-filled from the library where "
                   "possible.</i></p>")
    return "".join(h)


def steering_line(brain):
    """What the cross-night memory is doing right now, from a Brain that
    has had load_pair_memory() called on it."""
    if brain is None:
        return ""
    pm = getattr(brain, "pair_memory", {}) or {}
    cm = getattr(brain, "class_memory", {}) or {}
    sm = getattr(brain, "style_memory", {}) or {}
    if not (pm or cm or sm):
        return ("nothing yet — pair memory needs a few good/bad verdicts, "
                "class and style memory need 3+ weighted votes")
    bits = [f"<b>{len(pm)}</b> pair{'s' if len(pm) != 1 else ''} "
            f"remembered",
            f"<b>{len(cm)}</b> feature class"
            f"{'es' if len(cm) != 1 else ''}"]
    if sm:
        top = sorted(sm.items(), key=lambda kv: -abs(kv[1] - 1.0))[:5]
        bits.append("style leans " + ", ".join(
            f"<span style='color:{GOOD_C if v > 1 else BAD_C}'>{s} "
            f"×{v:.2f}</span>" for s, v in top))
    else:
        bits.append("no style lean yet")
    return " · ".join(bits)
