"""One-parameter probing by ABSOLUTE judgment, not comparison.

The first design asked "better or worse than the baseline?" - which
quietly assumes you remember what the baseline sounded like, on a
DIFFERENT pair of songs. Nobody can do that honestly, and a 10% shift is
far below what survives in memory across seams.

What a person CAN answer from a single listen is an absolute question
about this seam, in plain words:

    Where did the handover land?    too late / about right / too early
    How long was the blend?         too long / about right / too short
    The incoming track's entry?     too loud / about right / too quiet

Each answer is a direct constraint on the knob: "too late" means the
value is above where it should be, whatever the baseline was. The search
keeps an interval of still-plausible values per knob, probes inside it,
and shrinks it with each directional answer (damped, so one mistaken
answer cannot exclude the truth). "About right" votes accumulate; enough
of them settles the knob at their centre. There is nothing to remember
between seams.

The other answers keep their old meanings: "can't tell" means the seam
never made the attribute audible (persistent -> parked imperceptible);
"don't get it" means the question is badly worded (parked unclear - a
wording bug, and it says so).

Only the PRIORITY knobs are asked about: 39 questions is too many to
answer honestly, and most deferred knobs have measurable consequences -
tools/dj/dj_knobsweep.py estimates those from rendered audio instead.
"""
import json
import os
import time

from lib.dj.brain import TUNE_DEFAULTS

# knob -> (question, "too much" answer, "too little" answer).
# The "too much" answer ALWAYS corresponds to the knob being too HIGH.
QUESTIONS = {
    "beats_scale": ("How long did the blend feel?",
                    "Too long / dragged", "Too short / rushed"),
    "swap_pos": ("Where did the bass/melody handover land in the blend?",
                 "Too late", "Too early"),
    "fade_recede": ("During the fade, how present was the OUTGOING track "
                    "while the new one arrived?",
                    "Hung on too loud", "Dropped away too far"),
    "fade_b_stage1": ("On the fade, how did the INCOMING track arrive?",
                      "Too loud / barged in", "Too far back / timid"),
    "trim_cap": ("How was the incoming track's entry level?",
                 "Too loud", "Too quiet"),
    "b_mid0": ("The incoming track's mids (melody/voice) on entry?",
               "Too open - melodies clashed", "Too scooped - sounded thin"),
    "stage1_gain": ("How loud did the incoming track ride UNDER the "
                    "outgoing one before the swap?",
                    "Too loud - fought the outgoing track",
                    "Too quiet - blend felt empty"),
    "high_swap_at": ("When did the hats/air hand over?",
                     "Too late", "Too early"),
    "pre_dip_gain": ("The outgoing track's level just before the swap?",
                     "Stayed too loud", "Dipped too much"),
    # Deferred knobs keep questions too - they are asked only if promoted.
    "swap_beats": ("How abrupt was the handover itself?",
                   "Too drawn out", "Too abrupt"),
    "swap_beats_long": ("How abrupt was the handover on this long blend?",
                        "Too drawn out", "Too abrupt"),
    "fade_lead_a": ("When did the outgoing track start receding?",
                    "Too early - it left before the seam",
                    "Too late - the fade felt sudden"),
    "fade_lead_b": ("When did the incoming track arrive on the fade?",
                    "Too early", "Too late"),
    "fade_b_ramp1": ("How fast did the incoming track become present?",
                     "Too slow", "Too fast"),
    "fade_b_ramp2": ("How fast did the incoming track reach full level?",
                     "Too slow", "Too fast"),
    "fade_out_ramp": ("The outgoing track's final fade?",
                      "Too long", "Too abrupt"),
    "fade_stop_lead": ("How long did the outgoing track linger after the "
                       "seam?", "Too long", "Cut off too soon"),
    "b_mid0_hot": ("Against a mid-heavy outgoing track, the incoming "
                   "mids?", "Too open - clashed", "Too scooped - thin"),
    "b_mid0_long": ("Through the long blend, the incoming mids?",
                    "Too open - clashed", "Too scooped - thin"),
    "b_high0": ("The incoming track's highs on entry?",
                "Too bright", "Too dull"),
    "b_high0_hot": ("Against a bright outgoing track, the incoming highs?",
                    "Too bright", "Too dull"),
    "b_high0_long": ("Through the long blend, the incoming highs?",
                     "Too bright", "Too dull"),
    "stage1_frac": ("How quickly did the incoming track reach its riding "
                    "level?", "Too slow", "Too fast"),
    "pre_dip_at": ("When did the outgoing track start easing down?",
                   "Too late", "Too early"),
    "exit_res": ("After the swap, the outgoing track hung around...",
                 "Too long", "Left too abruptly"),
    "exit_res_long": ("After the long blend, the outgoing track hung "
                      "around...", "Too long", "Left too abruptly"),
    "duck_depth": ("How much of the outgoing vocal survived the duck?",
                   "Too much - voices clashed", "Too little - felt gutted"),
    "duck_beats": ("How fast did the vocal duck come in?",
                   "Too slow", "Too snappy"),
    "echo_lead_beats": ("On the echo exit, when did the incoming track "
                        "arrive?", "Too early", "Too late"),
    "echo_b_gain": ("Under the echo tail, the incoming track sat...",
                    "Too loud", "Too quiet"),
    "echo_delay_beats": ("The echo repeats were spaced...",
                         "Too far apart", "Too close together"),
    "echo_feedback": ("The echo repeats lasted...",
                      "Too long", "Died too fast"),
    "echo_wet": ("The echo against the dry signal was...",
                 "Too loud", "Too subtle"),
    "echo_tail_s": ("The echo tail rang on...",
                    "Too long", "Cut too short"),
    "spinback_s": ("The spinback wind-down was...",
                   "Too slow/long", "Too quick"),
    "brake_s": ("The brake into the cut was...",
                "Too slow/long", "Too quick"),
    "brake_chance": ("Brakes on this style happen...",
                     "Too often", "Not often enough"),
    "roll_shrink1": ("The loop roll's first halving came...",
                     "Too late", "Too soon"),
    "roll_shrink2": ("The loop roll's second halving came...",
                     "Too late", "Too soon"),
}

VERDICTS = ("too_much", "right", "too_little", "cant_tell", "dont_get_it")

# WORTH A HUMAN'S EAR - see module docstring. Everything else is deferred
# to the machine sweep unless promote()d.
PRIORITY = (
    "beats_scale", "swap_pos", "fade_recede", "fade_b_stage1", "trim_cap",
    "b_mid0", "stage1_gain", "high_swap_at", "pre_dip_gain",
)

RIGHT_STOP = 2         # "about right" votes to settle a knob
CANT_STOP = 3          # "can't tell" votes to park as imperceptible
UNCLEAR_STOP = 2       # "don't get it" votes to park as unclear
SHRINK = 0.6           # how far a directional answer pulls its bound in
                       # (damped: one wrong answer cannot exclude the truth)
MIN_WIDTH = 0.18       # settle when the plausible interval is this narrow
                       # (fraction of the knob's range)
APPLY_MIN = 0.08       # only move the live value if the settled estimate
                       # differs from it by at least this much of the range


def state_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))), "logs",
        "seam_probe.json")


def load():
    try:
        with open(state_path(), encoding="utf-8") as f:
            doc = json.load(f)
            if isinstance(doc, dict) and "knobs" in doc:
                # MIGRATION: entries from the abandoned better/worse
                # design (no interval bounds) answered a question nobody
                # could honestly answer - drop that state, keep the raw
                # trial log for the record.
                for k in [k for k, v in doc["knobs"].items()
                          if "lo" not in v]:
                    del doc["knobs"][k]
                return doc
    except (OSError, ValueError):
        pass
    return {"knobs": {}, "trials": [], "promoted": []}


def save(doc):
    p = state_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
    os.replace(tmp, p)


def _fresh(lo, hi):
    return {"status": "testing", "lo": lo, "hi": hi, "rights": [],
            "cant": 0, "unclear": 0, "trials": 0, "applied": None}


def _peek(doc, knob, ranges):
    st = doc["knobs"].get(knob)
    if st is None:
        lo, hi = ranges[knob]
        st = _fresh(lo, hi)
    return st


def status_of(doc, knob):
    return doc["knobs"].get(knob, {}).get("status", "testing")


def trials_of(doc, knob):
    return doc["knobs"].get(knob, {}).get("trials", 0)


def sweep_verdicts():
    """{knob: verdict} from the machine sweep results, where verdict is
    'machine_set' (a decisive measured optimum - no need to ask a human),
    'unreachable' (identical scores at every value: the rendered seams
    never exercised the knob, so a human seam probably wouldn't either),
    or 'taste' (measured and flat - genuinely the listener's call).

    Recomputed from the RAW scores rather than trusting the stored flat
    flag: early runs used an absolute spread bar that let a ~2% monotone
    drift on a fade's huge constant baseline read as signal."""
    out = {}
    logs = os.path.dirname(state_path())
    for fname in ("knob_sweep_priority.json", "knob_sweep.json"):
        try:
            with open(os.path.join(logs, fname), encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError):
            continue
        for r in doc.get("results", []):
            scores = [float(v) for v in (r.get("scores") or {}).values()]
            if len(scores) < 2:
                continue
            spread = max(scores) - min(scores)
            rel = spread / max(min(scores), 1e-9)
            if spread <= 1e-9:
                out[r["knob"]] = "unreachable"
            elif spread >= 0.8 and rel >= 0.10:
                out[r["knob"]] = "machine_set"
            else:
                out[r["knob"]] = "taste"
    return out


def open_knobs(ranges, doc=None):
    """Knobs still worth asking a HUMAN about: the priority tier (plus
    anything promote()d), minus whatever the machine sweep has already
    resolved or shown to be unreachable. The roster SHRINKS as sweep
    results land - only genuine taste calls remain."""
    doc = doc or load()
    promoted = set(doc.get("promoted") or [])
    sv = sweep_verdicts()
    return [k for k in ranges
            if k in QUESTIONS and status_of(doc, k) == "testing"
            and (k in PRIORITY or k in promoted)
            and sv.get(k, "taste") == "taste"]


def promote(knob, doc=None):
    """Ask about a deferred knob after all - e.g. when the machine sweep
    found its metrics flat and taste is the only arbiter left."""
    doc = doc or load()
    doc.setdefault("promoted", [])
    if knob not in doc["promoted"]:
        doc["promoted"].append(knob)
    save(doc)
    return doc


def next_probe(ranges, baseline, knob=None, doc=None, rng=None):
    """The next single-parameter probe: render at the middle of the
    still-plausible interval (with a little jitter so repeats are not
    identical). Returns {knob, value, question, too_much, too_little,
    trials} or None when nothing is left to ask."""
    doc = doc or load()
    pool = open_knobs(ranges, doc)
    if not pool:
        return None
    import random as _r
    rng = rng or _r
    if knob is None:
        fewest = min(trials_of(doc, k) for k in pool)
        knob = rng.choice([k for k in pool if trials_of(doc, k) == fewest])
    st = _peek(doc, knob, ranges)
    lo, hi = st["lo"], st["hi"]
    span = ranges[knob][1] - ranges[knob][0]
    mid = (lo + hi) / 2.0
    value = mid + rng.uniform(-0.15, 0.15) * max(hi - lo, 0.05 * span)
    value = max(ranges[knob][0], min(ranges[knob][1], value))
    q, too_much, too_little = QUESTIONS[knob]
    return {"knob": knob, "value": round(value, 4),
            "question": q, "too_much": too_much, "too_little": too_little,
            "trials": st["trials"]}


def record(probe, verdict, ranges, doc=None, now=None):
    """Fold one absolute judgment in. Returns (doc, note)."""
    doc = doc or load()
    knob = probe["knob"]
    st = doc["knobs"].setdefault(knob, _fresh(*ranges[knob]))
    r_lo, r_hi = ranges[knob]
    span = r_hi - r_lo
    v = float(probe["value"])
    st["trials"] += 1
    doc["trials"].append({"t": now or time.time(), "knob": knob,
                          "value": v, "verdict": verdict})
    doc["trials"] = doc["trials"][-2000:]
    note = ""

    def settle():
        est = (sum(st["rights"]) / len(st["rights"]) if st["rights"]
               else (st["lo"] + st["hi"]) / 2.0)
        st["status"] = "settled"
        cur = _current_value(knob)
        if abs(est - cur) >= APPLY_MIN * span:
            _apply(knob, est)
            st["applied"] = round(est, 4)
            return (f"{knob}: settled at {est:g} (was {cur:g}) - the "
                    f"engine now mixes with it")
        return (f"{knob}: settled - where it already is ({cur:g}) sits "
                f"inside what you called right")

    if verdict == "dont_get_it":
        st["unclear"] += 1
        if st["unclear"] >= UNCLEAR_STOP:
            st["status"] = "unclear"
            note = (f"{knob}: parked - the question isn't landing, so the "
                    f"answers wouldn't mean anything. The wording needs "
                    f"fixing, not your ears.")
    elif verdict == "cant_tell":
        st["cant"] += 1
        if st["cant"] >= CANT_STOP:
            st["status"] = "imperceptible"
            note = (f"{knob}: parked - {st['cant']} seams never made this "
                    f"audible. Probably not worth tuning by ear.")
    elif verdict == "right":
        st["rights"].append(v)
        if len(st["rights"]) >= RIGHT_STOP:
            note = settle()
        else:
            note = f"{knob}: noted as about right at {v:g}"
    elif verdict in ("too_much", "too_little"):
        # A direct constraint: the correct value lies below (too_much) or
        # above (too_little) the probed one. Pull that bound in, damped.
        if verdict == "too_much":
            st["hi"] = min(st["hi"], v + (1 - SHRINK) * (st["hi"] - v))
        else:
            st["lo"] = max(st["lo"], v - (1 - SHRINK) * (v - st["lo"]))
        if st["hi"] - st["lo"] <= MIN_WIDTH * span:
            note = settle()
        else:
            note = (f"{knob}: narrowing - now searching "
                    f"{st['lo']:g}..{st['hi']:g}")
    save(doc)
    return doc, note


def _current_value(knob):
    from lib.dj import tuning
    return float(tuning.value(knob, TUNE_DEFAULTS[knob]))


def _apply(knob, value):
    from lib.dj import tuning
    tuning.set_value(knob, value, why="probe: absolute judgments")


def summary(ranges, doc=None):
    doc = doc or load()
    promoted = set(doc.get("promoted") or [])
    sv = sweep_verdicts()
    rows = []
    for k in sorted(ranges):
        if k not in QUESTIONS:
            continue
        st = doc["knobs"].get(k)
        active = k in PRIORITY or k in promoted
        # The machine's word outranks the queue: a measured optimum or an
        # unreachable knob is not a question any more.
        if active and st is None and sv.get(k) == "machine_set":
            rows.append({"knob": k, "trials": 0, "status": "machine_set"})
            continue
        if active and st is None and sv.get(k) == "unreachable":
            rows.append({"knob": k, "trials": 0, "status": "unreachable"})
            continue
        if not st:
            rows.append({"knob": k, "trials": 0,
                         "status": "untested" if active else "deferred"})
            continue
        rows.append({"knob": k, "status": st["status"],
                     "trials": st["trials"],
                     "applied": st.get("applied")})
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    done = sum(counts.get(s, 0) for s in
               ("settled", "imperceptible", "unclear"))
    active_total = sum(1 for r in rows if r["status"] not in
                       ("deferred", "machine_set", "unreachable"))
    return {"rows": rows, "counts": counts, "done": done,
            "total": active_total,
            "machine": counts.get("machine_set", 0),
            "unreachable": counts.get("unreachable", 0),
            "deferred": counts.get("deferred", 0)}


_GOOD_C, _BAD_C, _DIM_C = "#2e8b57", "#c0392b", "#8a8f98"
_STATE_TEXT = {
    "settled": ("done", _GOOD_C, "answered"),
    "imperceptible": ("done", _DIM_C, "never audible - left alone"),
    "unclear": ("parked", _BAD_C, "question needs rewording"),
    "testing": ("asking", None, ""),
    "untested": ("queued", _DIM_C, ""),
    "machine_set": ("measured", _GOOD_C,
                    "the sweep found a decisive optimum - not asked"),
    "unreachable": ("skipped", _DIM_C,
                    "the sweep could not exercise it; a listening seam "
                    "would not either"),
}


def report_html(ranges, doc=None):
    doc = doc or load()
    sm = summary(ranges, doc)
    h = [f"<h4 style='margin:8px 0 2px'>Parameter questions &nbsp;"
         f"<span style='font-weight:normal;color:{_DIM_C}'>absolute "
         f"judgments - nothing to remember between seams</span></h4>"]
    c = sm["counts"]
    h.append(
        f"<p><b>{sm['done']} of {sm['total']} answered</b> "
        f"<span style='color:{_DIM_C}'>(only genuine taste calls are "
        f"asked)</span> &nbsp; "
        f"<span style='color:{_GOOD_C}'>{c.get('settled', 0)} settled"
        f"</span> · {c.get('imperceptible', 0)} never audible · "
        f"<span style='color:{_BAD_C}'>{c.get('unclear', 0)} need "
        f"rewording</span> &nbsp;·&nbsp; <span style='color:{_DIM_C}'>"
        f"machine: {sm['machine']} measured, {sm['unreachable']} "
        f"unreachable, {sm['deferred']} in its queue</span></p>")
    interesting = [r for r in sm["rows"] if r["status"] not in
                   ("deferred", "untested")]
    if interesting:
        h.append("<table width='100%' cellspacing='0' cellpadding='2'>")
        for r in sorted(interesting,
                        key=lambda r: (r["status"] != "settled",
                                       -r["trials"])):
            word, col, why = _STATE_TEXT.get(r["status"],
                                             (r["status"], None, ""))
            extra = (f" → <b>{r['applied']:g}</b> applied"
                     if r.get("applied") else "")
            h.append(f"<tr><td width='24%'>{r['knob']}</td>"
                     f"<td width='12%' style='color:{col or ''}'>{word}"
                     f"</td><td width='10%' align='right' "
                     f"style='color:{_DIM_C}'>{r['trials']} asked</td>"
                     f"<td>{why}{extra}</td></tr>")
        h.append("</table>")
    if sm["done"] == sm["total"] and sm["total"]:
        h.append(f"<p style='color:{_GOOD_C}'><b>Every question is "
                 f"answered.</b> Promote a deferred knob if the machine "
                 f"sweep leaves one to taste.</p>")
    return "".join(h)
