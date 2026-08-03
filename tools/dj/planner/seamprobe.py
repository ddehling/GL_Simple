"""One-at-a-time parameter probing against a known baseline.

The statistical approach this replaces asked you to rate whole seams
while several things varied at once, and inferred each parameter's effect
from the correlation. It cannot work at human rating volumes: the
pair-to-pair variance swamps a knob's effect, and measurement showed even
3,377 samples of one knob failed to identify a strong effect.

This asks a different question. Hold every parameter at the baseline,
move exactly ONE, say out loud what was moved and in which direction, and
ask about THAT. Four answers, and the third is the valuable one:

  good           - the change is an improvement; the baseline moves
  bad            - the change is worse; the baseline is better here
  can't tell     - below your threshold at this size; TEST BIGGER
  don't get it   - the question is not meaningful; stop asking it

"Can't tell" is what makes this converge. It measures the just-noticeable
difference directly instead of inferring it, and it drives the staircase:
imperceptible changes grow until they are perceptible or until the
parameter's whole range is exhausted - which is itself the finding that
the range is too tight, or that the parameter does not matter.

A knob stops being tested when it is:

  SETTLED       - worse in BOTH directions at sizes you could perceive,
                  so the baseline is a local optimum
  IMPERCEPTIBLE - indistinguishable even at the edge of its range
  UNCLEAR       - the question did not land; needs a better description
  IMPROVED      - moved, and the new value now needs re-confirming

Expect on the order of 6-12 trials per knob rather than thousands.
"""
import json
import os
import time

from lib.dj.brain import TUNE_DEFAULTS

# How a change SOUNDS, per knob: (raise it, lower it). Written for
# someone listening to the seam, not reading the code - if a description
# does not tell you what to listen for, the answer will be "don't get it"
# and that is a bug in the description, not in the listener.
DESCRIPTIONS = {
    "swap_pos": ("the bass/melody handover happens LATER in the blend",
                 "the handover happens EARLIER in the blend"),
    "swap_beats": ("the handover is SPREAD over more beats (gentler)",
                   "the handover is more ABRUPT"),
    "swap_beats_long": ("the handover is SPREAD over more beats (gentler)",
                        "the handover is more ABRUPT"),
    "trim_cap": ("a quiet incoming track is pushed LOUDER on entry",
                 "a quiet incoming track is left QUIETER on entry"),
    "b_mid0": ("the incoming track's MIDS are more open on entry",
               "the incoming track's MIDS are more scooped on entry"),
    "b_mid0_hot": ("the incoming MIDS are more open when the outgoing "
                   "track is mid-heavy",
                   "the incoming MIDS are more scooped when the outgoing "
                   "track is mid-heavy"),
    "b_mid0_long": ("the incoming MIDS are more open through a long blend",
                    "the incoming MIDS are more scooped through a long blend"),
    "b_high0": ("the incoming track's HIGHS/air are more open on entry",
                "the incoming track's HIGHS are rolled off on entry"),
    "b_high0_hot": ("the incoming HIGHS are more open when the outgoing "
                    "track is bright",
                    "the incoming HIGHS are rolled off when the outgoing "
                    "track is bright"),
    "b_high0_long": ("the incoming HIGHS are more open through a long blend",
                     "the incoming HIGHS are rolled off through a long blend"),
    "stage1_gain": ("the incoming track rides LOUDER under the outgoing "
                    "one before the swap",
                    "the incoming track rides QUIETER under the outgoing "
                    "one before the swap"),
    "stage1_frac": ("the incoming track takes LONGER to reach its riding "
                    "level", "the incoming track reaches its riding level "
                    "FASTER"),
    "high_swap_at": ("the hats/air hand over LATER",
                     "the hats/air hand over EARLIER"),
    "beats_scale": ("the whole blend is LONGER",
                    "the whole blend is SHORTER"),
    "pre_dip_at": ("the outgoing track starts easing down LATER",
                   "the outgoing track starts easing down EARLIER"),
    "pre_dip_gain": ("the outgoing track eases down LESS before the swap",
                     "the outgoing track eases down MORE before the swap"),
    "exit_res": ("the outgoing track hangs around LONGER after the swap",
                 "the outgoing track leaves SOONER after the swap"),
    "exit_res_long": ("the outgoing track hangs around LONGER after a long "
                      "blend", "the outgoing track leaves SOONER after a "
                      "long blend"),
    "duck_depth": ("the outgoing VOCAL is ducked less (more of it survives)",
                   "the outgoing VOCAL is ducked harder (more of it "
                   "disappears)"),
    "duck_beats": ("the vocal duck fades in more SLOWLY",
                   "the vocal duck snaps in more QUICKLY"),
    "fade_recede": ("on a fade, the outgoing track stays LOUDER as the new "
                    "one arrives",
                    "on a fade, the outgoing track drops FURTHER back as "
                    "the new one arrives"),
    "fade_lead_a": ("on a fade, the outgoing track starts receding EARLIER",
                    "on a fade, the outgoing track starts receding LATER"),
    "fade_lead_b": ("on a fade, the incoming track arrives EARLIER",
                    "on a fade, the incoming track arrives LATER"),
    "fade_b_stage1": ("on a fade, the incoming track arrives more PRESENT",
                      "on a fade, the incoming track arrives further BACK"),
    "fade_b_ramp1": ("on a fade, the incoming track takes LONGER to become "
                     "present", "on a fade, the incoming track becomes "
                     "present FASTER"),
    "fade_b_ramp2": ("on a fade, the incoming track takes LONGER to reach "
                     "full", "on a fade, the incoming track reaches full "
                     "FASTER"),
    "fade_out_ramp": ("on a fade, the outgoing track's final fade is LONGER",
                      "on a fade, the outgoing track's final fade is FASTER"),
    "fade_stop_lead": ("on a fade, the outgoing track lingers LONGER before "
                       "stopping", "on a fade, the outgoing track stops "
                       "SOONER"),
    "echo_lead_beats": ("on an echo exit, the incoming track arrives EARLIER "
                        "under the tail",
                        "on an echo exit, the incoming track arrives LATER"),
    "echo_b_gain": ("on an echo exit, the incoming track sits LOUDER under "
                    "the tail", "on an echo exit, the incoming track sits "
                    "QUIETER under the tail"),
    "echo_delay_beats": ("the echo repeats are FURTHER apart",
                         "the echo repeats are CLOSER together"),
    "echo_feedback": ("the echo repeats last LONGER (more repeats)",
                      "the echo repeats die away FASTER"),
    "echo_wet": ("the echo is LOUDER against the dry signal",
                 "the echo is more SUBTLE"),
    "echo_tail_s": ("the echo tail rings on LONGER before the deck stops",
                    "the echo tail is cut SHORTER"),
    "spinback_s": ("the spinback wind-down is SLOWER/longer",
                   "the spinback is a QUICKER snap"),
    "brake_s": ("the brake into the cut is SLOWER/longer",
                "the brake into the cut is QUICKER"),
    "brake_chance": ("a brake happens MORE often on this style",
                     "a brake happens LESS often on this style"),
    "roll_shrink1": ("the loop roll waits LONGER before halving",
                     "the loop roll halves SOONER"),
    "roll_shrink2": ("the loop roll waits LONGER before halving again",
                     "the loop roll halves again SOONER"),
}

VERDICTS = ("good", "bad", "cant_tell", "dont_get_it")

START_FRAC = 0.25      # first deviation, as a fraction of the knob's range
GROW = 1.7             # multiply the deviation after "can't tell"
CONFIRM = 2            # consistent answers needed to call a direction
UNCLEAR_STOP = 2       # "don't get it" answers before parking a knob
EDGE_STOP = 2          # "can't tell" at the range edge before parking


def state_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))), "logs",
        "seam_probe.json")


def load():
    try:
        with open(state_path(), encoding="utf-8") as f:
            doc = json.load(f)
            if isinstance(doc, dict):
                return doc
    except (OSError, ValueError):
        pass
    return {"knobs": {}, "trials": []}


def save(doc):
    p = state_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
    os.replace(tmp, p)


_FRESH = {"frac": START_FRAC, "status": "testing", "dir": 1,
          "bad_up": 0, "bad_down": 0, "good": 0, "cant": 0,
          "edge_cant": 0, "unclear": 0, "trials": 0, "moved": 0.0}


def _peek(doc, knob):
    """Read-only view - safe to call from the render thread."""
    return doc["knobs"].get(knob) or dict(_FRESH)


def _ks(doc, knob):
    return doc["knobs"].setdefault(knob, {
        "frac": START_FRAC, "status": "testing", "dir": 1,
        "bad_up": 0, "bad_down": 0, "good": 0, "cant": 0,
        "edge_cant": 0, "unclear": 0, "trials": 0, "moved": 0.0})


def status_of(doc, knob):
    return doc["knobs"].get(knob, {}).get("status", "testing")


def open_knobs(ranges, doc=None):
    """Knobs still worth asking about."""
    doc = doc or load()
    return [k for k in ranges
            if k in DESCRIPTIONS and status_of(doc, k) == "testing"]


def next_probe(ranges, baseline, knob=None, doc=None, rng=None):
    """Choose the next single-parameter probe.

    Returns {knob, value, baseline, direction, frac, description} or None
    when every knob has been parked."""
    doc = doc or load()
    pool = open_knobs(ranges, doc)
    if not pool:
        return None
    if knob is None:
        import random as _r
        rng = rng or _r
        # Fewest trials first, so attention spreads instead of grinding
        # one knob to death while others are untouched.
        fewest = min(_peek(doc, k)["trials"] for k in pool)
        knob = rng.choice([k for k in pool
                           if _peek(doc, k)["trials"] == fewest])
    st = _peek(doc, knob)
    lo, hi = ranges[knob]
    base = float(baseline.get(knob, TUNE_DEFAULTS[knob]))
    span = hi - lo
    d = st["frac"] * span * (1 if st["dir"] > 0 else -1)
    value = max(lo, min(hi, base + d))
    at_edge = abs(value - (hi if d > 0 else lo)) < 1e-9
    up, down = DESCRIPTIONS[knob]
    return {"knob": knob, "value": round(value, 4), "baseline": base,
            "direction": 1 if value >= base else -1,
            "frac": st["frac"], "at_edge": at_edge,
            "description": up if value > base else down,
            "trials": st["trials"]}


def record(probe, verdict, ranges, doc=None, now=None):
    """Fold one answer in. Returns (doc, note) - note is a human line
    describing what the answer changed, or "" when nothing changed."""
    doc = doc or load()
    knob = probe["knob"]
    st = _ks(doc, knob)
    lo, hi = ranges[knob]
    span = hi - lo
    st["trials"] += 1
    doc["trials"].append({"t": now or time.time(), "knob": knob,
                          "value": probe["value"], "base": probe["baseline"],
                          "frac": probe["frac"], "verdict": verdict})
    doc["trials"] = doc["trials"][-2000:]
    note = ""

    if verdict == "dont_get_it":
        st["unclear"] += 1
        if st["unclear"] >= UNCLEAR_STOP:
            st["status"] = "unclear"
            note = (f"{knob}: parked - the description isn't landing, so "
                    f"the answers wouldn't mean anything")
    elif verdict == "cant_tell":
        st["cant"] += 1
        if probe["at_edge"]:
            st["edge_cant"] += 1
            if st["edge_cant"] >= EDGE_STOP:
                st["status"] = "imperceptible"
                note = (f"{knob}: parked - inaudible even at the edge of "
                        f"its range. Either it doesn't matter, or the "
                        f"range is too tight to matter.")
            else:
                st["dir"] = -st["dir"]
        else:
            st["frac"] = min(st["frac"] * GROW, 1.0)
            note = (f"{knob}: no difference heard - trying a bigger change "
                    f"({st['frac']*100:.0f}% of its range)")
    elif verdict == "bad":
        key = "bad_up" if probe["direction"] > 0 else "bad_down"
        st[key] += 1
        if st["bad_up"] >= CONFIRM and st["bad_down"] >= CONFIRM:
            st["status"] = "settled"
            note = (f"{knob}: settled - worse in BOTH directions at sizes "
                    f"you could hear, so the current value is right")
        else:
            st["dir"] = -st["dir"]          # try the other side
            note = f"{knob}: worse that way - trying the other direction"
    elif verdict == "good":
        st["good"] += 1
        # Move the baseline halfway to the value that was preferred, then
        # keep probing around the NEW baseline. Halfway, not all the way:
        # one listen is one data point.
        base = float(probe["baseline"])
        new = max(lo, min(hi, base + 0.5 * (probe["value"] - base)))
        st["moved"] = round(new, 4)
        st["bad_up"] = st["bad_down"] = 0     # the old verdicts were about
        st["frac"] = START_FRAC               # a different baseline
        st["status"] = "improved"
        note = (f"{knob}: better that way - baseline {base:g} → {new:g}, "
                f"will re-check around the new value")
    save(doc)
    return doc, note


def pending_moves(doc=None):
    """{knob: new_value} for knobs whose baseline the answers moved."""
    doc = doc or load()
    return {k: v["moved"] for k, v in doc["knobs"].items()
            if v.get("moved") and v.get("status") == "improved"}


def reopen(doc, knob):
    """After a moved baseline is applied, resume testing around it."""
    st = _ks(doc, knob)
    st["status"] = "testing"
    st["moved"] = 0.0
    save(doc)
    return doc


_GOOD_C, _BAD_C, _DIM_C = "#2e8b57", "#c0392b", "#8a8f98"
_STATE_TEXT = {
    "settled": ("done", _GOOD_C,
                "worse both ways — current value is right"),
    "imperceptible": ("done", _DIM_C,
                      "inaudible across its whole range — either it does "
                      "not matter, or the range is too tight to matter"),
    "unclear": ("parked", _BAD_C,
                "the description did not land — needs rewording before "
                "asking again"),
    "improved": ("moved", _GOOD_C, "baseline moved; re-checking"),
    "testing": ("testing", None, ""),
    "untested": ("not started", _DIM_C, ""),
}


def report_html(ranges, doc=None):
    """The probe panel: what is finished, what is still being asked."""
    sm = summary(ranges, doc)
    h = [f"<h4 style='margin:8px 0 2px'>Parameter probes &nbsp;"
         f"<span style='font-weight:normal;color:{_DIM_C}'>one at a time, "
         f"against the baseline</span></h4>"]
    c = sm["counts"]
    h.append(
        f"<p><b>{sm['done']} of {sm['total']} answered</b> &nbsp; "
        f"<span style='color:{_GOOD_C}'>{c.get('settled',0)} settled</span> · "
        f"{c.get('imperceptible',0)} imperceptible · "
        f"<span style='color:{_BAD_C}'>{c.get('unclear',0)} need rewording"
        f"</span> · {c.get('testing',0)+c.get('untested',0)} still open</p>")
    done = [r for r in sm["rows"]
            if r["status"] in ("settled", "imperceptible", "unclear")]
    if done:
        h.append("<table width='100%' cellspacing='0' cellpadding='2'>")
        for r in sorted(done, key=lambda r: r["status"]):
            word, col, why = _STATE_TEXT[r["status"]]
            h.append(f"<tr><td width='24%'>{r['knob']}</td>"
                     f"<td width='12%' style='color:{col}'>{word}</td>"
                     f"<td width='10%' align='right' style='color:{_DIM_C}'>"
                     f"{r['trials']} trials</td><td>{why}</td></tr>")
        h.append("</table>")
    live = [r for r in sm["rows"] if r["status"] in ("testing", "improved")
            and r["trials"]]
    if live:
        h.append(f"<p style='color:{_DIM_C}'>In progress: " + " · ".join(
            f"{r['knob']} ({r['trials']})" for r in
            sorted(live, key=lambda r: -r["trials"])[:10]) + "</p>")
    if sm["done"] == sm["total"] and sm["total"]:
        h.append(f"<p style='color:{_GOOD_C}'><b>Every parameter has been "
                 f"answered.</b> Nothing further to test unless you widen a "
                 f"range or reword a parked description.</p>")
    return "".join(h)


def summary(ranges, doc=None):
    """Counts + per-knob lines for the panel."""
    doc = doc or load()
    rows = []
    for k in sorted(ranges):
        if k not in DESCRIPTIONS:
            continue
        st = doc["knobs"].get(k)
        if not st:
            rows.append({"knob": k, "status": "untested", "trials": 0})
            continue
        rows.append({"knob": k, "status": st["status"],
                     "trials": st["trials"], "frac": st.get("frac"),
                     "good": st.get("good", 0),
                     "bad": st.get("bad_up", 0) + st.get("bad_down", 0),
                     "cant": st.get("cant", 0),
                     "moved": st.get("moved") or None})
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    done = sum(counts.get(s, 0) for s in
               ("settled", "imperceptible", "unclear"))
    return {"rows": rows, "counts": counts, "done": done,
            "total": len(rows)}
