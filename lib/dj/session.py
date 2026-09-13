"""One process, one session id - and one place that writes the night's log.

Every row the DJ engines append to logs/dj_YYYYMMDD.jsonl carries `sid`, so a night's log can be split by
session (the live planner and a headless gate ran in the same hour once and their rows were indistinguishable).
Under DJ_NO_PLAY_LOG=1 (gates, smokes) nothing is written at all.
"""
import json
import os
import time
import uuid

SESSION_ID = uuid.uuid4().hex[:6]
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")


def headless():
    return os.environ.get("DJ_NO_PLAY_LOG", "") == "1"


def log_row(payload, log_dir=None):
    """Append one row (a dict) with the time and the session id; silent under DJ_NO_PLAY_LOG."""
    if headless():
        return
    try:
        d = log_dir or LOG_DIR
        os.makedirs(d, exist_ok=True)
        row = {"t": round(time.time(), 2), "sid": SESSION_ID, **payload}
        with open(os.path.join(d, f"dj_{time.strftime('%Y%m%d')}.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except (OSError, TypeError, ValueError):
        pass
