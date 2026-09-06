"""Authored hooks: an LLM writes the movement's theme (and its answer),
the rules develop it.

HookAuthor runs on its own thread and talks to the same Claude CLI the
director uses (lib/gen/director.py). request() asks for a few candidate
two-bar hooks with answer phrases for a style / key / tempo; replies are
validated (grid, range, note count) and cached in logs/gen_hooks.json
per style, so a night without the CLI still has hooks from earlier
nights. provider(rng) hands the melody one hook dict:
    {"steps": [..0..31..], "degrees": [..], "contour": "arch", "answer": {"steps": [...], "degrees": [...]}}
Never blocks the conductor: a request that has not returned yet simply
means the melody makes its own theme from the corpus model."""
from __future__ import annotations

import json
import os
import random
import threading
import time

PROMPT = """You are writing a lead hook for generative {label} at {bpm:.0f} BPM in {key}.
Return ONLY a JSON object, no prose:
{{"hooks": [ {{"name": "...", "steps": [16th-step onsets 0..31 for a 2-bar hook, ascending],
              "degrees": [scale-degree offsets from the chord root, -4..9, same length as steps],
              "contour": "arch|rise|fall|wave|flat",
              "answer": {{"steps": [0..31], "degrees": [...]}} }}, ... {n} of them ]}}
Rules: 5 to 10 notes per hook, syncopated but with at least 3 notes on beats (steps divisible by 4),
memorable (a repeated cell, a clear peak), mostly stepwise with one or two leaps, the hook ends on
degree 0 or 4; the answer keeps the rhythm feel but changes the contour and ends on 0.
{hint}"""


def validate(h: dict):
    """A hook dict cleaned up, or None when it is unusable."""
    try:
        steps = [int(s) for s in h.get("steps", [])]
        degs = [int(d) for d in h.get("degrees", [])]
    except Exception:
        return None
    n = min(len(steps), len(degs))
    steps, degs = steps[:n], degs[:n]
    pairs = sorted({s: d for s, d in zip(steps, degs) if 0 <= s <= 31}.items())
    if not (4 <= len(pairs) <= 12):
        return None
    steps = [s for s, _ in pairs]
    degs = [max(-4, min(9, d)) for _, d in pairs]
    if sum(1 for s in steps if s % 4 == 0) < 2:
        return None
    out = {"steps": steps, "degrees": degs, "contour": h.get("contour") if h.get("contour") in ("arch", "rise", "fall", "wave", "flat") else "flat",
           "name": str(h.get("name", ""))[:40]}
    ans = h.get("answer")
    if isinstance(ans, dict):
        a = validate({"steps": ans.get("steps", []), "degrees": ans.get("degrees", [])})
        if a:
            out["answer"] = {"steps": a["steps"], "degrees": a["degrees"]}
    return out


class HookAuthor:
    def __init__(self, path="logs/gen_hooks.json", transport=None, model=None, enabled=True):
        self.path = path
        self.enabled = enabled
        self.transport = transport            # callable(system_prompt, prompt) -> text; None = the director's CLI
        self.model = model
        self.cache = {}                       # style -> [hook dicts]
        self.pending = set()
        self.error = ""
        self.lock = threading.Lock()
        self._load()

    # -- storage ------------------------------------------------------------
    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as fh:
                self.cache = {k: [h for h in (validate(x) for x in v) if h] for k, v in json.load(fh).items()}
        except Exception:
            self.cache = {}

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump({k: v[-40:] for k, v in self.cache.items()}, fh)
        except Exception:
            pass

    def add(self, style: str, hooks: list):
        good = [h for h in (validate(x) for x in hooks) if h]
        with self.lock:
            self.cache.setdefault(style, []).extend(good)
            self.cache[style] = self.cache[style][-40:]
            self._save()
        return len(good)

    def count(self, style: str) -> int:
        return len(self.cache.get(style, []))

    # -- the ask ------------------------------------------------------------
    def _call(self, prompt: str) -> str:
        if self.transport is not None:
            return self.transport("", prompt)
        from lib.gen.director import find_claude_exe
        import subprocess
        exe = find_claude_exe()
        if not exe:
            raise RuntimeError("claude CLI not found")
        args = [exe, "-p", prompt, "--output-format", "text"]
        if self.model:
            args += ["--model", self.model]
        r = subprocess.run(args, capture_output=True, text=True, timeout=120, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout or "")[:200])
        return r.stdout

    def request(self, style: str, label: str, bpm: float, key: str, n: int = 4, hint: str = "", block=False):
        """Ask for n hooks in the background (or synchronously with block=True)."""
        if not self.enabled:
            return False
        k = (style, key)
        with self.lock:
            if k in self.pending:
                return False
            self.pending.add(k)

        def run():
            try:
                text = self._call(PROMPT.format(label=label, bpm=bpm, key=key, n=n, hint=hint))
                from lib.gen.director import _extract_json
                obj = _extract_json(text) or {}
                got = self.add(style, obj.get("hooks", []))
                self.error = "" if got else "no valid hooks in reply"
            except Exception as e:  # noqa: BLE001
                self.error = f"{type(e).__name__}: {e}"
            finally:
                with self.lock:
                    self.pending.discard(k)
        if block:
            run()
            return True
        threading.Thread(target=run, name="gen-hooks", daemon=True).start()
        return True

    # -- what the melody uses -----------------------------------------------
    def provider(self, style: str):
        """A callable(rng) -> hook dict or None, bound to a style."""
        def pick(rng: random.Random):
            hooks = self.cache.get(style) or []
            if not hooks:
                return None
            return dict(hooks[rng.randrange(len(hooks))])
        return pick

    def status(self):
        return {"enabled": self.enabled, "cached": {k: len(v) for k, v in self.cache.items()},
                "pending": len(self.pending), "error": self.error}
