"""
ElevenLabs Music batch generator.

Reads a prompts markdown file (default: projects/fan/forest_music_prompts.md),
prepends its "House style" block to every numbered prompt, and generates one
or more takes of each through the ElevenLabs Music API (POST /v1/music).

Each take is saved as an mp3 with a .txt sidecar holding the exact prompt and
parameters that produced it, so a good take can be reproduced or tweaked.

Requires a paid ElevenLabs plan (the Music API is not on the free tier).

Usage:
    python tools/media/music_elevenlabs.py --list
    python tools/media/music_elevenlabs.py --dry-run
    python tools/media/music_elevenlabs.py                     # everything, 1 take each
    python tools/media/music_elevenlabs.py --only 1,6,11 --takes 3
    python tools/media/music_elevenlabs.py --only dawn,storm --length 240
    python tools/media/music_elevenlabs.py --prompts other.md --out some/dir

API key priority:
    1. ELEVENLABS_API_KEY environment variable (recommended)
    2. --api-key argument
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS = REPO_ROOT / "projects" / "fan" / "forest_music_prompts.md"
API_URL = "https://api.elevenlabs.io/v1/music"
DEFAULT_MODEL = "music_v2_5"
DEFAULT_LENGTH_S = 240          # the section timings in the prompts assume 4:00
MAX_LENGTH_S = 600
MIN_LENGTH_S = 3
BUSY_WAIT_S = 10                # pause after a 429 (concurrency cap) before retrying
BUSY_MAX_WAITS = 18             # give up after this many in a row (3 minutes)
PROMPT_MAX_CHARS = 2000         # hard API limit for /v1/music prompts
PROMPT_WARN_CHARS = 1900        # leave headroom for edits


@dataclass
class Prompt:
    number: int
    title: str
    states: str
    body: str

    @property
    def slug(self) -> str:
        s = self.title.lower()
        s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
        return f"{self.number:02d}_{s}"


# ── Markdown parsing ───────────────────────────────────────────────────────

_SECTION_RE = re.compile(r"^## (\d+)\.\s+(.+?)(?:\s+\((.*?)\))?\s*$")


def parse_prompts(path: Path):
    """Return (house_style, [Prompt, ...]) from the markdown file."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    house = []
    prompts = []
    mode = None            # None | "house" | "prompt"
    current = None
    buf = []

    def flush():
        nonlocal current, buf
        if current is not None:
            current.body = "\n".join(buf).strip()
            prompts.append(current)
        current = None
        buf = []

    for line in lines:
        if line.startswith("## "):
            flush()
            m = _SECTION_RE.match(line)
            if m:
                mode = "prompt"
                current = Prompt(int(m.group(1)), m.group(2).strip(),
                                 (m.group(3) or "").strip(), "")
                buf = []
            elif line.lower().startswith("## house style"):
                mode = "house"
            else:
                mode = None
            continue
        if line.strip() == "---":
            flush()
            mode = None
            continue
        if mode == "house":
            house.append(line)
        elif mode == "prompt":
            buf.append(line)
    flush()

    house_text = "\n".join(house).strip()
    if not house_text:
        print(f"WARNING: no '## House style' block found in {path}")
    if not prompts:
        print(f"ERROR: no '## N. Title' prompt sections found in {path}")
        sys.exit(1)
    return house_text, prompts


def save_prompts(path: Path, house: str, prompts) -> None:
    """Write edited house style / prompt bodies back into the markdown,
    leaving headers, intro text and the troubleshooting section untouched."""
    lines = path.read_text(encoding="utf-8").splitlines()
    bodies = {p.number: p.body for p in prompts}
    out = []
    mode = None
    pending = None          # replacement text to emit when the section closes
    original_had_content = False

    def close():
        nonlocal pending, original_had_content
        if pending is not None:
            out.append("")
            out.append(pending.rstrip())
            out.append("")
        pending = None
        original_had_content = False

    for line in lines:
        is_header = line.startswith("## ")
        is_rule = line.strip() == "---"
        if is_header or is_rule:
            close()
            if is_header:
                m = _SECTION_RE.match(line)
                if m and int(m.group(1)) in bodies:
                    mode = "replace"; pending = bodies[int(m.group(1))]
                elif line.lower().startswith("## house style"):
                    mode = "replace"; pending = house
                else:
                    mode = None
            else:
                mode = None
            out.append(line)
            continue
        if mode == "replace":
            continue            # drop original body; replacement emitted at close()
        out.append(line)
    close()
    # Collapse runs of blank lines introduced by close().
    cleaned = []
    for line in out:
        if line == "" and cleaned and cleaned[-1] == "":
            continue
        cleaned.append(line)
    path.write_text("\n".join(cleaned).rstrip() + "\n", encoding="utf-8")


def take_path(out_dir: Path, p: Prompt, take: int) -> Path:
    return out_dir / f"forest_{p.slug}_take{take}.mp3"


def find_takes(out_dir: Path, p: Prompt):
    """Existing mp3 takes for a prompt, sorted by take number."""
    if not out_dir.exists():
        return []
    files = list(out_dir.glob(f"forest_{p.slug}_take*.mp3"))

    def key(f):
        m = re.search(r"_take(\d+)\.mp3$", f.name)
        return int(m.group(1)) if m else 0
    return sorted(files, key=key)


def next_take_number(out_dir: Path, p: Prompt) -> int:
    takes = find_takes(out_dir, p)
    if not takes:
        return 1
    m = re.search(r"_take(\d+)\.mp3$", takes[-1].name)
    return (int(m.group(1)) if m else len(takes)) + 1


def write_sidecar(out_path: Path, p: Prompt, model: str, length_ms: int,
                  seed, full_prompt: str) -> None:
    out_path.with_suffix(".txt").write_text(
        f"title: {p.title}\nstates: {p.states}\nmodel: {model}\n"
        f"length_ms: {length_ms}\nseed: {seed}\n\n{full_prompt}\n",
        encoding="utf-8")


def default_out_dir(prompts_path: Path) -> Path:
    return prompts_path.parent / "media" / "sounds" / prompts_path.stem


def build_prompt(house: str, p: Prompt) -> str:
    # Collapse hard-wrapped lines inside paragraphs; keep paragraph breaks.
    def unwrap(block: str) -> str:
        paras = re.split(r"\n\s*\n", block.strip())
        return "\n\n".join(" ".join(x.split()) for x in paras)
    parts = [unwrap(house)] if house else []
    parts.append(unwrap(p.body))
    return "\n\n".join(parts)


def select_prompts(prompts, only: str):
    if not only:
        return prompts
    wanted = [t.strip().lower() for t in only.split(",") if t.strip()]
    chosen = []
    for p in prompts:
        for w in wanted:
            if w.isdigit() and int(w) == p.number:
                chosen.append(p); break
            if not w.isdigit() and (w in p.title.lower() or w in p.states.lower()):
                chosen.append(p); break
    if not chosen:
        print(f"ERROR: --only '{only}' matched nothing. Use --list to see prompts.")
        sys.exit(1)
    return chosen


# ── API ────────────────────────────────────────────────────────────────────

def get_api_key(arg_key):
    key = os.environ.get("ELEVENLABS_API_KEY") or arg_key
    if not key:
        print("ERROR: No API key found. Set ELEVENLABS_API_KEY env var or pass --api-key.")
        sys.exit(1)
    return key


def compose(api_key: str, prompt: str, length_ms: int, model: str,
            seed, out_path: Path, retries: int = 3, log=print,
            should_stop=lambda: False) -> bool:
    """POST one prompt, stream the mp3 to out_path. Returns True on success.

    log(msg) receives progress/error lines; should_stop() is polled while
    downloading so a GUI can cancel mid-stream.
    """
    import requests

    print_ = log
    body = {
        "prompt": prompt,
        "music_length_ms": length_ms,
        "model_id": model,
        "force_instrumental": True,
    }
    if seed is not None:
        body["seed"] = seed
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    params = {"output_format": "mp3_44100_192"}

    busy_waits = 0
    attempt = 0
    while attempt < retries:
        attempt += 1
        if should_stop():
            return False
        try:
            r = requests.post(API_URL, headers=headers, params=params,
                              json=body, stream=True, timeout=(30, 900))
        except requests.RequestException as e:
            print_(f"    network error: {e}")
            if attempt < retries and not should_stop():
                time.sleep(5 * attempt)
                continue
            return False

        if r.status_code == 200:
            tmp = out_path.with_suffix(".part")
            try:
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if should_stop():
                            print_("    cancelled")
                            return False
                        if chunk:
                            f.write(chunk)
                tmp.replace(out_path)
                return True
            finally:
                if tmp.exists():
                    tmp.unlink()

        detail = _error_detail(r)
        if r.status_code in (401, 403):
            print_(f"    auth error {r.status_code}: {detail}")
            return False
        if r.status_code == 402 or "paid" in detail.lower():
            print_(f"    {r.status_code}: {detail}")
            print_("    The Music API needs a paid ElevenLabs plan.")
            return False
        if r.status_code == 422:
            print_(f"    request rejected (422): {detail}")
            return False
        if r.status_code == 429:
            # Concurrency cap or rate limit: wait for a slot. This is not a
            # real failure, so it doesn't count against the retry budget.
            busy_waits += 1
            if busy_waits > BUSY_MAX_WAITS:
                print_(f"    429 for {BUSY_MAX_WAITS * BUSY_WAIT_S}s straight, giving up: {detail}")
                return False
            print_(f"    busy (429), waiting {BUSY_WAIT_S}s for a slot: {detail}")
            attempt -= 1
            time.sleep(BUSY_WAIT_S)
            continue
        if r.status_code >= 500:
            wait = 15 * attempt
            print_(f"    {r.status_code}: {detail} — retrying in {wait}s")
            time.sleep(wait)
            continue
        print_(f"    unexpected {r.status_code}: {detail}")
        return False
    return False


def _error_detail(r) -> str:
    try:
        j = r.json()
        d = j.get("detail", j)
        if isinstance(d, dict):
            return d.get("message") or json.dumps(d)[:400]
        return str(d)[:400]
    except Exception:
        return (r.text or "")[:400]


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Batch-generate music from a prompts markdown via ElevenLabs")
    ap.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS,
                    help=f"prompts markdown (default: {DEFAULT_PROMPTS.relative_to(REPO_ROOT)})")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: <prompts dir>/media/sounds/<prompts stem>/)")
    ap.add_argument("--only", help="comma list of prompt numbers or name fragments, e.g. 1,6 or dawn,storm")
    ap.add_argument("--takes", type=int, default=1, help="takes per prompt (default 1)")
    ap.add_argument("--length", type=float, default=DEFAULT_LENGTH_S,
                    help=f"track length in seconds, {MIN_LENGTH_S}-{MAX_LENGTH_S} (default {DEFAULT_LENGTH_S})")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"model id (default {DEFAULT_MODEL})")
    ap.add_argument("--seed", type=int, help="fixed seed; takes use seed, seed+1, ...")
    ap.add_argument("--no-house-style", action="store_true", help="send the prompt body only")
    ap.add_argument("--force", action="store_true", help="regenerate takes that already exist")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds to wait between requests")
    # The CLI is serial; use tools/music_studio.py for concurrent generation.
    ap.add_argument("--list", action="store_true", help="list parsed prompts and exit")
    ap.add_argument("--dry-run", action="store_true", help="show what would be sent, no API calls")
    ap.add_argument("--api-key", help="ElevenLabs API key (prefer ELEVENLABS_API_KEY env var)")
    args = ap.parse_args()

    if not args.prompts.exists():
        print(f"ERROR: prompts file not found: {args.prompts}")
        sys.exit(1)
    if not (MIN_LENGTH_S <= args.length <= MAX_LENGTH_S):
        print(f"ERROR: --length must be between {MIN_LENGTH_S} and {MAX_LENGTH_S} seconds")
        sys.exit(1)

    house, prompts = parse_prompts(args.prompts)
    if args.no_house_style:
        house = ""

    if args.list:
        print(f"{len(prompts)} prompts in {args.prompts}\n")
        for p in prompts:
            print(f"{p.number:2d}. {p.title:<22} {p.states}")
        return

    chosen = select_prompts(prompts, args.only)
    out_dir = args.out or default_out_dir(args.prompts)
    length_ms = int(round(args.length * 1000))

    total = len(chosen) * args.takes
    print(f"{len(chosen)} prompts x {args.takes} take(s) = {total} tracks, "
          f"{args.length:.0f}s each, model {args.model}")
    print(f"output: {out_dir}\n")

    if args.dry_run:
        for p in chosen:
            full = build_prompt(house, p)
            print(f"=== {p.slug}  ({len(full)} chars) ===")
            print(full)
            print()
        return

    api_key = get_api_key(args.api_key)
    out_dir.mkdir(parents=True, exist_ok=True)

    done = skipped = failed = 0
    n = 0
    for p in chosen:
        full = build_prompt(house, p)
        if len(full) > PROMPT_MAX_CHARS:
            print(f"  SKIP {p.slug}: prompt is {len(full)} chars, the API limit is {PROMPT_MAX_CHARS}")
            failed += args.takes
            n += args.takes
            continue
        if len(full) > PROMPT_WARN_CHARS:
            print(f"  note: {p.slug} prompt is {len(full)} chars, close to the {PROMPT_MAX_CHARS} limit")
        for take in range(1, args.takes + 1):
            n += 1
            out_path = take_path(out_dir, p, take)
            print(f"[{n}/{total}] {out_path.name}")
            if out_path.exists() and not args.force:
                print("    exists, skipping (use --force to regenerate)")
                skipped += 1
                continue
            seed = None if args.seed is None else args.seed + (take - 1)
            t0 = time.time()
            ok = compose(api_key, full, length_ms, args.model, seed, out_path)
            if ok:
                size_mb = out_path.stat().st_size / 1e6
                print(f"    saved {size_mb:.1f} MB in {time.time() - t0:.0f}s")
                write_sidecar(out_path, p, args.model, length_ms, seed, full)
                done += 1
            else:
                failed += 1
                if failed and done == 0 and n == 1:
                    # First request failed outright: probably key/plan, don't burn the rest.
                    print("\nFirst request failed; stopping. Fix the issue and re-run.")
                    _summary(done, skipped, failed)
                    sys.exit(1)
            if n < total:
                time.sleep(args.delay)

    _summary(done, skipped, failed)
    if failed:
        sys.exit(1)


def _summary(done, skipped, failed):
    print(f"\ndone: {done}  skipped: {skipped}  failed: {failed}")


if __name__ == "__main__":
    main()
