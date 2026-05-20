"""
Standalone test — does the Claude CLI cache large prompts?

Calls ``claude -p`` exactly the way ``narrative_editor_qt.py`` does, with
a large system prompt + a small per-call question. Repeats with the
SAME system prompt and different questions, measuring wall-clock latency
each time.

If the CLI is caching the system-prompt prefix:
  • Call #1 is full-cost (cache write)
  • Calls #2+ are noticeably faster (cache read)
  • Output JSON (when available) shows ``cache_read_input_tokens > 0``

If not caching:
  • All calls take similar time

Run from repo root:
    python tools/test_claude_caching.py
    python tools/test_claude_caching.py --size 60000
    python tools/test_claude_caching.py --model claude-opus-4-7
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def find_claude() -> str:
    """Find the claude CLI. Checks PATH plus the Claude Code install
    locations on Windows / macOS / Linux."""
    found = shutil.which("claude")
    if found:
        return found

    candidates = [
        Path.home() / "AppData" / "Roaming" / "npm" / "claude",
        Path.home() / "AppData" / "Roaming" / "npm" / "claude.cmd",
        Path.home() / ".local" / "bin" / "claude",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    # Claude Code Windows install: AppData\Roaming\Claude\claude-code\<version>\claude.exe
    cc_root = Path.home() / "AppData" / "Roaming" / "Claude" / "claude-code"
    if cc_root.exists():
        version_dirs = sorted([p for p in cc_root.iterdir() if p.is_dir()],
                               reverse=True)  # newest first
        for vd in version_dirs:
            for name in ("claude.exe", "claude.cmd", "claude"):
                cand = vd / name
                if cand.exists():
                    return str(cand)

    raise RuntimeError("claude CLI not found on PATH or in standard locations")


# Stable "world bible" text. The questions only ever ask about facts in
# this bible, so the response is deterministic and the system prompt is
# what should get cached.
BIBLE_FRAGMENT = """\

The city of Aethelmark sits at the confluence of three rivers, each
carrying water from a different cardinal direction. Its name in the
old tongue means "watching-place by the joining." The original
settlement was a watch-fort built on the central island, but over
generations the fort fell, the walls were cannibalized for housing,
and the island became the densest neighborhood of the modern city.

The river to the north is called the Garnet; it runs swift and cold
from glaciers in the Spinescore Mountains. The river to the east is
the Loam, slow and silt-heavy, draining the fertile lowlands of Veldt.
The third river, joining from the south, is the Black — named not for
its color but for a fungal bloom that turns its banks dark in summer.

Aethelmark is ruled by a council of seven Chairmasters: one for each
of the four cardinal districts, plus three for the central island.
The current First Chairmaster is Inken Bel Soris, a former physician
who came to office sixteen years ago after the cholera reforms. Her
opposition on the council is led by the Black-quarter Chairmaster,
Vrael Hand, who advocates for tearing down the bridges over the
Black to control immigration from the southern villages.

The city's currency is the iron crown, divided into twelve copper
moons and one hundred forty-four glass pence. Iron crowns are struck
only at the Spire Mint on the central island and bear the silhouette
of the lost watch-fort tower. Glass pence — actually small disks of
fired clay coated in a thin glaze — break easily and are replaced
constantly; only the freshly issued ones are accepted in payment.

"""


def build_context(target_chars: int) -> str:
    """Repeat the bible fragment until target_chars is reached. The
    repetition is intentional: identical content across calls should
    be maximally cache-friendly."""
    out: list[str] = []
    while sum(len(p) for p in out) < target_chars:
        out.append(BIBLE_FRAGMENT)
    return ''.join(out)[:target_chars]


def call_claude(exe: str, system: str, prompt: str,
                model: str, output_format: str = 'json',
                timeout: int = 180):
    """Single call. Returns (elapsed_seconds, returncode, stdout, stderr)."""
    cmd = [
        exe,
        "--no-session-persistence",
        "--model", model,
        "--system-prompt", system,
        "--output-format", output_format,
        "-p", prompt,
    ]
    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return time.time() - t0, result.returncode, result.stdout, result.stderr


def find_cache_stats(json_str: str) -> dict:
    """Try to extract usage / cache stats from the JSON response.
    The Claude CLI's JSON shape varies across versions, so we scan
    recursively for any key containing 'token' or 'cache'."""
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return {}

    found: dict = {}

    def visit(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                kl = k.lower()
                if ('token' in kl or 'cache' in kl) and isinstance(v, (int, float)):
                    found[f"{path}.{k}".lstrip('.')] = v
                visit(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                visit(v, f"{path}[{i}]")

    visit(data)
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--size', type=int, default=30000,
                    help='Approximate context size in characters (default 30000)')
    ap.add_argument('--model', default='claude-sonnet-4-6',
                    help='Model id (default claude-sonnet-4-6)')
    ap.add_argument('--rounds', type=int, default=3,
                    help='Number of calls to make (default 3)')
    ap.add_argument('--format', default='json', choices=('json', 'text'),
                    help='Output format from CLI (default json — for cache stats)')
    args = ap.parse_args()

    try:
        exe = find_claude()
    except RuntimeError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"claude exe : {exe}")
    print(f"model      : {args.model}")
    print(f"output fmt : {args.format}")

    context = build_context(args.size)
    print(f"context    : {len(context):,} chars (~{len(context)//4:,} tokens)")

    system = (
        "You are a research assistant. Answer the user's question using "
        "ONLY the world-bible content below. Be concise — one short "
        "sentence is enough. Do not invent facts.\n\n"
        "----- WORLD BIBLE -----\n"
        + context +
        "\n----- END BIBLE -----"
    )

    questions = [
        "What is the name of the southernmost river?",
        "Who is the current First Chairmaster?",
        "What is the currency of Aethelmark called?",
        "How many Chairmasters sit on the council?",
        "What metal are the city's primary coins made of?",
    ][:args.rounds]

    print(f"\nMaking {len(questions)} calls back-to-back...\n")

    rows = []
    for i, q in enumerate(questions, 1):
        print(f"--- call {i}/{len(questions)} -----------------------------------")
        print(f"question: {q}")
        elapsed, rc, out, err = call_claude(
            exe, system, q, model=args.model, output_format=args.format)
        print(f"elapsed : {elapsed:.2f}s  rc={rc}  output bytes={len(out)}")
        if rc != 0:
            print(f"STDERR: {err.strip()[:400]}")
            continue

        # Try to extract cache stats
        cache_stats = find_cache_stats(out) if args.format == 'json' else {}
        if cache_stats:
            interesting = ', '.join(f"{k}={v}" for k, v in cache_stats.items())
            print(f"stats   : {interesting}")
        else:
            # Print the first 240 chars so the user can see what came back
            preview = out.strip().replace('\n', ' ')[:240]
            print(f"output  : {preview}{'…' if len(out) > 240 else ''}")
        rows.append((i, q, elapsed, cache_stats))
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if not rows:
        print("No successful calls.")
        return
    print(f"{'call':>4}  {'time':>7}  {'cache_read':>11}  {'cache_write':>11}  question")
    for i, q, elapsed, stats in rows:
        # Look for common cache-token keys
        read = (stats.get('cache_read_input_tokens')
                or stats.get('result.usage.cache_read_input_tokens')
                or stats.get('usage.cache_read_input_tokens') or '-')
        write = (stats.get('cache_creation_input_tokens')
                 or stats.get('result.usage.cache_creation_input_tokens')
                 or stats.get('usage.cache_creation_input_tokens') or '-')
        print(f"{i:>4}  {elapsed:>6.2f}s  {str(read):>11}  {str(write):>11}  {q[:40]}")

    times = [r[2] for r in rows]
    if len(times) >= 2:
        first = times[0]
        rest = sum(times[1:]) / (len(times) - 1)
        delta = (first - rest) / first * 100
        print()
        print(f"first-call latency : {first:.2f}s")
        print(f"avg of subsequent  : {rest:.2f}s")
        print(f"speedup            : {delta:+.1f}%")
        if delta > 25:
            print("VERDICT: subsequent calls ~25%+ faster — caching looks active.")
        elif rows[0][3] and any('cache_read' in k and v
                                for k, v in (rows[-1][3] or {}).items()):
            print("VERDICT: JSON shows cache_read_input_tokens > 0 — caching active.")
        else:
            print("VERDICT: no clear cache effect. CLI may not auto-cache the")
            print("         system prompt in `-p` (one-shot) mode, or context")
            print("         is too small to trigger the 1024-token threshold.")


if __name__ == '__main__':
    main()
