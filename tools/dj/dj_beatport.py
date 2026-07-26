"""Beatport CLI: log in, search, fit-check against your library, and manage
a buy-list you open in the browser to purchase.

Beatport has no cart API - discovery and fit-scoring are automated here; the
actual add-to-cart/buy is a browser click on the track page. See
lib/dj/beatport.py for the auth story.

Usage:
    # 1. Authenticate (pick ONE):
    python tools/dj/dj_beatport.py login --paste          # paste token JSON
    python tools/dj/dj_beatport.py login --pkce            # loopback OAuth (needs
                                                        # BEATPORT_CLIENT_ID)

    # 2. Discover + fit:
    python tools/dj/dj_beatport.py search "melodic house" --bpm 118-126
    python tools/dj/dj_beatport.py search "lane 8" --fit    # + fit vs your library
    python tools/dj/dj_beatport.py fit 12345678             # deep fit one track id

    # 3. Buy-list:
    python tools/dj/dj_beatport.py wish add 12345678
    python tools/dj/dj_beatport.py wish list
    python tools/dj/dj_beatport.py wish open                # open all in browser
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from lib.dj import resolve_music_dir
from lib.dj import beatport as BP


def _client():
    c = BP.BeatportClient()
    if not c.available():
        print("Not authenticated. Run:  python tools/dj/dj_beatport.py login "
              "--paste")
        sys.exit(2)
    return c


def _load_library(args):
    from lib.dj.db import LibraryDB
    from lib.dj.brain import load_library
    root = resolve_music_dir(args.dir)
    return load_library(LibraryDB(root)), root


def cmd_login(args):
    auth = BP.BeatportAuth()
    if args.pkce:
        print("Opening the browser for Beatport authorization...")
        BP.pkce_login(auth)
        print(f"Logged in. Token stored at {auth.token_path}")
        return 0
    # Manual paste.
    print("Log into the Beatport API docs in your browser, open the network "
          "tab,\nfind the token response, and paste the JSON here.\n"
          f"({BP.DOCS_LOGIN_URL})\n\nPaste token JSON, then Ctrl-D "
          "(Ctrl-Z on Windows) + Enter:")
    blob = sys.stdin.read().strip()
    auth.set_token_json(blob)
    print(f"Saved. Token stored at {auth.token_path}")
    return 0


def _bpm_range(spec):
    if not spec:
        return None, None
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return float(lo), float(hi)
    v = float(spec)
    return v - 3, v + 3


def cmd_search(args):
    client = _client()
    filters = {}
    lo, hi = _bpm_range(args.bpm)
    if lo:
        filters["bpm_low"], filters["bpm_high"] = int(lo), int(hi)
    tracks = client.search(args.query, per_page=args.limit, **filters)
    rows = [BP.beatport_row(t) for t in tracks]
    lib = None
    if args.fit:
        lib, _ = _load_library(args)
    print(f"{len(rows)} results\n")
    for r in rows:
        line = (f"  {r['bp_id']:>10}  {r['title'][:34]:34s} "
                f"{r['artist'][:22]:22s} {r['bpm']:5.0f} {r['camelot']:>3s} "
                f"{str(r['price'] or ''):>6s}")
        if lib is not None and r["camelot"]:
            f = BP.fit_vs_library(lib, BP.ghost_trackinfo(r), top=1)
            nb = f["mixable_neighbours"]
            best = f["best"][0] if f["best"] else None
            line += f"   fit: {nb} neighbours" + (
                f", best {best['title'][:16]} ({best['key_fit']})" if best
                else "")
        print(line)
    return 0


def cmd_fit(args):
    client = _client()
    lib, _ = _load_library(args)
    trk = client.track(args.track_id)
    row = BP.beatport_row(trk)
    print(f"{row['title']} - {row['artist']}  ({row['bpm']:.0f} bpm "
          f"{row['camelot']}, {row['price']})\n{row['url']}\n")
    if args.deep:
        print("Analyzing the preview clip...")
        ghost = BP.deep_ghost(client, row)
        print(f"  measured: {ghost.bpm:.1f} bpm  {ghost.camelot}  "
              f"grid_conf {ghost.bpm_conf:.2f}  {len(ghost.sections)} sections")
    else:
        ghost = BP.ghost_trackinfo(row)
    f = BP.fit_vs_library(lib, ghost)
    print(f"\n  {f['mixable_neighbours']}/{f['library_size']} library tracks "
          f"are tempo-reachable. Best matches:")
    for b in f["best"]:
        print(f"    {b['title'][:30]:30s} {b['artist'][:18]:18s} "
              f"{b['bpm']:5.0f} {b['key']:>3s}  key_fit {b['key_fit']}")
    return 0


def cmd_wish(args):
    root = resolve_music_dir(args.dir)
    wl = BP.Wishlist(root)
    if args.action == "list":
        if not wl.items:
            print("Wishlist is empty.")
            return 0
        for it in wl.items:
            print(f"  {it['bp_id']:>10}  {it['title'][:36]:36s} "
                  f"{it['artist'][:22]:22s} {it['bpm']:5.0f} "
                  f"{it['camelot']:>3s} {str(it.get('price') or ''):>6s}")
        print(f"\n{len(wl.items)} tracks. "
              f"`wish open` to add them to your cart on Beatport.")
    elif args.action == "add":
        client = _client()
        row = BP.beatport_row(client.track(args.track_id))
        print("added" if wl.add(row) else "already on the list",
              f": {row['title']} - {row['artist']}")
    elif args.action == "remove":
        print(f"removed {wl.remove(args.track_id)} item(s)")
    elif args.action == "open":
        n = wl.open_in_browser(args.track_id)
        print(f"opened {n} Beatport page(s) - add to cart there.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="", help="music library directory")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("login")
    p.add_argument("--paste", action="store_true", help="paste token JSON")
    p.add_argument("--pkce", action="store_true", help="loopback OAuth")
    p.set_defaults(fn=cmd_login)

    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--bpm", default="", help="e.g. 120 or 118-126")
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--fit", action="store_true",
                   help="score each result against your library")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("fit")
    p.add_argument("track_id", type=int)
    p.add_argument("--deep", action="store_true",
                   help="analyze the preview clip (measured bpm/key/grid)")
    p.set_defaults(fn=cmd_fit)

    p = sub.add_parser("wish")
    p.add_argument("action", choices=["list", "add", "remove", "open"])
    p.add_argument("track_id", type=int, nargs="?")
    p.set_defaults(fn=cmd_wish)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
