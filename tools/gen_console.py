"""Native generative-music console (PyQt6) - the operator surface WITHOUT
a browser. Renders the same spec as the web page (lib/gen/ui.py) with
native widgets and drives either this machine or the show box.

    # Play here (audio on this machine), full console:
    python tools/gen_console.py [--style groove --key 8A --fluid-slots keys,pad]

    # Remote-control the running show (or tools/gen/gen_server.py):
    python tools/gen_console.py --remote http://lucifera.local:5000

    # Dry run with no sound device (composer runs, nothing is played):
    python tools/gen_console.py --headless

Shortcuts: Space start/stop · Esc stop · Ctrl+Q quit.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--remote", metavar="URL", help="control a running show / gen_server instead of playing here")
    ap.add_argument("--headless", action="store_true", help="no audio device: the console pumps the composer itself")
    ap.add_argument("--style", default="groove")
    ap.add_argument("--bpm", type=float, default=None)
    ap.add_argument("--key", default="8A")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fluid-slots", default="")
    ap.add_argument("--soundfont", default=None)
    ap.add_argument("--set-length", type=float, default=10800.0)
    ap.add_argument("--autostart", action="store_true")
    args = ap.parse_args()
    from tools.gen.console.app import run
    if args.remote:
        from tools.gen.console.backend import RemoteBackend
        backend = RemoteBackend(args.remote)
    else:
        from tools.gen.console.backend import LocalBackend
        backend = LocalBackend({"style": args.style, "bpm": args.bpm, "key": args.key, "seed": args.seed,
                                "fluid_slots": args.fluid_slots, "soundfont": args.soundfont,
                                "set_length_s": args.set_length, "log_dir": "logs"}, audio=not args.headless)
        if args.autostart:
            backend.start()
    return run(backend, sys.argv)


if __name__ == "__main__":
    sys.exit(main())
