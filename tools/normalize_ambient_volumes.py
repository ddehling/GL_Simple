"""Normalize per-state Sound_volume so ambient backgrounds have roughly
matched loudness.

Scans every weather state's ambient_sound file, measures its RMS dBFS,
and computes a Sound_volume multiplier that brings it toward a target
level (default -20 dBFS). Prints a report; pass --write to persist the
volumes back to the project's weather_params.py.

BACKGROUND AMBIENT ONLY - one-shot / event sounds are not touched. The
engine applies Sound_volume to the active state's ambient track at mix
time (audio_engine.AudioEngine.ambient_volume).

Usage:
    python tools/normalize_ambient_volumes.py                # dry-run, fan project
    python tools/normalize_ambient_volumes.py --target -18   # louder target
    python tools/normalize_ambient_volumes.py --write        # apply changes
    python tools/normalize_ambient_volumes.py --project ocean

Files used by multiple states share one measurement (and therefore one
volume), so the result is consistent regardless of which state plays the
file.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import miniaudio

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def measure_rms_dbfs(path):
    """Return (rms_dbfs, error_msg). Decodes via miniaudio (handles WAV /
    FLAC / OGG / MP3) and computes the broadband RMS in dBFS. Mono mix at
    22050 Hz is plenty for a loudness estimate."""
    try:
        decoded = miniaudio.decode_file(
            str(path),
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=22050,
        )
    except Exception as e:
        return None, f"decode failed: {e}"
    samples = np.frombuffer(decoded.samples, dtype=np.int16).astype(np.float64) / 32768.0
    if samples.size == 0:
        return None, "no samples"
    rms = float(np.sqrt(np.mean(samples * samples)))
    if rms <= 1e-6:
        return None, "silent"
    return 20.0 * np.log10(rms), None


def load_project(name):
    proj = REPO_ROOT / "projects" / name
    wp_path = proj / "weather_params.py"
    if not wp_path.exists():
        sys.exit(f"no weather_params.py at {wp_path}")
    spec = importlib.util.spec_from_file_location("project_wp", wp_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return proj, mod


def _state_key(s):
    return s.value if hasattr(s, "value") else str(s)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--project", default="fan",
                    help="Project id under projects/ (default: fan)")
    ap.add_argument("--target", type=float, default=-20.0,
                    help="Target RMS dBFS (default: -20.0)")
    ap.add_argument("--min-volume", type=float, default=0.20,
                    help="Floor on Sound_volume multiplier (default: 0.20)")
    ap.add_argument("--max-volume", type=float, default=2.50,
                    help="Ceiling on Sound_volume multiplier (default: 2.50)")
    ap.add_argument("--write", action="store_true",
                    help="Update Sound_volume in the project's weather_params.py")
    args = ap.parse_args()

    proj, mod = load_project(args.project)
    sounds_dir = proj / "media" / "sounds"
    presets = mod.WEATHER_PRESETS

    # Collect file -> [(state, current_Sound_volume), ...]
    file_uses = {}
    for state, params in presets.items():
        fname = params.get("ambient_sound")
        if not fname:
            continue
        cur = float(params.get("Sound_volume", 1.0))
        file_uses.setdefault(fname, []).append((state, cur))

    if not file_uses:
        print("(no states reference an ambient_sound)")
        return

    print(f"Scanning {len(file_uses)} unique ambient files in {sounds_dir}")
    print(f"Target RMS = {args.target:+.1f} dBFS  "
          f"(clamp volume to {args.min_volume:.2f}..{args.max_volume:.2f})\n")
    header = f"  {'file':45s}  {'dBFS':>7s}  {'gain dB':>7s}  {'vol':>5s}  used by"
    print(header)
    print("  " + "-" * (len(header) + 30))

    file_volumes = {}
    for fname in sorted(file_uses.keys()):
        path = sounds_dir / fname
        if not path.exists():
            print(f"  [MISSING] {fname}")
            continue
        dbfs, err = measure_rms_dbfs(path)
        if dbfs is None:
            print(f"  [SKIP] {fname}: {err}")
            continue
        gain_db = args.target - dbfs
        vol = 10.0 ** (gain_db / 20.0)
        vol = max(args.min_volume, min(args.max_volume, vol))
        vol = round(vol, 2)
        file_volumes[fname] = vol

        used = file_uses[fname]
        states_str = ", ".join(_state_key(s) for s, _ in used[:3])
        if len(used) > 3:
            states_str += f"  (+{len(used) - 3} more)"
        # short filename for display
        short = fname if len(fname) <= 45 else (fname[:42] + "...")
        print(f"  {short:45s}  {dbfs:7.1f}  {gain_db:+7.1f}  {vol:5.2f}  {states_str}")

    if not args.write:
        print(f"\n(dry run; pass --write to update "
              f"projects/{args.project}/weather_params.py)")
        return

    # Build the new presets dict with updated Sound_volume, then save via the
    # (now-fixed) editor serializer so PARAMETER_DEFINITIONS and
    # DEFAULT_WEATHER_PARAMS are preserved verbatim.
    new_presets = {}
    changes = []
    for state, params in presets.items():
        p = dict(params)
        fname = p.get("ambient_sound")
        if fname in file_volumes:
            new_vol = file_volumes[fname]
            old_vol = float(p.get("Sound_volume", 1.0))
            if abs(old_vol - new_vol) > 1e-3:
                changes.append((_state_key(state), old_vol, new_vol))
            p["Sound_volume"] = new_vol
        new_presets[_state_key(state)] = p

    states_list = [_state_key(s) for s in mod.WeatherState]
    sets = {k: dict(v) for k, v in mod.WEATHER_SETS.items()}
    global_params = list(getattr(mod, "GLOBAL_PARAMETERS", []) or [])

    from lib.weather_editor_utils import save_weather_params
    result = save_weather_params(
        states_list, new_presets, sets,
        global_parameters=global_params or None,
        target_path=proj / "weather_params.py",
    )
    if not result.get("success"):
        sys.exit(f"save failed: {result.get('error')}")

    print(f"\n[OK] {len(changes)} Sound_volume changes written to "
          f"{proj / 'weather_params.py'}")
    if changes:
        print(f"     backup: {proj / 'weather_params.py.backup'}")
        print("\nDiffs:")
        for st, old, new in changes:
            print(f"  {st:28s}  {old:5.2f} -> {new:5.2f}")


if __name__ == "__main__":
    main()
