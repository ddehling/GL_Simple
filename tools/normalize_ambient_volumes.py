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


_SR = 22050   # mono mix rate for the loudness estimate


def decode_mono(path):
    """Decode a sound file to a float64 mono array at _SR Hz. Cached by caller.
    Returns (samples, error_msg)."""
    try:
        decoded = miniaudio.decode_file(
            str(path),
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=_SR,
        )
    except Exception as e:
        return None, f"decode failed: {e}"
    samples = np.frombuffer(decoded.samples, dtype=np.int16).astype(np.float64) / 32768.0
    if samples.size == 0:
        return None, "no samples"
    return samples, None


def window_rms_dbfs(samples, skip_sec, ari_sec):
    """Return (rms_dbfs, peak_dbfs, played_seconds, error_msg) for the SLICE
    of `samples` that actually plays under (skip_seconds, ARI):

      window = samples[skip_sec : skip_sec + ari_sec]   if ari_sec > 0
             = samples[skip_sec : end]                   if ari_sec == 0  (loop at EOF)

    Mirrors AudioEngine.play_ambient semantics so a state's measurement
    reflects only the audio it actually plays, not silent intros or peaks
    outside the loop window. ``peak_dbfs`` is the window's true-peak proxy
    (max |sample|), used to cap the volume so boosting a quiet track can't
    push its peaks into clipping.
    """
    total = samples.size
    if total == 0:
        return None, None, 0.0, "no samples"
    start = int(max(0.0, skip_sec) * _SR)
    if start >= total:
        return None, None, 0.0, "skip past end of file"
    if ari_sec > 0.0:
        end = min(total, start + int(ari_sec * _SR))
    else:
        end = total
    win = samples[start:end]
    if win.size == 0:
        return None, None, 0.0, "empty window"
    rms = float(np.sqrt(np.mean(win * win)))
    peak = float(np.max(np.abs(win)))
    played = win.size / _SR
    if rms <= 1e-6:
        return None, None, played, "silent window"
    peak_dbfs = 20.0 * np.log10(peak) if peak > 1e-9 else -120.0
    return 20.0 * np.log10(rms), peak_dbfs, played, None


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
    ap.add_argument("--max-peak", type=float, default=-1.5,
                    help="True-peak ceiling in dBFS; a track's volume is "
                         "capped so its loudest sample stays at/below this "
                         "after gain, preventing clipping when boosting a "
                         "quiet track (default: -1.5)")
    ap.add_argument("--write", action="store_true",
                    help="Update Sound_volume in the project's weather_params.py")
    args = ap.parse_args()

    proj, mod = load_project(args.project)
    sounds_dir = proj / "media" / "sounds"
    presets = mod.WEATHER_PRESETS

    # Per-state record: (state, file, skip, ari, current_vol)
    state_records = []
    for state, params in presets.items():
        fname = params.get("ambient_sound")
        if not fname:
            continue
        state_records.append((
            state,
            fname,
            float(params.get("skiptime", 0.0)),
            float(params.get("ARI", 0.0)),
            float(params.get("Sound_volume", 1.0)),
        ))

    if not state_records:
        print("(no states reference an ambient_sound)")
        return

    unique_files = sorted({r[1] for r in state_records})
    print(f"Scanning {len(state_records)} states ({len(unique_files)} unique files) "
          f"in {sounds_dir}")
    print(f"Target RMS = {args.target:+.1f} dBFS  "
          f"(clamp volume to {args.min_volume:.2f}..{args.max_volume:.2f})")
    print(f"Each state's RMS is measured over its ACTUAL play window "
          f"([skiptime, skiptime+ARI]).\n")
    header = (f"  {'state':28s}  {'skip':>5s}  {'ARI':>5s}  "
              f"{'win':>5s}  {'dBFS':>6s}  {'peak':>6s}  {'gain':>6s}  {'cur':>4s}  "
              f"{'new':>4s}  file")
    print(header)
    print("  " + "-" * (len(header) + 30))

    decoded_cache = {}    # fname -> samples or None (failed/missing)
    state_new_vols = {}   # state_key -> new volume
    rows = []             # for sorting

    for state, fname, skip, ari, cur_vol in state_records:
        path = sounds_dir / fname
        if fname not in decoded_cache:
            if not path.exists():
                decoded_cache[fname] = None
                rows.append((+999, _state_key(state), skip, ari, 0.0,
                             None, None, None, cur_vol, None, f"[MISSING] {fname}"))
                continue
            samples, err = decode_mono(path)
            if samples is None:
                decoded_cache[fname] = None
                rows.append((+999, _state_key(state), skip, ari, 0.0,
                             None, None, None, cur_vol, None, f"[SKIP] {fname}: {err}"))
                continue
            decoded_cache[fname] = samples
        samples = decoded_cache[fname]
        if samples is None:
            continue
        dbfs, peak_dbfs, played, err = window_rms_dbfs(samples, skip, ari)
        if dbfs is None:
            rows.append((+999, _state_key(state), skip, ari, played,
                         None, None, None, cur_vol, None, f"[SKIP] {fname}: {err}"))
            continue
        # Gain to hit the RMS target, then cap so the post-gain true peak
        # stays at/below --max-peak (prevents clipping when boosting a
        # quiet, low-peak track). The capped value is what we report.
        gain_db = min(args.target - dbfs, args.max_peak - peak_dbfs)
        vol = 10.0 ** (gain_db / 20.0)
        vol = max(args.min_volume, min(args.max_volume, vol))
        vol = round(vol, 2)
        state_new_vols[_state_key(state)] = vol
        # Sort key: most over-target first (high effective dBFS = louder than target)
        eff = dbfs + 20.0 * np.log10(max(cur_vol, 1e-3))
        rows.append((-(eff - args.target), _state_key(state), skip, ari, played,
                     dbfs, peak_dbfs, gain_db, cur_vol, vol, fname))

    rows.sort(key=lambda r: r[0])
    for _, st, skip, ari, played, dbfs, peak_dbfs, gain_db, cur_vol, new_vol, info in rows:
        if dbfs is None:
            print(f"  {st:28s}  {skip:5.1f}  {ari:5.1f}  -----   -----  ------  -----  "
                  f"{cur_vol:4.2f}  ----  {info}")
            continue
        short = info if len(info) <= 40 else (info[:37] + "...")
        print(f"  {st:28s}  {skip:5.1f}  {ari:5.1f}  {played:5.1f}  "
              f"{dbfs:6.1f}  {peak_dbfs:6.1f}  {gain_db:+6.1f}  {cur_vol:4.2f}  {new_vol:4.2f}  {short}")

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
        key = _state_key(state)
        if key in state_new_vols:
            new_vol = state_new_vols[key]
            old_vol = float(p.get("Sound_volume", 1.0))
            if abs(old_vol - new_vol) > 1e-3:
                changes.append((key, old_vol, new_vol))
            p["Sound_volume"] = new_vol
        new_presets[key] = p

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
