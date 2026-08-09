"""Check that a web weather-editor save doesn't damage a project's
weather_params.py.

The editor never patches that file — ``generate_weather_params_file`` rebuilds
it wholesale from the in-memory dicts. Everything the source carried but the
data does not (comments, imports, project helpers, the boot set) therefore has
to be explicitly carried through, and each of those has been silently dropped
at some point. This drives the REAL generator against the REAL project files
and asserts a no-op save stays a no-op.

Nothing is written to the project: generated text goes to a temp file.

    python tools/tests/_weather_editor_save_test.py
"""
from __future__ import annotations

import ast
import importlib
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.weather_editor_utils import generate_weather_params_file  # noqa: E402

PROJECTS = ("fan", "weight_of_light")

failures: list[str] = []
checks = 0


def check(ok, label, detail=""):
    global checks
    checks += 1
    if ok:
        print(f"  PASS  {label}")
    else:
        failures.append(f"{label}: {detail}")
        print(f"  FAIL  {label}\n        {detail}")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload(module):
    """The editor round-trips through JSON: enum keys become strings and
    numpy arrays become lists. Reproduce that before regenerating."""
    states = [s.value for s in module.WeatherState]
    presets = {}
    for state, params in module.WEATHER_PRESETS.items():
        clean = dict(params)
        for key, value in list(clean.items()):
            if hasattr(value, "tolist"):
                clean[key] = value.tolist()
        presets[state.value] = clean
    return states, presets, dict(module.WEATHER_SETS)


def regenerate(module, target, out_path, presets=None, sets=None):
    states, live_presets, live_sets = payload(module)
    text = generate_weather_params_file(
        states,
        presets if presets is not None else live_presets,
        sets if sets is not None else live_sets,
        getattr(module, "GLOBAL_PARAMETERS", None),
        existing_path=target,
    )
    out_path.write_text(text, encoding="utf-8")
    return text


def canon(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [canon(v) for v in value]
    if isinstance(value, bool):
        return ("b", value)
    if isinstance(value, (int, float)):
        return ("n", round(float(value), 6))
    return value


def comment_counts(path):
    return Counter(
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("#")
    )


def top_level_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            names.add(node.targets[0].id)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def run_project(project, tmp):
    root = Path(__file__).resolve().parents[2]
    target = root / "projects" / project / "weather_params.py"
    if not target.exists():
        print(f"\n== {project}: not deployed, skipped ==")
        return

    print(f"\n== {project} ==")
    original = importlib.import_module(f"projects.{project}.weather_params")

    # ---- an unedited save is a no-op -----------------------------------
    out = tmp / f"{project}_noop.py"
    regenerate(original, target, out)
    saved = load(out, f"{project}_noop")

    check(
        {s.value for s in original.WeatherState} == {s.value for s in saved.WeatherState},
        "every weather state survives",
    )

    lost_names = top_level_names(target) - top_level_names(out)
    check(not lost_names, "no top-level name disappears", f"lost {sorted(lost_names)}")

    before = {s.value: p for s, p in original.WEATHER_PRESETS.items()}
    after = {s.value: p for s, p in saved.WEATHER_PRESETS.items()}
    check(set(before) == set(after), "no preset disappears",
          f"lost {sorted(set(before) - set(after))}")

    lost_keys, changed = [], []
    for state in set(before) & set(after):
        for key in before[state]:
            if key not in after[state]:
                lost_keys.append(f"{state}.{key}")
            elif canon(before[state][key]) != canon(after[state][key]):
                changed.append(f"{state}.{key}")
    check(not lost_keys, "no preset parameter disappears", f"lost {lost_keys[:6]}")
    check(not changed, "no preset value changes", f"changed {changed[:6]}")

    check(original.WEATHER_SETS.keys() == saved.WEATHER_SETS.keys(),
          "no weather set disappears")
    check(getattr(original, "DEFAULT_WEATHER_SET", None)
          == getattr(saved, "DEFAULT_WEATHER_SET", None),
          "DEFAULT_WEATHER_SET is not re-derived from dict order",
          f"{getattr(original, 'DEFAULT_WEATHER_SET', None)!r} -> "
          f"{getattr(saved, 'DEFAULT_WEATHER_SET', None)!r}")

    if hasattr(original, "DEFAULT_WEATHER_PARAMS"):
        missing = set(original.DEFAULT_WEATHER_PARAMS) - set(saved.DEFAULT_WEATHER_PARAMS)
        check(not missing, "DEFAULT_WEATHER_PARAMS keeps every key",
              f"lost {sorted(missing)}")

    if hasattr(original, "AVAILABLE_BACKGROUND_EVENTS"):
        check(list(original.AVAILABLE_BACKGROUND_EVENTS)
              == list(saved.AVAILABLE_BACKGROUND_EVENTS),
              "AVAILABLE_BACKGROUND_EVENTS is not overwritten with the static list")

    src_comments, out_comments = comment_counts(target), comment_counts(out)
    lost = sum(max(0, n - out_comments.get(text, 0)) for text, n in src_comments.items())
    dup = sum(max(0, n - src_comments.get(text, 0))
              for text, n in out_comments.items() if text in src_comments)
    check(lost == 0, "no comment is deleted", f"{lost} comment lines lost")
    check(dup == 0, "no comment is duplicated", f"{dup} comment lines duplicated")

    # ---- an edited value still lands, and its neighbours don't move -----
    states, presets, sets = payload(original)
    victim = next(s for s in presets if "Sound_volume" in presets[s])
    presets[victim] = dict(presets[victim])
    presets[victim]["Sound_volume"] = 0.4242
    presets[victim]["a_brand_new_param"] = 7

    out2 = tmp / f"{project}_edited.py"
    regenerate(original, target, out2, presets=presets, sets=sets)
    edited = load(out2, f"{project}_edited")
    edited_presets = {s.value: p for s, p in edited.WEATHER_PRESETS.items()}

    check(edited_presets[victim]["Sound_volume"] == 0.4242,
          "an edited value is written")
    check(edited_presets[victim].get("a_brand_new_param") == 7,
          "a newly added parameter is written")

    untouched_ok = all(
        canon(before[state][key]) == canon(edited_presets[state][key])
        for state in before if state != victim
        for key in before[state]
    )
    check(untouched_ok, "editing one state leaves every other state byte-equal")

    edited_comments = comment_counts(out2)
    still_lost = sum(max(0, n - edited_comments.get(t, 0))
                     for t, n in src_comments.items())
    check(still_lost == 0, "comments survive an edited save",
          f"{still_lost} lost")


def run_fresh_project(tmp):
    """A project with no existing file at all still generates valid Python."""
    print("\n== bootstrap (no existing file) ==")
    text = generate_weather_params_file(
        ["only_state"],
        {"only_state": {"fog": 0.5, "fog_color": [0.1, 0.2, 0.3],
                        "possible_transitions": ["only_state"]}},
        {"only_set": {"states": ["only_state"], "allowed_parameters": ["fog"]}},
        ["possible_transitions"],
        existing_path=tmp / "does_not_exist.py",
    )
    out = tmp / "fresh.py"
    out.write_text(text, encoding="utf-8")
    try:
        module = load(out, "fresh_wp")
        ok, detail = True, ""
    except Exception as exc:                      # pragma: no cover - failure path
        ok, detail = False, repr(exc)
    check(ok, "a bootstrapped file is importable", detail)
    if ok:
        check(module.DEFAULT_WEATHER_SET == "only_set",
              "bootstrap falls back to the first set")
        check(hasattr(module, "DEFAULT_WEATHER_PARAMS"),
              "bootstrap emits the fallback schema blocks")


def main():
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        for project in PROJECTS:
            run_project(project, tmp)
        run_fresh_project(tmp)

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED  {len(failures)}/{checks} checks")
        for line in failures:
            print("  -", line)
        return 1
    print(f"OK  {checks}/{checks} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
