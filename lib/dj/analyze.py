"""The DJ analysis pipeline, Qt-free.

The 8-stage "Analyze all" flow (scan -> chroma -> [stems] -> rhythm ->
vocal curves -> enrich -> mood -> structure) used to exist ONLY inside the
planner's Library tab, so a library could not be analyzed overnight
without holding a Qt window open. The stage list, the WSL structure
handoff and a sequential subprocess runner live here now; the GUI keeps
its QProcess runner (cancellable, progress-labeled) but builds its stage
list from build_stages() so the two can never drift, and
tools/dj/dj_analyze.py is the headless CLI over run_stages().

Stage dicts: {name, args:[script, ...]} plus optional flags:
  skip       reason string - recorded and stage not run
  scanfile   stage reports via the scanner's progress JSON, not PROGRESS
  enrich     GUI runs this in-process (MusicBrainz rate-limited worker);
             headless stage lists get dj_enrich.py args instead
  structure  resolve native-vs-WSL at STAGE time (the WSL batch export
             must see the DB state after earlier stages ran)
"""
import json
import os
import shutil
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def tool(name):
    """Absolute path of a tools/dj/ CLI. The CLIs live in tools/dj/, NOT
    next to any caller - building this from a caller's __file__ is what
    silently broke every pipeline stage after the 2026-07-25 reorg."""
    return os.path.join(_REPO_ROOT, "tools", "dj", name)


# -------------------------------------------------- WSL structure handoff

def wsl_path(p):
    """C:\\foo\\bar -> /mnt/c/foo/bar (how WSL mounts Windows drives)."""
    p = os.path.abspath(p)
    return "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")


def structure_mode():
    """'native' when allin1 imports here, 'wsl' when only WSL can run it
    (NATTEN publishes no Windows wheels - see requirements-dj-structure.txt),
    else None."""
    from lib.dj import structure_ml
    if structure_ml.available():
        return "native"
    if shutil.which("wsl.exe"):
        return "wsl"
    return None


def structure_batch_paths(music_dir):
    return (os.path.join(music_dir, ".structure_batch.json"),
            os.path.join(music_dir, ".structure_results.jsonl"))


def structure_wsl_command(music_dir, db):
    """(program, args) for a SQLITE-FREE batch run through WSL, or
    (None, reason). The WSL side must never open the library DB: the
    caller holds it in WAL mode on the Windows side, and WAL's shared
    memory can't span /mnt/c (measured: 'disk I/O error'). So we export
    the todo list here, WSL appends JSONL results, and
    structure_import_results() folds them back into the DB we own.
    Env path overridable via $DJ_WSL_ALLIN1_PY."""
    rows = [r for r in db.all_tracks()
            if not r.get("missing") and not r.get("error")
            and not r.get("structure")]
    if not rows:
        return None, "structure: everything already labeled"
    tl, rp = structure_batch_paths(music_dir)
    batch = [{"id": r["id"], "title": (r.get("title") or r["path"])[:60],
              "path": wsl_path(db.abs(r["path"]))} for r in rows]
    with open(tl, "w", encoding="utf-8") as f:
        json.dump(batch, f)
    py = os.environ.get("DJ_WSL_ALLIN1_PY", "$HOME/allin1/bin/python")
    script = tool("dj_structure.py")
    cmd = (f'{py} "{wsl_path(script)}" --tracklist "{wsl_path(tl)}" '
           f'--results "{wsl_path(rp)}"')
    return ("wsl.exe", ["-e", "bash", "-lc", cmd]), None


def structure_import_results(db, music_dir):
    """Fold a WSL batch's JSONL results into the DB (which only the
    Windows side may open - see structure_wsl_command). Safe to call any
    time; imports leftovers from interrupted runs too. Returns the count
    imported."""
    tl, rp = structure_batch_paths(music_dir)
    n = 0
    if os.path.isfile(rp):
        with open(rp, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    db.set_structure(int(d["id"]), d["structure"])
                    n += 1
                except (ValueError, KeyError, TypeError):
                    continue
        os.remove(rp)
    try:
        os.remove(tl)
    except OSError:
        pass
    return n


# ------------------------------------------------------------- stage list

def build_stages(music_dir, include_stems=False, headless=False):
    """The ordered stage list of a full analysis run. Each stage skips
    already-done tracks internally, so re-running is cheap.

    NO --refine-grids here: tracks whose grid confidence stays low after
    refinement would re-queue a full re-analysis on EVERY pipeline run
    (~minutes each time). The planner's dedicated "Refine grids" button
    (or dj_scan.py --refine-grids) remains the deliberate way to chew
    that tail."""
    from lib.dj import vocals, mood_ml
    voc_ok = vocals.available()
    stages = [
        {"name": "scan", "scanfile": True,
         "args": [tool("dj_scan.py"), "--dir", music_dir]},
        {"name": "chroma",
         "args": [tool("dj_chroma.py"), "--dir", music_dir]},
    ]
    if include_stems:
        # Stems BEFORE vocal curves: with stems on disk, the vocal pass
        # derives its curve from the vocals stem - no second separation
        # of the same track.
        stages.append({"name": "stems", "skip": None if voc_ok
                       else "torch/demucs not installed",
                       "args": [tool("dj_stems.py"), "--dir", music_dir]})
    stages += [
        # Rhythm AFTER stems: with drum stems on disk the signature is
        # measured from the clean rhythm section instead of the mix.
        {"name": "rhythm",
         "args": [tool("dj_rhythm.py"), "--dir", music_dir]},
        {"name": "vocal curves", "skip": None if voc_ok
         else "torch/demucs not installed",
         "args": [tool("dj_scan.py"), "--dir", music_dir, "--revocals"]},
        ({"name": "enrich",
          "args": [tool("dj_enrich.py"), "--dir", music_dir]}
         if headless else {"name": "enrich", "enrich": True}),
        {"name": "mood", "skip": None if mood_ml.available()
         else "Music2Emotion model/torch missing",
         "args": [tool("dj_mood.py"), "--dir", music_dir]},
        # Structure resolves native-or-WSL at STAGE time.
        {"name": "structure", "structure": True,
         "skip": None if structure_mode() is not None
         else "no native allin1 and no WSL"},
    ]
    return stages


# ------------------------------------------------------ headless runner

def run_stages(music_dir, db, stages, on_line=None, on_stage=None):
    """Run the stages sequentially with subprocess, streaming output.

    on_stage(k, total, name) at each stage start; on_line(text) per output
    line. Returns {"completed": [...], "skipped": [(name, why)],
    "failed": [(name, why)]}. Exit-code semantics match the planner: 2 =
    "optional deps unavailable" (skipped), a missing script path is named
    a FAILURE before launch (a python interpreter handed a missing path
    ALSO exits 2, and that collision once let a whole pipeline "complete"
    in a second having analyzed nothing)."""
    summary = {"completed": [], "skipped": [], "failed": []}
    total = len(stages)
    for k, st in enumerate(stages, 1):
        name = st["name"]
        if on_stage:
            on_stage(k, total, name)
        if st.get("skip"):
            summary["skipped"].append((name, st["skip"]))
            continue
        program, args = sys.executable, list(st.get("args") or [])
        import_after = False
        if st.get("structure"):
            mode = structure_mode()
            if mode == "native":
                args = [tool("dj_structure.py"), "--dir", music_dir]
            elif mode == "wsl":
                structure_import_results(db, music_dir)
                cmd, why = structure_wsl_command(music_dir, db)
                if cmd is None:
                    summary["completed"].append(name)   # nothing to label
                    continue
                program, args = cmd
                import_after = True
            else:
                summary["skipped"].append((name, "became unavailable"))
                continue
        if program == sys.executable and args \
                and not os.path.exists(args[0]):
            summary["failed"].append((name,
                                      f"script missing: {args[0]}"))
            continue
        proc = subprocess.Popen(
            [program] + args, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1)
        for line in proc.stdout:
            if on_line:
                on_line(line.rstrip("\n"))
        code = proc.wait()
        if import_after:
            structure_import_results(db, music_dir)  # partial imports too
        if code == 2:            # the tools' "deps unavailable" exit
            summary["skipped"].append((name, "deps unavailable"))
        elif code != 0:
            summary["failed"].append((name, f"exit {code}"))
        else:
            summary["completed"].append(name)
    return summary
