"""
Utility for saving weather editor changes back to weather_params.py
Handles file writing while preserving Python syntax and formatting.
"""
import os
from pathlib import Path
import ast
import io
import tokenize
import numpy as np


def validate_weather_params(weather_states, weather_presets, weather_sets):
    """
    Validate weather parameters before saving.
    
    Returns:
        dict: {"valid": bool, "errors": list}
    """
    errors = []
    
    # Validate weather states
    if not weather_states or not isinstance(weather_states, list):
        errors.append("weather_states must be a non-empty list")
    
    for state in weather_states:
        if not isinstance(state, str) or not state:
            errors.append(f"Invalid weather state: {state}")
    
    # Validate weather presets
    if not isinstance(weather_presets, dict):
        errors.append("weather_presets must be a dictionary")
    else:
        for state, params in weather_presets.items():
            if state not in weather_states:
                errors.append(f"Preset '{state}' not in weather_states")
            if not isinstance(params, dict):
                errors.append(f"Preset '{state}' must be a dictionary")
    
    # Validate weather sets
    if not isinstance(weather_sets, dict):
        errors.append("weather_sets must be a dictionary")
    else:
        for set_id, set_data in weather_sets.items():
            if not isinstance(set_data, dict):
                errors.append(f"Set '{set_id}' must be a dictionary")
                continue
            
            if 'states' not in set_data:
                errors.append(f"Set '{set_id}' missing 'states' field")
            elif not isinstance(set_data['states'], list):
                errors.append(f"Set '{set_id}' states must be a list")
            else:
                for state in set_data['states']:
                    if state not in weather_states:
                        errors.append(f"Set '{set_id}' references unknown state '{state}'")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def _extract_assignments_text(file_path: Path, names) -> dict:
    """Return {name: source_text} for top-level `name = ...` assignments in
    the given file. Source text is the EXACT bytes of the assignment line(s),
    preserving comments, whitespace, np.array(...) literals, and ordering -
    so when the editor re-emits PARAMETER_DEFINITIONS / DEFAULT_WEATHER_PARAMS
    it doesn't strip project-specific entries it doesn't know about.

    Missing names or unreadable/unparseable files yield {}.
    """
    if not file_path.exists():
        return {}
    try:
        src = file_path.read_text(encoding='utf-8')
        tree = ast.parse(src)
    except Exception:
        return {}
    lines = src.splitlines(keepends=True)
    want = set(names)
    found = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in want):
            start = node.lineno - 1
            end = node.end_lineno or node.lineno
            found[node.targets[0].id] = ''.join(lines[start:end])
    return found


#: Top-level names the generator emits itself. Anything else a project
#: defines is carried through untouched by _extract_other_assignments.
_GENERATED_NAMES = frozenset({
    "GLOBAL_PARAMETERS", "AVAILABLE_BACKGROUND_EVENTS", "PARAMETER_DEFINITIONS",
    "DEFAULT_WEATHER_PARAMS", "OUTSTATE_PUBLISH", "WEATHER_PRESETS",
    "WEATHER_SETS", "DEFAULT_WEATHER_SET",
})


def _extract_top_level_comments(file_path: Path) -> dict:
    """Return {name: [comment_lines]} for the comment block above each
    top-level assignment, plus ``"__imports__"`` for the block above the
    first import.

    These are the module-scope notes — what OUTSTATE_PUBLISH is for, why the
    schema imports stay shared — that the generator would otherwise replace
    with its own canned one-liners. Emitting the project's block INSTEAD of
    the canned text keeps a generated file byte-identical (its existing
    comments are the canned ones) while letting a hand-written file keep its
    own words.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return {}
    try:
        src = file_path.read_text(encoding='utf-8')
        tree = ast.parse(src)
    except Exception:
        return {}
    standalone, _ = _collect_comments(src)
    if not standalone:
        return {}
    lines = src.splitlines()

    out = {}
    seen_import = False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) and not seen_import:
            seen_import = True
            block, _ = _leading_comment_block(standalone, lines, node.lineno)
            if block:
                out["__imports__"] = block
        elif (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            block, _ = _leading_comment_block(standalone, lines, node.lineno)
            if block:
                out[node.targets[0].id] = block
    return out


def _extract_other_assignments(file_path: Path) -> list:
    """Source text of every top-level assignment the generator does NOT emit.

    A project is free to define its own helpers at module scope — WoL builds
    its presets from a shared ``_BASE`` dict via ``**_BASE``. The generator
    flattens presets to literals, so the DATA survives, but the helper itself
    would disappear from the file. Carrying these through keeps a save from
    quietly deleting project code it doesn't understand.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return []
    try:
        src = file_path.read_text(encoding='utf-8')
        tree = ast.parse(src)
    except Exception:
        return []
    lines = src.splitlines()
    out = []
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id not in _GENERATED_NAMES):
            out.append((node.targets[0].id, '\n'.join(
                lines[node.lineno - 1:(node.end_lineno or node.lineno)])))
    return out


def _extract_module_header(file_path: Path) -> dict:
    """Return the existing file's module docstring and top-level imports.

    Returns ``{"docstring": str|None, "imports": [str, ...],
    "imported_names": {name, ...}}`` — all source text, verbatim.

    Two things depend on this. First, a project may satisfy the schema blocks
    by IMPORTING them rather than defining them (``weight_of_light`` does:
    ``from lib.weather_params import DEFAULT_WEATHER_PARAMS, ...``). Dropping
    that import and inlining the generator's static fallback replaced a live
    35-key table with a 25-key snapshot, deleting the params that project's
    states actually read. ``imported_names`` lets the generator recognise
    "this name already arrives by import" and emit nothing for it.

    Second, the docstring is usually where a project explains WHY its weather
    data lives locally — worth keeping.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return {"docstring": None, "imports": [], "imported_names": set()}
    try:
        src = file_path.read_text(encoding='utf-8')
        tree = ast.parse(src)
    except Exception:
        return {"docstring": None, "imports": [], "imported_names": set()}

    lines = src.splitlines()

    def text(node):
        return '\n'.join(lines[node.lineno - 1:(node.end_lineno or node.lineno)])

    docstring = None
    if (tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)):
        docstring = text(tree.body[0])

    standalone, _ = _collect_comments(src)
    imports, imported_names = [], set()
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        # Carry each import's own comment block with it — the explanation of
        # WHY a project re-exports lib's schema sits above the second import,
        # not the first, so a single "comments above the imports" grab misses it.
        block, _gap = _leading_comment_block(standalone, lines, node.lineno)
        imports.append('\n'.join(block + [text(node)]))
        for alias in node.names:
            imported_names.add(alias.asname or alias.name.split('.')[0])

    return {"docstring": docstring, "imports": imports,
            "imported_names": imported_names}


def _collect_comments(src: str):
    """Split a source file's comments into standalone vs trailing.

    Returns ``(standalone, trailing)``, both {lineno: comment_text}.
    A comment is *standalone* when it is the only thing on its line and
    *trailing* when it follows code. They're kept apart because the two
    re-emit differently: standalone comments go on their own line above
    whatever they describe, trailing ones get appended to it.

    Comments are invisible to ``ast``, so this is the only way to carry
    them through a regenerate-the-whole-file save.
    """
    lines = src.splitlines()
    standalone, trailing = {}, {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type != tokenize.COMMENT:
                continue
            lineno = tok.start[0]
            text = tok.string.rstrip()
            if lines[lineno - 1].lstrip().startswith('#'):
                standalone[lineno] = text
            else:
                trailing[lineno] = text
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return {}, {}
    return standalone, trailing


def _leading_comment_block(standalone, lines, start_line):
    """Standalone comments directly above `start_line`.

    Walks upward and stops at the first line of real code, so a block can
    only ever be claimed by one owner: an entry's banner stops at the
    previous entry's ``},`` and a first parameter's block stops at its own
    ``{``. A blank line ABOVE the comments also ends the walk (keeps
    unrelated blocks from merging); a blank line BETWEEN the comments and
    their owner is reported back so it can be reproduced.

    Blank lines BETWEEN comment blocks are kept (as empty strings), because a
    project may stack a section banner, a blank, and then a one-line label for
    the entry itself — stopping at the first blank dropped the banner. Blank
    lines directly above the owner are reported as ``gap`` instead.

    Returns ``(comment_lines, gap_before_owner)``.
    """
    collected = []
    lineno = start_line - 1
    while lineno >= 1:
        if lineno in standalone:
            collected.append(standalone[lineno])
        elif not lines[lineno - 1].strip():
            collected.append(None)          # blank
        else:
            break                           # real code — stop
        lineno -= 1
    collected.reverse()

    while collected and collected[0] is None:
        collected.pop(0)                    # blanks above the block: drop
    gap = False
    while collected and collected[-1] is None:
        collected.pop()                     # blanks under it: reproduce as gap
        gap = True

    return ['' if c is None else c for c in collected], gap


def _extract_enum_comments(file_path: Path) -> dict:
    """Return the WeatherState enum's docstring and per-member comments.

    ``{"docstring": str|None, "members": {MEMBER: {"before": [...],
    "inline": str|None}}}``. The generator rebuilds the enum from a plain
    list of state strings, so without this a project loses the notes that
    explain its own states (WoL annotates each Elements stage inline, and
    heads the group with a paragraph on how the loop runs).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return {"docstring": None, "members": {}}
    try:
        src = file_path.read_text(encoding='utf-8')
        tree = ast.parse(src)
    except Exception:
        return {"docstring": None, "members": {}}

    standalone, trailing = _collect_comments(src)
    lines = src.splitlines()

    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name == "WeatherState"):
            continue

        docstring = None
        if (node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            first = node.body[0]
            docstring = '\n'.join(
                lines[first.lineno - 1:(first.end_lineno or first.lineno)])

        members = {}
        for stmt in node.body:
            if not (isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)):
                continue
            before, _ = _leading_comment_block(standalone, lines, stmt.lineno)
            inline = trailing.get(stmt.end_lineno or stmt.lineno)
            if before or inline:
                members[stmt.targets[0].id] = {"before": before, "inline": inline}

        return {"docstring": docstring, "members": members}

    return {"docstring": None, "members": {}}


def _extract_entry_comments(file_path: Path) -> dict:
    """Harvest the layout and comments inside WEATHER_PRESETS / WEATHER_SETS.

    The generator rebuilds those two dicts from data, so without this every
    save silently deletes the "why" notes hand-written next to a state or a
    parameter (the rationale for a thunder value, the section banner above a
    story arc, the "time of day" annotations) and reformats every surviving
    line. Values round-trip fine; nothing else did.

    Returns::

        {"WEATHER_PRESETS": {entry_key: {
             "leading":     [comment lines above the entry],
             "blank_after": bool,          # blank line between them and it
             "order":       [param, ...],  # the file's own parameter order
             "groups":      [{"names":  [param, ...],   # sharing source lines
                              "before": [comment lines],
                              "inline": str|None,
                              "raw":    [source lines],
                              "values": {param: literal | _UNREADABLE}}],
             "param_group": {param: index into "groups"}}},
         "WEATHER_SETS":    {...}}

    ``entry_key`` is the enum ATTRIBUTE name for presets (``OCEAN_UPWELLING``)
    and the set-id string for sets, matching what the generator re-emits.

    ``raw`` is what makes an unedited save a no-op: a parameter whose value
    still matches ``values`` is copied through as source instead of being
    reformatted from data, so its number formatting, multi-line layout and any
    comments nested inside the value all survive. Comments attach to a
    parameter NAME, so they follow it if the order changes.

    An unreadable or unparseable file yields {} — this is a best-effort
    nicety and must never block a save.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return {}
    try:
        src = file_path.read_text(encoding='utf-8')
        tree = ast.parse(src)
    except Exception:
        return {}

    standalone, trailing = _collect_comments(src)
    lines = src.splitlines()

    def leading_block(start_line):
        return _leading_comment_block(standalone, lines, start_line)

    result = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("WEATHER_PRESETS", "WEATHER_SETS")
                and isinstance(node.value, ast.Dict)):
            continue

        entries = {}
        for key_node, val_node in zip(node.value.keys, node.value.values):
            if isinstance(key_node, ast.Attribute):
                entry_key = key_node.attr              # WeatherState.OCEAN_ABYSS
            elif (isinstance(key_node, ast.Constant)
                    and isinstance(key_node.value, str)):
                entry_key = key_node.value             # "ocean"
            else:
                continue

            leading, gap = leading_block(key_node.lineno)

            order, groups, param_group = [], [], {}
            if isinstance(val_node, ast.Dict):
                pairs = [(pk, pv)
                         for pk, pv in zip(val_node.keys, val_node.values)
                         if isinstance(pk, ast.Constant)
                         and isinstance(pk.value, str)]

                # A source line may pack several parameters (``"a": 1, "b": 2,``).
                # Such parameters can only be copied through as a UNIT — copying
                # per parameter would emit the shared line once for each of them.
                # Group any run whose spans touch the same line; most groups end
                # up with exactly one member.
                spans = []
                for pk, pv in pairs:
                    end = pv.end_lineno or pk.lineno
                    if spans and pk.lineno <= spans[-1]["end"]:
                        spans[-1]["members"].append((pk, pv))
                        spans[-1]["end"] = max(spans[-1]["end"], end)
                    else:
                        spans.append({"members": [(pk, pv)], "end": end})

                for span in spans:
                    first_key = span["members"][0][0]
                    last_val = span["members"][-1][1]
                    names = [pk.value for pk, _ in span["members"]]
                    for name in names:
                        param_group[name] = len(groups)
                        order.append(name)
                    groups.append({
                        "names": names,
                        "before": leading_block(first_key.lineno)[0],
                        "inline": (trailing.get(span["end"])
                                   if len(names) == 1 else None),
                        "raw": _raw_param_lines(lines, first_key, last_val),
                        "values": {pk.value: _literal_value(pv)
                                   for pk, pv in span["members"]},
                    })

            entries[entry_key] = {
                "leading": leading,
                "blank_after": gap,
                "order": order,
                "groups": groups,
                "param_group": param_group,
            }

        result[node.targets[0].id] = entries

    return result


#: Sentinel for "this value could not be read back from source" — a lambda,
#: a name reference, anything ast.literal_eval refuses. Such a parameter is
#: always re-emitted from data rather than copied verbatim.
_UNREADABLE = object()


def _literal_value(node):
    """Best-effort constant value of an AST value node, or _UNREADABLE.

    ``np.array([...])`` unwraps to the inner list so it compares equal to the
    plain list the editor sends back for the same parameter.
    """
    if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'array'
            and node.args):
        node = node.args[0]
    try:
        return ast.literal_eval(node)
    except Exception:
        return _UNREADABLE


def _canonical(value):
    """Comparable form of a parameter value, blind to int/float and
    list/tuple/ndarray distinctions — the round trip through JSON and numpy
    changes those without changing what the value MEANS."""
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, bool):
        return ('b', value)
    if isinstance(value, (int, float)):
        return ('n', round(float(value), 9))
    if isinstance(value, (list, tuple)):
        return ('l', tuple(_canonical(v) for v in value))
    return ('o', value)


def _raw_param_lines(lines, key_node, value_node):
    """Source lines for one ``"key": value,`` pair, re-indented to 8 spaces.

    Keeps everything the generator cannot reconstruct: the original number
    formatting, a multi-line layout, and any comments written BETWEEN the
    elements of a list (WoL annotates each of its 20-odd background_events
    that way). Re-indenting normalises the stray mis-indented lines that have
    accumulated in the hand-edited files.
    """
    start = key_node.lineno - 1
    end = value_node.end_lineno or key_node.lineno
    block = lines[start:end]
    if not block:
        return []

    # Make sure the pair still ends with a comma once it's re-emitted in the
    # middle of a dict — the source may have omitted it on a final entry.
    # ast reports col_offset in UTF-8 BYTES, not characters, so the split has
    # to happen on the encoded line; doing it on the str mis-indexes every
    # line containing an em-dash and duplicates the comma.
    raw = block[-1].encode('utf-8')
    cut = value_node.end_col_offset
    if 0 <= cut <= len(raw) and not raw[cut:].decode('utf-8', 'replace').lstrip().startswith(','):
        block[-1] = (raw[:cut] + b',' + raw[cut:]).decode('utf-8', 'replace')

    base = len(block[0]) - len(block[0].lstrip())
    out = []
    for i, line in enumerate(block):
        if i == 0:
            out.append(' ' * 8 + line.lstrip())
        elif not line.strip():
            out.append('')
        else:
            indent = len(line) - len(line.lstrip())
            out.append(' ' * (8 + max(0, indent - base)) + line.lstrip())
    return out


def _extract_set_allowed_parameters(file_path: Path) -> dict:
    """Return {set_id: [param, ...]} for each WEATHER_SETS entry's
    ``allowed_parameters`` list in the given file, read straight from the
    source via AST (no import, so project-specific modules don't need to be
    importable). Only string-literal list entries are extracted; anything
    unparseable yields {} so callers can treat a parse failure as "unknown,
    don't guard".

    Used by save_weather_params to detect a destructive save that would blank
    a set's allowed_parameters — the signature of the "control panel shows
    every set's sliders" corruption.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return {}
    try:
        src = file_path.read_text(encoding='utf-8')
        tree = ast.parse(src)
    except Exception:
        return {}
    out = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "WEATHER_SETS"
                and isinstance(node.value, ast.Dict)):
            continue
        for set_key, set_val in zip(node.value.keys, node.value.values):
            if not (isinstance(set_key, ast.Constant)
                    and isinstance(set_key.value, str)
                    and isinstance(set_val, ast.Dict)):
                continue
            set_id = set_key.value
            for k, v in zip(set_val.keys, set_val.values):
                if (isinstance(k, ast.Constant) and k.value == "allowed_parameters"
                        and isinstance(v, ast.List)):
                    params = [e.value for e in v.elts
                              if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                    out[set_id] = params
    return out


def save_weather_params(weather_states, weather_presets, weather_sets,
                        global_parameters=None, target_path=None):
    """
    Save weather parameters back to a weather_params.py file.

    Args:
        weather_states: List of weather state strings
        weather_presets: Dictionary of weather state -> parameters
        weather_sets: Dictionary of set_id -> set configuration
        global_parameters: List of parameter names that are global (optional)
        target_path: Path to write to. Defaults to lib/weather_params.py
            for backwards compatibility, but project-aware callers
            should pass the active project's weather_params.py path
            (e.g. ``projects/<id>/weather_params.py``) so per-project
            overrides actually persist — saves to the lib-level file
            get silently overridden at runtime by any project module
            that defines its own WEATHER_SETS / WEATHER_PRESETS.

    Returns:
        dict: {"success": bool, "message": str, "error": str (optional)}
    """
    try:
        # Default global parameters if not provided
        if global_parameters is None:
            global_parameters = ["possible_transitions", "transition_weights", "transition_duration", "Sound_volume"]

        # Validate first
        validation = validate_weather_params(weather_states, weather_presets, weather_sets)
        if not validation["valid"]:
            return {
                "success": False,
                "error": "Validation failed: " + "; ".join(validation["errors"])
            }

        # Resolve the target file path. ``target_path`` from the caller
        # wins; otherwise fall back to the legacy lib path.
        if target_path:
            weather_params_path = Path(target_path)
        else:
            current_dir = Path(__file__).parent.parent
            weather_params_path = current_dir / "lib" / "weather_params.py"

        # Guardrail against a destructive save that silently blanks
        # allowed_parameters. The web control panel filters its sliders to the
        # active set via each set's allowed_parameters; if a save wipes them,
        # _get_allowed_output_params() returns None and the panel falls back to
        # showing every set's parameters at once. A single set going empty can
        # be an intentional "no restrictions" edit, but MULTIPLE sets losing a
        # previously non-empty list in one save is the signature of corruption
        # (a partial/empty client payload), so refuse the write and say why.
        existing_allowed = _extract_set_allowed_parameters(weather_params_path)
        if existing_allowed:
            wiped = []
            for set_id, prev in existing_allowed.items():
                if not prev:
                    continue  # was already empty — nothing to protect
                incoming = weather_sets.get(set_id)
                if isinstance(incoming, dict) and not incoming.get("allowed_parameters"):
                    wiped.append(set_id)
            if len(wiped) >= 2:
                return {
                    "success": False,
                    "error": (
                        "Refusing to save: this would clear allowed_parameters "
                        f"for {len(wiped)} sets ({', '.join(sorted(wiped))}) that "
                        "currently have them. That blanks the control panel's "
                        "per-set slider filter (it would then show every set's "
                        "parameters at once), and a multi-set wipe like this is "
                        "almost always a bad/partial payload, not an intentional "
                        "edit. No changes were written. If you really meant to "
                        "clear them, edit weather_params.py directly."
                    ),
                }

        # Create backup
        backup_path = weather_params_path.with_suffix('.py.backup')
        if weather_params_path.exists():
            import shutil
            shutil.copy2(weather_params_path, backup_path)
        
        # Generate the Python file content. Pass the existing target path so
        # the generator can preserve project-specific PARAMETER_DEFINITIONS
        # and DEFAULT_WEATHER_PARAMS verbatim instead of stripping them.
        content = generate_weather_params_file(
            weather_states, weather_presets, weather_sets,
            global_parameters, existing_path=weather_params_path,
        )
        
        # Write to file
        with open(weather_params_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "success": True,
            "message": f"Successfully saved to {weather_params_path}. Backup created at {backup_path}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error saving file: {str(e)}"
        }


def _indent(text, spaces):
    """Indent a preserved comment line, leaving blank separators truly blank."""
    return f"{' ' * spaces}{text}" if text else ""


def _emit_default_params_fallback(lines):
    """Append the legacy static DEFAULT_WEATHER_PARAMS block.

    Only reached when the target file neither defines nor imports one — i.e.
    a fresh project being bootstrapped. Any real project hits the verbatim
    or the import path instead.
    """
    default_params = {
        "wind_speed": 0, "rain_rate": 0, "lightning_probability": 0,
        "starryness": 1.0, "spookyness": 0.0, "fog": 0.0,
        "fog_color": "np.array([0.7, 0.7, 0.7])",
        "possible_transitions": ["light_rain", "foggy", "windy_night"],
        "transition_weights": [1.0, 2.0, 0.5], "transition_duration": 20.0,
        "celestial_visibility": 1.0, "firefly_density": 0.0,
        "Aurora_probability": 0.0, "Wolfy": 0.0, "Switch_rate": 1.0,
        "meteor_rate": 0.0, "volcano_level": 0.0, "sand_density": 0.0,
        "skiptime": 0.0, "tree_prob": 0.0, "Weird": 0.0,
        "Sound_volume": 1.0, "season_preference": 0.375,
        "ambient_sound": None, "ARI": 0.0,
    }
    lines.append("DEFAULT_WEATHER_PARAMS = {")
    for key, value in default_params.items():
        if isinstance(value, str):
            lines.append(f'    "{key}": {value},')
        elif isinstance(value, list):
            lines.append(f'    "{key}": {repr(value)},')
        else:
            lines.append(f'    "{key}": {value},')
    lines.append("}")


def _emit_entry(lines, opening, params, comments):
    """Append one ``KEY: { ... },`` block, restoring its saved comments.

    `comments` is the entry's slice of _extract_entry_comments (or None).
    Parameters are still emitted in sorted order — each comment travels with
    the parameter NAME, so re-sorting moves the note along with the line it
    explains instead of stranding it above an unrelated value.
    """
    comments = comments or {}

    for text in comments.get("leading", []):
        lines.append(_indent(text, 4))
    if comments.get("leading") and comments.get("blank_after"):
        lines.append("")

    lines.append(opening)

    groups = comments.get("groups", [])
    param_group = comments.get("param_group", {})
    # Original order first, then anything new the editor added — sorted, so a
    # fresh parameter lands somewhere predictable. Keeping the file's own order
    # means an unrelated save no longer reshuffles hundreds of lines.
    order = [k for k in comments.get("order", []) if k in params]
    order += sorted(k for k in params if k not in param_group)

    done = set()
    for key in order:
        gi = param_group.get(key)
        group = groups[gi] if gi is not None else None

        if group is not None:
            if gi in done:
                continue                      # already copied with its line-mates
            unchanged = all(
                name in params
                and group["values"].get(name, _UNREADABLE) is not _UNREADABLE
                and _canonical(group["values"][name]) == _canonical(params[name])
                for name in group["names"]
            )
            if unchanged and group["raw"]:
                # Copy the source through untouched, so its formatting and any
                # comments nested inside the value survive.
                done.add(gi)
                lines.extend(_indent(t, 8) for t in group["before"])
                lines.extend(group["raw"])
                continue
            if key == group["names"][0]:
                lines.extend(_indent(t, 8) for t in group["before"])

        line = f'        "{key}": {format_python_value(key, params[key])},'
        if group is not None and group.get("inline"):
            line += f"  {group['inline']}"
        lines.append(line)

    lines.append("    },")
    lines.append("")


def generate_weather_params_file(weather_states, weather_presets, weather_sets,
                                 global_parameters=None, existing_path=None):
    """
    Generate the complete weather_params.py file content.
    
    Args:
        global_parameters: List of parameter names that should be global
    
    Returns:
        str: Python file content
    """
    if global_parameters is None:
        global_parameters = ["possible_transitions", "transition_weights", "transition_duration", "Sound_volume"]
    
    # Blocks copied through from the existing project file rather than
    # regenerated. See the note above the PARAMETER_DEFINITIONS emit for why
    # each one has to survive verbatim.
    preserved = {}
    entry_comments = {}
    enum_comments = {"docstring": None, "members": {}}
    header = {"docstring": None, "imports": [], "imported_names": set()}
    other_assignments = []
    if existing_path is not None:
        preserved = _extract_assignments_text(
            Path(existing_path),
            ('AVAILABLE_BACKGROUND_EVENTS', 'PARAMETER_DEFINITIONS',
             'DEFAULT_WEATHER_PARAMS', 'OUTSTATE_PUBLISH',
             'DEFAULT_WEATHER_SET'),
        )
        entry_comments = _extract_entry_comments(Path(existing_path))
        enum_comments = _extract_enum_comments(Path(existing_path))
        header = _extract_module_header(Path(existing_path))
        other_assignments = _extract_other_assignments(Path(existing_path))
        top_comments = _extract_top_level_comments(Path(existing_path))
    else:
        top_comments = {}
    preset_comments = entry_comments.get("WEATHER_PRESETS", {})
    set_comments = entry_comments.get("WEATHER_SETS", {})
    imported = header["imported_names"]

    def banner(name, *canned):
        """Emit the file's own comment block above `name`, or the
        generator's canned text when the file had none."""
        lines.extend(top_comments.get(name) or canned)

    # The body is built first so the header can tell whether `np` is actually
    # needed — a project whose presets carry no fog_color shouldn't gain an
    # unused numpy import just by being saved.
    lines = []

    # WeatherState enum
    lines.append("class WeatherState(Enum):")
    if enum_comments["docstring"]:
        # Extracted verbatim from source, so it already carries its own
        # indentation — re-indenting here would break a multi-line docstring
        # (only the first line would move, and the class body would then
        # disagree with itself about its indent level).
        lines.append(enum_comments["docstring"])
    members = enum_comments["members"]
    for state in weather_states:
        # Convert to UPPER_SNAKE_CASE for enum name
        enum_name = state.upper().replace('-', '_')
        note = members.get(enum_name)
        if note:
            for text in note["before"]:
                lines.append(_indent(text, 4))
        line = f'    {enum_name} = "{state}"'
        if note and note.get("inline"):
            line += f"  {note['inline']}"
        lines.append(line)
    lines.append("")

    # Global parameters
    banner("GLOBAL_PARAMETERS",
           "# Global parameters that are always available in every weather set",
           "# These cannot be removed from sets but their values can be customized per weather state")
    lines.append("GLOBAL_PARAMETERS = [")
    for param in global_parameters:
        lines.append(f'    "{param}",')
    lines.append("]")
    lines.append("")
    
    # Available background events. Skipped entirely when the project gets the
    # name from an import — re-emitting it would shadow the import with a
    # stale copy.
    if 'AVAILABLE_BACKGROUND_EVENTS' not in imported:
        banner("AVAILABLE_BACKGROUND_EVENTS",
               "# Available background events (always-active effects)",
               "# This is the single source of truth for which events can be used as continuous background effects.",
               "# Add new background-capable events here. They must also exist in Stories_OGL.py's event_map.")
        if 'AVAILABLE_BACKGROUND_EVENTS' in preserved:
            # Copy the project's own list. Regenerating it from the static list
            # below would silently overwrite any project whose set differs from
            # Fan's — a save made while another project was active used to
            # inject Fan-flavored event names into that project's file.
            lines.append(preserved['AVAILABLE_BACKGROUND_EVENTS'].rstrip('\n'))
        else:
            # Fallback for a fresh project with no list of its own yet.
            lines.append("AVAILABLE_BACKGROUND_EVENTS = [")
            background_events = [
                'clouds', 'firefly', 'stars', 'rain', 'fog', 'sandstorm', 'fog_beings', 'falling_leaves'
            ]
            for event in background_events:
                lines.append(f"    '{event}',")
            lines.append("]")
        lines.append("")

    # PARAMETER_DEFINITIONS + DEFAULT_WEATHER_PARAMS + OUTSTATE_PUBLISH:
    # preserve the project file's blocks VERBATIM (including comments,
    # ordering, np.array literals, lambdas, and any project-specific
    # params the editor doesn't know about). Without this the generator
    # drops every param not in the lib defaults - which is how all the
    # storm/cyber/etc. params kept getting stripped on save.
    # OUTSTATE_PUBLISH is the project's outstate publish table (see
    # lib/weather_state.py get_state_output) — it contains lambdas the
    # editor can never round-trip as data, so verbatim text is the ONLY
    # safe way to carry it through a save.
    # Either block is skipped when the project imports the name instead of
    # defining it — inlining a snapshot there would freeze a table that is
    # meant to track lib.
    if 'PARAMETER_DEFINITIONS' not in imported:
        banner("PARAMETER_DEFINITIONS",
               "# Parameter definitions for the weather editor",
               "# Defines the type and input configuration for each parameter")
        if 'PARAMETER_DEFINITIONS' in preserved:
            lines.append(preserved['PARAMETER_DEFINITIONS'].rstrip('\n'))
        else:
            # Fallback: emit the lib's view (only used when there's no existing
            # project file to copy from, e.g. a fresh project bootstrapped by
            # the editor).
            lines.append("PARAMETER_DEFINITIONS = {")
            from lib.weather_params import PARAMETER_DEFINITIONS as _live_defs
            for param_name, param_def in sorted(_live_defs.items()):
                lines.append(f"    '{param_name}': {repr(dict(param_def))},")
            lines.append("}")
        lines.append("")

    if 'DEFAULT_WEATHER_PARAMS' not in imported:
        banner("DEFAULT_WEATHER_PARAMS", "# Default weather parameters")
        if 'DEFAULT_WEATHER_PARAMS' in preserved:
            lines.append(preserved['DEFAULT_WEATHER_PARAMS'].rstrip('\n'))
        else:
            _emit_default_params_fallback(lines)
        lines.append("")

    # OUTSTATE_PUBLISH: verbatim only — there is no data fallback (a
    # project without one simply publishes the engine core outputs).
    if 'OUTSTATE_PUBLISH' in preserved:
        banner("OUTSTATE_PUBLISH")
        lines.append(preserved['OUTSTATE_PUBLISH'].rstrip('\n'))
        lines.append("")

    # Project-defined helpers the generator knows nothing about, kept
    # verbatim and placed ahead of the presets that may reference them.
    for name, block in other_assignments:
        banner(name)
        lines.append(block)
        lines.append("")

    # WEATHER_PRESETS
    banner("WEATHER_PRESETS", "# Weather presets", "# Weather state parameters")
    lines.append("WEATHER_PRESETS = {")
    
    for state, params in weather_presets.items():
        # Find the enum name for this state
        enum_name = state.upper().replace('-', '_')
        _emit_entry(lines, f"    WeatherState.{enum_name}: {{",
                    params, preset_comments.get(enum_name))
    
    lines.append("}")
    lines.append("")
    
    # WEATHER_SETS
    banner("WEATHER_SETS", "# Weather Sets - Mutually exclusive collections of weather states")
    lines.append("WEATHER_SETS = {")
    
    for set_id, set_data in weather_sets.items():
        _emit_entry(lines, f'    "{set_id}": {{',
                    set_data, set_comments.get(set_id))
    
    lines.append("}")
    lines.append("")
    
    # DEFAULT_WEATHER_SET is the project's BOOT set — keep whatever the file
    # already declares. Deriving it from dict order silently rebooted a
    # project into a different realm (WoL flipped wol_elements -> wol_natural)
    # just because someone saved an unrelated slider.
    if 'DEFAULT_WEATHER_SET' in preserved:
        lines.append(preserved['DEFAULT_WEATHER_SET'].rstrip('\n'))
    elif weather_sets:
        first_set = list(weather_sets.keys())[0]
        lines.append(f'DEFAULT_WEATHER_SET = "{first_set}"')
    else:
        lines.append('DEFAULT_WEATHER_SET = "full_spectrum"')
    lines.append("")

    # Preserve the import-time validation function so future missing
    # PARAMETER_DEFINITIONS entries still get surfaced loudly on startup.
    lines.append(_VALIDATION_FOOTER)

    body = '\n'.join(lines)

    # Header last: keep the project's own docstring and imports (a project may
    # satisfy the schema blocks by importing them — see _extract_module_header
    # — and rewriting the header would drop that import along with everything
    # it brings in), then top up with only what the body actually needs.
    head = []
    if header["docstring"]:
        head.append(header["docstring"])
        head.append("")
    head.extend(header["imports"])
    if 'np.array(' in body and 'np' not in imported and 'numpy' not in imported:
        head.append("import numpy as np")
    if 'Enum' not in imported:
        head.append("from enum import Enum")
    head.append("")

    return '\n'.join(head) + '\n' + body


_VALIDATION_FOOTER = '''

def _validate_parameter_definitions():
    """Sanity-check that every parameter referenced by a weather set or
    preset has a PARAMETER_DEFINITIONS entry.

    Missing entries cause the web weather editor to silently skip the
    parameter (see the `if (!paramDef) continue;` in weather_editor.html),
    so even though the parameter still affects rendering, the user can't
    see or change its value. Surfacing the problem at import time turns a
    "why can't I edit this" mystery into an obvious warning.
    """
    import sys

    known = set(PARAMETER_DEFINITIONS.keys())

    missing_in_sets = {}
    for set_name, set_data in WEATHER_SETS.items():
        for param in set_data.get("allowed_parameters", []):
            if param not in known:
                missing_in_sets.setdefault(param, []).append(set_name)

    missing_in_presets = {}
    for state, preset in WEATHER_PRESETS.items():
        for param in preset.keys():
            if param not in known:
                missing_in_presets.setdefault(param, []).append(state.value)

    if not missing_in_sets and not missing_in_presets:
        return

    bar = "=" * 72
    lines = [
        "",
        bar,
        "[weather_params] parameters missing from PARAMETER_DEFINITIONS",
        "These will be silently skipped by the web weather editor.",
        "Add an entry in PARAMETER_DEFINITIONS for each one.",
        bar,
    ]
    for param in sorted(missing_in_sets):
        sets = ", ".join(sorted(missing_in_sets[param]))
        lines.append(f"  [set]    {param}  (in allowed_parameters of: {sets})")
    for param in sorted(missing_in_presets):
        states = ", ".join(sorted(missing_in_presets[param]))
        lines.append(f"  [preset] {param}  (in states: {states})")
    lines.append(bar)
    print("\\n".join(lines), file=sys.stderr)


_validate_parameter_definitions()
'''


def format_python_value(key, value):
    """
    Format a value for Python code.
    Handles special cases like numpy arrays, lists, etc.
    """
    # Handle None
    if value is None:
        return "None"
    
    # Handle numpy-specific array for fog_color
    if key == "fog_color" and isinstance(value, (list, tuple)):
        return f"np.array({list(value)})"
    
    # Handle strings
    if isinstance(value, str):
        return f'"{value}"'
    
    # Handle lists
    if isinstance(value, list):
        # Check if it's a list of strings
        if all(isinstance(x, str) for x in value):
            formatted_items = [f'"{x}"' for x in value]
            return f"[{', '.join(formatted_items)}]"
        else:
            return repr(value)
    
    # Handle numbers
    if isinstance(value, (int, float)):
        return str(value)
    
    # Handle booleans
    if isinstance(value, bool):
        return str(value)
    
    # Default: use repr
    return repr(value)


def get_current_weather_params():
    """
    Load current weather parameters from weather_params.py
    
    Returns:
        dict: {"weather_states": list, "weather_presets": dict, "weather_sets": dict}
    """
    try:
        from lib.weather_params import (
            WeatherState, WEATHER_PRESETS, WEATHER_SETS
        )
        
        # Convert enum to list of strings
        weather_states = [state.value for state in WeatherState]
        
        # Convert WEATHER_PRESETS (with enum keys) to regular dict
        weather_presets = {}
        for state, params in WEATHER_PRESETS.items():
            state_key = state.value if hasattr(state, 'value') else str(state)
            params_copy = params.copy()
            
            # Convert numpy arrays to lists
            if 'fog_color' in params_copy:
                if hasattr(params_copy['fog_color'], 'tolist'):
                    params_copy['fog_color'] = params_copy['fog_color'].tolist()
            
            weather_presets[state_key] = params_copy
        
        return {
            "weather_states": weather_states,
            "weather_presets": weather_presets,
            "weather_sets": WEATHER_SETS
        }
        
    except Exception as e:
        raise Exception(f"Error loading current weather params: {str(e)}")


if __name__ == "__main__":
    # Test the utility
    print("Testing weather_editor_utils...")
    
    # Load current data
    data = get_current_weather_params()
    print(f"Loaded {len(data['weather_states'])} weather states")
    print(f"Loaded {len(data['weather_presets'])} weather presets")
    print(f"Loaded {len(data['weather_sets'])} weather sets")
    
    # Validate
    validation = validate_weather_params(
        data['weather_states'],
        data['weather_presets'],
        data['weather_sets']
    )
    print(f"Validation: {validation}")
    
    # Generate file content (but don't save)
    content = generate_weather_params_file(
        data['weather_states'],
        data['weather_presets'],
        data['weather_sets']
    )
    print(f"\nGenerated {len(content)} characters of Python code")
    print("\nFirst 500 characters:")
    print(content[:500])
