"""Set Copilot: a conversational set-builder inside the planner's Set tab.

Chat with Claude about the set you're building; the copilot has TOOLS over
the same machinery the planner uses - search the library, inspect tracks,
read the compiled plan (seam styles/scores/fade-risk/anchor drift), edit
the entry list, and run the planning operations (suggest / optimize /
autofill / bridge / alternatives). Every edit lands in the visible Set tab
and is revertible; the human always has the last word.

Backend (SetCopilot) is Qt-free and takes an injected client so the gate
test drives it with a scripted fake - no network, no key. The panel
(CopilotPanel) runs each turn on a QThread and streams tool activity into
the chat view.

AUTH - two transports, tried in this order:
  1. CLI: the `claude` binary (Claude Code) via subprocess - uses your
     existing Claude Code session, NO API key (same trick as the narrative
     editor). Tool use runs through a text command protocol. Preferred when
     `claude` is on PATH, so the copilot just works with zero setup.
  2. SDK: the `anthropic` package with a resolved key/token (paste one in
     the panel, ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or `ant` oauth).
Only if neither resolves does the panel show a key field.
"""
import json
import os
import shutil
import subprocess

from lib.dj import setlist as SL
from lib.dj.brain import Brain
from lib.dj.themes import BUILTIN_THEMES, get_theme

MODEL = os.environ.get("DJ_COPILOT_MODEL", "claude-opus-4-8")
MAX_TOOL_TURNS = 24              # runaway-loop backstop per user message
HISTORY_MAX_MSGS = 40            # trim old turns; the set state is re-read

# The full transition-style menu (mirrors brain.plan_transition's beats
# table) - the vocabulary pin_style validates against.
STYLES = ("long_blend", "bass_swap", "filter_sweep", "loop_roll_exit",
          "echo_out", "cut_at_drop", "double_drop", "loop_build",
          "bassline_layer", "stem_drum_swap", "acapella_out",
          "stem_bass_swap", "drum_bridge", "acapella_in", "melody_carry",
          "long_fade")


# --------------------------------------------------------------------------
# Credentials: resolve a CONCRETE key/token from several sources.
#
# The anthropic SDK validates auth at REQUEST time, not construction, so a
# bare Anthropic() constructs fine and then throws mid-turn ("Could not
# resolve authentication method"). We instead find an explicit credential
# up front - so `available()` is honest - from, in order: a key saved
# in-app, ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or the `ant` CLI's OAuth
# token (Claude Code / `ant auth login` users get this for free).
# --------------------------------------------------------------------------

def _key_config_path():
    return os.path.join(os.path.expanduser("~"), ".gl_simple_copilot.json")


def load_saved_key():
    try:
        with open(_key_config_path(), encoding="utf-8") as f:
            return (json.load(f).get("api_key") or "").strip() or None
    except (OSError, ValueError):
        return None


def save_api_key(key):
    """Persist an Anthropic API key for the copilot (user-level, out of the
    repo). Pass '' to clear it."""
    path = _key_config_path()
    if key and key.strip():
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"api_key": key.strip()}, f)
    else:
        try:
            os.remove(path)
        except OSError:
            pass


def _ant_oauth_token():
    """A short-lived OAuth access token from the `ant` CLI, if installed and
    logged in. None when unavailable."""
    exe = shutil.which("ant")
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, "auth", "print-credentials", "--access-token"],
            capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    tok = (r.stdout or "").strip()
    if r.returncode == 0 and tok and " " not in tok and len(tok) > 20:
        return tok
    return None


def find_claude_exe():
    """Locate the `claude` CLI (Claude Code). If present, the copilot can
    run through the user's existing Claude Code session with NO API key -
    the same trick the narrative editor uses. None when not installed."""
    exe = shutil.which("claude")
    if exe:
        return exe
    import glob
    home = os.path.expanduser("~")
    cands = [
        os.path.join(home, "AppData", "Roaming", "npm", "claude.cmd"),
        os.path.join(home, "AppData", "Roaming", "npm", "claude"),
        os.path.join(home, ".local", "bin", "claude"),
        "/usr/local/bin/claude",
    ]
    cands += glob.glob(os.path.join(
        home, ".vscode", "extensions",
        "anthropic.claude-code-*", "resources", "native-binary", "claude.exe"))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def resolve_anthropic_client():
    """Return (client, auth_mode) or (None, reason). Never raises."""
    try:
        import anthropic
    except ImportError:
        return None, ("the `anthropic` package isn't installed "
                      "(pip install anthropic)")
    key = load_saved_key() or os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return anthropic.Anthropic(api_key=key), "api key"
    tok = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if tok:
        return anthropic.Anthropic(auth_token=tok), "auth token"
    tok = _ant_oauth_token()
    if tok:
        return (anthropic.Anthropic(
            auth_token=tok,
            default_headers={"anthropic-beta": "oauth-2025-04-20"}),
            "ant oauth")
    return None, ("no API key set - paste one in the panel, or set "
                  "ANTHROPIC_API_KEY / run `ant auth login`")


# --------------------------------------------------------------------------
# Tool schemas (plain JSON schema - the manual loop needs no SDK helpers)
# --------------------------------------------------------------------------

def _tool(name, description, props=None, required=None):
    return {"name": name, "description": description,
            "input_schema": {"type": "object",
                             "properties": props or {},
                             "required": required or []}}


TOOLS = [
    _tool("search_library",
          "Search the music library. All filters optional; combine them. "
          "Returns compact rows (id, title, artist, bpm, key, energy 0..1, "
          "danceability 0..1, valence 0..1 [dark<->bright/happy], arousal 0..1 "
          "[calm<->intense, ML], moods [ML mood tags], genres, year, tags). "
          "Use tags/text for genre/vibe, danceability for 'driving' vs "
          "'listening', valence for 'dark/moody' vs 'uplifting', arousal/moods "
          "(present once the Mood ML pass has run) for finer emotional steering.",
          {"text": {"type": "string",
                    "description": "substring of title or artist"},
           "tags": {"type": "array", "items": {"type": "string"},
                    "description": "require ALL of these tags (genres too)"},
           "bpm_min": {"type": "number"}, "bpm_max": {"type": "number"},
           "energy_min": {"type": "number"}, "energy_max": {"type": "number"},
           "dance_min": {"type": "number"}, "dance_max": {"type": "number"},
           "valence_min": {"type": "number"}, "valence_max": {"type": "number"},
           "limit": {"type": "integer", "description": "default 25"}}),
    _tool("get_track",
          "Full analysis of one track: sections (kind/energy/vocalness), "
          "drops, loops, grid confidence, groove offset, mix points.",
          {"track_id": {"type": "integer"}}, ["track_id"]),
    _tool("get_set",
          "The current set COMPILED: per-slot start times and play lengths, "
          "every seam's style/score/fade-risk/groove-offset/blend-floor, "
          "anchor drift, report card, warnings. Read this before and after "
          "editing."),
    _tool("add_tracks",
          "Append (or insert) tracks into the set as suggestions.",
          {"track_ids": {"type": "array", "items": {"type": "integer"}},
           "position": {"type": "integer",
                        "description": "insert index; omit to append"}},
          ["track_ids"]),
    _tool("remove_slots", "Remove entries by slot index (0-based).",
          {"positions": {"type": "array", "items": {"type": "integer"}}},
          ["positions"]),
    _tool("move_slot", "Move one entry to a new index.",
          {"from_pos": {"type": "integer"}, "to_pos": {"type": "integer"}},
          ["from_pos", "to_pos"]),
    _tool("replace_slot", "Swap the track in one slot for another track.",
          {"position": {"type": "integer"}, "track_id": {"type": "integer"}},
          ["position", "track_id"]),
    _tool("pin_anchor",
          "Mark a slot as an ANCHOR (must play, in order), optionally at a "
          "target time into the set. Unpin by omitting... no - use unpin.",
          {"position": {"type": "integer"},
           "offset_min": {"type": "number",
                          "description": "target minutes into the set"}},
          ["position"]),
    _tool("unpin", "Turn an anchor back into a suggestion.",
          {"position": {"type": "integer"}}, ["position"]),
    _tool("set_theme",
          "Switch the working theme (bpm window + energy arc + style "
          "weights). Affects every planning operation.",
          {"theme": {"type": "string"}}, ["theme"]),
    _tool("build_set",
          "THE MAIN BUILD TOOL. Generate a COHERENT set of a requested "
          "vibe by confining the brain to a filtered pool: tags (genre/mood "
          "- ALL must match), bpm range, energy range. Use this whenever "
          "the user asks for a specific kind of set (a genre, an energy, a "
          "tempo band). Replaces the current set. Returns the pool size and "
          "the resulting tempo/energy shape - report those honestly.",
          {"minutes": {"type": "number"},
           "tags": {"type": "array", "items": {"type": "string"},
                    "description": "genre/mood tags the pool must ALL have"},
           "bpm_min": {"type": "number"}, "bpm_max": {"type": "number"},
           "energy_min": {"type": "number"}, "energy_max": {"type": "number"},
           "dance_min": {"type": "number", "description": "danceability floor "
                         "0..1 (driving/floor-focused)"},
           "dance_max": {"type": "number"},
           "valence_min": {"type": "number", "description": "valence floor "
                           "0..1 (uplifting); low = dark/moody"},
           "valence_max": {"type": "number"},
           "theme": {"type": "string",
                     "description": "theme to set first (for the arc/bpm "
                     "window); optional"},
           "shape": {"type": "string",
                     "enum": ["rise", "peak", "wind_down", "flat"],
                     "description": "shape the built set's tempo curve "
                     "(rise=build, peak=up-then-down); optional"}},
          ["minutes"]),
    _tool("suggest_set",
          "Whole-library set for the current theme (NO vibe constraint - "
          "use build_set for a specific genre/energy/tempo). Confirm before "
          "discarding a hand-built set.",
          {"minutes": {"type": "number"}}, ["minutes"]),
    _tool("shape_set",
          "Reorder the current set so TEMPO or ENERGY follows a curve - the "
          "way to make a set actually BUILD (optimize_order only cleans "
          "seams, it won't create an arc). metric: tempo|energy (tempo is "
          "the stronger lever; energy is compressed on this library). "
          "shape: rise (a climb/build), peak (up then down), wind_down "
          "(high->low), flat. Report the resulting curve from get_set.",
          {"metric": {"type": "string", "enum": ["tempo", "energy"]},
           "shape": {"type": "string",
                     "enum": ["rise", "peak", "wind_down", "flat"]}},
          ["shape"]),
    _tool("optimize_order",
          "Beam-search reorder of the suggestions (anchors stay at their "
          "positions) maximizing whole-set seam quality + arc fit. Seam "
          "cleanup only - does NOT build an arc (use shape_set for that)."),
    _tool("autofill",
          "Anchor timing solver: insert brain-chosen fills sized so every "
          "timed anchor lands near its target offset. Replaces non-anchor "
          "entries between anchors."),
    _tool("bridge",
          "Find and insert the best 1-2 track chain connecting slot N to "
          "slot N+1 (the repair for a tempo/key-remote seam).",
          {"position": {"type": "integer"}}, ["position"]),
    _tool("slot_alternatives",
          "Top replacement candidates for one slot, scored by the seam "
          "into it and out of it at that moment's arc.",
          {"position": {"type": "integer"}}, ["position"]),
    _tool("pin_style",
          "Pin the transition STYLE of the seam between slot `position` "
          "and the next slot. The pin is honored by the compiled preview "
          "AND the live night - it goes through the real style gates, so "
          "a style the seam can't do (stem styles without stems on both "
          "tracks, precision styles on shaky grids, drop styles with no "
          "drop) is REFUSED with a visible 'style pin ... refused' "
          "warning and the brain's own choice plays. Check get_set after "
          "pinning. Omit `style` to clear a pin.",
          {"position": {"type": "integer"},
           "style": {"type": "string", "enum": list(STYLES)}},
          ["position"]),
    _tool("night_history",
          "What the LIVE show actually measured and played: last-played "
          "recency per track (avoid building tonight's set out of last "
          "weekend's exact records) and per-pairing seam verdicts "
          "(clean/flam/hole, from the engine's own measurements) for any "
          "pairing that has played live before. Defaults to the current "
          "set's tracks. Check this before finalizing a set.",
          {"track_ids": {"type": "array", "items": {"type": "integer"},
                         "description": "omit = the current set"}}),
    _tool("save_set",
          "Save the current set to the library DB (as `name` if given - "
          "created or overwritten; else into the currently-loaded "
          "setlist). Executes through the planner UI right after this "
          "turn's reply.",
          {"name": {"type": "string"}}),
    _tool("push_to_live",
          "Save the set and load it into the RUNNING show (its theme, "
          "length and style pins ride along). mode 'order' = play top to "
          "bottom; 'pool' = the live brain steers inside the list. This "
          "changes what plays tonight - ALWAYS confirm with the user "
          "first. Executes right after this turn's reply; the planner "
          "reports if the show isn't reachable.",
          {"mode": {"type": "string", "enum": ["order", "pool"]}}),
]

# Beatport tools are appended only when an authenticated client is present.
BEATPORT_TOOLS = [
    _tool("beatport_search",
          "Search Beatport's catalog for NEW music to buy. Returns tracks "
          "(bp_id, title, artist, bpm, key, genre, price) with a fit score "
          "vs the set's last track and how many library neighbours each "
          "has. Use when the user wants fresh tracks the library lacks.",
          {"query": {"type": "string"},
           "bpm_min": {"type": "number"}, "bpm_max": {"type": "number"},
           "limit": {"type": "integer", "description": "default 20"}},
          ["query"]),
    _tool("beatport_wishlist_add",
          "Add a Beatport track (by bp_id from beatport_search) to the buy "
          "list. Beatport has no cart API - the user opens the list in a "
          "browser to purchase.",
          {"bp_id": {"type": "integer"}}, ["bp_id"]),
]


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------

class SetCopilot:
    """Qt-free conversational set-builder. Owns a working copy of the
    entries; the UI adopts it after each turn (revertibly)."""

    def __init__(self, library, theme_name="groove", pair_memory=None,
                 client=None, beatport=None, wishlist=None,
                 night_verdicts=None, last_played=None):
        self.library = library
        self.by_id = {t.id: t for t in library}
        self.theme_name = theme_name
        self.pair_memory = dict(pair_memory or {})
        # Night evidence (injected by the panel from the planner's parsed
        # logs): {(a_title, b_title): [verdict dicts]} and
        # {track_id: last_played_epoch}. Empty dicts when logs are absent.
        self.night_verdicts = dict(night_verdicts or {})
        self.last_played = dict(last_played or {})
        self.entries = []
        self.messages = []
        # UI actions (save / push-to-live) the tool loop may not run
        # directly: the DB and the show belong to the GUI thread. Tools
        # queue them here; the panel executes them after the turn.
        self.pending_ui = []
        self._client = client
        self._mode = "sdk" if client is not None else None
        self._claude_exe = None
        self._cli_hist = ""
        self._system = None
        self.last_usage = None
        # Optional Beatport discovery: only exposed when signed in.
        self.beatport = beatport if (beatport and beatport.available()) \
            else None
        self.wishlist = wishlist

    def set_library(self, library):
        """Adopt the current library (e.g. after tracks are flagged do-not-use
        or re-tagged). Resets the cached system prompt so its energy-range
        stats recompute from the new set."""
        self.library = library
        self.by_id = {t.id: t for t in library}
        self._system = None

    def tools(self):
        return TOOLS + (BEATPORT_TOOLS if self.beatport else [])

    # -- client -----------------------------------------------------------
    def available(self):
        """True when we can run a turn. PREFERS the Claude Code CLI (zero
        setup, uses the user's session like the narrative editor); falls
        back to the anthropic SDK with a resolved key/token."""
        if self._mode is not None:
            return True
        exe = find_claude_exe()
        if exe:                          # zero-config path
            self._claude_exe = exe
            self._mode = "cli"
            return True
        self._client, self._why = resolve_anthropic_client()
        if self._client is not None:
            self._mode = "sdk"
            return True
        return False

    def why_unavailable(self):
        return getattr(self, "_why", "no Claude access")

    def auth_mode(self):
        return self._mode

    def set_api_key(self, key):
        """Save a key and rebuild the SDK client (forces SDK mode)."""
        save_api_key(key)
        self._client, self._mode = None, None
        self._client, self._why = resolve_anthropic_client()
        if self._client is not None:
            self._mode = "sdk"
        return self._mode is not None

    def client(self):
        if self._client is None and not self.available():
            raise RuntimeError(self.why_unavailable())
        return self._client

    # -- state sync with the UI --------------------------------------------
    def sync(self, entries, theme_name):
        """Adopt the UI's current truth before a turn - the human may have
        edited the set since the copilot last saw it."""
        self.entries = [dict(e) for e in entries]
        self.theme_name = theme_name

    def _compile(self):
        return SL.compile_plan(self.library, self.entries,
                               get_theme(self.theme_name),
                               pair_memory=self.pair_memory)

    # -- system prompt ------------------------------------------------------
    def system_prompt(self):
        """Stable per-session prompt (cacheable): who the copilot is, the
        library's shape, the physics of mixability. The live set state is
        NOT here - it changes every turn and comes from get_set."""
        if self._system is not None:
            return self._system
        bpms = sorted(t.bpm for t in self.library if t.bpm)
        energies = sorted(t.energy_proxy() for t in self.library)
        tag_counts = {}
        for t in self.library:
            for tag in t.all_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        top_tags = sorted(tag_counts.items(), key=lambda kv: -kv[1])[:30]
        med = bpms[len(bpms) // 2] if bpms else 0
        # Per-theme arc SHAPE + tempo window so the copilot picks the right
        # theme for a requested energy journey instead of guessing by name.
        theme_lines = []
        for name in sorted(BUILTIN_THEMES):
            th = get_theme(name)
            shape = [round(th.arc_target(i / 6.0), 2) for i in range(7)]
            theme_lines.append(
                f"  {name}: arc={th.arc} {shape}  bpm {th.bpm_range[0]:.0f}-"
                f"{th.bpm_range[1]:.0f}")
        elo, ehi = energies[0], energies[-1]
        e_med = energies[len(energies) // 2]
        self._system = f"""You are the Set Copilot inside a DJ set-planning desktop app. You help \
the user build a setlist conversationally, using tools that operate on the \
REAL planning engine - the same brain that mixes live.

LIBRARY: {len(self.library)} playable tracks; bpm {bpms[0]:.0f}..{bpms[-1]:.0f} \
(median {med:.0f}). Track "energy" spans {elo:.2f}..{ehi:.2f} (median \
{e_med:.2f}) - it is COMPRESSED on this library, so energy arcs are gentle; \
tempo is the stronger lever for a "build". Top tags: \
{", ".join(f"{t}({n})" for t, n in top_tags)}.

THEMES (each sets a bpm window + an energy ARC over the set; arc is sampled \
start->end):
{chr(10).join(theme_lines)}
  arc shapes: flat=steady, rise=mellow->peak (use for "starts low, builds"), \
peak_wave=up-then-down (peak in the middle), wind_down=high->low, \
all_night=slow multi-hour climb. Pick the theme whose arc AND bpm window \
match what the user asked - do not default to groove.

PHYSICS OF MIXABILITY (why seams fail):
- Beat-matching reaches +/-8% tempo stretch (half/double-time reads count); \
beyond that a seam becomes a deliberate FADE, never a blend.
- Seams also degrade to fades when either track's beat-grid confidence is \
low, the seam is beatless, or two sung passages would overlap.
- The compiled plan flags per seam: fade risk, blend floor (near-silent \
stretch in the overlap = dead air), groove-offset delta (>35ms flams the \
punchy styles), and cross-night pair memory (mixed well / rough before).
- Seams also carry RHYTHM terms (when tracks have rhythm signatures): \
kick_agreement (low = kick patterns contradict, bad for open-bass blends), \
swing_delta (swung vs straight grooves flam on every offbeat - nothing \
fixes it), flam_ms (hits 20-80ms apart machine-gun), mult (2.0/0.5 = \
double/half-time read), conf (low = shaky grids, treat terms as guesses), \
regions (out/in = A's exit pattern was compared against B's intro pattern, \
the material the blend actually overlaps), meter (present only on a \
3/4-vs-4/4 clash - such a seam always fades). The "chips" list is the \
human-readable summary - use those words when explaining a seam. The \
planner already steers STYLE around these (kick clash -> one-low-bed \
styles, swing clash -> stem_drum_swap or short overlaps), so a warned seam \
with a sensible style choice may still play fine.

HOW TO BUILD (this is the fix for "it just gives random songs"):
- ANY specific request - a genre, a mood, a tempo band, "deep", "peak", \
"synthwave", "warmup", "dark", "uplifting", "driving" - use build_set with \
the matching filters: tags (genre), bpm, energy, danceability (dance_min \
for driving/floor sets), valence (valence_min uplifting / valence_max for \
dark & moody). That confines the brain to a COHERENT pool so the set is \
what was asked for, not a smattering of the whole library. Genre tags come from the \
library's enrichment (all_tags) - search_library first to see what tags \
exist and how many tracks match, so you pick filters that leave a usable \
pool (aim for >= ~15 tracks for a 60-min set).
- Only use suggest_set (whole library) when the user truly wants "anything \
that flows".
- To make a set BUILD (rise to a peak, wind down, etc.) use shape_set \
(metric=tempo usually) or pass shape= to build_set. This ORDERS the set to \
follow the curve while keeping seams mixable - it's the only tool that \
actually creates an arc. optimize_order does NOT build an arc (seam cleanup \
only). Tempo shaping is cleaner than energy (energy is compressed here).
- Call get_set before advising or editing; call it again after edits.
- If a request combines a vibe AND a shape (e.g. "a synthwave set that \
builds"), set the rising theme AND pass the genre tags to build_set.
- Confirm before destructive moves (suggest_set over a hand-built set, \
clearing anchors).

NIGHT EVIDENCE (night_history): the live engine measures every seam it \
plays. Before declaring a set done, check night_history for the set: a \
pairing marked rough has AUDIBLY flammed on a real night - replace it, \
bridge it, or pin a safer style; tracks with small played_days_ago are \
what the room heard last time - lean on variety unless asked otherwise.

STYLE PINS (pin_style): each seam's style is normally a weighted dice \
roll at night. Pinning locks it - through the REAL gates, so an \
impossible pin (stem styles need "stems" on BOTH tracks - see the stems \
flag in track rows; cut_at_drop needs strong grids and a pre-drop entry) \
is refused with a warning and the fallback plays. Pin sparingly: for a \
moment the user asked for (a double_drop at the peak), or to force \
long_fade/bass_swap across a seam night_history says flams.

PERSISTENCE: save_set stores the set; push_to_live saves AND loads it \
into the running show (theme, length and pins ride along). NEVER call \
push_to_live without the user's explicit confirmation in this \
conversation - it changes what plays tonight.

REPORTING - THIS IS A HARD RULE:
- Describe the set ONLY from get_set's actual numbers (bpm, energy, seam \
styles). NEVER claim a build, arc, drop, or progression the numbers don't \
show. If bpm is flat ~90 across the set, say "tempo stays flat ~90" - do \
NOT write "rides up to a peak".
- After building, CHECK the result against the request. If the arc/tempo \
didn't materialize (common when the library's energy is compressed or a \
tempo band is thin), say so plainly and offer a fix (different theme, \
Beatport for the missing energy) rather than declaring success.
- Be concise: lead with what the set ACTUALLY looks like (real bpm span, \
real energy trend, fade share), then what you'd do next.""" + (
            "\n\nBEATPORT: you can search Beatport for NEW music the library "
            "lacks (beatport_search) and add tracks to a buy-list "
            "(beatport_wishlist_add). Beatport has NO cart API - the user "
            "buys from the browser. Suggest this when the set has a gap the "
            "library can't fill (a tempo island, a missing energy, a key "
            "with no neighbours)." if self.beatport else "")
        return self._system

    # -- tool implementations ---------------------------------------------
    def _row(self, t):
        r = {"id": t.id, "title": t.title, "artist": t.artist,
             "bpm": round(t.bpm, 1), "key": t.camelot,
             "energy": round(t.energy_proxy(), 2),
             "dur_s": round(t.duration_s), "tags": t.all_tags[:8]}
        # MusicBrainz enrichment (present only after dj_enrich has run).
        if getattr(t, "genres", None):
            r["genres"] = t.genres[:4]
        if getattr(t, "year", None):
            r["year"] = t.year
        if getattr(t, "danceability", None) is not None:
            r["danceability"] = round(t.danceability, 2)
            r["valence"] = round(t.valence, 2)
        # Music2Emo ML descriptors (present only after the mood pass has run).
        if getattr(t, "arousal", None) is not None:
            r["arousal"] = round(t.arousal, 2)
        if getattr(t, "ml_moods", None):
            r["moods"] = t.ml_moods[:6]
        if getattr(t, "has_stems", False):
            r["stems"] = True            # unlocks stem_drum_swap/acapella_out
        return r

    def _t_search_library(self, args):
        text = (args.get("text") or "").strip().lower()
        tags = {s.lower() for s in (args.get("tags") or [])}
        out = []
        for t in self.library:
            if text and text not in (t.title or "").lower() \
                    and text not in (t.artist or "").lower():
                continue
            if tags and not tags <= {s.lower() for s in t.all_tags}:
                continue
            if args.get("bpm_min") and t.bpm < args["bpm_min"]:
                continue
            if args.get("bpm_max") and t.bpm > args["bpm_max"]:
                continue
            e = t.energy_proxy()
            if args.get("energy_min") and e < args["energy_min"]:
                continue
            if args.get("energy_max") and e > args["energy_max"]:
                continue
            if not self._char_ok(t, args):
                continue
            out.append(self._row(t))
        out.sort(key=lambda r: r["bpm"])
        limit = int(args.get("limit") or 25)
        return {"count": len(out), "tracks": out[:limit],
                "truncated": len(out) > limit}

    def _t_get_track(self, args):
        t = self.by_id.get(int(args["track_id"]))
        if t is None:
            raise ValueError(f"track {args['track_id']} not in library")
        from lib.dj.features import drop_moments
        row = self._row(t)
        sig = getattr(t, "rhythm_sig", None)
        if sig:
            row["groove"] = {
                "swing": sig.get("swing"),
                "feel": ("straight" if sig.get("swing", 0.5) < 0.54
                         else "swung" if sig.get("swing", 0.5) < 0.62
                         else "shuffle"),
                "density": sig.get("density"),
                "measured_from": sig.get("source"),
            }
        row.update({
            "bpm_conf": round(t.bpm_conf, 2),
            "has_stems": bool(getattr(t, "has_stems", False)),
            "kick_offset_ms": round(t.kick_offset_s * 1000),
            "drops_at_s": [round(d) for d in drop_moments(t.sections)],
            "n_loops": len(t.loops),
            "sections": [{"kind": s["kind"], "start_s": round(s["start_s"]),
                          "end_s": round(s["end_s"]),
                          "energy": s.get("energy"),
                          "vocal": s.get("vocalness")}
                         for s in t.sections],
            "mix_ins_s": [round(p["time_s"]) for p in t.mix_ins[:6]],
            "mix_outs_s": [round(p["time_s"]) for p in t.mix_outs[:6]],
        })
        return row

    def _t_get_set(self, _args):
        if not self.entries:
            return {"theme": self.theme_name, "slots": [],
                    "note": "set is empty"}
        plan = self._compile()
        slots = []
        for i, s in enumerate(plan["slots"]):
            t = s["track"]
            row = {"slot": i, "track_id": t.id, "title": t.title,
                   "bpm": round(t.bpm, 1), "key": t.camelot,
                   "energy": round(t.energy_proxy(), 2),
                   "anchor": s["entry"].get("pin_type") == "anchor",
                   "target_min": s["entry"].get("target_offset_min"),
                   "starts_at_s": round(s["start_offset_s"]),
                   "plays_s": round(s["play_s"])}
            p, si = s["transition"], s.get("seam_info") or {}
            if p:
                row["seam_to_next"] = {
                    "style": p["style"],
                    "score": p.get("pair_score"),
                    "stretch_pct": round((p["rate"] - 1) * 100, 1),
                    **{k: si[k] for k in
                       ("fade", "floor", "d_off", "key_fit", "pair_mem")
                       if si.get(k) is not None},
                }
                rt = si.get("rhythm")
                if rt:
                    rd = {k: rt[k] for k in
                          ("score", "kick_agreement", "swing_delta",
                           "flam_ms", "mult", "conf", "regions")
                          if rt.get(k) is not None}
                    if rt.get("meter_clash"):
                        rd["meter"] = (f"{rt.get('meter_a', 4)}/4 vs "
                                       f"{rt.get('meter_b', 4)}/4")
                    row["seam_to_next"]["rhythm"] = rd
                from lib.dj.rhythm import seam_chips
                chips = seam_chips(p, si)
                if chips:
                    row["seam_to_next"]["chips"] = chips
            if s["warnings"]:
                row["warnings"] = s["warnings"]
            slots.append(row)
        seams = [s["transition"] for s in plan["slots"][:-1]
                 if s["transition"]]
        n_bm = sum(1 for p in seams if p["style"] != "long_fade")
        # Aggregate SHAPE so reporting is grounded in fact, not narrative.
        bpms = [s["track"].bpm for s in plan["slots"]]
        es = [s["track"].energy_proxy() for s in plan["slots"]]
        n3 = max(1, len(es) // 3)
        first_e = sum(es[:n3]) / n3
        last_e = sum(es[-n3:]) / n3
        first_b = sum(bpms[:n3]) / n3
        last_b = sum(bpms[-n3:]) / n3

        def trend(a, b, eps):
            return ("rises" if b - a > eps else "falls" if a - b > eps
                    else "flat")
        summary = {
            "bpm_span": [round(min(bpms)), round(max(bpms))],
            "bpm_first_third": round(first_b), "bpm_last_third": round(last_b),
            "bpm_trend": trend(first_b, last_b, 3.0),
            "energy_span": [round(min(es), 2), round(max(es), 2)],
            "energy_first_third": round(first_e, 2),
            "energy_last_third": round(last_e, 2),
            "energy_trend": trend(first_e, last_e, 0.05),
            "note": "Report these actual trends - do NOT claim a build the "
                    "trends don't show.",
        }
        return {"theme": self.theme_name,
                "total_min": round(plan["total_s"] / 60.0, 1),
                "beat_matched": f"{n_bm}/{len(seams)}" if seams else "0/0",
                "summary": summary,
                "warnings": plan["warnings"], "slots": slots}

    def _check_ids(self, ids):
        missing = [i for i in ids if i not in self.by_id]
        if missing:
            raise ValueError(f"track ids not in library: {missing}")

    def _check_pos(self, i):
        if not (0 <= i < len(self.entries)):
            raise ValueError(f"slot {i} out of range (set has "
                             f"{len(self.entries)} entries)")

    @staticmethod
    def _entry(track_id):
        return {"track_id": int(track_id), "pin_type": "suggestion",
                "target_offset_min": None, "style_override": None,
                "target_play_s": None}

    def _t_add_tracks(self, args):
        ids = [int(i) for i in args["track_ids"]]
        self._check_ids(ids)
        pos = args.get("position")
        new = [self._entry(i) for i in ids]
        if pos is None:
            self.entries.extend(new)
        else:
            pos = max(0, min(int(pos), len(self.entries)))
            self.entries[pos:pos] = new
        return {"ok": True, "set_len": len(self.entries)}

    def _t_remove_slots(self, args):
        for i in args["positions"]:
            self._check_pos(int(i))
        for i in sorted({int(i) for i in args["positions"]}, reverse=True):
            self.entries.pop(i)
        return {"ok": True, "set_len": len(self.entries)}

    def _t_move_slot(self, args):
        i, j = int(args["from_pos"]), int(args["to_pos"])
        self._check_pos(i)
        e = self.entries.pop(i)
        self.entries.insert(max(0, min(j, len(self.entries))), e)
        return {"ok": True}

    def _t_replace_slot(self, args):
        i = int(args["position"])
        self._check_pos(i)
        self._check_ids([int(args["track_id"])])
        self.entries[i]["track_id"] = int(args["track_id"])
        return {"ok": True}

    def _t_pin_anchor(self, args):
        i = int(args["position"])
        self._check_pos(i)
        self.entries[i]["pin_type"] = "anchor"
        if args.get("offset_min") is not None:
            self.entries[i]["target_offset_min"] = float(args["offset_min"])
        return {"ok": True}

    def _t_unpin(self, args):
        i = int(args["position"])
        self._check_pos(i)
        self.entries[i]["pin_type"] = "suggestion"
        self.entries[i]["target_offset_min"] = None
        return {"ok": True}

    def _t_set_theme(self, args):
        name = str(args["theme"])
        if name not in BUILTIN_THEMES:
            raise ValueError(f"unknown theme '{name}'; themes: "
                             f"{sorted(BUILTIN_THEMES)}")
        self.theme_name = name
        return {"ok": True, "theme": name}

    def _char_ok(self, t, args):
        d = getattr(t, "danceability", None)
        v = getattr(t, "valence", None)
        if d is not None:
            if args.get("dance_min") and d < args["dance_min"]:
                return False
            if args.get("dance_max") and d > args["dance_max"]:
                return False
            if args.get("valence_min") and v < args["valence_min"]:
                return False
            if args.get("valence_max") and v > args["valence_max"]:
                return False
        return True

    def _pool_from_filters(self, args):
        tags = {s.lower() for s in (args.get("tags") or [])}
        pool = []
        for t in self.library:
            if tags and not tags <= {s.lower() for s in t.all_tags}:
                continue
            if args.get("bpm_min") and t.bpm < args["bpm_min"]:
                continue
            if args.get("bpm_max") and t.bpm > args["bpm_max"]:
                continue
            e = t.energy_proxy()
            if args.get("energy_min") and e < args["energy_min"]:
                continue
            if args.get("energy_max") and e > args["energy_max"]:
                continue
            if not self._char_ok(t, args):
                continue
            pool.append(t)
        return pool

    def _t_build_set(self, args):
        if args.get("theme"):
            self._t_set_theme({"theme": args["theme"]})
        pool = self._pool_from_filters(args)
        if len(pool) < 4:
            return {"ok": False, "pool_size": len(pool),
                    "note": "too few tracks match those filters to build a "
                    "coherent set - loosen the tags/tempo/energy, or (if a "
                    "genre) the library may not be enriched. Nothing changed."}
        self.entries = SL.suggest_set(
            self.library, get_theme(self.theme_name),
            minutes=float(args["minutes"]),
            pool_ids=[t.id for t in pool])
        if args.get("shape"):
            self.entries = SL.order_by_shape(
                self.library, self.entries, get_theme(self.theme_name),
                metric="tempo", shape=args["shape"])
        state = self._t_get_set({})
        summary = state.get("summary", {})
        got_min = state.get("total_min", 0)
        want_min = float(args["minutes"])
        out = {"ok": True, "pool_size": len(pool),
               "set_len": len(self.entries), "minutes": got_min,
               "shape": {k: summary.get(k) for k in
                         ("bpm_span", "bpm_trend", "energy_trend")}}
        # Planned sets are UNIQUE (no repeats); if the pool couldn't fill
        # the time, say so instead of padding with repeats.
        if got_min < 0.8 * want_min:
            out["note"] = (f"the pool has only {len(pool)} matching tracks, "
                           f"so the longest UNIQUE (no-repeat) set is "
                           f"~{got_min:.0f} min, short of the {want_min:.0f} "
                           f"you asked. Loosen the filters, enrich the "
                           f"library, or add tracks from Beatport for a "
                           f"longer set - I won't repeat songs to pad it.")
        return out

    def _t_suggest_set(self, args):
        self.entries = SL.suggest_set(self.library,
                                      get_theme(self.theme_name),
                                      minutes=float(args["minutes"]))
        return {"ok": True, "set_len": len(self.entries)}

    def _t_optimize_order(self, _args):
        self.entries = SL.optimize_order(self.library, self.entries,
                                         get_theme(self.theme_name))
        return {"ok": True}

    def _t_shape_set(self, args):
        if not self.entries:
            return {"ok": False, "note": "set is empty - build one first"}
        metric = args.get("metric", "tempo")
        self.entries = SL.order_by_shape(
            self.library, self.entries, get_theme(self.theme_name),
            metric=metric, shape=args["shape"])
        summary = self._t_get_set({}).get("summary", {})
        return {"ok": True, "metric": metric, "shape": args["shape"],
                "result": {k: summary.get(k) for k in
                           ("bpm_span", "bpm_trend", "energy_trend")}}

    def _t_autofill(self, _args):
        self.entries = SL.autofill(self.library, self.entries,
                                   get_theme(self.theme_name))
        return {"ok": True, "set_len": len(self.entries)}

    def _t_bridge(self, args):
        i = int(args["position"])
        self._check_pos(i)
        self._check_pos(i + 1)
        a = self.by_id[self.entries[i]["track_id"]]
        b = self.by_id[self.entries[i + 1]["track_id"]]
        chain, score = SL.bridge(
            self.library, a, b, get_theme(self.theme_name),
            exclude_ids={e["track_id"] for e in self.entries})
        if not chain:
            return {"ok": False,
                    "note": "no bridge beats the direct seam"}
        for k, t in enumerate(chain):
            self.entries.insert(i + 1 + k, self._entry(t.id))
        return {"ok": True, "inserted": [self._row(t) for t in chain],
                "bottleneck": round(score, 4)}

    def _t_slot_alternatives(self, args):
        i = int(args["position"])
        self._check_pos(i)
        brain = Brain(self.library, get_theme(self.theme_name))
        brain.pair_memory = dict(self.pair_memory)
        prev = self.by_id.get(self.entries[i - 1]["track_id"]) \
            if i > 0 else None
        nxt = self.by_id.get(self.entries[i + 1]["track_id"]) \
            if i + 1 < len(self.entries) else None
        cur_ids = {e["track_id"] for e in self.entries}
        scored = []
        for t in self.library:
            if t.id in cur_ids:
                continue
            if prev is not None and brain.rate_for(prev.bpm, t)[0] is None:
                continue
            if nxt is not None and brain.rate_for(t.bpm, nxt)[0] is None:
                continue
            s_in = (brain.score(prev, t, 0.6, prev.bpm)[0]
                    if prev is not None else 1.0)
            s_out = (brain.score(t, nxt, 0.6, t.bpm)[0]
                     if nxt is not None else 1.0)
            scored.append(((max(s_in, 1e-9) * max(s_out, 1e-9)) ** 0.5, t))
        scored.sort(key=lambda p: -p[0])
        return {"alternatives": [dict(self._row(t), fit=round(v, 4))
                                 for v, t in scored[:8]]}

    def _t_pin_style(self, args):
        i = int(args["position"])
        self._check_pos(i)
        if i + 1 >= len(self.entries):
            raise ValueError(f"slot {i} is the last slot - there is no "
                             "seam after it to pin")
        style = args.get("style")
        if style is not None and style not in STYLES:
            raise ValueError(f"unknown style '{style}'; styles: "
                             f"{list(STYLES)}")
        # style_override lives on the INCOMING entry (the seam INTO it) -
        # same convention compile_plan and the live order-mode read.
        self.entries[i + 1]["style_override"] = style
        return {"ok": True,
                "note": ("pin cleared" if style is None else
                         f"seam {i}->{i + 1} pinned to {style}. Call "
                         "get_set: a gate-refused pin shows a 'style pin "
                         "... refused' warning and plays the fallback.")}

    def _t_night_history(self, args):
        import time as _time
        ids = [int(x) for x in (args.get("track_ids") or [])] \
            or [e["track_id"] for e in self.entries]
        if not ids:
            return {"note": "set is empty and no track_ids given"}
        now = _time.time()
        tracks, titles = [], set()
        for i in ids:
            t = self.by_id.get(i)
            if t is None:
                continue
            titles.add(t.title)
            ts = self.last_played.get(i)
            tracks.append({
                "track_id": i, "title": t.title,
                "played_days_ago": round((now - ts) / 86400.0, 1)
                if ts else None})
        live = []
        for (a, b), evs in self.night_verdicts.items():
            if a in titles or b in titles:
                live.append({
                    "from": a, "to": b,
                    "rough": any(e["rough"] for e in evs),
                    "nights": [{"date": e["date"], "style": e["style"],
                                "verdict": e["verdict"],
                                "max_err_beats": e["max_err_beats"]}
                               for e in evs[-3:]]})
        live.sort(key=lambda r: not r["rough"])   # measured-rough first
        return {"tracks": tracks, "live_seams": live[:40],
                "note": "'rough' = measured flam/hole by the engine's own "
                        "bar (the same evidence behind pair memory); "
                        "played_days_ago null = never played live"}

    def _t_save_set(self, args):
        name = (args.get("name") or "").strip() or None
        self.pending_ui.append(("save", {"name": name}))
        return {"ok": True,
                "note": "queued - the planner saves the set (with its "
                        "theme, compiled length and notes) right after "
                        "this reply" + (f" as '{name}'" if name else "")}

    def _t_push_to_live(self, args):
        mode = args.get("mode") or "order"
        if mode not in ("order", "pool"):
            raise ValueError("mode must be 'order' or 'pool'")
        self.pending_ui.append(("push", {"mode": mode}))
        return {"ok": True,
                "note": f"queued - saved and loaded into the running show "
                        f"({mode} mode) right after this reply; the "
                        "planner's status line reports if the show isn't "
                        "reachable"}

    def _t_beatport_search(self, args):
        if not self.beatport:
            raise ValueError("not signed in to Beatport")
        from lib.dj import beatport as BP
        filters = {}
        if args.get("bpm_min"):
            filters["bpm_low"] = int(args["bpm_min"])
        if args.get("bpm_max"):
            filters["bpm_high"] = int(args["bpm_max"])
        tracks = self.beatport.search(
            str(args["query"]),
            per_page=int(args.get("limit") or 20), **filters)
        cur = self.by_id.get(self.entries[-1]["track_id"]) \
            if self.entries else None
        out = []
        for t in tracks:
            r = BP.beatport_row(t)
            ghost = BP.ghost_trackinfo(r)
            item = {"bp_id": r["bp_id"], "title": r["title"],
                    "artist": r["artist"], "bpm": r["bpm"],
                    "key": r["camelot"], "genre": r["genre"],
                    "price": r["price"]}
            if cur is not None and r["camelot"]:
                item["fit_vs_last"] = BP.fit_vs_track(cur, ghost)
            item["library_neighbours"] = BP.fit_vs_library(
                self.library, ghost)["mixable_neighbours"] \
                if r["camelot"] else 0
            out.append(item)
        return {"count": len(out), "tracks": out,
                "note": "buy on beatport.com - no cart API"}

    def _t_beatport_wishlist_add(self, args):
        if not (self.beatport and self.wishlist):
            raise ValueError("no wishlist available")
        from lib.dj import beatport as BP
        trk = self.beatport.track(int(args["bp_id"]))
        row = BP.beatport_row(trk)
        added = self.wishlist.add(row)
        return {"ok": True, "added": added, "title": row["title"],
                "url": row["url"]}

    def run_tool(self, name, args):
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            raise ValueError(f"unknown tool {name}")
        return fn(args or {})

    # -- the turn loop -------------------------------------------------------
    def run_turn(self, user_text, on_event=None):
        """One conversational turn: user message -> tool loop -> final text.
        on_event(kind, payload) narrates progress ('tool'). Returns the
        assistant's final text. Dispatches to the CLI or SDK transport."""
        if self._mode is None:
            self.available()
        self.pending_ui = []             # fresh queue per turn
        if self._mode == "cli":
            return self._run_turn_cli(user_text, on_event)
        return self._run_turn_sdk(user_text, on_event)

    # -- CLI transport (Claude Code session; no API key) ---------------------
    def _tools_manual(self):
        """Render the tool set as a text spec for the CLI command protocol."""
        lines = []
        for t in self.tools():
            props = t["input_schema"].get("properties", {})
            req = set(t["input_schema"].get("required", []))
            args = ", ".join(
                (k + "*" if k in req else k) for k in props) or "(none)"
            lines.append(f"- {t['name']}({args}): {t['description']}")
        return "\n".join(lines)

    @staticmethod
    def _extract_action(text):
        """Pull the first JSON object with a 'tool' key from a CLI reply
        (fenced ```json block or bare {...}). Returns dict or None."""
        import re
        for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text,
                             re.DOTALL):
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict) and "tool" in obj:
                    return obj
            except ValueError:
                continue
        # bare object fallback
        depth, start = 0, None
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict) and "tool" in obj:
                            return obj
                    except ValueError:
                        pass
        return None

    @staticmethod
    def _strip_json(text):
        import re
        return re.sub(r"```(?:json)?\s*\{.*?\}\s*```", "", text,
                      flags=re.DOTALL).strip()

    def _call_claude(self, system, prompt):
        import tempfile
        sf = tempfile.NamedTemporaryFile(
            mode="w", suffix=".sys.txt", delete=False, encoding="utf-8")
        sf.write(system)
        sf.close()
        try:
            cmd = [self._claude_exe, "--no-session-persistence",
                   "--model", MODEL, "--system-prompt-file", sf.name,
                   "--output-format", "text", "-p"]
            r = subprocess.run(cmd, input=prompt, capture_output=True,
                               text=True, encoding="utf-8", errors="replace",
                               timeout=360)
            if r.returncode != 0:
                raise RuntimeError((r.stderr or r.stdout or "claude failed")
                                   .strip()[:300])
            return (r.stdout or "").strip()
        finally:
            try:
                os.unlink(sf.name)
            except OSError:
                pass

    def _run_turn_cli(self, user_text, on_event):
        def emit(k, p):
            if on_event:
                on_event(k, p)
        proto = (
            "You are driving a set-builder through a TEXT command protocol "
            "(no native tool calls here). To use a tool, reply with EXACTLY "
            "ONE fenced json block and nothing else:\n"
            '```json\n{\"tool\": \"<name>\", \"input\": {..args..}}\n```\n'
            "I run it and reply TOOL_RESULT[name]: <json>. Repeat as needed. "
            "When you are done, reply with your final answer as PLAIN TEXT "
            "and NO json block. Call get_set before advising or after edits. "
            "TOOLS:\n" + self._tools_manual())
        system = self.system_prompt() + "\n\n" + proto
        self._cli_hist += f"\n\nUSER: {user_text}"
        if len(self._cli_hist) > 16000:      # keep the recent tail
            self._cli_hist = "...\n" + self._cli_hist[-16000:]
        final = ""
        for _ in range(MAX_TOOL_TURNS):
            out = self._call_claude(system, self._cli_hist.strip())
            self._cli_hist += f"\n\nASSISTANT: {out}"
            action = self._extract_action(out)
            if not action:
                final = self._strip_json(out) or out
                return final
            name, args = action.get("tool"), action.get("input", {})
            emit("tool", {"name": name, "input": args})
            try:
                res = json.dumps(self.run_tool(name, args))[:4000]
            except Exception as e:
                res = f"ERROR {type(e).__name__}: {e}"
            self._cli_hist += f"\n\nTOOL_RESULT[{name}]: {res}"
        return final or "(stopped: too many tool calls in one turn)"

    # -- SDK transport (anthropic API; needs a key) --------------------------
    def _run_turn_sdk(self, user_text, on_event=None):
        def emit(kind, payload):
            if on_event:
                on_event(kind, payload)

        self.messages.append({"role": "user", "content": user_text})
        if len(self.messages) > HISTORY_MAX_MSGS:
            # Trim the OLDEST turns whole (a dangling tool_use breaks the
            # API); the set state is always re-read via get_set anyway.
            cut = len(self.messages) - HISTORY_MAX_MSGS
            while cut < len(self.messages) \
                    and self.messages[cut]["role"] != "user":
                cut += 1
            self.messages = self.messages[cut:]

        client = self.client()
        final_text = ""
        for _ in range(MAX_TOOL_TURNS):
            response = client.messages.create(
                model=MODEL,
                max_tokens=16000,
                thinking={"type": "adaptive"},
                output_config={"effort": "medium"},
                system=[{"type": "text", "text": self.system_prompt(),
                         "cache_control": {"type": "ephemeral"}}],
                tools=self.tools(),
                messages=self.messages,
            )
            self.last_usage = getattr(response, "usage", None)
            if response.stop_reason == "refusal":
                self.messages.append(
                    {"role": "assistant", "content": "(declined)"})
                return ("I can't help with that request - let's get back "
                        "to the set.")
            texts = [b.text for b in response.content
                     if getattr(b, "type", "") == "text"]
            if texts:
                final_text = "\n".join(texts)
            if response.stop_reason != "tool_use":
                self.messages.append({"role": "assistant",
                                      "content": response.content})
                return final_text
            # Execute every requested tool; all results in ONE user message.
            self.messages.append({"role": "assistant",
                                  "content": response.content})
            results = []
            for block in response.content:
                if getattr(block, "type", "") != "tool_use":
                    continue
                emit("tool", {"name": block.name, "input": block.input})
                try:
                    out = self.run_tool(block.name, block.input)
                    results.append({"type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": json.dumps(out)})
                except Exception as e:
                    results.append({"type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": f"{type(e).__name__}: {e}",
                                    "is_error": True})
            self.messages.append({"role": "user", "content": results})
        return final_text or "(stopped: too many tool calls in one turn)"
