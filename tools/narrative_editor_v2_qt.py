#!/usr/bin/env python3
"""
Narrative Script Editor v2 (PySide6 + NodeGraphQt)

Visual node-graph editor for building narrative audio scripts.
Uses PySide6 + NodeGraphQt for the interface, Claude Code CLI for AI-assisted generation.

v2 builds on the v1 editor (tools/narrative_editor_qt.py, kept intact) by
anchoring stories around first-class recurring concepts:
  - CODEX: characters / locations / events / themes / ideas live as entity
    cards inside script.json ("entities" section) instead of only as prose
    in the world bible. Entity slugs double as node tags — the existing
    linkage mechanism — so save files stay fully compatible with the v1
    editor and the runtime NarrativePlayer (both ignore/preserve the new
    sections).
  - Cards can be imported copy-by-value from other scripts' codices; the
    "source" field on each card leaves room for a shared external codex
    database later.

Usage:
    python tools/narrative_editor_v2_qt.py
    python tools/narrative_editor_v2_qt.py media/sounds/my_story/script.json

Requirements (beyond base project):
    pip install PySide6 NodeGraphQt qtpy
"""

import datetime as _datetime
import faulthandler
import json
import logging
import math
import os
import queue
import random
import re
import subprocess
import sys
import textwrap
import threading
import time
import traceback
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Must be set before any Qt or NodeGraphQt imports
os.environ['QT_API'] = 'pyside6'

from PySide6.QtCore import Qt, QTimer, Signal, QObject, QEvent, QRectF
from PySide6.QtGui import QColor, QPalette, QAction, QTextOption, QTextCursor, QPen, QBrush
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog,
    QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
    QFrame, QGraphicsItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QSplitter,
    QStatusBar, QTabWidget, QTextEdit, QToolTip, QVBoxLayout, QWidget,
)

from NodeGraphQt import NodeGraph, BaseNode

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT  = Path(__file__).parent.parent
SOUNDS_DIR = REPO_ROOT / "media" / "sounds"
RECENTS_PATH = REPO_ROOT / "config" / "narrative_recents.json"
RECENTS_MAX = 10

# Crash / autosave diagnostics — kept separate from v1 so both editors can
# be open on the same machine (and even the same script) without clobbering
# each other's sidecars or logs.
LOG_DIR        = REPO_ROOT / "logs" / "narrative_editor_v2"
CRASH_LOG_PATH = LOG_DIR / "crash.log"
APP_LOG_PATH   = LOG_DIR / "app.log"
AUTOSAVE_SUFFIX = ".autosave_v2.json"

NODE_PREVIEW_LEN = 60        # chars shown inside a node box
# Maximum story_context size sent to AI per call. The previous 4000-char
# limit was conservative; modern Claude models handle 50K+ tokens easily.
# Combined with the CLI's automatic prompt caching (verified via
# tools/test_claude_caching.py — ~80% of the prefix gets cached on
# subsequent calls), there is no longer a reason to keep this small.
# Keeping the constant name FOCUSED_CONTEXT_MAX as an alias for back-compat
# with any external tools that read it.
CONTEXT_MAX = 60000
FOCUSED_CONTEXT_MAX = CONTEXT_MAX   # deprecated alias


# Per-node word-count presets. Selected via dropdown in the Node Length
# dialog (script-wide) and the Arc editor (per-arc override). Each entry
# is (label, min_words, max_words). Times are approximate at ~150 wpm.
NODE_LENGTH_PRESETS = [
    # label                            min   max
    ("Vignette  (40-100 words, 15-35 sec)",   40,  100),
    ("Scene     (60-140 words, 20-48 sec)",   60,  140),
    ("Passage   (80-180 words, 28-60 sec)",   80,  180),
    ("Monologue (120-250 words, 40-85 sec)", 120,  250),
    ("Long-form (180-350 words, 60-120 sec)", 180,  350),
]


def _find_node_length_preset_index(rng):
    """Return the index of the preset matching (min, max) or -1 if no
    preset matches. Used to set the dropdown to the right item when
    loading an existing script/arc value."""
    if not rng or len(rng) != 2:
        return -1
    lo, hi = int(rng[0]), int(rng[1])
    for i, (_lbl, p_lo, p_hi) in enumerate(NODE_LENGTH_PRESETS):
        if p_lo == lo and p_hi == hi:
            return i
    return -1

PARALLEL_WORKER_COUNT = 8   # concurrent AI calls for parallel generation

# Model ids passed to the `claude` CLI (--model). Latest usable tiers,
# verified against this machine's CLI 2026-07-16. Update these when new
# model generations ship — everything else references them.
MODEL_SONNET = 'claude-sonnet-5'
MODEL_OPUS   = 'claude-opus-4-8'


def _model_short_name(model_id: str) -> str:
    """'claude-sonnet-5' -> 'Sonnet 5' — for thinking-status labels."""
    parts = (model_id or '').replace('claude-', '').split('-')
    if not parts or not parts[0]:
        return 'Claude'
    name = parts[0].capitalize()
    if len(parts) > 1:
        name += ' ' + '.'.join(parts[1:])
    return name

# ElevenLabs voice parameter safe ranges (prevents slow/distorted speech)
VOICE_STABILITY_MIN = 0.35
VOICE_STABILITY_MAX = 0.80
VOICE_STYLE_MIN     = 0.0
VOICE_STYLE_MAX     = 0.0


def _clamp_voice_settings(vs: dict) -> bool:
    """Clamp voice_settings values to safe ranges. Returns True if anything changed."""
    changed = False
    if 'stability' in vs:
        clamped = max(VOICE_STABILITY_MIN, min(VOICE_STABILITY_MAX, vs['stability']))
        if clamped != vs['stability']:
            vs['stability'] = round(clamped, 2)
            changed = True
    if 'style' in vs:
        clamped = max(VOICE_STYLE_MIN, min(VOICE_STYLE_MAX, vs['style']))
        if clamped != vs['style']:
            vs['style'] = round(clamped, 2)
            changed = True
    return changed

LAYER_ORDER = ['arrival', 'presence', 'curiosity', 'discovery', 'complication',
               'intimacy', 'turn', 'consequence', 'echo', 'stillness']

# Migration map: old 8-layer names → new 10-layer names
_LAYER_MIGRATION = {
    'intro': 'arrival', 'opening': 'presence', 'development': 'discovery',
    'deepening': 'complication', 'bridge': 'intimacy', 'turn': 'turn',
    'descent': 'consequence', 'resolution': 'stillness',
}

GENERATION_PROFILES = {
    'full': {
        # Depth ceiling in BEATS. The story structure's own length is the
        # real bound for full runs — v2 structures may exceed the classic
        # 10 layers, so this must not clip them (it used to be 10, which
        # silently stopped >10-beat structures at beat 10).
        'max_depth': 999,
        # (min, max) children per parent — but total layer is capped by layer_caps
        'widths': {
            'arrival': (2, 4), 'presence': (2, 4), 'curiosity': (2, 3),
            'discovery': (3, 5), 'complication': (2, 4), 'intimacy': (2, 3),
            'turn': (2, 3), 'consequence': (1, 2), 'echo': (1, 2),
            'stillness': (1, 2),
        },
        # Hard cap on total nodes at each layer (global, across all branches)
        # Diamond shape: 2 branches, expand through discovery, gentle taper
        'layer_caps': {
            'arrival': 4, 'presence': 6, 'curiosity': 8,
            'discovery': 12, 'complication': 10, 'intimacy': 9,
            'turn': 8, 'consequence': 7, 'echo': 5, 'stillness': 5,
        },
    },
    'continue': {
        'max_depth': 999,   # structure-bounded, same as 'full'
        'widths': {
            'arrival': (2, 4), 'presence': (2, 4), 'curiosity': (2, 3),
            'discovery': (3, 5), 'complication': (2, 4), 'intimacy': (2, 3),
            'turn': (2, 3), 'consequence': (1, 2), 'echo': (1, 2),
            'stillness': (1, 2),
        },
        'layer_caps': {
            'arrival': 4, 'presence': 6, 'curiosity': 8,
            'discovery': 12, 'complication': 10, 'intimacy': 9,
            'turn': 8, 'consequence': 7, 'echo': 5, 'stillness': 5,
        },
    },
    'expand': {
        'max_depth': 2,
        'widths': {'*': (2, 5)},
        'layer_caps': {},
    },
}

# Global width preset scales — applied as a multiplier to a base profile's
# `widths` (children-per-parent) AND `layer_caps` (total cap per layer). The
# user picks one from the Story menu; it travels with the script as
# `script.width_preset`. 'medium' is the canonical default.
WIDTH_PRESETS = {
    'small':  0.55,
    'medium': 1.0,
    'large':  1.6,
}
DEFAULT_WIDTH_PRESET = 'medium'


def _scale_profile(profile: dict, preset: str) -> dict:
    """Return a copy of `profile` with widths and layer_caps multiplied by
    the named preset's scale factor. Width tuples are floored at (1, 2)
    to avoid degenerate (0, 0)/(0, 1) ranges that would produce nothing."""
    scale = WIDTH_PRESETS.get(preset, 1.0)
    if scale == 1.0:
        return profile  # no change needed
    widths = {}
    for k, v in profile.get('widths', {}).items():
        if isinstance(v, tuple) and len(v) == 2:
            lo, hi = v
            new_lo = max(1, int(round(lo * scale)))
            new_hi = max(new_lo + 1, int(round(hi * scale)))
            widths[k] = (new_lo, new_hi)
        else:
            widths[k] = v
    caps = {k: max(1, int(round(v * scale)))
            for k, v in profile.get('layer_caps', {}).items()}
    out = dict(profile)
    out['widths'] = widths
    out['layer_caps'] = caps
    return out

# Characters that break TTS and their plain-text replacements.
_TTS_REPLACEMENTS = [
    ('\u2014', ', '),    # em dash  —
    ('\u2013', ', '),    # en dash  –
    ('\u2026', '...'),   # ellipsis …
    ('\u2018', "'"),     # left single quote  '
    ('\u2019', "'"),     # right single quote '
    ('\u201c', '"'),     # left double quote  "
    ('\u201d', '"'),     # right double quote "
    ('\u00a0', ' '),     # non-breaking space
    ('\u200b', ''),      # zero-width space
    ('\u2022', ' '),     # bullet •
    ('-', ' '),          # hyphen / dash
]

def _sanitize_tts(text: str) -> str:
    """Replace characters that confuse TTS engines with safe equivalents."""
    for bad, good in _TTS_REPLACEMENTS:
        text = text.replace(bad, good)
    # Collapse any runs of spaces left by replacements
    text = re.sub(r'  +', ' ', text)
    return text

SCRIPT_TEMPLATE = {
    "name": "New Script",
    "description": "",
    "story_context": "",
    "voice": "Rachel",
    "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.3,
        "model": "eleven_multilingual_v2",
    },
    "gap_range": [4.0, 12.0],
    "no_repeat_window": 6,
    "variables": [],
    "start_nodes": [],
    "nodes": {},
    "entities": {},
}


# ─────────────────────────────────────────────────────────────────────────────
# Codex — first-class recurring concepts (v2)
#
# The codex replaces the monolithic world bible as the primary store of
# recurring concepts: each entity is a card living in script.json under
# "entities" (self-contained — the file is still the whole story). Entity
# slugs double as node tags, which is the existing node↔entity linkage that
# both the v1 editor and the runtime NarrativePlayer already preserve/ignore.
#
# The "source" field records provenance: "local" for cards authored here,
# "import:<script>" for cards copied from another script's codex. A future
# shared codex database can plug in behind the same accessors by using
# "db:<id>" sources — no format change needed.
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_KINDS = ['character', 'location', 'event', 'theme', 'idea', 'object']

ENTITY_TEMPLATE = {
    'kind': 'character',
    'name': '',
    'aliases': [],
    'essence': '',        # 1-3 sentences; what generation prompts lead with
    'facts': [],          # durable canon facts, one string each
    'sensory': [],        # sensory anchors (smell/sound/texture), one each
    'voice_notes': '',    # how the narrator handles this entity
    'relationships': [],  # [{"to": other_slug, "nature": "..."}]
    'notes': '',          # author-only; never sent to the AI
    'source': 'local',
}

# Slugs that can never name an entity: an entity slug doubles as a node tag,
# so it must not collide with layer tags, nor with the OLD layer names that
# _migrate_layers rewrites on every load.
RESERVED_ENTITY_SLUGS = frozenset(LAYER_ORDER) | frozenset(_LAYER_MIGRATION)


def _sanitize_entity_card(card: dict) -> dict:
    """Clamp an AI-proposed entity card to ENTITY_TEMPLATE field shapes.
    Returns only the AI-authored fields — caller merges notes/source/etc."""
    kind = card.get('kind', 'idea')
    if kind not in ENTITY_KINDS:
        kind = 'idea'
    rels = []
    for r in card.get('relationships', []) or []:
        if isinstance(r, dict) and r.get('to'):
            rels.append({'to': ScriptData.sanitize_entity_slug(str(r['to'])),
                         'nature': str(r.get('nature', ''))[:200]})
    return {
        'kind': kind,
        'name': str(card.get('name', '') or '')[:80],
        'aliases': [str(a)[:60] for a in (card.get('aliases') or [])][:8],
        'essence': str(card.get('essence', ''))[:1200],
        'facts': [str(f)[:300] for f in (card.get('facts') or [])][:12],
        'sensory': [str(s)[:200] for s in (card.get('sensory') or [])][:8],
        'voice_notes': str(card.get('voice_notes', ''))[:600],
        'relationships': rels[:10],
    }

# ─────────────────────────────────────────────────────────────────────────────
# The Web — story-to-story relations (v2 Phase 3)
#
# Relations are stored on the FROM arc under 'relations':
#   [{"to": arc_id, "type": <RELATION_TYPES key>, "note": str}, ...]
# v1 ignores-and-preserves the field (its arc saves use dict.update()).
# The runtime needs no changes: junctions become ordinary low-weight edges,
# callbacks live inside node text, and everything else is planning metadata.
#
# Canon events live top-level under 'canon_events':
#   [{"id": "ev_001", "summary": str, "entities": [slug], "established_by": arc_id}]
# They are the cross-pollination ledger: facts a generated story established
# that other stories can reference (callback) or retell (same_event).
# ─────────────────────────────────────────────────────────────────────────────

RELATION_TYPES = {
    'shares_character': 'the same codex entities appear in both stories',
    'same_event':       'both stories touch one canon event from different angles',
    'callback':         "FROM story's text references TO story's canon events",
    'sequel':           'FROM story takes place after TO story',
    'junction':         'runtime drift: low-weight edges let a walk slide FROM → TO',
}

RELATION_COLORS = {
    'shares_character': '#4a90d9',
    'same_event':       '#9b59b6',
    'callback':         '#e67e22',
    'sequel':           '#2ecc71',
    'junction':         '#e74c3c',
}

JUNCTION_EDGE_WEIGHT = 0.5   # drift edges are deliberately rare

# A junction relation is considered fully woven once this many node pairs
# connect the two stories (directly or via bridges) — further weave presses
# skip it, keeping repeated weaving idempotent and cost bounded.
MAX_JUNCTION_LINKS_PER_PAIR = 3


# ─────────────────────────────────────────────────────────────────────────────
# Storytelling discipline — story-construction theory injected into every
# node-generation system prompt. The goal: each node performs a real
# TRANSFORMATION of the protagonist's state, not just another atmospheric
# paragraph. To disable, set STORYTELLING_DISCIPLINE = "" and remove the
# LAYER FUNCTION blocks in generate_single_node_sync / generate_batch_sync.
# ─────────────────────────────────────────────────────────────────────────────

STORYTELLING_DISCIPLINE = """

STORYTELLING DISCIPLINE — read carefully. This overrides any temptation to
write atmospheric prose without story-work.

Every node performs ONE TRANSFORMATION of the protagonist's state. Before
writing, declare to yourself in your head:
  - What does the protagonist KNOW, where are they POSITIONED, and what
    RESOURCE do they have at the START of this node?
  - Which of {knowledge, position, resource} CHANGES by the end?

If none of those three changes, the node is fluff. Reject and try again.

A node should operate AT LEAST ONE engine — ideally two or three:
  CAUSAL    — "because the previous node did X, this node does Y"
  EMOTIONAL — "the protagonist now feels something they did not before"
  EPISTEMIC — "the protagonist now knows something they did not before"

If you cannot finish at least one of those three sentences for this node,
the node is fluff.

FORBIDDEN:
  - "You remember..." / "You feel..." without naming SPECIFICALLY what
    changed in what is remembered or felt.
  - Introducing a new evocative image that does no transformation work.
  - The arc's recurring motif appearing UNCHANGED from its previous
    appearance — it must mutate, advance, or be deliberately absent.
  - The same KIND of metaphor as the previous node (don't do "X is a Y"
    twice in a row; vary the figure).
  - Any sentence that could appear, unchanged, in a different beat of
    this arc.
  - Resolving a tension that belongs to a later beat (don't show the
    discovery in the curiosity beat; don't decide before the turn beat;
    don't deliver interior consequence before echo).
  - Reaction-shot interiority before the action has actually happened.

Sensory anchors from the world bible (tunnel iron, neon hum, condensation,
pale-green status light, bioluminescence, etc.) are INSTRUMENTS of the
transformation, not decoration. If a sensory detail is doing no work in
the transformation, cut it.
"""

# Per-layer function descriptions. Injected into the per-call prompt
# alongside (or instead of) the per-arc LAYER DIRECTION. Tells the AI
# what story-machine work this beat is supposed to do.
LAYER_FUNCTIONS = {
    'arrival': (
        "Establish the protagonist's blindspot by showing them in an "
        "equilibrium we already know is doomed. They are comfortable doing "
        "something they will not be doing by the end of the arc. Show them "
        "in action within their equilibrium — not just scene-setting."
    ),
    'presence': (
        "Establish the COST of the equilibrium. The protagonist's routine — "
        "but plant ONE small detail that, in retrospect, will be the first "
        "crack. Do not add new world; add internal pressure."
    ),
    'curiosity': (
        "The crack OPENS — the protagonist notices something specific they "
        "had been failing to notice. They do NOT yet understand what it "
        "means. Do not spell out the conclusion — that belongs to discovery."
    ),
    'discovery': (
        "The crack REVEALS what's behind it — the protagonist sees the shape "
        "of what they had been missing. Frame as recognition, not analysis. "
        "Discovery is felt before it is understood."
    ),
    'complication': (
        "The COST of knowing. The discovery creates an obligation, threat, "
        "debt, or trap. The story becomes IRREVERSIBLE at this beat. Show "
        "the bind specifically — every available path costs them. Do not "
        "have them decide yet — that's the turn."
    ),
    'intimacy': (
        "The protagonist comes into DIRECT CONTACT with the thing they "
        "care about most. The world NARROWS to a single point of contact. "
        "Not romantic, not confessional — they are now standing in front "
        "of whatever they had been moving toward. The story exhales."
    ),
    'turn': (
        "The CHOICE that IS the story. An irreversible action that NAMES "
        "the protagonist. They could not have done this before this beat; "
        "they cannot undo it after. Show the action, not the deliberation."
    ),
    'consequence': (
        "The world RECEIVES the choice. The system / corporation / city / "
        "institution responds. Show the RIPPLE, not the protagonist's "
        "reaction to it. They are now a smaller figure in a larger machine "
        "that is moving."
    ),
    'echo': (
        "The INTERIOR consequence — what the protagonist now carries that "
        "they did not before. Not regret, not pride — a DIFFERENT TEXTURE "
        "OF SELF. Felt, not narrated. Not a summary of what they did."
    ),
    'stillness': (
        "The world WITHOUT the protagonist's attention. The motif finishes "
        "on its own terms — the condensation line completes its path, the "
        "clock starts over, the room is still there. Withdraw the protagonist. "
        "Close by widening. Small."
    ),
}


# Per-layer premise role and weight. The premise stays visible across ALL
# layers — the weight modulates emphasis, not presence. The U-curve peaks at
# arrival (establish), complication (the bind IS the premise), and turn (the
# action IS the premise enacted); it dips slightly in the middle beats where
# moment-to-moment causation drives the prose, and tapers at stillness where
# the premise is afterimage.
#
# This replaces the original linear-fade `1.0 - 0.3 * layer_idx` formula,
# which dropped the premise to 0% by layer 4 (complication) — meaning the
# turn beat, the most important moment in any arc, was being generated
# without the premise present in the prompt at all.
LAYER_PREMISE_ROLES = {
    'arrival':     (1.00, "establish this premise's world — the equilibrium you show exists FOR this premise"),
    'presence':    (0.85, "extend the equilibrium implied by this premise; plant the small detail this premise will widen"),
    'curiosity':   (0.75, "the noticing arises because of this premise — let it pull the protagonist's attention"),
    'discovery':   (0.80, "this beat crystallizes the premise's shape — the protagonist begins to see what they have been missing"),
    'complication':(0.90, "the bind IS this premise made specific — name the cost the premise demands"),
    'intimacy':    (0.80, "the protagonist now stands in front of the heart of this premise — direct contact"),
    'turn':        (0.95, "the action you write here MUST BE this premise enacted — the choice the premise was always going to require"),
    'consequence': (0.80, "the world's response to this premise being acted on — the ripple, not the reaction"),
    'echo':        (0.70, "what the protagonist now carries from having acted on this premise — interior, felt, not narrated"),
    'stillness':   (0.55, "this premise's afterimage — the protagonist is gone but its residue remains"),
}


SYSTEM_GENERATE = """\
You are a narrative script writer for an immersive audio installation.
Scripts play as atmospheric spoken audio layered over weather and lighting effects.

Each node is one short spoken segment (40–100 words, ~15–35 seconds when read aloud).
Use evocative, atmospheric language suited to the theme.

RUNTIME BEHAVIOUR:
- The player randomly picks one entry from "next" (weighted) when a node finishes.
- Branches are NOT audience choices — they create organic variation so each playthrough differs.
- A no_repeat_window prevents the same node from playing twice in quick succession.

OUTPUT FORMAT — respond with ONLY this JSON, no markdown fences, no explanation:
{
  "name": "Script name",
  "description": "One-line description",
  "start_nodes": ["arrival_a"],
  "nodes": {
    "node_id": {
      "text": "Spoken text, 40-100 words.",
      "next": ["next_id"],
      "weights": [1.0],
      "tags": ["arrival"],
      "voice_settings": {"stability": 0.65, "similarity_boost": 0.75, "style": 0.1}
    }
  }
}

DESIGN PRINCIPLES:
Think in LAYERS. Each layer is a narrative beat. Within a layer, multiple alternative nodes
cover the same beat differently. Edges go forward through layers — no random cross-links.

CONTINUITY IS THE TOP PRIORITY:
Every path through the graph must read as a single, flowing spoken piece — not a collection
of independent vignettes. A listener hears one node at a time, in sequence. Each node must
feel like the natural next sentence or thought after whichever node preceded it.

To achieve this:
- Write nodes so the FIRST WORDS pick up the thread from where any parent could have left off.
  Avoid re-establishing the scene or re-introducing the subject — the listener is already there.
- A node with multiple parents must be written so it can follow any of them without jarring.
  Favour abstract, open continuations ("And yet—", "Still.", "Somewhere nearby...") over
  context-specific hooks that only make sense after one particular parent.
- Siblings at the same layer should cover the SAME narrative moment from different angles,
  so any of them can flow into the same child node without contradiction.
- Read every path aloud as you write. If a transition sounds like a non-sequitur, rewrite
  the child node until the seam disappears.

LAYER STRUCTURE (use as many layers as the content warrants, up to 10):
  Layer 1  (arrival)     : 1–3 nodes  — pure sensory immersion
  Layer 2  (presence)    : 2–4 nodes  — someone or something is here
  Layer 3  (curiosity)   : 2–3 nodes  — something doesn't fit, or beckons
  Layer 4  (discovery)   : 3–6 nodes  — the thing is encountered directly
  Layer 5  (complication): 2–5 nodes  — it's not what it seemed
  Layer 6  (intimacy)    : 2–3 nodes  — personal stakes, vulnerability
  Layer 7  (turn)        : 2–4 nodes  — the emotional pivot
  Layer 8  (consequence) : 1–3 nodes  — the weight of the turn
  Layer 9  (echo)        : 1–2 nodes  — reverberations, connections to something larger
  Layer 10 (stillness)   : 1–3 nodes  — rest, not closure

For shorter scripts, skip layers or collapse them — a 4-layer script is fine.
For longer scripts, use all 10 to create a full arc with genuine depth.

HARD LIMIT: generate no more than 12 nodes total. If the full arc needs more, compress layers,
reduce siblings per layer, or end earlier — but never exceed 12 nodes.

BRANCHING: any node in layer N may connect to 2–4 nodes in layer N+1.
MERGING:   multiple nodes in layer N may all point to the same node in layer N+1.
           This creates convergence points — moments every path passes through.

Good pattern:
  arrival → [pres_a, pres_b]             ← branch early
  pres_a, pres_b → [disc_a, disc_b, disc_c]
  disc_a, disc_b → [turn_x]              ← merge
  disc_c         → [turn_y]
  turn_x, turn_y → [still_a, still_b]    ← merge then branch again

Avoid:
  - Fully connected pools where every node points to every other node
  - Trees that only branch and never merge (too many dead-end leaves)
  - Chains with no branching at all (boring, no variation)
  - Nodes that re-establish context already given by their parents

WEIGHTS: use 1.0 as default. Use 2.0 to favour a path, 0.5 to make it rare.
TAGS: every node must have a tags array with:
1. Exactly one layer tag: arrival / presence / curiosity / discovery / complication / intimacy / turn / consequence / echo / stillness
2. Custom tags for everything present in the node text — characters, themes, locations, objects, moods.
   Use short lowercase snake_case words. Reuse the same tag across nodes whenever the same element recurs.
   Examples: "crow", "test_anxiety", "linoleum", "waiting", "silence", "rain"
node IDs: short_snake_case, layer-prefixed (e.g. "arrival_storm", "disc_pride", "turn_silence", "still_rest")

VOICE SETTINGS: set "voice_settings" on every node to match its emotional tone:
  stability      0.0–1.0  lower = more expressive/varied delivery
  similarity_boost         leave at 0.75 unless noted
  style          0.0–1.0  higher = more dramatic/theatrical

  Layer defaults:
    arrival     stability 0.65  style 0.10  (calm, orienting)
    presence    stability 0.60  style 0.15  (settling in, atmospheric)
    curiosity   stability 0.55  style 0.25  (drawn in, questioning)
    discovery   stability 0.50  style 0.35  (engaged, exploring)
    complication stability 0.45  style 0.45  (more invested, richer)
    intimacy    stability 0.42  style 0.48  (personal, vulnerable)
    turn        stability 0.38  style 0.55  (tense, expressive)
    consequence stability 0.35  style 0.55  (intense, committed)
    echo        stability 0.40  style 0.50  (resonant, reflective)
    stillness   stability 0.60  style 0.15  (settled, at rest)
  Adjust within layer if the content is notably more or less intense than usual.

TERMINAL NODES: stillness/ending nodes must have next: [].
NEVER create edges that point back toward arrival or start nodes.
When a terminal node finishes playing, the runtime will automatically restart
from a randomly chosen start_node — no explicit loop edges are needed or wanted.
"""

SYSTEM_GENERATE_SEED = """\
You are a narrative script writer for an immersive audio installation.
Scripts play as atmospheric spoken audio layered over weather and lighting effects.

Each node is one short spoken segment (40–100 words, ~15–35 seconds when read aloud).
Use evocative, atmospheric language suited to the theme.

Generate ONLY the arrival layer — exactly 4 opening nodes that establish pure sensory immersion.
Each node should set up a DISTINCT story branch — different perspectives, locations, or characters.
These are the first words the audience will hear. Leave "next" as [] for ALL nodes —
subsequent layers will be generated separately in a follow-up step.

OUTPUT FORMAT — respond with ONLY this JSON, no markdown fences, no explanation:
{
  "name": "Script name",
  "description": "One-line description",
  "start_nodes": ["arrival_a"],
  "nodes": {
    "arrival_a": {
      "text": "Spoken text, 40-100 words.",
      "next": [],
      "weights": [],
      "tags": ["arrival"],
      "voice_settings": {"stability": 0.65, "similarity_boost": 0.75, "style": 0.10}
    }
  }
}

Node IDs: short_snake_case, arrival-prefixed (e.g. "arrival_storm", "arrival_silence").
TAGS: "arrival" plus custom content tags — characters, themes, locations, moods.
VOICE SETTINGS: stability 0.65, similarity_boost 0.75, style 0.10 (calm, orienting).
"""

SYSTEM_GENERATE_LAYER = """\
You are writing one layer of a narrative graph for an immersive audio installation.
You receive multiple SOURCE NODES — the current frontier — and must generate the next layer.

Each node is a short spoken segment (40–100 words, ~15–35 seconds when read aloud).

OUTPUT FORMAT — respond with ONLY this JSON, no markdown fences, no explanation:
{
  "nodes": {
    "node_id": {
      "text": "Spoken text, 40-100 words.",
      "connect_from": ["source_id_a"],
      "next": [],
      "weights": [],
      "tags": ["discovery"],
      "voice_settings": {"stability": 0.50, "similarity_boost": 0.75, "style": 0.35}
    }
  }
}

"connect_from": which existing SOURCE NODE IDs lead into this new node.
  - One connect_from = this node continues from that one source
  - Multiple connect_from = convergence — multiple sources all lead here
  - Every source node must appear in at least one connect_from
  - Generate 2–5 total new nodes across the whole layer

BRANCHING AND MERGING: vary the structure. Do not produce a 1:1 mapping of source→child.
  source_a → [new_x, new_y]     (branch)
  source_b → [new_y, new_z]     (source_b shares new_y with source_a — merge)

LAYER PROGRESSION:
  arrival → presence → curiosity → discovery → complication → intimacy → turn → consequence → echo → stillness
  Determine the source nodes' layer from their tags. All new nodes go in the NEXT layer.
  If source nodes are in "echo", generate stillness nodes with next: [].
  If source nodes are in "turn", generate consequence nodes.

CONTINUITY: every new node must follow naturally from its source(s). The first words pick
  up the thread of whichever source led there. Shared convergence nodes must work after
  any of their sources without jarring.

THEMATIC CONTINUITY: stay in the same world, imagery, and atmosphere as the source nodes.

node IDs: short_snake_case, layer-prefixed (e.g. "disc_kelp", "turn_silence", "still_rest")
TAGS: one layer tag + custom content tags (carry forward tags from source nodes where applicable)
VOICE SETTINGS per layer:
  arrival     stab~0.65 style~0.10 | presence    stab~0.60 style~0.15
  curiosity   stab~0.55 style~0.25 | discovery   stab~0.50 style~0.35
  complicat   stab~0.45 style~0.45 | intimacy    stab~0.42 style~0.48
  turn        stab~0.38 style~0.55 | consequence stab~0.35 style~0.55
  echo        stab~0.40 style~0.50 | stillness   stab~0.60 style~0.15
"""

SYSTEM_CONTINUE = """\
You are continuing a narrative graph for an immersive audio installation.
You receive an existing SOURCE NODE and must generate the remaining story layers that come AFTER it.

Each node is one short spoken segment (40–100 words, ~15–35 seconds when read aloud).
Use evocative, atmospheric language suited to the theme.

OUTPUT FORMAT — respond with ONLY this JSON, no markdown fences, no explanation:
{
  "nodes": {
    "node_id": {
      "text": "Spoken text, 40-100 words.",
      "next": ["next_id"],
      "weights": [1.0],
      "tags": ["turn"],
      "voice_settings": {"stability": 0.38, "similarity_boost": 0.75, "style": 0.55}
    }
  },
  "start_nodes": ["first_node_id"]
}

CONTINUITY IS THE TOP PRIORITY:
Every path through the generated nodes must read as a single flowing spoken piece that begins
directly where the source node left off. A listener hears one node at a time, in sequence.

- The FIRST WORDS of each immediate child must pick up the thread of the source node's final
  thought — do not re-establish the scene or re-introduce the subject.
- Nodes with multiple parents must be written so they can follow any of their parents without
  jarring. Favour open, abstract continuations over hooks specific to one parent.
- Siblings at the same layer cover the same narrative moment from different angles, so any of
  them can flow into the same child without contradiction.
- Read every path aloud as you write. If a transition sounds like a non-sequitur, rewrite the
  child until the seam disappears.

RULES:
- start_nodes: IDs of nodes that connect DIRECTLY from the source node (immediate children only)
- Do NOT re-generate or include the source node itself
- Generate 2–4 layers forward from the source's layer, following the natural story arc
- Layer order: arrival → presence → curiosity → discovery → complication → intimacy → turn → consequence → echo → stillness
  Determine the source's layer from its tags, then generate only the layers that come AFTER it
- Use the same branching (2–4 children) and merging (multiple parents → 1 child) patterns as a full graph
- TERMINAL NODES: the final layer's nodes must have next: []
- TERMINAL NODES: stillness/ending nodes must have next: []. Never loop back to earlier layers.

BRANCHING and MERGING: same rules as full graph generation.
WEIGHTS: 1.0 default, 2.0 to favour, 0.5 to make rare.
TAGS: every node must have a tags array with:
1. Exactly one layer tag: arrival / presence / curiosity / discovery / complication / intimacy / turn / consequence / echo / stillness (whichever applies)
2. Custom tags for everything present in the node text — characters, themes, locations, objects, moods.
   Use short lowercase snake_case. Reuse the same tag across nodes whenever the same element recurs.
node IDs: short_snake_case, layer-prefixed (e.g. "turn_silence", "still_rest").

VOICE SETTINGS: match emotional tone to layer:
  arrival     stability 0.65  style 0.10
  presence    stability 0.60  style 0.15
  curiosity   stability 0.55  style 0.25
  discovery   stability 0.50  style 0.35
  complication stability 0.45  style 0.45
  intimacy    stability 0.40  style 0.50
  turn        stability 0.30  style 0.65
  consequence stability 0.25  style 0.70
  echo        stability 0.35  style 0.55
  stillness   stability 0.60  style 0.15
"""

SYSTEM_EXPAND = """\
You are expanding a single node in a narrative graph for an immersive audio installation.
You will receive one existing node and must generate new nodes that continue FROM it.

Each new node is a short spoken segment (40–100 words, ~15–35 seconds when read aloud).

AUTHOR DIRECTION — if an "AUTHOR DIRECTION" line appears in the prompt, it is the
HIGHEST PRIORITY instruction. Follow it even if it contradicts thematic continuity rules below.
The author's creative intent always takes precedence over default behavior.

THEMATIC CONTINUITY — important, but secondary to author direction:
The daughter nodes must feel like they are in the same room as the parent node.
- Keep the same specific imagery, sensory details, and atmosphere.
- If the parent mentions a specific character, place, or object, the daughters should
  stay in that world. Do not introduce unrelated new concepts.
- A listener should hear a daughter node immediately after the parent and feel continuity,
  not a scene change.
- The arc may deepen, shift in feeling, or reveal something new — but the SUBJECT stays close.
- Think of it as zooming in or turning a corner, not cutting to a new location.

Respond with ONLY a JSON object — no markdown fences, no explanation:
{
  "nodes": {
    "node_id": {
      "text": "Spoken text, 40-100 words.",
      "next": [],
      "weights": [],
      "tags": ["discovery", "toad", "revelation"],
      "voice_settings": {"stability": 0.50, "similarity_boost": 0.75, "style": 0.35}
    }
  },
  "connect_from": ["new_node_id_1", "new_node_id_2"]
}

TAGGING RULES — every node must have a tags array with:
1. Exactly one layer tag (arrival/presence/curiosity/discovery/complication/intimacy/turn/consequence/echo/stillness)
2. Custom tags for everything present in the node text — characters, themes, locations, objects.
   Any custom tag from the parent node should appear in daughters where that element is present.
   Prefer tags already used in the script (a list will be provided).

"connect_from": the new node IDs that the SOURCE node should gain edges to.
All other edges are between the new nodes themselves.

Layer progression rules (secondary to thematic continuity):
- Layer order: arrival → presence → curiosity → discovery → complication → intimacy → turn → consequence → echo → stillness
- Determine the source node's layer from its tags, then place new nodes in the NEXT layer(s)
- You may skip layers if the content calls for it, or span multiple layers in one expansion
- You may branch (source → multiple new nodes) or chain (source → A → B → C → ...)
- Branching then merging is encouraged: source → [A, B] and both A, B → C
- The prompt will specify an exact node count range — generate that many new nodes, no more, no fewer
- node IDs: short_snake_case, layer-prefixed (e.g. "disc_kelp_drift", "turn_silence", "still_rest")
- Weights default to 1.0 unless you have reason to favour one path
- Terminal nodes (stillness/echo end) must have next: [] — NEVER link back to arrival or start nodes.
  The runtime restarts automatically from a random start node when a terminal finishes.

VOICE SETTINGS: set voice_settings on every node (stability 0-1, similarity_boost 0.75, style 0-1).
  arrival     stab~0.65 style~0.10 | presence    stab~0.60 style~0.15
  curiosity   stab~0.55 style~0.25 | discovery   stab~0.50 style~0.35
  complicat   stab~0.45 style~0.45 | intimacy    stab~0.42 style~0.48
  turn        stab~0.38 style~0.55 | consequence stab~0.35 style~0.55
  echo        stab~0.40 style~0.50 | stillness   stab~0.60 style~0.15
  Adjust within the layer to match the specific emotional intensity of the node's text.
""" + STORYTELLING_DISCIPLINE

SYSTEM_GENERATE_SINGLE_NODE = """\
You are generating exactly ONE node for a narrative audio installation graph.
The node is a short spoken segment (40–100 words, ~15–35 seconds when read aloud).

THEMATIC CONTINUITY — most important rule:
- The node must feel like it naturally follows the parent node.
- Keep the same specific imagery, sensory details, and atmosphere.
- The arc may deepen, shift in feeling, or reveal something new — but the SUBJECT stays close.
- Think of it as zooming in or turning a corner, not cutting to a new location.

SIBLING DIFFERENTIATION — if sibling summaries are provided:
- Your node must take a DIFFERENT angle from already-generated siblings.
- Cover different aspects, emotions, or imagery — avoid retreading the same ground.

Respond with ONLY a JSON object — no markdown fences, no explanation:
{
  "node_id": "layer_prefix_descriptive_slug",
  "text": "Spoken text, 40-100 words.",
  "tags": ["layer_tag", "custom_tag_1", "custom_tag_2"],
  "voice_settings": {"stability": 0.50, "similarity_boost": 0.75, "style": 0.35},
  "vars": {},
  "vars_reasoning": "One short paragraph citing the specific words/moments in the node text that drive each non-zero variable. Variables at 0.00 don't need to be mentioned."
}

TAGGING RULES:
1. Exactly one layer tag (arrival/presence/curiosity/discovery/complication/intimacy/turn/consequence/echo/stillness)
2. Custom tags for characters, themes, locations, objects present in the text.
   Prefer tags already used in the script when applicable.

node_id: short_snake_case, layer-prefixed (e.g. "disc_kelp_drift", "turn_silence")

VOICE SETTINGS (stability 0-1, similarity_boost 0.75, style 0-1):
  arrival     stab~0.65 style~0.10 | presence    stab~0.60 style~0.15
  curiosity   stab~0.55 style~0.25 | discovery   stab~0.50 style~0.35
  complicat   stab~0.45 style~0.45 | intimacy    stab~0.42 style~0.48
  turn        stab~0.38 style~0.55 | consequence stab~0.35 style~0.55
  echo        stab~0.40 style~0.50 | stillness   stab~0.60 style~0.15
  Adjust to match the specific emotional intensity of the node's text.
""" + STORYTELLING_DISCIPLINE

SYSTEM_CROSS_LINK = """\
You are analyzing nodes in a narrative audio graph to find natural cross-branch connections.
Given a set of nodes at the same story layer and their existing children, suggest which nodes
from DIFFERENT branches could connect to each other's children.

A cross-link means: a listener who just heard node A could naturally hear node B next,
even though B was originally written as a continuation of a different branch.

Rules:
- Only suggest links where thematic or tonal continuity genuinely exists.
- Be conservative — fewer good links are better than many forced ones.
- Never link a node to its own children (those edges already exist).
- Cross-links go from a node at this layer to a child node on a different branch.

Respond with ONLY a JSON object:
{"cross_links": [{"from": "source_node_id", "to": "target_node_id"}, ...]}
No markdown fences, no explanation.
"""

SYSTEM_REWRITE = """\
You are rewriting spoken text for a single node in a narrative audio installation.
Text is spoken aloud over atmospheric lighting and weather effects.

Rules for the text:
- 40–100 words (~15–35 seconds when read aloud)
- Evocative, atmospheric, immersive language
- Fit naturally after any preceding context provided
- Match the emotional tone implied by the node's layer tag
- If a rough draft is provided, expand and improve it rather than replacing its intent

Rules for voice_settings (ElevenLabs):
- stability 0–1: higher = more consistent/calm, lower = more expressive/varied
- similarity_boost: keep at 0.75
- style 0–1: higher = more emotionally performed
- Layer guidance:
    arrival     stab~0.65 style~0.10 | presence    stab~0.60 style~0.15
    curiosity   stab~0.55 style~0.25 | discovery   stab~0.50 style~0.35
    complicat   stab~0.45 style~0.45 | intimacy    stab~0.42 style~0.48
    turn        stab~0.38 style~0.55 | consequence stab~0.35 style~0.55
    echo        stab~0.40 style~0.50 | stillness   stab~0.60 style~0.15
- Adjust to match the specific emotional intensity of the rewritten text

Respond with ONLY a JSON object in this exact format, no other text:
{
  "text": "...",
  "tags": ["discovery", "toad", "revelation"],
  "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.3}
}

TAGGING RULES — same as for expansion:
1. Exactly one layer tag (arrival/presence/curiosity/discovery/complication/intimacy/turn/consequence/echo/stillness)
2. Custom tags for characters, themes, locations, objects actually present in the rewritten text.
   Keep any existing custom tags that still apply. Add new ones if the rewrite introduces them.
   A list of existing script tags will be provided in the prompt — prefer those.
"""

SYSTEM_CHAT = """\
You are a creative collaborator helping develop narrative scripts for an immersive audio installation.
Scripts become spoken audio layered over atmospheric environments (ocean, storms, forest, etc.).

Help brainstorm themes, character voices, story arcs, and content.
Keep individual segments in mind — each will be ~15–35 seconds of spoken audio (40–100 words).

When the user clicks "Generate Graph", you will produce the actual JSON structure.
Until then, focus on ideas, themes, tone, and story development.

IMPORTANT — when a WORLD BIBLE block is appended below, treat it as REFERENCE MATERIAL ONLY
(voice, tone, setting, characters, places). It must NEVER override the subject of the user's
actual message. The user's message defines the topic. Always stay on that topic — use bible
details only insofar as they are RELEVANT to what the user is asking.
"""

SYSTEM_DETERMINE_VARS = """\
You are analyzing narrative script nodes to determine story variable values.
Given variable definitions and a list of nodes with their text, assign a value
(0.0–1.0) for each variable on each node based on its text content.

Rules:
- 0.0 = the variable's quality is NOT meaningfully present in the node text.
- 1.0 = the variable's quality is the dominant force in the node.
- Be decisive. Most variables on most nodes should be 0.0.
- Only assign non-zero values when the text explicitly contains that quality.
- Avoid clustering everything in the 0.2–0.6 range.
- Reserve 0.8–1.0 for nodes where that quality is unmistakably dominant.

Respond with ONLY a JSON object mapping node IDs to their variable values:
{"node_id": {"var_name": value, ...}, ...}
No markdown fences, no explanation — just the JSON object.
"""

SYSTEM_SUGGEST_VARS = """\
You are designing the NARRATIVE VARIABLES for a long-form audio narrative
intended to drive a real-time visual installation.

A narrative variable is a continuous lever in [0,1] that the AI sets per
node and that visual/audio effects read each frame. Good variables:

  - Name ONE clear emotional or structural dimension. Single words; lowercase.
  - Are STRUCTURALLY INDEPENDENT — knowing one tells you little about another.
    Variables that always co-vary (e.g. "fear" and "dread") are redundant.
  - Can swing across the full 0..1 range over the course of an arc — they
    are LEVERS, not statistics about the world. "weather" is a bad variable;
    "exposure" or "danger" are better.
  - Map intuitively to visual/audio cues a shader can read: color
    temperature, motion, intensity, density, focus.
  - Together cover the EMOTIONAL ARC the narrative is built around — the
    set should answer "what does the audience FEEL changing across an arc?"

Prefer ABSTRACT EMOTIONAL/STRUCTURAL terms over plot terms. Examples of
good variable families:
  - signal, dread, yearning, defiance, dissolution, velocity
    (cyberpunk: perception, surveillance pull, loss, push-back, fade, motion)
  - tension, intimacy, hope, grief, defiance, stillness
  - urgency, isolation, devotion, decay, recognition, motion

Output ONLY a JSON array (no markdown fences, no commentary, no explanation):
[
  {"name": "lowercase_word", "description": "One sentence: when this is high vs low, and what visual cue it might drive."},
  ...
]

Aim for exactly 6 variables (or 4-6 if fewer truly cover the arc). Names
should be single lowercase words (snake_case OK). Descriptions should be
ONE sentence, ≤ 120 chars each, framed in terms of "high = X, low = Y;
drives visual cue Z" where possible.
"""

SYSTEM_ARC_CHAT = """\
You are helping an author develop a story arc for an immersive audio installation.
Story arcs are used to guide generation of a narrative node graph — each arc has a premise,
recurring themes/motifs, and a beat for each of the 10 story layers (arrival through stillness).

Each node will become ~15–35 seconds of spoken audio. The full arc plays out over 10 layers:
  arrival → presence → curiosity → discovery → complication → intimacy → turn → consequence → echo → stillness

Your role: help the author refine their premise, suggest beats for specific layers,
develop themes and recurring motifs, identify character voices, and deepen the emotional arc.

When arc fields are shown, treat them as the current state. Respond conversationally.
Keep suggestions practical and grounded in the established tone.
Be specific — suggest actual text or directions, not just abstract advice.

── HOW THESE FIELDS ARE USED DOWNSTREAM (so your suggestions land at the right grain) ──

PREMISE — injected into every node call with a layer-specific role label ('establish this premise's
world' at arrival, 'the bind IS this premise made specific' at complication, 'the action MUST BE this
premise enacted' at turn, 'afterimage' at stillness). So premise suggestions you make must:
  - Be 600-1000 chars, several sentences. Long enough to carry both the character's durable anchors
    (who they are, key recurring facts) AND the situation/stakes of this arc.
  - Premise is the ONLY field that reliably reaches every node-generation call other than motif/themes/
    beats, so character identity belongs HERE, not in `notes` (notes is consulted only on the first
    seed call and ignored afterward).
  - Read cleanly through every layer role above (test it both ways).
  - Capture the IRREDUCIBLE thing — character + situation + what's at stake — not a plot summary.

BEATS — each beat is the per-layer steering text the generator sees as 'LAYER DIRECTION'. The arc's
story context is dense, so each beat needs to out-shout it by carrying three elements:
  1. SCENE/SITUATION ANCHOR — a concrete moment, not a theme.
  2. EMOTIONAL OR COGNITIVE MOVE — what specifically changes (one transformation).
  3. BOUNDARY — what this beat is NOT, or a restraint to keep the AI from over-citing.
Length per beat: 100-300 chars (2-3 sentences). Don't exceed ~400 chars — past that, the model
reads the beat as text-to-deliver rather than guidance.

MOTIF — must be CONCRETE and SENSORY (a smell, a sound, an object), not thematic. 60-150 chars.

When the author asks for a draft premise, beat, or motif, write it at the above grain directly,
not as advice about what one would look like.
"""

SYSTEM_EXTRACT_CODEX = """\
You are building a CODEX for a narrative audio-installation script: cards for the
recurring concepts the story world is built from, distilled from the world bible
and the existing node graph.

Entity kinds:
  character — a person, creature, or personified force that acts or speaks
  location  — a place the narrative returns to
  event     — ONE specific happening (past or ongoing) that stories can reference
              from different angles ("the 2041 quake", not "earthquakes")
  theme     — an emotional/structural undercurrent the narrative keeps touching
  idea      — a concept or institution of the world (a ritual, a technology, a rule)
  object    — a physical thing that recurs (a key, a drink, a bell)

RULES:
- slugs: short lowercase snake_case. If the TAG CENSUS in the prompt already has a
  tag naming the same concept, USE THAT TAG as the slug — this is what links the
  card to existing nodes. Never invent a near-duplicate of an existing tag.
- Do not re-propose slugs listed as already in the codex.
- essence: 1-3 sentences capturing the irreducible core — written to steer prose
  generation, not as encyclopedia text.
- facts: 3-8 short durable canon facts. Things that must stay true across every
  story that uses this entity. No atmosphere — atmosphere goes in sensory.
- sensory: 0-5 concrete sensory anchors (a smell, a sound, a texture).
- voice_notes: how the narrator should handle this entity (register, distance,
  what is never said aloud). One or two sentences; empty string if nothing special.
- relationships: links to OTHER slugs in this output or the existing codex,
  each with a short "nature" phrase.
- Propose the entities that CARRY the world — typically 8-25 cards. Prefer fewer,
  stronger cards over exhaustive coverage. Every character and location that
  recurs; only the most load-bearing themes/ideas/events/objects.

Respond with ONLY this JSON — no markdown fences, no explanation:
{
  "entities": {
    "slug": {
      "kind": "character",
      "name": "Display Name",
      "aliases": ["other name"],
      "essence": "...",
      "facts": ["..."],
      "sensory": ["..."],
      "voice_notes": "...",
      "relationships": [{"to": "other_slug", "nature": "..."}]
    }
  }
}
"""

SYSTEM_CODEX_CHAT = """\
You are a creative collaborator helping an author develop CODEX ENTITY CARDS for
a narrative audio installation. The codex holds the recurring concepts —
characters, locations, events, themes, ideas, objects — that every story in the
script is built from. Cards are injected into generation calls as CANON, so what
lands on a card shapes all future prose.

Card fields and their grain (so your suggestions land right):
  essence       — 1-3 sentences of irreducible core, written to STEER PROSE
                  generation, not encyclopedia text
  facts         — durable canon: 3-8 short facts that must stay true in EVERY
                  story that uses the entity (expensive — propose sparingly)
  sensory       — 0-5 concrete anchors: a smell, a sound, a texture
  voice_notes   — how the narrator handles this entity (register, distance,
                  what is never said aloud)
  relationships — links to other codex slugs with a short nature phrase
  aliases       — other names the entity goes by

Help the author deepen an entity: find its contradictions and hooks, propose
facts worth canonizing (and flag ones too limiting to canonize), suggest
relationships to other codex entities, and notice when the conversation is
really describing a SEPARATE entity that deserves its own card — say so.

When a WORLD BIBLE / CODEX block is appended below, treat it as reference
material. The user's message defines the topic. When asked for a draft field,
write it at the grain above directly — not advice about what it would look like.

When the author is happy, they press "Distill Chat → Entity" and the
conversation is distilled onto the card — so drive toward concrete,
distillable material.
"""

SYSTEM_DISTILL_ENTITY = """\
You are distilling a brainstorming conversation into a CODEX ENTITY CARD for a
narrative audio installation. Cards are injected into prose generation as CANON.

You receive the CURRENT CARD and the CONVERSATION. Return the updated card:
refine and extend, don't discard — keep existing card content unless the
conversation contradicts or supersedes it.

Field rules:
- essence: 1-3 sentences of irreducible core, written to steer prose.
- facts: durable canon only, 3-8 short facts, each true in every future story.
- sensory: 0-5 concrete anchors (a smell, a sound, a texture).
- voice_notes: how the narrator handles this entity; '' if nothing special.
- relationships: reference EXISTING codex slugs (list provided) or slugs you
  define in related_entities.
- related_entities: 0-3 OTHER concepts the conversation developed enough to
  deserve their own card. Slugs: short lowercase snake_case. Do not re-propose
  existing slugs. Omit or leave empty when the chat stayed on one entity.

Respond with ONLY this JSON — no markdown fences, no explanation:
{
  "entity": {"kind": "character", "name": "...", "aliases": ["..."],
             "essence": "...", "facts": ["..."], "sensory": ["..."],
             "voice_notes": "...",
             "relationships": [{"to": "slug", "nature": "..."}]},
  "related_entities": {"slug": {"kind": "...", "name": "...", "essence": "...",
                                 "facts": ["..."]}}
}
"""

SYSTEM_PROPOSE_WEB = """\
You are planning a WEB of interconnected stories for a narrative audio installation.
Multiple stories coexist in one script and cross-pollinate through shared codex
entities, canon events, callbacks, and junctions (runtime drift edges that let a
listener's random walk slide from one story into another).

You receive the world bible and the full codex (canon entity cards + canon events)
in the system prompt, and the EXISTING STORIES with their relations in the user
prompt. Propose a richer web: new stories anchored in the codex, plus typed
relations among stories (existing and new).

Relation types (relations point FROM one story TO another):
  shares_character — the same codex entities appear in both stories
  same_event       — both stories touch one canon event from different angles
  callback         — FROM story's text references TO story's canon events
  sequel           — FROM story takes place after TO story
  junction         — runtime drift: low-weight edges let a walk slide FROM → TO

RULES:
- New stories must be anchored in EXISTING codex entities — recombine the cast in
  new configurations rather than inventing new characters. 2-5 cast slugs each.
- When the user prompt lists DORMANT CODEX ENTITIES (cast in no story yet),
  PRIORITIZE them: build each new story around 1-3 dormant entities, pairing them
  with at most 1-2 well-used ones for continuity. The CAST USAGE counts show which
  entities are already heavily played — avoid adding more load to the top of that
  list. Dormant entities exist because the author cared enough to write them;
  giving them stories is a primary goal of this pass.
- premise: 300-700 chars carrying character anchors + specific situation + stakes.
  It reaches every node-generation call, so it must read cleanly as 'the bind is
  <premise>' and 'the action must be <premise> enacted'.
- structure: 4-10 beats, each {"layer": <archetype>, "direction": "..."}. layer is
  one of: arrival, presence, curiosity, discovery, complication, intimacy, turn,
  consequence, echo, stillness. Directions are concrete (scene anchor + one
  transformation + a boundary), 100-300 chars. Short structures suit vignette or
  callback stories; the classic 10 suits a full arc.
- themes: comma-separated. motif: one concrete sensory thread, 60-150 chars.
- relations: 1-3 per new story, referencing story NAMES (existing names verbatim,
  or the names you propose). Junctions only between tonally compatible stories;
  be conservative with same_event.
- Propose 2-4 new stories unless the instructions say otherwise.

Respond with ONLY this JSON — no markdown fences, no explanation:
{
  "stories": [
    {"name": "...", "premise": "...", "themes": "a, b", "motif": "...",
     "cast": ["slug"], "structure": [{"layer": "arrival", "direction": "..."}]}
  ],
  "relations": [
    {"from": "Story Name", "to": "Other Story", "type": "junction", "note": "..."}
  ]
}
"""

SYSTEM_EXTRACT_CANON = """\
You are extracting CANON from one generated story of a narrative audio installation:
durable facts about codex entities, and specific canon events that other stories can
later reference (callbacks) or retell from different angles.

The system prompt carries the story's entity cards and the EXISTING canon events.
The user prompt carries the story's premise and its node texts.

RULES:
- facts: only DURABLE, load-bearing facts this story establishes about a cast
  entity — things every future story must keep true. Not atmosphere, not one-off
  imagery. 0-4 per entity. NEVER restate a fact already on the entity's card.
- events: 0-4 SPECIFIC happenings this story establishes (something occurred,
  changed hands, was decided, was lost). One concrete sentence each, past tense,
  with the involved entity slugs. Skip anything already in the existing canon.
- When in doubt, extract LESS. Canon is expensive — every future story must honor it.

Respond with ONLY this JSON — no markdown fences, no explanation:
{
  "facts": {"entity_slug": ["new durable fact", "..."]},
  "events": [{"summary": "...", "entities": ["slug"]}]
}
"""

SYSTEM_WEAVE_JUNCTIONS = """\
You are weaving JUNCTIONS between two stories in a narrative audio graph: rare
cross-story edges where a listener's random walk drifts out of story A into story B
without noticing a boundary.

You receive nodes from story A (drift sources) and story B (drift targets), each
with its beat archetype, its position as a percentage through its own story, and a
text excerpt.

A good junction: the target reads as a natural next thought after the source —
shared imagery, tone, or subject — the positions fit (a mid-story source should
land in an early-to-mid target, never into a target's final beats), and the
EMOTIONAL REGISTER carries across: never drop a listener from a tense, charged
moment into a placid opening (or the reverse) without a step between.

For each link, judge the seam itself:
- If the target genuinely flows from the source as-is, emit the link alone.
- If the PAIRING is right but the seam needs a stepping stone, also write
  "bridge_text": one short spoken passage (25-60 words) that carries the
  listener from the source's ending into the target's opening — same narrator
  voice, concrete imagery shared with both sides, no exposition, no summary
  of either story. It will play between the two nodes.

RULES:
- Suggest 1-4 links. Fewer good links beat many forced ones; zero is acceptable.
- Never link nodes whose texts share nothing concrete.
- "from" is a story-A node id; "to" is a story-B node id.

Respond with ONLY this JSON — no markdown fences, no explanation:
{"links": [{"from": "node_id", "to": "node_id",
            "bridge_text": "...only when the seam needs it..."}]}
"""

SYSTEM_WEAVE_CALLBACKS = """\
You are weaving CALLBACKS into an existing story: rewriting 1-2 of its nodes so
they reference another story's canon — organically, as a memory, a rumor, an
overheard fragment, or an object carrying the other story's residue. Never as
exposition, never as a summary of the other story.

You receive the TARGET CANON (the other story's events and premise) and CANDIDATE
NODES from the story being rewritten.

RULES:
- Rewrite 1-2 candidates, no more. Choose the nodes where a callback lands most
  naturally; if none do, return an empty list.
- Keep each rewrite within ±20% of the original word count, in the same voice and
  register, doing the same beat's story-work. The callback is one thread woven
  through the existing fabric — the node's subject does not change.
- Reference the canon obliquely and concretely (a name, an object, a place), not
  by explaining what happened.
- add_tags: entity slugs newly present in the rewritten text.

Respond with ONLY this JSON — no markdown fences, no explanation:
{"rewrites": [{"node_id": "...", "new_text": "...", "add_tags": ["slug"]}]}
"""

SYSTEM_CONSISTENCY_AUDIT = """\
You are auditing one story's generated nodes against its CANON — the entity
cards and canon events provided in the system prompt.

Flag ONLY genuine conflicts:
  hard — the node states something a card fact or canon event makes impossible
  soft — the node strains canon: implies a contradiction, or uses an entity in
         a way its essence or voice notes rule out

Do NOT flag: atmosphere; new-but-compatible details; a node simply not
mentioning canon; style or quality concerns; claims about things that have no
card or event. False positives destroy trust in the audit — when unsure, stay
silent.

For each issue, quote or closely paraphrase the offending claim and name the
SPECIFIC fact, event, or card element it conflicts with.

Respond with ONLY this JSON — no markdown fences, no explanation:
{"issues": [{"node_id": "...", "claim": "...", "conflicts_with": "...",
             "severity": "hard"}]}
An empty issues list is a good outcome.
"""

SYSTEM_SEAM_AUDIT = """\
You are auditing SEAMS in a narrative audio graph: places where one spoken
node ends and a connected node begins. The listener hears the two back to
back with a few seconds of silence between. Flag seams that would jar:

  hard — a non-sequitur: the child re-establishes a different scene from
         nowhere, contradicts the parent's moment, or responds to something
         that never happened
  soft — hearable but rough: an abrupt emotional register drop or spike,
         the child re-introducing a subject the parent just covered as if
         new, or a topic swerve with no shared hook

Do NOT flag: intentional beat progression (a calm close after an intense
moment is fine when the imagery carries through), different angles on the
same moment, plain deepening of the scene, or seams that merely change pace.
Most seams in a healthy graph are fine — false positives destroy trust in
the audit; when unsure, stay silent.

Each seam shows the parent's ENDING and the child's OPENING, and whether the
edge crosses between two different stories (cross-story seams deserve the
closest listen).

Respond with ONLY this JSON — no markdown fences, no explanation:
{"issues": [{"from": "parent_id", "to": "child_id", "severity": "hard",
             "reason": "..."}]}
An empty issues list is a good outcome.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Data Model
# ─────────────────────────────────────────────────────────────────────────────

def _load_project_weather_sets(script_path: Optional[Path]):
    """Load the WEATHER_SETS dict from the project that owns this script.

    Walks up from script_path looking for a directory containing
    ``weather_params.py``, then imports it in an isolated namespace and
    returns the WEATHER_SETS dict. Returns ``{}`` on any failure
    (missing file, import error, missing/malformed WEATHER_SETS).
    """
    if script_path is None:
        return {}
    script_path = Path(script_path).resolve()
    project_dir = None
    candidate = script_path.parent
    for _ in range(6):
        if (candidate / 'weather_params.py').exists():
            project_dir = candidate
            break
        if candidate == candidate.parent:
            break
        candidate = candidate.parent
    if project_dir is None:
        return {}

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        '_narr_editor_weather_params', project_dir / 'weather_params.py')
    if spec is None or spec.loader is None:
        return {}
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        print(f"[narrative_editor] failed to load {project_dir}/"
              f"weather_params.py: {exc}")
        return {}

    weather_sets = getattr(mod, 'WEATHER_SETS', {})
    return weather_sets if isinstance(weather_sets, dict) else {}


def _reverse_lookup_weather_set(script_path: Optional[Path], weather_sets: dict):
    """Given a script path and a WEATHER_SETS dict, find the set whose
    narrative_script field matches this script. Returns the set name or
    None.

    Match strategy: a weather_set's narrative_script is typically
    relative to the project's media root (e.g.
    ``"sounds/cyberpunk/sounds.json"``). The script's absolute path
    contains that suffix, so suffix-matching is robust.
    """
    if script_path is None or not weather_sets:
        return None
    script_str = str(Path(script_path).resolve()).replace('\\', '/').lower()
    for set_name, set_data in weather_sets.items():
        narr = (set_data or {}).get('narrative_script')
        if not narr:
            continue
        narr_norm = str(narr).replace('\\', '/').lower()
        if script_str.endswith(narr_norm):
            return set_name
    return None


def _find_weather_states_for_script(script_path: Optional[Path],
                                     explicit_set: Optional[str] = None):
    """Given a script.json path, return ``(set_name, list_of_state_values)``
    for the trigger-state dropdown.

    If ``explicit_set`` is provided AND that set exists in the project's
    WEATHER_SETS, it wins — the user has overridden the association via
    the script's top-level ``weather_set`` field. Otherwise we
    reverse-lookup which set's ``narrative_script`` points at this file.

    Returns ``(None, [])`` if no association can be resolved.
    """
    weather_sets = _load_project_weather_sets(script_path)
    if not weather_sets:
        return None, []

    # Explicit override wins
    if explicit_set and explicit_set in weather_sets:
        states = list((weather_sets[explicit_set] or {}).get('states', []))
        return explicit_set, states

    # Otherwise, reverse-lookup via narrative_script
    set_name = _reverse_lookup_weather_set(script_path, weather_sets)
    if set_name is None:
        return None, []
    states = list((weather_sets[set_name] or {}).get('states', []))
    return set_name, states


class ScriptData:
    """In-memory representation of a script.json file."""

    def __init__(self, data: dict = None):
        self._data = deepcopy(data or SCRIPT_TEMPLATE)
        self.path: Optional[Path] = None
        self.dirty = False
        # Cache for trigger_state dropdown — populated when path is set.
        # (set_name, list[str]) of valid weather states for this script.
        self._associated_set: Optional[str] = None
        self._trigger_state_options: list = []
        self._migrate_layers()

    # ── Layer migration ───────────────────────────────────────────────────

    def _migrate_layers(self):
        """Migrate old 8-layer names to new 10-layer names in nodes and arcs."""
        if not _LAYER_MIGRATION:
            return
        # Migrate node tags
        for nd in self._data.get('nodes', {}).values():
            tags = nd.get('tags', [])
            nd['tags'] = [_LAYER_MIGRATION.get(t, t) for t in tags]
        # Migrate arc beat keys
        for arc in self._data.get('arcs', {}).values():
            beats = arc.get('beats', {})
            new_beats = {}
            for key, val in beats.items():
                new_key = _LAYER_MIGRATION.get(key, key)
                new_beats[new_key] = val
            # Ensure all new layer keys exist
            for layer in LAYER_ORDER:
                new_beats.setdefault(layer, '')
            arc['beats'] = new_beats

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def name(self): return self._data["name"]
    @name.setter
    def name(self, v): self._data["name"] = v; self.dirty = True

    @property
    def description(self): return self._data.get("description", "")

    @property
    def nodes(self) -> dict: return self._data["nodes"]

    @property
    def start_nodes(self) -> list: return self._data["start_nodes"]

    @property
    def story_context(self) -> str: return self._data.get("story_context", "")

    @property
    def story_context_focused(self) -> str:
        """Deprecated alias. The editor no longer distinguishes between
        a full context and a "focused" subset — the single
        ``story_context`` is what gets sent to the AI (with prompt
        caching handling the size). Kept as a read-only alias so older
        callsites that reference ``script.story_context_focused``
        silently work against the unified context."""
        return self._data.get("story_context", "")

    def set_story_context(self, text: str):
        self._data["story_context"] = text
        # Drop the deprecated field if present so re-saved scripts are
        # clean of the old dual-context layout.
        self._data.pop("story_context_focused", None)
        self.dirty = True

    def set_story_context_focused(self, text: str):
        """Deprecated. Behaves as a setter on the unified
        ``story_context`` field — silently routes through
        ``set_story_context`` so legacy code paths keep working."""
        self.set_story_context(text)

    @property
    def variables(self) -> list:
        """Story-level variable definitions: [{"name": ..., "description": ...}, ...]"""
        return self._data.setdefault("variables", [])

    def set_variables(self, var_list: list):
        """Replace all variable definitions (max 6)."""
        self._data["variables"] = list(var_list)[:6]
        self.dirty = True

    # ── Codex (entity) management ────────────────────────────────────────────
    # See the "Codex" comment block above ENTITY_KINDS for the design. The
    # short version: entities live in-file, slugs double as node tags, and
    # every accessor goes through here so a shared external codex can swap
    # in behind the same interface later.

    @property
    def entities(self) -> dict:
        return self._data.setdefault('entities', {})

    @staticmethod
    def sanitize_entity_slug(name: str) -> str:
        """Lowercase snake_case slug from a display name. '' if nothing usable."""
        return re.sub(r'[^a-z0-9]+', '_', (name or '').strip().lower()).strip('_')

    def _free_entity_slug(self, base: str) -> str:
        """First slug derived from `base` that is neither taken nor reserved."""
        final, counter = base, 2
        while final in self.entities or final in RESERVED_ENTITY_SLUGS:
            final = f'{base}_{counter}'
            counter += 1
        return final

    @staticmethod
    def _entity_match_key(text: str) -> str:
        """Normalized form for duplicate detection: sanitized slug with
        leading articles dropped, so 'The Goat' ≡ 'the_goat' ≡ 'goat'."""
        slug = ScriptData.sanitize_entity_slug(text)
        parts = [p for p in slug.split('_') if p]
        while len(parts) > 1 and parts[0] in ('the', 'a', 'an'):
            parts.pop(0)
        return '_'.join(parts)

    def find_equivalent_entity(self, name_or_slug: str) -> Optional[str]:
        """Slug of an existing entity this name/slug duplicates, or None.
        Matches article-insensitively against every existing slug, display
        name, and alias — the exact-slug checks at the AI apply sites let
        'goat' through when 'the_goat' already existed, silently creating
        duplicate cards."""
        key = self._entity_match_key(name_or_slug)
        if not key:
            return None
        for slug, card in self.entities.items():
            if not isinstance(card, dict):
                continue
            cands = [slug, card.get('name', '')]
            cands += [a for a in (card.get('aliases') or [])
                      if isinstance(a, str)]
            if any(self._entity_match_key(c) == key for c in cands if c):
                return slug
        return None

    def add_entity(self, name: str, kind: str = 'character',
                   slug: str = '') -> Optional[str]:
        """Create a new entity card. Returns the final slug, or None."""
        base = self.sanitize_entity_slug(slug or name)
        if not base:
            return None
        final = self._free_entity_slug(base)
        card = deepcopy(ENTITY_TEMPLATE)
        card['kind'] = kind if kind in ENTITY_KINDS else 'idea'
        card['name'] = (name or '').strip() or final
        self.entities[final] = card
        self.dirty = True
        return final

    def update_entity(self, slug: str, fields: dict):
        ent = self.entities.get(slug)
        if ent is None:
            return
        # Change-aware — codex form flushes fire on focus changes too.
        if all(ent.get(k) == v for k, v in fields.items()):
            return
        ent.update(fields)
        self.dirty = True

    def delete_entity(self, slug: str):
        """Remove a card and scrub relationship references to it. Node tags
        are content and are deliberately left untouched."""
        if self.entities.pop(slug, None) is None:
            return
        for card in self.entities.values():
            rels = card.get('relationships', [])
            card['relationships'] = [r for r in rels if r.get('to') != slug]
        self.dirty = True

    def rename_entity(self, old: str, new: str,
                      update_tags: bool = True) -> Optional[str]:
        """Rename a slug; fixes relationship refs and (optionally) node tags
        and node 'entities' lists. Returns the final slug, or None."""
        if old not in self.entities:
            return None
        base = self.sanitize_entity_slug(new)
        if not base or base == old:
            return None
        final = self._free_entity_slug(base)
        self.entities[final] = self.entities.pop(old)
        for card in self.entities.values():
            for r in card.get('relationships', []):
                if r.get('to') == old:
                    r['to'] = final
        if update_tags:
            for nd in self._data['nodes'].values():
                tags = nd.get('tags', [])
                if old in tags:
                    nd['tags'] = [final if t == old else t for t in tags]
                ents = nd.get('entities')
                if ents and old in ents:
                    nd['entities'] = [final if e == old else e for e in ents]
        self.dirty = True
        return final

    def entity_usage(self, slug: str) -> list:
        """Node IDs referencing this entity (via tag or explicit list)."""
        out = []
        for nid, nd in self._data['nodes'].items():
            if slug in nd.get('tags', []) or slug in (nd.get('entities') or []):
                out.append(nid)
        return out

    def codex_summary(self, slugs: Optional[list] = None) -> str:
        """Compact text block of entity cards for AI prompts. Stable kind-
        then-slug ordering so it stays cache-friendly as a prompt prefix.
        Pass `slugs` to scope to a cast subset (story generation later)."""
        pool = self.entities
        if slugs is not None:
            pool = {s: pool[s] for s in slugs if s in pool}
        if not pool:
            return ''
        def _order(kv):
            kind = kv[1].get('kind', 'idea')
            k = ENTITY_KINDS.index(kind) if kind in ENTITY_KINDS else 99
            return (k, kv[0])
        lines = []
        for slug, card in sorted(pool.items(), key=_order):
            lines.append(f'[{card.get("kind", "idea").upper()}] {card.get("name", slug)}  (tag: {slug})')
            if card.get('aliases'):
                lines.append(f'  aka: {", ".join(card["aliases"])}')
            if card.get('essence'):
                lines.append(f'  {card["essence"]}')
            for f in card.get('facts', []):
                lines.append(f'  - {f}')
            if card.get('sensory'):
                lines.append(f'  senses: {"; ".join(card["sensory"])}')
            if card.get('voice_notes'):
                lines.append(f'  voice: {card["voice_notes"]}')
            for r in card.get('relationships', []):
                # Only links whose target actually exists — dangling refs
                # (e.g. from imports) would pollute generation prompts.
                if r.get('to') in self.entities:
                    lines.append(f'  ↔ {r["to"]}: {r.get("nature", "")}')
            lines.append('')
        return '\n'.join(lines).rstrip()

    def import_entities_from(self, other_path, slugs: list) -> list:
        """Copy entity cards from another script.json (copy-by-value; the
        self-contained model). Slugs are kept IDENTICAL — they are the tag
        linkage, and renaming on import would break cross-script identity.
        Existing slugs are skipped, never overwritten. Returns imported slugs."""
        try:
            data = json.loads(Path(other_path).read_text(encoding='utf-8'))
        except Exception:
            return []
        pool = data.get('entities', {})
        src_label = f'import:{Path(other_path).parent.name}'
        imported = []
        for slug in slugs:
            card = pool.get(slug)
            if not isinstance(card, dict) or slug in self.entities \
                    or slug in RESERVED_ENTITY_SLUGS:
                continue
            new_card = deepcopy(ENTITY_TEMPLATE)
            new_card.update(deepcopy(card))
            new_card['source'] = src_label
            # Development chat stays with the donor script — cards travel clean.
            new_card.pop('chat_history', None)
            self.entities[slug] = new_card
            imported.append(slug)
        if imported:
            self.dirty = True
        return imported

    # ── Arc management ──────────────────────────────────────────────────────

    @property
    def arcs(self) -> dict:
        return self._data.setdefault('arcs', {})

    @property
    def active_arc_id(self) -> str:
        return self._data.get('active_arc_id', '')

    def active_arc(self) -> Optional[dict]:
        aid = self.active_arc_id
        return self.arcs.get(aid) if aid else None

    def set_active_arc(self, arc_id: str):
        if self._data.get('active_arc_id') == arc_id:
            return
        self._data['active_arc_id'] = arc_id
        self.dirty = True

    def add_arc(self) -> str:
        existing = set(self.arcs.keys())
        i = 1
        while f'arc_{i:03d}' in existing:
            i += 1
        arc_id = f'arc_{i:03d}'
        self.arcs[arc_id] = {
            'name': 'New Arc',
            'premise': '',
            'themes': '',
            'motif': '',
            'beats': {k: '' for k in LAYER_ORDER},
            'notes': '',
            'chat_history': [],
        }
        self.dirty = True
        return arc_id

    def delete_arc(self, arc_id: str):
        self.arcs.pop(arc_id, None)
        if self._data.get('active_arc_id') == arc_id:
            self._data['active_arc_id'] = ''
        self.dirty = True

    def save_arc(self, arc_id: str, data: dict):
        arc = self.arcs.get(arc_id)
        if arc is None:
            return
        # Change-aware: panel flushes (tab switches, selection changes, app
        # close) call this with unchanged values — those must not mark the
        # script dirty, or Ctrl+S is immediately followed by a phantom
        # "unsaved changes" prompt on quit.
        if all(arc.get(k) == v for k, v in data.items()):
            return
        arc.update(data)
        self.dirty = True

    def get_node_arc_id(self, node_id: str) -> str:
        nd = self._data.get('nodes', {}).get(node_id, {})
        return nd.get('arc_id', '') if isinstance(nd, dict) else ''

    def set_node_arc_id(self, node_id: str, arc_id: str):
        nd = self._data.get('nodes', {}).get(node_id)
        if isinstance(nd, dict):
            nd['arc_id'] = arc_id or ''
            self.dirty = True

    def get_arc(self, arc_id: str) -> dict:
        if not arc_id:
            return {}
        return self._data.get('arcs', {}).get(arc_id, {}) or {}

    # ── Story extensions on arcs (v2) ───────────────────────────────────────
    # v2 "stories" ARE arcs, extended in place with two optional fields:
    #   cast:      [entity_slug, ...] — which codex entities are in play
    #   structure: [{"layer": <LAYER_ORDER archetype>, "direction": str}, ...]
    #              — the flexible beat sequence (any length/order/repeats)
    # Storing them on the arc keeps v1 fully compatible: v1's arc editor
    # saves via dict.update() so unknown fields survive, and node arc_id
    # stays the single story-linkage field. The classic 10-layer `beats`
    # dict is kept in sync as a DERIVED view whenever v2 writes a structure,
    # so v1 still displays sensible beats. Ownership rule: once a script is
    # edited in v2, do story edits in v2 — v1 beat edits don't propagate
    # back into structure.

    @staticmethod
    def default_structure() -> list:
        """The classic 10-layer arc, as a structure list."""
        return [{'layer': layer, 'direction': ''} for layer in LAYER_ORDER]

    @staticmethod
    def beats_from_structure(structure: list) -> dict:
        """Derive the v1 beats dict from a structure list. Repeated
        archetypes concatenate their directions with ' / '."""
        beats = {layer: '' for layer in LAYER_ORDER}
        for entry in structure:
            layer = entry.get('layer', '')
            d = (entry.get('direction') or '').strip()
            if layer in beats and d:
                beats[layer] = d if not beats[layer] else beats[layer] + ' / ' + d
        return beats

    def get_story_structure(self, arc_id: str) -> list:
        """Resolved beat sequence for a story. Priority: the arc's explicit
        `structure` field; else derived from its classic beats dict; else
        the default 10-layer arc (also the path for arc_id ''). Always a
        non-empty list of {'layer', 'direction'} with valid archetypes."""
        arc = self.get_arc(arc_id)
        raw = arc.get('structure')
        if isinstance(raw, list):
            clean = [{'layer': e['layer'],
                      'direction': str(e.get('direction') or '')}
                     for e in raw
                     if isinstance(e, dict) and e.get('layer') in LAYER_ORDER]
            if clean:
                return clean
        beats = arc.get('beats', {}) if arc else {}
        return [{'layer': layer, 'direction': str(beats.get(layer) or '')}
                for layer in LAYER_ORDER]

    def set_story_structure(self, arc_id: str, structure: list):
        """Write a story's structure and re-derive the v1 beats view."""
        if arc_id not in self.arcs:
            return
        clean = [{'layer': e.get('layer'),
                  'direction': str(e.get('direction') or '')}
                 for e in structure if e.get('layer') in LAYER_ORDER]
        if not clean:
            clean = self.default_structure()
        if self.arcs[arc_id].get('structure') == clean:
            return   # beats are derived — unchanged structure means both match
        self.arcs[arc_id]['structure'] = clean
        self.arcs[arc_id]['beats'] = self.beats_from_structure(clean)
        self.dirty = True

    def get_story_cast(self, arc_id: str) -> list:
        """Entity slugs in play for this story (only ones still in the
        codex). Empty list = no cast = NOTHING from the codex is injected
        into that story's generation (explicit opt-in per story)."""
        arc = self.get_arc(arc_id)
        cast = arc.get('cast') if arc else None
        if not isinstance(cast, list):
            return []
        return [s for s in cast if s in self.entities]

    def set_story_cast(self, arc_id: str, cast: list):
        if arc_id in self.arcs and self.arcs[arc_id].get('cast') != list(cast):
            self.arcs[arc_id]['cast'] = list(cast)
            self.dirty = True

    @staticmethod
    def _canon_events_block(events: list) -> str:
        if not events:
            return ''
        lines = ['CANON EVENTS (established happenings — reference freely, never contradict):']
        for ev in events:
            ents = ', '.join(ev.get('entities', []))
            lines.append(f'- [{ev.get("id", "?")}] {ev.get("summary", "")}'
                         + (f'  (entities: {ents})' if ents else ''))
        return '\n'.join(lines)

    def cast_codex_for(self, arc_id: str) -> str:
        """Codex prompt block for a story's generation run: the cast's
        cards followed by the canon events relevant to that cast.

        NO CAST = NOTHING — a story with no checked entities gets no codex
        material at all (cards or events). The codex is opt-in per story;
        use full_codex_block() for script-level passes that genuinely need
        everything."""
        cast = self.get_story_cast(arc_id)
        if not cast:
            return ''
        block = self.codex_summary(cast)
        ev_block = self._canon_events_block(self.canon_events_for(arc_id))
        if ev_block:
            block = (block + '\n\n' if block else '') + ev_block
        return block

    def full_codex_block(self) -> str:
        """The WHOLE codex + every canon event — for script-level passes
        (web proposal, codex chat context) that legitimately survey
        everything, regardless of any story's cast."""
        block = self.codex_summary()
        ev_block = self._canon_events_block(self.canon_events())
        if ev_block:
            block = (block + '\n\n' if block else '') + ev_block
        return block

    # ── Canon events (v2 Phase 3) ────────────────────────────────────────────

    def canon_events(self) -> list:
        return self._data.setdefault('canon_events', [])

    def add_canon_event(self, summary: str, entities: list = None,
                        established_by: str = '') -> Optional[str]:
        """Append a canon event; returns its id (or None for empty summary)."""
        summary = (summary or '').strip()
        if not summary:
            return None
        evs = self.canon_events()
        taken = {e.get('id') for e in evs if isinstance(e, dict)}
        i = len(evs) + 1
        while f'ev_{i:03d}' in taken:
            i += 1
        ev = {
            'id': f'ev_{i:03d}',
            'summary': summary,
            'entities': [s for s in (entities or []) if s in self.entities],
            'established_by': established_by,
        }
        evs.append(ev)
        self.dirty = True
        return ev['id']

    def remove_canon_event(self, ev_id: str) -> bool:
        """Delete a canon event by id. Returns True if one was removed."""
        evs = self.canon_events()
        kept = [e for e in evs if not (isinstance(e, dict)
                                       and e.get('id') == ev_id)]
        if len(kept) == len(evs):
            return False
        self._data['canon_events'] = kept
        self.dirty = True
        return True

    def canon_events_for(self, arc_id: str) -> list:
        """Events relevant to a story: cast-entity overlap, entity-less
        (global) events, or events this story itself established. A story
        with NO cast sees none — same opt-in rule as the cards."""
        cast = set(self.get_story_cast(arc_id))
        if not cast:
            return []
        out = []
        for ev in self.canon_events():
            if not isinstance(ev, dict):
                continue
            ents = set(ev.get('entities', []))
            if not ents or (ents & cast) \
                    or ev.get('established_by') == arc_id:
                out.append(ev)
        return out

    # ── Story relations — the web (v2 Phase 3) ───────────────────────────────

    def get_story_relations(self, arc_id: str) -> list:
        """Valid outgoing relations of a story (targets must still exist)."""
        arc = self.get_arc(arc_id)
        rels = arc.get('relations') if arc else None
        if not isinstance(rels, list):
            return []
        return [r for r in rels
                if isinstance(r, dict) and r.get('to') in self.arcs
                and r.get('type') in RELATION_TYPES]

    def add_story_relation(self, from_id: str, to_id: str,
                           rtype: str, note: str = '') -> bool:
        """Add a relation FROM one story TO another. Dedupes on
        (to, type). Returns True if added."""
        if (from_id not in self.arcs or to_id not in self.arcs
                or from_id == to_id or rtype not in RELATION_TYPES):
            return False
        rels = self.arcs[from_id].setdefault('relations', [])
        for r in rels:
            if isinstance(r, dict) and r.get('to') == to_id \
                    and r.get('type') == rtype:
                return False
        rels.append({'to': to_id, 'type': rtype, 'note': (note or '').strip()})
        self.dirty = True
        return True

    def remove_story_relation(self, from_id: str, to_id: str, rtype: str):
        arc = self.arcs.get(from_id)
        if not arc:
            return
        rels = arc.get('relations', [])
        arc['relations'] = [r for r in rels
                            if not (isinstance(r, dict)
                                    and r.get('to') == to_id
                                    and r.get('type') == rtype)]
        self.dirty = True

    def all_story_relations(self) -> list:
        """Every relation in the web as (from_id, to_id, type, note)."""
        out = []
        for from_id in self.arcs:
            for r in self.get_story_relations(from_id):
                out.append((from_id, r['to'], r['type'], r.get('note', '')))
        return out

    # ── Node length range ──────────────────────────────────────────────────
    # Controls the per-node word-count target that gets injected into the
    # system prompt as a LENGTH OVERRIDE. Resolution order:
    #   1. arc.node_word_range (if the parent node belongs to an arc that
    #      overrides the default)
    #   2. script.node_word_range (story-wide default)
    #   3. The hardcoded fallback (40, 100) — original system-prompt spec
    DEFAULT_NODE_WORD_RANGE = (40, 100)

    @property
    def node_word_range(self) -> Optional[tuple]:
        """Script-wide default. None = use the hardcoded (40, 100)."""
        rng = self._data.get('node_word_range')
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            try:
                lo, hi = int(rng[0]), int(rng[1])
                if lo > 0 and hi >= lo:
                    return (lo, hi)
            except (TypeError, ValueError):
                pass
        return None

    def set_node_word_range(self, lo: Optional[int], hi: Optional[int]):
        """Set the script-wide range, or clear it (pass None for either)."""
        if lo is None or hi is None or lo <= 0 or hi < lo:
            self._data.pop('node_word_range', None)
        else:
            self._data['node_word_range'] = [int(lo), int(hi)]
        self.dirty = True

    @property
    def width_preset(self) -> str:
        """Generation width — one of 'small' / 'medium' / 'large'.
        Scales children-per-parent and per-layer caps in arc-driven
        generation. Defaults to 'medium' for any script that doesn't
        record a value."""
        val = self._data.get('width_preset')
        return val if val in WIDTH_PRESETS else DEFAULT_WIDTH_PRESET

    def set_width_preset(self, preset: str):
        if preset not in WIDTH_PRESETS:
            preset = DEFAULT_WIDTH_PRESET
        if self._data.get('width_preset') == preset:
            return
        self._data['width_preset'] = preset
        self.dirty = True

    def get_arc_width_preset(self, arc_id: str) -> Optional[str]:
        """Per-arc override of width preset. None = inherit from script."""
        arc = self.get_arc(arc_id)
        val = arc.get('width_preset') if arc else None
        return val if val in WIDTH_PRESETS else None

    def set_arc_width_preset(self, arc_id: str, preset: Optional[str]):
        """Set or clear a per-arc width override. None clears the override.
        No-op (not dirty) when nothing changes."""
        if arc_id not in self.arcs:
            return
        if preset is None or preset not in WIDTH_PRESETS:
            if 'width_preset' not in self.arcs[arc_id]:
                return
            self.arcs[arc_id].pop('width_preset', None)
        else:
            if self.arcs[arc_id].get('width_preset') == preset:
                return
            self.arcs[arc_id]['width_preset'] = preset
        self.dirty = True

    def get_effective_width_preset(self, arc_id: str) -> str:
        """Resolve the actual width preset to use for this arc:
        per-arc override if set, otherwise script-wide default."""
        return self.get_arc_width_preset(arc_id) or self.width_preset

    def get_arc_node_word_range(self, arc_id: str) -> Optional[tuple]:
        """Per-arc override. None = inherit from script-wide."""
        arc = self.get_arc(arc_id)
        rng = arc.get('node_word_range') if arc else None
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            try:
                lo, hi = int(rng[0]), int(rng[1])
                if lo > 0 and hi >= lo:
                    return (lo, hi)
            except (TypeError, ValueError):
                pass
        return None

    def set_arc_node_word_range(self, arc_id: str,
                                  lo: Optional[int], hi: Optional[int]):
        """Set or clear a per-arc override. No-op (not dirty) if unchanged."""
        if arc_id not in self.arcs:
            return
        if lo is None or hi is None or lo <= 0 or hi < lo:
            if 'node_word_range' not in self.arcs[arc_id]:
                return
            self.arcs[arc_id].pop('node_word_range', None)
        else:
            new = [int(lo), int(hi)]
            if self.arcs[arc_id].get('node_word_range') == new:
                return
            self.arcs[arc_id]['node_word_range'] = new
        self.dirty = True

    def get_effective_node_word_range(self, arc_id: str = '') -> tuple:
        """Final resolved (lo, hi) for a node generation. Tries
        per-arc → script-wide → DEFAULT_NODE_WORD_RANGE."""
        if arc_id:
            arc_rng = self.get_arc_node_word_range(arc_id)
            if arc_rng:
                return arc_rng
        script_rng = self.node_word_range
        if script_rng:
            return script_rng
        return self.DEFAULT_NODE_WORD_RANGE

    def summary(self, max_text: int = 80) -> str:
        """Compact text description of the script for AI context."""
        lines = [f'Script: "{self.name}"']
        if self.description:
            lines.append(f'Description: {self.description}')
        starts = self._data.get("start_nodes", [])
        if starts:
            lines.append(f'Start nodes: {", ".join(starts)}')
        lines.append(f'Nodes ({len(self._data["nodes"])}) — id [tags] "text..." → next:')
        for nid, nd in self._data["nodes"].items():
            tags = ", ".join(nd.get("tags", [])) or "—"
            text = nd.get("text", "").replace("\n", " ")
            if len(text) > max_text:
                text = text[:max_text] + "…"
            nexts = ", ".join(nd.get("next", [])) or "END"
            lines.append(f'  {nid} [{tags}] "{text}" → {nexts}')
        return "\n".join(lines)

    # ── Node operations ─────────────────────────────────────────────────────

    def add_node(self, node_id: str, text: str = "", pos=None) -> dict:
        node = {
            "text": text,
            "label": "",
            "hint": "",
            "file": None,
            "duration": None,
            "next": [],
            "weights": [],
            "tags": [],
            "voice": None,
            "voice_settings": {},
            "vars": {},
            "pos": pos or [100, 100],
        }
        self._data["nodes"][node_id] = node
        self.dirty = True
        return node

    def remove_node(self, node_id: str):
        self._data["nodes"].pop(node_id, None)
        if node_id in self._data["start_nodes"]:
            self._data["start_nodes"].remove(node_id)
        for node in self._data["nodes"].values():
            if node_id in node["next"]:
                idx = node["next"].index(node_id)
                node["next"].pop(idx)
                if idx < len(node.get("weights", [])):
                    node["weights"].pop(idx)
        self.dirty = True

    def rename_node(self, old_id: str, new_id: str) -> bool:
        if old_id not in self._data["nodes"] or new_id == old_id:
            return False
        if new_id in self._data["nodes"]:
            return False
        self._data["nodes"][new_id] = self._data["nodes"].pop(old_id)
        if old_id in self._data["start_nodes"]:
            idx = self._data["start_nodes"].index(old_id)
            self._data["start_nodes"][idx] = new_id
        for node in self._data["nodes"].values():
            node["next"] = [new_id if n == old_id else n for n in node["next"]]
        self.dirty = True
        return True

    def add_edge(self, from_id: str, to_id: str, weight: float = 1.0):
        node = self._data["nodes"].get(from_id)
        if node and to_id not in node["next"]:
            node["next"].append(to_id)
            node["weights"].append(weight)
            self.dirty = True

    def remove_edge(self, from_id: str, to_id: str):
        node = self._data["nodes"].get(from_id)
        if node and to_id in node["next"]:
            idx = node["next"].index(to_id)
            node["next"].pop(idx)
            if idx < len(node.get("weights", [])):
                node["weights"].pop(idx)
            self.dirty = True

    def nodes_that_cannot_end(self) -> set:
        """Node ids from which NO path reaches a terminal node.

        The player only finishes a walk on a node with an empty next[]
        (narrative_player._advance); recency penalties make loops WITH an
        exit escape quickly, but a pocket of nodes whose every path stays
        inside the pocket traps the walk forever — no ramp-down, no
        restart. Junction weaving can build such pockets if a story's
        ending gets deleted/rewired after it was woven into a cycle.

        Matches player semantics exactly: an edge to a nonexistent node id
        also ends playback (missing node → empty next), so only edges to
        EXISTING nodes count as continuations. Reverse-BFS from all
        terminals; complement = trapped set. O(nodes + edges)."""
        nodes = self._data.get('nodes', {})
        rev: dict = {nid: [] for nid in nodes}
        can_end = set()
        frontier = []
        for nid, nd in nodes.items():
            nxts = [n for n in nd.get('next', []) if n in nodes]
            if not nxts:
                can_end.add(nid)
                frontier.append(nid)
            for n in nxts:
                rev[n].append(nid)
        while frontier:
            cur = frontier.pop()
            for prv in rev[cur]:
                if prv not in can_end:
                    can_end.add(prv)
                    frontier.append(prv)
        return set(nodes) - can_end

    def update_text(self, node_id: str, text: str):
        if node_id in self._data["nodes"]:
            self._data["nodes"][node_id]["text"] = _sanitize_tts(text)
            self.dirty = True

    def update_tags(self, node_id: str, tags: list):
        if node_id in self._data["nodes"]:
            self._data["nodes"][node_id]["tags"] = tags
            self.dirty = True

    def update_hint(self, node_id: str, hint: str):
        if node_id in self._data["nodes"]:
            self._data["nodes"][node_id]["hint"] = hint
            self.dirty = True

    def update_label(self, node_id: str, label: str):
        if node_id in self._data["nodes"]:
            self._data["nodes"][node_id]["label"] = label
            self.dirty = True

    def set_start(self, node_id: str, is_start: bool):
        if is_start and node_id not in self._data["start_nodes"]:
            self._data["start_nodes"].append(node_id)
            self.dirty = True
        elif not is_start and node_id in self._data["start_nodes"]:
            self._data["start_nodes"].remove(node_id)
            self.dirty = True

    def update_pos(self, node_id: str, pos):
        if node_id in self._data["nodes"]:
            self._data["nodes"][node_id]["pos"] = list(pos)

    def set_default_voice(self, voice_id: str):
        self._data["voice"] = voice_id
        self.dirty = True

    def update_node_voice(self, node_id: str, voice_id: Optional[str]):
        if node_id in self._data["nodes"]:
            self._data["nodes"][node_id]["voice"] = voice_id
            self.dirty = True

    def update_node_voice_settings(self, node_id: str, settings: dict):
        if node_id in self._data["nodes"]:
            self._data["nodes"][node_id]["voice_settings"] = settings
            self.dirty = True

    # ── Serialisation ───────────────────────────────────────────────────────

    def save(self, path: Path = None):
        target = Path(path or self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        self.path = target
        self.dirty = False

    @property
    def associated_weather_set(self) -> Optional[str]:
        """The weather set this script is currently associated with.
        Reflects the explicit ``weather_set`` JSON field if present, or
        whatever reverse-lookup found, or None."""
        return self._associated_set

    @property
    def weather_set_explicit(self) -> Optional[str]:
        """The user-pinned weather set override from the script JSON.
        None means "auto-detect via reverse lookup"."""
        v = self._data.get('weather_set')
        return v if v else None

    def set_weather_set_explicit(self, set_name: Optional[str]):
        """Pin the script to a specific weather set (or pass None /
        empty string to clear the pin and fall back to auto-detect)."""
        if set_name:
            self._data['weather_set'] = set_name
        else:
            self._data.pop('weather_set', None)
        self.dirty = True
        self.refresh_weather_association()

    def available_weather_sets(self) -> list:
        """All weather sets known to this script's owning project.
        Used to populate the "Weather Set" selection dialog."""
        return sorted(_load_project_weather_sets(self.path).keys())

    @property
    def trigger_state_options(self) -> list:
        """List of weather-state value-strings the user can pick from for
        a node's trigger_state field. Empty if no weather set has been
        resolved for this script (in which case the editor will hide
        the dropdown)."""
        return list(self._trigger_state_options)

    def refresh_weather_association(self):
        """Re-resolve which weather set this script is associated with.
        Uses the explicit ``weather_set`` field if set, otherwise
        reverse-looks-up via narrative_script. Re-run after the user
        changes the explicit field or reloads weather_params.py."""
        self._associated_set, self._trigger_state_options = \
            _find_weather_states_for_script(self.path, self.weather_set_explicit)

    @classmethod
    def load(cls, path: Path) -> "ScriptData":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Migrate dual story_context / story_context_focused layout to
        # a single story_context field. If only the focused version is
        # set on an old script, promote it. If both are set, the longer
        # story_context wins (it's the "real" content; focused was the
        # truncated summary). Always drop the focused field so re-saved
        # scripts are clean.
        sc  = (data.get("story_context") or "").strip()
        scf = (data.get("story_context_focused") or "").strip()
        if not sc and scf:
            data["story_context"] = scf
        data.pop("story_context_focused", None)

        sd = cls(data)
        sd.path = Path(path)
        sd.refresh_weather_association()
        # Ensure all nodes have required fields; sanitize spoken text on load
        for node in sd._data["nodes"].values():
            node.setdefault("pos",      [100, 100])
            node.setdefault("next",     [])
            node.setdefault("weights",  [])
            node.setdefault("tags",     [])
            node.setdefault("file",     None)
            node.setdefault("duration", None)
            node.setdefault("voice",          None)
            node.setdefault("voice_settings", {})
            if node.get("text"):
                node["text"] = _sanitize_tts(node["text"])
        return sd

    @staticmethod
    def _sanitize_id(nid: str) -> str:
        return nid.replace('-', '_')

    @staticmethod
    def _sanitize_nodes(nodes: dict) -> dict:
        """Return a copy of `nodes` with IDs de-dashed and text TTS-sanitized."""
        clean = {}
        for nid, nd in nodes.items():
            if not isinstance(nd, dict):
                continue  # skip malformed entries
            safe_id = ScriptData._sanitize_id(nid)
            safe_nd = dict(nd)
            safe_nd['next'] = [ScriptData._sanitize_id(n) for n in nd.get('next', [])]
            if safe_nd.get('text'):
                safe_nd['text'] = _sanitize_tts(safe_nd['text'])
            clean[safe_id] = safe_nd
        return clean

    def _dedupe_ids(self, incoming: dict) -> dict:
        """Return a copy of `incoming` nodes with any IDs that collide with the
        existing script renamed (e.g. dev_kelp → dev_kelp_2, dev_kelp_3, …).
        All internal next-references are updated to match the new names."""
        taken = set(self._data["nodes"].keys())
        remap = {}
        for nid in list(incoming.keys()):
            new_id = nid
            if new_id in taken:
                counter = 2
                while f"{nid}_{counter}" in taken:
                    counter += 1
                new_id = f"{nid}_{counter}"
            remap[nid] = new_id
            taken.add(new_id)

        # Build renamed dict with updated next references
        result = {}
        for old_id, nd in incoming.items():
            new_id = remap[old_id]
            new_nd = dict(nd)
            new_nd['next'] = [remap.get(n, n) for n in nd.get('next', [])]
            result[new_id] = new_nd
        return result, remap

    def apply_generated(self, generated: dict):
        """Merge AI-generated graph into current script (additive).

        Positions nodes left-to-right by layer using tag hints:
        arrival < presence < curiosity < discovery < complication < intimacy < turn < consequence < echo < stillness
        Nodes within a layer are stacked vertically.
        """
        generated = dict(generated)
        generated['nodes'] = self._sanitize_nodes(generated.get('nodes', {}))
        generated['nodes'], remap = self._dedupe_ids(generated['nodes'])
        generated['start_nodes'] = [
            remap.get(self._sanitize_id(n), self._sanitize_id(n))
            for n in generated.get('start_nodes', [])
        ]
        if self._data["name"] == "New Script" and generated.get("name"):
            self._data["name"] = generated["name"]
        if generated.get("description"):
            self._data["description"] = generated["description"]

        LAYER_X     = {name: 80 + i * 310 for i, name in enumerate(LAYER_ORDER)}
        LAYER_X["_default"] = 80 + len(LAYER_ORDER) * 310

        # Count how many nodes already occupy each layer column (for vertical stacking)
        layer_counts: dict = {}

        new_nodes = generated.get("nodes", {})

        for nid, ndata in new_nodes.items():

            tags  = ndata.get("tags", [])
            layer = next((t for t in tags if t in LAYER_ORDER), "_default")
            x     = LAYER_X[layer]
            y_idx = layer_counts.get(layer, 0)
            layer_counts[layer] = y_idx + 1

            self._data["nodes"][nid] = {
                "text":           ndata.get("text", ""),
                "label":          "",
                "hint":           "",
                "file":           None,
                "duration":       None,
                "next":           ndata.get("next", []),
                "weights":        ndata.get("weights", [1.0] * len(ndata.get("next", []))),
                "tags":           tags,
                "voice":          ndata.get("voice", None),
                "voice_settings": ndata.get("voice_settings", {}),
                "vars":           ndata.get("vars", {}),
                "vars_reasoning": ndata.get("vars_reasoning", '').strip()
                                    if isinstance(ndata.get("vars_reasoning"), str)
                                    else '',
                "pos":            [x, 80 + y_idx * 170],
            }
            _clamp_voice_settings(self._data["nodes"][nid].get("voice_settings", {}))

        for nid in generated.get("start_nodes", []):
            if nid in self._data["nodes"] and nid not in self._data["start_nodes"]:
                self._data["start_nodes"].append(nid)

    def apply_expansion(self, source_id: str, expansion: dict):
        """Add expansion nodes and wire them from source_id."""
        expansion = dict(expansion)
        expansion['nodes'] = self._sanitize_nodes(expansion.get('nodes', {}))
        expansion['nodes'], remap = self._dedupe_ids(expansion['nodes'])
        expansion['connect_from'] = [
            remap.get(self._sanitize_id(n), self._sanitize_id(n))
            for n in expansion.get('connect_from', [])
        ]
        LAYER_X     = {name: 80 + i * 310 for i, name in enumerate(LAYER_ORDER)}
        LAYER_X["_default"] = 80 + len(LAYER_ORDER) * 310

        layer_counts: dict = {}
        for nd in self._data["nodes"].values():
            for tag in nd.get("tags", []):
                if tag in set(LAYER_ORDER):
                    layer_counts[tag] = layer_counts.get(tag, 0) + 1

        for nid, ndata in expansion.get("nodes", {}).items():
            tags  = ndata.get("tags", [])
            layer = next((t for t in tags if t in set(LAYER_ORDER)), "_default")
            x     = LAYER_X[layer]
            y_idx = layer_counts.get(layer, 0)
            layer_counts[layer] = y_idx + 1

            self._data["nodes"][nid] = {
                "text":           ndata.get("text", ""),
                "label":          "",
                "hint":           "",
                "file":           None,
                "duration":       None,
                "next":           ndata.get("next", []),
                "weights":        ndata.get("weights", [1.0] * len(ndata.get("next", []))),
                "tags":           tags,
                "voice":          ndata.get("voice", None),
                "voice_settings": ndata.get("voice_settings", {}),
                "vars":           ndata.get("vars", {}),
                "pos":            [x, 80 + y_idx * 170],
            }
            _clamp_voice_settings(self._data["nodes"][nid].get("voice_settings", {}))

        # Wire source node to the connect_from targets
        src = self._data["nodes"].get(source_id)
        if src:
            for nid in expansion.get("connect_from", []):
                if nid in self._data["nodes"] and nid not in src["next"]:
                    src["next"].append(nid)
                    src["weights"].append(1.0)

        self.dirty = True

    def apply_single_node(self, parent_id: str, node_data: dict) -> Optional[str]:
        """Add one AI-generated node to the script, wired from parent_id.

        Returns the final (possibly deduped) node_id, or None on failure.
        """
        node_id = node_data.get('node_id', '')
        if not node_id or not isinstance(node_data, dict):
            return None

        # Sanitize
        single = {node_id: {
            'text':           _sanitize_tts(node_data.get('text', '')),
            'next':           [],
            'weights':        [],
            'tags':           node_data.get('tags', []),
            'voice':          None,
            'voice_settings': node_data.get('voice_settings', {}),
            'vars':           node_data.get('vars', {}),
            'vars_reasoning': node_data.get('vars_reasoning', '').strip()
                                if isinstance(node_data.get('vars_reasoning'), str)
                                else '',
            'duration':       None,
        }}
        # Clamp voice settings from AI to safe ranges
        vs = single[node_id].get('voice_settings', {})
        if vs:
            _clamp_voice_settings(vs)
        single = self._sanitize_nodes(single)
        single, _remap = self._dedupe_ids(single)
        final_id = list(single.keys())[0] if single else None
        if not final_id:
            return None

        # Position new node to the right of its parent
        if parent_id and parent_id in self._data['nodes']:
            parent_nd = self._data['nodes'][parent_id]
            pp = parent_nd.get('pos', [100, 100])
            sibling_count = len(parent_nd.get('next', []))
            single[final_id]['pos'] = [pp[0] + 300, pp[1] + sibling_count * 120]

        # Add node
        self._data['nodes'][final_id] = single[final_id]

        # Wire parent → child
        if parent_id and parent_id in self._data['nodes']:
            parent = self._data['nodes'][parent_id]
            if final_id not in parent.get('next', []):
                parent.setdefault('next', []).append(final_id)
                parent.setdefault('weights', []).append(1.0)

        self.dirty = True
        return final_id

    def apply_layer(self, layer_data: dict):
        """Apply a batch layer expansion where each new node declares its connect_from sources."""
        layer_data = dict(layer_data)
        raw_nodes = layer_data.get('nodes', {})
        if not isinstance(raw_nodes, dict):
            return  # malformed — nothing to apply
        # Drop any node entries that aren't dicts
        raw_nodes = {k: v for k, v in raw_nodes.items() if isinstance(v, dict)}
        layer_data['nodes'] = self._sanitize_nodes(raw_nodes)
        layer_data['nodes'], remap = self._dedupe_ids(layer_data['nodes'])

        LAYER_X     = {name: 80 + i * 310 for i, name in enumerate(LAYER_ORDER)}
        LAYER_X["_default"] = 80 + len(LAYER_ORDER) * 310
        layer_counts: dict = {}
        for nd in self._data["nodes"].values():
            for tag in nd.get("tags", []):
                if tag in set(LAYER_ORDER):
                    layer_counts[tag] = layer_counts.get(tag, 0) + 1

        for nid, ndata in layer_data['nodes'].items():
            tags  = ndata.get("tags", [])
            layer = next((t for t in tags if t in set(LAYER_ORDER)), "_default")
            x     = LAYER_X[layer]
            y_idx = layer_counts.get(layer, 0)
            layer_counts[layer] = y_idx + 1

            self._data["nodes"][nid] = {
                "text":           ndata.get("text", ""),
                "label":          "",
                "hint":           "",
                "file":           None,
                "duration":       None,
                "next":           ndata.get("next", []),
                "weights":        ndata.get("weights", [1.0] * len(ndata.get("next", []))),
                "tags":           tags,
                "voice":          ndata.get("voice", None),
                "voice_settings": ndata.get("voice_settings", {}),
                "vars":           ndata.get("vars", {}),
                "pos":            [x, 80 + y_idx * 170],
            }

            # Wire each declared source node to point to this new node
            for src_id in ndata.get("connect_from", []):
                src = self._data["nodes"].get(src_id)
                if src and nid not in src["next"]:
                    src["next"].append(nid)
                    src["weights"].append(1.0)

        self.dirty = True


# ─────────────────────────────────────────────────────────────────────────────
# Parallel Node Generation
# ─────────────────────────────────────────────────────────────────────────────

class StoryStructure:
    """Resolved beat sequence for one story's generation run.

    Wraps the list from ScriptData.get_story_structure(): an ordered list
    of beats, each an archetype from LAYER_ORDER (which supplies the layer
    tag, LAYER_FUNCTIONS text, premise role, voice defaults, and width/cap
    profile defaults) plus this story's per-beat direction. Beats may
    repeat archetypes and come in any order — advancement is by beat INDEX,
    never by looking an archetype back up in LAYER_ORDER.
    """

    def __init__(self, beats: list):
        self._beats = beats or ScriptData.default_structure()

    @property
    def n_beats(self) -> int:
        return len(self._beats)

    def _entry(self, i: int) -> dict:
        return self._beats[max(0, min(i, len(self._beats) - 1))]

    def archetype(self, i: int) -> str:
        return self._entry(i)['layer']

    def direction(self, i: int) -> str:
        return self._entry(i).get('direction', '')

    def width(self, i: int, profile: dict) -> tuple:
        widths = profile.get('widths', {})
        return widths.get(self.archetype(i), widths.get('*', (2, 3)))

    def cap(self, i: int, profile: dict) -> int:
        caps = profile.get('layer_caps', {})
        return caps.get(self.archetype(i), 999)

    def beat_index_for_node(self, nd: dict) -> int:
        """Best-effort beat index of an existing node: the explicit 'beat'
        field v2 generation stamps, else the first beat whose archetype
        appears in the node's tags, else 0. (v1-generated nodes have no
        'beat' field — the tag fallback covers them.)"""
        b = nd.get('beat')
        if isinstance(b, int) and 0 <= b < len(self._beats):
            return b
        tags = set(nd.get('tags', []))
        for i, e in enumerate(self._beats):
            if e['layer'] in tags:
                return i
        return 0


@dataclass
class NodeTask:
    """One pending/active/complete batch generation task.

    Each task generates batch_size sibling nodes from a single parent
    in one AI call.
    """
    task_id:         str
    parent_id:       str                  # primary node to continue from
    parent_ids:      list  = field(default_factory=list)  # ALL parents (for convergence)
    root_id:         str   = ''           # seed node this branch descends from
    layer_name:      str   = ''           # target layer ARCHETYPE for this node
    layer_direction: str   = ''           # story beat guidance
    beat_idx:        int   = 0            # index into the story's structure
    arc_id:          str   = ''           # story (arc) this task belongs to
    batch_size:      int   = 1            # how many siblings to generate in one call
    status:          str   = 'pending'    # pending | dispatched | complete | failed
    result:          dict  = field(default_factory=dict)
    final_node_ids:  list  = field(default_factory=list)  # actual IDs after dedup


class ParallelNodeOrchestrator:
    """Generates a narrative graph in parallel, one node per AI call.

    Each worker (AIAssistant instance) generates exactly one node at a time.
    As nodes complete, their children are dispatched immediately.  Siblings
    at the same layer run concurrently on different workers.
    """

    def __init__(self, script: 'ScriptData', ui_queue: queue.SimpleQueue,
                 model: str = '', profile: str = 'full',
                 story_context: str = '', variables: list = None,
                 on_progress=None, on_complete=None, on_node_added=None,
                 thinking: str = '',
                 width_preset: str = DEFAULT_WIDTH_PRESET):
        self._script       = script
        self._ui_queue     = ui_queue
        # `profile` is a name from GENERATION_PROFILES, or a ready-made
        # profile dict (the expand flow passes a custom width override).
        if isinstance(profile, dict):
            base_profile = profile
        else:
            base_profile = GENERATION_PROFILES.get(profile,
                                                   GENERATION_PROFILES['full'])
        # Apply the width preset's scale factor to children-per-parent and
        # per-layer caps. 'medium' is a no-op; 'small'/'large' shrink/grow.
        self._profile      = _scale_profile(base_profile, width_preset)
        self._story_context = story_context
        self._variables    = variables or []
        self._on_progress  = on_progress   # callback(status_str)
        self._on_complete  = on_complete   # callback()
        self._on_node_added = on_node_added  # callback(set_of_new_ids)

        # One AIAssistant serves every worker thread: its sync calls are
        # stateless subprocess invocations, so concurrency comes from the
        # executor, not from instance count. (This used to be a pool of
        # PARALLEL_WORKER_COUNT instances of which only [0] was ever used.)
        self._worker = AIAssistant(model=model, thinking=thinking)
        self._worker_sem = threading.Semaphore(PARALLEL_WORKER_COUNT)
        self._executor = ThreadPoolExecutor(max_workers=PARALLEL_WORKER_COUNT)

        # Task tracking
        self._lock = threading.Lock()
        self._tasks: dict = {}          # task_id -> NodeTask
        self._completed_by_layer: dict = defaultdict(list)  # layer -> [NodeTask]
        self._sibling_summaries: dict = defaultdict(list)   # parent_id -> [(node_id, summary)]
        self._node_to_root: dict = {}   # node_id -> root_id (branch lineage)
        self._deferred_convergences: list = []  # [(parent_id, child_layer, root_id), ...]
        self._task_counter = 0
        self._active_count = 0
        self._total_created = 0
        self._total_completed = 0
        self._cancelled = threading.Event()
        # Prompt-cache priming flag. The first AI call writes ~11K tokens of
        # bible into Anthropic's prompt cache (1.25x rate). Subsequent calls
        # read it back at 0.1x. If we let all 8 workers fire at once, all 8
        # see a cache MISS and pay the 1.25x write rate — a 5-8x cost
        # multiplier on the initial burst. So we serialize the very first
        # call until it returns, then allow parallel dispatch.
        self._cache_primed = False

        # Existing tags for reuse hints
        layer_tags = set(LAYER_ORDER)
        self._existing_tags = list({
            t for n in script.nodes.values()
            for t in n.get('tags', []) if t not in layer_tags
        })

        # Per-story resolved structures + cast-codex blocks, cached per arc
        # id. The cast codex MUST stay byte-stable across the whole run so
        # it lives in the prompt-cache-friendly system prefix like the bible.
        self._structures: dict = {}
        self._cast_codices: dict = {}

    def _structure_for_arc(self, arc_id: str) -> StoryStructure:
        if arc_id not in self._structures:
            self._structures[arc_id] = StoryStructure(
                self._script.get_story_structure(arc_id))
        return self._structures[arc_id]

    def _cast_codex_for_arc(self, arc_id: str) -> str:
        if arc_id not in self._cast_codices:
            self._cast_codices[arc_id] = self._script.cast_codex_for(arc_id)
        return self._cast_codices[arc_id]

    def _story_fields(self, arc_id: str) -> tuple:
        """(premise, motif, themes) for a story; empty strings if it's gone.
        Beat direction is NOT here — it comes from the story structure by
        beat index (see StoryStructure)."""
        arc = self._script.get_arc(arc_id)
        return (arc.get('premise', ''), arc.get('motif', ''),
                arc.get('themes', ''))

    def cancel(self):
        """Signal cancellation. In-flight tasks finish but results are discarded."""
        self._cancelled.set()

    @property
    def running(self) -> bool:
        return self._active_count > 0

    def start_merged(self, parent_ids: list, batch_size: int = 3):
        """Generate children that share ALL parent_ids as parents (merge operation).

        The children's layer is determined by the deepest parent's next layer.
        """
        valid = [nid for nid in parent_ids if nid in self._script.nodes]
        if not valid:
            return
        print(f'[Parallel] Merged start from {len(valid)} parents: {valid}')

        # Determine the child beat from the deepest parent (first parent's
        # story structure governs the merge).
        arc_id = self._script.get_node_arc_id(valid[0])
        structure = self._structure_for_arc(arc_id)
        deepest = 0
        for nid in valid:
            deepest = max(deepest,
                          structure.beat_index_for_node(self._script.nodes[nid]))

        if deepest + 1 >= structure.n_beats:
            print(f'[Parallel] All parents at the final beat — nothing to generate')
            self._ui_queue.put(lambda: self._on_complete() if self._on_complete else None)
            return
        child_beat = deepest + 1
        child_layer = structure.archetype(child_beat)

        # Use first parent as root for branch tracking
        root_id = valid[0]
        for nid in valid:
            self._node_to_root[nid] = root_id

        direction = structure.direction(child_beat)
        tid = self._next_task_id()
        task = NodeTask(
            task_id=tid,
            parent_id=valid[0],
            parent_ids=list(valid),
            root_id=root_id,
            layer_name=child_layer,
            layer_direction=direction,
            beat_idx=child_beat,
            arc_id=arc_id,
            batch_size=batch_size,
        )
        self._tasks[tid] = task
        self._total_created += batch_size
        print(f'[Parallel] Merged batch: {batch_size}× beat {child_beat} '
              f'({child_layer}) from parents {valid}')

        threading.Thread(target=self._coordinator_loop, daemon=True).start()

    def start(self, seed_node_ids: list):
        """Begin parallel generation from a list of existing seed (arrival) nodes."""
        print(f'[Parallel] Starting with {len(seed_node_ids)} seed nodes: {seed_node_ids}')
        print(f'[Parallel] Profile: max_depth={self._profile["max_depth"]}, workers={PARALLEL_WORKER_COUNT}')
        for nid in seed_node_ids:
            nd = self._script.nodes.get(nid)
            if not nd:
                print(f'[Parallel]   {nid}: NOT FOUND in script — skipping')
                continue
            # Each seed is the root of its own branch
            self._node_to_root[nid] = nid
            arc_id = self._script.get_node_arc_id(nid)
            structure = self._structure_for_arc(arc_id)
            beat = structure.beat_index_for_node(nd)
            print(f'[Parallel]   {nid}: story={arc_id or "-"}, beat={beat} '
                  f'({structure.archetype(beat)}), branch={nid}')
            if beat + 1 >= structure.n_beats:
                print(f'[Parallel]   {nid}: already at the final beat — no children')
                continue
            self._spawn_children(nid, beat + 1, root_id=nid)

        print(f'[Parallel] Initial tasks queued: {self._total_created}')
        # Start coordinator thread
        threading.Thread(target=self._coordinator_loop, daemon=True).start()

    def _next_task_id(self) -> str:
        self._task_counter += 1
        return f'task_{self._task_counter:04d}'

    def _count_beat_nodes(self, arc_id: str, beat_idx: int) -> int:
        """Count expected nodes at a story beat across ALL branches (sum of
        batch sizes). Keyed by (story, beat index) so repeated archetypes in
        one structure each get their own budget."""
        return sum(t.batch_size for t in self._tasks.values()
                   if t.arc_id == arc_id and t.beat_idx == beat_idx)

    def _spawn_children(self, parent_id: str, child_beat: int, root_id: str = ''):
        """Create NodeTasks for children of parent_id at the story's beat
        index `child_beat`.

        Uses GLOBAL caps to limit total node count per beat.  When over cap,
        converges into existing same-branch nodes (no forced child creation).
        Cross-branch connections are handled by the cross-link AI pass.
        """
        if not root_id:
            root_id = self._node_to_root.get(parent_id, parent_id)

        arc_id = self._script.get_node_arc_id(parent_id)
        structure = self._structure_for_arc(arc_id)
        child_layer = structure.archetype(child_beat)
        lo, hi = structure.width(child_beat, self._profile)
        desired = random.randint(lo, hi)
        direction = structure.direction(child_beat)

        with self._lock:
            cap = structure.cap(child_beat, self._profile)
            global_count = self._count_beat_nodes(arc_id, child_beat)
            remaining_slots = max(0, cap - global_count)

            n_to_create = min(desired, remaining_slots)

            # Over global cap — converge into existing nodes at this beat
            if n_to_create == 0:
                # Queue a deferred convergence — will be resolved after all
                # tasks at this beat complete
                self._deferred_convergences.append(
                    (parent_id, arc_id, child_beat, root_id))
                print(f'[Parallel] ↗ Deferred converge: {parent_id} → beat {child_beat} '
                      f'({child_layer}) ({global_count}/{cap})')
                return  # hard stop — no new nodes when over cap

            print(f'[Parallel] Batch: {n_to_create} children for {parent_id} → '
                  f'beat:{child_beat} ({child_layer}) branch:{root_id} '
                  f'({global_count}+{n_to_create}/{cap})')

            tid = self._next_task_id()
            task = NodeTask(
                task_id=tid,
                parent_id=parent_id,
                parent_ids=[parent_id],
                root_id=root_id,
                layer_name=child_layer,
                layer_direction=direction,
                beat_idx=child_beat,
                arc_id=arc_id,
                batch_size=n_to_create,
            )
            self._tasks[tid] = task
            self._total_created += n_to_create  # count expected nodes, not tasks

    def _coordinator_loop(self):
        """Dispatch ready tasks to workers until done or cancelled."""
        print('[Parallel] Coordinator loop started')
        while not self._cancelled.is_set():
            with self._lock:
                pending = [t for t in self._tasks.values() if t.status == 'pending']
                if not pending and self._active_count == 0:
                    break  # all done

            if pending:
                print(f'[Parallel] Dispatching {len(pending)} pending tasks '
                      f'(active: {self._active_count}, total: {self._total_completed}/{self._total_created})')

            for task in pending:
                if self._cancelled.is_set():
                    break
                task.status = 'dispatched'
                with self._lock:
                    self._active_count += 1
                self._worker_sem.acquire()
                if self._cancelled.is_set():
                    self._worker_sem.release()
                    break
                print(f'[Parallel]   → Dispatch {task.task_id}: parent={task.parent_id} layer={task.layer_name}')
                self._executor.submit(self._execute_task, task)

                # Prompt-cache priming: hold off on parallel dispatch until
                # this first call returns. After it completes, Anthropic's
                # prompt cache holds the bible and subsequent calls read it
                # at 0.1x instead of racing to write at 1.25x. Adds ~one-call
                # wall time to the run; saves ~5-8x on the initial burst.
                if not self._cache_primed:
                    print(f'[Parallel] Priming prompt cache (waiting for {task.task_id} to return)...')
                    while not self._cancelled.is_set():
                        with self._lock:
                            if self._active_count == 0:
                                break
                        time.sleep(0.1)
                    self._cache_primed = True
                    print('[Parallel] Cache primed; parallel dispatch enabled.')
                    break  # re-scan pending — the first task may have spawned children

            # Brief sleep to avoid busy-waiting for new tasks from completions
            time.sleep(0.1)

        # Final: resolve deferred convergences, then cross-link passes
        if not self._cancelled.is_set():
            print(f'[Parallel] All tasks done ({self._total_completed} nodes).')
            self._resolve_deferred_convergences()
            print(f'[Parallel] Running cross-link passes...')
            self._run_cross_link_passes()

        print(f'[Parallel] Generation complete. {self._total_completed} nodes generated.')
        # Release the worker threads — otherwise each finished run leaves
        # PARALLEL_WORKER_COUNT idle threads behind for the process lifetime.
        self._executor.shutdown(wait=False)
        self._ui_queue.put(lambda: self._on_complete() if self._on_complete else None)

    def _resolve_deferred_convergences(self):
        """Wire deferred convergences now that all tasks are complete."""
        if not self._deferred_convergences:
            return
        print(f'[Parallel] Resolving {len(self._deferred_convergences)} deferred convergences...')
        for parent_id, arc_id, child_beat, root_id in self._deferred_convergences:
            # Find completed nodes at this story beat, prefer same branch
            candidates = []
            for t in self._tasks.values():
                if t.arc_id == arc_id and t.beat_idx == child_beat and t.final_node_ids:
                    if t.root_id == root_id:
                        candidates.extend(t.final_node_ids)
            if not candidates:
                for t in self._tasks.values():
                    if t.arc_id == arc_id and t.beat_idx == child_beat and t.final_node_ids:
                        candidates.extend(t.final_node_ids)
            if candidates:
                targets = random.sample(candidates, min(2, len(candidates)))
                for cid in targets:
                    def _wire(pid=parent_id, c=cid):
                        if pid in self._script.nodes and c in self._script.nodes:
                            src = self._script.nodes[pid]
                            if c not in src.get('next', []):
                                src.setdefault('next', []).append(c)
                                src.setdefault('weights', []).append(1.0)
                                self._script.dirty = True
                    self._ui_queue.put(_wire)
                print(f'[Parallel]   ↗ {parent_id} → {targets}')
            else:
                print(f'[Parallel]   ⚠ {parent_id}: no nodes at beat {child_beat} to converge into')
        self._deferred_convergences.clear()

    def _get_ancestor_chain(self, node_id: str, depth: int = 4) -> list:
        """Walk parents to build [(ancestor_id, text, tags), ...] oldest-first."""
        nodes = self._script.nodes
        # Build reverse map
        reverse = {}
        for nid, nd in nodes.items():
            for child in nd.get('next', []):
                reverse.setdefault(child, []).append(nid)
        chain = []
        current = node_id
        for _ in range(depth):
            parents = reverse.get(current, [])
            if not parents:
                break
            current = parents[0]
            nd = nodes.get(current, {})
            chain.append((current, nd.get('text', ''), nd.get('tags', [])))
        chain.reverse()
        return chain

    def _execute_task(self, task: NodeTask):
        """Run a batch node generation on a worker thread."""
        try:
            if self._cancelled.is_set():
                return

            # Retry briefly if parent hasn't been applied to script yet
            all_parent_ids = task.parent_ids if task.parent_ids else [task.parent_id]
            for _retry in range(10):
                valid_parents = [pid for pid in all_parent_ids
                                 if pid in self._script.nodes
                                 and self._script.nodes[pid].get('text')]
                if valid_parents:
                    break
                time.sleep(0.5)
            all_parent_ids = valid_parents if valid_parents else []

            if not all_parent_ids:
                print(f'[Parallel] ⚠ {task.task_id}: no valid parents after retries '
                      f'(wanted: {task.parent_ids or [task.parent_id]}) — skipping')
                task.status = 'failed'
                return

            primary_pid = all_parent_ids[0]
            primary_nd = self._script.nodes[primary_pid]
            ancestor_chain = self._get_ancestor_chain(primary_pid)

            if len(all_parent_ids) > 1:
                parent_texts = []
                all_tags = []
                for pid in all_parent_ids:
                    nd = self._script.nodes.get(pid, {})
                    parent_texts.append(f'[{pid}]: "{nd.get("text", "")}"')
                    all_tags.extend(nd.get('tags', []))
                parent_text = (
                    f'Merging {len(all_parent_ids)} branches — write nodes that '
                    f'naturally continue from ANY of these:\n'
                    + '\n'.join(parent_texts)
                )
                parent_tags = list(dict.fromkeys(all_tags))
            else:
                parent_text = primary_nd.get('text', '')
                parent_tags = primary_nd.get('tags', [])

            # Premise weight per layer — comes from LAYER_PREMISE_ROLES so the
            # premise stays visible across ALL beats (peaks at arrival, complication,
            # turn; tapers at stillness). The label text from the same dict tells
            # the generator what ROLE the premise plays at this specific beat.
            premise_weight, _ = LAYER_PREMISE_ROLES.get(task.layer_name, (0.0, ''))

            # Read the parent node's hint (author guidance for expansion)
            parent_hint = primary_nd.get('hint', '').strip()

            parent_label = (f'parents=[{", ".join(all_parent_ids)}]' if len(all_parent_ids) > 1
                           else f'parent={primary_pid}')
            premise_str = f', premise={premise_weight:.0%}' if premise_weight > 0 else ''
            hint_str = f', hint="{parent_hint[:40]}..."' if parent_hint else ''
            print(f'[Parallel] ▶ {task.task_id}: batch {task.batch_size}× {task.layer_name} '
                  f'from {parent_label} (ancestors={len(ancestor_chain)}{premise_str}{hint_str})')

            premise, motif, themes = self._story_fields(task.arc_id)
            cast_codex = self._cast_codex_for_arc(task.arc_id)

            # Resolve the node word-count range from the task's story;
            # falls through to script default then to (40, 100).
            node_word_range = self._script.get_effective_node_word_range(task.arc_id)

            result = self._worker.generate_batch_sync(
                parent_id=primary_pid,
                parent_text=parent_text,
                parent_tags=parent_tags,
                ancestor_chain=ancestor_chain,
                layer_name=task.layer_name,
                batch_size=task.batch_size,
                layer_direction=task.layer_direction,
                hint=parent_hint,
                motif=motif,
                themes=themes,
                story_context=self._story_context,
                existing_custom_tags=self._existing_tags,
                variables=self._variables,
                premise=premise,
                premise_weight=premise_weight,
                node_word_range=node_word_range,
                cast_codex=cast_codex,
            )

            if self._cancelled.is_set():
                return

            # Result is in SYSTEM_EXPAND format: {"nodes": {...}, "connect_from": [...]}
            nodes = result.get('nodes', {})
            if not isinstance(nodes, dict):
                nodes = {}
            # Filter out malformed entries (AI sometimes returns strings instead of dicts)
            nodes = {nid: nd for nid, nd in nodes.items() if isinstance(nd, dict)}

            n_got = len(nodes)
            print(f'[Parallel] ✓ {task.task_id}: got {n_got} nodes')
            for nid, nd in nodes.items():
                text_preview = ' '.join(nd.get('text', '').split()[:12]) + '...'
                print(f'[Parallel]   {nid}: {text_preview}')

            task.result = result
            task.status = 'complete'

            # Apply all nodes to script on main thread
            applied_event = threading.Event()

            def _apply(t=task, evt=applied_event, pids=all_parent_ids,
                       node_dict=nodes, inherit_arc=task.arc_id):
                applied_ids = []
                for nid, ndata in node_dict.items():
                    # Build a single-node result dict for apply_single_node
                    single = dict(ndata)
                    single['node_id'] = nid
                    final_id = self._script.apply_single_node(t.parent_id, single)
                    if final_id:
                        applied_ids.append(final_id)
                        # Stamp the exact beat index so advancement stays
                        # unambiguous even when a structure repeats archetypes.
                        self._script.nodes[final_id]['beat'] = t.beat_idx
                        if inherit_arc:
                            self._script.set_node_arc_id(final_id, inherit_arc)
                        # Wire additional convergence parents
                        for pid in pids[1:]:
                            if pid in self._script.nodes:
                                src = self._script.nodes[pid]
                                if final_id not in src.get('next', []):
                                    src.setdefault('next', []).append(final_id)
                                    src.setdefault('weights', []).append(1.0)
                        for tag in ndata.get('tags', []):
                            if tag not in set(LAYER_ORDER) and tag not in self._existing_tags:
                                self._existing_tags.append(tag)
                t.final_node_ids = applied_ids
                if applied_ids and self._on_node_added:
                    self._on_node_added(set(applied_ids))
                evt.set()

            self._ui_queue.put(_apply)

            if not applied_event.wait(timeout=30.0):
                print(f'[Parallel] ⚠ {task.task_id}: apply timed out — skipping children')
                task.status = 'failed'
                return

            # Record results and spawn children for each generated node
            root_id = task.root_id
            with self._lock:
                self._total_completed += len(task.final_node_ids)
                self._completed_by_layer[(task.arc_id, task.beat_idx)].append(task)
                for fid in task.final_node_ids:
                    self._node_to_root[fid] = root_id

            # Advancement is by beat INDEX in the story structure, not by
            # re-deriving a layer from the AI's tags — predictable, and it
            # supports structures that repeat archetypes. max_depth keeps
            # its historical meaning (absolute beat-position ceiling; the
            # expand profile's 2 ≈ one layer of children).
            structure = self._structure_for_arc(task.arc_id)
            max_depth_layers = self._profile['max_depth']
            next_beat = task.beat_idx + 1
            for final_id in task.final_node_ids:
                if next_beat < structure.n_beats and next_beat < max_depth_layers:
                    self._spawn_children(final_id, next_beat, root_id=root_id)
                else:
                    print(f'[Parallel]   {final_id} is terminal '
                          f'(beat={task.beat_idx}, {task.layer_name})')

            # Progress
            msg = (f"Parallel: {self._total_completed}/{self._total_created} nodes "
                   f"({self._active_count} active, layer: {task.layer_name})")
            print(f'[Parallel] {msg}')
            if self._on_progress:
                self._ui_queue.put(lambda m=msg: self._on_progress(m))

        except Exception as exc:
            task.status = 'failed'
            print(f'[Parallel] ✗ {task.task_id} FAILED: {exc}')
            import traceback
            traceback.print_exc()
            if self._on_progress:
                self._ui_queue.put(
                    lambda e=str(exc)[:80]: self._on_progress(f"Node error: {e}"))
        finally:
            with self._lock:
                self._active_count -= 1
            self._worker_sem.release()

    def _run_cross_link_passes(self):
        """After all generation is done, suggest cross-branch connections.

        Beat groups are independent, so their AI calls run concurrently on
        the (now otherwise idle) worker pool instead of serially — on a
        multi-story or wide run this was the slowest tail of generation."""
        if not self._completed_by_layer:
            return

        # Build all jobs on this thread first (reads script state), then
        # fan the AI calls out.
        jobs = []
        for (arc_id, beat_idx), completed in sorted(self._completed_by_layer.items()):
            structure = self._structure_for_arc(arc_id)
            if beat_idx >= structure.n_beats - 1:
                continue  # skip each story's final beat
            layer_nodes = []
            children_map = {}
            for task in completed:
                for nid in task.final_node_ids:
                    nd = self._script.nodes.get(nid, {})
                    if not nd:
                        continue
                    child_ids = nd.get('next', [])
                    layer_nodes.append((nid, nd.get('text', ''),
                                        nd.get('tags', []), child_ids))
                    for cid in child_ids:
                        cnd = self._script.nodes.get(cid, {})
                        children_map[cid] = (cnd.get('text', ''),
                                             cnd.get('tags', []))
            if len(layer_nodes) < 5 or not children_map:
                continue  # not enough nodes to be worth a cross-link call
            label = f'{structure.archetype(beat_idx)}@{beat_idx}'
            print(f'[Parallel] Cross-linking story {arc_id or "-"} {label} '
                  f'({len(layer_nodes)} nodes)...')
            jobs.append((label, layer_nodes, children_map))

        futures = [self._executor.submit(self._cross_link_one, *job)
                   for job in jobs]
        for f in futures:
            try:
                f.result()
            except Exception:
                pass  # cross-linking is best-effort

    def _cross_link_one(self, label: str, layer_nodes: list,
                        children_map: dict):
        """One cross-link AI call + edge application (worker thread)."""
        if self._cancelled.is_set():
            return
        links = self._worker.suggest_cross_links_sync(layer_nodes, children_map)
        print(f'[Parallel]   Cross-link suggestions for {label}: {len(links)} links')
        for link in links:
            print(f'[Parallel]     {link.get("from", "?")} → {link.get("to", "?")}')
        if not links or self._cancelled.is_set():
            return

        def _apply_links(cross_links=links):
            for link in cross_links:
                from_id = link.get('from', '')
                to_id = link.get('to', '')
                if (from_id in self._script.nodes
                        and to_id in self._script.nodes):
                    src = self._script.nodes[from_id]
                    if to_id not in src.get('next', []):
                        src.setdefault('next', []).append(to_id)
                        src.setdefault('weights', []).append(1.0)
                        self._script.dirty = True
        self._ui_queue.put(_apply_links)


def _seed_count_for_story(script: 'ScriptData', arc_id: str) -> int:
    """How many opening (beat-0) nodes to seed for a story: the scaled
    generation profile's cap for the first beat's archetype, bounded 1..4.
    Keeps a 'small' width preset from over-seeding (the old flow always
    seeded 4 regardless of preset, so a small story started over cap)."""
    structure = StoryStructure(script.get_story_structure(arc_id))
    profile = _scale_profile(GENERATION_PROFILES['full'],
                             script.get_effective_width_preset(arc_id))
    return max(1, min(4, structure.cap(0, profile)))


def make_full_orchestrator(script: 'ScriptData', ai: 'AIAssistant',
                           ui_queue: queue.SimpleQueue, *,
                           story_context: str, width_preset: str,
                           on_progress, on_complete,
                           on_node_added=None, register=None):
    """The one construction path for full-profile generation runs (AI chat
    panel and the Stories dialog). Guarantees the user's model AND thinking
    selection travel with the run — the chat panel used to drop thinking —
    and registers the run for Story ▸ Stop AI Generation."""
    orch = ParallelNodeOrchestrator(
        script=script,
        ui_queue=ui_queue,
        model=ai.model,
        thinking=ai.thinking,
        profile='full',
        width_preset=width_preset,
        story_context=story_context,
        variables=script.variables,
        on_progress=on_progress,
        on_complete=on_complete,
        on_node_added=on_node_added,
    )
    if register:
        register(orch)
    return orch


def _fan_out_ai_calls(jobs: list, call, on_progress=None,
                      max_workers: int = PARALLEL_WORKER_COUNT,
                      cancel_event=None) -> list:
    """Run `call(job)` for every job CONCURRENTLY on a transient thread
    pool; returns results in job order (None where a job raised).

    This is the engine behind the dialog AI passes (weave junctions/
    callbacks, seam + consistency audits), which used to run their
    independent calls sequentially — wall clock is now roughly the slowest
    single call instead of the sum. `on_progress(done, total)` fires as
    jobs complete (from worker threads — marshal to the UI queue yourself).

    cancel_event: optional threading.Event — once set, jobs that haven't
    started yet return None immediately; in-flight calls still run to
    completion (bounded by the per-call timeout).
    """
    if not jobs:
        return []
    results = [None] * len(jobs)
    done = 0

    def guarded(job):
        if cancel_event is not None and cancel_event.is_set():
            return None
        return call(job)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as ex:
        futures = {ex.submit(guarded, job): i for i, job in enumerate(jobs)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:
                print(f'[AI fan-out] job {i + 1}/{len(jobs)} failed: {exc}')
            done += 1
            if on_progress:
                on_progress(done, len(jobs))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# AI Assistant
# ─────────────────────────────────────────────────────────────────────────────

class AIAssistant:
    """Calls the `claude` CLI via subprocess — uses your Claude Code session,
    no separate API key required."""

    DEFAULT_MODEL = MODEL_SONNET

    # Per-attempt subprocess timeout (seconds). Class attr so tests can
    # shrink it; every timed-out attempt tree-kills the CLI process group.
    CLI_TIMEOUT_S = 360

    # Extended-thinking levels triggered by keywords in the prompt.
    # Claude Code escalates its thinking budget when it sees these tokens.
    THINKING_LEVELS = {
        'off':        '',
        'think':      'Think about this. ',
        'think_hard': 'Think hard about this. ',
        'ultrathink': 'Ultrathink about this. ',
    }
    DEFAULT_THINKING = 'think'

    def __init__(self, model: str = '', thinking: str = ''):
        self._history: list = []
        self._busy = False
        self._model: str = model or self.DEFAULT_MODEL
        self._thinking: str = thinking if thinking in self.THINKING_LEVELS else self.DEFAULT_THINKING
        self._claude_exe: Optional[str] = self._find_claude()

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str):
        self._model = value or self.DEFAULT_MODEL

    @property
    def thinking(self) -> str:
        return self._thinking

    @thinking.setter
    def thinking(self, value: str):
        if value in self.THINKING_LEVELS:
            self._thinking = value

    @staticmethod
    def _find_claude() -> Optional[str]:
        """Locate the claude executable, checking PATH and common install locations."""
        import shutil
        found = shutil.which("claude")
        if found:
            return found
        candidates = [
            # npm global (standard CLI install)
            Path.home() / "AppData" / "Roaming" / "npm" / "claude.cmd",
            Path.home() / "AppData" / "Roaming" / "npm" / "claude",
            # VSCode extension binary (Windows)
            *sorted(
                (Path.home() / ".vscode" / "extensions").glob(
                    "anthropic.claude-code-*/resources/native-binary/claude.exe"
                ),
                reverse=True,  # newest version first
            ),
            # macOS/Linux
            Path("/usr/local/bin/claude"),
            Path.home() / ".local" / "bin" / "claude",
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        return None

    @property
    def ready(self) -> bool:
        return self._claude_exe is not None

    @property
    def busy(self) -> bool:
        return self._busy

    def _transcript(self, limit: int = 8) -> str:
        if not self._history:
            return ""
        lines = ["Conversation so far:"]
        for msg in self._history[-limit:]:
            prefix = "You" if msg["role"] == "user" else "Claude"
            lines.append(f"{prefix}: {msg['content'][:500]}")
        return "\n".join(lines)

    def _reask_for_json(self, original_system: str, original_prompt: str,
                        bad_raw: str,
                        model_override: Optional[str] = None) -> str:
        """When _extract_json fails on a successful API response, ask the
        model to re-output its content as valid JSON only. One retry —
        recovers most parse failures (smart quotes, stray commas, fenced
        code blocks, trailing commentary). Uses the SAME system prompt so
        the cache stays warm for the retry.

        Returns the raw text of the retry response. Caller still has to
        call _extract_json on it (and is expected to let that raise if
        the retry also fails — no infinite-loop)."""
        fixer_prompt = (
            "Your previous response failed to parse as JSON (it may have "
            "been cut off mid-output). Below is the raw text you produced. "
            "Re-output the same content as a single COMPLETE valid JSON "
            "object only — no markdown fences, no commentary, no "
            "explanation. If the original output included preamble or "
            "thinking notes, drop them. If it was cut off, finish the "
            "remaining content more concisely so the whole object fits. "
            "Keep the actual data identical otherwise.\n\n"
            "----- YOUR PREVIOUS OUTPUT -----\n"
            f"{bad_raw[:24000]}\n"
            "----- END -----\n\n"
            "Re-output now as a single valid JSON object:"
        )
        return self._run_claude(original_system, fixer_prompt,
                                model_override=model_override)

    def _run_claude_json(self, system: str, prompt: str,
                         model_override: Optional[str] = None,
                         max_retries: int = 5) -> dict:
        """_run_claude + _extract_json with one re-ask repair pass.

        Every one-shot JSON pass (propose web, distills, canon/codex
        extraction, weaving, audit) should call THIS rather than pairing
        _run_claude with a bare _extract_json — a truncated or malformed
        response then gets salvaged or regenerated instead of failing.
        max_retries: fan-out passes (weave/audit) pass 2 — with many
        concurrent jobs, 5 attempts × 360s each reads as a hang."""
        raw = self._run_claude(system, prompt, max_retries=max_retries,
                               model_override=model_override)
        try:
            return self._extract_json(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f'[AI] JSON parse failed ({exc}); re-asking for valid JSON only...')
            return self._extract_json(self._reask_for_json(
                system, prompt, raw, model_override=model_override))

    @staticmethod
    def _extract_json(raw: str) -> dict:
        """Extract the first balanced top-level JSON object from *raw* text."""
        start = raw.find('{')
        if start == -1:
            preview = raw[:300] if raw else '(empty)'
            raise ValueError(f"No JSON object found in response. Preview: {preview}")
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(raw)):
            ch = raw[i]
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return json.loads(raw[start:i + 1])
        # Braces never balanced — the response was almost certainly cut off
        # mid-generation. Try to salvage the complete prefix before giving up.
        salvaged = AIAssistant._repair_truncated_json(raw[start:])
        if salvaged is not None:
            print(f'[AI] JSON response was truncated (len={len(raw)}); '
                  f'salvaged the complete prefix.')
            return salvaged
        # Show context around where it broke
        preview = raw[start:start+300] if len(raw) > start else raw
        raise ValueError(f"Unbalanced braces in JSON response (depth={depth}, "
                         f"len={len(raw)}, start={start}). Preview: {preview[:200]}")

    @staticmethod
    def _repair_truncated_json(text: str):
        """Best-effort parse of a TRUNCATED top-level JSON object (a
        response that stopped mid-generation). Each round: close any
        unterminated string, append closers for the open containers, and
        try to parse; on failure trim back to the last comma and retry —
        keeping every complete element and dropping the partial tail.
        Returns the parsed dict, or None if nothing parseable survives."""
        text = text.strip()
        for _ in range(200):
            if not text or text[0] != '{':
                return None
            stack = []
            in_string = False
            escape = False
            last_string_open = -1
            for i, ch in enumerate(text):
                if escape:
                    escape = False
                    continue
                if ch == '\\' and in_string:
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    if in_string:
                        last_string_open = i
                    continue
                if in_string:
                    continue
                if ch in '{[':
                    stack.append(ch)
                elif ch in '}]':
                    if not stack:
                        return None   # malformed, not truncated
                    stack.pop()
            if not in_string:
                # Ends at a structural point — try closing the open
                # containers. (Never close a cut-off string: that would
                # keep a silently truncated prose value; trimming below
                # drops the partial field instead.)
                candidate = text + ''.join('}' if c == '{' else ']'
                                           for c in reversed(stack))
                try:
                    parsed = json.loads(candidate)
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    pass
            # Drop the partial tail and retry. rfind may land inside a
            # string (prose commas) — the rescan on the next pass detects
            # that and trims again, converging on a structural boundary.
            idx = text.rfind(',')
            if idx <= 0:
                return None
            text = text[:idx].rstrip()
        return None

    def _run_claude(self, system: str, prompt: str, max_retries: int = 5,
                    model_override: Optional[str] = None) -> str:
        """Blocking call to `claude -p`. Retries with exponential backoff.

        model_override: if provided, use this model id instead of the
        AIAssistant's currently-selected model. Used to force Opus for
        bibles/arc work while keeping the user's selection for node gen.

        Windows' CreateProcess caps the entire command line at ~32,767
        chars, which becomes a problem once the bible (passed via
        --system-prompt) is large. We write the system prompt to a temp
        file and use --system-prompt-file instead; the path is short,
        argv stays small, and the CLI still applies cache_control to the
        system content (cache hits preserved)."""
        prefix = self.THINKING_LEVELS.get(self._thinking, '')
        full_prompt = prefix + prompt if prefix else prompt
        model = model_override or self._model

        # Write system prompt to a temp file so we never put it on argv.
        # NamedTemporaryFile(delete=False) is required on Windows because
        # the file must be closed before another process can open it.
        # The USER prompt is piped via stdin (claude -p with no arg reads
        # the prompt from stdin) for the same reason — long chat histories
        # or deep ancestor contexts can also blow Windows' argv limit.
        import tempfile, os
        sys_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.sysprompt.txt', delete=False, encoding='utf-8'
        )
        try:
            sys_file.write(system)
            sys_file.flush()
            sys_file.close()
            cmd = [
                self._claude_exe,
                "--no-session-persistence",
                "--model", model,
                "--system-prompt-file", sys_file.name,
                "--output-format", "text",
                "-p",
            ]
            return self._run_claude_cmd(cmd, max_retries=max_retries,
                                         stdin_data=full_prompt)
        finally:
            try:
                os.unlink(sys_file.name)
            except OSError:
                pass

    @staticmethod
    def _kill_process_tree(proc):
        """Kill a subprocess AND all its descendants.

        On Windows the claude CLI is a shim that spawns a node child which
        inherits our stdout/stderr pipes. proc.kill() (what subprocess.run's
        timeout path does) kills only the shim; the orphaned grandchild
        keeps the pipes open and the follow-up communicate() blocks FOREVER
        — this froze Weave Junctions with no error and no retry. taskkill /T
        takes down the whole tree so the pipes actually close."""
        try:
            if sys.platform == 'win32':
                subprocess.run(
                    ['taskkill', '/F', '/T', '/PID', str(proc.pid)],
                    capture_output=True, timeout=15)
            else:
                proc.kill()
        except Exception:
            try:
                proc.kill()
            except OSError:
                pass

    def _run_claude_cmd(self, cmd: list, max_retries: int = 5,
                         stdin_data: Optional[str] = None) -> str:
        """Inner retry loop. Pulled out so _run_claude's temp-file
        cleanup wraps it cleanly via try/finally.
        stdin_data: if given, piped to the subprocess's stdin (used for the
        user prompt so it doesn't sit on argv where Windows would cap it).

        Uses Popen + explicit tree-kill instead of subprocess.run(timeout=…):
        run()'s own timeout handling kills just the direct child and then
        blocks indefinitely on Windows when a grandchild still holds the
        pipes (see _kill_process_tree)."""
        last_error = None
        for attempt in range(max_retries):
            backoff = min(3 * (2 ** attempt), 30)  # 3, 6, 12, 24, 30 seconds
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            try:
                out_s, err_s = proc.communicate(input=stdin_data,
                                                timeout=self.CLI_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self._kill_process_tree(proc)
                try:
                    proc.communicate(timeout=15)   # bounded reap, never hangs
                except (subprocess.TimeoutExpired, OSError, ValueError):
                    for st in (proc.stdout, proc.stderr, proc.stdin):
                        try:
                            if st:
                                st.close()
                        except OSError:
                            pass
                last_error = RuntimeError(
                    f"claude CLI timed out ({self.CLI_TIMEOUT_S}s)")
                print(f'[AI] Attempt {attempt+1}/{max_retries}: timeout (retry in {backoff}s)')
                time.sleep(backoff)
                continue
            if proc.returncode != 0:
                err = (err_s or '').strip() or (out_s or '').strip() \
                    or "claude CLI returned non-zero"
                last_error = RuntimeError(err)
                # Don't retry model/auth errors
                if 'model' in err.lower() or 'auth' in err.lower() or 'access' in err.lower():
                    raise last_error
                print(f'[AI] Attempt {attempt+1}/{max_retries} failed: {err[:80]} (retry in {backoff}s)')
                time.sleep(backoff)
                continue
            out = (out_s or '').strip()
            if not out:
                last_error = RuntimeError(
                    f"claude produced no output (stderr: {(err_s or '').strip()!r})")
                print(f'[AI] Attempt {attempt+1}/{max_retries}: empty output (retry in {backoff}s)')
                time.sleep(backoff)
                continue
            return out
        raise last_error or RuntimeError("claude CLI failed after all retries")

    @classmethod
    def _augment_system_with_context(cls,
                                       system: str,
                                       story_context: str,
                                       node_word_range: Optional[tuple] = None,
                                       premise: str = '',
                                       themes: str = '',
                                       motif: str = '',
                                       variables: list = None,
                                       cast_codex: str = '') -> str:
        """Build the cache-friendly system prompt prefix.

        Everything that is STABLE within an arc-generation run belongs here so
        it lives in Anthropic's prompt cache across the run's many calls. Only
        per-call content (parent text, ancestor chain, beat direction,
        sibling summaries, batch_size instruction) stays in the user prompt.

        Blocks (in order, each cache-relevant chunk):
          - LENGTH OVERRIDE       — per-script/per-arc, stable
          - STORY PREMISE         — per-arc, stable
          - THEMATIC THREADS      — per-arc, stable
          - RECURRING MOTIF       — per-arc, stable
          - STORY VARIABLES       — per-script, stable
          - CAST CODEX            — per-story, stable across the run (v2)
          - WORLD BIBLE           — per-script, stable (biggest by far)

        Caller passes whatever they want appended. Empty / None → block skipped.
        The premise role-label and weight are NOT included here (they vary per
        layer) — those stay in the user prompt."""
        out = system
        if node_word_range:
            lo, hi = int(node_word_range[0]), int(node_word_range[1])
            # ~ 150 wpm narration -> roughly lo*0.4 .. hi*0.4 seconds
            t_lo = max(1, int(lo * 0.4))
            t_hi = max(t_lo + 1, int(hi * 0.4))
            out += (
                "\n\n----- LENGTH OVERRIDE -----\n"
                f"Override the word-count instruction in the rules above.\n"
                f"Generate each node as a spoken segment of {lo}-{hi} words\n"
                f"(approximately {t_lo}-{t_hi} seconds when read aloud).\n"
                "Stay inside that range. Stop when the transformation\n"
                "is complete; do not pad to hit the upper bound, but do\n"
                "develop the beat fully — don't cut at the lower bound.\n"
                "----- END LENGTH OVERRIDE -----"
            )
        if premise:
            out += (
                "\n\n----- STORY PREMISE (active arc, stable across this run) -----\n"
                + premise.strip() +
                "\n----- END STORY PREMISE -----"
            )
        if themes:
            out += (
                "\n\n----- THEMATIC THREADS (active arc) -----\n"
                "Let these resonate through the narrative as natural undercurrents:\n  "
                + themes.strip() +
                "\n----- END THEMATIC THREADS -----"
            )
        if motif:
            out += (
                "\n\n----- RECURRING MOTIF (active arc) -----\n"
                "Weave this naturally through the text. When it returns, it must\n"
                "mutate from its prior appearance or be absent altogether — never\n"
                "repeated verbatim.\n\n"
                + motif.strip() +
                "\n----- END RECURRING MOTIF -----"
            )
        if variables:
            vars_block = cls._vars_prompt_section(variables)
            if vars_block:
                out += (
                    "\n\n----- STORY VARIABLES (active script) -----\n"
                    + vars_block +
                    "\n----- END STORY VARIABLES -----"
                )
        if cast_codex:
            out += (
                "\n\n----- CAST CODEX (entities in play — canon) -----\n"
                "These entity cards are CANON for this story. The facts on a\n"
                "card must hold in every node — contradicting a card is an\n"
                "error. When one of these entities is present in a node's\n"
                "text, include its slug (the 'tag:' value) in that node's\n"
                "tags. Where a card and the world bible conflict, the card\n"
                "wins. Like the bible, cards are INSTRUMENTS of each node's\n"
                "transformation, not decoration to showcase.\n\n"
                + cast_codex +
                "\n----- END CAST CODEX -----"
            )
        if story_context:
            sc = story_context[:CONTEXT_MAX]
            if len(story_context) > CONTEXT_MAX:
                sc += '\n[...truncated...]'
            out += (
                "\n\n----- WORLD BIBLE (reference material) -----\n"
                "Use the following world bible as INSTRUMENTS of transformation,\n"
                "not decoration. The user prompt's PARENT NODE and LAYER\n"
                "FUNCTION take priority — write to advance THAT node's\n"
                "transformation, not to showcase world details.\n\n"
                + sc +
                "\n----- END WORLD BIBLE -----"
            )
        return out

    @staticmethod
    def _vars_prompt_section(variables: list) -> str:
        """Build a prompt section telling the AI to set story variables on
        every node. Strongly biases the model toward BIMODAL distributions
        (mostly 0.0, with high commitments at 0.7+) instead of clustering
        in the wishy-washy 0.2–0.6 middle. Also requires a one-line
        'vars_reasoning' explanation in the output so the editor can show
        the user WHY each value was chosen."""
        if not variables:
            return ''
        lines = ['STORY VARIABLES — set "vars" on every node (each value 0.0–1.0):']
        for v in variables:
            lines.append(f'  "{v["name"]}": {v["description"]}')
        lines.append(
            'BE DECISIVE — bimodal distribution, not a wash of middle values. '
            'Use ONE of these tiers for each variable, picking the closest fit:\n'
            '   0.00 — absent.   The variable\'s quality does NOT appear in the\n'
            '                    node text. Default; most variables on most nodes.\n'
            '   0.30 — trace.    A faint hint, almost background. Use SPARINGLY —\n'
            '                    only when the quality is genuinely glancing.\n'
            '   0.70 — present.  Clearly and explicitly part of this node.\n'
            '   1.00 — dominant. The defining force of this node.\n'
            'FORBIDDEN: clustering values in 0.40–0.60. If you\'re tempted by\n'
            'a "middle" value, pick 0.00 or 0.70 instead — commit to one.\n'
            'Most nodes have 1–2 variables non-zero and the rest at 0.00.'
        )
        lines.append(
            'OUTPUT TWO FIELDS for every node:\n'
            '  "vars": {"name": value, ...}     — the numeric assignments\n'
            '  "vars_reasoning": "..."           — ONE short paragraph explaining\n'
            '                                      WHY each non-zero variable is\n'
            '                                      non-zero. Cite specific words\n'
            '                                      or moments from the node text.\n'
            '                                      Variables at 0.00 do not need\n'
            '                                      to be mentioned.'
        )
        return '\n'.join(lines)

    def suggest_variables(self, story_context: str,
                            current_vars: list,
                            sample_nodes: list,
                            arcs_summary: str,
                            ui_queue: queue.SimpleQueue,
                            on_done, on_error,
                            instructions: str = ''):
        """Ask Claude to propose a fresh set of ~6 narrative variables
        for the active script.

        Inputs:
          - story_context: the world bible (goes into the SYSTEM prompt as usual)
          - current_vars: list of {"name","description"} the user already has
                          (so the AI can either iterate on them or replace)
          - sample_nodes: up to ~10 short node texts so the AI sees the
                          script's actual register
          - arcs_summary: short description of each arc (premise + themes)
                          so the AI sees what emotional arc each arc traces
          - instructions: optional free-form user guidance, e.g. "lean more
                          emotional, less plot-driven" — empty for a fresh pass

        Calls ``on_done(list_of_var_dicts)`` on success."""
        if self._busy:
            return
        self._busy = True

        parts = []
        if instructions.strip():
            parts.append(f"AUTHOR DIRECTION (highest priority):\n  {instructions.strip()}")
        if current_vars:
            cur_lines = ["CURRENT VARIABLES (the author has these already — you may keep, refine, or replace):"]
            for v in current_vars:
                cur_lines.append(f'  - "{v.get("name","")}": {v.get("description","")}')
            parts.append('\n'.join(cur_lines))
        if arcs_summary.strip():
            parts.append(f"ARC SUMMARIES (what emotional/structural arcs the narrative will trace):\n{arcs_summary.strip()}")
        if sample_nodes:
            sn_lines = ["SAMPLE NODE TEXTS (a few representative segments, for tone/register):"]
            for s in sample_nodes[:10]:
                sn_lines.append(f'  - "{s[:240]}{"…" if len(s) > 240 else ""}"')
            parts.append('\n'.join(sn_lines))
        parts.append(
            "Propose 4–6 narrative variables that together cover the emotional\n"
            "arc this script seems to be tracing. Output the JSON array only."
        )
        prompt = '\n\n'.join(parts)
        system = self._augment_system_with_context(SYSTEM_SUGGEST_VARS, story_context)

        def run():
            try:
                raw = self._run_claude(system, prompt)
                data = self._extract_json(raw)
                # The model returns an array. Sometimes it nests under a key
                # like {"variables": [...]}. Be permissive.
                if isinstance(data, list):
                    var_list = data
                elif isinstance(data, dict):
                    var_list = data.get('variables') or data.get('vars') or []
                else:
                    var_list = []
                # Sanitize: keep only well-formed entries
                cleaned = []
                for v in var_list:
                    if not isinstance(v, dict):
                        continue
                    name = (v.get('name') or '').strip()
                    desc = (v.get('description') or v.get('desc') or '').strip()
                    if name:
                        cleaned.append({'name': name[:30], 'description': desc[:300]})
                ui_queue.put(lambda v=cleaned: on_done(v))
            except json.JSONDecodeError as exc:
                ui_queue.put(lambda e=exc: on_error(f"JSON parse error: {e}"))
            except Exception as exc:
                ui_queue.put(lambda e=exc: on_error(str(e)))
            finally:
                self._busy = False

        threading.Thread(target=run, daemon=True).start()

    def extract_codex(self, story_context: str, tag_census: str,
                      sample_texts: list, existing_slugs: list,
                      ui_queue: queue.SimpleQueue, on_done, on_error,
                      instructions: str = ''):
        """Ask Claude to propose codex entity cards from the world bible +
        the node graph's tag census. Calls ``on_done(dict[slug -> card])``
        with sanitized, ENTITY_TEMPLATE-shaped cards on success."""
        if self._busy:
            return
        self._busy = True

        parts = []
        if instructions.strip():
            parts.append(f"AUTHOR DIRECTION (highest priority):\n  {instructions.strip()}")
        if existing_slugs:
            parts.append("ALREADY IN CODEX (do NOT re-propose these concepts "
                         "under ANY slug or name variant — no article/plural/"
                         "synonym renames of them either):\n  "
                         + ", ".join(sorted(existing_slugs)))
        if tag_census.strip():
            parts.append(
                "TAG CENSUS — custom tags on existing nodes, with usage count and a\n"
                "sample line of node text. PREFER these as entity slugs when they name\n"
                "the same concept:\n" + tag_census)
        if sample_texts:
            sn_lines = ["SAMPLE NODE TEXTS (register/tone):"]
            for s in sample_texts[:10]:
                sn_lines.append(f'  - "{s[:240]}{"…" if len(s) > 240 else ""}"')
            parts.append('\n'.join(sn_lines))
        parts.append("Propose the codex now. Output the JSON object only.")
        prompt = '\n\n'.join(parts)
        system = self._augment_system_with_context(SYSTEM_EXTRACT_CODEX, story_context)

        def run():
            try:
                data = self._run_claude_json(system, prompt)
                ents = data.get('entities', data) if isinstance(data, dict) else {}
                cleaned = {}
                for slug, card in (ents or {}).items():
                    if not isinstance(card, dict):
                        continue
                    slug = ScriptData.sanitize_entity_slug(str(slug))
                    if not slug or slug in RESERVED_ENTITY_SLUGS:
                        continue
                    clean = _sanitize_entity_card(card)
                    if not clean['name']:
                        clean['name'] = slug
                    clean['notes'] = ''
                    clean['source'] = 'local'
                    cleaned[slug] = clean
                ui_queue.put(lambda c=cleaned: on_done(c))
            except Exception as exc:
                ui_queue.put(lambda e=exc: on_error(str(e)))
            finally:
                self._busy = False

        threading.Thread(target=run, daemon=True).start()

    def chat(self, user_msg: str, ui_queue: queue.SimpleQueue,
             on_reply, on_error, script_summary: str = '', story_context: str = '',
             _system_override: str = '',
             node_word_range: Optional[tuple] = None,
             model_override: Optional[str] = None):
        if self._busy:
            return
        self._busy = True
        self._history.append({"role": "user", "content": user_msg})

        parts = []
        if script_summary:
            parts.append(f"Current script:\n{script_summary}")
        transcript = self._transcript()
        if transcript:
            parts.append(transcript)
        parts.append(f"User: {user_msg}")
        full_prompt = "\n\n".join(parts)
        # World bible goes into the SYSTEM prompt (cache-friendly prefix).
        system = self._augment_system_with_context(
            _system_override or SYSTEM_CHAT, story_context, node_word_range)

        def run():
            try:
                reply = self._run_claude(system, full_prompt,
                                         model_override=model_override)
                self._history.append({"role": "assistant", "content": reply})
                ui_queue.put(lambda: on_reply(reply))
            except Exception as exc:
                self._history.pop()
                ui_queue.put(lambda e=exc: on_error(str(e)))
            finally:
                self._busy = False

        threading.Thread(target=run, daemon=True).start()

    def generate_graph(self, prompt: str, ui_queue: queue.SimpleQueue,
                       on_done, on_error, story_context: str = '',
                       variables: list = None,
                       node_word_range: Optional[tuple] = None):
        if self._busy:
            return
        self._busy = True

        parts = []
        vars_sec = self._vars_prompt_section(variables or [])
        if vars_sec:
            parts.append(vars_sec)
        parts.append(f'SUBJECT (this is what the script must be about — prioritize above all else):\n{prompt}')
        full_prompt = '\n\n'.join(parts)
        # World bible into SYSTEM prompt — cache-friendly across calls.
        system = self._augment_system_with_context(SYSTEM_GENERATE, story_context, node_word_range)

        def run():
            try:
                raw   = self._run_claude(system, full_prompt)
                data = self._extract_json(raw)
                ui_queue.put(lambda: on_done(data))
            except json.JSONDecodeError as exc:
                ui_queue.put(lambda: on_error(f"JSON parse error: {exc}"))
            except Exception as exc:
                ui_queue.put(lambda e=exc: on_error(str(e)))
            finally:
                self._busy = False

        threading.Thread(target=run, daemon=True).start()

    def generate_seed(self, prompt: str, ui_queue: queue.SimpleQueue,
                      on_done, on_error, story_context: str = '',
                      layer_direction: str = '', motif: str = '',
                      variables: list = None,
                      node_word_range: Optional[tuple] = None,
                      cast_codex: str = '',
                      first_layer: str = 'arrival',
                      n_seeds: int = 4):
        """Generate only the opening beat — no children. Used to seed
        iterative generation. `first_layer` supports story structures that
        do not open on the classic 'arrival' archetype; `n_seeds` overrides
        the default of 4 opening nodes (see _seed_count_for_story)."""
        if self._busy:
            return
        self._busy = True

        parts = []
        if layer_direction:
            parts.append(f'OPENING BEAT DIRECTION (this is what the opening nodes must cover):\n{layer_direction}')
        # NOTE: motif and variables now live in the SYSTEM prompt (cache friendly).
        parts.append(f'SUBJECT (this is what the script must be about — prioritize above all else):\n{prompt}')
        if first_layer and first_layer != 'arrival' and first_layer in LAYER_ORDER:
            parts.append(
                f'FIRST BEAT OVERRIDE: this story does not open on the standard '
                f'"arrival" layer — its opening beat is "{first_layer}". Tag every '
                f'generated node with the layer tag "{first_layer}" instead of '
                f'"arrival", prefix node IDs accordingly (e.g. "{first_layer}_..."), '
                f'and write the nodes to perform that beat\'s story function: '
                f'{LAYER_FUNCTIONS.get(first_layer, "")}'
            )
        if n_seeds and n_seeds != 4:
            parts.append(
                f'SEED COUNT OVERRIDE: generate exactly {n_seeds} opening '
                f'node(s) instead of the default 4.'
            )
        full_prompt = '\n\n'.join(parts)
        # Push motif + variables + cast codex into the cached system prompt.
        # (The story's premise has already been baked into `prompt` as the
        # subject — no need to duplicate it in the system block here.)
        system = self._augment_system_with_context(
            SYSTEM_GENERATE_SEED, story_context, node_word_range,
            motif=motif, variables=variables, cast_codex=cast_codex,
        )

        def run():
            try:
                print(f'[AI] generate_seed: calling claude...')
                raw   = self._run_claude(system, full_prompt)
                print(f'[AI] generate_seed: got {len(raw)} chars, extracting JSON...')
                try:
                    data = self._extract_json(raw)
                except (ValueError, json.JSONDecodeError) as exc:
                    print(f'[AI] generate_seed JSON parse failed ({exc}); re-asking...')
                    data = self._extract_json(self._reask_for_json(system, full_prompt, raw))
                nodes = data.get('nodes', {})
                print(f'[AI] generate_seed: {len(nodes)} nodes parsed')
                ui_queue.put(lambda: on_done(data))
            except json.JSONDecodeError as exc:
                print(f'[AI] generate_seed JSON error: {exc}')
                ui_queue.put(lambda e=exc: on_error(f"JSON parse error: {e}"))
            except Exception as exc:
                print(f'[AI] generate_seed error: {exc}')
                # Include first 200 chars of raw response for debugging
                raw_preview = raw[:200] if 'raw' in dir() else '(no response)'
                print(f'[AI]   raw preview: {raw_preview}')
                ui_queue.put(lambda e=exc: on_error(str(e)))
            finally:
                self._busy = False

        threading.Thread(target=run, daemon=True).start()

    def generate_layer(self, frontier: list, ui_queue: queue.SimpleQueue,
                       on_done, on_error, story_context: str = '',
                       existing_custom_tags: list = None,
                       layer_direction: str = '', motif: str = '',
                       variables: list = None,
                       node_word_range: Optional[tuple] = None):
        """Generate the next layer for all frontier nodes in one AI call.

        frontier: list of (node_id, node_data) for all current leaf nodes.
        """
        if self._busy:
            return
        self._busy = True

        parts = []
        vars_sec = self._vars_prompt_section(variables or [])
        if vars_sec:
            parts.append(vars_sec)
        if layer_direction:
            parts.append(f'LAYER DIRECTION (this layer must cover this — follow it precisely, override default arc guidance):\n{layer_direction}')
        if motif:
            parts.append(f'RECURRING MOTIF (weave this through every new node naturally):\n{motif}')
        if existing_custom_tags:
            parts.append(f'EXISTING TAGS (reuse where applicable): {", ".join(sorted(existing_custom_tags))}')

        source_block = 'SOURCE NODES (generate the next layer continuing from all of these):'
        for nid, nd in frontier:
            tags_str = ', '.join(nd.get('tags', []))
            text     = nd.get('text', '')[:200]
            source_block += f'\n  [{nid}] tags: {tags_str}\n  "{text}"'
        parts.append(source_block)

        n_new = max(2, min(5, len(frontier) + 1))
        parts.append(f'Generate {n_new}–{n_new + 1} new nodes. Every source node must appear in at least one connect_from.')
        full_prompt = '\n\n'.join(parts)
        system = self._augment_system_with_context(SYSTEM_GENERATE_LAYER, story_context, node_word_range)

        def run():
            try:
                raw   = self._run_claude(system, full_prompt)
                data = self._extract_json(raw)
                ui_queue.put(lambda: on_done(data))
            except json.JSONDecodeError as exc:
                ui_queue.put(lambda e=exc: on_error(f"JSON parse error: {e}"))
            except Exception as exc:
                ui_queue.put(lambda e=exc: on_error(str(e)))
            finally:
                self._busy = False

        threading.Thread(target=run, daemon=True).start()

    def expand_node(self, source_id: str, source_text: str, source_tags: list,
                    hint: str, ui_queue: queue.SimpleQueue, on_done, on_error,
                    story_context: str = '', node_hint: str = '',
                    upstream_path: list = None,
                    node_min: int = 2, node_max: int = 5,
                    existing_custom_tags: list = None,
                    variables: list = None,
                    node_word_range: Optional[tuple] = None):
        if self._busy:
            return
        self._busy = True

        # Prompt is ordered lowest→highest weight so the most important content
        # is closest to the generation point (recency bias).
        parts = []

        # Background world bible is sent in the SYSTEM prompt below (cache
        # friendly across calls). Per-call USER prompt stays small.

        if existing_custom_tags:
            parts.append(f'EXISTING CUSTOM TAGS in this script (prefer these when applicable): {", ".join(sorted(existing_custom_tags))}')

        vars_sec = self._vars_prompt_section(variables or [])
        if vars_sec:
            parts.append(vars_sec)

        # ── Ancestor context: keywords and tags, oldest first ────────────────
        _STOPWORDS = frozenset({
            "the","a","an","and","or","but","in","on","at","to","for","of","with",
            "by","from","is","was","are","were","be","been","being","have","has",
            "had","do","does","did","will","would","could","should","may","might",
            "shall","can","that","this","it","its","they","their","them","there",
            "then","than","what","which","who","whom","when","where","how","not",
            "no","so","if","as","into","just","like","over","some","each","only",
            "also","very","about","up","out","all","more","one","two","said","he",
            "she","his","her","we","our","you","your","my",
        })
        ANCESTOR_KW_COUNTS = [6, 8, 12, 18]
        if upstream_path:
            parts.append('\nANCESTOR CONTEXT (thematic thread — maintain continuity):')
            for i, entry in enumerate(upstream_path):
                nid = entry[0]
                text = entry[1]
                tags = entry[2] if len(entry) > 2 else []
                n_kw = ANCESTOR_KW_COUNTS[i] if i < len(ANCESTOR_KW_COUNTS) else 6
                words = re.findall(r"\b[a-zA-Z]{3,}\b", text)
                keywords = list(dict.fromkeys(
                    w.lower() for w in words if w.lower() not in _STOPWORDS
                ))[:n_kw]
                custom_tags = [t for t in tags if t not in set(LAYER_ORDER)]
                parts.append(f'  [{nid}] tags: {custom_tags}, keywords: {", ".join(keywords)}')

        # ── Node intent / guidance (near-primary) ────────────────────────────
        if node_hint:
            parts.append(
                f'\nNODE INTENT (author direction — high weight):\n  {node_hint}'
            )
        if hint:
            parts.append(f'GUIDANCE: {hint}')

        # ── Source node (highest weight — closest to generation) ─────────────
        layer_tags = set(LAYER_ORDER)
        custom_tags = [t for t in source_tags if t not in layer_tags]

        words = re.findall(r"\b[a-zA-Z]{4,}\b", source_text)
        stopwords = {"that","this","with","from","have","they","their","there",
                     "when","what","which","will","been","were","then","than",
                     "just","into","like","over","some","each","only","also"}
        anchors = list(dict.fromkeys(
            w.lower() for w in words if w.lower() not in stopwords
        ))[:12]

        source_block = (
            f'\nSOURCE NODE — stay close to this:\n'
            f'  ID: {source_id}\n'
            f'  Tags: {source_tags}\n'
            f'  Text: "{source_text}"'
        )
        if custom_tags:
            source_block += (
                f'\n  Custom tags to carry into daughters: {", ".join(custom_tags)}'
            )
        if anchors:
            source_block += (
                f'\n  Key imagery to remain present: {", ".join(anchors)}'
            )
        parts.append(source_block)

        parts.append(f'\nGenerate {node_min}–{node_max} continuation nodes branching from this node.')
        prompt = '\n'.join(parts)
        system = self._augment_system_with_context(SYSTEM_EXPAND, story_context, node_word_range)

        def run():
            try:
                raw   = self._run_claude(system, prompt)
                data = self._extract_json(raw)
                ui_queue.put(lambda: on_done(data))
            except json.JSONDecodeError as exc:
                ui_queue.put(lambda e=exc: on_error(f"JSON parse error: {e}"))
            except Exception as exc:
                ui_queue.put(lambda e=exc: on_error(str(e)))
            finally:
                self._busy = False

        threading.Thread(target=run, daemon=True).start()

    def generate_single_node_sync(self, parent_id: str, parent_text: str,
                                   parent_tags: list, ancestor_chain: list,
                                   layer_name: str, layer_direction: str = '',
                                   motif: str = '', themes: str = '',
                                   sibling_summaries: list = None,
                                   story_context: str = '',
                                   existing_custom_tags: list = None,
                                   variables: list = None,
                                   premise: str = '',
                                   premise_weight: float = 1.0,
                                   node_word_range: Optional[tuple] = None) -> dict:
        """Blocking call that generates exactly one node. Returns parsed dict.

        Designed to be called from worker threads in ParallelNodeOrchestrator.
        Does NOT use _busy flag — caller is responsible for concurrency control.
        premise_weight: 0.0–1.0 controls how strongly the premise influences this node.
        """
        parts = []

        # Premise — full text is in the SYSTEM prompt (cached). Here we
        # only ship the per-layer role label + weight, which vary per beat.
        if premise and premise_weight > 0.05:
            weight_pct = int(premise_weight * 100)
            _, role = LAYER_PREMISE_ROLES.get(layer_name, (premise_weight, ''))
            if role:
                parts.append(
                    f'PREMISE ROLE AT THIS BEAT [{weight_pct}% influence]: {role}\n'
                    f'(The full premise text is in the SYSTEM prompt above.)'
                )
            else:
                parts.append(
                    f'PREMISE WEIGHT AT THIS BEAT: {weight_pct}%\n'
                    f'(The full premise text is in the SYSTEM prompt above.)'
                )

        # Bible, themes, motif, story-variables now live in the SYSTEM prompt.

        if existing_custom_tags:
            parts.append(f'EXISTING TAGS (prefer these): {", ".join(sorted(existing_custom_tags))}')

        # Ancestor context — keywords and tags, not raw truncated text.
        # Older ancestors are compressed more aggressively.
        _STOPWORDS = frozenset({
            "the","a","an","and","or","but","in","on","at","to","for","of","with",
            "by","from","is","was","are","were","be","been","being","have","has",
            "had","do","does","did","will","would","could","should","may","might",
            "shall","can","that","this","it","its","they","their","them","there",
            "then","than","what","which","who","whom","when","where","how","not",
            "no","so","if","as","into","just","like","over","some","each","only",
            "also","very","about","up","out","all","more","one","two","said","he",
            "she","his","her","we","our","you","your","my",
        })
        ANCESTOR_LABELS = [
            ('DISTANT ANCESTOR (themes only)',  6),   # keyword count
            ('GREAT-GRANDPARENT (key images)',   8),
            ('GRANDPARENT (imagery + mood)',    12),
            ('DIRECT PARENT (rich context)',    18),
        ]
        if ancestor_chain:
            parts.append('\nANCESTOR CONTEXT (thematic thread — do not copy, just maintain continuity):')
            for i, (nid, text, tags) in enumerate(ancestor_chain):
                label, n_keywords = ANCESTOR_LABELS[i] if i < len(ANCESTOR_LABELS) else ANCESTOR_LABELS[0]
                # Extract distinctive keywords
                words = re.findall(r"\b[a-zA-Z]{3,}\b", text)
                keywords = list(dict.fromkeys(
                    w.lower() for w in words if w.lower() not in _STOPWORDS
                ))[:n_keywords]
                custom_tags = [t for t in tags if t not in set(LAYER_ORDER)]
                parts.append(f'  {label}')
                parts.append(f'  [{nid}] tags: {custom_tags}, keywords: {", ".join(keywords)}')

        # Sibling differentiation
        if sibling_summaries:
            parts.append('\nSIBLINGS ALREADY GENERATED (take a DIFFERENT angle):')
            for sib_id, sib_summary in sibling_summaries:
                parts.append(f'  [{sib_id}]: {sib_summary}')

        # Layer function (story-structure work this beat is supposed to do).
        # Sits ABOVE layer_direction so the AI reads the universal beat-function
        # first, then the per-arc specific intent.
        layer_fn = LAYER_FUNCTIONS.get(layer_name, '')
        if layer_fn:
            parts.append(
                f'\nLAYER FUNCTION — what this beat MUST do in story terms '
                f'(universal across all arcs):\n  {layer_fn}'
            )

        # Layer direction (per-beat, varies per call).
        if layer_direction:
            parts.append(f'\nLAYER DIRECTION (this arc\'s specific intent for this beat): {layer_direction}')
        # NOTE: motif and themes now live in the SYSTEM prompt (cached).

        # Parent node (highest weight)
        parts.append(
            f'\nPARENT NODE — continue from this:\n'
            f'  ID: {parent_id}\n'
            f'  Tags: {parent_tags}\n'
            f'  Text: "{parent_text}"'
        )

        parts.append(
            f'\nGenerate exactly 1 node in the "{layer_name}" layer continuing from {parent_id}. '
            f'The node MUST perform exactly ONE transformation '
            f'(change in knowledge / position / resource between start and end of node) '
            f'and operate at least one engine (causal / emotional / epistemic). '
            f'See STORYTELLING DISCIPLINE in system prompt for forbidden patterns.'
        )
        prompt = '\n'.join(parts)
        system = self._augment_system_with_context(
            SYSTEM_GENERATE_SINGLE_NODE, story_context, node_word_range,
            premise=premise, themes=themes, motif=motif, variables=variables,
        )

        raw = self._run_claude(system, prompt)
        try:
            return self._extract_json(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f'[AI] generate_single_node_sync JSON parse failed ({exc}); re-asking...')
            return self._extract_json(self._reask_for_json(system, prompt, raw))

    def generate_batch_sync(self, parent_id: str, parent_text: str,
                             parent_tags: list, ancestor_chain: list,
                             layer_name: str, batch_size: int = 3,
                             layer_direction: str = '', hint: str = '',
                             motif: str = '',
                             sibling_summaries: list = None,
                             story_context: str = '',
                             existing_custom_tags: list = None,
                             variables: list = None,
                             premise: str = '',
                             premise_weight: float = 1.0,
                             themes: str = '',
                             node_word_range: Optional[tuple] = None,
                             cast_codex: str = '') -> dict:
        """Blocking call that generates multiple sibling nodes from one parent.

        Returns dict with 'nodes' and 'connect_from' keys (SYSTEM_EXPAND format).
        """
        parts = []

        # Premise — the full premise text now lives in the SYSTEM prompt (cached).
        # Here in the user prompt we only include the layer-specific ROLE LABEL
        # and WEIGHT, which vary per layer and would invalidate the cache if
        # they sat in the system prompt.
        if premise and premise_weight > 0.05:
            weight_pct = int(premise_weight * 100)
            _, role = LAYER_PREMISE_ROLES.get(layer_name, (premise_weight, ''))
            if role:
                parts.append(
                    f'PREMISE ROLE AT THIS BEAT [{weight_pct}% influence]: {role}\n'
                    f'(The full premise text is in the SYSTEM prompt above.)'
                )
            else:
                parts.append(
                    f'PREMISE WEIGHT AT THIS BEAT: {weight_pct}%\n'
                    f'(The full premise text is in the SYSTEM prompt above.)'
                )

        # Bible, themes, motif, story-variables all live in the SYSTEM prompt
        # below (cache friendly across this entire arc-generation run).

        if existing_custom_tags:
            parts.append(f'EXISTING TAGS (prefer these): {", ".join(sorted(existing_custom_tags))}')

        # Ancestor context — keywords and tags
        _STOPWORDS = frozenset({
            "the","a","an","and","or","but","in","on","at","to","for","of","with",
            "by","from","is","was","are","were","be","been","being","have","has",
            "had","do","does","did","will","would","could","should","may","might",
            "shall","can","that","this","it","its","they","their","them","there",
            "then","than","what","which","who","whom","when","where","how","not",
            "no","so","if","as","into","just","like","over","some","each","only",
            "also","very","about","up","out","all","more","one","two","said","he",
            "she","his","her","we","our","you","your","my",
        })
        ANCESTOR_KW = [6, 8, 12, 18]
        if ancestor_chain:
            parts.append('\nANCESTOR CONTEXT (thematic thread):')
            for i, entry in enumerate(ancestor_chain):
                nid, text = entry[0], entry[1]
                tags = entry[2] if len(entry) > 2 else []
                n_kw = ANCESTOR_KW[i] if i < len(ANCESTOR_KW) else 6
                words = re.findall(r"\b[a-zA-Z]{3,}\b", text)
                keywords = list(dict.fromkeys(
                    w.lower() for w in words if w.lower() not in _STOPWORDS
                ))[:n_kw]
                custom_tags = [t for t in tags if t not in set(LAYER_ORDER)]
                parts.append(f'  [{nid}] tags: {custom_tags}, keywords: {", ".join(keywords)}')

        # Layer function (story-structure work this beat is supposed to do).
        layer_fn = LAYER_FUNCTIONS.get(layer_name, '')
        if layer_fn:
            parts.append(
                f'\nLAYER FUNCTION — what this beat MUST do in story terms '
                f'(universal across all arcs):\n  {layer_fn}'
            )

        if layer_direction:
            parts.append(f'\nLAYER DIRECTION (this arc\'s specific intent for this beat): {layer_direction}')
        # NOTE: motif and themes now live in the SYSTEM prompt (cached).
        # They are not re-included here to avoid paying full rate every call.

        # Author hint (high priority — right before source node)
        # Source node
        parts.append(
            f'\nSOURCE NODE — stay close to this:\n'
            f'  ID: {parent_id}\n'
            f'  Tags: {parent_tags}\n'
            f'  Text: "{parent_text}"'
        )

        parts.append(
            f'\nGenerate exactly {batch_size} continuation nodes in the '
            f'"{layer_name}" layer branching from this node. '
            f'Each node MUST perform exactly ONE transformation '
            f'(change in knowledge / position / resource between start and end of node) '
            f'and operate at least one engine (causal / emotional / epistemic). '
            f'Siblings MUST perform DIFFERENT transformations from each other — '
            f'not just take different angles on the same one. '
            f'See STORYTELLING DISCIPLINE in system prompt for forbidden patterns.'
        )

        # Author hint LAST — highest recency weight, overrides all other guidance
        if hint:
            parts.append(f'\nCRITICAL — AUTHOR DIRECTION (this overrides thematic continuity): {hint}')
        prompt = '\n'.join(parts)
        # Premise / themes / motif / variables / cast codex go into the SYSTEM
        # prompt now, making them part of the cache-friendly prefix for the
        # whole story-generation run.
        system = self._augment_system_with_context(
            SYSTEM_EXPAND, story_context, node_word_range,
            premise=premise, themes=themes, motif=motif, variables=variables,
            cast_codex=cast_codex,
        )

        raw = self._run_claude(system, prompt)
        try:
            return self._extract_json(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            # JSON parse failure — re-ask once. The model's first response was
            # valid text but not valid JSON; one cheap retry usually recovers
            # the call without losing the work it already did.
            print(f'[AI] JSON parse failed ({exc}); re-asking for valid JSON only...')
            return self._extract_json(self._reask_for_json(system, prompt, raw))

    def suggest_cross_links_sync(self, layer_nodes: list,
                                  children_map: dict) -> list:
        """Blocking call that suggests cross-branch connections.

        layer_nodes: [(node_id, text, tags, child_ids), ...] — all nodes at one layer
        children_map: {child_id: (text, tags)} — all children of those nodes
        Returns: [{"from": str, "to": str}, ...]
        """
        parts = [f'NODES AT THIS LAYER ({len(layer_nodes)}):']
        for nid, text, tags, child_ids in layer_nodes:
            parts.append(f'  [{nid}] tags: {tags}')
            parts.append(f'    text: "{text[:150]}"')
            parts.append(f'    children: {child_ids}')

        parts.append(f'\nCHILDREN (potential cross-link targets):')
        for cid, (ctext, ctags) in children_map.items():
            parts.append(f'  [{cid}] tags: {ctags}')
            parts.append(f'    text: "{ctext[:100]}"')

        prompt = '\n'.join(parts)
        raw = self._run_claude(SYSTEM_CROSS_LINK, prompt)
        data = self._extract_json(raw)
        return data.get('cross_links', [])

    def continue_from_node(self, source_id: str, source_text: str, source_tags: list,
                           ui_queue: queue.SimpleQueue, on_done, on_error,
                           story_context: str = '', node_hint: str = '',
                           variables: list = None,
                           node_word_range: Optional[tuple] = None):
        if self._busy:
            return
        self._busy = True

        source_layer = next((t for t in source_tags if t in LAYER_ORDER), 'discovery')

        parts = []
        vars_sec = self._vars_prompt_section(variables or [])
        if vars_sec:
            parts.append(vars_sec)
        if node_hint:
            parts.append(f'Continuation guidance: {node_hint}')
        parts.append(
            f'SOURCE NODE (layer: {source_layer}, id: {source_id}):\n'
            f'  tags: {source_tags}\n'
            f'  text: "{source_text}"\n\n'
            f'Generate the continuation layers that come AFTER "{source_layer}".'
        )
        full_prompt = '\n\n'.join(parts)
        system = self._augment_system_with_context(SYSTEM_CONTINUE, story_context, node_word_range)

        def run():
            try:
                raw   = self._run_claude(system, full_prompt)
                data = self._extract_json(raw)
                ui_queue.put(lambda: on_done(data))
            except json.JSONDecodeError as exc:
                ui_queue.put(lambda e=exc: on_error(f"JSON parse error: {e}"))
            except Exception as exc:
                ui_queue.put(lambda e=exc: on_error(str(e)))
            finally:
                self._busy = False

        threading.Thread(target=run, daemon=True).start()

    def rewrite_text(self, prompt: str, ui_queue: queue.SimpleQueue,
                     on_done, on_error, story_context: str = '',
                     variables: list = None):
        if self._busy:
            return
        self._busy = True

        parts = []
        if story_context:
            parts.append(f'Story context:\n{story_context}')
        vars_sec = self._vars_prompt_section(variables or [])
        if vars_sec:
            parts.append(vars_sec)
        parts.append(prompt)
        full_prompt = '\n\n'.join(parts)

        def run():
            try:
                raw   = self._run_claude(SYSTEM_REWRITE, full_prompt)
                data = self._extract_json(raw)
                ui_queue.put(lambda: on_done(data))
            except json.JSONDecodeError as exc:
                ui_queue.put(lambda e=exc: on_error(f"JSON parse error: {e}"))
            except Exception as exc:
                ui_queue.put(lambda e=exc: on_error(str(e)))
            finally:
                self._busy = False

        threading.Thread(target=run, daemon=True).start()

    def generate_focused_context(self, full_context: str,
                                  ui_queue: queue.SimpleQueue, on_done, on_error):
        if self._busy:
            return
        self._busy = True
        system = (
            "You distill a detailed story/character bible into a focused AI context "
            "for use during node expansion in a narrative audio script.\n\n"
            "Rules:\n"
            "- Output plain text only — no JSON, no markdown, no headers\n"
            f"- Maximum {CONTEXT_MAX} characters\n"
            "- Keep: core character nature, tone/voice guidance, key sensory anchors\n"
            "- Drop: backstory detail, lists, repeated ideas, anything decorative\n"
            "- Write in present tense, dense and specific\n"
            "- This text will be labeled 'background flavour' so it must not overpower "
            "the node being expanded — keep it atmospheric, not directive"
        )
        def run():
            try:
                result = self._run_claude(system, full_context)
                ui_queue.put(lambda r=result: on_done(r.strip()))
            except Exception as exc:
                ui_queue.put(lambda e=exc: on_error(str(e)))
            finally:
                self._busy = False
        threading.Thread(target=run, daemon=True).start()

    # System prompt for the multi-turn story-context collaborator.
    # Used by chat_context(); kept as a class attribute so the dialog can
    # show it / tweak it later without poking at internals.
    SYSTEM_CONTEXT_CHAT = (
        "You are a collaborator helping develop a story / character bible for a "
        "narrative audio script. The bible is reference material — a dense, "
        "evocative portrait of premise, world, characters, voice, sensory "
        "anchors, and themes — that a human reads and edits.\n\n"
        "There are TWO commit paths the human can use on your reply:\n"
        "  (1) REPLACE — your reply becomes the new bible in full.\n"
        "  (2) APPEND  — your reply is added to the END of the existing bible.\n"
        "Choose your output shape based on what the user is asking for:\n\n"
        "- REWRITE / REVISE / REDRAFT requests (\"revise the whole bible\", "
        "  \"rewrite this to feel more bittersweet\", \"redo the protagonist\") → "
        "  output the FULL revised bible. Flowing prose. No preamble.\n\n"
        "- ADD / EXPAND / INTRODUCE requests (\"add a section on the harbor\", "
        "  \"introduce a sister character\", \"expand on the sound of the kiln\", "
        "  \"include the priestess subplot\") → output ONLY the new material as a "
        "  self-contained addition that will sit at the end of the existing bible. "
        "  Do NOT restate facts already established earlier in the conversation or "
        "  in the initial bible. Open in a way that flows naturally after a "
        "  paragraph break.\n\n"
        "- Questions, brainstorming, pushback → reply conversationally. Don't "
        "  dump prose. Just talk.\n\n"
        "If the request is ambiguous between rewrite and add, default to ADD "
        "(it's the safer non-destructive option) and tell the user which mode "
        "you chose in a brief one-line preface that they can ignore.\n\n"
        "Other rules:\n"
        "- No markdown, no headers, no bullet points in the prose itself. Present tense.\n"
        "- Full rewrites: 2000–4000 chars, dense, specific, sensory.\n"
        "- Additions: as long as the request warrants. Don't pad.\n"
        "- Always carry forward the user's prior decisions. If they said the "
        "  protagonist is a tile-setter, do not later make them a librarian."
    )

    def chat_context(self, history: list, user_msg: str,
                     ui_queue: queue.SimpleQueue, on_done, on_error,
                     model_override: Optional[str] = None):
        """Multi-turn collaboration on the Full Context.

        history: list of {"role": "user"|"assistant", "content": str} —
                 the dialog owns this list and appends to it.
        user_msg: the new message from the user.
        on_done(reply: str): called with Claude's reply text.

        The full transcript is rendered into a single text prompt because
        the `claude -p` CLI does not accept a structured chat history.
        """
        if self._busy:
            return
        self._busy = True

        parts = []
        if history:
            parts.append("Conversation so far:")
            for msg in history:
                who = "User" if msg["role"] == "user" else "You (Claude)"
                parts.append(f"{who}: {msg['content']}")
            parts.append("")
            parts.append(f"User: {user_msg}")
            parts.append("")
            parts.append("Reply now as Claude.")
        else:
            parts.append(user_msg)
        prompt = "\n".join(parts)

        def run():
            try:
                reply = self._run_claude(self.SYSTEM_CONTEXT_CHAT, prompt,
                                         model_override=model_override)
                ui_queue.put(lambda r=reply.strip(): on_done(r))
            except Exception as exc:
                ui_queue.put(lambda e=exc: on_error(str(e)))
            finally:
                self._busy = False
        threading.Thread(target=run, daemon=True).start()

    def determine_variables(self, node_texts: list, variables: list,
                            ui_queue: queue.SimpleQueue, on_done, on_error):
        """Analyze node texts and assign variable values via AI.

        node_texts: list of (node_id, text) tuples
        variables:  script.variables list [{"name": ..., "description": ...}, ...]
        on_done:    called with {node_id: {"var_name": float, ...}, ...}
        """
        if self._busy:
            return
        self._busy = True

        BATCH_SIZE = 30

        def run():
            try:
                all_results = {}
                for i in range(0, len(node_texts), BATCH_SIZE):
                    batch = node_texts[i:i + BATCH_SIZE]
                    parts = ['STORY VARIABLES (each 0.0–1.0):']
                    for v in variables:
                        parts.append(f'  "{v["name"]}": {v["description"]}')
                    parts.append('')
                    parts.append(f'NODES TO ANALYZE ({len(batch)}):')
                    for nid, text in batch:
                        parts.append(f'  [{nid}]: "{text}"')
                    prompt = '\n'.join(parts)

                    raw = self._run_claude(SYSTEM_DETERMINE_VARS, prompt)
                    try:
                        data = self._extract_json(raw)
                    except (ValueError, json.JSONDecodeError):
                        continue
                    if isinstance(data, dict):
                        all_results.update(data)

                    batch_num = i // BATCH_SIZE + 1
                    total_batches = (len(node_texts) + BATCH_SIZE - 1) // BATCH_SIZE
                    if total_batches > 1:
                        ui_queue.put(lambda b=batch_num, t=total_batches:
                            None)  # progress tracked in status bar by caller

                ui_queue.put(lambda r=all_results: on_done(r))
            except Exception as exc:
                ui_queue.put(lambda e=exc: on_error(str(e)))
            finally:
                self._busy = False

        threading.Thread(target=run, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Voice Manager
# ─────────────────────────────────────────────────────────────────────────────

VOICE_CACHE_PATH = REPO_ROOT / "config" / "voice_cache.json"


class VoiceManager:
    """Fetches ElevenLabs voices and generates audio files."""

    NO_OVERRIDE = "(use default)"

    def __init__(self):
        self._voices: list = []          # [(name, voice_id), ...]
        self._fetching = False
        self.api_key: str = os.environ.get("ELEVENLABS_API_KEY", "")
        self._load_cache()

    def _load_cache(self):
        try:
            if VOICE_CACHE_PATH.exists():
                data = json.loads(VOICE_CACHE_PATH.read_text(encoding="utf-8"))
                self._voices = [(v["name"], v["id"]) for v in data]
        except Exception:
            pass

    def _save_cache(self):
        try:
            VOICE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = [{"name": n, "id": vid} for n, vid in self._voices]
            VOICE_CACHE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    @property
    def fetching(self) -> bool:
        return self._fetching

    @property
    def names(self) -> list:
        return [n for n, _ in self._voices]

    def id_for_name(self, name: str) -> Optional[str]:
        return next((vid for n, vid in self._voices if n == name), None)

    def name_for_id(self, vid: str) -> Optional[str]:
        return next((n for n, v in self._voices if v == vid), None)

    def fetch_voices(self, ui_queue: queue.SimpleQueue, on_done, on_error):
        if self._fetching or not self.api_key:
            return
        self._fetching = True

        def run():
            try:
                from elevenlabs.client import ElevenLabs
                client = ElevenLabs(api_key=self.api_key)
                voices = client.voices.get_all().voices
                self._voices = sorted(
                    [(v.name, v.voice_id) for v in voices],
                    key=lambda x: x[0],
                )
                self._save_cache()
                ui_queue.put(lambda: on_done(self._voices))
            except Exception as exc:
                ui_queue.put(lambda e=exc: on_error(str(e)))
            finally:
                self._fetching = False

        threading.Thread(target=run, daemon=True).start()

    def generate(self, text: str, voice_id: str, out_path: Path,
                 settings: dict, ui_queue: queue.SimpleQueue, on_done, on_error):
        def run():
            try:
                from elevenlabs.client import ElevenLabs
                from elevenlabs import VoiceSettings
                # Clamp to safe ranges before sending to API
                safe = dict(settings)
                _clamp_voice_settings(safe)
                client = ElevenLabs(api_key=self.api_key)
                audio = client.text_to_speech.convert(
                    voice_id=voice_id,
                    text=text,
                    model_id=safe.get("model", "eleven_multilingual_v2"),
                    voice_settings=VoiceSettings(
                        stability=safe.get("stability", 0.5),
                        similarity_boost=safe.get("similarity_boost", 0.75),
                        style=0.0,
                        use_speaker_boost=True,
                    ),
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as f:
                    for chunk in audio:
                        f.write(chunk)
                ui_queue.put(lambda: on_done(out_path))
            except Exception as exc:
                ui_queue.put(lambda e=exc: on_error(str(e)))

        threading.Thread(target=run, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Playback helpers
# ─────────────────────────────────────────────────────────────────────────────

def _play_audio_blocking(path: Path, stop_event: threading.Event):
    """Decode MP3 with miniaudio and play via sounddevice; blocks until done or stopped."""
    import miniaudio
    import sounddevice as sd
    import numpy as np

    decoded = miniaudio.decode_file(
        str(path), nchannels=2, sample_rate=44100,
        output_format=miniaudio.SampleFormat.SIGNED16,
    )
    samples = np.frombuffer(decoded.samples, dtype=np.int16).astype(np.float32) / 32768.0
    samples = samples.reshape(-1, 2)
    sd.play(samples, samplerate=44100)
    try:
        while sd.get_stream().active:
            if stop_event.is_set():
                sd.stop()
                return
            time.sleep(0.05)
    except Exception:
        sd.stop()


def _weighted_choice(nexts: list, weights: list) -> str:
    total = sum(weights) or 1.0
    r = random.random() * total
    acc = 0.0
    for nid, w in zip(nexts, weights):
        acc += w
        if r <= acc:
            return nid
    return nexts[-1]


def _playback_loop(play_script: "ScriptData", stop_event: threading.Event,
                   get_delay, ui_queue: queue.SimpleQueue,
                   on_node, on_finish, on_error):
    try:
        starts = list(play_script.start_nodes)
        if not starts:
            starts = list(play_script.nodes.keys())
        if not starts:
            ui_queue.put(lambda: on_error("No nodes in script"))
            return

        current = random.choice(starts)

        while not stop_event.is_set():
            nd = play_script.nodes.get(current)
            if not nd:
                break

            ui_queue.put(lambda n=current: on_node(n))

            audio_dir  = play_script.path.parent if play_script.path else SOUNDS_DIR
            audio_file = audio_dir / f"{current}.mp3"

            if audio_file.exists():
                try:
                    _play_audio_blocking(audio_file, stop_event)
                except Exception as exc:
                    ui_queue.put(lambda e=exc: on_error(f"Audio error: {e}"))

            if stop_event.is_set():
                break

            # Inter-node delay
            delay    = get_delay()
            deadline = time.time() + delay
            while time.time() < deadline and not stop_event.is_set():
                remaining = deadline - time.time()
                ui_queue.put(lambda r=remaining, n=current: on_node(n, r))
                time.sleep(0.25)

            if stop_event.is_set():
                break

            nexts = nd.get("next", [])
            if not nexts:
                break
            weights = nd.get("weights", [1.0] * len(nexts))
            current = _weighted_choice(nexts, weights)

    except Exception as exc:
        ui_queue.put(lambda e=exc: on_error(str(e)))
    finally:
        ui_queue.put(lambda: on_finish())


# ─────────────────────────────────────────────────────────────────────────────
# NodeGraphQt node class and tag colors
# ─────────────────────────────────────────────────────────────────────────────

TAG_COLORS = {
    'arrival':      (60, 100, 180),
    'presence':     (70, 130, 170),
    'curiosity':    (80, 160, 140),
    'discovery':    (100, 170, 80),
    'complication': (170, 160, 50),
    'intimacy':     (180, 120, 70),
    'turn':         (190, 70, 70),
    'consequence':  (160, 50, 90),
    'echo':         (100, 90, 180),
    'stillness':    (160, 140, 180),
}


class NarrativeNode(BaseNode):
    __identifier__ = 'narrative'
    NODE_NAME = 'NarrativeNode'

    def __init__(self):
        super().__init__()
        self.add_input('in',  multi_input=True,  display_name=False)
        self.add_output('out', multi_output=True, display_name=False)


# ─────────────────────────────────────────────────────────────────────────────
# PropertiesPanel
# ─────────────────────────────────────────────────────────────────────────────

def _find_term_spans(text: str, term: str, whole_word: bool = False) -> list:
    """Case-insensitive (start, end) spans of `term` inside `text`.

    With ``whole_word``, occurrences flanked by word characters are
    excluded — searching 'rain' stops matching 'train' or 'raining'.
    Boundary guards are applied only where the TERM's own edge is a word
    character, so punctuation-edged terms behave sensibly: '...' still
    matches 'wait...', and 'rain,' matches 'rain,' but not 'train,'.

    Single source of truth for search matching: used by the node search
    (MainWindow._cmd_search) AND every highlight renderer (text view,
    label/tags line edits, arc-beat label), so what lights up is always
    exactly what matched.
    """
    if not term or not text:
        return []
    if whole_word:
        head = r'(?<!\w)' if re.match(r'\w', term[0]) else ''
        tail = r'(?!\w)' if re.match(r'\w', term[-1]) else ''
        pat = re.compile(head + re.escape(term) + tail, re.IGNORECASE)
        return [(m.start(), m.end()) for m in pat.finditer(text)]
    tlow, qlow = text.lower(), term.lower()
    spans = []
    start = 0
    while True:
        idx = tlow.find(qlow, start)
        if idx < 0:
            break
        spans.append((idx, idx + len(qlow)))
        start = idx + max(1, len(qlow))
    return spans


class _HighlightLineEdit(QLineEdit):
    """QLineEdit that paints a translucent yellow highlight rectangle over
    every substring matching a search term. QLineEdit has no native subrange
    highlighting API, so we override paintEvent and overlay rects computed
    via QFontMetrics + the style's content-rect for the widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._highlight_term: str = ''
        self._highlight_word: bool = False

    def set_highlight_term(self, term: str, whole_word: bool = False):
        new = term or ''
        word = bool(whole_word)
        if new == self._highlight_term and word == self._highlight_word:
            return
        self._highlight_term = new
        self._highlight_word = word
        self.update()  # trigger repaint

    def paintEvent(self, event):
        # Let QLineEdit paint normally first (border, bg, text, cursor).
        super().paintEvent(event)
        if not self._highlight_term:
            return
        text = self.text()
        if not text:
            return
        spans = _find_term_spans(text, self._highlight_term, self._highlight_word)
        if not spans:
            return

        from PySide6.QtWidgets import QStyle, QStyleOptionFrame
        from PySide6.QtGui import QPainter, QColor, QFontMetrics
        from PySide6.QtCore import QRect, Qt as _Qt

        # Find the content rect (where the text actually sits) via Qt's style.
        opt = QStyleOptionFrame()
        self.initStyleOption(opt)
        contents = self.style().subElementRect(
            QStyle.SubElement.SE_LineEditContents, opt, self
        )
        fm = QFontMetrics(self.font())
        text_h = fm.height()
        y_offset = contents.top() + (contents.height() - text_h) // 2
        x_base = contents.left()

        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            color = QColor(255, 220, 100, 150)  # translucent yellow
            painter.setBrush(color)
            painter.setPen(_Qt.PenStyle.NoPen)
            for s, e in spans:
                x = x_base + fm.horizontalAdvance(text[:s])
                w = fm.horizontalAdvance(text[s:e])
                painter.drawRect(QRect(x, y_offset, w, text_h))
        finally:
            painter.end()


class PropertiesPanel(QWidget):
    node_modified = Signal(str)  # node_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._script: Optional[ScriptData] = None
        self._node_id: Optional[str] = None
        self._blocking = False
        self._vm: Optional[VoiceManager] = None
        self._ai: Optional[AIAssistant] = None
        self._ui_queue: Optional[queue.SimpleQueue] = None
        self._search_term: str = ''  # current main-window search term for in-text highlighting
        self._search_text_only: bool = False  # mirror of the main window's Text-only search toggle
        self._search_whole_word: bool = False  # mirror of the main window's Word search toggle
        self._arc_beat_raw: str = ''  # raw arc_beat text (so we can re-render with highlights)
        self._build_ui()

    def set_context(self, script: ScriptData, vm: VoiceManager,
                    ai: AIAssistant, ui_queue: queue.SimpleQueue):
        self._script = script
        self._vm = vm
        self._ai = ai
        self._ui_queue = ui_queue

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Header
        hdr = QLabel("Node Properties")
        hdr.setStyleSheet("font-weight: bold; font-size: 13px; color: #aaddff;")
        layout.addWidget(hdr)

        # Node ID (hidden — internal use only)
        self.id_edit = QLineEdit()
        self.id_edit.setReadOnly(True)
        self.id_edit.hide()

        # Label (display name)
        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Name:"))
        self.label_edit = _HighlightLineEdit()
        self.label_edit.setPlaceholderText("Node name...")
        self.label_edit.textChanged.connect(self._autosave_label)
        self.label_edit.textChanged.connect(self._apply_search_highlights)
        label_row.addWidget(self.label_edit)
        layout.addLayout(label_row)

        # Arc — read-only, shows which story arc this node belongs to.
        self._arc_name_lbl = QLabel('')
        self._arc_name_lbl.setTextFormat(Qt.TextFormat.RichText)
        self._arc_name_lbl.setWordWrap(True)
        layout.addWidget(self._arc_name_lbl)

        # Text
        layout.addWidget(QLabel("Text:"))
        self.text_edit = QTextEdit()
        self.text_edit.setMinimumHeight(240)
        self.text_edit.setMaximumHeight(450)
        self.text_edit.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self.text_edit.textChanged.connect(self._autosave_text)
        self.text_edit.textChanged.connect(self._update_word_count)
        self.text_edit.textChanged.connect(self._apply_search_highlights)
        layout.addWidget(self.text_edit)

        self._word_count_lbl = QLabel("")
        self._word_count_lbl.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._word_count_lbl)

        self._arc_beat_lbl = QLabel("")
        self._arc_beat_lbl.setTextFormat(Qt.TextFormat.RichText)
        self._arc_beat_lbl.setWordWrap(True)
        layout.addWidget(self._arc_beat_lbl)

        # Rewrite hint
        self.rewrite_hint = QLineEdit()
        self.rewrite_hint.setPlaceholderText("AI rewrite hint...")
        layout.addWidget(self.rewrite_hint)

        rewrite_btn = QPushButton("AI Rewrite")
        rewrite_btn.clicked.connect(self._cmd_rewrite)
        layout.addWidget(rewrite_btn)

        self.rewrite_status = QLabel("")
        self.rewrite_status.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        layout.addWidget(self.rewrite_status)

        # Node hint
        lbl_hint = QLabel("Node Hint (for AI):")
        lbl_hint.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        layout.addWidget(lbl_hint)
        self.hint_edit = QTextEdit()
        self.hint_edit.setPlaceholderText("Optional context for AI expand/rewrite on this node...")
        self.hint_edit.setMinimumHeight(50)
        self.hint_edit.setMaximumHeight(80)
        self.hint_edit.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self.hint_edit.textChanged.connect(self._autosave_hint)
        layout.addWidget(self.hint_edit)

        # Expand button + node count
        expand_row = QHBoxLayout()
        self.expand_btn = QPushButton("Expand Node (AI)")
        self.expand_btn.clicked.connect(self._cmd_expand)
        expand_row.addWidget(self.expand_btn)
        expand_row.addWidget(QLabel("nodes:"))
        self.expand_min = QDoubleSpinBox()
        self.expand_min.setDecimals(0)
        self.expand_min.setRange(1, 20)
        self.expand_min.setValue(2)
        self.expand_min.setFixedWidth(44)
        self.expand_min.setToolTip("Min nodes to generate")
        expand_row.addWidget(self.expand_min)
        expand_row.addWidget(QLabel("–"))
        self.expand_max = QDoubleSpinBox()
        self.expand_max.setDecimals(0)
        self.expand_max.setRange(1, 20)
        self.expand_max.setValue(5)
        self.expand_max.setFixedWidth(44)
        self.expand_max.setToolTip("Max nodes to generate")
        expand_row.addWidget(self.expand_max)
        layout.addLayout(expand_row)

        # Layer tag (dropdown)
        layer_row = QHBoxLayout()
        layer_row.addWidget(QLabel("Layer:"))
        self.layer_combo = QComboBox()
        self.layer_combo.addItem("(none)", "")
        for _lt in LAYER_ORDER:
            self.layer_combo.addItem(_lt, _lt)
        self.layer_combo.currentIndexChanged.connect(self._autosave_tags)
        layer_row.addWidget(self.layer_combo)
        layout.addLayout(layer_row)

        # Custom tags
        custom_row = QHBoxLayout()
        custom_row.addWidget(QLabel("Tags:"))
        self.tags_edit = _HighlightLineEdit()
        self.tags_edit.setPlaceholderText("goat, toad, revelation, ...")
        self.tags_edit.setToolTip("Custom tags: characters, themes, locations — comma separated")
        self.tags_edit.textChanged.connect(self._autosave_tags)
        self.tags_edit.textChanged.connect(self._apply_search_highlights)
        custom_row.addWidget(self.tags_edit)
        layout.addLayout(custom_row)

        # ── Story variables (dynamic — rebuilt when definitions change) ────
        self._vars_container = QWidget()
        self._vars_layout = QGridLayout(self._vars_container)
        self._vars_layout.setContentsMargins(0, 4, 0, 4)
        self._vars_layout.setSpacing(4)
        self._vars_spins: dict = {}   # name -> QDoubleSpinBox
        self._vars_container.hide()
        layout.addWidget(self._vars_container)

        # ── vars_reasoning display ──────────────────────────────────────
        # Read-only explanation of WHY the AI assigned the current values.
        # Populated when the AI generates a node; user can edit it
        # manually too. Hidden if no value is set on the current node.
        self._vars_reasoning_lbl = QLabel("Variable reasoning:")
        self._vars_reasoning_lbl.setStyleSheet("color: #aaaaaa; font-size: 10px; margin-top: 4px;")
        self._vars_reasoning_edit = QTextEdit()
        self._vars_reasoning_edit.setPlaceholderText(
            "AI-generated explanation of why each variable was set this way. "
            "Cite specific words/moments. Edit freely.")
        self._vars_reasoning_edit.setFixedHeight(72)
        self._vars_reasoning_edit.textChanged.connect(self._autosave_vars_reasoning)
        self._vars_reasoning_lbl.hide()
        self._vars_reasoning_edit.hide()
        layout.addWidget(self._vars_reasoning_lbl)
        layout.addWidget(self._vars_reasoning_edit)

        # Is start
        self.is_start_cb = QCheckBox("Start node")
        self.is_start_cb.stateChanged.connect(self._autosave_start)
        layout.addWidget(self.is_start_cb)

        # Trigger weather state — optional. When this node STARTS, the
        # narrative_player will request a transition to the chosen state.
        # Populated with the script's associated-weather-set states; the
        # row is hidden entirely if no weather set references this script.
        self._trigger_row = QHBoxLayout()
        self._trigger_label = QLabel("Trigger state:")
        self._trigger_row.addWidget(self._trigger_label)
        self.trigger_state_combo = QComboBox()
        self.trigger_state_combo.setToolTip(
            "Optional. When this node begins playing, the env-system\n"
            "transitions to the chosen weather state. Most nodes leave\n"
            "this blank — only set on 'anchor' nodes that should pull\n"
            "the visuals to a specific state.")
        self.trigger_state_combo.currentIndexChanged.connect(
            self._autosave_trigger_state)
        self._trigger_row.addWidget(self.trigger_state_combo)
        # Wrap in a widget so we can show/hide the whole row at once
        self._trigger_row_widget = QWidget()
        self._trigger_row_widget.setLayout(self._trigger_row)
        layout.addWidget(self._trigger_row_widget)

        # Outgoing edges
        layout.addWidget(QLabel("Outgoing Edges:"))
        self.edge_list_widget = QWidget()
        self.edge_list_layout = QVBoxLayout(self.edge_list_widget)
        self.edge_list_layout.setContentsMargins(0, 0, 0, 0)
        self.edge_list_layout.setSpacing(2)
        layout.addWidget(self.edge_list_widget)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #555;")
        layout.addWidget(sep)

        # Voice override
        voice_row = QHBoxLayout()
        voice_row.addWidget(QLabel("Voice:"))
        self.voice_combo = QComboBox()
        self.voice_combo.addItem(VoiceManager.NO_OVERRIDE)
        self.voice_combo.currentTextChanged.connect(self._autosave_voice)
        voice_row.addWidget(self.voice_combo)
        layout.addLayout(voice_row)

        # Voice settings
        form = QFormLayout()
        form.setSpacing(4)

        self.stability_spin = QDoubleSpinBox()
        self.stability_spin.setRange(VOICE_STABILITY_MIN, VOICE_STABILITY_MAX)
        self.stability_spin.setSingleStep(0.05)
        self.stability_spin.setDecimals(2)
        self.stability_spin.setToolTip(f"Safe range: {VOICE_STABILITY_MIN}–{VOICE_STABILITY_MAX}")
        self.stability_spin.valueChanged.connect(self._autosave_voice_settings)
        form.addRow("Stability:", self.stability_spin)

        self.similarity_spin = QDoubleSpinBox()
        self.similarity_spin.setRange(0.0, 1.0)
        self.similarity_spin.setSingleStep(0.05)
        self.similarity_spin.setDecimals(2)
        self.similarity_spin.valueChanged.connect(self._autosave_voice_settings)
        form.addRow("Similarity Boost:", self.similarity_spin)


        layout.addLayout(form)

        gen_audio_btn = QPushButton("Generate Audio")
        gen_audio_btn.clicked.connect(self._cmd_generate_audio)
        layout.addWidget(gen_audio_btn)

        # Audio file path (editable)
        audio_file_row = QHBoxLayout()
        audio_file_row.addWidget(QLabel("Audio:"))
        self.audio_file_edit = QLineEdit()
        self.audio_file_edit.setPlaceholderText("(no file)")
        self.audio_file_edit.setToolTip("Relative path to audio file from project root")
        self.audio_file_edit.editingFinished.connect(self._autosave_audio_file)
        audio_file_row.addWidget(self.audio_file_edit, stretch=1)
        self.audio_browse_btn = QPushButton("…")
        self.audio_browse_btn.setFixedWidth(28)
        self.audio_browse_btn.setToolTip("Browse for audio file")
        self.audio_browse_btn.clicked.connect(self._cmd_browse_audio_file)
        audio_file_row.addWidget(self.audio_browse_btn)
        layout.addLayout(audio_file_row)

        audio_status_row = QHBoxLayout()
        self.audio_status = QLabel("")
        self.audio_status.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        audio_status_row.addWidget(self.audio_status, stretch=1)
        self.play_audio_btn = QPushButton("▶ Play Node")
        self.play_audio_btn.setFixedWidth(90)
        self.play_audio_btn.clicked.connect(self._cmd_play_node_audio)
        audio_status_row.addWidget(self.play_audio_btn)
        layout.addLayout(audio_status_row)

        self._audio_stop_event: Optional[threading.Event] = None

        layout.addStretch(1)

        del_btn = QPushButton("Delete Node")
        del_btn.setStyleSheet("background-color: #8B2222; color: white;")
        del_btn.clicked.connect(self._cmd_delete)
        layout.addWidget(del_btn)

        self.setEnabled(False)

    def set_search_term(self, term: str):
        """Called by MainWindow when the search bar text changes.
        Re-highlights any occurrences of `term` inside the node text view."""
        new = (term or '').strip()
        if new == self._search_term:
            return
        self._search_term = new
        self._apply_search_highlights()

    def set_search_scope(self, text_only: bool):
        """Called by MainWindow when the Text-only search toggle flips.
        In text-only mode the label / tags / arc-beat fields stop
        highlighting — they no longer participate in matching, and a lit
        highlight there would misreport what the search actually hit."""
        flag = bool(text_only)
        if flag == self._search_text_only:
            return
        self._search_text_only = flag
        self._apply_search_highlights()

    def set_search_word(self, whole_word: bool):
        """Called by MainWindow when the whole-Word search toggle flips.
        Highlights re-render so 'rain' stops lighting up 'train'."""
        flag = bool(whole_word)
        if flag == self._search_whole_word:
            return
        self._search_whole_word = flag
        self._apply_search_highlights()

    def _apply_search_highlights(self):
        """Highlight occurrences of the current search term across every
        property-panel field that participates in node search:
          - text_edit (QTextEdit): range highlights via ExtraSelections
          - label_edit, tags_edit (QLineEdit): tint the widget if it contains
            the term (Qt can't subrange-highlight a QLineEdit without
            subclassing it, so we tint as a 'contains-match' signal)
          - _arc_beat_lbl (QLabel): re-render with <mark> spans around hits
        """
        from PySide6.QtWidgets import QTextEdit as _QTE
        from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
        term = self._search_term
        qlow = term.lower() if term else ''

        # 1. text_edit — non-destructive range highlights.
        selections = []
        if qlow:
            text = self.text_edit.toPlainText()
            fmt = QTextCharFormat()
            fmt.setBackground(QColor(255, 220, 100, 180))
            fmt.setForeground(QColor(0, 0, 0))
            for s, e in _find_term_spans(text, term, self._search_whole_word):
                cursor = self.text_edit.textCursor()
                cursor.setPosition(s)
                cursor.setPosition(e, QTextCursor.MoveMode.KeepAnchor)
                sel = _QTE.ExtraSelection()
                sel.cursor = cursor
                sel.format = fmt
                selections.append(sel)
        self.text_edit.setExtraSelections(selections)

        # 2 + 3. label_edit & tags_edit — push the term into our
        # _HighlightLineEdit subclass, which overlays per-match rects.
        # Suppressed in text-only mode: those fields aren't searched.
        meta_term = '' if self._search_text_only else self._search_term
        self.label_edit.set_highlight_term(meta_term, self._search_whole_word)
        self.tags_edit.set_highlight_term(meta_term, self._search_whole_word)

        # 4. arc_beat_lbl — re-render with <mark> spans.
        self._render_arc_beat_label()

    def _render_arc_name_label(self, script: 'ScriptData', node_id: str):
        """Update the small label below the Name field to show which arc
        this node belongs to (or '(unassigned)' if it has no arc_id)."""
        arc_id = script.get_node_arc_id(node_id) if script else ''
        if arc_id:
            arc = script.get_arc(arc_id)
            name = (arc.get('name') or arc_id).strip() if arc else arc_id
            self._arc_name_lbl.setText(
                f'<span style="color:#aaccaa; font-size:10px;">'
                f'<b>Arc:</b> {name}</span>'
            )
        else:
            self._arc_name_lbl.setText(
                '<span style="color:#888888; font-size:10px; font-style:italic;">'
                'Arc: (unassigned)</span>'
            )

    def _render_arc_beat_label(self):
        """Render the arc-beat label, marking any search-term occurrences."""
        raw = self._arc_beat_raw
        if not raw:
            self._arc_beat_lbl.setText('')
            return
        import html as _html
        body = _html.escape(raw)
        # No arc-beat highlights in text-only mode — the field isn't searched.
        term = '' if self._search_text_only else self._search_term
        spans = _find_term_spans(raw, term, self._search_whole_word)
        if spans:
            # Splice <mark> spans into the escaped body, escaping each
            # segment separately so offsets stay aligned with `raw`.
            out = []
            i = 0
            for s, e in spans:
                out.append(_html.escape(raw[i:s]))
                out.append(
                    '<span style="background-color:#ffdc64; color:#000;">'
                    f'{_html.escape(raw[s:e])}</span>'
                )
                i = e
            out.append(_html.escape(raw[i:]))
            body = ''.join(out)
        self._arc_beat_lbl.setText(
            f'<span style="color:#7799cc; font-size:10px;">'
            f'<b>Arc beat:</b> {body}</span>'
        )

    def load_node(self, script: ScriptData, node_id: str):
        self._script = script
        self._node_id = node_id
        nd = script.nodes.get(node_id, {})

        self._blocking = True
        try:
            self.id_edit.setText(node_id)
            self.label_edit.setText(nd.get("label") or node_id)
            self._render_arc_name_label(script, node_id)
            self.text_edit.setPlainText(nd.get("text", ""))
            self.hint_edit.setPlainText(nd.get("hint", ""))
            self.rewrite_hint.clear()
            all_tags   = nd.get("tags", [])
            layer_tags = set(LAYER_ORDER)
            layer_tag  = next((t for t in all_tags if t in layer_tags), "")
            custom_tags = [t for t in all_tags if t not in layer_tags]
            idx = self.layer_combo.findData(layer_tag)
            self.layer_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.tags_edit.setText(", ".join(custom_tags))
            self.is_start_cb.setChecked(node_id in script.start_nodes)

            # Trigger-state dropdown: rebuild options from the script's
            # associated weather set (reverse-lookup). Hide the row if
            # the script isn't associated with any set.
            options = script.trigger_state_options
            self.trigger_state_combo.blockSignals(True)
            self.trigger_state_combo.clear()
            if options:
                self.trigger_state_combo.addItem("(none)", "")
                for sv in options:
                    self.trigger_state_combo.addItem(sv, sv)
                # Restore current value if any
                current_trig = nd.get('trigger_state') or ""
                idx = self.trigger_state_combo.findData(current_trig)
                self.trigger_state_combo.setCurrentIndex(idx if idx >= 0 else 0)
                self._trigger_row_widget.show()
            else:
                self._trigger_row_widget.hide()
            self.trigger_state_combo.blockSignals(False)

            vs = nd.get("voice_settings", {})
            if _clamp_voice_settings(vs):
                nd["voice_settings"] = vs
                script.dirty = True
            self.stability_spin.setValue(vs.get("stability", 0.5))
            self.similarity_spin.setValue(vs.get("similarity_boost", 0.75))

            # Voice combo
            voice_id = nd.get("voice")
            if voice_id and self._vm:
                name = self._vm.name_for_id(voice_id) or voice_id
                idx = self.voice_combo.findText(name)
                self.voice_combo.setCurrentIndex(idx if idx >= 0 else 0)
            else:
                self.voice_combo.setCurrentIndex(0)

            # Audio file path + status
            file_path = nd.get("file", "")
            self.audio_file_edit.setText(file_path)
            if file_path:
                full = REPO_ROOT / file_path
                if full.exists():
                    self.audio_status.setText(f"✓ {Path(file_path).name}")
                    self.audio_status.setStyleSheet("color: #88ee88; font-size: 10px;")
                else:
                    self.audio_status.setText(f"✗ File missing")
                    self.audio_status.setStyleSheet("color: #ff5555; font-size: 10px;")
            else:
                self.audio_status.setText("No audio file")
                self.audio_status.setStyleSheet("color: #aaaaaa; font-size: 10px;")

            arc_beat = nd.get('arc_beat', '')
            self._arc_beat_raw = arc_beat
            self._render_arc_beat_label()

            self.rebuild_edge_list(script, node_id)

            # Story variables
            node_vars = nd.get("vars", {})
            for name, spin in self._vars_spins.items():
                spin.setValue(node_vars.get(name, 0.0))

            # Variable reasoning (AI-generated explanation, user-editable)
            reasoning = (nd.get("vars_reasoning") or "").strip()
            self._vars_reasoning_edit.setPlainText(reasoning)
            # Only show the field if the script has variables defined
            has_vars = bool(self._vars_spins)
            self._vars_reasoning_lbl.setVisible(has_vars)
            self._vars_reasoning_edit.setVisible(has_vars)
        finally:
            self._blocking = False

        self.setEnabled(True)

    def preview_node(self, script: ScriptData, node_id: str):
        """Show another node's data without changing the active _node_id.
        All autosave signals are blocked so nothing gets overwritten."""
        if not node_id or node_id not in script.nodes:
            return
        saved_id = self._node_id
        self._node_id = None          # block autosave during load
        self.load_node(script, node_id)
        self._node_id = saved_id      # restore so saves still go to selected node

    def end_preview(self):
        """Restore the panel to the currently selected node (or clear if none)."""
        if self._node_id and self._script:
            self.load_node(self._script, self._node_id)
        else:
            self.clear()

    def clear(self, multi_select: bool = False):
        self._node_id = None
        self._blocking = True
        try:
            self.id_edit.setText("(multiple)" if multi_select else "")
            self.label_edit.setText("")
            self._arc_name_lbl.setText('')
            self.text_edit.setPlainText("")
            self.hint_edit.setPlainText("")
            self.rewrite_hint.clear()
            self.layer_combo.setCurrentIndex(0)
            self.tags_edit.setText("")
            self.is_start_cb.setChecked(False)
            # Hide the trigger-state row when no node is selected
            self._trigger_row_widget.hide()
            self.trigger_state_combo.blockSignals(True)
            self.trigger_state_combo.clear()
            self.trigger_state_combo.blockSignals(False)
            # Hide vars_reasoning when nothing is selected
            self._vars_reasoning_edit.clear()
            self._vars_reasoning_lbl.hide()
            self._vars_reasoning_edit.hide()
            self.stability_spin.setValue(0.5)
            self.similarity_spin.setValue(0.75)
            self.audio_file_edit.setText("")
            self.audio_status.setText("")
            self.rewrite_status.setText("")
            # Clear edge list
            for i in reversed(range(self.edge_list_layout.count())):
                w = self.edge_list_layout.itemAt(i).widget()
                if w:
                    w.deleteLater()
        finally:
            self._blocking = False
        self.setEnabled(False)

    def rebuild_edge_list(self, script: ScriptData, node_id: str):
        # Clear old rows
        for i in reversed(range(self.edge_list_layout.count())):
            w = self.edge_list_layout.itemAt(i).widget()
            if w:
                w.deleteLater()

        nd = script.nodes.get(node_id, {})
        nexts   = nd.get("next", [])
        weights = nd.get("weights", [])

        if not nexts:
            lbl = QLabel("(no outgoing edges)")
            lbl.setStyleSheet("color: #777;")
            self.edge_list_layout.addWidget(lbl)
            return

        self._weight_spinboxes: Dict[str, QDoubleSpinBox] = {}
        for i, target_id in enumerate(nexts):
            w = weights[i] if i < len(weights) else 1.0
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            lbl = QLabel(f"-> {target_id}")
            lbl.setStyleSheet("color: #aabbff;")
            row_layout.addWidget(lbl, 1)

            spin = QDoubleSpinBox()
            spin.setRange(0.0, 10.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(2)
            spin.setValue(w)
            spin.setFixedWidth(65)
            spin.valueChanged.connect(
                lambda val, tid=target_id: self._save_weight(tid, val)
            )
            row_layout.addWidget(spin)

            del_btn = QPushButton("×")
            del_btn.setFixedWidth(22)
            del_btn.setFixedHeight(22)
            del_btn.setStyleSheet("color: #ff6666; font-weight: bold; border: none; background: transparent;")
            del_btn.setToolTip(f"Delete edge → {target_id}")
            del_btn.clicked.connect(
                lambda _=False, tid=target_id: self.node_modified.emit(
                    f"__delete_edge__{self._node_id}__{tid}"
                )
            )
            row_layout.addWidget(del_btn)

            self._weight_spinboxes[target_id] = spin
            self.edge_list_layout.addWidget(row)

    def update_voice_list(self, names: list):
        self._blocking = True
        try:
            current = self.voice_combo.currentText()
            self.voice_combo.clear()
            self.voice_combo.addItem(VoiceManager.NO_OVERRIDE)
            for name in names:
                self.voice_combo.addItem(name)
            idx = self.voice_combo.findText(current)
            self.voice_combo.setCurrentIndex(idx if idx >= 0 else 0)
        finally:
            self._blocking = False

    def _save_weight(self, target_id: str, value: float):
        if self._blocking or not self._node_id or not self._script:
            return
        nd = self._script.nodes.get(self._node_id, {})
        nexts = nd.get("next", [])
        if target_id in nexts:
            idx = nexts.index(target_id)
            weights = nd.get("weights", [])
            if idx < len(weights):
                weights[idx] = value
                self._script.dirty = True

    def _autosave_text(self):
        if self._blocking or not self._node_id or not self._script:
            return
        raw       = self.text_edit.toPlainText()
        sanitized = _sanitize_tts(raw)
        if sanitized != raw:
            # Replace bad chars in the editor without moving the cursor far
            cursor = self.text_edit.textCursor()
            pos    = cursor.position()
            self._blocking = True
            try:
                self.text_edit.setPlainText(sanitized)
                cursor.setPosition(min(pos, len(sanitized)))
                self.text_edit.setTextCursor(cursor)
            finally:
                self._blocking = False
        self._script.update_text(self._node_id, sanitized)
        self.node_modified.emit(self._node_id)

    def _update_word_count(self):
        text  = self.text_edit.toPlainText().strip()
        words = len(text.split()) if text else 0
        if words == 0:
            self._word_count_lbl.setText('')
        else:
            if words < 40:
                color = '#ffaa44'
            elif words > 100:
                color = '#ff6666'
            else:
                color = '#888888'
            self._word_count_lbl.setText(
                f'<span style="color:{color}; font-size:10px;">'
                f'{words} words&nbsp;&nbsp;(40–100)</span>')

    def _autosave_hint(self):
        if self._blocking or not self._node_id or not self._script:
            return
        self._script.update_hint(self._node_id, self.hint_edit.toPlainText())

    def _autosave_label(self):
        if self._blocking or not self._node_id or not self._script:
            return
        self._script.update_label(self._node_id, self.label_edit.text().strip())
        self.node_modified.emit(self._node_id)

    def _autosave_tags(self):
        if self._blocking or not self._node_id or not self._script:
            return
        layer = self.layer_combo.currentData() or ""
        custom = [t.strip() for t in self.tags_edit.text().split(",") if t.strip()]
        tags = ([layer] if layer else []) + custom
        self._script.update_tags(self._node_id, tags)
        self.node_modified.emit(self._node_id)

    def _autosave_start(self):
        if self._blocking or not self._node_id or not self._script:
            return
        self._script.set_start(self._node_id, self.is_start_cb.isChecked())
        self.node_modified.emit(self._node_id)

    def _autosave_trigger_state(self):
        """Write the dropdown's current selection back to the node's
        trigger_state field. Empty string ("(none)") removes the field
        entirely so old scripts stay clean of empty values."""
        if self._blocking or not self._node_id or not self._script:
            return
        nd = self._script.nodes.get(self._node_id)
        if not nd:
            return
        val = self.trigger_state_combo.currentData() or ""
        if val:
            nd['trigger_state'] = val
        else:
            nd.pop('trigger_state', None)
        self._script.dirty = True
        self.node_modified.emit(self._node_id)

    def _autosave_vars(self):
        """Write current spinbox values back to the node's vars dict."""
        if self._blocking or not self._node_id or not self._script:
            return
        nd = self._script.nodes.get(self._node_id)
        if not nd:
            return
        nd.setdefault("vars", {})
        for name, spin in self._vars_spins.items():
            nd["vars"][name] = round(spin.value(), 2)
        self._script.dirty = True

    def _autosave_vars_reasoning(self):
        """Write the vars_reasoning text back to the node. Empty string
        removes the field entirely so re-saved scripts stay clean."""
        if self._blocking or not self._node_id or not self._script:
            return
        nd = self._script.nodes.get(self._node_id)
        if not nd:
            return
        text = self._vars_reasoning_edit.toPlainText().strip()
        if text:
            nd['vars_reasoning'] = text
        else:
            nd.pop('vars_reasoning', None)
        self._script.dirty = True

    def refresh_variable_widgets(self):
        """Rebuild the variable spinbox grid to match current script variable definitions."""
        # Clear old widgets
        while self._vars_layout.count():
            item = self._vars_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._vars_spins.clear()

        variables = self._script.variables if self._script else []
        if not variables:
            self._vars_container.hide()
            return

        # Header
        hdr = QLabel("Variables")
        hdr.setStyleSheet("color: #ccddaa; font-size: 10px; font-weight: bold;")
        self._vars_layout.addWidget(hdr, 0, 0, 1, 4)

        # 2-column grid: label + spin, label + spin
        for i, var in enumerate(variables[:6]):
            row = 1 + i // 2
            col = (i % 2) * 2
            lbl = QLabel(f"{var['name']}:")
            lbl.setStyleSheet("color: #aaaaaa; font-size: 10px;")
            lbl.setToolTip(var.get('description', ''))
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(2)
            spin.setFixedWidth(60)
            spin.setToolTip(var.get('description', ''))
            spin.valueChanged.connect(self._autosave_vars)
            self._vars_layout.addWidget(lbl, row, col)
            self._vars_layout.addWidget(spin, row, col + 1)
            self._vars_spins[var['name']] = spin

        self._vars_container.show()

    def _autosave_audio_file(self):
        if self._blocking or not self._node_id or not self._script:
            return
        val = self.audio_file_edit.text().strip()
        nd = self._script.nodes.get(self._node_id, {})
        if val:
            nd["file"] = val
            full = REPO_ROOT / val
            if full.exists():
                self.audio_status.setText(f"✓ {Path(val).name}")
                self.audio_status.setStyleSheet("color: #88ee88; font-size: 10px;")
            else:
                self.audio_status.setText("✗ File missing")
                self.audio_status.setStyleSheet("color: #ff5555; font-size: 10px;")
        else:
            nd.pop("file", None)
            self.audio_status.setText("No audio file")
            self.audio_status.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        self._script.dirty = True
        self.node_modified.emit(self._node_id)

    def _cmd_browse_audio_file(self):
        if not self._node_id or not self._script:
            return
        start_dir = str(self._script.path.parent) if self._script.path else str(SOUNDS_DIR)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", start_dir,
            "Audio Files (*.mp3 *.wav *.ogg *.flac);;All Files (*)")
        if not path:
            return
        try:
            rel = str(Path(path).relative_to(REPO_ROOT))
        except ValueError:
            rel = path
        self.audio_file_edit.setText(rel)
        self._autosave_audio_file()

    def _autosave_voice(self):
        if self._blocking or not self._node_id or not self._script:
            return
        val = self.voice_combo.currentText()
        if val == VoiceManager.NO_OVERRIDE or not val:
            voice_id = None
        elif self._vm:
            voice_id = self._vm.id_for_name(val) or val
        else:
            voice_id = val
        self._script.update_node_voice(self._node_id, voice_id)

    def _autosave_voice_settings(self):
        if self._blocking or not self._node_id or not self._script:
            return
        self._script.update_node_voice_settings(self._node_id, {
            "stability":        round(self.stability_spin.value(), 2),
            "similarity_boost": round(self.similarity_spin.value(), 2),
            "style":            0.0,
        })

    def _cmd_rewrite(self):
        if not self._node_id or not self._script or not self._ai or not self._ui_queue:
            return
        if not self._ai.ready:
            self.rewrite_status.setText("claude CLI not found")
            return
        if self._ai.busy:
            self.rewrite_status.setText("AI is busy...")
            return

        nd = self._script.nodes.get(self._node_id, {})
        current = self.text_edit.toPlainText().strip()
        hint    = self.rewrite_hint.text().strip()

        # Gather parent context
        parent_lines = []
        for pid, pnd in self._script.nodes.items():
            if self._node_id in pnd.get("next", []):
                ptxt = pnd.get("text", "").strip()
                if ptxt:
                    parent_lines.append(f"[{pid}]: {ptxt}")

        parts = []
        if parent_lines:
            parts.append("PRECEDING NODES:\n" + "\n".join(parent_lines))
        parts.append(f"CURRENT NODE [{self._node_id}]:\n{current or '(empty)'}")
        all_tags = nd.get('tags', [])
        layer_tags = set(LAYER_ORDER)
        layer_tag  = next((t for t in all_tags if t in layer_tags), "none")
        custom_tags = [t for t in all_tags if t not in layer_tags]
        parts.append(f"LAYER: {layer_tag}")
        if custom_tags:
            parts.append(f"CURRENT CUSTOM TAGS (keep those that still apply): {', '.join(custom_tags)}")
        all_script_tags = sorted({
            t for n in self._script.nodes.values()
            for t in n.get("tags", [])
            if t not in layer_tags
        })
        if all_script_tags:
            parts.append(f"EXISTING SCRIPT TAGS (prefer these when applicable): {', '.join(all_script_tags)}")
        if hint:
            parts.append(f"GUIDANCE: {hint}")
        parts.append("Rewrite the spoken text for this node based on the above context and guidance.")
        prompt = "\n\n".join(parts)

        self.rewrite_status.setText("Working...")
        self.rewrite_status.setStyleSheet("color: #cccc55; font-size: 10px;")
        node_id = self._node_id

        def on_done(data):
            new_text  = data.get("text", "").strip()
            vs        = data.get("voice_settings", {})
            new_tags  = data.get("tags", [])
            if self._script and node_id in self._script.nodes:
                self._script.update_text(node_id, new_text)
                if new_tags:
                    self._script.update_tags(node_id, new_tags)
                if vs:
                    self._script.update_node_voice_settings(node_id, {
                        "stability":        vs.get("stability",        0.5),
                        "similarity_boost": vs.get("similarity_boost", 0.75),
                        "style":            0.0,
                    })
                new_vars = data.get("vars", {})
                if new_vars:
                    self._script.nodes[node_id].setdefault("vars", {}).update(new_vars)
                    self._script.dirty = True
                if self._node_id == node_id:
                    self._blocking = True
                    try:
                        self.text_edit.setPlainText(new_text)
                        if new_tags:
                            _lt = set(LAYER_ORDER)
                            lt  = next((t for t in new_tags if t in _lt), "")
                            ct  = [t for t in new_tags if t not in _lt]
                            idx = self.layer_combo.findData(lt)
                            self.layer_combo.setCurrentIndex(idx if idx >= 0 else 0)
                            self.tags_edit.setText(", ".join(ct))
                        if vs:
                            self.stability_spin.setValue(vs.get("stability", 0.5))
                            self.similarity_spin.setValue(vs.get("similarity_boost", 0.75))
                        if new_vars:
                            for name, spin in self._vars_spins.items():
                                spin.setValue(new_vars.get(name, spin.value()))
                    finally:
                        self._blocking = False
            self.rewrite_status.setText("Done")
            self.rewrite_status.setStyleSheet("color: #88ee88; font-size: 10px;")
            self.node_modified.emit(node_id)

        def on_error(e):
            self.rewrite_status.setText(f"Error: {e[:60]}")
            self.rewrite_status.setStyleSheet("color: #ff5555; font-size: 10px;")

        self._ai.rewrite_text(prompt, self._ui_queue, on_done, on_error,
                              story_context=self._script.story_context_focused,
                              variables=self._script.variables)

    def _cmd_expand(self):
        if not self._node_id or not self._script or not self._ai or not self._ui_queue:
            return
        mn = int(self.expand_min.value())
        mx = int(self.expand_max.value())
        mx = max(mx, mn)
        self.node_modified.emit(f"__expand__{self._node_id}__{mn}__{mx}")

    def _cmd_delete(self):
        if not self._node_id:
            return
        self.node_modified.emit(f"__delete__{self._node_id}")

    def _cmd_generate_audio(self):
        if not self._node_id or not self._script or not self._vm or not self._ui_queue:
            return
        if not self._vm.api_key:
            self.audio_status.setText("Enter API key first")
            self.audio_status.setStyleSheet("color: #ffaa44; font-size: 10px;")
            return

        nd = self._script.nodes.get(self._node_id, {})
        text = _sanitize_tts(nd.get("text", "").strip())
        if not text:
            self.audio_status.setText("No text to generate")
            return

        raw      = nd.get("voice") or self._script._data.get("voice", "")
        voice_id = self._vm.id_for_name(raw) or raw
        if not voice_id:
            self.audio_status.setText("No voice selected")
            self.audio_status.setStyleSheet("color: #ffaa44; font-size: 10px;")
            return

        out_dir  = self._script.path.parent if self._script.path else SOUNDS_DIR
        out_path = out_dir / f"{self._node_id}.mp3"

        self.audio_status.setText("Generating...")
        self.audio_status.setStyleSheet("color: #cccc55; font-size: 10px;")
        node_id = self._node_id

        settings = {**self._script._data.get("voice_settings", {}),
                    **nd.get("voice_settings", {})}

        def on_done(path: Path):
            try:
                rel = str(path.relative_to(REPO_ROOT))
            except ValueError:
                rel = str(path)
            if self._script and node_id in self._script.nodes:
                self._script.nodes[node_id]["file"] = rel
                self._script.dirty = True
            if self._node_id == node_id:
                self.audio_file_edit.setText(rel)
                self.audio_status.setText(f"Saved: {path.name}")
                self.audio_status.setStyleSheet("color: #88ee88; font-size: 10px;")
            self.node_modified.emit(node_id)

        def on_error(e: str):
            self.audio_status.setText(f"Error: {e[:60]}")
            self.audio_status.setStyleSheet("color: #ff5555; font-size: 10px;")

        self._vm.generate(
            text=text, voice_id=voice_id, out_path=out_path,
            settings=settings, ui_queue=self._ui_queue,
            on_done=on_done, on_error=on_error,
        )

    def _cmd_play_node_audio(self):
        # If already playing, stop it
        if self._audio_stop_event and not self._audio_stop_event.is_set():
            self._audio_stop_event.set()
            try:
                import sounddevice as sd
                sd.stop()
            except Exception:
                pass
            self.play_audio_btn.setText("▶ Play")
            return

        if not self._node_id or not self._script:
            return
        nd = self._script.nodes.get(self._node_id, {})
        file_rel = nd.get("file")
        if not file_rel:
            self.audio_status.setText("No audio file — generate first")
            self.audio_status.setStyleSheet("color: #ffaa44; font-size: 10px;")
            return
        audio_file = REPO_ROOT / file_rel
        if not audio_file.exists():
            self.audio_status.setText(f"File missing: {file_rel}")
            self.audio_status.setStyleSheet("color: #ffaa44; font-size: 10px;")
            return

        self.audio_status.setText(f"Playing: {audio_file.name}")
        self.audio_status.setStyleSheet("color: #88ee88; font-size: 10px;")
        self._audio_stop_event = threading.Event()
        stop_event = self._audio_stop_event
        self.play_audio_btn.setText("■ Stop")

        def run():
            try:
                _play_audio_blocking(audio_file, stop_event)
            except Exception as e:
                self.audio_status.setText(f"Playback error: {e}")
                self.audio_status.setStyleSheet("color: #ff5555; font-size: 10px;")
            finally:
                if self._audio_stop_event is stop_event:
                    self._audio_stop_event = None
                    self.play_audio_btn.setText("▶ Play Node")

        threading.Thread(target=run, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# VoiceSettingsPanel
# ─────────────────────────────────────────────────────────────────────────────

class VoiceSettingsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._script: Optional[ScriptData] = None
        self._vm: Optional[VoiceManager] = None
        self._ui_queue: Optional[queue.SimpleQueue] = None
        self._props_panel: Optional[PropertiesPanel] = None
        self._build_ui()

    def set_context(self, script: ScriptData, vm: VoiceManager,
                    ui_queue: queue.SimpleQueue, props_panel: "PropertiesPanel"):
        self._script = script
        self._vm = vm
        self._ui_queue = ui_queue
        self._props_panel = props_panel
        # Pre-populate api key from vm
        self.api_key_edit.setText(vm.api_key)
        # Pre-populate default voice
        current_id = script._data.get("voice", "")
        if current_id and vm:
            name = vm.name_for_id(current_id) or current_id
            idx = self.default_voice_combo.findText(name)
            if idx >= 0:
                self.default_voice_combo.setCurrentIndex(idx)
        # Pre-populate TTS model
        saved_model = script._data.get("voice_settings", {}).get("model", "eleven_multilingual_v2")
        idx = self.tts_model_combo.findData(saved_model)
        if idx >= 0:
            self.tts_model_combo.setCurrentIndex(idx)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)

        hdr = QLabel("Voice Settings")
        hdr.setStyleSheet("font-weight: bold; font-size: 12px; color: #aaddff;")
        layout.addWidget(hdr)

        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("API Key:"))
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("sk-... or set ELEVENLABS_API_KEY env var")
        key_row.addWidget(self.api_key_edit)
        layout.addLayout(key_row)

        fetch_row = QHBoxLayout()
        fetch_btn = QPushButton("Fetch Voices")
        fetch_btn.clicked.connect(self._cmd_fetch_voices)
        fetch_row.addWidget(fetch_btn)
        self.fetch_status = QLabel("")
        self.fetch_status.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        fetch_row.addWidget(self.fetch_status)
        fetch_row.addStretch()
        layout.addLayout(fetch_row)

        voice_row = QHBoxLayout()
        voice_row.addWidget(QLabel("Default Voice:"))
        self.default_voice_combo = QComboBox()
        self.default_voice_combo.currentTextChanged.connect(self._autosave_default_voice)
        voice_row.addWidget(self.default_voice_combo)
        layout.addLayout(voice_row)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("TTS Model:"))
        self.tts_model_combo = QComboBox()
        self._tts_models = [
            ("eleven_multilingual_v2",  "Multilingual v2 (most expressive)"),
            ("eleven_turbo_v2_5",       "Turbo v2.5 (stable pacing)"),
            ("eleven_flash_v2_5",       "Flash v2.5 (fastest)"),
            ("eleven_monolingual_v1",   "Monolingual v1 (most stable)"),
            ("eleven_multilingual_v1",  "Multilingual v1"),
        ]
        for model_id, label in self._tts_models:
            self.tts_model_combo.addItem(label, model_id)
        self.tts_model_combo.currentIndexChanged.connect(self._autosave_tts_model)
        model_row.addWidget(self.tts_model_combo)
        layout.addLayout(model_row)

        self.skip_existing_chk = QCheckBox("Skip existing audio (> 1 KB)")
        self.skip_existing_chk.setChecked(True)
        layout.addWidget(self.skip_existing_chk)

        gen_all_btn = QPushButton("Generate All Audio")
        gen_all_btn.clicked.connect(self._cmd_generate_all_audio)
        layout.addWidget(gen_all_btn)

        self.gen_all_status = QLabel("")
        self.gen_all_status.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        self.gen_all_status.setWordWrap(True)
        layout.addWidget(self.gen_all_status)

    def _cmd_fetch_voices(self):
        if not self._vm or not self._ui_queue:
            return
        key = self.api_key_edit.text().strip()
        if key:
            self._vm.api_key = key
        if not self._vm.api_key:
            self.fetch_status.setText("Enter API key first")
            return
        if self._vm.fetching:
            return
        self.fetch_status.setText("Fetching...")
        self.fetch_status.setStyleSheet("color: #cccc55; font-size: 10px;")

        def on_done(voices):
            names = [n for n, _ in voices]
            self.default_voice_combo.clear()
            for n in names:
                self.default_voice_combo.addItem(n)
            # Restore current selection
            if self._script:
                current_id = self._script._data.get("voice", "")
                display = self._vm.name_for_id(current_id) or current_id
                idx = self.default_voice_combo.findText(display)
                if idx >= 0:
                    self.default_voice_combo.setCurrentIndex(idx)
            # Update props panel voice list
            if self._props_panel:
                self._props_panel.update_voice_list(names)
            self.fetch_status.setText(f"{len(voices)} voices")
            self.fetch_status.setStyleSheet("color: #88ee88; font-size: 10px;")

        def on_error(_):
            self.fetch_status.setText("Error")
            self.fetch_status.setStyleSheet("color: #ff5555; font-size: 10px;")

        self._vm.fetch_voices(self._ui_queue, on_done, on_error)

    def _cmd_generate_all_audio(self):
        if not self._script or not self._vm or not self._ui_queue:
            return
        if not self._vm.api_key:
            self.gen_all_status.setText("Enter API key first.")
            self.gen_all_status.setStyleSheet("color: #ffaa44; font-size: 10px;")
            return

        skip = self.skip_existing_chk.isChecked()
        out_dir = self._script.path.parent if self._script.path else SOUNDS_DIR

        # Build work list
        queue_list = []
        for node_id, nd in self._script.nodes.items():
            text = nd.get("text", "").strip()
            if not text:
                continue
            if skip:
                file_rel = nd.get("file")
                if file_rel:
                    p = REPO_ROOT / file_rel
                    if p.exists() and p.stat().st_size > 1024:
                        continue
            raw = nd.get("voice") or self._script._data.get("voice", "")
            voice_id = self._vm.id_for_name(raw) or raw
            if not voice_id:
                continue
            queue_list.append((node_id, nd, voice_id))

        if not queue_list:
            self.gen_all_status.setText("Nothing to generate.")
            self.gen_all_status.setStyleSheet("color: #aaaaaa; font-size: 10px;")
            return

        total = len(queue_list)
        self._gen_done = 0
        self._gen_errors = 0
        self.gen_all_status.setText(f"0 / {total}…")
        self.gen_all_status.setStyleSheet("color: #cccc55; font-size: 10px;")

        def generate_next(remaining):
            if not remaining:
                self.gen_all_status.setText(
                    f"Done — {self._gen_done} generated, {self._gen_errors} errors.")
                self.gen_all_status.setStyleSheet("color: #88ee88; font-size: 10px;")
                return

            node_id, nd, voice_id = remaining[0]
            rest = remaining[1:]
            out_path = out_dir / f"{node_id}.mp3"
            settings = {**self._script._data.get("voice_settings", {}),
                        **nd.get("voice_settings", {})}

            def on_done(path: Path):
                try:
                    rel = str(path.relative_to(REPO_ROOT))
                except ValueError:
                    rel = str(path)
                if self._script and node_id in self._script.nodes:
                    self._script.nodes[node_id]["file"] = rel
                    self._script.dirty = True
                if self._props_panel:
                    self._props_panel.node_modified.emit(node_id)
                self._gen_done += 1
                self.gen_all_status.setText(
                    f"{self._gen_done + self._gen_errors} / {total}…")
                generate_next(rest)

            def on_error(_: str):
                self._gen_errors += 1
                self.gen_all_status.setText(
                    f"{self._gen_done + self._gen_errors} / {total}… ({self._gen_errors} errors)")
                generate_next(rest)

            self._vm.generate(
                text=nd.get("text", ""), voice_id=voice_id, out_path=out_path,
                settings=settings, ui_queue=self._ui_queue,
                on_done=on_done, on_error=on_error,
            )

        generate_next(queue_list)

    def _autosave_default_voice(self, name: str):
        if not self._script or not self._vm:
            return
        voice_id = self._vm.id_for_name(name) or name
        if voice_id:
            self._script.set_default_voice(voice_id)

    def _autosave_tts_model(self):
        if not self._script:
            return
        model_id = self.tts_model_combo.currentData()
        if model_id:
            self._script._data.setdefault("voice_settings", {})["model"] = model_id
            self._script.dirty = True




# ─────────────────────────────────────────────────────────────────────────────
# AIChatPanel
# ─────────────────────────────────────────────────────────────────────────────

class AIChatPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._script: Optional[ScriptData] = None
        self._ai: Optional[AIAssistant] = None
        self._ui_queue: Optional[queue.SimpleQueue] = None
        self._on_graph_generated = None
        self._on_nodes_incremental = None  # (new_node_ids: set) -> None
        self._register_orchestrator = None
        self.selected_node_id: Optional[str] = None
        self._build_ui()

    def set_context(self, script: ScriptData, ai: AIAssistant,
                    ui_queue: queue.SimpleQueue, on_graph_generated,
                    on_nodes_incremental=None, register_orchestrator=None):
        self._script = script
        self._ai = ai
        self._ui_queue = ui_queue
        self._on_graph_generated = on_graph_generated
        self._on_nodes_incremental = on_nodes_incremental
        self._register_orchestrator = register_orchestrator

    def _build_script_context(self) -> str:
        """Return a focused context string for the AI chat prompt."""
        if not self._script:
            return ''
        node_id = self.selected_node_id
        if not node_id or node_id not in self._script.nodes:
            # No selection — just a bare structure map (IDs + tags + connections)
            lines = [f'Script: "{self._script.name}" ({len(self._script.nodes)} nodes)']
            starts = self._script.start_nodes
            if starts:
                lines.append(f'Start nodes: {", ".join(starts)}')
            for nid, nd in self._script.nodes.items():
                tags = ", ".join(nd.get("tags", [])) or "—"
                nexts = ", ".join(nd.get("next", [])) or "END"
                lines.append(f'  {nid} [{tags}] → {nexts}')
            return "\n".join(lines)

        # Node selected — show upstream path + selected node in full
        reverse = {}
        for nid, nd in self._script.nodes.items():
            for child in nd.get('next', []):
                reverse.setdefault(child, []).append(nid)

        chain = []
        current = node_id
        for _ in range(4):
            parents = reverse.get(current, [])
            if not parents:
                break
            current = parents[0]
            nd = self._script.nodes.get(current, {})
            chain.append((current, nd))
        chain.reverse()

        lines = [f'Script: "{self._script.name}"', 'Upstream path to selected node:']
        for nid, nd in chain:
            tags = ", ".join(nd.get("tags", [])) or "—"
            text = nd.get("text", "").replace("\n", " ")[:100]
            lines.append(f'  [{nid}] [{tags}] "{text}"')

        nd = self._script.nodes[node_id]
        tags = ", ".join(nd.get("tags", [])) or "—"
        lines.append(f'Selected node: {node_id} [{tags}]')
        lines.append(f'Text: "{nd.get("text", "")}"')
        nexts = ", ".join(nd.get("next", [])) or "END"
        lines.append(f'Connections out: {nexts}')
        return "\n".join(lines)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)

        hdr = QLabel("AI Chat")
        hdr.setStyleSheet("font-weight: bold; font-size: 12px; color: #aaddff;")
        layout.addWidget(hdr)

        self.chat_log = QTextEdit()
        self.chat_log.setReadOnly(True)
        self.chat_log.setMinimumHeight(100)
        self.chat_log.setMaximumHeight(200)
        self.chat_log.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        layout.addWidget(self.chat_log)

        self.chat_input = QTextEdit()
        self.chat_input.setMinimumHeight(50)
        self.chat_input.setMaximumHeight(80)
        self.chat_input.setPlaceholderText("Type a message... (Enter to send, Shift+Enter for newline)")
        self.chat_input.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self.chat_input.installEventFilter(self)
        layout.addWidget(self.chat_input)

        btn_row = QHBoxLayout()
        chat_btn = QPushButton("Chat")
        chat_btn.clicked.connect(self._cmd_chat_send)
        btn_row.addWidget(chat_btn)

        gen_btn = QPushButton("Generate Graph")
        gen_btn.setToolTip("Generate graph with parallel AI workers")
        gen_btn.clicked.connect(self._cmd_generate_parallel)
        btn_row.addWidget(gen_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._cmd_reset)
        btn_row.addWidget(reset_btn)
        layout.addLayout(btn_row)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        layout.addWidget(self.status_label)

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        if obj is self.chat_input and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                    self._cmd_chat_send()
                    return True
        return super().eventFilter(obj, event)

    def append_message(self, role: str, text: str):
        cursor = self.chat_log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        if role == "user":
            prefix_color = "#88bbff"
            prefix = "You"
        else:
            prefix_color = "#88ee88"
            prefix = "Claude"

        self.chat_log.append(f'<span style="color:{prefix_color}; font-weight:bold;">{prefix}:</span> {text}')
        self.chat_log.append("")
        self.chat_log.verticalScrollBar().setValue(
            self.chat_log.verticalScrollBar().maximum()
        )

    def _cmd_chat_send(self):
        if not self._ai or not self._ui_queue:
            return
        if not self._ai.ready:
            self.append_message("assistant",
                "[Error] claude CLI not found. Make sure Claude Code is installed: https://claude.ai/code")
            self.status_label.setText("claude not found")
            return
        if self._ai.busy:
            self.status_label.setText("AI is busy...")
            return

        msg = ' '.join(self.chat_input.toPlainText().split('\n')).strip()
        if not msg:
            return
        self.chat_input.setPlainText("")
        self.append_message("user", msg)
        self.status_label.setText("Waiting for Claude...")
        self.status_label.setStyleSheet("color: #cccc55; font-size: 10px;")

        self._ai.chat(
            msg, self._ui_queue,
            script_summary=self._build_script_context(),
            story_context=self._script.story_context_focused if self._script else '',
            on_reply=lambda r: (
                self.append_message("assistant", r),
                self.status_label.setText("Claude replied"),
                self.status_label.setStyleSheet("color: #88ee88; font-size: 10px;"),
            ),
            on_error=lambda e: (
                self.append_message("assistant", f"[Error] {e}"),
                self.status_label.setText(f"AI error: {e[:50]}"),
                self.status_label.setStyleSheet("color: #ff5555; font-size: 10px;"),
            ),
        )

    def _cmd_generate_parallel(self):
        """Generate graph using parallel orchestrator — one node per AI call."""
        if not self._ai or not self._ui_queue or not self._script:
            return
        if not self._ai.ready:
            self.status_label.setText("claude CLI not found")
            return

        prompt = ' '.join(self.chat_input.toPlainText().split('\n')).strip()
        if not prompt:
            prompt = "Generate a narrative graph based on our conversation so far."
        self.chat_input.setPlainText("")
        self.append_message("user", f"[Parallel generate]: {prompt}")
        self.status_label.setText("Generating seed nodes...")
        self.status_label.setStyleSheet("color: #cccc55; font-size: 10px;")

        def on_seed_done(data):
            before = set(self._script.nodes.keys())
            self._script.apply_generated(data)
            after = set(self._script.nodes.keys())
            seed_ids = sorted(after - before)

            if not seed_ids:
                self.status_label.setText("No seed nodes generated")
                return

            # Mark seeds as start nodes
            for nid in seed_ids:
                self._script.set_start(nid, True)

            # Add to visual graph
            if self._on_nodes_incremental:
                self._on_nodes_incremental(set(seed_ids))

            self.append_message("assistant", f"Seeds: {', '.join(seed_ids)}")
            self.status_label.setText(f"Seeds done. Starting parallel expansion...")

            # Launch orchestrator (shared factory — carries model+thinking,
            # registers the run for Stop AI Generation)
            self._orchestrator = make_full_orchestrator(
                self._script, self._ai, self._ui_queue,
                story_context=self._script.story_context,
                width_preset=self._script.width_preset,
                on_progress=lambda msg: (
                    self.status_label.setText(msg),
                    self.status_label.setStyleSheet("color: #cccc55; font-size: 10px;"),
                ),
                on_complete=lambda: (
                    self.status_label.setText(
                        f"Parallel generation complete — {len(self._script.nodes)} nodes"),
                    self.status_label.setStyleSheet("color: #88ee88; font-size: 10px;"),
                    self._on_graph_generated() if self._on_graph_generated else None,
                ),
                on_node_added=self._on_nodes_incremental,
                register=getattr(self, '_register_orchestrator', None),
            )
            self._orchestrator.start(seed_ids)

        def on_seed_error(e):
            self.status_label.setText(f"Seed error: {e[:50]}")
            self.status_label.setStyleSheet("color: #ff5555; font-size: 10px;")

        self._ai.generate_seed(
            prompt, self._ui_queue, on_seed_done, on_seed_error,
            story_context=self._script.story_context_focused if self._script else '',
            variables=self._script.variables if self._script else [],
        )

    def _cmd_reset(self):
        if self._ai:
            self._ai._history.clear()
        self.chat_log.clear()
        self.status_label.setText("Chat reset")


# ─────────────────────────────────────────────────────────────────────────────
# PlaybackPanel
# ─────────────────────────────────────────────────────────────────────────────

class PlaybackPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._script: Optional[ScriptData] = None
        self._ui_queue: Optional[queue.SimpleQueue] = None
        self._play_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._build_ui()

    def set_context(self, script: ScriptData, ui_queue: queue.SimpleQueue):
        self._script = script
        self._ui_queue = ui_queue

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)

        hdr = QLabel("Script Playback")
        hdr.setStyleSheet("font-weight: bold; font-size: 12px; color: #aaddff;")
        layout.addWidget(hdr)

        btn_row = QHBoxLayout()
        play_btn = QPushButton("▶ Play Script")
        play_btn.clicked.connect(self._cmd_play)
        btn_row.addWidget(play_btn)

        stop_btn = QPushButton("Stop")
        stop_btn.clicked.connect(self._cmd_stop)
        btn_row.addWidget(stop_btn)

        file_btn = QPushButton("Play from File...")
        file_btn.clicked.connect(self._cmd_play_file)
        btn_row.addWidget(file_btn)
        layout.addLayout(btn_row)

        delay_row = QHBoxLayout()
        delay_row.addWidget(QLabel("Delay (s):"))
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.0, 30.0)
        self.delay_spin.setSingleStep(0.5)
        self.delay_spin.setValue(3.0)
        delay_row.addWidget(self.delay_spin)
        delay_row.addStretch()
        layout.addLayout(delay_row)

        self.status_label = QLabel("Stopped")
        self.status_label.setStyleSheet("color: #999999; font-size: 10px;")
        layout.addWidget(self.status_label)

        self.current_node_label = QLabel("—")
        self.current_node_label.setStyleSheet("color: #aaddff; font-size: 10px;")
        layout.addWidget(self.current_node_label)

    def _cmd_play(self):
        if self._play_thread and self._play_thread.is_alive():
            return
        if not self._script or not self._ui_queue:
            return

        self._stop_event = threading.Event()
        stop_event = self._stop_event

        def get_delay():
            return self.delay_spin.value()

        def on_node(node_id, remaining=None):
            label = node_id
            if remaining is not None and remaining > 0:
                label = f"{node_id}  (next in {remaining:.1f}s)"
            self.current_node_label.setText(label)
            self.status_label.setText("Playing")
            self.status_label.setStyleSheet("color: #88ee88; font-size: 10px;")

        def on_finish():
            self.current_node_label.setText("—")
            self.status_label.setText("Stopped")
            self.status_label.setStyleSheet("color: #999999; font-size: 10px;")

        def on_error(msg):
            self.status_label.setText(f"Error: {msg[:60]}")
            self.status_label.setStyleSheet("color: #ff5555; font-size: 10px;")

        self._play_thread = threading.Thread(
            target=_playback_loop,
            args=(self._script, stop_event, get_delay,
                  self._ui_queue, on_node, on_finish, on_error),
            daemon=True,
        )
        self._play_thread.start()
        self.status_label.setText("Starting...")
        self.status_label.setStyleSheet("color: #cccc55; font-size: 10px;")

    def _cmd_stop(self):
        if self._stop_event:
            self._stop_event.set()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

    def _cmd_play_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Script", str(SOUNDS_DIR), "JSON files (*.json)"
        )
        if not path:
            return
        try:
            play_script = ScriptData.load(Path(path))
            # Temporarily swap script
            old_script = self._script
            self._script = play_script
            self._cmd_play()
            self._script = old_script
        except Exception as exc:
            self.status_label.setText(f"Load error: {exc}")
            self.status_label.setStyleSheet("color: #ff5555; font-size: 10px;")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _next_node_id(script: ScriptData) -> str:
    i = 1
    while f"node_{i:03d}" in script.nodes:
        i += 1
    return f"node_{i:03d}"


def _compute_layout(script: ScriptData) -> dict:
    """Return {node_id: [x, y]} using longest-path depth from root nodes.
    Root nodes (no incoming edges) go on the left; terminals go on the right."""
    nodes = script.nodes
    if not nodes:
        return {}

    children = {nid: [t for t in nd.get("next", []) if t in nodes]
                for nid, nd in nodes.items()}
    parents: dict = defaultdict(list)
    for nid, kids in children.items():
        for kid in kids:
            parents[kid].append(nid)

    # Topological sort (Kahn's) to handle DAG ordering
    in_deg = {nid: len(parents[nid]) for nid in nodes}
    q      = deque(nid for nid in nodes if in_deg[nid] == 0)
    topo: list = []
    while q:
        nid = q.popleft()
        topo.append(nid)
        for kid in children[nid]:
            in_deg[kid] -= 1
            if in_deg[kid] == 0:
                q.append(kid)
    # Append any cycle members at end
    topo.extend(nid for nid in nodes if nid not in set(topo))

    # Assign depth = longest path from any root
    depth: dict = {nid: 0 for nid in nodes}
    for nid in topo:
        for kid in children[nid]:
            depth[kid] = max(depth[kid], depth[nid] + 1)

    by_col: dict = defaultdict(list)
    for nid in nodes:
        by_col[depth[nid]].append(nid)

    LAYER_W = 220
    NODE_H_SPACING = 100
    result: dict = {}
    for col, nids in sorted(by_col.items()):
        x = 60 + col * LAYER_W
        for row, nid in enumerate(sorted(nids)):
            result[nid] = [x, 60 + row * NODE_H_SPACING]
    return result


def _layout_vertical(script: ScriptData) -> dict:
    """Top-to-bottom: roots at top, terminals at bottom, columns by depth."""
    nodes = script.nodes
    if not nodes:
        return {}
    children = {nid: [t for t in nd.get("next", []) if t in nodes]
                for nid, nd in nodes.items()}
    parents: dict = defaultdict(list)
    for nid, kids in children.items():
        for kid in kids:
            parents[kid].append(nid)
    in_deg = {nid: len(parents[nid]) for nid in nodes}
    q = deque(nid for nid in nodes if in_deg[nid] == 0)
    topo: list = []
    while q:
        nid = q.popleft(); topo.append(nid)
        for kid in children[nid]:
            in_deg[kid] -= 1
            if in_deg[kid] == 0: q.append(kid)
    topo.extend(nid for nid in nodes if nid not in set(topo))
    depth: dict = {nid: 0 for nid in nodes}
    for nid in topo:
        for kid in children[nid]:
            depth[kid] = max(depth[kid], depth[nid] + 1)
    by_row: dict = defaultdict(list)
    for nid in nodes:
        by_row[depth[nid]].append(nid)
    NODE_W, LAYER_H = 210, 180
    result: dict = {}
    for row, nids in sorted(by_row.items()):
        y = 60 + row * LAYER_H
        for col, nid in enumerate(sorted(nids)):
            result[nid] = [60 + col * NODE_W, y]
    return result


def _layout_tree(script: ScriptData) -> dict:
    """Hierarchical tree layout: roots on left, children beside parent.

    Lays out each root's subtree independently, then stacks the subtrees
    with visible gaps between them.  Within a subtree, parents are centered
    on their children.  Shared nodes (merge points reached from multiple
    roots) are assigned to their first root's subtree.
    """
    nodes = script.nodes
    if not nodes:
        return {}

    # ── Build graph ──
    children_map = {nid: [t for t in nd.get("next", []) if t in nodes]
                    for nid, nd in nodes.items()}
    parent_map: dict = defaultdict(list)
    for nid, kids in children_map.items():
        for kid in kids:
            parent_map[kid].append(nid)

    # ── Topological sort (Kahn's) ──
    in_deg = {nid: len(parent_map[nid]) for nid in nodes}
    q = deque(nid for nid in nodes if in_deg[nid] == 0)
    topo: list = []
    while q:
        nid = q.popleft()
        topo.append(nid)
        for kid in children_map[nid]:
            in_deg[kid] -= 1
            if in_deg[kid] == 0:
                q.append(kid)
    topo_set = set(topo)
    topo.extend(nid for nid in nodes if nid not in topo_set)

    # ── Assign depth (column) — longest path from any root ──
    depth: dict = {nid: 0 for nid in nodes}
    for nid in topo:
        for kid in children_map[nid]:
            depth[kid] = max(depth[kid], depth[nid] + 1)

    # ── Identify roots and partition into subtrees ──
    roots = [nid for nid in topo if not parent_map[nid]]
    if not roots:
        roots = [topo[0]]  # cycle fallback

    # BFS from each root to claim nodes into subtrees (first root wins)
    subtree_nodes: dict = {}  # root -> [nodes in topo order]
    claimed = set()
    for root in roots:
        members = []
        visit_q = deque([root])
        while visit_q:
            nid = visit_q.popleft()
            if nid in claimed or nid not in nodes:
                continue
            claimed.add(nid)
            members.append(nid)
            for kid in children_map.get(nid, []):
                if kid not in claimed:
                    visit_q.append(kid)
        if members:
            # Sort members in topo order for consistent processing
            member_set = set(members)
            subtree_nodes[root] = [n for n in topo if n in member_set]

    # Catch any orphans not reached from any root
    unclaimed = [nid for nid in topo if nid not in claimed]
    if unclaimed:
        subtree_nodes['__orphans__'] = unclaimed

    NODE_W, NODE_H = 260, 115
    MIN_GAP = 1.0        # minimum vertical slots between nodes in same column
    SUBTREE_GAP = 2.5    # extra gap between separate subtrees

    # ── Layout helper: lay out one subtree, returns y_pos dict (0-based) ──
    def _layout_subtree(member_list):
        members = set(member_list)
        sub_topo = [n for n in topo if n in members]

        # Group by depth
        sub_by_depth = defaultdict(list)
        for nid in sub_topo:
            sub_by_depth[depth[nid]].append(nid)
        sub_max_depth = max((depth[n] for n in members), default=0)

        y = {}

        # Place leaves first (deepest to shallowest)
        for d in range(sub_max_depth, -1, -1):
            col = sub_by_depth.get(d, [])
            if not col:
                continue

            # Sort by parent position (if placed) then child index
            def _key(nid):
                pars = [p for p in parent_map.get(nid, []) if p in members and p in y]
                if pars:
                    par = pars[0]
                    par_y = y[par]
                    try:
                        idx = children_map[par].index(nid)
                    except ValueError:
                        idx = 0
                    return (par_y, idx)
                return (0, sub_topo.index(nid) if nid in members else 0)

            col.sort(key=_key)

            for nid in col:
                if nid in y:
                    continue
                kids = [k for k in children_map.get(nid, []) if k in y and k in members]
                if kids:
                    # Center on children
                    y[nid] = sum(y[k] for k in kids) / len(kids)
                else:
                    # Stack at next available position
                    y[nid] = 0.0

            # Collision resolution for this column
            col.sort(key=lambda n: y.get(n, 0))
            for i in range(1, len(col)):
                min_y = y[col[i - 1]] + MIN_GAP
                if y[col[i]] < min_y:
                    y[col[i]] = min_y

        # Forward centroid pass — re-center parents on children
        for d in range(sub_max_depth + 1):
            for nid in sub_by_depth.get(d, []):
                kids = [k for k in children_map.get(nid, []) if k in y and k in members]
                if kids:
                    y[nid] = sum(y[k] for k in kids) / len(kids)

        # Collision resolution again
        for d in range(sub_max_depth + 1):
            col = sub_by_depth.get(d, [])
            col.sort(key=lambda n: y.get(n, 0))
            for i in range(1, len(col)):
                min_y = y[col[i - 1]] + MIN_GAP
                if y[col[i]] < min_y:
                    y[col[i]] = min_y

        # Backward centroid pass
        for d in range(sub_max_depth, -1, -1):
            for nid in sub_by_depth.get(d, []):
                kids = [k for k in children_map.get(nid, []) if k in y and k in members]
                if kids:
                    y[nid] = sum(y[k] for k in kids) / len(kids)

        # Final collision resolution
        for d in range(sub_max_depth + 1):
            col = sub_by_depth.get(d, [])
            col.sort(key=lambda n: y.get(n, 0))
            for i in range(1, len(col)):
                min_y = y[col[i - 1]] + MIN_GAP
                if y[col[i]] < min_y:
                    y[col[i]] = min_y

        # Normalize so minimum y is 0
        if y:
            min_val = min(y.values())
            for nid in y:
                y[nid] -= min_val

        return y

    # ── Lay out each subtree and stack with gaps ──
    y_pos: dict = {}
    y_offset = 0.0

    for root in list(subtree_nodes.keys()):
        members = subtree_nodes[root]
        sub_y = _layout_subtree(members)

        # Shift this subtree down by the current offset
        for nid, val in sub_y.items():
            y_pos[nid] = val + y_offset

        # Advance offset past this subtree
        if sub_y:
            y_offset += max(sub_y.values()) + SUBTREE_GAP

    # Place any still-unplaced nodes
    for nid in nodes:
        if nid not in y_pos:
            y_pos[nid] = y_offset
            y_offset += MIN_GAP

    result: dict = {}
    for nid in nodes:
        result[nid] = [60 + depth[nid] * NODE_W, 60 + y_pos.get(nid, 0) * NODE_H]
    return result




# ─────────────────────────────────────────────────────────────────────────────
# MainWindow
# ─────────────────────────────────────────────────────────────────────────────

class _GraphViewHoverFilter(QObject):
    """Single event filter on the NodeGraph viewer that tracks which node the
    mouse is over and fires enter/leave callbacks as it changes.
    Also intercepts right-click to fire on_right_click(node_id, global_pos)."""

    def __init__(self, get_node_at_pos, on_enter, on_leave, on_right_click=None,
                 on_delete_pipes=None, on_mouse_move=None, on_deselect=None,
                 get_selected=None, on_restore_selection=None,
                 on_marquee_select=None, on_shift_click=None,
                 viewer=None, parent=None):
        super().__init__(parent)
        self._get_node = get_node_at_pos
        self._on_enter = on_enter
        self._on_leave = on_leave
        self._on_right_click = on_right_click
        self._on_delete_pipes = on_delete_pipes
        self._on_mouse_move = on_mouse_move
        self._on_deselect = on_deselect
        self._get_selected = get_selected
        self._on_restore_selection = on_restore_selection
        self._on_marquee_select = on_marquee_select
        self._on_shift_click = on_shift_click
        self._viewer = viewer
        self._current: Optional[str] = None

        # ── Marquee (rubber-band) selection state ──
        self._marquee_origin: Optional[object] = None   # QPointF scene coords
        self._marquee_rect: Optional[QGraphicsRectItem] = None

    def _start_marquee(self, scene_pos):
        """Begin a rubber-band selection rectangle at *scene_pos*."""
        scene = self._viewer.scene() if self._viewer else None
        if not scene:
            return
        self._marquee_origin = scene_pos
        rect_item = QGraphicsRectItem(QRectF(scene_pos, scene_pos))
        rect_item.setPen(QPen(QColor(100, 150, 255), 1, Qt.PenStyle.DashLine))
        rect_item.setBrush(QBrush(QColor(100, 150, 255, 30)))
        rect_item.setZValue(999999)
        scene.addItem(rect_item)
        self._marquee_rect = rect_item

    def _update_marquee(self, scene_pos):
        """Resize the marquee rectangle to the current mouse position."""
        if self._marquee_rect and self._marquee_origin:
            self._marquee_rect.setRect(
                QRectF(self._marquee_origin, scene_pos).normalized()
            )

    def _finish_marquee(self):
        """Complete the marquee drag — select enclosed nodes and clean up."""
        if not self._marquee_rect:
            return
        final_rect = self._marquee_rect.rect()
        scene = self._marquee_rect.scene()
        if scene:
            scene.removeItem(self._marquee_rect)
        self._marquee_rect = None
        self._marquee_origin = None

        if self._on_marquee_select:
            self._on_marquee_select(final_rect)

    def _cancel_marquee(self):
        """Discard an in-progress marquee without selecting."""
        if self._marquee_rect:
            scene = self._marquee_rect.scene()
            if scene:
                scene.removeItem(self._marquee_rect)
        self._marquee_rect = None
        self._marquee_origin = None

    def eventFilter(self, obj, event):
        t = event.type()

        # ── Mouse move: update marquee if dragging, else do hover tracking ──
        if t == QEvent.Type.MouseMove:
            if self._marquee_origin is not None:
                try:
                    scene_pos = obj.parent().mapToScene(event.position().toPoint())
                    self._update_marquee(scene_pos)
                except Exception:
                    pass
                return True  # suppress other handling while dragging marquee

            if self._on_mouse_move:
                try:
                    scene_pos = obj.parent().mapToScene(event.position().toPoint())
                    self._on_mouse_move(scene_pos)
                except Exception:
                    pass
            node_id = self._get_node(event.position().toPoint())
            if node_id != self._current:
                if self._current:
                    self._on_leave(self._current)
                self._current = node_id
                if node_id:
                    self._on_enter(node_id)

        # ── Mouse press ──
        elif t == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.RightButton and self._on_right_click:
                node_id = self._get_node(event.position().toPoint())
                self._on_right_click(node_id, event.globalPosition().toPoint())
                return True  # suppress NodeGraphQt's built-in right-click menu
            if event.button() == Qt.MouseButton.LeftButton:
                node_id = self._get_node(event.position().toPoint())
                # Shift+click on a node: toggle selection without clearing others
                if node_id and event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    if self._on_shift_click:
                        self._on_shift_click(node_id)
                    return True
                if not node_id:
                    # Check if a pipe (connection line) is near the cursor
                    pipe_item = None
                    try:
                        scene_pos = obj.parent().mapToScene(event.position().toPoint())
                        # Use a small area for easier pipe clicking
                        hit_area = QRectF(scene_pos.x() - 6, scene_pos.y() - 6, 12, 12)
                        for item in obj.parent().scene().items(hit_area):
                            if 'Pipe' in type(item).__name__:
                                pipe_item = item
                                break
                    except Exception:
                        pass
                    if pipe_item:
                        # Clear other selections and toggle this pipe
                        for it in obj.parent().scene().selectedItems():
                            it.setSelected(False)
                        pipe_item.setSelected(True)
                        return True
                    # Empty canvas — start marquee selection
                    try:
                        scene_pos = obj.parent().mapToScene(event.position().toPoint())
                        self._start_marquee(scene_pos)
                    except Exception:
                        pass
                    return True
            if event.button() == Qt.MouseButton.MiddleButton:
                node_id = self._get_node(event.position().toPoint())
                if not node_id and self._on_restore_selection and self._get_selected:
                    # Capture the selected node ID NOW before Qt clears selection,
                    # then restore it after the middle-click event is fully processed.
                    nid = self._get_selected()
                    if nid:
                        QTimer.singleShot(0, lambda _nid=nid: self._on_restore_selection(_nid))
            if self._current:
                self._on_leave(self._current)
                self._current = None

        # ── Mouse release: finish marquee if active ──
        elif t == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton and self._marquee_origin is not None:
                self._finish_marquee()
                return True

        elif t == QEvent.Type.MouseButtonDblClick:
            if event.button() == Qt.MouseButton.LeftButton:
                node_id = self._get_node(event.position().toPoint())
                if not node_id and self._on_deselect:
                    self._on_deselect()
        elif t == QEvent.Type.Leave:
            if self._current:
                self._on_leave(self._current)
                self._current = None
            self._cancel_marquee()
        elif t == QEvent.Type.KeyPress and self._on_delete_pipes:
            from PySide6.QtCore import Qt as _Qt
            if event.key() in (_Qt.Key.Key_Delete, _Qt.Key.Key_Backspace):
                if self._on_delete_pipes():
                    return True  # consumed — don't let NodeGraphQt delete nodes
        return False


class CodexDialog(QDialog):
    """Modeless codex browser/editor — the v2 home of recurring concepts.

    Left: kind filter + entity list (usage counts). Right: the card form.
    Bottom-left: New / Delete / Import… / Extract (AI)….
    Spotlight pushes the entity slug into the main window's search bar
    (whole-word, all-fields scope) so its nodes light up on the graph.
    """

    def __init__(self, main_window, script: 'ScriptData', ai: 'AIAssistant',
                 ui_queue: queue.SimpleQueue, embedded: bool = False,
                 parent=None):
        super().__init__(parent)
        self._mw = main_window
        self.script = script
        self.ai = ai
        self.ui_queue = ui_queue
        self._embedded = embedded
        self._current_slug: Optional[str] = None
        self._loading = False          # guard: form population != user edits
        # Separate AI instance for entity chat so its transcript doesn't mix
        # with the main chat panel's. Its _history is re-seeded from the
        # entity's persisted chat_history whenever the selection changes.
        self._codex_ai = AIAssistant()
        self._call_start_time: Optional[float] = None
        self.setWindowTitle("Codex — Entities")
        if not embedded:
            self.setWindowFlags(self.windowFlags() | Qt.Window)
            self.resize(940, 860)
        self._build_ui()
        self._refresh_list()
        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._tick_thinking_status)
        self._tick_timer.start(250)

    def set_script(self, script: 'ScriptData'):
        """Re-point at a new ScriptData after the main window loads a file."""
        self.script = script
        self._current_slug = None
        self._clear_form()
        self._refresh_list()

    # Embedded guard: Escape must not blank the tab (QDialog.reject hides).
    def accept(self):
        if not self._embedded:
            super().accept()

    def reject(self):
        if not self._embedded:
            super().reject()

    # ── UI construction ─────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        root = QHBoxLayout()
        outer.addLayout(root, stretch=3)

        # Left column — filter, list, actions
        left = QVBoxLayout()
        self._kind_filter = QComboBox()
        self._kind_filter.addItem("All kinds", None)
        for k in ENTITY_KINDS:
            self._kind_filter.addItem(k.capitalize(), k)
        self._kind_filter.currentIndexChanged.connect(self._refresh_list)
        left.addWidget(self._kind_filter)

        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_row_selected)
        self._list.setMinimumWidth(280)
        left.addWidget(self._list, stretch=1)

        row1 = QHBoxLayout()
        btn_new = QPushButton("New Entity")
        btn_new.clicked.connect(self._cmd_new_entity)
        row1.addWidget(btn_new)
        self._btn_delete = QPushButton("Delete")
        self._btn_delete.clicked.connect(self._cmd_delete_entity)
        row1.addWidget(self._btn_delete)
        left.addLayout(row1)

        row2 = QHBoxLayout()
        btn_import = QPushButton("Import…")
        btn_import.setToolTip(
            "Copy entity cards from another script.json's codex.\n"
            "Copy-by-value: this file stays self-contained. Slugs are kept\n"
            "identical (they are the tag linkage); existing slugs are skipped.")
        btn_import.clicked.connect(self._cmd_import)
        row2.addWidget(btn_import)
        self._btn_extract = QPushButton("Extract (AI)…")
        self._btn_extract.setToolTip(
            "Ask the AI to propose entity cards from the world bible and the\n"
            "node graph's tags. You review and check which ones to keep.")
        self._btn_extract.clicked.connect(self._cmd_extract)
        row2.addWidget(self._btn_extract)
        left.addLayout(row2)

        self._status = QLabel("")
        self._status.setStyleSheet("color: #999; font-size: 11px;")
        self._status.setWordWrap(True)
        left.addWidget(self._status)

        root.addLayout(left)

        # Right column — the card form
        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setContentsMargins(8, 4, 4, 4)

        self._f_name = QLineEdit()
        self._f_name.editingFinished.connect(self._on_name_edited)
        form.addRow("Name", self._f_name)

        self._f_slug = QLineEdit()
        self._f_slug.setToolTip(
            "The entity's slug — doubles as the node tag that links nodes to\n"
            "this card. Renaming here also renames the tag on every node.")
        self._f_slug.editingFinished.connect(self._on_slug_edited)
        form.addRow("Slug (tag)", self._f_slug)

        self._f_kind = QComboBox()
        for k in ENTITY_KINDS:
            self._f_kind.addItem(k.capitalize(), k)
        self._f_kind.currentIndexChanged.connect(self._save_current)
        form.addRow("Kind", self._f_kind)

        self._f_aliases = QLineEdit()
        self._f_aliases.setPlaceholderText("comma, separated, names")
        self._f_aliases.editingFinished.connect(self._save_current)
        form.addRow("Aliases", self._f_aliases)

        def _text_edit(height, placeholder=''):
            te = QTextEdit()
            te.setAcceptRichText(False)
            te.setFixedHeight(height)
            if placeholder:
                te.setPlaceholderText(placeholder)
            te.textChanged.connect(self._save_current)
            return te

        self._f_essence = _text_edit(
            70, "1-3 sentences: the irreducible core, written to steer prose.")
        form.addRow("Essence", self._f_essence)

        self._f_facts = _text_edit(
            90, "One durable canon fact per line — must stay true in every story.")
        form.addRow("Facts", self._f_facts)

        self._f_sensory = _text_edit(
            56, "One sensory anchor per line (a smell, a sound, a texture).")
        form.addRow("Sensory", self._f_sensory)

        self._f_voice = _text_edit(
            56, "How the narrator handles this entity (register, distance).")
        form.addRow("Voice notes", self._f_voice)

        # Relationships as REAL links: each row picks an existing codex
        # entity (no free-form slugs), with a jump button to its card.
        rels_widget = QWidget()
        rels_v = QVBoxLayout(rels_widget)
        rels_v.setContentsMargins(0, 0, 0, 0)
        rels_v.setSpacing(2)
        self._rels_box = QVBoxLayout()
        self._rels_box.setSpacing(2)
        self._rel_rows: list = []   # [(row_widget, target_combo, nature_edit)]
        rels_v.addLayout(self._rels_box)
        add_rel_btn = QPushButton("+ Add link")
        add_rel_btn.setFixedWidth(80)
        add_rel_btn.clicked.connect(self._cmd_add_rel_row)
        rels_v.addWidget(add_rel_btn)
        form.addRow("Links to", rels_widget)

        self._incoming_list = QListWidget()
        self._incoming_list.setFixedHeight(84)
        self._incoming_list.setToolTip(
            "Entities whose cards link TO this one. → jumps to the source; "
            "⇄ adds a reciprocal 'Links to' line back to it.")
        form.addRow("Linked from", self._incoming_list)

        self._f_notes = _text_edit(
            56, "Author-only notes — never sent to the AI.")
        form.addRow("Notes", self._f_notes)

        usage_row = QHBoxLayout()
        self._usage_label = QLabel("")
        self._usage_label.setStyleSheet("color: #999;")
        usage_row.addWidget(self._usage_label, stretch=1)
        self._btn_spotlight = QPushButton("Spotlight on graph")
        self._btn_spotlight.setToolTip(
            "Highlight every node carrying this entity's tag in the main window.")
        self._btn_spotlight.clicked.connect(self._cmd_spotlight)
        usage_row.addWidget(self._btn_spotlight)
        form.addRow("Usage", usage_row)

        self._source_label = QLabel("")
        self._source_label.setStyleSheet("color: #777; font-size: 11px;")
        form.addRow("Source", self._source_label)

        root.addWidget(form_widget, stretch=1)

        # ── Entity development chat (full width, below list + card) ──────
        chat_hdr = QLabel("Entity Development Chat")
        chat_hdr.setStyleSheet("font-weight: bold; margin-top: 4px;")
        outer.addWidget(chat_hdr)

        self.chat_log = QTextEdit()
        self.chat_log.setReadOnly(True)
        self.chat_log.setStyleSheet(
            "background:#1a1a1a; color:#cccccc; font-size:11px;")
        outer.addWidget(self.chat_log, stretch=2)

        input_row = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText(
            "Develop the selected entity with AI — history is saved per card…")
        self.chat_input.returnPressed.connect(self._cmd_codex_chat)
        input_row.addWidget(self.chat_input)
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._cmd_codex_chat)
        input_row.addWidget(send_btn)
        self._distill_btn = QPushButton("Distill Chat → Entity")
        self._distill_btn.setToolTip(
            "Distill this conversation onto the selected entity's card\n"
            "(refines existing content; also proposes any NEW entities the\n"
            "chat surfaced, which you review before they're added).")
        self._distill_btn.clicked.connect(self._cmd_distill_entity)
        input_row.addWidget(self._distill_btn)
        outer.addLayout(input_row)

        self.chat_status = QLabel("")
        self.chat_status.setStyleSheet("color:#888888; font-size:10px;")
        outer.addWidget(self.chat_status)

    # ── List handling ────────────────────────────────────────────────────

    def _visible_slugs(self) -> list:
        kind = self._kind_filter.currentData()
        def _order(kv):
            k = kv[1].get('kind', 'idea')
            ki = ENTITY_KINDS.index(k) if k in ENTITY_KINDS else 99
            return (ki, (kv[1].get('name') or kv[0]).lower())
        return [slug for slug, card in sorted(self.script.entities.items(), key=_order)
                if kind is None or card.get('kind') == kind]

    def _refresh_list(self, *_):
        self._loading = True
        try:
            self._list.clear()
            for slug in self._visible_slugs():
                card = self.script.entities[slug]
                n_use = len(self.script.entity_usage(slug))
                kind = card.get('kind', 'idea')
                item = QListWidgetItem(
                    f"[{kind[:4]}] {card.get('name') or slug}   ·{n_use}")
                item.setData(Qt.UserRole, slug)
                item.setToolTip(f"{slug} — used by {n_use} node(s)")
                self._list.addItem(item)
        finally:
            self._loading = False
        # restore selection if the current slug is still visible
        if self._current_slug:
            for i in range(self._list.count()):
                if self._list.item(i).data(Qt.UserRole) == self._current_slug:
                    self._list.setCurrentRow(i)
                    return
        if self._list.count():
            self._list.setCurrentRow(0)
        else:
            self._current_slug = None
            self._clear_form()

    def _on_row_selected(self, row: int):
        if self._loading or row < 0:
            return
        item = self._list.item(row)
        if not item:
            return
        slug = item.data(Qt.UserRole)
        # CRASH RULE: _load_card rebuilds relationship rows + the incoming
        # list; rebuilding item views synchronously inside currentRowChanged
        # dispatch access-violates on this machine (same family as the
        # Stories-tab cast_list AV, crash.log 2026-07-17 19:09). Defer; the
        # guard drops the load if the selection moved again first.
        QTimer.singleShot(0, lambda s=slug: self._load_card_deferred(s))

    def _load_card_deferred(self, slug: str):
        item = self._list.currentItem()
        if item and item.data(Qt.UserRole) == slug \
                and slug in self.script.entities:
            self._load_card(slug)

    # ── Form handling ────────────────────────────────────────────────────

    def _clear_form(self):
        self._loading = True
        try:
            for w in (self._f_name, self._f_slug, self._f_aliases):
                w.clear()
            for w in (self._f_essence, self._f_facts, self._f_sensory,
                      self._f_voice, self._f_notes):
                w.clear()
            self._set_rel_rows([])
            self._incoming_list.clear()
            self._usage_label.setText("")
            self._source_label.setText("")
            self.chat_log.clear()
            self._codex_ai._history = []
        finally:
            self._loading = False

    def _load_card(self, slug: str):
        QToolTip.hideText()   # never rebuild widgets under an open tooltip
        card = self.script.entities.get(slug)
        if card is None:
            return
        self._current_slug = slug
        self._loading = True
        try:
            self._f_name.setText(card.get('name', ''))
            self._f_slug.setText(slug)
            kind = card.get('kind', 'idea')
            idx = ENTITY_KINDS.index(kind) if kind in ENTITY_KINDS else 0
            self._f_kind.setCurrentIndex(idx)
            self._f_aliases.setText(", ".join(card.get('aliases', [])))
            self._f_essence.setPlainText(card.get('essence', ''))
            self._f_facts.setPlainText("\n".join(card.get('facts', [])))
            self._f_sensory.setPlainText("\n".join(card.get('sensory', [])))
            self._f_voice.setPlainText(card.get('voice_notes', ''))
            self._set_rel_rows(card.get('relationships', []))
            self._refresh_incoming(slug)
            self._f_notes.setPlainText(card.get('notes', ''))
            n_use = len(self.script.entity_usage(slug))
            self._usage_label.setText(f"Used by {n_use} node(s)")
            self._source_label.setText(card.get('source', 'local'))
            # Per-entity chat: repopulate the log and re-seed the chat AI's
            # transcript so the conversation follows the selected card.
            self.chat_log.clear()
            history = card.get('chat_history', [])
            for entry in history:
                self._append_chat(entry.get('role', 'user'),
                                  entry.get('content', ''))
            self._codex_ai._history = list(history)
        finally:
            self._loading = False

    @staticmethod
    def _parse_lines(text: str) -> list:
        return [ln.strip() for ln in text.splitlines() if ln.strip()]

    # ── Relationship rows (real links, not free text) ────────────────────

    def _fill_rel_combo(self, combo: QComboBox, selected: str):
        """Populate a target picker with every OTHER codex entity. A saved
        target that no longer exists (e.g. arrived via import) is kept as
        an explicit '(missing)' item so the reference isn't silently lost."""
        combo.blockSignals(True)
        try:
            combo.clear()
            def _order(kv):
                kind = kv[1].get('kind', 'idea')
                ki = ENTITY_KINDS.index(kind) if kind in ENTITY_KINDS else 99
                return (ki, (kv[1].get('name') or kv[0]).lower())
            for slug, card in sorted(self.script.entities.items(), key=_order):
                if slug == self._current_slug:
                    continue
                combo.addItem(
                    f"[{card.get('kind', 'idea')[:4]}] "
                    f"{card.get('name') or slug}", slug)
            if selected and combo.findData(selected) < 0:
                combo.addItem(f"(missing) {selected}", selected)
            idx = combo.findData(selected) if selected else -1
            combo.setCurrentIndex(idx if idx >= 0 else max(0, combo.count() - 1)
                                  if combo.count() else -1)
            if not selected and combo.count():
                combo.setCurrentIndex(0)
        finally:
            combo.blockSignals(False)

    def _add_rel_row(self, to_slug: str = '', nature: str = ''):
        row_w = QWidget()
        h = QHBoxLayout(row_w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        combo = QComboBox()
        combo.setMinimumWidth(140)
        self._fill_rel_combo(combo, to_slug)
        combo.currentIndexChanged.connect(self._save_current)
        h.addWidget(combo, stretch=2)
        nat = QLineEdit(nature)
        nat.setPlaceholderText("nature of the link")
        nat.editingFinished.connect(self._save_current)
        h.addWidget(nat, stretch=3)
        go = QPushButton("→")
        go.setFixedWidth(24)
        go.setToolTip("Go to this entity's card")
        go.clicked.connect(lambda *_, c=combo: self._goto_entity(c.currentData()))
        h.addWidget(go)
        rm = QPushButton("✕")
        rm.setFixedWidth(24)
        rm.clicked.connect(lambda *_, w=row_w: self._remove_rel_row(w))
        h.addWidget(rm)
        self._rels_box.addWidget(row_w)
        self._rel_rows.append((row_w, combo, nat))

    def _cmd_add_rel_row(self):
        if not self._current_slug:
            return
        if len(self.script.entities) < 2:
            self.chat_status.setText(
                "Nothing to link to yet — the codex has no other entities.")
            return
        self._add_rel_row()
        self._save_current()

    def _remove_rel_row(self, row_w):
        for i, (w, _c, _n) in enumerate(self._rel_rows):
            if w is row_w:
                self._rel_rows.pop(i)
                self._rels_box.removeWidget(row_w)
                row_w.deleteLater()
                break
        self._save_current()

    def _set_rel_rows(self, rels: list):
        prev = self._loading
        self._loading = True
        try:
            for (w, _c, _n) in self._rel_rows:
                self._rels_box.removeWidget(w)
                w.deleteLater()
            self._rel_rows = []
            for r in rels:
                if isinstance(r, dict) and r.get('to'):
                    self._add_rel_row(r['to'], r.get('nature', ''))
        finally:
            self._loading = prev

    def _rels_from_rows(self) -> list:
        rels = []
        for (_w, combo, nat) in self._rel_rows:
            to = combo.currentData()
            if to and to != self._current_slug:
                rels.append({'to': to, 'nature': nat.text().strip()})
        return rels

    def _refresh_incoming(self, slug: str):
        """List every entity whose card links TO this one — each row has a
        jump button and a ⇄ button that adds the reciprocal 'Links to'
        line, so two-way relationships take one click instead of a trip
        to the other card."""
        self._incoming_list.clear()
        for src, card in self.script.entities.items():
            if src == slug:
                continue
            for r in card.get('relationships', []):
                if r.get('to') == slug:
                    label_text = (card.get('name') or src) \
                        + (f" — {r.get('nature')}" if r.get('nature') else '')
                    item = QListWidgetItem()
                    item.setData(Qt.ItemDataRole.UserRole, src)
                    row = QWidget()
                    h = QHBoxLayout(row)
                    h.setContentsMargins(4, 0, 4, 0)
                    h.setSpacing(4)
                    lbl = QLabel(label_text)
                    h.addWidget(lbl, stretch=1)
                    go = QPushButton("→")
                    go.setFixedWidth(24)
                    go.setToolTip("Go to this entity's card")
                    go.clicked.connect(
                        lambda *_, s2=src: self._goto_entity(s2))
                    h.addWidget(go)
                    back = QPushButton("⇄")
                    back.setFixedWidth(24)
                    back.setToolTip(
                        f"Add a reciprocal link: this card → "
                        f"{card.get('name') or src}")
                    back.clicked.connect(
                        lambda *_, s2=src: self._cmd_link_back(s2))
                    h.addWidget(back)
                    item.setSizeHint(row.sizeHint())
                    self._incoming_list.addItem(item)
                    self._incoming_list.setItemWidget(item, row)

    def _cmd_link_back(self, src_slug: str):
        """⇄ on a 'Linked from' row: add a 'Links to' line pointing back
        at the source entity (nature left blank for the author to fill —
        the reverse direction usually reads differently)."""
        if not self._current_slug or src_slug not in self.script.entities:
            return
        if any(combo.currentData() == src_slug
               for (_w, combo, _n) in self._rel_rows):
            self.chat_status.setText(
                f"Already links to '{src_slug}' — edit that row's nature instead.")
            return
        self._add_rel_row(src_slug, '')
        self._save_current()
        name = self.script.entities[src_slug].get('name') or src_slug
        self.chat_status.setText(
            f"Linked back to '{name}' — describe the nature of the link.")
        self._rel_rows[-1][2].setFocus()   # cursor into the nature field

    def _goto_entity(self, slug: Optional[str]):
        """Jump the browser + card form to another entity."""
        if not slug or slug not in self.script.entities:
            self.chat_status.setText(f"'{slug}' is not in the codex.")
            return
        self._save_current()
        self._current_slug = slug
        kind = self.script.entities[slug].get('kind')
        if self._kind_filter.currentData() not in (None, kind):
            self._kind_filter.setCurrentIndex(0)   # All kinds → refreshes
        else:
            self._refresh_list()

    def _save_current(self, *_):
        if self._loading or not self._current_slug:
            return
        self.script.update_entity(self._current_slug, {
            'name': self._f_name.text().strip(),
            'kind': self._f_kind.currentData(),
            'aliases': [a.strip() for a in self._f_aliases.text().split(',')
                        if a.strip()],
            'essence': self._f_essence.toPlainText().strip(),
            'facts': self._parse_lines(self._f_facts.toPlainText()),
            'sensory': self._parse_lines(self._f_sensory.toPlainText()),
            'voice_notes': self._f_voice.toPlainText().strip(),
            'relationships': self._rels_from_rows(),
            'notes': self._f_notes.toPlainText().strip(),
        })

    def _on_name_edited(self):
        if self._loading or not self._current_slug:
            return
        self._save_current()
        # Auto-follow: a card whose slug is still the 'new_entity'
        # placeholder tracks the name — even when nodes already carry the
        # tag (rename_entity cascades node tags safely), because a stuck
        # placeholder slug makes spotlight/prompts search for 'new_entity'.
        slug = self._current_slug
        if slug.startswith('new_entity'):
            want = ScriptData.sanitize_entity_slug(self._f_name.text())
            if want and want != slug and not want.startswith('new_entity'):
                final = self.script.rename_entity(slug, want)
                if final:
                    self._current_slug = final
                    self._loading = True
                    try:
                        self._f_slug.setText(final)
                    finally:
                        self._loading = False
                    self._refresh_list()

    def _on_slug_edited(self):
        if self._loading or not self._current_slug:
            return
        new = self._f_slug.text().strip()
        if not new or new == self._current_slug:
            return
        final = self.script.rename_entity(self._current_slug, new,
                                          update_tags=True)
        if final:
            self._current_slug = final
            self._status.setText(
                f"Renamed to '{final}' (node tags updated).")
            self._refresh_list()
            self._mw._rebuild_graph()
        else:
            # invalid / collision-only rename — restore the real slug
            self._f_slug.setText(self._current_slug)

    # ── Commands ─────────────────────────────────────────────────────────

    def _cmd_new_entity(self):
        slug = self.script.add_entity("New Entity", kind='character')
        if slug:
            self._current_slug = slug
            self._refresh_list()
            self._f_name.setFocus()
            self._f_name.selectAll()

    def _cmd_delete_entity(self):
        if not self._current_slug:
            return
        # Two-step inline confirm: first click arms, second click deletes.
        if self._btn_delete.text() != "Really delete?":
            self._btn_delete.setText("Really delete?")
            QTimer.singleShot(2500, lambda: self._btn_delete.setText("Delete"))
            return
        self._btn_delete.setText("Delete")
        slug = self._current_slug
        self.script.delete_entity(slug)
        self._current_slug = None
        self._status.setText(f"Deleted '{slug}' (node tags left untouched).")
        self._refresh_list()

    def _cmd_spotlight(self):
        if not self._current_slug:
            return
        # Rescue a card still carrying the 'new_entity' placeholder slug
        # (e.g. from an older session): give it its real slug first so the
        # spotlight searches something meaningful.
        slug = self._current_slug
        if slug.startswith('new_entity'):
            want = ScriptData.sanitize_entity_slug(self._f_name.text())
            if want and want != slug and not want.startswith('new_entity'):
                final = self.script.rename_entity(slug, want)
                if final:
                    self._current_slug = slug = final
                    self._refresh_list()
        self._mw._text_only_btn.setChecked(False)
        self._mw._word_btn.setChecked(True)
        self._mw._search_bar.setText(slug)
        # The search overlays live on the Graph tab — go there so the
        # spotlight is actually visible.
        self._mw.tabs.setCurrentIndex(0)
        n = len(self.script.entity_usage(slug))
        self._mw.status_bar.showMessage(
            f"Spotlight: '{slug}' — {n} node(s) carry the tag "
            "(glowing on the graph; clear the search box to dismiss).")

    # ── Entity development chat ──────────────────────────────────────────

    def _append_chat(self, role: str, text: str):
        color = '#88ccff' if role == 'assistant' else '#cccccc'
        label = 'Claude' if role == 'assistant' else 'You'
        self.chat_log.append(
            f'<span style="color:{color};"><b>{label}:</b> {text}</span><br>')

    def _tick_thinking_status(self):
        if self._call_start_time is None:
            return
        elapsed = time.time() - self._call_start_time
        name = _model_short_name(getattr(self, '_call_model', ''))
        self.chat_status.setText(f"Claude ({name}) is thinking… {elapsed:0.0f}s")

    def _chat_context(self) -> str:
        """Bible + full codex + the selected card's usage snippets — passed
        as the chat call's story_context (the cached system-prompt block)."""
        parts = []
        if self.script.story_context:
            parts.append(f'STORY CONTEXT:\n{self.script.story_context}')
        codex_sum = self.script.codex_summary()
        if codex_sum:
            parts.append(f'FULL CODEX (current entity cards):\n{codex_sum}')
        slug = self._current_slug
        if slug:
            lines = [f'CURRENTLY SELECTED ENTITY: {slug}']
            usage = self.script.entity_usage(slug)
            for nid in usage[:3]:
                text = self.script.nodes.get(nid, {}).get('text', '')
                if text:
                    lines.append(f'  used in [{nid}]: "{text[:200]}"')
            if usage:
                lines.append(f'  ({len(usage)} node(s) total)')
            parts.append('\n'.join(lines))
        return '\n\n'.join(parts)

    def _cmd_codex_chat(self):
        msg = self.chat_input.text().strip()
        if not msg:
            return
        if not self._current_slug:
            self.chat_status.setText("Select (or create) an entity first — "
                                     "the chat develops one card at a time.")
            return
        if not self._codex_ai.ready:
            self.chat_status.setText("claude CLI not found")
            return
        if self._codex_ai.busy:
            self.chat_status.setText("AI is busy…")
            return
        self.chat_input.clear()
        self._append_chat('user', msg)
        slug = self._current_slug
        self._call_start_time = time.time()
        self._call_model = self.ai.model
        self._tick_thinking_status()

        def on_reply(reply):
            elapsed = (time.time() - self._call_start_time) \
                if self._call_start_time else 0.0
            self._call_start_time = None
            self._append_chat('assistant', reply)
            self.chat_status.setText(f'(last call: {elapsed:.1f}s)')
            card = self.script.entities.get(slug)
            if card is not None:
                hist = card.setdefault('chat_history', [])
                hist.append({'role': 'user', 'content': msg})
                hist.append({'role': 'assistant', 'content': reply})
                self.script.dirty = True

        def on_error(e):
            self._call_start_time = None
            self.chat_status.setText(f'Error: {str(e)[:80]}')

        # The chat follows the AI Model menu selection; only the one-shot
        # "Distill Chat → Entity" pass (which writes canon) pins Opus.
        self._codex_ai.chat(msg, self.ui_queue,
                            on_reply=on_reply, on_error=on_error,
                            story_context=self._chat_context(),
                            _system_override=SYSTEM_CODEX_CHAT,
                            model_override=self.ai.model)

    def _cmd_distill_entity(self):
        if not self._current_slug:
            self.chat_status.setText("Select an entity first.")
            return
        if not self.ai.ready:
            self.chat_status.setText("claude CLI not found")
            return
        slug = self._current_slug
        card = self.script.entities.get(slug, {})
        history = card.get('chat_history', [])
        if not history:
            self.chat_status.setText("No chat history to distill.")
            return
        self._save_current()

        conv_lines = []
        for entry in history:
            who = 'Author' if entry.get('role') == 'user' else 'Claude'
            conv_lines.append(f'{who}: {entry.get("content", "")}')
        parts = [
            f'CURRENT CARD (slug: {slug}):\n'
            + (self.script.codex_summary([slug]) or '(empty card)'),
            'EXISTING CODEX SLUGS (do not re-propose): '
            + (', '.join(sorted(self.script.entities.keys())) or '(none)'),
            'CONVERSATION TO DISTILL:\n' + '\n'.join(conv_lines),
            'Distill the card now. Output the JSON object only.',
        ]
        prompt = '\n\n'.join(parts)

        self._call_start_time = time.time()
        self._call_model = MODEL_OPUS
        self._tick_thinking_status()
        self._append_chat('assistant', '[Distilling conversation onto the card...]')
        self._distill_btn.setEnabled(False)

        def run():
            try:
                # Opus: distillation writes canon.
                data = self.ai._run_claude_json(SYSTEM_DISTILL_ENTITY, prompt,
                                                model_override=MODEL_OPUS)
                self.ui_queue.put(lambda d=data, s=slug:
                                  self._apply_entity_distill(s, d))
            except Exception as exc:
                def fail(e=str(exc)):
                    self._call_start_time = None
                    self._distill_btn.setEnabled(True)
                    self.chat_status.setText(f"Distill error: {e[:80]}")
                self.ui_queue.put(fail)

        threading.Thread(target=run, daemon=True).start()

    def _apply_entity_distill(self, slug: str, data: dict):
        """Apply a distill result: update the card, then review any new
        related entities the chat surfaced."""
        elapsed = (time.time() - self._call_start_time) \
            if self._call_start_time else 0.0
        self._call_start_time = None
        self._distill_btn.setEnabled(True)
        if not isinstance(data, dict) or slug not in self.script.entities:
            self.chat_status.setText("Distill failed: invalid response.")
            return

        clean = _sanitize_entity_card(data.get('entity') or {})
        # Refine-not-discard: only overwrite fields the distill filled
        # (an empty field in the response leaves the card's value alone).
        updates = {k: v for k, v in clean.items() if v}
        self.script.update_entity(slug, updates)

        # Related entities → review picker (never overwrite existing slugs)
        related = data.get('related_entities')
        n_new = 0
        if isinstance(related, dict):
            fresh = {}
            for rslug, rcard in related.items():
                if not isinstance(rcard, dict):
                    continue
                rslug = ScriptData.sanitize_entity_slug(str(rslug))
                if not rslug or rslug in RESERVED_ENTITY_SLUGS:
                    continue
                rclean = _sanitize_entity_card(rcard)
                # Loose duplicate check (name/alias/article variants), not
                # just exact slug — see find_equivalent_entity.
                if self.script.find_equivalent_entity(rslug) or \
                        self.script.find_equivalent_entity(rclean['name']):
                    continue
                if not rclean['name']:
                    rclean['name'] = rslug
                fresh[rslug] = rclean
            if fresh:
                entries = [(s, f'[{c["kind"][:4]}] {c["name"]}  ({s}) — '
                               f'{c["essence"][:80]}')
                           for s, c in fresh.items()]
                for rslug in self._pick_entities_from_distill(entries):
                    new_card = deepcopy(ENTITY_TEMPLATE)
                    new_card.update(fresh[rslug])
                    self.script.entities[rslug] = new_card
                    n_new += 1
                if n_new:
                    self.script.dirty = True

        self._load_card(slug)
        self._refresh_list()
        extra = f", +{n_new} new entit{'y' if n_new == 1 else 'ies'}" if n_new else ''
        self.chat_status.setText(
            f"Card updated from chat{extra}. ({elapsed:.1f}s)")
        self._append_chat('assistant', f'[Distilled onto "{slug}"{extra}]')

    def _pick_entities_from_distill(self, entries: list) -> list:
        return _pick_checked(self, "New entities surfaced by the chat", entries)

    # ── Import ───────────────────────────────────────────────────────────

    def _cmd_import(self):
        start_dir = str(self.script.path.parent.parent) if self.script.path \
            else str(SOUNDS_DIR)
        path, _ = QFileDialog.getOpenFileName(
            self, "Import entities from script", start_dir,
            "Script JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding='utf-8'))
        except Exception as exc:
            self._status.setText(f"Could not read {Path(path).name}: {exc}")
            return
        pool = data.get('entities', {})
        if not pool:
            self._status.setText(
                f"{Path(path).name} has no codex (no 'entities' section).")
            return
        chosen = self._pick_entities_dialog(
            f"Import from {Path(path).parent.name}", pool,
            disabled=set(self.script.entities.keys()))
        if not chosen:
            return
        imported = self.script.import_entities_from(Path(path), chosen)
        self._status.setText(
            f"Imported {len(imported)} entit{'y' if len(imported) == 1 else 'ies'} "
            f"from {Path(path).parent.name}.")
        self._refresh_list()

    def _pick_entities_dialog(self, title: str, pool: dict,
                              disabled: set = frozenset()) -> list:
        """Checkable list of candidate cards; returns the checked slugs.
        Slugs in `disabled` are shown greyed-out and unchecked (already
        present — never overwritten)."""
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(560, 480)
        vbox = QVBoxLayout(dlg)
        info = QLabel("Check the entities to bring in. Greyed entries "
                      "already exist in this codex and are skipped.")
        info.setWordWrap(True)
        vbox.addWidget(info)
        lst = QListWidget()
        def _order(kv):
            k = kv[1].get('kind', 'idea') if isinstance(kv[1], dict) else 'idea'
            ki = ENTITY_KINDS.index(k) if k in ENTITY_KINDS else 99
            return (ki, kv[0])
        for slug, card in sorted(pool.items(), key=_order):
            if not isinstance(card, dict):
                continue
            kind = card.get('kind', 'idea')
            essence = (card.get('essence', '') or '')[:110]
            item = QListWidgetItem(
                f"[{kind[:4]}] {card.get('name') or slug}  ({slug})"
                + (f"\n      {essence}" if essence else ""))
            item.setData(Qt.UserRole, slug)
            if slug in disabled or slug in RESERVED_ENTITY_SLUGS:
                item.setFlags(Qt.ItemIsUserCheckable)   # visible, not selectable
                item.setCheckState(Qt.Unchecked)
            else:
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
            lst.addItem(item)
        vbox.addWidget(lst, stretch=1)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(dlg.reject)
        btn_row.addWidget(btn_cancel)
        btn_ok = QPushButton("Add checked")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_ok)
        vbox.addLayout(btn_row)
        if dlg.exec() != QDialog.Accepted:
            return []
        return [lst.item(i).data(Qt.UserRole) for i in range(lst.count())
                if lst.item(i).checkState() == Qt.Checked]

    # ── AI extraction ────────────────────────────────────────────────────

    def _tag_census(self) -> str:
        """Custom-tag frequency table with one sample text line per tag —
        the compact stand-in for dumping all node texts into the prompt."""
        layer_tags = set(LAYER_ORDER)
        counts: dict = {}
        samples: dict = {}
        for nid, nd in self.script.nodes.items():
            text = nd.get('text', '')
            for t in nd.get('tags', []):
                if t in layer_tags:
                    continue
                counts[t] = counts.get(t, 0) + 1
                if t not in samples and text:
                    samples[t] = text.replace('\n', ' ')[:160]
        lines = []
        for t, c in sorted(counts.items(), key=lambda kv: -kv[1])[:120]:
            s = samples.get(t, '')
            lines.append(f'  {t} (x{c}): "{s}"')
        return '\n'.join(lines)

    def _cmd_extract(self):
        if not self.ai.ready:
            self._status.setText("Claude CLI not found — cannot extract.")
            return
        if self.ai.busy:
            self._status.setText("AI is busy — try again in a moment.")
            return
        nodes = list(self.script.nodes.values())
        sample_texts = [nd.get('text', '') for nd in nodes[:60]
                        if nd.get('text')][:10]
        self._btn_extract.setEnabled(False)
        self._status.setText("Extracting codex… (this reads the bible + tags; "
                             "may take a minute)")

        def on_done(proposed: dict):
            self._btn_extract.setEnabled(True)
            # Drop anything that already exists — never overwrite. Match
            # loosely (slug, name, alias, article variants) rather than by
            # exact slug: the AI re-proposing 'goat' beside an existing
            # 'the_goat' used to slip through and create duplicate cards.
            fresh, dupes = {}, []
            for s, c in proposed.items():
                exist = self.script.find_equivalent_entity(s) or \
                    self.script.find_equivalent_entity(c.get('name', ''))
                if exist:
                    dupes.append(f'{s} = {exist}')
                else:
                    fresh[s] = c
            dupe_note = (f"  (skipped {len(dupes)} duplicate(s) of existing "
                         f"entries: {', '.join(dupes[:6])})") if dupes else ''
            if not fresh:
                self._status.setText("Extraction returned nothing new."
                                     + dupe_note)
                return
            chosen = self._pick_entities_dialog(
                f"Extracted {len(fresh)} proposed entities", fresh)
            added = 0
            for slug in chosen:
                card = deepcopy(ENTITY_TEMPLATE)
                card.update(fresh[slug])
                if slug not in self.script.entities \
                        and slug not in RESERVED_ENTITY_SLUGS:
                    self.script.entities[slug] = card
                    added += 1
            if added:
                self.script.dirty = True
            self._status.setText(
                f"Added {added} entit{'y' if added == 1 else 'ies'} "
                f"to the codex.{dupe_note}")
            self._refresh_list()

        def on_error(msg: str):
            self._btn_extract.setEnabled(True)
            self._status.setText(f"Extraction failed: {msg}")

        self.ai.extract_codex(
            story_context=self.script.story_context,
            tag_census=self._tag_census(),
            sample_texts=sample_texts,
            existing_slugs=list(self.script.entities.keys()),
            ui_queue=self.ui_queue,
            on_done=on_done, on_error=on_error,
        )


def _pick_checked(parent, title: str, entries: list) -> list:
    """Generic checkable review picker shared by the v2 AI passes.
    entries = [(key, label)]; returns the checked keys (all pre-checked),
    or [] on cancel."""
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(640, 460)
    vbox = QVBoxLayout(dlg)
    lst = QListWidget()
    for key, label in entries:
        item = QListWidgetItem(label)
        item.setData(Qt.ItemDataRole.UserRole, key)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        lst.addItem(item)
    vbox.addWidget(lst, stretch=1)
    row = QHBoxLayout()
    row.addStretch(1)
    btn_cancel = QPushButton("Cancel")
    btn_cancel.clicked.connect(dlg.reject)
    row.addWidget(btn_cancel)
    btn_ok = QPushButton("Apply checked")
    btn_ok.setDefault(True)
    btn_ok.clicked.connect(dlg.accept)
    row.addWidget(btn_ok)
    vbox.addLayout(row)
    if dlg.exec() != QDialog.Accepted:
        return []
    return [lst.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(lst.count())
            if lst.item(i).checkState() == Qt.CheckState.Checked]


class _ZoomableMapView(QGraphicsView):
    """Wheel-zoom / drag-pan QGraphicsView for the web mini-map.

    The map used to fitInView on every rebuild — with many stories that
    scaled everything toward unreadable. Now: wheel zooms around the
    cursor, left-drag pans, double-click (or the Fit button) refits."""

    def __init__(self, scene, on_user_zoom=None, on_click=None):
        super().__init__(scene)
        self._on_user_zoom = on_user_zoom
        self._on_click = on_click
        self._press_pos = None
        self._zoom = 1.0
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def mousePressEvent(self, event):
        self._press_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        # A release without meaningful movement is a CLICK (ScrollHandDrag
        # otherwise swallows clicks for panning) — report it in scene coords.
        #
        # CRASH-CRITICAL: the callback rebuilds the scene. It must run
        # AFTER Qt finishes this release (super() first, then a 0-ms timer)
        # — invoking it inline destroyed items the view was still touching
        # and caused native access-violation crashes (see crash.log).
        clicked = (self._on_click is not None and self._press_pos is not None
                   and (event.position().toPoint()
                        - self._press_pos).manhattanLength() < 6)
        scene_pos = (self.mapToScene(event.position().toPoint())
                     if clicked else None)
        self._press_pos = None
        super().mouseReleaseEvent(event)
        if clicked:
            QTimer.singleShot(0, lambda cb=self._on_click, p=scene_pos: cb(p))

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if not delta:
            return
        factor = 1.15 if delta > 0 else 1 / 1.15
        new_zoom = self._zoom * factor
        if not (0.05 <= new_zoom <= 8.0):
            return
        self._zoom = new_zoom
        self.scale(factor, factor)
        if self._on_user_zoom:
            self._on_user_zoom()

    def fit(self):
        rect = self.scene().itemsBoundingRect().adjusted(-40, -40, 40, 40)
        if rect.isEmpty():
            return
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = 1.0   # new relative-zoom baseline

    def mouseDoubleClickEvent(self, event):
        self.fit()
        super().mouseDoubleClickEvent(event)


class WebPlannerDialog(QDialog):
    """The Web — story-level planning (v2 Phase 3).

    Top: a read-only mini-map of the story web (stories on a circle, typed
    relation edges colored by kind, a dot marking the TO end). Middle: the
    relations editor. Bottom: the AI passes —
      Propose Web    — new story cards + relations from the codex + bible
      Weave Junctions— cross-story drift edges for every junction relation
      Weave Callbacks— rewrite 1-2 nodes per callback relation to reference
                       the target story's canon (reviewed before applying)
    """

    def __init__(self, main_window, script: 'ScriptData', ai: 'AIAssistant',
                 ui_queue: queue.SimpleQueue, embedded: bool = False,
                 parent=None):
        super().__init__(parent)
        self._mw = main_window
        self.script = script
        self.ai = ai
        self.ui_queue = ui_queue
        self._embedded = embedded
        self._ai_busy = False
        self._weave_chain = False   # Weave All: junctions → callbacks
        self._user_zoomed = False   # suppress auto-fit once the user zooms
        self._selected_story: Optional[str] = None
        self._path_stories: Optional[set] = None
        self._path_edges: Optional[set] = None
        self._up_edges: set = set()
        self._down_edges: set = set()
        self.setWindowTitle("Web Planner — Story Relations")
        if not embedded:
            self.setWindowFlags(self.windowFlags() | Qt.Window)
            self.resize(1000, 760)
        self._build_ui()
        self._refresh()

    def set_script(self, script: 'ScriptData'):
        self.script = script
        self._selected_story = None
        self._refresh()

    # Embedded guard: Escape must not blank the tab (QDialog.reject hides).
    def accept(self):
        if not self._embedded:
            super().accept()

    def reject(self):
        if not self._embedded:
            super().reject()

    # ── UI ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)

        # Map toolbar: fit, story filter, and per-type toggles that double
        # as the legend (click a color to hide/show that relation kind).
        map_bar = QHBoxLayout()
        btn_fit = QPushButton("Fit")
        btn_fit.setFixedWidth(40)
        btn_fit.setToolTip("Fit the whole web in view (double-click the map "
                           "does the same). Wheel zooms, drag pans.")
        btn_fit.clicked.connect(self._cmd_fit_map)
        map_bar.addWidget(btn_fit)

        self._trace_btn = QPushButton("Trace paths")
        self._trace_btn.setCheckable(True)
        self._trace_btn.setFixedHeight(22)
        self._trace_btn.setToolTip(
            "Off (default): selecting a story highlights only its DIRECT\n"
            "relations. On: highlights the full chains — every path that\n"
            "leads into the story and everything reachable onward from it.")
        self._trace_btn.setStyleSheet(
            "QPushButton { color: #777; border: 1px solid #444; "
            "border-radius: 3px; padding: 0 8px; font-size: 10px; }"
            "QPushButton:checked { color: #ffaa40; border: 1px solid #ffaa40; }")
        self._trace_btn.toggled.connect(self._on_trace_toggled)
        map_bar.addWidget(self._trace_btn)

        self._story_filter = QLineEdit()
        self._story_filter.setPlaceholderText(
            "filter stories… (non-matching dim; relations list follows)")
        self._story_filter.setClearButtonEnabled(True)
        self._story_filter.textChanged.connect(self._schedule_refresh)
        map_bar.addWidget(self._story_filter, stretch=1)

        self._type_toggles: dict = {}
        for rtype, color in RELATION_COLORS.items():
            btn = QPushButton(rtype)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setFixedHeight(22)
            btn.setToolTip(RELATION_TYPES.get(rtype, ''))
            btn.setStyleSheet(
                f"QPushButton {{ color: #555; border: 1px solid #444; "
                f"border-radius: 3px; padding: 0 6px; font-size: 10px; }}"
                f"QPushButton:checked {{ color: {color}; "
                f"border: 1px solid {color}; }}")
            btn.toggled.connect(self._schedule_refresh)
            self._type_toggles[rtype] = btn
            map_bar.addWidget(btn)
        root.addLayout(map_bar)

        self._scene = QGraphicsScene(self)
        # No BSP index: the scene is rebuilt wholesale on every refresh and
        # the index held dangling item pointers across clear() — implicated
        # in the native access-violation crashes. Linear hit-testing is
        # trivial at this item count.
        self._scene.setItemIndexMethod(
            QGraphicsScene.ItemIndexMethod.NoIndex)
        self._view = _ZoomableMapView(
            self._scene, on_user_zoom=self._on_user_zoomed,
            on_click=self._on_map_clicked)
        self._view.setToolTip(
            "Click a story to trace every path leading into and out of it.\n"
            "Click it again (or empty space) to clear. Wheel zooms, drag pans.")
        self._view.setMinimumHeight(280)
        self._view.setStyleSheet("background: #1a1a22; border: 1px solid #444;")
        root.addWidget(self._view, stretch=2)

        rel_hdr = QHBoxLayout()
        rel_hdr.addWidget(QLabel("Relations  (FROM —type→ TO)"))
        rel_hdr.addStretch(1)
        rel_hdr.addWidget(QLabel("Sort:"))
        self._sort_combo = QComboBox()
        for label, key in (("From story", "from"), ("Type", "type"),
                           ("To story", "to")):
            self._sort_combo.addItem(label, key)
        self._sort_combo.currentIndexChanged.connect(self._schedule_refresh)
        rel_hdr.addWidget(self._sort_combo)
        root.addLayout(rel_hdr)
        self._rel_list = QListWidget()
        root.addWidget(self._rel_list, stretch=1)

        add_row = QHBoxLayout()
        self._from_combo = QComboBox()
        add_row.addWidget(self._from_combo, stretch=1)
        self._type_combo = QComboBox()
        for rtype, desc in RELATION_TYPES.items():
            self._type_combo.addItem(rtype, rtype)
            self._type_combo.setItemData(
                self._type_combo.count() - 1, desc, Qt.ItemDataRole.ToolTipRole)
        add_row.addWidget(self._type_combo)
        self._to_combo = QComboBox()
        add_row.addWidget(self._to_combo, stretch=1)
        self._note_edit = QLineEdit()
        self._note_edit.setPlaceholderText("note (optional)")
        add_row.addWidget(self._note_edit, stretch=1)
        btn_add = QPushButton("Add")
        btn_add.clicked.connect(self._cmd_add_relation)
        add_row.addWidget(btn_add)
        btn_del = QPushButton("Remove selected")
        btn_del.clicked.connect(self._cmd_remove_relation)
        add_row.addWidget(btn_del)
        root.addLayout(add_row)

        # Context preview for the FROM/TO stories picked above — with many
        # stories a bare name gives nothing to relate on; this shows each
        # side's premise/cast/themes so a relation can be chosen on meaning.
        ctx_row = QHBoxLayout()
        self._from_info = QLabel("")
        self._to_info = QLabel("")
        for lbl in (self._from_info, self._to_info):
            lbl.setWordWrap(True)
            lbl.setStyleSheet(
                "color: #9a9aa8; font-size: 11px; background: #202028; "
                "border: 1px solid #333; border-radius: 3px; padding: 4px;")
            lbl.setMinimumHeight(40)
            ctx_row.addWidget(lbl, stretch=1)
        root.addLayout(ctx_row)
        self._from_combo.currentIndexChanged.connect(self._update_rel_context)
        self._to_combo.currentIndexChanged.connect(self._update_rel_context)

        ai_row = QHBoxLayout()
        self._instr = QLineEdit()
        self._instr.setPlaceholderText(
            "guidance for the AI web proposal (optional — e.g. 'two short "
            "vignette stories around the Toad')")
        ai_row.addWidget(self._instr, stretch=1)
        self._btn_propose = QPushButton("Propose Web (AI)")
        self._btn_propose.setToolTip(
            "Propose new story cards (premise, cast, structure) plus typed\n"
            "relations, anchored in the codex. You review before anything\n"
            "is created.")
        self._btn_propose.clicked.connect(self._cmd_propose_web)
        ai_row.addWidget(self._btn_propose)
        self._btn_junctions = QPushButton("Weave Junctions (AI)")
        self._btn_junctions.setToolTip(
            "For every 'junction' relation: suggest and add rare cross-story\n"
            f"edges (weight {JUNCTION_EDGE_WEIGHT}) where a listener's walk "
            "can drift between stories.")
        self._btn_junctions.clicked.connect(self._cmd_weave_junctions)
        ai_row.addWidget(self._btn_junctions)
        self._btn_callbacks = QPushButton("Weave Callbacks (AI)")
        self._btn_callbacks.setToolTip(
            "For every 'callback' relation: rewrite 1-2 nodes of the FROM\n"
            "story to reference the TO story's canon. Rewrites are reviewed\n"
            "side-by-side before applying.")
        self._btn_callbacks.clicked.connect(self._cmd_weave_callbacks)
        ai_row.addWidget(self._btn_callbacks)
        self._btn_weave_all = QPushButton("Weave All (AI)")
        self._btn_weave_all.setStyleSheet("font-weight: bold;")
        self._btn_weave_all.setToolTip(
            "One press for the whole weave: junctions first, then callbacks\n"
            "when they finish (callback rewrites still get their one review\n"
            "dialog at the end). Already-woven links are skipped, so press\n"
            "it again cheaply whenever new stories join the web.")
        self._btn_weave_all.clicked.connect(self._cmd_weave_all)
        ai_row.addWidget(self._btn_weave_all)
        self._btn_cancel_ai = QPushButton("Cancel")
        self._btn_cancel_ai.setToolTip(
            "Stop the running AI pass: calls not yet started are skipped;\n"
            "in-flight calls finish (bounded by the per-call timeout) and\n"
            "whatever completed is still applied.")
        self._btn_cancel_ai.setVisible(False)
        self._btn_cancel_ai.clicked.connect(self._cmd_cancel_ai)
        ai_row.addWidget(self._btn_cancel_ai)
        root.addLayout(ai_row)

        self._status = QLabel("")
        self._status.setStyleSheet("color: #999; font-size: 11px;")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        if not self._embedded:
            close_row = QHBoxLayout()
            close_row.addStretch(1)
            btn_close = QPushButton("Close")
            btn_close.clicked.connect(self.accept)
            close_row.addWidget(btn_close)
            root.addLayout(close_row)

    def _story_name(self, arc_id: str) -> str:
        return self.script.arcs.get(arc_id, {}).get('name') or arc_id

    def _on_user_zoomed(self):
        self._user_zoomed = True

    def _cmd_fit_map(self):
        self._user_zoomed = False
        self._view.fit()

    def _schedule_refresh(self, *_):
        """Coalesce and DEFER rebuilds to the next event-loop turn.

        Every widget-signal-driven refresh (filter typing, legend toggles,
        sort combo, add/remove buttons) must come through here: rebuilding
        the scene/list synchronously inside Qt's own event dispatch caused
        native access-violation crashes (see crash.log). Programmatic
        callers (set_script, _select_story, tests) may still call
        _refresh() directly — those run on a clean stack."""
        if getattr(self, '_refresh_pending', False):
            return
        self._refresh_pending = True
        QTimer.singleShot(0, self._do_scheduled_refresh)

    def _do_scheduled_refresh(self):
        self._refresh_pending = False
        self._refresh()
        if self._selected_story:
            self._update_selection_status()

    # ── Story selection + path tracing ───────────────────────────────────

    def _on_map_clicked(self, scene_pos):
        hit = None
        for arc_id, rect in getattr(self, '_box_rects', {}).items():
            if rect.contains(scene_pos):
                hit = arc_id
                break
        if hit == self._selected_story:
            hit = None   # clicking the selected story again clears
        self._select_story(hit)

    def _select_story(self, arc_id: Optional[str]):
        """Select a story on the map: highlights its DIRECT relations (or,
        with Trace paths on, every chain into and out of it), and the
        relations list narrows to match. None / empty-space click clears."""
        self._selected_story = arc_id if arc_id in self.script.arcs else None
        self._refresh()
        self._update_selection_status()

    def _on_trace_toggled(self, _checked: bool):
        self._schedule_refresh()

    def _update_selection_status(self):
        if not self._selected_story:
            self._status.setText("")
            return
        name = self._story_name(self._selected_story)
        if self._trace_btn.isChecked():
            self._status.setText(
                f"Selected '{name}' — full paths: {len(self._up_edges)} "
                f"relation(s) on chains leading in, {len(self._down_edges)} "
                "on chains leading onward. Click empty space to clear.")
        else:
            self._status.setText(
                f"Selected '{name}' — {len(self._up_edges)} direct "
                f"relation(s) in, {len(self._down_edges)} out. Turn on "
                "'Trace paths' for the full chains; click empty space to clear.")

    def _path_sets(self, sel: str) -> tuple:
        """(stories, edges) highlighted for the selected story.

        Default: only DIRECT relations touching `sel`. With the Trace
        paths toggle on: every directed chain that reaches `sel` and
        everything reachable onward from it. Hidden relation types
        (unchecked legend toggles) never participate. Also fills
        _up_edges/_down_edges for the status counts."""
        shown = self._shown_types()
        rels = [(f, t, ty) for f, t, ty, _ in self.script.all_story_relations()
                if ty in shown]
        if not self._trace_btn.isChecked():
            # Direct neighbors only — what a click most obviously means.
            self._down_edges = {(f, t, ty) for f, t, ty in rels if f == sel}
            self._up_edges = {(f, t, ty) for f, t, ty in rels if t == sel}
            edges = self._down_edges | self._up_edges
            stories = {sel} | {t for _, t, _ in self._down_edges} \
                | {f for f, _, _ in self._up_edges}
            return stories, edges
        fwd, rev = {}, {}
        for f, t, ty in rels:
            fwd.setdefault(f, []).append((t, ty))
            rev.setdefault(t, []).append((f, ty))
        self._down_edges = set()
        down = {sel}
        frontier = [sel]
        while frontier:
            cur = frontier.pop()
            for nxt, ty in fwd.get(cur, []):
                self._down_edges.add((cur, nxt, ty))
                if nxt not in down:
                    down.add(nxt)
                    frontier.append(nxt)
        self._up_edges = set()
        up = {sel}
        frontier = [sel]
        while frontier:
            cur = frontier.pop()
            for prv, ty in rev.get(cur, []):
                self._up_edges.add((prv, cur, ty))
                if prv not in up:
                    up.add(prv)
                    frontier.append(prv)
        return (down | up), (self._down_edges | self._up_edges)

    def _shown_types(self) -> set:
        return {t for t, b in self._type_toggles.items() if b.isChecked()}

    def _story_matches(self, arc_id: str) -> bool:
        term = self._story_filter.text().strip().lower()
        if not term:
            return True
        return (term in (self._story_name(arc_id) or '').lower()
                or term in arc_id.lower())

    def _story_brief(self, arc_id: str, premise_chars: int = 240) -> str:
        """Multi-line context card for one story — premise, themes, cast,
        beat structure, node count. Shown as combo/list/map tooltips and in
        the FROM/TO preview panes: with many stories a bare name gives
        nothing to relate on. Memoized per refresh — the node count is
        O(nodes) and this is called per combo item and per relation row."""
        cache = getattr(self, '_brief_cache', None)
        if cache is not None and arc_id in cache:
            return cache[arc_id]
        arc = self.script.arcs.get(arc_id) or {}
        lines = [self._story_name(arc_id)]
        prem = (arc.get('premise') or '').strip().replace('\n', ' ')
        lines.append(prem[:premise_chars]
                     + ('…' if len(prem) > premise_chars else '')
                     if prem else '(no premise)')
        themes = (arc.get('themes') or '').strip()
        motif = (arc.get('motif') or '').strip()
        if themes or motif:
            lines.append('themes: '
                         + ' · '.join(x for x in (themes, motif) if x)[:160])
        cast = [c for c in (arc.get('cast') or []) if isinstance(c, str)]
        if cast:
            lines.append('cast: ' + ', '.join(cast[:8])
                         + ('…' if len(cast) > 8 else ''))
        layers = [s.get('layer', '') for s in (arc.get('structure') or [])
                  if isinstance(s, dict) and s.get('layer')]
        if layers:
            lines.append('beats: ' + ' → '.join(layers))
        n = sum(1 for nid in self.script.nodes
                if self.script.get_node_arc_id(nid) == arc_id)
        lines.append(f'{n} node(s) generated' if n else 'no nodes yet')
        out = '\n'.join(lines)
        if cache is not None:
            cache[arc_id] = out
        return out

    def _update_rel_context(self, *_):
        f = self._from_combo.currentData()
        t = self._to_combo.currentData()
        self._from_info.setText(
            ('FROM — ' + self._story_brief(f)) if f else '')
        self._to_info.setText(
            ('TO — ' + self._story_brief(t)) if t else '')

    def _refresh(self, *_):
        if not hasattr(self, '_rel_list'):
            return   # signal fired mid-construction
        # A native tooltip may be open over an item we are about to delete;
        # destroying the item under it access-violates on this machine
        # (crash.log 2026-07-17 18:49 — AV at _rel_list.clear() while the
        # new context tooltips were in use). Hide it before any rebuild.
        QToolTip.hideText()
        self._brief_cache = {}   # per-refresh memo for _story_brief
        # combos — clear()+repopulate on every refresh is this machine's
        # native-crash vector (same PySide6 pattern as arc_list, see
        # crash.log); rebuild only when the story set actually changed,
        # otherwise update names/tooltips in place.
        arc_ids = list(self.script.arcs)
        for combo in (self._from_combo, self._to_combo):
            current_ids = [combo.itemData(i) for i in range(combo.count())]
            if current_ids != arc_ids:
                cur = combo.currentData()
                combo.blockSignals(True)
                combo.clear()
                for arc_id in arc_ids:
                    combo.addItem(self._story_name(arc_id), arc_id)
                idx = combo.findData(cur)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                combo.blockSignals(False)
            for i, arc_id in enumerate(arc_ids):
                combo.setItemText(i, self._story_name(arc_id))
                combo.setItemData(i, self._story_brief(arc_id),
                                  Qt.ItemDataRole.ToolTipRole)
        self._update_rel_context()   # signals were blocked above
        # Path tracing for the selected story (drives map highlight + list)
        sel = self._selected_story
        if sel not in self.script.arcs:
            sel = self._selected_story = None
        if sel:
            self._path_stories, self._path_edges = self._path_sets(sel)
        else:
            self._path_stories = self._path_edges = None
            self._up_edges = set()
            self._down_edges = set()
        # relations list — honors the type toggles, the story filter
        # (either endpoint may match), the path selection, and the sort
        # selector
        shown = self._shown_types()
        rels = [r for r in self.script.all_story_relations()
                if r[2] in shown
                and (self._story_matches(r[0]) or self._story_matches(r[1]))]
        if self._path_edges is not None:
            rels = [r for r in rels if (r[0], r[1], r[2]) in self._path_edges]
        sort_key = self._sort_combo.currentData() or 'from'
        if sort_key == 'type':
            rels.sort(key=lambda r: (r[2], self._story_name(r[0]).lower()))
        elif sort_key == 'to':
            rels.sort(key=lambda r: (self._story_name(r[1]).lower(), r[2]))
        else:
            rels.sort(key=lambda r: (self._story_name(r[0]).lower(), r[2]))
        # Same crash-avoidance as the combos: only clear()+rebuild when the
        # row set changed; tooltip refresh happens in place either way.
        desired = []
        for (f, t, rtype, note) in rels:
            label = f'{self._story_name(f)}  —{rtype}→  {self._story_name(t)}'
            if note:
                label += f'    · {note}'
            tip = (f"{rtype}: {RELATION_TYPES.get(rtype, '')}\n\n"
                   f"FROM — {self._story_brief(f)}\n\n"
                   f"TO — {self._story_brief(t)}")
            desired.append((label, (f, t, rtype), rtype, tip))
        current = [(self._rel_list.item(i).text(),
                    self._rel_list.item(i).data(Qt.ItemDataRole.UserRole))
                   for i in range(self._rel_list.count())]
        if current != [(lbl, key) for lbl, key, _, _ in desired]:
            self._rel_list.clear()
            for label, key, rtype, tip in desired:
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, key)
                item.setToolTip(tip)
                color = RELATION_COLORS.get(rtype)
                if color:
                    item.setForeground(QColor(color))
                self._rel_list.addItem(item)
        else:
            for i, (_label, _key, _rtype, tip) in enumerate(desired):
                self._rel_list.item(i).setToolTip(tip)
        self._rebuild_map()

    def _rebuild_map(self):
        self._scene.clear()
        # Alphabetical placement — stable positions as stories are added
        ids = sorted(self.script.arcs.keys(),
                     key=lambda a: (self._story_name(a) or a).lower())
        n = len(ids)
        if not n:
            self._scene.addSimpleText(
                "No stories yet — create some in Story ▸ Stories…, or use "
                "Propose Web (AI).").setBrush(QBrush(QColor('#888888')))
            return
        shown = self._shown_types()
        radius = max(140.0, 46.0 * n)
        centers = {}
        box_w, box_h = 150.0, 34.0
        for i, arc_id in enumerate(ids):
            ang = (2.0 * math.pi * i / n) - math.pi / 2.0
            cx, cy = radius * math.cos(ang), radius * math.sin(ang)
            centers[arc_id] = (cx, cy)
        # edges under boxes — hidden types skipped. With a story selected,
        # path edges glow and everything else recedes; otherwise the text
        # filter dims edges touching no matched story.
        sel = self._selected_story
        pair_seen: dict = {}
        for (f, t, rtype, _note) in self.script.all_story_relations():
            if rtype not in shown or f not in centers or t not in centers:
                continue
            (x1, y1), (x2, y2) = centers[f], centers[t]
            # perpendicular offset so parallel relations don't overlap
            k = pair_seen.get((f, t), 0) + pair_seen.get((t, f), 0)
            pair_seen[(f, t)] = pair_seen.get((f, t), 0) + 1
            dx, dy = x2 - x1, y2 - y1
            dist = math.hypot(dx, dy) or 1.0
            ox, oy = -dy / dist * 7.0 * k, dx / dist * 7.0 * k
            color = QColor(RELATION_COLORS.get(rtype, '#888888'))
            width = 1.6
            if sel:
                if (f, t, rtype) in (self._path_edges or set()):
                    width = 2.6
                else:
                    color.setAlpha(40)
            elif not (self._story_matches(f) or self._story_matches(t)):
                color.setAlpha(55)
            pen = QPen(color)
            pen.setWidthF(width)
            self._scene.addLine(x1 + ox, y1 + oy, x2 + ox, y2 + oy, pen)
            # direction dot at 78% toward the TO end
            px = x1 + ox + (x2 - x1) * 0.78
            py = y1 + oy + (y2 - y1) * 0.78
            self._scene.addEllipse(px - 3.5, py - 3.5, 7.0, 7.0,
                                   QPen(color), QBrush(color))
        # story boxes — selection outranks the text filter: the selected
        # story gets an accent border, path stories stay bright, the rest
        # recede. Rects are remembered for click hit-testing.
        self._box_rects = {}
        for arc_id, (cx, cy) in centers.items():
            if sel:
                on_path = arc_id in (self._path_stories or set())
                is_sel = arc_id == sel
                pen_c = '#ffaa40' if is_sel else ('#666688' if on_path
                                                 else '#3a3a4a')
                fill_c = '#2a2a3a' if on_path else '#202028'
                text_c = '#dddddd' if on_path else '#555566'
                pen = QPen(QColor(pen_c))
                if is_sel:
                    pen.setWidthF(2.4)
            else:
                matched = self._story_matches(arc_id)
                pen = QPen(QColor('#666688' if matched else '#3a3a4a'))
                fill_c = '#2a2a3a' if matched else '#202028'
                text_c = '#dddddd' if matched else '#555566'
            box_rect = QRectF(cx - box_w / 2, cy - box_h / 2, box_w, box_h)
            self._box_rects[arc_id] = box_rect
            rect = self._scene.addRect(box_rect, pen, QBrush(QColor(fill_c)))
            rect.setZValue(10)
            rect.setToolTip(self._story_brief(arc_id))
            name = self._story_name(arc_id)
            if len(name) > 20:
                name = name[:19] + '…'
            txt = self._scene.addSimpleText(name)
            txt.setBrush(QBrush(QColor(text_c)))
            br = txt.boundingRect()
            txt.setPos(cx - br.width() / 2, cy - br.height() / 2)
            txt.setZValue(11)
            txt.setToolTip(self._story_brief(arc_id))
        self._scene.setSceneRect(self._scene.itemsBoundingRect().adjusted(
            -40, -40, 40, 40))
        # Respect the user's zoom/pan; only auto-fit while they haven't
        # taken control (Fit button / double-click resets).
        if not self._user_zoomed:
            self._view.fit()

    # ── Manual relation editing ──────────────────────────────────────────

    def _cmd_add_relation(self):
        f = self._from_combo.currentData()
        t = self._to_combo.currentData()
        rtype = self._type_combo.currentData()
        if not f or not t or f == t:
            self._status.setText("Pick two different stories.")
            return
        if self.script.add_story_relation(f, t, rtype, self._note_edit.text()):
            self._note_edit.clear()
            self._schedule_refresh()
        else:
            self._status.setText("That relation already exists.")

    def _cmd_remove_relation(self):
        item = self._rel_list.currentItem()
        if not item:
            return
        f, t, rtype = item.data(Qt.ItemDataRole.UserRole)
        self.script.remove_story_relation(f, t, rtype)
        self._schedule_refresh()

    # ── Shared helpers ───────────────────────────────────────────────────

    def _set_ai_busy(self, busy: bool, msg: str = ''):
        self._ai_busy = busy
        for b in (self._btn_propose, self._btn_junctions,
                  self._btn_callbacks, self._btn_weave_all):
            b.setEnabled(not busy)
        if busy:
            self._cancel_ev = threading.Event()   # fresh per pass
        self._btn_cancel_ai.setVisible(busy)
        if msg:
            self._status.setText(msg)

    def _cmd_cancel_ai(self):
        self._weave_chain = False   # a cancelled weave never chains onward
        ev = getattr(self, '_cancel_ev', None)
        if ev is not None and not ev.is_set():
            ev.set()
            self._status.setText(
                "Cancelling — skipping queued calls, letting in-flight ones "
                "finish…")

    def _cmd_weave_all(self):
        """One press for the whole weave: junctions, then callbacks once
        the junction pass lands. Both passes skip already-woven material,
        so this is the standing 'weave whatever is new' button."""
        if not self._ai_guard():
            return
        self._weave_chain = True
        if any(r[2] == 'junction'
               for r in self.script.all_story_relations()):
            self._cmd_weave_junctions()
            if not self._ai_busy:
                # Junctions bailed without starting (all saturated / no
                # candidate nodes) — go straight to callbacks.
                self._weave_chain = False
                self._cmd_weave_callbacks()
        else:
            self._weave_chain = False
            self._cmd_weave_callbacks()

    def _ai_guard(self) -> bool:
        if not self.ai.ready:
            self._status.setText("claude CLI not found.")
            return False
        if self._ai_busy:
            self._status.setText("An AI pass is already running.")
            return False
        return True

    def _pick_dialog(self, title: str, entries: list) -> list:
        return _pick_checked(self, title, entries)

    @staticmethod
    def _sample_even(items: list, k: int) -> list:
        if len(items) <= k:
            return list(items)
        step = len(items) / k
        return [items[int(i * step)] for i in range(k)]

    def _annotated_story_nodes(self, arc_id: str) -> list:
        """[(node_id, node, archetype, fraction)] for a story's nodes with
        text, sorted by position through the story. Bridge nodes are
        excluded — they're seam material, not weave candidates (otherwise
        repeated weaving sprouts bridges from bridges)."""
        structure = StoryStructure(self.script.get_story_structure(arc_id))
        out = []
        for nid, nd in self.script.nodes.items():
            if self.script.get_node_arc_id(nid) != arc_id or not nd.get('text'):
                continue
            if nd.get('label') == 'bridge':
                continue
            b = structure.beat_index_for_node(nd)
            out.append((nid, nd, structure.archetype(b),
                        (b + 1) / structure.n_beats))
        out.sort(key=lambda x: x[3])
        return out

    def _existing_junction_pairs(self, from_id: str, to_id: str) -> set:
        """(source, target) node pairs already connected across this
        junction — directly, or through a bridge node. The weave uses this
        to tell the AI what exists, to skip saturated junctions, and to
        refuse duplicate applications (a bridged pair has no direct edge,
        so without this a re-press would build the same bridge again)."""
        pairs = set()
        arc_of = self.script.get_node_arc_id
        for nid, nd in self.script.nodes.items():
            # Bridges are conduits, not sources — counting bridge→target as
            # its own pair would double-count every bridged crossing.
            if arc_of(nid) != from_id or nd.get('label') == 'bridge':
                continue
            for nxt in nd.get('next', []):
                tnd = self.script.nodes.get(nxt)
                if tnd is None:
                    continue
                if arc_of(nxt) == to_id:
                    pairs.add((nid, nxt))
                elif tnd.get('label') == 'bridge':
                    for b2 in tnd.get('next', []):
                        if arc_of(b2) == to_id:
                            pairs.add((nid, b2))
        return pairs

    # ── Propose Web (AI) ─────────────────────────────────────────────────

    def _cmd_propose_web(self):
        if not self._ai_guard():
            return
        parts = []
        if self.script.arcs:
            lines = ['EXISTING STORIES:']
            for arc_id, arc in self.script.arcs.items():
                layers = ' → '.join(
                    e['layer'] for e in self.script.get_story_structure(arc_id))
                cast = ', '.join(self.script.get_story_cast(arc_id)) or '(none)'
                premise = (arc.get('premise') or '')[:300]
                lines.append(f'- "{arc.get("name") or arc_id}": cast: {cast}; '
                             f'structure: {layers}; premise: {premise}')
            parts.append('\n'.join(lines))
            rels = self.script.all_story_relations()
            if rels:
                parts.append('EXISTING RELATIONS:\n' + '\n'.join(
                    f'  {self._story_name(f)} —{rtype}→ {self._story_name(t)}'
                    for f, t, rtype, _ in rels))
        else:
            parts.append('EXISTING STORIES: (none yet)')
        # Cast-usage census: tell the AI which entities are dormant (cast
        # nowhere) so new stories activate them instead of piling more
        # load onto the already-popular ones.
        cast_counts = {slug: 0 for slug in self.script.entities}
        for arc_id in self.script.arcs:
            for slug in self.script.get_story_cast(arc_id):
                if slug in cast_counts:
                    cast_counts[slug] += 1
        dormant = sorted(s for s, n in cast_counts.items() if n == 0)
        used = sorted(((s, n) for s, n in cast_counts.items() if n > 0),
                      key=lambda kv: -kv[1])
        if dormant:
            parts.append(
                'DORMANT CODEX ENTITIES (cast in NO story yet — prioritize '
                'these as anchors for the new stories):\n  '
                + ', '.join(dormant))
        if used:
            parts.append('CAST USAGE (stories each entity is already cast '
                         'in):\n  '
                         + ', '.join(f'{s}×{n}' for s, n in used))
        instructions = self._instr.text().strip()
        if instructions:
            parts.append(f'AUTHOR DIRECTION (highest priority):\n  {instructions}')
        parts.append('Propose the web now. Output the JSON object only.')
        prompt = '\n\n'.join(parts)
        system = self.ai._augment_system_with_context(
            SYSTEM_PROPOSE_WEB, self.script.story_context,
            cast_codex=self.script.full_codex_block())

        self._set_ai_busy(True, "Proposing web… (bible + codex + existing "
                                "stories; may take a minute)")

        def run():
            try:
                data = self.ai._run_claude_json(system, prompt,
                                                model_override=MODEL_OPUS)
                if self._cancel_ev.is_set():
                    self.ui_queue.put(lambda: self._set_ai_busy(
                        False, "Propose cancelled — nothing applied."))
                    return
                self.ui_queue.put(lambda d=data: self._apply_web_proposal(d))
            except Exception as exc:
                self.ui_queue.put(lambda e=str(exc): self._set_ai_busy(
                    False, f"Propose failed: {e[:120]}"))

        threading.Thread(target=run, daemon=True).start()

    def _apply_web_proposal(self, data: dict):
        self._set_ai_busy(False)
        stories = [s for s in (data.get('stories') or [])
                   if isinstance(s, dict) and (s.get('name') or '').strip()]
        relations = [r for r in (data.get('relations') or [])
                     if isinstance(r, dict) and r.get('type') in RELATION_TYPES]
        if not stories and not relations:
            self._status.setText("Proposal came back empty.")
            return
        entries = []
        for i, s in enumerate(stories):
            entries.append((f'story:{i}',
                            f'[story] {s["name"]} — {(s.get("premise") or "")[:90]}'))
        for i, r in enumerate(relations):
            entries.append((f'rel:{i}',
                            f'[relation] {r.get("from", "?")} —{r["type"]}→ {r.get("to", "?")}'))
        chosen = set(self._pick_dialog(
            f'Web proposal: {len(stories)} stories, {len(relations)} relations',
            entries))
        if not chosen:
            self._status.setText("Proposal discarded.")
            return

        name_to_id = {(arc.get('name') or '').strip().lower(): arc_id
                      for arc_id, arc in self.script.arcs.items()}
        n_stories = 0
        for i, s in enumerate(stories):
            if f'story:{i}' not in chosen:
                continue
            structure = [{'layer': e.get('layer'),
                          'direction': str(e.get('direction') or '')}
                         for e in (s.get('structure') or [])
                         if isinstance(e, dict) and e.get('layer') in LAYER_ORDER]
            if not structure:
                structure = ScriptData.default_structure()
            cast = [c for c in (s.get('cast') or []) if c in self.script.entities]
            arc_id = self.script.add_arc()
            self.script.save_arc(arc_id, {
                'name':      str(s['name'])[:80],
                'premise':   str(s.get('premise') or '')[:1500],
                'themes':    str(s.get('themes') or '')[:300],
                'motif':     str(s.get('motif') or '')[:300],
                'structure': structure,
                'beats':     ScriptData.beats_from_structure(structure),
                'cast':      cast,
                'notes':     '',
            })
            name_to_id[str(s['name']).strip().lower()] = arc_id
            n_stories += 1
        n_rels = 0
        for i, r in enumerate(relations):
            if f'rel:{i}' not in chosen:
                continue
            f = name_to_id.get(str(r.get('from', '')).strip().lower())
            t = name_to_id.get(str(r.get('to', '')).strip().lower())
            if f and t and self.script.add_story_relation(
                    f, t, r['type'], str(r.get('note') or '')):
                n_rels += 1
        self._status.setText(
            f"Web proposal applied: {n_stories} new stor"
            f"{'y' if n_stories == 1 else 'ies'}, {n_rels} relation(s). "
            "Generate each story from Story ▸ Stories….")
        self._refresh()

    # ── Weave Junctions (AI) ─────────────────────────────────────────────

    def _junction_candidates(self, from_id: str, to_id: str) -> tuple:
        """(sources, targets) for one junction: mid-to-late FROM nodes and
        early-to-mid TO nodes (position guards live here in code; the AI
        only picks among pre-filtered candidates)."""
        src = [x for x in self._annotated_story_nodes(from_id) if x[3] >= 0.3]
        tgt = [x for x in self._annotated_story_nodes(to_id) if x[3] <= 0.7]
        if not src:
            src = self._annotated_story_nodes(from_id)
        if not tgt:
            tgt = self._annotated_story_nodes(to_id)
        return self._sample_even(src, 18), self._sample_even(tgt, 18)

    def _cmd_weave_junctions(self):
        if not self._ai_guard():
            return
        rels = [(f, t, note) for (f, t, rtype, note)
                in self.script.all_story_relations() if rtype == 'junction']
        if not rels:
            self._status.setText("No 'junction' relations to weave — add one first.")
            return
        # Build every prompt on the UI thread; the worker only calls the AI.
        # Already-woven junctions are skipped entirely (idempotent presses,
        # bounded cost as the web grows); partially-woven ones tell the AI
        # what exists so it doesn't re-suggest near-duplicates.
        jobs = []
        n_saturated = 0
        for f, t, note in rels:
            existing = self._existing_junction_pairs(f, t)
            if len(existing) >= MAX_JUNCTION_LINKS_PER_PAIR:
                n_saturated += 1
                continue
            src, tgt = self._junction_candidates(f, t)
            if not src or not tgt:
                continue
            # The seam the listener hears is the source's ENDING flowing
            # into the target's OPENING — show the AI those exact ends.
            lines = [f'STORY A "{self._story_name(f)}" — drift SOURCES:']
            for nid, nd, arch, frac in src:
                lines.append(f'  [{nid}] {arch} @{int(frac * 100)}% '
                             f'ending: "…{nd.get("text", "")[-200:]}"')
            lines.append(f'\nSTORY B "{self._story_name(t)}" — drift TARGETS:')
            for nid, nd, arch, frac in tgt:
                lines.append(f'  [{nid}] {arch} @{int(frac * 100)}% '
                             f'opening: "{nd.get("text", "")[:200]}…"')
            if note:
                lines.append(f'\nRELATION NOTE: {note}')
            if existing:
                lines.append(
                    '\nALREADY WOVEN (existing crossings for this junction — '
                    'do NOT re-suggest these pairs or near-duplicates; zero '
                    'new links is the right answer if the junction already '
                    'feels covered):')
                for a, b in sorted(existing):
                    lines.append(f'  {a} → {b}')
            lines.append('\nSuggest the junction links now. Output the JSON object only.')
            jobs.append((f, t, '\n'.join(lines),
                         {x[0] for x in src}, {x[0] for x in tgt}))
        if not jobs:
            if n_saturated:
                self._status.setText(
                    f"All {n_saturated} junction(s) are already fully woven "
                    f"(≥{MAX_JUNCTION_LINKS_PER_PAIR} crossings each). Delete "
                    "drift edges/bridges on the graph to re-weave.")
            else:
                self._status.setText("Junction stories have no generated nodes yet.")
            return
        extra = f" ({n_saturated} already fully woven, skipped)" if n_saturated else ""
        self._set_ai_busy(True, f"Weaving {len(jobs)} junction(s)…{extra}")

        def run():
            def one(job):
                f, t, prompt, vf, vt = job
                data = self.ai._run_claude_json(SYSTEM_WEAVE_JUNCTIONS, prompt,
                                                max_retries=2)
                out = []
                for link in data.get('links', []) or []:
                    if not isinstance(link, dict):
                        continue
                    a, b = link.get('from'), link.get('to')
                    bridge = str(link.get('bridge_text') or '').strip()
                    if a in vf and b in vt:
                        out.append((f, t, a, b, bridge))
                return out

            def prog(done, total):
                self.ui_queue.put(lambda d=done, n=total: self._status.setText(
                    f"Weaving junctions… {d}/{n} done"))

            ev = self._cancel_ev
            results = _fan_out_ai_calls(jobs, one, prog, cancel_event=ev)
            cancelled = ev.is_set()
            all_links = [lk for r in results if r for lk in r]

            def apply(links=all_links):
                n_edges = 0
                n_bridges = 0
                n_dupes = 0
                new_nodes = set()
                woven: dict = {}   # (from_arc, to_arc) -> existing pair set
                for from_arc, to_arc, a, b, bridge in links:
                    if a not in self.script.nodes or b not in self.script.nodes:
                        continue
                    key = (from_arc, to_arc)
                    if key not in woven:
                        woven[key] = self._existing_junction_pairs(from_arc,
                                                                   to_arc)
                    if (a, b) in woven[key]:
                        n_dupes += 1   # already connected (directly or bridged)
                        continue
                    woven[key].add((a, b))
                    if bridge:
                        bid = self._create_bridge_node(from_arc, a, b, bridge)
                        if bid:
                            new_nodes.add(bid)
                            n_bridges += 1
                            continue
                    src_nd = self.script.nodes.get(a, {})
                    if b not in src_nd.get('next', []):
                        self.script.add_edge(a, b, weight=JUNCTION_EDGE_WEIGHT)
                        n_edges += 1
                if new_nodes:
                    self._mw._add_nodes_incremental(new_nodes)
                if n_edges or new_nodes:
                    self._mw._sync_missing_edges()
                parts = []
                if n_edges:
                    parts.append(f"{n_edges} drift edge(s) at weight "
                                 f"{JUNCTION_EDGE_WEIGHT}")
                if n_bridges:
                    parts.append(f"{n_bridges} bridge node(s) — generate "
                                 "audio for them before playback")
                if n_dupes:
                    parts.append(f"{n_dupes} already-woven pair(s) skipped")
                head = ("Junctions woven (CANCELLED early — partial): "
                        if cancelled else "Junctions woven: ")
                msg = head + (", ".join(parts) if parts
                              else "nothing new to add") + \
                    ". Delete anything you dislike on the graph."
                # New cross-story edges can seal a cycle with no path to
                # any ending — a walk entering it never finishes. Check
                # right away, while undoing is one edge-delete.
                if n_edges or new_nodes:
                    trapped = self.script.nodes_that_cannot_end()
                    if trapped:
                        msg += (f"  ⚠ {len(trapped)} node(s) can no longer "
                                "reach an ending (trapped cycle) — see "
                                "Story ▸ Web Statistics for the list.")
                self._set_ai_busy(False, msg)
                # Weave All chain: junctions done → run callbacks.
                if self._weave_chain:
                    self._weave_chain = False
                    if not cancelled:
                        self._status.setText(
                            msg + "  → weaving callbacks next…")
                        QTimer.singleShot(0, self._cmd_weave_callbacks)
            self.ui_queue.put(apply)

        threading.Thread(target=run, daemon=True).start()

    def _create_bridge_node(self, from_arc: str, a: str, b: str,
                            text: str) -> Optional[str]:
        """Materialize an AI-authored bridge between two junction endpoints:
        a short transition node wired  a →(rare)→ bridge →(always)→ b.
        The bridge belongs to the FROM story, and averages the endpoints'
        story variables and voice settings so the visuals and delivery
        glide through the crossing instead of jumping."""
        s = self.script
        na, nb = s.nodes.get(a), s.nodes.get(b)
        if not na or not nb or not text.strip():
            return None
        base = f'bridge_{a[:16]}_{b[:16]}'
        nid, counter = base, 2
        while nid in s.nodes:
            nid = f'{base}_{counter}'
            counter += 1
        pa = na.get('pos', [100, 100])
        pb = nb.get('pos', [100, 100])
        node = s.add_node(nid, '', pos=[(pa[0] + pb[0]) / 2.0,
                                        (pa[1] + pb[1]) / 2.0])
        s.update_text(nid, text)   # TTS sanitation

        # Layer tag from the source (keeps layouts sensible) + any codex
        # entities actually present in the bridge text.
        arch = next((t for t in na.get('tags', []) if t in LAYER_ORDER), 'echo')
        tags = [arch]
        low = text.lower()
        for slug, card in s.entities.items():
            name = (card.get('name') or '').lower()
            if slug in low or (name and name in low):
                tags.append(slug)
        node['tags'] = tags
        node['label'] = 'bridge'

        # Transitional values: halfway between the endpoints
        vars_a = na.get('vars') or {}
        vars_b = nb.get('vars') or {}
        node['vars'] = {k: round((vars_a.get(k, 0.0) + vars_b.get(k, 0.0)) / 2, 2)
                        for k in (set(vars_a) | set(vars_b))}
        va = na.get('voice_settings') or {}
        vb = nb.get('voice_settings') or {}
        if va or vb:
            vs = {
                'stability': round((va.get('stability', 0.5)
                                    + vb.get('stability', 0.5)) / 2, 2),
                'similarity_boost': 0.75,
                'style': round((va.get('style', 0.3)
                                + vb.get('style', 0.3)) / 2, 2),
            }
            _clamp_voice_settings(vs)
            node['voice_settings'] = vs

        s.set_node_arc_id(nid, from_arc)
        src_beat = na.get('beat')
        if isinstance(src_beat, int):
            node['beat'] = src_beat

        # Drift is rare into the bridge; once on it, the crossing completes.
        s.add_edge(a, nid, weight=JUNCTION_EDGE_WEIGHT)
        s.add_edge(nid, b, weight=1.0)
        return nid

    # ── Weave Callbacks (AI) ─────────────────────────────────────────────

    def _cmd_weave_callbacks(self):
        if not self._ai_guard():
            return
        rels = [(f, t, note) for (f, t, rtype, note)
                in self.script.all_story_relations() if rtype == 'callback']
        if not rels:
            self._status.setText("No 'callback' relations to weave — add one first.")
            return
        jobs = []
        n_woven = 0
        for f, t, note in rels:
            # Idempotence: applied rewrites stamp 'callback_to' on the node;
            # a relation with one is done — repeated presses skip it instead
            # of re-rewriting text (or paying the AI call again).
            if any(nd.get('callback_to') == t
                   for nid, nd in self.script.nodes.items()
                   if self.script.get_node_arc_id(nid) == f):
                n_woven += 1
                continue
            candidates = [x for x in self._annotated_story_nodes(f)
                          if x[3] >= 0.35 and not x[1].get('callback_to')]
            if not candidates:
                candidates = [x for x in self._annotated_story_nodes(f)
                              if not x[1].get('callback_to')]
            candidates = self._sample_even(candidates, 10)
            if not candidates:
                continue
            target_arc = self.script.arcs.get(t, {})
            events = [ev for ev in self.script.canon_events()
                      if ev.get('established_by') == t] \
                or self.script.canon_events_for(t)
            parts = [f'TARGET STORY "{self._story_name(t)}":\n'
                     f'  premise: {(target_arc.get("premise") or "")[:600]}']
            if events:
                parts.append('TARGET CANON EVENTS:\n' + '\n'.join(
                    f'  - [{ev.get("id", "?")}] {ev.get("summary", "")}'
                    for ev in events[:12]))
            else:
                parts.append('TARGET CANON EVENTS: (none extracted yet — treat '
                             'the premise itself as the canon to reference)')
            if note:
                parts.append(f'RELATION NOTE: {note}')
            cand_lines = [f'CANDIDATE NODES (story "{self._story_name(f)}"):']
            for nid, nd, arch, frac in candidates:
                cand_lines.append(f'  [{nid}] ({arch}): "{nd.get("text", "")}"')
            parts.append('\n'.join(cand_lines))
            parts.append('Weave the callback now. Output the JSON object only.')
            jobs.append((f, t, '\n\n'.join(parts), {x[0] for x in candidates}))
        if not jobs:
            if n_woven:
                self._status.setText(
                    f"All {n_woven} callback(s) already woven. To redo one, "
                    "delete the relation's rewritten node or its "
                    "'callback_to' marker, then weave again.")
            else:
                self._status.setText("Callback stories have no generated nodes yet.")
            return
        extra = f" ({n_woven} already woven, skipped)" if n_woven else ""
        self._set_ai_busy(True, f"Weaving {len(jobs)} callback(s)…{extra}")

        def run():
            def one(job):
                f, t, prompt, valid = job
                data = self.ai._run_claude_json(SYSTEM_WEAVE_CALLBACKS, prompt,
                                                max_retries=2)
                out = []
                for rw in (data.get('rewrites') or [])[:2]:
                    if not isinstance(rw, dict):
                        continue
                    nid = rw.get('node_id')
                    new_text = (rw.get('new_text') or '').strip()
                    if nid in valid and new_text:
                        tags = [ScriptData.sanitize_entity_slug(str(x))
                                for x in (rw.get('add_tags') or [])]
                        out.append((nid, new_text, tags,
                                    self._story_name(f), self._story_name(t),
                                    t))
                return out

            def prog(done, total):
                self.ui_queue.put(lambda d=done, n=total: self._status.setText(
                    f"Weaving callbacks… {d}/{n} done"))

            results = _fan_out_ai_calls(jobs, one, prog,
                                        cancel_event=self._cancel_ev)
            rewrites = [rw for r in results if r for rw in r]
            self.ui_queue.put(lambda r=rewrites: self._review_callback_rewrites(r))

        threading.Thread(target=run, daemon=True).start()

    def _review_callback_rewrites(self, rewrites: list):
        self._set_ai_busy(False)
        if not rewrites:
            self._status.setText("No callback rewrites came back.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Callback rewrites ({len(rewrites)})")
        dlg.resize(820, 560)
        vbox = QVBoxLayout(dlg)
        vbox.addWidget(QLabel("Check the rewrites to apply; select a row to "
                              "compare old vs new text."))
        lst = QListWidget()
        for i, (nid, _new, _tags, fname, tname, _tid) in enumerate(rewrites):
            item = QListWidgetItem(f"{nid}   ({fname} → {tname})")
            item.setData(Qt.ItemDataRole.UserRole, i)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            lst.addItem(item)
        vbox.addWidget(lst, stretch=1)
        preview = QTextEdit()
        preview.setReadOnly(True)
        vbox.addWidget(preview, stretch=2)

        def show_row(row):
            if row < 0:
                preview.clear()
                return
            i = lst.item(row).data(Qt.ItemDataRole.UserRole)
            nid, new_text, _tags, _f, _t, _tid = rewrites[i]
            old_text = self.script.nodes.get(nid, {}).get('text', '')
            preview.setPlainText(f"── OLD ──\n{old_text}\n\n── NEW ──\n{new_text}")
        lst.currentRowChanged.connect(show_row)
        if lst.count():
            lst.setCurrentRow(0)

        row = QHBoxLayout()
        row.addStretch(1)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(dlg.reject)
        row.addWidget(btn_cancel)
        btn_ok = QPushButton("Apply checked")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(dlg.accept)
        row.addWidget(btn_ok)
        vbox.addLayout(row)

        if dlg.exec() != QDialog.Accepted:
            self._status.setText("Callback rewrites discarded.")
            return
        n = 0
        for j in range(lst.count()):
            if lst.item(j).checkState() != Qt.CheckState.Checked:
                continue
            i = lst.item(j).data(Qt.ItemDataRole.UserRole)
            nid, new_text, add_tags, _f, _t, to_id = rewrites[i]
            if nid not in self.script.nodes:
                continue
            self.script.update_text(nid, new_text)
            nd = self.script.nodes[nid]
            # Woven marker: makes the callback weave idempotent (this
            # relation is skipped on future presses). v1/player ignore it.
            nd['callback_to'] = to_id
            for tag in add_tags:
                if tag and tag in self.script.entities \
                        and tag not in nd.get('tags', []):
                    nd.setdefault('tags', []).append(tag)
            # The audio no longer matches the rewritten text — flag for
            # regeneration by clearing the file reference.
            if nd.get('file'):
                nd['file'] = None
            self._mw._refresh_node(nid)
            n += 1
        self.script.dirty = True
        self._status.setText(
            f"Applied {n} callback rewrite(s). Regenerate audio for the "
            "rewritten nodes (their file links were cleared).")


class CanonEventsPanel(QWidget):
    """Canon Events tab — the cross-story ledger built by Extract Canon.

    Formerly a cramped 96px strip inside the Web Planner; now its own tab
    with room to read summaries, a text filter, and removal. Events are
    injected into the generation prompts of every story whose cast touches
    the event's entities (entity-less events reach every story with a cast)."""

    def __init__(self, main_window, script: 'ScriptData', parent=None):
        super().__init__(parent)
        self._mw = main_window
        self.script = script
        self._refresh_queued = False
        self._build_ui()
        self._refresh()

    def set_script(self, script: 'ScriptData'):
        self.script = script
        self._refresh()

    def _build_ui(self):
        root = QVBoxLayout(self)

        hdr = QLabel(
            "Canon events are durable happenings a generated story "
            "established — extracted with 'Extract Canon (AI)' on the "
            "Stories tab. Each event reaches the generation prompts of "
            "every story whose cast includes one of its entities (events "
            "with no entities reach every story that has a cast). "
            "Reference freely, never contradict; remove an event to "
            "retire it from all future prompts.")
        hdr.setWordWrap(True)
        hdr.setStyleSheet("color: #999; font-size: 11px;")
        root.addWidget(hdr)

        bar = QHBoxLayout()
        self._filter = QLineEdit()
        self._filter.setPlaceholderText(
            "filter — matches summary, entity slugs, source story, or id")
        self._filter.textChanged.connect(self._schedule_refresh)
        bar.addWidget(self._filter, stretch=1)
        self._count_lbl = QLabel("")
        self._count_lbl.setStyleSheet("color: #888;")
        bar.addWidget(self._count_lbl)
        btn_promote = QPushButton("Promote to Codex Event")
        btn_promote.setToolTip(
            "Create a codex entity (kind 'event') from the selected canon\n"
            "event. The card's essence is the event summary, linked to the\n"
            "involved entities. Stories can then cast the event directly,\n"
            "and it becomes chattable/editable like any codex entry.\n"
            "The ledger entry stays (marked ⇧) and gains the new slug.")
        btn_promote.clicked.connect(self._cmd_promote_to_codex)
        bar.addWidget(btn_promote)
        btn_del = QPushButton("Remove selected")
        btn_del.clicked.connect(self._cmd_remove_canon_event)
        bar.addWidget(btn_del)
        root.addLayout(bar)

        self._canon_list = QListWidget()
        self._canon_list.setAlternatingRowColors(True)
        root.addWidget(self._canon_list, stretch=1)

        self._status = QLabel("")
        self._status.setStyleSheet("color: #999; font-size: 11px;")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

    def _story_name(self, arc_id: str) -> str:
        arc = self.script.arcs.get(arc_id) or {}
        return arc.get('name') or arc_id

    def _schedule_refresh(self, *_args):
        # CRASH RULE: never rebuild item views synchronously inside Qt
        # signal dispatch (see crash.log) — coalesce onto the event loop.
        if self._refresh_queued:
            return
        self._refresh_queued = True
        QTimer.singleShot(0, self._do_scheduled_refresh)

    def _do_scheduled_refresh(self):
        self._refresh_queued = False
        self._refresh()

    def _refresh(self):
        QToolTip.hideText()   # never delete the item under an open tooltip
        needle = self._filter.text().strip().lower()
        evs = [ev for ev in self.script.canon_events() if isinstance(ev, dict)]
        self._canon_list.clear()
        shown = 0
        for ev in evs:
            src = self._story_name(ev.get('established_by', '')) \
                if ev.get('established_by') in self.script.arcs else '?'
            ents = ', '.join(ev.get('entities', [])) or '—'
            hay = ' '.join([str(ev.get('summary', '')), ents, src,
                            str(ev.get('id', ''))]).lower()
            if needle and needle not in hay:
                continue
            promoted = ev.get('promoted_to')
            mark = ' ⇧' if promoted and promoted in self.script.entities \
                else ''
            item = QListWidgetItem(
                f"[{ev.get('id', '?')}]{mark} {ev.get('summary', '')}\n"
                f"    entities: {ents}   ·   from: {src}")
            if mark:
                item.setToolTip(f"Promoted to codex entity '{promoted}'")
            item.setData(Qt.ItemDataRole.UserRole, ev.get('id'))
            self._canon_list.addItem(item)
            shown += 1
        total = len(evs)
        self._count_lbl.setText(
            f"{shown}/{total} shown" if needle else f"{total} event(s)")
        if not total:
            self._status.setText(
                "No canon events yet — generate a story, then run "
                "'Extract Canon (AI)' on the Stories tab.")

    def _cmd_remove_canon_event(self):
        item = self._canon_list.currentItem()
        if not item:
            return
        ev_id = item.data(Qt.ItemDataRole.UserRole)
        if self.script.remove_canon_event(ev_id):
            self._status.setText(
                f"Removed canon event {ev_id} — it will no longer reach "
                "any story's prompts.")
            self._schedule_refresh()

    def _cmd_promote_to_codex(self):
        """Turn the selected ledger event into a codex entity (kind
        'event'): essence = the summary, relationships = the involved
        entities. The ledger entry is kept (its reach is by-cast; the card's
        is by-casting-the-event) and gains the new slug in its entities, so
        stories that cast the event entity also receive the ledger line."""
        item = self._canon_list.currentItem()
        if not item:
            self._status.setText("Select a canon event to promote.")
            return
        ev_id = item.data(Qt.ItemDataRole.UserRole)
        ev = next((e for e in self.script.canon_events()
                   if isinstance(e, dict) and e.get('id') == ev_id), None)
        if ev is None:
            return
        prior = ev.get('promoted_to')
        if prior and prior in self.script.entities:
            self._status.setText(
                f"{ev_id} is already promoted — codex entity '{prior}'.")
            return
        summary = str(ev.get('summary', '')).strip()
        default_name = summary[:60].rstrip(' .,;:') or ev_id
        name, ok = QInputDialog.getText(
            self, "Promote to Codex Event",
            "Name for the new codex event entity:", text=default_name)
        if not ok or not name.strip():
            return
        exist = self.script.find_equivalent_entity(name.strip())
        if exist:
            # Don't mint a duplicate card — link the ledger event to the
            # existing entity instead (its card stays untouched).
            ev['promoted_to'] = exist
            if exist not in ev.setdefault('entities', []):
                ev['entities'].append(exist)
            self.script.dirty = True
            self._status.setText(
                f"Codex already has '{exist}' matching that name — linked "
                f"{ev_id} to it instead of creating a duplicate.")
            self._schedule_refresh()
            return
        slug = self.script.add_entity(name.strip(), 'event')
        if not slug:
            self._status.setText("Could not create an entity from that name.")
            return
        involved = [s for s in (ev.get('entities') or [])
                    if s in self.script.entities]
        src = self._story_name(ev.get('established_by', '')) \
            if ev.get('established_by') in self.script.arcs else '?'
        self.script.update_entity(slug, {
            'essence': summary,
            'relationships': [{'to': s, 'nature': 'involved in this event'}
                              for s in involved],
            'notes': f"Promoted from canon event {ev_id} "
                     f"(established by: {src}).",
        })
        ev['promoted_to'] = slug
        if slug not in ev.setdefault('entities', []):
            ev['entities'].append(slug)
        self.script.dirty = True
        self._status.setText(
            f"Promoted {ev_id} → codex entity '{slug}' (kind: event). "
            "Edit its card on the Codex tab; cast it in stories to make "
            "them orbit this event.")
        self._schedule_refresh()


class ArcEditorDialog(QDialog):
    """Popup dialog for editing story arcs and wiring them into generation."""

    LAYER_NAMES = LAYER_ORDER

    def __init__(self, script: 'ScriptData', ai: 'AIAssistant',
                 ui_queue: queue.SimpleQueue, on_graph_generated=None,
                 register_orchestrator=None, embedded: bool = False,
                 parent=None):
        super().__init__(parent)
        self._register_orchestrator = register_orchestrator
        self._embedded = embedded
        self.setWindowTitle("Stories")
        if not embedded:
            self.setMinimumSize(940, 720)
        self.script = script
        self._main_ai = ai
        self.ui_queue = ui_queue
        self._on_graph_generated = on_graph_generated
        self._arc_ai = AIAssistant()   # separate instance for arc chat
        self._current_arc_id: Optional[str] = None
        # Generation queue: one full story run at a time (a run is already
        # 8 concurrent AI calls); further Generate clicks enqueue.
        self._gen_queue: list = []
        self._gen_running = False
        self._gen_current: Optional[str] = None
        self._loading = False           # suppress dirty callbacks while populating fields
        self._call_start_time: Optional[float] = None  # for elapsed-time status updates
        self._build_ui()
        self._refresh_arc_list()
        # Tick the chat-status while an AI call is in flight so the user
        # sees "thinking… 12s" instead of a frozen label.
        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._tick_thinking_status)
        self._tick_timer.start(250)
        # Manually load the first arc since _refresh_arc_list blocks signals
        if self.script.arcs:
            first_id = next(iter(self.script.arcs))
            self._current_arc_id = first_id
            self._load_arc(first_id)

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)

        # ── Top: list + editor side-by-side ──────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: arc list
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 4, 0)
        ll.addWidget(QLabel("Stories"))
        self.arc_list = QListWidget()
        self.arc_list.setMinimumWidth(180)
        self.arc_list.setMaximumWidth(250)
        self.arc_list.currentRowChanged.connect(self._on_arc_selected)
        ll.addWidget(self.arc_list)
        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ New")
        add_btn.clicked.connect(self._cmd_new_arc)
        btn_row.addWidget(add_btn)
        del_btn = QPushButton("Delete")
        del_btn.clicked.connect(self._cmd_delete_arc)
        btn_row.addWidget(del_btn)
        ll.addLayout(btn_row)
        splitter.addWidget(left)

        # Right: scrollable editor
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        rw = QWidget()
        rl = QVBoxLayout(rw)
        rl.setSpacing(6)

        form_top = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Arc name")
        self.name_edit.textChanged.connect(self._on_field_changed)
        form_top.addRow("Name:", self.name_edit)
        rl.addLayout(form_top)

        # Lifecycle status for the selected story: node count + canon state
        self.story_status = QLabel("")
        self.story_status.setStyleSheet("color: #9ab; font-size: 11px;")
        self.story_status.setWordWrap(True)
        rl.addWidget(self.story_status)

        rl.addWidget(QLabel("Premise:"))
        self.premise_edit = QTextEdit()
        self.premise_edit.setPlaceholderText(
            "What is this story about? Setting, characters, central conflict…")
        self.premise_edit.setFixedHeight(80)
        self.premise_edit.textChanged.connect(self._on_field_changed)
        rl.addWidget(self.premise_edit)

        form_mid = QFormLayout()
        self.themes_edit = QLineEdit()
        self.themes_edit.setPlaceholderText("isolation, memory, decay  (comma-separated)")
        self.themes_edit.textChanged.connect(self._on_field_changed)
        form_mid.addRow("Themes:", self.themes_edit)

        self.motif_edit = QLineEdit()
        self.motif_edit.setPlaceholderText(
            "Recurring thread woven through every layer, e.g. 'always reference the bell'")
        self.motif_edit.textChanged.connect(self._on_field_changed)
        form_mid.addRow("Recurring motif:", self.motif_edit)
        rl.addLayout(form_mid)

        # Per-arc node-length override — single dropdown. First option
        # is "Inherit from script default", then the canonical presets.
        # If an arc has a non-preset value, a "Custom (lo-hi)" entry is
        # added dynamically at load time.
        nl_row = QHBoxLayout()
        nl_row.addWidget(QLabel("Node length:"))
        self.arc_nl_combo = QComboBox()
        self.arc_nl_combo.addItem("Inherit from script default", None)
        for label, lo, hi in NODE_LENGTH_PRESETS:
            self.arc_nl_combo.addItem(label, (lo, hi))
        self.arc_nl_combo.setMinimumWidth(360)
        self.arc_nl_combo.currentIndexChanged.connect(self._on_field_changed)
        nl_row.addWidget(self.arc_nl_combo, stretch=1)
        rl.addLayout(nl_row)

        cast_sep = QLabel("Cast — entities in play")
        cast_sep.setStyleSheet("font-weight: bold; margin-top: 6px;")
        rl.addWidget(cast_sep)
        cast_hint = QLabel(
            "Checked codex entities are injected as canon cards into every "
            "generation call for this story. Nothing checked = NOTHING from "
            "the codex is injected — check what belongs in this story.")
        cast_hint.setStyleSheet("color: #888888; font-size: 10px;")
        cast_hint.setWordWrap(True)
        rl.addWidget(cast_hint)
        self.cast_list = QListWidget()
        self.cast_list.setFixedHeight(110)
        self.cast_list.itemChanged.connect(lambda *_: self._on_field_changed())
        rl.addWidget(self.cast_list)

        sep = QLabel("Structure — Story Beats")
        sep.setStyleSheet("font-weight: bold; margin-top: 6px;")
        rl.addWidget(sep)
        hint = QLabel(
            "One row per generation beat, in order. The archetype sets the beat's "
            "story function, node tag, and pacing defaults; the direction is this "
            "story's specific intent (blank = archetype default). Add, remove, and "
            "reorder freely — the classic 10-layer arc is just the default shape.")
        hint.setStyleSheet("color: #888888; font-size: 10px;")
        hint.setWordWrap(True)
        rl.addWidget(hint)

        self._structure_box = QVBoxLayout()
        self._structure_box.setSpacing(2)
        self._structure_rows: list = []   # [(row_widget, archetype_combo, direction_edit)]
        rl.addLayout(self._structure_box)

        add_beat_btn = QPushButton("+ Add beat")
        add_beat_btn.setFixedWidth(90)
        add_beat_btn.clicked.connect(self._cmd_add_beat)
        rl.addWidget(add_beat_btn)

        rl.addWidget(QLabel("Notes:"))
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText(
            "Character details, world-building, things to remember…")
        self.notes_edit.setFixedHeight(70)
        self.notes_edit.textChanged.connect(self._on_field_changed)
        rl.addWidget(self.notes_edit)
        rl.addStretch()

        scroll.setWidget(rw)
        splitter.addWidget(scroll)
        splitter.setSizes([170, 700])
        root.addWidget(splitter, stretch=3)

        # ── Arc development chat ──────────────────────────────────
        chat_hdr = QLabel("Arc Development Chat")
        chat_hdr.setStyleSheet("font-weight: bold; margin-top: 4px;")
        root.addWidget(chat_hdr)

        self.chat_log = QTextEdit()
        self.chat_log.setReadOnly(True)
        self.chat_log.setStyleSheet("background:#1a1a1a; color:#cccccc; font-size:11px;")
        root.addWidget(self.chat_log, stretch=1)

        input_row = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask AI to help develop this arc…")
        self.chat_input.returnPressed.connect(self._cmd_arc_chat)
        input_row.addWidget(self.chat_input)
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._cmd_arc_chat)
        input_row.addWidget(send_btn)
        root.addLayout(input_row)

        self.chat_status = QLabel("")
        self.chat_status.setStyleSheet("color:#888888; font-size:10px;")
        root.addWidget(self.chat_status)

        # ── Bottom buttons ────────────────────────────────────────
        bot = QHBoxLayout()
        distill_btn = QPushButton("Distill Chat → Story")
        distill_btn.setToolTip("Use AI to extract premise, themes, motif, and structure from the chat conversation")
        distill_btn.clicked.connect(self._cmd_distill_chat_to_arc)
        bot.addWidget(distill_btn)

        canon_btn = QPushButton("Extract Canon (AI)")
        canon_btn.setToolTip(
            "Read this story's generated nodes and extract durable canon:\n"
            "new facts for cast entity cards + canon events other stories\n"
            "can reference. You review before anything is written.")
        canon_btn.clicked.connect(lambda: self._cmd_extract_canon())
        bot.addWidget(canon_btn)

        self.gen_btn = QPushButton("Generate Graph from Story")
        self.gen_btn.setStyleSheet("font-weight: bold;")
        self.gen_btn.setToolTip(
            "Generate this story's node graph. While a run is active, "
            "clicking queues the selected story to generate next.")
        self.gen_btn.clicked.connect(self._cmd_generate_from_arc_parallel)
        bot.addWidget(self.gen_btn)

        gen_all_btn = QPushButton("Generate All Empty")
        gen_all_btn.setToolTip(
            "Queue every story that has no nodes yet; they generate one "
            "after another (each run is itself 8-way parallel).")
        gen_all_btn.clicked.connect(self._cmd_generate_all)
        bot.addWidget(gen_all_btn)

        self.auto_canon_cb = QCheckBox("Auto-canon")
        self.auto_canon_cb.setChecked(True)
        self.auto_canon_cb.setToolTip(
            "Pipeline mode: when a story finishes generating, extract its\n"
            "canon automatically (no review picker — everything the AI\n"
            "returns is applied; prune later on the Codex / Canon Events\n"
            "tabs). Runs alongside the next queued generation, so\n"
            "'Generate All Empty' leaves every story canon-stamped.\n"
            "Uncheck to review each extraction by hand instead.")
        bot.addWidget(self.auto_canon_cb)

        # Per-arc generation-width override. First entry inherits from the
        # script-wide setting; the other three pin a specific preset for
        # this arc. Saved with the arc on close / save.
        bot.addWidget(QLabel("Width:"))
        self.gen_width_combo = QComboBox()
        self.gen_width_combo.setToolTip(
            "Per-arc generation width. 'Inherit from script' uses the "
            "Story menu's Generation Width setting; the others pin this "
            "arc to a specific size regardless of the script default.")
        self.gen_width_combo.addItem("Inherit from script", None)
        for preset in ('small', 'medium', 'large'):
            self.gen_width_combo.addItem(preset.capitalize(), preset)
        self.gen_width_combo.currentIndexChanged.connect(self._on_field_changed)
        bot.addWidget(self.gen_width_combo)

        bot.addStretch()
        if not self._embedded:
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(self.accept)
            bot.addWidget(close_btn)
        root.addLayout(bot)

    # ── Structure rows + cast picker (v2) ─────────────────────────────────────

    def _cmd_add_beat(self):
        self._add_structure_row()
        self._on_field_changed()

    def _add_structure_row(self, layer: str = 'arrival', direction: str = ''):
        row_w = QWidget()
        h = QHBoxLayout(row_w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        combo = QComboBox()
        for lname in LAYER_ORDER:
            combo.addItem(lname, lname)
        idx = combo.findData(layer)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.setFixedWidth(110)
        combo.currentIndexChanged.connect(self._on_field_changed)
        h.addWidget(combo)
        edit = QLineEdit(direction)
        edit.setPlaceholderText("What this beat must cover (blank = archetype default)")
        edit.textChanged.connect(self._on_field_changed)
        h.addWidget(edit, stretch=1)
        for label, cb in (("↑", lambda *_, w=row_w: self._move_structure_row(w, -1)),
                          ("↓", lambda *_, w=row_w: self._move_structure_row(w, +1)),
                          ("✕", lambda *_, w=row_w: self._remove_structure_row(w))):
            b = QPushButton(label)
            b.setFixedWidth(24)
            b.clicked.connect(cb)
            h.addWidget(b)
        self._structure_box.addWidget(row_w)
        self._structure_rows.append((row_w, combo, edit))

    def _row_index(self, row_w) -> int:
        for i, (w, _, _) in enumerate(self._structure_rows):
            if w is row_w:
                return i
        return -1

    def _move_structure_row(self, row_w, delta: int):
        i = self._row_index(row_w)
        j = i + delta
        if i < 0 or not (0 <= j < len(self._structure_rows)):
            return
        self._structure_rows[i], self._structure_rows[j] = \
            self._structure_rows[j], self._structure_rows[i]
        self._structure_box.removeWidget(row_w)
        self._structure_box.insertWidget(j, row_w)
        self._on_field_changed()

    def _remove_structure_row(self, row_w):
        if len(self._structure_rows) <= 1:
            return  # a story keeps at least one beat
        i = self._row_index(row_w)
        if i < 0:
            return
        self._structure_rows.pop(i)
        self._structure_box.removeWidget(row_w)
        row_w.deleteLater()
        self._on_field_changed()

    def _set_structure_rows(self, structure: list):
        """Rebuild the beat rows from a structure list (suppresses the
        per-row change signals while populating)."""
        prev = self._loading
        self._loading = True
        try:
            for (w, _, _) in self._structure_rows:
                self._structure_box.removeWidget(w)
                w.deleteLater()
            self._structure_rows = []
            for e in (structure or ScriptData.default_structure()):
                self._add_structure_row(e.get('layer', 'arrival'),
                                        e.get('direction', ''))
        finally:
            self._loading = prev

    def _structure_from_rows(self) -> list:
        return [{'layer': combo.currentData(),
                 'direction': edit.text().strip()}
                for (_, combo, edit) in self._structure_rows]

    def _reload_cast_list(self, selected: set):
        """Repopulate the cast picker from the current codex."""
        self.cast_list.blockSignals(True)
        try:
            self.cast_list.clear()
            ents = self.script.entities
            def _order(kv):
                kind = kv[1].get('kind', 'idea')
                ki = ENTITY_KINDS.index(kind) if kind in ENTITY_KINDS else 99
                return (ki, (kv[1].get('name') or kv[0]).lower())
            if not ents:
                placeholder = QListWidgetItem(
                    "(codex is empty — Story ▸ Codex… to add entities)")
                placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
                self.cast_list.addItem(placeholder)
                return
            for slug, card in sorted(ents.items(), key=_order):
                kind = card.get('kind', 'idea')
                item = QListWidgetItem(
                    f"[{kind[:4]}] {card.get('name') or slug}  ({slug})")
                item.setData(Qt.ItemDataRole.UserRole, slug)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if slug in selected
                                   else Qt.CheckState.Unchecked)
                self.cast_list.addItem(item)
        finally:
            self.cast_list.blockSignals(False)

    def _cast_from_list(self) -> list:
        out = []
        for i in range(self.cast_list.count()):
            item = self.cast_list.item(i)
            slug = item.data(Qt.ItemDataRole.UserRole)
            if slug and item.checkState() == Qt.CheckState.Checked:
                out.append(slug)
        return out

    # ── Arc list management ──────────────────────────────────────────────────

    def _story_status_maps(self) -> tuple:
        """(node_counts, event_counts) keyed by arc_id — one pass each."""
        node_counts: dict = {}
        for nid in self.script.nodes:
            a = self.script.get_node_arc_id(nid)
            node_counts[a] = node_counts.get(a, 0) + 1
        event_counts: dict = {}
        for ev in self.script.canon_events():
            if isinstance(ev, dict):
                src = ev.get('established_by', '')
                event_counts[src] = event_counts.get(src, 0) + 1
        return node_counts, event_counts

    def _arc_list_label(self, arc_id: str, active_id: str,
                        node_counts: dict, event_counts: dict) -> tuple:
        """(text, tooltip) for a story list entry — one compact line:
        '★ Name · 42n ✓' (node count; ✓ = canon extracted). The tooltip
        carries the verbose status."""
        arc = self.script.arcs.get(arc_id, {})
        name = arc.get('name') or arc_id
        star = '★ ' if arc_id == active_id else '  '
        n = node_counts.get(arc_id, 0)
        n_ev = event_counts.get(arc_id, 0)
        extracted = bool(arc.get('canon_extracted_at')) or n_ev > 0
        text = f'{star}{name} · {n}n' + (' ✓' if extracted else '')
        tip = f'{n} node(s) generated' if n else 'No nodes generated yet'
        if extracted:
            tip += ' · canon extracted' + (f' ({n_ev} event(s))' if n_ev else '')
            if arc.get('canon_extracted_at'):
                tip += f" — {arc['canon_extracted_at']}"
        elif n:
            tip += ' · canon not extracted yet'
        return text, tip

    def _update_story_status(self):
        """Status line in the editor pane for the selected story."""
        arc_id = self._current_arc_id
        if not arc_id or arc_id not in self.script.arcs:
            self.story_status.setText('')
            return
        node_counts, event_counts = self._story_status_maps()
        n = node_counts.get(arc_id, 0)
        n_ev = event_counts.get(arc_id, 0)
        arc = self.script.arcs[arc_id]
        stamp = arc.get('canon_extracted_at')
        if not n:
            text = ("No nodes generated yet — use 'Generate Graph from "
                    "Story' below.")
        else:
            text = f"{n} node(s) generated"
            if stamp or n_ev:
                text += f" · canon extracted"
                if n_ev:
                    text += f" — {n_ev} event(s)"
                if stamp:
                    text += f" ({stamp})"
            else:
                text += (" · canon NOT extracted — run 'Extract Canon (AI)' "
                         "so other stories can reference this one.")
        self.story_status.setText(text)

    def _refresh_arc_list(self):
        # On the SELECTION-click path the arc set is unchanged — only the
        # ★ active marker shifts. The previous implementation called
        # clear() + addItem() each time, which (a) reset the scrollbar
        # and (b) caused a native crash in PySide6 on at least one
        # machine (see logs/narrative_editor/crash.log: native crash at
        # arc_list.clear() called from _on_arc_selected). Detect "same
        # arc IDs in same order" and update labels in place; only do a
        # full clear/repopulate when the arc set actually changed.
        scroll_value = self.arc_list.verticalScrollBar().value()
        active_id    = self.script.active_arc_id
        desired_ids  = list(self.script.arcs.keys())
        current_ids  = [self.arc_list.item(i).data(Qt.ItemDataRole.UserRole)
                        for i in range(self.arc_list.count())]

        node_counts, event_counts = self._story_status_maps()
        self.arc_list.blockSignals(True)
        try:
            if current_ids == desired_ids and current_ids:
                # In-place label update — no clear(), no item recreation.
                for i, arc_id in enumerate(desired_ids):
                    text, tip = self._arc_list_label(
                        arc_id, active_id, node_counts, event_counts)
                    self.arc_list.item(i).setText(text)
                    self.arc_list.item(i).setToolTip(tip)
            else:
                # Arc set genuinely changed (add / delete / rename / first
                # population) — full rebuild is unavoidable.
                self.arc_list.clear()
                for arc_id in self.script.arcs:
                    text, tip = self._arc_list_label(
                        arc_id, active_id, node_counts, event_counts)
                    item = QListWidgetItem(text)
                    item.setToolTip(tip)
                    item.setData(Qt.ItemDataRole.UserRole, arc_id)
                    self.arc_list.addItem(item)

            # Sync selection to _current_arc_id
            target = self._current_arc_id
            for i in range(self.arc_list.count()):
                if self.arc_list.item(i).data(Qt.ItemDataRole.UserRole) == target:
                    self.arc_list.setCurrentRow(i)
                    break
            else:
                if self.arc_list.count():
                    self.arc_list.setCurrentRow(0)
        finally:
            self.arc_list.blockSignals(False)
            self.arc_list.verticalScrollBar().setValue(scroll_value)

    def _on_arc_selected(self, row):
        self._save_current()
        if row < 0:
            self._current_arc_id = None
            QTimer.singleShot(0, self._clear_fields)
            return
        arc_id = self.arc_list.item(row).data(Qt.ItemDataRole.UserRole)
        self._current_arc_id = arc_id
        self.script.set_active_arc(arc_id)  # selected arc is always the active one
        self._refresh_arc_list()
        # CRASH RULE: _load_arc rebuilds the cast list and structure rows;
        # doing that synchronously inside currentRowChanged dispatch AV'd
        # natively (crash.log 2026-07-17 19:09 — cast_list.clear() in
        # _reload_cast_list). Defer to the event loop; the guard drops the
        # reload if the selection moved again before the timer fired.
        QTimer.singleShot(0, lambda a=arc_id: self._load_arc_deferred(a))

    def _load_arc_deferred(self, arc_id: str):
        if arc_id == self._current_arc_id and arc_id in self.script.arcs:
            self._load_arc(arc_id)

    def _load_arc(self, arc_id: str):
        # The story list rows have hover tooltips; never destroy widgets
        # under an open native tooltip (same AV family as the planner).
        QToolTip.hideText()
        arc = self.script.arcs.get(arc_id, {})
        self._loading = True
        self.name_edit.setText(arc.get('name', ''))
        self.premise_edit.setPlainText(arc.get('premise', ''))
        self.themes_edit.setText(arc.get('themes', ''))
        self.motif_edit.setText(arc.get('motif', ''))
        self._set_structure_rows(self.script.get_story_structure(arc_id))
        self._reload_cast_list(set(self.script.get_story_cast(arc_id)))
        self.notes_edit.setPlainText(arc.get('notes', ''))
        # Per-arc node-length override dropdown. Default: "Inherit from
        # script default" (item 0). If the arc has a custom range that
        # doesn't match any preset, we add it as a "Custom (lo-hi)" item
        # so the user can keep it without flipping back to inherit.
        self._reset_arc_nl_combo()
        arc_nwr = self.script.get_arc_node_word_range(arc_id)
        if arc_nwr:
            preset_idx = _find_node_length_preset_index(arc_nwr)
            if preset_idx >= 0:
                self.arc_nl_combo.setCurrentIndex(preset_idx + 1)   # +1 for inherit slot
            else:
                # Surface the custom value as an extra item
                self.arc_nl_combo.addItem(
                    f"Custom ({arc_nwr[0]}-{arc_nwr[1]} words)",
                    tuple(arc_nwr))
                self.arc_nl_combo.setCurrentIndex(self.arc_nl_combo.count() - 1)
        else:
            self.arc_nl_combo.setCurrentIndex(0)

        # Per-arc generation-width override (None = inherit from script).
        arc_width = self.script.get_arc_width_preset(arc_id)
        self.gen_width_combo.blockSignals(True)
        idx = self.gen_width_combo.findData(arc_width)
        self.gen_width_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.gen_width_combo.blockSignals(False)

        self.chat_log.clear()
        for entry in arc.get('chat_history', []):
            self._append_chat(entry.get('role', 'user'), entry.get('content', ''))
        self._loading = False
        self._update_story_status()

    def _reset_arc_nl_combo(self):
        """Rebuild the per-arc node-length combo to its canonical list
        (inherit + presets). Drops any stale 'Custom (...)' item that
        was added for a previously-loaded arc."""
        self.arc_nl_combo.blockSignals(True)
        self.arc_nl_combo.clear()
        self.arc_nl_combo.addItem("Inherit from script default", None)
        for label, lo, hi in NODE_LENGTH_PRESETS:
            self.arc_nl_combo.addItem(label, (lo, hi))
        self.arc_nl_combo.blockSignals(False)

    def _clear_fields(self):
        self._loading = True
        self.name_edit.clear()
        self.premise_edit.clear()
        self.themes_edit.clear()
        self.motif_edit.clear()
        self._set_structure_rows(ScriptData.default_structure())
        self._reload_cast_list(set())
        self.notes_edit.clear()
        self._reset_arc_nl_combo()
        self.arc_nl_combo.setCurrentIndex(0)
        self.gen_width_combo.blockSignals(True)
        self.gen_width_combo.setCurrentIndex(0)  # inherit
        self.gen_width_combo.blockSignals(False)
        self.chat_log.clear()
        self.story_status.setText('')
        self._loading = False

    def _save_current(self):
        if not self._current_arc_id or self._loading:
            return
        arc_id = self._current_arc_id
        if arc_id not in self.script.arcs:
            return
        structure = self._structure_from_rows()
        self.script.save_arc(arc_id, {
            'name':      self.name_edit.text(),
            'premise':   self.premise_edit.toPlainText(),
            'themes':    self.themes_edit.text(),
            'motif':     self.motif_edit.text(),
            'structure': structure,
            'beats':     ScriptData.beats_from_structure(structure),
            'cast':      self._cast_from_list(),
            'notes':     self.notes_edit.toPlainText(),
        })
        # Per-arc node-length override — None = inherit (combo index 0)
        data = self.arc_nl_combo.currentData()
        if data is None:
            self.script.set_arc_node_word_range(arc_id, None, None)
        else:
            lo, hi = data
            self.script.set_arc_node_word_range(arc_id, int(lo), int(hi))
        # Per-arc generation-width override — None = inherit
        self.script.set_arc_width_preset(arc_id, self.gen_width_combo.currentData())
        self._refresh_list_item(arc_id)

    def _refresh_list_item(self, arc_id: str):
        active_id = self.script.active_arc_id
        node_counts, event_counts = self._story_status_maps()
        for i in range(self.arc_list.count()):
            item = self.arc_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == arc_id:
                text, tip = self._arc_list_label(
                    arc_id, active_id, node_counts, event_counts)
                item.setText(text)
                item.setToolTip(tip)
                return

    def _on_field_changed(self):
        if not self._loading:
            self._save_current()

    def _cmd_new_arc(self):
        self._save_current()
        arc_id = self.script.add_arc()
        # If this is the first story, seed it with whatever is already in the fields
        if len(self.script.arcs) == 1:
            structure = self._structure_from_rows()
            name = self.name_edit.text().strip() or 'New Story'
            self.script.save_arc(arc_id, {
                'name':      name,
                'premise':   self.premise_edit.toPlainText(),
                'themes':    self.themes_edit.text(),
                'motif':     self.motif_edit.text(),
                'structure': structure,
                'beats':     ScriptData.beats_from_structure(structure),
                'cast':      self._cast_from_list(),
                'notes':     self.notes_edit.toPlainText(),
            })
        self._refresh_arc_list()
        for i in range(self.arc_list.count()):
            if self.arc_list.item(i).data(Qt.ItemDataRole.UserRole) == arc_id:
                self.arc_list.setCurrentRow(i)
                break

    def _cmd_delete_arc(self):
        if not self._current_arc_id:
            return
        arc_id = self._current_arc_id
        self._current_arc_id = None
        self.script.delete_arc(arc_id)
        self._refresh_arc_list()
        if not self.arc_list.count():
            self._clear_fields()

    def _cmd_generate_from_arc_parallel(self):
        """Generate the selected story — or queue it while a run is active."""
        if not self._current_arc_id:
            self.chat_status.setText("No story selected.")
            return
        self._save_current()
        if not self._main_ai.ready:
            self.chat_status.setText("claude CLI not found.")
            return
        arc_id = self._current_arc_id
        if self._gen_running:
            name = self.script.arcs.get(arc_id, {}).get('name') or arc_id
            if arc_id == self._gen_current:
                self.chat_status.setText(f"'{name}' is generating right now.")
            elif arc_id in self._gen_queue:
                self.chat_status.setText(f"'{name}' is already queued "
                                         f"(position {self._gen_queue.index(arc_id) + 1}).")
            else:
                self._gen_queue.append(arc_id)
                self.chat_status.setText(
                    f"Queued '{name}' — {len(self._gen_queue)} story(ies) "
                    "waiting; runs start automatically.")
            return
        self._start_generation(arc_id)

    def _cmd_generate_all(self):
        """Queue every story that has no nodes yet."""
        if not self._main_ai.ready:
            self.chat_status.setText("claude CLI not found.")
            return
        self._save_current()
        node_counts, _ = self._story_status_maps()
        pending = [aid for aid in self.script.arcs
                   if node_counts.get(aid, 0) == 0
                   and aid != self._gen_current
                   and aid not in self._gen_queue]
        if not pending:
            self.chat_status.setText(
                "Every story already has nodes (or is queued). Select one "
                "and Generate to extend it instead.")
            return
        self._gen_queue.extend(pending)
        self.chat_status.setText(
            f"Queued {len(pending)} empty story(ies); generating one at a "
            "time.")
        if not self._gen_running:
            self._start_next_generation()

    def _start_next_generation(self):
        while self._gen_queue and not self._gen_running:
            arc_id = self._gen_queue.pop(0)
            if arc_id in self.script.arcs:
                self._start_generation(arc_id)
                return

    def _finish_generation(self, message: str, cancelled: bool = False):
        self._gen_running = False
        self._gen_current = None
        self.gen_btn.setText("Generate Graph from Story")
        self._refresh_arc_list()
        self._update_story_status()
        if cancelled and self._gen_queue:
            n = len(self._gen_queue)
            self._gen_queue.clear()
            message += f" Queue cleared ({n} story(ies) dropped)."
        elif self._gen_queue:
            nxt = self.script.arcs.get(self._gen_queue[0], {})
            message += (f" Next in queue: "
                        f"'{nxt.get('name') or self._gen_queue[0]}'.")
            QTimer.singleShot(0, self._start_next_generation)
        self.chat_status.setText(message)

    def _start_generation(self, arc_id: str):
        """Run one story's seed + orchestration (called by the queue)."""
        arc = self.script.arcs.get(arc_id, {})

        prompt = arc.get('premise', '').strip() or arc.get('name', 'Generate a narrative graph.')
        structure = self.script.get_story_structure(arc_id)
        cast_codex = self.script.cast_codex_for(arc_id)
        arc_motif = arc.get('motif', '')
        arc_notes = arc.get('notes', '').strip()

        # Combine story context with arc notes so character/world details reach every node
        story_ctx = self.script.story_context_focused
        if arc_notes:
            story_ctx = (story_ctx + '\n\n' + arc_notes).strip() if story_ctx else arc_notes

        self._gen_running = True
        self._gen_current = arc_id
        self.gen_btn.setText("Queue Generation")
        self.chat_status.setText(
            f"Generating seed nodes for '{arc.get('name') or arc_id}'…")
        self._append_chat('assistant', f'[Parallel gen from arc: {arc.get("name", "")}]')

        arc_id_for_seeds = arc_id

        def on_seed_done(data):
            before = set(self.script.nodes.keys())
            self.script.apply_generated(data)
            after = set(self.script.nodes.keys())
            seed_ids = sorted(after - before)

            for nid in seed_ids:
                if nid in self.script.nodes:
                    self.script.set_start(nid, True)
                    self.script.set_node_arc_id(nid, arc_id_for_seeds)
                    # Seeds are beat 0 of this story's structure.
                    self.script.nodes[nid]['beat'] = 0

            self._append_chat('assistant', f'Seeds: {", ".join(seed_ids)}')
            self.chat_status.setText(f'Seeds done. Starting parallel expansion…')

            def on_complete():
                if self._on_graph_generated:
                    self._on_graph_generated()
                cancelled = orch._cancelled.is_set()
                if cancelled:
                    self._finish_generation("Generation stopped.",
                                            cancelled=True)
                else:
                    name = self.script.arcs.get(arc_id_for_seeds, {}).get(
                        'name') or arc_id_for_seeds
                    self._finish_generation(
                        f"'{name}' generated — {len(self.script.nodes)} "
                        "total nodes.")
                    # Pipeline: extraction runs in its own thread, so the
                    # generation queue advances in parallel with it.
                    if self.auto_canon_cb.isChecked():
                        QTimer.singleShot(0, lambda: self._cmd_extract_canon(
                            arc_id=arc_id_for_seeds, auto=True))

            orch = make_full_orchestrator(
                self.script, self._main_ai, self.ui_queue,
                story_context=story_ctx,
                width_preset=self.script.get_effective_width_preset(arc_id_for_seeds),
                on_progress=lambda msg: self.chat_status.setText(msg),
                on_complete=on_complete,
                on_node_added=None,  # don't rebuild graph per-node; full rebuild on_complete
                register=self._register_orchestrator,
            )
            self._orchestrator = orch
            orch.start(seed_ids)

        def on_seed_error(e):
            self._finish_generation(f"Seed error: {e[:60]} — moving on.")

        self._main_ai.generate_seed(
            prompt, self.ui_queue, on_seed_done, on_seed_error,
            story_context=story_ctx,
            layer_direction=structure[0].get('direction', ''),
            first_layer=structure[0].get('layer', 'arrival'),
            cast_codex=cast_codex,
            motif=arc_motif,
            variables=self.script.variables,
            node_word_range=self.script.get_effective_node_word_range(arc_id),
            n_seeds=_seed_count_for_story(self.script, arc_id),
        )

    # ── Arc chat ─────────────────────────────────────────────────────────────

    def _build_arc_context(self) -> str:
        if not self._current_arc_id:
            return ''
        arc = self.script.arcs.get(self._current_arc_id, {})
        parts = []
        if arc.get('name'):
            parts.append(f"Arc: {arc['name']}")
        if arc.get('premise'):
            parts.append(f"Premise: {arc['premise']}")
        if arc.get('themes'):
            parts.append(f"Themes: {arc['themes']}")
        if arc.get('motif'):
            parts.append(f"Recurring motif: {arc['motif']}")
        cast = self.script.get_story_cast(self._current_arc_id)
        if cast:
            names = [self.script.entities.get(s, {}).get('name') or s for s in cast]
            parts.append(f"Cast: {', '.join(names)}")
        structure = self.script.get_story_structure(self._current_arc_id)
        parts.append(f"Structure ({len(structure)} beats): "
                     + ' → '.join(e['layer'] for e in structure))
        filled = [(i, e) for i, e in enumerate(structure) if e['direction'].strip()]
        if filled:
            parts.append('Beat directions:')
            for i, e in filled:
                parts.append(f'  {i + 1}. {e["layer"]}: {e["direction"]}')
        if arc.get('notes'):
            parts.append(f"Notes: {arc['notes']}")
        return '\n'.join(parts)

    def _append_chat(self, role: str, text: str):
        color = '#88ccff' if role == 'assistant' else '#cccccc'
        label = 'Claude' if role == 'assistant' else 'You'
        self.chat_log.append(
            f'<span style="color:{color};"><b>{label}:</b> {text}</span><br>')

    def _tick_thinking_status(self):
        """Refresh chat_status with elapsed seconds while a call is in flight."""
        if self._call_start_time is None:
            return
        elapsed = time.time() - self._call_start_time
        name = _model_short_name(getattr(self, '_call_model', ''))
        self.chat_status.setText(f"Claude ({name}) is thinking… {elapsed:0.0f}s")

    def _cmd_extract_canon(self, arc_id: str = '', auto: bool = False):
        """Extract canon (entity facts + canon events) from a story's
        generated nodes — the cross-pollination ledger (v2 Phase 3).

        auto=True is the pipeline mode (fired when a generation finishes):
        operates on the given arc_id instead of the selection, applies
        everything without the review picker (prune later on the Codex /
        Canon Events tabs), and stays quiet in the chat pane unless the
        story is the one on screen."""
        if auto:
            if arc_id not in self.script.arcs:
                return
        else:
            if not self._current_arc_id:
                self.chat_status.setText("No story selected.")
                return
            self._save_current()
            arc_id = self._current_arc_id
        if not self._main_ai.ready:
            self.chat_status.setText("Claude CLI not found.")
            return
        arc = self.script.arcs.get(arc_id, {})

        story_nodes = [(nid, nd) for nid, nd in self.script.nodes.items()
                       if self.script.get_node_arc_id(nid) == arc_id
                       and nd.get('text')]
        if not story_nodes:
            self.chat_status.setText("This story has no generated nodes yet.")
            return

        # Cards + existing canon go in the (cached) system prompt so the
        # model can avoid restating what is already recorded.
        system = self._main_ai._augment_system_with_context(
            SYSTEM_EXTRACT_CANON, '',
            cast_codex=self.script.cast_codex_for(arc_id))
        parts = [f'STORY "{arc.get("name") or arc_id}":\n'
                 f'  premise: {(arc.get("premise") or "")[:600]}']
        text_lines = ['NODE TEXTS:']
        for nid, nd in story_nodes[:50]:
            text_lines.append(f'  [{nid}] "{nd.get("text", "")[:400]}"')
        parts.append('\n'.join(text_lines))
        parts.append('Extract the new canon now. Output the JSON object only.')
        prompt = '\n\n'.join(parts)

        if auto:
            self.chat_status.setText(
                f'Auto-extracting canon from "{arc.get("name") or arc_id}" '
                f'({len(story_nodes)} nodes)…')
        else:
            self._call_start_time = time.time()
            self._call_model = self._main_ai.model
            self._tick_thinking_status()
            self._append_chat('assistant',
                              f'[Extracting canon from {len(story_nodes)} nodes...]')

        def on_done(data):
            self._call_start_time = None
            facts = data.get('facts') if isinstance(data, dict) else None
            events = data.get('events') if isinstance(data, dict) else None
            facts = facts if isinstance(facts, dict) else {}
            events = [e for e in (events or [])
                      if isinstance(e, dict) and (e.get('summary') or '').strip()]
            # Only facts for entities that exist, and only genuinely new ones
            entries = []
            payload = {}
            for slug, new_facts in facts.items():
                card = self.script.entities.get(slug)
                if not card or not isinstance(new_facts, list):
                    continue
                have = set(card.get('facts', []))
                for j, fact in enumerate(new_facts):
                    fact = str(fact).strip()
                    if fact and fact not in have:
                        key = f'fact:{slug}:{j}'
                        payload[key] = (slug, fact)
                        entries.append((key, f'[fact] {slug}: {fact[:110]}'))
            for j, ev in enumerate(events):
                key = f'event:{j}'
                ents = [ScriptData.sanitize_entity_slug(str(x))
                        for x in (ev.get('entities') or [])]
                payload[key] = (str(ev['summary']).strip(), ents)
                entries.append((key, f'[event] {str(ev["summary"]).strip()[:120]}'))
            if not entries:
                # Extraction ran and found nothing new — still counts as
                # extracted for the lifecycle status.
                self.script.save_arc(arc_id, {
                    'canon_extracted_at': time.strftime('%Y-%m-%d %H:%M')})
                self._refresh_list_item(arc_id)
                self._update_story_status()
                self.chat_status.setText(
                    ('Auto-canon: ' if auto else 'Extraction ')
                    + 'found no new canon'
                    + (f' in "{arc.get("name") or arc_id}".' if auto else '.'))
                return
            if auto:
                # Pipeline mode: apply everything — the dedupe guards above
                # already dropped known facts/unknown slugs, and both facts
                # and events stay editable (Codex / Canon Events tabs).
                chosen = list(payload)
            else:
                chosen = _pick_checked(
                    self, f'Canon from "{arc.get("name") or arc_id}"',
                    entries)
            n_facts = n_events = 0
            for key in chosen:
                if key.startswith('fact:'):
                    slug, fact = payload[key]
                    card = self.script.entities.get(slug)
                    if card is not None and fact not in card.get('facts', []):
                        card.setdefault('facts', []).append(fact)
                        n_facts += 1
                else:
                    summary, ents = payload[key]
                    if self.script.add_canon_event(summary, ents,
                                                   established_by=arc_id):
                        n_events += 1
            if n_facts or n_events:
                self.script.save_arc(arc_id, {
                    'canon_extracted_at': time.strftime('%Y-%m-%d %H:%M')})
            self._refresh_list_item(arc_id)
            self._update_story_status()
            self.chat_status.setText(
                ('Auto-canon' if auto else 'Canon') +
                f' recorded for "{arc.get("name") or arc_id}": '
                f'{n_facts} fact(s), {n_events} event(s).')
            if not auto or arc_id == self._current_arc_id:
                self._append_chat('assistant',
                                  f'[Canon: +{n_facts} facts, +{n_events} events]')

        def on_error(e):
            self._call_start_time = None
            self.chat_status.setText(f"Canon extraction error: {str(e)[:80]}")

        def run():
            try:
                data = self._main_ai._run_claude_json(system, prompt)
                self.ui_queue.put(lambda: on_done(data))
            except Exception as exc:
                self.ui_queue.put(lambda e=exc: on_error(str(e)))

        threading.Thread(target=run, daemon=True).start()

    def _cmd_distill_chat_to_arc(self):
        """Use AI to extract structured arc fields from the chat conversation."""
        if not self._current_arc_id:
            self.chat_status.setText("No arc selected.")
            return
        if not self._main_ai.ready:
            self.chat_status.setText("Claude CLI not found.")
            return
        if self._main_ai.busy:
            self.chat_status.setText("AI is busy...")
            return

        arc = self.script.arcs.get(self._current_arc_id, {})
        chat_history = arc.get('chat_history', [])
        if not chat_history:
            self.chat_status.setText("No chat history to distill.")
            return

        # Build the conversation text
        conv_lines = []
        for entry in chat_history:
            role = 'Author' if entry.get('role') == 'user' else 'Claude'
            conv_lines.append(f'{role}: {entry.get("content", "")}')
        conversation = '\n'.join(conv_lines)

        # Include any existing arc fields as context
        existing = []
        if arc.get('name'):
            existing.append(f'Current name: {arc["name"]}')
        if arc.get('premise'):
            existing.append(f'Current premise: {arc["premise"]}')
        if arc.get('themes'):
            existing.append(f'Current themes: {arc["themes"]}')
        if arc.get('motif'):
            existing.append(f'Current motif: {arc["motif"]}')

        layer_names = ', '.join(LAYER_ORDER)

        system = (
            "You are distilling a brainstorming conversation into structured story fields "
            "for a narrative audio installation.\n\n"
            "The story drives generation of a node graph where each node is 15-35 seconds of "
            "spoken audio. A story is an ordered STRUCTURE of beats; each beat names one of "
            "the 10 layer archetypes: " + layer_names + ".\n"
            "The classic shape is all 10 archetypes in order, but the structure is flexible: "
            "choose 4-12 beats, skip archetypes, repeat one (e.g. two consecutive 'discovery' "
            "beats for a slow reveal), or reorder when the conversation genuinely calls for "
            "it. Default to the classic 10 when in doubt.\n\n"
            "Extract the following from the conversation and return ONLY a JSON object:\n"
            "{\n"
            '  "name": "Short story title (2-5 words)",\n'
            '  "premise": "Several sentences, 600-1000 chars. See PREMISE RULES.",\n'
            '  "themes": "Comma-separated themes (e.g. isolation, transformation, memory)",\n'
            '  "motif": "One concrete recurring sensory/symbolic thread (a smell, a sound, an object). 60-150 chars.",\n'
            '  "notes": "Character details, world-building, tone guidance — anything useful for generation",\n'
            '  "cast": ["entity_slug", ...],\n'
            '  "structure": [ {"layer": "arrival", "direction": "..."}, {"layer": "presence", "direction": "..."}, ... ]\n'
            "}\n\n"
            "── CAST RULES ──\n"
            "Pick the cast from the AVAILABLE CODEX ENTITIES list in the prompt — exact\n"
            "slugs only, never invented. Choose the entities genuinely central to THIS\n"
            "story (typically 2-6). The cast controls exactly which entity cards are\n"
            "injected into every node-generation call: an empty cast means NO codex\n"
            "material reaches generation at all, so include every entity the story\n"
            "needs — and none it doesn't. If no listed entity fits, return [].\n\n"
            "── PREMISE RULES ──\n"
            "The premise is injected into EVERY node generation call with a layer-specific role label\n"
            "('establish this premise's world' at arrival, 'the bind IS this premise made specific' at\n"
            "complication, 'the action MUST BE this premise enacted' at turn, 'afterimage' at stillness).\n"
            "So the premise must read cleanly through ALL those role-frames AND must carry the durable\n"
            "character anchors, because `notes` only reaches the first seed call and is ignored afterward.\n"
            "  - Length: 600-1000 chars, several sentences.\n"
            "  - Must include: who the character is, the key durable facts about them, the specific\n"
            "    situation/action of this arc, and the irreducible stakes.\n"
            "  - Test before finalizing: can you literally say 'the bind is <PREMISE>' and 'the action\n"
            "    MUST BE <PREMISE> enacted' and have both make narrative sense? If not, sharpen.\n"
            "  - Capture the ESSENCE — the irreducible thing the arc is about — not a plot summary.\n"
            "  - Avoid abstract themes ('isolation', 'transformation'). Those go in `themes`.\n"
            "  - Worldbuilding details that already live in the story context don't need restating here;\n"
            "    the character's specific anchors do, since the context is shared and the premise is not.\n\n"
            "── BEAT RULES ──\n"
            "Each beat's `direction` is the steering text the generator sees as 'LAYER DIRECTION'. It must\n"
            "out-shout the dense story context, which means each beat carries THREE elements:\n"
            "  1. SCENE/SITUATION ANCHOR — a concrete moment or setup, not just a theme.\n"
            "       Good: 'she returns to the commissary after an austerity-rain stretch'\n"
            "       Bad:  'isolation deepens'\n"
            "  2. EMOTIONAL OR COGNITIVE MOVE — what specifically changes in this beat (knowledge,\n"
            "       position, recognition, resource). One transformation per beat.\n"
            "       Good: 'she notices the thermos is already on the counter and decides not to ask why'\n"
            "       Bad:  'something shifts'\n"
            "  3. BOUNDARY — what this beat is NOT, or a restraint on the AI.\n"
            "       Good: 'do not name the Toad here — only the absence she leaves'\n"
            "       Bad:  (no boundary at all — model will reach for the bible's most striking anchors)\n"
            "Length per beat: 100-300 chars, typically 2-3 sentences carrying the three elements above.\n"
            "Do NOT exceed ~400 chars — beats longer than that start competing with the bible and the\n"
            "model reads them as text-to-deliver rather than guidance.\n\n"
            "── OTHER RULES ──\n"
            "- Motif must be CONCRETE and SENSORY (a smell, a sound, an object), not thematic.\n"
            "- If the conversation did not cover a beat, write one that fits the story's trajectory.\n"
            "- Every structure entry's `layer` MUST be one of the 10 archetype names.\n"
            "- No markdown fences, no explanation — just the JSON."
        )

        prompt_parts = []
        if self.script.story_context_focused:
            prompt_parts.append(f'STORY CONTEXT (shared across all arcs — incorporate this setting/tone):\n{self.script.story_context_focused}')
        if existing:
            prompt_parts.append('EXISTING ARC FIELDS (refine these, don\'t ignore them):\n' + '\n'.join(existing))
        if self.script.entities:
            ent_lines = ['AVAILABLE CODEX ENTITIES (for the "cast" field — use these exact slugs):']
            for e_slug, e_card in self.script.entities.items():
                ent_lines.append(
                    f'  {e_slug} [{e_card.get("kind", "idea")}] '
                    f'{e_card.get("name") or e_slug}: '
                    f'{(e_card.get("essence") or "")[:80]}')
            prompt_parts.append('\n'.join(ent_lines))
        prompt_parts.append(f'CONVERSATION TO DISTILL:\n{conversation}')
        prompt = '\n\n'.join(prompt_parts)

        self._call_start_time = time.time()
        self._call_model = MODEL_OPUS
        self._tick_thinking_status()
        self._append_chat('assistant', '[Distilling conversation into arc fields...]')

        def on_done(data):
            elapsed = (time.time() - self._call_start_time) if self._call_start_time else 0.0
            self._call_start_time = None
            if not isinstance(data, dict):
                self.chat_status.setText(f"Distill failed: invalid response ({elapsed:.1f}s)")
                return

            # Fill in the arc fields
            if data.get('name'):
                self.name_edit.setText(data['name'])
            if data.get('premise'):
                self.premise_edit.setPlainText(data['premise'])
            if data.get('themes'):
                self.themes_edit.setText(data['themes'])
            if data.get('motif'):
                self.motif_edit.setText(data['motif'])
            if data.get('notes'):
                self.notes_edit.setPlainText(data['notes'])
            # Preferred: flexible structure array. Fallback: legacy beats
            # dict (older cached responses) mapped onto the classic shape.
            structure = data.get('structure')
            if isinstance(structure, list):
                clean = [{'layer': e.get('layer'),
                          'direction': str(e.get('direction') or '')}
                         for e in structure
                         if isinstance(e, dict) and e.get('layer') in LAYER_ORDER]
                if clean:
                    self._set_structure_rows(clean)
            elif isinstance(data.get('beats'), dict):
                beats = data['beats']
                self._set_structure_rows(
                    [{'layer': layer, 'direction': str(beats.get(layer) or '')}
                     for layer in LAYER_ORDER])
            # Cast: check the distilled entities in the picker (invalid /
            # invented slugs filtered — cast is opt-in, so this is what
            # wires the story's codex material)
            cast = data.get('cast')
            if isinstance(cast, list):
                valid = {ScriptData.sanitize_entity_slug(str(c)) for c in cast}
                valid &= set(self.script.entities.keys())
                self._reload_cast_list(valid)

            self._on_field_changed()  # mark dirty
            self.chat_status.setText(f"Story fields updated from chat. ({elapsed:.1f}s)")
            self._append_chat('assistant',
                f'Distilled: "{data.get("name", "")}" — {data.get("premise", "")[:100]}...')

        def on_error(e):
            self._call_start_time = None
            self.chat_status.setText(f"Distill error: {str(e)[:60]}")
            self._append_chat('assistant', f'[Distill error: {e}]')

        def run():
            try:
                # Force Opus for arc distillation — extracts structured
                # premise/themes/beats from chat and quality matters.
                data = self._main_ai._run_claude_json(system, prompt,
                                                      model_override=MODEL_OPUS)
                self.ui_queue.put(lambda: on_done(data))
            except Exception as exc:
                self.ui_queue.put(lambda e=exc: on_error(str(e)))

        threading.Thread(target=run, daemon=True).start()

    def _cmd_arc_chat(self):
        msg = self.chat_input.text().strip()
        if not msg:
            return
        if not self._arc_ai.ready:
            self.chat_status.setText("claude CLI not found")
            return
        if self._arc_ai.busy:
            self.chat_status.setText("AI is busy…")
            return
        self.chat_input.clear()
        self._append_chat('user', msg)
        arc_ctx = self._build_arc_context()
        self._call_start_time = time.time()
        self._call_model = self._main_ai.model
        self._tick_thinking_status()

        def on_reply(reply):
            elapsed = (time.time() - self._call_start_time) if self._call_start_time else 0.0
            self._call_start_time = None
            self._append_chat('assistant', reply)
            self.chat_status.setText(f'(last call: {elapsed:.1f}s)')
            if self._current_arc_id and self._current_arc_id in self.script.arcs:
                hist = self.script.arcs[self._current_arc_id].setdefault('chat_history', [])
                hist.append({'role': 'user',      'content': msg})
                hist.append({'role': 'assistant', 'content': reply})
                self.script.dirty = True

        def on_error(e):
            self._call_start_time = None
            self.chat_status.setText(f'Error: {e[:80]}')

        # Combine script-level story context with arc-specific context
        full_ctx = ''
        if self.script.story_context_focused:
            full_ctx = f'STORY CONTEXT:\n{self.script.story_context_focused}\n\n'
        full_ctx += arc_ctx
        # Cast codex cards so the chat knows the canon for the entities in play
        cast_block = self.script.cast_codex_for(self._current_arc_id) \
            if self._current_arc_id else ''
        if cast_block:
            full_ctx += f'\n\nCAST CODEX (canon entity cards for this story):\n{cast_block}'

        # The chat follows the AI Model menu selection; only the one-shot
        # "Distill Chat → Story" pass (which writes the card) pins Opus.
        self._arc_ai.chat(msg, self.ui_queue,
                          on_reply=on_reply, on_error=on_error,
                          story_context=full_ctx,
                          _system_override=SYSTEM_ARC_CHAT,
                          model_override=self._main_ai.model)

    def set_script(self, script: 'ScriptData'):
        """Re-point at a new ScriptData after the main window loads a file.
        (As a floating dialog this class was recreated per open; as an
        embedded tab it lives forever, so it must follow script swaps.)"""
        self.script = script
        self._current_arc_id = None
        self._clear_fields()
        self._refresh_arc_list()
        if self.script.arcs:
            first_id = next(iter(self.script.arcs))
            self._current_arc_id = first_id
            self._load_arc(first_id)
            self._refresh_arc_list()

    # Embedded guard: QDialog hides itself on accept/reject (incl. the
    # Escape key) — as a tab page that would blank the tab.
    def accept(self):
        if not self._embedded:
            super().accept()

    def reject(self):
        if not self._embedded:
            super().reject()

    def closeEvent(self, event):
        self._save_current()
        super().closeEvent(event)


class _SearchBorderItem(QGraphicsItem):
    """Glowing border drawn around a node to indicate a search match."""

    def __init__(self, parent_item):
        super().__init__(parent_item)
        self._rect = parent_item.boundingRect()
        self.setZValue(200)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)

    def boundingRect(self):
        return self._rect.adjusted(-4, -4, 4, 4)

    def paint(self, painter, option, widget=None):
        painter.save()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        r = self._rect.adjusted(2, 2, -2, -2)
        dash = [4.0, 4.0]   # 4px on, 4px off
        width = 16.0

        # White dashes
        p1 = QPen(QColor(0, 230, 255, 255), width)
        p1.setCapStyle(Qt.PenCapStyle.FlatCap)
        p1.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        p1.setStyle(Qt.PenStyle.CustomDashLine)
        p1.setDashPattern(dash)
        p1.setDashOffset(0)
        painter.setPen(p1)
        painter.drawRect(r)

        # Black dashes — offset by half a dash to fill the gaps
        p2 = QPen(QColor(255, 0, 220, 255), width)
        p2.setCapStyle(Qt.PenCapStyle.FlatCap)
        p2.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        p2.setStyle(Qt.PenStyle.CustomDashLine)
        p2.setDashPattern(dash)
        p2.setDashOffset(4.0)
        painter.setPen(p2)
        painter.drawRect(r)

        painter.restore()


class StoryContextChatDialog(QDialog):
    """Multi-turn chat with Claude to develop the Full Context.

    Owns its own conversation history (independent of the main AI chat panel).
    The user types messages; each Claude reply is shown in the log and held
    as the "latest draft." A button commits the latest reply back to the
    parent dialog's Full Context box.
    """

    def __init__(self, parent, ai: AIAssistant, ui_queue: queue.SimpleQueue,
                 initial_full: str = ''):
        super().__init__(parent)
        self.setWindowTitle("Develop Full Context with AI")
        self.setMinimumWidth(820)
        self.setMinimumHeight(560)

        self._ai = ai
        self._ui_queue = ui_queue
        self._history: list = []        # [{"role": "user"|"assistant", "content": str}, ...]
        self._latest_reply: str = ''    # most recent assistant message
        self.committed_text: Optional[str] = None  # set on either commit path
        self.committed_mode: str = 'replace'        # 'replace' or 'append'
        self._initial_full = (initial_full or '').strip()
        self._call_start_time: Optional[float] = None  # set when an AI call is in-flight

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        info = QLabel(
            "Chat with Claude to draft, expand, refine, or discuss the story "
            "context. Two commit paths: <b>Append</b> adds the reply to the "
            "end of the existing context (non-destructive — use when you "
            "asked Claude to add a section, character, etc). <b>Replace</b> "
            "overwrites the whole context (use for full rewrites). The current "
            "context is sent on your first message so Claude can build on it."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        layout.addWidget(info)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self._log.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self._log.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self._log, stretch=1)

        self._input = QTextEdit()
        self._input.setAcceptRichText(False)
        self._input.setPlaceholderText(
            "What would you like to do? "
            "(e.g. \"Draft a bible for a story about two siblings reuniting after years apart\", "
            "\"Make the tone more bittersweet\", \"Expand the section on the protagonist\", "
            "\"Cut the lighthouse subplot\".  Enter to send, Shift+Enter for newline.)"
        )
        self._input.setMinimumHeight(70)
        self._input.setMaximumHeight(120)
        self._input.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self._input.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self._input.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._input.installEventFilter(self)
        layout.addWidget(self._input)

        btn_row = QHBoxLayout()
        self._status = QLabel("")
        self._status.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        btn_row.addWidget(self._status, stretch=1)

        self._send_btn = QPushButton("Send")
        self._send_btn.clicked.connect(self._on_send)
        btn_row.addWidget(self._send_btn)

        self._append_btn = QPushButton("Append Latest Reply")
        self._append_btn.setToolTip(
            "Add Claude's latest reply to the END of the existing context.\n"
            "Use when Claude wrote an addition (a new section, character, etc.)\n"
            "rather than a full rewrite — your existing bible is preserved.")
        self._append_btn.setEnabled(False)
        self._append_btn.clicked.connect(self._on_append)
        btn_row.addWidget(self._append_btn)

        self._commit_btn = QPushButton("Replace Context with Reply")
        self._commit_btn.setToolTip(
            "REPLACE the entire story context with Claude's latest reply.\n"
            "Use when Claude wrote a full rewrite of the whole bible.\n"
            "Destructive — your prior context is overwritten.")
        self._commit_btn.setEnabled(False)
        self._commit_btn.clicked.connect(self._on_commit)
        btn_row.addWidget(self._commit_btn)

        self._reset_btn = QPushButton("Reset Chat")
        self._reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(self._reset_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        # Pump for ui_queue callbacks while the modal dialog is up.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._drain_ui_queue)
        self._timer.start(50)

        if self._initial_full:
            self._append_system(
                f"(Existing Full Context will be included with your first message — "
                f"{len(self._initial_full)} chars.)"
            )

    # ── UI helpers ──────────────────────────────────────────────────────────

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        if obj is self._input and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                    self._on_send()
                    return True
        return super().eventFilter(obj, event)

    def _append_system(self, text: str):
        cursor = self._log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(
            f'<div style="color:#888888;font-style:italic;margin:4px 0;">{text}</div><br>'
        )
        self._log.moveCursor(QTextCursor.MoveOperation.End)

    def _append_message(self, role: str, text: str):
        cursor = self._log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if role == "user":
            color = "#88bbff"
            who = "You"
        else:
            color = "#aaee99"
            who = "Claude"
        # Escape minimally for HTML safety on user input
        safe = (text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace("\n", "<br>"))
        cursor.insertHtml(
            f'<div style="margin:4px 0;"><b style="color:{color};">{who}:</b> '
            f'<span style="color:#dddddd;">{safe}</span></div><br>'
        )
        self._log.moveCursor(QTextCursor.MoveOperation.End)

    # ── Actions ─────────────────────────────────────────────────────────────

    def _on_send(self):
        msg = self._input.toPlainText().strip()
        if not msg:
            return
        if self._ai.busy:
            self._status.setText("AI is busy...")
            return

        # On the very first send, prepend the existing full context (if any)
        # so Claude has something to build from.
        if not self._history and self._initial_full:
            msg_to_send = (
                f"Here is the current Full Context I'm working with:\n\n"
                f"{self._initial_full}\n\n"
                f"My request: {msg}"
            )
        else:
            msg_to_send = msg

        self._append_message("user", msg)
        self._input.clear()
        self._send_btn.setEnabled(False)
        # status will tick via _drain_ui_queue once _call_start_time is set

        def on_done(reply: str):
            elapsed = (time.time() - self._call_start_time) if self._call_start_time else 0.0
            self._call_start_time = None
            # Update internal history with the *original* user message (not
            # the wrapped one), so subsequent transcripts stay clean.
            self._history.append({"role": "user", "content": msg_to_send})
            self._history.append({"role": "assistant", "content": reply})
            self._latest_reply = reply
            self._append_message("assistant", reply)
            self._append_system(
                f"Latest reply: {len(reply)} chars in {elapsed:.1f}s. "
                f"Use <b>Append</b> if Claude wrote an addition, or "
                f"<b>Replace</b> if it's a full rewrite."
            )
            self._send_btn.setEnabled(True)
            self._commit_btn.setEnabled(True)
            self._append_btn.setEnabled(True)
            self._status.setText(f"Ready. (last call: {elapsed:.1f}s)")

        def on_error(err: str):
            self._call_start_time = None
            self._send_btn.setEnabled(True)
            self._status.setText(f"Error: {err[:80]}")
            self._append_system(f"Error: {err[:200]}")

        # The chat follows the AI Model menu selection (pick Opus in the
        # Story ▸ AI Model menu when drafting the bible warrants it).
        self._call_start_time = time.time()
        self._update_thinking_status()
        self._ai.chat_context(self._history, msg_to_send,
                              self._ui_queue, on_done, on_error,
                              model_override=self._ai.model)

    def _on_commit(self):
        """REPLACE: parent overwrites context with the latest reply."""
        if not self._latest_reply:
            return
        self.committed_text = self._latest_reply
        self.committed_mode = 'replace'
        self.accept()

    def _on_append(self):
        """APPEND: parent adds the latest reply to the end of the current context."""
        if not self._latest_reply:
            return
        self.committed_text = self._latest_reply
        self.committed_mode = 'append'
        self.accept()

    def _on_reset(self):
        self._history.clear()
        self._latest_reply = ''
        self._log.clear()
        self._commit_btn.setEnabled(False)
        self._append_btn.setEnabled(False)
        self._status.setText("Chat reset.")
        if self._initial_full:
            self._append_system(
                f"(Existing Full Context will be included with your first message — "
                f"{len(self._initial_full)} chars.)"
            )

    def _drain_ui_queue(self):
        """Run any pending UI callbacks posted by the AI worker thread,
        and update the elapsed-time status display while a call is in flight."""
        try:
            while True:
                cb = self._ui_queue.get_nowait()
                try:
                    cb()
                except Exception as e:
                    print(f"[StoryContextChatDialog] callback error: {e}")
        except queue.Empty:
            pass
        self._update_thinking_status()

    def _update_thinking_status(self):
        if self._call_start_time is None:
            return
        elapsed = time.time() - self._call_start_time
        self._status.setText(
            f"Claude ({_model_short_name(self._ai.model)}) is thinking… "
            f"{elapsed:0.0f}s"
        )


class MainWindow(QMainWindow):
    def __init__(self, script_path=None):
        super().__init__()
        self.script = ScriptData()
        self.ai     = AIAssistant()
        self.vm     = VoiceManager()
        self.ui_queue = queue.SimpleQueue()

        self._selected_node_id: Optional[str] = None
        self._node_items: Dict[str, NarrativeNode] = {}
        self._pending_connect_from: Optional[str] = None
        self._pending_connect_line = None   # QGraphicsLineItem rubber-band
        self._cycle_nodes: set = set()
        self._search_overlays: dict = {}   # node_id → _CrosshatchItem for active search matches
        self._orchestrators: list = []     # active ParallelNodeOrchestrator instances
        self._job_counter: int = 0
        self._recent_files: list = self._load_recent_files()
        self._recent_menu = None

        self._build_ui()
        self._build_menu()

        # Block Ctrl+Z/Y at the application level so NodeGraphQt's internal
        # undo/redo never fires.  Must be installed AFTER the graph is created
        # because NodeGraphQt registers its own shortcuts during widget init.
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QObject, QEvent, Qt as _Qt

        class _BlockUndoKeys(QObject):
            def eventFilter(self_, obj, event):
                if event.type() in (QEvent.Type.KeyPress,
                                    QEvent.Type.ShortcutOverride):
                    key  = event.key()
                    ctrl = bool(event.modifiers() & _Qt.KeyboardModifier.ControlModifier)
                    if ctrl and key in (_Qt.Key.Key_Z, _Qt.Key.Key_Y):
                        event.accept()  # claim it so QAction shortcuts don't fire
                        return True     # also swallow the key event
                return False

        self._block_undo = _BlockUndoKeys(parent=self)
        QApplication.instance().installEventFilter(self._block_undo)

        # Connect NodeGraph signals
        self.graph.node_selected.connect(self._on_node_selected)
        self.graph.port_connected.connect(self._on_port_connected)
        self.graph.port_disconnected.connect(self._on_port_disconnected)
        self.graph.nodes_deleted.connect(self._on_nodes_deleted)
        self.graph.scene().selectionChanged.connect(self._on_selection_changed)

        # Autosave every 60 seconds if dirty and file has a save path
        self._autosave_timer = QTimer(self)
        self._autosave_timer.timeout.connect(self._autosave)
        self._autosave_timer.start(60_000)

        # Set contexts
        self.props_panel.set_context(self.script, self.vm, self.ai, self.ui_queue)
        self.props_panel.refresh_variable_widgets()
        self.voice_panel.set_context(self.script, self.vm, self.ui_queue, self.props_panel)
        self.chat_panel.set_context(self.script, self.ai, self.ui_queue,
                                    self._on_graph_generated,
                                    on_nodes_incremental=self._add_nodes_incremental,
                                    register_orchestrator=self._register_orchestrator)
        self.play_panel.set_context(self.script, self.ui_queue)

        # Connect props signals
        self.props_panel.node_modified.connect(self._on_node_modified)

        # QTimer to drain ui_queue
        self._drain_timer = QTimer(self)
        self._drain_timer.setInterval(50)
        self._drain_timer.timeout.connect(self._drain_queue)
        self._drain_timer.start()

        if script_path:
            self._load_script(Path(script_path))

        self._update_title()
        self.resize(1400, 800)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # NodeGraphQt graph. Narrative graphs are legitimately CYCLIC
        # (loops, junction drift, callbacks), so acyclic enforcement must
        # be off: NodeGraphQt's acyclic_check has no visited-set — on a
        # graph that already contains a cycle its BFS re-queues the cycle
        # FOREVER, freezing the main thread on the next connect_to (this
        # hung Weave Junctions' apply step; py-spy showed acyclic_check
        # pinned). It also silently refuses cycle-closing edges, dropping
        # woven links from the view.
        self.graph = NodeGraph()
        self.graph.set_acyclic(False)
        self.graph.register_node(NarrativeNode)
        graph_widget = self.graph.widget

        self._graph_hover_filter = _GraphViewHoverFilter(
            self._get_node_at_view_pos,
            self._on_node_hover_enter,
            self._on_node_hover_leave,
            on_right_click=self._on_graph_right_click,
            on_delete_pipes=self._cmd_delete_selected_pipes,
            on_mouse_move=self._on_graph_mouse_move,
            on_deselect=self.graph.clear_selection,
            get_selected=lambda: self._selected_node_id,
            on_restore_selection=self._select_node,
            on_marquee_select=self._on_marquee_select,
            on_shift_click=self._on_shift_click,
            viewer=self.graph.viewer(),
        )
        # Events land on the viewport, not the view itself
        self.graph.viewer().viewport().installEventFilter(self._graph_hover_filter)
        # Key events (Delete / Backspace) land on the viewer itself
        self.graph.viewer().installEventFilter(self._graph_hover_filter)

        # Right panel (scrollable)
        right_widget = QWidget()
        right_widget.setMinimumWidth(310)
        right_widget.setMaximumWidth(380)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(6)

        self.voice_panel = VoiceSettingsPanel()

        self.props_panel = PropertiesPanel()
        right_layout.addWidget(self.props_panel)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("color: #555;")
        right_layout.addWidget(sep2)

        self.chat_panel = AIChatPanel()
        right_layout.addWidget(self.chat_panel)

        sep3 = QFrame()
        sep3.setFrameShape(QFrame.HLine)
        sep3.setStyleSheet("color: #555;")
        right_layout.addWidget(sep3)

        self.play_panel = PlaybackPanel()
        right_layout.addWidget(self.play_panel)

        right_layout.addStretch(1)

        scroll_area = QScrollArea()
        scroll_area.setWidget(right_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setMinimumWidth(320)
        scroll_area.setMaximumWidth(400)

        # Search bar + frequency toggle above graph
        self._search_bar = QLineEdit()
        self._search_bar.setPlaceholderText("Search nodes by text, tags, arc beat, ID…  (Ctrl+/)")
        self._search_bar.setClearButtonEnabled(True)
        self._search_bar.textChanged.connect(self._cmd_search)
        self._search_bar.setStyleSheet("padding: 2px 4px; font-size: 11px;")
        from PySide6.QtGui import QShortcut, QKeySequence
        QShortcut(QKeySequence("Ctrl+/"), self, activated=self._search_bar.setFocus)
        QShortcut(QKeySequence("Escape"), self, activated=self._cancel_pending_connect)

        # Text-only search scope: when checked, the search matches ONLY
        # each node's text body — not the ID, label, tags, or arc beat.
        self._text_only_btn = QPushButton("Text")
        self._text_only_btn.setCheckable(True)
        self._text_only_btn.setFixedWidth(46)
        self._text_only_btn.setFixedHeight(24)
        self._text_only_btn.setToolTip(
            "Search node text only.\n"
            "When on, matches ignore node IDs, labels, tags, and arc beats.")
        self._text_only_btn.setStyleSheet(
            "QPushButton { background: #2a2a3a; border: 1px solid #555; border-radius: 3px; font-size: 11px; }"
            "QPushButton:checked { background: #1a4a3a; border: 1px solid #20c080; color: #50e0a0; }"
        )
        self._text_only_btn.toggled.connect(self._cmd_search_scope_toggled)

        # Whole-word matching: 'rain' stops matching 'train' / 'raining'.
        self._word_btn = QPushButton("Word")
        self._word_btn.setCheckable(True)
        self._word_btn.setFixedWidth(46)
        self._word_btn.setFixedHeight(24)
        self._word_btn.setToolTip(
            "Match whole words only.\n"
            "When on, 'rain' no longer matches 'train' or 'raining'.")
        self._word_btn.setStyleSheet(
            "QPushButton { background: #2a2a3a; border: 1px solid #555; border-radius: 3px; font-size: 11px; }"
            "QPushButton:checked { background: #1a3a5a; border: 1px solid #2080e0; color: #50a0f0; }"
        )
        self._word_btn.toggled.connect(self._cmd_search_word_toggled)

        self._freq_btn = QPushButton("Freq")
        self._freq_btn.setCheckable(True)
        self._freq_btn.setFixedWidth(46)
        self._freq_btn.setFixedHeight(24)
        self._freq_btn.setToolTip("Toggle frequency heat map (Ctrl+Shift+A)")
        self._freq_btn.setStyleSheet(
            "QPushButton { background: #2a2a3a; border: 1px solid #555; border-radius: 3px; font-size: 11px; }"
            "QPushButton:checked { background: #5a3a1a; border: 1px solid #e08020; color: #ffaa40; }"
        )
        self._freq_btn.toggled.connect(self._cmd_frequency_analysis)
        self._freq_counts: dict = {}   # cached simulation results

        # Bulk-apply a trigger_state to every node currently highlighted
        # by the search. Disabled when there are no matches or no
        # associated weather set.
        self._apply_trigger_btn = QPushButton("Trigger…")
        self._apply_trigger_btn.setFixedHeight(24)
        self._apply_trigger_btn.setToolTip(
            "Set trigger_state on all nodes matched by the current search.\n"
            "Disabled until the search bar has matches and the script has\n"
            "an associated weather set.")
        self._apply_trigger_btn.setStyleSheet(
            "QPushButton { background: #2a2a3a; border: 1px solid #555; border-radius: 3px; font-size: 11px; padding: 0 8px; }"
            "QPushButton:disabled { color: #666; border-color: #333; }"
        )
        self._apply_trigger_btn.clicked.connect(self._cmd_bulk_set_trigger_state)
        self._apply_trigger_btn.setEnabled(False)

        search_row = QWidget()
        search_row.setMaximumHeight(30)
        search_row_layout = QHBoxLayout(search_row)
        search_row_layout.setContentsMargins(2, 2, 2, 2)
        search_row_layout.setSpacing(4)
        search_row_layout.addWidget(self._search_bar)
        search_row_layout.addWidget(self._text_only_btn)
        search_row_layout.addWidget(self._word_btn)
        search_row_layout.addWidget(self._apply_trigger_btn)
        search_row_layout.addWidget(self._freq_btn)

        graph_container = QWidget()
        graph_vbox = QVBoxLayout(graph_container)
        graph_vbox.setContentsMargins(0, 0, 0, 0)
        graph_vbox.setSpacing(0)
        graph_vbox.addWidget(search_row)
        graph_vbox.addWidget(graph_widget)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(graph_container)
        splitter.addWidget(scroll_area)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        # ── Main tabs: Graph | Codex | Stories | Web Planner | Canon ─────
        # The codex / stories / web sections used to be modeless floating
        # dialogs opened from the Story menu — hard to move between. They
        # are now persistent tab pages beside the graph; the old menu
        # entries (and their shortcuts) jump to the matching tab.
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(splitter, "Graph")

        self.codex_panel = CodexDialog(self, self.script, self.ai,
                                       self.ui_queue, embedded=True)
        self.tabs.addTab(self.codex_panel, "Codex")

        self.stories_panel = ArcEditorDialog(
            self.script, self.ai, self.ui_queue,
            on_graph_generated=self._on_graph_generated,
            register_orchestrator=self._register_orchestrator,
            embedded=True)
        self.tabs.addTab(self.stories_panel, "Stories")

        self.web_panel = WebPlannerDialog(self, self.script, self.ai,
                                          self.ui_queue, embedded=True)
        self.tabs.addTab(self.web_panel, "Web Planner")

        self.canon_panel = CanonEventsPanel(self, self.script)
        self.tabs.addTab(self.canon_panel, "Canon Events")

        self.tabs.currentChanged.connect(self._on_tab_changed)
        # Back-compat aliases (older commands/tests reach these names)
        self._codex_dlg = self.codex_panel
        self._arc_dlg = self.stories_panel
        self._web_dlg = self.web_panel

        main_layout.addWidget(self.tabs)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        self._stat_nodes = QLabel("")
        self._stat_nodes.setStyleSheet("color: #aaaaaa; padding-right: 16px;")
        self._stat_duration = QLabel("")
        self._stat_duration.setStyleSheet("color: #aaaaaa; padding-right: 8px;")
        self.status_bar.addPermanentWidget(self._stat_nodes)
        self.status_bar.addPermanentWidget(self._stat_duration)

    def _build_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        act_new = QAction("New", self)
        act_new.setShortcut("Ctrl+N")
        act_new.triggered.connect(self._cmd_new)
        file_menu.addAction(act_new)

        act_open = QAction("Open...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._cmd_open)
        file_menu.addAction(act_open)

        self._recent_menu = file_menu.addMenu("Open Recent")
        self._rebuild_recent_menu()

        act_save = QAction("Save", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self._cmd_save)
        file_menu.addAction(act_save)

        act_save_as = QAction("Save As...", self)
        act_save_as.setShortcut("Ctrl+Shift+S")
        act_save_as.triggered.connect(self._cmd_save_as)
        file_menu.addAction(act_save_as)

        # Edit menu
        edit_menu = menubar.addMenu("Edit")

        act_add = QAction("Add Node", self)
        act_add.setShortcut("Ctrl+A")
        act_add.triggered.connect(self._cmd_add_node)
        edit_menu.addAction(act_add)

        act_fit = QAction("Fit View", self)
        act_fit.setShortcut("Ctrl+F")
        act_fit.triggered.connect(self._cmd_fit_view)
        edit_menu.addAction(act_fit)

        layout_menu = edit_menu.addMenu("Layout")
        for label, fn in [
            ("Horizontal  (left → right)", _compute_layout),
            ("Vertical  (top → bottom)",   _layout_vertical),
            ("Tree  (branching)",           _layout_tree),
        ]:
            act = QAction(label, self)
            act.triggered.connect(lambda *_, f=fn: self._cmd_apply_layout(f))
            layout_menu.addAction(act)

        edit_menu.addSeparator()

        act_spread = QAction("Spread", self)
        act_spread.triggered.connect(self._cmd_spread)
        edit_menu.addAction(act_spread)

        act_compact = QAction("Compact", self)
        act_compact.triggered.connect(self._cmd_compact)
        edit_menu.addAction(act_compact)

        # Story menu
        story_menu = menubar.addMenu("Story")
        act_codex = QAction("Codex Tab", self)
        act_codex.setShortcut("Ctrl+Shift+E")
        act_codex.setToolTip(
            "Entity cards for recurring concepts — characters, locations, "
            "events, themes, ideas, objects. Slugs double as node tags.")
        act_codex.triggered.connect(self._cmd_open_codex)
        story_menu.addAction(act_codex)

        act_arcs = QAction("Stories Tab", self)
        act_arcs.setShortcut("Ctrl+Shift+R")
        act_arcs.setToolTip(
            "Story cards: premise, cast (codex entities), and a flexible "
            "beat structure — v2's generalization of arcs.")
        act_arcs.triggered.connect(self._cmd_open_arc_editor)
        story_menu.addAction(act_arcs)

        act_web = QAction("Web Planner Tab", self)
        act_web.setShortcut("Ctrl+Shift+W")
        act_web.setToolTip(
            "The story web: typed relations between stories, AI web "
            "proposals, junction weaving (cross-story drift edges), and "
            "callback weaving.")
        act_web.triggered.connect(self._cmd_open_web_planner)
        story_menu.addAction(act_web)

        act_canon = QAction("Canon Events Tab", self)
        act_canon.setShortcut("Ctrl+Shift+N")
        act_canon.setToolTip(
            "The cross-story ledger: events extracted from generated "
            "stories, injected into casting stories' prompts. Promote an "
            "event to a codex entity to make it castable.")
        act_canon.triggered.connect(self._cmd_open_canon_events)
        story_menu.addAction(act_canon)

        act_ctx = QAction("Story Context…", self)
        act_ctx.setShortcut("Ctrl+Shift+C")
        act_ctx.triggered.connect(self._cmd_open_story_context)
        story_menu.addAction(act_ctx)

        act_vars = QAction("Story Variables…", self)
        act_vars.setShortcut("Ctrl+Shift+B")
        act_vars.triggered.connect(self._cmd_open_story_variables)
        story_menu.addAction(act_vars)

        act_det_vars = QAction("Determine Variables", self)
        act_det_vars.setShortcut("Ctrl+Shift+D")
        act_det_vars.triggered.connect(self._cmd_determine_variables)
        story_menu.addAction(act_det_vars)

        act_weather = QAction("Weather Set…", self)
        act_weather.setToolTip(
            "Pin this script to a specific weather set so per-node "
            "trigger_state dropdowns show that set's states. By default "
            "the editor auto-detects via reverse lookup.")
        act_weather.triggered.connect(self._cmd_open_weather_set_dialog)
        story_menu.addAction(act_weather)

        # Node Length submenu — same style as AI Model. Selecting a preset
        # writes that range to the script's node_word_range field; selecting
        # "Editor default" clears the field so the 40-100 baseline applies.
        # Checkmarks are refreshed when the menu opens so they reflect the
        # currently-loaded script.
        node_len_menu = story_menu.addMenu("Node Length")
        node_len_menu.setToolTip(
            "Set the default per-node word range for this script. Each "
            "arc can override the script default via the Arc editor.")
        self._node_length_actions = {}   # key: (lo, hi) tuple or None
        # First item: editor default (clears the field)
        act_default = QAction(
            "Editor default (40-100 words, 15-35 sec, vignette)",
            self, checkable=True)
        act_default.setData(None)
        act_default.triggered.connect(
            lambda checked: self._set_script_node_length(None))
        node_len_menu.addAction(act_default)
        self._node_length_actions[None] = act_default
        node_len_menu.addSeparator()
        # Preset items
        for label, lo, hi in NODE_LENGTH_PRESETS:
            act = QAction(label, self, checkable=True)
            act.setData((lo, hi))
            act.triggered.connect(
                lambda checked, p_lo=lo, p_hi=hi:
                    self._set_script_node_length((p_lo, p_hi)))
            node_len_menu.addAction(act)
            self._node_length_actions[(lo, hi)] = act
        node_len_menu.aboutToShow.connect(self._refresh_node_length_checks)
        # Make sure the right item is checked at startup too
        self._refresh_node_length_checks()

        # Generation width — Small / Medium / Large. Scales children-per-parent
        # and per-layer caps in arc-driven generation.
        width_menu = story_menu.addMenu("Generation Width")
        width_menu.setToolTip(
            "Scale how many child nodes are generated at each beat when "
            "expanding a story arc. Small = tight & focused; Large = sprawling.")
        self._width_actions = {}
        for preset in ('small', 'medium', 'large'):
            label = preset.capitalize()
            if preset == 'medium':
                label += '  (default)'
            act = QAction(label, self, checkable=True)
            act.setData(preset)
            act.triggered.connect(
                lambda checked, p=preset: self._set_script_width_preset(p))
            width_menu.addAction(act)
            self._width_actions[preset] = act
        width_menu.aboutToShow.connect(self._refresh_width_checks)
        self._refresh_width_checks()

        act_audit = QAction("Audit Audio Files…", self)
        act_audit.triggered.connect(self._cmd_audit_audio)
        story_menu.addAction(act_audit)

        story_menu.addSeparator()
        model_menu = story_menu.addMenu("AI Model")
        self._model_actions = {}
        for model_id, short in [
            (MODEL_SONNET, 'Sonnet 5'),
            (MODEL_OPUS,   'Opus 4.8'),
        ]:
            act = QAction(short, self, checkable=True)
            act.setData(model_id)
            act.triggered.connect(lambda checked, m=model_id: self._set_ai_model(m))
            model_menu.addAction(act)
            self._model_actions[model_id] = act
        # Check the default
        default_act = self._model_actions.get(self.ai.model)
        if default_act:
            default_act.setChecked(True)

        thinking_menu = story_menu.addMenu("Thinking")
        self._thinking_actions = {}
        thinking_labels = [
            ('off',        'Off'),
            ('think',      'Think'),
            ('think_hard', 'Think Hard'),
            ('ultrathink', 'Ultrathink'),
        ]
        for level, label in thinking_labels:
            act = QAction(label, self, checkable=True)
            act.setData(level)
            act.triggered.connect(lambda checked, lv=level: self._set_ai_thinking(lv))
            thinking_menu.addAction(act)
            self._thinking_actions[level] = act
        default_think = self._thinking_actions.get(self.ai.thinking)
        if default_think:
            default_think.setChecked(True)

        story_menu.addSeparator()
        act_stop = QAction("Stop AI Generation", self)
        act_stop.setShortcut("Ctrl+Shift+G")
        act_stop.setToolTip(
            "Cancel every running generation job (full runs, expands, "
            "continues). In-flight AI calls finish and are discarded.")
        act_stop.triggered.connect(self._cmd_stop_generation)
        story_menu.addAction(act_stop)

        # Analysis menu
        analysis_menu = menubar.addMenu("Analysis")
        act_freq = QAction("Toggle Frequency Heat Map", self)
        act_freq.setShortcut("Ctrl+Shift+A")
        act_freq.triggered.connect(lambda: self._freq_btn.setChecked(not self._freq_btn.isChecked()))
        analysis_menu.addAction(act_freq)

        act_webstats = QAction("Web Statistics…", self)
        act_webstats.setToolTip(
            "Simulate 2000 listener walks and report per-story occupancy, "
            "drift between stories, and which junction edges carry it.")
        act_webstats.triggered.connect(self._cmd_web_statistics)
        analysis_menu.addAction(act_webstats)

        act_audit = QAction("Consistency Audit (AI)…", self)
        act_audit.setToolTip(
            "Check every story's node texts against its canon (entity cards "
            "+ canon events). Read-only; reports issues to review.")
        act_audit.triggered.connect(self._cmd_consistency_audit)
        analysis_menu.addAction(act_audit)

        act_seams = QAction("Seam Audit (AI)…", self)
        act_seams.setToolTip(
            "Listen-check edge transitions: every cross-story seam plus a "
            "sample of in-story seams. Flags non-sequiturs and register "
            "jumps; fix flagged children with AI Rewrite.")
        act_seams.triggered.connect(self._cmd_seam_audit)
        analysis_menu.addAction(act_seams)

        # Voice menu
        voice_menu = menubar.addMenu("Voice")
        act_voice = QAction("Voice Settings…", self)
        act_voice.setShortcut("Ctrl+Shift+V")
        act_voice.triggered.connect(self._cmd_open_voice_settings)
        voice_menu.addAction(act_voice)

    def _on_tab_changed(self, idx: int):
        """Refresh the tab the user just switched to — the graph and the
        section panels edit the same script, so counts/lists go stale.

        DEFERRED: refreshing synchronously inside the currentChanged signal
        rebuilt lists/scenes Qt was still dispatching events on — native
        access-violation crashes on this machine (see crash.log)."""
        w = self.tabs.widget(idx)
        if w is self.codex_panel:
            QTimer.singleShot(0, self.codex_panel._refresh_list)
        elif w is self.stories_panel:
            QTimer.singleShot(0, self.stories_panel._refresh_arc_list)
        elif w is self.web_panel:
            QTimer.singleShot(0, self.web_panel._refresh)
        elif w is self.canon_panel:
            QTimer.singleShot(0, self.canon_panel._refresh)
        else:
            self._update_title()   # story renames etc. reflect in title

    def _cmd_open_codex(self):
        # Refresh explicitly too — setCurrentWidget doesn't fire
        # currentChanged when the tab is already current.
        self.tabs.setCurrentWidget(self.codex_panel)
        self.codex_panel._refresh_list()

    def _cmd_open_web_planner(self):
        self.tabs.setCurrentWidget(self.web_panel)
        self.web_panel._refresh()

    def _cmd_open_canon_events(self):
        self.tabs.setCurrentWidget(self.canon_panel)
        self.canon_panel._refresh()

    def _cmd_open_arc_editor(self):
        self.tabs.setCurrentWidget(self.stories_panel)
        self.stories_panel._refresh_arc_list()

    def _set_script_width_preset(self, preset: str):
        """Apply the chosen Small/Medium/Large generation width to the
        active script. Affects future arc-driven generation only."""
        self.script.set_width_preset(preset)
        self._refresh_width_checks()
        self.status_bar.showMessage(
            f"Generation width: {preset} (affects future arc generation)", 3000)
        self._update_title()

    def _refresh_width_checks(self):
        current = self.script.width_preset
        for preset, act in getattr(self, '_width_actions', {}).items():
            try:
                act.setChecked(preset == current)
            except RuntimeError:
                # Widget destroyed (project reload, etc.)
                pass

    def _set_script_node_length(self, rng):
        """Apply a script-wide node-length selection. ``rng`` is None
        (editor default — clears the field) or a (lo, hi) tuple."""
        if rng is None:
            self.script.set_node_word_range(None, None)
        else:
            lo, hi = rng
            self.script.set_node_word_range(int(lo), int(hi))
        self._refresh_node_length_checks()
        # Status feedback like the AI Model menu does
        if rng is None:
            self.status_bar.showMessage("Node length: editor default (40-100 words)", 3000)
        else:
            self.status_bar.showMessage(
                f"Node length: {rng[0]}-{rng[1]} words", 3000)

    def _refresh_node_length_checks(self):
        """Update the Node Length submenu checkmarks to reflect the
        script's current node_word_range. Called on aboutToShow so the
        UI stays in sync if the script changed between menu opens, and
        on every set so the new selection sticks visually."""
        if not hasattr(self, '_node_length_actions') or not self._node_length_actions:
            return
        current = self.script.node_word_range if self.script else None
        # Normalize to a hashable key
        current_key = tuple(current) if current else None

        # Drop any stale "Custom" action from a previous load
        for key in list(self._node_length_actions):
            act = self._node_length_actions[key]
            if key is not None and key not in {
                (lo, hi) for (_lbl, lo, hi) in NODE_LENGTH_PRESETS
            }:
                # Stale custom item — remove
                act.parent().removeAction(act) if act.parent() else None
                act.deleteLater()
                del self._node_length_actions[key]

        # If the loaded script has a custom value not in any preset, add
        # a checkable "Custom (lo-hi)" item so the user sees it.
        if current_key is not None and current_key not in self._node_length_actions:
            menu = None
            # Reuse the parent menu from any existing action
            for act in self._node_length_actions.values():
                if act.parent() and hasattr(act.parent(), 'addAction'):
                    menu = act.parent()
                    break
            if menu is not None:
                custom = QAction(
                    f"Custom ({current_key[0]}-{current_key[1]} words)",
                    self, checkable=True)
                custom.setData(current_key)
                custom.triggered.connect(
                    lambda checked, r=current_key: self._set_script_node_length(r))
                menu.addAction(custom)
                self._node_length_actions[current_key] = custom

        # Set checkmarks
        for key, act in self._node_length_actions.items():
            act.setChecked(key == current_key)

    def _cmd_open_weather_set_dialog(self):
        """Pin (or unpin) the script's associated weather set. Affects
        which states appear in per-node trigger_state dropdowns.
        Default is auto-detect via reverse lookup on
        narrative_script field in WEATHER_SETS."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Weather Set")
        dlg.setMinimumWidth(440)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        all_sets = self.script.available_weather_sets()
        if not all_sets:
            layout.addWidget(QLabel(
                "No weather_params.py found in the project containing this "
                "script. Save the script inside a project directory first."))
            close = QPushButton("Close")
            close.clicked.connect(dlg.accept)
            layout.addWidget(close)
            dlg.exec()
            return

        # Detect what reverse-lookup would pick — useful info for the user
        wsets = _load_project_weather_sets(self.script.path)
        auto_pick = _reverse_lookup_weather_set(self.script.path, wsets)
        explicit = self.script.weather_set_explicit

        intro = QLabel(
            "Pin this script to a weather set so per-node trigger_state\n"
            "dropdowns show that set's states. Leave on auto-detect to\n"
            "use the set whose <code>narrative_script</code> field references this script."
        )
        intro.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        intro.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(intro)

        # Status line
        status_lines = []
        status_lines.append(
            f"<b>Currently associated set:</b> "
            f"<span style='color:#aaffaa'>{self.script.associated_weather_set or '(none)'}</span>"
        )
        if auto_pick:
            status_lines.append(
                f"<b>Auto-detect would pick:</b> "
                f"<span style='color:#aaccff'>{auto_pick}</span>")
        else:
            status_lines.append(
                f"<b>Auto-detect would pick:</b> "
                f"<span style='color:#cc8888'>(none — no weather set "
                f"references this script)</span>")
        status_lines.append(
            f"<b>Explicit pin:</b> "
            f"<span style='color:#ffcc88'>{explicit or '(none)'}</span>")
        status = QLabel("<br>".join(status_lines))
        status.setTextFormat(Qt.TextFormat.RichText)
        status.setStyleSheet("font-size: 11px; padding: 6px;")
        layout.addWidget(status)

        # Dropdown
        row = QHBoxLayout()
        row.addWidget(QLabel("Pin to set:"))
        combo = QComboBox()
        combo.addItem("(auto-detect)", "")
        for nm in all_sets:
            combo.addItem(nm, nm)
        # Restore the explicit value if set
        cur_idx = combo.findData(explicit or "")
        combo.setCurrentIndex(cur_idx if cur_idx >= 0 else 0)
        row.addWidget(combo)
        layout.addLayout(row)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(dlg.reject)
        btn_row.addWidget(cancel)
        save = QPushButton("Save")
        save.setDefault(True)

        def _save():
            pick = combo.currentData() or ""
            self.script.set_weather_set_explicit(pick or None)
            # Refresh whatever node-detail panel is currently showing so
            # the dropdown options pick up the new set immediately
            if self._selected_node_id and self._selected_node_id in self.script.nodes:
                self.props_panel.load_node(self.script, self._selected_node_id)
            dlg.accept()

        save.clicked.connect(_save)
        btn_row.addWidget(save)
        layout.addLayout(btn_row)

        dlg.exec()

    def _cmd_open_story_context(self):
        """Single-editor story context dialog. The previous split layout
        (Full + Focused) was eliminated when the editor moved to a
        unified context — the AI now sees the full story_context (capped
        at CONTEXT_MAX, currently 60,000 chars) via the SYSTEM prompt,
        and the CLI's automatic prompt caching keeps cost reasonable."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Story Context")
        dlg.setMinimumWidth(720)
        dlg.setMinimumHeight(520)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header = QLabel(
            "Story Context — sent to the AI on every generation call. "
            "Cached automatically after the first call in a session."
        )
        header.setStyleSheet("color: #aaddaa; font-size: 11px; font-weight: bold;")
        layout.addWidget(header)

        ctx_edit = QTextEdit()
        ctx_edit.setAcceptRichText(False)   # paste always lands as plain text
        ctx_edit.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        ctx_edit.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        ctx_edit.setLineWrapColumnOrWidth(0)
        ctx_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        ctx_edit.setPlainText(self.script.story_context)
        layout.addWidget(ctx_edit)

        def _char_label_text():
            n = len(ctx_edit.toPlainText())
            color = "#ff7777" if n > CONTEXT_MAX else "#888888"
            return (f'<span style="color:{color};font-size:10px;">'
                    f'{n:,} / {CONTEXT_MAX:,} characters</span>')

        char_lbl = QLabel(_char_label_text())
        char_lbl.setTextFormat(Qt.TextFormat.RichText)
        ctx_edit.textChanged.connect(lambda: char_lbl.setText(_char_label_text()))
        layout.addWidget(char_lbl)

        # Buttons
        btn_row = QHBoxLayout()
        status_lbl = QLabel("")
        status_lbl.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        btn_row.addWidget(status_lbl)
        btn_row.addStretch(1)

        chat_btn = QPushButton("Develop with AI…")
        chat_btn.setToolTip(
            "Open a chat with Claude to build, expand, refine, or iterate "
            "on the story context. Replies can be committed back here.")
        btn_row.addWidget(chat_btn)

        save_btn = QPushButton("Save & Close")
        save_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

        def on_chat():
            if not self.ai.ready:
                status_lbl.setText("Claude CLI not found.")
                return
            sub = StoryContextChatDialog(
                parent=dlg,
                ai=self.ai,
                ui_queue=self.ui_queue,
                initial_full=ctx_edit.toPlainText(),
            )
            if sub.exec() and sub.committed_text is not None:
                if sub.committed_mode == 'append':
                    existing = ctx_edit.toPlainText().rstrip()
                    addition = sub.committed_text.strip()
                    if existing:
                        merged = existing + "\n\n" + addition
                    else:
                        merged = addition
                    ctx_edit.setPlainText(merged)
                    status_lbl.setText(
                        f"Appended {len(addition):,} chars "
                        f"(total {len(merged):,}).")
                else:
                    ctx_edit.setPlainText(sub.committed_text)
                    status_lbl.setText(
                        f"Context replaced ({len(sub.committed_text):,} chars).")

        chat_btn.clicked.connect(on_chat)

        dlg.exec()
        self.script.set_story_context(ctx_edit.toPlainText())
        self._update_title()

    def _set_ai_model(self, model_id: str):
        """Switch the AI model used for node generation."""
        self.ai.model = model_id
        for mid, act in self._model_actions.items():
            act.setChecked(mid == model_id)
        short = model_id.split('-')[1].capitalize()
        self.status_bar.showMessage(f"AI model: {short}", 3000)

    def _set_ai_thinking(self, level: str):
        """Switch the extended-thinking level used for AI calls."""
        self.ai.thinking = level
        for lv, act in self._thinking_actions.items():
            act.setChecked(lv == level)
        pretty = {'off': 'Off', 'think': 'Think',
                  'think_hard': 'Think Hard', 'ultrathink': 'Ultrathink'}.get(level, level)
        self.status_bar.showMessage(f"Thinking: {pretty}", 3000)

    def _cmd_audit_audio(self):
        """Check all nodes for missing or unset audio files and show a report."""
        if not self.script:
            QMessageBox.information(self, "Audit Audio", "No script loaded.")
            return

        no_file = []       # nodes with no 'file' field set
        missing_file = []  # nodes with 'file' set but file doesn't exist
        ok_count = 0

        for nid, nd in self.script.nodes.items():
            if not isinstance(nd, dict):
                continue
            file_rel = nd.get("file")
            if not file_rel:
                no_file.append(nid)
            else:
                full = REPO_ROOT / file_rel
                if full.exists():
                    ok_count += 1
                else:
                    missing_file.append((nid, file_rel))

        total = len(self.script.nodes)
        lines = [f"Audio audit for {total} nodes:\n"]
        lines.append(f"  ✓  {ok_count} nodes have valid audio files")
        if no_file:
            lines.append(f"  —  {len(no_file)} nodes have no audio file assigned")
        if missing_file:
            lines.append(f"  ✗  {len(missing_file)} nodes have missing files:\n")
            for nid, rel in missing_file[:50]:
                label = self.script.nodes.get(nid, {}).get("label", nid)
                lines.append(f"      {label}  →  {rel}")
            if len(missing_file) > 50:
                lines.append(f"      … and {len(missing_file) - 50} more")

        if not no_file and not missing_file:
            lines.append("\nAll audio files are present! ✓")

        msg = QMessageBox(self)
        msg.setWindowTitle("Audit Audio Files")
        msg.setIcon(QMessageBox.Information if not missing_file else QMessageBox.Warning)
        msg.setText("\n".join(lines))
        if missing_file:
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Reset)
            btn_clear = msg.button(QMessageBox.Reset)
            btn_clear.setText(f"Clear {len(missing_file)} Missing")
        else:
            msg.setStandardButtons(QMessageBox.Ok)
        result = msg.exec()
        if missing_file and result == QMessageBox.Reset:
            for nid, _ in missing_file:
                nd = self.script.nodes.get(nid)
                if isinstance(nd, dict):
                    nd.pop("file", None)
                # Refresh graph node color
                self._refresh_node(nid)
            self.script.dirty = True
            # Refresh panel if the current node was affected
            if self.props_panel._node_id and self.props_panel._script:
                self.props_panel.load_node(self.props_panel._script,
                                           self.props_panel._node_id)
            self.status_bar.showMessage(
                f"Cleared audio file from {len(missing_file)} nodes", 5000)

    def _cmd_determine_variables(self):
        """Use AI to infer variable values for nodes that have all-zero vars."""
        variables = self.script.variables
        if not variables:
            self.status_bar.showMessage("No story variables defined — use Story → Story Variables first")
            return
        if not self.ai.ready:
            self.status_bar.showMessage("Claude CLI not found")
            return
        if self.ai.busy:
            self.status_bar.showMessage("AI is busy — wait for current task to finish")
            return

        var_names = [v['name'] for v in variables]
        # Collect nodes where ALL variable values are zero (or missing)
        candidates = []
        for nid, nd in self.script.nodes.items():
            node_vars = nd.get('vars', {})
            if not any(node_vars.get(vn, 0.0) != 0.0 for vn in var_names):
                text = nd.get('text', '').strip()
                if text:
                    candidates.append((nid, text))

        if not candidates:
            self.status_bar.showMessage("All nodes already have variable values set")
            return

        self.status_bar.showMessage(f"Determining variables for {len(candidates)} nodes…")

        def on_done(results):
            count = 0
            for nid, var_vals in results.items():
                if nid not in self.script.nodes or not isinstance(var_vals, dict):
                    continue
                nd = self.script.nodes[nid]
                node_vars = nd.setdefault('vars', {})
                for vn in var_names:
                    val = var_vals.get(vn)
                    if val is not None:
                        try:
                            node_vars[vn] = max(0.0, min(1.0, float(val)))
                        except (ValueError, TypeError):
                            pass
                count += 1
            self.script._dirty = True
            self._update_title()
            # Refresh properties panel if the selected node was updated
            if self._selected_node_id and self._selected_node_id in results:
                self.props_panel.load_node(self.script, self._selected_node_id)
            self.status_bar.showMessage(f"Variables set for {count} nodes")

        def on_error(e):
            self.status_bar.showMessage(f"Determine variables error: {str(e)[:80]}")

        self.ai.determine_variables(candidates, variables, self.ui_queue, on_done, on_error)

    def _cmd_open_story_variables(self):
        """Open dialog to define up to 6 story-level variables."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Story Variables")
        dlg.setMinimumWidth(520)
        dlg.setMinimumHeight(250)
        main_layout = QVBoxLayout(dlg)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        info_lbl = QLabel("Define up to 6 numeric variables (0–1) tracked per node. "
                          "AI will set values based on the description when generating nodes.")
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        main_layout.addWidget(info_lbl)

        # Rows container
        rows_widget = QWidget()
        rows_layout = QVBoxLayout(rows_widget)
        rows_layout.setContentsMargins(0, 0, 0, 0)
        rows_layout.setSpacing(4)
        main_layout.addWidget(rows_widget)

        row_widgets = []  # list of (name_edit, desc_edit, remove_btn, row_widget)

        def add_row(name: str = '', desc: str = ''):
            if len(row_widgets) >= 6:
                return
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(4)
            name_edit = QLineEdit()
            name_edit.setPlaceholderText("Name (e.g. tension)")
            name_edit.setMaxLength(20)
            name_edit.setFixedWidth(130)
            name_edit.setText(name)
            desc_edit = QLineEdit()
            desc_edit.setPlaceholderText("Description for AI (e.g. How tense the moment feels)")
            desc_edit.setText(desc)
            rm_btn = QPushButton("×")
            rm_btn.setFixedWidth(28)
            rm_btn.setStyleSheet("color: #ff6666;")
            rl.addWidget(name_edit)
            rl.addWidget(desc_edit)
            rl.addWidget(rm_btn)
            rows_layout.addWidget(row)
            entry = (name_edit, desc_edit, rm_btn, row)
            row_widgets.append(entry)

            def remove(e=entry):
                # Idempotent guard for the genuine re-fire cases: a rapid
                # double-click (slot can fire again before deleteLater tears the
                # button down) or the AI-suggest "replace" path clearing rows
                # out from under a lingering button.
                if e not in row_widgets:
                    return
                row_widgets.remove(e)
                e[3].deleteLater()
                add_btn.setEnabled(len(row_widgets) < 6)

            # Wrap in a lambda so QPushButton.clicked's `checked` bool is NOT
            # passed as `e`. Connecting `remove` directly made PyQt call
            # remove(False), clobbering the e=entry default, so the handler
            # operated on `False` instead of the row — nothing was ever removed
            # (and pre-guard it raised "list.remove(x): x not in list").
            rm_btn.clicked.connect(lambda *_: remove())
            add_btn.setEnabled(len(row_widgets) < 6)

        # Add variable button
        add_btn = QPushButton("+ Add Variable")
        add_btn.clicked.connect(lambda: add_row())
        main_layout.addWidget(add_btn)

        # AI-suggest row
        suggest_row = QHBoxLayout()
        suggest_lbl = QLabel("Optional guidance for AI:")
        suggest_lbl.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        suggest_row.addWidget(suggest_lbl)
        suggest_hint = QLineEdit()
        suggest_hint.setPlaceholderText(
            'e.g. "more emotional, less plot-driven" — or leave blank')
        suggest_row.addWidget(suggest_hint, stretch=1)
        suggest_btn = QPushButton("Suggest with AI…")
        suggest_btn.setToolTip(
            "Ask Claude to propose a set of 4-6 narrative variables for this\n"
            "script based on its story_context, current variables, sample node\n"
            "texts, and arc summaries. Suggestions can be accepted (replace\n"
            "current variables) or canceled.")
        suggest_row.addWidget(suggest_btn)
        main_layout.addLayout(suggest_row)

        # Status line for the AI call
        ai_status = QLabel("")
        ai_status.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        main_layout.addWidget(ai_status)

        main_layout.addStretch()

        # Save & Close
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        save_btn = QPushButton("Save && Close")
        save_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(save_btn)
        main_layout.addLayout(btn_row)

        # ── AI-suggest wiring ────────────────────────────────────────────
        def _on_suggest():
            if not self.ai.ready:
                ai_status.setText("Claude CLI not found.")
                return
            if self.ai.busy:
                ai_status.setText("AI is busy…")
                return
            # Collect current state for the prompt
            current_vars = []
            for ne, de, _, _ in row_widgets:
                nm = ne.text().strip()
                if nm:
                    current_vars.append({'name': nm, 'description': de.text().strip()})
            # Sample up to 10 representative node texts
            sample_texts = []
            for nid, nd in list(self.script.nodes.items())[:50]:
                t = (nd.get('text') or '').strip()
                if t:
                    sample_texts.append(t)
                if len(sample_texts) >= 10:
                    break
            # Arc summaries: name + premise + themes
            arc_lines = []
            for aid, arc in self.script.arcs.items():
                nm = arc.get('name') or aid
                pr = (arc.get('premise') or '').strip().replace('\n', ' ')
                th = (arc.get('themes') or '').strip()
                line = f"  - {nm}: {pr[:200]}"
                if th:
                    line += f"   [themes: {th[:120]}]"
                arc_lines.append(line)
            arcs_summary = '\n'.join(arc_lines)

            instructions = suggest_hint.text().strip()
            suggest_btn.setEnabled(False)
            ai_status.setText("Asking Claude for variable suggestions…")

            def on_done(suggested):
                suggest_btn.setEnabled(True)
                if not suggested:
                    ai_status.setText("AI returned no variables.")
                    return
                # Confirm replacement
                preview = '\n'.join(
                    f'  • {v["name"]}: {v["description"][:80]}'
                    + ('…' if len(v.get('description', '')) > 80 else '')
                    for v in suggested
                )
                resp = QMessageBox.question(
                    dlg,
                    "Replace variables with AI suggestions?",
                    f"Claude proposes {len(suggested)} variables:\n\n{preview}\n\n"
                    "Replace your current variables with these? (You can edit "
                    "each name/description afterward.)",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if resp != QMessageBox.Yes:
                    ai_status.setText("Suggestions discarded.")
                    return
                # Clear current rows, populate with suggestions
                for _ne, _de, _rb, rw in list(row_widgets):
                    row_widgets.remove((_ne, _de, _rb, rw))
                    rw.deleteLater()
                for v in suggested[:6]:
                    add_row(v.get('name', ''), v.get('description', ''))
                ai_status.setText(f"Added {len(suggested)} suggested variables — edit and Save.")

            def on_error(err):
                suggest_btn.setEnabled(True)
                ai_status.setText(f"AI error: {str(err)[:80]}")

            self.ai.suggest_variables(
                story_context=self.script.story_context,
                current_vars=current_vars,
                sample_nodes=sample_texts,
                arcs_summary=arcs_summary,
                ui_queue=self.ui_queue,
                on_done=on_done,
                on_error=on_error,
                instructions=instructions,
            )

        suggest_btn.clicked.connect(_on_suggest)

        # Populate from current data
        for v in self.script.variables:
            add_row(v.get('name', ''), v.get('description', ''))

        if dlg.exec():
            # Collect and save
            new_vars = []
            for name_edit, desc_edit, _, _ in row_widgets:
                name = name_edit.text().strip()
                if name:
                    new_vars.append({
                        'name': name,
                        'description': desc_edit.text().strip(),
                    })
            self.script.set_variables(new_vars)
            # Refresh the properties panel variable widgets
            self.props_panel.refresh_variable_widgets()
            # Reload current node to populate values
            if self.props_panel._node_id and self.props_panel._script:
                self.props_panel.load_node(self.script, self.props_panel._node_id)
            self._update_title()

    def _cmd_open_voice_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Voice Settings")
        dlg.setMinimumWidth(380)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.voice_panel)
        btn = QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
        dlg.exec()
        # Re-parent back so the panel isn't destroyed with the dialog
        self.voice_panel.setParent(None)

    def _drain_queue(self):
        while True:
            try:
                fn = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.status_bar.showMessage(f"Error: {exc}")

    # ── NodeGraph signal handlers ────────────────────────────────────────────

    def _on_node_selected(self, node):
        if node:
            node_id = next((nid for nid, n in self._node_items.items() if n is node), None)
            if not node_id:
                return
            # Complete a pending "Connect to…" if one is active
            if self._pending_connect_from and node_id != self._pending_connect_from:
                src = self._pending_connect_from
                self._cancel_pending_connect()
                if node_id not in self.script.nodes.get(src, {}).get('next', []):
                    self._node_items[src].output(0).connect_to(
                        self._node_items[node_id].input(0)
                    )
                    self.status_bar.showMessage(f"Connected: {src} → {node_id}")
                else:
                    self.status_bar.showMessage(f"Edge {src} → {node_id} already exists")
                return
            self._selected_node_id = node_id
            self.chat_panel.selected_node_id = node_id
            self.props_panel.load_node(self.script, node_id)
            self.status_bar.showMessage(f"Selected: {node_id}")
            self._apply_highlight(node_id)
        else:
            self._selected_node_id = None
            self.chat_panel.selected_node_id = None
            self._clear_highlight()

    def _on_selection_changed(self):
        selected = self.graph.selected_nodes()
        if not selected:
            self._selected_node_id = None
            self.chat_panel.selected_node_id = None
            self._clear_highlight()

    def _nid_of_node(self, node_obj):
        """O(1) lookup: NodeGraphQt node object → node_id."""
        # Build/use a cached reverse map
        if not hasattr(self, '_node_obj_to_id') or len(self._node_obj_to_id) != len(self._node_items):
            self._node_obj_to_id = {n: nid for nid, n in self._node_items.items()}
        return self._node_obj_to_id.get(node_obj)

    def _on_port_connected(self, in_port, out_port):
        from_id = self._nid_of_node(out_port.node())
        to_id   = self._nid_of_node(in_port.node())
        if not from_id or not to_id:
            return
        self.script.add_edge(from_id, to_id)
        self.props_panel.rebuild_edge_list(self.script, self._selected_node_id or from_id)
        self._update_title()
        self._maybe_refresh_freq()
        self._refresh_cycle_markers()
        self.status_bar.showMessage(f"Edge: {from_id} -> {to_id}")

    def _on_port_disconnected(self, in_port, out_port):
        from_id = self._nid_of_node(out_port.node())
        to_id   = self._nid_of_node(in_port.node())
        if not from_id or not to_id:
            return
        self.script.remove_edge(from_id, to_id)
        if self._selected_node_id:
            self.props_panel.rebuild_edge_list(self.script, self._selected_node_id)
        self._update_title()
        self._maybe_refresh_freq()
        self._refresh_cycle_markers()
        self.status_bar.showMessage(f"Deleted edge: {from_id} -> {to_id}")

    def _on_nodes_deleted(self, _):
        live = {id(n) for n in self.graph.all_nodes()}
        deleted = [nid for nid, n in self._node_items.items() if id(n) not in live]
        for nid in deleted:
            self.script.remove_node(nid)
            self._node_items.pop(nid, None)
        if deleted:
            self.props_panel.clear()
            self._selected_node_id = None
            self._update_title()

    # ── Graph management helpers ─────────────────────────────────────────────

    def _set_node_color(self, node: NarrativeNode, nd: dict, in_cycle: bool = False):
        tags = nd.get('tags', [])
        tag_color = next((TAG_COLORS[t] for t in tags if t in TAG_COLORS), (70, 70, 95))
        has_text  = bool(nd.get('text', '').strip())
        has_audio = bool(nd.get('file'))

        node.set_color(*tag_color)
        if in_cycle:
            node.view.border_color = (220, 50, 50, 255)    # red — cycle!
        elif has_audio:
            node.view.border_color = (60, 210, 120, 255)   # bright green — audio ready
        elif has_text:
            node.view.border_color = (255, 150, 0, 255)    # bright orange — needs audio
        else:
            node.view.border_color = (55, 55, 75, 255)     # dim — no text

    def _rebuild_graph(self):
        # Disconnect signals while rebuilding to prevent clear_session() from
        # wiping script data via nodes_deleted / port_disconnected callbacks.
        self.graph.nodes_deleted.disconnect(self._on_nodes_deleted)
        self.graph.port_disconnected.disconnect(self._on_port_disconnected)
        self.graph.port_connected.disconnect(self._on_port_connected)
        try:
            self.graph.clear_session()
            self._node_items.clear()
            for node_id, nd in self.script.nodes.items():
                pos = nd.get('pos', [100, 100])
                node = self.graph.create_node('narrative.NarrativeNode', name=(nd.get("label") or node_id))
                node.set_pos(float(pos[0]), float(pos[1]))
                self._set_node_color(node, nd, in_cycle=node_id in self._cycle_nodes)
                self._node_items[node_id] = node
            for from_id, nd in self.script.nodes.items():
                for to_id in nd.get('next', []):
                    if to_id in self._node_items and from_id in self._node_items:
                        try:
                            self._node_items[from_id].output(0).connect_to(
                                self._node_items[to_id].input(0)
                            )
                        except Exception:
                            pass
        finally:
            self.graph.port_connected.connect(self._on_port_connected)
            self.graph.port_disconnected.connect(self._on_port_disconnected)
            self.graph.nodes_deleted.connect(self._on_nodes_deleted)
        # Reset hover filter state after rebuild — stale refs would segfault
        if hasattr(self, '_graph_hover_filter'):
            self._graph_hover_filter._current = None
            self._graph_hover_filter._cancel_marquee()
        self._refresh_cycle_markers()
        # Node views were recreated — drop stale overlay refs and re-apply if search is active
        self._search_overlays.clear()
        if self._search_bar.text().strip():
            self._cmd_search(self._search_bar.text())

    def _get_node_at_view_pos(self, view_pos) -> Optional[str]:
        """Return node_id under the given viewport-space position, or None."""
        viewer = self.graph.viewer()
        scene_pos = viewer.mapToScene(view_pos)
        items_at = viewer.scene().items(scene_pos)
        for node_id, node in self._node_items.items():
            node_view = node.view
            for item in items_at:
                curr = item
                while curr is not None:
                    if curr is node_view:
                        return node_id
                    curr = curr.parentItem()
        return None

    def _get_connected_nodes(self, node_id: str):
        """Return (parent_ids, child_ids) directly connected to node_id using script data."""
        nd = self.script.nodes.get(node_id, {})
        children = set(nd.get('next', []))
        parents = set()
        for nid, n in self.script.nodes.items():
            if node_id in n.get('next', []):
                parents.add(nid)
        return parents, children

    def _pipe_connections(self):
        """Yield (pipe_item, from_node_id, to_node_id) for every pipe in the scene."""
        # Build a reverse lookup: view object → node_id (O(N) once)
        view_to_nid = {node.view: nid for nid, node in self._node_items.items()}

        viewer = self.graph.viewer()
        for item in viewer.scene().items():
            if 'Pipe' not in type(item).__name__:
                continue
            in_port  = getattr(item, 'input_port',  None)
            out_port = getattr(item, 'output_port', None)
            if in_port is None or out_port is None:
                continue
            from_nid = view_to_nid.get(out_port.parentItem())
            to_nid   = view_to_nid.get(in_port.parentItem())
            if from_nid and to_nid:
                yield item, from_nid, to_nid

    def _apply_highlight(self, node_id: str):
        """Directional opacity fade — ancestors and descendants only.

        Upstream (parents) and downstream (children) are traced separately
        so siblings, cousins, and other sideways connections stay dimmed.
        """
        nodes = self.script.nodes
        if node_id not in nodes:
            return

        # Build adjacency from script data (O(E))
        children_of = {}
        parents_of = defaultdict(set)
        for nid, nd in nodes.items():
            children_of[nid] = set(nd.get('next', []))
            for cid in nd.get('next', []):
                parents_of[cid].add(nid)

        # Trace upstream (ancestors) — follow parents only
        up_dist = {}
        frontier = {node_id}
        for level in range(1, 10):
            next_frontier = set()
            for nid in frontier:
                for parent in parents_of.get(nid, set()):
                    if parent not in up_dist:
                        up_dist[parent] = level
                        next_frontier.add(parent)
            frontier = next_frontier
            if not frontier:
                break

        # Trace downstream (descendants) — follow children only
        down_dist = {}
        frontier = {node_id}
        for level in range(1, 10):
            next_frontier = set()
            for nid in frontier:
                for child in children_of.get(nid, set()):
                    if child not in down_dist:
                        down_dist[child] = level
                        next_frontier.add(child)
            frontier = next_frontier
            if not frontier:
                break

        # Merge — use the closer distance if a node appears in both
        dist = {node_id: 0}
        for nid, d in up_dist.items():
            dist[nid] = d
        for nid, d in down_dist.items():
            if nid not in dist or d < dist[nid]:
                dist[nid] = d

        OPACITY = [1.0, 1.0, 0.85, 0.70, 0.55, 0.42, 0.32, 0.25, 0.20, 0.16]
        highlighted = set(dist.keys())

        for nid, n in self._node_items.items():
            d = dist.get(nid)
            if d is not None and d < len(OPACITY):
                n.view.setOpacity(OPACITY[d])
            else:
                n.view.setOpacity(0.05)

        # Set pipe opacity by walking output port pipes (no scene().items() scan)
        view_to_nid = {node.view: nid for nid, node in self._node_items.items()}
        try:
            for nid, node in self._node_items.items():
                out_view = node.output(0).view if node.output(0) else None
                if not out_view:
                    continue
                for pipe_item in out_view.connected_pipes:
                    in_pv = getattr(pipe_item, 'input_port', None)
                    if in_pv is None:
                        continue
                    from_nid = nid
                    to_nid = view_to_nid.get(in_pv.parentItem())
                    if from_nid in highlighted and to_nid in highlighted:
                        pipe_item.setOpacity(1.0)
                    else:
                        pipe_item.setOpacity(0.04)
        except Exception:
            pass

    def _apply_search_overlays(self, matched: set):
        """Add crosshatch overlays to all matched nodes; remove from nodes no longer matching."""
        # Remove overlays for nodes that no longer match
        for nid in list(self._search_overlays):
            if nid not in matched:
                item = self._search_overlays.pop(nid)
                try:
                    sc = item.scene()
                    if sc:
                        sc.removeItem(item)
                except Exception:
                    pass
        # Add overlays for newly matching nodes
        for nid in matched:
            if nid not in self._search_overlays and nid in self._node_items:
                self._search_overlays[nid] = _SearchBorderItem(self._node_items[nid].view)

    def _clear_search_overlays(self):
        """Remove all active search crosshatch overlays."""
        for item in self._search_overlays.values():
            try:
                sc = item.scene()
                if sc:
                    sc.removeItem(item)
            except Exception:
                pass
        self._search_overlays.clear()

    def _clear_highlight(self):
        """Restore opacity. Search overlays are managed separately and not touched here."""
        if self._freq_btn.isChecked():
            self._apply_frequency_heat()
            return
        for n in self._node_items.values():
            n.view.setOpacity(1.0)
        # Restore pipe opacity via port traversal (no scene scan)
        try:
            for node in self._node_items.values():
                out = node.output(0)
                if out is None:
                    continue
                try:
                    for pipe_item in out.view.connected_pipes:
                        pipe_item.setOpacity(1.0)
                except Exception:
                    pass
        except Exception:
            pass

    def _on_node_hover_enter(self, node_id: str):
        try:
            self._apply_highlight(node_id)
        except Exception:
            pass
        self.props_panel.preview_node(self.script, node_id)

    def _on_node_hover_leave(self, _node_id: str):
        try:
            if self._selected_node_id:
                self._apply_highlight(self._selected_node_id)
            elif self._freq_btn.isChecked():
                self._apply_frequency_heat()
            else:
                self._clear_highlight()
        except Exception:
            pass
        self.props_panel.end_preview()

    def _on_graph_right_click(self, node_id: Optional[str], global_pos):
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)

        if node_id:
            nd = self.script.nodes.get(node_id, {})
            is_start = node_id in self.script.start_nodes

            act_select = menu.addAction(f"Select  '{nd.get('label') or node_id}'")
            act_select.triggered.connect(lambda: self._select_node(node_id))
            menu.addSeparator()

            act_start = menu.addAction("Remove from Start Nodes" if is_start else "Set as Start Node")
            act_start.triggered.connect(lambda: self._toggle_start_node(node_id))

            act_connect = menu.addAction("Connect to…")
            act_connect.triggered.connect(lambda: self._cmd_start_connect(node_id))

            # Assign node(s) to a story (v2) — applies to the whole selection
            # when the clicked node is part of it. Nodes without a story are
            # invisible to cast-aware generation and the web weaving passes.
            if self.script.arcs:
                sel = self._get_selected_node_ids()
                targets = sel if (len(sel) > 1 and node_id in sel) else [node_id]
                label = (f"Assign to Story ({len(targets)} selected) →"
                         if len(targets) > 1 else "Assign to Story →")
                story_menu = menu.addMenu(label)
                current = self.script.get_node_arc_id(node_id)
                for arc_id, arc in self.script.arcs.items():
                    name = arc.get('name') or arc_id
                    mark = '● ' if arc_id == current else '   '
                    act_s = story_menu.addAction(mark + name)
                    act_s.triggered.connect(
                        lambda _=False, a=arc_id, ts=list(targets):
                            self._cmd_assign_story(ts, a))
                act_none = story_menu.addAction('   (no story)')
                act_none.triggered.connect(
                    lambda _=False, ts=list(targets):
                        self._cmd_assign_story(ts, ''))

            nexts = self.script.nodes.get(node_id, {}).get('next', [])
            if nexts:
                del_conn_menu = menu.addMenu("Delete Connection →")
                for tgt in nexts:
                    act_del = del_conn_menu.addAction(tgt)
                    act_del.triggered.connect(
                        lambda _=False, f=node_id, t=tgt: self._cmd_delete_edge(f, t)
                    )

            menu.addSeparator()

            act_expand = menu.addAction("Expand Node (AI)")
            act_expand.triggered.connect(lambda: self._cmd_expand_node(node_id))
            act_expand.setEnabled(self.ai.ready and not self.ai.busy)

            act_continue = menu.addAction("Continue from Here (AI)")
            act_continue.triggered.connect(lambda: self._cmd_continue_from_node(node_id))
            act_continue.setEnabled(self.ai.ready and not self.ai.busy)

            act_rewrite = menu.addAction("AI Rewrite Text")
            act_rewrite.triggered.connect(lambda: self._cmd_rewrite_node(node_id))
            act_rewrite.setEnabled(self.ai.ready and not self.ai.busy)

            menu.addSeparator()

            selected = self._get_selected_node_ids()
            if len(selected) > 1 and node_id in selected:
                act_gen_audio = menu.addAction(f"Generate Audio ({len(selected)} selected)")
                act_gen_audio.triggered.connect(lambda: self._cmd_generate_audio_for_nodes(selected))
            else:
                act_gen_audio = menu.addAction("Generate Audio")
                act_gen_audio.triggered.connect(lambda: self._cmd_generate_audio_for(node_id))
            act_gen_audio.setEnabled(bool(self.vm.api_key))

            act_play = menu.addAction("Play Audio")
            act_play.triggered.connect(lambda: self._cmd_play_audio_for(node_id))
            act_play.setEnabled(bool(nd.get("file")))

            menu.addSeparator()

            act_delete = menu.addAction("Delete Node")
            act_delete.triggered.connect(lambda: self._cmd_delete_node(node_id))
        else:
            act_add = menu.addAction("Add Node")
            act_add.triggered.connect(self._cmd_add_node)

            if self.ai.ready and not self.ai.busy:
                act_gen = menu.addAction("Generate Graph (AI)")
                act_gen.triggered.connect(self.chat_panel._cmd_generate_parallel)

            menu.addSeparator()
            act_freq = menu.addAction("Toggle Frequency Heat Map")
            act_freq.triggered.connect(lambda: self._freq_btn.setChecked(not self._freq_btn.isChecked()))

            menu.addSeparator()
            act_tree = QAction("Apply Tree Layout", menu)
            act_tree.triggered.connect(self._cmd_apply_tree_layout)
            menu.addAction(act_tree)

            act_fit = QAction("Fit View", menu)
            act_fit.triggered.connect(self._cmd_fit_view)
            menu.addAction(act_fit)

        menu.exec(global_pos)

    def _cmd_assign_story(self, node_ids: list, arc_id: str):
        """Set arc_id on the given nodes (''=none). Story membership drives
        node coloring, cast-aware generation, and the web weaving passes."""
        n = 0
        for nid in node_ids:
            if nid in self.script.nodes:
                self.script.set_node_arc_id(nid, arc_id)
                self._refresh_node(nid)
                n += 1
        name = (self.script.arcs.get(arc_id, {}).get('name') or arc_id) \
            if arc_id else 'no story'
        self.status_bar.showMessage(f"Assigned {n} node(s) to {name}.")
        if self._selected_node_id and self._selected_node_id in node_ids:
            self.props_panel.load_node(self.script, self._selected_node_id)

    def _select_node(self, node_id: str):
        node = self._node_items.get(node_id)
        if node:
            self.graph.clear_selection()
            node.set_selected(True)
            self.props_panel.load_node(self.script, node_id)
            self._selected_node_id = node_id
            self._apply_highlight(node_id)

    def _on_shift_click(self, node_id: str):
        """Toggle a node's selection without clearing other selections (Shift+click)."""
        node = self._node_items.get(node_id)
        if not node:
            return
        currently_selected = node.view.isSelected()
        node.set_selected(not currently_selected)

        # Update state
        selected = self._get_selected_node_ids()
        if len(selected) == 1:
            self._select_node(selected[0])
        elif len(selected) > 1:
            self._selected_node_id = None
            self.props_panel.clear(multi_select=True)
            self._clear_highlight()
            self.status_bar.showMessage(
                f"{len(selected)} nodes selected — Expand/Continue operates on all")
        else:
            self._selected_node_id = None
            self.props_panel.clear()
            self._clear_highlight()

    def _on_marquee_select(self, scene_rect):
        """Handle completion of a marquee drag — select all nodes and pipes within *scene_rect*."""
        self.graph.clear_selection()
        # Also deselect any pipes
        for it in self.graph.viewer().scene().selectedItems():
            it.setSelected(False)

        selected_ids = []
        for nid, node in self._node_items.items():
            item = node.view
            if item and scene_rect.intersects(item.sceneBoundingRect()):
                node.set_selected(True)
                selected_ids.append(nid)

        # Select pipes (connections) that intersect the marquee
        selected_pipes = 0
        for item in self.graph.viewer().scene().items():
            if 'Pipe' in type(item).__name__:
                if scene_rect.intersects(item.sceneBoundingRect()):
                    item.setSelected(True)
                    selected_pipes += 1

        if len(selected_ids) == 1 and selected_pipes == 0:
            self._select_node(selected_ids[0])
        elif selected_ids or selected_pipes:
            self._selected_node_id = None
            multi = len(selected_ids) > 1
            self.props_panel.clear(multi_select=multi)
            self._clear_highlight()
            parts = []
            if selected_ids:
                parts.append(f"{len(selected_ids)} node{'s' if len(selected_ids) != 1 else ''}")
            if selected_pipes:
                parts.append(f"{selected_pipes} connection{'s' if selected_pipes != 1 else ''}")
            self.status_bar.showMessage(f"{' + '.join(parts)} selected — Expand/Continue/Delete")
        else:
            self._selected_node_id = None
            self.props_panel.clear()
            self._clear_highlight()
            self.status_bar.showMessage("")

    def _toggle_start_node(self, node_id: str):
        is_start = node_id in self.script.start_nodes
        self.script.set_start(node_id, not is_start)
        self._refresh_node(node_id)
        self._update_title()
        if self._selected_node_id == node_id:
            self.props_panel.load_node(self.script, node_id)

    def _cmd_rewrite_node(self, node_id: str):
        """Select the node and trigger AI rewrite via props panel."""
        self._select_node(node_id)
        self.props_panel._cmd_rewrite()

    def _cmd_generate_audio_for(self, node_id: str):
        self._select_node(node_id)
        self.props_panel._cmd_generate_audio()

    def _cmd_generate_audio_for_nodes(self, node_ids: list):
        """Generate audio sequentially for a list of nodes, skipping those with existing audio."""
        if not self.vm.api_key:
            self.status_bar.showMessage("ElevenLabs API key not set")
            return

        out_dir = self.script.path.parent if self.script.path else SOUNDS_DIR

        # Build work list — skip nodes without text, voice, or with existing audio
        work = []
        for nid in node_ids:
            nd = self.script.nodes.get(nid, {})
            text = nd.get("text", "").strip()
            if not text:
                continue
            file_rel = nd.get("file")
            if file_rel:
                p = REPO_ROOT / file_rel
                if p.exists() and p.stat().st_size > 10240:
                    continue
            raw = nd.get("voice") or self.script._data.get("voice", "")
            voice_id = self.vm.id_for_name(raw) or raw
            if not voice_id:
                continue
            work.append((nid, nd, voice_id))

        if not work:
            self.status_bar.showMessage("No nodes with text + voice to generate")
            return

        total = len(work)
        self._batch_audio_done = 0
        self._batch_audio_errors = 0
        self.status_bar.showMessage(f"Generating audio: 0 / {total}…")

        def generate_next(remaining):
            if not remaining:
                self.status_bar.showMessage(
                    f"Audio done — {self._batch_audio_done} generated, "
                    f"{self._batch_audio_errors} errors")
                return

            nid, nd, voice_id = remaining[0]
            rest = remaining[1:]
            out_path = out_dir / f"{nid}.mp3"
            settings = {**self.script._data.get("voice_settings", {}),
                        **nd.get("voice_settings", {})}

            def on_done(path: Path):
                try:
                    rel = str(path.relative_to(REPO_ROOT))
                except ValueError:
                    rel = str(path)
                if nid in self.script.nodes:
                    self.script.nodes[nid]["file"] = rel
                    self.script.dirty = True
                self.props_panel.node_modified.emit(nid)
                self._batch_audio_done += 1
                self.status_bar.showMessage(
                    f"Generating audio: "
                    f"{self._batch_audio_done + self._batch_audio_errors} / {total}…")
                generate_next(rest)

            def on_error(_: str):
                self._batch_audio_errors += 1
                self.status_bar.showMessage(
                    f"Generating audio: "
                    f"{self._batch_audio_done + self._batch_audio_errors} / {total}… "
                    f"({self._batch_audio_errors} errors)")
                generate_next(rest)

            self.vm.generate(
                text=nd.get("text", "").strip(), voice_id=voice_id,
                out_path=out_path, settings=settings,
                ui_queue=self.ui_queue, on_done=on_done, on_error=on_error,
            )

        generate_next(work)

    def _cmd_play_audio_for(self, node_id: str):
        self._select_node(node_id)
        self.props_panel._cmd_play_node_audio()

    def _refresh_node(self, node_id: str):
        node = self._node_items.get(node_id)
        if node:
            nd = self.script.nodes.get(node_id, {})
            self._set_node_color(node, nd, in_cycle=node_id in self._cycle_nodes)
            display = nd.get("label") or node_id
            node.set_name(display)

    def _sync_positions(self):
        for node_id, node in self._node_items.items():
            pos = node.pos()
            self.script.update_pos(node_id, [pos[0], pos[1]])

    # ── Node/edge commands ───────────────────────────────────────────────────

    def _on_node_modified(self, signal: str):
        """Handle signals from PropertiesPanel (regular node_id or special commands)."""
        if signal.startswith("__delete__"):
            node_id = signal[len("__delete__"):]
            self._cmd_delete_node(node_id)
            return
        if signal.startswith("__expand__"):
            parts = signal[len("__expand__"):].split("__")
            node_id = parts[0]
            mn = int(parts[1]) if len(parts) > 1 else 2
            mx = int(parts[2]) if len(parts) > 2 else 5
            self._cmd_expand_node(node_id, node_min=mn, node_max=mx)
            return
        if signal.startswith("__delete_edge__"):
            parts = signal[len("__delete_edge__"):].split("__")
            if len(parts) == 2:
                self._cmd_delete_edge(parts[0], parts[1])
            return
        # Regular modification — refresh appearance
        self._refresh_node(signal)
        self._update_title()

    def _on_graph_generated(self):
        """Called after AI generates a graph.

        If the graph already has the right number of node items (incremental adds
        kept it in sync), skip the expensive full rebuild and just update layout.
        """
        n_script = len(self.script.nodes)
        n_items  = len(self._node_items)

        if n_items < n_script:
            # Some nodes were added to script but not to the graph — do incremental
            missing = set(self.script.nodes.keys()) - set(self._node_items.keys())
            if missing:
                self._add_nodes_incremental(missing)

        # Only do a full rebuild if the graph is badly out of sync
        if abs(len(self._node_items) - n_script) > n_script * 0.2:
            self._rebuild_graph()

        # Sync any edges that exist in script data but not in the visual graph
        self._sync_missing_edges()

        # Layout: reposition existing items (no recreation)
        try:
            self._cmd_apply_tree_layout()
        except Exception:
            pass  # layout failure shouldn't crash
        self._update_title()
        self._refresh_cycle_markers()
        self.status_bar.showMessage(f"Graph updated: {len(self.script.nodes)} nodes")

    def _sync_missing_edges(self):
        """Wire any script edges that don't have a visual pipe in the graph."""
        try:
            self._sync_missing_edges_inner()
        except Exception as exc:
            print(f'[Sync] Edge sync failed: {exc}')

    def _sync_missing_edges_inner(self):
        # Build set of existing visual edges
        view_to_nid = {node.view: nid for nid, node in self._node_items.items()}
        existing_edges = set()
        for item in self.graph.viewer().scene().items():
            if 'Pipe' not in type(item).__name__:
                continue
            out_port = getattr(item, 'output_port', None)
            in_port = getattr(item, 'input_port', None)
            if out_port and in_port:
                fn = view_to_nid.get(out_port.parentItem())
                tn = view_to_nid.get(in_port.parentItem())
                if fn and tn:
                    existing_edges.add((fn, tn))

        # Wire missing edges
        self.graph.port_connected.disconnect(self._on_port_connected)
        try:
            added = 0
            for from_id, nd in self.script.nodes.items():
                if from_id not in self._node_items:
                    continue
                for to_id in nd.get('next', []):
                    if to_id not in self._node_items:
                        continue
                    if (from_id, to_id) not in existing_edges:
                        try:
                            self._node_items[from_id].output(0).connect_to(
                                self._node_items[to_id].input(0)
                            )
                            added += 1
                        except Exception:
                            pass
            if added:
                print(f'[Sync] Wired {added} missing edges')
        finally:
            self.graph.port_connected.connect(self._on_port_connected)

    def _add_nodes_incremental(self, new_node_ids: set):
        """Add only newly-created nodes/edges to the live graph without a full rebuild.
        Used during iterative generation to avoid repeated clear_session() calls."""
        self.graph.port_connected.disconnect(self._on_port_connected)
        self.graph.port_disconnected.disconnect(self._on_port_disconnected)
        try:
            for node_id in new_node_ids:
                if node_id in self._node_items:
                    continue   # already present
                nd = self.script.nodes.get(node_id)
                if not nd:
                    continue
                pos  = nd.get('pos', [100, 100])
                node = self.graph.create_node('narrative.NarrativeNode',
                                              name=(nd.get('label') or node_id))
                node.set_pos(float(pos[0]), float(pos[1]))
                self._set_node_color(node, nd, in_cycle=node_id in self._cycle_nodes)
                self._node_items[node_id] = node
            # Wire only edges that touch the new nodes
            for node_id in new_node_ids:
                nd = self.script.nodes.get(node_id)
                if not nd:
                    continue
                for to_id in nd.get('next', []):
                    if to_id in self._node_items and node_id in self._node_items:
                        try:
                            self._node_items[node_id].output(0).connect_to(
                                self._node_items[to_id].input(0)
                            )
                        except Exception:
                            pass
            # Wire edges involving new nodes (both directions)
            involved = set(new_node_ids)
            for node_id in new_node_ids:
                nd = self.script.nodes.get(node_id)
                if not nd:
                    continue
                for to_id in nd.get('next', []):
                    if to_id in self._node_items:
                        try:
                            self._node_items[node_id].output(0).connect_to(
                                self._node_items[to_id].input(0)
                            )
                        except Exception:
                            pass
            for from_id, nd in self.script.nodes.items():
                if from_id in involved or from_id not in self._node_items:
                    continue
                for to_id in nd.get('next', []):
                    if to_id in involved and to_id in self._node_items:
                        try:
                            self._node_items[from_id].output(0).connect_to(
                                self._node_items[to_id].input(0)
                            )
                        except Exception:
                            pass
        finally:
            self.graph.port_connected.connect(self._on_port_connected)
            self.graph.port_disconnected.connect(self._on_port_disconnected)
        self._update_title()
        self._refresh_cycle_markers()
        if self._search_bar.text().strip():
            self._cmd_search(self._search_bar.text())

    def _cmd_add_node(self):
        node_id = _next_node_id(self.script)
        existing = [nd.get("pos", [0, 0]) for nd in self.script.nodes.values()]
        col_x = 60
        max_y = max((p[1] for p in existing if p[0] < col_x + 150), default=-40)
        pos = [col_x, max_y + 140]
        self.script.add_node(node_id, pos=pos)
        node = self.graph.create_node('narrative.NarrativeNode', name=node_id)
        node.set_pos(float(pos[0]), float(pos[1]))
        self._set_node_color(node, self.script.nodes[node_id])
        self._node_items[node_id] = node
        self._selected_node_id = node_id
        self.props_panel.load_node(self.script, node_id)
        self._update_title()
        self._maybe_refresh_freq()
        if self._search_bar.text().strip():
            self._cmd_search(self._search_bar.text())
        self.status_bar.showMessage(f"Added '{node_id}'")

    def _cmd_start_connect(self, node_id: str):
        self._pending_connect_from = node_id
        self.graph.viewer().viewport().setCursor(Qt.CursorShape.CrossCursor)
        label = self.script.nodes.get(node_id, {}).get('label') or node_id
        self.status_bar.showMessage(
            f"Connect from '{label}' — click the target node  (Escape to cancel)"
        )
        # Create a rubber-band line starting at the source node's output port
        try:
            from PySide6.QtWidgets import QGraphicsLineItem
            from PySide6.QtGui import QPen, QColor
            from PySide6.QtCore import Qt as _Qt2
            out_port = self._node_items[node_id].output(0)
            origin = out_port.view.scenePos()
            pen = QPen(QColor(200, 200, 200), 2, _Qt2.PenStyle.DashLine)
            line = QGraphicsLineItem(origin.x(), origin.y(), origin.x(), origin.y())
            line.setPen(pen)
            line.setZValue(1000)
            self.graph.scene().addItem(line)
            self._pending_connect_line = line
        except Exception:
            self._pending_connect_line = None

    def _on_graph_mouse_move(self, scene_pos):
        """Update rubber-band line endpoint as the mouse moves over the graph."""
        if self._pending_connect_line is None:
            return
        try:
            l = self._pending_connect_line.line()
            self._pending_connect_line.setLine(l.x1(), l.y1(),
                                               scene_pos.x(), scene_pos.y())
        except Exception:
            pass

    def _cancel_pending_connect(self):
        if self._pending_connect_from:
            self._pending_connect_from = None
            self.graph.viewer().viewport().unsetCursor()
            self.status_bar.showMessage("Connection cancelled")
        if self._pending_connect_line is not None:
            try:
                self.graph.scene().removeItem(self._pending_connect_line)
            except Exception:
                pass
            self._pending_connect_line = None

    def _cmd_delete_node(self, node_id: str = None):
        nid = node_id or self._selected_node_id
        if not nid or nid not in self.script.nodes:
            self.status_bar.showMessage("No node selected")
            return
        # Remove from NodeGraph if it still exists there
        node = self._node_items.pop(nid, None)
        if node:
            try:
                self.graph.delete_nodes([node])
            except Exception:
                pass
        self.script.remove_node(nid)
        if self._selected_node_id == nid:
            self._selected_node_id = None
            self.props_panel.clear()
        self._update_title()
        self._maybe_refresh_freq()
        self.status_bar.showMessage(f"Deleted '{nid}'")

    def _cmd_delete_selected_pipes(self) -> bool:
        """Delete selected pipes and/or nodes. Returns True if anything was deleted."""
        deleted = False
        for item, from_id, to_id in list(self._pipe_connections()):
            if item.isSelected():
                self._cmd_delete_edge(from_id, to_id)
                deleted = True
        selected_nids = [
            nid for nid, node in self._node_items.items()
            if node.view.isSelected()
        ]
        for nid in selected_nids:
            self._cmd_delete_node(nid)
            deleted = True
        return deleted

    def _cmd_delete_edge(self, from_id: str, to_id: str):
        src = self._node_items.get(from_id)
        tgt = self._node_items.get(to_id)
        if src and tgt:
            for p in src.output(0).connected_ports():
                if p.node() is tgt:
                    src.output(0).disconnect_from(p)
                    break
        else:
            # Nodes not in graph (shouldn't happen) — clean up data directly
            self.script.remove_edge(from_id, to_id)
            if self._selected_node_id:
                self.props_panel.rebuild_edge_list(self.script, self._selected_node_id)
            self._update_title()
            self._maybe_refresh_freq()

    def _get_selected_node_ids(self) -> list:
        """Return list of all currently selected node IDs."""
        return [nid for nid, node in self._node_items.items()
                if node.view.isSelected()]

    def _cmd_expand_node(self, node_id: str, node_min: int = 2, node_max: int = 5):
        """Expand node(s) using parallel orchestrator (1-2 layers deep).

        If multiple nodes are selected, new children share all selected nodes as parents.
        """
        selected = self._get_selected_node_ids()
        parent_ids = selected if len(selected) > 1 else [node_id]
        parent_ids = [nid for nid in parent_ids if nid in self.script.nodes]

        if not parent_ids:
            self.status_bar.showMessage("Select a node to expand")
            return
        if not self.ai.ready:
            self.status_bar.showMessage("claude CLI not found")
            return

        multi = len(parent_ids) > 1
        label = ', '.join(parent_ids[:5]) + ('...' if len(parent_ids) > 5 else '')
        action = "Merging" if multi else "Expanding"
        self.status_bar.showMessage(f"{action} {len(parent_ids)} node(s) (parallel)...")
        self.chat_panel.append_message("assistant",
            f"[Parallel {'merge-expand' if multi else 'expand'} {label} ({node_min}-{node_max} nodes)]")

        # Expand profile with the user's min/max as the width for every beat
        expand_profile = {
            'max_depth': 2,
            'widths': {'*': (node_min, node_max)},
        }

        self._job_counter += 1
        job_tag = f"expand#{self._job_counter}"

        def _make_expand_orch(jt):
            o = ParallelNodeOrchestrator(
                script=self.script,
                ui_queue=self.ui_queue,
                model=self.ai.model,
                thinking=self.ai.thinking,
                profile=expand_profile,
                story_context=self.script.story_context_focused,
                variables=self.script.variables,
                on_progress=lambda msg: self.status_bar.showMessage(f"[{jt}] {msg}"),
                on_complete=lambda: self._on_orchestrator_complete(o, jt, "Expand"),
                on_node_added=self._add_nodes_incremental,
            )
            return o

        orch = _make_expand_orch(job_tag)
        self._register_orchestrator(orch)
        if multi:
            orch.start_merged(parent_ids, batch_size=random.randint(node_min, node_max))
        else:
            orch.start(parent_ids)

    def _cmd_continue_from_node(self, node_id: str):
        """Continue from node(s) using parallel orchestrator (all remaining layers).

        If multiple nodes are selected, new children share all selected nodes as parents.
        """
        selected = self._get_selected_node_ids()
        parent_ids = selected if len(selected) > 1 else [node_id]
        parent_ids = [nid for nid in parent_ids if nid in self.script.nodes]

        if not parent_ids:
            self.status_bar.showMessage("Select a node to continue from")
            return
        if not self.ai.ready:
            self.status_bar.showMessage("claude CLI not found")
            return

        multi = len(parent_ids) > 1
        label = ', '.join(parent_ids[:5]) + ('...' if len(parent_ids) > 5 else '')
        action = "Merge-continuing" if multi else "Continuing"
        self.status_bar.showMessage(f"{action} from {len(parent_ids)} node(s) (parallel)...")
        self.chat_panel.append_message("assistant",
            f"[Parallel {'merge-continue' if multi else 'continue'} from {label}]")

        self._job_counter += 1
        job_tag = f"continue#{self._job_counter}"

        def _make_continue_orch(jt):
            o = ParallelNodeOrchestrator(
                script=self.script,
                ui_queue=self.ui_queue,
                model=self.ai.model,
                thinking=self.ai.thinking,
                profile='continue',
                story_context=self.script.story_context_focused,
                variables=self.script.variables,
                on_progress=lambda msg: self.status_bar.showMessage(f"[{jt}] {msg}"),
                on_complete=lambda: self._on_orchestrator_complete(o, jt, "Continue"),
                on_node_added=self._add_nodes_incremental,
            )
            return o

        orch = _make_continue_orch(job_tag)
        self._register_orchestrator(orch)
        if multi:
            orch.start_merged(parent_ids)
        else:
            orch.start(parent_ids)

    # ── Seam audit (flow naturalness) ───────────────────────────────────────

    def _collect_seams(self, max_seams: int = 120) -> list:
        """Seams to audit as (from_id, to_id, cross_story). Every cross-story
        edge is included (junctions and bridges are the riskiest seams);
        same-story edges fill the remaining budget as an even sample."""
        cross, intra = [], []
        for nid, nd in self.script.nodes.items():
            if not nd.get('text'):
                continue
            fa = self.script.get_node_arc_id(nid)
            for tgt in nd.get('next', []):
                tnd = self.script.nodes.get(tgt)
                if not tnd or not tnd.get('text'):
                    continue
                ta = self.script.get_node_arc_id(tgt)
                if fa != ta and fa and ta:
                    cross.append((nid, tgt, True))
                else:
                    intra.append((nid, tgt, False))
        seams = cross[:max_seams]
        room = max_seams - len(seams)
        if room > 0 and intra:
            step = max(1, len(intra) // room)
            seams += intra[::step][:room]
        return seams

    def _cmd_seam_audit(self):
        """AI pass: listen-check transitions between connected nodes.
        Read-only — reports jarring seams; fix via AI Rewrite on the child."""
        if not self.ai.ready:
            self.status_bar.showMessage("claude CLI not found")
            return
        if getattr(self, '_seam_audit_running', False):
            self.status_bar.showMessage("Seam audit already running")
            return
        seams = self._collect_seams()
        if not seams:
            self.status_bar.showMessage("No seams to audit — no connected nodes with text.")
            return

        # Chunk into calls; build every prompt on the UI thread.
        CHUNK = 30
        valid_edges = {(a, b) for a, b, _ in seams}
        chunks = []
        for i in range(0, len(seams), CHUNK):
            lines = [f'SEAMS ({i + 1}-{min(i + CHUNK, len(seams))} of {len(seams)}):']
            for a, b, is_cross in seams[i:i + CHUNK]:
                pa = self.script.nodes[a].get('text', '')
                pb = self.script.nodes[b].get('text', '')
                marker = '  [CROSS-STORY]' if is_cross else ''
                lines.append(f'- {a} → {b}{marker}')
                lines.append(f'    parent ending: "…{pa[-220:]}"')
                lines.append(f'    child opening: "{pb[:220]}…"')
            lines.append('\nAudit these seams now. Output the JSON object only.')
            chunks.append('\n'.join(lines))

        self._seam_audit_running = True
        n_cross = sum(1 for _, _, c in seams if c)
        self.status_bar.showMessage(
            f"Seam audit: {len(seams)} seam(s) ({n_cross} cross-story) in "
            f"{len(chunks)} call(s)…")

        def run():
            def one(prompt):
                data = self.ai._run_claude_json(SYSTEM_SEAM_AUDIT, prompt,
                                                max_retries=2)
                out = []
                for iss in (data.get('issues') or []):
                    if not isinstance(iss, dict):
                        continue
                    a, b = iss.get('from'), iss.get('to')
                    if (a, b) not in valid_edges:
                        continue
                    sev = 'hard' if iss.get('severity') == 'hard' else 'soft'
                    out.append((a, b, sev, str(iss.get('reason') or '')[:300]))
                return out

            def prog(done_n, total):
                self.ui_queue.put(
                    lambda d=done_n, n=total: self.status_bar.showMessage(
                        f"Seam audit… {d}/{n} batch(es) done"))

            results = _fan_out_ai_calls(chunks, one, prog)
            issues = [iss for r in results if r for iss in r]

            def done():
                self._seam_audit_running = False
                self._show_seam_results(issues, len(seams))
            self.ui_queue.put(done)

        threading.Thread(target=run, daemon=True).start()

    def _show_seam_results(self, issues: list, n_checked: int):
        if not issues:
            self.status_bar.showMessage(
                f"Seam audit: all {n_checked} checked seam(s) read clean.")
            return
        arc_of = self.script.get_node_arc_id
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Seam Audit — {len(issues)} rough seam(s) "
                           f"of {n_checked} checked")
        dlg.setWindowFlags(dlg.windowFlags() | Qt.Window)
        dlg.resize(780, 520)
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel(
            "Double-click a seam to select its CHILD node (the usual fix: "
            "AI Rewrite its opening to pick up the parent's thread). The "
            "audit changes nothing itself."))
        lst = QListWidget()
        for a, b, sev, reason in sorted(issues, key=lambda x: x[2] != 'hard'):
            cross = ' [cross-story]' if (arc_of(a) != arc_of(b)
                                         and arc_of(a) and arc_of(b)) else ''
            item = QListWidgetItem(
                f"[{sev.upper()}] {a} → {b}{cross}\n    {reason}")
            item.setData(Qt.ItemDataRole.UserRole, b)
            item.setForeground(QColor('#ff7060' if sev == 'hard' else '#e0b050'))
            lst.addItem(item)
        lst.itemDoubleClicked.connect(
            lambda item: self._select_node(item.data(Qt.ItemDataRole.UserRole)))
        v.addWidget(lst, stretch=1)
        row = QHBoxLayout()
        row.addStretch(1)
        btn = QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        row.addWidget(btn)
        v.addLayout(row)
        dlg.show()   # modeless so double-click selection stays usable
        self._seam_dlg = dlg

    def _register_orchestrator(self, orch):
        """Track a generation run so Stop AI Generation can cancel it.
        All entry paths (chat panel, Stories dialog, expand, continue)
        register here. Finished runs are pruned opportunistically."""
        self._orchestrators = [o for o in self._orchestrators if o.running]
        self._orchestrators.append(orch)

    def _cmd_stop_generation(self):
        """Cancel every running generation job."""
        active = [o for o in self._orchestrators if o.running]
        self._orchestrators = []
        if not active:
            self.status_bar.showMessage("No AI generation running.")
            return
        for o in active:
            o.cancel()
        self.status_bar.showMessage(
            f"Cancelled {len(active)} generation job(s) — in-flight calls "
            "finish in the background and are discarded.")

    def _on_orchestrator_complete(self, orch, job_tag: str, verb: str):
        """Called when any orchestrator finishes. Cleans up and refreshes UI."""
        if orch in self._orchestrators:
            self._orchestrators.remove(orch)
        n_active = len(self._orchestrators)
        if n_active:
            self.status_bar.showMessage(
                f"[{job_tag}] {verb} complete — {len(self.script.nodes)} nodes "
                f"({n_active} job{'s' if n_active != 1 else ''} still running)")
        else:
            self.status_bar.showMessage(
                f"{verb} complete — {len(self.script.nodes)} total nodes")
        self._sync_missing_edges()
        self._update_title()
        self._cmd_apply_tree_layout()

    def _run_frequency_simulation(self, n_runs: int = 2000,
                                  web_stats: Optional[dict] = None) -> dict:
        """Monte Carlo random walk with recency damping.

        Estimates each node's audio duration from word count (121 wpm) plus a
        3 s inter-node delay.  Tracks simulated time so recency counters decay
        at the same rate as the real player (1 count per 36000 s / 10 hours).

        Pass a dict as `web_stats` to additionally collect story-level
        statistics (v2 Phase 4): per-story visit occupancy, walk starts by
        story, story-boundary crossings with the edges that carried them,
        and how many walks drifted through 2+ stories.
        """
        nodes = self.script.nodes
        starts = self.script.start_nodes or list(nodes.keys())
        counts = {nid: 0 for nid in nodes}

        arc_map = None
        if web_stats is not None:
            arc_map = {nid: self.script.get_node_arc_id(nid) for nid in nodes}
            web_stats.update({
                'walks': n_runs,
                'occupancy': defaultdict(int),      # arc_id ('' = none) -> visits
                'starts': defaultdict(int),         # arc_id -> walk starts
                'transitions': defaultdict(int),    # (from_arc, to_arc) -> crossings
                'edge_usage': defaultdict(int),     # (from_nid, to_nid) -> crossings
                'drift_walks': 0,                   # walks visiting 2+ stories
            })

        INTER_NODE_DELAY = 3.0          # seconds between nodes
        WPM              = 121.0        # measured speech rate
        DECAY_PER_SEC    = 1.0 / 36000.0 # matches narrative_player (10 hours)

        # Pre-compute estimated duration for each node (speech + delay)
        node_dur = {}
        for nid, nd in nodes.items():
            words = len(nd.get('text', '').split())
            node_dur[nid] = (words / WPM) * 60.0 + INTER_NODE_DELAY

        for _ in range(n_runs):
            current = random.choice(starts)
            steps = 0
            recency: dict = {}
            sim_time = 0.0
            walk_stories: set = set()
            if web_stats is not None:
                web_stats['starts'][arc_map.get(current, '')] += 1
            while current and steps < 300:
                if current not in nodes:
                    break
                counts[current] += 1
                if web_stats is not None:
                    cur_arc = arc_map.get(current, '')
                    web_stats['occupancy'][cur_arc] += 1
                    if cur_arc:
                        walk_stories.add(cur_arc)

                # Advance simulated clock and decay recency
                dt = node_dur.get(current, INTER_NODE_DELAY)
                sim_time += dt
                decay = DECAY_PER_SEC * dt
                if recency:
                    expired = []
                    for rid in recency:
                        recency[rid] -= decay
                        if recency[rid] <= 0.0:
                            expired.append(rid)
                    for rid in expired:
                        del recency[rid]

                # Record visit
                recency[current] = recency.get(current, 0.0) + 1.0

                nd = nodes[current]
                nexts = nd.get('next', [])
                weights = nd.get('weights', [1.0] * len(nexts))
                if not nexts:
                    break

                # Apply recency penalty (same formula as narrative_player)
                effective = []
                for nid, w in zip(nexts, weights):
                    if nid in recency:
                        w *= 2.0 ** (-recency[nid])
                    effective.append(w)

                total = sum(effective) or 1.0
                r = random.random() * total
                acc = 0.0
                nxt = nexts[-1]
                for nid, w in zip(nexts, effective):
                    acc += w
                    if r <= acc:
                        nxt = nid
                        break
                if web_stats is not None and nxt in nodes:
                    next_arc = arc_map.get(nxt, '')
                    if next_arc != arc_map.get(current, ''):
                        web_stats['transitions'][
                            (arc_map.get(current, ''), next_arc)] += 1
                        web_stats['edge_usage'][(current, nxt)] += 1
                current = nxt
                steps += 1
            if web_stats is not None and len(walk_stories) >= 2:
                web_stats['drift_walks'] += 1
        return counts

    def _cmd_frequency_analysis(self, active: bool):
        """Toggle frequency heat map on/off. Bright = frequently visited, dim = rarely reached."""
        if not active:
            self._freq_counts = {}
            self._clear_highlight()
            self.status_bar.showMessage("Frequency map off")
            return
        if not self.script or not self.script.nodes:
            self._freq_btn.setChecked(False)
            self.status_bar.showMessage("No nodes to analyse")
            return

        N_RUNS = 2000
        self._freq_counts = self._run_frequency_simulation(N_RUNS)
        self._apply_frequency_heat()
        total_visits = sum(self._freq_counts.values()) or 1
        self.status_bar.showMessage(
            f"Frequency map: {N_RUNS} runs, {total_visits} total node visits — "
            "bright = frequent, dim = rare"
        )

    def _maybe_refresh_freq(self):
        """Re-run simulation and refresh heat map if the freq toggle is on."""
        if self._freq_btn.isChecked():
            self._freq_counts = self._run_frequency_simulation(2000)
            self._apply_frequency_heat()

    # ── Web statistics (v2 Phase 4) ─────────────────────────────────────────

    def _cmd_web_statistics(self):
        """Story-level Monte Carlo report: occupancy per story, walk starts,
        drift rate, and which edges carry story-boundary crossings."""
        if not self.script.nodes:
            self.status_bar.showMessage("No nodes to analyse")
            return
        N_RUNS = 2000
        stats: dict = {}
        counts = self._run_frequency_simulation(N_RUNS, web_stats=stats)

        def sname(arc_id):
            if not arc_id:
                return '(no story)'
            return self.script.arcs.get(arc_id, {}).get('name') or arc_id

        story_nodes = defaultdict(int)
        for nid in self.script.nodes:
            story_nodes[self.script.get_node_arc_id(nid)] += 1

        total_visits = sum(stats['occupancy'].values()) or 1
        lines = [f'Web statistics — {N_RUNS} simulated walks, '
                 f'{sum(counts.values())} node visits',
                 f'{len(self.script.arcs)} stor'
                 f'{"y" if len(self.script.arcs) == 1 else "ies"}, '
                 f'{len(self.script.nodes)} nodes, '
                 f'{len(self.script.all_story_relations())} relation(s)', '']

        lines.append('Occupancy (share of all node visits):')
        for arc_id, v in sorted(stats['occupancy'].items(),
                                key=lambda kv: -kv[1]):
            lines.append(f'  {v / total_visits * 100:5.1f}%  {sname(arc_id)}'
                         f'  ({story_nodes.get(arc_id, 0)} nodes)')
        # stories that never got visited at all
        for arc_id in self.script.arcs:
            if arc_id not in stats['occupancy']:
                lines.append(f'    0.0%  {sname(arc_id)}'
                             f'  ({story_nodes.get(arc_id, 0)} nodes) — unreachable')
        lines.append('')

        lines.append('Walk starts by story:')
        for arc_id, v in sorted(stats['starts'].items(), key=lambda kv: -kv[1]):
            lines.append(f'  {v / N_RUNS * 100:5.1f}%  {sname(arc_id)}')
        lines.append('')

        drift_pct = stats['drift_walks'] / max(1, stats['walks']) * 100
        lines.append(f'Drift: {drift_pct:.1f}% of walks visited 2+ stories.')
        lines.append('')

        if stats['transitions']:
            arc_of = {nid: self.script.get_node_arc_id(nid)
                      for nid in self.script.nodes}
            edges_by_pair = defaultdict(list)
            for (a, b), n in stats['edge_usage'].items():
                edges_by_pair[(arc_of.get(a, ''), arc_of.get(b, ''))].append(
                    (a, b, n))
            lines.append('Story boundary crossings:')
            for (fa, ta), n in sorted(stats['transitions'].items(),
                                      key=lambda kv: -kv[1]):
                lines.append(f'  {sname(fa)} → {sname(ta)}: {n} crossing(s)')
                for a, b, en in sorted(edges_by_pair.get((fa, ta), []),
                                       key=lambda x: -x[2])[:6]:
                    lines.append(f'      {a} → {b}   ({en})')
        else:
            lines.append('No story boundary crossings — the stories are '
                         'isolated. Add junction relations in the Web Planner '
                         'and run Weave Junctions to connect them.')

        n_un = story_nodes.get('', 0)
        if n_un and len(self.script.arcs) > 0:
            lines += ['', f'! {n_un} node(s) have no story — right-click → '
                          'Assign to Story to include them in the web.']

        # Ending reachability — a walk entering one of these nodes can
        # NEVER finish (no path to a terminal; recency can't escape a
        # pocket with no exit).
        trapped = self.script.nodes_that_cannot_end()
        lines.append('')
        if trapped:
            by_story = defaultdict(list)
            for nid in trapped:
                by_story[self.script.get_node_arc_id(nid)].append(nid)
            lines.append(f'!! {len(trapped)} node(s) can NEVER reach an '
                         'ending — walks entering them are trapped forever:')
            for arc_id, nids in sorted(by_story.items(),
                                       key=lambda kv: -len(kv[1])):
                sample = ', '.join(sorted(nids)[:5])
                more = f' … +{len(nids) - 5}' if len(nids) > 5 else ''
                lines.append(f'    {sname(arc_id)}: {len(nids)} — '
                             f'{sample}{more}')
            lines.append('    Fix: give the loop an exit edge toward a '
                         'terminal node, or a terminal of its own.')
        else:
            lines.append('Ending reachability: OK — every node can reach '
                         'a terminal.')

        dlg = QDialog(self)
        dlg.setWindowTitle("Web Statistics")
        dlg.resize(700, 560)
        v = QVBoxLayout(dlg)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        te.setPlainText('\n'.join(lines))
        v.addWidget(te, stretch=1)
        row = QHBoxLayout()
        row.addStretch(1)
        btn = QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        row.addWidget(btn)
        v.addLayout(row)
        dlg.exec()

    # ── Consistency audit (v2 Phase 4) ──────────────────────────────────────

    def _cmd_consistency_audit(self):
        """AI pass: check every story's node texts against its canon (entity
        cards + canon events). Read-only — reports issues, changes nothing."""
        if not self.ai.ready:
            self.status_bar.showMessage("claude CLI not found")
            return
        if getattr(self, '_audit_running', False):
            self.status_bar.showMessage("Audit already running")
            return
        jobs = []
        for arc_id, arc in self.script.arcs.items():
            story_nodes = [(nid, nd) for nid, nd in self.script.nodes.items()
                           if self.script.get_node_arc_id(nid) == arc_id
                           and nd.get('text')]
            if not story_nodes:
                continue
            cast_codex = self.script.cast_codex_for(arc_id)
            if not cast_codex:
                continue   # no canon to audit against
            system = self.ai._augment_system_with_context(
                SYSTEM_CONSISTENCY_AUDIT, '', cast_codex=cast_codex)
            parts = [f'STORY "{arc.get("name") or arc_id}":\n'
                     f'  premise: {(arc.get("premise") or "")[:600]}']
            tl = ['NODE TEXTS:']
            for nid, nd in story_nodes[:60]:
                tl.append(f'  [{nid}] "{nd.get("text", "")[:400]}"')
            parts.append('\n'.join(tl))
            parts.append('Audit the nodes now. Output the JSON object only.')
            jobs.append((arc.get('name') or arc_id, system,
                         '\n\n'.join(parts), {nid for nid, _ in story_nodes}))
        if not jobs:
            self.status_bar.showMessage(
                "Nothing to audit — no story has both nodes and a codex.")
            return
        self._audit_running = True
        self.status_bar.showMessage(
            f"Consistency audit: {len(jobs)} stor"
            f"{'y' if len(jobs) == 1 else 'ies'}…")

        def run():
            def one(job):
                name, system, prompt, valid = job
                data = self.ai._run_claude_json(system, prompt, max_retries=2)
                out = []
                for iss in (data.get('issues') or []):
                    if not isinstance(iss, dict):
                        continue
                    nid = iss.get('node_id')
                    if nid not in valid:
                        continue
                    sev = 'hard' if iss.get('severity') == 'hard' else 'soft'
                    out.append((name, nid, sev,
                                str(iss.get('claim') or '')[:300],
                                str(iss.get('conflicts_with') or '')[:300]))
                return out

            def prog(done_n, total):
                self.ui_queue.put(
                    lambda d=done_n, n=total: self.status_bar.showMessage(
                        f"Consistency audit… {d}/{n} stories done"))

            results = _fan_out_ai_calls(jobs, one, prog)
            issues = [iss for r in results if r for iss in r]

            def done():
                self._audit_running = False
                self._show_audit_results(issues)
            self.ui_queue.put(done)

        threading.Thread(target=run, daemon=True).start()

    def _show_audit_results(self, issues: list):
        if not issues:
            self.status_bar.showMessage(
                "Consistency audit: no canon conflicts found.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Consistency Audit — {len(issues)} issue(s)")
        dlg.setWindowFlags(dlg.windowFlags() | Qt.Window)
        dlg.resize(780, 540)
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel(
            "Double-click an issue to select its node on the graph. The audit "
            "never edits anything — fix by hand or with AI Rewrite."))
        lst = QListWidget()
        for name, nid, sev, claim, conflict in sorted(
                issues, key=lambda x: (x[2] != 'hard', x[0])):
            item = QListWidgetItem(
                f"[{sev.upper()}] {nid}   ({name})\n"
                f"    claim: {claim}\n"
                f"    conflicts with: {conflict}")
            item.setData(Qt.ItemDataRole.UserRole, nid)
            item.setForeground(QColor('#ff7060' if sev == 'hard' else '#e0b050'))
            lst.addItem(item)
        lst.itemDoubleClicked.connect(
            lambda item: self._select_node(item.data(Qt.ItemDataRole.UserRole)))
        v.addWidget(lst, stretch=1)
        row = QHBoxLayout()
        row.addStretch(1)
        btn = QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        row.addWidget(btn)
        v.addLayout(row)
        dlg.show()   # modeless so double-click node selection stays usable
        self._audit_dlg = dlg

    def _detect_cycles(self) -> set:
        """Return the set of node IDs that participate in at least one cycle (DFS)."""
        nodes = self.script.nodes
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in nodes}
        in_cycle: set = set()

        def dfs(start: str):
            stack = [(start, iter(nodes.get(start, {}).get('next', [])))]
            path: list = [start]
            color[start] = GRAY
            while stack:
                nid, children = stack[-1]
                try:
                    child = next(children)
                    if child not in color:
                        continue  # dangling reference — skip
                    if color[child] == GRAY:
                        # Back-edge found — mark the cycle portion of path
                        idx = path.index(child)
                        in_cycle.update(path[idx:])
                    elif color[child] == WHITE:
                        color[child] = GRAY
                        path.append(child)
                        stack.append((child, iter(nodes.get(child, {}).get('next', []))))
                except StopIteration:
                    color[nid] = BLACK
                    stack.pop()
                    if path and path[-1] == nid:
                        path.pop()

        for nid in list(nodes):
            if color[nid] == WHITE:
                dfs(nid)
        return in_cycle

    def _refresh_cycle_markers(self):
        """Re-detect cycles, update borders, and show a status warning if any found."""
        self._cycle_nodes = self._detect_cycles()
        for nid, node in self._node_items.items():
            nd = self.script.nodes.get(nid, {})
            self._set_node_color(node, nd, in_cycle=nid in self._cycle_nodes)
        if self._cycle_nodes:
            names = ', '.join(sorted(self._cycle_nodes))
            self.status_bar.showMessage(
                f"⚠ Cycle detected involving: {names}", 8000
            )

    def _apply_frequency_heat(self):
        """Apply opacity heat map from cached _freq_counts."""
        if not self._freq_counts:
            return
        max_count = max(self._freq_counts.values()) or 1
        for nid, n in self._node_items.items():
            c = self._freq_counts.get(nid, 0)
            opacity = max(0.04, c / max_count)
            n.view.setOpacity(opacity)
        for pipe, from_nid, to_nid in self._pipe_connections():
            c = (self._freq_counts.get(from_nid, 0) + self._freq_counts.get(to_nid, 0)) / 2
            pipe.setOpacity(max(0.03, c / max_count))

    def _cmd_search_scope_toggled(self, checked: bool):
        """The Text-only toggle changed: re-run the current search under
        the new scope and tell the properties panel so its field
        highlights track what is actually being matched."""
        self.props_panel.set_search_scope(checked)
        self._search_bar.setPlaceholderText(
            "Search node text only…  (Ctrl+/)" if checked
            else "Search nodes by text, tags, arc beat, ID…  (Ctrl+/)")
        self._cmd_search(self._search_bar.text())

    def _cmd_search_word_toggled(self, checked: bool):
        """The whole-Word toggle changed: re-run the current search and
        re-render the properties panel's highlights under the new rule."""
        self.props_panel.set_search_word(checked)
        self._cmd_search(self._search_bar.text())

    def _cmd_search(self, text: str):
        """Apply crosshatch overlays to nodes matching the search term, and
        highlight occurrences of the term inside the selected node's text view.
        Does not change node opacity — selection highlight is fully independent.
        The Text-only toggle narrows matching to the node text body; the Word
        toggle requires whole-word matches (via _find_term_spans, the shared
        matcher the highlight renderers also use)."""
        # Push the (possibly empty) term to the properties panel so in-node
        # highlights track the search bar even when the term is cleared.
        self.props_panel.set_search_term(text)
        if not text.strip():
            self._clear_search_overlays()
            self._update_apply_trigger_btn()
            return
        term = text.strip()
        text_only = self._text_only_btn.isChecked()
        whole_word = self._word_btn.isChecked()
        matched = set()
        for nid, nd in self.script.nodes.items():
            if text_only:
                haystack = nd.get('text', '') or ''
            else:
                haystack = ' '.join([
                    nid,
                    nd.get('label', '') or '',
                    nd.get('text', '') or '',
                    ' '.join(nd.get('tags', [])),
                    nd.get('arc_beat', '') or '',
                ])
            if _find_term_spans(haystack, term, whole_word):
                matched.add(nid)
        self._apply_search_overlays(matched)
        scope = "text-only" if text_only else "all fields"
        if whole_word:
            scope += ", whole word"
        self.status_bar.showMessage(f"Search ({scope}): {len(matched)} matching node(s)")
        self._update_apply_trigger_btn()

    def _update_apply_trigger_btn(self):
        """Sync the Trigger… button enabled-state with the current
        search-overlay set and whether the script has any trigger-state
        options."""
        n_matched = len(self._search_overlays)
        has_options = bool(self.script.trigger_state_options)
        self._apply_trigger_btn.setEnabled(n_matched > 0 and has_options)
        if n_matched > 0 and has_options:
            self._apply_trigger_btn.setText(f"Trigger ({n_matched})…")
        else:
            self._apply_trigger_btn.setText("Trigger…")

    def _cmd_bulk_set_trigger_state(self):
        """Open a dialog letting the user assign trigger_state to all
        nodes currently matching the search overlay. '(none)' clears
        the field entirely on those nodes."""
        matched = sorted(self._search_overlays.keys())
        if not matched:
            self.status_bar.showMessage("No search matches — nothing to do.")
            return

        options = self.script.trigger_state_options
        if not options:
            QMessageBox.information(
                self, "No Weather Set",
                "This script has no associated weather set, so there are no "
                "trigger-state options to choose from. Set the weather set "
                "first via the Story menu.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Bulk-set trigger_state")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel(
            f"Apply a trigger_state to all <b>{len(matched)}</b> nodes "
            f"matching the current search.<br>"
            f"Pick <i>(none)</i> to clear the field on those nodes."))
        combo = QComboBox()
        combo.addItem("(none)", "")
        for sv in options:
            combo.addItem(sv, sv)
        v.addWidget(combo)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        ok_btn = QPushButton("Apply")
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        v.addLayout(btn_row)
        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        ok_btn.setDefault(True)

        if dlg.exec() != QDialog.Accepted:
            return

        val = combo.currentData() or ""
        changed = 0
        for nid in matched:
            nd = self.script.nodes.get(nid)
            if nd is None:
                continue
            if val:
                if nd.get('trigger_state') != val:
                    nd['trigger_state'] = val
                    changed += 1
            else:
                if 'trigger_state' in nd:
                    nd.pop('trigger_state', None)
                    changed += 1

        if changed:
            self.script.dirty = True
            self._update_title()
            # Re-load the props panel if the currently selected node was
            # one of the ones we just changed (so its dropdown updates).
            if self._selected_node_id and self._selected_node_id in matched:
                self.props_panel.load_node(self.script, self._selected_node_id)
            label = val if val else "(none)"
            self.status_bar.showMessage(
                f"Set trigger_state '{label}' on {changed} node(s).")
        else:
            self.status_bar.showMessage(
                "No changes — matched nodes already had that trigger_state.")

    def _get_upstream_path(self, node_id: str, depth: int = 4) -> list:
        """Return [(ancestor_id, text), ...] oldest-first, up to `depth` hops."""
        reverse = {}
        for nid, nd in self.script.nodes.items():
            for child in nd.get('next', []):
                reverse.setdefault(child, []).append(nid)
        chain = []
        current = node_id
        for _ in range(depth):
            parents = reverse.get(current, [])
            if not parents:
                break
            current = parents[0]
            nd = self.script.nodes.get(current, {})
            chain.append((current, nd.get('text', '')))
        chain.reverse()
        return chain

    def _cmd_apply_layout(self, layout_fn):
        positions = layout_fn(self.script)
        for node_id, (x, y) in positions.items():
            if node_id in self._node_items:
                self._node_items[node_id].set_pos(float(x), float(y))
                self.script.update_pos(node_id, [x, y])
        self.graph.fit_to_selection()

    def _cmd_fit_view(self):
        layout = _layout_tree(self.script)
        for node_id, (x, y) in layout.items():
            if node_id in self._node_items:
                self._node_items[node_id].set_pos(float(x), float(y))
                self.script.update_pos(node_id, [x, y])
        self.graph.fit_to_selection()

    def _cmd_apply_tree_layout(self):
        """Rearrange nodes into tree layout and zoom to fit."""
        layout = _layout_tree(self.script)
        for node_id, (x, y) in layout.items():
            if node_id in self._node_items:
                self._node_items[node_id].set_pos(float(x), float(y))
                self.script.update_pos(node_id, [x, y])
        self.graph.fit_to_selection()
        self.status_bar.showMessage("Tree layout applied")

    def _cmd_spread(self):
        self._scale_positions(1.3)

    def _cmd_compact(self):
        self._scale_positions(1.0 / 1.3)

    def _scale_positions(self, factor: float):
        items = [(nid, node) for nid, node in self._node_items.items()]
        if not items:
            return
        positions = [node.pos() for _, node in items]
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        for (nid, node), (px, py) in zip(items, positions):
            new_x = cx + (px - cx) * factor
            new_y = cy + (py - cy) * factor
            node.set_pos(float(new_x), float(new_y))
            self.script.update_pos(nid, [new_x, new_y])

    def _cmd_new(self):
        if self.script.dirty:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Discard them?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        self.script = ScriptData()
        self._selected_node_id = None
        self._refresh_contexts()
        self._rebuild_graph()
        self.props_panel.clear()
        self._update_title()
        self.status_bar.showMessage("New script")

    def _cmd_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Script", str(SOUNDS_DIR), "JSON files (*.json)"
        )
        if path:
            self._load_script(Path(path))

    def _load_recent_files(self) -> list:
        try:
            if RECENTS_PATH.exists():
                data = json.loads(RECENTS_PATH.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return [str(p) for p in data if isinstance(p, str)]
        except Exception:
            pass
        return []

    def _save_recent_files(self):
        try:
            RECENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            RECENTS_PATH.write_text(
                json.dumps(self._recent_files, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _add_recent_file(self, path: Path):
        s = str(Path(path).resolve())
        # Move-to-front semantics, dedup case-insensitive on Windows
        norm = s.lower() if sys.platform == 'win32' else s
        self._recent_files = [
            p for p in self._recent_files
            if (p.lower() if sys.platform == 'win32' else p) != norm
        ]
        self._recent_files.insert(0, s)
        self._recent_files = self._recent_files[:RECENTS_MAX]
        self._save_recent_files()
        self._rebuild_recent_menu()

    @staticmethod
    def _label_for_recent(path_str: str) -> str:
        """Prefer the script's internal name; fall back to parent dir, then filename."""
        p = Path(path_str)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            name = (data.get("name") or "").strip()
            if name and name.lower() != "new script":
                parent = p.parent.name
                return f"{name}  ({parent})" if parent else name
        except Exception:
            pass
        return p.parent.name or p.name

    def _rebuild_recent_menu(self):
        if self._recent_menu is None:
            return
        self._recent_menu.clear()
        live = [p for p in self._recent_files if Path(p).exists()]
        if live != self._recent_files:
            self._recent_files = live
            self._save_recent_files()
        if not self._recent_files:
            empty = QAction("(empty)", self)
            empty.setEnabled(False)
            self._recent_menu.addAction(empty)
            return
        for i, p in enumerate(self._recent_files, 1):
            label = self._label_for_recent(p)
            act = QAction(f"{i}. {label}", self)
            act.setToolTip(p)
            act.triggered.connect(lambda _checked=False, path=p: self._load_script(Path(path)))
            self._recent_menu.addAction(act)
        self._recent_menu.addSeparator()
        clear = QAction("Clear Recent", self)
        clear.triggered.connect(self._cmd_clear_recent)
        self._recent_menu.addAction(clear)

    def _cmd_clear_recent(self):
        self._recent_files = []
        self._save_recent_files()
        self._rebuild_recent_menu()

    def _autosave_path_for(self, script_path: Path) -> Path:
        """Sidecar path used for crash-recovery snapshots."""
        return script_path.with_suffix(script_path.suffix + AUTOSAVE_SUFFIX)

    def _clear_autosave_sidecar(self):
        """Remove the .autosave.json sidecar once a real save has succeeded."""
        if self.script.path:
            sidecar = self._autosave_path_for(self.script.path)
            try:
                if sidecar.exists():
                    sidecar.unlink()
            except OSError as exc:
                logging.getLogger("narrative_editor").warning(
                    "could not remove autosave sidecar %s: %s", sidecar, exc)

    def _autosave(self):
        """Silently snapshot to a .autosave.json sidecar if dirty.

        Writes to <script>.autosave.json instead of the real file so a
        post-crash autosave can never clobber the user's last good save.
        Uses an atomic rename (write to .tmp, then os.replace) so a crash
        mid-write can't leave a half-written sidecar.
        """
        if not (self.script.dirty and self.script.path):
            return
        try:
            self._sync_positions()
            sidecar = self._autosave_path_for(self.script.path)
            tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.script._data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, sidecar)
            self._update_title()
        except Exception as exc:
            logging.getLogger("narrative_editor").warning("autosave failed: %s", exc)

    def _cmd_save(self):
        self._sync_positions()
        # Flush the Stories tab BEFORE saving: a legacy arc materializes
        # its structure/cast fields on first flush, and that must land in
        # this save — otherwise the close-time flush re-dirties the script
        # right after Ctrl+S and quit prompts about "unsaved changes".
        self.stories_panel._save_current()
        if self.script.path:
            try:
                self.script.save()
                self._clear_autosave_sidecar()
                self._update_title()
                self.status_bar.showMessage(f"Saved: {self.script.path.name}")
            except Exception as exc:
                self.status_bar.showMessage(f"Save error: {exc}")
        else:
            self._cmd_save_as()

    def _cmd_save_as(self):
        self._sync_positions()
        self.stories_panel._save_current()   # see _cmd_save
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Script", str(SOUNDS_DIR), "JSON files (*.json)"
        )
        if path:
            try:
                self.script.save(Path(path))
                self._clear_autosave_sidecar()
                self._update_title()
                self.status_bar.showMessage(f"Saved: {Path(path).name}")
                self._add_recent_file(Path(path))
            except Exception as exc:
                self.status_bar.showMessage(f"Save error: {exc}")

    def _load_script(self, path: Path):
        # Crash-recovery: if a sidecar exists and is newer than the real
        # file, the editor probably died between an autosave and a real
        # save. Offer to load the sidecar instead.
        sidecar = self._autosave_path_for(path)
        load_from = path
        try:
            if (sidecar.exists() and path.exists()
                    and sidecar.stat().st_mtime > path.stat().st_mtime):
                age_s = sidecar.stat().st_mtime - path.stat().st_mtime
                reply = QMessageBox.question(
                    self, "Recover Autosave?",
                    f"An autosave for '{path.name}' was found that is "
                    f"{age_s:.0f}s newer than the saved file — the editor "
                    f"may have crashed before you could save.\n\n"
                    f"Load the autosave?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    load_from = sidecar
        except OSError:
            pass
        try:
            self.script = ScriptData.load(load_from)
            # Always keep the canonical path pointing at the real file, even
            # if we loaded data from the sidecar.
            self.script.path = path
            if load_from is sidecar:
                self.script.dirty = True  # so the user is nudged to save
            self._selected_node_id = None
            self._refresh_contexts()
            for nid, pos in _layout_tree(self.script).items():
                self.script.update_pos(nid, pos)
            self._rebuild_graph()
            self.props_panel.clear()
            from PySide6.QtCore import QTimer
            def _fit():
                nodes = self.graph.all_nodes()
                if nodes:
                    self.graph.viewer().zoom_to_nodes([n.view for n in nodes])
            QTimer.singleShot(0, _fit)
            self._update_title()
            self.status_bar.showMessage(f"Loaded: {path.name}")
            self._add_recent_file(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))

    def _refresh_contexts(self):
        """Re-wire all panels after script is replaced."""
        self.props_panel.set_context(self.script, self.vm, self.ai, self.ui_queue)
        self.props_panel.refresh_variable_widgets()
        self.voice_panel.set_context(self.script, self.vm, self.ui_queue, self.props_panel)
        self.chat_panel.set_context(self.script, self.ai, self.ui_queue,
                                    self._on_graph_generated,
                                    on_nodes_incremental=self._add_nodes_incremental,
                                    register_orchestrator=self._register_orchestrator)
        self.play_panel.set_context(self.script, self.ui_queue)
        # Re-sync the Node Length / Width submenus to the newly-loaded script
        self._refresh_node_length_checks()
        self._refresh_width_checks()
        # The trigger-state options come from the script's weather set,
        # so the bulk-apply button's enabled-state may change.
        self._update_apply_trigger_btn()
        # Re-point the section tabs at the new ScriptData.
        self.codex_panel.set_script(self.script)
        self.stories_panel.set_script(self.script)
        self.web_panel.set_script(self.script)
        self.canon_panel.set_script(self.script)

    def _update_title(self):
        if self.script.path:
            name = self.script.path.name
        else:
            name = self.script.name or "New Script"
        dirty = "*" if self.script.dirty else ""
        self.setWindowTitle(f"Narrative Editor v2 — {name}{dirty}")

        nodes = self.script.nodes
        n = len(nodes)
        self._stat_nodes.setText(f"Nodes: {n}")

        total_words = sum(len(nd.get("text", "").split()) for nd in nodes.values())
        total_s = (total_words / 121) * 60  # ~121 wpm measured from bartiki audio
        if total_s < 60:
            dur_str = f"{total_s:.0f}s"
        elif total_s < 3600:
            dur_str = f"{int(total_s // 60)}m {int(total_s % 60)}s"
        else:
            dur_str = f"{int(total_s // 3600)}h {int((total_s % 3600) // 60)}m"
        self._stat_duration.setText(f"~{dur_str} speech")

    def closeEvent(self, event):
        # Flush any in-progress story-card edits before the dirty check
        # (the Stories tab autosaves per keystroke, but be certain).
        self.stories_panel._save_current()
        if self.script.dirty:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                self._cmd_save()
                event.accept()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
                return
        self._drain_timer.stop()
        # Stop playback
        self.play_panel._cmd_stop()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def _install_diagnostics():
    """Wire up crash logging so 'hitch then crash' leaves a useful trail.

    - faulthandler: native-level traceback for segfaults / aborts.
    - logging: rotating-ish app log + crash log (both append).
    - sys.excepthook: capture Python exceptions Qt would otherwise swallow.
    - threading.excepthook: same, for worker threads.
    - qInstallMessageHandler: capture Qt's own warnings (often print just
      before a native crash and otherwise go to a stderr nobody reads).
    Returns the open crash-log file (kept alive for faulthandler).
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # faulthandler writes a C-level traceback on segfault/abort/SIGFPE/etc.
    # We keep the file handle open for the lifetime of the process.
    crash_fp = open(CRASH_LOG_PATH, "a", buffering=1, encoding="utf-8")
    crash_fp.write(
        f"\n===== narrative_editor_v2 session start {_datetime.datetime.now().isoformat()} "
        f"pid={os.getpid()} =====\n"
    )
    faulthandler.enable(file=crash_fp, all_threads=True)

    # App log: timestamps + level for everything the editor logs explicitly.
    logging.basicConfig(
        filename=str(APP_LOG_PATH),
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("narrative_editor")
    log.info("session start pid=%s argv=%s", os.getpid(), sys.argv)

    # Surface unhandled Python exceptions (Qt slots sometimes swallow these).
    def _excepthook(exc_type, exc_value, tb):
        msg = "".join(traceback.format_exception(exc_type, exc_value, tb))
        log.error("UNHANDLED EXCEPTION:\n%s", msg)
        crash_fp.write(
            f"--- python exception {_datetime.datetime.now().isoformat()} ---\n{msg}\n"
        )
        crash_fp.flush()
        # Still print to stderr if someone is watching.
        sys.__excepthook__(exc_type, exc_value, tb)
    sys.excepthook = _excepthook

    # threading.excepthook is 3.8+; guard just in case.
    if hasattr(threading, "excepthook"):
        def _thread_excepthook(args):
            msg = "".join(traceback.format_exception(
                args.exc_type, args.exc_value, args.exc_traceback))
            log.error("THREAD EXCEPTION in %s:\n%s",
                      getattr(args.thread, "name", "?"), msg)
            crash_fp.write(
                f"--- thread exception {_datetime.datetime.now().isoformat()} "
                f"thread={getattr(args.thread, 'name', '?')} ---\n{msg}\n"
            )
            crash_fp.flush()
        threading.excepthook = _thread_excepthook

    # Qt's own message stream — warnings here often immediately precede a
    # native crash ("QObject::disconnect: ...", "QPainter::end: ...").
    try:
        from PySide6.QtCore import qInstallMessageHandler, QtMsgType
        _qt_level = {
            QtMsgType.QtDebugMsg:    logging.DEBUG,
            QtMsgType.QtInfoMsg:     logging.INFO,
            QtMsgType.QtWarningMsg:  logging.WARNING,
            QtMsgType.QtCriticalMsg: logging.ERROR,
            QtMsgType.QtFatalMsg:    logging.CRITICAL,
        }
        def _qt_handler(msg_type, ctx, message):
            lvl = _qt_level.get(msg_type, logging.INFO)
            where = ""
            if ctx and ctx.file:
                where = f" ({ctx.file}:{ctx.line})"
            log.log(lvl, "Qt: %s%s", message, where)
            if lvl >= logging.WARNING:
                crash_fp.write(
                    f"[qt {_datetime.datetime.now().isoformat()}] {message}{where}\n"
                )
                crash_fp.flush()
        qInstallMessageHandler(_qt_handler)
    except Exception as exc:
        log.warning("could not install Qt message handler: %s", exc)

    return crash_fp


def main():
    # Must happen before QApplication construction so we catch crashes
    # during startup too.
    _crash_fp = _install_diagnostics()  # kept alive for faulthandler

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(45, 45, 55))
    palette.setColor(QPalette.WindowText,      Qt.white)
    palette.setColor(QPalette.Base,            QColor(35, 35, 45))
    palette.setColor(QPalette.AlternateBase,   QColor(50, 50, 65))
    palette.setColor(QPalette.ToolTipBase,     QColor(50, 50, 65))
    palette.setColor(QPalette.ToolTipText,     Qt.white)
    palette.setColor(QPalette.Text,            Qt.white)
    palette.setColor(QPalette.Button,          QColor(55, 55, 70))
    palette.setColor(QPalette.ButtonText,      Qt.white)
    palette.setColor(QPalette.BrightText,      Qt.red)
    palette.setColor(QPalette.Highlight,       QColor(80, 120, 200))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(palette)

    path = sys.argv[1] if len(sys.argv) > 1 else None
    win  = MainWindow(path)
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
