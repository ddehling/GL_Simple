#!/usr/bin/env python3
"""
Narrative Script Editor (PySide6 + NodeGraphQt)

Visual node-graph editor for building narrative audio scripts.
Uses PyQt5 + NodeGraphQt for the interface, Claude Code CLI for AI-assisted generation.

Usage:
    python tools/narrative_editor_qt.py
    python tools/narrative_editor_qt.py media/sounds/my_story/script.json

Requirements (beyond base project):
    pip install PySide6 NodeGraphQt qtpy
"""

import json
import os
import queue
import random
import re
import subprocess
import sys
import textwrap
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
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
    QFrame, QGraphicsItem, QGraphicsRectItem, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QSplitter,
    QStatusBar, QTextEdit, QVBoxLayout, QWidget,
)

from NodeGraphQt import NodeGraph, BaseNode

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT  = Path(__file__).parent.parent
SOUNDS_DIR = REPO_ROOT / "media" / "sounds"

NODE_PREVIEW_LEN = 60        # chars shown inside a node box
FOCUSED_CONTEXT_MAX = 1500  # max chars of story_context_focused sent to AI

PARALLEL_WORKER_COUNT = 8   # concurrent AI calls for parallel generation

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
        'max_depth': 10,
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
        'max_depth': 10,
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
    "story_context_focused": "",
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
"""

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
  "vars": {}
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
"""

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

IMPORTANT — when a "Story context" block is provided, treat it as BACKGROUND ATMOSPHERE ONLY
(voice, tone, setting). It must NEVER override the subject of the user's actual message.
The user's message defines the topic. Always stay on that topic.
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
"""

# ─────────────────────────────────────────────────────────────────────────────
# Data Model
# ─────────────────────────────────────────────────────────────────────────────

class ScriptData:
    """In-memory representation of a script.json file."""

    def __init__(self, data: dict = None):
        self._data = deepcopy(data or SCRIPT_TEMPLATE)
        self.path: Optional[Path] = None
        self.dirty = False
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
    def story_context_focused(self) -> str: return self._data.get("story_context_focused", "")

    def set_story_context(self, text: str):
        self._data["story_context"] = text
        self.dirty = True

    def set_story_context_focused(self, text: str):
        self._data["story_context_focused"] = text
        self.dirty = True

    @property
    def variables(self) -> list:
        """Story-level variable definitions: [{"name": ..., "description": ...}, ...]"""
        return self._data.setdefault("variables", [])

    def set_variables(self, var_list: list):
        """Replace all variable definitions (max 4)."""
        self._data["variables"] = list(var_list)[:4]
        self.dirty = True

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
        if arc_id in self.arcs:
            self.arcs[arc_id].update(data)
            self.dirty = True

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

    @classmethod
    def load(cls, path: Path) -> "ScriptData":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        sd = cls(data)
        sd.path = Path(path)
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
    layer_name:      str   = ''           # target layer for this node
    layer_direction: str   = ''           # arc beat guidance
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
                 story_context: str = '', motif: str = '',
                 premise: str = '', themes: str = '',
                 arc_beats: dict = None, variables: list = None,
                 on_progress=None, on_complete=None, on_node_added=None):
        self._script       = script
        self._ui_queue     = ui_queue
        self._profile      = GENERATION_PROFILES.get(profile, GENERATION_PROFILES['full'])
        self._story_context = story_context
        self._motif        = motif
        self._premise      = premise
        self._themes       = themes
        self._arc_beats    = arc_beats or {}
        self._variables    = variables or []
        self._on_progress  = on_progress   # callback(status_str)
        self._on_complete  = on_complete   # callback()
        self._on_node_added = on_node_added  # callback(set_of_new_ids)

        # Worker pool
        self._workers = [AIAssistant(model=model) for _ in range(PARALLEL_WORKER_COUNT)]
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

        # Existing tags for reuse hints
        layer_tags = set(LAYER_ORDER)
        self._existing_tags = list({
            t for n in script.nodes.values()
            for t in n.get('tags', []) if t not in layer_tags
        })

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

        # Determine child layer from the deepest parent
        deepest_idx = 0
        for nid in valid:
            tags = self._script.nodes[nid].get('tags', [])
            layer = next((t for t in tags if t in LAYER_ORDER), 'arrival')
            idx = LAYER_ORDER.index(layer) if layer in LAYER_ORDER else 0
            deepest_idx = max(deepest_idx, idx)

        next_idx = min(deepest_idx + 1, len(LAYER_ORDER) - 1)
        child_layer = LAYER_ORDER[next_idx]
        if child_layer == LAYER_ORDER[deepest_idx] and deepest_idx == len(LAYER_ORDER) - 1:
            print(f'[Parallel] All parents at stillness — nothing to generate')
            self._ui_queue.put(lambda: self._on_complete() if self._on_complete else None)
            return

        # Use first parent as root for branch tracking
        root_id = valid[0]
        for nid in valid:
            self._node_to_root[nid] = root_id

        direction = self._arc_beats.get(child_layer, '')
        tid = self._next_task_id()
        task = NodeTask(
            task_id=tid,
            parent_id=valid[0],
            parent_ids=list(valid),
            root_id=root_id,
            layer_name=child_layer,
            layer_direction=direction,
            batch_size=batch_size,
        )
        self._tasks[tid] = task
        self._total_created += batch_size
        print(f'[Parallel] Merged batch: {batch_size}× {child_layer} from parents {valid}')

        threading.Thread(target=self._coordinator_loop, daemon=True).start()

    def start(self, seed_node_ids: list):
        """Begin parallel generation from a list of existing seed (arrival) nodes."""
        print(f'[Parallel] Starting with {len(seed_node_ids)} seed nodes: {seed_node_ids}')
        print(f'[Parallel] Profile: max_depth={self._profile["max_depth"]}, workers={PARALLEL_WORKER_COUNT}')
        print(f'[Parallel] Arc beats: {list(self._arc_beats.keys())}')
        for nid in seed_node_ids:
            nd = self._script.nodes.get(nid)
            if not nd:
                print(f'[Parallel]   {nid}: NOT FOUND in script — skipping')
                continue
            # Each seed is the root of its own branch
            self._node_to_root[nid] = nid
            tags = nd.get('tags', [])
            layer = next((t for t in tags if t in LAYER_ORDER), 'arrival')
            layer_idx = LAYER_ORDER.index(layer) if layer in LAYER_ORDER else 0
            next_layer_idx = min(layer_idx + 1, len(LAYER_ORDER) - 1)
            next_layer = LAYER_ORDER[next_layer_idx]
            print(f'[Parallel]   {nid}: tags={tags}, layer={layer}, children→{next_layer}, branch={nid}')
            if next_layer == layer:
                print(f'[Parallel]   {nid}: already at stillness — no children')
                continue
            self._spawn_children(nid, next_layer, root_id=nid)

        print(f'[Parallel] Initial tasks queued: {self._total_created}')
        # Start coordinator thread
        threading.Thread(target=self._coordinator_loop, daemon=True).start()

    def _next_task_id(self) -> str:
        self._task_counter += 1
        return f'task_{self._task_counter:04d}'

    def _get_width(self, layer_name: str) -> tuple:
        widths = self._profile['widths']
        return widths.get(layer_name, widths.get('*', (2, 3)))

    def _get_layer_cap(self, layer_name: str) -> int:
        caps = self._profile.get('layer_caps', {})
        return caps.get(layer_name, 999)

    def _count_global_layer_nodes(self, layer_name: str) -> int:
        """Count expected nodes at a layer across ALL branches (sum of batch sizes)."""
        return sum(t.batch_size for t in self._tasks.values() if t.layer_name == layer_name)

    def _spawn_children(self, parent_id: str, child_layer: str, root_id: str = ''):
        """Create NodeTasks for children of parent_id at child_layer.

        Uses GLOBAL caps to limit total node count per layer.  When over cap,
        converges into existing same-branch nodes (no forced child creation).
        Cross-branch connections are handled by the cross-link AI pass.
        """
        if not root_id:
            root_id = self._node_to_root.get(parent_id, parent_id)

        lo, hi = self._get_width(child_layer)
        desired = random.randint(lo, hi)
        direction = self._arc_beats.get(child_layer, '')

        with self._lock:
            cap = self._get_layer_cap(child_layer)
            global_count = self._count_global_layer_nodes(child_layer)
            remaining_slots = max(0, cap - global_count)

            n_to_create = min(desired, remaining_slots)

            # Over global cap — converge into existing nodes at this layer
            if n_to_create == 0:
                # Queue a deferred convergence — will be resolved after all
                # tasks at this layer complete
                self._deferred_convergences.append(
                    (parent_id, child_layer, root_id))
                print(f'[Parallel] ↗ Deferred converge: {parent_id} → layer {child_layer} '
                      f'({global_count}/{cap})')
                return  # hard stop — no new nodes when over cap

            print(f'[Parallel] Batch: {n_to_create} children for {parent_id} → '
                  f'layer:{child_layer} branch:{root_id} ({global_count}+{n_to_create}/{cap})')

            tid = self._next_task_id()
            task = NodeTask(
                task_id=tid,
                parent_id=parent_id,
                parent_ids=[parent_id],
                root_id=root_id,
                layer_name=child_layer,
                layer_direction=direction,
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

            # Brief sleep to avoid busy-waiting for new tasks from completions
            time.sleep(0.1)

        # Final: resolve deferred convergences, then cross-link passes
        if not self._cancelled.is_set():
            print(f'[Parallel] All tasks done ({self._total_completed} nodes).')
            self._resolve_deferred_convergences()
            print(f'[Parallel] Running cross-link passes...')
            self._run_cross_link_passes()

        print(f'[Parallel] Generation complete. {self._total_completed} nodes generated.')
        self._ui_queue.put(lambda: self._on_complete() if self._on_complete else None)

    def _resolve_deferred_convergences(self):
        """Wire deferred convergences now that all tasks are complete."""
        if not self._deferred_convergences:
            return
        print(f'[Parallel] Resolving {len(self._deferred_convergences)} deferred convergences...')
        for parent_id, child_layer, root_id in self._deferred_convergences:
            # Find completed nodes at this layer, prefer same branch
            candidates = []
            for t in self._tasks.values():
                if t.layer_name == child_layer and t.final_node_ids:
                    if t.root_id == root_id:
                        candidates.extend(t.final_node_ids)
            if not candidates:
                for t in self._tasks.values():
                    if t.layer_name == child_layer and t.final_node_ids:
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
                print(f'[Parallel]   ⚠ {parent_id}: no nodes at layer {child_layer} to converge into')
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

            # Calculate premise weight — 100% at layer 0, decreasing 30% per layer
            # Layers 0-4 get premise; layer 5+ gets none
            layer_idx = LAYER_ORDER.index(task.layer_name) if task.layer_name in LAYER_ORDER else 0
            premise_weight = max(0.0, 1.0 - 0.3 * layer_idx) if layer_idx < 5 else 0.0

            # Read the parent node's hint (author guidance for expansion)
            parent_hint = primary_nd.get('hint', '').strip()

            parent_label = (f'parents=[{", ".join(all_parent_ids)}]' if len(all_parent_ids) > 1
                           else f'parent={primary_pid}')
            premise_str = f', premise={premise_weight:.0%}' if premise_weight > 0 else ''
            hint_str = f', hint="{parent_hint[:40]}..."' if parent_hint else ''
            print(f'[Parallel] ▶ {task.task_id}: batch {task.batch_size}× {task.layer_name} '
                  f'from {parent_label} (ancestors={len(ancestor_chain)}{premise_str}{hint_str})')

            worker = self._workers[0]

            result = worker.generate_batch_sync(
                parent_id=primary_pid,
                parent_text=parent_text,
                parent_tags=parent_tags,
                ancestor_chain=ancestor_chain,
                layer_name=task.layer_name,
                batch_size=task.batch_size,
                layer_direction=task.layer_direction,
                hint=parent_hint,
                motif=self._motif,
                themes=self._themes,
                story_context=self._story_context,
                existing_custom_tags=self._existing_tags,
                variables=self._variables,
                premise=self._premise,
                premise_weight=premise_weight,
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

            def _apply(t=task, evt=applied_event, pids=all_parent_ids, node_dict=nodes):
                applied_ids = []
                for nid, ndata in node_dict.items():
                    # Build a single-node result dict for apply_single_node
                    single = dict(ndata)
                    single['node_id'] = nid
                    final_id = self._script.apply_single_node(t.parent_id, single)
                    if final_id:
                        applied_ids.append(final_id)
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
                self._completed_by_layer[task.layer_name].append(task)
                for fid in task.final_node_ids:
                    self._node_to_root[fid] = root_id

            max_depth_layers = self._profile['max_depth']
            for final_id in task.final_node_ids:
                nd = self._script.nodes.get(final_id, {})
                result_tags = nd.get('tags', [])
                actual_layer = next((t for t in result_tags if t in LAYER_ORDER), task.layer_name)
                layer_idx = LAYER_ORDER.index(actual_layer) if actual_layer in LAYER_ORDER else 0
                if layer_idx + 1 < len(LAYER_ORDER) and layer_idx + 1 < max_depth_layers:
                    next_layer = LAYER_ORDER[layer_idx + 1]
                    self._spawn_children(final_id, next_layer, root_id=root_id)
                else:
                    print(f'[Parallel]   {final_id} is terminal (layer={actual_layer})')

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
        """After all generation is done, suggest cross-branch connections."""
        if not self._completed_by_layer:
            return
        worker = self._workers[0]

        for layer_name in LAYER_ORDER[:-1]:  # skip stillness
            completed = self._completed_by_layer.get(layer_name, [])
            if len(completed) < 5:
                continue  # not enough nodes to cross-link
            print(f'[Parallel] Cross-linking layer "{layer_name}" ({len(completed)} nodes)...')

            # Build layer_nodes list from all nodes produced by completed batch tasks
            layer_nodes = []
            children_map = {}
            for task in completed:
                for nid in task.final_node_ids:
                    nd = self._script.nodes.get(nid, {})
                    if not nd:
                        continue
                    text = nd.get('text', '')
                    tags = nd.get('tags', [])
                    child_ids = nd.get('next', [])
                    layer_nodes.append((nid, text, tags, child_ids))
                    for cid in child_ids:
                        cnd = self._script.nodes.get(cid, {})
                        children_map[cid] = (cnd.get('text', ''), cnd.get('tags', []))

            if not children_map:
                continue

            try:
                links = worker.suggest_cross_links_sync(layer_nodes, children_map)
                print(f'[Parallel]   Cross-link suggestions for {layer_name}: {len(links)} links')
                for link in links:
                    print(f'[Parallel]     {link.get("from", "?")} → {link.get("to", "?")}')
                if links:
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
            except Exception:
                pass  # cross-linking is best-effort


# ─────────────────────────────────────────────────────────────────────────────
# AI Assistant
# ─────────────────────────────────────────────────────────────────────────────

class AIAssistant:
    """Calls the `claude` CLI via subprocess — uses your Claude Code session,
    no separate API key required."""

    DEFAULT_MODEL = 'claude-sonnet-4-6'

    def __init__(self, model: str = ''):
        self._history: list = []
        self._busy = False
        self._model: str = model or self.DEFAULT_MODEL
        self._claude_exe: Optional[str] = self._find_claude()

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str):
        self._model = value or self.DEFAULT_MODEL

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
        # Show context around where it broke
        preview = raw[start:start+300] if len(raw) > start else raw
        raise ValueError(f"Unbalanced braces in JSON response (depth={depth}, "
                         f"len={len(raw)}, start={start}). Preview: {preview[:200]}")

    def _run_claude(self, system: str, prompt: str, max_retries: int = 5) -> str:
        """Blocking call to `claude -p`. Retries with exponential backoff."""
        cmd = [
            self._claude_exe,
            "--no-session-persistence",
            "--model", self._model,
            "--system-prompt", system,
            "--output-format", "text",
            "-p", prompt,
        ]
        last_error = None
        for attempt in range(max_retries):
            backoff = min(3 * (2 ** attempt), 30)  # 3, 6, 12, 24, 30 seconds
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=360,
                )
                if result.returncode != 0:
                    err = result.stderr.strip() or result.stdout.strip() or "claude CLI returned non-zero"
                    last_error = RuntimeError(err)
                    # Don't retry model/auth errors
                    if 'model' in err.lower() or 'auth' in err.lower() or 'access' in err.lower():
                        raise last_error
                    print(f'[AI] Attempt {attempt+1}/{max_retries} failed: {err[:80]} (retry in {backoff}s)')
                    time.sleep(backoff)
                    continue
                out = result.stdout.strip()
                if not out:
                    last_error = RuntimeError(
                        f"claude produced no output (stderr: {result.stderr.strip()!r})")
                    print(f'[AI] Attempt {attempt+1}/{max_retries}: empty output (retry in {backoff}s)')
                    time.sleep(backoff)
                    continue
                return out
            except subprocess.TimeoutExpired:
                last_error = RuntimeError("claude CLI timed out (360s)")
                print(f'[AI] Attempt {attempt+1}/{max_retries}: timeout (retry in {backoff}s)')
                time.sleep(backoff)
                continue
        raise last_error or RuntimeError("claude CLI failed after all retries")

    @staticmethod
    def _vars_prompt_section(variables: list) -> str:
        """Build a prompt section telling the AI to set story variables on every node."""
        if not variables:
            return ''
        lines = ['STORY VARIABLES — set "vars" on every node (each value 0.0–1.0):']
        for v in variables:
            lines.append(f'  "{v["name"]}": {v["description"]}')
        lines.append('CRITICAL: Use the FULL range. Default is 0.0 — use 0.0 when the variable\'s '
                      'quality is NOT meaningfully present in the node text. Only use values above 0 '
                      'when the text explicitly contains elements described by the variable. '
                      'Most nodes should be 0.0 for most variables. '
                      'Reserve 0.8–1.0 for nodes where that quality is the dominant force. '
                      'Avoid clustering everything in the 0.2–0.6 range — be decisive.')
        lines.append('Include "vars": {"name": value, ...} in every node object.')
        return '\n'.join(lines)

    def chat(self, user_msg: str, ui_queue: queue.SimpleQueue,
             on_reply, on_error, script_summary: str = '', story_context: str = '',
             _system_override: str = ''):
        if self._busy:
            return
        self._busy = True
        self._history.append({"role": "user", "content": user_msg})

        parts = []
        if story_context:
            parts.append(f"BACKGROUND ATMOSPHERE (tone/voice/setting only — do not let this override the topic of the user's message):\n{story_context}")
        if script_summary:
            parts.append(f"Current script:\n{script_summary}")
        transcript = self._transcript()
        if transcript:
            parts.append(transcript)
        parts.append(f"User: {user_msg}")
        full_prompt = "\n\n".join(parts)
        system = _system_override or SYSTEM_CHAT

        def run():
            try:
                reply = self._run_claude(system, full_prompt)
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
                       variables: list = None):
        if self._busy:
            return
        self._busy = True

        parts = []
        if story_context:
            parts.append(f'BACKGROUND ATMOSPHERE (tone/voice/setting only — do not let this dilute or override the subject below):\n{story_context}')
        vars_sec = self._vars_prompt_section(variables or [])
        if vars_sec:
            parts.append(vars_sec)
        parts.append(f'SUBJECT (this is what the script must be about — prioritize above all else):\n{prompt}')
        full_prompt = '\n\n'.join(parts)

        def run():
            try:
                raw   = self._run_claude(SYSTEM_GENERATE, full_prompt)
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
                      variables: list = None):
        """Generate only the arrival layer — no children. Used to seed iterative generation."""
        if self._busy:
            return
        self._busy = True

        parts = []
        if story_context:
            parts.append(f'BACKGROUND ATMOSPHERE (tone/voice/setting only — do not let this dilute or override the subject below):\n{story_context}')
        vars_sec = self._vars_prompt_section(variables or [])
        if vars_sec:
            parts.append(vars_sec)
        if layer_direction:
            parts.append(f'ARRIVAL LAYER DIRECTION (this is what the arrival nodes must cover):\n{layer_direction}')
        if motif:
            parts.append(f'RECURRING MOTIF (weave this through the text naturally in every node):\n{motif}')
        parts.append(f'SUBJECT (this is what the script must be about — prioritize above all else):\n{prompt}')
        full_prompt = '\n\n'.join(parts)

        def run():
            try:
                print(f'[AI] generate_seed: calling claude...')
                raw   = self._run_claude(SYSTEM_GENERATE_SEED, full_prompt)
                print(f'[AI] generate_seed: got {len(raw)} chars, extracting JSON...')
                data = self._extract_json(raw)
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
                       variables: list = None):
        """Generate the next layer for all frontier nodes in one AI call.

        frontier: list of (node_id, node_data) for all current leaf nodes.
        """
        if self._busy:
            return
        self._busy = True

        parts = []
        if story_context:
            sc = story_context[:FOCUSED_CONTEXT_MAX] + '...' if len(story_context) > FOCUSED_CONTEXT_MAX else story_context
            parts.append(f'BACKGROUND ATMOSPHERE (tone/voice/setting only — do NOT let this dilute or override the topic defined by the source nodes):\n  {sc}')
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

        def run():
            try:
                raw   = self._run_claude(SYSTEM_GENERATE_LAYER, full_prompt)
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
                    variables: list = None):
        if self._busy:
            return
        self._busy = True

        # Prompt is ordered lowest→highest weight so the most important content
        # is closest to the generation point (recency bias).
        parts = []

        # ── Background: story context (lowest weight, read first / least recent) ──
        if story_context:
            sc = story_context[:1000] + '...' if len(story_context) > 1000 else story_context
            parts.append(f'BACKGROUND (story flavour only):\n  {sc}')

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

        def run():
            try:
                raw   = self._run_claude(SYSTEM_EXPAND, prompt)
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
                                   premise_weight: float = 1.0) -> dict:
        """Blocking call that generates exactly one node. Returns parsed dict.

        Designed to be called from worker threads in ParallelNodeOrchestrator.
        Does NOT use _busy flag — caller is responsible for concurrency control.
        premise_weight: 0.0–1.0 controls how strongly the premise influences this node.
        """
        parts = []

        # Premise — fades over layers via premise_weight
        if premise and premise_weight > 0.05:
            weight_pct = int(premise_weight * 100)
            if premise_weight > 0.8:
                label = "STORY PREMISE (this is the core vision — stay true to it)"
            elif premise_weight > 0.5:
                label = "STORY PREMISE (keep this present as an undercurrent)"
            elif premise_weight > 0.3:
                label = "STORY PREMISE (a distant echo — let it inform tone, not dictate content)"
            else:
                label = "STORY PREMISE (faint background influence only)"
            parts.append(f'{label} [{weight_pct}% influence]:\n  {premise}')

        # Background (lowest weight)
        if story_context:
            sc = story_context[:FOCUSED_CONTEXT_MAX] + '...' \
                if len(story_context) > FOCUSED_CONTEXT_MAX else story_context
            parts.append(f'BACKGROUND (story flavour only):\n  {sc}')

        if existing_custom_tags:
            parts.append(f'EXISTING TAGS (prefer these): {", ".join(sorted(existing_custom_tags))}')

        vars_sec = self._vars_prompt_section(variables or [])
        if vars_sec:
            parts.append(vars_sec)

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

        # Layer direction / motif
        if layer_direction:
            parts.append(f'\nLAYER DIRECTION: {layer_direction}')
        if motif:
            parts.append(f'RECURRING MOTIF (weave naturally): {motif}')
        if themes:
            parts.append(f'THEMATIC THREADS (let these resonate through the narrative): {themes}')

        # Parent node (highest weight)
        parts.append(
            f'\nPARENT NODE — continue from this:\n'
            f'  ID: {parent_id}\n'
            f'  Tags: {parent_tags}\n'
            f'  Text: "{parent_text}"'
        )

        parts.append(f'\nGenerate exactly 1 node in the "{layer_name}" layer continuing from {parent_id}.')
        prompt = '\n'.join(parts)

        raw = self._run_claude(SYSTEM_GENERATE_SINGLE_NODE, prompt)
        return self._extract_json(raw)

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
                             themes: str = '') -> dict:
        """Blocking call that generates multiple sibling nodes from one parent.

        Returns dict with 'nodes' and 'connect_from' keys (SYSTEM_EXPAND format).
        """
        parts = []

        # Premise — fades over layers
        if premise and premise_weight > 0.05:
            weight_pct = int(premise_weight * 100)
            if premise_weight > 0.8:
                label = "STORY PREMISE (core vision — stay true to it)"
            elif premise_weight > 0.5:
                label = "STORY PREMISE (keep as undercurrent)"
            elif premise_weight > 0.3:
                label = "STORY PREMISE (distant echo — inform tone only)"
            else:
                label = "STORY PREMISE (faint background)"
            parts.append(f'{label} [{weight_pct}% influence]:\n  {premise}')

        if story_context:
            sc = story_context[:FOCUSED_CONTEXT_MAX] + '...' \
                if len(story_context) > FOCUSED_CONTEXT_MAX else story_context
            parts.append(f'BACKGROUND (story flavour only):\n  {sc}')

        if existing_custom_tags:
            parts.append(f'EXISTING TAGS (prefer these): {", ".join(sorted(existing_custom_tags))}')

        vars_sec = self._vars_prompt_section(variables or [])
        if vars_sec:
            parts.append(vars_sec)

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

        if layer_direction:
            parts.append(f'\nLAYER DIRECTION: {layer_direction}')
        if motif:
            parts.append(f'RECURRING MOTIF (weave naturally): {motif}')
        if themes:
            parts.append(f'THEMATIC THREADS (let these resonate through the narrative): {themes}')

        # Author hint (high priority — right before source node)
        # Source node
        parts.append(
            f'\nSOURCE NODE — stay close to this:\n'
            f'  ID: {parent_id}\n'
            f'  Tags: {parent_tags}\n'
            f'  Text: "{parent_text}"'
        )

        parts.append(f'\nGenerate exactly {batch_size} continuation nodes in the '
                     f'"{layer_name}" layer branching from this node.')

        # Author hint LAST — highest recency weight, overrides all other guidance
        if hint:
            parts.append(f'\nCRITICAL — AUTHOR DIRECTION (this overrides thematic continuity): {hint}')
        prompt = '\n'.join(parts)

        raw = self._run_claude(SYSTEM_EXPAND, prompt)
        return self._extract_json(raw)

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
                           variables: list = None):
        if self._busy:
            return
        self._busy = True

        source_layer = next((t for t in source_tags if t in LAYER_ORDER), 'discovery')

        parts = []
        if story_context:
            sc = story_context[:FOCUSED_CONTEXT_MAX] + '...' if len(story_context) > FOCUSED_CONTEXT_MAX else story_context
            parts.append(f'BACKGROUND (story flavour only):\n  {sc}')
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

        def run():
            try:
                raw   = self._run_claude(SYSTEM_CONTINUE, full_prompt)
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
            "- Maximum 800 characters\n"
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
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("Node name...")
        self.label_edit.textChanged.connect(self._autosave_label)
        label_row.addWidget(self.label_edit)
        layout.addLayout(label_row)

        # Text
        layout.addWidget(QLabel("Text:"))
        self.text_edit = QTextEdit()
        self.text_edit.setMinimumHeight(240)
        self.text_edit.setMaximumHeight(450)
        self.text_edit.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self.text_edit.textChanged.connect(self._autosave_text)
        self.text_edit.textChanged.connect(self._update_word_count)
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
        self.tags_edit = QLineEdit()
        self.tags_edit.setPlaceholderText("goat, toad, revelation, ...")
        self.tags_edit.setToolTip("Custom tags: characters, themes, locations — comma separated")
        self.tags_edit.textChanged.connect(self._autosave_tags)
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

        # Is start
        self.is_start_cb = QCheckBox("Start node")
        self.is_start_cb.stateChanged.connect(self._autosave_start)
        layout.addWidget(self.is_start_cb)

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

    def load_node(self, script: ScriptData, node_id: str):
        self._script = script
        self._node_id = node_id
        nd = script.nodes.get(node_id, {})

        self._blocking = True
        try:
            self.id_edit.setText(node_id)
            self.label_edit.setText(nd.get("label") or node_id)
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
            if arc_beat:
                self._arc_beat_lbl.setText(
                    f'<span style="color:#7799cc; font-size:10px;">'
                    f'<b>Arc beat:</b> {arc_beat}</span>')
            else:
                self._arc_beat_lbl.setText('')

            self.rebuild_edge_list(script, node_id)

            # Story variables
            node_vars = nd.get("vars", {})
            for name, spin in self._vars_spins.items():
                spin.setValue(node_vars.get(name, 0.0))
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
            self.text_edit.setPlainText("")
            self.hint_edit.setPlainText("")
            self.rewrite_hint.clear()
            self.layer_combo.setCurrentIndex(0)
            self.tags_edit.setText("")
            self.is_start_cb.setChecked(False)
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
        for i, var in enumerate(variables[:4]):
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
        self.selected_node_id: Optional[str] = None
        self._build_ui()

    def set_context(self, script: ScriptData, ai: AIAssistant,
                    ui_queue: queue.SimpleQueue, on_graph_generated,
                    on_nodes_incremental=None):
        self._script = script
        self._ai = ai
        self._ui_queue = ui_queue
        self._on_graph_generated = on_graph_generated
        self._on_nodes_incremental = on_nodes_incremental

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

        arc = self._script.active_arc() if self._script else None
        arc_beats = arc.get('beats', {}) if arc else {}
        arc_motif = arc.get('motif', '') if arc else ''

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

            # Launch orchestrator
            self._orchestrator = ParallelNodeOrchestrator(
                script=self._script,
                ui_queue=self._ui_queue,
                model=self._ai.model,
                profile='full',
                story_context=self._script.story_context_focused,
                motif=arc_motif,
                themes=arc.get('themes', '') if arc else '',
                premise=prompt,
                arc_beats=arc_beats,
                variables=self._script.variables,
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
            )
            self._orchestrator.start(seed_ids)

        def on_seed_error(e):
            self.status_label.setText(f"Seed error: {e[:50]}")
            self.status_label.setStyleSheet("color: #ff5555; font-size: 10px;")

        self._ai.generate_seed(
            prompt, self._ui_queue, on_seed_done, on_seed_error,
            story_context=self._script.story_context_focused if self._script else '',
            layer_direction=arc_beats.get('arrival', ''),
            motif=arc_motif,
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


class ArcEditorDialog(QDialog):
    """Popup dialog for editing story arcs and wiring them into generation."""

    LAYER_NAMES = LAYER_ORDER

    def __init__(self, script: 'ScriptData', ai: 'AIAssistant',
                 ui_queue: queue.SimpleQueue, on_graph_generated=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Story Arcs")
        self.setMinimumSize(940, 720)
        self.script = script
        self._main_ai = ai
        self.ui_queue = ui_queue
        self._on_graph_generated = on_graph_generated
        self._arc_ai = AIAssistant()   # separate instance for arc chat
        self._current_arc_id: Optional[str] = None
        self._loading = False           # suppress dirty callbacks while populating fields
        self._build_ui()
        self._refresh_arc_list()
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
        ll.addWidget(QLabel("Arcs"))
        self.arc_list = QListWidget()
        self.arc_list.setMinimumWidth(150)
        self.arc_list.setMaximumWidth(200)
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

        sep = QLabel("Story Beats")
        sep.setStyleSheet("font-weight: bold; margin-top: 6px;")
        rl.addWidget(sep)
        hint = QLabel(
            "Each beat guides one generation layer. Leave blank for full AI freedom.")
        hint.setStyleSheet("color: #888888; font-size: 10px;")
        rl.addWidget(hint)

        beat_form = QFormLayout()
        self._beat_edits: dict = {}
        for layer in self.LAYER_NAMES:
            edit = QLineEdit()
            edit.setPlaceholderText(f"What should the {layer} layer cover?")
            edit.textChanged.connect(self._on_field_changed)
            self._beat_edits[layer] = edit
            beat_form.addRow(f"{layer.capitalize()}:", edit)
        rl.addLayout(beat_form)

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
        distill_btn = QPushButton("Distill Chat → Arc")
        distill_btn.setToolTip("Use AI to extract premise, themes, motif, and beats from the chat conversation")
        distill_btn.clicked.connect(self._cmd_distill_chat_to_arc)
        bot.addWidget(distill_btn)

        self.gen_btn = QPushButton("Generate Graph from Arc")
        self.gen_btn.setStyleSheet("font-weight: bold;")
        self.gen_btn.clicked.connect(self._cmd_generate_from_arc_parallel)
        bot.addWidget(self.gen_btn)

        bot.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        bot.addWidget(close_btn)
        root.addLayout(bot)

    # ── Arc list management ──────────────────────────────────────────────────

    def _refresh_arc_list(self):
        self.arc_list.blockSignals(True)
        try:
            self.arc_list.clear()
            active_id = self.script.active_arc_id
            for arc_id, arc in self.script.arcs.items():
                name = arc.get('name') or arc_id
                label = ('★ ' if arc_id == active_id else '  ') + name
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, arc_id)
                self.arc_list.addItem(item)
            # Restore selection while signals are still blocked
            target = self._current_arc_id
            for i in range(self.arc_list.count()):
                if self.arc_list.item(i).data(Qt.ItemDataRole.UserRole) == target:
                    self.arc_list.setCurrentRow(i)
                    return
            if self.arc_list.count():
                self.arc_list.setCurrentRow(0)
        finally:
            self.arc_list.blockSignals(False)

    def _on_arc_selected(self, row):
        self._save_current()
        if row < 0:
            self._current_arc_id = None
            self._clear_fields()
            return
        arc_id = self.arc_list.item(row).data(Qt.ItemDataRole.UserRole)
        self._current_arc_id = arc_id
        self.script.set_active_arc(arc_id)  # selected arc is always the active one
        self._refresh_arc_list()
        self._load_arc(arc_id)

    def _load_arc(self, arc_id: str):
        arc = self.script.arcs.get(arc_id, {})
        self._loading = True
        self.name_edit.setText(arc.get('name', ''))
        self.premise_edit.setPlainText(arc.get('premise', ''))
        self.themes_edit.setText(arc.get('themes', ''))
        self.motif_edit.setText(arc.get('motif', ''))
        beats = arc.get('beats', {})
        for layer, edit in self._beat_edits.items():
            edit.setText(beats.get(layer, ''))
        self.notes_edit.setPlainText(arc.get('notes', ''))
        self.chat_log.clear()
        for entry in arc.get('chat_history', []):
            self._append_chat(entry.get('role', 'user'), entry.get('content', ''))
        self._loading = False

    def _clear_fields(self):
        self._loading = True
        self.name_edit.clear()
        self.premise_edit.clear()
        self.themes_edit.clear()
        self.motif_edit.clear()
        for edit in self._beat_edits.values():
            edit.clear()
        self.notes_edit.clear()
        self.chat_log.clear()
        self._loading = False

    def _save_current(self):
        if not self._current_arc_id or self._loading:
            return
        arc_id = self._current_arc_id
        if arc_id not in self.script.arcs:
            return
        beats = {layer: edit.text() for layer, edit in self._beat_edits.items()}
        self.script.save_arc(arc_id, {
            'name':    self.name_edit.text(),
            'premise': self.premise_edit.toPlainText(),
            'themes':  self.themes_edit.text(),
            'motif':   self.motif_edit.text(),
            'beats':   beats,
            'notes':   self.notes_edit.toPlainText(),
        })
        self._refresh_list_item(arc_id)

    def _refresh_list_item(self, arc_id: str):
        active_id = self.script.active_arc_id
        name = self.script.arcs.get(arc_id, {}).get('name') or arc_id
        for i in range(self.arc_list.count()):
            item = self.arc_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == arc_id:
                item.setText(('★ ' if arc_id == active_id else '  ') + name)
                return

    def _on_field_changed(self):
        if not self._loading:
            self._save_current()

    def _cmd_new_arc(self):
        self._save_current()
        arc_id = self.script.add_arc()
        # If this is the first arc, seed it with whatever is already in the fields
        if len(self.script.arcs) == 1:
            beats = {layer: edit.text() for layer, edit in self._beat_edits.items()}
            name = self.name_edit.text().strip() or 'New Arc'
            self.script.save_arc(arc_id, {
                'name':    name,
                'premise': self.premise_edit.toPlainText(),
                'themes':  self.themes_edit.text(),
                'motif':   self.motif_edit.text(),
                'beats':   beats,
                'notes':   self.notes_edit.toPlainText(),
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
        """Parallel generation from arc — one node per AI call."""
        if not self._current_arc_id:
            self.chat_status.setText("No arc selected.")
            return
        self._save_current()
        arc = self.script.arcs.get(self._current_arc_id, {})

        if not self._main_ai.ready:
            self.chat_status.setText("claude CLI not found.")
            return

        prompt = arc.get('premise', '').strip() or arc.get('name', 'Generate a narrative graph.')
        arc_beats = arc.get('beats', {})
        arc_motif = arc.get('motif', '')
        arc_notes = arc.get('notes', '').strip()

        # Combine story context with arc notes so character/world details reach every node
        story_ctx = self.script.story_context_focused
        if arc_notes:
            story_ctx = (story_ctx + '\n\n' + arc_notes).strip() if story_ctx else arc_notes

        self.gen_btn.setEnabled(False)
        self.chat_status.setText("Generating seed nodes…")
        self._append_chat('assistant', f'[Parallel gen from arc: {arc.get("name", "")}]')

        def on_seed_done(data):
            before = set(self.script.nodes.keys())
            self.script.apply_generated(data)
            after = set(self.script.nodes.keys())
            seed_ids = sorted(after - before)

            for nid in seed_ids:
                if nid in self.script.nodes:
                    self.script.set_start(nid, True)

            self._append_chat('assistant', f'Seeds: {", ".join(seed_ids)}')
            self.chat_status.setText(f'Seeds done. Starting parallel expansion…')

            arc_themes = arc.get('themes', '')
            self._orchestrator = ParallelNodeOrchestrator(
                script=self.script,
                ui_queue=self.ui_queue,
                model=self._main_ai.model,
                profile='full',
                story_context=story_ctx,
                motif=arc_motif,
                themes=arc_themes,
                premise=prompt,
                arc_beats=arc_beats,
                variables=self.script.variables,
                on_progress=lambda msg: self.chat_status.setText(msg),
                on_complete=lambda: (
                    self.gen_btn.setEnabled(True),
                    self.chat_status.setText(
                        f'Parallel generation complete — {len(self.script.nodes)} nodes'),
                    self._on_graph_generated() if self._on_graph_generated else None,
                ),
                on_node_added=None,  # don't rebuild graph per-node; full rebuild on_complete
            )
            self._orchestrator.start(seed_ids)

        def on_seed_error(e):
            self.chat_status.setText(f'Seed error: {e[:60]}')
            self.gen_btn.setEnabled(True)

        self._main_ai.generate_seed(
            prompt, self.ui_queue, on_seed_done, on_seed_error,
            story_context=story_ctx,
            layer_direction=arc_beats.get('arrival', ''),
            motif=arc_motif,
            variables=self.script.variables,
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
        filled_beats = [(k, v) for k, v in arc.get('beats', {}).items() if v.strip()]
        if filled_beats:
            parts.append('Story beats:')
            for layer, beat in filled_beats:
                parts.append(f'  {layer}: {beat}')
        if arc.get('notes'):
            parts.append(f"Notes: {arc['notes']}")
        return '\n'.join(parts)

    def _append_chat(self, role: str, text: str):
        color = '#88ccff' if role == 'assistant' else '#cccccc'
        label = 'Claude' if role == 'assistant' else 'You'
        self.chat_log.append(
            f'<span style="color:{color};"><b>{label}:</b> {text}</span><br>')

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
            "You are distilling a brainstorming conversation into structured story arc fields "
            "for a narrative audio installation.\n\n"
            "The arc drives generation of a node graph where each node is 15-35 seconds of "
            "spoken audio. The 10 story layers are: " + layer_names + ".\n\n"
            "Extract the following from the conversation and return ONLY a JSON object:\n"
            "{\n"
            '  "name": "Short arc title (2-5 words)",\n'
            '  "premise": "The core story premise — what is this about? 1-3 sentences.",\n'
            '  "themes": "Comma-separated themes (e.g. isolation, transformation, memory)",\n'
            '  "motif": "One recurring sensory/symbolic thread to weave through every node",\n'
            '  "notes": "Character details, world-building, tone guidance — anything useful for generation",\n'
            '  "beats": {\n'
            '    "arrival": "What the arrival layer should establish",\n'
            '    "presence": "What presence should introduce",\n'
            '    "curiosity": "...",\n'
            '    "discovery": "...",\n'
            '    "complication": "...",\n'
            '    "intimacy": "...",\n'
            '    "turn": "...",\n'
            '    "consequence": "...",\n'
            '    "echo": "...",\n'
            '    "stillness": "What the final resting point should feel like"\n'
            '  }\n'
            "}\n\n"
            "Rules:\n"
            "- The premise should capture the ESSENCE of what was discussed, not summarize the conversation\n"
            "- Beats should be specific and actionable — not vague ('something changes') but concrete "
            "('the character realizes the sound was always there')\n"
            "- The motif should be a sensory detail that can appear in every node naturally\n"
            "- If the conversation didn't cover a beat, write one that fits the arc's trajectory\n"
            "- No markdown fences, no explanation — just the JSON"
        )

        prompt_parts = []
        if self.script.story_context_focused:
            prompt_parts.append(f'STORY CONTEXT (shared across all arcs — incorporate this setting/tone):\n{self.script.story_context_focused}')
        if existing:
            prompt_parts.append('EXISTING ARC FIELDS (refine these, don\'t ignore them):\n' + '\n'.join(existing))
        prompt_parts.append(f'CONVERSATION TO DISTILL:\n{conversation}')
        prompt = '\n\n'.join(prompt_parts)

        self.chat_status.setText("Distilling chat to arc fields...")
        self._append_chat('assistant', '[Distilling conversation into arc fields...]')

        def on_done(data):
            if not isinstance(data, dict):
                self.chat_status.setText("Distill failed: invalid response")
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
            beats = data.get('beats', {})
            for layer, text in beats.items():
                if layer in self._beat_edits and text:
                    self._beat_edits[layer].setText(text)

            self._on_field_changed()  # mark dirty
            self.chat_status.setText("Arc fields updated from chat.")
            self._append_chat('assistant',
                f'Distilled: "{data.get("name", "")}" — {data.get("premise", "")[:100]}...')

        def on_error(e):
            self.chat_status.setText(f"Distill error: {str(e)[:60]}")
            self._append_chat('assistant', f'[Distill error: {e}]')

        def run():
            try:
                raw = self._main_ai._run_claude(system, prompt)
                data = self._main_ai._extract_json(raw)
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
        self.chat_status.setText("Thinking…")

        def on_reply(reply):
            self._append_chat('assistant', reply)
            self.chat_status.setText('')
            if self._current_arc_id and self._current_arc_id in self.script.arcs:
                hist = self.script.arcs[self._current_arc_id].setdefault('chat_history', [])
                hist.append({'role': 'user',      'content': msg})
                hist.append({'role': 'assistant', 'content': reply})
                self.script.dirty = True

        def on_error(e):
            self.chat_status.setText(f'Error: {e[:80]}')

        # Combine script-level story context with arc-specific context
        full_ctx = ''
        if self.script.story_context_focused:
            full_ctx = f'STORY CONTEXT:\n{self.script.story_context_focused}\n\n'
        full_ctx += arc_ctx

        self._arc_ai.chat(msg, self.ui_queue,
                          on_reply=on_reply, on_error=on_error,
                          story_context=full_ctx,
                          _system_override=SYSTEM_ARC_CHAT)

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
                                    on_nodes_incremental=self._add_nodes_incremental)
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

        # NodeGraphQt graph
        self.graph = NodeGraph()
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

        search_row = QWidget()
        search_row.setMaximumHeight(30)
        search_row_layout = QHBoxLayout(search_row)
        search_row_layout.setContentsMargins(2, 2, 2, 2)
        search_row_layout.setSpacing(4)
        search_row_layout.addWidget(self._search_bar)
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

        main_layout.addWidget(splitter)

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
        act_arcs = QAction("Story Arcs…", self)
        act_arcs.setShortcut("Ctrl+Shift+R")
        act_arcs.triggered.connect(self._cmd_open_arc_editor)
        story_menu.addAction(act_arcs)

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

        act_audit = QAction("Audit Audio Files…", self)
        act_audit.triggered.connect(self._cmd_audit_audio)
        story_menu.addAction(act_audit)

        story_menu.addSeparator()
        model_menu = story_menu.addMenu("AI Model")
        self._model_actions = {}
        for model_id in [
            'claude-sonnet-4-6',
            'claude-opus-4-6',
            'claude-haiku-4-5-20251001',
        ]:
            short = model_id.split('-')[1].capitalize()  # Sonnet / Opus / Haiku
            act = QAction(short, self, checkable=True)
            act.setData(model_id)
            act.triggered.connect(lambda checked, m=model_id: self._set_ai_model(m))
            model_menu.addAction(act)
            self._model_actions[model_id] = act
        # Check the default
        default_act = self._model_actions.get(self.ai.model)
        if default_act:
            default_act.setChecked(True)

        # Analysis menu
        analysis_menu = menubar.addMenu("Analysis")
        act_freq = QAction("Toggle Frequency Heat Map", self)
        act_freq.setShortcut("Ctrl+Shift+A")
        act_freq.triggered.connect(lambda: self._freq_btn.setChecked(not self._freq_btn.isChecked()))
        analysis_menu.addAction(act_freq)

        # Voice menu
        voice_menu = menubar.addMenu("Voice")
        act_voice = QAction("Voice Settings…", self)
        act_voice.setShortcut("Ctrl+Shift+V")
        act_voice.triggered.connect(self._cmd_open_voice_settings)
        voice_menu.addAction(act_voice)

    def _cmd_open_arc_editor(self):
        dlg = ArcEditorDialog(self.script, self.ai, self.ui_queue,
                              on_graph_generated=self._on_graph_generated,
                              parent=self)
        dlg.exec()
        self._update_title()

    def _cmd_open_story_context(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Story Context")
        dlg.setMinimumWidth(820)
        dlg.setMinimumHeight(460)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: full context (not sent to AI) ─────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_lbl = QLabel("Full Context  (reference only — not seen by AI)")
        left_lbl.setStyleSheet("color: #aaaaaa; font-size: 10px; font-weight: bold;")
        left_layout.addWidget(left_lbl)
        full_edit = QTextEdit()
        full_edit.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        full_edit.setPlainText(self.script.story_context)
        left_layout.addWidget(full_edit)
        full_char_lbl = QLabel(f"{len(self.script.story_context)} characters")
        full_char_lbl.setStyleSheet("color: #888888; font-size: 10px;")
        full_edit.textChanged.connect(
            lambda: full_char_lbl.setText(f"{len(full_edit.toPlainText())} characters"))
        left_layout.addWidget(full_char_lbl)
        splitter.addWidget(left)

        # ── Right: focused context (sent to AI) ─────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_lbl = QLabel(f"Focused Context  (seen by AI — max {FOCUSED_CONTEXT_MAX} characters)")
        right_lbl.setStyleSheet("color: #aaddaa; font-size: 10px; font-weight: bold;")
        right_layout.addWidget(right_lbl)
        focused_edit = QTextEdit()
        focused_edit.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        focused_edit.setPlainText(self.script.story_context_focused)
        right_layout.addWidget(focused_edit)

        def _focused_char_text():
            n = len(focused_edit.toPlainText())
            color = "#ff7777" if n > FOCUSED_CONTEXT_MAX else "#888888"
            return f'<span style="color:{color};font-size:10px;">{n} / {FOCUSED_CONTEXT_MAX} characters</span>'

        focused_char_lbl = QLabel(_focused_char_text())
        focused_char_lbl.setTextFormat(Qt.TextFormat.RichText)
        focused_edit.textChanged.connect(lambda: focused_char_lbl.setText(_focused_char_text()))
        right_layout.addWidget(focused_char_lbl)

        gen_btn = QPushButton("Generate Focused Context from Full (AI)")
        gen_btn.setToolTip("Use Claude to distill the full context into a focused ~800-char version")
        right_layout.addWidget(gen_btn)
        splitter.addWidget(right)

        splitter.setSizes([400, 400])
        layout.addWidget(splitter)

        btn_row = QHBoxLayout()
        status_lbl = QLabel("")
        status_lbl.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        btn_row.addWidget(status_lbl)
        btn_row.addStretch()
        save_btn = QPushButton("Save & Close")
        save_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

        def on_generate():
            full = full_edit.toPlainText().strip()
            if not full:
                status_lbl.setText("Full context is empty.")
                return
            if not self.ai.ready:
                status_lbl.setText("Claude CLI not found.")
                return
            if self.ai.busy:
                status_lbl.setText("AI is busy...")
                return
            gen_btn.setEnabled(False)
            status_lbl.setText("Generating...")
            def on_done(text):
                focused_edit.setPlainText(text)
                gen_btn.setEnabled(True)
                status_lbl.setText("Done.")
            def on_error(e):
                gen_btn.setEnabled(True)
                status_lbl.setText(f"Error: {e[:60]}")
            self.ai.generate_focused_context(full, self.ui_queue, on_done, on_error)

        gen_btn.clicked.connect(on_generate)

        dlg.exec()
        self.script.set_story_context(full_edit.toPlainText())
        self.script.set_story_context_focused(focused_edit.toPlainText())
        self._update_title()

    def _set_ai_model(self, model_id: str):
        """Switch the AI model used for node generation."""
        self.ai.model = model_id
        for mid, act in self._model_actions.items():
            act.setChecked(mid == model_id)
        short = model_id.split('-')[1].capitalize()
        self.status_bar.showMessage(f"AI model: {short}", 3000)

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
        """Open dialog to define up to 4 story-level variables."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Story Variables")
        dlg.setMinimumWidth(520)
        dlg.setMinimumHeight(250)
        main_layout = QVBoxLayout(dlg)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        info_lbl = QLabel("Define up to 4 numeric variables (0–1) tracked per node. "
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
            if len(row_widgets) >= 4:
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
                row_widgets.remove(e)
                e[3].deleteLater()
                add_btn.setEnabled(len(row_widgets) < 4)

            rm_btn.clicked.connect(remove)
            add_btn.setEnabled(len(row_widgets) < 4)

        # Add variable button
        add_btn = QPushButton("+ Add Variable")
        add_btn.clicked.connect(lambda: add_row())
        main_layout.addWidget(add_btn)

        main_layout.addStretch()

        # Save & Close
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        save_btn = QPushButton("Save && Close")
        save_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(save_btn)
        main_layout.addLayout(btn_row)

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

        arc = self.script.active_arc() if self.script else None
        arc_beats = arc.get('beats', {}) if arc else {}
        arc_motif = arc.get('motif', '') if arc else ''

        # Override the expand profile's width with the user's min/max
        expand_profile = {
            'max_depth': 2,
            'widths': {'*': (node_min, node_max)},
        }

        arc_premise = arc.get('premise', '') if arc else ''
        arc_themes = arc.get('themes', '') if arc else ''

        self._job_counter += 1
        job_tag = f"expand#{self._job_counter}"

        def _make_expand_orch(jt):
            o = ParallelNodeOrchestrator(
                script=self.script,
                ui_queue=self.ui_queue,
                model=self.ai.model,
                profile='expand',
                story_context=self.script.story_context_focused,
                motif=arc_motif,
                themes=arc_themes,
                premise=arc_premise,
                arc_beats=arc_beats,
                variables=self.script.variables,
                on_progress=lambda msg: self.status_bar.showMessage(f"[{jt}] {msg}"),
                on_complete=lambda: self._on_orchestrator_complete(o, jt, "Expand"),
                on_node_added=self._add_nodes_incremental,
            )
            return o

        orch = _make_expand_orch(job_tag)
        orch._profile = expand_profile
        self._orchestrators.append(orch)
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

        arc = self.script.active_arc() if self.script else None
        arc_beats = arc.get('beats', {}) if arc else {}
        arc_motif = arc.get('motif', '') if arc else ''
        arc_premise = arc.get('premise', '') if arc else ''
        arc_themes = arc.get('themes', '') if arc else ''

        self._job_counter += 1
        job_tag = f"continue#{self._job_counter}"

        def _make_continue_orch(jt):
            o = ParallelNodeOrchestrator(
                script=self.script,
                ui_queue=self.ui_queue,
                model=self.ai.model,
                profile='continue',
                story_context=self.script.story_context_focused,
                motif=arc_motif,
                themes=arc_themes,
                premise=arc_premise,
                arc_beats=arc_beats,
                variables=self.script.variables,
                on_progress=lambda msg: self.status_bar.showMessage(f"[{jt}] {msg}"),
                on_complete=lambda: self._on_orchestrator_complete(o, jt, "Continue"),
                on_node_added=self._add_nodes_incremental,
            )
            return o

        orch = _make_continue_orch(job_tag)
        self._orchestrators.append(orch)
        if multi:
            orch.start_merged(parent_ids)
        else:
            orch.start(parent_ids)

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

    def _run_frequency_simulation(self, n_runs: int = 2000) -> dict:
        """Monte Carlo random walk with recency damping.

        Estimates each node's audio duration from word count (121 wpm) plus a
        3 s inter-node delay.  Tracks simulated time so recency counters decay
        at the same rate as the real player (1 count per 36000 s / 10 hours).
        """
        nodes = self.script.nodes
        starts = self.script.start_nodes or list(nodes.keys())
        counts = {nid: 0 for nid in nodes}

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
            while current and steps < 300:
                if current not in nodes:
                    break
                counts[current] += 1

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
                current = nxt
                steps += 1
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

    def _cmd_search(self, text: str):
        """Apply crosshatch overlays to nodes matching the search term.
        Does not change node opacity — selection highlight is fully independent."""
        if not text.strip():
            self._clear_search_overlays()
            return
        term = text.strip().lower()
        matched = set()
        for nid, nd in self.script.nodes.items():
            haystack = ' '.join([
                nid,
                nd.get('label', '') or '',
                nd.get('text', '') or '',
                ' '.join(nd.get('tags', [])),
                nd.get('arc_beat', '') or '',
            ]).lower()
            if term in haystack:
                matched.add(nid)
        self._apply_search_overlays(matched)
        self.status_bar.showMessage(f"Search: {len(matched)} matching node(s)")

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

    def _autosave(self):
        """Silently save if dirty and a file path exists."""
        if self.script.dirty and self.script.path:
            try:
                self._sync_positions()
                self.script.save()
                self._update_title()
            except Exception:
                pass  # autosave should never interrupt the user

    def _cmd_save(self):
        self._sync_positions()
        if self.script.path:
            try:
                self.script.save()
                self._update_title()
                self.status_bar.showMessage(f"Saved: {self.script.path.name}")
            except Exception as exc:
                self.status_bar.showMessage(f"Save error: {exc}")
        else:
            self._cmd_save_as()

    def _cmd_save_as(self):
        self._sync_positions()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Script", str(SOUNDS_DIR), "JSON files (*.json)"
        )
        if path:
            try:
                self.script.save(Path(path))
                self._update_title()
                self.status_bar.showMessage(f"Saved: {Path(path).name}")
            except Exception as exc:
                self.status_bar.showMessage(f"Save error: {exc}")

    def _load_script(self, path: Path):
        try:
            self.script = ScriptData.load(path)
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
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))

    def _refresh_contexts(self):
        """Re-wire all panels after script is replaced."""
        self.props_panel.set_context(self.script, self.vm, self.ai, self.ui_queue)
        self.props_panel.refresh_variable_widgets()
        self.voice_panel.set_context(self.script, self.vm, self.ui_queue, self.props_panel)
        self.chat_panel.set_context(self.script, self.ai, self.ui_queue,
                                    self._on_graph_generated,
                                    on_nodes_incremental=self._add_nodes_incremental)
        self.play_panel.set_context(self.script, self.ui_queue)

    def _update_title(self):
        if self.script.path:
            name = self.script.path.name
        else:
            name = self.script.name or "New Script"
        dirty = "*" if self.script.dirty else ""
        self.setWindowTitle(f"Narrative Editor — {name}{dirty}")

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

def main():
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
