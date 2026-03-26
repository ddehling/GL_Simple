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
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

# Must be set before any Qt or NodeGraphQt imports
os.environ['QT_API'] = 'pyside6'

from PySide6.QtCore import Qt, QTimer, Signal, QObject, QEvent
from PySide6.QtGui import QColor, QPalette, QAction, QTextOption, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog,
    QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPushButton, QScrollArea, QSplitter,
    QStatusBar, QTextEdit, QVBoxLayout, QWidget,
)

from NodeGraphQt import NodeGraph, BaseNode

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT  = Path(__file__).parent.parent
SOUNDS_DIR = REPO_ROOT / "media" / "sounds"

NODE_PREVIEW_LEN = 60   # chars shown inside a node box

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
  "start_nodes": ["intro_a"],
  "nodes": {
    "node_id": {
      "text": "Spoken text, 40-100 words.",
      "next": ["next_id"],
      "weights": [1.0],
      "tags": ["intro"],
      "voice_settings": {"stability": 0.65, "similarity_boost": 0.75, "style": 0.2}
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

LAYER STRUCTURE (use as many layers as the content warrants, up to 8):
  Layer 1 (intro)      : 1–3 nodes  — establish tone, place, or speaker
  Layer 2 (opening)    : 2–4 nodes  — widen the scene, first impressions
  Layer 3 (development): 3–6 nodes  — explore the theme from different angles
  Layer 4 (deepening)  : 2–5 nodes  — go further, add texture or contrast
  Layer 5 (bridge)     : 2–4 nodes  — transitional energy, shift is coming
  Layer 6 (turn)       : 2–4 nodes  — complication, revelation, or emotional shift
  Layer 7 (descent)    : 1–3 nodes  — lean into the change, consequences felt
  Layer 8 (resolution) : 1–3 nodes  — landing, conclusion, or open question

For shorter scripts, skip layers or collapse them — a 4-layer script is fine.
For longer scripts, use all 8 to create a full arc with genuine depth.

HARD LIMIT: generate no more than 12 nodes total. If the full arc needs more, compress layers,
reduce siblings per layer, or end earlier — but never exceed 12 nodes.

BRANCHING: any node in layer N may connect to 2–4 nodes in layer N+1.
MERGING:   multiple nodes in layer N may all point to the same node in layer N+1.
           This creates convergence points — moments every path passes through.

Good pattern:
  intro → [open_a, open_b]             ← branch early
  open_a, open_b → [dev_a, dev_b, dev_c]
  dev_a, dev_b → [turn_x]              ← merge
  dev_c        → [turn_y]
  turn_x, turn_y → [close_a, close_b]  ← merge then branch again

Avoid:
  - Fully connected pools where every node points to every other node
  - Trees that only branch and never merge (too many dead-end leaves)
  - Chains with no branching at all (boring, no variation)
  - Nodes that re-establish context already given by their parents

WEIGHTS: use 1.0 as default. Use 2.0 to favour a path, 0.5 to make it rare.
TAGS: every node must have a tags array with:
1. Exactly one layer tag: intro / opening / development / deepening / bridge / turn / descent / resolution
2. Custom tags for everything present in the node text — characters, themes, locations, objects, moods.
   Use short lowercase snake_case words. Reuse the same tag across nodes whenever the same element recurs.
   Examples: "crow", "test_anxiety", "linoleum", "waiting", "silence", "rain"
node IDs: short_snake_case, layer-prefixed (e.g. "intro_storm", "dev_pride", "turn_silence", "res_still")

VOICE SETTINGS: set "voice_settings" on every node to match its emotional tone:
  stability      0.0–1.0  lower = more expressive/varied delivery
  similarity_boost         leave at 0.75 unless noted
  style          0.0–1.0  higher = more dramatic/theatrical

  Layer defaults:
    intro       stability 0.65  style 0.10  (calm, orienting)
    opening     stability 0.60  style 0.20  (settling in, atmospheric)
    development stability 0.50  style 0.35  (engaged, exploring)
    deepening   stability 0.45  style 0.45  (more invested, richer)
    bridge      stability 0.45  style 0.40  (transitional energy)
    turn        stability 0.30  style 0.65  (tense, expressive)
    descent     stability 0.25  style 0.70  (intense, committed)
    resolution  stability 0.60  style 0.15  (settled, reflective)
  Adjust within layer if the content is notably more or less intense than usual.

TERMINAL NODES: resolution/ending nodes must have next: [].
NEVER create edges that point back toward intro or start nodes.
When a terminal node finishes playing, the runtime will automatically restart
from a randomly chosen start_node — no explicit loop edges are needed or wanted.
"""

SYSTEM_GENERATE_SEED = """\
You are a narrative script writer for an immersive audio installation.
Scripts play as atmospheric spoken audio layered over weather and lighting effects.

Each node is one short spoken segment (40–100 words, ~15–35 seconds when read aloud).
Use evocative, atmospheric language suited to the theme.

Generate ONLY the intro layer — 1 to 3 opening nodes that establish the tone and world.
These are the first words the audience will hear. Leave "next" as [] for ALL nodes —
subsequent layers will be generated separately in a follow-up step.

OUTPUT FORMAT — respond with ONLY this JSON, no markdown fences, no explanation:
{
  "name": "Script name",
  "description": "One-line description",
  "start_nodes": ["intro_a"],
  "nodes": {
    "intro_a": {
      "text": "Spoken text, 40-100 words.",
      "next": [],
      "weights": [],
      "tags": ["intro"],
      "voice_settings": {"stability": 0.65, "similarity_boost": 0.75, "style": 0.10}
    }
  }
}

Node IDs: short_snake_case, intro-prefixed (e.g. "intro_storm", "intro_silence").
TAGS: "intro" plus custom content tags — characters, themes, locations, moods.
VOICE SETTINGS: stability 0.65, similarity_boost 0.75, style 0.10 (calm, orienting).
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
      "voice_settings": {"stability": 0.30, "similarity_boost": 0.75, "style": 0.65}
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
- Layer order: intro → opening → development → deepening → bridge → turn → descent → resolution
  Determine the source's layer from its tags, then generate only the layers that come AFTER it
- Use the same branching (2–4 children) and merging (multiple parents → 1 child) patterns as a full graph
- TERMINAL NODES: the final layer's nodes must have next: []
- TERMINAL NODES: resolution/ending nodes must have next: []. Never loop back to earlier layers.

BRANCHING and MERGING: same rules as full graph generation.
WEIGHTS: 1.0 default, 2.0 to favour, 0.5 to make rare.
TAGS: every node must have a tags array with:
1. Exactly one layer tag: deepening / bridge / turn / descent / resolution (whichever applies)
2. Custom tags for everything present in the node text — characters, themes, locations, objects, moods.
   Use short lowercase snake_case. Reuse the same tag across nodes whenever the same element recurs.
node IDs: short_snake_case, layer-prefixed (e.g. "turn_silence", "res_still").

VOICE SETTINGS: match emotional tone to layer:
  deepening   stability 0.45  style 0.45
  bridge      stability 0.45  style 0.40
  turn        stability 0.30  style 0.65
  descent     stability 0.25  style 0.70
  resolution  stability 0.60  style 0.15
"""

SYSTEM_EXPAND = """\
You are expanding a single node in a narrative graph for an immersive audio installation.
You will receive one existing node and must generate new nodes that continue FROM it.

Each new node is a short spoken segment (40–100 words, ~15–35 seconds when read aloud).

THEMATIC CONTINUITY — this is the most important rule:
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
      "tags": ["development", "toad", "revelation"],
      "voice_settings": {"stability": 0.50, "similarity_boost": 0.75, "style": 0.35}
    }
  },
  "connect_from": ["new_node_id_1", "new_node_id_2"]
}

TAGGING RULES — every node must have a tags array with:
1. Exactly one layer tag (intro/opening/development/deepening/bridge/turn/descent/resolution)
2. Custom tags for everything present in the node text — characters, themes, locations, objects.
   Any custom tag from the parent node should appear in daughters where that element is present.
   Prefer tags already used in the script (a list will be provided).

"connect_from": the new node IDs that the SOURCE node should gain edges to.
All other edges are between the new nodes themselves.

Layer progression rules (secondary to thematic continuity):
- Layer order: intro → opening → development → deepening → bridge → turn → descent → resolution
- Determine the source node's layer from its tags, then place new nodes in the NEXT layer(s)
- You may skip layers if the content calls for it, or span multiple layers in one expansion
- You may branch (source → multiple new nodes) or chain (source → A → B → C → ...)
- Branching then merging is encouraged: source → [A, B] and both A, B → C
- The prompt will specify an exact node count range — generate that many new nodes, no more, no fewer
- node IDs: short_snake_case, layer-prefixed (e.g. "dev_kelp_drift", "turn_silence", "res_still")
- Weights default to 1.0 unless you have reason to favour one path
- Terminal nodes (resolution/descent end) must have next: [] — NEVER link back to intro or start nodes.
  The runtime restarts automatically from a random start node when a terminal finishes.

VOICE SETTINGS: set voice_settings on every node (stability 0-1, similarity_boost 0.75, style 0-1).
  intro   stab~0.65 style~0.10 | opening  stab~0.60 style~0.20
  develop stab~0.50 style~0.35 | deepen   stab~0.45 style~0.45
  bridge  stab~0.45 style~0.40 | turn     stab~0.30 style~0.65
  descent stab~0.25 style~0.70 | resolut  stab~0.60 style~0.15
  Adjust within the layer to match the specific emotional intensity of the node's text.
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
    intro   stab~0.65 style~0.10 | opening  stab~0.60 style~0.20
    develop stab~0.50 style~0.35 | deepen   stab~0.45 style~0.45
    bridge  stab~0.45 style~0.40 | turn     stab~0.30 style~0.65
    descent stab~0.25 style~0.70 | resolut  stab~0.60 style~0.15
- Adjust to match the specific emotional intensity of the rewritten text

Respond with ONLY a JSON object in this exact format, no other text:
{
  "text": "...",
  "tags": ["development", "toad", "revelation"],
  "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.3}
}

TAGGING RULES — same as for expansion:
1. Exactly one layer tag (intro/opening/development/deepening/bridge/turn/descent/resolution)
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
            self._data["nodes"][node_id]["text"] = text
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
        intro < opening < development < deepening < bridge < turn < descent < resolution
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

        LAYER_ORDER = ["intro", "opening", "development", "deepening", "bridge", "turn", "descent", "resolution"]
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
        LAYER_ORDER = ["intro", "opening", "development", "deepening", "bridge", "turn", "descent", "resolution"]
        LAYER_X     = {name: 80 + i * 310 for i, name in enumerate(LAYER_ORDER)}
        LAYER_X["_default"] = 80 + len(LAYER_ORDER) * 310

        layer_counts: dict = {}
        for nd in self._data["nodes"].values():
            for tag in nd.get("tags", []):
                if tag in LAYER_ORDER:
                    layer_counts[tag] = layer_counts.get(tag, 0) + 1

        for nid, ndata in expansion.get("nodes", {}).items():
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

        # Wire source node to the connect_from targets
        src = self._data["nodes"].get(source_id)
        if src:
            for nid in expansion.get("connect_from", []):
                if nid in self._data["nodes"] and nid not in src["next"]:
                    src["next"].append(nid)
                    src["weights"].append(1.0)

        self.dirty = True


# ─────────────────────────────────────────────────────────────────────────────
# AI Assistant
# ─────────────────────────────────────────────────────────────────────────────

class AIAssistant:
    """Calls the `claude` CLI via subprocess — uses your Claude Code session,
    no separate API key required."""

    def __init__(self):
        self._history: list = []
        self._busy = False
        self._claude_exe: Optional[str] = self._find_claude()

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

    def _run_claude(self, system: str, prompt: str) -> str:
        """Blocking call to `claude -p`. Raises on non-zero exit."""
        cmd = [
            self._claude_exe,
            "--no-session-persistence",
            "--system-prompt", system,
            "--output-format", "text",
            "-p", prompt,
        ]
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
            raise RuntimeError(err)
        out = result.stdout.strip()
        if not out:
            raise RuntimeError(
                f"claude produced no output (stderr: {result.stderr.strip()!r})"
            )
        return out

    def chat(self, user_msg: str, ui_queue: queue.SimpleQueue,
             on_reply, on_error, script_summary: str = '', story_context: str = ''):
        if self._busy:
            return
        self._busy = True
        self._history.append({"role": "user", "content": user_msg})

        parts = []
        if story_context:
            parts.append(f"Story context:\n{story_context}")
        if script_summary:
            parts.append(f"Current script:\n{script_summary}")
        transcript = self._transcript()
        if transcript:
            parts.append(transcript)
        parts.append(f"User: {user_msg}")
        full_prompt = "\n\n".join(parts)

        def run():
            try:
                reply = self._run_claude(SYSTEM_CHAT, full_prompt)
                self._history.append({"role": "assistant", "content": reply})
                ui_queue.put(lambda: on_reply(reply))
            except Exception as exc:
                self._history.pop()
                ui_queue.put(lambda e=exc: on_error(str(e)))
            finally:
                self._busy = False

        threading.Thread(target=run, daemon=True).start()

    def generate_graph(self, prompt: str, ui_queue: queue.SimpleQueue,
                       on_done, on_error, story_context: str = ''):
        if self._busy:
            return
        self._busy = True

        context = self._transcript(limit=6)
        parts = []
        if story_context:
            parts.append(f'Story context:\n{story_context}')
        parts.append(prompt)
        if context:
            parts.append(context)
        full_prompt = '\n\n'.join(parts)

        def run():
            try:
                raw   = self._run_claude(SYSTEM_GENERATE, full_prompt)
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if not match:
                    ui_queue.put(lambda: on_error("No JSON found in response"))
                    return
                data = json.loads(match.group(0))
                ui_queue.put(lambda: on_done(data))
            except json.JSONDecodeError as exc:
                ui_queue.put(lambda: on_error(f"JSON parse error: {exc}"))
            except Exception as exc:
                ui_queue.put(lambda e=exc: on_error(str(e)))
            finally:
                self._busy = False

        threading.Thread(target=run, daemon=True).start()

    def generate_seed(self, prompt: str, ui_queue: queue.SimpleQueue,
                      on_done, on_error, story_context: str = ''):
        """Generate only the intro layer — no children. Used to seed iterative generation."""
        if self._busy:
            return
        self._busy = True

        context = self._transcript(limit=6)
        parts = []
        if story_context:
            parts.append(f'Story context:\n{story_context}')
        parts.append(prompt)
        if context:
            parts.append(context)
        full_prompt = '\n\n'.join(parts)

        def run():
            try:
                raw   = self._run_claude(SYSTEM_GENERATE_SEED, full_prompt)
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if not match:
                    ui_queue.put(lambda: on_error("No JSON found in response"))
                    return
                data = json.loads(match.group(0))
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
                    existing_custom_tags: list = None):
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

        # ── Ancestor context: oldest first, least influential ────────────────
        ANCESTOR_LABELS = [
            ('EARLIER CONTEXT (faint background)',     40),
            ('GREAT-GRANDPARENT (minimal influence)',  60),
            ('GRANDPARENT (light influence)',         100),
            ('DIRECT PARENT (moderate influence)',    200),
        ]
        if upstream_path:
            parts.append('\nANCESTOR CONTEXT (for arc continuity only — do not let this pull the theme away from the source node):')
            for i, (nid, text) in enumerate(upstream_path):  # oldest first
                label, max_chars = ANCESTOR_LABELS[i] if i < len(ANCESTOR_LABELS) else ANCESTOR_LABELS[0]
                excerpt = text[:max_chars] + '...' if len(text) > max_chars else text
                parts.append(f'  {label}\n  [{nid}]: "{excerpt}"')

        # ── Node intent / guidance (near-primary) ────────────────────────────
        if node_hint:
            parts.append(
                f'\nNODE INTENT (author direction — high weight):\n  {node_hint}'
            )
        if hint:
            parts.append(f'GUIDANCE: {hint}')

        # ── Source node (highest weight — closest to generation) ─────────────
        layer_tags = {"intro","opening","development","deepening","bridge","turn","descent","resolution"}
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
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if not match:
                    ui_queue.put(lambda: on_error("No JSON found in response"))
                    return
                data = json.loads(match.group(0))
                ui_queue.put(lambda: on_done(data))
            except json.JSONDecodeError as exc:
                ui_queue.put(lambda e=exc: on_error(f"JSON parse error: {e}"))
            except Exception as exc:
                ui_queue.put(lambda e=exc: on_error(str(e)))
            finally:
                self._busy = False

        threading.Thread(target=run, daemon=True).start()

    def continue_from_node(self, source_id: str, source_text: str, source_tags: list,
                           ui_queue: queue.SimpleQueue, on_done, on_error,
                           story_context: str = '', node_hint: str = ''):
        if self._busy:
            return
        self._busy = True

        layer_order = ["intro", "opening", "development", "deepening",
                       "bridge", "turn", "descent", "resolution"]
        source_layer = next((t for t in source_tags if t in layer_order), 'development')

        parts = []
        if story_context:
            sc = story_context[:1000] + '...' if len(story_context) > 1000 else story_context
            parts.append(f'BACKGROUND (story flavour only):\n  {sc}')
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
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if not match:
                    ui_queue.put(lambda: on_error("No JSON found in response"))
                    return
                data = json.loads(match.group(0))
                ui_queue.put(lambda: on_done(data))
            except json.JSONDecodeError as exc:
                ui_queue.put(lambda e=exc: on_error(f"JSON parse error: {e}"))
            except Exception as exc:
                ui_queue.put(lambda e=exc: on_error(str(e)))
            finally:
                self._busy = False

        threading.Thread(target=run, daemon=True).start()

    def rewrite_text(self, prompt: str, ui_queue: queue.SimpleQueue,
                     on_done, on_error, story_context: str = ''):
        if self._busy:
            return
        self._busy = True

        full_prompt = (f'Story context:\n{story_context}\n\n{prompt}'
                       if story_context else prompt)

        def run():
            try:
                raw   = self._run_claude(SYSTEM_REWRITE, full_prompt)
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if not match:
                    ui_queue.put(lambda: on_error("No JSON in response"))
                    return
                data = json.loads(match.group(0))
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
                client = ElevenLabs(api_key=self.api_key)
                audio = client.text_to_speech.convert(
                    voice_id=voice_id,
                    text=text,
                    model_id=settings.get("model", "eleven_multilingual_v2"),
                    voice_settings=VoiceSettings(
                        stability=settings.get("stability", 0.5),
                        similarity_boost=settings.get("similarity_boost", 0.75),
                        style=settings.get("style", 0.3),
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
    'intro':       (60,  100, 180),
    'opening':     (60,  130, 160),
    'development': (50,  140, 70),
    'deepening':   (80,  130, 60),
    'bridge':      (140, 120, 50),
    'turn':        (180, 80,  50),
    'descent':     (160, 50,  60),
    'resolution':  (120, 60,  150),
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
        layout.addWidget(self.text_edit)

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
        expand_btn = QPushButton("Expand Node (AI)")
        expand_btn.clicked.connect(self._cmd_expand)
        expand_row.addWidget(expand_btn)
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
        for _lt in ["intro", "opening", "development", "deepening",
                    "bridge", "turn", "descent", "resolution"]:
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
        self.stability_spin.setRange(0.0, 1.0)
        self.stability_spin.setSingleStep(0.05)
        self.stability_spin.setDecimals(2)
        self.stability_spin.valueChanged.connect(self._autosave_voice_settings)
        form.addRow("Stability:", self.stability_spin)

        self.similarity_spin = QDoubleSpinBox()
        self.similarity_spin.setRange(0.0, 1.0)
        self.similarity_spin.setSingleStep(0.05)
        self.similarity_spin.setDecimals(2)
        self.similarity_spin.valueChanged.connect(self._autosave_voice_settings)
        form.addRow("Similarity Boost:", self.similarity_spin)

        self.style_spin = QDoubleSpinBox()
        self.style_spin.setRange(0.0, 1.0)
        self.style_spin.setSingleStep(0.05)
        self.style_spin.setDecimals(2)
        self.style_spin.valueChanged.connect(self._autosave_voice_settings)
        form.addRow("Style:", self.style_spin)

        layout.addLayout(form)

        gen_audio_btn = QPushButton("Generate Audio")
        gen_audio_btn.clicked.connect(self._cmd_generate_audio)
        layout.addWidget(gen_audio_btn)

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
            all_tags   = nd.get("tags", [])
            layer_tags = {"intro","opening","development","deepening","bridge","turn","descent","resolution"}
            layer_tag  = next((t for t in all_tags if t in layer_tags), "")
            custom_tags = [t for t in all_tags if t not in layer_tags]
            idx = self.layer_combo.findData(layer_tag)
            self.layer_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.tags_edit.setText(", ".join(custom_tags))
            self.is_start_cb.setChecked(node_id in script.start_nodes)

            vs = nd.get("voice_settings", {})
            self.stability_spin.setValue(vs.get("stability", 0.5))
            self.similarity_spin.setValue(vs.get("similarity_boost", 0.75))
            self.style_spin.setValue(vs.get("style", 0.3))

            # Voice combo
            voice_id = nd.get("voice")
            if voice_id and self._vm:
                name = self._vm.name_for_id(voice_id) or voice_id
                idx = self.voice_combo.findText(name)
                self.voice_combo.setCurrentIndex(idx if idx >= 0 else 0)
            else:
                self.voice_combo.setCurrentIndex(0)

            # Audio status
            file_path = nd.get("file")
            if file_path:
                self.audio_status.setText(f"File: {Path(file_path).name}")
                self.audio_status.setStyleSheet("color: #88ee88; font-size: 10px;")
            else:
                self.audio_status.setText("No audio file")
                self.audio_status.setStyleSheet("color: #aaaaaa; font-size: 10px;")

            self.rebuild_edge_list(script, node_id)
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

    def clear(self):
        self._node_id = None
        self._blocking = True
        try:
            self.id_edit.setText("")
            self.label_edit.setText("")
            self.text_edit.setPlainText("")
            self.hint_edit.setPlainText("")
            self.layer_combo.setCurrentIndex(0)
            self.tags_edit.setText("")
            self.is_start_cb.setChecked(False)
            self.stability_spin.setValue(0.5)
            self.similarity_spin.setValue(0.75)
            self.style_spin.setValue(0.3)
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
            "style":            round(self.style_spin.value(), 2),
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
        layer_tags = {"intro","opening","development","deepening","bridge","turn","descent","resolution"}
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
                        "style":            vs.get("style",            0.3),
                    })
                if self._node_id == node_id:
                    self._blocking = True
                    try:
                        self.text_edit.setPlainText(new_text)
                        if new_tags:
                            _lt = {"intro","opening","development","deepening",
                                   "bridge","turn","descent","resolution"}
                            lt  = next((t for t in new_tags if t in _lt), "")
                            ct  = [t for t in new_tags if t not in _lt]
                            idx = self.layer_combo.findData(lt)
                            self.layer_combo.setCurrentIndex(idx if idx >= 0 else 0)
                            self.tags_edit.setText(", ".join(ct))
                        if vs:
                            self.stability_spin.setValue(vs.get("stability", 0.5))
                            self.similarity_spin.setValue(vs.get("similarity_boost", 0.75))
                            self.style_spin.setValue(vs.get("style", 0.3))
                    finally:
                        self._blocking = False
            self.rewrite_status.setText("Done")
            self.rewrite_status.setStyleSheet("color: #88ee88; font-size: 10px;")
            self.node_modified.emit(node_id)

        def on_error(e):
            self.rewrite_status.setText(f"Error: {e[:60]}")
            self.rewrite_status.setStyleSheet("color: #ff5555; font-size: 10px;")

        self._ai.rewrite_text(prompt, self._ui_queue, on_done, on_error,
                              story_context=self._script.story_context_focused)

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
                self.audio_status.setText(f"Saved: {path.name}")
                self.audio_status.setStyleSheet("color: #88ee88; font-size: 10px;")

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
        gen_btn.clicked.connect(self._cmd_generate_graph)
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

    def _cmd_generate_graph(self):
        if not self._ai or not self._ui_queue or not self._script:
            return
        if not self._ai.ready:
            self.status_label.setText("claude CLI not found")
            return
        if self._ai.busy:
            self.status_label.setText("AI is busy...")
            return

        prompt = ' '.join(self.chat_input.toPlainText().split('\n')).strip()
        if not prompt:
            prompt = "Generate a narrative graph based on our conversation so far."
        self.chat_input.setPlainText("")
        self.append_message("user", f"[Generate graph]: {prompt}")
        self.status_label.setText("Generating seed nodes...")
        self.status_label.setStyleSheet("color: #cccc55; font-size: 10px;")

        def on_seed_done(data):
            before = set(self._script.nodes.keys())
            self._script.apply_generated(data)
            after  = set(self._script.nodes.keys())
            seed_ids = after - before
            if seed_ids and self._on_nodes_incremental:
                try:
                    self._on_nodes_incremental(seed_ids)
                except Exception as exc:
                    import traceback; traceback.print_exc()
                    self.status_label.setText(f"Graph update error: {exc}")
            names = ', '.join(sorted(seed_ids))
            self.append_message("assistant", f"Seed: {names}")
            self.status_label.setText(f"Seed done ({len(seed_ids)} nodes). Expanding...")
            self._expand_leaves(set(), 0, generation_set=seed_ids)

        def on_seed_error(e):
            self.status_label.setText(f"Error: {e[:50]}")
            self.status_label.setStyleSheet("color: #ff5555; font-size: 10px;")
            self.append_message("assistant", f"Seed error: {e}")

        self._ai.generate_seed(
            prompt, self._ui_queue, on_seed_done, on_seed_error,
            story_context=self._script.story_context_focused if self._script else '',
        )

    def _expand_leaves(self, expanded_ids: set, total_calls: int, generation_set: set = None):
        """Expand one unexpanded leaf node within generation_set, then recurse."""
        TERMINAL_LAYERS = {'resolution', 'descent'}
        MAX_CALLS = 10

        if total_calls >= MAX_CALLS:
            self.status_label.setText("Generation complete (call limit).")
            self.status_label.setStyleSheet("color: #88ee88; font-size: 10px;")
            return

        # Only consider nodes created in this generation session
        candidate_ids = generation_set if generation_set is not None else set(self._script.nodes.keys())

        leaves = [
            nid for nid in candidate_ids
            if nid in self._script.nodes
            and not self._script.nodes[nid].get('next')
            and nid not in expanded_ids
            and not any(t in TERMINAL_LAYERS for t in self._script.nodes[nid].get('tags', []))
        ]

        if not leaves:
            count = len(generation_set) if generation_set is not None else len(self._script.nodes)
            self.status_label.setText(f"Complete — {count} new nodes, {total_calls} expansions.")
            self.status_label.setStyleSheet("color: #88ee88; font-size: 10px;")
            self.append_message("assistant",
                f"Graph complete: {count} new nodes across {total_calls} expansions.")
            # Final full rebuild to sync everything cleanly
            if self._on_graph_generated:
                self._on_graph_generated()
            return

        nid = leaves[0]
        nd  = self._script.nodes[nid]
        self.status_label.setText(
            f"Expanding '{nid}'... ({total_calls + 1}/{MAX_CALLS})"
        )

        layer_tags = {"intro","opening","development","deepening",
                      "bridge","turn","descent","resolution"}
        existing_custom_tags = list({
            t for n in self._script.nodes.values()
            for t in n.get('tags', []) if t not in layer_tags
        })

        def on_done(data):
            before = set(self._script.nodes.keys())
            # Only allow new nodes to connect to each other, not to pre-existing nodes.
            new_node_ids = set(data.get('nodes', {}).keys())
            for nd_data in data.get('nodes', {}).values():
                nd_data['next'] = [n for n in nd_data.get('next', [])
                                   if n in new_node_ids]
            self._script.apply_expansion(nid, data)
            after   = set(self._script.nodes.keys())
            new_ids = after - before
            # Incremental update: only add the new nodes/edges to the live graph.
            if new_ids and self._on_nodes_incremental:
                try:
                    self._on_nodes_incremental(new_ids)
                except Exception as exc:
                    import traceback; traceback.print_exc()
                    self.status_label.setText(f"Graph update error: {exc}")
            self.append_message("assistant",
                f"  '{nid}' → {', '.join(sorted(new_ids)) or '(no new nodes)'}")
            self._expand_leaves(
                expanded_ids | {nid}, total_calls + 1,
                generation_set=(generation_set | new_ids) if generation_set is not None else None,
            )

        def on_error(e):
            self.status_label.setText(f"Error expanding '{nid}': {e[:40]}")
            self.append_message("assistant", f"  Error on '{nid}': {e[:60]}")
            self._expand_leaves(expanded_ids | {nid}, total_calls + 1, generation_set=generation_set)

        self._ai.expand_node(
            nid, nd.get('text', ''), nd.get('tags', []),
            hint='',
            ui_queue=self._ui_queue,
            on_done=on_done,
            on_error=on_error,
            story_context=self._script.story_context_focused if self._script else '',
            node_min=2, node_max=3,
            existing_custom_tags=existing_custom_tags,
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
    """Hierarchical tree: roots on left, children spread vertically beside parent."""
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

    NODE_W, NODE_H = 260, 115

    # Count UNIQUE leaf nodes reachable from each node.
    # Using frozensets prevents double-counting when DAG paths converge.
    reachable: dict = {}
    for nid in reversed(topo):
        kids = [c for c in children[nid] if c in nodes]
        if not kids:
            reachable[nid] = frozenset([nid])
        else:
            s: frozenset = frozenset()
            for k in kids:
                s = s | reachable.get(k, frozenset([k]))
            reachable[nid] = s
    leaf_count = {nid: max(1, len(reachable[nid])) for nid in nodes}

    # Allocate a contiguous y range to every node.
    # Roots get sequential ranges with a small gap between them.
    # Each node divides its range among children proportionally.
    ROOT_GAP = 0.25
    y_ranges: dict = {}
    y_cursor = 0.0
    for nid in topo:
        if not parents[nid]:
            size = leaf_count[nid]
            y_ranges[nid] = (y_cursor, y_cursor + size)
            y_cursor += size + ROOT_GAP

    for nid in topo:
        if nid not in y_ranges:
            continue
        y_lo, y_hi = y_ranges[nid]
        kids = [c for c in children[nid] if c in nodes]
        if not kids:
            continue
        counts = [leaf_count.get(k, 1) for k in kids]
        total  = sum(counts) or 1
        span   = y_hi - y_lo
        y = y_lo
        for kid, cnt in zip(kids, counts):
            kid_hi = y + span * cnt / total
            if kid not in y_ranges:
                y_ranges[kid] = (y, kid_hi)
            y = kid_hi

    # Place each node at the midpoint of its allocated range
    y_pos: dict = {}
    for nid in nodes:
        if nid in y_ranges:
            lo, hi = y_ranges[nid]
            y_pos[nid] = (lo + hi) / 2
        else:
            y_pos[nid] = y_cursor
            y_cursor += 1

    # Collision pass: ensure at least 1 slot gap within each depth column
    by_depth: dict = defaultdict(list)
    for nid in nodes:
        by_depth[depth[nid]].append(nid)
    for col_nodes in by_depth.values():
        col_nodes.sort(key=lambda n: y_pos.get(n, 0))
        for i in range(1, len(col_nodes)):
            min_y = y_pos[col_nodes[i - 1]] + 0.85
            if y_pos[col_nodes[i]] < min_y:
                y_pos[col_nodes[i]] = min_y

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
                 on_delete_pipes=None, on_mouse_move=None, parent=None):
        super().__init__(parent)
        self._get_node = get_node_at_pos
        self._on_enter = on_enter
        self._on_leave = on_leave
        self._on_right_click = on_right_click
        self._on_delete_pipes = on_delete_pipes
        self._on_mouse_move = on_mouse_move
        self._current: Optional[str] = None

    def eventFilter(self, obj, event):
        t = event.type()
        if t == QEvent.Type.MouseMove:
            if self._on_mouse_move:
                viewer = obj.parent() if hasattr(obj, 'parent') else None
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
        elif t == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.RightButton and self._on_right_click:
                node_id = self._get_node(event.position().toPoint())
                self._on_right_click(node_id, event.globalPosition().toPoint())
                return True  # suppress NodeGraphQt's built-in right-click menu
            if self._current:
                self._on_leave(self._current)
                self._current = None
        elif t == QEvent.Type.Leave:
            if self._current:
                self._on_leave(self._current)
                self._current = None
        elif t == QEvent.Type.KeyPress and self._on_delete_pipes:
            from PySide6.QtCore import Qt as _Qt
            if event.key() in (_Qt.Key.Key_Delete, _Qt.Key.Key_Backspace):
                if self._on_delete_pipes():
                    return True  # consumed — don't let NodeGraphQt delete nodes
        return False


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

        # Set contexts
        self.props_panel.set_context(self.script, self.vm, self.ai, self.ui_queue)
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
        self._search_bar.setPlaceholderText("Search nodes by text, label, or ID...  (Ctrl+/)")
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
        act_ctx = QAction("Story Context…", self)
        act_ctx.setShortcut("Ctrl+Shift+C")
        act_ctx.triggered.connect(self._cmd_open_story_context)
        story_menu.addAction(act_ctx)

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
        right_lbl = QLabel("Focused Context  (seen by AI — keep under 1000 characters)")
        right_lbl.setStyleSheet("color: #aaddaa; font-size: 10px; font-weight: bold;")
        right_layout.addWidget(right_lbl)
        focused_edit = QTextEdit()
        focused_edit.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        focused_edit.setPlainText(self.script.story_context_focused)
        right_layout.addWidget(focused_edit)
        focused_char_lbl = QLabel(f"{len(self.script.story_context_focused)} characters")
        focused_char_lbl.setStyleSheet("color: #888888; font-size: 10px;")
        focused_edit.textChanged.connect(
            lambda: focused_char_lbl.setText(f"{len(focused_edit.toPlainText())} characters"))
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

    def _on_port_connected(self, in_port, out_port):
        from_id = next((nid for nid, n in self._node_items.items() if n is out_port.node()), None)
        to_id   = next((nid for nid, n in self._node_items.items() if n is in_port.node()), None)
        if not from_id or not to_id:
            return
        self.script.add_edge(from_id, to_id)
        self.props_panel.rebuild_edge_list(self.script, self._selected_node_id or from_id)
        self._update_title()
        self._maybe_refresh_freq()
        self._refresh_cycle_markers()
        self.status_bar.showMessage(f"Edge: {from_id} -> {to_id}")

    def _on_port_disconnected(self, in_port, out_port):
        from_id = next((nid for nid, n in self._node_items.items() if n is out_port.node()), None)
        to_id   = next((nid for nid, n in self._node_items.items() if n is in_port.node()), None)
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
                    if to_id in self._node_items:
                        self._node_items[from_id].output(0).connect_to(
                            self._node_items[to_id].input(0)
                        )
        finally:
            self.graph.port_connected.connect(self._on_port_connected)
            self.graph.port_disconnected.connect(self._on_port_disconnected)
            self.graph.nodes_deleted.connect(self._on_nodes_deleted)
        # Reset hover filter's current node after rebuild
        if hasattr(self, '_graph_hover_filter'):
            self._graph_hover_filter._current = None
        self._refresh_cycle_markers()

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
        """Return (parent_ids, child_ids) directly connected to node_id."""
        node = self._node_items.get(node_id)
        if not node:
            return set(), set()
        def nid_of(n):
            return next((k for k, v in self._node_items.items() if v is n), None)
        parents  = {nid_of(p.node()) for p in node.input(0).connected_ports()}
        children = {nid_of(p.node()) for p in node.output(0).connected_ports()}
        return parents - {None}, children - {None}

    def _pipe_connections(self):
        """Yield (pipe_item, from_node_id, to_node_id) for every pipe in the scene."""
        viewer = self.graph.viewer()
        for item in viewer.scene().items():
            if 'Pipe' not in type(item).__name__:
                continue
            in_port  = getattr(item, 'input_port',  None)
            out_port = getattr(item, 'output_port', None)
            if in_port is None or out_port is None:
                continue
            in_view  = in_port.parentItem()
            out_view = out_port.parentItem()
            from_nid = to_nid = None
            for nid, node in self._node_items.items():
                if node.view is out_view:
                    from_nid = nid
                if node.view is in_view:
                    to_nid = nid
            if from_nid and to_nid:
                yield item, from_nid, to_nid

    def _apply_highlight(self, node_id: str):
        """Multi-level opacity fade: focus → 1st order → 2nd order → unrelated.
        Traversal is strictly directional: upstream follows inputs only,
        downstream follows outputs only, so siblings are never included."""

        def nid_of(n):
            return next((k for k, v in self._node_items.items() if v is n), None)

        def upstream(nid):
            n = self._node_items.get(nid)
            if not n:
                return set()
            return {r for p in n.input(0).connected_ports() if (r := nid_of(p.node()))}

        def downstream(nid):
            n = self._node_items.get(nid)
            if not n:
                return set()
            return {r for p in n.output(0).connected_ports() if (r := nid_of(p.node()))}

        up1   = upstream(node_id)
        down1 = downstream(node_id)
        first = up1 | down1

        up2   = set()
        for nid in up1:
            up2 |= upstream(nid)
        down2 = set()
        for nid in down1:
            down2 |= downstream(nid)
        second = (up2 | down2) - first - {node_id}

        up3   = set()
        for nid in up2:
            up3 |= upstream(nid)
        down3 = set()
        for nid in down2:
            down3 |= downstream(nid)
        third = (up3 | down3) - second - first - {node_id}

        up4   = set()
        for nid in up3:
            up4 |= upstream(nid)
        down4 = set()
        for nid in down3:
            down4 |= downstream(nid)
        fourth = (up4 | down4) - third - second - first - {node_id}

        up5   = set()
        for nid in up4:
            up5 |= upstream(nid)
        down5 = set()
        for nid in down4:
            down5 |= downstream(nid)
        fifth = (up5 | down5) - fourth - third - second - first - {node_id}

        up6   = set()
        for nid in up5:
            up6 |= upstream(nid)
        down6 = set()
        for nid in down5:
            down6 |= downstream(nid)
        sixth = (up6 | down6) - fifth - fourth - third - second - first - {node_id}

        up7   = set()
        for nid in up6:
            up7 |= upstream(nid)
        down7 = set()
        for nid in down6:
            down7 |= downstream(nid)
        seventh = (up7 | down7) - sixth - fifth - fourth - third - second - first - {node_id}

        up8   = set()
        for nid in up7:
            up8 |= upstream(nid)
        down8 = set()
        for nid in down7:
            down8 |= downstream(nid)
        eighth = (up8 | down8) - seventh - sixth - fifth - fourth - third - second - first - {node_id}

        up9   = set()
        for nid in up8:
            up9 |= upstream(nid)
        down9 = set()
        for nid in down8:
            down9 |= downstream(nid)
        ninth = (up9 | down9) - eighth - seventh - sixth - fifth - fourth - third - second - first - {node_id}

        highlighted = first | second | third | fourth | fifth | sixth | seventh | eighth | ninth | {node_id}

        for nid, n in self._node_items.items():
            if nid == node_id:
                n.view.setOpacity(1.0)
            elif nid in first:
                n.view.setOpacity(1.0)
            elif nid in second:
                n.view.setOpacity(0.90)
            elif nid in third:
                n.view.setOpacity(0.78)
            elif nid in fourth:
                n.view.setOpacity(0.67)
            elif nid in fifth:
                n.view.setOpacity(0.57)
            elif nid in sixth:
                n.view.setOpacity(0.47)
            elif nid in seventh:
                n.view.setOpacity(0.38)
            elif nid in eighth:
                n.view.setOpacity(0.31)
            elif nid in ninth:
                n.view.setOpacity(0.25)
            else:
                n.view.setOpacity(0.07)

        for pipe, from_nid, to_nid in self._pipe_connections():
            if from_nid in highlighted and to_nid in highlighted:
                pipe.setOpacity(1.0)
            else:
                pipe.setOpacity(0.05)

    def _clear_highlight(self):
        """Restore opacity, respecting active search or frequency map."""
        if self._search_bar.text().strip():
            self._cmd_search(self._search_bar.text())
            return
        if self._freq_btn.isChecked():
            self._apply_frequency_heat()
            return
        for n in self._node_items.values():
            n.view.setOpacity(1.0)
        for pipe, _, _ in self._pipe_connections():
            pipe.setOpacity(1.0)

    def _on_node_hover_enter(self, node_id: str):
        self._apply_highlight(node_id)
        self.props_panel.preview_node(self.script, node_id)

    def _on_node_hover_leave(self, _node_id: str):
        if self._selected_node_id:
            self._apply_highlight(self._selected_node_id)
        elif self._search_bar.text().strip():
            self._cmd_search(self._search_bar.text())
        elif self._freq_btn.isChecked():
            self._apply_frequency_heat()
        else:
            self._clear_highlight()
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
                act_gen.triggered.connect(self.chat_panel._cmd_generate_graph)

            menu.addSeparator()
            act_freq = menu.addAction("Toggle Frequency Heat Map")
            act_freq.triggered.connect(lambda: self._freq_btn.setChecked(not self._freq_btn.isChecked()))

            menu.addSeparator()
            act_fit = menu.addAction("Fit View")
            act_fit.triggered.connect(self._cmd_fit_view)

        menu.exec(global_pos)

    def _select_node(self, node_id: str):
        node = self._node_items.get(node_id)
        if node:
            self.graph.clear_selection()
            node.set_selected(True)
            self.props_panel.load_node(self.script, node_id)
            self._selected_node_id = node_id
            self._apply_highlight(node_id)

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
        """Called after AI generates a graph."""
        self._rebuild_graph()
        self._update_title()
        self._maybe_refresh_freq()
        self._refresh_cycle_markers()
        self.status_bar.showMessage(f"Graph updated: {len(self.script.nodes)} nodes")

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
            # Also wire edges FROM existing nodes TO new nodes (parent → new child)
            for from_id, nd in self.script.nodes.items():
                if from_id in new_node_ids:
                    continue
                for to_id in nd.get('next', []):
                    if to_id in new_node_ids and from_id in self._node_items:
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

    def _cmd_expand_node(self, node_id: str, node_min: int = 2, node_max: int = 5):
        if not node_id or node_id not in self.script.nodes:
            self.status_bar.showMessage("Select a node to expand")
            return
        if not self.ai.ready:
            self.status_bar.showMessage("claude CLI not found")
            return
        if self.ai.busy:
            self.status_bar.showMessage("AI is busy...")
            return

        nd = self.script.nodes[node_id]
        hint = self.chat_panel.chat_input.toPlainText().strip()

        _layer_tags = {"intro","opening","development","deepening","bridge","turn","descent","resolution"}
        existing_custom_tags = sorted({
            t for n in self.script.nodes.values()
            for t in n.get("tags", [])
            if t not in _layer_tags
        })

        self.status_bar.showMessage(f"Expanding '{node_id}'...")

        def on_done(data):
            n = len(data.get("nodes", {}))
            self.script.apply_expansion(node_id, data)
            for nid, pos in _layout_tree(self.script).items():
                self.script.update_pos(nid, pos)
            self._rebuild_graph()
            if node_id in self.script.nodes:
                self._select_node(node_id)
            self._update_title()
            self.status_bar.showMessage(f"Expanded '{node_id}' -> {n} new nodes")
            self.chat_panel.append_message("assistant",
                f"Expanded '{node_id}' with {n} new nodes.")

        def on_error(e):
            self.status_bar.showMessage(f"Expand error: {e[:60]}")
            self.chat_panel.append_message("assistant", f"[Expand error] {e}")

        self.ai.expand_node(
            source_id=node_id,
            source_text=nd.get("text", ""),
            source_tags=nd.get("tags", []),
            hint=hint,
            ui_queue=self.ui_queue,
            on_done=on_done,
            on_error=on_error,
            story_context=self.script.story_context_focused,
            node_hint=nd.get("hint", ""),
            upstream_path=self._get_upstream_path(node_id),
            node_min=node_min,
            node_max=node_max,
            existing_custom_tags=existing_custom_tags,
        )

    def _cmd_continue_from_node(self, node_id: str):
        """Generate 2-4 forward layers from an existing node and wire them to it."""
        if not node_id or node_id not in self.script.nodes:
            self.status_bar.showMessage("Select a node to continue from")
            return
        if not self.ai.ready:
            self.status_bar.showMessage("claude CLI not found")
            return
        if self.ai.busy:
            self.status_bar.showMessage("AI is busy...")
            return

        nd = self.script.nodes[node_id]
        hint = self.chat_panel.chat_input.toPlainText().strip()
        self.status_bar.showMessage(f"Continuing from '{node_id}'...")

        def on_done(data):
            # Reuse apply_expansion: treat start_nodes as connect_from targets
            expansion_data = dict(data)
            expansion_data['connect_from'] = data.get('start_nodes', [])
            self.script.apply_expansion(node_id, expansion_data)
            for nid, pos in _layout_tree(self.script).items():
                self.script.update_pos(nid, pos)
            self._rebuild_graph()
            self._select_node(node_id)
            n = len(data.get("nodes", {}))
            self._update_title()
            self.status_bar.showMessage(f"Continued '{node_id}' → {n} new nodes")
            self.chat_panel.append_message("assistant",
                f"Continued from '{node_id}' with {n} new nodes.")

        def on_error(e):
            self.status_bar.showMessage(f"Continue error: {e[:60]}")
            self.chat_panel.append_message("assistant", f"[Continue error] {e}")

        self.ai.continue_from_node(
            source_id=node_id,
            source_text=nd.get("text", ""),
            source_tags=nd.get("tags", []),
            ui_queue=self.ui_queue,
            on_done=on_done,
            on_error=on_error,
            story_context=self.script.story_context_focused,
            node_hint=nd.get("hint", "") or hint,
        )

    def _run_frequency_simulation(self, n_runs: int = 2000) -> dict:
        """Monte Carlo random walk. Returns {node_id: visit_count}."""
        nodes = self.script.nodes
        starts = self.script.start_nodes or list(nodes.keys())
        counts = {nid: 0 for nid in nodes}
        for _ in range(n_runs):
            current = random.choice(starts)
            steps = 0
            while current and steps < 300:
                if current not in nodes:
                    break
                counts[current] += 1
                nd = nodes[current]
                nexts = nd.get('next', [])
                weights = nd.get('weights', [1.0] * len(nexts))
                if not nexts:
                    break
                total = sum(weights) or 1.0
                r = random.random() * total
                acc = 0.0
                nxt = nexts[-1]
                for nid, w in zip(nexts, weights):
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
        """Highlight nodes whose text/label/ID contains the search string."""
        if not text.strip():
            self._clear_highlight()
            return
        term = text.strip().lower()
        matched = set()
        for nid, nd in self.script.nodes.items():
            haystack = ' '.join([
                nid,
                nd.get('label', '') or '',
                nd.get('text', '') or '',
                ' '.join(nd.get('tags', [])),
            ]).lower()
            if term in haystack:
                matched.add(nid)
        for nid, n in self._node_items.items():
            n.view.setOpacity(1.0 if nid in matched else 0.08)
        for pipe, from_nid, to_nid in self._pipe_connections():
            pipe.setOpacity(1.0 if (from_nid in matched and to_nid in matched) else 0.04)
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
