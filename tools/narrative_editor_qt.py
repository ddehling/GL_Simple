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
    ('\u2014', ' - '),   # em dash  —
    ('\u2013', ' - '),   # en dash  –
    ('\u2026', '...'),   # ellipsis …
    ('\u2018', "'"),     # left single quote  '
    ('\u2019', "'"),     # right single quote '
    ('\u201c', '"'),     # left double quote  "
    ('\u201d', '"'),     # right double quote "
    ('\u00a0', ' '),     # non-breaking space
    ('\u200b', ''),      # zero-width space
    ('\u2022', '-'),     # bullet •
]

def _sanitize_tts(text: str) -> str:
    """Replace characters that confuse TTS engines with safe equivalents."""
    for bad, good in _TTS_REPLACEMENTS:
        text = text.replace(bad, good)
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

LAYER STRUCTURE (adapt depth to the content):
  Layer 1 (opening)   : 1–3 nodes  — establish tone/setting
  Layer 2 (development): 3–6 nodes — explore the theme, each from a different angle
  Layer 3 (turn)      : 2–4 nodes  — complication, deepening, or shift in feeling
  Layer 4 (resolution): 1–3 nodes  — landing, conclusion, or open question

BRANCHING: any node in layer N may connect to 2–4 nodes in layer N+1.
MERGING:   multiple nodes in layer N may all point to the same node in layer N+1.
           This creates convergence points — moments every path passes through.

Good pattern:
  intro → [dev_a, dev_b, dev_c]        ← branch
  dev_a, dev_b → [turn_x]              ← merge
  dev_c        → [turn_y]
  turn_x, turn_y → [close_a, close_b]  ← merge then branch again

Avoid:
  - Fully connected pools where every node points to every other node
  - Trees that only branch and never merge (too many dead-end leaves)
  - Chains with no branching at all (boring, no variation)

WEIGHTS: use 1.0 as default. Use 2.0 to favour a path, 0.5 to make it rare.
TAGS: label each node's layer role: "intro", "development", "turn", "resolution", "bridge"
node IDs: short_snake_case, layer-prefixed (e.g. "intro_storm", "dev_pride", "close_silence")

VOICE SETTINGS: set "voice_settings" on every node to match its emotional tone:
  stability      0.0–1.0  lower = more expressive/varied delivery
  similarity_boost         leave at 0.75 unless noted
  style          0.0–1.0  higher = more dramatic/theatrical

  Layer defaults:
    intro       stability 0.65  style 0.15  (calm, scene-setting)
    development stability 0.50  style 0.35  (engaged, exploring)
    bridge      stability 0.45  style 0.40  (transitional energy)
    turn        stability 0.30  style 0.65  (tense, expressive)
    resolution  stability 0.60  style 0.15  (settled, reflective)
  Adjust within layer if the content is notably more or less intense than usual.

TERMINAL NODES: resolution/ending nodes must have next: [].
NEVER create edges that point back toward intro or start nodes.
When a terminal node finishes playing, the runtime will automatically restart
from a randomly chosen start_node — no explicit loop edges are needed or wanted.
"""

SYSTEM_EXPAND = """\
You are expanding a single node in a narrative graph for an immersive audio installation.
You will receive one existing node and must generate new nodes that continue FROM it.

Each new node is a short spoken segment (40–100 words, ~15–35 seconds when read aloud).
Use evocative, atmospheric language consistent with the source node's tone and theme.

Respond with ONLY a JSON object — no markdown fences, no explanation:
{
  "nodes": {
    "node_id": {
      "text": "Spoken text, 40-100 words.",
      "next": [],
      "weights": [],
      "tags": ["development"],
      "voice_settings": {"stability": 0.50, "similarity_boost": 0.75, "style": 0.35}
    }
  },
  "connect_from": ["new_node_id_1", "new_node_id_2"]
}

"connect_from": the new node IDs that the SOURCE node should gain edges to.
All other edges are between the new nodes themselves.

Layer progression rules:
- Determine the source node's layer from its tags (intro→development→turn→resolution)
- New nodes should be in the NEXT layer
- You may branch (source → multiple new nodes) or chain (source → A → B → C → ...)
- Branching then merging is encouraged: source → [A, B] and both A, B → C
- 2–5 new nodes is typical; more for a rich expansion, fewer for a tight chain
- node IDs: short_snake_case, layer-prefixed (e.g. "dev_kelp_drift", "turn_silence")
- Weights default to 1.0 unless you have reason to favour one path
- Terminal nodes (resolution/end) must have next: [] — NEVER link back to intro or start nodes.
  The runtime restarts automatically from a random start node when a terminal finishes.

VOICE SETTINGS: set voice_settings on every node (stability 0-1, similarity_boost 0.75, style 0-1).
  intro stability~0.65 style~0.15 | development stability~0.50 style~0.35
  turn  stability~0.30 style~0.65 | resolution  stability~0.60 style~0.15
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
- Layer guidance: intro stability~0.65 style~0.15 | development stability~0.50 style~0.35
                  turn stability~0.30 style~0.65 | resolution stability~0.60 style~0.15
- Adjust to match the specific emotional intensity of the rewritten text

Respond with ONLY a JSON object in this exact format, no other text:
{"text": "...", "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.3}}
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

    def set_story_context(self, text: str):
        self._data["story_context"] = text
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
        # Ensure all nodes have required fields
        for node in sd._data["nodes"].values():
            node.setdefault("pos",      [100, 100])
            node.setdefault("next",     [])
            node.setdefault("weights",  [])
            node.setdefault("tags",     [])
            node.setdefault("file",     None)
            node.setdefault("duration", None)
            node.setdefault("voice",          None)
            node.setdefault("voice_settings", {})
        return sd

    @staticmethod
    def _sanitize_id(nid: str) -> str:
        return nid.replace('-', '_')

    @staticmethod
    def _sanitize_nodes(nodes: dict) -> dict:
        """Return a copy of `nodes` with all node IDs and next-references de-dashed."""
        clean = {}
        for nid, nd in nodes.items():
            safe_id = ScriptData._sanitize_id(nid)
            safe_nd = dict(nd)
            safe_nd['next'] = [ScriptData._sanitize_id(n) for n in nd.get('next', [])]
            clean[safe_id] = safe_nd
        return clean

    def apply_generated(self, generated: dict):
        """Merge AI-generated graph into current script (additive).

        Positions nodes left-to-right by layer using tag hints:
        intro < development < bridge < turn < resolution
        Nodes within a layer are stacked vertically.
        """
        generated = dict(generated)
        generated['nodes']       = self._sanitize_nodes(generated.get('nodes', {}))
        generated['start_nodes'] = [self._sanitize_id(n) for n in generated.get('start_nodes', [])]
        if self._data["name"] == "New Script" and generated.get("name"):
            self._data["name"] = generated["name"]
        if generated.get("description"):
            self._data["description"] = generated["description"]

        LAYER_ORDER = ["intro", "development", "bridge", "turn", "resolution"]
        LAYER_X     = {name: 80 + i * 310 for i, name in enumerate(LAYER_ORDER)}
        LAYER_X["_default"] = 80 + len(LAYER_ORDER) * 310

        # Count how many nodes already occupy each layer column (for vertical stacking)
        layer_counts: dict = {}

        new_nodes = generated.get("nodes", {})

        for nid, ndata in new_nodes.items():
            # Never overwrite an existing node
            if nid in self._data["nodes"]:
                continue

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
        expansion['nodes']        = self._sanitize_nodes(expansion.get('nodes', {}))
        expansion['connect_from'] = [self._sanitize_id(n) for n in expansion.get('connect_from', [])]
        LAYER_ORDER = ["intro", "development", "bridge", "turn", "resolution"]
        LAYER_X     = {name: 80 + i * 310 for i, name in enumerate(LAYER_ORDER)}
        LAYER_X["_default"] = 80 + len(LAYER_ORDER) * 310

        layer_counts: dict = {}
        for nd in self._data["nodes"].values():
            for tag in nd.get("tags", []):
                if tag in LAYER_ORDER:
                    layer_counts[tag] = layer_counts.get(tag, 0) + 1

        for nid, ndata in expansion.get("nodes", {}).items():
            # Never overwrite an existing node — the AI may echo back the source node
            if nid in self._data["nodes"]:
                continue

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

    def expand_node(self, source_id: str, source_text: str, source_tags: list,
                    hint: str, ui_queue: queue.SimpleQueue, on_done, on_error,
                    story_context: str = '', node_hint: str = '',
                    upstream_path: list = None):
        if self._busy:
            return
        self._busy = True

        parts = []
        if story_context:
            parts.append(f'Story context:\n{story_context}\n')
        if upstream_path:
            path_str = '\n'.join(f'  [{nid}]: "{text}"'
                                 for nid, text in upstream_path)
            parts.append(f'Upstream path leading to source node:\n{path_str}\n')
        parts.append(
            f'Source node: "{source_id}"\n'
            f'Tags: {source_tags}\n'
            f'Text: "{source_text}"'
        )
        if node_hint:
            parts.append(f'\nNode hint: {node_hint}')
        if hint:
            parts.append(f'\nGuidance: {hint}')
        parts.append('\nGenerate continuation nodes branching from this node.')
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
    'development': (50,  140, 70),
    'bridge':      (140, 100, 50),
    'turn':        (180, 80,  50),
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
        self.text_edit.setMinimumHeight(80)
        self.text_edit.setMaximumHeight(150)
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

        # Tags
        tags_row = QHBoxLayout()
        tags_row.addWidget(QLabel("Tags:"))
        self.tags_edit = QLineEdit()
        self.tags_edit.setPlaceholderText("intro, development, ...")
        self.tags_edit.textChanged.connect(self._autosave_tags)
        tags_row.addWidget(self.tags_edit)
        layout.addLayout(tags_row)

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

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("color: #555;")
        layout.addWidget(sep2)

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

        expand_btn = QPushButton("Expand Node (AI)")
        expand_btn.clicked.connect(self._cmd_expand)
        layout.addWidget(expand_btn)

        del_btn = QPushButton("Delete Node")
        del_btn.setStyleSheet("background-color: #8B2222; color: white;")
        del_btn.clicked.connect(self._cmd_delete)
        layout.addWidget(del_btn)

        layout.addStretch(1)

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
            self.tags_edit.setText(", ".join(nd.get("tags", [])))
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

    def clear(self):
        self._node_id = None
        self._blocking = True
        try:
            self.id_edit.setText("")
            self.label_edit.setText("")
            self.text_edit.setPlainText("")
            self.hint_edit.setPlainText("")
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
        raw = self.tags_edit.text()
        self._script.update_tags(self._node_id,
                                  [t.strip() for t in raw.split(",") if t.strip()])
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
        parts.append(f"NODE TAGS: {', '.join(nd.get('tags', [])) or 'none'}")
        if hint:
            parts.append(f"GUIDANCE: {hint}")
        parts.append("Rewrite the spoken text for this node based on the above context and guidance.")
        prompt = "\n\n".join(parts)

        self.rewrite_status.setText("Working...")
        self.rewrite_status.setStyleSheet("color: #cccc55; font-size: 10px;")
        node_id = self._node_id

        def on_done(data):
            new_text = data.get("text", "").strip()
            vs       = data.get("voice_settings", {})
            if self._script and node_id in self._script.nodes:
                self._script.update_text(node_id, new_text)
                if self._node_id == node_id:
                    self._blocking = True
                    try:
                        self.text_edit.setPlainText(new_text)
                    finally:
                        self._blocking = False
                if vs:
                    self._script.update_node_voice_settings(node_id, {
                        "stability":        vs.get("stability",        0.5),
                        "similarity_boost": vs.get("similarity_boost", 0.75),
                        "style":            vs.get("style",            0.3),
                    })
                    if self._node_id == node_id:
                        self._blocking = True
                        try:
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
                              story_context=self._script.story_context)

    def _cmd_expand(self):
        if not self._node_id or not self._script or not self._ai or not self._ui_queue:
            return
        # Signal parent to handle expansion
        self.node_modified.emit(f"__expand__{self._node_id}")

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
        self.selected_node_id: Optional[str] = None
        self._build_ui()

    def set_context(self, script: ScriptData, ai: AIAssistant,
                    ui_queue: queue.SimpleQueue, on_graph_generated):
        self._script = script
        self._ai = ai
        self._ui_queue = ui_queue
        self._on_graph_generated = on_graph_generated

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
            story_context=self._script.story_context if self._script else '',
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
        self.status_label.setText("Generating graph...")
        self.status_label.setStyleSheet("color: #cccc55; font-size: 10px;")

        def on_done(data):
            n = len(data.get("nodes", {}))
            self._script.apply_generated(data)
            for nid, pos in _compute_layout(self._script).items():
                self._script.update_pos(nid, pos)
            if self._on_graph_generated:
                self._on_graph_generated()
            self.status_label.setText(f"Added {n} nodes")
            self.status_label.setStyleSheet("color: #88ee88; font-size: 10px;")
            self.append_message("assistant",
                f"Generated {n} nodes: " +
                ", ".join(list(data.get("nodes", {}).keys())[:8]))

        def on_error(e):
            self.status_label.setText(f"Error: {e[:50]}")
            self.status_label.setStyleSheet("color: #ff5555; font-size: 10px;")
            self.append_message("assistant", f"Error: {e}")

        self._ai.generate_graph(prompt, self._ui_queue, on_done, on_error,
                                story_context=self._script.story_context if self._script else '')

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


# ─────────────────────────────────────────────────────────────────────────────
# MainWindow
# ─────────────────────────────────────────────────────────────────────────────

class _GraphViewHoverFilter(QObject):
    """Single event filter on the NodeGraph viewer that tracks which node the
    mouse is over and fires enter/leave callbacks as it changes.
    Also intercepts right-click to fire on_right_click(node_id, global_pos)."""

    def __init__(self, get_node_at_pos, on_enter, on_leave, on_right_click=None, parent=None):
        super().__init__(parent)
        self._get_node = get_node_at_pos
        self._on_enter = on_enter
        self._on_leave = on_leave
        self._on_right_click = on_right_click
        self._current: Optional[str] = None

    def eventFilter(self, obj, event):
        t = event.type()
        if t == QEvent.Type.MouseMove:
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

        self._build_ui()
        self._build_menu()

        # Connect NodeGraph signals
        self.graph.node_selected.connect(self._on_node_selected)
        self.graph.port_connected.connect(self._on_port_connected)
        self.graph.port_disconnected.connect(self._on_port_disconnected)
        self.graph.nodes_deleted.connect(self._on_nodes_deleted)
        self.graph.scene().selectionChanged.connect(self._on_selection_changed)

        # Set contexts
        self.props_panel.set_context(self.script, self.vm, self.ai, self.ui_queue)
        self.voice_panel.set_context(self.script, self.vm, self.ui_queue, self.props_panel)
        self.chat_panel.set_context(self.script, self.ai, self.ui_queue, self._on_graph_generated)
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
        )
        # Events land on the viewport, not the view itself
        self.graph.viewer().viewport().installEventFilter(self._graph_hover_filter)

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

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(graph_widget)
        splitter.addWidget(scroll_area)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

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

        # Voice menu
        voice_menu = menubar.addMenu("Voice")
        act_voice = QAction("Voice Settings…", self)
        act_voice.setShortcut("Ctrl+Shift+V")
        act_voice.triggered.connect(self._cmd_open_voice_settings)
        voice_menu.addAction(act_voice)

    def _cmd_open_story_context(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Story Context")
        dlg.setMinimumWidth(480)
        dlg.setMinimumHeight(300)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(10, 10, 10, 10)
        lbl = QLabel("Global story context — visible to all AI operations (expand, rewrite, generate):")
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        layout.addWidget(lbl)
        edit = QTextEdit()
        edit.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        edit.setPlainText(self.script.story_context)
        layout.addWidget(edit)
        btn = QPushButton("Save & Close")
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
        dlg.exec()
        self.script.set_story_context(edit.toPlainText())
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
                fn()
            except queue.Empty:
                break

    # ── NodeGraph signal handlers ────────────────────────────────────────────

    def _on_node_selected(self, node):
        if node:
            node_id = next((nid for nid, n in self._node_items.items() if n is node), None)
            if not node_id:
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

    def _set_node_color(self, node: NarrativeNode, nd: dict):
        tags = nd.get('tags', [])
        color = next((TAG_COLORS[t] for t in tags if t in TAG_COLORS), (70, 70, 95))
        node.set_color(*color)

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
                self._set_node_color(node, nd)
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

        highlighted = first | second | third | {node_id}

        for nid, n in self._node_items.items():
            if nid == node_id:
                n.view.setOpacity(1.0)
            elif nid in first:
                n.view.setOpacity(1.0)
            elif nid in second:
                n.view.setOpacity(0.75)
            elif nid in third:
                n.view.setOpacity(0.45)
            else:
                n.view.setOpacity(0.12)

        for pipe, from_nid, to_nid in self._pipe_connections():
            if from_nid in highlighted and to_nid in highlighted:
                pipe.setOpacity(1.0)
            else:
                pipe.setOpacity(0.08)

    def _clear_highlight(self):
        """Restore all nodes and pipes to full opacity."""
        for n in self._node_items.values():
            n.view.setOpacity(1.0)
        for pipe, _, _ in self._pipe_connections():
            pipe.setOpacity(1.0)

    def _on_node_hover_enter(self, node_id: str):
        self._apply_highlight(node_id)

    def _on_node_hover_leave(self, _node_id: str):
        # On mouse leave, restore to selection-based highlight if a node is selected
        if self._selected_node_id:
            self._apply_highlight(self._selected_node_id)
        else:
            self._clear_highlight()

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

            menu.addSeparator()

            act_expand = menu.addAction("Expand Node (AI)")
            act_expand.triggered.connect(lambda: self._cmd_expand_node(node_id))
            act_expand.setEnabled(self.ai.ready and not self.ai.busy)

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
            self._set_node_color(node, nd)
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
            node_id = signal[len("__expand__"):]
            self._cmd_expand_node(node_id)
            return
        # Regular modification — refresh appearance
        self._refresh_node(signal)
        self._update_title()

    def _on_graph_generated(self):
        """Called after AI generates a graph."""
        self._rebuild_graph()
        self._update_title()
        self.status_bar.showMessage(f"Graph updated: {len(self.script.nodes)} nodes")

    def _cmd_add_node(self):
        node_id = _next_node_id(self.script)
        existing = [nd.get("pos", [0, 0]) for nd in self.script.nodes.values()]
        max_x = max((p[0] for p in existing), default=60)
        pos = [max_x + 220, 60]
        self.script.add_node(node_id, pos=pos)
        node = self.graph.create_node('narrative.NarrativeNode', name=node_id)
        node.set_pos(float(pos[0]), float(pos[1]))
        self._set_node_color(node, self.script.nodes[node_id])
        self._node_items[node_id] = node
        self._selected_node_id = node_id
        self.props_panel.load_node(self.script, node_id)
        self._update_title()
        self.status_bar.showMessage(f"Added '{node_id}'")

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
        self.status_bar.showMessage(f"Deleted '{nid}'")

    def _cmd_expand_node(self, node_id: str):
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

        self.status_bar.showMessage(f"Expanding '{node_id}'...")

        def on_done(data):
            n = len(data.get("nodes", {}))
            self.script.apply_expansion(node_id, data)
            for nid, pos in _compute_layout(self.script).items():
                self.script.update_pos(nid, pos)
            self._rebuild_graph()
            # Re-select same node
            if node_id in self.script.nodes:
                self.props_panel.load_node(self.script, node_id)
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
            story_context=self.script.story_context,
            node_hint=nd.get("hint", ""),
            upstream_path=self._get_upstream_path(node_id),
        )

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

    def _cmd_fit_view(self):
        layout = _compute_layout(self.script)
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
            self._rebuild_graph()
            self.props_panel.clear()
            self._update_title()
            self.status_bar.showMessage(f"Loaded: {path.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))

    def _refresh_contexts(self):
        """Re-wire all panels after script is replaced."""
        self.props_panel.set_context(self.script, self.vm, self.ai, self.ui_queue)
        self.voice_panel.set_context(self.script, self.vm, self.ui_queue, self.props_panel)
        self.chat_panel.set_context(self.script, self.ai, self.ui_queue, self._on_graph_generated)
        self.play_panel.set_context(self.script, self.ui_queue)

    def _update_title(self):
        if self.script.path:
            name = self.script.path.name
        else:
            name = self.script.name or "New Script"
        dirty = "*" if self.script.dirty else ""
        self.setWindowTitle(f"Narrative Editor — {name}{dirty}")

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
