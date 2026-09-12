"""Perform Copilot: steer the live DJ in plain words.

"darker and slower changes", "more vocals", "drop it", "hold this", "that
was good" - one line in the Perform tab; the copilot reads the live state
(what plays on each lane, the controls, the last moves) and maps the wish
onto the SAME controls the sliders and buttons drive (RemixConductor in
Remix mode, DJSystem in Automix). Every action goes through a bridge the
tab drains on its own thread: the copilot never touches a widget and never
touches the engine directly.

Transport and auth are SetCopilot's (Claude Code CLI first, no key; the
anthropic SDK with a key otherwise) - this class swaps the tools and the
system prompt. Qt-free; the gate drives it with a scripted CLI.
"""
from tools.dj.planner.copilot import SetCopilot

TOOLS = [
    {"name": "get_state",
     "description": "The live state: mode, what plays on each lane (Remix) or the playing / next track (Automix), "
                    "the controls as set, the arc position and energy target, the last moves. Call it first.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "steer",
     "description": "Set one or more controls. Remix: blend 0..1 (0 = song follows song as a morph, 1 = free "
                    "recombination of three songs), vocals 0..1 (how freely the vocal lane crosses), change_bars "
                    "4|8|16|32 (bars between moves: faster = fewer bars), energy_lean -0.4..0.4 (on the theme's "
                    "arc), tempo 0..0.06 (how far the clock may travel with the arc), auto 0..1 (autopilot amount: "
                    "0 = only the operator moves lanes, 1 = the conductor moves every phrase), theme (a theme name). "
                    "Automix: energy_lean, theme, mix_style (a seam style or 'auto'), mix_speed short|normal|long|marathon.",
     "input_schema": {"type": "object", "properties": {
         "blend": {"type": "number"}, "vocals": {"type": "number"}, "change_bars": {"type": "integer"},
         "energy_lean": {"type": "number"}, "tempo": {"type": "number"}, "auto": {"type": "number"}, "theme": {"type": "string"},
         "mix_style": {"type": "string"}, "mix_speed": {"type": "string"}}}},
    {"name": "act",
     "description": "One performance action, on the next bar: next (a move now / the next transition now), hold "
                    "(freeze the combination), unhold, drop (every lane to the newest song / the drop moment), "
                    "break (all lanes but one rest four bars), loop4, loop8, unloop, save (remember this "
                    "combination), recall (bring the last saved one back), rec, rec_stop.",
     "input_schema": {"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"]}},
    {"name": "rate",
     "description": "Rate the last move (Remix) or the last seam (Automix): good true / false. The system learns.",
     "input_schema": {"type": "object", "properties": {"good": {"type": "boolean"}}, "required": ["good"]}},
]

ACTIONS = ("next", "hold", "unhold", "drop", "break", "loop4", "loop8", "unloop", "save", "recall", "rec", "rec_stop")


class PerformCopilot(SetCopilot):
    def __init__(self, bridge, library, theme_name="groove", client=None):
        super().__init__(library, theme_name=theme_name, client=client)
        self.bridge = bridge

    def tools(self):
        return TOOLS

    def system_prompt(self):
        if self._system is None:
            from lib.dj.themes import BUILTIN_THEMES
            self._system = (
                "You steer a LIVE DJ system from one line of the operator's words. Two modes. REMIX: two or three "
                "songs play at once, each of four stem lanes (drums, bass, other, vocals) belongs to one song, the "
                "system makes one move per phrase; the operator's words map onto blend, vocals, change_bars, "
                "energy_lean, tempo, theme and the actions. AUTOMIX: whole songs one seam at a time; words map "
                "onto energy_lean, theme, mix_style, mix_speed and next / hold / drop.\n"
                "Rules: call get_state first. Translate the wish into the fewest control changes that do it "
                "('darker' = energy_lean down and a darker theme; 'slower changes' = more change_bars; 'more "
                "vocals' = vocals up; 'calmer' = blend down and change_bars up; 'wilder' = blend up, change_bars "
                "down; 'faster' = tempo up with energy_lean up; 'that was good/bad' = rate). Apply with steer / "
                "act, never describe what you would do. Then answer in ONE short sentence saying what you set.\n"
                f"Themes: {', '.join(sorted(BUILTIN_THEMES))}.")
        return self._system

    # -- tools --------------------------------------------------------------------------
    def _t_get_state(self, args):
        return self.bridge.state()

    def _t_steer(self, args):
        clean = {}
        for k, v in (args or {}).items():
            if k in ("blend", "vocals", "energy_lean", "tempo", "auto"):
                try:
                    clean[k] = float(v)
                except (TypeError, ValueError):
                    continue
            elif k == "change_bars":
                try:
                    clean[k] = min((4, 8, 16, 32), key=lambda b: abs(b - int(v)))
                except (TypeError, ValueError):
                    continue
            elif k in ("theme", "mix_style", "mix_speed") and isinstance(v, str):
                clean[k] = v[:40]
        if not clean:
            return {"ok": False, "why": "nothing to set"}
        return self.bridge.steer(clean)

    def _t_act(self, args):
        action = str((args or {}).get("action", "")).strip().lower().replace(" ", "")
        if action not in ACTIONS:
            return {"ok": False, "why": f"unknown action; one of {', '.join(ACTIONS)}"}
        return self.bridge.act(action)

    def _t_rate(self, args):
        return self.bridge.rate(bool((args or {}).get("good")))
