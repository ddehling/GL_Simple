"""MIDI performance surface for the console: a Korg nanoKONTROL2 (the
controller lib/midi_controller.py already supports) drives the same
whitelisted actions the widgets do, so the console becomes an instrument.

  sliders 1-8   energy bias, density, swing, brightness, humanize, master,
                reverb lane, low-pass lane
  knobs 1-8     mute depth is not a thing - knobs move the mix lanes:
                hp, lp, duck, verb, delay_fb (1-5); 6-8 spare
  M buttons 1-8 mute kick, snare, hat, bass, lead, pad, arp, keys (toggle)
  S buttons 1-4 gestures: build, breakdown, groove, keep
  S buttons 5-6 modulate up / down; 7 darker; 8 brighter
  R buttons 1-2 feedback up / down; 3 reseed; 4 automation on/off
  transport     play = start, stop = stop, record = end (outro), cycle = hold

No device -> the plugin stays quiet (status bar says so). Values are
only sent when a control moves, so the console and the controller never
fight over a value."""
from __future__ import annotations

import time

SLIDERS = {
    "slider_1": ("energy", -0.5, 0.5), "slider_2": ("density", 0.0, 1.5), "slider_3": ("swing", 0.0, 0.33),
    "slider_4": ("brightness", 0.4, 1.6), "slider_5": ("humanize", 0.0, 1.5), "slider_6": ("master", 0.0, 1.0),
    "slider_7": ("lane:verb", 0.0, 2.5), "slider_8": ("lane:lp", 300.0, 20000.0),
}
KNOBS = {
    "knob_1": ("lane:hp", 10.0, 1500.0), "knob_2": ("lane:lp", 300.0, 20000.0), "knob_3": ("lane:duck", 0.0, 0.9),
    "knob_4": ("lane:verb", 0.0, 2.5), "knob_5": ("lane:delay_fb", 0.0, 0.85),
}
MUTES = {f"m_button_{i + 1}": s for i, s in enumerate(("kick", "snare", "hat", "bass", "lead", "pad", "arp", "keys"))}
S_BUTTONS = {"s_button_1": ("gesture", "build"), "s_button_2": ("gesture", "breakdown"), "s_button_3": ("gesture", "groove"),
             "s_button_4": ("gesture", "keep"), "s_button_5": ("gesture", "modulate_up"), "s_button_6": ("gesture", "modulate_down"),
             "s_button_7": ("gesture", "darker"), "s_button_8": ("gesture", "brighter")}
R_BUTTONS = {"r_button_1": ("feedback", True), "r_button_2": ("feedback", False), "r_button_3": ("reseed", None)}
TRANSPORT = {"play": ("start", None), "stop": ("stop", None), "record": ("end", None)}


class MidiSurface:
    def __init__(self, console):
        self.console = console
        self.ctl = None
        self.muted = set()
        self.auto_on = True
        self.hold = False
        self._last_try = 0.0
        self._connect()

    def _connect(self):
        self._last_try = time.time()
        try:
            from lib.midi_controller import KorgNanoKontrol2
            ctl = KorgNanoKontrol2(auto_connect=True)
            if getattr(ctl, "input_device", None):
                self.ctl = ctl
                self.console.notify("MIDI: nanoKONTROL2 connected")
        except Exception as e:  # noqa: BLE001 - no pygame.midi / no device is normal
            self.ctl = None
            self.error = f"{type(e).__name__}: {e}"

    @staticmethod
    def _scale(v, lo, hi):
        v = max(0.0, min(1.0, float(v)))
        if lo > 0 and hi / lo > 20:                  # frequencies: log scale
            return lo * (hi / lo) ** v
        return lo + (hi - lo) * v

    def emit(self, action, value):
        self.console.ctx.emit(action, value)

    def poll(self, state):
        if self.ctl is None:
            if time.time() - self._last_try > 10.0:
                self._connect()
            return
        try:
            changes = self.ctl.update()
        except Exception:  # noqa: BLE001
            self.ctl = None
            return
        for name, value in (changes or {}).items():
            if name in SLIDERS or name in KNOBS:
                action, lo, hi = (SLIDERS.get(name) or KNOBS.get(name))
                v = self._scale(value if value <= 1.0 else value / 127.0, lo, hi)
                if action.startswith("lane:"):
                    self.emit("lane", {"lane": action[5:], "to": v, "ramp_s": 0.25})
                else:
                    self.emit(action, v)
            elif name in MUTES and value:
                slot = MUTES[name]
                on = slot not in self.muted
                (self.muted.add if on else self.muted.discard)(slot)
                self.emit("mute", {"slot": slot, "on": on})
            elif name in S_BUTTONS and value:
                self.emit(*S_BUTTONS[name])
            elif name in R_BUTTONS and value:
                self.emit(*R_BUTTONS[name])
            elif name == "r_button_4" and value:
                self.auto_on = not self.auto_on
                self.emit("automation", self.auto_on)
            elif name == "cycle" and value:
                self.hold = not self.hold
                self.emit("hold", self.hold)
            elif name in TRANSPORT and value:
                self.emit(*TRANSPORT[name])


def register(console):
    surface = MidiSurface(console)
    console.on_state(surface.poll)
    try:
        from PyQt6.QtWidgets import QLabel
        lbl = QLabel("MIDI: nanoKONTROL2" if surface.ctl is not None else "MIDI: none")
        console.add_status(lbl)
    except Exception:  # noqa: BLE001
        pass
    console.midi = surface
