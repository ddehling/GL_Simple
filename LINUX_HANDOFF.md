# Linux Handoff — Bluetooth audio + beloved set

Notes for picking this work up on the Linux box (Pop!_OS / Raspberry Pi).
Written 2026-06-21 from the Windows dev machine.

---

## 0. CRITICAL FIRST STEP — commit + push, on Windows, before you switch

This session's new work is **uncommitted**. It only reaches the Linux box if
you commit AND push it from Windows first. Two separate repos are involved
(the engine, and the `projects/fan` standalone repo).

```bash
# --- engine repo (Bluetooth audio feature) ---
cd /c/Users/ddehl/Desktop/Devel/GL_Simple
git add Stories_OGL.py lib/audio_analyzer.py lib/bluetooth_audio.py \
        web/web_controller.py web/templates/control_panel.html \
        web/static/js/control_panel.js bin/linux-install.sh \
        docs/BLUETOOTH_AUDIO.md LINUX_HANDOFF.md
git commit -m "bluetooth: A2DP audio-sink input source (lucifera), Linux-only"
git push

# --- fan project repo (cyberpunk PARAMETER_DEFINITIONS fix) ---
cd projects/fan
git add weather_params.py
git commit -m "weather_params: define cyber_* + bird_density (silence startup validator)"
git push
```

Verify both show `## main...origin/main` with nothing ahead afterward.
Then on Linux: `./bin/linux-run.sh` (pulls engine + every deployed project).

---

## 1. What is ALREADY committed + pushed (arrives via normal pull — no action)

The whole **beloved ("Weather of the Heart") shader rebuild** is already in the
repos' history and pushed. You do **not** need to re-do or re-commit any of it;
it'll be on Linux after the run script pulls. It includes:

- 9 `HEART_*` states + 8 love-themed STATE params (`heart_tint`, `heart_warmth`,
  `ember_density`, `ember_lift`, `veil_density`, `stillness`, `turmoil`,
  `pulse_rate`) in `projects/fan/weather_params.py`.
- 5 shaders in `projects/fan/shaders/`: `heart_sky` (0.97 backdrop),
  `memory_veil` (0.82), `ember_drift` (0.60), `heart_tide` (0.20),
  `two_lights` (0.30) — plus `heart_pulse` rebuilt as an exact SDF heart with
  rim/fill/glint, fan-radius aspect correction, and sizing fixes.
- Engine-side: `lib/weather_state.py` publishes the heart params;
  `docs/shader_info.txt` has the beloved z-band table; `renderer/effects/`
  `warm_bloom.py` + `distant_lights.py` got z_centroid + depth-mask fixes.

**On Linux, verify it visually** (it was only compile-checked on Windows, never
displayed): temporarily set the fan project to start in the beloved set and run.
```bash
# projects/fan/project.yaml
startup_weather_set: beloved
startup_weather_state: heart_dawn
```
Then `venv/bin/python Stories_OGL.py` (or the run script) and watch the fan /
the web preview at http://localhost (port 80 per config.yaml) → /preview.
Revert project.yaml when done.

---

## 2. What is UNCOMMITTED right now (the list `git add` above covers)

**Engine repo (Bluetooth feature):**
- NEW `lib/bluetooth_audio.py` — BlueZ A2DP-sink controller + inert stub.
- `lib/audio_analyzer.py` — new `"bluetooth"` capture source + `set_bluetooth_hint()`.
- `web/web_controller.py` — `toggle_bluetooth_audio` / `approve_pairing` /
  `deny_pairing` Socket.IO handlers + bluetooth block in the state snapshot.
- `web/templates/control_panel.html` + `web/static/js/control_panel.js` — toggle,
  per-device approval cards, connected list (hidden when unavailable).
- `Stories_OGL.py` — owns the receiver, drains web actions, auto-routes the
  analyzer on connect/disconnect, shuts it down cleanly.
- `bin/linux-install.sh` — installs BT deps into the venv + bluetooth group.
- NEW `docs/BLUETOOTH_AUDIO.md` — setup + on-device test plan.

**Fan repo:** `weather_params.py` — adds `PARAMETER_DEFINITIONS` entries for the
7 `cyber_*` params + `bird_density` (silences the startup "parameters missing"
banner; makes them editable in the web weather editor). Pure additive.

---

## 3. Bluetooth audio ("lucifera") — Linux is where it actually runs

This **could not be tested on Windows** (Windows can't be an A2DP sink at all;
the receiver is an inert stub there). Linux is the real target.

**Setup:** `./bin/linux-install.sh` now does it automatically (non-fatal if any
part fails). It installs `bluez` + `libspa-0.2-bluetooth` (PipeWire BT sink) +
the D-Bus/GLib build deps, pip-installs `dbus-python` + `PyGObject` **into the
venv**, and adds you to the `bluetooth` group (re-login for that to take
effect). Manual steps + the PulseAudio alternative are in
`docs/BLUETOOTH_AUDIO.md`.

**Quick import check in the venv** (must pass or the UI shows "unavailable"):
```bash
source venv/bin/activate
python -c "import dbus, dbus.mainloop.glib; from gi.repository import GLib; print('BT deps OK')"
```

**Then run the 6-step on-device test plan in `docs/BLUETOOTH_AUDIO.md`:**
availability → enable (phone sees "lucifera") → approve a pairing in the web UI
→ audio routes + visuals react → disconnect falls back → disable hides it.

**Gotchas to expect:**
- Bindings must be importable *inside the venv* (the install handles it; if you
  recreate the venv, redo `pip install dbus-python PyGObject`).
- Admin gating: `config.yaml` sets `admin_password: "admin123"`, so the toggle +
  approve/deny require logging in at `/admin` first (same browser session).
- If "Bluetooth unavailable" persists: `systemctl status bluetooth`,
  `rfkill unblock bluetooth`, and confirm `pactl list short sources | grep bluez`
  shows a node once a phone connects.

---

## 4. What does NOT transfer (machine-local to the Windows box)

- **Claude Code permission settings** — this session also added `PowerShell` +
  `Bash` tool-wide allow rules to `~/.claude/settings.json` on *Windows*. That's
  a per-machine user file, not in the repo. On Linux, only `Bash` is relevant;
  re-add it there if you want prompt-free Bash (`PowerShell` is moot on Linux).
- `active_project.yaml` (if present) is a gitignored per-machine override.

---

## 5. Open / unverified

- Beloved set: compile-clean + shape-verified by ASCII render, but never shown
  on a real fan. Tune preset numbers (ember densities, pulse rates, heart_tint
  colors) against the actual display.
- Bluetooth: entirely unverified on hardware — the BlueZ agent's deferred-reply
  approval flow and PipeWire node routing need a real phone + adapter.
- This `LINUX_HANDOFF.md` is a transient note; delete it once you're settled.
```
