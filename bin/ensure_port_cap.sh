#!/usr/bin/env bash
# Ensure the venv's Python can bind the configured web port.
#
# Binding a port below 1024 (the default web port is 80) needs either root or
# the cap_net_bind_service file capability on the python binary. We set that
# capability at install time, BUT any later apt upgrade/reinstall of the python
# package replaces the binary and silently strips the capability — after which
# the app fails to bind port 80 ("unable to connect" in the browser).
#
# This script re-applies the capability if it's missing, so the binding
# self-heals on every launch. It is:
#   - a no-op when the capability is already present (the common case),
#   - a no-op when the web port is >= 1024 (no capability needed),
#   - non-interactive: uses `sudo -n` so it never blocks an unattended boot.
#     A matching /etc/sudoers.d rule (installed by bin/linux-install.sh) lets
#     just this one setcap command run without a password.
# Always exits 0 so it can be wired as a non-fatal pre-launch step.

set -u
cd "$(dirname "$0")/.."

PY="$(readlink -f venv/bin/python3 2>/dev/null || true)"
[ -n "$PY" ] && [ -x "$PY" ] || { echo "[cap] venv python not found; skipping"; exit 0; }

# Read the web port from config.yaml (fall back to 80, the app's default).
PORT="$("$PY" - <<'PYEOF' 2>/dev/null || true
try:
    import yaml
    print((yaml.safe_load(open("config.yaml")) or {}).get("web", {}).get("port", 80))
except Exception:
    print(80)
PYEOF
)"
[ -n "${PORT:-}" ] || PORT=80

if [ "$PORT" -ge 1024 ] 2>/dev/null; then
    echo "[cap] web port $PORT >= 1024; no capability needed"
    exit 0
fi

if getcap "$PY" 2>/dev/null | grep -q cap_net_bind_service; then
    exit 0  # already capable — silent no-op
fi

echo "[cap] cap_net_bind_service missing on $PY (needed for port $PORT) — re-applying"
SETCAP="$(command -v setcap || echo /usr/sbin/setcap)"
if sudo -n "$SETCAP" 'cap_net_bind_service=+ep' "$PY" 2>/dev/null; then
    echo "[cap] re-applied cap_net_bind_service"
else
    echo "[cap] could NOT re-apply (no passwordless sudo rule?). Run once:"
    echo "      sudo $SETCAP 'cap_net_bind_service=+ep' $PY"
    echo "      or re-run bin/linux-install.sh. (Or set web.port >= 1024 in config.yaml.)"
fi
exit 0
