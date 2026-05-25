#!/usr/bin/env bash
# GL_Simple all-in-one installer for Linux/macOS.
#
# Assumes the GL_Simple engine repo is already cloned. Runs the
# interactive project picker (deploy.sh) then installs system deps,
# creates a Python venv, installs Python deps, and launches the app
# (setup_and_run.sh).
#
# After the first run, just use ./bin/setup_and_run.sh to launch -
# this script is the one-and-done for fresh installs.

set -euo pipefail
cd "$(dirname "$0")/.."

./bin/deploy.sh
./bin/setup_and_run.sh
