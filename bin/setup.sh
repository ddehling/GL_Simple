#!/usr/bin/env bash
# GL_Simple all-in-one setup for Linux (Ubuntu/Debian).
#
# Run from the repo root after `git clone`. Does everything needed to
# get from a freshly cloned engine to a running app:
#
#   1. Installs git + gh (GitHub CLI) via apt if missing.
#   2. Signs you into GitHub via gh's browser-based device flow if
#      not already signed in. gh wires itself as git's credential
#      helper, so private project repos clone without any PAT pasting.
#   3. Reads deploy/catalog.yaml, probes each project repo to mark it
#      [ACCESSIBLE] or [NO ACCESS] under your GitHub auth.
#   4. Prompts you to pick which project(s) to install and which is
#      the primary (the one that runs on launch).
#   5. Clones the selected project repos into projects/<id>/.
#   6. Writes the primary id to config.yaml's `project:` field.
#   7. Installs system deps (python, PortAudio, libsndfile, ALSA dev).
#   8. Creates ./venv and installs requirements.txt.
#   9. Grants the venv Python permission to bind port 80.
#  10. Launches Stories_OGL.py.
#
# Subsequent app launches can use ./bin/quick_run.sh to skip the
# deps-verification dance. Re-running this script is also fine - it's
# idempotent (skips already-installed bits and already-cloned repos).
#
# Cross-platform note: the Windows equivalent is bin/setup.ps1 - both
# do the same thing and use the same catalog.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "====================================="
echo "  GL_Simple Setup (Linux)"
echo "====================================="

# ---------------------------------------------------------------------
# 1. git
# ---------------------------------------------------------------------
if ! command -v git >/dev/null 2>&1; then
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "ERROR: git is required and apt-get isn't available. Install git manually." >&2
        exit 1
    fi
    echo "[1/8] Installing git..."
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git
else
    echo "[1/8] git OK ($(git --version))"
fi

# ---------------------------------------------------------------------
# 2. gh + auth
# ---------------------------------------------------------------------
# Always install gh from GitHub's official apt source. Ubuntu's
# universe carries very old gh (Jammy: 2.4 from 2021) that's missing
# flags we use. The GitHub source guarantees a current version.
install_gh_from_github() {
    echo "      Adding GitHub's apt source for current gh..."
    sudo apt-get install -y --no-install-recommends curl ca-certificates
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        | sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null
    sudo apt-get update -y
    sudo apt-get install -y --only-upgrade gh \
        || sudo apt-get install -y gh
}

GH_OK=false
if command -v gh >/dev/null 2>&1; then
    # Require gh >= 2.20 (covers --git-protocol, modern device-flow UX).
    GH_VER=$(gh --version 2>/dev/null | awk '/^gh version/ {print $3; exit}')
    GH_MAJOR=$(echo "$GH_VER" | cut -d. -f1)
    GH_MINOR=$(echo "$GH_VER" | cut -d. -f2)
    if [ "${GH_MAJOR:-0}" -gt 2 ] || { [ "${GH_MAJOR:-0}" -eq 2 ] && [ "${GH_MINOR:-0}" -ge 20 ]; }; then
        GH_OK=true
        echo "[2/8] gh OK (version $GH_VER)"
    else
        echo "[2/8] gh present but too old (version $GH_VER); upgrading..."
    fi
fi
if ! $GH_OK; then
    install_gh_from_github
fi

if ! gh auth status >/dev/null 2>&1; then
    echo ""
    echo "      Not signed in to GitHub yet."
    echo "      gh will print a URL and a one-time code. Open the URL"
    echo "      on any device, enter the code, sign in, and grant access."
    echo "      The script will resume automatically."
    echo ""
    gh auth login --hostname github.com --web \
        || { echo "ERROR: gh auth login failed." >&2; exit 1; }
fi
GH_USER=$(gh api user -q .login 2>/dev/null || echo "?")
echo "      Signed in to GitHub as: $GH_USER"
gh auth setup-git >/dev/null 2>&1 || true

# ---------------------------------------------------------------------
# 3. Load catalog
# ---------------------------------------------------------------------
CATALOG="deploy/catalog.yaml"
if [ ! -f "$CATALOG" ]; then
    echo "ERROR: catalog file not found at $CATALOG" >&2
    exit 1
fi

IDS=()
URLS=()
NAMES=()
current_id=""
while IFS= read -r line; do
    if [[ "$line" =~ ^[[:space:]]{2}([A-Za-z_][A-Za-z0-9_]*):[[:space:]]*$ ]]; then
        current_id="${BASH_REMATCH[1]}"
        IDS+=("$current_id")
        URLS+=("")
        NAMES+=("")
        idx=$((${#IDS[@]} - 1))
    elif [[ -n "$current_id" ]]; then
        if [[ "$line" =~ ^[[:space:]]{4}repo:[[:space:]]*(.+)$ ]]; then
            URLS[$idx]=$(echo "${BASH_REMATCH[1]}" | sed 's/[[:space:]]*$//')
        elif [[ "$line" =~ ^[[:space:]]{4}display_name:[[:space:]]*(.+)$ ]]; then
            NAMES[$idx]=$(echo "${BASH_REMATCH[1]}" | sed 's/[[:space:]]*$//; s/^"//; s/"$//')
        fi
    fi
done < "$CATALOG"

if [ ${#IDS[@]} -eq 0 ]; then
    echo "ERROR: no projects found in $CATALOG" >&2
    exit 1
fi
echo "[3/8] Loaded ${#IDS[@]} projects from $CATALOG"

# ---------------------------------------------------------------------
# 4. Probe access for each project
# ---------------------------------------------------------------------
echo "[4/8] Checking access to each project repo..."
ACCESS=()
for i in "${!IDS[@]}"; do
    url="${URLS[$i]}"
    if [ -z "$url" ]; then
        ACCESS+=("NO_URL")
        continue
    fi
    if GIT_TERMINAL_PROMPT=0 git ls-remote "$url" HEAD >/dev/null 2>&1; then
        ACCESS+=("OK")
    else
        ACCESS+=("DENIED")
    fi
done

echo ""
echo "Available projects:"
echo ""
for i in "${!IDS[@]}"; do
    num=$((i + 1))
    id="${IDS[$i]}"
    name="${NAMES[$i]:-$id}"
    case "${ACCESS[$i]}" in
        OK)     status="[ACCESSIBLE]" ;;
        DENIED) status="[NO ACCESS] " ;;
        *)      status="[NO URL]    " ;;
    esac
    printf "  %d) %-22s %s  %s\n" "$num" "$id" "$status" "$name"
done
echo ""
echo "  NO ACCESS = your GitHub account doesn't have access to that repo."
echo "  Ask the project owner to add you (Settings -> Manage access)"
echo "  then re-run this script."
echo ""

# ---------------------------------------------------------------------
# 5. Pick projects + primary
# ---------------------------------------------------------------------
CHOSEN_IDX=()
while [ ${#CHOSEN_IDX[@]} -eq 0 ]; do
    read -rp "Enter project numbers to install (space-separated, e.g. '1 2'): " input
    CHOSEN_IDX=()
    valid=true
    for tok in $input; do
        if ! [[ "$tok" =~ ^[0-9]+$ ]]; then
            echo "  '$tok' is not a number." >&2; valid=false; break
        fi
        idx=$((tok - 1))
        if [ "$idx" -lt 0 ] || [ "$idx" -ge ${#IDS[@]} ]; then
            echo "  $tok is out of range." >&2; valid=false; break
        fi
        if [ "${ACCESS[$idx]}" != "OK" ]; then
            echo "  $tok (${IDS[$idx]}) is not accessible." >&2; valid=false; break
        fi
        CHOSEN_IDX+=("$idx")
    done
    if ! $valid; then CHOSEN_IDX=(); fi
done

CHOSEN_IDS=()
for i in "${CHOSEN_IDX[@]}"; do CHOSEN_IDS+=("${IDS[$i]}"); done
echo "  Will install: ${CHOSEN_IDS[*]}"

PRIMARY_ID="${CHOSEN_IDS[0]}"
if [ ${#CHOSEN_IDS[@]} -gt 1 ]; then
    echo ""
    echo "Which project is the primary (runs on app launch)?"
    for i in "${!CHOSEN_IDS[@]}"; do
        printf "  %d) %s\n" "$((i + 1))" "${CHOSEN_IDS[$i]}"
    done
    while true; do
        read -rp "Primary [1]: " primary_input
        primary_input="${primary_input:-1}"
        if [[ "$primary_input" =~ ^[0-9]+$ ]] && [ "$primary_input" -ge 1 ] && [ "$primary_input" -le ${#CHOSEN_IDS[@]} ]; then
            PRIMARY_ID="${CHOSEN_IDS[$((primary_input - 1))]}"
            break
        fi
        echo "  Invalid choice." >&2
    done
fi
echo "  Primary: $PRIMARY_ID"

# ---------------------------------------------------------------------
# 6. Clone selected
# ---------------------------------------------------------------------
echo ""
echo "[5/8] Cloning selected projects..."
for i in "${CHOSEN_IDX[@]}"; do
    id="${IDS[$i]}"
    url="${URLS[$i]}"
    dest="projects/$id"
    if [ -d "$dest/.git" ]; then
        echo "      $id already deployed at $dest (skipping clone)"
        continue
    fi
    echo "      cloning $id from $url"
    rm -rf "$dest"
    git clone "$url" "$dest"
done

# ---------------------------------------------------------------------
# 7. Write config.yaml's project: field
# ---------------------------------------------------------------------
echo "[6/8] Setting active project to '$PRIMARY_ID' in config.yaml..."
if [ ! -f config.yaml ]; then
    echo "ERROR: config.yaml not found at repo root." >&2
    exit 1
fi
if grep -qE '^project:' config.yaml; then
    sed -i.bak -E "s|^project:.*$|project: $PRIMARY_ID|" config.yaml
    rm -f config.yaml.bak
else
    echo "project: $PRIMARY_ID" >> config.yaml
fi

# ---------------------------------------------------------------------
# 8. System deps + Python venv + pip install
# ---------------------------------------------------------------------
echo "[7/8] Installing system + Python dependencies..."
sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip build-essential \
    libportaudio2 portaudio19-dev libsndfile1-dev libasound2-dev pkg-config libcap2-bin

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# Verify sounddevice imports; if not, force-reinstall after audio libs.
if ! python -c "import sounddevice" 2>/dev/null; then
    echo "      sounddevice import failed; reinstalling..."
    python -m pip install --force-reinstall --no-cache-dir sounddevice
    if ! python -c "import sounddevice" 2>/dev/null; then
        echo "ERROR: sounddevice still failing. Check PortAudio install." >&2
        exit 1
    fi
fi

# Let Python bind port 80 without sudo.
VENV_PYTHON="$(readlink -f venv/bin/python3)"
if ! getcap "$VENV_PYTHON" 2>/dev/null | grep -q cap_net_bind_service; then
    sudo setcap 'cap_net_bind_service=+ep' "$VENV_PYTHON"
fi

# ---------------------------------------------------------------------
# 9. Launch
# ---------------------------------------------------------------------
echo "[8/8] Launching application..."
echo "      Web control panel: http://lucifera.local"
echo "      Press Ctrl+C to stop."
echo ""
python Stories_OGL.py
