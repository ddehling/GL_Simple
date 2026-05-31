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
#
# It does NOT launch the app at the end. Run it with ./bin/linux-run.sh, or
# set it to start on boot with ./bin/linux-autostart.sh. Identical on Pop!_OS
# and modern Raspberry Pi OS - no Pi-specific GPU tuning is needed (Full KMS
# is the default driver and the V3D stack allocates GPU memory from CMA).
#
# Re-running this script is fine - it's idempotent (skips already-installed
# bits and already-cloned repos).
#
# Cross-platform note: the Windows equivalent is bin/windows-install.ps1 - both
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
    echo "[1/7] Installing git..."
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git
else
    echo "[1/7] git OK ($(git --version))"
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

# Run GitHub's OAuth device flow ourselves and PRINT the one-time code with
# our own echo. gh's interactive prompt does not reliably show the code in
# every terminal (xterm / SSH / when stdout is not a TTY) - that was the whole
# problem. Returns: 0 = signed in, 1 = code expired/denied (caller retries),
# 2 = could not start the flow (caller offers the token fallback).
github_device_signin() {
    local client_id="178c6fc778ccc68e1d6a"   # GitHub CLI's public OAuth client id
    command -v curl >/dev/null 2>&1 || sudo apt-get install -y --no-install-recommends curl
    local resp uc dc uri intvl
    resp="$(curl -fsS -X POST https://github.com/login/device/code \
        -H 'Accept: application/json' \
        --data-urlencode "client_id=${client_id}" \
        --data-urlencode 'scope=repo read:org')" || return 2
    uc="$(printf '%s' "$resp"  | grep -o '"user_code":"[^"]*"'        | cut -d'"' -f4)"
    dc="$(printf '%s' "$resp"  | grep -o '"device_code":"[^"]*"'      | cut -d'"' -f4)"
    uri="$(printf '%s' "$resp" | grep -o '"verification_uri":"[^"]*"' | cut -d'"' -f4)"
    intvl="$(printf '%s' "$resp" | grep -o '"interval":[0-9]*'        | cut -d: -f2)"
    [ -n "$uc" ] && [ -n "$dc" ] || return 2
    [ -n "$uri" ] || uri="https://github.com/login/device"
    [ -n "$intvl" ] || intvl=5

    echo ""
    echo "      ================================================================"
    echo "        1. On any device, open:   $uri"
    echo "        2. Enter this code:        $uc"
    echo "      ================================================================"
    echo "      Waiting for you to authorize (Ctrl+C to cancel)..."

    local tresp access err
    while true; do
        sleep "$intvl"
        tresp="$(curl -fsS -X POST https://github.com/login/oauth/access_token \
            -H 'Accept: application/json' \
            --data-urlencode "client_id=${client_id}" \
            --data-urlencode "device_code=${dc}" \
            --data-urlencode 'grant_type=urn:ietf:params:oauth:grant-type:device_code')" || continue
        access="$(printf '%s' "$tresp" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)"
        if [ -n "$access" ]; then
            printf '%s' "$access" | gh auth login --hostname github.com --git-protocol https --with-token && return 0
            return 2
        fi
        err="$(printf '%s' "$tresp" | grep -o '"error":"[^"]*"' | cut -d'"' -f4)"
        case "$err" in
            authorization_pending|"") : ;;                  # keep polling
            slow_down) intvl=$((intvl + 5)) ;;
            *) echo "      GitHub: ${err:-unknown error}; restarting sign-in..." >&2; return 1 ;;
        esac
    done
}

GH_OK=false
if command -v gh >/dev/null 2>&1; then
    # Require gh >= 2.20 (covers --git-protocol, modern device-flow UX).
    GH_VER=$(gh --version 2>/dev/null | awk '/^gh version/ {print $3; exit}')
    GH_MAJOR=$(echo "$GH_VER" | cut -d. -f1)
    GH_MINOR=$(echo "$GH_VER" | cut -d. -f2)
    if [ "${GH_MAJOR:-0}" -gt 2 ] || { [ "${GH_MAJOR:-0}" -eq 2 ] && [ "${GH_MINOR:-0}" -ge 20 ]; }; then
        GH_OK=true
        echo "[2/7] gh OK (version $GH_VER)"
    else
        echo "[2/7] gh present but too old (version $GH_VER); upgrading..."
    fi
fi
if ! $GH_OK; then
    install_gh_from_github
fi

if ! gh auth status >/dev/null 2>&1; then
    if [ ! -t 0 ]; then
        echo "ERROR: GitHub sign-in is required but this is not an interactive" >&2
        echo "       terminal. Re-run ./bin/linux-install.sh directly in a shell." >&2
        exit 1
    fi
    echo ""
    echo "      GitHub sign-in required (to clone the private project repos)."
    echo "      This script prints the one-time code itself (below) - you do not"
    echo "      need a browser on this machine."
    # Loop until authenticated. The device flow prints the code via our own
    # echo, so it is always visible; if it can't even start, fall back to a
    # pasted token.
    while ! gh auth status >/dev/null 2>&1; do
        rc=0
        github_device_signin || rc=$?
        gh auth status >/dev/null 2>&1 && break
        if [ "$rc" -eq 2 ]; then
            echo ""
            echo "      Could not run the code sign-in. Paste a GitHub token instead"
            echo "      (create one at https://github.com/settings/tokens - classic,"
            echo "      'repo' scope), or press Ctrl+C to abort."
            printf "      Paste token, then Enter (input hidden): "
            read -rs _token || true
            echo ""
            if [ -n "${_token:-}" ]; then
                printf '%s' "$_token" \
                    | gh auth login --hostname github.com --git-protocol https --with-token || true
            fi
            unset _token
        fi
        # rc==1 (code expired/denied) just loops and requests a fresh code.
    done
    echo ""
    echo "      GitHub sign-in confirmed."
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
echo "[3/7] Loaded ${#IDS[@]} projects from $CATALOG"

# ---------------------------------------------------------------------
# 4. Probe access for each project
# ---------------------------------------------------------------------
echo "[4/7] Checking access to each project repo..."
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
echo "[5/7] Cloning selected projects..."
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
# 7. Write active_project.yaml (per-machine, gitignored)
# ---------------------------------------------------------------------
# Kept separate from config.yaml so 'switch active project on this
# machine' doesn't dirty a tracked file - no merge churn between
# operators with different active projects.
echo "[6/7] Setting active project to '$PRIMARY_ID' in active_project.yaml..."
cat > active_project.yaml <<EOF
# Per-machine active project selection. Gitignored.
# Override at launch with --project <id>.
project: $PRIMARY_ID
EOF

# ---------------------------------------------------------------------
# 8. System deps + Python venv + pip install
# ---------------------------------------------------------------------
echo "[7/7] Installing system + Python dependencies..."
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
# Done
# ---------------------------------------------------------------------
echo ""
echo "====================================="
echo "  Setup complete."
echo "====================================="
echo "  Run the app now:    ./bin/linux-run.sh"
echo "  Start it on boot:   ./bin/linux-autostart.sh   (choose 'enable')"
