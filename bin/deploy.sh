#!/usr/bin/env bash
# GL_Simple Deploy: interactive multi-project deployment for Linux/macOS.
#
# On a fresh machine:
#   1. Ensures git is installed (apt on Linux).
#   2. Reads deploy/catalog.yaml for the list of available projects.
#   3. Probes each catalog entry to see which the current GitHub
#      credentials can access (via `git ls-remote`).
#   4. Prompts the operator to pick which project(s) to install and
#      which one is the primary (auto-runs on app launch).
#   5. Clones chosen projects into projects/<id>/.
#   6. Writes config.yaml's `project:` field to the primary.
#   7. Hands off to bin/setup_and_run.sh.
#
# This script is for the *first* deployment on a machine. After that,
# bin/setup_and_run.sh will auto-clone the active project if missing,
# so you can just edit config.yaml's `project:` and run setup_and_run.
#
# The Windows equivalent is bin/deploy.ps1 - keep their UX identical.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "====================================="
echo "  GL_Simple Deploy (Linux/macOS)"
echo "====================================="

# --- 1. Ensure git is present ---
if ! command -v git >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        echo "[1/6] git not found - installing..."
        sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git
    else
        echo "ERROR: git is required but not installed and apt-get isn't available." >&2
        echo "       Install git manually and re-run." >&2
        exit 1
    fi
else
    echo "[1/6] git OK ($(git --version))"
fi

# --- 2. Load catalog ---
CATALOG="deploy/catalog.yaml"
if [ ! -f "$CATALOG" ]; then
    echo "ERROR: catalog file not found at $CATALOG" >&2
    exit 1
fi

# Build parallel arrays: ID, URL, NAME, DESC. Parse the YAML by hand
# (no yq dependency). Assumes the format committed in deploy/catalog.yaml:
#
#   projects:
#     <id>:
#       repo: <url>
#       display_name: <name>
#       description: "..."
#
declare -a IDS URLS NAMES
while IFS= read -r line; do
    # Match "  <id>:" - exactly two-space indented, ending in ":"
    if [[ "$line" =~ ^[[:space:]]{2}([A-Za-z_][A-Za-z0-9_]*):[[:space:]]*$ ]]; then
        current_id="${BASH_REMATCH[1]}"
        IDS+=("$current_id")
        URLS+=("")
        NAMES+=("")
        idx=$((${#IDS[@]} - 1))
    elif [[ -n "${current_id:-}" ]]; then
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

echo "[2/6] Loaded ${#IDS[@]} projects from $CATALOG"

# --- 3. Probe access for each project ---
# `git ls-remote URL HEAD` returns 0 if the current creds can read the
# repo. Quiet failures (auth required, repo not found, network) all
# come back as nonzero. We don't distinguish - operator sees one of
# two states: ACCESSIBLE or NO ACCESS.
echo "[3/6] Checking access to each project repo..."
declare -a ACCESS
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

# --- 4. Show picker ---
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
echo "  NO ACCESS = current GitHub credentials can't read that repo."
echo "  Set up a Personal Access Token or SSH key (see docs/DEPLOYMENT.md)"
echo "  then re-run this script."
echo ""

# --- 5. Read selection ---
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
            echo "  $tok (${IDS[$idx]}) is not accessible with current credentials." >&2
            valid=false; break
        fi
        CHOSEN_IDX+=("$idx")
    done
    if ! $valid; then CHOSEN_IDX=(); fi
done

CHOSEN_IDS=()
for i in "${CHOSEN_IDX[@]}"; do CHOSEN_IDS+=("${IDS[$i]}"); done
echo "  Will install: ${CHOSEN_IDS[*]}"

# --- 6. Choose primary if more than one ---
PRIMARY_ID="${CHOSEN_IDS[0]}"
if [ ${#CHOSEN_IDS[@]} -gt 1 ]; then
    echo ""
    echo "Which project is the primary (runs on app launch)?"
    for i in "${!CHOSEN_IDS[@]}"; do
        num=$((i + 1))
        printf "  %d) %s\n" "$num" "${CHOSEN_IDS[$i]}"
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

# --- 7. Clone selected projects ---
echo ""
echo "[4/6] Cloning selected projects..."
for i in "${CHOSEN_IDX[@]}"; do
    id="${IDS[$i]}"
    url="${URLS[$i]}"
    dest="projects/$id"
    if [ -d "$dest/.git" ]; then
        echo "  - $id already deployed at $dest (skipping clone)"
        continue
    fi
    echo "  - cloning $id from $url"
    rm -rf "$dest"   # remove any stale dir/leftover from prior runs
    git clone "$url" "$dest"
done

# --- 8. Write config.yaml's project: field ---
echo ""
echo "[5/6] Setting active project to '$PRIMARY_ID' in config.yaml..."
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
echo "  Done."

# --- 9. Hand off to setup_and_run ---
echo ""
echo "[6/6] Deployment complete."
echo ""
echo "Run the app with:    bin/setup_and_run.sh"
echo "Switch primary with: edit config.yaml's 'project:' field, then setup_and_run."
