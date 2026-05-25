# Deployment

Each art-piece project lives in its own **private GitHub repo** named
`GL_Simple_<id>` and is cloned into `projects/<id>/` on the deployment
machine. The main `GL_Simple` repo holds only the engine (rendering,
scheduling, web, geometry) and is project-agnostic. A given deployment
chooses which projects to install; collaborators see only the projects
they have access to.

| Repo | Path | Holds |
|---|---|---|
| `GL_Simple` | (main repo) | Engine code, shared utilities, [deploy/catalog.yaml](../deploy/catalog.yaml). Public-friendly. |
| `GL_Simple_<id>` | `projects/<id>/` (standalone clone, gitignored) | One project's entire source: `project.yaml`, `event_map.py`, `weather_params.py`, `shaders/`, `media/`. Private. |

The catalog of available projects (id → repo URL) lives at
[deploy/catalog.yaml](../deploy/catalog.yaml). The setup script reads
this catalog at install time.

## Fresh-machine setup

One command per platform. The script handles git install, GitHub
authentication, project selection, cloning, dep install, and launch.

**Linux/macOS:**
```bash
git clone https://github.com/ddehling/GL_Simple.git
cd GL_Simple
./bin/setup.sh
```

**Windows:**
```
git clone https://github.com/ddehling/GL_Simple.git
cd GL_Simple
bin\setup.bat
```
(Windows needs git pre-installed for the initial `git clone`; the
script installs it via winget if you launch via `bin\setup.bat`
from a different location, but the curl-pipe path is easier.)

What `bin/setup.*` does, in order:

1. Installs `git` if missing (apt on Linux, winget on Windows).
2. Installs the **GitHub CLI** (`gh`) if missing.
3. If not already signed in, runs `gh auth login` — gh prints a URL
   and a one-time code, you open the URL on any device (phone, laptop,
   browser tab), enter the code, sign in, grant access. The script
   resumes automatically. gh also wires itself as git's credential
   helper, so private project repos clone with no further prompts.
4. Reads `deploy/catalog.yaml` and probes each entry with `git
   ls-remote` to mark it `[ACCESSIBLE]` or `[NO ACCESS]` under the
   signed-in account.
5. Prompts which project(s) to install and which is primary.
6. Clones the selected project repos into `projects/<id>/`.
7. Writes the primary id to `config.yaml`'s `project:` field.
8. Installs Python + system deps (PortAudio, libsndfile, ALSA dev
   headers on Linux; Python via winget on Windows).
9. Creates `./venv`, installs `requirements.txt`.
10. Launches `Stories_OGL.py`.

Re-running `bin/setup.*` is fine — every step is idempotent (skips
already-installed bits, skips already-cloned projects).

## Launching after setup

After the first run, launch the app directly without re-running setup:

```bash
venv/bin/python Stories_OGL.py        # Linux/macOS
venv\Scripts\python Stories_OGL.py    # Windows
```

Re-run `bin/setup.*` if you change the active project, add a project,
or just want to refresh deps.

## Adding a new project

1. Scaffold the project tree locally: `projects/<id>/{project.yaml,
   event_map.py, weather_params.py, shaders/, media/}`.

2. Create the private repo on GitHub:
   ```bash
   gh repo create GL_Simple_<id> --private \
       --description "Source for the <Name> art piece."
   ```

3. Push the project source:
   ```bash
   cd projects/<id>
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/ddehling/GL_Simple_<id>.git
   git push -u origin main
   ```

4. Add the project to [deploy/catalog.yaml](../deploy/catalog.yaml):
   ```yaml
   projects:
     <id>:
       repo: https://github.com/ddehling/GL_Simple_<id>.git
       display_name: <Name>
       description: "..."
   ```

5. Grant collaborators access to the new repo (Settings → Manage
   access) and they can deploy it on their machines.

## Switching the active project on a machine

Edit `config.yaml`:
```yaml
project: <new-id>
```

Re-run `bin/setup.*` (it'll auto-clone the new project if needed and
launch). Or if the new project is already deployed, just launch
directly with `venv/bin/python Stories_OGL.py` — `Stories_OGL.py`
re-reads `config.yaml` on startup.

## Updating a project that's already deployed

```bash
cd projects/<id> && git pull
```

Or update everything deployed on the machine:
```bash
# Linux/macOS
for d in projects/*/; do [ -d "$d/.git" ] && git -C "$d" pull; done

# Windows PowerShell
Get-ChildItem projects -Directory | Where-Object { Test-Path "$($_.FullName)\.git" } | ForEach-Object { git -C $_.FullName pull }
```

## Modifying a project locally and pushing back

`projects/<id>/` is a full git repo. Edit, commit, push as usual:
```bash
cd projects/<id>
git add .
git commit -m "Tune cyberpunk pulse rate"
git push
```

## Troubleshooting

**`bin/setup.*` says "Project 'X' has no entry in deploy/catalog.yaml".**
Add `X` to the catalog (with its private repo URL) or change
`config.yaml`'s `project:` to a deployed project id.

**`bin/setup.*` says "gh auth login failed".**
Your machine couldn't open the GitHub device-auth URL. Try copying the
URL from the script output and opening it manually on a phone or
another machine — the device-flow works across devices.

**A project shows `[NO ACCESS]` in the picker.**
Your signed-in GitHub account doesn't have access to that repo. Ask
the project owner to add you (the repo's GitHub Settings → Manage
access). Re-run `bin/setup.*` after.

**The app starts but complains about missing shaders or sounds.**
The project was cloned but is missing files. Confirm `projects/<id>/`
contains `project.yaml`, `shaders/`, and `media/sounds/`. If not,
delete the dir and re-run setup to re-clone.

**Want to sign in as a different GitHub account.**
```bash
gh auth logout
./bin/setup.sh
```

**Want to roll back the fan or WoL migration.**
Pre-migration state of each project repo is preserved under the
`pre-restructure` tag:
```bash
git -C projects/<id> fetch --tags
git -C projects/<id> checkout pre-restructure
```
