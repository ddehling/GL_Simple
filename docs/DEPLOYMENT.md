# Deployment

Each art-piece project lives in its own **private GitHub repo** and is
cloned into `projects/<id>/` on the deployment machine. The main
`GL_Simple` repo holds only the engine (rendering, scheduling, web,
geometry) and is project-agnostic. A given deployment chooses which
projects to install, and collaborators see only the projects they have
access to.

| Repo | Path | Holds |
|---|---|---|
| `GL_Simple` | (main repo) | Engine code, shared utilities, deploy catalog. Public-friendly. |
| `GL_Simple_<id>` | `projects/<id>/` (standalone clone, gitignored) | One project's entire source: `project.yaml`, `event_map.py`, `weather_params.py`, `shaders/`, `media/`. Private. |

The catalog of available projects (id → repo URL) lives at
[deploy/catalog.yaml](../deploy/catalog.yaml). The deployment scripts
([bin/setup_and_run.sh](../bin/setup_and_run.sh) and
[bin/setup_and_run.ps1](../bin/setup_and_run.ps1)) read this catalog
plus `config.yaml`'s `project:` field, and clone the active project
into `projects/<id>/` if it isn't already present.

## Fresh-machine setup

The recommended path uses `bin/deploy.*` to interactively pick which
projects to install and which is the primary:

1. Install **git** on the machine if it isn't already (Linux: `sudo
   apt-get install git`; Windows: download from
   <https://git-scm.com/download/win>, or use `winget install Git.Git`).
   The deploy and setup scripts attempt to install git automatically if
   missing, but git is also needed to clone this engine repo in the
   first place.
2. Clone the engine repo (public):
   ```bash
   git clone https://github.com/ddehling/GL_Simple.git
   cd GL_Simple
   ```
3. Set GitHub auth so private project repos can be cloned (one-time per
   machine - see [Authenticating to private project repos](#authenticating-to-private-project-repos)).
4. Run the deploy tool:
   ```bash
   ./bin/deploy.sh             # Linux/macOS
   bin\deploy.bat              # Windows (or bin\deploy.ps1 directly)
   ```
   The tool lists every project in `deploy/catalog.yaml`, marks each
   as `[ACCESSIBLE]` or `[NO ACCESS]` based on whether the current
   GitHub credentials can read its private repo, lets the operator
   pick which to install, asks which is the primary, clones the
   selected repos into `projects/<id>/`, and writes the primary id
   into `config.yaml`.
5. Run the app:
   ```bash
   ./bin/setup_and_run.sh      # Linux/macOS
   bin\setup_and_run.bat       # Windows
   ```
   On first run, `setup_and_run.*`:
   - verifies git is installed (installs it on Linux/Windows if missing),
   - auto-clones the active project from the catalog if not already deployed,
   - verifies Python is installed (installs it on Linux/Windows if missing),
   - installs Python deps,
   - on Linux: installs PortAudio / libsndfile / ALSA dev headers,
   - launches the app.

### Shortcut: skip the deploy tool

If you only want one project and you already know its id, you can
write `project: <id>` directly into `config.yaml` and run
`bin/setup_and_run.*` — it auto-clones the named project on first run.
The deploy tool is mostly useful when (a) installing several projects
at once or (b) the operator wants to see what they have access to.

## Authenticating to private project repos

The project repos are private, so `git clone` needs credentials. Both
methods below work identically on Linux and Windows.

### Option A — Personal Access Token (PAT) (recommended for collaborators)

GitHub PATs are strings that act like a password for command-line git.
They can be scoped to specific repos, which means a collaborator's
token can only access the projects they're entitled to.

1. On GitHub: **Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → Generate new token**.
2. **Repository access:** "Only select repositories" → choose the
   specific `GL_Simple_<id>` repos the operator should see.
3. **Permissions → Repository → Contents:** Read-only.
4. Copy the token (looks like `github_pat_11ABC...`).
5. Configure git to store credentials so it's used automatically:
   ```bash
   git config --global credential.helper store          # Linux/macOS
   git config --global credential.helper manager        # Windows
   ```
6. First `git clone` will prompt for username/password; paste the
   token as the password. Subsequent clones use the cached credential.

To revoke a collaborator's access: delete or regenerate the PAT on
GitHub. No machine-side action needed.

### Option B — SSH keys

If `git@github.com:...` URLs in `deploy/catalog.yaml` are preferred,
collaborators can add an SSH key on their account
(**Settings → SSH and GPG keys**). One-time setup, then no further
prompts.

## Adding a new project

1. Scaffold the project on a development machine (use the layout editor
   or manually create `projects/<id>/{project.yaml, event_map.py,
   weather_params.py, shaders/, media/}`).

2. Create the private repo on GitHub:
   ```bash
   gh repo create GL_Simple_<id> --private \
       --description "Source for the <Name> art piece. Cloned into projects/<id>/ on deployment machines."
   ```

3. Push the project source into the new repo:
   ```bash
   cd projects/<id>
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/ddehling/GL_Simple_<id>.git
   git push -u origin main
   ```

4. Add the project to [deploy/catalog.yaml](../deploy/catalog.yaml) so
   other deployments can find it.

5. Grant collaborators access to the new repo (Settings → Manage access)
   and they can deploy it on their machines.

## Switching the active project on a machine

Edit `config.yaml`:
```yaml
project: <new-id>
```

Re-run `bin/setup_and_run.*`. If the new project isn't cloned yet, the
script auto-clones it from the catalog.

## Updating a project that's already deployed

```bash
cd projects/<id>
git pull
```

Or update everything deployed on the machine:
```bash
# Linux/macOS
for d in projects/*/; do [ -d "$d/.git" ] && git -C "$d" pull; done

# Windows PowerShell
Get-ChildItem projects -Directory | Where-Object { Test-Path "$($_.FullName)\.git" } | ForEach-Object { git -C $_.FullName pull }
```

## Modifying a project locally and pushing back

`projects/<id>/` is a full git repo:
```bash
cd projects/<id>
# edit shaders, weather params, media...
git add .
git commit -m "Tune cyberpunk pulse rate"
git push
```
No two-step push (unlike the old submodule model).

## Legacy: media-only submodule (transitional)

The `weight_of_light` project still uses the older split model: its
code lives in the main `GL_Simple` repo under
`projects/weight_of_light/`, while only its media is a submodule at
`projects/weight_of_light/media/` pointing at `GL_Simple_wol_media`.
The setup scripts handle both models simultaneously during the
transition. WoL will be migrated to the standalone-clone model in a
subsequent pass.

## Troubleshooting

**`bin/setup_and_run.*` says "Project 'X' has no entry in deploy/catalog.yaml".**
Either add `X` to the catalog (with its private repo URL) or change
`config.yaml`'s `project:` to a deployed project id.

**`bin/setup_and_run.*` says "clone failed. Set up GitHub auth".**
The script tried to clone a private repo and git rejected the request.
Set up a PAT or SSH key per the
[Authenticating](#authenticating-to-private-project-repos) section
above, then re-run.

**The app starts but complains about missing shaders or sounds.**
The project was cloned but is missing files. Confirm `projects/<id>/`
contains `project.yaml`, `shaders/`, and `media/sounds/`. If not,
delete the dir and re-run setup to re-clone.

**Want to roll back the fan migration entirely.**
The pre-migration state of the old `GL_Simple_fan_media` repo is
preserved under the `pre-restructure` tag in the now-renamed
`GL_Simple_fan` repo:
```bash
git -C projects/fan fetch --tags
git -C projects/fan checkout pre-restructure
```
This brings back the old `sounds/`-at-the-top layout without project
code. The main repo's tracked `projects/fan/` files were removed in
this migration; they're still in main's git history if needed.
