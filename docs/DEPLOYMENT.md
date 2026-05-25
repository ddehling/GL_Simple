# Per-Project Media Submodules

Each art-piece project's heavy media (sounds, narrative TTS clips, images)
lives in its own private GitHub repo, mounted as a git submodule at
`projects/<id>/media/`. Operators clone the main repo + opt in to one
project's media only — Fan and Weight of Light don't push each other's
assets around.

| Repo | Path | Holds |
|---|---|---|
| `GL_Simple` | (main repo) | All code + project config + shared `media/sounds/` |
| `GL_Simple_fan_media` | `projects/fan/media/` (submodule) | Fan-specific sounds (~656 MB) |
| `GL_Simple_wol_media` | `projects/weight_of_light/media/` (submodule) | WoL-specific sounds (~4 MB) |

Shared sounds usable by both pieces stay in the **top-level `media/sounds/`**
inside the main repo. Only project-specific narrative / ambient / TTS goes
in the per-project submodules.

## Daily workflow

### Pulling updates

Code changes — same as before:
```bash
git pull
```

Media changes (optional; only if someone else updated your project's media):
```bash
git submodule update --remote projects/<id>/media
```

### Fresh-machine setup

```bash
git clone https://github.com/ddehling/GL_Simple.git
cd GL_Simple
bin/setup_and_run.sh    # or bin/setup_and_run.bat on Windows
```

The setup script reads `config.yaml`'s `project:` field and auto-inits
only that project's submodule. The other project's media stays untouched.

To init a project's media manually (e.g. switching which project a
machine runs):
```bash
git submodule update --init projects/<id>/media
```

If the active project's media submodule isn't initialized, `Stories_OGL.py`
prints a warning at boot with the exact fix command — you can't miss it.

## Adding or changing media for an existing project

The `projects/<id>/media/` directory **is** a separate git repo. Treat
it like any other:

```bash
cd projects/fan/media
cp ~/new-track.mp3 sounds/
git add sounds/new-track.mp3
git commit -m "add new ambient track"
git push                       # → GL_Simple_fan_media

cd ../../..                    # back to main repo root
git add projects/fan/media     # update the submodule pointer
git commit -m "fan media: bump submodule for new track"
git push                       # → GL_Simple
```

**The two-step matters.** Pushing the submodule first puts the commit on
GitHub; pushing main second moves the *pointer* main holds to that commit.
If you push main with a pointer to a commit nobody else can fetch, other
operators get errors on `git submodule update`.

## Creating a new project

1. **Scaffold the project** in the layout editor: click **+ New**, fill
   in id + display name + canvas size, click Create. The scaffolder
   writes `projects/<id>/{project.yaml, weather_params.py, event_map.py,
   shaders/, geometry.yaml, media/}` — `media/` is just an empty
   placeholder at this point.

2. **Create the media repo on GitHub** before adding any sounds:
   ```bash
   gh repo create GL_Simple_<id>_media --private \
       --description "Media for the <Name> art piece. Mounted as submodule at projects/<id>/media/."
   ```

3. **Replace the placeholder with the submodule**:
   ```bash
   rm -rf projects/<id>/media
   git submodule add https://github.com/ddehling/GL_Simple_<id>_media.git projects/<id>/media
   git commit -m "wire <id> media submodule"
   git push
   ```

4. **Add media files** inside the submodule and follow the two-step push
   from "Adding or changing media" above.

## Troubleshooting

**`Stories_OGL.py` warns that the media directory is empty.**
The submodule hasn't been initialized on this machine. Run:
```bash
git submodule update --init projects/<active>/media
```

**`git submodule update` errors with "fatal: needed a single revision".**
The main repo's pointer references a commit that hasn't been pushed to the
media repo. Whoever updated the submodule last needs to `git push` from
inside `projects/<id>/media/` first.

**Other operator says they can't fetch the media submodule.**
Make sure the GitHub repo (`GL_Simple_<id>_media`) is shared with them.
Settings → Manage access → Add people.

**Want to roll back the per-project media migration entirely.**
A backup tag was pushed before the rewrite. To restore:
```bash
git fetch --tags
git checkout pre-media-submodules-backup
# (review state, then if you want to make it the new main:)
git branch -f main pre-media-submodules-backup
git push --force origin main
```
This reintroduces the old single-repo layout. The media submodule repos
on GitHub can stay or be deleted; they don't affect the rolled-back main.

## How big should the main repo stay?

After this migration, the main repo on GitHub is ~700 MB (was ~1.4 GB).
That includes git history. The shared `media/sounds/` at the top level
is still in there (a few hundred MB). If shared sounds grow significantly,
consider a third `GL_Simple_shared_media` submodule on the same pattern.
