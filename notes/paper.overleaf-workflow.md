---
id: xriyis21e8g0y47zuwhkk9s
title: Overleaf Workflow
desc: ''
updated: 1781573688303
created: 1781573688303
---

## 2026.06.15 - Paper writing + Overleaf publishing process

How the manuscript pipeline currently works, and best practices. Companion to the
content outline [[paper.nature-biotech-cgt-outline]]. The live manuscript lives in
`paper/nature-biotech/` (on `main`); figure-prep specifics are in
`paper/nature-biotech/figures/README.md`.

### Architecture: two tiers (workshop vs shared)

- **Workshop (canonical, private, full):** `paper/nature-biotech/` in the torchcell
  repo, on `main`. You edit here. Versioned with torchcell. Holds everything:
  section sources, all build wrappers, char-budget tags, draft scaffolding.
- **Shared (Overleaf, curated):** `~/Documents/projects/torchcell-overleaf` -- a
  clone of the Overleaf project's git repo. Collaborators see this. Receives a
  curated subset, published by a script. Never hand-edited locally.

The workshop is the **source of truth for the manuscript text**. Overleaf is the
compile/preview + co-author surface.

### The Overleaf git repo

- Overleaf (Student plan or higher) exposes each project as a git remote:
  `https://git.overleaf.com/<project-id>` (derive it by swapping
  `www.overleaf.com/project/` -> `git.overleaf.com/` in the project URL, or
  project sidebar -> Integrations -> Git). Current project id:
  `6a29f691da9f11a8cd60a308`.
- Auth is **token-based**: username `git`, password = a Git authentication token
  (Overleaf Account Settings -> Git integration). Cache it with
  `git config --global credential.helper store` so pushes are non-interactive.
- "Git integration" (direct clone URL) is distinct from "GitHub sync" (links to a
  GitHub repo). We use the direct git integration.

### File structure (paper/nature-biotech/)

- `content.tex` -- orchestrator; `\input`s `sections/{frontmatter,introduction,
  results,discussion,methods,backmatter}.tex`. Edit the section files.
- Build wrappers (each `\input{preamble}` + `\input{content}`):
  - `submission.tex` -- official Springer Nature single-column (journal upload).
  - `editing.tex` -- single-column, print-approx margins + visible char-budget tags
    (`\cb{}`). Our drafting/typeset look.
  - `twocolumn.tex` -- `iicol`, published-like Nature double column.
  - `figure-proto.tex` -- true-scale (180x170 mm) figure-sizing canvas.
- `preamble.tex` -- shared packages + helpers (`\figph` placeholder; `\cb` is
  defined per-wrapper: gray in editing, empty elsewhere).
- `sn-jnl.cls` + `sn-nature.bst` -- Springer Nature template (covers Nature
  Portfolio). `references.bib` -- bibliography.
- `figures/` -- exported figure PDFs + the figure-prep guide (README.md).

### Build (make targets; run from repo root or paper dir)

- `make paper` -> submission + editing + twocolumn PDFs (via Tectonic).
- `make paper-submission` / `-editing` / `-twocolumn` / `-figproto` -> one PDF.
- `make paper-figures` -> export draw.io sources to vector PDF (headless).
- `make paper-flat` -> flatten to a single `.tex` for journal submission.
- `make paper-sync` -> publish curated subset to Overleaf + push.
- `make paper-clean` -> remove generated PDFs.
- Tectonic: auto-uses installed `tectonic`, else `/tmp/tectonic`. Install permanently
  with `conda install -c conda-forge tectonic`.

### Publishing to Overleaf (`make paper-sync` = sync-overleaf.sh)

1. Registers a local-only manifest exclude, then `git pull` (gets collaborator
   changes from Overleaf).
2. Removes files **we** previously published but no longer do (manifest diff) --
   our own deletions propagate.
3. Copies the curated set: `submission.tex` -> `main.tex`, plus `content.tex`,
   `preamble.tex`, `sn-jnl.cls`, `sn-nature.bst`, `references.bib`, `editing.tex`,
   `twocolumn.tex`, `sections/*`, `figures/*`.
4. Commits + pushes to Overleaf.

Curate what crosses over by editing `SHARE_FILES` in `sync-overleaf.sh`. Workshop-
only files (figproto, READMEs, Makefile, flatten_tex.py, sn-article.tex) are not shared.

### Figures (true-to-size, from assets)

- Compose in **draw.io**; sources live in `notes/assets/drawio/` (per the
  copy-from-assets rule -- never write image data straight to Overleaf).
- Export to **vector PDF** programmatically: `make paper-figures` runs drawio-desktop
  headless via `xvfb-run` + the AppImage (`APPIMAGE_EXTRACT_AND_RUN=1 ... --no-sandbox
  --disable-gpu -x -f pdf --crop`). One Makefile rule per figure.
- **True-to-size:** include with `\includegraphics{figures/figN.pdf}` (NO width/
  height) -> prints at the size drawn in draw.io. Resize by changing the drawing,
  not the LaTeX. Adding `width=`/`height=` rescales and breaks WYSIWYG.
- Print box: full-width **180 mm** (709 draw.io units), single column **88 mm**
  (347 units), max height **170 mm** (669 units); fonts 5-7 pt. See
  `paper/nature-biotech/figures/README.md`.

### Journal submission constraints (Springer Nature / Nature Biotechnology)

- **Single flat `.tex`** -- no `\input`; `.bbl` pasted in. Use `make paper-flat`
  (-> `submission-flat.tex`). Multi-file is only for the workshop + Overleaf.
- Abstract <= 150 words; main text <= 3,000 words; <= 6 display items; <= 50 refs.
- Figures: vector (PDF/EPS) for line art, >= 300 dpi for photos.
- Confirmed via nature.com/nbt + springernature.com (2026-06).

### Collaboration model (how it currently works)

- Collaborators **adding new files** in Overleaf (images, etc.): **preserved** --
  the sync pulls first, and the manifest only deletes files we publish, so
  never-published files are never touched.
- Collaborators **editing files we publish** (`content.tex`, `sections/*`,
  `main.tex`, ...): **overwritten** on the next `make paper-sync` (workshop is the
  source of truth; their edit survives in Overleaf git history but the live file
  reverts to ours).
- Best practice for now: collaborators add **figures/new files** + use Overleaf
  **comments / track-changes** for prose; we fold prose into the workshop.

### #future -- full bidirectional pull-back

Desired: since Overleaf is a real git repo, support true two-way editing on managed
files -- **commit our state, `git pull` from Overleaf, review/merge collaborator
edits (resolve conflicts), then push** -- instead of the current copy-over that
clobbers managed files. Likely a `make paper-pull` (merge Overleaf -> workshop)
companion to `make paper-sync`, or restructure so the Overleaf clone is a genuine
bidirectional mirror (git subtree, or workshop-as-Overleaf-clone). Deferred.

**[Done 2026.06.15 -- see the section below: `make paper-pull` implements this via a
per-file 3-way merge. Collaborators now edit managed files directly in Overleaf and
we pull them in; no collaborator-owned-section split needed.]**

### Tooling notes / gotchas

- **Tectonic** is the LaTeX engine (handles sn-jnl's newer packages that the
  frozen system TeX Live 2020 lacks, e.g. `cuted.sty`). `/tmp/tectonic` is the
  musl static binary; install via conda-forge for durability.
- **drawio-desktop AppImage** at `/tmp/drawio.AppImage` is ephemeral and needs
  `chmod +x`; move to a stable path and set `DRAWIO=` in the Makefile/env so
  `make paper-figures` survives a reboot. Runs headless via `xvfb-run` (Electron
  needs a display, even to export).
- **Overleaf main document:** set in Overleaf to `main.tex` (default), or switch to
  `editing.tex` / `twocolumn.tex` to compile those views.
- **VS Code:** `torchcell-overleaf` is in the workspace; the "Overleaf Workshop"
  extension (cookie login) edits the Overleaf project directly; "LaTeX Workshop"
  has a two-page spread view (`tomoki1207.pdf` does not).
- **Authorship:** co-first authors via `\equalcont{...}` in `sections/frontmatter.tex`
  (adds a dagger + "These authors contributed equally" note); `\author*` marks
  corresponding authors.
- **Local `main`** has been running ahead of `origin/main` -- remember to
  `git push origin main` to land the paper work on GitHub.
## 2026.06.15 - Bidirectional pull-back implemented (`make paper-pull`)

Collaborators now **edit managed files directly in Overleaf** (`content.tex`,
`sections/*`, ...) and we **pull their edits back into the workshop** -- no
collaborator-owned section split, no clobbering. This closes the loop the `#future`
section above described.

### `make paper-pull` (= `paper/nature-biotech/paper-pull.sh`)

The mirror image of `paper-sync`. For every file we publish it does a **per-file
3-way merge** (`git merge-file`):

- **BASE** = the state we last published (the common ancestor)
- **OURS** = current workshop file
- **THEIRS** = current Overleaf file (after collaborator edits)

Non-overlapping edits merge silently; lines **both** sides changed get
`<<<<<<<`/`=======`/`>>>>>>>` conflict markers. Results are written into the
**workshop** files only -- it **never pushes and never auto-commits**. Because the
workshop is itself tracked by torchcell git, every pulled-in change is just a
reviewable working-tree diff with a full undo (`git checkout`).

Handles the `submission.tex` <-> `main.tex` rename (workshop `submission.tex` is
published as Overleaf `main.tex`). **Figures** are binary -> reported, not merged.
**New files** collaborators added are reported but left on Overleaf (already
preserved by `sync-overleaf.sh`'s manifest logic; pull them in by hand if wanted).

### Conflicts -> VS Code Merge Editor

When a file conflicts (we and a collaborator changed the **same region** since the
last publish), `paper-pull` doesn't just leave bare markers -- it registers a real
**git unmerged entry** in the torchcell repo: index stage 1 = base, stage 2 = ours
(workshop), stage 3 = theirs (Overleaf), with the working file holding the
`<<<<<<<`/`=======`/`>>>>>>>` markers. `git status` then shows the file as `UU`, so
VS Code lists it under **Source Control -> Merge Changes** and offers **"Resolve in
Merge Editor"** (Current = workshop, Incoming = overleaf, toggle **Base** to see the
last-published ancestor). Resolve, save, `git add` -> the `UU` clears to a normal
staged change. Non-VS-Code users can still edit the markers by hand. `paper-pull`
exits **2** when there are conflicts (0 = clean, 2 = needs resolution). The unmerged
staging only happens when the workshop lives in a git repo (real use); a plain
`WORKSHOP_DIR` copy just gets markers.

Key Overleaf gotcha behind all this: **a web-app save is NOT a per-save git commit**.
Overleaf's git bridge **squashes everything since your last pull into one
`"Update on Overleaf."` commit**, so you can't merge against its commit graph -- which
is exactly why we diff against our own recorded BASE instead.

### Merge base tracking

`sync-overleaf.sh` now writes the just-published commit SHA to
`.torchcell-sync-base` in the Overleaf clone (local-only, `.git/info/exclude`'d
like the manifest). `paper-pull` reads it as the merge BASE, falling back to the
most recent "Publish manuscript from torchcell workshop" commit if absent.

### The reconciliation loop

1. Collaborators edit managed files in Overleaf.
2. `make paper-pull` -> their edits merge into the workshop; resolve any conflicts
   via VS Code's Merge Editor (Merge Changes group).
3. Review with `git diff`, commit in torchcell (workshop is now the reconciled
   source of truth).
4. `make paper-sync` -> republish. No clobber, because the workshop already
   contains their changes.

### Validation

Verified end-to-end in a local sandbox (a bare repo standing in for the Overleaf
remote, so the real shared project is never touched): a collaborator's
non-overlapping `content.tex` edit merged cleanly alongside our edit; a same-line
`introduction.tex` edit produced proper conflict markers; a new collaborator file
was reported and left on Overleaf. Also verified against the **live** Overleaf
project (a real web-app save pulled back via `git pull` from `git.overleaf.com`),
and verified that a conflict in a git-backed workshop produces a real `UU` unmerged
entry with stages 1/2/3 (so VS Code's Merge Editor opens) that clears on
`git add` after resolution. Test it yourself with
`OVERLEAF_DIR=... WORKSHOP_DIR=... bash paper/nature-biotech/paper-pull.sh` against
copies.
