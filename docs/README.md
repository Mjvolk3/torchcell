# Building the documentation

The site is built with Sphinx from `docs/source/` and deployed to
<https://mjvolk3.github.io/torchcell/> by `.github/workflows/docs.yaml` on every push to
`main`. The steps below reproduce that build locally.

## 1. Environment

Autodoc imports every documented module, so the docs need the full torchcell
environment plus the Sphinx stack in `docs/requirements.txt`. The simplest recipe is a
virtual environment layered on an existing torchcell conda environment, so torch and the
PyG extensions are reused rather than reinstalled:

```bash
CONDA_PY=~/miniconda3/envs/torchcell/bin/python   # your torchcell env's interpreter
VENV=/path/to/docs-venv                           # anywhere outside the repo

$CONDA_PY -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install -r docs/requirements.txt
```

Without a conda environment, follow the workflow instead: install CPU torch, the matching
`torch-scatter` wheel, `docs/requirements.txt`, then `pip install -e .`.

## 2. Build

Run from the repository root:

```bash
# The ontology explorer, published at /ontology/ through html_extra_path.
PYTHONPATH=$PWD "$VENV/bin/python" paper/nature-biotech/scripts/generate_ontology_diagram.py \
  --explorer-only docs/source/_extra/ontology/index.html

# The site itself.
PYTHONPATH=$PWD "$VENV/bin/sphinx-build" -b html docs/source docs/build/html
```

`PYTHONPATH` makes autodoc import this checkout (for example a git worktree) rather than
another installed copy of torchcell. `make -C docs html` runs the same `sphinx-build`
with the Sphinx found on `PATH`, writing to `docs/build/html`.

Open `docs/build/html/index.html` to view the result.

## 3. Layout

- `source/conf.py`: Sphinx configuration.
- `source/index.rst`: the landing page and the three table-of-contents trees.
- `source/guide/`, `source/database/`: narrative pages, written in Markdown (MyST).
- `source/modules/<package>.rst`: one API page per `torchcell` subpackage. Each lists
  the package's members in `autosummary` blocks; Sphinx writes one page per member into
  `source/generated/`, which is git-ignored and rebuilt on every run.
  The pages are generated: after adding or removing a public class, function or module,
  run `PYTHONPATH=$PWD "$VENV/bin/python" docs/gen_api_pages.py` from the repository
  root (`--check` writes nothing and exits 1 when a page is stale).
- `source/_templates/autosummary/`: the member page templates.
- `source/_extra/`: copied verbatim into the site root (the ontology explorer).

A member that fails to import produces an `autodoc` warning in the build log and an
empty page, so check the log after adding a module to an API page.
