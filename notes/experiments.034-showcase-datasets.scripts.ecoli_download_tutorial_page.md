---
id: f1q4zor4lij6uo9nrb6ob03
title: Ecoli_download_tutorial_page
desc: ''
updated: 1794934994226
created: 1794934994226
---


## 2026.11.17 - Runs of the download tutorial and the page fragments

Two subcommands, standard library only.

`run` executes [[experiments.034-showcase-datasets.scripts.ecoli_download_tutorial]] with a given interpreter in an empty work directory, with `TC_DATA_URL`, `DATA_ROOT`, `PYTHONPATH` and `VIRTUAL_ENV` removed from the environment (`--pythonpath` puts one checkout back). It writes one result file per host, `experiments/034-showcase-datasets/results/ecoli_download_tutorial.<host>.json`: the standard output and its sha256, the sha256 of the tutorial file that ran, the seconds each loader took, the bytes each dataset occupies on disk, the four index rows of the endpoint, and the versions of torchcell, Python, pandas, numpy and torch.

`page` reads every result file and every `install_benchmark.<host>.json` ([[experiments.034-showcase-datasets.scripts.install_benchmark]]) and writes the fragments under `docs/source/datasets/ecoli/_generated/download-tutorial/`: one `output_<section>.md` per section of the tutorial, `datasets_table.md`, `timings.md`, `runs.md`, `archives.md` and `install_benchmark.md`. It exits 1 and writes nothing when a host's output sha256 differs from the reference host's, or when the hosts ran different tutorial files, so the page's statement that the output was byte-identical is checked at generation time.

Commands, from the repository root:

```bash
python experiments/034-showcase-datasets/scripts/ecoli_download_tutorial_page.py run \
  --python ~/miniconda3/envs/torchcell/bin/python --pythonpath "$PWD" \
  --installed-from "repository checkout (conda environment)" --host gilahyper \
  --work <empty dir> \
  --out experiments/034-showcase-datasets/results/ecoli_download_tutorial.gilahyper.json
python experiments/034-showcase-datasets/scripts/ecoli_download_tutorial_page.py page
```

On the second machine only the two scripts are copied over, `run` is started with the venv's own interpreter, and the result file is copied back into `results/`.

## 2026.11.17 - Release check table, and the runs redone on torchcell 1.8.5

`page` also reads `experiments/034-showcase-datasets/results/install_release_check.<host>.json`, the result of [[experiments.034-showcase-datasets.scripts.install_benchmark]] run on the release after the one the comparison timed, and writes `install_release_check.md`: one row per host and arm with the install seconds, the venv size and whether a loader imports in the new environment.

Both tutorial result files were replaced after torchcell 1.8.5 reached PyPI. `ecoli_download_tutorial.gilahyper.json` is the checkout at the 1.8.5 bump commit, and `ecoli_download_tutorial.radiant.json` is a new uv venv on the Radiant VM that installed `torchcell==1.8.5` from PyPI with the CPU build of torch. The first versions of the two files, from the 1.8.4 checkout and from a wheel built on the branch, are in the history of PR #1166. The venv that `pip install torchcell==1.8.5` made on Radiant was run as well; its result is `ecoli_download_tutorial_pip_check.radiant.json`, a name `page` does not read, so it is not a row on the page. All five runs printed output with the same sha256, `0c5323ce7d5b6aec76345c6945a3ad7927f0b3f7db5913e1140bceab27a48068`.

## 2026.11.17 - Fragments split over three pages, figures copied by run

- `run --figures-out <dir>` copies the SVG figures the tutorial drew into the docs tree; the reference host's run is the source of the figures on the page.
- `page` writes the printed output, the dataset table, the download seconds and the archive table to `docs/source/datasets/ecoli/_generated/tutorial/`, and the install timings, the release check and the run table to `docs/source/guide/_generated/install-timings/`.
- One output fragment per section banner of the tutorial, so a new cell needs no change here.
