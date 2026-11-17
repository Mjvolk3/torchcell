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
