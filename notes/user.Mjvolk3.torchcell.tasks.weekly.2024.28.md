---
id: 0j3fly72kssdbapr5eqchti
title: '28'
desc: ''
updated: 1720114915301
created: 1719334149824
---
## 2024.06.27

- [x] Run tests → can get db to build but slurm doesn't talk with docker. → `cgroup`?
- 🔲 Try to link docker and slurm with `cgroup`
- 🔲 Run build bash script for testing.
- 🔲 `gh` Test build under resource constraints.
- 🔲 Change logo on docs → to do this we need a `torchcell_sphinx_theme`. → cloned, changed all `pyg_spinx_theme` to `torchcell_sphinx_theme`, pushed, trying rebuild.

## 2024.06.26

- [x] `gh` Test build from bash script → worked first time, text again...
- [x] `gh` Test build on bash script → works. Now we need to try to rerun under resource limitations
- [x] Use directory initialization for more reproducibility in setting up builds
- [x] Troubleshoot builds → ongoing...
- 🔲 `gh` Test build under resource constraints.
- 🔲 Change logo on docs → to do this we need a `torchcell_sphinx_theme`. → cloned, changed all `pyg_spinx_theme` to `torchcell_sphinx_theme`, pushed, trying rebuild.

## 2024.06.25

- [x] Sync and stop runs on `gh` → also made new plots
- [x] Add gpu to gilahyper, now we have 4 gpus.
- [x] Update cgroup → not cgroup but `gres.conf` and `slurm.conf`
- [x] Bash build → only works inside docker container `docker exec -it tc-neo4j /bin/bash`.

- [x] Get docs going on readthedocs. should should make the update cycles on docs much faster which might get me to document. Next step would be to `make` documentation with every release. → requirements issues with need to install `torchcell` therefore `pytorch-scatter` which needs to be installed before... trying to do in `requirements.txt`. → works and think that it is pushes to read the docs on push.
- 🔲 `sbatch` build
- 🔲 Troubleshoot why docker container is crashing
- 🔲 Verify connection to database from M1
- 🔲 Compare GPU accelerated Random Forest v CPU random forest.
- 🔲 Per model, per scale, performance v num_params for all models. Double check to see if we can get curving lines, look for examples first.
