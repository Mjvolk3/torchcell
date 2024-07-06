---
id: goislb7hg0rffoj4hhlzawl
title: '29'
desc: ''
updated: 1720233379356
created: 1719839145091
---

## 2024.07.06

- [ ] While getting coffee make sure that we can connect to the neo4j browser.

## 2024.07.05

- [ ] Expand [[paper-outline-02|dendron://torchcell/paper.outline.02]]
- [ ] Make sure ports are getting forwarded correctly and that we can connect to the database over the network. We need to verify that we can connect with the neo4j browser.
- [ ] Try to link docker and slurm with `cgroup`
- [ ] Run build bash script for testing.
- [ ] `gh` Test build under resource constraints.
- [ ] Change logo on docs → to do this we need a `torchcell_sphinx_theme`. → cloned, changed all `pyg_spinx_theme` to `torchcell_sphinx_theme`, pushed, trying rebuild.

## 2024.07.04

- [x] Get docker going again on `gh` run test sbatch → need to bump version for small.
- [x] Bump version → fix. colon? → looks like we do need colon.
- [x] Run small kg → Verify that things update on push to main.
- [x] Correct setup script so we can start fresh from `/scratch`. We deleted `wandb-experiments` but they are saved online. → I think setting `wandb-experiments` will automatically create them when they are necessary.
- [x] Make sure that permissions are set such that we can still view all of the files in `/database`
