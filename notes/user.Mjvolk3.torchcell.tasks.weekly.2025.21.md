---
id: byqxlgenke415eplp31l2fk
title: '21'
desc: ''
updated: 1748899481801
created: 1747592127726
---

## 2025.05.19

- [x] Dcell with strata.

## 2025.05.20

- [x] [[2025.05.20 - Investigatin DCell Absurdly Slow Iteration|dendron://torchcell/torchcell.models.dcell#20250520---investigatin-dcell-absurdly-slow-iteration]]
- [x] [[2025.05.20 - DCell with Strata|dendron://torchcell/torchcell.scratch.load_batch_005#20250520---dcell-with-strata]]
- [x] Get `DCell` to work on device.

**BATCH SIZE = 32**

On Delta GPU:

`Loss: 0.001867, Corr: 0.4769, Time: 100.151s/epoch:  62%|███  62/100 [1:44:31<1:04:03, 101.15s/it]`

On M1 CPU:

Much faster. This is when we realized we need some other solution.

`20s/it`

## 2025.05.21

- [x] [[2025.05.21 - Data Structure for Pyg Message Passing|dendron://torchcell/torchcell.scratch.load_batch_005#20250521---data-structure-for-pyg-message-passing]]
- [x] [[2025.05.21 - Data Structure Don't Propagate Edge Info|dendron://torchcell/torchcell.scratch.load_batch_005#20250521---data-structure-dont-propagate-edge-info]]
- [x] Revert `DCell` I think we can make this better with mutant state matrix. Get mutant state right.
- [x] Model initialization.

## 2025.05.22

- [x][[2025.05.23 - One DCell Module Processing|dendron://torchcell/torchcell.models.dcell#20250523---one-dcell-module-processing]]
- [x] Get `Dcell` working. [[2025.05.22 - DCell overfit on M1|dendron://torchcell/torchcell.models.dcell#20250522---dcell-overfit-on-m1]]

## 2025.05.23

- [x] Experiment without alpha. [[2025.05.23 - DCell overfit on M1 without alpha|dendron://torchcell/torchcell.models.dcell#20250523---dcell-overfit-on-m1-without-alpha]]
- 🔲 Consolidate and commit.
- 🔲 Test speed on GPU.

| device        | batch_size | workers | time/epoch (s) | time/instance (s) | notes        |
|:--------------|:-----------|:--------|:---------------|:------------------|--------------|
| M1            | 32         | 4       | 11.0           | 0.344             |              |
| A100 80g mmli | 32         | 4       | 7.0            | 0.219             |              |
| A100 80g mmli | 256        | 8       | 49.33          | 0.193             | more gpu use |
| A100 80g mmli | 1,500      | 16      | 292.14         | 0.195             | uses 5b vRAM |
| A100 80g mmli | 3,000      | 8       | 582.93         | 0.194             | gpu 35% 20 gb |

Things aren't looking that good.

**Target is time/instance = 0.05 s**

## 2025.05.24

```bash
mjvolk3@biologin-3 ~/projects/torchcell $ squeue -u $USER
  JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
  1808464      mmli max_DCel  mjvolk3 PD       0:00      1 (Resources)
  1796148      mmli    DCell  mjvolk3  R 8-17:35:54      1 compute-5-7
  1799739      mmli    DCell  mjvolk3  R 4-17:44:17      1 compute-5-7
  1807864      mmli     bash  mjvolk3  R      48:54      1 compute-5-7
```

