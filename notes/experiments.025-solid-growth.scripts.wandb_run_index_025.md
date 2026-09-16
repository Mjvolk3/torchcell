---
id: xy390ldyzimrcjupdcex9ay
title: Wandb_run_index_025
desc: ''
updated: 1789527913114
created: 1789527913114
---

## 2026.09.15 - The run registry behind the additive-baselines document

Every transformer number in `notes-tex/025-additive-baselines` traces to one of eight runs: the three 010 checkpoints (arm R reference, project `torchcell_010-kuzmin-tmi_equivariant_cell_graph_transformer`), GH 1598 and GH 1640, and the three IGB sequence-embedding runs on arm Q. The registry is one list of pydantic records in the script, each with the configuration it ran, the gene input, schedule, readout, graph penalty and whether it was scored on test. The script verifies each against the W&B API (state, epochs logged, best validation Pearson from the history), writes `results/wandb_run_index_025.json` and the hyperlinked table `tables/t5-wandb-runs.tex`, so no run id is typed into the document.

<https://wandb.ai/zhao-group/torchcell_010-kuzmin-tmi_equivariant_cell_graph_transformer/runs/lzs9pcj3>
<https://wandb.ai/zhao-group/torchcell_010-kuzmin-tmi_equivariant_cell_graph_transformer/runs/yv4r30bi>
<https://wandb.ai/zhao-group/torchcell_010-kuzmin-tmi_equivariant_cell_graph_transformer/runs/c7671wgj>
<https://wandb.ai/zhao-group/torchcell_025-solid-growth_equivariant_cell_graph_transformer/runs/0yw7moue>
<https://wandb.ai/zhao-group/torchcell_025-solid-growth_equivariant_cell_graph_transformer/runs/327csnlk>
<https://wandb.ai/zhao-group/torchcell_025-solid-growth_equivariant_cell_graph_transformer/runs/s1vx2zgw>
<https://wandb.ai/zhao-group/torchcell_025-solid-growth_equivariant_cell_graph_transformer/runs/8aa08xx0>
<https://wandb.ai/zhao-group/torchcell_025-solid-growth_equivariant_cell_graph_transformer/runs/pmkzwwzw>
