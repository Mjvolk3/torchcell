---
id: x9lt2zhuquois6k4051rj2f
title: 08
desc: ''
updated: 1740007227713
created: 1739703537035
---
## 2025.02.16

🏆 - `HeteroCell` Random grid 20 epochs each, small model, `lambda` loss search, `weight_decay`, norms, `num_layers`, 2 layer mlp

- [x] `HeteroCell` make trainable
- [x] `HeteroCell` validate plotting and logging
- [x] `HeteroCell` log metric to determine degeneracy in node embeddings. e.g. #oversmoothing  →  #oversquashing more difficult as you need adjacency information and it is less likely that this is a huge issue during training. We should be able to compute it over the graph prior to training. This will be a nice characterization of potential issues with graphs and could give good justification for expander edges.
- [x] Fixed bug inability to setup `optuna.db` by downgrading sqalalchemy → "To prevent incompatible upgrades prior to updating applications, ensure requirements files are pinned to "sqlalchemy<2.0"

## 2025.02.19

- [x] Followed up on radiant resources
- [x] Followed up on `GH` testing
- [x] Fixed Overfit on batch. bone head error ☠. [[2025.02.19 - Equations How Interacting Wildtype and Integrated|dendron://torchcell/torchcell.models.hetero_cell#20250219---equations-how-interacting-wildtype-and-integrated]]
- [x] Note [[Sync Agent Dirs|dendron://torchcell/wandb.igb.sync-agent-dirs]]
- [ ] Try to relaunch sweep with new name to see if we can add study to db.
- [ ] Fix logging with oputna so that scripts don't write over each other in sbatch launcher. I couldn't tell why the last run failed because of this we wanted
- [ ] Fix `wandb-experiment` logging we sometimes log what it seems like is multiple runs in the same dir. We  

- [ ] Retry optuna sweep with narrower set of hyper parameters. Use `choice` to match `qm9` study.
