---
id: x9lt2zhuquois6k4051rj2f
title: 08
desc: ''
updated: 1739703542115
created: 1739703537035
---
## 2025.02.16

🏆 - `HeteroCell` Random grid 20 epochs each, small model, `lambda` loss search, `weight_decay`, norms, `num_layers`, 2 layer mlp

- [x] `HeteroCell` make trainable
- [x] `HeteroCell` validate plotting and logging
- [x] `HeteroCell` log metric to determine degeneracy in node embeddings. e.g. #oversmoothing  →  #oversquashing more difficult as you need adjacency information and it is less likely that this is a huge issue during training. We should be able to compute it over the graph prior to training. This will be a nice characterization of potential issues with graphs and could give good justification for expander edges.
- [ ]
