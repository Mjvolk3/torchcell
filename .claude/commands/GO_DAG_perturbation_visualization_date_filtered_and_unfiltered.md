## Relevant Files

Relevant files:
experiments/005-kuzmin2018-tmi/scripts/GO_graph_date_filter_comparison.py
experiments/005-kuzmin2018-tmi/scripts/dcell_go_perturbation_visualization.py
torchcell/graph/graph_analysis.py
/Users/michaelvolk/Documents/projects/torchcell/notes/assets/images/go_graph_unfiltered_visualization_2025-05-13-17-54-25.png
/Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/scripts/dcell_batch_005_verify_mutant_state.py
[[Dcell_batch_005_verify_mutant_state|dendron://torchcell/experiments.005-kuzmin2018-tmi.scripts.dcell_batch_005_verify_mutant_state]]
/Users/michaelvolk/Documents/projects/torchcell/torchcell/torchcell.mplstyle
/Users/michaelvolk/Documents/projects/torchcell/torchcell/scratch/load_batch_005.py
[[torchcell.scratch.load_batch_005]]
[[20|dendron://torchcell/user.Mjvolk3.torchcell.tasks.weekly.2025.20]]
dcell.md
/Users/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell_new.py

We want the images to be structured like this... They should have the tree-like structure with the croot node at the top and the different go at their different assigned levels. Leaf nodes are at the bottom.

/Users/michaelvolk/Documents/projects/torchcell/notes/assets/images/scratch.2025.05.13.191447-dcell-dag-small-img.md.2025-05-13-19-14-58.png

## Task

Edit this file experiments/005-kuzmin2018-tmi/scripts/dcell_go_perturbation_visualization.py

We want to plot the GO DAG with root node at the top and leaf nodes at the bottom. We want to color all nodes and edges light gray. Color the perturbed GO nodes using 3 colors. The perturbations using up to 3 colors so for each use a different color. Each Gene perturbation is likely to have many associated GO perturbation nodes. Then also color the connected paths of the GO nodes associated with the gene all the way up to the root node. If a given gene shared a GO term we can apply color and path coloring but lets use alpha value for opaqueness so we can see overlap. We want to do this for every sample in load_batch_005. We want to do this for a graph that is filtered by `2017-07-19` and one that is not filtered by date. Since the `DCell` model is a visible neural network this will help us compare why the correlations on the unfiltered model torchcell/models/dcell_new.py are so much higher ~0.78 versus the date filtered model ~0.45. Please save the images in `notes/assets/images/005_GO_DAG_perturbed_date_filtered` and `notes/assets/images/005_GO_DAG_perturbed_date_filtered`. Since the bach size is 32 we should have 32 images for each dir.

```python
# G_go = filter_by_date(G_go, "2017-07-19")
# print(f"After date filter: {G_go.number_of_nodes()}")
```

- [ ] First change torchcell/scratch/load_batch_005.py to allow for GO_DAG parameterization by date. If `None` then don't date filter.
- [ ] Overwrite experiments/005-kuzmin2018-tmi/scripts/dcell_go_perturbation_visualization.py according to specifications. Loop over every sample. Just start with the first 4 sample because I can see that sample 3 gets learned without date filter and does not get leared with date filter. /Users/michaelvolk/Documents/projects/torchcell/notes/assets/images/dcell_batch_tanh_predictions_vs_targets_2025-05-13-18-12-21.png
