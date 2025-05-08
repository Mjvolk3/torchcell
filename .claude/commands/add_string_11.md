## Goal

We are writing a pipeline for training the Dango Model. The way they set lambda is based on the difference

## Dango Pipeline Files

We have a pipeline for training Dango model that we are working on.

### Dango Paper

'/Users/michaelvolk/Zotero/storage/AFBC5E89/Zhang et al. - 2020 - DANGO Predicting higher-order genetic interaction.pdf'

From the paper: "The hyperparameter λ reflects to which degree the zeros in the network are considered no interactions. In this work, we calculated the percentage of decreased zeroes from STRING database v9.1 to v11.0 for each network, ranging from 0.02% (co-occurrence) to 2.42% (co-expression)."

This means we need to include v11.0 in our graph data object.

### Dango Pipeline - Main

torchcell/models/dango.py
experiments/005-kuzmin2018-tmi/scripts/dango.py
experiments/005-kuzmin2018-tmi/conf/dango_kuzmin2018_tmi.yaml
torchcell/trainers/int_dango.py

### Dango - Graph Objects Used

/Users/michaelvolk/Documents/projects/torchcell/torchcell/graph/graph.py

## Tasks

Steps:

1. Update /Users/michaelvolk/Documents/projects/torchcell/torchcell/graph/graph.py add v11.0 of string. Also support it in `build_gene_multigraph` method.

2. Write this file /Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/scripts/dango_lambda_determination.py and print out the `labmda` value according to the papers calculations based on string 9 and string 11 comparison.

3. Then we will write a lambda_values function based on this in experiments/005-kuzmin2018-tmi/scripts/dango.py 

This is the current version.

```python
    lambda_values = {}
    for edge_type in wandb.config.cell_dataset["graphs"]:
        if edge_type in ["string9_1_coexpression"]:  # > 1% zeros decreased
            lambda_values[edge_type] = 0.1
        else:  # <= 1% zeros decreased
            lambda_values[edge_type] = 1.0
```

***

Summarize the tasks you need to do and wait on my command.