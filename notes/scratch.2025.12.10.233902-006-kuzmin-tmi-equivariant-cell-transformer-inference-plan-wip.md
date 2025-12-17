---
id: zeriob9n9d275a04ohqj2sw
title: 233902 006 Kuzmin Tmi Equivariant Cell Transformer Inference Plan Wip
desc: ''
updated: 1765831209576
created: 1765431566080
---

## Where Current Inference Data is Stored

```bash
michaelvolk@M1-MV-6 torchcell % tree /Users/michaelvolk/Documents/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi
/Users/michaelvolk/Documents/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi
├── 001-small-build
│   ├── aggregation
│   │   └── lmdb
│   │       ├── data.mdb
│   │       └── lock.mdb
│   ├── data_module_cache
│   │   ├── index_details_seed_42.json
│   │   ├── index_seed_42.json
│   │   ├── perturbation_subset_1e01_metabolism
│   │   │   ├── index_1e01_seed_42.json
│   │   │   └── index_details_1e01_seed_42.json
│   │   ├── perturbation_subset_1e02_metabolism
│   │   │   ├── index_1e02_seed_42.json
│   │   │   └── index_details_1e02_seed_42.json
│   │   ├── perturbation_subset_1e03_metabolism
│   │   │   ├── index_1e03_seed_42.json
│   │   │   └── index_details_1e03_seed_42.json
│   │   ├── perturbation_subset_1e04_metabolism
│   │   │   ├── index_1e04_seed_42.json
│   │   │   └── index_details_1e04_seed_42.json
│   │   ├── perturbation_subset_5e03_metabolism
│   │   │   ├── index_5e03_seed_42.json
│   │   │   └── index_details_5e03_seed_42.json
│   │   └── perturbation_subset_5e04_metabolism
│   │       ├── index_5e04_seed_42.json
│   │       └── index_details_5e04_seed_42.json
│   ├── deduplication
│   │   └── lmdb
│   │       ├── data.mdb
│   │       └── lock.mdb
│   ├── processed
│   │   ├── dataset_name_index.json
│   │   ├── dataset_name_index.json.lock
│   │   ├── experiment_types.json
│   │   ├── experiment_types.json.lock
│   │   ├── gene_set.json
│   │   ├── gene_set.json.lock
│   │   ├── is_any_perturbed_gene_index.json
│   │   ├── is_any_perturbed_gene_index.json.lock
│   │   ├── label_df.parquet
│   │   ├── lmdb
│   │   │   ├── data.mdb
│   │   │   └── lock.mdb
│   │   ├── perturbation_count_index.json
│   │   ├── perturbation_count_index.json.lock
│   │   ├── phenotype_label_index.json
│   │   ├── phenotype_label_index.json.lock
│   │   ├── pre_filter.pt
│   │   └── pre_transform.pt
│   └── raw
│       ├── experiment_reference_index.json
│       ├── gene_set.json
│       └── lmdb
│           ├── data.mdb
│           └── lock.mdb
├── inference_0
│   ├── inference_predictions_2025-07-03-19-09-04.csv
│   ├── inference_predictions_2025-07-03-21-47-30.csv
│   ├── inference_predictions_2025-07-07-17-17-03.csv
│   ├── inference_predictions_2025-07-08-16-21-55.csv
│   ├── processed
│   │   ├── lmdb
│   │   │   ├── data.mdb
│   │   │   └── lock.mdb
│   │   ├── pre_filter.pt
│   │   └── pre_transform.pt
│   └── raw
│       └── triple_combinations_list_2025-07-03-19-02-25.txt
└── projected_results
    └── wandb_export_2025-10-22T11_16_25.781-05_00.csv

22 directories, 49 files
```

## Relevant Files For Creating the Dataset

`experiments/006-kuzmin-tmi/scripts/inference_dataset.py`
`experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_gi_inference.py`
`experiments/006-kuzmin-tmi/scripts/generate_triple_combinations.py`
`experiments/006-kuzmin-tmi/scripts/igb_optuna_hetero_cell_bipartite_dango_gi-ddp_inference.slurm`

the optuna naming is a misnomer. Don't think optuna is used anywhere.

"""
Generate triple gene combinations from selected genes.
Filters out triples where any pair has fitness < 0.5 in DMF datasets.
Also filters out triples that already exist in TMI datasets.
"""

## Current Model That We Need Inference To work For on 006-kuzmin-tmi

The data loading has changed and so we will need to save a different inference dataset. Let's call this one `006-kuzmin-tmi/inference_1`.

`/Users/michaelvolk/Documents/projects/torchcell/torchcell/models/equivariant_cell_graph_transformer.py`

```python
Note how this file uses load batch. @hydra.main(
    version_base=None,
    config_path=osp.join(os.getcwd(), "experiments/006-kuzmin-tmi/conf"),
    config_name="equivariant_cell_graph_transformer",
)
def main(cfg: DictConfig) -> None:
    """Main training function for Equivariant Cell Graph Transformer."""
    import matplotlib.pyplot as plt
    from dotenv import load_dotenv
    from torchcell.timestamp import timestamp
    from torchcell.scratch.load_batch_006_perturbation import (
        load_perturbation_batch,
    )
```

This will tell you how the data is being loading/processed. This difference from previous time we did inference with GeneINteractionDango model.

## To Start Plan Discussion

- We still want real time monitoring as in `/Users/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_gi_inference.py`
- Inference is being done on triples still so we need to generate these as in `/Users/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/generate_triple_combinations.py` read here for more info [[Generate_triple_combinations|experiments.006-kuzmin-tmi.scripts.generate_triple_combinations]]
- For inference we want to create dataset like this. `experiments/006-kuzmin-tmi/scripts/inference_dataset.py`
- in `experiments/006-kuzmin-tmi/scripts/generate_triple_combinations.py` we find a file outputted by `/Users/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/rank_metabolic_genes.py` ... We want to refine this selection. The list of genes that is used to makes triples comes from the follwowing `/Users/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/results/inference_preprocessing/selected_genes_list_2025-07-03-11-37-27.txt`... we want to expand this to include more genes.

## Selecting Additional Genes to Add For Creation of Triples

Here are the three datasets we need to consider for collecting genes:

Note that `Sameith` has two datasets. Singles and doubles.

- `torchcell/datasets/scerevisiae/kemmeren2014.py`
- `torchcell/datasets/scerevisiae/ohya2005.py`
- `torchcell/datasets/scerevisiae/sameith2015.py`

We have nearly all viable single gene deletion mutants expression profiles with kemmeren2014. We have another study with single gene deletion expression profiles in sameith2015. We also have all single gene deletion morphology in ohya2005. Then we have some double mutant gene knockouts with expression profiles. The idea is to use these datasets to select some genes in the process of designing triple mutants. We have to construct them sequentially and will be creating some double mutant for which we don't have morphology data for because this hasn't been measured or morphology. I want to try to select some of the double mutants from `sameith` in our gene pool so that we can have controls when we compare against our own transcriptomic data.

Our experimental budget is limited to starting with 2 potential scenarios depending on some post hoc analysis of the inference results. We will either start with 12 or 24 single mutants. Getting this first set of genes is probably the most important step as we can then pick and choose how we want to construct combinations as results come in. The idea is to explore the boundaries of interestingness when it comes to the mutant space. Our inference model currently is just being used to predict high order interactions. ie triple interactions. We have all the single mutant and double mutant fitness and we are interesting in fast growing yeast strains for bioreactor applications. But we are also interested in the tradeoffs that are required for achieving positive interactions. if you have high single mutant fitness, positve double and triple interactions you will have a mutant that grows much faster than wildtype and you could justify module changes from engineering efforts. No regression. It is for this reason that we are primarily interested in predicting triple mutant interactions.

| num single mutants | $n^2$ | $n^3$  |
|:-------------------|:------|:-------|
| 12                 | 144   | 1,728  |
| 24                 | 576   | 13,824 |

We will measure fitness as we create these mutants with SGA imaging method. Then for select mutants we record their RNAseq data, possibly for all if budget permits. After this we will also record their morphology. This means we want to select mutants that have increasingly positive growth patterns and also some high likelihood to show something interesting in terms of morphology and expression. It is for this reason that we want to also consider the morophology and expression datasets in the gene selection.

Criterion are as follows for the construction of triple mutants:

1. I want to expand the original 220 genes from `experiments/006-kuzmin-tmi/results/inference_preprocessing/selected_genes_list_2025-07-03-11-37-27.txt`. I want to include genes that are have a significant amount of morphological change. I think we can start by adding the top 1000 genes with many changed morphological factors. We can get that from some intermediate data that comes from the ohya paper. We want to basically just rank systematic gene names by the last column which is the most stringent threshold for number of significant parameters changed for the morphological mutant. 2000 genes should be global param at top of script. hard coded for now is fine. I imagine when we take intersection between a few different genes sets this will reduce to a number we can do inference on.

`data/torchcell/experiments/006-kuzmin-tmi/inference_1/Ohya - SI - table3 ORF Statisitcs.xlsx`

from excel table

```html
"<span style=""font-weight: bold;text-decoration: underline;font-family: Arial;font-size: 11px;""> Table 3 </span><span style=""text-decoration: underline;font-family: Arial;font-size: 11px;"">ORF statistics                                                                                                                                                                                
</span><span style=""font-family: Arial;font-size: 11px;"">Number of parameters in which the disruptant is judged to be a morphological mutant with the threshold</span>"        
<span style="font-family: Arial;font-size: 11px;">ORF name</span> <span style="font-family: Arial;font-size: 11px;">gene name</span>  0.001  0.000100 0.000010 0.000001 
<span style="font-family: Arial;font-size: 11px;">YAL002W</span> <span style="font-family: Arial;font-size: 11px;">VPS8</span>  3  0 0 0 
<span style="font-family: Arial;font-size: 11px;">YAL004W</span> <span style="font-family: Arial;font-size: 11px;">YAL004W</span>  2  1 0 0 
```

2. The next step is to select a set of genes with changing transcriptional activity from kemmeren. I think for this we want to pick the genes with that are responsive. For now we can just pick from the responsive mutants list.

`torchcell/datasets/scerevisiae/kemmeren2014.py`

```python
@register_dataset
class MicroarrayKemmeren2014Dataset(ExperimentDataset):
    # GEO accessions for the dataset
    geo_accession_responsive = "GSE42527"  # Responsive mutants
    geo_accession_nonresponsive = "GSE42526"  # Non-responsive mutants.
```

We can take all of the genes from this list.

3. We want all of the genes from the doubles in sameith added to the inference list.

4. We want to keep metabolic and kinase genes as originally laid out here `experiments/006-kuzmin-tmi/scripts/rank_metabolic_genes.py`  but we want to expand the selection a little more. We want genes associated with beta carotene, a mutant yeast Betaxanthin strain, and amino acid production. This expansion is absolutely necessary.

5. FITNESS_THRESHOLD = 1.0  # Pairs with fitness below this are excluded ... we have upped from 0.5 so we can try to see if any combinations have continual improvement... we will run it again if we get to

- Last time we had 0.5 and think started with 200 genes and got down to a little under a million triple combinations to run inference on. This is pretty good target as we can complete this inference in a reasonable amount of time.
Last thing is the slurm script for running infernece. We need to get to that stage to run inference then our this task will be complete

## WIP Task List
