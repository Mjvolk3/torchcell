# Linear-probe baselines

A first set of baselines: a frozen representation of each deleted gene and a ridge regression on top. They say how much of each label a linear readout of a gene representation recovers under each kind of split, and they are the floor a model that uses the cell graph should clear.

## Method

Five representations of a gene:

- **Random**: 1,280 standard-normal numbers per gene, fixed seed. It carries no biology.
- **One-hot**: the identity of the gene.
- **ESM-2 650M**: mean over residues of the last layer, protein sequence (1,280 numbers).
- **ProtT5-XL**: mean over residues of the ProtT5-XL-UniRef50 encoder, protein sequence (1,024).
- **CaLM**: mean over codons of the last layer, coding sequence (768).

A single-gene sample is its gene's representation. A gene-pair sample is the two representations concatenated in stored order (`concat`) or added (`sum`). Features are standardized with the training mean and standard deviation, and one ridge regression is fit per task, split and representation, on all targets at once. The penalty is the one of eight values from 0.1 to 1,000,000 with the lowest validation mean squared error. The test part is unlocked once per split, after that choice.

Sequences longer than 1,022 residues or codons are truncated (53 of 4,276 distinct proteins).

## Results

Each cell is the test Pearson correlation, averaged over the targets of the task, as the mean ± sample standard deviation over the seeds of a split family (five seeds for random splits, three for the others). `undefined` means the model predicted a constant, which is what a one-hot representation does for a gene it never saw in training.

```{figure} _generated/linear_probes.svg
:width: 100%

Test Pearson correlation of each linear probe, by split family. Bars are means over seeds and whiskers are sample standard deviations. In the pair panel, plain bars concatenate the two gene representations and hatched bars add them. A missing bar is an undefined correlation.
```

### Growth rate (Campos 2018)

```{include} _generated/probes_ecoli-growth-rate-campos2018.md
```

### Cell shape and cell cycle (Campos 2018)

Mean over the 15 morphology features.

```{include} _generated/probes_ecoli-morphology-campos2018.md
```

### Metabolome (Fuhrer 2017)

Mean over the 7,534 ions.

```{include} _generated/probes_ecoli-metabolome-fuhrer2017.md
```

### Genetic interactions (Babu 2014)

```{include} _generated/probes_ecoli-gene-interaction-babu2014.md
```

Under gene holdout the test pairs are of two kinds: one gene was never in a training pair, or both were not.

```{include} _generated/probes_ecoli-gene-interaction-babu2014_novelty.md
```

## What the numbers say

All of the following are read from the 380 runs in the tables (one run per task, split, representation and pair combination).

- **Single-gene tasks are close to unexplained by a linear readout of one gene's sequence.** The highest test Pearson of any growth-rate run is 0.20, and the three sequence models average 0.08 to 0.10 on growth rate against 0.00 for random features. On the imaging and metabolome tasks no representation is separated from random features by more than the spread over seeds.
- **One-hot is undefined on every single-gene split,** because each gene appears once: a test gene was never seen, and the model predicts the training mean.
- **On the pair task, random features and one-hot do as well as the sequence models.** With random pair splits all five representations reach 0.43 to 0.46. A linear probe is therefore reading which genes are in the pair (each gene's average interaction score), and the sequence models add nothing to that here.
- **Gene holdout removes most of it.** Test Pearson falls to 0.18 to 0.32, and for test pairs in which both genes are new it is 0.17 or lower for every representation, with seed spreads as large as the means.

## Limits

- The ridge penalty chosen on validation is the largest value of the grid (1,000,000) in 121 of the 380 runs, almost all on the single-gene tasks. In those runs the best linear model is nearly the training mean, and a larger penalty was not tried.
- Three seeds per split family (five for random) give wide standard deviations; differences between sequence models are within them in most rows.
- These are frozen mean-pooled embeddings with a linear readout. Nothing here says what a fine-tuned model, a nonlinear readout or a model that uses the cell graph reaches.
- The pair representation `concat` depends on the stored order of the two genes; the pairs were not also presented in the other order.

## Reproducing

Every number comes from `experiments/050-ecoli-benchmark/`: `embed_genes.py` (embeddings), `cluster_proteins.py` and `build_splits.py` (splits), `linear_probes.py` (the runs), `results_page.py` (these tables and the figure). The table of every run, with validation scores, the chosen penalty and four more metrics, is [`linear_probes.csv`](https://github.com/Mjvolk3/torchcell/blob/main/experiments/050-ecoli-benchmark/results/linear_probes.csv).
