# E. coli tasks

Four tasks from four built *Escherichia coli* K-12 datasets. Each one asks a model to predict a measured phenotype of a deletion strain from the genes that were deleted.

```{toctree}
:maxdepth: 1

processing
results
```

## The tasks

```{include} _generated/tasks.md
```

A record of a dataset is a benchmark sample when it perturbs the number of genes the task is about (one, or two for the pair task) and each of them resolves to one gene of the genome that has a protein sequence. The other records stay in `task.excluded`, each with its reason, and the datasets themselves are not filtered.

## The short import

```python
from torchcell.tasks import load_task

task = load_task("ecoli-growth-rate-campos2018")
split = task.split("seqid30_s0")

task.genes[split.train]   # locus tags of the deleted genes, (n, 1) or (n, 2)
task.y[split.train]       # labels, (n, n_targets); nan where a target was not measured
task.genome               # the genome: task.genome["BW25113_0002"].protein.seq
task.score(task.y[split.val], prediction)   # Pearson, Spearman, R2, MSE, MAE
```

`load_task` downloads the built dataset store when `TC_DATA_URL` is set, fetches the genome the records name when it is absent, matches records to genes, reads the labels and attaches the pinned splits. The split files ship inside the package and are checked against a sha256 recorded in the code, and the samples built on your machine must equal the samples the file was written for, row for row, or the call raises.

Because every script reaches the processed task through this one import, a submitted script can be checked for it: a script that builds its own samples or splits does not contain `load_task`, and a script that reads the test part contains `unlock_test`.

## The test part is locked

`split.train` and `split.val` are plain index arrays. `split.test` raises `TestSplitLocked`. The test indices come from

```python
test = split.unlock_test(reason="final evaluation of the model selected on validation")
```

which needs a reason and emits a `TestSplitWarning` on every call. Fit on train, choose the model and every hyperparameter on validation, and unlock the test part once for the final number.

## The splits

Train, validation and test hold about 80 %, 10 % and 10 % of the unit that is split, and a unit is never divided. Each family has several seeds; the name of a split is the family, `_s` and the seed.

- **Random**: genes at random. For pairs, pairs at random, with the two orders of a pair and repeated measurements of it kept together.
- **Sequence identity 30 % and 50 %**: whole protein clusters. A cluster is a connected component of an all-against-all MMseqs2 search at that identity over 80 % of both sequences, so a test protein has no reported hit to a training protein at or above the threshold.
- **Chromosome position**: whole chromosome blocks (40 contiguous blocks of equal gene count), so the genes of an operon stay on one side.
- **Gene holdout** (pair task): the genes are split first, at random, by cluster or by block. A pair is test when either gene is a test gene, validation when neither is test and either is validation, and train when both are training genes.

```{include} _generated/splits.md
```

The growth-rate and imaging tasks hold the same strains and share one split file.
