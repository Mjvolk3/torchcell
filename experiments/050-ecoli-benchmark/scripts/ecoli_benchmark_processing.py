# experiments/050-ecoli-benchmark/scripts/ecoli_benchmark_processing.py
# [[experiments.050-ecoli-benchmark.scripts.ecoli_benchmark_processing]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/050-ecoli-benchmark/scripts/ecoli_benchmark_processing
#
# Source of experiments/050-ecoli-benchmark/notebooks/ecoli_benchmark_processing.ipynb:
# a line that starts with "# %%" opens a cell (notebook_page.py builds and runs it).
# %% [markdown]
# # Processing the *E. coli* benchmark tasks
#
# How four built *Escherichia coli* K-12 datasets become benchmark tasks: the dataset,
# the genome its records are written against, the records matched to genes of that
# genome, the labels, and the pinned train, validation and test splits. Every step is a
# function of `torchcell.tasks`, and Step 5 shows the one import a benchmark script
# needs.
#
# Run it from a checkout of the repository, because the cells that rebuild the
# sequence-similarity splits read the committed protein clusters.

# %%
import os
import warnings

import numpy as np
import pandas as pd

from torchcell.endpoint_http import DEFAULT_URL
from torchcell.sequence.genome.standard import genome_for
from torchcell.tasks import TestSplitLocked, TestSplitWarning, load_task
from torchcell.tasks.ecoli import (
    ECOLI_TASKS,
    build_task,
    match_records,
    open_dataset,
    read_labels,
)
from torchcell.tasks.splits import (
    assign_groups,
    block_groups,
    pair_parts,
    part_sizes,
    random_split,
)

os.environ.setdefault("TC_DATA_URL", DEFAULT_URL)  # download the built dataset stores
os.environ.setdefault("DATA_ROOT", ".")  # stores and genome files go under here
pd.set_option("display.width", 200)

# %% [markdown]
# ## The four tasks
#
# A task pins one dataset, one genome assembly and one label.

# %%
for spec in ECOLI_TASKS.values():
    print(spec.name)
    print(f"  dataset  {spec.dataset_class}  ({spec.kind})")
    print(f"  genome   {spec.assembly_set}")
    print(f"  label    {spec.label}: {spec.n_targets_note}")

# %% [markdown]
# ## Step 1: the dataset and its genome
#
# `open_dataset` runs the dataset's loader. `genome_for` reads the assembly the records
# name and returns that genome, fetching its files when they are absent.

# %%
spec = ECOLI_TASKS["ecoli-metabolome-fuhrer2017"]
dataset = open_dataset(spec)
genome = genome_for(dataset)
print(f"{len(dataset):,} records")
print(f"genome {type(genome).__name__} ({genome.ASSEMBLY_SET})")
print(f"{len(genome.gene_set):,} genes")

# %% [markdown]
# ## Step 2: records matched to genes of the genome
#
# A dataset stores each perturbed gene under the identifier its source released (here
# Keio JW numbers). `match_records` resolves every one against the genome and decides
# whether the record is a benchmark sample: it must perturb the number of genes the
# task is about, and each gene must resolve to one gene of the genome that has a
# protein sequence. A record that is not a sample keeps its row and its reason.
#
# The `status` column says how an identifier resolved: `current` (it is a gene of the
# genome), `renamed` (an alias of exactly one gene), `non_gene_feature` (a locus the
# annotation does not class as a gene) or `retired` (not in the annotation). A record
# that perturbs several genes joins their values with `|`.

# %%
matched = match_records(dataset, genome, n_genes=1)
print(matched.head(5).to_string(index=False))
print()
print(matched["status"].value_counts().to_string())
print()
print(matched.loc[~matched["eligible"], "exclusion_reason"].value_counts().to_string())

# %% [markdown]
# ## Step 3: labels
#
# `read_labels` returns one row per sample and one column per target. A label that maps
# names to values (ions here, morphology features for the imaging task) becomes one
# column per name, with `nan` where a sample does not hold that name.

# %%
samples = matched[matched["eligible"]]
y, targets = read_labels(dataset, spec, samples["record_index"].to_numpy())
print("labels", y.shape, "first targets", targets[:3])

# %% [markdown]
# `build_task` is steps 1 to 3 in one call. The table counts the result for all four.

# %%
built = {name: build_task(name) for name in ECOLI_TASKS}
print(
    pd.DataFrame(
        {
            "records": {n: len(t) + len(t.excluded) for n, t in built.items()},
            "samples": {n: len(t) for n, t in built.items()},
            "excluded": {n: len(t.excluded) for n, t in built.items()},
            "genes": {n: len(set(t.genes.ravel())) for n, t in built.items()},
            "targets": {n: t.y.shape[1] for n, t in built.items()},
        }
    ).to_string()
)

# %% [markdown]
# ## Step 4: splits
#
# A split assigns whole units to train, validation and test in about 80%, 10% and
# 10%. The pinned split files were written by `build_splits.py` with the functions
# below; the cells rebuild one split of each kind and compare it with the pinned one.
# For a single-gene task the unit is the gene. `pinned` reads the part of every sample
# under a pinned split, which means unlocking its test part.


# %%
def pinned(task, name):
    """The part (train, val or test) of every sample under one pinned split."""
    split = task.split(name)
    test = split.unlock_test(reason="comparing a rebuilt split with the pinned one")
    parts = np.full(len(task), "train", dtype="<U5")
    parts[split.val] = "val"
    parts[test] = "test"
    return parts


task = load_task("ecoli-growth-rate-campos2018")
genes = [str(g) for g in task.genes[:, 0]]
print(task.split_names)

rebuilt = random_split(genes, seed=0)
same = (rebuilt == pinned(task, "random_s0")).all()
print("random_s0  ", part_sizes(rebuilt), "equals the pinned split:", same)

# %% [markdown]
# A sequence-similarity split keeps a protein cluster whole. The clusters are connected
# components of an all-against-all MMseqs2 search at 30% (or 50%) identity over 80%
# of both sequences (`cluster_proteins.py`), so a test protein has no reported hit to a
# training protein at or above the threshold.

# %%
CLUSTERS = "experiments/050-ecoli-benchmark/results/clusters"
path = f"{CLUSTERS}/{task.spec.assembly_set}.seqid30.tsv"
clusters = pd.read_csv(path, sep="\t", dtype=str)
cluster_of = dict(zip(clusters["locus_tag"], clusters["cluster"]))
rebuilt = assign_groups([cluster_of[g] for g in genes], seed=0)
same = (rebuilt == pinned(task, "seqid30_s0")).all()
print("seqid30_s0 ", part_sizes(rebuilt), "equals the pinned split:", same)
parts_of_cluster = pd.Series(rebuilt).groupby([cluster_of[g] for g in genes]).nunique()
print("clusters divided between parts:", int((parts_of_cluster > 1).sum()))

# %% [markdown]
# A position split keeps a chromosome block whole (40 contiguous blocks of equal gene
# count), so the genes of an operon stay on one side.

# %%
starts = [int(task.genome[g].start) for g in sorted(set(genes))]
block_of = dict(zip(sorted(set(genes)), block_groups(starts, n_blocks=40)))
rebuilt = assign_groups([block_of[g] for g in genes], seed=0)
same = (rebuilt == pinned(task, "position_s0")).all()
print("position_s0", part_sizes(rebuilt), "equals the pinned split:", same)

# %% [markdown]
# For the gene-pair task a random split keeps the two orders of a pair together. A
# gene-holdout split divides the genes first: a pair is test when either gene is a
# test gene, validation when neither is test and either is validation, and train when
# both are training genes. Holding genes out by cluster or by block works the same way.

# %%
pairs = load_task("ecoli-gene-interaction-babu2014")
a = [str(g) for g in pairs.genes[:, 0]]
b = [str(g) for g in pairs.genes[:, 1]]
print(pairs.split_names)

pair_genes = sorted(set(a) | set(b))
gene_part = dict(zip(pair_genes, assign_groups(pair_genes, seed=0)))
rebuilt = pair_parts([gene_part[g] for g in a], [gene_part[g] for g in b])
same = (rebuilt == pinned(pairs, "gene_holdout_s0")).all()
print("gene_holdout_s0", part_sizes(rebuilt), "equals the pinned split:", same)

# %% [markdown]
# ## Step 5: the one import
#
# Everything above is what `load_task` does. A benchmark script needs this import and
# nothing else from the processing, which is also what a submission is checked for.

# %%
task = load_task("ecoli-growth-rate-campos2018")
split = task.split("seqid30_s0")
print(f"{len(task):,} samples, {task.y.shape[1]} target, sizes {split.sizes()}")
print(
    "first training genes",
    task.genes[split.train[:3], 0],
    "labels",
    task.y[split.train[:3], 0],
)

# %% [markdown]
# The test part is locked. Reading it raises, and the message says what to do instead.

# %%
try:
    split.test
except TestSplitLocked as error:
    print(error)

# %% [markdown]
# Model selection uses validation. As an example, this scores the constant prediction
# of the training mean on validation with the task's metrics.

# %%
prediction = np.full((len(split.val), 1), task.y[split.train].mean())
print({k: round(v, 4) for k, v in task.score(task.y[split.val], prediction).items()})

# %% [markdown]
# `unlock_test` returns the test indices and warns every time. It is for one final
# evaluation of a model chosen on validation, and the reason is part of the call.

# %%
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", TestSplitWarning)
    test = split.unlock_test(reason="showing the warning in the processing notebook")
print(len(test), "test samples")
print(caught[0].category.__name__ + ":", caught[0].message)
