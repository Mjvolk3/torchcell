# examples/benchmark/load_ecoli_tasks.py
"""Load the four *E. coli* benchmark tasks with their splits and genomes. No model.

For each task this prints what a model needs: the samples, the deleted genes, the
labels, the pinned train, validation and test splits, and the genome with the DNA,
coding sequence and protein of every gene.

Needs only ``pip install "torchcell>=1.8.8"``. Run it in an empty directory::

    python load_ecoli_tasks.py

The first run downloads the four dataset stores and two genomes into the current
directory (the metabolome store is the large one, about 80 MB).
"""

from __future__ import annotations

import os

from torchcell.endpoint_http import DEFAULT_URL
from torchcell.tasks import ECOLI_TASKS, load_task

os.environ.setdefault("TC_DATA_URL", DEFAULT_URL)  # download the built dataset stores
os.environ.setdefault("DATA_ROOT", ".")  # the stores and genome files go under here

for name in ECOLI_TASKS:  # the four task names
    task = load_task(name)
    print(f"\n{name}: {task.spec.title}")
    print(f"  samples  {len(task):,} ({len(task.excluded):,} records set aside)")
    print(f"  genes    task.genes {task.genes.shape}, first {task.genes[0].tolist()}")
    print(f"  labels   task.y {task.y.shape}, first targets {task.targets[:3]}")

    # Splits are named <family>_s<seed>. Each one gives row indices into task.genes
    # and task.y for train and validation; the test part is locked.
    families = sorted(
        {split_name.rsplit("_s", 1)[0] for split_name in task.split_names}
    )
    print(f"  splits   {len(task.split_names)} in families {families}")
    split = task.split(task.split_names[0])
    print(f"  {split.name}: sizes {split.sizes()}")
    x_genes, y = task.genes[split.train], task.y[split.train]  # training inputs, labels
    print(f"  train    genes {x_genes.shape}, labels {y.shape}")
    print(f"  val      first rows {split.val[:5].tolist()}")
    # split.test raises. For one final evaluation, and nothing else, call:
    # test = split.unlock_test(reason="final evaluation of the model chosen on val")

    # The genome the records are written against: one entry per locus tag.
    gene = task.genome[task.genes[0, 0]]
    print(f"  genome   {task.spec.assembly_set}, {len(task.genome.gene_set):,} genes")
    dna, protein = len(gene.seq), len(gene.protein.seq)
    print(f"  {gene.id} {gene.symbol}: DNA {dna:,} bp, protein {protein:,} aa")
    print(f"    gene.seq[:30]          {str(gene.seq)[:30]}")
    print(f"    gene.cds.seq[:30]      {str(gene.cds.seq)[:30]}")
    print(f"    gene.protein.seq[:30]  {str(gene.protein.seq)[:30]}")
