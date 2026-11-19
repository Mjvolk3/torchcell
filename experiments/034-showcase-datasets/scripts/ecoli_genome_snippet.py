# experiments/034-showcase-datasets/scripts/ecoli_genome_snippet.py
# [[experiments.034-showcase-datasets.scripts.ecoli_genome_snippet]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/034-showcase-datasets/scripts/ecoli_genome_snippet
"""Load each built E. coli dataset together with the genome its records are written against.

The snippet a collaborator copies. It needs an installed ``torchcell`` and a network
route to the public endpoint, nothing else; run it in an empty directory::

    python ecoli_genome_snippet.py

It downloads the four dataset stores into ``./data/torchcell/<slug>/`` and the genome
files into ``./torchcell-genomes/<assembly_set>/``, every file checked against its
sha256. For each dataset it prints which genome was looked up, how the dataset's gene
identifiers resolve against that genome, and one record turned into the two things a
cell graph transformer consumes: the node set (the genome's genes) and the indices of
the nodes a strain perturbs, with the measured label.
"""

from __future__ import annotations

# [setup]
import os
from collections import Counter

from torchcell.datasets.ecoli.babu2014 import GeneInteractionBabu2014Dataset
from torchcell.datasets.ecoli.campos2018 import (
    GrowthRateCampos2018Dataset,
    MorphologyCampos2018Dataset,
)
from torchcell.datasets.ecoli.fuhrer2017 import MetabolomeFuhrer2017Dataset
from torchcell.endpoint_http import DEFAULT_URL
from torchcell.sequence.genome.standard import genome_for, standard_genomes

os.environ.setdefault("TC_DATA_URL", DEFAULT_URL)  # download built stores and genomes
os.environ.setdefault("DATA_ROOT", ".")  # genome files go to ./torchcell-genomes/
# [/setup]

# [standard]
for row in standard_genomes():  # every genome that loads by its assembly-set id
    print(f"{row.assembly_set:32} {row.organism} {row.strain} -> {row.class_name}")
# [/standard]

# [load]
DATASETS = {
    "ecoli_growth_rate_campos2018": GrowthRateCampos2018Dataset,
    "ecoli_morphology_campos2018": MorphologyCampos2018Dataset,
    "gene_interaction_babu2014": GeneInteractionBabu2014Dataset,
    "metabolome_fuhrer2017": MetabolomeFuhrer2017Dataset,
}

for slug, loader in DATASETS.items():
    dataset = loader(root=f"data/torchcell/{slug}")  # records: genotype -> phenotype
    genome = genome_for(dataset)  # looked up from the records' assembly_set

    # Nodes of the cell graph: one per gene of the genome, in sorted order.
    nodes = list(genome.gene_set)
    node_index = {gene: i for i, gene in enumerate(nodes)}

    # The dataset writes genes in the identifiers its source released (locus tags,
    # b-numbers or Keio JW numbers); the genome resolves each one to a locus tag.
    resolved = {name: genome.resolve_gene_name(name) for name in dataset.gene_set}
    status = Counter(r.status.value for r in resolved.values())

    # One record as model input: the perturbed node indices and the label.
    record = dataset[0]["experiment"]
    perturbed = [p["systematic_gene_name"] for p in record["genotype"]["perturbations"]]
    targets = [resolved[name].systematic_name for name in perturbed]
    gene = genome[targets[0]]
    label_name = record["phenotype"]["label_name"]

    print(f"\n{slug}")
    print(f"  records {len(dataset):,}   genome {type(genome).__name__}")
    print(f"  assembly set {genome.ASSEMBLY_SET}   nodes {len(nodes):,}")
    print(f"  dataset genes {len(dataset.gene_set):,}: {dict(sorted(status.items()))}")
    print(f"  record 0 perturbs {perturbed} -> {targets}")
    print(f"  node indices {[node_index[t] for t in targets]}   label {label_name}")
    print(
        f"  {gene.id} {gene.symbol}: {len(gene.seq):,} bp, "
        f"protein {len(gene.protein.seq):,} aa, GO terms {len(gene.go)}"
    )
# [/load]
