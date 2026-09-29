# User Guide

torchcell is a provenance-first experimental-data ontology and dataset substrate for the
yeast (*Saccharomyces cerevisiae*) virtual cell. Every record is a typed experiment,
genotype x environment -> phenotype, paired with the reference it is measured against and
traceable to the source file it was parsed from. Records are built into LMDB-backed
dataset loaders, served together as a Neo4j knowledge graph, and queried back into
PyTorch Geometric datasets for model training.

```{toctree}
:maxdepth: 2

installation
quickstart
data-model
datasets
knowledge-graph
contributing
```

Released versions of the served knowledge graph, and which torchcell versions they
correspond to, are listed under [Database releases](../database/index.md).
