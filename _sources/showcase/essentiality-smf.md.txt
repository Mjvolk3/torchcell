# Gene essentiality and single-mutant fitness

Two datasets describe what happens to *S. cerevisiae* when one gene is perturbed. `GeneEssentialitySgdDataset` records the genes whose null mutant the Saccharomyces Genome Database (SGD) annotates as inviable. `SmfCostanzo2016Dataset` records single-mutant fitness (SMF), the colony size of a mutant strain relative to wild type, for the array strains of the Costanzo et al. 2016 synthetic genetic array (SGA) screens. The two meet in one supported query, where each essential gene becomes a fitness of 0 beside the fitness Costanzo measured for the same gene.

Terms used below:

- **DAmP**: decreased abundance by mRNA perturbation, a hypomorphic allele of an essential gene made by disrupting its 3' untranslated region.
- **entry**: one experiment and its reference, as the query returns it.
- **fitness**: colony size of a mutant relative to wild type; 1.0 is wild-type growth.
- **KanMX, NatMX deletion**: a gene replaced by a kanamycin or nourseothricin resistance marker; the two deletion collections of the SGA array.
- **processed record**: every entry of one gene set, grouped by `GenotypeAggregator`.
- **source key**: the screen an entry came from (`costanzo2016@26`, `costanzo2016@30`, `converted_zero`), as `torchcell.data.label_policy.source_key` names it.
- **TS allele**: a temperature-sensitive allele of an essential gene.

## What the experiments measured

```{figure} _generated/essentiality-smf/showcase-essentiality-smf.svg
:name: fig-essentiality-smf-diagram
:width: 100%

**a**, the two sources: SGD null-mutant "inviable" annotations, and the Costanzo et al. 2016 SGA colony-size screens at 26 °C and 30 °C over four strain types. **b**, the stored record, an experiment (genotype, environment, phenotype), its reference (genome and reference phenotype) and the publication. **c**, the supported query and the conversion that turns an essential gene into a fitness-0 entry, the aggregation that keeps every entry of a gene in one record, and the read-time label policy. Diagram: `notes/assets/drawio/showcase-essentiality-smf.drawio`, composed from the loaders and the converter, not redrawn from a published figure.
```

## The records

One record of each dataset as its loader stored it, re-validated through the pydantic class and printed from `model_dump()`. The dumps are cut to the named fields; each cut is marked with the number of fields it hides.

### Gene essentiality (SGD)

```{include} _generated/essentiality-smf/record_essentiality.md
```

### Single-mutant fitness (Costanzo 2016)

```{include} _generated/essentiality-smf/record_smf.md
```

## The data

Counts and distributions over the two dataset stores. Where a table names one temperature it is 30 °C; the Caveats section says why.

```{include} _generated/essentiality-smf/summary_tables.md
```

```{include} _generated/essentiality-smf/figures.md
```

## Querying the graph

The supported query ships with the package as `torchcell/knowledge_graphs/queries/essentiality_smf.cql`. It selects both datasets from the served graph, keeps experiments whose perturbed genes are all in `$gene_set` and whose medium is solid, and returns each experiment's serialized record with its reference:

```{literalinclude} ../../../torchcell/knowledge_graphs/queries/essentiality_smf.cql
:language: text
```

The query result is built into a dataset with no deduplication, so every entry survives:

```python
from pathlib import Path

import torchcell
from torchcell.data.genotype_aggregate import GenotypeAggregator
from torchcell.data.graph_processor import SubgraphRepresentation
from torchcell.data.neo4j_cell import Neo4jCellDataset
from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter

query = (
    Path(torchcell.__file__).parent / "knowledge_graphs" / "queries" / "essentiality_smf.cql"
).read_text()
dataset = Neo4jCellDataset(
    root="data/torchcell/showcase_essentiality_smf",
    query=query,
    gene_set=genome.gene_set,  # SCerevisiaeGenome(...).gene_set
    graphs=None,
    incidence_graphs=None,
    node_embeddings=None,
    converter=CompositeFitnessConverter,
    deduplicator=None,
    aggregator=GenotypeAggregator,
    graph_processor=SubgraphRepresentation(),
)
```

`CompositeFitnessConverter` passes Costanzo fitness entries through and hands each essentiality entry to `GeneEssentialityToFitnessConverter`, whose experiment and reference functions are:

```{literalinclude} ../../../torchcell/datamodels/gene_essentiality_to_fitness_conversion.py
:language: python
:pyobject: gene_essentiality_to_fitness_experiment
```

```{literalinclude} ../../../torchcell/datamodels/gene_essentiality_to_fitness_conversion.py
:language: python
:pyobject: gene_essentiality_to_fitness_reference
```

A converted 0 is a statement about the gene, not a measurement of a strain. `torchcell.data.label_policy` gives it its own source key, and the default `LabelPolicy` admits it only when the record holds no measured fitness:

```{literalinclude} ../../../torchcell/data/label_policy.py
:language: python
:start-at: "# A converted 0 is a statement about the gene"
:end-at: "_CONVERTED_ZERO_DATASETS ="
```

What the query returned from the served release, and what the conversion, the aggregation, `label_df` and the default policy make of it:

```{include} _generated/essentiality-smf/query_results.md
```

`Neo4jCellDataset.label_df` is a convenience table, not a label policy: it keeps the last non-missing entry of each record, so its value for a record that holds a converted 0 and a measurement depends on entry order. Read labels for training through a `LabelPolicy` (`torchcell.data.label_table.build_label_table`).

## Caveats

**Temperature shown, and issue #410.** For deletion and DAmP strains Costanzo 2016 released one temperature-combined fitness. The supplementary text says: "Because we observed a close correlation between fitness measured at [26 °C] and [30 °C] for deletion mutants , we combined measurements from different temperatures in the average for each deletion mutant. Fitness associated with TS mutants was computed separately at either [26 °C] or [30 °C] ." (mirror OCR `costanzoGlobalGeneticInteraction2016/si/si1.md`, line 96, sha256 `1828703b0ff739fd...`; bracketed temperatures replace OCR-garbled tokens, the rest is verbatim). `SmfCostanzo2016Dataset` nevertheless emits each deletion and DAmP strain twice, once at 26 °C and once at 30 °C, with the same value (issue [#410](https://github.com/Mjvolk3/torchcell/issues/410); the twin counts are in the summary above). The two temperature panels of the histogram therefore differ only through the TS alleles. The per-strain-type tables use the 30 °C records: for deletion and DAmP strains that record carries the one combined value exactly once, for TS alleles it is the 30 °C measurement, and 30 °C is the first Costanzo source in the default `LabelPolicy`. The 26 °C label on a deletion or DAmP record is not a 26 °C measurement.

**SGD "inviable" includes conditionally inviable genes.** The essentiality store holds every SGD null-mutant "inviable" annotation in strain S288C. Some annotated genes have viable deletion-collection strains (the `cooper2010.py` docstring lists 21 such hits, among them ATG1, ERG24 and VPS30), and those genes still become a fitness-0 entry through the converter. The overlap table above shows the same tension: SGD-essential genes that carry a KanMX or NatMX deletion strain with a measured fitness well above 0.

**The essentiality environment is assumed, not recorded.** SGD phenotype annotations carry no medium or temperature. The loader writes the same environment on every record, marked in the source:

```{literalinclude} ../../../torchcell/datasets/scerevisiae/sgd.py
:language: python
:start-at: "# HACK for this dataset all meta data is guessed"
:end-at: "temperature=Temperature(value=30)"
```

The KanMX deletion perturbation type and the strain id `S288C` on those records are fixed by the same function, not read from the annotation.

**One graph node per distinct experiment.** The graph identifies an experiment by the sha256 of its serialized record, which excludes the publication. SGD records of one gene that differ only by publication therefore collapse to one node, which is why the query returns one essentiality entry per gene while the store holds one per annotation. SMF records of genes outside `$gene_set` are not returned.

## Provenance

```{include} _generated/essentiality-smf/provenance.md
```

Generating scripts:

- `experiments/034-showcase-datasets/scripts/essentiality_smf.py`: record dumps, summary tables, figures and the provenance table, from the two dev-tree stores.
- `experiments/034-showcase-datasets/scripts/query_essentiality_smf.py`, run by `experiments/034-showcase-datasets/scripts/gh_query_essentiality_smf.slurm`: the query results, from the served graph; the slurm job id is in the query fragment above and in `experiments/034-showcase-datasets/results/essentiality_smf_query.json`.
- `notes/assets/drawio/showcase-essentiality-smf.drawio`: the diagram, exported headless with draw.io 31.4.5.
