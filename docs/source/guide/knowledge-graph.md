# Knowledge graph

The served knowledge graph is a Neo4j database built with BioCypher from the dataset
LMDBs (see [Datasets](datasets.md)). Every experiment record becomes an `Experiment`
node joined to nodes for its dataset, genotype, perturbations, environment, phenotype
and reference, so a single Cypher query can select records across datasets by any of
those parts.

A Neo4j Browser over the served graph is at
<https://torchcell-database.ncsa.illinois.edu:7473/browser/>.

## Schema

The graph schema is `biocypher/config/torchcell_schema_config.yaml`. BioCypher gives
each node the label of its class and of the classes it descends from, which is why every
phenotype node, whatever its class, also carries the label `PhenotypicFeature`.

**Nodes used by the queries under `experiments/*/queries/`:**

| Label | What it is | Selected properties |
| :-- | :-- | :-- |
| `Dataset` | one loader class | `id`, the class name (`'SmfCostanzo2016Dataset'`) |
| `Experiment` | one experiment record | `id` (sha256 of the serialized record), `serialized_data` |
| `ExperimentReference` | the reference the experiment is measured against | `serialized_data` |
| `Genotype` | the set of perturbations in a strain | `systematic_gene_names`, `perturbation_types` |
| `Perturbation` | one gene perturbation | `systematic_gene_name`, `perturbation_type` |
| `PhenotypicFeature` | any phenotype node (`FitnessPhenotype`, `GeneInteractionPhenotype`, ...) | `graph_level`, `label_name` |
| `Environment` | medium, temperature and environment perturbations | `serialized_data` |
| `Media` | the growth medium | `name`, `state` (`solid`, `liquid` or `gas`) |
| `Temperature` | the growth temperature | `value`, `unit` |

**Relationships**, in the direction they are stored:

| Relationship | From | To |
| :-- | :-- | :-- |
| `ExperimentMemberOf` | `Experiment` | `Dataset` |
| `ExperimentReferenceOf` | `ExperimentReference` | `Experiment` |
| `GenotypeMemberOf` | `Genotype` | `Experiment` |
| `PerturbationMemberOf` | `Perturbation` | `Genotype` |
| `PhenotypeMemberOf` | phenotype (`PhenotypicFeature`) | `Experiment`, `ExperimentReference` |
| `EnvironmentMemberOf` | `Environment` | `Experiment`, `ExperimentReference` |
| `MediaMemberOf` | `Media` | `Environment` |
| `TemperatureMemberOf` | `Temperature` | `Environment` |

The schema file also declares `Publication`, `Genome`, `EnvironmentPerturbation`,
`CrisprConstruct` and `SegregantGenotype` nodes and the relationships that attach them.
Each served database also holds one `KgRelease` node describing the release (see
[Database releases](../database/index.md)).

## Writing a query

A query returns `Experiment` and `ExperimentReference` pairs; `Neo4jQueryRaw` (see
[Quickstart](quickstart.md)) rebuilds the typed records from them. The excerpt below is
the first block of `experiments/003-fit-int/queries/fit-int.cql`, annotated:

```text
// Start from one dataset. Dataset.id is the loader class name.
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'TmiKuzmin2018Dataset'
// Walk from the experiment to each of its parts.
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (g)<-[:PerturbationMemberOf]-(p:Perturbation)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature)
// Filter on the phenotype's graph level and the environment.
WHERE phen.graph_level = 'hyperedge'
 AND m.name = 'YEPD'
 AND t.value = 30
 // Keep a genotype only if EVERY perturbation is a deletion of a gene in $gene_set,
 // the parameter Neo4jQueryRaw passes through cypher_kwargs.
 AND ALL(pert IN [(g)<-[:PerturbationMemberOf]-(p) | p]
WHERE pert.perturbation_type = 'deletion'
 AND pert.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(p) | p]) > 0
// One row per experiment; ordering by the content-hash id gives a stable order.
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e, ref
```

Blocks for further datasets are joined with `UNION ALL`, each with the same `RETURN`
columns.

This query is a historical record of what experiment 003 selected, and two of its
filters no longer select the same records on the served graph. The SGA fitness datasets
now record their actual selection medium rather than YEPD, and the Kuzmin screens are
recorded at 26 C, so `m.name = 'YEPD'` and `t.value = 30` drop them. The comments at the
top of `experiments/025-solid-growth/queries/001_all_solid_growth.cql` record these
changes; that query filters on `m.state = 'solid'` instead and returns the property
shape:

```cypher
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized
```

Use the property shape for large queries. Returning whole nodes makes the Neo4j driver
keep every node it has read for the life of the result, measured in the
`Neo4jQueryRaw.process` comments at 13.6 KB per record on the 025 build against 3 B per
record for the property shape.

(kg-choosing-a-release)=

## Choosing a release

Clients name a **version**, not a database. `Neo4jQueryRaw` resolves the version to a
database name each time it runs a query, through
`torchcell.knowledge_graphs.releases.resolve_database`:

- `latest` (the default) and `pinned` are Neo4j database aliases on the served instance,
  retargeted when a release is published. `latest` follows the newest release; `pinned`
  is held on a release that a paper or an experiment series depends on.
- A release id (`2026.09.21-ab6d8c5d`) or a `major.minor` version (`1.2`) is looked up
  in the `KgRelease` node of every online database.
- Anything else raises `LookupError`.

The version comes from, in order, the `version` argument of `Neo4jQueryRaw` and the
`TORCHCELL_KG_VERSION` environment variable, and is `latest` when neither is set.
`Neo4jCellDataset` does not pass a version, so set `TORCHCELL_KG_VERSION` to pin a
whole run:

```bash
export TORCHCELL_KG_VERSION=pinned
```

### The releases CLI

**Requires the served database.** `python -m torchcell.knowledge_graphs.releases` reads
`NEO4J_URI`, `NEO4J_USER` and `NEO4J_PASSWORD`, or takes `--uri`, `--user` and
`--password` before the subcommand.

```bash
# One row per served database: version, release id, commit, datasets, nodes, aliases, status.
python -m torchcell.knowledge_graphs.releases status

# Every dataset a version serves: class name, record count, content sha256.
python -m torchcell.knowledge_graphs.releases datasets --version latest

# Which served datasets this checkout would serialize differently (exit status 1 on drift).
python -m torchcell.knowledge_graphs.releases --repo . compat --version latest

# Datasets unchanged, changed, added and removed between two versions.
python -m torchcell.knowledge_graphs.releases diff 1.2 latest
```

A dataset's content hash is the sha256 of its sorted experiment ids. Experiment ids are
themselves sha256 hashes of the serialized records, so two releases that report the same
hash for a dataset serve byte-identical experiment records for it.

(kg-converters)=

## Converters

Some phenotypes are statements that imply a fitness value. A **converter** rewrites
such records into another experiment type between the raw query and deduplication, so
they can be pooled with measured values. Converters subclass
`torchcell.datamodels.conversion.Converter` and declare a `ConversionMap` of input
type, output type and conversion function for both the experiment and its reference.

| Converter | Input | Output experiment | Output reference |
| :-- | :-- | :-- | :-- |
| `GeneEssentialityToFitnessConverter` | `GeneEssentialityExperiment` with `is_essential=True` | `FitnessExperiment`, fitness 0.0 | `FitnessExperimentReference`, fitness 1.0 |
| `SyntheticLethalityToFitnessConverter` | `SyntheticLethalityExperiment` with `is_synthetic_lethal=True` | `FitnessExperiment`, fitness 0.0 | `FitnessExperimentReference`, fitness 1.0 |
| `CompositeFitnessConverter` | either of the above | tries essentiality first, then synthetic lethality | as above |

The genotype and environment are carried over unchanged, and records of any other type
pass through the converter untouched. A record whose phenotype is
false (a non-essential gene, a pair that is not synthetic lethal) has no fitness
equivalent: its conversion function returns `None`, and `Converter.process` logs the
record as an error and leaves it out of the converted LMDB.

A converted zero is a statement about a gene or a pair, not a measurement of a strain.
`torchcell/data/label_policy.py` keeps it distinguishable: `source_key` maps every entry
from a `GeneEssentialitySgd` or `SynthLethality` dataset to the single source
`CONVERTED_ZERO` (`"converted_zero"`), and the default `LabelPolicy` ranks it last in
`fitness_precedence` and, with `converted_zero_only_without_measurement=True`, admits it
only for a genotype that has no measured fitness.

## Changing what the graph serves

The served store carries a build manifest (`torchcell/knowledge_graphs/kg_manifest.py`)
recording, for every served dataset, the schema-contract fingerprints, the graph schema
and the adapter code it was serialized under. A new dataset is **admitted
incrementally**, without rebuilding the others, when nothing already served would
change; a dataset that is already served can be re-admitted only as a superset of the
records the store holds for it. Any change to the schema closure of a served dataset, to
an existing node class, or to adapter code a served dataset uses requires a **full
rebuild**. The first bumps the minor version of the release, the second the major (see
[Database releases](../database/index.md)).
