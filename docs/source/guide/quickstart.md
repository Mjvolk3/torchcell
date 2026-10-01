# Quickstart

Three entry points cover most work: the reference genome, a single-source experiment
dataset built from a publication, and a multi-source dataset queried out of the served
knowledge graph. Every block below states what it needs. Blocks marked
**requires `DATA_ROOT`** read files from the data root described in
[Installation](installation.md); blocks marked **requires the served database** open a
Neo4j connection.

## Load the reference genome

**Requires `DATA_ROOT`.** `SCerevisiaeGenome`
(`torchcell.sequence.genome.scerevisiae.s288c`) resolves the SGD R64-4-1 S288C release
from the genomes tier at `$DATA_ROOT/torchcell-genomes/`, verifying each file's sha256
against the tier's `manifest.json` on every load. There is no fallback location: a
machine without the tier raises `FileNotFoundError` with the command that seeds it.
`genome_root` is only a cache directory for the gffutils database (`data.db`) and must
already exist. With `overwrite=False` (the default) the database is built only when it
is absent and is otherwise opened after its record is verified; `overwrite=True`
rebuilds it. `go_root` holds
`go.obo`, which is downloaded from geneontology.org only when it is missing.

```python
import os
import os.path as osp

from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

genome = SCerevisiaeGenome(
    genome_root=osp.join(os.environ["DATA_ROOT"], "data/sgd/genome"),
    go_root=osp.join(os.environ["DATA_ROOT"], "data/go"),
    overwrite=False,
)
print(len(genome.gene_set))
gene = genome["YAL001C"]
print(gene.id, gene.chromosome, gene.strand, gene.start, gene.end, len(gene.seq))
```

Output (run with `genome_root` pointed at a scratch cache directory, which does not
change the result):

```text
6607
YAL001C 1 - 147594 151166 3573
```

## Load an experiment dataset

**Requires `DATA_ROOT`; the first build also needs the network.** Each publication-backed
loader is an `ExperimentDataset` subclass (see [Datasets](datasets.md)) whose default
`root` is relative, for example `data/torchcell/smf_costanzo2016`. Pass an absolute root
under `$DATA_ROOT`. If `processed/lmdb` is absent, the loader downloads its raw files
(for `SmfCostanzo2016Dataset`, from thecellmap.org) and builds the LMDB; if it is
present, the loader reads it and never touches the network.

```python
import os
import os.path as osp

from torchcell.datasets.scerevisiae.costanzo2016 import SmfCostanzo2016Dataset

dataset = SmfCostanzo2016Dataset(
    root=osp.join(os.environ["DATA_ROOT"], "data/torchcell/smf_costanzo2016")
)
print(dataset)
item = dataset[0]
print(sorted(item))
typed = dataset.transform_item(item)
print(type(typed["experiment"]).__name__, typed["experiment"].phenotype.fitness)
print(len(dataset.gene_set), len(dataset.experiment_reference_index))
```

Output (run against a read-only copy of a built `processed/` and `preprocess/`):

```text
SmfCostanzo2016Dataset(20484)
['experiment', 'publication', 'reference']
FitnessExperiment 0.9395
5493 2
```

`dataset[i]` returns plain dictionaries; `transform_item` validates them into the typed
`Experiment`, `ExperimentReference` and `Publication` models. `experiment_reference_index`
groups records by the reference they share; this dataset has two (its 26 C and 30 C
screens carry different references).

## Query the served knowledge graph

**Requires the served database** (connection from `NEO4J_URI`, `NEO4J_USER`,
`NEO4J_PASSWORD`) and `DATA_ROOT` for the output directory. None of the code in this
section was run for this page.

### `Neo4jQueryRaw`

`torchcell.data.neo4j_query_raw.Neo4jQueryRaw` runs one Cypher query and caches every
returned record in an LMDB at `<root_dir>/raw/lmdb`. It is an `attrs` class; its
constructor arguments are:

```python
Neo4jQueryRaw(
    uri: str,
    username: str,
    password: str,
    root_dir: str,
    query: str,
    io_workers: int | None = None,
    num_workers: int | None = None,
    cypher_kwargs: dict[str, str | int | float | list[Any]] = {},  # attrs factory=dict
    version: str | None = None,
)
```

The query runs once, on construction, when `<root_dir>/raw/lmdb/data.mdb` does not yet
exist; later constructions open the cached LMDB read-only. `cypher_kwargs` are passed to
`session.run` as query parameters, so a query that filters on `$gene_set` gets
`cypher_kwargs={"gene_set": [...]}`. `version` names the knowledge-graph release to open
and defaults to `TORCHCELL_KG_VERSION` (see {ref}`kg-choosing-a-release`).

Each result row must have one of two shapes:

| Shape | `RETURN` clause | When to use |
| :-- | :-- | :-- |
| Property | `RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized` | Required for large queries: the driver does not cache hydrated nodes, so memory stays flat. |
| Node | `RETURN e, ref` | Supported for the historical queries under `experiments/`. |

`e` is the `Experiment` node and `ref` its `ExperimentReference`. Each stored record is
revalidated through `EXPERIMENT_TYPE_MAP` and `EXPERIMENT_REFERENCE_TYPE_MAP` from
`torchcell.datamodels.schema`, keyed by the record's `experiment_type`.

### `Neo4jCellDataset`

`torchcell.data.neo4j_cell.Neo4jCellDataset` wraps the raw query in a processing
pipeline and returns graph-ready items. Its constructor arguments:

```python
Neo4jCellDataset(
    root: str,
    query: str | None = None,
    gene_set: GeneSet | None = None,
    graphs: GeneMultiGraph | None = None,
    incidence_graphs: dict[str, nx.Graph | hnx.Hypergraph] | None = None,
    node_embeddings: dict[str, BaseEmbeddingDataset] | None = None,
    graph_processor: GraphProcessor | None = None,
    add_remaining_gene_self_loops: bool = True,
    converter: type[Converter] | None = None,
    deduplicator: type[Deduplicator] | None = None,
    aggregator: type[Aggregator] | None = None,
    overwrite_intermediates: bool = False,
    uri: str | None = None,
    username: str | None = None,
    password: str | None = None,
    transform: Callable[..., Any] | None = None,
    pre_transform: Callable[..., Any] | None = None,
    pre_filter: Callable[..., Any] | None = None,
    phenotype_labels: list[str] | None = None,
)
```

`converter`, `deduplicator` and `aggregator` are passed as classes; `process()`
instantiates them. `uri`, `username` and `password` default to the environment. The
`gene_set` is sent to the query as `$gene_set`. On first construction `process()` runs
the stages in order, each writing its own LMDB under `root`:

| Stage | LMDB | Runs when |
| :-- | :-- | :-- |
| raw | `raw/lmdb` | always (the `Neo4jQueryRaw` cache) |
| conversion | `conversion/lmdb` | `converter` is given |
| deduplication | `deduplication/lmdb` | `deduplicator` is given |
| aggregation | `aggregation/lmdb` | `aggregator` is given |
| processed | `processed/lmdb` | always |

A finished stage writes a `STAGE_COMPLETE` file beside its LMDB, and a restarted build
skips stages that carry one. `phenotype_labels` selects and orders the phenotypes each
item carries by `label_name` (for example `["fitness", "gene_interaction"]`); leave it
`None` only when the build holds a single phenotype.

The block below is the construction used by
`experiments/025-solid-growth/scripts/query.py`:

```python
# Requires the served database and DATA_ROOT.
import os
import os.path as osp

from dotenv import load_dotenv

from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator
from torchcell.data.graph_processor import SubgraphRepresentation
from torchcell.data.neo4j_cell import Neo4jCellDataset
from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

load_dotenv()
data_root = os.environ["DATA_ROOT"]

with open("experiments/025-solid-growth/queries/001_all_solid_growth.cql") as f:
    query = f.read()

genome = SCerevisiaeGenome(
    genome_root=osp.join(data_root, "data/sgd/genome"),
    go_root=osp.join(data_root, "data/go"),
)

dataset = Neo4jCellDataset(
    root=osp.join(data_root, "data/torchcell/experiments/025-solid-growth/001-full-build"),
    query=query,
    gene_set=genome.gene_set,
    converter=CompositeFitnessConverter,
    deduplicator=MeanExperimentDeduplicator,
    aggregator=GenotypeAggregator,
    graph_processor=SubgraphRepresentation(),
)
print(len(dataset))
print({k: len(v) for k, v in dataset.phenotype_label_index.items()})
```

`CompositeFitnessConverter` turns gene-essentiality and synthetic-lethality records into
fitness records (see {ref}`kg-converters`), which is what lets that query pool them with
measured fitness. `dataset.phenotype_label_index`, `dataset.dataset_name_index` and
`dataset.perturbation_count_index` map a label, a source dataset, or a perturbation count
to the item indices that carry it.

## Inspect a stored record

A built experiment LMDB stores one pickled dictionary per record under the keys
`"0"`, `"1"`, and so on. Constant sub-objects that repeat across a dataset (the
environment, the reference, the publication) are stored once in a sibling `interned`
LMDB when their canonical JSON is at least `INTERN_MIN_BYTES` (512) long, and the record holds a `{"$ref": <sha256>, "name": <hint>}` pointer in their place.
`torchcell.data.experiment_dataset.resolve_interned` splices them back; a reader that
bypasses the dataset class must call it.

The script below opens the development build of `SmfCostanzo2016Dataset` read-only and
prints record `0`:

```python
import os.path as osp
import pickle
from pprint import pprint

import lmdb

from torchcell.data.experiment_dataset import resolve_interned
from torchcell.datamodels.schema import FitnessExperiment

root = "/scratch/projects/torchcell-scratch/data/torchcell/smf_costanzo2016/processed"

# Records and interned constants live in two sibling LMDB environments.
env = lmdb.open(osp.join(root, "lmdb"), readonly=True, lock=False)
ienv = lmdb.open(osp.join(root, "interned"), readonly=True, lock=False)
with ienv.begin() as txn:
    interned = {k.decode(): pickle.loads(v) for k, v in txn.cursor()}
with env.begin() as txn:
    print("records:", txn.stat()["entries"])
    raw = pickle.loads(txn.get(b"0"))

print("keys:", sorted(raw))
print("stored environment:", raw["experiment"]["environment"])
record = resolve_interned(raw, interned)
experiment = FitnessExperiment(**record["experiment"])

print("experiment_type:", experiment.experiment_type)
print("dataset_name:", experiment.dataset_name)
pprint(experiment.genotype.model_dump(), width=88)
env_dump = experiment.environment.model_dump()
print("media.name:", env_dump["media"]["name"], "| state:", env_dump["media"]["state"])
print("temperature:", env_dump["temperature"])
pprint(experiment.phenotype.model_dump(), width=88)
print("publication:", record["publication"])
```

Output, verbatim (build of 2026-09-14):

```text
records: 20484
keys: ['experiment', 'publication', 'reference']
stored environment: {'$ref': '424e6d43e0fcd66aa5490fdd3c0a9f92485038ab7157c1a4456769e91461d9aa', 'name': 'SGA double-mutant selection (SD-MSG, -His/Arg/Lys, +canavanine/thialysine/G418/clonNAT)'}
experiment_type: fitness
dataset_name: SmfCostanzo2016Dataset
{'perturbations': [{'damp_description': 'Damp Perturbation information specific to SGA '
                                        'experiments.',
                    'damp_perturbation_type': 'SGA',
                    'description': '4-10 decreased expression via KANmx insertion at '
                                   'the ',
                    'expression_range': {'max': 0.25, 'min': 0.1},
                    'mechanism_so_id': 'SO:0001060',
                    'mechanism_so_name': 'sequence_variant',
                    'perturbation_type': 'damp',
                    'perturbed_gene_name': 'tfc3_damp',
                    'provenance': 'engineered',
                    'strain_id': 'YAL001C_damp174',
                    'systematic_gene_name': 'YAL001C'}]}
media.name: SGA double-mutant selection (SD-MSG, -His/Arg/Lys, +canavanine/thialysine/G418/clonNAT) | state: solid
temperature: {'value': 26.0, 'unit': <TemperatureUnit.celsius: 'Celsius'>}
{'fitness': 0.9395,
 'fitness_se': 0.0988,
 'fitness_std': 0.0988,
 'fitness_uncertainty': 0.0988,
 'fitness_uncertainty_type': <UncertaintyType.bootstrap_se: 'bootstrap_se'>,
 'graph_level': 'global',
 'label_name': 'fitness',
 'label_statistic_name': 'fitness_se',
 'n_samples': 17,
 'provenance_gaps': [],
 'sample_unit': <SampleUnit.screen: 'screen'>}
publication: {'pubmed_id': '27708008', 'pubmed_url': 'https://pubmed.ncbi.nlm.nih.gov/27708008/', 'doi': '10.1126/science.aaf1420', 'doi_url': 'https://www.science.org/doi/10.1126/science.aaf1420'}
```

The record reads as one experiment: a genotype of one perturbation (a DAmP allele of
*TFC3*, systematic name `YAL001C`), an environment (the SGA selection medium, solid, at
26 C), and a fitness phenotype of 0.9395 with its uncertainty named as a bootstrap
standard error over 17 screens. [Data model](data-model.md) describes each part.
