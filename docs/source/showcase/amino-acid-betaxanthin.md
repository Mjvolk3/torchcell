# Amino acids and betaxanthin

Three datasets describe the small-molecule content of *S. cerevisiae* deletion strains. `AminoAcidMulleder2016Dataset` records the intracellular concentration of 19 amino acids in the prototrophic deletion collection (Mulleder et al. 2016). `AminoAcidCooper2010Dataset` records free amine-containing pools of the deletion collection as a ratio to the plate mean (Cooper et al. 2010). `BetaxanthinCachera2023Dataset` records the yellow color of deletion strains that carry a transferred betaxanthin pathway (Cachera et al. 2023). All three store a `MetabolitePhenotype` and meet in one supported query.

Terms used below:

- **Btx cassette**: the four betaxanthin genes Cachera et al. integrate at chromosomal site XII-5 of every strain: *CYP76AD1* (from *Beta vulgaris*), *DOD* (from *Mirabilis jalapa*), and feedback-resistant alleles of the yeast genes *ARO4* (K229L) and *ARO7* (G141S).
- **CE-LIF**: capillary electrophoresis with laser-induced fluorescence, the Cooper 2010 platform; amine metabolites are labeled with NBD-F and read as peaks.
- **CRI-SPA**: the Cachera 2023 method that moves a cassette from a donor strain into every strain of an arrayed library by mating and selection.
- **entry**: one experiment and its reference, as the query returns it.
- **LC-SRM/MS**: liquid chromatography with selected reaction monitoring mass spectrometry, the Mulleder 2016 platform.
- **MCD robust mean**: the Minimum Covariance Determinant estimate of the mean over all strains; Mulleder et al. use it as the unperturbed level of each amino acid.
- **processed record**: every entry of one perturbed gene set, grouped by `GenotypeAggregator`.
- **reference**: the level a record's value is compared with (`reference.phenotype_reference`): the MCD robust mean (Mulleder), 1.0 (Cooper, the plate mean) or 0 (Cachera, the population-centered score).

## What the experiments measured

```{figure} _generated/amino-acid-betaxanthin/showcase-amino-acid-betaxanthin.svg
:name: fig-amino-acid-betaxanthin-diagram
:width: 100%

**a**, the three sources: the Mulleder 2016 LC-SRM/MS amino-acid metabolome, the Cooper 2010 CE-LIF amine peaks, and the Cachera 2023 CRI-SPA betaxanthin screen with its Btx cassette. **b**, the stored record, an experiment (genotype, environment, `MetabolitePhenotype`), its reference (genome and reference level) and the publication. **c**, the supported query and the aggregation, under which a Mulleder and a Cooper entry of the same deletion share one processed record while a Cachera entry, whose gene set also holds the cassette genes, is a record of its own. Diagram: `notes/assets/drawio/showcase-amino-acid-betaxanthin.drawio`, composed from the loaders and the query, not redrawn from a published figure.
```

## The records

One record of each dataset as its loader stored it, re-validated through the pydantic class and printed from `model_dump()`. The three dumps show the same deletion, the first gene held by all three stores. Each dump is cut to the named fields, and each cut is marked with the number of fields it hides.

### Amino acids (Mulleder 2016)

```{include} _generated/amino-acid-betaxanthin/record_mulleder.md
```

### Amine peaks (Cooper 2010)

```{include} _generated/amino-acid-betaxanthin/record_cooper.md
```

### Betaxanthin (Cachera 2023)

```{include} _generated/amino-acid-betaxanthin/record_cachera.md
```

## The data

Counts and distributions over the three dev-tree stores. The Caveats section says where a store differs from the served release.

```{include} _generated/amino-acid-betaxanthin/summary_tables.md
```

```{include} _generated/amino-acid-betaxanthin/figures.md
```

## Querying the graph

The supported query ships with the package as `torchcell/knowledge_graphs/queries/amino_acid_betaxanthin.cql`. It selects the three datasets from the served graph, keeps experiments whose deletions are all of genes in `$gene_set` and that carry at least one deletion, lets gene additions pass, and returns each experiment's serialized record with its reference:

```{literalinclude} ../../../torchcell/knowledge_graphs/queries/amino_acid_betaxanthin.cql
:language: text
```

Every entry is already a `MetabolitePhenotype`, so the query result is built into a dataset with no converter and no deduplicator:

```python
from pathlib import Path

import torchcell
from torchcell.data.genotype_aggregate import GenotypeAggregator
from torchcell.data.graph_processor import SubgraphRepresentation
from torchcell.data.neo4j_cell import Neo4jCellDataset

query = (
    Path(torchcell.__file__).parent
    / "knowledge_graphs"
    / "queries"
    / "amino_acid_betaxanthin.cql"
).read_text()
dataset = Neo4jCellDataset(
    root="data/torchcell/showcase_amino_acid_betaxanthin",
    query=query,
    gene_set=genome.gene_set,  # SCerevisiaeGenome(...).gene_set
    graphs=None,
    incidence_graphs=None,
    node_embeddings=None,
    converter=None,
    deduplicator=None,
    aggregator=GenotypeAggregator,
    graph_processor=SubgraphRepresentation(),
)
```

`GenotypeAggregator` keys a record on the set of every perturbed gene's systematic name, gene additions included:

```{literalinclude} ../../../torchcell/data/genotype_aggregate.py
:language: python
:pyobject: _hash_gene_set
```

What the query returned from the served release, and what the aggregation makes of it:

```{include} _generated/amino-acid-betaxanthin/query_results.md
```

A metabolite label is a dictionary, so `Neo4jCellDataset.label_df` leaves it missing; read the levels from each entry's `phenotype.metabolite_level`. `DeletionKeyedGenotypeAggregator` (same module) keys a record on its deletions alone, treating a constant cassette as part of the reference strain; with it, a Cachera entry would share a record with the Mulleder and Cooper entries of the same deletion.

## Getting the data

Each dataset's built store is packaged as an archive for the `tc-data` download endpoint ({doc}`../guide/downloads`). The public endpoint is not yet deployed (its deployment on Radiant is pending), so today every loader below builds its store from the source files instead, and the commands in this section apply once an endpoint URL and key are issued.

| dataset | loader | store slug |
|---|---|---|
| Mulleder 2016 | `torchcell.datasets.scerevisiae.mulleder2016.AminoAcidMulleder2016Dataset` | `amino_acid_mulleder2016` |
| Cooper 2010 | `torchcell.datasets.scerevisiae.cooper2010.AminoAcidCooper2010Dataset` | `amino_acid_cooper2010` |
| Cachera 2023 | `torchcell.datasets.scerevisiae.cachera2023.BetaxanthinCachera2023Dataset` | `betaxanthin_cachera2023` |

From Python, with `TC_DATA_URL` and `TC_DATA_API_KEY` set:

```python
from pathlib import Path

from torchcell.datasets.client import DatasetClient, unpack_artifact

client = DatasetClient.from_env()  # TC_DATA_URL, TC_DATA_API_KEY
for slug in ("amino_acid_mulleder2016", "amino_acid_cooper2010", "betaxanthin_cachera2023"):
    artifact = client.select(slug)  # newest supported row on this major.minor
    if artifact is None:
        raise SystemExit(f"nothing for {slug} compatible with the installed torchcell")
    archive = client.download(artifact, dest=Path("artifacts") / artifact.archive)  # verifies sha256
    unpack_artifact(archive, Path("data/torchcell") / slug)
```

With curl, one dataset at a time (the archive name comes from the index row):

```bash
curl -H "X-API-Key: $TC_DATA_API_KEY" "$TC_DATA_URL/datasets/amino_acid_mulleder2016"
curl -C - -H "X-API-Key: $TC_DATA_API_KEY" \
  -o "$ARCHIVE" "$TC_DATA_URL/datasets/amino_acid_mulleder2016/$ARCHIVE"
sha256sum "$ARCHIVE"   # must equal the row's archive_sha256
mkdir -p data/torchcell/amino_acid_mulleder2016
tar -xf "$ARCHIVE" -C data/torchcell/amino_acid_mulleder2016
```

The loaders do this themselves. With both variables set, constructing a loader whose `processed/lmdb` is absent selects, downloads, verifies and unpacks that dataset's archive instead of running `process()`, and raises, naming the slug, when the endpoint has nothing compatible with the installed version; it never falls back to the source files while `TC_DATA_URL` is set. With the variables unset, the loader builds from its source: Mulleder from Table S3 (pinned by sha256), Cachera from `GA1_2_4_6.csv` (pinned by sha256; the loader also needs an `SCerevisiaeGenome` to resolve gene names), and Cooper from its sha256-verified raw mirror only, because the Genome Research supplement is behind an institutional login and the loader has no network path.

## Caveats

**The served records lag the dev stores.** The record dumps, tables and figures above read the dev-tree stores; the query section reads release `2026.09.21-ab6d8c5d`. The Mulleder and Cachera stores were rebuilt after that release, so their content differs from what the graph serves:

```{include} _generated/amino-acid-betaxanthin/served_vs_dev.md
```

The Mulleder difference is the medium: the dev store carries the sourced `SM_AGAR` recipe of issue #143 (dataset note `torchcell.datasets.scerevisiae.mulleder2016`, section 2026.09.23), the release the bare `SM` stub. The Cachera difference is `perturbed_gene_name`: the dev store takes the genome's standard name (issue #195, closed), the release repeats the systematic name. No `phenotype` field differs in either dataset (the field tables above). The Cachera deletions outside `SCerevisiaeGenome.gene_set` are why the query returns fewer Cachera entries than the release serves.

**Mulleder 2016: source of the table.** The loader pins `Table_S3_Complete_Dataset.xls` by sha256 `a7fcb4bc8aa5e394e7f6e2b99e327eaa88fa04111ab5602fc7cb3445f653802e`. The paper's own supplement (Cell SI `mmc3.xls`, PMC5055083) is byte-identical to that pin (verification report `notes/assets/verification/2026.09.29/mulleder2016.md`, claim M1), so the pinned bytes are the published Table S3 and the Mendeley Data deposit is not a live dependency of the records. No raw-mirror record for the file exists under `$DATA_ROOT/torchcell-raw/` yet (issue #487). The paper also deposited raw-level data in MetaboLights study MTBLS434 (6,475 samples, a 19-metabolite table equal to the workbook's `data_raw` sheet to 3 decimals, and 6,475 mzML files); torchcell does not use it (same report, M1 and new finding 2; issue #487).

**Mulleder 2016: replicates.** Every record stores `n_replicates = 1`. Among the 4,678 released strains, 4,487 have one raw measurement and 191 have two to four (167 with 2, 20 with 3, 4 with 4), so the stored count understates those 191 (report, claim M2; issue #488). The reference `n_replicates = 1` describes the MCD robust mean, a statistic over every strain, as one measurement (report, new finding 5; issue #489).

**Mulleder 2016: 19 amino acids, not 18.** An earlier project note said the paper counts 18 amino acids; no sentence in the paper, its supplement or the PMC page gives 18, and the paper's Table S5 lists 19 (report, claim M3). The count is 20 proteinogenic amino acids minus cysteine: "Cysteine was omitted from further analysis, due to its property of being quickly oxidized upon cell lysis and hence giving imprecise results." (mirror OCR `mullederFunctionalMetabolomicsDescribes2016/paper.md`, line 395, sha256 `20412bec5b930d1f...`).

**Mulleder 2016: medium state.** The record stores the SM agar recipe with `state = 'solid'`, but the amino acids were extracted from a liquid SM subculture that the agar spots inoculated (report, claim M5; issue #143, open).

**Cooper 2010: the values are linear ratios, whatever the legend says.** The deposited legend and the Methods call Supplemental Table 4 log2-transformed ratios, but the released values are non-negative, include exact zeros, and the dense peaks average close to 1.0 (see the per-peak table above), which is what a linear ratio to the plate mean gives. The loader stores the values as released, names the linear ratio in `measurement_type`, and sets the reference to 1.0 (`torchcell/datasets/scerevisiae/cooper2010.py`, module docstring, "LEGEND VS VALUES"). Several peaks hold two or three amino acids that the platform did not separate (`glutamine+valine`, `methionine+proline`, `asparagine+tyrosine`, `leucine+isoleucine+citrulline`).

**Cooper 2010: strains, replicates and temperature.** Later rows of a duplicated Table 4 identifier are written to the build ledger, not served, and genes renamed onto one current systematic name carry two Cooper entries in one processed record (counts in the ledger line and in the query tables above). `n_replicates` is 1 on every key, the conservative lower end: strains were screened in duplicate but a strain with one quality trace was used alone, and no per-row count was released. The paper states no growth temperature, so the record stores `temperature = None` with a typed provenance gap (module docstring, "RECORD SHAPE" and "ENVIRONMENT").

**Mulleder and Cooper do not agree on the same gene.** Over the deletions both stores hold, the Spearman correlation between the two measurements of each shared amino acid is close to 0, and Cooper's own duplicate strains correlate about as weakly (tables and figure above). The two screens differ in medium (minimal vs synthetic complete), platform and normalization; which of these removes the agreement is not measured here.

**Cachera 2023: the cassette and the query filter.** Every Cachera genotype holds one deletion and the four Btx-cassette genes as `gene_addition` perturbations. Two of them, *CYP76AD1* and *DOD*, are not yeast genes, so a rule requiring every perturbed gene to be in `$gene_set` would drop every Cachera record; the query therefore constrains deletions only, as its header states:

```{literalinclude} ../../../torchcell/knowledge_graphs/queries/amino_acid_betaxanthin.cql
:language: text
:lines: 15-21
```

The cassette's *ARO4* and *ARO7* alleles are ectopic copies of native yeast genes, so a Cachera deletion of the native *ARO4* or *ARO7* has the cassette's own gene set as its aggregation key; those two strains share one processed record (the Btx-cassette line under the Cachera table, and the "2 Cachera" row of the query tables).

**Cachera 2023: medium, temperature and readout.** The record stores `Media(name="SC", state="solid")` at 30 °C (`torchcell/datasets/scerevisiae/cachera2023.py`, `create_experiment`). The paper places the screen on YPD with G418, and its OCR states no temperature; its readout is colony yellowness from color, not fluorescence, although `measurement_type` says `cri_spa_corrected_fluorescence_intensity_24h`:

```{include} _generated/amino-acid-betaxanthin/cachera_sources.md
```

Issue [#509](https://github.com/Mjvolk3/torchcell/issues/509) tracks the medium, temperature and measurement-type correction; it changes every served Cachera record, so it waits for the next full KG build.

## Provenance

```{include} _generated/amino-acid-betaxanthin/provenance.md
```

Generating scripts:

- `experiments/034-showcase-datasets/scripts/amino_acid_betaxanthin.py`: record dumps, summary tables, figures, the served-versus-dev comparison, the Cachera source lines and the provenance table, from the three dev-tree stores and the query job's cached result.
- `experiments/034-showcase-datasets/scripts/query_amino_acid_betaxanthin.py`, run by `experiments/034-showcase-datasets/scripts/gh_query_amino_acid_betaxanthin.slurm`: the query results, from the served graph; the slurm job id is in the query fragment above and in `experiments/034-showcase-datasets/results/amino_acid_betaxanthin_query.json`.
- `notes/assets/drawio/showcase-amino-acid-betaxanthin.drawio`: the diagram, exported headless with draw.io.
