# Datasets

torchcell provides built datasets the way `torchvision.datasets` provides image datasets: a loader class per source publication, a versioned archive of the built records (`{doc}`../guide/downloads``), and a supported query that returns the records from the served knowledge graph. The pages under each organism take one group of datasets and show what the experiments measured, what one stored record looks like, how the values are distributed, the query that returns them, and how to download the built stores. Every table, figure, record dump and query result on these pages is written by a script under `experiments/034-showcase-datasets/scripts/`, and query results come from a slurm job against the served release named on the page. Caveats that change how a value should be read are stated on the page that shows the value.

## Size of the collection

```{include} _generated/served_counts.md
```

The full list of loaders, with the phenotype class, the genotype count and the record count of each, is on the guide's {doc}`../guide/datasets` page; the served counts above come from the committed release snapshot, so the two agree only for datasets admitted at that release.

## What every dataset page provides

A dataset page is a fixed set of sections, in this order, so two pages can be read the same way. `essentiality-smf` and `amino-acid-betaxanthin` are the reference pages; a new page copies their structure.

1. **Introduction and terms.** One paragraph naming each loader class, what it records and from which paper, and the supported query that joins them. A term list, alphabetical, defines every abbreviation and every torchcell term (entry, processed record, reference, source key) before the text uses it.
2. **What the experiments measured.** One draw.io diagram (`notes/assets/drawio/<page>.drawio`, exported headless to SVG) with three panels: **a** the sources and what each screen measured, in which strain, medium and temperature; **b** the stored record (experiment, reference, publication) and the phenotype class; **c** the query, the conversion or aggregation, and what one processed record holds. The diagram is composed from the loaders and the query, never redrawn from a published figure, and follows the repository figure standard (Nature print size, the palette, the font ladder).
3. **The records.** One record per dataset as its loader stored it, re-validated through the pydantic class and printed from `model_dump()`, cut to the named fields with every cut marked by the number of fields it hides. The records show the same gene where the datasets share one.
4. **The data.** Counts per dataset (records, genes, measurement type, medium, temperature, parent strain, replicate counts, uncertainty stored), the overlap of genes between the datasets, the distribution of every stored value as a table and at least one figure, and one exploration that a reader of the group would ask for (agreement between two datasets on the same gene, a histogram by strain type, a per-key summary). Figures follow the repository plotting standard (`torchcell.utils.apply_paper_style`: Arial 6 pt, the palette, boxed axes, a standard panel width, true-size SVG). A figure with more than one panel carries bold lowercase panel letters drawn by `torchcell.utils.panel_label`, and its caption names the panels in the paper's form: `**a**, ...; **b**, ...`. Every figure has a `:name:`, Sphinx numbers it (`Fig. 1`), and the text refers to it with `{numref}`, never "the figure above".
5. **Querying the graph.** The supported query as a `literalinclude` of its `.cql` file, the `Neo4jCellDataset` construction with the converter, deduplicator and aggregator the group needs, the source of any converter as a `literalinclude`, and what the query returned from a named served release: the slurm job id, entries per dataset at each pipeline stage, processed records by dataset membership, and one processed record per membership pattern.
6. **Getting the data.** The `tc-data` slug of each dataset, the `DatasetClient` and curl forms, what the loader does with and without `TC_DATA_URL`, and the state of the public endpoint.
7. **Caveats.** Every statement that changes how a value should be read, each with its source: the mirror OCR file, line and sha256, the verification report and claim id, the loader line, or the issue number. A caveat with no issue names that fact.
8. **Provenance.** A table per dataset: loader, dev store, raw input with sha256, source of record, paper mirror with sha256, DOI and PubMed id. Then the generating scripts.

Two rules hold for every number on a page. Every table, figure, record dump and query count is written by a committed script under `experiments/034-showcase-datasets/scripts/` into the page's `_generated/` directory, and the page only includes those fragments; nothing is typed by hand. Every query against the served graph runs as a slurm job whose id the fragment records, and the page names the release the job read. The page path is registered as `docs_page` on its supported query in `torchcell/knowledge_graphs/supported_queries/registry.json`, and the Sphinx build must add no warnings.

```{toctree}
:maxdepth: 2

scerevisiae/index
```
