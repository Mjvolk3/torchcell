---
id: cutd0mltas3wg2bhhavl5na
title: Bacteria_common
desc: ''
updated: 1791377722232
created: 1791377722232
---

## 2026.10.07 - Bacterial loader skeleton (plan Step 7)

`torchcell/datasets/bacteria_common.py` is the shared skeleton of the E. coli K-12 and
P. putida KT2440 loaders, the bacterial counterpart of
[[torchcell.datasets.scerevisiae.gene_name_reconcile]]. Design: section 4 and section 5
item 7 of [[plan.bacteria-ontology-genome]]. The loader packages
`torchcell/datasets/ecoli/` and `torchcell/datasets/pputida/` exist with no loader yet;
they are imported by `torchcell/datasets/__init__.py`, by
`torchcell.database.build_dataset_lmdb.resolve_dataset_class` and by
`torchcell.knowledge_graphs.kg_manifest._dataset_class`, beside the existing
`import torchcell.datasets.scerevisiae`, so a loader added to either package and
imported in its `__init__.py` is in `dataset_registry` everywhere a dataset is looked up
by name.

### API

| name | what it does |
|---|---|
| `bacterial_genome(host, strain, data_root=None)` | The genome of `strain` (`EcoliK12MG1655Genome`, `EcoliK12BW25113Genome`, `PPutidaKT2440Genome`) on its default cache root `<data_root>/data/ecoli/mg1655/genome`, `.../bw25113/genome` or `data/pputida/kt2440/genome`, with `overwrite=False`. `data_root` defaults to `$DATA_ROOT`. A strain of the other host is refused. |
| `reconcile_locus_tags(genome, names, *, label)` | Retain-all mapping of a dataset's source names to the genome's GenBank locus tags. Returns `(stored_names, LocusTagReconciliation)`. |
| `LocusTagReconciliation` | Pydantic report: `status_histogram` (every `GeneNameStatus`, zeros included), `layer_histogram` (which resolver layer decided each name), `remapped`, `kept_on_collision`, `retired_kept`, `ambiguous_kept` (with candidates), `case_insensitive`, `outside_namespace` (stored names a bacterial leaf will refuse), `resolved_fraction`, and `require_resolved(min_fraction)`, which raises `LocusTagResolutionError`. |
| `resolution_layer(genome, resolution)` | The layer that decided one resolution, read from the resolver's note; a note of no known form is refused. |
| `assembly_reference(strain, *, background=None, data_root=None)` | The `AssemblyReferenceGenome` a record stores: `species` from the genome class, `assembly_set` and the GenBank `GCA_` accession read from the set's deposited `_assembly_report.txt`. With a background, `strain` is the background's `name` and the background must be an edit of `strain`. |
| `read_assembly_report(strain, data_root=None)` | The parsed report header (`AssemblyReport`); refuses a report naming another assembly, an accession pair other than the schema's `ASSEMBLY_SET_ACCESSIONS`, a repeated key, or a header line it cannot read. |
| `eck_crosswalk(mg1655_genome, bw25113_genome)` | The ECK synonym join (`EckCrosswalk` of `torchcell.sequence.genome.ecoli.k12`): `pairs` (4,423 one-to-one on the deposited sets) and `numeric_disagreements` (11). |
| `BacterialGenomeInjector(data_root).genome_kwargs(dataset_class)` | The host-aware injection rule shared by the build entry points (below). |
| `STRAIN_GENE_NAMESPACES`, `HOST_STRAINS`, `BACTERIAL_GENOME_CLASSES`, `host_of_strain`, `strain_of_assembly_set`, `gene_namespace_of`, `declared_reference_strain` | The strain, host, namespace and class vocabulary. |
| `BacterialGeneNamespace`, `BACTERIAL_LOCUS_TAG_PATTERNS`, `BACTERIAL_LOCUS_TAG_PATTERN`, `BACTERIAL_ASSEMBLY_SETS`, `LOCUS_TAG_PATTERNS` | Imported from `torchcell.datamodels.schema` and re-exported (the same objects); `LOCUS_TAG_PATTERNS` is those patterns compiled. Nothing is restated here. |

Strains are `MG1655`, `BW25113` and `KT2440` (the schema's `BacterialReferenceStrain`).
Their namespaces are `ecoli_k12_mg1655_bnumber`, `ecoli_k12_bw25113_locus_tag` and
`pputida_kt2440_locus_tag`. E. coli always names a strain: a `BW25113_` number is not an
MG1655 b-number (plan D5), so there is no strain-free E. coli genome.

The GenBank accession is the one pinned because the records' identifiers are GenBank
locus tags (plan D1); RefSeq retags BW25113 (`BW25113_RS...`) and KT2440 (`PP_RS...`).

### Resolution order and the retain-all policy

`reconcile_locus_tags` calls `BacterialGenome.resolve_gene_name` once per distinct name.
The layers, in order:

1. locus tag (`CURRENT` for a gene, `NON_GENE_FEATURE` for a pseudogene);
2. `old_locus_tag`;
3. RefSeq locus tag, through that RefSeq gene's `old_locus_tag`;
4. gene symbol (`/gene`);
5. `gene_synonym` (ECK numbers in E. coli; KT2440's GenBank file has none);
6. not found: `RETIRED`, kept as given.

Within each layer an exact-case match wins and a case-insensitive match is used only
when there is none; such names are listed in `case_insensitive` for review.

Retention, identical to the yeast `reconcile_systematic_names`: a name resolving to one
locus (`CURRENT`, `RENAMED`, `NON_GENE_FEATURE`) is stored as that locus tag, unless a
second distinct source name resolves to the same locus; then both are kept as given, so
two source strains never collapse onto one genotype identity. `RETIRED` and `AMBIGUOUS`
names are kept as given. No record is dropped for a naming reason. A kept name that is
not a locus tag of the namespace shows up in `outside_namespace`; the loader decides what
to do with those records and states it.

Measured on the real MG1655 annotation (`tests/torchcell/datasets/test_bacteria_common.py`,
data-gated): `b0001` and `b0006` are `CURRENT`, `thrB` resolves at the symbol layer to
`b0003`, `ECK0004` and `ECK0006` at the synonym layer to `b0004` and `b0006`, so `b0006`
and `ECK0006` collide and are both kept as given, and `b9999` is `RETIRED`. The MG1655
GenBank file carries ECK synonyms but no JW numbers (`JW0003` resolves `RETIRED`).

### The injection rule

A loader receives genomes by `__init__` parameter NAME, in all three build entry points
(`torchcell/knowledge_graphs/create_kg.py`,
`torchcell/knowledge_graphs/create_scerevisiae_kg_small.py`,
`torchcell/database/build_dataset_lmdb.py`):

- `genome` receives `SCerevisiaeGenome`, and `scerevisiae_graph` the graph on it, exactly
  as before;
- `ecoli_genome` receives the `EcoliK12Genome` and `pputida_genome` the
  `PPutidaKT2440Genome` of the strain the loader class states in
  `REFERENCE_STRAIN` (`"MG1655"`, `"BW25113"` or `"KT2440"`; annotate it
  `ClassVar[EcoliK12StrainName]` or `ClassVar[Literal["KT2440"]]` so mypy checks it).

`BacterialGenomeInjector` builds each strain's genome on the first loader that asks for
it and hands the same object to every later loader of that strain. A loader naming
neither parameter gets nothing, so a yeast-only build constructs no bacterial genome and
never reads the bacterial tier. Refused before any genome is built: a bacterial loader
(one naming a bacterial parameter or stating `REFERENCE_STRAIN`) that also names
`genome`, a loader naming both bacterial parameters, a bacterial parameter without
`REFERENCE_STRAIN`, a strain of the other host, and a strain outside the vocabulary.

`torchcell/verification/runners.py` has no loader to inject into; its counterpart is
selection by the record's own reference. `_reference_assembly_set(genome_reference)`
reads the stored reference: an `AssemblyReferenceGenome` names its set, and a reference
without `assembly_set` must say Saccharomyces cerevisiae (SGD R64). On that set,
`_gene_set_for_reference` returns the L4 gene universe and `_genome_for_reference` the
genome whose `resolve_gene_name` the canonical-name rule uses. The bacterial universes
are `_ecoli_k12_gene_set(data_root, strain)` and `_pputida_gene_set(data_root)`: the locus
tag of every `gene` row of the set's `_feature_table.txt.gz`, with every protein of
`_protein.faa.gz` required to be the product of a `CDS` row of one of those genes (a
disagreement is refused). On the deposited sets: 4,651 MG1655, 4,490 BW25113 and 5,786
KT2440 loci, equal to the GenBank gene-feature counts. Pseudogenes are included, since a
pseudogene tag is a locus a record can carry.

A bacterial loader therefore looks like this (sketch, not a landed loader):

```python
@register_dataset
class FitnessWetmore2015Dataset(ExperimentDataset):
    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = "BW25113"

    def __init__(
        self,
        root: str = "data/torchcell/fitness_wetmore2015",
        io_workers: int = 0,
        ecoli_genome: EcoliK12Genome | None = None,
        ...
    ) -> None:
        self.ecoli_genome = ecoli_genome
        ...

    def process(self) -> None:
        if self.ecoli_genome is None:  # a direct run; the build entry points inject it
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        stored, report = reconcile_locus_tags(
            self.ecoli_genome, raw["locusId"], label=self.name
        )
        report.require_resolved(0.95)  # the PR states and justifies the threshold
        reference = assembly_reference(self.REFERENCE_STRAIN)
        ...
```

### Per-dataset checklist (verbatim from [[plan.bacteria-ontology-genome]] section 4)

**Per-dataset checklist** (the bacterial form of the "Adding Datasets (Modular)" rules in
`CLAUDE.md`), to be copied into each loader's PR body:

1. **Pin the paper.** `citation_key` plus the mirrored `paper.md` sha256 from that key's
   `manifest.json` on `tc-lit`; every sourced value quotes a verbatim substring of those
   bytes.
2. **Identify the exact SI table and the exact column.** Name the file, the sheet or table
   number, and the column the loader consumes. Different columns carry different replicate
   structures; the statistic stored is the one belonging to the consumed column.
3. **Settle the background strain before anything else.** MG1655 or BW25113 for an *E.
   coli* row (the expansion document's `K-12` label is deliberately ambiguous and is NOT
   the answer), which assembly set it pins, and the quote that establishes it. A row whose
   paper does not say gets a typed `ProvenanceGap`, not a guess.
4. **Map every reported identifier to the pinned assembly's locus tags** through
   `reconcile_locus_tags`, and report the status histogram: how many CURRENT, how many
   resolved from a symbol or an ECK synonym, how many retired and kept, how many ambiguous.
   A row whose identifiers resolve below a stated threshold stops and reports rather than
   dropping records.
5. **Source `n_samples` and the uncertainty TYPE with a verbatim quote.** Combing the full
   Methods and the SI column descriptions, searching every synonym (`replicate`, `colony`,
   `biological replicate`, `bootstrap`, `standard deviation`, `n =`), and following
   deferrals into the cited method paper (RB-TnSeq rows will defer to Wetmore 2015, which is
   itself row 2, so that paper is mirrored anyway). On a range with no per-record column,
   apply the documented resolution order: back-solve from a companion statistic, else the
   conservative lower end, and say which rule was used.
6. **Media and environment from the Methods**, through the `MEDIA_LIBRARY` additions of
   section 3d, with the carbon source as a component or as a typed physical factor
   depending on whether it is fixed or varied.
7. **Deduplicate against the supersets.** The expansion document names two: the dense *P.
   putida* 4,732-gene-by-332-sample fitness matrix is a superset of the other KT2440
   transposon rows, and the *E. coli* compendium is a superset of the method paper's own
   *E. coli* experiments. The superset is loaded; the subsumed rows become provenance
   records naming which source experiments the matrix covers, not separate loaders.
8. **Write the dataset's Dendron note** with the sourcing decisions, the dropped-record
   counts and the identifier histogram, as the yeast loaders do
   ([[torchcell.datasets.scerevisiae.vanacloig2022]] is the model).
9. **Build the LMDB in the dev tree and report the manifest**:
   `python -m torchcell.database.build_dataset_lmdb --dataset <Class>`, which writes
   `preprocess/build_manifest.json`; the KG build's freshness gate reads exactly that file.

Acceptance per loader: the LMDB builds, `len(dataset)` matches the record count the PR
states, the build manifest reads fresh under
`python -m torchcell.provenance.build_manifest`, and the family verifier passes L0 to L4.

## 2026.10.07 - E. coli B REL606 as a fourth strain

`REL606` joins the strain vocabulary ([[torchcell.sequence.genome.ecoli.rel606]]):
`HOST_STRAINS["ecoli"]` is `("MG1655", "BW25113", "REL606")`,
`STRAIN_GENE_NAMESPACES["REL606"]` is `ecoli_b_rel606_locus_tag` (pattern
`^ECB_[rt]?\d{5}$`), and `BACTERIAL_GENOME_CLASSES["REL606"]` is `EcoliBREL606Genome`,
built on `data/ecoli/rel606/genome`. REL606 is an E. coli B strain, so its genome is not
an `EcoliK12Genome`; a REL606 loader still names `ecoli_genome` and states
`REFERENCE_STRAIN: ClassVar[EcoliBStrainName] = "REL606"`, and the injector hands it the B
genome. `bacterial_genome("ecoli", "REL606")` has its own overload returning
`EcoliBREL606Genome`; `BacterialStrainGenome` names the union of the three genome
classes that `bacterial_genome` and the injector return. `assembly_reference("REL606")`
reads the deposited report (taxid 413997) and pins `GCA_000017985.1`. `eck_crosswalk`
stays K-12 only.
