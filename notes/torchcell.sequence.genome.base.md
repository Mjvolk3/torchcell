---
id: 6gj5k35sly0vf3h74b1jhdz
title: Base
desc: ''
updated: 1791371982347
created: 1791371982347
---

## 2026.10.07 - Extracted from s288c: the organism-agnostic genome

Step 2 of [[plan.bacteria-ontology-genome]] (`refactor/genome-organism-agnostic`). Everything `SCerevisiaeGenome` did that is not an SGD convention moved here, so the bacterial genomes of step 3 subclass it instead of copying 2,000 lines. No bacterial class exists yet; the diff is a pure refactor whose acceptance is byte-level equivalence.

### Layers

| Layer | Module | Holds |
|---|---|---|
| 1 | `torchcell/sequence/genome/base.py` | `AnnotatedGenome`, `AnnotatedGene`, `GenomeReleaseFiles`, the `data.db` machinery, `GeneNameStatus` / `GeneNameResolution` |
| 2 | `torchcell/sequence/genome/scerevisiae/s288c.py` | `SCerevisiaeGenome`, `SCerevisiaeGene`: the SGD specifics only |
| 3 (step 3) | `ecoli/k12.py`, `pputida/kt2440.py` | the bacterial subclasses |

`base.py` imports nothing from `torchcell.sequence.genome.scerevisiae` and nothing that downloads (`test_base_imports_nothing_organism_specific` pins its torchcell imports to `literature.manifest`, `sequence`, `sequence.db_connection` and `sequence.genome.registry`).

### What moved, and how

- **The `data.db` machinery, verbatim.** Lines 506-1419 of the pre-refactor `s288c.py` (`GENOME_DB_FILENAME` through `_remove_private_copy`: the record models, `RECORD_VERSION`, the build/install/migrate/rebuild functions, torn-file and hot-journal detection, the dead-build sweep) were sliced into `base.py` by line range, not retyped, so every comment documenting an incident moved unchanged. The `AnnotatedGenome` class docstring (the cache contract and its known limits) is the old `SCerevisiaeGenome` docstring, with only the first line and the `drop_*` sentence generalized.
- **The genome methods**: `__attrs_post_init__` (now through the hooks), `db`, `go_dag`, `__reduce_ex__`, `_writable_db`, `_apply_write`, `_write`, `remove_deprecated_go_terms`, `_deprecated_go_updates`, `alias_to_systematic`, `feature_index`, `resolve_gene_name`, `go`, `go_subset`, `go_genes`, `go_subset_genes`, `get_seq`, `gene_attribute_table`, `feature_types`, `compute_gene_set`, `drop_empty_go`, `__getitem__`. `drop_chrmt` stayed in the yeast subclass: a mitochondrion is a eukaryote fact.
- **The gene**: coordinates, strand check, sequence, protein and CDS lookup, `codon_frequency`, the three windows and `__repr__` moved into `AnnotatedGene`.
- **`GeneNameStatus` / `GeneNameResolution`** moved with organism-neutral docstrings; the enum values are unchanged.

### Hooks (what a subclass sets)

| Hook | Kind | SGD value |
|---|---|---|
| `ASSEMBLY_SET`, `GENOME_VERSION` | `ClassVar[str]` | `sgd_S288C_R64-4-1_20230830`, `R64-4-1_20230830` |
| `LOCUS_FEATURE_TYPES` | `ClassVar[frozenset[str]]` | `SGD_LOCUS_FEATURE_TYPES` (10 types) |
| `ANNOTATION_NAME`, `ANNOTATION_RELEASE` | `ClassVar[str]` | `R64`, `R64-4-1` (the resolver's `note` text, which tests pin) |
| `release_files()` | classmethod -> `GenomeReleaseFiles` | the four SGD release files |
| `fasta_chromosome(record)` | classmethod | `[chromosome=<roman>]` / `[location=mitochondrion]` -> 0 |
| `gene_class()` | classmethod | `SCerevisiaeGene` |
| `_prepare_go_obo()` | method | `<go_root>/go.obo`, downloaded when absent (the one behavior bacteria must not inherit, D8) |
| `AnnotatedGene.GO_ATTRIBUTE` | `ClassVar[str]` | `Ontology_term` |
| `AnnotatedGene.seqid_to_chromosome(seqid)` | classmethod | `chrI`..`chrXVI` -> 1..16, `chrmt` -> 0 |
| `AnnotatedGene.coding_feature()` | method | the five-prime-UTR-intron CDS selection (default: the gene row) |
| `AnnotatedGene.annotate(gene_feature)` | method | adds `ontology_term`, `display`, `dbxref`, `orf_classification` |

`AnnotatedGenome` is generic in its gene type (`AnnotatedGenome[GeneT: AnnotatedGene]`), so `SCerevisiaeGenome.__getitem__` is typed `SCerevisiaeGene | None` and strict mypy still sees `gene.orf_classification`. Its init fields are `genome_root` and `overwrite`; `SCerevisiaeGenome` redeclares `genome_root` (default `data/sgd/genome`), adds `go_root`, and redeclares `overwrite`, so its positional order stays `(genome_root, go_root, overwrite)`. The refusal messages render the subclass's own init fields (`_constructor_call`), so the yeast text is unchanged and a bacterial refusal will not name a `go_root` it has no field for. Pickling reopens through `_restore_annotated_genome(cls, init_kwargs, private_db_path)`; `s288c._restore_genome(cls, genome_root, go_root, private_db_path)` stays as the target pickles written before this change name.

### Compatibility of `s288c`

Every name `s288c` defined before is re-exported from it (explicit `X as X` imports, so strict mypy sees them). Tests rebind many of those names on the `s288c` module (`monkeypatch.setattr(s288c, "write_genome_database", ...)`, `s288c.filecmp`, `s288c.resolve`), and the code that looks them up now lives in `base`, so a plain re-export would leave those patches inert. The module's type is `_BaseMirroringModule`: assigning or deleting a name `s288c` shares with `base` (bound to the identical object) does the same in `base`. `test_s288c_reexports.py` pins the contract.

### Renames (D3)

- `SYSTEMATIC_GENE_PATTERN` -> `SGD_SYSTEMATIC_GENE_PATTERN` in `torchcell/datamodels/schema.py`. `BackgroundAllele._validate_systematic` references it, so the schema-impact report marks `BackgroundAllele` stale, reaching 4 served loader modules (HetHillenmeyer2008Dataset, HomHillenmeyer2008Dataset, EnvChemgenHoepfner2014Dataset, EnvChemgenVanacloig2022Dataset, EnvChemgenWildenhain2015Dataset); 0 breaking. Accepted by the plan: the program takes a full rebuild.
- `_LOCUS_FEATURE_TYPES` -> `SGD_LOCUS_FEATURE_TYPES` (stays in `s288c`).
- `SGD_GENE_FASTAS` and `_sgd_gene_set` in `torchcell/verification/runners.py` keep their names; no bacterial siblings yet.

### Equivalence (measured)

Fresh `data.db` builds into scratch roots, once with `PYTHONPATH` at the primary checkout on origin/main `fb5adf027` and once at this worktree, both with `overwrite=False` on an empty root:

| Quantity | Before | After |
|---|---|---|
| `database_content_digest` | `9fae73b71df938427cd82c2068978cb6948d885ecbbe2a8d0bc498cd31e8c1f4` | identical |
| `len(genome.gene_set)` | 6,607 | 6,607 |
| `len(genome.go)` | 4,686 | 4,686 |
| sha256 over every gene's fields, GO and four windows | `7ff6b8cf...a05` | identical |

The feature index, `go_genes`, `chr_to_nc`, `chr_to_len`, the attribute-table shape, `alias_to_systematic` size, a `get_seq` slice and eight `resolve_gene_name` results were also identical (14 of 14 keys).

### Tests

- `tests/torchcell/sequence/genome/test_base.py`: a `ToyGenome` with one linear replicon, its own locus types, GO in a `go_terms` attribute and no SGD convention; pins resolution through the tier, the recorded source, chromosome 1, the gene set, both strands, GO from the hook (an `Ontology_term` row is ignored), every resolver layer with the toy labels, the subclass refusal text, the generic pickle restore, `remove_deprecated_go_terms` rewriting only the hook attribute, `drop_empty_go`, `get_seq`, and the abstract surface.
- `tests/torchcell/sequence/genome/scerevisiae/test_s288c.py` and `test_s288c_synthetic.py` are unchanged and pass.
