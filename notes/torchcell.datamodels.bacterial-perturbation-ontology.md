---
id: zcs5vdr7a3p07abq26we8kh
title: Bacterial Perturbation Ontology
desc: ''
updated: 1791374899318
created: 1791374899318
---

## 2026.10.07 - The bacterial schema layer (plan step 4)

Step 4 of [[plan.bacteria-ontology-genome]]: the pydantic layer the fifty bacterial
datasets (35 *E. coli*, 15 *P. putida* KT2440) need, added so that **no served dataset's
schema closure moves**. Everything here is new classes and module-level union/map
appends; no existing class was edited. The measurement that proves it is in the PR body
and is reproduced at the end of this note.

Decisions this implements: **D1** (locus tag plus a namespace field), **D2** (keep
`systematic_gene_name`, namespace in a sibling field), **D6** (the assembly pin as a
subclass), **D7** (keep the additive list minimal). Deliberately out of scope and left to
later steps: `media.py` (step 5), the BioCypher graph classes and adapter methods
(step 6), the loaders (step 8).

### Why a namespace field at all

`Genotype` derives its content identity from the sorted `systematic_gene_name`s of its
perturbations. Without a namespace a b-number and a `PP_` tag are two anonymous strings
in one id space. The three tag forms are disjoint today, so the namespace is not needed
to disambiguate them *now*; it is needed so that a future host whose tags collide with an
existing one stays separable without a second rename, and so a record says which genome
it is written against rather than leaving it to be inferred from the dataset.

`BacterialGeneNamespace` is a `Literal` with one member per deposited strain set:

| namespace | locus-tag pattern | gene features matching |
|---|---|---|
| `ecoli_k12_mg1655_bnumber` | `^b\d{4}$` | 4,651 / 4,651 |
| `ecoli_k12_bw25113_locus_tag` | `^BW25113_\d{4}$` | 4,490 / 4,490 |
| `pputida_kt2440_locus_tag` | `^PP_(?:\d{4}\|(?:5\|16\|23)S[A-G]\|t\d{2}\|tm\d{2}\|mr\d{2}\|r\d{2})$` | 5,786 / 5,786 |

Measured by parsing the `gene` features of each set's deposited GenBank flat file, not
recalled. The KT2440 pattern is not digits-only on purpose: **165 of its 5,786 gene
features carry a named structural-RNA tag** (`PP_16SA`-`PP_16SG`, `PP_23SA`-`PP_23SG`,
`PP_5SA`-`PP_5SG`, 75 `PP_tNN`, 67 `PP_mrNN`, `PP_tm01`, `PP_r01`), so a `PP_\d{4}`-only
pattern would have silently refused every rRNA, tRNA and `PP_mr` gene. All three patterns
are pairwise disjoint and none matches a yeast systematic name
(`tests/torchcell/datamodels/test_schema.py::test_the_three_locus_tag_patterns_are_pairwise_disjoint`,
`::test_a_kt2440_structural_rna_tag_is_admitted`).

Each leaf validates its own name with the union of the three patterns and then requires
`gene_namespace` to AGREE with the pattern the tag actually matched, so a KT2440 tag
cannot be filed under the MG1655 namespace. **The yeast `GenePerturbation`
validator is untouched**, so `b0002` still fails on every yeast leaf: widening it was
measured to move 35 of 36 served loader closures and would have admitted a b-number into
an SGA deletion.

### The five perturbation leaves (plan 3a)

Each is a subclass of the axis leaf its OUTCOME already belongs to, so existing filters
keep working (`issubclass(_, DeletionPerturbation)` still catches a bacterial knockout).

| class | parent | `perturbation_type` | state | SO mechanism | rows of the fifty it serves |
|---|---|---|---|---|---|
| `BacterialDeletionPerturbation` | `DeletionPerturbation` | `bacterial_deletion` | `absent` | SO:0000159 `deletion` | the 9 `K-12-KO` rows (Keio plus the chemical-genomics screens) and the Keio-derived `K-12-KO x KO` pairs |
| `TransposonInsertionPerturbation` | `PresenceAbsencePerturbation` | `transposon_insertion` | `absent` | SO:0001218 `transgenic_insertion` | the 10 `K-12+transposon` / `KT2440+transposon` rows |
| `BacterialCrisprInterferencePerturbation` | `CrisprInterferencePerturbation` | `bacterial_crispr_interference` | `present` | SO:0001998 `sgRNA` | the 10 `+guide` rows |
| `PromoterReplacementPerturbation` | `ExpressionModulationPerturbation` | `promoter_replacement` | `present` | SO:1000032 `delins` | the `KT2440+promoter` row and the promoter arms of the combinatorial designs |
| `HeterologousPathwayPerturbation` | `GeneAdditionPerturbation` | `heterologous_pathway` | `present` | SO:0000667 `insertion` | the 6 `engineered-chassis` production rows |

Both new SO id/name pairs were read from the Sequence Ontology release of **2026-08-07**
(`so.obo`, sha256 `22a8f3ec2b49125dbb6cee8456f0bca86dd8c98f433165ffa4b554da4f155204`),
not recalled, and are pinned in `test_ontology_all_trees.py`'s `SO_ALLOWED`:

- **SO:0001218 `transgenic_insertion`**, "An insertion that derives from another
  organism, via the use of recombinant DNA technology" (`is_a` SO:0000667 `insertion`).
- **SO:1000032 `delins`**, "A sequence alteration which included an insertion and a
  deletion, affecting 2 or more bases" -- a promoter replacement is the native part
  removed and the new one put in its place, which is neither a plain insertion nor a
  plain deletion.

Three judgment calls worth recording:

- **A transposon insertion is `state="absent"` with an *insertion* mechanism.** The axis
  field is the biological fact (the gene's function is gone) and the SO term is the
  mechanism (what was done to the DNA); the pairing is deliberate, not a mismatch.
  The leaf is a **realized** genotype: a pooled library is built by random insertion and
  then sequenced, so `barcode`, `insertion_position` and `insertion_strand` are
  measurements, not a design.
- **`PromoterReplacementPerturbation` re-declares `crispr` as optional.** The expression
  axis base requires a `CrisprConstruct` because it was built for the guide-directed
  leaves, and a promoter swap introduces no guide. Widening the base class instead would
  have moved the served Lian and Mormino closures, so the narrowing lives on the leaf.
  `expression_direction` is REQUIRED with no default there: a promoter swap can raise or
  lower expression depending on the part, and a default would assert a direction the
  record never stated.
- **`gene_namespace` on `HeterologousPathwayPerturbation` names the HOST**, not the
  namespace of `systematic_gene_name` (usually a heterologous symbol such as `Efa:mvaE`).
  That leaf inherits the gene-addition axis's relaxed name validator, so the
  tag-to-namespace check applies only when the identifier IS a host locus tag, which it
  legitimately is for an extra copy of a native gene in the same pathway. One rule covers
  both: `gene_namespace` names the host genome the record is written against, and when the
  identifier is one of that host's tags the two must agree.

### The assembly pin (plan 3b) and the serialization caveat

`AssemblyReferenceGenome(ReferenceGenome)` adds `assembly_set` (a `Literal` over the
three deposited set ids), `assembly_accession` (one of that set's GCA/GCF pair) and an
optional `background: BacterialStrainBackground`.

**The caveat, which is the single most important thing in this note.** pydantic v2
serializes a field by its DECLARED type. An `AssemblyReferenceGenome` placed in a slot
annotated `ReferenceGenome` keeps the subclass as a python attribute but **dumps only
`{species, strain, ploidy}`**, and re-validating that dump returns a plain
`ReferenceGenome`. The assembly pin is lost silently, with no error anywhere. So the pin
survives **only where a field is re-annotated to the subclass**, which is why every
bacterial `ExperimentReference` below declares `genome_reference: AssemblyReferenceGenome`
-- the same narrowing `StrainEnvironmentResponseExperimentReference` already does for
`StrainReferenceGenome`. Pinned as a test, not a comment:
`test_schema.py::test_the_assembly_pin_survives_only_where_the_field_is_re_annotated`
exercises both halves (base-typed slot loses it, narrowed slot keeps it through dict and
JSON).

The alternative, `assembly_set` on `ReferenceGenome` itself, was measured to move **36 of
36** served closures; the subclass moves 0. The cost of the subclass route is that the 51
served yeast datasets keep storing `{species, strain, ploidy}` with no assembly pin, so
"which assembly" stays implicit for yeast and explicit for bacteria until some later
rebuild unifies them.

`assembly_set` is a `Literal` in `schema.py` rather than a call into
`torchcell.sequence.genome.registry` because **`torchcell.sequence` imports
`torchcell.datamodels`**, so importing the registry from `schema.py` is an import cycle
(and would drag the literature subsystem's FastAPI/Zotero imports into every process that
constructs a record). Equality with the registry's own constants is pinned by
`test_schema.py::test_the_assembly_set_vocabulary_equals_the_registry_ids`, which is what
keeps the restatement from drifting. The accession pairs come from each set's deposited
`_assembly_report.txt`:

| assembly set | strain | GenBank | RefSeq |
|---|---|---|---|
| `ecoli_K12_MG1655_ASM584v2` | MG1655 | `GCA_000005845.2` | `GCF_000005845.2` |
| `ecoli_K12_BW25113_ASM75055v1` | BW25113 | `GCA_000750555.1` | `GCF_000750555.1` |
| `pputida_KT2440_ASM756v2` | KT2440 | `GCA_000007565.2` | `GCF_000007565.2` |

A record must name one of its own set's two accessions, so a (set, accession) pair cannot
disagree.

### The bacterial strain background (plan 3e)

`BacterialStrainBackground` and `BacterialBackgroundAllele` are **siblings** of the yeast
`StrainBackground` / `BackgroundAllele`, not widenings: widening
`StrainBackground.reference_strain` past `Literal["S288C"]` or touching
`BackgroundAllele` moves 4 of 36 served closures (the four chemogenomic loaders). The
duplication is the measured price, and unifying the two families later is a rebuild whose
purpose is that unification.

Two differences follow from the host rather than from taste: the allele's identifier is a
namespaced locus tag, and **there is no `zygosity` field at all** -- these strains are
haploid, so a copy count would have one possible value and would be a field that cannot
carry information. `functional_copies` is correspondingly 1 or 0.

**BW25113's background, and what is deliberately NOT asserted.** The genotype is sourced
from the tier, not from a paper: the `source` feature of
`GCA_000750555.1_ASM75055v1_genomic.gbff.gz` carries
`/note="from B. L. Wanner laboratory; genotype: rrnB3 lacZ4787 hsdR514 (araBAD)567 (rhaBAD)568 rph-1"`,
and that file's bytes are sha256-verified by `registry.resolve` on every call. So
`BW25113_BACKGROUND_GENOTYPE` is the genotype clause verbatim and
`BW25113_BACKGROUND_LESIONS` its six designations, re-read from the real bytes by
`test_schema.py::test_the_bw25113_genotype_is_verbatim_from_the_deposited_genbank_bytes`
(data-gated).

They are **not** expanded into six `BacterialBackgroundAllele` records here, and the
reason is evidence discipline. The `/note` gives each lesion's DESIGNATION and nothing
else, while an allele record also asserts an edit kind, a functional status and a locus.
Measured from BW25113's own annotation: three of the six name a gene it still annotates
(`lacZ` at `BW25113_0344`, `hsdR` at `BW25113_4350`, `rph` at `BW25113_3643`), and the two
operon deletions name genes it does **not** annotate at all (`araB`, `araA`, `rhaB`,
`rhaA` are absent, which is what being deleted looks like), so giving those a locus would
mean borrowing MG1655 b-numbers -- the cross-strain inference D9 refuses. The loader that
needs typed alleles sources the edit kinds from the BW25113 construction paper and records
which lesion it mapped how. `genotype_statement` carries the verbatim string beside
whatever typed alleles a loader does source, so nothing the source said is lost.

### The three new phenotypes and the eleven experiment pairs (plan 3c)

| class | label | graph level | rows it serves |
|---|---|---|---|
| `ProductTiterPhenotype` | `titer` (stat `titer_se`) | `global` | the 8 production-campaign rows plus the 2 combinatorial designs |
| `ProteinTurnoverPhenotype` | `degradation_rate` (stat `degradation_rate_se`) | `node` | Gupta 2024 per-protein degradation and turnover rates |
| `FluxPhenotype` | `net_flux` (no single stat) | `metabolism` | Li 2021 fitted net flux for 198 reactions with confidence bounds, and Ishii 2007's 13C flux arm |

`ProductTiterPhenotype` is the one the Vision cares about most, being the phenotype side
of inverse strain design. Three choices in it are load-bearing:

- **The product is a typed `Compound`, not a product-name string.** That is what makes
  the molecule a strain MAKES the same entity another screen DOSES, so "everything about
  isoprenol" is one query across a tolerance screen and a production run
  (`test_identity.py::test_a_product_made_and_a_compound_dosed_are_one_compound_node`).
  It is also what puts `Compound` in both the environment and the phenotype lane, which
  the ontology-checks note anticipated as legitimate; `test_ontology_coherence.py` now
  names the shared class explicitly instead of asserting the lanes are disjoint.
- **The value is a plain float with a typed `titer_unit`**, so the ML-facing label stays
  scalar, and the uncertainty ontology is `FitnessPhenotype`'s exactly (reported number
  plus its `UncertaintyType`, `n_samples` plus `SampleUnit`, derived `titer_se`). Yield
  and productivity are optional and each is paired with its own unit enum, since neither
  is a concentration; a value without its unit is refused.
- **The fermentation context is NOT on the phenotype.** Medium, carbon source, oxygen
  regime, vessel and duration are all `Environment` / `CultureEnvironment` slots already,
  and putting them here would make "what was measured" carry part of "what it was grown
  in".

`FluxPhenotype` sets `label_statistic_name = None` on purpose: a fitted flux's
uncertainty is a two-sided interval, and naming one bound as "the" statistic would
misreport it. `net_flux` is signed and nothing is clamped, because the sign is the
direction.

Eight further pairs exist for ONE reason -- to carry the assembly pin for a phenotype
family that is **reused unchanged**: `bacterial_fitness`,
`bacterial_environment_response`, `bacterial_gene_interaction`,
`bacterial_gene_essentiality`, `bacterial_protein_abundance`, `bacterial_metabolite`,
`bacterial_rnaseq_expression`, `bacterial_visual_score`. The existing yeast pairs cannot
be narrowed instead: their `genome_reference` admits a plain `ReferenceGenome` and all 51
served yeast records store one, so narrowing the shared class would invalidate every one
of them. All eleven pairs keep `environment: Environment`, so the cross-dataset
medium-level aggregate still forms across hosts.

**A finding about union resolution, worth knowing before writing a loader.**
`experiment_type` is a plain `str` on every experiment class, not a `Literal`, so
`ExperimentType` is an undiscriminated union. An assembly-pinned family whose phenotype is
reused has the same field set on the experiment side as its yeast sibling, so
`TypeAdapter(ExperimentType)` may return the sibling class (the values and the tag all
survive). This is not new behavior and not a defect introduced here --
`StrainEnvironmentResponseExperiment` resolves correctly only because it adds fields --
and it is exactly why every reconstruction path in the codebase keys on
`EXPERIMENT_TYPE_MAP[experiment_type]`. A family with a NEW phenotype class
(`ProductTiterExperiment`) is unambiguous either way. Both behaviors are pinned in
`test_schema.py` so the difference stays visible.

### Test surface added

In `tests/torchcell/datamodels/`:

- `test_schema.py`: the identifier families stay separate (a yeast leaf refuses `b0002`,
  the bacterial leaf accepts it, and the reverse), namespace-to-tag agreement, pattern
  disjointness, the KT2440 RNA tags, a `Genotype` round trip per leaf and all five at
  once, the assembly pin's rejections and the serialization caveat, the registry-id
  agreement, the BW25113 verbatim genotype (data-gated), the background's invariants, and
  the three phenotypes' validators and derivations.
- `test_ontology_invariants.py`: the identity regex gains the three bacterial patterns
  plus a test that the three identifier families are disjoint; the leaf factory covers
  the five new leaves with REAL tags of each host.
- `test_ontology_all_trees.py`: `SO_ALLOWED` gains the two verified SO pairs; the M2
  dosage sweep gains `HeterologousPathwayPerturbation`.
- `test_ontology_coherence.py`: `AssemblyReferenceGenome` named as the third sanctioned
  genome narrowing; lane disjointness becomes "shares only named identity classes" with
  a guard that a lane root can never be listed as shared; the phenotype-to-node-class
  bijection names the three classes awaiting step 6's graph classes.
- `test_identity.py`: the product of a titer joins the compound layer; a measurement is
  never part of an identity.
- `test_datamodels_roundtrip.py`: `EXAMPLES` for the classes whose validators refuse the
  generic builder's placeholders. `HeterologousPathwayPerturbation` deliberately needs no
  entry, which confirms its relaxed validator behaves as designed.

### The measurement that proves the step stayed additive

`python -m torchcell.provenance.schema_impact --base origin/main`:

```
Changed symbols (35):   every one (added)
Impacted datasets: none (no built loader depends on the changed symbols).
exit 0
```

**35 added symbols, 0 modified, 0 impacted datasets.** Step 4 reaches no served dataset
at all.

Measured twice, and the difference is worth recording. Before this branch was rebased,
the base was the pre-step-2 commit and the report read *35 added, 1 modified, 4 impacted,
0 breaking*: the modified symbol was `BackgroundAllele` and the four datasets
(Hillenmeyer HET and HOM, Hoepfner, Vanacloig, Wildenhain) were the ones step 2's
`SYSTEMATIC_GENE_PATTERN` to `SGD_SYSTEMATIC_GENE_PATTERN` rename reaches. That rename
has since landed on `main`, so it is part of the base and the only remaining delta is
this step's own additions. Either way the conclusion is the same and is the one the step
was designed for: **nothing authored here moves a served dataset's closure.**

Verification runs, all green on the rebased tree: `pytest tests/torchcell/datamodels -x`
(1,177 passed with the figure tests, 2 skipped, 8 xfailed) and `pytest
tests/torchcell/sequence tests/torchcell/datamodels tests/torchcell/verification -x`
(1,750 passed, 3 skipped, 8 xfailed). The data-gated BW25113 test passes under
`--data` with the real `DATA_ROOT`.

### One figure consequence

The schematic in `notes/assets/images/schema-ontology/` is regenerated from the schema by
a pre-commit hook, and the eleven new experiment pairs pushed its EXPERIMENT box over its
line budget for the first time. The generator's own overflow notice ("+N more lines --
full list in the interactive map") was wider than the box whose overflow it announces, so
`_truncate` clipped it mid-word. Shortened in `torchcell/paper/ontology_svg.py` to fit,
with a test that measures the notice against the real box width rather than re-checking a
render by eye; the figure footer already carries the explorer URL, so the notice does not
need to repeat it.
