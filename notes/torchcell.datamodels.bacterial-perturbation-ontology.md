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

## 2026.10.07 - Follow-ups from the first loaders

Source: `torchcell/datamodels/schema.py`, `torchcell/datamodels/identity.py`,
`biocypher/config/torchcell_schema_config.yaml`, `torchcell/adapters/cell_adapter.py`,
`torchcell/verification/{common,fitness,rnaseq,runners}.py`,
`torchcell/literature/retrieve.py`.
Branch: `feat/bacterial-schema-followups`. The needs are the ones the seven landed
loaders stated: [[torchcell.datasets.ecoli.mutalik2020]] (a typed phage),
[[torchcell.datasets.pputida.carruthers2025]] (the titer's vessel and its product's
identity), [[torchcell.datasets.ecoli.tong2020]] (two verifier rules),
[[torchcell.datasets.ecoli.fuhrer2017]] and [[torchcell.datasets.ecoli.wetmore2015]] (a
derived-mapping field), [[torchcell.datasets.ecoli.lamoureux2023]] and
[[torchcell.datasets.pputida.lim2022]] (a replicate-aware RNA-seq verifier, the Zenodo
User-Agent).

### `PhagePerturbation`, the fourth environment leaf

`EnvironmentPerturbationType` gains `PhagePerturbation`. A virion is none of the other
three: a small molecule is keyed by InChIKey, `PhysicalFactor` is a scalar variable
(pH, osmolarity, carbon source, nitrogen source, ionic strength, nutrient dropout,
radiation), and `BiologicAgentClass` is peptide / protein / antibody / toxin, which
would call a replicating nucleoprotein particle a proteinaceous agent. Identity is the
phage itself: `name` verbatim, plus `ncbi_taxid` and `genome_accession` when stated.

**The dose is not a `Concentration`, and that is the load-bearing decision.** A
multiplicity of infection is a dimensionless ratio of two counts, so
`multiplicity_of_infection` is a float field of its own and `titer_pfu_per_ml` carries
the culture concentration; adding an `moi` member to `ConcentrationUnit` would make a
particle-to-cell ratio look like a dose unit (and would move every served closure, see
the measurement below). A phage challenge always has a dose, so an unstated MOI is a
typed `ProvenanceGap`, never a silent `None` (`_require_value_or_gap`, the strain-
background contract's rule).

The identity projection (`PHAGE_PERTURBATION_IDENTITY_FIELDS`) is the discriminator, the
folded name, the taxon, the accession and the two dose numbers. `family`, `genome_type`
and `host_of_propagation` are deliberately OUT: two sources may classify or propagate
one phage differently, and including them would split a node that should join. The dose
IS in, which is what keeps Mutalik's 68 challenges of 14 phages from collapsing onto one
environment identity.

### The graph side: a sibling of `environment perturbation`, under `biotic exposure`

`phage perturbation` is additive and is NOT a child of `environment perturbation` (a
child would carry that served label, so every served `MATCH (:EnvironmentPerturbation)`
would start returning phage nodes). Its `is_a` is Biolink's `biotic exposure`, whose own
definition names viruses -- "An external biotic exposure is an intake of (sometimes
pathological) biological organisms (including viruses)", altLabel `viral exposure` --
while `environmental exposure` is defined as ABIOTIC ("a factor relating to abiotic
processes in the environment"). The two are siblings under `exposure event`, so
`MATCH (:ExposureEvent)` still reaches both. Measured in-process on the pinned Biolink
3.2.1 mirror: `phage perturbation -> biotic exposure -> attribute -> named thing ->
entity`, and no path to `environment perturbation` in either direction.

`environment perturbation member of` gains `phage perturbation` as a second source, so
it needs `input_label` (with a list source and no `input_label`,
`OntologyMapping._horizontal_inheritance_source` zips the list against a string and
builds one virtual leaf per CHARACTER -- the same trap `perturbation member of`
documented). The adapter's relationship label is unchanged, so the written type stays
`EnvironmentPerturbationMemberOf`.

`CellAdapter._phage_perturbation_node` emits only `PhagePerturbation` leaves, with the
same composition-projection id every environment-side node uses, so
`_environment_perturbation_to_environment_edges` addresses it unchanged. The served
`_environment_perturbation_node` is NOT touched, which is what keeps the step additive.

**The consequence to settle before the first phage dataset is served**, stated rather
than discovered later: the served method emits EVERY perturbation of an environment, a
phage included, under the `environment perturbation` label, so a conf that enables both
classes writes one content id under two classes and the import keeps whichever row it
reads first. `test_no_conf_enables_both_environment_perturbation_classes` fails such a
conf. The cost is that a phage-only conf would not write the assay's other environment
perturbations (a Bar-seq kanamycin, a physical factor). Resolving that needs either a
filter in the served method (adapter drift on served datasets, so a full rebuild) or a
separate id space for the phage node. No conf is in that position yet: Mutalik's loader
has no dataset class.

### `DerivedIdentifierMapping` on the five perturbation leaves

Plan section 4 requires a crosswalk use to be "recorded on the record as a derived
mapping, never silently", and three landed loaders had nowhere to put it: Fuhrer dropped
17 rescuable JW ids for the lack of it, Wetmore's subsumption record names the ECK route
as the one its compendium loader will take, and Tong crosswalked 3,644 Keio b-numbers to
BW25113 tags with the map only in `preprocess/identifier_reconciliation.json`.

The field is `identifier_mapping: DerivedIdentifierMapping | None = None` on all five
leaves (`source_identifier` verbatim plus a `route` of `eck_crosswalk | jw_synonym |
gene_symbol`), so every record built before it existed reads exactly as before. Three
refusals keep it honest:

- a route must match the identifier it starts from (an `eck_crosswalk` starts from
  another strain's locus tag, a `jw_synonym` from a Keio `JW` id, a `gene_symbol` from
  something that is not a locus tag at all);
- a tag cannot be derived from itself;
- an `eck_crosswalk` cannot start from a tag of the record's OWN namespace, since the
  join exists to cross namespaces.

`JW_IDENTIFIER_PATTERN` is `^JW[RS]?\d{4}$`, measured on
`GCA_000750555.1_ASM75055v1_genomic.gbff.gz`: 4,181 `JW` plus four digits, 153 `JWR`
(RNA genes) and one `JWS`, nothing else. A `JW\d{4}`-only pattern would have refused the
154 structural-RNA synonyms.

### `ProductTiterExperiment` narrows its environment to `CultureEnvironment`

A titer cannot be read without its vessel: a 1.5 mL flower plate at 1000 RPM and a
shake flask are different fermentations. `ProductTiterExperiment.environment` and
`ProductTiterExperimentReference.environment_reference` are therefore
`CultureEnvironment`, which is the same narrowing the chemogenomic family already makes
and the only way the fields survive -- pydantic v2 serializes by the DECLARED type, so a
`CultureEnvironment` in an `Environment`-typed slot keeps `culture_format` as a python
attribute and dumps without it, with no error anywhere. Both halves are pinned
(`test_a_culture_environment_survives_the_product_titer_slot`). The medium-level
cross-dataset aggregate is unchanged: `media` is the same field holding the same `Media`,
and a culture environment stating no protocol has the plain environment's identity.

This is the one BREAKING symbol change on the branch, and it reaches no served dataset:
`ProductTiterExperiment` is imported by no landed loader (Carruthers 2025 is PR #720,
still open).

### The two enum additions, measured and DEFERRED

Both were asked for; both were measured to move served closures, so neither landed.

| addition | impacted datasets | why it is deferred |
|---|---|---|
| `ConcentrationUnit.mg_per_l` | **41** (39 served yeast loaders plus the three bacterial dev builds) | `ConcentrationUnit` is in every served dataset's closure through `Concentration` and `Media`. Carruthers stores the identical `ug/mL` meanwhile (1 mg/L IS 1 ug/mL exactly), so nothing is lost. |
| `MeasurementType.normalized_colony_size` | **13 served** environment-response loaders | same shape. Tong 2020 uses `FitnessPhenotype` meanwhile, which its own PR argues for independently (each carbon source is normalized on its own plates, so the baseline is 1 by construction). |

Added-member changes classify as `stale`, not `breaking`: no stored record becomes
invalid, but every affected dataset's build manifest reads stale, which is the
rebuild signal. They belong in the next deliberate full rebuild, together.

### The measurement

`python -m torchcell.provenance.schema_impact --base origin/main`:

```text
Changed symbols (9):
  [stale]    BacterialCrisprInterferencePerturbation  added optional field 'identifier_mapping'
  [stale]    BacterialDeletionPerturbation            added optional field 'identifier_mapping'
  [stale]    DerivedIdentifierMapping                 new symbol
  [stale]    HeterologousPathwayPerturbation          added optional field 'identifier_mapping'
  [stale]    PhagePerturbation                        new symbol
  [BREAKING] ProductTiterExperiment                   added required field 'environment'
  [BREAKING] ProductTiterExperimentReference          added required field 'environment_reference'
  [stale]    PromoterReplacementPerturbation          added optional field 'identifier_mapping'
  [stale]    TransposonInsertionPerturbation          added optional field 'identifier_mapping'

Impacted datasets (3; 0 breaking) -> rebuild:
  [stale] MetabolomeFuhrer2017Dataset     via BacterialDeletionPerturbation, DerivedIdentifierMapping
  [stale] RnaseqLamoureux2023Dataset      via BacterialDeletionPerturbation, DerivedIdentifierMapping
  [stale] PutidaPrecise321Lim2022Dataset  via BacterialDeletionPerturbation, DerivedIdentifierMapping
```

**Zero SERVED loaders are impacted.** The three are the bacterial dev builds, none of
which is in the served store (51 yeast datasets, KG 2.0, built at `4b293d34`); each
reads stale under `torchcell.provenance.build_manifest` until its dev LMDB is rebuilt,
which the KG build's freshness gate requires before the next full build. That rebuild is
not run here.

`kg_manifest drift`, against a scratch copy of the served manifest
(`/scratch/projects/torchcell/database/kg_manifest.json`, sha256 `662eaab2...05cd480`,
unchanged after the run):

```text
  graph schema CHANGED: crispr construct
  graph schema ADDED: bacterial perturbation, flux phenotype, phage perturbation, product titer phenotype, protein turnover phenotype
  served edge classes widened (additive): crispr construct member of: +bacterial perturbation; environment perturbation member of: +phage perturbation; perturbation member of: +bacterial perturbation; phenotype member of: +flux phenotype, +product titer phenotype, +protein turnover phenotype
  adapter drift touching served datasets: plumbing: _crispr_construct_node_from
  adapter methods ADDED: ... _get_phage_perturbation_reference_nodes, _phage_perturbation_node, _phage_perturbation_node_from ...
  value surface: CHANGED: compound_identity.py, compound_identity_table.json, media.py
```

The delta against the previous round is `phage perturbation`, its edge widening and its
three methods. The one CHANGED class and the one plumbing drift are pre-existing (main
`caaf2295f`'s `crispr construct` property rename). The value-surface entries are the
compound row and the media keys below.

### Verification follow-ups (plan section 5 item 8)

Four changes, each named by a loader note, plus the registry entries:

- **`verify_rnaseq_dataset(replicate_aware=True)`** swaps `strain_uniqueness` for
  `replicate_groups`. The strain rule cannot be satisfied by a compendium whose rows are
  sequenced LIBRARIES: replicates of one condition share both genotype and environment,
  and a wild-type record has no perturbation to hold a `strain_id`. The replacement
  checks what such records can get wrong -- no two records share an expression profile
  (a library counted twice) and every (genotype, environment) group measures one gene set
  -- and reports the group-size histogram.
- **`fitness._environment_signature` now includes the environment's typed EDITS**, keyed
  by the same `(perturbation_type, compound or agent name, factor, dose value, unit,
  basis)` tuple the environment-response verifier uses. Tong 2020 grows every strain on
  one medium at one temperature and varies only the carbon source, so each of its 3,644
  Keio strains read as thirty duplicates. A yeast fitness dataset carries no environment
  perturbations, so its key gains an empty tuple and is unchanged.
- **`common._gene_name_result` accepts a pseudogene locus that resolves to ITSELF**
  (`non_gene_feature` with `systematic_name == systematic`), counted and reported in the
  result's details. A bacterial annotation's `gene` features exclude `/pseudo` loci by
  construction, so a pseudogene can never come back `current`, yet the Keio collection
  and the sRNA library deleted 108 BW25113 and 1 MG1655 pseudogene loci whose stored
  identifier is exactly right. A name that resolves to ANOTHER locus, or to none, still
  fails.
- **`run_metabolite`, `run_rnaseq` and `run_fitness` select the L4 universe (and the
  resolver) from each dataset's OWN records**, through `_dataset_assembly_sets` /
  `_dataset_gene_universe` and the new `_l4_assembly_gene_containment`. A bacterial
  dataset's genes are neither Ohya deletion-collection ORFs nor S288C genes, so the
  yeast rules would have failed it by construction while saying nothing about its
  identifiers. `run_fitness` refuses a dataset pinned to TWO assemblies, since one
  resolver cannot serve two hosts: that is Tong 2020, which runs its own per-background
  verification in its loader module.

Registered and run against the dev LMDBs under `$DATA_ROOT/data/torchcell/` (read-only;
the reports are written to each dataset's own writable `preprocess/`):

| dataset | runner | records | verdict |
|---|---|---|---|
| `metabolome_fuhrer2017` | `run_metabolite` | 3,735 | **PASS** L0-L4; L4 1.000 of 3,735 deleted loci are `ecoli_K12_BW25113_ASM75055v1` genes |
| `rnaseq_lamoureux2023` | `run_rnaseq` (replicate-aware) | 241 | **PASS** L0-L4; L1 241 distinct profiles over 114 groups; L4 0.999 of 4,257 measured genes are ASM584v2 loci (floor 0.99; the three retired b-numbers) |
| `putida_precise321_lim2022` | `run_rnaseq` (replicate-aware) | 180 | **PASS** L0-L4; L1 180 distinct profiles over 65 groups; L4 1.000 of 5,564 measured genes are ASM756v2 loci |

Tong 2020 is NOT registered: its class is PR #719, still open, and a registry row for a
dataset whose loader is not on `main` is a row nobody can rebuild. The two rules it
asked for are the two above, so its entry is one line when it lands.

### The Zenodo User-Agent

`retrieve._get` now picks the agent by host. Zenodo answers the shared Chrome 120 string
with HTTP **403** and a non-browser agent with 200/206 (measured 2026-10-07 by the
Lamoureux round: `torchcell-literature`, `python-httpx/0.27.0` and
`Mozilla/5.0 torchcell-literature` all 206), so PRECISE-1K's recorded `zip_member`
retrieval did not re-run. `PLAIN_UA_HOSTS = {"zenodo.org"}` now gets
`torchcell-literature (+https://github.com/Mjvolk3/torchcell)`, matched on a dotted
boundary (so `sandbox.zenodo.org` is covered and `notzenodo.org` is not). The browser
string stays the default, because the publisher CDNs the other retrievers use were
chosen against it.

## 2026.10.09 - Called bacterial variants: three leaves and one composed call (#731)

An evolved clone's genotype is its parent plus the variants a caller found in its
resequencing, and before this step no leaf could hold one of those calls. Two landed
loaders had already measured the blockage on their own bytes and counted every refused
row rather than forcing it into a class that did not fit: de Siqueira 2025 (173 Geneious
calls over 5 sequenced clones) and Lim 2025 (159 breseq rows over 49 clone columns).

### What the releases actually carry, measured

| release | rows | in a locus | no locus | spanning loci | position end | frequency |
|---|---|---|---|---|---|---|
| de Siqueira Data Set S2 (`aem.02123-24-s0002.xlsx`, sha256 `7ba609170ec7233e09e0ffbbfdb59a190353180ad48b50c45ab60850cd713633`) | 173 | 58 GenBank tag + 10 RefSeq only | 105 | 0 | released (`Maximum`) | 157 bare fractions, 16 percent RANGES |
| Lim `Fig 2B_Mutation List` (`si2.xlsx`, sha256 `a3cfd6014cc611b206af9c7e7c8770c23189ae96986d23981da0b11236721612`) | 159 | 123 | 17 intergenic | 19 multi-locus (25 counting single-locus DEL rows with an empty `Details`) | NOT released, only a start plus a length | 431 cells `1`, 12 cells `0.9` |

Four further shapes the first design would have missed, each measured:

- **A zero-length insertion interval.** 21 of de Siqueira's 173 rows are insertions
  released with `Maximum == Minimum - 1` and `Length` 0, which is the interval BETWEEN
  two reference bases. The coordinates are stored as released; `span_length` is then 0.
- **The frequency column is not one scale.** 157 de Siqueira cells are fractions in
  (0, 1] and 16 are percent RANGES (`95.1% -> 97.6%`). No float holds a range, so those
  calls carry the verbatim cell with a null numeric frequency, and the scale is a stated
  field rather than inferred from whether a value exceeds 1.
- **A RefSeq-only locus does not resolve.** The 10 RefSeq-only de Siqueira calls name
  `PP_RS21780` (9 rows) and `PP_RS19075` (1), and `resolve_gene_name` on the pinned
  KT2440 GenBank assembly returns `retired`, "not found in GCA_000007565.2_ASM756v2;
  retained as given", for both. So "no locus tag of the pinned assembly holds this call"
  has TWO reasons, not one.
- **One side of a call can legitimately be blank.** One de Siqueira insertion row has an
  empty `Sequence` cell, because an insertion replaces no reference bases.

### The classes

`BacterialVariantCall(ProvenanceGapMixin)` is COMPOSED onto all three leaves, so what a
call is is defined once and the leaves differ only in where it sits. It carries the
replicon verbatim, the 1-based end-inclusive interval, the released change and type
cells verbatim, the annotation / amino-acid / codon cells, the caller, `call_mode`
(`clone` vs `population`), and a frequency that is nullable AND gappable with its
`frequency_basis` required whenever a number is present.

| class | parent | `perturbation_type` | state | keyed on |
|---|---|---|---|---|
| `BacterialSequenceVariantPerturbation` | `BacterialVariantPerturbation` (abstract, AXIS 3) | `bacterial_sequence_variant` | n/a | a locus tag |
| `BacterialSiteVariantPerturbation` | the same | `bacterial_site_variant` | n/a | the derived site id `<replicon>:<position>` |
| `BacterialSpanDeletionPerturbation` | `BacterialDeletionPerturbation` (AXIS 1) | `bacterial_span_deletion` | `absent` | a locus tag, one per covered locus |

No new Sequence Ontology term: `BACTERIAL_VARIANT_TYPE_SO` maps the four variant kinds
onto pairs the schema already pins (SNV, insertion, deletion, `delins`). The leaves keep
the generic `SO:0001060 sequence_variant` as their class-default `mechanism_so_id`, true
of every call whatever its kind, and expose the specific pair as a property, so the
mechanism field cannot desync from `variant_type`.

### Six decisions worth recording

- **A site id, not a borrowed neighbor.** An intergenic call is keyed on
  `<replicon>:<position>`, DERIVED from the call and checked against it by a model
  validator, and the leaf REFUSES a locus tag in that field. Keying it to a flanking
  gene would assert the variant is in that gene, which is the inference both loaders
  refused to make. `GeneAdditionPerturbation` already establishes the shape: a
  perturbation whose identifier is legitimately not a host locus tag.
- **A site id is not a gene, so it is not in the gene set.**
  `ExperimentDataset.extract_systematic_gene_names` leaves a `bacterial_site_variant`
  identifier out and collects its `flanking_systematic_gene_names` instead. Without this
  a site id becomes a gene node for a place that is not a gene; measured on the de
  Siqueira proteome store, the gene set went 71 -> 30 and the loader's own L4 gene
  containment went from 41 identifiers outside the KT2440 universe to 0.
- **A partial deletion of a locus is a sequence variant, not an absence.** Lim's
  `PP_2675` row reads `coding (4-459/462 nt)`: the locus is still there. `state="absent"`
  would assert a consequence the release did not.
- **A span deletion is one perturbation PER COVERED LOCUS**, not one per event, because a
  gene-keyed consumer must see all 53 genes of `PP_3024-PP_5558` as absent and not only
  an endpoint. The event stays one query through `span_designation`, which every locus of
  one deletion carries identically, the way `pathway_name` groups a pathway's genes.
- **The frequency is part of the call, so the same change at two frequencies is two
  perturbation identities.** Deliberate: an isolate that is 90% mutant at a site is not
  genotypically the same as one that is 100% mutant. A convergence query joins on
  `systematic_gene_name` and `position_start`, never on the node id.
- **Which reference the call is against is on the call.** `reference_sequence` is
  required, because a position means nothing without it and because it is the only thing
  that exposes a call made against a genome that is not the record's own host. Niu 2019
  aligned a BW25113 derivative to MG1655, so its calls mix evolution-acquired mutations
  with strain-background differences; the disagreement between this field and the
  record's `AssemblyReferenceGenome` is what a consumer can see it by.

### The graph side: a sibling of `bacterial perturbation`

`bacterial sequence variant perturbation`, `is_a: genotype`, serves all three leaves.
Its own class rather than rows of `bacterial perturbation`, for the reason
`phage perturbation` is its own class: the replicon, the interval, the variant kind, the
frequency and the call mode are what a variant record is asked about, and they would be
null on all five of the other bacterial leaves, which is exactly the argument that keeps
`strain_id` off that class. A SIBLING, so no served `MATCH (:BacterialPerturbation)`
changes. `BACTERIAL_PERTURBATION_LEAVES` and `BACTERIAL_VARIANT_PERTURBATION_LEAVES`
PARTITION the namespaced leaves and the plain method excludes the variant tuple
explicitly: `BacterialSpanDeletionPerturbation` IS a `BacterialDeletionPerturbation`, so
an isinstance test alone would write it under both labels. The served
`perturbation member of` edge gains the class as a third source and the adapter's edge
methods are untouched, because the node id is the same sha256 of the leaf's
`model_dump` they already address.

### Schema-impact verdict

`scripts/schema_impact_check.py --base origin/main`: 13 changed symbols, **80 impacted
datasets, 0 breaking**, exit 0. Every dataset whose closure reaches `Genotype` is marked
stale because `GenePerturbationType` gained three members; nothing is breaking, and the
KG 4.0 full rebuild that follows this wave covers the rebuild.

`python -m torchcell.provenance.build_manifest` then reads **113 dev stores STALE**, all
of them naming `GenePerturbationType` as the changed symbol and nothing else (a handful
also name symbols a parallel branch already landed). That is the measured consequence of
an additive union member, and the full rebuild is what clears it. The eight stores this
step touched were rebuilt by name and read fresh: the three de Siqueira proteome classes,
its titer class, Lim's tolerance and proteome classes, and Menasalvas and Kang, whose
loaders did not change but whose closures moved. All eight pass their own verification.

## 2026.10.09 - Round-2 bacterial leaves and a dose that is not a concentration (#749, #792, #799)

Three more gene-perturbation leaves and one environment leaf, each because a LANDED
loader measured released rows it could not type. The block in `schema.py` is additive and
self-contained, delimited by its own comment banner; nothing above it is edited.

| class | axis | mechanism SO | the measurement that asked for it |
|---|---|---|---|
| `BacterialMarkedAllelePerturbation` | gene presence/absence | `SO:0001218 transgenic_insertion` | Shiver 2016 dropped 121 of the 134 non-deletion columns of its array (114 `-SPA`, 7 `-kan`) (#749), and Babu 2014 dropped 3,420 of 42,705 released interaction pairs whose donor or recipient is one of Butland 2008's kan-marked, SPA-tagged essential-gene strains (#792) |
| `BacterialDegronPerturbation` | gene expression | `SO:0001218` with the proteolysis fields | Shiver's 5 `-DAS` / `-DAS+4` columns, whose construct asserts regulated PROTEOLYSIS conditional on an adaptor, which a marked allele does not assert (#749) |
| `BacterialCrisprActivationPerturbation` | gene expression | inherited from `CrisprActivationPerturbation` | Niu 2019's 57-target activation arm, four fifths of that release's 94 numeric cells (#799) |
| `PhysicalExposurePerturbation` | environment | n/a | Shiver's UV conditions, released as `UV [12 sec] {4}` with no irradiance anywhere in the paper or its SI (#749 item 3) |

### Why ONE marked-allele leaf serves two issues

#749 and #792 describe the same physical construct: a selectable cassette integrated at a
gene's terminus with the open reading frame intact, with or without an in-frame tag.
Splitting them by issue would file one physical strain collection under two class names.
The leaf's optional `cassette`, `insertion_site`, `tag` and `terminus` let a release say
as much as it says, and `allele_effect` is a required three-valued field
(`hypomorphic | unaffected | not_stated`) rather than a bool: a gene-perturbation leaf
cannot carry a `ProvenanceGap`, so without an explicit `not_stated` member a silent
release would have to be written as `False`, which asserts the allele is NOT hypomorphic.
Babu 2014 stores `hypomorphic` on the strength of its own sentence; Shiver stores
`not_stated`, because its array defers to Nichols 2011, which has no mirror entry (#691).

### Why a CRISPRa leaf rather than reusing what exists

Measured in
`experiments/036-dataset-fixes-before-kg-build/results/niu2019_release_loadability.json`:
`CrisprActivationPerturbation` inherits the R64 ORF validator and refuses both `b3417`
and `dxs`; of the five bacterial leaves that existed, the only one on the expression axis
with a direction was `BacterialCrisprInterferencePerturbation`, whose
`expression_direction` is fixed to `decreased`. `PromoterReplacementPerturbation` can
state `increased` but asserts `SO:1000032 delins`, a native promoter removed and a
characterized part put in its place, which a guide-directed activator never does. So the
new leaf subclasses `CrisprActivationPerturbation` exactly as the interference leaf
subclasses `CrisprInterferencePerturbation`: a required `gene_namespace`, the bacterial
locus-tag validator overriding the R64 one, an optional `identifier_mapping`, and the
inherited `crispr` construct, so the guide payload stays defined once.

### Why an exposure dose is not a `ConcentrationUnit` member

`EnvironmentPhysicalPerturbation` carries a scalar factor whose magnitude is a
`Concentration`, which works for pH and osmolarity and fails for an irradiation: a UV
dose is an irradiance times a time, and Shiver releases only the time. Adding `sec` or
`J/m2` to `ConcentrationUnit` would make an exposure time look like a concentration for
every other dataset. `PhagePerturbation` already set the precedent, giving the
multiplicity of infection its own named field, so `PhysicalExposurePerturbation` gives the
dose three: `exposure_duration_seconds`, `irradiance_w_per_m2`, `fluence_j_per_m2`. They
are not redundant, because a release may state the time alone (Shiver), the fluence alone,
or the irradiance and the time. No cross-field arithmetic is asserted: an irradiance a
source reports may be nominal or time-averaged, and multiplying it would manufacture a
fluence the source never released. A model validator requires one of the three to be
stated or gapped, so an exposure never silently has no dose, and `factor` reuses the
`PhysicalFactor` vocabulary so a radiation exposure and a qualitative
`EnvironmentPhysicalPerturbation(factor=radiation)` name the same variable.

### Graph side

The three gene leaves join `cell_adapter.BACTERIAL_PERTURBATION_LEAVES` (5 -> 8). Each
projects the same five properties the `bacterial perturbation` graph class declares;
their leaf-specific fields (`cassette`, `tag`, `degron`, `insertion_site`, `collection`)
carry no node property, as `BacterialDeletionPerturbation.collection` does not, and
travel in the serialized record. So no graph class changed, which is what keeps this
additive. `PhysicalExposurePerturbation` is served by the existing
`environment perturbation` lane, which since #756 partitions the environment's
perturbations with the phage lane.

### Schema impact

`python scripts/schema_impact_check.py --base origin/main`: **6 changed symbols, 81
impacted datasets, 0 breaking.** The four new classes are new symbols; the two modified
symbols are `GenePerturbationType` and `EnvironmentPerturbationType`, which gained union
members. Every impacted dataset is stale through those two unions rather than through a
changed field, which is the measured consequence of an additive union member and is what
the KG 4.0 full rebuild clears. The three stores this step's loaders touched were rebuilt
by name and read fresh: Shiver 2016 (204,033 -> 210,998), Babu 2014 (38,579 -> 41,988),
and the new Niu 2019 (51). All three pass L0 to L4.

### Still open, named here so it is not lost

`MeasurementType` has no `fold_change` member, and two things are blocked on it: the
released sample SD of a growth RATIO (an SD does not transform with its statistic, so
`log2_ratio` cannot carry it), and every dimensionless product ratio, of which Niu 2019
releases 40. Issue #770 owns it, and it is proposed rather than taken in this branch.

## 2026.10.09 - The protein fold-change family, and five fields two landed loaders asked for

Issues #770 and #753, both on the protein phenotype side of `schema.py`. Every addition is
additive and sits in a block delimited `# --- begin #770 ... # --- end #770 ---` or the
same with `#753`, so parallel branches rebase cleanly.

### #770, a relative axis beside the absolute one

`ProteinAbundancePhenotype`'s docstring forbids a ratio in capitals ("absolute per-strain
quantity on a log signal scale, NOT a ratio"), and the P. putida supplementary-data audit
found five papers releasing a protein-level fold change with a significance value beside
it, plus a landed loader (Yunus 2026) storing a ratio under that class anyway. The fix is
option 1 of the three the issue lists: a sibling class, not a discriminator on the
absolute one, because an absolute level and a ratio answer different questions and the
contract that makes the absolute case unambiguous is worth keeping.

`ProteinFoldChangePhenotype` carries the ratio, `fold_change_scale`
(`linear` | `log2` | `log10`), `reference_basis` (the denominator in the source's own
terms), a per-protein SE, `protein_fold_change_p_value`, its BH-adjusted companion with
`p_value_adjustment_method` naming the correction, `n_replicates` and `measurement_type`.
Three reasons for that shape:

- **The scale is required, not inferred.** `0.5` is halved on the linear scale and a
  1.41-fold increase on log2, so without the field a linear and a log2 column of the same
  contrast average into nothing.
- **The reference is the neutral value by definition**, which is why
  `FoldChangeScale.neutral_value` and `ProteinFoldChangePhenotype.neutral_reference()`
  exist: experiment over reference reproduces the released number exactly and nothing is
  imputed. This is the rule the Yunus loader already followed by hand with a `1.0`
  reference; now it is on the class.
- **The basis is per column, not per paper.** Carruthers 2025's Source Data carries
  `POI:Control` beside `dCas9:Control` on one strain, so the pair (scale, basis) is what
  makes two columns comparable.

`gene_interaction_p_value` was the only p-value anywhere in `schema.py` before this, which
is why a fold-change table's test result had nowhere to go. Generic and bacterial
experiment/reference pairs follow the existing families
(`BacterialProteinFoldChangeExperimentReference` takes `genome_reference:
AssemblyReferenceGenome`, as the bacterial abundance pair does), and the BioCypher node
class `protein fold change phenotype` plus three `CellAdapter` methods complete the
leaf/node-class/adapter-method triple the bijection checks in
`torchcell/datamodels/ontology_checks.py` require. The lane is read from the `is_a`, so
`LANE_OF_LABEL` needs no edit.

The PRODUCT-side gap the issue's comment records (Niu 2019's 40 released pinene ratios,
which `ProductTiterPhenotype` admits only by lying about `titer_unit`) is NOT closed here:
it wants the typed `Compound`, so it is a sibling of `ProductTiterPhenotype` rather than a
member of this class, and that is a separate decision.

### #753, five fields the Gupta 2024 and Rapp 2026 loaders typed as gaps

| # | addition | what it recovers |
|---|---|---|
| 1 | `ProteinTurnoverPhenotype.degradation_rate_lower` / `_upper` + `confidence_level` + `interval_method` | a published interval stays an interval. `FluxPhenotype`'s triple is reused verbatim rather than inventing a second interval pattern; the validator requires `lower <= rate <= upper` per key and refuses bounds that state neither a level nor a method unless each carries a typed gap |
| 2 | `ProteinTurnoverPhenotype.censoring`, a `dict[str, Censoring]` | the right-censored cells Gupta 2024 writes to `preprocess/ceiling_cells.csv` travel with the record. `Censoring` has `uncensored` as a member rather than spelling it by absence, so a complete oracle can state it per key and a MISSING key means the source does not say |
| 3 | `Environment.dilution_rate_per_hour` | in a chemostat the dilution rate IS the controlled variable, so two cultures differing only in it are two environments. Ishii 2007 drops its wild-type dilution-rate series under `culture_not_batch` for exactly this reason |
| 4 | `DerivedIdentifierRoute` gains `uniprot_db_xref` and `locus_tag_synonym` | the primary route of any proteomics release (Gupta: 3,225 of 3,262 accessions) and a retired tag inside the pinned strain's own namespace (Rapp: 1 dropped record). Five landed loaders (`babu2014`, `butland2008`, `girgis2009`, `rapp2026`, `rapp2026_platforms`) record the second one in a drop ledger today |
| 5 | `ProvenanceGap.keys` + the per-key branch of `ProvenanceGapMixin` | a gap can sit beside a PARTIALLY populated map (Rapp's `target_metabolite_ids`: 1,077 sourced, 244 merged isobaric features with 2 to 11 candidates each). The honesty invariant gets stronger, not weaker: storing a value and declaring it missing is still refused, now key by key |

`uniprot_db_xref` outranks the symbol layer, and Gupta 2024 measured why rather than
asserting it: `sp|P0A6E9|BIOD2_ECOLI` carries the gene name `bioD`, whose symbol resolves
to `b0778` (bioD1) while its accession resolves to `b1593` (bioD2). A symbol-only route
mis-keys that protein. `DerivedIdentifierMapping.uniprot_accession()` is the one place
that reads the accession out of either a bare accession or a `sp|ACC|ENTRY` header, so no
consumer parses the string itself.

Two placements were considered and rejected. `dilution_rate_per_hour` could have gone on
`CultureEnvironment`, which exists precisely so a new field does not touch every served
dataset's closure; it is on `Environment` because the experiment classes that need it
declare `environment: Environment` and would each have to narrow their annotation, which
is the more invasive change, and because a dilution rate is part of the environment
IDENTITY in the same way `duration_generations` is rather than a protocol detail.
`ProvenanceGap.keys` could have been a second gap class, which would have left every
existing record's bytes untouched; one class won because a per-key absence is the same
concept narrowed, and KG 4.0 is a full rebuild either way.

### Schema impact, measured

`PYTHONPATH=<worktree> python scripts/schema_impact_check.py --base origin/main`:
**BREAKING, 81 impacted datasets of which 37 breaking.** 19 changed symbols. The breaking
set is the bacterial datasets, via the module-level `_check_gene_namespace` (which gained
the `locus_tag_synonym` namespace rule) together with `DerivedIdentifierMapping` and
`DerivedIdentifierRoute`. The 44 remaining are stale, not breaking, via `Environment`
(one new optional field) and `ProvenanceGapMixin` (one changed validator); that set is
every served yeast dataset, from `SmfCostanzo2016Dataset` through
`ProteomeZelezniak2018Dataset`, because `Environment` is in the schema closure of all of
them.

That blast radius is the price of putting the dilution rate on the shared `Environment`
and of relaxing the gap invariant in the shared mixin, and it is the right price for this
wave: KG 4.0 is a full rebuild, so no served store is being updated in place, and both
fields are the honest home for what the sources release.

## 2026.10.09 - The #770/#753 branch rebased onto the #731 variant leaves: every pin re-derived

The branch was rebased onto main after #830, #831 and #835 landed, so every count the
#770 leaf moves had to be re-derived from the MERGED schema rather than incremented.
What the merged state reads, each value taken from the tool that computes it and not from
arithmetic on the old pin:

| pin | before #731 | after #731 (main) | with #770 | read from |
|---|---|---|---|---|
| schema config nodes | 33 | 34 | **35** | `print_schema_mappings(compact=True)` |
| explicit under a Biolink parent | 29 | 30 | **31** | same |
| phenotypic-feature children | 16 | 17 | **18** | same |
| edges | 13 | 13 | 13 | same |
| Biolink concepts | 11 | 11 | 11 | same |
| mermaid `(nodes, edges, data lines)` | (33, 13, 51) | (34, 13, 52) | **(35, 13, 54)** | `test_real_schema_diagram` |
| mermaid line count | 147 | 150 | **154** | same |

One node adds FOUR mermaid lines, not two: its declaration, its `is_a` line, and one
data line per `phenotype member of` target, of which there are two. Incrementing the old
pin by two was wrong and the generator said so.

The Neo4j Browser stylesheet and the local-storage seed are generated, so they were
regenerated rather than merged (`python -m torchcell.database.browser_style`, 47 node
rules), and the three schema-ontology SVGs likewise through
`scripts/run-ontology-figure.sh`. Both now carry `ProteinFoldChangePhenotype` and #731's
`BacterialSequenceVariantPerturbation`, which is the check that the regeneration saw the
merged schema and not one branch's.

`ProvenanceGap.keys` is a LIST, not a tuple. A tuple dumps to a JSON array and comes back
from an LMDB as a list, so a live model and the record read from the store compared
unequal, which showed up as four reference-index pins failing in the yeast synthetic
suites (Baryshnikova 2010, Bloom 2019, Hoepfner 2014, Lian 2019) rather than anywhere
near Rapp 2026.

Two content addresses moved as a direct consequence of the new `Environment` field and
`ProvenanceGap.keys`, and were re-derived, not guessed:
`test_serialize_for_hashing_sorts_a_model_at_every_depth_like_its_dump`'s three reference
digests, and `_ENV_JSON` in `test_neo4j_query_raw.py`, which now carries
`dilution_rate_per_hour: None` for a batch culture.

Schema impact after the rebase is unchanged from the pre-rebase measurement: **BREAKING,
81 impacted datasets of which 37 breaking, 19 changed symbols**.

### The supported-query gate is expected RED, and this is why

`python -m torchcell.knowledge_graphs.supported_queries check` reports
`contract_changed` on all four supported queries (`amino_acid_betaxanthin`,
`essentiality_smf`, `expression_proteome_morphology`, `solid_growth_025`), in every case
naming `ProvenanceGapMixin` and no other symbol: its validator gained the per-key branch,
so a stored record's `ProvenanceGap` serializes with `keys` where the
`2026.10.06-4b293d34` closure has none. `Environment.dilution_rate_per_hour` produces NO
query drift, which is the measured evidence that the new field is outside the phenotype
surface closure those queries read.

That drift is a true statement about the change, and the lifecycle has exactly one
resolution for it: re-validate each query against the release the next build stamps
(`validate <id> --release <new>`), which cannot be done before KG 4.0 exists. The
pre-commit hook takes `TORCHCELL_QUERY_DRIFT_ACK=1` for a deliberate revision; the CI job
`query-drift` has no acknowledgment path and fails a pull request on any drift the base
branch does not already carry. So this branch lands with `query-drift` red by
construction. Deprecating the four queries to make it green would be false: they are
supported, and they return the same records.

## 2026.10.10 - A third reference baseline: a ratio readout whose control is 1.0

Hawkins 2020's mismatch-CRISPRi relative fitness (strain doublings divided by the same
run's wild-type doublings) is 1.0 at its control, not 0, and the paper states it:
"Strains with a relative fitness of 1 grow as well as the wild-type does; lower values
imply slower growth." The environment-response verifier had two reference branches and
neither fit. The numeric `reference_zero` branch requires the reference to be identically
0, which is what a DIFFERENCE-scaled relative readout (`log2_ratio`, `z_score`,
`differential_fitness`, `control_regression_residual`) is at its control. The
`reference_centered=False` absolute branch refuses any record whose `measurement_type` is
not in `ABSOLUTE_MEASUREMENT_TYPES`, deliberately, and `relative_growth_rate` is not and
should not be a member: it has a control in its own units.

Added, additively, with the same gating shape as the #776 absolute relief:

- `RATIO_MEASUREMENT_TYPES = frozenset({MeasurementType.relative_growth_rate})` and
  `RATIO_REFERENCE_VALUE = 1.0` in `torchcell/datamodels/schema.py`, beside
  `ABSOLUTE_MEASUREMENT_TYPES`. The two sets are disjoint and no difference-scaled type
  is in either.
- `reference_unit_scaled: bool = False` on both
  `verify_environment_response_dataset` and
  `verify_environment_response_dataset_streaming`. When set, the `reference_zero` row
  becomes the ratio rule: every reference is present, finite and exactly 1.0, every
  reference is on the same measurement scale as its experiment, and every record's
  `measurement_type` is a member of `RATIO_MEASUREMENT_TYPES`. Asking for the relief on a
  log2-ratio dataset FAILS (measured in
  `test_ratio_branch_refuses_a_difference_scaled_measurement_type`) rather than replacing
  a zero check with a one check.

The two existing branches are untouched, so every served dataset's report keeps its rows
verbatim. The precedent this generalizes is the private Bioscreen loader, which had to
build its own report with a hand-written `l3_reference_one` rule because the shared
verifier could not express the baseline; a ratio dataset can now take the shared verifier
whole. Schema impact: both new symbols are new module-level bindings, impacted datasets
none.

First consumer: `[[torchcell.datasets.ecoli.hawkins2020]]` (24,149 records, the ratio row
PASSes with `rule: ratio_reference`).

## 2026.10.10 - The mRNA number-fraction family: a count-less transcriptome (#854)

`MrnaNumberFractionPhenotype` joins the expression families beside
`RNASeqExpressionPhenotype`, with `MrnaNumberFractionExperiment` and an
`AssemblyReferenceGenome`-pinned reference, served as the `mrna number fraction
phenotype` node class under `phenotypic feature`. Fields: `mrna_number_fraction` (per
gene, in [0, 1], a key subset whose sum may not exceed 1 +
`MRNA_NUMBER_FRACTION_SUM_ATOL`), `n_libraries`, and a required `measurement_type`.
First consumer: [[torchcell.datasets.ecoli.balakrishnan2022]].

Why not an optional `expression_count` on the RNA-seq leaf: that would change the schema
closure of four served transcriptomes (Caudal 2024, Caglar 2017, Lamoureux 2023, Lim
2022) for one release without counts, and a `psi x 1e6` written into `expression_tpm`
would claim a length-normalized count pipeline the release does not document. The
sibling is additive: `schema_impact_check --base origin/main` reports 0 breaking, with
the usual stale-via-`ExperimentType` set. The ontology moves to 37 nodes and 58 mermaid
data lines; the RNA-seq verifier gains a `number_fraction` branch (L2 values in [0, 1],
L3 `fraction_sum_at_most_one`).

## 2026.10.10 - A protein synthesis-rate leaf for releases with no degradation rate (#857)

`ProteinSynthesisRatePhenotype` with its `ProteinSynthesisRateExperiment` / `...Reference` pair (pinning `AssemblyReferenceGenome`), the `SynthesisRateUnit` enum, and the `protein synthesis rate phenotype` graph class under `phenotypic feature`. First consumer: Li 2014 ribosome-profiling synthesis rates, [[torchcell.datasets.ecoli.li2014]].

- Why a sibling and not an optional `ProteinTurnoverPhenotype.degradation_rate`: the turnover class's primary label is the degradation rate, and an optional one would make `label_name` switch per record, so the lane would mean two things.
- `rate_unit` types the time basis. A rate per generation requires `generation_time_minutes`, since only the doubling time converts it to a rate per hour.
- Schema impact vs `origin/main`: 0 breaking; nine datasets stale only through the grown unions, and their stored records re-serialize identically (0 of 622 records differ, measured in [[experiments.036-dataset-fixes-before-kg-build.scripts.li2014_synthesis_rate_checks]]).
