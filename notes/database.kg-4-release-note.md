---
id: xhezrgjjzd5wcdr5hepodmi
title: Kg 4 Release Note
desc: ''
updated: 1792084483185
created: 1792084483185
---

## 2026.10.15 - KG 4.0 release note, prepared before the build

**Status: nothing described here is served yet.** This is the note the KG 4.0 release
ships with, written from the branch that will be built (`ontology-unify`, at
`b5c5549d8` when this was written, rebased on `main` at `d9eefcbd7`) and from the
measurements lanes A to U recorded while that branch was made
([[scratch.2026.10.15.ontology-unification-decision]],
[[database.kg-4-pre-build-verification]]). Lanes Q, M, R, T and U are integrated and the
branch head is `21456d696`; the numbers below are that head's, not the per-lane ones.
Lanes V (the remaining content-id collisions), S (the generation profile) and W (the
environment-identity verdict) are still out. The release id, the node and relationship
totals and the final per-dataset counts are filled in by the build; every other number
below is measured and cited. The store being served today is KG 3.0
(`2026.10.06-4b293d34`), paired with package `v1.6.2`.

The one thing a current user must read is **The KG 3.0 defect a 4.0 query no longer
has**, below. It is a disclosure about the store serving right now, not about 4.0.

### Terms used here

- **Served store**: the Neo4j database a client queries, built by the full knowledge
  graph (KG) build and named by a release id of the form `<YYYY.MM.DD>-<commit>`.
- **Dev store**: the per-dataset LMDB a loader writes under `$DATA_ROOT/data/torchcell/`,
  which the build reads. The 170 dev stores are rebuilt ("swept") before a full build.
- **Schema closure**: the set of pydantic classes a dataset's records reach, each with a
  contract fingerprint. A release snapshot records one closure per served dataset, and
  that is what decides whether a client version can read the release.
- **Content id**: a served node's id, the sha256 of the record's own `model_dump()`.
  Two records with the same content get one node, whatever produced them.
- **Interned constant**: a node that holds one large repeated payload (a whole genotype,
  an environment) once, which the records then point at, so the payload is stored once
  rather than per record.
- **Typed gap**: a `ProvenanceGap` stored in place of a value, carrying the reason and
  where the loader looked, so an absent value is countable instead of silent.
- **Assembly set**: the registry id of a deposited reference assembly
  (`sgd_S288C_R64-4-1_20230830`, `ecoli_K12_BW25113_ASM75055v1`, ...), the deposited
  reference a record's identifiers are written against.
- **Gene id namespace**: the closed vocabulary naming which host's identifier space a
  stored gene name belongs to, on every gene perturbation.
- **Lane**: the adapter method that writes one graph label's rows. A record whose leaf
  class sits in no enabled lane gets edges and no node.

### What KG 4.0 is

KG 4.0 serves **170 mapped datasets** (169 public plus 1 private,
`dataset_adapter_map` and `PRIVATE_DATASET_ADAPTER_MAP`) across *S. cerevisiae*,
*E. coli* and *P. putida*, against KG 3.0's 51 yeast datasets. It is the first store
built after the ontology unification, which removes the split that let the same physical
measurement exist as two classes depending on which host the data came from
([[database.constitution]], principle 1). Ten schema changes carry it, each with its own
record under `database/schema_changes/`, nine of them signed off by the owner on
2026-10-15 and the tenth (Q's `construction` field) adding an optional field, which needs
no sign-off. Five were not in this note's first version: Q's deletion `construction`, the
three names it listed as deliberately left alone and which lane R corrected instead, and
T's compound construct span. So KG 4.0 carries the whole unification and no bacterial
class survives into the next release.

The ten records partition the change exactly: 146 schema symbols moved against
`origin/main`, every one of them is named by exactly one record, and no record names a
symbol that did not move (measured on the integrated head).

### The ten schema changes, and what each makes true that was not true before

| record | what it makes true | symbols | served datasets | loaders |
|---|---|---:|---:|---:|
| `2026-10-15-gene-id-namespace-on-gene-perturbation.json` (P1, PR #1112) | Every gene perturbation states which host's identifier space its gene name is written in, as a required typed field rather than by the choice of class | 42 | 51 | 121 |
| `2026-10-15-released-identifier-leaf.json` (P3, PR #1113) | One class says "this is the identifier the source released"; four did, split by host, with different field names for the same fact | 15 | 18 | 34 |
| `2026-10-15-assembly-pin-on-reference-genome-host-neutral-experiment-pairs.json` (P2, PR #1117) | Every record's reference genome pins the deposited assembly set its identifiers are written against, or a typed gap; and a fitness experiment is a fitness experiment on any host | 56 | 51 | 121 |
| `2026-10-15-construct-placement-on-gene-addition.json` (P4, PR #1116) | An added gene points at real deposited sequence through a hash-carrying reference, or states in a typed gap that the source released none | 7 | 51 | 121 |
| `2026-10-15-strain-background-shared-base-eukaryotic-subclass.json` (P5, PR #1124) | One strain background and one background allele for every host, so a strain's own edits carry the same namespace field every perturbation does, and an integrated cassette is placed like any other added DNA | 11 | 51 | 121 |
| `2026-10-15-strain-construction-on-kanmx-and-released-identifier-deletions.json` (Q, PR #1120) | A deletion record states how the strain was constructed, so two strains made differently at the same locus are two records and not one | 1 | 42 | 35 |
| `2026-10-15-the-called-variant-vocabulary-and-call-object-drop-the-host-prefix.json` (R, PR #1122) | A called variant is a called variant on any host: the vocabulary and the call object are composed by three yeast loaders, so the prefix named the author and not the biology | 6 | 51 | 121 |
| `2026-10-15-morphology-phenotype-is-the-general-class-calmorph-is-the-specific-one.json` (R, PR #1122) | The general morphology class is `MorphologyPhenotype` and the specific one is `CalMorphPhenotype`, which was backwards | 3 | 0 | 1 |
| `2026-10-15-bw25113-background-genotype-leaves-the-schema-surface.json` (R, PR #1122) | A strain's verbatim background genotype is data about one deposited assembly set, so it lives in the genomes tier beside that set and not on the schema surface | 2 | 0 | 0 |
| `2026-10-15-construct-span-holds-the-parts-of-a-compound-location.json` (T, PR #1123) | A feature's location is the ordered parts it is made of plus how they compose, so an origin-wrapping or joined span is the span the source wrote and not its envelope | 3 | 51 | 121 |

The served and loader columns are the record's own tool-computed scope on the integrated
head, which is not the scope each lane measured alone, and the difference is the
unification composing: P5 puts a `ConstructPlacement` on `IntegratedCassette`, which
hangs off every `StrainReferenceGenome`, so P4's placement classes, T's span parts and
R's variant vocabulary reach every served closure (51 datasets, 121 loaders) instead of
the 3 datasets each measured on its own branch. Every one of them was already a
`full_kg_rebuild`, so the build decision does not move; the recorded radius does.

In more detail, and with what each closes:

- **P1, one namespace on `GenePerturbation`.** `gene_namespace: GeneIdNamespace` is
  required on the base class and means the host. The vocabulary was measured, not
  assumed: `experiments/database/scripts/measure_gene_id_namespaces.py` read all 51
  served dev stores and classified **130,685,237 stored identifiers**
  (130,537,788 SGD S288C systematic names in 51 of 51 stores; 79,016 Peter 2018
  pangenome ids, Caudal 2024 only; 68,433 non-host identifiers on relaxed leaves). Eight
  members, pairwise disjoint by test. Before this, 12 leaf classes declared the field and
  every yeast leaf stated no host at all, so the Carruthers 2025 native `phnW` and `phnX`
  copies were stored with no host and fell out of the served node lane.
- **P3, one released-identifier leaf.** `ReleasedIdentifierPerturbation` replaces
  `ReleasedGeneNamePerturbation`, `BacterialReleasedStrainPerturbation`,
  `BacterialReleasedCrisprPerturbation` and `BacterialReleasedTransposonPerturbation`;
  `ReleasedKind` replaces `ReleasedStrainKind` and `ReleasedGeneKind`. The host stays in
  `gene_namespace` and the class is the statement that the identifier is unresolved, so
  no value of the namespace vocabulary means "unresolved". `cassette` and `marker` were
  the same fact under two names and are one field; `released_gene_name` was a required
  copy of `systematic_gene_name` and is retired; `expression_direction` becomes a
  `crispr_activation` member of the kind vocabulary (one fact, one field).
- **P2, the assembly pin and the host-neutral experiment pairs.** `ReferenceGenome`
  gains `assembly_set` and `assembly_accession`, required with no default on a gap
  mixin, so the pin is a value or a typed gap. Before this a bacterial record pinned a
  sha256-deposited assembly and a yeast record pinned three free strings, by
  construction, because the pin lived on a subclass. `assembly_accession` stays
  **required** whenever a set is pinned, because R64 has an accession and it was
  verified rather than assumed: all 16 nuclear chromosomes of the deposited
  `S288C_reference_sequence_R64-4-1_20230830.fsa` are byte-identical to
  GCA_000146045.2. The 13 `Bacterial*Experiment` pairs and the
  `YeastProteinTurnover` pair retire: 10 into the general twin, 4 renamed with the host
  prefix dropped. `RETIRED_EXPERIMENT_TYPES` names all 14 retired tags and their
  successors, and `torchcell/data/experiment_types.py` raises
  `RetiredExperimentTypeError` naming the tag, its successor and the rebuild, instead of
  a bare `KeyError`, when a store written under an old tag is read.
- **P4, an added gene points at sequence.** `placement: ConstructPlacement | None` is
  required and gap-able on `GeneAdditionPerturbation`, and the two free strings
  (`plasmid_contig_id`, `locus_tag`) are gone. They were `None` in 26 of 26 served
  gene-addition stores, so a hand-typed parts list and a sequence-backed record were the
  same record. Of 24 call sites: **1 real placement and 12 typed gaps**, each gap
  carrying its own paper's pinned path, sha256 and section. Cachera 2023's four Btx genes
  now point at the deposited pBTX1 contig, with ARO4 K229L at contig 3368-3370 and ARO7
  G141S at 820-822, each confirmed by translating the construct's own codon.
- **P5, one strain background.** `StrainBackground` and
  `BacterialStrainBackground` become one base with a eukaryotic subclass, and
  `BackgroundAllele` and `BacterialBackgroundAllele` become one class carrying
  `gene_namespace` (the same required field P1 puts on every gene perturbation) with
  `zygosity` optional. A strain's own edits are the last pair of classes in the ontology
  that assert the same fact and are held apart by the host: the bacterial allele carries
  `gene_namespace: BacterialGeneNamespace` and no `zygosity` field at all, the yeast
  allele carries `zygosity` and no namespace, and both describe one allele a background
  carries as an edit against its own reference. The loosened field is `zygosity`, from required-or-gapped to
  optional, which needs the owner's sign-off and gets it in the record. P5 also places
  `IntegratedCassette` the way P4 places a gene addition, so a cassette integrated into
  the background points at deposited sequence or states a typed gap, and it supplies the
  identity projections both classes need: `strain_background_identity` projects
  `mating_type` and `ploidy` unconditionally today, which is wrong for a host that has
  neither, and `BACKGROUND_ALLELE_IDENTITY_FIELDS` does not include `gene_namespace`.
  The plan estimated **6 served closures** (Lian 2019, Hoepfner 2014, Vanacloig 2022,
  Wildenhain 2015, both Hillenmeyer 2008, the datasets whose records carry a
  `StrainReferenceGenome.background`) and 13 dev stores; the record's own tool-computed
  scope is 51 served datasets over 121 loaders, because `StrainReferenceGenome` is a
  member of the `genome_reference` union on every `ExperimentReference`.
- **R, the three audit flags.** The three names this note first listed as deliberately
  left alone were corrected in this rebuild instead, which is what spared a second one.
  `BacterialVariantType` becomes `VariantType` and `BacterialVariantCall` becomes
  `GenomeVariantCall`: the vocabulary is composed onto `ConstructVariantCall`, so it sits
  in 15 loader closures, three of them yeast, and `cachera2023.py` writes
  `VariantType.substitution` by name. `BacterialMorphologyPhenotype` becomes
  `MorphologyPhenotype` while `CalMorphPhenotype` KEEPS its name, because the program name
  already states what it is specific to and a non-CalMorph yeast screen belongs on the
  general class. The BW25113 background genotype and its six lesions move to the genomes
  tier beside the set id whose GenBank `/note` they are read from, which is where data
  about one deposited set belongs; that record's scope is `none`. The same lane made three
  verification rules `report` whose own messages read "a finding, never a gate"
  (`perturbed_locus_in_bw25113_annotation`, `released_b_numbers_on_mg1655_annotation`,
  `released_tags_on_kt2440_annotation`), so no record is gated on an annotation any more.
- **T, a construct span holds its parts.** `ConstructSpan` is `parts: list[SpanPart]`
  plus a composition and an orientation, with the vocabulary taken from the INSDC location
  grammar rather than invented, and `start` and `end` are properties that serve the simple
  case and RAISE on a compound span, so no reader silently receives an envelope. All 17
  Cachera maps now index (6 to 17 contigs, 208 to 570 features), and every one of the 570
  features regenerates its own file's location string byte for byte from the stored parts.
  The scale of the old fallback is what it closes: `pBTX002`'s `pUC ori` was read as the
  whole 12,096 bp plasmid and is 666 bp.
- **Q, a deletion states its construction.** `construction: StrainConstruction | None` on
  `KanMxDeletionPerturbation` and `ReleasedIdentifierPerturbation`, so Ozaydin 2013's 31
  pairs of byte-identical records are 31 pairs of distinct records rather than 31 nodes
  lost to content-id dedup. A scan of 176 dev stores and 30,543,838 records found the
  collision in that one store ([[experiments.database.scripts.content_id_collision_scan]]).

The records chain into one coherent change: on the integrated head of all ten,
`tests/torchcell/datamodels/test_schema_change_records.py` and
`scripts/check_schema_change_records.py` both pass **without**
`TORCHCELL_SCHEMA_RECORD_ACK`, with each of the 146 moved symbols owned by exactly one
record.

### Two bacterial graph classes are retired, and what replaces them

| retired graph class (KG 3.0) | served label | replaced by |
|---|---|---|
| `bacterial perturbation` | `BacterialPerturbation` | `perturbation` (`Perturbation`), which now carries every member of the perturbation union whatever the host, plus a mandatory `host_genome` property |
| `bacterial sequence variant perturbation` | `BacterialSequenceVariantPerturbation` | `sequence variant perturbation` (`SequenceVariantPerturbation`), which is now `is_a: perturbation` instead of a sibling under `genotype`, and carries `host_genome` in place of `gene_namespace` |

Neither retired label appears in any registered supported query (measured, see below),
and neither was ever served: KG 3.0 holds 51 yeast datasets and no bacterial dataset, so
no served row carries either label today. The retirement matters for client code and for
hand-written Cypher that was written against the schema rather than against the store.

What replaces them is a value join the unification exists for. `perturbation.host_genome`
is the deposited assembly set the record's identifiers are written against, in the same
vocabulary as `genome.assembly_set`, so
`perturbation.host_genome = genome.assembly_set` resolves with no hop through
genotype, experiment and experiment reference. Lane P verified it on real data: on every
perturbation row of all nine verified stores, across three organisms, `host_genome`
equals the `assembly_set` of that adapter's genome node. It could not hold for yeast
while the pin lived on a bacterial subclass, and it does now.

Two more graph-side changes a reader should know: `genome` gains `assembly_set` (the
value the join is on, so BY4741, BY4742 and S288C are three genome nodes all pinning the
one S288C set), and an optional `perturbation of genome` edge is declared and **off by
default**, as the traversal form of the same join; it is on the order of
`PerturbationMemberOf` (47.3 M rows in the job 4901 partial) and is measured on the
first build before it is enabled.

### Every served node id re-keys: content ids are not comparable with KG 3.0

A perturbation node id is `sha256(json.dumps(model_dump()))`, and a genotype id inlines
its perturbations while an experiment id inlines its genotype. P1 adds two fields to
`GenePerturbation` and P2 adds two to `ReferenceGenome`, so **every perturbation,
genotype, experiment, experiment reference, genome and reference node of every served
dataset gets a new id**, including where the new value is a class default. P5 moves the
strain background the same way, so the genotype and experiment ids of the six datasets
whose records carry a background re-key for a second reason.

Measured, not asserted: of KG 3.0's **23,650 served experiment ids** in lane P's sample,
**0 appear in the new CSV output** (lane P, nine stores). Interned-constant ids move too,
for a separate reason given in the next section.

What this means in practice:

- Do not join a KG 3.0 id to a KG 4.0 id, in a notebook, a cached file or a saved query
  result. The id spaces are disjoint by construction, and a join across them silently
  returns nothing rather than failing.
- Incremental admission by content id is impossible across this change. That is why
  every record computes `full_kg_rebuild` and every one of the pull requests carries the
  `schema-breaking` label.
- Nothing reachable in 3.0 became unreachable. Lane P compared 3.0 and 4.0 **by record
  content** rather than by id on the six of nine sampled datasets that 3.0 serves: 0
  records lost.
- A dev store built under a retired experiment-type tag is **unreadable** by
  `torchcell.data.aggregate` and `torchcell.data.deduplicate` until it is rebuilt, not
  merely stale. That is what `RetiredExperimentTypeError` says when it happens.

### The KG 3.0 defect a 4.0 query no longer has

**This is a disclosure about the store serving today, 2026.10.06-4b293d34 (KG 3.0). If
you ran a genotype-scoped analysis over Bloom 2019 segregants or Caudal 2024 isolates
against that store, your population was incomplete and you need to re-run it against KG
4.0.**

Measured read-only over bolt against the live `tc-neo4j-readonly` by
`experiments/database/scripts/interned_id_collision_served.py` (lane N, landed as
`d9eefcbd7`):

- **518,883 of 52,668,931 `GenotypeMemberOf` relationships in the currently served store
  start at an `InternedConstant` node instead of at a genotype node.**
- **14,573 of the store's 21,020 `InternedConstant` nodes are whole genotypes**:
  **Bloom 2019, 13,630 segregant genotypes carrying 517,940 of those rows, of which only
  320 survive as `SegregantGenotype` nodes**; **Caudal 2024, 943 natural isolates,
  943 rows**.
- `PerturbationMemberOf` is untouched in the served store (0 of 45.3 M).

The cause: the id of a whole-genotype interned constant equaled the genotype node's own
id, because both are the sha256 of the same payload, and the CSV sink deduplicates ids
globally across labels, so it kept the constant and never wrote the genotype node.

The consequence, stated plainly: **`MATCH (g:Genotype)` cannot reach those records in the
graph we are serving.** Any genotype-scoped query over Bloom 2019 segregants has been
returning an incomplete population since interning landed: 13,630 of that dataset's
segregant genotypes are reachable only as an `InternedConstant`, and only 320
`SegregantGenotype` nodes survive for a genotype-label query to find. The query loader,
which follows the blob pointers rather than the graph, returns them intact: all
8,577,230 blob pointers resolve, which is why the loader never saw this and the graph
did.

What to do about it:

- **If your result came through the query loader** (`Neo4jQueryRaw` and the dataset
  classes on top of it), it is unaffected. The pointers resolve.
- **If your result came from Cypher that matches `Genotype` or `SegregantGenotype`**
  against KG 3.0, treat it as incomplete and re-run it against KG 4.0. For Bloom 2019 it
  is severely incomplete, not marginally: 320 surviving genotype nodes against 13,630
  that the graph holds only as interned constants.
- **If you need to know whether a given 3.0 result was affected**, the test is whether
  the query traverses `GenotypeMemberOf` from a genotype label. A query that starts at
  `Experiment` or `Dataset` and reads phenotypes is not affected.
- **Do not re-point an old analysis at 4.0 by id.** Ids re-key (previous section); re-run
  the query.

How 4.0 fixes it, and how we know the fix holds. The interned id is namespaced,
`constant_id(kind, payload) = sha256(label NUL kind NUL payload)`, so its preimage starts
with the label and is disjoint from every bare-dump content id by construction, and the
node's `kind` is inside the id that names it. Per-label deduplication was not available
as an alternative: `neo4j-admin import` has one id space and `--skip-duplicate-nodes`
keeps the first row whatever its label, so that option only moves the silent drop into
the import. Not interning whole genotypes was rejected too: it removes one instance and
leaves the mechanism (`genotype.background` collides the same way) while re-bloating the
Bloom, Boocock and Hale payloads. The sink now raises when an id arrives under a second
label, and the CSV endpoint gate checks every endpoint's label ancestry against the graph
schema's own declared `source` and `target` rather than a hardcoded table. Verified on
real data by lane P: Caudal's 943 `GenotypeMemberOf` rows start at 943 `Genotype` nodes
in the new output, where all 943 start at an `InternedConstant` in the served 3.0 dump.

On the job 4901 partial the same defect would have cost **218,258 genotypes across 16
datasets**, Hale 2024 first, which is what 4.0 would have shipped had lane N not measured
it.

### Record counts against KG 3.0

Every dataset KG 3.0 serves whose record count changes in 4.0, measured read-only by
`scripts/served_count_drift.py` against the committed served-counts table and the dev
stores on 2026-10-15: **of the 51 datasets KG 3.0 serves, 19 increased, 32 are unchanged
and 0 decreased.** The five rows the pre-build verification attributed commit by commit
(lane P, nine stores built from the integrated branch and walked record by record) are
these:

| dataset | KG 3.0 | KG 4.0 | delta | what changed | commit on `main` |
|---|---:|---:|---:|---|---|
| `CrispriChemgenSmith2016Dataset` | 7,053 | 14,463 | +7,410 | The released sheet has 14,463 rows, one per (pool, guide, drug, concentration). The 3.0 loader dropped 7,410 of them at `resolve_compound_identity` as proprietary, because the drug is a vendor catalog code; they are now stored with a typed `inchikey` gap. Not a double emission: 0 duplicated keys, and 3.0's 7,053 are a strict subset | `ce74c97d5` (#1045), 2026-10-12 |
| `CarotenoidOzaydin2013Dataset` | 4,474 | 4,607 | +133 | 133 released rows whose gene cell has no ORF form are stored under their released names | `e542fd365`, `0036fa30a` (#1045), 2026-10-12 |
| `BetaxanthinCachera2023Dataset` | 4,719 | 4,759 | +40 | 40 records whose released gene cell has no ORF form are stored on the released leaf. 339 served records also change `perturbed_gene_name` from the genome's standard name to the released cell (MRX12 to YJR003C) and 3,610 more differ on the released systematic name the same way | `0036fa30a` (#1045), 2026-10-12 |
| `SmfKuzmin2018Dataset` | 1,539 | 1,539 | 0 | 339 records gain a `fitness_se` that 3.0 held as null: the query-strain standard deviations of Additional Data S4, stored as `bootstrap_se` with `n_samples` 12, the conservative end of the released 12-to-24 range. 19 strains S4 lists with a blank value carry a typed gap instead. The companion `dmf_kuzmin2018` store gains 165 values and 7 gaps the same way | `207fe218d`, 2026-10-12 |
| `CaudalPanTranscriptome2024Dataset` | 943 | 943 | 0 | Every record's genotype is now a `natural_isolate` genotype with Peter 2018's loss-of-function calls, the HO deletion and Table S1's ploidy, zygosity and aneuploidy, and the bad-annotation rows are served under their pangenome ids. The TPM vector is 6,017 genes against 6,036 | `7e7444439`, `cd2d6e5f4`, `211293a81`, 2026-10-12 |

Three of those five rows are count increases (Smith 2016, Ozaydin 2013, Cachera 2023);
the Kuzmin and Caudal rows change stored values at an unchanged count. The other 16
increases, measured by the same script and attributed at release level to the eight
schema gates and the genome-gate reversals that landed between `v1.7.2` and `v1.7.3`
(#1049 to #1094, [[versioning]]), not commit by commit:

| dataset | KG 3.0 | KG 4.0 | delta |
|---|---:|---:|---:|
| `EnvChemgenHoepfner2014Dataset` | 3,083,827 | 30,308,574 | +27,224,747 |
| `HetHillenmeyer2008Dataset` | 2,712,677 | 3,074,132 | +361,455 |
| `HomHillenmeyer2008Dataset` | 1,063,034 | 1,228,676 | +165,642 |
| `EnvChemgenVanacloig2022Dataset` | 118,662 | 164,205 | +45,543 |
| `EnvChemgenWildenhain2015Dataset` | 430,820 | 449,033 | +18,213 |
| `FattyAcidSmith2006Dataset` | 12,747 | 14,310 | +1,563 |
| `IsobutanolScreenLopez2024Dataset` | 4,554 | 4,805 | +251 |
| `EnvChemgenCostanzo2021Dataset` | 61,430 | 61,640 | +210 |
| `GeneEssentialitySgdDataset` | 1,140 | 1,329 | +189 |
| `AminoAcidCooper2010Dataset` | 4,313 | 4,382 | +69 |
| `SmfBaryshnikova2010Dataset` | 5,993 | 6,023 | +30 |
| `EnvChemgenMota2024Dataset` | 1,270 | 1,289 | +19 |
| `SynthRescueYeastSynthLethDbDataset` | 6,942 | 6,948 | +6 |
| `SynthLethalityYeastSynthLethDbDataset` | 13,996 | 14,000 | +4 |
| `ScmdOhnuki2022Dataset` | 1,979 | 1,982 | +3 |
| `IsobutanolValidatedLopez2024Dataset` | 224 | 227 | +3 |

Sixteen rows there plus the three above is the 19 of 51 served datasets that increased.

Two caveats on the KG 4.0 column, both honest about what has and has not run. It is the
dev-store count as it stands on 2026-10-15, which is the count the build will serialize
only if the sweep reproduces it; the branch reproduces the main-built dev tree record for
record on all nine stores lane P verified (0 lost, 0 extra). And one row is a known
undercount in the graph rather than in the store: Ozaydin 2013's 4,607 records contain 31
pairs that are byte-identical as records, because the released sheet distinguishes them
by record number, plate and well and the record carries none of those, so the graph holds
4,576 `Experiment` nodes for 4,607 records and **31 released measurements are in the LMDB
and not in the graph**. That is pre-existing on `main`, not caused by the unification, and
whether the loader carries the source's record number before or after 4.0 is the owner's
call.

### Supported queries: expected drift

All four registered supported queries parse, and **none names a retired label**. Measured
on the branch by `experiments/database/scripts/kg4_release_prep_checks.py queries`, which
checks each query's labels, relationship types and `Label.prop` reads against this
checkout's own graph schema (the schema 4.0 will serve):

| query | parses | labels | relationship types | datasets | retired label named | drift vs 2026.10.06-4b293d34 |
|---|---|---:|---:|---:|---|---|
| `amino_acid_betaxanthin` | yes | 5 | 5 | 3 | none | 1 x contract_changed |
| `essentiality_smf` | yes | 7 | 7 | 2 | none | 2 x contract_changed |
| `expression_proteome_morphology` | yes | 5 | 5 | 5 | none | 3 x contract_changed |
| `solid_growth_025` | yes | 7 | 7 | 15 | none | 4 x contract_changed |

- **No query file needs editing in this release.** Every label and relationship type each
  query names exists in the branch's graph schema, every `Label.prop` read is carried by
  every class under that label, and every dataset each query selects is still served.
  `BacterialPerturbation` is absent from all four.
- The `contract_changed` drifts are **expected and pre-existing**, not caused by the
  unification: the identical ten drift details are reported from `main` at `7da9bfac0`
  and from the branch at `b5c5549d8`, so they describe the distance from KG 3.0's
  recorded closures, which this release closes by rebuilding.
- Each query's `dataset_composite` **will change**, and the new value cannot be computed
  before the build. The composite is the sha256 over the selected datasets'
  `content_sha256` values, which are the swept dev stores' content hashes recorded by
  `releases stamp` at the end of the build. Every one of the 51 datasets' content hashes
  moves, because every record re-serializes, so all four composites move.
- Re-validation is a step of the build, not of this branch, and it is the step that
  records the new composite:
  `python -m torchcell.knowledge_graphs.supported_queries validate <id> --release <new release>`
  for each of the four ids, after the snapshot is committed. `validate` resolves a
  `composite_changed` drift and refuses any other kind, so a `contract_changed` that
  survives the rebuild is a real problem and must be read, not validated away.

### Client compatibility

KG 4.0 pairs with package **`v1.8.0`** (a minor bump; see [[versioning]] for why, and
for the command that cuts it). The pairing is exact: the client gate
(`releases.require_paired`, run by `Neo4jQueryRaw._connect` before a driver is opened)
accepts a release only when the installed package's schema surface reproduces every
served dataset's recorded closure fingerprints, and there is no override.

Measured by `kg4_release_prep_checks.py compat`, which reads each tag's schema surface
with `git show <tag>:torchcell/datamodels/schema.py` and `pydant.py`, the same inputs
`scripts/kg_compat_page.py` renders the compatibility page from:

| schema surface | verdict against KG 3.0 | drifted | unverified | of served |
|---|---|---:|---:|---:|
| `v1.6.2` | compatible | 0 | 0 | 51 |
| `v1.7.3` | incompatible | 51 | 0 | 51 |
| `v1.7.4` | incompatible | 51 | 0 | 51 |
| `v1.7.5` | incompatible | 51 | 0 | 51 |
| branch `b5c5549d8` | incompatible | 51 | 0 | 51 |

So for a **`v1.7.5` client**:

- It cannot read KG 3.0 today, and that is measured above rather than inferred: all 51
  served closures drift. `v1.6.2` is the only package version that reads the store
  serving now. `v1.7.3`, `v1.7.4` and `v1.7.5` were cut as build pairs, not as client
  releases, and no store was ever built from them.
- It will not read KG 4.0 either. The records move every served closure
  (`served_datasets` 51 in three of the first four, 3 in P4, and 6 by the plan's estimate
  in P5), so the gate will refuse the 4.0 store under `v1.7.5` as well.
- The error it gets is `IncompatibleReleaseError` at connect, before any driver opens,
  reading (the message is assembled by `releases.require_paired`): `knowledge-graph
  release <release> (KG 4.0, database 'torchcell') is paired with torchcell v1.8.0; the
  installed torchcell 1.7.5 is not a pair:`, then one line per drifted dataset naming the
  symbols whose contract changed, then `<n> drifted and 0 unverified of <m> served
  datasets. Install the paired package (pip install torchcell==1.8.0) or set
  TORCHCELL_KG_VERSION to a release built under the installed schema.`
- The remedy is the first clause of that sentence: `pip install torchcell==1.8.0`. There
  is no partial-read mode; the client refuses the whole release rather than reading the
  datasets that happen to match, because the names in the message are evidence and not a
  usable subset.

`docs/source/database/compatibility.md` is generated and must not be hand-edited; it is
regenerated by `python scripts/kg_compat_page.py` once `database/releases/<release>.json`
is committed, which adds the 4.0 column and the `v1.7.3` to `v1.8.0` rows. Its `--check`
mode runs in CI (`docs.yaml`, job `query-drift`) and fails while the page is stale, so
the regeneration is a step of the release, not optional.

### Migrating client code: every retired symbol and its successor

Built from the records' own symbol lists. The first four records remove **48 symbols**
(42 pydantic classes, 2 string enumerations and 4 module-level bindings) and introduce
**15 names as renames**, counted from the records and classified against the `v1.7.5`
schema surface. P5's symbols are pending and its rows are marked so; its own record
replaces them when lane U lands.

Gene perturbation leaves (P1):

| removed | use instead | note |
|---|---|---|
| `BacterialDeletionPerturbation` | `DeletionStrainPerturbation` | Renamed. Tag `bacterial_deletion` becomes `deletion_strain`; `deletion_type` `bacterial` becomes `strain`. The name is the material the leaf states, a deletion strain, because 13 of its 25 loader sites state neither a collection nor a cassette, so "cataloged" would misstate half of them |
| `BacterialMarkedAllelePerturbation` | `MarkedAllelePerturbation` | Renamed; tag `marked_allele` |
| `BacterialDegronPerturbation` | `DegronPerturbation` | Renamed; tag `degron` |
| `BacterialVariantPerturbation` | `VariantCallPerturbation` | Renamed; abstract base of the three call leaves |
| `BacterialSequenceVariantPerturbation` | `LocusVariantCallPerturbation` | Renamed; tag `locus_variant_call` |
| `BacterialSiteVariantPerturbation` | `SiteVariantCallPerturbation` | Renamed; tag `site_variant_call` |
| `BacterialSpanDeletionPerturbation` | `SpanDeletionCallPerturbation` | Renamed; tag `span_deletion_call`, `deletion_type` `called_span` |
| `BacterialCrisprInterferencePerturbation` | `CrisprInterferencePerturbation` | Folded into the general leaf, which differed by the namespace alone; tag `bacterial_crispr_interference` becomes `crispr_interference` |
| `BacterialCrisprActivationPerturbation` | `CrisprActivationPerturbation` | Folded the same way; tag becomes `crispr_activation` |

Released-identifier leaves (P3):

| removed | use instead | note |
|---|---|---|
| `ReleasedGeneNamePerturbation` | `ReleasedIdentifierPerturbation` | Merged. `released_gene_name` is retired (it was required to equal `systematic_gene_name`); state `identifier_kind` instead |
| `BacterialReleasedStrainPerturbation` | `ReleasedIdentifierPerturbation` | Merged |
| `BacterialReleasedCrisprPerturbation` | `ReleasedIdentifierPerturbation` | Merged; `expression_direction` is retired in favor of the `crispr_activation` member of `ReleasedKind` |
| `BacterialReleasedTransposonPerturbation` | `ReleasedIdentifierPerturbation` | Merged |
| `ReleasedGeneKind` | `ReleasedKind` | Merged vocabulary |
| `ReleasedStrainKind` | `ReleasedKind` | Merged vocabulary |
| `RELEASED_GENE_KIND_SO` | `RELEASED_KIND_SO` | One Sequence Ontology map for the merged kind |
| `RELEASED_STRAIN_KIND_SO` | `RELEASED_KIND_SO` | The same |
| `marker` (field) | `cassette` (field) | The same fact under two names |

Experiment and reference pairs (P2). Ten pairs are removed into their general twin and
four are renamed with the host prefix dropped; `RETIRED_EXPERIMENT_TYPES` is the
authoritative map of the 14 stored tags:

| removed class pair | use instead | stored tag |
|---|---|---|
| `BacterialFitnessExperiment`, `...Reference` | `FitnessExperiment`, `...Reference` | `bacterial_fitness` to `fitness` |
| `BacterialEnvironmentResponseExperiment`, `...Reference` | `EnvironmentResponseExperiment`, `...Reference` | `bacterial_environment_response` to `environment_response` |
| `BacterialGeneInteractionExperiment`, `...Reference` | `GeneInteractionExperiment`, `...Reference` | `bacterial_gene_interaction` to `gene interaction` |
| `BacterialGeneEssentialityExperiment`, `...Reference` | `GeneEssentialityExperiment`, `...Reference` | `bacterial_gene_essentiality` to `gene essentiality` |
| `BacterialProteinAbundanceExperiment`, `...Reference` | `ProteinAbundanceExperiment`, `...Reference` | `bacterial_protein_abundance` to `protein_abundance` |
| `BacterialProteinFoldChangeExperiment`, `...Reference` | `ProteinFoldChangeExperiment`, `...Reference` | `bacterial_protein_fold_change` to `protein_fold_change` |
| `BacterialMetaboliteExperiment`, `...Reference` | `MetaboliteExperiment`, `...Reference` | `bacterial_metabolite` to `metabolite` |
| `BacterialRNASeqExpressionExperiment`, `...Reference` | `RNASeqExpressionExperiment`, `...Reference` | `bacterial_rnaseq_expression` to `rnaseq_expression` |
| `BacterialVisualScoreExperiment`, `...Reference` | `VisualScoreExperiment`, `...Reference` | `bacterial_visual_score` to `visual_score` |
| `YeastProteinTurnoverExperiment`, `...Reference` | `ProteinTurnoverExperiment`, `...Reference` | `yeast_protein_turnover` to `protein_turnover` |
| `BacterialGradedGeneEssentialityExperiment`, `...Reference` | `GradedGeneEssentialityExperiment`, `...Reference` | `bacterial_graded_gene_essentiality` to `graded_gene_essentiality` (renamed, no twin existed) |
| `BacterialPhosphositeIntensityExperiment`, `...Reference` | `PhosphositeIntensityExperiment`, `...Reference` | `bacterial_phosphosite_intensity` to `phosphosite_intensity` (renamed) |
| `BacterialMorphologyExperiment`, `...Reference` | `MorphologyExperiment`, `...Reference` | `bacterial_morphology` to `morphology` (renamed) |
| `BacterialFitnessContrastExperiment`, `...Reference` | `FitnessContrastExperiment`, `...Reference` | `bacterial_fitness_contrast` to `fitness_contrast` (renamed) |

Reference genome and gene addition:

| removed | use instead | note |
|---|---|---|
| `AssemblyReferenceGenome` | `ReferenceGenome` (with `assembly_set` and `assembly_accession`), or `StrainReferenceGenome` when a background is stated | The pin moved to the base, so no family needs a narrowed subclass to carry it. `StrainReferenceGenome.background` stays required |
| `GeneAdditionPerturbation.plasmid_contig_id` | `placement.contig_ref` (an `ArtifactRef` carrying its own sha256) | Required-or-gapped; the string pointed at a store that did not exist |
| `GeneAdditionPerturbation.locus_tag` | `placement.host.locus_tag` | The same |
| `localization` free strings | the two-member `Localization` vocabulary, with the source's words kept verbatim in `replicon_statement` | The replicon's identity is a property of the contig, read from its origin-of-replication feature, not of the gene riding on it |

Strain background (P5, pending: the successor names below are lane U's design as the
coordinator stated it, and each row is confirmed or corrected against P5's record when it
lands):

| removed | use instead | note |
|---|---|---|
| `BacterialStrainBackground` | the shared strain-background base | `StrainBackground` becomes that base with a eukaryotic subclass, so the fields a host does not have (`mating_type`, `ploidy`, `reference_strain: Literal["S288C"]`) sit on the subclass and the shared ones (`name`, `parents`, `construction`, `alleles`, `provenance`) on the base |
| `BacterialBackgroundAllele` | `BackgroundAllele`, which gains `gene_namespace` | One class for one fact. The merged class carries `gene_namespace` in P1's general vocabulary rather than `BacterialGeneNamespace`, and `zygosity` becomes optional instead of required-or-gapped, which is the loosened field the record signs off |
| `BackgroundAllele.zygosity` as required-or-gapped | `BackgroundAllele.zygosity` as optional | A haploid host has one possible copy count, so the field cannot carry information there; a yeast record states it exactly as before |
| `BacterialGeneNamespace`, `BACTERIAL_LOCUS_TAG_PATTERNS` | `GeneIdNamespace`, `GENE_ID_NAMESPACE_PATTERNS` | They survived P1 only because `BacterialBackgroundAllele` read them, and they go with it. `BACTERIAL_LOCUS_TAG_PATTERNS` is already derived from the general patterns, so membership does not change |
| `IntegratedCassette` with no placement | `IntegratedCassette.placement` | Placed the way P4 places a gene addition: a hash-carrying reference to deposited sequence, or a typed gap |
| `StrainReferenceGenome.background: StrainBackground \| BacterialStrainBackground` | `StrainReferenceGenome.background` on the shared base | P2 left the union in place for exactly this step; it collapses when P5 lands. The field stays required |

Client code that reads an identity projection should expect both to move:
`strain_background_identity` stops projecting `mating_type` and `ploidy` unconditionally,
and `BACKGROUND_ALLELE_IDENTITY_FIELDS` gains `gene_namespace`.

What has **not** changed, so client code need not move: `Experiment.experiment_type` is
still a plain string and `EXPERIMENT_TYPE_MAP` is still its only reader; the query loader
interface and the blob-pointer path are unchanged.

### Where every number here comes from

| claim | source |
|---|---|
| The records, their symbol counts, served datasets and loaders | `database/schema_changes/2026-10-15-*.json` (tool-computed scope, owner sign-off quoted in each); P5's is pending |
| P5's shape, its 6 served closures and 13 dev stores | the plan's estimate in [[scratch.2026.10.15.ontology-unification-decision]] (stage 10) and the P1 and P2 records, which name the 6 datasets carrying a background; superseded by P5's own record |
| 130,685,237 identifiers, 8 namespace members | `experiments/database/scripts/measure_gene_id_namespaces.py`, `experiments/database/results/gene_id_namespace_vocabulary.{json,md}` |
| R64 equals GCA_000146045.2 on all 16 nuclear chromosomes | P2 record, correctness argument |
| 518,883 rows, 14,573 interned genotypes, Bloom 13,630 with 320 surviving, Caudal 943 | `experiments/database/scripts/interned_id_collision_served.py` (read-only over bolt), [[database.kg-4-pre-build-verification]] |
| 23,650 of 23,650 experiment ids re-key; `host_genome` equals `assembly_set` on every row; 0 records lost | lane P, nine stores, jobs 4925, 4934, 4961, 4962, `experiments/database/results/prebuild_verify_stores.json` |
| 19 of 51 datasets increased, 0 decreased | `scripts/served_count_drift.py`, read-only, 2026-10-15 |
| The five attributed count changes and their commits | [[database.kg-4-pre-build-verification]], lane P's "Counts against KG 3.0" table |
| The four queries parse, name no retired label, and their drift kinds | `experiments/database/scripts/kg4_release_prep_checks.py queries`, which writes `experiments/database/results/kg4_release_prep_checks.json` (untracked by that directory's rule, since it reproduces from the committed snapshots, registry and tags alone) |
| The per-tag compatibility verdicts | `experiments/database/scripts/kg4_release_prep_checks.py compat --tag v1.6.2 --tag v1.7.3 --tag v1.7.4 --tag v1.7.5 --worktree`, same results file |
| 170 mapped datasets | `torchcell/knowledge_graphs/dataset_adapter_map.py` (169 public, 1 private) |

One gap worth naming rather than papering over: the KG 3.0 disclosure above cannot be
written into the 4.0 store's own release metadata by any flag that exists today.
`releases snapshot --note` is accepted only together with `--torchcell-version`, which is
refused when the manifest already records a version, and a build from a tagged commit
always records one; `kg_manifest bootstrap` takes no note. So the disclosure lives in this
note, in the `v1.8.0` GitHub release body and on the compatibility page. Putting it in
`events[].note` of the snapshot would need a code change, which is not in this release.
