---
id: 7ebj2bl30d1pujtsq0ukjhp
title: Babu2014
desc: ''
updated: 1791450343406
created: 1791450343406
---

## 2026.10.08 - Loader added: the genome-wide eSGA digenic interaction map

Row 44 of the fifty bacterial datasets, tier 1. Babu, Arnold, Bundalovic-Torma et al.
2014, PLoS Genetics 10(2):e1004120, doi:10.1371/journal.pgen.1004120, PMC3930520,
PMID 24586182, citation key `babuQuantitativeGenomeWideGenetic2014`.

- loader: `torchcell/datasets/ecoli/babu2014.py`, `GeneInteractionBabu2014Dataset`
- adapter: `torchcell/adapters/babu2014_adapter.py`, conf
  `torchcell/adapters/conf/gene_interaction_babu2014_adapter.yaml`
- tests: `tests/torchcell/datasets/ecoli/test_babu2014.py`,
  `tests/torchcell/adapters/test_babu2014_adapter.py`
- dev store: `$DATA_ROOT/data/torchcell/gene_interaction_babu2014`
- raw mirrors: `$DATA_ROOT/torchcell-raw/babuQuantitativeGenomeWideGenetic2014/data/`
  (Table S1, Table S2) and
  `$DATA_ROOT/torchcell-raw/butlandESGAColiSynthetic2008/data/` (the deferral file)

### What the paper measured

163 Hfr-Cavalli (Hfr C) donor query mutants, each marked with a `cat` cassette, were
transferred by conjugation into an arrayed F- recipient collection of 3,968 Keio
single-gene deletions plus 149 hypomorphic strains, each marked with `kan`. Conjugants
were selected on LB plates carrying both drugs, grown 36 h at 32 C, imaged, and the
colony sizes normalized into a signed interaction score (S). Only the high-confidence
tail is released: 42,705 pairs, 25,239 aggravating and 17,466 alleviating.

### Which of the 122 mirrored SI files is authoritative

The release is 21 PDFs (Figures S1-S5 = `si1`-`si5`, Protocols S1-S16 = `si6`-`si21`) and
16 spreadsheets (Tables S1-S16 = `si22`-`si37`); the mirror holds 122 files once the OCR
artifacts are counted.

**`si23.xls`, sheet `WG_GI_Score_Mar_06_2013`, column `GI score`** is the only
authoritative source of a stored value. It is the one released table whose unit of
observation IS a (donor, recipient) pair with its own score; its row count is exactly
the 42,705 that both the Results and Protocol S14 state; and its sign split is exactly
the 25,239 / 17,466 the Results state. `si22.xls` (Table S1) is the second build input,
read for two columns only: `Essentiality` (which donors are hypomorphs) and
`Donors screened in this and previous study` (which becomes `screen_id`).

Two other sheets also carry a per-pair `GI score` and are NOT loaded, because both are
proper subsets of Table S2 re-listed with the annotation a figure needs: Table S12
(`si33.xls`, sheet `sig gis - intra and intermod`) and Table S13 (`si34.xls`, sheet
`Module pairs - overview network`). Loading either would duplicate records under a
second provenance. The remaining files and their reasons are in the loader's
`NOT_LOADED` and in `preprocess/files_not_loaded.json`.

### Sourcing

| value | source | quote anchor |
|---|---|---|
| 163 donors | `paper.md` | "In total, a set of 163 query 'donor' genes ..." |
| recipient array = 3,968 + 149 | `paper.md` | "This collection, contains 3,968 non-essential single gene deletions ... and 149 hypomorphic mutant strains [13,16] ..." |
| n_samples = 8 colonies | `si/si7.md` (Protocol S2) | "These eight replicate measurements of each gene pair were subsequently averaged into a single GI S-score" |
| 42,705 released pairs | `si/si19.md` (Protocol S14) | "We mapped 42,705 putative digenic interactions ..." |
| sign split 25,239 / 17,466 | `paper.md` | "S-scores of -3 or lower (25,239 in total) ... +3 or higher (17,466)" |
| cutoffs -3.3 / 3.1 | `si/si7.md` | "a GI S-score of <= -3.3 for aggravating and >= 3.1 for alleviating" |
| medium, 32 C, 36 h | `paper.md` | "selected on rich medium (Luria Broth) containing both marker drugs (Kan+Cm). After outgrowth for 36 hrs at 32 C" |
| strains | `si/si21.md` (Protocol S16) | "were from the Keio mutant library [1]. The Hfr C non-essential donor gene deletion mutant strains or essential gene hypomorphic mutations ..." |
| S-score definition | Butland 2008 `paper.md` | "Negative S scores correspond to putative aggravating interactions and positive S scores, to putative alleviating interactions." |
| `cat` / `kan` markers | Butland 2008 `paper.md` | Figure 1 legend |
| 149 SPA-tag hypomorphs | Butland 2008 `paper.md` | "we also added 149 F- potentially hypomorphic kan-marked strains ... C-terminal sequential peptide affinity tag (SPA)" |
| agar plates | Butland 2008 `si/si1.md` | "pinned onto a LB-Kan-Cm agar plate" |

`n_samples` is EXACT, not a range: two replicate screens x four biological replicate
recipient colonies. No back-solve or range rule was needed.

**Two deferrals, both followed.** (1) The S-score derivation: Protocol S2 says the
scores were computed "essentially as described in our previous genomewide study [1]",
and ref [1] of that protocol is Butland 2008, which is in the mirror and is where the
sign convention is quoted. Butland in turn defers the S formula to Collins 2006
(Genome Biol 7:R63), which is NOT mirrored, so the operational description is as far as
the chain reaches and the loader says so. (2) The identity of the 149 hypomorphic
recipients: the Results give the count and cite refs [13,16] (Butland 2008, Babu 2011).
Babu 2011 is not mirrored; Butland 2008's Supplementary Table 1 (`si2.xls`, sheet `st1`)
is the array roster and labels each b-number `Non-essential` or `SPA-tag essential`, with
exactly 149 of the latter. That file is the third build input, read from its OWN citation
key's raw mirror (the shape Rousset 2018 uses to read Cui 2018's guide table), never
copied into Babu's mirror.

### The schema question: a signed double-deletion interaction score HAS a home

`BacterialGeneInteractionExperiment` / `BacterialGeneInteractionExperimentReference` with
`GeneInteractionPhenotype` already existed in `schema.py` and were exported but unused;
this is their first consumer. `GeneInteractionPhenotype.gene_interaction` is a plain
signed float with no clamping, so the S score is stored as released. It must never go
into `FitnessPhenotype`, which clamps non-positive values: 25,239 of the 42,705 released
numbers are negative.

The graph side needs NO new class. `gene interaction phenotype` is already declared in
`biocypher/config/torchcell_schema_config.yaml` and served by the yeast interaction
datasets, `bacterial perturbation` is already declared, and `CellAdapter` already has
both the `gene interaction phenotype` node method and its reference method. Nothing in
`schema.py`, `torchcell_schema_config.yaml` or `cell_adapter.py` changed, so no served
dataset's fingerprint moves and no rebuild is implied.

**Two sourced facts have no slot, and both are filed rather than forced.**

- `n_samples` / `sample_unit`: `GeneInteractionPhenotype` has neither field. Adding one
  would change a SERVED graph class and force a full knowledge-graph rebuild, so the
  sourced eight-colony design is written to `preprocess/replicate_structure.json` with
  its verbatim quote and sha256 instead. Filed as issue #793.
- A hypomorphic allele has no bacterial gene-perturbation leaf. A `kan` cassette in the
  3'-UTR that alters transcript abundance is not a deletion, not a mapped transposon
  insertion, not a CRISPRi knockdown and not a promoter replacement; the yeast
  `DampPerturbation` is the right concept but carries no `gene_namespace` and is not a
  bacterial leaf. The 3,420 pairs involving one are dropped rather than typed as
  deletions. Filed as issue #792.

### Strain: MG1655 is the pin, the conjugant chassis is the background

Every released identifier is an MG1655 b-number (`b0002__thrA`), and 3,863 of the 3,880
distinct names ARE locus tags of GCA_000005845.2. A b-number is not derivable from a
`BW25113_` tag by string surgery, and the ECK crosswalk would put a derived mapping on
every record. So records pin `assembly_reference("MG1655", background=...)` and the
physical strain is a `BacterialStrainBackground` named
"Hfr Cavalli x Keio (K-12 BW25113) eSGA conjugant" whose `parents` are
`["Hfr Cavalli", "Keio collection (K-12 BW25113)"]`. This is the shape Rapp 2026 uses
for its BW25993-derived host. `genotype_statement` is a typed gap: the two parental
genotype strings are released only in Table S16's spreadsheet cells, which the
provenance audit reads as bytes rather than text.

### Table S2 subsumes Butland 2008

Table S1 attributes 124 of the 163 donors to `This Study` and 39 to `Butland et al.`,
and Protocol S2 says the new scores were "combined with our previously published GI
datasets from 39 genome-wide screens". Each record therefore stores its screen set
verbatim in `GeneInteractionPhenotype.screen_id`: 37,852 records from `This Study` and
727 from `Butland et al.`. Schedule row 43 (Butland 2008) is a provenance record rather
than a second loader, and its own release (Supplementary Tables 2-4) is listed in
`NOT_LOADED`. Butland's schedule row calls itself blocked "pending a paywalled
retrieval", which is now stale: its SI IS in the mirror and every count the row calls
unverified is readable from it. Recorded as issue #794 rather than edited here.

### Retention: 38,579 of 42,705

| rule | pairs | why |
|---|---|---|
| `hypomorphic_allele_has_no_bacterial_perturbation_leaf` | 3,420 | 1,024 with one of the 7 essential (hypomorphic) donors, 2,479 with one of the 149 SPA-tagged recipients, 83 with both |
| `b_number_is_not_a_locus_tag_of_the_pinned_annotation` | 183 | 17 released ids GCA_000005845.2 does not carry under any layer: `JW5447`, `JW5661`, `cscR`, nine `bNNNN.1` sub-numbered Keio entries, and `b0370`, `b0510`, `b4091`, `b4223`, `b4274`, `b4574` |
| `b_number_remapped_by_the_annotation` | 519 | 52 b-numbers the annotation carries as a `/gene_synonym` of a DIFFERENT locus (51 of a pseudogene, 1 of a current gene `b0519`->`b4572`). Storing the merged locus needs a `DerivedIdentifierMapping` and `DerivedIdentifierRoute` has no member for a retired tag of the pinned strain's own namespace (issue #753) |
| `contradictory_duplicate_pair` | 4 | the two ordered pairs Table S2 releases twice with different scores: `b1396__paaI`/`b2863__ygeQ` at -5.19537 and +4.33814, `b2675__nrdE`/`b4462__ygaR` at -7.12262 and -5.67666. The paaI pair disagrees in SIGN and the release gives no rule for choosing |

Measured build: 38,579 records, 22,732 aggravating and 15,847 alleviating, over 155
donors and 3,658 recipients.

### Duplication inside the release

- 2 ordered (donor, recipient) pairs appear twice with different scores; both rows of
  both pairs are dropped (above).
- 97 UNORDERED gene pairs appear twice because each gene was a donor in its own screen,
  and 24 of those 97 disagree in sign (`b0436__tig` / `b0957__ompA`: -7.63085 as
  donor-tig, +6.96912 as donor-ompA). These are KEPT as two records, because they are
  two different strains: in one `tig` carries `cat` and `ompA` carries `kan`, in the
  other the markers are swapped, so the two `Genotype` objects differ. The census is
  `preprocess/reciprocal_pairs.json` (97 pairs, 194 records, 24 sign disagreements).
- Table S12 and Table S13 re-list subsets of Table S2 (above), and Tables S6, S7, S8 and
  S10 are per-gene and per-complex COUNTS recomputable from the stored records.

### Observation: the alleviating cut in the data is 3.09, not 3.1

Protocol S2 states the thresholds as `<= -3.3` and `>= 3.1`. Measured on Table S2 the
most negative kept score is -3.38787, which matches the -3.3 cut exactly, while the
smallest kept positive score is 3.08809, which matches the Results' "+3 or higher"
rather than the protocol's 3.1: 213 kept rows lie between 3.08809 and 3.1. Recorded
rather than corrected; every row the release published is stored.

### Verification

`python -m torchcell.datasets.ecoli.babu2014 verify` reports PASS on all 20 checks over
the 38,579-record dev store: L0 structural, L1 count and digenic-pair shape, L1 gap
census and canonical names, L2 every stored score equals its Table S2 cell, L2
uncertainty sanity, L3 signed axis with a zero reference and both signs present, L3
screen-set membership, L3 compound and media identity, L3 media membership, L3 the
provenance audit of all 18 sourced values, and L4 3,661 of 3,661 perturbed loci are
MG1655 GenBank loci.

## 2026.10.09 - The eight-colony design moves onto the record (#793)

`GeneInteractionPhenotype` gained `n_samples`, `sample_unit`,
`gene_interaction_uncertainty` and its type
([[torchcell.datamodels.schema]], section "2026.10.09"), so Protocol S2's sourced count
is a field of every record instead of a file beside the build.

Every record now carries `n_samples = 8`, `sample_unit = colony`, and the number is read
from `SOURCED_VALUES["n_samples"]` rather than retyped in the phenotype builder, so the
stored value and the quote cannot drift apart. The quote
(`si/si7.md`, sha256
`b060ce3f5277f708675e7f26239d5ee2d357ffb2306a04da326417caad135db9`), verbatim:

> Each genome-wide screen was performed twice by replica pinning the conjugants arrayed
> in a 384 density format using four biological replicate recipient colonies, which is
> selected further on double antibiotics at 1,536 colony density, to generate double
> mutant colonies. These eight replicate measurements of each gene pair were subsequently
> averaged into a single GI S-score to account for colony plate variance.

`preprocess/replicate_structure.json` stays, and its role changes: it is now the
PROVENANCE COPY of the same two numbers. It carries the verbatim quote, the citation key,
the source path and its sha256 beside them, which is what makes the stored property
auditable rather than asserted; a float on a graph node cannot carry any of that.

The uncertainty pair stays None. The paper releases no per-pair dispersion, which is a
different statement from the p-value's typed `not_reported_by_primary` gap (the source
tested the SET: `P<=0.05` for the released `|Z|>=2` cut, never per pair). Those two
absences read the same from a query only while the fields are absent from the class.

The reference phenotype carries no replicate design: it is the unperturbed chassis scoring
0 by construction, not a measured cell.

### Measured, before and after

The change is additive, so no record count moves:

| | records | `n_samples` | `sample_unit` |
|---|---|---|---|
| before (origin/main) | 38,579 | absent from the class | absent from the class |
| after | 38,579 | 8 on 38,579 of 38,579 | `colony` on 38,579 of 38,579 |

42,705 released pairs, 4,126 dropped, 38,579 kept: 37,852 from the 124 `This Study`
screens and 727 from the 39 `Butland et al.` screens.

### L0 to L4

Verified on a build of this branch's loader, PASS, 31 rows, 0 failures. The rows this
change touches:

| level | rule | result |
|---|---|---|
| L0 | `structural` | 38,579 records validated |
| L1 | `count` | observed 38,579, expected 38,579 |
| L1 | `provenance_gaps` | 154,316 documented gaps over 38,579/38,579 records |
| L2 | `gene_interaction_equals_table_s2_cell` | 38,579 of 38,579 stored scores equal their Table S2 cell |
| L2 | `uncertainty_sanity` | 0 labeled uncertainties, none a zero dispersion; 38,579 records report `n_samples >= 2` with no uncertainty |
| L3 | `signed_interaction_score_with_zero_reference` | 22,732 aggravating, 15,847 alleviating, 0 zeros, reference scores [0.0] |
| L3 | `screen_id_is_a_table_s1_screen_set` | `This Study` 37,852, `Butland et al.` 727 |
| L3 | `provenance_audit` x 18 | every sourced value backed by a verbatim quote |
| L4 | `gene_containment_mg1655_locus_tags` | 3,661 of 3,661 perturbed loci are MG1655 GenBank loci |

The `uncertainty_sanity` row is the one that reads differently now: before this change no
record declared `n_samples`, so the "reports `n_samples >= 2` with no uncertainty" count
was 0 by absence. It is 38,579 by measurement now, and that is the honest reading of a
release that states its replicate count and no dispersion.

### Why the canonical dev store was NOT rebuilt by this branch

`$DATA_ROOT/data/torchcell/gene_interaction_babu2014` was rebuilt at
2026-10-09T07:45 UTC from commit `ccfb82ccd` of the parallel #792 branch, which adds the
marked-allele (hypomorph) leaf and 41,988 kept records. Rebuilding it from this branch
would discard that. The build and the L0-to-L4 run above are therefore from an isolated
root, and the canonical store needs ONE rebuild once both #792 and #793 are on `main`
(each change alone already moves the Babu schema closure, so a rebuild is due either way,
and the KG 4.0 full build remakes it regardless).

`build_manifest` on the shared tree reports `gene_interaction_babu2014` STALE on four
symbols, of which exactly one is this branch's (`GeneInteractionPhenotype`); the other
three (`BacterialMarkedAllelePerturbation`, whose current fingerprint is `None` because
this branch's closure has no such symbol, plus `EnvironmentPerturbationType` and
`GenePerturbationType`) are the parallel branches whose builds are in that tree.

Related: [[torchcell.datasets.ecoli.butland2008]], [[torchcell.datamodels.schema]].
