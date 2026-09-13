---
id: 1h311pacpj31w68ev0xvhap
title: Hillenmeyer2008
desc: ''
updated: 1789271722921
created: 1789271722921
---

## 2026.09.12 - Serve-50 round: raw mirror, typed media, per-control-set references

The serve-50 review of `HetHillenmeyer2008Dataset` / `HomHillenmeyer2008Dataset` returned
FIX-THEN-SERVE with a DROP-RECORDS rule for both classes and named the raw mirror as a hard
precondition. This section records what changed, why, and what it measured.

Loader: `torchcell/datasets/scerevisiae/hillenmeyer2008.py`. Adapters:
`torchcell/adapters/hillenmeyer2008_adapter.py` with
`torchcell/adapters/conf/{het,hom}_hillenmeyer2008_adapter.yaml`. Tests:
`tests/torchcell/datasets/scerevisiae/test_hillenmeyer2008.py`,
`tests/torchcell/adapters/test_hillenmeyer2008_adapter.py`.

### Raw mirror, first, because the dev tree held the only copy

FitDb's portals (`fitdb.stanford.edu`, `chemogenomics.stanford.edu`) are DNS-dead and the
Science SI 403s, so the Stanford static supplement exists only as an Internet Archive
snapshot and as the bytes someone downloaded into
`$DATA_ROOT/data/torchcell/env_chemgen_hillenmeyer2008/raw/` on 2026-07-11. That directory
is not in `scripts/backup_mirrors_to_bulk.sh`'s rsync, so a dev-tree clear would have
destroyed the dataset's only copy.

The seven files the loader and its provenance consume now live at
`$DATA_ROOT/torchcell-raw/hillenmeyerChemicalGenomicPortrait2008/data/` with a
`manifest.json` (`torchcell.literature.manifest.Manifest`) recording per file the exact
archived URL, the `direct_url` retriever, the sha256 and `retrieved_at=2026-07-11`:

| file | Wayback timestamp | bytes |
|---|---|---|
| `het.ratio_result_nm.pub` | 20151207003548 | 41,016,044 |
| `hom.z_result_nm.pub` | 20151207003024 | 16,961,587 |
| `het.txt` | 20151207063659 | 53,484 |
| `hom.txt` | 20151207011015 | 30,311 |
| `het_controls.txt` | 20151207003758 | 12,513 |
| `hom_controls.txt` | 20151207004537 | 11,740 |
| `README_CEL_files.txt` | 20151207004902 | 863 |

The timestamps come from the CDX index
(`curl "http://web.archive.org/cdx/search/cdx?url=chemogenomics.stanford.edu/supplements/global*&output=text&fl=original,timestamp,statuscode,length"`).
`deposit_raw_mirror(..., verify_source=True)` RE-RAN every recorded retrieval on
2026-09-12 and all seven reproduced the deposited bytes exactly, so each record carries a
`last_check` with `matches=true`. `download()` now symlinks the mirror files into `raw/`
and verifies each sha256 against the manifest, and `process()` re-verifies the matrix
sha256 before reading it.

`hom.z_tdist_pval_nm.pub` (the companion p-value matrix, Wayback timestamp
20151207002137) is NOT deposited: the loader does not consume it, and the mirror holds
exactly what was consumed. It remains re-retrievable from that snapshot, which was
verified live on 2026-09-12.

### What the record is now

One record per (strain, environment, CONTROL SET). The third key is new and is what the
SOM's own scoring definition requires:

> Each treatment array was compared to an associated control set of no-drug arrays that
> matched (a) the deletion pool used in the experiment, (b) the number of growth
> generations of the experiment $( 2 )$ , and (c) the scanner for the experiment (one of
> two scanners).

`het.txt` / `hom.txt` release the per-array control-set assignment and
`het_controls.txt` / `hom_controls.txt` its membership; the old loader read neither and
emitted ONE reference per matrix. Now the control-set id (`<pool>::<scanner>::
<generations>::tag3::YPD::dmso::0`) is stored on
`EnvironmentResponsePhenotype.screen_id`, one
`EnvironmentResponseExperimentReference` is emitted per control set with that set's
control-array count as `n_samples` (the n of the z-score's denominator and of the
t-distribution's df), and the reference's `duration_generations` matches the treatment's
instead of being `None`.

Keeping the control set in the identity also solves the generation-sign problem. The sign
is a protocol flag, not a negative duration:

> A negative sign preceding the number indicates that the pool was taken directly from the
> freezer, thawed, diluted and grown in the condition. Absence of a negative sign indicates
> that the pool was thawed, inoculated into YPD and grown overnight until log phase
> (OD600= 2.0)(~10 generations of recovery) before drug addition.

`duration_generations` therefore stores the MAGNITUDE (`-5gen` -> 5.0, where the old build
stored -5.0 for 5,784 het and 207,995 hom records), and the signed form survives inside
`screen_id`, so a `-5gen` array and a `5gen` array of one condition never merge.

### Media are library objects, temperature is a typed gap

Every record's base medium is the shared `YPD_LIQUID`, never the old name-only
`Media(name="YPD", state="liquid", is_synthetic=False)`. Table S1's "media change" class
maps onto the library: `minimal media` -> `SD`, `synthetic complete` -> `SC`,
`YP glycerol` -> `YP_GLYCEROL_LIQUID` (which gaps its glycerol percentage to Pierce 2006
rather than borrowing Bloom 2019's 3%). The 15 named-nutrient drop-outs are DERIVED media
(`HILLENMEYER_DROPOUT_MEDIA`, SC minus the nutrient) rather than
`EnvironmentPhysicalPerturbation(factor=nutrient_dropout)` on YPD, which is what makes a
tryptophan drop-out join `SC` and `SC_URA`; `vitamin drop-out control media` is that
series' control, so it maps to plain `SC` rather than to a `Compound(name="vitamin")`.

Temperature is NOT defaulted to 30 C any more. The SOM states no growth temperature and
defers the whole protocol to ref (2):

> The protocol for pooled, competitive growth of the deletion strains, genomic DNA
> purification and PCR, and tag hybridization follows Ref. (2).

Pierce 2006 (Nat Methods 3:601) is not mirrored, so non-temperature conditions carry
`temperature=None` with a
`ProvenanceGap(field="temperature", reason=deferred_pending_source_review)` whose
`resolve_with` names that paper. The four temperature-shift conditions carry the stated
value and no gap. The adapter already tolerates a `None` temperature (it emits no
temperature node or edge), and the L3 `environment_perturbed` rule still passes because a
base-medium swap counts as an edit.

### Parse fixes the review found

- **`pH7.5 + FK506` was merged into plain `pH7.5`.** The pH branch returned before `cond2`
  was read, so a calcineurin-inhibitor arm was averaged into a pH stress. `cond2` is now
  appended in EVERY branch, not only in the small-molecule fall-through.
- **`irradiated` collapsed three conditions into one.** `"irradiated" in label` matched
  `no drug irradiated`, `angelicin irradiated` and `psoralen irradiated`, discarded the
  compound and the dose from the last two, and merged all nine arrays. `no drug irradiated`
  is now matched exactly; `<compound> irradiated` emits the compound at its dose AND the
  radiation factor, which is what SOM Table S1 says (it files angelicin and psoralen under
  ALKYLATING small molecules, not under radiation).
- **One dose spelled two ways became two environments.** `sorbitol 1.5 m` and
  `sorbitol 1.5e+06 um` at 15 generations are the same condition; so are
  `streptozotocin 2 mm` / `2000 um`, `mitomycin c 1 mm` / `1000 um` and
  `carboplatin 15 mm` / `15000 um`. `canonical_concentration` converts the molar family in
  `decimal.Decimal` (exact, so both spellings land on the identical float) and picks the
  largest unit leaving the value >= 1. The cost is that a sub-unit dose moves down a unit
  (`0.5 uM` is stored as `500 nM`); the gain is that physically identical doses can never
  fragment.
- **Gene names go through `resolve_gene_name` first.** The old loader kept only names
  present in the R64 FASTA headers and logged a bare count. Now a RENAMED ORF is stored
  under its current systematic name and only retired / non-gene ids are dropped, each with
  its status written to `preprocess/dropped_genes.json`.
- **Rows of one ORF are not replicates.** "The number of strains exceeds the number of
  genes because some gene deletions were constructed more than once, in different
  batches", so within a group each ARRAY contributes one value (the mean over that ORF's
  construction rows) and the record is the across-array mean with the across-array sample
  SD; `n_samples` is the array count, which is the replicate count the SOM defines.

### The HIP background-mutation risk, flagged the way Hoepfner's is

Hillenmeyer reports this artifact itself, six years before Hoepfner, and names the
batches. The row id in both matrices is `ORF:batch`:

> Batches are denoted by $\mathrm { c h r X \_ Y }$ , where X is the chromosome and Y is
> an index.

and:

> In the heterozygous cluster analysis, we noted a few abnormally strong clusters whose
> genes were functionally unrelated but the strains in each cluster originated from a
> common deletion collection generation batch. We believe this is due to unrelated
> artifacts, e.g. a secondary site mutation in the parent strain used to make each batch.

and:

> 647 strains belonged to these suspicious batches, and we exclude these strains from most
> analyses.

and:

> The homozygous strains did not show this pattern.

The old loader threw the batch away (`cells[0].split(":")[0]`), so nothing in either LMDB
could identify those strains. No served perturbation class carries a construction batch
(`strain_id` exists only on `MarkerDeletionPerturbation`), and adding one to
`GenePerturbation` would change a class in every served dataset's closure, so the batch
stays OUT of the record and the authoritative flag is written beside the build, exactly as
the Hoepfner purge list is:

- `preprocess/strain_batches.json` -- every ORF's construction batches, the two SOM
  quotes and the sha256 of the quoted `paper.md`.
- `preprocess/suspicious_batch_strains.json` -- the ten named batches, the three quotes,
  and the ORFs with at least one row in them.

Measured: **772 het ORFs and 634 hom ORFs** have a row in a suspicious batch. The het
figure exceeds the SOM's 647 because the SOM counts the strains inside its own analysis
set. Records for those ORFs are SERVED with the flag recorded, not dropped: the exclusion
is an analysis choice, not a defect of the measurement.

Also measured, and worth keeping: **86 het ORFs and 41 hom ORFs** carry more than one
construction row. That is higher than the 59 / 17 the review measured on the raw row ids,
because `resolve_gene_name` now folds a renamed ORF onto its current name and some of those
targets already had a row of their own.

### Build results (2026.09.12)

| | HET (HIP) | HOM (HOP) |
|---|---|---|
| arrays | 726 | 418 |
| kept environments / arrays | 474 / 658 | 257 / 381 |
| dropped environments / arrays | 57 / 68 | 32 / 37 |
| control sets (references) | 10 | 13 |
| ORFs kept | 5,863 | 4,695 |
| ORFs dropped (retired / non-gene) | 24 | 22 |
| **records kept** | **2,698,797** | **1,088,620** |
| records dropped | 324,848 (10.74%) | 134,586 (11.00%) |
| LMDB size | 9.6 GB | 4.0 GB |

Drop rules and their counts:

| rule | HET envs / records | HOM envs / records |
|---|---|---|
| `unidentifiable_agent` | 56 / 319,114 | 30 / 125,967 |
| `heat_shock_cycle_not_representable` (`37c, 45c`) | 1 / 5,734 | 0 |
| `unnamed_agent_dosed_into_a_media_swap` (`minimal media:400:um`) | 0 | 2 / 8,619 |

36 distinct condition labels resolve to no structure identifier across the two matrices:
12 are `PROPRIETARY` (the ten ChemDiv `chemical diversity labs` codes plus the two Biomol
catalog codes) and 24 are `UNRESOLVED_PUBLIC`, each carrying an `unresolved_reason` in the
pinned compound table that says why it is not resolvable rather than not yet resolved
(`amphotericin` and `latrunculin` are genus names that do not pick a congener, `FeCl4` does
not exist as written, `tyrphostin` and `phosphatase inhibitor` are activity classes, `ptp2`
is a condition label, the `DMSO <n>%` and `<compound>, <n>ul total up/dn` labels fuse a
dose or a hybridization volume into the compound name). The two largest single
contributions are `latrunculin` (39,907 het records) and `amphotericin` (39,739), both
genus names.

### Decisions taken against the review, with reasons

- **`DMSO 1% / 2% / 4%` and the four `, <n>ul total up/dn` labels are DROPPED, not
  repaired.** Their compounds are identifiable in principle. For DMSO the released unit
  contradicts the name (`DMSO 1%` ships as `1:um`, and 1% v/v DMSO is about 141 mM), and
  the hom `DMSO 4%` column carries no dose at all, so parsing the percent out of the name
  would mean overriding the released dose field with a number the release does not state.
  For the volume-suffixed labels, folding them into the base compound would pool a
  different hybridization protocol with the plain arrays. Both are counted under the
  unidentifiable-agent rule, which is where the review's own drop table put them.
- **`sorbitol` stays a `SmallMoleculePerturbation`.** Table S1 files it under "media
  change", but the media library has no sorbitol medium and inventing one here would put a
  dataset-specific recipe outside the shared library. The concentration bug is fixed
  (1.5 M and 1.5e+06 uM are now one environment), which was the part that corrupted data.
- **No `Solvent` on the reference.** The review asked for `Solvent(DMSO)` because every
  control set reads `...::YPD::dmso::0`. A reference environment carries no perturbation
  object, and `Solvent` hangs off `SmallMoleculePerturbation`, so there is nowhere to put
  it without asserting a 0% DMSO dose as an edit. The vehicle is recorded verbatim in the
  reference phenotype's `units` instead.
- **Records are keyed on ORF, not on (ORF, batch).** The review proposed (ORF, batch)
  keying to make the suspicious strains addressable. Two records that delete the same ORF
  in the same environment have identical genotype and condition signatures, so that keying
  would fail the L1 `pair_uniqueness` rule; distinguishing them needs a strain
  discriminator on the perturbation, which is a served-class change. The batch artifacts
  above carry the same information without touching the schema.

### Open flags

- `Environment` has no slot for the outgrowth-protocol flag, so the `-5gen` fact (no
  recovery outgrowth) survives only inside `screen_id`. A typed
  `Environment.pre_culture_generations` would make it queryable, at the cost of a full
  rebuild.
- The SOM's IC-15 prescreen ("those compounds/treatments that produced a measurable
  $1 0 { - } 1 5 \%$ inhibition of wildtype growth (IC-15) were chosen for further
  full-genome screening") is dose PROVENANCE with no home: `DoseBasis` has no `IC15`
  member and adding one changes a served class. Every dose here is a released numeric
  value, so nothing is misstated; the fact is simply not on the record.
- `torchcell-library/hillenmeyerChemicalGenomicPortrait2008/si/si1.*` is a condensed-matter
  physics supplement, not Hillenmeyer SI. The top-level `paper.md` / `paper.pdf` are
  correct and are what every quote above is anchored to, so this does not block the loader,
  but the `si/` slot needs a re-fetch.

### Verifier result

Run from a scratch driver that mirrors `run_environment_response` exactly
(`verify_environment_response_dataset_streaming` with the same `sgd_genes`,
`resolve_gene_name` resolver and `MIN_RNASEQ_GENE_CONTAINMENT`), carrying the proposed
`expected_count` and provenance for each entry, so `torchcell/verification/runners.py`
itself was not edited. `env_chemgen_hillenmeyer2008_hom`, verbatim:

```
env_chemgen_hillenmeyer2008_hom: FAIL
  [ok] L0 structural: 1088620 records validated
  [ok] L1 count: observed 1088620, expected 1088620
  [ok] L1 pair_uniqueness: 1088620 unique (strain, condition) records, one each
  [ok] L1 provenance_gaps: 1062896 documented provenance gaps over 1062895/1088620 records; 1 deferred field(s): ['temperature']; 23664224 undeclared None values over 11 carrier fields (top: Compound.inchi x6794212, Compound.chebi_id x5642876, Compound.inchikey x1942542, Compound.pubchem_cid x1942542, Compound.smiles x1942542)
  [ok] L1 canonical_gene_names: 4675 systematic names, one canonical spelling each, each current in the genome
  [ok] L2 value_fidelity: 1088620 values checked
  [ok] L2 se_nonnegative: 377403 values checked
  [ok] L2 uncertainty_sanity: 377403 labelled uncertainties, none a zero dispersion; 1 records report n_samples >= 2 with no uncertainty
  [ok] L3 measurement_type_consistent: single measurement_type: <MeasurementType.z_score: 'z_score'>
  [ok] L3 reference_zero: numeric rule: reference response == 0 for all 1088620 records
  [XX] L3 environment_perturbed: 8619 experiments have no environmental edit (no perturbation, baseline temperature 25.0, baseline media)
  [ok] L3 compound_identity: environment edits: 901329 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable)
  [ok] L3 media_compound_identity: medium components: 3889685 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable)
  [ok] L3 media_membership: 1088620 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (19 distinct media)
  [ok] L4 gene_containment_sgd: 1.000 of 4675 measured genes are S288C reference genes (>= 0.9)
  [ok] L4 current_genome_genes: every one of the 4675 measured systematic names is a gene of the current genome
```

L0 now validates 100% of records (the old build failed L0 on 100% of them, on
`Environment.media.is_synthetic Field required`), every compound carries a structure
identifier, every medium is a shared library object, and the reference-zero, pair-
uniqueness and gene rules pass.

**One rule fails, and it is a rule gap rather than a data defect.**
`_l3_environment_perturbed` derives the dataset's baseline temperature as
`_modal_scalar(...)`, which SKIPS `None`. With the baseline temperature gapped (the
review's own recommendation, since the SOM states none), the modal NON-None temperature
becomes one of the temperature-CHANGE conditions itself, and that condition's own records
are then flagged as carrying no edit. In hom the modal stated temperature is 25.0 C, so
the 8,619 records of the two `25 degrees C` arrays are flagged. Every other record passes,
including every media swap, because a non-baseline medium is recognized as an edit.

The one-line fix is to compute the baseline INCLUDING `None`: a dataset whose records
mostly gap the temperature has an UNSTATED baseline, and a record that states one does
differ from it. That is a change to a shared verification file, so it is written up as a
shared-layer request instead of being made here. The alternative (store 30 C as a
documented representative, the Bloom 2019 precedent) is rejected: the SOM states no
temperature anywhere except Table S1's list of temperature-CHANGE conditions
(20, 23, 25, 37 C), so 30 C on 3.8 million records would be a fabricated value.

The other failure the first run reported is FIXED rather than argued away. It read

```
  [XX] L2 uncertainty_sanity: 1/377404 labelled uncertainties are a sample dispersion of exactly 0
```

and the cause, measured against the raw matrices, is that exactly one (strain,
environment, control set) group per matrix has two arrays printing the identical score to
the release's last decimal (`YHR096C` at 0.428325 on het arrays `03_03_28_06` and
`03_03_28_12`; `YAL068C` at 4.7108 on hom arrays `03_06_06_01` and `03_06_06_06`). That is
a rounding coincidence at four to six printed decimals, not a measurement of zero
dispersion, and storing it would have given those two records a standard error of 0. The
loader now reports `environment_response_uncertainty=None` with a
`ProvenanceGap(not_reported_by_primary)` for that case, which is what the second run
above shows.

`env_chemgen_hillenmeyer2008_het`, verbatim:

```
env_chemgen_hillenmeyer2008_het: FAIL
  [ok] L0 structural: 2698797 records validated
  [ok] L1 count: observed 2698797, expected 2698797
  [ok] L1 pair_uniqueness: 2698797 unique (strain, condition) records, one each
  [ok] L1 provenance_gaps: 2693182 documented provenance gaps over 2670237/2698797 records; 1 deferred field(s): ['temperature']; 50153593 undeclared None values over 11 carrier fields (top: Compound.inchi x10879976, Compound.chebi_id x8876304, Compound.pubchem_cid x5397405, Compound.smiles x5397405, Compound.inchikey x5374461)
  [ok] L1 canonical_gene_names: 5825 systematic names, one canonical spelling each, each current in the genome
  [ok] L2 value_fidelity: 2698797 values checked
  [ok] L2 se_nonnegative: 654913 values checked
  [ok] L2 uncertainty_sanity: 654913 labelled uncertainties, none a zero dispersion; 1 records report n_samples >= 2 with no uncertainty
  [ok] L3 measurement_type_consistent: single measurement_type: <MeasurementType.log2_ratio: 'log2_ratio'>
  [ok] L3 reference_zero: numeric rule: reference response == 0 for all 2698797 records
  [XX] L3 environment_perturbed: 11472 experiments have no environmental edit (no perturbation, baseline temperature 23.0, baseline media)
  [ok] L3 compound_identity: environment edits: 2783585 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable)
  [ok] L3 media_compound_identity: medium components: 2721930 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable)
  [ok] L3 media_membership: 2698797 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (3 distinct media)
  [ok] L4 gene_containment_sgd: 1.000 of 5825 measured genes are S288C reference genes (>= 0.9)
  [ok] L4 current_genome_genes: every one of the 5825 measured systematic names is a gene of the current genome
```

Same single failure, same cause: het's modal STATED temperature is 23.0 C, so the 11,472
records of the two `23 degrees C` groups are the ones flagged. Both datasets pass every
other level.

### Proposed `runners.py` entry changes

`torchcell/verification/runners.py` was deliberately not edited (a shared file this round
touches from several places at once). The two entries should become:

- `env_chemgen_hillenmeyer2008_het`: `expected_count` 2921078 -> **2698797**; the comment
  "726 arrays -> 514 unique environments" -> "726 arrays -> 474 kept environments (658
  arrays; 57 environments / 68 arrays dropped) x 5863 R64 strains, one record per
  (strain, environment, control set)".
- `env_chemgen_hillenmeyer2008_hom`: `expected_count` 1179520 -> **1088620**; the comment
  "418 arrays -> 284 unique environments" -> "418 arrays -> 257 kept environments (381
  arrays; 32 environments / 37 arrays dropped) x 4695 R64 strains, one record per
  (strain, environment, control set)".

Both `Provenance` blocks should take the archived URL as `source_uri`
(`http://web.archive.org/web/<TS>id_/http://chemogenomics.stanford.edu/supplements/global/download/data/<matrix>`,
TS 20151207003548 het / 20151207003024 hom) and a `method` string describing the
control-set keying, the typed media, the gapped temperature, the duration magnitude, the
dose canonicalization and the three drop rules. The exact strings this round ran with are
in the scratch driver `hm_verify.py`.
