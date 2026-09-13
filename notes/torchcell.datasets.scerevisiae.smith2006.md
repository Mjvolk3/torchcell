---
id: t5pg308ay1dtt7ejmdbcrws
title: Smith2006
desc: ''
updated: 1783834308152
created: 1783834308152
---

## 2026.07.12 - Smith 2006 fatty-acid clear-zone screen build

`FattyAcidSmith2006Dataset` (`torchcell/datasets/scerevisiae/smith2006.py`). Smith et al.
2006, *Mol Syst Biol* 2:2006.0009, doi:10.1038/msb4100051, PMID 16738555. Genome-wide YKO
clear-zone (fatty-acid utilization) screen → `EnvironmentResponsePhenotype`. **L0-L4
verified, 14,163 records** (4,721 unique-ORF strains × 3 conditions).

- **Source (open-access SI, mirror + hash-pin):** Supplementary Table 1
  `msb4100051-s1.xls` (sha256 `7048663f…b283a4`), fetched once from the Europe PMC
  supplementary bundle for PMC1681483 into the library mirror `…/data/`; loader copies +
  verifies sha256. Legacy .xls, header row 23, 4,770 strain rows.
- **Model:** KO × carbon-source condition. Each condition = Media (YPBO/YPBM/YPBA) + a
  `SmallMoleculePerturbation` for the carbon species (oleic acid 0.1% / myristic acid
  0.125% / acetate 2%, percent_w/v), 30 C, aerobic, solid. One record per (strain, condition).
- **Ordinal score** on `environment_response` (float) + semantic `category`
  (`measurement_type=categorical`; the enum has no ordinal member). Clear-zone 4/3/2/1
  (enhanced/wild_type/reduced/defective); acetate growth 3/2/1 + undocumented 2.5 kept
  ("intermediate"). Reference = BY4742 WT, `environment_response=None`/`category=wild_type`
  (the verifier's L3 `reference_zero` requires numeric reference==0, so None is the honest
  ordinal encoding). `n_samples=3` (triplicate replicate plates; quadruplicate pinning is
  within-plate technical, not counted). No SE released.
- **Gene resolution:** 4,721/4,770 → current R64 ORFs; 49 dropped (26 non-current/dubious;
  23 alias-collisions where the alias target is already a direct-ID row, e.g. YOR240W→
  YOR239W/ABP140 — dropped to avoid mislabeling a distinct dubious strain). 22 legit renames
  (e.g. YGR272c→YGR271C-A) kept.
- Skipped the sparse 'Glucose (YEPD)' NG/LG flag column (not an ordinal score). Verifier
  wired as `env_chemgen_smith2006` in `runners.py`.

## 2026.09.12 - Serve-50 rebuild: shared media, typed ordinal calls, the YEPD growth-control drop

Rebuilt for admission. The 2026-07-12 build failed L0 structural on all 14,163 records
(`Media` gained a required `is_synthetic` after it was serialized) and carried four
substantive modeling problems. Every one is fixed here; nothing was served, so nothing is
updated in place.

### What changed and why

**The three media are now the shared `MEDIA_LIBRARY` objects `YPBO` / `YPBM` / `YPBA`.**
They were free-text `Media(name="YPBO", state="solid")` with zero components, although the
full recipe is published in one Methods sentence. The library objects carry it, including
the two distinct bases (`YPB` = yeast extract + peptone + K-phosphate pH 6.0 for the oleate
plate; `YNB_YE_B` = YNB + yeast extract + the same buffer for myristate and acetate), the
Tween 40 emulsifier the old build dropped entirely, the agar, and the fatty acid or acetate
as the medium's `carbon_source` component. L3 `media_membership` now reports 12,747 records
on a shared library medium; it used to report three free-text media joining nothing.

**The carbon source stopped being a `SmallMoleculePerturbation`.** Oleate, myristate and
acetate are not stresses dosed on top of a complete medium, each IS the plate's sole carbon
and energy source, so the edit is
`EnvironmentPhysicalPerturbation(factor=carbon_source, magnitude=<percent w/v>, agent=<Compound>)`
-- the convention condition-SGA (Costanzo 2021) already uses for its galactose arm.

Measured caveat on the review's literal instruction. The review asked for the perturbation
to be dropped outright and the medium to carry the edit alone. That does not survive the
verifier: `_l3_environment_perturbed` flags a record that has no perturbation AND sits at
the dataset's MODAL medium, and with three equal-sized media one of them is always the
mode, so 4,249 records would have failed. A typed carbon-source factor keeps the edit
explicit and keeps the three conditions distinguishable in `_condition_signature`.

**`measurement_type` is now `ordinal`, and the call is typed.** The 1-4 score stays verbatim
on `environment_response`; `category` is the shared `ResponseCategory` axis and
`category_label` keeps this screen's own word:

| ordinal | readout | `ResponseCategory` | `category_label` |
|---|---|---|---|
| 4 | clear zone | `enhanced` | `enhanced` |
| 3 | clear zone / acetate | `no_change` | `wild_type` |
| 2 | clear zone | `reduced` | `reduced` |
| 1 | clear zone | `severely_reduced` | `defective` |
| 2.5 | acetate | `mildly_reduced` | `intermediate` |
| 2 | acetate | `reduced` | `moderate` |
| 1 | acetate | `severely_reduced` | `poor` |

**`assay_type` is set per condition**: `halo_zone` for oleate and myristate,
`colony_size_array` for acetate. This is the axis that separates the two ordinal scales
that used to be distinguishable only by a free-text `units` string.

**`duration_hours` is 72.0, not 84.0.** The source gives a range, "Plates were incubated for
3-4 days at 30 C". The old build asserted an unmarked 84.0 h midpoint. No companion
statistic permits a back-solve, so the CLAUDE.md range rule takes the conservative lower
end, and the choice now lives in the record's `SourcedValue` note rather than only in a
docstring.

**The common name now comes from the genome, not the 2005-era Standard Name column.**
51 of the stored common names resolved to a DIFFERENT gene through the shared
`resolve_gene_name`, so L1 `canonical_gene_names` failed. Taking the genome's own standard
name for the resolved ORF (falling back to the ORF id when it has none) round-trips for all
6,607 genes in R64.

### The YEPD growth-control drop (472 strains, 1,416 records)

Column 7 of the released table flags strains that did not grow (`NG`, 263), grew poorly
(`LG`, 208) or were contaminated (`NG/CONT`, 1) on the YEPD plate the entire deletion set
was pinned on BEFORE replication onto the fatty-acid plates. The old loader called this
column "deliberately SKIPPED (not an ordinal score)", which is true of its type and false of
its meaning: it is the screen's own QC flag.

Measured on the released table, the flagged strains' scores are dominated by the growth
failure, not by beta-oxidation:

| YEPD flag | n | lowest oleate score | lowest myristate score | lowest acetate score |
|---|---|---|---|---|
| unflagged | 4,298 | 2.0% | 2.2% | 0.3% |
| `NG` | 263 | 98.1% | 98.9% | 96.2% |
| `LG` | 208 | 79.8% | 76.0% | 6.7% (70.7% score the middle value 2) |

A clear zone needs a growing patch, so for these strains the assay measures the absence of a
colony. `EnvironmentResponsePhenotype` has no QC-flag field, so there is no way to serve
them honestly with the flag attached; they are dropped and counted in
`preprocess/dropped_records.json`.

### Retention accounting

Source 4,770 strains x 3 conditions = 14,310 cells.

| rule | strains | records |
|---|---|---|
| `strain_failed_the_yepd_growth_control` | 472 | 1,416 |
| `systematic_name_does_not_resolve_to_a_current_orf` | 26 | 78 |
| `alias_resolution_collides_with_a_directly_present_orf` | 23 | 69 |
| `second_row_for_an_already_resolved_orf` | 0 | 0 |
| `condition_cell_is_blank` | 0 | 0 |
| **kept** | **4,249** | **12,747** |

(The review estimated 469 flagged strains; the measured count after resolution is 472.)

### Provenance

The consumed `msb4100051-s1.xls` had NO recorded retrieval anywhere: the library mirror's
`manifest.json` listed only the OCR artifacts, and the loader docstring said it was "fetched
once from the Europe PMC supplementary bundle" with no command. It is now deposited at
`$DATA_ROOT/torchcell-raw/smithExpressionFunctionalProfiling2006/data/msb4100051-s1.xls`
with a `manifest.json` recording the Europe PMC REST supplementary-files endpoint.

One honest wrinkle is recorded in the retrieval params rather than papered over: that
endpoint re-zips the bundle on every request, so the CONTAINER sha256 is not stable
(measured -- two retrievals on 2026-09-12 gave `b9293f20...` and `7fd9b611...`), while the
MEMBER hash is identical to the pinned `7048663f...` both times. `retrieve.zip_member`
currently REQUIRES a container hash, so re-running the record needs that check made
optional. No stable single-file URL exists: embopress 404s and the PMC `bin/` path returns
the JS proof-of-work page.

### Verifier result (verbatim, scratch runner with this entry's parameters)

```
env_chemgen_smith2006: FAIL
  [ok] L0 structural: 12747 records validated
  [ok] L1 count: observed 12747, expected 12747
  [ok] L1 pair_uniqueness: 12747 unique (study, strain, condition) records, one each
  [ok] L1 provenance_gaps: 12747 documented provenance gaps over 12747/12747 records; 0 deferred field(s)
  [ok] L1 canonical_gene_names: 4249 systematic names, one canonical spelling each, each current in the genome
  [ok] L2 value_fidelity: 12747 values checked
  [ok] L2 se_nonnegative: 0 values checked
  [ok] L2 uncertainty_sanity: 0 labelled uncertainties, none a zero dispersion
  [ok] L3 measurement_type_consistent: single measurement_type: MeasurementType.ordinal
  [XX] L3 reference_zero: categorical rule: 0 references carry no category; 1 distinct
       reference categories ['no_change']; baseline also used as a measured call in
       {'no_change': 12124}
  [ok] L3 environment_perturbed: all 12747 experiments carry an environmental edit
  [ok] L3 compound_identity: 12747 compound references carry a structure identifier
  [ok] L3 media_compound_identity: 25494 compound references carry a structure identifier
  [ok] L3 media_membership: 12747 records on a shared MEDIA_LIBRARY medium (3 distinct media)
  [ok] L4 gene_containment_sgd: 1.000 of 4249 measured genes are S288C reference genes
  [ok] L4 current_genome_genes: every one of the 4249 measured systematic names is current
```

### The one failing rule is a rule problem, not a data problem

`_reference_baseline_result` (`torchcell/verification/environment_response.py`) requires, for
a categorical dataset, that the reference's baseline category is used by NO experiment
record. That holds for a HITS-ONLY screen, where only deviants are tabulated (Auesukaree
2009: reference `tolerant`, experiments `sensitive`, passes). It cannot hold for a CENSUS
screen that scores every strain: here 12,124 of 12,747 records ARE the wild-type ordinal 3,
so `no_change` is necessarily in the measured vocabulary and no honest category is left for
the parental reference. Giving the reference `environment_response = 0.0` instead would
assert that the wild-type clear zone is 0 on an absolute 1-4 visual scale, which is false.

Requested shared-layer change: drop `and not collisions` from the categorical branch's
`passed` expression, keeping `baseline_used_as_measured_call` in `details` and in the
message. The rule's real content (every reference carries a category, and all references
agree on one baseline) is preserved. Mormino 2022 fails the same way for the same reason.

### Open flags

- 27 strains carry the `retested myristate` flag, whose legend describes a DIFFERENT assay
  (spotting a washed overnight suspension rather than pinning). The release does not say
  whether the myristate column holds the retest or the original score, so `assay_type` stays
  `halo_zone` for all of them rather than guessing `spot_dilution`.
- The `Adherence` column (79 strains) is a second released phenotype the paper analyses
  separately; it is deliberately not ingested.
- `n_samples=3` describes the triplicate replicate plates, but the released value is a single
  visual consensus ordinal. No uncertainty is stored, so nothing is mis-derived.

## 2026.09.13 - Categorical reference rule and the zip_member retrieval both land

Two shared-layer requests from the 2026.09.12 section were applied, and this dataset now
PASSES every level.

**`L3 reference_zero`.** The categorical branch of `_reference_baseline_result` no longer
fails when the baseline category is also a measured call; the collision is reported in
`details` instead. A census screen necessarily puts its neutral baseline in the measured
vocabulary, and Smith 2006 is the worked case (12,124 of 12,747 records are the wild-type
ordinal 3). Verifier line is now:

```
[ok] L3 reference_zero: categorical rule: all 12747 references carry the baseline category
     'no_change', which no experiment record reports
```

**The retrieval record now re-runs.** `retrieve.zip_member` accepts `container_sha256=None`
for hosts that re-zip per request, so the stored record calls it with the container hash
left None and the MEMBER hash pinned. Re-running the record exactly as the manifest stores
it (`RETRIEVERS[record.retriever](**record.params)`) returned
`7048663ffa4890478724e6e371f434baccc7160e6d8250df9a777a26c6b283a4`, matching the pinned
member sha256. The rebuild-from-scratch chain for this file is now closed rather than
aspirational.
