---
id: i7p9rjr8caxx8yq2q0t0m4f
title: Yoshikawa2009
desc: ''
updated: 1791801520991
created: 1791801520991
---

## 2026.10.12 - Loaded: ethanol and NaCl growth rates from the publisher's Table S1

Yeast Zotero audit rank 3 ([[database.yeast-zotero-collection-audit.2026.10.11]], off-schedule, Zotero `DCF6RHD6`). Yoshikawa et al. 2009, FEMS Yeast Res 9:32-44 (doi 10.1111/j.1567-1364.2008.00456.x, PMID 19054128) grew 4,729 MATalpha deletion strains one per well in static 96-well YPD at 30 C, added 25 uL of sterile water, 40% (v/v) ethanol or 5 M NaCl at 4 h, and fitted each well's specific growth rate. Loader `torchcell/datasets/scerevisiae/yoshikawa2009.py`, class `EnvChemgenYoshikawa2009Dataset`, store `$DATA_ROOT/data/torchcell/env_chemgen_yoshikawa2009/` (slurm job 4166, built at `f994bbbbe`).

### The source: the publisher's file, cell-identical to YeastPhenome's

The owner downloaded the publisher's Table S1 by hand (`FYR_456_sm_tableS1.xls`, sha256 `1861ba64...f759`); the YeastPhenome copy (`.xlsx`, yp-data v1.0, sha256 `f8567d2c...3871`) sits beside it. Measured cell by cell ([[experiments.database.scripts.yoshikawa2009_release_checks]]): one sheet, 4,731 x 14, the same merged headers, 0 differences in value or kind over 66,234 cells. The `.xls` is the authors' own file, last saved by "Katsunori Yoshikawa2" on 2008-09-19, the paper's revision date; the `.xlsx` was re-saved by Anastasia Baryshnikova on 2013-03-06. **Both carry the 5% ethanol columns**: the condition is in the authors' release, not a curator's addition. The loader reads the `.xls`; the `.xlsx` is recorded as the cross-check (manifest role `si_data`, scriptable `direct_url` retrieval at commit `83e2917b`, re-fetched 2026-10-12 to the same sha256).

### Records (measured from the built store)

| quantity | count |
|---|---|
| Table S1 strain rows | 4,729 |
| released rate cells (4,729 x 4 conditions x 2 cultures) | 37,832 |
| refused: the 5% ethanol columns (4,729 x 2) | 9,458 |
| refused: 39 strains x 3 kept conditions x 2 cultures | 234 |
| served records | **28,140** |
| strains served | 4,690 (21 under an SGD-renamed ORF), on 4,689 genes |
| nonstress (base-condition) records | 9,380 |
| references (one per kept condition) | 3 |

Refused strains (`preprocess/dropped_records.json`, `preprocess/systematic_name_ledger.json`), through the systematic-name currency tables of #893 (the SGA tables plus seven pins for names only this release carries, each quoted from SGD's cached locus history or checked against the R64-4-1 GFF):

| rule | strains | names |
|---|---|---|
| `released_name_retired` | 10 | YAR037W, YAR040C, YAR043C-a, YCL006C, YCL013W, YCL026C, YDL061W, YHR034W, YHR096W, YKL147W |
| `released_name_merged_into_locus` | 28 | YAR044W, YBR075W, YBR100W, YCL003W, YCL012W, YCL060C, YCL062W, YCR062W, YDL038C, YDR474C, YER108C, YFL006W, YFL043C, YFR024C, YGL046W, YGR272C, YIL168W, YIR044C, YJL017W, YJL021C, YKL158W, YKL199C, YKL200C, YLR391W, YML033W, YOR088W, YOR240W, YPR090W |
| `released_name_change_unsourced` | 1 | YDL134C-A: R64 lists it as an Alias of YDL133C-A (RPL41B) alone, and SGD's history of YDL133C-A records "Name: YDL134C-A" and no annotation change, so neither the rename rule nor the merge rule is sourced |

25 of the 28 merged loci are also released as their own strain; YCL004W, YDR475C and YGR271C-A are not.

### What the record is

- Phenotype: `growth_rate` in 1/h, ONE culture per record (`replicate_id` "1st experiment" / "2nd experiment", the release's column names; `n_samples` 1, `biological_replicate`), `liquid_od_growth`. The released rate is normalized: "specific growth rates were normalized with those of the standard strain cultured in each plate using the formula", i.e. multiplied by the condition's median standard-strain rate over the plate's mean of four standard-strain wells. Nothing is averaged: "All strains were separately cultured twice under nonstress and stress conditions." `category` is the release's call, `0: sensitive` / `1: no-change` / `2: tolerant` mapped to `sensitive` / `no_change` / `resistant`, made from both cultures ("We defined stress-sensitive deletion strains as those whose growth rates in both independent cultures were lower than the threshold value."), so it is the same on a strain's two records of a condition. The calls reproduce Table 2 exactly, which the build checks.
- Genotype: `BarcodedKanMxDeletionPerturbation`, collection "Open Biosystems" ("purchased from Open Biosystems"), cassette kanMX4 (the paper names the Winzeler 1999 collection; Giaever 2014 states its cassette), barcodes gapped pending Giaever 2002. The 21 renamed strains keep their released name on a `ConstructedOrf` (relation `alias`, sourced by the SGD nomenclature pin). That is what keeps RPL22B's two released strains apart: the release carries both `YFL034C-A` and its pre-1998 name `YFL035C-B`, with different rates (0.4501 / 0.4652 against 0.4457 / 0.4435 nonstress).
- Environment: the shared liquid YPD restated with the paper's recipe ("YPD medium $1 \%$ Bacto yeast extract, $2 \%$ Bacto peptone, and $2 \%$ glucose)"), 30 C, a static 96-well culture of 127 uL after the addition (100 uL of YPD + 2 uL of pre-culture + 25 uL, arithmetic on three stated volumes), a 24 h shaken YPD pre-culture, `duration_hours` 9 (nonstress) or 11 (stress), inoculation to the last OD reading. The stresses are `SmallMoleculePerturbation`s: ethanol at 8% `percent_v/v` (the final concentration of a "40% (v/v)" stock) and sodium chloride at 1 M; the stock's diluent is a typed gap. Nonstress has no perturbation (the 25 uL of water is a dilution) and is declared as the base condition to the verifier. Aerobic: an open static well; the paper describes no oxygen limitation.
- Reference: per condition, the standard strain ("We used the HIS3 deletion strain of BY4742 (BY4742Dhis3) as a standard") in the same condition, Table 1's average rate (0.452, 0.178, 0.203 h-1) with Table 1's SD (4.4, 5.6, 7.6 x 10-3) as `sample_sd` over its 720 wells (4 per plate x 90 plates x 2 runs), on a BY4742 `StrainReferenceGenome`. The OCR and the PDF text layer write "MATa his3D1"; the rendered page 2 reads "MATα his3Δ1 leu2Δ0 lys2Δ0 ura3Δ0", recorded on the quote's note. The background's `provenance` is a typed gap: "derived from S. cerevisiae BY4742 (...) or BY4739 (...)" and Table S1 never says which strain is which; BY4739 carries HIS3.

### Owner decisions taken by recommendation

1. **The 5% ethanol columns are refused** (`condition_not_described_by_primary`, 9,458 records), although the authors' own file carries them. The Methods describe no 5% condition (its dose basis, addition time and regression window are unstated) and Table 1 gives it no standard-strain rate, so neither the environment nor the reference is sourced. The alternative is to serve them with the 8% protocol assumed and a reference back-solved from the calls (the bracket is in the release-checks note); that would be the first reference value this loader did not read from the paper.
2. **Merged ORFs are refused** (28 strains), following `gene_alias_resolution.py`'s `MERGED_INTO_LOCUS` and the #893 currency tables Costanzo 2016 and Kuzmin 2020 use. Parsons 2006 instead keeps the same kind of strain (YDL038C under YDL039C) with a `ConstructedOrf(relation=merged)`; the two loaders differ here, and the leaf can hold either reading.
3. **One record per culture**, not a two-culture mean the release never states.
4. **Renamed strains carry their released name** on a `ConstructedOrf(alias)` rather than being stored bare under the current ORF, which is what lets the release's two RPL22B strains coexist.
5. **YDL134C-A is refused** rather than mapped to YDL133C-A by its single candidate.

### Schema impact

`scripts/schema_impact_check.py --base origin/main`: "No schema contract changes vs origin/main." The only verifier change is additive: a registry entry may declare `reference_centered: False` and `expected_unperturbed`, which `verify_environment_response_entry` now passes to both verifiers.

### Raw mirror (`$DATA_ROOT/torchcell-raw/yoshikawaComprehensivePhenotypicAnalysis2009/`)

| path | bytes | sha256 | retrieval |
|---|---|---|---|
| `si/FYR_456_sm_tableS1.xls` | 1,399,296 | `1861ba647e382f28a39041b2b590276ad30e67ef334b0d8be968415f56d9f759` | `manual_browser`, the owner, 2026-10-12 03:41 CDT |
| `si/FYR_456_sm_tableS1.xlsx` | 553,420 | `f8567d2cbfe974a87cdb8629086884bc5e3c280513b99ab944065fb53d873871` | `direct_url`, yp-data `83e2917b` |

`manifest.json` is written by `write_raw_manifest()` from the pins; `si/DEPOSIT.md` and `si/SHA256SUMS.txt` record the deposit. No genomes-tier deposit: the host is the S288C reference.

### Verification, L0 to L4

`ENVIRONMENT_RESPONSE_DATASETS["env_chemgen_yoshikawa2009"]` (eager, `reference_centered: False`, `expected_unperturbed: 9380`) plus the loader's L2 `release_fidelity`, run by `verify_build` under `build_dataset_lmdb --verify` in slurm job 4166 (array script, one class; build 30 s, verification about 5 min). Report at `preprocess/verification_report.json`. All rows pass:

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 28140 records validated |
| L1 | `count` | pass | observed 28140, expected 28140 |
| L1 | `pair_uniqueness` | pass | 28140 unique (study, strain, condition) records, one each |
| L1 | `provenance_gaps` | pass | 159586 documented provenance gaps over 28140/28140 records; 2 deferred field(s): ['barcode', 'downtag_barcode']; 1228654 undeclared None values over 24 carrier fields (top: Compound.inchi x187600, Compound.chebi_id x168840, ...) |
| L1 | `canonical_gene_names` | pass | 4689 systematic names, one canonical spelling each, each current in the genome; 13 common names the resolver cannot place; 13 are pseudogene loci the genome resolves to themselves |
| L2 | `value_fidelity` | pass | 28140 values checked |
| L2 | `se_nonnegative` | pass | 0 values checked |
| L2 | `interval_orientation` | pass | 0 of 0 stored intervals do not bracket their value, as declared |
| L2 | `uncertainty_sanity` | pass | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L2 | `release_fidelity` | pass | all 28140 stored rates and calls equal their released Table S1 cells, one record per kept (strain, condition, culture) |
| L3 | `measurement_type_consistent` | pass | single measurement_type: 'growth_rate' |
| L3 | `reference_zero` | pass | absolute rule: reference value finite and on the record's own scale for all 28140 records (['growth_rate']) |
| L3 | `environment_perturbed` | pass | 9380 of 28140 experiments carry no environmental edit, as declared (baseline temp=30.0, media='YPD (yeast extract / peptone / dextrose), liquid') |
| L3 | `compound_identity` | pass | environment edits: 18760 compound references carry a structure identifier; 0 declare a typed gap |
| L3 | `media_compound_identity` | pass | medium components: 28140 compound references carry a structure identifier; 0 declare a typed gap |
| L3 | `media_membership` | pass | 0 records on a shared MEDIA_LIBRARY medium, 28140 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | pass | 1.000 of 4689 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | pass | every one of the 4689 measured systematic names is a gene of the current genome |

The 13 placeless common names are exactly the 13 self-resolving non-gene loci (7 pseudogenes, 5 blocked reading frames and the transposable-element gene YJR026W, per the report's details), stored under their own ids with the id as the common name, since the R64 GFF gives them no `gene=` name.

`--list-stale --include-private` after the build names 48 stores, none of them this one.

### L4 cross-check against YeastPhenome (#1000)

[[experiments.database.scripts.yoshikawa2009_yeastphenome_crosscheck]]: the yeastphenome dev store holds 0 Yoshikawa rows, so PMID 19054128 must stay off its `SCREENS` (pinned by `test_yeastphenome_screens_do_not_serve_this_paper`). Recomputing yp-data's curated value (stress mean over nonstress mean) from the store, 9,302 of 9,348 joined cells agree to 1e-12 (Pearson 0.99977); all 46 others are reproduced exactly when the strains this loader refuses are pooled into their loci, so yp-data averages each merged ORF's partial-deletion strain with its locus strain. Against the NPV: Pearson 0.872, Spearman 0.719.

### Candidate gate

[[experiments.database.scripts.yoshikawa2009_candidate_verdict]] (the key is off-schedule, so `candidate-gate gate` has no row): G1 to G5 pass, outcome admissible; G3 now inventories the raw mirror (2 files, 4,731 worksheet rows, all intact) where the audit read `unmeasured`.
