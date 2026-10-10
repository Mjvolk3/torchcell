---
id: lpsb8ddshhoum7pdq81z6g1
title: Thompson2019_valerolactam
desc: ''
updated: 1791615286934
created: 1791615286934
---

## 2026.10.10 - Row 59 ingested: two loaders, and the RB-TnSeq half measured subsumed

Schedule row 59, `Thompson 2019 valerolactam` ([`build_bacteria_candidate_datasets_table.py`](https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/build_bacteria_candidate_datasets_table.py)), doi:10.1016/j.mec.2019.e00098, P. putida KT2440. The row is filed under "Transposon fitness" with an estimate of 4,778 insertion mutants over 2 conditions. Measured, the RB-TnSeq half is already served and the loadable payload is the engineering ladder the row's own "Why" text points at.

### Mirror route: raw-mirror deposit from PMC open access

The paper is in neither Zotero library, so nothing was in `torchcell-library`. It is PMC open access (PMC6838509, CC BY-NC-ND), and the whole release is five scriptable objects of the PMC Article Datasets bucket. `deposit_raw_mirror` writes them to `$DATA_ROOT/torchcell-raw/thompsonOmicsdrivenIdentificationElimination2019/` with a retrieval record each:

| mirror path | role | bytes | sha256 |
|---|---|---|---|
| `paper/PMC6838509.1.txt` | paper_text | 46,473 | `f0244882bdfc6c4789079dbe76a14c47be04acb3a7ab6083f5a80f8f23c2da84` |
| `paper/PMC6838509.1.xml` | paper_text | 97,776 | `0a1f88d7987bfe5b0f71eb14ced2987aa22f5c20413a45c02ed6d4e257cb7ed4` |
| `paper/PMC6838509.1.pdf` | paper_pdf | 1,194,524 | `ed6c14563b2b48d10f269b1b1abc624b4cbfbd634c49cbf2e19b4a1b53238b2f` |
| `si/mmc1.pdf` | si_pdf | 2,303,927 | `73032e933ca57b6a0d20badd8dea04b89ba27d6039c2748b1d6f034085a4483f` |
| `si/mmc1.txt` | si_text | 3,760 | `197a98d3c122da930a6b4ae3b000767950e626d48f194487d07eefe7008067b1` |

`si/mmc1.txt` is derived, not retrieved: it is the text layer of `si/mmc1.pdf` rendered by pypdf 6.19.0 (every page's `extract_text()` joined by a newline), carried with a `ProcessingRecord` so Table S1 and the SI quotes are auditable against bytes rather than against a re-render. Nothing was added to Zotero.

### Duplication verdict: SUBSUMED at the sample level, no RB-TnSeq loader

Measured by [`thompson2019_valerolactam_release_inventory.py`](https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/thompson2019_valerolactam_release_inventory.py):

- The Methods describe ONE growth, "diluted 1:50 in MOPS minimal medium with 10 mM valerolactam", and the Results name "two valerolactam RB-TnSeq experiments".
- The Borchert 2024 compendium release (`fModule_Metadata.xlsx`, sha256 `4d649385...`) holds exactly two carbon-source samples at `2-Piperidinone` 10 mM: `set6IT064` (library `Putida_ML5`, 48-well microplate, Tecan Infinite F200, which is the vessel and reader the Methods state) and `set7IT045` (`Putida_ML5_JBEI`).
- The served `RbTnseqBorchert2024Dataset` (1,372,280 records) carries both in full: 4,732 records over 4,732 loci each, so an RB-TnSeq loader for this row would store 9,464 values a second time.
- The paper's own release is the Fitness Browser alone, with no per-experiment accession, so there are no other bytes to load. This is the Thompson 2020 and Schmidt 2022 state.
- The 5-aminovalerate samples beside them in Fig. S2 (`set7IT044`, `set7IT057`) are NOT this paper's: the SI says "All non-valerolactam fitness experiments are from Thompson et. al 2019", the lysine paper (row 55).

### What is loaded

Two dataset classes in `torchcell/datasets/pputida/thompson2019_valerolactam.py`.

`ValerolactamTiterThompson2019Dataset`, 4 records, `ProductTiterExperiment`: the 24 h column of the four-strain ladder (wild type 0.43, dOplBA 4.47, dOplBA dDavT 19.29, dOplBA dDavT dAlr 63.66 mg/L), stored verbatim as `ug/mL` since 1 mg/L is exactly 1 ug/mL. Every record carries `pBADT-davBA-ORF26` as three `HeterologousPathwayPerturbation`s: `davB` (PP_0383) and `davA` (PP_0382) with `is_heterologous=False`, an extra copy of the host's own genes, and `ORF26` from *Streptomyces aizunensis*. The reference is the wild-type producer at the same 24 h, which is the paper's own fold-change denominator.

`LactamGrowthRateThompson2019Dataset`, 9 records, `BacterialEnvironmentResponseExperiment`: Supplementary Table S1 in full, three strains on three 10 mM carbon sources, `MeasurementType.growth_rate` carrying the maximal specific rate in h^-1. Three of the nine are released zeros (`ΔdavT` on 5-aminovalerate and both mutants on valerolactam), not gaps. No record carries the plasmid: Figs. 2A to 2C grow plasmid-free strains. Each record's reference is the SAME strain on glucose, the carbon source all three grow on identically, so `reference_centered=False` and the absolute branch of the environment-response verifier applies.

### Four titers refused, and the schema finding behind it

The 48 h family is NOT loaded, and the reason is a missing denominator rather than a missing field in this loader. The paper states "no valerolactam could be detected after 48 h" for the wild type, which the abstract restates as "undetectable": a value below the 8-point calibration floor (0.78125 to 100 uM standards). `ProductTiterExperimentReference` requires a `phenotype_reference` and `ProductTiterPhenotype.titer` is a required non-negative float with no `Censoring` slot, unlike `ProteinTurnoverPhenotype`. Writing 0.0 would state a measurement nobody made; reusing the 24 h reference would compare two sampling times. So the wild type's 48 h cell and the three engineered 48 h titers measured against it (9.27, 85.19, 91.97 mg/L) are all refused, counted in `preprocess/build_accounting.json` and listed in `preprocess/refused_titers.csv`. The sibling Kang 2026 and Yunus 2026 loaders refuse a titer family for the same reason. The schema gap (a left-censoring carrier on `ProductTiterPhenotype`) is filed as its own issue.

### davT is placed by the genomes tier, not by the paper

The paper names no locus tag anywhere. Five of the six symbols resolve through the pinned GenBank annotation of GCA_000007565.2: `oplB` -> PP_3514, `oplA` -> PP_3515, `alr` -> PP_3722, `davB` -> PP_0383, `davA` -> PP_0382. `davT` returns "not found in GCA_000007565.2_ASM756v2". The genomes tier's own UniProt GOA proteome file for the same assembly set (`109.P_putida_KT2440.goa`) carries `davT` as the DB-object symbol of exactly one protein, Q88RB9, with `davT|PP_0214` in its synonym column and the product name "5-aminovalerate aminotransferase DavT", which is the enzyme the paper describes ("davT, which catalyzes the first step in 5AVA catabolism"). `goa_symbol_locus` reads that file and refuses anything but one locus, so 6 of 6 symbols reach a locus and `MIN_RESOLVED_FRACTION` is 1.0.

### This paper's MOPS is its own object

The Methods write out a modified MOPS minimal medium component by component (LaBauve and Wargo 2012). It is not the library's `MOPS_MINIMAL` (Neidhardt as Price 2018 tabulates it): calcium chloride is 32.5 uM against 0.5 uM, potassium sulfate 0.29 against 0.276 mM, ammonium chloride 9.52 against 9.5 mM, the magnesium salt is weighed as the chloride at 0.52 mM, and 8 uM iron(II) chloride is added. `MOPS_MODIFIED_THOMPSON2019` lives in the loader module with `base_medium="MOPS_MINIMAL"` so the family still joins, and stays out of `MEDIA_LIBRARY` for the Ishii 2007 and Choe 2019 reason. One component carries a typed identity gap: `iron(II) chloride` has no row in `compound_identity_table.json` under that name or any synonym measured here (`FeCl2`, `iron chloride`, `ferrous chloride`, `iron dichloride`), so it is the honest typed absence and the curation is raised in the PR.

### Verification

Both stores built in the dev tree with `python -m torchcell.datasets.pputida.thompson2019_valerolactam`; `torchcell.provenance.build_manifest` reports both fresh.

| level | `ValerolactamTiterThompson2019Dataset` | `LactamGrowthRateThompson2019Dataset` |
|---|---|---|
| L0 | structural: 4 records validated | structural: 9 records validated |
| L1 | count 4 of 4 | count 9 of 9; pair_uniqueness 9 unique (study, strain, condition); provenance_gaps 57 over 9/9 |
| L2 | value_fidelity 4 values, minimum 0.0 | value_fidelity 9 values; se_nonnegative 0; interval_orientation 0 of 0; uncertainty_sanity 0 labeled |
| L3 | titer stored as ug/mL; every uncertainty a typed gap; every record carries the plasmid; reference at the same sampling time | measurement_type single `growth_rate`; reference_zero absolute branch, finite on each record's own scale; environment_perturbed 9 of 9; compound and media identity |
| L4 | `stored_titer_vs_results_prose`, 4 entities within 1e-9, each quote re-read from the pinned bytes | `stored_growth_rate_vs_table_s1`, 9 entities within 1e-9, re-read from the pinned SI text layer |

Both reports are `PASS` and are written to each store's `preprocess/verification_report.json`.

### Schema impact

`python scripts/schema_impact_check.py --base origin/main` reports no schema-contract changes versus `origin/main`: no new class, no changed field, no impacted served dataset. Both datasets are additive admissions (new dataset classes, new adapter modules, new confs), which is the incremental-admission case rather than the full-rebuild case.

## 2026.10.10 - #867: the 48 h family stored, the wild type as a left-censored floor

Closes the refusal above. `ProductTiterPhenotype` now carries `titer_censoring` (a `Censoring | None`, the `ProteinTurnoverPhenotype` precedent of #753), so the 48 h wild type is stored as a bound and becomes the 48 h reference of the three engineered 48 h titers.

### What the paper says, verbatim (`paper/PMC6838509.1.txt`, sha256 `f0244882...2da84`)

- Results 2.3: "Wild type P. putida produced 0.43 mg/L valerolactam after 24 h, but no valerolactam could be detected after 48 h, presumably due to host consumption (Fig. 3B)."
- Abstract: "increased the titer of valerolactam from undetectable after 48 h of production to ~90 mg/L."
- Methods 4.4, the calibration: "Lactams were quantified by comparison with 8-point calibration curves of authentic chemical standards from 0.78125 μM to 100 μM."
- Methods 4.4, the sample handling: "with 200 μL of culture being quenched with an equal volume of ice cold methanol and then stored at −20 °C until analysis"

(The mirror text has narrow no-break spaces before every unit; the loader's quote constants carry them and `verify_quotes` re-reads all 26 paper quotes against the pinned bytes.)

### What is stored, and how the floor is derived

The paper states no limit of detection. The lowest calibration standard (0.78125 uM in the vial) was quantified by the curve, so a sample in which nothing was detected holds less than that in the vial. The Methods state one dilution, the equal-volume methanol quench (twofold). Valerolactam's molar mass, 99.133 g/mol, is derived (C5H9NO; rdkit 2026.03.6 `Descriptors.MolWt` over `O=C1CCCCN1`, which equals the IUPAC standard atomic weights summed by hand). The bound in the culture is therefore

`TITER_FLOOR_MG_PER_L = 0.78125 uM x 2 x 99.133 g/mol / 1000 = 0.1548953125 mg/L`

stored as `titer = 0.1548953125` (ug/mL, which is mg/L) with `titer_censoring = left`, no `titer_uncertainty` and no `titer_se`. The bound is conservative in one direction and conditional in another: the true limit of detection may be lower than the lowest standard (so the true titer is below a tighter number than the one stored, and the stored bound is still true), and the bound assumes the stated quench is the only dilution between culture and injection, which is what the Methods describe. Every stated number is stored with `titer_censoring = uncensored`, and every one lies above the floor (the smallest, 0.43 mg/L, is 2.8 times it).

Each record's reference is the wild type at its own sampling time: 0.43 mg/L (uncensored) at 24 h, the floor (left) at 48 h. The L3 rule `the_reference_is_the_wild_type_at_the_same_sampling_time` holds for all eight. A consumer computing a fold change against the 48 h reference gets a LOWER bound on the ratio (the denominator is an upper bound), so it must read `titer_censoring` on the reference before treating that ratio as an estimate. Not measured: no consumer of `ProductTiterPhenotype` labels in `torchcell/` reads the field yet (a grep for titer label consumers outside datasets, the schema, the adapter and the verifiers finds none).

### Counts, from the built dev store

`$DATA_ROOT/data/torchcell/valerolactam_titer_thompson2019`, rebuilt with `python -m torchcell.database.build_dataset_lmdb --dataset ValerolactamTiterThompson2019Dataset --retire-existing --verify`:

| | records in the LMDB | candidates | dropped | rule |
|---|---|---|---|---|
| before (origin/main 5382a5400) | 4 (24 h) | 8 | 4 | `no_released_reference_titer_at_this_time` |
| after (this branch) | 8 (24 and 48 h) | 8 | 0 | none |

`refused_titers.csv` is no longer written; `titer_rows.csv` gains `stored_titer_mg_per_l` and `titer_censoring`.

### Verification, L0 to L4 (`runners.verify_bacterial_dataset('valerolactam_titer_thompson2019')`, PASS)

| level | check | result |
|---|---|---|
| L0 | structural | 8 records validated |
| L1 | count | observed 8, expected 8 |
| L2 | value_fidelity | 8 values, minimum 0.0 |
| L3 | titer_unit_is_the_sources_mg_per_l_as_ug_per_ml | stated numbers verbatim; the floor is derived |
| L3 | every_uncertainty_is_a_typed_gap_not_a_guess | pass |
| L3 | every_record_carries_the_production_plasmid | pass |
| L3 | the_reference_is_the_wild_type_at_the_same_sampling_time | 8 of 8 |
| L3 | only_the_undetected_cell_is_a_bound_and_it_carries_no_error | exactly one left-censored record, 48 h wild type, no SE |
| L3 | every_stated_titer_lies_above_the_floor | 7 of 7 uncensored above 0.1548953125 |
| L4 | stored_titer_vs_results_prose | 8 entities within 1e-9; the censored cell joins to its floor, and the calibration and quench quotes are re-read from the pinned bytes |
| L4 | perturbed_gene_containment_assembly | 1.000 of 4 measured genes are loci of pputida_KT2440_ASM756v2 |

### Schema impact

`PYTHONPATH=<wt> python scripts/schema_impact_check.py --base origin/main`: one changed symbol, `ProductTiterPhenotype` (modified: added optional field `titer_censoring`; validator `_check_titer` changed). 6 dataset modules impacted, **0 breaking**, all stale -> rebuild: Foo 2014, Carruthers 2025 (4 classes), de Siqueira 2025 (4), Kang 2026, Thompson 2019 (2), Yunus 2026 (the two mapped classes `--list-stale` named). All 14 stale dev stores were rebuilt with `--retire-existing --verify`; 13 verifications PASS and `LactamGrowthRateThompson2019Dataset` has no CLI verifier (its module's `verify_growth_build` passes in the `--data` test). Afterwards `--list-stale --include-private` names none of them. The phenotype node id is a sha256 of the phenotype's `model_dump`, so every served titer node's id moves: this belongs to the KG 4.0 full rebuild, not to an incremental admission.

The second gap the issue records, `iron(II) chloride` and `valerolactam` without a `compound_identity_table.json` row, is untouched here: adding a row is a curation act.
