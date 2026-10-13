---
id: ip0argx2ste0hpxdrb17418
title: Li2021
desc: ''
updated: 1791822576848
created: 1791822576848
---

## 2026.10.12 - The three released 13C-MFA maps, loaded with the control's map as a typed gap

Li et al. 2021 (Microb Cell Fact 20:32, doi:10.1186/s12934-021-01526-1, `liFineTuningGlycolytic2021`), issue #800, PR #1070. `FluxLi2021Dataset`, slug `flux_li2021`, 3 records.

**What is stored.** One `FluxExperiment` per `NETWORK` block of Additional file 2 (`12934_2021_1526_MOESM2_ESM.xlsx`, sha256 `24c836f8459e5cce4f6a322477a7554e1f7dd16e8bb6b7cdd0ac4890b24c585a`, deposited in `$DATA_ROOT/torchcell-raw/liFineTuningGlycolytic2021/data/` from the library copy with its PMC Article Datasets retrieval), under the block's own label: BW-P08, BW-P04 and BW-P10. Each record carries the block's 99 net fluxes in mmol/gdw/h (the header cell reads `Absolute  Flux  value (mmol/gdw/h)`), signed and unclamped, the `LB90` and `UB90` bounds at `confidence_level=0.90`, and the exchange table under the block (24, 23 and 23 rows) as `exchange_flux` with its bounds. Every number is the workbook cell, checked bit for bit by the L2 rule.

**The reference.** BW-P BF (BW25113 `∆pfkA:FRT` carrying pBSA and pFF, stated verbatim as the strain background) on the labeling medium. Its map is a `ProvenanceGap` on `net_flux`: the paper names it as analysed ("MFA was performed to detect the metabolic fux distribution in strains BW-P08 BF and BW-P10 BF (which had high MVA titers) and the control strain BW-P BF.") and draws it in Fig. 5, and the workbook releases no block for it.

**Genotype.** Each record's edit relative to BW-P BF is its Table 1 `zwf` promoter (BBa-J 23108, 23104, 23100), a `PromoterReplacementPerturbation` keyed `BW25113_1852` with the released symbol `zwf` kept as `identifier_mapping.source_identifier` (route `gene_symbol`).

**Environment.** The 13C-MFA paragraph states the tracer, 100% [1-13C1]glucose at 10 g/L, and nothing else about the labeling culture, so the medium's name, state and synthetic flag and the temperature are typed gaps. `aerobicity` carries the field's default `aerobic`, which is NOT sourced: the schema has no gap for it (owner question in PR #1070).

**Findings about the release, recorded and not acted on.** The workbook's BW-P04 block is never mentioned in the paper. 50 of the 99 reaction rows are bit-identical (best fit and both bounds) between the BW-P04 and BW-P10 blocks, against 2 for BW-P08 vs BW-P04 and 0 for BW-P08 vs BW-P10, so the BW-P10 block carries BW-P04's extracellular constraints; it is stored under the label the release gives it. The `χ2 90%` cell reads 102.8 for BW-P08 and 102.08 for the other two blocks, which all state 85 degrees of freedom. No interval procedure and no replicate count are stated, so `n_samples` and `sample_unit` are gaps. The paper calls the file "MFA simulated result." while its Methods describe a least-squares fit.

**Not stored** (`preprocess/not_stored.json`): the normalized columns (891 numbers, the absolute map divided by the block's own released `vUPTU`), the 12 fit-diagnostic cells, and the three input sheets.

### L0-L4, dev store built and verified under slurm (job 4387, task 1, source snapshot `0d8bc1f43`)

`$DATA_ROOT/data/torchcell/flux_li2021`: 3 records, 1 reference. Verdict PASS.

| level | rule | result |
|---|---|---|
| L0 | structural | 3 records validate as `FluxExperiment` |
| L1 | count | 3 observed, 3 expected |
| L1 | one_record_per_released_block | promoters BBa-J 23100, 23104, 23108, one each |
| L2 | value_fidelity_to_workbook | 1,101 stored values equal the re-read workbook cells, 0 mismatched |
| L3 | flux_interval_and_gapped_reference | one measurement type; 0 incomplete intervals; 3 of 3 references carry the control-map gap |
| L3 | provenance_audit | 11 of 11 paper quotes verbatim in `paper.md` (sha256 `7d174c8e...`) |
| L4 | perturbed_locus_in_bw25113_annotation | 1 of 1 keyed locus (`BW25113_1852`) is a BW25113 gene row; a finding, never a gate |

## 2026.10.12 - Rebuilt on main after the rebase (slurm 4425)

The dev store was rebuilt from the rebased tree of PR #1070 (snapshot `accd94cfb`, slurm 4425, `--retire-existing --verify`) and verifies PASS with the same counts and the same L0-L4 rows as job 4387 above; the build manifest now records that tree's schema closure.
