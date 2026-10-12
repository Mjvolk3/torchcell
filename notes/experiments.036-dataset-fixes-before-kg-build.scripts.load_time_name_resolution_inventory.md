---
id: 4lxlrtbxdey5qlwo91m82n2
title: Load_time_name_resolution_inventory
desc: ''
updated: 1791817923694
created: 1791817923694
---

## 2026.10.12 - The loaders left for the sweep

Lists every module under `torchcell/datasets` that resolves a gene name against a genome at LOAD time (owner, 2026-10-12: "the data is the data!!!! ... the genome is separate.. it doesn't gate anything"), with the branch that holds it. Read with `ast`, nothing imported. Calls counted: `.resolve_gene_name(`, `resolve_gene_name_strict`, `check_ambiguous_aliases`, `candidate_orfs`, `check_pin`, `reconcile_locus_tags`, `require_resolved`, `sga_systematic_name_currency`, `SystematicNameCurrency`, and any use of `SGA_SYSTEMATIC_REFUSALS`, `MERGED_INTO_LOCUS`, `GeneNameRefused`. A call inside a verification function is counted apart (it is a reported finding since the same day). Writes `results/load_time_name_resolution_inventory.json`.

Measured on `fix/genome-rules-report-not-gate`: 84 modules resolve at load time. By holder: Babu 2014, Butland 2008, Girgis 2009 and `bacteria_common` on `fix/audit-bacterial-refusals-1045`; 51 more bacterial loaders in its bacterial set; Costanzo 2016, Parsons 2006, Yoshikawa 2009 and `systematic_name_currency` on `fix/audit-yeast-refusals-1045`; Kuzmin 2020 on both yeast branches; Xue 2025 on `fix/keep-records-887-sd-zero`; and 23 unheld yeast loaders for the sweep.
