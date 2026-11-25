---
id: yntifzp80gyim7p5qkta8ax
title: Fang2025
desc: ''
updated: 1791848119118
created: 1791848119118
---

## 2026.10.12 - Every cluster member stored as Wang 2018 released it (genome-gate reversal)

Module: `torchcell/datasets/ecoli/fang2025.py` (`CrispriGuideFfaEnrichmentFang2025Dataset`); earlier build history is in [[torchcell.datasets.ecoli.wang2018]] and the sweep note [[database.genome-gating-sweep.2026.10.12]] (row 244).

The owner's rule of 2026-10-12: the loader reads no annotation. `reconcile_locus_tags`, `MIN_RESOLVED_FRACTION`, the `ecoli_genome` constructor parameter and the `retired_target` drop rule are gone. Every member of a guide's BLASTN cluster is a knockdown leaf keyed by the b-number and symbol Wang 2018's Supplementary Data 2 releases for it (`ClusterMember.bnumber`, `ClusterMember.symbol`), including `b4700` (`sokE`) and `b4590` (`ybfK`), which GCA_000005845.2 no longer carries: their 3 rows above the read floor (`sokEb4700_32` and `sokEb4700_33` in round 1, `ybfKb4590_11` in round 2) are records, and the two members are no longer pruned from the genotypes of the clusters that carry them beside a current b-number. The L4 genome-reference rows report them as findings (#1069). `stored_gene_names` (the annotation's symbol where it round-trips) is no longer used: the symbol is the release's own.

Pins, measured on the pinned bytes: `EXPECTED_RECORDS` 15,708 -> 15,711, `EXPECTED_PERTURBATIONS` 16,187 -> 16,191, `EXPECTED_GENES` 2,628 -> 2,630; round 1 keeps 15,380 records and round 2 331. `build_accounting.json` drops `retired_targets_dropped` and `reconciliation`; `RoundAccounting` drops `dropped_retired_target`.

Released vs stored: 111,342 released (guide, round) rows = 15,711 records + 95,631 rows below the authors' own 20-read floor (the one remaining rule, the release's own exclusion); 0 dropped on a genome ground.

### 2026.10.12 - The per-round pins follow the restored records

Job 4464 failed at `round_1_cf: wrote 15380 records, the pinned bytes give 15378`: `EXPECTED_RECORDS` moved to 15,711 with the reversal but each `Round.kept_records` still carried its retired-target subtraction. A round's kept records are now exactly its `floor_kept_rows`, 15,380 and 331, which is the 15,711 total.

## 2026.11.25 - The per-strain FFA titers, one record per biological replicate (#1178)

Module: `torchcell/datasets/ecoli/fang2025.py`, new class `FfaTiterFang2025Dataset` (`ProductTiterExperiment` / `ProductTiterExperimentReference`), adapter `torchcell/adapters/fang2025_titer_adapter.py` with conf `ffa_titer_fang2025_adapter.yaml`, verifier entry `ffa_titer_fang2025` in `torchcell/verification/runners.py`. The screen dataset is unchanged: 15,711 records, same bytes, same pins.

### What the release holds

Supplementary Data 6 (`MOESM8`) lays every quantitative panel out as `Strain | 1 | 2 | 3 | Mean | SD`. 13 of those blocks carry `FFAs (mg L-1)`: Figure 1e (47 rows), 1g (9), 1h (2), 4 (14), 5d (37), 5f (4), 5g (2), 6c (4), 7a (3), S2b (7), S3a (4), S9b (6) and S12 (5) = **144 strain rows, 432 released culture values**. The loader's earlier docstring said "13 blocks over 165 labelled rows"; 165 is wrong, the count is 144 (measured). The sweep's own scan said 8 blocks over 90 rows, which missed the Figure 5 and Figure 1g/1h blocks.

The three numbered columns are the three biological replicates behind the panel's error bar:

- `paper.md` L129 "Tube cultivations were performed in three biological replicates."
- L131 "Flask cultivations followed the same procedures as glass-tube fermentations and were performed in three biological replicates."
- L207 "Quantitative data of three biological replicates are presented as mean $\pm \mathsf { S D }$ , as indicated in legends of figures."
- L207 "No statistical method was used to predetermine sample size, and no data was excluded from the analysis." (no sample is withdrawn)

Measured on the pinned bytes, over all 144 rows: the mean of the three columns reproduces the released `Mean` to **9.095e-13** and their sample SD reproduces the released `SD` to **3.922e-12**. That is what fixes the column alignment; `read_titer_blocks` re-checks both on every build and refuses a row that does not reproduce.

### What a record is

One biological replicate culture: the released value verbatim (mg/L stored under the numerically identical `ug/mL`), `n_samples=1`, `sample_unit=biological_replicate`, `replicate_id` = the panel's own two labels for the culture, its strain and its column number (`"pcnBi 1"`), and `titer_uncertainty` / `titer_uncertainty_type` as `ProvenanceGap`s with reason `not_applicable_single_measurement`. The released Mean and SD are a derivation over the stored records and live only in `preprocess/titer_rows.csv`.

The strain half of `replicate_id` is load-bearing. The schema states a guide without the plasmid carrying it, so `pcnB_956` (pCF + pSg-pcnB_956), `pcnBi` (the single pCF-pcnBi plasmid) and `pcnBiF-` (pC + pSg-pcnB_956) all carry ONE pcnB knockdown; their titers differ, so a bare `"1"` would key three different cultures the same way.

**306 records over 102 strains**, built at `/scratch/projects/torchcell-scratch/tmp/scratch/per-replicate-fang2025/ffa_titer_fang2025` (scratch, not the shared dev tree).

### The same cultures are re-plotted, and the store holds them once

144 rows cover 116 distinct (strain, medium) pairs: CF appears in six panels and `pcnBi` in seven, with the same three numbers each time. A record is keyed by (strain, medium, replicate); every panel that released it is in the ledger; the build refuses a key whose panels disagree by more than 1e-3 mg/L. Two repeats disagree at all: Figure S9b rounds the `pcnB-` cultures Figure S3a prints in full, max **3.911e-4** mg/L, and the full-precision cell is the one stored. Accounting closes as 306 stored + 42 declined + 84 re-releases = 432.

### Genotypes come from the source's own two tables

Supplementary Data 1 states each strain as a sentence ("MG with plasmids pCF-pcnBi and pSg-acrD_2252") and Supplementary Data 2 states each plasmid's parts ("PLlacO-1: sgRNA acrD_2252, pBR322, AmpR"), so the build parses the sentence and turns each plasmid's parts into perturbations. Both tables, and the primer table (Supplementary Data 3), are now **deposited in the raw mirror** with their recorded `pmc_cloud` retrieval (`MOESM3/4/5`, re-fetched 2026-11-25 and reproducing their pins); they were quoted from the literature mirror before.

- `sgRNA <guide>` -> one `CrisprInterferencePerturbation` per member of that guide's Wang 2018 BLASTN cluster, carrying Wang's 20-mer. The panels write the short form (`acrD_2252`) and it resolves on (symbol, position): no short form of the 55,671 released ids is ambiguous, and all 86 guides of the plasmid table resolve. Every resolved spacer is re-checked against the 4-base-tagged primer Fang's own Supplementary Data 3 carries for that guide name (86 of 86).
- `PBAD: fadR` / `PT7: fadR` -> a `GeneAdditionPerturbation`, an extra native copy on its own replicon, `is_heterologous=False`, the plasmid's words in `replicon_statement`, `placement` a typed gap.
- `MG1655(DE3) ΔfadEΔpcnB` -> a `DeletionStrainPerturbation` on `b0143`, the lambda-RED deletion, no cassette asserted.

Two backgrounds: `pCF` is "PTrc: dCas9, PT7: tesA', p15A, CmR" and `pC` is the same without the thioesterase, so `CF-`, `pcnBiF-` and `pcnB-F-` carry their own `StrainBackground` and their own reference. Each record points at the released control mean of ITS background: CF 746.8201 mg/L (SD 20.4727, n = 3) for the pCF family, CF- 86.33773 (SD 21.1876) for the pC family, both read out of the same panels, both with `replicate_id=None`. The CF and CF- cultures are records too, with an empty genotype on their stated background.

The product is a mixture and is stored as one: "Total concentrations of FFAs were calculated as the sum of $\mathbf { C } _ { 1 2 }$ to ${ \sf C } _ { 1 8 }$ (saturated and monounsaturated)", so `resolved_compound("free fatty acids")` returns the name with a typed gap on the structure. `culture_format` is a typed gap: the Methods describe tube (5 mL) and flask (50 mL) fermentations in the same medium, temperature, induction and duration, and only Figure 7a's legend says which vessel it used.

### The overexpression decline was wrong

Before this change the titers were not loaded at all, the reason given being that `pcnBi-acrDi-fadR+` carries "a P_BAD-driven fadR OVEREXPRESSION that no bacterial gene-perturbation leaf types". `GeneAdditionPerturbation` is exactly that leaf ("a gene ADDED to the strain (heterologous expression or an extra native copy), carried on a plasmid"), and Supplementary Data 2 states the plasmid as `PBAD: fadR, ColE1, AmpR`, so the arm is a gene addition and needs no schema change. The inducer is where the schema says it goes: 10 mM arabinose as a `SmallMoleculePerturbation` on that strain's environment ("When the $\mathsf { P } _ { \mathsf { B A D } }$ promoter was used, $1 0 \mathrm { m M }$ arabinose was utilized to induce gene expression"). `fadRT7` is `PT7: fadR` and answers to the IPTG every record already carries.

### 42 of the 432 values are not stored, and none is a replicate of a stored strain

| reason | rows | why |
| --- | --- | --- |
| `alternate_host_no_deposited_assembly` | 10 | JM109(DE3), AD494(DE3), BL21-trxB(DE3), OrigamiB(DE3) as -CF and -pcnBi (Figures 4, S11). `BacterialReferenceStrain` holds MG1655, BW25113, W3110, REL606, KT2440; a record needs the assembly its reference pins. Needs four new reference strains with deposited assemblies. |
| `glucose_medium_not_in_the_media_layer` | 2 | CF and pcnBi on "30 g/L glucose, specifically used in Fig. 4". `torchcell.datamodels.media` holds only the glycerol recipe, and the registry note at `media.py` already anticipated this row. |
| `named_pcnB_point_allele_leaf_undecided` | 2 | `pcnBmE` (pcnB E108A) and `pcnBmN` (N138H), Figure S3a. The allele name is the whole of what tells the two strains apart, and which leaf carries a named bacterial point allele (`NamedAllelePerturbation`, documented for yeast, versus a site variant call) is an owner decision. |

Figure 4's two MG1655 glycerol rows (CF, pcnBi) are stored: their cells are identical to the ones Figures 1e/1h/6c/S9b release, so nothing of Figure 4's glycerol content is lost.

### Verification

`verify_titer_build` runs the shared `verify_product_titer_dataset` gate plus four rules of its own: the panel ledger (`panels_are_accounted`), `three_cultures_per_strain` (every strain carries cultures 1, 2 and 3 exactly once, and the strain count is the pinned 102), `titer_vs_released_source_data` (every stored titer is a released cell) and `spacer_vs_released_primer_table` (every stored spacer carries a Fang primer). All 15 levels PASS on the scratch build; the runner's L4 containment also holds, 69 of 69 perturbed b-numbers are loci of GCA_000005845.2.

### Source quirks worth remembering

- The titer panels write `folK_38/188/400`, Supplementary Data 1 writes `folk_38/188/400`; the strain lookup is case-folded.
- Supplementary Data 1's `paaG_218` row names its plasmid `pSg-aaG_218` (a dropped `p`); Supplementary Data 2 carries `pSg-paaG_218`. The parser completes that one typo from the strain's own name and refuses any other unknown plasmid.
- `pcnBi-pcnB_956` (Figure 5d) carries the SAME guide on two plasmids, so its genotype holds two identical knockdown leaves.
