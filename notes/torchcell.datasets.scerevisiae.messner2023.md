---
id: zyn8xo41fsy0q63qkrvackr
title: Messner2023
desc: ''
updated: 1783832523640
created: 1783832523640
---

## 2026.07.12 - Messner 2023 genome-wide KO proteome build

`ProteomeMessner2023Dataset` (`torchcell/datasets/scerevisiae/messner2023.py`).
Messner et al. 2023, *Cell* 186:2018, doi:10.1016/j.cell.2023.03.026, PMID 37080200.
Proteomes of the whole *S. cerevisiae* (S288C) haploid MATa gene-deletion collection
(restored prototrophy) by microflow-SWATH-MS (DIA). Maps each KO strain to
`ProteinAbundancePhenotype` (WS9). **L0-L4 verified, 4,699 records.**

### Data source -- "mirror once + hash-pin" (NOT a live Mendeley dependency)

The curated protein matrix is deposited **only on Mendeley** (doi:10.17632/w8jtmnszd9.1).
We do NOT use Mendeley as a live source ([[no-mendeley-data-source]] preference). Checked
all alternatives:

- **PRIDE/ProteomeXchange PXD036062** -> actually **MassIVE MSV000090136**; only
  processed object is a **74.8 GB** raw DIA-NN precursor report (not the protein matrix).
- **Cell SI (Tables S1-S7)** -> none is the abundance matrix; SI defers to Mendeley.
- **y5k web app** (<https://y5k.bio.ed.ac.uk/>) -> R Shiny, no scriptable download.

So the matrix is Mendeley-only. User approved the **mirror-once + hash-pin** path: fetch
from Mendeley ONCE into the library mirror
(`$DATA_ROOT/torchcell-library/messnerProteomicLandscapeGenomewide2023/data/`), pin by
sha256, and have the loader read from the mirror (`download()` copies + verifies sha256;
no live Mendeley call). Files + provenance recorded in the mirror `manifest.json`:

- `yeast5k_noimpute_wide.csv` -- sha256 `69a9df05...aa1df9`, 167,754,298 B. Proteins
  (rows, UniProt `Protein.Group`) x samples (cols). Batch-corrected MaxLFQ, **no
  imputation** (measured values only; NaN dropped per strain).
- `yeast5k_metadata.csv` -- sha256 `48864282...878b`, 377,047 B. Per-sample: `Filename`
  (matrix col id), `sampletype` (`ko`|`HIS3`|`qc`), `ORF` (deleted gene), plate nr.

### Design decisions (all sourced, not guessed)

- **Value = linear MaxLFQ** (`measurement_type = "swath_ms_maxlfq_batch_corrected_quantity"`).
  Values ~340-430 confirm linear; log2 is applied only downstream in the paper's
  differential analysis.
- **noimpute over impute** -- store only actually-measured quantities; never persist an
  imputed value as if measured (honest-to-source). First strain stored 1,830/1,850.
- **Single-replicate KOs.** "Strains were not measured in replicates" (STAR Methods) ->
  per-strain `n_replicates = 1`, `protein_abundance_se = None`. (L2 `se_nonnegative`
  checks 0 values -- the correct signature.)
- **388-replicate WT reference.** Control = his3D::kanMX complemented by HIS3
  (`sampletype == HIS3`, ORF YOR202W), 388 reps across 57 batches -> per-protein
  mean + SE + n. Reference is **restricted per record to the strain's measured proteins**
  (as in the Zelezniak metabolite loader), so reference keys == experiment keys (L3
  `reference_finite` passes) and every reference value is a real WT measurement.
- **UniProt -> systematic ORF, 100%.** Matrix ids are UniProt accessions; mapped to ORFs
  via the SGD S288C GFF `protein_id=UniProtKB:` cross-refs (`build_uniprot_to_orf_map`),
  keeping the proteome joinable with the ORF-keyed Zelezniak proteome. 1,850/1,850 map
  (incl. mito `P00410` -> `Q0250`/COX2 via the `Q\d{4}` pattern); an unmapped id RAISES.
- **Duplicate strains kept per-instance.** One experiment per KO SAMPLE (4,699), not per
  ORF. The verifier's L1 `orf_uniqueness` was parameterized (`allow_duplicate_orfs=True`)
  for this dataset.
- **Background** = BY4741 MATa deletion collection, restored prototrophy (Mulleder); the
  prototrophy marker is not yet modeled as a GeneAddition (as in the Mulleder loader).
  Environment = synthetic minimal (SM) liquid, 30 C.

### Two data quirks handled (more rigorous than the source)

1. **Inconsistent ORF casing.** Metadata has `YML009c` and `YAL043C-a`; a case-sensitive
   ORF regex would have silently dropped 2 real strains. Fixed by uppercasing the deletion
   ORF (systematic names are uppercase) -> full 4,699. Gene-name parse from `Filename` is
   case-insensitive so `MRPL39` is still recovered.
2. **146 duplicated ORFs vs the paper's 145.** The raw metadata splits gene MRPL39 across
   `YML009C` and `YML009c`; case-sensitive grouping counts them as two singletons (the
   paper's 145). Normalizing case recognizes MRPL39 has **2 strains** -> 146 duplicated
   ORFs. Total strains unchanged (4,699); we are simply more rigorous than the source.

### L0-L4 verification (`torchcell/verification/runners.py` `proteome_messner2023`)

All pass: L0 structural (4,699), L1 count (4,699), L1 orf_uniqueness (4,549 ORFs, 146
multi-strain, expected), L2 value_fidelity (8,466,210 finite values), L2 se_nonnegative
(0 values), L3 reference_finite (key-matched, all 8,466,210), L3 measurement_type_consistent.

### Follow-ups

- Growth rates (`yeast5k_growthrates_byORF.csv`) + differential tables (`yeast5k_stat_DE*`)
  are separate phenotypes for later.
- Model the prototrophy-restoring marker as a `GeneAddition` (shared with Mulleder/Zelezniak).
- Consider an `impute` variant if a dense matrix is needed downstream.
