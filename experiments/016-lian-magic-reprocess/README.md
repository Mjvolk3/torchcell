# 016 — Lian 2019 MAGIC per-guide enrichment: reproducibility pipeline

Regenerates `guide_enrichment_replicates.tsv` (**sha256 `46eaf7957ef410395177acf842ed0d36c12fd7c7f20ac5f5b2ee7dfd47d698ab`**),
the sha256-pinned input consumed by `torchcell/datasets/scerevisiae/lian2019.py`
(`CrisprMagicLian2019Dataset`, one record per guide x round x biological replicate), and its
mean-and-SD ancestor `guide_enrichment_final.tsv` (**sha256 `f9af849f97a2d460c3a6d628308491ec3966c6cc2a7f6cad130848d2bad32647`**),
**from raw NGS**. Lian did NOT release the per-guide furfural
enrichment (only the designed libraries + the reference); it is reprocessed from SRA. Because
this pipeline fully regenerates the table, the ~12 GB of downloaded fastqs are **safe to
delete** — this is the durable record of how they were used.

Background + validation: memory `[[lian2019-magic-data-availability]]`; dataset note
`[[torchcell.datasets.scerevisiae.lian2019]]`.

## Inputs (sha256-pinned, durable in the library mirror)

`$DATA_ROOT/torchcell-library/lianMultifunctionalGenomewideCRISPR2019/data/inputs/`
(checksums in `scripts/inputs.sha256`):

| file | sha256 (prefix) | role |
|---|---|---|
| `41467_2019_13621_MOESM6_ESM.xlsx` | `4e3f225a` | Supplementary Data 4 — the 100,493-guide reference |
| `activation_final_random linker-all.xlsx` | `da55d9e9` | CRISPRa designed library (Supp Data 1) |
| `interference_final_random linker-ALL.xlsx` | `4818fab9` | CRISPRi designed library (Supp Data 2) |
| `deletion_final_random linker-ALL.xlsx` | `9ab9a1be` | CRISPRd designed library (Supp Data 3) |
| `all symbols.xlsx` | `e11ffbbf` | gene-symbol list |

Raw reads: **NCBI SRA `PRJNA504483`**, 21 runs (`scripts/run_manifest.tsv`): 3 furfural
rounds × {before(untreated), after(furfural)} × 3 biological triplicates + `ecLibA/I/D`
plasmid baselines. Two runs (`SRR8293140`, `SRR8293144`) 404 on `prefetch` and are pulled
directly from ENA (`https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR829/00<lastdigit>/<run>/<run>.fastq.gz`).

## Environment

- `torchcell` conda env (pandas) — the Python steps.
- Dedicated `lian-sra` conda env: **`sra-tools` 3.4.1 + `bowtie` 1.3.1** (create with
  `conda create -n lian-sra -c bioconda -c conda-forge sra-tools bowtie samtools`).

## Method (deterministic)

1. **download** (`download.sh`) — `prefetch` + `fasterq-dump` the 21 runs; gzip.
2. **guide map** (`build_guide_map.py`) — reference `guide_id → (modality, gene, spacer)` via
   exact spacer join (100% resolve). Barcode collisions are all same-gene+same-modality.
3. **count** (`count_guides.py`) — barcode = `read[27:70]` (43 bp activation) or `read[27:71]`
   (44 bp interference/deletion), forward (offset 27, empirically scanned); exact-match to the
   reference. Mapping 74–78 % furfural / 58–74 % plasmid.
4. **enrich** (`enrichment.py`) — CPM (+1 pseudocount) per library; per round per replicate
   `log2(after/before)`; mean ± SD over the 3 triplicates.
5. **finalize** (`finalize_enrichment.py`) — add `is_control` (100 random guides/library, blank
   design Score) + `corrupted_gene` (Excel date/serial artifact) flags → `guide_enrichment_final.tsv`.
6. **per replicate** (`replicate_enrichment.py`) — the same CPM `log2(after/before)` written out
   one column per (round, replicate), `r{rnd}_rep{rep}_log2fc`, with the flags carried over →
   `guide_enrichment_replicates.tsv`, the loader's input. The mean and sample SD over the three
   emitted columns must reproduce `guide_enrichment_final.tsv` for every guide and round
   (asserted at 1e-9; measured 1.3e-15), which is what ties the per-replicate table to the
   pinned numbers earlier builds stored. Replicate labels are the authors' own SRA library
   names (`Ff1a/Ff1b` ... `3Ff3a/3Ff3b`), which is what the loader stores as `replicate_id`.

**Validation**: recovers the paper's hits — PDR1i round-3 rank 1, SLX5i round-1 rank 1,
SAP30d round-1 rank 2.

## Run

```bash
export DATA_ROOT=/scratch/projects/torchcell-scratch   # inputs + work dir derive from this
bash experiments/016-lian-magic-reprocess/scripts/reproduce.sh
# -> verifies guide_enrichment_final.tsv sha256 == f9af849f...; prints REPRODUCE_OK
```

Override `LIAN_INPUTS` / `LIAN_WORK` / `TORCHCELL_PYTHON` / `LIAN_SRA_BIN` if your paths differ.
