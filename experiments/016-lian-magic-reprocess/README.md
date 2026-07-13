# 016 — Lian 2019 MAGIC per-guide enrichment: reproducibility pipeline

Regenerates `guide_enrichment_final.tsv` (**sha256 `f9af849f97a2d460c3a6d628308491ec3966c6cc2a7f6cad130848d2bad32647`**),
the sha256-pinned input consumed by `torchcell/datasets/scerevisiae/lian2019.py`
(`CrisprMagicLian2019Dataset`), **from raw NGS**. Lian did NOT release the per-guide furfural
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

**Validation**: recovers the paper's hits — PDR1i round-3 rank 1, SLX5i round-1 rank 1,
SAP30d round-1 rank 2.

## Run

```bash
export DATA_ROOT=/scratch/projects/torchcell-scratch   # inputs + work dir derive from this
bash experiments/016-lian-magic-reprocess/scripts/reproduce.sh
# -> verifies guide_enrichment_final.tsv sha256 == f9af849f...; prints REPRODUCE_OK
```

Override `LIAN_INPUTS` / `LIAN_WORK` / `TORCHCELL_PYTHON` / `LIAN_SRA_BIN` if your paths differ.
