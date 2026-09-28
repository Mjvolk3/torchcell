---
id: rpkkbwtk91loupc5kxho7fk
title: Embed_compounds
desc: ''
updated: 1790380030640
created: 1790380030640
---

## 2026.09.25 - Embedding every dosed compound with every encoder

`experiments/031-env-chemgen-inhibitor-tolerance/scripts/embed_compounds.py` reads the
flattened parquet files from `flatten_records.py`
(`results/records_{vanacloig2022,hillenmeyer2008_hom,hillenmeyer2008_het}.parquet`,
`inchikey` and `compound` split on `|` for two-compound records), resolves each InChIKey
to a SMILES through the sha256-pinned `torchcell/datamodels/compound_identity_table.json`,
and runs every encoder in `torchcell.molecule.ENCODERS` ([[torchcell.molecule.encoders]])
over the union of compounds. Outputs:

- `results/embeddings/<encoder>.npz` with `inchikey` (str) and `X` (float32).
- `results/embeddings/failures.json`: per encoder, InChIKey -> error message.
- `results/embedding_coverage.csv`: one row per (dataset, encoder).
- `results/encoder_timing.md`: one row per encoder over the union.

A compound is one distinct InChIKey. A dosed name with no InChIKey (a mixture,
proprietary code or unresolved name) is not a compound here and is printed per dataset:
0 in vanacloig2022, 0 in hillenmeyer2008_hom, 1 in hillenmeyer2008_het (tunicamycin, a
homologue mixture, `RESOLVED_MIXTURE` in the identity table). A compound whose InChIKey
has no SMILES would be counted in `n_compounds` and left out of `n_with_smiles`; none
occurred: every one of the 41, 114 and 290 InChIKeys (343 in the union) has a SMILES.

Coverage (run of 2026.09.25, `embedding_coverage.csv`): all 12 encoders embed 343/343
except `unimol_v1`, which rejects 11 (0 of 41 in vanacloig2022, 11 of 114 in
hillenmeyer2008_hom, 7 of 290 in hillenmeyer2008_het; the two Hillenmeyer sets share
7) because unimol_tools cannot generate a 3D conformer and would otherwise embed
stand-in coordinates: four ionic salts (NaCl, LiCl, NaF, CoCl2: all-zero coordinates)
and seven ETKDG failures (MnCl2, ZnCl2, HgCl2, CdCl2, nitric oxide, sodium arsenite,
bleomycin). Each is named in `failures.json`.

| dataset | n_compounds | n_with_smiles | names without InChIKey | unimol_v1 n_failed |
|---|---|---|---|---|
| vanacloig2022 | 41 | 41 | 0 | 0 |
| hillenmeyer2008_hom | 114 | 114 | 0 | 11 |
| hillenmeyer2008_het | 290 | 290 | 1 | 7 |

Timing columns: `seconds_union` is the measured wall time over the union (check +
encode, model loaded), `seconds` prorates it by the dataset's share, `seconds_per_100 =
100 * seconds_union / n_embedded`. The full per-encoder table is `encoder_timing.md`;
the slowest is `unimol_v1` at 15.2 s per 100 (conformer generation runs twice, once in
`check` and once inside `get_repr`), everything else is under 0.7 s per 100.
