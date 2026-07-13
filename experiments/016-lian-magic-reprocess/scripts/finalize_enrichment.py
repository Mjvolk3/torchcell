#!/usr/bin/env python
# experiments/016-lian-magic-reprocess/scripts/finalize_enrichment.py
# [[experiments.016-lian-magic-reprocess.scripts.finalize_enrichment]]
"""Add control + corrupted-gene flags to produce the loader input guide_enrichment_final.tsv.

Negative controls = the 100 random guides per library (blank design Score). Corrupted-gene
rows = an Excel date/serial artifact present identically in the reference + library (a gene
whose name Excel mangled). Reads ``$LIAN_WORK/guide_enrichment.tsv`` -> writes
``guide_enrichment_final.tsv`` (sha256 f9af849f...). See ``[[lian2019-magic-data-availability]]``.
"""

import os

import pandas as pd

INP = os.environ["LIAN_INPUTS"]
WORK = os.environ["LIAN_WORK"]
LIBS = [
    ("a", "activation_final_random linker-all.xlsx"),
    ("i", "interference_final_random linker-ALL.xlsx"),
    ("d", "deletion_final_random linker-ALL.xlsx"),
]

ctrl: set[tuple[str, str]] = set()
for mod, fn in LIBS:
    df = pd.read_excel(os.path.join(INP, fn))
    for seq in df[df["Score"].isna()]["Sequence"].astype(str).str.strip():
        ctrl.add((mod, seq if mod != "d" else seq[:44]))

enr = pd.read_csv(os.path.join(WORK, "guide_enrichment.tsv"), sep="\t")
enr["gene"] = enr["gene"].astype(str)
enr["is_control"] = enr.apply(
    lambda r: (
        (r["mod"], r["spacer"] if r["mod"] != "d" else str(r["refseq"])[:44]) in ctrl
    ),
    axis=1,
)
enr["corrupted_gene"] = enr["gene"].str.match(r"^\d{4}-\d{2}-\d{2}") | enr[
    "gene"
].str.fullmatch(r"\d{6,}")
cols = [
    "guide_id",
    "mod",
    "gene",
    "spacer",
    "is_control",
    "corrupted_gene",
    "r1_log2fc_mean",
    "r1_log2fc_sd",
    "r2_log2fc_mean",
    "r2_log2fc_sd",
    "r3_log2fc_mean",
    "r3_log2fc_sd",
]
enr[cols].to_csv(
    os.path.join(WORK, "guide_enrichment_final.tsv"), sep="\t", index=False
)
print(
    f"wrote guide_enrichment_final.tsv: {len(enr)} guides, "
    f"{int(enr['is_control'].sum())} controls, {int(enr['corrupted_gene'].sum())} corrupted"
)
