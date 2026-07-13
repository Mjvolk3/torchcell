#!/usr/bin/env python
# experiments/016-lian-magic-reprocess/scripts/build_guide_map.py
# [[experiments.016-lian-magic-reprocess.scripts.build_guide_map]]
"""Build the Lian 2019 MAGIC guide_id -> (modality, gene, spacer) map.

Reads the sha256-pinned reprocessing inputs (Supplementary Data 4 reference + the three
designed-guide libraries) from ``$LIAN_INPUTS`` and writes ``guide_map.tsv`` to ``$LIAN_WORK``.
Every reference guide resolves (100% join) via an exact spacer match; barcode collisions are
all within the same gene+modality (safe to collapse for gene/modality-level enrichment).
"""

import os

import pandas as pd

INP = os.environ["LIAN_INPUTS"]
WORK = os.environ["LIAN_WORK"]
REF = os.path.join(INP, "41467_2019_13621_MOESM6_ESM.xlsx")
LIBS = {
    "a": "activation_final_random linker-all.xlsx",
    "i": "interference_final_random linker-ALL.xlsx",
    "d": "deletion_final_random linker-ALL.xlsx",
}

ref = pd.read_excel(REF, header=None)
ref.columns = ["guide_id", "refseq"]
ref["refseq"] = ref["refseq"].astype(str).str.strip()
ref["mod"] = ref["guide_id"].astype(str).str.extract(r"^\d+_([aid])_")[0]


def spacer(row: pd.Series) -> str:
    """Per-modality spacer window inside the 43/44 bp reference barcode."""
    s = row["refseq"]
    if row["mod"] == "a":
        return s[20:43]  # 20 bp Cas12a DR + 23 bp spacer
    if row["mod"] == "i":
        return s[0:20]  # 20 bp spacer + scaffold
    return s  # deletion: full 44 bp window (guide + donor start)


ref["spacer"] = ref.apply(spacer, axis=1)

libmap: dict[tuple[str, str], str] = {}
for mod, fn in LIBS.items():
    df = pd.read_excel(os.path.join(INP, fn))
    df["Name"] = df["Name"].ffill()  # gene named only on the first guide of each block
    df["Sequence"] = df["Sequence"].astype(str).str.strip()
    for gene, seq in zip(df["Name"], df["Sequence"]):
        key = (mod, seq[:44]) if mod == "d" else (mod, seq)
        libmap[key] = gene


def lookup(row: pd.Series) -> str | None:
    m = row["mod"]
    key = (m, row["refseq"]) if m == "d" else (m, row["spacer"])
    return libmap.get(key)


ref["gene"] = ref.apply(lookup, axis=1)
assert ref["gene"].notna().all(), "guide_map: some guides did not resolve to a gene"
ref[["guide_id", "mod", "gene", "spacer", "refseq"]].to_csv(
    os.path.join(WORK, "guide_map.tsv"), sep="\t", index=False
)
print(f"wrote guide_map.tsv: {len(ref)} guides, all resolved")
