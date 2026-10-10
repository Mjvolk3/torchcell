# experiments/036-dataset-fixes-before-kg-build/scripts/royetHighThroughputTnSeqScreens2025_release_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.royetHighThroughputTnSeqScreens2025_release_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/royetHighThroughputTnSeqScreens2025_release_inventory

r"""What Royet 2025 released, measured off the mirrored SI bytes, and its overlap with
the Borchert 2024 KT2440 compendium.

Row 53 of the bacterial candidate table. Every number the loader
``torchcell/datasets/pputida/royet2025.py`` declares as a build oracle is measured here
first, on the sha256-pinned workbooks in the literature mirror
(``$DATA_ROOT/torchcell-library/royetHighThroughputTnSeqScreens2025/si/``):

1. **Table S3** (``si7.xlsx``): the sequenced pools and their replicate structure.
2. **Table S4** (``si8.xlsx``): the two HMM essentiality sheets, rows and state counts.
3. **Table S5** (``si9.xlsx``): the four LB-vs-metal RESAMPLING sheets and the
   all-metals summary sheet: rows, value typing, missing cells, the q <= 0.05 counts the
   paper prints, whether the summary sheet equals the four sheets, and how often the
   LB-arm ``Mean A`` differs between sheets.
4. **Identifiers**: every ``#Orf`` resolved against the KT2440 genome tier.
5. **Duplication against Borchert 2024**, by content: the compendium's media and
   condition columns searched for any metal or LB sample, its gene set intersected with
   this release's, and its mutant libraries listed.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/royetHighThroughputTnSeqScreens2025_release_inventory.py
"""

import hashlib
import json
import os
import os.path as osp
import re
from collections import Counter
from typing import Any

import openpyxl
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from torchcell.datasets.bacteria_common import (  # noqa: E402
    bacterial_genome,
    reconcile_locus_tags,
)

KEY = "royetHighThroughputTnSeqScreens2025"
DATA_ROOT = os.environ["DATA_ROOT"]
LIBRARY = osp.join(DATA_ROOT, "torchcell-library", KEY)
BORCHERT_RELEASE = osp.join(
    DATA_ROOT,
    "torchcell-raw",
    "borchertMachineLearningAnalysis2024",
    "data",
    "fModule_Metadata.xlsx",
)
BORCHERT_SOURCE_STUDIES = osp.join(
    DATA_ROOT,
    "data",
    "torchcell",
    "rbtnseq_borchert2024",
    "preprocess",
    "source_studies.json",
)
RESULTS = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")
OUT = osp.join(RESULTS, f"{KEY}_release_inventory.json")

METAL_SHEETS = ("LB-Co", "LB-Cu", "LB-Zn", "LB-Cd")
#: Table S5 column positions (0-based) in each per-metal sheet.
COL_ORF, COL_MEAN_A, COL_MEAN_B, COL_LOG2FC, COL_P, COL_Q = 0, 11, 12, 14, 15, 16
HEADER_ROWS = 4
#: The per-metal q <= 0.05 counts the Results print before the read-count cutoff.
PAPER_Q_COUNTS = {"LB-Co": 9, "LB-Cu": 14, "LB-Zn": 3, "LB-Cd": 8}
#: Patterns that would name a metal salt or a rich medium in the compendium metadata.
METAL_PATTERN = (
    r"\bCo\b|CoCl|Cobalt|Copper|CuCl|CuSO|Zinc|ZnCl|ZnSO|Cadmium|CdCl|Nickel|NiCl"
)


def sha256(path: str) -> str:
    """Hex sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pinned(relpath: str) -> str:
    """The mirrored file's path after checking its bytes against the key's manifest."""
    with open(osp.join(LIBRARY, "manifest.json")) as handle:
        manifest = json.load(handle)
    recorded = {f["path"]: f["sha256"] for f in manifest["files"]}[relpath]
    path = osp.join(LIBRARY, relpath)
    if sha256(path) != recorded:
        raise RuntimeError(f"{relpath} does not match its manifest sha256")
    return path


def rows_of(path: str, sheet: str) -> list[tuple[Any, ...]]:
    """Every row of one sheet, values only."""
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return list(book[sheet].iter_rows(values_only=True))
    finally:
        book.close()


def table_s3(path: str) -> dict[str, Any]:
    """The sequenced pools: name, replicate index, mapped reads."""
    rows = rows_of(path, "Feuil1")
    pools = [
        r
        for r in rows[3:]
        if isinstance(r[0], str) and re.fullmatch(r"\w+ #\d+", r[0].strip())
    ]
    arms: dict[str, list[str]] = {}
    for row in pools:
        arm, _, rep = str(row[0]).strip().partition(" #")
        arms.setdefault(arm, []).append(rep)
    return {
        "title": rows[0][0],
        "header": [c for c in rows[2] if c is not None],
        "n_pools": len(pools),
        "replicates_per_arm": {arm: len(reps) for arm, reps in arms.items()},
        "pools": [[str(c) if c is not None else None for c in r[:8]] for r in pools],
    }


def table_s4(path: str) -> dict[str, Any]:
    """The two HMM essentiality sheets: rows and the four-state histogram."""
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheets = book.sheetnames
    book.close()
    out: dict[str, Any] = {"title": rows_of(path, sheets[0])[0][0], "sheets": {}}
    for sheet in sheets:
        rows = rows_of(path, sheet)
        body = [r for r in rows[3:] if r[0] is not None]
        out["sheets"][sheet] = {
            "header": list(rows[2]),
            "n_genes": len(body),
            "n_unique_genes": len({r[0] for r in body}),
            "state_counts": dict(Counter(str(r[10]) for r in body)),
        }
    return out


def table_s5(path: str) -> tuple[dict[str, Any], list[str]]:
    """The four LB-vs-metal sheets and the summary sheet, measured."""
    per: dict[str, dict[str, tuple[Any, ...]]] = {}
    out: dict[str, Any] = {"sheets": {}}
    for sheet in METAL_SHEETS:
        rows = rows_of(path, sheet)
        body = [r for r in rows[HEADER_ROWS:] if r[COL_ORF] is not None]
        per[sheet] = {str(r[COL_ORF]): r for r in body}
        log2fc = [r[COL_LOG2FC] for r in body]
        q = [r[COL_Q] for r in body]
        out["sheets"][sheet] = {
            "title_cell": rows[0][0],
            "group_header": [c for c in rows[1] if c is not None],
            "header": list(rows[2]),
            "n_rows_total": len(rows),
            "n_genes": len(body),
            "n_duplicate_orfs": len(body) - len(per[sheet]),
            "log2fc_cell_types": dict(Counter(type(v).__name__ for v in log2fc)),
            "q_cell_types": dict(Counter(type(v).__name__ for v in q)),
            "n_missing_log2fc": sum(1 for v in log2fc if v in (None, "")),
            "n_missing_q": sum(1 for v in q if v in (None, "")),
            "n_q_le_0_05": sum(1 for v in q if float(v) <= 0.05),
            "paper_q_le_0_05": PAPER_Q_COUNTS[sheet],
            "log2fc_min": min(float(v) for v in log2fc),
            "log2fc_max": max(float(v) for v in log2fc),
            "n_log2fc_exactly_zero": sum(1 for v in log2fc if float(v) == 0.0),
            "n_q_exactly_one": sum(1 for v in q if float(v) == 1.0),
        }
    gene_sets = [set(v) for v in per.values()]
    out["same_genes_in_every_sheet"] = all(s == gene_sets[0] for s in gene_sets)
    first = per[METAL_SHEETS[0]]
    out["mean_a_differs_from_lb_co"] = {
        sheet: sum(
            1 for g in first if per[sheet][g][COL_MEAN_A] != first[g][COL_MEAN_A]
        )
        for sheet in METAL_SHEETS[1:]
    }
    out["hmm_lb_columns_differ_from_lb_co"] = {
        sheet: sum(1 for g in first if per[sheet][g][3:11] != first[g][3:11])
        for sheet in METAL_SHEETS[1:]
    }
    summary = rows_of(path, "Log2FC all metals")
    body = [r for r in summary[HEADER_ROWS:] if r[0] is not None]
    mismatches = 0
    for row in body:
        for index, sheet in enumerate(METAL_SHEETS):
            source = per[sheet][str(row[0])]
            if float(source[COL_LOG2FC]) != float(row[3 + 2 * index]) or float(
                source[COL_Q]
            ) != float(row[4 + 2 * index]):
                mismatches += 1
    out["summary_sheet"] = {
        "metal_header": [c for c in summary[2] if c is not None],
        "n_genes": len(body),
        "cells_disagreeing_with_per_metal_sheets": mismatches,
    }
    out["product_records"] = len(gene_sets[0]) * len(METAL_SHEETS)
    return out, sorted(gene_sets[0])


def identifiers(genes: list[str]) -> dict[str, Any]:
    """Every released ``#Orf`` against the KT2440 genome tier."""
    genome = bacterial_genome("pputida", "KT2440")
    _, reconciliation = reconcile_locus_tags(genome, pd.Series(genes), label=KEY)
    return {
        "assembly_loci": len(genome.genbank.loci),
        "reconciliation": reconciliation.model_dump(mode="json"),
    }


def borchert_overlap(genes: list[str]) -> dict[str, Any]:
    """Content overlap with the served Borchert 2024 compendium."""
    meta = pd.read_excel(BORCHERT_RELEASE, sheet_name="metadata")
    fitness = pd.read_excel(
        BORCHERT_RELEASE, sheet_name="fitness_measurements", usecols=["locusId"]
    )
    text = (
        meta[["condition_1", "condition_2", "expDesc", "media"]]
        .astype(str)
        .agg(" ".join, axis=1)
    )
    metal = text.str.contains(METAL_PATTERN, case=False, regex=True)
    lb = meta["media"].astype(str).str.contains(r"\bLB\b|Luria|lysogeny", regex=True)
    with open(BORCHERT_SOURCE_STUDIES) as handle:
        studies = json.load(handle)["studies"]
    borchert_genes = set(fitness["locusId"].astype(str))
    royet = set(genes)
    return {
        "borchert_release_sha256": sha256(BORCHERT_RELEASE),
        "n_samples": int(len(meta)),
        "media_counts": {
            str(k): int(v) for k, v in meta["media"].value_counts().items()
        },
        "mutant_libraries": sorted(meta["mutantLibrary"].astype(str).unique()),
        "n_samples_naming_a_metal": int(metal.sum()),
        "n_samples_on_lb": int(lb.sum()),
        "source_studies": sorted(studies),
        "royet_named_as_source_study": any(
            re.search("royet", json.dumps(v), re.IGNORECASE) for v in studies.values()
        ),
        "n_borchert_genes": len(borchert_genes),
        "n_royet_genes": len(royet),
        "n_shared_genes": len(royet & borchert_genes),
        "n_royet_only_genes": len(royet - borchert_genes),
        "n_shared_conditions": 0
        if int(metal.sum()) == 0 and int(lb.sum()) == 0
        else None,
    }


def main() -> None:
    """Measure the release and write the results JSON."""
    files = {
        name: pinned(f"si/{name}") for name in ("si7.xlsx", "si8.xlsx", "si9.xlsx")
    }
    s5, genes = table_s5(files["si9.xlsx"])
    result = {
        "citation_key": KEY,
        "files_sha256": {name: sha256(path) for name, path in files.items()},
        "table_s3": table_s3(files["si7.xlsx"]),
        "table_s4": table_s4(files["si8.xlsx"]),
        "table_s5": s5,
        "identifiers": identifiers(genes),
        "borchert2024_overlap": borchert_overlap(genes),
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(result, handle, indent=2, default=str)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "table_s3"}, indent=1, default=str
        )[:6000]
    )
    print(OUT)


if __name__ == "__main__":
    main()
