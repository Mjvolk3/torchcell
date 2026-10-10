# experiments/036-dataset-fixes-before-kg-build/scripts/wangGenomeWideScreenRevealsCellular2024_release_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.wangGenomeWideScreenRevealsCellular2024_release_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/wangGenomeWideScreenRevealsCellular2024_release_inventory
"""Measure the Wang 2024 rifampicin Tn-seq release before (and against) its loader.

Reads the sha256-pinned Table S2 workbook out of the raw mirror
(``$DATA_ROOT/torchcell-raw/wangGenomeWideScreenRevealsCellular2024/``) with openpyxl and
measures, without using the loader's own reader for the counts:

1. **Shape.** Sheets, header, data rows and columns per sheet, distinct ``#Orf`` values,
   and whether the input-sample columns (``Sites``, ``Mean Ctrl``, ``Sum Ctrl``) agree
   across the six sheets, which they must since one input sample is the control of
   every comparison.
2. **Reads.** Genes with no input reads, with no post-treatment reads, and with neither,
   per sheet, counted on the ``Sum`` columns (the ``Mean`` columns are rounded to 0.1).
3. **The statistic.** How closely ``log2FC`` is matched by ``log2((Mean Exp + 1) /
   (Mean Ctrl + 1))`` on the rounded means. This is an observation about the release, NOT
   a sourced formula: the paper states only that the summed counts were compared and
   "expressed as a log2 fold change".
4. **Hit calls.** Genes passing the paper's own cut (|log2FC| > 1 and adjusted p < 0.05)
   per sheet and in the union, against the paper's "a total of 365 genes were selected".
5. **Identifiers.** How the 4,419 b-numbers resolve on MG1655 GCA_000005845.2, and the
   resulting stored-record count under the loader's two retention rules.
6. **Duplication.** Spearman correlation of each Wang condition's log2FC against the
   rifampicin records already in the dev stores of Choe 2025 (CRISPRi, MG1655 b-numbers)
   and Shiver 2016 (Keio, joined on gene symbol), per screen, on the shared genes. A
   measured r near 1 would mean the row re-serves a dataset; a different library and
   modality with a low r is an independent measurement.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/
wangGenomeWideScreenRevealsCellular2024_release_inventory.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/wangGenomeWideScreenRevealsCellular2024_release_inventory.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import os.path as osp
from collections import defaultdict
from typing import Any

import openpyxl
from dotenv import load_dotenv
from scipy.stats import spearmanr

from torchcell.datasets.bacteria_common import bacterial_genome
from torchcell.datasets.ecoli import wang2024
from torchcell.verification.runners import stream_records

OUT = osp.join(
    "experiments",
    "036-dataset-fixes-before-kg-build",
    "results",
    "wangGenomeWideScreenRevealsCellular2024_release_inventory.json",
)
COMPARE_STORES = {
    "Choe 2025 (CRISPRi, MG1655)": "crispri_chemgen_choe2025",
    "Shiver 2016 (Keio, BW25113)": "ecoli_env_chemgen_shiver2016",
}


def sha256(path: str) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sheets(path: str) -> dict[str, list[dict[str, Any]]]:
    """Each sheet's data rows as dicts keyed by the header, header located by '#Orf'."""
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out: dict[str, list[dict[str, Any]]] = {}
    for sheet in book.worksheets:
        rows = list(sheet.iter_rows(values_only=True))
        header_index = next(i for i, row in enumerate(rows) if row[0] == "#Orf")
        header = rows[header_index]
        out[sheet.title] = [
            dict(zip(header, row, strict=True)) for row in rows[header_index + 1 :]
        ]
    book.close()
    return out


def rifampicin_by_screen(store: str) -> dict[str, dict[str, float]]:
    """``{screen key: {join key: response}}`` of a dev store's rifampicin records.

    The join key is the b-number for an MG1655 store and the gene symbol otherwise; a
    gene measured more than once in one screen keeps the mean of its values.
    """
    sums: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    root = osp.join(os.environ["DATA_ROOT"], "data", "torchcell", store)
    for record in stream_records(root):
        experiment = record["experiment"]
        drugs = [
            p
            for p in experiment["environment"]["perturbations"]
            if "rifamp" in str((p.get("compound") or {}).get("name", "")).lower()
        ]
        if not drugs:
            continue
        perturbations = experiment["genotype"]["perturbations"]
        if len(perturbations) != 1:
            continue
        pert = perturbations[0]
        namespace = pert.get("gene_namespace", "")
        key = (
            pert["systematic_gene_name"]
            if "mg1655" in namespace
            else str(pert.get("perturbed_gene_name"))
        )
        dose = drugs[0]["concentration"]
        screen = (
            f"{experiment['phenotype'].get('screen_id')} "
            f"[{dose.get('value')} {dose.get('unit')}]"
        )
        sums[screen][key].append(float(experiment["phenotype"]["environment_response"]))
    return {
        screen: {k: sum(v) / len(v) for k, v in genes.items()}
        for screen, genes in sums.items()
    }


def main() -> None:
    """Measure and write the inventory JSON."""
    load_dotenv()
    path = str(wang2024.raw_mirror_dir() / "data" / wang2024.DATA_FILE)
    digest = sha256(path)
    if digest != wang2024.DATA_SHA256[wang2024.DATA_FILE]:
        raise RuntimeError(f"{path} sha256 {digest} is not the pinned one")
    sheets = read_sheets(path)
    names = list(sheets)
    first = sheets[names[0]]

    shape = {
        name: {
            "data_rows": len(rows),
            "columns": len(rows[0]),
            "distinct_orf": len({r["#Orf"] for r in rows}),
            "input_columns_equal_first_sheet": all(
                a[c] == b[c]
                for a, b in zip(first, rows, strict=True)
                for c in ("#Orf", "Name", "Sites", "Mean Ctrl", "Sum Ctrl")
            ),
            "non_numeric_value_cells": sum(
                1
                for r in rows
                for c, v in r.items()
                if c not in ("#Orf", "Name") and not isinstance(v, (int, float))
            ),
        }
        for name, rows in sheets.items()
    }
    total_cells = sum(len(rows) for rows in sheets.values())

    reads = {}
    statistic = {}
    hits: dict[str, set[str]] = {}
    for name, rows in sheets.items():
        no_input = [r for r in rows if r["Sum Ctrl"] == 0]
        reads[name] = {
            "no_input_reads": len(no_input),
            "no_post_treatment_reads": sum(1 for r in rows if r["Sum Exp"] == 0),
            "no_reads_in_either": sum(
                1 for r in rows if r["Sum Ctrl"] == 0 and r["Sum Exp"] == 0
            ),
            "no_input_reads_but_post_treatment_reads": sum(
                1 for r in no_input if r["Sum Exp"] > 0
            ),
            "log2fc_of_no_reads_rows": sorted(
                {r["log2FC"] for r in rows if r["Sum Ctrl"] == 0 and r["Sum Exp"] == 0}
            ),
        }
        errors = [
            abs(math.log2((r["Mean Exp"] + 1) / (r["Mean Ctrl"] + 1)) - r["log2FC"])
            for r in rows
        ]
        statistic[name] = {
            "within_0.011": sum(1 for e in errors if e <= 0.011),
            "within_0.1": sum(1 for e in errors if e <= 0.1),
            "max_abs_error": round(max(errors), 4),
            "n": len(errors),
        }
        hits[name] = {
            r["#Orf"] for r in rows if abs(r["log2FC"]) > 1 and r["Adj. p-value"] < 0.05
        }
    no_input_sets = [
        {r["#Orf"] for r in rows if r["Sum Ctrl"] == 0} for rows in sheets.values()
    ]

    genome = bacterial_genome("ecoli", wang2024.REFERENCE_STRAIN_NAME)
    b_numbers = [r["#Orf"] for r in first]
    kept, ledger = wang2024.resolve_b_numbers(genome, b_numbers, label="Table S2 #Orf")
    stored = sum(
        1
        for rows in sheets.values()
        for r in rows
        if r["#Orf"] in kept and not (r["Sum Ctrl"] == 0 and r["Sum Exp"] == 0)
    )

    wang_by_sheet = {
        name: {
            r["#Orf"]: float(r["log2FC"])
            for r in rows
            if r["#Orf"] in kept and not (r["Sum Ctrl"] == 0 and r["Sum Exp"] == 0)
        }
        for name, rows in sheets.items()
    }
    symbol_of = dict(kept)
    duplication: dict[str, Any] = {}
    for label, store in COMPARE_STORES.items():
        screens = rifampicin_by_screen(store)
        by_symbol = "Shiver" in label
        rows_out: list[dict[str, Any]] = []
        for screen, other in sorted(screens.items()):
            for name, wang in wang_by_sheet.items():
                keyed = (
                    {symbol_of[b]: v for b, v in wang.items()} if by_symbol else wang
                )
                shared = sorted(set(keyed) & set(other))
                rho = spearmanr([keyed[g] for g in shared], [other[g] for g in shared])
                rows_out.append(
                    {
                        "other_screen": screen,
                        "wang_sheet": name,
                        "shared_genes": len(shared),
                        "spearman_r": round(float(rho.statistic), 4),
                    }
                )
        duplication[label] = {
            "store": store,
            "rifampicin_screens": {s: len(g) for s, g in screens.items()},
            "pairs": rows_out,
            "max_abs_spearman_r": max(abs(float(r["spearman_r"])) for r in rows_out),
        }

    result = {
        "file": f"$DATA_ROOT/{wang2024.RAW_DIR_REL}/data/{wang2024.DATA_FILE}",
        "sha256": digest,
        "sheets": names,
        "shape": shape,
        "total_cells": total_cells,
        "reads": reads,
        "no_input_reads_same_genes_every_sheet": all(
            s == no_input_sets[0] for s in no_input_sets
        ),
        "statistic_vs_log2_mean_plus_one": statistic,
        "hits_per_sheet_paper_cut": {k: len(v) for k, v in hits.items()},
        "hits_union_paper_cut": len(set().union(*hits.values())),
        "identifiers": {
            "distinct": len(set(b_numbers)),
            "locus_tags_of_pinned_annotation": len(kept),
            "not_a_locus_tag": ledger.not_a_locus_tag,
            "layer_histogram": ledger.reconciliation.layer_histogram,
        },
        "stored_records": stored,
        "dropped_records": total_cells - stored,
        "duplication": duplication,
    }
    os.makedirs(osp.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(result, handle, indent=2, default=str)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "total_cells",
                    "stored_records",
                    "hits_union_paper_cut",
                    "hits_per_sheet_paper_cut",
                    "no_input_reads_same_genes_every_sheet",
                )
            },
            indent=2,
        )
    )
    for label, d in duplication.items():
        print(label, d["rifampicin_screens"], d["max_abs_spearman_r"])


if __name__ == "__main__":
    main()
