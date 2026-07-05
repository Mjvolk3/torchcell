# torchcell/verification/runners
# [[torchcell.verification.runners]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/runners
"""Runnable L0-L4 verification over the abstract's built datasets (roadmap WS5/WS6).

This is the in-`src` home for the verification RUNNERS (the harness logic that loads
each built LMDB, runs the per-family verifier from
:mod:`torchcell.verification.{expression,morphology}`, adds the cross-source L4 checks,
and writes a ``verification_report.json`` sibling of each
``experiment_reference_index.json``). Keeping the runners in the package -- not in
``scripts/`` -- lets them be imported, tested, and reused.

Run everything::

    ~/miniconda3/envs/torchcell/bin/python -m torchcell.verification.runners

Or import and call :func:`run_expression`, :func:`run_morphology`, or :func:`run_all`.
"""

from __future__ import annotations

import os
import os.path as osp
import pickle
from typing import Any

import lmdb

from torchcell.verification.expression import (
    measured_gene_universe,
    verify_expression_dataset,
)
from torchcell.verification.levels import l4_cross_source
from torchcell.verification.morphology import (
    perturbed_gene_set,
    verify_morphology_dataset,
)
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)
from torchcell.verification.visual_score import (
    verify_visual_score_dataset,
    visual_score_gene_set,
)

# --------------------------------------------------------------------------- #
# Expression datasets (WS5)
# --------------------------------------------------------------------------- #
EXPRESSION_DATASETS: dict[str, dict[str, Any]] = {
    "dm_microarray_sameith2015": {
        "root": "data/torchcell/dm_microarray_sameith2015",
        "expected_count": 72,
        "provenance": Provenance(
            source_uri="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42536",
            citation_key="sameithHighresolutionGeneExpression2015",
            method="GEOparse GSE42536 (GPL11232), dye-swap-corrected log2(mutant/refpool)",
            page="Cell Reports, Expression Profiling; SI 'Double mutants - info'",
        ),
    },
    "sm_microarray_sameith2015": {
        "root": "data/torchcell/sm_microarray_sameith2015",
        "expected_count": 82,
        "provenance": Provenance(
            source_uri="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42536",
            citation_key="sameithHighresolutionGeneExpression2015",
            method="GEOparse GSE42536 (GPL11232), dye-swap-corrected log2(mutant/refpool)",
            page="Cell Reports, Expression Profiling; SI 'Single mutants - info'",
        ),
    },
    "microarray_kemmeren2014": {
        "root": "data/torchcell/microarray_kemmeren2014",
        "expected_count": 1450,
        "provenance": Provenance(
            source_uri="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42527",
            citation_key="kemmerenLargeScaleGeneticPerturbations2014",
            method="GEOparse GSE42527/GSE42526 (GPL11232), log2(mutant/wt-reference)",
            page="Cell 2014, Expression Profiling",
        ),
    },
}

# --------------------------------------------------------------------------- #
# Morphology dataset (WS6). The Ohya LMDB lives under the KG-build tree
# (`database/data/...`), which may be owned by the KG-build user and read-only; we READ
# it there and WRITE the report to the writable `data/torchcell/...` tree.
# --------------------------------------------------------------------------- #
OHYA_SOURCE_ROOT = "database/data/torchcell/scmd_ohya2005"
OHYA_REPORT_ROOT = "data/torchcell/scmd_ohya2005"
OHYA_EXPECTED_COUNT = 4718  # mutant records; 122 WT aggregated into the reference
OHYA_PROVENANCE = Provenance(
    source_uri="http://www.yeast.ib.k.u-tokyo.ac.jp/SCMD/ (mt4718data.tsv, wt122data.tsv)",
    citation_key="ohyaHighdimensionalLargescalePhenotyping2005a",
    method="SCMD CalMorph TSV (mt4718 mutants + wt122 wildtype)",
    page="Ohya et al. 2005 PNAS; PMID 16365294",
)
# Expression deletion datasets whose genes should be CONTAINED in Ohya's set.
DELETION_GENE_SOURCES = {
    "microarray_kemmeren2014": "data/torchcell/microarray_kemmeren2014",
    "sm_microarray_sameith2015": "data/torchcell/sm_microarray_sameith2015",
    "dm_microarray_sameith2015": "data/torchcell/dm_microarray_sameith2015",
}
# Empirically Kemmeren=0.970, SM Sameith=0.988; 0.90 leaves headroom for genes
# legitimately profiled for expression but absent from the morphology screen.
MIN_GENE_OVERLAP = 0.90


def load_records(abs_root: str) -> list[dict[str, Any]]:
    """Read every LMDB entry under ``<abs_root>/processed/lmdb``."""
    env = lmdb.open(osp.join(abs_root, "processed", "lmdb"), readonly=True, lock=False)
    records: list[dict[str, Any]] = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for _, value in cursor:
            records.append(pickle.loads(value))
    env.close()
    return records


def _write_report(report: VerificationReport, report_dir: str) -> str:
    """Write a report as ``verification_report.json`` in ``report_dir``; return path."""
    os.makedirs(report_dir, exist_ok=True)
    out = osp.join(report_dir, "verification_report.json")
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return out


def run_expression(data_root: str) -> bool:
    """Verify all expression datasets (L0-L4) and write their reports. True if all pass."""
    reports: dict[str, VerificationReport] = {}
    universes: dict[str, set[str]] = {}
    for name, spec in EXPRESSION_DATASETS.items():
        abs_root = osp.join(data_root, spec["root"])
        records = load_records(abs_root)
        report = verify_expression_dataset(
            records,
            dataset_name=name,
            provenance=spec["provenance"],
            expected_count=spec["expected_count"],
        )
        reports[name] = report
        universes[name] = measured_gene_universe(records)

    # L4: every expression dataset measures the same platform gene universe.
    ref_name = next(iter(universes))
    ref_universe = universes[ref_name]
    for name, universe in universes.items():
        if name == ref_name:
            continue
        all_genes = sorted(ref_universe | universe)
        shared = [
            (g, 1.0 if g in ref_universe else 0.0, 1.0 if g in universe else 0.0)
            for g in all_genes
        ]
        result = l4_cross_source(shared, tol=0.0).model_copy(
            update={"name": f"gene_universe_vs_{ref_name}"}
        )
        reports[name].add(result)

    all_passed = True
    for name, report in reports.items():
        report_dir = osp.join(
            data_root, EXPRESSION_DATASETS[name]["root"], "preprocess"
        )
        out = _write_report(report, report_dir)
        print(report.summary())
        print(f"  -> wrote {out}\n")
        all_passed = all_passed and report.passed
    return all_passed


def _l4_gene_containment(
    ohya_genes: set[str], other_name: str, other_genes: set[str]
) -> LevelResult:
    """L4: fraction of ``other_genes`` in Ohya's deletion set is >= MIN_GENE_OVERLAP."""
    overlap = len(other_genes & ohya_genes) / len(other_genes) if other_genes else 0.0
    return LevelResult(
        level=Level.L4,
        name=f"gene_containment_{other_name}",
        passed=overlap >= MIN_GENE_OVERLAP,
        message=(
            f"{overlap:.3f} of {other_name}'s {len(other_genes)} deletion genes "
            f"are in Ohya (>= {MIN_GENE_OVERLAP})"
        ),
        details={
            "n_other": len(other_genes),
            "n_in_ohya": len(other_genes & ohya_genes),
            "overlap": overlap,
            "missing_examples": sorted(other_genes - ohya_genes)[:20],
        },
    )


def run_morphology(data_root: str) -> bool:
    """Verify the Ohya morphology dataset (L0-L4) and write its report. True if it passes."""
    ohya_abs = osp.join(data_root, OHYA_SOURCE_ROOT)
    ohya_records = load_records(ohya_abs)
    report = verify_morphology_dataset(
        ohya_records,
        dataset_name="scmd_ohya2005",
        provenance=OHYA_PROVENANCE,
        expected_count=OHYA_EXPECTED_COUNT,
    )

    ohya_genes = perturbed_gene_set(ohya_records)
    for name, root in DELETION_GENE_SOURCES.items():
        other_abs = osp.join(data_root, root)
        if not osp.exists(osp.join(other_abs, "processed", "lmdb")):
            continue
        report.add(
            _l4_gene_containment(
                ohya_genes, name, perturbed_gene_set(load_records(other_abs))
            )
        )

    out = _write_report(report, osp.join(data_root, OHYA_REPORT_ROOT, "preprocess"))
    print(report.summary())
    print(f"  -> verified LMDB: {osp.join(ohya_abs, 'processed', 'lmdb')}")
    print(f"  -> wrote report:  {out}\n")
    return report.passed


# --------------------------------------------------------------------------- #
# Visual-score datasets (WS7)
# --------------------------------------------------------------------------- #
VISUAL_SCORE_DATASETS: dict[str, dict[str, Any]] = {
    "carotenoid_ozaydin2013": {
        "root": "data/torchcell/carotenoid_ozaydin2013",
        "provenance": Provenance(
            source_uri="https://ars.els-cdn.com/content/image/1-s2.0-S109671761200081X-mmc1.xlsx",
            citation_key="ozaydinCarotenoidbasedPhenotypicScreen2013a",
            method="Elsevier ESM xlsx; colony-color visual carotenoid screen (-5..+5)",
            page="SI Sheet 1 'Color scores of all deletions'",
        ),
    }
}


def run_visual_score(data_root: str) -> bool:
    """Verify visual-score datasets (L0-L4) and write reports. True if all pass."""
    ohya_abs = osp.join(data_root, OHYA_SOURCE_ROOT)
    ohya_genes: set[str] = set()
    if osp.exists(osp.join(ohya_abs, "processed", "lmdb")):
        ohya_genes = perturbed_gene_set(load_records(ohya_abs))

    all_passed = True
    for name, spec in VISUAL_SCORE_DATASETS.items():
        abs_root = osp.join(data_root, spec["root"])
        records = load_records(abs_root)
        report = verify_visual_score_dataset(
            records,
            dataset_name=name,
            provenance=spec["provenance"],
            expected_count=len(records),
        )
        if ohya_genes:
            report.add(
                _l4_gene_containment(
                    ohya_genes, "scmd_ohya2005", visual_score_gene_set(records)
                )
            )
        out = _write_report(report, osp.join(abs_root, "preprocess"))
        print(report.summary())
        print(f"  -> wrote {out}\n")
        all_passed = all_passed and report.passed
    return all_passed


def run_all(data_root: str) -> bool:
    """Run every dataset-family verification. True only if all pass."""
    expression_ok = run_expression(data_root)
    morphology_ok = run_morphology(data_root)
    visual_ok = run_visual_score(data_root)
    return expression_ok and morphology_ok and visual_ok


def main() -> int:
    """Verify all abstract datasets; write reports; return a shell exit code."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    all_passed = run_all(data_root)
    print("=" * 60)
    print("ALL DATASETS PASS" if all_passed else "SOME DATASETS FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
