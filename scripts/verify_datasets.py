#!/usr/bin/env python
"""Run L0-L4 verification for a named subset of built datasets and tabulate the result.

Thin driver over the committed runners in ``torchcell.verification.runners``: it narrows each
family registry to the requested slugs, calls that family's entry point (which writes
``preprocess/verification_report.json`` into the writable dev tree), then reads the written
reports back and prints a per-dataset per-level markdown table.

Default slug list is the 11 datasets named in issue #92 (L0-structural regression from the
required ``Media.is_synthetic`` field), so the table in
``notes/torchcell.verification.runners.md`` is regenerable::

    PYTHONPATH=. python scripts/verify_datasets.py
    PYTHONPATH=. python scripts/verify_datasets.py --datasets scmd_ohya2005 --markdown table.md
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys

from torchcell.verification import runners
from torchcell.verification.report import Level, VerificationReport

# The 11 datasets issue #92 reported as L0-FAIL (media stored without is_synthetic).
ISSUE_92_DATASETS = [
    "microarray_kemmeren2014",
    "sm_microarray_sameith2015",
    "dm_microarray_sameith2015",
    "scmd_ohya2005",
    "scmd_ohnuki2018",
    "scmd_ohnuki2022",
    "carotenoid_ozaydin2013",
    "betaxanthin_cachera2023",
    "amino_acid_mulleder2016",
    "metabolite_zelezniak2018",
    "metabolite_dasilveira2014",
]

# Dataset slug -> (family entry point, registry attribute on runners or None).
# A registry-backed family runs every slug left in its dict; the single-dataset
# morphology runners take no registry.
FAMILIES: dict[str, tuple[str, str | None]] = {
    "microarray_kemmeren2014": ("run_expression", "EXPRESSION_DATASETS"),
    "sm_microarray_sameith2015": ("run_expression", "EXPRESSION_DATASETS"),
    "dm_microarray_sameith2015": ("run_expression", "EXPRESSION_DATASETS"),
    "carotenoid_ozaydin2013": ("run_visual_score", "VISUAL_SCORE_DATASETS"),
    "betaxanthin_cachera2023": ("run_metabolite", "METABOLITE_DATASETS"),
    "amino_acid_mulleder2016": ("run_metabolite", "METABOLITE_DATASETS"),
    "amino_acid_cooper2010": ("run_metabolite", "METABOLITE_DATASETS"),
    "metabolite_zelezniak2018": ("run_metabolite", "METABOLITE_DATASETS"),
    "metabolite_dasilveira2014": ("run_metabolite", "METABOLITE_DATASETS"),
    "organic_acid_yoshida2012": ("run_metabolite", "METABOLITE_DATASETS"),
    "isobutanol_screen_lopez2024": ("run_metabolite", "METABOLITE_DATASETS"),
    "isobutanol_validated_lopez2024": ("run_metabolite", "METABOLITE_DATASETS"),
    "ffa_xue2025": ("run_metabolite", "METABOLITE_DATASETS"),
    "proteome_zelezniak2018": ("run_protein", "PROTEIN_DATASETS"),
    "proteome_messner2023": ("run_protein", "PROTEIN_DATASETS"),
    "caudal_pantranscriptome2024": ("run_rnaseq", "RNASEQ_DATASETS"),
    "nadal_ribelles_perturbseq2025": ("run_rnaseq", "RNASEQ_DATASETS"),
    "scmd_ohya2005": ("run_morphology", None),
    "scmd_ohnuki2018": ("run_morphology_ohnuki", None),
    "scmd_ohnuki2022": ("run_ohnuki_morphology", None),
}

# Where each dataset's report lands (the writable dev tree), for the read-back table.
REPORT_ROOTS: dict[str, str] = {
    "scmd_ohya2005": runners.OHYA_REPORT_ROOT,
    "scmd_ohnuki2018": runners.OHNUKI_REPORT_ROOT,
    "scmd_ohnuki2022": runners.OHNUKI2022_REPORT_ROOT,
}


def _report_root(slug: str) -> str:
    """Relative dataset root whose ``preprocess/`` holds this slug's report."""
    if slug in REPORT_ROOTS:
        return REPORT_ROOTS[slug]
    return f"data/torchcell/{slug}"


def _narrow_registries(slugs: list[str]) -> None:
    """Delete every registry entry not requested, so only the requested slugs run."""
    for attr in {reg for _, reg in FAMILIES.values() if reg}:
        registry = getattr(runners, attr)
        for name in list(registry):
            if name not in slugs:
                del registry[name]


def _level_marks(report: VerificationReport) -> dict[Level, str]:
    """PASS/FAIL per level (``-`` when the verifier records no check at that level)."""
    marks: dict[Level, str] = {}
    for level in Level:
        results = [r for r in report.results if r.level is level]
        if not results:
            marks[level] = "-"
        else:
            marks[level] = "PASS" if all(r.passed for r in results) else "FAIL"
    return marks


def _n_records(report: VerificationReport) -> str:
    """Record count as the L0 check saw it."""
    for result in report.results:
        if result.level is Level.L0 and "n_records" in result.details:
            return str(result.details["n_records"])
    return "?"


def main(argv: list[str] | None = None) -> int:
    """Run the requested datasets' verification, then print the level table."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ISSUE_92_DATASETS,
        help="dataset slugs to verify (default: the 11 datasets of issue #92)",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="root holding data/torchcell/<slug>/ (default: $DATA_ROOT)",
    )
    parser.add_argument(
        "--markdown", default=None, help="also write the markdown table to this path"
    )
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = args.data_root or os.environ["DATA_ROOT"]
    slugs: list[str] = args.datasets
    unknown = [s for s in slugs if s not in FAMILIES]
    if unknown:
        parser.error(f"no verification family registered for: {unknown}")

    _narrow_registries(slugs)

    entry_points: list[str] = []
    for slug in slugs:
        entry, _ = FAMILIES[slug]
        if entry not in entry_points:
            entry_points.append(entry)

    failures: dict[str, str] = {}
    for entry in entry_points:
        print(f"===== {entry} =====", flush=True)
        getattr(runners, entry)(data_root)

    rows: list[tuple[str, ...]] = []
    for slug in slugs:
        path = osp.join(
            data_root, _report_root(slug), "preprocess", "verification_report.json"
        )
        with open(path) as handle:
            report = VerificationReport.model_validate(json.load(handle))
        marks = _level_marks(report)
        rows.append(
            (
                slug,
                _n_records(report),
                marks[Level.L0],
                marks[Level.L1],
                marks[Level.L2],
                marks[Level.L3],
                marks[Level.L4],
                "PASS" if report.passed else "FAIL",
            )
        )
        if not report.passed:
            failures[slug] = report.summary()

    header = (
        "| dataset | records | L0 | L1 | L2 | L3 | L4 | overall |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    table = header + "".join("| " + " | ".join(r) + " |\n" for r in rows)
    print()
    print(table)
    if args.markdown:
        with open(args.markdown, "w") as handle:
            handle.write(table)
        print(f"-> wrote {args.markdown}")

    for slug, summary in failures.items():
        print(f"\nFAILING REPORT {slug}:\n{summary}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
