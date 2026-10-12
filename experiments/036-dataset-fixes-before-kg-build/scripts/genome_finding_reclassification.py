# experiments/036-dataset-fixes-before-kg-build/scripts/genome_finding_reclassification.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.genome_finding_reclassification]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/genome_finding_reclassification.py
"""Which stores flip from FAIL to PASS once genome-reference rules report, never gate.

Owner, 2026-10-12: "the genome is separate.. it doesn't gate anything". A store's verdict
is now :attr:`torchcell.verification.report.VerificationReport.passed`, which reads only
the ``gate`` rules; a ``report`` rule (containment, name currency, assembly pins, marker
alleles, ...) keeps its computation and is listed under the verdict.

Two inputs, both read-only on the dev tree:

``--reports``
    every mapped store's ``preprocess/verification_report*.json`` as it sits on disk,
    judged twice: under the old rule (every rule gates) and the new one (gate rules
    only). A report older than the store's ``build_manifest.json`` is marked stale.
``--sweep <json> [<json> ...]``
    the rows of a ``kg4_pre_build_verification_sweep.py`` rerun (one file per lane) (run with ``DATA_ROOT``
    pointed at an overlay, so no report lands in the shared tree): its
    ``failed_rules`` are the gate failures and ``genome_findings`` the open findings,
    so the old verdict is FAIL when either is non-empty.

A store passing every rule cannot fail when fewer rules gate, so only a FAIL under the
old rule can flip. Writes ``results/genome_finding_reclassification.json`` and prints the
markdown tables the pre-build note carries.

Usage::

    PYTHONPATH=. python experiments/036-dataset-fixes-before-kg-build/scripts/genome_finding_reclassification.py --reports
    PYTHONPATH=. python experiments/036-dataset-fixes-before-kg-build/scripts/genome_finding_reclassification.py --sweep <rerun.json>
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import os.path as osp
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

Verdict = Literal["PASS", "FAIL", "NO_REPORT", "ERROR"]

RESULTS = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))),
    "results",
    "genome_finding_reclassification.json",
)


class Reclassified(BaseModel):
    """One mapped store judged under the old and the new rule."""

    model_config = ConfigDict(extra="forbid")

    dataset_class: str
    root: str
    old_verdict: Verdict
    new_verdict: Verdict
    failing_gate_rules: list[str] = Field(default_factory=list)
    genome_findings: list[str] = Field(default_factory=list)
    reports: list[str] = Field(default_factory=list)
    report_stale: bool = False
    manifest: Literal["present", "retired", "absent"] = "absent"
    note: str | None = None

    @property
    def flips(self) -> bool:
        """FAIL under the old rule, PASS under the new one."""
        return self.old_verdict == "FAIL" and self.new_verdict == "PASS"


class Reclassification(BaseModel):
    """Every mapped store, by source of evidence."""

    model_config = ConfigDict(extra="forbid")

    source: str
    data_root: str
    measured_at: str
    stores: dict[str, Reclassified] = Field(default_factory=dict)


def mapped_roots() -> dict[str, str]:
    """Every mapped dataset class (private included) -> its default root."""
    from torchcell.knowledge_graphs.dataset_adapter_map import build_adapter_map

    roots: dict[str, str] = {}
    for cls in build_adapter_map(include_private=True):
        params = inspect.signature(cls.__init__).parameters  # type: ignore[misc]
        roots[cls.__name__] = str(params["root"].default)
    return roots


def manifest_state(preprocess: str) -> Literal["present", "retired", "absent"]:
    """Whether the store still carries its build manifest, or a retired one only."""
    if osp.isfile(osp.join(preprocess, "build_manifest.json")):
        return "present"
    if osp.isdir(preprocess) and any(
        name.startswith("build_manifest.json.unverified.")
        for name in os.listdir(preprocess)
    ):
        return "retired"
    return "absent"


def judge_reports(name: str, root: str, data_root: str) -> Reclassified:
    """Judge a store's on-disk reports under the old and the new rule."""
    from torchcell.verification.report import VerificationReport

    preprocess = osp.join(data_root, root, "preprocess")
    row = Reclassified(
        dataset_class=name,
        root=root,
        old_verdict="NO_REPORT",
        new_verdict="NO_REPORT",
        manifest=manifest_state(preprocess),
    )
    if not osp.isdir(preprocess):
        return row
    paths = sorted(
        osp.join(preprocess, f)
        for f in os.listdir(preprocess)
        if f.startswith("verification_report") and f.endswith(".json")
    )
    if not paths:
        return row
    row.reports = [osp.relpath(p, data_root) for p in paths]
    manifest = osp.join(preprocess, "build_manifest.json")
    if osp.isfile(manifest):
        built = osp.getmtime(manifest)
        row.report_stale = any(osp.getmtime(p) < built for p in paths)
    old_pass = True
    new_pass = True
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            report = VerificationReport.model_validate_json(handle.read())
        old_pass = (
            old_pass and bool(report.results) and all(r.passed for r in report.results)
        )
        new_pass = new_pass and report.passed
        row.failing_gate_rules.extend(
            f"{r.level.name} {r.name}" for r in report.gate_results if not r.passed
        )
        row.genome_findings.extend(
            f"{r.level.name} {r.name}" for r in report.open_findings
        )
    row.old_verdict = "PASS" if old_pass else "FAIL"
    row.new_verdict = "PASS" if new_pass else "FAIL"
    return row


def judge_sweep(path: str, data_root: str) -> dict[str, Reclassified]:
    """Judge a sweep rerun's rows: old FAIL when any rule failed, new by the gates.

    Every rule that did not hold (the row's ``failed_rules`` and ``genome_findings``
    together) is reclassified here by name, so the current registry decides, not the
    one the lane ran under.
    """
    from torchcell.verification.report import FindingClass, finding_class_of

    with open(path, encoding="utf-8") as handle:
        sweep = json.load(handle)
    out: dict[str, Reclassified] = {}
    for name, row in sweep["datasets"].items():
        preprocess = osp.join(data_root, row["root"], "preprocess")
        failing = [*row["failed_rules"], *row.get("genome_findings", [])]
        gate = [
            f"{r['level']} {r['name']}"
            for r in failing
            if finding_class_of(r["name"]) is FindingClass.gate
        ]
        findings = [
            f"{r['level']} {r['name']}"
            for r in failing
            if finding_class_of(r["name"]) is FindingClass.report
        ]
        if row["verdict"] in ("ERROR", "NO_VERIFIER"):
            verdict: Verdict = "ERROR" if row["verdict"] == "ERROR" else "NO_REPORT"
            out[name] = Reclassified(
                dataset_class=name,
                root=row["root"],
                old_verdict=verdict,
                new_verdict=verdict,
                manifest=manifest_state(preprocess),
                note=row.get("error"),
            )
            continue
        out[name] = Reclassified(
            dataset_class=name,
            root=row["root"],
            old_verdict="FAIL" if gate or findings else "PASS",
            new_verdict="FAIL" if gate else "PASS",
            failing_gate_rules=gate,
            genome_findings=findings,
            reports=list(row["reports"]),
            manifest=manifest_state(preprocess),
        )
    return out


def render(results: Reclassification) -> str:
    """The flip table, the still-failing table and the totals, as markdown."""
    stores = list(results.stores.values())
    lines = [
        f"Source: {results.source}; `DATA_ROOT={results.data_root}`; measured "
        f"{results.measured_at}",
        "",
    ]
    old = sum(r.old_verdict == "FAIL" for r in stores)
    new = sum(r.new_verdict == "FAIL" for r in stores)
    flipped = [r for r in stores if r.flips]
    lines.append(
        f"{len(stores)} mapped stores: FAIL {old} under every-rule gating, FAIL {new} "
        f"under gate-only; {len(flipped)} flip to PASS; "
        f"{sum(r.old_verdict in ('NO_REPORT', 'ERROR') for r in stores)} without a "
        "verdict"
    )
    lines += ["", "| Store | Flips | Genome findings | Report stale | Manifest |"]
    lines.append("|---|---|---|---|---|")
    for r in flipped:
        lines.append(
            f"| `{r.root.rsplit('/', 1)[-1]}` | FAIL -> PASS | "
            f"{', '.join(f'`{f}`' for f in r.genome_findings)} | "
            f"{'yes' if r.report_stale else 'no'} | {r.manifest} |"
        )
    still = [r for r in stores if r.new_verdict == "FAIL"]
    lines += ["", "| Store still FAIL | Failing gate rules | Genome findings |"]
    lines.append("|---|---|---|")
    for r in still:
        lines.append(
            f"| `{r.root.rsplit('/', 1)[-1]}` | "
            f"{', '.join(f'`{f}`' for f in r.failing_gate_rules)} | "
            f"{', '.join(f'`{f}`' for f in r.genome_findings) or '-'} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Judge the reports or a sweep rerun, write the results, print the tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reports", action="store_true")
    source.add_argument("--sweep", nargs="+", default=None)
    parser.add_argument("--out", default=RESULTS)
    args = parser.parse_args(argv)
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    results = Reclassification(
        source="on-disk reports"
        if args.reports
        else "sweep rerun " + ", ".join(osp.basename(p) for p in args.sweep),
        data_root=data_root,
        measured_at=datetime.now().isoformat(timespec="seconds"),
    )
    if args.reports:
        for name, root in mapped_roots().items():
            results.stores[name] = judge_reports(name, root, data_root)
    else:
        for path in args.sweep:
            results.stores.update(judge_sweep(path, data_root))
    os.makedirs(osp.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(results.model_dump_json(indent=2))
    print(render(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
