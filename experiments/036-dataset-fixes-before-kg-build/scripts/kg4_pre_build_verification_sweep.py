# experiments/036-dataset-fixes-before-kg-build/scripts/kg4_pre_build_verification_sweep.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.kg4_pre_build_verification_sweep]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/kg4_pre_build_verification_sweep
"""Run every mapped dataset's own L0-L4 verification against the dev bytes on disk.

``build_dataset_lmdb`` builds a store; it does not verify one. After the 2026.10.09
rebuild wave (slurm array 3565) ``--list-stale --include-private`` read 0 against main,
yet 115 of the 123 mapped stores carried no ``preprocess/verification_report.json``
produced from the bytes that rebuild left behind. This sweep closes that gap before
KG 4.0 is built from this tree: for each mapped dataset it resolves HOW that dataset is
verified, runs that route, and tabulates the verdict rule by rule.

Three routes exist, and the route is a property of the dataset, not a choice here:

``registry``
    the store's root is an entry in one of ``torchcell.verification.runners``' family
    registries, so the family runner owns its battery. The runner is called with its
    registry narrowed to the one dataset, which is what makes a crash in one store stop
    at that store instead of taking the family's other stores with it. The per-dataset
    L4 rules a runner adds (Ohya containment, the bacterial assembly containment) still
    run, because the runner reads those stores itself. ``run_expression`` is the one
    exception and is NOT narrowed: its L4 compares the three expression datasets'
    measured universes against each other, so the family only means anything whole.

``module``
    the loader module owns the battery (``run_verification``, ``verify_build``,
    ``verify``, ...), which is how every bacterial dataset outside the four
    bioproduction families verifies. The entry point, its argument order and its
    per-arm selector are recorded per dataset in :data:`MODULE_ROUTES`; nothing is
    guessed from the signature at runtime.

``none``
    no verifier reaches this store at all. These are listed, not silently passed.

Every route is READ-ONLY on ``processed/``: the readers in
``torchcell.verification.runners`` open each LMDB ``readonly=True, lock=False``, and the
only bytes this sweep writes are the ``preprocess/verification_report*.json`` files each
route writes by its own convention, in the dev tree. ``$DATA_ROOT/database/`` is never
touched and no store is ever rebuilt.

Each dataset runs in its own subprocess (``--one``), so an OOM or an exception in one
store cannot end the sweep, and the 12 GB Hoepfner atlas does not have to share an
address space with the 5.3 GB Borchert one.

Usage::

    python experiments/036-dataset-fixes-before-kg-build/scripts/\
kg4_pre_build_verification_sweep.py --list-routes
    python experiments/036-dataset-fixes-before-kg-build/scripts/\
kg4_pre_build_verification_sweep.py --route registry
    python experiments/036-dataset-fixes-before-kg-build/scripts/\
kg4_pre_build_verification_sweep.py --dataset ScmdOhya2005Dataset

Results land in ``experiments/036-dataset-fixes-before-kg-build/results/
kg4_pre_build_verification_sweep.json`` (``--out`` to move them); a partial run merges
into whatever is already there, so the sweep can be driven in waves.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import os.path as osp
import subprocess
import sys
import time
from typing import Any, Literal

import lmdb
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

RouteKind = Literal["registry", "module", "none"]
Verdict = Literal["PASS", "FAIL", "ERROR", "NO_VERIFIER"]

#: The runner that owns each family registry in ``torchcell.verification.runners``.
#: Keyed by the registry's module-level name so the sweep narrows the same dict the
#: runner reads.
REGISTRY_RUNNERS: dict[str, str] = {
    "EXPRESSION_DATASETS": "run_expression",
    "VISUAL_SCORE_DATASETS": "run_visual_score",
    "METABOLITE_DATASETS": "run_metabolite",
    "PROTEIN_DATASETS": "run_protein",
    "RNASEQ_DATASETS": "run_rnaseq",
    "ENVIRONMENT_RESPONSE_DATASETS": "run_environment_response",
    "FITNESS_DATASETS": "run_fitness",
    "SEGREGANT_GROWTH_DATASETS": "run_segregant_growth",
    "PRODUCT_TITER_DATASETS": "run_product_titer",
    "BACTERIAL_PROTEIN_ABUNDANCE_DATASETS": "run_bacterial_protein_abundance",
    "BACTERIAL_METABOLITE_DATASETS": "run_bacterial_metabolite",
    "BACTERIAL_PROTEIN_FOLD_CHANGE_DATASETS": "run_bacterial_protein_fold_change",
}

#: The three morphology runners keep their dataset's root in module constants rather
#: than a registry dict, so each is named here with the constant that holds its root.
MORPHOLOGY_RUNNERS: dict[str, str] = {
    "OHYA_SOURCE_ROOT": "run_morphology",
    "OHNUKI_SOURCE_ROOT": "run_morphology_ohnuki",
    "OHNUKI2022_SOURCE_ROOT": "run_ohnuki_morphology",
}

#: Runners whose L4 compares the family's datasets to EACH OTHER, so narrowing the
#: registry to one dataset would drop a rule. These run whole.
WHOLE_FAMILY_RUNNERS: frozenset[str] = frozenset({"run_expression"})

#: Argument sentinels resolved per dataset in :func:`_resolve_argument`.
DATASET_ROOT = "<DATASET_ROOT>"
DATA_ROOT = "<DATA_ROOT>"
DATASET_CLASS = "<DATASET_CLASS>"


class ModuleRoute(BaseModel):
    """A loader module's own build verifier, with the exact call that runs it.

    ``positional`` and ``keywords`` hold either a sentinel (:data:`DATASET_ROOT`,
    :data:`DATA_ROOT`, :data:`DATASET_CLASS`) or a literal string the entry point
    selects an arm with (``family="selection"``, ``name="flux_ishii2007"``). Recording
    the call rather than inferring it keeps a renamed parameter a loud failure instead
    of a dataset that quietly verified the wrong arm.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    module: str
    entry: str
    positional: tuple[str, ...] = ()
    keywords: dict[str, str] = Field(default_factory=dict)


#: Every mapped dataset whose verifier lives in its loader module, with the call that
#: runs it. Derived by reading each module's signature and CLI ``verify`` subcommand;
#: the arm selectors are the modules' own registry keys
#: (``ishii2007.DATASET_ROOTS``, ``yunus2026.DATASETS``) and ``Family`` literals.
MODULE_ROUTES: dict[str, ModuleRoute] = {
    "GeneInteractionBabu2014Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.babu2014",
        entry="run_verification",
        positional=(DATA_ROOT,),
    ),
    "GeneInteractionButland2008Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.butland2008",
        entry="run_verification",
        positional=(DATA_ROOT,),
    ),
    "RnaseqCaglar2017Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.caglar2017",
        entry="run_verification",
        positional=("rnaseq", DATA_ROOT),
    ),
    "DoublingTimeCaglar2017Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.caglar2017_doubling_time",
        entry="verify_build",
        positional=(DATASET_ROOT, DATA_ROOT),
    ),
    "CrispriChemgenChoe2025Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.choe2025",
        entry="run_verification",
        positional=(DATA_ROOT,),
    ),
    "CrispriKnockdownCui2018Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.cui2018",
        entry="verify_build",
        positional=(DATASET_ROOT, DATA_ROOT),
    ),
    "GrowthRateCampos2018Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.campos2018",
        entry="verify_build",
        positional=(DATASET_ROOT,),
        keywords={"data_root": DATA_ROOT},
    ),
    "MorphologyCampos2018Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.campos2018",
        entry="verify_morphology_build",
        positional=(DATASET_ROOT,),
        keywords={"data_root": DATA_ROOT},
    ),
    "GrowthRateChoe2019Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.choe2019_growth_rate",
        entry="verify_growth_build",
        positional=(DATASET_ROOT, DATA_ROOT),
    ),
    "TranscriptionFactorKnockoutChoe2019Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.choe2019_growth_rate",
        entry="verify_tf_build",
        positional=(DATASET_ROOT, DATA_ROOT),
    ),
    "EnvChemgenGirgis2009Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.girgis2009",
        entry="verify_build",
        positional=(DATASET_ROOT,),
        keywords={"data_root": DATA_ROOT},
    ),
    "GeneEssentialityGoodall2018Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.goodall2018",
        entry="run_verification",
        positional=(DATA_ROOT,),
    ),
    "ProteinTurnoverGupta2024Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.gupta2024",
        entry="verify_build",
        positional=(DATASET_ROOT,),
        keywords={"data_root": DATA_ROOT},
    ),
    "FluxIshii2007Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.ishii2007",
        entry="run_verification",
        positional=("flux_ishii2007", DATA_ROOT),
    ),
    "GrowthRateLamoureux2023Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.lamoureux2023_growth",
        entry="verify_build",
        positional=(DATASET_ROOT, DATA_ROOT),
    ),
    "PromoterReporterMohiuddin2022Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.mohiuddin2022",
        entry="run_verification",
        positional=(DATA_ROOT,),
    ),
    "ProteomeMori2021Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.mori2021",
        entry="verify_build",
        positional=(DATASET_ROOT, DATA_ROOT),
    ),
    "PhageRbTnseqMutalik2020Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.mutalik2020",
        entry="verify_build",
        positional=(DATASET_ROOT,),
        keywords={"data_root": DATA_ROOT},
    ),
    "CrisprPineneToleranceNiu2019Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.niu2019",
        entry="verify_build",
        positional=(DATASET_ROOT,),
        keywords={"data_root": DATA_ROOT},
    ),
    "RbTnseqPrice2018EcoliDataset": ModuleRoute(
        module="torchcell.datasets.ecoli.price2018",
        entry="verify",
        positional=(DATA_ROOT,),
    ),
    "GeneEssentialityPrice2018EcoliDataset": ModuleRoute(
        module="torchcell.datasets.ecoli.price2018",
        entry="verify_essentiality",
        positional=(DATA_ROOT,),
    ),
    "CrispriCrossRachwalski2024Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.rachwalski2024",
        entry="run_verification",
        positional=(DATA_ROOT,),
    ),
    "CrispriScreenRousset2018Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.rousset2018",
        entry="verify_build",
        positional=(DATASET_ROOT,),
        keywords={"data_root": DATA_ROOT},
    ),
    "MetabolomeSchastnaya2021Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.schastnaya2021",
        entry="verify_build",
        positional=(DATASET_ROOT,),
        keywords={"data_root": DATA_ROOT},
    ),
    "EnvChemgenShiver2016Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.shiver2016",
        entry="verify_build",
        positional=(DATASET_ROOT,),
        keywords={"data_root": DATA_ROOT},
    ),
    "ProteomeSchmidt2016Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.schmidt2016",
        entry="verify_build",
        positional=(DATASET_ROOT, DATA_ROOT),
    ),
    "ProteomeSrmSet1Schmidt2016Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.schmidt2016_srm",
        entry="verify_build",
        positional=(DATASET_ROOT, DATA_ROOT),
        keywords={"arm": DATASET_CLASS},
    ),
    "ProteomeSrmSet2Schmidt2016Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.schmidt2016_srm",
        entry="verify_build",
        positional=(DATASET_ROOT, DATA_ROOT),
        keywords={"arm": DATASET_CLASS},
    ),
    "GrowthRateSchmidt2016Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.schmidt2016_growth_rate",
        entry="verify_build",
        positional=(DATASET_ROOT, DATA_ROOT),
    ),
    "GrowthRateS23Schmidt2016Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.schmidt2016_s23_growth_rate",
        entry="verify_build",
        positional=(DATASET_ROOT, DATA_ROOT),
    ),
    "CarbonSourceTong2020Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.tong2020",
        entry="verify_build",
        positional=(DATASET_ROOT, DATA_ROOT),
    ),
    "EnvChemgenWang2015Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.wang2015",
        entry="run_verification",
        positional=(DATA_ROOT,),
    ),
    "GrowthWang2015Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.wang2015_growth",
        entry="verify_build",
        positional=(DATASET_ROOT, DATA_ROOT),
    ),
    "CrispriGuideFfaEnrichmentFang2025Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.fang2025",
        entry="verify_build",
        positional=(DATASET_ROOT,),
        keywords={"data_root": DATA_ROOT},
    ),
    "CrispriGuideFitnessWang2018Dataset": ModuleRoute(
        module="torchcell.datasets.ecoli.wang2018",
        entry="verify_build",
        positional=(DATASET_ROOT,),
        keywords={"data_root": DATA_ROOT},
    ),
    "RbTnseqBorchert2023Dataset": ModuleRoute(
        module="torchcell.datasets.pputida.borchert2023",
        entry="verify_build",
        positional=(DATASET_ROOT,),
        keywords={"data_root": DATA_ROOT},
    ),
    "RbTnseqBorchert2024Dataset": ModuleRoute(
        module="torchcell.datasets.pputida.borchert2024",
        entry="verify_build",
        positional=(DATASET_ROOT,),
        keywords={"data_root": DATA_ROOT},
    ),
    "IsoprenolToleranceLim2025Dataset": ModuleRoute(
        module="torchcell.datasets.pputida.lim2025",
        entry="run_tolerance_verification",
        positional=(DATA_ROOT,),
    ),
    "IsoprenolSelectionMenasalvas2025Dataset": ModuleRoute(
        module="torchcell.datasets.pputida.menasalvas2025",
        entry="verify_build",
        positional=(DATASET_ROOT, DATA_ROOT),
        keywords={"family": "selection"},
    ),
    "CrispriArrayYunus2026Dataset": ModuleRoute(
        module="torchcell.datasets.pputida.yunus2026",
        entry="run_verification",
        positional=("crispri_array_yunus2026", DATA_ROOT),
    ),
    "CrispriDifferentialProteomeYunus2026Dataset": ModuleRoute(
        module="torchcell.datasets.pputida.yunus2026",
        entry="run_verification",
        positional=("crispri_differential_proteome_yunus2026", DATA_ROOT),
    ),
    "CrispriKnockdownYunus2026Dataset": ModuleRoute(
        module="torchcell.datasets.pputida.yunus2026",
        entry="run_verification",
        positional=("crispri_knockdown_yunus2026", DATA_ROOT),
    ),
    "InhibitorBioscreenVolk2021Dataset": ModuleRoute(
        module="torchcell.datasets.private_torchcell.volk2021_inhibitor_bioscreen",
        entry="verify_build",
        positional=(DATASET_ROOT, DATA_ROOT),
    ),
}


class RuleOutcome(BaseModel):
    """One L-level rule's verdict, as the written report states it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    level: str
    name: str
    passed: bool
    message: str


class DatasetSweep(BaseModel):
    """One mapped dataset's route, verdict and report, measured by this sweep."""

    model_config = ConfigDict(extra="forbid")

    dataset_class: str
    root: str
    records: int
    route: RouteKind
    route_detail: str
    verdict: Verdict
    n_rules: int = 0
    failed_rules: list[RuleOutcome] = Field(default_factory=list)
    reports: list[str] = Field(default_factory=list)
    error: str | None = None
    seconds: float = 0.0


class SweepResults(BaseModel):
    """The whole sweep, one row per mapped dataset."""

    model_config = ConfigDict(extra="forbid")

    data_root: str
    git_head: str
    datasets: dict[str, DatasetSweep] = Field(default_factory=dict)

    def totals(self) -> dict[str, int]:
        """Count the rows by verdict."""
        counts: dict[str, int] = {}
        for row in self.datasets.values():
            counts[row.verdict] = counts.get(row.verdict, 0) + 1
        return counts


class RegistryRoute(BaseModel):
    """A family runner that owns a dataset's battery, and where it is registered."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runner: str
    registry: str | None
    key: str | None

    @property
    def detail(self) -> str:
        """``runner[registry:key]``, or the bare runner for a morphology constant."""
        if self.registry is None:
            return self.runner
        return f"{self.runner}[{self.registry}:{self.key}]"


def _data_root() -> str:
    """``DATA_ROOT`` from the environment; a missing one is a configuration error."""
    load_dotenv()
    return os.environ["DATA_ROOT"]


def _git_head() -> str:
    """The HEAD commit of the tree this sweep ran from."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=osp.dirname(osp.abspath(__file__)),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def mapped_datasets() -> dict[str, type]:
    """Every dataset the knowledge graph may build, private included, in map order."""
    from torchcell.knowledge_graphs.dataset_adapter_map import build_adapter_map

    return {cls.__name__: cls for cls in build_adapter_map(include_private=True)}


def dataset_root(dataset_class: type) -> str:
    """The loader's default ``root``, relative to ``DATA_ROOT``."""
    params = inspect.signature(dataset_class.__init__).parameters  # type: ignore[misc]
    return str(params["root"].default)


def registry_routes() -> dict[str, RegistryRoute]:
    """Map each registered store root to the family runner that verifies it."""
    runners = importlib.import_module("torchcell.verification.runners")
    routes: dict[str, RegistryRoute] = {}
    for registry, runner in REGISTRY_RUNNERS.items():
        for key, spec in getattr(runners, registry).items():
            routes[spec["root"]] = RegistryRoute(
                runner=runner, registry=registry, key=key
            )
    for constant, runner in MORPHOLOGY_RUNNERS.items():
        routes[getattr(runners, constant)] = RegistryRoute(
            runner=runner, registry=None, key=None
        )
    return routes


def route_of(dataset_class_name: str, root: str) -> tuple[RouteKind, str]:
    """The route that verifies one dataset, and a human-readable rendering of it."""
    registry = registry_routes()
    if root in registry:
        return "registry", registry[root].detail
    module_route = MODULE_ROUTES.get(dataset_class_name)
    if module_route is not None:
        return (
            "module",
            f"{module_route.module.rsplit('.', 1)[-1]}.{module_route.entry}",
        )
    return "none", "-"


def record_count(abs_root: str) -> int:
    """The number of entries in a store's LMDB, read-only; -1 when there is no store."""
    lmdb_dir = osp.join(abs_root, "processed", "lmdb")
    if not osp.isdir(lmdb_dir):
        return -1
    env = lmdb.open(lmdb_dir, readonly=True, lock=False)
    try:
        return int(env.stat()["entries"])
    finally:
        env.close()


def _resolve_argument(argument: str, abs_root: str, data_root: str, cls: type) -> Any:
    """Turn one recorded argument into the value the entry point takes."""
    if argument == DATASET_ROOT:
        return abs_root
    if argument == DATA_ROOT:
        return data_root
    if argument == DATASET_CLASS:
        return cls
    return argument


def run_registry_route(
    route: RegistryRoute, data_root: str
) -> None:  # pragma: no cover - drives the real stores
    """Call the family runner that owns this dataset, narrowed to it where sound.

    Narrowing replaces the runner's module-level registry with the single entry under
    verification, so one store's failure stops at that store. A runner in
    :data:`WHOLE_FAMILY_RUNNERS` is called unnarrowed because its L4 compares the
    family's datasets to each other.
    """
    runners = importlib.import_module("torchcell.verification.runners")
    if route.registry is not None and route.runner not in WHOLE_FAMILY_RUNNERS:
        whole = getattr(runners, route.registry)
        setattr(runners, route.registry, {route.key: whole[route.key]})
    getattr(runners, route.runner)(data_root)


def run_module_route(
    route: ModuleRoute, abs_root: str, data_root: str, cls: type
) -> None:  # pragma: no cover - drives the real stores
    """Call the loader module's own build verifier with its recorded arguments."""
    module = importlib.import_module(route.module)
    entry = getattr(module, route.entry)
    args = [_resolve_argument(a, abs_root, data_root, cls) for a in route.positional]
    kwargs = {
        name: _resolve_argument(value, abs_root, data_root, cls)
        for name, value in route.keywords.items()
    }
    entry(*args, **kwargs)


def written_reports(abs_root: str, since: float) -> list[str]:
    """Every ``preprocess/verification_report*.json`` written at or after ``since``."""
    preprocess = osp.join(abs_root, "preprocess")
    if not osp.isdir(preprocess):
        return []
    return sorted(
        osp.join(preprocess, name)
        for name in os.listdir(preprocess)
        if name.startswith("verification_report")
        and name.endswith(".json")
        and osp.getmtime(osp.join(preprocess, name)) >= since
    )


def read_outcomes(paths: list[str]) -> tuple[int, list[RuleOutcome], bool]:
    """Rule count, the failures, and whether every rule in every report passed."""
    from torchcell.verification.report import VerificationReport

    n_rules = 0
    failures: list[RuleOutcome] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            report = VerificationReport.model_validate_json(handle.read())
        n_rules += len(report.results)
        failures.extend(
            RuleOutcome(
                level=result.level.name,
                name=result.name,
                passed=False,
                message=result.message,
            )
            for result in report.results
            if not result.passed
        )
    return n_rules, failures, bool(paths) and not failures


def sweep_one(name: str, data_root: str) -> DatasetSweep:
    """Verify one mapped dataset through its own route and read back the verdict."""
    cls = mapped_datasets()[name]
    root = dataset_root(cls)
    abs_root = osp.join(data_root, root)
    kind, detail = route_of(name, root)
    row = DatasetSweep(
        dataset_class=name,
        root=root,
        records=record_count(abs_root),
        route=kind,
        route_detail=detail,
        verdict="NO_VERIFIER",
    )
    if kind == "none":
        return row
    started = time.time()
    try:
        if kind == "registry":
            run_registry_route(registry_routes()[root], data_root)
        else:
            run_module_route(MODULE_ROUTES[name], abs_root, data_root, cls)
    except BaseException as exc:  # noqa: BLE001 - the sweep reports, it does not hide
        row.verdict = "ERROR"
        row.error = f"{type(exc).__name__}: {exc}"
        row.seconds = round(time.time() - started, 1)
        return row
    row.seconds = round(time.time() - started, 1)
    row.reports = written_reports(abs_root, started)
    n_rules, failures, passed = read_outcomes(row.reports)
    row.n_rules = n_rules
    row.failed_rules = failures
    row.verdict = "PASS" if passed else "FAIL"
    if not row.reports:
        row.verdict = "ERROR"
        row.error = "the route wrote no verification report"
    return row


def _short_report(paths: list[str], data_root: str) -> str:
    """The written reports as ``<store>/preprocess/<file>``, relative to ``DATA_ROOT``."""
    if not paths:
        return "-"
    return ", ".join(f"`{osp.relpath(p, data_root)}`" for p in paths)


def render_tables(results: SweepResults) -> str:
    """The sweep as the three markdown tables the dendron note carries.

    The note quotes this output rather than restating it by hand, so every row,
    count and report path in the note is the one this sweep measured.
    """
    lines: list[str] = []
    totals = results.totals()
    verified = [r for r in results.datasets.values() if r.route != "none"]
    lines.append(f"HEAD `{results.git_head[:10]}`, `DATA_ROOT={results.data_root}`")
    lines.append("")
    lines.append(
        f"{len(results.datasets)} mapped datasets: "
        + ", ".join(f"{k} {totals[k]}" for k in sorted(totals))
    )
    lines.append("")
    lines.append("| Store | Records | Route | Result | Report |")
    lines.append("|---|---:|---|---|---|")
    for row in verified:
        failed = (
            "PASS"
            if row.verdict == "PASS"
            else (
                f"FAIL: {', '.join(f'{f.level} {f.name}' for f in row.failed_rules)}"
                if row.verdict == "FAIL"
                else f"ERROR: {row.error}"
            )
        )
        lines.append(
            f"| `{row.root.rsplit('/', 1)[-1]}` | {row.records} | "
            f"`{row.route_detail}` | {failed} ({row.n_rules} rules) | "
            f"{_short_report(row.reports, results.data_root)} |"
        )
    lines.append("")
    lines.append("### Datasets with no verifier")
    lines.append("")
    lines.append("| Store | Dataset class | Records |")
    lines.append("|---|---|---:|")
    for row in results.datasets.values():
        if row.route == "none":
            lines.append(
                f"| `{row.root.rsplit('/', 1)[-1]}` | `{row.dataset_class}` | "
                f"{row.records} |"
            )
    lines.append("")
    lines.append("### Failing rules")
    lines.append("")
    failing = [r for r in verified if r.failed_rules]
    if not failing:
        lines.append("No rule failed.")
    else:
        lines.append("| Store | Level | Rule | Message |")
        lines.append("|---|---|---|---|")
        for row in failing:
            for rule in row.failed_rules:
                message = rule.message.replace("|", "\\|").replace("\n", " ")
                lines.append(
                    f"| `{row.root.rsplit('/', 1)[-1]}` | {rule.level} | "
                    f"`{rule.name}` | {message} |"
                )
    return "\n".join(lines)


def default_out() -> str:
    """The sweep's results file inside this experiment's ``results/`` directory."""
    here = osp.dirname(osp.abspath(__file__))
    return osp.join(
        osp.dirname(here), "results", "kg4_pre_build_verification_sweep.json"
    )


def load_results(path: str, data_root: str) -> SweepResults:
    """The results file, or a fresh one; a partial run merges into what is there."""
    if not osp.isfile(path):
        return SweepResults(data_root=data_root, git_head=_git_head())
    with open(path, encoding="utf-8") as handle:
        return SweepResults.model_validate_json(handle.read())


def save_results(results: SweepResults, path: str) -> None:
    """Write the results file, dataset rows in adapter-map order."""
    os.makedirs(osp.dirname(path), exist_ok=True)
    order = list(mapped_datasets())
    results.datasets = {
        name: results.datasets[name] for name in order if name in results.datasets
    }
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(results.model_dump_json(indent=2))


def selected(args: argparse.Namespace, data_root: str) -> list[str]:
    """The dataset classes this invocation sweeps, in adapter-map order."""
    names = list(mapped_datasets())
    if args.dataset:
        unknown = sorted(set(args.dataset) - set(names))
        if unknown:
            raise KeyError(f"not mapped datasets: {', '.join(unknown)}")
        return [n for n in names if n in set(args.dataset)]
    if args.route:
        return [
            n
            for n in names
            if route_of(n, dataset_root(mapped_datasets()[n]))[0] == args.route
        ]
    return names


def print_routes(data_root: str) -> None:
    """One line per mapped dataset: route, records and store root."""
    classes = mapped_datasets()
    counts: dict[str, int] = {}
    for name, cls in classes.items():
        root = dataset_root(cls)
        kind, detail = route_of(name, root)
        counts[kind] = counts.get(kind, 0) + 1
        records = record_count(osp.join(data_root, root))
        print(f"{name:46s} {kind:9s} {records:>9d}  {detail}")
    print(
        f"\n{len(classes)} mapped datasets: "
        + ", ".join(f"{kind}={counts[kind]}" for kind in sorted(counts))
    )


def main(argv: list[str] | None = None) -> int:
    """Sweep the selected datasets in subprocesses and merge their rows into the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--route", choices=["registry", "module", "none"], default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--list-routes", action="store_true", help="print the route table and exit"
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="render the results file as the dendron note's markdown tables and exit",
    )
    parser.add_argument(
        "--merge",
        action="append",
        default=[],
        help="merge another results file's rows into --out and exit (repeatable)",
    )
    parser.add_argument(
        "--one",
        default=None,
        help="verify exactly this dataset IN THIS PROCESS and print its row as JSON "
        "(what the parent sweep runs per dataset)",
    )
    args = parser.parse_args(argv)
    data_root = _data_root()

    if args.list_routes:
        print_routes(data_root)
        return 0

    if args.one:
        print("<<<ROW>>>" + sweep_one(args.one, data_root).model_dump_json())
        return 0

    out = args.out or default_out()

    if args.table:
        print(render_tables(load_results(out, data_root)))
        return 0

    if args.merge:
        merged = load_results(out, data_root)
        for path in args.merge:
            for name, row in load_results(path, data_root).datasets.items():
                merged.datasets[name] = row
        save_results(merged, out)
        print(json.dumps(merged.totals(), indent=2))
        return 0

    results = load_results(out, data_root)
    names = selected(args, data_root)
    print(f"sweeping {len(names)} datasets -> {out}")
    for index, name in enumerate(names, start=1):
        completed = subprocess.run(
            [sys.executable, osp.abspath(__file__), "--one", name],
            capture_output=True,
            text=True,
        )
        marker = "<<<ROW>>>"
        rows = [
            line[len(marker) :]
            for line in completed.stdout.splitlines()
            if line.startswith(marker)
        ]
        if rows:
            row = DatasetSweep.model_validate_json(rows[-1])
        else:
            cls = mapped_datasets()[name]
            root = dataset_root(cls)
            kind, detail = route_of(name, root)
            row = DatasetSweep(
                dataset_class=name,
                root=root,
                records=record_count(osp.join(data_root, root)),
                route=kind,
                route_detail=detail,
                verdict="ERROR",
                error=f"subprocess exited {completed.returncode} without a row; "
                f"stderr tail: {completed.stderr.strip()[-400:]}",
            )
        results.datasets[name] = row
        save_results(results, out)
        print(
            f"[{index}/{len(names)}] {name} {row.verdict} "
            f"rules={row.n_rules} failed={len(row.failed_rules)} {row.seconds}s"
        )
    print(json.dumps(results.totals(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
