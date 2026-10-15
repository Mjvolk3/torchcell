# torchcell/datasets/pputida/borchert2023_contrast
# [[torchcell.datasets.pputida.borchert2023_contrast]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/pputida/borchert2023_contrast
# Test file: tests/torchcell/datasets/pputida/test_borchert2023_contrast.py
"""Borchert 2023: the 64,853 released two-arm fitness contrasts, with their statistics.

Borchert, Bleem and Beckham 2023 (Metab Eng 77:208-218, doi:10.1016/j.ymben.2023.04.007,
citation key ``borchertRBTnSeqIdentifiesGenetic2023``, issue #913) release 13 comparison
sheets in Supplementary File 1. Each row is one gene in one comparison of a condition
arm against a reference arm: the three replicate fitnesses and the mean of each arm,
then the ``t-statistic``, ``p-value``, ``q-value`` and ``adjusted_q-value`` of the
two-sample t test between the arms. The sibling :mod:`borchert2023` (with the Borchert
2024 compendium) stores the per-culture fitness; nothing stored the test until now.

WHAT IS STORED. One ``FitnessContrastExperiment`` per released (gene, sheet)
row, 64,853 in all: the genotype is the released ``old_locus_tag`` (and ``gene_name``,
or the tag where the release gives no name) as a transposon insertion in
``Putida_ML5_JBEI``; the environment is the condition arm; the reference's environment
is the reference arm; the phenotype is a ``FitnessContrastPhenotype`` carrying the four
released numbers verbatim, the test the Methods name, the pFDR correction behind
``q-value`` and the monotonicity adjustment behind ``adjusted_q-value``. 582 released
``q-value`` cells exceed 1 and are stored as released. The reference's statistics are
typed gaps: a reference arm tests nothing.

A CONTRAST POOLS THREE CULTURES PER ARM. Table S4 releases each replicate culture's
growth duration, and the three replicates of an arm differ, so the contrast's
environments carry no ``duration_hours`` (a typed gap naming the three released
durations) rather than one replicate's.

FINDING, measured and not acted on: on all 64,853 rows the released t-statistic has the
sign of (first-named arm mean minus second-named arm mean), so it is positive where the
condition arm's fitness is lower. The Methods do not state the orientation.

NOT STORED here: the per-replicate fitness and the ``_mean`` columns (the sibling
datasets hold the replicates; the means are measured derived from them), and the
``new_locus_tag`` and ``description`` columns, which are annotation the release copies
in beside the released identity.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Final

from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
    FitnessContrastExperiment,
    FitnessContrastExperimentReference,
    FitnessContrastPhenotype,
    Genotype,
    Publication,
    TransposonInsertionPerturbation,
)
from torchcell.datasets.bacteria_common import assembly_reference
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.pputida import borchert2023 as b23
from torchcell.datasets.pputida import borchert2024 as b24
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    audit_sourced_value,
)

log = logging.getLogger(__name__)

CITATION_KEY: Final = "borchertRBTnSeqIdentifiesGenetic2023"
DATASET_ROOT_REL: Final = "data/torchcell/fitness_contrast_borchert2023"
REFERENCE_STRAIN: Final = "KT2440"
GENE_NAMESPACE: Final = "pputida_kt2440_locus_tag"

MEASUREMENT_TYPE: Final = "rbtnseq_gene_fitness_two_sample_t_test"
TEST_NAME: Final = "two-sample t test"
Q_VALUE_METHOD: Final = "positive_false_discovery_rate"
Q_VALUE_ADJUSTMENT_METHOD: Final = "monotonicity"
N_REPLICATES: Final = 3

#: Build oracles, measured on the pinned workbook.
EXPECTED_RECORDS: Final = 64853
EXPECTED_Q_ABOVE_ONE: Final = 582

SOURCED_VALUES: Final = {
    "significance_test": b23.SOURCED_VALUES["significance_test"],
    "adjusted_q": b23.SOURCED_VALUES["adjusted_q"],
    "triplicate_filter": b23.SOURCED_VALUES["triplicate_filter"],
}

_GAP_NOTE: Final = (
    "the reference arm of a contrast is not itself tested; the release states the "
    "statistics of the contrast only"
)
REFERENCE_GAPS: Final = tuple(
    ProvenanceGap(
        field=field,
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=b23._RELEASE,
        note=_GAP_NOTE,
    )
    for field in ("t_statistic", "p_value", "q_value")
)


# --------------------------------------------------------------------------- #
# Reading the released rows
# --------------------------------------------------------------------------- #
class ContrastRow(BaseModel):
    """One released (gene, comparison) row: its identity and its four statistics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sheet: str
    old_locus_tag: str
    gene_name: str | None
    t_statistic: float
    p_value: float
    q_value: float
    q_value_adjusted: float
    first_mean: float
    second_mean: float


def read_contrast_rows(path: str | Path) -> list[ContrastRow]:
    """Every row of the 13 comparison sheets, after ``b23.read_comparisons``'s checks.

    ``read_comparisons`` refuses any sheet list, header or empty cell that is not the
    pinned one; this reads the released ``gene_name`` column beside it, which that
    reader does not keep.
    """
    import openpyxl

    comparisons = b23.read_comparisons(path)
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows: list[ContrastRow] = []
    for comparison, plan in zip(comparisons, b23.COMPARISONS, strict=True):
        body = list(workbook[plan.sheet].iter_rows(min_row=2, values_only=True))
        if len(body) != len(comparison.tags):
            raise ValueError(
                f"{plan.sheet}: {len(body)} rows, parsed {len(comparison.tags)}"
            )
        reference = b23.COLUMNS_BY_EXPERIMENT[plan.reference_experiment].prefix
        condition = b23.COLUMNS_BY_EXPERIMENT[plan.condition_experiment].prefix
        first = comparison.column(f"{reference}_mean")
        second = comparison.column(f"{condition}_mean")
        for i, (cells, tag) in enumerate(zip(body, comparison.tags, strict=True)):
            if str(cells[0]) != tag:
                raise ValueError(
                    f"{plan.sheet} row {i + 2}: tag {cells[0]!r} != {tag!r}"
                )
            t, p, q, adjusted = (float(v) for v in comparison.stats[i])
            rows.append(
                ContrastRow(
                    sheet=plan.sheet,
                    old_locus_tag=tag,
                    gene_name=None if cells[2] is None else str(cells[2]),
                    t_statistic=t,
                    p_value=p,
                    q_value=q,
                    q_value_adjusted=adjusted,
                    first_mean=float(first[i]),
                    second_mean=float(second[i]),
                )
            )
    workbook.close()
    return rows


# --------------------------------------------------------------------------- #
# Records (pure)
# --------------------------------------------------------------------------- #
def arm_environment(experiment: int) -> Environment:
    """An arm's environment with the replicates' differing durations as a typed gap."""
    columns = b23.COLUMNS_BY_EXPERIMENT[experiment]
    per_replicate = b23.build_environment(columns, b23.REPLICATES[0])
    durations = b23.LAYOUT_BY_EXPERIMENT[experiment].duration
    return Environment(
        media=per_replicate.media,
        temperature=per_replicate.temperature,
        perturbations=per_replicate.perturbations,
        aerobicity=per_replicate.aerobicity,
        duration_hours=None,
        provenance_gaps=[
            *per_replicate.provenance_gaps,
            ProvenanceGap(
                field="duration_hours",
                reason=ProvenanceGapReason.not_reported_by_primary,
                looked_in=b23._SI2,
                note="the contrast pools the arm's three replicate cultures, whose "
                f"Table S4 durations are {', '.join(durations)}; no single duration is "
                "the contrast's",
            ),
        ],
    )


def phenotype(row: ContrastRow) -> FitnessContrastPhenotype:
    """One released contrast row, verbatim."""
    return FitnessContrastPhenotype(
        measurement_type=MEASUREMENT_TYPE,
        test_name=TEST_NAME,
        contrast_label=row.sheet,
        t_statistic=row.t_statistic,
        p_value=row.p_value,
        q_value=row.q_value,
        q_value_method=Q_VALUE_METHOD,
        q_value_adjusted=row.q_value_adjusted,
        q_value_adjustment_method=Q_VALUE_ADJUSTMENT_METHOD,
        n_replicates_condition=N_REPLICATES,
        n_replicates_reference=N_REPLICATES,
    )


def reference_phenotype(sheet: str) -> FitnessContrastPhenotype:
    """The reference arm: tests nothing, so its statistics are typed gaps."""
    return FitnessContrastPhenotype(
        measurement_type=MEASUREMENT_TYPE,
        test_name=TEST_NAME,
        contrast_label=sheet,
        t_statistic=None,
        p_value=None,
        q_value=None,
        n_replicates_condition=N_REPLICATES,
        n_replicates_reference=N_REPLICATES,
        provenance_gaps=list(REFERENCE_GAPS),
    )


def genotype(row: ContrastRow) -> Genotype:
    """The released locus as a transposon insertion of the ML5 library."""
    return Genotype(
        perturbations=[
            TransposonInsertionPerturbation(
                systematic_gene_name=row.old_locus_tag,
                perturbed_gene_name=row.gene_name or row.old_locus_tag,
                gene_namespace=GENE_NAMESPACE,
                transposon=b24.TRANSPOSON,
                library_pool=b23.MUTANT_LIBRARY,
            )
        ]
    )


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class FitnessContrastBorchert2023Dataset(ExperimentDataset):
    """Borchert 2023's 64,853 released two-arm fitness contrasts."""

    organism = Organism.pputida

    REFERENCE_STRAIN: ClassVar[str] = REFERENCE_STRAIN

    def __init__(
        self,
        root: str = DATASET_ROOT_REL,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize at the dev-tree root."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return FitnessContrastExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return FitnessContrastExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Supplementary File 1."""
        return [b23.DATA_FILE]

    def download(self) -> None:
        """Link the mirrored workbook after checking the manifest pin and sha256."""
        data_root = os.environ["DATA_ROOT"]
        recorded = b23.manifest_sha256(b23.load_manifest(data_root))
        check_manifest_pin(b23.DATA_RELPATH, recorded, b23.DATA_SHA256)
        os.makedirs(self.raw_dir, exist_ok=True)
        link_verified(
            b23.raw_mirror_dir(data_root) / b23.DATA_RELPATH,
            osp.join(self.raw_dir, b23.DATA_FILE),
            b23.DATA_SHA256,
        )

    @post_process
    def process(self) -> None:
        """One record per released (gene, comparison) row."""
        verify_raw_files(self.raw_dir, {b23.DATA_FILE: b23.DATA_SHA256})
        rows = read_contrast_rows(osp.join(self.raw_dir, b23.DATA_FILE))
        if len(rows) != EXPECTED_RECORDS:
            raise RuntimeError(f"{len(rows)} rows, expected {EXPECTED_RECORDS}")
        genome_reference = assembly_reference(REFERENCE_STRAIN)
        publication = Publication(doi=b23.DOI, doi_url=f"https://doi.org/{b23.DOI}")
        plans = {plan.sheet: plan for plan in b23.COMPARISONS}
        environments = {
            plan.sheet: (
                arm_environment(plan.condition_experiment),
                arm_environment(plan.reference_experiment),
            )
            for plan in b23.COMPARISONS
        }
        references = {
            sheet: FitnessContrastExperimentReference(
                dataset_name=self.name,
                genome_reference=genome_reference,
                environment_reference=reference_env,
                phenotype_reference=reference_phenotype(sheet),
            )
            for sheet, (_, reference_env) in environments.items()
        }
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, row in enumerate(tqdm(rows, desc=self.name)):
                experiment = FitnessContrastExperiment(
                    dataset_name=self.name,
                    genotype=genotype(row),
                    environment=environments[plans[row.sheet].sheet][0],
                    phenotype=phenotype(row),
                )
                txn.put(
                    f"{index}".encode(),
                    self._intern_record(
                        experiment, references[row.sheet], publication, itxn
                    ),
                )
        env.close()
        interned_env.close()
        self._write_ledgers(rows)
        log.info("Borchert 2023 contrasts: %d records", len(rows))

    def _write_ledgers(self, rows: Sequence[ContrastRow]) -> None:
        """Per-sheet counts, the q > 1 count and the t-sign finding."""
        out = Path(self.preprocess_dir)
        out.mkdir(parents=True, exist_ok=True)
        per_sheet: dict[str, int] = {}
        for row in rows:
            per_sheet[row.sheet] = per_sheet.get(row.sheet, 0) + 1
        (out / "contrasts.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    "records": len(rows),
                    "per_sheet": per_sheet,
                    "q_value_above_one": sum(r.q_value > 1.0 for r in rows),
                    "t_sign_is_first_minus_second": sum(sign_agrees(r) for r in rows),
                    "rows_without_gene_name": sum(r.gene_name is None for r in rows),
                },
                indent=2,
            )
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process()."""
        raise NotImplementedError("built in process()")


def sign_agrees(row: ContrastRow) -> bool:
    """The released t has the sign of (first-named arm mean minus second-named)."""
    difference = row.first_mean - row.second_mean
    return (row.t_statistic > 0) == (difference > 0)


# --------------------------------------------------------------------------- #
# L0-L4 verification
# --------------------------------------------------------------------------- #
Record = Mapping[str, Any]

VERIFIER_PROVENANCE: Final = Provenance(
    source_uri=f"$DATA_ROOT/{b23.RAW_DIR_REL}/{b23.DATA_RELPATH}",
    citation_key=CITATION_KEY,
    sha256=b23.DATA_SHA256,
    method="one FitnessContrastExperiment per released (gene, comparison) row "
    "of the 13 comparison sheets: t, p, pFDR q and monotonicity-adjusted q verbatim",
    page="comparison sheets",
)

_STATS: Final = ("t_statistic", "p_value", "q_value", "q_value_adjusted")


def _l1_unique(records: Sequence[Record]) -> LevelResult:
    keys = {
        (
            r["experiment"]["phenotype"]["contrast_label"],
            r["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"],
        )
        for r in records
    }
    return LevelResult(
        level=Level.L1,
        name="one_record_per_released_row",
        passed=len(keys) == len(records),
        message=f"{len(keys)} distinct (sheet, locus) keys over {len(records)} records",
    )


def _l2_values_equal_release(
    records: Sequence[Record], rows: Sequence[ContrastRow]
) -> LevelResult:
    released = {(row.sheet, row.old_locus_tag): row for row in rows}
    mismatched: list[str] = []
    for r in records:
        phenotype = r["experiment"]["phenotype"]
        tag = r["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
        row = released[(phenotype["contrast_label"], tag)]
        if any(phenotype[name] != getattr(row, name) for name in _STATS):
            mismatched.append(f"{row.sheet}:{tag}")
    return LevelResult(
        level=Level.L2,
        name="value_fidelity_to_release",
        passed=not mismatched and len(records) == len(rows),
        message=f"{len(records) - len(mismatched)} of {len(rows)} released rows stored "
        "with their four statistics bit for bit",
        details={"mismatched": mismatched[:20]},
    )


def _l3_contract(records: Sequence[Record]) -> LevelResult:
    bad_reference = sum(
        r["reference"]["phenotype_reference"]["t_statistic"] is not None
        for r in records
    )
    methods = {
        (
            r["experiment"]["phenotype"]["q_value_method"],
            r["experiment"]["phenotype"]["q_value_adjustment_method"],
        )
        for r in records
    }
    q_above_one = sum(r["experiment"]["phenotype"]["q_value"] > 1.0 for r in records)
    passed = (
        bad_reference == 0
        and methods == {(Q_VALUE_METHOD, Q_VALUE_ADJUSTMENT_METHOD)}
        and q_above_one == EXPECTED_Q_ABOVE_ONE
    )
    return LevelResult(
        level=Level.L3,
        name="contrast_contract",
        passed=passed,
        message=f"{bad_reference} references carry a statistic; corrections {sorted(methods)}; "
        f"{q_above_one} released q-values above 1, {EXPECTED_Q_ABOVE_ONE} declared",
        details={"q_value_above_one": q_above_one},
    )


def _l4_tags_in_kt2440(records: Sequence[Record], universe: set[str]) -> LevelResult:
    tags = {
        r["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
        for r in records
    }
    missing = sorted(tags - universe)
    return LevelResult(
        level=Level.L4,
        name="released_tags_on_kt2440_annotation",
        passed=True,
        message=f"{len(tags) - len(missing)} of {len(tags)} released tags are KT2440 "
        "gene rows (a finding: the genome never gates storage)",
        details={"missing": missing[:50], "n_missing": len(missing)},
    )


def verify_build(dataset_root: str, data_root: str | None = None) -> VerificationReport:
    """L0-L4 over the built store; writes ``preprocess/verification_report.json``."""
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import BACTERIAL_ASSEMBLY_SETS, ExperimentType
    from torchcell.verification.levels import l0_structural, l1_count
    from torchcell.verification.runners import (
        BACTERIAL_GENE_ASSEMBLIES,
        _bacterial_gene_set,
        load_records,
    )

    base = data_root or os.environ["DATA_ROOT"]
    records = load_records(dataset_root)
    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python
    report = VerificationReport(
        dataset_name=osp.basename(osp.normpath(dataset_root)),
        provenance=VERIFIER_PROVENANCE,
    )
    report.add(l0_structural((r["experiment"] for r in records), validate))
    report.add(l1_count(len(records), EXPECTED_RECORDS))
    report.add(_l1_unique(records))
    rows = read_contrast_rows(b23.raw_mirror_dir(base) / b23.DATA_RELPATH)
    report.add(_l2_values_equal_release(records, rows))
    report.add(_l3_contract(records))
    for value in SOURCED_VALUES.values():
        report.add(audit_sourced_value(value, Path(base) / "torchcell-library"))
    universe = _bacterial_gene_set(
        BACTERIAL_GENE_ASSEMBLIES[BACTERIAL_ASSEMBLY_SETS[REFERENCE_STRAIN]], base
    )
    report.add(_l4_tags_in_kt2440(records, universe))
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI: ``build`` the dev store or ``verify`` it."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        prog="python -m torchcell.datasets.pputida.borchert2023_contrast"
    )
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args(argv)
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(data_root, DATASET_ROOT_REL)
    if args.command == "build":
        print(f"len = {len(FitnessContrastBorchert2023Dataset(root=root))}")
        return 0
    report = verify_build(root, data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
