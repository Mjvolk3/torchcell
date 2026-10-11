# torchcell/verification/natural_isolate_response
# [[torchcell.verification.natural_isolate_response]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/natural_isolate_response
# Test file: tests/torchcell/verification/test_natural_isolate_response.py
"""L0-L4 verifier for natural-isolate environment-response datasets (Peter 2018).

A natural isolate's genotype is ~5,000 perturbations off S288C (sequence variants,
accessory ORFs present, reference ORFs absent), shared by every condition the isolate
was grown on. The environment-response verifier assumes a screened DELETION genotype:
its strain key drops ``strain_id`` (two isolates with the same variant genes but
different alleles would collide), and its L4 gene rules require every systematic name
to be a current S288C gene, which an accessory pangenome ORF is not by definition. This
module keeps that verifier's per-record rules (it imports them) and replaces the
genotype rules with ones that hold for an isolate:

- L0 ``structural``: every experiment validates against ``ExperimentType``; an
  isolate's genotype (one interned object shared by its records) is validated once and
  spliced into each of its records, so every record's own bytes are still validated.
- L1 ``count``, ``isolate_condition_uniqueness`` (one record per (isolate, condition)),
  ``isolate_genotype_identity`` (one ``strain_id`` per genotype, one genotype per
  isolate, every isolate on every condition), plus the shared census, canonical
  spellings, uncertainty, compound and media rules of
  :class:`~torchcell.verification.common.SharedRecordRules`.
- L2 ``value_fidelity`` (finite and non-negative), ``release_fidelity`` (every stored
  ratio equals its released cell bit for bit, every released cell stored once),
  ``genotype_round_trip`` (each isolate's present accessory ORFs, absent reference ORFs
  and their copy numbers re-derived from Peter's matrices; every variant's pointer names
  the pinned tarball and its own isolate).
- L3 ``measurement_type_consistent``, ``reference_zero`` (the ratio branch: 1.0),
  ``environment_perturbed``, ``conditions_documented`` (35 distinct environments, each
  the loader's condition, each Table 2 row re-read from the mirrored workbook), and
  ``sourced_values`` (every quote still in its sha256-pinned source).
- L4 ``s288c_gene_containment`` (variant and absence names are S288C genes),
  ``pangenome_orf_containment`` (present accessory ORFs are pangenome ORFs) and
  ``published_md5`` (the release's md5 equals the host's ``md5.txt``).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from torchcell.datasets.scerevisiae import peter2018 as p
from torchcell.datasets.scerevisiae.caudal2024 import COPYNUMBER_NAME, PRESENCE_NAME
from torchcell.verification.common import SharedRecordRules
from torchcell.verification.environment_response import (
    _condition_signature,
    _environment_perturbed_result,
    _measurement_type_result,
    _ratio_reference_accumulator,
    _value_problem,
    _value_result,
)
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
    sha256_file,
)
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

Record = dict[str, Any]

SEQUENCE_VARIANT = "sequence_variant"
NATURAL_PRESENCE = "natural_gene_presence"
NATURAL_ABSENCE = "natural_gene_absence"


def _result(
    level: Level, name: str, passed: bool, message: str, **details: Any
) -> LevelResult:
    return LevelResult(
        level=level, name=name, passed=passed, message=message, details=details
    )


def expected_condition_signatures(
    conditions: dict[str, p.ConditionSpec],
) -> dict[tuple[Any, ...], str]:
    """Condition signature of each loader condition's environment -> release header."""
    out: dict[tuple[Any, ...], str] = {}
    for code, spec in conditions.items():
        env = p.condition_environment(spec).model_dump()
        out[_condition_signature({"environment": env, "phenotype": {}})] = code
    return out


def _genotype_strain(perturbations: list[dict[str, Any]]) -> set[str]:
    return {str(pert.get("strain_id")) for pert in perturbations}


class _GenotypeChecks:
    """Per-isolate genotype rules, run once per distinct genotype object."""

    def __init__(self, pangenome: p.PangenomeMatrices, tarball_sha256: str) -> None:
        self.pangenome = pangenome
        self.tarball_sha256 = tarball_sha256
        self.n_genotypes = 0
        self.multi_strain: list[str] = []
        self.round_trip_failures: list[str] = []
        self.pointer_failures: list[str] = []
        self.type_counts: Counter[str] = Counter()
        self.s288c_names: Counter[str] = Counter()
        self.pangenome_ids: Counter[str] = Counter()

    def add(self, perturbations: list[dict[str, Any]]) -> str:
        """Check one genotype; return its isolate code."""
        self.n_genotypes += 1
        strains = _genotype_strain(perturbations)
        strain = sorted(strains)[0]
        if len(strains) != 1:
            self.multi_strain.append(",".join(sorted(strains)))
        present: dict[str, float] = {}
        absent: set[str] = set()
        for pert in perturbations:
            kind = str(pert["perturbation_type"])
            self.type_counts[kind] += 1
            if kind == NATURAL_PRESENCE:
                present[str(pert["pangenome_orf_id"])] = float(pert["copy_number"])
                self.pangenome_ids[str(pert["pangenome_orf_id"])] += 1
            elif kind == NATURAL_ABSENCE:
                absent.add(str(pert["systematic_gene_name"]))
                self.s288c_names[str(pert["systematic_gene_name"])] += 1
            elif kind == SEQUENCE_VARIANT:
                self.s288c_names[str(pert["systematic_gene_name"])] += 1
                self._check_pointer(strain, pert)
            else:
                self.round_trip_failures.append(f"{strain}: unexpected {kind}")
        self._round_trip(strain, present, absent)
        return strain

    def _check_pointer(self, strain: str, pert: dict[str, Any]) -> None:
        ref = pert.get("sequence_ref") or {}
        gene = pert["systematic_gene_name"]
        member = str(ref.get("member") or "")
        if (
            ref.get("sha256") != self.tarball_sha256
            or not member.startswith(f"{gene}.fasta#{strain}_{gene}_")
            or pert.get("strain_id") != strain
        ):
            if len(self.pointer_failures) < 20:
                self.pointer_failures.append(f"{strain}/{gene}: {member!r}")
            else:
                self.pointer_failures.append("...")

    def _round_trip(
        self, strain: str, present: dict[str, float], absent: set[str]
    ) -> None:
        pg = self.pangenome
        row = pg.row_of.get(strain)
        if row is None:
            self.round_trip_failures.append(f"{strain}: not in the presence matrix")
            return
        expected_present: dict[str, float] = {}
        expected_absent: set[str] = set()
        for j, orf in enumerate(pg.orf_ids):
            if pg.presence[row, j] == 1 and not pg.s288c_mask[j]:
                cn = pg.copy_number[row, j]
                expected_present[orf] = (
                    float(cn) if math.isfinite(cn) and cn > 0 else 1.0
                )
            elif pg.presence[row, j] == 0 and pg.s288c_mask[j]:
                expected_absent.add(pg.s288c_names[j] or orf)
        if present != expected_present:
            self.round_trip_failures.append(
                f"{strain}: {len(present)} present accessory ORFs stored, "
                f"{len(expected_present)} in the matrix"
            )
        if absent != expected_absent:
            self.round_trip_failures.append(
                f"{strain}: {len(absent)} absent reference ORFs stored, "
                f"{len(expected_absent)} in the matrix"
            )


def _l3_conditions_documented(
    observed: Counter[str],
    unmatched: int,
    conditions: dict[str, p.ConditionSpec],
    table_s2_path: Path,
) -> LevelResult:
    """35 distinct environments, each a loader condition, each Table 2 row re-read."""
    integrity = sha256_file(table_s2_path) == p.SI_TABLES_XLS_SHA256
    mismatched: list[str] = []
    if integrity:
        sheet = pd.read_excel(table_s2_path, sheet_name=p.TABLE_S2_SHEET, header=None)
        for code, spec in conditions.items():
            row = sheet.iloc[spec.table_s2_row - 1]
            if (str(row[0]), str(row[1])) != (
                spec.table_s2_condition,
                spec.table_s2_media,
            ):
                mismatched.append(f"{code}: row {spec.table_s2_row} reads {row[0]!r}")
        control = sheet.iloc[p.YPD_ROW - 1]
        if (str(control[0]), str(control[1])) != (p.CONTROL_CONDITION, p.YPD_CELL):
            mismatched.append(f"control: row {p.YPD_ROW} reads {control[0]!r}")
    passed = (
        integrity
        and not mismatched
        and unmatched == 0
        and set(observed) == set(conditions)
        and len(set(observed.values())) == 1
    )
    return _result(
        Level.L3,
        "conditions_documented",
        passed,
        f"{len(observed)} environments, each a Supplementary Table 2 row re-read from "
        f"{p.SI_TABLES_XLS} ({next(iter(set(observed.values())), 0)} records each)"
        if passed
        else f"{unmatched} records on no loader condition; {len(mismatched)} Table 2 rows "
        f"differ; workbook sha256 ok={integrity}; {len(observed)} conditions observed",
        n_conditions=len(observed),
        records_per_condition=dict(sorted(observed.items())),
        n_unmatched_records=unmatched,
        table_s2_mismatches=mismatched,
        workbook_sha256_ok=integrity,
    )


def _l3_sourced_values(library_root: Path) -> LevelResult:
    """Every text quote still in its pinned source; every Table 2 media quote a cell."""
    text_values = list(p.PHENOTYPE_SOURCED_VALUES.values()) + [
        SourcedValue(
            value=p.PAPER_MEASUREMENTS,
            provenance=Provenance(
                source_uri=p.PAPER_MD,
                citation_key=p.CITATION_KEY,
                sha256=p.PAPER_MD_SHA256,
            ),
            quote=p.MEASUREMENT_COUNT_QUOTE,
        ),
        SourcedValue(
            value="ratio normalization",
            provenance=Provenance(
                source_uri=p.PAPER_MD,
                citation_key=p.CITATION_KEY,
                sha256=p.PAPER_MD_SHA256,
            ),
            quote=p.NORMALIZATION_QUOTE,
        ),
        SourcedValue(
            value="pre-culture",
            provenance=Provenance(
                source_uri=p.PAPER_MD,
                citation_key=p.CITATION_KEY,
                sha256=p.PAPER_MD_SHA256,
            ),
            quote=p.PRECULTURE_QUOTE,
        ),
        SourcedValue(
            value="no repeat of the experiment",
            provenance=Provenance(
                source_uri=p.REPORTING_SUMMARY_MD,
                citation_key=p.CITATION_KEY,
                sha256=p.REPORTING_SUMMARY_MD_SHA256,
            ),
            quote=p.REPLICATION_QUOTE,
        ),
    ]
    failed: list[str] = []
    n_audited = 0
    for sv in text_values:
        if sv.provenance.source_uri == p.SI_TABLES_XLS:
            continue
        audit = audit_sourced_value(sv, library_root)
        n_audited += 1
        if not audit.passed:
            failed.append(f"{sv.quote[:50]!r}: {audit.message}")
    xls = library_root / p.CITATION_KEY / p.SI_TABLES_XLS
    cells: set[str] = set()
    if sha256_file(xls) == p.SI_TABLES_XLS_SHA256:
        sheet = pd.read_excel(xls, sheet_name=p.TABLE_S2_SHEET, header=None)
        cells = {str(v) for v in sheet.to_numpy().ravel().tolist() if str(v) != "nan"}
    else:
        failed.append(f"sha256 drift on {p.SI_TABLES_XLS}")
    media_quotes = [p.PETER_YPD.provenance[-1]] + [
        sv
        for spec in p.build_conditions().values()
        for sv in [
            *spec.media.provenance,
            *(sv for c in spec.media.components for sv in c.provenance),
        ]
        if sv.provenance.source_uri == p.SI_TABLES_XLS
    ]
    for sv in media_quotes:
        n_audited += 1
        if sv.quote not in cells:
            failed.append(f"Table 2 cell {sv.quote!r} not in the sheet")
    if p.TABLE_S2_TEMPERATURE_QUOTE not in cells:
        failed.append("the 30 C default is not in Table 2")
    n_audited += 1
    return _result(
        Level.L3,
        "sourced_values",
        not failed,
        f"{n_audited} sourced values audited against the library mirror"
        if not failed
        else f"{len(failed)}/{n_audited} sourced values failed audit",
        n_audited=n_audited,
        failures=failed[:20],
    )


def _l4_published_md5(raw_dir: Path, raw_mirror: Path) -> LevelResult:
    md5_txt = raw_mirror / p.MD5_REL
    published = p.read_published_md5(md5_txt)
    observed = p.md5_file(raw_dir / p.PHENO_NAME)
    passed = published.get(p.PHENO_NAME) == observed == p.PHENO_PUBLISHED_MD5
    return _result(
        Level.L4,
        "published_md5",
        passed,
        f"release md5 {observed} equals the host's md5.txt and the loader pin"
        if passed
        else f"release md5 {observed}; md5.txt {published.get(p.PHENO_NAME)}; pin "
        f"{p.PHENO_PUBLISHED_MD5}",
        observed=observed,
        published=published.get(p.PHENO_NAME),
    )


def verify_natural_isolate_response_streaming(
    records: Iterable[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    raw_dir: str | Path,
    raw_mirror: str | Path,
    library_root: str | Path,
    tarball_sha256: str,
    sgd_genes: set[str],
    min_s288c_containment: float,
) -> VerificationReport:
    """Single streaming pass over the store plus the raw re-derivations (see module)."""
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType, Genotype

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python
    raw_dir = Path(raw_dir)
    conditions = p.build_conditions()
    signatures = expected_condition_signatures(conditions)
    release = p.read_release(raw_dir / p.PHENO_NAME, conditions)
    released_cells = {code: release[code].to_dict() for code in conditions}
    pangenome = p.read_pangenome(raw_dir / PRESENCE_NAME, raw_dir / COPYNUMBER_NAME)
    genotype_checks = _GenotypeChecks(pangenome, tarball_sha256)
    shared = SharedRecordRules()
    ratio_reference = _ratio_reference_accumulator()

    n_records = 0
    l0_failures: list[dict[str, Any]] = []
    validated: dict[int, Any] = {}
    strain_of: dict[int, str] = {}
    genotypes_of_isolate: dict[str, set[int]] = {}
    pairs: set[tuple[str, str]] = set()
    n_duplicate_pairs = 0
    per_condition: Counter[str] = Counter()
    per_isolate: Counter[str] = Counter()
    n_unmatched = 0
    bad_values: list[dict[str, Any]] = []
    n_release_mismatch = 0
    release_mismatches: list[str] = []
    measurement_types: set[str] = set()
    temp_counts: Counter[Any] = Counter()
    media_counts: Counter[Any] = Counter()
    no_pert_env: list[tuple[Any, Any]] = []

    for i, rec in enumerate(records):
        n_records += 1
        exp = rec["experiment"]
        genotype = exp["genotype"]
        key = id(genotype)
        first_of_isolate = key not in validated
        if first_of_isolate:
            validated[key] = Genotype.model_validate(genotype)
            strain_of[key] = genotype_checks.add(genotype["perturbations"])
        strain = strain_of[key]
        genotypes_of_isolate.setdefault(strain, set()).add(key)
        try:
            validate({**exp, "genotype": validated[key]})
        except (ValueError, TypeError) as err:
            l0_failures.append({"index": i, "error": str(err)[:500]})
        shared.add(
            rec
            if first_of_isolate
            else {**rec, "experiment": {**exp, "genotype": {"perturbations": []}}}
        )

        signature = _condition_signature(exp)
        code = signatures.get(signature)
        if code is None:
            n_unmatched += 1
            code = f"unmatched:{i}"
        per_condition[code] += 1
        per_isolate[strain] += 1
        if (strain, code) in pairs:
            n_duplicate_pairs += 1
        pairs.add((strain, code))

        value = exp["phenotype"]["environment_response"]
        if (bad := _value_problem(i, float(value), minimum=0.0)) is not None:
            bad_values.append(bad)
        if code in conditions:
            released = float(released_cells[code][strain])
            if float(value) != released:
                n_release_mismatch += 1
                if len(release_mismatches) < 20:
                    release_mismatches.append(f"{strain}/{code}: {value} != {released}")
        measurement_types.add(str(exp["phenotype"]["measurement_type"]))
        ratio_reference.add(rec)
        environment = exp["environment"]
        temp = (environment.get("temperature") or {}).get("value")
        media = (environment.get("media") or {}).get("name")
        temp_counts[temp] += 1
        media_counts[media] += 1
        if not environment.get("perturbations"):
            no_pert_env.append((temp, media))

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(
        _result(
            Level.L0,
            "structural",
            not l0_failures,
            f"{n_records} records validated ({len(validated)} genotypes validated once "
            "each and spliced into their records)"
            if not l0_failures
            else f"{len(l0_failures)}/{n_records} records failed schema validation",
            n_records=n_records,
            n_genotypes=len(validated),
            failures=l0_failures[:10],
        )
    )
    report.add(
        _result(
            Level.L1,
            "count",
            n_records == expected_count,
            f"observed {n_records}, expected {expected_count}",
            observed=n_records,
            expected=expected_count,
        )
    )
    report.add(
        _result(
            Level.L1,
            "isolate_condition_uniqueness",
            n_duplicate_pairs == 0,
            f"{len(pairs)} unique (isolate, condition) records"
            if n_duplicate_pairs == 0
            else f"{n_duplicate_pairs} records repeat an (isolate, condition)",
            n_pairs=len(pairs),
            n_duplicated=n_duplicate_pairs,
        )
    )
    split_isolates = sorted(
        s for s, keys in genotypes_of_isolate.items() if len(keys) > 1
    )
    uneven = sorted(s for s, n in per_isolate.items() if n != len(conditions))
    identity_ok = not (genotype_checks.multi_strain or split_isolates or uneven) and (
        len(per_isolate) == p.N_ISOLATES == len(release.index)
    )
    report.add(
        _result(
            Level.L1,
            "isolate_genotype_identity",
            identity_ok,
            f"{len(per_isolate)} isolates, one genotype and one strain_id each, "
            f"{len(conditions)} records each"
            if identity_ok
            else f"{len(genotype_checks.multi_strain)} genotypes mix strain_ids; "
            f"{len(split_isolates)} isolates carry two genotypes; {len(uneven)} isolates "
            f"lack a condition; {len(per_isolate)} isolates",
            n_isolates=len(per_isolate),
            multi_strain=genotype_checks.multi_strain[:10],
            split_isolates=split_isolates[:10],
            uneven_isolates=uneven[:10],
        )
    )
    report.add(_value_result("value_fidelity", n_records, bad_values))
    n_cells = int(release.size)
    release_ok = n_release_mismatch == 0 and len(pairs) == n_cells and n_unmatched == 0
    report.add(
        _result(
            Level.L2,
            "release_fidelity",
            release_ok,
            f"{n_cells} released cells, each stored once and equal to its stored ratio"
            if release_ok
            else f"{n_release_mismatch} stored ratios differ from the release; "
            f"{len(pairs)} of {n_cells} cells stored",
            n_cells=n_cells,
            n_mismatched=n_release_mismatch,
            mismatches=release_mismatches,
        )
    )
    round_trip_ok = not (
        genotype_checks.round_trip_failures or genotype_checks.pointer_failures
    )
    report.add(
        _result(
            Level.L2,
            "genotype_round_trip",
            round_trip_ok,
            f"{genotype_checks.n_genotypes} genotypes re-derived from the presence and "
            "copy-number matrices; every variant pointer names the pinned tarball and its "
            "own isolate"
            if round_trip_ok
            else f"{len(genotype_checks.round_trip_failures)} matrix round-trip failures; "
            f"{len(genotype_checks.pointer_failures)} variant pointers wrong",
            perturbation_types=dict(genotype_checks.type_counts),
            round_trip_failures=genotype_checks.round_trip_failures[:20],
            pointer_failures=genotype_checks.pointer_failures[:20],
        )
    )
    report.add(_measurement_type_result(measurement_types))
    report.add(ratio_reference.result())
    baseline_temp = temp_counts.most_common(1)[0][0] if temp_counts else None
    baseline_media = media_counts.most_common(1)[0][0] if media_counts else None
    n_env_missing = sum(
        1
        for temp, media in no_pert_env
        if not (temp is not None and temp != baseline_temp)
        and not (media is not None and media != baseline_media)
    )
    report.add(
        _environment_perturbed_result(
            n_records=n_records,
            n_missing=n_env_missing,
            expected=0,
            baseline_temp=baseline_temp,
            baseline_media=baseline_media,
        )
    )
    library_root = Path(library_root)
    report.add(
        _l3_conditions_documented(
            per_condition,
            n_unmatched,
            conditions,
            library_root / p.CITATION_KEY / p.SI_TABLES_XLS,
        )
    )
    report.add(_l3_sourced_values(library_root))
    for result in shared.results():
        report.add(result)

    measured = set(genotype_checks.s288c_names)
    missing = sorted(measured - sgd_genes)
    overlap = len(measured & sgd_genes) / len(measured) if measured else 0.0
    report.add(
        _result(
            Level.L4,
            "s288c_gene_containment",
            bool(measured) and overlap >= min_s288c_containment,
            f"{overlap:.4f} of {len(measured)} sequence-variant and absence names are "
            f"S288C genes (>= {min_s288c_containment}); {len(missing)} are not"
            if measured
            else "no S288C-keyed perturbation",
            n_measured=len(measured),
            n_missing=len(missing),
            overlap=overlap,
            missing_examples=missing[:20],
        )
    )
    pangenome_set = set(pangenome.orf_ids)
    stray = sorted(set(genotype_checks.pangenome_ids) - pangenome_set)
    report.add(
        _result(
            Level.L4,
            "pangenome_orf_containment",
            not stray and bool(genotype_checks.pangenome_ids),
            f"all {len(genotype_checks.pangenome_ids)} present accessory ORFs are "
            "pangenome ORFs of the presence matrix"
            if not stray
            else f"{len(stray)} present ORFs are not pangenome ORFs",
            n_orfs=len(genotype_checks.pangenome_ids),
            stray=stray[:20],
        )
    )
    report.add(_l4_published_md5(raw_dir, Path(raw_mirror)))
    return report


__all__ = ["expected_condition_signatures", "verify_natural_isolate_response_streaming"]
