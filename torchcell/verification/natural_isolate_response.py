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
  ``isolate_condition_coverage`` (one genotype object per isolate, every isolate on
  every condition), plus the shared census, canonical spellings, uncertainty, compound
  and media rules of :class:`~torchcell.verification.common.SharedRecordRules`.
- L2 ``value_fidelity`` (finite and non-negative), ``release_fidelity`` (every stored
  ratio equals its released cell bit for bit, every released cell stored once).
- The four isolate-genotype rules every natural-isolate store shares
  (:class:`~torchcell.verification.natural_isolate_genotype.IsolateGenotypeAudit`): L1
  ``isolate_genotype_identity``, L2 ``isolate_genotype_round_trip`` (every stored
  genotype equals its re-derivation from the pinned releases), L4
  ``isolate_descriptor_matches_table_s1`` and ``loss_of_function_calls_match_release``.
- L3 ``measurement_type_consistent``, ``reference_zero`` (the ratio branch: 1.0),
  ``environment_perturbed``, ``conditions_documented`` (35 distinct environments, each
  the loader's condition, each Table 2 row re-read from the mirrored workbook), and
  ``sourced_values`` (every quote still in its sha256-pinned source).
- L4 ``s288c_gene_containment`` (variant, absence, deletion and loss-of-function names
  are S288C genes; a frameshift call released without a pangenome code is keyed by the
  release's ``Gene``, not an S288C name, and is counted apart),
  ``pangenome_orf_containment`` (present accessory ORFs are pangenome ORFs) and
  ``published_md5`` (the release's md5 equals the host's ``md5.txt``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from torchcell.datasets.scerevisiae import peter2018 as p
from torchcell.sequence import natural_isolate as ni
from torchcell.verification.common import (
    SharedRecordRules,
    declared_first_union_validator,
)
from torchcell.verification.environment_response import (
    _condition_signature,
    _environment_perturbed_result,
    _measurement_type_result,
    _ratio_reference_accumulator,
    _value_problem,
    _value_result,
)
from torchcell.verification.natural_isolate_genotype import IsolateGenotypeAudit
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
    sha256_file,
)
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

Record = dict[str, Any]

NATURAL_PRESENCE = "natural_gene_presence"
LOSS_OF_FUNCTION = "natural_loss_of_function"


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


class _GenotypeCensus:
    """Names the stored genotypes carry, for the two L4 containment rules."""

    def __init__(self) -> None:
        self.type_counts: Counter[str] = Counter()
        self.s288c_names: Counter[str] = Counter()
        self.pangenome_ids: Counter[str] = Counter()
        self.release_gene_keys: Counter[str] = Counter()

    def add(self, perturbations: list[dict[str, Any]]) -> None:
        """Count one genotype's names: pangenome ids of presences, the release's
        ``Gene`` of a code-less frameshift call, S288C names else.
        """
        for pert in perturbations:
            kind = str(pert["perturbation_type"])
            self.type_counts[kind] += 1
            if kind == NATURAL_PRESENCE:
                self.pangenome_ids[str(pert["pangenome_orf_id"])] += 1
            elif kind == LOSS_OF_FUNCTION and pert["release_code"] is None:
                self.release_gene_keys[str(pert["systematic_gene_name"])] += 1
            else:
                self.s288c_names[str(pert["systematic_gene_name"])] += 1


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
    genotypes: ni.NaturalIsolateGenotypes,
    sgd_genes: set[str],
    min_s288c_containment: float,
) -> VerificationReport:
    """Single streaming pass over the store plus the raw re-derivations (see module).

    ``genotypes`` re-derives each isolate's genotype from the pinned releases (built over
    the release's isolates, reading the store's own allele-diff cache).
    """
    from torchcell.datamodels.schema import Genotype

    validate: Callable[[Any], object] = declared_first_union_validator()
    raw_dir = Path(raw_dir)
    conditions = p.build_conditions()
    signatures = expected_condition_signatures(conditions)
    release = p.read_release(raw_dir / p.PHENO_NAME, conditions)
    released_cells = {code: release[code].to_dict() for code in conditions}
    pangenome = ni.read_pangenome(
        raw_dir / ni.PRESENCE_NAME, raw_dir / ni.COPYNUMBER_NAME
    )
    census = _GenotypeCensus()
    audit = IsolateGenotypeAudit(genotypes, library_root)
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
            census.add(genotype["perturbations"])
            strain_of[key] = str(audit.add(genotype))
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
    coverage_ok = not (split_isolates or uneven) and (
        len(per_isolate) == p.N_ISOLATES == len(release.index)
    )
    report.add(
        _result(
            Level.L1,
            "isolate_condition_coverage",
            coverage_ok,
            f"{len(per_isolate)} isolates, one genotype object each, "
            f"{len(conditions)} records each"
            if coverage_ok
            else f"{len(split_isolates)} isolates carry two genotype objects; "
            f"{len(uneven)} isolates lack a condition; {len(per_isolate)} isolates",
            n_isolates=len(per_isolate),
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
    for result in audit.results():
        report.add(result)
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

    measured = set(census.s288c_names)
    missing = sorted(measured - sgd_genes)
    overlap = len(measured & sgd_genes) / len(measured) if measured else 0.0
    report.add(
        _result(
            Level.L4,
            "s288c_gene_containment",
            bool(measured) and overlap >= min_s288c_containment,
            f"{overlap:.4f} of {len(measured)} S288C-keyed perturbation names are "
            f"S288C genes (>= {min_s288c_containment}); {len(missing)} are not"
            if measured
            else "no S288C-keyed perturbation",
            n_measured=len(measured),
            n_missing=len(missing),
            overlap=overlap,
            missing_examples=missing[:20],
            perturbation_types=dict(census.type_counts),
            n_release_gene_keyed_calls=sum(census.release_gene_keys.values()),
            n_release_gene_keys=len(census.release_gene_keys),
        )
    )
    pangenome_set = set(pangenome.orf_ids)
    stray = sorted(set(census.pangenome_ids) - pangenome_set)
    report.add(
        _result(
            Level.L4,
            "pangenome_orf_containment",
            not stray and bool(census.pangenome_ids),
            f"all {len(census.pangenome_ids)} present accessory ORFs are "
            "pangenome ORFs of the presence matrix"
            if not stray
            else f"{len(stray)} present ORFs are not pangenome ORFs",
            n_orfs=len(census.pangenome_ids),
            stray=stray[:20],
        )
    )
    report.add(_l4_published_md5(raw_dir, Path(raw_mirror)))
    return report


__all__ = ["expected_condition_signatures", "verify_natural_isolate_response_streaming"]
