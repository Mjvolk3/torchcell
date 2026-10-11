# torchcell/verification/pseudobulk_umi_count
# [[torchcell.verification.pseudobulk_umi_count]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/pseudobulk_umi_count
# Test file: tests/torchcell/verification/test_pseudobulk_umi_count.py
"""L0-L4 verifier for pseudobulk UMI-count datasets (Jackson 2020).

A record is the summed single-cell UMI counts of one barcoded strain in one culture, so
the strongest check there is is to re-sum the released matrix and compare every number.
The rules:

- L0 ``structural``: every experiment validates as ``PseudobulkUmiCountExperiment`` and
  every distinct reference as ``PseudobulkUmiCountExperimentReference``.
- L1 ``count``, ``strain_condition_uniqueness`` (one record per (strain, condition), every
  record on a loader condition), ``strain_identity`` (each strain label carries one
  barcode and one deleted gene, the barcode is Table S2's and Source code 4's, six
  strains per gene), plus the shared census, canonical spellings, uncertainty, compound
  and media rules of :class:`~torchcell.verification.common.SharedRecordRules`.
- L2 ``release_fidelity``: every record's gene and marker counts, total, cell barcodes and
  per-cell totals equal a fresh sum of the released matrix, every (strain, condition)
  holding a cell is stored, and the empty pairs are exactly the loader's ledger;
  ``reference_fidelity``: each condition's reference equals the pooled sum of its
  HO-deletion control strains; ``release_totals``: the cells and counts the records sum to
  are the paper's 38,225 and 83,703,440.
- L3 ``measurement_type_consistent``, ``feature_universe`` (one universe pin shared by
  every record and reference, equal to the loader's feature map, every key inside it),
  ``conditions_documented`` (eleven environments, each a loader condition whose Table 1
  row is in the paper and whose release label is the authors' map), ``sourced_values``
  (every paper quote in its pinned OCR, every xlsx quote a row rendering of the pinned
  workbook).
- L4 ``feature_containment`` (the universe's ids that are SGD systematic names, against a
  floor) and ``feature_ids_corroborated`` (the un-mangled release header equals the bulk
  matrix's header from the same release).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from torchcell.datasets.scerevisiae import jackson2020 as j
from torchcell.verification.common import SharedRecordRules, declared_member_validator
from torchcell.verification.environment_response import _measurement_type_result
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
    sha256_file,
)
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

Record = dict[str, Any]


def _result(
    level: Level, name: str, passed: bool, message: str, **details: Any
) -> LevelResult:
    return LevelResult(
        level=level, name=name, passed=passed, message=message, details=details
    )


def _environment_key(environment: dict[str, Any]) -> str:
    return json.dumps(environment, sort_keys=True, default=str)


def expected_environments(conditions: dict[str, j.ConditionSpec]) -> dict[str, str]:
    """Canonical JSON of each loader condition's environment -> Table 1 abbreviation."""
    return {
        _environment_key(j.condition_environment(spec).model_dump()): abbreviation
        for abbreviation, spec in conditions.items()
    }


def _expected_phenotype(
    entry: j.StrainConditionSum, features: j.FeatureMap
) -> dict[str, Any]:
    return j.pseudobulk_phenotype(entry, features).model_dump()


_COMPARED_FIELDS = (
    "umi_count",
    "marker_umi_count",
    "total_umis",
    "n_features",
    "feature_universe_sha256",
    "n_cells",
    "cell_barcodes",
    "cell_total_umis",
)


def _phenotype_mismatch(stored: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Fields whose stored value differs from the re-derived one."""
    return [
        name
        for name in _COMPARED_FIELDS
        if (dict(stored[name]) if isinstance(stored[name], dict) else stored[name])
        != (
            dict(expected[name]) if isinstance(expected[name], dict) else expected[name]
        )
    ]


def _l3_conditions_documented(
    observed: Counter[str],
    unmatched: int,
    conditions: dict[str, j.ConditionSpec],
    settings: j.AuthorSettings,
    paper_md: Path,
) -> LevelResult:
    """Eleven environments, each a loader condition, Table 1 rows and labels re-read."""
    text = paper_md.read_text(encoding="utf-8")
    integrity = sha256_file(paper_md) == j.PAPER_MD_SHA256
    missing_rows = [a for a, spec in conditions.items() if spec.table1_row not in text]
    mislabeled = [
        a
        for a, spec in conditions.items()
        if settings.condition_label.get(spec.release_label) != a
    ]
    passed = (
        integrity
        and not missing_rows
        and not mislabeled
        and unmatched == 0
        and set(observed) == set(conditions)
    )
    return _result(
        Level.L3,
        "conditions_documented",
        passed,
        f"{len(observed)} environments, each a Table 1 row of {j.PAPER_MD} and an "
        "authors' release label"
        if passed
        else f"{unmatched} records on no loader condition; Table 1 rows missing "
        f"{missing_rows}; labels not the authors' {mislabeled}; paper sha256 "
        f"ok={integrity}",
        records_per_condition=dict(sorted(observed.items())),
        n_unmatched_records=unmatched,
        missing_table1_rows=missing_rows,
        mislabeled=mislabeled,
        paper_sha256_ok=integrity,
    )


def _l3_sourced_values(
    library_root: Path, specs: list[j.StrainSpec], settings_path: Path
) -> LevelResult:
    """Every paper quote in its OCR; every xlsx quote a row of the workbook; the two
    condition-label lines of the authors' settings file verbatim.
    """
    failed: list[str] = []
    n_audited = 0
    for quote in j.PAPER_QUOTES:
        audit = audit_sourced_value(
            SourcedValue(
                value="quote",
                provenance=Provenance(
                    source_uri=j.PAPER_MD,
                    citation_key=j.CITATION_KEY,
                    sha256=j.PAPER_MD_SHA256,
                ),
                quote=quote,
            ),
            library_root,
        )
        n_audited += 1
        if not audit.passed:
            failed.append(f"{quote[:50]!r}: {audit.message}")
    xlsx = library_root / j.CITATION_KEY / j.SI_XLSX
    if sha256_file(xlsx) != j.SI_XLSX_SHA256:
        failed.append(f"sha256 drift on {j.SI_XLSX}")
        sheet_text = ""
    else:
        sheet_text = j.sheet_text(xlsx)
    background = j.fy_background()
    xlsx_values = [
        *(
            sv
            for sv in background.provenance or []
            if sv.provenance.source_uri == j.SI_XLSX
        ),
        *(
            sv
            for sv in j.JACKSON_YPD.provenance
            if sv.provenance.source_uri == j.SI_XLSX
        ),
    ]
    for spec in j.build_conditions(j.read_author_settings(settings_path)).values():
        xlsx_values.extend(
            sv
            for sv in [
                *spec.media.provenance,
                *(sv for c in spec.media.components for sv in c.provenance),
            ]
            if sv.provenance.citation_key == j.CITATION_KEY
            and sv.provenance.source_uri == j.SI_XLSX
        )
    quotes = {sv.quote for sv in xlsx_values} | {s.diploid_row_quote for s in specs}
    for quote in sorted(quotes):
        n_audited += 1
        if quote not in sheet_text:
            failed.append(f"xlsx row {quote[:60]!r} not in the workbook")
    settings = j.read_author_settings(settings_path)
    settings_text = settings_path.read_text()
    for line in (settings.condition_levels_line, settings.condition_labels_line):
        n_audited += 1
        if line not in settings_text:
            failed.append(f"settings line {line[:50]!r} not verbatim")
    return _result(
        Level.L3,
        "sourced_values",
        not failed,
        f"{n_audited} sourced values audited against the library and raw mirrors"
        if not failed
        else f"{len(failed)}/{n_audited} sourced values failed audit",
        n_audited=n_audited,
        failures=failed[:20],
    )


def verify_pseudobulk_umi_count_streaming(
    records: Iterable[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    raw_dir: str | Path,
    library_root: str | Path,
    genome: Any,
    sgd_genes: set[str],
    min_feature_containment: float,
) -> VerificationReport:
    """One streaming pass over the store against a fresh re-sum of the release."""
    validate_experiment: Callable[[Any], object] = declared_member_validator(
        "PseudobulkUmiCountExperiment"
    )
    validate_reference: Callable[[Any], object] = declared_member_validator(
        "PseudobulkUmiCountExperimentReference", union="ExperimentReferenceType"
    )
    raw = Path(raw_dir)
    library_root = Path(library_root)
    specs = j.read_strain_table(raw / j.SUPP1_NAME)
    strains = {s.strain: s for s in specs}
    fasta = j.read_barcode_fasta(raw / j.BARCODES_NAME)
    settings = j.read_author_settings(raw / j.SETTINGS_NAME)
    conditions = j.build_conditions(settings)
    environment_of = expected_environments(conditions)
    abbreviation_of_label = {c.release_label: a for a, c in conditions.items()}
    bulk_header = pd.read_csv(raw / j.BULK_NAME, sep="\t", nrows=0).columns.tolist()
    released = j.read_release_header(raw / j.MATRIX_NAME)
    features = j.build_feature_map(released, bulk_header, genome)
    sums, n_cells = j.read_release(
        raw / j.MATRIX_NAME, strains, set(settings.condition_label)
    )
    expected_pairs = {(strain, abbreviation_of_label[label]) for strain, label in sums}
    shared = SharedRecordRules(resolve_gene_name=genome.resolve_gene_name)

    n_records = 0
    l0_failures: list[dict[str, Any]] = []
    references: dict[str, dict[str, Any]] = {}
    pairs: Counter[tuple[str, str]] = Counter()
    per_condition: Counter[str] = Counter()
    n_unmatched = 0
    barcode_of: dict[str, set[str]] = {}
    gene_of: dict[str, set[str]] = {}
    release_mismatches: list[str] = []
    n_release_mismatch = 0
    measurement_types: set[str] = set()
    universes: Counter[tuple[int, str]] = Counter()
    stray_keys: set[str] = set()
    universe_keys = set(features.gene_keys)
    total_cells = 0
    total_umis = 0

    for i, rec in enumerate(records):
        n_records += 1
        exp = rec["experiment"]
        try:
            validate_experiment(exp)
        except (ValueError, TypeError) as err:
            l0_failures.append({"index": i, "error": str(err)[:500]})
        shared.add(rec)
        perturbation = exp["genotype"]["perturbations"][0]
        strain = str((perturbation.get("construction") or {}).get("strain_accession"))
        barcode_of.setdefault(strain, set()).add(str(perturbation.get("barcode")))
        gene_of.setdefault(strain, set()).add(str(perturbation["systematic_gene_name"]))
        abbreviation = environment_of.get(_environment_key(exp["environment"]))
        if abbreviation is None:
            n_unmatched += 1
            abbreviation = f"unmatched:{i}"
        per_condition[abbreviation] += 1
        pairs[(strain, abbreviation)] += 1
        reference = rec["reference"]
        references.setdefault(abbreviation, reference)

        phenotype = exp["phenotype"]
        measurement_types.add(str(phenotype["measurement_type"]))
        universes[
            (int(phenotype["n_features"]), str(phenotype["feature_universe_sha256"]))
        ] += 1
        stray_keys.update(set(phenotype["umi_count"]) - universe_keys)
        total_cells += int(phenotype["n_cells"])
        total_umis += int(phenotype["total_umis"])
        if abbreviation in conditions:
            key = (strain, conditions[abbreviation].release_label)
            if key not in sums:
                diff = ["absent from the release"]
            else:
                diff = _phenotype_mismatch(
                    phenotype, _expected_phenotype(sums[key], features)
                )
            if diff:
                n_release_mismatch += 1
                if len(release_mismatches) < 20:
                    release_mismatches.append(f"{strain}/{abbreviation}: {diff}")

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    l0_reference_failures: list[str] = []
    for abbreviation, reference in references.items():
        try:
            validate_reference(reference)
        except (ValueError, TypeError) as err:
            l0_reference_failures.append(f"{abbreviation}: {str(err)[:300]}")
    l0_ok = not l0_failures and not l0_reference_failures
    report.add(
        _result(
            Level.L0,
            "structural",
            l0_ok,
            f"{n_records} records and {len(references)} references validated"
            if l0_ok
            else f"{len(l0_failures)}/{n_records} records and "
            f"{len(l0_reference_failures)} references failed schema validation",
            n_records=n_records,
            n_references=len(references),
            failures=l0_failures[:10],
            reference_failures=l0_reference_failures[:10],
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
    duplicated = sorted(f"{s}/{a}" for (s, a), n in pairs.items() if n > 1)
    observed_pairs = set(pairs)
    missing_pairs = sorted(f"{s}/{a}" for s, a in expected_pairs - observed_pairs)
    extra_pairs = sorted(f"{s}/{a}" for s, a in observed_pairs - expected_pairs)
    unique_ok = not duplicated and not missing_pairs and not extra_pairs
    report.add(
        _result(
            Level.L1,
            "strain_condition_uniqueness",
            unique_ok,
            f"{len(observed_pairs)} unique (strain, condition) records, exactly the "
            f"{len(expected_pairs)} pairs holding a cell in the release"
            if unique_ok
            else f"{len(duplicated)} duplicated, {len(missing_pairs)} released pairs not "
            f"stored, {len(extra_pairs)} stored pairs not released",
            n_pairs=len(observed_pairs),
            duplicated=duplicated[:10],
            missing=missing_pairs[:10],
            extra=extra_pairs[:10],
        )
    )
    multi = sorted(
        s for s in barcode_of if len(barcode_of[s]) != 1 or len(gene_of[s]) != 1
    )
    wrong_barcode = sorted(
        s
        for s, codes in barcode_of.items()
        if len(codes) == 1 and (next(iter(codes)) != fasta.get(s) or s not in strains)
    )
    gene_ids = j.TfDeletionPseudobulkJackson2020Dataset._deleted_gene_ids(
        specs, settings, genome
    )
    wrong_gene = sorted(
        s
        for s, genes in gene_of.items()
        if s in strains and genes != {gene_ids[strains[s].gene_token][0]}
    )
    per_gene = Counter(next(iter(g)) for g in gene_of.values() if len(g) == 1)
    expected_per_gene = Counter(gene_ids[s.gene_token][0] for s in specs)
    identity_ok = (
        not multi
        and not wrong_barcode
        and not wrong_gene
        and set(barcode_of) == set(strains)
        and per_gene == expected_per_gene
    )
    report.add(
        _result(
            Level.L1,
            "strain_identity",
            identity_ok,
            f"{len(barcode_of)} strains, one barcode and one deleted gene each, every "
            "barcode Table S2's and Source code 4's, every gene the label's; "
            f"{len(per_gene)} genes, strains per gene as Table S2 lists them"
            if identity_ok
            else f"{len(multi)} strains carry two barcodes or genes; "
            f"{len(wrong_barcode)} barcodes differ from the FASTA; {len(wrong_gene)} "
            f"genes differ from the label; strains per gene {dict(per_gene)}, Table S2 "
            f"{dict(expected_per_gene)}",
            n_strains=len(barcode_of),
            strains_per_gene=dict(sorted(per_gene.items())),
            multi=multi[:10],
            wrong_barcode=wrong_barcode[:10],
            wrong_gene=wrong_gene[:10],
        )
    )
    release_ok = n_release_mismatch == 0 and unique_ok
    report.add(
        _result(
            Level.L2,
            "release_fidelity",
            release_ok,
            f"{n_records} records equal a fresh sum of the released matrix (genes, "
            "markers, totals, cell barcodes and per-cell totals)"
            if release_ok
            else f"{n_release_mismatch} records differ from the re-summed release",
            n_mismatched=n_release_mismatch,
            mismatches=release_mismatches,
        )
    )
    wild_type = [s.strain for s in specs if s.is_wild_type]
    reference_mismatches: list[str] = []
    for abbreviation, spec in conditions.items():
        stored = references.get(abbreviation)
        entries = [
            sums[(s, spec.release_label)]
            for s in wild_type
            if (s, spec.release_label) in sums
        ]
        if stored is None or not entries:
            reference_mismatches.append(
                f"{abbreviation}: no reference or no control cell"
            )
            continue
        diff = _phenotype_mismatch(
            stored["phenotype_reference"],
            _expected_phenotype(j.pooled(entries), features),
        )
        if diff:
            reference_mismatches.append(f"{abbreviation}: {diff}")
    report.add(
        _result(
            Level.L2,
            "reference_fidelity",
            not reference_mismatches,
            f"{len(conditions)} references, each the pooled sum of the condition's "
            "HO-deletion control strains"
            if not reference_mismatches
            else f"{len(reference_mismatches)} references differ from the pooled controls",
            mismatches=reference_mismatches[:20],
        )
    )
    released_umis = int(sum(int(e.counts.sum()) for e in sums.values()))
    totals_ok = (
        total_cells == n_cells == j.PAPER_CELLS
        and total_umis == released_umis == j.PAPER_TRANSCRIPT_COUNTS
    )
    report.add(
        _result(
            Level.L2,
            "release_totals",
            totals_ok,
            f"records sum to {total_cells} cells and {total_umis} UMIs, the release's "
            "and the paper's 38,225 cells and 83,703,440 transcript counts"
            if totals_ok
            else f"records sum to {total_cells} cells / {total_umis} UMIs; release "
            f"{n_cells} / {released_umis}; paper {j.PAPER_CELLS} / "
            f"{j.PAPER_TRANSCRIPT_COUNTS}",
            record_cells=total_cells,
            record_umis=total_umis,
            released_cells=n_cells,
            released_umis=released_umis,
        )
    )
    report.add(_measurement_type_result(measurement_types))
    reference_universes = {
        (
            int(r["phenotype_reference"]["n_features"]),
            str(r["phenotype_reference"]["feature_universe_sha256"]),
        )
        for r in references.values()
    }
    expected_universe = (len(features.gene_columns), features.universe_sha256)
    universe_ok = (
        set(universes) == {expected_universe}
        and reference_universes == {expected_universe}
        and not stray_keys
    )
    report.add(
        _result(
            Level.L3,
            "feature_universe",
            universe_ok,
            f"one universe of {expected_universe[0]} gene features pinned by every "
            "record and reference, equal to the loader's feature map; every key inside it"
            if universe_ok
            else f"universes {sorted(universes)}; references {sorted(reference_universes)}; "
            f"{len(stray_keys)} keys outside the universe",
            expected=list(expected_universe),
            stray_keys=sorted(stray_keys)[:20],
        )
    )
    report.add(
        _l3_conditions_documented(
            per_condition,
            n_unmatched,
            conditions,
            settings,
            library_root / j.CITATION_KEY / j.PAPER_MD,
        )
    )
    report.add(_l3_sourced_values(library_root, specs, raw / j.SETTINGS_NAME))
    for result in shared.results():
        report.add(result)

    gene_keys = features.gene_keys
    outside = sorted(k for k in gene_keys if k not in sgd_genes)
    overlap = 1.0 - len(outside) / len(gene_keys) if gene_keys else 0.0
    report.add(
        _result(
            Level.L4,
            "feature_containment",
            bool(gene_keys) and overlap >= min_feature_containment,
            f"{overlap:.5f} of {len(gene_keys)} gene features are SGD systematic names "
            f"(>= {min_feature_containment}); {len(outside)} are not: {outside}",
            n_features=len(gene_keys),
            n_outside=len(outside),
            outside=outside[:20],
            legacy=features.legacy(),
        )
    )
    corroborated = [j.unmangle(name) for name in released] == bulk_header
    report.add(
        _result(
            Level.L4,
            "feature_ids_corroborated",
            corroborated,
            f"the {len(released)} un-mangled release columns equal TRIZOL_BULK.tsv's "
            "header, in order"
            if corroborated
            else "the un-mangled release header differs from TRIZOL_BULK.tsv's",
            n_columns=len(released),
            markers=list(j.MARKER_FEATURES),
        )
    )
    return report


__all__ = ["expected_environments", "verify_pseudobulk_umi_count_streaming"]
