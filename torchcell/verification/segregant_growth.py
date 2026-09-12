# torchcell/verification/segregant_growth
# [[torchcell.verification.segregant_growth]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/segregant_growth
"""L0-L4 record-level verifier for segregant growth datasets (haplotype-mosaic genotypes).

A ``SegregantGrowthExperiment`` carries a ``SegregantGenotype`` (a haplotype mosaic
against two pinned parent assemblies) rather than gene-keyed perturbations, so the
environment-response verifier's strain signature (which reads ``perturbations``), its
"every record carries an environmental edit" rule (the control plates are the point of
the design) and its single-measurement-type rule (residual and absolute columns are one
experiment) do not apply. This module is a single streaming pass over the LMDB records
plus a raw-file pass that re-derives what the records claim:

- L0 ``structural`` -- every experiment validates against ``ExperimentType`` (the
  segregant experiment pair is in the union).
- L1 ``count``; ``pair_uniqueness`` on (segregant id, condition signature); ``id_bijection``
  (every phenotype row id is in exactly one genotype file and every genotype row has a
  phenotype row); the ``provenance_gaps`` census.
- L2 ``value_fidelity`` (finite, and every stored value equals the released tsv cell
  exactly); ``mosaic_round_trip`` (every segregant's blocks re-expanded at the cross's
  sorted marker positions equal its released row); ``block_invariants`` (ordered,
  non-overlapping, parents alternate, ``n_markers`` sums to the cross's marker count).
- L3 ``measurement_partition`` (exactly 36 residual and 2 absolute conditions);
  ``reference_zero``; ``conditions_documented`` (every environment is one of the 38 and
  exactly two carry no environmental edit); ``sourced_values`` (the text anchors in the
  raw mirror still hold their quotes, and the xls still carries each quoted cell).
- L4 ``parents_pinned`` (every parent's Peter id is in the assembly member index);
  ``cross_pairs`` (xls parents match the README pairs); ``segregant_counts`` (per-cross
  counts equal the xls); ``marker_reference`` (a marker sample's ref allele equals the
  S288C base at that position, which settles the coordinate system empirically);
  ``gene_containment_sgd`` (the derived gene set is a subset of the SGD gene universe).
"""

from __future__ import annotations

import math
import os.path as osp
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from torchcell.datamodels.schema import HaplotypeBlock
from torchcell.datasets.scerevisiae import bloom2019 as b
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)
from torchcell.verification.sourced import (
    ProvenanceGapCensus,
    ProvenanceGapReason,
    SourcedValue,
    audit_sourced_value,
    provenance_gap_level_result,
)

Record = dict[str, Any]


def _condition_signature(experiment: dict[str, Any]) -> tuple[Any, ...]:
    """Canonical environment identity (perturbations, temperature, media, durations)."""
    env = experiment["environment"]
    perts: list[tuple[str, ...]] = []
    for p in env.get("perturbations") or []:
        compound = p.get("compound") if isinstance(p.get("compound"), dict) else {}
        agent = p.get("agent") if isinstance(p.get("agent"), dict) else {}
        dose = p.get("concentration") or p.get("magnitude") or {}
        dose = dose if isinstance(dose, dict) else {}
        fields = (
            p.get("perturbation_type"),
            compound.get("name") or agent.get("name"),
            p.get("factor"),
            dose.get("value"),
            dose.get("unit"),
            dose.get("basis"),
        )
        perts.append(tuple("" if v is None else str(v) for v in fields))
    return (
        tuple(sorted(perts)),
        (env.get("temperature") or {}).get("value"),
        (env.get("media") or {}).get("name"),
        env.get("duration_hours"),
    )


def _result(
    level: Level, name: str, passed: bool, message: str, **details: Any
) -> LevelResult:
    return LevelResult(
        level=level, name=name, passed=passed, message=message, details=details
    )


def verify_segregant_growth_streaming(
    records: Iterable[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    raw_dir: str | Path,
    raw_mirror: str | Path,
    assembly_index_path: str | Path,
    genome: Any,
    sgd_genes: set[str],
    gene_set: set[str],
    marker_sample: int = 2000,
    min_marker_reference: float = 0.95,
    skip_sourced_values: bool = False,
) -> VerificationReport:
    """Single-pass L0-L4 gate for a segregant growth dataset (see the module docstring).

    ``raw_dir`` is the dataset's ``raw/`` (symlinks into the mirror), ``raw_mirror`` the
    mirror root holding ``code/`` and ``paper/``, ``genome`` the S288C genome object
    (``fasta_dna`` + ``chr_to_nc``) for the marker reference check.
    ``skip_sourced_values`` omits the L3 provenance audit, which needs the real mirror
    (a synthetic test release has no XML, R files or xls to audit against).
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType, MeasurementType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python
    conditions = b.build_conditions()
    phenotypes = pd.read_csv(
        osp.join(raw_dir, b.PHENOTYPES_NAME),
        sep="\t",
        compression="gzip",
        index_col=0,
        float_precision="round_trip",
    )
    phenotypes.index = phenotypes.index.astype(str)
    column_of_env: dict[tuple[Any, ...], str] = {}
    probe = b.Bloom2019Dataset.__new__(b.Bloom2019Dataset)
    probe.conditions = conditions
    signature_of_column: dict[str, tuple[Any, ...]] = {}
    for col, spec in conditions.items():
        env = probe._environment(spec).model_dump()
        signature_of_column[col] = _condition_signature({"environment": env})
        column_of_env[signature_of_column[col]] = col
    # A condition carries no environmental edit iff its environment equals its control
    # plate's environment; a carbon-source swap or a temperature shift is an edit even
    # with no perturbation object, and only the two control plates should qualify.
    unedited_columns = {
        col
        for col, spec in conditions.items()
        if signature_of_column[col]
        == signature_of_column[b.CONTROL_ENVIRONMENT_COLUMN[spec.control_column]]
    }
    absolute_columns = {
        c
        for c, s in conditions.items()
        if s.measurement_type is MeasurementType.colony_size
    }

    # ---- record pass -----------------------------------------------------------
    n_records = 0
    l0_failures: list[dict[str, Any]] = []
    pairs: set[tuple[str, tuple[Any, ...]]] = set()
    n_dups = 0
    n_values = 0
    bad_values: list[dict[str, Any]] = []
    n_mismatch = 0
    mismatch_examples: list[dict[str, Any]] = []
    measurement_by_column: dict[str, set[str]] = {}
    ref_worst = 0.0
    n_ref = 0
    unknown_env = 0
    no_edit_columns: set[str] = set()
    seen_segregants: dict[str, str] = {}  # segregant id -> cross
    blocks_by_segregant: dict[str, list[dict[str, Any]]] = {}
    gap_records_with = 0
    gap_total = 0
    gap_by_reason: Counter[str] = Counter()
    gap_by_field: Counter[str] = Counter()
    gap_worklist: set[str] = set()

    for i, rec in enumerate(records):
        exp = rec["experiment"]
        n_records += 1
        try:
            validate(exp)
        except (ValueError, TypeError) as err:
            l0_failures.append({"index": i, "error": str(err)[:500]})
        genotype = exp["genotype"]
        seg_id = str(genotype["segregant_id"])
        seen_segregants[seg_id] = str(genotype["cross"])
        if seg_id not in blocks_by_segregant:
            blocks_by_segregant[seg_id] = genotype["blocks"]
        signature = _condition_signature(exp)
        column: str | None = column_of_env.get(signature)
        if column is None:
            unknown_env += 1
        key = (seg_id, signature)
        if key in pairs:
            n_dups += 1
        else:
            pairs.add(key)
        value = exp["phenotype"]["environment_response"]
        n_values += 1
        if value is None or math.isnan(value) or math.isinf(value):
            bad_values.append({"index": i, "value": repr(value)})
        elif column is not None:
            released = float(cast(Any, phenotypes.at[seg_id, column]))
            if released != float(value):
                n_mismatch += 1
                if len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        {
                            "segregant": seg_id,
                            "column": column,
                            "stored": value,
                            "released": released,
                        }
                    )
        if column is not None:
            measurement_by_column.setdefault(column, set()).add(
                str(exp["phenotype"]["measurement_type"])
            )
            if column in unedited_columns:
                no_edit_columns.add(column)
        ref_val = rec["reference"]["phenotype_reference"]["environment_response"]
        if ref_val is not None:
            n_ref += 1
            ref_worst = max(ref_worst, abs(float(ref_val)))
        gaps = (exp["phenotype"].get("provenance_gaps") or []) + (
            exp["environment"].get("provenance_gaps") or []
        )
        if gaps:
            gap_records_with += 1
        for gap in gaps:
            gap_total += 1
            reason = str(gap["reason"])
            field = str(gap["field"])
            gap_by_reason[reason] += 1
            gap_by_field[field] += 1
            if reason == ProvenanceGapReason.deferred_pending_source_review:
                gap_worklist.add(field)

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(
        _result(
            Level.L0,
            "structural",
            not l0_failures,
            f"{n_records} records validated"
            if not l0_failures
            else f"{len(l0_failures)}/{n_records} records failed schema validation",
            n_records=n_records,
            n_failures=len(l0_failures),
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
            "pair_uniqueness",
            n_dups == 0,
            f"{len(pairs)} unique (segregant, condition) records, one each"
            if n_dups == 0
            else f"{n_dups} records duplicate an existing (segregant, condition) pair",
            n_pairs=len(pairs),
            n_duplicated=n_dups,
        )
    )

    # ---- raw-file pass: ids, mosaics, markers -----------------------------------
    genotype_rows: dict[str, str] = {}  # id -> cross (from the released matrices)
    n_round_trip = 0
    round_trip_failures: list[str] = []
    invariant_failures: list[str] = []
    counts_by_cross: Counter[str] = Counter()
    block_count_by_cross: dict[str, list[int]] = {}
    marker_ok = 0
    marker_checked = 0
    rng = np.random.default_rng(0)
    for cross in b.CROSSES:
        frame = b.read_marker_matrix(osp.join(raw_dir, f"genotype_{cross}.tsv.gz"))
        markers = b.sorted_markers([str(c) for c in frame.columns])
        values = frame.to_numpy(dtype=np.int8)[:, [m.column for m in markers]]
        n_markers = len(markers)
        for row_id, calls in zip(frame.index.astype(str), values):
            if row_id in genotype_rows:
                invariant_failures.append(f"{row_id}: in two genotype files")
            genotype_rows[row_id] = cross
            counts_by_cross[cross] += 1
            blocks = blocks_by_segregant.get(row_id)
            if blocks is None:
                continue
            block_count_by_cross.setdefault(cross, []).append(len(blocks))
            if sum(int(blk["n_markers"]) for blk in blocks) != n_markers:
                invariant_failures.append(
                    f"{row_id}: n_markers does not sum to {n_markers}"
                )
            typed = [HaplotypeBlock.model_validate(blk) for blk in blocks]
            n_round_trip += 1
            try:
                expanded = b.expand_blocks(typed, markers)
            except ValueError as err:
                round_trip_failures.append(f"{row_id}: {err}")
                continue
            if not np.array_equal(expanded, calls):
                round_trip_failures.append(f"{row_id}: re-expanded calls differ")
        sample = rng.choice(n_markers, min(marker_sample, n_markers), replace=False)
        for idx in sample:
            m = markers[idx]
            key = genome.chr_to_nc[b.CHROM_TO_GENOME_INDEX[m.chromosome]]
            seq = str(
                genome.fasta_dna[key].seq[m.position - 1 : m.position - 1 + len(m.ref)]
            )
            marker_checked += 1
            marker_ok += seq.upper() == m.ref.upper()

    pheno_ids = set(phenotypes.index)
    missing_geno = sorted(pheno_ids - set(genotype_rows))
    missing_pheno = sorted(set(genotype_rows) - pheno_ids)
    wrong_cross = [s for s, c in seen_segregants.items() if genotype_rows.get(s) != c]
    report.add(
        _result(
            Level.L1,
            "id_bijection",
            not missing_geno
            and not missing_pheno
            and not wrong_cross
            and len(seen_segregants) == len(pheno_ids),
            f"{len(pheno_ids)} phenotype ids <-> {len(genotype_rows)} genotype rows; "
            f"{len(seen_segregants)} segregants in the store",
            n_phenotype_ids=len(pheno_ids),
            n_genotype_rows=len(genotype_rows),
            n_stored_segregants=len(seen_segregants),
            missing_genotype=missing_geno[:10],
            missing_phenotype=missing_pheno[:10],
            wrong_cross=wrong_cross[:10],
        )
    )
    report.add(
        provenance_gap_level_result(
            ProvenanceGapCensus(
                n_records=n_records,
                n_records_with_gaps=gap_records_with,
                n_gaps=gap_total,
                by_reason=dict(gap_by_reason),
                by_field=dict(gap_by_field),
                worklist_fields=sorted(gap_worklist),
            )
        )
    )
    report.add(
        _result(
            Level.L2,
            "value_fidelity",
            not bad_values and n_mismatch == 0,
            f"{n_values} values finite and equal to the released tsv cells"
            if not bad_values and n_mismatch == 0
            else f"{len(bad_values)} non-finite, {n_mismatch} differ from the released tsv",
            n_values=n_values,
            n_bad=len(bad_values),
            n_mismatch=n_mismatch,
            bad=bad_values[:20],
            mismatch=mismatch_examples,
        )
    )
    report.add(
        _result(
            Level.L2,
            "mosaic_round_trip",
            not round_trip_failures and n_round_trip == len(seen_segregants),
            f"{n_round_trip} mosaics re-expand to their released marker rows"
            if not round_trip_failures
            else f"{len(round_trip_failures)}/{n_round_trip} mosaics fail the round trip",
            n_checked=n_round_trip,
            failures=round_trip_failures[:20],
        )
    )
    distribution = {
        cross: {
            "min": int(min(c)),
            "median": float(np.median(c)),
            "mean": float(np.mean(c)),
            "max": int(max(c)),
        }
        for cross, c in block_count_by_cross.items()
    }
    report.add(
        _result(
            Level.L2,
            "block_invariants",
            not invariant_failures,
            "blocks ordered, non-overlapping, alternating, n_markers sums per cross"
            if not invariant_failures
            else f"{len(invariant_failures)} block-invariant failures",
            failures=invariant_failures[:20],
            blocks_per_segregant_by_cross=distribution,
        )
    )
    mixed = {c: sorted(t) for c, t in measurement_by_column.items() if len(t) != 1}
    residual_cols = {
        c
        for c, t in measurement_by_column.items()
        if t == {"control_regression_residual"}
    }
    absolute_cols = {
        c for c, t in measurement_by_column.items() if t == {"colony_size"}
    }
    report.add(
        _result(
            Level.L3,
            "measurement_partition",
            not mixed
            and len(residual_cols) == 36
            and absolute_cols == absolute_columns,
            f"{len(residual_cols)} residual + {len(absolute_cols)} absolute conditions",
            n_residual=len(residual_cols),
            n_absolute=len(absolute_cols),
            mixed=mixed,
        )
    )
    report.add(
        _result(
            Level.L3,
            "reference_zero",
            ref_worst == 0.0,
            f"reference response == 0 for all {n_ref} records"
            if ref_worst == 0.0
            else f"reference response not identically 0: max|v|={ref_worst:.3g}",
            n_values=n_ref,
            worst_abs=ref_worst,
        )
    )
    report.add(
        _result(
            Level.L3,
            "conditions_documented",
            unknown_env == 0
            and set(measurement_by_column) == set(conditions)
            and no_edit_columns == absolute_columns,
            f"{len(measurement_by_column)}/{len(conditions)} documented conditions seen; "
            f"{unknown_env} records with an undocumented environment; "
            f"no-edit columns = {sorted(no_edit_columns)}",
            n_unknown=unknown_env,
            missing_conditions=sorted(set(conditions) - set(measurement_by_column)),
            no_edit_columns=sorted(no_edit_columns),
        )
    )
    if not skip_sourced_values:
        report.add(_l3_sourced_values(conditions, raw_mirror))
    report.add(_l4_parents(raw_dir, assembly_index_path))
    report.add(_l4_cross_pairs_and_counts(raw_dir, counts_by_cross))
    ratio = marker_ok / marker_checked if marker_checked else 0.0
    report.add(
        _result(
            Level.L4,
            "marker_reference",
            ratio >= min_marker_reference,
            f"{ratio:.4f} of {marker_checked} sampled markers carry the S288C reference "
            f"base as ref (>= {min_marker_reference})",
            n_checked=marker_checked,
            n_match=marker_ok,
            ratio=ratio,
        )
    )
    overlap = len(gene_set & sgd_genes) / len(gene_set) if gene_set else 0.0
    report.add(
        _result(
            Level.L4,
            "gene_containment_sgd",
            overlap == 1.0 and bool(gene_set),
            f"{overlap:.3f} of {len(gene_set)} spanned genes are S288C reference genes",
            n_measured=len(gene_set),
            n_in_sgd=len(gene_set & sgd_genes),
            overlap=overlap,
            missing_examples=sorted(gene_set - sgd_genes)[:20],
        )
    )
    return report


def _l3_sourced_values(
    conditions: dict[str, Any], raw_mirror: str | Path
) -> LevelResult:
    """Audit the text-anchored SourcedValues and the xls-anchored dose quotes."""
    from torchcell.datamodels import media as media_module

    audits: list[LevelResult] = []
    text_values: list[SourcedValue] = [
        SourcedValue(
            value=b.N_SAMPLES,
            provenance=Provenance(
                source_uri=f"paper/{b.XML_NAME}",
                citation_key=b.CITATION_KEY,
                sha256=b.manifest_sha256(
                    b.load_manifest(str(Path(raw_mirror).parents[1])),
                    f"paper/{b.XML_NAME}",
                ),
            ),
            quote=b.DUPLICATE_QUOTE,
        ),
        SourcedValue(
            value=b.INCUBATION_HOURS,
            provenance=Provenance(
                source_uri=f"paper/{b.XML_NAME}",
                citation_key=b.CITATION_KEY,
                sha256=b.manifest_sha256(
                    b.load_manifest(str(Path(raw_mirror).parents[1])),
                    f"paper/{b.XML_NAME}",
                ),
            ),
            quote=b.INCUBATION_QUOTE,
        ),
        SourcedValue(
            value=b.CALL_METHOD,
            provenance=Provenance(
                source_uri="code/mapping.R",
                citation_key=b.CITATION_KEY,
                sha256=b.manifest_sha256(
                    b.load_manifest(str(Path(raw_mirror).parents[1])), "code/mapping.R"
                ),
            ),
            quote=b.CALL_METHOD_QUOTE,
        ),
    ]
    library_root = Path(raw_mirror).parent
    for sv in text_values:
        audits.append(audit_sourced_value(sv, library_root))
    for constant in media_module.MEDIA_LIBRARY.values():
        for component in constant.components:
            for sv in component.provenance:
                if sv.provenance.citation_key != b.CITATION_KEY:
                    continue
                if sv.provenance.source_uri.endswith(".xls"):
                    audits.append(_audit_xls_quote(sv, library_root))
    for spec in conditions.values():
        if spec.dose_quote:
            audits.append(
                _audit_xls_quote(
                    SourcedValue(
                        value=spec.column,
                        provenance=Provenance(
                            source_uri=f"data/{b.XLS_NAME}",
                            citation_key=b.CITATION_KEY,
                            sha256=media_module._BLOOM2019_XLS_SHA,
                        ),
                        quote=spec.dose_quote,
                    ),
                    library_root,
                )
            )
    failed = [a.message for a in audits if not a.passed]
    return _result(
        Level.L3,
        "sourced_values",
        not failed,
        f"{len(audits)} sourced values audited against the raw mirror"
        if not failed
        else f"{len(failed)}/{len(audits)} sourced values failed audit",
        n_audited=len(audits),
        failures=failed[:20],
    )


def _xls_row_strings(path: str | Path) -> set[str]:
    """Every row of both sheets as ``a | b | c`` of its non-empty cells, the quote form."""
    rows: set[str] = set()
    for sheet in ("Crosses and Strains", "Phenotypes"):
        df = pd.read_excel(path, sheet_name=sheet, header=None, engine="xlrd")
        for _, row in df.iterrows():
            cells = [str(x).strip() for x in row.tolist() if str(x) != "nan"]
            rows.add(" | ".join(cells))
    return rows


def _audit_xls_quote(sv: SourcedValue, library_root: Path) -> LevelResult:
    """The binary xls cannot be substring-searched; re-read it and match the row form.

    The quote is the ``a | b | c`` join of a row's non-empty cells, or a prefix of it.
    """
    path = sv.source_path(library_root)
    from torchcell.verification.report import sha256_file

    integrity = sha256_file(path) == sv.provenance.sha256
    present = integrity and any(
        row.startswith(sv.quote) for row in _xls_row_strings(path)
    )
    return _result(
        Level.L3,
        "provenance_audit",
        integrity and present,
        f"xls row {'found' if present else 'NOT found'} for {sv.quote[:40]!r}"
        if integrity
        else f"sha256 drift on {path.name}",
        source=sv.provenance.source_uri,
        quote=sv.quote,
        sha256_ok=integrity,
        quote_present=present,
    )


def _l4_parents(raw_dir: str | Path, assembly_index_path: str | Path) -> LevelResult:
    index = b.read_assembly_index(assembly_index_path)
    missing = [
        f"{label} -> {peter}"
        for label, peter in b.PARENT_PETER_ID.items()
        if peter is not None and peter not in index
    ]
    return _result(
        Level.L4,
        "parents_pinned",
        not missing,
        f"{sum(1 for p in b.PARENT_PETER_ID.values() if p)} parents resolve in the 1011 "
        "assembly member index; BY is the S288C reference"
        if not missing
        else f"parents absent from the assembly index: {missing}",
        missing=missing,
        assembly_sha256=b.ASSEMBLY_TAR_SHA256,
    )


def _l4_cross_pairs_and_counts(
    raw_dir: str | Path, counts: Counter[str]
) -> LevelResult:
    info = b.read_cross_table(
        osp.join(raw_dir, b.XLS_NAME), osp.join(raw_dir, b.README_NAME)
    )
    bad = {
        cross: (counts.get(cross), i.n_segregants_xls, b.EXPECTED_SEGREGANTS[cross])
        for cross, i in info.items()
        if counts.get(cross) != i.n_segregants_xls
        or i.n_segregants_xls != b.EXPECTED_SEGREGANTS[cross]
    }
    return _result(
        Level.L4,
        "cross_pairs_and_counts",
        not bad and sum(counts.values()) == b.N_SEGREGANTS,
        f"16 crosses: xls parent pairs match the README pairs; per-cross segregant "
        f"counts match the xls (sum {sum(counts.values())})"
        if not bad
        else f"count mismatches (stored, xls, expected): {bad}",
        counts=dict(counts),
        mismatches=bad,
    )


def segregant_gene_set(preprocess_dir: str | Path) -> set[str]:
    """The dataset's derived gene set (``preprocess/gene_set.json``)."""
    import json

    with open(osp.join(preprocess_dir, "gene_set.json")) as fh:
        return set(json.load(fh))


__all__ = ["verify_segregant_growth_streaming", "segregant_gene_set", "Record"]


if __name__ == "__main__":
    sys.exit(0)
