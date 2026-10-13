# torchcell/verification/segregant_expression
# [[torchcell.verification.segregant_expression]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/segregant_expression
# Test file: tests/torchcell/verification/test_segregant_expression.py
"""L0-L4 verifier for segregant single-cell expression datasets (Boocock 2025).

A ``SegregantExpressionExperiment`` carries a ``SegregantGenotype`` (haplotype mosaic)
and a ``SingleCellExpressionPhenotype`` (UMI counts), so neither the segregant growth
verifier (Bloom's conditions and residuals) nor the expression verifiers (gene-keyed
perturbations) apply. One streaming pass over the records, panel by panel, re-reads the
panel's release from ``raw/`` and compares:

- L0 ``structural``: every experiment validates against ``ExperimentType`` and is a
  ``segregant_expression`` record.
- L1 ``count`` and ``panel_counts`` (per-panel record counts equal the release's cells,
  or segregants for the pooled panel); ``segregant_uniqueness``; the shared rules of
  :mod:`torchcell.verification.common` (gap census, compound identity, media
  membership, uncertainty sanity).
- L2 ``value_fidelity``: every record's counts, ``total_umis``, cell-cycle and library
  partitions, and summed cells (barcodes, libraries, per-cell ``nUMI``) equal the
  release's row (de novo) or the segregant's matched cells (pooled), and every record
  gaps ``marker_umi_count`` (no cassette is counted apart from the genes);
  ``mosaic_round_trip``: de novo blocks re-expand to the released posterior (called
  parent exact, posterior within half the grid step), pooled blocks to the lookup calls
  exactly; ``reference_pseudobulk``: each panel's reference is the sum of its records
  and lists every cell of the panel.
- L3 ``table_s4``: cells, analysis transcripts and median UMI per panel equal Table S4;
  ``pooled_panel``: 393 segregants, median 17 cells, 277 with more than ten, and the
  median per-cell genotype agreement with whole-genome sequencing within a stated
  tolerance of the paper's 92.5% (measured here on the LD-pruned markers shared with the
  lookup genotypes, not the paper's full marker set); ``sourced_values``: every quote is
  still in the pinned ``paper.md``.
- L4 ``parents_pinned``; ``mat_orientation`` (the MATa-sorted cells inherit the MATa
  parent at the MAT locus, which fixes which parent the posterior is a probability of);
  ``marker_reference`` (sampled markers' ref allele is the S288C base);
  ``gene_containment_sgd`` (the analysis transcripts and the derived gene set are S288C
  reference genes).
"""

from __future__ import annotations

import os.path as osp
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from torchcell.datamodels.schema import HaplotypeBlock
from torchcell.datasets.scerevisiae import boocock2025 as bc
from torchcell.datasets.scerevisiae.bloom2019 import expand_blocks
from torchcell.verification.common import (
    SharedRecordRules,
    declared_first_union_validator,
)
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

Record = dict[str, Any]

#: The MAT locus on chrIII (SGD R64: MATALPHA2/MATALPHA1, ~198.7-201.2 kb).
MAT_CHROMOSOME = "chrIII"
MAT_POSITION = 200_000
#: Tolerance on the paper's median per-cell genotype agreement (92.5%), in fraction:
#: the release does not carry the full-marker comparison the paper ran, only the
#: LD-pruned posterior, so the recomputation is a check of the same quantity on a
#: subset of its markers.
AGREEMENT_TOLERANCE = 0.01
PAPER_AGREEMENT = 0.925
PAPER_CELLS_PER_SEGREGANT = 17.0
PAPER_SEGREGANTS_OVER_TEN = 277


def _result(
    level: Level, name: str, passed: bool, message: str, **details: Any
) -> LevelResult:
    return LevelResult(
        level=level, name=name, passed=passed, message=message, details=details
    )


def panel_of(genotype: dict[str, Any], library_panel: dict[str, str]) -> str:
    """The panel of a stored genotype: the library prefix of a cell id, else pooled."""
    segregant = str(genotype["segregant_id"])
    if ":" in segregant:
        return library_panel[segregant.split(":", 1)[0]]
    return next(p.panel for p in bc.PANELS.values() if p.pooled)


def _counts_row(release: bc.PanelRelease, rows: list[int]) -> dict[str, int]:
    summed = np.asarray(release.counts[rows].sum(axis=0)).ravel()
    return {release.transcripts[j]: int(summed[j]) for j in np.flatnonzero(summed)}


def _nearest_marker(markers: list[Any]) -> int:
    on_chrom = [i for i, m in enumerate(markers) if m.chromosome == MAT_CHROMOSOME]
    return min(on_chrom, key=lambda i: abs(markers[i].position - MAT_POSITION))


class _PanelState:
    """Running tallies for the panel whose records are streaming past."""

    def __init__(self, spec: bc.PanelSpec, raw_dir: str | Path) -> None:
        self.spec = spec
        self.release = bc.read_panel_release(osp.join(raw_dir, spec.segdata_name))
        self.row_of = {
            f"{lib}:{bc_}": i
            for i, (lib, bc_) in enumerate(
                zip(self.release.libraries, self.release.barcodes)
            )
        }
        self.n_records = 0
        self.reference_sum: Counter[str] = Counter()
        self.reference: dict[str, Any] | None = None
        self.cells_of: dict[str, list[int]] = {}
        # Only the pooled panel has a match table and lookup genotypes; a de novo panel
        # keeps them empty and never reads them.
        self.match = bc.SegregantMatch(segregant=[], correlation=[], log_n_umi=[])
        self.lookup: tuple[list[str], list[Any], Any] = ([], [], np.zeros((0, 0)))
        if spec.pooled:
            self.match = bc.read_segregant_match(osp.join(raw_dir, bc.SEGMATCH_NAME))
            for row, segregant in enumerate(self.match.segregant):
                self.cells_of.setdefault(segregant, []).append(row)
            self.lookup = bc.read_lookup_genotypes(osp.join(raw_dir, bc.GDATA_NAME))


def verify_segregant_expression_streaming(
    records: Iterable[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    raw_dir: str | Path,
    library_root: str | Path,
    assembly_index: dict[str, str],
    genome: Any,
    sgd_genes: set[str],
    gene_set: set[str],
    panels: dict[str, bc.PanelSpec] | None = None,
    sourced: list[SourcedValue] | None = None,
    marker_sample: int = 500,
    min_marker_reference: float = 0.95,
    min_transcript_containment: float = 0.99,
    expected_pooled_segregants: int = bc.N_POOLED_SEGREGANTS,
    check_paper_numbers: bool = True,
) -> VerificationReport:
    """Single-pass L0-L4 gate (see the module docstring).

    ``panels`` defaults to the four released panels; a synthetic test release passes its
    own. ``check_paper_numbers`` gates the L3 comparisons against the paper's printed
    numbers (Table S4 medians, 17 cells, 277, 92.5%), which a synthetic release cannot
    reproduce. ``genome`` supplies ``fasta_dna`` and ``chr_to_nc`` for the marker
    reference check.
    """
    specs = panels if panels is not None else bc.PANELS
    library_panel = {lib: p.panel for p in specs.values() for lib in p.libraries}
    validate: Callable[[Any], object] = declared_first_union_validator()
    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    shared = SharedRecordRules()

    n_records = 0
    l0_failures: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    n_dup = 0
    finished: dict[str, dict[str, Any]] = {}
    value_failures: list[str] = []
    round_trip_failures: list[str] = []
    worst_posterior_error = 0.0
    blocks_by_panel: dict[str, list[int]] = {}
    states: dict[str, _PanelState] = {}

    def close(state: _PanelState) -> None:
        release = state.release
        all_rows = list(range(len(release.barcodes)))
        expected_reference = _counts_row(release, all_rows)
        reference = state.reference or {}
        finished[state.spec.panel] = {
            "n_records": state.n_records,
            "reference_ok": (
                dict(state.reference_sum) == expected_reference
                and reference.get("umi_count") == expected_reference
                and reference.get("n_cells") == len(all_rows)
                and reference.get("total_umis") == sum(release.n_umi)
                and reference.get("cell_barcodes") == release.barcodes
                and reference.get("cell_libraries") == release.libraries
                and reference.get("cell_total_umis") == release.n_umi
            ),
            "n_cells": len(release.barcodes),
            "n_transcripts": len(release.transcripts),
            "median_umi": float(np.median(release.n_umi)),
            "mat_mean_parent_2": float(
                np.mean(release.posterior[:, _nearest_marker(release.markers)])
            ),
            "markers": release.markers,
            "transcripts": list(release.transcripts),
        }
        if state.spec.pooled:
            finished[state.spec.panel]["pooled"] = _pooled_numbers(state)

    for i, rec in enumerate(records):
        exp = rec["experiment"]
        n_records += 1
        shared.add(rec)
        try:
            validate(exp)
            if exp["experiment_type"] != "segregant_expression":
                raise TypeError(f"experiment_type {exp['experiment_type']!r}")
        except (ValueError, TypeError) as err:
            l0_failures.append({"index": i, "error": str(err)[:300]})
            continue
        genotype = exp["genotype"]
        panel = panel_of(genotype, library_panel)
        # LMDB keys iterate in byte order ("1", "10", "100", ...), so the panels
        # interleave: every panel's release stays loaded until the pass ends.
        if panel not in states:
            states[panel] = _PanelState(specs[panel], raw_dir)
            states[panel].reference = rec["reference"]["phenotype_reference"]
        state = states[panel]
        reference = rec["reference"]["phenotype_reference"]
        if reference is not state.reference and reference != state.reference:
            value_failures.append(f"{panel}: a record names a different reference")
        key = (panel, str(genotype["segregant_id"]))
        n_dup += key in seen
        seen.add(key)
        state.n_records += 1
        phenotype = exp["phenotype"]
        state.reference_sum.update(phenotype["umi_count"])
        blocks = [HaplotypeBlock.model_validate(b) for b in genotype["blocks"]]
        blocks_by_panel.setdefault(panel, []).append(len(blocks))
        release = state.release
        if state.spec.pooled:
            rows = state.cells_of.get(key[1], [])
            ids, markers, calls = state.lookup
            coded = np.where(calls[ids.index(key[1])] > 0, 2, 1).astype(np.int8)
            try:
                if not np.array_equal(expand_blocks(blocks, markers), coded):
                    round_trip_failures.append(f"{key[1]}: calls differ")
            except ValueError as err:
                round_trip_failures.append(f"{key[1]}: {err}")
        else:
            row = state.row_of.get(key[1])
            rows = [] if row is None else [row]
            if row is not None:
                try:
                    parent, level = bc.expand_posterior_blocks(blocks, release.markers)
                except ValueError as err:
                    round_trip_failures.append(f"{key[1]}: {err}")
                else:
                    p2 = release.posterior[row]
                    called = np.where(p2 > 0.5, 2, 1)
                    q = np.where(called == 2, p2, 1.0 - p2)
                    error = float(np.max(np.abs(q - level)))
                    worst_posterior_error = max(worst_posterior_error, error)
                    if not np.array_equal(parent, called):
                        round_trip_failures.append(f"{key[1]}: called parent differs")
                    elif error > bc.POSTERIOR_STEP / 2 + 1e-9:
                        round_trip_failures.append(
                            f"{key[1]}: posterior off by {error:.4f}"
                        )
        if not rows:
            value_failures.append(f"{key}: no release cells")
            continue
        expected = {
            "umi_count": _counts_row(release, rows),
            "marker_umi_count": None,
            "total_umis": sum(release.n_umi[r] for r in rows),
            "n_cells": len(rows),
            "cell_cycle_stage_counts": dict(Counter(release.stages[r] for r in rows)),
            "library_cell_counts": dict(Counter(release.libraries[r] for r in rows)),
            "cell_barcodes": [release.barcodes[r] for r in rows],
            "cell_libraries": [release.libraries[r] for r in rows],
            "cell_total_umis": [release.n_umi[r] for r in rows],
        }
        for field, value in expected.items():
            if phenotype[field] != value:
                value_failures.append(f"{key}: {field} differs from the release")
    for state in states.values():
        close(state)

    report.add(
        _result(
            Level.L0,
            "structural",
            not l0_failures and n_records > 0,
            f"{n_records - len(l0_failures)}/{n_records} records validate as "
            "segregant_expression experiments",
            n_records=n_records,
            failures=l0_failures[:10],
        )
    )
    report.add(
        _result(
            Level.L1,
            "count",
            n_records == expected_count,
            f"{n_records} records (expected {expected_count})",
            n_records=n_records,
            expected=expected_count,
        )
    )
    expected_panel = {
        p.panel: (
            expected_pooled_segregants if p.pooled else finished[p.panel]["n_cells"]
        )
        for p in specs.values()
        if p.panel in finished
    }
    observed_panel = {p: v["n_records"] for p, v in finished.items()}
    report.add(
        _result(
            Level.L1,
            "panel_counts",
            observed_panel == expected_panel and set(finished) == set(specs),
            f"records per panel {observed_panel}",
            observed=observed_panel,
            expected=expected_panel,
        )
    )
    report.add(
        _result(
            Level.L1,
            "segregant_uniqueness",
            n_dup == 0,
            f"{len(seen)} unique (panel, segregant) keys, {n_dup} duplicates",
            n_unique=len(seen),
            n_duplicates=n_dup,
        )
    )
    for result in shared.results():
        report.add(result)
    report.add(
        _result(
            Level.L2,
            "value_fidelity",
            not value_failures,
            "every record's counts, totals, partitions and summed cells equal the "
            "release; every record gaps marker_umi_count"
            if not value_failures
            else f"{len(value_failures)} fields differ from the release",
            failures=value_failures[:20],
        )
    )
    distribution = {
        panel: {
            "min": int(min(c)),
            "median": float(np.median(c)),
            "mean": round(float(np.mean(c)), 1),
            "max": int(max(c)),
        }
        for panel, c in blocks_by_panel.items()
    }
    report.add(
        _result(
            Level.L2,
            "mosaic_round_trip",
            not round_trip_failures,
            f"every mosaic re-expands to its release (worst de novo posterior error "
            f"{worst_posterior_error:.4f}, bound {bc.POSTERIOR_STEP / 2})"
            if not round_trip_failures
            else f"{len(round_trip_failures)} mosaics fail the round trip",
            failures=round_trip_failures[:20],
            worst_posterior_error=worst_posterior_error,
            blocks_per_record_by_panel=distribution,
        )
    )
    report.add(
        _result(
            Level.L2,
            "reference_pseudobulk",
            all(v["reference_ok"] for v in finished.values()),
            "each panel's reference is the sum of its cells",
            by_panel={p: v["reference_ok"] for p, v in finished.items()},
        )
    )
    if check_paper_numbers:
        report.add(_l3_table_s4(specs, finished))
        pooled = [v["pooled"] for v in finished.values() if "pooled" in v]
        report.add(_l3_pooled(pooled[0] if pooled else None))
    report.add(_l3_sourced(sourced or [], library_root))
    report.add(_l4_parents(specs, assembly_index))
    report.add(_l4_mat(specs, finished))
    report.add(
        _l4_marker_reference(finished, genome, marker_sample, min_marker_reference)
    )
    report.add(_l4_genes(finished, gene_set, sgd_genes, min_transcript_containment))
    return report


def _pooled_numbers(state: _PanelState) -> dict[str, Any]:
    """Segregant count, cells per segregant, and per-cell genotype agreement."""
    per = np.array([len(rows) for rows in state.cells_of.values()])
    ids, markers, calls = state.lookup
    key = {(m.chromosome, m.position, m.ref, m.alt): j for j, m in enumerate(markers)}
    pairs = [
        (i, key[(m.chromosome, m.position, m.ref, m.alt)])
        for i, m in enumerate(state.release.markers)
        if (m.chromosome, m.position, m.ref, m.alt) in key
    ]
    cell_idx = np.array([a for a, _ in pairs])
    lookup_idx = np.array([c for _, c in pairs])
    row = {s: i for i, s in enumerate(ids)}
    hmm = state.release.posterior[:, cell_idx] > 0.5
    wgs = calls[[row[s] for s in state.match.segregant]][:, lookup_idx] > 0
    agreement = (hmm == wgs).mean(axis=1)
    mat = _nearest_marker(markers)
    matched = sorted(state.cells_of)
    return {
        "n_segregants": len(state.cells_of),
        "median_cells": float(np.median(per)),
        "over_ten": int((per > 10).sum()),
        "n_shared_markers": len(pairs),
        "median_agreement": float(np.median(agreement)),
        "mat_parent_2_fraction": float(
            np.mean(calls[[row[s] for s in matched], mat] > 0)
        ),
    }


def _l3_table_s4(
    specs: dict[str, bc.PanelSpec], finished: dict[str, dict[str, Any]]
) -> LevelResult:
    rows = {
        p: (v["n_cells"], v["n_transcripts"], v["median_umi"])
        for p, v in finished.items()
    }
    expected = {
        p.panel: (p.expected_cells, p.expected_transcripts, p.median_umi)
        for p in specs.values()
    }
    return _result(
        Level.L3,
        "table_s4",
        rows == expected,
        "cells, analysis transcripts and median UMI per panel equal Table S4"
        if rows == expected
        else "a panel differs from Table S4",
        observed=rows,
        expected=expected,
    )


def _l3_pooled(numbers: dict[str, Any] | None) -> LevelResult:
    if numbers is None:
        return _result(Level.L3, "pooled_panel", False, "no pooled panel was streamed")
    passed = (
        numbers["n_segregants"] == bc.N_POOLED_SEGREGANTS
        and numbers["median_cells"] == PAPER_CELLS_PER_SEGREGANT
        and numbers["over_ten"] == PAPER_SEGREGANTS_OVER_TEN
        and abs(numbers["median_agreement"] - PAPER_AGREEMENT) <= AGREEMENT_TOLERANCE
    )
    return _result(
        Level.L3,
        "pooled_panel",
        passed,
        f"{numbers['n_segregants']} segregants, median {numbers['median_cells']} cells, "
        f"{numbers['over_ten']} with more than ten; median per-cell agreement "
        f"{numbers['median_agreement']:.4f} on {numbers['n_shared_markers']} shared "
        f"markers (paper {PAPER_AGREEMENT}, tolerance {AGREEMENT_TOLERANCE})",
        **numbers,
    )


def _l3_sourced(sourced: list[SourcedValue], library_root: str | Path) -> LevelResult:
    audits = [audit_sourced_value(sv, library_root) for sv in sourced]
    failed = [a.details for a in audits if not a.passed]
    return _result(
        Level.L3,
        "sourced_values",
        bool(audits) and not failed,
        f"{len(audits) - len(failed)}/{len(audits)} quotes present in the pinned bytes",
        failures=failed[:10],
    )


def _l4_parents(
    specs: dict[str, bc.PanelSpec], assembly_index: dict[str, str]
) -> LevelResult:
    from torchcell.datasets.scerevisiae.bloom2019 import PARENT_PETER_ID

    labels = {label for p in specs.values() for label in (p.parent_1, p.parent_2)}
    peter = {label: PARENT_PETER_ID[bc.BLOOM_LABEL[label]] for label in labels}
    missing = sorted(
        label
        for label, pid in peter.items()
        if pid is not None and pid not in assembly_index
    )
    return _result(
        Level.L4,
        "parents_pinned",
        not missing,
        f"{len(labels)} parents; Peter ids {sorted(p for p in peter.values() if p)} "
        "in the assembly member index; BY is the S288C reference",
        missing=missing,
        peter_ids=peter,
    )


def _l4_mat(
    specs: dict[str, bc.PanelSpec], finished: dict[str, dict[str, Any]]
) -> LevelResult:
    """MATa-sorted cells carry their MATa parent at MAT: P(parent 2) < 0.5 iff it is 1."""
    observed: dict[str, float] = {}
    wrong: list[str] = []
    for panel, values in finished.items():
        spec = specs[panel]
        if spec.pooled:
            fraction = float(values["pooled"]["mat_parent_2_fraction"])
        else:
            fraction = float(values["mat_mean_parent_2"])
        observed[panel] = fraction
        if (fraction > 0.5) != (spec.mata_parent == 2):
            wrong.append(panel)
    return _result(
        Level.L4,
        "mat_orientation",
        not wrong and bool(observed),
        f"P(parent 2) at the marker nearest MAT per panel {observed}; the MATa parent "
        "is parent 2 only where the value exceeds 0.5",
        observed=observed,
        wrong=wrong,
    )


def _l4_marker_reference(
    finished: dict[str, dict[str, Any]],
    genome: Any,
    marker_sample: int,
    min_ratio: float,
) -> LevelResult:
    from torchcell.datasets.scerevisiae.bloom2019 import CHROM_TO_GENOME_INDEX

    rng = np.random.default_rng(0)
    checked = matched = 0
    for values in finished.values():
        markers = values["markers"]
        for idx in rng.choice(
            len(markers), min(marker_sample, len(markers)), replace=False
        ):
            m = markers[idx]
            key = genome.chr_to_nc[CHROM_TO_GENOME_INDEX[m.chromosome]]
            seq = str(
                genome.fasta_dna[key].seq[m.position - 1 : m.position - 1 + len(m.ref)]
            )
            checked += 1
            matched += seq.upper() == m.ref.upper()
    ratio = matched / checked if checked else 0.0
    return _result(
        Level.L4,
        "marker_reference",
        ratio >= min_ratio,
        f"{ratio:.4f} of {checked} sampled markers carry the S288C base as ref "
        f"(>= {min_ratio})",
        n_checked=checked,
        n_match=matched,
        ratio=ratio,
    )


def _l4_genes(
    finished: dict[str, dict[str, Any]],
    gene_set: set[str],
    sgd_genes: set[str],
    min_transcript_containment: float,
) -> LevelResult:
    transcripts = {t for v in finished.values() for t in v["transcripts"]}
    outside = sorted(transcripts - sgd_genes)
    containment = 1 - len(outside) / len(transcripts) if transcripts else 0.0
    gene_ok = bool(gene_set) and gene_set <= sgd_genes
    return _result(
        Level.L4,
        "gene_containment_sgd",
        gene_ok and containment >= min_transcript_containment,
        f"{len(transcripts) - len(outside)}/{len(transcripts)} analysis transcripts are "
        f"S288C reference genes ({containment:.4f}, >= {min_transcript_containment}); "
        f"{len(gene_set)} spanned genes, all reference: {gene_ok}",
        n_transcripts=len(transcripts),
        outside_reference=outside[:50],
        n_gene_set=len(gene_set),
    )
