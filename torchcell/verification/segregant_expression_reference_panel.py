# torchcell/verification/segregant_expression_reference_panel
# [[torchcell.verification.segregant_expression_reference_panel]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/segregant_expression_reference_panel
# Test file: tests/torchcell/verification/test_segregant_expression_reference_panel.py
"""L0-L4 verifier for pooled segregant expression matched to a reference panel.

The pooled design of N'Guessan 2025: cells of many pre-genotyped segregants grow in one
culture, each cell is assigned to a strain of a reference panel, and a record is one
strain, its assigned cells summed and the panel's genotype posterior of that strain. The
release holds the cell-to-strain assignment, not a per-cell genotype, and the panel's
genotypes come from another deposit (Nguyen Ba 2022 on Dryad), joined through a
batch-1 index. So beyond re-reading the release, this verifier tests the JOIN: if the
index-to-strain map were wrong, the cells' expression would be independent of the
genotype they are stored with.

One streaming pass over the records re-reads the release from ``raw/`` and compares:

- L0 ``structural``: every experiment validates against ``ExperimentType`` and is a
  ``segregant_expression`` record.
- L1 ``count``; ``segregant_uniqueness``; the shared rules of
  :mod:`torchcell.verification.common`.
- L2 ``value_fidelity``: each record's counts, ``total_umis``, ``n_cells`` and library
  partition equal the sum over its assigned cells, and the stage partition is a typed
  gap; ``mosaic_round_trip``: every mosaic re-expands to the strain's released posterior
  (called parent exact, an exact 0.5 a tie, posterior within half the grid step);
  ``reference_pseudobulk``: the shared reference is the sum of every assigned cell.
- L3 ``release_numbers``: barcodes, median UMI and median genes per cell, mean cells per
  assigned strain and the batch-1 size equal the paper's Table 1 and Methods;
  ``sourced_values``: every quote is still in its pinned bytes (paper quotes in the
  literature mirror, release quotes in the raw mirror).
- L4 ``parents_pinned``; ``marker_reference`` (sampled SNPs' BY allele is the S288C
  base); ``gene_containment_sgd``; ``mat_fixation`` (the panel's expression is MATa, so
  its genotype at MAT must be fixed); ``cis_concordance`` (expression tracks the stored
  genotype at the gene's own locus far beyond a permuted strain-to-genotype map).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from torchcell.datamodels.schema import HaplotypeBlock
from torchcell.datasets.scerevisiae import nguessan2025 as ng
from torchcell.datasets.scerevisiae.boocock2025 import POSTERIOR_STEP
from torchcell.verification.common import SharedRecordRules
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
#: Mating-type-specific transcripts (SGD): a-specific and alpha-specific genes.
A_SPECIFIC = ("YFL026W", "YDR461W", "YNL145W", "YIL015W", "YKL209C", "YGL032C")
ALPHA_SPECIFIC = ("YPL187W", "YGL089C", "YKL178C", "YJR004C")
#: A MATa panel's reference must carry this many a-specific UMIs per alpha-specific one.
MIN_A_TO_ALPHA = 10.0
#: A fixed locus: the mean posterior at MAT within this distance of 0 or 1.
MAT_FIXATION_TOLERANCE = 0.05
#: cis test: genes detected in at least this fraction of strains are tested.
CIS_MIN_DETECTION = 0.5
#: cis test: the null threshold is this quantile of |r| under permuted strain maps, so
#: the null exceedance rate is 1 - CIS_NULL_QUANTILE by construction.
CIS_NULL_QUANTILE = 0.99
#: cis test passes when the observed exceedance is at least this multiple of the null
#: rate (5x of 1% = 5% of tested genes). Set before the build, not tuned on it.
MIN_CIS_ENRICHMENT = 5.0


def _result(
    level: Level, name: str, passed: bool, message: str, **details: Any
) -> LevelResult:
    return LevelResult(
        level=level, name=name, passed=passed, message=message, details=details
    )


class PanelRelease:
    """Everything the checks re-read from ``raw/``, loaded once."""

    def __init__(self, raw_dir: str | Path) -> None:
        """Read the matrices, assignment, panel index, SNPs and assigned genotypes."""
        raw = Path(raw_dir)
        self.release = ng.read_expression([raw / n for n in ng.EXPRESSION_PARTS])
        self.genes = ng.read_gene_table(raw / ng.GENE_TABLE)
        self.assignment = ng.read_assignment(raw / ng.ASSIGNMENT_TABLE)
        self.cells = ng.assigned_cells(self.assignment)
        self.numbers = ng.read_batch_numbers(raw / ng.SEGREGANT_INFO_NAME)
        self.markers, self.order = ng.read_snp_markers(raw / ng.SNP_LIST_NAME)
        self.index_of = {self.numbers[i]: i for i in self.cells}
        genotypes = ng.read_reference_genotypes(
            raw / ng.GENO_NAME, set(self.index_of), len(self.markers)
        )
        self.posterior = {n: g[self.order] for n, g in genotypes.items()}

    def counts(self, rows: list[int]) -> dict[str, int]:
        """Summed nonzero counts of the release rows."""
        summed = np.asarray(self.release.counts[rows].sum(axis=0)).ravel()
        return {self.release.genes[j]: int(summed[j]) for j in np.flatnonzero(summed)}


def verify_reference_panel_expression_streaming(
    records: Iterable[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    raw_dir: str | Path,
    library_root: str | Path,
    raw_root: str | Path,
    assembly_index: dict[str, str],
    genome: Any,
    sgd_genes: set[str],
    gene_set: set[str],
    gene_midpoints: dict[str, tuple[str, int]],
    sourced: list[SourcedValue],
    release_sourced: list[SourcedValue],
    marker_sample: int = 500,
    min_marker_reference: float = 0.95,
    min_gene_containment: float = 0.99,
    cis_permutations: int = 20,
    seed: int = 0,
    check_paper_numbers: bool = True,
) -> VerificationReport:
    """Single-pass L0-L4 gate (see the module docstring).

    ``gene_midpoints`` maps a gene to its (chromosome, midpoint) on S288C, for the cis
    test; ``genome`` supplies ``fasta_dna`` and ``chr_to_nc`` for the marker reference
    check. ``check_paper_numbers`` gates the L3 comparison with the paper's printed
    numbers, which a synthetic release cannot reproduce.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    panel = PanelRelease(raw_dir)
    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python
    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    shared = SharedRecordRules()

    n_records = 0
    l0_failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    n_dup = 0
    value_failures: list[str] = []
    round_trip_failures: list[str] = []
    worst_error = 0.0
    n_blocks: list[int] = []
    reference: dict[str, Any] | None = None
    reference_mismatch = 0
    reference_sum: dict[str, int] = {}
    parents: list[dict[str, Any]] = []

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
        if reference is None:
            reference = rec["reference"]["phenotype_reference"]
        elif rec["reference"]["phenotype_reference"] != reference:
            reference_mismatch += 1
        genotype = exp["genotype"]
        if not parents:
            parents = [genotype["parent_1"], genotype["parent_2"]]
        key = str(genotype["segregant_id"])
        n_dup += key in seen
        seen.add(key)
        phenotype = exp["phenotype"]
        for gene, count in phenotype["umi_count"].items():
            reference_sum[gene] = reference_sum.get(gene, 0) + count
        number = int(key)
        index = panel.index_of.get(number)
        if index is None:
            value_failures.append(f"{key}: not an assigned batch-1 strain")
            continue
        rows = panel.cells[index]
        expected = {
            "umi_count": panel.counts(rows),
            "total_umis": int(panel.release.cell_totals[rows].sum()),
            "n_cells": len(rows),
            "library_cell_counts": {ng.LIBRARY_LABEL: len(rows)},
            "cell_cycle_stage_counts": None,
        }
        for field, value in expected.items():
            if phenotype[field] != value:
                value_failures.append(f"{key}: {field} differs from the release")
        if "cell_cycle_stage_counts" not in {
            g["field"] for g in phenotype["provenance_gaps"]
        }:
            value_failures.append(f"{key}: the stage partition carries no gap")
        blocks = [HaplotypeBlock.model_validate(b) for b in genotype["blocks"]]
        n_blocks.append(len(blocks))
        try:
            parents_ok, error = ng.round_trip_error(
                blocks, panel.markers, panel.posterior[number]
            )
        except ValueError as err:
            round_trip_failures.append(f"{key}: {err}")
        else:
            worst_error = max(worst_error, error)
            if not parents_ok:
                round_trip_failures.append(f"{key}: called parent differs")
            elif error > POSTERIOR_STEP / 2 + 1e-9:
                round_trip_failures.append(f"{key}: posterior off by {error:.4f}")

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
    report.add(
        _result(
            Level.L1,
            "segregant_uniqueness",
            n_dup == 0 and len(seen) == len(panel.cells),
            f"{len(seen)} unique strains, {n_dup} duplicates; {len(panel.cells)} "
            "strains carry a significant match",
            n_unique=len(seen),
            n_duplicates=n_dup,
            n_assigned_strains=len(panel.cells),
        )
    )
    for result in shared.results():
        report.add(result)
    report.add(
        _result(
            Level.L2,
            "value_fidelity",
            not value_failures,
            "every record's counts, totals and partitions equal its assigned cells"
            if not value_failures
            else f"{len(value_failures)} fields differ from the release",
            failures=value_failures[:20],
        )
    )
    distribution = (
        {
            "min": int(min(n_blocks)),
            "median": float(np.median(n_blocks)),
            "mean": round(float(np.mean(n_blocks)), 1),
            "max": int(max(n_blocks)),
        }
        if n_blocks
        else {}
    )
    report.add(
        _result(
            Level.L2,
            "mosaic_round_trip",
            not round_trip_failures and bool(n_blocks),
            f"every mosaic re-expands to its strain's posterior (worst error "
            f"{worst_error:.4f}, bound {POSTERIOR_STEP / 2}); blocks per record "
            f"{distribution}"
            if not round_trip_failures
            else f"{len(round_trip_failures)} mosaics fail the round trip",
            failures=round_trip_failures[:20],
            worst_posterior_error=worst_error,
            blocks_per_record=distribution,
        )
    )
    all_rows = sorted(r for rows in panel.cells.values() for r in rows)
    expected_reference = panel.counts(all_rows)
    reference_ok = (
        reference is not None
        and reference_mismatch == 0
        and reference["umi_count"] == expected_reference
        and reference_sum == expected_reference
        and reference["n_cells"] == len(all_rows)
        and reference["total_umis"] == int(panel.release.cell_totals[all_rows].sum())
    )
    report.add(
        _result(
            Level.L2,
            "reference_pseudobulk",
            reference_ok,
            f"the shared reference is the sum of the {len(all_rows)} assigned cells, "
            "and so is the sum of the records",
            n_cells=len(all_rows),
            records_naming_another_reference=reference_mismatch,
        )
    )
    if check_paper_numbers:
        report.add(_l3_release_numbers(panel))
    report.add(_l3_sourced(sourced, library_root, release_sourced, raw_root))
    report.add(_l4_parents(parents, assembly_index))
    report.add(
        _l4_marker_reference(panel.markers, genome, marker_sample, min_marker_reference)
    )
    report.add(_l4_genes(panel.genes, gene_set, sgd_genes, min_gene_containment))
    report.add(_l4_mat(panel, reference))
    report.add(_l4_cis(panel, gene_midpoints, cis_permutations, seed))
    return report


def _l3_release_numbers(panel: PanelRelease) -> LevelResult:
    """Table 1 and the Methods: barcodes, medians, mean cells per strain, batch size."""
    release = panel.release
    genes_per_cell = np.diff(release.counts.indptr)
    n_assigned = sum(len(rows) for rows in panel.cells.values())
    observed = {
        "barcodes": len(release.barcodes),
        "median_umi": float(np.median(release.cell_totals)),
        "median_genes": float(np.median(genes_per_cell)),
        "cells_per_strain": round(n_assigned / len(panel.cells), 2),
        "batch1_strains": len(panel.numbers),
    }
    expected = {
        "barcodes": ng.N_BARCODES,
        "median_umi": 5158.0,
        "median_genes": 1311.0,
        "cells_per_strain": 4.83,
        "batch1_strains": ng.BATCH1_STRAINS,
    }
    return _result(
        Level.L3,
        "release_numbers",
        observed == expected,
        f"barcodes, median UMI and genes per cell, mean cells per assigned strain and "
        f"the batch-1 size {observed} (paper {expected})",
        observed=observed,
        expected=expected,
    )


def _l3_sourced(
    sourced: list[SourcedValue],
    library_root: str | Path,
    release_sourced: list[SourcedValue],
    raw_root: str | Path,
) -> LevelResult:
    audits = [audit_sourced_value(sv, library_root) for sv in sourced]
    audits += [audit_sourced_value(sv, raw_root) for sv in release_sourced]
    failed = [a.details for a in audits if not a.passed]
    return _result(
        Level.L3,
        "sourced_values",
        bool(audits) and not failed,
        f"{len(audits) - len(failed)}/{len(audits)} quotes present in the pinned bytes "
        f"({len(sourced)} paper, {len(release_sourced)} release)",
        failures=failed[:10],
    )


def _l4_parents(
    parents: list[dict[str, Any]], assembly_index: dict[str, str]
) -> LevelResult:
    """BY on the S288C reference file, RM11-1a on its assembly-index member."""
    names = [p["name"] for p in parents]
    problems: list[str] = []
    for parent in parents:
        ref = parent["assembly_ref"]
        peter = parent["peter_strain_id"]
        if peter is None:
            if ref["member"] is not None or not ref["path"].endswith(".fsa"):
                problems.append(f"{parent['name']}: not the S288C reference sequence")
        elif assembly_index.get(peter) != ref["member"]:
            problems.append(f"{parent['name']}: {peter} is not member {ref['member']}")
    passed = names == ["BY4741", "RM11-1a"] and not problems
    return _result(
        Level.L4,
        "parents_pinned",
        passed,
        f"parents {names}; peter ids {[p['peter_strain_id'] for p in parents]}; "
        f"{len(problems)} pin problems",
        problems=problems,
    )


def _l4_marker_reference(
    markers: list[Any], genome: Any, marker_sample: int, min_ratio: float
) -> LevelResult:
    from torchcell.datasets.scerevisiae.bloom2019 import CHROM_TO_GENOME_INDEX

    rng = np.random.default_rng(0)
    checked = matched = 0
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
        f"{ratio:.4f} of {checked} sampled SNPs carry the S288C base as their BY "
        f"allele (>= {min_ratio})",
        n_checked=checked,
        n_match=matched,
        ratio=ratio,
    )


def _l4_genes(
    genes: list[str], gene_set: set[str], sgd_genes: set[str], min_containment: float
) -> LevelResult:
    outside = sorted(set(genes) - sgd_genes)
    containment = 1 - len(outside) / len(genes) if genes else 0.0
    gene_ok = bool(gene_set) and gene_set <= sgd_genes
    return _result(
        Level.L4,
        "gene_containment_sgd",
        gene_ok and containment >= min_containment,
        f"{len(genes) - len(outside)}/{len(genes)} released genes are S288C reference "
        f"genes ({containment:.4f}, >= {min_containment}); {len(gene_set)} spanned "
        f"genes, all reference: {gene_ok}",
        n_genes=len(genes),
        outside_reference=outside[:50],
        n_gene_set=len(gene_set),
    )


def _nearest_marker(markers: list[Any], chromosome: str, position: int) -> int | None:
    on_chrom = [i for i, m in enumerate(markers) if m.chromosome == chromosome]
    if not on_chrom:
        return None
    return min(on_chrom, key=lambda i: abs(markers[i].position - position))


def _l4_mat(panel: PanelRelease, reference: dict[str, Any] | None) -> LevelResult:
    """A MATa panel by expression must be fixed at MAT by genotype."""
    counts = reference["umi_count"] if reference is not None else {}
    a_umis = sum(counts.get(g, 0) for g in A_SPECIFIC)
    alpha_umis = sum(counts.get(g, 0) for g in ALPHA_SPECIFIC)
    ratio = a_umis / alpha_umis if alpha_umis else float("inf")
    mat = _nearest_marker(panel.markers, MAT_CHROMOSOME, MAT_POSITION)
    values = (
        [float(p[mat]) for p in panel.posterior.values()] if mat is not None else []
    )
    mean = float(np.mean(values)) if values else float("nan")
    fixed = bool(values) and min(mean, 1 - mean) <= MAT_FIXATION_TOLERANCE
    passed = ratio >= MIN_A_TO_ALPHA and fixed
    return _result(
        Level.L4,
        "mat_fixation",
        passed,
        f"reference a-specific/alpha-specific UMIs {a_umis}/{alpha_umis}; mean P(RM) "
        f"at the SNP nearest MAT {mean:.4f} (fixed within {MAT_FIXATION_TOLERANCE}); "
        f"the MATa parent is {'RM11-1a' if mean > 0.5 else 'BY4741'}",
        a_specific_umis=a_umis,
        alpha_specific_umis=alpha_umis,
        mean_rm_at_mat=mean,
        mat_marker=None if mat is None else panel.markers[mat].model_dump(),
    )


def cis_statistics(
    expression: np.ndarray[Any, Any],
    genotype: np.ndarray[Any, Any],
    permutations: int,
    seed: int,
) -> dict[str, float]:
    """Observed vs permuted |Pearson r| of each gene with its own-locus genotype.

    ``expression`` and ``genotype`` are strains x tested genes (log CPM and P(RM) at the
    SNP nearest each gene). The null permutes which strain's genotype goes with which
    strain's expression; its ``CIS_NULL_QUANTILE`` quantile of |r| is the threshold.
    """

    def correlations(x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> Any:
        xc = x - x.mean(axis=0)
        yc = y - y.mean(axis=0)
        denom = np.sqrt((xc**2).sum(axis=0) * (yc**2).sum(axis=0))
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (xc * yc).sum(axis=0) / denom
        return np.abs(np.nan_to_num(r, nan=0.0))

    rng = np.random.default_rng(seed)
    observed = correlations(expression, genotype)
    null = np.concatenate(
        [
            correlations(expression, genotype[rng.permutation(len(genotype))])
            for _ in range(permutations)
        ]
    )
    threshold = float(np.quantile(null, CIS_NULL_QUANTILE))
    return {
        "threshold": threshold,
        "observed_rate": float(np.mean(observed > threshold)),
        "null_rate": float(np.mean(null > threshold)),
        "median_observed_r": float(np.median(observed)),
    }


def _l4_cis(
    panel: PanelRelease,
    gene_midpoints: dict[str, tuple[str, int]],
    permutations: int,
    seed: int,
) -> LevelResult:
    """Expression of the assigned cells tracks the stored genotype at each gene's locus."""
    strains = sorted(panel.index_of)
    summed = np.vstack(
        [
            np.asarray(
                panel.release.counts[panel.cells[panel.index_of[n]]].sum(axis=0)
            ).ravel()
            for n in strains
        ]
    ).astype(np.float64)
    totals = summed.sum(axis=1, keepdims=True)
    log_cpm = np.log2(summed / totals * 1e6 + 1.0)
    detected = (summed > 0).mean(axis=0) >= CIS_MIN_DETECTION
    columns: list[int] = []
    loci: list[int] = []
    for j, gene in enumerate(panel.release.genes):
        if not detected[j] or gene not in gene_midpoints:
            continue
        chrom, mid = gene_midpoints[gene]
        marker = _nearest_marker(panel.markers, chrom, mid)
        if marker is not None:
            columns.append(j)
            loci.append(marker)
    if not columns:
        return _result(Level.L4, "cis_concordance", False, "no gene could be tested")
    genotype = np.vstack([panel.posterior[n][loci] for n in strains])
    stats = cis_statistics(log_cpm[:, columns], genotype, permutations, seed)
    required = MIN_CIS_ENRICHMENT * (1 - CIS_NULL_QUANTILE)
    return _result(
        Level.L4,
        "cis_concordance",
        stats["observed_rate"] >= required,
        f"{stats['observed_rate']:.4f} of {len(columns)} tested genes exceed the "
        f"permuted-map |r| threshold {stats['threshold']:.4f} (null rate "
        f"{stats['null_rate']:.4f}; required >= {required:.2f}) over {len(strains)} "
        "strains",
        n_tested_genes=len(columns),
        n_strains=len(strains),
        permutations=permutations,
        **stats,
    )


def gene_midpoints_from_genome(
    genome: Any, genes: Iterable[str]
) -> dict[str, tuple[str, int]]:
    """(chromosome, midpoint) of every gene the genome annotates; others are skipped."""
    out: dict[str, tuple[str, int]] = {}
    db = genome.db
    for gene in genes:
        if gene in genome.gene_set:
            feature = db[gene]
            out[gene] = (
                str(feature.chrom),
                (int(feature.start) + int(feature.end)) // 2,
            )
    return out
