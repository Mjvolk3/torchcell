# torchcell/verification/segregant_perturbation
# [[torchcell.verification.segregant_perturbation]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/segregant_perturbation
# Test file: tests/torchcell/verification/test_segregant_perturbation.py
"""L0-L4 verifier for perturbation-on-segregant datasets (Hale 2024).

A ``SegregantPerturbationExperiment`` carries a ``PerturbedSegregantGenotype``: an
inferred haplotype mosaic (the background) plus designed gene-keyed perturbations. The
segregant growth verifier reads ``segregant_id``/``blocks`` at the genotype's top level
and the environment-response verifier reads only ``perturbations``, so neither sees both
halves. This module is one streaming pass over the records plus a pass over the released
files that re-derives what the records claim:

- L0 ``structural`` -- every experiment validates as ``SegregantPerturbationExperiment``,
  the class its ``experiment_type`` names (each distinct interned background and
  environment is validated once and reused).
- L1 ``count``; ``lineage_uniqueness`` (one record per (assay, double barcode));
  ``screen_partition`` (per-assay record counts equal the release's gene-targeting rows);
  ``ledger_balance`` (served + ledgered rows = released rows, per assay); plus the shared
  rules (gap census, canonical gene names, uncertainty sanity, compound identity, media
  membership).
- L2 ``value_fidelity`` (every stored fitness, error, background, guide and ORF equals
  its released cell); ``normalization_reproduced`` (each assay's released ``fitness`` is
  its ``raw_fitness`` minus ONE constant, and that constant is recomputed from the 77
  retained control gRNAs and the CON mean); ``mosaic_round_trip`` (every background
  re-expands to its released column at that column's called markers, and the parent role
  is held only by the BY column).
- L3 ``reference_zero``; ``environment_documented`` (ATC1/ATC2 carry anhydrotetracycline
  at the sourced dose, CON nothing; 21 C, SC-Ura, 10 generations); ``measurement_typed``;
  ``sourced_values`` (every quote still in its sha256-pinned source).
- L4 ``matrix_subset_agreement`` (Supplemental Data 2 is a subset of 12 and agrees at
  every called cell); ``variant_reference`` (every SNV and deletion of the 3S variant
  table carries the S288C R64-4-1 base as its reference); ``guide_targets`` (each served
  guide's released ORF equals Supplemental Data 3's); ``parents_pinned`` (both parent
  refs resolve to bytes with the pinned sha256); gene containment in SGD (shared rule).
"""

from __future__ import annotations

import math
import os.path as osp
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from torchcell.datasets.scerevisiae import hale2024 as h
from torchcell.verification.common import SharedRecordRules
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


class ReleasedAssay:
    """One assay's released gene-targeting rows, indexed by double barcode."""

    def __init__(self, spec: h.AssaySpec, frame: pd.DataFrame) -> None:
        """Index ``frame`` (the full released table) by ``<geno_bc>_<guide_bc>``."""
        self.spec = spec
        self.n_released = len(frame)
        gene = frame[frame["gene_orf"].notna()]
        self.n_gene_rows = len(gene)
        ids = (
            gene["geno_bc"].astype(str) + "_" + gene["guide_bc"].astype(str)
        ).tolist()
        self.row_of = {rid: i for i, rid in enumerate(ids)}
        self.fitness = gene["fitness"].to_numpy(dtype=float)
        self.error = gene["error"].to_numpy(dtype=float)
        self.genotype = gene["genotype"].astype(str).tolist()
        self.guide = gene["guide_seq"].astype(str).tolist()
        self.orf = gene["gene_orf"].astype(str).tolist()
        self.seen = np.zeros(len(gene), dtype=bool)


def normalization_offsets(
    raw: dict[str, pd.DataFrame],
) -> tuple[dict[str, float], dict[str, float], int]:
    """(released offset per assay, recomputed offset per assay, retained control count).

    Recomputed as ``analysis_code.R`` does: the per-guide mean ``raw_fitness`` in each
    assay; control gRNAs (``gene_orf`` NA) whose mean is negative in ANY assay are
    dropped; each assay is shifted by the median of the remaining control means; then
    every assay by the mean of the shifted CON values. The released offset is the
    constant ``raw_fitness - fitness`` (its spread is returned by the caller's check).
    """
    means = {k: d.groupby("guide_seq")["raw_fitness"].mean() for k, d in raw.items()}
    first = next(iter(raw.values()))
    controls = set(first.loc[first["gene_orf"].isna(), "guide_seq"])
    low = {g for g in controls for m in means.values() if m[g] < 0}
    kept = sorted(controls - low)
    medians = {k: float(np.median(m[kept])) for k, m in means.items()}
    con_mean = float((raw["CON"]["raw_fitness"] - medians["CON"]).mean())
    recomputed = {k: medians[k] + con_mean for k in raw}
    released = {
        k: float((d["raw_fitness"] - d["fitness"]).mean()) for k, d in raw.items()
    }
    return released, recomputed, len(kept)


def read_released_tables(
    raw_dir: str | Path,
) -> tuple[dict[str, ReleasedAssay], dict[str, pd.DataFrame]]:
    """Every assay's lineage table: the indexed gene rows, and the raw normalization view."""
    import zipfile

    assays: dict[str, ReleasedAssay] = {}
    raw: dict[str, pd.DataFrame] = {}
    for spec in h.ASSAYS:
        path = osp.join(raw_dir, spec.released_file)
        with zipfile.ZipFile(path) as archive, archive.open(spec.zip_member) as handle:
            frame = pd.read_csv(
                handle,
                usecols=[*h.LINEAGE_COLUMNS, "raw_fitness"],
                dtype={"gene_orf": str, "genotype": str, "guide_seq": str},
                float_precision="round_trip",
            )
        assays[spec.screen_id] = ReleasedAssay(spec, frame)
        raw[spec.screen_id] = frame[["guide_seq", "gene_orf", "raw_fitness", "fitness"]]
    return assays, raw


def verify_segregant_perturbation_streaming(
    records: Iterable[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    raw_dir: str | Path,
    preprocess_dir: str | Path,
    s288c_fasta: str | Path,
    variant_table: str | Path,
    parent_files: dict[str, str | Path],
    library_root: str | Path | None,
    raw_root: str | Path | None,
    resolve_gene_name: Any,
    sgd_genes: set[str],
) -> VerificationReport:
    """Single-pass L0-L4 gate (see the module docstring).

    ``raw_dir`` holds the seven released files (the store's ``raw/``), ``s288c_fasta``
    the SGD R64-4-1 sequence and ``variant_table`` the 3S table (both resolved from the
    genomes tier by the caller), ``parent_files`` maps each parent's pinned tier path to
    its file on disk. ``resolve_gene_name`` is the genome's resolver (the shared
    canonical-name rule). The stored target ORF is compared with the released
    ``gene_orf`` cell byte for byte: the loader stores it as released and resolves
    nothing. ``library_root`` / ``raw_root`` are the roots the paper and code quotes
    audit against; ``None`` skips the L3 provenance audit (a synthetic release has no
    mirror to audit).
    """
    from torchcell.datamodels.schema import (
        EXPERIMENT_TYPE_MAP,
        Environment,
        SegregantGenotype,
        SegregantPerturbationExperiment,
    )

    # L0 validates each record as the class its experiment_type names (how the dataset
    # rebuilds it); the bare ExperimentType union tries every experiment class in turn
    # and measured 10.3 ms per record here, 25 h over the store. The constant
    # sub-objects (each distinct background and environment) are validated once and
    # their instances reused, as the dataset's own read path does.
    experiment_class = EXPERIMENT_TYPE_MAP["segregant_perturbation"]
    if experiment_class is not SegregantPerturbationExperiment:
        raise TypeError("segregant_perturbation does not name its experiment class")
    validated_environments: dict[int, Environment] = {}
    assays, raw = read_released_tables(raw_dir)
    matrix = h.read_genotype_matrix(osp.join(raw_dir, h.SD12))

    # ---- record pass -----------------------------------------------------------
    n_records = 0
    l0_failures: list[dict[str, Any]] = []
    validated_backgrounds: dict[int, SegregantGenotype] = {}
    per_screen: Counter[str] = Counter()
    n_duplicate = 0
    n_unknown_lineage = 0
    mismatches: list[dict[str, Any]] = []
    n_mismatch = 0
    nonfinite = 0
    backgrounds: dict[str, dict[str, Any]] = {}
    roles: dict[str, set[str]] = {}
    env_by_screen: dict[str, set[str]] = {}
    references: dict[str, float] = {}
    typed_failures: Counter[str] = Counter()
    shared = SharedRecordRules(resolve_gene_name=resolve_gene_name, sgd_genes=sgd_genes)

    for rec in records:
        exp = rec["experiment"]
        n_records += 1
        shared.add(rec)
        genotype = exp["genotype"]
        background = genotype["background"]
        key = id(background)
        try:
            if exp["experiment_type"] != "segregant_perturbation":
                raise TypeError(f"experiment_type {exp['experiment_type']!r}")
            if key not in validated_backgrounds:
                validated_backgrounds[key] = SegregantGenotype.model_validate(
                    background
                )
            env_key = id(exp["environment"])
            if env_key not in validated_environments:
                validated_environments[env_key] = Environment.model_validate(
                    exp["environment"]
                )
            SegregantPerturbationExperiment.model_validate(
                {
                    **exp,
                    "genotype": {**genotype, "background": validated_backgrounds[key]},
                    "environment": validated_environments[env_key],
                }
            )
        except (ValueError, TypeError) as err:
            l0_failures.append({"record": n_records, "error": str(err)[:300]})
        phenotype = exp["phenotype"]
        screen = str(phenotype["screen_id"])
        rid = str(phenotype["replicate_id"])
        per_screen[screen] += 1
        seg_id = str(background["segregant_id"])
        backgrounds.setdefault(seg_id, background)
        roles.setdefault(seg_id, set()).add(str(genotype["background_role"]))
        perts = exp["environment"]["perturbations"]
        env_by_screen.setdefault(screen, set()).add(
            ";".join(
                f"{p['compound']['name']}@{p['concentration']['value']}"
                f"{p['concentration']['unit']}"
                for p in perts
            )
        )
        ref_pheno = rec["reference"]["phenotype_reference"]
        references[str(ref_pheno["screen_id"])] = float(
            ref_pheno["environment_response"]
        )
        if (
            phenotype["measurement_type"] != "relative_fitness"
            or phenotype["assay_type"] != "pooled_competitive_growth_barcode"
            or phenotype["environment_response_uncertainty_type"] != "standard_error"
            or phenotype["n_samples"] != 1
        ):
            typed_failures[screen] += 1
        released = assays.get(screen)
        row = None if released is None else released.row_of.get(rid)
        if released is None or row is None:
            n_unknown_lineage += 1
            continue
        if released.seen[row]:
            n_duplicate += 1
            continue
        released.seen[row] = True
        value = float(phenotype["environment_response"])
        error = float(phenotype["environment_response_uncertainty"])
        if not (math.isfinite(value) and math.isfinite(error)):
            nonfinite += 1
        pert = genotype["perturbations"]
        stored = (
            value,
            error,
            seg_id,
            pert[0]["crispr"]["guide_sequence"] if len(pert) == 1 else None,
            pert[0]["systematic_gene_name"] if len(pert) == 1 else None,
        )
        expected = (
            float(released.fitness[row]),
            float(released.error[row]),
            released.genotype[row],
            released.guide[row],
            released.orf[row],
        )
        if stored != expected:
            n_mismatch += 1
            if len(mismatches) < 20:
                mismatches.append(
                    {
                        "screen": screen,
                        "lineage": rid,
                        "stored": stored,
                        "released": expected,
                    }
                )

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(
        _result(
            Level.L0,
            "structural",
            not l0_failures,
            f"{n_records - len(l0_failures)}/{n_records} records validate as "
            "SegregantPerturbationExperiment",
            failures=l0_failures[:20],
            n_distinct_backgrounds=len(validated_backgrounds),
        )
    )
    report.add(
        _result(
            Level.L1,
            "count",
            n_records == expected_count,
            f"{n_records} records (expected {expected_count})",
            n_records=n_records,
        )
    )
    unseen = {k: int((~a.seen).sum()) for k, a in assays.items()}
    report.add(
        _result(
            Level.L1,
            "lineage_uniqueness",
            n_duplicate == 0 and n_unknown_lineage == 0 and not any(unseen.values()),
            f"{n_duplicate} duplicate and {n_unknown_lineage} unreleased lineages; "
            f"released gene rows never served: {unseen}",
            n_duplicate=n_duplicate,
            n_unknown=n_unknown_lineage,
            unseen=unseen,
        )
    )
    expected_screens = {k: a.n_gene_rows for k, a in assays.items()}
    report.add(
        _result(
            Level.L1,
            "screen_partition",
            dict(per_screen) == expected_screens,
            f"records per assay {dict(per_screen)} vs released gene rows {expected_screens}",
            per_screen=dict(per_screen),
        )
    )
    report.add(_l1_ledger_balance(preprocess_dir, assays))
    for result in shared.results():
        report.add(result)
    report.add(
        _result(
            Level.L2,
            "value_fidelity",
            n_mismatch == 0 and nonfinite == 0,
            f"{n_records - n_mismatch} records equal their released cell "
            f"(fitness, error, background, guide, ORF); {nonfinite} non-finite",
            n_mismatch=n_mismatch,
            examples=mismatches,
        )
    )
    report.add(_l2_normalization(raw))
    report.add(_l2_mosaics(backgrounds, roles, matrix))
    report.add(
        _result(
            Level.L3,
            "reference_zero",
            set(references) == set(expected_screens)
            and all(v == 0.0 for v in references.values()),
            f"reference response per assay {references}",
            references=references,
        )
    )
    dose = f"anhydrotetracycline@{h.ATC_DOSE_UG_PER_ML.value}ug/mL"
    expected_env = {a.screen_id: {dose if a.induced else ""} for a in h.ASSAYS}
    env_seen = {k: sorted(v) for k, v in env_by_screen.items()}
    report.add(
        _result(
            Level.L3,
            "environment_documented",
            {k: set(v) for k, v in env_seen.items()} == expected_env,
            f"environment perturbations per assay {env_seen}",
            per_screen=env_seen,
        )
    )
    report.add(
        _result(
            Level.L3,
            "measurement_typed",
            not typed_failures,
            "every phenotype is relative_fitness / pooled barcode / standard_error / "
            f"n=1; off-type per assay: {dict(typed_failures)}",
            failures=dict(typed_failures),
        )
    )
    if library_root is not None and raw_root is not None:
        report.add(_l3_sourced_values(library_root, raw_root))
    report.add(_l4_matrix_subset(raw_dir, matrix))
    report.add(_l4_variant_reference(variant_table, s288c_fasta))
    report.add(_l4_guide_targets(raw_dir, assays))
    report.add(_l4_parents_pinned(backgrounds, parent_files))
    return report


def _l1_ledger_balance(
    preprocess_dir: str | Path, assays: dict[str, ReleasedAssay]
) -> LevelResult:
    """Served + every ledgered rule's per-assay count = released rows, per assay."""
    import json

    path = osp.join(preprocess_dir, "dropped_records.json")
    with open(path) as fh:
        rules = json.load(fh)
    ledgered: Counter[str] = Counter()
    for rule in rules:
        for screen, n in rule["per_assay"].items():
            ledgered[screen] += int(n)
    balance = {k: a.n_gene_rows + ledgered[k] - a.n_released for k, a in assays.items()}
    controls = [r for r in rules if r["rule"] == "control_guide_targets_no_gene"]
    n_control_guides = len(controls[0]["items"]) if controls else 0
    return _result(
        Level.L1,
        "ledger_balance",
        all(v == 0 for v in balance.values())
        and n_control_guides == h.CONTROL_GUIDES.value,
        f"served + ledgered - released per assay {balance}; "
        f"{n_control_guides} control gRNAs ledgered (paper: {h.CONTROL_GUIDES.value})",
        balance=balance,
        ledger=rules,
    )


def _l2_normalization(raw: dict[str, pd.DataFrame]) -> LevelResult:
    """The released fitness is raw_fitness minus one recomputable constant per assay."""
    released, recomputed, n_kept = normalization_offsets(raw)
    spread = {
        k: float((d["raw_fitness"] - d["fitness"] - released[k]).abs().max())
        for k, d in raw.items()
    }
    diff = {k: abs(released[k] - recomputed[k]) for k in raw}
    passed = (
        all(v < 1e-12 for v in spread.values())
        and all(v < 1e-12 for v in diff.values())
        and n_kept == h.CONTROL_GUIDES.value - h.CONTROL_GUIDES_REMOVED.value
    )
    return _result(
        Level.L2,
        "normalization_reproduced",
        passed,
        f"offsets {released}; recomputed from {n_kept} control gRNAs within "
        f"{max(diff.values()):.2g}; within-assay spread {max(spread.values()):.2g}",
        released=released,
        recomputed=recomputed,
        n_controls_kept=n_kept,
        spread=spread,
    )


def _l2_mosaics(
    backgrounds: dict[str, dict[str, Any]],
    roles: dict[str, set[str]],
    matrix: pd.DataFrame,
) -> LevelResult:
    """Every served background re-expands to its released column; one role each."""
    from torchcell.datamodels.schema import HaplotypeBlock

    failures: list[str] = []
    for seg_id, background in sorted(backgrounds.items()):
        markers = h.called_markers(matrix, seg_id)
        blocks = [HaplotypeBlock.model_validate(b) for b in background["blocks"]]
        try:
            if h.expand_blocks(blocks, markers) != markers.calls:
                failures.append(f"{seg_id}: calls differ")
        except ValueError as err:
            failures.append(f"{seg_id}: {err}")
        expected_role = {h.background_role(seg_id)}
        if roles[seg_id] != expected_role:
            failures.append(
                f"{seg_id}: roles {sorted(roles[seg_id])} != {expected_role}"
            )
    return _result(
        Level.L2,
        "mosaic_round_trip",
        not failures and len(backgrounds) == h.N_BACKGROUNDS,
        f"{len(backgrounds) - len(failures)}/{len(backgrounds)} backgrounds re-expand to "
        f"their released column (expected {h.N_BACKGROUNDS})",
        failures=failures[:20],
        n_blocks={k: len(v["blocks"]) for k, v in sorted(backgrounds.items())},
    )


def _l3_sourced_values(library_root: str | Path, raw_root: str | Path) -> LevelResult:
    """Every quote is still in its sha256-pinned source (paper, SI, Matsui, code)."""
    failures: list[dict[str, Any]] = []
    audited: list[tuple[SourcedValue, str | Path]] = [
        *((sv, library_root) for sv in h.SOURCED_VALUES),
        *((sv, raw_root) for sv in h.CODE_SOURCED_VALUES),
    ]
    for sv, root in audited:
        result = audit_sourced_value(sv, root)
        if not result.passed:
            failures.append({"quote": sv.quote[:80], "message": result.message})
    return _result(
        Level.L3,
        "sourced_values",
        not failures,
        f"{len(audited) - len(failures)}/{len(audited)} quotes present in their "
        "pinned sources",
        failures=failures,
    )


def _l4_matrix_subset(raw_dir: str | Path, sd12: pd.DataFrame) -> LevelResult:
    """Supplemental Data 2 is a marker and column subset of 12 with no disagreement."""
    sd2 = h.read_genotype_matrix(osp.join(raw_dir, h.SD2))
    merged = sd2.merge(sd12, on=["c", "p"], suffixes=("_2", "_12"))
    columns = [c for c in sd2.columns[3:] if c in sd12.columns]
    disagree = 0
    compared = 0
    for col in columns:
        a = merged[f"{col}_2"].to_numpy(dtype=float)
        b = merged[f"{col}_12"].to_numpy(dtype=float)
        both = ~np.isnan(a) & ~np.isnan(b)
        compared += int(both.sum())
        disagree += int((a[both] != b[both]).sum())
    passed = (
        len(merged) == len(sd2)
        and len(columns) == len(sd2.columns) - 3
        and disagree == 0
    )
    return _result(
        Level.L4,
        "matrix_subset_agreement",
        passed,
        f"{len(merged)}/{len(sd2)} SD2 markers and {len(columns)}/{len(sd2.columns) - 3} "
        f"columns are in SD12; {disagree} of {compared} called cells disagree",
        n_markers_sd2=len(sd2),
        n_markers_sd12=len(sd12),
        compared=compared,
        disagree=disagree,
    )


def read_fasta_by_chromosome(path: str | Path) -> dict[int, str]:
    """SGD R64 FASTA keyed by Supplemental Data chromosome code (17 = mitochondrion)."""
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII",
             "XIII", "XIV", "XV", "XVI"]  # fmt: skip
    seqs: dict[int, list[str]] = {}
    current: int | None = None
    with open(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if "[location=mitochondrion]" in line:
                    current = 17
                else:
                    current = (
                        roman.index(line.split("[chromosome=")[1].split("]")[0]) + 1
                    )
                seqs[current] = []
            elif current is not None:
                seqs[current].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}


def _l4_variant_reference(
    variant_table: str | Path, s288c_fasta: str | Path
) -> LevelResult:
    """Every SNV and deletion of the 3S table carries the R64-4-1 base as reference."""
    table = pd.read_csv(
        variant_table, sep="\t", header=None, names=["chr", "pos", "ref", "alt"]
    )
    seqs = read_fasta_by_chromosome(s288c_fasta)
    codes = table["chr"].str.replace("_S288C", "", regex=False).astype(int)
    anchored = table["ref"] != "-"
    matches = sum(
        seqs[c][p - 1].upper() == r.upper()
        for c, p, r in zip(
            codes[anchored], table["pos"][anchored], table["ref"][anchored], strict=True
        )
    )
    n_anchored = int(anchored.sum())
    kinds = {
        "snv": int(((table["ref"] != "-") & (table["alt"] != "-")).sum()),
        "deletion": int((table["alt"] == "-").sum()),
        "insertion": int((table["ref"] == "-").sum()),
    }
    return _result(
        Level.L4,
        "variant_reference",
        matches == n_anchored and n_anchored > 0,
        f"{matches}/{n_anchored} SNV and deletion rows carry the S288C R64-4-1 base "
        f"(rows by kind {kinds})",
        kinds=kinds,
        n_match=matches,
    )


def _l4_guide_targets(
    raw_dir: str | Path, assays: dict[str, ReleasedAssay]
) -> LevelResult:
    """Each served guide's released ORF equals its Supplemental Data 3 ORF."""
    sd3 = pd.read_csv(osp.join(raw_dir, h.SD3), dtype={"ORF": str})
    orf_of = sd3.drop_duplicates("guide").set_index("guide")["ORF"]
    served: dict[str, str] = {}
    for released in assays.values():
        served.update(zip(released.guide, released.orf, strict=True))
    missing = sorted(g for g in served if g not in orf_of.index)
    differ = sorted(
        g for g in served if g in orf_of.index and str(orf_of[g]) != served[g]
    )
    return _result(
        Level.L4,
        "guide_targets",
        not missing and not differ,
        f"{len(served) - len(missing) - len(differ)}/{len(served)} served guides carry "
        "their Supplemental Data 3 ORF",
        missing=missing[:20],
        differ=differ[:20],
    )


def _l4_parents_pinned(
    backgrounds: dict[str, dict[str, Any]], parent_files: dict[str, str | Path]
) -> LevelResult:
    """Both parents' refs name a tier file whose bytes hash to the pinned sha256."""
    failures: list[str] = []
    refs: dict[str, dict[str, Any]] = {}
    for background in backgrounds.values():
        for key in ("parent_1", "parent_2"):
            parent = background[key]
            refs[parent["name"]] = parent["assembly_ref"]
    for name, ref in sorted(refs.items()):
        path = parent_files.get(ref["path"])
        if path is None:
            failures.append(f"{name}: no file for {ref['key']}/{ref['path']}")
        elif sha256_file(path) != ref["sha256"]:
            failures.append(f"{name}: {ref['path']} does not hash to the pinned sha256")
    return _result(
        Level.L4,
        "parents_pinned",
        not failures and set(refs) == {h.BY_NAME, h.THREE_S_NAME},
        f"parents {sorted(refs)} pinned to {[r['key'] for _, r in sorted(refs.items())]}",
        failures=failures,
    )
