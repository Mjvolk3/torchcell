# experiments/026-metabolism-flux/scripts/kcat_sigma_table.py
# [[experiments.026-metabolism-flux.kcat-uncertainty-2026-09-14]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/kcat_sigma_table.py

r"""Build the per-unit :math:`\log_{10}k_{cat}` width, and prove it moves nothing that shipped.

WHAT THIS IS FOR. The flux layer propagates a correlated uncertainty for reaction energies
(``||S_j^T Sigma^{1/2}||_2``, consumed as :math:`\mu+\Sigma^{1/2}m` with an
:math:`m\sim N(0,I)` prior) and propagates nothing at all for turnover numbers: ``kcat_per_h``
is one point value per catalytic unit. This script builds the missing half, a per-unit
:math:`\sigma_u` in decades, and writes it beside the shipped values rather than over them.

THE LADDER, AND WHY EVERY RUNG IS MEASURED HERE RATHER THAN ASSUMED. The four tiers of the
resolver are a PROVENANCE ordering. They are not an accuracy ordering, and the difference is
the whole design:

* a measured :math:`k_{cat}` reproduces across independent papers to 0.497 decades,
* the homolog tier's own cross-organism spread is 0.733 decades, the WIDEST of the three,
* the admitted ensemble's residual against a measurement is 0.469 decades.

So a cascade that gave the measured tier a near-zero width would be reporting where a number
came from, not how well it is known. Each statistic is recomputed from the Open Enzyme
Database mirror and from ``results/predictor_vs_measured.csv`` on every run and asserted
against the rung :data:`~torchcell.metabolism.parameters.KCAT_SIGMA_LADDER` ships, so the
constants cannot drift away from the data that set them.

THE NEUTRALITY PROOF. Two tables are resolved in one run from the same mirror: one through
``ParameterTable.build`` without the new argument, which is byte-for-byte the call the
shipped code makes, and one with ``log10_sigma``. ``values``, ``provenance``, ``known_mask``
and ``experimental_mask`` are compared element-wise between them, the tier counts are
asserted against the committed 97 / 782 / 2,831 / 18, and every predictor parquet's sha256 is
checked against ``results/kinetic_table_versions.json``. Any inequality is a bug, not a
result.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/kcat_sigma_table.py

Writes, all NEW files beside the shipped ones:

* ``results/kcat_unit_sigma_v2026.09.14.parquet`` -- one row per catalytic unit
* ``results/kcat_unit_shipped_v2026.09.14.parquet`` -- the same table with no sigma column,
  the object the element-wise comparison is made against
* ``results/kcat_sigma_ladder.json`` -- the ladder, the recomputed statistics, the asserts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from torchcell.metabolism.constraints import build_gem_tensors
from torchcell.metabolism.parameters import (
    KCAT_SIGMA_LADDER,
    KINETICS_MIRROR,
    PLAUSIBILITY_RESULT,
    VERIFIED_PAIRS_FEATURE,
    PredictorSelection,
    SubstrateMatch,
    load_predicted_unit_table,
    resolve_kcat_table,
)
from torchcell.metabolism.yeast_GEM import YeastGEM

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")

OED_ALL = osp.join(DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/all")
OED_SCEREVISIAE = osp.join(
    DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/scerevisiae"
)

#: The committed tier split of the verified table, from
#: [[experiments.026-metabolism-flux.kinetic-coverage-closure]]. A run that resolves
#: anything else read a different mirror and its sigma table is not comparable.
SHIPPED_TIER_COUNTS = {"measured": 97, "homolog": 782, "predicted": 2831, "default": 18}

#: Assay temperature window for the measured rung, in Celsius. The resolver picks the row
#: nearest 30 C because the phenotype data is at 26-30 C, so the replicate spread has to be
#: measured on rows the resolver would actually accept.
TEMPERATURE_WINDOW_C = (25.0, 35.0)

#: How far a recomputed statistic may sit from the rung it ships, in decades. The rungs are
#: the statistics rounded to two decimals, so anything past this is a real change in the
#: mirror rather than a rounding difference.
SIGMA_TOLERANCE = 0.006

VERSION = "2026.09.14"


class LadderCheck(BaseModel):
    """One rung's number, recomputed, and whether it still matches what ships."""

    model_config = ConfigDict(extra="forbid")

    tier: str
    shipped_log10_sigma: float
    recomputed: float
    n_groups: int | None = None
    n_rows: int | None = None
    statistic: str
    source: str
    matches: bool


class SigmaTableReport(BaseModel):
    """Everything one run of this script measured and asserted."""

    model_config = ConfigDict(extra="forbid")

    version: str
    built_at: str
    ladder_version: str
    checks: list[LadderCheck]
    tier_counts: dict[str, int]
    shipped_tier_counts: dict[str, int]
    n_units: int
    predictor_sha256: dict[str, str] = Field(default_factory=dict)
    predictor_sha256_matches_shipped: dict[str, bool] = Field(default_factory=dict)
    elementwise_equal: dict[str, bool] = Field(default_factory=dict)
    sigma_by_tier: dict[str, dict[str, float]] = Field(default_factory=dict)
    per_unit_sd_used_fraction: float = Field(
        description="Share of predicted units whose own across-predictor sd beat the floor."
    )
    out_parquet: str
    out_parquet_shipped: str
    sha256: dict[str, str] = Field(default_factory=dict)


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _oed_frame(mirror_dir: str) -> pd.DataFrame:
    """Wild-type rows of one OED slice with a usable turnover number, in log10."""
    with open(osp.join(mirror_dir, "oed_records.json")) as handle:
        frame = pd.DataFrame(json.load(handle))
    frame["kcat_value"] = pd.to_numeric(frame["kcat_value"], errors="coerce")
    frame = frame[np.isfinite(frame["kcat_value"]) & (frame["kcat_value"] > 0)]
    frame = frame[frame["enzymetype"] == "wildtype"].copy()
    frame["lk"] = np.log10(frame["kcat_value"])
    frame["temp"] = pd.to_numeric(frame["temperature"], errors="coerce")
    return frame


def _pooled_within_group_sd(grouped) -> tuple[float, int, int]:
    r"""Pooled within-group sample sd, and the groups and rows it was pooled over.

    :math:`s_p=\sqrt{\sum(n_g-1)s_g^2/\sum(n_g-1)}`. Pooled rather than a median of
    per-group sds because a two-row group's sd is a one-degree-of-freedom estimate and
    the median over many of them is dominated by that noise.
    """
    counts = grouped.count()
    keep = counts[counts >= 2]
    sds = grouped.std()[keep.index]
    numerator = ((keep - 1) * sds**2).sum()
    return (
        float(np.sqrt(numerator / (keep - 1).sum())),
        int(len(keep)),
        int(keep.sum()),
    )


def measured_rung_statistic() -> tuple[float, int, int]:
    """Cross-PAPER replicate spread of a measured k_cat, all organisms, 25-35 C.

    Reduced to one value per ``(accession, substrate, pubmed id)`` FIRST, so a within-paper
    condition series (a pH or temperature scan) cannot pose as replication and shrink the
    width. That reduction is what takes the statistic from 0.324 to 0.497 decades.
    """
    frame = _oed_frame(OED_ALL)
    low, high = TEMPERATURE_WINDOW_C
    warm = frame[np.isfinite(frame["temp"]) & frame["temp"].between(low, high)]
    per_paper = (
        warm.dropna(subset=["uniprot", "substrate", "kcat_pubmedid"])
        .groupby(["uniprot", "substrate", "kcat_pubmedid"], as_index=False)["lk"]
        .median()
    )
    return _pooled_within_group_sd(per_paper.groupby(["uniprot", "substrate"])["lk"])


def homolog_rung_statistic() -> tuple[float, int, int]:
    """Cross-ORGANISM spread at a shared EC number and substrate, every temperature.

    This is exactly what the homolog tier accepts, so it is the width of the evidence that
    tier carries: the same chemistry measured in a protein that is not this organism's.
    """
    frame = _oed_frame(OED_ALL)
    return _pooled_within_group_sd(
        frame.dropna(subset=["ec", "substrate"]).groupby(["ec", "substrate"])["lk"]
    )


def predicted_rung_statistic() -> tuple[float, int, float]:
    """Sd of the admitted ensemble's signed residual against the 58 measured yeast pairs.

    Returns ``(sd, n, signed_median)``. The signed median is reported alongside because a
    biased-low ensemble and a wide one are different defects and the width alone hides the
    first.
    """
    frame = pd.read_csv(osp.join(RESULTS, "predictor_vs_measured.csv"))
    admitted = PredictorSelection.from_study(
        "k_cat", osp.join(EXPERIMENT_ROOT, PLAUSIBILITY_RESULT)
    ).selected
    kcat = frame[
        (frame["parameter"] == "k_cat") & frame["predictor"].isin(admitted)
    ].copy()
    kcat["log_predicted"] = np.log10(kcat["predicted"])
    ensemble = kcat.groupby(["uniprot", "substrate"]).agg(
        log_predicted=("log_predicted", "median"), measured=("measured", "first")
    )
    residual = ensemble["log_predicted"] - np.log10(ensemble["measured"])
    return float(residual.std(ddof=1)), int(len(residual)), float(residual.median())


def default_rung_statistic() -> tuple[float, int]:
    """Width of the whole yeast wild-type turnover distribution.

    The organism default is a central value for a unit nothing is known about, so the width
    that belongs to it is the width of the distribution it was drawn from.
    """
    frame = _oed_frame(OED_SCEREVISIAE)
    return float(frame["lk"].std(ddof=1)), int(len(frame))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=VERSION)
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    checks: list[LadderCheck] = []

    sd, groups, rows = measured_rung_statistic()
    rung = KCAT_SIGMA_LADDER.rung("measured")
    checks.append(
        LadderCheck(
            tier="measured",
            shipped_log10_sigma=rung.log10_sigma,
            recomputed=sd,
            n_groups=groups,
            n_rows=rows,
            statistic=rung.statistic,
            source=rung.source,
            matches=abs(sd - rung.log10_sigma) <= SIGMA_TOLERANCE,
        )
    )

    sd, groups, rows = homolog_rung_statistic()
    rung = KCAT_SIGMA_LADDER.rung("homolog")
    checks.append(
        LadderCheck(
            tier="homolog",
            shipped_log10_sigma=rung.log10_sigma,
            recomputed=sd,
            n_groups=groups,
            n_rows=rows,
            statistic=rung.statistic,
            source=rung.source,
            matches=abs(sd - rung.log10_sigma) <= SIGMA_TOLERANCE,
        )
    )

    sd, n_pairs, signed_median = predicted_rung_statistic()
    rung = KCAT_SIGMA_LADDER.rung("predicted")
    checks.append(
        LadderCheck(
            tier="predicted",
            shipped_log10_sigma=rung.log10_sigma,
            recomputed=sd,
            n_groups=n_pairs,
            n_rows=n_pairs,
            statistic=f"{rung.statistic}; signed median {signed_median:+.4f} decades",
            source=rung.source,
            matches=abs(sd - rung.log10_sigma) <= SIGMA_TOLERANCE,
        )
    )

    sd, rows = default_rung_statistic()
    rung = KCAT_SIGMA_LADDER.rung("default")
    checks.append(
        LadderCheck(
            tier="default",
            shipped_log10_sigma=rung.log10_sigma,
            recomputed=sd,
            n_rows=rows,
            statistic=rung.statistic,
            source=rung.source,
            matches=abs(sd - rung.log10_sigma) <= SIGMA_TOLERANCE,
        )
    )
    for check in checks:
        print(
            f"[ladder] {check.tier:<9} shipped {check.shipped_log10_sigma:.2f} "
            f"recomputed {check.recomputed:.4f} "
            f"(groups {check.n_groups}, rows {check.n_rows}) "
            f"{'OK' if check.matches else 'MISMATCH'}",
            flush=True,
        )
    failed = [c.tier for c in checks if not c.matches]
    assert not failed, (
        f"the ladder no longer matches the data it was measured on: {failed}. "
        "The mirror changed; re-derive the rung rather than loosening the tolerance."
    )

    source = YeastGEM()
    gem = build_gem_tensors(source.model, model_dir=source.model_dir)
    units = gem.catalytic_units
    kinetics_dir = osp.join(DATA_ROOT, KINETICS_MIRROR)
    substrates = SubstrateMatch.from_verified_pairs(
        osp.join(kinetics_dir, VERIFIED_PAIRS_FEATURE)
    )
    predicted = load_predicted_unit_table(
        "k_cat",
        units.n_units,
        kinetics_dir,
        osp.join(EXPERIMENT_ROOT, PLAUSIBILITY_RESULT),
    )
    homolog_mirror = OED_ALL
    shipped = resolve_kcat_table(
        units,
        source.model_dir,
        OED_SCEREVISIAE,
        predicted=predicted,
        substrates=substrates,
        homolog_mirror_dir=homolog_mirror,
    )
    counts = shipped.tier_counts()
    print(f"[table] tier counts {counts}", flush=True)
    assert counts == SHIPPED_TIER_COUNTS, (
        f"resolved {counts} against the committed {SHIPPED_TIER_COUNTS}. The mirror is not "
        "the one the shipped table was built on, so the sigma table is not comparable."
    )

    per_unit_sd = predicted.log10_sd
    assert per_unit_sd is not None, "load_predicted_unit_table did not carry log10_sd"
    sigma = KCAT_SIGMA_LADDER.sigma(shipped.provenance, per_unit_sd)
    with_sigma = type(shipped).build(
        shipped.values.clone(),
        list(shipped.provenance),
        unit=shipped.unit,
        notes=shipped.notes,
        log10_sigma=sigma,
    )

    equal = {
        "values": bool(torch.equal(shipped.values, with_sigma.values)),
        "provenance": shipped.provenance == with_sigma.provenance,
        "known_mask": bool(torch.equal(shipped.known_mask, with_sigma.known_mask)),
        "experimental_mask": bool(
            torch.equal(shipped.experimental_mask, with_sigma.experimental_mask)
        ),
    }
    print(f"[neutrality] element-wise equality {equal}", flush=True)
    assert all(equal.values()), (
        f"adding log10_sigma changed a shipped field: {equal}. This is a bug, not a result."
    )

    versions_path = osp.join(RESULTS, "kinetic_table_versions.json")
    with open(versions_path) as handle:
        shipped_versions = json.load(handle)
    shipped_hashes = {
        (entry["predictor"], entry["parameter"], entry["version"]): entry["sha256"]
        for entry in shipped_versions["tables"]
    }
    hash_matches: dict[str, bool] = {}
    for name, digest in predicted.sha256.items():
        key = (name, "k_cat", predicted.version)
        hash_matches[name] = shipped_hashes.get(key) == digest
    print(f"[mirror] predictor sha256 matches shipped {hash_matches}", flush=True)

    tier_of = KCAT_SIGMA_LADDER.tier_of
    tiers = [tier_of(p) for p in shipped.provenance]
    genes_per_unit = np.zeros(units.n_units, dtype=np.int64)
    np.add.at(genes_per_unit, units.unit_gene_index[0].numpy(), 1)
    reaction_of_unit = [gem.rxn_ids[j] for j in units.unit_reaction.tolist()]
    sigma_np = sigma.numpy()
    sd_np = per_unit_sd.numpy()
    frame = pd.DataFrame(
        {
            "unit_id": np.arange(units.n_units, dtype=np.int64),
            "reaction_id": reaction_of_unit,
            "n_genes": genes_per_unit,
            "k_cat_per_s": shipped.values.numpy(),
            "provenance": [str(p) for p in shipped.provenance],
            "tier": tiers,
            "known": shipped.known_mask.numpy(),
            "experimental": shipped.experimental_mask.numpy(),
            "log10_sd_across_predictors": sd_np,
            "n_predictors": predicted.n_predictors.numpy(),
            "log10_sigma": sigma_np,
        }
    )
    out_sigma = osp.join(RESULTS, f"kcat_unit_sigma_v{args.version}.parquet")
    out_shipped = osp.join(RESULTS, f"kcat_unit_shipped_v{args.version}.parquet")
    frame.to_parquet(out_sigma, index=False)
    frame.drop(columns=["log10_sigma", "log10_sd_across_predictors"]).to_parquet(
        out_shipped, index=False
    )

    by_tier: dict[str, dict[str, float]] = {}
    for tier in ("measured", "homolog", "predicted", "default"):
        selector = np.array([t == tier for t in tiers])
        if not selector.any():
            continue
        block = sigma_np[selector]
        by_tier[tier] = {
            "n": float(selector.sum()),
            "p10": float(np.percentile(block, 10)),
            "p50": float(np.median(block)),
            "p90": float(np.percentile(block, 90)),
            "max": float(block.max()),
        }
    predicted_sel = np.array([t == "predicted" for t in tiers])
    floor = KCAT_SIGMA_LADDER.rung("predicted").log10_sigma
    beat_floor = float(
        np.mean(
            np.isfinite(sd_np[predicted_sel]) & (sd_np[predicted_sel] > floor)
        )
    )

    report = SigmaTableReport(
        version=args.version,
        built_at=datetime.now(UTC).isoformat(),
        ladder_version=KCAT_SIGMA_LADDER.version,
        checks=checks,
        tier_counts=counts,
        shipped_tier_counts=SHIPPED_TIER_COUNTS,
        n_units=units.n_units,
        predictor_sha256=dict(predicted.sha256),
        predictor_sha256_matches_shipped=hash_matches,
        elementwise_equal=equal,
        sigma_by_tier=by_tier,
        per_unit_sd_used_fraction=beat_floor,
        out_parquet=out_sigma,
        out_parquet_shipped=out_shipped,
        sha256={
            osp.basename(out_sigma): _sha256(out_sigma),
            osp.basename(out_shipped): _sha256(out_shipped),
        },
    )
    report_path = osp.join(RESULTS, "kcat_sigma_ladder.json")
    with open(report_path, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    print(f"[sigma] by tier {json.dumps(by_tier, indent=2)}", flush=True)
    print(
        f"[sigma] predicted units whose own sd beats the {floor} floor: {beat_floor:.4f}",
        flush=True,
    )
    print(f"wrote {out_sigma}\nwrote {out_shipped}\nwrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
