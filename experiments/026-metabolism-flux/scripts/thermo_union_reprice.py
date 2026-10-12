# experiments/026-metabolism-flux/scripts/thermo_union_reprice.py
# [[experiments.026-metabolism-flux.scripts.thermo_union_reprice]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/thermo_union_reprice.py

r"""Regenerate the union reaction table so the flux layer can consume the new values.

Two corrections landed upstream of the flux layer and neither reaches it on its own.

**The extra metabolites.** ``thermo_pipeline_yeast9.py`` prices 2,528 of 2,806 species by
running the staged pipeline, and the extra rows over the plain eQuilibrator recomputation
are the written-structure lipids and the re-priced external-pKa targets. Those live in JSON
files that ``parameters.resolve_reaction_delta_g`` does not read: it reads ONE parquet. So
the staged result is written back out as
``metabolite_dgf_prime_with_external.parquet``, and ``ThermoSource.UNION_EXTERNAL`` reads
that file. A new file rather than an overwrite, so a run that predates it still reproduces,
and so a config records which table it consumed.

**The mixed-unit reaction column.** 33 of the shipped reaction column's 3,210 rows are
dGPredictor output in kJ/mol while the rest are yETFL in kcal/mol, and the parser converts
the whole column. ``resolve_reaction_delta_g`` now divides those rows back, and the before
and after are reported here for every reaction the union actually fills from that column.

Writes ``results/thermo_union_reprice.json`` and the parquet under
``$DATA_ROOT/data/torchcell/thermo_equilibrator/``.

Run from the repository root under the torchcell env::

    ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/thermo_union_reprice.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
from datetime import UTC, datetime
from typing import cast

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from torchcell.metabolism.constraints import build_gem_tensors
from torchcell.metabolism.parameters import (
    METABOLITE_TABLE_WITH_EXTERNAL_FILENAME,
    ReactionThermoTable,
    ThermoSource,
    resolve_reaction_delta_g,
)
from torchcell.metabolism.thermo_pipeline import read_result
from torchcell.metabolism.yeast_GEM import YeastGEM

load_dotenv()

DATA_ROOT = cast(str, os.environ["DATA_ROOT"])
EXPERIMENT_ROOT = cast(str, os.environ["EXPERIMENT_ROOT"])
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
THERMO_DIR = osp.join(DATA_ROOT, "data", "torchcell", "thermo_equilibrator")
PIPELINE_JSON = osp.join(RESULTS, "thermo_pipeline_yeast9.json")
OUT_JSON = osp.join(RESULTS, "thermo_union_reprice.json")

#: The factor the parser applies to the whole shipped column, and therefore the factor the
#: dGPredictor rows are divided back by. Read from the correction record so there is one
#: number rather than two.
DGPREDICTOR_ROWS = osp.join(RESULTS, "dgpredictor_reaction_rows.json")


class FilledReaction(BaseModel):
    """One reaction the union filled from the shipped REACTION column, before and after."""

    reaction_id: str
    from_dgpredictor: bool
    delta_r_g0_before_kj_per_mol: float
    delta_r_g0_after_kj_per_mol: float
    change_kj_per_mol: float


class SourceSummary(BaseModel):
    """Coverage of one ThermoSource over the whole model."""
    source: str
    n_reactions: int
    n_priced: int
    fraction_priced: float
    n_with_sigma: int
    provenance_counts: dict[str, int]
    metabolite_counts: dict[str, int]


class UnionRepriceReport(BaseModel):
    """What the regenerated union table contains and how it differs from the old one."""
    built_at: str
    metabolite_table_written: str
    n_metabolite_rows_written: int
    n_rows_from_the_recomputation: int
    n_rows_added_by_the_staged_pipeline: int
    added_by_route: dict[str, int]
    kj_per_kcal: float
    n_dgpredictor_rows: int
    filled_from_the_reaction_column: list[FilledReaction]
    sources: list[SourceSummary]
    flux_layer_note: str = Field(
        default=(
            "the flux layer consumes this through FluxLayerConfig.thermo_source; set it to "
            "'union_external' to read the extended table, and the layer refuses a table "
            "whose source disagrees with the config"
        )
    )


def _summarize(table: ReactionThermoTable) -> SourceSummary:
    counts = dict(table.counts)
    return SourceSummary(
        source=str(table.source),
        n_reactions=int(counts.pop("n_reactions")),
        n_priced=int(counts.pop("known")),
        fraction_priced=float(int(table.mask.sum()) / len(table.provenance)),
        n_with_sigma=int(counts.pop("sigma_known")),
        provenance_counts=counts,
        metabolite_counts=dict(table.met_counts),
    )


def main() -> None:
    source = YeastGEM()
    model = source.model
    gem = build_gem_tensors(model, model_dir=source.model_dir)

    # 1. The staged pipeline's metabolite table, written where resolve_reaction_delta_g
    #    can read it. Only PRICED rows go in: an unpriced row with a NaN would be read as a
    #    value by anything that forgets to check the mask.
    result = read_result(PIPELINE_JSON)
    rows = [
        {
            "met_id": price.met_id,
            "dgf_prime_kj_per_mol": price.dgf_prime_kj_per_mol,
            "sigma_kj_per_mol": price.sigma_kj_per_mol,
            "route": str(price.route),
            "p_h": price.p_h,
            "ionic_strength_m": price.ionic_strength_m,
            "p_mg": price.p_mg,
            "temperature_k": price.temperature_k,
            "detail": price.detail,
        }
        for price in result.metabolites
        if price.priced
    ]
    frame = pd.DataFrame(rows)
    out_parquet = osp.join(THERMO_DIR, METABOLITE_TABLE_WITH_EXTERNAL_FILENAME)
    frame.to_parquet(out_parquet, index=False)

    recomputation = pd.read_parquet(
        osp.join(THERMO_DIR, "metabolite_dgf_prime.parquet"),
        columns=["met_id", "dgf_prime_kj_per_mol"],
    )
    n_recomputed = int(recomputation["dgf_prime_kj_per_mol"].notna().sum())
    added = frame[frame["route"] != "cache_accession"]
    print(f"wrote {out_parquet}: {len(frame)} priced metabolites")

    # 2. The union, before and after the unit correction, on the reactions it fills from the
    #    shipped reaction column.
    with open(DGPREDICTOR_ROWS) as handle:
        correction = json.load(handle)
    factor = float(correction["kj_per_kcal"])
    dgpred = set(correction["dgpredictor_reaction_ids"])

    tables = {
        name: resolve_reaction_delta_g(gem, THERMO_DIR, source=name)
        for name in (
            ThermoSource.SHIPPED,
            ThermoSource.EQUILIBRATOR,
            ThermoSource.UNION,
            ThermoSource.UNION_EXTERNAL,
        )
    }
    union = tables[ThermoSource.UNION_EXTERNAL]
    filled: list[FilledReaction] = []
    for j, tag in enumerate(union.provenance):
        if tag != "shipped_reaction_column":
            continue
        reaction_id = gem.rxn_ids[j]
        after = float(union.delta_r_g0[j])
        # Before the correction the whole column had been multiplied by 4.184, so a
        # dGPredictor row's old value is exactly the corrected one times that factor.
        before = after * factor if reaction_id in dgpred else after
        filled.append(
            FilledReaction(
                reaction_id=reaction_id,
                from_dgpredictor=reaction_id in dgpred,
                delta_r_g0_before_kj_per_mol=before,
                delta_r_g0_after_kj_per_mol=after,
                change_kj_per_mol=after - before,
            )
        )

    report = UnionRepriceReport(
        built_at=datetime.now(UTC).isoformat(),
        metabolite_table_written=out_parquet,
        n_metabolite_rows_written=len(frame),
        n_rows_from_the_recomputation=n_recomputed,
        n_rows_added_by_the_staged_pipeline=len(added),
        added_by_route={
            str(route): int(count)
            for route, count in added["route"].value_counts().items()
        },
        kj_per_kcal=factor,
        n_dgpredictor_rows=len(dgpred),
        filled_from_the_reaction_column=filled,
        sources=[_summarize(table) for table in tables.values()],
    )
    with open(OUT_JSON, "w") as handle:
        handle.write(report.model_dump_json(indent=1))
    print(f"wrote {OUT_JSON}")
    for summary in report.sources:
        print(
            f"  {summary.source:16s} {summary.n_priced:5d}/{summary.n_reactions}"
            f" = {summary.fraction_priced:.3%}, sigma on {summary.n_with_sigma}"
        )
    print("  reactions filled from the shipped reaction column:")
    for row in filled:
        print(
            f"    {row.reaction_id} {'dGPredictor' if row.from_dgpredictor else 'yETFL'}"
            f" {row.delta_r_g0_before_kj_per_mol:12.3f} ->"
            f" {row.delta_r_g0_after_kj_per_mol:10.3f}"
            f" ({row.change_kj_per_mol:+.3f})"
        )


if __name__ == "__main__":
    main()
