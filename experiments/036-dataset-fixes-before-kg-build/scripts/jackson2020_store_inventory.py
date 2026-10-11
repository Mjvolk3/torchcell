# experiments/036-dataset-fixes-before-kg-build/scripts/jackson2020_store_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.jackson2020_store_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/jackson2020_store_inventory.py
"""Measure the built Jackson 2020 pseudobulk store and the release it was summed from.

Every count in the 2026.10.11 section of ``[[torchcell.datasets.scerevisiae.jackson2020]]``
that is not a verifier row comes from this script. It reads the dev store
``$DATA_ROOT/data/torchcell/tf_deletion_pseudobulk_jackson2020`` (read only: its records,
its ``release_ledger.json``) and the raw-mirror files the loader consumed.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/jackson2020_store_inventory.json``.

    PYTHONPATH=<worktree> python experiments/036-dataset-fixes-before-kg-build/scripts/jackson2020_store_inventory.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
from collections import Counter
from statistics import median

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from torchcell.datasets.scerevisiae import jackson2020 as j
from torchcell.verification.runners import stream_records

STORE_REL = "data/torchcell/tf_deletion_pseudobulk_jackson2020"
OUT = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))),
    "results",
    "jackson2020_store_inventory.json",
)


class Spread(BaseModel):
    """Minimum, median and maximum of one per-record quantity."""

    model_config = ConfigDict(extra="forbid")

    min: float
    median: float
    max: float


def _spread(values: list[int]) -> Spread:
    return Spread(min=min(values), median=median(values), max=max(values))


class StoreInventory(BaseModel):
    """The counts the dendron note quotes."""

    model_config = ConfigDict(extra="forbid")

    n_records: int
    n_references: int
    n_strains: int
    n_genotypes: int
    records_per_condition: dict[str, int]
    cells_per_condition: dict[str, int]
    reference_cells_per_condition: dict[str, int]
    cells_per_record: Spread
    umis_per_record: Spread
    nonzero_genes_per_record: Spread
    gene_features_in_universe: int
    gene_features_nonzero_somewhere: int
    marker_umis: dict[str, int]
    marker_share_of_umis: float
    strain_conditions_without_cells: list[str]
    table_s2_genotype_disagreements: list[str]
    legacy_features: dict[str, str]
    remapped_feature_count: int
    raw_mirror_sha256: dict[str, str]


def main() -> None:
    """Measure and write the inventory."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    store = osp.join(data_root, STORE_REL)
    with open(osp.join(store, "preprocess", "release_ledger.json")) as handle:
        ledger = json.load(handle)
    environment_of = {
        json.dumps(
            j.condition_environment(spec).model_dump(), sort_keys=True, default=str
        ): a
        for a, spec in j.build_conditions(
            j.read_author_settings(osp.join(store, "raw", j.SETTINGS_NAME))
        ).items()
    }
    records_per_condition: Counter[str] = Counter()
    cells_per_condition: Counter[str] = Counter()
    reference_cells: dict[str, int] = {}
    cells: list[int] = []
    umis: list[int] = []
    nonzero: list[int] = []
    seen_genes: set[str] = set()
    strains: set[str] = set()
    genotypes: set[str] = set()
    markers: Counter[str] = Counter()
    universe = 0
    for rec in stream_records(store):
        exp = rec["experiment"]
        condition = environment_of[
            json.dumps(exp["environment"], sort_keys=True, default=str)
        ]
        phenotype = exp["phenotype"]
        records_per_condition[condition] += 1
        cells_per_condition[condition] += int(phenotype["n_cells"])
        reference_cells[condition] = int(
            rec["reference"]["phenotype_reference"]["n_cells"]
        )
        cells.append(int(phenotype["n_cells"]))
        umis.append(int(phenotype["total_umis"]))
        nonzero.append(len(phenotype["umi_count"]))
        seen_genes.update(phenotype["umi_count"])
        markers.update({k: int(v) for k, v in phenotype["marker_umi_count"].items()})
        universe = int(phenotype["n_features"])
        (pert,) = exp["genotype"]["perturbations"]
        strains.add(str(pert["construction"]["strain_accession"]))
        genotypes.add(str(pert["systematic_gene_name"]))
    inventory = StoreInventory(
        n_records=len(cells),
        n_references=len(reference_cells),
        n_strains=len(strains),
        n_genotypes=len(genotypes),
        records_per_condition=dict(records_per_condition),
        cells_per_condition=dict(cells_per_condition),
        reference_cells_per_condition=reference_cells,
        cells_per_record=_spread(cells),
        umis_per_record=_spread(umis),
        nonzero_genes_per_record=_spread(nonzero),
        gene_features_in_universe=universe,
        gene_features_nonzero_somewhere=len(seen_genes),
        marker_umis=dict(markers),
        marker_share_of_umis=sum(markers.values()) / sum(umis),
        strain_conditions_without_cells=ledger["strain_conditions_without_cells"],
        table_s2_genotype_disagreements=ledger["table_s2_genotype_disagreements"],
        legacy_features=ledger["legacy_features"],
        remapped_feature_count=len(ledger["remapped_features"]),
        raw_mirror_sha256=dict(j.RAW_FILES),
    )
    os.makedirs(osp.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        handle.write(inventory.model_dump_json(indent=2) + "\n")
    print(inventory.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
