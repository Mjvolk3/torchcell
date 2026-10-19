# experiments/database/scripts/isolate_gene_content_vs_s288c.py
# [[experiments.database.scripts.isolate_gene_content_vs_s288c]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/isolate_gene_content_vs_s288c
"""How far each natural isolate's gene content sits from the S288C reference.

Reads the built natural-isolate stores (Caudal 2024 transcriptomes, Muenzner 2024
proteomes) and counts, per isolate, the perturbations its genotype carries against
S288C: accessory ORFs present that the reference lacks, reference ORFs absent, sequence
variants and natural loss-of-function alleles. Nothing is inferred; the counts are the
stored ``SegregantGenotype``-free isolate genotypes exactly as the loaders built them,
where presence and absence mean reference MEMBERSHIP (whether the pangenome column maps
to an S288C name), with copy number on the presence record.

Writes ``experiments/database/results/isolate_gene_content_vs_s288c.json`` with the
per-dataset distribution (min, median, mean, max) of each count, the number of distinct
non-reference ORFs seen in at least one isolate, and the number of distinct reference
ORFs absent in at least one. The figures are quoted in
notes-tex/database/database-expansion-200/sections/state.tex.

Run from the repo root:
  python experiments/database/scripts/isolate_gene_content_vs_s288c.py
"""

from __future__ import annotations

import json
import os
import statistics as st
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from torchcell.datasets.scerevisiae.caudal2024 import CaudalPanTranscriptome2024Dataset
from torchcell.datasets.scerevisiae.muenzner2024 import ProteomeMuenzner2024Dataset

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
OUT = SCRIPT.parent.parent / "results" / "isolate_gene_content_vs_s288c.json"

STORES = {
    "CaudalPanTranscriptome2024Dataset": (
        CaudalPanTranscriptome2024Dataset,
        "caudal_pantranscriptome2024",
    ),
    "ProteomeMuenzner2024Dataset": (ProteomeMuenzner2024Dataset, "proteome_muenzner2024"),
}


class Distribution(BaseModel):
    """Summary of one per-isolate count over a store."""

    min: int
    median: float
    mean: float
    max: int


class StoreGeneContent(BaseModel):
    """Per-isolate gene-content distance from S288C over one built store."""

    dataset: str
    records: int
    accessory_orfs_present: Distribution
    reference_orfs_absent: Distribution
    sequence_variants: Distribution
    natural_loss_of_function: Distribution
    all_perturbations: Distribution
    distinct_accessory_orfs_present: int
    distinct_reference_orfs_absent: int
    perturbation_type_totals: dict[str, int]


def _dist(xs: list[int]) -> Distribution:
    return Distribution(min=min(xs), median=st.median(xs), mean=st.mean(xs), max=max(xs))


def _gene(p: dict[str, Any]) -> str:
    for key in ("systematic_gene_name", "gene_name", "pangenome_id", "name"):
        if p.get(key):
            return str(p[key])
    raise KeyError(f"perturbation carries no gene identifier: {sorted(p)}")


def measure(dataset: str, cls: type, subpath: str, data_root: Path) -> StoreGeneContent:
    ds = cls(root=str(data_root / "data" / "torchcell" / subpath))
    present, absent, variants, lof, total = [], [], [], [], []
    present_ids: set[str] = set()
    absent_ids: set[str] = set()
    totals: Counter[str] = Counter()
    for i in range(len(ds)):
        experiment = ds[i]["experiment"]
        record = experiment if isinstance(experiment, dict) else experiment.model_dump()
        perts = record["genotype"]["perturbations"]
        kinds = Counter(p["perturbation_type"] for p in perts)
        totals.update(kinds)
        present.append(sum(n for k, n in kinds.items() if "presence" in k))
        absent.append(sum(n for k, n in kinds.items() if "absence" in k))
        variants.append(kinds.get("sequence_variant", 0))
        lof.append(kinds.get("natural_loss_of_function", 0))
        total.append(len(perts))
        for p in perts:
            if "presence" in p["perturbation_type"]:
                present_ids.add(_gene(p))
            elif "absence" in p["perturbation_type"]:
                absent_ids.add(_gene(p))
    return StoreGeneContent(
        dataset=dataset,
        records=len(ds),
        accessory_orfs_present=_dist(present),
        reference_orfs_absent=_dist(absent),
        sequence_variants=_dist(variants),
        natural_loss_of_function=_dist(lof),
        all_perturbations=_dist(total),
        distinct_accessory_orfs_present=len(present_ids),
        distinct_reference_orfs_absent=len(absent_ids),
        perturbation_type_totals=dict(totals),
    )


def main() -> None:
    load_dotenv(REPO / ".env")
    data_root = Path(os.environ["DATA_ROOT"])
    results = [measure(name, cls, sub, data_root) for name, (cls, sub) in STORES.items()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "date": date.today().isoformat(),
                "script": str(SCRIPT.relative_to(REPO)),
                "stores": [r.model_dump() for r in results],
            },
            indent=2,
        )
        + "\n"
    )
    for r in results:
        print(
            f"{r.dataset}: {r.records} isolates; accessory present median "
            f"{r.accessory_orfs_present.median:.0f} (max {r.accessory_orfs_present.max}); "
            f"reference absent median {r.reference_orfs_absent.median:.0f} "
            f"(max {r.reference_orfs_absent.max}); variants median "
            f"{r.sequence_variants.median:.0f}; LoF median {r.natural_loss_of_function.median:.0f}; "
            f"distinct accessory {r.distinct_accessory_orfs_present}, distinct absent "
            f"{r.distinct_reference_orfs_absent}"
        )
    print(f"Wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
