# experiments/tcdb-002-build-speed/scripts/test_validated_cache.py
# [[experiments.tcdb-002-build-speed.scripts.test_validated_cache]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/test_validated_cache
"""Cached-constant transform_item: same model_dump as fresh validation, and faster."""

import json
import os
import time

from dotenv import load_dotenv

load_dotenv(
    "/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/tcdb-002-build-speed/.env"
)
from torchcell.datamodels.schema import Publication
from torchcell.datasets.scerevisiae.costanzo2016 import DmfCostanzo2016Dataset
from torchcell.datasets.scerevisiae.kuzmin2018 import SmfKuzmin2018Dataset

for cls, slug in [
    (DmfCostanzo2016Dataset, "dmf_costanzo2016"),
    (SmfKuzmin2018Dataset, "smf_kuzmin2018"),
]:
    ds = cls(root=os.path.join(os.environ["DATA_ROOT"], "data/torchcell", slug))
    N = min(2000, len(ds))
    raw = [ds[i] for i in range(N)]
    # fresh validation, no cache
    t = time.perf_counter()
    fresh = [
        {
            "experiment": ds.experiment_class(**r["experiment"]),
            "reference": ds.reference_class(**r["reference"]),
            "publication": Publication(**r["publication"]),
        }
        for r in raw
    ]
    fresh_ms = (time.perf_counter() - t) / N * 1e3
    t = time.perf_counter()
    cached = [ds.transform_item(r) for r in raw]
    cached_ms = (time.perf_counter() - t) / N * 1e3
    same = all(
        json.dumps(a[k].model_dump(), sort_keys=True)
        == json.dumps(b[k].model_dump(), sort_keys=True)
        for a, b in zip(fresh, cached)
        for k in ("experiment", "reference", "publication")
    )
    print(
        f"{slug}: fresh {fresh_ms:.2f} ms/record, cached {cached_ms:.2f} ms/record, cache entries {len(ds._validated_interned)}, model_dump identical: {same}"
    )
    # raw items still plain dicts for other readers
    print(
        "  raw item type",
        type(raw[0]["experiment"]["environment"]).__name__,
        "is dict:",
        isinstance(raw[0]["experiment"]["environment"], dict),
    )
    assert same
