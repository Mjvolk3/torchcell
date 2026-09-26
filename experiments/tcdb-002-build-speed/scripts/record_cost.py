# experiments/tcdb-002-build-speed/scripts/record_cost.py
# [[experiments.tcdb-002-build-speed.scripts.record_cost]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/record_cost
"""Per-record cost of load, rehydration, serialization and hashing on one dev LMDB.

Numbers quoted in the experiment note come from this script run against the dev
dmf_costanzo2016 LMDB. Run from the repo root with the torchcell env.
"""

import hashlib
import json
import os
import time
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv

from torchcell.datamodels.schema import Publication
from torchcell.datasets.scerevisiae.costanzo2016 import DmfCostanzo2016Dataset

N = 2000


def per_record_ms(items: list[Any], fn: Callable[[Any], Any]) -> float:
    """Milliseconds per item for ``fn`` over ``items``."""
    t = time.perf_counter()
    for item in items:
        fn(item)
    return (time.perf_counter() - t) / len(items) * 1e3


def main() -> None:
    """Print the per-record millisecond cost of each build step."""
    load_dotenv()
    root = os.path.join(os.environ["DATA_ROOT"], "data/torchcell/dmf_costanzo2016")
    ds = DmfCostanzo2016Dataset(root=root)
    print("records", len(ds))
    t = time.perf_counter()
    raw = [ds[i] for i in range(N)]
    load = (time.perf_counter() - t) / N * 1e3
    items = [ds.transform_item(r) for r in raw]
    rehydrate = per_record_ms(raw, ds.transform_item)
    exp_cls, ref_cls = ds.experiment_class, ds.reference_class
    env_cls = type(items[0]["experiment"].environment)
    print(f"raw lmdb+pickle {load:.2f} ms | transform_item {rehydrate:.2f} ms")
    print(
        "validate alone: experiment %.2f | reference %.2f | publication %.2f | environment %.2f ms"
        % (
            per_record_ms(raw, lambda r: exp_cls(**r["experiment"])),
            per_record_ms(raw, lambda r: ref_cls(**r["reference"])),
            per_record_ms(raw, lambda r: Publication(**r["publication"])),
            per_record_ms(raw, lambda r: env_cls(**r["experiment"]["environment"])),
        )
    )
    dump = per_record_ms(items, lambda it: json.dumps(it["experiment"].model_dump()))
    hsh = per_record_ms(
        items,
        lambda it: hashlib.sha256(
            json.dumps(it["experiment"].model_dump()).encode()
        ).hexdigest(),
    )
    exp = items[0]["experiment"]
    print(
        "experiment json bytes",
        len(json.dumps(exp.model_dump())),
        "environment bytes",
        len(json.dumps(exp.environment.model_dump())),
    )
    print(f"experiment model_dump+json {dump:.2f} ms | +sha256 {hsh:.2f} ms")


if __name__ == "__main__":
    main()
