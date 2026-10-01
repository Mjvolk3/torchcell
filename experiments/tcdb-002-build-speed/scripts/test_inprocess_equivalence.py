# experiments/tcdb-002-build-speed/scripts/test_inprocess_equivalence.py
# [[experiments.tcdb-002-build-speed.scripts.test_inprocess_equivalence]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/test_inprocess_equivalence
"""Prove the r1 in-process and r3 single-pass paths emit the pool path's nodes and edges.

Runs SmfKuzmin2018Adapter four ways over the dev LMDB and compares the sorted node id
set and edge id set by sha256. Exit 0 only when all four agree.
"""

import hashlib
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()
os.environ["WANDB_MODE"] = "disabled"
import wandb  # noqa: E402

wandb.init(mode="disabled")
from torchcell.adapters import SmfKuzmin2018Adapter  # noqa: E402
from torchcell.datasets.scerevisiae.kuzmin2018 import SmfKuzmin2018Dataset  # noqa: E402

root = os.path.join(os.environ["DATA_ROOT"], "data/torchcell/smf_kuzmin2018")
ds = SmfKuzmin2018Dataset(root=root)
print("records", len(ds))


def digest(items: list[str]) -> str:
    """Short sha256 of a sorted id list."""
    return hashlib.sha256("\n".join(items).encode()).hexdigest()[:16]


def run(inproc: int, single_pass: bool = False) -> tuple[str, str, int, int, float]:
    """Emit every node and edge one way; return id digests, counts, and wall time."""
    t = time.time()
    ad = SmfKuzmin2018Adapter(
        dataset=ds,
        process_workers=4,
        io_workers=1,
        chunk_size=10000,
        loader_batch_size=1000,
    )
    ad.inprocess_max_records = inproc
    ad.single_pass = single_pass
    node_ids = sorted(n.get_id() + "|" + n.get_label() for n in ad.get_nodes())
    edge_ids = sorted(
        e.get_source_id() + ">" + e.get_target_id() + "|" + e.get_label()
        for e in ad.get_edges()
    )
    return (
        digest(node_ids),
        digest(edge_ids),
        len(node_ids),
        len(edge_ids),
        time.time() - t,
    )


results = {
    "pool": run(0),
    "inproc": run(25000),
    "single": run(0, single_pass=True),
    "sp+inpr": run(25000, single_pass=True),
}
for name, (nh, eh, n, e, sec) in results.items():
    print(f"{name:8s} nodes {nh} edges {eh} n={n} e={e} {sec:.1f}s")
ok = len({r[:4] for r in results.values()}) == 1
print("IDENTICAL" if ok else "DIFFER")
sys.exit(0 if ok else 1)
