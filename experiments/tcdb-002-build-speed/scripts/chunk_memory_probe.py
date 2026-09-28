# experiments/tcdb-002-build-speed/scripts/chunk_memory_probe.py
# [[experiments.tcdb-002-build-speed.scripts.chunk_memory_probe]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/chunk_memory_probe
"""Peak memory of ONE single-pass node chunk, measured in-process.

    python experiments/tcdb-002-build-speed/scripts/chunk_memory_probe.py <Dataset> <n>

Builds the adapter over the first ``n`` records of the dev-tree dataset and runs the
single-pass node traversal in-process (no pool, no loader children), which is exactly
one worker's job for an ``n``-record chunk. Prints one TSV line: dataset, n, resolved
JSON bytes per record (the byte budget's estimate), baseline RSS, peak RSS, and the
peak-over-baseline per record and as a multiple of the resolved JSON. Rows are
BioCypher objects, not rendered CSV lines (the conda env's BioCypher is not the
container's), so this is an upper bound on a fast-writer worker's rows and an exact
measure of the records and pydantic objects it holds. Run each size in a fresh
process: ru_maxrss is a high-water mark.
"""

import inspect
import os
import os.path as osp
import resource
import sys

os.environ["WANDB_MODE"] = "disabled"
import wandb  # noqa: E402

wandb.init(mode="disabled")
from dotenv import load_dotenv  # noqa: E402

from torchcell.knowledge_graphs.dataset_adapter_map import (  # noqa: E402
    dataset_adapter_map,
)

load_dotenv(osp.join(osp.dirname(osp.abspath(__file__)), "../../../.env"))


def rss_gb() -> float:
    """Current resident set size of this process in GB."""
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9


def main(name: str, n: int) -> None:
    """Run one n-record single-pass node chunk in-process and report peak memory."""
    dataset_class, adapter_class = next(
        (d, a) for d, a in dataset_adapter_map.items() if d.__name__ == name
    )
    root = osp.join(
        os.environ["DATA_ROOT"],
        inspect.signature(dataset_class.__init__).parameters["root"].default,
    )
    full = dataset_class(root=root)
    dataset = full[list(range(min(n, len(full))))] if n < len(full) else full
    adapter = adapter_class(
        dataset=dataset,
        process_workers=1,
        io_workers=1,
        chunk_size=10**9,
        loader_batch_size=1000,
    )
    adapter.single_pass = True
    adapter.inprocess_max_records = 10**9
    record_bytes = adapter._estimate_record_bytes()
    base = rss_gb()
    rows = sum(1 for _ in adapter.get_nodes())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    over = peak - base
    print(
        f"{name}\t{len(dataset)}\t{rows}\t{record_bytes}\t{base:.2f}\t{peak:.2f}\t"
        f"{over * 1e9 / len(dataset) / 1e3:.1f}\t{over * 1e9 / len(dataset) / record_bytes:.1f}"
    )


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
