# experiments/tcdb-002-build-speed/scripts/test_fast_csv_equivalence.py
# [[experiments.tcdb-002-build-speed.scripts.test_fast_csv_equivalence]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/test_fast_csv_equivalence
"""Prove the fast CSV sink writes what BioCypher's writer writes, byte for byte.

Runs the same adapters twice into two BioCypher output directories: once through
``bc.write_nodes`` / ``bc.write_edges`` (production), once through
``FastCsvSink`` with rows rendered in the workers (r5). Then compares, per label, the
header file and the multiset of data lines across all part files, and the set of
labels in the import call. Meant to run INSIDE the tc-neo4j container (BioCypher
0.15.2, the production version), with the worktree on PYTHONPATH:

    DATASETS=SmfKuzmin2018Dataset,DmfCostanzo2016Dataset CAP=20000 python -m \
        experiments.tcdb-002-build-speed.scripts.test_fast_csv_equivalence
"""

import filecmp
import glob
import os
import os.path as osp
import sys
import tempfile
from collections import Counter

os.environ.setdefault("WANDB_MODE", "disabled")
import wandb  # noqa: E402

wandb.init(mode="disabled")
from dotenv import load_dotenv  # noqa: E402

from biocypher import BioCypher  # type: ignore[attr-defined]  # noqa: E402
from torchcell.fast_csv import FastCsvSink, build_row_specs  # noqa: E402
from torchcell.knowledge_graphs.dataset_adapter_map import (  # noqa: E402
    dataset_adapter_map,
)
from torchcell.knowledge_graphs.subset import subset_dataset  # noqa: E402

load_dotenv("/.env")
DATA_ROOT = os.environ["DATA_ROOT"]
NAMES = os.environ.get("DATASETS", "SmfKuzmin2018Dataset").split(",")
CAP = int(os.environ.get("CAP", "0")) or None
WORKERS = int(os.environ.get("WORKERS", "4"))


def build(out: str, fast: bool) -> dict[str, int]:
    """Generate CSVs for NAMES into ``out`` one way; return rows written per file."""
    bc = BioCypher(
        output_directory=out,
        biocypher_config_path=os.environ["BIOCYPHER_CONFIG_PATH"],
        schema_config_path=os.environ["SCHEMA_CONFIG_PATH"],
    )
    specs = build_row_specs(bc) if fast else None
    sink = FastCsvSink(bc, specs) if specs is not None else None
    for dataset_class, adapter_class in dataset_adapter_map.items():
        if dataset_class.__name__ not in NAMES:
            continue
        import inspect

        root = osp.join(
            DATA_ROOT,
            inspect.signature(dataset_class.__init__).parameters["root"].default,
        )
        dataset = subset_dataset(dataset_class(root=root), CAP, 42, None)
        adapter = adapter_class(
            dataset=dataset,
            process_workers=WORKERS,
            io_workers=1,
            chunk_size=10000,
            loader_batch_size=1000,
        )
        adapter.inprocess_max_records = 25000
        adapter.single_pass = True
        adapter.row_specs = specs
        if sink is not None:
            sink.write_nodes(adapter.get_nodes())
            sink.write_edges(adapter.get_edges())
        else:
            bc.write_nodes(adapter.get_nodes())
            bc.write_edges(adapter.get_edges())
    if sink is not None:
        sink.finish()
    bc.write_import_call()
    bc.write_schema_info(as_node=True)
    counts = {}
    for path in glob.glob(osp.join(out, "*-part*.csv")):
        with open(path) as handle:
            counts[osp.basename(path)] = sum(1 for _ in handle)
    return counts


def lines_by_label(out: str) -> dict[str, Counter[str]]:
    """Multiset of data lines per label across its part files."""
    result: dict[str, Counter[str]] = {}
    for path in sorted(glob.glob(osp.join(out, "*-part*.csv"))):
        label = osp.basename(path).split("-part")[0]
        with open(path) as handle:
            result.setdefault(label, Counter()).update(handle)
    return result


def main() -> None:
    """Build both ways and compare."""
    base = tempfile.mkdtemp(prefix="fastcsv-", dir=osp.join(DATA_ROOT, "biocypher-out"))
    slow_dir, fast_dir = osp.join(base, "biocypher"), osp.join(base, "fast")
    print("biocypher writer ->", slow_dir)
    build(slow_dir, fast=False)
    print("fast sink ->", fast_dir)
    build(fast_dir, fast=True)
    slow, fast = lines_by_label(slow_dir), lines_by_label(fast_dir)
    ok = True
    for label in sorted(set(slow) | set(fast)):
        a, b = slow.get(label, Counter()), fast.get(label, Counter())
        same = a == b
        ok &= same
        print(
            f"{'OK ' if same else 'DIFF'} {label:32s} biocypher {sum(a.values()):>9,} fast {sum(b.values()):>9,}"
        )
        if not same:
            only_a = list((a - b).elements())[:2]
            only_b = list((b - a).elements())[:2]
            print("   only biocypher:", [x[:160] for x in only_a])
            print("   only fast:     ", [x[:160] for x in only_b])
    for header in sorted(glob.glob(osp.join(slow_dir, "*-header.csv"))):
        other = osp.join(fast_dir, osp.basename(header))
        same = osp.exists(other) and filecmp.cmp(header, other, shallow=False)
        ok &= same
        print(f"{'OK ' if same else 'DIFF'} header {osp.basename(header)}")
    # BioCypher keeps the --nodes/--relationships entries in sets, so their order in
    # the script is hash-randomized per process; compare the token multiset.
    call_a = open(osp.join(slow_dir, "neo4j-admin-import-call.sh")).read()
    call_b = open(osp.join(fast_dir, "neo4j-admin-import-call.sh")).read()
    same_call = sorted(call_a.replace(slow_dir, "X").split()) == sorted(
        call_b.replace(fast_dir, "X").split()
    )
    ok &= same_call
    print(f"{'OK ' if same_call else 'DIFF'} import call")
    print("IDENTICAL" if ok else "DIFFER")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
