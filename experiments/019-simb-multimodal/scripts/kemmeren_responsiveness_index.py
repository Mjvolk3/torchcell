# experiments/019-simb-multimodal/scripts/kemmeren_responsiveness_index.py
# [[experiments.019-simb-multimodal.scripts.kemmeren_responsiveness_index]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/kemmeren_responsiveness_index
"""Recover Kemmeren2014 responsive/non-responsive labels and write a STRATIFIED split index.

THE PROBLEM. `seed` feeds `CellDataModule(random_seed=...)`, so it selects the train/val
SPLIT, not just the init. With only ~155 val strains that makes the headline metric
wildly split-dependent -- measured on the A0 baseline, identical config, three seeds:

    seed 0 ROLL5 = 0.1527    seed 1 = 0.0235    seed 2 = 0.0661     (a 6x spread)

The likely cause is dataset composition. Kemmeren2014 profiled deletion mutants in TWO
GEO series by design:

    GSE42527  RESPONSIVE mutants      (4 replicates each)
    GSE42526  NON-RESPONSIVE mutants

`pearson_per_feature` correlates predictions across val strains PER GENE. A val set drawn
mostly from the non-responsive series has near-flat targets -- almost no variance to
correlate against -- so the metric collapses regardless of the model. A responsive-rich
val set inflates it. Nothing about the architecture changed between those three runs.

WHY THIS SCRIPT AND NOT A LOADER CHANGE. `kemmeren2014.py` already COMPUTES the flag
(`sample_info["is_responsive_mutant"]`, line ~412) but never persists it to the schema, so
it is absent from the built LMDB. Adding it is the right long-term fix (tracked as a GitHub
issue) but costs a dataset rebuild + re-query, which we do not want mid-round. Instead:

  1. Recover the labels directly from the MIRRORED GEO SOFT files -- the raw artifacts the
     loader itself downloaded, already sha-pinned under `$DATA_ROOT/data/torchcell/
     microarray_kemmeren2014/raw/`. No network, no rebuild.
  2. Write a stratified `index_seed_<N>.json` + `index_details_seed_<N>.json` straight into
     the datamodule's cache dir. `CellDataModule._load_or_compute_index` loads those files
     verbatim when they exist, so a training run picks the stratified split up with NO code
     change -- just `seed=<N>`.

The stratification balances the RESPONSIVE FRACTION across train/val/test, so the three
splits are comparable in the one property that drives the metric.

Usage
-----
    python experiments/019-simb-multimodal/scripts/kemmeren_responsiveness_index.py \
        --seeds 100 101 102 --base-seed 0
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import os.path as osp
import re

from dotenv import load_dotenv

load_dotenv()

RESPONSIVE_GSE = "GSE42527"
NONRESPONSIVE_GSE = "GSE42526"
# GEO sample titles are `<gene>-del-<biological replicate>-<dye swap>` -- e.g.
# "bck1-del-1-a" -- OPTIONALLY prefixed by a hybridization-batch tag in square brackets,
# e.g. "[hs1991] abf2-del-1-d". Two parsing traps, both hit here:
#   * a character class containing '-' swallows the whole sample id, giving per-ARRAY
#     tokens instead of per-GENE ones;
#   * anchoring at '^' without allowing the bracket prefix makes `re.match` return None
#     for the MAJORITY of titles (1,891 of 2,633), which a naive comprehension then drops
#     SILENTLY -- the gene sets looked clean and complete at 230/148 while omitting 72%
#     of the data.
_TITLE_GENE = re.compile(r"^\s*(?:\[[^\]]*\]\s*)?([A-Za-z0-9]+)")


def parse_soft_gene_names(soft_path: str) -> set[str]:
    """Extract the per-sample strain/gene tokens from a GEO family SOFT file.

    Reads only ``!Sample_title`` lines -- the full SOFT carries every probe measurement and
    is far too large to parse in full for a label we can read off the titles.

    Args:
        soft_path: Path to a gzipped ``*_family.soft.gz``.

    Returns:
        Upper-cased gene/strain tokens, one per sample title.
    """
    names: set[str] = set()
    n_titles = 0
    unmatched: list[str] = []
    with gzip.open(soft_path, "rt", errors="replace") as handle:
        for line in handle:
            if not line.startswith("!Sample_title"):
                continue
            n_titles += 1
            value = line.split("=", 1)[1].strip()
            match = _TITLE_GENE.match(value)
            if match:
                names.add(match.group(1).upper())
            else:
                unmatched.append(value)
    # FAIL LOUDLY on unparsed titles. Dropping them silently is what produced a plausible
    # but 72%-incomplete label set on the first pass.
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} of {n_titles} sample titles in {osp.basename(soft_path)} "
            f"did not parse, e.g. {unmatched[:5]}"
        )
    print(f"  {osp.basename(soft_path)}: {n_titles} arrays -> {len(names)} genes")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="dir holding the mirrored *_family.soft.gz (default: DATA_ROOT mirror)",
    )
    parser.add_argument("--out", default=None, help="results dir for the label JSON")
    args = parser.parse_args()

    data_root = os.environ["DATA_ROOT"]
    raw_dir = args.raw_dir or osp.join(
        data_root, "data/torchcell/microarray_kemmeren2014/raw"
    )
    out_dir = args.out or osp.join(
        os.environ["EXPERIMENT_ROOT"], "019-simb-multimodal", "results"
    )
    os.makedirs(out_dir, exist_ok=True)

    responsive = parse_soft_gene_names(
        osp.join(raw_dir, f"{RESPONSIVE_GSE}_family.soft.gz")
    )
    nonresponsive = parse_soft_gene_names(
        osp.join(raw_dir, f"{NONRESPONSIVE_GSE}_family.soft.gz")
    )
    # A token appearing in both series is ambiguous; report rather than silently pick.
    overlap = responsive & nonresponsive

    payload = {
        "responsive_gse": RESPONSIVE_GSE,
        "nonresponsive_gse": NONRESPONSIVE_GSE,
        "n_responsive": len(responsive),
        "n_nonresponsive": len(nonresponsive),
        "n_overlap": len(overlap),
        "responsive": sorted(responsive),
        "nonresponsive": sorted(nonresponsive),
        "overlap": sorted(overlap),
    }
    out_path = osp.join(out_dir, "kemmeren_responsiveness.json")
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"{RESPONSIVE_GSE} (responsive)     : {len(responsive):>5} strain tokens")
    print(
        f"{NONRESPONSIVE_GSE} (non-responsive) : {len(nonresponsive):>5} strain tokens"
    )
    print(f"overlap (ambiguous)              : {len(overlap):>5}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
