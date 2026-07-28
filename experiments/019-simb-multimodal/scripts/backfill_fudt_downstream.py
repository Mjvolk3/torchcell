# experiments/019-simb-multimodal/scripts/backfill_fudt_downstream.py
# [[experiments.019-simb-multimodal.scripts.backfill_fudt_downstream]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/backfill_fudt_downstream
"""Rebuild the FUDT downstream (species LM 3') embedding over the FULL gene set.

THE DEFECT. `species_downstream.pt` holds 6,579 of 6,607 genes; `species_upstream.pt`
holds all 6,607. The 28 absentees are exactly the mitochondrial `Q0*` genes:

    Q0010 Q0017 Q0032 Q0045 Q0050 Q0055 Q0060 Q0065 Q0070 Q0075 Q0080 Q0085 Q0092
    Q0105 Q0110 Q0115 Q0120 Q0130 Q0140 Q0142 Q0143 Q0144 Q0160 Q0182 Q0250 Q0255
    Q0275 Q0297

NOT a data limitation -- verified that `window_three_prime(300, True, allow_undersize=True)`
returns a full 300 bp selection for Q0010, Q0250 and a nuclear control alike. The two files
were simply built at different times against different gene sets, and only upstream was
rebuilt after the mitochondrial genes entered. `process()` has no try/except, so a genuine
per-gene failure would have crashed the build rather than silently dropping 28 genes.

WHY IT MATTERS. A partial embedding is worse than an absent one: the model concatenates
`[fudt_upstream, fudt_downstream]` by summing feature dims, so 28 genes would arrive with
missing or zero-filled 3' features -- silently information-free for a subset, with nothing
in the logs to say so. That is the same failure mode as the esm2 `no_dubious` variants
emitting 684 all-zero vectors.

Moves the stale artifact to the deprecation graveyard (never deletes) and re-runs
`process()` over the full gene set: 6,607 sequences of 300 bp through the downstream
species LM.

Run from repo root (or via gh_backfill_fudt_downstream.slurm):
    python experiments/019-simb-multimodal/scripts/backfill_fudt_downstream.py
"""

from __future__ import annotations

import os
import os.path as osp
import shutil

from dotenv import load_dotenv

_WT_ENV = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", "..", ".env"))
load_dotenv(
    _WT_ENV
    if osp.exists(_WT_ENV)
    else osp.expanduser("~/Documents/projects/torchcell/.env")
)

import torch  # noqa: E402

from torchcell.datasets.fungal_up_down_transformer import (  # noqa: E402
    FungalUpDownTransformerDataset,
)
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402

MODEL_NAME = "species_downstream"
DEPRECATED = "/scratch/projects/torchcell-deprecated"


def _ids(path: str) -> set[str]:
    data, _ = torch.load(path, map_location="cpu", weights_only=False)
    return set(data.id)


def main() -> None:
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(data_root, "data/scerevisiae/fudt_embedding")
    processed = osp.join(root, "processed", f"{MODEL_NAME}.pt")

    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    gene_set = set(genome.gene_set)
    print(f"genome gene_set: {len(gene_set)}")

    if osp.exists(processed):
        before = _ids(processed)
        missing = sorted(gene_set - before)
        print(f"existing {MODEL_NAME}.pt: {len(before)} ids, missing {len(missing)}")
        if not missing:
            print("already complete -- nothing to do")
            return
        print(f"missing: {missing}")
        graveyard = osp.join(DEPRECATED, "fudt-downstream-partial-20260727")
        os.makedirs(graveyard, exist_ok=True)
        dest = osp.join(graveyard, f"{MODEL_NAME}.pt")
        shutil.move(processed, dest)
        print(f"moved stale artifact -> {dest}")

    # Absent processed file makes InMemoryDataset re-run process() over the full gene set.
    ds = FungalUpDownTransformerDataset(root=root, genome=genome, model_name=MODEL_NAME)
    print(f"rebuilt: {len(ds)} records")

    after = _ids(processed)
    still = sorted(gene_set - after)
    dim = len(next(iter(ds[0].embeddings.values())).flatten())
    print(f"\n{MODEL_NAME}.pt: {len(after)} ids, dim={dim}, still missing {len(still)}")
    if still:
        raise RuntimeError(f"backfill incomplete, still missing {len(still)}: {still}")
    print("COMPLETE -- all genes covered")


if __name__ == "__main__":
    main()
