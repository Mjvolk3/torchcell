# experiments/043-expression-proteome-morphology/scripts/alcf/aurora_batch_census_043.py
# [[experiments.043-expression-proteome-morphology.scripts.alcf.aurora_batch_census_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/alcf/aurora_batch_census_043.py
"""Which training micro-batches are heavy on Aurora? Distinct contexts and marks per batch.

Aurora cell a1024_lodo_zelezniak_onto_hlgauss_equal_s0 (PBS 8959387) page-faulted at
micro-batch 2 near optimizer step 2,400 of epoch 1 (W&B logged train/loss_step up to step
2,400; accumulate 4, so micro-batches about 9,600 to 9,800). XPU forms the attention score
matrix, so a micro-batch's memory grows with its distinct contexts (trunk) and with
records x marks (operator). This replays the records sampler's order (seed * 100003 +
epoch, sliced by the micro-batch) for given hydra overrides and prints, for epoch 1, the
heaviest micro-batches and the ones around a given micro-batch index.

    python aurora_batch_census_043.py --around 9700 -- arm=all \
        arm.exclude_from_training=[zelezniak] data.prep_version=prep_v4 seed=0
"""

from __future__ import annotations

import argparse
import json
import os.path as osp
import sys

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.dirname(HERE))


def main() -> None:
    """Replay epoch 1's micro-batch order and print the census."""
    import train_043 as tr
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    ap = argparse.ArgumentParser()
    ap.add_argument("--around", type=int, default=None)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--epoch", type=int, default=1)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    with initialize_config_dir(
        version_base=None, config_dir=osp.join(osp.dirname(HERE), "..", "conf")
    ):
        dcfg = compose("train_043_default", overrides=a.overrides)
    raw = OmegaConf.to_container(dcfg, resolve=True)
    assert isinstance(raw, dict)
    raw.pop("hydra", None)
    cfg = tr.TrainConfig043.model_validate(raw)
    gen = tr.store_paths(cfg.data.store_version)
    store = tr.PreparedStore043.load(
        cfg.paths.resolved_prep_dir(cfg.data.route, gen, cfg.data.prep_version)
    )
    view = tr.build_view(
        store,
        cfg.arm,
        cfg.data.mark_cap,
        cfg.data.target_transform,
        cfg.data.target_stats_by,
        cfg.weighting.by,
    )
    train_pos = view.positions(tr.ROLE_CODE[tr.Role.TRAIN])
    if cfg.arm.exclude_from_training:
        train_pos = train_pos[view.train_mask[train_pos]]
    order = train_pos.copy()
    np.random.default_rng(cfg.seed * 100003 + a.epoch).shuffle(order)
    chunks = [order[i : i + a.micro] for i in range(0, len(order), a.micro)]
    rec = view.rec
    marks = (store.mark_ptr[rec + 1] - store.mark_ptr[rec]).astype(np.int64)
    ctx = store.rec_context[rec]
    rows = []
    for i, c in enumerate(chunks):
        rows.append(
            {
                "micro": i,
                "n_contexts": int(len(np.unique(ctx[c]))),
                "max_marks": int(marks[c].max()),
                "records_x_max_marks": int(len(c) * marks[c].max()),
                "datasets": [view.datasets[view.local_dataset[p]] for p in c],
            }
        )
    n_ctx = np.asarray([r["n_contexts"] for r in rows])
    mm = np.asarray([r["max_marks"] for r in rows])
    print(
        json.dumps(
            {
                "n_micro": len(rows),
                "n_contexts_counts": {
                    int(k): int((n_ctx == k).sum()) for k in np.unique(n_ctx)
                },
                "max_marks_quantiles": [
                    int(np.quantile(mm, q)) for q in (0.5, 0.9, 0.99, 1.0)
                ],
                "two_contexts_and_marks_over_2048": int(
                    ((n_ctx == 2) & (mm > 2048)).sum()
                ),
            }
        )
    )
    heavy = sorted(rows, key=lambda r: (r["n_contexts"], r["max_marks"]), reverse=True)[
        :10
    ]
    print("heaviest:", json.dumps(heavy))
    if a.around is not None:
        lo, hi = max(0, a.around - 200), min(len(rows), a.around + 200)
        near = sorted(
            rows[lo:hi], key=lambda r: (r["n_contexts"], r["max_marks"]), reverse=True
        )[:8]
        print(f"heaviest in micro {lo}..{hi}:", json.dumps(near))


if __name__ == "__main__":
    main()
