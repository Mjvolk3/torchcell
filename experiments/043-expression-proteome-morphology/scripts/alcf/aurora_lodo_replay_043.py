# experiments/043-expression-proteome-morphology/scripts/alcf/aurora_lodo_replay_043.py
# [[experiments.043-expression-proteome-morphology.scripts.alcf.aurora_lodo_replay_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/alcf/aurora_lodo_replay_043.py
"""Bounds audit and CPU replay of the micro-batches an Aurora lodo cell faulted on.

Both Aurora leave-one-dataset-out cells (lodo_hu at optimizer step ~2,950, lodo_zelezniak
at ~2,400, PBS 8959387) died on a GPU write page fault, and no other cell of 24 did. An
out-of-bounds index raises IndexError on CPU where XPU can write out of bounds silently.

1. audit: every micro-batch of epoch 1 (records sampler, seed * 100003 + 1, micro-batch
   ``--micro``) is collated with ``make_batch`` and every integer index it carries is
   checked against the size of the table it indexes in the model built for the arm.
2. replay: micro-batches ``--lo`` to ``--hi`` run forward and backward on CPU in training
   mode (bf16 autocast off), and the gradient norm is checked finite.

    python aurora_lodo_replay_043.py --lo 11700 --hi 11900 -- arm=all \
        arm.exclude_from_training=[hu] data.prep_version=prep_v4 \
        model.conditioning.source=ontology seed=0
"""

from __future__ import annotations

import argparse
import json
import math
import os.path as osp
import sys
import time

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.dirname(HERE))


def main() -> None:
    """Audit every micro-batch of epoch 1, then replay the window on CPU."""
    import train_043 as tr
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    with initialize_config_dir(
        version_base=None, config_dir=osp.join(osp.dirname(HERE), "..", "conf")
    ):
        dcfg = compose("train_043_default", overrides=[*a.overrides, "device=cpu"])
    raw = OmegaConf.to_container(dcfg, resolve=True)
    assert isinstance(raw, dict)
    raw.pop("hydra", None)
    cfg = tr.TrainConfig043.model_validate(raw)
    torch.manual_seed(cfg.seed)
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
    tables = [
        tr.h5_vector_source(cfg.paths, gen)(e, view.used_ids[e])
        for e in range(len(tr.ENCODERS))
    ]
    model = tr.build_model(cfg, view, tables)
    objective = tr.ReadoutLossFamily043(model.config, None)
    mc = model.config
    sizes = [int(t.shape[0]) for t in tables]
    G = len(store.meta.genes)
    train_pos = view.positions(tr.ROLE_CODE[tr.Role.TRAIN])
    if cfg.arm.exclude_from_training:
        assert view.train_mask is not None
        train_pos = train_pos[view.train_mask[train_pos]]
    order = train_pos.copy()
    np.random.default_rng(cfg.seed * 100003 + 1).shuffle(order)
    chunks = [order[i : i + a.micro] for i in range(0, len(order), a.micro)]
    print(
        json.dumps(
            {
                "datasets": list(view.datasets),
                "n_train": len(train_pos),
                "n_micro": len(chunks),
                "table_rows": sizes,
                "n_parents": mc.n_parents,
                "n_datasets": mc.n_datasets,
                "n_envs": mc.n_envs,
                "n_exp_types": len(mc.exp_type_classes),
                "conditioning": mc.conditioning,
                "weight_group": None
                if view.weight_group is None
                else [int(x) for x in view.weight_group],
            }
        ),
        flush=True,
    )

    def check(i: int, b: object, tg: object) -> list[str]:
        bad = []
        ctx, mk = b.contexts, b.marks  # type: ignore[attr-defined]
        C = int(ctx.key_rows.shape[0])
        B = int(b.record_context.shape[0])  # type: ignore[attr-defined]

        def over(name: str, t: torch.Tensor, n: int, lo: int = 0) -> None:
            if t.numel() and (int(t.max()) >= n or int(t.min()) < lo):
                bad.append(
                    f"{name} range [{int(t.min())}, {int(t.max())}] vs [{lo}, {n})"
                )

        for e in range(len(sizes)):
            over(f"ctx.key_rows[{e}]", ctx.key_rows[..., e], sizes[e], -1)
            over(f"marks.key_rows[{e}]", mk.key_rows[..., e], sizes[e], -1)
        over("record_context", b.record_context, C)  # type: ignore[attr-defined]
        over("marks.gene+1", mk.gene + 1, G + 1)
        over("marks.parent", mk.parent, mc.n_parents)
        over("dataset_ids", b.dataset_ids, mc.n_datasets)  # type: ignore[attr-defined]
        over("env_ids", b.env_ids, mc.n_envs)  # type: ignore[attr-defined]
        over("exp_type_ids", b.exp_type_ids, len(mc.exp_type_classes))  # type: ignore[attr-defined]
        over("tg.dataset_ids", tg.dataset_ids, len(view.datasets))  # type: ignore[attr-defined]
        over("tg.gene_rows", tg.gene_rows, B)  # type: ignore[attr-defined]
        over("tg.record_rows", tg.record_rows, B)  # type: ignore[attr-defined]
        if tg.groups is not None:  # type: ignore[attr-defined]
            over("tg.groups", tg.groups, max(1, int(tg.groups.max()) + 1))  # type: ignore[attr-defined]
        if view.weight_group is not None:
            over("weight_group", torch.as_tensor(view.weight_group), len(view.datasets))
        if b.ontology is not None:  # type: ignore[attr-defined]
            o = b.ontology  # type: ignore[attr-defined]
            if not bool(torch.isfinite(o).all()):
                bad.append("ontology non-finite")
        return bad

    if not a.skip_audit:
        t0 = time.time()
        n_bad = 0
        for i, c in enumerate(chunks):
            b, tg = tr.make_batch(view, c)
            bad = check(i, b, tg)
            if bad:
                n_bad += 1
                if n_bad <= 20:
                    print(f"[audit] micro {i}: {bad}", flush=True)
        print(
            f"[audit] {len(chunks)} micro-batches, {n_bad} with an index out of range "
            f"({time.time() - t0:.0f} s)",
            flush=True,
        )

    model.train()
    t0 = time.time()
    n_inf = 0
    for i in range(a.lo, min(a.hi, len(chunks))):
        b, tg = tr.make_batch(view, chunks[i])
        out = model(b, tg.gene_rows, tg.record_rows)
        loss, _ = tr.batch_loss(
            out.gene,
            out.record,
            tg,
            objective,
            len(view.datasets),
            cfg.weighting.kind,
            None if view.weight_group is None else torch.from_numpy(view.weight_group),
        )
        model.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9))
        if not math.isfinite(gn) or not math.isfinite(float(loss)):
            n_inf += 1
            print(f"[replay] micro {i}: loss {float(loss)} grad norm {gn}", flush=True)
        if (i - a.lo) % 50 == 0:
            print(
                f"[replay] micro {i}: loss {float(loss):.4f} grad norm {gn:.3f} "
                f"({time.time() - t0:.0f} s)",
                flush=True,
            )
    print(
        f"[replay] micro {a.lo}..{a.hi}: {n_inf} non-finite, no exception "
        f"({time.time() - t0:.0f} s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
