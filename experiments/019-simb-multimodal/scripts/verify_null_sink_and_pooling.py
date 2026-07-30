# experiments/019-simb-multimodal/scripts/verify_null_sink_and_pooling.py
# [[experiments.019-simb-multimodal.scripts.verify_null_sink_and_pooling]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/verify_null_sink_and_pooling
"""Acceptance contract for the null sink and for sum pooling, before any GPU time is spent.

Six assertions. All are CPU-cheap and all must pass before the wave launches; each one is a
way the wave could produce an uninterpretable result if it silently failed.

  C1  null_sink=false is UNCHANGED behaviour -- the added code path must not perturb the
      arms that do not use it (A1_ref, A2_prop_sparse, A3_deep2 all run with it off).
  C2  SHAM CONTRACT: null_sink=true, bias_init=-20, trainable=false must be numerically
      identical to null_sink=false (p_null ~ 2e-9). A5_sham is the empirical paired null,
      so if it is not an identity it measures something and the null is not a null.
  C3  The sink RESTORES QUERY-DEPENDENCE at |S|=1: per-gene spread of the context must go
      from float32 epsilon to something real. This is the mechanism the arm claims.
  C4  GRADIENT REACHES the sink and revives the dead projections: null_bias must get a
      gradient, and W_Q/W_K must go from EXACTLY zero grad (today, |S|=1) to nonzero.
  C5  Sum pooling is INERT at |S|=1 -- identical to mean. This is why it is landing as a
      default rather than as an arm: on 95.4% of the expression build it changes nothing.
  C6  Sum pooling is CARDINALITY-AWARE at |S|>1 -- sum = |S| * mean, and z_S for S={A,A}
      must differ from S={A} (under mean they were bit-identical at -0.3520988).

Run:
    PYTHONPATH=<worktree> ~/miniconda3/envs/torchcell/bin/python \
        experiments/019-simb-multimodal/scripts/verify_null_sink_and_pooling.py
"""

import json
import os
import os.path as osp

import torch
from dotenv import load_dotenv

from torchcell.models.equivariant_cell_graph_transformer import (
    EquivariantPerturbationTransform,
    PerturbationHead,
)

load_dotenv()
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")

HIDDEN_DIM = 90
NUM_HEADS = 9
NUM_GENES = 6607
SEED = 0


def build(**kwargs) -> EquivariantPerturbationTransform:
    """Construct the transform with a fixed init so two builds are comparable."""
    torch.manual_seed(SEED)
    return EquivariantPerturbationTransform(
        hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, dropout=0.0, **kwargs
    ).eval()


def context_of(
    transform: EquivariantPerturbationTransform, h_genes: torch.Tensor, pert: list[int]
) -> torch.Tensor:
    """Return the raw attended context [N, d] for one strain."""
    idx = torch.tensor(pert, dtype=torch.long)
    _, context = transform(h_genes, idx, torch.zeros(len(pert), dtype=torch.long))
    return context[0]


def main() -> None:
    torch.manual_seed(SEED)
    h_genes = torch.randn(NUM_GENES, HIDDEN_DIM)
    results: dict[str, object] = {}
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results[name] = {"pass": bool(ok), "detail": detail}
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    print("null sink")
    with torch.no_grad():
        off = context_of(build(), h_genes, [7])
        sham = context_of(
            build(
                null_sink=True, null_sink_bias_init=-20.0, null_sink_trainable=False
            ),
            h_genes,
            [7],
        )
        on = context_of(
            build(null_sink=True, null_sink_bias_init=-4.0), h_genes, [7]
        )

    off_spread = float((off.max(0).values - off.min(0).values).mean())
    on_spread = float((on.max(0).values - on.min(0).values).mean())
    sham_delta = float((sham - off).abs().max())

    check(
        "C1_sink_off_unchanged",
        off_spread < 1e-6,
        f"|S|=1 context is query-independent without the sink (spread {off_spread:.2e})",
    )
    check(
        "C2_sham_is_identity",
        sham_delta < 1e-6,
        f"bias=-20 non-trainable matches sink=off to {sham_delta:.2e}",
    )
    check(
        "C3_sink_restores_query_dependence",
        on_spread > 1e-3,
        f"spread {off_spread:.2e} -> {on_spread:.4f} ({on_spread / off_spread:.2e}x)",
    )

    # C4 -- gradient. Today W_Q/W_K get EXACTLY zero at |S|=1; the sink must revive them.
    def grads(**kwargs) -> tuple[float, float | None]:
        t = build(**kwargs)
        idx = torch.tensor([7], dtype=torch.long)
        _, context = t(h_genes, idx, torch.zeros(1, dtype=torch.long))
        context.sum().backward()
        qk = t.cross_attn_layers[0].in_proj_weight.grad[: 2 * HIDDEN_DIM]
        bias_grad = (
            float(t.null_bias.grad.abs().sum()) if kwargs.get("null_sink") else None
        )
        return float(qk.abs().sum()), bias_grad

    qk_off, _ = grads()
    qk_on, bias_grad = grads(null_sink=True, null_sink_bias_init=-4.0)
    check(
        "C4_gradient_revives_qk",
        qk_off == 0.0 and qk_on > 0.0 and bias_grad is not None and bias_grad > 0.0,
        f"W_Q/W_K grad {qk_off:.1f} -> {qk_on:.1f}; null_bias grad {bias_grad:.1f}",
    )
    results["qk_grad_off"] = qk_off
    results["qk_grad_on"] = qk_on
    results["null_bias_grad"] = bias_grad

    print("pooling")
    torch.manual_seed(SEED)
    head_sum = PerturbationHead(HIDDEN_DIM, dropout=0.0, pooling="sum").eval()
    torch.manual_seed(SEED)
    head_mean = PerturbationHead(HIDDEN_DIM, dropout=0.0, pooling="mean").eval()

    h_pert = torch.randn(1, NUM_GENES, HIDDEN_DIM)
    h_cls = torch.randn(HIDDEN_DIM)

    def z_of(head: PerturbationHead, pert: list[int]) -> float:
        idx = torch.tensor(pert, dtype=torch.long)
        with torch.no_grad():
            return float(
                head(h_cls, h_pert, idx, torch.zeros(len(pert), dtype=torch.long))
            )

    single_sum, single_mean = z_of(head_sum, [7]), z_of(head_mean, [7])
    check(
        "C5_sum_inert_at_S1",
        abs(single_sum - single_mean) < 1e-6,
        f"sum {single_sum:.7f} == mean {single_mean:.7f} at |S|=1",
    )

    # Multiplicity: S={A} vs S={A,A}. Under MEAN these are bit-identical.
    dup_sum, dup_mean = z_of(head_sum, [7, 7]), z_of(head_mean, [7, 7])
    check(
        "C6_sum_is_cardinality_aware",
        abs(dup_mean - single_mean) < 1e-6 and abs(dup_sum - single_sum) > 1e-3,
        f"S={{A,A}} vs S={{A}}: mean {dup_mean:.7f} vs {single_mean:.7f} (blind); "
        f"sum {dup_sum:.7f} vs {single_sum:.7f} (aware)",
    )

    out_path = osp.join(
        EXPERIMENT_ROOT,
        "019-simb-multimodal",
        "results",
        "verify_null_sink_and_pooling.json",
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")

    if failures:
        raise SystemExit(f"FAILED: {', '.join(failures)} -- do NOT launch the wave")
    print("all contracts hold -- safe to launch")


if __name__ == "__main__":
    main()
