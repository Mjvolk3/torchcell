# experiments/019-simb-multimodal/scripts/perturbation_selector_degeneracy.py
# [[experiments.019-simb-multimodal.scripts.perturbation_selector_degeneracy]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/perturbation_selector_degeneracy
"""Measure the |S|=1 degeneracy of the perturbation cross-attention, and what a null key fixes.

The claim under test: in `EquivariantPerturbationTransform`, every one of the N gene tokens
QUERIES the |S| perturbed tokens. At |S|=1 the softmax runs over a ONE-element key set, so the
attention weight is exactly 1.0 for every query regardless of the logit -- the attended vector is
query-independent and W_Q/W_K receive no gradient. Since ~80% of rows across the expression
datasets are single deletions, that is most of the training signal.

Four measurements, all on the REAL module (not a stand-in):
  1. |S|=1 -- the attention weight matrix, and the spread of the attended vector across queries.
  2. |S|=1 -- an ABLATION: re-randomize W_Q and W_K with std=10 and measure the output change.
     Exactly 0.0 is the decisive evidence that half the attention parameters are dead weight.
  3. |S|=2 -- the same measurement, showing the selector is alive as soon as there are two keys.
  4. |S|=1 + a NULL KEY (one learned token prepended to K/V) -- shows the weight becomes
     query-dependent again, which is the (p, i) pair term the readout currently cannot see.

Run (PYTHONPATH must pin the worktree; the primary checkout has a different model):
    PYTHONPATH=<worktree> ~/miniconda3/envs/torchcell/bin/python \
        experiments/019-simb-multimodal/scripts/perturbation_selector_degeneracy.py
"""

import json
import os
import os.path as osp

import torch
import torch.nn as nn
from dotenv import load_dotenv

from torchcell.models.equivariant_cell_graph_transformer import (
    EquivariantPerturbationTransform,
)

load_dotenv()
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")

# _006's measured winner, so the numbers describe the configuration actually being swept.
HIDDEN_DIM = 90
NUM_HEADS = 9
NUM_GENES = 6607
SEED = 0


def attend(
    attn: nn.MultiheadAttention, h_genes: torch.Tensor, kv: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the cross-attention: all genes query the given key/value set.

    Args:
        attn: The module's `cross_attn`.
        h_genes: [N, d] wildtype gene embeddings (the queries).
        kv: [K, d] the key/value set.

    Returns:
        out: [N, d] attended context per query gene.
        weights: [heads, N, K] per-head attention weights.
    """
    out, weights = attn(
        h_genes.unsqueeze(0),
        kv.unsqueeze(0),
        kv.unsqueeze(0),
        need_weights=True,
        average_attn_weights=False,
    )
    return out.squeeze(0), weights.squeeze(0)


def query_spread(out: torch.Tensor) -> float:
    """Mean per-dimension range of the attended vector across query genes.

    A value at float32 epsilon (~1e-7 relative) means the context is the SAME for every
    measured gene, i.e. no term depends on the pair (perturbed gene, measured gene).
    """
    return float((out.max(dim=0).values - out.min(dim=0).values).mean())


def main() -> None:
    torch.manual_seed(SEED)
    transform = EquivariantPerturbationTransform(
        hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS
    ).eval()
    attn = transform.cross_attn

    h_genes = torch.randn(NUM_GENES, HIDDEN_DIM)
    pert_single = h_genes[[7]]  # |S| = 1, the Kemmeren regime
    pert_double = h_genes[[7, 512]]  # |S| = 2, the Sameith regime

    results: dict[str, object] = {
        "hidden_dim": HIDDEN_DIM,
        "num_heads": NUM_HEADS,
        "num_genes": NUM_GENES,
        "seed": SEED,
    }

    with torch.no_grad():
        # 1. |S| = 1 -- the degenerate case.
        out1, w1 = attend(attn, h_genes, pert_single)
        results["s1_weights_unique"] = sorted({round(float(v), 6) for v in w1.flatten()})
        results["s1_query_spread"] = query_spread(out1)
        results["s1_output_magnitude"] = float(out1.abs().mean())
        results["s1_relative_spread"] = results["s1_query_spread"] / results[
            "s1_output_magnitude"
        ]

        # 2. |S| = 1 -- ablate W_Q and W_K. If the selector is a no-op this is EXACTLY 0.
        in_proj = attn.in_proj_weight.clone()
        attn.in_proj_weight[: 2 * HIDDEN_DIM] = (
            torch.randn(2 * HIDDEN_DIM, HIDDEN_DIM) * 10.0
        )
        out1_ablated, _ = attend(attn, h_genes, pert_single)
        results["s1_wq_wk_ablation_max_change"] = float(
            (out1_ablated - out1).abs().max()
        )
        results["s1_dead_params"] = 2 * HIDDEN_DIM * HIDDEN_DIM
        results["s1_total_attn_params"] = int(
            sum(p.numel() for p in attn.parameters())
        )
        attn.in_proj_weight.copy_(in_proj)  # restore

        # 3. |S| = 2 -- the selector is alive.
        out2, w2 = attend(attn, h_genes, pert_double)
        results["s2_weight_min"] = float(w2.min())
        results["s2_weight_max"] = float(w2.max())
        results["s2_query_spread"] = query_spread(out2)

        # 4. |S| = 1 + NULL KEY. One learned token prepended to K/V. It carries no
        #    strain information -- it is identical for every row -- but it gives the
        #    softmax a second term in its denominator, which is what makes the weight a
        #    function of the QUERY gene.
        torch.manual_seed(SEED)
        null_token = nn.Parameter(torch.randn(1, HIDDEN_DIM) * 0.02)
        kv_null = torch.cat([null_token, pert_single], dim=0)
        out_null, w_null = attend(attn, h_genes, kv_null)
        real_key_weight = w_null[:, :, 1]  # mass on the actual deleted gene
        results["null_real_key_weight_min"] = float(real_key_weight.min())
        results["null_real_key_weight_max"] = float(real_key_weight.max())
        results["null_query_spread"] = query_spread(out_null)
        results["null_spread_gain_vs_s1"] = (
            results["null_query_spread"] / results["s1_query_spread"]
        )
        results["null_added_params"] = HIDDEN_DIM

    out_path = osp.join(
        EXPERIMENT_ROOT,
        "019-simb-multimodal",
        "results",
        "perturbation_selector_degeneracy.json",
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"|S|=1  attention weights (unique values) : {results['s1_weights_unique']}")
    print(f"|S|=1  query spread                      : {results['s1_query_spread']:.3e}")
    print(f"|S|=1  relative spread                   : {results['s1_relative_spread']:.2e}")
    print(
        f"|S|=1  W_Q/W_K ablation (std=10) change  : "
        f"{results['s1_wq_wk_ablation_max_change']:.1f}  "
        f"({results['s1_dead_params']} of {results['s1_total_attn_params']} attn params)"
    )
    print(
        f"|S|=2  weight range                      : "
        f"[{results['s2_weight_min']:.4f}, {results['s2_weight_max']:.4f}]"
    )
    print(f"|S|=2  query spread                      : {results['s2_query_spread']:.4f}")
    print(
        f"null   real-key weight range             : "
        f"[{results['null_real_key_weight_min']:.4f}, {results['null_real_key_weight_max']:.4f}]"
    )
    print(
        f"null   query spread                      : {results['null_query_spread']:.4f}  "
        f"({results['null_spread_gain_vs_s1']:.2e}x the |S|=1 spread, "
        f"+{results['null_added_params']} params)"
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
