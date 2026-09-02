# experiments/026-metabolism-flux/scripts/flux_sampling_demo.py
# [[experiments.026-metabolism-flux.scripts.flux_sampling_demo]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/flux_sampling_demo.py

r"""Amortized flux sampling: draw a flux distribution per genotype in one forward pass.

Run from the worktree root, after ``fva_reference.py``::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/flux_sampling_demo.py --epochs 8

WHAT IS BEING DEMONSTRATED
--------------------------
Classical flux sampling runs one Markov chain per genotype over that genotype's feasible
polytope. It ignores data, targets a uniform distribution, and costs minutes per genotype:
Merzbacher et al. (2025) report 1,159 yeast deletions at 124 samples each. It does not
amortize, so :math:`\binom{1161}{2} = 673{,}380` deletion pairs would be that many separate
random walks.

Here the distribution is put on a **latent** and pushed through the same deterministic map
the deterministic head uses:

.. math::
    z \sim q_\phi(z \mid H_{\mathrm{pert}}), \qquad
    v = v^{\ell}(\varepsilon,p) + \big(v^{u}(\varepsilon,p) - v^{\ell}(\varepsilon,p)\big)\,\sigma(z)

Every draw lands inside the genotype's box by construction, so a sample is feasible with
respect to bounds, directionality and capacity without any rejection step. Drawing 128
samples for a new genotype is 128 evaluations of one network, and a double deletion is the
same forward pass as a single.

**The trap this avoids.** A per-reaction marginal distribution is not a distribution over
flux vectors: :math:`Sv=0` constrains the JOINT, so sampling each :math:`v_j` from an
independent marginal almost surely violates it. Merzbacher hit the same wall from the other
side, reporting that their deep models failed, which they attribute to "the fluxes being
linearly correlated through :math:`Sv=0`". Those correlations are the constraint. Sampling a
shared latent and mapping it to all 4,131 reactions at once is what keeps the coordinates
coupled.

**The evaluation, and it needs no labels.** Per reaction, compare the model's posterior
interval width against the flux-variability width from the constraints alone. A narrower
model interval means the phenotype data added information. Because it is label-free it can
be run on double deletions where no production measurement exists, which is the route to
scoring an inverse-design extrapolation before the experiment is done.

*Caveat stated plainly:* classical sampling targets a uniform distribution over the
polytope; this targets whatever the data and priors induce. Widths are comparable, the
distributions are not the same object.
"""

import argparse
import json
import os
import os.path as osp
import sys

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from train_flux import ARMS, build_dataset, run_arm  # noqa: E402

load_dotenv()

RESULTS_DIR = osp.join(os.environ["EXPERIMENT_ROOT"], "026-metabolism-flux", "results")


def main() -> None:
    """Train a stochastic flux head briefly, then sample and score against FVA."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--genotypes", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    dataset = build_dataset()

    # Train the anchored arm with the stochastic head switched on. `run_arm` returns the
    # trained model through the sampler hook so the samples come from a FITTED posterior
    # rather than from an untrained prior, which would make the width comparison
    # meaningless.
    arm = dict(ARMS["flux_anchored"])
    arm["stochastic"] = True
    result = run_arm(
        "flux_anchored_stochastic",
        args.seed,
        dataset,
        args,
        arm_override=arm,
        return_model=True,
    )
    model = result.pop("model")
    device = next(model.parameters()).device
    layer = model.flux_layer
    layer.config.stochastic = True

    from torch_geometric.loader import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=min(args.genotypes, 32),
        shuffle=False,
        follow_batch=["perturbation_indices", "phenotype_values"],
    )
    batch = next(iter(loader)).to(device)
    cell_graph = dataset.cell_graph.to(device)

    # Sampling: the layer stays in train mode ONLY so the reparameterized draw is taken;
    # dropout is disabled explicitly so the spread measured is the flux posterior, not
    # dropout noise wearing its clothes.
    model.eval()
    layer.train()
    for module in layer.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()

    draws = []
    with torch.no_grad():
        for _ in range(args.samples):
            _, reps = model(cell_graph, batch)
            draws.append(reps["flux"]["v"].cpu())
    samples = torch.stack(draws, dim=0)  # [S, B, r]
    lo = samples.quantile(0.025, dim=0)
    hi = samples.quantile(0.975, dim=0)
    model_width = (hi - lo).mean(dim=0).numpy()  # mean over genotypes, per reaction

    fva = pd.read_csv(osp.join(RESULTS_DIR, "fva_wildtype.csv"), index_col=0)
    fva_width = fva.reindex(layer.rxn_ids)["width"].to_numpy()

    licensed = np.isfinite(fva_width) & (fva_width > 1e-9) & (fva_width <= 1.0)
    n_licensed = int(licensed.sum())
    narrower = int((model_width[licensed] < fva_width[licensed]).sum())

    report = {
        "n_samples_per_genotype": args.samples,
        "n_genotypes": int(samples.shape[1]),
        "n_reactions": int(samples.shape[2]),
        "epochs_trained": args.epochs,
        "best_val_betaxanthin": result["best"]["val_betaxanthin"],
        "fva_licensed_reactions_width_le_1": n_licensed,
        "n_model_interval_narrower_than_fva": narrower,
        "frac_model_interval_narrower": (
            narrower / n_licensed if n_licensed else None
        ),
        "median_model_width_licensed": float(np.median(model_width[licensed]))
        if n_licensed
        else None,
        "median_fva_width_licensed": float(np.median(fva_width[licensed]))
        if n_licensed
        else None,
        "median_width_reduction": float(
            np.median(fva_width[licensed] - model_width[licensed])
        )
        if n_licensed
        else None,
        # Feasibility of the SAMPLES, not of a mean. A sampler that is feasible only on
        # average is not a feasible sampler.
        "sample_feasibility": {
            k: float(v)
            for k, v in result["final"].items()
            if k.startswith("feas_")
        },
        "note": (
            "Classical sampling targets a uniform distribution over the polytope; this "
            "targets whatever the data and priors induce. Widths are comparable, the "
            "distributions are not the same object."
        ),
    }
    np.save(osp.join(RESULTS_DIR, "sampler_widths.npy"), model_width)
    with open(osp.join(RESULTS_DIR, "flux_sampling_demo.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
