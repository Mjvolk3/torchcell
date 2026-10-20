# experiments/040-inhibitor-synergy-wetlab/scripts/check_round11_pieces.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.check_round11_pieces]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/check_round11_pieces
"""CPU checks of round 11's five new pieces, against the model's own code.

    python experiments/040-inhibitor-synergy-wetlab/scripts/check_round11_pieces.py


A CPU smoke of train_mixture.py is impossible (SDPA materializes the gene-by-gene
matrix; gene batch 32 hit 71 GB), so the pieces are checked directly on a small
MixtureModel with no encoder: gene_logits, host_logits and medium_state do not touch it.
"""

import os.path as osp
import sys

import numpy as np
import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from train_mixture import (  # noqa: E402
    Medium,
    MixtureConfig,
    MixtureModel,
    loewe_fitness,
)

N_NODES, FEATURES, D = 50, 8, 16
torch.manual_seed(0)


def model_of(**fields) -> MixtureModel:
    cfg = MixtureConfig(
        name="check",
        readout="cell",
        compound_pool="bliss",
        cgt_dim=D,
        cgt_heads=4,
        hidden=32,
        dropout=0.0,
        dose="off",
        **fields,
    )
    model = MixtureModel(
        cfg, n_nodes=N_NODES, n_sources=1, feature_dim=FEATURES, encoder=None
    )
    model.eval()
    model.set_dose_scale(0.8)
    return model


def medium_of(k: int, n: int = 3, dose: float | None = None) -> Medium:
    x = torch.randn(n, k, FEATURES)
    z = torch.full((n, k), 0.0) if dose is None else torch.full((n, k), dose)
    return Medium(k=k, rows=list(range(n)), x=x, dose=z)


def host_read(model: MixtureModel, h: torch.Tensor, medium: Medium) -> torch.Tensor:
    e = model.tokens(medium)
    gain = model.gain(medium)
    _, pooled, token = model.medium_state(h, e, None, gain)
    return model.host_logits(pooled, token, e, gain, medium.dose)


h = torch.randn(N_NODES, D)
ok = []

# ---- 1. loewe_fitness solves its own equation ------------------------------- #
z = torch.tensor([[0.3, -0.2, 0.1]])
z50 = torch.tensor([[[0.0, 0.5, -0.5]]])
slope = torch.tensor([[[2.0, 1.5, 3.0]]])
scale = torch.tensor(1.84)
fit = loewe_fitness(z, z50, slope, scale)
logit = torch.log((1 - fit) / fit)
ratio = torch.exp(scale * (z - z50 - logit.unsqueeze(-1) / slope)).sum(-1)
ok.append(("loewe residual sum of dose ratios is 1", abs(float(ratio) - 1.0) < 1e-3))

# two identical curves at the same dose: Loewe is that curve at twice the dose
z50 = torch.tensor([[[0.0, 0.0]]])
slope = torch.tensor([[[2.0, 2.0]]])
z_one = torch.tensor([[0.2, 0.2]])
pair = float(loewe_fitness(z_one, z50, slope, scale))
double = float(torch.sigmoid(torch.tensor(-2.0 * (0.2 + np.log10(2.0) / 0.8))))
ok.append(
    (
        f"identical curves = single at 2x dose ({pair:.4f} vs {double:.4f})",
        abs(pair - double) < 2e-3,
    )
)

# Loewe never reads fitter than the highest single agent of the same curves
z50 = torch.randn(1, 4, 3) * 0.4
slope = torch.rand(1, 4, 3) * 2 + 1
z_any = torch.randn(4, 3) * 0.5
singles = torch.sigmoid(-slope * (z_any.unsqueeze(0) - z50))
lo = loewe_fitness(z_any, z50, slope, scale)
ok.append(
    ("loewe <= highest single agent", bool((lo <= singles.min(-1).values + 1e-3).all()))
)

# ---- 2. host_form: hill is a sigmoid in dose, one curve per compound -------- #
hill = model_of(host_form="hill")
low = float(host_read(hill, h, medium_of(1, 1, dose=-2.0))[0])
high = float(host_read(hill, h, medium_of(1, 1, dose=2.0))[0])
ok.append((f"hill falls with dose ({low:.3f} at low, {high:.3f} at high)", low > high))
ok.append(
    (
        "hill at the mean dose is one half",
        abs(float(host_read(hill, h, medium_of(1, 1, dose=0.0))[0]) - 0.5) < 1e-3,
    )
)
# the compound-free medium reads fitness one
free = host_read(
    hill,
    h,
    Medium(k=0, rows=[0], x=torch.zeros(1, 0, FEATURES), dose=torch.zeros(1, 0)),
)
ok.append(
    ("hill reads the compound-free medium at one", abs(float(free[0]) - 1.0) < 1e-6)
)

# ---- 3. compose: the three rules order as the pharmacology says ------------- #
torch.manual_seed(3)
shared = medium_of(3, 4, dose=0.4)
reads = {}
for rule in ("bliss", "loewe", "hsa"):
    m = model_of(host_form="hill", compose=rule)
    m.load_state_dict(hill.state_dict(), strict=False)
    reads[rule] = host_read(m, h, shared).detach().numpy()
ok.append(
    (
        f"hsa >= bliss ({reads['hsa'].mean():.3f} vs {reads['bliss'].mean():.3f})",
        bool((reads["hsa"] >= reads["bliss"] - 1e-6).all()),
    )
)
ok.append(
    (
        f"hsa >= loewe ({reads['hsa'].mean():.3f} vs {reads['loewe'].mean():.3f})",
        bool((reads["hsa"] >= reads["loewe"] - 1e-3).all()),
    )
)

# at one compound every rule is the same reading
one = medium_of(1, 3, dose=0.2)
single = [
    host_read(model_of(host_form="hill", compose=r), h, one)
    for r in ("bliss", "loewe", "hsa")
]
ok.append(
    (
        "the three rules agree at one compound",
        all(torch.allclose(single[0], s, atol=1e-3) for s in single[1:]),
    )
)

# ---- 4. deviation: pair is zero at the start, and only at k > 1 ------------- #
plain = model_of()
dev = model_of(deviation="pair")
dev.load_state_dict(plain.state_dict(), strict=False)
for k in (1, 2, 3):
    m = medium_of(k, 3, dose=0.1)
    ok.append(
        (
            f"deviation is zero at the start, k={k}",
            torch.allclose(host_read(plain, h, m), host_read(dev, h, m), atol=1e-5),
        )
    )
with torch.no_grad():
    dev.deviation[-1].weight.fill_(0.05)
moved = (
    (
        host_read(dev, h, medium_of(2, 3, dose=0.1))
        - host_read(plain, h, medium_of(2, 3, dose=0.1))
    )
    .abs()
    .max()
)
ok.append(
    (f"a trained deviation moves a pair ({float(moved):.4f})", float(moved) > 1e-4)
)

# ---- 5. env_mode: displace starts as the compound-free cell ---------------- #
disp = model_of(env_mode="displace")
e = disp.tokens(medium_of(2, 3))
_, pooled, _ = disp.medium_state(h, e, None, None)
ok.append(
    (
        "displace starts at the wildtype gene mean",
        torch.allclose(pooled, h.mean(0).expand_as(pooled), atol=1e-5),
    )
)
ok.append(("displace builds no medium transformer", len(disp.layers) == 0))
with torch.no_grad():
    disp.displace[-1].weight.fill_(0.1)
_, pooled_moved, _ = disp.medium_state(h, e, None, None)
ok.append(
    (
        "a trained displacement moves the cell",
        float((pooled_moved - pooled).abs().max()) > 1e-4,
    )
)
# the deleted-gene rows are read from the displaced cell
strain = torch.tensor([[1, 2], [3, 4]])
env, _, _ = disp.medium_state(h, e, strain, None)
ok.append(
    (
        "displace returns the perturbed rows",
        env is not None and tuple(env.shape) == (2, 6, D),
    )
)

# ---- 6. cell_only zeroes the token block and the offset -------------------- #
co = model_of(cell_only=True, env_mode="displace")
m = medium_of(1, 3, dose=0.3)
e = co.tokens(m)
_, pooled, token = co.medium_state(h, e, None, None)
with torch.no_grad():
    direct = co.host_logits(pooled, torch.randn_like(token), e, None, m.dose)
    masked = co.host_logits(pooled, token, e, None, m.dose)
ok.append(
    (
        "cell_only ignores the compound token block",
        torch.allclose(direct, masked, atol=1e-5),
    )
)

# ---- 7. the defaults are unchanged ----------------------------------------- #
base = MixtureConfig(name="base")
ok.append(
    (
        "the new fields default to the round-10 model",
        base.env_mode == "attend"
        and base.host_form == "head"
        and base.compose == "bliss"
        and base.deviation == "off"
        and base.cell_only is False,
    )
)

width = max(len(name) for name, _ in ok)
for name, passed in ok:
    print(f"{'PASS' if passed else 'FAIL'}  {name:<{width}}")
print(f"\n{sum(p for _, p in ok)} of {len(ok)} checks passed")
sys.exit(0 if all(p for _, p in ok) else 1)
