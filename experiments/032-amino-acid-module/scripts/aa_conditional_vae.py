# experiments/032-amino-acid-module/scripts/aa_conditional_vae.py
# [[experiments.032-amino-acid-module.scripts.aa_conditional_vae]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/032-amino-acid-module/scripts/aa_conditional_vae

r"""How autoencodable is the amino-acid profile from gene embeddings and graphs: a conditional VAE.

The question (2026-09-29): encode a strain's measured profile into a latent conditioned
on its genotype, decode the profile, and put a betaxanthin head on the latent. If the
latent can be reached from the genotype alone, a new strain's profile and betaxanthin
could be decoded, or sampled, from its deletion. The model has two routes to the latent
and the answer is the distance between them:

- the POSTERIOR route $q(z \mid y, x)$ sees the measured profile $y$ and the genotype
  features $x$; it is what training uses, and on a held-out strain it is a
  reconstruction ceiling that needs the measurement;
- the PRIOR route $p(z \mid x)$ sees the genotype alone; it is the only route a strain
  nobody measured has, so it is the inductive read.

Training strains are the trainer's train split; pinned genes of the betaxanthin study
(``merzbacher_nested_split.json``) are held out of every fit; a validation fifth of the
training strains selects the epoch and the grid cell on the prior-route mean Spearman.
Loss: Gaussian reconstruction of the standardized $\log(1+\text{mM})$ profile from
$(z, h_x)$, $\beta$ times the KL from the posterior to the conditional prior, and
$\lambda$ times the squared error of a linear betaxanthin head on the posterior latent
(the discriminator; the Cachera score, ``SM_gem_bxscore``, standardized, present on the
strains the pigment screen measured). Genotype features $x$ of the deleted gene: the
propagated GO multi-hot (``SM_gem_go``), its one-hop neighbor mean over the graphs
(``SM_aa_gonbr``, inductive, no label), ProtT5, and the three together, each
standardized on the training strains and embedded by a two-layer MLP into $h_x$.

Reported on the pinned strains, per grid cell: posterior-route reconstruction (mean
Spearman over the 19 amino acids and pooled $R^2$), prior-route prediction (the same
two, the study's statistic), prior-route betaxanthin Spearman from the head, calibration
of 64 prior samples (the fraction of true values inside the central 90 percent of the
decoded samples, 0.90 when calibrated), and beside them a plain MLP from $x$ to the
profile and from $x$ to betaxanthin with no latent. Writes
``results/aa_conditional_vae.json``. Run under slurm (``gh_aa_conditional_vae.slurm``)::

    PYTHONHASHSEED=0 PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/032-amino-acid-module/scripts/aa_conditional_vae.py
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from pydantic import BaseModel
from scipy.stats import spearmanr
from torch import nn

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "scripts"))
from linear_baselines_bx import embedding_matrix  # noqa: E402

from torchcell.graph import SCerevisiaeGraph  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402

TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
SPLIT = osp.join(EXPERIMENT_ROOT, "020-cachera-betaxanthin", "results", "merzbacher_nested_split.json")
WIDTH = 128
EPOCHS = 300
LR = 1e-3
BATCH = 128
SEEDS = (0, 1, 2)
N_SAMPLES = 64
FEATURES = ("go", "gonbr", "t5", "go+gonbr+t5")
GRID = [
    # (latent, beta, lambda_bx)
    (4, 1.0, 0.0),
    (8, 1.0, 0.0),
    (8, 0.1, 0.0),
    (8, 1.0, 1.0),
    (8, 0.1, 1.0),
]


class CellRead(BaseModel):
    features: str
    latent: int
    beta: float
    lambda_bx: float
    seed: int
    select_epoch: int
    val_prior_mean_spearman: float
    posterior_mean_spearman: float
    posterior_r2: float
    prior_mean_spearman: float
    prior_r2: float
    prior_bx_spearman: float | None
    posterior_bx_spearman: float | None
    calibration_90: float
    kl_per_strain: float


class BaselineRead(BaseModel):
    features: str
    seed: int
    select_epoch: int
    mlp_profile_mean_spearman: float
    mlp_profile_r2: float
    mlp_bx_spearman: float | None


def load_store(name: str) -> dict[str, np.ndarray]:
    z = np.load(osp.join(TABLE_DIR, f"ess_solve_encodings_{name}.npz"), allow_pickle=False)
    return {str(g): v.astype(np.float32) for g, v in zip(z["genes"], z["v_deletion"], strict=True)}


def mean_spearman(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean([spearmanr(pred[:, j], true[:, j])[0] for j in range(true.shape[1])]))


def r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = ((pred - true) ** 2).sum()
    ss_tot = ((true - true.mean(axis=0)) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def mlp(d_in: int, d_out: int, width: int = WIDTH) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, d_out))


class CVAE(nn.Module):
    def __init__(self, d_x: int, d_y: int, k: int) -> None:
        super().__init__()
        self.hx = mlp(d_x, WIDTH)
        self.enc = mlp(d_y + WIDTH, 2 * k)
        self.prior = mlp(WIDTH, 2 * k)
        self.dec = mlp(k + WIDTH, d_y)
        self.bx = nn.Linear(k, 1)
        self.k = k

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        h = F.relu(self.hx(x))
        mu_q, lv_q = self.enc(torch.cat([y, h], 1)).chunk(2, 1)
        mu_p, lv_p = self.prior(h).chunk(2, 1)
        z = mu_q + torch.randn_like(mu_q) * torch.exp(0.5 * lv_q)
        y_hat = self.dec(torch.cat([z, h], 1))
        kl = 0.5 * (lv_p - lv_q + (torch.exp(lv_q) + (mu_q - mu_p) ** 2) / torch.exp(lv_p) - 1).sum(1)
        return y_hat, kl, mu_q, mu_p, lv_p, h

    def decode(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.dec(torch.cat([z, h], 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    assert os.environ.get("PYTHONHASHSEED") == "0"
    torch.set_num_threads(args.threads)
    started = time.time()

    pool = load_store("YPD_bx_aapool")
    bx = {g: float(v[0]) for g, v in load_store("SM_gem_bxscore").items()}
    go = load_store("SM_gem_go")
    gonbr = load_store("SM_aa_gonbr")
    genome = SCerevisiaeGenome(genome_root=osp.join(DATA_ROOT, "data/sgd/genome"), go_root=osp.join(DATA_ROOT, "data/go"), overwrite=False)
    graph = SCerevisiaeGraph(sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"), string_root=osp.join(DATA_ROOT, "data/string"), tflink_root=osp.join(DATA_ROOT, "data/tflink"), genome=genome)
    t5 = embedding_matrix(("prot_T5_all",), DATA_ROOT, genome, graph)
    feats = {"go": go, "gonbr": gonbr, "t5": t5}
    genes = sorted(g for g in pool if all(g in f for f in feats.values()))
    with open(SPLIT) as fh:
        pinned = set(json.load(fh)["split"]["test"])
    test = np.array([g in pinned for g in genes])
    rng = np.random.default_rng(0)
    train_idx = np.flatnonzero(~test)
    val_idx = np.sort(rng.choice(train_idx, size=len(train_idx) // 5, replace=False))
    fit_idx = np.setdiff1d(train_idx, val_idx)
    test_idx = np.flatnonzero(test)
    y = np.log1p(np.stack([pool[g] for g in genes]))
    mu, sd = y[fit_idx].mean(0), y[fit_idx].std(0)
    y = ((y - mu) / sd).astype(np.float32)
    b = np.array([bx.get(g, np.nan) for g in genes], dtype=np.float32)
    has_b = np.isfinite(b)
    b_mu, b_sd = np.nanmean(b[fit_idx]), np.nanstd(b[fit_idx])
    b_z = np.where(has_b, (b - b_mu) / b_sd, 0.0).astype(np.float32)
    print(f"  {len(genes)} strains: fit {len(fit_idx)}, val {len(val_idx)}, pinned {len(test_idx)}; betaxanthin score on {int(has_b.sum())}", flush=True)

    def feature_matrix(name: str) -> np.ndarray:
        parts = [feats[n] for n in name.split("+")]
        x = np.concatenate([np.stack([p[g] for g in genes]) for p in parts], axis=1).astype(np.float32)
        if "gonbr" in name:
            pass
        xm, xs = x[fit_idx].mean(0), x[fit_idx].std(0)
        xs[xs == 0] = 1.0
        return ((x - xm) / xs).astype(np.float32)

    yt = torch.from_numpy(y)
    bt = torch.from_numpy(b_z)
    bm = torch.from_numpy(has_b)
    cells: list[CellRead] = []
    baselines: list[BaselineRead] = []
    for fname in FEATURES:
        x = feature_matrix(fname)
        xt = torch.from_numpy(x)
        # ---- plain MLP baselines, no latent ---------------------------------------------
        for seed in SEEDS:
            torch.manual_seed(seed)
            net_y, net_b = mlp(x.shape[1], y.shape[1]), mlp(x.shape[1], 1)
            opt = torch.optim.Adam(list(net_y.parameters()) + list(net_b.parameters()), lr=LR, weight_decay=1e-4)
            best, best_state, best_ep = -1.0, None, 0
            for ep in range(args.epochs):
                net_y.train(); net_b.train()
                perm = torch.from_numpy(rng.permutation(fit_idx))
                for i in range(0, len(perm), BATCH):
                    idx = perm[i : i + BATCH]
                    opt.zero_grad()
                    loss = F.mse_loss(net_y(xt[idx]), yt[idx])
                    mb = bm[idx]
                    if mb.any():
                        loss = loss + F.mse_loss(net_b(xt[idx]).squeeze(1)[mb], bt[idx][mb])
                    loss.backward(); opt.step()
                net_y.eval()
                with torch.no_grad():
                    v = mean_spearman(net_y(xt[val_idx]).numpy(), y[val_idx])
                if v > best:
                    best, best_ep = v, ep
                    best_state = ({k: t.clone() for k, t in net_y.state_dict().items()}, {k: t.clone() for k, t in net_b.state_dict().items()})
            net_y.load_state_dict(best_state[0]); net_b.load_state_dict(best_state[1]); net_y.eval(); net_b.eval()
            with torch.no_grad():
                p = net_y(xt[test_idx]).numpy()
                pb = net_b(xt[test_idx]).squeeze(1).numpy()
            tb = has_b[test_idx]
            baselines.append(BaselineRead(features=fname, seed=seed, select_epoch=best_ep, mlp_profile_mean_spearman=mean_spearman(p, y[test_idx]), mlp_profile_r2=r2(p, y[test_idx]), mlp_bx_spearman=float(spearmanr(pb[tb], b[test_idx][tb])[0]) if tb.sum() > 3 else None))
            print(f"  MLP {fname:14s} seed {seed}: profile rho {baselines[-1].mlp_profile_mean_spearman:+.3f} R2 {baselines[-1].mlp_profile_r2:+.3f} bx rho {baselines[-1].mlp_bx_spearman}", flush=True)
        # ---- conditional VAE grid ---------------------------------------------------------
        for k, beta, lam in GRID:
            for seed in SEEDS:
                torch.manual_seed(seed)
                net = CVAE(x.shape[1], y.shape[1], k)
                opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
                best, best_state, best_ep = -1.0, None, 0
                for ep in range(args.epochs):
                    net.train()
                    perm = torch.from_numpy(rng.permutation(fit_idx))
                    for i in range(0, len(perm), BATCH):
                        idx = perm[i : i + BATCH]
                        opt.zero_grad()
                        y_hat, kl, mu_q, _, _, _ = net(xt[idx], yt[idx])
                        loss = F.mse_loss(y_hat, yt[idx]) + beta * kl.mean() / y.shape[1]
                        mb = bm[idx]
                        if lam > 0 and mb.any():
                            loss = loss + lam * F.mse_loss(net.bx(mu_q).squeeze(1)[mb], bt[idx][mb])
                        loss.backward(); opt.step()
                    net.eval()
                    with torch.no_grad():
                        _, _, _, mu_p, _, h = net(xt[val_idx], yt[val_idx])
                        v = mean_spearman(net.decode(mu_p, h).numpy(), y[val_idx])
                    if v > best:
                        best, best_ep = v, ep
                        best_state = {kk: t.clone() for kk, t in net.state_dict().items()}
                net.load_state_dict(best_state); net.eval()
                with torch.no_grad():
                    _, kl, mu_q, mu_p, lv_p, h = net(xt[test_idx], yt[test_idx])
                    post = net.decode(mu_q, h).numpy()
                    prior = net.decode(mu_p, h).numpy()
                    samples = np.stack([net.decode(mu_p + torch.randn_like(mu_p) * torch.exp(0.5 * lv_p), h).numpy() for _ in range(N_SAMPLES)])
                    bx_prior = net.bx(mu_p).squeeze(1).numpy()
                    bx_post = net.bx(mu_q).squeeze(1).numpy()
                lo, hi = np.quantile(samples, 0.05, axis=0), np.quantile(samples, 0.95, axis=0)
                cal = float(((y[test_idx] >= lo) & (y[test_idx] <= hi)).mean())
                tb = has_b[test_idx]
                read = CellRead(features=fname, latent=k, beta=beta, lambda_bx=lam, seed=seed, select_epoch=best_ep, val_prior_mean_spearman=best,
                                posterior_mean_spearman=mean_spearman(post, y[test_idx]), posterior_r2=r2(post, y[test_idx]),
                                prior_mean_spearman=mean_spearman(prior, y[test_idx]), prior_r2=r2(prior, y[test_idx]),
                                prior_bx_spearman=float(spearmanr(bx_prior[tb], b[test_idx][tb])[0]) if tb.sum() > 3 else None,
                                posterior_bx_spearman=float(spearmanr(bx_post[tb], b[test_idx][tb])[0]) if tb.sum() > 3 else None,
                                calibration_90=cal, kl_per_strain=float(kl.mean()))
                cells.append(read)
                print(f"  CVAE {fname:14s} k={k} beta={beta} lam={lam} seed {seed}: posterior rho {read.posterior_mean_spearman:+.3f} R2 {read.posterior_r2:+.3f} | prior rho {read.prior_mean_spearman:+.3f} R2 {read.prior_r2:+.3f} | bx prior {read.prior_bx_spearman} post {read.posterior_bx_spearman} | cal90 {cal:.2f} kl {read.kl_per_strain:.2f} ep {best_ep}", flush=True)
    out = osp.join(EXPERIMENT_ROOT, "032-amino-acid-module", "results", "aa_conditional_vae.json")
    with open(out, "w") as fh:
        json.dump({
            "generated_by": "experiments/032-amino-acid-module/scripts/aa_conditional_vae.py",
            "split": SPLIT, "n_strains": len(genes), "n_fit": int(len(fit_idx)), "n_val": int(len(val_idx)), "n_pinned": int(len(test_idx)),
            "n_pinned_with_betaxanthin": int(has_b[test_idx].sum()), "epochs": args.epochs, "seeds": list(SEEDS), "grid": GRID, "features": list(FEATURES),
            "statistic": ("pinned held-out mean Spearman over the 19 amino acids and pooled R2 of the decoded standardized log1p profile; posterior route decodes "
                          "from q(z|y,x) (needs the measurement), prior route from p(z|x) (genotype only); calibration_90 is the fraction of true values inside "
                          "the central 90 percent of 64 prior samples; epoch and cell selected on the validation prior-route mean Spearman"),
            "cells": [c.model_dump() for c in cells], "baselines": [b_.model_dump() for b_ in baselines], "seconds": time.time() - started,
        }, fh, indent=2)
    print(out)


if __name__ == "__main__":
    main()
