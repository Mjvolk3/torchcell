---
id: rpl5rfrpn06srv9krjx6wgj
title: Flux_sampling_demo
desc: ''
updated: 1788314360622
created: 1788314360622
---

## 2026.09.01 - Amortized flux sampling

Draws a flux distribution per genotype in one forward pass, replacing the per-genotype MCMC
of classical flux sampling. The distribution goes on a LATENT and is pushed through the same
box map, so every draw is feasible with no rejection step and the reaction coordinates stay
coupled through Sv = 0.

The trap it avoids: a per-reaction marginal is not a distribution over flux vectors.
Merzbacher hit the same wall from the other side, reporting that their deep models failed
"attributed to the fluxes being linearly correlated through Sv = 0".

Full write-up: [[experiments.026-metabolism-flux.enzyme-constrained-thermodynamic-flux-layer]]
