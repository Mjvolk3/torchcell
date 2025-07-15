---
id: clp031ghpb1q34vxm34sur6
title: 183123 Torchcell Basic Supervised Formulation
desc: ''
updated: 1751931146182
created: 1751931103468
---
$$
\hat{\theta}=\arg \min _\theta \mathbb{E}_{(\bar{G}, \bar{E}, \bar{P}, y) \sim D}\left[\mathcal{L}\left(\hat{f}_\theta(\tilde{G}, \tilde{E}, \tilde{P}), y\right)\right]
$$

- $\tilde{G}$ = cellular graph (genome structure with gene networks)
- $\tilde{E}$ = environment (growth conditions, media)
- $\tilde{P}$ = perturbation operator (gene deletions/modifications)