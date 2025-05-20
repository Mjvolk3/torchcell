---
id: ay2rrmwfoe89y0u5er1b01n
title: 140200 Torchcell Main Model Setup
desc: ''
updated: 1747422361607
created: 1747422134012
---
$$\hat f_\theta: \widetilde{\mathcal G} \times \widetilde{\mathcal E} \times \widetilde{\mathcal P} \rightarrow \mathcal Y$$

$$\hat\theta = \arg\min_{\theta} \mathbb{E}_{(\tilde G,\tilde E,\tilde P,y)\sim D} \left[ \mathcal L\left(\hat f_\theta(\tilde G,\tilde E,\tilde P), y\right) \right]$$

**Where:**

- $\widetilde{\mathcal G}$: cellular graphs with vertex/edge features  
- $\widetilde{\mathcal E}$: real-valued environment vectors  
- $\widetilde{\mathcal P}$: perturbation operators  
- $\mathcal Y$: phenotype space  
- $y \in \mathcal Y$: observed phenotype  
- $D$: data distribution over $(\tilde G,\tilde E,\tilde P,y)$  
- $\mathcal L$: loss function  
- $\theta$: learnable parameters  
