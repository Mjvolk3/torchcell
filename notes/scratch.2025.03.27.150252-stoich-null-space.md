---
id: cc17yltnesylskxwtco3r79
title: 150252 Stoich Null Space
desc: ''
updated: 1743107823694
created: 1743105787694
---

$$
\begin{aligned}
\mathbf{v} &= \text{MLP}(\mathbf{z_e}) &&\text{(Reaction embeddings to flux vector)}\\[6pt]
f &= \mathbf{w}^\top \mathbf{v} &&\text{(Extract maximal growth rate from flux vector)}\\[6pt]
\mathcal{L}_{\text{null}} &= \|S \mathbf{v}\|_2^2 &&\text{(Null-space constraint loss)}\\[6pt]
\mathcal{L}_{\text{bound}} &= \|\text{ReLU}(\mathbf{v}_{lb} \odot \boldsymbol{\Theta} - \mathbf{v})\|_2^2 + \|\text{ReLU}(\mathbf{v} - \mathbf{v}_{ub} \odot \boldsymbol{\Theta})\|_2^2 &&\text{(Flux bounds constraint loss)}\\[6pt]
\mathcal{L}_{\text{pFBA}} &= \|\mathbf{v}\|_1 &&\text{(Parsimonious flux balance objective)} \\[8pt]
\mathcal{L}_{\text{total}} &= \mathcal{L}_{\text{pFBA}} + \lambda_{\text{null}}\mathcal{L}_{\text{null}} + \lambda_{\text{bound}}\mathcal{L}_{\text{bound}} &&\text{(Total PINN loss)}
\end{aligned}
$$

- **$\mathbf{z_e}$** is the reaction embedding vector mapped via MLP to the flux vector **$\mathbf{v}$**.
- **$\mathbf{w}$** is binary vector indicating the biomass pseudoreaction.
- **$f$** is maximal achievable growth rate under given constraints.
- **$S$** is the stoichiometric matrix.
- **$\mathbf{v}_{lb}$**, **$\mathbf{v}_{ub}$** are flux bounds, scaled by the transcriptional condition **$\boldsymbol{\Theta}$**, derived from gene expression through GPR logic.

***

Here's your markdown-formatted LaTeX formulation without the constraint bounds:

$$
\begin{aligned}
\mathbf{v} &= \text{MLP}(\mathbf{z_e}) &&\text{(Reaction embeddings to flux vector)}\\[6pt]
f &= \mathbf{w}^\top \mathbf{v} &&\text{(Extract maximal growth rate from flux vector)}\\[6pt]
\mathcal{L}_{\text{null}} &= \|S \mathbf{v}\|_2^2 &&\text{(Null-space constraint loss)}\\[6pt]
\mathcal{L}_{\text{pFBA}} &= \|\mathbf{v}\|_1 &&\text{(Parsimonious flux balance objective)} \\[8pt]
\mathcal{L}_{\text{total}} &= \mathcal{L}_{\text{pFBA}} + \lambda_{\text{null}}\mathcal{L}_{\text{null}} &&\text{(Total PINN loss)}
\end{aligned}
$$

- **$\mathbf{z_e}$** is the reaction embedding vector mapped via MLP to the flux vector **$\mathbf{v}$**.
- **$\mathbf{w}$** is a binary vector indicating the biomass pseudoreaction.
- **$f$** is the maximal achievable growth rate under given constraints.
- **$S$** is the stoichiometric matrix enforcing the null-space constraint.

***

Here's the revised markdown-formatted LaTeX formulation incorporating your fitness constraint:

$$
\begin{aligned}
\mathbf{v} &= \text{MLP}(\mathbf{z_e}) &&\text{(Reaction embeddings to flux vector)}\\[6pt]
\text{fitness} &= \mathbf{w}^\top \mathbf{v} &&\text{(Predicted fitness from flux vector)}\\[6pt]
\mathcal{L}_{\text{null}} &= \|S \mathbf{v}\|_2^2 &&\text{(Null-space constraint loss)}\\[6pt]
\mathcal{L}_{\text{pFBA}} &= \|\mathbf{v}\|_1 &&\text{(Parsimonious flux balance objective)} \\[6pt]
\mathcal{L}_{\text{fitness}} &= (\mathbf{w}^\top \mathbf{v} - \text{fitness})^2 &&\text{(Fitness constraint loss)} \\[8pt]
\mathcal{L}_{\text{total}} &= \mathcal{L}_{\text{pFBA}} + \lambda_{\text{null}}\mathcal{L}_{\text{null}} + \lambda_{\text{fitness}}\mathcal{L}_{\text{fitness}} &&\text{(Total PINN loss)}
\end{aligned}
$$

- **$\mathbf{z_e}$** is the reaction embedding vector mapped via MLP to the flux vector **$\mathbf{v}$**.
- **$\mathbf{w}$** is a binary vector indicating the biomass pseudoreaction.
- **$\text{fitness}$** is the experimentally measured fitness (growth ratio mutant/wildtype).
- **$S$** is the stoichiometric matrix enforcing the null-space constraint.
