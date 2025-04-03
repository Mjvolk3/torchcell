---
id: ui5ybzmags4oraov2bobc3r
title: '02'
desc: ''
updated: 1743640164280
created: 1743634873634
---

## Model

## Objective

## Dataset

## Issues

##

## Alternative Idea 1

<!-- [[150252 Stoich Null Space|dendron://torchcell/scratch.2025.03.27.150252-stoich-null-space]] -->

[Brief other](#brief-of-other-attempts)

[url](https://chatgpt.com/c/67edc350-ed30-8002-bd5f-7bbc8a354cbd)

Here is some text @corsoGraphNeuralNetworks2024

Idea:

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

## Brief of Other Attempts

If you think that these directions are interesting I can will expand on them. For brevity I just highlight my findings.

## References
