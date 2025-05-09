---
id: kuz3smsc6sr7hd598k2rjd9
title: Mechanisitc Aware Fba Inspired
desc: ''
updated: 1744955611708
created: 1744149743802
---
## 2025.04.08 - Possible Inclusions

Taken from here [[02|dendron://torchcell/experiments.003-fit-int.2025.04.02]].

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

***

**Claude**

## Implementing Bound Constraints in Mechanistic-Aware FBA

Continuing from the PINN formulation, we need to handle the bound constraints properly:

## Handling Flux Bounds as Loss

Since you've split reversible reactions, the bound constraints simplify to:

$$0 \leq \mathbf{v} \leq \mathbf{v_{ub}}$$

We can implement this as a penalty loss using a ReLU-based hinge formulation:

$$\mathcal{L}_{\text{bounds}} = \sum_i \text{ReLU}(-v_i) + \sum_i \text{ReLU}(v_i - v_{ub,i})$$

- First term penalizes violations of lower bound (0)
- Second term penalizes violations of upper bound

## Complete Loss Function

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pFBA}} + \lambda_{\text{null}}\mathcal{L}_{\text{null}} + \lambda_{\text{fitness}}\mathcal{L}_{\text{fitness}} + \lambda_{\text{bounds}}\mathcal{L}_{\text{bounds}}$$

## Implementing L1 Loss on Flux Vector

For L1 regularization on v (not weights):

```python
def l1_flux_loss(v):
    return torch.norm(v, p=1)  # L1 norm of flux vector
```

This directly implements the `min ||v||₁` objective from your pFBA formulation.

## Implementation Sketch

```python
def forward(self, reaction_embeddings):
    # Map embeddings to flux vector
    v = self.mlp(reaction_embeddings)
    
    # Compute losses
    l_null = torch.norm(torch.matmul(S, v), p=2)**2
    l_fitness = (torch.matmul(w.T, v) - target_fitness)**2
    l_pfba = torch.norm(v, p=1)
    l_bounds = torch.sum(F.relu(-v)) + torch.sum(F.relu(v - v_ub))
    
    # Total loss
    loss = l_pfba + lambda_null*l_null + lambda_fitness*l_fitness + lambda_bounds*l_bounds
    
    return v, loss
```

Since you're splitting reversible reactions into forward-only reactions, the implementation is cleaner as all fluxes have a lower bound of 0. This approach directly optimizes the original pFBA objective while incorporating the required constraints through soft penalties.

**Claude**

# Growth Equations in GEMs

Growth equations in genome-scale metabolic models (GEMs) are formulated as biomass objective functions that represent the synthesis of cellular components required for growth.

## Summary Table

| Component | Typical Representation | Role in Growth Equation |
|-----------|------------------------|-------------------------|
| Amino acids | Stoichiometric coefficients | Protein synthesis requirements |
| Nucleotides | Stoichiometric coefficients | DNA/RNA synthesis requirements |
| Lipids | Stoichiometric coefficients | Membrane components |
| Carbohydrates | Stoichiometric coefficients | Cell wall, storage molecules |
| Cofactors | Stoichiometric coefficients | Metabolic catalysts |
| ATP maintenance | ATP → ADP + Pi | Energy for non-growth processes |

## Growth Encoding

The growth equation encodes growth by:

1. Representing growth as a pseudo-reaction: $\sum_i s_i M_i \rightarrow \text{Biomass}$
   where $s_i$ is the stoichiometric coefficient of metabolite $M_i$

2. Serving as the objective function in FBA:
   $$\max Z = v_{\text{biomass}}$$
   subject to:
   $$S \cdot v = 0$$
   $$v_{\min} \leq v \leq v_{\max}$$

3. Reflecting cellular composition based on experimental measurements of macromolecular content

4. Including growth-associated maintenance energy (GAM) and non-growth-associated maintenance energy (NGAM) requirements

## 2025.04.10 - Combining Terms

$$
\begin{aligned}
\mathbf{v} &= \text{MLP}(\mathbf{z_e}) && \text{(Reaction embeddings to flux vector)} \\[6pt]
\text{fitness} &= \mathbf{w}^\top \mathbf{v} && \text{(Predicted fitness from flux vector)} \\[6pt]
\mathcal{L}_{\text{null}} &= \| S \mathbf{v} \|_2^2 && \text{(Null-space constraint loss)} \\[6pt]
\mathcal{L}_{\text{pFBA}} &= \| \mathbf{v} \|_1 && \text{(Parsimonious flux balance objective)} \\[6pt]
\mathcal{L}_{\text{fitness}} &= (\mathbf{w}^\top \mathbf{v} - \text{fitness})^2 && \text{(Fitness constraint loss)} \\[6pt]
\mathcal{L}_{\text{bounds}} &= \sum_i \text{ReLU}(-v_i) + \sum_i \text{ReLU}(v_i - v_{\text{ub},i}) && \text{(Flux bound constraint loss)} \\[8pt]
\mathcal{L}_{\text{total}} &= \mathcal{L}_{\text{pFBA}} + \lambda_{\text{null}} \mathcal{L}_{\text{null}} + \lambda_{\text{fitness}} \mathcal{L}_{\text{fitness}} + \lambda_{\text{bounds}} \mathcal{L}_{\text{bounds}} && \text{(Total PINN loss)}
\end{aligned}
$$

- $\mathbf{z_e}$ is the reaction embedding vector, mapped via MLP to the flux vector $\mathbf{v}$  
- $\mathbf{w}$ is a binary vector indicating the biomass pseudoreaction  
- $\text{fitness}$ is the experimentally measured fitness (growth ratio of mutant to wild type)  
- $S$ is the stoichiometric matrix enforcing the null-space constraint  
- $\mathbf{v}_{\text{ub}}$ is the upper bound vector for fluxes  
