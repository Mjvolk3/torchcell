---
id: vrhjq0r8uruk1isu16mack9
title: Gene_interaction
desc: ''
updated: 1738551936531
created: 1738036266319
---
A convenient way to see the general rule is via "inclusion-exclusion" on the lower-order terms. Concretely, label your fitness (or response) for a subset of genes $S \subseteq\{1, \ldots, n\}$ by $f_S$. Then define the interaction $\epsilon_S$ for that subset by recursively subtracting off all interactions belonging to strict sub-subsets of $S$. In symbols,

$$
\epsilon_S=f_S-\sum_{\varnothing \neq T \subset S} \epsilon_T
$$

Here are the small cases to see the pattern:

- Singles: $\epsilon_{\{i\}}=f_{\{i\}}$.
- Pairs: $\epsilon_{\{i, j\}}=f_{\{i, j\}}-\epsilon_{\{i\}}-\epsilon_{\{j\}}=f_{i j}-f_i-f_j$.
- Triples:

$$
\epsilon_{\{i, j, k\}}=f_{i j k}-\left(\epsilon_{\{i\}}+\epsilon_{\{j\}}+\epsilon_{\{k\}}\right)-\left(\epsilon_{\{i, j\}}+\epsilon_{\{i, k\}}+\epsilon_{\{j, k\}}\right)
$$

and so forth.
Equivalently, one may write this as an inclusion-exclusion formula

$$
\epsilon_S=\sum_{T \subseteq S}(-1)^{|S|-|T|} f_T
$$

provided you set $f_{\varnothing}=0$. Either expression generalizes the notion that, to isolate a genuine $k$-way interaction, you must "peel off" all interactions from sub-collections off the same genes.

## 2025.01.27 - Derivation of Tau

The derivation for $\tau_{i j k}$, the triple interaction term, follows from the recursive definition of interactions. Let's walk through the steps systematically.

We start with the general recursive formula:

$$
\tau_{i j k}=f_{i j k}-\sum_{S \subset\{i, j, k\},|S|=2} \tau_S-\sum_{T \subset\{i, j, k\},|T|=1} f_T
$$

Here's the breakdown step by step:

1. Subtract contributions from single elements ( $f_i, f_j, f_k$ ):

The single fitness terms contribute:

$$
\sum_{T \subset\{i, j, k\},|T|=1} f_T=f_i+f_j+f_k
$$

These are the individual effects of the single genes $i, j, k$.
2. Subtract contributions from pairs $\left(\tau_{i j}, \tau_{i k}, \tau_{j k}\right)$ :

The pairwise interaction terms contribute:

$$
\sum_{S \subset\{i, j, k\},|S|=2} \tau_S=\tau_{i j}+\tau_{i k}+\tau_{j k}
$$

Each of these pairwise terms has already been recursively defined as:

$$
\begin{aligned}
\tau_{i j} & =f_{i j}-f_i-f_j \\
\tau_{i k} & =f_{i k}-f_i-f_k \\
\tau_{j k} & =f_{j k}-f_j-f_k
\end{aligned}
$$

3. Combine terms:

The triple interaction term $\tau_{i j k}$ is then:

$$
\tau_{i j k}=f_{i j k}-\left(f_{i j}-f_i-f_j\right)-\left(f_{i k}-f_i-f_k\right)-\left(f_{j k}-f_j-f_k\right)-f_i-f_j-f_k
$$

Simplify step by step:

- Combine the single terms $f_i, f_j, f_k$ :

$$
\tau_{i j k}=f_{i j k}-f_{i j}-f_{i k}-f_{j k}+f_i+f_j+f_k
$$

Final Form:

$$
\tau_{i j k}=f_{i j k}-f_{i j}-f_{i k}-f_{j k}+f_i+f_j+f_k
$$

This is the full expanded form of the triple interaction term $\tau_{i j k}$. It subtracts all pairwise and single contributions to isolate the unique interaction between $i, j, k$.

## 2025.02.02 - Gene Interaction Not Regressive Form

**Digenic Interaction**:

$$
\epsilon_{ij} = f_{ij} - f_i f_j, \quad \text{where } f_i \text{ is fitness,}
$$

$$
\epsilon_{ij} = \mathcal{F}(f_i, f_j, f_{ij}).
$$

**Trigenic Interaction**:

$$
\tau_{ijk} = f_{ijk} - f_i f_j f_k - \varepsilon_{ij} f_k - \varepsilon_{ik} f_j - \varepsilon_{jk} f_i.
$$

**Full Recursive Definition** (subtracting out effects of digenic interactions):

$$
\begin{aligned}
\tau_{ijk} &= f_{ijk} - f_i f_j f_k \\
&\quad - \left(f_{ij} - f_i f_j\right) f_k \\
&\quad - \left(f_{ik} - f_i f_k\right) f_j \\
&\quad - \left(f_{jk} - f_j f_k\right) f_i, \\
\tau_{ijk} &= \mathcal{F}\left(f_i, f_j, f_k, f_{ij}, f_{ik}, f_{jk}, f_{ijk}\right).
\end{aligned}
$$

***

**Fitness**

$$
f_i = \mathcal{F}(g_i, g_{wt})
$$

**Digenic Interaction**:

$$
\epsilon_{ij} = \mathcal{F}(f_i, f_j, f_{ij})
$$

**Trigenic Interaction**:

$$
\tau_{ijk} = \mathcal{F}\left(f_i, f_j, f_k, f_{ij}, f_{ik}, f_{jk}, f_{ijk}\right)
$$
