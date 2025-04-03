---
id: vrhjq0r8uruk1isu16mack9
title: Gene_interaction
desc: ''
updated: 1743649163250
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

## 2025.04.02 - Gene Interaction in Terms of Fitness Only

I'll derive the expression for interactions (epistasis) $\epsilon_S$ in terms of fitness values $f_S$ using the inclusion-exclusion principle.

Starting with the recursive definition:
$$\epsilon_S = f_S - \sum_{\emptyset \neq T \subset S} \epsilon_T$$

Let's solve this by induction to arrive at the inclusion-exclusion formula.

For $|S| = 1$, say $S = \{i\}$:
$$\epsilon_{\{i\}} = f_{\{i\}} - \sum_{\emptyset \neq T \subset \{i\}} \epsilon_T = f_{\{i\}}$$
since there are no non-empty strict subsets of a singleton.

For $|S| = 2$, say $S = \{i,j\}$:
$$\epsilon_{\{i,j\}} = f_{\{i,j\}} - \epsilon_{\{i\}} - \epsilon_{\{j\}} = f_{\{i,j\}} - f_{\{i\}} - f_{\{j\}}$$

For $|S| = 3$, say $S = \{i,j,k\}$:
$$\begin{align}
\epsilon_{\{i,j,k\}} &= f_{\{i,j,k\}} - \epsilon_{\{i\}} - \epsilon_{\{j\}} - \epsilon_{\{k\}} - \epsilon_{\{i,j\}} - \epsilon_{\{i,k\}} - \epsilon_{\{j,k\}} \\
&= f_{\{i,j,k\}} - f_{\{i\}} - f_{\{j\}} - f_{\{k\}} - (f_{\{i,j\}} - f_{\{i\}} - f_{\{j\}}) - (f_{\{i,k\}} - f_{\{i\}} - f_{\{k\}}) - (f_{\{j,k\}} - f_{\{j\}} - f_{\{k\}}) \\
&= f_{\{i,j,k\}} - f_{\{i\}} - f_{\{j\}} - f_{\{k\}} - f_{\{i,j\}} + f_{\{i\}} + f_{\{j\}} - f_{\{i,k\}} + f_{\{i\}} + f_{\{k\}} - f_{\{j,k\}} + f_{\{j\}} + f_{\{k\}} \\
&= f_{\{i,j,k\}} - f_{\{i,j\}} - f_{\{i,k\}} - f_{\{j,k\}} + f_{\{i\}} + f_{\{j\}} + f_{\{k\}}
\end{align}$$

Looking at this pattern, we can derive the inclusion-exclusion formula:

$$\epsilon_S = \sum_{T \subseteq S} (-1)^{|S|-|T|} f_T$$

To prove this formula satisfies our recursive definition, let's substitute it back:

$$\begin{align}
f_S - \sum_{\emptyset \neq T \subset S} \epsilon_T &= f_S - \sum_{\emptyset \neq T \subset S} \left(\sum_{U \subseteq T} (-1)^{|T|-|U|} f_U\right) \\
&= f_S - \sum_{U \subseteq S, U \neq S} f_U \sum_{\{U \subseteq T \subset S\}} (-1)^{|T|-|U|}
\end{align}$$

For any fixed $U \subset S$, the inner sum $\sum_{\{U \subseteq T \subset S\}} (-1)^{|T|-|U|}$ equals 0 except when $U = S$, in which case it equals 1. This is a combinatorial identity that follows from the binomial theorem.

Therefore, the right side reduces to $f_S - \sum_{U \subset S} f_U \cdot 0 + f_S \cdot 1 = f_S$, confirming that our inclusion-exclusion formula is correct.

The formula $\epsilon_S = \sum_{T \subseteq S} (-1)^{|S|-|T|} f_T$ provides a direct way to compute interactions from fitness values without recursion.

***

I'll rewrite the derivation with simplified notation, removing set brackets for subscripts while clearly indicating these are set-based.

## 2025.04.02 - Gene Interaction in Terms of Fitness Only

I'll derive the expression for interactions (epistasis) $\epsilon_S$ in terms of fitness values $f_S$ using the inclusion-exclusion principle.

Starting with the recursive definition:
$$\epsilon_S = f_S - \sum_{\emptyset \neq T \subset S} \epsilon_T$$

Let's solve this by induction to arrive at the inclusion-exclusion formula.

For $|S| = 1$, say $S = i$:
$$\epsilon_i = f_i - \sum_{\emptyset \neq T \subset i} \epsilon_T = f_i$$
since there are no non-empty strict subsets of a singleton.

For $|S| = 2$, say $S = ij$:
$$\epsilon_{ij} = f_{ij} - \epsilon_i - \epsilon_j = f_{ij} - f_i - f_j$$

For $|S| = 3$, say $S = ijk$:
$$\begin{align}
\epsilon_{ijk} &= f_{ijk} - \epsilon_i - \epsilon_j - \epsilon_k - \epsilon_{ij} - \epsilon_{ik} - \epsilon_{jk} \\
&= f_{ijk} - f_i - f_j - f_k - (f_{ij} - f_i - f_j) - (f_{ik} - f_i - f_k) - (f_{jk} - f_j - f_k) \\
&= f_{ijk} - f_i - f_j - f_k - f_{ij} + f_i + f_j - f_{ik} + f_i + f_k - f_{jk} + f_j + f_k \\
&= f_{ijk} - f_{ij} - f_{ik} - f_{jk} + f_i + f_j + f_k
\end{align}$$

Looking at this pattern, we can derive the inclusion-exclusion formula:

$$\epsilon_S = \sum_{T \subseteq S} (-1)^{|S|-|T|} f_T$$

To prove this formula satisfies our recursive definition, let's substitute it back:

$$\begin{align}
f_S - \sum_{\emptyset \neq T \subset S} \epsilon_T &= f_S - \sum_{\emptyset \neq T \subset S} \left(\sum_{U \subseteq T} (-1)^{|T|-|U|} f_U\right) \\
&= f_S - \sum_{U \subseteq S, U \neq S} f_U \sum_{U \subseteq T \subset S} (-1)^{|T|-|U|}
\end{align}$$

For any fixed $U \subset S$, the inner sum $\sum_{U \subseteq T \subset S} (-1)^{|T|-|U|}$ equals 0 except when $U = S$, in which case it equals 1. This is a combinatorial identity that follows from the binomial theorem.

Therefore, the right side reduces to $f_S - \sum_{U \subset S} f_U \cdot 0 + f_S \cdot 1 = f_S$, confirming that our inclusion-exclusion formula is correct.

The formula $\epsilon_S = \sum_{T \subseteq S} (-1)^{|S|-|T|} f_T$ provides a direct way to compute interactions from fitness values without recursion.

***

# Derivation of Gene Interactions in Terms of Fitness Values

In epistasis analysis, we aim to quantify the interactions between genes by examining how combinations of genetic variants affect phenotypes beyond what would be expected from individual effects. Here, I present a formal derivation of gene interactions using the inclusion-exclusion principle.

## Definition and Notation

Let $f_S$ represent the fitness (or phenotypic response) for a subset of genes $S \subseteq \{1, \ldots, n\}$. The interaction term $\epsilon_S$ for that subset can be defined recursively by subtracting all interactions belonging to strict subsets of $S$:

$$\epsilon_S = f_S - \sum_{\emptyset \neq T \subset S} \epsilon_T$$

## Examples for Small Sets

### Single Gene ($|S| = 1$)
For a single gene $i$:
$$\epsilon_i = f_i$$
since there are no non-empty strict subsets.

### Gene Pairs ($|S| = 2$)
For a pair of genes $i$ and $j$:
$$\epsilon_{ij} = f_{ij} - \epsilon_i - \epsilon_j = f_{ij} - f_i - f_j$$

### Gene Triples ($|S| = 3$)
For three genes $i$, $j$, and $k$:
$$\begin{align}
\epsilon_{ijk} &= f_{ijk} - \epsilon_i - \epsilon_j - \epsilon_k - \epsilon_{ij} - \epsilon_{ik} - \epsilon_{jk} \\
&= f_{ijk} - f_i - f_j - f_k - (f_{ij} - f_i - f_j) - (f_{ik} - f_i - f_k) - (f_{jk} - f_j - f_k) \\
&= f_{ijk} - f_{ij} - f_{ik} - f_{jk} + f_i + f_j + f_k
\end{align}$$

## General Formula via Inclusion-Exclusion

Examining the pattern in these examples suggests a general formula using inclusion-exclusion:

$$\epsilon_S = \sum_{T \subseteq S} (-1)^{|S|-|T|} f_T$$

This formula directly computes the interaction term $\epsilon_S$ from fitness values without requiring recursive calculation.

## Proof of Equivalence

To verify this formula satisfies our recursive definition, we substitute it back:

$$\begin{align}
f_S - \sum_{\emptyset \neq T \subset S} \epsilon_T &= f_S - \sum_{\emptyset \neq T \subset S} \left(\sum_{U \subseteq T} (-1)^{|T|-|U|} f_U\right) \\
&= f_S - \sum_{U \subseteq S, U \neq S} f_U \sum_{U \subseteq T \subset S} (-1)^{|T|-|U|}
\end{align}$$

For any fixed $U \subset S$, the inner sum $\sum_{U \subseteq T \subset S} (-1)^{|T|-|U|}$ equals 0 except when $U = S$, which is excluded from the range. This follows from a combinatorial identity related to the binomial theorem.

Therefore, the right side reduces to $f_S$, confirming that our inclusion-exclusion formula is correct.

## Interpretation

The formula $\epsilon_S = \sum_{T \subseteq S} (-1)^{|S|-|T|} f_T$ isolates genuine $|S|$-way interactions by systematically "peeling off" all interactions from subsets of the same genes. This allows researchers to determine whether genes interact synergistically or antagonistically beyond what would be expected from their individual and lower-order combined effects.
