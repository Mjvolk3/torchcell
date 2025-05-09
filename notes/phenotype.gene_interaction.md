---
id: vrhjq0r8uruk1isu16mack9
title: Gene_interaction
desc: ''
updated: 1744955518246
created: 1738036266319
---
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

**Simplified**

$$
\begin{aligned}
\tau_{ijk} &= f_{ijk} - f_i f_j f_k - (f_{ij} - f_i f_j) f_k - (f_{ik} - f_i f_k) f_j - (f_{jk} - f_j f_k) f_i \\
&= f_{ijk} - f_i f_j f_k - f_{ij}f_k + f_i f_j f_k - f_{ik}f_j + f_i f_k f_j - f_{jk}f_i + f_j f_k f_i \\
\end{aligned}
$$

Notice that $f_i f_j f_k$ appears once with negative sign, and then three more times with positive signs. Also, $f_j f_k f_i = f_i f_j f_k$. Simplifying:

$$
\begin{aligned}
\tau_{ijk} &= f_{ijk} - f_i f_j f_k - f_{ij}f_k + f_i f_j f_k - f_{ik}f_j + f_i f_j f_k - f_{jk}f_i + f_i f_j f_k \\
&= f_{ijk} - f_{ij}f_k - f_{ik}f_j - f_{jk}f_i + 3f_i f_j f_k - f_i f_j f_k \\
&= f_{ijk} - f_{ij}f_k - f_{ik}f_j - f_{jk}f_i + 2f_i f_j f_k \\
\end{aligned}
$$

Therefore, the simplified trigenic interaction equation is:

$$\tau_{ijk} = f_{ijk} - f_{ij}f_k - f_{ik}f_j - f_{jk}f_i + 2f_i f_j f_k$$

To generalize the interaction equations to higher-order interactions (4-way, 5-way, etc.), I'll develop a recursive formula using set notation.

Let $S$ be a set of elements (genes in this context), and for any subset $T \subseteq S$, let $f_T$ represent the fitness of that subset.

For any set $S$, the interaction term $\eta_S$ can be defined recursively as:

$$\eta_S = f_S - \sum_{T \subset S, T \neq \emptyset} \eta_T \prod_{i \in S \setminus T} f_i$$

Where:

- $\eta_{\{i\}} = f_i$ for singleton sets (single genes)
- $S \setminus T$ represents the set difference (elements in $S$ but not in $T$)

This recursive definition gives us:

For digenic (2-way): $\eta_{\{i,j\}} = f_{ij} - f_i f_j$ (which is your $\epsilon_{ij}$)

For trigenic (3-way): $\eta_{\{i,j,k\}} = f_{ijk} - f_{ij}f_k - f_{ik}f_j - f_{jk}f_i + 2f_i f_j f_k$ (which is your $\tau_{ijk}$)

For 4-way interactions, the formula would expand to:
$$\eta_{\{i,j,k,l\}} = f_{ijkl} - \sum_{|T|=3} \eta_T f_{S \setminus T} - \sum_{|T|=2} \eta_T \prod_{m \in S \setminus T} f_m - \sum_{|T|=1} \eta_T \prod_{m \in S \setminus T} f_m$$

This pattern continues for higher-order interactions, with the inclusion-exclusion principle determining the signs of the terms.
