---
id: bz84sm0ua8aiclkafohceu0
title: '133148'
desc: ''
updated: 1736365287261
created: 1736364711618
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
