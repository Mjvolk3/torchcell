---
id: mswlvlny1fqujz68493zk9d
title: Problem Formulation
desc: ''
updated: 1739316741723
created: 1739316734799
---
$$
\mathcal{X} \in \mathbb{R}^{N \times d}
$$

$$
\mathcal{X} \in \mathbb{R}^{N-n \times d}
$$

$$
\begin{align}
\hat{y} &\in \mathbb{R}^{N \times 2} \quad \text{(Predicted Labels)}\\
\mathcal{Z_I} &\in \mathbb{R}^{n_I \times h} \quad \text{(Integrated Perturbation Set)} \\
\mathcal{Z_P} &\in \mathbb{R}^{n_P \times h} \quad \text{(Perturbation Set)} \\
\mathcal{Z_W} &\in \mathbb{R}^{n_W \times h} \quad \text{(Wildtype or Whole Set)}
\end{align}
$$

$$
\mathcal{L}_\text{con}(Z,\hat{Y}) \\
\mathcal{L}_\text{dist}(Y, \hat{Y})
$$

$$
\mathcal{L}_{\text{MLE}}(y_i, \hat{y}_i)
$$

$$
\hat{f}(\mathcal{\tilde{G}}, \mathcal{\tilde{E}, \mathcal{\tilde{P}}} \rightarrow \hat{Y})
$$

where $\mathcal{\tilde{G}}$ - Cell as Graphs

where $\mathcal{\tilde{E}}$ - Environment

where $\mathcal{\tilde{P}}$ - Perturbations over Graphs and Environment

The goal is learn function $\hat{f}$ that approximates labels $Y$

***

Here's a more precise formalization:

$$
\begin{aligned}
& \text{Given spaces:} \\
& \tilde{\mathcal{G}} \subset \{\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{X}) \mid \mathcal{V} \text{ vertices}, \mathcal{E} \text{ edges}, \mathbf{X} \text{ features}\} \\
& \tilde{\mathcal{E}} \subset \mathbb{R}^{d_e} \\
& \tilde{\mathcal{P}} \subset \{\mathcal{P}: \tilde{\mathcal{G}} \times \tilde{\mathcal{E}} \to \tilde{\mathcal{G}} \times \tilde{\mathcal{E}}\} \\
& \mathcal{Y} \subset \mathbb{R}^{d_y} \\
\\
& \text{Find function:} \\
& \hat{f}: \tilde{\mathcal{G}} \times \tilde{\mathcal{E}} \times \tilde{\mathcal{P}} \to \mathcal{Y} \\
\\
& \text{Optimization objective:} \\
& \hat{f} = \arg\min_f \mathbb{E}_{(\tilde{G}, \tilde{E}, \tilde{P}, Y) \sim \mathcal{D}}[\mathcal{L}(f(\tilde{G}, \tilde{E}, \tilde{P}), Y)]
\end{aligned}
$$

Where:

- $\tilde{\mathcal{G}}$ is the space of cellular graphs with vertex/edge features
- $\tilde{\mathcal{E}}$ is a real-valued environment space
- $\tilde{\mathcal{P}}$ is the space of perturbation operators
- $\mathcal{D}$ is the data distribution
- $\mathcal{L}$ is a suitable loss function

***

$$
\hat{f}: \tilde{\mathcal{G}} \times \tilde{\mathcal{E}} \times \tilde{\mathcal{P}} \rightarrow \mathcal{Y}
$$

$$
\hat{f}=\arg \min_f \mathbb{E}_{(\tilde{G}, \tilde{E}, \tilde{P}, Y) \sim \mathcal{D}}[\mathcal{L}(f(\tilde{G}, \tilde{E}, \tilde{P}), Y)]
$$

Where:

- $\tilde{\mathcal{G}}$: cellular graphs with vertex/edge features
- $\tilde{\mathcal{E}}$: real-valued environment space  
- $\tilde{\mathcal{P}}$: perturbation operators
- $\mathcal{D}$: data distribution
- $\mathcal{L}$: loss function

***

$$
\begin{aligned}
\hat{f}_\theta&: \tilde{\mathcal{G}} \times \tilde{\mathcal{E}} \times \tilde{\mathcal{P}} \rightarrow \mathcal{Y} \\[1em]
\hat{\theta}&=\arg \min_\theta \mathbb{E}_{(\tilde{G}, \tilde{E}, \tilde{P}, Y) \sim \mathcal{D}}[\mathcal{L}(\hat{f}_\theta(\tilde{G}, \tilde{E}, \tilde{P}), Y)]
\\[2em]
\text{where: }& \\
& \tilde{\mathcal{G}} \text{ - cellular graphs with vertex/edge features} \\
& \tilde{\mathcal{E}} \text{ - real-valued environment space} \\
& \tilde{\mathcal{P}} \text{ - perturbation operators} \\
& \mathcal{D} \text{ - data distribution} \\
& \mathcal{L} \text{ - loss function} \\
& \theta \text{ - learnable parameters}
\end{aligned}
$$
