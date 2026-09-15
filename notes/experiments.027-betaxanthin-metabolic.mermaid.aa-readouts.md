---
id: tt3hjv3qxll0y9ru5frivsp
title: Aa Readouts
desc: ''
updated: 1789437901337
created: 1789437901337
---

## 2026.09.14 - The three amino-acid readouts

Labels are KaTeX (`$$...$$`); multi-line labels use the triple-backslash row separator the
repo's mermaid rules require. Rendered by `notes/assets/publish/scripts/mermaid_pdf.sh`;
the PDFs are the figures of notes-tex/027-metabolic-flux-module section 5.7. Colors follow
the module figure: orange the gene-state path, purple the fixed stoichiometry and the
concentration state, yellow the parameter tables and the flux layer (dashed border = a fixed
table), red the prediction and the loss.

### 1. context_concentration (the control: no flux layer)

```mermaid
flowchart LR
  G["$$\text{Genotype (deleted genes)}$$"] --> B["$$\begin{gathered}\text{CGT pooled backbone}\\\ \text{context } h\end{gathered}$$"]
  B --> S["$$\log c = \ell + (u-\ell)\,\sigma(b + W h)$$"]
  S --> C["$$\begin{gathered}\log c \in \mathbb{R}^{2806}\\\ \text{(compartment species)}\end{gathered}$$"]
  C --> M["$$[A]_{\mathrm{total}} = \sum_n f_n\,[A]_n$$"]
  M --> P["$$\hat{y} \in \mathbb{R}^{19}$$"]
  P --> L["$$\mathrm{CRPS}\big(\hat{y},\, y_{\text{Mulleder}}\big)$$"]
  X["$$\begin{gathered}\text{no } S,\ \text{no } k_{cat},\\\ \text{no } v\end{gathered}$$"]
  style G fill:#FFFFFF,stroke:#666666
  style B fill:#FFE6CC,stroke:#D79B00
  style S fill:#E1D5E7,stroke:#9673A6
  style C fill:#E1D5E7,stroke:#9673A6
  style M fill:#E1D5E7,stroke:#9673A6
  style P fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
  style X fill:#FFF2CC,stroke:#D6B656
```

### 2. flux_concentration (the wave-7 arm: same readout inside the anchored flux layer)

```mermaid
flowchart LR
  G["$$\text{Genotype}$$"] --> B["$$\begin{gathered}\text{CGT pooled backbone}\\\ \text{context } h\end{gathered}$$"]
  B --> S["$$\log c = \ell + (u-\ell)\,\sigma(b + W h)$$"]
  S --> C["$$\log c \in \mathbb{R}^{2806}$$"]
  C --> M["$$[A]_{\mathrm{total}} = \sum_n f_n\,[A]_n$$"]
  M --> P["$$\hat{y} \in \mathbb{R}^{19}$$"]
  P --> L["$$\mathrm{CRPS}(\hat{y}, y)$$"]
  B --> Z["$$z \sim q(z \mid h)$$"]
  B --> E["$$\begin{gathered}E_g = \gamma_g\,E^0_g\,e^{\,r\tanh(w^\top h_g)}\\\ \gamma_g = 0 \text{ for a deleted gene}\end{gathered}$$"]
  K["$$\begin{gathered}k_{cat,u}\ \text{table (OED)}\\\ \text{measured 97 / homolog 782}\\\ \text{predicted 2831 / default 18}\end{gathered}$$"] --> CAP
  E --> CAP["$$\mathrm{cap}_j = \sum_{u \in \mathrm{GPR}(j)} k_{cat,u}\,E_u$$"]
  GEM["$$\begin{gathered}\text{Yeast9 GEM}\\\ S,\ [\ell_j, u_j],\ \text{GPR}\end{gathered}$$"] --> V
  CAP --> V["$$\begin{gathered}v = \Pi_{Sv=0}\!\big(\mathrm{clip}(v_z;\ \ell_j, u_j, \pm\mathrm{cap}_j)\big)\\\ \text{K alternations}\end{gathered}$$"]
  Z --> V
  DG0["$$\begin{gathered}\Delta_r G^\circ\ \text{table}\\\ \text{eQuilibrator} \cup \text{shipped}\end{gathered}$$"] --> DG
  C --> DG["$$\Delta_r G' = \Delta_r G^\circ + RT\,S^\top \log c$$"]
  DG --> T["$$\begin{gathered}\lambda \sum_j \mathrm{relu}(v_j\,\Delta_r G'_j)\ +\ \lambda\,\mathrm{relu}(g_{\mathrm{diss}} - g_{\lim})\\\ \lambda = 0.1\end{gathered}$$"]
  V --> T
  T -. "gradient only, no path to the prediction" .-> C
  style G fill:#FFFFFF,stroke:#666666
  style B fill:#FFE6CC,stroke:#D79B00
  style E fill:#FFE6CC,stroke:#D79B00
  style S fill:#E1D5E7,stroke:#9673A6
  style C fill:#E1D5E7,stroke:#9673A6
  style M fill:#E1D5E7,stroke:#9673A6
  style GEM fill:#E1D5E7,stroke:#9673A6,stroke-dasharray:4 3
  style P fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
  style Z fill:#FFF2CC,stroke:#D6B656
  style K fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style DG0 fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style CAP fill:#FFF2CC,stroke:#D6B656
  style V fill:#FFF2CC,stroke:#D6B656
  style DG fill:#FFF2CC,stroke:#D6B656
  style T fill:#FFF2CC,stroke:#D6B656
```

### 3. flux_metabolite (the turnover readout: the flux is the prediction)

```mermaid
flowchart LR
  G["$$\text{Genotype}$$"] --> B["$$\begin{gathered}\text{CGT pooled backbone}\\\ \text{context } h\end{gathered}$$"]
  B --> Z["$$z \sim q(z \mid h)$$"]
  B --> E["$$E_g = \gamma_g\,E^0_g\,e^{\,r\tanh(w^\top h_g)}$$"]
  K["$$k_{cat,u}\ \text{table (OED)}$$"] --> CAP
  E --> CAP["$$\mathrm{cap}_j = \sum_{u \in \mathrm{GPR}(j)} k_{cat,u}\,E_u$$"]
  GEM["$$\begin{gathered}\text{Yeast9 GEM}\\\ S,\ [\ell_j, u_j],\ \text{GPR}\end{gathered}$$"] --> V
  CAP --> V["$$v = \Pi_{Sv=0}\!\big(\mathrm{clip}(v_z;\ \ell_j, u_j, \pm\mathrm{cap}_j)\big)$$"]
  Z --> V
  V --> W["$$\omega_i = \tfrac{1}{2}\sum_j |S_{ij}\,v_j|,\quad i \in \text{19 amino acids}$$"]
  W --> P["$$\hat{y} \in \mathbb{R}^{19}$$"]
  P --> L["$$\mathrm{CRPS}(\hat{y}, y)$$"]
  W -. "collapses to a constant: validation spread 0.000" .-> P
  style G fill:#FFFFFF,stroke:#666666
  style B fill:#FFE6CC,stroke:#D79B00
  style E fill:#FFE6CC,stroke:#D79B00
  style GEM fill:#E1D5E7,stroke:#9673A6,stroke-dasharray:4 3
  style Z fill:#FFF2CC,stroke:#D6B656
  style K fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style CAP fill:#FFF2CC,stroke:#D6B656
  style V fill:#FFF2CC,stroke:#D6B656
  style W fill:#E1D5E7,stroke:#9673A6
  style P fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
```
