---
id: kwd1y1sj3juyhjhdm1usgcz
title: Readouts
desc: ''
updated: 1789592140730
created: 1789592140730
---

## 2026.09.16 - The three essentiality readouts

Labels are KaTeX (`$$...$$`); multi-line labels use the triple-backslash row separator the
repo's mermaid rules require. Rendered by `notes/assets/publish/scripts/mermaid_pdf.sh`; the
PDFs are the figures of notes-tex/028-gene-essentiality section 4. Colors follow the module
figure: orange the gene-state path, purple the fixed stoichiometry, yellow the parameter
tables and the flux layer (dashed border = a fixed table), red the prediction and the loss.

### 0. color key (one row, rendered once and placed above the three readouts)

```mermaid
flowchart LR
  A["$$\text{input}$$"] ~~~ B["$$\begin{gathered}\text{backbone,}\\\ \text{learned}\end{gathered}$$"] ~~~ C["$$\begin{gathered}\text{flux layer,}\\\ \text{learned}\end{gathered}$$"] ~~~ D["$$\begin{gathered}\text{fixed table}\\\ \text{(dashed)}\end{gathered}$$"] ~~~ E["$$\begin{gathered}\text{GEM,}\\\ \text{fixed}\end{gathered}$$"] ~~~ F["$$\begin{gathered}\text{prediction}\\\ \text{and loss}\end{gathered}$$"]
  style A fill:#FFFFFF,stroke:#666666
  style B fill:#FFE6CC,stroke:#D79B00
  style C fill:#FFF2CC,stroke:#D6B656
  style D fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style E fill:#E1D5E7,stroke:#9673A6,stroke-dasharray:4 3
  style F fill:#F8CECC,stroke:#B85450
```

### 1. pooled (the backbone alone, no flux layer)

```mermaid
flowchart LR
  G["$$\text{Genotype (deleted gene)}$$"] --> B["$$\begin{gathered}\text{CGT pooled backbone}\\\ \text{context } h\end{gathered}$$"]
  B --> H["$$\ell = \mathrm{MLP}(h)$$"]
  H --> L["$$\mathrm{BCE}\big(\ell,\ y_{\text{essential}}\big)$$"]
  X["$$\begin{gathered}\text{no } S,\ \text{no } k_{cat},\\\ \text{no } v\end{gathered}$$"]
  style G fill:#FFFFFF,stroke:#666666
  style B fill:#FFE6CC,stroke:#D79B00
  style H fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
  style X fill:#FFF2CC,stroke:#D6B656
```

### 2. biomass (forced: the growth column is the only path to the logit)

```mermaid
flowchart LR
  G["$$\begin{gathered}\text{Genotype}\\\ \text{(deleted gene)}\end{gathered}$$"] --> B["$$\begin{gathered}\text{CGT backbone}\\\ \text{context } h\end{gathered}$$"]
  B --> E["$$\begin{gathered}E_g = \gamma_g E^0_g\\\ \times\, e^{r\tanh(w^\top h_g)}\\\ \gamma_g = 0 \text{ if deleted}\end{gathered}$$"]
  K["$$k_{cat}\ \text{table}$$"] --> CAP
  E --> CAP["$$\begin{gathered}\mathrm{cap}_j =\\\ \sum_{u \in \mathrm{GPR}(j)} k_{cat,u} E_u\end{gathered}$$"]
  GEM["$$\begin{gathered}\text{Yeast9 GEM}\\\ S,\ [\ell_j, u_j],\ \text{GPR}\end{gathered}$$"] --> V
  CAP --> V["$$\begin{gathered}v = \Pi_{Sv=0}\big(\mathrm{clip}(v_z)\big)\\\ \text{bounds } \ell_j, u_j, \pm\mathrm{cap}_j\\\ K \text{ alternations}\end{gathered}$$"]
  V --> H["$$\begin{gathered}\ell = a\,v_{\text{growth}} + b\\\ v_{\text{growth}} = v_{r\_2111}\end{gathered}$$"]
  H --> L["$$\mathrm{BCE}(\ell,\ y)$$"]
  V --> T["$$\begin{gathered}\lambda \sum_j \mathrm{relu}(v_j\,\Delta_r G'_j)\\\ \lambda \in \{0, 0.1\}\end{gathered}$$"]
  style G fill:#FFFFFF,stroke:#666666
  style B fill:#FFE6CC,stroke:#D79B00
  style E fill:#FFE6CC,stroke:#D79B00
  style GEM fill:#E1D5E7,stroke:#9673A6,stroke-dasharray:4 3
  style K fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style CAP fill:#FFF2CC,stroke:#D6B656
  style V fill:#FFF2CC,stroke:#D6B656
  style T fill:#FFF2CC,stroke:#D6B656
  style H fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
```

### 3. gated (optional: the pooled head plus a zero-initialized gate on the growth affine)

```mermaid
flowchart LR
  G["$$\begin{gathered}\text{Genotype}\\\ \text{(deleted gene)}\end{gathered}$$"] --> B["$$\begin{gathered}\text{CGT backbone}\\\ \text{context } h\end{gathered}$$"]
  B --> H["$$\ell_0 = \mathrm{MLP}(h)$$"]
  B --> E["$$\begin{gathered}E_g = \gamma_g E^0_g\\\ \times\, e^{r\tanh(w^\top h_g)}\\\ \gamma_g = 0 \text{ if deleted}\end{gathered}$$"]
  K["$$k_{cat}\ \text{table}$$"] --> CAP
  E --> CAP["$$\begin{gathered}\mathrm{cap}_j =\\\ \sum_{u \in \mathrm{GPR}(j)} k_{cat,u} E_u\end{gathered}$$"]
  GEM["$$\begin{gathered}\text{Yeast9 GEM}\\\ S,\ [\ell_j, u_j],\ \text{GPR}\end{gathered}$$"] --> V
  CAP --> V["$$\begin{gathered}v = \Pi_{Sv=0}\big(\mathrm{clip}(v_z)\big)\\\ \text{bounds } \ell_j, u_j, \pm\mathrm{cap}_j\\\ K \text{ alternations}\end{gathered}$$"]
  V --> S["$$\begin{gathered}\ell = \ell_0 + g\,(a\,v_{\text{growth}} + b)\\\ v_{\text{growth}} = v_{r\_2111}\\\ g_0 = 0,\ g \text{ logged}\end{gathered}$$"]
  H --> S
  S --> L["$$\mathrm{BCE}(\ell,\ y)$$"]
  style G fill:#FFFFFF,stroke:#666666
  style B fill:#FFE6CC,stroke:#D79B00
  style E fill:#FFE6CC,stroke:#D79B00
  style GEM fill:#E1D5E7,stroke:#9673A6,stroke-dasharray:4 3
  style K fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style CAP fill:#FFF2CC,stroke:#D6B656
  style V fill:#FFF2CC,stroke:#D6B656
  style H fill:#F8CECC,stroke:#B85450
  style S fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
```

## 2026.09.21 - One diagram per module form that was trained

Same palette, key and node vocabulary as the three readouts above, one diagram for each
form of the metabolic module that was trained after them, in the order the forms first
ran. `mermaid_pdf.sh` numbers the PDFs by position in this file, so diagram `N` below is
`experiments.028-gene-essentiality.mermaid.readouts-(N+1).pdf` and lands in
`notes-tex/028-gene-essentiality/figures/readout_N.pdf`. Keep new diagrams at the END of
this file so the numbering of the earlier ones never shifts. The layer itself (the
alternating projections, the layer that solves, the elastic program, the flux head, the
solve on the model's own box) is drawn in draw.io by `notes/assets/drawio/028-module.gen.py`.

### 4. fba affine (round 5: FBA's growth ratios as an affine on the logit, no layer)

```mermaid
flowchart LR
  G["$$\begin{gathered}\text{Genotype}\\\ \text{(deleted gene)}\end{gathered}$$"] --> B["$$\begin{gathered}\text{CGT backbone}\\\ \text{context } h\end{gathered}$$"]
  B --> H["$$\ell_0 = \mathrm{MLP}(h)$$"]
  GEM["$$\begin{gathered}\text{Yeast9 GEM}\\\ S,\ [\ell_j, u_j],\ \text{GPR}\end{gathered}$$"] --> F
  G --> F["$$\begin{gathered}\text{FBA per deletion, offline}\\\ r_m = \mu^{\text{KO}}_m / \mu^{\text{WT}}_m\\\ m \in 4 \text{ media}\end{gathered}$$"]
  F --> S["$$\begin{gathered}\ell = \ell_0 + w^\top r + b\\\ 5 \text{ parameters}\end{gathered}$$"]
  H --> S
  S --> L["$$\mathrm{BCE}(\ell,\ y)$$"]
  style G fill:#FFFFFF,stroke:#666666
  style B fill:#FFE6CC,stroke:#D79B00
  style GEM fill:#E1D5E7,stroke:#9673A6,stroke-dasharray:4 3
  style F fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style H fill:#F8CECC,stroke:#B85450
  style S fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
```

### 5. gated plus fba affine (round 6: the alternating layer beside the solve)

```mermaid
flowchart LR
  G["$$\begin{gathered}\text{Genotype}\\\ \text{(deleted gene)}\end{gathered}$$"] --> B["$$\begin{gathered}\text{CGT backbone}\\\ \text{context } h\end{gathered}$$"]
  B --> H["$$\ell_0 = \mathrm{MLP}(h)$$"]
  B --> E["$$\begin{gathered}E_g = \gamma_g E^0_g\\\ \times\, e^{r\tanh(w^\top h_g)}\\\ \gamma_g = 0 \text{ if deleted}\end{gathered}$$"]
  K["$$k_{cat}\ \text{table}$$"] --> CAP
  E --> CAP["$$\begin{gathered}\mathrm{cap}_j =\\\ \sum_{u \in \mathrm{GPR}(j)} k_{cat,u} E_u\end{gathered}$$"]
  GEM["$$\begin{gathered}\text{Yeast9 GEM}\\\ S,\ [\ell_j, u_j],\ \text{GPR}\end{gathered}$$"] --> V
  CAP --> V["$$\begin{gathered}v = \Pi_{Sv=0}\big(\mathrm{clip}(v_z)\big)\\\ \text{bounds } \ell_j, u_j, \pm\mathrm{cap}_j\\\ K \text{ alternations}\end{gathered}$$"]
  GEM --> F["$$\begin{gathered}\text{FBA per deletion, offline}\\\ \text{growth ratios } r,\ 4 \text{ media}\end{gathered}$$"]
  V --> S["$$\begin{gathered}\ell = \ell_0 + g\,(a\,v_{\text{growth}} + b)\\\ +\, w^\top r,\quad g_0 = 0\end{gathered}$$"]
  H --> S
  F --> S
  S --> L["$$\mathrm{BCE}(\ell,\ y)$$"]
  style G fill:#FFFFFF,stroke:#666666
  style B fill:#FFE6CC,stroke:#D79B00
  style E fill:#FFE6CC,stroke:#D79B00
  style GEM fill:#E1D5E7,stroke:#9673A6,stroke-dasharray:4 3
  style K fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style F fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style CAP fill:#FFF2CC,stroke:#D6B656
  style V fill:#FFF2CC,stroke:#D6B656
  style H fill:#F8CECC,stroke:#B85450
  style S fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
```

### 6. box form (round 7: the bounds exact, no balance projection)

```mermaid
flowchart LR
  G["$$\begin{gathered}\text{Genotype}\\\ \text{(deleted gene)}\end{gathered}$$"] --> B["$$\begin{gathered}\text{CGT backbone}\\\ \text{context } h\end{gathered}$$"]
  B --> E["$$\begin{gathered}E_g = \gamma_g E^0_g\\\ \times\, e^{r\tanh(w^\top h_g)}\\\ \gamma_g = 0 \text{ if deleted}\end{gathered}$$"]
  K["$$k_{cat}\ \text{table}$$"] --> CAP
  E --> CAP["$$\begin{gathered}\mathrm{cap}_j =\\\ \sum_{u \in \mathrm{GPR}(j)} k_{cat,u} E_u\end{gathered}$$"]
  GEM["$$\begin{gathered}\text{Yeast9 GEM}\\\ S,\ [\ell_j, u_j],\ \text{GPR}\end{gathered}$$"] --> V
  CAP --> V["$$\begin{gathered}v_j = \ell_j + (u_j - \ell_j)\,\sigma(z_j)\\\ \text{bounds exact, capped by } \pm\mathrm{cap}_j\\\ \text{no projection onto } Sv = 0\end{gathered}$$"]
  V --> H["$$\begin{gathered}\ell = a\,v_{\text{growth}} + b\\\ v_{\text{growth}} = v_{r\_2111}\end{gathered}$$"]
  H --> L["$$\mathrm{BCE}(\ell,\ y)$$"]
  style G fill:#FFFFFF,stroke:#666666
  style B fill:#FFE6CC,stroke:#D79B00
  style E fill:#FFE6CC,stroke:#D79B00
  style GEM fill:#E1D5E7,stroke:#9673A6,stroke-dasharray:4 3
  style K fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style CAP fill:#FFF2CC,stroke:#D6B656
  style V fill:#FFF2CC,stroke:#D6B656
  style H fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
```

### 7. warm start, anchored (round 10: the proposal starts at the deletion's own pFBA flux)

```mermaid
flowchart LR
  G["$$\begin{gathered}\text{Genotype}\\\ \text{(deleted gene)}\end{gathered}$$"] --> B["$$\begin{gathered}\text{CGT backbone}\\\ \text{context } h\end{gathered}$$"]
  B --> E["$$\begin{gathered}E_g = \gamma_g E^0_g\\\ \times\, e^{r\tanh(w^\top h_g)}\\\ \gamma_g = 0 \text{ if deleted}\end{gathered}$$"]
  K["$$k_{cat}\ \text{table}$$"] --> CAP
  E --> CAP["$$\begin{gathered}\mathrm{cap}_j =\\\ \sum_{u \in \mathrm{GPR}(j)} k_{cat,u} E_u\end{gathered}$$"]
  REF["$$\begin{gathered}\text{pFBA per deletion,}\\\ \text{YPD, offline: } v_{\text{ref}}\end{gathered}$$"] --> P
  B --> P["$$\begin{gathered}v_{\text{pre}} = \mathrm{clip}(v_{\text{ref}}) + \alpha\,\rho\,\hat{n}\\\ \rho = \lVert \mathrm{clip}(v_{\text{ref}}) \rVert,\ \hat{n} \in \ker S\\\ \alpha = 0.01 \text{ frozen, or learned}\end{gathered}$$"]
  GEM["$$\begin{gathered}\text{Yeast9 GEM}\\\ S,\ [\ell_j, u_j],\ \text{GPR}\end{gathered}$$"] --> V
  CAP --> V["$$\begin{gathered}v = \Pi_{Sv=0}\big(\mathrm{clip}(v_{\text{pre}})\big)\\\ \text{bounds } \ell_j, u_j, \pm\mathrm{cap}_j\\\ K \text{ alternations}\end{gathered}$$"]
  P --> V
  V --> H["$$\begin{gathered}\ell = a\,v_{\text{growth}} + b\\\ v_{\text{growth}} = v_{r\_2111}\end{gathered}$$"]
  H --> L["$$\mathrm{BCE}(\ell,\ y)$$"]
  style G fill:#FFFFFF,stroke:#666666
  style B fill:#FFE6CC,stroke:#D79B00
  style E fill:#FFE6CC,stroke:#D79B00
  style GEM fill:#E1D5E7,stroke:#9673A6,stroke-dasharray:4 3
  style K fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style REF fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style CAP fill:#FFF2CC,stroke:#D6B656
  style P fill:#FFF2CC,stroke:#D6B656
  style V fill:#FFF2CC,stroke:#D6B656
  style H fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
```

### 8. warm start, GEM box, gated (round 13: the same proposal on the stoichiometry-only layer)

```mermaid
flowchart LR
  G["$$\begin{gathered}\text{Genotype}\\\ \text{(deleted gene)}\end{gathered}$$"] --> B["$$\begin{gathered}\text{CGT backbone}\\\ \text{context } h\end{gathered}$$"]
  B --> H["$$\ell_0 = \mathrm{MLP}(h)$$"]
  B --> A["$$\begin{gathered}\gamma_g = \sigma(w^\top h_g)\\\ \gamma_g = 0 \text{ if deleted}\\\ c_j = \text{GPR fold of } \gamma\end{gathered}$$"]
  REF["$$\begin{gathered}\text{pFBA per deletion,}\\\ \text{YPD, offline: } v_{\text{ref}}\end{gathered}$$"] --> P
  B --> P["$$\begin{gathered}v_{\text{pre}} = \mathrm{clip}(v_{\text{ref}}) + \alpha\,\rho\,\hat{n}\\\ \hat{n} \in \ker S,\ \alpha = 0.01 \text{ frozen}\end{gathered}$$"]
  GEM["$$\begin{gathered}\text{Yeast9 GEM}\\\ S,\ [\ell_j, u_j],\ \text{GPR}\end{gathered}$$"] --> V
  A --> V["$$\begin{gathered}v = \Pi_{Sv=0}\big(\mathrm{clip}(v_{\text{pre}})\big)\\\ \text{bounds } \ell_j c_j,\ u_j c_j\\\ \text{no } \mathrm{cap}_j,\ \text{no second law}\end{gathered}$$"]
  P --> V
  V --> S["$$\begin{gathered}\ell = \ell_0 + g\,(a\,v_{\text{growth}} + b)\\\ g_0 = 0,\ g \text{ logged}\end{gathered}$$"]
  H --> S
  S --> L["$$\mathrm{BCE}(\ell,\ y)$$"]
  style G fill:#FFFFFF,stroke:#666666
  style B fill:#FFE6CC,stroke:#D79B00
  style A fill:#FFE6CC,stroke:#D79B00
  style GEM fill:#E1D5E7,stroke:#9673A6,stroke-dasharray:4 3
  style REF fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style P fill:#FFF2CC,stroke:#D6B656
  style V fill:#FFF2CC,stroke:#D6B656
  style H fill:#F8CECC,stroke:#B85450
  style S fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
```

### 9. widened affine (round 11, rerun in round 31: solve features as standardized affine columns)

```mermaid
flowchart LR
  G["$$\begin{gathered}\text{Genotype}\\\ \text{(deleted gene)}\end{gathered}$$"] --> B["$$\begin{gathered}\text{CGT backbone}\\\ \text{context } h\end{gathered}$$"]
  B --> H["$$\ell_0 = \mathrm{MLP}(h)$$"]
  GEM["$$\begin{gathered}\text{Yeast9 GEM}\\\ S,\ [\ell_j, u_j],\ \text{GPR}\end{gathered}$$"] --> F["$$\begin{gathered}\text{FBA per deletion, offline}\\\ \text{growth ratios } r,\ 4 \text{ media}\end{gathered}$$"]
  GEM --> T["$$\begin{gathered}\text{pFBA per deletion, YPD, offline}\\\ v \in \mathbb{R}^{4131},\ 8 \text{ solve statistics}\\\ x:\ \text{each column standardized}\end{gathered}$$"]
  F --> S["$$\begin{gathered}\ell = \ell_0 + w^\top [r;\, x] + b\\\ w \text{ default-initialized}\\\ 12 \text{ to } 4{,}139 \text{ columns}\end{gathered}$$"]
  T --> S
  H --> S
  S --> L["$$\mathrm{BCE}(\ell,\ y)$$"]
  style G fill:#FFFFFF,stroke:#666666
  style B fill:#FFE6CC,stroke:#D79B00
  style GEM fill:#E1D5E7,stroke:#9673A6,stroke-dasharray:4 3
  style F fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style T fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style H fill:#F8CECC,stroke:#B85450
  style S fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
```

### 10. solve on the GEM box with a flux head (round 24, cached in round 27)

```mermaid
flowchart LR
  G["$$\begin{gathered}\text{Genotype}\\\ \text{(deleted gene)}\end{gathered}$$"] --> B["$$\begin{gathered}\text{CGT backbone}\\\ \text{context } h\end{gathered}$$"]
  B --> H["$$\ell_0 = \mathrm{MLP}(h)$$"]
  B --> A["$$\begin{gathered}\gamma_g = \sigma(w^\top h_g)\\\ \gamma_g = 0 \text{ if deleted}\\\ c_j = \text{GPR fold of } \gamma\end{gathered}$$"]
  GEM["$$\begin{gathered}\text{Yeast9 GEM}\\\ S,\ [\ell_j, u_j],\ \text{GPR}\end{gathered}$$"] --> V
  A --> V["$$\begin{gathered}z^* = \max\ v_{\text{growth}}\\\ Sv = 0,\ \ell_j c_j \le v_j \le u_j c_j\\\ \text{no } \mathrm{cap}_j;\ \text{HiGHS, cached}\end{gathered}$$"]
  V --> S["$$\begin{gathered}\ell = \ell_0 + g\,(a\,z^* + b)\\\ +\, w^\top \log(1 + |v^*|)\\\ g_0 = 0,\ w_0 = 0,\ v^* \text{ detached}\end{gathered}$$"]
  H --> S
  S --> L["$$\mathrm{BCE}(\ell,\ y)$$"]
  style G fill:#FFFFFF,stroke:#666666
  style B fill:#FFE6CC,stroke:#D79B00
  style A fill:#FFE6CC,stroke:#D79B00
  style GEM fill:#E1D5E7,stroke:#9673A6,stroke-dasharray:4 3
  style V fill:#FFF2CC,stroke:#D6B656
  style H fill:#F8CECC,stroke:#B85450
  style S fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
```

### 11. layer-free (round 33: the solve is a stored transform, read by a zero-initialized affine)

```mermaid
flowchart LR
  G["$$\begin{gathered}\text{Genotype}\\\ \text{(deleted gene)}\end{gathered}$$"] --> B["$$\begin{gathered}\text{CGT backbone}\\\ \text{pathway} + \text{ProtT5},\ \text{context } h\end{gathered}$$"]
  B --> H["$$\ell_0 = \mathrm{MLP}(h)$$"]
  G --> P["$$\begin{gathered}\text{GEM program per deletion}\\\ \text{its own box } [\ell_j, u_j]\\\ \text{deleted gene's reactions closed}\end{gathered}$$"]
  P --> F["$$\begin{gathered}\text{FBA, offline}\\\ \text{growth ratios } r,\ 4 \text{ media}\end{gathered}$$"]
  P --> ST["$$\begin{gathered}\text{pFBA on YPD, once, offline}\\\ \text{store } x = \log(1 + |v^*|)\\\ 4{,}131 \text{ columns}\end{gathered}$$"]
  F --> S["$$\begin{gathered}\ell = \ell_0 + w_r^\top r + w_x^\top x + b\\\ w_x \text{ zero-initialized}\\\ \text{no solver in training}\end{gathered}$$"]
  ST --> S
  H --> S
  S --> L["$$\mathrm{BCE}(\ell,\ y)$$"]
  style G fill:#FFFFFF,stroke:#666666
  style B fill:#FFE6CC,stroke:#D79B00
  style P fill:#E1D5E7,stroke:#9673A6,stroke-dasharray:4 3
  style F fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style ST fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style H fill:#F8CECC,stroke:#B85450
  style S fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
```

### 12. layer-free, a store from another solver's point (rounds 36, 38 and 42)

```mermaid
flowchart LR
  G["$$\begin{gathered}\text{Genotype}\\\ \text{(deleted gene)}\end{gathered}$$"] --> B["$$\begin{gathered}\text{CGT backbone}\\\ \text{pathway} + \text{ProtT5},\ \text{context } h\end{gathered}$$"]
  B --> H["$$\ell_0 = \mathrm{MLP}(h)$$"]
  G --> P["$$\begin{gathered}\text{GEM program per deletion}\\\ \max\ v_{\text{growth}},\ \text{its own box}\\\ \text{YPD, or SC in round 42}\end{gathered}$$"]
  P --> F["$$\begin{gathered}\text{FBA, offline}\\\ \text{growth ratios } r,\ 4 \text{ media}\end{gathered}$$"]
  P --> ST["$$\begin{gathered}\text{one point of the optimal face:}\\\ \text{pFBA, dual-simplex vertex,}\\\ \text{interior point, or PDLP}\\\ \text{store } x = \log(1 + |v^*|)\end{gathered}$$"]
  F --> S["$$\begin{gathered}\ell = \ell_0 + w_r^\top r + w_x^\top x + b\\\ w_x \text{ zero-initialized}\end{gathered}$$"]
  ST --> S
  H --> S
  S --> L["$$\mathrm{BCE}(\ell,\ y)$$"]
  style G fill:#FFFFFF,stroke:#666666
  style B fill:#FFE6CC,stroke:#D79B00
  style P fill:#E1D5E7,stroke:#9673A6,stroke-dasharray:4 3
  style F fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style ST fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style H fill:#F8CECC,stroke:#B85450
  style S fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
```

### 13. layer-free, a hidden layer in the affine (round 41)

```mermaid
flowchart LR
  G["$$\begin{gathered}\text{Genotype}\\\ \text{(deleted gene)}\end{gathered}$$"] --> B["$$\begin{gathered}\text{CGT backbone}\\\ \text{pathway} + \text{ProtT5},\ \text{context } h\end{gathered}$$"]
  B --> H["$$\ell_0 = \mathrm{MLP}(h)$$"]
  G --> P["$$\begin{gathered}\text{GEM program per deletion}\\\ \text{its own box } [\ell_j, u_j]\end{gathered}$$"]
  P --> F["$$\begin{gathered}\text{FBA, offline}\\\ \text{growth ratios } r,\ 4 \text{ media}\end{gathered}$$"]
  P --> ST["$$\begin{gathered}\text{pFBA on YPD, once, offline}\\\ \text{store } x = \log(1 + |v^*|)\end{gathered}$$"]
  F --> S["$$\begin{gathered}\ell = \ell_0 + W_2\,\mathrm{GELU}(W_1 [r;\, x])\\\ \text{one hidden layer,}\\\ \text{width } 64 \text{ or } 256\end{gathered}$$"]
  ST --> S
  H --> S
  S --> L["$$\mathrm{BCE}(\ell,\ y)$$"]
  style G fill:#FFFFFF,stroke:#666666
  style B fill:#FFE6CC,stroke:#D79B00
  style P fill:#E1D5E7,stroke:#9673A6,stroke-dasharray:4 3
  style F fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style ST fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style H fill:#F8CECC,stroke:#B85450
  style S fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
```

### 14. the solve on the node (rounds 43 and 44: the stored vector as a gene embedding)

```mermaid
flowchart LR
  G["$$\begin{gathered}\text{Genotype}\\\ \text{(deleted gene)}\end{gathered}$$"] --> B
  P["$$\begin{gathered}\text{GEM program}\\\ \text{per gene,}\\\ \text{its own box}\end{gathered}$$"] --> ST["$$\begin{gathered}\text{pFBA on YPD, once, offline}\\\ x_g = \log(1 + |v^*_g|) \text{ on gene } g\text{'s node}\\\ \text{raw (round 43), or}\\\ \text{64 or 16 PCs (round 44)}\end{gathered}$$"]
  ST --> B["$$\begin{gathered}\text{CGT backbone}\\\ \text{pathway} + \text{ProtT5}\\\ +\ \text{solve, context } h\end{gathered}$$"]
  B --> H["$$\ell_0 = \mathrm{MLP}(h)$$"]
  F["$$\begin{gathered}\text{FBA, offline}\\\ \text{growth ratios } r,\\\ 4 \text{ media}\end{gathered}$$"] --> S
  H --> S["$$\begin{gathered}\ell = \ell_0 + w_r^\top r + b\\\ \text{optionally } +\, w_x^\top x\end{gathered}$$"]
  S --> L["$$\mathrm{BCE}(\ell,\ y)$$"]
  style G fill:#FFFFFF,stroke:#666666
  style B fill:#FFE6CC,stroke:#D79B00
  style P fill:#E1D5E7,stroke:#9673A6,stroke-dasharray:4 3
  style F fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style ST fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style H fill:#F8CECC,stroke:#B85450
  style S fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
```

### 15. layer-free, the clamp columns dropped from the vertex store (rounds 68 to 70)

```mermaid
flowchart LR
  G["$$\begin{gathered}\text{Genotype}\\\ \text{(deleted gene)}\end{gathered}$$"] --> B["$$\begin{gathered}\text{CGT backbone}\\\ \text{pathway} + \text{ProtT5},\ \text{context } h\end{gathered}$$"]
  B --> H["$$\ell_0 = \mathrm{MLP}(h)$$"]
  G --> P["$$\begin{gathered}\text{GEM program per deletion}\\\ \max\ v_{\text{growth}},\ \text{its own box}\end{gathered}$$"]
  P --> F["$$\begin{gathered}\text{FBA, offline}\\\ \text{growth ratios } r,\ 4 \text{ media}\end{gathered}$$"]
  P --> ST["$$\begin{gathered}\text{dual-simplex vertex,}\\\ \text{once, offline}\\\ \log(1 + |v^*|)\end{gathered}$$"]
  ST --> C["$$\begin{gathered}\text{zero each column that}\\\ \text{reaches the } \pm 100 \text{ clamp:}\\\ 429 \text{ of } 4{,}131\ \to\ x\end{gathered}$$"]
  C --> S["$$\begin{gathered}\ell = \ell_0 + w_r^\top r + w_x^\top x + b\\\ w_x \text{ zero-initialized}\end{gathered}$$"]
  F --> S
  H --> S
  S --> L["$$\mathrm{BCE}(\ell,\ y)$$"]
  style G fill:#FFFFFF,stroke:#666666
  style B fill:#FFE6CC,stroke:#D79B00
  style P fill:#E1D5E7,stroke:#9673A6,stroke-dasharray:4 3
  style F fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style ST fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style C fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style H fill:#F8CECC,stroke:#B85450
  style S fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
```

### 16. the store at the database's scale (the 025 solid-growth build)

```mermaid
flowchart LR
  G["$$\begin{gathered}\text{025 record:}\\\ \text{deleted genes, medium}\end{gathered}$$"] --> B["$$\begin{gathered}\text{025 transformer}\\\ \hat{y}_0 = f(z)\end{gathered}$$"]
  G --> KEY["$$\begin{gathered}\text{key} = (\text{GEM genes deleted,}\\\ \text{medium});\ 469{,}474 \text{ keys}\\\ \text{for } 13.5\text{M records}\end{gathered}$$"]
  P["$$\begin{gathered}\text{GEM program per key}\\\ \text{its own box, SC medium}\end{gathered}$$"] --> ST["$$\begin{gathered}\text{interior point, once per key,}\\\ \text{offline: } \log(1 + |v^*_{\text{key}}|)\\\ 8.0 \text{ GB on the host}\end{gathered}$$"]
  KEY --> GA["$$\begin{gathered}\text{gather one row } x\\\ \text{per record, to the card}\end{gathered}$$"]
  ST --> GA
  GA --> S["$$\hat{y} = \hat{y}_0 + w^\top x + b$$"]
  B --> S
  S --> L["$$\begin{gathered}\text{loss on the measured}\\\ \text{phenotype}\end{gathered}$$"]
  style G fill:#FFFFFF,stroke:#666666
  style B fill:#FFE6CC,stroke:#D79B00
  style P fill:#E1D5E7,stroke:#9673A6,stroke-dasharray:4 3
  style KEY fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style ST fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style GA fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style S fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
```

### 17. layer-free plus the gene-rule counts (round 77)

```mermaid
flowchart LR
  G["$$\begin{gathered}\text{Genotype}\\\ \text{(deleted gene)}\end{gathered}$$"] --> B["$$\begin{gathered}\text{CGT backbone}\\\ \text{pathway} + \text{ProtT5},\ \text{context } h\end{gathered}$$"]
  B --> H["$$\ell_0 = \mathrm{MLP}(h)$$"]
  G --> P["$$\begin{gathered}\text{GEM program per deletion}\\\ \text{its own box } [\ell_j, u_j]\end{gathered}$$"]
  P --> F["$$\begin{gathered}\text{FBA, offline}\\\ \text{growth ratios } r,\ 4 \text{ media}\end{gathered}$$"]
  P --> ST["$$\begin{gathered}\text{pFBA on YPD, once, offline}\\\ \text{store } \log(1 + |v^*|)\end{gathered}$$"]
  G --> R["$$\begin{gathered}\text{GPR rules in normal form:}\\\ 6 \text{ counts } c \text{ per gene}\end{gathered}$$"]
  ST --> S["$$\begin{gathered}\ell = \ell_0 + w_r^\top r + w_x^\top x + b\\\ x = \log(1 + [\,|v^*|;\ c\,])\\\ w_x \text{ zero-initialized}\end{gathered}$$"]
  R --> S
  F --> S
  H --> S
  S --> L["$$\mathrm{BCE}(\ell,\ y)$$"]
  style G fill:#FFFFFF,stroke:#666666
  style B fill:#FFE6CC,stroke:#D79B00
  style P fill:#E1D5E7,stroke:#9673A6,stroke-dasharray:4 3
  style F fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style ST fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style R fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style H fill:#F8CECC,stroke:#B85450
  style S fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
```

### 18. layer-free with the class token through the perturbation operator (round 79)

```mermaid
flowchart LR
  G["$$\begin{gathered}\text{Genotype}\\\ \text{(deleted gene)}\end{gathered}$$"] --> PO
  ENC["$$\begin{gathered}\text{CGT encoder,}\\\ \text{wild-type graph:}\\\ h_{\text{CLS}},\ H_{\text{genes}}\end{gathered}$$"] --> PO["$$\begin{gathered}\text{perturbation operator}\\\ \text{queries: } H_{\text{genes}} \text{ and } h_{\text{CLS}}\\\ \text{keys: the deleted genes}\end{gathered}$$"]
  PO --> H["$$\begin{gathered}\ell_0 = \mathrm{MLP}\big([\,h^{\text{pert}}_{\text{CLS}};\ z_S\,]\big)\\\ z_S:\ \text{pooled perturbed genes}\end{gathered}$$"]
  ST["$$\begin{gathered}\text{pFBA on YPD, once, offline}\\\ \text{store } x = \log(1 + |v^*|)\end{gathered}$$"] --> S
  F["$$\begin{gathered}\text{FBA, offline}\\\ \text{growth ratios } r,\ 4 \text{ media}\end{gathered}$$"] --> S
  H --> S["$$\begin{gathered}\ell = \ell_0 + w_r^\top r + w_x^\top x + b\\\ w_x \text{ zero-initialized}\end{gathered}$$"]
  S --> L["$$\mathrm{BCE}(\ell,\ y)$$"]
  style G fill:#FFFFFF,stroke:#666666
  style ENC fill:#FFE6CC,stroke:#D79B00
  style PO fill:#FFE6CC,stroke:#D79B00
  style F fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style ST fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style H fill:#F8CECC,stroke:#B85450
  style S fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
```

### 19. the counts without the flux vector (round 82, and the fewer-media recipes of rounds 84 to 98)

```mermaid
flowchart LR
  G["$$\begin{gathered}\text{Genotype}\\\ \text{(deleted gene)}\end{gathered}$$"] --> B["$$\begin{gathered}\text{CGT backbone}\\\ \text{pathway} + \text{ProtT5},\ \text{context } h\end{gathered}$$"]
  B --> H["$$\ell_0 = \mathrm{MLP}(h)$$"]
  G --> P["$$\begin{gathered}\text{GEM program per deletion}\\\ \text{its own box } [\ell_j, u_j]\end{gathered}$$"]
  P --> F["$$\begin{gathered}\text{FBA, offline: growth ratios } r\\\ 4 \text{ media; or YPD and SC;}\\\ \text{or YPD alone}\end{gathered}$$"]
  G --> R["$$\begin{gathered}\text{GPR rules in normal form:}\\\ 6 \text{ counts } c \text{ per gene}\end{gathered}$$"]
  F --> S["$$\begin{gathered}\ell = \ell_0 + w_r^\top r + w_c^\top \log(1 + c) + b\\\ w_c \text{ zero-initialized}\\\ \text{no flux vector stored}\end{gathered}$$"]
  R --> S
  H --> S
  S --> L["$$\mathrm{BCE}(\ell,\ y)$$"]
  style G fill:#FFFFFF,stroke:#666666
  style B fill:#FFE6CC,stroke:#D79B00
  style P fill:#E1D5E7,stroke:#9673A6,stroke-dasharray:4 3
  style F fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style R fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style H fill:#F8CECC,stroke:#B85450
  style S fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
```

### 20. every column inside one experiment (round 83)

```mermaid
flowchart LR
  G["$$\begin{gathered}\text{Record: genotype}\\\ \text{(deleted gene) on YPD}\end{gathered}$$"] --> B["$$\begin{gathered}\text{CGT backbone}\\\ \text{pathway} + \text{ProtT5},\ \text{context } h\end{gathered}$$"]
  B --> H["$$\ell_0 = \mathrm{MLP}(h)$$"]
  G --> P["$$\begin{gathered}\text{GEM program per deletion}\\\ \text{its own box, YPD only}\end{gathered}$$"]
  P --> F["$$\begin{gathered}\text{FBA, offline: one ratio}\\\ r_{\text{YPD}} = \mu^{\text{KO}} / \mu^{\text{WT}}\end{gathered}$$"]
  P --> ST["$$\begin{gathered}\text{pFBA on YPD, once, offline}\\\ \text{store } \log(1 + |v^*|)\end{gathered}$$"]
  G --> R["$$\begin{gathered}\text{GPR rules: } 6 \text{ counts } c\\\ \text{(no environment)}\end{gathered}$$"]
  ST --> S["$$\begin{gathered}\ell = \ell_0 + w_r\, r_{\text{YPD}} + w_x^\top x + b\\\ x = \log(1 + [\,|v^*|;\ c\,])\\\ \text{one genotype, one medium}\end{gathered}$$"]
  R --> S
  F --> S
  H --> S
  S --> L["$$\mathrm{BCE}(\ell,\ y)$$"]
  style G fill:#FFFFFF,stroke:#666666
  style B fill:#FFE6CC,stroke:#D79B00
  style P fill:#E1D5E7,stroke:#9673A6,stroke-dasharray:4 3
  style F fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style ST fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style R fill:#FFF2CC,stroke:#D6B656,stroke-dasharray:4 3
  style H fill:#F8CECC,stroke:#B85450
  style S fill:#F8CECC,stroke:#B85450
  style L fill:#F8CECC,stroke:#B85450
```
