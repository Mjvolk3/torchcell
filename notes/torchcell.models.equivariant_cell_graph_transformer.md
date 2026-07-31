---
id: dodckzudna75nmc7zdm8wxi
title: Equivariant_cell_graph_transformer
desc: ''
updated: 1765213721280
created: 1765213721280
---

## Overview

The `EquivariantCellGraphTransformer` implements a two-stage virtual instrument architecture for modeling cellular phenotypes as $y = f(G, S)$, where $G$ is the wildtype multi-graph and $S \subseteq V(G)$ is the set of perturbed genes. Unlike the non-equivariant variant, this model preserves per-gene structure through the perturbation transformation, enabling multi-task learning.

**Location:** `torchcell/models/equivariant_cell_graph_transformer.py`

**Architecture diagrams:** [[torchcell.models.equivariant_cell_graph_transformer.mermaid]]

## Mathematical Framework

### Core Reparametrization

Both transformer variants implement the **reparametrized form**:

$$
f(G,S) \approx g_\psi\big(F_\theta(G), S\big)
$$

This exploits the fact that $(G, S)$ uniquely determines $(G \setminus S)$, so any function of $(G \setminus S)$ can be reparametrized via $(F_\theta(G), S)$.

**Computational Cost Reduction:**

$$
\text{cost} \sim \mathcal{O}(B L) \quad \text{vs} \quad \mathcal{O}(B L |E|) \text{ (original)}
$$

The $|E|$ dependence is eliminated because we only encode the wildtype graph $G$ once, not per-sample perturbed graphs.

### Gene Sequence with CLS Token

Input sequence including CLS token:

$$
X = \big(x_{\mathrm{CLS}}, x_1,\dots,x_N\big) \in \mathbb{R}^{(N+1) \times d_{\text{in}}}
$$

where:

- $x_{\mathrm{CLS}}$ is a learnable whole-cell token
- $x_i = \text{Embed}(g_i)$ for gene $g_i \in \mathcal{G}$ with $|\mathcal{G}| = N = 6607$

### Transformer Encoder

$$
H = F_\theta(X) = \big(h_{\mathrm{CLS}}, h_1,\dots,h_N\big) \in \mathbb{R}^{(N+1) \times d}
$$

**Interpretation:**

- $h_{\mathrm{CLS}} \in \mathbb{R}^d$ is the **whole-cell representation**
- $h_i \in \mathbb{R}^d$ are **gene embeddings**

### Graph-Regularized Attention

Given $K$ biological graphs $G^{(k)} = (V, E^{(k)})$ with adjacencies $A^{(k)} \in \{0,1\}^{N \times N}$:

**Row-Normalized Adjacency:**

$$
\tilde{A}_{i,:}^{(k)} = \frac{A_{i,:}^{(k)}}{\sum_j A_{ij}^{(k)} + \varepsilon} = \frac{A_{i,:}^{(k)}}{d_i^{(k)} + \varepsilon}
$$

**Graph Regularization Loss** for graph $k$ assigned to layer $\ell_k$, head $h_k$:

$$
\mathcal{L}_{\text{graph}}^{(k)} = \sum_{i \in \mathcal{I}_k} \text{KL}\left(\tilde{A}_{i,:}^{(k)} \| \alpha_{i,:}^{(\ell_k, h_k)}\right)
$$

where $\mathcal{I}_k$ is a sampled row set (e.g., 50% of nodes with positive degree).

**Total Loss:**

$$
\mathcal{L} = \mathcal{L}_{\text{phenotype}} + \sum_{k=1}^K \lambda_k \mathcal{L}_{\text{graph}}^{(k)}
$$

## Type I / Type II Virtual Instrument Architecture

The key architectural innovation is the **separation of perturbation transformation (Type I) from task-specific readout (Type II)**.

### Type I: Equivariant Perturbation Transform

**Mathematical Formulation:**

$$
H_{\text{pert}} = \mathcal{T}_\psi(H^{(L)}, S) \in \mathbb{R}^{B \times N \times d}
$$

**Cross-Attention Implementation:**

For each sample $b \in \{1, \dots, B\}$ with perturbation set $S_b$:

1. **Cross-attention from all genes to perturbation context:**

$$
\begin{align}
\text{Query} &: H_{\text{genes}} \in \mathbb{R}^{N \times d} \quad \text{(all genes)} \\
\text{Key, Value} &: H_{\text{genes}}[S_b] \in \mathbb{R}^{|S_b| \times d} \quad \text{(perturbed genes)}
\end{align}
$$

$$
H_{\text{attn}}^{(b)} = \text{CrossAttn}\big(H_{\text{genes}}, H_{\text{genes}}[S_b], H_{\text{genes}}[S_b]\big) \in \mathbb{R}^{N \times d}
$$

2. **Residual connection + LayerNorm:**

$$
H_{\text{res}}^{(b)} = \text{LayerNorm}\big(H_{\text{genes}} + \text{Dropout}(H_{\text{attn}}^{(b)})\big)
$$

3. **Feed-forward refinement:**

$$
H_{\text{pert}}^{(b)} = \text{LayerNorm}\big(H_{\text{res}}^{(b)} + \text{Dropout}(\text{FFN}(H_{\text{res}}^{(b)}))\big) \in \mathbb{R}^{N \times d}
$$

4. **Stack across batch:**

$$
H_{\text{pert}} = \text{stack}([H_{\text{pert}}^{(1)}, \dots, H_{\text{pert}}^{(B)}]) \in \mathbb{R}^{B \times N \times d}
$$

**Key Property:** Output is **EQUIVARIANT** - preserves per-gene structure for all samples.

### Type II: Task-Specific Readout

**Gene Interaction Readout:**

1. **Aggregate perturbed genes per sample:**

$$
z_S^{(b)} = \frac{1}{|S_b|} \sum_{i \in S_b} H_{\text{pert}}^{(b)}[i] \in \mathbb{R}^d
$$

2. **Concatenate with CLS token:**

$$
\text{input}^{(b)} = [h_{\mathrm{CLS}} \,\|\, z_S^{(b)}] \in \mathbb{R}^{2d}
$$

3. **Predict gene interaction:**

$$
\hat{y}_{\text{GI}}^{(b)} = \text{MLP}(\text{input}^{(b)}) \in \mathbb{R}
$$

**Future Type II Instruments:**

| Task | Formula | Output Shape |
|------|---------|--------------|
| **Fitness** (invariant) | $\text{MLP}(\text{GlobalPool}(H_{\text{pert}}))$ | $\mathbb{R}^B$ |
| **Expression** (equivariant) | $\text{MLP}_{\text{gene}}(H_{\text{pert}})$ | $\mathbb{R}^{B \times N}$ |
| **Morphology** (gene-set) | $\text{MLP}(\text{GeneSetPool}(H_{\text{pert}}))$ | $\mathbb{R}^{B \times m}$ |

## Comparison: Non-Equivariant vs Equivariant

### Mathematical Difference

**Non-Equivariant:**

$$
\hat{y} = g_\psi(h_{\mathrm{CLS}}, H_{\text{genes}}, S) \quad \text{→ Immediate collapse to scalar}
$$

**Equivariant:**

$$
\hat{y} = \mathcal{R}_\phi(\mathcal{T}_\psi(H, S)) \quad \text{→ Preserves } H_{\text{pert}} \in \mathbb{R}^{B \times N \times d}
$$

### Summary Table

| Aspect | Non-Equivariant | Equivariant |
|--------|-----------------|-------------|
| **Output Structure** | Invariant ($[B, 1]$) | Equivariant ($[B, N, d]$ available) |
| **Multi-Task Support** | Single task | Multiple tasks |
| **Architecture** | Single fused module | Two-stage (Type I + Type II) |
| **Parameter Count** | ~748K | ~781K (+4.5%) |
| **Max Batch Size (4×A100)** | 512 | 128 (memory constrained) |
| **Modularity** | Fixed architecture | Swappable Type II heads |

## Memory Constraints

### Why No Propagation Layers?

The architecture originally proposed additional **propagation layers** after Type I:

$$
H_{\text{pert}}^{(\ell+1)} = \text{TransformerLayer}(H_{\text{pert}}^{(\ell)})
$$

However, this is **memory-prohibitive**:

$$
\text{Attention memory} = B \times H \times N^2 \times 4 \text{ bytes}
$$

For $B=256$, $H=8$, $N=6607$:

$$
256 \times 8 \times 6607^2 \times 4 \approx 357 \text{ GB per layer}
$$

**Solution:** Use single cross-attention transform (Type I) without propagation.

### Batch Size Trade-offs

| Batch Size | Equivariant Transform | With 1 Prop Layer | Status |
|------------|----------------------|-------------------|--------|
| 256 | ~17 GB | ~357 GB | ✓ Safe / ❌ OOM |
| 64 | ~17 GB | ~89 GB | ✓ Safe / ❌ OOM |
| 16 | ~16 GB | ~22 GB | ✓ Safe / ⚠️ Tight |

## Parameter Count Breakdown

**Configuration:** $d=64$, $L=6$, $H=8$

| Component | Non-Equivariant | Equivariant | Change |
|-----------|-----------------|-------------|--------|
| gene_embedding | 422,848 | 422,848 | 0 |
| cls_token | 64 | 64 | 0 |
| transformer_layers | 299,904 | 299,904 | 0 |
| **perturbation_transform** | — | **49,984** | **+49,984** |
| perturbation_head | 24,961 | 8,321 | -16,640 |
| **TOTAL** | **747,777** | **781,121** | **+4.5%** |

**New Parameters from Type I Transform:**

- Cross-attention (Q, K, V, O): 16,640 params
- FFN ($d \to 4d \to d$): 33,088 params
- LayerNorm (2×): 256 params

## Implementation Details

**Default Configuration:**

- $N = 6607$ genes
- $d = 64$ or $96$ hidden dimensions
- $L = 6$ or $8$ transformer layers
- $H = 8$ or $12$ attention heads per layer
- $K = 9$ biological graphs (physical, regulatory, tflink, 6× STRING layers)
- Graph regularization at mid-depth layers (e.g., layer 4)

## When to Use Which Model?

### Use Non-Equivariant if:

- Single task (gene interaction prediction)
- Speed is critical
- Limited GPU memory
- Simple deployment requirements

### Use Equivariant if:

- Multi-task learning (fitness, expression, morphology)
- Biological interpretability of perturbed cell states is important
- Future extensibility to new readout tasks
- Sufficient GPU memory (reduce batch size as needed)

## Future Directions

### Enabling Propagation Layers

1. **Flash Attention 2**: 2-4× memory reduction
2. **Sparse Attention**: Only attend to k-nearest neighbors ($O(B \times N \times k)$)
3. **Gradient Checkpointing**: Trade compute for memory
4. **Biological Sparsity**: Use graph structure to define attention neighborhoods

### Multi-Task Type II Instruments

**Expression Prediction** (equivariant):

$$
\hat{y}_{\text{expr}}^{(b)} = \text{MLP}_{\text{gene}}(H_{\text{pert}}^{(b)}) \in \mathbb{R}^N
$$

**Morphology Prediction** (gene-set specific):

$$
\hat{y}_{\text{morph}}^{(b)} = \text{MLP}\big(\text{GeneSetPool}(H_{\text{pert}}^{(b)}, \mathcal{G}_{\text{cytoskeleton}})\big) \in \mathbb{R}^m
$$

### Sparse Multi-Task Loss

With PyG-style pointers for missing labels:

$$
\mathcal{L} = \sum_{t \in \{\text{fit, GI, morph, expr}\}} w_t \sum_{b : \text{label}_t^{(b)} \text{ available}} \ell_t\big(\hat{y}_t^{(b)}, y_t^{(b)}\big)
$$

## Related Modules

- [[torchcell.models.cell_graph_transformer]] - Non-equivariant variant
- [[torchcell.viz.graph_recovery]] - Edge recovery metrics for graph regularization
- [[torchcell.viz.transformer_diagnostics]] - Attention health diagnostics
- [[torchcell.trainers.int_hetero_cell]] - Trainer for both model variants

## 2026.07.31 - Giving the Perturbation Operator a Pair Term, and Buying the Graph Prior Back with a Mask

Two facts forced this round's rewrite of the module (+1476/-73 on branch `019-decoder-pair-routing`). First, at $|S|=1$ -- **95.4%** of the expression build (1484 singles vs 72 Sameith doubles, `fig3_overlap_census.json`) -- the perturbation operator is **degenerate**: softmax over a one-element key set gives $\alpha = 1$ for every query, so the model reduces to $g(h_i + c_b)$ and **no term depends on the pair $(p, i)$**. Second, the graph prior was being paid for with a materialized attention matrix, which is a per-epoch tax on every run whether or not the KL is even on. The module now exposes a monotone ladder of pair-term mechanisms, and takes its graph prior structurally instead of through a penalty.

### Hard edge masking replaces the graph_reg KL

`attention_mask_config` burns one gene-gene relation into each attention head as a `[heads, N+1, N+1]` boolean mask (`CellGraphTransformer._build_head_mask`), so the graph prior is exact and has no $\lambda$ to calibrate. This is now the **frozen default**, not an arm (`conf/cgt_expr_010.yaml`).

The speedup is structural, not incidental. The KL needs attention WEIGHTS, so `GraphRegularizedTransformerLayer` must take the manual matmul-then-softmax path and materialize a `[1, 9, 6608, 6608]` tensor per regularized layer ($9 \times 6608^2 \times 4\,\text{B} = 1.57$ GB). Dropping the KL is what allows `scaled_dot_product_attention`'s fused kernel, which never forms that matrix; the mask is what makes dropping the KL acceptable. Measured on 8 runs launched within 21 s of each other, 2-per-GPU on the same box (`conf/cgt_expr_010.yaml` header):

| run | mask | lambda | s/epoch |
|---|---|---|---|
| `D2_mask` (au8fu60o) | on | 0 | **28.0** (235 epochs / 6584 s) |
| `D1_bilinear32` | off | 2e-7 | 42.3 |
| `H0_factor` | off | 2e-7 | 42.9 |
| `F1_attn4` | off | 2e-7 | 45.5 |
| `A0_baseline` | off | 2e-7 | 47.1 |
| `GEARS_crossgene` | off | 2e-7 | 47.8 |
| `E0_perceiver_on` | off | 2e-7 | 48.6 |
| `F1_deep2` | off | 2e-7 | 48.7 |

1.51x-1.74x faster per epoch (1.68x against `A0_baseline`), well outside the +/-5 s/epoch spread among the KL runs; memory ~35.3 vs ~40 GB per GPU pair.

- **The fused path is gated on `return_attention` ALONE, not on having a mask.** The earlier condition `head_mask is not None and not return_attention` forfeited the fused kernel on every UNMASKED layer, so with `attention_mask.layers=[1]` on a four-layer encoder only ONE layer fused and the other three still built and discarded the 1.57 GB matrix.
- **`dropout_p` is now passed explicitly to SDPA.** The old fused branch passed none, so a masked layer silently trained at $p = 0$ while every manual layer applied $p = 0.1$ -- a regularization difference confounded with the mask itself. SDPA also consumes RNG differently from the manual softmax, so trajectories do not match earlier runs at an identical seed; this is why masking lands on a new wave boundary rather than pooling with wave3/wave4a.
- **It fails loudly where the KL failed silently.** `compute_graph_regularization_loss` skipped an unmatched graph name with a bare `continue`, which is how the config names `physical`/`regulatory` never matched the `cell_graph` names `physical_interaction`/`regulatory_interaction` and **2 of 9 heads went unregularized for three rounds**. `_build_head_mask` raises. The layer indices are validated too: `layers=[7]` on a four-layer stack previously applied NO mask, raised nothing, and still logged `attention_mask.enabled=true, n_graphs=9`.
- **Masking preserves the organism-transfer story** -- graphs stay STRUCTURE ON ATTENTION rather than input features, so "no graph" is simply "no mask". That is the criterion the graph-propagation arms fail.
- **Not established by this freeze:** mask-vs-KL on ACCURACY (D2_mask's score was resolved to the literal string `A0_baseline` by the pre-fix scorer and averaged into the baseline), and the mask-vs-`lambda=0` attribution (that run changed both). A paired-seed A/B is still owed, and must run against `cgt_expr_010` as reference.

### The pair-rank ladder

Every wave-6 arm differs ONLY in how much (perturbation, gene) interaction the architecture can express, so the axis is monotone and the question is where it saturates:

| arm | form | pair rank | class |
|---|---|---|---|
| `V_ref` | $g(h_i + c_b)$ | 0 | additive reference |
| `V_sink` | one bounded scalar per head | 9 | `EquivariantPerturbationTransform(null_sink=True)` |
| `V_basis{16,32,64}` | $\sum_j b_{ij}(h_i)\, a_j(z_S)$ | $r$ | `ResponseBasisHead` |
| `V_film` | $h_i \odot (1 + \gamma(z_S)) + \beta(z_S)$ | $d = 90$ | `PerGeneHead(film_dim>0)`, at the READOUT |
| `V_hadamard(_add)` | $h_i \odot (1 + \gamma(c_b))$ | $d = 90$ | `EquivariantPerturbationTransform(hadamard=...)`, at the OPERATOR |

- **Headroom is not the constraint**, so a null on this axis is about the MECHANISM, not capacity: the rank-$r$ reconstruction ceiling is **0.7265** at $r=32$ and **0.7799** at $r=64$ (`results/lowrank_output_ceiling.json`, train-basis arm) against a 0.198 best.
- **`ResponseBasisHead` rank 32 is measured, not chosen**: the participation-ratio effective rank of the reproducible residual gene-gene correlation is 32.8 (split-half $r = 0.869$). Its amplitude output is zero-initialized, so the head is an exact identity at init.
- It **supersedes `LowRankBilinear`**, whose first version applied a LayerNorm to $a \odot b$ and so normalized away the magnitude the module existed to expose. The previously reported null (-0.0006) was that bug, not evidence.
- **FiLM is conditioned on $z_S$ alone, so `film_dim = hidden_channels`, not $2d$.** `pert_cond` is $[z_S ; h_{CLS}]$ and $h_{CLS}$ comes from the encoder on the WILDTYPE graph -- byte-identical across strains (measured across-strain sd 0.0, against 0.973 for $z_S$). Feeding it meant half the conditioner's first-layer weights saw a constant and could only contribute a fixed offset.

### The Hadamard operator: two modes, two different questions

```python
mod = H_cur * self.hadamard_gamma(attended)          # gamma's last layer is zero-init
attended = mod if self.hadamard == "replace" else attended + mod
```

- **`replace`** -- $g(h_i \odot (1 + \gamma(c_b)))$. The additive path is GONE, so at init ($\gamma = 0$) the block is the identity on $h_i$ and the model sees NO perturbation at all; it must learn the entire pathway through $\gamma$. This is "assertion instead of cross-attention" literally, and it starts strictly worse than the reference.
- **`add`** -- $g(h_i + c_b + h_i \odot \gamma(c_b))$. **Bit-identical to the additive reference at init (verified: max|diff| = 0.000e+00)**, so a null here reads as "the multiplicative term does not help" rather than "the arm never got off the ground". Both are run because `replace` answers the question that was asked and `add` is the one whose null is interpretable.
- Routing the modulation through the EXISTING residual exploits $x + x \odot \gamma(c) \equiv x \odot (1 + \gamma(c))$, which keeps postln/rezero, dropout and the FFN byte-identical between arm and reference -- only the meaning of `attended` changes.

### The null sink: restoring query dependence at |S| = 1

Appending one ZERO-valued key/value column plus a learned scalar `null_bias` to that column's logit makes the real-key weight $\alpha_i = \sigma(q_i \cdot k_p / \sqrt{d_k} - b)$, which DEPENDS ON $i$ -- a genuine pair-$(p,i)$ term learned inside the attention, rank 1 per head and rank `num_heads` across heads.

- The defect it fixes is measured (`results/perturbation_selector_degeneracy.json`): at $|S|=1$ the per-query weight spread is 3.31e-09, re-drawing $W_Q, W_K$ at std 10 changes the output by **exactly 0.0**, and **16,200 of 32,760** attention parameters get no gradient. (The module's block comment says 16,380 for the same defect: that count includes the $2 \times 90$ Q/K biases, while the script counts the $2 \times 90 \times 90$ weights.)
- The fix is verified end to end (`results/verify_null_sink_and_pooling.json`): query spread 2.98e-09 -> 0.0624 (2.09e+07x), $W_Q/W_K$ gradient norm 0.0 -> 56645.7, `null_bias` gradient 418.7, and the sham arm (`bias_init=-20`, non-trainable) matches sink-off to 5.96e-08 on the SAME code path.
- **`null_sink_magnitude_match` is what makes the arm interpretable.** A zero-valued sink diverts mass to the origin and permanently ATTENUATES the context ($\lVert c_{sink}\rVert / \lVert c_{ref}\rVert = 0.5139$ at bias 0); a bare scalar 0.5 on the REFERENCE context, with no sink at all, reproduces **97.8%** of the arm's downstream deviation. Dividing by $\sigma(-b_{init})$ restores the norm (ratio 1.0258) while leaving the per-query spread untouched (CV 0.0971), so the arm differs from the reference by the query-dependent GATE alone.
- It does NOT restore cardinality: $\lVert c \rVert$ shrinks with the sink rather than growing with $|S|$ (4.75/2.48/1.91 with sink vs 5.43/2.65/1.95 without, at $|S| = 1/2/3$). Cardinality is the `pooling` lever instead -- `sum` is cardinality-aware where `mean` is blind (C5/C6 in `verify_null_sink_and_pooling.json`: $S=\{A,A\}$ vs $S=\{A\}$ gives mean 0.1022 vs 0.1022, sum 0.1868 vs 0.1022). `pert_pooling` is read once and shared with the per-gene $z_S$ site so the two can never disagree.
- `null_sink_bias_init=0.0` starts the sink carrying ~50% of the mass so that the model closing it is itself an observable result; a strongly negative init can never engage (at -4.0 the gate moved 0.07 over 187 epochs).

### Post-perturbation mixing, and the observed-labels encoder the v9 objective needs

After `EquivariantPerturbationTransform` there was NO gene-gene interaction anywhere, so downstream cascades ("delete p -> changes j -> changes i") were structurally unrepresentable. Three blocks now sit in that slot, in forward order: `PerturbationGraphPropagation` (5b), `ObservedLabelEncoder` (5b2), `CrossGeneMixing` (5b3), `PerceiverMixing` (5c).

- **`PerceiverMixing`** routes genes -> M latents -> genes, so cost is $O(NM)$ not $O(N^2)$: ~4.9 GFLOP and ~0.12 GB at $M = 32$, against ~503 GFLOP and a `[32, 9, 6607, 6607]` (~25 GB bf16) tensor for dense post-perturbation self-attention. Graph-FREE, so it transfers to organisms with no curated interactome.
- **`gate_mode="on"` exists because a closed ReZero gate is not evidence against a mechanism.** The gate receives gradient only in proportion to how useful the block is with UNTRAINED internals, which is random. Measured: the Perceiver's gate finished at 1.0e-5 (never opened) while the propagation module's reached 0.125 under the identical mechanism. Forcing the gate to 1.0 removes the confound.
- **`ObservedLabelEncoder` is the input side of the v9 teacher-forced masked-label objective.** It projects `(value, observed-flag)` per gene into a token-space offset -- the flag is required, since values are zeroed where unobserved and without it "observed and happens to be 0" is indistinguishable from "not observed". It is injected BEFORE the mixing blocks so revealed values have a pathway to other genes; without mixing the conditioning is inert (the Perceiver alone moved seed 0 by only +0.0045). It defaults to `gate_mode="on"` because the point is to push gradient through the pathway, and because at validation everything is masked, the encoded features are all zero and the forward pass is an exact identity -- which is what keeps the per-feature metric comparable to every other arm.
- **`PerturbationGraphPropagation`** adds `len(graphs) * hops + 1` reachability scalars per gene, the +1 being a hop-0 self-indicator ("is gene $i$ itself deleted in strain $b$?") -- the cheapest possible pair term, and the deleted gene's own down-regulation is the most predictable single value in a Kemmeren profile. The sparse path runs in fp32 with autocast disabled: CUDA has no bf16 `addmm_sparse` kernel, and $O(|E|)$ mat-vecs cost nothing next to the encoder anyway. Note this pathway needs curated organism graphs, so unlike masking it does NOT transfer.

Round context and the arm results: [[experiments.019-simb-multimodal.expression-round-retrospective]]; the $|S|=1$ derivation with line refs: [[experiments.019-simb-multimodal.multiplicative-perturbation-conditioning]].
