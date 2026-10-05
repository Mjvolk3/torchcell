# Reviewer 5 of 10: architecture and perturbation operator, from the code (2026-10-04)
Read-only audit by an independent agent; every hypothesis is labeled "Hypothesis (untested)", everything else was read from code or measured by the CPU probes in the appendix.

Legend (abbreviations used below):

- M = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/torchcell/models/equivariant_cell_graph_transformer.py
- W30 M = /Users/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-per-entry-dataset-token/torchcell/models/equivariant_cell_graph_transformer.py
- T = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/train_cgt_multitask.py
- C = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/conf/
- methods.tex = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/paper/nature-biotech/sections/methods.tex
- Fig 1 = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes/assets/drawio/Fig1-torchcell-overview.drawio.svg
- multiplicative note = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes/experiments.019-simb-multimodal.multiplicative-perturbation-conditioning.md
- expression-fit-review note = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes/experiments.019-simb-multimodal.expression-fit-review.md
- Probes: /tmp/rev5/count.py, /tmp/rev5/grad.py, /tmp/rev5/time.py (reproduced in the appendix)

## 1. ESTABLISHED FROM CODE

- **Effective v19 config** (following the `defaults:` chain C/cgt_expr_v19_joint_clean.yaml → v16 → v14 → v13 → v12 → v11 → v9 → 012 → 011 → 010 → 008 → 006 → embed_005 → decoder_003):
  - d=90, 6 layers, 9 heads.
  - Embeddings `[fudt_upstream, calm, prot_T5_all, fudt_downstream]` concatenated to 3,328 dims; no learnable embedding.
  - Hard attention mask on layer 1 only; graph-reg λ=0.
  - Operator: post-LN, 1 layer, `null_sink` off, sum pooling.
  - Perceiver with 32 latents, gate fixed on; observed-label encoder on.
  - `mask_schedule [0]`; heads `per_gene` + `per_gene_aux` with 19 quantile parameters each.
- **(1) Parameters, instantiated:** 6,681,548 in total.
  - Embedding preprocessor 5,846,759 (87.5%). Its first layer alone, `Linear(3328→1709)`, is 5.69M (85%) (M:2191-2213).
  - Encoder 590,220. Operator 98,370. Perceiver 101,430.
  - Interaction head 16,381; each per-gene head 9,919; observed-label encoder 8,460.
  - The 6.67M figure is v13, which has one head; v19 is 6.68M.
- **(2) What reaches gene i at one deletion p.**
  - The encoder runs once at batch 1 on the wild-type graph (M:2928, 2994-2996), so h_i and h_CLS are the same for every strain.
  - The operator's keys and values are only `H[p]` (M:688). At |S|=1 the softmax weight is identically 1 in eval mode (probe: max|c_i−c_0| = 0.0). So c_b = W_O(W_V h_p + b_V) + b_O is one vector shared by every gene.
  - Output is H_pert_i = LN2(x + FFN(x)), with x = LN1(h_i + c_b) (M:635-645).
  - Pair paths that do exist:
    - (a) The curvature of the shared LN, FFN and head MLP applied to h_i + c_b. Probe without the Perceiver: the strain-effect difference y_p − y_q has a standard deviation of 0.061 across reporters, against 0.163 for y itself. So the effect is not additive, but it is implicit.
    - (b) The Perceiver read step (M:1249), a softmax of q(h_i + c_b) against latents written from all strain tokens. This is a genuine query-by-strain product.
  - There is no explicit bilinear term, and nothing marks which gene was deleted. The hop-0 indicator exists only in the disabled propagation module (M:813).
  - In training, the operator's 0.1 attention dropout acts on that single weight of 1. It randomly drops whole heads of the deletion signal per gene: only 53 distinct context rows out of 300 queries.
  - **Null sink:** α_p = sigmoid(q_i·W_K h_p/√d_k − null_bias). The b_K term cancels. The sink value is b_V, not zero. One scalar bias is shared across all heads (M:538, 695-705).
- **(4) CLS token.**
  - On this branch the CLS is wild-type for every strain (M:2995).
  - Morphology `s1_pool` reads [wild-type CLS ; mean over all 6,607 H_pert tokens] (M:1413-1416). So morphology is read from the pool, not the CLS.
  - W30 `perturb_cls` (W30 M:2978-2991) adds the CLS as query row 0. At |S|=1 this gives h_CLS_pert = g(h_CLS + c_b), a function of c_b only.
  - It adds no information and no parameters. Gene rows never attend to it, and the Perceiver skips it.
  - Porting it needs about 30 lines: the constructor flag, the forward branch, and `GlobalHead.features` accepting a [B,d] CLS (W30 M:1455-1465). CrossAttnHead also assumes a [d] CLS (M:1516).
- **W30 differences that matter:**
  - W30 lacks `per_gene_aux`, `per_gene_weight`, `linear_readout`, `context_readout`, and the NaN-target masking in the loss (M:2018-2026).
  - W30 adds a configurable preprocessor `hidden_dim` (W30 M:2113), which is the knob for shrinking the 5.69M layer, plus a dataset token on the readout.
- **(5) Natural isolates.**
  - Gene features are a fixed reference [N,3328] taken from `cell_graph["gene"].x` (M:2898).
  - Per-strain embeddings would mean running the preprocessor plus a 6-layer dense encoder over 6,608 tokens per strain. Measured: 12.3 s per CPU forward against 2.6 s for operator, Perceiver and heads at B=32.
  - The cheaper route is variant tokens: the reference h_j plus φ(e_alt − e_ref). The Methods' type and magnitude tokens (methods.tex:242) are not implemented.
  - At thousands of keys, the softmax splits mass over K, roughly 1/K each, which destroys cardinality. Logits are 6,607×K×9 per strain (0.7 GB fp32 per strain at K=3,000) unless a fused kernel is used. The per-strain Python loop (M:677) serializes this.

## 2. CLAIMS THE CODE DOES NOT SUPPORT

- **Fig 1f** (decoded MathJax in Fig 1) shows β = σ((h_iW_Q)·(p_tW_K)/√d_k), a per-pair sigmoid.
  - The code runs a softmax (M:706). Methods eq. pert (methods.tex:256) matches the code, so Fig 1f and Methods contradict each other.
- **Methods, other mismatches:**
  - Encoder pre-LN (methods.tex:218-223); the code is post-LN (M:177, 181).
  - Graphs as a soft KL prior (methods.tex:192, 287); the code uses a hard mask on layer 1 with λ=0.
  - Typed perturbation tokens (methods.tex:242); not implemented.
  - h_CLS^pert in the interaction head and 1/|p| mean pooling (methods.tex:305); the code uses the wild-type CLS and sum pooling (M:1333, 1346).
  - Morphology pooled over a fixed gene set G_s (methods.tex:316); the code pools all genes.
  - Environment conditions every head (methods.tex:319); this is absent from the code.
- **M:399-401** says the encoder "closes with a ReZero residual"; it is post-LN.
- **M:755-757 and the multiplicative note at lines 43-44** say "no term depends on the pair (p, i)" and "rank 0". This is contradicted by the 0.061 reporter-dependent strain effect in the probe, and by the Perceiver.
- **M:977-979 and M:989** say the observed-label encoder is the identity when everything is masked. In fact proj([0,0]) = W2·ReLU(b1) + b2 ≠ 0, so it adds a learned constant.
- **M:1434** says the S3 head reads `h_CLS_pert`; it reads the wild-type CLS.

## 3. BUGS, DEAD PARAMETERS, RISKS

- **High:** `_dump_split_predictions` calls `task(batch)` with no observed values (T:2378), which skips the observed-label constant (M:3027). Training and validation always pass zeros through `_masked_step` (T:1463, 1282).
  - At init, dumped predictions differ by up to 0.237, against a prediction sd of 0.161.
  - Every `val-predictions`/`test-predictions` JSON from a v9+ run is therefore from a different function than the logged metric.
- **High (dead parameters, measured as zero gradient):**
  - Operator W_Q, W_K, b_Q, b_K: 16,380.
  - Interaction head, which is inactive: 16,381.
  - Observed-label first layer: 180.
- **Medium:**
  - `mask_schedule [0]` makes `_masked_step` run an extra no-grad forward pass every step (T:1254).
  - Operator attention dropout zeroes whole heads of the only strain signal.
  - The CLS row and column are unmasked (M:2620-2621), which gives every masked head a 2-hop global bypass.
- **Low:** `num_parameters` omits the aux head, the Perceiver and the observed-label encoder, so it reports 6,561,739 instead of 6,681,548 (M:3154-3180).

### Fundamental assumptions

| Assumption | Status |
|---|---|
| A1. Encode the wild type once and perturb afterwards (M:2928) | Untested. No per-strain re-encoding arm exists. |
| A2. Softmax over perturbed keys (M:706) | Contradicted as a selector at \|S\|=1. Its sigmoid alternative (null sink) was underpowered: +0.0024 ± 0.0143, n=4 (multiplicative note, lines 85-91). |
| A3. Frozen sequence embeddings through one large projection (M:2178) | Contradicted-leaning. The comment at M:1650-1652 cites a parameter-free prot_T5 kNN at 0.117 against the model's 0.080; I did not verify this. |
| A4. The nine graphs act only as masks on layer 1 | Untested in v19. |
| A5. One shared per-gene MLP (M:1665) | Alternatives exist (M:1702-1766) but are off in v19. |
| A6. The CLS is wild-type | Contradicted: sd 0.0 across strains (M:2399-2401). |
| A7. All 6,607 tokens are processed per strain | Unnecessary except for the Perceiver, because the operator and head are per-token. |
| A8. The deleted gene is unmarked | Contradicted by evidence: predicted +0.036 against a true −2.42 (expression-fit-review note, line 21). |

## 4. DESIGN OPTIONS (ranked by expected value over cost)

1. **Explicit pair readout.**
   - Head input [H_pert_i ; h_i⊙c_i ; 1{i∈S}], with c_i = Σ_t σ(q_i·k_t/√d + b)·v_t, i.e. sigmoid gating with no normalization over keys.
   - About 8K extra parameters per head and no extra compute. It handles 1, n or thousands of keys and is cardinality-aware.
   - Code changes: M:686-711 (the custom sigmoid attention), M:3061-3075 (head input), M:2393 (`in_mult`).
2. **Shrink the input side.**
   - Frozen PCA-128 of the embeddings followed by `Linear(128,90)`: 5.85M becomes 12K.
   - Or W30's `hidden_dim` set to 64: 5.85M becomes about 219K.
3. **Cache the encoder output and subsample reporters.**
   - Freeze or cache H. Without the Perceiver, a per-gene output depends only on (h_i, h_p), so reporter subsampling is exact.
   - Measured on CPU: full v19 step 42 s; cached encoder 2.6 s (about 90 s per epoch); 512 reporters with no Perceiver 0.23 s (about 8 s per epoch).
   - At d=32 the operator plus head is about 15K parameters. GPU times are unmeasured; Hypothesis (untested): well under 1 s per epoch.
4. **Variant tokens for isolates.** The reference h_j plus a delta embedding, using the sigmoid gate from option 1. Requires an allele embedding cache.

## 5. TOP THREE RECOMMENDATIONS

**R1 (days 1-3): a discriminating small-model harness.**

- **Change:**
  - Cache the e_i features.
  - Fit three models on the fixed split at n = 137, 275, 550 and 1,100 strains, with 5 split seeds: (a) per-gene mean; (b) Kronecker bilinear ridge Y ≈ E_R W E_Pᵀ with a 64×64 W, in closed form; (c) kNN.
  - Add a planted-signal control: labels E_R W* E_Pᵀ at the measured noise ceiling.
- **Reading the outcome:**
  - The ridge validation score keeps rising with n → too few strains.
  - Train far above validation at the best ridge → overfitting.
  - Ridge no better than kNN, while the planted signal is recovered → the pair term carries no signal in these embeddings.
  - Planted signal not recovered by the CGT → the architecture fails.
- **Expected effect:** Hypothesis (untested): the slope is still positive at n=1,100.
- **Stop:** a learning-curve slope CI that excludes 0, or one that does not.

**R2 (days 4-10): option 1 combined with options 2 and 3, as a sweep.**

- **Control:** the identical small model without the pair features. Use 5 initialization seeds at a fixed split, plus 3 split seeds.
- **Expected effect:** Hypothesis (untested): validation +0.02 to +0.05, mostly from the self-indicator.
- **Stop:** adopt if the paired Δ CI excludes 0; otherwise declare the pair term not the bottleneck.

**R3 (days 1-2): fix and align.**

- **Change:**
  - Fix the dump bug and re-dump the existing checkpoints.
  - Drop the inactive interaction head.
  - Make Methods, Fig 1f and the code name one operator (the sigmoid).
  - Fix the `num_parameters` count.
- **Control:** dumped Pearson must equal the logged validation Pearson.
- **Stop:** they agree to 1e-4.

Files:

/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/torchcell/models/equivariant_cell_graph_transformer.py
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/train_cgt_multitask.py
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/paper/nature-biotech/sections/methods.tex
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes/assets/drawio/Fig1-torchcell-overview.drawio.svg
/Users/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-per-entry-dataset-token/torchcell/models/equivariant_cell_graph_transformer.py
/tmp/rev5/count.py
/tmp/rev5/grad.py
/tmp/rev5/time.py

## Appendix: probe scripts, commands and printed output

All three were run on the M1 Mac CPU with the `torchcell` env (Python 3.13). PyTorch deprecation warnings are filtered out of the outputs shown.

### A1. count.py (parameter breakdown of the v19 model)

Command:

```bash
PYTHONPATH=/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective /Users/michaelvolk/miniconda3/envs/torchcell/bin/python /tmp/rev5/count.py 2>&1 | tail -30
```

Script:

```python
import torch, types
from torch_geometric.data import HeteroData
from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer
N=6607
g=HeteroData(); g["gene"].num_nodes=N; g["gene"].x=torch.randn(N,3328)
def ds(d):
    o=types.SimpleNamespace(embeddings={"e":torch.zeros(1,d)}); return [o]
ne={"fudt_upstream":ds(768),"calm":ds(768),"prot_T5_all":ds(1024),"fudt_downstream":ds(768)}
heads={"per_gene":{"output_dim":1,"param_dim":19},"per_gene_aux":{"output_dim":1,"param_dim":19}}
m=CellGraphTransformer(gene_num=N,hidden_channels=90,num_transformer_layers=6,num_attention_heads=9,cell_graph=g,
  graph_regularization_config=None,perturbation_head_config=dict(num_heads=9,dropout=0.1,residual="postln",num_layers=1,ffn_mult=4,hadamard="off",null_sink=False,pooling="sum"),
  dropout=0.1,graph_reg_lambda=0.0,node_embeddings=ne,learnable_embedding_config=dict(enabled=False,size=90,preprocessor=dict(num_layers=2,dropout=0.1)),
  heads_config=heads,post_perturbation_mixing_config=dict(enabled=True,num_latents=32,gate_mode="on"),
  observed_label_config=dict(enabled=True,gate_mode="on"),perturbation_propagation_config=dict(enabled=False))
tot=0
for n,c in m.named_children():
    k=sum(p.numel() for p in c.parameters()); tot+=k; print(f"{n:32s}{k:>10,}")
print("cls",m.cls_token.numel()); print("TOTAL",sum(p.numel() for p in m.parameters()))
for n,p in m.embedding_preprocessor.named_parameters(): print(" pre",n,tuple(p.shape))
print(m.num_parameters)
torch.save(m.state_dict(),"/tmp/rev5/sd.pt")
```

Output:

```text
embedding_preprocessor           5,846,759
transformer_layers                 590,220
perturbation_transform              98,370
perturbation_head                   16,381
per_gene_head                        9,919
per_gene_aux_head                    9,919
observed_label_encoder               8,460
post_perturbation_mixing           101,430
cls 90
TOTAL 6681548
 pre 0.weight (1709, 3328)
 pre 0.bias (1709,)
 pre 1.weight (1709,)
 pre 1.bias (1709,)
 pre 4.weight (90, 1709)
 pre 4.bias (90,)
 pre 5.weight (90,)
 pre 5.bias (90,)
{'gene_embedding': 0, 'embedding_preprocessor': 5846759, 'cls_token': 90, 'transformer_layers': 590220, 'perturbation_transform': 98370, 'perturbation_head': 16381, 'per_gene_head': 9919, 'total': 6561739}
```

The per-embedding widths (768, 768, 1024, 768; sum 3,328) are an assumption that matches the 3,328 → 1,709 → 90 preprocessor shape reported in the expression-fit-review note; they were not read from the built embedding datasets.

### A2. grad.py (dead gradients, attention-dropout query dependence, reporter-dependent strain effect, dump-vs-validation discrepancy)

Command (first run, before the final dump-vs-validation block was appended):

```bash
PYTHONPATH=/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective /Users/michaelvolk/miniconda3/envs/torchcell/bin/python /tmp/rev5/grad.py 2>&1 | grep -v Warn | grep -v warn
```

Second run, after appending the last block; only its final line was kept:

```bash
PYTHONPATH=/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective /Users/michaelvolk/miniconda3/envs/torchcell/bin/python /tmp/rev5/grad.py 2>&1 | tail -1
```

Script (final form):

```python
import torch, types
torch.manual_seed(0)
from torch_geometric.data import HeteroData
from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer
N=300
g=HeteroData(); g["gene"].num_nodes=N; g["gene"].x=torch.randn(N,3328)
ne={"a":[types.SimpleNamespace(embeddings={"e":torch.zeros(1,3328)})]}
heads={"per_gene":{"output_dim":1,"param_dim":19},"per_gene_aux":{"output_dim":1,"param_dim":19}}
def build(mix=True, sink=False):
    torch.manual_seed(0)
    return CellGraphTransformer(gene_num=N,hidden_channels=90,num_transformer_layers=6,num_attention_heads=9,cell_graph=g,
      graph_regularization_config=None,perturbation_head_config=dict(num_heads=9,dropout=0.1,null_sink=sink,null_sink_bias_init=0.0,pooling="sum"),
      dropout=0.1,graph_reg_lambda=0.0,node_embeddings=ne,learnable_embedding_config=dict(enabled=False,size=90,preprocessor=dict(num_layers=2,dropout=0.1)),
      heads_config=heads,post_perturbation_mixing_config=dict(enabled=mix,num_latents=32,gate_mode="on"),
      observed_label_config=dict(enabled=True,gate_mode="on"))
def batch(idx):
    b=HeteroData(); b["gene"].perturbation_indices=torch.tensor(idx); b["gene"].perturbation_indices_batch=torch.arange(len(idx)); return b
m=build(); m.train()
B=8; z=torch.zeros(B,N)
_,r=m(g,batch(list(range(10,10+B))),observed_values=z,observed_mask=z)
loss=sum((v*torch.randn_like(v)).sum() for v in r["head_outputs"].values()); loss.backward()
dead=0
for n,p in m.named_parameters():
    if p.grad is None or p.grad.abs().max()==0:
        print("ZERO/None grad:",n,tuple(p.shape),p.numel()); dead+=p.numel()
    elif "perturbation_transform.cross_attn_layers.0.in_proj" in n:
        gq=p.grad[:90].abs().max().item(); gk=p.grad[90:180].abs().max().item(); gv=p.grad[180:].abs().max().item()
        print(n,"Q",gq,"K",gk,"V",gv)
print("dead params",dead)
# attention-dropout query dependence in train mode at |S|=1
H=torch.randn(N,90)
pt=m.perturbation_transform; pt.train()
_,ctx=pt(H,torch.tensor([5]),torch.tensor([0]))
c=ctx[0]; print("train: distinct context rows", torch.unique(c.round(decimals=5),dim=0).shape[0], "of",N)
pt.eval(); _,ctx=pt(H,torch.tensor([5]),torch.tensor([0])); print("eval: max |c_i-c_0|",(ctx[0]-ctx[0][0]).abs().max().item())
# pair interaction: double difference at per_gene output (eval)
for mix in [False,True]:
  for sink in [False,True]:
    m=build(mix,sink); m.eval()
    with torch.no_grad():
        _,r=m(g,batch([5,7]),observed_values=torch.zeros(2,N),observed_mask=torch.zeros(2,N))
        y=r["head_outputs"]["per_gene"][...,9]  # median knot
        d=y[0]-y[1]  # strain effect per gene
        print(f"mix={mix} sink={sink}: sd over reporters of (y_p - y_q) = {d.std():.4g}; sd of y = {y.std():.4g}")
m=build(); m.eval()
with torch.no_grad():
    b=batch([5,7]); _,r1=m(g,b,observed_values=torch.zeros(2,N),observed_mask=torch.zeros(2,N)); _,r2=m(g,b)
    a=r1["head_outputs"]["per_gene"][...,9]; c=r2["head_outputs"]["per_gene"][...,9]
    print("dump-vs-val max abs diff at init", (a-c).abs().max().item(), "sd y", a.std().item())
```

Output (first run):

```text
perturbation_transform.cross_attn_layers.0.in_proj_weight Q 0.0 K 0.0 V 40.314910888671875
perturbation_transform.cross_attn_layers.0.in_proj_bias Q 0.0 K 0.0 V 18.237775802612305
ZERO/None grad: perturbation_head.mlp.0.weight (90, 180) 16200
ZERO/None grad: perturbation_head.mlp.0.bias (90,) 90
ZERO/None grad: perturbation_head.mlp.3.weight (1, 90) 90
ZERO/None grad: perturbation_head.mlp.3.bias (1,) 1
ZERO/None grad: observed_label_encoder.proj.0.weight (90, 2) 180
dead params 16561
train: distinct context rows 53 of 300
eval: max |c_i-c_0| 0.0
mix=False sink=False: sd over reporters of (y_p - y_q) = 0.06144; sd of y = 0.1631
mix=False sink=True: sd over reporters of (y_p - y_q) = 0.03553; sd of y = 0.1535
mix=True sink=False: sd over reporters of (y_p - y_q) = 0.08096; sd of y = 0.1608
mix=True sink=True: sd over reporters of (y_p - y_q) = 0.04477; sd of y = 0.1648
```

Output (second run, final line):

```text
dump-vs-val max abs diff at init 0.23703423142433167 sd y 0.16083306074142456
```

Notes. N=300 genes stands in for 6,607 to keep the probe fast. The script's "dead params 16561" counts only parameters whose whole tensor gets zero gradient. The operator's Q/K rows (16,380 parameters) live inside the shared `in_proj` tensors, so they are reported separately by the Q/K/V lines rather than added to that sum. All probe numbers are at random init, not from a trained checkpoint.

### A3. time.py (CPU timings per component, B=32, N=6,607)

Command:

```bash
PYTHONPATH=/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective /Users/michaelvolk/miniconda3/envs/torchcell/bin/python /tmp/rev5/time.py 2>&1 | grep -v -i warn
```

Script:

```python
import torch, time, torch.nn as nn
from torchcell.models.equivariant_cell_graph_transformer import GraphRegularizedTransformerLayer, EquivariantPerturbationTransform, PerceiverMixing, PerGeneHead
torch.manual_seed(0); N=6607; d=90; B=32
def T(f,rep=2):
    f(); t=time.time()
    for _ in range(rep): f()
    return (time.time()-t)/rep
X=torch.randn(N,3328)
pre=nn.Sequential(nn.Linear(3328,1709),nn.LayerNorm(1709),nn.GELU(),nn.Linear(1709,d),nn.LayerNorm(d))
enc=nn.ModuleList(GraphRegularizedTransformerLayer(d,9) for _ in range(6))
op=EquivariantPerturbationTransform(d,9); mix=PerceiverMixing(d,32,9,gate_mode="on"); head=PerGeneHead(d,1,param_dim=19)
def trunk():
    H=pre(X); H=torch.cat([torch.zeros(1,d),H]).unsqueeze(0)
    for l in enc: H,_=l(H)
    return H[0,1:]
def full():
    H=trunk(); Hp,_=op(H,torch.arange(B)*7,torch.arange(B)); y=head(mix(Hp)); y.sum().backward()
Hc=trunk().detach()
def cached():
    Hp,_=op(Hc,torch.arange(B)*7,torch.arange(B)); y=head(mix(Hp)); y.sum().backward()
def cached_sub(R=512):
    idx=torch.randperm(N)[:R]; Hs=Hc[idx]
    Hp,_=op(Hs,torch.arange(B)%R,torch.arange(B)); y=head(Hp); y.sum().backward()
with torch.no_grad(): print("preproc fwd s",T(lambda: pre(X)))
with torch.no_grad(): print("trunk fwd s",T(trunk))
print("full step fwd+bwd s",T(full,1))
print("cached-encoder step (op+perceiver+head) s",T(cached))
print("cached, 512 reporters, no perceiver s",T(cached_sub))
```

Output:

```text
preproc fwd s 0.05256640911102295
trunk fwd s 12.32420301437378
full step fwd+bwd s 42.19218611717224
cached-encoder step (op+perceiver+head) s 2.5943539142608643
cached, 512 reporters, no perceiver s 0.22933614253997803
```

Notes. These are CPU times; the attention mask is off and dropout is in train mode. Per-epoch figures in the report assume 35 steps per epoch (1,100 training strains at batch 32, from the v19 config comment). GPU times were not measured.
