# Reviewer 7 of 10: data, splits and metrics (2026-10-04)
Read-only audit by an independent agent (no repository file modified, no training run); every untested claim is labeled "Hypothesis (untested)".

**Reviewer 7 report: data, splits and metrics**

The headline numbers are not yet trustworthy. One real bug and two leakage-type risks are untested, and the evaluation inflates results through the order statistic, the lucky split 0 and unmatched baselines. I modified nothing. My checks ran on the local Kemmeren and Sameith LMDBs, with scratch files under `/tmp/r7_*.py`.

## 1. ESTABLISHED

- **Unit of splitting.** It is the aggregated genotype record. The data module shuffles each index key with `random.seed(split_seed)` in an 80/10/10 split (`torchcell/datamodules/cell.py:308-309, 386-410`). Overlapping records are reassigned once (`:439-488`). `split_seed` is separate from the init seed (`train_cgt_multitask.py:2680-2688`).
- **Validation sizes per split seed.** Expression has 151-155 strains (`split_indices_manifest.json`). Proteome has 448 of 477-484 records (`split_indices_manifest_fig3_proteome.json`).
- **`require_modalities` filters after the partition is drawn** (`train_cgt_multitask.py:2767-2791`). So the v19 arms (protein and expression required) score expression on about 125 strains with no doubles, while v16/v17 score 155. The denominators differ (`split_gene_overlap_audit.json`, `both` = 119-126).
- **Standardization is fit on train only** (`:3114-3128`, `:742-773`, nanmean and nanstd). NaN proteome entries go to the sparse Pearson, which needs at least 3 finite pairs (`:314-330`).
- **Duplicate key ignores environment, background and allele.** It is experiment type plus gene names only (`mean_experiment_deduplicate.py`, `duplicate_check`). All 82 Sameith single mutants are mean-merged with their Kemmeren twin (measured: 82/82 genotypes shared, identical 6,169-key sets). Exact duplicates across splits: 0 (audit).
- **Gene-sharing rows.** 14-21 of 155 expression validation strains share a gene with train, all single-versus-double relations through the 72 Sameith doubles (audit, `fig3_core_expr`).
- **Self-knockout entry is in target and metric.** The decode does no masking (`:1026-1175`). Measured on Kemmeren: the deleted gene's own value is present in 1,479 of 1,484 strains, median log2 ratio -2.48, and 86% of them are below -1. For comparison, the median strain's 99th percentile of |other genes| is 0.56.
- **Media differ.** Expression is SC (all three LMDBs). Messner is SM, HIS3-complemented (`messner2023.py:53, 344`).
- **Caudal 2024.** The reference is the population mean of absolute TPM (`caudal2024.py:33-36`). FY4-6 is dropped (`:415`).

## 2. NOT ESTABLISHED OR CONTRADICTED

- **"Incumbent clears the bilinear by 0.093, 4.2 sd"** (`1-findings.tex:45-46`) is contradicted.
  - The baselines used the numpy "oracle-family" permutation, not the trained arms' partition (`expression_baselines.py:41-48`).
  - On matched split 0, the bilinear baseline (B2) scores 0.135 on validation and ranks 1 of 12 split draws (`expression_baselines_split/summary.json`).
  - The model figure, 0.1965, is a `roll_max` averaged over 8 arms.
- **"Baselines match the model"** holds on validation only roughly. On test (v17 `L_ref` at its best-validation checkpoint, 3 seeds per split) the model reads 0.113/0.169/0.154/0.143 for s0-s3. The best matched baseline per split reads 0.085/0.162/0.124/0.142. The mean edge is about +0.017, and most of it comes from s0.
- **Does the model actually predict the self-knockdown?** Not measured.

## 3. BUGS, LEAKAGE RISKS AND INCONSISTENCIES

- **HIGH: self-knockout reporter in the metric.**
  - A predictor that only puts -2.48 at the deleted gene and nothing elsewhere scores mean per-feature r **0.0175 ± 0.0013** over 50 random 155-strain draws.
  - That equals the model's test edge over the matched baselines. Hypothesis (untested): part of the CGT advantage is this entry.
  - Cheap test: rescore the dumped predictions (`_dump_split_predictions`) with the self entries set to NaN.
- **HIGH: v16 expression baselines are from the wrong partition.** `v16_joint_expr_readout.json["baselines"]` equals the `fig3_core` values (s0 B2 validation 0.1349), but `fig3_proteome` partitions overlap only 9/155 at seed 0 (audit, `v13_v16_overlap`).
- **HIGH: constant-column dropping depends on the model.**
  - `per_feature_pearson` drops any column where the denominator is at most 1e-8 (`:307`), so the set of features scored can differ between arms.
  - The self-only predictor with exact zeros elsewhere reads **0.72** on the 155 columns it kept.
  - Test: log the number of valid features per arm.
- **MEDIUM: stale split cache.** `index_seed_{k}.json` is reused if it exists, with no check against dataset length or content, and a load error silently regenerates it (`cell.py:362-382`). Test: assert the stored genotype hash.
- **MEDIUM: two different "278" morphology feature sets.**
  - The ceiling and the decoder/optuna configs drop A113_A, D203 and D205 (`morphology_noise_ceiling.json`, `cgt_decoder_003.yaml:115`).
  - The gh/igb/train configs drop A113_A1B, A113_C and C123_C (`train_cgt_multitask.yaml:102`).
- **MEDIUM: conditioned arms log to the same key as unconditioned ones.** v20 feeds the same strain's other-modality labels as input and logs `val/<head>/pearson_per_feature` (`:1403-1456`). This is not a genotype-only score.
- **LOW: per-rank Pearson under DDP.** Pearson is computed per rank and then averaged (`:1615-1616`). This affects only the 4-GPU `gh_cgt_multitask_*` configs.
- **Not audited: paralogs and protein complexes across the split.** The audit checks shared genes only.

## 4. EVALUATION CHANGES (ranked)

1. **Matched baselines and model on the same strains, on test, from saved checkpoints.** Report window means, never `roll_max`. The `roll_max` minus window gap is +0.009 for v17 expression and up to +0.034 for v16 proteome s0.
2. **Self-knockout entries blanked**, and doubles plus gene-sharing rows reported separately.
3. **A fixed, reliability-restricted feature set** decided before scoring:
   - Proteome: drop the 28% of proteins with reliability 0 (`01_data_ceilings.md:83`).
   - Morphology: 162 of 278 features have reliability at or below 0.5 and 24 have 0 (`morphology_feature_ceiling.csv`).
   - Use a fixed denominator.
4. **Splits grouped by gene family and complex, with a sealed test set.**

## 5. TOP THREE RECOMMENDATIONS

1. **Rescore everything already trained with one harness (days 1-3, CPU plus a few GPU hours).**
   - Covers items 4.1-4.3 for v16-v18 checkpoints and the v13/v14 checkpoints that were never tested.
   - Hypothesis (untested): the model's test edge over baselines falls from about 0.017 toward 0.
   - Stopping result: if the 4-partition CI of model minus baseline includes 0, Figure 3 reports parity and architecture work stops.
2. **Ten grouped partitions with a sealed test set, two arms by two seeds, about 1,200 epochs each (days 3-14).**
   - Measured inputs: between-partition sd is 0.028-0.030 (v13/v17 readouts) and the partition-level arm sd is about 0.017.
   - Minimum detectable effect at 80% power, two-sided 0.05: about 0.036 at P=4, 0.024 at P=6, 0.017 at P=10.
   - A sign-flip test cannot reach p<0.05 with only 4 partitions.
   - Stopping result: the test-set contrast at P=10 is reported once, whatever it shows.
3. **Pipeline fixes before any new build (days 1-4).**
   - Hash-check the split cache.
   - Unify the morphology drop set.
   - Separate the conditioned metric key.
   - Record Kemmeren and Sameith as replicates instead of mean-merging them.
   - Keep Caudal out of the deletion pool until it carries an in-batch reference: FY4-6 or a declared S288C-like strain, log2 with a pseudocount, NaN for accessory genes, ploidy recorded.
   - Stopping result: all audit scripts pass with the new checks.

**Numbers I would not trust:**
- 0.1965 and its 0.093 / 4.2 sd margin
- any `roll_max`
- the v16 expression-versus-baseline comparison
- the v19 versus v16/v17 expression comparison
- the morphology fraction of ceiling (0.135, from a 65-epoch `roll_max`)
- every split-0 validation figure

Files:
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/train_cgt_multitask.py
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/torchcell/datamodules/cell.py
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/torchcell/data/mean_experiment_deduplicate.py
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/expression_baselines.py
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v16_joint_expr_readout.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/split_gene_overlap_audit.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/expression_baselines_split/summary.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/morphology_noise_ceiling.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/conf/cgt_decoder_003.yaml
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/torchcell/datasets/scerevisiae/caudal2024.py
/tmp/r7_self.py
/tmp/r7_keys.py
/tmp/r7_kem.py

## Path legend (abbreviated names used above)

Line references written as `:NNN` with no file name refer to the file named most recently before them in the same bullet; bare `:NNN` in sections 1 and 3 without a preceding name refer to `train_cgt_multitask.py`, except `:439-488` (cell.py) and `:415` (caudal2024.py).

| abbreviation | absolute path |
|---|---|
| `train_cgt_multitask.py` | /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/train_cgt_multitask.py |
| `torchcell/datamodules/cell.py`, `cell.py` | /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/torchcell/datamodules/cell.py |
| `mean_experiment_deduplicate.py` | /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/torchcell/data/mean_experiment_deduplicate.py |
| `messner2023.py` | /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/torchcell/datasets/scerevisiae/messner2023.py |
| `caudal2024.py` | /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/torchcell/datasets/scerevisiae/caudal2024.py |
| `expression_baselines.py` | /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/expression_baselines.py |
| `split_indices_manifest.json` | /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/split_indices_manifest.json |
| `split_indices_manifest_fig3_proteome.json` | /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/split_indices_manifest_fig3_proteome.json |
| `split_gene_overlap_audit.json`, "audit" | /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/split_gene_overlap_audit.json |
| `expression_baselines_split/summary.json` | /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/expression_baselines_split/summary.json |
| `v16_joint_expr_readout.json` | /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v16_joint_expr_readout.json |
| v13/v17 readouts | /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v13_split_readout.json and /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v17_locality_readout.json |
| `morphology_noise_ceiling.json` | /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/morphology_noise_ceiling.json |
| `morphology_feature_ceiling.csv` | /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/morphology_feature_ceiling.csv |
| `cgt_decoder_003.yaml` | /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/conf/cgt_decoder_003.yaml |
| `train_cgt_multitask.yaml` | /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/conf/train_cgt_multitask.yaml |
| `1-findings.tex` | /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/sections/1-findings.tex |
| `01_data_ceilings.md` | /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/review/2026-09-27-joint-review/01_data_ceilings.md |

## Appendix: scratch scripts and their printed output

All three read the local LMDBs under /Users/michaelvolk/Documents/projects/torchcell/data/torchcell/ (built July 2026, which may predate the current loaders). Outputs below are from a rerun on 2026-10-04 for this file. `/tmp/r7_keys.py` is shown as run after a one-line sed edit that added the pickle fallback (its first run failed on a pickled record). `/tmp/r7_self.py` includes two lines appended after the first run (the 82-strain responsiveness check).

### A1. /tmp/r7_kem.py (self-knockout value in its own strain)

Command:

```bash
/Users/michaelvolk/miniconda3/envs/torchcell/bin/python /tmp/r7_kem.py
```

```python
import lmdb, json, pickle, numpy as np
p="/Users/michaelvolk/Documents/projects/torchcell/data/torchcell/microarray_kemmeren2014/processed/lmdb"
env=lmdb.open(p,readonly=True,lock=False)
with env.begin() as txn:
    print("entries",txn.stat()["entries"])
    cur=txn.cursor(); 
    self_vals=[];other_abs=[];n=0;missing=0; nperts=[]
    for k,v in cur:
        try: d=json.loads(v)
        except Exception: d=pickle.loads(v)
        if n==0: print(type(d), list(d.keys()) if isinstance(d,dict) else None); print(str(d)[:600])
        e=d["experiment"]; g=[pp["systematic_gene_name"] for pp in e["genotype"]["perturbations"]]
        nperts.append(len(g))
        ph=e["phenotype"]; r=ph["expression_log2_ratio"]
        vals=np.array([x for x in r.values() if x is not None and np.isfinite(x)])
        for gg in g:
            if gg in r and r[gg] is not None and np.isfinite(r[gg]): self_vals.append(r[gg]); other_abs.append(np.percentile(np.abs(vals),99))
            else: missing+=1
        n+=1
print("n",n,"nperts",np.bincount(nperts),"self present",len(self_vals),"missing",missing)
s=np.array(self_vals); print("self log2 ratio median",np.median(s),"q10",np.percentile(s,10),"q90",np.percentile(s,90),"frac < -1",np.mean(s<-1))
print("median per-strain 99th pct |other|",np.median(other_abs))
```

Output:

```
entries 1484
<class 'dict'> ['experiment', 'reference', 'publication']
{'experiment': {'experiment_type': 'microarray_expression', 'dataset_name': 'MicroarrayKemmeren2014Dataset', 'genotype': {'perturbations': [{'systematic_gene_name': 'YJL095W', 'perturbed_gene_name': 'YJL095W', 'provenance': 'engineered', 'state': 'absent', 'mechanism_so_id': 'SO:0000159', 'mechanism_so_name': 'deletion', 'description': 'Deletion via KanMX or NatMX gene replacement', 'perturbation_type': 'kanmx_deletion', 'deletion_description': 'Deletion via KanMX gene replacement.', 'deletion_type': 'KanMX'}]}, 'environment': {'provenance_gaps': [], 'media': {'name': 'SC', 'state': 'liquid', 
n 1484 nperts [   0 1484] self present 1479 missing 5
self log2 ratio median -2.4786809222452844 q10 -4.078660634543458 q90 -0.6920069601967352 frac < -1 0.8553076402974983
median per-strain 99th pct |other| 0.5555857360339419
```

### A2. /tmp/r7_keys.py (key sets, media and duplicate genotypes across Kemmeren and Sameith)

Command:

```bash
/Users/michaelvolk/miniconda3/envs/torchcell/bin/python /tmp/r7_keys.py
```

```python
import lmdb, json, numpy as np, collections
base="/Users/michaelvolk/Documents/projects/torchcell/data/torchcell/"
sets={}
for name in ["microarray_kemmeren2014","sm_microarray_sameith2015","dm_microarray_sameith2015"]:
    env=lmdb.open(base+name+"/processed/lmdb",readonly=True,lock=False)
    sizes=collections.Counter(); keysets=collections.Counter(); genes=[]; nan=[]
    allk=None
    with env.begin() as txn:
        for k,v in txn.cursor():
            import pickle
            try: d=json.loads(v)
            except Exception: d=pickle.loads(v)
            if isinstance(d,list): d=d[0]
            e=d["experiment"]; r=e["phenotype"]["expression_log2_ratio"]
            ks=tuple(sorted(r)); sizes[len(ks)]+=1; keysets[hash(ks)]+=1
            allk = set(ks) if allk is None else allk
            genes.append(tuple(sorted(p["systematic_gene_name"] for p in e["genotype"]["perturbations"])))
            nan.append(sum(1 for x in r.values() if x is None or not np.isfinite(x)))
            sets.setdefault(name,set()).update(ks)
            env_media=e["environment"]["media"]["name"]
    print(name,"n",len(genes),"sizes",dict(sizes),"distinct keysets",len(keysets),"nan per rec median",np.median(nan),"max",max(nan),"media",env_media,"dup genotypes",len(genes)-len(set(genes)))
    sets[name+"_genos"]=set(genes)
k=sets["microarray_kemmeren2014"]; s=sets["sm_microarray_sameith2015"]; dm=sets["dm_microarray_sameith2015"]
print("kem keys",len(k),"sm keys",len(s),"sm==kem",s==k,"dm==kem",dm==k, "kem-sm",len(k-s),"sm-kem",len(s-k), "union",len(k|s))
kg=sets["microarray_kemmeren2014_genos"]; sg=sets["sm_microarray_sameith2015_genos"]
print("sm genotypes also in kemmeren",len(kg&sg),"of",len(sg))
```

Output:

```
microarray_kemmeren2014 n 1484 sizes {6169: 1484} distinct keysets 1 nan per rec median 0.0 max 0 media SC dup genotypes 0
sm_microarray_sameith2015 n 82 sizes {6169: 82} distinct keysets 1 nan per rec median 0.0 max 0 media SC dup genotypes 0
dm_microarray_sameith2015 n 72 sizes {6169: 72} distinct keysets 1 nan per rec median 0.0 max 0 media SC dup genotypes 0
kem keys 6169 sm keys 6169 sm==kem True dm==kem True kem-sm 0 sm-kem 0 union 6169
sm genotypes also in kemmeren 82 of 82
```

### A3. /tmp/r7_self.py (self-only predictor score, cross-study ceiling with and without self entries)

Command:

```bash
/Users/michaelvolk/miniconda3/envs/torchcell/bin/python /tmp/r7_self.py
```

```python
import lmdb, json, pickle, numpy as np
base="/Users/michaelvolk/Documents/projects/torchcell/data/torchcell/"
def load(name):
    env=lmdb.open(base+name+"/processed/lmdb",readonly=True,lock=False)
    G=[];Y=[];keys=None
    with env.begin() as txn:
        for k,v in txn.cursor():
            try: d=json.loads(v)
            except Exception: d=pickle.loads(v)
            if isinstance(d,list): d=d[0]
            e=d["experiment"]; r=e["phenotype"]["expression_log2_ratio"]
            if keys is None: keys=sorted(r)
            G.append(tuple(sorted(p["systematic_gene_name"] for p in e["genotype"]["perturbations"])))
            Y.append([r[x] for x in keys])
    return G,np.array(Y,float),keys
Gk,Yk,keys=load("microarray_kemmeren2014"); Gs,Ys,_=load("sm_microarray_sameith2015")
col={k:i for i,k in enumerate(keys)}
def pf(P,T,drop_const=True):
    pc=P-P.mean(0);tc=T-T.mean(0);num=(pc*tc).sum(0);den=np.linalg.norm(pc,axis=0)*np.linalg.norm(tc,axis=0)
    v=den>1e-8; r=num[v]/den[v]; return r.mean(), v.sum()
# 1) cross-study ceiling on 82 shared, with and without self entries
ik={g:i for i,g in enumerate(Gk)}
idx=[ik[g] for g in Gs]
A=Yk[idx].copy();B=Ys.copy()
print("cross-study per-feature r (82 shared), with self:",pf(A,B))
for j,g in enumerate(Gs):
    for gg in g:
        if gg in col: A[j,col[gg]]=np.nan;B[j,col[gg]]=np.nan
# nan-> column mean replace for simplicity
def fill(M):
    M=M.copy(); m=np.nanmean(M,0); i=np.where(np.isnan(M)); M[i]=np.take(m,i[1]); return M
print("cross-study per-feature r (82 shared), self blanked:",pf(fill(A),fill(B)))
# 2) self-only oracle on random 155-strain draws of Kemmeren
rng=np.random.default_rng(0); res=[];res0=[];frac=[]
for rep in range(50):
    s=rng.choice(len(Gk),155,replace=False); T=Yk[s]; P=np.zeros_like(T)
    for j,i in enumerate(s):
        for gg in Gk[i]:
            if gg in col: P[j,col[gg]]=T[j,col[gg]]
    m,nv=pf(P,T); res.append(m)
    # counted over all columns (constant->0)
    pc=P-P.mean(0);tc=T-T.mean(0);num=(pc*tc).sum(0);den=np.linalg.norm(pc,axis=0)*np.linalg.norm(tc,axis=0)
    r=np.where(den>1e-8,num/np.where(den>1e-8,den,1),0); res0.append(r.mean())
    # fraction of total per-column variance from self entries
    tot=(tc**2).sum(); selfv=sum((tc[j,col[gg]])**2 for j,i in enumerate(s) for gg in Gk[i] if gg in col); frac.append(selfv/tot)
print("self-only oracle: mean r over non-constant cols",np.mean(res),"n cols",nv,"; averaged over all 6169 cols",np.mean(res0),"sd",np.std(res0))
print("fraction of total centered target SS from self entries",np.mean(frac))
# realistic: model = self-drop guess of -2.48 constant at the self entry
res1=[]
for rep in range(50):
    s=rng.choice(len(Gk),155,replace=False); T=Yk[s]; P=np.zeros_like(T)+rng.normal(0,1e-3,T.shape)
    for j,i in enumerate(s):
        for gg in Gk[i]:
            if gg in col: P[j,col[gg]]=-2.48
    pc=P-P.mean(0);tc=T-T.mean(0);num=(pc*tc).sum(0);den=np.linalg.norm(pc,axis=0)*np.linalg.norm(tc,axis=0)
    r=num/den; res1.append(r.mean())
print("constant -2.48 at self entry + tiny noise elsewhere, mean r over all cols:",np.mean(res1),np.std(res1))
sd=Yk.std(1); sh=set(idx)
print("per-strain sd median: 82 Sameith-shared",np.median(sd[idx]),"other Kemmeren",np.median(np.delete(sd,idx)))
# per-feature variance across 82 vs random 82
rv=[]
for rep in range(20):
    s=rng.choice(len(Gk),82,replace=False); rv.append(np.median(Yk[s].var(0)))
print("median per-gene var across 82 shared",np.median(Yk[idx].var(0)),"random 82",np.mean(rv))
```

Output:

```
cross-study per-feature r (82 shared), with self: (np.float64(0.6110432525511217), np.int64(6169))
cross-study per-feature r (82 shared), self blanked: (np.float64(0.6074746419427016), np.int64(6169))
self-only oracle: mean r over non-constant cols 0.7235232167989462 n cols 155 ; averaged over all 6169 cols 0.018115785947384323 sd 0.0004966168436198745
fraction of total centered target SS from self entries 0.03283041446548259
constant -2.48 at self entry + tiny noise elsewhere, mean r over all cols: 0.017540762446991674 0.001281543264356955
per-strain sd median: 82 Sameith-shared 0.18762518221550858 other Kemmeren 0.18603844807141873
median per-gene var across 82 shared 0.01606452995259379 random 82 0.019842558689280445
```
