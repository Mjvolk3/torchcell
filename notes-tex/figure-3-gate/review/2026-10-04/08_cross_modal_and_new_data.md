# Reviewer 8 of 10: cross-modal evidence and new data (2026-10-04)
Read-only audit by an independent agent; hypotheses, web sources and memory are labeled as such.

I've finished the audit. The short version: more single-deletion expression data can't move the metric enough to detect in 20 days. Public data adds about 245 to 460 strains on top of 1,484. Below, WT = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective, R19 = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results, R28 = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/028-knockout-expression/results, JR = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/review/2026-09-27-joint-review, F3 = /Users/michaelvolk/Documents/projects/torchcell.worktrees/fig3-priority-candidates/experiments/database/scripts/build_candidate_datasets_table.py, SP = /private/tmp/claude-501/-Users-michaelvolk-Documents-projects-torchcell/0e7a8bd2-1c56-433f-848a-c25de6ee5824/scratchpad/spell (scratch scripts), MAIN = /Users/michaelvolk/Documents/projects/torchcell. Full paths are at the end.

**1. ESTABLISHED**

- **Morphology triangle** (R19/proteome_morphology_covariation.json, expression_morphology_covariation.json):
  - Proteome to morphology: median per-feature r 0.169 over all 278 features, 0.280 on the 116 "moving" ones; n=4,314 strains. With PC1 removed: 0.118.
  - Expression to morphology: 0.215 all, 0.324 moving; n=1,438.
  - Reverse directions are weak: morphology to proteome 0.081, morphology to expression 0.087. Nulls are about 0.
  - So "0.22 to 0.32" is the best direction on selected features, not the general level.
- **Proteome and expression** (JR/01_data_ceilings.md §4, `part2.json`; n=1,349 strains):
  - Proteome to expression 0.226 and expression to proteome 0.218, against genotype alone at 0.101 and 0.056.
  - Adding genotype to the observed modality gains +0.009 / -0.004. The two channels barely overlap.
- **Cross-study conditioning** (R19/cross_study_conditioning_oracle.json, 82 shared strains, 5 draws):
  - At m=1000: within-study 0.78; Kemmeren-to-Sameith 0.48; Sameith-to-Kemmeren 0.48.
  - About 38% of the within-study conditioning is tied to the individual array.
  - Separately, conditioning gain survives removing a genotype predictor (97.5 to 100.6%; R19/conditioning_gain_after_genotype.json).
- **Gene co-variation** (R28/gene_covariation_all.json): Kemmeren vs Caudal Spearman 0.470 on 1,645 protein-measured genes and 0.393 on 4,337 expression genes. Kemmeren vs Sameith is 0.645; Caudal vs Nadal is -0.016.
- **Nadal-Ribelles** (R28/cross_study_recomputed.json): per-strain median r against Kemmeren 0.003 (914 shared deletions); 0.008 to 0.013 under recomputed statistics.
- **Learning curve** (JR/01 §5, `part6.json`, one split, fixed penalty):
  - Expression: 622 to 1,244 training strains gives +0.021.
  - Proteome: 895 to 1,790 gives +0.017; 1,790 to 3,581 gives +0.009. The proteome curve is flattening.
- **Amino acids** (R28/expression_metabolic_yko_correlation.json, ridge out of fold, median over metabolites):
  - Messner proteome to Mülleder 0.409 (n=4,400) vs to Cooper 0.082 (n=3,912).
  - Kemmeren expression to Mülleder 0.179 (n=1,416) vs to Cooper 0.096 (n=1,317).
  - Mülleder vs Cooper directly: Spearman -0.006 to 0.059 over about 3,750 to 4,070 genes. Cooper's own duplicate strains agree at only -0.021 to 0.133 (39 to 47 pairs) (MAIN/notes/experiments.034-showcase-datasets.scripts.amino_acid_betaxanthin.md). Cooper doesn't replicate itself.
- **SPELL count** (SP/classify.py; 603 studies, 778 PCL files, 16,160 columns):
  - **Method:** a column counts as a genetic perturbation if it names a gene (via the SGD R64-4-1 GFF) plus a perturbation marker (Δ, del, null, ts allele, overexpression, tet), or is a bare gene name in a study whose README says deletion or mutant.
  - **Result:** 3,918 columns in **200 studies** (159 with at least 3 columns), covering **1,175 distinct genes**. 697 are among Kemmeren's 1,484 deleted genes (read from the LMDB); Sameith's genes all lie inside Kemmeren. **478 are new.**
  - **Where the new genes come from:** Mnaimneh 2004 tet-promoter knockdowns of essential genes, 216; Hughes 2000, 146; Komili 2007, 10; Hu 2007, only 14. 393 of the 478 come from 14 compendium studies.
  - **Overlap:** Hughes shares 124 genes with Kemmeren; Hu shares 255.
  - **Error rates:**
    - Precision: 60/60 sampled positives are true perturbations. 3/60 picked up a spurious gene (MET3 promoter, URA3 marker, "set1" as a replicate label).
    - Recall: 11/80 sampled negatives were missed perturbations, about 14% (roughly 1,500 columns, wide interval), mostly in studies already flagged.
    - A liberal rule gives 599 new genes, but at much lower precision.
  - **Multi-gene in SPELL beyond Sameith:** 21 explicit doubles (van Wageningen, Verzijlbergen, Apweiler) plus about 10 TF doubles in Carter 2007.
- **Fitness overlap** (R19/fig3_overlap_census.json): 1,360 Kemmeren strains have single-mutant fitness; 57 of the 72 Sameith doubles have double-mutant fitness; 770,498 double-mutant genotypes in total.

**2. NOT ESTABLISHED OR CONTRADICTED**

- No measured fitness-to-expression or fitness-to-proteome relationship exists in R19 or R28. The "slow-growth axis" in the proteome-morphology script is PC1 labelled as growth, never checked against fitness.
- Morphology vs either amino-acid metabolome: not measured.
- The 0.226/0.218 triangle and the learning curve exist only as scratch files on the remote machine (`part*.json`), not committed. Both rest on one split.
- Proteome-to-Mülleder 0.41 may be inflated: both are Ralser-lab prototrophic collections (from memory). Hypothesis, untested; no plate-matched null was run.
- "Agreement" of gene co-variation shows shared gene modules. It does not show that genotype-to-expression will transfer across panels.
- Distinct allele count across the Caudal isolates and the cost of embedding them: never measured.

**3. ERRORS AND INCONSISTENCIES**

- The Kemmeren-vs-Caudal "0.47" is the protein-gene subset; on all expression genes it is 0.393.
- The genotype-alone "0.06" for proteome is on the 1,349 shared strains; on the full set it is 0.076 (JR/01).
- The covariation ridges pick λ=10,000, the top of their grid (R19/*_covariation.json), so the reported r may be slightly low.
- F3's Figure 3 set says the Hu and Hughes overlaps are "not yet counted". They are now: Hu shares 255 of 269 (14 new), Hughes 124 (146 new). Hu 2007 is ranked 5th but adds almost no new strains.
- Mnaimneh 2004, the largest SPELL source of new genes, is absent from F3.
- The SPELL metadata CSV covers 568 of 603 studies, with mean extraction confidence 0.176. Its mutant_strain tag hits 66 studies against 200 by my rule.
- SPELL's Kemmeren entry holds only wild-type arrays (672 columns).
- The Hughes SPELL file is KNN-imputed and averaged (`flt.knn.avg`), so it isn't raw data.
- The Caudal loader drops FY4-6, an S288C derivative that could serve as the in-panel reference.
- All embedding builders (protT5.py, esm2.py, nucleotide_transformer.py, codon_language_model.py) key on `genome[gene_id]` for the S288C sequence. None accepts an allele sequence.

**4. FEASIBILITY TABLE** (days and value are my judgment; expected value is a hypothesis)

| Dataset | Exists | Missing | Days | Expected value (hypothesis) | Main risk |
|---|---|---|---|---|---|
| Hughes 2000 | SPELL PCL locally | loader, ploidy, raw GEO | 2–3 | +146 strains, about +0.003 | imputed matrix, other lab |
| Mnaimneh 2004 | SPELL PCL | promoter-knockdown genotype | 2–3 | +216 essential genes, about +0.004 | not deletions |
| Hu 2007 | SPELL PCL | loader | 2 | about 0 new; replication only | redundant |
| Caudal relative to reference | loader on main; store remote | reference record, re-centering | 1–2 | gene-structure signal for the decoder | out-of-distribution genotypes |
| Per-allele embeddings | none | refactor, allele table, GPU | 5–8 | unknown | promoter variants not captured |
| Teyssonniere / Muenzner | candidate rows only | loaders; possibly the same data twice | 5+ each | isolate proteome | duplication |
| Albert 2018 | Bloom 2019 segregant schema on main | loader | 5–8 | segregant expression | BY×RM is a different input type |
| Jakobson 2025 | candidate row | everything | >10 | proteome | won't land |
| Fitness (O'Duibhir 1,312 / Costanzo) | loaders on main | analysis | 0.5–1 | tests the growth axis | |
| Perturb-seq multi-gene | costing docs | guide-detection pilot ($4,493), multi-guide pilot | >20 | combinations | Nadal-style non-replication |

**5. TOP THREE RECOMMENDATIONS**

1. **Do a one-day "do more strains help" test first.**
   - Action: rerun the ridge learning curve over 5 resampled splits, then project the gain from +245 and +460 strains.
   - Check: compare the projected gain with the between-partition SD of the trained model (0.013 to 0.035, JR/01).
   - Cost: 1 day.
   - Stop if the projection is below 0.01. On the current curve it is about +0.004 to +0.008 (hypothesis), so I expect this test to stop the deletion-loader work.
2. **Test fitness as a signal with an oracle bound before using it for pretraining.**
   - Action: add observed O'Duibhir growth (measured in the same cultures as Kemmeren, 1,312 strains) or Costanzo fitness as an input to the genotype-to-expression ridge, and also regress it out.
   - Control: a shuffled-fitness input.
   - Cost: 1 day.
   - Stop result: if observed fitness adds less than +0.01, a fitness-pretrained trunk can't gain more, at least linearly.
3. **Use the isolates for gene-gene structure, not for the genotype mapping.**
   - Action: rebuild Caudal relative to a reference, with FY4-6 as the reference record. Then fit the expression head's covariance or low-rank prior on Caudal and evaluate genotype-to-expression on Kemmeren validation.
   - Controls: a Kemmeren-only prior and a random prior.
   - Cost: 3 days.
   - Stop result: below +0.01.

   In parallel, the cheapest replication gate for any SPELL loader is Hu vs Kemmeren on the 255 shared deletions (1 day). Drop SPELL if the per-strain median r approaches Nadal's 0.003. Per-allele embeddings, the isolate proteomes, Jakobson and Perturb-seq can't land in 20 days.

Files:
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/cross_study_conditioning_oracle.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/028-knockout-expression/results/expression_metabolic_yko_correlation.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/028-knockout-expression/results/gene_covariation_all.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/review/2026-09-27-joint-review/01_data_ceilings.md
/Users/michaelvolk/Documents/projects/torchcell.worktrees/fig3-priority-candidates/experiments/database/scripts/build_candidate_datasets_table.py
/Users/michaelvolk/Documents/projects/torchcell/torchcell/datasets/scerevisiae/caudal2024.py
/Users/michaelvolk/Documents/projects/torchcell/torchcell/datasets/scerevisiae/oduibhir2014.py
/private/tmp/claude-501/-Users-michaelvolk-Documents-projects-torchcell/0e7a8bd2-1c56-433f-848a-c25de6ee5824/scratchpad/spell/classify.py
/private/tmp/claude-501/-Users-michaelvolk-Documents-projects-torchcell/0e7a8bd2-1c56-433f-848a-c25de6ee5824/scratchpad/spell/cols.json

## Appendix: SPELL classifier (evidence for the SPELL count)

Inputs were prepared by two companion scratch scripts in SP: `base.py` reads the Kemmeren 2014 (1,484 records, 1,484 genes), Sameith 2015 Sm (82) and Dm (72 records, 82 genes) LMDBs under /Users/michaelvolk/Documents/projects/torchcell/data/torchcell/ (keys `experiment.genotype.perturbations[].systematic_gene_name`) and builds a standard-name to systematic-name map from /Users/michaelvolk/Documents/projects/torchcell/data/sgd/genome/S288C_reference_genome_R64-4-1_20230830/saccharomyces_cerevisiae_R64-4-1_20230830.gff (7,129 systematic IDs, 15,247 names); `dump.py` reads the README and PCL header of every one of the 603 study directories under /Users/michaelvolk/Documents/projects/torchcell/data/sgd/spell/ (778 PCL files, 16,160 condition columns).

Command:

```bash
SP=/private/tmp/claude-501/-Users-michaelvolk-Documents-projects-torchcell/0e7a8bd2-1c56-433f-848a-c25de6ee5824/scratchpad/spell
/Users/michaelvolk/miniconda3/envs/torchcell/bin/python $SP/base.py
/Users/michaelvolk/miniconda3/envs/torchcell/bin/python $SP/dump.py
/Users/michaelvolk/miniconda3/envs/torchcell/bin/python $SP/classify.py
```

Script (`classify.py`):

```python
import json, re, html, random, collections
OUT='/private/tmp/claude-501/-Users-michaelvolk-Documents-projects-torchcell/0e7a8bd2-1c56-433f-848a-c25de6ee5824/scratchpad/spell'
S=json.load(open(f'{OUT}/studies.json')); B=json.load(open(f'{OUT}/base.json'))
nm=B['nm']; known=set(B['known'])
kemset=set(g for r in B['kem'] for g in r)
GENE=re.compile(r'(?<![a-z0-9])(y[a-p][lr]\d{3}[wc](?:-[a-z](?![a-z]))?|[a-z]{3}\d{1,3}[ab]?)(?![0-9])')
MARK=re.compile(r'(δ|deleion|anchor.?away|[a-z]{3}\d{1,3}d(?![a-z])|\b[a-z]{3}\d{1,3}[a-z]{3}\d{1,3}\b|[a-z]{3}\d{1,3}[ -][a-z]\d+[a-z]\b|del\b|-del|delet|delta|Δ|∆|knock|\bko\b|mutant|\bmut\b|null|overexp|over-exp|\boe\b|\bop\b|gal1?-?pr|pgal|gal-|tet[- ]?(off|o|promoter)|degron|-ts\b|\bts\b|-aid|::|disrupt|allele|[a-z]{3}\d{1,3}-\d+|[a-z]{3}\d{1,3}d\b|[a-z]{3}\d{1,3}\^|\b[a-z]{3}\d{1,3}/[a-z]{3}\d{1,3}\b)')
README_MARK=re.compile(r'(deletion|deleted|knock-?out|mutant|overexpress|tet-?promoter|promoter-shutoff|strains lacking|null)',re.I)
TIMEWORDS=re.compile(r'(\bmin\b|\d+ ?min|\bhr?\b|\d+ ?h\b|time|\bt\d+|heat|stress|mm\b|\bnm\b|%|glucose|galactose|nitrogen|carbon|limit|alpha|arrest|cycle|drug|rapamycin|h2o2|dtt|sorbitol|nacl|mms|temperature|\d+c\b|°)',re.I)
NONGENE={'wt','ref'}
cols=[]
for st,v in S.items():
    rm=bool(README_MARK.search(v['readme']))
    for p,pv in v['pcls'].items():
        for c in pv['cols']:
            raw=c; s=html.unescape(c).lower()
            genes=set()
            for m in GENE.finditer(s):
                t=m.group(1)
                if t in nm: genes.add(nm[t])
                elif t[-1] in 'ab' and t[:-1] in nm: genes.add(nm[t[:-1]])
            mark=bool(MARK.search(s))
            stripped=re.sub(r'(rep(licate)?\.?\s*\d+|[-_ ]?\b[abcd]\b|\(haploid\)|\(diploid\)|\b\d+\b|\*+|[",;:/()\[\]\-_.#])',' ',s).split()
            gene_only = genes and all((t in nm) or t in ('and','vs','wt','x','strain') for t in stripped) and len(stripped)<=4
            cond=bool(TIMEWORDS.search(s))
            if genes and mark: cls='gene+marker'
            elif gene_only and rm: cls='gene_only+readme'
            elif genes and rm: cls='liberal_gene+readme'
            else: cls='none'
            cols.append(dict(study=st,pcl=p,col=raw,genes=sorted(genes),cls=cls,cond=cond,readme_mark=rm))
json.dump(cols,open(f'{OUT}/cols.json','w'))
pos=[c for c in cols if c['cls'] in ('gene+marker','gene_only+readme')]
lib=[c for c in cols if c['cls']!='none']
lg=set(g for c in lib for g in c['genes'])
print('LIBERAL cols',len(lib),'studies',len(set(c['study'] for c in lib)),'genes',len(lg),'new',len(lg-kemset))
print('columns',len(cols),'positive',len(pos), collections.Counter(c['cls'] for c in pos))
st_pos=collections.Counter(c['study'] for c in pos)
print('studies with >=1 positive col',len(st_pos), 'with >=3',sum(1 for v in st_pos.values() if v>=3))
genes=set(g for c in pos for g in c['genes'])
print('distinct genes',len(genes),'in kemmeren',len(genes&kemset),'new',len(genes-kemset))
nocond=[c for c in pos if not c['cond']]
g2=set(g for c in nocond for g in c['genes'])
print('positives without condition/time words',len(nocond),'studies',len(set(c['study'] for c in nocond)),'genes',len(g2),'new',len(g2-kemset))
# per study genes new
per=collections.defaultdict(set)
for c in pos:
    for g in c['genes']: per[c['study']].add(g)
top=sorted(per.items(),key=lambda x:-len(x[1]-kemset))[:30]
for s,g in top: print(s,len(g),len(g-kemset),st_pos[s])
```

Printed summary (`base.py`, then `dump.py`, then the head of `classify.py`):

```text
kem records 1484 genes 1484 sm 82 82 dm 72 82 union 1484
systematic 7129 names 15247
603 778 16160
LIBERAL cols 5336 studies 248 genes 1322 new 599
columns 16160 positive 3918 Counter({'gene+marker': 3363, 'gene_only+readme': 555})
studies with >=1 positive col 200 with >=3 159
distinct genes 1175 in kemmeren 697 new 478
positives without condition/time words 3039 studies 169 genes 967 new 341
```

Per-study table of the 14 compendium studies (at least 10 distinct perturbed genes under the strict rule); together they contribute 393 of the 478 new genes, the other 85 come from the remaining strict-positive studies:

| study | distinct perturbed genes | new beyond Kemmeren |
|---|---|---|
| Mnaimneh_2004_PMID_15242642 | 217 | 216 |
| Hughes_2000_PMID_10929718 | 270 | 146 |
| Hu_2007_PMID_17417638 | 269 | 14 |
| Komili_2007_PMID_17981122 | 11 | 10 |
| Albulescu_2012_PMID_22479188 | 11 | 5 |
| Chua_2006_PMID_16880382 | 76 | 5 |
| Apweiler_2012_PMID_22697265 | 89 | 3 |
| Lenstra_2013_PMID_24324601 | 10 | 3 |
| van_Wageningen_2010_PMID_21145464 | 133 | 2 |
| Huang_2002_PMID_12077337 | 12 | 1 |
| Garcia-Oliver_2013_PMID_23599000 | 16 | 0 |
| Lenstra_2011_PMID_21596317 | 162 | 0 |
| Sameith_2015_PMID_26700642 | 82 | 0 |
| van_de_Pasch_2013_PMID_23785440 | 17 | 0 |

Audit of the classifier: precision on 60 random strict positives was 60/60 correct class, with 3/60 columns carrying a spurious extra gene; recall on 80 random negatives found 11 missed genetic perturbations (about 14 percent). Gene counts carry these errors.
