---
id: qvn92igwmzytitfv83t8ee8
title: Construction Validation Handoff
desc: ''
updated: 1784691672638
created: 1784691672638
---

## 2026.07.21 - Handoff: fold the construction+validation doubles into the wet-lab plate

Instructions for the next Cloud Code session to pick this up cold. Everything below is
committed to `origin/main` (tip `43d6edab` at handoff). Entry-point note:
[[experiments.010-kuzmin-tmi.scripts.construction_validation_doubles]].

### What this is

A double-mutant construction list for the **echo-plating SGA fitness assay** being developed
in `experiments/019-echo-crispr-array/` (that experiment currently lives ONLY on the M1 Mac —
NOT yet pushed; see Blockers). The list serves two goals at once: (a) reconstruct the top-ranked
trigenic-interaction triples of the constructed genes, and (b) span DMF / interaction variance so
the new assay has real signal to validate against a public reference (Costanzo2016).

### The gene set (read this first — three panels are easy to confuse)

- **inference_3 panel-12** = the model-selected panel ([[experiments.010-kuzmin-tmi.scripts.topk_triples_from_constructed_10]]).
- **the 10 "constructed" genes** = inference_3 panel-12 MINUS YIL174W and LCL2/YLR104W:
  YBR203W(COS111), YDR057W(YOS9), YER079W, YGL087C(MMS2), YJR060W(CBF1), YKL033W-A,
  YLL012W(YEH1), YLR312C-B, YPL046C(ELC1), YPL081W(RPS9A). All selection here is over these 10.
- **the wet-lab plate-12 (exp-019)** = the 10 + SPH1/YLR313C + LCL1/YPL056C (two swaps). SPH1 and
  LCL1 have NO model predictions, so they are not in the double selection.

### Deliverables (all committed, `experiments/010-kuzmin-tmi/`)

- Selection script: `scripts/construction_validation_doubles.py` (re-run: reads committed CSVs +
  writes CSV+figures; `~/miniconda3/envs/torchcell/bin/python scripts/construction_validation_doubles.py`).
- **SI table (the canonical output)**: `results/construction_validation_doubles.csv` — ALL 45
  within-10 pairs = C(10,2); `tier` blank when unselected (coverage / validation / novel otherwise);
  Costanzo DMF±SD + derived SE + ε + p, AND Kuzmin2018/2020 DMF±SD side by side.
- Figures: `notes/assets/images/010-kuzmin-tmi/construction_validation_doubles.{svg,png}` (DMF×ε
  scatter) and `…_forest.{svg,png}` (all 44 measured doubles ranked by DMF, tier-colored).

### The list to construct (13 selected + 1 novel = 14)

- **8 coverage** doubles reconstruct all 31 within-10 top-k triples (the set-cover).
- **5 validation** doubles add dynamic range + the 3 significant interactions: YER079W+YJR060W
  (low-DMF anchor, 0.610±0.018, covers 3), YER079W+YPL081W (ε −0.130 ✓), YJR060W+YKL033W-A
  (ε −0.082 ✓), YDR057W+YGL087C (ε +0.098 ✓, covers 3), YDR057W+YLR312C-B (high-DMF anchor).
- **1 novel**: **YPL046C+YPL081W (ELC1+RPS9A)** — the ONE pair unmeasured in Costanzo, Kuzmin,
  SynthLethDB, AND BioGRID/SGD; NOT synthetic-lethal → build it to fill the 10×10 matrix.

### Facts the assay comparison depends on (do not re-derive wrong)

- Costanzo **DMF** SD = **sample SD over 4 colonies** → directly comparable to the assay's colony
  SD. Costanzo **SMF** "std" = **bootstrap SE** (NOT comparable). Typed columns already exist.
- **Kuzmin disagrees with Costanzo** for the CBF1 doubles (up to ~0.5, e.g. YJR060W+YLL012W:
  Costanzo 0.605 vs Kuzmin2020 1.119). Report BOTH references for low-DMF/CBF1 targets.
- **Depth beats SD**: SE = SD/√n and n (colonies) is ours to choose; plate a few doubles deep
  rather than many shallow. Interesting *point estimates* are chosen here; resolving them to
  significance is a bench sampling decision.

### Next actions ("add into the fold")

1. [ ] **Push exp-019 from the M1** so a GilaHyper/cloud session can read the ECHO picklist and
   the plate gene identities (blocker for everything below).
2. [ ] **Merge this 14-double list into the exp-019 plate/ECHO-picklist design** — add the 8+5
   selected doubles and the novel ELC1+RPS9A; carry the `tier` so the plate map records why each
   double is present.
3. [ ] **Confirm the LCL2→LCL1 swap intent** on the plate (LCL1/YPL056C was never in the model
   scoring; it may be a gene-identity swap worth double-checking against design intent).
4. [ ] For low-DMF/CBF1 validation doubles, put the Kuzmin value beside Costanzo on the plate map.
5. [ ] (optional) figure→script manifest for code+image submission.

### Blockers / open

- exp-019 is M1-only (unpushed) — cannot design the plate integration from a cloud session until pushed.
- ELC1+RPS9A "not a *known* SL" ≠ proven non-SL (no source tested it); low risk, flagged.

Related: [[experiments.010-kuzmin-tmi.scripts.construction_validation_doubles]],
[[experiments.010-kuzmin-tmi.scripts.constructed_10_dmf_reference]],
[[experiments.010-kuzmin-tmi.scripts.topk_triples_from_constructed_10]].
