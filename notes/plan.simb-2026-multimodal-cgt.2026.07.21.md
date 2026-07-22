---
id: simb2026multimodalcgt0721
title: SIMB 2026 Multimodal CGT Roadmap
desc: 'Query → convert/dedup/agg → subset → model-expand → train the multimodal phenotype results (Figs 3/4/6) for the SIMB 2026 abstract'
updated: 1784691637024
created: 1784691637024
---

Roadmap to substantiate the SIMB 2026 abstract's multimodal-phenotype claims
(Figs 3, 4, 6). Each `## WS<n>` below is a self-contained workstream = ONE GitHub issue linking
back to this note (`(issue: WS<n>)` swaps to `#N` once cut). Work them **linearly
by phase**; within a phase the dependency graph is noted per-WS. Time flows
top→bottom (CLAUDE.md convention); append dated reconciliation H2s, never rewrite.

**Deadline:** SIMB 2026 — results needed in ~10 days (2026-07-31), submit ~12
(2026-08-02). Scope is deliberately narrow: substantiate the abstract's
*"same architecture predicts expression (single+double KO) and morphology (single
KO)"* (Fig 3) and *"recommend gene deletions for β-carotene / betaxanthin"* (Fig
6), plus the natural-variation contrast (Fig 4). **Fig 5 (robustness) and the
enzyme-constrained-FBA / kcat North Star are OUT of scope** and parked at the
bottom.

## Progress

Legend: ✅ done · 🔨 in progress · ⬜ not started · ⏸ deferred (North Star)

| WS     | Title                                                       | Phase | Status | Issue / PR           |
|--------|-------------------------------------------------------------|-------|--------|----------------------|
| WS0    | Live-DB verification + head/media truth table               | —     | ✅      | (this note, Context) |
| WS1    | Phenotype dedup/convert branches + graph_processor coverage | A     | ✅      | PR #161 → 29814103   |
| WS2    | Fig 3 unified build (expr + morphology + fitness)           | A     | ✅      | PR #162 → c8387cd9   |
| WS2b   | Aggregator gene-set key + Perturbation processor for build  | A     | ✅      | PR #164 → 349bc54b (expr+fit 90→1416) |
| WS3    | Fig 4 build (Caudal natural isolates + doubles)             | A     | ⬜      | (issue: WS3)         |
| WS4    | Fig 6 build (β-carotene + betaxanthin + metabolome)         | A     | ✅      | PR #165 → a0fe1de3   |
| WS5    | Subset indices (modality / leave-out)                       | A     | ⬜      | (issue: WS5)         |
| WS6    | Media/environment ε-conditioning                            | A     | ⬜      | (issue: WS6)         |
| WS7    | Multi-head decoder (per-gene / global / per-metabolite)     | B     | ✅      | PR #158 → 5c6c17fa   |
| WS8    | Metabolite readout + metabolic-graph annotation             | B     | ⬜      | (issue: WS8)         |
| WS9    | Natural-isolate embeddings (ESM2 + FUDT)                    | B     | 🔨      | ESM2 done (#159); FUDT FEASIBLE (WS9b #168 → 5d0300a6); full 943 run pending |
| WS10   | Individual per-phenotype models (fast baselines)            | C     | 🔨      | WS10a smoke ✅ PR #166 → 018bd0c8; full cluster runs next |
| WS11   | Joint / leave-out ablations ("does more data help")         | C     | ⬜      | (issue: WS11)        |
| WS12   | Fig 6 strain-design inference (deletion ranking)            | C     | ⬜      | (issue: WS12)        |
| WS13   | Compute prep (IGB mmli/cabbi + Delta SLURM + configs)       | C     | ✅      | PR #163 → 4c10f694   |
| WS14   | Error-bar provenance fix (trigenic τ ±0.006)                | D     | ✅      | PR #157 → cd3b928d   |
| WS-NS1 | Additive perturbation operator (inverse design)             | NS    | ⏸      | (issue: WS-NS1)      |
| WS-NS2 | Enzyme-constrained FBA regularizer + kcat/KM                | NS    | ⏸      | (issue: WS-NS2)      |

## Context — verified 2026-07-21

**The served NCSA DB is the full multimodal build, and it already holds every
dataset the abstract needs.** `neo4j+s://torchcell-database.ncsa.illinois.edu:7687`
(db name `torchcell`, read-only creds `readonly`/`ReadOnly`) has 30 labels,
34 `Dataset` nodes, 2,947,850 `Experiment` nodes. Confirmed present: Kemmeren2014,
Sm+Dm Sameith2015, Mülleder2016, Caudal2024, Ohya2005 (+Ohnuki 2018/2022),
Ozaydin2013, Cachera2023, Zelezniak2018 (proteome+metabolome), da Silveira2014,
all Costanzo/Kuzmin fitness+interaction, gene-essentiality + SL/SR. **Only gap:
Messner2023 proteome is built+L0–L4 but NOT wired into the adapter map → absent
from the graph. It is an *auxiliary* proteome anchor, not on the critical path.**

**Head mapping is already encoded by the graph's `graph_level` property** — no
guessing needed. This is the single most useful discovery: the phenotype's
`graph_level` in the KG maps 1:1 onto the user's predicted readout type.

| Dataset (Fig)                 | id                                  | n_exp | `graph_level` → head              | media / temp |
|-------------------------------|-------------------------------------|------:|-----------------------------------|--------------|
| Kemmeren2014 expr (3)         | `MicroarrayKemmeren2014Dataset`     | 1,484 | `node` → **per-gene**             | SC / 30      |
| Sameith Sm expr (3)           | `SmMicroarraySameith2015Dataset`    |    82 | `node` → per-gene                 | SC / 30      |
| Sameith Dm expr (3)           | `DmMicroarraySameith2015Dataset`    |    72 | `node` → per-gene                 | SC / 30      |
| Ohya morphology (3)           | `ScmdOhya2005Dataset`               | 4,718 | `global` → **CLS/global**         | **YPD / 25** |
| Caudal transcriptome (4)      | `CaudalPanTranscriptome2024Dataset` |   943 | `node` → per-gene                 | SC / 30      |
| Mülleder AA (6)               | `AminoAcidMulleder2016Dataset`      | 4,678 | `metabolism` → **per-metabolite** | SM / 30      |
| Zelezniak metab (6)           | `MetaboliteZelezniak2018Dataset`    |    95 | `metabolism` → per-metabolite     | SM / 30      |
| Zelezniak proteome            | `ProteomeZelezniak2018Dataset`      |    97 | `node` → per-gene                 | SM / 30      |
| Ozaydin β-carotene (6)        | `CarotenoidOzaydin2013Dataset`      | 4,474 | `global` → CLS                    | SC-URA / 30  |
| Cachera betaxanthin (6)       | `BetaxanthinCachera2023Dataset`     | 4,735 | `metabolism` → per-metabolite     | SC / 30      |
| da Silveira lipids ("SR VCA") | `MetaboliteDaSilveira2014Dataset`   |   127 | `metabolism` → per-metabolite     | YPD / 30     |

Three consequences drive the whole plan:

1. **Everything is small.** The largest per-phenotype set is ~4.7 k single-KO
   records; expression is tiny (1,484 + 82 + 72). **Individual models train in
   minutes** — the user's intuition is correct, so we baseline each phenotype
   alone first, cheaply, then test whether joining data helps.
2. **`graph_level` = the head type.** `global`→class-token (morphology, β-carotene
   score), `node`→per-gene MLP (expression, proteome), `metabolism`→per-metabolite
   (AA/betaxanthin/lipid). The model must grow these three heads (WS7/WS8); today
   it has exactly ONE (scalar gene-interaction).
3. **Media is bounded but Ohya is an outlier.** Only 4 (media,temp) combos across
   the 11 datasets; **Ohya morphology at YPD/25 °C is the sole real divergence**
   (everything else is 30 °C, SC/SM family). Individual models: media is constant,
   ignore. Joint models: condition the decoder on ε or it confounds
   temperature with dataset (WS6). Methods already spec `ŷ = R_φ(H_pert, ε)`.

**Model reality (verified in `equivariant_cell_graph_transformer.py`):**

- Nodes are **genes only**, fixed `N=6607`. A `cls_token` exists but currently
  feeds ONLY the GI head. Per-gene `H_genes_pert [B,N,d]` is produced but unused.
- The **metabolic bipartite graph is loaded into `cell_graph`** (reaction +
  metabolite nodes, `gpr`/`rmr` stoichiometry edges via `yeast_GEM.bipartite_graph`)
  **but the model skips all non-gene edges** (`_normalize_adjacency_matrices`
  filters `src/dst != "gene"`). Methods deliberately treat metabolism as a
  *representation annotation, never an attention prior* — so WS8 wires a metabolite
  readout, NOT a metabolic attention loss.
- The perturbation operator is **deletion/type-agnostic** (cross-attends to the
  perturbed genes' embeddings). It **cannot represent additions** (fixed `N`, no
  embedding rows for new entities, no perturbation-type signal). **This is fine for
  the conference:** the pigment screens are single-KO *deletions on a fixed
  engineered background* (the crt / betaxanthin cassette is constant across all
  ~4.5 k members → it belongs to the *reference cell*, not the perturbation).
  Dynamic additions are inverse-design North Star (WS-NS1).
- The "extremely slow subgraphing" is `SubgraphRepresentation` in
  `graph_processor.py` (per-sample `subgraph()` over every edge type). The
  equivariant CGT already avoids it: encode `G` once, inject perturbation indices
  late via the light `Perturbation` processor. **Do not reintroduce per-sample
  subgraphing** when adding phenotypes.

**Pipeline reality (verified in `neo4j_cell.py`):** the build is a pluggable
LMDB→LMDB chain `Cypher → Neo4jQueryRaw(raw) → Converter(conversion) →
Deduplicator(deduplication) → Aggregator(aggregation) → processed → graph_processor
→ HeteroData`. `GenotypeAggregator` is phenotype-agnostic and **is the enabler of
"does fitness help expression"** — it groups every phenotype measured on the same
genotype into one HeteroData instance (so a genotype can carry expression AND
fitness labels together). `MeanExperimentDeduplicator` currently **raises** on
anything but `fitness`/`gene interaction` → WS1.

**Where the results live:** the `paper.results-and-discussion.{1..6}.md` notes are
empty stubs. Authoritative content is the abstract, `paper/nature-biotech/sections/
{results,methods,backmatter}.tex` (Fig 3=`fig:multitask`, Fig 4=`fig:variation`,
Fig 6=`fig:metabolism`; **Supp Fig 7 = `fig:conversion-dedup-agg`**, the
conversion/dedup/agg schematic), and `paper.results-and-discussion.6.experimental-
plan.md` (the only one with real content — holds "Arm B: does the metabolome
improve isobutanol/betaxanthin prediction"). Fig-3 numbers exist in the abstract
(expr r=0.543±0.023, morph r=0.619±0.037) but **morphology is not yet reproducible
in-repo** and the trigenic ±0.006 error bar is an unresolved SE-vs-SD blocker
(WS14). Fig 4 and Fig 6 prose are entirely `[FILLER]`.

## Core Design Decisions

1. **Group builds by figure, not one-per-dataset, and subset with indices.**
   Per the user: query all the datasets we've added into a few figure-scoped
   builds, then carve targeted training sets with precomputed indices — do NOT mint
   a new dataset per experiment. Three builds: WS2 (Fig 3), WS3 (Fig 4), WS4 (Fig 6).
2. **`graph_level` selects the head; the decoder is multitask + masked.** One
   forward, N heads, loss masks to whichever phenotype(s) a genotype actually has
   (sparse supervision). Reuse `cls_token` for `global`, `H_genes_pert` for `node`,
   a new metabolite readout for `metabolism`.
3. **Deletion-only stays; cassette = reference cell.** No additive operator for the
   conference. β-carotene/betaxanthin tasks predict *over deletions* of a
   pathway-carrying reference strain.
4. **Metabolism enters as representation annotation, not attention prior** (matches
   Methods). Per-metabolite head maps to Yeast9 metabolite IDs via `gpr`/`rmr`
   incidence already in `cell_graph`.
5. **Individual-first, then joint.** Cheap per-phenotype baselines (WS10) establish
   the abstract numbers; joint leave-out runs (WS11) test the thesis that adding
   cheaper/denser data (fitness, metabolome, isolates) lifts a scarcer target.
6. **Minimal indices.** Add exactly two new index axes — `phenotype_class`
   (modality) and reuse `dataset_name` — enough to leave out metabolism / fitness /
   a modality. Resist adding more.

## Relevant Files

| Path                                                      | Action           | Purpose                                     |
|-----------------------------------------------------------|------------------|---------------------------------------------|
| `experiments/019-simb-multimodal/queries/*.cql`           | NEW              | Fig 3/4/6 UNION queries                     |
| `experiments/019-simb-multimodal/scripts/query.py`        | NEW              | Build runner (copy 010 pattern)             |
| `experiments/019-simb-multimodal/conf/*.yaml`             | NEW              | Hydra configs per run                       |
| `experiments/019-simb-multimodal/scripts/*.slurm`         | NEW              | IGB mmli/cabbi + Delta                      |
| `torchcell/data/mean_experiment_deduplicate.py`           | MODIFY           | Add phenotype branches (WS1)                |
| `torchcell/data/graph_processor.py`                       | REFERENCE/MODIFY | Confirm node/global/metabolism → HeteroData |
| `torchcell/data/neo4j_cell.py`                            | REFERENCE        | Build orchestration, indices                |
| `torchcell/datamodules/{cell,perturbation_subset}.py`     | REFERENCE/MODIFY | Subset indices (WS5)                        |
| `torchcell/models/equivariant_cell_graph_transformer.py`  | MODIFY           | Multi-head decoder (WS7/WS8)                |
| `experiments/embeddings/compute_esm2_embeddings.py`       | REFERENCE        | Isolate ESM2 (WS9)                          |
| `torchcell/datasets/{esm2,fungal_up_down_transformer}.py` | REFERENCE        | Embedding datasets (WS9)                    |
| `experiments/010-kuzmin-tmi/scripts/query.py`             | REFERENCE        | Canonical build+query pattern               |
| `experiments/010-kuzmin-tmi/scripts/*.slurm`              | REFERENCE        | Canonical SLURM headers                     |

## Phasing

- **Phase A (data):** WS1 → {WS2, WS4} in parallel; WS3 gated on WS9. WS5, WS6 after builds.
- **Phase B (model):** WS7, WS8 in parallel with A; WS9 starts NOW (long pole).
- **Phase C (train):** WS10 after WS2+WS7; WS11 after WS10+WS5; WS12 after WS4+WS8; WS13 continuous.
- **Phase D:** WS14 anytime (independent, ~1 day).
- **Critical path to a safe headline:** WS1 → WS2 → WS7 → WS10 (Fig 3 expr+morph
  reproduced). Fig 6 deletion-ranking (WS4→WS8→WS12) second. **Fig 4 is the risk**
  (WS9 embeddings) — treat as stretch.

---

## WS1. Phenotype dedup/convert branches + graph_processor coverage

**Status:** ⬜ not started. **Depends:** none. **Blocks:** WS2/WS3/WS4.

**Goal.** Make the build chain accept the new phenotype families without crashing,
and confirm each becomes a valid `HeteroData` label.

**Scope / key files.**

- `torchcell/data/mean_experiment_deduplicate.py`: `create_deduplicate_entry`
  raises for `experiment_type ∉ {fitness, gene interaction}`. Add mean-merge
  branches (or a pass-through) for `MicroarrayExpressionPhenotype`,
  `RnaseqExpressionPhenotype`, `CalMorphPhenotype`, `MetabolitePhenotype`,
  `ProteinAbundancePhenotype`, `VisualScorePhenotype`. For vector phenotypes,
  duplicate-merge = elementwise mean + RMS-pooled std (mirror the fitness rule).
- Decision knob: for the *individual* builds we may pass `deduplicator=None`
  (these screens are ~1 record/genotype) — but the *unified* Fig-3 build needs a
  working dedup because fitness+expression on the same genotype must NOT be
  collapsed across modality (dedup keys on `experiment_type` + gene set, so cross-
  modality records are already distinct — verify).
- `torchcell/data/graph_processor.py`: confirm the light `Perturbation` processor
  (used by the equivariant CGT) attaches `global` / `node` / `metabolism`
  phenotype tensors onto `HeteroData` correctly; add handling if a `graph_level` is
  unsupported. Do NOT route these through `SubgraphRepresentation`.

**Checks that must pass.**

- A 50-record smoke build of each phenotype family completes and `dataset[0]` is a
  `HeteroData` with the label tensor of the expected shape (per-gene vector,
  501-D morphology, per-metabolite vector, scalar visual score).
- `GenotypeAggregator` produces a single instance carrying two modalities when a
  genotype has both (construct a synthetic 2-modality genotype to verify).

## WS2. Fig 3 unified build — expression + morphology + fitness

**Status:** ⬜. **Depends:** WS1. **Blocks:** WS10, WS11.

**Goal.** One LMDB build under `experiments/019-simb-multimodal/` unioning the
Fig-3 phenotypes over a shared single/double-KO gene universe, aggregated per
genotype so a strain can carry expression + fitness together.

**Scope.**

- `.cql`: `UNION ALL` blocks (010 pattern) for `MicroarrayKemmeren2014Dataset`
  (node), `Sm`/`DmMicroarraySameith2015Dataset` (node), `ScmdOhya2005Dataset`
  (global), plus `Smf/DmfCostanzo2016` + `Smf/DmfKuzmin2018/2020` fitness on the
  overlapping genes. Filter `graph_level` per block; keep media/temp in the return
  so WS6 can condition.
- Runner: copy `experiments/010-kuzmin-tmi/scripts/query.py`; wire
  `converter=None, deduplicator=<WS1>, aggregator=GenotypeAggregator`,
  `graph_processor=Perturbation`.
- **Key science hook (user's question):** Sameith Dm expression genotypes have
  *paired* double-KO fitness — this is the substrate for "does one fitness scalar
  improve the expression vector prediction?" Ensure the aggregator co-locates them.

**Checks.** Build completes; `dataset_name_index` shows all expected datasets;
count of genotypes carrying ≥2 modalities is reported; a genotype with paired
expression+fitness is exhibited.

## WS3. Fig 4 build — Caudal natural isolates (+ doubles contrast)

**Status:** ⬜. **Depends:** WS1, **WS9 (embeddings)**. **Blocks:** WS11(c).

**Goal.** Build the natural-isolate transcriptome set and the comparison substrate
for "natural variation vs model-designed perturbation," plus test whether isolate
data helps double-KO prediction (user suspects not — isolates vary largely in the
auxiliary genome).

**Scope.** `.cql` for `CaudalPanTranscriptome2024Dataset` (node, 943). Isolates are
perturbation SETS off S288C (SequenceVariant + CopyNumberVariant + Deletion) →
require per-isolate ORF embeddings (WS9), unlike the S288C-fixed KO datasets.
Optionally union Dm fitness for the contrast. **Loader defect to verify first**
(MEMORY): Caudal paralog + `SACE_` gene drops; Sameith sign bug.

**Checks.** Build completes with isolate genotypes resolved to embeddings; report
how many isolate ORFs lack an embedding (drives WS9 coverage).

## WS4. Fig 6 build — β-carotene + betaxanthin + metabolome

**Status:** ⬜. **Depends:** WS1. **Blocks:** WS12, WS11(b).

**Goal.** Build the production-screen + metabolome set for strain-design ranking
and the Arm-B "does metabolome help" test.

**Scope.** `.cql` for `CarotenoidOzaydin2013Dataset` (global visual score, 4474),
`BetaxanthinCachera2023Dataset` (metabolism, 4735), `AminoAcidMulleder2016Dataset`
(metabolism, 4678), `MetaboliteZelezniak2018Dataset` (metabolism, 95),
`MetaboliteDaSilveira2014Dataset` (metabolism lipids, 127). Cassette genes belong
to each dataset's *reference strain* (Design Decision 3) — verify the reference
carries them and the perturbation is the ORF deletion only.

**Checks.** Build completes; per-metabolite label vectors have the right length
(Mülleder 19 AA, Zelezniak ~50); β-carotene score is scalar-global.

## WS5. Subset indices — modality / leave-out

**Status:** ⬜. **Depends:** WS2/WS3/WS4. **Blocks:** WS11.

**Goal.** Precompute exactly the indices needed to carve leave-out training sets,
without index sprawl.

**Scope.** Reuse existing `phenotype_label_index`, `dataset_name_index`,
`perturbation_count_index`, `is_any_perturbed_gene_index` (all already built in
`neo4j_cell.py`) and `PerturbationSubsetDataModule`. Add ONE new axis:
`phenotype_class_index` ({expression, morphology, metabolite, proteome, fitness,
interaction} → row indices) so a run can `leave_out=metabolism` or
`only=expression+fitness`. Persist as JSON next to the dataset (file-locked
pattern). Define the concrete named splits WS11 consumes:
`expr_only`, `expr+fitness`, `metab_off`, `metab_on`, `isolates_off`, `isolates_on`.

**Checks.** Each named split loads via `CellDataModule(split_indices=...)`; train/
val/test ratios preserved; leave-out actually removes the modality (assert counts).

## WS6. Media / environment ε-conditioning

**Status:** ⬜. **Depends:** WS2/WS4. **Blocks:** WS11 joint runs.

**Goal.** Prevent the joint model from confounding media/temp with dataset.

**Scope.** Audit is DONE (4 combos; Ohya YPD/25 °C the outlier). Decide + wire ε
into the decoder (`ŷ = R_φ(H_pert, ε)`, already in Methods). Minimal encoding:
one-hot/(learned) over the small (media,temp) vocabulary observed in the build.
For individual models ε is constant → no-op. **Open question flagged below:**
whether to drop Ohya/25 °C into a separate head vs condition — recommend condition.

**Checks.** Joint run with ε-conditioning does not degrade an individual baseline;
ablating ε on the mixed-media set measurably changes morphology loss.

## WS7. Multi-head decoder — per-gene / global / per-metabolite

**Status:** ⬜. **Depends:** none (parallel with A). **Blocks:** WS10, WS11.

**Goal.** Grow the single-head model into the documented multitask decoder.

**Scope (`equivariant_cell_graph_transformer.py`).**

- **Global head** (morphology 501-D, β-carotene scalar): new `MLP(h_CLS)` (+ optional
  `GlobalPool(H_genes_pert)`), instantiate next to `self.perturbation_head` (~:643),
  call after :896. Smallest change — `h_CLS` already carries whole-cell state.
- **Per-gene head** (expression, proteome): `MLP(H_genes_pert) → [B,N]` (equivariant,
  per-node). Reuses the already-produced `H_genes_pert [B,N,d]`.
- **Per-metabolite head:** see WS8 (needs metabolite indexing).
- **Masked multitask loss:** sum per-head losses, each masked to genotypes that
  carry that phenotype (sparse). Keep the existing graph-reg attention loss term.

**Checks.** Forward returns all heads; loss masks correctly on a mixed batch; a
single-head (expression) config reproduces WS10's individual baseline (no regression
from the plumbing).

## WS8. Metabolite readout + metabolic-graph annotation

**Status:** ⬜. **Depends:** WS7. **Blocks:** WS12, WS11(b).

**Goal.** A per-metabolite head mapping predictions onto Yeast9 metabolite IDs,
using the metabolic incidence already in `cell_graph` — as *representation
annotation, not attention prior* (Design Decision 4).

**Scope.** Add a metabolite embedding/index table; readout options: (i) metabolite-
node embeddings if we promote metabolites to encoder nodes (heavier, breaks fixed-N)
or (ii) **GPR-pooled** — aggregate `H_genes_pert` over the genes catalyzing each
reaction/metabolite via `gpr`/`rmr` incidence (lighter, no N change) — **recommend
(ii) for the conference.** Map head outputs to the stored metabolite label vector
(Mülleder AA IDs, Zelezniak `s_NNNN`).

**Checks.** Per-metabolite predictions align to the correct metabolite IDs;
Mülleder AA baseline trains; GPR pooling touches the right gene sets (spot-check one
amino-acid pathway).

## WS9. Natural-isolate embeddings (ESM2 + FUDT) — LONG POLE, START NOW

**Status:** ⬜. **Depends:** none. **Blocks:** WS3, WS11(c).

**Goal.** Per-isolate sequence features for Caudal's 943 isolates: ESM2 over each
isolate's ORF proteins + FUDT (species-aware SpeciesLM) over promoter/terminator
windows. The S288C KO datasets reuse the existing reference embeddings; **only the
natural isolates need fresh per-strain embeddings.**

**Scope.** Reuse `experiments/embeddings/compute_esm2_embeddings.py` +
`torchcell/datasets/esm2.py` (`Esm2Dataset`) and `fungal_up_down_transformer.py`
(`FungalUpDownTransformerDataset`), driven per-isolate rather than over the single
reference genome. Needs each isolate's ORF sequences (from the Caudal genotype
SequenceVariant/CNV set applied to S288C, or the isolate assemblies if mirrored).
Compute on Delta (`gpuA40x4`, `bbub-delta-gpu`) — the runners are already Delta-
shaped. **This is the largest compute item; schedule day 1.**

**Checks.** Embedding coverage report per isolate (fraction of ORFs embedded);
a spot isolate's ESM2 tensor shape matches the reference pipeline.

## WS10. Individual per-phenotype models — fast baselines

**Status:** ⬜. **Depends:** WS2, WS7. **Blocks:** WS11.

**Goal.** Cheap per-phenotype baselines establishing the abstract numbers and a
floor for the joint runs. Expected minutes-scale given dataset sizes.

**Scope.** Verify small runs on the workstation, then scale on IGB. Targets:
expression (Kemmeren+Sameith, reproduce r≈0.543), morphology (Ohya, reproduce
r≈0.619 — **currently not reproducible in-repo, highest-priority baseline**),
metabolite (Mülleder/Zelezniak), pigment (Ozaydin/Cachera). One head active each.

**Checks.** Each baseline logs a test Pearson r; morphology r reproduced or the gap
diagnosed; runtimes recorded to size the joint runs.

## WS11. Joint / leave-out ablations — "does more data help?"

**Status:** ⬜. **Depends:** WS10, WS5, WS6. **Blocks:** Figs 3/4/6 panels.

**Goal.** Test the abstract's central inductive-bias claim in three concrete
leave-out contrasts, each a WS5 named split:

- **(a) Fig 3 — does fitness help expression?** expression-only vs expression+fitness
  (Sameith Dm expression has paired double-KO fitness). One scalar added to a vector
  target — the user's "predict expensive states from cheap states" probe.
- **(b) Fig 6 — does metabolome help production?** Arm B: with/without Mülleder +
  Zelezniak metabolome → isobutanol (valine pool) / betaxanthin (tyrosine pool)
  prediction.
- **(c) Fig 4 — do natural isolates help double-KO?** isolates_on vs isolates_off
  (user expects little lift — auxiliary-genome variation).

**Checks.** Each contrast reports Δr (joint − individual) with matched splits/seeds;
sign and magnitude recorded even if null (a clean null for (c) is a legitimate
result).

## WS12. Fig 6 strain-design inference — deletion ranking

**Status:** ⬜. **Depends:** WS4, WS8. **Blocks:** Fig 6 recommendation panel.

**Goal.** Rank single-gene-deletion recommendations for improved β-carotene /
betaxanthin, reusing the 010 inference-at-scale pattern.

**Scope.** Score the ~4.5 k ORF deletions (or the full singles space) with the
trained Fig-6 model; produce a ranked recommendation table (cf.
`experiments/010-kuzmin-tmi/results/inference_*/singles_table_*`). Pair narrative
with the iBioFoundry DBTL loop.

**Checks.** Ranked CSV produced by a committed script; top hits sanity-checked
against known β-carotene/betaxanthin modifiers.

## WS13. Compute prep — IGB (mmli/cabbi) + Delta SLURM + configs

**Status:** ⬜. **Depends:** continuous. **Blocks:** all cluster runs.

**Goal.** Author every SLURM script + Hydra config HERE (no Claude Code on IGB);
document the transfer + launch procedure.

**Scope.** Templates (verified headers):

- **IGB mmli:** `#SBATCH -p mmli` … `--gres=gpu:4`, Singularity `rockylinux_9.sif`,
  `conda activate torchcell`.
- **IGB cabbi:** `#SBATCH -p cabbi` … same filesystem `/home/a-m/mjvolk3/…`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Delta:** `--partition=gpuA40x4 --account=bbub-delta-gpu --time=48:00:00`,
  apptainer with `/projects/bbub` + `/scratch/bbub/mjvolk3` binds.
Rapid-verify configs for the workstation; long-run configs for IGB.

**Checks.** A dry `--help`/1-step run of each config launches locally; SLURM scripts
lint; transfer checklist (code + dataset LMDB + embeddings) written.

## WS14. Error-bar provenance fix — trigenic τ ±0.006

**Status:** ⬜. **Depends:** none. **Blocks:** abstract submission.

**Goal.** Resolve the flagged blocker: DANGO/DCell ± are SEM over 3 replicates,
TorchCell's ±0.006 is a hardcoded reported SE with no replicate array, Yeast9 is
deterministic. Confirm TorchCell's value is SE/SEM (comparable) vs SD (~√3× too
large) so all bars represent the same quantity.

**Checks.** `experiments/010-kuzmin-tmi/scripts/trigenic_tau_model_comparison.py`
regenerates with matched-statistic error bars; abstract number updated if needed.

---

## North Star (DEFERRED — not for SIMB 2026)

## WS-NS1. Additive perturbation operator (inverse design)

**Status:** ⏸. Variable-length additions (genes + metabolites) + a perturbation-
type/sign signal `e_τ` fused into the operator's K/V context, relaxing the fixed-`N`
assumption. Enables representing heterologous cassette *additions* dynamically
(β-carotene/betaxanthin design by construction, not just deletion). Structural — the
largest model change; explicitly out of the 10-day scope.

## WS-NS2. Enzyme-constrained FBA regularizer + kcat/KM

**Status:** ⏸. Add GECKO/ecYeastGEM-style soft constraints to the objective:
capacity `|v_i| ≤ k_cat,i · E_i` and shared protein budget `Σ MW_i E_i ≤ P_avail`
(the `E_i` as pool-allocated variables, not measured). Populate `k_cat` (and
optionally `K_M`) by prediction over ORF sequences — KcatNet
(`iBioFoundry/KcatNet`) and the Wu 2026 method (`wuSystematicallyExploringYeast2026`)
— including the natural-isolate variants. Two variants worth interpreting: a HARD
stoichiometric burn-in vs a SOFT learned mapping (compare what the soft map
recovers). Reference: `/tmp/screenshots/TorchCell - Enzyme-constrained vs Kinetic
Models.pdf`. This is the mechanistic-grounding North Star; kinetic models are a
further tier still.

## Open Questions

1. **Ohya media/temp (YPD/25 °C):** condition the decoder on ε (recommended) vs a
   separate morphology head vs exclude from joint runs. → WS6.
2. **Metabolite readout:** GPR-pooled `H_genes_pert` (recommended, no N change) vs
   promoting metabolites to encoder nodes (heavier). → WS8.
3. **Fig 4 scope under deadline:** if WS9 isolate embeddings slip, is a reduced Fig 4
   (fewer isolates, or descriptive contrast only) acceptable for the abstract? →
   affects critical path.
4. **Messner2023 proteome:** wire into the adapter map for a second proteome anchor,
   or leave out for the conference? Currently out of scope (auxiliary).
5. **Dedup for vector phenotypes:** pass-through vs mean-merge — confirm no target
   dataset has true within-genotype replicates that must be averaged. → WS1.

## Verification (universal gate)

Every build passes L0–L4 (existing verifier runners). Every artifact (figure/table/
number) is produced by a committed script in `experiments/019-simb-multimodal/`
reading real result files (CLAUDE.md provenance rule). Model changes keep whole-tree
mypy + ruff clean; individual-baseline configs must reproduce after the multi-head
plumbing lands (no regression).

## 2026.07.22 - Wave-1 landings + reconciliation

Parallel uber-implement fan-out (isolated worktrees, serial enqueue-merge). Landed:

- **WS14** `cd3b928d` (PR #157) — τ error bars unified to **SEM**. Root cause: CGT
  had three real replicates (0.462/0.452/0.447 from wandb `inf_1`); the old ±0.006
  was their *population SD*. Abstract now **CGT r=0.454±0.004, ρ=0.421±0.003** (means
  unchanged); caveat marked RESOLVED. Open Question resolved.
- **WS7** `5c6c17fa` (PR #158) — multitask heads `GlobalHead`/`PerGeneHead`/
  `PerMetaboliteHead` (GPR-pooled sparse incidence) + `MaskedMultitaskLoss`, config-
  selectable via `heads_config`; `heads_config=None` byte-identical to the old single
  head (allclose regression test). Proteome uses the per-gene head.
- **WS1** `29814103` (PR #161) — dedup + graph_processor coverage for all six new
  phenotype families incl. ProteinAbundance (Messner-ready). Verified vs live DB:
  Kemmeren per-gene (6169,), Ohya 501-D, Mülleder 19-metabolite; `GenotypeAggregator`
  co-locates expression+fitness into ONE HeteroData. Touched a **3rd file**
  (`neo4j_cell.py` `label_df` — dict labels left NaN, not crashed).
- **WS9** `ff350819` (PR #159) — per-isolate **ESM2 runner works** (CDS already
  materialized in the Peter 2018 mirror; translate→embed, smoke-proven). **FUDT is
  the gap** (needs per-isolate promoter/terminator flank from `1011Assemblies` +
  per-strain locus map; store holds CDS only) — stubbed with `NotImplementedError`.

**Corrections to the plan above:**

- **Expression is *also* not yet reproducible in-repo** (not just morphology, as WS10
  implied). Both r=0.543 (expr) and r=0.619 (morph) need in-repo regeneration → WS10.
- **Metabolite→Yeast9 map is a hard gap:** live DB shows Mülleder
  `target_metabolite_ids = null`, so WS8's per-metabolite head needs an explicit
  amino-acid-name → `s_NNNN` map (this is where the enzyme-constrained-regularizer
  idea, [[TorchCell - Enzyme-constrained vs Kinetic Models]], plugs in).
- **Messner adapter already landed** on main (`402733a3` from the DB-build session);
  overnight full rebuild will serve it → then Fig-3 proteome + rebuild verification.
  So Open Question 4 is decided: Messner IS coming in.

**New open decision (FUDT, from WS9):** for the deadline, take **ESM2-only per-isolate
features for Fig 4** (fastest, defers the assembly-flank extraction), or invest in the
per-isolate promoter/terminator path now? Recommend ESM2-only for the abstract.

**In flight:** WS2 (Fig-3 build) — key output is the measured expression↔fitness
genotype overlap that gates WS11(a).

### 2026.07.22b - WS2 census + two substrate blockers (→ WS2b)

WS2 `c8387cd9` landed the Fig-3 build harness (`experiments/019-simb-multimodal/`)

- overlap census. **DB substrate is healthy**: Sameith-Dm expression with matching
double-mutant fitness = 57/72; single-KO expr+fitness = 1,360; gene-set-predicted
≥2-modality genotypes = 4,425 (expr+fitness 1,417, expr+morph 1,440, all-three 1,326).

**But the real build realized only 90 expr+fitness** (2 blockers → WS2b, in flight):

1. **Aggregator key.** `MeanExperimentDeduplicator` rewrites only *replicated*
   genotypes to `mean_deletion`; singletons keep `kanmx_deletion`/`sga_kanmx_deletion`.
   `GenotypeAggregator` keys on `(gene, perturbation_type)` → expression (singleton
   kanmx) and fitness (replicated mean_deletion) never co-locate. **Fix (WS2b): key
   the aggregator on the sorted gene-SET only** (matches the deduplicator). Runs at
   dataloader time only — does NOT touch the served graph. TODO: axis-aware key when
   non-deletion perturbations arrive.
2. **Processor.** The build used `SubgraphRepresentation` (crashes on dict/vector
   labels). **Fix (WS2b): use the WS1-fixed `Perturbation` processor.**

**Caudal #71 — NOT a blocker (verified against served DB 2026.07.22):** served
Caudal isolates carry 165–175 `natural_gene_absence` perturbations each (would be ~0
if the bug were live) + off-graph `sequence_uri`/`sha256` pointers. The loader gates
absence on `s288c_mask` (reference membership) on main; #71/paralog/SACE_ are all
FIXED. **Rebuild needs Messner only, not Caudal.** (Caudal perts are stored in
`serialized_data`, NOT as `:Perturbation` nodes — pull Caudal whole, not via a
perturbation gene-set filter.)

**Isobutanol IS in the served DB** (IsobutanolScreenLopez2024 4,554 + Validated 224,
both `metabolism`) → keep in the full Fig-6 query, select via indices (Design Dec 1).

**Launched wave (2026.07.22):** WS2b (aggregator+processor), WS13 (compute harness:
train_cgt_multitask.py + IGB mmli/cabbi + Delta configs/slurm). Messner rebuild (job
990) is the DB-build session's; this branch's work is all Messner-independent.

### 2026.07.22c - WS2b/WS13/WS4 landed; the cassette-aggregation fork

Landed: WS2b `349bc54b` (expr+fit 90→**1416**), WS13 `4c10f694` (harness), WS4
`a0fe1de3` (Fig-6 build, 13,948 genotypes).

**WS4 census — cassette design fork (connects to Design Decision 3 + user's earlier
ambivalence).** Pigment strains carry their heterologous pathway as `gene_addition`
perturbations (β-carotene 3 genes, betaxanthin 4). Because the WS2b aggregator keys
on the FULL perturbed gene-set (deletions + additions), pigment strains sit in a
genotype space DISJOINT from single-KO metabolome screens:

- metabolome+**isobutanol** = **4367** (isobutanol is cassette-free single-KO → works now)
- metabolome+betaxanthin = **0**, metabolome+β-carotene = **0**
- cassettes-stripped hypothetical: 4439 / 4226 — the cassette key-isolation is exactly
  what zeroes pigment co-location.

Implications:

- **WS11b headline ("does metabolome help production") is already well-powered via
  isobutanol** — no change needed for the abstract.
- **Fig-6 pigment deletion-ranking (WS12) works either way** — the cassette is CONSTANT
  within a pigment dataset, so members still differ by their deleted gene; the model
  predicts pigment as f(deletion) on a fixed cassette background.
- **Only the cross-dataset metabolome↔PIGMENT transfer test needs a choice:** treat the
  cassette as reference-strain background (aggregate on the DELETION gene-set only →
  ~4226-4439 co-location, Design Decision 3 / "predict over deletions of a pathway-
  carrying reference strain") vs keep it an explicit addition (disjoint, North-Star
  WS-NS1). This IS the axis-aware-key TODO WS2b left. **Recommend: isobutanol for the
  WS11b headline now; cassette-as-background aggregation only if we also want the
  pigment metabolome-transfer analysis.** → user decision.

**In flight:** WS10a (harness smoke + expression/morphology baselines — the gate before
cluster runs).

### 2026.07.22d - WS10a landed: harness VALIDATED, both phenotypes train

WS10a `018bd0c8`. The multitask harness trains on real Fig-3 batches (workstation, tiny
model, 4 epochs): **expression val Pearson 0.024→0.115**, **morphology val Pearson
0.050→0.139** (both rising). Closes the "expression AND morphology not reproducible
in-repo" gap at the mechanism level — full-scale numbers are the next (cluster) run.
Fixed WS13's decode (3 real-batch bugs: wrong phenotype strings → real are
`expression_log2_ratio`/`calmorph`/`fitness`; `phenotype_sample_indices` not batch-
offset → use `phenotype_values_batch`; per-graph list-of-lists collation). Core fix:
`graph_processor.py` Perturbation always emits `phenotype_stat_*` keys.

Real-batch facts for the full runs: expression 6169 vals (6127 map to gene nodes),
CalMorph **281** (not 501), fitness 1. **CalMorph is unnormalized → needs a
normalization transform before the real morphology run.** Timing (tiny model): ~15
s/epoch full expression, ~47 s/epoch full morphology → scale up for the cluster model.

**Next to get the abstract NUMBERS (WS10 full + WS11a):** (1) morphology normalization
transform; (2) full-size cluster configs (WS13 has templates); (3) user sbatch on
IGB/Delta (no Claude there). Pending decisions unchanged: cassette aggregation (Fig-6
pigment transfer, optional), FUDT (ESM2-only recommended). Pre-existing: 3
`test_cell_data.py::test_stoichiometric_matrix_*` FAIL on main (unrelated; likely
go.obo/data-env).

**SIMB workstreams landed this session (9):** WS1, WS2, WS2b, WS4, WS7, WS9, WS10a,
WS13, WS14. Stack complete: Fig-3 + Fig-6 builds · multitask model+heads · harness ·
validated training.

### 2026.07.22e - WS10b (CalMorph norm) + WS9b (FUDT feasible) landed

**WS10b `58ad13ae`** — morphology normalization, SI-sourced (Ohya 2005, sha256-pinned):
served `calmorph` = 281 BASE params, `_coefficient_of_variation` = 220 CV (=501). Ohya's
own method = per-param Box–Cox (fit on WT) + z-score; we implemented train-split z-score
(loss 4.5M→1.14; val 0.912 < variance floor 1.0 = ~9% variance explained). **Keep/drop
= user decision:** 0 constant, 3 degenerate (A113_A1B/A113_C actin ratios, C123_C
small-bud) — recommend KEEP all 281 (post-z-score they're harmless noise). **METRIC
CORRECTION:** the earlier "morphology Pearson 0.05→0.14" was a flatten-Pearson
feature-scale ARTIFACT (CalMorph spans ~8 orders of magnitude); post-norm it's ~0 on the
smoke. Real runs must report **per-feature/per-gene-averaged Pearson** (same care for
expression). Decisions pending: keep-281 vs drop-3; z-score vs +Box-Cox.

**WS9b `5d0300a6`** — per-isolate FUDT FEASIBLE. BLASTN each gene's known CDS →
isolate's own assembly → slice 5'/3' flanks → `FungalUpDownTransformer` → (1,768).
~100% identity, contig-end truncation 2.5–6.25%, ~22 min/isolate → 943 ≈ ~1–2 days
parallel+GPU. To scale: genome-wide mapping validation (PoC used paralog-rich
subtelomere), BLAST+ in run env, wire SCerevisiaeGenome reference-fallback.

### Remaining for the abstract numbers (mostly cluster compute = user sbatch)

1. **Metric fix** (in-session): per-feature/per-gene-averaged Pearson — the honest metric.
2. **WS10 full baselines** (cluster): expression + morphology at full scale → r numbers.
3. **WS11a** (cluster): does fitness help expression, on the 1,416 pairs.
4. **Full isolate embeddings** (cluster ~1-2 days): ESM2 + FUDT over 943 isolates → Fig 4.
5. **WS12** Fig-6 deletion ranking. **WS11b** metabolome→isobutanol (co-location ready).
6. **Messner** proteome (other session's rebuild 990) → proteome per-gene ablation.

### Pending user decisions (defaults will proceed if unanswered)

- CalMorph: keep all 281 (default) vs drop 3 degenerate.
- Morphology norm: z-score (default) vs + Ohya Box–Cox.
- Cassette aggregation for pigment metabolome-transfer (optional; isobutanol powers WS11b).
- Small cleanup: `chown`/temp-dir fix for the pre-existing stoich-test perms failure.
