---
id: ipx9kooy7pqclpilxnkih9w
title: '15'
desc: ''
updated: 1784102143779
created: 1784102143779
---

# Plan: YeastPhenome Ingestion Harness (per-PMID adapter, one-study proof)

## Context — why

YeastPhenome (Turco & Baryshnikova 2023, "Global analysis of the yeast knockout
phenome", primary PMID 37205762, `turcoGlobalAnalysisYeast2023`, paper OCR'd + mirrored,
Zotero `database` collection) is a **curated meta-aggregation of ~534 per-PMID
knockout×condition screens**, not a single experiment. Its GitHub layout is
`Datasets/<PMID>/` each holding `<author>_<year>_value.txt` (the ORIGINAL per-study
values), `<author>_<year>_valuez.txt` (YeastPhenome-computed z-scores), and `code.ipynb`;
`Utils/` holds the consensus tested-gene lists + `yp_sgd_features.txt` (name→SGD map).
Data is archived on Zenodo: `10.5281/zenodo.7714347` (values + code) and
`10.5281/zenodo.7714354` (NPVs / cross-study aggregate).

Why torchcell touches it at all: it is the single largest lever on deletion×condition
coverage in the roadmap (row 79 of `notes/paper.north-star.dataset-triage.md`), feeding
the SIMB dataset-coverage roadmap (`#30`). But ~42% of its screens are Kemmeren
expression (`kemmeren2014.py`, already built — exclude), and many other PMIDs are already
built as torchcell primaries (Costanzo, Kuzmin, Baryshnikova, Auesukaree, Wildenhain,
Hoepfner, Smith, Hillenmeyer/FitDb, Cachera). So the win is NOT a monolithic
"YeastPhenome dataset" — it is a **uniform per-PMID adapter** that wraps each remaining
screen into torchcell's EXISTING modular `EnvironmentResponseExperiment` schema (the 11
shipped WS15 loaders), sourcing ORIGINAL per-study values with the primary PMID canonical
and YeastPhenome as a *curation layer* (the value it re-hosts, not its z-score).

This plan proves the per-study fit **end-to-end on ONE representative screen** (L0-L4) —
a non-expression, non-already-built homozygous-deletion chemogenomic/stress screen — and
in doing so *defines what the harness looks like*. It EXPLICITLY leaves the cross-study
NPV aggregate + its 2-hop-provenance model as an **open user decision** (that is the real
supervised call; see Open Questions), and does not fan out to 534 papers.

Open issues this plan navigates: `#92` (`Media.is_synthetic` now required — breaks L0 on
11/11 legacy datasets until they rebuild; a NEW build passes cleanly), `#73` (the
caudal-class silent gene-drop bug — resolved here with drop-and-log + a review threshold),
`#30` (SIMB dataset-coverage roadmap this feeds), `#70` (URL-rot precedent — why a pinned
Zenodo archive beats a live git `master`).

## Relevant Files

| path | action | purpose | stance |
|---|---|---|---|
| `torchcell/datasets/scerevisiae/<study>.py` | NEW | the proof loader; one LMDB for the chosen PMID, patterned on `vanacloig2022.py` | new code, high-scrutiny |
| `torchcell/datasets/scerevisiae/__init__.py` | MODIFY | export the new dataset class | stable, append-only (mirror L52-53, L97, L138) |
| `torchcell/verification/runners.py` | MODIFY | add `ENVIRONMENT_RESPONSE_DATASETS` entry (~L756, before `env_chemgen_mota2024` at L779) | stable/extension-designed per `notes/torchcell.verification.runners.md` (2026.07.08) |
| `torchcell/literature/manifest.py` | MODIFY | add `zenodo` member to `RetrievalMethod` (StrEnum at **L59**, members L65-69) | in-flux core, high-scrutiny — this is THE enum, verified by grep |
| `tests/torchcell/datasets/scerevisiae/test_<study>.py` | NEW | record-construction + count test, patterned on `test_cachera2023.py` | new test |
| `tests/torchcell/verification/test_environment_response_verification.py` | NEW | closes a real coverage gap — no env-response verification test exists (only expression/metabolite/morphology/protein/visual_score in `tests/torchcell/verification/`) | new test |
| `torchcell/datasets/scerevisiae/vanacloig2022.py` | REFERENCE | primary template (download+verify sha256, `process()` builds triples, one LMDB); no genome needed (rows carry systematic ORFs) | production, undocumented but stable |
| `torchcell/datasets/scerevisiae/auesukaree2009.py` | REFERENCE | template for the **genome-injected** name→ORF path (L236 `genome` REQUIRED, L344 `alias_to_systematic`, L372-392 drop-and-log) — closest to YeastPhenome rows keyed by common name | production, stable |
| `torchcell/datasets/scerevisiae/mota2024.py` | REFERENCE | template for categorical + n-not-released precedent (per-record dedup, conservative rep) | production, stable |
| `torchcell/data/experiment_dataset.py` | REFERENCE | `ExperimentDataset` base ABC (subclassed by the loader) | stable |
| `torchcell/datamodels/schema.py` | REFERENCE | `EnvironmentResponsePhenotype` L2640, `EnvironmentResponseExperiment` L2766, `MeasurementType` L2612, `SmallMoleculePerturbation` L1356, `Environment` L1412, `Compound` L1081, `Concentration` L1145, `KanMxDeletionPerturbation` L265, `EnvironmentPhysicalPerturbation` L1381, `EngineeredCopyNumberPerturbation` L770 | schema is authoritative |
| `torchcell/datamodels/media.py` | REFERENCE | media library constants w/ `is_synthetic` set (YPD L242, SD_MINIMAL L353, SC L390, SGA_* L207/L218); pass one of these, do not hand-roll `Media(...)` | stable |
| `torchcell/verification/environment_response.py` | REFERENCE | the verifier reused UNCHANGED (`verify_environment_response_dataset`, `environment_response_gene_set`) | stable, do not touch |
| `torchcell/sequence/genome/scerevisiae/s288c.py` | REFERENCE | `SCerevisiaeGenome.alias_to_systematic` for name→ORF | stable |
| `torchcell/literature/manifest.py` + `provenance.py` | REFERENCE | `RetrievalRecord`/`Provenance` shape for the loader + runner provenance | stable |
| `notes/paper.north-star.dataset-triage.md` (row 79) | REFERENCE | roadmap justification | note |
| `experiments/database/scripts/build_supported_datasets_table.py` | REFERENCE | LMDB-scanning inventory used to build the PMID skip list | tooling |

## Key Design Decisions

1. **Per-PMID loader = a normal `EnvironmentResponseExperiment` dataset; NOT a monolithic
   YeastPhenome dataset.** Why: it matches the 11 shipped WS15 env-response loaders exactly
   and needs zero new schema. The harness IS the uniform shape of these loaders, not a new
   container class. The "one LMDB per PMID" rule is load-bearing: Hoepfner (~3M records,
   `stream:True` at runner L820) and Costanzo (43GB) are the standing warning that a
   many-study monolith is unbuildable and un-verifiable. A per-PMID LMDB stays small,
   independently L0-L4-gated, and independently landable.

2. **Store ORIGINAL values from Zenodo `7714347` `<author>_<year>_value.txt`**, with
   `MeasurementType` = the primary paper's NATIVE statistic (`growth_rate` /
   `log2_ratio` / `sensitivity_score` / `categorical` — all already in the enum, schema
   L2632-2637). Reject the `_valuez.txt` z-score AND the Zenodo `7714354` NPV as the stored
   phenotype: both are YeastPhenome-derived transforms that discard the primary's units and
   fold in cross-study normalization we do not want baked into a single-study record. Record
   the NPV only as a *derived* annotation if ever, never as `environment_response`. Reject
   `FitnessPhenotype` outright — it clamps ≤0 and would corrupt a signed score (many screens
   report signed sensitivity).

3. **Provenance: pin ONE sha256 on the fetched Zenodo `7714347` archive; mirror only the
   proof PMID's subtree** (`Datasets/<PMID>/`) into the loader `raw_dir`; add `zenodo` to
   `RetrievalMethod` (`manifest.py` L59). Reject git-cloning the ~1GB live `master` — it is
   mutable and has thousands of unpinnable files (this is the `#70` URL-rot lesson: pin the
   archive bytes, not a moving branch). Reject Zenodo `7714354` (NPV aggregate = the
   deferred layer, Decision-per Open Q1). **Curation-vs-primary stance for v1:**
   `Publication` = the primary PMID (37205762 is the YeastPhenome meta-paper; the SCREEN's
   own PMID is the record's publication), but the VALUE is sourced from YeastPhenome's
   Zenodo re-host. Acceptable for v1; flag in the dendron note that ideal rigor cross-checks
   the primary paper's own SI. This stance sets `source_url`/`retrieval_method`/citation for
   every future YeastPhenome study, so it is Open Q3.

4. **Genotype axis = single-ORF homozygous YKO deletion → `KanMxDeletionPerturbation`.**
   Resolve gene names via an INJECTED `SCerevisiaeGenome.alias_to_systematic`
   (auesukaree2009.py L236/L344 pattern) with **drop-and-log + a per-study drop-rate REVIEW
   THRESHOLD**: any nonzero drop is a flag, not a silent pass (`#73` class). Cross-check the
   resolved set against `Utils/yp_sgd_features.txt` (YeastPhenome's own SGD map) as an
   independent second opinion — disagreement is logged. Reject the vanacloig no-genome path
   only IF the chosen study's rows are keyed by common name (most YeastPhenome value files
   are); if rows already carry systematic ORFs, skip the genome (vanacloig pattern).
   Heterozygous-diploid screens → `EngineeredCopyNumberPerturbation` (the FitDb/Hoepfner HIP
   copy-1/2 pattern, runner L834) — **OUT OF SCOPE for v1** (homozygous only).

5. **Environment axis = reuse the landed `EnvironmentPerturbation` /
   `SmallMoleculePerturbation` / `EnvironmentPhysicalPerturbation` exactly.**
   `Compound.name` = the primary paper's REAL chemical name (a condition-id placeholder is
   allowed only if the compound is genuinely unnameable, and then flagged);
   `inchikey`/`chebi`/`smiles` = `None`. `Concentration` may be basis-only (e.g.
   `DoseBasis.IC30`, `value=None`, as vanacloig L302-303). Temperature/oxidative stresses →
   `Environment.temperature` / `EnvironmentPhysicalPerturbation` (auesukaree stress pattern),
   not a fake compound. **Do NOT block v1 on ChEBI/InChIKey resolution** — structure IDs are
   an enrichment pass, not a gate.

6. **Double-count avoidance via a HAND-BUILT PMID reconciliation map** (there is no machine
   index: `paper/tables.py` has no `DATASET_ROWS`; inventory = run
   `experiments/database/scripts/build_supported_datasets_table.py` to scan built LMDBs +
   read `notes/paper.supported-datasets-and-databases.md` + `grep -rn pubmed_id`/`doi` across
   `torchcell/datasets/scerevisiae/*.py`). The proof PMID MUST NOT be in the skip list:
   Costanzo2016, Kemmeren2014, Cachera2023 (`37572348`), Baryshnikova2010, Auesukaree2009,
   Kuzmin2018/2020, Wildenhain2015, Hoepfner2014, Smith2006, Hillenmeyer2008/FitDb (plus any
   other primary already in the built inventory). Record the map + its evidence in the
   dendron note so the next YeastPhenome study reuses it.

7. **Uncertainty / `n_samples` sourced from the ONE proof paper's SI** with a verbatim
   quote + sha256 + section (CLAUDE.md "Adding Datasets" rule). If the primary genuinely
   does not report `n` (the mota2024 precedent — categorical calls with no released rep
   count), record `not-released` + a conservative/representative value + a dendron flag —
   **never guess**. v1 is BOUND to the single study; per-paper SI sourcing across 534 papers
   is the dominant manual cost and is explicitly out of scope.

8. **`Media.is_synthetic` (`#92`): pass `is_synthetic` on every `Media`** — use the
   `media.py` constants (which already set it) rather than constructing `Media(...)`
   inline. A NEW build passes L0 cleanly; the 11 legacy datasets that predate the required
   field are a SEPARATE scheduled rebuild session, out of scope here.

## Approach

The v1 slice is one vertical, provable line — not a framework.

**Pick the proof PMID.** Criteria, all four required: (a) NOT already built as a torchcell
primary (Decision 6 skip list); (b) non-expression, homozygous deletion × chemical-or-stress
condition; (c) a continuous *scalar original* value present in the Zenodo `7714347`
`<author>_<year>_value.txt` (so `MeasurementType` is `growth_rate`/`log2_ratio`/
`sensitivity_score`, not just categorical — a continuous readout best exercises the schema
and verifier); (d) `n_samples` + uncertainty are sourceable from THAT paper's SI. Selection
is Open Q2 (needs user approval of criteria / the actual study) — the plan defines the
criteria; it does not silently pick.

**Mirror its Zenodo subtree.** Fetch the `7714347` archive, pin its sha256 (runner
`Provenance` + loader `DATA_SHA256`), extract only `Datasets/<PMID>/` (+ the relevant
`Utils/*` maps) into `raw_dir`. Add `zenodo` to `RetrievalMethod` in `manifest.py` first
(the loader/runner provenance references it).

**Write the loader**, subclassing `ExperimentDataset`, patterned on `vanacloig2022.py`
(structure) with the auesukaree2009 genome-injection path if rows are name-keyed:
`download()` fetches+verifies the archive; `raw_file_names` lists the mirrored value file;
`process()` reads `_value.txt`, resolves each ORF (drop-and-log + threshold, Decision 4),
builds the `EnvironmentResponseExperiment` / `...Reference` / `...Phenotype` triples, and
writes ONE LMDB. The record shape (this is the only place a snippet disambiguates):

```python
EnvironmentResponseExperiment(
    dataset_name=self.name,
    genotype=Genotype(perturbations=[
        KanMxDeletionPerturbation(systematic_gene_name=orf, perturbed_gene_name=common)]),
    environment=Environment(              # media from media.py constant (is_synthetic set)
        media=YPD, temperature=Temperature(value=30.0),
        perturbations=[SmallMoleculePerturbation(
            compound=Compound(name="<paper chemical>"),
            concentration=Concentration(basis=DoseBasis.IC30))]),  # or physical/temperature
    phenotype=EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.<native>, environment_response=<original value>,
        environment_response_uncertainty=<sd or None>,
        environment_response_uncertainty_type=<... or None>,
        n_samples=<sourced>, sample_unit=SampleUnit.<...>, units="<paper-native units>"))
```

The `EnvironmentResponseExperimentReference` carries the parent-strain baseline in the same
environment (vanacloig L310-326: `environment_response=0.0`, `ReferenceGenome`).

**Wire it up:** export in `__init__.py`; add the `ENVIRONMENT_RESPONSE_DATASETS` runner
entry (root, `expected_count`, `background_genes=frozenset()`, `Provenance` with the Zenodo
sha256 + `citation_key` + method string). **Run L0-L4** via
`~/miniconda3/envs/torchcell/bin/python` (build LMDB synchronously, then
`run_environment_response`). Add both tests. **Land via `/enqueue-merge`** (one at a time —
shared `.git`).

**OUT OF SCOPE for v1 (explicit):** the NPV cross-study aggregate entity + its
2-hop-provenance model (the deferred supervised decision, Open Q1); the 534-paper fan-out;
heterozygous/diploid screens; ChEBI/InChIKey/SMILES resolution; the 11 legacy `is_synthetic`
rebuilds (`#92`).

## Gotchas

1. **No machine PMID index.** There is no `DATASET_ROWS` in `paper/tables.py` and no
   pubmed→built lookup. The skip list is hand-assembled from three sources (Decision 6);
   miss one and you double-count a screen already in the DB. Sidestep: build the map,
   commit it to the dendron note, `grep -rn "pubmed_id\|doi=" torchcell/datasets/` as the
   backstop.
2. **1GB mutable git `master` vs the pinned Zenodo archive.** Cloning `master` gives
   unpinnable, drifting bytes (`#70`). Sidestep: fetch the `7714347` archive, pin its
   sha256, extract only the proof subtree.
3. **Silent gene-drop (`#73` class).** `alias_to_systematic` misses (dead/merged/renamed
   names) must be dropped-AND-logged with a review threshold, never silently skipped — a
   silent drop is exactly the caudal2024 defect. Sidestep: log each dropped token, assert a
   drop-rate flag, cross-check `Utils/yp_sgd_features.txt`.
4. **Unnameable compounds still require a name.** `Compound.name` is required; a bare
   condition-id is a placeholder only when the paper truly gives no chemical name, and must
   be flagged in the note (not silently shipped as if resolved).
5. **Per-study SI uncertainty sourcing is the real cost.** One study's `n_samples` may take a
   full Methods/SI comb (CLAUDE.md rule; retry, check column descriptions + synonyms). This
   is why v1 is one study — 534× this cost is the out-of-scope wall.
6. **Do NOT branch off stale `origin/feat/media-component-schema`.** That branch reverts
   `torchcell/literature/` — branching off it would silently undo the merged literature
   subsystem. Sidestep: branch off current `main` in a fresh worktree.
7. **Build synchronously; never detach.** The LMDB build must complete before the verifier
   runs; do not background-and-forget. ONE LMDB per PMID (never a shared store).

## Verification

- **L0-L4** via `torchcell/verification` `run_environment_response(data_root)` (runners.py
  L1203) over the new `ENVIRONMENT_RESPONSE_DATASETS` entry — reuses
  `torchcell/verification/environment_response.py` UNCHANGED.
- **pytest** `tests/torchcell/datasets/scerevisiae/test_<study>.py` (record construction +
  `expected_count`, patterned on `test_cachera2023.py`) AND the NEW
  `tests/torchcell/verification/test_environment_response_verification.py` (closes the
  verification-test coverage gap).
- **mypy** whole-tree locally (CI is diff-scoped and would miss a whole-tree regression —
  the standing `torchcell-ci-mypy-diff-scoped` lesson), **ruff** on changed files.
- **Manual smoke:** build the one LMDB with `~/miniconda3/envs/torchcell/bin/python
  torchcell/datasets/scerevisiae/<study>.py`, assert `len(dataset) == expected_count`, and
  spot-check ONE record — its `environment_response` must equal the paper's / value-file's
  original value for that (gene, condition) pair (not a z-score), and its `n_samples`/units
  must match the sourced SI value.

## Open Questions

Flagged AMBIGUOUS — for the user, prose dialogue not `AskUserQuestion` cards:

1. **The aggregate 2-hop-provenance / NPV curation-layer model.** How (and whether) to
   represent YeastPhenome's cross-study NPV as a first-class derived entity that points back
   at each per-PMID record — the real supervised decision. Out of scope for v1; v1
   deliberately builds only the per-study layer so this stays open.
2. **Proof-study PMID selection.** Approve the four selection criteria (Approach) and/or pick
   the specific study. The plan defines criteria; it does not silently choose.
3. **Curation-vs-primary sourcing stance.** For v1 the value comes from YeastPhenome's Zenodo
   re-host with the primary PMID as `Publication` (Decision 3). This sets
   `source_url`/`retrieval_method`/citation for EVERY future YeastPhenome study — confirm
   before it becomes the harness default, vs. always cross-checking each primary paper's own
   SI (higher rigor, much higher cost).

## 2026.07.15 - `ProvenanceGap`: the honest-typed-absence affordance (name settled)

Resolves the affordance that was tentatively named "capture gap." **Name settled:
`ProvenanceGap`** (NOT "capture gap"). This reframes v1's ambition slightly and is the
mechanism that makes it safe.

### Reframed v1 goal (per user)

v1 = **capture the ENTIRE dataset as YeastPhenome presents it** and map every field they
hand us into the EXISTING ontology (`EnvironmentResponseExperiment` family). Wherever a
schema field cannot be sourced from the curation layer we are consuming, we do NOT guess
and do NOT silently `None` it — we attach a **`ProvenanceGap`** naming the field and the
typed reason it is absent. The union of all gaps across the build is a **queryable
worklist** of exactly what remains to be done (the per-paper SI comb). This is what
generalizes cleanly to **SPELL** and to the eventual 534-paper fan-out: capture-now,
document-the-gap, close-later.

This does NOT relax the CLAUDE.md "Adding Datasets" sourcing rule — it is the rule's
honest escape hatch. A value that IS sourceable from the curation layer is still stored
with its `SourcedValue`; only genuinely-absent fields get a `ProvenanceGap`. The rule
"never guess" is preserved verbatim; `ProvenanceGap` is how you obey it without dropping
the record.

### The model (complement of `SourcedValue`, same module)

Lives beside `SourcedValue` in `torchcell/verification/sourced.py` — the two are the
two halves of one idea: a value WITH provenance vs. a documented ABSENCE of one.

```python
class ProvenanceGapReason(StrEnum):
    """Where in the provenance chain the value died -- an honest, typed absence."""
    not_reported_by_primary = "not_reported_by_primary"        # original screen never measured/reported it
    not_carried_by_curation = "not_carried_by_curation"        # primary had it; the secondary DB (YeastPhenome/SPELL) dropped it in aggregation
    deferred_pending_source_review = "deferred_pending_source_review"  # recoverable -- the per-paper SI comb just isn't done yet (the ONLY actionable reason)


class ProvenanceGap(BaseModel):
    """The complement of SourcedValue: a documented, typed ABSENCE of a sourced value.

    SourcedValue binds a value to (citation_key, sha256, quote). ProvenanceGap binds a
    MISSING value to a typed reason, so an unfilled n_samples / uncertainty is an honest
    typed absence -- never a guess, never a silent None. The set of gaps across a build
    is a queryable worklist.
    """
    model_config = ConfigDict(extra="forbid")

    field: str = Field(description="name of the field on THIS record that is unsourced (e.g. 'n_samples')")
    reason: ProvenanceGapReason
    looked_in: Provenance | None = Field(
        default=None,
        description="the secondary source we DID consult (e.g. the YeastPhenome Zenodo re-host) -- anchors even the absence",
    )
    resolve_with: Provenance | None = Field(
        default=None,
        description="deferred_pending_source_review only: the primary artifact whose SI would close the gap",
    )
    note: str | None = None
```

### Placement (the one genuine fork — for the user)

Recommended: attach `provenance_gaps: list[ProvenanceGap] = []` on the **`Phenotype`
base** (schema L1456), inherited by every phenotype. Two validators, both mirroring the
existing `Phenotype.validate_label_fields` pattern (L1486):

1. **Field-name check** — each `gap.field` must name a real field on the concrete
   subclass (`type(self).__annotations__`), exactly as `label_name` is checked. Self-
   validating, no dotted paths.
2. **Honesty invariant** — a field named by a gap must be `None` on the record
   (`getattr(self, gap.field) is None`). You cannot both store a value and declare it
   missing. THIS is what makes the affordance machine-checkable, not decorative.

Why phenotype-level for v1: the fields that actually go missing in YeastPhenome
(`n_samples`, `environment_response_uncertainty`, `sample_unit`) are ALL phenotype
fields, so the gap sits next to the value it is about and the field-name check is exact.
No conflict with `EnvironmentResponsePhenotype._check` (L2733): that validator only
requires `n_samples`/`sample_unit` when a `sample_sd`/`variance` uncertainty is present;
if uncertainty itself is a gap, uncertainty is `None` and the requirement never fires.

Alternative (deferred, not v1): a record-level `Experiment.provenance_gaps` so gaps can
also cover environment/genotype-axis fields, at the cost of dotted-path field references
(`"environment.perturbations[0].concentration"`) that are harder to validate. Phenotype-
level covers v1; Experiment-level is the natural later extension when a non-phenotype
field needs a gap.

### What this changes elsewhere

- **Verifier (L0-L4):** add an optional pass that COLLECTS the gap-set from a built LMDB
  and reports it (count by reason, list of `deferred_pending_source_review` fields = the
  worklist). Does not fail the build — a documented gap is a PASS, a silent `None` or a
  guess is the failure mode this prevents.
- **Generalizes to SPELL** unchanged: same three reasons, same complement-of-SourcedValue
  shape; SPELL's per-series metadata gaps map to `not_carried_by_curation` /
  `deferred_pending_source_review` identically.

### Resolved + IMPLEMENTED (2026-07-16)

User confirmed **Phenotype-base placement** and **verifier pass in v1**. Both are now
built on branch `plan/yeastphenome-ingestion-harness` (not yet landed):

- `torchcell/verification/sourced.py` — added `ProvenanceGapReason` (StrEnum),
  `ProvenanceGap` (complement of `SourcedValue`, same module), `ProvenanceGapCensus`,
  `provenance_gap_census()`, `provenance_gap_level_result()`, `l1_provenance_gaps()`.
- `torchcell/datamodels/schema.py` — `Phenotype.provenance_gaps: list[ProvenanceGap]`
  - `validate_provenance_gaps` (field-name-real via `model_fields` + honesty invariant:
  a gapped field must be `None`).
- `torchcell/verification/environment_response.py` — both the eager and streaming
  verifiers now emit the L1 `provenance_gaps` census (always-pass; deferred fields =
  worklist).
- `torchcell/verification/__init__.py` — exports the six new names.
- Tests: extended `tests/torchcell/verification/test_sourced.py`; NEW
  `tests/torchcell/verification/test_environment_response_verification.py` (also closes
  the env-response verifier coverage gap). 21 new/extended assertions; full
  verification + datamodels suites green (395 passed), mypy + ruff clean.

Still open (unchanged): Open Q1 (NPV 2-hop aggregate),
Q3 (curation-vs-primary sourcing stance). The `ProvenanceGap` machinery is what makes
the reframed "capture-whole-then-gap-mark" v1 safe.

## 2026.07.16 - Proof-study PMID SELECTED (Open Q2 resolved): Khozoie & Avery 2009 (quinine)

**Pick: `khozoie_avery_2009` — PMID `19416971`** — "The antimalarial drug quinine
disrupts Tat2p-mediated tryptophan transport and causes tryptophan starvation" (Khozoie,
Pleass & Avery, *J Biol Chem* 2009). YeastPhenome dataset `Datasets/19416971/`.

### How the catalog was built (reusable — this IS the Decision-6 PMID reconciliation harness)

The YeastPhenome data lives in GitHub `yeastphenome/yp-data` (= Zenodo `7714347` =
`yp-data-v1.0.zip`, 609.6 MB); its website is **out of service** (funding lapse), so the
catalog was reconstructed FROM THE REPO, not the site:

1. `git/trees/<sha>?recursive=1` (ONE call) → all **534** `Datasets/<PMID>/<author>_<year>_value.txt`
   paths (avoids the 60/hr unauth API limit; `raw.githubusercontent.com` has no such limit).
2. Range-fetched the **first line** of all 534 value files. The header is a machine catalog:
   each data column is `<hom|het> | <phenotype> | <condition> | <base media> | <author>`, so
   `hom`/`het`, phenotype class, and condition count are all parseable WITHOUT the website DB.
3. Filtered: purely `hom` + a `growth` phenotype + PMID not in our built skip list + author
   not matching a built loader → **110** homozygous-growth candidates; **54** single-condition.
4. Downloaded the value data for the cleanest single-stress candidates and checked the value
   distribution to separate CONTINUOUS scalars from binned/hit-list files.

Skip list (from our loaders' `pubmed_id`/`doi`, cross-checked by author name in the header):
costanzo 27708008, kemmeren 24766815, cachera 37572348, caudal 38778243, kuzmin 29674565/32586993,
dasilveira 25143408, messner 37080200, mulleder 27693354, ohnuki 29768403/35087094, ohya 16365294,
ozaydin 22918085, sameith 26687005, yoshida 22277779, zelezniak 30195436, smith 16738555,
oduibhir 24952590, lian 31857575, lopez 35022416, mormino 36284296, xue 23899824, hillenmeyer
18420932, costanzo2021 33958448 (+ baryshnikova/hoepfner/wildenhain/auesukaree/vanacloig/mota
excluded by author-name match). **Reusable artifacts left in the session scratchpad**
(`valpaths.txt`, `hdrs/`, the parse script) so the next YeastPhenome study reuses the catalog.

### Why this study wins the four criteria

| criterion | evidence |
|---|---|
| (a) NOT already built | Avery-lab quinine screen; absent from the skip list above |
| (b) non-expression, homozygous KO × chemical | homozygous diploid **BY4743** collection × **quinine 2 mM** in **YEPD, 30 °C**, OD600 read at 7 h (paper Methods) |
| (c) continuous scalar original value | value.txt = **4199 distinct** continuous values (`GR2_adjusted` = plate-median-normalized inverse growth ratio), range **[0.457, 2.68]**, 4228 ORFs — a clean scalar, NOT a hit list or binned class |
| (d) n_samples + uncertainty sourceable | paper Methods (OPEN ACCESS **PMC2709357**) state the design verbatim: *"a mean growth ratio ... (n = 2) from the initial screen ... re-arrayed ... screened three further times in duplicate"*; growth ratio = A600(control)/A600(quinine) |

**Bonuses.** (i) OPEN ACCESS (PMC2709357) → mirror-able + the replicate design is sourceable
(no paywall wall like the FEMS-YR alternatives). (ii) The quinine → Tat2p → tryptophan-
starvation mechanism ties directly to the flagship YeastPhenome finding that *tryptophan
biosynthesis* is the exceptional pathway required for resistance to >1000 compounds — a clean
narrative thread. (iii) Bioproduction-adjacent chemical-stress readout, consistent with the
existing Vanacloig anaerobic-hydrolysate-toxin anchor.

### Runners-up (kept as fallbacks)

- `khozoie`'s sibling `islahudin_avery_2013` (chloroquine, PMID 23733464) — same Avery QGA
  growth-ratio method, ~4196 continuous values; a near-identical second proof if needed.
- `ando_shima_2006` (high-sucrose osmotic stress, PMID 16487347) — cleanest continuous
  distribution centred at 1.0 and MORE on-theme (fermentation osmotolerance), BUT the paper
  is FEMS Yeast Res (paywalled) so criterion (d) sourcing is harder → deprioritised.
- **Rejected on inspection:** `endo_shima_2008` vanillin (PMID 18471310) — thematically ideal
  but the value.txt has only **26 distinct values** (binned), fails (c); `zhao_deng_2020` zinc
  (PMID 31836620) — only **108 rows all `-1`** (a hit list, not a screen), fails (c).

### ProvenanceGap angle for THIS study (the demonstration)

The value.txt carries ONE `GR2_adjusted` per ORF and **no per-strain n or SD**. So the loader
demonstrates BOTH paths: store the value (`measurement_type` = `sensitivity_score`, a
growth-ratio; `units` = "plate-median-normalized inverse OD600 growth ratio, control/quinine";
Decision-2 native-statistic stance) AND either (a) source `n_samples = 2` +
`sample_unit = biological_replicate` from PMC2709357 as a `SourcedValue`-backed constant
(criterion d met), OR (b) if we decide the per-record replicate structure isn't cleanly
2-for-all (hits were re-screened 5×), gap-mark `n_samples` as
`deferred_pending_source_review` — the exact honest-absence the affordance exists for. The
loader PR resolves which, and records the decision in the study's dendron note.

### NEXT

Mirror `Datasets/19416971/` + PMC2709357 (sha256-pinned), add `zenodo` to `RetrievalMethod`,
write `torchcell/datasets/scerevisiae/khozoie2009.py` on the `vanacloig2022.py` + genome-
injection pattern, wire the runner + tests, run L0-L4, `/enqueue-merge`. Open Q3 (curation-
vs-primary sourcing stance) gets its first concrete answer here.

## 2026.07.16 - Encoding PROBE + two design decisions (temperature gap; NPV z-score label)

Ran a **5-record probe** building real curated khozoie records with the schema (NOT the
loader) to answer: can the ontology hold a curated YeastPhenome record, or is it
underspecified? **Answer: structurally YES (L0 passes), with 4 underspecifications, 2
sharp.** The two sharp ones each forced a decision (below); both now RESOLVED + IMPLEMENTED.

### What curated YeastPhenome gives per record

`value.txt`/`valuez.txt` first col = systematic `orf`; second col = one float. Header meta
= `<hom|het> | <phenotype> | <condition> | <base media> | <author>`. So we get: ORF, zygosity
(hom), phenotype class (growth), compound+dose (quinine 2 mM), base media (YPD+EtOH), the
value. We do NOT get: temperature, n_samples, uncertainty, replicate design, common gene name.

### Decision A -- temperature = Environment-level `ProvenanceGap` (user: "type it with gaps")

`Environment.temperature` was REQUIRED but curation does not carry it, and `ProvenanceGap`
only lived on `Phenotype`. RESOLVED: extracted a shared **`ProvenanceGapMixin`** (schema.py)
holding `provenance_gaps` + the 2 honesty validators; **`Phenotype` AND `Environment` now
both inherit it**; **`Environment.temperature` is now `Temperature | None = None`**. A curated
record with no temperature sets `temperature=None` + `ProvenanceGap(field="temperature",
reason=not_carried_by_curation, looked_in=Zenodo7714347)` -- a typed absence, never a guessed
30 C. The verifier L1 census now aggregates BOTH phenotype + environment gaps. User's
"worst case everything gets gapped, don't know why" concern is answered by design: every gap
carries a `reason` + `looked_in`, so a fully-gapped record is fully EXPLAINED (not a silent
None), and the census turns gap-density into a queryable per-dataset fidelity score. Fill
temperature when curation DOES carry it (heat-stress screens encode it in the condition
string); gap only when genuinely absent.

### Decision B -- store the NPV z-score as the label (flips plan Decision 2)

Reread the mirrored Turco paper (`$DATA_ROOT/torchcell-library/turcoGlobalAnalysisYeast2023/
paper.md`). YeastPhenome's canonical label output = **NPV (normalized phenotypic value): a
per-screen, mode-referenced modified z-score** (paper L28/L44): the MODE of each screen (the
"most typical mutant" ~ WT) is the 0-reference, values are standardized deviations (SD units),
`|NPV|>3` = strong. Growth of any modality (colony size / turbidity / pooled abundance) is ONE
phenotype (L146). Two data forms shipped: `value.txt` (original native units) + `valuez.txt`
(NPV). The exact modified-z formula is DEFERRED to note S4 + the (down) website -> itself a
`deferred_pending_source_review` gap on the normalization method. **DECISION (user-approved):
store the NPV `valuez.txt` as `environment_response` with `measurement_type=z_score`,
reference=0** -- this is what YeastPhenome PRESENTS as its label, and it fits our schema's
signed-score-centred-at-0 convention. This FLIPS plan Decision 2 (which said store original,
reject z-score): under "capture as presented," the NPV is the presented label; the original
value.txt is the pre-transform input (store as an annotation if wanted, not the label).

### Probe result (both decisions applied, real NPV + gapped temperature)

`verify_environment_response_dataset` on 5 real khozoie NPV records: **PASS L0-L4**, incl.
`reference_zero` (WT NPV=0 -- FAILED before with the ratio-centred original). Gap census: 20
gaps / 5 records, `by_field {n_samples:5, environment_response_uncertainty:5, sample_unit:5,
temperature:5}`, all `not_carried_by_curation`. So a curated YeastPhenome record encodes
cleanly + honestly with the ontology as now extended.

### Implemented (branch `plan/yeastphenome-ingestion-harness`, not landed)

`schema.py`: `ProvenanceGapMixin` (Phenotype+Environment inherit); `Environment.temperature`
optional. `environment_response.py`: census aggregates phenotype+environment gaps (eager +
streaming). Tests: +4 in `test_environment_response_verification.py` (env temp gap optional /
honesty / census eager+streaming). **398 verification+datamodels tests pass, mypy + ruff clean.**

### Softer underspecifications (noted, not blocking)

- Deletion MECHANISM (KanMX) is inferred from "it's the YKO collection" -- curated file says
  only "hom deletion". Homozygous diploid = total absence -> `KanMxDeletionPerturbation` +
  `ReferenceGenome(ploidy="diploid")` (schema's HOP convention). Fine, but mechanism is
  assumed not stated.
- Only the ORF is given -> `perturbed_gene_name` set to the ORF (no common name in curation;
  resolvable via genome later).

## 2026.07.16 - ARCHITECTURE REVISION: ONE YeastPhenome growth loader (Decision 1 overturned)

User pushed back on per-PMID loaders: we consume ONE pre-processed intermediate (yp-data /
Zenodo 7714347), so writing 534 loaders is wrong. **Decision 1 is OVERTURNED.**

**NEW architecture: a SINGLE `YeastPhenome` (Turco 2023) env-response loader.** One dataset
class, one mirror record (Zenodo 7714347), one LMDB. Each record = `(homozygous deletion x
screen-condition -> NPV z-score)`; the **screen (PMID + YeastPhenome dataset_id) becomes
per-record provenance + a study identifier**, NOT a separate loader. khozoie (PMID 19416971)
is the FIRST screen validated INSIDE this loader (the probe), not a `khozoie2009.py` file.

**Why one homogeneous loader is coherent:** YeastPhenome already harmonized every screen to
the NPV (mode-referenced z-score, Decision B), so a single `measurement_type=z_score` +
reference=0 spans ALL screens. Storing each screen's original native units would be
incommensurable and force per-screen handling; consuming the harmonized NPV is what makes
"one loader" possible. The condition (compound/stress) is the environment axis -- exactly
what `EnvironmentResponseExperiment` is for (many conditions, one phenotype type).

**Scope = the GROWTH phenome only (the 53%).** YeastPhenome splits ~53% growth / ~42%
expression / ~5% mosaic (paper L23), and that split maps onto PHENOTYPE FAMILIES:

- **53% growth** -> THIS loader (`EnvironmentResponseExperiment`, NPV label). The 110
  homozygous-growth non-skip screens from the header catalog.
- **42% expression** (Kemmeren) -> ALREADY BUILT directly (`kemmeren2014.py`); EXCLUDED
  (user 2026-07-16: "we already have processed kemmeren so we don't need it again").
- **5% mosaic** -> ~670 specialized non-growth phenotypes (proteome / metabolome / morphology
  / localization / intracellular pH / genome state), each a DIFFERENT phenotype family
  (`ProteinAbundance`/`Metabolite`/`CalMorph`/...). Several already built from primaries
  (Zelezniak, Messner, Mulleder, da Silveira, Ohya, Ohnuki); the rest not-yet-modeled. NONE
  belong in the growth loader. So "one YeastPhenome loader" = the growth slice, the only part
  that is both one homogeneous family AND not already covered.

**Exclude already-built primaries** (skip list) -- don't re-ingest Costanzo/Kemmeren/Kuzmin/...
via YeastPhenome (Open Q3 default = exclude, user-confirmed for Kemmeren). Build the loader
incrementally: seed with khozoie (1 screen, L0-L4 green), then widen the screen set to the
110 growth-minus-skip screens (same code, longer screen list).

## 2026.07.17 - LOADER BUILT + L0-L4 GREEN (khozoie seed)

`YeastPhenomeDataset` (`torchcell/datasets/scerevisiae/yeastphenome.py`) written + built +
verified end-to-end. **The single-loader architecture works.**

- **Source pin:** raw files fetched from `yeastphenome/yp-data` @ **v1.0 tag** (commit
  `83e2917bf86955ec6ba66dc70ff2ed0fe24ecbe8` = the Zenodo 7714347 freeze, NOT `master` which
  drifted to 2025), each verified against its pinned sha256 (valuez
  `99eb77d2...`, value `d4c2ba35...`). `RetrievalMethod.zenodo` added.
- **`SCREENS` list** drives the loader (seed = khozoie PMID 19416971); widening = append.
- **Label = NPV** from `valuez.txt` -> `measurement_type=z_score`, reference 0.
- **Gaps:** env `temperature`=None + gap; phenotype `n_samples`/uncertainty/`sample_unit` gaps;
  all `not_carried_by_curation`, `looked_in`=Zenodo. **16912 gaps over 4228/4228 records.**
- **Genotype:** homozygous diploid = `KanMxDeletionPerturbation` + `ReferenceGenome(ploidy=
  "diploid")`; ORF-only (no common name carried).
- **Build:** 4228 records, 0 non-ORF dropped. **L0-L4 PASS** (incl. `reference_zero` on NPV=0,
  `gene_containment_sgd`=1.000). Runner entry `yeastphenome` added to
  `ENVIRONMENT_RESPONSE_DATASETS`.
- **Tests:** `test_yeastphenome.py` (6 build-smoke tests: count, NPV-label round-trip, hom-
  diploid genotype, condition/media, **typed-gaps-not-guesses**, source-PMID). 404
  verification+datamodels+dataset tests pass; mypy + ruff clean.
- **Double-count guard (Kemmeren concern):** Kemmeren is EXPRESSION so the growth filter
  already excludes it; the growth-screen primaries (Costanzo/Kuzmin/Hoepfner/...) that DO pass
  the growth filter are caught by the PMID+author skip list. When widening, log the excluded
  screens so nothing already-built slips in.

Build gotchas hit + fixed: (1) running the loader as a plain script imported the PRIMARY
checkout's torchcell (no ProvenanceGap) -- must run `PYTHONPATH=$PWD python -m
torchcell.datasets.scerevisiae.yeastphenome` from the worktree. (2) `Publication` requires a
URL, not just `pubmed_id` -> pass `pubmed_url`. (3) a failed first build left an empty
`processed/lmdb` that PyG then reused (len=0) -> `/deprecate` the stale `processed/` to force
reprocess (`rm -rf` blocked; deprecate.sh lives in the PRIMARY checkout, not this worktree).

### NEXT (widen)

Add the remaining ~109 homozygous-growth non-skip screens to `SCREENS` (each: pmid, stem, both
sha256s). Generalize `_parse_condition` (physical stresses, ranges, multi-compound) +
`_parse_media`; drop-and-log screens it cannot parse (never guess). Confirm the skip list
against each candidate PMID + LOG exclusions. Re-verify L0-L4 on the widened build.

## 2026.07.18 - WIDENED to 20 screens / 38 environments (L0-L4 GREEN)

`YeastPhenomeDataset` now ingests **20 homozygous-growth screens** (v1.0 pin), each parsed
COLUMN-BY-COLUMN with drop-and-log. **140,264 records over 38 environments; L0-L4 PASS.**

- **Pin corrected to the v1.0 TREE** (`9de026a9...`), not master: v1.0 has **513** valuez files
  (master drifted to 534). Skip list made airtight by resolving the DOI-only built loaders to
  PMIDs via NCBI esearch (auesukaree 19638689, baryshnikova 21076421, hoepfner 24360837,
  wildenhain 27136353, costanzo2021 33958448, smith2016 26956608, vanacloig 35883225, mota
  38419072, nadal 40102404, hillenmeyer 18420932). **6 already-built screens excluded** (ohya,
  smith2006, auesukaree, baryshnikova, ozaydin, costanzo2021); Kemmeren absent (expression, no
  growth valuez) -- user's double-count concern satisfied on both axes.
- **Column-aware loader:** `SCREENS` = {pmid, stem, valuez_sha256}; the loader parses EVERY
  data column, keeps hom + growth + single-dosed-compound + known-media columns, drop-and-logs
  the rest (18 unparseable + 8 unknown-media + 2 het WITHIN the 20 screens; ~488 screens have 0
  encodable columns -> the worklist). valuez-only now (value.txt no longer mirrored; at the pin
  if needed).
- **Two verifier issues the multi-screen build surfaced + fixed:**
  1. `_parse_condition` mis-parsed multi-component conditions (`time [5 gen], X [1 uM]`) ->
     TIGHTENED to reject any name with `[`/`,` (drop-and-log), never mis-parse.
  2. `pair_uniqueness` FAILED (12,742 dup pairs): YeastPhenome's NEAR-REPLICATE screens --
     `berry_gasch_2011` runs the SAME conditions by microarray AND barseq (separate screens).
     Fix (principled + backward-compatible): the record's READOUT METHOD (from the phenotype
     string) is recorded in phenotype `units`, and the verifier's uniqueness key is now
     **(study+units, strain, condition)** via `_study_key` (returns `(pubmed_id/doi, units)`).
     Single-study/single-assay datasets unaffected (context constant). After the fix: **140,264
     unique (study, strain, condition) records, one each.**
- **Verify:** L0 140264 validated; L1 count + uniqueness OK; 561,056 gaps (4/record);
  L3 reference_zero (NPV=0) + measurement_type(z_score) OK; L4 gene_containment 1.000 over 4974
  genes. `expected_count` in runner = 140264. Test rewritten to a 2-screen subset (khozoie +
  berry_gasch) via monkeypatched SCREENS -- asserts NPV label, hom-diploid, typed gaps, and the
  **near-replicate distinctness** (same gene+condition, microarray vs barseq -> 2 records). 404
  tests + mypy + ruff green.
- **Build gotcha:** the stale `preprocess/` from the seed build had `gene_set.json` owned by
  **uid 7474** (KG-build user) -> `PermissionError` on the widened post_process. `/deprecate`
  the preprocess dir (I own the dir, so `mv` works even with 7474-owned files inside) before
  rebuilding.

## 2026.07.19 - CORRECTION: Decision 4 "homozygous only" was WRONG -- haploid screens included

**User caught a scoping error that cost more than half the data.** Plan Decision 4 said
"homozygous only", and the loader filtered `zygosity == "hom"` -- which silently excluded all
HAPLOID screens too.

**Why it was wrong.** Decision 4's real intent was to exclude **heterozygous** diploid screens
(gene DOSAGE / haploinsufficiency = a different perturbation type needing
`EngineeredCopyNumberPerturbation`). A **haploid** deletion is COMPLETE loss-of-function --
the SAME total-absence `DeletionPerturbation` as a homozygous diploid; only
`ReferenceGenome.ploidy` differs, which we already model. And haploid is torchcell's DOMINANT
convention: across our loaders `BY4741` x16, `BY4742` x7, vs `BY4743` x2. Excluding haploid
contradicted our own reference-strain practice.

**Impact (encodable growth+single-compound+known-media columns, non-skip screens):**
`hap a` **40** cols / 24 screens; `hom` 38 / 20; `hap alpha` **4** / 3; `het` 4 (correctly
excluded). So `hap a` ALONE exceeded `hom` -- we were ingesting less than half.

**Fix.** New `_ZYGOSITY` map (background -> strain, ploidy): `hom`->(BY4743, diploid),
`hap a`->(BY4741, haploid), `hap alpha`->(BY4742, haploid), + post-SGA variants. `_reference`
takes strain/ploidy per column. Only `het` and ambiguous/mixed collections ("hap ?",
"hap a/hap alpha") are dropped-and-logged.

**Result: 20 -> 47 screens, 38 -> 83 environments, 140,264 -> 296,777 records. L0-L4 PASS**
(1,187,108 gaps; pair_uniqueness 296,777 unique; gene_containment 1.000 over **5011** genes;
67 distinct compounds; 47 source PMIDs). `expected_count` = 296777. New test
`test_haploid_screens_are_ingested_as_haploid_background` (pagani_arino 17630978 -> BY4741,
ploidy=haploid, same `kanmx_deletion`/`state=absent`) locks the fix in. 404 tests + mypy +
ruff green.

**Double-count re-checked after widening: still 0 leaks** (see the reconciliation below).

### Overlap reconciliation (built torchcell datasets vs YeastPhenome v1.0)

Answering "besides Kemmeren, what else overlaps?" -- checked programmatically over all 517
v1.0 `Datasets/` dirs vs all 32 built loaders:

- **6 built datasets appear in YeastPhenome WITH an NPV file, ALL excluded by the skip list:**
  ohya2005 (16365294), smith2006 (16738555), auesukaree2009 (19638689), baryshnikova2010
  (21076421), ozaydin2013 (22918085), costanzo2021 (33958448). NOTE each is *also* excluded
  independently by the zygosity/phenotype filter (all are haploid-or-non-growth), so the skip
  list is belt-and-braces.
- **3 built datasets are YeastPhenome dirs with NO `valuez` at all** (nothing ingestible):
  kemmeren2014 (24766815), hillenmeyer2008 (18420932), hoepfner2014 (24360837). Only 4 of 517
  dirs lack a valuez and 3 of those 4 are ours -- the three biggest double-count risks ship no
  NPV matrix in the freeze.
- **23 built datasets are absent from YeastPhenome v1.0 entirely** (costanzo2016, kuzmin2018/
  2020, wildenhain2015, smith2016, vanacloig2022, oduibhir2014, sameith2015, + the
  metabolome/proteome/morphology sets).
- **`retained ∩ built = {}`** -- verified empty after the widening.

### NEXT (further widen -- optional)

The remaining coverage is behind the parser's conservative scope: capture **physical-stress**
conditions (temperature/anaerobic/desiccation/irradiation -> `EnvironmentPhysicalPerturbation`
or `Environment` scalars), **dose ranges**, **multi-compound**, **time/generation** components,
and **non-standard media** (extend the media map). Each is a drop-and-log line today (the
worklist); adding them lifts coverage past the current 38 environments. Also: `cell_adapter.py`
None-safe temperature (the standing follow-up) before KG ingest.

### FOLLOW-UP (do before the KG build ingests a gapped-temperature dataset)

**Only `torchcell/adapters/cell_adapter.py`** (the CPython BioCypher adapter) accesses
`environment.temperature.value` (attribute access on the pydantic object) UNGUARDED -> a
gapped `temperature=None` record would AttributeError in the KG build. Latent today (no
dataset emits None), mypy can't catch it (accessed off dict-typed `Any`). **Make
`cell_adapter.py` None-safe when the khozoie loader lands.** NOT a concern: the **pypy
adapters are DEPRECATED** (not used); `datasets/experiment.py:163` is SAFE (reads temperature
only under an explicit `preprocess["temperature"]` filter via dict access + `!=` compare, so
a None temperature just skips the record -- graceful, no crash).
