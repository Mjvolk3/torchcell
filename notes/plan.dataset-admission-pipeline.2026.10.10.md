---
id: sxo4nxko6dco17mbrg49qlh
title: '10'
desc: ''
updated: 1791613583748
created: 1791613583748
---

## Context

Hand selection let three aggregation corpora through to worktrees before anyone refused
them: D2Cell 2026 (`experiments/database/scripts/build_bacteria_candidate_datasets_table.py:1382`),
MCF2Chem 2023 (`:2180`), CeCaFDB (`:3913`). All three carry `status="aggregation"`, but
that field is read only for a table glyph (`:5819`) and a summary count (`:6428`); nothing
refuses on it, and two of the rows still rank into the reserve. Wave 4 (rows 51-61,
2026-10-10) measured the cost of deciding late: D2Cell's primary-measurement refusal
(PR `#858`), Thompson 2019 lysine subsumed inside Borchert 2024 (`#856`, 156 values, max
diff 0.046), Balakrishnan 2022's retrieval wall plus a count-less expression table
(`#853`, `#854`), Li 2014's missing degradation rate (`#857`). Every one was found AFTER a
worktree, a deposit, or a loader attempt.

This plan adds a typed candidate gate that runs BEFORE loader effort, an agentic
`/add-dataset` pipeline from candidate to landed PR, a re-audit of the bacterial and yeast
candidate lists against the gates, and a CI tripwire so a dataset class cannot register
without a recorded verdict. It replaces nothing downstream: `kg_manifest admit`
(`torchcell/knowledge_graphs/kg_manifest.py:1278`) stays the last stage and is unchanged
("an admission is a minor release" holds). It also retires the planned pydantic-ai role
loop in `notes/torchcell.knowledge_graphs.dataset-admission-loop.md:45` in favor of the
Agent-tool orchestration the repo already runs. Related open issues: `#758` (sha256 pin
does not make a quote verbatim), `#699` `#788` `#853` `#691` (retrieval walls), `#726`
(PubChem 429 from GilaHyper), `#800` (Li 2021 internally inconsistent release), `#805`
(`wt_cleanup` removed an unpushed commit), `#495` (`lit_sync` treats a raw-mirror key as a
broken capture), `#833` (`check_store` not consulted by `admit`).

## Relevant Files

| path | action | purpose | stance |
|---|---|---|---|
| `torchcell/candidates/{__init__,verdict}.py` | NEW | frozen pydantic verdict + gate records; `model_json_schema()` for agent prompts | n/a |
| `torchcell/candidates/gates.py` | NEW | deterministic G1-G5 evaluators over row fields, registry, mirror bytes | n/a |
| `torchcell/candidates/store.py` | NEW | read/write `database/candidates/<citation_key>.json`; GRANDFATHERED tuple | n/a |
| `torchcell/candidates/ledger.py` | NEW | render the dendron ledger section from the store | n/a |
| `torchcell/candidates/cli.py` | NEW | `candidate-gate` subcommands (`[project.scripts]` entry) | n/a |
| `database/candidates/` | NEW | tracked verdict JSON, one per citation key | n/a |
| `scripts/check_candidate_verdicts.py` | NEW | diff-scoped AST tripwire on new `@register_dataset` | n/a |
| `tests/torchcell/candidates/test_{verdict,gates,store,ledger,cli}.py` | NEW | paired tests (paired-tests hook) | n/a |
| `tests/torchcell/datasets/test_candidate_verdicts.py` | NEW | set-based enforcement over `dataset_registry` | n/a |
| `tests/torchcell/scripts/test_check_candidate_verdicts.py` | NEW | tripwire test, `test_check_paired_tests.py` shape | n/a |
| `experiments/database/scripts/reaudit_candidate_gates.py` | NEW | re-audit over 61 bacterial rows + yeast list | n/a |
| `.claude/skills/add-dataset/SKILL.md` | NEW | the pipeline skill | n/a |
| `notes/torchcell.candidates.ledger.md` | NEW | generated dendron ledger (dated sections) | n/a |
| `experiments/database/scripts/build_bacteria_candidate_datasets_table.py` | MODIFY | render a verdict column from the store; never touch `sort_key` (`:308`) | in-flux |
| `experiments/database/scripts/build_candidate_datasets_table.py` | MODIFY | same column; widen `Status` (`:108`) and `Excluded.rule` (`:241`) | in-flux |
| `.pre-commit-config.yaml` | MODIFY | add `candidate-verdicts` local hook beside `paired-tests` (`:66`) | stable |
| `pyproject.toml` | MODIFY | `candidate-gate` entry point; `[tool.torchcell.candidate_grandfathered]` is NOT added (tuple lives in code) | stable |
| `scripts/ops.sh` | MODIFY | `candidates` action beside `status`, `releases`, `health`, `sync` (`:308-314`) | stable |
| `notes/torchcell.knowledge_graphs.dataset-admission-loop.md` | MODIFY | dated section: Agent-tool orchestration replaces the pydantic-ai plan | provisional |
| `paper/nature-biotech/sections/backmatter.tex:1044-1083` | REFERENCE | panel c caption names pydantic-ai; figure change scheduled, no edit without go-ahead | tent/final, owner-gated |
| `torchcell/knowledge_graphs/kg_manifest.py` | REFERENCE | `AdmissionReport` (`:399`), `check_admission` (`:1278`): downstream, unchanged | stable |
| `torchcell/sequence/genome/registry.py` | REFERENCE | `resolve` (`:166`), W3110 "in no schema vocabulary yet" (`:53-62`) | stable |
| `torchcell/datamodels/schema.py` | REFERENCE | `BacterialReferenceStrain` (`:1021`), `HaplotypeBlock` (`:6634`), `SegregantGenotype` (`:6683`) | in-flux; never touched by this plan |
| `torchcell/verification/sourced.py` | REFERENCE | `SourcedValue` (`:43`), `ProvenanceGap` (`:147`): evidence types, outside the fingerprinted surface | provisional |
| `torchcell/literature/manifest.py` | REFERENCE | `RetrievalMethod.manual_browser` (`:109`), roles `paper_text`/`si_text` (`:30-35`) | stable |
| `torchcell/datasets/ecoli/{cai2023,typas2008,teteneva2024,mohiuddin2022,wetmore2015}.py` | REFERENCE | settled-row record shapes the verdict composes | stable |
| `torchcell/datasets/dataset_registry.py` | REFERENCE | `dataset_registry` (`:7`), `register_dataset` (`:10`): the enforcement target | stable |
| `experiments/036-dataset-fixes-before-kg-build/scripts/si_audit_loadable_ledger.py` | REFERENCE | `RowState` (`:86`), `markdown_table` (`:1021`): ledger pattern | provisional |
| `experiments/database/scripts/check_candidate_overlap.py` | REFERENCE | DOI/accession/PMID keys (`:23-27`): G4-key reuse | provisional |
| `scripts/check_paired_tests.py` | REFERENCE | `changed_modules` (`:75`), `load_exceptions` (`:43`): tripwire template | stable |
| `scripts/provision_bacterial_genomes.py` | REFERENCE | the `blocked:provision` prerequisite | stable |

## Key Design Decisions

1. **Name it `CandidateVerdict` / `candidate-gate`, never `Admission*`.** `AdmissionReport`
   already means serving admission (`kg_manifest.py:399`); a second `AdmissionVerdict` is the
   collision to avoid. "Dataset-admission pipeline" stays as the program name in prose
   because the loop note already uses it for the whole arc, and one line in every document
   says `kg_manifest admit` is the last stage, unchanged. (D1)
2. **The contract is the pydantic model, orchestration is the Agent tool.** No pydantic-ai
   (absent from every requirements file and the env; the v2 line, 2.55.0 at 2026-10-10,
   would add an API-key-bearing dependency to a package whose CI pins Python 3.13.0 and torch
   cpu) and no Workflow tool (needs an opt-in the owner has not given, and its Explore/Plan
   agents receive no CLAUDE.md, the file the gates depend on). The CLI emits
   `CandidateVerdict.model_json_schema()` into each agent prompt and validates the returned
   JSON with `model_validate_json`; `/add-dataset` fans out with the Agent tool the way
   `uber-implement` and `wt-implement` do. The panel-c role names (retriever, proposer,
   critic, implementer, judge) survive as stage names. (D2)
3. **The model lives in a new `torchcell/candidates/` package.** The schema-impact hook
   fires only on `torchcell/datamodels/(schema|pydant).py` (`.pre-commit-config.yaml:83`)
   and supported-queries on those plus `database/releases/*.json` (`:99`); `candidates/`
   touches neither. Not `provenance/` (served-store freshness and fingerprints), not
   `agents/` (the verdict is not agent code). `SourcedValue` / `ProvenanceGap` stay the
   evidence types because they are deliberately outside the fingerprinted surface. (D3)
4. **Verdict files go in `database/candidates/<citation_key>.json`.** Tracked (CI reads
   them), beside `database/releases/` (verdicts are release-adjacent records, not code),
   and outside the supported-queries pattern, which anchors on `database/releases/`.
   Rejected: `torchcell/candidates/ledger/` as package data, because semantic-release would
   ship every verdict in the PyPI wheel; `experiments/database/results/`, because results
   are untracked by convention. (uncovered 1)
5. **G2 is a typed two-branch precheck keyed off `seq_basis`, with four outcomes.** Branch
   `assembly_set`: every strain's host resolves through `registry.resolve` AND is readable
   by its genome class. Branch `haplotype_mosaic`: BOTH parent sets resolve and read; the
   mosaic is already a schema object (`SegregantGenotype`, `schema.py:6683`) and
   `bloom2019.py:818` resolves parents through the registry with no fallback, so the
   probabilistic representation is a schema decision the registry already serves, not a
   registry fallback. A pooled bacterial population ("the frequency is part of the call")
   is the assembly branch with a designed perturbation. Outcomes: `resolvable_readable`;
   `resolvable_not_ingestible` (W3110, `registry.py:53-62`); `absent_provisionable` (an
   accession is named, so a genomes-tier deposit PR via `provision_bacterial_genomes.py` is
   the prerequisite and the row is `blocked:provision`, not refused); `absent_unnamed`
   (refused, same as `Excluded.rule="no-sequence"`). An engineered chassis (BL21, W, Nissle)
   is therefore `blocked:provision`, never refused; its cassette's declared sequence is a G5
   requirement on the perturbation class. (D4, uncovered 2)
6. **Mohiuddin is the G3 precedent for a paper absent from Zotero.** `mohiuddin2022.py:698`
   retrieves PMC OA bucket objects into `torchcell-raw` under `pmc_cloud` with
   `paper_text`/`si_text` roles; `teteneva2024.py:425-447` refused the identical fact
   pattern one day later. The Zotero rule forbids WRITING Zotero, not retrieving a
   scriptable OA artifact into the loader-owned raw mirror with a `RetrievalRecord`, and
   rows 55 and 61 already followed Mohiuddin. The Teteneva refusal survives only as
   `blocked_by_wall` (nothing scriptable: `#699` `#788` `#853` `#691`), recorded with the
   `manual_browser` recipe, never a retry loop. The verdict carries `zotero_item:
   present|absent`; absent files an owner by-hand issue without blocking the gate. (D5)
7. **The verdict store is authoritative; the tables RENDER it.** The schedule table is a
   triage snapshot not edited when a row settles (cai2023 note 2026.10.08), and the Li 2014
   agent correctly declined a row edit that would re-rank reserve rows other branches own.
   So both table scripts read the store and add a verdict column; no row literal is
   hand-edited for a verdict and `sort_key` never consumes it. The dendron ledger is a
   generated markdown table appended as a dated section. Row-literal corrections (Li 2014's
   media count, strain) remain separate human edits landed when the reserve is idle. (D6)
8. **Enforcement is set-based pytest plus a diff-scoped pre-commit tripwire.** The pytest
   iterates `dataset_registry` (the registration mechanism; the adapter map is downstream
   and excludes record-only modules) and links class to verdict through `CITATION_KEY` (62
   of 88 registered classes already declare it). Existing classes go into an explicit,
   sorted, frozen GRANDFATHERED tuple that can only shrink. A fifth positional pin is
   rejected: the four pin places (`test_bacterial_adapters.py`,
   `_bacterial_adapter_cases.py`, `test_build_time_projection.py`, `test_runners.py`)
   already move on every dataset. (D7)
9. **The re-audit consumes; it never re-fetches, and it marks `unmeasured` honestly.** G1
   and G2 are decidable offline from row fields and the registry, so they run fresh on every
   row. G3/G4/G5 run only where bytes exist (mirror `manifest.json`, raw-mirror deposits,
   the 036 inventory JSONs, settled-row modules, open issues); elsewhere the gate records
   `unmeasured`, never `pass`: a queue's score is not evidence it understood the row. It
   lands AFTER the pre-build verification sweep (branch `docs/pre-build-verification-sweep`)
   and after main is re-read; main moved 17 PRs on 2026-10-09 (Schmidt S23, Wang 2015,
   Lamoureux, Lim 2025, Butland landed), and every verdict records the commit it read. (D8)
10. **G4 is two sub-checks, both recorded, neither builds.** G4-key runs off the row (DOI,
    accession, PMID, sample ids against the served-store manifest and
    `check_candidate_overlap.py`'s offline cache) and can fail a row before fan-out. G4-value
    runs after G3: released table versus the dev LMDB on matched samples, reporting max abs
    diff and nearest-wrong-sample margin (the Thompson measurement: 0.046 vs >= 0.58), and
    reuses the aggregation convention's clause 3 (measured net-new). Byte-hash identity is
    one G4-value result, not the test, because Thompson's 156 values hide inside Borchert's
    bytes. (D9)
11. **Deterministic stages are everything that reads bytes we hold or the registry:** G1
    from row fields, G2 resolution, G3 inventory, G4-key, G4-value, G5 class-exists,
    verdict validation, ledger rendering, pin re-derivation, enqueue. Agentic stages are the
    ones that need reading: G1's aggregation judgment on a borderline row (verbatim quote as
    evidence), G3's mapping of SI file to described table, G5's proposed phenotype class and
    gap text, the loader itself. Every agentic output is a typed record the CLI validates
    before the next stage runs. (D10)
12. **The quote-slice audit is `candidate-gate audit`, run by hand or under slurm, never a
    crontab line.** `#758` needs a data-gated check (file + sha256 + byte slice, whitespace
    and soft-hyphen normalized) that CI cannot run, and a crontab beside `lit-sync` would be
    a project-owned machine-level job. It is an `ops.sh candidates` action
    (`scripts/ops.sh:308`) wrapping the subcommand, plus a slurm script when the mirror walk
    is long. (uncovered 5)
13. **Yeast table gains the vocabulary, not the schema.** `Status` (`build_candidate_datasets_table.py:108`)
    gets `aggregation`; `Excluded.rule` (`:241`) gets `no-per-record-data`. Additive Literal
    widening in an experiments script, no schema surface touched, so G1 reads the same
    vocabulary from both tables. (uncovered 4)
14. **The figure follows the design, through a scheduled follow-up.** Panel c of
    `FigS-dataset-admission-loop` (`backmatter.tex:1044-1083`, source
    `notes/assets/drawio/FigS-dataset-admission-loop.drawio`) is captioned as the
    pydantic-ai loop. The loop note gets a dated section now; the figure and caption change
    go through `/drawio-diagram` or `/fig-swap` after the author's go-ahead, because the
    backmatter carries status chips. (uncovered 3)
15. **Aggregations are marked and counted, not refused (owner, 2026-10-10).** Served data
    already includes aggregations: SynthLethDB (`torchcell/datasets/scerevisiae/synth_leth_db.py`,
    literature-curated pairs with a PubMed reference per pair) on the yeast side, and the
    bacterial compendia Borchert 2024, Lim 2022 PRECISE and Price 2018, each naming a source
    study per record. Refusing the class would contradict the store. So G1 classifies the
    source kind and an aggregation passes when it carries an `AggregationRecord`: how many
    source studies it aggregates, the per-record field that names them, whether its values
    are re-measured or copied (copied values carry `derived_by_aggregator`), how many
    sources are mirrored, and the measured net-new against what is served. What G1 still
    refuses is a corpus whose values cannot be traced to a byte of the measuring paper:
    LLM or review-table transcriptions (D2Cell, MCF2Chem) and predictions. The backfill
    writes an `AggregationRecord` for every served aggregation, starting with SynthLethDB
    (its source-study count is the number of distinct PubMed ids in the stored records,
    measured, not recalled), so the GRANDFATHERED tuple shrinks by those classes first.
    Rejected: the earlier draft's "fails unless the aggregation convention holds", which
    would have refused CeCaFDB's 33 attributed source workbooks on the same clause that
    refuses a transcription.

## Approach

**The verdict.** `CandidateVerdict` is frozen, `extra="forbid"`, pinned to pydantic 2.12.3
semantics (the env; 2.13.0 turns `PydanticUserError` into `RuntimeError` and stops
discriminated-union serialization falling back, per the pydantic.dev changelog read
2026-10-10, so no behavior here may depend on either). It composes the settled-row shapes
rather than inventing a third vocabulary: evidence is a `SourcedValue`-style
file + sha256 + byte slice; a G3 blocker reuses `cai2023.SchemaBlocker` (`:319`) and
`typas2008.DepositReconciliation`; a G2 record reuses `teteneva2024.DepositedAssemblySet`
(`:172`). The field list that disambiguates the ordering and outcome rules:

```python
class GateResult(BaseModel):            # frozen, extra="forbid"
    gate: Literal["G1", "G2", "G3", "G4", "G5"]
    outcome: Literal["pass", "fail", "blocked", "gap", "unmeasured"]
    reason: str; evidence: tuple[SourcedValue, ...]   # file + sha256 + byte slice
    issue: int | None = None             # required when outcome == "gap"
class CandidateVerdict(BaseModel):
    citation_key: str; table: Literal["bacteria", "yeast"]; row_name: str
    gates: tuple[GateResult, ...]        # exactly G1..G5 in order; first fail stops
    zotero_item: Literal["present", "absent"]; g2_branch: ...; g2_outcome: ...
    source_kind: Literal["primary", "aggregation", "transcription", "prediction"]
    aggregation: AggregationRecord | None   # required when source_kind == "aggregation":
        # n_source_studies, attribution_field, value_origin (re_measured | derived_by_aggregator),
        # n_sources_mirrored, net_new_vs_served
    g4_key: ...; g4_value: ...; g5_class: str | None; decided_at: str; torchcell_commit: str
```

A validator enforces order and the stop rule: after a `fail`, later gates are
`unmeasured`; a `blocked` (wall, provision) also stops; a `gap` continues. The verdict
outcome derives from the gates: `admissible` (all pass), `admissible_with_gaps` (every
non-pass is a `gap` with an issue number), `blocked:<kind>`, `refused`.

**The five gates.** G1 (primary measurement) asks one question of every stored value:
does it sit in a deposited artifact and name the study that measured it. The answer
classifies the source as `primary`, `aggregation`, `transcription` or `prediction`
(decision 15). `primary` passes. `aggregation` passes WITH a filled `AggregationRecord`:
the count of distinct source studies, the per-record attribution field that names them,
whether the aggregator re-measured raw data or copied published numbers (copied numbers
are stored as `derived_by_aggregator`, never as the source's measurement), how many
sources are mirrored, and the measured net-new against what is served (Lim 2022 `:2561`
and Borchert 2024 `:2501` re-measure; CeCaFDB copies and rescales 300 maps from 33 named
workbooks, so it passes G1 as a marked aggregation and its values carry the derived flag).
`transcription` (D2Cell: LLM-extracted rows with no source sentence; MCF2Chem: review
tables) and `prediction` fail, because no stored value can be checked against a byte of
the measuring paper. Deterministic off `status`, `klass`, `accession_confirmed` and the
deposit's attribution column; borderline rows go to an agent with the verbatim quote as
evidence. G2 is decision 5, entirely deterministic. G3 (deposit inventory off the bytes) walks the mirror
`manifest.json` `si/` and `data/` entries and the raw mirror separately (`#495`), runs
`zipfile.testzip()`, records full-bytes sha256 and,
when concatenated, the leading-archive sha256 (`tong2020.py:220`), counts rows with
openpyxl 3.1.5 in `read_only` mode, and compares to the abstract's claim (Glebes,
Casanova-Hampton, Mei: hundreds deposited vs tens of thousands claimed). Its mapping of
file to described table is agentic. Nothing scriptable yields `blocked_by_wall` with the
`manual_browser` recipe. G4 is decision 10. G5 checks that a phenotype class exists in
`schema.py` for the released columns (Balakrishnan: `RNASeqExpressionPhenotype` `:5499`
needs integer counts; Li 2014: `ProteinTurnoverPhenotype` `:6944` refuses an empty
degradation rate) and checks compound membership offline in
`torchcell/datamodels/compound_identity_table.json` (`#726`); a missing class is a `gap`
with a filed issue and a proposed class name from the agent, never a schema edit in the
same PR. "Check for an existing unused schema class before declaring a gap."

**The CLI.** `candidate-gate` subcommands: `evaluate --table bacteria|yeast --row <name>`
(G1, G2, G4-key offline); `inventory --citation-key` (G3 off mirror bytes); `overlap
--citation-key` (G4-value against a dev LMDB, read-only); `schema-fit --citation-key`
(G5 deterministic half); `schema` (prints `model_json_schema()` for agent prompts);
`validate <json>` (agent output into the store or a typed refusal); `ledger` (append a
`## YYYY.MM.DD` section to `notes/torchcell.candidates.ledger.md`: citation key, row, G1-G5
glyphs, outcome, issue, decided_at, commit; the `markdown_table` pattern from
`si_audit_loadable_ledger.py:1021`); `audit` (decision 12); `status` (store summary).

**The `/add-dataset` skill.** Stages, in order, one worktree per candidate: (1) `gate`,
deterministic, refuses or blocks before any agent is spawned; (2) `retrieve`, agent maps SI
files to described tables and deposits scriptable artifacts (Mohiuddin route), CLI validates
G3; (3) `propose`, agent drafts the G5 class fit and the loader skeleton, CLI validates G5;
(4) `implement`, agent writes the loader with `CITATION_KEY`, `deposit_raw_mirror`, paired
test, dev build through the slurm array when permitted; (5) `verify`, deterministic:
pytest, mypy, ruff, L0-L4 via `resolve_verifier`, `kg_manifest admit`; (6) `land`,
orchestrator only: rebase, re-derive the four pin places, `gh pr create`, `/enqueue-merge`.
Fan-out rules, verbatim from the program state: no Agent calls inside stage prompts; under
ten concurrent; unique scratch filenames per candidate; the orchestrator does all git;
every agent stage ends with a WIP commit, never a stash, and no `wt_cleanup` runs while
candidate worktrees exist (`#805`); landing goes only through the single-writer queue. The
critic/judge roles are the CLI validators, not extra agents.

**The re-audit.** `experiments/database/scripts/reaudit_candidate_gates.py` imports
`ranked()` from both table scripts, runs G1/G2 fresh on every row, and reads G3/G4/G5 from
what exists: `experiments/036-dataset-fixes-before-kg-build/results/*.json`, the
pre-build sweep tables, `si_audit_loadable_ledger` output, mirror and raw-mirror manifests,
settled-row modules, and the issue list. It writes one verdict per row into
`database/candidates/` and a dated ledger section. Expected named failures from what is
already measured: D2Cell,
MCF2Chem (G1 `transcription`); CeCaFDB passes G1 as a marked `aggregation` (33 attributed
workbooks, values `derived_by_aggregator`) and is then subsumed by its sources at G4
(`#864`); Thompson 2019 lysine `:2936` (G4, inside Borchert 2024); Li 2021
`:5291` (`#800`), Balakrishnan `:4181` (`#853` `#854`), Li 2014 `:3777` (`#857`) (G3/G5);
Glebes `:5579`, Casanova-Hampton `:5596`, Mei `:5612` (G3 count vs abstract). Expect several
of the 20 `engineered-chassis` rows (BL21, W, Nissle, industrial yeast) to move to
`blocked:provision`. Rows 51-54, 56, 58, 59 are in flight: verdicts are recorded against
the commit read; no row is re-ranked or edited.

**Enforcement.** `tests/torchcell/datasets/test_candidate_verdicts.py`: every registered
class not in `GRANDFATHERED` has `CITATION_KEY` and a store verdict whose outcome is
`admissible` or `admissible_with_gaps`; a second test asserts the tuple is sorted and every
member still registered (it shrinks as backfill verdicts land, never grows silently).
`scripts/check_candidate_verdicts.py` mirrors `check_paired_tests.py:75`: index-vs-merge-base
diff, AST scan for a new `@register_dataset` under `torchcell/datasets/`, exit 1 unless
`database/candidates/<CITATION_KEY>.json` is in the same index; a local hook with
`files: ^torchcell/datasets/.*\.py$`.

**Execution order.** (1) `torchcell/candidates/` package, CLI, tests, store directory, the
yeast vocabulary widening, GRANDFATHERED seeded from the registry at that commit; one PR.
(2) Enforcement test + hook + loop-note section; one PR. (3) `/add-dataset` skill; one PR.
(4) Table-script verdict columns; one PR, landed when the reserve is idle. (5) The
re-audit, after the pre-build verification sweep lands and main is re-read. (6) Figure
follow-up after the author's go-ahead.

**Out of scope, deliberately not automated (E).** By-hand retrieval walls (`#699` `#788`
`#853` `#691`); any Zotero write; the compound curator from GilaHyper (`#726`); schema
changes for one candidate (`#854`: human PR with `TORCHCELL_SCHEMA_ACK=1`); retiring or
rebuilding dev stores while `live_rebuild_kg` is queued or running; verdicts on internally
inconsistent releases (`#800`): the script flags, a human refuses; promoting any section
status chip in the paper; wiring `check_store` into `admit` (`#833`, separate decision).

## Gotchas

1. **A sha256 pin is not a verbatim quote (`#758`).** Hazard: evidence drifts from the
   mirror file after an OCR re-run. Sidestep: byte slice in every `SourcedValue`, normalized
   for whitespace and soft hyphens; `candidate-gate audit` re-reads the slice on GilaHyper.
2. **Retrieval walls are typed and recurring.** Hazard: an agent retries a PMC
   proof-of-work bin or a science.org 403. Sidestep: `blocked_by_wall` keyed to
   `RetrievalMethod.manual_browser` (`manifest.py:109`) with the recipe copied from the
   provenance record; "in the schedule but not in the collection" (`#788`) is its own G3
   failure mode, not a mirror hit.
3. **Pre-commit hooks a candidate PR trips.** schema-impact and supported-queries fire on
   `schema.py` and `database/releases/*.json`; markdownlint on `notes/`; test-quality rejects
   truthiness-only tests; paired-tests wants a test per new module; `legacy_partition.py
   --check` (`:54`) wants every module reachable from a root. Sidestep: no `schema.py` edit
   in a gate PR; `database/candidates/` is outside the hook pattern; the CLI entry point and
   the test are the roots; tests assert exact outcomes.
4. **CI pins Python 3.13.0 (`test.yaml:40`).** Hazard: recomputing fingerprints in pytest
   diverges by patch version. Sidestep: the verdict stores `torchcell_commit`, never a
   recomputed fingerprint.
5. **Worktree env.** `scripts/run-in-env.sh` prepends PATH only. Sidestep: every CLI
   stage exports `PYTHONPATH=$WORKTREE` and reads `DATA_ROOT` from the worktree `.env`.
6. **Shared dev tree during a live rebuild.** Hazard: a `torchcell_dirty: true`
   `build_manifest.json` at a non-main commit killed job 3547, and a dev build under
   `live_rebuild_kg` killed its successor. Sidestep: G4 reads `processed/lmdb` and
   `build_manifest.json` read-only and refuses a dirty store with a typed reason; any build
   is the slurm array and refuses while `squeue` shows `live_rebuild_kg`;
   `--retire-existing`, never `deprecate.sh` (graveyard name collision on `ts__basename`).
7. **Fan-out failure modes already paid for.** Seven agents at once and nested fan-out
   hit rate limits; a shared scratchpad clobbered `pr_body.md`; agents in one worktree
   clobbered via git; `wt_cleanup` removed an unpushed commit (`#805`). Sidestep: the
   skill's fan-out rules are in every stage prompt verbatim.
8. **The drainer blocks on a rebase conflict and never consults checks**
   (`drain_merge_queue.py:224-225`). Sidestep: `land` re-derives the four pin places after
   rebase, confirms the seven CI checks on the head sha, then `add`; blocked = re-queue.
9. **Zotero duplicate collection names.** Sidestep: G3 resolves collections by key;
   missing PDF = by-hand issue in the `#691` template.
10. **One `Manifest` class for both mirrors (`#495`).** Hazard: `lit_sync` treats a
    raw-mirror key as a broken capture. Sidestep: G3 keys the two trees separately; the
    hazard is named, not fixed here.

## Verification

- Unit tests, paired per module:
  `~/miniconda3/envs/torchcell/bin/python -m pytest tests/torchcell/candidates -xvs`
  covering: gate ordering and stop rule; outcome derivation; G1 on the three aggregation
  rows (fail) and Lim 2022 / Borchert 2024 (pass); G2 four outcomes on fixtures for
  MG1655, W3110, a named-but-absent accession, an unnamed strain, and the Bloom 2019
  mosaic; G3 on a concatenated-archive fixture; G4-value on a Thompson-shaped fixture
  (max diff and nearest-wrong margin); G5 on an integer-count column vs a number-fraction
  column; store round trip; ledger rendering byte-exact.
- Enforcement:
  `~/miniconda3/envs/torchcell/bin/python -m pytest tests/torchcell/datasets/test_candidate_verdicts.py tests/torchcell/datasets/test_dataset_registry.py -xvs`
  (GRANDFATHERED sorted, every member registered, every non-member has `CITATION_KEY` and
  an admissible verdict; the registry test's `len >= 52` and `__all__` subset still hold).
- Tripwire: `~/miniconda3/envs/torchcell/bin/python -m pytest tests/torchcell/scripts/test_check_candidate_verdicts.py -xvs`;
  manual smoke: register a throwaway class in a scratch worktree, `git add`, `pre-commit
  run candidate-verdicts --all-files` exits 1 with the citation key named; add the verdict
  JSON, exits 0.
- Static: `bash scripts/run-mypy.sh torchcell/candidates`, `ruff check torchcell/candidates
  tests/torchcell/candidates experiments/database/scripts/reaudit_candidate_gates.py`,
  `pre-commit run --all-files` (schema-impact and supported-queries must NOT fire),
  `~/miniconda3/envs/torchcell/bin/python scripts/legacy_partition.py --check`,
  `~/miniconda3/envs/torchcell/bin/python scripts/check_paired_tests.py --base main`.
- CLI smoke on GilaHyper with the mirror mounted: `candidate-gate evaluate --table bacteria
  --row "D2Cell 2026"` prints G1 fail; `--row "Lim 2022 putidaPRECISE321"` passes G1;
  `candidate-gate inventory --citation-key mohiuddin2022` reports the raw-mirror files;
  `candidate-gate schema | python -m json.tool` validates; `candidate-gate validate` on a
  hand-broken JSON (missing `issue` on a `gap`) refuses with the field name.
- Re-audit: `~/miniconda3/envs/torchcell/bin/python experiments/database/scripts/reaudit_candidate_gates.py`
  from the repo root; the ledger section must name every row the re-audit paragraph
  expects, at the expected gate; every row without bytes shows `unmeasured` on G3-G5, none
  shows `pass` there; no row literal in either table script changed
  (`git diff --stat` on the two scripts is empty after the run).
- Table rendering: both table scripts run from the repo root and emit the verdict column;
  `sort_key` output is byte-identical before and after (diff the ranked list).
- Skill dry run: `/add-dataset` on a row the gate refuses (D2Cell) spawns zero agents and
  writes one verdict; on `Lim 2022` it stops at `implement` when `squeue` shows
  `live_rebuild_kg`, with the typed reason in the log.
