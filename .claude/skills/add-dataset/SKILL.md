---
name: add-dataset
description: Add or revise a torchcell dataset loader without losing a single released record; the data is the data, the genome is a separate layer, schema changes need a record, and every landing proves its record counts against the release and the served graph
---

# Add or revise a dataset

Use this skill for every new loader and every change to an existing one, on yeast or bacteria. It exists because on 2026-10-11 a loader change refused 317,234 measured double-mutant records because a later genome annotation had renamed 35 genes. The owner's words: "the data is the data. It is what they reported. We don't drop raw data unless they themselves pull it. The genome is separate; it doesn't gate anything; we do that at modeling time." Everything below follows from that.

Read the database constitution first (`/db-constitution`, [[database.constitution]]): the five rules below are its principles 2, 3, 7, 8 and 10 applied to a loader.

## The five rules

1. **The data is the data.** Every record the authors released is stored under its released identity: released gene, ORF, locus tag, strain and compound names, released values, released uncertainty (an SD of exactly 0 is stored as 0). The ONLY drop is an author withdrawal, quoted verbatim with its sha256 (Wetmore 2015 is the precedent). Nothing else is ever refused, filtered, collapsed or deduplicated away. A value the source does not state is a typed provenance gap; a reported value that contradicts an annotation (an essential gene reported deleted) is stored as reported, with the contradiction written in the dataset note and the paper note as a follow-up, not as a record field.
2. **The genome is a separate layer, and no modeling happens before the data enters.** A loader never consults the annotation, a synonym table or another reference to decide whether a measurement exists, how it is keyed, or what status it carries. It stores the identifiers the authors released, verbatim: gene symbols, ORFs, locus tags, b-numbers, strain names, SNP positions with the reference the authors name. The genome tier and the experiment tier are loaded independently and meet downstream, in the adapter and at modeling time, through their gene sets; a released name with no matching gene node simply has no gene edge, and a contradiction found there is dealt with there. Liftovers, name resolution and synonym routes are downstream utilities or derived artifacts, never loader logic and never record fields. L4 genome-reference verifier rules report findings and never fail a store, retire a manifest, or block registration. Candidate gate G2 checks that the STRAIN BACKGROUND resolves to a genomes-tier assembly or a declared probabilistic representation; it is not a per-record gate.
3. **Trust the old schema; correctness decides changes.** A served class, field or enum member is presumed correct. Additive first: a new leaf, a sibling, a typed gap on an existing required field. Never make a required field optional for a thin source (media stays required); never remove a dataset class to satisfy a field (the typed gap is the route). Place a new measurement type by the physical measurement, not the analysis: different technology, two leaves with explicit conversion; same technology diverging in specifics, shared parent, diverge in the leaf. Every schema change ships a `SchemaChangeRecord` (`python -m torchcell.provenance.schema_change new`), the impact report, and the `schema-breaking` label when a served closure moves. Policy: `notes/torchcell.datamodels.schema-change-policy.md`.
4. **Source every schema value from the paper or SI,** with a verbatim quote and sha256 from the mirror; retry before concluding "not in the SI"; follow deferrals to the cited paper; a value the source does not state is a typed gap, never a guess. Raw files live in the raw mirror with a retrieval record; a by-hand file gets `DEPOSIT.md`, `SHA256SUMS.txt` and `manual_browser` provenance (`torchcell/literature/manual_deposit.py`). The primary paper gets a Zotero item in the `database/<Organism>` collection (`scripts/zotero_add_paper.py`).
5. **Prove the counts before landing.** The PR states, per store: released row count, stored record count, and every difference with its typed reason and count (withdrawals, duplicates the release really carries). For a revised loader of a SERVED dataset, run `python scripts/served_count_drift.py` and paste its table: a decrease against the served release is a block unless it is an author withdrawal. CI green and pin counts are not evidence that no record was lost.

## Mechanics

- Worktree per branch, PR for every branch, land by rebase and ff through the merge queue (`/enqueue-merge`); never `gh pr merge`.
- Class declares `organism = Organism.<scerevisiae|ecoli|pputida>`; registry, KG mapping yaml, adapter map, verification family runner.
- Re-derive the four pin places from the merged tree, never add deltas: `tests/torchcell/adapters/test_bacterial_adapters.py`, `tests/torchcell/adapters/_bacterial_adapter_cases.py`, `tests/torchcell/knowledge_graphs/test_build_time_projection.py`, `tests/torchcell/verification/test_runners.py`.
- Regenerate after any schema change: `bash scripts/run-ontology-figure.sh`, `python -m torchcell.database.browser_style`.
- Build the dev store under slurm with `--retire-existing --verify`, never on the login shell; append the L0-L4 table to the dataset's dendron note; gate the candidate (`candidate-gate gate`) with the G3 inventory script committed under `experiments/database/scripts/`.
- A served dataset's loader change is a full-rebuild case: say so in the PR, and the KG build follows, never an incremental admission.

## Before you report done

- `python scripts/served_count_drift.py` shows no unexplained decrease.
- `scripts/check_schema_change_records.py` and `scripts/check_candidate_verdicts.py` pass.
- The note's L0-L4 table shows genome-reference rules as findings, not gates.
- The PR lists what the source carries that you did not store, and why, each with a count.
