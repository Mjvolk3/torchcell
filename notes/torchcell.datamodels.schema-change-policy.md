---
id: fjwggx10nln2aj0a6fy5t80
title: Schema Change Policy
desc: ''
updated: 1791803403506
created: 1791803403506
---

## 2026.10.12 - The schema-change policy and its enforcement

The schema is the fingerprinted surface `torchcell/datamodels/schema.py` + `pydant.py`: every class and every module-level binding a class resolves through, 284 symbols at main `eb07bd213` (229 classes, 55 bindings). Its contract (fields, required-ness, semantic `Field` arguments, validators and serializers, enum members, bases, `Config`) is what every dev store and the served graph were serialized under, so it is the most sensitive file in the database. The owner, 2026-10-12:

> we need much tighter rules i think for what justifies updating the schema ... the schema is probably the most sensitive / important file in this entire db ... we trust old schema by default because we know compute went into building it ... we are fine with irreversible updates to schema, as long as you are convinced it is correct, that is the criterion

The rules below each carry their reason and a case from the repository's history. The enforcement (a record per change, a pre-commit tripwire, an enforcement test) is the second half of this section.

### Rule 1 - Trust the served schema by default

A served class, field or enum member is presumed correct. Compute went into it (dev LMDB builds, the L0-L4 verification of every store that imports it, a full KG build), and every served dataset whose closure holds it was checked against it. The burden of proof is on the change, never on the existing contract.

- **Reason.** A served symbol has evidence behind it; a proposed change has an argument. Moving a served contract invalidates the stores built under it, and the incremental import cannot repair served nodes, so the cost of a wrong change is a full rebuild plus whatever the change silently broke.
- **Case.** Hu 2007 (#1022) releases error-model `X` scores, not log ratios. Relaxing `MicroarrayExpressionPhenotype` to hold them was rejected in the class's own docstring: "relaxing either field would move the schema closure of the three served microarray stores (Kemmeren 2014, both Sameith 2015 sets) for one release". The release got `MicroarrayExpressionScorePhenotype`, a sibling, and no served record changed. Likewise #793 added a replicate quartet to `GeneInteractionPhenotype` but no derived SE field: "inventing a second statistic name on a served class is a separate decision" ([[torchcell.datamodels.schema]], 2026.10.09).

### Rule 2 - Correctness is the criterion, not reversibility

An irreversible change is acceptable when it is demonstrably more correct. "Demonstrably" means a measurement or a source: a round-trip count over every stored record, a verbatim quote, a field the release carries. "Convenient for a new source" is never a reason, and neither is "keeps the diff small".

- **Reason.** Reversibility protects nothing once a build has run on the change; correctness is what the stores are for.
- **Case, accepted.** `ArtifactRef` (2026.10.07) removed the `sequence_uri` / `sequence_sha256`, `effector_plasmid_uri` / `effector_plasmid_sha256` and `assembly_member` / `assembly_sha256` string pairs, a breaking change for 5 datasets, because a pointer without its hash can no longer be written. #1016 removed `PseudobulkUmiCountPhenotype` into `SingleCellExpressionPhenotype` after "every stored phenotype of both dev stores (79,516 + 4 Boocock, 784 + 11 Jackson) was measured to round-trip through the merged class with no field dropped or changed" (PR #1038, commit `133ef4a04`).
- **Case, refused.** #985's first draft made `Environment.media` optional so the SGD essentiality and SynLethDB compendia could load. That is convenience for a thin source, and the owner reversed it (Rule 5).

### Rule 3 - Additive first

Prefer, in order: a new leaf, a new sibling, a new experiment pair sharing an existing leaf, a new optional-by-construction field (absent means "the release does not carry it", never "unknown"), a typed `ProvenanceGap` on an existing required field (value or typed gap, never a silent `None`), a new enum member. Each needs a correctness argument in its record and nothing more.

A **removal** (class, field, enum member, binding), a **rename**, **loosening a required field** (required to optional, or a validator that accepts more without a typed gap) or **changing a field's meaning** (unit, statistic, reference, what a key denotes) needs a written correctness argument AND the owner's sign-off: the owner's words verbatim, or the URL of the issue or PR comment holding them.

- **Reason.** An additive change cannot alter what a served record means; the four listed changes can, silently, for every consumer.
- **Case, additive.** #776 and #793 added two optional blocks to served phenotypes: 34 impacted datasets, 0 breaking. Commit `6c763b673` let a source without stage labels gap `cell_cycle_stage_counts` (value or `ProvenanceGap`) instead of dropping the requirement.
- **Case, sign-off.** The #1016 merge removed a leaf, so it went to the owner as an issue first. PR #1038 states the decision it implements, "one class for the two single-cell UMI-count leaves, unless it loses data", and the round-trip measurement that met its condition.

### Rule 4 - Never remove a dataset class to satisfy a required field

When a served dataset cannot fill a required field, the route is a typed provenance gap on the field's carrier, quoting the source's silence, which keeps the field required and the dataset served.

- **Reason.** A removal throws away verified records to protect one field, when a typed absence keeps both and makes the gap queryable.
- **Case.** #985's second draft turned `GeneEssentialitySgdDataset`, `SynthLethalityYeastSynthLethDbDataset` and `SynthRescueYeastSynthLethDbDataset` into provenance records, 22,277 served records leaving the graph, because no record names a medium. The owner, 2026-10-12: "we are fine with irreversible updates to schema, as long as you are convinced it is correct, that is the criterion for now ... why would you get rid of old classes?" The classes stay; `Media` carries the gap.

### Rule 5 - Never weaken a required field for a thin source

A required field stays required. When a source cannot supply it, the question is whether the SOURCE belongs, not whether the field can bend: the record is refused with a typed reason and a count, or the field's carrier takes a typed gap (Rule 4).

- **Reason.** The schema is meant to be more rigorous than its sources; bending it to the weakest one inverts that for every source.
- **Case.** `Environment.media` stays required. The owner, 2026-10-11, verbatim in PR #985: "i don't think env / media should be optional... if there are some cell growth under some condition i think we need it ... if we are blocked by the compendia... then we shouldn't use that compendia." The same reasoning refused a `SourceWithdrawal` type for author-withdrawn data (2026.10.11 in [[torchcell.datamodels.schema]]): withdrawn data is refused, not stored behind a flag.

### Rule 6 - Place a new measurement type by the physical measurement, using the DAG

The owner, 2026-10-12: "if they are fundamentally different technology, two [leaves] and we convert; or they can be the same at a given level but diverge in specifics ... use the DAG-ness."

**What decides "fundamentally different".** The physical measurement, never the analysis: what is captured (a hybridized labeled cDNA, a sequenced fragment, a UMI-deduplicated molecule, a fluorescence, a colony area), by what readout, and what number the release carries before any modeling. Pooling cells, mapping eQTLs, contrasting against a deletion, normalizing, and kit versions are analysis or specifics; they never justify a new leaf on their own. Decide in this order:

1. **Same physical measurement and the same released quantity as an existing leaf:** reuse the leaf and one `measurement_type` tag. Divergence in design goes into the experiment class above the leaf (a new `*Experiment` / `*ExperimentReference` pair that shares the phenotype leaf: the DAG, one leaf with several parents). Source-specific metadata goes into value-or-gap fields on the leaf.
2. **Same at the parent's level, divergent in specifics that change which fields exist or what they mean** (another UMI platform with no droplet barcodes, say): a sibling leaf under the same parent, so a query over the shared parent stays meaningful.
3. **A different physical measurement** (array hybridization against sequencing; full-length TPM against 3' UMI counts): a separate leaf, and any comparison between the two goes through an explicit converter (the `torchcell/datamodels/conversion.py` `Converter` pattern, as `gene_essentiality_to_fitness_conversion.py` does), never through a field reused with a different meaning.

A released quantity that differs in meaning from an existing field (a significance score where the leaf stores a ratio) is a sibling leaf even on the same technology, because reusing the field is a meaning change (Rule 3): `MicroarrayExpressionScorePhenotype` beside `MicroarrayExpressionPhenotype` is the case.

**Worked example: single-cell UMI counts (#1016).** Both releases are 10x Chromium 3' UMI counts:

- Boocock 2025, `torchcell-library/boocockSinglecellEQTLMapping2025/paper.md` (sha256 `a2920750e12fb9977f36a687bb9f73aa0382527de84270b44609a3338bbeafe4`), section "Single-cell library preparation":

  > loaded onto the Chromium device using either the Single Cell $3 ^ { \prime }$ Solution V2 or V3 kit.

  and the Figure 1 legend: "Unique molecular identifier (UMI) counts are tallied for each transcript in each segregant."
- Jackson 2020, `torchcell-library/jacksonGeneRegulatoryNetwork2020/paper.md` (sha256 `453924f3d5ed998e99695768cf5b4067f2dbac3fb0af53dcbf8fd9017f199697`), section "Library preparation and sequencing":

  > Single cell library preparation was done using the $ 1 0 \times$ Genomics Chromium ${ \boldsymbol { 3 ^ { \prime } } }$ v2 Single Cell Gene Expression Kit

  and section "Supplementary files": "gene expression values are unmodified transcript counts after deartifacting using UMIs".

Same physical measurement, same released quantity, so case 1: one phenotype leaf, `SingleCellExpressionPhenotype`, and one `measurement_type` tag. The two diverge in the experiment classes above the leaf, which is where the designs differ: `SegregantExpressionExperiment` (a haplotype-mosaic `SegregantGenotype`, eQTL mapping) and `PseudobulkUmiCountExperiment` (a gene-keyed TF-deletion `Genotype` pooled per strain and condition, a `CultureEnvironment`). V2 against V3 chemistry is a specific, not a technology. What only one release can state (Boocock's cell-cycle stage per cell, Jackson's cassette-marker transcripts) is a value-or-gap field on the shared leaf. A non-10x single-cell platform would be case 2: a sibling leaf under the same parent (`Phenotype` today). `RNASeqExpressionPhenotype` is case 3 against both: it stores TPM, "which a 3' UMI assay does not produce" (the schema comment above the single-cell leaf).

**Open finding, not changed here.** The loaders do not yet follow the one-tag half of case 1. By the loader source, Boocock 2025 and N'Guessan 2025 store the leaf default `measurement_type="scrnaseq_umi_count"` and Jackson 2020 writes `MEASUREMENT_TYPE = "scrnaseq_pseudobulk_umi_count"` (`torchcell/datasets/scerevisiae/jackson2020.py`). Pooling is analysis and `n_cells` already records it, so the rule asks for one tag; aligning it rebuilds Jackson's dev store and is left to its own change with a record.

### Rule 7 - What every schema change ships with

- A schema-change record, `database/schema_changes/<date>-<slug>.json` (below), naming every changed symbol that is not exempt.
- The impact report, `PYTHONPATH=<worktree> python scripts/schema_impact_check.py --base origin/main`, pasted in the PR body.
- The served datasets that rebuild (the record's `served_datasets`), in the PR body.
- The `schema-breaking` label on the PR when a served closure changes (the record's `rebuild_scope` is `full_kg_rebuild`); such a change lands with a full KG build, never an incremental admission.

### Rule 8 - What needs no record

- Docstrings, comments, `Field` descriptions and field order. The contract fingerprint excludes them, so the tripwire never sees them.
- A NEW symbol that no served dataset's closure holds yet. The tripwire prints it; list it in the PR body. Once a served loader imports it, or a served class references it, it needs a record.
- Test-only changes.

### The record

`torchcell.provenance.schema_change.SchemaChangeRecord`, one JSON file per change, every field required:

| field | content |
|---|---|
| `record_version` | `1` |
| `slug`, `date` | the file is `<date>-<slug>.json`; kebab-case slug, unique across records |
| `pr` | the PR that lands it; `null` only until the PR is open, and the enforcement test fails a record without one |
| `author_kind` | `owner` or `agent` |
| `base_commit`, `python_version` | the merge base the before fingerprints were read at, and the interpreter (fingerprints hash `ast.unparse` text, which differs across 3.13 patch releases; CI pins 3.13.0) |
| `symbols` | per symbol: `status` (added, removed, modified), `kinds`, `fingerprint_before`, `fingerprint_after`, `impact` (breaking or stale) and `reasons`, the last two verbatim from `schema_impact` |
| `correctness_argument` | free text, non-blank |
| `measurement_type_decision`, `source_quotes` | a measurement-type decision carries at least one verbatim `SourcedValue` (citation key, mirror path, sha256, quote) |
| `alternatives_rejected` | each with why; at least one for a sign-off kind or a measurement-type decision |
| `served_release`, `affected_loaders`, `served_datasets`, `rebuild_scope` | from the impact tool and the newest committed release snapshot; `rebuild_scope` is `none`, `dev_stores` or `full_kg_rebuild` and must agree with the loaders |
| `owner_sign_off` | `signed` plus the owner's quote or a `https://github.com/` comment URL; required when any symbol carries a sign-off kind |

**Kinds.** A symbol may carry several. Each claim is checked against `schema_impact`'s reasons in both directions: every reason must be covered by a claimed kind (a removed field cannot be called `additive_field`), and a kind that names a structural event must have a reason that shows it (`enum_member_added` needs a gained member).

| kind | meaning | owner sign-off |
|---|---|---|
| `additive_leaf` | a new class or binding: leaf, sibling, experiment pair, enum | no |
| `additive_field` | a new optional-by-construction field | no |
| `typed_gap` | a required field becomes value-or-gap (`X -> X \| None`, still required, plus the gap check) | no |
| `enum_member_added` | a vocabulary gains a member | no |
| `equivalent_refactor` | the fingerprint moved but the accepted records and their serialization did not (a validator message, a reordered check) | no |
| `contract_tightened` | a new required field or a stricter check; breaking, and correct only with the argument | no |
| `field_meaning` | same name, different unit, statistic or semantics | yes |
| `field_required_loosened` | accepts more, without a typed gap | yes |
| `rename` | a class, field, member or binding under a new name | yes |
| `removal` | a class, field, member or binding is gone | yes |
| `enum_member_removed` | a vocabulary loses a member | yes |

Three kinds go beyond the eight first proposed: `additive_field`, `equivalent_refactor` and `contract_tightened`. The fingerprint moves for each, and none of the eight describes them honestly.

### How the tripwire decides

`scripts/check_schema_change_records.py`, the `schema-change-records` pre-commit hook (fires on the schema surface, `torchcell/datasets/`, `database/schema_changes/` and `database/releases/`):

1. Diff the STAGED surface (`git show :<path>`) against the merge base with `origin/main` (`schema_impact.diff_surfaces`).
2. Read every record staged under `database/schema_changes/` since that merge base; an unstaged record does not count, and an invalid one is reported with its validation error (an unsigned removal is invalid by construction).
3. For each changed symbol: no record and the symbol is new with no served dataset's closure holding it (served = the datasets of the newest `database/releases/<release>.json`, mapped through `schema_impact.map_impacts`) is a note; no record otherwise is a problem; two records naming it is a problem; one record whose status, before and after fingerprints, reasons or impact differ from the diff is stale.
4. A staged record naming a symbol that did not change, or whose affected loaders or served release differ from the staged tree, is stale.

Exit 1 on any problem. `TORCHCELL_SCHEMA_RECORD_ACK=1` passes the commit. It is deliberately not `TORCHCELL_SCHEMA_ACK`: that variable is set on every breaking schema commit to acknowledge the rebuild, and sharing it would switch this check off exactly when it matters.

### The enforcement test and the grandfathered baseline

`tests/torchcell/datamodels/test_schema_change_records.py` runs in CI. `database/schema_changes/grandfathered.json` pins all 284 surface fingerprints at `eb07bd213ca1b8f4138cc4327664558109024dd3` (Python 3.13.0), the commit this policy branched from; no record is backfilled. For every symbol, the records naming it must chain from its baseline fingerprint to its current one (one record's `fingerprint_before` equals the previous one's `fingerprint_after`). A moved or removed baseline symbol no record names fails; a record whose `after` no longer matches the schema is stale and fails; so does a record without its PR or computed under another interpreter. `TORCHCELL_SCHEMA_RECORD_ACK=1` skips the drift test for a local emergency run; CI never sets it, so the override lets a commit through and never a landing.

The baseline is written once. If a schema change lands on main before this policy does, the baseline is re-pinned at the new base (`python -m torchcell.provenance.schema_change baseline --ref origin/main --write --force`) together with `BASELINE_COMMIT` in the test, in the same PR, so the re-pin is visible in review.

### Workflow

```bash
# after editing schema.py, in the worktree
python -m torchcell.provenance.schema_change template > statement.json   # reasons on stderr
# fill slug, kinds per symbol, correctness_argument, quotes, alternatives, sign-off
python -m torchcell.provenance.schema_change new --statement statement.json
git add torchcell/datamodels/schema.py database/schema_changes/<date>-<slug>.json
# open the PR, then set "pr" in statement.json and rewrite the record
python -m torchcell.provenance.schema_change new --statement statement.json --refresh
gh pr edit <N> --add-label schema-breaking        # when rebuild_scope is full_kg_rebuild
python -m torchcell.provenance.schema_change check   # the enforcement test, from the CLI
python -m torchcell.provenance.schema_change list
```

After a rebase or a further schema edit, rerun `new --refresh`; the record keeps its date. `schema-change` is the same CLI as a console script once the package is reinstalled (`pip install -e .`).

Related: [[torchcell.provenance.schema-dependency-tracking]] (fingerprints and closures), [[torchcell.knowledge_graphs.incremental-admission]] (the rebuild policy), [[torchcell.provenance.schema_change]], [[scripts.check_schema_change_records]].
