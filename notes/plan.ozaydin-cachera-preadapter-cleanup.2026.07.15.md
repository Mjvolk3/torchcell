---
id: fpnxpn1ee2fsjmdq8kyys6f
title: Ozaydin + Cachera Pre-Adapter Cleanup
desc: ''
updated: 1784099445852
created: 1784099445852
---

## 2026.07.15 - Plan: Pre-Adapter Cleanup + Canonical Rebuild

Cleanup pass on the β-carotene (`CarotenoidOzaydin2013Dataset`) and betaxanthin
(`BetaxanthinCachera2023Dataset`) loaders **before** BioCypher adapters are built for
the graph-DB rebuild. Audit found: loader code is current + rebuilds clean (Ozaydin
4474, Cachera 4735 records), but the on-disk LMDBs are STALE (predate required
`Media.is_synthetic` → fail schema round-trip), loaders skip sha256 verification, a
verifier test is stale, and there are no dataset-build tests. Related:
[[torchcell.datasets.scerevisiae.ozaydin2013]],
[[torchcell.datasets.scerevisiae.cachera2023]], memory `ozaydin-cachera-followups`.

### Constraints

- **Worktree only** (`/setup-worktree`); never touch the primary `main` checkout.
- No fallbacks / unnecessary try-except (CLAUDE.md). Provenance-first; pydantic-first.
- mypy is **strict** (`pyproject.toml`); ruff `select=[E,F,I,UP,D]`, google docstrings.
  UP007/UP045 PyG caveat does NOT apply to these files (loaders/verifiers use modern
  `X | None` freely).
- Rebuilt LMDBs live under `$DATA_ROOT` → NOT in the git commit; the commit is
  code/test/note only. Rebuild is a data-artifact step to run + verify + report.
- torchcell env interpreter: `~/miniconda3/envs/torchcell/bin/python`.

### Item 1 — Fix stale verifier test (test-only)

`tests/torchcell/verification/test_metabolite_verification.py:106` filters
`report.results` for `name == "orf_uniqueness"`, but the metabolite verifier renamed
that L1 check to `"genotype_uniqueness"` (`torchcell/verification/metabolite.py:85`;
keys on the whole deletion set for combinatorial collections). Replace the string on
line 106 only: `"orf_uniqueness"` → `"genotype_uniqueness"`. Do NOT touch the verifier
and do NOT touch `test_visual_score_verification.py:126` (that verifier legitimately
emits `orf_uniqueness`). Gate: run metabolite + visual_score + ontology-invariant suites
→ 0 failures (was 189 passed / 1 failed → 190 / 0).

### Item 2 — sha256 verification in both `download()` (provenance)

House idiom (scouted): inline `hashlib.sha256(data).hexdigest()` compared to a
module-level UPPER_SNAKE constant, `raise RuntimeError` on mismatch (see
`costanzo2021.py:111,259-263`; `smith2006.py`; `hoepfner2014.py`). NOT the shared
`sha256_file` helper (no loader uses it). Add `import hashlib` to each stdlib block.

- **ozaydin2013.py**: constant `_SI_SHA256 = "4818726e...7507f0"` (comment: recorded in
  `torchcell-library/ozaydinCarotenoidbasedPhenotypicScreen2013a/manifest.json`, role
  `si_data`). `download()` currently short-circuits `if osp.exists(dest): return` (176-190)
  — restructure so sha256 is verified in BOTH branches: if the file already exists, read
  - hash + compare; else download, size-check, hash + compare, then write. Mismatch →
  `RuntimeError(f"... sha256 mismatch: got {digest}, expected {_SI_SHA256}")`.
- **cachera2023.py**: constant `DATA_SHA256 = "71f55609...cac8a"` (alongside `DATA_URL`/
  `DATA_FILENAME` at 120-121; comment → cachera manifest.json role `si_data`). Same
  both-branch verification in `download()` (159-173).
- Existing on-disk raw files already match these hashes (verified) → verification passes
  on the real files; a smoke run of each `download()` against the populated raw dir must
  not raise.

### Item 3 — Document the Cachera gene drop (decision: document, do NOT backfill)

Resolver drops 5 unresolved names: `WT` (control), `YLR287-A` (malformed), and real
genes `AAD6`/`CRS5`/`FLO8` — whose SGD systematic ids (YFL056C/YOR031W/YER109C) are NOT
in the genome `gene_set` (6607), so including them would orphan gene nodes in the KG.
Backfill is wrong (would guess ids + create dangling refs). Actions:

- In `process()` log (`cachera2023.py:227-235`): log the FULL sorted unresolved list (only
  5) rather than `[:5]`, and add an inline comment that AAD6/CRS5/FLO8 are real genes
  intentionally dropped (absent from reference gene_set → would orphan). Keep behavior
  identical (no backfill map). 20 ORF collisions deduped (first-kept) already logged.
- Append a dated `## 2026.07.15 -` section to
  `notes/torchcell.datasets.scerevisiae.cachera2023.md` recording: the 5 unresolved names,
  WHY the 3 real genes are dropped, that it is intentional (3/4735), the 20 collisions,
  and a follow-up to revisit if the reference annotation is updated.

### Item 4 — Build-smoke tests (new)

New files `tests/torchcell/datasets/scerevisiae/test_ozaydin2013.py` +
`test_cachera2023.py`. No `conftest.py` exists today; only a `gpu` pytest marker is
registered. Register a `slow` marker in `pyproject.toml` `[tool.pytest.ini_options]
markers` and tag both tests `@pytest.mark.slow` (genome load is heavy). Idiom (scouted
from `test_s288c.py` / `test_graph.py`): `load_dotenv(); DATA_ROOT=os.getenv("DATA_ROOT")`;
module-level `pytest.skip(..., allow_module_level=True)` if DATA_ROOT unset; `skipif` on
the specific raw file (and genome dir for cachera) so CI without the data skips cleanly.
Copy mirror raw (`$DATA_ROOT/data/torchcell/<name>/raw/<file>`) into `tmp_path/<name>/raw/`
with `shutil.copy` (the `download()` `if osp.exists(dest): return` guard means no network).
`SCerevisiaeGenome` import = `torchcell.sequence.genome.scerevisiae.s288c`
(`genome_root`, `go_root`, `overwrite`). `dataset[i]` returns typed instances
(`experiment_class(**...)`), so assert:

- `len(dataset) == 4474` (Ozaydin) / `== 4735` (Cachera) — baseline counts from this audit.
- `isinstance(dataset[0]["experiment"], VisualScoreExperiment)` / `MetaboliteExperiment`.
- round-trip: `exp.model_validate(exp.model_dump()).model_dump() == exp.model_dump()`.
- perturbation types: Ozaydin `{kanmx_deletion:1, gene_addition:3}`; Cachera
  `{kanmx_deletion:1, gene_addition:4}` (assert the multiset over `dataset[0]`).
- `exp.environment.media.is_synthetic is True`.
- `dataset[0]["publication"].pubmed_id == "22918085"` / `"37572348"`.

### Item 5 — Rebuild both canonical LMDBs in place

`$DATA_ROOT/data/torchcell/{carotenoid_ozaydin2013,betaxanthin_cachera2023}`: delete
`<root>/processed` + `<root>/preprocess`, KEEP `<root>/raw`; re-run the build (Ozaydin
no genome; Cachera inject `SCerevisiaeGenome` from `$DATA_ROOT`). Post-rebuild verify:
record 0 of each LMDB round-trips through `VisualScoreExperiment` / `MetaboliteExperiment`
and `media.is_synthetic` present. Report counts. NOT part of the commit.

### Item 6 — Notes: promote the adopted cassette design

Both notes still say design PENDING though code implemented recommendation A. Edit the
PENDING headings (ozaydin2013.md:98 text = `DESIGN SIGN-OFF STILL PENDING` incl. "STILL";
cachera2023.md:83 = `DESIGN SIGN-OFF PENDING`) → mark ADOPTED, and state the design:
per-gene `GeneAdditionPerturbation` with `localization`/`source_organism`/`construct_name`
(+ `integration_locus` for Cachera), native feedback-resistant alleles (ARO4^K229L /
ARO7^G141S) carried as `variant=` on `GeneAdditionPerturbation` (NOT `AllelePerturbation`,
since integrated ectopically at XII-5). Preserve dendron frontmatter; append as dated
content where it reads best.

### Verification gates (all must pass before hand-back)

1. ruff + mypy-strict clean on every changed `.py` (loaders, verifier test, 2 new tests).
2. metabolite + visual_score + ontology-invariant suites: 0 failures.
3. both new build-smoke tests pass under the torchcell env.
4. both canonical LMDBs rebuilt + record-0 round-trip verified (report counts 4474/4735).
5. No `--no-verify` / force-push / fallback. Rebase onto `main`; hand back for
   `/enqueue-merge` (run from the parent session).

### Out of scope (flag, don't do)

- The VisualScore/Metabolite BioCypher adapters (the next step after this lands).
- Plasmid-sequence store (`plasmid_contig_id`/`locus_tag` stay None).
- Alias-map fix for AAD6/CRS5/FLO8 (documented as follow-up, not fixed here).
