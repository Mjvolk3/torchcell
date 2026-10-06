---
id: 1ri3t9rtjy956o6gftoznhj
title: Test_hillenmeyer2008_synthetic
desc: ''
updated: 1790562883031
created: 1790562883031
---

## 2026.09.27 - The Hillenmeyer 2008 loader through a real genomes registry

Twelve tests with the registry manifest and FASTAs under a `tmp_path` `DATA_ROOT`; the raw mirror is written by the loader's own `deposit_raw_mirror` with no network. Loader coverage 90%; the `genome=None` branch (lines 986 to 991) builds a real genome and stays untested. Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.

## 2026.10.06 - Phase 21: verify_source, the default resolver, small branches

- `deposit_raw_mirror(..., verify_source=True)` with `torchcell.literature.retrieve.direct_url` stubbed: one call per file on its Wayback URL in `raw_relpaths` order, each retrieval carries `last_check = {checked_at: 2026-10-06 (date stubbed), produced_sha256: sha256 of the staged text, matches: True}`; each retrieval record is otherwise `{method: direct_url, source_url: <Wayback URL>, retriever: torchcell.literature.retrieve.direct_url, params: {url: <Wayback URL>}, sha256: <digest>, retrieved_at: 2026-07-11}` (`RAW_RETRIEVED_AT`); drifted bytes raise naming both digests and write no manifest and no file.
- `_resolver` with `genome=None` builds `SCerevisiaeGenome(genome_root=<DATA_ROOT>/data/sgd/genome, go_root=<DATA_ROOT>/data/go, overwrite=False)` once (class and `load_dotenv` stubbed) and returns its `resolve_gene_name`.
- `canonical_concentration("0.5", "nM")` falls through the family to 0.5 nM; `read_key_conditions` maps filename to condition and refuses a header whose third column is not `control_set`.
- Left uncovered: the 250,000-record commit branch of `process` (a local constant; would need that many records) and `main`.
