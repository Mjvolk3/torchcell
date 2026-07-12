---
id: 52fsf019nw7vx7vtom9xt4j
title: Provenance
desc: ''
updated: 1783563742010
created: 1783563742010
---

## 2026.07.08 - Behavior that makes "where did this come from?" answerable + re-runnable

This module exists so that, for ANY captured artifact, we can answer "where did this come from and how was it done?" purely from our own records -- and can PROVE that answer on rebuild rather than trusting it. The pydantic record types live in [[torchcell.literature.manifest]] (pure data, bottom of the import graph); this module holds the behavior that acts on them: run the recorded retriever, verify the stored bytes against their sha256, and re-check the source for upstream drift. The point is that the artifact + its sha256 is canonical while the URL is only historical metadata, so drift or link-rot is DETECTED (a sha256 mismatch) instead of silently followed.

- `run_retriever` / `check_source`: resolve a `RetrievalRecord` to versioned source code via a registry key + params (a testable recipe, never a free command string), re-run it, and report whether it still yields our recorded bytes.
- `verify_artifact`: confirm the on-disk mirror is intact by matching the stored file's sha256 -- the integrity check every rebuild depends on.
- A failed `check_source` never overwrites our canonical artifact; the caller decides whether to version a NEW record for the changed upstream. Retrievers themselves are in [[torchcell.literature.retrieve]].
