---
id: si6kyxz3lh8qn3kq0bej8sg
title: Supported Queries
desc: ''
updated: 1790725777181
created: 1790725777181
---

## 2026.09.29 - Lifecycle, validation, and the drift issue flow

Plan: [[plan.data-release-program.2026.09.29]], Decisions 3 and 4; issue #468. Code:
[[torchcell.knowledge_graphs.supported_queries.registry]],
[[torchcell.knowledge_graphs.supported_queries.cypher_deps]],
[[torchcell.knowledge_graphs.supported_queries.check]].

A supported query is a `.cql` file under `torchcell/knowledge_graphs/queries/` with an entry
in `torchcell/knowledge_graphs/supported_queries/registry.json`. The file is package data,
so an installed torchcell carries it.

### Lifecycle

1. **Add.** Write the `.cql` (one `UNION ALL` block per dataset, `dataset.id` literals, the
   property-shaped `RETURN`), add an entry with `status: supported` and
   `validated_release`/`dataset_composite`/`since_kg_version` null, then run
   `python -m torchcell.knowledge_graphs.supported_queries validate <id> --release <newest>`.
   Validation refuses on any drift except `composite_changed` and records the release,
   the composite over the selected datasets' `content_sha256`, and `since_kg_version`.
2. **Keep.** Every commit that touches the schema surface, the BioCypher schema config, the
   cell adapter, a `.cql`, the registry package, or a release snapshot runs the
   `supported-queries` pre-commit hook. A drifted `supported` query blocks the commit;
   `TORCHCELL_QUERY_DRIFT_ACK=1` lets a deliberate ontology revision through after the
   report is read.
3. **After each KG release.** The stamp writes a new snapshot; the check then compares every
   query with it, and each query whose selected datasets changed shows
   `composite_changed`. Re-run `validate <id> --release <new>` for each one that still
   returns what its consumers expect (read the diff of the datasets first), commit the
   registry, and close the matching issue by hand.
4. **Deprecate.** When a query should no longer be maintained, set `status: deprecated` and
   `deprecated_in: <KG version>`. A deprecated query is still checked and reported, never
   fails the hook or CI, and files no issue. Removing the entry and the file is a separate,
   later decision.

### Issue flow

The CI job `query-drift` (`.github/workflows/docs.yaml`) installs only pydantic, PyYAML and
python-dotenv and runs `check --json` against the newest committed snapshot.

- On a pull request, a drifted `supported` query fails the job.
- On `main`, each drifted `supported` query opens one issue titled
  `Before the next KG build: supported query <id> drifts against <release>`, labeled
  `before-next-kg-build`, whose body is that query's report (drift kinds and details, the
  release, the checkout commit, the re-validation command). An open issue with the exact
  title is updated instead of duplicated. The job passes after filing: the issue is the
  queue for the next KG build (plan Decision 4). The pre-commit hook never files issues (no
  token, and it runs on every machine).
- Closing is manual, after `validate` on the new release or a deprecation.

The label `before-next-kg-build` is the same queue the dataset re-verification issues use,
so one filter shows everything that must land before the next build.
