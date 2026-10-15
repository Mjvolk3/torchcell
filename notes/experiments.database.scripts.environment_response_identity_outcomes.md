---
id: et1hjktkwdhux9h7guxlh7h
title: Environment_response_identity_outcomes
desc: ''
updated: 1792091135944
created: 1792091135944
---

## 2026.10.15 - Bracketing the environment-response identity keys

Lane W of the ontology unification. Re-keys every store in `ENVIRONMENT_RESPONSE_DATASETS` under five L1 uniqueness keys in one read-only streaming pass per store, so the question "does classifying the unread identity fields change any outcome" is answered by two bounds instead of by arguing about a field:

| key | the strain | the condition |
|---|---|---|
| `head` | the shipped genotype signature | the seven-element condition signature the rule carried before lane W |
| `shipped` | `_pair_key` as it stands | `_pair_key` as it stands |
| `genotype_base` | `(systematic_gene_name, perturbation_type, perturbed_gene_name)` only | `head`'s |
| `genotype_full` | the WHOLE stored perturbation dump | `head`'s |
| `environment_full` | the shipped signature | the WHOLE stored environment |

A key that reads more fields can only split a group, so `head - genotype_full` is the number of records that collapse although an unread genotype field distinguishes them, and `head - environment_full` the same on the environment axis. `--field-table` prints which of the 92 beyond-base genotype fields each of the three identity sites reads, from the schema union and the per-site field sets.

Run (from the repo root, with the torchcell env):

```bash
PYTHONPATH=. python experiments/database/scripts/environment_response_identity_outcomes.py --field-table \
  --out experiments/database/results/environment_response_identity_fields.json
PYTHONPATH=. python experiments/database/scripts/environment_response_identity_outcomes.py \
  --dataset yeastphenome --data-root $DATA_ROOT \
  --out experiments/database/results/environment_response_identity_outcomes.json
```

One store per invocation keeps the passes independent, so the 16 can run in parallel and the results files merge. `env_chemgen_hoepfner2014` is 118 GB and 30.3M records, which is hours; every other store is minutes.

Three dev stores of this family were pickled under class names the unification renamed and raise `AttributeError: ReleasedGeneKind` under plain `pickle`, so the reader resolves a renamed class through `RENAMED_CLASSES` and reports every substitution in `renamed_classes_read`. That is read-only and reports what it did; it is not a fallback that hides a stale store.

Findings: [[torchcell.verification.environment-response-identity]].
