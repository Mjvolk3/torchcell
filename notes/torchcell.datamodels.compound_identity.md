---
id: 8qc2ra55yg8h9ve0gah7v61
title: Compound_identity
desc: ''
updated: 1784601915913
created: 1784601915914
---

## 2026.07.20 - UI-2 compound-identity resolver

Shared, offline, pure resolver that fills `Compound` structure IDs from a committed,
sha256-pinned name->structure table, and gap-marks the unresolved residue. Part of the
env/chemogenomic audit pipeline (UI-2; plan [[plan.env-compound-identity-resolver.2026.07.20]]).

- **`resolve_compound_identity(name=, pubchem_cid=, known_proprietary=)`** -> typed
  `CompoundIdentityResolution` (status RESOLVED / UNRESOLVED_PUBLIC / PROPRIETARY). Reads
  ONLY `compound_identity_table.json` (sha256-self-checked at import); NEVER hits the
  network at build/CI/test time. Mirrors the single-shared-resolver shape of
  `SCerevisiaeGenome.resolve_gene_name`.
- **`resolved_compound(...)`** loader helper builds a `Compound` fill-or-gap ADDITIVELY:
  fills only structure fields that are None (never clobbers a caller's smiles/pubchem_cid,
  e.g. hoepfner2014), and attaches `ProvenanceGap(field="inchikey", reason=...)` only when
  inchikey is truly None -- `deferred_pending_source_review` for unresolved public names,
  `not_reported_by_primary` for known-proprietary (hoepfner CMB codes, smith2016 vendor IDs).
- **Table** = 33 pinned records (32 resolved to a valid InChIKey; `tunicamycin` a mixture ->
  UNRESOLVED_PUBLIC gap, honest not guessed). Built ONCE offline by
  `scripts/build_compound_identity_table.py` (PubChem PUG REST, `RetrievalMethod.pubchem_api`);
  the committed JSON is canonical thereafter.
- Wired into 13 loaders + the yeastphenome plant-defensin peptide retype (-> BiologicPerturbation).
- NOT done here (follow-ups): the strict structure-or-gap `Compound` validator (UI-2b -- ~65
  name-only sites incl. media.py); `Compound` gaps vs `Media.open_gaps` reconciliation; the
  full DB rebuild the loaders' new output implies (UI-3 / KG build).
