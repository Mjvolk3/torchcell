---
name: db-constitution
description: Load the torchcell database constitution (the twelve ranked principles of the ontology) before any change to the schema, a loader, the adapter, the graph yaml, the verifiers or a build, and check the change against each principle in rank order. Use whenever a task touches torchcell/datamodels/schema.py, torchcell/datasets/, torchcell/adapters/, biocypher/config/, torchcell/verification/ or a KG build, and before writing a schema-change record.
---

# The database constitution

Read `notes/database.constitution.md` ([[database.constitution]]) in full before the first edit. It is the ranked list of principles the ontology is built on. Correctness has no direct measurement, so a change is judged by how faithfully it upholds the principles, highest rank first. This skill is the checklist; the note is the authority, and a new ruling goes into the note under a dated section, never only into a conversation.

## When this skill applies

Every change to `torchcell/datamodels/schema.py` or `pydant.py`, every new or revised loader under `torchcell/datasets/`, every adapter or conf change under `torchcell/adapters/`, every edit to `biocypher/config/*.yaml`, every verifier rule, and every schema-change record. `/add-dataset` and `/torchcell-db-build` assume it has been read.

## The checklist, in rank order

Answer each in the PR body or the schema-change record's `correctness_argument`. A "no" on a higher-ranked line outweighs any number of "yes" lines below it.

1. **Experiment, not organism.** Does the change add or keep a class whose only difference from an existing one is the host, the author, or a tag string? If so, it is wrong: the host is a field (principle 4), and the pair retires.
2. **The data is the data, once the experiment can be stated.** Can every stored record say what went into the flask: a perturbation as a named edit to a stated reference, and every environment agent as identified material (structure, recipe or released name)? A record that cannot is held out with its count and quote in the ledger, never stored under a bare label with a gap. Given representability, does the change drop, refuse, remap or collapse any released record for a reason other than an author withdrawal quoted with its sha256? Does any new rule consult the genome to decide whether a record loads? If so, stop.
3. **Pinned bytes.** Does every new value point at a sha256-pinned artifact with a verbatim quote, and does every new file carry `source_url`, `retrieval_method`, `retrieval_command`, `sha256`, `retrieved_at`? A pointer to a store is an `ArtifactRef`, never a string pair.
4. **Sequence-level material.** Does the record state the deposited reference its identifiers are written against (`gene_namespace`) and the assembly it resolves to (`assembly_set`)? Does added DNA point at a contig with coordinates, or carry a typed gap saying why not?
5. **Strict typing.** Is every new property a typed field with a closed vocabulary measured from the stored records? Is there any free `str` where an enum belongs, any `Any`, any enum that carries two facts, any fact expressed by class choice?
6. **External grounding.** Does every new node or edge class carry an `is_a` to a Biolink 3.2.1 class chosen by its definition, with the reasoning in a yaml comment? Does every new mechanism carry an `SOTerm`?
7. **Required and typed absence.** Is any required field loosened? Is any absence a silent `None` rather than a `ProvenanceGap` with `reason` and `looked_in` per dataset?
8. **Correctness, not cost.** Is any shape chosen because it moves fewer served closures, fewer dev stores, or avoids a rebuild? If so, name the more correct shape and take it; the rebuild is the price of being right.
9. **Placement by physical measurement.** Same technology shares the leaf and diverges above it; a different technology is its own leaf with a converter; no field changes meaning.
10. **More rigorous than the source.** Every count, uncertainty type, threshold and unit is quoted from the paper or SI; deferrals are followed; ranges resolve conservatively; implicit values are declared implicit.
11. **Superset interoperability.** Emitted GenBank, GFF3 or SBOL is regenerated from the canonical bytes; nothing treats the projection as canonical.
12. **Record and gates.** One schema-change record per proposal (disjoint symbols), with a dated or linked sign-off quote; the impact report; `schema-breaking` when a served closure moves; the exhaustiveness test and the build manifest updated; the release note names every retired label.

## Hard stops

- A class whose name carries a host prefix and whose body differs from the general class by a tag or a pin: retire it, do not extend it.
- A `released`, `unknown`, `other` or `mixed` member added to a host or assembly vocabulary: refuse; the leaf class or a typed gap carries that statement.
- A default that asserts a host or an assembly on a leaf whose validator does not already prove it from the stored name's form.
- `rebuild_scope` chosen to avoid a rebuild rather than computed by the record tool.
- A perturbation or environment agent with no identity (a vendor or registry number with no released name or structure) stored under its label with an `inchikey` or similar gap: hold it out in the ledger instead; a gap is for an attribute of an identified thing, never for its identity.

## Related

- [[torchcell.datamodels.schema-change-policy]]: the mechanical rules, the record format, the tripwire.
- `/add-dataset`: the loader rules that follow from principles 2, 3, 7 and 10.
- `/torchcell-db-build`: the build and release path that principle 12 gates.
