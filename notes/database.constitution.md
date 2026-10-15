---
id: lg5ct0f2bxat3i7ecl35rbn
title: Constitution
desc: ''
updated: 1792052346839
created: 1792052346839
---

## 2026.10.15 - The torchcell database constitution

The ranked principles that decide what the ontology says, how a record is allowed to say it, and what keeps it true over time. Correctness of the ontology has no direct measurement; the best estimate of it is how faithfully every change upholds these principles, so a conflict between two of them is resolved in favor of the higher-ranked one. The owner's rulings they come from are quoted where they exist. The companion skill `/db-constitution` loads this note before any database change, and `[[torchcell.datamodels.schema-change-policy]]` holds the mechanical rules and the record format.

### Part I. What the ontology says

1. **Ontologize the experiment, not the organism.** The record is a typed `genotype x environment -> phenotype` with perturbations and provenance, and structure grows from what real papers report. A class exists because it names a different physical thing or a different physical measurement, never because of which host the data came from or who wrote the loader. Owner, 2026-10-15: "like why a bacterial experiment! it seems this is unnecessary distinction. we can run many same fitness experiments on both yeast and bacteria." A `BacterialFitnessExperiment` that differs from `FitnessExperiment` by a tag string is a violation of this principle, whatever it costs to retire.

2. **The data is the data.** Every released record is stored under the names, positions and values the source gave. The only drop is a sample the authors themselves withdrew, quoted. The genome never gates a record: every comparison with an annotation is a report-class finding for modeling time. Owner, 2026-10-12: "the data is the data!!!! it is what they reported ... we don't drop raw data unless they themselves pull it! ... the genome is separate.. it doesn't gate anything." When a principle below collides with this one, this one wins.

3. **Every value traces to pinned bytes.** A stored value carries a sha256-pinned artifact and a verbatim quote, and the artifact is canonical, not the URL. Rebuild from the recorded retrieval, verify the hash, detect drift, never follow it silently. An added gene points at sequence through an `ArtifactRef`, not a free string; a construct set is provenance-complete only when every file carries its own retrieval record.

4. **The material is a sequence-level fact on every record.** A perturbation is an edit to the total genomic content of the cell: deletions, integrations, alleles, present plasmids, heterologous cassettes. Every record states the deposited reference its identifiers are written against and the assembly set it resolves to, and added DNA is a contig with coordinates. The host is a property of the material, so it lives on the reference genome and on the perturbation, and nowhere above them.

5. **Strictly typed: one fact, one field, one closed vocabulary.** A property of a record (which host, resolved versus released, which assembly, which replicon class) is a typed field with a closed vocabulary, never a choice of class, never a free string, never `Any`. Two classes that assert the same fact are a defect; so is one enum that carries two facts (a `released` member on a host namespace would say "unresolved" and erase the host in the same value). Pydantic `ModelStrict` is the floor; the vocabulary is measured from the stored records, not assumed.

6. **Ground every served class in an external ontology.** Every BioCypher node and edge class carries an `is_a` to a Biolink class (3.2.1 is the pinned head ontology), chosen by the definition's meaning and not by label convenience: a phage challenge is a `biotic exposure` and not an `environmental exposure` because Biolink defines the latter as abiotic; a guide plus effector plasmid is a `nucleic acid entity`. Where Biolink has no term, the Sequence Ontology names the mechanism (`SOTerm`, one map for the repo). The grounding is part of the class, so a class with no honest parent is a design question, not a yaml gap to paper over.

### Part II. How a record is allowed to say it

7. **A required field is required, and its absence is typed.** Never weaken a required field for a thin source and never default silently. A loader states the value, or a `ProvenanceGap` with its reason and where it looked, per dataset, never from one shared constant. A silent `None` is how 26 of 26 gene-addition stores lost their plasmid pointers.

8. **Correctness decides a change; cost never does.** A served class is presumed correct until a more correct shape is shown, and then the change is made whatever it moves. Owner, 2026-10-15: "we care most about correctness and are willing right now to do full rebuilds if we identify even small semantically prudent change in onto." Additive-first is the usual route because an additive change is usually the correct one, not because it is cheaper; "moves 0 of 36 served closures instead of 36 of 36" is never a reason, and the 13 experiment pairs are what that reason produced. Convenience for a new source is never a reason either.

9. **Place by the physical measurement, using the DAG.** Same technology and same released quantity share a leaf and diverge in the experiment class above it; same parent with divergent specifics is a sibling leaf; a different technology is its own leaf with an explicit converter. A field is never reused with a different meaning.

10. **Be more rigorous than the source.** Capture to the bit, follow deferrals to the cited method paper, source every count and uncertainty type from the SI with its quote, back-solve ranges and take the conservative end, and say in the record when the source left a value implicit rather than guessing it.

### Part III. What keeps it true over time

11. **Interoperate as a superset.** Own the pydantic object; ingest GenBank first, then GFF3 and FASTA; emit GenBank, GFF3 and SBOL. An interchange representation is a projection of the canonical bytes, regenerated from them, never the canonical thing.

12. **Every change is a record, and the gates are mechanical.** A schema-change record per proposal with its correctness argument, alternatives and dated sign-off; fingerprinted closures and the build manifest; the loader-to-lane exhaustiveness test; the CSV endpoint gate before import; the release note and the supported-query revalidation. They exist so that a violation of principles 1 to 11 fails a landing instead of surfacing in a served store a release later.

### Reading the ranking

Principles 1 to 6 decide what the ontology says. Principles 7 to 10 decide how a record is allowed to say it. Principles 11 and 12 keep it true over time. A proposal that is cheaper, simpler or faster but weaker on a higher-ranked principle is the wrong proposal.

### Provenance of this note

Sources: `CLAUDE.md` (Vision, Schema changes, Provenance and Reproducibility, Adding Datasets), [[torchcell.datamodels.schema-change-policy]], the owner's rulings of 2026-10-12 and 2026-10-15 quoted above, and the ontology audits and reviews of 2026-10-15 ([[scratch.2026.10.15.ontology-unification-decision]]) whose findings each principle names.
