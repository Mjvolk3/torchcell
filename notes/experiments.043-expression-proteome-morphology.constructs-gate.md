---
id: 543wadenv4c5g0kcb381ny4
title: Constructs Gate
desc: ''
updated: 1792634618355
created: 1792634618355
---

## 2026.10.21 - Design: a report-class rule for unresolved construct references

Design only. Nothing here is implemented, and no number below is a prediction: every
count is measured on this branch and the command that measured it is given.

### What went wrong, concretely

The Vaishnav 2022 loader has named Addgene 127546 since it landed. Its record carries
`construct_name="pTpA-YFP dual-reporter CEN plasmid (Addgene 127546)"`, a
`replicon_statement` of `"CEN plasmid"`, and `placement=None` beside a typed
`ProvenanceGap(field="placement", reason=not_released_by_primary)` whose note reads
"the method names the backbone by an Addgene handle, 'within a CEN plasmid (Addgene:
127546)', and deposits no map". Every one of those statements is true of the PRIMARY
paper, and the gap is correctly typed. The store built, verified, and would have been
admitted.

What nobody was told is that the plasmid is public and fetchable. The owner's point,
2026-10-21: "this should have been caught on build because we need all of the genomic
structures etc that will be used for reconstituting the cell repr." The 110 bases the
release carries are 11 percent of the 1,003 bp window a FUDT embedding reads; the other
893 bp are backbone that was simply not in the repository
(`experiments/043-expression-proteome-morphology/results/vaishnav_promoter_windows.json`).

### The rule that was already there, and the hole in it

`torchcell/verification/construct_placement.py` emits `construct_placement_states`, a
`report`-class L1 row (registered in `GENOME_REFERENCE_RULES` in
`torchcell/verification/report.py`, so `finding_class_of` returns `FindingClass.report`).
It sorts each addition into `sequence_backed`, `host_placed_without_coordinate` or
`composition_only`, and counts an untyped absence separately.

It would have classified Vaishnav's 71,158 additions as `composition_only`, reason
`not_released_by_primary`, and `passed=True`. That is the hole: the row asks whether a
record SAYS it has no sequence, never whether the construct it NAMES could be resolved.
A construct nobody ever deposited and a construct sitting in a public registry under an
accession the loader prints are the same row today.

### The proposed rule: `named_constructs_resolve`

One new `report`-class L1 row, beside the existing one and never replacing it.

**Subject.** For every addition perturbation (the family derived by
`addition_perturbation_types()`, which is the four subclasses of
`GeneAdditionPerturbation` in `torchcell/datamodels/schema.py`) and every
`IntegratedCassette`, take the construct the record NAMES and ask whether it resolves to
a constructs-tier contig with a sha256.

**The fields it reads**, all in `torchcell/datamodels/schema.py`:

| class | line | field | what it holds |
|---|---|---|---|
| `GeneAdditionPerturbation` | 3023 | `construct_name` | the plasmid or cassette name, verbatim |
| `GeneAdditionPerturbation` | 3017 | `replicon_statement` | the source's words for the replicon |
| `GeneAdditionPerturbation` | 3031 | `placement` | `ConstructPlacement` or `None` with a typed gap |
| `IntegratedCassette` | 2722 | `name` | the cassette designation, verbatim |
| `IntegratedCassette` | 2734 | `elements` | the cassette's parts, in order |
| `IntegratedCassette` | 2745 | `placement` | same type, required and gap-able |
| `ConstructPlacement` | 2604 | `contig_sha256` | the construct's identity |
| `ConstructPlacement` | 2605 | `contig_ref` | the `ArtifactRef` pinning the same bytes |
| `ConstructPlacement` | 2608 | `feature_key` | label, SO id, 1-based span within the contig |
| `AlleleEdit` / leaves | 969, 1648, 1699 | `cassette` | the replacement cassette, e.g. `kanMX4` |
| `CrisprGuidePerturbation` | 1455 | `effector_plasmid_ref` | an `ArtifactRef`, `None` on every record today |

**Resolution.** A named construct resolves when a `ConstructManifest` under
`$DATA_ROOT/torchcell-constructs/` holds a contig whose `source_name.value` is that name
and whose bytes re-hash, checked through `resolve_contig` in
`torchcell/sequence/construct/registry.py`, which already re-hashes on every call. The
name is matched against `source_name`, the SOURCED name, never against a filename and
never against the GenBank LOCUS line, for the reason the registry's own docstring gives:
two of the 17 Cachera files carry a LOCUS naming a different replicon.

**Four outcomes per named construct**, which is what makes the row a worklist:

1. `resolved` -- a constructs-tier contig carries the name and its bytes re-hash.
2. `registry_handle_unresolved` -- the name or the replicon statement carries a public
   registry accession (`Addgene \d+`, `addgene:\d+`, an NCBI accession, a JBEI ICE id)
   and no contig resolves. **This is the Vaishnav state, and it is the finding the rule
   exists to raise**: the bytes are fetchable, so the gap is a worklist item, not a
   property of the source.
3. `named_unresolved` -- a name with no handle and no contig: a construct that may be in
   a supplement nobody has mirrored.
4. `unnamed` -- no `construct_name`, no cassette name. Nothing to look for.

**Class: `report`, and why.** By the standing rule in `FindingClass`'s own docstring, a
rule whose subject is the record against OUR references is listed under the verdict with
its statistic and never fails a store. A source that released no map is correctly stored
with a typed gap, and failing its build would be the genome gating the data. So the row
is added to `GENOME_REFERENCE_RULES` beside `construct_placement_states`, `passed` is
False only on an untyped absence, and outcome 2 is an open finding carrying the handle it
parsed. The exhaustiveness test over the fleet's rule names
(`tests/torchcell/verification/test_finding_class.py`) keeps it classified.

**Where it would have fired.** `admit` in `torchcell/knowledge_graphs/kg_manifest.py` is
the one place a dataset crosses into the served graph, so the row belongs in the
admission report a human reads there, as well as in each family verifier's verdict. A
`registry_handle_unresolved` finding at admission is a question to the owner, which is
exactly the exchange that happened on 2026-10-21, except two months late.

### Measured state today

Served graph, `/scratch/projects/torchcell-scratch/database/data/torchcell/` (19
datasets: the Costanzo and Kuzmin SGA stores, SGD essentiality, Ohya morphology, the two
SynLethDB stores): **no served dataset references a construct at all.** Every
construct-naming loader is a dev-tree build today, so the first run of this rule is a
run over the admission queue, not over the live store.

Eighteen loaders carry a construct reference. Measured with
`grep -rl "construct_name=" torchcell/datasets/` (13 files),
`grep -rl "IntegratedCassette(" torchcell/datasets/` (1 file) and
`grep -rl "effector_plasmid_ref" torchcell/datasets/` (4 files); `ConstructPlacement(`
occurs in exactly 1 of the 18.

| loader | reference | placement | resolves |
|---|---|---|---|
| `scerevisiae/cachera2023.py` | `construct_name="Btx-cassette"` | `ConstructPlacement` on `PBTX1_SHA256` | yes, `cachera2023_cri_spa_plasmids` |
| `scerevisiae/vaishnav2022.py` | `construct_name` naming Addgene 127546 | `None` + typed gap | the contig now exists (`vaishnav2022_addgene_127546`); the loader still points at nothing |
| `scerevisiae/mormino2022.py` | `construct_name="pMM4_14L"` | `None` + typed gap | no contig |
| `scerevisiae/ozaydin2013.py` | `construct_name="YB/I/BTS1"` | `None` + typed gap | no contig |
| `scerevisiae/hackett2020.py` | `IntegratedCassette(...)` | `None` + typed gap | no contig |
| `pputida/carruthers2025.py` | `construct_name="pIY670"` and three more | `None` + typed gap | no contig |
| `pputida/menasalvas2025.py` | `construct_name="RSF1010 PpedF-RBS-pyrF"` | `None` + typed gap | no contig |
| `pputida/banerjee2024.py`, `pputida/desiqueira2025.py`, `pputida/kang2026.py`, `pputida/thompson2019_valerolactam.py`, `pputida/yunus2026.py`, `ecoli/brunk2016.py`, `ecoli/foo2014.py` | `construct_name=` per record | `None` + typed gap | no contig |
| `ecoli/cui2018.py`, `ecoli/rachwalski2024.py`, `ecoli/rapp2026.py`, `ecoli/rapp2026_platforms.py` | `effector_plasmid_ref` | `None` on every record | no contig; the field is the declared "field now, plasmid later" slot |

So: **1 of the 18 resolved before 2026-10-21, and the constructs tier holds 2 sets** --
`cachera2023_cri_spa_plasmids` (17 contigs) and `vaishnav2022_addgene_127546` (1 contig,
deposited today). Which of the remaining 16 carry a public registry handle rather than an
unmirrored supplement is exactly what outcome 2 would answer, and that triage has not
been run. `scerevisiae/lian2019.py`, `pputida/banerjee2025.py` and `ecoli/mohiuddin2022.py`
mention these classes only in their docstrings and are not counted.

### What this design does not decide

Whether a `registry_handle_unresolved` finding should also block an incremental
admission is the owner's call, not the rule's. Being `report`-class it cannot, by
construction, and that is the conservative default. Raising it to a gate would mean
asserting that a store is wrong because a registry we can reach has bytes we have not
fetched, which is a different claim from the twelve principles' "the data is the data"
and needs its own argument.
