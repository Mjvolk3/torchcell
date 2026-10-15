---
id: lp67yyqxp9zjb3md83im93z
title: Registry
desc: ''
updated: 1792054836634
created: 1792054836634
---

## 2026.10.15 - The construct store, built as an index over the tiers that exist

`torchcell/sequence/construct/registry.py` is the construct store: an INDEX from a
construct's identity to an `ArtifactRef` into the `library`, `raw` or `objects` tier,
mirroring `torchcell/sequence/genome/registry.py` member for member. It is lane B of the
Path U ontology unification ([[scratch.2026.10.15.ontology-unification-decision]]),
built to the spec in [[scratch.2026.10.15.ontology-review-2-material]] "Recommended
changes" and audit B section 2 ([[scratch.2026.10.15.ontology-audit-B_material]]).

### Why an index and not a fifth tier

`ArtifactTier` is `Literal["raw","genomes","library","objects"]` and that Literal sits
in 120 of 146 loader contract closures, so a `"constructs"` member is a full-rebuild
change bought for a directory name. The bytes stay where they are and
`$DATA_ROOT/torchcell-constructs/<set>/manifest.json` holds manifests only. Every contig
reaches its bytes through the ONE resolver, `torchcell.artifacts.resolve`, which
re-hashes on every read; `resolve_contig` adds the set's own claim on top, so the bytes
must also hash to the sha256 the contig IS.

### The three identity rules, each measured on the 17 Cachera maps

| rule | the measurement that forced it |
|---|---|
| identity is the file's sha256, handle = first 12 hex | `pBTX002.gb` carries `LOCUS pBTX001`; `pBTX001(ante113).gb` carries `LOCUS ANTE113_with_NatMX`. A LOCUS-derived id names the wrong replicon |
| `source_name` is a quoted `SourcedValue` or a typed gap | the replicon's name lives in the paper text and Supplementary Table S2, never in the file |
| feature key = label + SO id + 1-based end-inclusive span | 0 of 570 non-source features carry a `/locus_tag`; label alone collides in 4 of 17 maps; label plus span still collides in 1 of 17 (pHO22's `Ampicillin` gene/CDS pair at 1543..2403); the triple is unique in all 17 |

A primer is INDEXED and REFUSED as a placement target (`PLACEMENT_TARGET_REFUSED_SO`,
SO:0005850). The data is the data, so the annotation is stored; what is refused is
pointing an added gene at an oligo's footprint. 297 of 570 features in the set are
primers, so without the refusal the modal feature of the store would be a legal target.

Units are converted at this boundary: `plasmid.Location` is 0-based half-open,
`ConstructSpan` and `GenomicSpan` are 1-based end-inclusive. One store serving native
and construct genes carries one convention.

### Parser hardening (no warn-and-fix)

`parse_genbank_component` now raises `GenBankLocationError`, naming the file and the
feature, for two cases it used to flatten:

- an **origin-wrapping** location Biopython rewrites (its warning is caught under
  `catch_warnings` with `simplefilter("always")`; the default `once` registry hides a
  repeat, which is why 7 of 17 maps warn by default and 11 of 17 warn in truth);
- a **compound `join`**, which a single-range `Location` can only hold as its envelope.

Measured consequence on the mirror: **6 of 17 maps index, 11 are blocked**, each with
exactly one offending feature (an `AmpR` CDS, or the `pUC\ori` of `pBTX002.gb`). The
paper's **pBTX2 is blocked**, so the 8,104 bp XII-5 insert of
[[torchcell.sequence.plasmid-and-genomic-content-design]] is not indexable until a span
can hold parts. The previous behavior stored that `pUC\ori` as the whole 12,096 bp
plasmid. A blocked map stays in `files` with its retrieval record and sha256 and is
named in `blocked` with the parser's verbatim refusal.

`plasmid.SORole` is now `schema.SOTerm` (one SO map for the repo), which additionally
validates the `SO:NNNNNNN` shape the local duplicate never checked, and the module
docstring states the `Component` is a PROJECTION of the canonical bytes: every qualifier
but `/label` and `/gene` is dropped, `source` is dropped, and GenBank -> Component ->
GenBank is not identity, so any emission is regenerated from the bytes.

### Deposited set

`cachera2023_cri_spa_plasmids`, 17 files, 6 contigs (pBTX1, pHO8, pHO22, pHO24, pHO25,
pHO29), 11 blocked, `sequence_status: design`, `provenance_complete: True`. See
[[scripts.provision_construct_sets]] for how the completeness claim was earned.

## 2026.10.15 - A span holds parts, so 17 of 17 maps index

Lane T. `ConstructSpan` is now ordered `parts` plus a `composition`, so the parser no
longer has to refuse what it cannot hold. The set's own numbers:
**17 contigs, 0 blocked, 570 features** (was 6, 11 and 208). See
[[experiments.036-dataset-fixes-before-kg-build.scripts.cachera2023_construct_span_parts]]
for the full table and the per-map features.

### What the registry had to add

- **`BlockedReason` is now the parser's own vocabulary**, `plasmid.LocationRefusal`,
  re-exported under the name the index uses, exactly as `SORole` is `SOTerm`. The reason
  comes off `GenBankLocationError.refusal` instead of being hardcoded where the refusal
  is caught, which is how all 11 Cachera refusals came to be recorded as
  `origin_wrapping_location` whatever they were. Those two members stay in the
  vocabulary, documented as no longer producible, because the index deposited earlier
  that day recorded them.
- **`ConstructContig._check_span` proves an `origin_spanning` claim against the bytes.**
  A span can check the shape of its parts but not their relation to the replicon: that
  needs the length and the topology, which the contig holds. Every part must be within
  the contig's length, a wrap is refused on a linear contig, and a wrap's first part must
  end at the contig's LAST base.
- **`TOPOLOGY_SO` + `ConstructContig.topology_term`** ground `topology` in SO's
  `topology_attribute` (SO:0000986), whose two children are exactly the two members of
  the Literal: `SO:0000987 linear` and `SO:0000988 circular` (principle 6). A derived map
  and a read-only property, never a second stored field.
- **`_project` carries the parts and the composition across** range for range, in the
  order the source wrote them. That order matters: Biopython hands a reverse-strand
  compound location's parts back in 5'->3' order, so the parser re-reverses them and the
  stored order is the file's.

### Keys did not churn

`FeatureKey.token` reads `<label>@<so_id>:<parts><strand>` where the parts carry a
composition sigil: `2684-3796` for a simple span (unchanged, so every key already
recorded for a single-interval feature still reads the same string),
`wrap:11559-12096,1-128` for an origin crossing, `join:18-20,30-40` for a join. The sigil
is what keeps a wrap and a join over the same two intervals from colliding.

### Names for the 11

The nine `pBTX0xx` gene-targeting vectors are never named in the paper body, so their
names are quoted from SUPPLEMENTARY TABLE S2 of `Supplementary Material.docx`, a member
of the already-pinned SI container, extracted and re-hashed in memory on every
provisioning run (`SI_TEXT_METHOD = "docx-document-xml-paragraph-text"`); pBTX2 and
pCCM023 are named from `paper.md`. 8 from the paper, 9 from the table, 0 typed gaps.
Open question left for the owner: the nine `pBTX0xx` and `pCCM023` are filed as
`ConstructKind.plasmid`, and Table S2 plus the paper's "released from purified plasmids
by digestion with NotI" read more like `integration_vector`; `plasmid` is literally true
of all of them, so lane B's assignment is left alone.
