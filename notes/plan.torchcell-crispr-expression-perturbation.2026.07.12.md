---
id: ot86if1hiuijma0fsx2y39z
title: Torchcell CRISPR Expression-Modulation Perturbation (AXIS 4)
desc: ''
updated: 1783889035662
created: 1783889035662
---

## 2026.07.12 - CRISPR expression-modulation perturbation: design (with user)

Extends the landed perturbation ontology (`[[plan.torchcell-perturbation-ontology.2026.07.08]]`,
`[[perturbation-ontology-refactor-landed]]`) with a **fourth axis** to represent CRISPRi/CRISPRa
screens. Target datasets: **Lian 2019 MAGIC** (`lianMultifunctionalGenomewideCRISPR2019`) and
**Mormino 2022** (`morminoIdentificationAceticAcid2022`). Chosen path = **Option B (full sequence
fidelity)**: capture the guide RNA spacer + effector identity as engineered genomic content.

### The three prior axes (recap) + why CRISPRi/a is a fourth

Every genotype difference from S288C was one of three axes x provenance x SO mechanism:
`PresenceAbsence` (present/absent), `CopyNumberVariant` (DNA dosage of a present gene),
`Sequence` (SNP/indel allele). A CRISPRi/CRISPRa perturbation fits NONE of them:

- gene is **present** (not a presence/absence edit),
- gene **DNA copy number is unchanged** (not a CNV),
- gene **sequence is unedited** (not a Sequence allele).

What changes is **expression output**, effected in **trans** by an inserted dCas9/dCas12a
effector + a gene-specific guide RNA. Hence a new axis:
**`ExpressionModulationPerturbation`** -- engineered modulation of a present, sequence-unedited,
copy-number-unchanged gene's expression level.

### Scout findings that shaped the design (2026.07.12)

- **Effector is NOT always dCas9.** Lian: activation = `dLbCas12a-VP`, interference =
  `dSpCas9-RD1152`, deletion = active `SaCas9`. Mormino: `dCas9-Mxi1`. => effector must be a
  **sourced field**, and leaves are named by FUNCTION (`CrisprActivation`/`CrisprInterference`),
  never by protein.
- **SO term for a CRISPR guide = `SO:0001998` (sgRNA)** -- "a small RNA oligo, typically ~20
  bases, that guides the cas nuclease to a target DNA sequence in the CRISPR/cas mutagenesis
  method." NOT `SO:0000602` (guide_RNA = trypanosome mRNA-editing guide; wrong meaning).
  Verified via EBI OLS. `insertion` confirmed `SO:0000667`.
- **Option B achievability:** **Lian = YES** (whole genome-scale library released; `Sequence`
  col = spacer, `Name` col = target common name, effector fixed per file; 37,817 CRISPRa /
  37,870 CRISPRi / 24,806 CRISPRd guides; the `Score` col is a guide-DESIGN rank, NOT
  phenotype -- per-guide screen fitness is raw NGS in SRA PRJNA504483). **Mormino = PARTIAL**
  (only ~6 enriched target genes + uniform `dCas9-Mxi1`; spacers live upstream in Smith 2017)
  => Mormino scaffolds-and-defers the guide (nullable `guide_sequence`).

### DECISIONS (with user, 2026.07.12)

1. **Effector = a FIELD on the leaf** (not a separate per-strain `GeneAddition`). The dCas9/
   dCas12a cassette is IDENTICAL across every strain in a library (a background constant), so
   duplicating it per record is wasteful; the leaf names the effector identity string and the
   full cassette sequence is documented once at the dataset/ReferenceGenome level. **Forward
   compat ("field now, plasmid later"):** the leaf also carries nullable off-graph pointers
   `effector_plasmid_uri`/`effector_plasmid_sha256` (None today) so upgrading to full-plasmid /
   SBOL capture is a non-breaking extension, aligning with the CLAUDE.md "grow toward
   SBOL-level design capture" path. Record shape does not change when fidelity is upgraded.
2. **CRISPRd = `CrisprDeletionPerturbation(DeletionPerturbation)` (IN scope, revised).**
   Model-by-state: the OUTCOME is an absent gene, so it belongs on the presence/absence axis
   under `DeletionPerturbation`, as a THIRD sibling mechanism beside `KanMxDeletion`/
   `NatMxDeletion` -- exactly the axis on which those two already differ. This is NOT a new
   *kind* of thing (which is what felt weird), it is another *mechanism* of the same absent
   state, and it KEEPS the guide (resolving the earlier "plain deletion discards the certain
   input" objection). `issubclass(_, DeletionPerturbation)` still catches every KO. The
   residual DESIGNED-vs-REALIZED uncertainty (pooled screens assert an unverified outcome)
   stays ORTHOGONAL -- no certainty axis this pass; the conversion / BioCypher modules must map
   designed<->realized (CRISPRi<->knockdown-hypomorph, CRISPRa<->overexpression,
   CRISPRd<->deletion). Tracked in `[[designed-vs-realized-perturbation-material-entity]]`.
3. **Shared `CrisprConstruct` component.** The CRISPR guide payload now appears on TWO axes
   (absence via active `SaCas9`; expression-modulation via dead Cas). Factor it into one
   composed sub-model (a first-class MATERIAL ENTITY -- the guide+effector we introduced),
   rather than duplicating fields across three leaves. The CRISPR *tool* is orthogonal to the
   *outcome axis* -- same guide-directed machinery, different consequence by effector.
4. **This pass lands THREE leaves:** `CrisprActivation`, `CrisprInterference`, `CrisprDeletion`.

### The schema

**`CrisprConstruct(ModelStrict)`** -- composed field `crispr` on every CRISPR leaf. The
guide+effector material entity; NOT itself a `GenePerturbation` (no systematic name), so it
does not enter the perturbation union / hierarchy tests.

| field | type / default | note |
|---|---|---|
| `effector` | `str` (required) | Cas effector fusion, e.g. `SaCas9`, `dSpCas9-RD1152`, `dLbCas12a-VP`; sourced, never guessed |
| `guide_sequence` | `str \| None = None` | spacer, INLINED as identity; None => Mormino scaffold-defer |
| `n_guides` | `int \| None = None` | Mormino "1-16 gRNAs/gene" |
| `effector_plasmid_uri` | `str \| None = None` | off-graph plasmid/SBOL pointer (future full-plasmid capture) |
| `effector_plasmid_sha256` | `str \| None = None` | required-if-uri (validator) |

**AXIS 1 (presence/absence) -- new deletion mechanism leaf:**

- `CrisprDeletionPerturbation(DeletionPerturbation)` -- `perturbation_type="crispr_deletion"`;
  inherits `state="absent"`, `mechanism_so_id="SO:0000159"` (deletion), `provenance="engineered"`;
  adds `crispr: CrisprConstruct` (Lian effector `SaCas9`) + optional `donor_sequence: str | None`
  (Lian's CRISPRd `Sequence` col = guide+HR-donor; loader splits).

**AXIS 4 (NEW) -- `ExpressionModulationPerturbation(GenePerturbation)`**, ABC, never
instantiated. Indexed by the **target** gene (`systematic_gene_name`). Engineered modulation of
a PRESENT, sequence-unedited, copy-number-unchanged gene:

| field | type / default | note |
|---|---|---|
| `state` | `str = "present"` | gene remains present (parity w/ `EngineeredCopyNumber`) |
| `provenance` | `str = "engineered"` | |
| `mechanism_so_id` | `str = "SO:0001998"` | sgRNA (guide-directed effector) |
| `mechanism_so_name` | `str = "sgRNA"` | |
| `expression_direction` | `str` (ABC declares; leaf defaults) | validator `{increased, decreased}` |
| `crispr` | `CrisprConstruct` | shared guide+effector material entity |

- `CrisprActivationPerturbation` -- `perturbation_type="crispr_activation"`,
  `expression_direction="increased"` (Lian effector `dLbCas12a-VP`).
- `CrisprInterferencePerturbation` -- `perturbation_type="crispr_interference"`,
  `expression_direction="decreased"` (Lian `dSpCas9-RD1152`; Mormino `dCas9-Mxi1`).

```
GenePerturbation
├─ PresenceAbsencePerturbation ─ DeletionPerturbation (ABC)
│     ├─ KanMxDeletion / NatMxDeletion / MeanDeletion / MarkerDeletion
│     └─ CrisprDeletionPerturbation         crispr: CrisprConstruct (SaCas9) + donor_sequence
└─ ExpressionModulationPerturbation (ABC, AXIS 4)      crispr: CrisprConstruct + expression_direction
      ├─ CrisprActivationPerturbation       (increased, dLbCas12a-VP)
      └─ CrisprInterferencePerturbation     (decreased, dSpCas9-RD1152 / dCas9-Mxi1)
```

### Integration checklist (schema.py + tests)

- `schema.py`: add `CrisprConstruct`; add `CrisprDeletionPerturbation` under
  `DeletionPerturbation`; add `ExpressionModulationPerturbation` ABC + its 2 leaves (AXIS-4
  comment banner mirroring the existing axis banners); add all THREE leaves to the
  `GenePerturbationType` union.
- `tests/torchcell/datamodels/test_ontology_invariants.py`: add
  `ExpressionModulationPerturbation` to `ABSTRACT` (`DeletionPerturbation` already there);
  add `CrisprDeletionPerturbation` + `CrisprActivationPerturbation` +
  `CrisprInterferencePerturbation` to `FACTORY` (minimal valid instance:
  `_SYS` + `crispr=CrisprConstruct(effector=...)`). Existing generic invariants then auto-cover
  them (union==leaves, discriminator uniqueness, round-trip, SO well-formedness, provenance,
  identity, Liskov). Add targeted tests: `expression_direction` validator; `effector` required;
  `effector_plasmid_uri`-implies-`sha256`; `issubclass(CrisprDeletion, DeletionPerturbation)`.
- Loader-side (NOT this pass): Lian uses COMMON names (`Name` col, e.g. `ACS1`) -> loader maps
  common->systematic (via `all symbols.xlsx` / SGD) before constructing the leaf, since the
  `GenePerturbation` validator requires systematic `Y..`/`Q..`/`YNC..` names.

### Sequencing

1. Land the schema + integrity tests (this note) via this worktree + merge queue.
2. THEN build **Lian** CRISPRa/i/d (guides populated) and **Mormino** CRISPRi (guides deferred),
   each on its own worktree, L0-L4, enqueue-merge -- per the
   `[[metabolic-overnight-build-campaign]]` pattern.
