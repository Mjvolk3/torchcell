---
id: 6dwr9boibcgbixu8mho928l
title: Strain Background
desc: ''
updated: 1790979778668
created: 1790979778668
---

## 2026.10.02 - Schema decision for #507 (strain background, chemogenomic leaves, culture protocol)

Umbrella issue #507 (audits #500 Vanacloig, #504 Wildenhain, #505 Hillenmeyer, #506 Hoepfner). Standard from #500: the served genotype is the strain's genome content as the researchers understood it, or a typed `ProvenanceGap` per missing element. Code: `torchcell/datamodels/schema.py` (record classes, fingerprinted), `torchcell/datamodels/strain_background.py` (shared allele specs and constructors), `torchcell/datamodels/identity.py` (projections). Tests: `tests/torchcell/datamodels/test_strain_background.py`, `tests/torchcell/adapters/test_strain_background_genome_node.py`.

### Placement: new classes only, opted into by the four loaders

Every class a served dataset already imports is left with its contract unchanged except `BarcodedKanMxDeletionPerturbation` (imported only by Vanacloig and Wildenhain). The new content rides on NEW classes that a chemogenomic loader opts into:

| new class | subclass of | carries |
|---|---|---|
| `StrainEnvironmentResponseExperiment` | `EnvironmentResponseExperiment` | `experiment_type="strain_environment_response"`, `environment: CultureEnvironment` |
| `StrainEnvironmentResponseExperimentReference` | `EnvironmentResponseExperimentReference` | `genome_reference: StrainReferenceGenome`, `environment_reference: CultureEnvironment` |
| `StrainReferenceGenome` | `ReferenceGenome` | `background: StrainBackground` (required) |
| `CultureEnvironment` | `Environment` | `culture_format`, `pre_culture`, `auxotroph_supplements` |

Why subclasses with their own `experiment_type`: pydantic v2 serializes a field by its DECLARED type, so a `CultureEnvironment` placed in an `Environment`-typed slot would lose its fields, and the reconstruction maps (`EXPERIMENT_TYPE_MAP`, used by the neo4j query loader and the deduplicator) need a tag that resolves to the class declaring them. The phenotype is the unchanged `EnvironmentResponsePhenotype`; `isinstance(x, EnvironmentResponseExperiment)` still holds. The coherence test that pinned `environment` to exactly `Environment` now admits this closed list of slot-adding subclasses and still requires the shared `media: Media` field (`test_every_experiment_family_uses_the_shared_environment_class`).

Why the background hangs off the reference genome, not `Genotype`: the screened edit stays the only perturbation, so pooled one-perturbation semantics, perturbation counts and every gene-keyed consumer are untouched; and the reference is stored once per distinct reference, not per record (the LMDB interns the whole reference once its canonical JSON reaches 512 bytes, which a BY4743 reference does: `test_strain_reference_is_interned_in_the_lmdb`; the graph writes it on the `genome` and `experiment reference` nodes). No adapter change was needed: the `genome` node's `serialized_data` and the reference blob carry it (`test_strain_background_genome_node.py`).

`ProvenanceGapMixin` is NOT added to `Genotype`: the mixin's rule is "a gapped field is None", and `Genotype` has no optional field to gap. Gaps live where the field is: `StrainBackground`, `BackgroundAllele`, `ConstructedOrf`, `CultureFormat`, `PreCulture`, `CultureEnvironment`, and the gap-carrying perturbation leaves (`HashableProvenanceGapMixin`, which hashes canonical JSON so `Genotype.__eq__`'s set comparison still works).

### The eight decisions

1. **Strain background.** `StrainBackground(name, reference_strain="S288C", parents, construction, mating_type, ploidy, alleles, provenance)`. Each `BackgroundAllele(systematic_gene_name, gene_name, allele_name, edit, functional, zygosity, cassette, deleted_span, provenance)` is an edit against R64: `AlleleEdit` = `full_deletion` (BY delta0), `partial_deletion` (his3-delta1), `cassette_replacement` (requires `cassette`), `sequence_variant`. Every allele and the background itself is `provenance` (quotes) or a `ProvenanceGap` on `provenance`, never neither; an unmirrored literature-standard allele is ASSERTED with a `deferred_pending_source_review` gap naming its resolver (`BRACHMANN_1998`).
2. **Mating type.** `MatingType` = `a`, `alpha`, `a/alpha` on the background; set or gapped. R64 is MATalpha (`REFERENCE_MATING_TYPE`; MATALPHA1 YCR040W, MATALPHA2 YCR039C in the R64-4-1 GFF), so MATa is a sequence difference at MAT. A haploid cannot be `a/alpha`.
3. **Ploidy and per-locus dosage.** Zygosity per allele (`haploid` / `homozygous` / `heterozygous`, checked against ploidy). `StrainBackground.functional_copies(gene)` gives the unperturbed dose (BY4743: HIS3 0, LYS2 1, MET17 1, LEU2 0, untouched 2). HIP data moves from `EngineeredCopyNumberPerturbation(1 of 2)` to the new `HeterozygousDeletionPerturbation` (#506 point 3): an allele edit with cassette, barcodes, collection, construction, `constructed_orf` and `replaced_allele`; `heterozygous_deletion_functional_copies(background, pert)` derives the dose (HIS3 in BY4743 -> 0; LYS2 -> 1 if the null allele was replaced, 0 if LYS2 was, `None` when `replaced_allele` is unset). It is `state="present"` and NOT a `DeletionPerturbation` subclass, so "every knockout" filters do not count a heterozygote. `EngineeredCopyNumberPerturbation` is unchanged and stays for genuine dosage edits.
4. **Typed gaps.** See placement above; `_require_value_or_gap` enforces "set or gapped" on `mating_type`, background and allele `provenance`, allele `zygosity`, `ConstructedOrf.relation` / `deleted_span`, and `ConditionalAllelePerturbation.allele_class`.
5. **Cassette, barcode, collection, construction.** `BarcodedKanMxDeletionPerturbation` (haploid deletion in a haploid background, homozygous in a diploid one: HOP) gains `cassette` (e.g. `kanMX4`, sourced by `KANMX4_CASSETTE`, Giaever 2014), `downtag_barcode` (`barcode` is the UPTAG or the only tag named), `construction: StrainConstruction(strain_accession, lab, batch, plate, well)` and `constructed_orf`, all nullable and gappable. `HeterozygousDeletionPerturbation` carries the same set. Two records whose construction or constructed ORF differ are different perturbations, so they never merge.
6. **Conditional alleles.** `ConditionalAllelePerturbation(allele_class, allele_name, marker, collection, construction)`; `ConditionalAlleleClass` = `temperature_sensitive`, `damp`, `promoter_replacement`. Unknown class is `allele_class=None` plus a gap (one encoding of unknown, no `unknown` member).
7. **Merged historical ORFs.** `ConstructedOrf(source_systematic_name, relation, deleted_span)` with `OrfHistoryRelation` = `merged` / `reannotated` / `alias`; `systematic_gene_name` stays the current gene for joins. A YAR044W strain served on YAR042W is a different perturbation from the YAR042W strain.
8. **Culture protocol.** `CultureEnvironment.culture_format: CultureFormat(vessel, working_volume_ul, shaking_rpm, inoculum_cells, inoculum_cells_per_strain, inoculum_od600, endpoint: EndpointRule)`; `pre_culture: PreCulture(source: PreCultureSource, medium, generations, duration_hours, od600_at_transfer, source_label)`; `auxotroph_supplements: list[MediaComponent]`. A signed generation count splits into `duration_generations` (magnitude) and `pre_culture.source` (sign: `frozen_stock` for Hillenmeyer `-5gen`, `log_phase_culture` for `5gen`). The VEHICLE gap above solubility is `SmallMoleculePerturbation(solvent=None, provenance_gaps=[gap("solvent")])`; `Solvent` itself is unchanged because 13 served closures contain it. Environment identity adds the three slots only when set, so a `CultureEnvironment` stating none of them joins the plain `Environment` node.

### Blast radius (measured)

- Contract fingerprints, base `main` vs this branch, mapped onto the 51 datasets of served release `2026.10.02-833970cd`: changed `BarcodedKanMxDeletionPerturbation` only; 21 symbols added; 2 of 51 served closures move (`EnvChemgenVanacloig2022Dataset`, `EnvChemgenWildenhain2015Dataset`); the per-dataset drift against the manifest's stored closures is the same set. Script `experiments/036-dataset-fixes-before-kg-build/scripts/strain_background_fingerprint_impact.py`, output `experiments/036-dataset-fixes-before-kg-build/results/strain_background_fingerprint_impact.{json,csv}`.
- Dev stores, `python -m torchcell.provenance.build_manifest`: before (main) 7 stale of 79; after 9 stale. The two new ones are `env_chemgen_wildenhain2015` and one more `env_chemgen_vanacloig2022` build, both on `BarcodedKanMxDeletionPerturbation`. Hoepfner and Hillenmeyer stores go stale only when their loaders import the new classes.

### Worked examples (the calls a loader writes)

Shared imports:

```python
from torchcell.datamodels.schema import (
    BarcodedKanMxDeletionPerturbation, ConditionalAlleleClass, ConditionalAllelePerturbation,
    ConstructedOrf, CultureEnvironment, CultureFormat, EndpointRule, Genotype,
    HeterozygousDeletionPerturbation, MatingType, OrfHistoryRelation, PreCulture,
    PreCultureSource, StrainBackground, StrainConstruction, StrainEnvironmentResponseExperiment,
    StrainEnvironmentResponseExperimentReference, StrainReferenceGenome, Zygosity,
)
from torchcell.datamodels.strain_background import (
    BRACHMANN_1998, GIAEVER_2002, KANMX4_CASSETTE, pending_source_review,
    standard_allele, standard_background,
)
```

The loader's `experiment_class` / `reference_class` properties return the two `StrainEnvironmentResponse*` classes, and every record uses `CultureEnvironment` (with or without protocol fields).

**Wildenhain 2015 (BY4741 haploid, MATa his3Δ1 leu2Δ0 met15Δ0 ura3Δ0 + xxxΔ::kanMX4).** The BY4741 genotype is quoted in the mirrored Sci Data paper, so the background is fully sourced:

```python
BY4741_SV = SourcedValue(
    value="BY4741",
    provenance=Provenance(source_uri="paper.md",
        citation_key="wildenhainSystematicChemicalgeneticChemicalchemical2016",
        sha256="89ff4d9bf1d31719ab15c18ab7aca0b7caf10f55c7c239c1b021908c95439e33"),
    quote="isogenic to BY4741, which has the genotype MATa his3Δ1 leu2Δ0 met15Δ0 ura3Δ0",
)
genome = StrainReferenceGenome(species="Saccharomyces cerevisiae", strain="BY4741",
    ploidy="haploid", background=standard_background("BY4741", provenance=[BY4741_SV]))
deletion = BarcodedKanMxDeletionPerturbation(systematic_gene_name=orf, perturbed_gene_name=sym,
    collection=COLLECTION.value, cassette=KANMX4_CASSETTE.value,
    provenance_gaps=[pending_source_review("barcode", TABLE_1, "barcodes not released")])
# the 33 essential-gene strains (#504): no kanMX null of an essential gene in a haploid
conditional = ConditionalAllelePerturbation(systematic_gene_name=orf, perturbed_gene_name=sym,
    allele_class=None, provenance_gaps=[pending_source_review("allele_class", TABLE_1,
        "essential gene; allele class in Sci Data Table 1, not mirrored")])
env = CultureEnvironment(media=SC, temperature=..., perturbations=[...],
    culture_format=CultureFormat(vessel="96-well plate", working_volume_ul=100.0,
        shaking_rpm=0.0, inoculum_cells=50000.0, endpoint=EndpointRule.until_control_saturation))
```

`TABLE_1` is a `Provenance(source_uri=...)` naming Sci Data Table 1 (not mirrored); each value in `CultureFormat` gets its quote in the loader's `SOURCED_VALUES` (and may also go in `CultureFormat.provenance`).

**Vanacloig 2022 (SGA MATa progeny of Y13206 x BY4741-derived array).** Sourced from Ohnuki 2022 (Y13206 and its parent Y8835) and Piotrowski 2017 (MATa selection); the name must equal `strain`:

```python
OHNUKI = Provenance(source_uri="paper.md", citation_key="ohnukiHighthroughputPlatformYeast2022",
    sha256="de2cad9b33c5e0f7e9ce7b7d56feb17a10dbb83de34f65330d6e35846b757ee1")
PIOTROWSKI = Provenance(source_uri="paper.md", citation_key="piotrowskiFunctionalAnnotationChemical2017",
    sha256="9314a0dd932c1b20b4dc4297de4ce9452b09f314bf100f05f2fe718fb3415fc1")
Y8835 = SourcedValue(value="Y8835", provenance=OHNUKI, quote="its parent strain Y8835 (MATα "
    "ura3Δ0:: natMX4 can 1Δ:: STE2pr-Sp_his5 lyp1Δ his3Δ1 leu2Δ0 met15Δ0 LYS2)")
QUERY = SourcedValue(value="Y13206", provenance=PIOTROWSKI, quote="The MATα pdr1Δ::natMX "
    "pdr3Δ::KI.URA3 snq2Δ::KI.LEU2 (y13206) query strain carried the can1Δ::STEpr-SP_his5 "
    "and lypΔ SGA reporters")
MATA = SourcedValue(value="a", provenance=PIOTROWSKI, quote="to select for the MATa meiotic progeny")
name = "Y13206 x BY4741 SGA MATa progeny"
alleles = [standard_allele(a, Zygosity.haploid, provenance=[Y8835, QUERY])
           for a in ("can1Δ::STE2pr-Sp_his5", "lyp1Δ", "his3Δ1", "leu2Δ0", "ura3Δ0",
                     "met15Δ0", "pdr1Δ::natMX", "pdr3Δ::KlURA3", "snq2Δ::KlLEU2")]
background = StrainBackground(name=name, parents=["Y13206", "BY4741 xxxΔ::kanMX4 array"],
    construction="SGA: MATα Y13206 x MATa deletion array; MATa meiotic progeny selected",
    mating_type=MatingType.a, ploidy="haploid", alleles=alleles, provenance=[MATA, QUERY])
genome = StrainReferenceGenome(species="Saccharomyces cerevisiae", strain=name,
    ploidy="haploid", background=background)
genotype = Genotype(perturbations=[BarcodedKanMxDeletionPerturbation(
    systematic_gene_name=orf, perturbed_gene_name=sym, barcode=uptag,
    collection=LIBRARY_COLLECTION.value, cassette="kanMX")])
```

With the 3Δ alleles in the background, a Vanacloig genotype holds ONE perturbation (it holds four today); the follow-up must confirm that change. A per-allele source list must quote the specific allele; the loader should split `[Y8835, QUERY]` per allele (Y8835 for can1/lyp1/his3/leu2/met15, QUERY for pdr1/pdr3/snq2) and gap `ura3Δ0` / `met15Δ0` if the two quotes disagree with the array parent (the Ohnuki OCR spells Y13206's alleles `met15Δ` and garbles its can1 token).

**Hillenmeyer 2008 (BY4743, het and hom).** BY4743 is named only by the released key file `hom.txt` (sha256 `312f76309547ae2e2870029a2ca1f3580176ba0b97a24308eeb5afe17de9131b`, "synthetic complete for BY4743"); its genotype is not in any mirrored source, so every element is asserted pending Brachmann 1998:

```python
background = standard_background("BY4743", resolve_with=BRACHMANN_1998,
    note="BY4741 x BY4742 genotype not in a mirrored source")
genome = StrainReferenceGenome(species="Saccharomyces cerevisiae", strain="BY4743",
    ploidy="diploid", background=background)
het = HeterozygousDeletionPerturbation(systematic_gene_name=orf, perturbed_gene_name=sym,
    cassette=KANMX4_CASSETTE.value, construction=StrainConstruction(batch="chr4_3"),
    provenance_gaps=[pending_source_review("barcode", GIAEVER_2002)])
# at LYS2 / MET17 the background is heterozygous; state which allele was replaced or gap it
het_lys2 = HeterozygousDeletionPerturbation(systematic_gene_name="YBR115C",
    perturbed_gene_name="LYS2", cassette="kanMX4", replaced_allele=None,
    provenance_gaps=[pending_source_review("replaced_allele", GIAEVER_2002)])
hom = BarcodedKanMxDeletionPerturbation(systematic_gene_name=orf, perturbed_gene_name=sym,
    cassette="kanMX4", construction=StrainConstruction(batch="chr4_3"),
    constructed_orf=ConstructedOrf(source_systematic_name="YAR044W",
        relation=None, deleted_span=None, provenance_gaps=[
            pending_source_review("relation", SGD_HISTORY), pending_source_review("deleted_span", SGD_HISTORY)]))
env = CultureEnvironment(media=YPD_LIQUID, ..., duration_generations=5.0,
    pre_culture=PreCulture(source=PreCultureSource.frozen_stock, source_label="-5gen"))
env_pre = CultureEnvironment(media=YPD_LIQUID, ..., duration_generations=5.0,
    pre_culture=PreCulture(source=PreCultureSource.log_phase_culture, medium=YPD_LIQUID,
        od600_at_transfer=2.0, generations=10.0, source_label="5gen"))
```

`SGD_HISTORY` names the SGD ORF-history record (not mirrored). The SOM's "~10 generations of recovery" sources `generations=10.0`; quote it in `SOURCED_VALUES`. For the minimal-medium arrays: `auxotroph_supplements=None` with a gap (the supplement is implied, never named).

**Hoepfner 2014 (BY4743, HIP YSC1055 and HOP YSC1056).** The paper names the collections ("YSC1055 and YSC1056, OpenBiosystems", paper.md sha256 `a9877549...0aeb`) but never BY4743 for them, so the background name and every allele are pending:

```python
background = standard_background("BY4743", resolve_with=BRACHMANN_1998,
    note="the paper names YSC1055/YSC1056, not BY4743")
hip = HeterozygousDeletionPerturbation(systematic_gene_name=orf, perturbed_gene_name=sym,
    cassette="kanMX4", collection="YSC1055 OpenBiosystems",
    construction=StrainConstruction(lab=row.Lab, batch=row.Batch, plate=row.Plate,
        well=row.Row_Column),
    provenance_gaps=[pending_source_review("barcode", GIAEVER_2002)])
hop = BarcodedKanMxDeletionPerturbation(systematic_gene_name=orf, perturbed_gene_name=sym,
    cassette="kanMX4", collection="YSC1056 OpenBiosystems")
high_dose = SmallMoleculePerturbation(compound=..., concentration=..., solvent=None,
    provenance_gaps=[ProvenanceGap(field="solvent",
        reason=ProvenanceGapReason.not_reported_by_primary,
        note="above 200 uM: concentrated stocks 'if solubility permitted', vehicle unstated")])
env = CultureEnvironment(media=YPD_LIQUID, ..., culture_format=CultureFormat(
    vessel="24-well plate (Greiner 662102)", working_volume_ul=1600.0, shaking_rpm=550.0,
    inoculum_cells_per_strain=250.0, endpoint=EndpointRule.fixed_generations))
```

Table S5 (mirrored, `si/Table_S5.xls`) gives `Lab`, `Batch`, `Plate`, `Row_Column` per HIP strain.

### Open

- Retrievals that would turn pending gaps into sources (each needs a go-ahead): Brachmann 1998 (BY4741/BY4742/BY4743 genotypes and the delta0 / his3-delta1 edit kinds), Giaever 2002 and Winzeler 1999 (YKO construction, hom diploids by mating), Pierce 2007, the Wildenhain strain tables (Sci Data Table 1, Cell Systems Table S3), the Hoepfner Supplementary Material.
- The `BRACHMANN_1998` DOI in `strain_background.py` was written from the citation, not checked against a mirrored copy.
- Hypothesis (untested): `lyp1Δ` is a marker-free full ORF deletion (`STANDARD_ALLELES`); no mirrored source describes its construction.
- The loaders' per-record `n_perturbations` changes for Vanacloig (4 to 1) if the 3Δ alleles move to the background; this is the recommended reading of #500 but is the follow-up's call.
- `genome` node properties are unchanged (`species`, `strain`, `serialized_data`); mating type and ploidy are queryable only through `serialized_data` until a property is added (a served graph-class change).
