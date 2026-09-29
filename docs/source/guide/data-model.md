# Data model

The unit of data in torchcell is the **experiment record**: one genotype, grown in one
environment, observed to have one phenotype, paired with the reference it is measured
against and the publication it came from. The models live in
`torchcell/datamodels/schema.py`. They are pydantic models built on `ModelStrict`
(`torchcell/datamodels/pydant.py`), which forbids extra fields and freezes instances, so
a record either validates completely or is rejected.

An interactive view of the class hierarchy, generated from the live models at build
time, is the <a href="../ontology/index.html">ontology explorer</a>.

## The experiment record

```text
Experiment
  experiment_type   str          key into EXPERIMENT_TYPE_MAP ("fitness", "gene interaction", ...)
  dataset_name      str          the loader class that produced it, e.g. "SmfCostanzo2016Dataset"
  genotype          Genotype     what was changed in the strain
  environment       Environment  what the strain was grown in
  phenotype         Phenotype    what was measured

ExperimentReference
  experiment_reference_type  str
  dataset_name               str
  genome_reference           ReferenceGenome   species, strain, ploidy
  environment_reference      Environment
  phenotype_reference        Phenotype         the reference value (e.g. wild-type fitness 1.0)

Publication
  pubmed_id, pubmed_url, doi, doi_url           at least one id and one URL are required
```

Each experiment type pairs a subclass of `Experiment` with a subclass of
`ExperimentReference` that narrows `phenotype` to one phenotype class, for example
`FitnessExperiment` and `FitnessExperimentReference` with `FitnessPhenotype`. Two
dictionaries at the end of `schema.py`, `EXPERIMENT_TYPE_MAP` and
`EXPERIMENT_REFERENCE_TYPE_MAP`, map the `experiment_type` string to the class; every
reader that deserializes a record (the dataset loaders, `Neo4jQueryRaw`, the converters)
dispatches through them.

### Genotype

`Genotype.perturbations` is a list of gene perturbations, sorted on validation by
`(systematic_gene_name, perturbation_type, perturbed_gene_name)` so that equal genotypes
serialize identically. Every perturbation extends `GenePerturbation`:

- `systematic_gene_name`: validated against the S288C naming patterns (coding genes
  such as `YAL001C`, mitochondrial `Q0010`-style names, noncoding `YNC...` genes).
- `perturbed_gene_name`: the allele or common name, such as `tfc3_damp`.
- `provenance`: `"engineered"` (made in the lab) or `"natural"` (found in an isolate).

The concrete perturbation classes are organized by what they change relative to the
reference genome: presence or absence of a gene (deletions, marker deletions, gene
additions, natural gene absence and presence), sequence-level alleles (DAmP,
temperature-sensitive, suppressor and sequence-variant alleles), copy number (natural
CNVs and engineered copy number), and expression modulation (CRISPRa and CRISPRi). Each
names its mechanism with a Sequence Ontology term (`mechanism_so_id`,
`mechanism_so_name`).

### Environment

`Environment` holds a `media` (a `Media` resolved into typed components, with `name`,
`state` and `is_synthetic`), an optional `temperature`, a list of environment
`perturbations` (added small molecules, physical factors, biologic agents), an
`aerobicity`, and optional `duration_hours` and `duration_generations`.

### Phenotype

`Phenotype` is the base class of every measured label. It carries three fields that
describe the label rather than its value:

- `graph_level`: where the label sits on a gene graph. Validated against `edge`,
  `node`, `hyperedge`, `subgraph`, `global`, `metabolism` and `gene ontology`.
- `label_name`: the name of the field that holds the value. A validator checks that the
  concrete class declares that field.
- `label_statistic_name`: the name of the field holding its uncertainty or confidence
  statistic, or `None`.

Numeric phenotypes record their uncertainty with its kind: `FitnessPhenotype`, for
example, stores the source-reported number in `fitness_uncertainty`, names it with
`fitness_uncertainty_type` (an `UncertaintyType`: `sample_sd`, `standard_error`,
`bootstrap_se`, `variance`, `ci95`), records the replicate design in `n_samples` and
`sample_unit`, and derives the standard error `fitness_se` from those when it is not
supplied (a bootstrap SE is used as is; a sample SD is divided by the square root of
`n_samples`). A value the
source does not report is declared as a typed `ProvenanceGap` in `provenance_gaps`, and
the field itself must then be `None`.

### Phenotype classes

The defaults below were read from each class's `model_fields` in `schema.py`.
`graph_level` is a default, and a loader may set another value (the digenic-interaction
datasets store `GeneInteractionPhenotype` records at `edge`, the trigenic ones at
`hyperedge`).

| Phenotype class | Value field (`label_name`) | Value type | `graph_level` | `label_statistic_name` |
| :-- | :-- | :-- | :-- | :-- |
| `CalMorphPhenotype` | `calmorph` | `dict[str, float]` | `global` | `calmorph_coefficient_of_variation` |
| `EnvironmentResponsePhenotype` | `environment_response` | `float \| None` | `global` | `environment_response_se` |
| `FitnessPhenotype` | `fitness` | `float` | `global` | `fitness_se` |
| `GeneEssentialityPhenotype` | `is_essential` | `bool` | `node` | `None` |
| `GeneInteractionPhenotype` | `gene_interaction` | `float` | `hyperedge` | `gene_interaction_p_value` |
| `MetabolitePhenotype` | `metabolite_level` | `dict[str, float]` | `metabolism` | `metabolite_level_se` |
| `MicroarrayExpressionPhenotype` | `expression_log2_ratio` | `dict[str, float]` | `node` | `expression_log2_ratio_se` |
| `ProteinAbundancePhenotype` | `protein_abundance` | `dict[str, float]` | `node` | `protein_abundance_se` |
| `PseudobulkExpressionPhenotype` | `expression_log2_ratio` | `dict[str, float]` | `node` | `dispersion` |
| `RNASeqExpressionPhenotype` | `expression_tpm` | `dict[str, float]` | `node` | `None` |
| `SyntheticLethalityPhenotype` | `is_synthetic_lethal` | `bool` | `edge` | `synthetic_lethality_statistic_score` |
| `VisualScorePhenotype` | `visual_score` | `float` | `global` | `None` |

`schema.py` also defines `SyntheticRescuePhenotype` (`is_synthetic_rescue`, `edge`), and
`EXPERIMENT_TYPE_MAP` has a fourteenth entry, `segregant_growth`, whose experiment pairs a
`SegregantGenotype` (a haplotype mosaic of two parent assemblies) with an
`EnvironmentResponsePhenotype`.

## Verification levels

Built datasets are checked record by record against five ordered levels, defined as
`Level` in `torchcell/verification/report.py` and implemented as reusable checks in
`torchcell/verification/levels.py`:

| Level | Name | Checks |
| :-- | :-- | :-- |
| L0 | structural | every record instantiates its schema model |
| L1 | completeness | record counts, known keys and contiguity against an oracle |
| L2 | value fidelity | cross-method agreement, types, ranges, no NaN |
| L3 | semantic | units and conventions (fitness as mutant over wild type; log2 of sample over reference) |
| L4 | cross-source consistency | entities that two sources share agree |

Each check returns a `LevelResult`; a per-dataset verifier (for example
`torchcell/verification/morphology.py`) assembles them into a `VerificationReport`
together with a `Provenance` record of the source.

## Provenance of hand-entered values

A constant that a loader cannot read from a data column (a replicate count, the kind of
an uncertainty, a strain background) is stored as a `SourcedValue`
(`torchcell/verification/sourced.py`): the value, a `Provenance` naming the source file
by citation key and sha256, and a verbatim `quote` from that file that justifies it.
