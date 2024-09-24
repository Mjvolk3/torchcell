# torchcell/datamodels/__init__
# [[torchcell.datamodels.__init__]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/__init__.py


from .pydant import ModelStrict, ModelStrictArbitrary

from .schema import (
    Environment,
    Genotype,
    Phenotype,
    Experiment,
    GenePerturbation,
    Media,
    ModelStrict,
    ReferenceGenome,
    Temperature,
    DeletionPerturbation,
    FitnessPhenotype,
    FitnessExperiment,
    DampPerturbation,
    TsAllelePerturbation,
    FitnessExperimentReference,
    ExperimentReference,
    KanMxDeletionPerturbation,
    NatMxDeletionPerturbation,
    SgaKanMxDeletionPerturbation,
    SgaNatMxDeletionPerturbation,
    SgaTsAllelePerturbation,
    SgaDampPerturbation,
    SuppressorAllelePerturbation,
    SgaSuppressorAllelePerturbation,
    AllelePerturbation,
    SgaAllelePerturbation,
    MeanDeletionPerturbation,
    GeneInteractionPhenotype,
    GeneInteractionExperimentReference,
    GeneInteractionExperiment,
    Publication,
    ExperimentType,
    ExperimentReferenceType,
    EXPERIMENT_TYPE_MAP,
    EXPERIMENT_REFERENCE_TYPE_MAP,
)

from .conversion import ConversionEntry, ConversionMap, Converter

# from .gene_essentiality_to_fitness_conversion import GeneEssentialityToFitnessConverter

core_models = ["ModelStrict", "ModelStrictArbitrary"]

schema_classes = [
    "Environment",
    "Genotype",
    "Phenotype",
    "Experiment",
    "GenePerturbation",
    "Media",
    "ReferenceGenome",
    "Temperature",
    "DeletionPerturbation",
    "FitnessPhenotype",
    "FitnessExperiment",
    "DampPerturbation",
    "TsAllelePerturbation",
    "FitnessExperimentReference",
    "ExperimentReference",
    "KanMxDeletionPerturbation",
    "NatMxDeletionPerturbation",
    "SgaKanMxDeletionPerturbation",
    "SgaNatMxDeletionPerturbation",
    "SgaTsAllelePerturbation",
    "SgaDampPerturbation",
    "SuppressorAllelePerturbation",
    "SgaSuppressorAllelePerturbation",
    "AllelePerturbation",
    "SgaAllelePerturbation",
    "MeanDeletionPerturbation",
    "GeneInteractionPhenotype",
    "GeneInteractionExperimentReference",
    "GeneInteractionExperiment",
    "ExperimentType",
    "ExperimentReferenceType",
    "Publication",
]

conversion = ["ConversionEntry", "ConversionMap", "Converter"]

# converters = ["GeneEssentialityToFitnessConverter"]

maps = ["EXPERIMENT_TYPE_MAP", "EXPERIMENT_REFERENCE_TYPE_MAP"]

# __all__ = core_models + schema_classes + conversion + converters
__all__ = core_models + schema_classes + conversion
