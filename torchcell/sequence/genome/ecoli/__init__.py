"""E. coli genome implementations (K-12 MG1655 and BW25113)."""

from .k12 import (
    BW25113_ASSEMBLY,
    MG1655_ASSEMBLY,
    DerivedGoAnnotation,
    EckCrosswalk,
    EckPair,
    EcoliK12BW25113Genome,
    EcoliK12Gene,
    EcoliK12Genome,
    EcoliK12MG1655Genome,
    eck_crosswalk,
)

__all__ = [
    "BW25113_ASSEMBLY",
    "MG1655_ASSEMBLY",
    "DerivedGoAnnotation",
    "EckCrosswalk",
    "EckPair",
    "EcoliK12BW25113Genome",
    "EcoliK12Gene",
    "EcoliK12Genome",
    "EcoliK12MG1655Genome",
    "eck_crosswalk",
]
