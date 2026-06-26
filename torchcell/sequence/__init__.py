"""Sequence package: genome, gene, and DNA window data structures and helpers."""

# torchcell/sequence/__init__.py
from .data import (
    DnaSelectionResult,
    DnaWindowResult,
    Gene,
    GeneSet,
    Genome,
    ParsedGenome,
    calculate_window_bounds,
    calculate_window_bounds_symmetric,
    compute_codon_frequency,
    get_chr_from_description,
    mismatch_positions,
    roman_to_int,
)

__all__ = [
    "DnaSelectionResult",
    "DnaWindowResult",
    "Gene",
    "GeneSet",
    "Genome",
    "ParsedGenome",
    "calculate_window_bounds",
    "calculate_window_bounds_symmetric",
    "get_chr_from_description",
    "mismatch_positions",
    "roman_to_int",
    "compute_codon_frequency",
]
