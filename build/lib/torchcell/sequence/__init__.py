"""Sequence package: genome, gene, and DNA window data structures and helpers."""

# torchcell/sequence/__init__.py
from .data import DnaSelectionResult as DnaSelectionResult
from .data import DnaWindowResult as DnaWindowResult
from .data import Gene as Gene
from .data import GeneSet as GeneSet
from .data import Genome as Genome
from .data import ParsedGenome as ParsedGenome
from .data import calculate_window_bounds as calculate_window_bounds
from .data import calculate_window_bounds_symmetric as calculate_window_bounds_symmetric
from .data import compute_codon_frequency as compute_codon_frequency
from .data import get_chr_from_description as get_chr_from_description
from .data import mismatch_positions as mismatch_positions
from .data import roman_to_int as roman_to_int

# Curated groupings consumed by the docs: docs/source/modules/sequence.rst iterates
# `data_classes` and `helper_functions` via autosummary, mirroring the pattern in
# torchcell/models/__init__.py (`models`). Keep in sync with the imports above.
data_classes = [
    "DnaSelectionResult",
    "DnaWindowResult",
    "Gene",
    "GeneSet",
    "Genome",
    "ParsedGenome",
]

helper_functions = [
    "calculate_window_bounds",
    "calculate_window_bounds_symmetric",
    "compute_codon_frequency",
    "get_chr_from_description",
    "mismatch_positions",
    "roman_to_int",
]

__all__ = data_classes + helper_functions
