# torchcell/datamodels/genome_architecture_features.py
# [[torchcell.datamodels.genome_architecture_features]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/genome_architecture_features.py
# Test file: tests/torchcell/datamodels/test_genome_architecture_features.py
"""Vocabularies for a whole-genome-sequencing GENOME-ARCHITECTURE profile.

One record of :class:`~torchcell.datamodels.schema.GenomeArchitecturePhenotype` holds
several numbers measured off one sequenced sample: the copy number of a tandem repeat
array, the copy number of an extrachromosomal replicon, the copy number of each
chromosome, a telomere-length proxy, and a count of chromosomal rearrangements. Those
numbers are NOT in one unit and must never be pooled: 125 rDNA repeats per haploid
genome, 17.5 mtDNA copies per cell, 2.0 copies of chromosome IV and 756 telomeric reads
per million are four different quantities.

So the unit basis is a property of the FEATURE, declared here, exactly as
``morphology_features`` makes the statistic a property of the feature. A
record names the assay it was measured by, that assay's symbols are the only permitted
keys, and each symbol says what its number means and which genomic region it was read
from. A later sequencing screen registers its own assay rather than widening this one:
``puddu2019_wgs_coverage`` keys on the exact column headers of Puddu 2019's
Supplementary Table 3, which is the release this vocabulary transcribes.

Design + measurements: ``[[torchcell.datasets.scerevisiae.puddu2019]]``.
"""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "GENOME_ARCHITECTURE_ASSAYS",
    "PUDDU2019_WGS_COVERAGE",
    "ArchitectureBasis",
    "ArchitectureAssay",
    "ArchitectureFeature",
    "architecture_assay",
]


class ArchitectureBasis(StrEnum):
    """WHAT one genome-architecture number is, so two of them are never pooled.

    - ``copies_per_haploid_genome``: coverage over the region divided by the
      genome-wide median, scaled to the haploid genome (a tandem repeat array, a
      transposon family).
    - ``copies_per_cell``: the same ratio multiplied by the sample's expected ploidy,
      which is the right basis for an extrachromosomal replicon (mtDNA, a plasmid).
    - ``chromosome_copies_per_cell``: the copy number of ONE chromosome, 2.0 in an
      unperturbed diploid; a non-integer value is real (it reads as a mixed culture),
      so it is a float and not rounded.
    - ``reads_per_million``: a read-count proxy with no copy-number interpretation at
      all (telomere length as the abundance of telomeric reads).
    - ``ploidy_deviation_units``: a SUM of per-chromosome deviations from the expected
      ploidy, in chromosome units; 0 means every chromosome is at the expected copy
      number. It is a derived aggregate of the per-chromosome features, which is why a
      record carrying it also carries them.
    - ``rearrangement_count``: a non-negative INTEGER count of called events.
    """

    copies_per_haploid_genome = "copies_per_haploid_genome"
    copies_per_cell = "copies_per_cell"
    chromosome_copies_per_cell = "chromosome_copies_per_cell"
    reads_per_million = "reads_per_million"
    ploidy_deviation_units = "ploidy_deviation_units"
    rearrangement_count = "rearrangement_count"


#: Bases whose values are counts of events, so a stored value must be integral.
INTEGRAL_BASES: frozenset[ArchitectureBasis] = frozenset(
    {ArchitectureBasis.rearrangement_count}
)

#: Every basis is a non-negative quantity: a copy number, a read share and an absolute
#: deviation are all bounded below by zero, and none of them has an upper bound a
#: definition supplies (a strain can carry 250 rDNA repeats or 385 plasmid copies).
BASIS_LOWER_BOUND: float = 0.0


class ArchitectureFeature(BaseModel):
    """One measurable element of genome architecture, as one assay names it.

    ``symbol`` is the release's own column header, verbatim, so a stored key is
    comparable to the released bytes without a translation table. ``region`` is the
    genomic interval the number was read from, verbatim from the source's Methods, and
    is ``None`` only for a feature that is not read from an interval (an aggregate).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    name: str
    basis: ArchitectureBasis
    region: str | None = Field(
        default=None, description="the measured interval, verbatim from the Methods"
    )
    note: str | None = None

    @model_validator(mode="after")
    def _check_feature(self) -> ArchitectureFeature:
        """A feature has a symbol and a human name."""
        if not self.symbol.strip() or not self.name.strip():
            raise ValueError("an ArchitectureFeature needs a symbol and a name")
        return self


class ArchitectureAssay(BaseModel):
    """The feature vocabulary of one genome-architecture assay.

    ``name`` is the key a record stores; ``platform`` and ``reference_assembly`` are the
    measurement's own context, since a coverage ratio is only interpretable against the
    assembly the reads were mapped to.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    platform: str
    reference_assembly: str
    features: tuple[ArchitectureFeature, ...]

    @model_validator(mode="after")
    def _check_assay(self) -> ArchitectureAssay:
        """Symbols are distinct and non-empty."""
        symbols = [feature.symbol for feature in self.features]
        if not symbols:
            raise ValueError(f"assay {self.name!r} declares no features")
        if len(set(symbols)) != len(symbols):
            raise ValueError(f"assay {self.name!r} repeats a feature symbol")
        return self

    @property
    def by_symbol(self) -> dict[str, ArchitectureFeature]:
        """``{symbol: feature}`` of this assay."""
        return {feature.symbol: feature for feature in self.features}

    def symbols_with_basis(self, basis: ArchitectureBasis) -> frozenset[str]:
        """The symbols of this assay whose number has ``basis``."""
        return frozenset(f.symbol for f in self.features if f.basis is basis)

    def validate_value(self, symbol: str, value: float) -> None:
        """Raise when ``value`` is impossible for ``symbol``'s declared basis."""
        feature = self.by_symbol[symbol]
        if not math.isfinite(value) or value < BASIS_LOWER_BOUND:
            raise ValueError(
                f"{symbol} is a {feature.basis.value} and must be a finite value "
                f">= {BASIS_LOWER_BOUND}, got {value}"
            )
        if feature.basis in INTEGRAL_BASES and float(value) != int(value):
            raise ValueError(
                f"{symbol} is a {feature.basis.value} and must be integral, got {value}"
            )


_CHROMOSOME_NUMERAL: dict[str, str] = {
    "chr01": "I",
    "chr02": "II",
    "chr03": "III",
    "chr04": "IV",
    "chr05": "V",
    "chr06": "VI",
    "chr07": "VII",
    "chr08": "VIII",
    "chr09": "IX",
    "chr10": "X",
    "chr11": "XI",
    "chr12": "XII",
    "chr13": "XIII",
    "chr14": "XIV",
    "chr15": "XV",
    "chr16": "XVI",
}

#: Ty regions, verbatim from Puddu 2019's Methods ("Quantification of Ty1-Ty5, 2u and
#: definition of genetic mating type"): the artificial-reference chromosome each Ty
#: family was mapped to, and the window the median coverage was taken over.
_TY_REGIONS: dict[str, str] = {
    "Ty1": "YDRWTy1-5 = chromosome IV: 1203704-1215621, window 4000-6999",
    "Ty2": "YLRWTy2-1 = chromosome XII: 938192-950150, window 4000-6999",
    "Ty3": "YILWTy3-1 = chromosome IX: 202220-213647, window 4000-6999",
    "Ty4": "YHLWTy4-1 = chromosome VIII: 82539-94761, window 4000-6999",
    "Ty5": "YCLWTy5-1 = chromosome III: 679-7322, window 2000-2999",
}

PUDDU2019_WGS_COVERAGE = ArchitectureAssay(
    name="puddu2019_wgs_coverage",
    platform="Illumina HiSeqX whole-genome sequencing, copy number from read coverage",
    reference_assembly="S288c_R64-1-1_Ensembl",
    features=(
        ArchitectureFeature(
            symbol="rDNA",
            name="ribosomal DNA repeat copy number",
            basis=ArchitectureBasis.copies_per_haploid_genome,
            region="chromosome XII:452000-459000",
            note="coverage over the region / genome-wide median x 2 (the reference "
            "carries two copies of the minimal repeat unit), then corrected for the "
            "observed ploidy of chromosome XII",
        ),
        ArchitectureFeature(
            symbol="CUP1",
            name="CUP1 tandem-repeat copy number",
            basis=ArchitectureBasis.copies_per_haploid_genome,
            region="chromosome VIII: 212986-213525",
            note="same ratio as rDNA, corrected for the observed ploidy of "
            "chromosome VIII",
        ),
        ArchitectureFeature(
            symbol="mitochondria",
            name="mitochondrial DNA copy number",
            basis=ArchitectureBasis.copies_per_cell,
            region="mito: 14000-20000 (a region of COX1)",
            note="coverage / genome-wide median x the sample's expected ploidy; a "
            "colony below one copy is the paper's rho0 class",
        ),
        ArchitectureFeature(
            symbol="2-micron",
            name="2 micron plasmid copy number",
            basis=ArchitectureBasis.copies_per_cell,
            region="2u = [01347.1, window 2000-4500",
            note="per-haploid ratio multiplied by the expected ploidy, as for mtDNA",
        ),
        *(
            ArchitectureFeature(
                symbol=symbol,
                name=f"{symbol} transposon family copy number",
                basis=ArchitectureBasis.copies_per_haploid_genome,
                region=region,
            )
            for symbol, region in _TY_REGIONS.items()
        ),
        ArchitectureFeature(
            symbol="Telomeres",
            name="telomere length proxy",
            basis=ArchitectureBasis.reads_per_million,
            region="reads carrying at least 5 telomeric repeat units",
            note="telomeric reads in both orientations and read pairs, normalized by "
            "the total number of reads; an abundance, NOT a length in base pairs",
        ),
        *(
            ArchitectureFeature(
                symbol=symbol,
                name=f"chromosome {numeral} copy number",
                basis=ArchitectureBasis.chromosome_copies_per_cell,
                region=f"10-kb window centred on the centromere of chromosome {numeral}",
                note="coverage / genome-wide median x 2; an estimate within 0.2 of an "
                "integer was rounded to it, every other value kept as a rational "
                "number (a fractional value reads as a mixed culture)",
            )
            for symbol, numeral in _CHROMOSOME_NUMERAL.items()
        ),
        ArchitectureFeature(
            symbol="ANtot",
            name="total deviation from the expected ploidy",
            basis=ArchitectureBasis.ploidy_deviation_units,
            region=None,
            note="sum over the 16 chromosomes of |copy number - 2|, in chromosome units",
        ),
        ArchitectureFeature(
            symbol="GCR",
            name="chromosomal rearrangements detected",
            basis=ArchitectureBasis.rearrangement_count,
            region=None,
            note="regions longer than 20 kb whose ploidy differs from the ploidy at "
            "their chromosome's centromere by more than 0.5 units",
        ),
    ),
)
"""Puddu et al. 2019's whole-genome-sequencing screen of the YKOC, keyed by the column
headers of its Supplementary Table 3."""

#: Registered assays. A new sequencing screen adds its own entry; nothing here is edited
#: in place, since a record stores the assay name and its keys are read back through it.
GENOME_ARCHITECTURE_ASSAYS: dict[str, ArchitectureAssay] = {
    PUDDU2019_WGS_COVERAGE.name: PUDDU2019_WGS_COVERAGE
}


def architecture_assay(name: str) -> ArchitectureAssay:
    """The registered assay ``name``; an unregistered name raises."""
    try:
        return GENOME_ARCHITECTURE_ASSAYS[name]
    except KeyError:
        raise ValueError(
            f"unknown genome-architecture assay {name!r}; registered: "
            f"{sorted(GENOME_ARCHITECTURE_ASSAYS)}"
        ) from None
