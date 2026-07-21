# torchcell/sga/__init__.py
# [[torchcell.sga]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sga
"""SGAtools-style colony-fitness pipeline for ECHO-arrayed CRISPR knockouts.

An adaptation of the Boone-lab SGAtools normalization & scoring stage (Wagih
et al. 2013; math from Baryshnikova et al. 2010) to a non-SGA design: single
CRISPR-Cas knockouts, acoustic (ECHO) dispensing, randomized plate positions,
an on-plate BY4741 wild-type reference and a Blank_media no-cell control. We
consume the same gitter DAT SGAtools does but redefine "score" as fitness
relative to the on-plate wild-type (there is no query x array cross).

Typical use:
    from torchcell.sga import (
        read_gitter_dat, read_echo_picklist, merge_layout,
        normalize_plate, score_plate, NormalizationConfig,
    )
    dat = read_gitter_dat("plate.dat")
    layout = read_echo_picklist("picklist.csv")
    df = normalize_plate(merge_layout(dat, layout))
    report = score_plate(df, plate_id="plate")
"""

from torchcell.sga.assay import (
    recommend_volume,
    volume_assay_metrics,
    volume_position_confound,
    zfactor,
)
from torchcell.sga.image import quantify_plate_image
from torchcell.sga.io import (
    merge_layout,
    read_echo_picklist,
    read_gitter_dat,
    well_to_rowcol,
)
from torchcell.sga.models import NormalizationConfig, ScoreReport, StrainScore
from torchcell.sga.normalize import normalize_plate
from torchcell.sga.register import resolve_orientation
from torchcell.sga.score import score_plate, score_table

__all__ = [
    "read_gitter_dat",
    "read_echo_picklist",
    "merge_layout",
    "well_to_rowcol",
    "NormalizationConfig",
    "ScoreReport",
    "StrainScore",
    "normalize_plate",
    "score_plate",
    "score_table",
    "quantify_plate_image",
    "resolve_orientation",
    "volume_assay_metrics",
    "volume_position_confound",
    "recommend_volume",
    "zfactor",
]
