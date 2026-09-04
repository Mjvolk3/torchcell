# torchcell/datasets/kinetics/__init__.py
# [[torchcell.datasets.kinetics.__init__]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/kinetics/__init__.py

"""One module per kinetic predictor, selected by name.

The registry is the point of the package: a config string picks which predictor feeds the
flux layer, so no model is wired in as the one true source of a turnover number. Adding a
predictor is one module plus one line here.
"""

from torchcell.datasets.kinetics.boost_km import BoostKmDataset, BoostKmEsm1bDataset
from torchcell.datasets.kinetics.deepenzyme import DeepEnzymeDataset
from torchcell.datasets.kinetics.dlkcat import DlkcatDataset
from torchcell.datasets.kinetics.eitlem import EitlemDataset
from torchcell.datasets.kinetics.turnup import TurnupDataset
from torchcell.datasets.kinetics.unikp import UniKPDataset

KINETICS_CONFIGS = {
    "dlkcat": DlkcatDataset,
    "unikp": UniKPDataset,
    "deepenzyme": DeepEnzymeDataset,
    "eitlem": EitlemDataset,
    "turnup": TurnupDataset,
    # The ESM-1b variant, because that is the one Wu et al. ran; see boost_km.py.
    "boost_km": BoostKmEsm1bDataset,
}

__all__ = [
    "BoostKmDataset",
    "BoostKmEsm1bDataset",
    "DeepEnzymeDataset",
    "DlkcatDataset",
    "EitlemDataset",
    "TurnupDataset",
    "UniKPDataset",
    "KINETICS_CONFIGS",
]
