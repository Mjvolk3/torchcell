# tests/torchcell/datasets/scerevisiae/test_ozaydin2013
# [[tests.torchcell.datasets.scerevisiae.test_ozaydin2013]]
"""Build-smoke test for the Ozaydin 2013 beta-carotene visual-screen dataset.

Builds the dataset into a tmp root from the sha256-pinned library-mirror SI (no
network), then asserts record count, schema round-trip, the engineered genotype
(single KO + 3-gene carotenogenic cassette), synthetic media, and publication id.
Skipped when the `$DATA_ROOT` mirror is absent (CI without the data).
"""

import os
import os.path as osp
import shutil
from collections import Counter

import pytest
from dotenv import load_dotenv

from torchcell.datamodels.schema import Genotype, VisualScoreExperiment

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
if DATA_ROOT is None:
    pytest.skip("requires DATA_ROOT data (absent in CI)", allow_module_level=True)

_MIRROR_SI = osp.join(
    DATA_ROOT,
    "torchcell-library/ozaydinCarotenoidbasedPhenotypicScreen2013a/si",
    "1-s2.0-S109671761200081X-mmc1.xlsx",
)

pytestmark = pytest.mark.skipif(
    not osp.exists(_MIRROR_SI),
    reason=f"requires Ozaydin SI mirror at {_MIRROR_SI} (absent in CI)",
)


@pytest.mark.slow
def test_ozaydin_build_smoke(tmp_path):
    """Ozaydin dataset builds from the mirror and produces schema-valid records."""
    from torchcell.datasets.scerevisiae.ozaydin2013 import CarotenoidOzaydin2013Dataset

    root = tmp_path / "carotenoid_ozaydin2013"
    (root / "raw").mkdir(parents=True)
    shutil.copy(_MIRROR_SI, root / "raw" / "1-s2.0-S109671761200081X-mmc1.xlsx")

    dataset = CarotenoidOzaydin2013Dataset(root=str(root))
    assert len(dataset) == 4474

    # dataset[i] returns the stored dicts; validating through the schema IS the
    # round-trip test (the stale on-disk LMDB failed exactly here on media.is_synthetic).
    record = dataset[0]
    exp = VisualScoreExperiment.model_validate(record["experiment"])
    dumped = exp.model_dump()
    assert type(exp).model_validate(dumped).model_dump() == dumped

    # genotype: one KO deletion + the 3-gene YB/I/BTS1 cassette
    assert isinstance(exp.genotype, Genotype)
    ptypes = Counter(p.perturbation_type for p in exp.genotype.perturbations)
    assert ptypes == {"kanmx_deletion": 1, "gene_addition": 3}

    # synthetic medium (the field whose absence made the stale on-disk LMDB fail)
    assert exp.environment.media.is_synthetic is True

    assert record["publication"]["pubmed_id"] == "22918085"
