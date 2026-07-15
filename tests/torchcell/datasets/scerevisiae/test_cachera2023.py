# tests/torchcell/datasets/scerevisiae/test_cachera2023
# [[tests.torchcell.datasets.scerevisiae.test_cachera2023]]
"""Build-smoke test for the Cachera 2023 betaxanthin CRI-SPA dataset.

Builds the dataset into a tmp root from the sha256-pinned library-mirror CSV (no
network), injecting a real ``SCerevisiaeGenome`` to resolve common gene names, then
asserts record count, schema round-trip, the engineered genotype (single KO + 4-gene
Btx-cassette), synthetic media, and publication id. Skipped when the ``$DATA_ROOT``
mirror or the SGD genome is absent (CI without the data).
"""

import os
import os.path as osp
import shutil
from collections import Counter

import pytest
from dotenv import load_dotenv

from torchcell.datamodels.schema import Genotype, MetaboliteExperiment

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
if DATA_ROOT is None:
    pytest.skip("requires DATA_ROOT data (absent in CI)", allow_module_level=True)

_MIRROR_CSV = osp.join(
    DATA_ROOT,
    "torchcell-library/cacheraCRISPAHighthroughputMethod2023/si",
    "GA1_2_4_6.csv",
)
_GENOME_DIR = osp.join(DATA_ROOT, "data/sgd/genome")
_GO_DIR = osp.join(DATA_ROOT, "data/go")

pytestmark = pytest.mark.skipif(
    not (osp.exists(_MIRROR_CSV) and osp.isdir(_GENOME_DIR) and osp.isdir(_GO_DIR)),
    reason="requires Cachera CSV mirror + SGD genome at $DATA_ROOT (absent in CI)",
)


@pytest.mark.slow
def test_cachera_build_smoke(tmp_path):
    """Cachera dataset builds from the mirror + genome and yields schema-valid records."""
    from torchcell.datasets.scerevisiae.cachera2023 import BetaxanthinCachera2023Dataset
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    root = tmp_path / "betaxanthin_cachera2023"
    (root / "raw").mkdir(parents=True)
    shutil.copy(_MIRROR_CSV, root / "raw" / "GA1_2_4_6.csv")

    genome = SCerevisiaeGenome(
        genome_root=_GENOME_DIR, go_root=_GO_DIR, overwrite=False
    )
    dataset = BetaxanthinCachera2023Dataset(root=str(root), genome=genome)
    assert len(dataset) == 4735

    # dataset[i] returns the stored dicts; validating through the schema IS the
    # round-trip test (the stale on-disk LMDB failed exactly here on media.is_synthetic).
    record = dataset[0]
    exp = MetaboliteExperiment.model_validate(record["experiment"])
    dumped = exp.model_dump()
    assert type(exp).model_validate(dumped).model_dump() == dumped

    # genotype: one KO deletion + the 4-gene Btx-cassette (natMX marker omitted)
    assert isinstance(exp.genotype, Genotype)
    ptypes = Counter(p.perturbation_type for p in exp.genotype.perturbations)
    assert ptypes == {"kanmx_deletion": 1, "gene_addition": 4}

    # synthetic medium (the field whose absence made the stale on-disk LMDB fail)
    assert exp.environment.media.is_synthetic is True

    assert record["publication"]["pubmed_id"] == "37572348"
