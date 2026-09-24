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
    # 4,788 raw rows minus 28 control/NaN rows, 11 unresolvable names (2 ambiguous, 9
    # retired) and 30 ORF collisions. The earlier 4735 predated the layered resolver.
    assert len(dataset) == 4719

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


@pytest.mark.slow
def test_cachera_varying_deletion_carries_the_canonical_common_name(tmp_path):
    """The varying KO stores a common name, not a second copy of the ORF id.

    Issue #195: the CRI-SPA file is common-named, the resolver turns those names into
    systematic ids, and the loader used to store the id in both fields. It now stores the
    genome's own standard name (the L1 ``canonical_gene_names`` spelling shared with the
    Smith, Mormino and Lian loaders), falling back to the id for an ORF with no
    round-tripping standard name. The nine genes a pre-resolver build dropped are the
    check that both halves are present.
    """
    from torchcell.datasets.scerevisiae.cachera2023 import BetaxanthinCachera2023Dataset
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    root = tmp_path / "betaxanthin_cachera2023"
    (root / "raw").mkdir(parents=True)
    shutil.copy(_MIRROR_CSV, root / "raw" / "GA1_2_4_6.csv")

    genome = SCerevisiaeGenome(
        genome_root=_GENOME_DIR, go_root=_GO_DIR, overwrite=False
    )
    dataset = BetaxanthinCachera2023Dataset(root=str(root), genome=genome)

    # (systematic id -> stored common name) for the varying deletion of every record.
    ko_names: dict[str, str] = {}
    for i in range(len(dataset)):
        exp = MetaboliteExperiment.model_validate(dataset[i]["experiment"])
        assert isinstance(exp.genotype, Genotype)
        kos = [
            p
            for p in exp.genotype.perturbations
            if p.perturbation_type == "kanmx_deletion"
        ]
        assert len(kos) == 1
        ko_names[kos[0].systematic_gene_name] = kos[0].perturbed_gene_name

    # The nine genes the pre-resolver build dropped, with the common name the file gives.
    for systematic, common in {
        "YEL024W": "RIP1",
        "YHL011C": "PRS3",
        "YKL148C": "SDH1",
        "YML022W": "APT1",
        "YML100W": "TSL1",
        "YMR205C": "PFK2",
        "YNL129W": "NRK1",
        "YPR047W": "MSF1",
        "YPR128C": "ANT1",
    }.items():
        assert ko_names[systematic] == common

    # Most ORFs have a standard name, so most records must differ between the two fields;
    # an ORF with no round-tripping standard name legitimately stores the id in both.
    n_common = sum(1 for sysn, name in ko_names.items() if name != sysn)
    assert n_common == 3930
    assert len(ko_names) - n_common == 789

    # The spelling comes from the genome, not from the 2023-era source column, so a name
    # SGD has since superseded is stored under its current standard name.
    assert ko_names["YDR511W"] == "SDH7"  # source column says ACN9
    assert ko_names["YAL046C"] == "BOL3"  # source column says AIM1

    # Every (systematic, perturbed) pair stays unique, so the L1 ORF-uniqueness check
    # cannot be collapsed by two source spellings mapping onto one standard name.
    assert len({(s, n) for s, n in ko_names.items()}) == len(dataset)

    # The fixed Btx-cassette keeps the same convention (ARO4 at YBR249C, ARO7 at YPR060C).
    exp0 = MetaboliteExperiment.model_validate(dataset[0]["experiment"])
    assert isinstance(exp0.genotype, Genotype)
    additions = {
        p.systematic_gene_name: p.perturbed_gene_name
        for p in exp0.genotype.perturbations
        if p.perturbation_type == "gene_addition"
    }
    assert additions["YBR249C"] == "ARO4"
    assert additions["YPR060C"] == "ARO7"
