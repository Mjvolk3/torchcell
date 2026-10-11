# tests/torchcell/datasets/scerevisiae/test_peter2018.py
# [[tests.torchcell.datasets.scerevisiae.test_peter2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_peter2018.py
"""Peter 2018 loader against the mirrors it was built from.

The synthetic end-to-end build lives in ``test_peter2018_synthetic.py``. These tests read
the real bytes and are skipped where a mirror is not on disk: the release in the raw
mirror (971 x 35, 197 zeros, its first cell 0.518089736, md5 equal to the host's
``md5.txt``), the quotes against the library mirror's ``paper.md``, ``si/si2.md`` and
Supplementary Table 2, and the genomes-tier records of the two loss-of-function
matrices.
"""

from __future__ import annotations

import os
import os.path as osp
from pathlib import Path

import pytest

from torchcell.datasets.scerevisiae import peter2018 as p
from torchcell.verification.natural_isolate_response import (
    _l3_conditions_documented,
    _l3_sourced_values,
)


def _data_root() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ.get("DATA_ROOT", "")


_RAW = osp.join(_data_root(), p.RAW_DIR_REL)
_LIBRARY = osp.join(_data_root(), "torchcell-library")
_TIER = osp.join(_data_root(), "torchcell-genomes", "peter2018_1011_assemblies")


@pytest.mark.data
@pytest.mark.skipif(
    not osp.isfile(osp.join(_RAW, "manifest.json")),
    reason="Peter 2018 raw mirror not on disk",
)
def test_the_release_is_the_pinned_971_by_35_matrix() -> None:
    manifest = p.load_manifest(_data_root())
    assert p.manifest_sha256(manifest, p.PHENO_REL) == p.PHENO_SHA256
    path = Path(_RAW) / p.PHENO_REL
    assert p.md5_file(path) == p.PHENO_PUBLISHED_MD5
    assert p.read_published_md5(Path(_RAW) / p.MD5_REL)[p.PHENO_NAME] == (
        p.PHENO_PUBLISHED_MD5
    )
    frame = p.read_release(path, p.build_conditions())
    assert frame.shape == (971, 35)
    assert int((frame.to_numpy() == 0).sum()) == 197
    assert frame.at["AAA", "YPACETATE"] == 0.518089736
    assert sum(code.startswith("SACE_") for code in frame.index) == 65
    assert frame.size + p.N_ISOLATES == p.PAPER_MEASUREMENTS


@pytest.mark.data
@pytest.mark.skipif(
    not osp.isfile(osp.join(_LIBRARY, p.CITATION_KEY, p.SI_TABLES_XLS)),
    reason="Peter 2018 library mirror not on disk",
)
def test_every_quote_and_table_s2_row_reads_from_the_mirror() -> None:
    sourced = _l3_sourced_values(Path(_LIBRARY))
    assert (sourced.passed, sourced.details["failures"]) == (True, [])
    conditions = p.build_conditions()
    from collections import Counter

    documented = _l3_conditions_documented(
        Counter(dict.fromkeys(conditions, 971)),
        0,
        conditions,
        Path(_LIBRARY) / p.CITATION_KEY / p.SI_TABLES_XLS,
    )
    assert (documented.passed, documented.details["table_s2_mismatches"]) == (True, [])


@pytest.mark.data
@pytest.mark.skipif(
    not osp.isfile(osp.join(_TIER, "manifest.json")),
    reason="Peter genomes tier not on disk",
)
def test_the_loss_of_function_matrices_are_registered_in_the_tier() -> None:
    from torchcell.sequence.genome.registry import PETER2018_1011, load_genome_manifest

    manifest = load_genome_manifest(PETER2018_1011, _data_root())
    for name in (p.LOF_NAME, p.FRAMESHIFT_NAME):
        record = manifest.record(name)
        assert record.role == "matrix"
        assert record.source == p.DATA_HOST + name
        assert p.md5_file(osp.join(_TIER, name)) == p.LOF_PUBLISHED_MD5[name]
    assert manifest.record(p.LOF_NAME).sha256 == (
        "067327e1970c48f278dd08abb18f140622458829a6abc320207245c8c4b61c34"
    )
    assert manifest.record(p.FRAMESHIFT_NAME).sha256 == (
        "110debdf846e48ff71d21c69ec66b1dfde0239be3e199ba432544d218634bee5"
    )


def test_quotes_carry_the_pinned_sources() -> None:
    """Every phenotype SourcedValue is pinned to paper.md's sha256 at a stated line."""
    for sv in p.PHENOTYPE_SOURCED_VALUES.values():
        assert sv.provenance.citation_key == p.CITATION_KEY
        assert sv.provenance.sha256 == p.PAPER_MD_SHA256
        assert sv.provenance.page in {"paper.md line 250", "paper.md line 292"}
    assert p.PHENOTYPE_SOURCED_VALUES["n_samples"].value == p.N_SAMPLES == 4
    statement = p.PETER_YPD.provenance[-1]
    assert (statement.quote, statement.provenance.page) == (
        p.YPD_CELL,
        "sheet 'Table S2' row 5",
    )
