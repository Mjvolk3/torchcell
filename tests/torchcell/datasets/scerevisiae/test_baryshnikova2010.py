# tests/torchcell/datasets/scerevisiae/test_baryshnikova2010.py
"""Baryshnikova 2010 loader: allele parsing, the temperature split, the n_samples split.

The synthetic tests need no data on disk. The mirror-backed tests re-open the pinned
artifacts: one audits every ``SourcedValue`` (sha256 + verbatim quote), the other reads
the publisher's Supplementary Data 1 and checks the release facts the loader asserts.
"""

from __future__ import annotations

import os
import os.path as osp
from pathlib import Path

import pandas as pd
import pytest

from torchcell.datamodels.media import SGA_DM_SELECTION
from torchcell.datamodels.schema import (
    SampleUnit,
    SgaDampPerturbation,
    SgaKanMxDeletionPerturbation,
    SgaTsAllelePerturbation,
    UncertaintyType,
)
from torchcell.datasets.scerevisiae import baryshnikova2010 as b
from torchcell.verification.sourced import ProvenanceGapReason, audit_sourced_value


def _dataset() -> b.SmfBaryshnikova2010Dataset:
    """A loader instance with only the attribute the record builders read."""
    dataset = object.__new__(b.SmfBaryshnikova2010Dataset)
    dataset.name = "SmfBaryshnikova2010Dataset"
    return dataset


def _temperature(kind: str) -> float:
    """The environment temperature of one allele kind, asserted present."""
    temperature = b.SmfBaryshnikova2010Dataset.environment(kind).temperature
    assert temperature is not None
    return temperature.value


def test_parse_allele_splits_the_three_released_kinds() -> None:
    assert b._parse_allele("YML062C") == ("YML062C", "deletion")
    assert b._parse_allele("YAL001C_damp") == ("YAL001C", "damp")
    assert b._parse_allele("YBR156C_tsq236") == ("YBR156C", "ts")
    # a TS id keeps only the ORF token, never the allele serial
    assert b._parse_allele("YAL041W_tsq148")[0] == "YAL041W"


def test_perturbation_leaf_per_kind_carries_the_raw_id_on_strain_id() -> None:
    build = b.SmfBaryshnikova2010Dataset._perturbation
    deletion = build("deletion", "YML062C", "YML062C")
    damp = build("damp", "YAL001C", "YAL001C_damp")
    ts = build("ts", "YBR156C", "YBR156C_tsq236")
    assert isinstance(deletion, SgaKanMxDeletionPerturbation)
    assert isinstance(damp, SgaDampPerturbation)
    assert isinstance(ts, SgaTsAllelePerturbation)
    assert ts.strain_id == "YBR156C_tsq236"
    # the release carries no common names, so the ORF is both names
    assert ts.systematic_gene_name == ts.perturbed_gene_name == "YBR156C"


def test_temperature_is_split_by_allele_kind_on_one_shared_medium() -> None:
    environment = b.SmfBaryshnikova2010Dataset.environment
    assert _temperature("deletion") == 30.0
    assert _temperature("damp") == 30.0
    assert _temperature("ts") == 26.0
    for kind in ("deletion", "damp", "ts"):
        assert environment(kind).media == SGA_DM_SELECTION
        assert environment(kind).media.state == "solid"
    # the two halves are sourced to different mirrored papers
    assert b._SV_TEMPERATURE_30.provenance.citation_key == b._TONG2006
    assert b._SV_TEMPERATURE_26.provenance.citation_key == b._COSTANZO2016


def test_n_samples_is_stored_only_where_the_si_sources_it() -> None:
    dataset = _dataset()
    deletion = dataset._phenotype("deletion", 0.98, 0.005)
    assert deletion.n_samples == 80 and deletion.provenance_gaps == []
    assert deletion.sample_unit is SampleUnit.screen
    # bootstrap SE is already an SE: used as-is, never divided by sqrt(n)
    assert deletion.fitness_uncertainty_type is UncertaintyType.bootstrap_se
    assert deletion.fitness_se == 0.005
    for kind in ("damp", "ts"):
        query_side = dataset._phenotype(kind, 0.75, 0.02)
        assert query_side.n_samples is None
        assert [g.field for g in query_side.provenance_gaps] == ["n_samples"]
        gap = query_side.provenance_gaps[0]
        assert gap.reason is ProvenanceGapReason.not_reported_by_primary
        assert gap.looked_in is not None
        assert gap.looked_in.citation_key == b.CITATION_KEY
        # the SE is still stored; only the replicate count is absent
        assert query_side.fitness_se == 0.02


def test_reference_is_the_normalization_convention_not_a_measurement() -> None:
    reference = _dataset()._reference("ts")
    phenotype = reference.phenotype_reference
    assert phenotype.fitness == 1.0
    assert sorted(g.field for g in phenotype.provenance_gaps) == [
        "n_samples",
        "sample_unit",
    ]
    assert phenotype.n_samples is None and phenotype.sample_unit is None
    assert reference.genome_reference.strain == "BY4741"
    # the reference environment follows its allele kind, so a TS reference is 26 C
    environment_reference = reference.environment_reference
    assert environment_reference.temperature is not None
    assert environment_reference.temperature.value == 26.0


def test_record_count_oracle_is_derived_from_the_release() -> None:
    assert b._EXPECTED_ROWS == 6023
    assert len(b._UNRESOLVABLE) == 30
    assert b.EXPECTED_RECORDS == b._EXPECTED_ROWS - len(b._UNRESOLVABLE) == 5993
    # the frozen set is 29 deletions + the one DAmP allele
    kinds = sorted(b._UNRESOLVABLE.values())
    assert kinds.count("deletion") == 29 and kinds.count("damp") == 1
    assert b._SV_COMPOSITION.value == {"deletion": 4635, "damp": 1082, "ts": 306}
    assert sum(b._SV_COMPOSITION.value.values()) == b._EXPECTED_ROWS


def test_read_smf_rejects_a_drifted_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = _dataset()
    monkeypatch.setattr(
        type(dataset), "raw_dir", property(lambda self: str(tmp_path)), raising=False
    )
    frame = pd.DataFrame(
        {"allele": ["YML062C", "YAL001C_damp"], "fitness": [1.0, 0.9], "se": [0.1, 0.1]}
    )
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: frame)
    with pytest.raises(RuntimeError, match="row-count self-checksum"):
        dataset._read_smf()

    padded = pd.concat([frame] * 1, ignore_index=True)
    padded = pd.concat(
        [padded]
        + [
            pd.DataFrame(
                {
                    "allele": [f"Y{i:06d}W" for i in range(b._EXPECTED_ROWS - 2)],
                    "fitness": [1.0] * (b._EXPECTED_ROWS - 2),
                    "se": [0.1] * (b._EXPECTED_ROWS - 2),
                }
            )
        ],
        ignore_index=True,
    )
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: padded)
    with pytest.raises(RuntimeError, match="composition self-checksum"):
        dataset._read_smf()


def _library_root() -> str | None:
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ.get("DATA_ROOT")
    if not data_root:
        return None
    root = osp.join(data_root, "torchcell-library")
    return root if osp.isdir(root) else None


@pytest.mark.skipif(
    _library_root() is None, reason="torchcell-library mirror not on disk"
)
def test_every_sourced_value_audits_against_its_pinned_artifact() -> None:
    root = _library_root()
    assert root is not None
    failures = []
    for sourced in b.SOURCED_VALUES:
        result = audit_sourced_value(sourced, root)
        if not result.passed:
            failures.append((sourced.provenance.source_uri, result.message))
    assert not failures, failures
    # the chain reaches all three mirrored papers
    assert {s.provenance.citation_key for s in b.SOURCED_VALUES} == {
        b.CITATION_KEY,
        b._TONG2006,
        b._COSTANZO2016,
    }


def _mirror_root() -> str | None:
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ.get("DATA_ROOT")
    if not data_root:
        return None
    root = osp.join(data_root, b.RAW_DIR_REL)
    return root if osp.exists(osp.join(root, "manifest.json")) else None


@pytest.mark.skipif(
    _mirror_root() is None, reason="Baryshnikova 2010 raw mirror not on disk"
)
def test_raw_mirror_manifest_pins_the_publisher_supplement() -> None:
    root = _mirror_root()
    assert root is not None
    manifest = b.load_manifest(osp.dirname(osp.dirname(root)))
    assert b.manifest_sha256(manifest, b.XLS_REL) == b.XLS_SHA256
    record = next(f for f in manifest.files if f.path == b.XLS_REL)
    assert record.retrieval is not None
    assert record.retrieval.method.value == "springer_esm"
    assert record.retrieval.source_url == b.XLS_URL
    assert b._sha256(osp.join(root, b.XLS_REL)) == b.XLS_SHA256


@pytest.mark.skipif(
    _mirror_root() is None, reason="Baryshnikova 2010 raw mirror not on disk"
)
def test_released_supplement_matches_the_frozen_release_facts() -> None:
    root = _mirror_root()
    assert root is not None
    frame = pd.read_excel(
        osp.join(root, b.XLS_REL),
        sheet_name=b.XLS_SHEET,
        header=None,
        names=["allele", "fitness", "se"],
    )
    assert len(frame) == b._EXPECTED_ROWS
    kinds = frame["allele"].map(lambda a: b._parse_allele(str(a))[1])
    assert kinds.value_counts().to_dict() == dict(b._SV_COMPOSITION.value)
    # YDL227C (HO, the neutral SGA marker locus) is the one repeated id, with two
    # different fitness values that the loader keeps as two records
    duplicated = sorted(set(frame["allele"][frame["allele"].duplicated(keep=False)]))
    assert duplicated == ["YDL227C"]
    assert frame.loc[frame["allele"] == "YDL227C", "fitness"].nunique() == 2
