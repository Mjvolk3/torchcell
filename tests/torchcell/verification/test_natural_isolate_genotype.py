# tests/torchcell/verification/test_natural_isolate_genotype.py
# [[tests.torchcell.verification.test_natural_isolate_genotype]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_natural_isolate_genotype.py
"""The four isolate-genotype rules, on the synthetic five-isolate panel.

The panel, its matrices and its Table S1 are ``tests/torchcell/sequence/
test_natural_isolate.py``'s. The library mirror holds the same workbook bytes as the
tier (the published workbook is one file in both places), pinned through
``natural_isolate.TABLES_SHA256``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.torchcell.sequence._peter_tier import TableS1Row, tables_workbook, write_tier
from tests.torchcell.sequence.test_natural_isolate import (
    _SGD_FSA,
    _TABLE_S1,
    ISOLATES,
    _peter_files,
)
from torchcell.sequence import natural_isolate as ni
from torchcell.sequence.genome.registry import PETER2018_1011, SGD_S288C_R64
from torchcell.verification.natural_isolate_genotype import IsolateGenotypeAudit
from torchcell.verification.report import Level


@pytest.fixture
def audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> IsolateGenotypeAudit:
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    files = _peter_files()
    write_tier(data_root, SGD_S288C_R64, {ni.SGD_FSA_NAME: _SGD_FSA.encode()})
    write_tier(data_root, PETER2018_1011, files)
    library = tmp_path / "library"
    tables = library / ni.CITATION_KEY / ni.LIBRARY_TABLES_REL
    tables.parent.mkdir(parents=True)
    tables.write_bytes(files[ni.SUPPLEMENTARY_TABLES_NAME])
    monkeypatch.setattr(
        ni,
        "TABLES_SHA256",
        hashlib.sha256(files[ni.SUPPLEMENTARY_TABLES_NAME]).hexdigest(),
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    ni.link_genotype_files(str(raw), str(data_root))
    genotypes = ni.NaturalIsolateGenotypes(raw, tmp_path, ISOLATES, str(data_root))
    return IsolateGenotypeAudit(genotypes, library)


def _dumps(audit: IsolateGenotypeAudit) -> dict[str, dict[str, Any]]:
    return {i: audit.genotypes.genotype(i).model_dump() for i in ISOLATES}


def _by_name(audit: IsolateGenotypeAudit) -> dict[str, Any]:
    return {r.name: r for r in audit.results()}


def test_the_built_genotypes_pass_all_four_rules(audit: IsolateGenotypeAudit) -> None:
    for isolate, dump in _dumps(audit).items():
        assert audit.add(dump) == isolate
    results = _by_name(audit)
    assert {name: r.passed for name, r in results.items()} == {
        "isolate_genotype_identity": True,
        "isolate_genotype_round_trip": True,
        "isolate_descriptor_matches_table_s1": True,
        "loss_of_function_calls_match_release": True,
    }
    assert [r.level for r in audit.results()] == [
        Level.L1,
        Level.L2,
        Level.L4,
        Level.L4,
    ]
    assert results["isolate_descriptor_matches_table_s1"].details["n_ho_deleted"] == 2
    assert results["loss_of_function_calls_match_release"].details == {
        "totals": {"nonsense_or_sift": 5, "homozygous_frameshift": 8},
        "differ": [],
    }
    assert results["loss_of_function_calls_match_release"].message == (
        "stored loss-of-function calls are the release's 1s, row for row by the "
        "matrix's own key, over 5 isolates: "
        "{'nonsense_or_sift': 5, 'homozygous_frameshift': 8}"
    )


def test_identity_names_a_missing_descriptor_a_foreign_strain_and_a_repeat(
    audit: IsolateGenotypeAudit,
) -> None:
    dumps = _dumps(audit)
    assert audit.add({"perturbations": dumps["AAA"]["perturbations"]}) is None
    foreign = json.loads(json.dumps(dumps["BBB"], default=str))
    foreign["perturbations"][0]["strain_id"] = "ZZZ"
    audit.add(foreign)
    audit.add(dumps["CCC"])
    audit.add(dumps["CCC"])
    result = _by_name(audit)["isolate_genotype_identity"]
    assert result.passed is False
    assert result.message == (
        "1 genotypes without natural_isolate; 1 name another strain; 1 isolates with "
        "two genotypes; 3 builder isolates not stored"
    )
    assert result.details["foreign_strain"] == ["BBB: ['ZZZ']"]
    assert result.details["repeated"] == ["CCC"]
    assert result.details["missing"] == ["AAA", "DDD", "SACE_YAU"]


def test_round_trip_and_release_keys_catch_a_dropped_or_rekeyed_call(
    audit: IsolateGenotypeAudit,
) -> None:
    dumps = _dumps(audit)
    dropped = dict(dumps["AAA"])
    dropped["perturbations"] = [
        q
        for q in dumps["AAA"]["perturbations"]
        if not (
            q["perturbation_type"] == "natural_loss_of_function"
            and q["systematic_gene_name"] == "YBR001W"
        )
    ]
    rekeyed = json.loads(json.dumps(dumps["DDD"], default=str))
    for q in rekeyed["perturbations"]:
        if q["release_gene_name"] == "LTO1":
            q["release_gene_name"] = q["systematic_gene_name"] = "YNL260C"
    for isolate, dump in dumps.items():
        audit.add({"AAA": dropped, "DDD": rekeyed}.get(isolate, dump))
    results = _by_name(audit)
    assert results["isolate_genotype_round_trip"].details["failures"] == [
        "AAA: differs from the rebuild",
        "DDD: differs from the rebuild",
    ]
    assert results["loss_of_function_calls_match_release"].details["differ"] == [
        "AAA/nonsense_or_sift: stored 1, release 2; extra [], missing ['YBR001W']",
        "DDD/homozygous_frameshift: stored 2, release 2; extra ['YNL260C'], "
        "missing ['LTO1']",
    ]


def test_descriptors_are_checked_against_the_library_table_s1(
    audit: IsolateGenotypeAudit, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A library Table S1 that marks AAA HO-deleted disagrees with the stored AAA
    (no manipulation, no HO deletion); a workbook off its pin fails integrity.
    """
    other = [TableS1Row(isolate="AAA", ho_deletion="yes"), *_TABLE_S1[1:]]
    data = tables_workbook(other)
    audit.library_tables.write_bytes(data)
    monkeypatch.setattr(ni, "TABLES_SHA256", hashlib.sha256(data).hexdigest())
    for dump in _dumps(audit).values():
        audit.add(dump)
    result = _by_name(audit)["isolate_descriptor_matches_table_s1"]
    assert result.details["problems"] == [
        "AAA: {'genetic_manipulation': None, 'ploidy': 2, 'zygosity': 'homozygous', "
        "'chromosome_copy_changes': []} != Table S1 {'genetic_manipulation': "
        "'ho_deletion', 'ploidy': 2, 'zygosity': 'homozygous', "
        "'chromosome_copy_changes': []}",
        "AAA: HO deletion True in Table S1",
    ]
    monkeypatch.setattr(ni, "TABLES_SHA256", "0" * 64)
    drifted = _by_name(audit)["isolate_descriptor_matches_table_s1"]
    assert (drifted.passed, drifted.details["workbook_sha256_ok"]) == (False, False)


def test_an_isolate_outside_the_builder_fails_the_round_trip(
    audit: IsolateGenotypeAudit,
) -> None:
    dump = _dumps(audit)["AAA"]
    dump["natural_isolate"] = {**dump["natural_isolate"], "isolate": "EEE"}
    for q in dump["perturbations"]:
        if "strain_id" in q:
            q["strain_id"] = "EEE"
    audit.add(dump)
    assert audit.round_trip_failures == ["EEE: not an isolate of the build"]
