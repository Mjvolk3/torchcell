# tests/torchcell/verification/test_segregant_growth.py
"""The segregant growth verifier on a synthetic two-segregant, three-condition panel.

Builds a tiny release in ``tmp_path`` (two crosses' marker matrices, a phenotypes table,
the README pairs, an xls with the cross sheet, a stand-in genome) and runs every level;
then breaks one record at a time and checks the right level fails.
"""

from __future__ import annotations

import gzip
import os.path as osp
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from torchcell.datamodels.schema import SegregantGenotype, SegregantGrowthExperiment
from torchcell.datasets.scerevisiae import bloom2019 as b
from torchcell.verification.report import Level, Provenance
from torchcell.verification.segregant_growth import verify_segregant_growth_streaming

PROV = Provenance(source_uri="test://synthetic", citation_key=b.CITATION_KEY)


class _Seq:
    def __init__(self, s: str) -> None:
        self.seq = s


class _Genome:
    """Just enough of SCerevisiaeGenome for the marker-reference check."""

    def __init__(self) -> None:
        self.chr_to_nc = {i + 1: f"nc{i + 1}" for i in range(16)}
        self.fasta_dna = {f"nc{i + 1}": _Seq("A" * 300_000) for i in range(16)}


def _write_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    raw = tmp_path / "raw"
    raw.mkdir()
    # two crosses with three markers on chrI and two on chrII; ref allele always A
    header = [
        "chrI_100_A_T_1",
        "chrI_200_A_T_2",
        "chrI_300_A_T_3",
        "chrII_50_A_T_4",
        "chrII_60_A_T_5",
    ]
    matrices: dict[str, dict[str, list[int]]] = {
        "A": {"A01_01": [1, 1, 2, 2, 2], "A01_02": [2, 2, 2, 1, 1]},
        "375": {"375_G1_01": [1, 2, 2, 1, 1]},
    }
    for cross in b.CROSSES:
        rows = matrices.get(cross, {f"{cross}_G1_01": [1, 1, 1, 2, 2]})
        with gzip.open(raw / f"genotype_{cross}.tsv.gz", "wt") as fh:
            fh.write("\t" + "\t".join(header) + "\n")
            for rid, calls in rows.items():
                fh.write(rid + "\t" + "\t".join(map(str, calls)) + "\n")
    conditions = b.build_conditions()
    cols = list(conditions)
    ids = [
        rid
        for cross in b.CROSSES
        for rid in (matrices.get(cross) or {f"{cross}_G1_01": []})
    ]
    rng = np.random.default_rng(1)
    pheno = pd.DataFrame(
        rng.normal(size=(len(ids), len(cols))), index=ids, columns=cols
    )
    pheno["YPD;;1"] = 50.0
    pheno["YNB;;1"] = 40.0
    pheno["YPD;;2"] = 51.0
    pheno["YPD;;3"] = 52.0
    pheno.to_csv(raw / b.PHENOTYPES_NAME, sep="\t", compression="gzip")
    # README pairs + a cross sheet whose counts match this release
    pairs = {c: (p1, p2) for c, (p1, p2) in zip(b.CROSSES, [
        ("BYa", "M22"), ("BYa", "RMx"), ("RMx", "YPS163a"), ("YJM145x", "YPS163a"),
        ("CLIB413a", "YJM145x"), ("CLIB413a", "YJM978x"), ("YJM454a", "YJM978x"),
        ("YJM454a", "YPS1009x"), ("I14a", "YPS1009x"), ("I14a", "Y10x"), ("PW5a", "Y10x"),
        ("273614xa", "PW5a"), ("273614xa", "YJM981x"), ("CBS2888a", "YJM981x"),
        ("CBS2888a", "CLIB219x"), ("CLIB219x", "M22"),
    ])}  # fmt: skip
    lines = (
        ["header", "", "cross\tParent1\tParent2"]
        + [f"{c}\t{p1}\t{p2}\t" for c, (p1, p2) in pairs.items()]
        + ["", "trailer"]
    )
    (raw / b.README_NAME).write_text("\n".join(lines))
    counts = {c: len(matrices.get(c, {"x": 1})) for c in b.CROSSES}
    monkeypatch.setattr(b, "EXPECTED_SEGREGANTS", counts)
    monkeypatch.setattr(b, "N_SEGREGANTS", sum(counts.values()))
    sheet = pd.DataFrame(
        {
            "Diploid Parent": [f"y{c}" for c in b.CROSSES],
            "Strain ID of Parent 1 in Peter et al. 2018": ["x"] * 16,
            "Parent 1": [b.PARENT_XLS_PREFIX[pairs[c][1]] + "MatA" for c in b.CROSSES],
            "Parent 2": [
                b.PARENT_XLS_PREFIX[pairs[c][0]] + "MatAlpha" for c in b.CROSSES
            ],
            "name of cross used in provided code": b.CROSSES,
            "Magic Marker Plasmid (if used)": ["(none)"] * 16,
            "Number of Segregants Analyzed": [counts[c] for c in b.CROSSES],
        }
    )
    xls = raw / b.XLS_NAME
    # xlrd reads .xls only; write via the sheet reader's own path by monkeypatching read_excel
    real_read_excel = pd.read_excel

    def fake_read_excel(path: Any, sheet_name: Any = 0, **kw: Any) -> pd.DataFrame:
        if str(path).endswith(b.XLS_NAME):
            if sheet_name == "Crosses and Strains" and "header" not in kw:
                return sheet.copy()
            return pd.DataFrame([[c] for c in ["YNB | 20"]])
        return cast(pd.DataFrame, real_read_excel(path, sheet_name=sheet_name, **kw))

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)
    xls.write_bytes(b"xls")
    index = tmp_path / "index.tsv"
    index.write_text(
        "\n".join(
            f"{p}\tGENOMES_ASSEMBLED/{p}.re.fa" for p in b.PARENT_PETER_ID.values() if p
        )
    )
    return {
        "raw": raw,
        "index": index,
        "matrices": matrices,
        "conditions": conditions,
        "pheno": pheno,
        "counts": counts,
    }


def _records(setup: dict[str, Any]) -> list[dict[str, Any]]:
    """Records the loader would write, built through the loader's own helpers."""
    ds = b.Bloom2019Dataset.__new__(b.Bloom2019Dataset)
    ds.conditions = setup["conditions"]
    ds.name = "Bloom2019Dataset"
    index = b.read_assembly_index(setup["index"])
    out: list[dict[str, Any]] = []
    header = b.read_marker_names(osp.join(setup["raw"], "genotype_A.tsv.gz"))
    markers = b.sorted_markers(header)
    for cross in b.CROSSES:
        frame = b.read_marker_matrix(osp.join(setup["raw"], f"genotype_{cross}.tsv.gz"))
        p1 = b.build_parent(*_pair(setup, cross)[0], index)
        p2 = b.build_parent(*_pair(setup, cross)[1], index)
        for rid, calls in zip(frame.index.astype(str), frame.to_numpy(dtype=np.int8)):
            genotype = SegregantGenotype(
                cross=cross, segregant_id=rid, parent_1=p1, parent_2=p2,
                blocks=b.encode_blocks(calls, markers), call_method="test", marker_matrix_sha256="0",
            )  # fmt: skip
            for col, spec in setup["conditions"].items():
                exp = SegregantGrowthExperiment(
                    dataset_name=ds.name,
                    genotype=genotype,
                    environment=ds._environment(spec),
                    phenotype=ds._phenotype(spec, float(setup["pheno"].at[rid, col])),
                )
                out.append(
                    {
                        "experiment": exp.model_dump(),
                        "reference": ds._reference(spec).model_dump(),
                    }
                )
    return out


def _pair(setup: dict[str, Any], cross: str) -> list[tuple[str, str]]:
    pairs = b.read_readme_pairs(osp.join(setup["raw"], b.README_NAME))
    p1, p2 = pairs[cross]
    return [(p1, f"{p1} genotype"), (p2, f"{p2} genotype")]


def _run(
    setup: dict[str, Any], records: list[dict[str, Any]], tmp_path: Path, **kw: Any
) -> Any:
    n_segregants = sum(setup["counts"].values())
    return verify_segregant_growth_streaming(
        records,
        dataset_name="bloom2019-test",
        provenance=PROV,
        expected_count=kw.pop("expected_count", n_segregants * 38),
        raw_dir=setup["raw"],
        raw_mirror=tmp_path / "mirror" / b.CITATION_KEY,
        assembly_index_path=setup["index"],
        genome=_Genome(),
        sgd_genes={"YAL001C", "YAL002W"},
        gene_set={"YAL001C"},
        marker_sample=5,
        skip_sourced_values=True,
        **kw,
    )


def test_good_panel_passes_every_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _write_release(tmp_path, monkeypatch)
    report = _run(setup, _records(setup), tmp_path)
    failed = [r.name for r in report.results if not r.passed]
    assert failed == [], report.summary()
    by_name = {r.name: r for r in report.results}
    assert by_name["measurement_partition"].details["n_residual"] == 36
    assert by_name["conditions_documented"].details["no_edit_columns"] == [
        "YNB;;1",
        "YPD;;1",
    ]
    assert by_name["marker_reference"].details["ratio"] == 1.0
    assert {r.level for r in report.results} == {
        Level.L0,
        Level.L1,
        Level.L2,
        Level.L3,
        Level.L4,
    }


def test_tampered_value_fails_l2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _write_release(tmp_path, monkeypatch)
    records = _records(setup)
    records[3]["experiment"]["phenotype"]["environment_response"] += 1.0
    report = _run(setup, records, tmp_path)
    by_name = {r.name: r for r in report.results}
    assert not by_name["value_fidelity"].passed
    assert by_name["value_fidelity"].details["n_mismatch"] == 1


def test_tampered_block_fails_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _write_release(tmp_path, monkeypatch)
    records = _records(setup)
    first = records[0]["experiment"]["genotype"]["segregant_id"]
    for rec in records:
        g = rec["experiment"]["genotype"]
        if g["segregant_id"] == first:
            g["blocks"][0]["parent"] = 3 - g["blocks"][0]["parent"]
    report = _run(setup, records, tmp_path)
    by_name = {r.name: r for r in report.results}
    assert not by_name["mosaic_round_trip"].passed
    assert not by_name["structural"].passed  # adjacent blocks now share a parent


def test_duplicate_record_fails_uniqueness_and_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _write_release(tmp_path, monkeypatch)
    records = _records(setup)
    records.append(records[0])
    report = _run(setup, records, tmp_path)
    by_name = {r.name: r for r in report.results}
    assert not by_name["pair_uniqueness"].passed and not by_name["count"].passed
