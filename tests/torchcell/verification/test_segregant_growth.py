# tests/torchcell/verification/test_segregant_growth.py
# [[tests.torchcell.verification.test_segregant_growth]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_segregant_growth.py
"""The segregant growth verifier on a synthetic two-segregant, three-condition panel.

Builds a tiny release in ``tmp_path`` (two crosses' marker matrices, a phenotypes table,
the README pairs, an xls with the cross sheet, a stand-in genome) and runs every level;
then breaks one record at a time and checks the right level fails.

2026.09.30 (Phase 12). The release has 17 segregants (cross A carries two, the other 15
crosses one each) and the 38 conditions of ``build_conditions()``, so the store holds
17 * 38 = 646 records; each cross's matrix has five markers (three on chrI, two on
chrII), all with reference base A, and ``marker_sample=5`` samples all five per cross,
16 * 5 = 80 checks. Added, each pinning the verdict and the exact message:

- the passing panel: the ordered (level, name) list of all 20 results and the exact
  message of each of this module's 14 own results (the six shared rules keep their names
  and order; their messages belong to ``test_common.py``);
- one table per failure mode: a schema failure at record 5 (L0), an undocumented
  environment (temperature 37), a NaN value, a nonzero and a missing reference, a
  segregant absent from the store (608 = 16 * 38 records), a block whose ``n_markers``
  is one short (both the sum invariant and the re-expansion fail), a mixed measurement
  column, a genotype row present in two cross files, the chrII reference base changed to
  C (48 / 80 = 0.6 of markers match), a gene outside SGD (1 / 2 = 0.5), a parent missing
  from the assembly index, a per-cross count differing from the expected table;
- the L3 sourced-value audit against a synthetic raw mirror under ``tmp_path``: 3 text
  anchors plus 26 media xls quotes plus 24 dose quotes = 53 audits; with the synthetic
  xls bytes the 50 xls audits fail on sha256 drift, with the pinned hash substituted they
  all pass, and one removed dose quote fails as "NOT found";
- ``segregant_gene_set`` and the module's ``__main__`` (a no-op exit 0; Finding).
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import os.path as osp
import runpy
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import torchcell.verification.report as report_module
from torchcell.datamodels import media as media_module
from torchcell.datamodels.schema import SegregantGenotype, SegregantGrowthExperiment
from torchcell.datasets.scerevisiae import bloom2019 as b
from torchcell.verification.report import Level, Provenance
from torchcell.verification.segregant_growth import (
    segregant_gene_set,
    verify_segregant_growth_streaming,
)

PROV = Provenance(source_uri="test://synthetic", citation_key=b.CITATION_KEY)


class _Seq:
    def __init__(self, s: str) -> None:
        self.seq = s


class _Genome:
    """Just enough of SCerevisiaeGenome for the marker-reference check."""

    def __init__(self, chr_ii_base: str = "A") -> None:
        self.chr_to_nc = {i + 1: f"nc{i + 1}" for i in range(16)}
        self.fasta_dna = {f"nc{i + 1}": _Seq("A" * 300_000) for i in range(16)}
        self.fasta_dna["nc2"] = _Seq(chr_ii_base * 300_000)


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
        genome=kw.pop("genome", _Genome()),
        sgd_genes={"YAL001C", "YAL002W"},
        gene_set=kw.pop("gene_set", {"YAL001C"}),
        marker_sample=5,
        skip_sourced_values=kw.pop("skip_sourced_values", True),
        **kw,
    )


def test_good_panel_passes_every_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _write_release(tmp_path, monkeypatch)
    report = _run(setup, _records(setup), tmp_path)
    failed = [r.name for r in report.results if not r.passed]
    # ``media_compound_identity`` used to fail here on the SHARED library rather than on
    # this dataset: YPD's dextrose and the YNB vitamins were built with
    # Compound(name=...) instead of through resolved_compound, so they carried neither
    # an identifier nor a gap. torchcell/datamodels/media.py now routes every
    # single-substance component through the resolver, so the list is empty.
    assert failed == [], report.summary()
    by_name = {r.name: r for r in report.results}
    # the dataset's OWN compounds (the stress conditions) are identified or typed-gapped
    assert by_name["compound_identity"].passed
    assert by_name["media_compound_identity"].details["name_only_records"] == {}
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


OWN_MESSAGES = {
    "structural": "646 records validated",
    "count": "observed 646, expected 646",
    "pair_uniqueness": "646 unique (segregant, condition) records, one each",
    "id_bijection": "17 phenotype ids <-> 17 genotype rows; 17 segregants in the store",
    "value_fidelity": "646 values finite and equal to the released tsv cells",
    "mosaic_round_trip": "17 mosaics re-expand to their released marker rows",
    "block_invariants": (
        "blocks ordered, non-overlapping, alternating, n_markers sums per cross"
    ),
    "measurement_partition": "36 residual + 2 absolute conditions",
    "reference_zero": "reference response == 0 for all 646 records",
    "conditions_documented": (
        "38/38 documented conditions seen; 0 records with an undocumented "
        "environment; no-edit columns = ['YNB;;1', 'YPD;;1']"
    ),
    "parents_pinned": (
        "15 parents resolve in the 1011 assembly member index; BY is the S288C reference"
    ),
    "cross_pairs_and_counts": (
        "16 crosses: xls parent pairs match the README pairs; per-cross segregant "
        "counts match the xls (sum 17)"
    ),
    "marker_reference": (
        "1.0000 of 80 sampled markers carry the S288C reference base as ref (>= 0.95)"
    ),
    "gene_containment_sgd": "1.000 of 1 spanned genes are S288C reference genes",
}
ORDER = [
    (Level.L0, "structural"),
    (Level.L1, "count"),
    (Level.L1, "pair_uniqueness"),
    (Level.L1, "id_bijection"),
    (Level.L1, "provenance_gaps"),
    (Level.L1, "canonical_gene_names"),
    (Level.L2, "uncertainty_sanity"),
    (Level.L3, "compound_identity"),
    (Level.L3, "media_compound_identity"),
    (Level.L3, "media_membership"),
    (Level.L2, "value_fidelity"),
    (Level.L2, "mosaic_round_trip"),
    (Level.L2, "block_invariants"),
    (Level.L3, "measurement_partition"),
    (Level.L3, "reference_zero"),
    (Level.L3, "conditions_documented"),
    (Level.L4, "parents_pinned"),
    (Level.L4, "cross_pairs_and_counts"),
    (Level.L4, "marker_reference"),
    (Level.L4, "gene_containment_sgd"),
]


def _failed(report: Any) -> dict[str, Any]:
    return {r.name: r for r in report.results if not r.passed}


def test_passing_panel_pins_every_message_and_the_result_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """17 segregants * 38 conditions = 646 records; 16 crosses * 5 markers = 80 checks."""
    setup = _write_release(tmp_path, monkeypatch)
    report = _run(setup, _records(setup), tmp_path)
    assert [(r.level, r.name) for r in report.results] == ORDER
    assert report.passed
    own = {r.name: r.message for r in report.results if r.name in OWN_MESSAGES}
    assert own == OWN_MESSAGES
    assert report.summary().splitlines()[0] == "bloom2019-test: PASS"


def test_schema_failure_is_reported_at_its_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A None ``dataset_name`` on record 5: L0 fails with 1/646, every other level passes."""
    setup = _write_release(tmp_path, monkeypatch)
    records = _records(setup)
    records[5]["experiment"]["dataset_name"] = None
    failed = _failed(_run(setup, records, tmp_path))
    assert list(failed) == ["structural"]
    assert failed["structural"].message == "1/646 records failed schema validation"
    assert failed["structural"].details["n_failures"] == 1
    assert failed["structural"].details["failures"][0]["index"] == 5
    assert len(failed["structural"].details["failures"][0]["error"]) == 500


def test_undocumented_environment_is_counted_and_not_value_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Record 0 moved to 37 C: its signature matches no condition; only L3 fails."""
    setup = _write_release(tmp_path, monkeypatch)
    records = _records(setup)
    records[0]["experiment"]["environment"]["temperature"]["value"] = 37.0
    records[0]["experiment"]["phenotype"]["environment_response"] += 1.0
    failed = _failed(_run(setup, records, tmp_path))
    # the tampered value is not compared: an unknown environment has no released column
    assert list(failed) == ["conditions_documented"]
    assert failed["conditions_documented"].message == (
        "38/38 documented conditions seen; 1 records with an undocumented "
        "environment; no-edit columns = ['YNB;;1', 'YPD;;1']"
    )
    assert failed["conditions_documented"].details["n_unknown"] == 1


def test_non_finite_value_fails_value_fidelity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A NaN at record 3: the schema rejects it (L0) and L2 lists it as non-finite."""
    setup = _write_release(tmp_path, monkeypatch)
    records = _records(setup)
    records[3]["experiment"]["phenotype"]["environment_response"] = math.nan
    failed = _failed(_run(setup, records, tmp_path))
    assert sorted(failed) == ["structural", "value_fidelity"]
    fidelity = failed["value_fidelity"]
    assert fidelity.message == "1 non-finite, 0 differ from the released tsv"
    assert fidelity.details["bad"] == [{"index": 3, "value": "nan"}]
    assert fidelity.details["mismatch"] == []


def test_tampered_value_reports_the_mismatch_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """+1.0 on record 3 (segregant 375_G1_01): stored = released + 1, message 0 / 1."""
    setup = _write_release(tmp_path, monkeypatch)
    records = _records(setup)
    exp = records[3]["experiment"]
    released = exp["phenotype"]["environment_response"]
    exp["phenotype"]["environment_response"] = released + 1.0
    column = list(setup["conditions"])[3]
    fidelity = _failed(_run(setup, records, tmp_path))["value_fidelity"]
    assert fidelity.message == "0 non-finite, 1 differ from the released tsv"
    assert fidelity.details["mismatch"] == [
        {
            "segregant": "375_G1_01",
            "column": column,
            "stored": released + 1.0,
            "released": float(setup["pheno"].at["375_G1_01", column]),
        }
    ]


def test_reference_response_nonzero_and_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """-0.25 fails with max|v| 0.25; a None reference is skipped (645 of 646 counted)."""
    setup = _write_release(tmp_path, monkeypatch)
    records = _records(setup)
    nonzero = copy.deepcopy(records)
    nonzero[2]["reference"]["phenotype_reference"]["environment_response"] = -0.25
    failed = _failed(_run(setup, nonzero, tmp_path))
    assert list(failed) == ["reference_zero"]
    assert failed["reference_zero"].message == (
        "reference response not identically 0: max|v|=0.25"
    )
    assert failed["reference_zero"].details["worst_abs"] == 0.25
    records[2]["reference"]["phenotype_reference"]["environment_response"] = None
    report = _run(setup, records, tmp_path)
    by_name = {r.name: r for r in report.results}
    assert by_name["reference_zero"].passed
    assert by_name["reference_zero"].message == (
        "reference response == 0 for all 645 records"
    )


def test_segregant_absent_from_the_store_breaks_the_bijection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping A01_02's 38 records: 608 observed; 17 ids against 16 stored segregants."""
    setup = _write_release(tmp_path, monkeypatch)
    records = [
        r
        for r in _records(setup)
        if r["experiment"]["genotype"]["segregant_id"] != "A01_02"
    ]
    report = _run(setup, records, tmp_path)
    failed = _failed(report)
    assert sorted(failed) == ["count", "id_bijection"]
    assert failed["count"].message == "observed 608, expected 646"
    assert failed["id_bijection"].message == (
        "17 phenotype ids <-> 17 genotype rows; 16 segregants in the store"
    )
    by_name = {r.name: r for r in report.results}
    assert by_name["mosaic_round_trip"].message == (
        "16 mosaics re-expand to their released marker rows"
    )
    # measured against the store, not the release: 16 checked, so the verdict passes
    assert by_name["mosaic_round_trip"].passed


def test_short_block_fails_the_sum_and_the_re_expansion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A01_01's last block (chrII 50-60, 2 markers) set to 1: sums to 4 of 5, and the
    block's end 60 no longer lines up with marker 3 (chrII 50).
    """
    setup = _write_release(tmp_path, monkeypatch)
    records = _records(setup)
    for rec in records:
        genotype = rec["experiment"]["genotype"]
        if genotype["segregant_id"] == "A01_01":
            genotype["blocks"][-1]["n_markers"] -= 1
    failed = _failed(_run(setup, records, tmp_path))
    assert sorted(failed) == ["block_invariants", "mosaic_round_trip"]
    assert failed["block_invariants"].message == "1 block-invariant failures"
    assert failed["block_invariants"].details["failures"] == [
        "A01_01: n_markers does not sum to 5"
    ]
    assert failed["mosaic_round_trip"].message == "1/17 mosaics fail the round trip"
    assert failed["mosaic_round_trip"].details["failures"] == [
        "A01_01: block chromosome='chrII' start=50 end=60 parent=2 posterior=1.0 "
        "n_markers=1 does not align with the marker list at 3"
    ]


def test_mixed_measurement_column_fails_the_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Record 0 (a residual column) relabeled colony_size: 35 residual + 2 absolute."""
    setup = _write_release(tmp_path, monkeypatch)
    records = _records(setup)
    records[0]["experiment"]["phenotype"]["measurement_type"] = "colony_size"
    column = list(setup["conditions"])[0]
    failed = _failed(_run(setup, records, tmp_path))
    assert list(failed) == ["measurement_partition"]
    assert failed["measurement_partition"].message == (
        "35 residual + 2 absolute conditions"
    )
    assert failed["measurement_partition"].details["mixed"] == {
        column: ["colony_size", "control_regression_residual"]
    }


def test_row_in_two_genotype_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A01_01 appended to the last cross's (3028) file: an invariant failure, a wrong
    cross, a per-cross count of 2 against the xls 1, and 18 round trips for 17 segregants.
    """
    setup = _write_release(tmp_path, monkeypatch)
    records = _records(setup)
    last = b.CROSSES[-1]
    assert last == "3028"
    with gzip.open(setup["raw"] / f"genotype_{last}.tsv.gz", "at") as fh:
        fh.write("A01_01\t1\t1\t2\t2\t2\n")
    failed = _failed(_run(setup, records, tmp_path))
    assert sorted(failed) == [
        "block_invariants",
        "cross_pairs_and_counts",
        "id_bijection",
        "mosaic_round_trip",
    ]
    assert failed["block_invariants"].details["failures"] == [
        "A01_01: in two genotype files"
    ]
    assert failed["id_bijection"].details["wrong_cross"] == ["A01_01"]
    assert failed["cross_pairs_and_counts"].message == (
        "count mismatches (stored, xls, expected): {'3028': (2, 1, 1)}"
    )


def test_round_trip_message_claims_success_when_the_count_disagrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: ``mosaic_round_trip`` picks its message from the failure list alone
    (``segregant_growth.py:357-360``), so a duplicated row (18 round trips for 17 stored
    segregants) fails with the success wording. Pinned until the message also checks
    the count.
    """
    setup = _write_release(tmp_path, monkeypatch)
    records = _records(setup)
    with gzip.open(setup["raw"] / f"genotype_{b.CROSSES[-1]}.tsv.gz", "at") as fh:
        fh.write("A01_01\t1\t1\t2\t2\t2\n")
    result = _failed(_run(setup, records, tmp_path))["mosaic_round_trip"]
    assert result.message == "18 mosaics re-expand to their released marker rows"
    assert result.details == {"n_checked": 18, "failures": []}


def test_marker_reference_ratio_and_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ChrII base C: 16 crosses * 3 chrI markers = 48 of 80 match, 0.6; fails at 0.95,
    passes at exactly 0.6 (the comparison is ``>=``).
    """
    setup = _write_release(tmp_path, monkeypatch)
    records = _records(setup)
    genome = _Genome(chr_ii_base="C")
    failed = _failed(_run(setup, records, tmp_path, genome=genome))
    assert list(failed) == ["marker_reference"]
    assert failed["marker_reference"].message == (
        "0.6000 of 80 sampled markers carry the S288C reference base as ref (>= 0.95)"
    )
    assert failed["marker_reference"].details["n_match"] == 48
    at_threshold = _run(
        setup, records, tmp_path, genome=genome, min_marker_reference=0.6
    )
    assert at_threshold.passed


def test_gene_containment_outside_sgd_and_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """{YAL001C, YZZ999W} against SGD {YAL001C, YAL002W}: 1/2 = 0.500; an empty set fails."""
    setup = _write_release(tmp_path, monkeypatch)
    records = _records(setup)
    failed = _failed(_run(setup, records, tmp_path, gene_set={"YAL001C", "YZZ999W"}))
    assert list(failed) == ["gene_containment_sgd"]
    result = failed["gene_containment_sgd"]
    assert result.message == "0.500 of 2 spanned genes are S288C reference genes"
    assert result.details["missing_examples"] == ["YZZ999W"]
    empty = _failed(_run(setup, records, tmp_path, gene_set=set()))
    assert empty["gene_containment_sgd"].message == (
        "0.000 of 0 spanned genes are S288C reference genes"
    )


def test_parent_missing_from_the_assembly_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The index loses M22's Peter id ADR (after the records are built): L4 names it."""
    setup = _write_release(tmp_path, monkeypatch)
    records = _records(setup)
    assert b.PARENT_PETER_ID["M22"] == "ADR"
    lines = setup["index"].read_text().splitlines()
    setup["index"].write_text("\n".join(x for x in lines if not x.startswith("ADR\t")))
    failed = _failed(_run(setup, records, tmp_path))
    assert list(failed) == ["parents_pinned"]
    assert failed["parents_pinned"].message == (
        "parents absent from the assembly index: ['M22 -> ADR']"
    )


def test_expected_count_differing_from_the_xls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expected table says cross A has 3 while the store and xls say 2."""
    setup = _write_release(tmp_path, monkeypatch)
    records = _records(setup)
    monkeypatch.setattr(b, "EXPECTED_SEGREGANTS", {**setup["counts"], "A": 3})
    failed = _failed(_run(setup, records, tmp_path))
    assert list(failed) == ["cross_pairs_and_counts"]
    assert failed["cross_pairs_and_counts"].message == (
        "count mismatches (stored, xls, expected): {'A': (2, 2, 3)}"
    )


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_mirror(tmp_path: Path) -> Path:
    """``<data_root>/torchcell-raw/<key>/`` with the XML, mapping.R, the xls, a manifest."""
    mirror = tmp_path / "dr" / b.RAW_DIR_REL
    (mirror / "paper").mkdir(parents=True)
    (mirror / "code").mkdir()
    (mirror / "data").mkdir()
    xml = f"<p>{b.DUPLICATE_QUOTE}</p><p>{b.INCUBATION_QUOTE}</p>".encode()
    code = f"g=1\n{b.CALL_METHOD_QUOTE}\n".encode()
    (mirror / "paper" / b.XML_NAME).write_bytes(xml)
    (mirror / "code" / "mapping.R").write_bytes(code)
    (mirror / "data" / b.XLS_NAME).write_bytes(b"synthetic xls")
    manifest = {
        "citation_key": b.CITATION_KEY,
        "files": [
            {
                "path": f"paper/{b.XML_NAME}",
                "role": "paper_ocr",
                "bytes": len(xml),
                "sha256": _sha(xml),
            },
            {
                "path": "code/mapping.R",
                "role": "software",
                "bytes": len(code),
                "sha256": _sha(code),
            },
        ],
    }
    (mirror / "manifest.json").write_text(json.dumps(manifest))
    return mirror


def _xls_quotes(conditions: dict[str, Any]) -> list[str]:
    quotes = [
        sv.quote
        for constant in media_module.MEDIA_LIBRARY.values()
        for sv in [
            *constant.provenance,
            *(p for c in constant.components for p in c.provenance),
        ]
        if sv.provenance.citation_key == b.CITATION_KEY
        and sv.provenance.source_uri.endswith(".xls")
    ]
    return quotes + [s.dose_quote for s in conditions.values() if s.dose_quote]


def _run_sourced(setup: dict[str, Any], tmp_path: Path, mirror: Path) -> Any:
    report = verify_segregant_growth_streaming(
        _records(setup),
        dataset_name="bloom2019-test",
        provenance=PROV,
        expected_count=646,
        raw_dir=setup["raw"],
        raw_mirror=mirror,
        assembly_index_path=setup["index"],
        genome=_Genome(),
        sgd_genes={"YAL001C"},
        gene_set={"YAL001C"},
        marker_sample=5,
    )
    return {r.name: r for r in report.results}["sourced_values"]


def test_sourced_values_fail_on_xls_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3 text anchors pass; the 26 + 24 = 50 xls audits see the synthetic bytes: 50/53."""
    setup = _write_release(tmp_path, monkeypatch)
    quotes = _xls_quotes(setup["conditions"])
    assert len(quotes) == 50
    result = _run_sourced(setup, tmp_path, _write_mirror(tmp_path))
    assert not result.passed
    assert result.message == "50/53 sourced values failed audit"
    assert result.details["n_audited"] == 53
    assert result.details["failures"] == [f"sha256 drift on {b.XLS_NAME}"] * 20


def _pin_xls(monkeypatch: pytest.MonkeyPatch, rows: list[str]) -> None:
    """Substitute the pinned xls hash for the synthetic file and serve ``rows``."""
    real_sha = report_module.sha256_file

    def sha(path: str | Path, **kw: Any) -> str:
        if Path(path).name == b.XLS_NAME:
            return media_module._BLOOM2019_XLS_SHA
        return real_sha(path, **kw)

    monkeypatch.setattr(report_module, "sha256_file", sha)
    served = pd.read_excel  # the release fixture's fake

    def rows_reader(path: Any, sheet_name: Any = 0, **kw: Any) -> pd.DataFrame:
        if kw.get("header", 0) is None and str(path).endswith(b.XLS_NAME):
            cells = [q.split(" | ") for q in rows]
            width = max(len(c) for c in cells)
            return pd.DataFrame([c + [math.nan] * (width - len(c)) for c in cells])
        return cast(pd.DataFrame, served(path, sheet_name=sheet_name, **kw))

    monkeypatch.setattr(pd, "read_excel", rows_reader)


def test_sourced_values_pass_with_the_pinned_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every quote served as an xls row (cells joined by `` | ``): 53 audited, all pass."""
    setup = _write_release(tmp_path, monkeypatch)
    _pin_xls(monkeypatch, _xls_quotes(setup["conditions"]))
    result = _run_sourced(setup, tmp_path, _write_mirror(tmp_path))
    assert result.passed
    assert result.message == "53 sourced values audited against the raw mirror"
    assert result.details["failures"] == []


def test_sourced_values_name_a_missing_quote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 6-azauracil dose row removed: 1/53, reported by its first 40 characters
    (11 + 3 + 8 + 3 + 5 + 3 + 4 + 3, ending in the separator).
    """
    setup = _write_release(tmp_path, monkeypatch)
    quotes = _xls_quotes(setup["conditions"])
    missing = next(q for q in quotes if q.startswith("6-azauracil"))
    _pin_xls(monkeypatch, [q for q in quotes if q != missing])
    result = _run_sourced(setup, tmp_path, _write_mirror(tmp_path))
    assert result.message == "1/53 sourced values failed audit"
    assert result.details["failures"] == [
        "xls row NOT found for '6-azauracil | 10 mg/mL | 20 mL | DMSO | '"
    ]


def test_sourced_values_flag_an_edited_text_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mapping.R edited after the manifest pinned it: that one text audit drifts."""
    setup = _write_release(tmp_path, monkeypatch)
    _pin_xls(monkeypatch, _xls_quotes(setup["conditions"]))
    mirror = _write_mirror(tmp_path)
    with open(mirror / "code" / "mapping.R", "a") as fh:
        fh.write("# edited\n")
    result = _run_sourced(setup, tmp_path, mirror)
    assert result.message == "1/53 sourced values failed audit"
    assert result.details["failures"] == [
        "sha256 drift: source re-OCR'd or edited (mapping.R)"
    ]


def test_segregant_gene_set_reads_the_preprocess_json(tmp_path: Path) -> None:
    """``gene_set.json`` with a repeated name loads as a two-element set."""
    (tmp_path / "gene_set.json").write_text('["YAL001C", "YAL002W", "YAL001C"]')
    assert segregant_gene_set(tmp_path) == {"YAL001C", "YAL002W"}


def test_module_main_is_a_no_op_exit_zero() -> None:
    """Finding: ``python -m torchcell.verification.segregant_growth`` runs nothing and
    exits 0 (``segregant_growth.py:645-646``), so a shell step that calls it always
    succeeds. Pinned until it gets a CLI or loses the ``__main__`` block.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(SystemExit) as exit_info:
            runpy.run_module(
                "torchcell.verification.segregant_growth", run_name="__main__"
            )
    assert exit_info.value.code == 0
