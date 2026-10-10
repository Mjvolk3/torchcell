# tests/torchcell/datasets/ecoli/test_zhang2015.py
# [[tests.torchcell.datasets.ecoli.test_zhang2015]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_zhang2015.py
"""The CeCaFDB 2015 provenance record: the decision, and the measurement behind it.

Synthetic tests run everywhere: they build Download-page HTML and workbook grids in the
release's template shape, plus counterfactual workbooks that DO carry an interval
column or an uncertainty word, so the inventory rule is shown to separate the two rather
than merely to return ``False`` on the real bytes.

The ``@pytest.mark.data`` tests read the deposited raw mirror: every quote is audited
against the pinned PMC text, and the measured inventory is asserted exactly.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from pydantic import BaseModel

import torchcell.datasets.ecoli.zhang2015 as z
from torchcell.datamodels.schema import (
    FluxExperiment,
    FluxExperimentReference,
    FluxPhenotype,
)
from torchcell.literature.manifest import Manifest
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

DATA_ROOT = os.environ.get("DATA_ROOT")


# --------------------------------------------------------------------------- #
# Synthetic release
# --------------------------------------------------------------------------- #
def _grid(
    *,
    strains: list[str],
    values: list[list[z.Cell]],
    extra: list[z.Cell] | None = None,
    genotype_label: str = "genotype",
    trailer: bool = False,
    remark: str = "",
) -> list[list[z.Cell]]:
    """A workbook grid in the CeCaFDB ``V1.0F`` template shape."""
    n = len(strains)
    width = 3 + n + (1 if extra is not None else 0)

    def row(*cells: z.Cell) -> list[z.Cell]:
        out: list[z.Cell] = list(cells) + [""] * (width - len(cells))
        return out[:width]

    grid = [
        row("", "", "", "Internal Info"),
        row("", "", "", "V1.0F"),
        row("Experiment"),
        row("Start of Experiment (Date)", "15/02/12"),
        row("Remark", remark),
        row("Experiment Name (ID)", "J Test. 2000;1:1-2. A  flux paper. Doe J."),
        row("Coordinator", "http://example.org/paper"),
        row(""),
        row("Growth condition"),
        row("Strains", *strains),
        row(genotype_label, *[f"geno {s}" for s in strains]),
        row("culture medium", *["M9"] * n),
        row("carbon source", *["Glucose"] * n),
        row("growth rate", *["0.1/h"] * n),
        row("Case-specific description", *[f"case {i}" for i in range(n)]),
        row(""),
        row("Conditions", *[f"case {i + 1}" for i in range(n)]),
        row(""),
        row("Measurements", "", "Conditions", *[float(i + 1) for i in range(n)]),
        row("", "", "Time"),
        row("Reaction", "Reactionname", "Unit"),
    ]
    for i, vals in enumerate(values):
        tail = [extra[i]] if extra is not None else []
        grid.append(row(f"A{i} <==> B{i}", f"R0000{i}", "relative flux", *vals, *tail))
    grid.append(row(""))
    if trailer:
        grid.append(row("carbon source", "a second, free-text block"))
    return grid


def _index_html(rows: list[tuple[str, str, str, str]]) -> str:
    """Download-page HTML for ``(species, reference, code, file)`` rows."""
    by_species: dict[str, list[tuple[str, str, str]]] = {}
    for species, reference, code, file in rows:
        by_species.setdefault(species, []).append((reference, code, file))
    out = ["<table>"]
    for species, items in by_species.items():
        out.append(f"<tr><td>{species}</td><td>first</td></tr>")
        out.append(f'<tr><td>{species}</td><td colspan="2"><ul>')
        for reference, code, file in items:
            out.append(
                f'<li class="flip">{reference}</li>'
                f'<p class="flip" id="{code}" align="center">Click here</p>'
                f'<div class="{code}" style="display: none;">'
                f'<span><a href="./download_files/{file}">download--{file}</a></span>'
                "</div>"
            )
        out.append("</ul></td></tr>")
    out.append("</table>")
    return "\n".join(out)


REFERENCE = "J Test. 2000;1:1-2. A flux paper. Doe J."


# --------------------------------------------------------------------------- #
# The decision
# --------------------------------------------------------------------------- #
def test_the_module_registers_no_dataset() -> None:
    from torchcell.datasets.dataset_registry import dataset_registry

    assert not [
        name for name, cls in dataset_registry.items() if cls.__module__ == z.__name__
    ]


def test_the_interval_gap_is_typed_as_dropped_by_curation() -> None:
    assert z.INTERVAL_GAP.reason.value == "not_carried_by_curation"
    assert "net_flux_lower" in z.INTERVAL_GAP.field


def test_every_field_fit_names_a_live_schema_field() -> None:
    classes: dict[str, type[BaseModel]] = {
        "FluxPhenotype": FluxPhenotype,
        "FluxExperimentReference": FluxExperimentReference,
        "FluxExperiment": FluxExperiment,
    }
    for fit in z.FLUX_FIELD_FIT:
        cls_name, field = fit.field.split(".")
        assert field in classes[cls_name].model_fields, fit.field
    assert [f.field for f in z.FLUX_FIELD_FIT if f.carried] == [
        "FluxPhenotype.net_flux",
        "FluxPhenotype.measurement_type",
    ]


def test_the_reference_map_is_required_on_the_live_schema() -> None:
    assert FluxExperimentReference.model_fields["phenotype_reference"].is_required()


def test_pins_cover_both_hosts_with_unique_files() -> None:
    assert len(z.WORKBOOKS) == 33
    assert len(z.WORKBOOKS_BY_FILE) == 33
    assert sum(w.species == z.ECOLI for w in z.WORKBOOKS) == 31
    assert sum(w.species == z.PPUTIDA for w in z.WORKBOOKS) == 2
    assert len({w.doi.lower() for w in z.WORKBOOKS}) == 33


def test_a_workbook_pin_carries_a_rerunnable_retrieval() -> None:
    pin = z.WORKBOOKS[0]
    assert pin.relpath == f"data/xls/{pin.file}"
    assert pin.retrieval.params == {"url": z.WORKBOOK_URL_PREFIX + pin.file}
    assert pin.retrieval.sha256 == pin.sha256
    assert z.OTHER_FILES[0].retrieval.params == {"key": "PMC4383945.1/PMC4383945.1.txt"}


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def test_the_download_index_is_read_per_species_and_reference() -> None:
    page = _index_html(
        [
            (z.ECOLI, "Ref &amp; one.", "00000001", "aa.xls"),
            (z.ECOLI, "Ref two.", "00000002", "bb.xls"),
            ("Homo sapiens", "Ref three.", "00000001", "cc.xls"),
        ]
    )
    rows = z.parse_download_index(page)
    assert [(r.species, r.reference, r.reference_code, r.files) for r in rows] == [
        (z.ECOLI, "Ref & one.", "00000001", ["aa.xls"]),
        (z.ECOLI, "Ref two.", "00000002", ["bb.xls"]),
        ("Homo sapiens", "Ref three.", "00000001", ["cc.xls"]),
    ]


def test_a_workbook_is_parsed_by_its_template() -> None:
    grid = _grid(
        strains=["E. coli MG1655", "E. coli JWK1"],
        values=[[100.0, 100.0], [54.0, "X"], ["", -12.5]],
        genotype_label="Genotype",
        trailer=True,
    )
    parsed = z.parse_workbook(grid)
    assert parsed.experiment_name == REFERENCE
    assert parsed.coordinator == "http://example.org/paper"
    assert parsed.case_ids == ["1.0", "2.0"]
    assert parsed.strains == ["E. coli MG1655", "E. coli JWK1"]
    assert parsed.genotypes == ["geno E. coli MG1655", "geno E. coli JWK1"]
    assert parsed.carbon_sources == ["Glucose", "Glucose"]
    assert parsed.reaction_codes == ["R00000", "R00001", "R00002"]
    assert parsed.units == ["relative flux"] * 3
    assert parsed.values == [[100.0, 100.0], [54.0, "X"], ["", -12.5]]
    assert parsed.n_extra_value_columns == 0
    assert parsed.n_interval_term_cells == 0


def test_an_interval_column_and_word_are_detected() -> None:
    grid = _grid(
        strains=["E. coli MG1655"],
        values=[[100.0], [50.0]],
        extra=[2.0, 3.0],
        remark="values are mean ± SD",
    )
    parsed = z.parse_workbook(grid)
    assert parsed.n_extra_value_columns == 1
    assert parsed.n_interval_term_cells == 1


def test_a_missing_template_label_is_refused() -> None:
    grid = [row for row in _grid(strains=["s"], values=[[1.0]]) if row[0] != "Strains"]
    with pytest.raises(ValueError, match="expected one 'Strains' row, found 0"):
        z.parse_workbook(grid)


def test_an_unexpected_value_header_is_refused() -> None:
    grid = _grid(strains=["s"], values=[[1.0]])
    header = next(i for i, row in enumerate(grid) if row[0] == "Reaction")
    grid[header][2] = "Units"
    with pytest.raises(ValueError, match="unexpected value header"):
        z.parse_workbook(grid)


def test_non_contiguous_case_columns_are_refused() -> None:
    grid = _grid(strains=["s", "t"], values=[[1.0, 2.0]])
    measurements = next(i for i, row in enumerate(grid) if row[0] == "Measurements")
    grid[measurements][3] = ""
    with pytest.raises(ValueError, match="not contiguous"):
        z.parse_workbook(grid)


def test_a_fill_series_is_the_longest_consecutive_suffix_run() -> None:
    labels = [f"E. coli BW{25113 + i}" for i in range(5)]
    assert z.longest_label_run(labels) == 5
    assert z.longest_label_run(["E. coli MG1655", "E. coli JWK1", "x"]) == 1
    assert z.longest_label_run(["W3110", "VH33", "VH34", "VH35", "no digits"]) == 3
    assert z.longest_label_run(["no digits", "none"]) == 0


def test_genome_tier_strains_are_matched_as_words() -> None:
    assert z.names_genome_tier_strain("Escherichia coli K-12 MG1655")
    assert z.names_genome_tier_strain("Pseudomonas putida. KT2440")
    assert not z.names_genome_tier_strain("Escherichia coli BW25114")
    assert not z.names_genome_tier_strain("Escherichia coli JM101")


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
def _release(
    interval_file: str | None = None,
) -> tuple[list[z.IndexRow], dict[str, z.ParsedWorkbook]]:
    index = [
        z.IndexRow(
            species=w.species,
            reference=REFERENCE,
            reference_code=w.reference_code,
            files=[w.file],
        )
        for w in z.WORKBOOKS
    ] + [
        z.IndexRow(
            species="Homo sapiens", reference="other", reference_code="1", files=["o"]
        )
    ]
    parsed = {}
    for i, w in enumerate(z.WORKBOOKS):
        n = 1 if i % 11 == 0 else 2
        extra: list[z.Cell] | None = [0.5] if w.file == interval_file else None
        grid = _grid(
            strains=["E. coli MG1655"] + ["E. coli JM101"] * (n - 1),
            values=[[100.0] * n],
            extra=extra,
        )
        parsed[w.file] = z.parse_workbook(grid)
    return index, parsed


def test_the_inventory_totals_each_host() -> None:
    index, parsed = _release()
    inv = z.inventory_release(index, parsed)
    assert inv.n_index_references == 34
    assert inv.units == ["relative flux"]
    assert inv.all_attributed
    assert inv.n_workbooks_with_interval == 0
    assert not inv.loadable_as_flux_with_interval
    single = sum(1 for i in range(33) if i % 11 == 0)
    assert (
        inv.totals[z.ECOLI].single_case_references
        + inv.totals[z.PPUTIDA].single_case_references
        == single
    )
    assert inv.totals[z.ECOLI].cases + inv.totals[z.PPUTIDA].cases == 66 - single
    assert inv.totals[z.ECOLI].cases_naming_genome_tier_strain == 31


def test_a_release_that_did_carry_an_interval_would_be_found() -> None:
    index, parsed = _release(interval_file=z.WORKBOOKS[3].file)
    inv = z.inventory_release(index, parsed)
    assert inv.n_workbooks_with_interval == 1
    assert inv.loadable_as_flux_with_interval


def test_a_misattributed_workbook_is_reported() -> None:
    index, parsed = _release()
    index[0] = index[0].model_copy(update={"reference": "a different paper"})
    inv = z.inventory_release(index, parsed)
    assert not inv.all_attributed
    assert [w.index_reference_matches for w in inv.workbooks].count(False) == 1


def test_an_index_that_moved_is_refused() -> None:
    index, parsed = _release()
    with pytest.raises(ValueError, match="no longer lists exactly"):
        z.inventory_release(index[1:], parsed)


def test_placeholders_are_counted_apart_from_numbers() -> None:
    parsed = z.parse_workbook(_grid(strains=["a", "b"], values=[[1.0, "X"], ["x", ""]]))
    inv = z.inventory_workbook(z.WORKBOOKS[0], parsed, REFERENCE)
    assert (inv.n_numeric_values, inv.n_placeholder_values) == (1, 2)


# --------------------------------------------------------------------------- #
# Deposit and retrieval
# --------------------------------------------------------------------------- #
def _fake_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Pins replaced by two tiny files so the deposit runs without the network."""
    src = tmp_path / "src"
    src.mkdir()
    pins = []
    for name in ("a.xls", "b.xls"):
        data = name.encode()
        (src / name).write_bytes(data)
        pins.append(
            z.WorkbookPin(
                file=name,
                sha256=hashlib.sha256(data).hexdigest(),
                n_bytes=len(data),
                species=z.ECOLI,
                reference_code="00000001",
                pmid="1",
                doi="10.1/x",
            )
        )
    others = []
    for rel in (z.PAPER_TEXT_REL, z.DOWNLOAD_INDEX_REL):
        data = rel.encode()
        path = src / Path(rel).name
        path.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        others.append(
            z.OtherFile(
                relpath=rel,
                sha256=sha,
                role="raw_data",
                retrieval=z.OTHER_FILES[0].retrieval.model_copy(update={"sha256": sha}),
            )
        )
    monkeypatch.setattr(z, "WORKBOOKS", tuple(pins))
    monkeypatch.setattr(z, "OTHER_FILES", tuple(others))
    sources = {p.relpath: src / p.file for p in pins}
    sources.update({o.relpath: src / Path(o.relpath).name for o in others})
    return sources


def test_the_deposit_writes_files_and_a_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _fake_release(tmp_path, monkeypatch)
    root = z.deposit_raw_mirror(sources=sources, data_root=str(tmp_path / "data"))
    assert root == tmp_path / "data" / z.RAW_DIR_REL
    manifest = Manifest.model_validate_json((root / "manifest.json").read_text())
    assert manifest.citation_key == z.CITATION_KEY
    assert [f.path for f in manifest.files] == [
        "data/xls/a.xls",
        "data/xls/b.xls",
        z.PAPER_TEXT_REL,
        z.DOWNLOAD_INDEX_REL,
    ]
    assert (root / "data/xls/a.xls").read_bytes() == b"a.xls"
    # Idempotent: a second deposit over identical bytes succeeds.
    z.deposit_raw_mirror(sources=sources, data_root=str(tmp_path / "data"))


def test_the_deposit_refuses_drifted_or_missing_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _fake_release(tmp_path, monkeypatch)
    with pytest.raises(KeyError, match="no source given"):
        z.deposit_raw_mirror(
            sources={"data/xls/a.xls": sources["data/xls/a.xls"]},
            data_root=str(tmp_path / "data"),
        )
    sources["data/xls/a.xls"].write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        z.deposit_raw_mirror(sources=sources, data_root=str(tmp_path / "data"))


def test_the_deposit_never_overwrites_other_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _fake_release(tmp_path, monkeypatch)
    dest = tmp_path / "data" / z.RAW_DIR_REL / "data/xls/a.xls"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"other")
    with pytest.raises(RuntimeError, match="different sha256; refusing"):
        z.deposit_raw_mirror(sources=sources, data_root=str(tmp_path / "data"))


def test_retrieval_reruns_every_record_and_checks_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _fake_release(tmp_path, monkeypatch)
    by_sha = {hashlib.sha256(p.read_bytes()).hexdigest(): p for p in sources.values()}
    monkeypatch.setattr(
        z, "run_retriever", lambda record: by_sha[record.sha256].read_bytes()
    )
    out = z.retrieve_release(tmp_path / "fetched")
    assert set(out) == set(sources)
    assert out["data/xls/b.xls"].read_bytes() == b"b.xls"
    monkeypatch.setattr(z, "run_retriever", lambda record: b"drifted")
    with pytest.raises(RuntimeError, match="sha256 drift"):
        z.retrieve_release(tmp_path / "again")


def test_raw_mirror_dir_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", "/some/root")
    assert z.raw_mirror_dir() == Path("/some/root") / z.RAW_DIR_REL


# --------------------------------------------------------------------------- #
# The real deposit
# --------------------------------------------------------------------------- #
needs_data = pytest.mark.skipif(DATA_ROOT is None, reason="DATA_ROOT is not set")


@pytest.mark.data
@needs_data
def test_every_quote_audits_against_the_pinned_paper() -> None:
    values = [v for v in vars(z).values() if isinstance(v, SourcedValue)]
    assert len(values) == 9
    library_root = Path(str(DATA_ROOT)) / "torchcell-raw"
    for value in values:
        result = audit_sourced_value(value, library_root)
        assert result.passed, (value.quote, result.message)


@pytest.mark.data
@needs_data
def test_the_real_release_measures_as_recorded() -> None:
    inv = z.release_inventory()
    assert inv.n_index_references == 118
    assert inv.all_attributed
    assert inv.units == ["relative flux"]
    assert inv.n_workbooks_with_interval == 0
    ecoli, pputida = inv.totals[z.ECOLI], inv.totals[z.PPUTIDA]
    assert (ecoli.references, ecoli.cases) == (31, 297)
    assert (pputida.references, pputida.cases) == (2, 3)
    assert (ecoli.numeric_values, ecoli.placeholder_values) == (8144, 717)
    assert (pputida.numeric_values, pputida.placeholder_values) == (88, 6)
    assert (ecoli.single_case_references, pputida.single_case_references) == (5, 1)
    assert ecoli.cases_naming_genome_tier_strain == 33
    haverkorn = next(w for w in inv.workbooks if w.doi == "10.1038/msb.2011.9")
    assert (haverkorn.n_cases, haverkorn.longest_strain_label_run) == (190, 190)
    assert inv.largest_reference_case_fraction == pytest.approx(190 / 297)


@pytest.mark.data
@needs_data
def test_the_committed_results_match_a_fresh_measurement() -> None:
    path = (
        Path(__file__).resolve().parents[4]
        / "experiments/036-dataset-fixes-before-kg-build/results"
        / "cecafdb2015_release_inventory.json"
    )
    committed = json.loads(path.read_text())
    assert committed["release"] == z.release_inventory().model_dump(mode="json")
    assert committed["paper_uncertainty_mentions"] == 0
    assert committed["duplication"]["source_dois_in_loaders"] == []
    assert committed["duplication"]["source_dois_in_candidate_table"] == [
        "10.1038/msb.2011.9"
    ]
