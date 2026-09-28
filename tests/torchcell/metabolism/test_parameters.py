# tests/torchcell/metabolism/test_parameters.py
# [[tests.torchcell.metabolism.test_parameters]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/metabolism/test_parameters.py
"""Kinetic and physical parameter tables on a hand-written GEM checkout under tmp_path.

``<model_dir>/data/databases/swissprot.tsv`` (tab separated, columns ``gene_id``,
``uniprot``, ``MW``, ``sequence``):

====================================  =======  =======  ========
gene_id (alias list)                  uniprot  MW (Da)  sequence
====================================  =======  =======  ========
``RAM2 YKL019W``                      P1       38000    MKT
``ERG20 BOT3 YJL167W J0525``          P2       40482    MAS
``COX1 Q0045``                        P3       (empty)  (empty)
``FOO1 BAR2``                         P4       1000     MF
``DUP YKL019W``                       P5       99       MD
``YDR034C-A``                         (empty)  5000     MQ
====================================  =======  =======  ========

Only the token that matches the systematic-name pattern keys a row, and the FIRST row
per key wins (``setdefault``), so ``YKL019W`` -> P1 and ``FOO1 BAR2`` is dropped.

The OED mirror holds three rows with a manifest pinning the sha256 of the exact bytes:
P1 kcat 10 at 30 C and 50 at 37 C (wildtype both), P2 kcat 40 at 25 C. The catalytic
units are unit 0 = {YKL019W, YJL167W}, unit 1 = {Q0045}, unit 2 = {YDR034C-A}, unit 3 = {}
(no gene). ``YKL019W`` resolves to 10 (nearest 30 C; the 37 C row would give 50),
``YJL167W`` to 40, ``Q0045`` has no OED row and no sequence, ``YDR034C-A`` has no
accession but a sequence, so the registered predictor (``realkcat``, returning 6) fills
it. Measured = [10, 40, 6], median 10, so the table is [min(10, 40), 10 default,
6 realkcat, 10 default] = [10, 10, 6, 10]. Unit 0 == 10 is what exposes the nearest-30 C
choice: picking the 37 C row would make it min(50, 40) = 40.

``YMDBconcentrations.csv`` holds ChEBI 17234 at 250 uM and 15377 at 1000 uM, i.e.
2.5e-4 M and 1e-3 M.
"""

import csv
import hashlib
import json
from pathlib import Path

import cobra
import pytest
import torch

from torchcell.metabolism.constraints import CatalyticUnits, TableCoverage
from torchcell.metabolism.parameters import (
    EXPERIMENTAL_SOURCES,
    FALLBACK_KCAT_PER_S,
    KcatPredictor,
    ParameterProvenance,
    ParameterTable,
    PredictorRegistry,
    concentration_prior,
    load_measured_concentrations,
    load_swissprot,
    molecular_weight_table,
    resolve_kcat_table,
    uniprot_for_genes,
)

SWISSPROT_ROWS = [
    ("RAM2 YKL019W", "P1", "38000", "MKT"),
    ("ERG20 BOT3 YJL167W J0525", "P2", "40482", "MAS"),
    ("COX1 Q0045", "P3", "", ""),
    ("FOO1 BAR2", "P4", "1000", "MF"),
    ("DUP YKL019W", "P5", "99", "MD"),
    ("YDR034C-A", "", "5000", "MQ"),
]

OED_ROWS = [
    {
        "uniprot": "P1",
        "enzymetype": "wildtype",
        "temperature": 30.0,
        "kcat_value": 10.0,
    },
    {
        "uniprot": "P1",
        "enzymetype": "wildtype",
        "temperature": 37.0,
        "kcat_value": 50.0,
    },
    {
        "uniprot": "P2",
        "enzymetype": "wildtype",
        "temperature": 25.0,
        "kcat_value": 40.0,
    },
]

GENE_IDS = ["YKL019W", "YJL167W", "Q0045", "YDR034C-A"]

P = ParameterProvenance


def _row(gene_id: str, uniprot: str, mw: str, seq: str) -> dict[str, str]:
    return {"gene_id": gene_id, "uniprot": uniprot, "MW": mw, "sequence": seq}


@pytest.fixture
def model_dir(tmp_path: Path) -> str:
    """A GEM checkout holding swissprot.tsv and YMDBconcentrations.csv."""
    db = tmp_path / "gem" / "data" / "databases"
    db.mkdir(parents=True)
    with open(db / "swissprot.tsv", "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["gene_id", "uniprot", "MW", "sequence"])
        writer.writerows(SWISSPROT_ROWS)
    (db / "YMDBconcentrations.csv").write_text(
        "chebi,mean\nCHEBI:17234,250\nchebi:15377,1000\n,5\nCHEBI:1,\nHMDB0001,3\n"
    )
    return str(tmp_path / "gem")


@pytest.fixture
def oed_dir(tmp_path: Path) -> str:
    """An OED mirror whose manifest pins the sha256 of oed_records.json."""
    mirror = tmp_path / "oed"
    mirror.mkdir()
    payload = json.dumps(OED_ROWS).encode()
    (mirror / "oed_records.json").write_bytes(payload)
    (mirror / "manifest.json").write_text(
        json.dumps(
            {
                "source_url": "https://example.invalid/oed",
                "retrieval_command": "fetch_oed_records(...)",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "retrieved_at": "2026-09-27T00:00:00",
                "n_records": 3,
                "organism": "Saccharomyces cerevisiae",
            }
        )
    )
    return str(mirror)


def _units() -> CatalyticUnits:
    return CatalyticUnits(
        unit_gene_index=torch.tensor([[0, 0, 1, 2], [0, 1, 2, 3]]),
        unit_reaction=torch.tensor([0, 0, 1, 2]),
        n_units=4,
        n_multigene_units=1,
        n_reactions_with_gpr=3,
        gene_ids=GENE_IDS,
    )


class RecordingPredictor:
    """A KcatPredictor that returns a fixed (6.0, 0.5) and records its inputs."""

    name = "realkcat"
    emits_km = True

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.calls: list[tuple[str, str | None]] = []

    def predict(
        self, sequence: str, substrate_smiles: str | None = None
    ) -> tuple[float, float | None]:
        """Record the inputs and return (6.0, 0.5)."""
        self.calls.append((sequence, substrate_smiles))
        return 6.0, 0.5


class KcatOnly:
    """A predictor without K_M."""

    name = "kcatnet"
    emits_km = False

    def predict(
        self, sequence: str, substrate_smiles: str | None = None
    ) -> tuple[float, float | None]:
        """Return (1.0, None)."""
        return 1.0, None


def _table_dump(table: ParameterTable) -> dict[str, object]:
    return {
        "values": table.values.tolist(),
        "provenance": table.provenance,
        "known_mask": table.known_mask.tolist(),
        "experimental_mask": table.experimental_mask.tolist(),
        "unit": table.unit,
        "coverage": table.coverage.model_dump(),
        "experimental_coverage": table.experimental_coverage.model_dump(),
        "notes": table.notes,
    }


def test_load_swissprot_keys_on_the_systematic_token(model_dir: str) -> None:
    """Four keys; ``FOO1 BAR2`` has no systematic token; first ``YKL019W`` row wins."""
    assert load_swissprot(model_dir) == {
        "YKL019W": _row(*SWISSPROT_ROWS[0]),
        "YJL167W": _row(*SWISSPROT_ROWS[1]),
        "Q0045": _row(*SWISSPROT_ROWS[2]),
        "YDR034C-A": _row(*SWISSPROT_ROWS[5]),
    }


def test_molecular_weight_table(model_dir: str) -> None:
    """KDa = MW / 1000; empty MW and unknown genes take 40000 Da = 40 kDa, tagged default.

    [38000, (empty), 40482, (absent)] -> [38.0, 40.0, 40.482, 40.0]; 2 of 4 known.
    """
    table = molecular_weight_table(
        model_dir, ["YKL019W", "Q0045", "YJL167W", "YXX000W"]
    )
    assert torch.equal(
        table.values, torch.tensor([38.0, 40.0, 40.482, 40.0], dtype=torch.float32)
    )
    assert _table_dump(table) | {"values": None} == {
        "values": None,
        "provenance": [
            P.SWISSPROT,
            P.ORGANISM_DEFAULT,
            P.SWISSPROT,
            P.ORGANISM_DEFAULT,
        ],
        "known_mask": [True, False, True, False],
        "experimental_mask": [True, False, True, False],
        "unit": "kDa",
        "coverage": {"n_total": 4, "n_known": 2, "fraction": 0.5},
        "experimental_coverage": {"n_total": 4, "n_known": 2, "fraction": 0.5},
        "notes": "yeast-GEM data/databases/swissprot.tsv, joined on systematic ORF name.",
    }


def test_uniprot_for_genes_skips_empty_accessions(model_dir: str) -> None:
    """YDR034C-A has an empty accession and YXX000W no row; both are omitted."""
    assert uniprot_for_genes(
        model_dir, ["YKL019W", "YDR034C-A", "Q0045", "YXX000W"]
    ) == {"YKL019W": "P1", "Q0045": "P3"}


def test_resolve_kcat_table_database_then_predictor_then_default(
    model_dir: str, oed_dir: str
) -> None:
    """[min(10, 40), median 10, predicted 6, median 10] with the provenance of each.

    The predicted 6 differs from the default 10, so a predictor value dropped in favor
    of the default would fail; the median of [10, 40, 6] is 10.
    """
    predictor = RecordingPredictor()
    table = resolve_kcat_table(
        _units(), model_dir, oed_dir, registry=PredictorRegistry(predictors=[predictor])
    )
    assert predictor.calls == [("MQ", None)]
    assert _table_dump(table) == {
        "values": [10.0, 10.0, 6.0, 10.0],
        "provenance": [
            P.OPEN_ENZYME_DATABASE,
            P.ORGANISM_DEFAULT,
            P.REALKCAT,
            P.ORGANISM_DEFAULT,
        ],
        "known_mask": [True, False, True, False],
        "experimental_mask": [True, False, False, False],
        "unit": "1/s",
        "coverage": {"n_total": 4, "n_known": 2, "fraction": 0.5},
        "experimental_coverage": {"n_total": 4, "n_known": 1, "fraction": 0.25},
        "notes": (
            "Open Enzyme Database first (3 organism rows), then registered sequence "
            "predictors, then the organism default 10 1/s = the median of the resolved "
            "values. A complex takes the min over its subunits."
        ),
    }


def test_resolve_kcat_table_without_mirror_uses_fallback(
    model_dir: str, tmp_path: Path
) -> None:
    """No mirror and no registry: every unit is the 13.7 1/s fallback, 0 % known."""
    table = resolve_kcat_table(_units(), model_dir, str(tmp_path / "absent"))
    assert FALLBACK_KCAT_PER_S == 13.7
    assert torch.equal(table.values, torch.full((4,), 13.7, dtype=torch.float32))
    assert table.provenance == [P.ORGANISM_DEFAULT] * 4
    assert table.coverage == TableCoverage(n_total=4, n_known=0, fraction=0.0)
    assert table.notes == (
        "Open Enzyme Database first (0 organism rows), then registered sequence "
        "predictors, then the organism default 13.7 1/s = the median of the resolved "
        "values. A complex takes the min over its subunits."
    )


def test_parameter_table_build_masks() -> None:
    """Predictions are known but not experimental; only ORGANISM_DEFAULT is unknown."""
    provenance = [P.BRENDA, P.KCATNET, P.DEKP, P.YMDB, P.ORGANISM_DEFAULT]
    table = ParameterTable.build(torch.arange(5.0), provenance, unit="mM", notes="n")
    assert table.known_mask.tolist() == [True, True, True, True, False]
    assert table.experimental_mask.tolist() == [True, False, False, True, False]
    assert table.coverage == TableCoverage(n_total=5, n_known=4, fraction=0.8)
    assert table.experimental_coverage == TableCoverage(
        n_total=5, n_known=2, fraction=0.4
    )
    assert EXPERIMENTAL_SOURCES == frozenset(
        {P.BRENDA, P.OPEN_ENZYME_DATABASE, P.SWISSPROT, P.YMDB}
    )


def test_predictor_registry_first_emitting_km() -> None:
    """Priority order: the first predictor with ``emits_km`` True; none when absent."""
    kcat_only, with_km = KcatOnly(), RecordingPredictor()
    assert PredictorRegistry(predictors=[kcat_only, with_km]).first_emitting_km() is (
        with_km
    )
    assert PredictorRegistry(predictors=[kcat_only]).first_emitting_km() is None
    assert PredictorRegistry().first_emitting_km() is None
    assert isinstance(with_km, KcatPredictor)
    assert not isinstance(object(), KcatPredictor)


def test_load_measured_concentrations(model_dir: str) -> None:
    """UM -> M; case-insensitive ChEBI; blank id, blank mean and non-ChEBI rows skipped."""
    assert load_measured_concentrations(model_dir) == {
        "17234": 250 * 1e-6,
        "15377": 1000 * 1e-6,
    }


def test_concentration_prior(model_dir: str) -> None:
    """A_c (str chebi) and B_c (list, second id matches) are covered; C_c is not."""
    model = cobra.Model("toy")
    a = cobra.Metabolite("A_c", compartment="c")
    b = cobra.Metabolite("B_c", compartment="c")
    c = cobra.Metabolite("C_c", compartment="c")
    a.annotation = {"chebi": "CHEBI:17234"}
    b.annotation = {"chebi": ["CHEBI:999", "CHEBI:15377"]}
    model.add_metabolites([a, b, c])
    values, mask = concentration_prior(model, ["A_c", "B_c", "C_c"], model_dir)
    assert torch.equal(values, torch.tensor([2.5e-4, 1e-3, 0.0]))
    assert mask.tolist() == [True, True, False]
