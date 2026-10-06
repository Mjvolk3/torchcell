# tests/torchcell/datasets/scerevisiae/test_spell.py
# [[tests.torchcell.datasets.scerevisiae.test_spell]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_spell.py
"""SPELL loader: PCL parsing, the eight condition-name parsers, export, quality, plots.

Everything is hermetic. PCL files are hand-written under ``tmp_path``; SPELL study
archives are zips built in the test; no network, no ``DATA_ROOT``, no real archive.

Fixture ``_PCL`` (three genes, three conditions, one missing value, one empty EWEIGHT):

    YORF     NAME  GWEIGHT  WT_30C_10min  WT_37C_10min  WT_42C_10min
    EWEIGHT                 1             (empty)       2
    YAL001C  TFC3  1        0.12          0.45          1.23
    YAL002W  VPS8  1        -0.34         (empty)       0.89
    YAL003W  EFB1  1        0.5           0.25          -1

so ``conditions`` is the header from column 4 on, ``eweights`` is ``[1.0, 1.0, 2.0]``
(an empty weight reads as 1.0), ``n_genes = 3``, ``n_conditions = 3``, and the empty
expression cell is NaN.

Condition strings in the parser tables are taken verbatim from the real SPELL condition
vocabulary in ``experiments/015-spell/results/spell_knockout_conditions.csv`` (2,871 rows,
column ``condition_name``) wherever that vocabulary has an example of the branch; a branch
with no real example uses the pattern written in the source comment (``37°C``, ``1.5 hr``)
and is marked ``synthetic`` in its id. Several real strings land in the wrong field (an ORF
suffix read as a temperature, a gene name read as a pH or a cell-cycle phase, ``min`` read
as a molar unit); those rows are Findings and are pinned as the code behaves.

Extraction confidence (``export_condition_metadata``) is
``min(0.9, 0.1 + k / 14 * 0.8)`` rounded to 3 places for ``k`` extracted fields:
k=0 -> 0.1, k=1 -> 0.157, k=2 -> 0.214, k=3 -> 0.271.
"""

import os
import os.path as osp
import re
import statistics
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from matplotlib.axes import Axes

from torchcell.datasets.scerevisiae import spell

_PCL = (
    "YORF\tNAME\tGWEIGHT\tWT_30C_10min\tWT_37C_10min\tWT_42C_10min\n"
    "EWEIGHT\t\t\t1\t\t2\n"
    "YAL001C\tTFC3\t1\t0.12\t0.45\t1.23\n"
    "YAL002W\tVPS8\t1\t-0.34\t\t0.89\n"
    "YAL003W\tEFB1\t1\t0.5\t0.25\t-1\n"
)
_CONDITIONS = ["WT_30C_10min", "WT_37C_10min", "WT_42C_10min"]

# A second dataset for the multi-study tests: YAL001C only, two conditions.
_PCL_B = "YORF\tNAME\tGWEIGHT\tc1\tc2\nEWEIGHT\t\t\t1\t1\nYAL001C\tTFC3\t1\t2.0\t\n"


def _write(path: Path, text: str) -> str:
    """Write ``text`` to ``path`` and return it as a string path."""
    path.write_text(text)
    return str(path)


def _expected_fixture_frame() -> pd.DataFrame:
    """The ``_PCL`` fixture as a DataFrame, built by hand (GWEIGHT parses as int64)."""
    frame = pd.DataFrame(
        {
            "NAME": ["TFC3", "VPS8", "EFB1"],
            "GWEIGHT": np.array([1, 1, 1], dtype=np.int64),
            "WT_30C_10min": [0.12, -0.34, 0.5],
            "WT_37C_10min": [0.45, np.nan, 0.25],
            "WT_42C_10min": [1.23, 0.89, -1.0],
        },
        index=pd.Index(["YAL001C", "YAL002W", "YAL003W"], name="YORF"),
    )
    return frame


# ---------------------------------------------------------------------------
# read_pcl_file
# ---------------------------------------------------------------------------


def test_read_pcl_file_exact_frame_and_metadata(tmp_path: Path) -> None:
    """The fixture parses to the hand-built frame and metadata.

    eweights: header slots 4..6 of row 2 are "1", "", "2" -> [1.0, 1.0 (empty), 2.0].
    The empty expression cell of YAL002W / WT_37C_10min is NaN.
    """
    df, metadata = spell.read_pcl_file(_write(tmp_path / "a.pcl", _PCL))
    pd.testing.assert_frame_equal(df, _expected_fixture_frame())
    assert metadata == {
        "conditions": _CONDITIONS,
        "eweights": [1.0, 1.0, 2.0],
        "n_genes": 3,
        "n_conditions": 3,
    }


def test_read_pcl_file_trailing_empty_eweight_drops_the_weight(tmp_path: Path) -> None:
    r"""Finding: a trailing empty EWEIGHT cell is lost, so eweights is shorter than conditions.

    Row 2 ``"EWEIGHT\t\t\t1\t0.5\t"`` goes through ``.strip()`` before ``split``
    (spell.py:48), which removes the final tab, so the split is
    ``["EWEIGHT", "", "", "1", "0.5"]`` and ``eweights = [1.0, 0.5]`` for three
    conditions. The empty middle cell of the main fixture reads as 1.0 because it is not
    at the end. Pinned until the EWEIGHT row is split without stripping trailing tabs.
    """
    text = _PCL.replace("EWEIGHT\t\t\t1\t\t2\n", "EWEIGHT\t\t\t1\t0.5\t\n")
    _, metadata = spell.read_pcl_file(_write(tmp_path / "a.pcl", text))
    assert metadata["eweights"] == [1.0, 0.5]
    assert metadata["n_conditions"] == 3


def test_read_pcl_file_without_eweight_row_eats_the_first_gene(tmp_path: Path) -> None:
    """Finding: row 2 is taken as EWEIGHT unconditionally, so a file without it loses a gene.

    With the EWEIGHT row removed, row 2 is the YAL001C data line; its fields 4..6
    (0.12, 0.45, 1.23) become the eweights, ``skiprows=2`` drops it from the frame,
    and ``n_genes`` is 2 (spell.py:46-51). Nothing checks that row 2 starts with
    ``EWEIGHT``. Pinned until the loader checks the row label.
    """
    text = _PCL.replace("EWEIGHT\t\t\t1\t\t2\n", "")
    df, metadata = spell.read_pcl_file(_write(tmp_path / "a.pcl", text))
    assert metadata["eweights"] == [0.12, 0.45, 1.23]
    assert metadata["n_genes"] == 2
    assert df.index.tolist() == ["YAL002W", "YAL003W"]


def test_read_pcl_file_gid_column_counts_gweight_as_a_condition(tmp_path: Path) -> None:
    """Finding: conditions are ``header[3:]`` by position, so a GID-first PCL is off by one.

    Header ``GID YORF NAME GWEIGHT c1 c2``: ``header[3:]`` is ``["GWEIGHT", "c1", "c2"]``
    and ``n_conditions = len(header) - 3 = 3`` (spell.py:55-58); eweights read slots
    3..5 of row 2 ``"EWEIGHT", "", "", "", "1", "1"`` -> ``["", "1", "1"]`` ->
    ``[1.0, 1.0, 1.0]``. ``set_index("YORF")`` still succeeds, so nothing fails.
    Pinned until the condition columns are located by name instead of position.
    """
    text = (
        "GID\tYORF\tNAME\tGWEIGHT\tc1\tc2\n"
        "EWEIGHT\t\t\t\t1\t1\n"
        "G1\tYAL001C\tTFC3\t1\t0.5\t-0.5\n"
    )
    df, metadata = spell.read_pcl_file(_write(tmp_path / "gid.pcl", text))
    assert metadata["conditions"] == ["GWEIGHT", "c1", "c2"]
    assert metadata["n_conditions"] == 3
    assert metadata["eweights"] == [1.0, 1.0, 1.0]
    assert df.index.tolist() == ["YAL001C"]
    assert df.loc["YAL001C", "GID"] == "G1"


def test_read_pcl_file_without_yorf_column_raises(tmp_path: Path) -> None:
    """A header with no ``YORF`` column fails at ``set_index`` with pandas' KeyError.

    ``str(KeyError(msg))`` is the repr of the message, so the anchored match includes
    the surrounding double quotes.
    """
    path = _write(
        tmp_path / "bad.pcl", "ORF\tNAME\tGWEIGHT\tc1\nEWEIGHT\t\t\t1\nA\tB\t1\t2\n"
    )
    message = "None of ['YORF'] are in the columns"
    with pytest.raises(KeyError, match='^"' + re.escape(message) + '"$'):
        spell.read_pcl_file(path)


# ---------------------------------------------------------------------------
# The eight condition-name parsers
# ---------------------------------------------------------------------------

_NO_MATCH = "CI25E delta Cot1 replicate 1"  # real; matches no field of seven parsers


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        pytest.param(
            "F82G pho4D 1 mM 1NaPP1 10min",
            {"time_min": 10.0, "is_timeseries": False},
            id="min",
        ),
        pytest.param(
            "ADH1 deletion mutant grown in glycerol (2 hour)",
            {"time_min": 120.0, "is_timeseries": False},
            id="hour-x60",
        ),
        pytest.param(
            "pho85D DMSO 24 hrs. vs. F82G 10 mM 1-Na PP1 24 hrs.",
            {"time_min": 1440.0, "is_timeseries": False},
            id="hrs-x60",
        ),
        pytest.param(
            "mer1D_SK1_t=9hr_A",
            {"time_min": 540.0, "is_timeseries": True},
            id="t=-and-hr",
        ),
        pytest.param(
            "1.5 hr", {"time_min": 90.0, "is_timeseries": False}, id="decimal-synthetic"
        ),
        pytest.param(
            "90 sec", {"time_min": 1.5, "is_timeseries": False}, id="sec-synthetic"
        ),
        pytest.param(
            "SEC4/YFL005W",
            {"time_min": None, "is_timeseries": False},
            id="SEC-gene-needs-a-leading-number",
        ),
        pytest.param(
            "Transcriptomic regulation during meiosis t=1h, Clb5Delta-Clb6Delta mutant",
            {"time_min": None, "is_timeseries": True},
            id="finding-bare-h-not-parsed",
        ),
        pytest.param(
            "stb3-null 3 days in YPD + 2% glucose for 10 minutes A",
            {"time_min": 10.0, "is_timeseries": False},
            id="days-ignored",
        ),
        pytest.param(
            "heat shock 2 hr 30 min",
            {"time_min": 30.0, "is_timeseries": False},
            id="finding-min-beats-hr-synthetic",
        ),
        pytest.param(
            "A-nmd2D-time 0 min, biological replicate 1",
            {"time_min": 0.0, "is_timeseries": False},
            id="time-word-is-not-a-keyword",
        ),
        pytest.param(_NO_MATCH, {"time_min": None, "is_timeseries": False}, id="none"),
    ],
)
def test_extract_time_info(condition: str, expected: dict[str, Any]) -> None:
    """Unit patterns are tried in the order min, hr/hour(s), sec (spell.py:384-397).

    The first unit pattern that matches anywhere wins, not the first one in the string,
    so "2 hr 30 min" is 30.0 (Finding: the hours are dropped; pinned until the time is
    summed or taken by position). "t=1h" has no ``hr``/``hour`` token, so its time is
    None although the module note lists ``4h`` as a supported pattern (Finding, pinned
    until a bare ``h`` is parsed). 90 sec = 90 * (1/60) = 1.5 exactly in float.
    """
    assert spell.extract_time_info(condition) == expected


@pytest.mark.parametrize(
    "keyword",
    ["time course", "time series", "time zero", "timepoint", "time point", "t="],
)
def test_extract_time_info_every_timeseries_keyword(keyword: str) -> None:
    """Each of the six keywords sets ``is_timeseries``; none carries a number."""
    assert spell.extract_time_info(f"WT {keyword.upper()} sample") == {
        "time_min": None,
        "is_timeseries": True,
    }


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        pytest.param("37°C heat shock", 37.0, id="degree-sign-synthetic"),
        pytest.param("slt2D - 25C", 25.0, id="C-at-end"),
        pytest.param("WT 30C, YPD", 30.0, id="C-comma-synthetic"),
        pytest.param(
            "HHO1_delta_37_degrees_vs_Wild_type_25_degrees",
            None,
            id="finding-underscore-degrees-missed",
        ),
        pytest.param("S288C ΔGCN5, biological rep1", 288.0, id="finding-S288C-strain"),
        pytest.param("30 deg", 30.0, id="deg-synthetic"),
        pytest.param("ykl020c deletion", 20.0, id="finding-orf-suffix-c"),
        pytest.param("SPC34/YKR037C", 37.0, id="finding-orf-suffix-C-at-end"),
        pytest.param(_NO_MATCH, None, id="none"),
    ],
)
def test_extract_temperature_info(condition: str, expected: float | None) -> None:
    r"""Temperature patterns: ``°C``, then ``N C`` before space/end/comma, then ``deg``.

    Finding: the second pattern (spell.py:424) runs with ``re.IGNORECASE``, so the
    trailing ``c`` of a systematic ORF name is read as Celsius: "ykl020c deletion" ->
    20.0, "SPC34/YKR037C" -> 37.0, and the strain name "S288C" -> 288.0. Of the 2,804
    distinct real labels, 278 get a temperature and 272 of those come from an ORF
    suffix. Pinned until the pattern is case-sensitive and refuses an ORF token.
    Finding: the real form "HHO1_delta_37_degrees" gets None, because ``\s*deg``
    does not allow the underscore separator SPELL labels use. Pinned until the
    separator class includes ``_``.
    """
    assert spell.extract_temperature_info(condition) == {"temperature_c": expected}


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        pytest.param("∆zta1_vs_wt_H2O2_repl1", ("H2O2", None, None), id="name-only"),
        pytest.param(
            "nup42-delta 0.4mM H2O2 t10", ("H2O2", 0.4, "mM"), id="name-and-mM"
        ),
        pytest.param(
            "∆pde2 +4mM cAMP rep1", (None, 4.0, "mM"), id="concentration-only"
        ),
        pytest.param(
            "stb3-null 3 days in YPD + 2% glucose for 10 minutes A",
            ("glucose", 2.0, "%"),
            id="percent",
        ),
        pytest.param("hog1D +KCl rep1", ("KCl", None, None), id="KCl"),
        pytest.param(
            "0.3 mM hydrogen peroxide",
            ("hydrogen peroxide", 0.3, "mM"),
            id="list-order-before-peroxide-synthetic",
        ),
        pytest.param(
            "10 µM menadione", ("menadione", 10.0, "µM"), id="micro-sign-synthetic"
        ),
        pytest.param(
            "2 ug/ml tunicamycin",
            ("tunicamycin", 2.0, "ug/ml"),
            id="ug-per-ml-synthetic",
        ),
        pytest.param("1 M sorbitol", ("sorbitol", 1.0, "M"), id="molar-synthetic"),
        pytest.param(
            "GAL-SIC1Δ3P_biological_rep1_20min",
            (None, 20.0, "m"),
            id="finding-min-read-as-molar",
        ),
        pytest.param(
            "ADH1 deletion mutant- glycerol vs. glucose at log phase- #1",
            ("glucose", None, None),
            id="list-order-not-string-order",
        ),
        pytest.param(_NO_MATCH, (None, None, None), id="none"),
    ],
)
def test_extract_chemical_info(
    condition: str, expected: tuple[str | None, float | None, str | None]
) -> None:
    """Chemical name is the first entry of the fixed list found as a substring.

    Finding: the concentration pattern (spell.py:494) runs with ``re.IGNORECASE`` and
    its unit alternation includes ``M``, so the ``m`` of ``20min`` is a molar unit:
    "..._rep1_20min" -> concentration 20.0, unit "m" (138 of the 2,804 distinct real
    labels get unit "m"). Pinned until the unit match is
    case-sensitive and anchored on a word boundary. The chemical list is searched in its
    own order, so a string naming glycerol before glucose reports glucose.
    """
    name, conc, unit = expected
    assert spell.extract_chemical_info(condition) == {
        "chemical_name": name,
        "concentration": conc,
        "concentration_unit": unit,
    }


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        pytest.param(
            "Yeast gcr1 null mutant in galactose rep1",
            ("galactose", None, None),
            id="galactose",
        ),
        pytest.param(
            "ADH1 deletion mutant grown in glycerol (30 minutes)",
            ("glycerol", None, None),
            id="glycerol",
        ),
        pytest.param(
            "far1Δ, ethanol, biological rep1", ("ethanol", None, None), id="ethanol"
        ),
        pytest.param(
            "ADH1 deletion mutant- glycerol vs. glucose at log phase- #1",
            ("glucose", None, None),
            id="dict-order-glucose-first",
        ),
        pytest.param("YP + 2% dextrose", ("glucose", None, None), id="dextrose-synth"),
        pytest.param("YP + raffinose", ("raffinose", None, None), id="raffinose-synth"),
        pytest.param("YP + 2% acetate", ("acetate", None, None), id="acetate-synth"),
        pytest.param(
            "0.5% lactic acid", ("lactate", None, None), id="lactic-acid-synthetic"
        ),
        pytest.param(
            "(NH4)2SO4 sole nitrogen", (None, "ammonium", None), id="nh4-synthetic"
        ),
        pytest.param("proline medium", (None, "proline", None), id="proline-synth"),
        pytest.param("glutamine medium", (None, "glutamine", None), id="gln-synth"),
        pytest.param("urea medium", (None, "urea", None), id="urea-synthetic"),
        pytest.param(
            "leu3 delta C-lim1; src: Chemostat sampled cells",
            (None, None, "carbon_limited"),
            id="C-lim",
        ),
        pytest.param(
            "leu3 delta N-lim1; src: chemostat sampled cells",
            (None, None, "nitrogen_limited"),
            id="N-lim",
        ),
        pytest.param(
            "phosphate limitation", (None, None, "phosphate_limited"), id="P-synth"
        ),
        pytest.param(
            "sulfur-limited chemostat", (None, None, "sulfur_limited"), id="S-synth"
        ),
        pytest.param(
            "ammonium to proline shift",
            (None, "ammonium", None),
            id="dict-order-ammonium-first-synthetic",
        ),
        pytest.param(
            "C-lim to N-lim shift",
            (None, None, "carbon_limited"),
            id="dict-order-carbon-limited-first-synthetic",
        ),
        pytest.param(_NO_MATCH, (None, None, None), id="none"),
    ],
)
def test_extract_nutrient_info(
    condition: str, expected: tuple[str | None, str | None, str | None]
) -> None:
    """Carbon, nitrogen and limitation are each the first dict entry whose keyword occurs.

    Dict order decides ties in all three dicts, not position in the string:
    "glycerol vs. glucose" is glucose (first carbon entry, spell.py:515-523),
    "ammonium to proline shift" is ammonium (first nitrogen entry), and
    "C-lim to N-lim shift" is carbon_limited (first limitation entry), although the
    shift ends on proline and N-lim.
    """
    carbon, nitrogen, limitation = expected
    assert spell.extract_nutrient_info(condition) == {
        "carbon_source": carbon,
        "nitrogen_source": nitrogen,
        "limitation_type": limitation,
    }


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        pytest.param("YPD pH=5.5", (5.5, None), id="pH-equals-synthetic"),
        pytest.param("YPD pH: 4", (4.0, None), id="pH-colon-synthetic"),
        pytest.param("YPD (pH 3)", (3.0, None), id="pH-parenthesized-synthetic"),
        pytest.param("gph1-del-1-a", (1.0, None), id="finding-gph1-is-pH-1"),
        pytest.param("pph21-del-1-a", (21.0, None), id="finding-pph21-is-pH-21"),
        pytest.param("gph1 (pH 7)", (1.0, None), id="finding-gene-beats-real-pH"),
        pytest.param(
            "Steady-state based anaerobic glucose pulse, sfp1delta_SS_2",
            (None, "anaerobic"),
            id="anaerobic",
        ),
        pytest.param("anoxic", (None, "anaerobic"), id="anoxic-synthetic"),
        pytest.param(
            "Aerobic/hypoxic expression in hap1 deletion S.cerevisiae rep1",
            (None, "aerobic"),
            id="aerobic-before-hypoxic",
        ),
        pytest.param("low oxygen", (None, "hypoxic"), id="low-oxygen-synthetic"),
        pytest.param("hyperoxic", (None, "hyperoxic"), id="hyperoxic-synthetic"),
        pytest.param(_NO_MATCH, (None, None), id="none"),
    ],
)
def test_extract_physical_params(
    condition: str, expected: tuple[float | None, str | None]
) -> None:
    r"""Physical parameters: pH and oxygen level.

    Finding: the pH pattern ``pH\s*[=:]?\s*(\d+...)`` (spell.py:566) runs with
    ``re.IGNORECASE`` and needs no word boundary, so the gene names gph1, rph1, pph3,
    pph21, pph22 of the real vocabulary read as pH 1, 1, 3, 21, 22 (14 of the 2,804
    distinct real labels). Pinned until the
    pattern requires a word boundary and a 0-14 value. Finding: the second pH pattern
    ``\(pH\s+(\d+...)\)`` is unreachable, since every string it matches contains
    ``pH\s+digits``, which the first pattern always matches first, sometimes on a
    different token: "gph1 (pH 7)" returns 1.0, not 7.0; see
    ``test_second_ph_pattern_is_subsumed``.
    """
    ph, oxygen = expected
    assert spell.extract_physical_params(condition) == {
        "ph": ph,
        "oxygen_level": oxygen,
    }


def test_second_ph_pattern_is_subsumed() -> None:
    r"""Finding: the parenthesized pH pattern never runs, because pattern 1 always matches first.

    The two patterns are read from the constants of ``extract_physical_params`` itself
    (spell.py:566). Every string pattern 2 matches contains ``pH\s+digits``, so pattern
    1 matches it too and the loop breaks before pattern 2 is tried (spell.py:568-572).
    Pattern 1 sometimes matches a different token: "gph1 (pH 7)" has pattern 2 capture
    "7" but pattern 1 captures "1" from "gph1", and the function returns 1.0. Pinned
    until the dead pattern is removed.
    """
    consts = spell.extract_physical_params.__code__.co_consts
    first, second = [
        c for c in consts if isinstance(c, str) and c.startswith(("pH", r"\(pH"))
    ]
    assert (first, second) == (r"pH\s*[=:]?\s*(\d+\.?\d*)", r"\(pH\s+(\d+\.?\d*)\)")
    cases = {
        "(pH 3)": 3.0,
        "medium (pH  5.5) at 30": 5.5,
        "x(PH 7)": 7.0,
        "gph1 (pH 7)": 1.0,
    }
    for text, value in cases.items():
        m2 = re.search(second, text, re.IGNORECASE)
        m1 = re.search(first, text, re.IGNORECASE)
        assert m2 is not None and m1 is not None
        assert float(m1.group(1)) == value
        assert spell.extract_physical_params(text)["ph"] == value
    gph1_second = re.search(second, "gph1 (pH 7)", re.IGNORECASE)
    assert gph1_second is not None and gph1_second.group(1) == "7"


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        pytest.param("∆zta1_vs_wt_H2O2_repl1", "oxidative_stress", id="h2o2"),
        pytest.param("heat shock 37C", "heat_shock", id="heat-shock-synthetic"),
        pytest.param("temperature shift", "heat_shock", id="temp-shift-synthetic"),
        pytest.param("1 M sorbitol", "osmotic_stress", id="sorbitol-synthetic"),
        pytest.param("0.1% MMS", "dna_damage", id="mms-synthetic"),
        pytest.param("tunicamycin 2h", "er_stress", id="tunicamycin-synthetic"),
        pytest.param("calcofluor white", "cell_wall_stress", id="calcofluor-synth"),
        pytest.param(
            "heat shock + H2O2", "heat_shock", id="dict-order-heat-first-synthetic"
        ),
        pytest.param(_NO_MATCH, None, id="none"),
    ],
)
def test_extract_stress_info(condition: str, expected: str | None) -> None:
    """Stress type is the first dict entry (heat, oxidative, osmotic, dna, er, wall) hit."""
    assert spell.extract_stress_info(condition) == {"stress_type": expected}


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        pytest.param("glg1-del-1-a", ("G1", None), id="finding-glg1-is-G1"),
        pytest.param("hog1 (haploid)", ("G1", None), id="finding-hog1-is-G1"),
        pytest.param("glg2-del-1-a", ("G2", None), id="finding-glg2-is-G2"),
        pytest.param("cells in S phase", ("S", None), id="S-phase-synthetic"),
        pytest.param("mitosis", ("M", None), id="mitosis-synthetic"),
        pytest.param(
            "alpha factor arrest", (None, "alpha_factor"), id="alpha-factor-synthetic"
        ),
        pytest.param(
            "α-factor release", (None, "alpha_factor"), id="greek-alpha-synthetic"
        ),
        pytest.param("elutriation", (None, "elutriation"), id="elutriation-synthetic"),
        pytest.param("cdc15 block", (None, "cdc_arrest"), id="cdc15-synthetic"),
        pytest.param("cdc73-del-1-a", (None, None), id="cdc73-is-not-cdc15"),
        pytest.param(_NO_MATCH, (None, None), id="none"),
    ],
)
def test_extract_cell_cycle_info(
    condition: str, expected: tuple[str | None, str | None]
) -> None:
    """Phase and synchronization method, first dict entry by substring.

    Finding: the G1 and G2 keywords are bare ``"g1"`` / ``"g2"`` substrings
    (spell.py:622, 624), so real deletion names glg1, hog1, mig1, gpg1, glg2, mig2 are
    tagged as cell-cycle phases (102 of the 2,804 distinct real labels). Pinned until the phase
    keywords require word boundaries.
    """
    phase, method = expected
    assert spell.extract_cell_cycle_info(condition) == {
        "cell_cycle_phase": phase,
        "synchronization_method": method,
    }


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        pytest.param("delta rnh201 #1SNM106", (1, "unknown"), id="hash"),
        pytest.param("CI25E delta Cot1 replicate 2", (2, "unknown"), id="replicate"),
        pytest.param(
            "hmt1[delta], biological replicate 3", (3, "biological"), id="biological"
        ),
        pytest.param("gal11∆ med3∆ YPD rep1", (1, "unknown"), id="number-at-end"),
        pytest.param("tech rep 2", (2, "technical"), id="tech-rep-synthetic"),
        pytest.param("bio rep 2", (2, "biological"), id="bio-rep-synthetic"),
        pytest.param("technical replicate 1", (1, "technical"), id="technical-synth"),
        pytest.param(
            "GAL-SIC1Δ3P_biological_rep1_20min",
            (None, "biological"),
            id="rep1-glued-not-at-end",
        ),
        pytest.param("erg6", (6, "unknown"), id="finding-gene-number-is-replicate"),
        pytest.param("elm1-del-3-a", (None, None), id="none"),
    ],
)
def test_extract_replicate_info(
    condition: str, expected: tuple[int | None, str | None]
) -> None:
    """Replicate number: ``#N``, then ``rep[licate] N``, then a trailing number.

    Finding: the trailing-number pattern (spell.py:658) takes any final digits, so a
    bare gene name like "erg6" (real; also clb6, fre6, cat8, ecm10) is replicate 6 of
    type "unknown". 440 of the 2,804 distinct real labels take their replicate number
    from trailing digits alone (no ``#N`` and no ``rep N``); 135 of those are bare
    gene names. Pinned until the trailing pattern requires a replicate marker.
    """
    number, kind = expected
    assert spell.extract_replicate_info(condition) == {
        "replicate_number": number,
        "replicate_type": kind,
    }


# ---------------------------------------------------------------------------
# export_condition_metadata
# ---------------------------------------------------------------------------

_EXPORT_COLUMNS = [
    "study_name",
    "dataset_name",
    "condition_name",
    "condition_index",
    "n_genes",
    "categories",
    "primary_category",
    "secondary_categories",
    "time_min",
    "is_timeseries",
    "temperature_c",
    "chemical_name",
    "concentration",
    "concentration_unit",
    "carbon_source",
    "nitrogen_source",
    "limitation_type",
    "ph",
    "oxygen_level",
    "stress_type",
    "cell_cycle_phase",
    "synchronization_method",
    "replicate_number",
    "replicate_type",
    "extraction_confidence",
    "needs_manual_review",
]


def _md(conditions: list[str], n_genes: int) -> dict[str, Any]:
    """Metadata as ``read_pcl_file`` returns it (eweights unused by the consumers)."""
    return {
        "conditions": conditions,
        "eweights": [1.0] * len(conditions),
        "n_genes": n_genes,
        "n_conditions": len(conditions),
    }


def _export_input() -> dict[tuple[str, str], tuple[pd.DataFrame, dict[str, Any]]]:
    """Two studies, six real-vocabulary conditions (the frames are never read)."""
    empty = pd.DataFrame()
    return {
        ("StudyA_2014", "dsA"): (
            empty,
            _md(
                [
                    "ykl020c deletion",
                    "GAL-SIC1Δ3P_biological_rep1_20min",
                    "Aerobic/hypoxic expression in hap1 deletion S.cerevisiae rep1",
                ],
                2,
            ),
        ),
        ("StudyB_2004", "dsB"): (
            empty,
            _md(
                [
                    "1234",
                    "Htb1_K123R;htb2_delta0 vs. htb2_delta0 (array1)",
                    "Yeast gcr1 null mutant in galactose rep1",
                ],
                4,
            ),
        ),
    }


def _row(
    study: str,
    dataset: str,
    condition: str,
    idx: int,
    n_genes: int,
    categories: list[str],
    confidence: float,
    review: bool,
    **fields: Any,
) -> dict[str, Any]:
    """One expected export row: every parser field None/False unless given."""
    row: dict[str, Any] = {
        "study_name": study,
        "dataset_name": dataset,
        "condition_name": condition,
        "condition_index": idx,
        "n_genes": n_genes,
        "categories": "|".join(categories),
        "primary_category": categories[0],
        "secondary_categories": "|".join(categories[1:]),
        "time_min": None,
        "is_timeseries": False,
        "temperature_c": None,
        "chemical_name": None,
        "concentration": None,
        "concentration_unit": None,
        "carbon_source": None,
        "nitrogen_source": None,
        "limitation_type": None,
        "ph": None,
        "oxygen_level": None,
        "stress_type": None,
        "cell_cycle_phase": None,
        "synchronization_method": None,
        "replicate_number": None,
        "replicate_type": None,
        "extraction_confidence": confidence,
        "needs_manual_review": review,
    }
    row.update(fields)
    return row


def _expected_export_rows() -> list[dict[str, Any]]:
    """The six rows, derived by hand.

    - "ykl020c deletion": category mutant_strain ("deletion"); k=1 (temperature 20.0
      from the ORF suffix) -> 0.157 < 0.2 -> review.
    - "..._rep1_20min": time_series ("min"); k=2 (time 20, concentration 20 "m")
      -> 0.214; review because a concentration has no chemical name.
    - "Aerobic/hypoxic ...": osmotic_stress ("hypo" in "hypoxic") then mutant_strain;
      k=2 (oxygen aerobic, replicate 1) -> 0.214; no review.
    - "1234": no keyword, all digits -> numeric_id; k=1 (replicate 1234) -> 0.157 -> review.
    - "... (array1)": no keyword, contains "array" -> array_id; k=0 -> 0.1 -> review.
    - "Yeast gcr1 null mutant in galactose rep1": mutant_strain; k=3 (chemical, carbon,
      replicate) -> 0.271; no review.
    """
    a, b = ("StudyA_2014", "dsA"), ("StudyB_2004", "dsB")
    return [
        _row(
            *a,
            "ykl020c deletion",
            0,
            2,
            ["mutant_strain"],
            0.157,
            True,
            temperature_c=20.0,
        ),
        _row(
            *a,
            "GAL-SIC1Δ3P_biological_rep1_20min",
            1,
            2,
            ["time_series"],
            0.214,
            True,
            time_min=20.0,
            concentration=20.0,
            concentration_unit="m",
            replicate_type="biological",
        ),
        _row(
            *a,
            "Aerobic/hypoxic expression in hap1 deletion S.cerevisiae rep1",
            2,
            2,
            ["osmotic_stress", "mutant_strain"],
            0.214,
            False,
            oxygen_level="aerobic",
            replicate_number=1,
            replicate_type="unknown",
        ),
        _row(
            *b,
            "1234",
            0,
            4,
            ["numeric_id"],
            0.157,
            True,
            replicate_number=1234,
            replicate_type="unknown",
        ),
        _row(
            *b,
            "Htb1_K123R;htb2_delta0 vs. htb2_delta0 (array1)",
            1,
            4,
            ["array_id"],
            0.1,
            True,
        ),
        _row(
            *b,
            "Yeast gcr1 null mutant in galactose rep1",
            2,
            4,
            ["mutant_strain"],
            0.271,
            False,
            chemical_name="galactose",
            carbon_source="galactose",
            replicate_number=1,
            replicate_type="unknown",
        ),
    ]


def test_export_condition_metadata_exact_rows_and_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Returned frame, the written CSV, and the printed category breakdown are exact.

    Category breakdown over 6 rows: mutant_strain 2 (33.3%), the other four 1 each
    (16.7%), in ``value_counts`` order (count descending, then first appearance). Needs
    manual review: 4 of 6 (66.7%). Mean confidence (0.157+0.214+0.214+0.157+0.1+0.271)/6
    = 1.113/6 = 0.1855, which is 0.18549999... in float -> printed "0.185"
    (``statistics.fmean`` oracle below).
    """
    out_path = tmp_path / "conditions.csv"
    df = spell.export_condition_metadata(_export_input(), output_path=str(out_path))
    expected = pd.DataFrame(_expected_export_rows(), columns=_EXPORT_COLUMNS)
    pd.testing.assert_frame_equal(df, expected)

    lines = out_path.read_text().splitlines()
    assert lines[0] == ",".join(_EXPORT_COLUMNS)
    assert lines[1] == (
        "StudyA_2014,dsA,ykl020c deletion,0,2,mutant_strain,mutant_strain,,,False,20.0,"
        ",,,,,,,,,,,,,0.157,True"
    )
    assert len(lines) == 7

    printed = capsys.readouterr().out
    mean = statistics.fmean([0.157, 0.214, 0.214, 0.157, 0.1, 0.271])
    for line in [
        "\n✓ Exported 6 condition records to: " + str(out_path),
        "  mutant_strain            :      2 ( 33.3%)",
        "  time_series              :      1 ( 16.7%)",
        "  numeric_id               :      1 ( 16.7%)",
        "  array_id                 :      1 ( 16.7%)",
        "  osmotic_stress           :      1 ( 16.7%)",
        f"  Mean confidence:          {mean:.3f}",
        "  Needs manual review:           4 conditions ( 66.7%)",
        "  Temperature:                   1 conditions ( 16.7%)",
    ]:
        assert line in printed
    assert f"{mean:.3f}" == "0.185"


def test_export_condition_metadata_confidence_comment_overstates(
    tmp_path: Path,
) -> None:
    """Finding: the confidence comment says 5 fields -> 0.5; the formula gives 0.386.

    spell.py:843 documents "0 fields = 0.1, 5 fields = 0.5, 10+ fields = 0.9+", but
    the code is ``0.1 + k / 14 * 0.8``: 5 -> 0.1 + 0.2857 = 0.386, 10 -> 0.671, and
    0.9 needs all 14. A condition built from fields that each parser fills
    independently: "0.5 mM H2O2 37°C 30 min time course" ->
    time_min 30, is_timeseries, temperature 37, chemical H2O2, concentration 0.5
    (oxidative stress is a sixth), so k = 6 -> 0.1 + 6/14*0.8 = 0.443. Pinned until the
    comment or the formula is changed to agree.
    """
    condition = "0.5 mM H2O2 37°C 30 min time course"
    all_data = {("S", "D"): (pd.DataFrame(), _md([condition], 1))}
    df = spell.export_condition_metadata(all_data, output_path=str(tmp_path / "c.csv"))
    row = df.iloc[0]
    assert (row["time_min"], row["is_timeseries"], row["temperature_c"]) == (
        30.0,
        True,
        37.0,
    )
    assert (row["chemical_name"], row["concentration"], row["stress_type"]) == (
        "H2O2",
        0.5,
        "oxidative_stress",
    )
    assert row["extraction_confidence"] == round(0.1 + 6 / 14 * 0.8, 3) == 0.443


def test_export_condition_metadata_all_fourteen_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A synthetic label that fills all 14 counted fields scores the 0.9 cap.

    "0.5 mM H2O2 37°C 30 min time course pH 5 anaerobic proline glucose N-lim G1 phase
    alpha factor rep 2": time 30, time course, 37, H2O2, 0.5 mM, glucose, proline,
    nitrogen_limited, pH 5, anaerobic, oxidative (h2o2), G1, alpha_factor, replicate 2
    -> k = 14 -> min(0.9, 0.1 + 0.8) = 0.9, no review. 1,000 copies of it also reach
    the progress line printed every 1,000 conditions (spell.py:779).
    """
    label = (
        "0.5 mM H2O2 37°C 30 min time course pH 5 anaerobic proline glucose N-lim "
        "G1 phase alpha factor rep 2"
    )
    all_data = {("S", "D"): (pd.DataFrame(), _md([label] * 1000, 1))}
    df = spell.export_condition_metadata(all_data, output_path=str(tmp_path / "c.csv"))
    fields = [
        "time_min", "is_timeseries", "temperature_c", "chemical_name", "concentration",
        "concentration_unit", "carbon_source", "nitrogen_source", "limitation_type",
        "ph", "oxygen_level", "stress_type", "cell_cycle_phase",
        "synchronization_method", "replicate_number", "replicate_type",
        "extraction_confidence", "needs_manual_review",
    ]  # fmt: skip
    assert df[fields].iloc[0].tolist() == [
        30.0, True, 37.0, "H2O2", 0.5, "mM", "glucose", "proline", "nitrogen_limited",
        5.0, "anaerobic", "oxidative_stress", "G1", "alpha_factor", 2, "unknown", 0.9,
        False,
    ]  # fmt: skip
    assert len(df) == 1000
    assert "  Processed 1,000/1,000 conditions...\n" in capsys.readouterr().out


def test_export_condition_metadata_default_path_ignores_env_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: the default output path is under a hard-coded ``~/Documents/projects/torchcell``.

    spell.py:25 sets the module ``DATA_ROOT`` from ``expanduser`` even though the
    module calls ``load_dotenv()``; the ``DATA_ROOT`` environment variable (set to the
    test sentinel by tests/conftest.py) is never read. With ``output_path=None`` the CSV
    goes to ``<module DATA_ROOT>/data/sgd/spell/spell_conditions_metadata_enhanced.csv``,
    not the docstring's ``DATA_ROOT/spell_conditions_metadata.csv``. The module constant
    is redirected to ``tmp_path`` here so nothing is written to the home directory.
    Pinned until the module reads ``os.environ["DATA_ROOT"]``.
    """
    assert os.environ["DATA_ROOT"] != spell.DATA_ROOT
    assert spell.DATA_ROOT == osp.expanduser(
        osp.join("~", "Documents", "projects", "torchcell")
    )
    monkeypatch.setattr(spell, "DATA_ROOT", str(tmp_path))
    (tmp_path / "data" / "sgd" / "spell").mkdir(parents=True)
    all_data = {("S", "D"): (pd.DataFrame(), _md(["erg6"], 1))}
    df = spell.export_condition_metadata(all_data)
    written = pd.read_csv(
        tmp_path / "data" / "sgd" / "spell" / "spell_conditions_metadata_enhanced.csv"
    )
    assert (
        written["condition_name"].tolist() == df["condition_name"].tolist() == ["erg6"]
    )
    assert written["replicate_number"].tolist() == [6]


def test_export_condition_metadata_category_substrings_on_real_names(
    tmp_path: Path,
) -> None:
    """Finding: category keywords are bare substrings, so real gene names pick categories.

    spell.py:718 ``"hs"`` makes "hsp12-del-1-a" heat_shock; spell.py:761 ``"dna"``
    makes a reference-channel note "... yeast genomic DNA ..." dna_damage; spell.py:734
    ``"hypo"`` makes "hypoxic" osmotic_stress (asserted in the exact-rows test). Pinned
    until the keywords match whole words.
    """
    conditions = [
        "hsp12-del-1-a",
        "prp17 null cells, time 5min; src: prp17 null, 5min CH1 Cy5<->yeast genomic "
        "DNA CH2 Cy3",
    ]
    all_data = {("S", "D"): (pd.DataFrame(), _md(conditions, 1))}
    df = spell.export_condition_metadata(all_data, output_path=str(tmp_path / "c.csv"))
    assert df["categories"].tolist() == ["heat_shock", "dna_damage|time_series"]


def test_export_condition_metadata_empty_input_raises(tmp_path: Path) -> None:
    """No conditions: the empty frame has no ``primary_category`` column, so it raises."""
    with pytest.raises(KeyError, match="^'primary_category'$"):
        spell.export_condition_metadata({}, output_path=str(tmp_path / "c.csv"))


# ---------------------------------------------------------------------------
# check_condition_metadata_quality
# ---------------------------------------------------------------------------


def test_check_condition_metadata_quality_exact_stats(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each label falls in the first matching bucket: empty, generic, numeric, descriptive.

    - "" and "   " are empty (example carries the study/dataset prefix).
    - generic needs a generic word AND at most two words: "Condition 1", "sample",
      "chip7", "Array A", "experiment B", "condition X" -> 6 generic, 5 examples kept.
    - "sample A B" has three words, so it is not generic, and not numeric -> descriptive.
    - "3", "-1.5" (dots and dashes stripped before ``isdigit``) are numeric.
    - "hog1D +KCl rep1" and the 93-character real prp17 label (truncated to 60) are
      descriptive.
    Totals: 13 = 2 empty + 6 generic + 2 numeric + 3 descriptive.
    """
    long_label = (
        "prp17 null cells, time 0 hours; src: prp17 null, 0 hours CH1 Cy5<->yeast "
        "genomic DNA CH2 Cy3"
    )
    conds = [
        "",
        "   ",
        "Condition 1",
        "sample",
        "chip7",
        "Array A",
        "experiment B",
        "condition X",
        "sample A B",
        "3",
        "-1.5",
        "hog1D +KCl rep1",
        long_label,
    ]
    all_data = {("Study", "ds"): (pd.DataFrame(), _md(conds, 1))}
    stats = spell.check_condition_metadata_quality(all_data)
    assert stats == {
        "total_conditions": 13,
        "empty_conditions": 2,
        "generic_conditions": 6,
        "numeric_only_conditions": 2,
        "descriptive_conditions": 3,
        "examples": {
            "empty": ["Study/ds: ''", "Study/ds: '   '"],
            "generic": ["Condition 1", "sample", "chip7", "Array A", "experiment B"],
            "numeric": ["3", "-1.5"],
            "descriptive": ["sample A B", "hog1D +KCl rep1", long_label[:60]],
        },
    }
    printed = capsys.readouterr().out
    assert "  ✓ Descriptive labels: 3 (23.1%)\n" in printed
    assert "  ⚠ Generic labels: 6 (46.2%)\n" in printed
    assert "  ⚠ Numeric-only labels: 2 (15.4%)\n" in printed
    assert "  ✗ Empty labels: 2 (15.4%)\n" in printed
    assert "\nExample empty conditions:\n  Study/ds: ''\n  Study/ds: '   '\n" in printed


def test_check_condition_metadata_quality_caps_examples_at_five(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Six labels of each bucket: counts are 6, each example list stops at 5.

    Empty examples are formatted ``"<study>/<dataset>: '<label>'"``; numeric and
    descriptive keep the label. With a second all-descriptive study, only that bucket's
    example block prints for it.
    """
    conds = (
        [""] * 6 + [str(i) for i in range(6)] + [f"hog1D +KCl rep{i}" for i in range(6)]
    )
    stats = spell.check_condition_metadata_quality(
        {("St", "ds"): (pd.DataFrame(), _md(conds, 1))}
    )
    assert (
        stats["empty_conditions"],
        stats["numeric_only_conditions"],
        stats["descriptive_conditions"],
        stats["generic_conditions"],
    ) == (6, 6, 6, 0)
    assert stats["examples"] == {
        "empty": ["St/ds: ''"] * 5,
        "generic": [],
        "numeric": ["0", "1", "2", "3", "4"],
        "descriptive": [f"hog1D +KCl rep{i}" for i in range(5)],
    }
    capsys.readouterr()
    only = spell.check_condition_metadata_quality(
        {("St", "ds"): (pd.DataFrame(), _md(["hog1D +KCl rep1"], 1))}
    )
    printed = capsys.readouterr().out
    assert only["descriptive_conditions"] == 1
    assert "Example empty" not in printed
    assert "Example generic" not in printed
    assert "Example numeric" not in printed
    assert "\nExample descriptive conditions:\n  'hog1D +KCl rep1'\n" in printed


def test_check_condition_metadata_quality_empty_input_divides_by_zero() -> None:
    """Finding: with no conditions the percentage lines divide by zero.

    ``100 * stats['descriptive_conditions'] / stats['total_conditions']`` with total 0
    (spell.py:1043) raises before the stats are returned. Pinned until the summary
    guards an empty input.
    """
    with pytest.raises(ZeroDivisionError, match="^division by zero$"):
        spell.check_condition_metadata_quality({})


# ---------------------------------------------------------------------------
# extract_and_load_all_spell_studies
# ---------------------------------------------------------------------------


def _make_zip(root: Path, study: str, files: dict[str, str]) -> None:
    """Write ``<root>/<study>.zip`` holding ``<study>/<name>`` for each file."""
    with zipfile.ZipFile(root / f"{study}.zip", "w") as zf:
        for name, text in files.items():
            zf.writestr(f"{study}/{name}", text)


def _two_study_root(root: Path) -> Path:
    """Alpha (two datasets a1, a2) and Beta (one dataset b1)."""
    root.mkdir(parents=True, exist_ok=True)
    _make_zip(root, "Alpha_2001_PMID_1", {"a2.pcl": _PCL_B, "a1.pcl": _PCL})
    _make_zip(root, "Beta_2002_PMID_2", {"b1.pcl": _PCL_B})
    return root


def _expected_b_frame() -> pd.DataFrame:
    """``_PCL_B`` by hand: c2 is all empty, so it is float NaN."""
    return pd.DataFrame(
        {
            "NAME": ["TFC3"],
            "GWEIGHT": np.array([1], dtype=np.int64),
            "c1": [2.0],
            "c2": [np.nan],
        },
        index=pd.Index(["YAL001C"], name="YORF"),
    )


def test_extract_and_load_all_spell_studies_two_studies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both archives are extracted; keys follow sorted zip, then sorted PCL order."""
    root = _two_study_root(tmp_path / "spell")
    data = spell.extract_and_load_all_spell_studies(str(root))
    assert list(data) == [
        ("Alpha_2001_PMID_1", "a1"),
        ("Alpha_2001_PMID_1", "a2"),
        ("Beta_2002_PMID_2", "b1"),
    ]
    pd.testing.assert_frame_equal(
        data[("Alpha_2001_PMID_1", "a1")][0], _expected_fixture_frame()
    )
    pd.testing.assert_frame_equal(
        data[("Beta_2002_PMID_2", "b1")][0], _expected_b_frame()
    )
    assert data[("Alpha_2001_PMID_1", "a2")][1] == {
        "conditions": ["c1", "c2"],
        "eweights": [1.0, 1.0],
        "n_genes": 1,
        "n_conditions": 2,
    }
    assert (root / "Alpha_2001_PMID_1" / "a1.pcl").read_text() == _PCL
    assert capsys.readouterr().out == (
        "Found 2 study archives\n"
        "\n[1/2] Processing: Alpha_2001_PMID_1\n"
        "  Extracting...\n"
        "  ✓ Loaded 2 dataset(s)\n"
        "\n[2/2] Processing: Beta_2002_PMID_2\n"
        "  Extracting...\n"
        "  ✓ Loaded 1 dataset(s)\n"
    )


def test_extract_and_load_studies_to_load_and_max_studies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``studies_to_load`` filters archives; ``max_studies=1`` stops after the first."""
    root = _two_study_root(tmp_path / "spell")
    only_beta = spell.extract_and_load_all_spell_studies(
        str(root), studies_to_load=["Beta_2002_PMID_2"]
    )
    assert list(only_beta) == [("Beta_2002_PMID_2", "b1")]
    pd.testing.assert_frame_equal(
        only_beta[("Beta_2002_PMID_2", "b1")][0], _expected_b_frame()
    )
    capsys.readouterr()

    first = spell.extract_and_load_all_spell_studies(str(root), max_studies=1)
    assert list(first) == [("Alpha_2001_PMID_1", "a1"), ("Alpha_2001_PMID_1", "a2")]
    # Only Beta was extracted by the first call, so Alpha is extracted now.
    assert capsys.readouterr().out == (
        "Found 2 study archives\n"
        "\n[1/1] Processing: Alpha_2001_PMID_1\n"
        "  Extracting...\n"
        "  ✓ Loaded 2 dataset(s)\n"
        "Reached max_studies limit (1)\n"
    )


def test_extract_and_load_max_studies_zero_loads_everything(tmp_path: Path) -> None:
    """Finding: ``max_studies=0`` is falsy, so it means unlimited, not zero studies.

    ``if max_studies and studies_loaded >= max_studies`` (spell.py:229) skips the limit
    for 0, and the progress total falls back to ``len(zip_files)`` (spell.py:226).
    Pinned until the check is ``max_studies is not None``.
    """
    root = _two_study_root(tmp_path / "spell")
    data = spell.extract_and_load_all_spell_studies(str(root), max_studies=0)
    assert list(data) == [
        ("Alpha_2001_PMID_1", "a1"),
        ("Alpha_2001_PMID_1", "a2"),
        ("Beta_2002_PMID_2", "b1"),
    ]


def test_extract_and_load_study_filter_matches_the_root_path(tmp_path: Path) -> None:
    """Finding: ``studies_to_load`` is a substring test on the full zip path.

    spell.py:222 tests ``study in z`` with ``z`` the absolute path, so a token that
    also occurs in the root directory name selects every archive: root
    ``.../Beta_2002_mirror`` with ``studies_to_load=["Beta_2002"]`` loads Alpha too.
    Pinned until the filter compares the archive basename.
    """
    root = _two_study_root(tmp_path / "Beta_2002_mirror")
    data = spell.extract_and_load_all_spell_studies(
        str(root), studies_to_load=["Beta_2002"]
    )
    assert sorted({study for study, _ in data}) == [
        "Alpha_2001_PMID_1",
        "Beta_2002_PMID_2",
    ]


def test_extract_and_load_skips_bad_archives_and_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A corrupt zip, a zip with no PCL, a broken PCL, and an already-extracted study.

    - ``A_bad.zip`` is not a zip: printed error, skipped, not counted toward
      ``max_studies``.
    - ``B_nopcl.zip`` holds only a README: "No PCL files found", not counted.
    - ``C_mixed`` is already extracted on disk with ``good.pcl`` (the ``_PCL_B``
      content) and ``bad.pcl`` (no YORF column); its zip holds a different file
      (``zip_only.pcl``) that is never extracted because the directory exists.
    With ``max_studies=1``, C is the first study counted, so the loop ends after it.
    """
    root = tmp_path / "spell"
    root.mkdir()
    (root / "A_bad.zip").write_bytes(b"not a zip")
    _make_zip(root, "B_nopcl", {"README.txt": "nothing"})
    _make_zip(root, "C_mixed", {"zip_only.pcl": _PCL})
    _make_zip(root, "D_after", {"d.pcl": _PCL})
    (root / "C_mixed").mkdir()
    (root / "C_mixed" / "good.pcl").write_text(_PCL_B)
    (root / "C_mixed" / "bad.pcl").write_text("ORF\tNAME\nEWEIGHT\t\nX\tY\n")

    data = spell.extract_and_load_all_spell_studies(str(root), max_studies=1)
    assert list(data) == [("C_mixed", "good")]
    pd.testing.assert_frame_equal(data[("C_mixed", "good")][0], _expected_b_frame())
    assert not (root / "C_mixed" / "zip_only.pcl").exists()
    assert capsys.readouterr().out == (
        "Found 4 study archives\n"
        "\n[1/1] Processing: A_bad\n"
        "  Extracting...\n"
        "  ✗ Error extracting: File is not a zip file\n"
        "\n[2/1] Processing: B_nopcl\n"
        "  Extracting...\n"
        "  ⚠ No PCL files found\n"
        "\n[3/1] Processing: C_mixed\n"
        "  ✗ bad: \"None of ['YORF'] are in the columns\"\n"
        "  ✓ Loaded 1 dataset(s)\n"
        "Reached max_studies limit (1)\n"
    )


# ---------------------------------------------------------------------------
# Plot functions (Agg backend from tests/conftest.py; Axes.hist recorded)
# ---------------------------------------------------------------------------

HistCall = tuple[Axes, npt.NDArray[np.float64], dict[str, Any]]


@pytest.fixture
def hist_calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[HistCall]]:
    """Record every ``Axes.hist`` call (axes, data, kwargs); stub ``plt.show``."""
    calls: list[HistCall] = []
    original = Axes.hist

    def recorder(self: Axes, x: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append((self, np.asarray(x, dtype=float).copy(), dict(kwargs)))
        return original(self, x, *args, **kwargs)

    monkeypatch.setattr(Axes, "hist", recorder)
    monkeypatch.setattr(plt, "show", lambda: None)
    yield calls
    plt.close("all")


_HIST_STYLE = {"edgecolor": "black", "alpha": 0.7, "color": "steelblue"}


def test_plot_expression_histograms_default_genes(  # test-quality: allow output is read from the hist_calls recorder fixture
    tmp_path: Path, hist_calls: list[HistCall]
) -> None:
    """Default gene list is the first five ORFs (three here); NaN is dropped per gene."""
    df, metadata = spell.read_pcl_file(_write(tmp_path / "a.pcl", _PCL))
    spell.plot_expression_histograms(df, metadata)
    assert [c[2] for c in hist_calls] == [{"bins": 30, **_HIST_STYLE}] * 3
    np.testing.assert_array_equal(hist_calls[0][1], [0.12, 0.45, 1.23])
    np.testing.assert_array_equal(hist_calls[1][1], [-0.34, 0.89])
    np.testing.assert_array_equal(hist_calls[2][1], [0.5, 0.25, -1.0])
    assert [c[0].get_title() for c in hist_calls] == [
        "YAL001C (TFC3) - 3 conditions",
        "YAL002W (VPS8) - 2 conditions",
        "YAL003W (EFB1) - 3 conditions",
    ]
    assert plt.gcf().get_suptitle() == "SPELL - Gene Expression Distributions"


def test_plot_expression_histograms_single_gene_saves(
    tmp_path: Path,
    hist_calls: list[HistCall],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing ORF is filtered; one gene takes the single-axes branch; savefig args."""
    saved: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(plt, "savefig", lambda path, **kw: saved.append((path, kw)))
    df, metadata = spell.read_pcl_file(_write(tmp_path / "a.pcl", _PCL))
    out = str(tmp_path / "h.png")
    spell.plot_expression_histograms(
        df, metadata, gene_list=["YBR999W", "YAL002W"], title_prefix="X", save_path=out
    )
    assert len(hist_calls) == 1
    np.testing.assert_array_equal(hist_calls[0][1], [-0.34, 0.89])
    assert saved == [(out, {"dpi": 300, "bbox_inches": "tight"})]
    assert capsys.readouterr().out == f"Saved figure to: {out}\n"


def test_plot_expression_histograms_no_gene_found(
    tmp_path: Path, hist_calls: list[HistCall], capsys: pytest.CaptureFixture[str]
) -> None:
    """No requested gene is present: a message and no histogram."""
    df, metadata = spell.read_pcl_file(_write(tmp_path / "a.pcl", _PCL))
    spell.plot_expression_histograms(df, metadata, gene_list=["YBR999W"])
    assert hist_calls == []
    assert capsys.readouterr().out == "None of the specified genes found in dataset\n"


def test_plot_global_expression_distribution(  # test-quality: allow output is read from the hist_calls recorder fixture
    tmp_path: Path,
    hist_calls: list[HistCall],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """All 9 cells flattened row-major minus the one NaN: 8 values, 100 bins; saved.

    Values [0.12, 0.45, 1.23, -0.34, 0.89, 0.5, 0.25, -1.0]: sum 2.1, mean 0.2625;
    sorted middle pair (0.25, 0.45) -> median 0.35; population std (np.std, ddof 0) is
    checked against ``statistics.pstdev``.
    """
    saved: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(plt, "savefig", lambda path, **kw: saved.append((path, kw)))
    df, metadata = spell.read_pcl_file(_write(tmp_path / "a.pcl", _PCL))
    out = str(tmp_path / "global.png")
    spell.plot_global_expression_distribution(df, metadata, save_path=out)
    assert saved == [(out, {"dpi": 300, "bbox_inches": "tight"})]
    assert capsys.readouterr().out == f"Saved figure to: {out}\n"
    values = [0.12, 0.45, 1.23, -0.34, 0.89, 0.5, 0.25, -1.0]
    assert len(hist_calls) == 1
    ax, data, kwargs = hist_calls[0]
    np.testing.assert_array_equal(data, values)
    assert kwargs == {"bins": 100, **_HIST_STYLE}
    assert ax.get_title() == (
        "SPELL - Global Expression Distribution\n3 genes × 3 conditions = 8 measurements"
    )
    assert [t.get_text() for t in ax.texts] == [
        f"Mean: {statistics.fmean(values):.3f}\n"
        f"Median: {statistics.median(values):.3f}\n"
        f"Std: {statistics.pstdev(values):.3f}"
    ]
    assert statistics.median(values) == 0.35


def test_plot_genes_across_all_studies(  # test-quality: allow output is read from the hist_calls recorder fixture
    tmp_path: Path, hist_calls: list[HistCall]
) -> None:
    """Values pool across datasets in dict order; a gene with no data gets a text panel.

    Alpha/a1 is the fixture, Beta/b1 is ``_PCL_B`` (YAL001C only, c2 empty). YAL001C
    pools [0.12, 0.45, 1.23] + [2.0] = 4 values. Total conditions 3 + 2 = 5.

    Finding: the per-gene title says "across {total_studies} studies" with
    ``total_studies`` = every loaded study (spell.py:342), so YAL002W, measured only in
    Alpha, is titled "across 2 studies". Pinned until the count is per gene.
    """
    all_data = {
        ("Alpha", "a1"): spell.read_pcl_file(_write(tmp_path / "a1.pcl", _PCL)),
        ("Beta", "b1"): spell.read_pcl_file(_write(tmp_path / "b1.pcl", _PCL_B)),
    }
    spell.plot_genes_across_all_studies(all_data, ["YAL001C", "YAL002W", "YCR999C"])
    assert len(hist_calls) == 2
    np.testing.assert_array_equal(hist_calls[0][1], [0.12, 0.45, 1.23, 2.0])
    np.testing.assert_array_equal(hist_calls[1][1], [-0.34, 0.89])
    assert [c[2] for c in hist_calls] == [{"bins": 100, **_HIST_STYLE}] * 2
    axes = hist_calls[0][0].figure.axes
    assert [ax.get_title() for ax in axes] == [
        "YAL001C (TFC3) - 4 measurements across 2 studies",
        "YAL002W (VPS8) - 2 measurements across 2 studies",
        "YCR999C - No expression data available",
    ]
    assert [t.get_text() for t in axes[2].texts] == ["No data found for YCR999C"]
    pooled = [0.12, 0.45, 1.23, 2.0]
    assert [t.get_text() for t in axes[0].texts] == [
        f"Mean: {statistics.fmean(pooled):.3f}\n"
        f"Median: {statistics.median(pooled):.3f}\n"
        f"Std: {statistics.pstdev(pooled):.3f}"
    ]
    assert plt.gcf().get_suptitle() == (
        "SPELL Database - Gene Expression Across Multiple Studies\n"
        "2 studies, 2 datasets, 5 total conditions"
    )


def test_plot_genes_across_all_studies_single_gene_saves(
    tmp_path: Path,
    hist_calls: list[HistCall],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One gene takes the single-axes branch; savefig gets the path, dpi 300, tight."""
    saved: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(plt, "savefig", lambda path, **kw: saved.append((path, kw)))
    all_data = {("Beta", "b1"): spell.read_pcl_file(_write(tmp_path / "b.pcl", _PCL_B))}
    out = str(tmp_path / "g.png")
    spell.plot_genes_across_all_studies(all_data, ["YAL001C"], save_path=out)
    assert len(hist_calls) == 1
    np.testing.assert_array_equal(hist_calls[0][1], [2.0])
    assert saved == [(out, {"dpi": 300, "bbox_inches": "tight"})]
    assert capsys.readouterr().out == f"\n✓ Saved figure to: {out}\n"
