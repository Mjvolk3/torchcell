# tests/torchcell/datasets/scerevisiae/test_media_stubs_issue622.py
# [[tests.torchcell.datasets.scerevisiae.test_media_stubs_issue622]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_media_stubs_issue622.py
"""Issue #622: seven loaders emitted four name-only Media stubs into the served store.

The census of release 2026.10.02-833970cd found ``SC`` liquid (Caudal 2024), ``SC-URA``
solid (Ozaydin 2013), ``YEPD`` solid (SynLethDB SL and SR) and ``YPD`` liquid (Ohya 2005,
Ohnuki 2018, Nadal-Ribelles 2025, Yoshida 2012, da Silveira 2014) with no components and
no provenance. Each loader now emits a sourced medium object. Pinned here:

- the five "paper names the library medium" loaders carry the library object's exact
  composition (``media_identity`` equal to ``SC`` / ``YPD_LIQUID``) and the library
  provenance followed by the paper's own sentence(s), in that order;
- da Silveira's YPD is its own printed recipe (yeast extract and peptone swapped against
  the library YPD, MES buffer, Trp/Ura/Ade), so its identity is NOT ``YPD_LIQUID``'s,
  and the tryptophan amount printed as "40 mg/ml" is held as an open gap;
- SynLethDB carries no medium and no temperature, so its environment is a
  composition-deferred placeholder medium plus a ``not_carried_by_curation`` gap on
  ``temperature``;
- every loader-side quote is a verbatim substring of its pinned mirror file (skipped
  when the mirror is not mounted).
"""

from __future__ import annotations

import os
import os.path as osp

import pytest

from torchcell.datamodels.identity import media_identity
from torchcell.datamodels.media import SC, SC_URA, YPD_LIQUID
from torchcell.datamodels.schema import (
    ComponentDefinition,
    Concentration,
    ConcentrationUnit,
    Media,
)
from torchcell.datasets.scerevisiae import (
    caudal2024,
    dasilveira2014,
    nadal_ribelles2025,
    ohnuki2018,
    ohya2005,
    ozaydin2013,
    synth_leth_db,
    yoshida2012,
)
from torchcell.verification.sourced import (
    ProvenanceGapReason,
    SourcedValue,
    audit_sourced_value,
)

#: (loader medium, the library object it restates, the loader's own statements).
_RESTATED: list[tuple[str, Media, Media, list[SourcedValue]]] = [
    (
        "caudal2024",
        caudal2024.CAUDAL_SC,
        SC,
        list(caudal2024.MEDIUM_SOURCED_VALUES.values()),
    ),
    (
        "ohya2005",
        ohya2005.OHYA_YPD,
        YPD_LIQUID,
        list(ohya2005.MEDIUM_SOURCED_VALUES.values()),
    ),
    (
        "ohnuki2018",
        ohnuki2018.OHNUKI_YPD,
        YPD_LIQUID,
        list(ohnuki2018.MEDIUM_SOURCED_VALUES.values()),
    ),
    (
        "nadal_ribelles2025",
        nadal_ribelles2025.NADAL_RIBELLES_YPD,
        YPD_LIQUID,
        list(nadal_ribelles2025.MEDIUM_SOURCED_VALUES.values()),
    ),
    (
        "yoshida2012",
        yoshida2012.YOSHIDA_YPD,
        YPD_LIQUID,
        list(yoshida2012.MEDIUM_SOURCED_VALUES.values()),
    ),
]


@pytest.mark.parametrize(
    ("loader", "media", "library", "statements"),
    _RESTATED,
    ids=[row[0] for row in _RESTATED],
)
def test_restated_medium_is_the_library_composition_plus_the_paper_sentence(
    loader: str, media: Media, library: Media, statements: list[SourcedValue]
) -> None:
    """Same composition as the library object, provenance = library + paper."""
    assert media_identity(media) == media_identity(library), loader
    assert media.name == library.name
    assert media.components == library.components
    assert media.provenance == [*library.provenance, *statements]
    assert all(sv.provenance.source_uri for sv in statements)


def test_restated_statement_counts_and_mirror_keys() -> None:
    """Each loader quotes its own paper; Ohya also quotes the Ohnuki 2018 recipe that
    names Ohya 2005 (its ref 15) as the preparation.
    """
    keys = {
        loader: [sv.provenance.citation_key for sv in statements]
        for loader, _, _, statements in _RESTATED
    }
    assert keys == {
        "caudal2024": ["caudalPantranscriptomeRevealsLarge2024"] * 2,
        "ohya2005": [
            "ohyaHighdimensionalLargescalePhenotyping2005",
            "ohnukiHighdimensionalSinglecellPhenotyping2018",
        ],
        "ohnuki2018": ["ohnukiHighdimensionalSinglecellPhenotyping2018"] * 2,
        "nadal_ribelles2025": ["nadal-ribellesSinglecellResolvedGenotypephenotype2025"],
        "yoshida2012": ["yoshidaIdentificationCharacterizationGenes2012"] * 3,
    }


def test_ozaydin_plate_is_sc_ura_made_solid_with_an_unquantified_agar_row() -> None:
    """``SC_URA`` + agar (no amount) on a solid plate: a different identity from the
    liquid library ``SC_URA``, the same base and dropout.
    """
    media = ozaydin2013.OZAYDIN_SC_URA_AGAR
    assert (media.state, media.is_synthetic, media.base_medium) == ("solid", True, "SC")
    assert media.dropouts == SC_URA.dropouts
    assert media.components[:-1] == SC_URA.components
    assert media.components[-1].compound.name == "agar"
    assert media.components[-1].concentration is None
    assert media_identity(media) != media_identity(SC_URA)
    assert media.provenance == [
        *SC_URA.provenance,
        ozaydin2013.SOURCED_VALUES["scoring_medium"],
    ]


def test_da_silveira_ypd_is_its_own_printed_recipe() -> None:
    """Seven rows, amounts exactly as printed; tryptophan, uracil, adenine unquantified."""
    media = dasilveira2014.DA_SILVEIRA_YPD
    assert (media.state, media.is_synthetic, media.base_medium) == (
        "liquid",
        False,
        "YPD",
    )
    pct = ConcentrationUnit.percent_w_v
    assert [(c.compound.name, c.concentration) for c in media.components] == [
        ("D-glucose", Concentration(value=2.0, unit=pct)),
        ("peptone", Concentration(value=1.0, unit=pct)),
        ("yeast extract", Concentration(value=2.0, unit=pct)),
        (
            "2-(N-morpholino)ethanesulfonic acid",
            Concentration(value=10.0, unit=ConcentrationUnit.millimolar),
        ),
        ("L-tryptophan", None),
        ("uracil", None),
        ("adenine", None),
    ]
    assert media_identity(media) != media_identity(YPD_LIQUID)
    assert media.open_gaps == [
        "peptone",
        "yeast extract",
        "L-tryptophan",
        "uracil",
        "adenine",
    ]


def test_synlethdb_environment_carries_no_invented_medium_or_temperature() -> None:
    """The old ``YEPD`` solid at 30 C was a default no SynLethDB field states."""
    env = synth_leth_db.SYNLETHDB_ENVIRONMENT
    assert env.temperature is None
    assert [(g.field, g.reason) for g in env.provenance_gaps] == [
        ("temperature", ProvenanceGapReason.not_carried_by_curation)
    ]
    media = env.media
    assert media.base_medium is None
    assert len(media.components) == 1
    only = media.components[0]
    assert only.definition is ComponentDefinition.composition_deferred
    assert only.concentration is None
    assert media.provenance[0].quote == synth_leth_db.SYNLETHDB_ENTRY_FIELDS_QUOTE
    assert "YEPD" not in media.name


def _all_loader_media() -> list[tuple[str, Media]]:
    return [
        *((loader, media) for loader, media, _, _ in _RESTATED),
        ("ozaydin2013", ozaydin2013.OZAYDIN_SC_URA_AGAR),
        ("dasilveira2014", dasilveira2014.DA_SILVEIRA_YPD),
        ("synth_leth_db", synth_leth_db.SYNLETHDB_MEDIUM_NOT_CARRIED),
    ]


def test_no_loader_medium_is_a_stub() -> None:
    """The census definition of a stub: zero components and zero provenance."""
    for loader, media in _all_loader_media():
        assert media.components, loader
        assert media.provenance, loader


def _library_root() -> str | None:
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        return None
    root = osp.join(data_root, "torchcell-library")
    return root if osp.isdir(root) else None


def test_loader_quotes_are_verbatim_in_the_mirrored_papers() -> None:
    """Every SourcedValue a loader added is still a substring of its pinned bytes."""
    root = _library_root()
    if root is None:
        pytest.skip("torchcell-library mirror not mounted")
    loader_values: list[SourcedValue] = [
        *(sv for _, _, _, statements in _RESTATED for sv in statements),
        *ozaydin2013.SOURCED_VALUES.values(),
        *dasilveira2014.SOURCED_VALUES.values(),
        *synth_leth_db.SYNLETHDB_MEDIUM_NOT_CARRIED.provenance,
    ]
    assert len(loader_values) == 21
    for value in loader_values:
        result = audit_sourced_value(value, root)
        assert result.passed, f"{value.provenance.citation_key}: {result.message}"
