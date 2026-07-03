# tests/torchcell/literature/test_citation_keys.py
# [[tests.torchcell.literature.test_citation_keys]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_citation_keys.py

"""Tests for torchcell.literature.citation_keys."""

from torchcell.literature.citation_keys import generate_citation_key


def test_basic_key():
    assert (
        generate_citation_key(["John Smith"], "2022", "Biodegradation of PET")
        == "smithBiodegradationPET2022"
    )


def test_skipwords_and_five_word_cap():
    # Leading "A" and "a" are skipwords; title segment caps at five words.
    key = generate_citation_key(
        ["Michael Costanzo"],
        "2016",
        "A global genetic interaction network maps a wiring diagram",
    )
    assert key == "costanzoGlobalGeneticInteractionNetworkMaps2016"


def test_unicode_surname_transliterated():
    assert (
        generate_citation_key(["Müller"], "2020", "Enzyme design")
        == "mullerEnzymeDesign2020"
    )


def test_missing_metadata_falls_back():
    # No creators -> "unknown"; no 4-digit year -> "XXXX".
    assert generate_citation_key([], "", "") == "unknownXXXX"


def test_deterministic():
    args = (["Jane Doe", "John Roe"], "2018-05", "Systematic complex interactions")
    assert generate_citation_key(*args) == generate_citation_key(*args)
