"""Tests for torchcell.verification.sourced (SourcedValue + audit)."""

import pickle

import pytest
from pydantic import BaseModel

from torchcell.verification import (
    Provenance,
    SourcedValue,
    audit_sourced_value,
    library_available,
    sha256_file,
)

# The Kuzmin 2018 SI quote that gives the replicate design (si1.md:29).
KUZMIN_QUOTE = (
    "screened alongside its two single mutant control strains, in two "
    "independent replicates"
)


def _make_library(tmp_path, citation_key, rel, text):
    """Create a fake <library>/<citation_key>/<rel> artifact; return (root, sha)."""
    art = tmp_path / citation_key / rel
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_text(text, encoding="utf-8")
    return tmp_path, sha256_file(art)


def _sourced(citation_key, rel, sha, value=2, quote=KUZMIN_QUOTE):
    return SourcedValue(
        value=value,
        quote=quote,
        provenance=Provenance(
            source_uri=rel, citation_key=citation_key, sha256=sha, page="S-methods"
        ),
    )


class _ProvenanceHolder(BaseModel):
    """Module-level so its instances are picklable (mirrors MediaComponent)."""

    provenance: list[SourcedValue]


# --------------------------------------------------------------------------- #
# Construction requires an auditable provenance
# --------------------------------------------------------------------------- #
def test_requires_citation_key_and_sha():
    with pytest.raises(ValueError):
        SourcedValue(
            value=2, quote="x", provenance=Provenance(source_uri="si/si1.md")
        )  # no citation_key/sha256


def test_requires_nonempty_quote():
    with pytest.raises(ValueError):
        SourcedValue(
            value=2,
            quote="   ",
            provenance=Provenance(
                source_uri="si/si1.md", citation_key="k", sha256="abc"
            ),
        )


def test_value_reads_without_library():
    # The whole point: .value needs nothing but Python.
    sv = _sourced("kuzmin2018", "si/si1.md", "deadbeef")
    assert sv.value == 2
    assert sv.provenance.citation_key == "kuzmin2018"


# --------------------------------------------------------------------------- #
# Audit against the artifact
# --------------------------------------------------------------------------- #
def test_audit_passes_when_hash_and_quote_match(tmp_path):
    text = f"... {KUZMIN_QUOTE}, for a total of 1,092 screens. ..."
    root, sha = _make_library(tmp_path, "kuzmin2018", "si/si1.md", text)
    sv = _sourced("kuzmin2018", "si/si1.md", sha)
    result = audit_sourced_value(sv, root)
    assert result.passed
    assert result.details["sha256_ok"] and result.details["quote_present"]


def test_audit_fails_on_hash_drift(tmp_path):
    root, sha = _make_library(
        tmp_path, "kuzmin2018", "si/si1.md", f"a {KUZMIN_QUOTE} b"
    )
    sv = _sourced("kuzmin2018", "si/si1.md", sha)
    # Simulate a re-OCR that reflowed the file (bytes change -> hash drift).
    (tmp_path / "kuzmin2018" / "si" / "si1.md").write_text(
        f"a {KUZMIN_QUOTE} b   (re-ocr'd)", encoding="utf-8"
    )
    result = audit_sourced_value(sv, root)
    assert not result.passed
    assert not result.details["sha256_ok"]


def test_audit_fails_when_quote_absent(tmp_path):
    root, sha = _make_library(
        tmp_path, "kuzmin2018", "si/si1.md", "no replicate statement here"
    )
    sv = _sourced(
        "kuzmin2018", "si/si1.md", sha
    )  # sha matches, but quote won't be found
    result = audit_sourced_value(sv, root)
    assert not result.passed
    assert result.details["sha256_ok"] and not result.details["quote_present"]


def test_audit_raises_when_artifact_missing(tmp_path):
    sv = _sourced("kuzmin2018", "si/si1.md", "abc")
    assert not library_available(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        audit_sourced_value(sv, tmp_path)  # citation_key dir does not exist


# --------------------------------------------------------------------------- #
# Picklability across process boundaries (regression)
# --------------------------------------------------------------------------- #
def test_pickles_bare_and_when_embedded_in_a_model_field():
    """A SourcedValue must pickle by qualified name, even after field coercion.

    SourcedValue is deliberately NON-generic: a parametrized pydantic generic
    (``SourcedValue[Any]``) gets a bracketed ``__qualname__`` that ``pickle``
    cannot resolve, so any object embedding one (a Media provenance list inside a
    dataset's reference index) fails to cross a ``multiprocessing.Queue`` /
    ``ProcessPoolExecutor`` -- surfacing as a silent adapter build hang.
    """
    sv = _sourced("kuzmin2018", "si/si1.md", "abc")
    assert pickle.loads(pickle.dumps(sv)).value == sv.value

    holder = _ProvenanceHolder(provenance=[sv])
    # Coercion into the field must keep a module-findable class name.
    assert type(holder.provenance[0]).__qualname__ == "SourcedValue"
    restored = pickle.loads(pickle.dumps(holder))
    assert restored.provenance[0].value == sv.value
