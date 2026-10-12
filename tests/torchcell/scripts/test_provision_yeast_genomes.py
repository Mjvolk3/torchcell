# tests/torchcell/scripts/test_provision_yeast_genomes.py
# [[tests.torchcell.scripts.test_provision_yeast_genomes]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/scripts/test_provision_yeast_genomes.py
"""``scripts/provision_yeast_genomes.py``: the URL it fetches from, and the md5 gate.

Hermetic: the retriever is stubbed, so no byte leaves the machine. What is checked is
what makes a deposit trustworthy -- that the script reads NCBI's own ``md5checksums.txt``
and stops when a member's bytes do not match it, that a member the listing does not name
stops the run, and that the manifest it writes pins the sha256 of the bytes that passed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from scripts import provision_yeast_genomes as p
from torchcell.literature.retrieve import RETRIEVERS
from torchcell.sequence.genome.registry import ROLE_INDEX

MEMBERS = {
    "GCA_002571405.2_ASM257140v2_genomic.gbff.gz": b"flat file",
    "GCA_002571405.2_ASM257140v2_genomic.gff.gz": b"gff",
    "GCA_002571405.2_ASM257140v2_genomic.fna.gz": b"fasta",
    "GCA_002571405.2_ASM257140v2_protein.faa.gz": b"proteins",
    "GCA_002571405.2_ASM257140v2_assembly_report.txt": b"report",
}


def _md5(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def _listing(payloads: dict[str, bytes]) -> bytes:
    return "".join(f"{_md5(v)}  ./{k}\n" for k, v in payloads.items()).encode()


def _serve(
    monkeypatch: pytest.MonkeyPatch, payloads: dict[str, bytes], listing: bytes
) -> list[str]:
    """Stub the direct-URL retriever with the given payloads; record every URL."""
    urls: list[str] = []

    def fetch(url: str, *args: Any, **kwargs: Any) -> bytes:
        urls.append(url)
        name = url.rsplit("/", 1)[-1]
        if name == p.MD5_LISTING:
            return listing
        if name not in payloads:
            raise AssertionError(f"unexpected fetch: {url}")
        return payloads[name]

    monkeypatch.setitem(RETRIEVERS, p.DIRECT_URL, fetch)
    return urls


def test_the_directory_url_is_the_accessions_own_ncbi_path() -> None:
    """GCA_002571405.2 lives under GCA/002/571/405, which is how NCBI shards them."""
    assert p.CENPK113_7D.directory_url == (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/002/571/405/"
        "GCA_002571405.2_ASM257140v2"
    )
    assert set(p.CENPK113_7D.members) == set(MEMBERS)
    assert (
        p.CENPK113_7D.members["GCA_002571405.2_ASM257140v2_assembly_report.txt"]
        == ROLE_INDEX
    )


def test_the_manifest_pins_the_sha256_of_every_member_that_passed_its_md5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every member is fetched, md5-checked against the listing, then sha256-pinned."""
    urls = _serve(monkeypatch, MEMBERS, _listing(MEMBERS))
    manifest = p.build_manifest(p.CENPK113_7D, tmp_path / "staged")
    assert manifest.assembly_set == p.CENPK113_7D.assembly_set
    assert manifest.organism == "Saccharomyces cerevisiae"
    assert manifest.strain_or_population == "CEN.PK113-7D"
    assert [record.path for record in manifest.files] == list(MEMBERS)
    for record in manifest.files:
        assert record.sha256 == hashlib.sha256(MEMBERS[record.path]).hexdigest()
        assert record.retrieval is not None
        assert record.retrieval.method.value == "direct_url"
    # The listing is fetched first, then the members, each from the assembly directory.
    assert urls[0].endswith(p.MD5_LISTING)
    assert len(urls) == len(MEMBERS) + 1
    assert all(url.startswith(p.CENPK113_7D.directory_url) for url in urls)


def test_a_member_whose_bytes_do_not_match_the_listing_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NCBI's own digest is the gate: a drifted member is never deposited."""
    drifted = dict(MEMBERS)
    listing = _listing(MEMBERS)
    drifted["GCA_002571405.2_ASM257140v2_genomic.fna.gz"] = b"other bytes"
    _serve(monkeypatch, drifted, listing)
    with pytest.raises(RuntimeError, match="does not match the published"):
        p.build_manifest(p.CENPK113_7D, tmp_path / "staged")


def test_a_member_the_listing_does_not_name_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unlisted member has no published digest, so there is nothing to check it against."""
    partial = {k: v for k, v in MEMBERS.items() if not k.endswith("protein.faa.gz")}
    _serve(monkeypatch, MEMBERS, _listing(partial))
    with pytest.raises(RuntimeError, match="is not in the directory's"):
        p.build_manifest(p.CENPK113_7D, tmp_path / "staged")


def test_a_staged_file_whose_bytes_changed_since_its_sidecar_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reused fetch is reused only while it still matches what was recorded."""
    _serve(monkeypatch, MEMBERS, _listing(MEMBERS))
    staged = tmp_path / "staged"
    p.build_manifest(p.CENPK113_7D, staged)
    member = next(iter(MEMBERS))
    (staged / member).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="no longer matches its recorded sha256"):
        p.build_manifest(p.CENPK113_7D, staged)


def test_published_md5s_reads_the_listings_own_paths(tmp_path: Path) -> None:
    """The listing's ``./`` prefix is stripped, so its keys are member names."""
    path = tmp_path / p.MD5_LISTING
    path.write_bytes(_listing(MEMBERS))
    assert p.published_md5s(path) == {k: _md5(v) for k, v in MEMBERS.items()}
