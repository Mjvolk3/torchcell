# tests/torchcell/literature/test_manual_deposit.py
# [[tests.torchcell.literature.test_manual_deposit]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_manual_deposit.py
"""The by-hand deposit record: verification, the two record files, the manifest merge."""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from torchcell.literature.manifest import (
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.literature.manual_deposit import (
    CHECKSUMS_NAME,
    DEPOSIT_RECORD_NAME,
    MANUAL_RETRIEVER,
    DepositedFile,
    DepositMismatchError,
    ManualDeposit,
    ScriptedEquivalent,
    artifact_records,
    beside,
    checksums_text,
    deposit_markdown,
    extend_manifest,
    verify_deposit,
    write_deposit_records,
)

KEY = "zzExampleDeposit2026"
PDF = b"%PDF-1.4 example supplement"
MEMBER = b"PK-less member bytes of table s3"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _zip(path: Path) -> bytes:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("table_s3.xlsx", MEMBER)
    return path.read_bytes()


def _file(path: str, data: bytes, **kw: object) -> DepositedFile:
    fields: dict[str, object] = {
        "path": path,
        "role": "si_pdf",
        "sha256": _sha(data),
        "bytes": len(data),
        "original_filename": Path(path).name,
        "arrived_at": "2026-10-12T03:41:18-05:00",
        "description": "an example",
    }
    fields.update(kw)
    return DepositedFile.model_validate(fields)


def _deposit(root: Path) -> ManualDeposit:
    """Write a three-file deposit (a PDF, an archive, one unpacked member) under root."""
    (root / "si").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "si" / "sm.pdf").write_bytes(PDF)
    archive = _zip(root / "data" / "tables.zip")
    (root / "data" / "table_s3.xlsx").write_bytes(MEMBER)
    return ManualDeposit(
        citation_key=KEY,
        doi="10.1000/example",
        title="An example",
        source_url="https://publisher.example/suppl",
        retrieval_command="open the page, download, copy under si/ and data/",
        retrieved_by="the owner",
        deposited_at="2026-10-12",
        issues=(1, 2),
        note="a test deposit",
        files=(
            _file("si/sm.pdf", PDF),
            _file("data/tables.zip", archive, role="raw_data"),
            _file(
                "data/table_s3.xlsx",
                MEMBER,
                role="raw_data",
                container="data/tables.zip",
                original_filename="table_s3.xlsx",
                scripted_equivalent=ScriptedEquivalent(
                    retriever="torchcell.literature.retrieve.pmc_cloud_object",
                    params={"key": "PMC1.1/table_s3.xlsx"},
                    url="https://pmc-oa-opendata.s3.amazonaws.com/PMC1.1/table_s3.xlsx",
                    checked_at="2026-10-12",
                ),
            ),
        ),
    )


def test_a_deposited_file_refuses_a_malformed_record() -> None:
    with pytest.raises(ValidationError, match="64 lowercase hex"):
        _file("si/a.pdf", PDF, sha256="ABC")
    with pytest.raises(ValidationError, match="at least one byte"):
        _file("si/a.pdf", PDF, bytes=0)
    with pytest.raises(ValidationError, match="inside the key directory"):
        _file("../a.pdf", PDF)
    with pytest.raises(ValidationError, match="inside the key directory"):
        _file("/abs/a.pdf", PDF)


def test_a_deposit_refuses_repeats_record_names_and_a_missing_container(
    tmp_path: Path,
) -> None:
    good = _deposit(tmp_path)
    base = good.model_dump()
    with pytest.raises(ValidationError, match="at least one file"):
        ManualDeposit.model_validate({**base, "files": []})
    with pytest.raises(ValidationError, match="repeated"):
        ManualDeposit.model_validate({**base, "files": [base["files"][0]] * 2})
    named = {**base["files"][0], "path": "si/DEPOSIT.md"}
    with pytest.raises(ValidationError, match="record names"):
        ManualDeposit.model_validate({**base, "files": [named]})
    orphan = {**base["files"][2], "container": "data/other.zip"}
    with pytest.raises(ValidationError, match="not a deposited file"):
        ManualDeposit.model_validate({**base, "files": [orphan]})


def test_directories_and_files_are_grouped_and_sorted(tmp_path: Path) -> None:
    deposit = _deposit(tmp_path)
    assert deposit.directories() == ("data", "si")
    assert [f.name for f in deposit.files_in("data")] == ["table_s3.xlsx", "tables.zip"]
    assert deposit.source_of(deposit.files[0]) == "https://publisher.example/suppl"
    other = deposit.files[0].model_copy(update={"source_url": "https://doi.org/x"})
    assert deposit.source_of(other) == "https://doi.org/x"
    assert beside(deposit.files[0], CHECKSUMS_NAME) == "si/SHA256SUMS.txt"


def test_verify_accepts_the_deposit_and_names_each_kind_of_mismatch(
    tmp_path: Path,
) -> None:
    deposit = _deposit(tmp_path)
    verify_deposit(deposit, tmp_path)
    (tmp_path / "si" / "sm.pdf").write_bytes(PDF + b"!")
    with pytest.raises(DepositMismatchError, match="recorded"):
        verify_deposit(deposit, tmp_path)
    (tmp_path / "si" / "sm.pdf").unlink()
    with pytest.raises(DepositMismatchError, match="MANUAL RECIPE: open the page"):
        verify_deposit(deposit, tmp_path)


def test_verify_requires_an_unpacked_member_to_equal_its_archive_member(
    tmp_path: Path,
) -> None:
    deposit = _deposit(tmp_path)
    renamed = deposit.files[2].model_copy(update={"original_filename": "absent.xlsx"})
    lying = deposit.model_copy(
        update={"files": (deposit.files[0], deposit.files[1], renamed)}
    )
    with pytest.raises(KeyError, match="absent.xlsx"):
        verify_deposit(lying, tmp_path)
    with zipfile.ZipFile(tmp_path / "data" / "tables.zip", "w") as archive:
        archive.writestr("table_s3.xlsx", b"different")
    archive_bytes = (tmp_path / "data" / "tables.zip").read_bytes()
    rezipped = deposit.files[1].model_copy(
        update={"sha256": _sha(archive_bytes), "bytes": len(archive_bytes)}
    )
    swapped = deposit.model_copy(
        update={"files": (deposit.files[0], rezipped, deposit.files[2])}
    )
    with pytest.raises(DepositMismatchError, match="is not the member table_s3.xlsx"):
        verify_deposit(swapped, tmp_path)


def test_the_record_files_name_every_file_of_their_directory_only(
    tmp_path: Path,
) -> None:
    deposit = _deposit(tmp_path)
    assert checksums_text(deposit, "data") == (
        f"{_sha(MEMBER)}  table_s3.xlsx\n{deposit.files[1].sha256}  tables.zip\n"
    )
    text = deposit_markdown(deposit, "data")
    assert "- retrieval_method: manual_browser\n" in text
    assert "- retrieval_command (manual recipe): open the page" in text
    assert "- issues: #1, #2\n" in text
    assert "| table_s3.xlsx | table_s3.xlsx (member of data/tables.zip) |" in text
    assert "sm.pdf" not in text
    assert "## Scripted equivalents" in text
    assert "returned these exact bytes on 2026-10-12" in text
    assert "## Scripted equivalents" not in deposit_markdown(deposit, "si")


def test_write_records_is_idempotent_and_refuses_to_overwrite(tmp_path: Path) -> None:
    deposit = _deposit(tmp_path)
    written = write_deposit_records(deposit, tmp_path)
    assert sorted(str(p.relative_to(tmp_path)) for p in written) == [
        "data/DEPOSIT.md",
        "data/SHA256SUMS.txt",
        "si/DEPOSIT.md",
        "si/SHA256SUMS.txt",
    ]
    assert write_deposit_records(deposit, tmp_path) == written
    (tmp_path / "si" / DEPOSIT_RECORD_NAME).write_text("edited by hand\n")
    with pytest.raises(DepositMismatchError, match="other content"):
        write_deposit_records(deposit, tmp_path)


def test_artifact_records_carry_the_recipe_the_container_and_the_equivalent(
    tmp_path: Path,
) -> None:
    deposit = _deposit(tmp_path)
    records = {r.path: r for r in artifact_records(deposit)}
    pdf = records["si/sm.pdf"]
    assert pdf.retrieval is not None
    assert pdf.retrieval.method is RetrievalMethod.manual_browser
    assert pdf.retrieval.retriever == MANUAL_RETRIEVER
    assert pdf.retrieval.retrieved_at == "2026-10-12T03:41:18-05:00"
    assert pdf.retrieval.params["deposit_record"] == "si/DEPOSIT.md"
    assert pdf.retrieval.params["issues"] == [1, 2]
    assert "container" not in pdf.retrieval.params
    member = records["data/table_s3.xlsx"]
    assert member.retrieval is not None
    assert member.retrieval.params["container"] == "data/tables.zip"
    assert member.retrieval.params["container_sha256"] == deposit.files[1].sha256
    assert member.retrieval.params["member"] == "table_s3.xlsx"
    assert member.retrieval.params["scripted_equivalent"]["checked_at"] == "2026-10-12"
    assert member.original_filename == "table_s3.xlsx"


def test_extend_manifest_creates_merges_and_refuses_a_disagreeing_sha(
    tmp_path: Path,
) -> None:
    deposit = _deposit(tmp_path)
    records = artifact_records(deposit)
    path = extend_manifest(
        tmp_path,
        records[:1],
        citation_key=KEY,
        doi=deposit.doi,
        title=deposit.title,
        si_expected=["Table S9: not released"],
        si_data_sources=["https://elsewhere.example"],
    )
    first = Manifest.model_validate_json(path.read_text())
    assert [r.path for r in first.files] == ["si/sm.pdf"]
    assert first.si_data_sources == [
        "https://publisher.example/suppl",
        "https://elsewhere.example",
    ]
    held = ArtifactRecord(
        path="data/other.csv",
        role="raw_data",
        bytes=1,
        sha256="0" * 64,
        retrieval=RetrievalRecord(
            method=RetrievalMethod.direct_url,
            retriever="torchcell.literature.retrieve.direct_url",
            sha256="0" * 64,
            retrieved_at="2026-10-01",
        ),
    )
    data = json.loads(path.read_text())
    data["files"].append(json.loads(held.model_dump_json()))
    path.write_text(json.dumps(data))
    extend_manifest(
        tmp_path, records, citation_key=KEY, doi=deposit.doi, title=deposit.title
    )
    merged = Manifest.model_validate_json(path.read_text())
    assert [r.path for r in merged.files] == [
        "data/other.csv",
        "data/table_s3.xlsx",
        "data/tables.zip",
        "si/sm.pdf",
    ]
    assert merged.si_expected == ["Table S9: not released"]
    assert merged.created_at == first.created_at
    derived = ArtifactRecord(
        path="si/sm.layout.txt",
        role="si_text",
        bytes=3,
        sha256="1" * 64,
        source="derived from si/sm.pdf",
    )
    extend_manifest(
        tmp_path, [derived], citation_key=KEY, doi=deposit.doi, title=deposit.title
    )
    after = Manifest.model_validate_json(path.read_text())
    assert "derived from si/sm.pdf" not in after.si_data_sources
    assert after.si_data_sources == merged.si_data_sources
    bad = records[0].model_copy(update={"sha256": "f" * 64})
    with pytest.raises(DepositMismatchError, match="refusing"):
        extend_manifest(
            tmp_path, [bad], citation_key=KEY, doi=deposit.doi, title=deposit.title
        )
    with pytest.raises(DepositMismatchError, match="not zzOther"):
        extend_manifest(tmp_path, records, citation_key="zzOther", doi="d", title="t")
