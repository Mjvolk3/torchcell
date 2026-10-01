# tests/torchcell/datasets/scerevisiae/test_raw_pins.py
# [[tests.torchcell.datasets.scerevisiae.test_raw_pins]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_raw_pins.py
"""Build-time sha256 pins across every pinned loader (issue #561).

``tests/torchcell/conftest.py`` swaps each loader's ``verify_raw_files`` for a recorder
so the synthetic-raw build tests can run, which means a build test alone cannot tell a
loader that verifies its pins from one that forgot to. This module closes that gap for
every loader module ``PINNED_LOADERS`` finds (one ``pinned_loader`` parameter each):

* each dataset class's ``process()``, run on a ``raw/`` of empty files, reaches
  ``verify_raw_files`` before reading any of them, exactly once, with the full
  ``{file: pin}`` mapping: every file in ``raw_file_names``, each against the pin the
  module declares for it;
* a test marked ``slow`` or ``data`` (a real-data build) gets the real check back in
  every loader, so it meets the real pin, while every other test keeps the recorder;
* the seven loaders whose ``download()`` read a pin from the raw-mirror manifest now
  verify against the module constant and refuse a manifest that records another digest
  with ``ManifestPinMismatchError``.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from torchcell.data import ExperimentDataset, ManifestPinMismatchError

PACKAGE = "torchcell.datasets.scerevisiae"
SHA256_HEX = re.compile(r"[0-9a-f]{64}")

#: Stand-in digests the manifest stubs return, so a manifest-pinned file is traceable
#: to the manifest record (or genomes-tier record) of exactly its own path.
MANIFEST_PIN = "manifest:{}"
GENOME_PIN = "genome:{}"


class _Verified(Exception):
    """Raised by the stand-in check to stop ``process()`` right after the call."""


def _pins_from_constants(module: ModuleType, cls: type) -> dict[str, str]:
    """The ``{file: pin}`` mapping each class declares through its module constants."""
    m = module
    by_class: dict[str, Callable[[], dict[str, str]]] = {
        "EnvChemgenAuesukaree2009Dataset": lambda: {m._PDF_FILENAME: m._PDF_SHA256},
        "SmfBaryshnikova2010Dataset": lambda: {m.XLS_NAME: m.XLS_SHA256},
        "BetaxanthinCachera2023Dataset": lambda: {m.DATA_FILENAME: m.DATA_SHA256},
        "CaudalPanTranscriptome2024Dataset": lambda: {
            m.CAUDAL_ZIP_BASENAME: m.CAUDAL_ZIP_SHA256,
            m.REFGENE_TAR_NAME: m.REFGENE_TAR_SHA256,
            m.PRESENCE_NAME: GENOME_PIN.format(m.PRESENCE_NAME),
            m.COPYNUMBER_NAME: GENOME_PIN.format(m.COPYNUMBER_NAME),
        },
        "AminoAcidCooper2010Dataset": lambda: {m.TABLE4_NAME: m.TABLE4_SHA256},
        "EnvChemgenCostanzo2021Dataset": lambda: {m._S1_FILENAME: m._S1_SHA256},
        "MetaboliteDaSilveira2014Dataset": lambda: {
            m.DATA_FILENAME: m.DATA_SHA256,
            m.CHEBI_FILENAME: m.CHEBI_SHA256,
        },
        "EnvChemgenHoepfner2014Dataset": lambda: {
            name: spec["sha256"] for name, spec in m._DRYAD_FILES.items()
        },
        "CrisprMagicLian2019Dataset": lambda: {
            m.TSV_FILENAME: m.TSV_SHA256,
            m.DESIGN_D_FILENAME: m.DESIGN_D_SHA256,
        },
        "IsobutanolScreenLopez2024Dataset": lambda: {m._XLSX_FILENAME: m._XLSX_SHA256},
        "IsobutanolValidatedLopez2024Dataset": lambda: {
            m._XLSX_FILENAME: m._XLSX_SHA256
        },
        "ProteomeMessner2023Dataset": lambda: {
            m.MATRIX_FILENAME: m.MATRIX_SHA256,
            m.METADATA_FILENAME: m.METADATA_SHA256,
        },
        "CrispriMormino2022Dataset": lambda: {
            m.PDF_FILENAME: m.PDF_SHA256,
            m.PAPER_MD: m.PAPER_MD_SHA256,
        },
        "EnvChemgenMota2024Dataset": lambda: {
            spec["filename"]: spec["sha256"] for spec in m._ACID_SPECS
        },
        "AminoAcidMulleder2016Dataset": lambda: {m.DATA_FILENAME: m.DATA_SHA256},
        "NadalRibellesPerturbSeq2025Dataset": lambda: {
            name: m.SHA256_EXPECTED[name] for name in m.RAW_FILES
        },
        "SmfODuibhir2014Dataset": lambda: {m._RAW_FILENAME: m._DATASET_S2_SHA256},
        "ScmdOhnuki2018Dataset": lambda: {
            name: spec["sha256"] for name, spec in m._RAW_FILES.items()
        },
        "ScmdOhnuki2022Dataset": lambda: {
            m.MUTANT_FILE: m.MUTANT_SHA256,
            m.WT_FILE: m.WT_SHA256,
        },
        "ScmdOhya2005Dataset": lambda: {
            name: spec["sha256"] for name, spec in m._RAW_FILES.items()
        },
        "CarotenoidOzaydin2013Dataset": lambda: {
            m.CarotenoidOzaydin2013Dataset.si_filename: m._SI_SHA256
        },
        "FattyAcidSmith2006Dataset": lambda: {m.XLS_FILENAME: m.XLS_SHA256},
        "CrispriChemgenSmith2016Dataset": lambda: {
            m.EFFECT_FILENAME: m.EFFECT_SHA256,
            m.GUIDE_FILENAME: m.GUIDE_SHA256,
        },
        "EnvChemgenVanacloig2022Dataset": lambda: {m.DATA_FILENAME: m.DATA_SHA256},
        "EnvChemgenWildenhain2015Dataset": lambda: {
            m.DATA_FILENAME: m.DATA_SHA256,
            m.AID_FILENAME: m.AID_SHA256,
        },
        "FattyAcidXue2025Dataset": lambda: {m.DATA_FILENAME: m.DATA_SHA256},
        "YeastPhenomeDataset": lambda: {
            f"{s['pmid']}_{s['stem']}_valuez.txt": s["valuez_sha256"] for s in m.SCREENS
        },
        "OrganicAcidYoshida2012Dataset": lambda: {m.PDF_FILENAME: m.PDF_SHA256},
        "ProteomeZelezniak2018Dataset": lambda: {m.DATA_FILENAME: m.DATA_SHA256},
        "MetaboliteZelezniak2018Dataset": lambda: {
            m.METABOLITE_DATA_FILENAME: m.METABOLITE_DATA_SHA256
        },
    }
    return by_class[cls.__name__]()


#: Classes whose pins are the raw-mirror manifest's records rather than constants.
MANIFEST_PINNED = {
    "Bloom2019Dataset",
    "HetHillenmeyer2008Dataset",
    "HomHillenmeyer2008Dataset",
}


def _dataset_classes(module: ModuleType) -> list[type[ExperimentDataset]]:
    """The concrete dataset classes a loader module defines, by name."""
    return sorted(
        (
            obj
            for obj in vars(module).values()
            if inspect.isclass(obj)
            and obj.__module__ == module.__name__
            and issubclass(obj, ExperimentDataset)
            and not obj.__name__.startswith("_")
        ),
        key=lambda c: c.__name__,
    )


def _stub_pin_sources(
    module: ModuleType,
    cls: type[ExperimentDataset],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Point the manifest and genomes-tier reads at stand-ins naming the path asked."""
    if cls.__name__ in MANIFEST_PINNED:
        monkeypatch.setattr(module, "_data_root", lambda: str(tmp_path))
        monkeypatch.setattr(module, "load_manifest", lambda data_root: "manifest")
        monkeypatch.setattr(
            module, "manifest_sha256", lambda manifest, rel: MANIFEST_PIN.format(rel)
        )
    if cls.__name__.endswith("Hillenmeyer2008Dataset"):
        monkeypatch.setattr(module, "_load_sgd_genes", lambda data_root: set())
    if cls.__name__ == "CaudalPanTranscriptome2024Dataset":
        monkeypatch.setattr(cls, "_data_root", lambda self: str(tmp_path))
        monkeypatch.setattr(
            module,
            "load_genome_manifest",
            lambda *args: SimpleNamespace(
                record=lambda name: SimpleNamespace(sha256=GENOME_PIN.format(name))
            ),
        )


def _expected_pins(
    module: ModuleType, cls: type[ExperimentDataset], raw_files: list[str]
) -> dict[str, str]:
    if cls.__name__ in MANIFEST_PINNED:
        rel = module.raw_relpaths()
        return {name: MANIFEST_PIN.format(rel[name]) for name in raw_files}
    return _pins_from_constants(module, cls)


def test_process_verifies_every_raw_file_against_its_pin_before_reading(
    pinned_loader: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For every dataset class of the loader: ``process()`` on a ``raw/`` of EMPTY files
    calls ``verify_raw_files`` once, before any read (a read of an empty file would
    raise something other than the stand-in's ``_Verified``), with ``self.raw_dir`` and a
    mapping whose files are exactly ``raw_file_names`` and whose pins are the module's
    declared pins (a 64-hex sha256 each, or the manifest record of that very file).
    """
    module = importlib.import_module(f"{PACKAGE}.{pinned_loader}")
    classes = _dataset_classes(module)
    assert classes, f"{pinned_loader} defines no concrete dataset class"
    for cls in classes:
        _stub_pin_sources(module, cls, tmp_path, monkeypatch)
        dataset = cls.__new__(cls)
        dataset.root = str(tmp_path / cls.__name__)
        raw_files = list(dataset.raw_file_names)
        raw = Path(dataset.raw_dir)
        raw.mkdir(parents=True)
        for name in raw_files:
            (raw / name).write_bytes(b"")
        seen: list[tuple[str, dict[str, str]]] = []

        def stop(raw_dir: str, pins: Mapping[str, str]) -> None:
            seen.append((raw_dir, dict(pins)))
            raise _Verified

        monkeypatch.setattr(module, "verify_raw_files", stop)
        with pytest.raises(_Verified):
            dataset.process()
        expected = _expected_pins(module, cls, raw_files)
        assert seen == [(str(raw), expected)], cls.__name__
        assert sorted(expected) == sorted(raw_files), cls.__name__
        if cls.__name__ not in MANIFEST_PINNED | {"CaudalPanTranscriptome2024Dataset"}:
            off_hex = {k: v for k, v in expected.items() if not SHA256_HEX.fullmatch(v)}
            assert off_hex == {}, cls.__name__


class _Node:
    """A test-item stand-in carrying exactly the named markers."""

    def __init__(self, *markers: str) -> None:
        self.markers = set(markers)

    def get_closest_marker(self, name: str) -> Any:
        return pytest.Mark(name, (), {}) if name in self.markers else None


def _loader_modules() -> list[ModuleType]:
    package_dir = Path(importlib.import_module(PACKAGE).__path__[0])
    return [
        importlib.import_module(f"{PACKAGE}.{path.stem}")
        for path in sorted(package_dir.glob("*.py"))
        if "verify_raw_files(" in path.read_text(encoding="utf-8")
    ]


@pytest.mark.parametrize(
    "markers",
    [("slow",), ("data",), ("slow", "data")],
    ids=["slow", "data", "slow+data"],
)
def test_a_real_data_test_meets_the_real_pin(
    markers: tuple[str, ...],
    pin_restorer: Callable[[_Node, pytest.MonkeyPatch], list[str]],
    real_verify_raw_files: Callable[[str, Mapping[str, str]], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A test marked ``slow`` or ``data`` (a real build, run only under ``--slow`` or
    ``--data``) gets the real ``verify_raw_files`` back in all 30 pinned loader modules
    for its duration, and the module recorder returns when its monkeypatch unwinds.
    """
    modules = _loader_modules()
    assert len(modules) == 30
    recorders = {m.__name__: m.verify_raw_files for m in modules}
    assert [n for n, f in recorders.items() if f is real_verify_raw_files] == []
    with monkeypatch.context() as mp:
        restored = pin_restorer(_Node(*markers), mp)
        assert restored == sorted(m.__name__.rsplit(".", 1)[1] for m in modules)
        held = [
            m.__name__
            for m in modules
            if m.verify_raw_files is not real_verify_raw_files
        ]
        assert held == []
    assert {m.__name__: m.verify_raw_files for m in modules} == recorders


@pytest.mark.parametrize(
    "markers", [(), ("gpu",), ("network",)], ids=["unmarked", "gpu", "network"]
)
def test_every_other_test_keeps_the_recorder(
    markers: tuple[str, ...],
    pin_restorer: Callable[[_Node, pytest.MonkeyPatch], list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a ``slow``/``data`` marker nothing is restored: each loader still holds
    the recorder, whatever flags the run carries.
    """
    modules = _loader_modules()
    recorders = {m.__name__: m.verify_raw_files for m in modules}
    assert pin_restorer(_Node(*markers), monkeypatch) == []
    assert {m.__name__: m.verify_raw_files for m in modules} == recorders


def test_the_recorder_checks_presence_and_records_without_hashing(
    raw_pin_calls: list[tuple[str, str, dict[str, str]]], tmp_path: Path
) -> None:
    """The module recorder takes a present file off any pin, recording
    ``(module, raw dir, pins)``, and refuses an absent one by name.
    """
    cachera = importlib.import_module(f"{PACKAGE}.cachera2023")
    (tmp_path / "present.txt").write_bytes(b"not the pinned bytes")
    before = len(raw_pin_calls)
    cachera.verify_raw_files(str(tmp_path), {"present.txt": "0" * 64})
    with pytest.raises(FileNotFoundError) as err:
        cachera.verify_raw_files(str(tmp_path), {"absent.txt": "0" * 64})
    assert str(err.value) == "pinned raw files absent: ['absent.txt']"
    assert raw_pin_calls[before:] == [
        ("cachera2023", str(tmp_path), {"present.txt": "0" * 64})
    ]


#: The seven loaders whose ``download()`` used to take its pin from the manifest:
#: ``(module, class, first mirror relpath checked, its pin constant)``.
MANIFEST_CHECKED_DOWNLOADS = [
    ("baryshnikova2010", "SmfBaryshnikova2010Dataset", "XLS_REL", "XLS_SHA256"),
    ("lian2019", "CrisprMagicLian2019Dataset", "TSV_REL", "TSV_SHA256"),
    ("mormino2022", "CrispriMormino2022Dataset", "PDF_REL", "PDF_SHA256"),
    ("smith2006", "FattyAcidSmith2006Dataset", "XLS_REL", "XLS_SHA256"),
    ("smith2016", "CrispriChemgenSmith2016Dataset", "EFFECT_REL", "EFFECT_SHA256"),
    ("vanacloig2022", "EnvChemgenVanacloig2022Dataset", "DATA_REL", "DATA_SHA256"),
    ("wildenhain2015", "EnvChemgenWildenhain2015Dataset", "DATA_REL", "DATA_SHA256"),
]


@pytest.mark.parametrize(
    ("loader", "class_name", "rel_attr", "pin_attr"),
    MANIFEST_CHECKED_DOWNLOADS,
    ids=[row[0] for row in MANIFEST_CHECKED_DOWNLOADS],
)
def test_download_refuses_a_manifest_digest_off_the_module_pin(
    loader: str,
    class_name: str,
    rel_attr: str,
    pin_attr: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The module constant is the one pin; the manifest is the retrieval record. A
    manifest recording another digest for a mirror file raises
    ``ManifestPinMismatchError`` naming the path, the recorded digest and the pin, and
    nothing is linked into ``raw/``.
    """
    module = importlib.import_module(f"{PACKAGE}.{loader}")
    relpath: str = getattr(module, rel_attr)
    pin: str = getattr(module, pin_attr)
    assert SHA256_HEX.fullmatch(pin)
    data_root = tmp_path / "dr"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    mirror = module.raw_mirror_dir(str(data_root))
    (mirror / relpath).parent.mkdir(parents=True, exist_ok=True)
    (mirror / relpath).write_bytes(b"mirror bytes")
    (mirror / "manifest.json").write_text(
        json.dumps(
            {
                "citation_key": module.CITATION_KEY,
                "files": [
                    {
                        "path": relpath,
                        "role": "raw_data",
                        "bytes": 12,
                        "sha256": "0" * 64,
                    }
                ],
            }
        )
    )
    cls = getattr(module, class_name)
    dataset = cls.__new__(cls)
    dataset.root = str(tmp_path / "build")
    with pytest.raises(ManifestPinMismatchError) as err:
        dataset.download()
    assert str(err.value) == (
        f"raw-mirror manifest records sha256 {'0' * 64} for {relpath}, but the loader "
        f"pins {pin}"
    )
    assert (err.value.relpath, err.value.recorded, err.value.pin) == (
        relpath,
        "0" * 64,
        pin,
    )
    linked = [p for p in (tmp_path / "build").rglob("*") if not p.is_dir()]
    assert linked == []
