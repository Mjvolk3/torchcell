# tests/torchcell/datasets/scerevisiae/test_raw_pins.py
# [[tests.torchcell.datasets.scerevisiae.test_raw_pins]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_raw_pins.py
"""Build-time sha256 pins across every pinned loader (issue #561).

``tests/torchcell/conftest.py`` swaps each loader's ``verify_raw_files`` for a recorder
so the synthetic-raw build tests can run, which means a build test alone cannot tell a
loader that verifies its pins from one that forgot to. This module closes that gap for
every loader module ``PINNED_LOADERS`` finds (one ``pinned_loader`` parameter each):

* each dataset class's ``process()``, run on an empty ``raw/``, calls
  ``verify_raw_files`` before opening any raw file, and that first call carries the
  full ``{file: pin}`` mapping: every file in ``raw_file_names``, each against the pin
  the module declares for it (a new pinned class needs a row in
  ``_pins_from_constants`` and fails with ``KeyError`` until it has one);
* every loader module with a ``process()`` is pinned or on the ``UNPINNED_LOADERS``
  debt list, and no debt-list module carries a pin;
* a test marked ``slow`` or ``data`` (a real-data build) gets the real check back in
  every loader, so it meets the real pin, while every other test keeps the recorder;
* the seven loaders whose ``download()`` read a pin from the raw-mirror manifest now
  verify against the module constant and refuse a manifest that records another digest
  with ``ManifestPinMismatchError``.
"""

from __future__ import annotations

import ast
import hashlib
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

#: Stand-in digests the manifest stubs return. Each encodes every argument of the read
#: (data root, genomes-tier key, path), so a loader asking with the wrong root, key or
#: path gets a value the expected mapping does not carry.
MANIFEST_PIN = "manifest:{root}:{rel}"
GENOME_PIN = "genome:{key}:{root}:{name}"


class _Verified(Exception):
    """Raised by the stand-in check to stop ``process()`` right after the call."""


def _pins_from_constants(
    module: ModuleType, cls: type, data_root: str
) -> dict[str, str]:
    """The ``{file: pin}`` mapping each class declares through its module constants.

    A new pinned class has no row here and raises ``KeyError`` until one is added: that
    is the tripwire, so a new loader is parametrized automatically but asserted only once
    its expected mapping is written down.
    """
    m = module

    def genome(name: str) -> str:
        return GENOME_PIN.format(key=m.PETER2018_1011, root=data_root, name=name)

    by_class: dict[str, Callable[[], dict[str, str]]] = {
        "EnvChemgenAuesukaree2009Dataset": lambda: {m._PDF_FILENAME: m._PDF_SHA256},
        "SmfBaryshnikova2010Dataset": lambda: {m.XLS_NAME: m.XLS_SHA256},
        "BetaxanthinCachera2023Dataset": lambda: {m.DATA_FILENAME: m.DATA_SHA256},
        "CaudalPanTranscriptome2024Dataset": lambda: {
            m.CAUDAL_ZIP_BASENAME: m.CAUDAL_ZIP_SHA256,
            m.REFGENE_TAR_NAME: m.REFGENE_TAR_SHA256,
            m.PRESENCE_NAME: genome(m.PRESENCE_NAME),
            m.COPYNUMBER_NAME: genome(m.COPYNUMBER_NAME),
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
        "SynthLethalityYeastSynthLethDbDataset": lambda: {
            m.SL_CSV_NAME: m.SL_CSV_SHA256
        },
        "SynthRescueYeastSynthLethDbDataset": lambda: {m.SR_CSV_NAME: m.SR_CSV_SHA256},
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
    data_root: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Point the manifest and genomes-tier reads at stand-ins naming what was asked."""
    if cls.__name__ in MANIFEST_PINNED:
        monkeypatch.setattr(module, "_data_root", lambda: data_root)
        monkeypatch.setattr(
            module, "load_manifest", lambda root: SimpleNamespace(root=root)
        )
        monkeypatch.setattr(
            module,
            "manifest_sha256",
            lambda manifest, rel: MANIFEST_PIN.format(root=manifest.root, rel=rel),
        )
    if cls.__name__.endswith("Hillenmeyer2008Dataset"):
        monkeypatch.setattr(module, "_load_sgd_genes", lambda root: set())
    if cls.__name__ == "CaudalPanTranscriptome2024Dataset":
        monkeypatch.setattr(cls, "_data_root", lambda self: data_root)

        def genome_manifest(key: str, root: str) -> SimpleNamespace:
            return SimpleNamespace(
                record=lambda name: SimpleNamespace(
                    sha256=GENOME_PIN.format(key=key, root=root, name=name)
                )
            )

        monkeypatch.setattr(module, "load_genome_manifest", genome_manifest)


def _expected_pins(
    module: ModuleType,
    cls: type[ExperimentDataset],
    raw_files: list[str],
    data_root: str,
) -> dict[str, str]:
    if cls.__name__ in MANIFEST_PINNED:
        rel = module.raw_relpaths()
        return {
            name: MANIFEST_PIN.format(root=data_root, rel=rel[name])
            for name in raw_files
        }
    return _pins_from_constants(module, cls, data_root)


def test_process_verifies_every_raw_file_against_its_pin_before_reading(
    pinned_loader: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For every dataset class of the loader, ``process()`` on an EMPTY ``raw/`` (no raw
    file exists, so any open before the check raises something other than the
    stand-in's ``_Verified``) calls ``verify_raw_files`` before opening any raw file,
    and that first call carries ``self.raw_dir`` and the complete mapping: its files are
    exactly ``raw_file_names`` and its pins are the module's declared pins (a 64-hex
    sha256 each, or the manifest or genomes-tier record of that very file, asked of the
    right data root and key). The stand-in raises, so later calls are not observed.
    """
    module = importlib.import_module(f"{PACKAGE}.{pinned_loader}")
    classes = _dataset_classes(module)
    assert classes, f"{pinned_loader} defines no concrete dataset class"
    data_root = str(tmp_path / "data_root")
    for cls in classes:
        _stub_pin_sources(module, cls, data_root, monkeypatch)
        dataset = cls.__new__(cls)
        dataset.root = str(tmp_path / cls.__name__)
        raw_files = list(dataset.raw_file_names)
        raw = Path(dataset.raw_dir)
        raw.mkdir(parents=True)
        seen: list[tuple[str, dict[str, str]]] = []

        def stop(raw_dir: str, pins: Mapping[str, str]) -> None:
            seen.append((raw_dir, dict(pins)))
            raise _Verified

        monkeypatch.setattr(module, "verify_raw_files", stop)
        with pytest.raises(_Verified):
            dataset.process()
        expected = _expected_pins(module, cls, raw_files, data_root)
        assert seen == [(str(raw), expected)], cls.__name__
        assert sorted(expected) == sorted(raw_files), cls.__name__
        assert list(raw.iterdir()) == [], cls.__name__
        if cls.__name__ not in MANIFEST_PINNED | {"CaudalPanTranscriptome2024Dataset"}:
            off_hex = {k: v for k, v in expected.items() if not SHA256_HEX.fullmatch(v)}
            assert off_hex == {}, cls.__name__


#: Loader modules with a ``process()`` and no build-time pin yet: the pin debt. A module
#: leaves this list only by gaining a ``verify_raw_files`` call.
UNPINNED_LOADERS = frozenset(
    {"costanzo2016", "kemmeren2014", "kuzmin2018", "kuzmin2020", "sameith2015", "sgd"}
)


def _module_trees() -> dict[str, ast.Module]:
    package_dir = Path(importlib.import_module(PACKAGE).__path__[0])
    return {
        path.stem: ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(package_dir.glob("*.py"))
    }


def _defines_process(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ClassDef)
        and any(
            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == "process"
            for item in node.body
        )
        for node in ast.walk(tree)
    )


def _pin_evidence(tree: ast.Module) -> list[str]:
    """``sha256``-named names, attributes, args and defs, and 64-hex string constants."""
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and "sha256" in node.id.lower():
            found.append(node.id)
        elif isinstance(node, ast.Attribute) and "sha256" in node.attr.lower():
            found.append(node.attr)
        elif isinstance(node, ast.arg) and "sha256" in node.arg.lower():
            found.append(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and (
            "sha256" in node.name.lower()
        ):
            found.append(node.name)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and SHA256_HEX.fullmatch(node.value)
        ):
            found.append(node.value)
    return found


def test_every_loader_is_pinned_or_on_the_debt_list(
    pinned_loaders: tuple[str, ...],
) -> None:
    """(A) Every module defining a ``process()`` either calls ``verify_raw_files`` (and so
    is in ``PINNED_LOADERS``) or is on ``UNPINNED_LOADERS``: a new loader with no call,
    or a pinned loader losing its call, changes the difference. (B) No module outside
    ``PINNED_LOADERS`` carries a pin (a ``sha256``-named identifier or attribute, or a
    64-hex string constant, by AST, so comments do not count): a debt-list module that
    gains a pin without the build-time call fails here.
    """
    trees = _module_trees()
    with_process = {name for name, tree in trees.items() if _defines_process(tree)}
    assert with_process - set(pinned_loaders) == UNPINNED_LOADERS
    carrying = {
        name: _pin_evidence(tree)
        for name, tree in trees.items()
        if name not in pinned_loaders and _pin_evidence(tree)
    }
    assert carrying == {}


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
    ``--data``) gets the real ``verify_raw_files`` back in every pinned loader module
    for its duration, and the module recorder returns when its monkeypatch unwinds.
    """
    modules = _loader_modules()
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


#: The seven loaders whose ``download()`` used to take its pin from the manifest, one
#: row per mirror file (11): ``(module, class, relpath constant, pin constant, files
#: checked before it)``, each earlier file as ``(relpath, pin, raw file name)`` constants.
MANIFEST_CHECKED_DOWNLOADS: list[
    tuple[str, str, str, str, tuple[tuple[str, str, str], ...]]
] = [
    ("baryshnikova2010", "SmfBaryshnikova2010Dataset", "XLS_REL", "XLS_SHA256", ()),
    ("lian2019", "CrisprMagicLian2019Dataset", "TSV_REL", "TSV_SHA256", ()),
    (
        "lian2019",
        "CrisprMagicLian2019Dataset",
        "DESIGN_D_REL",
        "DESIGN_D_SHA256",
        (("TSV_REL", "TSV_SHA256", "TSV_FILENAME"),),
    ),
    ("mormino2022", "CrispriMormino2022Dataset", "PDF_REL", "PDF_SHA256", ()),
    (
        "mormino2022",
        "CrispriMormino2022Dataset",
        "PAPER_MD_REL",
        "PAPER_MD_SHA256",
        (("PDF_REL", "PDF_SHA256", "PDF_FILENAME"),),
    ),
    ("smith2006", "FattyAcidSmith2006Dataset", "XLS_REL", "XLS_SHA256", ()),
    ("smith2016", "CrispriChemgenSmith2016Dataset", "EFFECT_REL", "EFFECT_SHA256", ()),
    (
        "smith2016",
        "CrispriChemgenSmith2016Dataset",
        "GUIDE_REL",
        "GUIDE_SHA256",
        (("EFFECT_REL", "EFFECT_SHA256", "EFFECT_FILENAME"),),
    ),
    ("vanacloig2022", "EnvChemgenVanacloig2022Dataset", "DATA_REL", "DATA_SHA256", ()),
    (
        "wildenhain2015",
        "EnvChemgenWildenhain2015Dataset",
        "DATA_REL",
        "DATA_SHA256",
        (),
    ),
    (
        "wildenhain2015",
        "EnvChemgenWildenhain2015Dataset",
        "AID_REL",
        "AID_SHA256",
        (("DATA_REL", "DATA_SHA256", "DATA_FILENAME"),),
    ),
]

MIRROR_BYTES = b"mirror bytes"


@pytest.mark.parametrize(
    ("loader", "class_name", "rel_attr", "pin_attr", "before"),
    MANIFEST_CHECKED_DOWNLOADS,
    ids=[f"{row[0]}-{row[2]}" for row in MANIFEST_CHECKED_DOWNLOADS],
)
def test_download_refuses_a_manifest_digest_off_the_module_pin(
    loader: str,
    class_name: str,
    rel_attr: str,
    pin_attr: str,
    before: tuple[tuple[str, str, str], ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The module constant is the one pin; the manifest is the retrieval record. A
    manifest recording another digest for a mirror file raises
    ``ManifestPinMismatchError`` naming the path, the recorded digest and the pin, and
    that file is not linked into ``raw/`` (only the files checked before it, whose pin
    is repointed at the staged bytes and whose manifest record agrees, are linked).
    """
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    module = importlib.import_module(f"{PACKAGE}.{loader}")
    relpath: str = getattr(module, rel_attr)
    pin: str = getattr(module, pin_attr)
    assert SHA256_HEX.fullmatch(pin)
    staged_digest = hashlib.sha256(MIRROR_BYTES).hexdigest()
    data_root = tmp_path / "dr"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    mirror = module.raw_mirror_dir(str(data_root))
    records = []
    for earlier_rel_attr, earlier_pin_attr, _ in before:
        monkeypatch.setattr(module, earlier_pin_attr, staged_digest)
        records.append((getattr(module, earlier_rel_attr), staged_digest))
    records.append((relpath, "0" * 64))
    for rel, _ in records:
        (mirror / rel).parent.mkdir(parents=True, exist_ok=True)
        (mirror / rel).write_bytes(MIRROR_BYTES)
    (mirror / "manifest.json").write_text(
        json.dumps(
            {
                "citation_key": module.CITATION_KEY,
                "files": [
                    {
                        "path": rel,
                        "role": "raw_data",
                        "bytes": len(MIRROR_BYTES),
                        "sha256": digest,
                    }
                    for rel, digest in records
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
    linked = sorted(p.name for p in (tmp_path / "build").rglob("*") if not p.is_dir())
    assert linked == sorted(getattr(module, name_attr) for _, _, name_attr in before)
