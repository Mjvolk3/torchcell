# tests/torchcell/datasets/test_loader_mains.py
# [[tests.torchcell.datasets.test_loader_mains]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_loader_mains.py
"""The loader ``main()`` build entries: which tree each builds into, and with what.

``experiments/database/datasets.sh`` calls each loader's ``main()``. The contract is
the root directory every dataset class is built under (``$DATA_ROOT/data/torchcell/
<name>``), the genome kwargs (``genome_root=$DATA_ROOT/data/sgd/genome``,
``go_root=$DATA_ROOT/data/go``, ``overwrite=False``), whether the dataset receives
``genome=``, and the lines it prints.

Fixture: ``DATA_ROOT`` is a ``tmp_path`` directory (some mains read the drop log
under the root, so a real readable directory is needed); ``dotenv.load_dotenv`` is a
no-op so the repo ``.env`` cannot override it; ``SCerevisiaeGenome`` and each dataset
class are replaced ON THE LOADER MODULE by recording fakes. Every fake dataset has
two hand-built records (``RECORDS``), so ``len = 2`` is printed and ``dataset[0]`` is
``RECORDS[0]``. Nothing touches the network or the sentinel ``DATA_ROOT``.
"""

import importlib
import json
import os.path as osp
from pathlib import Path
from typing import Any

import pytest

RECORDS: list[dict[str, Any]] = [
    {
        "experiment": {
            "genotype": {"cross": "BYxRM", "segregant_id": "A01", "blocks": [1, 2, 3]},
            "phenotype": {"environment_response": 0.25},
        }
    },
    {
        "experiment": {
            "genotype": {"cross": "BYx3S", "segregant_id": "B02", "blocks": [4]},
            "phenotype": {"environment_response": -1.5},
        }
    },
]

DROP_RULES = [{"rule": "missing_value", "n_records": 3}]


class FakeGenome:
    """Records the constructor kwargs and the drop calls ``main`` makes."""

    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        """Record the kwargs on the class."""
        FakeGenome.calls.append(kwargs)
        self.dropped: list[str] = []

    def drop_chrmt(self) -> None:
        """Record the mitochondrial-chromosome drop."""
        self.dropped.append("chrmt")

    def drop_empty_go(self) -> None:
        """Record the empty-GO drop."""
        self.dropped.append("empty_go")


def make_fake_dataset(log: list[tuple[str, dict[str, Any]]], name: str) -> type:
    """A dataset class that logs ``(name, kwargs)`` and serves ``RECORDS``."""

    class FakeDataset:
        def __init__(self, **kwargs: Any) -> None:
            log.append((name, kwargs))

        def __len__(self) -> int:
            return len(RECORDS)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return RECORDS[idx]

    return FakeDataset


# (module, [(dataset class attr, root subdir under $DATA_ROOT/data/torchcell)],
#  builds a genome and passes it, drop log the main reads under the root)
MAINS: list[tuple[str, list[tuple[str, str]], bool, str | None]] = [
    (
        "lopez2024",
        [
            ("IsobutanolScreenLopez2024Dataset", "isobutanol_screen_lopez2024"),
            ("IsobutanolValidatedLopez2024Dataset", "isobutanol_validated_lopez2024"),
        ],
        True,
        None,
    ),
    ("bloom2019", [("Bloom2019Dataset", "bloom2019")], True, None),
    (
        "hillenmeyer2008",
        [
            ("HetHillenmeyer2008Dataset", "env_chemgen_hillenmeyer2008_het"),
            ("HomHillenmeyer2008Dataset", "env_chemgen_hillenmeyer2008_hom"),
        ],
        False,
        None,
    ),
    (
        "smith2006",
        [("FattyAcidSmith2006Dataset", "env_chemgen_smith2006")],
        True,
        "preprocess/dropped_records.json",
    ),
    ("xue2025", [("FattyAcidXue2025Dataset", "ffa_xue2025")], True, None),
    (
        "costanzo2021",
        [("EnvChemgenCostanzo2021Dataset", "env_chemgen_costanzo2021")],
        True,
        "dropped_records.json",
    ),
    (
        "smith2016",
        [("CrispriChemgenSmith2016Dataset", "crispri_chemgen_smith2016")],
        True,
        "preprocess/dropped_records.json",
    ),
    (
        "vanacloig2022",
        [("EnvChemgenVanacloig2022Dataset", "env_chemgen_vanacloig2022")],
        False,
        "preprocess/dropped_records.json",
    ),
    (
        "zelezniak2018",
        [
            ("ProteomeZelezniak2018Dataset", "proteome_zelezniak2018"),
            ("MetaboliteZelezniak2018Dataset", "metabolite_zelezniak2018"),
        ],
        False,
        None,
    ),
    (
        "auesukaree2009",
        [("EnvChemgenAuesukaree2009Dataset", "env_chemgen_auesukaree2009")],
        True,
        None,
    ),
    (
        "lian2019",
        [("CrisprMagicLian2019Dataset", "crispr_magic_lian2019")],
        True,
        "preprocess/dropped_records.json",
    ),
    ("mormino2022", [("CrispriMormino2022Dataset", "crispri_mormino2022")], True, None),
    ("mota2024", [("EnvChemgenMota2024Dataset", "env_chemgen_mota2024")], True, None),
    (
        "wildenhain2015",
        [("EnvChemgenWildenhain2015Dataset", "env_chemgen_wildenhain2015")],
        False,
        None,
    ),
    (
        "baryshnikova2010",
        [("SmfBaryshnikova2010Dataset", "smf_baryshnikova2010")],
        True,
        None,
    ),
    (
        "cooper2010",
        [("AminoAcidCooper2010Dataset", "amino_acid_cooper2010")],
        False,
        None,
    ),
]

REC0 = repr(RECORDS[0])
RULES_JSON = json.dumps(DROP_RULES, indent=2)

# The exact stdout of each main on the fixture.
EXPECTED_STDOUT: dict[str, str] = {
    "lopez2024": f"screen len = 2\n{REC0}\nvalidated len = 2\n{REC0}\n",
    "bloom2019": "len = 2\ncross BYxRM segregant A01 blocks 3\nphenotype 0.25\n",
    "hillenmeyer2008": f"het: len = 2\n{REC0}\nhom: len = 2\n{REC0}\n",
    "smith2006": f"len = 2\n{REC0}\n{RULES_JSON}\n",
    "xue2025": f"len = 2\n{REC0}\n",
    "costanzo2021": f"len = 2\n{REC0}\n7\n",
    "smith2016": f"len = 2\n{REC0}\n{RULES_JSON}\n",
    "vanacloig2022": f"len = 2\n{REC0}\n{RULES_JSON}\n",
    "zelezniak2018": f"proteome len = 2\n{REC0}\nmetabolite len = 2\n{REC0}\n",
    "auesukaree2009": f"len = 2\n{REC0}\n",
    "lian2019": f"len = 2\n{RULES_JSON}\n",
    "mormino2022": "len = 2\n"
    + json.dumps(RECORDS[0]["experiment"], indent=2, default=str)
    + "\n",
    "mota2024": f"len = 2\n{REC0}\n",
    "wildenhain2015": f"len = 2\n{REC0}\n",
    "baryshnikova2010": f"len = 2\n{REC0}\n",
    "cooper2010": f"len = 2\n{REC0}\n",
}


_DOTENV_CALLS: list[tuple[Any, ...]] = []


def _run_main(
    module_name: str,
    datasets: list[tuple[str, str]],
    drop_log: str | None,
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[tuple[str, dict[str, Any]]], Any]:
    """Patch the module, write the drop log the main reads, run ``main()``."""
    import dotenv

    mod = importlib.import_module(f"torchcell.datasets.scerevisiae.{module_name}")
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _DOTENV_CALLS.clear()

    def record_load_dotenv(*args: Any, **kwargs: Any) -> bool:
        _DOTENV_CALLS.append(args)
        return True

    monkeypatch.setattr(dotenv, "load_dotenv", record_load_dotenv)
    FakeGenome.calls = []
    monkeypatch.setattr(mod, "SCerevisiaeGenome", FakeGenome, raising=False)
    log: list[tuple[str, dict[str, Any]]] = []
    for attr, _ in datasets:
        monkeypatch.setattr(mod, attr, make_fake_dataset(log, attr))
    if drop_log is not None:
        root = data_root / "data/torchcell" / datasets[0][1]
        path = root / drop_log
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"rules": DROP_RULES, "n_dropped_records": 7}))
    mod.main()
    return log, mod


@pytest.mark.parametrize(
    ("module_name", "datasets", "passes_genome", "drop_log"),
    MAINS,
    ids=[m[0] for m in MAINS],
)
def test_main_builds_each_dataset_under_data_torchcell_root(
    module_name: str,
    datasets: list[tuple[str, str]],
    passes_genome: bool,
    drop_log: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pins each dataset class's exact kwargs: ``root`` and ``genome`` only if passed.

    A genome-building main constructs exactly one ``SCerevisiaeGenome`` with the
    three kwargs below and hands the SAME object to every dataset; the others never
    construct one and pass ``root`` alone.
    """
    data_root = tmp_path / "dr"
    log, _ = _run_main(module_name, datasets, drop_log, data_root, monkeypatch)
    capsys.readouterr()
    # every main loads the repo .env exactly once (three of them through _data_root)
    assert _DOTENV_CALLS == [()]
    assert [name for name, _ in log] == [attr for attr, _ in datasets]
    for (_, kwargs), (_, sub) in zip(log, datasets, strict=True):
        expected_root = f"{data_root}/data/torchcell/{sub}"
        if passes_genome:
            assert set(kwargs) == {"root", "genome"}
            assert type(kwargs["genome"]) is FakeGenome
        else:
            assert set(kwargs) == {"root"}
        assert kwargs["root"] == expected_root
    if passes_genome:
        assert FakeGenome.calls == [
            {
                "genome_root": f"{data_root}/data/sgd/genome",
                "go_root": f"{data_root}/data/go",
                "overwrite": False,
            }
        ]
        genomes = {id(kwargs["genome"]) for _, kwargs in log}
        assert len(genomes) == 1
    else:
        assert FakeGenome.calls == []


@pytest.mark.parametrize(
    ("module_name", "datasets", "passes_genome", "drop_log"),
    MAINS,
    ids=[m[0] for m in MAINS],
)
def test_main_prints_length_record_and_drop_log(
    module_name: str,
    datasets: list[tuple[str, str]],
    passes_genome: bool,
    drop_log: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pins the exact stdout: ``len = 2``, ``dataset[0]`` and the drop-log field read.

    ``bloom2019`` prints the genotype fields (``cross``, ``segregant_id``, the block
    count ``len([1, 2, 3]) == 3``) and the phenotype value; ``mormino2022`` prints
    ``json.dumps`` of the experiment; the drop-log mains print ``json.dumps`` of
    ``rules`` (``costanzo2021`` prints ``n_dropped_records`` == 7 instead).
    """
    _run_main(module_name, datasets, drop_log, tmp_path / "dr", monkeypatch)
    assert capsys.readouterr().out == EXPECTED_STDOUT[module_name]


class FakeGraph:
    """Records the ``SCerevisiaeGraph`` kwargs; ``G_gene`` is a sentinel string."""

    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        """Record the kwargs on the class."""
        FakeGraph.calls.append(kwargs)
        self.G_gene = "G_gene-sentinel"


def test_sgd_gene_graph_main_builds_one_dataset_per_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pins the genome kwargs + drops, the graph kwargs, one dataset per model, stdout.

    ``main`` imports ``SCerevisiaeGraph`` from ``torchcell.graph`` and
    ``SCerevisiaeGenome`` from the s288c module inside the function, so both are
    patched there. ``MODEL_TO_WINDOW`` keys are ``normalized_chrom_pathways`` then
    ``chrom_pathways``; the fake reports 17 chromosomes and 5 pathways.
    """
    import torchcell.graph
    import torchcell.sequence.genome.scerevisiae.s288c as s288c
    from torchcell.datasets import sgd_gene_graph

    data_root = str(tmp_path / "dr")
    monkeypatch.setenv("DATA_ROOT", data_root)
    FakeGenome.calls = []
    FakeGraph.calls = []
    genomes: list[FakeGenome] = []

    class RecordingGenome(FakeGenome):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            genomes.append(self)

    monkeypatch.setattr(s288c, "SCerevisiaeGenome", RecordingGenome)
    monkeypatch.setattr(torchcell.graph, "SCerevisiaeGraph", FakeGraph)
    built: list[dict[str, Any]] = []

    class FakeEmbedding:
        MODEL_TO_WINDOW = sgd_gene_graph.GraphEmbeddingDataset.MODEL_TO_WINDOW

        def __init__(self, **kwargs: Any) -> None:
            built.append(kwargs)
            self.categorical_features = {
                "chromosome": {"num_values": 17},
                "pathways": {"num_values": 5},
            }

        def __getitem__(self, idx: int) -> str:
            return f"datum-{idx}"

    monkeypatch.setattr(sgd_gene_graph, "GraphEmbeddingDataset", FakeEmbedding)
    sgd_gene_graph.main()

    assert FakeGenome.calls == [
        {
            "genome_root": osp.join(data_root, "data/sgd/genome"),
            "go_root": osp.join(data_root, "data/go"),
            "overwrite": False,
        }
    ]
    assert genomes[0].dropped == ["chrmt", "empty_go"]
    assert FakeGraph.calls == [
        {
            "sgd_root": f"{data_root}/data/sgd/genome",
            "string_root": f"{data_root}/data/string",
            "tflink_root": f"{data_root}/data/tflink",
            "genome": genomes[0],
        }
    ]
    root = f"{data_root}/data/scerevisiae/sgd_gene_graph"
    assert built == [
        {
            "root": root,
            "graph": "G_gene-sentinel",
            "model_name": name,
            "categorical_features": {"chromosome": {}, "pathways": {}},
        }
        for name in ("normalized_chrom_pathways", "chrom_pathways")
    ]
    block = (
        "Processing model: {m}\nCompleted processing for model: {m}\n"
        "Number of unique chromosomes: 17\nNumber of unique pathways: 5\n"
        "Example data point:\ndatum-0\n\n"
    )
    assert capsys.readouterr().out == block.format(
        m="normalized_chrom_pathways"
    ) + block.format(m="chrom_pathways")
