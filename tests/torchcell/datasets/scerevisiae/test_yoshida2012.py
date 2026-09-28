# tests/torchcell/datasets/scerevisiae/test_yoshida2012.py
# [[tests.torchcell.datasets.scerevisiae.test_yoshida2012]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_yoshida2012.py
"""Hermetic build of the Yoshida 2012 organic-acid loader from its embedded Table 3.

The only raw file is ``paper.pdf``, whose presence is all PyG checks; a placeholder is
written so ``download()`` is never called, and the values come from the module-level
``TABLE_3`` literal. ``build_metabolite_s_id_map`` (which loads Yeast9) is replaced by a
stub that records its argument and returns ``s_0001`` to ``s_0005`` for the five acids.
The genome stub carries ``alias_to_systematic`` for the 16 common names; ``YDR379C-A``
matches the systematic-name regex and is never looked up.

Record 0 = ASM4 (Table 3 mean, SD; n = 3):
    acetate 4.02 +/- 0.30, citrate 0.10 +/- 0.03, malate 0.15 +/- 0.02,
    pyruvate 0.13 +/- 0.04, succinate 1.31 +/- 0.06, phosphate 0.78 +/- 0.07
SE = SD / sqrt(3) per analyte; ``target_metabolite_ids`` covers the five acids only.
Reference = the WT row for every record: acetate 4.21 +/- 0.30, citrate 0.10 +/- 0.04,
malate 0.15 +/- 0.01, pyruvate 0.18 +/- 0.05, succinate 1.29 +/- 0.05, phosphate 0.69
+/- 0.11, same SE rule; BY4742, static liquid YPD at 25 C.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import cast

import pytest

from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.scerevisiae import yoshida2012 as m
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

_ORF_BY_NAME = {
    "ASM4": "YDL088C",
    "EMI5": "YOL071W",
    "GTR1": "YML121W",
    "GTR2": "YGR163W",
    "LIP5": "YOR196C",
    "LSM1": "YJL124C",
    "MKS1": "YNL076W",
    "NFU1": "YKL040C",
    "PCK1": "YKR097W",
    "PHO85": "YPL031C",
    "PLM2": "YDR501W",
    "RTG1": "YOL067C",
    "RTG2": "YGL252C",
    "TIF3": "YPR163C",
    "UBA3": "YPR066W",
    "UBP3": "YER151C",
}
_S_IDS = {
    "acetate": "s_0001",
    "citrate": "s_0002",
    "malate": "s_0003",
    "pyruvate": "s_0004",
    "succinate": "s_0005",
}


class _StubGenome:
    def __init__(self, names: dict[str, str]) -> None:
        self.alias_to_systematic = {k: [v] for k, v in names.items()}


def _genome(names: dict[str, str] = _ORF_BY_NAME) -> SCerevisiaeGenome:
    return cast(SCerevisiaeGenome, _StubGenome(names))


@pytest.fixture
def s_id_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str]]:
    calls: list[dict[str, str]] = []

    def fake_map(kegg_by_metabolite: dict[str, str]) -> dict[str, str]:
        calls.append(dict(kegg_by_metabolite))
        return dict(_S_IDS)

    monkeypatch.setattr(m, "build_metabolite_s_id_map", fake_map)
    return calls


def _root(tmp_path: Path, slug: str = "organic_acid_yoshida2012") -> Path:
    root = tmp_path / slug
    (root / "raw").mkdir(parents=True)
    (root / "raw" / m.PDF_FILENAME).write_bytes(b"%PDF-1.4 synthetic placeholder")
    return root


@pytest.fixture
def dataset(
    tmp_path: Path, s_id_calls: list[dict[str, str]]
) -> m.OrganicAcidYoshida2012Dataset:
    return m.OrganicAcidYoshida2012Dataset(root=str(_root(tmp_path)), genome=_genome())


_ENVIRONMENT = Environment(
    media=Media(name="YPD", state="liquid", is_synthetic=False),
    temperature=Temperature(value=25),
)
_PUBLICATION = Publication(
    pubmed_id="22277779",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/22277779/",
    doi="10.1016/j.jbiosc.2011.12.017",
    doi_url="https://doi.org/10.1016/j.jbiosc.2011.12.017",
).model_dump()
_ANALYTES = ["acetate", "citrate", "malate", "pyruvate", "succinate", "phosphate"]


def _phenotype(means: list[float], sds: list[float]) -> MetabolitePhenotype:
    return MetabolitePhenotype(
        metabolite_level=dict(zip(_ANALYTES, means, strict=True)),
        metabolite_level_se={
            a: sd / math.sqrt(3) for a, sd in zip(_ANALYTES, sds, strict=True)
        },
        n_replicates=dict.fromkeys(_ANALYTES, 3),
        measurement_type="hplc_organic_acid_titer_mM",
        target_metabolite_ids=dict(_S_IDS),
    )


_WT_PHENOTYPE = _phenotype(
    [4.21, 0.10, 0.15, 0.18, 1.29, 0.69], [0.30, 0.04, 0.01, 0.05, 0.05, 0.11]
)
_REFERENCE = MetaboliteExperimentReference(
    dataset_name="OrganicAcidYoshida2012Dataset",
    genome_reference=ReferenceGenome(
        species="Saccharomyces cerevisiae", strain="BY4742"
    ),
    environment_reference=_ENVIRONMENT,
    phenotype_reference=_WT_PHENOTYPE,
).model_dump()


def test_record_0_is_asm4_with_se_from_sd_over_sqrt_3(
    dataset: m.OrganicAcidYoshida2012Dataset, s_id_calls: list[dict[str, str]]
) -> None:
    """The mapper is called once with ``ACID_KEGG_IDS``; record 0 stores the six ASM4
    analytes (OD dropped), SE = SD / sqrt(3), n = 3, Yeast9 ids for the five acids and
    none for phosphate, against the measured WT reference.
    """
    assert s_id_calls == [m.ACID_KEGG_IDS]
    assert len(dataset) == 17
    assert (
        dataset[0]["experiment"]
        == MetaboliteExperiment(
            dataset_name="OrganicAcidYoshida2012Dataset",
            genotype=Genotype(
                perturbations=[
                    KanMxDeletionPerturbation(
                        systematic_gene_name="YDL088C", perturbed_gene_name="ASM4"
                    )
                ]
            ),
            environment=_ENVIRONMENT,
            phenotype=_phenotype(
                [4.02, 0.10, 0.15, 0.13, 1.31, 0.78],
                [0.30, 0.03, 0.02, 0.04, 0.06, 0.07],
            ),
        ).model_dump()
    )
    assert dataset[0]["reference"] == _REFERENCE
    assert dataset[0]["publication"] == _PUBLICATION
    assert dataset[0]["experiment"]["phenotype"]["metabolite_level_se"]["acetate"] == (
        0.30 / math.sqrt(3)
    )


def test_seventeen_records_in_table_order_with_one_shared_reference(
    dataset: m.OrganicAcidYoshida2012Dataset,
) -> None:
    """Records follow ``TABLE_3`` order minus WT; the last is YDR379C-A, resolved by the
    regex rather than the alias table, with the ORF as its perturbed name. Every strain
    measures all six analytes, so the WT reference is shared: one index entry.
    """
    genes = [g for g in m.TABLE_3 if g != "WT"]
    stored = [
        (
            dataset[i]["experiment"]["genotype"]["perturbations"][0][
                "systematic_gene_name"
            ],
            dataset[i]["experiment"]["genotype"]["perturbations"][0][
                "perturbed_gene_name"
            ],
        )
        for i in range(17)
    ]
    assert stored == [(_ORF_BY_NAME.get(g, g), g) for g in genes]
    assert stored[-1] == ("YDR379C-A", "YDR379C-A")
    assert dataset.experiment_class is MetaboliteExperiment
    assert dataset.reference_class is MetaboliteExperimentReference
    preprocess = Path(dataset.root) / "preprocess"
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [list(range(17))]
    assert (preprocess / "data.csv").read_text() == "orf,gene\n" + "".join(
        f"{_ORF_BY_NAME.get(g, g)},{g}\n" for g in genes
    )
    assert json.loads((preprocess / "gene_set.json").read_text()) == sorted(
        [*_ORF_BY_NAME.values(), "YDR379C-A"]
    )
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == "organic_acid_yoshida2012"
    assert manifest["loader_class"] == "OrganicAcidYoshida2012Dataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.yoshida2012"


def test_unresolvable_common_name_raises(
    tmp_path: Path, s_id_calls: list[dict[str, str]]
) -> None:
    names = {k: v for k, v in _ORF_BY_NAME.items() if k != "ASM4"}
    with pytest.raises(
        RuntimeError, match="Yoshida2012: could not resolve gene name 'ASM4'"
    ):
        m.OrganicAcidYoshida2012Dataset(
            root=str(_root(tmp_path)), genome=_genome(names)
        )


def test_requires_an_injected_genome(tmp_path: Path) -> None:
    """The genome check precedes the Yeast9 mapper call, so no stub is needed here."""
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "OrganicAcidYoshida2012Dataset requires an injected SCerevisiaeGenome to "
            "resolve common gene names to systematic ORF ids (Table 3 uses names)."
        ),
    ):
        m.OrganicAcidYoshida2012Dataset(root=str(_root(tmp_path)), genome=None)


def test_download_stages_the_mirror_pdf_only_after_verifying_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, s_id_calls: list[dict[str, str]]
) -> None:
    """An empty root looks for ``$DATA_ROOT/torchcell-library/<key>/paper.pdf`` and names
    it when absent; a mirror PDF holding ``b"wrong pdf"`` is rejected with its digest
    (43d5ed94...) and not copied into ``raw/``.

    Finding: with ``paper.pdf`` already in ``raw/`` the method returns without hashing
    it (source line 334), so the staged copy is not re-verified on later builds.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    mirror = data_root / "torchcell-library" / m.LIBRARY_CITATION_KEY / m.PDF_FILENAME
    with pytest.raises(
        RuntimeError, match=re.escape(f"Yoshida2012 mirror PDF not found: {mirror}")
    ):
        m.OrganicAcidYoshida2012Dataset(root=str(tmp_path / "empty"), genome=_genome())
    mirror.parent.mkdir(parents=True)
    mirror.write_bytes(b"wrong pdf")
    digest = hashlib.sha256(b"wrong pdf").hexdigest()
    assert digest.startswith("43d5ed94")
    with pytest.raises(
        RuntimeError,
        match=f"Yoshida2012 paper.pdf sha256 mismatch: got {digest}, "
        f"expected {m.PDF_SHA256}",
    ):
        m.OrganicAcidYoshida2012Dataset(root=str(tmp_path / "empty2"), genome=_genome())
    assert not (tmp_path / "empty2" / "raw" / m.PDF_FILENAME).exists()
    dataset = m.OrganicAcidYoshida2012Dataset(
        root=str(_root(tmp_path)), genome=_genome()
    )
    dest = Path(dataset.root) / "raw" / m.PDF_FILENAME
    assert hashlib.sha256(dest.read_bytes()).hexdigest() != m.PDF_SHA256
    dataset.download()
    assert dest.read_bytes() == b"%PDF-1.4 synthetic placeholder"
