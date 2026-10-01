# tests/torchcell/datasets/scerevisiae/test_kuzmin2018
# [[tests.torchcell.datasets.scerevisiae.test_kuzmin2018]]
"""Tests for the five Kuzmin 2018 loaders (Smf, Dmf, Tmf, Dmi, Tmi).

The data-gated tests at the bottom (``--data``) call ``preprocess_raw`` /
``create_experiment`` directly on the raw Data S1 table, so nothing is built into LMDB.
They cover both Dmf record kinds: the digenic query x array crosses, which must stay
byte-identical to what is already served, and the double-mutant QUERY strains of the
trigenic screens (one record per distinct strain, which the loader used to drop
entirely). They skip when the ``$DATA_ROOT`` raw mirror is absent (CI without the data).

2026.09.30 (Phase 15): hermetic builds added (the import-time ``load_dotenv()`` and the
module-level skip removed, the data gate moved onto the three mirror tests). The
deletion-only paths are pinned in ``test_kuzmin2018_synthetic.py``; this file pins the
ALLELE, blank-SD, unknown-array, subset and download paths on a four-row
``aao1729_data_s1.tsv`` under ``tmp_path`` in the released column names:

====  ========================  ===============  ===============  ====  ====  =====  =====  =====  =====
row   query strain              query alleles    array strain     type  comb  sd     query  eps    p
====  ========================  ===============  ===============  ====  ====  =====  =====  =====  =====
0     YBR160W+YDL227C_tsq508    cdc28-4+hoΔ      YBR001C_tsa100   dig   0.61  blank  0.83   -0.12  0.03
1     YBR160W+YDL227C_tsq508    cdc28-4+hoΔ      YAL048C_dma5203  dig   0.72  0.05   0.83   0.04   0.4
2     YBR160W+YML107C_tm801     cdc28-4+pml39Δ   YBR001C_tsa100   tri   0.3   0.02   0.55   -0.2   0.001
3     YAR002W+YBR160W_tm802     nup60Δ+cdc28-4   YAL048C_dma5203  tri   0.5   0.04   0.66   0.07   0.2
====  ========================  ===============  ===============  ====  ====  =====  =====  =====  =====

(array single fitness: nth2-5001 0.8, gem1Δ 0.95). ``cdc28-4`` carries no ``Δ``, so it is
an ``SgaAllelePerturbation`` wherever it appears: the digenic query (after the ``hoΔ`` /
``YDL227C`` half is stripped), query 1 of tm801 and query 2 of tm802; ``tsa`` arrays are
``SgaTsAllelePerturbation``. Expected values, all from the table:

- Smf: array singles first in first-seen order (nth2-5001 0.8, gem1_delta 0.95), then the
  digenic query single cdc28-4 0.83 (strain id the full query strain); no SD on any. The
  reference SD is the mean over ALL rows of the SD column, NaN skipped:
  (0.05 + 0.02 + 0.04) / 3 = 0.036666666666666674, se half of it.
- Dmf: rows 0 and 1 (0.61 with the blank SD, stored as None; 0.72 / 0.05, ``sample_sd`` over
  ``N_SAMPLES_COMBINED_MUTANT = 4`` colonies, se 0.05 / 2 = 0.025), then the query pairs
  tm801 (0.55) and tm802 (0.66) with no uncertainty (the 12 to 24 colony bootstrap SD of a
  query strain lives in Data File S4, not a raw file of this loader). The reference SD is
  the digenic mean with the blank skipped: 0.05.
- Tmf: rows 2 and 3 (0.3 / 0.02, se 0.01; 0.5 / 0.04, se 0.02); reference (0.02 + 0.04) / 2
  = 0.03, se 0.015. Dmi: epsilon -0.12 (p 0.03) and 0.04 (p 0.4), edge level. Tmi: tau
  -0.2 (p 0.001) and 0.07 (p 0.2), hyperedge level.
- ``subset_n``: ``df.sample(n, random_state=42)`` picks positions [1, 3] of four rows and
  [1] of two, so Dmf with ``subset_n=2`` keeps records 1 and 3 (0.72, 0.66) and Tmf, Dmi,
  Tmi with ``subset_n=1`` keep their second record.
- ``download``: ``download_url`` is faked to drop a zip holding the table; every loader
  asks for the same hosted archive, extracts it and deletes the zip.
- An array strain that is neither ``tsa`` nor ``dma`` (``YCR002C_sn77``) is refused by
  all five loaders with the same named ``ValueError``.

2026.10.01 (issue #533): the Phase 15 findings are retired. Smf refuses an unknown array
type by name instead of failing with ``UnboundLocalError`` (and the other four refuse it
before their genotype assertion); a blank Dmf or Tmf SD is stored as ``fitness_std`` None,
not a float NaN (the SI defines no meaning for a blank, so it is "no value reported");
a repeated digenic (query strain, array strain) cross is refused by Dmf and Dmi instead of
being stored twice. On the real Data S1 table each of these inputs occurs 0 times (0
other array strains, 0 blank SDs and 0 repeated crosses in 501,510 rows), so no stored
record changes.
"""

import os
import os.path as osp
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from torchcell.datamodels.media import SGA_TM_SELECTION
from torchcell.datamodels.schema import (
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    GeneInteractionExperiment,
    GeneInteractionExperimentReference,
    GeneInteractionPhenotype,
    Genotype,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SgaAllelePerturbation,
    SgaKanMxDeletionPerturbation,
    SgaTsAllelePerturbation,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.scerevisiae import kuzmin2018 as k

RAW_NAME = "aao1729_data_s1.tsv"
URL = (
    "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/data/host/kuzmin2018/"
    "aao1729_data_s1.zip"
)
COLUMNS = [
    "Query strain ID",
    "Query allele name",
    "Array strain ID",
    "Array allele name",
    "Combined mutant type",
    "Combined mutant fitness",
    "Combined mutant fitness standard deviation",
    "Query single/double mutant fitness",
    "Array single mutant fitness",
    "Adjusted genetic interaction score (epsilon or tau)",
    "P-value",
]
TSQ = "YBR160W+YDL227C_tsq508"
TM801 = "YBR160W+YML107C_tm801"
TM802 = "YAR002W+YBR160W_tm802"
TSA = "YBR001C_tsa100"
DMA = "YAL048C_dma5203"
ROWS: list[list[Any]] = [
    [TSQ, "cdc28-4+hoΔ", TSA, "nth2-5001", "digenic", 0.61, None, 0.83, 0.8, -0.12, 0.03],
    [TSQ, "cdc28-4+hoΔ", DMA, "gem1Δ", "digenic", 0.72, 0.05, 0.83, 0.95, 0.04, 0.4],
    [TM801, "cdc28-4+pml39Δ", TSA, "nth2-5001", "trigenic", 0.3, 0.02, 0.55, 0.8, -0.2, 0.001],
    [TM802, "nup60Δ+cdc28-4", DMA, "gem1Δ", "trigenic", 0.5, 0.04, 0.66, 0.95, 0.07, 0.2],
]  # fmt: skip
UNKNOWN_DIGENIC = ["YAR002W+YDL227C_tm3180", "nup60Δ+hoΔ", "YCR002C_sn77", "cdc10-1", "digenic", 0.5, 0.01, 0.9, 0.7, 0.0, 0.9]  # fmt: skip
UNKNOWN_TRIGENIC = ["YAR002W+YML107C_tm2550", "nup60Δ+pml39Δ", "YCR002C_sn77", "cdc10-1", "trigenic", 0.5, 0.01, 0.9, 0.7, 0.0, 0.9]  # fmt: skip

ENVIRONMENT = Environment(media=SGA_TM_SELECTION, temperature=Temperature(value=26))
GENOME = ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")
PUBLICATION = Publication(
    pubmed_id="29674565",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/29674565/",
    doi="10.1126/science.aao1729",
    doi_url="https://www.science.org/doi/10.1126/science.aao1729",
)
NTH2_TS = SgaTsAllelePerturbation(
    systematic_gene_name="YBR001C", perturbed_gene_name="nth2-5001", strain_id=TSA
)
GEM1 = SgaKanMxDeletionPerturbation(
    systematic_gene_name="YAL048C", perturbed_gene_name="gem1_delta", strain_id=DMA
)


def cdc28(strain: str) -> SgaAllelePerturbation:
    """The cdc28-4 allele tagged with ``strain``."""
    return SgaAllelePerturbation(
        systematic_gene_name="YBR160W", perturbed_gene_name="cdc28-4", strain_id=strain
    )


PML39_TM801 = SgaKanMxDeletionPerturbation(
    systematic_gene_name="YML107C", perturbed_gene_name="pml39_delta", strain_id=TM801
)
NUP60_TM802 = SgaKanMxDeletionPerturbation(
    systematic_gene_name="YAR002W", perturbed_gene_name="nup60_delta", strain_id=TM802
)


def write_raw(root: Path, rows: list[list[Any]]) -> None:
    """Write ``rows`` as the tab-separated raw table under ``root/raw``."""
    (root / "raw").mkdir(parents=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(
        root / "raw" / RAW_NAME, sep="\t", index=False
    )


def build(
    tmp_path: Path, cls: type[Any], rows: list[list[Any]] = ROWS, **kw: Any
) -> Any:
    """Build ``cls`` under ``tmp_path/<class name>`` from ``rows``."""
    root = tmp_path / cls.__name__
    write_raw(root, rows)
    return cls(root=str(root), **kw)


def sample_sd(sd: float) -> dict[str, Any]:
    """The uncertainty fields a combined-mutant SD carries (n = 4 colonies)."""
    return {
        "fitness_uncertainty": sd,
        "fitness_uncertainty_type": UncertaintyType.sample_sd,
        "n_samples": 4,
        "sample_unit": SampleUnit.colony,
    }


def fitness(name: str, perturbations: list[Any], value: float, sd: float | None) -> Any:
    """A Kuzmin 2018 fitness experiment dump (SD labeled only when given)."""
    extra = sample_sd(sd) if sd is not None else {}
    return FitnessExperiment(
        dataset_name=name,
        genotype=Genotype(perturbations=perturbations),
        environment=ENVIRONMENT,
        phenotype=FitnessPhenotype(fitness=value, fitness_std=sd, **extra),
    ).model_dump()


def fitness_reference(name: str, sd: float) -> dict[str, Any]:
    """The labeled fitness reference every Kuzmin 2018 fitness record shares."""
    return FitnessExperimentReference(
        dataset_name=name,
        genome_reference=GENOME,
        environment_reference=ENVIRONMENT,
        phenotype_reference=FitnessPhenotype(
            fitness=1.0, fitness_std=sd, **sample_sd(sd)
        ),
    ).model_dump()


def interaction(
    name: str, perturbations: list[Any], value: float, p: float, level: str
) -> dict[str, Any]:
    """A Kuzmin 2018 interaction experiment dump at graph level ``level``."""
    return GeneInteractionExperiment(
        dataset_name=name,
        genotype=Genotype(perturbations=perturbations),
        environment=ENVIRONMENT,
        phenotype=GeneInteractionPhenotype(
            gene_interaction=value, gene_interaction_p_value=p, graph_level=level
        ),
    ).model_dump()


def interaction_reference(name: str, level: str) -> dict[str, Any]:
    """The zero-interaction reference of a Kuzmin 2018 interaction loader."""
    return GeneInteractionExperimentReference(
        dataset_name=name,
        genome_reference=GENOME,
        environment_reference=ENVIRONMENT,
        phenotype_reference=GeneInteractionPhenotype(
            gene_interaction=0.0, gene_interaction_p_value=None, graph_level=level
        ),
    ).model_dump()


def stored(ds: Any, key: str) -> list[dict[str, Any]]:
    """The ``key`` dictionary of every stored record, in LMDB order."""
    return [ds[i][key] for i in range(len(ds))]


def test_smf_allele_query_and_ts_array_singles(tmp_path: Path) -> None:
    """Smf: nth2-5001 (ts, 0.8) and gem1_delta (KanMX, 0.95) array singles, then the
    digenic query single cdc28-4 (allele, 0.83, full query strain id); the shared
    reference SD is the NaN-skipping mean over all four rows, 0.036666666666666674.
    """
    name = "SmfKuzmin2018Dataset"
    ds = build(tmp_path, k.SmfKuzmin2018Dataset)
    assert stored(ds, "experiment") == [
        fitness(name, [NTH2_TS], 0.8, None),
        fitness(name, [GEM1], 0.95, None),
        fitness(name, [cdc28(TSQ)], 0.83, None),
    ]
    reference_sd = (0.05 + 0.02 + 0.04) / 3
    assert reference_sd == 0.036666666666666674
    assert stored(ds, "reference") == [fitness_reference(name, reference_sd)] * 3
    assert ds[0]["reference"]["phenotype_reference"]["fitness_se"] == reference_sd / 2
    assert stored(ds, "publication") == [PUBLICATION.model_dump()] * 3
    assert sorted(ds.gene_set) == ["YAL048C", "YBR001C", "YBR160W"]


def test_dmf_allele_pairs_and_blank_sd(tmp_path: Path) -> None:
    """Dmf: the two digenic crosses (allele query x ts array, allele query x KanMX
    array), then the two trigenic query strains, each an allele + deletion pair tagged
    with the full strain id and carrying no uncertainty. The reference SD is the digenic
    mean with the blank skipped, 0.05. Record 0's blank SD is stored as ``fitness_std``
    None with every uncertainty field None, not as a float NaN.
    """
    name = "DmfKuzmin2018Dataset"
    ds = build(tmp_path, k.DmfKuzmin2018Dataset)
    experiments = stored(ds, "experiment")
    assert experiments[0]["phenotype"]["fitness_std"] is None
    expected = [
        fitness(name, [cdc28(TSQ), NTH2_TS], 0.61, None),
        fitness(name, [cdc28(TSQ), GEM1], 0.72, 0.05),
        fitness(name, [cdc28(TM801), PML39_TM801], 0.55, None),
        fitness(name, [NUP60_TM802, cdc28(TM802)], 0.66, None),
    ]
    assert experiments == expected
    assert [e["phenotype"]["fitness_se"] for e in experiments] == [
        None,
        0.025,
        None,
        None,
    ]
    assert stored(ds, "reference") == [fitness_reference(name, 0.05)] * 4
    assert ds.df["record_kind"].tolist() == [
        k.RECORD_KIND_DIGENIC_ARRAY_CROSS,
        k.RECORD_KIND_DIGENIC_ARRAY_CROSS,
        k.RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN,
        k.RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN,
    ]


def test_tmf_allele_in_either_query_slot(tmp_path: Path) -> None:
    """Tmf: tm801 has the allele as query 1 (with the ts array), tm802 as query 2; SDs
    0.02 and 0.04 are ``sample_sd`` over 4 colonies (se 0.01, 0.02); the reference SD is
    their mean, 0.03 (se 0.015).
    """
    name = "TmfKuzmin2018Dataset"
    ds = build(tmp_path, k.TmfKuzmin2018Dataset)
    assert stored(ds, "experiment") == [
        fitness(name, [cdc28(TM801), PML39_TM801, NTH2_TS], 0.3, 0.02),
        fitness(name, [NUP60_TM802, cdc28(TM802), GEM1], 0.5, 0.04),
    ]
    assert [ds[i]["experiment"]["phenotype"]["fitness_se"] for i in range(2)] == [
        0.01,
        0.02,
    ]
    assert stored(ds, "reference") == [fitness_reference(name, 0.03)] * 2
    assert ds[0]["reference"]["phenotype_reference"]["fitness_se"] == 0.015


def test_tmf_blank_sd_is_stored_as_none(tmp_path: Path) -> None:
    """A trigenic row with a blank SD (tm801, 0.3) is stored with ``fitness_std`` None and
    no uncertainty, not a float NaN; the reference SD is the mean of the one reported SD,
    0.04 (se 0.02).
    """
    name = "TmfKuzmin2018Dataset"
    blank = [*ROWS[2][:6], None, *ROWS[2][7:]]
    ds = build(tmp_path, k.TmfKuzmin2018Dataset, [blank, ROWS[3]])
    experiments = stored(ds, "experiment")
    assert experiments[0]["phenotype"]["fitness_std"] is None
    assert experiments == [
        fitness(name, [cdc28(TM801), PML39_TM801, NTH2_TS], 0.3, None),
        fitness(name, [NUP60_TM802, cdc28(TM802), GEM1], 0.5, 0.04),
    ]
    assert stored(ds, "reference") == [fitness_reference(name, 0.04)] * 2


def test_dmi_and_tmi_allele_records(tmp_path: Path) -> None:
    """Dmi stores the two digenic rows as edge-level epsilons (-0.12 p 0.03, 0.04 p 0.4);
    Tmi the two trigenic rows as hyperedge-level taus (-0.2 p 0.001, 0.07 p 0.2); each
    against a zero-interaction reference at its own level.
    """
    dmi = build(tmp_path, k.DmiKuzmin2018Dataset)
    assert stored(dmi, "experiment") == [
        interaction("DmiKuzmin2018Dataset", [cdc28(TSQ), NTH2_TS], -0.12, 0.03, "edge"),
        interaction("DmiKuzmin2018Dataset", [cdc28(TSQ), GEM1], 0.04, 0.4, "edge"),
    ]
    assert (
        stored(dmi, "reference")
        == [interaction_reference("DmiKuzmin2018Dataset", "edge")] * 2
    )
    tmi = build(tmp_path, k.TmiKuzmin2018Dataset)
    assert stored(tmi, "experiment") == [
        interaction(
            "TmiKuzmin2018Dataset",
            [cdc28(TM801), PML39_TM801, NTH2_TS],
            -0.2,
            0.001,
            "hyperedge",
        ),
        interaction(
            "TmiKuzmin2018Dataset",
            [NUP60_TM802, cdc28(TM802), GEM1],
            0.07,
            0.2,
            "hyperedge",
        ),
    ]
    assert (
        stored(tmi, "reference")
        == [interaction_reference("TmiKuzmin2018Dataset", "hyperedge")] * 2
    )


@pytest.mark.parametrize(
    ("cls", "rows"),
    [
        (k.SmfKuzmin2018Dataset, [UNKNOWN_DIGENIC]),
        (k.DmfKuzmin2018Dataset, [UNKNOWN_DIGENIC, UNKNOWN_TRIGENIC]),
        (k.DmiKuzmin2018Dataset, [UNKNOWN_DIGENIC]),
        (k.TmfKuzmin2018Dataset, [UNKNOWN_TRIGENIC]),
        (k.TmiKuzmin2018Dataset, [UNKNOWN_TRIGENIC]),
    ],
)
def test_unknown_array_strain_type_refuses_the_build(
    tmp_path: Path, cls: type[Any], rows: list[list[Any]]
) -> None:
    """``YCR002C_sn77`` is neither ``tsa`` nor ``dma``, so nothing records its
    perturbation class: all five loaders refuse it by name while classifying the array
    column, before any genotype is built (Smf used to fail with ``UnboundLocalError``,
    the other four at their genotype-size assertion). No LMDB store is written.
    """
    with pytest.raises(ValueError) as info:
        build(tmp_path, cls, rows)
    assert str(info.value) == (
        "array strain 'YCR002C_sn77' is neither a 'tsa' (temperature-sensitive allele) "
        "nor a 'dma' (KanMX deletion) strain; its perturbation type is unknown"
    )
    assert not (tmp_path / cls.__name__ / "processed" / "lmdb").exists()


@pytest.mark.parametrize("cls", [k.DmfKuzmin2018Dataset, k.DmiKuzmin2018Dataset])
def test_repeated_digenic_cross_is_refused_by_dmf_and_dmi(
    tmp_path: Path, cls: type[Any]
) -> None:
    """The cdc28-4 x gem1 cross listed twice, the second time with a different combined
    fitness (0.73 against 0.72), is refused by name, naming the repeated (query strain,
    array strain) pair: the key is the cross, not the whole row, so two disagreeing
    measurements of one cross are not both stored. No LMDB store is written.
    """
    repeat = [*ROWS[1][:5], 0.73, *ROWS[1][6:]]
    with pytest.raises(ValueError) as info:
        build(tmp_path, cls, [ROWS[1], repeat])
    assert str(info.value) == (
        "Kuzmin 2018 Data S1 lists 1 digenic cross(es) more than once (query strain, "
        "array strain): [('YBR160W+YDL227C_tsq508', 'YAL048C_dma5203')]"
    )
    assert not (tmp_path / cls.__name__ / "processed" / "lmdb").exists()


def test_repeated_digenic_row_gives_smf_one_single_per_allele(tmp_path: Path) -> None:
    """Smf deduplicates the array allele, so the repeated cdc28-4 x gem1 row still gives
    one gem1 single plus the one query single.
    """
    rows = [ROWS[1], ROWS[1]]
    smf = build(tmp_path, k.SmfKuzmin2018Dataset, rows)
    assert stored(smf, "experiment") == [
        fitness("SmfKuzmin2018Dataset", [GEM1], 0.95, None),
        fitness("SmfKuzmin2018Dataset", [cdc28(TSQ)], 0.83, None),
    ]


def test_subset_n_samples_rows_with_seed_42(tmp_path: Path) -> None:
    """``subset_n`` samples the preprocessed frame with ``random_state=42`` AFTER the
    reference SD is taken: Dmf ``subset_n=2`` keeps positions [1, 3] (0.72 and the tm802
    query strain 0.66) and still carries the full-frame reference SD 0.05; Tmf, Dmi and
    Tmi ``subset_n=1`` keep position 1 (0.5, epsilon 0.04, tau 0.07). ``data.csv`` holds
    only the sampled rows.
    """
    assert pd.DataFrame({"x": range(4)}).sample(
        n=2, random_state=42
    ).index.tolist() == [1, 3]
    dmf = build(tmp_path, k.DmfKuzmin2018Dataset, subset_n=2)
    assert stored(dmf, "experiment") == [
        fitness("DmfKuzmin2018Dataset", [cdc28(TSQ), GEM1], 0.72, 0.05),
        fitness("DmfKuzmin2018Dataset", [NUP60_TM802, cdc28(TM802)], 0.66, None),
    ]
    assert (
        stored(dmf, "reference")
        == [fitness_reference("DmfKuzmin2018Dataset", 0.05)] * 2
    )
    csv = pd.read_csv(Path(dmf.preprocess_dir) / "data.csv")
    assert csv["Query strain ID"].tolist() == [TSQ, TM802]
    tmf = build(tmp_path, k.TmfKuzmin2018Dataset, subset_n=1)
    assert [e["phenotype"]["fitness"] for e in stored(tmf, "experiment")] == [0.5]
    assert stored(tmf, "reference") == [fitness_reference("TmfKuzmin2018Dataset", 0.03)]
    dmi = build(tmp_path, k.DmiKuzmin2018Dataset, subset_n=1)
    assert [e["phenotype"]["gene_interaction"] for e in stored(dmi, "experiment")] == [
        0.04
    ]
    tmi = build(tmp_path, k.TmiKuzmin2018Dataset, subset_n=1)
    assert [e["phenotype"]["gene_interaction"] for e in stored(tmi, "experiment")] == [
        0.07
    ]


@pytest.mark.parametrize(
    ("cls", "n"),
    [
        (k.SmfKuzmin2018Dataset, 3),
        (k.DmfKuzmin2018Dataset, 4),
        (k.TmfKuzmin2018Dataset, 2),
        (k.DmiKuzmin2018Dataset, 2),
        (k.TmiKuzmin2018Dataset, 2),
    ],
)
def test_download_extracts_the_hosted_zip_and_removes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cls: type[Any], n: int
) -> None:
    """With no raw file and no ``TC_DATA_URL``, the build calls ``download_url`` once
    with the hosted archive URL and the raw directory; the zip is extracted there and
    deleted, and the build proceeds on the extracted table.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    table = tmp_path / "table.tsv"
    pd.DataFrame(ROWS, columns=COLUMNS).to_csv(table, sep="\t", index=False)
    calls: list[tuple[str, str]] = []

    def fake_download_url(url: str, folder: str) -> str:
        calls.append((url, folder))
        path = osp.join(folder, "aao1729_data_s1.zip")
        with zipfile.ZipFile(path, "w") as archive:
            archive.write(table, RAW_NAME)
        return path

    monkeypatch.setattr(k, "download_url", fake_download_url)
    root = tmp_path / cls.__name__
    ds = cls(root=str(root))
    assert calls == [(URL, str(root / "raw"))]
    assert cls.url == URL
    assert sorted(os.listdir(root / "raw")) == [RAW_NAME]
    assert len(ds) == n


@pytest.mark.parametrize(
    ("cls", "experiment_class", "reference_class"),
    [
        (k.SmfKuzmin2018Dataset, FitnessExperiment, FitnessExperimentReference),
        (k.DmfKuzmin2018Dataset, FitnessExperiment, FitnessExperimentReference),
        (k.TmfKuzmin2018Dataset, FitnessExperiment, FitnessExperimentReference),
        (
            k.DmiKuzmin2018Dataset,
            GeneInteractionExperiment,
            GeneInteractionExperimentReference,
        ),
        (
            k.TmiKuzmin2018Dataset,
            GeneInteractionExperiment,
            GeneInteractionExperimentReference,
        ),
    ],
)
def test_items_retype_through_the_declared_classes(
    tmp_path: Path,
    cls: type[Any],
    experiment_class: type[Any],
    reference_class: type[Any],
) -> None:
    """``transform_item`` rebuilds record 1 of each loader through its declared classes,
    and both dump back to exactly the stored dictionaries (an interaction item read
    through a fitness class would fail validation or drop its label).
    """
    ds = build(tmp_path, cls)
    item = ds[1]
    typed = ds.transform_item(item)
    assert type(typed["experiment"]) is experiment_class
    assert type(typed["reference"]) is reference_class
    assert typed["experiment"].model_dump() == item["experiment"]
    assert typed["reference"].model_dump() == item["reference"]
    assert typed["publication"] == PUBLICATION


# ---------------------------------------------------------------------------------------
# Data-gated tests on the real Data S1 table.
# ---------------------------------------------------------------------------------------

_RAW = osp.join(
    os.environ["DATA_ROOT"], "data/torchcell/dmf_kuzmin2018/raw/aao1729_data_s1.tsv"
)
_needs_raw = pytest.mark.skipif(
    not osp.exists(_RAW),
    reason=f"requires the Kuzmin 2018 raw Data S1 TSV at {_RAW} (absent in CI)",
)

# The digenic row used by the regression test, addressed by strain ids (stable under
# any reordering). Values read off the raw table.
_DIGENIC_QUERY_STRAIN = "YAR002W+YDL227C_tm3180"
_DIGENIC_ARRAY_STRAIN = "YAL048C_dma5203"
_DIGENIC_FITNESS = 0.8103
_DIGENIC_FITNESS_SD = 0.0463

# A double-mutant query strain of the trigenic screens, with its released fitness.
_QUERY_STRAIN = "YAR002W+YML107C_tm2550"
_QUERY_STRAIN_FITNESS = 0.5128


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    """The raw Kuzmin 2018 Data S1 table."""
    return pd.read_csv(_RAW, sep="\t")


@pytest.fixture(scope="module")
def built(raw: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
    """The loader instance (for phenotype_reference_std) and its preprocessed frame."""
    from torchcell.datasets.scerevisiae.kuzmin2018 import DmfKuzmin2018Dataset

    dataset = DmfKuzmin2018Dataset.__new__(DmfKuzmin2018Dataset)
    return dataset, dataset.preprocess_raw(raw.copy())


@pytest.mark.data
@_needs_raw
def test_trigenic_query_strains_are_kept_one_record_each(
    raw: pd.DataFrame, built: tuple[Any, pd.DataFrame]
) -> None:
    """The trigenic rows are no longer dropped, and collapse to one record/strain."""
    from torchcell.datasets.scerevisiae.kuzmin2018 import (
        RECORD_KIND_DIGENIC_ARRAY_CROSS,
        RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN,
    )

    _, df = built
    trigenic = raw[raw["Combined mutant type"] == "trigenic"]
    n_query_strains = trigenic.dropna(subset=["Query single/double mutant fitness"])[
        "Query strain ID"
    ].nunique()

    counts = df["record_kind"].value_counts().to_dict()
    assert counts[RECORD_KIND_DIGENIC_ARRAY_CROSS] == int(
        (raw["Combined mutant type"] == "digenic").sum()
    )
    assert counts[RECORD_KIND_DIGENIC_ARRAY_CROSS] == 410_399
    assert counts[RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN] == n_query_strains == 172

    # One record per strain, NOT one per trigenic row (91,111 rows, 182 strains, of
    # which 172 report a fitness).
    assert len(trigenic) == 91_111
    query_rows = df[df["record_kind"] == RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN]
    assert query_rows["Query strain ID"].is_unique
    assert query_rows["Query single/double mutant fitness"].notna().all()


@pytest.mark.data
@_needs_raw
def test_query_strain_record_is_the_query_pair(built: tuple[Any, pd.DataFrame]) -> None:
    """Genotype is the two QUERY genes, both tagged with the full query strain id."""
    from torchcell.datamodels.schema import Genotype
    from torchcell.datasets.scerevisiae.kuzmin2018 import (
        RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN,
        DmfKuzmin2018Dataset,
    )

    dataset, df = built
    rows = df[
        (df["record_kind"] == RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN)
        & (df["Query strain ID"] == _QUERY_STRAIN)
    ]
    assert len(rows) == 1
    row = rows.iloc[0]

    experiment, _, _ = DmfKuzmin2018Dataset.create_experiment(
        "dmf_kuzmin2018", row, phenotype_reference_std=dataset.phenotype_reference_std
    )
    genotype = experiment.genotype
    assert isinstance(genotype, Genotype)
    perturbations = genotype.perturbations
    assert len(perturbations) == 2
    assert [p.systematic_gene_name for p in perturbations] == ["YAR002W", "YML107C"]
    assert [p.perturbed_gene_name for p in perturbations] == [
        "nup60_delta",
        "pml39_delta",
    ]
    assert {p.model_dump()["strain_id"] for p in perturbations} == {_QUERY_STRAIN}
    # The array gene of the trigenic rows this value was read off is NOT in the
    # genotype.
    assert row["Array systematic name"] not in {
        p.systematic_gene_name for p in perturbations
    }

    # Released value, no uncertainty (the SD lives in Data File S4, not a raw file
    # of this loader).
    assert experiment.phenotype.fitness == pytest.approx(_QUERY_STRAIN_FITNESS)
    assert experiment.phenotype.fitness_std is None
    assert experiment.phenotype.fitness_se is None
    assert experiment.phenotype.fitness_uncertainty is None
    assert experiment.phenotype.fitness_uncertainty_type is None


@pytest.mark.data
@_needs_raw
def test_digenic_record_unchanged(built: tuple[Any, pd.DataFrame]) -> None:
    """Regression: a digenic record equals the object the old path produced."""
    from torchcell.datamodels.media import SGA_TM_SELECTION
    from torchcell.datamodels.schema import (
        Environment,
        FitnessExperiment,
        FitnessPhenotype,
        Genotype,
        SampleUnit,
        SgaKanMxDeletionPerturbation,
        Temperature,
        UncertaintyType,
    )
    from torchcell.datasets.scerevisiae.kuzmin2018 import (
        RECORD_KIND_DIGENIC_ARRAY_CROSS,
        DmfKuzmin2018Dataset,
    )

    dataset, df = built
    rows = df[
        (df["record_kind"] == RECORD_KIND_DIGENIC_ARRAY_CROSS)
        & (df["Query strain ID"] == _DIGENIC_QUERY_STRAIN)
        & (df["Array strain ID"] == _DIGENIC_ARRAY_STRAIN)
    ]
    assert len(rows) == 1

    experiment, _, _ = DmfKuzmin2018Dataset.create_experiment(
        "dmf_kuzmin2018",
        rows.iloc[0],
        phenotype_reference_std=dataset.phenotype_reference_std,
    )

    expected = FitnessExperiment(
        dataset_name="dmf_kuzmin2018",
        genotype=Genotype(
            perturbations=[
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name="YAR002W",
                    perturbed_gene_name="nup60_delta",
                    strain_id=_DIGENIC_QUERY_STRAIN,
                ),
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name="YAL048C",
                    perturbed_gene_name="gem1_delta",
                    strain_id=_DIGENIC_ARRAY_STRAIN,
                ),
            ]
        ),
        environment=Environment(
            media=SGA_TM_SELECTION, temperature=Temperature(value=26)
        ),
        phenotype=FitnessPhenotype(
            fitness=_DIGENIC_FITNESS,
            fitness_std=_DIGENIC_FITNESS_SD,
            fitness_uncertainty=_DIGENIC_FITNESS_SD,
            fitness_uncertainty_type=UncertaintyType.sample_sd,
            n_samples=4,
            sample_unit=SampleUnit.colony,
        ),
    )
    assert experiment.model_dump() == expected.model_dump()
