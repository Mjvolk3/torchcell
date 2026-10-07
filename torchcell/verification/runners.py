# torchcell/verification/runners
# [[torchcell.verification.runners]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/runners
"""Runnable L0-L4 verification over the abstract's built datasets (roadmap WS5/WS6).

This is the in-`src` home for the verification RUNNERS (the harness logic that loads
each built LMDB, runs the per-family verifier from
:mod:`torchcell.verification.{expression,morphology}`, adds the cross-source L4 checks,
and writes a ``verification_report.json`` sibling of each
``experiment_reference_index.json``). Keeping the runners in the package -- not in
``scripts/`` -- lets them be imported, tested, and reused.

Run everything::

    ~/miniconda3/envs/torchcell/bin/python -m torchcell.verification.runners

Or import and call :func:`run_expression`, :func:`run_morphology`, or :func:`run_all`.

The two bioproduction families work the other way round: each of their datasets already
carries its own L0-L3 gate and its own raw-mirror cross-source oracles in the loader
module that owns its readers, so :func:`run_product_titer` and
:func:`run_bacterial_protein_abundance` dispatch to those entry points and add the one
rule that is the family's own, the host-aware locus containment.

The L4 gene universe and the canonical-name resolver belong to the host a record is
written against. The yeast runners use S288C (:func:`_sgd_gene_set`, :func:`_genome`);
a bacterial runner selects both from each record's own ``genome_reference``
(:func:`_gene_set_for_reference`, :func:`_genome_for_reference`), whose
``assembly_set`` names the strain: :func:`_ecoli_k12_gene_set` for MG1655 or BW25113,
:func:`_ecoli_rel606_gene_set` for E. coli B REL606, :func:`_pputida_gene_set` for KT2440.
"""

from __future__ import annotations

import gzip
import os
import os.path as osp
import pickle
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import lmdb

from torchcell.data.experiment_dataset import resolve_interned
from torchcell.datamodels.schema import BACTERIAL_ASSEMBLY_SETS
from torchcell.sequence.genome.bacterial import BacterialAssembly
from torchcell.sequence.genome.ecoli.k12 import (
    BW25113_ASSEMBLY,
    MG1655_ASSEMBLY,
    EcoliK12StrainName,
)
from torchcell.sequence.genome.ecoli.rel606 import REL606_ASSEMBLY
from torchcell.sequence.genome.pputida.kt2440 import KT2440_ASSEMBLY
from torchcell.sequence.genome.registry import PETER2018_1011, SGD_S288C_R64, resolve
from torchcell.verification.environment_response import (
    verify_environment_response_dataset,
    verify_environment_response_dataset_streaming,
)
from torchcell.verification.expression import (
    measured_gene_universe,
    verify_expression_dataset,
)
from torchcell.verification.fitness import verify_fitness_dataset
from torchcell.verification.levels import l4_cross_source
from torchcell.verification.metabolite import (
    metabolite_gene_set,
    verify_metabolite_dataset,
)
from torchcell.verification.morphology import (
    perturbed_gene_set,
    verify_morphology_dataset,
)
from torchcell.verification.protein import protein_gene_set, verify_protein_dataset
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)
from torchcell.verification.rnaseq import rnaseq_gene_set, verify_rnaseq_dataset
from torchcell.verification.segregant_growth import (
    segregant_gene_set,
    verify_segregant_growth_streaming,
)
from torchcell.verification.visual_score import (
    verify_visual_score_dataset,
    visual_score_gene_set,
)

# --------------------------------------------------------------------------- #
# Expression datasets (WS5)
# --------------------------------------------------------------------------- #
EXPRESSION_DATASETS: dict[str, dict[str, Any]] = {
    "dm_microarray_sameith2015": {
        "root": "data/torchcell/dm_microarray_sameith2015",
        "expected_count": 72,
        "provenance": Provenance(
            source_uri="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42536",
            citation_key="sameithHighresolutionGeneExpression2015",
            method="GEOparse GSE42536 (GPL11232), dye-swap-corrected log2(mutant/refpool)",
            page="Cell Reports, Expression Profiling; SI 'Double mutants - info'",
        ),
    },
    "sm_microarray_sameith2015": {
        "root": "data/torchcell/sm_microarray_sameith2015",
        "expected_count": 82,
        "provenance": Provenance(
            source_uri="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42536",
            citation_key="sameithHighresolutionGeneExpression2015",
            method="GEOparse GSE42536 (GPL11232), dye-swap-corrected log2(mutant/refpool)",
            page="Cell Reports, Expression Profiling; SI 'Single mutants - info'",
        ),
    },
    "microarray_kemmeren2014": {
        "root": "data/torchcell/microarray_kemmeren2014",
        # 1484 = every deletion strain in the source Excel (the current loader's
        # deterministic output; all L0-valid, full 6169-gene vector each). The prior
        # 1450 was a stale oracle from an older LMDB build (loader logic predates it).
        "expected_count": 1484,
        "provenance": Provenance(
            source_uri="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42527",
            citation_key="kemmerenLargeScaleGeneticPerturbations2014",
            method="GEOparse GSE42527/GSE42526 (GPL11232), log2(mutant/wt-reference)",
            page="Cell 2014, Expression Profiling",
        ),
    },
}

# --------------------------------------------------------------------------- #
# Morphology dataset (WS6). The Ohya LMDB lives under the KG-build tree
# (`database/data/...`), which may be owned by the KG-build user and read-only; we READ
# it there and WRITE the report to the writable `data/torchcell/...` tree.
# --------------------------------------------------------------------------- #
OHYA_SOURCE_ROOT = "data/torchcell/scmd_ohya2005"
OHYA_REPORT_ROOT = "data/torchcell/scmd_ohya2005"
OHYA_EXPECTED_COUNT = (
    4718  # all strains retained; ORF names reconciled to R64, 0 dropped
)
OHYA_PROVENANCE = Provenance(
    source_uri="http://www.yeast.ib.k.u-tokyo.ac.jp/SCMD/ (mt4718data.tsv, wt122data.tsv)",
    citation_key="ohyaHighdimensionalLargescalePhenotyping2005a",
    method="SCMD CalMorph TSV (mt4718 mutants + wt122 wildtype)",
    page="Ohya et al. 2005 PNAS; PMID 16365294",
)
# Ohnuki 2022 high-throughput CalMorph morphology (drug-hypersensitive 3Delta quadruple
# deletions). 1982 mutant rows minus 1 all-NaN row (YGL141W) minus 2 background-gene
# collision rows (PDR1=YGL013C, SNQ2=YDR011W already deleted in the 3Delta background).
OHNUKI2022_SOURCE_ROOT = "data/torchcell/scmd_ohnuki2022"
OHNUKI2022_REPORT_ROOT = "data/torchcell/scmd_ohnuki2022"
OHNUKI2022_EXPECTED_COUNT = 1979
OHNUKI2022_PROVENANCE = Provenance(
    source_uri=(
        "http://www.yeast.ib.k.u-tokyo.ac.jp/SCMD/ "
        "(quad1982data.tsv, wt749data.tsv; pj=quadruple)"
    ),
    citation_key="ohnukiHighthroughputPlatformYeast2022",
    method=(
        "SCMD2 CalMorph quadruple-set TSV (quad1982 mutants + wt749 3Delta reference); "
        "raw per-strain CalMorph averages; target KanMX deletion in the fixed 3Delta "
        "pdr1Delta::NATMX pdr3Delta::KlURA3 snq2Delta::KlLEU2 background (strain Y13206); "
        "liquid YPD, 25 C. quad1982data.tsv "
        "sha256=5a1d45005c1249a77b0608ee7e6678c045c14464cf67bfc149b24fcaeb4854c0, "
        "wt749data.tsv "
        "sha256=4603aadf6ae5a3187e447c5d0df1f6bfb30a2292ddb687beb9a871b14769094d"
    ),
    page="npj Syst Biol Appl 2022 8:3 (doi:10.1038/s41540-022-00212-1); PMID 35087094",
)

# Expression deletion datasets whose genes should be CONTAINED in Ohya's set.
DELETION_GENE_SOURCES = {
    "microarray_kemmeren2014": "data/torchcell/microarray_kemmeren2014",
    "sm_microarray_sameith2015": "data/torchcell/sm_microarray_sameith2015",
    "dm_microarray_sameith2015": "data/torchcell/dm_microarray_sameith2015",
}
# Empirically Kemmeren=0.970, SM Sameith=0.988; 0.90 leaves headroom for genes
# legitimately profiled for expression but absent from the morphology screen.
MIN_GENE_OVERLAP = 0.90

# --------------------------------------------------------------------------- #
# Ohnuki 2018 morphology dataset (essential-gene heterozygous diploid CalMorph).
# Same CalMorph verifier as Ohya, but the perturbations are 50%-dosage HIP het
# deletions (EngineeredCopyNumberPerturbation), NOT full KOs, and the gene set is
# essential genes -- so the L4 cross-source check is SGD R64 containment (essential
# ORFs are a subset of the reference gene universe), not the nonessential-deletion
# expression-dataset containment used for Ohya.
# --------------------------------------------------------------------------- #
OHNUKI_SOURCE_ROOT = "data/torchcell/scmd_ohnuki2018"
OHNUKI_REPORT_ROOT = "data/torchcell/scmd_ohnuki2018"
OHNUKI_EXPECTED_COUNT = 1112  # essential-gene het diploids; 114 WT aggregated into ref
OHNUKI_PROVENANCE = Provenance(
    source_uri=(
        "http://www.yeast.ib.k.u-tokyo.ac.jp/SCMD/ "
        "(ess1112data.tsv, wt114data.tsv; SCMD2 portal per Data Availability)"
    ),
    citation_key="ohnukiHighdimensionalSinglecellPhenotyping2018",
    method=(
        "SCMD2 CalMorph TSV (ess1112 essential-gene heterozygous BY4743 diploids + "
        "wt114 WT replicate averages); optimal arm only (liquid YPD, 25 C); genotype = "
        "EngineeredCopyNumberPerturbation(copy 1 of 2, KanMX); ORFs validated vs SGD R64"
    ),
    page=(
        "Ohnuki & Ohya 2018 PLoS Biol 16(5):e2005130 (PMID 29768403); "
        "ess1112data.tsv "
        "sha256=2d168bd1c436c7edae0ab3eb07e99e4c41f3b12f69607a7f3a00092eed7c4b03, "
        "wt114data.tsv "
        "sha256=f48d42da2c727854b83b70e8768cfb42ada0e5118f6074765698d3045862804e"
    ),
)


def _load_interned(abs_root: str) -> dict[str, Any]:
    """Load the sibling ``interned`` env (content-addressed sub-objects) into a RAM dict.

    Empty dict when there is no ``interned`` dir (a legacy inline LMDB), so resolving is a
    no-op there. Mirrors :meth:`ExperimentDataset._load_interned` so these raw cursor readers
    resolve ``{"$ref": ...}`` pointers exactly as ``get_single_item`` does. The interned env
    holds only a handful of constant sub-objects, so loading it whole is cheap even for the
    streaming path (the records are what's large, not the interned set).
    """
    interned_dir = osp.join(abs_root, "processed", "interned")
    if not osp.isdir(interned_dir):
        return {}
    env = lmdb.open(interned_dir, readonly=True, lock=False)
    interned: dict[str, Any] = {}
    with env.begin() as txn:
        for key, value in txn.cursor():
            interned[key.decode()] = pickle.loads(value)
    env.close()
    return interned


def load_records(abs_root: str) -> list[dict[str, Any]]:
    """Read every LMDB entry under ``<abs_root>/processed/lmdb``, resolving interned sub-objects."""
    interned = _load_interned(abs_root)
    env = lmdb.open(osp.join(abs_root, "processed", "lmdb"), readonly=True, lock=False)
    records: list[dict[str, Any]] = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for _, value in cursor:
            records.append(resolve_interned(pickle.loads(value), interned))
    env.close()
    return records


def stream_records(abs_root: str) -> Any:
    """Yield every LMDB entry under ``<abs_root>/processed/lmdb``, resolving interned sub-objects.

    Memory-bounded alternative to :func:`load_records` for very large datasets (e.g.
    the 30M-record Hoepfner HIP-HOP atlas) whose full materialization would exceed RAM.
    The interned set is small and loaded once up front; only the records stream.
    """
    interned = _load_interned(abs_root)
    env = lmdb.open(osp.join(abs_root, "processed", "lmdb"), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                yield resolve_interned(pickle.loads(value), interned)
    finally:
        env.close()


def _write_report(report: VerificationReport, report_dir: str) -> str:
    """Write a report as ``verification_report.json`` in ``report_dir``; return path."""
    os.makedirs(report_dir, exist_ok=True)
    out = osp.join(report_dir, "verification_report.json")
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return out


def run_expression(data_root: str) -> bool:
    """Verify all expression datasets (L0-L4) and write their reports. True if all pass."""
    reports: dict[str, VerificationReport] = {}
    universes: dict[str, set[str]] = {}
    for name, spec in EXPRESSION_DATASETS.items():
        abs_root = osp.join(data_root, spec["root"])
        records = load_records(abs_root)
        report = verify_expression_dataset(
            records,
            dataset_name=name,
            provenance=spec["provenance"],
            expected_count=spec["expected_count"],
        )
        reports[name] = report
        universes[name] = measured_gene_universe(records)

    # L4: every expression dataset measures the same platform gene universe.
    ref_name = next(iter(universes))
    ref_universe = universes[ref_name]
    for name, universe in universes.items():
        if name == ref_name:
            continue
        all_genes = sorted(ref_universe | universe)
        shared = [
            (g, 1.0 if g in ref_universe else 0.0, 1.0 if g in universe else 0.0)
            for g in all_genes
        ]
        result = l4_cross_source(shared, tol=0.0).model_copy(
            update={"name": f"gene_universe_vs_{ref_name}"}
        )
        reports[name].add(result)

    all_passed = True
    for name, report in reports.items():
        report_dir = osp.join(
            data_root, EXPRESSION_DATASETS[name]["root"], "preprocess"
        )
        out = _write_report(report, report_dir)
        print(report.summary())
        print(f"  -> wrote {out}\n")
        all_passed = all_passed and report.passed
    return all_passed


def _l4_gene_containment(
    ohya_genes: set[str], other_name: str, other_genes: set[str]
) -> LevelResult:
    """L4: fraction of ``other_genes`` in Ohya's deletion set is >= MIN_GENE_OVERLAP."""
    overlap = len(other_genes & ohya_genes) / len(other_genes) if other_genes else 0.0
    return LevelResult(
        level=Level.L4,
        name=f"gene_containment_{other_name}",
        passed=overlap >= MIN_GENE_OVERLAP,
        message=(
            f"{overlap:.3f} of {other_name}'s {len(other_genes)} deletion genes "
            f"are in Ohya (>= {MIN_GENE_OVERLAP})"
        ),
        details={
            "n_other": len(other_genes),
            "n_in_ohya": len(other_genes & ohya_genes),
            "overlap": overlap,
            "missing_examples": sorted(other_genes - ohya_genes)[:20],
        },
    )


def run_morphology(data_root: str) -> bool:
    """Verify the Ohya morphology dataset (L0-L4) and write its report. True if it passes."""
    ohya_abs = osp.join(data_root, OHYA_SOURCE_ROOT)
    ohya_records = load_records(ohya_abs)
    report = verify_morphology_dataset(
        ohya_records,
        dataset_name="scmd_ohya2005",
        provenance=OHYA_PROVENANCE,
        expected_count=OHYA_EXPECTED_COUNT,
    )

    ohya_genes = perturbed_gene_set(ohya_records)
    for name, root in DELETION_GENE_SOURCES.items():
        other_abs = osp.join(data_root, root)
        if not osp.exists(osp.join(other_abs, "processed", "lmdb")):
            continue
        report.add(
            _l4_gene_containment(
                ohya_genes, name, perturbed_gene_set(load_records(other_abs))
            )
        )

    out = _write_report(report, osp.join(data_root, OHYA_REPORT_ROOT, "preprocess"))
    print(report.summary())
    print(f"  -> verified LMDB: {osp.join(ohya_abs, 'processed', 'lmdb')}")
    print(f"  -> wrote report:  {out}\n")
    return report.passed


def run_morphology_ohnuki(data_root: str) -> bool:
    """Verify the Ohnuki 2018 morphology dataset (L0-L4) and write its report.

    Same CalMorph L0-L3 gate as Ohya; L4 is SGD R64 containment (the essential-gene
    heterozygote ORFs are a subset of the S288C reference gene universe).
    """
    ohnuki_abs = osp.join(data_root, OHNUKI_SOURCE_ROOT)
    ohnuki_records = load_records(ohnuki_abs)
    report = verify_morphology_dataset(
        ohnuki_records,
        dataset_name="scmd_ohnuki2018",
        provenance=OHNUKI_PROVENANCE,
        expected_count=OHNUKI_EXPECTED_COUNT,
    )
    sgd_genes = _sgd_gene_set(data_root)
    report.add(
        _l4_rnaseq_gene_containment(
            sgd_genes, perturbed_gene_set(ohnuki_records)
        ).model_copy(update={"name": "gene_containment_sgd"})
    )
    out = _write_report(report, osp.join(data_root, OHNUKI_REPORT_ROOT, "preprocess"))
    print(report.summary())
    print(f"  -> verified LMDB: {osp.join(ohnuki_abs, 'processed', 'lmdb')}")
    print(f"  -> wrote report:  {out}\n")
    return report.passed


def run_ohnuki_morphology(data_root: str) -> bool:
    """Verify the Ohnuki 2022 morphology dataset (L0-L4); write its report. True if pass.

    L4 = the screened Ohnuki genes (a diagnostic subset selected from the same 4718
    non-essential morphology genes) are CONTAINED in Ohya 2005's deletion set.
    """
    ohnuki_abs = osp.join(data_root, OHNUKI2022_SOURCE_ROOT)
    ohnuki_records = load_records(ohnuki_abs)
    report = verify_morphology_dataset(
        ohnuki_records,
        dataset_name="scmd_ohnuki2022",
        provenance=OHNUKI2022_PROVENANCE,
        expected_count=OHNUKI2022_EXPECTED_COUNT,
    )
    ohya_abs = osp.join(data_root, OHYA_SOURCE_ROOT)
    if osp.exists(osp.join(ohya_abs, "processed", "lmdb")):
        ohya_genes = perturbed_gene_set(load_records(ohya_abs))
        report.add(
            _l4_gene_containment(
                ohya_genes, "scmd_ohnuki2022", perturbed_gene_set(ohnuki_records)
            )
        )
    out = _write_report(
        report, osp.join(data_root, OHNUKI2022_REPORT_ROOT, "preprocess")
    )
    print(report.summary())
    print(f"  -> verified LMDB: {osp.join(ohnuki_abs, 'processed', 'lmdb')}")
    print(f"  -> wrote report:  {out}\n")
    return report.passed


# --------------------------------------------------------------------------- #
# Visual-score datasets (WS7)
# --------------------------------------------------------------------------- #
VISUAL_SCORE_DATASETS: dict[str, dict[str, Any]] = {
    "carotenoid_ozaydin2013": {
        "root": "data/torchcell/carotenoid_ozaydin2013",
        "provenance": Provenance(
            source_uri="https://ars.els-cdn.com/content/image/1-s2.0-S109671761200081X-mmc1.xlsx",
            citation_key="ozaydinCarotenoidbasedPhenotypicScreen2013a",
            method="Elsevier ESM xlsx; colony-color visual carotenoid screen (-5..+5)",
            page="SI Sheet 1 'Color scores of all deletions'",
        ),
    }
}


def run_visual_score(data_root: str) -> bool:
    """Verify visual-score datasets (L0-L4) and write reports. True if all pass."""
    ohya_abs = osp.join(data_root, OHYA_SOURCE_ROOT)
    ohya_genes: set[str] = set()
    if osp.exists(osp.join(ohya_abs, "processed", "lmdb")):
        ohya_genes = perturbed_gene_set(load_records(ohya_abs))

    all_passed = True
    for name, spec in VISUAL_SCORE_DATASETS.items():
        abs_root = osp.join(data_root, spec["root"])
        records = load_records(abs_root)
        report = verify_visual_score_dataset(
            records,
            dataset_name=name,
            provenance=spec["provenance"],
            expected_count=len(records),
        )
        if ohya_genes:
            report.add(
                _l4_gene_containment(
                    ohya_genes, "scmd_ohya2005", visual_score_gene_set(records)
                )
            )
        out = _write_report(report, osp.join(abs_root, "preprocess"))
        print(report.summary())
        print(f"  -> wrote {out}\n")
        all_passed = all_passed and report.passed
    return all_passed


# --------------------------------------------------------------------------- #
# Metabolite datasets (WS8)
# --------------------------------------------------------------------------- #
METABOLITE_DATASETS: dict[str, dict[str, Any]] = {
    "betaxanthin_cachera2023": {
        "root": "data/torchcell/betaxanthin_cachera2023",
        "provenance": Provenance(
            source_uri="https://raw.githubusercontent.com/pc2912/CRI-SPA_repo/main/GA1_2_4_6.csv",
            citation_key="cacheraCRISPAHighthroughputMethod2023",
            method="CRI-SPA corrected colony HSV yellowness score (24h) as betaxanthin proxy",
            page="CRI-SPA GitHub GA1_2_4_6.csv (replicates 1/2/4/6)",
        ),
    },
    "amino_acid_mulleder2016": {
        "root": "data/torchcell/amino_acid_mulleder2016",
        "expected_count": 4678,
        "reference_centered": False,  # absolute mM concentrations, not centered scores
        "provenance": Provenance(
            source_uri="https://data.mendeley.com/datasets/bnzdhd6ck8/1",
            citation_key="mullederFunctionalMetabolomicsDescribes2016",
            method="LC-SRM intracellular amino-acid concentration (mM), batch-normalised",
            page="Mendeley 10.17632/bnzdhd6ck8.1 Table_S3 intracellular_concentration_mM",
        ),
    },
    "amino_acid_cooper2010": {
        "root": "data/torchcell/amino_acid_cooper2010",
        # 4382 Table 4 rows - 22 resolver drops (non-gene / retired) - 47 duplicate
        # identifier rows (ledgered, not averaged) = 4313 kept records.
        "expected_count": 4313,
        # linear ratio to the plate mean, reference 1.0 per key; NOT centered on 0
        "reference_centered": False,
        "provenance": Provenance(
            source_uri="torchcell-raw/cooperHighthroughputProfilingAmino2010/data/SupplementalTable4.txt",
            citation_key="cooperHighthroughputProfilingAmino2010",
            sha256="3c56cd492b4cab51249358e90c7e9731982960eb619e45b8850093e616a471fb",
            method=(
                "manual_browser deposit of Genome Research Supplemental Table 4 (login "
                "wall); CE-LIF NBD-F peak area as a linear ratio to the plate mean, "
                "duplicate traces averaged by the authors; legends .doc sha256 "
                "ca4983cda4c34318dbc3a05d10edf8e05df11f1560ffc99442df83a6a4399184"
            ),
            page="Genome Res 20:1288 Supplemental Table 4 (DC1 gr.105825.110)",
        ),
    },
    "metabolite_zelezniak2018": {
        "root": "data/torchcell/metabolite_zelezniak2018",
        # One record per (strain, protocol) since issue #595: 95 protocol-1 + 18
        # protocol-2 + 16 protocol-3 strains (scratch build of the pinned release).
        "expected_count": 129,
        "reference_centered": False,  # per-protocol WT baseline, not centered
        # Must equal the loader's ZELEZNIAK_METABOLITE_PROTOCOLS measurement_types
        # (pinned by tests/torchcell/datasets/scerevisiae/test_zelezniak2018.py).
        "protocol_measurement_types": frozenset(
            {
                "lc_srm_signal_batch_corrected_uncalibrated_dataset1",
                "lc_srm_ion_pair_external_calibration_batch_corrected_unit_unstated_dataset2",
                "lc_srm_hilic_external_calibration_batch_corrected_unit_unstated_dataset3",
            }
        ),
        "provenance": Provenance(
            source_uri="https://zenodo.org/api/records/1320289/files/metabolites_dataset.data_prep.tsv/content",
            citation_key="zelezniakMachineLearningPredicts2018",
            method="LC-SRM targeted metabolomics, three protocols kept separate (Dataset 1 uncalibrated signal, Datasets 2/3 externally calibrated); per-(strain, protocol) mean over that protocol's replicates vs the same-protocol WT",
            page="Zenodo 10.5281/zenodo.1320288 metabolites_dataset.data_prep.tsv",
        ),
    },
    "metabolite_dasilveira2014": {
        "root": "data/torchcell/metabolite_dasilveira2014",
        "expected_count": 127,  # 130 Quant rows - 3 WT controls (measured reference)
        "reference_centered": False,  # relative abundance (a.u.) vs measured WT baseline
        "provenance": Provenance(
            source_uri="https://doi.org/10.1091/mbc.E14-03-0851",
            citation_key="daSystematicLipidomicAnalysis2014",
            method="MS lipidomics relative abundance (a.u.), Table S4 Quant sheet; measured WT baseline from 3 WT control rows",
            page="library mirror daSystematicLipidomicAnalysis2014 TableS4 Quant sheet",
        ),
    },
    "organic_acid_yoshida2012": {
        "root": "data/torchcell/organic_acid_yoshida2012",
        "expected_count": 17,
        "reference_centered": False,  # absolute HPLC titers (mM) vs measured WT baseline
        "provenance": Provenance(
            source_uri="https://doi.org/10.1016/j.jbiosc.2011.12.017",
            citation_key="yoshidaIdentificationCharacterizationGenes2012",
            method="HPLC organic-acid titers (mM), static YPD 25C 72h; per-strain mean+/-SD (n=3) vs measured WT (BY4742)",
            page="J Biosci Bioeng 113:556 Table 3 (born-digital pdftotext -layout)",
        ),
    },
    "isobutanol_screen_lopez2024": {
        "root": "data/torchcell/isobutanol_screen_lopez2024",
        # First genome-wide biosensor screen (Table S2): median GFP fluorescence per YKO
        # strain; FC = median fluor(deletion) / median fluor(WT same plate). Aggregated to
        # ONE record per current-R64 ORF (mean FC; n_replicates=row count; SE=SD/sqrt-n when
        # >=2 rows). 4805 rows - 57 unresolved = 4748 resolved -> 4554 unique ORFs.
        "expected_count": 4554,
        "reference_centered": False,  # FC ratio vs WT control (reference metabolite_level=1.0)
        "provenance": Provenance(
            source_uri=(
                "https://dataspace.princeton.edu/handle/88435/dsp019s161956t (Montano Lopez "
                "2024 Princeton dissertation; data = supplementary_tables.xlsx sha256 f97cf13c "
                "in the library mirror). DOI-less dissertation -> Publication cites the same-lab "
                "biosensor methods paper (Montano-Lopez et al. Nat Commun 2022, PMID 35022416)"
            ),
            citation_key="lopezSystemsMetabolicEngineering2024",
            method=(
                "GFP alpha-ketoisovalerate/isobutanol-pathway biosensor (Leu1 promoter) "
                "integrated in each BY4741 (ura3D0) YKO strain; median GFP by flow cytometry "
                "in SC liquid at exponential phase, measured once (n=1). Table S2 fold change = "
                "deletion / same-plate WT. MetabolitePhenotype biosensor_gfp_fluorescence_fold_"
                "change, metabolite_level={isobutanol: FC}; genes resolved to current R64"
            ),
            page=(
                "Systems Metabolic Engineering of Isobutanol Production (Princeton 2024) Ch 3.3, "
                "Supplementary Table S2; xlsx sha256 f97cf13c..., thesis.pdf sha256 525e03b4..."
            ),
        ),
    },
    "isobutanol_validated_lopez2024": {
        "root": "data/torchcell/isobutanol_validated_lopez2024",
        # Validated re-screen (Table S3): FC>=2 (66) + FC<=0.5 (161) strains re-measured in
        # TRIPLICATE (n=3); FC average + STD (SE=STD/sqrt3). 227 block rows - 2 YBL071W-A
        # (contradictory 3.469 up / 0.0757 down, dropped) - 1 unresolved YIR043C = 224.
        "expected_count": 224,
        "reference_centered": False,
        "provenance": Provenance(
            source_uri=(
                "https://dataspace.princeton.edu/handle/88435/dsp019s161956t (Montano Lopez "
                "2024 Princeton dissertation; data = supplementary_tables.xlsx sha256 f97cf13c). "
                "DOI-less dissertation -> Publication cites Montano-Lopez et al. Nat Commun 2022"
            ),
            citation_key="lopezSystemsMetabolicEngineering2024",
            method=(
                "Table S3 = first-screen hits (FC>=2 or FC<=0.5) re-screened in triplicate "
                "(n=3); FC average + sample STD -> SE=STD/sqrt(3). Same biosensor/FC definition "
                "as the first screen. MetabolitePhenotype biosensor_gfp_fluorescence_fold_change"
            ),
            page=(
                "Systems Metabolic Engineering of Isobutanol Production (Princeton 2024) Ch 3.3, "
                "Supplementary Table S3; xlsx sha256 f97cf13c..., thesis.pdf sha256 525e03b4..."
            ),
        ),
    },
    "ffa_xue2025": {
        "root": "data/torchcell/ffa_xue2025",
        # In-house Xue 2025 combinatorial TF-deletion FFA titers. 177 genotype rows -> 176
        # experiment records + 1 measured WT reference (wt BY4741). Each strain = the
        # POX1-FAA1-FAA4 FFA-overproduction baseline + 0-3 TF deletions (letters decoded via
        # the Abbreviations sheet; N delta = 3 + #TF letters). Combinatorial -> L1 keys on the
        # genotype signature (deletion set), not per-ORF.
        "expected_count": 176,
        "reference_centered": False,  # absolute FFA titers (mg/L) vs measured WT baseline
        "provenance": Provenance(
            source_uri=(
                "$DATA_ROOT/torchcell-library/xue2025/data/Supplementary Data 1_Raw titers.xlsx "
                "(in-house/unpublished; sha256 023de80e...). No DOI -> Publication anchors to "
                "the faa1D faa4D pox1D FFA-chassis paper Runguphan & Keasling 2014 (PMID 23899824)"
            ),
            citation_key="xue2025",
            method=(
                "Combinatorial TF-deletion strains on the POX1-FAA1-FAA4 FFA-overproduction "
                "chassis; 5 free-fatty-acid species (C14:0/C16:0/C18:0/C16:1/C18:1) titered in "
                "mg/L, mean +- sample SD over up to 3 replicates (per-FFA replicate count varies; "
                "n=1 -> SE NaN). MetabolitePhenotype measurement_type=titer_mg_per_l; genotype = "
                "KanMxDeletionPerturbation per gene (markers unknown/mixed for in-house combos -> "
                "KanMX representative); env SC/30C aerobic (in-house-assumed); ref = measured WT"
            ),
            page=(
                "In-house Xue 2025 Supplementary Data 1_Raw titers.xlsx "
                "sha256=023de80ec51a4d0d1ccd6fc3506e42a44fcea6d3c3101ee7a83a005e49bbc779"
            ),
        ),
    },
    # The first bacterial metabolome. L4 is the BW25113 locus universe, not the Ohya
    # yeast deletion collection (selected from the records' own assembly pin), and the
    # reference is the deposit's measured `wt` profile on the z-score scale, so it is
    # not centered on 0.
    "metabolome_fuhrer2017": {
        "root": "data/torchcell/metabolome_fuhrer2017",
        # 3,807 deposit columns - 25 pooled Keio names - 46 JW ids off a BW25113 locus
        # - the `wt` reference = 3,735 kept records.
        "expected_count": 3735,
        "reference_centered": False,
        "provenance": Provenance(
            source_uri="https://www.ebi.ac.uk/biostudies/studies/S-BSST5",
            citation_key="fuhrerGenomewideLandscapeGene2017",
            method=(
                "FIA-TOF-MS modified ion z-scores per Keio deletion strain (7,534 ions, "
                "negative and positive mode), two biological replicates per strain; "
                "reference = the deposit's measured `wt` profile on the same scale. "
                "Deposit column names are gene NAMES, joined to Table EV1A and stored "
                "under the strain's BW25113 locus tag through its Keio JW synonym"
            ),
            page="Mol Syst Biol 13:907; S-BSST5 zscore_neg.tsv, zscore_pos.tsv",
        ),
    },
    # The CRISPRi metabolome. L4 is the MG1655 locus universe (the records' own
    # assembly pin), and the reference is the measured profile of the 15 empty-sgRNA
    # control strains on the per-batch-median scale, so it is not centered on 0.
    "metabolome_rapp2026": {
        "root": "data/torchcell/metabolome_rapp2026",
        # 1,513 Table S4 strain tokens - 15 control strains (the reference) - 1 strain
        # with no assigned target (argR) - 1 b-number the annotation remaps (phnE
        # b4104) = 1,496 kept records.
        "expected_count": 1496,
        "reference_centered": False,
        "provenance": Provenance(
            source_uri=("torchcell-raw/rappMetabolomeColiCRISPRi2026/data/si5.xlsx"),
            citation_key="rappMetabolomeColiCRISPRi2026",
            sha256=("fdc5ac2c759ad82d5cfb5ee1bb3c59fea427a88279e1178ad8e894e072498b84"),
            method=(
                "FI-MS fold changes of 1,321 annotated iML1515 m/z features relative "
                "to the per-batch median, the mean of each strain's two independent "
                "plates (exactly the paper's own Table S5 Mean_FC); reference = the 15 "
                "empty-sgRNA control strains on the same scale. Strains are stored "
                "under the iML1515 b-number Table S3 releases"
            ),
            page="Cell Syst 2026 Table S4 (mmc5.xlsx, sheet Table_S4)",
        ),
    },
}


def run_metabolite(data_root: str) -> bool:
    """Verify metabolite datasets (L0-L4) and write reports. True if all pass.

    L4 is selected by the HOST a dataset's own records name. A yeast dataset's deleted
    genes are checked against the Ohya deletion collection, as before. An
    assembly-pinned dataset (Fuhrer 2017's Keio metabolome) is checked against its own
    strain's locus universe instead: its b-numbers and ``BW25113_`` tags are not yeast
    ORFs, so the Ohya rule would fail it by construction and would say nothing about its
    identifiers.
    """
    ohya_abs = osp.join(data_root, OHYA_SOURCE_ROOT)
    ohya_genes: set[str] = set()
    if osp.exists(osp.join(ohya_abs, "processed", "lmdb")):
        ohya_genes = perturbed_gene_set(load_records(ohya_abs))

    all_passed = True
    for name, spec in METABOLITE_DATASETS.items():
        abs_root = osp.join(data_root, spec["root"])
        records = load_records(abs_root)
        report = verify_metabolite_dataset(
            records,
            dataset_name=name,
            provenance=spec["provenance"],
            expected_count=spec.get("expected_count", len(records)),
            reference_centered=spec.get("reference_centered", True),
            protocol_measurement_types=spec.get("protocol_measurement_types"),
        )
        assembly_sets = _dataset_assembly_sets(records)
        if assembly_sets != (SGD_S288C_R64,):
            universe, _ = _dataset_gene_universe(records, data_root)
            report.add(
                _l4_assembly_gene_containment(
                    universe,
                    assembly_sets,
                    metabolite_gene_set(records),
                    min_containment=1.0,
                )
            )
        elif ohya_genes:
            report.add(
                _l4_gene_containment(
                    ohya_genes, "scmd_ohya2005", metabolite_gene_set(records)
                )
            )
        out = _write_report(report, osp.join(abs_root, "preprocess"))
        print(report.summary())
        print(f"  -> wrote {out}\n")
        all_passed = all_passed and report.passed
    return all_passed


PROTEIN_DATASETS: dict[str, dict[str, Any]] = {
    "proteome_zelezniak2018": {
        "root": "data/torchcell/proteome_zelezniak2018",
        "expected_count": 97,
        "provenance": Provenance(
            source_uri="https://zenodo.org/records/1320289/files/proteins_dataset.data_prep.tsv",
            citation_key="zelezniakMachineLearningPredicts2018",
            method="SWATH-MS label-free protein signal, SVA batch-corrected; per-strain mean over replicates",
            page="Zenodo 10.5281/zenodo.1320288 proteins_dataset.data_prep.tsv",
        ),
    },
    "proteome_messner2023": {
        "root": "data/torchcell/proteome_messner2023",
        "expected_count": 4699,
        "allow_duplicate_orfs": True,
        "provenance": Provenance(
            source_uri="https://data.mendeley.com/datasets/w8jtmnszd9/1",
            citation_key="messnerProteomicLandscapeGenomewide2023",
            method=(
                "microflow-SWATH-MS (DIA-NN MaxLFQ), plate-median batch-corrected, "
                "no imputation; single-replicate KOs vs a 388-replicate HIS3 WT "
                "reference; UniProt->ORF via SGD GFF"
            ),
            page="Cell 186:2018; Mendeley 10.17632/w8jtmnszd9.1 yeast5k_noimpute_wide.csv",
        ),
    },
}


def run_protein(data_root: str) -> bool:
    """Verify protein-abundance datasets (L0-L4) and write reports. True if all pass."""
    ohya_abs = osp.join(data_root, OHYA_SOURCE_ROOT)
    ohya_genes: set[str] = set()
    if osp.exists(osp.join(ohya_abs, "processed", "lmdb")):
        ohya_genes = perturbed_gene_set(load_records(ohya_abs))

    all_passed = True
    for name, spec in PROTEIN_DATASETS.items():
        abs_root = osp.join(data_root, spec["root"])
        records = load_records(abs_root)
        report = verify_protein_dataset(
            records,
            dataset_name=name,
            provenance=spec["provenance"],
            expected_count=spec.get("expected_count", len(records)),
            allow_duplicate_orfs=spec.get("allow_duplicate_orfs", False),
        )
        if ohya_genes:
            report.add(
                _l4_gene_containment(
                    ohya_genes, "scmd_ohya2005", protein_gene_set(records)
                )
            )
        out = _write_report(report, osp.join(abs_root, "preprocess"))
        print(report.summary())
        print(f"  -> wrote {out}\n")
        all_passed = all_passed and report.passed
    return all_passed


# --------------------------------------------------------------------------- #
# RNA-seq pan-transcriptome datasets (WS10)
# --------------------------------------------------------------------------- #
RNASEQ_DATASETS: dict[str, dict[str, Any]] = {
    "caudal_pantranscriptome2024": {
        "root": "data/torchcell/caudal_pantranscriptome2024",
        "expected_count": 943,
        "provenance": Provenance(
            source_uri="http://1002genomes.u-strasbg.fr/files/",
            citation_key="caudalPantranscriptomeRevealsLarge2024",
            method=(
                "Caudal 2024 pan-transcriptome (final_data_annotated_merged); per-isolate "
                "absolute TPM + raw counts, genotype vs S288C from Peter 2018 genomes"
            ),
            page="Nat. Genet. 56:1278; final_data_annotated_merged_04052022.tab",
        ),
    },
    "nadal_ribelles_perturbseq2025": {
        "root": "data/torchcell/nadal_ribelles_perturbseq2025",
        "expected_count": 6188,
        "provenance": Provenance(
            source_uri="https://doi.org/10.5281/zenodo.14062629",
            citation_key="nadal-ribellesSinglecellResolvedGenotypephenotype2025",
            sha256=("c210fe541b0b91bc6eead28aa2265065afceec763ade1abd682c58896299a240"),
            method=(
                "Nadal-Ribelles 2025 genome-scale single-cell Perturb-seq collapsed to "
                "pseudobulk: per (deletion genotype, condition) log2 fold-change vs "
                "same-condition WT (scanpy logfoldchanges; Wilcoxon DE) + per-genotype "
                "dispersion (sd_lvscore_scaledFU2) + n_cells (cell_number); FC_genotype.Rdata "
                "+ ptb_summary.Rdata read with the pure-Python rdata reader"
            ),
            page="Nat. Commun. 16 (2025), doi:10.1038/s41467-025-57600-4; FC_genotype.Rdata",
        ),
    },
    # The two bacterial compendia. Both release one row per sequenced LIBRARY, so L1 is
    # `replicate_groups` (`replicate_aware`), and L4 is the pinned strain's locus-tag
    # universe. `min_containment` is 1.0 where every measured id must be a locus of the
    # pin, and PRECISE-1K's floor is the 0.99 its own loader states, since three
    # released b-numbers (b3036, b4223, b4590) are not in ASM584v2 and are kept as given.
    "rnaseq_lamoureux2023": {
        "root": "data/torchcell/rnaseq_lamoureux2023",
        # 1,035 released samples - 794 dropped by the genotype and environment rules.
        "expected_count": 241,
        "replicate_aware": True,
        "min_containment": 0.99,
        "provenance": Provenance(
            source_uri="https://doi.org/10.5281/zenodo.8284223",
            citation_key="lamoureuxMultiscaleExpressionRegulation2023",
            sha256="7c7008f2c8bcd66aebecbdb97b8c1a0314637e3873ec09e12c26d0ccaaa35172",
            method=(
                "PRECISE-1K: one record per biological-replicate RNA-seq library of "
                "E. coli K-12 MG1655 (wild type or whole-gene deletions), "
                "log_tpm_qc.csv back-transformed to TPM as 2**x - 1 (the pseudocount "
                "back-solved from the release) with featureCounts counts.csv beside it"
            ),
            page=(
                "Nucleic Acids Res. 51:10184; Zenodo SBRG/precise1k-v1.0.zip, "
                "data/precise1k/{log_tpm_qc,counts,metadata_qc}.csv"
            ),
        ),
    },
    "putida_precise321_lim2022": {
        "root": "data/torchcell/putida_precise321_lim2022",
        # 321 compendium samples - 141 dropped (engineered, evolved, plasmid-bearing,
        # unresolved deletion symbols) = 180 on the KT2440 reference strain.
        "expected_count": 180,
        "replicate_aware": True,
        "min_containment": 1.0,
        "provenance": Provenance(
            source_uri="https://doi.org/10.1016/j.ymben.2022.04.004",
            citation_key="limMachinelearningPseudomonasPutida2022",
            sha256="10e81a18fdfd08b9e27581f877dd0ae2a169946508b1ed5dabbe444210420970",
            method=(
                "putidaPRECISE321: one record per compendium sample whose genotype can "
                "be written against KT2440, Supplementary Data X matrix "
                "(log2(TPM + 1)) back-transformed to TPM, counts paired from "
                "SBRG/modulome_ppu@f63a0df counts.csv by reproducing the released value"
            ),
            page="Metab. Eng. 72:297; si/si2.xlsx sheets 1-Sample_list and X",
        ),
    },
}

# S288C reference gene universe (ORF + RNA-coding systematic names) for L4 containment.
SGD_GENE_FASTAS = [
    "orf_coding_all_R64-4-1_20230830.fasta",
    "rna_coding_R64-4-1_20230830.fasta",
]
# Empirically the Caudal measured-gene union is 0.943 contained in the SGD gene set (the
# remainder are accessory/novel ORFs legitimately absent from S288C); 0.90 leaves headroom.
MIN_RNASEQ_GENE_CONTAINMENT = 0.90


def _sgd_gene_set(data_root: str) -> set[str]:
    """Build the S288C systematic-name universe from the SGD ORF + RNA FASTA headers."""
    genes: set[str] = set()
    for name in SGD_GENE_FASTAS:
        with open(resolve(SGD_S288C_R64, name, data_root=data_root)) as handle:
            for line in handle:
                if line.startswith(">"):
                    genes.add(line[1:].split()[0])
    return genes


def _genome(data_root: str) -> Any:
    """The S288C genome, for the resolver the canonical-gene-name rule needs.

    ``overwrite=False`` is mandatory: a rebuild here would race any other process holding
    the same gffutils database.
    """
    from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

    return SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )


# --------------------------------------------------------------------------- #
# Bacterial gene universes, and selecting a record's universe by its own reference
# --------------------------------------------------------------------------- #
#: The GenBank assembly each bacterial strain set's locus-tag universe is read from.
BACTERIAL_GENE_ASSEMBLIES: dict[str, BacterialAssembly] = {
    assembly.assembly_set: assembly
    for assembly in (
        MG1655_ASSEMBLY,
        BW25113_ASSEMBLY,
        KT2440_ASSEMBLY,
        REL606_ASSEMBLY,
    )
}
#: The species a reference without an ``assembly_set`` must name (a yeast record).
YEAST_SPECIES = "Saccharomyces cerevisiae"


def _bacterial_gene_set(assembly: BacterialAssembly, data_root: str) -> set[str]:
    """The locus-tag universe of one bacterial GenBank assembly, for L4 containment.

    The bacterial sibling of :func:`_sgd_gene_set`: the locus tag of every ``gene`` row of
    the set's ``_feature_table.txt.gz`` (protein-coding, RNA and pseudogene loci alike,
    since a pseudogene tag is a locus a record may carry), cross-checked against the
    ``_protein.faa.gz``: every protein accession must be the product of a ``CDS`` row
    whose locus tag is one of those genes. A disagreement between the two members is
    refused, never unioned. Both members are resolved (sha256-verified) from the tier.
    """
    table = resolve(
        assembly.assembly_set,
        f"{assembly.genbank_assembly}_feature_table.txt.gz",
        data_root=data_root,
    )
    genes: set[str] = set()
    cds_locus: dict[str, str] = {}
    with gzip.open(table, "rt") as handle:
        columns = handle.readline().removeprefix("# ").rstrip("\n").split("\t")
        feature, accession, locus_tag = (
            columns.index(name)
            for name in ("feature", "product_accession", "locus_tag")
        )
        for line in handle:
            row = line.rstrip("\n").split("\t")
            if row[feature] == "gene":
                genes.add(row[locus_tag])
            elif row[feature] == "CDS" and row[accession]:
                cds_locus[row[accession]] = row[locus_tag]
    proteins: set[str] = set()
    fasta = resolve(
        assembly.assembly_set, assembly.protein_fasta_member, data_root=data_root
    )
    with gzip.open(fasta, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                proteins.add(line[1:].split()[0])
    no_cds = sorted(proteins - set(cds_locus))
    off_gene = sorted({cds_locus[p] for p in proteins if p in cds_locus} - genes)
    if no_cds or off_gene:
        raise ValueError(
            f"{assembly.assembly_set}: {len(no_cds)} proteins have no CDS row in the "
            f"feature table {no_cds[:10]}; {len(off_gene)} CDS locus tags are no gene "
            f"row {off_gene[:10]}"
        )
    return genes


def _ecoli_k12_gene_set(data_root: str, strain: EcoliK12StrainName) -> set[str]:
    """The E. coli K-12 locus-tag universe of ``strain``: MG1655 b-numbers or BW25113
    ``BW25113_`` tags (never one standing in for the other).
    """
    return _bacterial_gene_set(
        BACTERIAL_GENE_ASSEMBLIES[BACTERIAL_ASSEMBLY_SETS[strain]], data_root
    )


def _ecoli_rel606_gene_set(data_root: str) -> set[str]:
    """The E. coli B REL606 locus-tag universe (``ECB_`` tags, tRNA and rRNA included);
    a B strain, so no K-12 universe stands in for it.
    """
    return _bacterial_gene_set(REL606_ASSEMBLY, data_root)


def _pputida_gene_set(data_root: str) -> set[str]:
    """The P. putida KT2440 locus-tag universe (``PP_`` tags, RNA tags included)."""
    return _bacterial_gene_set(KT2440_ASSEMBLY, data_root)


def _reference_assembly_set(genome_reference: Mapping[str, Any]) -> str:
    """The assembly set a record's own stored ``genome_reference`` is written against.

    An ``AssemblyReferenceGenome`` names its bacterial set. A reference without
    ``assembly_set`` is a yeast record: it must name Saccharomyces cerevisiae, whose set
    is the SGD R64 release. Anything else is refused, so a record's gene universe and
    resolver are never borrowed from another host.
    """
    if "assembly_set" in genome_reference:
        assembly_set = str(genome_reference["assembly_set"])
        if assembly_set not in BACTERIAL_GENE_ASSEMBLIES:
            raise ValueError(
                f"genome reference names assembly set {assembly_set!r}; known: "
                f"{sorted(BACTERIAL_GENE_ASSEMBLIES)}"
            )
        return assembly_set
    if genome_reference["species"] != YEAST_SPECIES:
        raise ValueError(
            f"a genome reference without assembly_set must be {YEAST_SPECIES!r}, got "
            f"{genome_reference['species']!r}"
        )
    return SGD_S288C_R64


def _gene_set_for_reference(
    genome_reference: Mapping[str, Any], data_root: str
) -> set[str]:
    """The L4 gene universe of the genome a record's own reference names: the S288C
    ORF + RNA set for a yeast reference, the strain's locus-tag set for an
    assembly-pinned bacterial one. Call once per distinct reference of a dataset.
    """
    assembly_set = _reference_assembly_set(genome_reference)
    if assembly_set == SGD_S288C_R64:
        return _sgd_gene_set(data_root)
    return _bacterial_gene_set(BACTERIAL_GENE_ASSEMBLIES[assembly_set], data_root)


def _genome_for_reference(genome_reference: Mapping[str, Any], data_root: str) -> Any:
    """The genome whose ``resolve_gene_name`` the canonical-name rule applies to a record.

    S288C (:func:`_genome`) for a yeast reference; for an assembly-pinned bacterial one,
    the strain's genome from its default cache root with ``overwrite=False``
    (``bacteria_common.bacterial_genome``), so bacterial names are never resolved
    against S288C. Call once per distinct reference: construction reads the tier.
    """
    assembly_set = _reference_assembly_set(genome_reference)
    if assembly_set == SGD_S288C_R64:
        return _genome(data_root)
    # Imported here, not at the top: this module must not import loaders, and the
    # bacteria_common import runs the torchcell.datasets package (every loader).
    from torchcell.datasets.bacteria_common import (
        bacterial_genome,
        host_of_strain,
        strain_of_assembly_set,
    )

    strain = strain_of_assembly_set(assembly_set)
    return bacterial_genome(host_of_strain(strain), strain, data_root)


def _dataset_assembly_sets(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """The assembly sets a dataset's own records are written against, sorted.

    One dataset may hold several: Tong 2020 pins its Keio strains to BW25113 and its
    sRNA-library strains to MG1655, so the L4 universe is the union of both. A yeast
    dataset yields the SGD release id, since that is what
    :func:`_reference_assembly_set` calls its universe.
    """
    return tuple(
        sorted(
            {
                _reference_assembly_set(record["reference"]["genome_reference"])
                for record in records
            }
        )
    )


def _dataset_gene_universe(
    records: Sequence[Mapping[str, Any]], data_root: str
) -> tuple[set[str], tuple[str, ...]]:
    """The union of the L4 gene universes of the genomes a dataset's records name.

    Selected from each record's OWN ``genome_reference``, never from the runner's host:
    the L4 rule fails a record keyed to a name its genome does not carry, which is the
    behavior we want per host and nonsense across hosts. Returns the universe and the
    assembly sets it came from, so the report can name them.
    """
    universe: set[str] = set()
    assembly_sets = _dataset_assembly_sets(records)
    for assembly_set in assembly_sets:
        if assembly_set == SGD_S288C_R64:
            universe |= _sgd_gene_set(data_root)
        else:
            universe |= _bacterial_gene_set(
                BACTERIAL_GENE_ASSEMBLIES[assembly_set], data_root
            )
    return universe, assembly_sets


def _l4_assembly_gene_containment(
    universe: set[str],
    assembly_sets: tuple[str, ...],
    measured: set[str],
    *,
    min_containment: float = MIN_RNASEQ_GENE_CONTAINMENT,
) -> LevelResult:
    """L4: the measured genes are loci of the assemblies the records themselves pin.

    The host-aware sibling of :func:`_l4_rnaseq_gene_containment` and of the Ohya
    deletion-set containment: a bacterial dataset's genes are not in the yeast deletion
    collection and are not S288C ORFs, so those two rules would fail it by construction
    while saying nothing about its identifiers.
    """
    overlap = len(measured & universe) / len(measured) if measured else 0.0
    pins = ", ".join(assembly_sets)
    return LevelResult(
        level=Level.L4,
        name="gene_containment_assembly",
        # An empty measured set FAILS, as it does in the two sibling rules: every family
        # this serves measures genes, so an empty set is a broken build, not a vacuous
        # pass. The message says that rather than reporting a fabricated 0.000 overlap.
        passed=bool(measured) and overlap >= min_containment,
        message=(
            f"{overlap:.3f} of {len(measured)} measured genes are loci of {pins} "
            f"(>= {min_containment})"
            if measured
            else f"no measured genes, so containment in {pins} has nothing to check"
        ),
        details={
            "assembly_sets": list(assembly_sets),
            "n_measured": len(measured),
            "n_in_universe": len(measured & universe),
            "n_universe": len(universe),
            "overlap": overlap,
            "missing_examples": sorted(measured - universe)[:20],
        },
    )


def _l4_rnaseq_gene_containment(sgd_genes: set[str], measured: set[str]) -> LevelResult:
    """L4: the measured expression gene universe is contained in the SGD gene set."""
    overlap = len(measured & sgd_genes) / len(measured) if measured else 0.0
    return LevelResult(
        level=Level.L4,
        name="gene_containment_sgd",
        passed=overlap >= MIN_RNASEQ_GENE_CONTAINMENT,
        message=(
            f"{overlap:.3f} of {len(measured)} measured genes are S288C reference genes "
            f"(>= {MIN_RNASEQ_GENE_CONTAINMENT})"
        ),
        details={
            "n_measured": len(measured),
            "n_in_sgd": len(measured & sgd_genes),
            "overlap": overlap,
            "missing_examples": sorted(measured - sgd_genes)[:20],
        },
    )


def run_rnaseq(data_root: str) -> bool:
    """Verify RNA-seq expression datasets (L0-L4) and write reports. True if pass.

    Two per-dataset selections, both read off the records rather than configured twice:
    ``replicate_aware`` picks the L1 rule (one record per library, for the bacterial
    compendia), and the L4 universe is the genome a dataset's own records name -- the
    S288C ORF + RNA set for a yeast dataset, the pinned strain's locus-tag set for an
    assembly-pinned one. The yeast gene set is read only if a yeast dataset is present.
    """
    sgd_genes: set[str] | None = None
    all_passed = True
    for name, spec in RNASEQ_DATASETS.items():
        abs_root = osp.join(data_root, spec["root"])
        records = load_records(abs_root)
        report = verify_rnaseq_dataset(
            records,
            dataset_name=name,
            provenance=spec["provenance"],
            expected_count=spec.get("expected_count", len(records)),
            replicate_aware=spec.get("replicate_aware", False),
        )
        assembly_sets = _dataset_assembly_sets(records)
        if assembly_sets != (SGD_S288C_R64,):
            universe, _ = _dataset_gene_universe(records, data_root)
            report.add(
                _l4_assembly_gene_containment(
                    universe,
                    assembly_sets,
                    rnaseq_gene_set(records),
                    min_containment=spec.get(
                        "min_containment", MIN_RNASEQ_GENE_CONTAINMENT
                    ),
                )
            )
        else:
            if sgd_genes is None:
                sgd_genes = _sgd_gene_set(data_root)
            report.add(_l4_rnaseq_gene_containment(sgd_genes, rnaseq_gene_set(records)))
        out = _write_report(report, osp.join(abs_root, "preprocess"))
        print(report.summary())
        print(f"  -> wrote {out}\n")
        all_passed = all_passed and report.passed
    return all_passed


# --------------------------------------------------------------------------- #
# Environment-response chemogenomic datasets (WS15)
# --------------------------------------------------------------------------- #
ENVIRONMENT_RESPONSE_DATASETS: dict[str, dict[str, Any]] = {
    "yeastphenome": {
        "root": "data/torchcell/yeastphenome",
        # Curated YeastPhenome GROWTH screens (NPV z-score label): 47 complete-loss-of-
        # function deletion screens (haploid BY4741/BY4742 + homozygous-diploid BY4743;
        # heterozygous/dosage screens excluded) minus already-built primaries, x their
        # encodable single-dosed-compound growth conditions = 83 environments, 296777
        # records (one per (ORF, condition, screen); non-ORF rows + unparseable /
        # het-or-ambiguous / unknown-media columns drop-and-logged). Widen via SCREENS.
        "expected_count": 296777,
        "background_genes": frozenset(),
        "provenance": Provenance(
            source_uri="https://doi.org/10.5281/zenodo.7714347",
            citation_key="turcoGlobalAnalysisYeast2023",
            method=(
                "curated YeastPhenome NPV (mode-referenced modified z-score) from "
                "yp-data @ v1.0 (commit 83e2917bf86955ec6ba66dc70ff2ed0fe24ecbe8), the "
                "freeze Zenodo 7714347 archives; label = NPV (valuez.txt); readout method "
                "recorded in units (distinguishes near-replicate screens); temperature + "
                "n_samples/uncertainty/sample_unit are not_carried_by_curation "
                "ProvenanceGaps; exact modified-z formula deferred to Turco2023 note S4"
            ),
            page="Sci Adv 2023 adg5702; yp-data Datasets/<PMID>/<stem>_valuez.txt",
        ),
    },
    "env_chemgen_vanacloig2022": {
        "root": "data/torchcell/env_chemgen_vanacloig2022",
        # 3606 retained library rows x 34 Fig 1B conditions (45 matrix tokens minus the
        # 11 the paper never reports) minus the 3942 all-three-replicates-zero cells.
        # Rules + counts: preprocess/dropped_records.json (issue #501).
        "expected_count": 118662,
        # The sensitizing and reporter alleles ride on the reference's
        # StrainReferenceGenome background (#500), so the genotype holds only the
        # screened deletion and no background gene needs excluding.
        "background_genes": frozenset(),
        "provenance": Provenance(
            source_uri=(
                "$DATA_ROOT/torchcell-raw/"
                "vanacloig-pedrosComparativeChemicalGenomic2022/data/"
                "GSE186866_ChemGenomics_Raw_Counts_matrix.txt.gz (raw mirror; retrieved "
                "from https://ftp.ncbi.nlm.nih.gov/geo/series/GSE186nnn/GSE186866/suppl/)"
            ),
            citation_key="vanacloig-pedrosComparativeChemicalGenomic2022",
            sha256="e29eb02769ce2180d632020dc612a7f3e14a124fc7f1e0e33f9d41b6f4e4a85a",
            method=(
                "GEO GSE186866 raw up-tag barcode counts; per condition, edgeR 3.26.8 "
                "TMM factors (calcNormFactors defaults, ported to numpy) over its 3 "
                "replicates and paired control columns, then per gene "
                "log2((TMM-CPM_rep+1)/(mean TMM-CPM of the SAME CG00n batch's "
                "inhibitor-free control columns+1)) (the paper's paired design), "
                "pooled over all 16 control columns for MMS only (analyzed unpaired); "
                "response = mean of 3 biological replicates, uncertainty = their "
                "sample SD (SE=SD/sqrt(3)); recomputed readout, NOT the paper's edgeR "
                "glmQLFit logFC. Anaerobic static SynBase (shared MEDIA_LIBRARY object) "
                "at 30 C for 48 h / 6.5 doublings in 1.5 mL 24-well cultures, pH 5.0 "
                "as an EnvironmentPhysicalPerturbation; the 34 Fig 1B conditions at "
                "their IC30 basis (Table S1 molar values unavailable) except Benomyl "
                "34.4 uM (Piotrowski 2017), MMS (fixed, unit unstated) and DMSO 1% "
                "v/v; MBO adjudicated to 2-methyl-3-buten-2-ol. Strain background: "
                "the SGA MATa progeny of Y13206 x the MATa kanMX array. Dropped: 11 "
                "matrix tokens Fig 1B does not list, 2 all-NaN rows, 4 rows at a "
                "selected background locus, 22 retired ORFs, 17 legacy ORF spellings "
                "(typed ConstructedOrf ledger entries), and 3942 all-replicates-zero "
                "cells"
            ),
            page=(
                "FEMS Yeast Res 2022 foac036; paper.md "
                "sha256=0b5d938b54b8424fa08203a4357bc8f7c7dfae3fbe1a6d07d422848b92f37ba3"
            ),
        ),
    },
    "env_chemgen_mota2024": {
        "root": "data/torchcell/env_chemgen_mota2024",
        "expected_count": 1270,
        "background_genes": frozenset(),
        "provenance": Provenance(
            source_uri=(
                "$DATA_ROOT/torchcell-raw/motaSharedMoreSpecific2024/si/"
                "12934_2024_2309_MOESM{1,2,3}_ESM.xlsx (raw mirror; Springer ESM is "
                "scriptable and re-yielded all three bit-identically on 2026-09-12)"
            ),
            citation_key="motaSharedMoreSpecific2024",
            method=(
                "BMC open-access supplementary spreadsheets (Additional files 1-3 = Tables "
                "S1/S2/S3). ORDINAL spot-assay susceptibility grade, measurement_type="
                "ordinal: environment_response carries the rank (0/1/2), category the "
                "shared call (no_change / reduced / severely_reduced) and category_label "
                "the source symbol ('0' / '+' / '++'). The reference is the PARENTAL "
                "BY4741 in empty wells of the SAME plates, scored '0' ('an absence of a "
                "detectable susceptibility phenotype'), so its rank is 0.0. 75 mM acetic / "
                "14 mM butyric / 0.30 mM octanoic acid -> SmallMoleculePerturbation, plus a "
                "typed EnvironmentPhysicalPerturbation(factor=ph, magnitude=4.5 pH, "
                "agent=hydrochloric acid) on BOTH environments -- pH is never a medium "
                "name. Media = the shared media.YPD_AGAR (base_medium='YPD'), 30 C, "
                "duration_hours=48.0 (RULE: the time the stored call was made, anchored by "
                "'(++) if no growth was observed after 48 h of incubation'; the 36-48 h "
                "photograph window and the 24 h control reading are recorded as notes). "
                "assay_type=spot_dilution; n_samples and sample_unit carry "
                "ProvenanceGap(not_reported_by_primary) -- the screen states no replicate "
                "count. Every token, including systematic-looking ones, goes through the "
                "SHARED resolve_gene_name: 4 RENAMED names that the old resolver stored "
                "unchecked are corrected (YGR272C->YGR271C-A, YJL021C->YJL020C, "
                "YML010W-A->YML009W-B, YML013C-A->YML012C-A) and 6 RETIRED tokens (13 "
                "records) are dropped. Dedup keeps the MORE SEVERE grade per (ORF, acid), "
                "ties broken lexicographically: the RNR4 source duplicate (3 records) and "
                "the EFG1/YGR272C pair (3 records), which the SI itself says are one gene "
                "('merged with an adjacent ORF into a single reading frame, designated "
                "YGR271C-A'). 1289 raw - 3 - 3 - 13 = 1270"
            ),
            page=(
                "Microb Cell Fact 2024 s12934-024-02309-0 (PMC10903034); "
                "MOESM1 sha256=b23ad28141e70b307048fc69475aedd4e3cf880118ae9d0d806b6d9f91205e42, "
                "MOESM2 sha256=a7a1aaee1c76e52d8fe435326790c89170ab43ec96b92ef272903d8e78a1e81f, "
                "MOESM3 sha256=27f1508641ad5e7cc29ab8611739d4940da355c1dffbe9c9a908c267cdf5d455; "
                "paper.md sha256=a19769f757fd912139551f39736dd2b67581cb03f83a7a9b9385e28516b1f1b6"
            ),
        ),
    },
    "env_chemgen_hoepfner2014": {
        "root": "data/torchcell/env_chemgen_hoepfner2014",
        # ENCODABLE-COMPOUNDS-ONLY build (only compounds with a released SMILES in Table S1;
        # the proprietary CMBxxx columns are counted in the ledger's encodable_filter),
        # MINUS the one kept compound that resolves to no structure identifier (CMB409
        # Boromycin), MINUS the six CMB4019 "D-Glucose (starvation)" columns excluded by
        # rule (#506: HIP 17,289 + HOP 13,479 records), MINUS the rows scored in fewer
        # than half of their arm's deposited sensitivity columns (#506 detection rule:
        # HIP 72 rows / 2,263 records, HOP 452 rows / 7,344 records, 328 of the HOP rows
        # SGD-essential). Row ORFs through the shared resolver (CURRENT kept; RENAMED kept
        # under the current name with a ConstructedOrf; NON_GENE_FEATURE / RETIRED
        # dropped, 30 rows per assay). HIP 1,739,664 (302 encodable heterozygous kanMX4
        # deletion experiments) + HOP 1,344,163 (300 barcoded kanMX4 deletion
        # experiments) = 3,083,827 records, measured by
        # experiments/036-dataset-fixes-before-kg-build/scripts/hoepfner2014_inputs.py
        # over the real record path. Drop ledger: <root>/dropped_records.json.
        "expected_count": 3083827,
        # ~3.1M records -> single-pass streaming gate (retained from the 30M full-atlas build).
        "stream": True,
        # HIP heterozygous + HOP homozygous kanMX4 deletion diploid collections (BY4743):
        # no constant background genes in the genotype (the BY4743 alleles ride on the
        # StrainReferenceGenome background, not on Genotype).
        "background_genes": frozenset(),
        "provenance": Provenance(  # noqa: F821  # resolved in runners.py's namespace
            source_uri=(
                "https://datadryad.org/downloads/file_stream/4834608 (HIP_scores.txt); "
                "https://datadryad.org/downloads/file_stream/4834609 (HOP_scores.txt); "
                "Dryad doi:10.5061/dryad.v5m8v"
            ),
            citation_key="hoepfnerHighresolutionChemicalDissection2014",
            method=(
                "Novartis HIP-HOP chemogenomic atlas; deposited (adjusted) MADL "
                "sensitivity score = (r_L - med(r_L))/MAD(r_L) per (deletion strain x "
                "compound/concentration) at IC30 in YPD_LIQUID, 30 C, 24-well culture "
                "(CultureEnvironment); 2% DMSO vehicle up to 200 uM, a solvent "
                "ProvenanceGap above it, a pH ProvenanceGap for HCl / NaOH / sodium "
                "acetate; encodable compounds only (released Table S1 SMILES), and every "
                "kept compound carries a structure identifier -- curated "
                "compound_identity_table row first, else an InChIKey derived from the "
                "released SMILES with RDKit; the one compound resolving to no identifier "
                "(CMB409 Boromycin) is DROPPED; CMB4019 glucose starvation is EXCLUDED "
                "(medium change not expressible with a sourced value); rows scored in "
                "fewer than half of their arm's columns are DROPPED (detection rule). "
                "StrainEnvironmentResponseExperiment records on a BY4743 "
                "StrainReferenceGenome whose name and alleles are pending-review gaps. "
                "HIP = HeterozygousDeletionPerturbation (kanMX4, YSC1055, Table S5 "
                "construction lab / batch / plate / well) incl. essential genes, HIP "
                "duration hours, generations and endpoint all gaps (Fig. S2); HOP = "
                "BarcodedKanMxDeletionPerturbation (kanMX4, YSC1056) at 16 h / ~5 "
                "generations; RENAMED source ORFs carry a ConstructedOrf; n_samples = 2 "
                "(Ad. columns) / 1 (MADL columns), technical duplicate, reference "
                "n_samples = 4 (conservative lower end of the paper's 4-8 control "
                "replicates); screen_id = the deposited study number; assay_type = "
                "pooled_competitive_growth_barcode; Table S5 background-mutation HIP "
                "strains are KEPT and flagged in <root>/table_s5_affected_strains.json; "
                "row ORFs go through the SHARED resolve_gene_name (CURRENT kept; RENAMED "
                "kept under the current name; NON_GENE_FEATURE and RETIRED dropped to "
                "<root>/dropped_records.json)"
            ),
            page=(
                "Microbiol Res 2014 (doi:10.1016/j.micres.2013.11.004); Dryad "
                "doi:10.5061/dryad.v5m8v HIP_scores.txt "
                "sha256=dbc5041defea9c046da0890d5e569f97d5f7afbf50ea0885f539ea8e5980cd24, "
                "HOP_scores.txt "
                "sha256=99b386a84384eae847657ed41bf222c9550a87ef961f0ab191833c918771ffd7, "
                "Table_S1.xls "
                "sha256=115bb31cc5e696588d1ecb4ffa262475e05025e22347f7e004f77fd635898209"
            ),
        ),
    },
    "env_chemgen_wildenhain2015": {
        "root": "data/torchcell/env_chemgen_wildenhain2015",
        # (strain, compound-identity) cells minus the strain_label_unresolved cells
        # (TSCII, YGL11, wtn01) and the SID-only compounds' cells (#504). Rules +
        # counts: preprocess/dropped_records.json.
        "expected_count": 430820,
        "background_genes": frozenset(),
        "stream": True,
        "provenance": Provenance(
            source_uri=(
                "$DATA_ROOT/torchcell-raw/"
                "wildenhainPredictionSynergismChemicalGenetic2015/data/1159580.csv.gz "
                "+ data/aid_1159580_description.json (raw mirror)"
            ),
            citation_key="wildenhainPredictionSynergismChemicalGenetic2015",
            sha256="c461c679b63ac56045cef0f03ed9bcbb8e7f9c12146f1fc7cc8ac0c113188d64",
            method=(
                "PubChem BioAssay AID 1159580 datapoint export (member of the FTP range "
                "archive, container sha256 d1fd5dc2bf7c526ad9845e0a14ae9981256fb820aaf42"
                "28b48a3ba0724ee59b0 asserted before the member is read), the extended "
                "CGM of the 2016 Sci Data descriptor (doi:10.1038/sdata.2016.95; 242 "
                "strains, 492,126 tests); released z_score (N(1, IQR) fit within the "
                "strain's own screen) per (strain x compound) at 20 uM in 1.96% v/v DMSO, "
                "SC + 2% glucose, 30 C, static 100 uL 96-well culture read at solvent-"
                "control saturation (StrainEnvironmentResponseExperiment, "
                "CultureEnvironment). Genotypes on a BY4741 StrainReferenceGenome: "
                "kanMX4 Euroscarf deletions; the 33 SGD-essential ORFs as "
                "ConditionalAllelePerturbation with typed gaps; NA/NNK1 mapped to "
                "YKL171W; the wild-type screen as the empty genotype. One reference: "
                "z = 0, no compound. Response = mean over a cell's distinct released "
                "screens, n_samples = screens, sample SD or a typed gap at n=1. "
                "PUBCHEM_ACTIVITY_OUTCOME + bioactivity map onto ResponseCategory. "
                "Held/dropped: the TSCII / YGL11 / wtn01 labels and the SID-only "
                "compounds"
            ),
            page=(
                "Cell Systems 2015 (doi:10.1016/j.cels.2015.12.003); AID description "
                "sha256=23c5f8c56af94786cfe8e22c93fdde0b719ca2165975305944557ab39087b0e4; "
                "paper.md sha256=f46409eb8f23412c9c1015d0f8f5bb581bfddfe2796d319d407585e23c757ac2; "
                "Sci Data 2016 paper.md sha256=89ff4d9bf1d31719ab15c18ab7aca0b7caf10f55c7c239c1b021908c95439e33"
            ),
        ),
    },
    "env_chemgen_auesukaree2009": {
        "root": "data/torchcell/env_chemgen_auesukaree2009",
        "expected_count": 525,
        "background_genes": frozenset(),
        "provenance": Provenance(
            source_uri=(
                "$DATA_ROOT/torchcell-raw/auesukareeGenomewideIdentificationGenes2009/"
                "paper/paper.pdf (raw mirror; bytes from Zotero attachment VIJCFVIA, "
                "PMC2747848 file downloads being JS-proof-of-work and not scriptable)"
            ),
            citation_key="auesukareeGenomewideIdentificationGenes2009",
            sha256="01b945443c0ce41642c76fd737e12b4c31cacb5f384049a5c0a7e4bf9e1eb5a1",
            method=(
                "Article Tables 1-6 extracted from the born-digital PDF text layer "
                "(pdftotext -layout; per-class parenthetical counts as self-checksums; the "
                "poppler version is recorded in the raw mirror's manifest). Categorical "
                "spot-assay sensitivity call, and the comparator is WITHIN a strain, not vs "
                "the parent: 'Deletion mutants showing significantly reduced growth on "
                "plates under stress conditions, as compared to those under non-stress "
                "conditions, were defined as sensitive mutants.' So the reference "
                "environment is the UNPERTURBED base environment (shared media.YPD, 30 C, "
                "no perturbation, 72 h) -- 30 C for the heat records too -- and its "
                "phenotype is ResponseCategory.no_change with category_label 'tolerant'; "
                "the experiment is ResponseCategory.sensitive with category_label "
                "'sensitive' (an UNGRADED hit call, deliberately not mapped to the graded "
                "'reduced'). 10% ethanol / 16% methanol / 7% 1-propanol (percent v/v is a "
                "CONVENTION -- the paper writes neither v/v nor w/v) and 1 M NaCl / 5 mM "
                "H2O2 -> SmallMoleculePerturbation (typed Compound); 37 C heat -> raised "
                "Environment.temperature (no perturbation, M2); media = the SHARED "
                "media.YPD, whose three components the paper independently restates; "
                "assay_type=spot_dilution; n_samples=3 ('performed in triplicate'; "
                "biological vs technical is not stated, and sample_unit="
                "biological_replicate is the independent-experiment reading). Gene tokens "
                "go through the SHARED resolve_gene_name and AMBIGUOUS is a hard stop: "
                "PPA1 -> YHR026W (VMA16) and FEN1 -> YCR034W (ELO2) are adjudicated by "
                "source evidence, never first-matched"
            ),
            page=(
                "J Appl Genet 2009 50(3):301-310 (doi:10.1007/BF03195688; PMC2747848); "
                "paper.pdf sha256=01b945443c0ce41642c76fd737e12b4c31cacb5f384049a5c0a7e4bf9e1eb5a1; "
                "paper.md sha256=d0f3885d1f5027fc29beab7a4327ff377d2bc9c42dd5b87f580049eeda4223b2"
            ),
        ),
    },
    "env_chemgen_smith2006": {
        "root": "data/torchcell/env_chemgen_smith2006",
        # 4249 unique-ORF strains x 3 conditions = 12,747 ordinal records. Of 4770
        # released strains, 521 are dropped: 472 flagged NG/LG/NG-CONT on the YEPD growth
        # control (the screen's own QC column), 26 non-current systematic names, and 23
        # alias-resolutions that would collide with a directly-present R64 ORF.
        "expected_count": 12747,
        # matalpha haploid single-deletion set (BY4742): no constant background.
        "background_genes": frozenset(),
        "provenance": Provenance(
            source_uri=(
                "$DATA_ROOT/torchcell-raw/smithExpressionFunctionalProfiling2006/"
                "data/msb4100051-s1.xls (raw mirror; Supplementary Table 1, retrieved as "
                "one member of the Europe PMC REST supplementaryFiles zip for PMC1681483. "
                "That endpoint re-zips per request, so the CONTAINER hash is not stable; the "
                "record calls retrieve.zip_member with container_sha256=None and pins "
                "the MEMBER hash, which a re-run of the stored record reproduced on "
                "2026-09-13)"
            ),
            citation_key="smithExpressionFunctionalProfiling2006",
            sha256="7048663ffa4890478724e6e371f434baccc7160e6d8250df9a777a26c6b283a4",
            method=(
                "Supplementary Table 1 (.xls, header row 23) per-strain ordinal scores for "
                "the fatty-acid clear-zone screen; one record per (deletion strain x "
                "condition). Clear zone on oleate (YPBO) and myristate (YPBM) scored "
                "4=larger/3=wild type/2=less/1=small-or-absent (assay_type=halo_zone); "
                "growth on acetate (YPBA) scored 3=wild-type/2=moderate/1=little-no with "
                "an undocumented 2.5 kept verbatim (assay_type=colony_size_array). "
                "measurement_type=ordinal: the 1-4 score is stored verbatim on "
                "environment_response, `category` is the shared ResponseCategory call "
                "(4->enhanced, 3->no_change, 2->reduced, 1->severely_reduced on the clear "
                "zone; 3->no_change, 2.5->mildly_reduced, 2->reduced, 1->severely_reduced "
                "on acetate) and `category_label` keeps the screen's own word. Media = the "
                "SHARED media.YPBO / YPBM / YPBA objects, whose typed components carry the "
                "published recipe INCLUDING the fatty acid or acetate as the medium's "
                "carbon_source; the experimental edit is therefore "
                "EnvironmentPhysicalPerturbation(factor=carbon_source, agent=<Compound>, "
                "magnitude=<percent w/v>), the convention condition-SGA uses for galactose, "
                "NOT a SmallMoleculePerturbation (none of the three is a stress on top of a "
                "complete medium; each IS the plate's sole carbon source). 30 C, aerobic, "
                "duration_hours=72.0 -- the source gives a RANGE ('Plates were incubated "
                "for 3-4 days at 30 C') with no companion statistic to back-solve, so the "
                "CLAUDE.md rule takes the CONSERVATIVE lower end rather than the previous "
                "build's unmarked 84.0 h midpoint. n_samples=3 sample_unit="
                "biological_replicate ('Colonies were replicated in triplicate onto "
                "acetate, oleate or myristate agar omnitrays'; the quadruplicate YEPD "
                "pinning is within-plate technical replication and is NOT counted). "
                "Reference = parental BY4742, category no_change / label 'wild_type', no "
                "numeric baseline (the 1-4 scale is an absolute visual ordinal, so the WT "
                "value on it is 3 and asserting 0 would be false). Common names come from "
                "the GENOME's standard name for the resolved ORF via the shared "
                "resolve_gene_name, not the 2005-era Standard Name column (51 of those "
                "resolved to a different gene). Systematic names -> current R64 with "
                "collision-aware alias resolution"
            ),
            page=(
                "Mol Syst Biol 2006 2:2006.0009 (doi:10.1038/msb4100051; PMID 16738555; "
                "PMC1681483); msb4100051-s1.xls "
                "sha256=7048663ffa4890478724e6e371f434baccc7160e6d8250df9a777a26c6b283a4; "
                "paper.md sha256=eb5ab21b842365e2138528bbce936bd68134dd97ee99eb9a58502c25ca2948c6"
            ),
        ),
    },
    "crispr_magic_lian2019": {
        "root": "data/torchcell/crispr_magic_lian2019",
        # 266,304 (guide x round) records of 301,479 cells. Rounds are iterative in
        # accumulating backgrounds (R1 bAID / R2 +SIZ1i / R3 +SIZ1i+NAT1a), so the
        # background is NOT constant across the dataset and belongs in the genotype
        # signature -> background_genes stays empty.
        "expected_count": 266304,
        "background_genes": frozenset(),
        # 266,304 records: the eager path materializes them all (~25 min wall clock).
        "stream": True,
        "provenance": Provenance(
            source_uri=(
                "$DATA_ROOT/torchcell-raw/lianMultifunctionalGenomewideCRISPR2019/"
                "data/guide_enrichment_final.tsv (DERIVED from NCBI SRA PRJNA504483 by the "
                "versioned pipeline in experiments/016-lian-magic-reprocess/, with a "
                "ProcessingRecord naming its inputs) + si/si_data/"
                "41467_2019_13621_MOESM5_ESM.xlsx (Supplementary Data 3, the CRISPRd design "
                "library; springer_esm retrieval re-verified 2026-09-12)"
            ),
            citation_key="lianMultifunctionalGenomewideCRISPR2019",
            sha256="f9af849f97a2d460c3a6d628308491ec3966c6cc2a7f6cad130848d2bad32647",
            method=(
                "The furfural per-guide enrichment is NOT a released supplement; it is "
                "reprocessed from raw reads (SRA PRJNA504483): barcode = read[27:70] (43bp "
                "activation) | read[27:71] (44bp interference/deletion), forward, "
                "exact-match to the 100,493-guide Supplementary Data 4 reference (sha256 "
                "4e3f225a...); CPM(+1)/library; per round per replicate log2(furfural-after "
                "/ untreated-before); mean +- SD over 3 biological triplicates. Validated "
                "vs the paper's hits (PDR1i round-3 rank 1, SLX5i round-1 rank 1, SAP30d "
                "round-1 rank 2). One record per (guide x round): genotype = the library "
                "member as a CrisprActivation/Interference/Deletion perturbation (target "
                "gene + guide spacer + orthogonal effector dLbCas12a-VP / dSpCas9-RD1152 / "
                "SaCas9) PLUS the round's integrated background (SIZ1 interference r2/r3, "
                "NAT1 activation r3, guide unspecified). For CRISPRd the 121 nt design "
                "cassette is SPLIT: the last 21 nt is the SaCas9 spacer on "
                "crispr.guide_sequence and the leading 100 nt is the HR donor on "
                "donor_sequence. The boundary is MEASURED against S288C, not assumed: the "
                "two 50 nt donor arms flank a gap of exactly 28 bp in 193/193 resolvable "
                "sampled designs (matching 'the deletion of 28 bp nucleotides'), the last "
                "21 nt starts at that gap, and for all 24,706 non-control designs it sits "
                "in the genome with a canonical SaCas9 NNGRRT PAM immediately 3' (0 "
                "exceptions). The previous build stored the 44 nt amplicon BARCODE (donor "
                "sequence) as the spacer on 62,793 records. The split is read from the "
                "PUBLISHED Supplementary Data 3, whose Sequence column is element-for-"
                "element identical to the archived lab design file and whose first 44 nt "
                "reproduce the enrichment table's d barcode for all 24,806 rows -- the "
                "positional join the loader asserts at build time. Environment = furfural "
                "(5/10/15 mM by round) as a SmallMoleculePerturbation on the SHARED "
                "media.SED_URA_G418 object, 30 C, aerobic, assay_type="
                "pooled_competitive_growth_barcode; duration_hours AND duration_generations "
                "are typed ProvenanceGaps (harvested at mid-log, neither reported). The "
                "medium is SED-URA/G418, CORRECTED from the previous build's SED/G418, "
                "which is the medium for the integrated validation strains in the same "
                "paragraph and lacks the uracil dropout selecting the guide plasmid. "
                "Phenotype = log2_ratio, uncertainty = SD (sample_sd, n=3 -> SE=SD/sqrt(3)); "
                "reference = no-enrichment baseline (log2FC 0) in the bAID host (kept as "
                "the reference strain: pAID6's integration locus is never stated, so its "
                "three Cas cassettes cannot be sourced as gene-keyed additions, and the "
                "effectors ride on CrisprConstruct.effector instead). Common names from the "
                "genome's standard name via the shared resolve_gene_name. Dropped: 300 "
                "controls, 16 source-corrupted names, 2,670 unresolved-gene guides, 26,169 "
                "undetected guide-rounds and 48 self-background guide-rounds. The 318 records in "
                "150 groups where two CRISPRd designs share a gene AND a 21 nt spacer "
                "(multicopy loci, differing only in their donor arms) are KEPT: "
                "donor_sequence is what tells those strains apart"
            ),
            page=(
                "Nat Commun 2019 10:5794 (doi:10.1038/s41467-019-13621-4; PMID 31857575); "
                "guide_enrichment_final.tsv "
                "sha256=f9af849f97a2d460c3a6d628308491ec3966c6cc2a7f6cad130848d2bad32647; "
                "Supplementary Data 3 "
                "sha256=737074a76b9eee2dc015be8b17e29b4fbe65c8be5565e6fcbe71505dca4109e2; "
                "paper.md sha256=63fe2b7101fc48feb297f9e34b83d108b74f03f28bbc280e08c7219bc975086c"
            ),
        ),
    },
    "crispri_mormino2022": {
        "root": "data/torchcell/crispri_mormino2022",
        # 12 individually-isolated CRISPRi strains (Table 1), one categorical
        # acetic-acid biosensor call each. No record is dropped.
        "expected_count": 12,
        # The pMM4_14L biosensor cassette (two GeneAdditionPerturbations integrated at HO)
        # is in EVERY strain and in the comparator, so it is a constant background: it
        # must be excluded from the L1 strain identity and from the L4 gene rules, which
        # a heterologous cassette cannot satisfy by construction.
        "background_genes": frozenset({"BM3R1-HAA1-mTurquoise2", "sfpHluorin"}),
        "provenance": Provenance(
            source_uri=(
                "$DATA_ROOT/torchcell-raw/morminoIdentificationAceticAcid2022/paper.pdf "
                "(raw mirror; BMC counter URL, direct_url, re-verified 2026-09-12) + "
                "paper.md (the sha256-pinned OCR whose machine-readable Table 1 the build "
                "audits every stored row against). No SI data file was released"
            ),
            citation_key="morminoIdentificationAceticAcid2022",
            sha256="388f8e922b0b94fba3a41965035eeee0f6180a110869073a426df96b5e63746a",
            method=(
                "Table 1 'Properties of isolated strains' (12 rows), a module literal whose "
                "every row is CHECKED verbatim against the OCR'd HTML table in paper.md "
                "(sha256 f5d38e48...) before any record is written. Each strain = one "
                "CrisprInterferencePerturbation (target gene, effector dCas9-Mxi1, "
                "n_guides=1, guide_sequence=None -- Mormino releases no per-strain spacer, "
                "the guides live upstream in the Smith 2017 BY4742-derived library) PLUS "
                "the constant pMM4_14L biosensor cassette as two GeneAdditionPerturbations "
                "integrated at the HO locus ('was integrated into the HO locus of the "
                "CRISPRi library strains'). Environment = the SHARED media.SC object "
                "(CORRECTED from the previous build's 'SD, pH 3.5': the paper says "
                "synthetic complete, and SC's own 0.77 g/L CSM / 6.9 g/L YNB w/o AA / 20 "
                "g/L glucose are sourced from THIS paper) carrying three typed edits -- "
                "acetic acid 50 mM (NaOH-titrated to the medium pH), anhydrotetracycline 2 "
                "ug/mL with a DMSO Solvent (the CRISPRi inducer, absent from the previous "
                "build although released), and pH 3.5 as EnvironmentPhysicalPerturbation"
                "(factor=ph) -- at 30 C, aerobic. Phenotype = measurement_type=categorical, "
                "assay_type=biosensor_readout: Table 1's '+' (reporter expression 30% "
                "higher than the CBL) -> ResponseCategory.enhanced, '=' (similar) -> "
                "no_change, with category_label keeping the source symbol. n_samples=2 "
                "sample_unit=biological_replicate ('Screening of the pooled and single cell "
                "cultures sorted by FACS was performed in two biological replicates'). "
                "Reference = the CBL pool (no_change / '='), CORRECTED from the previous "
                "build's CC23 control strain: Table 1's own footnote says '*Reporter "
                "expression 30% higher (+) or similar (=) compared to the CBL', and CC23 is "
                "the comparator for the Growth column and the separate 150 mM experiments. "
                "The Growth and essentiality columns and the figure-only FI / sfpHluorin / "
                "growth values are NOT ingested. 12 targets resolve to current R64 genes"
            ),
            page=(
                "Microb Cell Fact 2022 21:214 (doi:10.1186/s12934-022-01938-7; PMID "
                "36284296; PMC9571444), Table 1; paper.pdf "
                "sha256=388f8e922b0b94fba3a41965035eeee0f6180a110869073a426df96b5e63746a; "
                "paper.md sha256=f5d38e486148527bfba9dc9e40a9eb06ba051766aeca3e8ff67663551bf043c3"
            ),
        ),
    },
    "env_chemgen_costanzo2021": {
        "root": "data/torchcell/env_chemgen_costanzo2021",
        "expected_count": 61430,
        "background_genes": frozenset(),
        "provenance": Provenance(
            source_uri=(
                "$DATA_ROOT/torchcell-raw/costanzoEnvironmentalRobustnessGlobal2021/"
                "data/Costanzo et al_Data File S1_Conditions_Strains_Fitness.xlsx "
                "(raw mirror; Science SI is HTTP 403 behind a Cloudflare challenge, so "
                "the file is manual-once -> deposited); Methods sourced from the library "
                "mirror's paper.md"
            ),
            citation_key="costanzoEnvironmentalRobustnessGlobal2021",
            sha256="f6c313de416ce8cc6ae87e2020b4389bd4adeb07cdb6a438aecaf1e45e6228ad",
            method=(
                "condition-SGA single-mutant fitness: DIFFERENTIAL mutant fitness = "
                "normalized colony-size fitness in a test condition minus the matched "
                "reference condition ('To obtain condition-specific fitness estimates, we "
                "computed the difference in colony size measured in a particular test "
                "condition versus the matched reference condition for each mutant.'); "
                "measurement_type=differential_fitness, assay_type=colony_size_array, "
                "n_samples=3 sample_unit=screen ('an average of three replicate control "
                "screens conducted per each of 14 test conditions as well as the reference "
                "condition at 26 C'). 26 C is a DERIVATION, not a quote: the clause closes "
                "a list ending with the reference condition, and 26.0 is applied to all 15 "
                "conditions because one screen's three array copies share an incubator "
                "('every double-mutant array generated from a singlequery SGA screen was "
                "copied three times') and the TS array needs a permissive temperature. "
                "Media = the SHARED media.SGA_DM_SELECTION (identical to served Costanzo "
                "2016), except the galactose condition, which is the derived "
                "media.SGA_DM_SELECTION_GALACTOSE with NO perturbation ('an alternative "
                "carbon source' = replacement); the other 13 conditions are "
                "SmallMoleculePerturbations. Environment.duration_hours is a typed "
                "ProvenanceGap (the source's reference sheet splits 3-day vs 5-day "
                "incubation; the differential's choice is in the un-mirrored SI). ORFs are "
                "resolved with the SHARED SCerevisiaeGenome.resolve_gene_name: CURRENT "
                "(4396) + RENAMED (18) kept under the current systematic name, "
                "NON_GENE_FEATURE (12 ORFs, 168 cells) and RETIRED (3 ORFs, 42 cells) "
                "dropped -- 4414 strains x 14 conditions minus empty cells = 61,430"
            ),
            page=(
                "Science 2021 372(6542):eabf8424 (doi:10.1126/science.abf8424; PMID "
                "33958448; PMC9132594); Data File S1 sheet 'Diff. Mutant fitness_Conditions'; "
                "S1 sha256=f6c313de416ce8cc6ae87e2020b4389bd4adeb07cdb6a438aecaf1e45e6228ad; "
                "paper.md sha256=ba22973ed0c53c00c37bcfb9f659d3b0373c451a3f7633158afae274035559fb. "
                "Three deposited concentrations are recorded verbatim and FLAGGED as SI unit "
                "anomalies (bortezomib '1300 mM'; actinomycin D '20 mM'; geldanamycin '10 mM'), "
                "and two bare fractions are read as percent (galactose '0.02' -> 2% w/v; MMS "
                "'0.0001' -> 0.01% v/v); each reading lives in its SourcedValue note, not in "
                "the phenotype's units string"
            ),
        ),
    },
    "env_chemgen_hillenmeyer2008_het": {
        "root": "data/torchcell/env_chemgen_hillenmeyer2008_het",
        # one record per (strain row, environment, control set) after the array
        # and strain drop rules (#505); counted by
        # experiments/036-dataset-fixes-before-kg-build/scripts/hillenmeyer2008_inputs.py
        # through the loader's own functions; rules + counts in preprocess/.
        "expected_count": 2712677,
        "background_genes": frozenset(),
        "stream": True,
        "provenance": Provenance(
            source_uri=(
                "$DATA_ROOT/torchcell-raw/hillenmeyerChemicalGenomicPortrait2008/data/het.ratio_result_nm.pub "
                "(raw mirror with manifest.json; Wayback capture 20151207003548 of the "
                "chemogenomics.stanford.edu FitDb download, re-retrieved bit-identically 2026-09-12)"
            ),
            citation_key="hillenmeyerChemicalGenomicPortrait2008",
            sha256="c0ddefaeb44c9760481dc443d1c2e82570bad4eb1a5458acd74b169bec47cb44",
            method="genome-scale HIP fitness-defect log2_ratio = log2(mean control intensity / treatment intensity), up+down tag mean; positive = fitness defect. HET = HeterozygousDeletionPerturbation (kanMX4, Giaever 2014; construction batch, pool, barcodes gapped; replaced_allele gapped at the BY marker loci), incl. essential genes. One record per (constructed STRAIN = matrix row ORF:batch, environment, CONTROL SET): the SOM defines the score against a matched no-drug control set (same pool, generation count and scanner), so the control-set id is stored on phenotype.screen_id and one reference is emitted per control set with that set's control-array count as n_samples; rows of one ORF are different constructions and are never averaged. Reference genome StrainReferenceGenome BY4743 (named by the hom.txt key file; alleles, mating type and parents pending Brachmann 1998 / Giaever 2002). Renamed source ORFs keep their own records under the current gene with a ConstructedOrf ('merged' when two source ORFs land on one gene, else a gap). Every array's header is cross-checked against its key file: pH conflicts serve the header (Table S1 lists only pH7.5/8), the 18 hom 'minimal media' arrays the key calls 'synthetic complete for BY4743' are served on SC, compound-name conflicts are dropped. Conditions per SOM Table S1: temperature on Environment.temperature; pH; nutrient drop-outs as DERIVED SC media (partial levels checked against the release); minimal media as SD with a typed auxotroph-supplement gap; '<compound> irradiated' as compound + radiation; cond2 appended in every branch; base medium YPD_LIQUID; compounds carry a typed vehicle gap. CultureEnvironment: duration_generations = magnitude, pre_culture from the SIGN (negative frozen_stock, positive YPD log phase to OD600 2.0, ~10 generations), culture_format and temperature gapped (Pierce 2006). Molar doses canonicalized exactly (Decimal). DROPPED: 64 environments / 75 arrays -- 66 arrays naming a compound with no structure identifier, the '37c, 45c' heat-shock cycle, 4 header/key compound conflicts, 4 0gen arrays; strains: the 11 YDL227C:ctrl_* HO control rows (preprocess/dropped_strains.json). ORFs go through resolve_gene_name (renamed kept, retired/non-gene dropped to preprocess/dropped_genes.json)",
            page=(
                "Science 2008 320:362 (doi:10.1126/science.1150021); SOM Table S1; raw mirror "
                "manifest.json pinning every file; het.ratio_result_nm.pub sha256=c0ddefaeb44c9760481dc443d1c2e82570bad4eb1a5458acd74b169bec47cb44"
            ),
        ),
    },
    "env_chemgen_hillenmeyer2008_hom": {
        "root": "data/torchcell/env_chemgen_hillenmeyer2008_hom",
        # one record per (strain row, environment, control set) after the array
        # and strain drop rules (#505); counted by
        # experiments/036-dataset-fixes-before-kg-build/scripts/hillenmeyer2008_inputs.py
        # through the loader's own functions; rules + counts in preprocess/.
        "expected_count": 1063034,
        "background_genes": frozenset(),
        "stream": True,
        "provenance": Provenance(
            source_uri=(
                "$DATA_ROOT/torchcell-raw/hillenmeyerChemicalGenomicPortrait2008/data/hom.z_result_nm.pub "
                "(raw mirror with manifest.json; Wayback capture 20151207003024 of the "
                "chemogenomics.stanford.edu FitDb download, re-retrieved bit-identically 2026-09-12)"
            ),
            citation_key="hillenmeyerChemicalGenomicPortrait2008",
            sha256="0b7d5e4dad0e5b4336b1dac97fb32ef14d7d0bf9f5d1b0d5ddb382c393362b71",
            method="genome-scale HOP fitness-defect z_score = (mean control - treatment)/SD control; positive = fitness defect. HOM = BarcodedKanMxDeletionPerturbation (kanMX4; construction batch, pool; barcodes gapped; constructed_orf gapped at the BY marker loci). One record per (constructed STRAIN = matrix row ORF:batch, environment, CONTROL SET): the SOM defines the score against a matched no-drug control set (same pool, generation count and scanner), so the control-set id is stored on phenotype.screen_id and one reference is emitted per control set with that set's control-array count as n_samples; rows of one ORF are different constructions and are never averaged. Reference genome StrainReferenceGenome BY4743 (named by the hom.txt key file; alleles, mating type and parents pending Brachmann 1998 / Giaever 2002). Renamed source ORFs keep their own records under the current gene with a ConstructedOrf ('merged' when two source ORFs land on one gene, else a gap). Every array's header is cross-checked against its key file: pH conflicts serve the header (Table S1 lists only pH7.5/8), the 18 hom 'minimal media' arrays the key calls 'synthetic complete for BY4743' are served on SC, compound-name conflicts are dropped. Conditions per SOM Table S1: temperature on Environment.temperature; pH; nutrient drop-outs as DERIVED SC media (partial levels checked against the release); minimal media as SD with a typed auxotroph-supplement gap; '<compound> irradiated' as compound + radiation; cond2 appended in every branch; base medium YPD_LIQUID; compounds carry a typed vehicle gap. CultureEnvironment: duration_generations = magnitude, pre_culture from the SIGN (negative frozen_stock, positive YPD log phase to OD600 2.0, ~10 generations), culture_format and temperature gapped (Pierce 2006). Molar doses canonicalized exactly (Decimal). DROPPED: 44 environments / 50 arrays -- 32 arrays naming a compound with no structure identifier, the two 'minimal media:400:um' arrays whose agent is named nowhere, 7 header/key compound conflicts, 9 0gen arrays; strains: hom PDR5 YOR153W:chr15_2 (the SOM: wrong gene deleted) and the 11 YDL227C:ctrl_* rows. ORFs go through resolve_gene_name (renamed kept, retired/non-gene dropped to preprocess/dropped_genes.json)",
            page=(
                "Science 2008 320:362 (doi:10.1126/science.1150021); SOM Table S1; raw mirror "
                "manifest.json pinning every file; hom.z_result_nm.pub sha256=0b7d5e4dad0e5b4336b1dac97fb32ef14d7d0bf9f5d1b0d5ddb382c393362b71"
            ),
        ),
    },
    "crispri_chemgen_smith2016": {
        "root": "data/torchcell/crispri_chemgen_smith2016",
        # 7,053 of the 14,463 rows of Additional file 10. 7,410 drop on the compound
        # rule: ten Drug labels are ChemDiv / ChemBridge / TimTec catalog ids for which the
        # primary released no structure, so no InChIKey / ChEBI / CID exists. L1 strain
        # identity keys on (gene, crispr_interference, guide spacer, library_pool).
        "expected_count": 7053,
        "background_genes": frozenset(),
        "provenance": Provenance(
            source_uri=(
                "$DATA_ROOT/torchcell-raw/smithQuantitativeCRISPRInterference2016/"
                "si/si_data/13059_2016_900_MOESM10_ESM.xlsx (Additional file 10, sheet "
                "'Fitness and Effect Data') + 13059_2016_900_MOESM4_ESM.xlsx (Additional "
                "file 4, sheet 'gRNAs', guide->spacer); raw mirror, both with a "
                "springer_esm retrieval re-verified 2026-09-12"
            ),
            citation_key="smithQuantitativeCRISPRInterference2016",
            sha256="02962e51e492b0505e8595fc1c80fab5fca8a8c8f05e05969dbff18ddff71cd0",
            method=(
                "One EnvironmentResponseExperiment per (pool, guide, drug, concentration) "
                "row of Additional file 10. measurement_type=log2_ratio, assay_type="
                "pooled_competitive_growth_barcode, environment_response = column 'A' = the "
                "ATc-induced fold change (A_ijk = f_ijk+ - f_ijk-, the difference of log2 "
                "median-centred guide read-count fitness between induced +ATc and uninduced "
                "-ATc cultures in that drug; negative = repression is a growth defect). "
                "Uncertainty = column 'var(A)' stored VERBATIM as UncertaintyType.variance "
                "with n_samples=1, DELIBERATELY: var(A) is the variance of the SINGLE "
                "released estimate, already inverse-variance-combined across the 8 and 3 "
                "replicate experiments for the 1% DMSO and 20 uM fluconazole conditions, so "
                "derive_se must divide by 1 (recording 8 would shrink the SE by sqrt(8)); "
                "the replicate structure is recorded in the loader's REPLICATE_STRUCTURE "
                "SourcedValue instead. Genotype = CrisprInterferencePerturbation(target ORF, "
                "effector dCas9-Mxi1, guide_sequence = the 18/20 nt Specificity_sequence "
                "spacer joined by guide name from Additional file 4 (sha256 e5eb4e3c...), "
                "library_pool = #Pool; the same spacer in two pools is two independent "
                "pooled measurements). Environment = the SHARED media.SC_URA object (the "
                "paper's SCM-Ura; Smith releases no recipe, so SC_URA's own gaps are the "
                "honest state) + the row's Drug at its released dose as a "
                "SmallMoleculePerturbation (uM/nM; the 1% DMSO vehicle control -> 1.0 "
                "percent_v/v), 30 C, aerobic, duration_generations=20.0 ('approximately 20 "
                "culture doublings') with duration_hours a typed ProvenanceGap. ATc is NOT "
                "asserted: the pooled Methods say only '+/- ATc' and the one released "
                "number (250 ng/mL) is from the qPCR section; no Solvent is asserted "
                "either, since 'Drugs were dissolved in DMSO' sits in the individual-strain "
                "section. The ~20%-growth-inhibition dose rule is real (Additional file 8 "
                "ReadMe) but DoseBasis has no IC20 member and adding one is a full-rebuild "
                "trigger, so it is documented not typed. Reference = uninduced -ATc "
                "baseline (A=0) in BY4741. Common names from the genome's standard name. "
                "20 target ORFs all current R64"
            ),
            page=(
                "Genome Biol 2016 17:45 (doi:10.1186/s13059-016-0900-9); Additional file 10 "
                "'Fitness and Effect Data' "
                "sha256=02962e51e492b0505e8595fc1c80fab5fca8a8c8f05e05969dbff18ddff71cd0; "
                "Additional file 4 'gRNAs' "
                "sha256=e5eb4e3c7856782e36edff5ef55e680cb43fa8e9944bf8e40d7cb7b4f376c1e5; "
                "paper.md sha256=346be6968eced82706cffb163a76dfc5adf381e602bd11006fea608436ca7b2f"
            ),
        ),
    },
}


def run_environment_response(data_root: str) -> bool:
    """Verify environment-response datasets (L0-L4) and write reports. True if all pass."""
    sgd_genes = _sgd_gene_set(data_root)
    resolve_gene_name = _genome(data_root).resolve_gene_name
    all_passed = True
    for name, spec in ENVIRONMENT_RESPONSE_DATASETS.items():
        abs_root = osp.join(data_root, spec["root"])
        background = spec.get("background_genes", frozenset())
        if spec.get("stream"):
            # Large dataset: single-pass, memory-bounded verification (never materialized).
            report = verify_environment_response_dataset_streaming(
                stream_records(abs_root),
                dataset_name=name,
                provenance=spec["provenance"],
                expected_count=spec["expected_count"],
                sgd_genes=sgd_genes,
                background_genes=background,
                min_containment=MIN_RNASEQ_GENE_CONTAINMENT,
                resolve_gene_name=resolve_gene_name,
            )
        else:
            records = load_records(abs_root)
            report = verify_environment_response_dataset(
                records,
                dataset_name=name,
                provenance=spec["provenance"],
                expected_count=spec.get("expected_count", len(records)),
                background_genes=background,
                resolve_gene_name=resolve_gene_name,
                sgd_genes=sgd_genes,
                min_containment=MIN_RNASEQ_GENE_CONTAINMENT,
            )
        out = _write_report(report, osp.join(abs_root, "preprocess"))
        print(report.summary())
        print(f"  -> wrote {out}\n")
        all_passed = all_passed and report.passed
    return all_passed


# --------------------------------------------------------------------------- #
# Single-mutant fitness datasets
# --------------------------------------------------------------------------- #
FITNESS_DATASETS: dict[str, dict[str, Any]] = {
    "smf_oduibhir2014": {
        "root": "data/torchcell/smf_oduibhir2014",
        "expected_count": 1312,
        "provenance": Provenance(
            source_uri="https://doi.org/10.15252/msb.20145172",
            citation_key="oduibhirCellCyclePopulation2014",
            method=(
                "per-deletion relative growth: fitness = 2^(-log2relT) where log2relT = "
                "log2(doubling_mut/doubling_wt) (Supplementary Dataset S2); WT == 1.0, "
                "sick < 1 (matches stored Costanzo SMF convention); n_samples=2 biological "
                "replicate cultures, no released per-strain SE"
            ),
            page="Mol Syst Biol 10:732; Supplementary Dataset S2 'data set 2.txt'",
            sha256=("37ef19ee249c64c0557c84870e59b2fd7a8bbaf14371fd355775e650f2a39f1c"),
        ),
    },
    "smf_baryshnikova2010": {
        "root": "data/torchcell/smf_baryshnikova2010",
        # 6023 released alleles minus the 30 frozen in the loader's _UNRESOLVABLE
        # (== baryshnikova2010.EXPECTED_RECORDS; kept literal here because this module
        # must not import loaders).
        "expected_count": 5993,
        "provenance": Provenance(
            source_uri=(
                "$DATA_ROOT/torchcell-raw/baryshnikovaQuantitativeAnalysisFitness2010/"
                "data/SupplementaryData1_SMF.xls (Springer ESM MOESM168, retrieved and "
                "re-hashed before deposit)"
            ),
            citation_key="baryshnikovaQuantitativeAnalysisFitness2010",
            method=(
                "genome-scale SGA single-mutant fitness (Supplementary Data 1, sheet "
                "S1_SMF_standard_100209, publisher xls from Springer ESM MOESM168 deposited "
                "in torchcell-raw/baryshnikovaQuantitativeAnalysisFitness2010): "
                "WT-normalized so the fitness distribution mode == 1.0; uncertainty = "
                "bootstrap SE of the median (SI Note 1); n_samples=80 control screens is "
                "stored for the 4635 ARRAY-side kanMX deletions only, the 1388 query-side "
                "(_damp/_tsq) rows carry a typed ProvenanceGap on n_samples; environment = "
                "SGA_DM_SELECTION at 30 C for deletions and DAmP (Tong and Boone 2006 SGA "
                "protocol step 18) and at 26 C for the TS alleles (Costanzo 2016 SI "
                "semipermissive final selection, back-solved against Costanzo's 26/30 C "
                "SMF columns on 291 shared strain ids); 6023 released alleles minus the 30 "
                "frozen in _UNRESOLVABLE = 5993, with the raw allele id on strain_id so the "
                "TS allelic series stays distinct"
            ),
            page="Nat Methods 7:1017; Supplementary Data 1 sheet 'S1_SMF_standard_100209'",
            sha256="086bfadf2684f28940500dd87e3be74c53a957448d2016f7a02370540da8a04e",
        ),
    },
}


def run_fitness(data_root: str) -> bool:
    """Verify single-mutant fitness datasets (L0-L4) and write reports. True if all pass.

    The gene universe and the canonical-name resolver belong to the HOST a dataset's own
    records are written against (plan section 5 item 8): a bacterial fitness dataset's
    locus tags resolved against S288C would fail every record for the wrong reason. Both
    are built once per assembly set and reused, since each construction reads the tier.
    """
    universes: dict[tuple[str, ...], set[str]] = {}
    resolvers: dict[tuple[str, ...], Any] = {}
    all_passed = True
    for name, spec in FITNESS_DATASETS.items():
        abs_root = osp.join(data_root, spec["root"])
        records = load_records(abs_root)
        assembly_sets = _dataset_assembly_sets(records)
        if assembly_sets not in universes:
            universes[assembly_sets] = _dataset_gene_universe(records, data_root)[0]
            # One resolver per dataset's host. A dataset pinned to TWO assemblies
            # (Tong 2020's Keio plus sRNA library) cannot share one resolver, so it
            # runs its own per-background verification in its loader module and is not
            # registered here until that is settled.
            if len(assembly_sets) > 1:
                raise ValueError(
                    f"{name}: records name {len(assembly_sets)} assembly sets "
                    f"{assembly_sets}; one resolver cannot serve two hosts"
                )
            resolvers[assembly_sets] = _genome_for_reference(
                records[0]["reference"]["genome_reference"], data_root
            ).resolve_gene_name
        report = verify_fitness_dataset(
            records,
            dataset_name=name,
            provenance=spec["provenance"],
            expected_count=spec.get("expected_count", len(records)),
            resolve_gene_name=resolvers[assembly_sets],
            sgd_genes=universes[assembly_sets],
            min_containment=MIN_RNASEQ_GENE_CONTAINMENT,
        )
        out = _write_report(report, osp.join(abs_root, "preprocess"))
        print(report.summary())
        print(f"  -> wrote {out}\n")
        all_passed = all_passed and report.passed
    return all_passed


# --------------------------------------------------------------------------- #
# Segregant growth datasets (haplotype-mosaic genotypes)
# --------------------------------------------------------------------------- #
SEGREGANT_GROWTH_DATASETS: dict[str, dict[str, Any]] = {
    "bloom2019": {
        "root": "data/torchcell/bloom2019",
        "raw_mirror": "torchcell-raw/bloomRareVariantsContribute2019",
        # the member index is resolved from the genomes tier (Peter 2018 set)
        "assembly_index": "1011Assemblies.tar.gz.member_index.tsv",
        # 13,950 segregants x 38 served conditions (YPD;;2 / YPD;;3 are regressors,
        # not conditions; 4NQO was removed upstream).
        "expected_count": 530100,
        "provenance": Provenance(
            source_uri="https://doi.org/10.7554/eLife.49212",
            citation_key="bloomRareVariantsContribute2019",
            method=(
                "16 biparental crosses of 1011-collection strains, 13,950 haploid "
                "segregants genotyped by R/qtl argmax hard calls (released "
                "genotype_<cross>.tsv.gz, 1 = parent 1 / 2 = parent 2) and encoded as "
                "haplotype-block mosaics; 38 end-point colony-growth conditions from "
                "phenotypes.tsv.gz: 36 are residuals of colony mean radius regressed on "
                "the same-batch control plate (process_images.R), 2 are absolute radii; "
                "duplicate plates averaged, missing cells mean-imputed (mapping.R); "
                "raw mirror torchcell-raw/bloomRareVariantsContribute2019 with manifest.json "
                "(Dropbox share zip sha256 78ded0db..., xls sha256 990e7516..., eLife XML "
                "sha256 0cfa345e..., code at joshsbloom/yeast-16-parents c913c9ae)"
            ),
            page="eLife 8:e49212; Methods 'Phenotyping by endpoint colony growth'; Figure 1 source data 1",
            sha256=("3942dbbc9280536f90cf4fc41ce39649cd2c638be623706a152b3a5406765575"),
        ),
    }
}


def run_segregant_growth(data_root: str) -> bool:
    """Verify segregant growth datasets (L0-L4) and write reports. True if all pass."""
    from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

    sgd_genes = _sgd_gene_set(data_root)
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    all_passed = True
    for name, spec in SEGREGANT_GROWTH_DATASETS.items():
        abs_root = osp.join(data_root, spec["root"])
        report = verify_segregant_growth_streaming(
            stream_records(abs_root),
            dataset_name=name,
            provenance=spec["provenance"],
            expected_count=spec["expected_count"],
            raw_dir=osp.join(abs_root, "raw"),
            raw_mirror=osp.join(data_root, spec["raw_mirror"]),
            assembly_index_path=resolve(
                PETER2018_1011, spec["assembly_index"], data_root=data_root
            ),
            genome=genome,
            sgd_genes=sgd_genes,
            gene_set=segregant_gene_set(osp.join(abs_root, "preprocess")),
        )
        out = _write_report(report, osp.join(abs_root, "preprocess"))
        print(report.summary())
        print(f"  -> wrote {out}\n")
        all_passed = all_passed and report.passed
    return all_passed


# --------------------------------------------------------------------------- #
# Product titer and bacterial protein abundance (the bioproduction families)
#
# Both families are bacterial, and each dataset's L0-L3 gate plus its cross-source
# oracles live in the loader module that owns its readers: the raw-mirror joins a titer
# oracle needs (a released per-target table, a fed-batch time course, a titer quoted in
# the Results prose) cannot be written once for the family, and the proteome datasets
# additionally re-audit every ``SourcedValue`` quote against pinned library bytes. So
# these two runners DISPATCH to each dataset's own entry point rather than re-stating
# its checks in a second place, and add the one rule that is the family's own: every
# host identifier a record carries is a locus of the assembly that record's own
# ``genome_reference`` pins.
# --------------------------------------------------------------------------- #
def _verify_carruthers_titer(dataset_root: str, data_root: str) -> VerificationReport:
    """Carruthers 2025 isoprenol titer: the shared titer gate plus its own two rules."""
    from torchcell.datasets.pputida import carruthers2025

    return carruthers2025.verify_build(dataset_root, data_root, family="titer")


def _verify_carruthers_proteome(
    dataset_root: str, data_root: str
) -> VerificationReport:
    """Carruthers 2025 proteome: the shared protein gate plus its own three rules."""
    from torchcell.datasets.pputida import carruthers2025

    return carruthers2025.verify_build(dataset_root, data_root, family="proteome")


def _verify_desiqueira_titer(dataset_root: str, data_root: str) -> VerificationReport:
    """The de Siqueira 2025 titer battery, with its Results-text cross-source row."""
    from torchcell.datasets.pputida import desiqueira2025

    return desiqueira2025.verify_build(dataset_root, data_root, family="titer")


def _verify_desiqueira_proteome(
    dataset_root: str, data_root: str
) -> VerificationReport:
    """The de Siqueira 2025 proteome: the shared gate, its pin and its containment."""
    from torchcell.datasets.pputida import desiqueira2025

    return desiqueira2025.verify_build(dataset_root, data_root, family="proteome")


def _verify_kang_titer(dataset_root: str, data_root: str) -> VerificationReport:
    """Kang 2026: the Table 1 / Table S4 / Table S9 titers and their two L4 joins."""
    from torchcell.datasets.pputida import kang2026

    return kang2026.verify_build(dataset_root, data_root)


def _verify_lim_proteome(dataset_root: str, data_root: str) -> VerificationReport:
    """Lim 2025 proteome: the shared protein gate, its containment and quote audits.

    ``dataset_root`` is unused: this entry point resolves its own root under
    ``data_root``, and the registry's root is what the runner reads and writes.
    """
    from torchcell.datasets.pputida import lim2025

    return lim2025.run_proteome_verification(data_root)


def _verify_caglar_proteome(dataset_root: str, data_root: str) -> VerificationReport:
    """Caglar 2017 proteome: the shared protein gate, the VST back-solve and REL606.

    ``dataset_root`` is unused, as in :func:`_verify_lim_proteome`: this entry point
    takes the family name and resolves the root itself.
    """
    from torchcell.datasets.ecoli import caglar2017

    return caglar2017.run_verification("proteome", data_root)


#: Every landed ``ProductTiterExperiment`` dataset, with the entry point that verifies it.
PRODUCT_TITER_DATASETS: dict[str, dict[str, Any]] = {
    "isoprenol_titer_carruthers2025": {
        "root": "data/torchcell/isoprenol_titer_carruthers2025",
        "verify": _verify_carruthers_titer,
    },
    "isoprenol_titer_desiqueira2025": {
        "root": "data/torchcell/isoprenol_titer_desiqueira2025",
        "verify": _verify_desiqueira_titer,
    },
    "isoprenyl_acetate_titer_kang2026": {
        "root": "data/torchcell/isoprenyl_acetate_titer_kang2026",
        "verify": _verify_kang_titer,
    },
}

#: Every landed ``BacterialProteinAbundanceExperiment`` dataset. Caglar 2017 is here
#: with the three P. putida proteomes: it is the same experiment class on a different
#: host, and the containment rule reads its REL606 pin off its own records.
BACTERIAL_PROTEIN_ABUNDANCE_DATASETS: dict[str, dict[str, Any]] = {
    "proteome_carruthers2025": {
        "root": "data/torchcell/proteome_carruthers2025",
        "verify": _verify_carruthers_proteome,
    },
    "proteome_desiqueira2025": {
        "root": "data/torchcell/proteome_desiqueira2025",
        "verify": _verify_desiqueira_proteome,
    },
    "proteome_lim2025": {
        "root": "data/torchcell/proteome_lim2025",
        "verify": _verify_lim_proteome,
    },
    "proteome_caglar2017": {
        "root": "data/torchcell/proteome_caglar2017",
        "verify": _verify_caglar_proteome,
    },
}


def host_perturbed_gene_set(records: Sequence[Mapping[str, Any]]) -> set[str]:
    """Every perturbed identifier a dataset's records assert is a locus of their host.

    The perturbed systematic names MINUS the heterologous ones. A
    ``HeterologousPathwayPerturbation`` carries ``source_organism``, and when that
    organism is not the record's own species the identifier is a gene of ANOTHER genome
    (``MvaSEf``, ``ATF1``) -- which is exactly what the class exists to say, so checking
    it against the host's locus universe would fail every production record for the
    wrong reason. An extra copy of a NATIVE gene names its real locus tag and stays in.
    """
    genes: set[str] = set()
    for record in records:
        species = record["reference"]["genome_reference"]["species"]
        for perturbation in record["experiment"]["genotype"]["perturbations"]:
            name = perturbation.get("systematic_gene_name")
            if name is None or perturbation.get("source_organism", species) != species:
                continue
            genes.add(str(name))
    return genes


def bacterial_protein_locus_set(records: Sequence[Mapping[str, Any]]) -> set[str]:
    """The host loci a protein-abundance dataset names: its quantified proteins and its
    perturbed genes.

    Both are identifiers the records claim are loci of their own pinned assembly, so
    containment over their union is the whole L4 question for this family and is never
    empty (a panel whose arms are environmental, like Caglar's, perturbs no gene but
    quantifies thousands of proteins).
    """
    measured = host_perturbed_gene_set(records)
    for record in records:
        measured |= set(record["experiment"]["phenotype"]["protein_abundance"])
    return measured


def _run_bacterial_family(
    datasets: Mapping[str, Mapping[str, Any]],
    data_root: str,
    *,
    measured_set: Callable[[Sequence[Mapping[str, Any]]], set[str]],
    l4_name: str,
) -> bool:
    """Verify one bioproduction family: each dataset's own gate plus the L4 containment.

    ``min_containment`` is 1.0 and not a floor with headroom: every identifier these
    records carry was written by a loader that resolved it against the pinned assembly,
    so one that is not a locus of that assembly is a build error, not an accepted edge.
    """
    all_passed = True
    for spec in datasets.values():
        abs_root = osp.join(data_root, spec["root"])
        report = spec["verify"](abs_root, data_root)
        records = load_records(abs_root)
        universe, assembly_sets = _dataset_gene_universe(records, data_root)
        report.add(
            _l4_assembly_gene_containment(
                universe, assembly_sets, measured_set(records), min_containment=1.0
            ).model_copy(update={"name": l4_name})
        )
        out = _write_report(report, osp.join(abs_root, "preprocess"))
        print(report.summary())
        print(f"  -> verified LMDB: {osp.join(abs_root, 'processed', 'lmdb')}")
        print(f"  -> wrote report:  {out}\n")
        all_passed = all_passed and report.passed
    return all_passed


def run_product_titer(data_root: str) -> bool:
    """Verify every product-titer dataset (L0-L4) and write reports. True if all pass."""
    return _run_bacterial_family(
        PRODUCT_TITER_DATASETS,
        data_root,
        measured_set=host_perturbed_gene_set,
        l4_name="perturbed_gene_containment_assembly",
    )


def run_bacterial_protein_abundance(data_root: str) -> bool:
    """Verify every bacterial protein-abundance dataset (L0-L4). True if all pass."""
    return _run_bacterial_family(
        BACTERIAL_PROTEIN_ABUNDANCE_DATASETS,
        data_root,
        measured_set=bacterial_protein_locus_set,
        l4_name="protein_and_perturbed_locus_containment_assembly",
    )


def run_all(data_root: str) -> bool:
    """Run every dataset-family verification. True only if all pass."""
    expression_ok = run_expression(data_root)
    morphology_ok = run_morphology(data_root)
    morphology_ohnuki_ok = run_morphology_ohnuki(data_root)
    ohnuki_ok = run_ohnuki_morphology(data_root)
    visual_ok = run_visual_score(data_root)
    metabolite_ok = run_metabolite(data_root)
    protein_ok = run_protein(data_root)
    rnaseq_ok = run_rnaseq(data_root)
    environment_ok = run_environment_response(data_root)
    fitness_ok = run_fitness(data_root)
    segregant_ok = run_segregant_growth(data_root)
    titer_ok = run_product_titer(data_root)
    bacterial_protein_ok = run_bacterial_protein_abundance(data_root)
    return (
        expression_ok
        and morphology_ok
        and morphology_ohnuki_ok
        and ohnuki_ok
        and visual_ok
        and metabolite_ok
        and protein_ok
        and rnaseq_ok
        and environment_ok
        and fitness_ok
        and segregant_ok
        and titer_ok
        and bacterial_protein_ok
    )


def main() -> int:
    """Verify all abstract datasets; write reports; return a shell exit code."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    all_passed = run_all(data_root)
    print("=" * 60)
    print("ALL DATASETS PASS" if all_passed else "SOME DATASETS FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
