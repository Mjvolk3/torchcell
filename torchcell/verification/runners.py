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
"""

from __future__ import annotations

import os
import os.path as osp
import pickle
from typing import Any

import lmdb

from torchcell.verification.environment_response import (
    environment_response_gene_set,
    verify_environment_response_dataset,
    verify_environment_response_dataset_streaming,
)
from torchcell.verification.expression import (
    measured_gene_universe,
    verify_expression_dataset,
)
from torchcell.verification.fitness import fitness_gene_set, verify_fitness_dataset
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


def load_records(abs_root: str) -> list[dict[str, Any]]:
    """Read every LMDB entry under ``<abs_root>/processed/lmdb``."""
    env = lmdb.open(osp.join(abs_root, "processed", "lmdb"), readonly=True, lock=False)
    records: list[dict[str, Any]] = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for _, value in cursor:
            records.append(pickle.loads(value))
    env.close()
    return records


def stream_records(abs_root: str) -> Any:
    """Yield every LMDB entry under ``<abs_root>/processed/lmdb`` one at a time.

    Memory-bounded alternative to :func:`load_records` for very large datasets (e.g.
    the 30M-record Hoepfner HIP-HOP atlas) whose full materialization would exceed RAM.
    """
    env = lmdb.open(osp.join(abs_root, "processed", "lmdb"), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                yield pickle.loads(value)
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
            method="CRI-SPA corrected colony fluorescence intensity (24h) as betaxanthin proxy",
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
    "metabolite_zelezniak2018": {
        "root": "data/torchcell/metabolite_zelezniak2018",
        "expected_count": 95,
        "reference_centered": False,  # arbitrary batch-corrected SRM signal, not centered
        "provenance": Provenance(
            source_uri="https://zenodo.org/api/records/1320289/files/metabolites_dataset.data_prep.tsv/content",
            citation_key="zelezniakMachineLearningPredicts2018",
            method="SRM-MS/MS targeted metabolomics, batch-corrected signal; per-strain mean over pooled replicates",
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
}


def run_metabolite(data_root: str) -> bool:
    """Verify metabolite datasets (L0-L4) and write reports. True if all pass."""
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
        )
        if ohya_genes:
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
}

# S288C reference gene universe (ORF + RNA-coding systematic names) for L4 containment.
SGD_GENE_FASTAS = [
    "data/sgd/genome/S288C_reference_genome_R64-4-1_20230830/"
    "orf_coding_all_R64-4-1_20230830.fasta",
    "data/sgd/genome/S288C_reference_genome_R64-4-1_20230830/"
    "rna_coding_R64-4-1_20230830.fasta",
]
# Empirically the Caudal measured-gene union is 0.943 contained in the SGD gene set (the
# remainder are accessory/novel ORFs legitimately absent from S288C); 0.90 leaves headroom.
MIN_RNASEQ_GENE_CONTAINMENT = 0.90


def _sgd_gene_set(data_root: str) -> set[str]:
    """Build the S288C systematic-name universe from the SGD ORF + RNA FASTA headers."""
    genes: set[str] = set()
    for rel in SGD_GENE_FASTAS:
        with open(osp.join(data_root, rel)) as handle:
            for line in handle:
                if line.startswith(">"):
                    genes.add(line[1:].split()[0])
    return genes


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
    """Verify RNA-seq pan-transcriptome datasets (L0-L4) and write reports. True if pass."""
    sgd_genes = _sgd_gene_set(data_root)
    all_passed = True
    for name, spec in RNASEQ_DATASETS.items():
        abs_root = osp.join(data_root, spec["root"])
        records = load_records(abs_root)
        report = verify_rnaseq_dataset(
            records,
            dataset_name=name,
            provenance=spec["provenance"],
            expected_count=spec.get("expected_count", len(records)),
        )
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
    "env_chemgen_vanacloig2022": {
        "root": "data/torchcell/env_chemgen_vanacloig2022",
        # 3647 screened ORFs (3651 barcodes - 2 all-NaN QC rows - 2 background-gene rows)
        # x 45 compound columns.
        "expected_count": 164115,
        # 3DeltaAlpha background PDR1/PDR3/SNQ2 -- excluded from the screened-ORF key.
        "background_genes": frozenset({"YGL013C", "YBL005W", "YDR011W"}),
        "provenance": Provenance(
            source_uri=(
                "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE186nnn/GSE186866/suppl/"
                "GSE186866_ChemGenomics_Raw_Counts_matrix.txt.gz"
            ),
            citation_key="vanacloig-pedrosComparativeChemicalGenomic2022",
            sha256="e29eb02769ce2180d632020dc612a7f3e14a124fc7f1e0e33f9d41b6f4e4a85a",
            method=(
                "GEO GSE186866 raw barcode counts; per-sample CPM, per-gene "
                "log2((CPM_compound_rep+1)/(CPM_pooled_control_mean+1)) mean of 3 biol. "
                "reps (SE = SD/sqrt(3)); recomputed readout, NOT the paper's edgeR logFC"
            ),
            page="FEMS Yeast Res 2022 foac036; GEO GSE186866 raw counts (Table S1 SI unscriptable)",
        ),
    },
    "env_chemgen_mota2024": {
        "root": "data/torchcell/env_chemgen_mota2024",
        # acetic 373 + butyric 416 + octanoic 484 = 1273 records: susceptible-mutant
        # rows from Tables S1/S2/S3, deduplicated per (ORF, acid) keeping the more
        # severe score (RNR4 source duplicate) minus 6 unresolvable gene tokens.
        "expected_count": 1273,
        # BY4741 Euroscarf single-deletion collection: no constant background genes.
        "background_genes": frozenset(),
        "provenance": Provenance(
            source_uri=(
                "https://static-content.springer.com/esm/"
                "art%3A10.1186%2Fs12934-024-02309-0/MediaObjects/"
                "12934_2024_2309_MOESM{1,2,3}_ESM.xlsx"
            ),
            citation_key="motaSharedMoreSpecific2024",
            method=(
                "BMC open-access supplementary spreadsheets (Additional files 1-3 = "
                "Tables S1/S2/S3); categorical spot-assay susceptibility calls "
                "(+ minor/moderate, ++ total growth inhibition) for 75 mM acetic / "
                "14 mM butyric / 0.30 mM octanoic acid, YPD pH 4.5, 30 C, 48 h; "
                "gene names -> SGD R64 ORFs via genome alias table"
            ),
            page=(
                "Microb Cell Fact 2024 s12934-024-02309-0 (PMC10903034); "
                "MOESM1 sha256=b23ad28141e70b307048fc69475aedd4e3cf880118ae9d0d806b6d9f91205e42, "
                "MOESM2 sha256=a7a1aaee1c76e52d8fe435326790c89170ab43ec96b92ef272903d8e78a1e81f, "
                "MOESM3 sha256=27f1508641ad5e7cc29ab8611739d4940da355c1dffbe9c9a908c267cdf5d455"
            ),
        ),
    },
    "env_chemgen_hoepfner2014": {
        "root": "data/torchcell/env_chemgen_hoepfner2014",
        # ENCODABLE-COMPOUNDS-ONLY build (only compounds with a released SMILES in Table S1;
        # ~92% proprietary black-box CMBxxx dropped, incl. the named-but-structureless CMB222):
        # HIP 1,753,367 (306 encodable het-CNV experiments) + HOP 1,359,513 (304 encodable
        # deletion experiments) = 3,112,880 records over 610 of 5879 sensitivity columns
        # (150 encodable compounds of 1852 deposited). One record per (ORF, sensitivity column);
        # compound_name embeds the unique experiment tag. See the loader docstring +
        # experiments/017-hoepfner-background-mutations/compound_encodability.json.
        "expected_count": 3112880,
        # ~3M records -> single-pass streaming gate (retained from the 30M full-atlas build).
        "stream": True,
        # HIP het-CNV + HOP homozygous-deletion diploid collections (BY4743): no constant
        # background genes.
        "background_genes": frozenset(),
        "provenance": Provenance(
            source_uri=(
                "https://datadryad.org/downloads/file_stream/4834608 (HIP_scores.txt); "
                "https://datadryad.org/downloads/file_stream/4834609 (HOP_scores.txt); "
                "Dryad doi:10.5061/dryad.v5m8v"
            ),
            citation_key="hoepfnerHighresolutionChemicalDissection2014",
            method=(
                "Novartis HIP-HOP chemogenomic atlas; deposited (adjusted) MADL "
                "sensitivity score = (r_L - med(r_L))/MAD(r_L) per (deletion strain x "
                "compound/concentration) at IC30 in YPD, 30 C, ~16 h, 2% DMSO; HIP = "
                "heterozygous (EngineeredCopyNumberPerturbation copy 1/2, KanMX) diploid "
                "incl. essential genes, HOP = homozygous KanMx deletion diploid; n_samples "
                "= 2 (Ad. columns) / 1 (MADL columns), technical duplicate; ORFs resolved "
                "to SGD R64 (non-R64 names dropped)"
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
        # 428573 = 242 screened ORFs x compounds, one cell per (ORF, CID-else-SID);
        # 484830 systematic-ORF datapoints collapsed (46195 multi-library-repeat cells
        # averaged), 7296 NA/NULL non-strain control rows dropped.
        "expected_count": 428573,
        # Plain haploid single-deletion collection (isogenic to BY4741): no constant
        # background genes.
        "background_genes": frozenset(),
        "provenance": Provenance(
            source_uri=(
                "https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/CSV/Data/"
                "1159001_1160000.zip (member 1159001_1160000/1159580.csv.gz)"
            ),
            citation_key="wildenhainPredictionSynergismChemicalGenetic2015",
            sha256="c461c679b63ac56045cef0f03ed9bcbb8e7f9c12146f1fc7cc8ac0c113188d64",
            method=(
                "PubChem BioAssay AID 1159580 datapoint export; normalized-OD600 growth-"
                "inhibition Z-score per (deletion strain x compound) at 20 uM in DMSO, "
                "SC + 2% glucose, 30 C, ~18 h; duplicate replicate screens (read 1/2) => "
                "n_samples=2 per screen; multi-library-repeat cells averaged (paper's "
                "'Z scores averaged for the replicate screens'); compounds -> PubChem CID"
            ),
            page=(
                "Cell Systems 2015 (doi:10.1016/j.cels.2015.12.003); ACCESSION NUMBERS "
                "PubChem BioAssay AID 1159580; inner 1159580.csv.gz sha256=c461c679... "
                "(bit-identical to PUG-REST /assay/aid/1159580/CSV, 492126 datapoints)"
            ),
        ),
    },
    "env_chemgen_auesukaree2009": {
        "root": "data/torchcell/env_chemgen_auesukaree2009",
        # ethanol 95 + methanol 55 + 1-propanol 125 + heat 178 + NaCl 42 + H2O2 30 = 525
        # sensitive-mutant records, one categorical call per (deletion, stress). Methanol
        # Table 2 lists 55 (abstract headline says 54 -- Table authoritative); all 525
        # listed gene tokens resolve to unique R64 ORFs (YGR272c -> current YGR271C-A).
        "expected_count": 525,
        # BY4742 nonessential haploid single-deletion collection: no constant background.
        "background_genes": frozenset(),
        "provenance": Provenance(
            source_uri=(
                "$DATA_ROOT/torchcell-library/"
                "auesukareeGenomewideIdentificationGenes2009/paper.pdf (library mirror; "
                "PMC2747848; publisher PMC download is JS-proof-of-work, not scriptable)"
            ),
            citation_key="auesukareeGenomewideIdentificationGenes2009",
            sha256="01b945443c0ce41642c76fd737e12b4c31cacb5f384049a5c0a7e4bf9e1eb5a1",
            method=(
                "Article Tables 1-6 extracted from the born-digital PDF text layer "
                "(pdftotext -layout; per-class parenthetical counts as self-checksums); "
                "categorical spot-assay sensitivity ('sensitive' vs parental BY4742, "
                "tolerant) for 10% ethanol / 16% methanol / 7% 1-propanol (v/v) and "
                "1 M NaCl / 5 mM H2O2 -> SmallMoleculePerturbation (typed Compound); "
                "37 C heat -> raised Environment.temperature (no perturbation, M2); "
                "YPD solid, 30 C (37 C heat), 3 days, aerobic; n_samples=3 (triplicate); "
                "gene names -> SGD R64 ORFs via genome alias table"
            ),
            page=(
                "J Appl Genet 2009 50(3):301-310 (doi:10.1007/BF03195688; PMC2747848); "
                "paper.pdf sha256=01b945443c0ce41642c76fd737e12b4c31cacb5f384049a5c0a7e4bf9e1eb5a1"
            ),
        ),
    },
    "env_chemgen_smith2006": {
        "root": "data/torchcell/env_chemgen_smith2006",
        # 4721 unique-ORF strains x 3 conditions (oleate/myristate clear-zone + acetate
        # growth) = 14163 ordinal-categorical records, one per (strain, condition). Every
        # screened row carries all three scores (no blank condition cells). Of 4770 screened
        # strains, 49 are dropped: 26 non-current/dubious systematic names + 23 alias-
        # resolutions that would collide with a directly-present R64 ORF (e.g. YOR240W ->
        # YOR239W, present as ABP140). Records store the ordinal (1-4) on
        # environment_response with a semantic category; measurement_type=categorical.
        "expected_count": 14163,
        # matalpha haploid single-deletion set (BY4742): no constant background.
        "background_genes": frozenset(),
        "provenance": Provenance(
            source_uri=(
                "$DATA_ROOT/torchcell-library/smithExpressionFunctionalProfiling2006/"
                "data/msb4100051-s1.xls (library mirror; Supplementary Table 1, fetched "
                "once from the Europe PMC supplementary bundle for PMC1681483)"
            ),
            citation_key="smithExpressionFunctionalProfiling2006",
            sha256="7048663ffa4890478724e6e371f434baccc7160e6d8250df9a777a26c6b283a4",
            method=(
                "Supplementary Table 1 (.xls, header row 23) per-strain ordinal scores for "
                "the fatty-acid clear-zone screen; one record per (deletion strain x "
                "condition). Clear-zone size on oleate (YPBO, 0.1% oleic acid w/v) and "
                "myristate (YPBM, 0.125% myristic acid w/v) scored 4=larger/3=wild type/"
                "2=less/1=small-or-absent; growth on acetate (YPBA, 2% acetate w/v) scored "
                "3=wild-type/2=moderate/1=little-no (undocumented 2.5=intermediate kept). "
                "Ordinal stored on environment_response with semantic category; "
                "measurement_type=categorical (no ordinal enum member). Each condition = "
                "SmallMoleculePerturbation (added carbon/fatty-acid species, percent w/v) on "
                "a solid-agar Environment; 30 C, 3-4 days (duration_hours=84.0), aerobic; "
                "n_samples=3 (triplicate replicate plates; quadruplicate pinning is within-"
                "plate technical); reference = parental BY4742 (category wild_type, no "
                "numeric baseline). Systematic names -> current SGD R64 ORFs (collision-"
                "aware alias resolution; 49/4770 strains dropped)"
            ),
            page=(
                "Mol Syst Biol 2006 2:2006.0009 (doi:10.1038/msb4100051; PMID 16738555; "
                "PMC1681483); msb4100051-s1.xls "
                "sha256=7048663ffa4890478724e6e371f434baccc7160e6d8250df9a777a26c6b283a4"
            ),
        ),
    },
    "crispr_magic_lian2019": {
        "root": "data/torchcell/crispr_magic_lian2019",
        # Genome-scale MAGIC CRISPRa/i/d furfural screen, per-GUIDE enrichment reprocessed
        # from raw NGS (SRA PRJNA504483). 100,493 designed guides; drop 300 random controls
        # + 16 source-corrupted-gene guides + 2,633 unresolved-gene guides (165 ncRNA/rDNA
        # genes absent from the ORF genome); of the rest, each (guide x round) with a defined
        # enrichment is a record (26,169 guide-rounds undetected; 48 guides skipped in the
        # round where they target their own integrated background) = 266,415 records. Rounds
        # are iterative in accumulating backgrounds (R1 bAID / R2 +SIZ1i / R3 +SIZ1i+NAT1a),
        # so a record's genotype is a 1-/2-/3-perturbation mixed-modality CRISPR combo; the
        # background is NOT constant across the dataset, so it is part of the genotype
        # signature (background_genes empty). L1 strain identity keys on (gene, mode, guide
        # spacer): sibling guides of one gene are distinct strains, like a TS-allele series.
        "expected_count": 266415,
        "background_genes": frozenset(),
        "provenance": Provenance(
            source_uri=(
                "NCBI SRA PRJNA504483 (raw NGS, 21 runs) + Supplementary Data 4 reference "
                "(41467_2019_13621_MOESM6_ESM.xlsx, 100,493 guides, sha256 4e3f225a...); "
                "derived enrichment table + reprocessing scripts in the library mirror "
                "$DATA_ROOT/torchcell-library/lianMultifunctionalGenomewideCRISPR2019/data/"
                " (guide_enrichment_final.tsv + PROVENANCE.md)"
            ),
            citation_key="lianMultifunctionalGenomewideCRISPR2019",
            sha256="f9af849f97a2d460c3a6d628308491ec3966c6cc2a7f6cad130848d2bad32647",
            method=(
                "The furfural per-guide enrichment is NOT a released supplement; it is "
                "reprocessed from raw reads (SRA PRJNA504483): barcode = read[27:70] (43bp "
                "activation) | read[27:71] (44bp interference/deletion), forward, exact-match "
                "to the 100,493-guide reference; CPM(+1)/library; per round per replicate "
                "log2(furfural-after / untreated-before); mean +- SD over 3 biological "
                "triplicates. Validated vs the paper's hits (PDR1i round-3 rank 1, SLX5i "
                "round-1 rank 1, SAP30d round-1 rank 2). One EnvironmentResponseExperiment per "
                "(guide x round): genotype = the library member as a Crispr"
                "Activation/Interference/Deletion perturbation (target gene + guide spacer + "
                "orthogonal effector dLbCas12a-VP/dSpCas9-RD1152/SaCas9) PLUS the round's "
                "integrated background (SIZ1 interference for r2/r3, NAT1 activation for r3, "
                "guide unspecified); environment = furfural (5/10/15 mM by round) as a "
                "SmallMoleculePerturbation on SED/G418 liquid, 30 C, aerobic; phenotype = "
                "EnvironmentResponsePhenotype measurement_type=log2_ratio, environment_response"
                "=mean log2FC, uncertainty=SD (sample_sd, n=3 -> SE=SD/sqrt(3)); reference = "
                "no-enrichment baseline (log2FC 0) in the bAID host. Common gene names -> "
                "current R64 ORFs via the genome (5,060/5,226 resolved)"
            ),
            page=(
                "Nat Commun 2019 10:5794 (doi:10.1038/s41467-019-13621-4; PMID 31857575); "
                "guide_enrichment_final.tsv "
                "sha256=f9af849f97a2d460c3a6d628308491ec3966c6cc2a7f6cad130848d2bad32647"
            ),
        ),
    },
    "crispri_mormino2022": {
        "root": "data/torchcell/crispri_mormino2022",
        # 12 individually-isolated CRISPRi strains (Table 1) each -> one categorical
        # acetic-acid-sensitivity call from the Haa1 biosensor RFP (+ -> sensitive, = ->
        # no_effect). Genome-wide enrichment is figure-only (not ingested); guides live
        # upstream in Smith 2017 (guide_sequence=None). One record per strain.
        "expected_count": 12,
        "background_genes": frozenset(),
        "provenance": Provenance(
            source_uri=(
                "$DATA_ROOT/torchcell-library/morminoIdentificationAceticAcid2022/paper.pdf "
                "(library mirror; Table 1 embedded as a literal — no SI data file released)"
            ),
            citation_key="morminoIdentificationAceticAcid2022",
            sha256="388f8e922b0b94fba3a41965035eeee0f6180a110869073a426df96b5e63746a",
            method=(
                "Table 1 'Properties of isolated strains' (12 rows) from the sha256-pinned "
                "mirror PDF. Each strain = one CrisprInterferencePerturbation (target gene, "
                "effector dCas9-Mxi1, guide_sequence=None; library from Smith 2017, BY4742) "
                "in an acetic-acid environment (50 mM, pH 3.5, 30 C, aerobic). Phenotype = "
                "EnvironmentResponsePhenotype measurement_type=categorical: RFP '+' (enhanced "
                "Haa1-biosensor signal -> more acetic-acid sensitive) -> category 'sensitive', "
                "'=' (as control) -> 'no_effect'; reference = CC23 control (no_effect). Growth "
                "column and figure-only FI/sfpHluorin/growth values are NOT ingested; "
                "n_samples None (qualitative summary call). 12 targets resolve to R64 ORFs"
            ),
            page=(
                "Microb Cell Fact 2022 21:214 (doi:10.1186/s12934-022-01938-7; PMID 36284296; "
                "PMC9571444), Table 1; paper.pdf "
                "sha256=388f8e922b0b94fba3a41965035eeee0f6180a110869073a426df96b5e63746a"
            ),
        ),
    },
    "env_chemgen_costanzo2021": {
        "root": "data/torchcell/env_chemgen_costanzo2021",
        # 4406 R64-resolved strains (3624 dma KanMX deletions + 782 tsa TS alleles after
        # dropping 23 old/merged non-R64 ORF names) x 14 conditions minus 366 empty cells
        # = 61,318 records. Essential genes are screened as allelic series (one ORF, up to
        # 18 ts alleles); the L1 uniqueness unit is the STRAIN (genotype signature).
        "expected_count": 61318,
        # SGA deletion/TS array (BY4741-derived): no constant drug-sensitized background.
        "background_genes": frozenset(),
        "provenance": Provenance(
            source_uri=(
                "https://www.science.org/doi/10.1126/science.abf8424 Data File S1 "
                "(Science SI bot-blocked/403, deposited manually to the raw mirror); "
                "Methods sourced from https://pmc.ncbi.nlm.nih.gov/articles/PMC9132594/"
            ),
            citation_key="costanzoEnvironmentalRobustnessGlobal2021",
            sha256="f6c313de416ce8cc6ae87e2020b4389bd4adeb07cdb6a438aecaf1e45e6228ad",
            method=(
                "condition-SGA single-mutant fitness: DIFFERENTIAL mutant fitness = "
                "normalized colony-size fitness in a test condition minus the matched "
                "reference condition ('the difference in colony size measured in a "
                "particular test condition versus the matched reference condition for each "
                "mutant', PMC9132594 Methods); measurement_type=differential_fitness, "
                "n_samples=3 sample_unit=screen ('an average of 3 replicate control screens "
                "conducted per each of 14 test conditions as well as the reference condition "
                "at 26 C'); 3624 KanMX deletions (SgaKanMxDeletion) + 782 TS alleles "
                "(SgaTsAllele, essential-gene allelic series) x 14 conditions; galactose -> "
                "EnvironmentPhysicalPerturbation(carbon_source), sorbitol/12 drugs -> "
                "SmallMoleculePerturbation; 26 C on Environment.temperature (M2); ORFs "
                "validated vs SGD R64 (23 old/merged names dropped)"
            ),
            page=(
                "Science 2021 372(6542):eabf8424 (doi:10.1126/science.abf8424; PMID "
                "33958448; PMC9132594); Data File S1 sheet 'Diff. Mutant fitness_Conditions'; "
                "S1 sha256=f6c313de416ce8cc6ae87e2020b4389bd4adeb07cdb6a438aecaf1e45e6228ad. "
                "Several deposited concentrations recorded verbatim + flagged as SI unit "
                "anomalies (bortezomib 1300 mM; actinomycin D 20 mM; geldanamycin 10 mM)"
            ),
        ),
    },
    "env_chemgen_hillenmeyer2008_het": {
        "root": "data/torchcell/env_chemgen_hillenmeyer2008_het",
        # HIP (heterozygous) fitness-defect log2_ratio. 726 arrays -> 514 unique
        # environments (replicates aggregated) x 5984 R64 strains minus NA-only cells.
        "expected_count": 2921078,
        "background_genes": frozenset(),
        "stream": True,
        "provenance": Provenance(
            source_uri=(
                "Internet Archive (Wayback) mirror of the Stanford supplement "
                "chemogenomics.stanford.edu/supplements/global/download/data/"
                "het.ratio_result_nm.pub; live FitDb portals are DNS-dead, Science SI 403s"
            ),
            citation_key="hillenmeyerChemicalGenomicPortrait2008",
            sha256="c0ddefaeb44c9760481dc443d1c2e82570bad4eb1a5458acd74b169bec47cb44",
            method=(
                "genome-scale HIP fitness-defect log2_ratio = log2(mean control intensity / "
                "treatment intensity), up+down tag mean; positive = fitness defect. "
                "Per-array atlas AGGREGATED to one record per (strain, environment): arrays "
                "with identical compound/concentration/generations are replicates -> mean, "
                "n_samples=n_arrays, sample SD -> derived SE. HET = "
                "EngineeredCopyNumberPerturbation (copy 1 of 2, diploid) incl. essential "
                "genes. Conditions classified per SOM Table S1: temperature "
                "(Environment.temperature, baseline 30 C), pH, amino-acid/vitamin "
                "nutrient_dropout (agent=Compound), radiation, media swap "
                "(Environment.media), carbon source (YP glycerol), else small molecule; "
                "combination = 2 small molecules. ORFs -> SGD R64 (non-R64 dropped)"
            ),
            page=(
                "Science 2008 320(5874):362-365 (doi:10.1126/science.1150021); "
                "het.ratio_result_nm.pub "
                "sha256=c0ddefaeb44c9760481dc443d1c2e82570bad4eb1a5458acd74b169bec47cb44"
            ),
        ),
    },
    "env_chemgen_hillenmeyer2008_hom": {
        "root": "data/torchcell/env_chemgen_hillenmeyer2008_hom",
        # HOP (homozygous) fitness-defect z_score. 418 arrays -> 284 unique environments
        # (replicates aggregated) x 4769 R64 strains minus NA-only cells.
        "expected_count": 1179520,
        "background_genes": frozenset(),
        "stream": True,
        "provenance": Provenance(
            source_uri=(
                "Internet Archive (Wayback) mirror of the Stanford supplement "
                "chemogenomics.stanford.edu/supplements/global/download/data/"
                "hom.z_result_nm.pub; live FitDb portals are DNS-dead, Science SI 403s"
            ),
            citation_key="hillenmeyerChemicalGenomicPortrait2008",
            sha256="0b7d5e4dad0e5b4336b1dac97fb32ef14d7d0bf9f5d1b0d5ddb382c393362b71",
            method=(
                "genome-scale HOP fitness-defect z_score = (mean control - treatment)/SD "
                "control; positive = fitness defect. Per-array atlas AGGREGATED to one "
                "record per (strain, environment): arrays with identical "
                "compound/concentration/generations are replicates -> mean, "
                "n_samples=n_arrays, sample SD -> derived SE. HOM = KanMx deletion diploid. "
                "Conditions classified per SOM Table S1 (temperature, pH, nutrient_dropout, "
                "radiation, media swap, carbon source, else small molecule). ORFs -> SGD "
                "R64 (non-R64 dropped)"
            ),
            page=(
                "Science 2008 320(5874):362-365 (doi:10.1126/science.1150021); "
                "hom.z_result_nm.pub "
                "sha256=0b7d5e4dad0e5b4336b1dac97fb32ef14d7d0bf9f5d1b0d5ddb382c393362b71"
            ),
        ),
    },
    "crispri_chemgen_smith2016": {
        "root": "data/torchcell/crispri_chemgen_smith2016",
        # Per-guide dCas9-Mxi1 CRISPRi chemical-genetic screen: 14,463 rows of Additional
        # file 10 = one record per (pool x guide x drug-condition), 0 dropped (all 20 target
        # ORFs are current R64; A/var(A) complete; every guide resolves to a spacer). 977
        # guides x 26 (drug, concentration) conditions across 5 pools. L1 strain identity
        # keys on (gene, crispr_interference, guide spacer, library_pool): sibling guides of
        # a gene are distinct strains, AND the SAME 20 bp spacer screened in both broad_tiling
        # and gene_tiling_20bp (272 such pairs) is two independent pooled measurements, kept
        # distinct by library_pool (pool-relative fitness differs up to ~8 log2 units).
        "expected_count": 14463,
        # Single-guide CRISPRi strains: no constant background.
        "background_genes": frozenset(),
        "provenance": Provenance(
            source_uri=(
                "$DATA_ROOT/torchcell-library/smithQuantitativeCRISPRInterference2016/"
                "si/si_data/13059_2016_900_MOESM10_ESM.xlsx (Additional file 10, sheet "
                "'Fitness and Effect Data') + 13059_2016_900_MOESM4_ESM.xlsx (Additional "
                "file 4, sheet 'gRNAs', guide->spacer); Springer ESM mirror"
            ),
            citation_key="smithQuantitativeCRISPRInterference2016",
            sha256="02962e51e492b0505e8595fc1c80fab5fca8a8c8f05e05969dbff18ddff71cd0",
            method=(
                "Additional file 10 (14,463 rows x 24 cols, sha256 02962e51...): one "
                "EnvironmentResponseExperiment per (pool, guide, drug, concentration) row. "
                "measurement_type=log2_ratio, environment_response = column 'A' = the "
                "ATc-induced fold change (Methods 'ATc-induced fold change': A_ijk = f_ijk+ - "
                "f_ijk-, the difference of log2 median-centred guide read-count fitness "
                "between induced +ATc, i.e. dCas9-Mxi1 CRISPRi ON, and uninduced -ATc "
                "cultures in that drug; negative = repression is a growth defect). Uncertainty "
                "= column 'var(A)' stored VERBATIM as UncertaintyType.variance with "
                "n_samples=1 (var(A) is the variance of the single released A estimate -- a "
                "Gamma read-count-resampling posterior variance s2_+ + s2_-, already inverse-"
                "variance-combined across the 8/3 replicate experiments for the 1% DMSO / "
                "20 uM fluconazole conditions), so derive_se -> SE = sqrt(var(A)). Genotype = "
                "CrisprInterferencePerturbation(target ORF, effector dCas9-Mxi1, "
                "guide_sequence = the 18/20 nt Specificity_sequence spacer joined by guide "
                "name from Additional file 4 (sha256 e5eb4e3c...), library_pool = #Pool). "
                "Environment = row Drug + Concentration as one SmallMoleculePerturbation "
                "(uM/nM; the 1% DMSO vehicle control -> 1.0 percent_v/v) on SCM-Ura liquid, "
                "30 C, aerobic; ATc dose not released so not asserted. Reference = uninduced "
                "-ATc baseline (A=0) in BY4741. 20 target ORFs all current R64 (0 dropped)"
            ),
            page=(
                "Genome Biol 2016 17:45 (doi:10.1186/s13059-016-0900-9); Additional file 10 "
                "'Fitness and Effect Data' "
                "sha256=02962e51e492b0505e8595fc1c80fab5fca8a8c8f05e05969dbff18ddff71cd0; "
                "Additional file 4 'gRNAs' "
                "sha256=e5eb4e3c7856782e36edff5ef55e680cb43fa8e9944bf8e40d7cb7b4f376c1e5"
            ),
        ),
    },
}


def run_environment_response(data_root: str) -> bool:
    """Verify environment-response datasets (L0-L4) and write reports. True if all pass."""
    sgd_genes = _sgd_gene_set(data_root)
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
            )
        else:
            records = load_records(abs_root)
            report = verify_environment_response_dataset(
                records,
                dataset_name=name,
                provenance=spec["provenance"],
                expected_count=spec.get("expected_count", len(records)),
                background_genes=background,
            )
            report.add(
                _l4_rnaseq_gene_containment(
                    sgd_genes, environment_response_gene_set(records, background)
                ).model_copy(update={"name": "gene_containment_sgd"})
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
        # 6023 raw alleles minus 30 unresolvable in current R64 (dubious/merged/removed ORFs).
        "expected_count": 5993,
        "provenance": Provenance(
            source_uri="https://doi.org/10.1038/nmeth.1534",
            citation_key="baryshnikovaQuantitativeAnalysisFitness2010",
            method=(
                "genome-scale SGA single-mutant fitness (Supplementary Data 1, "
                "S1_SMF_standard): WT-normalized so the fitness distribution mode == 1.0; "
                "uncertainty = bootstrap SE of the median (SI Note 1); n_samples=80 control "
                "screens; 6023 alleles = 4635 KanMx deletions + 1082 DAmP + 306 TS, with the "
                "raw allele id on strain_id so the TS allelic series stays distinct"
            ),
            page="Nat Methods 7:1017; Supplementary Data 1 'S1_SMF_standard_100209.txt'",
            sha256=("c8114f88c96f3b605dc5837c8958de30c34e0077558fd78f26440465a19f6b5b"),
        ),
    },
}


def run_fitness(data_root: str) -> bool:
    """Verify single-mutant fitness datasets (L0-L4) and write reports. True if all pass."""
    sgd_genes = _sgd_gene_set(data_root)
    all_passed = True
    for name, spec in FITNESS_DATASETS.items():
        abs_root = osp.join(data_root, spec["root"])
        records = load_records(abs_root)
        report = verify_fitness_dataset(
            records,
            dataset_name=name,
            provenance=spec["provenance"],
            expected_count=spec.get("expected_count", len(records)),
        )
        report.add(
            _l4_rnaseq_gene_containment(
                sgd_genes, fitness_gene_set(records)
            ).model_copy(update={"name": "gene_containment_sgd"})
        )
        out = _write_report(report, osp.join(abs_root, "preprocess"))
        print(report.summary())
        print(f"  -> wrote {out}\n")
        all_passed = all_passed and report.passed
    return all_passed


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
