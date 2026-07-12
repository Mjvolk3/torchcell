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
OHYA_EXPECTED_COUNT = 4718  # mutant records; 122 WT aggregated into the reference
OHYA_PROVENANCE = Provenance(
    source_uri="http://www.yeast.ib.k.u-tokyo.ac.jp/SCMD/ (mt4718data.tsv, wt122data.tsv)",
    citation_key="ohyaHighdimensionalLargescalePhenotyping2005a",
    method="SCMD CalMorph TSV (mt4718 mutants + wt122 wildtype)",
    page="Ohya et al. 2005 PNAS; PMID 16365294",
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
    }
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
        # HIP 16,939,418 (5807 R64 ORFs x 2956 het-CNV experiments) + HOP 13,056,820
        # (4912 R64 ORFs x 2923 deletion experiments) = 29,996,238 records. 62 HIP + 58
        # HOP old/merged ORF names not in SGD R64 dropped (343,356 records). One record
        # per (ORF, sensitivity column); compound_name embeds the unique experiment tag.
        "expected_count": 29996238,
        # 30M records -> verify via a single-pass streaming gate (materializing would
        # need ~450 GB RAM).
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


def run_all(data_root: str) -> bool:
    """Run every dataset-family verification. True only if all pass."""
    expression_ok = run_expression(data_root)
    morphology_ok = run_morphology(data_root)
    visual_ok = run_visual_score(data_root)
    metabolite_ok = run_metabolite(data_root)
    protein_ok = run_protein(data_root)
    rnaseq_ok = run_rnaseq(data_root)
    environment_ok = run_environment_response(data_root)
    return (
        expression_ok
        and morphology_ok
        and visual_ok
        and metabolite_ok
        and protein_ok
        and rnaseq_ok
        and environment_ok
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
