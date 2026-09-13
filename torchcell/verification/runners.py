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

from torchcell.data.experiment_dataset import resolve_interned
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
        # 3608 retained library rows x 41 retained compounds (45 columns minus the DMSO
        # vehicle control and the 3 compounds with no structure identifier) minus the
        # 4710 all-three-replicates-zero cells. Rules + counts:
        # preprocess/dropped_records.json.
        "expected_count": 143218,
        "background_genes": frozenset({"YGL013C", "YBL005W", "YDR011W"}),
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
                "GEO GSE186866 raw up-tag barcode counts; per-sample CPM, then per gene "
                "log2((CPM_treated_rep+1)/(CPM_control+1)) where the control is the mean "
                "of the SAME CG00n batch's inhibitor-free control columns (the paper's "
                "paired design), pooled over all 16 control columns for MMS only (the "
                "one retained compound the paper analyzed unpaired); response = mean of "
                "3 biological replicates, uncertainty = their sample SD (SE=SD/sqrt(3)); "
                "recomputed readout, NOT the paper's edgeR logFC. Anaerobic SynBase "
                "(shared MEDIA_LIBRARY object) at 30 C for 48 h / 6.5 doublings, pH 5.0 "
                "as an EnvironmentPhysicalPerturbation; compounds at their IC30 basis "
                "(Table S1 molar values unavailable) except Benomyl 10 ug/mL and MMS "
                "(fixed, unit unstated). Dropped: the DMSO vehicle column, MBO and the "
                "two QUADRIS doses (no structure identifier), 22 retired ORFs, 17 legacy "
                "ORF spellings, and 4710 all-replicates-zero cells"
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
        # ~92% proprietary black-box CMBxxx dropped, incl. the named-but-structureless
        # CMB222), MINUS the one kept compound that resolves to no structure identifier
        # (CMB409 Boromycin: 2 columns, 10,161 records). HIP 1,747,659 (305 encodable
        # het-CNV experiments) + HOP 1,355,060 (303 encodable deletion experiments) =
        # 3,102,719 records over 608 of 5,879 sensitivity columns (149 identified compounds,
        # 148 distinct InChIKeys). One record per (ORF, sensitivity column); the compound is
        # keyed by structure and the deposited study number lives on phenotype.screen_id.
        # Drop ledger: <root>/dropped_records.json. See the loader docstring +
        # experiments/017-hoepfner-background-mutations/compound_encodability.json.
        "expected_count": 3102719,
        # ~3.1M records -> single-pass streaming gate (retained from the 30M full-atlas build).
        "stream": True,
        # HIP het-CNV + HOP homozygous-deletion diploid collections (BY4743): no constant
        # background genes.
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
                "compound/concentration) at IC30 in YPD_LIQUID, 30 C, 2% DMSO vehicle; "
                "encodable compounds only (released Table S1 SMILES), and every kept "
                "compound carries a structure identifier -- curated "
                "compound_identity_table row first, else an InChIKey derived from the "
                "released SMILES with RDKit; the one compound resolving to no identifier "
                "(CMB409 Boromycin) is DROPPED (2 columns, 10,161 records). HIP = "
                "heterozygous (EngineeredCopyNumberPerturbation copy 1/2, KanMX) diploid "
                "incl. essential genes at ~20 generations over four 16 h passages "
                "(duration_hours a ProvenanceGap), HOP = homozygous KanMx deletion diploid "
                "at 16 h / ~5 generations; n_samples = 2 (Ad. columns) / 1 (MADL columns), "
                "technical duplicate, reference n_samples = 4 (conservative lower end of "
                "the paper's 4-8 control replicates); screen_id = the deposited study "
                "number, which keeps same-compound same-dose columns from two screens "
                "L1-distinct; assay_type = pooled_competitive_growth_barcode; the 157 "
                "Table S5 background-mutation HIP strains are KEPT and flagged in "
                "<root>/table_s5_affected_strains.json; ORFs resolved to SGD R64 (non-R64 "
                "names dropped)"
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
        # 428573 (ORF, compound-identity) cells minus the 367 cells of the 5 SID-only
        # compounds. Rules + counts: preprocess/dropped_records.json.
        "expected_count": 428206,
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
                "28b48a3ba0724ee59b0 asserted before the member is read); released "
                "normalized-OD600 z_score per (deletion strain x compound) at 20 uM in "
                "DMSO, SC + 2% glucose (shared MEDIA_LIBRARY object), 30 C, ~18 h. A "
                "cell's contributing SCREENS are its distinct released datapoints (the "
                "export re-emits the same datapoint under two gene-symbol spellings); "
                "response = mean over screens, n_samples = screens (sample_unit=screen), "
                "uncertainty = sample SD across screens or a typed ProvenanceGap at n=1. "
                "PUBCHEM_ACTIVITY_OUTCOME + bioactivity map onto ResponseCategory "
                "(Inactive->no_change, Active->sensitive/resistant, Inconclusive and "
                "disagreeing screens->not_determined) with the source words verbatim in "
                "category_label. Dropped: the 367 cells of the 5 SID-only compounds"
            ),
            page=(
                "Cell Systems 2015 (doi:10.1016/j.cels.2015.12.003); AID description "
                "sha256=23c5f8c56af94786cfe8e22c93fdde0b719ca2165975305944557ab39087b0e4; "
                "paper.md sha256=f46409eb8f23412c9c1015d0f8f5bb581bfddfe2796d319d407585e23c757ac2"
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
        # kept records after the identifier rule and the two unrepresentable
        # conditions; rules + counts in preprocess/dropped_records.json.
        "expected_count": 2698797,
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
            method="genome-scale HIP fitness-defect log2_ratio = log2(mean control intensity / treatment intensity), up+down tag mean; positive = fitness defect. HET = EngineeredCopyNumberPerturbation (copy 1 of 2, diploid) incl. essential genes. One record per (strain, environment, CONTROL SET): the SOM defines the score against a matched no-drug control set (same pool, generation count and scanner), so the control-set id is stored on phenotype.screen_id and one reference is emitted per control set with that set's control-array count as n_samples. Within a group each array contributes one value (the mean over that ORF's construction rows, which are DIFFERENT strains, not replicates); the record is the across-array mean with the sample SD, n_samples = arrays. Conditions per SOM Table S1: temperature on Environment.temperature; pH; nutrient drop-outs as DERIVED SC media (HILLENMEYER_DROPOUT_MEDIA); media swaps to SD / SC / YP_GLYCEROL_LIQUID; '<compound> irradiated' as compound + radiation; cond2 appended in every branch; base medium YPD_LIQUID. Temperature is NOT defaulted -- the SOM defers the growth protocol to Pierce 2006 (not mirrored), so non-temperature conditions carry a ProvenanceGap(deferred_pending_source_review) on it. duration_generations stores the MAGNITUDE (the sign is the no-recovery protocol flag, preserved in screen_id). Molar doses canonicalized exactly (Decimal) so '1.5 m' and '1.5e+06 um' are one environment. DROPPED: 57 environments / 68 arrays -- 56 naming a compound with no structure identifier, plus the '37c, 45c' heat-shock cycle. ORFs go through resolve_gene_name (renamed mapped, retired/non-gene dropped to preprocess/dropped_genes.json)",
            page=(
                "Science 2008 320:362 (doi:10.1126/science.1150021); SOM Table S1; raw mirror "
                "manifest.json pinning every file; het.ratio_result_nm.pub sha256=c0ddefaeb44c9760481dc443d1c2e82570bad4eb1a5458acd74b169bec47cb44"
            ),
        ),
    },
    "env_chemgen_hillenmeyer2008_hom": {
        "root": "data/torchcell/env_chemgen_hillenmeyer2008_hom",
        # kept records after the identifier rule and the two unrepresentable
        # conditions; rules + counts in preprocess/dropped_records.json.
        "expected_count": 1088620,
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
            method="genome-scale HOP fitness-defect z_score = (mean control - treatment)/SD control; positive = fitness defect. HOM = KanMx deletion in a diploid. One record per (strain, environment, CONTROL SET): the SOM defines the score against a matched no-drug control set (same pool, generation count and scanner), so the control-set id is stored on phenotype.screen_id and one reference is emitted per control set with that set's control-array count as n_samples -- the n of the z-score's denominator. Within a group each array contributes one value (the mean over that ORF's construction rows, which are DIFFERENT strains, not replicates); the record is the across-array mean with the sample SD, n_samples = arrays. Conditions per SOM Table S1: temperature on Environment.temperature; pH (with cond2 parsed, so pH7.5+FK506 is no longer merged into plain pH7.5); the 15 nutrient drop-outs as DERIVED SC media plus SC for the drop-out control; media swaps to SD / SC / YP_GLYCEROL_LIQUID; 'no drug irradiated' as radiation alone and angelicin / psoralen irradiated as compound + radiation; base medium YPD_LIQUID. Temperature is NOT defaulted (ProvenanceGap deferring to Pierce 2006). duration_generations stores the MAGNITUDE (the sign is the no-recovery protocol flag, preserved in screen_id). Molar doses canonicalized exactly (Decimal). DROPPED: 32 environments / 37 arrays -- 30 naming a compound with no structure identifier, plus the two 'minimal media:400:um' arrays whose agent is named nowhere. ORFs go through resolve_gene_name (renamed mapped, retired/non-gene dropped to preprocess/dropped_genes.json)",
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
    """Verify single-mutant fitness datasets (L0-L4) and write reports. True if all pass."""
    sgd_genes = _sgd_gene_set(data_root)
    resolve_gene_name = _genome(data_root).resolve_gene_name
    all_passed = True
    for name, spec in FITNESS_DATASETS.items():
        abs_root = osp.join(data_root, spec["root"])
        records = load_records(abs_root)
        report = verify_fitness_dataset(
            records,
            dataset_name=name,
            provenance=spec["provenance"],
            expected_count=spec.get("expected_count", len(records)),
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
# Segregant growth datasets (haplotype-mosaic genotypes)
# --------------------------------------------------------------------------- #
SEGREGANT_GROWTH_DATASETS: dict[str, dict[str, Any]] = {
    "bloom2019": {
        "root": "data/torchcell/bloom2019",
        "raw_mirror": "torchcell-raw/bloomRareVariantsContribute2019",
        "assembly_index": (
            "torchcell-library/peterGenomeEvolution10112018/data/"
            "1011Assemblies.tar.gz.member_index.tsv"
        ),
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
            assembly_index_path=osp.join(data_root, spec["assembly_index"]),
            genome=genome,
            sgd_genes=sgd_genes,
            gene_set=segregant_gene_set(osp.join(abs_root, "preprocess")),
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
    segregant_ok = run_segregant_growth(data_root)
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
