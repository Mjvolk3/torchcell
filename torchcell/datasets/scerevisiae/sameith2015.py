# torchcell/datasets/scerevisiae/sameith2015
# [[torchcell.datasets.scerevisiae.sameith2015]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/sameith2015
# Test file: tests/torchcell/datasets/scerevisiae/test_sameith2015.py
"""Sameith 2015 GSTF single/double deletion-mutant microarray expression datasets."""

import logging
import math
import os
import os.path as osp
import pickle
import re
import urllib.request
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import GEOparse
import lmdb
import numpy as np
import pandas as pd
from sortedcontainers import SortedDict
from tqdm import tqdm

from torchcell.data import ExperimentDataset, Organism, post_process
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
    GenePerturbationType,
    Genotype,
    Media,
    MicroarrayExpressionExperiment,
    MicroarrayExpressionExperimentReference,
    MicroarrayExpressionPhenotype,
    Publication,
    ReferenceGenome,
    SgaKanMxDeletionPerturbation,
    SgaNatMxDeletionPerturbation,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ============================================================================
# One record per array: what a replicate is in GSE42536 (#1174, #1172, #1178)
# ============================================================================
#
# ONE RECORD PER GEO ARRAY, AND ONE ARRAY IS ONE CULTURE. Every array GSE42536 releases
# is its own record: value verbatim from that array, ``n_replicates = 1`` per gene, no
# SE and no variance (one culture has no dispersion), ``replicate_id`` = the array's
# GEO sample accession (``GSM1044376``). SM: 144 arrays over 82 single mutants (62 x 2,
# 20 x 1); DM: 143 arrays over 72 pairs (71 x 2, 1 x 1). Before 2026-11-24 the loaders
# stored one record per mutant holding the mean, SD / sqrt(n) and variance over its
# arrays (82 and 72 records); that mean is now a modeling-time derivation over the
# records of one genotype.
#
# Sources (sha256 of each file read; line numbers are lines of the decompressed file
# or of the OCR markdown):
#   Sameith 2015 paper.md (library mirror sameithHighresolutionGeneExpression2015,
#   sha256 7064a1ebcf6dddf51ab23658b92457ded9b56e2eecb27b88694fd852e16c9161)
#   L33: "With the exception of single mutants that behave like WT, each mutant was
#         grown and profiled four times from two independent cultures."
#   L119 (Methods, "Gene expression profiling"): "Full details of all gene expression
#         profiling procedures have been described before [46]." and "Single mutants
#         differing from WT as well as all double mutants were profiled another two
#         times from an independently inoculated culture. The reported FC is then the
#         average of four replicate mutant expression profiles versus the average of
#         all WTs."
#   L115: "Double mutants were generated in duplicate" and "Of the selected 90 double
#         mutants, 18 failed quality control criteria even after remake. They were
#         therefore excluded from further analyses." (the 18 are not in GSE42536)
#   Reference [46] is Kemmeren et al. 2014, Cell (L232); its paper.md (library mirror
#   kemmerenLargeScaleGeneticPerturbations2014, sha256
#   299eac4b07fc977a89c24ce190d05cfd182342a9f0d56a05e7e477d72e4ad08d) L307:
#         "Two independent cultures were hybridized on two separate microarrays. For
#         the first hybridization the Cy5 (red) labeled cRNA from the deletion mutant
#         is hybridized together with the Cy3 (green) labeled cRNA from the common
#         reference. For the replicate hybridization from the independent cultures,
#         the labels are swapped. Each gene is represented twice on the microarray,
#         resulting in four measurements per mutant."
#   GSE42536_family.soft.gz (sha256 below) line 14, ``!Series_overall_design``: "Two
#         independent cultures were hybridized on two separate microarrays. [...] For
#         the replicate hybridization, the labels are swapped. Each gene is represented
#         twice on the microarray, resulting in four measurements per mutant."
#
# So the "four measurements" are 2 arrays (one per independent culture, dye orientation
# swapped BETWEEN the two cultures) x 2 spots per gene, never 4 arrays: GSE42536 holds
# at most 2 arrays per mutant. A two-array pair is two biological replicates and so two
# records; it is not one culture dye-swapped. The 20 one-array single mutants are
# exactly the Additional file 1 "Single mutants - info" rows whose ``comments`` reads
# "2 replicates" (L33's WT-like singles, profiled from one culture: 1 array x 2 spots),
# and the one-array pair cin5+yap6 is the one "Double mutants - info" row with that
# comment. No released array is dropped and the authors withdrew none.
#
# THE TECHNICAL LEVEL IS THE SPOT, COMBINED INSIDE THE RECORD. GPL11232 prints 6,157
# ORFs twice and 12 four times (``ORF`` column, 15,552 rows). Kemmeren L390 obtains its
# statistics from limma "(Smyth et al., 2005)", which its reference list (L481) names
# "Use of within-array replicate spots for assessing differential expression in
# microarray experiments": the spots are within-array replicates of one culture. A
# record's per-gene value combines all of the gene's spots on its array: ``expression``
# and the reference ``expression`` are the geometric means of the spots' mutant and
# refpool ``Signal Norm`` values, and ``expression_log2_ratio`` is the mean of the
# spots' log2(mutant / refpool), which equals log2 of the ratio of the two geometric
# means. Until 2026-11-24 the loaders kept only the last spot of each gene (the probe
# map overwrote the first). ``MicroarrayExpressionPhenotype`` has no field for the
# technical count (2, or 4 for the 12 ORFs) or the array's dye orientation, so the
# record cannot state either; both are fixed by the platform and by
# ``source_name_ch1``, and adding the fields is an owner decision (schema change).
#
# The reference of a record is the refpool channel of that same array (spots
# combined), with ``n_replicates = 1`` per gene and ``replicate_id = None``: GEO line 14
# "RNA isolated from a large amount of wt yeast from a single culture was used as a
# common reference", so the refpool is one RNA batch measured on every array, the
# denominator of that array's ratio, not a set of control replicates. GSE42536 holds no
# wild-type-vs-refpool array (every one of the 287 titles names a deletion), so there
# is no control replicate to store as a record.

# PubMed ID of Sameith et al. 2015, BMC Biology (DOI 10.1186/s12915-015-0222-5).
# Source: the GEO series record this loader reads, GSE42536_family.soft.gz (sha256
# f8842af9768043fde560fa7cab737cde2fca0463c4b655511d54678d7b85cc7d, the file in the
# dev tree data/torchcell/{sm,dm}_microarray_sameith2015/raw/), decompressed line 12:
#     "!Series_pubmed_id = 26700642"
# Checked 2026-10-02. The loaders stored 26687005 until then, the PubMed ID of an
# unrelated eLife 2015 paper (#478).
SAMEITH2015_PUBMED_ID = "26700642"

# ============================================================================
# Gene names in GEO sample titles: the release's own symbol table
# ============================================================================
# A GEO sample title names a deleted gene by a lower-case token ("dot6-del-1-a",
# "rpn4-del+ydr026c-del"). The release itself pairs every gene symbol it uses with a
# systematic name: Additional file 1 (``12915_2015_222_MOESM1_ESM.xlsx``, in the raw
# dir) lists them in three sheets, "List of 215 GSTFs" (``systematic name``,
# ``gene symbol``), "Single mutants - info" (the same two columns) and "Double mutants -
# info" (``GSTF1,           systematic name`` / ``GSTF1,           gene symbol`` and the
# GSTF2 pair, with that spacing). The workbook writes systematic names as ``Yil130W``
# and symbols as ``Asg1``, so both sides are upper-cased. That table is the ONLY route
# from a title token to an ORF: no genome, alias table or pin is consulted (the genome
# is a separate layer and gates nothing). Measured 2026-10-12 over all 287 titles of
# GSE42536 (143 with "+", 144 without, none wildtype): every title resolves through the
# workbook (396 symbol tokens) or as an ORF-shaped token (34) to exactly the gene set the
# KG 3.0 build stored, including SUT2 -> YPR009W and GAT1 -> YFL021W, the two tokens that
# are ambiguous in R64 and that the workbook names directly ("Ypr009W | Sut2",
# "Yfl021W | Gat1"). The other title tokens (DEL, MATA, REP) are not genes.
WORKBOOK_FILE_NAME = "12915_2015_222_MOESM1_ESM.xlsx"
_WORKBOOK_NAME_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("List of 215 GSTFs", "systematic name", "gene symbol"),
    ("Single mutants - info", "systematic name", "gene symbol"),
    (
        "Double mutants - info",
        "GSTF1,           systematic name",
        "GSTF1,           gene symbol",
    ),
    (
        "Double mutants - info",
        "GSTF2,           systematic name",
        "GSTF2,           gene symbol",
    ),
)
_SYSTEMATIC_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$")


def release_symbol_map(path: str) -> dict[str, str]:
    """Upper-cased gene symbol -> upper-cased systematic name, from Additional file 1.

    Reads every (systematic name, gene symbol) pair of the three sheets of the
    release's own workbook; a row with either cell blank names no pair. A symbol the
    workbook pairs with two different ORFs raises ``ValueError``, since the release
    would then not say which gene a title means. A symbol equal to its ORF (the workbook
    writes ``Ypr015C | Ypr015C`` for a gene with no symbol) maps to itself.
    """
    symbol_map: dict[str, str] = {}
    for sheet, orf_column, symbol_column in _WORKBOOK_NAME_COLUMNS:
        table = pd.read_excel(path, sheet_name=sheet)
        for orf_cell, symbol_cell in zip(
            table[orf_column], table[symbol_column], strict=True
        ):
            if pd.isna(orf_cell) or pd.isna(symbol_cell):
                continue
            orf = str(orf_cell).strip().upper()
            symbol = str(symbol_cell).strip().upper()
            if symbol in symbol_map and symbol_map[symbol] != orf:
                raise ValueError(
                    f"{WORKBOOK_FILE_NAME} pairs the symbol {symbol} with both "
                    f"{symbol_map[symbol]} and {orf} (sheet {sheet!r})"
                )
            symbol_map[symbol] = orf
    return symbol_map


def release_gene_symbols(symbol_map: dict[str, str]) -> dict[str, str]:
    """Systematic name -> the release's gene symbol for it, from ``release_symbol_map``.

    An ORF whose only workbook symbol is the ORF itself has no symbol and is absent.
    The workbook names YML081W ``Tda9`` in "List of 215 GSTFs" and "Single mutants -
    info" but ``Yml081W`` in "Double mutants - info"; the ORF-as-symbol entry is not a
    symbol, so YML081W -> TDA9. Two different non-ORF symbols for one ORF raise
    ``ValueError``.
    """
    symbols: dict[str, str] = {}
    for symbol, orf in sorted(symbol_map.items()):
        if symbol == orf:
            continue
        if orf in symbols and symbols[orf] != symbol:
            raise ValueError(
                f"{WORKBOOK_FILE_NAME} names {orf} both {symbols[orf]} and {symbol}"
            )
        symbols[orf] = symbol
    return symbols


_MUTANT_AND_REFPOOL_COLUMNS = ("Signal Norm_Cy5", "Signal Norm_Cy3")


def combine_spots(
    accession: str, gene: str, spots: list[tuple[float, float]]
) -> tuple[float, float, float]:
    """(mutant, refpool, log2 ratio) of one gene on one array, from all its spots.

    ``spots`` holds one (mutant signal, refpool signal) pair per spot of the gene.
    The channel values are the geometric means over the spots and the log2 ratio is
    the mean of the spots' log2(mutant / refpool), so it equals log2 of the ratio of
    the two geometric means. One spot returns its own values unchanged. A signal that
    is not a positive number has no log ratio, so it raises instead of being skipped:
    GSE42536 holds none (all 4,463,424 signals of the 287 arrays are positive).
    """
    for mutant, refpool in spots:
        if not (mutant > 0 and refpool > 0):
            raise ValueError(
                f"{accession} {gene}: spot signals mutant={mutant!r}, "
                f"refpool={refpool!r} are not both positive numbers"
            )
    n = len(spots)
    mutant_value = math.prod(m for m, _ in spots) ** (1 / n)
    refpool_value = math.prod(r for _, r in spots) ** (1 / n)
    log2_ratio = sum(float(np.log2(m / r)) for m, r in spots) / n
    return mutant_value, refpool_value, log2_ratio


def extract_array_expression(
    gsm: Any, probe_to_gene_map: dict[str, str] | None
) -> tuple[SortedDict, SortedDict, SortedDict]:
    """(mutant, refpool, log2 ratio) per gene of one GEO array, spots combined.

    The mutant channel is Cy5 unless ``source_name_ch1`` names the refpool, then Cy3
    (``label_ch1 = Cy5`` on all 287 arrays of GSE42536). The log2 ratio is recomputed
    from the two ``Signal Norm`` columns, never read from ``VALUE``: as numbers, VALUE
    is log2(Cy5/Cy3) on 217 arrays and log2(Cy3/Cy5) on 70, and on those 70 the
    ``#VALUE`` header says the opposite of the numbers (#72, #481), while the signals
    are orientation-proof. Every probe the platform map names a gene contributes one
    spot to that gene (``combine_spots``). A GSM without a table, without ``ID_REF`` or
    either signal column, or a missing probe map yields three empty maps.
    """
    mutant_data = SortedDict()
    refpool_data = SortedDict()
    log2_ratio_data = SortedDict()
    table = getattr(gsm, "table", None)
    if table is None or "ID_REF" not in table.columns or not probe_to_gene_map:
        return mutant_data, refpool_data, log2_ratio_data
    if not set(_MUTANT_AND_REFPOOL_COLUMNS) <= set(table.columns):
        log.warning(f"Missing required columns. Available: {list(table.columns)}")
        return mutant_data, refpool_data, log2_ratio_data
    metadata = getattr(gsm, "metadata", {})
    source_ch1 = metadata.get("source_name_ch1", [""])[0]
    accession = metadata.get("geo_accession", [getattr(gsm, "name", "")])[0]
    cy5, cy3 = _MUTANT_AND_REFPOOL_COLUMNS
    mutant_column, refpool_column = (
        (cy3, cy5) if "refpool" in source_ch1.lower() else (cy5, cy3)
    )
    spots: dict[str, list[tuple[float, float]]] = {}
    for probe, mutant, refpool in zip(
        table["ID_REF"], table[mutant_column], table[refpool_column], strict=True
    ):
        gene = probe_to_gene_map.get(str(int(probe)))
        if gene is None:
            continue
        spots.setdefault(gene, []).append((float(mutant), float(refpool)))
    for gene, gene_spots in spots.items():
        mutant_data[gene], refpool_data[gene], log2_ratio_data[gene] = combine_spots(
            accession, gene, gene_spots
        )
    return mutant_data, refpool_data, log2_ratio_data


def title_token_to_systematic(symbol_map: dict[str, str], token: str) -> str | None:
    """The ORF a title token names through the release, or None when it names no gene.

    GEO titles are lower case, so the token is upper-cased first. A systematic-shaped
    token is taken verbatim (upper-cased); otherwise the token is looked up in the
    workbook's symbol map. A token that is neither (``DEL``, ``MATA``, a replicate
    suffix) is not a gene token.
    """
    upper = token.upper()
    if _SYSTEMATIC_RE.match(upper):
        return upper
    return symbol_map.get(upper)


@register_dataset
class SmMicroarraySameith2015Dataset(ExperimentDataset):
    """Single mutant GSTF expression profiling from Sameith et al. 2015.

    Microarray gene expression data for 82 single mutant yeast strains with
    deletions in general stress transcription factors (GSTFs), one record per GEO
    array (one culture): 144 records, 62 mutants with 2 arrays and 20 with 1 (see the
    module comment). All single mutants
    are from the yeast deletion library (BY4742, MATalpha, KanMX marker). Title tokens
    resolve to ORFs only through the release's Additional file 1
    (``release_symbol_map``), and ``perturbed_gene_name`` is the workbook's symbol.

    Data source: GEO accession GSE42536
    Paper: Sameith et al. (2015) BMC Biology
    DOI: 10.1186/s12915-015-0222-5
    """

    organism = Organism.scerevisiae

    geo_accession = "GSE42536"

    def __init__(
        self,
        root: str = "data/torchcell/sm_microarray_sameith2015",
        io_workers: int = 0,
        process_workers: int = 0,
        batch_size: int = 10,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ):
        """Initialize the dataset; gene names come from the release's own workbook."""
        self.process_workers = process_workers
        self.batch_size = batch_size

        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment model produced by this dataset."""
        return MicroarrayExpressionExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference model produced by this dataset."""
        return MicroarrayExpressionExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Raw GEO SOFT family file required before processing."""
        return [f"{self.geo_accession}_family.soft.gz"]

    def download(self) -> None:
        """Download single mutant microarray expression data from GEO."""
        log.info(f"Downloading GEO dataset {self.geo_accession}...")

        try:
            gse = GEOparse.get_GEO(
                geo=self.geo_accession, destdir=self.raw_dir, silent=False
            )
            log.info(f"Successfully downloaded {self.geo_accession}")

            # Save the parsed GEO object for use in process()
            geo_pkl_path = osp.join(self.raw_dir, f"{self.geo_accession}.pkl")
            with open(geo_pkl_path, "wb") as f:
                pickle.dump(gse, f)
            log.info(f"Saved GEO object to {geo_pkl_path}")

        except Exception as e:
            log.error(f"Failed to download GEO dataset: {e}")
            raise RuntimeError("GEO download failed")

        # Download supplementary Excel file
        suppl_url = "https://static-content.springer.com/esm/art%3A10.1186%2Fs12915-015-0222-5/MediaObjects/12915_2015_222_MOESM1_ESM.xlsx"
        suppl_path = osp.join(self.raw_dir, "12915_2015_222_MOESM1_ESM.xlsx")

        try:
            log.info("Downloading supplementary file...")
            urllib.request.urlretrieve(suppl_url, suppl_path)
            log.info(f"Downloaded supplementary file to {suppl_path}")
        except Exception as e:
            log.error(f"Failed to download supplementary file: {e}")
            raise RuntimeError("Failed to download supplementary data")

    def _load_authoritative_single_mutants(self) -> dict[str, dict[str, Any]]:
        """Load authoritative single mutants from supplementary Excel file."""
        suppl_path = osp.join(self.raw_dir, "12915_2015_222_MOESM1_ESM.xlsx")
        df = pd.read_excel(suppl_path, sheet_name="Single mutants - info")
        valid_df = df[df["systematic name"].notna()].copy()

        log.info(f"Loaded {len(valid_df)} single mutants from supplementary file")

        single_mutants: dict[str, dict[str, Any]] = {}
        for _, row in valid_df.iterrows():
            systematic_name = row["systematic name"].strip().upper()
            single_mutants[systematic_name] = {
                "systematic_name": systematic_name,
                "gene_symbol": row.get("gene symbol", ""),
                "strain": "BY4742",  # All from deletion library
            }

        return single_mutants

    @post_process
    def process(self) -> None:
        """Parse GEO expression data into single-mutant experiments and write LMDB."""
        # Title tokens resolve through the release's own workbook only.
        self._load_release_names()

        # Load the GEO object
        geo_pkl_path = osp.join(self.raw_dir, f"{self.geo_accession}.pkl")
        if osp.exists(geo_pkl_path):
            with open(geo_pkl_path, "rb") as f:
                gse = pickle.load(f)
        else:
            # Re-download if pickle doesn't exist
            gse = GEOparse.get_GEO(
                geo=self.geo_accession, destdir=self.raw_dir, silent=False
            )

        # Load authoritative single mutants from supplementary file
        self.single_mutants = self._load_authoritative_single_mutants()

        log.info("Processing GEO samples for single mutants...")

        # Extract platform annotation for probe-to-gene mapping
        probe_to_gene_map = self._extract_probe_to_gene_mapping(gse)
        log.info(f"Found {len(probe_to_gene_map)} probe-to-gene mappings")

        # Parse samples and extract metadata
        samples_data = []
        single_mutant_samples: dict[str, list[Any]] = {}  # Group by gene name
        wt_samples = []
        multi_gene_titles = 0  # Drop rule "double_deletion_title" (#479)

        for gsm_name, gsm in gse.gsms.items():
            sample_info = {
                "geo_accession": gsm_name,
                "title": gsm.metadata.get("title", [""])[0],
            }

            # Extract gene names from title
            gene_names = self._extract_gene_names_from_title(sample_info["title"])

            # Determine sample type
            is_wildtype = (
                "wt" in sample_info["title"].lower()
                or "wildtype" in sample_info["title"].lower()
            )
            # A single-deletion array names exactly one gene and joins none with "+".
            # GSE42536 also holds the double-deletion arrays ("rpn4-del+ydr026c-del");
            # they belong to DmMicroarraySameith2015Dataset and are dropped here under
            # the rule "double_deletion_title" (#479).
            is_double_title = (
                "+" in sample_info["title"] or len(gene_names) > 1
            ) and not is_wildtype
            is_single = not is_double_title and len(gene_names) == 1
            if is_double_title:
                multi_gene_titles += 1

            sample_info["is_wildtype"] = is_wildtype
            sample_info["is_single_mutant"] = is_single
            sample_info["gene_names"] = gene_names
            sample_info["gsm_object"] = gsm

            if is_wildtype:
                wt_samples.append(gsm)
            elif is_single:
                gene = gene_names[0]
                if gene not in single_mutant_samples:
                    single_mutant_samples[gene] = []
                single_mutant_samples[gene].append(sample_info)

            samples_data.append(sample_info)

        log.info(f"Found {len(single_mutant_samples)} unique single mutant genes")
        log.info(f"Found {len(wt_samples)} wildtype samples")
        self.dropped_double_deletion_titles = multi_gene_titles
        log.info(
            f"Dropped {multi_gene_titles} arrays under rule double_deletion_title "
            "(title joins genes with '+' or names more than one gene)"
        )

        # Calculate wildtype reference expression
        if wt_samples:
            self.wt_reference_expression, self.wt_std_expression = (
                self._calculate_wt_reference_with_std(wt_samples, probe_to_gene_map)
            )
        else:
            log.warning("No wildtype samples found, using synthetic reference")
            self.wt_reference_expression = SortedDict()
            self.wt_std_expression = SortedDict()

        # Group single mutant samples by gene using authoritative list
        single_mutant_groups = self._group_single_mutant_replicates(
            single_mutant_samples
        )
        log.info(
            f"Grouped into {len(single_mutant_groups)} replicate groups for processing"
        )

        # Process single mutants sequentially
        log.info(f"Processing {len(single_mutant_groups)} single mutant groups...")
        self._process_sequential(single_mutant_groups, probe_to_gene_map)

        # Save preprocessed data - filter for single mutants only
        os.makedirs(self.preprocess_dir, exist_ok=True)
        samples_df = pd.DataFrame(samples_data)
        samples_df = samples_df.drop("gsm_object", axis=1)
        # Filter to only include single mutant samples
        single_mutant_df = samples_df[samples_df["is_single_mutant"]]
        single_mutant_df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)
        log.info(
            f"Saved {len(single_mutant_df)} single mutant samples to preprocess data.csv"
        )

        log.info(f"Processed {len(single_mutant_groups)} single mutant experiments")

    def _group_single_mutant_replicates(
        self, single_mutant_samples: dict[str, list[Any]]
    ) -> list[Any]:
        """Group the arrays of each single mutant (one array per culture) by gene.

        A group orders the records of one genotype; every array of a group is its own
        record. A gene outside the SI's single-mutant list is kept.

        Args:
            single_mutant_samples: Dict mapping gene -> list of sample_info dicts

        Returns:
            list: List of replicate groups
        """
        groups = []
        matched_count = 0
        unmatched_count = 0

        for gene, sample_list in single_mutant_samples.items():
            if gene in self.single_mutants:
                groups.append(sample_list)
                matched_count += len(sample_list)
            else:
                log.warning(f"Gene {gene} not in authoritative list")
                unmatched_count += len(sample_list)
                groups.append(sample_list)  # Still process

        log.info(f"Matched {matched_count} samples, {unmatched_count} unmatched")
        log.info(f"Found {len(groups)} unique genotypes")

        return groups

    def _process_sequential(
        self, single_mutant_groups: list[Any], probe_to_gene_map: dict[str, str]
    ) -> None:
        """Write one record per array of every single-mutant group, in group order.

        A group not flagged single or naming no gene writes nothing. An array that
        yields no expression value raises: every released array is a record, so none
        is skipped.
        """
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(5e12))

        idx = 0
        with env.begin(write=True) as txn:
            for replicate_group in tqdm(single_mutant_groups):
                sample_info = replicate_group[0]
                if (
                    not sample_info["is_single_mutant"]
                    or len(sample_info["gene_names"]) < 1
                ):
                    continue
                for array_info in replicate_group:
                    accession = array_info["geo_accession"]
                    mutant_data, refpool_data, log2_ratio_data = (
                        self._extract_expression_from_gsm(
                            array_info["gsm_object"], probe_to_gene_map
                        )
                    )
                    if not mutant_data:
                        raise RuntimeError(
                            f"array {accession} ({array_info['gene_names']}) yields "
                            "no expression value; a released array is never skipped"
                        )
                    experiment, reference, publication = (
                        self.create_single_mutant_expression_experiment(
                            self.name,
                            array_info,
                            mutant_data,
                            refpool_data,
                            log2_ratio_data,
                            replicate_id=accession,
                            gene_symbols=self.gene_symbols,
                        )
                    )
                    serialized_data = pickle.dumps(
                        {
                            "experiment": experiment.model_dump(),
                            "reference": reference.model_dump(),
                            "publication": publication.model_dump(),
                        }
                    )
                    txn.put(f"{idx}".encode(), serialized_data)
                    idx += 1

        env.close()
        log.info(f"Wrote {idx} single mutant array records to LMDB")

    def _extract_gene_names_from_title(self, title: str) -> list[str]:
        """Every gene a sample title names: systematic names, then workbook symbols.

        The list is not truncated. GSE42536 holds single- and double-deletion arrays in
        one series, and a double-deletion title joins two genes with ``+``
        (``rpn4-del+ydr026c-del``); returning only the first gene made every such
        array look like a single mutant and folded all 143 double-deletion arrays into
        45 of the 82 single-mutant records (#479). ``process()`` keeps an array as a
        single mutant only when its title names exactly one gene and has no ``+``.
        """
        gene_names: list[str] = []

        # Look for systematic names (e.g., YAL001C or YBR089C-A)
        # Match the full systematic name, including optional suffix like -A
        systematic_pattern = r"\b(Y[A-P][LR]\d{3}[WC](?:-[A-Z])?)\b"
        matches = re.findall(systematic_pattern, title.upper())

        # Validate each match
        for match in matches:
            if self._is_valid_systematic_name(match) and match not in gene_names:
                gene_names.append(match)

        # Common names too, also when a systematic name was found: a double-deletion
        # title can mix the two ("rpn4-del+ydr026c-del").
        common_pattern = r"\b([A-Z][A-Z0-9]{2,})\b"
        for gene in re.findall(common_pattern, title.upper()):
            # Skip common non-gene words
            if gene in ["DEL", "WT", "WILDTYPE", "MUTANT", "SINGLE", "DOUBLE"]:
                continue
            systematic = self._convert_to_systematic(gene)
            if (
                systematic
                and self._is_valid_systematic_name(systematic)
                and systematic not in gene_names
            ):
                gene_names.append(systematic)

        return gene_names

    def _load_release_names(self) -> None:
        """Read the workbook's symbol -> ORF map and its ORF -> symbol inverse."""
        path = osp.join(self.raw_dir, WORKBOOK_FILE_NAME)
        self.symbol_map = release_symbol_map(path)
        self.gene_symbols = release_gene_symbols(self.symbol_map)

    def _is_valid_systematic_name(self, name: str) -> bool:
        """Validate that a systematic name matches the expected format."""
        if not name:
            return False
        # Must match exact pattern: Y + chromosome letter + L/R + 3 digits + C/W (with optional suffix like -A)
        return bool(re.match(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$", name.upper()))

    def _convert_to_systematic(self, gene_name: str) -> str | None:
        """The ORF a title token names through the release's workbook, else None."""
        if not gene_name:
            return None
        return title_token_to_systematic(self.symbol_map, gene_name)

    def _extract_probe_to_gene_mapping(self, gse: Any) -> dict[str, str]:
        """Extract probe ID to gene name mapping from GEO platform annotation."""
        probe_to_gene: dict[str, str] = {}

        if not hasattr(gse, "gpls") or not gse.gpls:
            log.warning("No platform annotation found in GEO dataset")
            return probe_to_gene

        for gpl_name, gpl in gse.gpls.items():
            log.info(f"Processing platform {gpl_name}")

            if hasattr(gpl, "table") and gpl.table is not None:
                table = gpl.table

                # Look for gene symbol columns
                gene_columns = [
                    "ORF",
                    "Gene",
                    "Gene Symbol",
                    "Gene_Symbol",
                    "GENE_SYMBOL",
                    "gene_symbol",
                    "SystematicName",
                    "Systematic_Name",
                    "SYSTEMATIC_NAME",
                ]

                id_column = None
                gene_column = None

                # Find ID and gene columns
                if "ID" in table.columns:
                    id_column = "ID"
                elif "SPOT" in table.columns:
                    id_column = "SPOT"

                for col in gene_columns:
                    if col in table.columns:
                        gene_column = col
                        break

                if id_column and gene_column:
                    for _, row in table.iterrows():
                        probe_id = str(int(row[id_column]))
                        gene_name = str(row[gene_column])

                        # Clean and validate gene name
                        if gene_name and gene_name != "nan" and gene_name != "":
                            gene_name_upper = gene_name.upper()
                            if re.match(r"Y[A-Z]{2}\d{3}[CW]", gene_name_upper):
                                probe_to_gene[probe_id] = gene_name_upper
                            elif re.match(r"Q\d{4}", gene_name_upper):
                                probe_to_gene[probe_id] = gene_name_upper
                            else:
                                probe_to_gene[probe_id] = gene_name_upper

                    log.info(
                        f"Extracted {len(probe_to_gene)} probe-to-gene mappings from {gpl_name}"
                    )
                else:
                    log.warning(
                        f"Could not find appropriate columns in platform {gpl_name}"
                    )
                    log.info(f"Available columns: {list(table.columns)}")

        return probe_to_gene

    def _extract_expression_from_gsm(
        self, gsm: Any, probe_to_gene_map: dict[str, str] | None = None
    ) -> tuple[SortedDict, SortedDict, SortedDict]:
        """(mutant, refpool, log2 ratio) per gene of one array, spots combined.

        Delegates to ``extract_array_expression``; the dye orientation of the array
        (the two cultures of a mutant are hybridized in opposite orientations) is read
        from ``source_name_ch1``.
        """
        return extract_array_expression(gsm, probe_to_gene_map)

    def _calculate_wt_reference_with_std(
        self, wt_gsm_list: list[Any], probe_to_gene_map: dict[str, str]
    ) -> tuple[SortedDict, SortedDict]:
        """Calculate average wildtype expression values and std from GSM objects."""
        if len(wt_gsm_list) == 0:
            log.warning("No wildtype samples found, returning empty reference")
            return SortedDict(), SortedDict()

        log.info(f"Calculating reference from {len(wt_gsm_list)} wildtype samples")

        # Collect all expression values per gene
        all_expressions: dict[str, list[float]] = {}

        for gsm in wt_gsm_list:
            mutant_data, refpool_data, log2_ratio_data = (
                self._extract_expression_from_gsm(gsm, probe_to_gene_map)
            )
            # Use reference pool data for wildtype reference
            for gene, value in refpool_data.items():
                if gene not in all_expressions:
                    all_expressions[gene] = []
                all_expressions[gene].append(value)

        # Calculate mean and std
        wt_mean_expression = SortedDict()
        wt_std_expression = SortedDict()

        for gene, values in all_expressions.items():
            wt_mean_expression[gene] = np.mean(values)
            wt_std_expression[gene] = np.std(values, ddof=1) if len(values) > 1 else 0.0

        log.info(f"Calculated reference for {len(wt_mean_expression)} genes")

        return wt_mean_expression, wt_std_expression

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocess raw data - for Sameith this is handled in process()."""
        return df

    def create_experiment(self) -> None:
        """Required by base class but not used."""
        pass

    @staticmethod
    def create_single_mutant_expression_experiment(
        dataset_name: str,
        sample_info: dict[str, Any],
        mutant_data: SortedDict,
        refpool_data: SortedDict,
        log2_ratio_data: SortedDict,
        *,
        replicate_id: str,
        gene_symbols: dict[str, str],
    ) -> tuple[Any, Any, Any]:
        """Create the record of one single-mutant array (one culture).

        The phenotype is that array's values (``array_phenotypes``): ``n_replicates``
        1 per gene, no SE or variance, ``replicate_id`` the GEO sample accession. The
        reference is the refpool channel of the same array.

        Args:
            dataset_name: Name of the dataset
            sample_info: Sample metadata dictionary of the array
            mutant_data: The array's mutant-channel values, spots combined
            refpool_data: The array's refpool-channel values, spots combined
            log2_ratio_data: The array's log2(mutant/refpool), spots combined
            replicate_id: The array's GEO sample accession
            gene_symbols: The workbook's ORF -> gene symbol map
                (``release_gene_symbols``); ``perturbed_gene_name`` is that symbol, or
                the ORF when the workbook gives the ORF no symbol
        """
        # BY4742 (MATalpha) for every single mutant. Paper: "All single mutants and
        # most double mutants carry the mating type matα and are in the genetic
        # background of BY4742."
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="BY4742",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )

        # Create genotype for single deletion mutant
        gene_names = sample_info["gene_names"]
        perturbations: list[GenePerturbationType] = []

        # Single deletion (KanMX marker from deletion library)
        if len(gene_names) > 0:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=gene_names[0],
                    perturbed_gene_name=gene_symbols.get(gene_names[0], gene_names[0]),
                    strain_id=f"KanMX_{gene_names[0]}",
                )
            )

        genotype = Genotype(perturbations=perturbations)

        # Environment - SC medium at 30°C
        environment = Environment(
            media=Media(name="SC", state="liquid", is_synthetic=True),
            temperature=Temperature(value=30),
        )
        environment_reference = environment.model_copy()

        phenotype, phenotype_reference = array_phenotypes(
            mutant_data, refpool_data, log2_ratio_data, replicate_id
        )

        # Create reference
        reference = MicroarrayExpressionExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        # Create experiment
        experiment = MicroarrayExpressionExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )

        return experiment, reference, sameith2015_publication()


def sameith2015_publication() -> Publication:
    """The publication every Sameith 2015 record cites."""
    return Publication(
        pubmed_id=SAMEITH2015_PUBMED_ID,
        pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{SAMEITH2015_PUBMED_ID}/",
        doi="10.1186/s12915-015-0222-5",
        doi_url="https://doi.org/10.1186/s12915-015-0222-5",
    )


def array_phenotypes(
    mutant_data: SortedDict,
    refpool_data: SortedDict,
    log2_ratio_data: SortedDict,
    replicate_id: str,
) -> tuple[MicroarrayExpressionPhenotype, MicroarrayExpressionPhenotype]:
    """(phenotype, phenotype_reference) of one array record.

    The phenotype is that one culture's values: ``n_replicates`` 1 per gene and no SE
    or variance (one value has no dispersion; the two fields accept no typed gap, so
    they are ``None``), ``replicate_id`` the array's GEO accession. The reference is
    the refpool channel of the same array, ``n_replicates`` 1 per gene, its log2
    ratios 0 (self-referential) and ``replicate_id`` None: it is the common reference
    RNA batch, not a control replicate.
    """
    phenotype = MicroarrayExpressionPhenotype(
        expression=mutant_data,
        expression_log2_ratio=log2_ratio_data,
        expression_log2_ratio_se=None,
        expression_log2_ratio_variance=None,
        n_replicates=dict.fromkeys(mutant_data, 1),
        replicate_id=replicate_id,
    )
    phenotype_reference = MicroarrayExpressionPhenotype(
        expression=refpool_data,
        expression_log2_ratio=dict.fromkeys(refpool_data, 0.0),
        expression_log2_ratio_se=None,
        expression_log2_ratio_variance=None,
        n_replicates=dict.fromkeys(refpool_data, 1),
    )
    return phenotype, phenotype_reference


@register_dataset
class DmMicroarraySameith2015Dataset(ExperimentDataset):
    """Double mutant GSTF expression profiling from Sameith et al. 2015.

    Microarray gene expression data for double-deletion yeast strains pairing
    general stress transcription factors (GSTFs), profiled against a common
    wildtype reference pool, one record per GEO array (one culture): 143 records, 71
    pairs with 2 arrays and cin5+yap6 with 1 (see the module comment). Authoritative
    GSTF pairs and per-sample strain are
    read from the paper's supplementary table. Title tokens resolve to ORFs only
    through the release's Additional file 1 (``release_symbol_map``), and
    ``perturbed_gene_name`` is the workbook's symbol.

    Data source: GEO accession GSE42536
    Paper: Sameith et al. (2015) BMC Biology
    DOI: 10.1186/s12915-015-0222-5
    """

    organism = Organism.scerevisiae

    # GEO accession for double mutant microarray expression dataset
    geo_accession = "GSE42536"

    def __init__(
        self,
        root: str = "data/torchcell/dm_microarray_sameith2015",
        io_workers: int = 0,
        process_workers: int = 0,
        batch_size: int = 10,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ):
        """Initialize the dataset; gene names come from the release's own workbook."""
        self.process_workers = process_workers
        self.batch_size = batch_size

        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment model produced by this dataset."""
        return MicroarrayExpressionExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference model produced by this dataset."""
        return MicroarrayExpressionExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Raw GEO SOFT family file required before processing."""
        return [f"{self.geo_accession}_family.soft.gz"]

    def download(self) -> None:
        """Download double mutant microarray expression data from GEO and supplementary data."""
        log.info(f"Downloading GEO dataset {self.geo_accession}...")

        try:
            gse = GEOparse.get_GEO(
                geo=self.geo_accession, destdir=self.raw_dir, silent=False
            )
            log.info(f"Successfully downloaded {self.geo_accession}")

            # Save the parsed GEO object for use in process()
            geo_pkl_path = osp.join(self.raw_dir, f"{self.geo_accession}.pkl")
            with open(geo_pkl_path, "wb") as f:
                pickle.dump(gse, f)

        except Exception as e:
            log.error(f"Failed to download GEO data: {e}")
            raise RuntimeError(f"Failed to download {self.geo_accession} from GEO")

        # Download supplementary Excel file with authoritative GSTF pairs
        log.info("Downloading supplementary data file...")
        suppl_url = "https://static-content.springer.com/esm/art%3A10.1186%2Fs12915-015-0222-5/MediaObjects/12915_2015_222_MOESM1_ESM.xlsx"
        suppl_path = osp.join(self.raw_dir, "12915_2015_222_MOESM1_ESM.xlsx")

        try:
            if not osp.exists(suppl_path):
                urllib.request.urlretrieve(suppl_url, suppl_path)
                log.info(f"Successfully downloaded supplementary file to {suppl_path}")
            else:
                log.info("Supplementary file already exists")
        except Exception as e:
            log.error(f"Failed to download supplementary file: {e}")
            raise RuntimeError("Failed to download supplementary data")

    def _load_authoritative_gstf_pairs(self) -> dict[tuple[str, str], dict[str, Any]]:
        """Load authoritative GSTF pairs WITH PER-SAMPLE STRAIN from supplementary Excel file.

        Returns:
            dict: Dictionary mapping (gene1, gene2) tuples to their metadata including strain
        """
        suppl_path = osp.join(self.raw_dir, "12915_2015_222_MOESM1_ESM.xlsx")

        # Read the "Double mutants - info" sheet
        df = pd.read_excel(suppl_path, sheet_name="Double mutants - info")

        # Filter for pairs that passed QC
        passed_df = df[df["curation"] == "passed"].copy()

        log.info(
            f"Loaded {len(passed_df)} GSTF pairs that passed QC from supplementary file"
        )

        # Create a dictionary of genotype pairs
        gstf_pairs: dict[tuple[str, str], dict[str, Any]] = {}
        strain_counts = {"BY4742": 0, "BY4741": 0}

        for _, row in passed_df.iterrows():
            gstf1_sys = row["GSTF1,           systematic name"].strip()
            gstf2_sys = row["GSTF2,           systematic name"].strip()

            # Create canonical genotype key (sorted to handle order-independence)
            genotype_key = tuple(sorted([gstf1_sys.upper(), gstf2_sys.upper()]))

            # Strain from the comments column. Paper: "All single mutants and most
            # double mutants carry the mating type matα and are in the genetic
            # background of BY4742. Few double mutants carry the mating type matA and
            # are in the genetic background of BY4741." The SI marks those few with a
            # "MATa" comment (4 of the 72 passed pairs), and GEO's own sample
            # characteristics carry "strain: BY4741" on exactly those four (checked
            # 2026-09-28 on GSE42536). BY4741 is the MATa strain and BY4742 the
            # MATalpha strain, so a MATa comment means BY4741; "mata" is a prefix of
            # "matalpha", hence the order of the tests.
            comments = row.get("comments", "")
            if pd.notna(comments):
                lowered = str(comments).lower()
                if "matα" in lowered or "matalpha" in lowered:
                    strain = "BY4742"
                elif "mata" in lowered:
                    strain = "BY4741"
                else:
                    strain = "BY4742"  # Default for comments without mating type
            else:
                strain = "BY4742"  # Default for blank comments

            strain_counts[strain] += 1

            gstf_pairs[genotype_key] = {
                "gstf1_systematic": gstf1_sys.upper(),
                "gstf2_systematic": gstf2_sys.upper(),
                "gstf1_symbol": row["GSTF1,           gene symbol"],
                "gstf2_symbol": row["GSTF2,           gene symbol"],
                "selection": row["selection"],
                "comments": comments,
                "strain": strain,  # NEW: Per-sample strain extracted from comments
            }

        log.info(
            f"Strain distribution: BY4742={strain_counts['BY4742']}, BY4741={strain_counts['BY4741']}"
        )
        return gstf_pairs

    @post_process
    def process(self) -> None:
        """Parse GEO expression data into double-mutant experiments and write LMDB."""
        # Title tokens resolve through the release's own workbook only.
        self._load_release_names()

        # Load the GEO object
        geo_pkl_path = osp.join(self.raw_dir, f"{self.geo_accession}.pkl")
        if osp.exists(geo_pkl_path):
            with open(geo_pkl_path, "rb") as f:
                gse = pickle.load(f)
        else:
            # Re-download if pickle doesn't exist
            gse = GEOparse.get_GEO(
                geo=self.geo_accession, destdir=self.raw_dir, silent=False
            )

        # Load authoritative GSTF pairs from supplementary file
        self.gstf_pairs = self._load_authoritative_gstf_pairs()

        log.info("Processing GEO samples for double mutants...")

        # Extract platform annotation for probe-to-gene mapping
        probe_to_gene_map = self._extract_probe_to_gene_mapping(gse)
        log.info(f"Found {len(probe_to_gene_map)} probe-to-gene mappings")

        # Parse samples and extract metadata
        samples_data = []
        single_mutant_samples: dict[str, list[Any]] = {}  # Group by gene name
        double_mutant_samples: list[Any] = []  # List of double mutant sample infos
        wt_samples = []

        for gsm_name, gsm in gse.gsms.items():
            sample_info = {
                "geo_accession": gsm_name,
                "title": gsm.metadata.get("title", [""])[0],
            }

            # Extract gene names from title
            gene_names = self._extract_gene_names_from_title(sample_info["title"])

            # Determine sample type
            is_wildtype = (
                "wt" in sample_info["title"].lower()
                or "wildtype" in sample_info["title"].lower()
            )
            is_double = len(gene_names) >= 2
            is_single = len(gene_names) == 1

            sample_info["is_wildtype"] = is_wildtype
            sample_info["is_double_mutant"] = is_double
            sample_info["is_single_mutant"] = is_single
            sample_info["gene_names"] = gene_names
            sample_info["gsm_object"] = gsm

            if is_wildtype:
                wt_samples.append(gsm)
            elif is_single:
                gene = gene_names[0]
                if gene not in single_mutant_samples:
                    single_mutant_samples[gene] = []
                single_mutant_samples[gene].append(gsm)
            elif is_double:
                double_mutant_samples.append(sample_info)

            samples_data.append(sample_info)

        log.info(f"Found {len(single_mutant_samples)} unique single mutant genes")
        log.info(f"Found {len(double_mutant_samples)} double mutant samples")
        log.info(f"Found {len(wt_samples)} wildtype samples")

        # Calculate wildtype reference expression
        if wt_samples:
            self.wt_reference_expression, self.wt_std_expression = (
                self._calculate_wt_reference_with_std(wt_samples, probe_to_gene_map)
            )
        else:
            log.warning("No wildtype samples found, using synthetic reference")
            self.wt_reference_expression = SortedDict()
            self.wt_std_expression = SortedDict()

        log.info(
            f"Found {len(single_mutant_samples)} single mutant genes (not processed separately)"
        )

        # Group the arrays of each pair (one array per culture; each is a record)
        double_mutant_groups = self._group_arrays_by_pair(double_mutant_samples)
        log.info(
            f"Grouped {len(double_mutant_samples)} arrays into {len(double_mutant_groups)} pair groups"
        )

        # Choose processing method based on process_workers
        if self.process_workers > 0:
            log.info(
                f"Processing {len(double_mutant_groups)} double mutant groups in parallel with {self.process_workers} workers..."
            )
            self._process_parallel(double_mutant_groups, probe_to_gene_map)
        else:
            log.info(
                f"Processing {len(double_mutant_groups)} double mutant groups sequentially..."
            )
            self._process_sequential(double_mutant_groups, probe_to_gene_map)

        # Save preprocessed data - filter for double mutants only
        os.makedirs(self.preprocess_dir, exist_ok=True)
        samples_df = pd.DataFrame(samples_data)
        samples_df = samples_df.drop("gsm_object", axis=1)
        # Filter to only include double mutant samples
        double_mutant_df = samples_df[samples_df["is_double_mutant"]]
        double_mutant_df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)
        log.info(
            f"Saved {len(double_mutant_df)} double mutant samples to preprocess data.csv"
        )

        log.info(f"Processed {len(double_mutant_samples)} double mutant arrays")

    def _group_arrays_by_pair(self, double_mutant_samples: list[Any]) -> list[Any]:
        """Group the arrays of each double mutant by its pair of deleted genes.

        Each array is one culture (two biological replicates per pair, module
        comment), and every array of a group becomes its own record; the group only
        orders them. An array whose pair is not a ``passed`` row of the SI's "Double
        mutants - info" is left out and logged: the paper excludes the 18 pairs that
        "failed quality control criteria even after remake" (L115), and GSE42536
        holds no array of them (0 unmatched on the release).
        """
        groups: dict[Any, list[Any]] = {}
        matched_count = 0
        unmatched_count = 0

        for sample_info in double_mutant_samples:
            # Extract gene names from sample title
            gene_names = sample_info["gene_names"]

            if len(gene_names) >= 2:
                # Create canonical genotype key (sorted)
                extracted_key = tuple(sorted(gene_names[:2]))

                # Try to match against authoritative pairs
                if extracted_key in self.gstf_pairs:
                    # Use the authoritative pair as the key
                    genotype_key = extracted_key
                    matched_count += 1
                else:
                    # Log unmatched sample for debugging
                    log.warning(
                        f"Sample {sample_info['geo_accession']} with genes {extracted_key} "
                        f"not found in authoritative GSTF pairs"
                    )
                    unmatched_count += 1
                    continue  # Skip unmatched samples

                if genotype_key not in groups:
                    groups[genotype_key] = []
                groups[genotype_key].append(sample_info)

        log.info(
            f"Matched {matched_count} samples to authoritative pairs, "
            f"{unmatched_count} unmatched"
        )
        log.info(f"Found {len(groups)} unique genotypes")

        return list(groups.values())

    def _process_sequential(
        self, double_mutant_groups: list[Any], probe_to_gene_map: dict[str, str]
    ) -> None:
        """Write one record per array of every pair group, in group order.

        Builds the records through ``_process_batch`` (the worker path), so the
        sequential and parallel builds cannot diverge.
        """
        records = self._process_batch(
            double_mutant_groups,
            probe_to_gene_map,
            self.name,
            self.gstf_pairs,
            self.gene_symbols,
        )
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(5e12))
        with env.begin(write=True) as txn:
            for idx, serialized_data in enumerate(records):
                txn.put(f"{idx}".encode(), serialized_data)
        env.close()
        log.info(f"Wrote {len(records)} double mutant array records to LMDB")

    def _process_parallel(
        self, double_mutant_groups: list[Any], probe_to_gene_map: dict[str, str]
    ) -> None:
        """Write the same records as ``_process_sequential``, batches built in workers."""
        batches = [
            double_mutant_groups[i : i + self.batch_size]
            for i in range(0, len(double_mutant_groups), self.batch_size)
        ]
        log.info(f"Created {len(batches)} batches of size {self.batch_size}")

        all_results: list[bytes] = []
        with ProcessPoolExecutor(max_workers=self.process_workers) as executor:
            futures = [
                executor.submit(
                    self._process_batch,
                    batch,
                    probe_to_gene_map,
                    self.name,
                    self.gstf_pairs,
                    self.gene_symbols,
                )
                for batch in batches
            ]
            for future in tqdm(futures, desc="Processing batches"):
                all_results.extend(future.result())

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(5e12))
        with env.begin(write=True) as txn:
            for idx, serialized_data in enumerate(all_results):
                txn.put(f"{idx}".encode(), serialized_data)
        env.close()
        log.info(f"Wrote {len(all_results)} double mutant array records to LMDB")

    @staticmethod
    def _process_batch(
        batch_items: list[Any],
        probe_to_gene_map: dict[str, str],
        dataset_name: str,
        gstf_pairs: dict[tuple[str, str], dict[str, Any]],
        gene_symbols: dict[str, str],
    ) -> list[bytes]:
        """One serialized record per array of each pair group (static, for workers).

        A group not flagged double or naming fewer than two genes writes nothing. An
        array that yields no expression value raises: every released array is a
        record, so none is skipped. The strain is looked up under the sorted,
        upper-cased pair.
        """
        results = []
        for replicate_group in batch_items:
            sample_info = replicate_group[0]
            if (
                not sample_info["is_double_mutant"]
                or len(sample_info["gene_names"]) < 2
            ):
                continue
            gene_names = sample_info["gene_names"]
            genotype_key = tuple(sorted([gene_names[0].upper(), gene_names[1].upper()]))
            strain = gstf_pairs.get(genotype_key, {}).get("strain", "BY4742")
            for array_info in replicate_group:
                accession = array_info["geo_accession"]
                mutant_data, refpool_data, log2_ratio_data = (
                    DmMicroarraySameith2015Dataset._extract_expression_from_gsm_static(
                        array_info["gsm_object"], probe_to_gene_map
                    )
                )
                if not mutant_data:
                    raise RuntimeError(
                        f"array {accession} ({array_info['gene_names']}) yields no "
                        "expression value; a released array is never skipped"
                    )
                experiment, reference, publication = (
                    DmMicroarraySameith2015Dataset.create_double_mutant_expression_experiment(
                        dataset_name,
                        array_info,
                        mutant_data,
                        refpool_data,
                        log2_ratio_data,
                        strain=strain,
                        replicate_id=accession,
                        gene_symbols=gene_symbols,
                    )
                )
                results.append(
                    pickle.dumps(
                        {
                            "experiment": experiment.model_dump(),
                            "reference": reference.model_dump(),
                            "publication": publication.model_dump(),
                        }
                    )
                )
        return results

    def _extract_gene_names_from_title(self, title: str) -> list[str]:
        """Up to two genes a sample title names: systematic names, then workbook symbols."""
        gene_names = []

        # Look for systematic names (e.g., YAL001C or YBR089C-A)
        # Match the full systematic name, including optional suffix like -A
        systematic_pattern = r"\b(Y[A-P][LR]\d{3}[WC](?:-[A-Z])?)\b"
        matches = re.findall(systematic_pattern, title.upper())

        # Validate each match
        for match in matches:
            if self._is_valid_systematic_name(match):
                gene_names.append(match)

        # Also try to extract common names (even if we found systematic names)
        # This handles cases like "rpn4-del+ydr026c-del" where one is common and one is systematic
        if len(gene_names) < 2:
            # Extract potential gene names (all caps words)
            common_pattern = r"\b([A-Z][A-Z0-9]{2,})\b"
            potential_genes = re.findall(common_pattern, title.upper())

            for gene in potential_genes:
                # Skip common non-gene words
                if gene in ["DEL", "WT", "WILDTYPE", "MUTANT", "DOUBLE"]:
                    continue

                # Try to convert to systematic
                systematic = self._convert_to_systematic(gene)
                if systematic and self._is_valid_systematic_name(systematic):
                    # Avoid duplicates
                    if systematic not in gene_names:
                        gene_names.append(systematic)
                        # Stop if we have 2 genes
                        if len(gene_names) >= 2:
                            break

        # Log if we couldn't extract valid gene names
        if len(gene_names) < 2 and not hasattr(self, "_logged_extraction_warning"):
            log.warning(f"Could not extract 2 valid gene names from title: {title}")
            log.warning(f"Extracted: {gene_names}")
            self._logged_extraction_warning = True

        return gene_names[:2]  # Return at most 2 genes for double mutants

    def _load_release_names(self) -> None:
        """Read the workbook's symbol -> ORF map and its ORF -> symbol inverse."""
        path = osp.join(self.raw_dir, WORKBOOK_FILE_NAME)
        self.symbol_map = release_symbol_map(path)
        self.gene_symbols = release_gene_symbols(self.symbol_map)

    def _is_valid_systematic_name(self, name: str) -> bool:
        """Validate that a systematic name matches the expected format."""
        if not name:
            return False
        # Must match exact pattern: Y + chromosome letter + L/R + 3 digits + C/W (with optional suffix like -A)
        return bool(re.match(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$", name.upper()))

    def _convert_to_systematic(self, gene_name: str) -> str | None:
        """The ORF a title token names through the release's workbook, else None."""
        if not gene_name:
            return None
        return title_token_to_systematic(self.symbol_map, gene_name)

    def _extract_probe_to_gene_mapping(self, gse: Any) -> dict[str, str]:
        """Extract probe ID to gene name mapping from GEO platform annotation."""
        probe_to_gene: dict[str, str] = {}

        if not hasattr(gse, "gpls") or not gse.gpls:
            log.warning("No platform annotation found in GEO dataset")
            return probe_to_gene

        for gpl_name, gpl in gse.gpls.items():
            log.info(f"Processing platform {gpl_name}")

            if hasattr(gpl, "table") and gpl.table is not None:
                table = gpl.table

                # Look for gene symbol columns
                gene_columns = [
                    "ORF",
                    "Gene",
                    "Gene Symbol",
                    "Gene_Symbol",
                    "GENE_SYMBOL",
                    "gene_symbol",
                    "SystematicName",
                    "Systematic_Name",
                    "SYSTEMATIC_NAME",
                ]

                id_column = None
                gene_column = None

                # Find ID and gene columns
                if "ID" in table.columns:
                    id_column = "ID"
                elif "SPOT" in table.columns:
                    id_column = "SPOT"

                for col in gene_columns:
                    if col in table.columns:
                        gene_column = col
                        break

                if id_column and gene_column:
                    for _, row in table.iterrows():
                        probe_id = str(int(row[id_column]))
                        gene_name = str(row[gene_column])

                        # Clean and validate gene name
                        if gene_name and gene_name != "nan" and gene_name != "":
                            gene_name_upper = gene_name.upper()
                            if re.match(r"Y[A-Z]{2}\d{3}[CW]", gene_name_upper):
                                probe_to_gene[probe_id] = gene_name_upper
                            elif re.match(r"Q\d{4}", gene_name_upper):
                                probe_to_gene[probe_id] = gene_name_upper
                            else:
                                probe_to_gene[probe_id] = gene_name_upper

                    log.info(
                        f"Extracted {len(probe_to_gene)} probe-to-gene mappings from {gpl_name}"
                    )
                else:
                    log.warning(
                        f"Could not find appropriate columns in platform {gpl_name}"
                    )
                    log.info(f"Available columns: {list(table.columns)}")

        return probe_to_gene

    def _extract_expression_from_gsm(
        self, gsm: Any, probe_to_gene_map: dict[str, str] | None = None
    ) -> tuple[SortedDict, SortedDict, SortedDict]:
        """(mutant, refpool, log2 ratio) per gene of one array, spots combined.

        Delegates to ``extract_array_expression``.
        """
        return extract_array_expression(gsm, probe_to_gene_map)

    @staticmethod
    def _extract_expression_from_gsm_static(
        gsm: Any, probe_to_gene_map: dict[str, str] | None = None
    ) -> tuple[SortedDict, SortedDict, SortedDict]:
        """Static form of ``_extract_expression_from_gsm`` for worker processes."""
        return extract_array_expression(gsm, probe_to_gene_map)

    def _calculate_wt_reference_with_std(
        self, wt_gsm_list: list[Any], probe_to_gene_map: dict[str, str]
    ) -> tuple[SortedDict, SortedDict]:
        """Calculate average wildtype expression values and std from GSM objects."""
        if len(wt_gsm_list) == 0:
            log.warning("No wildtype samples found, returning empty reference")
            return SortedDict(), SortedDict()

        log.info(f"Calculating reference from {len(wt_gsm_list)} wildtype samples")

        # Collect all expression values per gene
        all_expressions: dict[str, list[float]] = {}

        for gsm in wt_gsm_list:
            mutant_data, refpool_data, log2_ratio_data = (
                self._extract_expression_from_gsm(gsm, probe_to_gene_map)
            )
            # Use reference pool data for wildtype reference
            for gene, value in refpool_data.items():
                if gene not in all_expressions:
                    all_expressions[gene] = []
                all_expressions[gene].append(value)

        # Calculate mean and std
        wt_mean_expression = SortedDict()
        wt_std_expression = SortedDict()

        for gene, values in all_expressions.items():
            wt_mean_expression[gene] = np.mean(values)
            wt_std_expression[gene] = np.std(values, ddof=1) if len(values) > 1 else 0.0

        log.info(f"Calculated reference for {len(wt_mean_expression)} genes")

        return wt_mean_expression, wt_std_expression

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocess raw data - for Sameith this is handled in process()."""
        return df

    def create_experiment(self) -> None:
        """Required by base class but not used."""
        pass

    @staticmethod
    def create_double_mutant_expression_experiment(
        dataset_name: str,
        sample_info: dict[str, Any],
        mutant_data: SortedDict,
        refpool_data: SortedDict,
        log2_ratio_data: SortedDict,
        strain: str = "BY4742",
        *,
        replicate_id: str,
        gene_symbols: dict[str, str],
    ) -> tuple[Any, Any, Any]:
        """Create the record of one double-mutant array (one culture), with its strain.

        The phenotype is that array's values (``array_phenotypes``): ``n_replicates``
        1 per gene, no SE or variance, ``replicate_id`` the GEO sample accession. The
        reference is the refpool channel of the same array.

        Args:
            dataset_name: Name of the dataset
            sample_info: Sample metadata dictionary of the array
            mutant_data: The array's mutant-channel values, spots combined
            refpool_data: The array's refpool-channel values, spots combined
            log2_ratio_data: The array's log2(mutant/refpool), spots combined
            strain: Strain background (BY4742 or BY4741), extracted from Excel comments
            replicate_id: The array's GEO sample accession
            gene_symbols: The workbook's ORF -> gene symbol map
                (``release_gene_symbols``); ``perturbed_gene_name`` is that symbol, or
                the ORF when the workbook gives the ORF no symbol
        """
        # Per-pair strain from the SI comments column: "MATa" -> BY4741, otherwise
        # BY4742 (see _load_authoritative_gstf_pairs).
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain=strain,
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )

        # Create genotype for double deletion mutant
        gene_names = sample_info["gene_names"]
        perturbations: list[GenePerturbationType] = []

        # First deletion (KanMX marker)
        if len(gene_names) > 0:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=gene_names[0],
                    perturbed_gene_name=gene_symbols.get(gene_names[0], gene_names[0]),
                    strain_id=f"KanMX_{gene_names[0]}",
                )
            )

        # Second deletion (NatMX marker for SGA)
        if len(gene_names) > 1:
            perturbations.append(
                SgaNatMxDeletionPerturbation(
                    systematic_gene_name=gene_names[1],
                    perturbed_gene_name=gene_symbols.get(gene_names[1], gene_names[1]),
                    strain_id=f"NatMX_{gene_names[1]}",
                )
            )

        genotype = Genotype(perturbations=perturbations)

        # Environment - SC medium at 30°C
        environment = Environment(
            media=Media(name="SC", state="liquid", is_synthetic=True),
            temperature=Temperature(value=30),
        )
        environment_reference = environment.model_copy()

        phenotype, phenotype_reference = array_phenotypes(
            mutant_data, refpool_data, log2_ratio_data, replicate_id
        )

        # Create reference
        reference = MicroarrayExpressionExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        # Create experiment
        experiment = MicroarrayExpressionExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )

        return experiment, reference, sameith2015_publication()


if __name__ == "__main__":
    raise SystemExit(
        "the demo main moved to torchcell/scratch/sameith2015_demo.py on 2026-10-06"
    )
