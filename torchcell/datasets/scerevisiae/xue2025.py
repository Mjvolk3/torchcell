# torchcell/datasets/scerevisiae/xue2025
# [[torchcell.datasets.scerevisiae.xue2025]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/xue2025
# Test file: tests/torchcell/datasets/scerevisiae/test_xue2025.py
"""Xue 2025 in-house free-fatty-acid (FFA) combinatorial-deletion titer dataset.

An IN-HOUSE (unpublished) engineering screen: an FFA-overproduction yeast chassis
(BY4741 background carrying the standard free-fatty-acid chassis triple deletion
POX1 + FAA1 + FAA4) into which up to THREE ADDITIONAL transcription-factor (TF) deletions
(1-3 of the 10 TFs; max strain = 6 total deletions) were combinatorially stacked, then
assayed by GC for the absolute titer (mg/L) of five
free fatty acids per strain: C14:0, C16:0, C18:0, C16:1, C18:1. 177 genotype rows: one
true wild type (``wt BY4741``), one baseline-only positive control (``+ve Ctrl`` = the
three-deletion chassis alone), and 175 combinatorial TF-deletion strains.

Maps to ``MetaboliteExperiment`` / ``MetabolitePhenotype`` (one record per genotype row,
EXCEPT ``wt BY4741`` which is the measured reference). Each record's
``metabolite_level`` carries all five FFA titers at once (multi-metabolite pattern, like
``dasilveira2014``): ``metabolite_level = {FFA -> mean of the replicate titers (mg/L)}``,
``metabolite_level_se = {FFA -> sample-SD / sqrt(n)}``, ``n_replicates = {FFA -> n}``,
``measurement_type = "titer_mg_per_l"`` (ABSOLUTE FFA titer, not a ratio nor a centered
score). ``target_metabolite_ids = None`` (the FFAs are acyl species; the Yeast9
``s_NNNN`` / ChEBI mapping is deferred). ``reference_centered = False`` for verification
(the reference is a MEASURED WT baseline, not an identically-zero centered origin; this is
a verifier parameter set in ``verification/runners.py``, not a schema field).

DATA SOURCE (sha256-pinned in the library mirror ``$DATA_ROOT/torchcell-library/
xue2025/data/``):
- ``Supplementary Data 1_Raw titers.xlsx`` (sha256 pinned below). Two sheets:
  - ``Abbreviations`` (header=None): col0 = TF common name, col1 = single-letter code
    (F=FKH1, G=GCN5, M=MED4, O=OPI1, X=RFX1, R=RGR1, P=RPD3, S=SPT3, Y=YAP6, T=TFC7).
    Read programmatically and asserted against this expected map.
  - ``raw-titer (mg-L)``: col0 = genotype string; then 5 FFAs x 3 replicate columns of raw
    per-replicate titer (mg/L): C14:0 = cols 1-3, C16:0 = cols 4-6, C18:0 = cols 7-9,
    C16:1 = cols 10-12, C18:1 = cols 13-15. Cols 16-21 are the authors' per-FFA averages +
    total titer and are IGNORED (we recompute mean/SD from the 3 replicate columns).

GENOTYPE DECODING (validated): every non-WT strain carries the IMPLICIT FFA-chassis
baseline of THREE deletions (POX1, FAA1, FAA4). The genotype string is
``<letters> <N>d`` where ``letters`` (split on '-') are ADDITIONAL TF deletions and
``N`` = 3 + (number of TF letters). ``wt BY4741`` -> no deletions (true WT, the
reference); ``+ve Ctrl`` -> the baseline only; e.g. ``P-S-Y 6d`` ->
[POX1, FAA1, FAA4, RPD3, SPT3, YAP6]. We ASSERT ``len(TF letters) + 3 == N`` for every
parsed row. All 13 genes (baseline + 10 TFs) are stored under their released names.

GENE IDENTITY (#1045): the release names every gene by its common name only, and each
is stored under that name verbatim on ``ReleasedGeneNamePerturbation`` (kind
``deletion``, marker KanMX); a systematic name, had the release used one, would go on
``KanMxDeletionPerturbation``. No genome is consulted, so ``TFC7`` (the standard name of
YOR110W and an alias of YNL039W) is stored as the symbol the release wrote, and the join
to R64 is a modeling-time act. Before #1045 the loader resolved every name through
``resolve_gene_name_strict`` with a pin for TFC7 (#886).

REPLICATE STRUCTURE: 3 GC replicate columns per FFA. Some strains released fewer than 3
replicates (an entire replicate column is blank across all FFAs -- 10 strains have 2, one
strain (``G-O-T 6d``) has 1). We therefore compute ``n_replicates`` PER FFA from the count
of non-blank replicates (never hardcode 3), the mean over the present replicates, and the
sample SD (ddof=1) over the present replicates; ``SE = SD / sqrt(n)``. For ``n == 1`` the
sample SD is undefined, so that FFA's SE is stored as NaN (the schema permits NaN SE) and
``n_replicates = 1``.

DELETION MARKER (documented flag, NOT asserted fidelity): these are IN-HOUSE COMBINATORIAL
strains carrying up to SIX deletions. Six independent KanMX cassettes cannot be stacked
in one strain (mixed/rotated markers are used in practice), and the actual per-locus markers
are not released. We therefore model each deletion as ``KanMxDeletionPerturbation`` as a
DOCUMENTED REPRESENTATIVE -- the asserted fact is ``state = "absent"`` (the gene is deleted),
NOT the KanMX marker identity. Do not read marker fidelity into these records.

ESSENTIAL GENES STATED AS DELETIONS (#887): the release states RGR1 (YLR071C), MED4
(YOR174W) and TFC7 (YOR110W) as deletions in the haploid chassis (``R 4Δ``, ``M 4Δ``,
``T 4Δ``; 46 records each, 112 of 176 carry at least one). SGD records the null of RGR1
and of MED4 as inviable in S288C (Giaever 2002; the two ``gene_essentiality_sgd``
records), and the null of TFC7 as inviable only in a non-S288C background (Manaud 1998,
strain "Other"). The release states no allele: its one file names each strain by a
deletion count, and no cell of either sheet names a truncation, DAmP, degron,
conditional, hypomorph or partial allele. Owner decision of 2026-10-12: "the data is
the data! ... if these are the genes reported then that is what they are", so every
record is stored as the spreadsheet reports it, with no flag; under which conditions
these genes are essential, and what allele the strains carry, is a follow-up for the
strain builders (issue #887; experiments.008-xue-ffa.results.002-interaction-models-
report, 2026.10.12). Measured by ``experiments/036-dataset-fixes-before-kg-build/
scripts/keep_records_887_sd_zero_inventory.py``.

ENVIRONMENT (in-house-assumed, FLAGGED): the media name and temperature are NOT in this data
file (in-house engineering data). We use ``Media(name="SC (FFA production)", state="liquid")``
and ``Temperature(value=30.0)`` (standard yeast cultivation temperature; user-approved
assumption) with aerobic cultivation. There is no small-molecule environment perturbation.
Both the media name and 30 C are IN-HOUSE ASSUMPTIONS, flagged here for review.

PUBLICATION (in-house/unpublished; methodology anchor, FLAGGED): this FFA titer data is
unpublished in-house data with no DOI/PMID. ``Publication`` requires a resolvable
identifier, so -- as with ``lopez2024`` -- the citation anchors to the peer-reviewed paper
that established the exact FFA-overproduction chassis used here (the faa1d faa4d pox1d
free-fatty-acid chassis): Runguphan & Keasling, "Metabolic engineering of Saccharomyces
cerevisiae for production of fatty acid-derived biofuels and chemicals," Metab Eng 2014
(DOI 10.1016/j.ymben.2013.07.003, PMID 23899824). The DATA source is the sha256-pinned
xlsx in the library mirror. Flagged for review.
"""

import logging
import math
import os
import os.path as osp
import pickle
import re
from collections.abc import Callable
from typing import Any

import lmdb
import pandas as pd
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    copy_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
    GenePerturbationType,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    Publication,
    ReferenceGenome,
    ReleasedGeneKind,
    ReleasedGeneNamePerturbation,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# The stored level is an ABSOLUTE free-fatty-acid titer in mg/L (not a ratio/centered score).
MEASUREMENT_TYPE = "titer_mg_per_l"

_LIBRARY_CITATION_KEY = "xue2025"
DATA_FILENAME = "Supplementary Data 1_Raw titers.xlsx"
DATA_SHA256 = "023de80ec51a4d0d1ccd6fc3506e42a44fcea6d3c3101ee7a83a005e49bbc779"

_ABBREV_SHEET = "Abbreviations"
_TITER_SHEET = "raw-titer (mg-L)"

# Implicit FFA-overproduction chassis baseline present in EVERY non-WT strain.
_BASELINE_GENES = ["POX1", "FAA1", "FAA4"]

# Expected single-letter TF code -> common name (asserted against the Abbreviations sheet).
_EXPECTED_CODE_TO_GENE = {
    "F": "FKH1",
    "G": "GCN5",
    "M": "MED4",
    "O": "OPI1",
    "X": "RFX1",
    "R": "RGR1",
    "P": "RPD3",
    "S": "SPT3",
    "Y": "YAP6",
    "T": "TFC7",
}

#: An S288C systematic name, the form that goes on the placed deletion leaf.
_SYSTEMATIC_RE = re.compile(
    r"^(Y[A-P][LR]\d{3}[WC](-[A-Z])?|Q\d{4}|YNC[A-Q]\d{4}[WC])$"
)


def _released_deletion(
    gene: str,
) -> KanMxDeletionPerturbation | ReleasedGeneNamePerturbation:
    """The deletion under the name the release gives (#1045)."""
    if _SYSTEMATIC_RE.match(gene):
        return KanMxDeletionPerturbation(
            systematic_gene_name=gene, perturbed_gene_name=gene
        )
    return ReleasedGeneNamePerturbation(
        systematic_gene_name=gene,
        gene_namespace="scerevisiae_s288c_sgd_systematic",
        perturbed_gene_name=gene,
        released_gene_name=gene,
        released_kind=ReleasedGeneKind.deletion,
        marker="KanMX",
    )


# 5 FFAs x 3 replicate columns (0-based positions in the titer sheet; col0 = genotype).
_FFA_COLUMNS: dict[str, list[int]] = {
    "C14:0": [1, 2, 3],
    "C16:0": [4, 5, 6],
    "C18:0": [7, 8, 9],
    "C16:1": [10, 11, 12],
    "C18:1": [13, 14, 15],
}

# True wild-type row = the measured reference (not an experiment record).
_WT_LABELS = {"wt BY4741", "BY4741"}
# Baseline-only positive control = the three-deletion chassis alone (an experiment record).
_POS_CTRL_LABEL = "+ve Ctrl"

# Methodology / chassis anchor (see module docstring PUBLICATION note; in-house data).
_PMID = "23899824"
_DOI = "10.1016/j.ymben.2013.07.003"

# Genotype string pattern: "<letters> <N>d" (the delta glyph is stripped before matching).
_GENOTYPE_RE = re.compile(r"^(?P<letters>[A-Z](?:-[A-Z])*)\s+(?P<n>\d+)$")


@register_dataset
class FattyAcidXue2025Dataset(ExperimentDataset):
    """Xue 2025 in-house FFA combinatorial-deletion titer dataset (5 FFAs/strain)."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/ffa_xue2025",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset; every gene is stored under its released name."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return MetaboliteExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return MetaboliteExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The mirrored raw-titer xlsx (Abbreviations + raw-titer sheets)."""
        return [DATA_FILENAME]

    def download(self) -> None:
        """Copy the pinned library xlsx into raw_dir and verify its sha256."""
        os.makedirs(self.raw_dir, exist_ok=True)
        dest = osp.join(self.raw_dir, DATA_FILENAME)
        if not osp.exists(dest):
            data_root = os.environ["DATA_ROOT"]
            src = osp.join(
                data_root,
                "torchcell-library",
                _LIBRARY_CITATION_KEY,
                "data",
                DATA_FILENAME,
            )
            if not osp.exists(src):
                raise RuntimeError(
                    f"library mirror data file not found: {src}. This dataset's source is "
                    f"the sha256-pinned {DATA_FILENAME} in the torchcell-library mirror."
                )
            copy_verified(src, dest, DATA_SHA256)
            log.info("Verified %s (sha256 %s)", dest, DATA_SHA256)

    def _code_to_gene(self) -> dict[str, str]:
        """Read the Abbreviations sheet (code -> common name) and assert the expected map."""
        path = osp.join(self.raw_dir, DATA_FILENAME)
        ab = pd.read_excel(path, sheet_name=_ABBREV_SHEET, header=None)
        code_to_gene = {
            str(row[1]).strip(): str(row[0]).strip() for _, row in ab.iterrows()
        }
        if code_to_gene != _EXPECTED_CODE_TO_GENE:
            raise RuntimeError(
                "Abbreviations sheet does not match the expected TF code map: "
                f"got {code_to_gene}, expected {_EXPECTED_CODE_TO_GENE}"
            )
        return code_to_gene

    def _decode_genotype(
        self, label: str, code_to_gene: dict[str, str]
    ) -> list[str] | None:
        """Decode a genotype string into a list of deleted common gene names.

        Returns [] for the true wild type (``wt BY4741``), the baseline chassis for the
        positive control, and baseline + TF deletions otherwise. Asserts the deletion
        count encoded in the string (``<N>d``) equals 3 (baseline) + the TF-letter count.
        """
        label = label.strip()
        if label in _WT_LABELS:
            return []
        if label == _POS_CTRL_LABEL:
            return list(_BASELINE_GENES)
        # Strip the trailing delta glyph (rendered variably); keep only "<letters> <N>".
        core = label.rstrip().rstrip("Δ∆d ").strip()
        match = _GENOTYPE_RE.match(core)
        if match is None:
            raise RuntimeError(f"unparseable genotype string {label!r} (core {core!r})")
        letters = match.group("letters").split("-")
        n_total = int(match.group("n"))
        if len(letters) + len(_BASELINE_GENES) != n_total:
            raise RuntimeError(
                f"{label!r}: TF letters ({len(letters)}) + baseline "
                f"({len(_BASELINE_GENES)}) != declared deletion count {n_total}"
            )
        tf_genes = [code_to_gene[c] for c in letters]
        return list(_BASELINE_GENES) + tf_genes

    def _phenotype_from_row(self, row: pd.Series) -> MetabolitePhenotype:
        """Compute a MetabolitePhenotype (5 FFA titers) from one titer-sheet row."""
        level: dict[str, float] = {}
        se: dict[str, float] = {}
        n_rep: dict[str, int] = {}
        for ffa, cols in _FFA_COLUMNS.items():
            values = [float(row[c]) for c in cols if pd.notna(row[c])]
            if not values:
                raise RuntimeError(
                    f"FFA {ffa} has no replicate values in row {row[0]!r}"
                )
            n = len(values)
            mean = sum(values) / n
            level[ffa] = mean
            n_rep[ffa] = n
            if n >= 2:
                var = sum((v - mean) ** 2 for v in values) / (
                    n - 1
                )  # sample SD (ddof=1)
                se[ffa] = math.sqrt(var) / math.sqrt(n)
            else:
                se[ffa] = math.nan  # single replicate: SD undefined
        return MetabolitePhenotype(
            metabolite_level=level,
            metabolite_level_se=se,
            n_replicates=n_rep,
            measurement_type=MEASUREMENT_TYPE,
            target_metabolite_ids=None,  # FFAs are acyl species; s_NNNN/ChEBI mapping deferred
        )

    def _environment(self) -> Environment:
        """FFA-production medium, 30 C, aerobic (media name + 30 C are in-house-assumed)."""
        return Environment(
            media=Media(name="SC (FFA production)", state="liquid", is_synthetic=True),
            temperature=Temperature(value=30.0),
            aerobicity="aerobic",
        )

    @staticmethod
    def _genotype(deleted_genes: list[str]) -> Genotype:
        """Build a Genotype: one deletion per deleted gene, under its released name."""
        perturbations: list[GenePerturbationType] = [
            _released_deletion(common) for common in deleted_genes
        ]
        return Genotype(perturbations=perturbations)

    @post_process
    def process(self) -> None:
        """Parse the raw-titer sheet into per-strain Metabolite experiments; write LMDB."""
        verify_raw_files(self.raw_dir, {DATA_FILENAME: DATA_SHA256})
        code_to_gene = self._code_to_gene()
        path = osp.join(self.raw_dir, DATA_FILENAME)
        raw = pd.read_excel(path, sheet_name=_TITER_SHEET, header=None)
        df = raw.iloc[1:].reset_index(drop=True)  # row 0 = header labels (FFA groups)

        environment = self._environment()
        publication = Publication(
            pubmed_id=_PMID,
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{_PMID}/",
            doi=_DOI,
            doi_url=f"https://doi.org/{_DOI}",
        )
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="BY4741",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )

        # Measured WT reference: the wt BY4741 row's 5 FFA titers (constant across records).
        wt_rows = df[df[0].astype(str).str.strip().isin(_WT_LABELS)]
        if len(wt_rows) != 1:
            raise RuntimeError(
                f"expected exactly 1 wild-type row, found {len(wt_rows)}: "
                f"{wt_rows[0].tolist()}"
            )
        wt_phenotype = self._phenotype_from_row(wt_rows.iloc[0])
        reference = MetaboliteExperimentReference(
            dataset_name=self.name,
            genome_reference=genome_reference,
            environment_reference=environment.model_copy(),
            phenotype_reference=wt_phenotype,
        )
        ref_dump = reference.model_dump()
        pub_dump = publication.model_dump()

        # Every record is built BEFORE anything is written.
        experiments: list[MetaboliteExperiment] = []
        summary: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            label = str(row[0]).strip()
            deleted = self._decode_genotype(label, code_to_gene)
            if not deleted:  # true WT -> reference only, not an experiment record
                continue
            genotype = self._genotype(deleted)
            experiments.append(
                MetaboliteExperiment(
                    dataset_name=self.name,
                    genotype=genotype,
                    environment=environment,
                    phenotype=self._phenotype_from_row(row),
                )
            )
            summary.append(
                {
                    "genotype": label,
                    "n_deletions": len(deleted),
                    "genes": ";".join(genotype.systematic_gene_names),
                }
            )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        with env.begin(write=True) as txn:
            for experiment in tqdm(experiments, desc="xue2025"):
                txn.put(
                    f"{idx}".encode(),
                    pickle.dumps(
                        {
                            "experiment": experiment.model_dump(),
                            "reference": ref_dump,
                            "publication": pub_dump,
                        }
                    ),
                )
                idx += 1
        env.close()
        pd.DataFrame(summary).to_csv(
            osp.join(self.preprocess_dir, "data.csv"), index=False
        )
        log.info(
            "Wrote %d Xue2025 FFA metabolite experiments to LMDB (+ 1 measured WT reference)",
            idx,
        )

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


def main() -> None:
    """Build/load the dataset for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(data_root, "data/torchcell/ffa_xue2025")
    dataset = FattyAcidXue2025Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()
