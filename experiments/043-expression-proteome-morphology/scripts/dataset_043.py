# experiments/043-expression-proteome-morphology/scripts/dataset_043.py
# [[experiments.043-expression-proteome-morphology.scripts.dataset_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/dataset_043
"""A torch ``Dataset`` over the 043 raw store, one sample per served record.

The store is the raw LMDB written by ``build_043_store.py`` (``$DATA_ROOT/data/torchcell/
experiments/043-expression-proteome-morphology/raw_043/raw/lmdb``, keys ``data_<i>``, one
JSON record per value: ``experiment`` + ``experiment_reference``). Records are addressed per
dataset through ``raw/raw_dataset_name_index.json``, the head-only index that
``data_shapes_043.py`` writes (the processed store's index is in processed order, not raw
order).

Every sample is a dict over ONE shared gene index, the 6,607 genes of the store's
``processed/gene_set.json`` (``SCerevisiaeGenome.gene_set``, sorted):

* ``gene_index``: LongTensor of the marked genes (perturbed, or carrying a non-reference
  allele), indices into the 6,607 set; ``pert_type``: one ``PertType`` per marked gene
  (deletion, variant, lof, absent, promoter); ``allele_parent``: one parent token per
  marked gene (segregants; -1 elsewhere).
* ``promoter_seq`` / ``scaffold`` (Vaishnav 2022): the released promoter sequence verbatim
  and the scaffold name; the reporter has no node, so ``gene_index`` is empty.
* ``env_id`` (sha256 of the environment JSON, first 15 hex digits as an int; ``env_key``
  holds the hex), ``dataset_id`` (position in ``config.datasets``), ``exp_type_id``
  (``ExpType``, the schema phenotype class the record serializes), ``y_scale`` (``YScale``,
  the released scale), ``strain``, ``raw_index``.
* ``y``: FloatTensor (6,607,) over the gene index, NaN where the record measures nothing;
  ``y_scalar`` (Vaishnav: the sort-seq expression level, NaN elsewhere); ``y_global``
  (Ohya 2005: the 281 CalMorph features in sorted key order, NaN elsewhere). Values are the
  RELEASED values, no conversion (``y_scale`` says what they are). With
  ``include_reference`` the record's ``experiment_reference`` label is returned the same way
  as ``y_ref`` / ``y_ref_scalar``.

Label keys that are not systematic ORFs are mapped at read time, never at load time:
Messner 2023 UniProt accession -> ORF (``build_uniprot_to_orf_map``), Nadal-Ribelles 2025
standard name -> ORF (``build_orf_to_gene_name_map`` inverted); a perturbed gene name
outside the index is also tried through the standard-name map (``ENA1`` -> ``YDR040C``),
which is how a ``released_identifier`` perturbation (the released symbol in
``systematic_gene_name``, ``released_kind`` deletion) finds its ORF. A key with no index
position is counted (``n_label_keys_outside_index``, ``n_marked_outside_index``), never
guessed, and the record is kept; with a representation, an engineered deletion of a gene
with no position is an ``off_index_gene`` extra token (gap ``NO_S288C_KEY``).

Genotype classes:

* ``Genotype`` (engineered deletions, 1 or 2 per record): every perturbation marks its
  gene as DELETION.
* ``Genotype`` + ``natural_isolate`` (Caudal 2024, Muenzner 2024): the marking rule is
  ``config.isolate_marking``. ``all_gene_level`` (default) marks ``sequence_variant`` as
  VARIANT, ``natural_loss_of_function`` as LOF, ``natural_gene_absence`` as ABSENT and the
  HO ``deletion`` as DELETION; ``lof_absent_deletion`` leaves the sequence variants
  unmarked. ``natural_gene_presence`` names a pangenome ORF with no node and is counted in
  ``n_accessory_present``.
* ``SegregantGenotype`` (Boocock 2025, N'Guessan 2025): the definite rule. Each gene's
  midpoint (SGD R64 GFF coordinates; the block positions are the released marker positions
  on the S288C reference, taken here as the same coordinate system) is looked up in the
  record's haplotype blocks on its chromosome. The gene is marked VARIANT with
  ``allele_parent`` = that parent when the covering block calls the NON-REFERENCE parent
  with posterior >= ``config.segregant_posterior_min`` (0.9). The reference parent is the
  one with ``peter_strain_id`` None (BY); for a cross with no such parent (Boocock crosses
  3004 and B) parent 1 stands as the reference-like background and parent-2 genes are
  marked, which ``allele_parent`` and ``parents`` make explicit. Genes in an unassigned gap
  between blocks, or under a block below the posterior threshold, are not marked; their
  counts are returned (``segregant_coverage``). Probabilistic marking is a later step.

The full input (``config.representation`` set): ``sample["representation"]`` is the
``RecordRepresentation`` of ``torchcell.data.representation_043`` built by
``representation_043.resolve_genotype``: the reference context (S288C, or the isolate under
route A), per gene the key ids of the four composite encoders (-1 = typed gap), the
change-type vector under the precedence rule, the allele parent, the operator mask, the
position features (chromosome index, relative midpoint, chromosome copy number, gene copy
number), the extra off-graph tokens (accessory ORFs, the Vaishnav reporter) and every typed
gap. Segregants there use whole-gene containment in a confident run of one parent, which
supersedes the midpoint rule of the prototype fields above (``gene_index`` / ``pert_type``
/ ``allele_parent`` are kept unchanged for the scripts that read them). Vectors are read
with ``representation_043.gene_vectors`` through the lazy h5 readers; a missing vector
raises.

No fallbacks: an unexpected perturbation type, experiment type, parent name, dataset name
or label name raises.

    python experiments/043-expression-proteome-morphology/scripts/dataset_043.py
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import os.path as osp
import sys
from enum import IntEnum, StrEnum
from typing import Any

import lmdb
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from torch.utils.data import Dataset

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import representation_043 as rep043  # noqa: E402
from store_version_043 import active as active_store  # noqa: E402

from torchcell.data.representation_043 import RepresentationConfig  # noqa: E402
from torchcell.datasets.scerevisiae.messner2023 import (  # noqa: E402
    _sgd_gff_path,
    build_orf_to_gene_name_map,
    build_uniprot_to_orf_map,
)

EXPERIMENT = "043-expression-proteome-morphology"
RAW_INDEX_NAME = "raw_dataset_name_index.json"
N_CALMORPH = 281


class PertType(IntEnum):
    """How a marked gene differs from the reference strain."""

    DELETION = 0
    VARIANT = 1
    LOF = 2
    ABSENT = 3
    PROMOTER = 4


class ExpType(IntEnum):
    """The schema phenotype class a record serializes (the ontology leaf)."""

    MICROARRAY_EXPRESSION = 0  # MicroarrayExpressionPhenotype
    MICROARRAY_EXPRESSION_SCORE = 1  # MicroarrayExpressionScorePhenotype
    SINGLE_CELL_EXPRESSION = 2  # SingleCellExpressionPhenotype (umi_count)
    PSEUDOBULK_EXPRESSION = 3  # PseudobulkExpressionPhenotype (log2 fc)
    RNASEQ_EXPRESSION = 4  # RNASeqExpressionPhenotype (TPM)
    PROTEIN_ABUNDANCE = 5  # ProteinAbundancePhenotype
    PROTEIN_TURNOVER = 6  # ProteinTurnoverPhenotype
    POOLED_SORT_SEQ_EXPRESSION = 7  # PooledSortSeqExpressionPhenotype
    CALMORPH = 8  # CalMorphPhenotype


class YScale(IntEnum):
    """The released scale of ``y``."""

    LOG2_RATIO = 0
    SCORE = 1
    UMI_COUNT = 2
    TPM = 3
    LINEAR_ABUNDANCE = 4
    LOG_ABUNDANCE = 5
    RATE_PER_HOUR = 6
    SORT_BIN_INDEX = 7
    CALMORPH = 8


class IsolateMarking(StrEnum):
    """Which isolate perturbation types mark a gene."""

    ALL_GENE_LEVEL = "all_gene_level"
    LOF_ABSENT_DELETION = "lof_absent_deletion"


# experiment_type (record field) -> (ExpType, label_name the class carries).
EXP_TYPES: dict[str, tuple[ExpType, str]] = {
    "microarray_expression": (ExpType.MICROARRAY_EXPRESSION, "expression_log2_ratio"),
    "microarray_expression_score": (
        ExpType.MICROARRAY_EXPRESSION_SCORE,
        "expression_score",
    ),
    "pseudobulk_umi_count": (ExpType.SINGLE_CELL_EXPRESSION, "umi_count"),
    "segregant_expression": (ExpType.SINGLE_CELL_EXPRESSION, "umi_count"),
    "pseudobulk_expression": (ExpType.PSEUDOBULK_EXPRESSION, "expression_log2_ratio"),
    "rnaseq_expression": (ExpType.RNASEQ_EXPRESSION, "expression_tpm"),
    "protein_abundance": (ExpType.PROTEIN_ABUNDANCE, "protein_abundance"),
    "protein_turnover": (ExpType.PROTEIN_TURNOVER, "degradation_rate"),
    "pooled_sort_seq_expression": (
        ExpType.POOLED_SORT_SEQ_EXPRESSION,
        "expression_level",
    ),
    "calmorph": (ExpType.CALMORPH, "calmorph"),
}

# dataset -> released scale (the record's measurement_type names it where present;
# the two-channel arrays and the CalMorph table carry none).
Y_SCALE: dict[str, YScale] = {
    "MicroarrayKemmeren2014Dataset": YScale.LOG2_RATIO,
    "SmMicroarraySameith2015Dataset": YScale.LOG2_RATIO,
    "DmMicroarraySameith2015Dataset": YScale.LOG2_RATIO,
    "MicroarrayScoreHu2007Dataset": YScale.SCORE,
    "TfDeletionPseudobulkJackson2020Dataset": YScale.UMI_COUNT,
    "NadalRibellesPerturbSeq2025Dataset": YScale.LOG2_RATIO,
    "ProteomeMessner2023Dataset": YScale.LINEAR_ABUNDANCE,
    "ProteomeZelezniak2018Dataset": YScale.LOG_ABUNDANCE,
    "ScmdOhya2005Dataset": YScale.CALMORPH,
    "CaudalPanTranscriptome2024Dataset": YScale.TPM,
    "ProteomeMuenzner2024Dataset": YScale.LINEAR_ABUNDANCE,
    "ProteinTurnoverMuenzner2024Dataset": YScale.RATE_PER_HOUR,
    "Boocock2025Dataset": YScale.UMI_COUNT,
    "Nguessan2025Dataset": YScale.UMI_COUNT,
    "PooledPromoterExpressionVaishnav2022Dataset": YScale.SORT_BIN_INDEX,
}
UNIPROT_KEYED = {"ProteomeMessner2023Dataset"}
STANDARD_NAME_KEYED = {"NadalRibellesPerturbSeq2025Dataset"}

#: ``released_kind`` values of a ``released_identifier`` perturbation marked as DELETION
#: (``rep043`` maps the same kinds through ``RELEASED_KIND_TO_CHANGE``).
RELEASED_DELETION_KINDS = {"deletion", "truncation"}
DELETION_TYPES = {
    "kanmx_deletion",
    "sga_kanmx_deletion",
    "sga_natmx_deletion",
    "marker_deletion",
    "barcoded_kanmx_deletion",
    "deletion",
}

# The owner's training set: every gene-level readout. Sameith 2015 singles are the
# Kemmeren 2014 arrays (82 of 82 vectors identical) and are left out; Ohya 2005 is the
# morphology hold-out, probed from the learned cell state, never a training target.
TRAINING_DATASETS: list[str] = [
    "MicroarrayKemmeren2014Dataset",
    "DmMicroarraySameith2015Dataset",
    "MicroarrayScoreHu2007Dataset",
    "TfDeletionPseudobulkJackson2020Dataset",
    "NadalRibellesPerturbSeq2025Dataset",
    "CaudalPanTranscriptome2024Dataset",
    "Boocock2025Dataset",
    "Nguessan2025Dataset",
    "PooledPromoterExpressionVaishnav2022Dataset",
    "ProteomeMessner2023Dataset",
    "ProteomeZelezniak2018Dataset",
    "ProteomeMuenzner2024Dataset",
    "ProteinTurnoverMuenzner2024Dataset",
]
HOLDOUT_DATASETS: list[str] = ["ScmdOhya2005Dataset"]
DUPLICATE_DATASETS: list[str] = ["SmMicroarraySameith2015Dataset"]


class Dataset043Config(BaseModel):
    """Subset choices of the 043 dataset."""

    data_root: str = Field(description="$DATA_ROOT")
    datasets: list[str] = Field(
        default_factory=lambda: list(TRAINING_DATASETS),
        description="dataset classes to serve, in dataset_id order",
    )
    label_keys: dict[str, str] = Field(
        default_factory=dict,
        description="dataset -> phenotype field to read as y; default is the record's "
        "label_name (e.g. 'half_life' for the turnover set instead of degradation_rate)",
    )
    segregant_posterior_min: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="blocks below this posterior do not mark",
    )
    isolate_marking: IsolateMarking = IsolateMarking.ALL_GENE_LEVEL
    include_reference: bool = Field(
        default=False, description="also return the experiment_reference label as y_ref"
    )
    segregant_parent_vocab: list[str] = Field(
        default_factory=lambda: [
            "BY",
            "RM",
            "YJM145",
            "YPS163",
            "CBS2888",
            "YJM981",
            "BY4741",
            "RM11-1a",
        ],
        description="parent names as the records carry them; allele_parent indexes this",
    )
    max_records_per_dataset: int | None = Field(
        default=None, description="truncate each dataset (debugging)"
    )
    representation: RepresentationConfig | None = Field(
        default=None,
        description="when set, every sample carries its full input representation "
        "(representation_043.py --stage build must have run)",
    )

    @property
    def store_root(self) -> str:
        """``$DATA_ROOT/data/torchcell/experiments/<EXPERIMENT>/<raw tag>`` of the active
        store generation (``store_version_043``: ``raw_043`` v1, ``raw_043_v2`` v2).
        """
        return osp.join(
            self.data_root,
            "data/torchcell/experiments",
            EXPERIMENT,
            active_store().raw_tag,
        )


def gene_coordinates(gff_path: str, genes: set[str]) -> dict[str, tuple[str, int, int]]:
    """ORF -> (chromosome, start, end) from the SGD GFF ``gene`` features."""
    coords: dict[str, tuple[str, int, int]] = {}
    with open(gff_path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != "gene":
                continue
            attrs = dict(kv.split("=", 1) for kv in cols[8].split(";") if "=" in kv)
            gid = attrs["ID"]
            if gid in genes:
                coords[gid] = (cols[0], int(cols[3]), int(cols[4]))
    missing = genes - set(coords)
    if missing:
        raise KeyError(
            f"{len(missing)} genes have no GFF gene feature: {sorted(missing)[:5]}"
        )
    return coords


class Raw043Dataset(Dataset):
    """One sample per raw record of the chosen datasets (see the module docstring)."""

    def __init__(self, config: Dataset043Config) -> None:
        """Load the index, the gene set, the coordinate table and the id maps."""
        self.config = config
        root = config.store_root
        self.lmdb_path = osp.join(root, "raw", "lmdb")
        index_path = osp.join(root, "raw", RAW_INDEX_NAME)
        if not osp.exists(index_path):
            raise FileNotFoundError(
                f"{index_path} missing; run data_shapes_043.py once to write it"
            )
        with open(index_path) as f:
            index = json.load(f)["index"]
        for name in config.datasets:
            if name not in index:
                raise KeyError(
                    f"{name} is not in the raw store; datasets: {sorted(index)}"
                )
            if name not in Y_SCALE:
                raise KeyError(f"{name} has no YScale entry")
        self.items: list[tuple[str, int]] = []
        self.dataset_id = {name: i for i, name in enumerate(config.datasets)}
        for name in config.datasets:
            ids = index[name]
            if config.max_records_per_dataset is not None:
                ids = ids[: config.max_records_per_dataset]
            self.items.extend((name, i) for i in ids)

        with open(osp.join(root, "processed", "gene_set.json")) as f:
            self.genes: list[str] = json.load(f)
        if self.genes != sorted(self.genes):
            raise ValueError(
                "gene_set.json is not sorted; the gene index must be stable"
            )
        self.gene_to_idx = {g: i for i, g in enumerate(self.genes)}
        self.n_genes = len(self.genes)
        self.coords = gene_coordinates(_sgd_gff_path(config.data_root), set(self.genes))
        self.up2orf = build_uniprot_to_orf_map(config.data_root)
        self.name2orf: dict[str, str] = {}
        for orf, name in build_orf_to_gene_name_map(config.data_root).items():
            self.name2orf.setdefault(name.upper(), orf)
        self.parent_id = {p: i for i, p in enumerate(config.segregant_parent_vocab)}
        self.calmorph_features: list[str] | None = None
        self._env: lmdb.Environment | None = None
        self.resolver = (
            None
            if config.representation is None
            else rep043.load_resolver(
                config.data_root,
                config.representation,
                config.segregant_parent_vocab,
                self.genes,
                _sgd_gff_path(config.data_root),
            )
        )

    # ------------------------------------------------------------------ LMDB access

    def _txn(self) -> lmdb.Transaction:
        if (
            self._env is None
        ):  # opened lazily so DataLoader workers get their own handle
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=512,
            )
        return self._env.begin()

    def record(self, raw_index: int) -> dict[str, Any]:
        """The raw record ``data_<raw_index>``."""
        with self._txn() as txn:
            value = txn.get(f"data_{raw_index}".encode())
        if value is None:
            raise KeyError(f"data_{raw_index} missing from {self.lmdb_path}")
        return json.loads(value)

    def __len__(self) -> int:
        """Records served."""
        return len(self.items)

    # ------------------------------------------------------------------ helpers

    def _gene_idx(self, name: str) -> int | None:
        idx = self.gene_to_idx.get(name)
        if idx is None:
            orf = self.name2orf.get(name.upper())
            if orf is not None:
                idx = self.gene_to_idx.get(orf)
        return idx

    def _label_vector(
        self, name: str, label: dict[str, float | int | None]
    ) -> tuple[torch.Tensor, int]:
        """Dict label -> (6,607,) tensor with NaN; also the count of keys outside the index."""
        y = torch.full((self.n_genes,), float("nan"))
        outside = 0
        for key, value in label.items():
            if name in UNIPROT_KEYED:
                orf = self.up2orf.get(key)
                idx = None if orf is None else self.gene_to_idx.get(orf)
            elif name in STANDARD_NAME_KEYED:
                orf = self.name2orf.get(key.upper())
                idx = None if orf is None else self.gene_to_idx.get(orf)
            else:
                idx = self.gene_to_idx.get(key)
            if idx is None:
                outside += 1
                continue
            if value is not None:
                y[idx] = float(value)
        return y, outside

    def _calmorph_vector(self, label: dict[str, float]) -> torch.Tensor:
        if self.calmorph_features is None:
            self.calmorph_features = sorted(label)
            if len(self.calmorph_features) != N_CALMORPH:
                raise ValueError(
                    f"CalMorph record has {len(label)} features, expected {N_CALMORPH}"
                )
        if sorted(label) != self.calmorph_features:
            raise ValueError("CalMorph feature set differs between records")
        return torch.tensor([float(label[k]) for k in self.calmorph_features])

    def _mark_perturbations(
        self, perturbations: list[dict[str, Any]], is_isolate: bool
    ) -> tuple[list[int], list[int], int, int, dict[str, str] | None]:
        """Marked gene indices and types from a perturbation list.

        Returns (gene_index, pert_type, n_marked_outside_index, n_accessory_present,
        promoter) where ``promoter`` is the Vaishnav construct or None.
        """
        gene_index: list[int] = []
        pert_type: list[int] = []
        outside = 0
        accessory = 0
        promoter: dict[str, str] | None = None
        for p in perturbations:
            t = p["perturbation_type"]
            if t in DELETION_TYPES:
                kind = PertType.DELETION
            elif t == "released_identifier":
                # the source's own gene name; the change is its released_kind
                if p["released_kind"] not in RELEASED_DELETION_KINDS:
                    raise ValueError(
                        f"released_identifier {p['systematic_gene_name']!r} has "
                        f"released_kind {p['released_kind']!r}"
                    )
                kind = PertType.DELETION
            elif t == "sequence_variant":
                if not is_isolate:
                    raise ValueError("sequence_variant outside a natural isolate")
                if self.config.isolate_marking == IsolateMarking.LOF_ABSENT_DELETION:
                    continue
                kind = PertType.VARIANT
            elif t == "natural_loss_of_function":
                kind = PertType.LOF
            elif t == "natural_gene_absence":
                kind = PertType.ABSENT
            elif t == "natural_gene_presence":
                accessory += 1  # a pangenome ORF: no node in the 6,607 index
                continue
            elif t == "variable_promoter_reporter":
                if promoter is not None:
                    raise ValueError("two promoter reporters in one genotype")
                promoter = {
                    "sequence": p["promoter_sequence"],
                    "scaffold": p["scaffold_name"],
                    "variant_id": p["promoter_variant_id"],
                    "variable_region": f"{p['variable_region_start']}-{p['variable_region_end']}",
                }
                continue
            else:
                raise ValueError(f"unexpected perturbation_type {t!r}")
            idx = self._gene_idx(p["systematic_gene_name"])
            if idx is None:
                outside += 1
                continue
            gene_index.append(idx)
            pert_type.append(int(kind))
        return gene_index, pert_type, outside, accessory, promoter

    def _mark_segregant(
        self, genotype: dict[str, Any]
    ) -> tuple[list[int], list[int], list[int], dict[str, Any]]:
        """The definite segregant rule (module docstring)."""
        parents = (genotype["parent_1"], genotype["parent_2"])
        for parent in parents:
            if parent["name"] not in self.parent_id:
                raise KeyError(
                    f"segregant parent {parent['name']!r} not in segregant_parent_vocab"
                )
        reference = [
            i for i, p in enumerate(parents, start=1) if p["peter_strain_id"] is None
        ]
        if len(reference) > 1:
            raise ValueError("both parents have peter_strain_id None")
        reference_parent = reference[0] if reference else 1
        non_reference = 2 if reference_parent == 1 else 1

        by_chrom: dict[str, tuple[list[int], list[dict[str, Any]]]] = {}
        for block in genotype["blocks"]:  # validated ascending and non-overlapping
            starts, blocks = by_chrom.setdefault(block["chromosome"], ([], []))
            starts.append(block["start"])
            blocks.append(block)

        gene_index: list[int] = []
        pert_type: list[int] = []
        allele_parent: list[int] = []
        n_unassigned = n_low_posterior = n_reference = 0
        threshold = self.config.segregant_posterior_min
        for gene in self.genes:
            chrom, start, end = self.coords[gene]
            mid = (start + end) // 2
            if chrom not in by_chrom:
                n_unassigned += 1
                continue
            starts, blocks = by_chrom[chrom]
            k = bisect.bisect_right(starts, mid) - 1
            if k < 0 or mid > blocks[k]["end"]:
                n_unassigned += 1
                continue
            block = blocks[k]
            if block["posterior"] < threshold:
                n_low_posterior += 1
                continue
            if block["parent"] != non_reference:
                n_reference += 1
                continue
            gene_index.append(self.gene_to_idx[gene])
            pert_type.append(int(PertType.VARIANT))
            allele_parent.append(self.parent_id[parents[non_reference - 1]["name"]])
        coverage = {
            "n_marked_non_reference": len(gene_index),
            "n_reference_allele": n_reference,
            "n_low_posterior": n_low_posterior,
            "n_unassigned": n_unassigned,
            "reference_parent": reference_parent,
            "parents": (parents[0]["name"], parents[1]["name"]),
            "cross": genotype["cross"],
            "n_blocks": len(genotype["blocks"]),
        }
        return gene_index, pert_type, allele_parent, coverage

    # ------------------------------------------------------------------ samples

    def __getitem__(self, i: int) -> dict[str, Any]:
        """The sample dict of record i (module docstring)."""
        name, raw_index = self.items[i]
        record = self.record(raw_index)
        ex = record["experiment"]
        if ex["dataset_name"] != name:
            raise ValueError(
                f"data_{raw_index} is {ex['dataset_name']}, index says {name}"
            )
        exp_type, class_label = EXP_TYPES[ex["experiment_type"]]
        phenotype = ex["phenotype"]
        if phenotype["label_name"] != class_label:
            raise ValueError(
                f"{name}: label_name {phenotype['label_name']!r} does not match "
                f"{ex['experiment_type']} ({class_label})"
            )
        label_key = self.config.label_keys.get(name, class_label)
        genotype = ex["genotype"]
        environment = ex["environment"]
        env_hex = hashlib.sha256(
            json.dumps(environment, sort_keys=True).encode()
        ).hexdigest()

        sample: dict[str, Any] = {
            "dataset": name,
            "dataset_id": self.dataset_id[name],
            "raw_index": raw_index,
            "exp_type_id": int(exp_type),
            "y_scale": int(Y_SCALE[name]),
            "env_id": int(env_hex[:15], 16),
            "env_key": env_hex[:12],
            "media": environment["media"]["name"],
            "allele_parent": [],
            "promoter_seq": None,
            "scaffold": None,
            "strain": None,
            "segregant_coverage": None,
            "n_accessory_present": 0,
            "n_marked_outside_index": 0,
        }

        # ---- genotype -> marked genes
        if "perturbations" in genotype:
            isolate = genotype.get("natural_isolate")
            gene_index, pert_type, outside, accessory, promoter = (
                self._mark_perturbations(
                    genotype["perturbations"], is_isolate=isolate is not None
                )
            )
            sample["genotype_class"] = (
                "Genotype+natural_isolate" if isolate else "Genotype"
            )
            sample["strain"] = isolate["isolate"] if isolate else None
            sample["n_marked_outside_index"] = outside
            sample["n_accessory_present"] = accessory
            if promoter is not None:
                sample["promoter_seq"] = promoter["sequence"]
                sample["scaffold"] = promoter["scaffold"]
                sample["promoter_variant_id"] = promoter["variant_id"]
                sample["promoter_variable_region"] = promoter["variable_region"]
            sample["allele_parent"] = [-1] * len(gene_index)
        else:
            gene_index, pert_type, allele_parent, coverage = self._mark_segregant(
                genotype
            )
            sample["genotype_class"] = "SegregantGenotype"
            sample["strain"] = f"{genotype['cross']}:{genotype['segregant_id']}"
            sample["segregant_coverage"] = coverage
            sample["allele_parent"] = allele_parent
        sample["gene_index"] = torch.tensor(gene_index, dtype=torch.long)
        sample["pert_type"] = torch.tensor(pert_type, dtype=torch.long)
        sample["allele_parent"] = torch.tensor(
            sample["allele_parent"], dtype=torch.long
        )

        # ---- phenotype -> y over the shared gene index
        value = phenotype[label_key]
        y = torch.full((self.n_genes,), float("nan"))
        y_scalar = float("nan")
        y_global = torch.full((N_CALMORPH,), float("nan"))
        outside_keys = 0
        if exp_type == ExpType.CALMORPH:
            y_global = self._calmorph_vector(value)
        elif isinstance(value, dict):
            y, outside_keys = self._label_vector(name, value)
        elif value is not None:
            y_scalar = float(value)
        sample["y"] = y
        sample["y_scalar"] = y_scalar
        sample["y_global"] = y_global
        sample["n_label_keys"] = len(value) if isinstance(value, dict) else 1
        sample["n_label_keys_outside_index"] = outside_keys
        sample["label_key"] = label_key

        if self.resolver is not None:
            sample["representation"] = rep043.resolve_genotype(
                self.resolver, genotype, self._gene_idx
            )

        if self.config.include_reference:
            ref_value = record["experiment_reference"]["phenotype_reference"][label_key]
            y_ref = torch.full((self.n_genes,), float("nan"))
            y_ref_scalar = float("nan")
            if exp_type == ExpType.CALMORPH:
                pass  # the CalMorph reference is a global vector; not served here
            elif isinstance(ref_value, dict):
                y_ref, _ = self._label_vector(name, ref_value)
            elif ref_value is not None:
                y_ref_scalar = float(ref_value)
            sample["y_ref"] = y_ref
            sample["y_ref_scalar"] = y_ref_scalar
        return sample


def collate_043(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Batch: per-sample tensors stacked, marked genes concatenated with a batch vector."""
    gene_index = torch.cat([s["gene_index"] for s in samples])
    batch = torch.cat(
        [
            torch.full((len(s["gene_index"]),), b, dtype=torch.long)
            for b, s in enumerate(samples)
        ]
    )
    out: dict[str, Any] = {
        "gene_index": gene_index,
        "gene_batch": batch,
        "pert_type": torch.cat([s["pert_type"] for s in samples]),
        "allele_parent": torch.cat([s["allele_parent"] for s in samples]),
        "n_marked": torch.tensor(
            [len(s["gene_index"]) for s in samples], dtype=torch.long
        ),
        "promoter_seq": [s["promoter_seq"] for s in samples],
        "scaffold": [s["scaffold"] for s in samples],
        "env_id": torch.tensor([s["env_id"] for s in samples], dtype=torch.long),
        "dataset_id": torch.tensor(
            [s["dataset_id"] for s in samples], dtype=torch.long
        ),
        "exp_type_id": torch.tensor(
            [s["exp_type_id"] for s in samples], dtype=torch.long
        ),
        "y_scale": torch.tensor([s["y_scale"] for s in samples], dtype=torch.long),
        "raw_index": torch.tensor([s["raw_index"] for s in samples], dtype=torch.long),
        "y": torch.stack([s["y"] for s in samples]),
        "y_mask": torch.stack([~torch.isnan(s["y"]) for s in samples]),
        "y_scalar": torch.tensor([s["y_scalar"] for s in samples]),
        "y_global": torch.stack([s["y_global"] for s in samples]),
        "strain": [s["strain"] for s in samples],
    }
    if "representation" in samples[0]:
        out["representation"] = [s["representation"] for s in samples]
    if "y_ref" in samples[0]:
        out["y_ref"] = torch.stack([s["y_ref"] for s in samples])
        out["y_ref_scalar"] = torch.tensor([s["y_ref_scalar"] for s in samples])
    return out


def _describe(sample: dict[str, Any]) -> str:
    y_meas = int((~torch.isnan(sample["y"])).sum())
    pert = sample["pert_type"].tolist()
    counts = {PertType(k).name: pert.count(k) for k in sorted(set(pert))}
    extra = ""
    if sample["segregant_coverage"] is not None:
        c = sample["segregant_coverage"]
        extra = (
            f" | segregant {c['cross']} {c['parents']} ref parent {c['reference_parent']}: "
            f"marked {c['n_marked_non_reference']}, ref allele {c['n_reference_allele']}, "
            f"low posterior {c['n_low_posterior']}, unassigned {c['n_unassigned']}, "
            f"blocks {c['n_blocks']}"
        )
    if sample["promoter_seq"] is not None:
        extra = (
            f" | promoter {len(sample['promoter_seq'])} nt, scaffold {sample['scaffold']}, "
            f"variable {sample['promoter_variable_region']}, y_scalar {sample['y_scalar']:.3f}"
        )
    if not torch.isnan(sample["y_global"]).all():
        extra = f" | y_global {tuple(sample['y_global'].shape)} measured {int((~torch.isnan(sample['y_global'])).sum())}"
    return (
        f"{sample['dataset']} data_{sample['raw_index']} [{sample['genotype_class']}] "
        f"exp_type {ExpType(sample['exp_type_id']).name} y_scale {YScale(sample['y_scale']).name} "
        f"env {sample['env_key']} | gene_index {tuple(sample['gene_index'].shape)} {counts} "
        f"outside index {sample['n_marked_outside_index']} accessory {sample['n_accessory_present']} "
        f"| y {tuple(sample['y'].shape)} measured {y_meas} of {sample['n_label_keys']} keys "
        f"(outside index {sample['n_label_keys_outside_index']})" + extra
    )


if __name__ == "__main__":
    load_dotenv()
    # BREAKPOINT: config. Change the subset here (datasets, posterior threshold, marking rule).
    config = Dataset043Config(
        data_root=os.environ["DATA_ROOT"],
        datasets=TRAINING_DATASETS + HOLDOUT_DATASETS + DUPLICATE_DATASETS,
        max_records_per_dataset=3,
        include_reference=True,
    )
    dataset = Raw043Dataset(config)
    print(
        f"{len(dataset)} samples over {len(config.datasets)} datasets; gene index "
        f"{dataset.n_genes}; isolate marking {config.isolate_marking}; "
        f"segregant posterior >= {config.segregant_posterior_min}"
    )
    first_of: dict[str, dict[str, Any]] = {}
    for i in range(len(dataset)):
        # BREAKPOINT: one sample. `sample` is the dict the model consumes for record i.
        sample = dataset[i]
        print(_describe(sample))
        first_of.setdefault(sample["dataset"], sample)

    # BREAKPOINT: one mixed batch, one sample per dataset, through the collate function.
    batch = collate_043(list(first_of.values()))
    print("\nbatch:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:14s} {str(value.dtype):14s} {tuple(value.shape)}")
        else:
            print(f"  {key:14s} list[{len(value)}] e.g. {str(value[0])[:40]!r}")
    print(
        f"  y measured per row: {batch['y_mask'].sum(1).tolist()}; "
        f"marked genes per row: {batch['n_marked'].tolist()}"
    )

    # BREAKPOINT: the full input of one record per dataset (keys; vectors read lazily).
    from torchcell.data.representation_043 import GapKind

    rep_config = config.model_copy(
        update={
            "representation": RepresentationConfig(route="A"),
            "max_records_per_dataset": 1,
            "include_reference": False,
        }
    )
    rep_dataset = Raw043Dataset(rep_config)
    for i in range(len(rep_dataset)):
        r = rep_dataset[i]["representation"]
        gaps = {
            GapKind(k).name: int((r.key_gap == k).sum())
            for k in sorted(set(r.key_gap.ravel().tolist()))
            if k
        }
        print(
            f"{rep_dataset.items[i][0]}: {r.genotype_class} context {r.context_id} "
            f"route {r.route} | marks {len(r.marked)} operator {int(r.operator_mask.sum())} "
            f"| extra tokens {len(r.extra_tokens)} | gap slots {gaps}"
        )

    # BREAKPOINT: the other isolate marking rule on the same Caudal record.
    lof_config = config.model_copy(
        update={
            "isolate_marking": IsolateMarking.LOF_ABSENT_DELETION,
            "datasets": ["CaudalPanTranscriptome2024Dataset"],
            "max_records_per_dataset": 1,
        }
    )
    lof_sample = Raw043Dataset(lof_config)[0]
    print("\nCaudal under lof_absent_deletion:", _describe(lof_sample))
