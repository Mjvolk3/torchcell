# torchcell/datasets/scerevisiae/caudal2024
# [[torchcell.datasets.scerevisiae.caudal2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/caudal2024
# Test file: tests/torchcell/datasets/scerevisiae/test_caudal2024.py
"""Caudal 2024 S. cerevisiae natural-isolate pan-transcriptome (RNA-seq, WS10).

Caudal et al. 2024 (Nat. Genet. 56:1278, doi:10.1038/s41588-024-01769-9) measured the
whole-transcriptome expression of ~1000 natural S. cerevisiae isolates (the 1002-genomes
panel), each mapped to its OWN genome. We model every isolate as a PERTURBATION SET off
the S288C reference: its genotype is the full set of sequence/copy-number differences from
S288C, and its phenotype is the absolute per-gene expression (TPM + raw counts).

GENOTYPE (three natural-variation perturbation families, all off-graph pointers):
  - ``SequenceVariantPerturbation`` -- a native S288C gene whose isolate allele differs
    from the S288C reference sequence. The reference allele is reconstructed EXACTLY in
    Peter's representation: the coordinate in a gene's first FASTA header
    (``chromosomeN:start-end +/-``) slices the SGD R64 chromosome (reverse-complemented on
    the minus strand); an isolate has a variant iff its sequence != that slice. Source =
    Peter et al. 2018 ``allReferenceGenesWithSNPsAndIndelsInferred`` (gene-keyed store).
  - ``NaturalGenePresencePerturbation`` for an ACCESSORY (non-reference) pangenome ORF that
    is PRESENT in the isolate (AXIS-1 state=present, natural insertion; ``copy_number`` from
    the copy-number matrix, default 1.0).
  - ``NaturalGeneAbsencePerturbation`` for a CORE (reference) ORF that is ABSENT in the
    isolate (AXIS-1 state=absent, natural deletion). EVERY absence is recorded -- an ORF
    with no S288C systematic name is kept by its pangenome id, never dropped (a dropped
    absence would wrongly reconstruct the gene as present).
These are NATURAL genome-content edits off S288C, distinct from engineered CNV (true dosage
of a PRESENT gene stays on the copy-number axis). Core vs accessory is Peter's
presence/absence matrix over the 1011 isolates: an ORF present in >= 99% of isolates is core.

PHENOTYPE (``RNASeqExpressionPhenotype``): per-isolate ``expression_tpm`` +
``expression_count`` for the genes that isolate carries (a gene absent from an isolate is
KEY-ABSENT, never 0). ``measurement_type = "rnaseq_tpm"``. The shared
``phenotype_reference`` is the POPULATION MEAN over the 943 built isolates (mean TPM /
rounded mean count per gene) -- an absolute WT-equivalent baseline, NOT a centered 0
(reference is not the record itself; ``reference_centered = False`` for verification).

STRAIN SET: the 943 isolates that are the intersection of Caudal's ``Strain`` codes (969)
with Peter's genome panel. The 26 Caudal-only strains (25 ``XTRA_*`` + ``FY4-6``) have no
Peter genome and are EXCLUDED -- only isolates with a matched genome are built.

Sources (hash-pinned, local library mirror):
  - Caudal expression: ``caudalPantranscriptomeRevealsLarge2024/data/
    final_data_annotated_merged_04052022.tab.zip`` (comma-delimited, latin-1;
    sha256 8b55ccd76e1d19476d8f5f718e9e061cb9e4693e343965114dd4cd65d5f8d26b).
  - Peter genome: ``peterGenomeEvolution10112018/data/`` --
    ``allReferenceGenesWithSNPsAndIndelsInferred.tar.gz`` (sha256 b5400b89...),
    ``genesMatrix_PresenceAbsence.tab.gz``, ``genesMatrix_CopyNumber.tab.gz``.
  - S288C reference: SGD R64-4-1 ``S288C_reference_sequence_R64-4-1_20230830.fsa``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import os.path as osp
import pickle
import re
import tarfile
from typing import Any

import lmdb
import numpy as np
import pandas as pd
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    Media,
    NaturalGeneAbsencePerturbation,
    NaturalGenePresencePerturbation,
    Publication,
    ReferenceGenome,
    RNASeqExpressionExperiment,
    RNASeqExpressionExperimentReference,
    RNASeqExpressionPhenotype,
    SequenceVariantPerturbation,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MEASUREMENT_TYPE = "rnaseq_tpm"
SEQUENCE_SOURCE = "peterGenomeEvolution10112018"
CORE_PRESENCE_THRESHOLD = 0.99  # ORF present in >= 99% of the 1011 isolates -> core

# Library-mirror-relative paths (under $DATA_ROOT). The stored artifact + sha256 is
# canonical; these are referenced (symlinked) in-place, never re-downloaded/copied.
CAUDAL_ZIP_REL = (
    "torchcell-library/caudalPantranscriptomeRevealsLarge2024/data/"
    "final_data_annotated_merged_04052022.tab.zip"
)
CAUDAL_ZIP_SHA256 = "8b55ccd76e1d19476d8f5f718e9e061cb9e4693e343965114dd4cd65d5f8d26b"
PETER_DIR_REL = "torchcell-library/peterGenomeEvolution10112018/data"
REFGENE_TAR_NAME = "allReferenceGenesWithSNPsAndIndelsInferred.tar.gz"
REFGENE_TAR_SHA256 = "b5400b89499fe84b1feada51abd7742c29838ae1f28c0cbd208b6622ca533f25"
PRESENCE_NAME = "genesMatrix_PresenceAbsence.tab.gz"
COPYNUMBER_NAME = "genesMatrix_CopyNumber.tab.gz"
SGD_FSA_REL = (
    "data/sgd/genome/S288C_reference_genome_R64-4-1_20230830/"
    "S288C_reference_sequence_R64-4-1_20230830.fsa"
)

CAUDAL_ZIP_BASENAME = "final_data_annotated_merged_04052022.tab.zip"
RAW_FILES = [CAUDAL_ZIP_BASENAME, REFGENE_TAR_NAME, PRESENCE_NAME, COPYNUMBER_NAME]

_ROMAN = [
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "XI",
    "XII",
    "XIII",
    "XIV",
    "XV",
    "XVI",
]
_S288C_RE = re.compile(r"^(Y[A-P][LR]\d{3}[WC](-[A-Z])?|Q\d{4}|YNC[A-Q]\d{4}[WC])$")
_EXCLUDED_STRAIN_RE = re.compile(r"^XTRA_")
_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA string (A/C/G/T/N, case-preserving)."""
    return seq.translate(_COMPLEMENT)[::-1]


def _sgd_chromosomes(fsa_path: str) -> dict[str, str]:
    """Map each SGD chromosome id (roman I..XVI, ``MT``) to its uppercase sequence."""
    chrom: dict[str, str] = {}
    key: str | None = None
    buf: list[str] = []
    with open(fsa_path) as handle:
        for line in handle:
            if line.startswith(">"):
                if key is not None:
                    chrom[key] = "".join(buf).upper()
                match = re.search(r"\[chromosome=([IVX]+)\]", line)
                if match:
                    key = match.group(1)
                elif "[location=mitochondrion]" in line:
                    key = "MT"
                else:
                    key = None
                buf = []
            else:
                buf.append(line.strip())
    if key is not None:
        chrom[key] = "".join(buf).upper()
    return chrom


def _reference_slice(header: str, chrom: dict[str, str]) -> str:
    """Reconstruct the S288C reference allele from a Peter gene-file header coordinate."""
    match = re.search(r"chromosome(\d+):(\d+)-(\d+)\s+([+-])", header)
    if match is None:
        raise ValueError(f"unparseable coordinate header: {header!r}")
    chrom_n = int(match.group(1))
    start = int(match.group(2))
    end = int(match.group(3))
    strand = match.group(4)
    key = "MT" if chrom_n == 17 else _ROMAN[chrom_n - 1]
    seq = chrom[key][start - 1 : end]
    if strand == "-":
        seq = _reverse_complement(seq)
    return seq.upper()


def _demangle_orf(col: str) -> str:
    """Reverse R ``make.names`` on a pangenome-ORF matrix column -> the raw ORF id.

    ``X1834.YAL063C`` -> ``1834-YAL063C``; ``X1.EC1118_1F14_0012g`` ->
    ``1-EC1118_1F14_0012g`` (leading ``X`` dropped; the first ``.`` was the ``-`` after the
    ORF number).
    """
    stem = col[1:] if col.startswith("X") else col
    return stem.replace(".", "-", 1)


def _orf_to_s288c(orf_id: str) -> str | None:
    """Return the S288C systematic name a pangenome ORF id encodes, else None.

    A pangenome ORF id is ``<number>-<name>``; ``<name>`` is an S288C systematic name for
    reference ORFs (``1834-YAL063C`` -> ``YAL063C``) and an assembly id otherwise
    (``1-EC1118_1F14_0012g`` -> None). Residual ``.`` (from mangled ``-B`` suffixes) is
    restored before the pattern test.

    A ``_NumOfGenes_N`` suffix marks a pangenome ORF that collapses N paralogous copies
    of a gene into one cluster (``1771-YAL005C_NumOfGenes_3`` -> ``YAL005C``). These ARE
    reference genes -- 793 of them -- so we strip the suffix and map the cluster to its
    S288C name. This conflates paralogs for the presence/absence question (the cluster is
    present iff any copy is), which is the right granularity here; every reference name
    maps from exactly one pangenome column, so no de-duplication is needed.
    """
    suffix = re.sub(r"^\d+-", "", orf_id).replace(".", "-")
    suffix = re.sub(r"_NumOfGenes_\d+$", "", suffix)
    return suffix if _S288C_RE.match(suffix) else None


@register_dataset
class CaudalPanTranscriptome2024Dataset(ExperimentDataset):
    """Natural-isolate pan-transcriptome: each isolate a perturbation set off S288C."""

    def __init__(
        self,
        root: str = "data/torchcell/caudal_pantranscriptome2024",
        io_workers: int = 0,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (isolates keyed by their 3-letter Caudal strain code)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return RNASeqExpressionExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return RNASeqExpressionExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The Caudal + Peter raw files (symlinked from the library mirror)."""
        return RAW_FILES

    def _data_root(self) -> str:
        """Return $DATA_ROOT (the library-mirror parent)."""
        from dotenv import load_dotenv

        load_dotenv()
        return os.environ["DATA_ROOT"]

    def download(self) -> None:
        """Symlink the hash-pinned mirror files into ``raw_dir`` and verify sha256.

        Large artifacts are referenced in place (never copied). Only the two files with a
        pinned sha256 (Caudal zip, Peter reference-gene tarball) are hash-verified; the
        presence/copy-number matrices are existence-checked.
        """
        data_root = self._data_root()
        os.makedirs(self.raw_dir, exist_ok=True)
        sources = {
            CAUDAL_ZIP_BASENAME: osp.join(data_root, CAUDAL_ZIP_REL),
            REFGENE_TAR_NAME: osp.join(data_root, PETER_DIR_REL, REFGENE_TAR_NAME),
            PRESENCE_NAME: osp.join(data_root, PETER_DIR_REL, PRESENCE_NAME),
            COPYNUMBER_NAME: osp.join(data_root, PETER_DIR_REL, COPYNUMBER_NAME),
        }
        sha256_expected = {
            CAUDAL_ZIP_BASENAME: CAUDAL_ZIP_SHA256,
            REFGENE_TAR_NAME: REFGENE_TAR_SHA256,
        }
        for name, src in sources.items():
            if not osp.exists(src):
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            expected = sha256_expected.get(name)
            if expected is not None:
                got = _sha256(src)
                if got != expected:
                    raise RuntimeError(
                        f"{name} sha256 mismatch: got {got}, expected {expected}"
                    )
            dest = osp.join(self.raw_dir, name)
            if not osp.exists(dest):
                os.symlink(src, dest)
        log.info(
            "Caudal/Peter raw files linked into %s (sha256 verified)", self.raw_dir
        )

    @post_process
    def process(self) -> None:
        """Build the 943 per-isolate pan-transcriptome experiments and write LMDB."""
        data_root = self._data_root()
        os.makedirs(self.preprocess_dir, exist_ok=True)

        # 1. Peter presence/absence + copy-number matrices -> core/accessory + ORF maps.
        presence = pd.read_csv(
            osp.join(self.raw_dir, PRESENCE_NAME), sep="\t", index_col=0
        )
        copynumber = pd.read_csv(
            osp.join(self.raw_dir, COPYNUMBER_NAME), sep="\t", index_col=0
        ).reindex(columns=presence.columns)
        cols = list(presence.columns)
        orf_ids = [_demangle_orf(c) for c in cols]
        s288c_names = [_orf_to_s288c(o) for o in orf_ids]
        s288c_mask = np.array([n is not None for n in s288c_names], dtype=bool)
        presence_vals = presence.to_numpy(dtype=float)
        copynumber_vals = copynumber.to_numpy(dtype=float)
        core_mask = presence_vals.mean(axis=0) >= CORE_PRESENCE_THRESHOLD
        peter_isolates = set(presence.index.astype(str))
        log.info(
            "Peter pangenome: %d ORFs (%d core, %d accessory), %d isolates",
            len(cols),
            int(core_mask.sum()),
            int((~core_mask).sum()),
            len(peter_isolates),
        )

        # 2. Caudal expression -> per-(strain, gene) tpm/count, restricted to matched
        #    strains; population-mean reference over those strains.
        strains, per_strain, ref_tpm, ref_count = self._load_caudal(
            data_root, peter_isolates
        )
        log.info("Matched isolates (Caudal ∩ Peter): %d", len(strains))

        # 3. Sequence variants vs the S288C reference slice (heavy; resumable parquet).
        variants_by_strain = self._sequence_variants(data_root, set(strains))

        # 4. Shared population-mean phenotype reference (absolute TPM baseline).
        self._reference_phenotype = RNASeqExpressionPhenotype(
            expression_tpm=ref_tpm,
            expression_count=ref_count,
            measurement_type=MEASUREMENT_TYPE,
            n_mapped_reads=None,
        )

        # Row index per strain into the numpy presence/copy-number matrices.
        strain_row = {str(s): i for i, s in enumerate(presence.index.astype(str))}
        pd.DataFrame({"strain_id": strains}).to_csv(
            osp.join(self.preprocess_dir, "data.csv"), index=False
        )

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(2e11))
        n_seq_total = 0
        n_cnv_acc_total = 0
        n_cnv_core_total = 0
        idx = 0
        with env.begin(write=True) as txn:
            for strain in tqdm(strains, desc="assembling isolates"):
                row = strain_row[strain]
                presence_perts, absence_perts = self._content_perturbations(
                    strain,
                    presence_vals[row],
                    copynumber_vals[row],
                    s288c_mask,
                    orf_ids,
                    s288c_names,
                )
                seq_perts = self._sequence_perturbations(
                    strain, variants_by_strain.get(strain, [])
                )
                n_seq_total += len(seq_perts)
                n_cnv_acc_total += len(presence_perts)
                n_cnv_core_total += len(absence_perts)
                experiment, reference, publication = self.create_experiment(
                    {
                        "strain": strain,
                        "perturbations": seq_perts + presence_perts + absence_perts,
                        "expression_tpm": per_strain[strain]["tpm"],
                        "expression_count": per_strain[strain]["count"],
                    }
                )
                txn.put(
                    f"{idx}".encode(),
                    pickle.dumps(
                        {
                            "experiment": experiment.model_dump(),
                            "reference": reference.model_dump(),
                            "publication": publication.model_dump(),
                        }
                    ),
                )
                idx += 1
        env.close()
        n = max(idx, 1)
        log.info(
            "Wrote %d isolates. Per-isolate means: sequence_variants=%.1f, "
            "accessory-present=%.1f, core-absent=%.2f",
            idx,
            n_seq_total / n,
            n_cnv_acc_total / n,
            n_cnv_core_total / n,
        )

    def _load_caudal(
        self, data_root: str, peter_isolates: set[str]
    ) -> tuple[
        list[str],
        dict[str, dict[str, dict[str, Any]]],
        dict[str, float],
        dict[str, int],
    ]:
        """Aggregate Caudal expression per (strain, gene); return matched strains + ref."""
        import zipfile

        path = osp.join(self.raw_dir, CAUDAL_ZIP_BASENAME)
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".tab"))
            with zf.open(name) as handle:
                df = pd.read_csv(
                    handle,
                    encoding="latin-1",
                    low_memory=False,
                    usecols=["Strain", "systematic_name", "count", "tpm"],
                )
        df["Strain"] = df["Strain"].astype(str)
        df = df[~df["Strain"].str.match(_EXCLUDED_STRAIN_RE)]
        df = df[df["Strain"] != "FY4-6"]
        df = df[df["Strain"].isin(peter_isolates)]
        # Aggregate multi-allele rows per (strain, gene): tpm sum, count sum (defensive;
        # the merged file already carries one row per pair).
        agg = (
            df.groupby(["Strain", "systematic_name"], sort=False)
            .agg(tpm=("tpm", "sum"), count=("count", "sum"))
            .reset_index()
        )
        agg["systematic_name"] = agg["systematic_name"].astype(str)
        per_strain: dict[str, dict[str, dict[str, Any]]] = {}
        for strain, sub in agg.groupby("Strain", sort=False):
            genes = sub["systematic_name"].tolist()
            tpm = {g: float(t) for g, t in zip(genes, sub["tpm"].tolist())}
            count = {
                g: int(round(float(c))) for g, c in zip(genes, sub["count"].tolist())
            }
            per_strain[str(strain)] = {"tpm": tpm, "count": count}
        # Population-mean reference over the matched isolates (mean per gene).
        ref = agg.groupby("systematic_name", sort=False).agg(
            tpm=("tpm", "mean"), count=("count", "mean")
        )
        ref_tpm = {str(g): float(t) for g, t in ref["tpm"].items()}
        ref_count = {str(g): int(round(float(c))) for g, c in ref["count"].items()}
        strains = sorted(per_strain)
        return strains, per_strain, ref_tpm, ref_count

    def _sequence_variants(
        self, data_root: str, matched: set[str]
    ) -> dict[str, list[tuple[str, str, str]]]:
        """Diff every isolate's allele vs the S288C reference slice, per gene (resumable).

        Returns ``strain_id -> [(systematic_gene_name, symbol, header_token), ...]``. The
        result is cached to ``preprocess/sequence_variants.parquet`` so re-assembly need not
        re-diff the ~6015-gene x 1011-isolate store.
        """
        cache = osp.join(self.preprocess_dir, "sequence_variants.parquet")
        if osp.exists(cache):
            log.info("Loading cached sequence variants from %s", cache)
            vdf = pd.read_parquet(cache)
            out: dict[str, list[tuple[str, str, str]]] = {}
            for strain, sub in vdf.groupby("strain_id", sort=False):
                out[str(strain)] = list(
                    zip(sub["systematic_gene_name"], sub["symbol"], sub["header_token"])
                )
            return out

        chrom = _sgd_chromosomes(osp.join(data_root, SGD_FSA_REL))
        tar_path = osp.join(self.raw_dir, REFGENE_TAR_NAME)
        strain_col: list[str] = []
        sys_col: list[str] = []
        sym_col: list[str] = []
        tok_col: list[str] = []
        with tarfile.open(tar_path, "r:gz") as tf:
            for member in tqdm(tf, desc="diffing reference genes"):
                if not member.isfile():
                    continue
                sys_name = member.name.replace(".fasta", "")
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                records = _parse_fasta(extracted.read().decode("latin-1"))
                if not records:
                    continue
                ref = _reference_slice(records[0][0], chrom)
                split_key = f"_{sys_name}_"
                for header, seq in records:
                    token = header.split()[0].split("\t")[0]
                    if split_key not in token:
                        continue
                    iso, symbol = token.split(split_key, 1)
                    if iso not in matched:
                        continue
                    if seq.upper() != ref:
                        strain_col.append(iso)
                        sys_col.append(sys_name)
                        sym_col.append(symbol)
                        tok_col.append(token)
        vdf = pd.DataFrame(
            {
                "strain_id": strain_col,
                "systematic_gene_name": sys_col,
                "symbol": sym_col,
                "header_token": tok_col,
            }
        )
        vdf.to_parquet(cache, index=False)
        log.info(
            "Diffed %d sequence variants across %d isolates -> %s",
            len(vdf),
            vdf["strain_id"].nunique(),
            cache,
        )
        out = {}
        for strain, sub in vdf.groupby("strain_id", sort=False):
            out[str(strain)] = list(
                zip(sub["systematic_gene_name"], sub["symbol"], sub["header_token"])
            )
        return out

    @staticmethod
    def _sequence_perturbations(
        strain: str, variants: list[tuple[str, str, str]]
    ) -> list[SequenceVariantPerturbation]:
        """Build the SequenceVariantPerturbations for one isolate."""
        perts: list[SequenceVariantPerturbation] = []
        for sys_name, symbol, token in variants:
            perts.append(
                SequenceVariantPerturbation(
                    systematic_gene_name=sys_name,
                    perturbed_gene_name=symbol or sys_name,
                    strain_id=strain,
                    sequence_source=SEQUENCE_SOURCE,
                    sequence_uri=f"{sys_name}.fasta#{token}",
                    sequence_sha256=REFGENE_TAR_SHA256,
                )
            )
        return perts

    @staticmethod
    def _content_perturbations(
        strain: str,
        presence_row: np.ndarray,
        copynumber_row: np.ndarray,
        s288c_mask: np.ndarray,
        orf_ids: list[str],
        s288c_names: list[str | None],
    ) -> tuple[
        list[NaturalGenePresencePerturbation], list[NaturalGeneAbsencePerturbation]
    ]:
        """AXIS-1 presence/absence edits for one isolate, RELATIVE TO S288C.

        Presence/absence vs S288C is a question of **reference membership** (is this ORF
        in S288C?), NOT population frequency (is it in >=99% of isolates?). The two sets
        differ -- a reference ORF can be variable, an accessory ORF can be near-ubiquitous
        -- so both loops gate on ``s288c_mask`` (does the pangenome column map to an S288C
        systematic name, paralog clusters included), never on a frequency ``core_mask``:

        * non-reference ORF PRESENT -> a natural PRESENCE (S288C lacks it, the isolate has
          it), with observed copy number;
        * reference ORF ABSENT -> a natural ABSENCE (S288C has it, the isolate lost it).

        A reference ORF present and an accessory ORF absent are BOTH no-ops vs S288C. The
        earlier core/accessory gating silently dropped every *variable* reference ORF that
        was absent (~133 per isolate), so those isolates reconstructed as if they still
        carried the gene -- the exact failure the "never dropped" invariant forbids.
        copy_number != 1 on a present gene is a separate dosage axis, not modelled here.
        """
        presence: list[NaturalGenePresencePerturbation] = []
        for j in np.nonzero((~s288c_mask) & (presence_row == 1))[0]:
            cn = copynumber_row[j]
            copy_number = float(cn) if np.isfinite(cn) and cn > 0 else 1.0
            orf_id = orf_ids[j]
            presence.append(
                NaturalGenePresencePerturbation(
                    systematic_gene_name=orf_id,
                    perturbed_gene_name=orf_id,
                    copy_number=copy_number,
                    strain_id=strain,
                    pangenome_orf_id=orf_id,
                    origin=None,
                    sequence_source=SEQUENCE_SOURCE,
                )
            )
        absence: list[NaturalGeneAbsencePerturbation] = []
        for j in np.nonzero(s288c_mask & (presence_row == 0))[0]:
            # s288c_mask guarantees a systematic name here; keep the pangenome-id fallback
            # defensively so an absence is never dropped.
            name = s288c_names[j] or orf_ids[j]
            absence.append(
                NaturalGeneAbsencePerturbation(
                    systematic_gene_name=name,
                    perturbed_gene_name=name,
                    strain_id=strain,
                    pangenome_orf_id=orf_ids[j],
                    sequence_source=SEQUENCE_SOURCE,
                )
            )
        return presence, absence

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(  # type: ignore[override]
        self, row: dict[str, Any]
    ) -> tuple[
        RNASeqExpressionExperiment, RNASeqExpressionExperimentReference, Publication
    ]:
        """Build the RNA-seq experiment/reference/publication for one isolate."""
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        )
        genotype = Genotype(perturbations=row["perturbations"])
        # SC liquid medium, 30 C, harvested at mid-log (OD ~0.3) -- Caudal Methods.
        environment = Environment(
            media=Media(name="SC", state="liquid", is_synthetic=True),
            temperature=Temperature(value=30),
        )
        phenotype = RNASeqExpressionPhenotype(
            expression_tpm=row["expression_tpm"],
            expression_count=row["expression_count"],
            measurement_type=MEASUREMENT_TYPE,
            n_mapped_reads=None,
        )
        experiment = RNASeqExpressionExperiment(
            dataset_name=self.name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        reference = RNASeqExpressionExperimentReference(
            dataset_name=self.name,
            genome_reference=genome_reference,
            environment_reference=environment.model_copy(),
            phenotype_reference=self._reference_phenotype,
        )
        publication = Publication(
            pubmed_id="38778243",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/38862621/",
            doi="10.1038/s41588-024-01769-9",
            doi_url="https://doi.org/10.1038/s41588-024-01769-9",
        )
        return experiment, reference, publication


def _parse_fasta(text: str) -> list[tuple[str, str]]:
    """Parse FASTA text into ``[(header_without_gt, sequence), ...]``."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    seq: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq)))
            header = line[1:]
            seq = []
        else:
            seq.append(line.strip())
    if header is not None:
        records.append((header, "".join(seq)))
    return records


def _sha256(path: str, chunk_size: int = 1 << 20) -> str:
    """Return the hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Build/load the dataset for interactive debugging.

    Loads the existing LMDB if already built. To rebuild, delete ``<root>/processed``
    (and ``<root>/preprocess/sequence_variants.parquet`` to re-diff) first.
    """
    from dotenv import load_dotenv

    load_dotenv()
    root = osp.join(
        os.environ["DATA_ROOT"], "data/torchcell/caudal_pantranscriptome2024"
    )
    dataset = CaudalPanTranscriptome2024Dataset(root=root)
    print(f"len = {len(dataset)}")
    record = dataset[0]
    exp = record["experiment"]
    ptypes: dict[str, int] = {}
    for pert in exp["genotype"]["perturbations"]:
        ptypes[pert["perturbation_type"]] = ptypes.get(pert["perturbation_type"], 0) + 1
    print("record[0] perturbation-type counts:", ptypes)
    print("record[0] phenotype gene count:", len(exp["phenotype"]["expression_tpm"]))
    print("record[0] genome_reference:", record["reference"]["genome_reference"])


if __name__ == "__main__":
    main()
