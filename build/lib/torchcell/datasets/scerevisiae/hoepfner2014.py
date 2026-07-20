# torchcell/datasets/scerevisiae/hoepfner2014
# [[torchcell.datasets.scerevisiae.hoepfner2014]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/hoepfner2014
# Test file: tests/torchcell/datasets/scerevisiae/test_hoepfner2014.py
"""Hoepfner 2014 HIP-HOP chemogenomic atlas: env x geno -> sensitivity score.

Hoepfner et al. 2014 (Microbiol Res, doi:10.1016/j.micres.2013.11.004) is the Novartis
genome-wide chemogenomic resource: 2956 HIP + 2923 HOP experiments for 1776 discrete
compounds (>40M data points), each compound profiled in duplicate at (or near) its
S. cerevisiae IC30.

ENCODABLE COMPOUNDS ONLY (build filter): ~92% of the profiled compounds are PROPRIETARY
Novartis ``CMBxxx`` entries with no released structure -- black-box perturbations no
molecular encoder can represent. This loader keeps ONLY the compounds with a released
SMILES in Table S1 (150 of the 1852 deposited compounds; the one named-but-structureless
case, CMB222 "Enniatin derivative", is also dropped), so every stored (ORF, compound)
record is featurisable by the cell graph transformer. That is ~610 of the 5879 sensitivity
columns; the full atlas is recoverable by removing the ``smiles is None`` skip in
``_column_meta``. Provenance: paper.md line 110 ("In addition to 1641 proprietary compounds
(named CMBxxx), we included 135 reference compounds ... Table S1"); analysis in
``experiments/017-hoepfner-background-mutations`` (``compound_encodability.json``).

It is the canonical use case for the WS15 schema extension
(``EngineeredCopyNumberPerturbation`` + ``ReferenceGenome.ploidy``), because HIP and HOP
are two DIPLOID deletion collections:

- HIP (haploinsufficiency profiling) uses the HETEROZYGOUS deletion collection (YSC1055):
  in a DIPLOID one of the two autosomal copies is deleted (KanMX), leaving one copy ->
  reduced dosage. This collection INCLUDES ESSENTIAL genes (they are viable as
  heterozygotes), so HIP reaches genes the homozygous collection cannot. A HIP strain is
  modeled as ``EngineeredCopyNumberPerturbation(copy_number=1, reference_copy_number=2,
  marker="KanMX")`` in a ``ReferenceGenome(ploidy="diploid")``.
- HOP (homozygous deletion profiling) uses the HOMOZYGOUS deletion collection (YSC1056):
  BOTH copies deleted in a diploid -> total absence; only non-essential genes are viable.
  A HOP strain is the existing absence leaf ``KanMxDeletionPerturbation`` in a
  ``ReferenceGenome(ploidy="diploid")``.

Both collections descend from the diploid BY4743; the ``ReferenceGenome.strain`` string
records which collection (YSC1055 vs YSC1056), and the genotype LEAF TYPE (engineered-CNV
vs deletion) keeps HIP and HOP records structurally distinguishable in one LMDB.

READOUT -- the stored score is the (adjusted) MADL SENSITIVITY score
(``measurement_type=sensitivity_score``), the paper's DEFINED per-experiment quantity:
*"We defined the sensitivity score for a given gene deletion strain for each
compound/concentration combination based on the logarithmic ratio of treated versus
control measurements"* (Results, paper.md line 110); computed as
``(r_L - med(r_L)) / MAD(r_L)`` over all strains in a sample (a robust experiment-wise
z-score), then adjusted for replicate variability (Methods "Processing of TAG16K v2 data",
paper.md line 84). The deposited files ALSO carry a companion gene-wise ``z-score`` column
per experiment (a second normalization across all experiments); this loader stores the
sensitivity score (the atomic per-experiment readout with a clean replicate count) and
records the z-score's existence in ``units``.

SOURCED PROVENANCE (all quotes from the mirrored paper.md; sha256-pinned raw files):
- Ploidy / het-vs-hom collections: *"the haploinsufficiency profiling (HIP) and the
  homozygous deletion profiling (HOP) assays ... HIP exploits the increased sensitivity of
  a heterozygous deletion strain ... The HOP assay provides a genome-wide overview ..."*
  (paper.md line 34); *"The heterozygous and homozygous deletion strain collections were
  acquired (YSC1055 and YSC1056, OpenBiosystems)"* (line 64). The deletion collections are
  diploid (BY4743-derived); HIP essential-gene strains are drawn as gray boxes throughout
  (Figs 1-4).
- Replicate structure -> n_samples: *"Experimental compounds were tested at n = 2 within
  the same plate at or close to their IC30 concentration"* (HIP HOP assay, line 68); the
  score files' column prefix encodes this per experiment -- *"If a column name starts with
  'Ad.', then the MADL score of each strain is adjusted for the variability between the two
  measurements of the two samples ... In some cases only one of the two samples for a
  compound passed QC; in this case there is only one measurement ... indicated by the
  prefix 'MADL'"* (Supplemental data organization, line 52). Hence ``n_samples=2`` for
  ``Ad.`` columns and ``n_samples=1`` for ``MADL`` columns, ``sample_unit=technical_replicate``
  (the two duplicate wells in the same plate). No per-cell SE is released.
- Concentration (IC30, uM): each column header carries the exact per-experiment
  concentration -- *"<Conc.>: The uM concentration of a compound"* (line 50); *"HIP HOP
  experiments were run at the corresponding ... IC30 ... Testing at IC30 resulted in best
  expected strain sensitivity"* (line 58). Stored as ``Concentration(value=<header uM>,
  unit="uM", basis="IC30")``.
- Medium / temperature / duration / solvent: YPD (*"YPD 2% glucose, 2% BactoPeptone, 1%
  yeast extract"*, line 58), 30 C, ~16 h incubation (lines 68, 70), 2% DMSO vehicle
  (*"In all experiments DMSO was normalized to 2%"*, line 58).

DATA SOURCE (scriptable + sha256-pinned): Dryad doi:10.5061/dryad.v5m8v files
``HIP_scores.txt`` (644 MB) and ``HOP_scores.txt`` (505 MB) -- the deposited processed
score matrices (rows = systematic ORF names, columns = per-experiment scores) -- and
``Table_S1.xls`` (compound IC30 + common name + SMILES for the reference/novel MoA
compounds; the ~1641 proprietary CMB compounds have no released name). Dryad's frontend is
behind an Anubis SHA-256 proof-of-work; ``download()`` solves it deterministically (the PoW
is fully scriptable), streams each file, and verifies its pinned sha256.

BUILD / SOURCE QUIRKS handled deterministically (no fabrication):
- Each experiment column is an atomic measurement (a distinct CMB/concentration/study);
  the same compound recurs across replicate/dose experiments, so ``compound_name`` embeds
  the exact experiment tag ``<CMB>_<conc>_<HIP|HOP>_<study>`` (keeping the human name /
  SMILES when known) so every (ORF, experiment) pair is unique (L1) and HIP/HOP never
  collide. The chemical concentration and assay are ALSO stored structurally.
- Systematic ORF names are validated against the SGD R64 gene universe (ORF + RNA-coding
  FASTA headers); names not resolving to R64 (old/merged deletion-collection features) are
  DROPPED and counted (never guessed) -> L4 containment == 1.000. Empty cells are skipped.
"""

import hashlib
import json
import logging
import os
import os.path as osp
import pickle
import re
import time
from collections.abc import Callable
from typing import Any

import lmdb
import requests
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.schema import (
    Compound,
    Concentration,
    ConcentrationUnit,
    DoseBasis,
    EngineeredCopyNumberPerturbation,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    KanMxDeletionPerturbation,
    MeasurementType,
    Media,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SmallMoleculePerturbation,
    Solvent,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DOI = "10.1016/j.micres.2013.11.004"

# Dryad doi:10.5061/dryad.v5m8v file-stream ids + pinned sha256 of the deposited files.
_DRYAD_FILES: dict[str, dict[str, str]] = {
    "HIP_scores.txt": {
        "url": "https://datadryad.org/downloads/file_stream/4834608",
        "sha256": "dbc5041defea9c046da0890d5e569f97d5f7afbf50ea0885f539ea8e5980cd24",
    },
    "HOP_scores.txt": {
        "url": "https://datadryad.org/downloads/file_stream/4834609",
        "sha256": "99b386a84384eae847657ed41bf222c9550a87ef961f0ab191833c918771ffd7",
    },
    "Table_S1.xls": {
        "url": "https://datadryad.org/downloads/file_stream/4834600",
        "sha256": "115bb31cc5e696588d1ecb4ffa262475e05025e22347f7e004f77fd635898209",
    },
}

_ASSAYS = (("HIP_scores.txt", "HIP"), ("HOP_scores.txt", "HOP"))

# S288C reference gene universe (systematic ORF + RNA-coding names) for R64 resolution.
_SGD_GENE_FASTAS = (
    "data/sgd/genome/S288C_reference_genome_R64-4-1_20230830/"
    "orf_coding_all_R64-4-1_20230830.fasta",
    "data/sgd/genome/S288C_reference_genome_R64-4-1_20230830/"
    "rna_coding_R64-4-1_20230830.fasta",
)

# Experiment column header: '(Ad.|MADL) scores for Exp. <CMB>_<conc>_<HIP|HOP>_<study>'
# with an optional trailing ' z-score' (the companion gene-wise z-score column).
_COL_RE = re.compile(
    r"^(?P<prefix>Ad\.|MADL) scores for Exp\. "
    r"(?P<cmb>\d+)_(?P<conc>[\d.]+)_(?P<assay>HIP|HOP)_(?P<study>\S+?)(?P<z> z-score)?$"
)

MEASUREMENT_UNITS = (
    "adjusted MADL sensitivity score = (r_L - med(r_L)) / MAD(r_L) over all pool strains, "
    "r_L = log ratio of treated vs control strain abundance (Hoepfner 2014); negative = "
    "hypersensitive, positive = resistant; 0 = growth equal to control. A companion "
    "gene-wise z-score column is deposited per experiment but not stored here."
)

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _solve_anubis(random_data: str, difficulty: int) -> tuple[str, int]:
    """Solve an Anubis 'fast' proof-of-work: sha256(random_data + nonce) with
    ``difficulty`` leading zero hex nibbles. Deterministic and scriptable.
    """
    p = difficulty // 2
    odd = difficulty % 2
    nonce = 0
    while True:
        digest = hashlib.sha256((random_data + str(nonce)).encode()).digest()
        if all(digest[i] == 0 for i in range(p)) and not (
            odd and (digest[p] >> 4) != 0
        ):
            return digest.hex(), nonce
        nonce += 1


def _dryad_get(session: requests.Session, url: str) -> requests.Response:
    """GET a Dryad file-stream URL, clearing the Anubis PoW challenge if present.

    Returns a streamed response for the file body. Retries on WAF throttling.
    """
    for _ in range(80):
        resp = session.get(url, timeout=180, stream=True)
        if not resp.headers.get("content-type", "").startswith("text/html"):
            return resp
        body = resp.text
        resp.close()
        challenge_match = re.search(
            r'id="anubis_challenge" type="application/json">(.*?)</script>', body, re.S
        )
        if challenge_match is None:
            time.sleep(15)  # throttled: back off and retry
            continue
        challenge = json.loads(challenge_match.group(1))["challenge"]
        base_match = re.search(
            r'id="anubis_base_prefix" type="application/json">(.*?)</script>',
            body,
            re.S,
        )
        base = json.loads(base_match.group(1)) if base_match else ""
        digest, nonce = _solve_anubis(challenge["randomData"], challenge["difficulty"])
        pass_url = (
            f"https://datadryad.org{base}"
            "/.within.website/x/cmd/anubis/api/pass-challenge"
        )
        cleared = session.get(
            pass_url,
            params={
                "id": challenge["id"],
                "response": digest,
                "nonce": nonce,
                "redir": url,
                "elapsedTime": 100,
            },
            stream=True,
            allow_redirects=True,
            timeout=1800,
        )
        if not cleared.headers.get("content-type", "").startswith("text/html"):
            return cleared
        cleared.close()
        time.sleep(15)
    raise RuntimeError(f"could not clear Anubis challenge for {url}")


def _load_sgd_genes(data_root: str) -> set[str]:
    """S288C R64 systematic-name universe from the ORF + RNA-coding FASTA headers."""
    genes: set[str] = set()
    for rel in _SGD_GENE_FASTAS:
        with open(osp.join(data_root, rel)) as handle:
            for line in handle:
                if line.startswith(">"):
                    genes.add(line[1:].split()[0])
    return genes


def _load_compound_meta(table_s1_path: str) -> dict[str, dict[str, str | None]]:
    """CMB id -> {common_name, smiles} from Table_S1 (reference + novel MoA + structures).

    Only the reference/novel MoA compounds carry a name; the ~1641 proprietary CMB
    compounds are absent (name/SMILES stay None).
    """
    import pandas as pd

    def cmb_ids(value: Any) -> list[str]:
        """Parse a CMB-ID cell; some structure rows list several ids ('244, 1818')."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        if isinstance(value, (int, float)):
            return [str(int(value))]
        return [tok.strip() for tok in str(value).split(",") if tok.strip().isdigit()]

    meta: dict[str, dict[str, str | None]] = {}
    xl = pd.ExcelFile(table_s1_path)
    for sheet in ("Reference Substances known MoA", "Substances novel MoA"):
        frame = xl.parse(sheet, header=0)
        for _, row in frame.iterrows():
            name = row.get("Common Name")
            name = None if (name is None or pd.isna(name)) else str(name).strip()
            for cmb_id in cmb_ids(row.get("CMB ID")):
                meta.setdefault(cmb_id, {"common_name": None, "smiles": None})
                if name:
                    meta[cmb_id]["common_name"] = name
    structures = xl.parse("All Structures", header=0)
    for _, row in structures.iterrows():
        smiles = row.get("SMILE string")
        smiles = None if (smiles is None or pd.isna(smiles)) else str(smiles).strip()
        for cmb_id in cmb_ids(row.get("CMB ID")):
            meta.setdefault(cmb_id, {"common_name": None, "smiles": None})
            if smiles:
                meta[cmb_id]["smiles"] = smiles
    return meta


class _ColumnMeta:
    """Per-experiment (sensitivity column) metadata + the treated environment object."""

    __slots__ = ("index", "cmb", "conc", "study", "n_samples", "env")

    def __init__(
        self,
        index: int,
        cmb: str,
        conc: float,
        study: str,
        n_samples: int,
        env: Environment,
    ) -> None:
        self.index = index
        self.cmb = cmb
        self.conc = conc
        self.study = study
        self.n_samples = n_samples
        self.env = env


@register_dataset
class EnvChemgenHoepfner2014Dataset(ExperimentDataset):
    """Hoepfner 2014 HIP-HOP atlas: env x (het-CNV | hom-deletion) -> sensitivity score."""

    def __init__(
        self,
        root: str = "data/torchcell/env_chemgen_hoepfner2014",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset. ORFs are already systematic, so no genome is required."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return EnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return EnvironmentResponseExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The Dryad score matrices + compound table required before processing."""
        return ["HIP_scores.txt", "HOP_scores.txt", "Table_S1.xls"]

    def download(self) -> None:
        """Fetch the Dryad files (solving the Anubis PoW), verify each pinned sha256."""
        os.makedirs(self.raw_dir, exist_ok=True)
        session = requests.Session()
        session.headers.update({"User-Agent": _UA})
        for name, spec in _DRYAD_FILES.items():
            dest = osp.join(self.raw_dir, name)
            if osp.exists(dest):
                continue
            log.info("Downloading Hoepfner2014 %s from %s", name, spec["url"])
            resp = _dryad_get(session, spec["url"])
            digest = hashlib.sha256()
            with open(dest, "wb") as handle:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
                        digest.update(chunk)
            resp.close()
            got = digest.hexdigest()
            if got != spec["sha256"]:
                raise RuntimeError(
                    f"{name} sha256 mismatch: got {got}, expected {spec['sha256']}"
                )
            log.info("Wrote %s (sha256 verified)", dest)

    def _compound_name(
        self,
        cmb: str,
        conc: float,
        assay: str,
        study: str,
        meta: dict[str, dict[str, str | None]],
    ) -> str:
        """Human name (or CMB id) plus the unique experiment tag for L1 pair-uniqueness."""
        base = (meta.get(cmb) or {}).get("common_name") or f"CMB{cmb}"
        conc_str = f"{conc:f}".rstrip("0").rstrip(".")
        return f"{base} [{cmb}_{conc_str}_{assay}_{study}]"

    def _reference_strain(self, assay: str) -> str:
        """Diploid deletion-collection strain string (records HIP vs HOP collection)."""
        return (
            "BY4743 heterozygous deletion collection (YSC1055)"
            if assay == "HIP"
            else "BY4743 homozygous deletion collection (YSC1056)"
        )

    def _reference_dump(self, assay: str) -> dict[str, Any]:
        """One shared per-assay reference: the untreated (DMSO-vehicle, no-compound)
        control of the diploid collection, sensitivity score 0.

        The reference is deliberately compound-INDEPENDENT (the biological control is the
        no-drug DMSO culture, identical across every compound), so the whole atlas has
        exactly TWO unique references (HIP, HOP) -- keeping the base-class reference index
        O(2 x N) instead of O(experiments x N).
        """
        control_env = Environment(
            media=Media(
                name="YPD (1% yeast extract, 2% BactoPeptone, 2% glucose), 2% DMSO vehicle",
                state="liquid",
                is_synthetic=False,  # YPD = complex/rich medium (yeast extract + peptone)
                base_medium="YPD",
            ),
            temperature=Temperature(value=30.0),
            aerobicity="aerobic",
            duration_hours=16.0,
        )
        return EnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae",
                strain=self._reference_strain(assay),
                ploidy="diploid",
            ),
            environment_reference=control_env,
            phenotype_reference=EnvironmentResponsePhenotype(
                measurement_type=MeasurementType.sensitivity_score,
                environment_response=0.0,
                units=MEASUREMENT_UNITS,
            ),
        ).model_dump()

    def _column_meta(
        self, header: list[str], assay: str, meta: dict[str, dict[str, str | None]]
    ) -> list[_ColumnMeta]:
        """Parse sensitivity (non-z-score) columns; build the treated environment per column."""
        columns: list[_ColumnMeta] = []
        for index, raw in enumerate(header):
            name = raw.strip().strip('"')
            match = _COL_RE.match(name)
            if match is None or match.group("z") is not None:
                continue  # column 0 (Systematic Name) or a z-score companion column
            if match.group("assay") != assay:
                continue
            cmb = match.group("cmb")
            conc = float(match.group("conc"))
            study = match.group("study")
            n_samples = 2 if match.group("prefix") == "Ad." else 1
            compound_name = self._compound_name(cmb, conc, assay, study, meta)
            smiles = (meta.get(cmb) or {}).get("smiles")
            if smiles is None:
                # ENCODABLE-COMPOUNDS-ONLY build: ~92% of the atlas is proprietary Novartis
                # CMBxxx with no released structure (plus the lone named-but-structureless
                # CMB222 "Enniatin derivative"); these are black-box perturbations a
                # molecular encoder cannot represent, so they are dropped at build time.
                # Only the 150 compounds with a released SMILES (Table S1) are kept, so every
                # stored (ORF, compound) record is featurisable by the cell graph transformer.
                continue
            env = Environment(
                media=Media(
                    name="YPD (1% yeast extract, 2% BactoPeptone, 2% glucose)",
                    state="liquid",
                    is_synthetic=False,  # YPD = complex/rich medium (yeast extract + peptone)
                    base_medium="YPD",
                ),
                temperature=Temperature(value=30.0),
                perturbations=[
                    SmallMoleculePerturbation(
                        compound=Compound(name=compound_name, smiles=smiles),
                        concentration=Concentration(
                            value=conc,
                            unit=ConcentrationUnit.micromolar,
                            basis=DoseBasis.IC30,
                        ),
                        solvent=Solvent(name="DMSO", percent=2.0),
                    )
                ],
                aerobicity="aerobic",
                duration_hours=16.0,
            )
            columns.append(_ColumnMeta(index, cmb, conc, study, n_samples, env))
        return columns

    def _genotype(self, assay: str, orf: str) -> Genotype:
        """HIP -> heterozygous engineered-CNV (copy 1 of 2); HOP -> homozygous deletion."""
        if assay == "HIP":
            return Genotype(
                perturbations=[
                    EngineeredCopyNumberPerturbation(
                        systematic_gene_name=orf,
                        perturbed_gene_name=orf,
                        copy_number=1,
                        reference_copy_number=2,
                        marker="KanMX",
                    )
                ]
            )
        return Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=orf
                )
            ]
        )

    def _iter_records(
        self,
        path: str,
        assay: str,
        sgd_genes: set[str],
        meta: dict[str, dict[str, str | None]],
        pub_dump: dict[str, Any],
    ) -> Any:
        """Yield pickled records for one score matrix (one per (R64 ORF, sensitivity col))."""
        dropped: set[str] = set()
        n_written = 0
        ref_dump = self._reference_dump(assay)
        with open(path) as handle:
            header = handle.readline().rstrip("\n").split("\t")
            columns = self._column_meta(header, assay, meta)
            log.info("Hoepfner2014 %s: %d sensitivity experiments", assay, len(columns))
            for line in tqdm(handle, desc=f"Hoepfner2014 {assay}"):
                parts = line.rstrip("\n").split("\t")
                orf = parts[0].strip().strip('"')
                if orf not in sgd_genes:
                    dropped.add(orf)
                    continue
                genotype = self._genotype(assay, orf)
                for col in columns:
                    if col.index >= len(parts):
                        continue
                    cell = parts[col.index].strip().strip('"')
                    if cell == "":
                        continue
                    experiment = EnvironmentResponseExperiment(
                        dataset_name=self.name,
                        genotype=genotype,
                        environment=col.env,
                        phenotype=EnvironmentResponsePhenotype(
                            measurement_type=MeasurementType.sensitivity_score,
                            environment_response=float(cell),
                            n_samples=col.n_samples,
                            sample_unit=SampleUnit.technical_replicate,
                            units=MEASUREMENT_UNITS,
                        ),
                    )
                    n_written += 1
                    yield pickle.dumps(
                        {
                            "experiment": experiment.model_dump(),
                            "reference": ref_dump,
                            "publication": pub_dump,
                        }
                    )
        log.info(
            "Hoepfner2014 %s: wrote %d records; dropped %d non-R64 ORF names: %s",
            assay,
            n_written,
            len(dropped),
            sorted(dropped),
        )

    @post_process
    def process(self) -> None:
        """Stream both score matrices into the LMDB (one record per (ORF, experiment)).

        Commits in batches so a 30M-record build never accumulates one giant
        (multi-GB dirty-page) LMDB write transaction.
        """
        from dotenv import load_dotenv

        load_dotenv()
        data_root = os.environ["DATA_ROOT"]
        sgd_genes = _load_sgd_genes(data_root)
        meta = _load_compound_meta(osp.join(self.raw_dir, "Table_S1.xls"))
        pub_dump = Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}").model_dump()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(2e12))
        batch_size = 500_000
        idx = 0
        txn = env.begin(write=True)
        for filename, assay in _ASSAYS:
            for value in self._iter_records(
                osp.join(self.raw_dir, filename), assay, sgd_genes, meta, pub_dump
            ):
                txn.put(f"{idx}".encode(), value)
                idx += 1
                if idx % batch_size == 0:
                    txn.commit()
                    txn = env.begin(write=True)
        txn.commit()
        env.close()
        log.info("Wrote %d Hoepfner2014 environment-response experiments to LMDB", idx)

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
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
    root = osp.join(data_root, "data/torchcell/env_chemgen_hoepfner2014")
    dataset = EnvChemgenHoepfner2014Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()
