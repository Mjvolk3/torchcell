# torchcell/datasets/scerevisiae/zelezniak2018
# [[torchcell.datasets.scerevisiae.zelezniak2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/zelezniak2018
# Test file: tests/torchcell/datasets/scerevisiae/test_zelezniak2018.py
"""Zelezniak 2018 kinase-knockout multi-omics datasets (proteome + metabolome).

Zelezniak et al. 2018 (Cell Systems, doi:10.1016/j.cels.2018.08.001) profiled the
S. cerevisiae kinase-deletion collection. This module holds the two matching loaders,
both from the same Zenodo record 1320289 (concept DOI 10.5281/zenodo.1320288):

- ``ProteomeZelezniak2018Dataset`` -- the quantitative PROTEOME of 97 kinase-deletion
  strains (plus WT) by SWATH-MS (data-independent acquisition). We ingest the authors'
  processed per-strain matrix (``proteins_dataset.data_prep.tsv``): a long-format table
  of batch-corrected (SVA-adjusted) label-free protein signal per (protein, sample, KO
  strain, replicate), covering 726 proteins. We aggregate the replicate samples of each
  knockout strain to a per-protein mean + standard error, mapping to
  ``ProteinAbundancePhenotype`` (WS9): ``protein_abundance = {protein_ORF -> mean log
  signal}`` with ``measurement_type = "swath_ms_label_free_log_signal_sva"``. The parent
  **WT** strain (``KO_ORF == "WT"``) supplies the reference profile. Every protein has
  >=2 replicate samples per strain, so the standard error is always defined.

- ``MetaboliteZelezniak2018Dataset`` -- the targeted central-carbon/amino-acid
  METABOLOME of the same 95 kinase-KO strains (plus a measured WT) by SRM-MS/MS. The file
  ``metabolites_dataset.data_prep.tsv`` is a long-format table of batch-corrected SRM
  signal per (metabolite, strain, replicate). We aggregate to a per-metabolite mean +
  standard error, mapping to ``MetabolitePhenotype`` (WS9): ``metabolite_level =
  {metabolite_id -> mean signal}`` with ``measurement_type =
  "srm_ms_signal_batch_corrected"`` (the value is an ARBITRARY batch-corrected SRM
  signal, NOT a concentration; range ~0.004-58995). This is the first dataset to
  populate ``target_metabolite_ids`` (metabolite -> Yeast9 ``s_NNNN``), sourced from
  ``YeastGEM`` (never invented), enabling constraint-based-model linkage.
    - Columns (differ from the Zenodo README): ``metabolite_id, kegg_id, official_name,
      dataset, genotype, replicate, value``. ``genotype`` is the strain (systematic
      kinase ORF, or literal ``WT``); NOT a KO_ORF column. ``metabolite_id`` is a
      BiGG-style id (50 total); ~5 are co-elution merges joined with ``;`` (e.g.
      ``3pg;2pg``, ``ala-L;ala-B``, ``g6p;f6p;g6p-B``) which we KEEP verbatim as dict
      keys (honest to source; resolved via the FIRST sub-id). ``dataset`` is the protocol
      used for generation (1/2/3); we POOL rows across it per (metabolite, strain) so
      ``n_replicates`` is the pooled row count.

For both, the background is BY4741 made prototrophic by the pHLUM minichromosome. The
proteome ``?download=1`` URL works, but it 403s for the metabolome file, which is fetched
via the Zenodo API content endpoint instead.
"""

import hashlib
import logging
import math
import os
import os.path as osp
import pickle
import re
import urllib.request
from typing import Any

import lmdb
import pandas as pd
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    ProteinAbundanceExperiment,
    ProteinAbundanceExperimentReference,
    ProteinAbundancePhenotype,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.metabolism.yeast_GEM import YeastGEM

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MEASUREMENT_TYPE = "swath_ms_label_free_log_signal_sva"
_SYSTEMATIC_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$")
_WT = "WT"

# Zenodo record 1320289 -> proteins_dataset.data_prep.tsv (direct, pinned).
DATA_URL = (
    "https://zenodo.org/records/1320289/files/proteins_dataset.data_prep.tsv?download=1"
)
DATA_FILENAME = "proteins_dataset.data_prep.tsv"
DATA_SHA256 = "9ff81ecb1e2dd44d2f6e072ce5b628f0be1abdf57cdbd90d645db4d1fb64bfeb"

# The metabolome value is an ARBITRARY batch-corrected SRM signal, NOT a concentration.
METABOLITE_MEASUREMENT_TYPE = "srm_ms_signal_batch_corrected"

# Zenodo record 1320289 -> metabolites_dataset.data_prep.tsv. The ?download=1 URL 403s
# for this file, so we hit the Zenodo API content endpoint.
METABOLITE_DATA_URL = (
    "https://zenodo.org/api/records/1320289/files/"
    "metabolites_dataset.data_prep.tsv/content"
)
METABOLITE_DATA_FILENAME = "metabolites_dataset.data_prep.tsv"
METABOLITE_DATA_SHA256 = (
    "c4429fd8cef675d96ffacba1ed51e52ea483fd72d6978a22c04fa405f4e1b07d"
)


@register_dataset
class ProteomeZelezniak2018Dataset(ExperimentDataset):
    """SWATH-MS proteome of the yeast kinase-knockout collection (97 strains)."""

    def __init__(
        self,
        root: str = "data/torchcell/proteome_zelezniak2018",
        io_workers: int = 0,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (KO_ORF is systematic; no genome injection needed)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return ProteinAbundanceExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return ProteinAbundanceExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The Zenodo proteome matrix required before processing."""
        return [DATA_FILENAME]

    def download(self) -> None:
        """Download the proteome matrix from Zenodo and verify its sha256."""
        dest = osp.join(self.raw_dir, DATA_FILENAME)
        if osp.exists(dest):
            return
        os.makedirs(self.raw_dir, exist_ok=True)
        log.info("Downloading Zelezniak proteome from %s", DATA_URL)
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
        got = hashlib.sha256(data).hexdigest()
        if got != DATA_SHA256:
            raise RuntimeError(
                f"Zelezniak proteome sha256 mismatch: got {got}, expected {DATA_SHA256}"
            )
        with open(dest, "wb") as handle:
            handle.write(data)
        log.info("Wrote %s (%d bytes, sha256 verified)", dest, len(data))

    @staticmethod
    def _aggregate(sub: pd.DataFrame) -> dict[str, Any]:
        """Aggregate one strain's replicate rows to per-protein mean/se/n dicts."""
        grp = sub.groupby("ORF")["value"].agg(["mean", "std", "count"])
        abundance: dict[str, float] = {}
        se: dict[str, float] = {}
        n_reps: dict[str, int] = {}
        for orf, r in grp.iterrows():
            n = int(r["count"])
            abundance[str(orf)] = float(r["mean"])
            n_reps[str(orf)] = n
            se[str(orf)] = float(r["std"]) / math.sqrt(n) if n > 1 else float("nan")
        return {"abundance": abundance, "se": se, "n": n_reps}

    @post_process
    def process(self) -> None:
        """Aggregate the proteome matrix into per-strain experiments and write LMDB."""
        df = pd.read_csv(osp.join(self.raw_dir, DATA_FILENAME), sep="\t")
        bad = df[~df["ORF"].astype(str).str.match(_SYSTEMATIC_RE)]
        if len(bad):
            raise RuntimeError(f"non-systematic protein ORF ids present: {len(bad)}")

        wt_rows = df[df["KO_ORF"] == _WT]
        if wt_rows.empty:
            raise RuntimeError("Zelezniak matrix missing the WT reference strain")
        self._reference = self._aggregate(wt_rows)

        os.makedirs(self.preprocess_dir, exist_ok=True)
        n_bad_orf = 0
        rows: list[dict[str, Any]] = []
        for ko_orf, sub in df[df["KO_ORF"] != _WT].groupby("KO_ORF"):
            if not _SYSTEMATIC_RE.match(str(ko_orf)):
                n_bad_orf += 1
                continue
            gene = str(sub["KO_gene_name"].iloc[0])
            rows.append({"orf": str(ko_orf), "gene": gene, "agg": self._aggregate(sub)})
        log.info(
            "Zelezniak: %d knockout strains, WT reference with %d proteins, "
            "%d non-systematic KO_ORF skipped",
            len(rows),
            len(self._reference["abundance"]),
            n_bad_orf,
        )
        pd.DataFrame([{"orf": r["orf"], "gene": r["gene"]} for r in rows]).to_csv(
            osp.join(self.preprocess_dir, "data.csv"), index=False
        )

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        with env.begin(write=True) as txn:
            for record_row in tqdm(rows):
                experiment, reference, publication = self.create_experiment(record_row)
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
        log.info("Wrote %d Zelezniak proteome experiments to LMDB", idx)

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(  # type: ignore[override]
        self, row: dict[str, Any]
    ) -> tuple[
        ProteinAbundanceExperiment, ProteinAbundanceExperimentReference, Publication
    ]:
        """Build the ProteinAbundance experiment/reference/publication for one strain."""
        # Background = BY4741 kinase-deletion collection made prototrophic by the pHLUM
        # minichromosome (restores HIS3/LEU2/URA3/MET17); pHLUM not yet modeled here.
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        )
        genotype = Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=row["orf"], perturbed_gene_name=row["gene"]
                )
            ]
        )
        # SWATH-MS on cells in synthetic minimal (SM) liquid medium, 30 C.
        environment = Environment(
            media=Media(name="SM", state="liquid", is_synthetic=True),
            temperature=Temperature(value=30),
        )
        agg = row["agg"]
        phenotype = ProteinAbundancePhenotype(
            protein_abundance=agg["abundance"],
            protein_abundance_se=agg["se"],
            n_replicates=agg["n"],
            measurement_type=MEASUREMENT_TYPE,
        )
        ref = self._reference
        phenotype_reference = ProteinAbundancePhenotype(
            protein_abundance=dict(ref["abundance"]),
            protein_abundance_se=dict(ref["se"]),
            n_replicates=dict(ref["n"]),
            measurement_type=MEASUREMENT_TYPE,
        )
        experiment = ProteinAbundanceExperiment(
            dataset_name=self.name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        reference = ProteinAbundanceExperimentReference(
            dataset_name=self.name,
            genome_reference=genome_reference,
            environment_reference=environment.model_copy(),
            phenotype_reference=phenotype_reference,
        )
        publication = Publication(
            pubmed_id="30195436",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/30195436/",
            doi="10.1016/j.cels.2018.08.001",
            doi_url="https://doi.org/10.1016/j.cels.2018.08.001",
        )
        return experiment, reference, publication


def build_metabolite_s_id_map(kegg_by_metabolite: dict[str, str]) -> dict[str, str]:
    """Map each metabolite id -> a Yeast9 ``s_NNNN`` species id from ``YeastGEM``.

    Hybrid, model-sourced resolution (ids come only FROM the model, never invented):
    prefer the KEGG-compound annotation (matched on the first ``;``-separated ``kegg_id``
    token), fall back to the BiGG-metabolite annotation (matched on the first ``;``
    token of the metabolite id). For a ``;``-merged key, the FIRST sub-id is used.
    Cytosolic (compartment ``c``) hits are preferred; when a metabolite has no cytosolic
    form the available compartment is used (e.g. ``b124tc`` -> mitochondrial ``s_0454``).

    Reusable across targeted-metabolome loaders (e.g. Mulleder) by passing the loader's
    ``{metabolite_id: kegg_id}`` map; only this dataset wires it in for now.
    """
    model = YeastGEM().model
    kegg_index: dict[str, list[Any]] = {}
    bigg_index: dict[str, list[Any]] = {}
    for met in model.metabolites:
        for key, index in (
            ("kegg.compound", kegg_index),
            ("bigg.metabolite", bigg_index),
        ):
            ann = met.annotation.get(key)
            if ann is None:
                continue
            for token in ann if isinstance(ann, list) else [ann]:
                index.setdefault(str(token), []).append(met)

    def pick(cands: list[Any]) -> Any:
        cyto = [m for m in cands if m.compartment == "c"]
        return cyto[0] if cyto else cands[0]

    s_ids: dict[str, str] = {}
    for metabolite_id, kegg_id in kegg_by_metabolite.items():
        first_kegg = str(kegg_id).split(";")[0].strip() if kegg_id else ""
        first_bigg = metabolite_id.split(";")[0].strip()
        cands = kegg_index.get(first_kegg) or bigg_index.get(first_bigg)
        if not cands:
            raise RuntimeError(
                f"no Yeast9 s_NNNN found for metabolite '{metabolite_id}' "
                f"(kegg '{first_kegg}', bigg '{first_bigg}')"
            )
        s_ids[metabolite_id] = str(pick(cands).id)
    return s_ids


@register_dataset
class MetaboliteZelezniak2018Dataset(ExperimentDataset):
    """SRM-MS/MS targeted metabolome of the yeast kinase-knockout collection (95 strains)."""

    def __init__(
        self,
        root: str = "data/torchcell/metabolite_zelezniak2018",
        io_workers: int = 0,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (genotype is systematic; no genome injection needed)."""
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
        """The Zenodo metabolome matrix required before processing."""
        return [METABOLITE_DATA_FILENAME]

    def download(self) -> None:
        """Download the metabolome matrix from Zenodo and verify its sha256."""
        dest = osp.join(self.raw_dir, METABOLITE_DATA_FILENAME)
        if osp.exists(dest):
            return
        os.makedirs(self.raw_dir, exist_ok=True)
        log.info("Downloading Zelezniak metabolome from %s", METABOLITE_DATA_URL)
        req = urllib.request.Request(
            METABOLITE_DATA_URL, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
        got = hashlib.sha256(data).hexdigest()
        if got != METABOLITE_DATA_SHA256:
            raise RuntimeError(
                f"Zelezniak metabolome sha256 mismatch: got {got}, "
                f"expected {METABOLITE_DATA_SHA256}"
            )
        with open(dest, "wb") as handle:
            handle.write(data)
        log.info("Wrote %s (%d bytes, sha256 verified)", dest, len(data))

    @staticmethod
    def _aggregate(sub: pd.DataFrame) -> dict[str, Any]:
        """Aggregate one strain's rows to per-metabolite mean/se/n dicts.

        Rows are POOLED across the ``dataset`` protocol column (1/2/3) and the
        ``replicate`` column per metabolite: mean = pooled mean, SE = sample_SD * n**-0.5
        when n>1 else NaN, n = pooled row count. (README: ``dataset`` = "Protocol used for
        generation"; pooling protocols is an explicit decision -- see module docstring.)
        """
        grp = sub.groupby("metabolite_id")["value"].agg(["mean", "std", "count"])
        level: dict[str, float] = {}
        se: dict[str, float] = {}
        n_reps: dict[str, int] = {}
        for metabolite_id, r in grp.iterrows():
            n = int(r["count"])
            level[str(metabolite_id)] = float(r["mean"])
            n_reps[str(metabolite_id)] = n
            se[str(metabolite_id)] = (
                float(r["std"]) / math.sqrt(n) if n > 1 else float("nan")
            )
        return {"level": level, "se": se, "n": n_reps}

    @post_process
    def process(self) -> None:
        """Aggregate the metabolome matrix into per-strain experiments and write LMDB."""
        df = pd.read_csv(osp.join(self.raw_dir, METABOLITE_DATA_FILENAME), sep="\t")
        nonwt = df[df["genotype"] != _WT]["genotype"].astype(str)
        bad = nonwt[~nonwt.str.match(_SYSTEMATIC_RE)]
        if len(bad):
            raise RuntimeError(
                f"non-systematic strain genotype ids present: {len(bad)}"
            )

        # metabolite_id -> Yeast9 s_NNNN, sourced from YeastGEM (never invented).
        kegg_by_metabolite = {
            str(r["metabolite_id"]): str(r["kegg_id"])
            for _, r in df[["metabolite_id", "kegg_id"]].drop_duplicates().iterrows()
        }
        self._s_id_map = build_metabolite_s_id_map(kegg_by_metabolite)

        wt_rows = df[df["genotype"] == _WT]
        if wt_rows.empty:
            raise RuntimeError("Zelezniak metabolome missing the WT reference strain")
        self._reference = self._aggregate(wt_rows)

        os.makedirs(self.preprocess_dir, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for genotype, sub in df[df["genotype"] != _WT].groupby("genotype"):
            rows.append({"orf": str(genotype), "agg": self._aggregate(sub)})
        log.info(
            "Zelezniak metabolome: %d knockout strains, WT reference with %d "
            "metabolites, %d metabolite ids mapped to Yeast9 s_NNNN",
            len(rows),
            len(self._reference["level"]),
            len(self._s_id_map),
        )
        pd.DataFrame(
            [{"orf": r["orf"], "n_metabolites": len(r["agg"]["level"])} for r in rows]
        ).to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        with env.begin(write=True) as txn:
            for record_row in tqdm(rows):
                experiment, reference, publication = self.create_experiment(record_row)
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
        log.info("Wrote %d Zelezniak metabolome experiments to LMDB", idx)

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def _phenotype(self, agg: dict[str, Any]) -> MetabolitePhenotype:
        """Build a MetabolitePhenotype from an aggregated per-metabolite dict."""
        level = dict(agg["level"])
        se = dict(agg["se"])
        # SE keys must be a subset of level keys; the all-NaN case collapses to None.
        level_se: dict[str, float] | None = se
        if all(math.isnan(v) for v in se.values()):
            level_se = None
        targets = {m: self._s_id_map[m] for m in level}
        return MetabolitePhenotype(
            metabolite_level=level,
            metabolite_level_se=level_se,
            n_replicates=dict(agg["n"]),
            measurement_type=METABOLITE_MEASUREMENT_TYPE,
            target_metabolite_ids=targets,
        )

    def create_experiment(  # type: ignore[override]
        self, row: dict[str, Any]
    ) -> tuple[MetaboliteExperiment, MetaboliteExperimentReference, Publication]:
        """Build the Metabolite experiment/reference/publication for one strain."""
        # Background = BY4741 kinase-deletion collection made prototrophic by the pHLUM
        # minichromosome (restores HIS3/LEU2/URA3/MET17); pHLUM not yet modeled here.
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        )
        # No gene-name column in the metabolome file (only the systematic ORF), so the
        # systematic id is used for both names (as in the Mulleder loader).
        genotype = Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=row["orf"], perturbed_gene_name=row["orf"]
                )
            ]
        )
        # SRM-MS/MS on cells in synthetic minimal (SM) liquid medium, 30 C (as proteome).
        environment = Environment(
            media=Media(name="SM", state="liquid", is_synthetic=True),
            temperature=Temperature(value=30),
        )
        phenotype = self._phenotype(row["agg"])
        # Reference = the measured WT baseline RESTRICTED to the metabolites this strain
        # measured. Targeted-metabolomics coverage is sparse and per-strain (WT itself
        # measured only 45 of the 50 ids -- never adp/amp/atp/e4p/fum), so a strain can
        # measure metabolites the WT lacks; those simply have no WT baseline. Restricting
        # keeps reference keys a subset of the experiment's and every reference value a
        # real WT measurement (never invented). Every strain shares >=1 metabolite with WT.
        exp_keys = set(row["agg"]["level"])
        ref = self._reference
        ref_agg = {
            "level": {k: v for k, v in ref["level"].items() if k in exp_keys},
            "se": {k: v for k, v in ref["se"].items() if k in exp_keys},
            "n": {k: v for k, v in ref["n"].items() if k in exp_keys},
        }
        phenotype_reference = self._phenotype(ref_agg)
        experiment = MetaboliteExperiment(
            dataset_name=self.name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        reference = MetaboliteExperimentReference(
            dataset_name=self.name,
            genome_reference=genome_reference,
            environment_reference=environment.model_copy(),
            phenotype_reference=phenotype_reference,
        )
        publication = Publication(
            pubmed_id="30195436",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/30195436/",
            doi="10.1016/j.cels.2018.08.001",
            doi_url="https://doi.org/10.1016/j.cels.2018.08.001",
        )
        return experiment, reference, publication


def main() -> None:
    """Build/load the datasets for interactive debugging.

    Loads the existing LMDB if already built. To step through
    ``process()``/``create_experiment`` under a debugger, delete ``<root>/processed``
    first so the build re-runs.
    """
    from dotenv import load_dotenv

    load_dotenv()
    proteome_root = osp.join(
        os.environ["DATA_ROOT"], "data/torchcell/proteome_zelezniak2018"
    )
    proteome = ProteomeZelezniak2018Dataset(root=proteome_root)
    print(f"proteome len = {len(proteome)}")
    print(proteome[0])

    metabolite_root = osp.join(
        os.environ["DATA_ROOT"], "data/torchcell/metabolite_zelezniak2018"
    )
    metabolite = MetaboliteZelezniak2018Dataset(root=metabolite_root)
    print(f"metabolite len = {len(metabolite)}")
    print(metabolite[0])


if __name__ == "__main__":
    main()
