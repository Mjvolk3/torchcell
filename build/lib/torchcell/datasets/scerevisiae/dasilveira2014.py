# torchcell/datasets/scerevisiae/dasilveira2014
# [[torchcell.datasets.scerevisiae.dasilveira2014]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/dasilveira2014
# Test file: tests/torchcell/datasets/scerevisiae/test_dasilveira2014.py
"""da Silveira dos Santos 2014 yeast kinase/phosphatase lipidome dataset.

da Silveira dos Santos A et al. 2014, "Systematic lipidomic analysis of yeast protein
kinase and phosphatase mutants" (Mol Biol Cell 25(20):3234-3246,
doi:10.1091/mbc.E14-03-0851, PMID 25143408, PMC4196872). A mass-spectrometry lipidomic
screen of the nonessential protein-kinase/phosphatase deletion collection, quantifying
147 acyl-chain-resolved lipid species (glycerophospholipids + sphingolipids by MRM-MS /
Orbitrap direct infusion; ergosterol by GC-MS) as RELATIVE abundances in arbitrary units
(a.u.), each species quantified against a spiked internal standard.

Source (sha256-pinned, in the library mirror ``$DATA_ROOT/torchcell-library/
daSystematicLipidomicAnalysis2014/data/``):

- ``TableS4_complete_dataset_all_lipids.xlsx`` -- the lipid matrix. We ingest the
  ``Quant`` sheet (relative quantities in a.u., one aggregated value per mutant x lipid;
  the absolute-per-mutant input appropriate for ``MetabolitePhenotype`` levels + a
  measured-WT baseline). Orientation: rows = strains, cols = ``Systematic Name`` (col 0),
  ``Standard Name`` (col 1), then 147 lipid species. 130 rows total.
- ``TableS10_lipidX_chebi_ids.xlsx`` -- per-lipid LipidX id + Name + Formula + ChEBI id.
  Copied for provenance; its lipid-name -> ChEBI map is written to
  ``preprocess/lipid_chebi.csv`` for a deferred follow-up (see target ids below).

Maps to ``MetaboliteExperiment`` / ``MetabolitePhenotype`` (one record per mutant):
``metabolite_level = {lipid_species_name -> relative abundance (a.u.)}`` with
``measurement_type = "lipidomics_ms_relative_abundance_au"``. The value is an ARBITRARY
relative abundance, NOT a concentration.

WILD-TYPE REFERENCE (deviates from a first-pass "no WT row" assumption -- verified against
the data): the ``Quant`` sheet's 130 rows are NOT all mutants. THREE rows are wild-type
controls -- ``Y7092`` (WT1), ``Y7220`` (WT2), ``BY4741`` (WT3) -- which are strain
identifiers, not ORFs, so they cannot be deletion records. Excluding them leaves 127
mutant rows (matching the paper's screened-mutant count), and their per-lipid MEAN is used
as the measured WT baseline (like the Zelezniak metabolome loader's measured WT). This is
strictly more faithful than a population mean over mutants: there is an actual measured WT.
Each record's reference is RESTRICTED to the lipids that strain measured (and the WT
measured), keeping reference keys a subset of the experiment's (WT covers all 147 lipids
collectively; every mutant overlaps WT by >=125 lipids). ``reference_centered=False``.

Missing values are NaN in the ``Quant`` sheet (1038 cells) and are DROPPED per strain, so
each record carries only the lipids that strain measured (125-147 lipids/strain).

Replicate structure (sourced, Methods "Lipid extraction and analysis", MRM-MS paragraph):
"Two independent biological replicates were analyzed, each of which comprised up to six
technical replicates." (also Results: "biological duplicates with up to six technical
replicates per biological replicate"). The ``Quant`` value is a single aggregated relative
quantity per mutant x lipid (batch-adjusted/normalized across the extraction batch), with
no released per-strain SE, so ``n_replicates = 2`` per lipid (the INDEPENDENT biological-
replicate unit; the up-to-six technical replicates are not independent) and
``metabolite_level_se = None``. The WT reference's ``n_replicates`` is the number of the
three WT control rows measuring each lipid.

Background strain (sourced): Methods "Strains" defer the strain list to Supplemental Table
S1, which states "Kinase and phosphatase knockout collection was obtained from Claudine
Kraft and Matthias Peter (ETH Zurich)" without naming BY4741/BY4742 in prose. The screen's
own WT controls include a row explicitly labeled ``BY4741`` (WT3), and the collection is
the standard nonessential KanMX kinase/phosphatase deletion set (BY4741, MATa background),
so ``ReferenceGenome(strain="BY4741")`` is used with this evidence recorded (not guessed).

Growth (sourced, Methods "Strains"): YPD rich medium (2% glucose, 1% Bacto Peptone, 2%
Bacto Yeast Extract, + MES/Trp/uracil/adenine), grown to early exponential phase
(1-2 OD600 units/ml) at 30 C.

Target ids: ``MetabolitePhenotype.target_metabolite_ids`` is specifically a Yeast9
``s_NNNN`` map for constraint-based-model linkage. These lipids map to ChEBI (Table S10),
not Yeast9 ``s_NNNN``, so target ids are left ``None`` here and the ChEBI mapping is
deferred to a follow-up (as Mulleder deferred its amino-acid -> ``s_NNNN`` ids); the
lipid-name -> ChEBI table is emitted to ``preprocess/lipid_chebi.csv`` for that follow-up.

Counting note: the paper prose reports 129 screened mutants; the released ``Quant`` matrix
has 127 mutant rows (+ 3 WT controls). We keep all 127 released mutant rows and flag the
129-vs-127 discrepancy.
"""

import hashlib
import logging
import os
import os.path as osp
import pickle
import re
import shutil
from typing import Any, cast

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
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# The value is an ARBITRARY relative abundance (a.u.), NOT a concentration.
MEASUREMENT_TYPE = "lipidomics_ms_relative_abundance_au"

_LIBRARY_CITATION_KEY = "daSystematicLipidomicAnalysis2014"
_SYSTEMATIC_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$")

# Wild-type control rows in the Quant sheet: strain ids (not ORFs), used as the measured
# WT reference, NOT as deletion records.
_WT_ROW_IDS = {"Y7092", "Y7220", "BY4741"}

_QUANT_SHEET = "Quant"
_SYS_COL = "Systematic Name"
_STD_COL = "Standard Name"

# TableS4 -- the lipid matrix (Quant sheet consumed).
DATA_FILENAME = "TableS4_complete_dataset_all_lipids.xlsx"
DATA_SHA256 = "91409229756c132823e6e7a8dbe552d4d7451833b2ff902740f24a29bced3894"

# TableS10 -- per-lipid LipidX/ChEBI ids (copied for provenance; ChEBI mapping deferred).
CHEBI_FILENAME = "TableS10_lipidX_chebi_ids.xlsx"
CHEBI_SHA256 = "29a7ed0a11af4700fa051b99717e4686e0399a59c7a7b9be130ae674ed5d58f9"

# Biological-replicate count per lipid (sourced: "Two independent biological replicates").
_N_BIOLOGICAL_REPLICATES = 2


@register_dataset
class MetaboliteDaSilveira2014Dataset(ExperimentDataset):
    """MS lipidome of the yeast kinase/phosphatase deletion collection (127 mutants)."""

    def __init__(
        self,
        root: str = "data/torchcell/metabolite_dasilveira2014",
        io_workers: int = 0,
        genome: SCerevisiaeGenome | None = None,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset; a genome resolves/validates ORFs against R64."""
        self.genome = genome
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
        """The Table S4 lipid matrix + Table S10 ChEBI ids required before processing."""
        return [DATA_FILENAME, CHEBI_FILENAME]

    def _copy_pinned(self, filename: str, sha256: str) -> None:
        """Copy one sha256-pinned file from the library mirror into raw_dir + verify."""
        dest = osp.join(self.raw_dir, filename)
        if not osp.exists(dest):
            data_root = os.environ["DATA_ROOT"]
            src = osp.join(
                data_root, "torchcell-library", _LIBRARY_CITATION_KEY, "data", filename
            )
            if not osp.exists(src):
                raise RuntimeError(
                    f"library mirror data file not found: {src}. This dataset's source is "
                    f"the sha256-pinned {filename} in the torchcell-library mirror."
                )
            shutil.copyfile(src, dest)
        digest = hashlib.sha256(open(dest, "rb").read()).hexdigest()
        if digest != sha256:
            raise RuntimeError(
                f"{filename} sha256 mismatch: got {digest}, expected {sha256}"
            )
        log.info("Verified %s (sha256 %s)", dest, sha256)

    def download(self) -> None:
        """Copy the pinned Table S4 + Table S10 from the library mirror and verify."""
        os.makedirs(self.raw_dir, exist_ok=True)
        self._copy_pinned(DATA_FILENAME, DATA_SHA256)
        self._copy_pinned(CHEBI_FILENAME, CHEBI_SHA256)

    def _resolve_systematic(self, name: str) -> str | None:
        """Validate/resolve a source systematic name against the R64 genome."""
        genome = cast(SCerevisiaeGenome, self.genome)
        name = name.strip()
        if name in genome.gene_set:
            return name
        candidates = genome.alias_to_systematic.get(name.upper(), [])
        if candidates:
            return candidates[0]
        return None

    def _lipid_chebi_map(self) -> dict[str, str]:
        """Lipid Name -> ChEBI id from Table S10 (header on the 3rd row)."""
        s10 = pd.read_excel(osp.join(self.raw_dir, CHEBI_FILENAME), header=2)
        s10 = s10.rename(
            columns={
                s10.columns[0]: "ID",
                s10.columns[1]: "Name",
                s10.columns[2]: "Formula",
                s10.columns[3]: "ChEBI",
            }
        )
        return {
            str(r["Name"]).strip(): str(r["ChEBI"]).strip()
            for _, r in s10.iterrows()
            if pd.notna(r["Name"]) and pd.notna(r["ChEBI"])
        }

    @post_process
    def process(self) -> None:
        """Parse the Quant sheet into per-mutant Metabolite experiments and write LMDB."""
        if self.genome is None:
            raise RuntimeError(
                "MetaboliteDaSilveira2014Dataset requires an injected SCerevisiaeGenome "
                "to validate systematic ORF ids against R64."
            )
        path = osp.join(self.raw_dir, DATA_FILENAME)
        df = pd.read_excel(path, sheet_name=_QUANT_SHEET)
        lipid_cols = [c for c in df.columns if c not in (_SYS_COL, _STD_COL)]

        # Measured WT baseline: per-lipid mean over the three WT control rows (skip NaN).
        wt_df = df[df[_SYS_COL].isin(_WT_ROW_IDS)]
        if len(wt_df) != len(_WT_ROW_IDS):
            raise RuntimeError(
                f"expected {len(_WT_ROW_IDS)} WT control rows, found {len(wt_df)}"
            )
        wt_mean = wt_df[lipid_cols].mean(axis=0, skipna=True)
        wt_count = wt_df[lipid_cols].notna().sum(axis=0)
        self._wt_level = {
            c: float(wt_mean[c]) for c in lipid_cols if wt_mean.notna()[c]
        }
        self._wt_n = {c: int(wt_count[c]) for c in self._wt_level}

        # ChEBI map (Table S10) -> emitted for the deferred target-id follow-up (not used
        # for target_metabolite_ids, which is Yeast9 s_NNNN specific).
        chebi_by_lipid = self._lipid_chebi_map()
        chebi_missing = [c for c in lipid_cols if c.strip() not in chebi_by_lipid]
        log.info(
            "da Silveira: %d/%d lipids have a Table S10 ChEBI id (%d missing)",
            len(lipid_cols) - len(chebi_missing),
            len(lipid_cols),
            len(chebi_missing),
        )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        pd.DataFrame(
            [
                {"lipid": c, "chebi": chebi_by_lipid.get(c.strip(), "")}
                for c in lipid_cols
            ]
        ).to_csv(osp.join(self.preprocess_dir, "lipid_chebi.csv"), index=False)

        mut_df = df[~df[_SYS_COL].isin(_WT_ROW_IDS)]
        n_unresolved = 0
        unresolved: list[str] = []
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for _, row in mut_df.iterrows():
            source_orf = str(row[_SYS_COL]).strip()
            orf = self._resolve_systematic(source_orf)
            if orf is None:
                n_unresolved += 1
                unresolved.append(source_orf)
                continue
            if orf in seen:
                log.warning("duplicate ORF %s after resolution; keeping first", orf)
                continue
            seen.add(orf)
            std = row[_STD_COL]
            gene_name = str(std).strip() if pd.notna(std) else orf
            level = {c: float(row[c]) for c in lipid_cols if pd.notna(row[c])}
            rows.append({"orf": orf, "gene": gene_name, "level": level})
        log.info(
            "da Silveira: %d mutant records, %d WT control rows -> measured reference "
            "(%d lipids), %d unresolved ORFs%s",
            len(rows),
            len(wt_df),
            len(self._wt_level),
            n_unresolved,
            f" ({unresolved})" if unresolved else "",
        )
        pd.DataFrame(
            [
                {"orf": r["orf"], "gene": r["gene"], "n_lipids": len(r["level"])}
                for r in rows
            ]
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
        log.info("Wrote %d da Silveira lipidome experiments to LMDB", idx)

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(  # type: ignore[override]
        self, row: dict[str, Any]
    ) -> tuple[MetaboliteExperiment, MetaboliteExperimentReference, Publication]:
        """Build the Metabolite experiment/reference/publication for one mutant."""
        # BY4741 nonessential kinase/phosphatase KanMX deletion collection (see docstring:
        # BY4741 appears as a WT control row in the released matrix).
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
        # YPD rich medium, early exponential phase, 30 C (Methods "Strains").
        environment = Environment(
            media=Media(name="YPD", state="liquid", is_synthetic=False),
            temperature=Temperature(value=30),
        )
        level = row["level"]
        phenotype = MetabolitePhenotype(
            metabolite_level=dict(level),
            metabolite_level_se=None,  # single aggregated Quant value; no released SE
            n_replicates={lip: _N_BIOLOGICAL_REPLICATES for lip in level},
            measurement_type=MEASUREMENT_TYPE,
            target_metabolite_ids=None,  # deferred: lipid -> ChEBI (Table S10), not s_NNNN
        )
        # Reference = measured WT baseline RESTRICTED to the lipids this strain measured
        # (keeps reference keys a subset of the experiment's; every mutant overlaps WT).
        ref_level = {k: v for k, v in self._wt_level.items() if k in level}
        phenotype_reference = MetabolitePhenotype(
            metabolite_level=ref_level,
            metabolite_level_se=None,
            n_replicates={k: self._wt_n[k] for k in ref_level},
            measurement_type=MEASUREMENT_TYPE,
            target_metabolite_ids=None,
        )
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
            pubmed_id="25143408",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/25143408/",
            doi="10.1091/mbc.E14-03-0851",
            doi_url="https://doi.org/10.1091/mbc.E14-03-0851",
        )
        return experiment, reference, publication


def main() -> None:
    """Build/load the dataset for interactive debugging.

    A genome is REQUIRED (ORFs are validated against R64). Loads the existing LMDB if
    already built; to step through ``process()``/``create_experiment`` under a debugger,
    delete ``<root>/processed`` first so the build re-runs.
    """
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    root = osp.join(data_root, "data/torchcell/metabolite_dasilveira2014")
    dataset = MetaboliteDaSilveira2014Dataset(root=root, genome=genome)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()
