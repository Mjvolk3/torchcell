# torchcell/datasets/ecoli/fitness_browser_ecoli
# [[torchcell.datasets.ecoli.fitness_browser_ecoli]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/fitness_browser_ecoli
# Test file: tests/torchcell/datasets/ecoli/test_fitness_browser_ecoli.py
"""Fitness Browser, E. coli BW25113 (``Keio``): the browser's own experiments, keyed by
publication.

The Price 2018 decomposition (#931, plan decision 16; owner policy of 2026-10-10: "a served
compendium is a sanity check, not the source; load each original study from its own
release"). ONE loader for the organism, reading the Fitness Browser deposit
(:mod:`torchcell.datasets.fitness_browser`), with every stored record's ``Publication``
the browser's ``pubId`` for its sample (:data:`SOURCE_STUDIES`).

WHAT IS STORED, by the deposit's 183 Keio experiments (measured 2026-10-12, the inventory
of ``experiments/database/scripts/fitness_browser_inventory.py``):

* ``Price21b`` (GapMind for carbon sources, doi:10.1101/2021.11.02.466981), 6 samples:
  D-mannitol at 10, 5 and 2.5 mM as sole carbon source on ``RCH2_defined_noCarbon``, in
  duplicate. The browser is this study's own release (``source_kind = primary``); no
  compendium carries these samples, so they have no L4 counterpart.

NOT STORED, each with its fate in ``preprocess/samples.json`` and counted in
``preprocess/dropped_records.json``:

* ``Wetmore15`` (87) and ``Price18`` (71): each study is already loaded from its own
  release, ``RbTnseqWetmore2015Dataset`` (statistics 1.0.0) and
  ``RbTnseqPrice2018EcoliDataset`` (the compendium paper's own release, statistics 1.0.3),
  so storing the browser's copy would store the same sample twice (G4 duplication by
  content). The browser equals the Price 2018 release to 8.9e-16 over 158 samples x 3,789
  genes (the 2026-10-11 audit); this loader's verifier re-measures it against the served
  store as its L4 row. Those datasets serve the sample or refuse it under their own
  rules (the 2021 withdrawal, the motility assay, the MOPS Rich medium).
* the 19 experiments with an empty ``pubId`` (set57's ``Ying_*`` substrate mixes on
  ``SDM_noCarbon``, set5's sulfur sources on ``M63_no_sulfur``): an unattributed release;
  whether it counts as a study to load is an open owner decision of #931.

RECORDS. One ``BacterialEnvironmentResponseExperiment`` per (released locus, stored sample):

* the value is the browser's gene fitness (``FitByExp_Keio.fit``), a signed log2 ratio
  normalized so the typical gene is 0; the browser's moderated t is stored verbatim in
  ``environment_response_t_statistic``, and the standard error ``sqrt((f / t)^2 - 0.1^2)``
  (:func:`torchcell.datasets.fitness_browser.derived_se`) in ``environment_response_se``;
* genotype: one ``TransposonInsertionPerturbation`` per gene in the sample's own
  ``mutantLibrary`` (``Keio_ML9a`` for all six); the transposon is None, because no
  deposited byte states it for ``Keio_ML9a`` (Price 2018 Table S20 states Tn5 for
  ``KEIO_ML9``, and the relation of the two library names is not stated). Every gene is
  stored under the locus tag the deposit gives it, ``Gene.sysName`` (an MG1655 b-number,
  namespace ``ecoli_k12_mg1655_bnumber``), with ``Gene.gene`` as its common name; no
  assembly is consulted and none decides what is stored (owner, 2026-10-12: "the genome
  is separate.. it doesn't gate anything"). The reference genome states the library
  strain, BW25113 (``Organism.strain``);
* environment: the sample's ``media`` through :data:`MEDIA_BY_LABEL`, its carbon source as
  an ``EnvironmentPhysicalPerturbation``, ``temperature``, aerobic. ``nGenerations`` is
  empty for all six, so the duration is a typed gap;
* reference: the typical gene of the same sample, response 0 by the normalization.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
from collections.abc import Callable, Iterator, Sequence
from typing import Any, ClassVar, Final

from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import RCH2_DEFINED_NOCARBON_PRICE2018
from torchcell.datamodels.schema import (
    AssayType,
    AssemblyReferenceGenome,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    Media,
    PhysicalFactor,
    Publication,
    Temperature,
    TransposonInsertionPerturbation,
)
from torchcell.datasets import fitness_browser as fb
from torchcell.datasets.bacteria_common import (
    STRAIN_GENE_NAMESPACES,
    assembly_reference,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.ecoli import price2018
from torchcell.datasets.ecoli import wetmore2015 as wetmore
from torchcell.sequence.genome.ecoli.k12 import EcoliK12StrainName
from torchcell.verification.report import Level, LevelResult, VerificationReport
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

log = logging.getLogger(__name__)

#: The raw-mirror key the deposit sits under; the records' Publication is the study's.
CITATION_KEY: Final = "priceMutantPhenotypesThousands2018"
ORG_ID: Final = "Keio"
REFERENCE_STRAIN: Final = "BW25113"
DATASET_ROOT_REL: Final = "data/torchcell/rbtnseq_fitness_browser_ecoli"
RAW_FILE: Final = "feba.db"

# --------------------------------------------------------------------------- #
# Studies: the deposit's Keio publications
# --------------------------------------------------------------------------- #
SOURCE_STUDIES: Final[dict[str, fb.SourceStudy]] = {
    "Wetmore15": fb.SourceStudy(
        pub_id="Wetmore15",
        browser_title="Rapid Quantification of Mutant Fitness in Diverse Bacteria by "
        "Sequencing Randomly Bar-Coded Transposons",
        browser_url="https://doi.org/10.1128/mBio.00306-15",
        doi=wetmore.PAPER_DOI,
        citation_key=wetmore.CITATION_KEY,
        source_kind="primary",
        own_release_dataset="RbTnseqWetmore2015Dataset",
        note="the authors' own data page (statistics 1.0.0) is loaded by "
        "RbTnseqWetmore2015Dataset",
    ),
    "Price18": fb.SourceStudy(
        pub_id="Price18",
        browser_title="Mutant Phenotypes for Thousands of Bacterial Genes of Unknown "
        "Function",
        browser_url="http://genomics.lbl.gov/supplemental/bigfit/",
        doi=price2018.PAPER_DOI,
        citation_key=price2018.CITATION_KEY,
        source_kind="compendium_only",
        own_release_dataset="RbTnseqPrice2018EcoliDataset",
        note="the compendium's own paper: its release (bigfit, statistics 1.0.3) is "
        "loaded by RbTnseqPrice2018EcoliDataset and equals the browser to 8.9e-16",
    ),
    "Price21b": fb.SourceStudy(
        pub_id="Price21b",
        browser_title="GapMind for Carbon Sources: Automated annotations of catabolic "
        "pathways",
        browser_url="https://doi.org/10.1101/2021.11.02.466981",
        doi="10.1101/2021.11.02.466981",
        citation_key=None,
        source_kind="primary",
        issues=(939,),
        note="the browser's URL names this DOI (a preprint); no mirror holds the paper",
    ),
}

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
#: The browser's ``media`` label -> the library object. A label outside it is a fate.
MEDIA_BY_LABEL: Final[dict[str, Media]] = {
    "RCH2_defined_noCarbon": RCH2_DEFINED_NOCARBON_PRICE2018
}
#: The browser's ``expGroup`` -> the factor its one condition varies.
FACTOR_BY_GROUP: Final[dict[str, PhysicalFactor]] = {
    "carbon source": PhysicalFactor.carbon_source
}
#: The media a group may name; a pairing outside this is refused.
GROUP_MEDIA: Final[dict[str, frozenset[str]]] = {
    "carbon source": frozenset({"RCH2_defined_noCarbon"})
}
UNITS_BY_LABEL: Final[dict[str, ConcentrationUnit]] = {
    "mM": ConcentrationUnit.millimolar
}

#: The publications the 2021 withdrawal covers: the note sits on Price 2018's page and
#: names "this publication"; its four samples were first reported by Wetmore 2015.
WITHDRAWAL_SCOPE: Final = frozenset({"Wetmore15", "Price18"})


def unrepresentable(e: fb.FbExperiment) -> str | None:
    """Why the sample's condition columns have no typed representation, or None."""
    if e.aerobic != "Aerobic" or e.liquid != "Liquid":
        return f"aerobic={e.aerobic!r}, liquid={e.liquid!r}"
    if e.exp_group not in FACTOR_BY_GROUP:
        return f"group {e.exp_group!r}"
    if e.media not in GROUP_MEDIA[e.exp_group]:
        return f"group {e.exp_group!r} on medium {e.media!r}"
    if len(e.conditions) != 1:
        return f"{len(e.conditions)} condition columns"
    (c,) = e.conditions
    if c.condition is None or c.concentration is None:
        return f"condition {c.condition!r} at {c.concentration_text!r}"
    if c.units not in UNITS_BY_LABEL:
        return f"unit {c.units!r}"
    if e.temperature_c is None:
        return "no temperature"
    return None


def decide(e: fb.FbExperiment) -> fb.SampleDecision:
    """The sample's fate, by the first rule that applies.

    Order: no publication, the 2021 withdrawal, a study loaded from its own release, a
    medium outside the library, an unrepresentable condition. A publication missing from
    :data:`SOURCE_STUDIES` is refused (the table is the attribution, never a default).
    """
    if e.pub_id is None:
        return fb.decision(e, "unattributed_release")
    if e.pub_id not in SOURCE_STUDIES:
        raise KeyError(f"{e.exp_name}: pubId {e.pub_id!r} is not in SOURCE_STUDIES")
    study = SOURCE_STUDIES[e.pub_id]
    named = {c.condition for c in e.conditions}
    if e.pub_id in WITHDRAWAL_SCOPE and named & wetmore.AUTHOR_WITHDRAWAL.conditions:
        return fb.decision(e, "author_withdrawn", wetmore.AUTHOR_WITHDRAWAL.rule)
    if study.own_release_dataset is not None:
        return fb.decision(e, "own_release_loaded", study.own_release_dataset)
    if e.media not in MEDIA_BY_LABEL:
        return fb.decision(e, "medium_not_in_media_library", e.media)
    reason = unrepresentable(e)
    if reason is not None:
        return fb.decision(e, "condition_not_representable", reason)
    return fb.decision(e, "stored")


DURATION_GAP_NOTE = (
    "the deposit's Experiment.nGenerations is empty for this sample and no growth-time "
    "column exists"
)


def build_environment(e: fb.FbExperiment) -> Environment:
    """A stored sample's medium, carbon source, temperature, and generations or a gap."""
    reason = unrepresentable(e)
    if reason is not None:
        raise ValueError(f"{e.exp_name}: {reason}")
    (c,) = e.conditions
    if c.condition is None or c.concentration is None or c.units is None:
        raise ValueError(f"{e.exp_name}: incomplete condition {c}")
    if e.temperature_c is None:
        raise ValueError(f"{e.exp_name}: no temperature")
    gaps = [
        ProvenanceGap(
            field="duration_hours",
            reason=ProvenanceGapReason.not_reported_by_primary,
            looked_in=fb.FEBA_PROVENANCE,
            note=DURATION_GAP_NOTE,
        )
    ]
    if e.n_generations is None:
        gaps.append(
            ProvenanceGap(
                field="duration_generations",
                reason=ProvenanceGapReason.not_reported_by_primary,
                looked_in=fb.FEBA_PROVENANCE,
                note=DURATION_GAP_NOTE,
            )
        )
    return Environment(
        media=MEDIA_BY_LABEL[e.media],
        temperature=Temperature(value=e.temperature_c),
        perturbations=[
            EnvironmentPhysicalPerturbation(
                factor=FACTOR_BY_GROUP[e.exp_group],
                agent=resolved_compound(c.condition),
                magnitude=Concentration(
                    value=c.concentration, unit=UNITS_BY_LABEL[c.units]
                ),
            )
        ],
        aerobicity="aerobic",
        duration_generations=e.n_generations,
        provenance_gaps=gaps,
    )


# --------------------------------------------------------------------------- #
# Genes, as the deposit names them
# --------------------------------------------------------------------------- #
#: The browser's Keio ``Gene.sysName`` is an MG1655 b-number; the records keep it in that
#: namespace. The library strain is BW25113 (``Organism.strain``), which the reference
#: genome states; no assembly decides what is stored (owner, 2026-10-12: "the genome is
#: separate.. it doesn't gate anything").
GENE_NAMESPACE: Final = STRAIN_GENE_NAMESPACES["MG1655"]
DESCRIPTION = (
    "Transposon insertion disrupting a gene; gene-level call over the gene's usable "
    "insertion strains (no per-strain barcode or mapped site). Locus tag as the Fitness "
    "Browser gives it (Gene.sysName, an MG1655 b-number); the library strain is BW25113"
)


def build_genotype(gene: fb.ReleasedGene, library: str) -> Genotype:
    """A transposon insertion in one gene of the sample's library, as released."""
    return Genotype(
        perturbations=[
            TransposonInsertionPerturbation(
                systematic_gene_name=gene.tag,
                perturbed_gene_name=gene.name,
                gene_namespace=GENE_NAMESPACE,
                description=DESCRIPTION,
                library_pool=library,
            )
        ]
    )


# --------------------------------------------------------------------------- #
# Phenotype and reference
# --------------------------------------------------------------------------- #
UNITS = (
    "RB-TnSeq gene fitness (Fitness Browser FitByExp.fit): log2 ratio of the gene's "
    "insertion-strain barcode abundance at the end of growth over the time-zero sample, "
    "weighted over the gene's usable strains, typical gene = 0; t = the browser's "
    "moderated t (Wetmore 2015, sigma 0.1); SE = sqrt((fit / t)^2 - 0.1^2)"
)
UNITS_REFERENCE = "the typical gene of this sample (gene fitness is normalized to 0)"
#: Gene fitness averages a gene's strains; how many is not in the deposit.
_GAPS: Final = (
    ProvenanceGap(
        field="environment_response_uncertainty",
        reason=ProvenanceGapReason.not_reported_by_primary,
        note="the deposit releases the moderated t, not an uncertainty; the standard "
        "error is derived from it into environment_response_se",
    ),
    ProvenanceGap(
        field="n_samples", reason=ProvenanceGapReason.not_reported_by_primary
    ),
    ProvenanceGap(
        field="sample_unit", reason=ProvenanceGapReason.not_reported_by_primary
    ),
)


def build_phenotype(
    fitness: float, t: float, screen_id: str
) -> EnvironmentResponsePhenotype:
    """One gene's fitness in one sample, its t verbatim and its derived standard error."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=fitness,
        environment_response_se=fb.derived_se(fitness, t),
        environment_response_t_statistic=t,
        screen_id=screen_id,
        units=UNITS,
        provenance_gaps=list(_GAPS),
    )


def build_reference(
    dataset_name: str,
    environment: Environment,
    genome_reference: AssemblyReferenceGenome,
    screen_id: str,
) -> BacterialEnvironmentResponseExperimentReference:
    """The typical gene of the same sample: fitness 0 by the normalization."""
    return BacterialEnvironmentResponseExperimentReference(
        dataset_name=dataset_name,
        genome_reference=genome_reference,
        environment_reference=environment,
        phenotype_reference=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.log2_ratio,
            assay_type=AssayType.pooled_competitive_growth_barcode,
            environment_response=0.0,
            screen_id=screen_id,
            units=UNITS_REFERENCE,
        ),
    )


def iter_records(
    dataset_name: str,
    stored: Sequence[fb.FbExperiment],
    values: fb.FbValues,
    genes: Sequence[fb.ReleasedGene],
    genome_reference: AssemblyReferenceGenome,
) -> Iterator[
    tuple[
        BacterialEnvironmentResponseExperiment,
        BacterialEnvironmentResponseExperimentReference,
        Publication,
    ]
]:
    """Every record, sample by sample then gene by sorted locusId."""
    if tuple(g.locus_id for g in genes) != values.loci:
        raise ValueError("the gene rows are not the value matrix's loci")
    for j, e in enumerate(stored):
        if values.experiments[j] != e.exp_name:
            raise ValueError(f"column {j} is {values.experiments[j]}, not {e.exp_name}")
        environment = build_environment(e)
        reference = build_reference(
            dataset_name, environment, genome_reference, e.screen_id
        )
        publication = SOURCE_STUDIES[str(e.pub_id)].publication
        for i, gene in enumerate(genes):
            yield (
                BacterialEnvironmentResponseExperiment(
                    dataset_name=dataset_name,
                    genotype=build_genotype(gene, e.mutant_library),
                    environment=environment,
                    phenotype=build_phenotype(
                        float(values.fitness[i, j]), float(values.t[i, j]), e.screen_id
                    ),
                ),
                reference,
                publication,
            )


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
#: Measured on the 2026-10-12 deposit: 6 stored samples x 3,789 released loci.
EXPECTED_SAMPLES: Final = 6
EXPECTED_GENES: Final = 3789
EXPECTED_RECORDS: Final = EXPECTED_SAMPLES * EXPECTED_GENES


class LoaderInventory(BaseModel):
    """The loader's sample, gene and record accounting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_experiments: int
    fates: dict[str, int]
    fates_by_publication: dict[str, dict[str, int]]
    fate_descriptions: dict[fb.Fate, str]
    stored_samples: tuple[str, ...]
    n_loci: int
    stored_records: int
    studies: dict[str, fb.SourceStudy]


def loader_inventory(
    decisions: Sequence[fb.SampleDecision], n_loci: int
) -> LoaderInventory:
    """The accounting of one build."""
    stored = tuple(d.exp_name for d in decisions if d.fate == "stored")
    return LoaderInventory(
        n_experiments=len(decisions),
        fates=fb.fate_counts(decisions),
        fates_by_publication=fb.fate_counts_by_publication(decisions),
        fate_descriptions=dict(fb.FATE_DESCRIPTIONS),
        stored_samples=stored,
        n_loci=n_loci,
        stored_records=len(stored) * n_loci,
        studies=dict(SOURCE_STUDIES),
    )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@register_dataset
class FitnessBrowserEcoliDataset(ExperimentDataset):
    """Fitness Browser Keio experiments not loaded from another release, by publication."""

    organism = Organism.ecoli

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = "BW25113"

    def __init__(
        self,
        root: str = DATASET_ROOT_REL,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; no genome is read, the deposit names every gene."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialEnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialEnvironmentResponseExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The deposit, linked into ``raw/`` by ``download``."""
        return [RAW_FILE]

    def download(self) -> None:
        """Link the pinned deposit into ``raw/`` after verifying its size and sha256."""
        source = fb.feba_path()
        fb.verify_feba(source)
        os.makedirs(self.raw_dir, exist_ok=True)
        link_verified(source, osp.join(self.raw_dir, RAW_FILE), fb.FEBA.sha256)

    @post_process
    def process(self) -> None:
        """Build one record per (released locus, stored sample) and the preprocess reports."""
        verify_raw_files(self.raw_dir, {RAW_FILE: fb.FEBA.sha256})
        feba = fb.Feba(osp.join(self.raw_dir, RAW_FILE), verify=False)
        fb.check_studies(SOURCE_STUDIES, feba.publications())
        experiments = feba.experiments(ORG_ID)
        decisions = [decide(e) for e in experiments]
        stored = [
            e for e, d in zip(experiments, decisions, strict=True) if d.fate == "stored"
        ]
        values = feba.values(ORG_ID, [e.exp_name for e in stored])
        genes = fb.released_genes(values.loci, feba.genes(ORG_ID))
        feba.close()
        counts = loader_inventory(decisions, len(genes))

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for experiment, reference, publication in tqdm(
                iter_records(
                    self.name,
                    stored,
                    values,
                    genes,
                    assembly_reference(REFERENCE_STRAIN),
                ),
                total=counts.stored_records,
                desc=self.name,
            ):
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, publication, itxn),
                )
                idx += 1
        env.close()
        interned_env.close()
        if idx != counts.stored_records:
            raise RuntimeError(
                f"wrote {idx} records, inventory says {counts.stored_records}"
            )
        payloads: dict[str, Any] = {
            "dropped_records.json": counts.model_dump(mode="json"),
            "samples.json": [
                {**d.model_dump(mode="json"), "experiment": e.model_dump(mode="json")}
                for e, d in zip(experiments, decisions, strict=True)
            ],
            "genes.json": [g.model_dump(mode="json") for g in genes],
        }
        for name, payload in payloads.items():
            with open(osp.join(self.preprocess_dir, name), "w") as handle:
                json.dump(payload, handle, indent=2)
        log.info("Wrote %d Fitness Browser Keio records", idx)

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError(
            "FitnessBrowserEcoliDataset builds records in process()"
        )


# --------------------------------------------------------------------------- #
# Verification (L0 to L4)
# --------------------------------------------------------------------------- #
#: The served stores the L4 rows read, read-only.
PRICE_STORE_REL: Final = wetmore.SUPERSET_DATASET_ROOT_REL
WETMORE_STORE_REL: Final = wetmore.DATASET_ROOT_REL
#: Two copies of one release must agree exactly up to float formatting.
IDENTITY_MIN_PEARSON_R: Final = 0.999999


def deposit_fitness(feba: fb.Feba, names: Sequence[str]) -> dict[str, dict[str, float]]:
    """The deposit's fitness of ``names``, keyed by the released ``Gene.sysName``.

    MEASUREMENT ONLY: these are samples this loader does not store (G4). The Price 2018
    and Wetmore 2015 stores key a gene by the b-number their release gives (#1057), as
    this deposit does, so the values align on the released tag with no crosswalk.
    Nothing here decides what this loader stores.
    """
    values = feba.values(ORG_ID, names)
    genes = fb.released_genes(values.loci, feba.genes(ORG_ID))
    out: dict[str, dict[str, float]] = {}
    for j, name in enumerate(values.experiments):
        out[name] = {g.tag: float(values.fitness[i, j]) for i, g in enumerate(genes)}
    return out


def stored_publications_match(dataset_root: str) -> LevelResult:
    """SUPPLEMENTARY L1: each stored sample's records name its browser publication."""
    from torchcell.verification.runners import stream_records

    seen: dict[str, set[str]] = {}
    for record in stream_records(dataset_root):
        sample = str(record["experiment"]["phenotype"]["screen_id"])
        seen.setdefault(sample, set()).add(str(record["publication"].get("doi")))
    with open(osp.join(dataset_root, "preprocess", "samples.json")) as handle:
        rows = json.load(handle)
    expected = {
        f"{ORG_ID}:{r['exp_name']}": str(SOURCE_STUDIES[r["pub_id"]].publication.doi)
        for r in rows
        if r["fate"] == "stored"
    }
    wrong = sorted(s for s, dois in seen.items() if dois != {expected.get(s)})
    passed = not wrong and set(seen) == set(expected)
    return LevelResult(
        level=Level.L1,
        name="records_carry_their_browser_publication",
        passed=passed,
        message=(
            f"SUPPLEMENTARY: {len(seen)} stored samples, {len(expected)} expected; "
            f"{len(wrong)} carry a publication other than their pubId's"
        ),
        details={
            "wrong": wrong[:20],
            "by_sample": {k: sorted(v) for k, v in seen.items()},
        },
    )


def verify_build(
    dataset_root: str,
    *,
    data_root: str | None = None,
    expected_count: int = EXPECTED_RECORDS,
) -> VerificationReport:
    """The environment-response L0-L4 gate on a built tree, plus the L4 compendium rows.

    The gene universe of the L1 and L4 gene rules is the deposit's own Keio
    ``Gene.sysName`` set, with no resolver: the stored tags are the deposit's, and no
    assembly decides anything about them. The L4 rows read the deposit's values of every Keio sample the served Price 2018 and
    Wetmore 2015 stores hold, against those stores read-only: the samples this loader
    does NOT store because their publication is loaded from its own release (G4). Price
    2018's release is the browser's, so its bar is identity (Pearson r 0.999999); Wetmore
    2015 is a different analysis (1.0.0) of the same reads, so its bar is Price 2018's
    median exact-replicate r 0.87 (``wetmore2015.COMPENDIUM_MIN_PEARSON_R``).
    """
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset_streaming,
    )
    from torchcell.verification.runners import stream_records

    root = fb.data_root(data_root)
    feba = fb.Feba(osp.join(dataset_root, "raw", RAW_FILE), verify=False)
    released = {g.sys_name for g in feba.genes(ORG_ID).values() if g.sys_name}
    report = verify_environment_response_dataset_streaming(
        stream_records(dataset_root),
        dataset_name=FitnessBrowserEcoliDataset.__name__,
        provenance=fb.FEBA_PROVENANCE,
        expected_count=expected_count,
        sgd_genes=released,
        gene_universe_label="the deposit's Keio Gene.sysName",
        min_containment=1.0,
    )
    report.add(stored_publications_match(dataset_root))
    experiments = {e.exp_name for e in feba.experiments(ORG_ID)}
    checks: dict[str, fb.CompendiumCheck] = {}
    for rel, bar, name in (
        (PRICE_STORE_REL, IDENTITY_MIN_PEARSON_R, "compendium_cross_check_price2018"),
        (
            WETMORE_STORE_REL,
            wetmore.COMPENDIUM_MIN_PEARSON_R,
            "own_release_cross_check_wetmore2015",
        ),
    ):
        store = osp.join(root, rel)
        if not osp.isdir(osp.join(store, "processed", "lmdb")):
            raise FileNotFoundError(f"{store}: the store the L4 row reads is absent")
        served = fb.stored_fitness(stream_records(store), ORG_ID)
        shared = [n for n in served if n in experiments]
        check = fb.compendium_check(
            deposit_fitness(feba, shared), served, {}, against=rel, min_pearson_r=bar
        )
        checks[name] = check
        report.add(fb.compendium_check_result(check, name))
    feba.close()
    preprocess = osp.join(dataset_root, "preprocess")
    os.makedirs(preprocess, exist_ok=True)
    with open(osp.join(preprocess, "verification_report.json"), "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    with open(osp.join(preprocess, "compendium_cross_check.json"), "w") as handle:
        json.dump(
            {k: v.model_dump(mode="json") for k, v in checks.items()}, handle, indent=2
        )
    return report


def main(argv: list[str] | None = None) -> None:
    """``verify`` the built dev store, or print the ``inventory`` off the deposit."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("verify", "inventory"))
    args = parser.parse_args(argv)
    root = fb.data_root()
    if args.command == "verify":
        print(verify_build(osp.join(root, DATASET_ROOT_REL), data_root=root).summary())
        return
    feba = fb.Feba(fb.feba_path(root))
    decisions = [decide(e) for e in feba.experiments(ORG_ID)]
    feba.close()
    print(
        json.dumps(
            {
                "fates": fb.fate_counts(decisions),
                "by_publication": fb.fate_counts_by_publication(decisions),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
