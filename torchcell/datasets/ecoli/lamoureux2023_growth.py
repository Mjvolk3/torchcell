# torchcell/datasets/ecoli/lamoureux2023_growth
# [[torchcell.datasets.ecoli.lamoureux2023_growth]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/lamoureux2023_growth
# Test file: tests/torchcell/datasets/ecoli/test_lamoureux2023_growth.py
r"""Lamoureux 2023 PRECISE-1K: the released per-sample growth rate, as an absolute readout.

Lamoureux et al. 2023 (Nucleic Acids Res 51:10184, doi:10.1093/nar/gkad750) releases its
sample metadata with a ``Growth Rate (1/hr)`` column. 354 of ``metadata_qc.csv``'s 1,035
rows carry a value; this module serves the 89 of them the schema can hold as
:class:`GrowthRateLamoureux2023Dataset`, one ``BacterialEnvironmentResponseExperiment``
per released sample carrying the ABSOLUTE rate, against a declared base condition.

Rank 11 of the E. coli SI audit, blocked "in its absolute form by gap 1". Gap 1's second
half is closed: PR #836 (#776) gave the environment-response verifier an absolute branch
(``reference_centered=False``, admitted only for a ``measurement_type`` in
``ABSOLUTE_MEASUREMENT_TYPES``, which is ``{growth_rate, colony_size}``), and
``MeasurementType.growth_rate`` is documented "absolute or normalized growth rate /
doubling time", so an absolute rate in 1/hr is exactly that member.

THE COUNTS, each measured rather than recalled. Of the 354 rate cells, every one is a
``p1k_*`` id of the PRECISE-1K index, so the sibling Public K-12 arm contributes NOTHING:
0 of its 240 built records carry a rate. Applying the expression loader's OWN genotype
and environment rules (``settle_genotype`` / ``settle_environment``, imported, never
restated) to the 354 keeps 103 and drops 251 -- 138 evolved isolates, 94 heterologous
constructs, 12 non-MG1655 strains and 7 point-mutation alleles. Measured: those 103 are
exactly the built ``rnaseq_lamoureux2023`` records that carry a rate.

THE 14 ZERO-RATE ROWS ARE DROPPED, AND THAT IS A DECISION. 14 of the 103 release the
string ``0.0``. A stored 0 h^-1 asserts a fully arrested culture from which a sequencing
library was nonetheless prepared, and the release gives no legend that says so: a 0 in a
released numeric column is indistinguishable from an unrecorded cell, and whether the
authors meant 0 or blank is NOT stated anywhere in the paper or the release. The
conservative reading is that the cell is uninterpretable, so the 14 rows are dropped
under ``DROP_ZERO_RATE`` with every id, condition and value written verbatim to
``preprocess/dropped_records.json``, which makes the decision reversible. All 14 sit in 7
stress conditions (``fur:wt_dpd``, ``fur:delfur_dpd``, ``ompr:wt_nacl``,
``oxidative:wt_pq``, ``oxidative:deloxyr_pq``, ``oxidative:delsoxr_pq``,
``oxidative:delsoxs_pq``), and the cost is recorded too: the ``oxyR``, ``soxR`` and
``soxS`` deletions appear ONLY in those rows, so they are absent from the stored gene
set. The remaining 89 rates run 0.07 to 1.42 h^-1 and none is negative.

THE REFERENCE IS A DECLARED BASE CONDITION, NEVER A BORROWED ROW. The absolute branch
needs the reference to state its own finite value on the record's own scale. The
expression loader's reference, the ``control:wt_glc`` pair, CANNOT carry it: measured,
both ``p1k_00001`` and ``p1k_00002`` release an empty rate cell. So the reference is the
declared base condition -- wild-type MG1655 in M9 with ``glucose(2)`` at 37 C and pH 7.0,
no supplement -- and its value is the mean of the rates of EVERY stored record whose
genotype is wild type and whose environment serializes identically to that base
environment. Measured: exactly 8 records do (``ica:wt_glc`` six and ``ytf:wt_glc`` two),
releasing 0.58, 0.58, 0.63, 0.63, 0.66, 0.66, 0.68 and 0.69, mean 0.63875 h^-1. The ninth
``*:wt_glc`` row with a rate, ``ssw__wt_glc__1`` at 0.73, is NOT in the aggregate: it was
grown on ``glucose(4)``, a different environment, which the byte comparison catches
without a hand-written exclusion. An aggregate rather than one row because the eight span
0.58 to 0.69, 19%, which is the Caglar 2017 situation verbatim ("the three released base
measurements span 0.2174 log2 ... so borrowing one is not neutral").

``screen_id`` IS THE RELEASED PROJECT, AND IT IS LOAD-BEARING. Measured over the 89: the
L1 key (genotype, environment, project, ``rep_id``) is unique for all 89, and dropping the
project leaves 2 duplicate triples, because ``ica:wt_glc`` and ``ytf:wt_glc`` are the same
declared environment and the same wild-type genotype measured in two different projects,
which repeat ``rep_id`` 1 and 2. This is Caglar 2017's use of ``screen_id``: the source's
own run label, not a synthesized encoding of a factor the schema cannot hold.

WHAT IS A TYPED GAP, AND WHY EACH ONE IS. The release ties the rate to a library row and
never states the design behind it, so nothing about that design is asserted:

* ``n_samples`` and ``sample_unit``. ``Biological Replicates`` counts the condition's
  RNA-SEQ libraries, not measurements of the rate, and nothing in the release ties them.
  Measured evidence that a row is not a culture: within ``ica:wt_glc`` the six libraries
  carry 0.58, 0.58, 0.66, 0.66, 0.63 and 0.63, three values on three pairs, so the rate
  looks like a per-culture number repeated across a culture's libraries.
* ``environment_response_uncertainty``, its type, and ``environment_response_se``. No
  released column states a dispersion for the rate: measured, the only metadata column
  whose name matches ``sd|std|err|ci|sem`` is ``Sequencing Machine``.
* ``assay_type``. The paper mentions a growth rate once, in a sentence that does not
  define the column, and never names the instrument, the OD range or the fitting method.
  Unlike Schmidt 2016, where the assay IS stated and ``AssayType.other`` carries it, here
  the assay is not reported, so the field is a typed absence rather than ``other``.

A SEPARATE MODULE, measured. ``build_manifest`` keys a built store's staleness on the
schema closure of the loader MODULE's own ``torchcell.datamodels`` imports
(``provenance/schema_deps.py:loader_closure``). ``lamoureux2023.py``'s closure is 61
symbols and equals the 61 the served ``rnaseq_lamoureux2023`` store records; adding the
environment-response symbols raises it to 72, which would mark that store (and nothing
else) STALE for a change touching none of its 241 records. The pinned artifacts, the
metadata reader, the genotype and environment rules, the media library and the sourced
values are imported FROM ``lamoureux2023``, so there is one copy of each.

DATA. The one consumed file is ``metadata_qc.csv``, one of the four files
``lamoureux2023.py`` already mirrors from the Zenodo archive under the same sha256 pins.
Nothing is retrieved here that that module does not already record.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
import statistics
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.schema import (
    BacterialDeletionPerturbation,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    Environment,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    Publication,
)
from torchcell.datasets.bacteria_common import (
    assembly_reference,
    bacterial_genome,
    reconcile_locus_tags,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.ecoli import lamoureux2023 as lm
from torchcell.sequence.genome.ecoli.k12 import EcoliK12Genome, EcoliK12StrainName
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

log = logging.getLogger(__name__)

DATASET_ROOT_REL = "data/torchcell/growth_rate_lamoureux2023"

#: The released column, by its header verbatim. It is also the stored ``units``.
COL_GROWTH_RATE = "Growth Rate (1/hr)"
UNITS = COL_GROWTH_RATE

#: Build oracles, every one measured on the pinned metadata before being written here.
EXPECTED_RATE_CELLS = 354
EXPECTED_AFTER_LOADER_RULES = 103
EXPECTED_ZERO_RATES = 14
EXPECTED_RECORDS = 89
EXPECTED_UNPERTURBED = 0
EXPECTED_BASE_RECORDS = 8
#: Measured over the 89 stored rates.
MIN_RATE = 0.07
MAX_RATE = 1.42
#: Measured mean of the 8 base-condition rates, recomputed by the build and compared.
BASE_RATE_MEAN = 0.63875
BASE_RATE_ATOL = 1e-9

#: The base condition, as the metadata columns that define it. The reference is the
#: aggregate over the records whose ENVIRONMENT serializes to this one and whose genotype
#: is wild type, so the declaration is a selection of released cells, never a value.
BASE_CONDITION: dict[str, str] = {
    lm.COL_STRAIN: "MG1655",
    lm.COL_MEDIA: "M9",
    lm.COL_CARBON: "glucose(2)",
    lm.COL_TEMPERATURE: "37",
    lm.COL_PH: "7",
    lm.COL_SUPPLEMENT: "",
    lm.COL_ACCEPTOR: "O2",
    lm.COL_NITROGEN: "NH4Cl(1)",
    lm.COL_TRACE: "sauer trace element mixture",
    lm.COL_ANTIBIOTIC: "",
    lm.COL_CULTURE: "Batch",
    lm.COL_EVOLVED: "No",
}
#: A human label for the base condition in the ledgers; not a stored value.
BASE_CONDITION_LABEL = "wild-type MG1655, M9 + glucose(2) g/L, 37 C, pH 7.0"

GROWTH_RATE_COLUMN = SourcedValue(
    value=COL_GROWTH_RATE,
    quote=COL_GROWTH_RATE,
    provenance=lm.metadata_provenance(COL_GROWTH_RATE),
    note="the released header is the ONLY statement of the readout and its unit: the "
    "paper mentions a growth rate once, in 'growth rate and oxidative stress during "
    "naphthoquinone-based aerobic respiration', which defines nothing, and never names "
    "the instrument, the OD range or the fitting method",
)

SOURCED_VALUES: dict[str, SourcedValue] = {
    "growth_rate_column": GROWTH_RATE_COLUMN,
    "strain": lm.STRAIN,
    "control_medium": lm.CONTROL_MEDIUM,
    "harvest": lm.HARVEST,
    "condition_space": lm.CONDITION_SPACE,
    "data_availability": lm.DATA_AVAILABILITY,
}

ASSAY_GAP = ProvenanceGap(
    field="assay_type",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=lm.metadata_provenance(COL_GROWTH_RATE),
    note="the release states neither the instrument nor the fit behind the rate, and the "
    "paper's single mention of a growth rate defines nothing, so no AssayType member is "
    "sourced; liquid_od_growth would be a reading rather than a released fact",
)
UNCERTAINTY_GAP = ProvenanceGap(
    field="environment_response_uncertainty",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=lm.metadata_provenance(COL_GROWTH_RATE),
    note="no released column states a dispersion of the rate: measured, the only "
    "metadata column whose name matches sd|std|err|ci|sem is 'Sequencing Machine'",
)
SE_GAP = ProvenanceGap(
    field="environment_response_se",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=lm.metadata_provenance(COL_GROWTH_RATE),
    note="no dispersion is released, so none can be reduced to a standard error",
)
N_SAMPLES_GAP = ProvenanceGap(
    field="n_samples",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=lm.metadata_provenance(lm.COL_REPLICATES),
    note="'Biological Replicates' counts the condition's RNA-seq libraries, not "
    "measurements of the rate, and nothing in the release ties them. Measured evidence "
    "that a released row is not a culture: within ica:wt_glc the six libraries carry "
    "0.58, 0.58, 0.66, 0.66, 0.63 and 0.63, three values on three pairs",
)
SAMPLE_UNIT_GAP = ProvenanceGap(
    field="sample_unit",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=lm.metadata_provenance(lm.COL_REPLICATES),
    note="the released row is a sequencing library; the release never says what one "
    "measurement of the rate is",
)
RECORD_GAPS: tuple[ProvenanceGap, ...] = (
    ASSAY_GAP,
    UNCERTAINTY_GAP,
    SE_GAP,
    N_SAMPLES_GAP,
    SAMPLE_UNIT_GAP,
)
REFERENCE_SCREEN_GAP = ProvenanceGap(
    field="screen_id",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=lm.metadata_provenance(lm.COL_PROJECT),
    note="the reference is the mean over the base condition's released rates, which "
    "span two projects (ica and ytf), so no single released screen label names it",
)
REFERENCE_REPLICATE_GAP = ProvenanceGap(
    field="replicate_id",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=lm.metadata_provenance(lm.COL_REP),
    note="the reference aggregates several released replicates, so no one rep_id "
    "names it",
)
REFERENCE_GAPS: tuple[ProvenanceGap, ...] = (
    *RECORD_GAPS,
    REFERENCE_SCREEN_GAP,
    REFERENCE_REPLICATE_GAP,
)


# --------------------------------------------------------------------------- #
# Drop rules
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One reason a released rate cell is not a record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule: str
    description: str
    stage: str
    n_records: int
    items: list[str] = Field(default_factory=list)


DROP_ZERO_RATE = "zero_rate_not_distinguishable_from_unrecorded"
ZERO_RATE_DESCRIPTION = (
    "the released cell is exactly 0.0. A stored 0 h^-1 asserts a fully arrested culture "
    "from which a sequencing library was nonetheless prepared, and the release carries "
    "no legend that says so, so the cell is indistinguishable from an unrecorded one. "
    "Whether the authors meant 0 or blank is stated nowhere in the paper or the "
    "release, so the conservative reading is that the cell is uninterpretable. Every "
    "dropped id, condition and value is listed here, so the decision is reversible"
)


class RateRow(BaseModel):
    """One released sample that carries a growth rate, with its released context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample: str = Field(description="the release's own ``p1k_*`` index value")
    sample_name: str = Field(description="``sample_id``, the human label")
    project: str
    full_name: str
    rep_id: str
    rate: float
    deleted_symbols: tuple[str, ...]


def read_rate_rows(path: str) -> tuple[list[RateRow], list[DropRule]]:
    """Every metadata row with a rate, filtered by the expression loader's own rules.

    The genotype and environment rules are ``lamoureux2023``'s, imported rather than
    restated, so this dataset keeps exactly the samples the expression dataset keeps.
    The rate cell is then required to be a positive number; the zero cells go under
    :data:`DROP_ZERO_RATE`.
    """
    metadata = lm.read_metadata(path)
    samples = sorted(str(sample) for sample in metadata.index)
    rows = {sample: lm.metadata_row(metadata, sample) for sample in samples}
    with_rate = [
        sample for sample in samples if str(rows[sample][COL_GROWTH_RATE]).strip()
    ]
    if len(with_rate) != EXPECTED_RATE_CELLS:
        raise RuntimeError(
            f"{len(with_rate)} metadata rows carry a {COL_GROWTH_RATE!r}, the module "
            f"declares {EXPECTED_RATE_CELLS}"
        )
    dropped: dict[tuple[str, str], list[str]] = {}
    reasons: dict[str, str] = {}
    kept: list[RateRow] = []
    for sample in with_rate:
        row = rows[sample]
        genotype = lm.settle_genotype(row)
        if not genotype.kept:
            dropped.setdefault(("genotype", genotype.rule.value), []).append(sample)
            reasons[genotype.rule.value] = lm.GENOTYPE_RULE_REASONS[genotype.rule]
            continue
        environment = lm.settle_environment(row)
        if environment.rule is not None:
            dropped.setdefault(("environment", environment.rule.value), []).append(
                sample
            )
            reasons[environment.rule.value] = lm.ENVIRONMENT_RULE_REASONS[
                environment.rule
            ]
            continue
        rate = float(str(row[COL_GROWTH_RATE]).strip())
        if rate <= 0.0:
            dropped.setdefault(("readout", DROP_ZERO_RATE), []).append(
                f"{sample} ({row[lm.COL_FULL_NAME]}, {rate})"
            )
            reasons[DROP_ZERO_RATE] = ZERO_RATE_DESCRIPTION
            continue
        kept.append(
            RateRow(
                sample=sample,
                sample_name=str(row[lm.COL_SAMPLE]),
                project=str(row[lm.COL_PROJECT]),
                full_name=str(row[lm.COL_FULL_NAME]),
                rep_id=str(row[lm.COL_REP]),
                rate=rate,
                deleted_symbols=tuple(sorted(genotype.deleted_symbols)),
            )
        )
    rules = [
        DropRule(
            rule=rule,
            description=reasons[rule],
            stage=stage,
            n_records=len(items),
            items=sorted(items),
        )
        for (stage, rule), items in sorted(dropped.items())
    ]
    survivors = len(with_rate) - sum(rule.n_records for rule in rules)
    if survivors != len(kept):
        raise RuntimeError("the drop rules do not account for every released rate cell")
    return kept, rules


def environments_of(path: str, rows: Sequence[RateRow]) -> dict[str, Environment]:
    """``{sample: environment}`` through ``lamoureux2023``'s own environment builder."""
    metadata = lm.read_metadata(path)
    out: dict[str, Environment] = {}
    cache: dict[Any, Environment] = {}
    for row in rows:
        verdict = lm.settle_environment(lm.metadata_row(metadata, row.sample))
        if verdict.spec is None:
            raise RuntimeError(f"kept sample {row.sample} has no parsed condition")
        if verdict.spec not in cache:
            cache[verdict.spec] = lm.build_environment(verdict.spec)
        out[row.sample] = cache[verdict.spec]
    return out


def base_condition_environment(path: str) -> Environment:
    """The declared base condition's environment, built from ITS released row.

    The base condition is declared as a set of metadata cells (:data:`BASE_CONDITION`),
    and the rows matching all of them must build ONE environment, which is the reference's
    environment. Declaring the selection rather than the value is what keeps the
    reference a released measurement: the aggregate below is taken over whichever stored
    records land on this environment, so a re-released medium cannot silently change
    which rows the reference is computed from without changing the environment too.
    """
    metadata = lm.read_metadata(path)
    matches = [
        str(sample)
        for sample in metadata.index
        if all(
            str(lm.metadata_row(metadata, str(sample))[column]).strip() == value
            for column, value in BASE_CONDITION.items()
        )
    ]
    if not matches:
        raise RuntimeError(
            f"no released row matches the declared base condition {BASE_CONDITION}"
        )
    built = {
        lm.build_environment(
            _spec_or_raise(lm.settle_environment(lm.metadata_row(metadata, sample)))
        ).model_dump_json()
        for sample in matches
    }
    if len(built) != 1:
        raise RuntimeError(
            f"the {len(matches)} rows of the declared base condition build "
            f"{len(built)} distinct environments, not one"
        )
    return lm.build_environment(
        _spec_or_raise(lm.settle_environment(lm.metadata_row(metadata, matches[0])))
    )


def _spec_or_raise(verdict: lm.EnvironmentVerdict) -> lm.ConditionSpec:
    """The verdict's parsed condition, or a refusal naming the rule that blocked it."""
    if verdict.spec is None:
        raise RuntimeError(
            f"the base condition's row was dropped by the rule {verdict.rule!r}"
        )
    return verdict.spec


class BaseAggregate(BaseModel):
    """The reference's value and every released rate it was computed from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    samples: list[str]
    rates: list[float]
    mean: float
    spread: float = Field(description="max minus min over the contributing rates")


def base_aggregate(
    rows: Sequence[RateRow], environments: Mapping[str, Environment], base: Environment
) -> BaseAggregate:
    """Mean of the stored wild-type rates whose environment IS the base environment.

    Byte comparison of the serialized environment, not a condition label, so a row grown
    on another carbon concentration drops out without a hand-written exclusion. Measured
    on the pinned metadata: 8 records qualify (``ica:wt_glc`` six, ``ytf:wt_glc`` two) and
    ``ssw__wt_glc__1``, which was grown on ``glucose(4)``, does not.
    """
    key = base.model_dump_json()
    members = [
        row
        for row in rows
        if not row.deleted_symbols and environments[row.sample].model_dump_json() == key
    ]
    if len(members) != EXPECTED_BASE_RECORDS:
        raise RuntimeError(
            f"{len(members)} stored wild-type records carry the base environment, the "
            f"module declares {EXPECTED_BASE_RECORDS}"
        )
    rates = [row.rate for row in members]
    mean = statistics.fmean(rates)
    if abs(mean - BASE_RATE_MEAN) > BASE_RATE_ATOL:
        raise RuntimeError(
            f"the base condition's mean rate is {mean}, the module declares "
            f"{BASE_RATE_MEAN}"
        )
    return BaseAggregate(
        label=BASE_CONDITION_LABEL,
        samples=[row.sample for row in members],
        rates=rates,
        mean=mean,
        spread=max(rates) - min(rates),
    )


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def phenotype(row: RateRow) -> EnvironmentResponsePhenotype:
    """One released sample's absolute growth rate, with its five typed gaps."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.growth_rate,
        environment_response=row.rate,
        units=UNITS,
        screen_id=row.project,
        replicate_id=row.rep_id,
        provenance_gaps=list(RECORD_GAPS),
    )


def reference_phenotype(aggregate: BaseAggregate) -> EnvironmentResponsePhenotype:
    """The base condition's mean released rate, on the records' own scale."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.growth_rate,
        environment_response=aggregate.mean,
        units=UNITS,
        provenance_gaps=list(REFERENCE_GAPS),
    )


def genotype_of(row: RateRow, tags: Mapping[str, tuple[str, str]]) -> Genotype:
    """The released deletions of one sample, on their MG1655 b-numbers."""
    return Genotype(
        perturbations=[
            BacterialDeletionPerturbation(
                systematic_gene_name=tags[symbol][0],
                perturbed_gene_name=tags[symbol][1],
                gene_namespace=lm.GENE_NAMESPACE,
            )
            for symbol in row.deleted_symbols
        ]
    )


def deletion_tags(
    genome: EcoliK12Genome, rows: Sequence[RateRow], *, label: str
) -> tuple[dict[str, tuple[str, str]], Any]:
    """Each deleted symbol -> (b-number, the genome's own symbol), with the report.

    Every symbol must resolve to one MG1655 locus and the genome's symbol must resolve
    back to it, which is the rule ``lamoureux2023`` applies to the same symbols.
    """
    symbols = sorted({symbol for row in rows for symbol in row.deleted_symbols})
    names = pd.Series(symbols, dtype=object)
    stored, report = reconcile_locus_tags(genome, names, label=label)
    report.require_resolved(1.0)
    if report.outside_namespace:
        raise RuntimeError(
            f"deleted-gene symbols not stored as b-numbers: {report.outside_namespace}"
        )
    tags: dict[str, tuple[str, str]] = {}
    for symbol, tag in zip(symbols, stored.tolist(), strict=True):
        canonical = genome.genbank.loci[str(tag)].symbol or symbol
        if genome.resolve_gene_name(canonical).systematic_name != str(tag):
            raise RuntimeError(
                f"{canonical!r}, the genome's symbol of {tag}, does not resolve back"
            )
        tags[symbol] = (str(tag), canonical)
    return tags, report


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class GrowthRateLamoureux2023Dataset(ExperimentDataset):
    """PRECISE-1K's released per-sample growth rate: 89 absolute rates in 1/hr."""

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = lm.REFERENCE_STRAIN_NAME

    def __init__(
        self,
        root: str = DATASET_ROOT_REL,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        ecoli_genome: EcoliK12Genome | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; ``ecoli_genome`` is injected by the build entry points."""
        self.ecoli_genome = ecoli_genome
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
        """The one consumed file, the release's sample metadata."""
        return [lm.METADATA.name]

    def download(self) -> None:
        """Link the mirrored metadata table after checking the manifest and sha256."""
        data_root = os.environ["DATA_ROOT"]
        manifest = lm.load_manifest(data_root)
        check_manifest_pin(
            lm.METADATA.relpath,
            lm.manifest_sha256(manifest, lm.METADATA.relpath),
            lm.METADATA.sha256,
        )
        src = lm.raw_mirror_dir(data_root) / lm.METADATA.relpath
        if not src.exists():
            raise RuntimeError(f"required raw artifact missing from mirror: {src}")
        os.makedirs(self.raw_dir, exist_ok=True)
        link_verified(src, osp.join(self.raw_dir, lm.METADATA.name), lm.METADATA.sha256)
        log.info(
            "Lamoureux 2023 metadata linked into %s (sha256 verified)", self.raw_dir
        )

    def _genome(self) -> EcoliK12Genome:
        """The injected MG1655 genome, or the default cache opened read-only."""
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        return self.ecoli_genome

    @post_process
    def process(self) -> None:
        """Write one record per stored released rate, 89 in all."""
        verify_raw_files(self.raw_dir, {lm.METADATA.name: lm.METADATA.sha256})
        path = osp.join(self.raw_dir, lm.METADATA.name)
        rows, rules = read_rate_rows(path)
        by_rule = {rule.rule: rule.n_records for rule in rules}
        if by_rule.get(DROP_ZERO_RATE) != EXPECTED_ZERO_RATES:
            raise RuntimeError(
                f"{by_rule.get(DROP_ZERO_RATE)} rows release a zero rate, the module "
                f"declares {EXPECTED_ZERO_RATES}"
            )
        if len(rows) + by_rule[DROP_ZERO_RATE] != EXPECTED_AFTER_LOADER_RULES:
            raise RuntimeError(
                f"{len(rows) + by_rule[DROP_ZERO_RATE]} rate cells survive the "
                f"expression loader's rules, the module declares "
                f"{EXPECTED_AFTER_LOADER_RULES}"
            )
        if len(rows) != EXPECTED_RECORDS:
            raise RuntimeError(
                f"{len(rows)} records, the module declares {EXPECTED_RECORDS}"
            )
        if min(row.rate for row in rows) != MIN_RATE or (
            max(row.rate for row in rows) != MAX_RATE
        ):
            raise RuntimeError(
                f"the stored rates span {min(row.rate for row in rows)} to "
                f"{max(row.rate for row in rows)}; the module declares {MIN_RATE} to "
                f"{MAX_RATE}"
            )
        environments = environments_of(path, rows)
        unperturbed = [
            row.sample for row in rows if not environments[row.sample].perturbations
        ]
        if len(unperturbed) != EXPECTED_UNPERTURBED:
            raise RuntimeError(
                f"{len(unperturbed)} records carry no environmental edit, the module "
                f"declares {EXPECTED_UNPERTURBED}"
            )
        base = base_condition_environment(path)
        aggregate = base_aggregate(rows, environments, base)

        genome = self._genome()
        tags, report = deletion_tags(genome, rows, label=f"{self.name} deleted genes")

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        reference = BacterialEnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=assembly_reference(self.REFERENCE_STRAIN),
            environment_reference=base,
            phenotype_reference=reference_phenotype(aggregate),
        )
        publication = Publication(
            doi=lm.PAPER_DOI, doi_url=f"https://doi.org/{lm.PAPER_DOI}"
        )
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, row in enumerate(tqdm(rows, desc="lamoureux2023 growth rate")):
                experiment = BacterialEnvironmentResponseExperiment(
                    dataset_name=self.name,
                    genotype=genotype_of(row, tags),
                    environment=environments[row.sample],
                    phenotype=phenotype(row),
                )
                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, publication, itxn),
                )
        env.close()
        interned_env.close()

        self._write_ledgers(rows, rules, aggregate, report, environments)
        log.info(
            "Lamoureux 2023 growth rate: %d records over %d conditions and %d projects; "
            "rate %.4f to %.4f 1/hr; reference %s = %.5f 1/hr over %d released rows",
            len(rows),
            len({row.full_name for row in rows}),
            len({row.project for row in rows}),
            min(row.rate for row in rows),
            max(row.rate for row in rows),
            aggregate.label,
            aggregate.mean,
            len(aggregate.samples),
        )

    def _write_ledgers(
        self,
        rows: Sequence[RateRow],
        rules: Sequence[DropRule],
        aggregate: BaseAggregate,
        report: Any,
        environments: Mapping[str, Environment],
    ) -> None:
        """The drop log, the base aggregate, the sourcing table and the rate table."""
        out = Path(self.preprocess_dir)
        (out / "dropped_records.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    "source_rows": EXPECTED_RATE_CELLS,
                    "kept_records": len(rows),
                    "dropped_records": sum(rule.n_records for rule in rules),
                    "rules": [rule.model_dump(mode="json") for rule in rules],
                    "notes": [
                        "the genotype and environment rules are lamoureux2023's own, "
                        "imported rather than restated, so this dataset keeps exactly "
                        "the samples the expression dataset keeps; measured, the 103 "
                        "cells they admit are exactly the built rnaseq_lamoureux2023 "
                        "records that carry a rate",
                        "every one of the 354 released rate cells is a p1k_* id, so the "
                        "Public K-12 arm contributes nothing: 0 of its 240 built "
                        "records carry a rate",
                        "dropping the 14 zero-rate rows also drops the oxyR, soxR and "
                        "soxS deletions entirely, because they appear only in those "
                        "rows; that is the cost of the decision and it is recorded here",
                    ],
                },
                indent=2,
            )
        )
        (out / "base_condition.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    "declared_cells": BASE_CONDITION,
                    "aggregate": aggregate.model_dump(mode="json"),
                    "why_an_aggregate": "the eight released base rates span "
                    f"{aggregate.spread:.2f} 1/hr, so borrowing one would not be "
                    "neutral; this is the Caglar 2017 situation, where the three "
                    "released base measurements span 0.2174 log2",
                    "why_not_the_expression_loaders_reference": "the control:wt_glc "
                    "pair p1k_00001 and p1k_00002, which is the expression dataset's "
                    "reference, releases an EMPTY rate cell, measured",
                    "excluded_by_the_environment_comparison": "ssw__wt_glc__1 releases "
                    "a rate of 0.73 and is a *:wt_glc row, but it was grown on "
                    "glucose(4) rather than glucose(2), so its environment differs and "
                    "the byte comparison drops it without a hand-written exclusion",
                },
                indent=2,
            )
        )
        (out / "locus_tag_reconciliation.json").write_text(
            json.dumps({"deleted_genes": report.model_dump(mode="json")}, indent=2)
        )
        (out / "sourced_values.json").write_text(
            json.dumps(
                {
                    name: value.model_dump(mode="json")
                    for name, value in SOURCED_VALUES.items()
                },
                indent=2,
            )
        )
        (out / "record_samples.json").write_text(
            json.dumps([row.sample for row in rows], indent=2)
        )
        pd.DataFrame(
            [
                {
                    "record": index,
                    "sample": row.sample,
                    "sample_id": row.sample_name,
                    "project": row.project,
                    "full_name": row.full_name,
                    "rep_id": row.rep_id,
                    "growth_rate_per_hour": row.rate,
                    "deleted_symbols": ";".join(row.deleted_symbols),
                    "is_base_condition": row.sample in set(aggregate.samples),
                    "n_environment_perturbations": len(
                        environments[row.sample].perturbations
                    ),
                }
                for index, row in enumerate(rows)
            ]
        ).to_csv(out / "growth_rates.csv", index=False)

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError(
            "GrowthRateLamoureux2023Dataset builds records in process()"
        )


# --------------------------------------------------------------------------- #
# L0-L4 verification
# --------------------------------------------------------------------------- #
VERIFIER_PROVENANCE = Provenance(
    source_uri=f"$DATA_ROOT/{lm.RAW_DIR_REL}/{lm.METADATA.relpath}",
    citation_key=lm.CITATION_KEY,
    sha256=lm.METADATA.sha256,
    method=(
        "the release's 'Growth Rate (1/hr)' metadata column: one "
        "BacterialEnvironmentResponseExperiment per released sample the expression "
        "loader's own genotype and environment rules keep and whose rate is positive, "
        "MeasurementType.growth_rate carrying the ABSOLUTE rate; the reference is the "
        "mean of the declared base condition's released rates"
    ),
    page=f"column {COL_GROWTH_RATE!r}",
    retrieved=lm.DATA_RETRIEVED_AT,
)


def l4_deleted_genes_are_mg1655_loci(
    records: Sequence[Mapping[str, Any]], genome: EcoliK12Genome
) -> LevelResult:
    """L4: every stored deletion names a gene row of the pinned MG1655 annotation."""
    deleted = {
        perturbation["systematic_gene_name"]
        for record in records
        for perturbation in record["experiment"]["genotype"]["perturbations"]
    }
    universe = set(genome.genbank.loci)
    missing = sorted(deleted - universe)
    return LevelResult(
        level=Level.L4,
        name="gene_containment_mg1655_b_numbers",
        passed=not missing,
        message=(
            f"{len(deleted) - len(missing)} of {len(deleted)} deleted loci are MG1655 "
            "GenBank gene rows"
        ),
        details={
            "n_deleted": len(deleted),
            "n_universe": len(universe),
            "missing": missing[:20],
        },
    )


def verify_build(
    dataset_root: str,
    data_root: str | None = None,
    *,
    genome: EcoliK12Genome | None = None,
    expected_count: int = EXPECTED_RECORDS,
) -> VerificationReport:
    """Run the environment-response L0-L4 gate over a built tree and write the report.

    ``reference_centered=False`` selects the absolute branch: the reference states its
    own finite rate on the records' own scale, and the branch refuses any record whose
    ``measurement_type`` is not in ``ABSOLUTE_MEASUREMENT_TYPES``.
    """
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset,
    )
    from torchcell.verification.runners import load_records

    records = load_records(dataset_root)
    if genome is None:
        genome = bacterial_genome(
            "ecoli", GrowthRateLamoureux2023Dataset.REFERENCE_STRAIN, data_root
        )
    report = verify_environment_response_dataset(
        records,
        dataset_name=osp.basename(osp.normpath(dataset_root)),
        provenance=VERIFIER_PROVENANCE,
        expected_count=expected_count,
        reference_centered=False,
        expected_unperturbed=EXPECTED_UNPERTURBED,
        resolve_gene_name=genome.resolve_gene_name,
    )
    report.add(l4_deleted_genes_are_mg1655_loci(records, genome))
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI: ``build`` the dev-tree LMDB, or ``verify`` an already built one."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        prog="python -m torchcell.datasets.ecoli.lamoureux2023_growth"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="build (or load) the dev-tree LMDB")
    sub.add_parser("verify", help="run L0-L4 on the built dev-tree LMDB")
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(data_root, DATASET_ROOT_REL)
    if args.command == "build":
        dataset = GrowthRateLamoureux2023Dataset(root=root)
        print(f"len = {len(dataset)}")
        print(Path(root, "preprocess", "base_condition.json").read_text())
        return 0
    report = verify_build(root, data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
