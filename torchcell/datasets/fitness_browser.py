# torchcell/datasets/fitness_browser
# [[torchcell.datasets.fitness_browser]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/fitness_browser
# Test file: tests/torchcell/datasets/test_fitness_browser.py
"""The Fitness Browser deposit (``feba.db``): the pinned reader both browser loaders share.

The Fitness Browser (https://fit.genomics.lbl.gov) is the Deutschbauer and Arkin labs'
release of every RB-TnSeq experiment they have run, one SQLite database with an
``Experiment`` row per sample (its medium, condition columns, temperature, library and the
``pubId`` of the paper that reported it), a ``Publication`` row per paper, and the gene
fitness and moderated t of every sample in ``FitByExp_<orgId>``. Price 2018's Keio
compendium release equals it to 8.9e-16 and Borchert 2024's Putida compendium copies it to
three decimals (the 2026-10-11 decomposition audit,
[[torchcell.candidates.aggregation-decomposition]]), so under the owner policy of
2026-10-10 ("a served compendium is a sanity check, not the source; load each original
study from its own release") the browser is the own release of every study it attributes
by ``pubId``, and the two compendia are the L4 cross-checks.

The deposit sits in the Price 2018 raw mirror,
``$DATA_ROOT/torchcell-raw/priceMutantPhenotypesThousands2018/data/fitness_browser/``,
retrieved once (10.2 GB) and pinned by :data:`FEBA` (sha256, bytes, URL, retrieval command,
``Last-Modified`` and ``ETag``, verbatim from ``DEPOSIT.md`` and ``headers.txt`` beside it).
Everything here reads it read-only and immutable; nothing writes to it.

What this module holds for the two loaders
(:mod:`torchcell.datasets.ecoli.fitness_browser_ecoli`,
:mod:`torchcell.datasets.pputida.fitness_browser_putida`):

* :class:`Feba`: the pinned, read-only reader (experiments, publications, genes, the
  fitness and t matrix of an organism);
* :class:`FbExperiment`: one ``Experiment`` row, typed, with empty strings as ``None``;
* :class:`SourceStudy`: a browser publication as a study, with ``source_kind``
  (``primary`` when the browser is the study's own release, ``compendium_only`` when the
  study is a compendium's own paper and the compendium release is its only other
  artifact) and the registered dataset, if any, that already loads its own release;
* :class:`SampleDecision`: what a loader does with one browser sample, and why;
* :func:`compendium_check`: the L4 agreement of two value sets over their shared (gene,
  sample) cells, pooled and per sample.
"""

from __future__ import annotations

import math
import os
import re
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from torchcell.data import file_sha256
from torchcell.datamodels.schema import Publication
from torchcell.verification.report import Level, LevelResult, Provenance

# --------------------------------------------------------------------------- #
# The deposit
# --------------------------------------------------------------------------- #
#: The raw-mirror key the deposit sits under (the Price 2018 decomposition audit put it
#: there; DEPOSIT.md records why).
DEPOSIT_CITATION_KEY: Final = "priceMutantPhenotypesThousands2018"
RAW_DIR_REL: Final = f"torchcell-raw/{DEPOSIT_CITATION_KEY}"
FEBA_RELPATH: Final = "data/fitness_browser/feba.db"


class DepositPin(BaseModel):
    """The deposited file's identity and retrieval, verbatim from ``DEPOSIT.md``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    relpath: str = Field(description="path under the raw mirror key")
    sha256: str
    n_bytes: int
    source_url: str
    retrieval_method: Literal["direct_url"]
    retrieval_command: str
    retrieved_at: str
    last_modified: str = Field(description="the response's Last-Modified header")
    etag: str = Field(description="the response's ETag header")


FEBA: Final = DepositPin(
    relpath=FEBA_RELPATH,
    sha256="250d4a03ffc41ea918027ed7adcb2fa9ed3ddc127ad3de29b40602fc6264b4ef",
    n_bytes=10_205_732_864,
    source_url="https://fit.genomics.lbl.gov/cgi_data/feba.db",
    retrieval_method="direct_url",
    retrieval_command="curl -L -o feba.db https://fit.genomics.lbl.gov/cgi_data/feba.db",
    retrieved_at="2026-10-11T04:29:05Z",
    last_modified="Fri, 09 Oct 2026 18:08:11 GMT",
    etag='"2604f2000-65d6c3b3c80ac"',
)

FEBA_PROVENANCE: Final = Provenance(
    source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{FEBA_RELPATH}",
    citation_key=DEPOSIT_CITATION_KEY,
    sha256=FEBA.sha256,
    method="the Fitness Browser SQLite database (fit.genomics.lbl.gov/cgi_data/feba.db), "
    "read-only: Experiment, Publication, Gene and FitByExp_<orgId> (fit, t)",
    page="tables Experiment, Publication, Gene, FitByExp_<orgId>",
    retrieved=FEBA.retrieved_at,
)


def data_root(root: str | None = None) -> str:
    """``root``, else ``DATA_ROOT`` from the repo ``.env``."""
    if root is not None:
        return root
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ["DATA_ROOT"]


def feba_path(root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/priceMutantPhenotypesThousands2018/data/fitness_browser/feba.db``."""
    return Path(data_root(root)) / RAW_DIR_REL / FEBA_RELPATH


class FebaPinError(RuntimeError):
    """The file at the deposit path is not the pinned deposit."""


def verify_feba(path: str | Path) -> None:
    """Refuse a file whose size or sha256 differs from :data:`FEBA`.

    The size is checked first (instant), then the sha256 (a full read of 10.2 GB).
    """
    size = Path(path).stat().st_size
    if size != FEBA.n_bytes:
        raise FebaPinError(f"{path}: {size} bytes, pinned {FEBA.n_bytes}")
    digest = file_sha256(path)
    if digest != FEBA.sha256:
        raise FebaPinError(f"{path}: sha256 {digest}, pinned {FEBA.sha256}")


# --------------------------------------------------------------------------- #
# Typed rows
# --------------------------------------------------------------------------- #
class FbOrganism(BaseModel):
    """One ``Organism`` row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    org_id: str
    genus: str
    species: str
    strain: str


class FbPublication(BaseModel):
    """One ``Publication`` row; ``url`` is stripped of surrounding whitespace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pub_id: str
    title: str
    url: str


class FbGene(BaseModel):
    """One ``Gene`` row (the columns a loader reads)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    locus_id: str
    sys_name: str | None
    gene: str | None
    gene_type: int = Field(
        description="1 protein-coding, 2 rRNA, 5 tRNA, 6 ncRNA, 7 pseudogene, 9 and 10 "
        "CRISPR, 11 antisense RNA, 99 other (the Gene table's own comment)"
    )


class FbCondition(BaseModel):
    """One ``condition_<k>`` / ``concentration_<k>`` / ``units_<k>`` triple, verbatim.

    ``concentration`` is the cell as a number when it reads as one, else None with the
    text kept in ``concentration_text``. A triple with a dose and no condition is kept as
    released (``condition`` None): the browser has such rows (``set9IT093`` of one
    organism), and the inventory reports them; a loader refuses them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    position: int = Field(description="k of condition_k")
    condition: str | None
    concentration_text: str | None
    concentration: float | None
    units: str | None


class FbQuality(BaseModel):
    """The browser's per-experiment quality metrics (Price 2018's success criteria)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_used: int
    g_med: int
    g_med_t0: int
    cor12: float
    mad12: float
    adjcor: float
    gccor: float
    max_fit: float


class FbExperiment(BaseModel):
    """One ``Experiment`` row, typed. Empty strings (the table's absent value) are None."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    org_id: str
    exp_name: str
    exp_desc: str
    exp_desc_long: str
    exp_group: str
    set_name: str
    mutant_library: str
    person: str
    date_started: str
    time_zero_set: str
    media: str
    media_strength: float
    temperature_c: float | None
    ph: str | None
    vessel: str | None
    aerobic: str | None
    liquid: str | None
    shaking: str | None
    conditions: tuple[FbCondition, ...] = Field(
        description="the non-empty condition_1..condition_4 columns, in order"
    )
    growth_plate: str | None
    growth_wells: str | None
    n_generations: float | None
    pub_id: str | None
    quality: FbQuality

    @property
    def screen_id(self) -> str:
        """``<orgId>:<expName>``: names are unique only within an organism."""
        return f"{self.org_id}:{self.exp_name}"


def _text(value: Any) -> str | None:
    """A TEXT cell, None when empty."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _number(value: Any, label: str) -> float | None:
    """A numeric cell stored as TEXT or REAL, None when empty; a non-number is refused."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label}: non-finite {value!r}")
    return number


#: A plain decimal number, as the browser's numeric TEXT cells are written.
_DECIMAL: Final = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def _maybe_number(text: str | None) -> float | None:
    """``text`` as a finite number, or None when it does not read as one."""
    if text is None or _DECIMAL.fullmatch(text) is None:
        return None
    return float(text)


def _conditions(row: Mapping[str, Any]) -> tuple[FbCondition, ...]:
    """Every condition triple with any non-empty cell, in column order, verbatim."""
    out: list[FbCondition] = []
    for k in (1, 2, 3, 4):
        condition = _text(row[f"condition_{k}"])
        text = _text(row[f"concentration_{k}"])
        units = _text(row[f"units_{k}"])
        if condition is None and text is None and units is None:
            continue
        out.append(
            FbCondition(
                position=k,
                condition=condition,
                concentration_text=text,
                concentration=_maybe_number(text),
                units=units,
            )
        )
    return tuple(out)


def experiment_from_row(row: Mapping[str, Any]) -> FbExperiment:
    """An :class:`FbExperiment` from one ``Experiment`` row as a column dict."""
    return FbExperiment(
        org_id=str(row["orgId"]),
        exp_name=str(row["expName"]),
        exp_desc=str(row["expDesc"]),
        exp_desc_long=str(row["expDescLong"]),
        exp_group=str(row["expGroup"]),
        set_name=str(row["setName"]),
        mutant_library=str(row["mutantLibrary"]),
        person=str(row["person"]),
        date_started=str(row["dateStarted"]),
        time_zero_set=str(row["timeZeroSet"]),
        media=str(row["media"]),
        media_strength=float(row["mediaStrength"]),
        temperature_c=_number(row["temperature"], "temperature"),
        ph=_text(row["pH"]),
        vessel=_text(row["vessel"]),
        aerobic=_text(row["aerobic"]),
        liquid=_text(row["liquid"]),
        shaking=_text(row["shaking"]),
        conditions=_conditions(row),
        growth_plate=_text(row["growthPlate"]),
        growth_wells=_text(row["growthWells"]),
        n_generations=_number(row["nGenerations"], "nGenerations"),
        pub_id=_text(row["pubId"]),
        quality=FbQuality(
            n_used=int(row["nUsed"]),
            g_med=int(row["gMed"]),
            g_med_t0=int(row["gMedt0"]),
            cor12=float(row["cor12"]),
            mad12=float(row["mad12"]),
            adjcor=float(row["adjcor"]),
            gccor=float(row["gccor"]),
            max_fit=float(row["maxFit"]),
        ),
    )


class FbValues(BaseModel):
    """An organism's gene fitness and moderated t: loci x experiments, as released."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    org_id: str
    loci: tuple[str, ...] = Field(description="locusId, sorted")
    experiments: tuple[str, ...] = Field(description="expName, in the requested order")
    fitness: npt.NDArray[np.float64] = Field(description="loci x experiments")
    t: npt.NDArray[np.float64] = Field(description="loci x experiments")


class Feba:
    """Read-only, immutable access to the pinned ``feba.db``.

    ``verify=True`` (the default) checks the size and sha256 against :data:`FEBA` before
    opening; the tests pass ``verify=False`` on a small fixture database with the same
    schema.
    """

    def __init__(self, path: str | Path, *, verify: bool = True) -> None:
        """Open ``path`` read-only after checking it is the pinned deposit."""
        if verify:
            verify_feba(path)
        self.path = Path(path)
        self.connection = sqlite3.connect(
            f"file:{self.path}?mode=ro&immutable=1", uri=True
        )
        self.connection.row_factory = sqlite3.Row
        self._fit_tables = {
            str(row[0]).removeprefix("FitByExp_")
            for row in self.connection.execute(
                "select name from sqlite_master where type = 'table' and name like "
                "'FitByExp\\_%' escape '\\'"
            )
        }

    def close(self) -> None:
        """Close the connection."""
        self.connection.close()

    def organisms(self) -> tuple[FbOrganism, ...]:
        """Every ``Organism`` row, by orgId."""
        return tuple(
            FbOrganism(
                org_id=str(r["orgId"]),
                genus=str(r["genus"]),
                species=str(r["species"]),
                strain=str(r["strain"]),
            )
            for r in self.connection.execute(
                "select orgId, genus, species, strain from Organism order by orgId"
            )
        )

    def experiments(self, org_id: str) -> tuple[FbExperiment, ...]:
        """Every experiment of an organism, in the browser's ``num`` order."""
        rows = self.connection.execute(
            "select * from Experiment where orgId = ? order by num", (org_id,)
        ).fetchall()
        experiments = tuple(experiment_from_row(dict(r)) for r in rows)
        names = [e.exp_name for e in experiments]
        if len(set(names)) != len(names):
            raise ValueError(f"{org_id}: an expName repeats")
        return experiments

    def publications(self) -> dict[str, FbPublication]:
        """``Publication`` rows by pubId."""
        return {
            str(r["pubId"]): FbPublication(
                pub_id=str(r["pubId"]), title=str(r["title"]), url=str(r["URL"]).strip()
            )
            for r in self.connection.execute(
                "select pubId, title, URL from Publication"
            )
        }

    def genes(self, org_id: str) -> dict[str, FbGene]:
        """``Gene`` rows of an organism by locusId."""
        return {
            str(r["locusId"]): FbGene(
                locus_id=str(r["locusId"]),
                sys_name=_text(r["sysName"]),
                gene=_text(r["gene"]),
                gene_type=int(r["type"]),
            )
            for r in self.connection.execute(
                "select locusId, sysName, gene, type from Gene where orgId = ?",
                (org_id,),
            )
        }

    def _fit_table(self, org_id: str) -> str:
        """``FitByExp_<orgId>``, refusing an organism with no such table."""
        if org_id not in self._fit_tables:
            raise KeyError(f"no FitByExp table for orgId {org_id!r}")
        return f"FitByExp_{org_id}"

    def gene_counts(self, org_id: str) -> dict[str, int]:
        """Per experiment, the loci with both a fitness and a t value."""
        table = self._fit_table(org_id)
        return {
            str(r[0]): int(r[1])
            for r in self.connection.execute(
                f"select expName, count(*) from {table} "  # noqa: S608  # table name checked against sqlite_master
                "where fit is not null and t is not null group by expName"
            )
        }

    def values(self, org_id: str, exp_names: Sequence[str]) -> FbValues:
        """The fitness and t of ``exp_names``, every locus each holds.

        Refused: a requested experiment with no value, a (locus, experiment) cell held by
        one experiment and not another (the matrix must be complete), and a null value.
        """
        table = self._fit_table(org_id)
        wanted = {name: j for j, name in enumerate(exp_names)}
        if len(wanted) != len(exp_names):
            raise ValueError(f"{org_id}: an experiment is requested twice")
        cells: dict[str, dict[int, tuple[float, float]]] = {}
        for exp, locus, fit, t in self.connection.execute(
            f"select expName, locusId, fit, t from {table}"  # noqa: S608  # table name checked against sqlite_master
        ):
            j = wanted.get(str(exp))
            if j is None:
                continue
            if fit is None or t is None:
                raise ValueError(f"{org_id}:{exp}:{locus}: a null fit or t")
            cells.setdefault(str(locus), {})[j] = (float(fit), float(t))
        loci = tuple(sorted(cells))
        fitness = np.full((len(loci), len(exp_names)), np.nan)
        t_values = np.full((len(loci), len(exp_names)), np.nan)
        for i, locus in enumerate(loci):
            for j, (fit, t) in cells[locus].items():
                fitness[i, j] = fit
                t_values[i, j] = t
        missing_cols = [
            exp_names[j] for j in range(len(exp_names)) if np.isnan(fitness[:, j]).all()
        ]
        if missing_cols:
            raise ValueError(f"{org_id}: no values for {missing_cols}")
        if np.isnan(fitness).any():
            raise ValueError(
                f"{org_id}: {int(np.isnan(fitness).sum())} (locus, experiment) cells are "
                "absent; the requested experiments do not share one locus set"
            )
        return FbValues(
            org_id=org_id,
            loci=loci,
            experiments=tuple(exp_names),
            fitness=fitness,
            t=t_values,
        )


# --------------------------------------------------------------------------- #
# Genes, as the deposit names them
# --------------------------------------------------------------------------- #
class ReleasedGene(BaseModel):
    """One locus with values, under the locus tag the deposit gives it.

    ``tag`` is the ``Gene.sysName`` (the table's own comment: "a locus tag like SO_1446
    or b2338"), ``name`` the ``Gene.gene`` symbol, else the tag. No assembly is consulted:
    the genome is a separate layer and decides nothing about what is stored.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    locus_id: str
    tag: str
    name: str


def released_genes(
    loci: Sequence[str], genes: Mapping[str, FbGene]
) -> tuple[ReleasedGene, ...]:
    """Each locus with values as :class:`ReleasedGene`, in ``loci`` order.

    Refused, as a change in the deposit's shape rather than resolved: a locus with no
    ``Gene`` row, one with no ``sysName``, and two loci sharing a ``sysName``.
    """
    out: list[ReleasedGene] = []
    seen: dict[str, str] = {}
    for locus in loci:
        if locus not in genes:
            raise ValueError(f"locusId {locus} has values but no Gene row")
        gene = genes[locus]
        if gene.sys_name is None:
            raise ValueError(f"locusId {locus} has no sysName")
        if gene.sys_name in seen:
            raise ValueError(
                f"sysName {gene.sys_name} names loci {seen[gene.sys_name]} and {locus}"
            )
        seen[gene.sys_name] = locus
        out.append(
            ReleasedGene(
                locus_id=locus, tag=gene.sys_name, name=gene.gene or gene.sys_name
            )
        )
    return tuple(out)


# --------------------------------------------------------------------------- #
# Studies
# --------------------------------------------------------------------------- #
SourceKind = Literal["primary", "compendium_only"]


class SourceStudy(BaseModel):
    """A browser publication as the study its samples are attributed to.

    ``browser_title`` and ``browser_url`` are the deposit's ``Publication`` row verbatim
    (checked by :func:`check_studies` on every build). ``doi`` and ``citation_key`` are
    the journal version the records' ``Publication`` names, from the literature or raw
    mirror's manifest when the study is mirrored, else from the browser's URL; a browser
    URL naming no DOI leaves ``doi`` None and ``pubmed_id`` set.

    ``source_kind`` follows [[torchcell.candidates.aggregation-decomposition]]:
    ``primary`` when the study released its own per-record values (the browser is that
    release, or the study's own data page is), ``compendium_only`` when the study is a
    compendium's own paper, whose only artifact besides the browser is the compendium
    release. ``own_release_dataset`` names the registered dataset that already loads the
    study's own release; a loader does not store that study's samples again (G4).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pub_id: str
    browser_title: str
    browser_url: str
    doi: str | None
    pubmed_id: str | None = None
    citation_key: str | None
    source_kind: SourceKind
    own_release_dataset: str | None = None
    issues: tuple[int, ...] = Field(
        default=(), description="the decomposition issues (#932-#976) naming this study"
    )
    note: str | None = None

    @property
    def publication(self) -> Publication:
        """The ``Publication`` every stored record of this study carries."""
        if self.doi is not None:
            return Publication(doi=self.doi, doi_url=f"https://doi.org/{self.doi}")
        if self.pubmed_id is None:
            raise ValueError(f"{self.pub_id}: neither a DOI nor a PubMed id")
        return Publication(
            pubmed_id=self.pubmed_id,
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{self.pubmed_id}/",
        )


def check_studies(
    studies: Mapping[str, SourceStudy], publications: Mapping[str, FbPublication]
) -> None:
    """Every study's browser title and URL equal the deposit's row, verbatim."""
    for pub_id, study in studies.items():
        if pub_id != study.pub_id:
            raise ValueError(f"{pub_id}: keyed under another pubId ({study.pub_id})")
        row = publications.get(pub_id)
        if row is None:
            raise ValueError(f"{pub_id}: not in the deposit's Publication table")
        if (row.title, row.url) != (study.browser_title, study.browser_url):
            raise ValueError(
                f"{pub_id}: the deposit says ({row.title!r}, {row.url!r}), the study "
                f"table ({study.browser_title!r}, {study.browser_url!r})"
            )


# --------------------------------------------------------------------------- #
# Decisions
# --------------------------------------------------------------------------- #
Fate = Literal[
    "stored",
    "unattributed_release",
    "own_release_loaded",
    "author_withdrawn",
    "medium_not_in_media_library",
    "condition_not_representable",
]

FATE_DESCRIPTIONS: Final[dict[Fate, str]] = {
    "stored": "stored by this loader, attributed to its browser publication",
    "unattributed_release": "the browser names no publication for the sample (empty "
    "pubId); whether an unattributed release counts as a study to load is an open "
    "owner decision of #931, so nothing of it is stored",
    "own_release_loaded": "the sample's publication is already loaded from its own "
    "release by another registered dataset (G4 duplication by content); that dataset "
    "serves the sample or refuses it under its own rule, and this loader does not "
    "store it again",
    "author_withdrawn": "the authors told readers to disregard it (the 2021 note on "
    "the compendium page, wetmore2015.AUTHOR_WITHDRAWAL); refused, never stored flagged",
    "medium_not_in_media_library": "the sample's medium has no MEDIA_LIBRARY entry",
    "condition_not_representable": "a condition column has no typed representation "
    "(named per sample)",
}


class SampleDecision(BaseModel):
    """What a browser loader does with one sample, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exp_name: str
    pub_id: str | None
    fate: Fate
    detail: str | None = Field(
        default=None,
        description="the dataset class, medium or condition the fate names",
    )


def decision(e: FbExperiment, fate: Fate, detail: str | None = None) -> SampleDecision:
    """``e``'s :class:`SampleDecision`."""
    return SampleDecision(
        exp_name=e.exp_name, pub_id=e.pub_id, fate=fate, detail=detail
    )


def fate_counts(decisions: Iterable[SampleDecision]) -> dict[str, int]:
    """Samples per fate."""
    return dict(Counter(d.fate for d in decisions))


def fate_counts_by_publication(
    decisions: Iterable[SampleDecision],
) -> dict[str, dict[str, int]]:
    """Samples per (pubId, fate); an empty pubId is ``no-pubId``."""
    out: dict[str, Counter[str]] = {}
    for d in decisions:
        out.setdefault(d.pub_id or "no-pubId", Counter())[d.fate] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


# --------------------------------------------------------------------------- #
# Standard error from the moderated t
# --------------------------------------------------------------------------- #
#: Wetmore 2015's normalization constant in the t denominator ("we use 0.1";
#: wetmore2015.T_STATISTIC_SIGMA quotes it).
T_SIGMA: Final = 0.1


def derived_se(fitness: float, t: float) -> float:
    """``max(se_obs, se_naive) = sqrt((f / t)^2 - 0.1^2)``, from the moderated t.

    The browser releases ``t = f / sqrt(0.1^2 + max(V_e, V_n))`` (Wetmore 2015 Methods;
    Price 2018 defines the standard error as "the maximum of two estimates"), so its
    standard error is recoverable at the browser's full precision. A zero f or t, or a
    pair below the identity's floor ``|f / t| >= 0.1``, is refused: on the 2026-10-11
    deposit no Keio or Putida pair is zero and the smallest ``|f / t|`` is 0.1020 (Keio)
    and 0.1011 (Putida), so a breach means the statistic is not that t.
    """
    if fitness == 0.0 or t == 0.0:
        raise ValueError(f"f={fitness}, t={t}: the standard error is undefined")
    q = abs(fitness / t)
    variance = q * q - T_SIGMA * T_SIGMA
    if variance <= 0.0:
        raise ValueError(f"|f/t| = {q} is below the identity's floor {T_SIGMA}")
    return math.sqrt(variance)


# --------------------------------------------------------------------------- #
# L4: agreement with a served compendium
# --------------------------------------------------------------------------- #
class SampleAgreement(BaseModel):
    """One shared sample: the two value sets over the genes both hold."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample: str
    n_genes: int
    n_only_here: int
    n_only_there: int
    pearson_r: float
    max_abs_difference: float


class CompendiumCheck(BaseModel):
    """The L4 agreement of this loader's values with a served compendium's store."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    against: str = Field(
        description="the store compared against, relative to DATA_ROOT"
    )
    statistic: Literal["pearson_r"] = "pearson_r"
    min_pearson_r: float = Field(description="the per-sample bar")
    compared: tuple[str, ...]
    absent_there: dict[str, str] = Field(
        description="this side's samples the compendium store does not hold -> why"
    )
    pooled_pearson_r: float = Field(
        description="Pearson r over every shared (gene, sample) value, pooled"
    )
    n_values: int
    max_abs_difference: float
    agreements: tuple[SampleAgreement, ...]

    @property
    def worst_sample_r(self) -> float:
        """The smallest per-sample Pearson r."""
        return min(a.pearson_r for a in self.agreements)

    @property
    def passed(self) -> bool:
        """Every compared sample agrees at least as well as the bar."""
        return bool(self.agreements) and all(
            a.pearson_r >= self.min_pearson_r for a in self.agreements
        )


def compendium_check(
    own: Mapping[str, Mapping[str, float]],
    served: Mapping[str, Mapping[str, float]],
    absent_there: Mapping[str, str],
    *,
    against: str,
    min_pearson_r: float,
) -> CompendiumCheck:
    """Pearson r of ``own`` against ``served`` per shared sample and pooled.

    Both map sample -> gene tag -> fitness. Every sample of ``own`` that ``served`` does
    not hold must be named in ``absent_there`` with its reason, else the check refuses; a
    shared sample with no shared gene is refused.
    """
    unexplained = sorted(s for s in own if s not in served and s not in absent_there)
    if unexplained:
        raise ValueError(
            f"samples absent from {against} without a reason: {unexplained}"
        )
    compared = sorted(s for s in own if s in served)
    agreements: list[SampleAgreement] = []
    xs: list[npt.NDArray[np.float64]] = []
    ys: list[npt.NDArray[np.float64]] = []
    for sample in compared:
        tags = sorted(set(own[sample]) & set(served[sample]))
        if not tags:
            raise ValueError(f"{sample}: no gene shared with {against}")
        x = np.array([own[sample][g] for g in tags], dtype=np.float64)
        y = np.array([served[sample][g] for g in tags], dtype=np.float64)
        xs.append(x)
        ys.append(y)
        agreements.append(
            SampleAgreement(
                sample=sample,
                n_genes=len(tags),
                n_only_here=len(set(own[sample]) - set(served[sample])),
                n_only_there=len(set(served[sample]) - set(own[sample])),
                pearson_r=float(np.corrcoef(x, y)[0, 1]),
                max_abs_difference=float(np.abs(x - y).max()),
            )
        )
    if not agreements:
        raise ValueError(f"no sample is shared with {against}")
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    return CompendiumCheck(
        against=against,
        min_pearson_r=min_pearson_r,
        compared=tuple(compared),
        absent_there={s: absent_there[s] for s in sorted(own) if s not in served},
        pooled_pearson_r=float(np.corrcoef(x_all, y_all)[0, 1]),
        n_values=int(x_all.size),
        max_abs_difference=float(np.abs(x_all - y_all).max()),
        agreements=tuple(agreements),
    )


def compendium_check_result(check: CompendiumCheck, name: str) -> LevelResult:
    """The check as a SUPPLEMENTARY L4 row."""
    return LevelResult(
        level=Level.L4,
        name=name,
        passed=check.passed,
        message=(
            f"SUPPLEMENTARY: {len(check.compared)} shared samples against "
            f"{check.against}, {check.n_values} (gene, sample) values; pooled Pearson r "
            f"{check.pooled_pearson_r:.6f}, smallest per-sample r "
            f"{check.worst_sample_r:.6f} against the bar {check.min_pearson_r}; largest "
            f"single-value difference {check.max_abs_difference:.3g}; "
            f"{len(check.absent_there)} samples not in that store"
        ),
        details=check.model_dump(mode="json"),
    )


def stored_fitness(
    records: Iterable[Mapping[str, Any]], org_id: str
) -> dict[str, dict[str, float]]:
    """Sample -> stored locus tag -> stored fitness, from an RB-TnSeq store's records.

    The sample is the record's ``screen_id`` without an ``<orgId>:`` prefix (Price 2018,
    Wetmore 2015 and these loaders write ``Keio:<name>``; Borchert 2024 writes the bare
    name). A record with more than one perturbation, or a (sample, tag) seen twice, is
    refused.
    """
    out: dict[str, dict[str, float]] = {}
    prefix = f"{org_id}:"
    for record in records:
        phenotype = record["experiment"]["phenotype"]
        perturbations = record["experiment"]["genotype"]["perturbations"]
        if len(perturbations) != 1:
            raise ValueError(f"expected one perturbation, got {len(perturbations)}")
        sample = str(phenotype["screen_id"]).removeprefix(prefix)
        tag = str(perturbations[0]["systematic_gene_name"])
        cell = out.setdefault(sample, {})
        if tag in cell:
            raise ValueError(f"{sample}:{tag} is stored twice")
        cell[tag] = float(phenotype["environment_response"])
    return out
