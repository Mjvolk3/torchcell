# experiments/036-dataset-fixes-before-kg-build/scripts/si_audit_loadable_ledger.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.si_audit_loadable_ledger]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/si_audit_loadable_ledger
"""Measure every row of the two SI audits' "Loadable now" tables against the tree.

The two bacterial SI audits each close with a ranked "Loadable now" table:
``notes/plan.bacteria-si-phenotype-audit-ecoli.md`` (21 ranks over 14 *E. coli* papers)
and ``notes/plan.bacteria-si-phenotype-audit-pputida.md`` (9 ranks over the *P. putida*
isoprenol group plus Caglar 2017). Twelve of the thirty have been implemented since, in
seven separate pull requests, and the audits' own dated sections record the corrections
each build forced. Nothing recorded which rank is still open, and the umbrella issue
#826 asks for exactly that before the KG 4.0 rebuild.

This script is that ledger, and every cell of it is a measurement on the checked-out
tree rather than a recollection:

- **landed** is the dataset registry plus the dev store. A rank declares the loader
  class or classes that would serve it; the rank is landed when every one is in
  ``torchcell.datasets.dataset_registry`` AND its dev LMDB
  (``$DATA_ROOT/<loader root>/processed/lmdb``) holds entries, counted with
  ``lmdb.Environment.stat()``, the same call ``ExperimentDataset.len`` makes. A rank
  that enriches records already served instead declares a module symbol, which is
  looked up in the loader source.
- **the landing commit** is ``git log --reverse -S'<class ...>' -- <module>``, the first
  commit that introduced that class name into that file, so the attribution survives the
  rebase-and-fast-forward landing this repo uses. ``--resolve-prs`` then maps that
  commit's SUBJECT onto a pull request by asking GitHub for each candidate PR's commit
  list and taking the PR with the FEWEST commits among those carrying the subject. That
  rule is what makes a stacked branch attribute correctly: PRs #779 to #784 were stacked,
  so six PRs carry the Carruthers commit and only #783 is the Carruthers PR.
- **in flight** is ``git diff --name-only origin/main...<branch>`` over every local
  branch, intersected with the rank's declared owner paths, plus a declared probe string
  that must appear as an ADDED line of that diff. A rank whose file some other branch is
  rewriting is reported as in flight even when it is otherwise open, because landing it
  here would collide.
- **refused** is a verbatim substring of a loader module or dendron note that states the
  decline. ``grep -F`` on the pinned text, never a judgment made here.
- **open** is what is left, and the ledger prints the blocker each open rank declares.

ONE RANK IS EXCLUDED BY INSTRUCTION, and the exclusion is data rather than silence:
*E. coli* rank 2, Shiver 2016's Nichols 2011 batch-0 block (835,337 values). Nichols 2011
has no PDF in the literature mirror (#691, a by-hand retrieval) and the owner's standing
decision is that the Shiver block is not to be landed before that paper is mirrored and
the duplication question settled. :data:`EXCLUDED_ROWS` carries the reason.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/si_audit_loadable_ledger.py
    python experiments/036-dataset-fixes-before-kg-build/scripts/si_audit_loadable_ledger.py --resolve-prs

``--resolve-prs`` is the only networked step (it shells out to ``gh``); without it the
ledger reports the landing commit and leaves ``landing_pr`` null, because guessing a PR
number from a commit subject is exactly the recollection this script exists to replace.
Results land beside the other row measurements as ``si_audit_loadable_ledger.json`` and
``si_audit_loadable_ledger.csv``, and the printed Markdown table is what the dated
section of each audit note carries.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

import lmdb
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO: Final = Path(__file__).resolve().parents[3]
RESULTS: Final = str(Path(__file__).resolve().parent.parent / "results")

GITHUB_REPO: Final = "Mjvolk3/torchcell"

#: The lowest PR number ``--resolve-prs`` asks GitHub about. Every SI-audit rank landed
#: after the audits were written (2026-10-07), and #690 predates them by weeks, so a
#: lower floor only adds requests that cannot match.
PR_FLOOR: Final = 690

RowState = Literal["landed", "in_flight", "open", "refused", "excluded"]


class ClassProbe(BaseModel):
    """A loader class that would serve a rank, and where to find it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    module: str = Field(description="repo-relative path of the loader module")


class SymbolProbe(BaseModel):
    """A verbatim string that must appear in a tracked file for the probe to pass."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(description="repo-relative path of the file to read")
    text: str = Field(description="verbatim substring; matched with str.__contains__")


class RetrievalProbe(BaseModel):
    """A release file a rank needs, and the raw-mirror key it would live under.

    A rank whose file is not on disk is retrieval-gated and therefore OPEN, never
    refused: the release carries the values and this project has not fetched them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    citation_key: str
    rel_path: str = Field(description="path under $DATA_ROOT/torchcell-raw/<key>/")


class AuditRow(BaseModel):
    """One row of one audit's "Loadable now" table, with its probes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    table: Literal["ecoli", "pputida"]
    rank: str
    item: str
    audit_records: str = Field(
        description="the record count the audit's own table gives"
    )
    classes: tuple[ClassProbe, ...] = ()
    enrichment: tuple[SymbolProbe, ...] = ()
    refusal: tuple[SymbolProbe, ...] = ()
    retrieval: tuple[RetrievalProbe, ...] = ()
    assigned_to: str | None = Field(
        default=None,
        description="a branch another agent owns this rank on, DECLARED not measured",
    )
    owner_paths: tuple[str, ...] = ()
    in_flight_probe: str | None = None
    blocker: str | None = Field(
        default=None, description="what the rank declares as its blocker while open"
    )
    excluded_reason: str | None = None

    @model_validator(mode="after")
    def _probe_declared(self) -> AuditRow:
        if self.excluded_reason is not None:
            return self
        if not (self.classes or self.enrichment or self.refusal or self.retrieval):
            raise ValueError(f"{self.table} rank {self.rank} declares no probe")
        return self


class StoreCount(BaseModel):
    """A dataset class's registry and dev-store measurement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    registered: bool
    root: str | None
    store_path: str | None
    entries: int | None
    landing_commit: str | None
    landing_subject: str | None
    landing_pr: int | None


class SymbolMeasure(BaseModel):
    """An enrichment probe's presence and the commit that first introduced it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    text: str
    present: bool
    landing_commit: str | None
    landing_subject: str | None
    landing_pr: int | None


class RowMeasurement(BaseModel):
    """One ledger line: a row's declared probes and what they measured."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    table: str
    rank: str
    item: str
    audit_records: str
    state: RowState
    stores: tuple[StoreCount, ...]
    records_now: int | None
    enrichment: tuple[SymbolMeasure, ...]
    enrichment_missing: tuple[str, ...]
    refusal_found: tuple[str, ...]
    retrieval_present: tuple[str, ...]
    retrieval_absent: tuple[str, ...]
    in_flight_branches: tuple[str, ...]
    landing_prs: tuple[int, ...]
    blocker: str | None
    excluded_reason: str | None


# --------------------------------------------------------------------------- #
# The rows, transcribed from the two audits' own tables
# --------------------------------------------------------------------------- #
ECOLI_AUDIT: Final = "notes/plan.bacteria-si-phenotype-audit-ecoli.md"
PPUTIDA_AUDIT: Final = "notes/plan.bacteria-si-phenotype-audit-pputida.md"

EXCLUDED_ROWS: Final = {
    ("ecoli", "2"): (
        "Nichols 2011 is not in the literature mirror (#691, a by-hand retrieval), and "
        "the owner's standing decision is that Shiver 2016's batch-0 block is not "
        "landed before that paper is mirrored and the duplication question is settled."
    )
}

ECOLI_ROWS: Final = (
    AuditRow(
        table="ecoli",
        rank="1",
        item="Price 2018 per-strain fitness (strain_fit.tab, strain_se)",
        audit_records="<= 24,626,916 (upper bound)",
        retrieval=(
            RetrievalProbe(
                citation_key="priceMutantPhenotypesThousands2018",
                rel_path="data/bigfit/html/Keio/strain_fit.tab",
            ),
        ),
        owner_paths=("torchcell/datasets/ecoli/price2018.py",),
        blocker=(
            "retrieval: strain_fit.tab is not in the raw mirror, and the 24,626,916 is "
            "an upper bound (152,018 barcodes x 162 samples), not a count"
        ),
    ),
    AuditRow(
        table="ecoli",
        rank="2",
        item="Shiver 2016 Nichols batch-0 block of S1 Dataset, 235 conditions",
        audit_records="835,337",
        excluded_reason=EXCLUDED_ROWS[("ecoli", "2")],
    ),
    AuditRow(
        table="ecoli",
        rank="3",
        item="Lamoureux 2023 Public K-12, 1,675 public RNA-seq samples",
        audit_records="1,675",
        classes=(
            ClassProbe(
                name="RnaseqPublicK12Lamoureux2023Dataset",
                module="torchcell/datasets/ecoli/lamoureux2023_public_k12.py",
            ),
        ),
    ),
    AuditRow(
        table="ecoli",
        rank="4",
        item="Rapp 2026 growth AUC over Table S2's curves",
        audit_records="1,514",
        classes=(
            ClassProbe(
                name="GrowthAucRapp2026Dataset",
                module="torchcell/datasets/ecoli/rapp2026_platforms.py",
            ),
        ),
    ),
    AuditRow(
        table="ecoli",
        rank="5",
        item="Rapp 2026 targeted LC-MS/MS fold change (Table S6)",
        audit_records="411 records, 1,256 values",
        classes=(
            ClassProbe(
                name="TargetedMetabolomeRapp2026Dataset",
                module="torchcell/datasets/ecoli/rapp2026_platforms.py",
            ),
        ),
    ),
    AuditRow(
        table="ecoli",
        rank="6",
        item="Rapp 2026 FI-MS absolute intensities (Table S5)",
        audit_records="411 records, 1,385 values",
        classes=(
            ClassProbe(
                name="MetaboliteIntensityRapp2026Dataset",
                module="torchcell/datasets/ecoli/rapp2026_platforms.py",
            ),
        ),
    ),
    AuditRow(
        table="ecoli",
        rank="7",
        item="Rapp 2026 Table S6 Intensity PrecMz",
        audit_records="411 records",
        refusal=(
            SymbolProbe(
                path="torchcell/datasets/ecoli/rapp2026_platforms.py",
                text="NOT loaded: an instrument-scale intensity with no normalization",
            ),
        ),
        blocker=None,
    ),
    AuditRow(
        table="ecoli",
        rank="8",
        item="Rapp 2026 Table S7's 2,847 annotated features",
        audit_records="254 records, 2,847 values",
        refusal=(
            SymbolProbe(
                path="torchcell/datasets/ecoli/rapp2026_platforms.py",
                text="Table S7 is gated on a release",
            ),
        ),
    ),
    AuditRow(
        table="ecoli",
        rank="9",
        item="Price 2018 Table S1 likely-essential E. coli genes",
        audit_records="324",
        classes=(
            ClassProbe(
                name="GeneEssentialityPrice2018EcoliDataset",
                module="torchcell/datasets/ecoli/price2018.py",
            ),
        ),
    ),
    AuditRow(
        table="ecoli",
        rank="10",
        item="Price 2018 Tables S2 and S3 wild-type growth calls",
        audit_records="192",
        refusal=(
            SymbolProbe(
                path="notes/torchcell.datasets.ecoli.price2018.md",
                text="**Agreed with the audit: not loadable as they stand.**",
            ),
        ),
    ),
    AuditRow(
        table="ecoli",
        rank="11",
        item="Lamoureux 2023 per-sample growth rate (metadata_qc.csv)",
        audit_records="103 of 241, ceiling 354",
        classes=(
            ClassProbe(
                name="GrowthRateLamoureux2023Dataset",
                module="torchcell/datasets/ecoli/lamoureux2023_growth.py",
            ),
        ),
        owner_paths=(
            "torchcell/datasets/ecoli/lamoureux2023.py",
            "torchcell/datasets/ecoli/lamoureux2023_public_k12.py",
        ),
        blocker=(
            "the audit calls it blocked by gap 1; PR #836 (#776) landed the absolute "
            "branch, so the remaining question is whether a matched reference rate is "
            "released per condition"
        ),
    ),
    AuditRow(
        table="ecoli",
        rank="12",
        item="Wang 2015 no-isoprenol OD600",
        audit_records="46 plus 1 reference",
        classes=(
            ClassProbe(
                name="GrowthWang2015Dataset",
                module="torchcell/datasets/ecoli/wang2015_growth.py",
            ),
        ),
        owner_paths=("torchcell/datasets/ecoli/wang2015.py",),
        blocker=(
            "a recorded decision rather than a release defect: it would be a second "
            "phenotype family beside the stored chemical-genomic records"
        ),
    ),
    AuditRow(
        table="ecoli",
        rank="13",
        item="Schmidt 2016 Table S23 per-condition growth rate + Stdev",
        audit_records="26",
        classes=(
            ClassProbe(
                name="GrowthRateS23Schmidt2016Dataset",
                module="torchcell/datasets/ecoli/schmidt2016_s23_growth_rate.py",
            ),
        ),
        owner_paths=(
            "torchcell/datasets/ecoli/schmidt2016.py",
            "torchcell/datasets/ecoli/schmidt2016_growth_rate.py",
        ),
        blocker=(
            "three blockers measured in the schmidt2016_growth_rate note; PR #836 "
            "(#776) lifted the first, and the Stdev's replicate design stays a gap"
        ),
    ),
    AuditRow(
        table="ecoli",
        rank="14",
        item="Schmidt 2016 Tables S2 and S3 SRM absolute abundances",
        audit_records="up to 22 records, 1,461 values",
        classes=(
            ClassProbe(
                name="ProteomeSrmSet1Schmidt2016Dataset",
                module="torchcell/datasets/ecoli/schmidt2016_srm.py",
            ),
            ClassProbe(
                name="ProteomeSrmSet2Schmidt2016Dataset",
                module="torchcell/datasets/ecoli/schmidt2016_srm.py",
            ),
        ),
    ),
    AuditRow(
        table="ecoli",
        rank="15",
        item="Caglar 2017 doubling time",
        audit_records="19",
        classes=(
            ClassProbe(
                name="DoublingTimeCaglar2017Dataset",
                module="torchcell/datasets/ecoli/caglar2017_doubling_time.py",
            ),
        ),
    ),
    AuditRow(
        table="ecoli",
        rank="16",
        item="Schmidt 2016 Table S24 deletion-strain growth rates",
        audit_records="6 plus 2 references",
        classes=(
            ClassProbe(
                name="GrowthRateSchmidt2016Dataset",
                module="torchcell/datasets/ecoli/schmidt2016_growth_rate.py",
            ),
        ),
    ),
    AuditRow(
        table="ecoli",
        rank="17",
        item="Gupta 2024 absolute protein concentration (Supplementary Data 6)",
        audit_records="1 record, 2,994 values",
        classes=(
            ClassProbe(
                name="ProteomeAbsoluteGupta2024Dataset",
                module="torchcell/datasets/ecoli/gupta2024_absolute.py",
            ),
        ),
        owner_paths=("torchcell/datasets/ecoli/gupta2024.py",),
        in_flight_probe="class ",
        blocker=(
            "two unresolved sourcing questions, neither a schema change: n_replicates "
            "is unsourced and the release never says which of the 13 conditions the "
            "label-free run was"
        ),
    ),
    AuditRow(
        table="ecoli",
        rank="18",
        item="Price 2018 Table S4 Solvent into the stress records",
        audit_records="0 new, enriches 207,240",
        enrichment=(
            SymbolProbe(
                path="torchcell/datasets/ecoli/price2018.py", text="def stress_solvent("
            ),
            SymbolProbe(
                path="torchcell/datasets/ecoli/price2018.py",
                text="SOLVENT_IS_THE_PRESCREEN_STOCK",
            ),
        ),
        classes=(
            ClassProbe(
                name="RbTnseqPrice2018EcoliDataset",
                module="torchcell/datasets/ecoli/price2018.py",
            ),
        ),
    ),
    AuditRow(
        table="ecoli",
        rank="19a",
        item="Lamoureux 2023 Public K-12 aerobicity",
        audit_records="0 standalone (1,425 inside rank 3)",
        enrichment=(
            SymbolProbe(
                path="torchcell/datasets/ecoli/lamoureux2023_public_k12.py",
                text="AEROBICITY_CELLS",
            ),
        ),
    ),
    AuditRow(
        table="ecoli",
        rank="19b",
        item="Lamoureux 2023 Public K-12 time",
        audit_records="0 standalone (391 inside rank 3)",
        refusal=(
            SymbolProbe(
                path="torchcell/datasets/ecoli/lamoureux2023_public_k12.py",
                text="the 'time' column carries no unit in its header",
            ),
        ),
    ),
    AuditRow(
        table="ecoli",
        rank="20",
        item="Lamoureux 2023 the 98 short / low-FPKM genes",
        audit_records="0 new, 23,618 values on built records",
        refusal=(
            SymbolProbe(
                path="notes/plan.bacteria-si-phenotype-audit-ecoli.md",
                text=(
                    "Already recorded, and the source's own QC removed them, so this "
                    "is a note rather than a recommendation"
                ),
            ),
        ),
    ),
    AuditRow(
        table="ecoli",
        rank="21",
        item="Lamoureux 2023 the 20 QC-failed libraries",
        audit_records="<= 20, uncounted",
        refusal=(
            SymbolProbe(
                path="torchcell/datasets/ecoli/lamoureux2023.py",
                text="1035 highquality RNA-seq samples",
            ),
        ),
        blocker=(
            "the release's own QC excluded them and the compendium the loader reads is "
            "the 1,035-sample one, so there is no released value to store"
        ),
    ),
)

PPUTIDA_ROWS: Final = (
    AuditRow(
        table="pputida",
        rank="1",
        item="Caglar 2017 doubling time per replicate growth curve",
        audit_records="55 (or 19 condition means)",
        classes=(
            ClassProbe(
                name="DoublingTimeCaglar2017Dataset",
                module="torchcell/datasets/ecoli/caglar2017_doubling_time.py",
            ),
        ),
    ),
    AuditRow(
        table="pputida",
        rank="2",
        item="Carruthers 2025 four unstored isoprenol titer sheets",
        audit_records="49 records over 190 cultures, plus 14 references",
        classes=(
            ClassProbe(
                name="IsoprenolTiterCarruthers2025Dataset",
                module="torchcell/datasets/pputida/carruthers2025.py",
            ),
        ),
        enrichment=(
            SymbolProbe(
                path="torchcell/datasets/pputida/carruthers2025.py",
                text='SHEET_OFFTARGET_TITER = "Supplementary Figure 13d"',
            ),
            SymbolProbe(
                path="torchcell/datasets/pputida/carruthers2025.py",
                text='SHEET_KO_ARRAYS = "Figure 6d"',
            ),
        ),
    ),
    AuditRow(
        table="pputida",
        rank="3",
        item="Carruthers 2025 two unstored per-protein abundance sheets",
        audit_records="21 records, about 630 values",
        classes=(
            ClassProbe(
                name="ProteomeCarruthers2025Dataset",
                module="torchcell/datasets/pputida/carruthers2025.py",
            ),
        ),
        enrichment=(
            SymbolProbe(
                path="torchcell/datasets/pputida/carruthers2025.py",
                text='SHEET_CONTROL_PROTEOME = "Supplementary Figure 15"',
            ),
            SymbolProbe(
                path="torchcell/datasets/pputida/carruthers2025.py",
                text='SHEET_OVEREXPRESSION_PROTEOME = "Supplementary Figure 12ac"',
            ),
        ),
    ),
    AuditRow(
        table="pputida",
        rank="4",
        item="Kang 2026 fed-batch isoprenol titer and residual sugars (Table S9)",
        audit_records="21 (7 titer + 14 metabolite)",
        refusal=(
            SymbolProbe(
                path="torchcell/datasets/pputida/kang2026.py",
                text="ISOPRENOL_NOT_A_RECORD",
            ),
        ),
    ),
    AuditRow(
        table="pputida",
        rank="5",
        item="de Siqueira 2025 two further proteomics normalizations",
        audit_records="10 (2 normalizations x 5 samples)",
        classes=(
            ClassProbe(
                name="ProteomePercentDeSiqueira2025Dataset",
                module="torchcell/datasets/pputida/desiqueira2025.py",
            ),
            ClassProbe(
                name="ProteomeLog10PercentDeSiqueira2025Dataset",
                module="torchcell/datasets/pputida/desiqueira2025.py",
            ),
        ),
    ),
    AuditRow(
        table="pputida",
        rank="6",
        item="Menasalvas 2025 metabolites and five proteomics sheets (Dryad)",
        audit_records="about 4 metabolite records plus the designed-strain arms",
        classes=(
            ClassProbe(
                name="MetaboliteMenasalvas2025Dataset",
                module="torchcell/datasets/pputida/menasalvas2025.py",
            ),
        ),
        retrieval=(
            RetrievalProbe(
                citation_key="menasalvasBiosensordrivenStrainEngineering2025",
                rel_path="data/dryad/doi_10_5061_dryad_sbcc2frjq__v20250919.zip",
            ),
        ),
        assigned_to="feat/739-788-manual-deposits-consumed",
        owner_paths=("torchcell/datasets/pputida/menasalvas2025.py",),
        blocker=(
            "the Dryad deposit the audit calls unfetched IS now in the raw mirror, so "
            "the row is no longer retrieval-gated; it is assigned to the #739/#788 "
            "branch, which has not yet touched the loader (measured)"
        ),
    ),
    AuditRow(
        table="pputida",
        rank="7",
        item="Yunus 2026 per-protein fold change, PP_4188 strain (Tables S4, S5)",
        audit_records="1 record carrying 338 protein keys",
        classes=(
            ClassProbe(
                name="CrispriDifferentialProteomeYunus2026Dataset",
                module="torchcell/datasets/pputida/yunus2026.py",
            ),
        ),
    ),
    AuditRow(
        table="pputida",
        rank="8",
        item="Carruthers 2025 knockdown ratios (Figure 3c), CONDITIONAL on gap R",
        audit_records="92 records of a 2-protein profile",
        classes=(
            ClassProbe(
                name="ProteomeFoldChangeCarruthers2025Dataset",
                module="torchcell/datasets/pputida/carruthers2025_fold_change.py",
            ),
        ),
        owner_paths=("torchcell/datasets/pputida/carruthers2025.py",),
        blocker="gap R: ProteinAbundancePhenotype's docstring forbids a ratio",
    ),
    AuditRow(
        table="pputida",
        rank="9",
        item="Lim 2025 three IPL400 production-proteome arms, CONDITIONAL",
        audit_records="0 to 2",
        classes=(
            ClassProbe(
                name="ProteomeProductionLim2025Dataset",
                module="torchcell/datasets/pputida/lim2025_production.py",
            ),
        ),
        owner_paths=("torchcell/datasets/pputida/lim2025.py",),
        blocker=(
            "needs a MEDIA_LIBRARY entry for the production medium and a pairing "
            "decision against an evolved strain"
        ),
    ),
)

ROWS: Final = ECOLI_ROWS + PPUTIDA_ROWS


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #
def git(*args: str) -> str:
    """Run a read-only git command in the repo and return its stdout."""
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout


def local_branches() -> list[str]:
    """Every local branch except ``main``, in the order git lists them."""
    names = git("branch", "--format=%(refname:short)").split("\n")
    return [name for name in names if name and name != "main"]


def branch_added_lines(branch: str, path: str) -> str:
    """The ADDED lines of ``branch``'s diff against ``origin/main`` for one path."""
    diff = subprocess.run(
        ["git", "-C", str(REPO), "diff", f"origin/main...{branch}", "--", path],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return "\n".join(
        line[1:] for line in diff.split("\n") if line.startswith("+") and line != "+++"
    )


def first_commit_introducing(needle: str, path: str) -> tuple[str, str] | None:
    """The oldest commit whose diff for ``path`` changed the occurrence count of ``needle``.

    ``git log -S`` counts occurrences of the string, so the first commit in
    ``--reverse`` order is the one that introduced it. The landing is a rebase plus a
    fast-forward in this repo, so this is the commit that is on ``main``, and its
    SUBJECT is what the pull request's own commit list still carries.
    """
    out = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "log",
            "--reverse",
            "--format=%H%x09%s",
            "-S",
            needle,
            "--",
            path,
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if not out:
        return None
    sha, _, subject = out.split("\n")[0].partition("\t")
    return sha, subject


def pr_commit_subjects() -> dict[str, list[tuple[int, int]]]:
    """Map a commit SUBJECT onto ``(pr_number, n_commits_in_that_pr)`` pairs.

    Built by asking GitHub for every pull request from :data:`PR_FLOOR` up and listing
    each one's commits. A stacked branch puts one commit in several PRs, so the caller
    resolves the attribution by taking the PR with the fewest commits (see
    :func:`attribute_pr`).
    """
    listing = subprocess.run(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            GITHUB_REPO,
            "--state",
            "all",
            "--limit",
            "500",
            "--json",
            "number",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    numbers = sorted(
        item["number"] for item in json.loads(listing) if item["number"] >= PR_FLOOR
    )
    subjects: dict[str, list[tuple[int, int]]] = {}
    for number in numbers:
        commits = subprocess.run(
            [
                "gh",
                "api",
                f"repos/{GITHUB_REPO}/pulls/{number}/commits",
                "--paginate",
                "--jq",
                '.[].commit.message | split("\\n")[0]',
            ],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        heads = [line for line in commits.split("\n") if line.strip()]
        for subject in heads:
            subjects.setdefault(subject, []).append((number, len(heads)))
    return subjects


def attribute_pr(
    subject: str, subjects: dict[str, list[tuple[int, int]]]
) -> int | None:
    """The PR that introduced ``subject``: of the PRs carrying it, the smallest one.

    "Smallest" is by commit count, then by number. PRs #779 to #784 were a stack, so
    six of them carry the Carruthers commit; #783 is the one with three commits and is
    the Carruthers pull request.
    """
    candidates = subjects.get(subject)
    if not candidates:
        return None
    return min(candidates, key=lambda pair: (pair[1], pair[0]))[0]


def store_entries(path: str) -> int | None:
    """The LMDB entry count at ``path``, or None when no store is built there."""
    if not osp.isdir(path):
        return None
    env = lmdb.open(path, readonly=True, lock=False, subdir=True)
    try:
        return int(env.stat()["entries"])
    finally:
        env.close()


def measure_classes(
    row: AuditRow,
    registry: dict[str, type],
    data_root: str,
    subjects: dict[str, list[tuple[int, int]]] | None,
) -> tuple[StoreCount, ...]:
    """Measure every class a row declares against the registry and the dev tree."""
    from torchcell.database.build_dataset_lmdb import dataset_default_root

    counts: list[StoreCount] = []
    for probe in row.classes:
        cls = registry.get(probe.name)
        root = dataset_default_root(cls) if cls is not None else None
        store = osp.join(data_root, root, "processed", "lmdb") if root else None
        landing = first_commit_introducing(f"class {probe.name}(", probe.module)
        counts.append(
            StoreCount(
                name=probe.name,
                registered=cls is not None,
                root=root,
                store_path=store,
                entries=store_entries(store) if store else None,
                landing_commit=landing[0] if landing else None,
                landing_subject=landing[1] if landing else None,
                landing_pr=(
                    attribute_pr(landing[1], subjects)
                    if landing and subjects is not None
                    else None
                ),
            )
        )
    return tuple(counts)


def found_probes(probes: tuple[SymbolProbe, ...]) -> tuple[list[str], list[str]]:
    """Split declared probes into those present in their file and those absent."""
    found: list[str] = []
    missing: list[str] = []
    for probe in probes:
        path = REPO / probe.path
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        (found if probe.text in text else missing).append(
            f"{probe.path}::{probe.text[:60]}"
        )
    return found, missing


def measure_enrichment(
    probes: tuple[SymbolProbe, ...], subjects: dict[str, list[tuple[int, int]]] | None
) -> tuple[SymbolMeasure, ...]:
    """Measure each enrichment probe's presence and the commit that introduced it.

    A row that fills a field on records some earlier pull request already served cannot
    be attributed by its loader class, because the class predates the enrichment: the
    Price 2018 stress records landed in #725 and Table S4's solvent reached them in
    #782. The probe string is what ``git log -S`` is asked about instead.
    """
    measures: list[SymbolMeasure] = []
    for probe in probes:
        path = REPO / probe.path
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        landing = first_commit_introducing(probe.text, probe.path)
        measures.append(
            SymbolMeasure(
                path=probe.path,
                text=probe.text,
                present=probe.text in text,
                landing_commit=landing[0] if landing else None,
                landing_subject=landing[1] if landing else None,
                landing_pr=(
                    attribute_pr(landing[1], subjects)
                    if landing and subjects is not None
                    else None
                ),
            )
        )
    return tuple(measures)


def found_retrievals(
    probes: tuple[RetrievalProbe, ...], data_root: str
) -> tuple[list[str], list[str]]:
    """Split declared release files into those on disk in the raw mirror and those not."""
    present: list[str] = []
    absent: list[str] = []
    for probe in probes:
        path = osp.join(data_root, "torchcell-raw", probe.citation_key, probe.rel_path)
        (present if osp.exists(path) else absent).append(path)
    return present, absent


def in_flight_for(row: AuditRow, branches: list[str]) -> tuple[str, ...]:
    """Branches other than main whose diff touches a path this row owns.

    The declared classes' own modules count as owned paths too, so a branch that is
    already writing the loader a row needs is reported even when the row declares no
    ``owner_paths`` of its own.
    """
    paths = set(row.owner_paths) | {probe.module for probe in row.classes}
    hits: list[str] = []
    for branch in branches:
        for path in sorted(paths):
            added = branch_added_lines(branch, path)
            if not added:
                continue
            if row.in_flight_probe is None or row.in_flight_probe in added:
                hits.append(branch)
                break
    return tuple(hits)


def classify(
    row: AuditRow,
    stores: tuple[StoreCount, ...],
    enrichment_missing: list[str],
    refusal_found: list[str],
    in_flight: tuple[str, ...],
) -> RowState:
    """Derive the row's state from its measurements, in a fixed order."""
    if row.excluded_reason is not None:
        return "excluded"
    classes_served = bool(stores) and all(
        store.registered and (store.entries or 0) > 0 for store in stores
    )
    if classes_served and not enrichment_missing:
        return "landed"
    if not row.classes and row.enrichment and not enrichment_missing:
        return "landed"
    if refusal_found and not row.classes:
        return "refused"
    if refusal_found and not classes_served:
        return "refused"
    if in_flight:
        return "in_flight"
    return "open"


def landing_prs(
    row: AuditRow, stores: tuple[StoreCount, ...], enrichment: tuple[SymbolMeasure, ...]
) -> tuple[int, ...]:
    """The pull requests that landed this row: the enrichment's when one is declared.

    An enrichment row's deliverable IS the enrichment, so its loader class's own
    pull request would misattribute it; a row without enrichment is attributed by the
    class that serves it.
    """
    source = (
        [measure.landing_pr for measure in enrichment]
        if row.enrichment
        else [store.landing_pr for store in stores]
    )
    return tuple(sorted({number for number in source if number}))


def measure_row(
    row: AuditRow,
    registry: dict[str, type],
    data_root: str,
    branches: list[str],
    subjects: dict[str, list[tuple[int, int]]] | None,
) -> RowMeasurement:
    """Run every probe one row declares and classify the result."""
    stores = measure_classes(row, registry, data_root, subjects)
    enrichment = measure_enrichment(row.enrichment, subjects)
    _, enrichment_missing = found_probes(row.enrichment)
    refusal_found, _ = found_probes(row.refusal)
    retrieval_present, retrieval_absent = found_retrievals(row.retrieval, data_root)
    in_flight = in_flight_for(row, branches)
    state = classify(row, stores, enrichment_missing, refusal_found, in_flight)
    records = sum(store.entries or 0 for store in stores) if stores else None
    return RowMeasurement(
        table=row.table,
        rank=row.rank,
        item=row.item,
        audit_records=row.audit_records,
        state=state,
        stores=stores,
        records_now=records if state == "landed" else records,
        enrichment=enrichment,
        enrichment_missing=tuple(enrichment_missing),
        refusal_found=tuple(refusal_found),
        retrieval_present=tuple(retrieval_present),
        retrieval_absent=tuple(retrieval_absent),
        in_flight_branches=in_flight if state != "landed" else (),
        landing_prs=landing_prs(row, stores, enrichment),
        blocker=row.blocker,
        excluded_reason=row.excluded_reason,
    )


def markdown_table(rows: list[RowMeasurement]) -> str:
    """Render the ledger as the Markdown table the audit notes carry."""
    head = (
        "| rank | item | audit's count | state | records now | where |\n"
        "|---|---|---|---|---|---|"
    )
    lines = [head]
    for row in rows:
        if row.state == "landed":
            prs = ", ".join(f"PR #{n}" for n in row.landing_prs)
            where = prs if prs else "landed, PR unresolved"
        elif row.state == "in_flight":
            where = ", ".join(f"`{b}`" for b in row.in_flight_branches)
        elif row.state == "refused":
            where = row.refusal_found[0].split("::")[0] if row.refusal_found else ""
        elif row.state == "excluded":
            where = "excluded by owner decision"
        else:
            where = row.blocker or ""
            if row.retrieval_absent:
                where = f"{where}; file absent from the raw mirror"
        records = "" if row.records_now in (None, 0) else f"{row.records_now:,}"
        lines.append(
            f"| {row.rank} | {row.item} | {row.audit_records} | **{row.state}** | "
            f"{records} | {where} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Measure both audits' loadable-now tables and write the ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolve-prs",
        action="store_true",
        help="ask GitHub which pull request introduced each landing commit",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]

    import torchcell.datasets.ecoli  # noqa: F401  # populates the registry
    import torchcell.datasets.pputida  # noqa: F401  # populates the registry
    from torchcell.datasets.dataset_registry import dataset_registry

    registry = dict(dataset_registry)
    branches = local_branches()
    subjects = pr_commit_subjects() if args.resolve_prs else None

    measured = [
        measure_row(row, registry, data_root, branches, subjects) for row in ROWS
    ]
    by_state = Counter(row.state for row in measured)

    ecoli = [row for row in measured if row.table == "ecoli"]
    pputida = [row for row in measured if row.table == "pputida"]

    results: dict[str, Any] = {
        "measured_at": datetime.now(UTC).isoformat(),
        "head": git("rev-parse", "HEAD").strip(),
        "data_root": data_root,
        "ecoli_audit": ECOLI_AUDIT,
        "pputida_audit": PPUTIDA_AUDIT,
        "prs_resolved": args.resolve_prs,
        "n_rows": len(measured),
        "totals": dict(sorted(by_state.items())),
        "rows": [row.model_dump() for row in measured],
    }

    os.makedirs(RESULTS, exist_ok=True)
    json_path = osp.join(RESULTS, "si_audit_loadable_ledger.json")
    with open(json_path, "w") as handle:
        json.dump(results, handle, indent=2)
    frame = pd.DataFrame(
        [
            {
                "table": row.table,
                "rank": row.rank,
                "item": row.item,
                "audit_records": row.audit_records,
                "state": row.state,
                "records_now": row.records_now,
                "landing_prs": ";".join(str(n) for n in row.landing_prs),
                "in_flight_branches": ";".join(row.in_flight_branches),
                "blocker": row.blocker or "",
            }
            for row in measured
        ]
    )
    csv_path = osp.join(RESULTS, "si_audit_loadable_ledger.csv")
    frame.to_csv(csv_path, index=False)

    print(f"rows measured: {len(measured)}")
    print(json.dumps(dict(sorted(by_state.items())), indent=2))
    print("\n### E. coli audit, loadable-now ledger\n")
    print(markdown_table(ecoli))
    print("\n### P. putida audit, loadable-now ledger\n")
    print(markdown_table(pputida))
    print(f"\nwrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
