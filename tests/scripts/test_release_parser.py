# tests/scripts/test_release_parser.py
# [[tests.scripts.test_release_parser]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/scripts/test_release_parser.py
"""The configured semantic-release parser on the last 30 real subjects of ``main``.

The parser and its options are loaded exactly as python-semantic-release loads them:
``commit_parser`` from ``pyproject.toml`` through ``semantic_release.helpers
.dynamic_import`` (the ``path.py:Class`` form), options from
``[tool.semantic_release.commit_parser_options]``. ``SUBJECTS`` is
``git log --format=%s -30 origin/main`` on 2026-09-29 (head ``901d5ec31``), verbatim.
Under the previous ``scipy`` parser 23 of these did not parse (every ``TAG(scope):``
subject); under this parser only the version-bump commit ``1.2.1`` and the untagged
``fig(008)`` subject do not, and the eight ``FIX``/``MAINT``/``PERF`` commits bump a patch.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from semantic_release.enums import LevelBump
from semantic_release.helpers import dynamic_import

REPO = Path(__file__).resolve().parents[2]

SUBJECTS: list[tuple[str, str | None]] = [
    ("NOTE(plan): data release program, versioning spine, docs, showcase, download API, query drift gate", "NO_RELEASE"),
    ("DOCS(readme): status badges from the branch head's check runs, not the workflow run list", "NO_RELEASE"),
    ("FIX(sga/viz): boxplot tick_labels for matplotlib 3.11; CI-conditional tests (cellpose, tight bbox)", "PATCH"),
    ("TST(utils): read PAPER_RC values by iterating rcParams (CI matplotlib stubs)", "NO_RELEASE"),
    ("TST: test-suite build-out Phase 9, sga, verification runners, viz, scheduler, KG build scripts, adapters (audited)", "NO_RELEASE"),
    ("TST(yeast_GEM): drop hypernetx-internal edge properties before comparing (CI hypernetx 2.4.3)", "NO_RELEASE"),
    ("TST: test-suite build-out Phase 8, adapter, manifest, training stack, remaining loaders, genome, GEM, literature (audited)", "NO_RELEASE"),
    ("FIX(neo4j_query_raw): phenotype label field, parallel reference key, len closes its store", "PATCH"),
    ("TST(loaders): close the PyG subset handle before reopening the store (CI py-lmdb)", "NO_RELEASE"),
    ("TST(loaders): type the LMDB reads for both lmdb stub versions (CI mypy)", "NO_RELEASE"),
    ("NOTE(weekly): drop the stale loss-fix decision bullet, name the Phase 7 audit model", "NO_RELEASE"),
    ("TST: test-suite build-out Phase 7, the remaining loaders, graph and release code, datamodule and model leftovers (audited)", "NO_RELEASE"),
    ("FIX(losses): type the soft-sort apply call without a type: ignore (CI mypy)", "PATCH"),
    ("FIX(losses): soft-sort gradient sign, DistLoss pairing and default weight, SupCR tie rule, combined weights, scheduled temperature", "PATCH"),
    ("fig(008): Fig 1b draws the positive case, Fig 3e and f recast as hypotheses", None),
    ("NOTE(losses): keep the composite weight arithmetic out of markdown emphasis", "NO_RELEASE"),
    ("TST: test-suite build-out Phase 6, the item pipeline (graph processors, losses, LMDB stages, Neo4jCellDataset) (audited)", "NO_RELEASE"),
    ("TST(kuzmin): close the first LMDB handle before reopening the store (CI py-lmdb)", "NO_RELEASE"),
    ("TST(kuzmin2018): guard the LMDB read before unpickling (CI mypy)", "NO_RELEASE"),
    ("TST: test-suite build-out Phase 5, dataset loaders on synthetic raw files (audited)", "NO_RELEASE"),
    ("TST(knowledge_graphs): exercise the lazy import path in-process", "NO_RELEASE"),
    ("TST(knowledge_graphs): assert the unknown-attribute error in-process for diff-cover", "NO_RELEASE"),
    ("FIX(knowledge_graphs): lazy names are the submodules; the dict is imported from its module", "PATCH"),
    ("PERF(ops): make ops-fast, OPS_HOSTS, and a lazy knowledge_graphs package import", "PATCH"),
    ("1.2.1", None),
    ("MAINT(ci): the mypy job skips deleted files in its diff scope", "PATCH"),
    ("MAINT: test-suite build-out Phase 0c, deprecate yeastmine and cpu_benchmark_system_monitor", "PATCH"),
    ("DOCS(skills): test-campaign skill, one coverage phase from targets to a landed PR", "NO_RELEASE"),
    ("TST: test-suite build-out Phase 4, hermetic script tests (audited)", "NO_RELEASE"),
    ("TST: test-suite build-out Phase 3, kg/graph/data/ontology (audited)", "NO_RELEASE"),
]  # fmt: skip


@pytest.fixture(scope="module")
def parser() -> object:
    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    section = config["tool"]["semantic_release"]
    assert section["commit_parser"] == "scripts/release_parser.py:TorchcellCommitParser"
    cls = dynamic_import(
        str(REPO / "scripts" / "release_parser.py") + ":TorchcellCommitParser"
    )
    options = cls.parser_options(
        **{k: tuple(v) for k, v in section["commit_parser_options"].items()}
    )
    return cls(options)


def _bump(parser: object, subject: str) -> str | None:
    result = parser.parse_message(subject)  # type: ignore[attr-defined]
    return None if result is None else result.bump.name


def test_the_last_30_real_subjects_bump_as_recorded(parser: object) -> None:
    assert [(s, _bump(parser, s)) for s, _ in SUBJECTS] == SUBJECTS
    assert sum(1 for _, bump in SUBJECTS if bump == "PATCH") == 8
    assert sum(1 for _, bump in SUBJECTS if bump is None) == 2


def test_tag_map_levels_and_the_breaking_markers(parser: object) -> None:
    """API is major; FEAT, ENH, DEP, DEV, REV minor; FIX, BUG, BLD, MAINT, PERF patch;
    DOC, DOCS, NOTE, TST, TEST, STY, CI, REL, BENCH no bump; "!" and a BREAKING CHANGE
    paragraph force a major on any tag; an unknown tag does not parse.
    """
    levels = {
        "API": "MAJOR",
        "FEAT": "MINOR", "ENH": "MINOR", "DEP": "MINOR", "DEV": "MINOR", "REV": "MINOR",
        "FIX": "PATCH", "BUG": "PATCH", "BLD": "PATCH", "MAINT": "PATCH", "PERF": "PATCH",
        "DOC": "NO_RELEASE", "DOCS": "NO_RELEASE", "NOTE": "NO_RELEASE",
        "TST": "NO_RELEASE", "TEST": "NO_RELEASE", "STY": "NO_RELEASE",
        "CI": "NO_RELEASE", "REL": "NO_RELEASE", "BENCH": "NO_RELEASE",
    }  # fmt: skip
    assert {tag: _bump(parser, f"{tag}: subject") for tag in levels} == levels
    assert {tag: _bump(parser, f"{tag}(scope): subject") for tag in levels} == levels
    assert _bump(parser, "TST!: subject") == "MAJOR"
    assert _bump(parser, "FIX(a): b\n\nBREAKING CHANGE: c") == "MAJOR"
    assert _bump(parser, "fix(a): lowercase is not in the map") is None
    assert _bump(parser, "WIP: unknown tag") is None
    assert parser.options.tag_to_level["API"] is LevelBump.MAJOR  # type: ignore[attr-defined]
