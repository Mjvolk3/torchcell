# scripts/release_parser.py
# [[scripts.release_parser]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/release_parser.py
# Test file: tests/scripts/test_release_parser.py
"""The semantic-release commit parser for torchcell's ``TAG(scope): subject`` commits.

python-semantic-release 10.4.1's ``scipy`` parser (the repo's parser until 2026-09-29)
accepts only ``TAG: subject`` and ``TAG: scope: subject``: its prefix regex puts a colon,
never a parenthesis, after the tag, so the parenthesized scope most commits carry
(``FIX(losses): ...``, ``DOCS(readme): ...``) does not parse and never bumps a release,
whatever ``allowed_tags`` says. Measured on the last 30 subjects of ``main``: 23 did not
parse under scipy, among them every ``FIX(...)`` commit. The ``conventional`` parser
reads ``TAG(scope)!: subject`` but has no ``major_tags``: a major bump comes only from
``!`` or a ``BREAKING CHANGE:`` paragraph.

This parser is the conventional parser plus ``major_tags``, so the tag map in
``pyproject.toml`` keeps its meaning. Since the second 2026.09.29 revision a bump is
deliberate: ``API`` major; ``REL`` minor (a source release, cut before a KG build); ``DB``
patch (a database compatibility change) and, since 2026.10.01, ``PATCH`` patch (a source
patch release with no interface or database change); every other allowed tag, ``FEAT``
and ``FIX`` included, parses with no bump. ``!`` and ``BREAKING CHANGE:`` still force a major. Loaded by file path:
``commit_parser = "scripts/release_parser.py:TorchcellCommitParser"`` (python-semantic-
release resolves ``path.py:Class`` relative to the working directory, which is the
checkout in the release action). Imports only ``semantic_release``, so it needs no
torchcell install where the action runs. Record: ``notes/versioning.md`` (2026.09.29).
"""

from __future__ import annotations

from pydantic.dataclasses import dataclass
from semantic_release.commit_parser.conventional import (
    ConventionalCommitParser,
    ConventionalCommitParserOptions,
)
from semantic_release.enums import LevelBump


@dataclass
class TorchcellParserOptions(ConventionalCommitParserOptions):
    """Conventional parser options plus the tags that bump a MAJOR version."""

    major_tags: tuple[str, ...] = ("API",)
    """Commit-type prefixes that should result in a major release bump."""

    def __post_init__(self) -> None:
        """Build the conventional tag map, then lift ``major_tags`` to MAJOR."""
        super().__post_init__()
        for tag in self.major_tags:
            self._tag_to_level[str(tag)] = LevelBump.MAJOR


class TorchcellCommitParser(ConventionalCommitParser):
    """``TAG(scope): subject`` with the scipy-style tag names and a major tag."""

    parser_options = TorchcellParserOptions
