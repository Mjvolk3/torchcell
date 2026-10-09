# tests/torchcell/test_markdownlint_quote_protection.py
# [[tests.torchcell.test_markdownlint_quote_protection]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/test_markdownlint_quote_protection.py
"""The markdownlint autofix path may not edit characters inside a line (#846).

A dendron note carries the same verbatim quote a loader pins in
``SourcedValue.quote``, and the ``markdownlint-cli2 --fix`` pre-commit hook rewrote two
of them in ``notes/torchcell.datasets.pputida.menasalvas2025.md``: MD034 turned the bare
DOI ``doi:10.1126/sciadv.ady2677`` into an autolink and MD037 deleted the spaces in
``ΔPP_ 2428``. The sha256 pin on the SOURCE did not catch it, because the source was
never touched; only a re-read of the mirror bytes did.

Two things are pinned here, and both are hermetic (no node, no network, no mirror):

1. ``.markdownlint.jsonc`` disables every markdownlint rule whose fix edits characters
   inside a line. That one file governs BOTH autofix paths, the pre-commit hook and the
   VS Code extension's on-save ``source.fixAll.markdownlint``.
2. ``tests/torchcell/data/markdownlint_canary/quote_canary.md`` still holds each shape
   verbatim. The hook lints that file on every commit that touches it, so a
   reintroduced inline autofix rewrites the canary and fails this test rather than
   silently editing a note.
"""

import json
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / ".markdownlint.jsonc"
PRE_COMMIT = REPO / ".pre-commit-config.yaml"
CANARY_REL = "tests/torchcell/data/markdownlint_canary/quote_canary.md"
CANARY = REPO / CANARY_REL

# Every markdownlint rule that ships an autofix which rewrites characters INSIDE a line
# of prose, a blockquote, a backtick span, or a fenced block. Measured against
# markdownlint v0.36.1 (markdownlint-cli2 v0.16.0, the pinned `rev`). Add a rule here
# when markdownlint ships a new inline autofix.
INLINE_FIX_RULES = (
    "MD011",  # reversed-link-syntax: `(text)[url]` -> `[text](url)`
    "MD027",  # multiple-spaces-after-blockquote-symbol: drops a quoted line's indent
    "MD034",  # no-bare-urls: wraps a bare URL or DOI in `<>`
    "MD037",  # no-space-in-emphasis: deletes the spaces beside `_` or `*`
    "MD038",  # no-space-in-code: deletes the spaces beside a backtick delimiter
    "MD039",  # no-space-in-links: deletes the spaces inside link text
    "MD044",  # proper-names: rewrites capitalization, code spans included
    "MD049",  # emphasis-style: swaps `_` for `*`
    "MD050",  # strong-style: swaps `__` for `**`
    "MD053",  # link-reference-definitions: deletes a definition it reads as unused
)

# The exact byte sequences #846 found rewritten, plus one per other inline rule, each as
# it appears in the canary. A sequence that has been autofixed away no longer matches.
CANARY_SHAPES = {
    "MD034": "the deposit at https://doi.org/10.5061/dryad.sbcc2frjq.",
    "MD034_doi": "doi:10.1126/sciadv.ady2677",
    "MD037": "deletions in ΔPP_ 2428, ΔPP_ 4622, ΔPP_ 3540, and ΔPP_4373.",
    "MD038": "the OCR cell ` ΔPP_ 2428 ` carries its own padding.",
    "MD011": '"see (Table 4)[S4] of the Supplementary Material"',
    "MD049": '"the _mvaS_ overexpression and the',
    "MD050": '__PP_2074__ deletion"',
    "MD027": '>  "    PP_2428    PP_4622" is how the OCR laid out the table row',
    "MD039": '"see [ Table 4 ](S4) of the Supplementary Material"',
    "MD053": "[dryad-deposit]: https://doi.org/10.5061/dryad.sbcc2frjq",
    "MD010": "TEAM-3185\tPp TEAM-2777 ΔPP_ 2428",
}


def strip_json_comments(text: str) -> str:
    """Drop ``//`` line comments outside string literals, as markdownlint-cli2 does.

    String-aware on purpose: the config's comments hold URLs, and a naive split on
    ``//`` would also cut a ``https://`` that appeared in a value.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if char == "/" and text[index + 1 : index + 2] == "/":
            while index < len(text) and text[index] != "\n":
                index += 1
            continue
        out.append(char)
        index += 1
    return "".join(out)


def load_config() -> dict[str, object]:
    """The markdownlint rule table the hook and the editor both read."""
    config: dict[str, object] = json.loads(
        strip_json_comments(CONFIG.read_text(encoding="utf-8"))
    )
    return config


def markdownlint_hook() -> dict[str, object]:
    """The single ``markdownlint-cli2`` hook entry in ``.pre-commit-config.yaml``."""
    config = yaml.safe_load(PRE_COMMIT.read_text(encoding="utf-8"))
    hooks = [
        hook
        for repo in config["repos"]
        for hook in repo["hooks"]
        if hook["id"] == "markdownlint-cli2"
    ]
    assert len(hooks) == 1, f"expected exactly one markdownlint-cli2 hook, got {hooks}"
    hook: dict[str, object] = hooks[0]
    return hook


def test_comment_stripper_keeps_a_double_slash_inside_a_string() -> None:
    """The stripper cuts a comment and keeps a URL that lives in a value."""
    source = '{\n  // a comment with https://example.com\n  "MD044": "https://a//b"\n}'
    assert json.loads(strip_json_comments(source)) == {"MD044": "https://a//b"}


def test_exactly_one_markdownlint_config_file_exists() -> None:
    """Two config files would make precedence, and therefore the guarantee, ambiguous."""
    found = sorted(
        path.name
        for path in REPO.iterdir()
        if re.fullmatch(
            r"\.markdownlint(-cli2)?\.(json|jsonc|yaml|yml|cjs|mjs)", path.name
        )
    )
    assert found == [".markdownlint.jsonc"]


@pytest.mark.parametrize("rule", INLINE_FIX_RULES)
def test_every_inline_autofix_rule_is_disabled(rule: str) -> None:
    """An inline-fix rule is set to ``false``, so neither autofix path can run it."""
    config = load_config()
    assert config[rule] is False, f"{rule} must be false; got {config[rule]!r}"


def test_md010_leaves_fenced_blocks_alone() -> None:
    """MD010 fixes tabs in prose and never inside a fence, where tabs are source bytes."""
    assert load_config()["MD010"] == {"code_blocks": False}


def test_layout_rules_stay_enabled() -> None:
    """The whitespace-between-lines rules are untouched, so --fix stays useful.

    These move whitespace between lines and blocks and cannot change a quote's text,
    which is why the fix is a rule-level disable rather than check-only mode for notes/.
    """
    config = load_config()
    for rule in ("MD009", "MD012", "MD022", "MD031", "MD032", "MD047"):
        assert rule not in config, f"{rule} is a layout rule and must stay enabled"
    assert config["MD007"] == {"indent": 2}


def test_hook_lints_the_canary_and_the_notes_tree() -> None:
    """The canary is in the hook's scope, so the hook rewriting it fails this suite."""
    hook = markdownlint_hook()
    pattern = re.compile(str(hook["files"]))
    exclude = re.compile(str(hook["exclude"]))
    assert pattern.search(CANARY_REL)
    assert not exclude.search(CANARY_REL)
    assert pattern.search("notes/torchcell.datasets.pputida.menasalvas2025.md")
    assert exclude.search("notes/assets/verification/2026.09.29/messner2023.md")
    assert not pattern.search(
        "tests/torchcell/data/markdownlint_canary/quote_canary.py"
    )
    assert hook["args"] == ["--fix"]


@pytest.mark.parametrize("shape", sorted(CANARY_SHAPES))
def test_canary_still_holds_each_unfixed_shape(shape: str) -> None:
    """Each shape is byte-present in the canary, i.e. no autofix has reached it."""
    text = CANARY.read_text(encoding="utf-8")
    assert CANARY_SHAPES[shape] in text, f"{shape} shape was rewritten in {CANARY_REL}"


def test_canary_covers_every_inline_rule_that_can_fire() -> None:
    """Only MD044 is unexercised, and it cannot fire under this config.

    MD044 reports nothing without a ``names`` list, which the config does not define, so
    there is no shape to put in the canary; disabling it is forward protection for the
    day a name list is added.
    """
    covered = {shape.split("_")[0] for shape in CANARY_SHAPES}
    assert set(INLINE_FIX_RULES) - covered == {"MD044"}
